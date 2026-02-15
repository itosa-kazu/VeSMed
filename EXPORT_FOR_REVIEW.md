# VeSMed v2 完全技術仕様書（外部レビュー用）

※ 本書はengine.py（2457行）、web.py、index.py、config.pyの全コードを精読した上で記述。
コードの実際の動作に基づいており、コメントや変数名から推測した情報ではない。

---

## 1. テクノロジースタック

| 役割 | 技術 |
|------|------|
| LLM（推論・判断） | Vertex AI `gemini-3-flash-preview`（プライマリ）→ lemonapi → 12ai（3段フォールバック） |
| Embedding | `Qwen3-Embedding-8B` via OpenRouter, **4096次元** |
| ベクトルDB | ChromaDB（cosine距離） |
| Web UI | Gradio |

---

## 2. データ資産

### 2.1 diseases.jsonl（519件、うち488件がChromaDB indexed）

**フィールド一覧（16種）:**
`disease_name`, `icd10`, `category`, `description_for_embedding`, `findings_description`,
`findings_description_v2`, `urgency`, `age_peak`, `gender_tendency`, `risk_factors`,
`core_symptoms`, `common_symptoms`, `rare_but_specific`, `core_signs`, `differential_top5`,
`relevant_tests`

**統計:**
| フィールド | 件数 | 平均文字数 | min | max |
|---|---|---|---|---|
| `description_for_embedding` | 519 | 291字 | 205 | 376 |
| `findings_description` | 488 | **18,512字** | 10,527 | 39,896 |
| `findings_description_v2` | 110 | ~31,898字 | - | - |

**urgency分布:** 超緊急79 / 緊急157 / 準緊急141 / 通常141 / その他1

**★重要: ChromaDBに格納されているのは `findings_description`（18K字）である。**
`description_for_embedding`（291字）はどこにも使われていない死んだフィールドである。
index.pyのコメントには「description_for_embeddingを格納」と書いてあるが、
実際のコード（L58）は `d.get("findings_description", "")` で `findings_description` を使用している。
`findings_description` が存在しない31件がChromaDB未登録（519-488=31）。

### 2.2 tests.jsonl（340件、うち331件がsim_matrixに参加）

**フィールド一覧:**
`test_name`, `category`, `hypothesis_text`, `findings_description`,
`quality_description`, `risk_description`, `description_for_embedding`,
`turnaround_minutes`, `sample_type`, `contraindications`, `notes`

**統計:**
| フィールド | 件数 | 平均文字数 |
|---|---|---|
| `hypothesis_text` | 340 | 49字 |
| `findings_description` | 331 | 11,457字 |
| `quality_description` | 331 | 1,239字 |
| `risk_description` | 40 | - |

検査の`findings_description`はsim_matrix計算には使われない（フィルタ条件としてのみ使用）。
`hypothesis_text`がsim_matrix計算に使われる。
`quality_description`は侵襲性スコアと検査の質スコア計算に使われる。

### 2.3 hpe_items.jsonl（274件: Hx 173, PE 101）

**フィールド一覧:** `item_name`, `category`, `subcategory`, `hypothesis`, `instruction`

### 2.4 事前計算行列

| 行列 | サイズ | 計算方法 |
|---|---|---|
| `sim_matrix` | (488, 331) | `disease_embs_normed @ hypothesis_embs_normed.T` |
| `sim_matrix_hpe` | (488, 274) | `disease_embs_normed @ hpe_hypothesis_embs_normed.T` |

---

## 3. パイプライン（web.py → engine.py）

ユーザーが自由記述テキストを入力してからランキングが返るまでの全処理。
コードの実際の呼び出し順に記述。

### Step 0: テキスト3分類（LLM）
```
入力: 自由記述テキスト
出力: (positive_text, negative_findings[], result_lines[])
方法: LLMが3カテゴリに分離
  - positive_findings: 陽性症状・所見（→ Step 1へ）
  - negative_findings: 陰性所見（→ Step 2のLLMフィルタへ）
  - results: 検査結果の数値（→ Step 3へ）
フォールバック: embedding類似度ベース分離（陰性分離不可）
```

### Step 1: 疾患検索（embedding — 知覚）
```
入力: positive_text
方法: positive_textをQwen3-Embedding-8Bでembed → ChromaDBで全488疾患と
     cosine距離で類似度検索 → 全疾患を類似度降順でソート
出力: candidates[] — 各要素に {disease_name, similarity, category, urgency}

★ ChromaDBに格納されている疾患embeddingは findings_description（平均18,512字）を
  1本の4096次元ベクトルに圧縮したものである。
  患者テキスト（数十〜数百字）のembeddingとの cosine類似度で検索される。
```

### Step 2: 並行処理（3タスク同時実行 — ThreadPoolExecutor）

**2a. LLMフィルタ（判断）:**
```
入力: candidates上位20件 + 患者テキスト全文 + negative_findings[]
方法: LLMが論理的矛盾を検出し疾患を除外
  例: ANA陰性 → SLE除外、男性 → PID除外
出力: (filtered_candidates[], exclusion_reasons[])
```

**2b. 検査結果更新（polarity + 反実仮想法）:**
```
入力: candidates + result_lines[] + positive_text
方法:
  1. LLMアノテーション（または基準範囲テーブル）で各結果行に方向語を付与
  2. 極性判定: polarity = dot(result_emb, polarity_axis)
     polarity_axis = normalize(abnormal_anchor_emb - normal_anchor_emb)
  3. 異常結果 (polarity > 0):
     sims = result_emb @ disease_embs → excess = max(0, sim[d] - mean)
     similarity *= exp(+excess)  — 関連疾患を増幅
  4. 正常結果 (polarity ≤ 0): 反実仮想法
     定量検査: 「検査名 異常 上昇」「検査名 異常 低下」の双方向仮想結果をembed
       similarity *= exp(-(excess_up + excess_down))  — 両方向の関連疾患を抑制
     定性検査: 「検査名 異常」の単方向仮想
       similarity *= exp(-excess)
```

**2c. 統合Novelty（1 LLMコール）:**
```
入力: 患者テキスト全文 + 検査マスタ全331件 + HPEマスタ全274件
方法: LLMが3タスクを同時実行
  1. 実施済み検査の特定 → novelty_tests[] (二値: 0 or 1)
  2. 聴取済みHPE項目の特定 → novelty_hpe[] (二値: 0 or 1)
  3. HPE所見の極性判定 → hpe_findings[] (item, polarity +1/-1)
フォールバック: 個別LLMコール3回 → embeddingギャップ検出
```

### Step 3: HPE所見による疾患重み更新（★現在の問題点★）
```python
# 現在のコード（engine.py L2239-2269）:
for f in hpe_findings:              # N件のHPE所見（例: 発熱, 腹痛, 頭痛なし...）
    sims = sim_matrix_hpe[:, idx]    # この所見と全疾患の関連度
    bg = mean(sims)                  # 全疾患平均（背景）
    for each candidate:
        excess = max(0, sims[d] - bg)
        if excess > 0:
            similarity *= exp(polarity * excess)  # 乗算（1所見ごとに1回）

# 問題: N所見あればexp()がN回乗算される → 指数的発散
# 例: N=28所見 → 一部疾患のsimilarityが0.58→0.88に急騰
# → 重みが2-3疾患に集中 → Part Aの加重分散がゼロに崩壊

# 検証済み改善案: delta/√N 方式
# delta = Σ polarity_k * excess_k  （N所見分を合算）
# similarity *= exp(delta / √N)     （1回だけ乗算）
# テスト結果: 10ケース比較で√N方式が最適（分散3倍改善、発散防止）
```

### Step 4: ランキング（4つのPart）

全Partで共通の重み計算:
```python
raw_sims = [candidate.similarity for each candidate]  # HPE更新済みの値
sim_centered = max(0, raw_sims - mean(raw_sims))       # 平均以上のみ
w = sim_centered * exp(cos_critical + cos_curable)      # 2C重み付け
w = w / sum(w)                                           # 正規化
```

**Part A: 鑑別推奨（prior加重分散）**
```
Var_w[j] = Σ w_i × (sim_matrix[i][j] - μ_j)²
μ_j = Σ w_i × sim_matrix[i][j]
utility_j = Var_w[j] × novelty_j × invasive_discount_j

意味: 候補疾患間で「バラつく」検査 = どちらかの疾患を肯定/否定できる = 鑑別に有用
弱点: 候補群に均一に関連する検査（血培、CBC等）は分散が小さく沈む
```

**Part B: Critical Hit（致命疾患排除）**
```
critical_hit_j = max_i [ exp(cos_critical_i) × similarity_i × sim_matrix[i][j] ]
utility_j = critical_hit_j × novelty_j × invasive_discount_j

意味: 「見逃したら死ぬ疾患」を最も効率的に排除/確認できる検査
max演算: 最も致命的な1疾患との命中度で決まる（分散ではなく最大値）
```

**Part C: 確認・同定推奨（クラスタ特異度）**
```
cluster_mu_j = Σ w_i × sim_matrix[i][j]   — 候補群の加重平均
global_mu_j = mean(sim_matrix[:][j])        — 全488疾患の非加重平均
confirm_j = cluster_mu_j - global_mu_j
utility_j = confirm_j × novelty_j × invasive_discount_j

意味: 候補群に「特異的に」関連する検査（背景ノイズを除去）
CRP（汎用）はglobal_muも高い → 差分小 → 沈む
血液培養（特異的）はcluster_mu >> global_mu → 浮上
```

**Part D: 問診・身体診察推奨（Part Aと同一数学）**
```
sim_matrix_hpe (488, 274) を使用
utility_k = Var_w[k] × novelty_hpe_k
侵襲性バランシングなし（問診・身体診察は非侵襲）
```

### 動的侵襲性バランシング（Part A/B/Cに適用）

```python
# 各検査の侵襲度（事前計算、quality_descriptionから）
cos_invasive_j = cos(quality_desc_emb_j, invasive_anchor_emb)

# 候補群の重症度
expected_criticality = Σ w_i × cos_critical_i

# ペナルティ
penalty_j = max(0, cos_invasive_j - expected_criticality)
utility *= exp(-penalty_j)

# 意味: 候補群が軽症なのに侵襲的検査が上位に来るのを防ぐ
# 重症群（expected_criticality高）ではペナルティ消失
```

---

## 4. 事前計算コンポーネント（engine.__init__で全て実行）

### 4.1 2Cスコア（Critical / Curable）
```
critical_anchor: "未治療の場合、数時間以内にバイタルサイン急速悪化...心停止に至る"
curable_anchor: "治療開始後、数時間から数日で検査値の正常化...症状消失"
cos_critical_d = cos(disease_emb_d, critical_anchor_emb)
cos_curable_d = cos(disease_emb_d, curable_anchor_emb)
weight_d = exp(cos_critical_d + cos_curable_d)
```

### 4.2 sim_matrix（仮説embedding方式）
```
# 検査側: hypothesis_text をembed（平均49字）
# 疾患側: ChromaDB格納済みembedding（findings_descriptionの18K字をembed済み）
sim_matrix = disease_embs_normed @ hypothesis_embs_normed.T  # (488, 331)

# 例: hypothesis_text = "血液培養で微生物が検出された。グラム陽性球菌..."（49字）
#     disease_emb = E("60代以上の高齢者や糖尿病...悪寒戦慄...ショック..."）（18K字のemb）
#     sim = cos(hypothesis_emb, disease_emb)
```

### 4.3 検査の質スコア（差分ベクトル射影）
```
good_anchor: "採血のみで実施可能...15分以内に結果判明...致命的疾患を検出..."
bad_anchor: "全身麻酔下でカテーテル挿入...数日を要し..."
quality_axis = normalize(good_emb - bad_emb)
quality_score_j = dot(quality_desc_emb_j, quality_axis)

用途: 現在は直接ランキングに使用していない（侵襲性スコアに統合）
```

### 4.4 極性軸（正常←→異常）
```
normal_anchor: "検査値は基準範囲内であり正常...陰性...検出されず"
abnormal_anchor: "検査値は基準範囲を逸脱し異常...陽性...上昇...低下...検出"
polarity_axis = normalize(abnormal_emb - normal_emb)

用途: update_from_results で検査結果の方向判定
polarity = dot(result_emb, polarity_axis)
> 0 → 異常結果, ≤ 0 → 正常結果
```

---

## 5. データサンプル

### 5.1 疾患サンプル

#### 敗血症性ショック（urgency: 超緊急）
```
description_for_embedding (313字)（★使われていない）:
60代以上の高齢者や糖尿病、悪性腫瘍、免疫抑制状態にある患者が、数日前からの発熱、
悪寒戦慄、全身倦怠感を主訴に来院し、急激な意識レベルの低下や呼吸困難を呈する
臨床像が典型的である。来院時、収縮期血圧90mmHg未満の低血圧、頻脈、頻呼吸を認め、
十分な輸液負荷（30mL/kg以上）を行っても平均血圧65mmHg以上を維持できず、
循環作動薬を必要とする状態に陥る。（略）血清乳酸値の上昇（>2mmol/L）を認め...
```

```
findings_description (18K字前後)（★これがChromaDBに格納されている）:
※ LLM（Gemini）が生成した極めて詳細な臨床記述。症状、身体所見、検査所見、
  鑑別ポイント、重症度評価を網羅。全文18,000字前後。
  このテキスト全体が1本の4096次元ベクトルに圧縮されてChromaDBに格納される。
```

#### 前頭側頭型認知症（urgency: 通常）
```
description_for_embedding (231字):
50代後半で発症。以前は温厚であったが、次第に万引きや信号無視などの社会的脱抑制行動が
目立つようになり、家族の制止を聞かず、注意しても反省の態度が見られない。（略）
人格変化と行動異常が主訴となる典型的な前頭側頭葉変性症の臨床像である。
```

### 5.2 検査サンプル

#### 血液培養 (好気/嫌気)
```
hypothesis_text (49字):
血液培養で微生物が検出された。グラム陽性球菌（ブドウ球菌、連鎖球菌、腸球菌）、
グラム陰性桿菌（大腸菌、クレブシエラ、緑膿菌）、嫌気性菌、または真菌（カンジダ属）
が分離培養された
```

sim_matrix計算: `cos(E(hypothesis_text), disease_emb)` で各疾患との関連度を計算。
検査のfindings_description（11K字）はsim_matrix計算には使用されない。

### 5.3 HPEサンプル

```json
{"item_name": "発熱", "hypothesis": "発熱 熱発 体温上昇 微熱 高熱", ...}
{"item_name": "Murphy徴候", "hypothesis": "Murphy徴候 右季肋部圧痛 吸気停止 胆嚢炎", ...}
{"item_name": "咳嗽", "hypothesis": "咳嗽 咳 乾性咳嗽 湿性咳嗽 遷延性 3週間以上", ...}
```

時間軸項目:
```
突然発症（数秒）: "突然発症 数秒で最大に達する 雷鳴様"
急性発症（分〜時間）: "急性発症 分から時間単位で進行 増悪"
亜急性発症（日〜週）: "亜急性発症 日から週単位で徐々に進行"
緩徐発症（週〜月）: "緩徐発症 週から月単位で緩やかに進行"
```

---

## 6. 判明している課題と検証結果

### 6.1 ★致命的★ HPE更新の乗算による分散崩壊

**症状:** 陰性所見が多い症例（N≥12）でPart Aの全検査がutility=0.00に崩壊。

**根本原因:** `update_from_hpe()` が各HPE所見ごとに `similarity *= exp(polarity * excess)` を
N回乗算する。N=28所見で上位2-3疾患のsimilarityが指数的に増大 → 重みが極度に集中 →
加重分散がゼロに収束。LLMの非決定性で小さな出力差が指数的に増幅され、
同じ入力でも結果が不安定。

**検証済み解決策:** delta/√N方式。10テストケースで3方式を比較:
- 方式A（現行・乗算式）: 平均分散0.0059, 平均top1重み0.493
- 方式B（delta/N）: 平均分散0.0207, 平均top1重み0.431（保守的すぎ）
- **方式C（delta/√N）: 平均分散0.0171, 平均top1重み0.458（最適バランス）**

### 6.2 ★重要★ 血液培養がランキングに入らない

**症状:** ショック状態の敗血症疑い症例で、血液培養がPart A/B/CいずれのTop10にも入らない。

**原因（Part Aの構造的限界）:**
血液培養は敗血症関連疾患で「均一に高い」sim値を持つ → 分散が小さい →
「全候補に共通して必要な検査」を推薦する仕組みが欠けている。

**注意:** 前回のレビューで「291字のサマリーに起炎菌の記述がないから」と分析されたが、
これは誤りである。ChromaDBには18K字のfindings_descriptionが格納されており、
そこには血液培養に関する詳細な記述が含まれている。問題はテキストの解像度ではなく、
Part Aの分散ベースの数学が「鑑別に有用な検査」のみを選び、
「候補群に共通して必須の検査」を構造的に見つけられないことにある。

**対応案:** Part E（cluster_mu）を新設。
```
utility_e_j = cluster_mu_j × novelty_j
cluster_mu_j = Σ w_i × sim_matrix[i][j]  — 候補群の加重平均
```
全候補に「共通して関連する」検査が上位に来る。Part Cの confirm_score（cluster_mu - global_mu）
とは異なり、global_muを引かない。Part Cは「特異性」、Part Eは「共通必要度」。

### 6.3 HPE時間粒度の不足

「咳嗽」は1項目のみ。急性咳嗽 vs 慢性咳嗽の区別がHPE levelでは不可能。
発症様式（突然/急性/亜急性/緩徐）の4項目はあるが、症状×時間のクロスバリアントはない。

### 6.4 疾患カバレッジの穴

以下のコモンディジーズが diseases.jsonl に未登録:
- 咳喘息、後鼻漏/上気道咳嗽症候群(UACS)、好酸球性気管支炎
- 急性気管支炎、感冒/急性上気道炎
- ACE阻害薬誘発性咳嗽
- GERD関連咳嗽

### 6.5 18K字→1ベクトル圧縮の情報損失（未検証）

findings_description（平均18,512字）を1本の4096次元ベクトルに圧縮している。
この圧縮で重要な臨床情報（特異的な身体所見、マイナーな検査所見等）が
ベクトル空間上で希釈されている可能性がある。
Semantic Chunking（疫学/身体所見/検査所見でチャンク分割→max-pooling）は
検討課題だが、現時点では未検証。

---

## 7. 哲学と設計原則

- **「知覚はembedding、判断はLLM」**: embeddingは類似性の天才だが論理の盲目。
  否定・矛盾・文脈依存の判断はLLMに委譲する。
- **ハイパーパラメータなし**: 閾値、重み係数、チューニングパラメータを排除。
  全てをembedding空間の幾何学とLLMの推論に委ねる。
- **自然言語的統一**: 全層が自然言語で通信。数値化・離散化を最小限に。
- **所見即所得**: テキストに書かれた概念のみがベクトル空間に存在する。
  天井 = LLM記述の質 × LLM推論の質。

---

## 8. 議論したい問い

1. 18K字を1ベクトルに圧縮する現方式で、Qwen3-Embedding-8Bの表現力は足りているか？
   Semantic Chunkingの優先度はどの程度か？
2. Part E (cluster_mu) で血液培養問題は本当に解決するか？
   cluster_muとconfirm_score (cluster_mu - global_mu) の使い分けは正しいか？
3. HPEの症状×時間クロスバリアント（咳嗽(急性) / 咳嗽(慢性)）は追加すべきか？
   追加する場合、LLMの抽出精度は維持できるか？
4. コモンディジーズ（感冒、急性気管支炎等）の欠落はランキング精度にどう影響するか？
   「低重心の防波堤」としてbase-rate neglectを防ぐ効果はあるか？
