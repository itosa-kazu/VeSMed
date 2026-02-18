# VeSMed 検査マスタ全体レビュー相談文書

## 1. VeSMedとは

**Vector Space Medicine（ベクトル空間医学）** — 臨床所見のテキスト入力から、鑑別疾患ランキング・推奨検査・推奨問診/身体診察を算出する臨床意思決定支援システム。

### v2の理念と哲学

- **知覚はembedding、判断はLLM** — embeddingは意味的類似性の天才だが論理の盲目。論理的判断（矛盾検出、因果推論、極性判定）はLLMに任せる。
- **ハイパーパラメータなし** — 閾値やweight係数を人手で調整しない。全ての判断はデータ駆動またはLLM推論。
- **自然言語的に統一** — システムの全層が自然言語で通信する。embeddingの入出力も、LLMへのクエリも、全て自然言語。
- **所見即所得** — 入力した所見がそのまま結果に反映される。ブラックボックスの中間層を排除。
- **天井 = LLM記述の質 × LLM推論の質** — マスタデータの記述品質がembedding検索の精度を決め、ランタイムLLMの推論品質がフィルタ・novelty・分類の精度を決める。この2つの積がシステムの上限。

---

## 2. システムアーキテクチャ概要

```
[患者テキスト入力]
    ↓
[Step 0] LLM 3分類: 陽性所見 / 陰性所見 / 検査結果
    ↓                    ↓              ↓
[Step 1]          [Step 2 並行]     [Step 2 並行]
embedding検索     LLMフィルタ       統合Novelty
(陽性所見→         (矛盾除外)        (検査+HPE)
 ChromaDB)             ↓              ↓
    ↓            検査結果→update    HPE所見→update
候補疾患522件     (反実仮想法)       (背景情報)
    ↓
[Step 3] 数学ランキング
    ├─ Part A: 分散ベース検査推奨（鑑別）
    ├─ Part B: Critical Hit検査推奨（致死的疾患除外）
    ├─ Part C: 特異度ベース検査推奨（確認）
    ├─ Part D: 問診推奨（HPE）
    └─ Part E: 共通必要度検査推奨（基本）
```

### データ資産

| コンポーネント | 格納場所 | 件数 | 用途 |
|---|---|---|---|
| 疾患マスタ | diseases.jsonl | 527件 (522件ChromaDB収録) | fd_*セクション別記述、embedding検索のソース |
| **検査マスタ** | **tests.jsonl** | **344件** | **検査メタデータ、findings_description、hypothesis_text** |
| HPEマスタ | hpe_items.jsonl | 279件 (Hx:178, PE:101) | 問診・身体診察項目 |
| ChromaDB | chroma_db/ | 2976チャンク (522疾患×6ドメイン) | **疾患のみ**をembedding索引 |
| sim_matrix | sim_matrix.npz | (522, 335) | 疾患×検査のcos類似度行列 |
| sim_matrix_hpe | sim_matrix_hpe.npz | (522, 279) | 疾患×HPEのcos類似度行列 |

**重要**: ChromaDBには**疾患のみ**がindex。検査はChromaDBに入っていない。検査の推奨は`sim_matrix`（疾患embedding × 検査hypothesis embedding）の数学演算で算出。

---

## 3. 検査マスタの現状（★今回のレビュー対象★）

### 3.1 レコード構造

```json
{
  "test_name": "CRP",
  "category": "血液検査（生化学・一般）",
  "description_for_embedding": "（手技的説明、~291字、現在未使用）",
  "turnaround_minutes": 30,
  "sample_type": "採血",
  "contraindications": [],
  "notes": "（臨床的意義の短い説明）",
  "findings_description": "（全疾患横断の検査所見記述、平均11,483字、6セクション構成）",
  "quality_description": "（検査の質・安全性・コスト記述、侵襲性スコア計算に使用）",
  "hypothesis_text": "（1行の仮説文、平均48字、★sim_matrixの唯一の入力★）"
}
```

### 3.2 統計

| 項目 | 値 |
|---|---|
| 総検査数 | 344件 |
| findings_description有り | 335件 (97.4%) |
| findings_description無し | 9件 (D-dimer, 補体, クームス試験, 末梢血塗抹, KL-6, 尿中薬物スクリーニング, ハプトグロビン, ガラクトマンナン抗原, プレセプシン) |
| fd_*セクション分離 | **0件（未実施、疾患は完了済み）** |
| hypothesis_text有り | 344件 (100%) |
| quality_description有り | 335件 |
| findings_description文字数 | min 5,009 / max 19,654 / mean 11,483 / median 11,715 |
| hypothesis_text文字数 | min 22 / max 112 / mean 48 |

### 3.3 カテゴリ体系（29種類、統一性に問題あり）

```
血液検査（生化学・一般）:        48件
血液検査（内分泌・ホルモン）:     31件
血液検査（免疫・血清・感染症）:   29件
血液検査（自己抗体・アレルギー）: 22件
尿・便・穿刺液検査:              20件
侵襲的・カテーテル・生検:        16件
血液検査（腫瘍マーカー）:        15件
微生物学的検査（同定・遺伝子）:  15件
画像検査（超音波）:              14件
画像検査（CT）:                  13件
画像検査（X線・造影）:           12件
画像検査（MRI）:                 12件
生理機能検査（神経・筋・その他）: 11件
画像検査（核医学）:              11件
専門科検査（眼科）:              11件
生理機能検査（循環・呼吸）:      10件
内視鏡検査:                       9件
専門科検査（耳鼻咽喉科）:        8件
血液検査（凝固・線溶）:           7件
遺伝子・染色体検査:               7件
専門科検査（産婦人科・泌尿器科）: 7件
専門科検査（皮膚・アレルギー）:   6件
血液検査・免疫:        3件  ← 表記揺れ
血液検査・感染症:      2件  ← 表記揺れ
穿刺・採取検査:        1件  ← 表記揺れ
血液検査:              1件  ← 表記揺れ
尿検査:                1件  ← 表記揺れ
血液検査・生化学:      1件  ← 表記揺れ
血液検査・凝固:        1件  ← 表記揺れ
```

### 3.4 hypothesis_text のサンプル

これがsim_matrix計算の**唯一の入力**（平均48字）:

```
"血清CK上昇を認めた。横紋筋融解、心筋障害、筋炎、甲状腺ホルモン欠乏による筋酵素逸脱を示唆する"

"CRP上昇を認めた。急性炎症、感染症、自己免疫疾患、悪性腫瘍、組織破壊を示唆する"

"血液培養で微生物が検出された。グラム陽性球菌（ブドウ球菌、連鎖球菌、腸球菌）、グラム陰性桿菌（大腸菌、クレブシエラ、緑膿菌）、嫌気性菌、または真菌（カンジダ属）が分離培養された"

"胸部X線検査で異常陰影を認めた。肺炎、胸水、気胸、肺腫瘤、心拡大を示唆する"
```

### 3.5 findings_descriptionの6セクション構成

generate.pyのTEST_SECTION_PROMPTSで生成:

| # | セクション | 内容 |
|---|---|---|
| T1 | 適応臨床像 | どんな主訴・バイタル・身体所見で検査を想起するか |
| T2 | 疾患別異常パターン（主要群） | 主要15-20疾患の具体的異常パターン・数値・特異度 |
| T3 | 疾患別異常パターン（追加群） | 稀少・重篤・小児・代謝・薬剤性の追加疾患群 |
| T4 | 鑑別パターン | 重症度別・時間経過別・他検査組合せ別の鑑別論理 |
| T5 | 偽陽性・偽陰性・限界 | 薬剤・生理的変動・検体前処理・paradox・単独検査の限界 |
| T6 | 緊急値・時間軸・モニタリング | panic value、発症-異常値の時間差、ピーク、正常化、治療反応 |

**注意**: これら11,483字の記述はsim_matrixには使われていない。hypothesis_text（48字）のみが使用される。

---

## 4. sim_matrixの構成方法（★核心的課題★）

```python
# 検査側: hypothesis_text をembed（平均48字）
hypothesis_embs = embed([t["hypothesis_text"] for t in tests])  # (335, 4096)

# 疾患側: ChromaDB格納済みembedding（findings_descriptionの6ドメイン max-pooling）
disease_embs = normalized_disease_embeddings  # (522, 4096)

# sim_matrix = cos類似度
sim_matrix = disease_embs @ hypothesis_embs.T  # (522, 335)
```

**情報の非対称性**:
- 疾患側: findings_description（平均18,000字、6ドメインmax-pooling） → 4096次元
- 検査側: hypothesis_text（平均48字、1行テキスト） → 4096次元

この48字が4つの検査ランキング関数全ての基盤。

---

## 5. 検査ランキングの4つの軸

### Part A: 分散ベース（鑑別）
```
utility = variance × novelty × invasive_discount
variance = Σ w_i × (sim_matrix[i][j] - μ_j)²
```
候補疾患間で「バラつく」検査 = 鑑別に有用。
例: SLEと感染症が候補 → ANA検査は高分散（SLEに高類似、感染症に低類似）

### Part B: Critical Hit（致死的疾患除外）
```
utility = critical_hit × novelty × invasive_discount
critical_hit = max_i [ exp(cos_critical_i) × similarity_i × sim_matrix[i][j] ]
```
「見逃したら死ぬ疾患」を効率的に排除/確認できる検査。max演算。

### Part C: 特異度（確認）
```
utility = (cluster_μ - global_μ) × novelty × invasive_discount
```
候補疾患群に特異的な検査。CRP等の汎用検査はglobal_μが高いため沈む。

### Part E: 共通必要度（基本）
```
utility = cluster_μ × novelty × invasive_discount
```
候補疾患群に共通して必要な検査。Part Cと違いglobal_μを引かない。ルーチン検査が上位に。

### 共通要素
- **novelty**: LLM二値判定（実施済み=0, 未実施=1）
- **invasive_discount**: `exp(-max(0, cos_invasive - expected_criticality))` — 動的侵襲ペナルティ
- **w（疾患重み）**: `exp(cos_critical + cos_curable)` — 2C重み付け

---

## 6. 検査結果処理（update_from_results）

検査結果入力時の処理:

1. **LLMアノテーション**: 「CRP 15」→「CRP 15 mg/dL、著明な炎症反応上昇」に変換
2. **極性判定**: embedding × 極性軸 → 異常/正常を自動判定
3. **異常値**: 疾患embeddingとの直接cos類似度で候補疾患を増幅 `similarity *= exp(+excess)`
4. **正常値（反実仮想法）**: 「もしこの疾患ならこの検査は異常のはず」→ 正常なので抑制 `similarity *= exp(-excess)`

---

## 7. 実症例テスト結果（暫定）

47歳女性、来院1日前から左脇〜背中の痛み、全身痛、嘔吐、水様便、39度発熱。

### 候補疾患（良好）
- embedding検索は臨床的に妥当な候補を返す
- LLMフィルタの過剰除外問題を修正済み（リスク因子1つの否定で除外しない等）

### 検査推奨（改善余地あり）
- 詳細は臨床テスト進行中
- 感覚的に「なぜこの検査がこの順位なのか」の違和感がある場面あり
- 根本的にはhypothesis_text（48字）の情報量とsim_matrixの精度の問題と推測

---

## 8. 認識している課題まとめ

| # | 課題 | 深刻度 |
|---|---|---|
| 1 | hypothesis_text（48字）の情報量不足 → sim_matrix精度の天井 | ★★★ |
| 2 | 検査findings_description（11,483字）がsim_matrixに未活用 | ★★★ |
| 3 | 9件の重要検査がfindings_description未生成 | ★★ |
| 4 | カテゴリの表記揺れ（29種中6件以上） | ★ |
| 5 | fd_*セクション分離が未実施（検査側） | ★★ |
| 6 | hypothesis_textの方向性問題（上昇/低下が1行に収まらない） | ★★ |
| 7 | 344件の検査リストに重要な欠落がある可能性 | 不明 |

---

## 9. レビュアーへの質問

1. **hypothesis_textの設計**: 現在1行48字。この情報量でsim_matrix精度は十分か？リッチにすべきならどういう構造が良いか？

2. **sim_matrixの構成方法**: 疾患embedding × 検査hypothesis embeddingのcos類似度という方法は最適か？ 検査のfindings_description（11,483字）を活用する方法はあるか？

3. **4軸ランキングの設計**: 分散(A)、Critical Hit(B)、特異度(C)、共通必要度(E)の4軸は妥当か？不足・不要な軸はあるか？

4. **カテゴリ体系**: 表記揺れの修正以外に、カテゴリ情報をランキングに活用すべきか？

5. **検査マスタの網羅性**: 344件で十分か？重要な欠落はあるか？

6. **VeSMedの哲学（天井 = LLM記述の質 × LLM推論の質）を前提に**、検査推奨システム全体の改善方針として最も効果的なアプローチは何か？
