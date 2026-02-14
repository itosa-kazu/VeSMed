# VeSMed Novelty問題 — 技術文書

## 1. VeSMedとは何か

VeSMed（Vector Space Medicine）は、臨床テキストからベクトル空間検索で候補疾患を見つけ、次に行うべき検査・問診・身体診察をランキングする診断支援システム。

### v2アーキテクチャ: 「知覚はembedding、判断はLLM」

```
テキスト入力（自由記述）
    │
    ▼
[LLM] 症状と検査結果を自動分離
    │
    ├── 症状テキスト ──→ [Embedding] ベクトル検索 → 候補疾患リスト（知覚）
    │                          │
    │                          ▼
    │                     [LLM] 論理矛盾疾患を除外（判断）
    │                          │
    ├── 検査結果 ─────→ 極性判定 × 反実仮想法で疾患similarity更新
    │                          │
    ├── [LLM] Novelty ──→ 実施済み検査を二値判定（0 or 1）  ← ★ここが問題
    │                          │
    │                          ▼
    │                     数学層: ランキング計算
    │                     ├── Part A: 鑑別推奨（分散ベース）
    │                     ├── Part B: Critical排除（命中ベース）
    │                     ├── Part C: 確認・同定（特異度ベース）
    │                     └── Part D: 問診・身体診察（分散ベース）
    ▼
    出力: 推奨検査ランキング
```

### データ構造

- **疾患DB**: 519疾患（うち488疾患がChromaDBにembedding済み）
- **検査リスト**: 372件（`test_names[0..371]`）。各検査に `hypothesis_text`（仮説テキスト）あり
- **HPE（問診・身体診察）リスト**: 274項目（Hx: 173, PE: 101）
- **sim_matrix**: (488疾患 × 372検査) の余弦類似度行列。ランキングの心臓部
- **sim_matrix_hpe**: (488疾患 × 274項目) の余弦類似度行列

### sim_matrixの構築

```
sim_matrix[i][j] = cos(disease_emb[i], hypothesis_emb[j])
```

- `disease_emb[i]`: 疾患iの全記述テキスト（平均19,844字）をembedding（dim=4096）
- `hypothesis_emb[j]`: 検査jの仮説テキスト（例: 「血清アミラーゼが異常高値を示す患者では急性膵炎、慢性膵炎の急性増悪…」）をembedding
- 両方をL2正規化した後、行列積: `sim_matrix = disease_embs_normed @ hypothesis_embs_normed.T`

## 2. ランキングの計算式

### 共通: 局所重み w

候補疾患iに対する重み w_i:

```
raw_sim_i     = cos(patient_emb, disease_emb_i)   ← ベクトル検索の類似度
2c_weight_i   = exp(cos_critical_i + cos_curable_i)  ← 臨床重要度
sim_centered_i = max(0, raw_sim_i - mean(raw_sims))   ← 平均以下はゼロ
w_i = sim_centered_i × 2c_weight_i
w   = w / sum(w)                                       ← 正規化
```

- `cos_critical_i` = cos(disease_emb_i, critical_anchor_emb)
  - critical_anchor = 「未治療の場合、数時間以内にバイタルサイン急速悪化…心停止に至る」
- `cos_curable_i` = cos(disease_emb_i, curable_anchor_emb)
  - curable_anchor = 「治療開始後、数時間から数日で検査値の正常化…画像所見改善」

### Part A: 鑑別推奨（prior加重分散）

```
μ_j  = Σ_i  w_i × sim_matrix[i][j]
var_j = Σ_i  w_i × (sim_matrix[i][j] - μ_j)²

utility_j = var_j × novelty_j
```

**意味**: 候補疾患間で検査jへの類似度がバラつくほど、その検査は鑑別に有用。

### Part B: Critical排除（最大命中）

```
critical_hit_j = max_i [ exp(cos_critical_i) × raw_sim_i × sim_matrix[i][j] ]

utility_j = critical_hit_j × novelty_j
```

**意味**: 「見逃したら死ぬ疾患」を最も強く排除/確認できる検査が上位。

### Part C: 確認・同定（クラスタ特異度）

```
cluster_mu_j = Σ_i  w_i × sim_matrix[i][j]       ← 候補疾患群の加重平均
global_mu_j  = mean over ALL diseases of sim_matrix[·][j]  ← 全疾患の非加重平均
confirm_j    = cluster_mu_j - global_mu_j

utility_j = confirm_j × novelty_j
```

**意味**: CRPのような汎用検査はglobal_muも高い→差分小→沈む。血液培養のような特異的検査はcluster_mu >> global_mu→差分大→浮上。

### Part D: 問診・身体診察（Part Aと同一数学）

```
μ_k  = Σ_i  w_i × sim_matrix_hpe[i][k]
var_k = Σ_i  w_i × (sim_matrix_hpe[i][k] - μ_k)²

utility_k = var_k × novelty_hpe_k
```

## 3. Noveltyの役割

### 定義

```
novelty_j ∈ {0, 1}   （二値）
  0 = 検査jは実施済み（推奨しない）
  1 = 検査jは未実施（推奨する）
```

### utilityへの影響

全パート(A/B/C/D)で同じ構造:

```
utility_j = score_j × novelty_j
```

- `novelty_j = 0` → `utility_j = 0` → **ランキングから事実上消える**
- `novelty_j = 1` → `utility_j = score_j` → 純粋なスコアで競争

### なぜ二値にしたか

以前はembeddingの余弦類似度（0~1の連続値）でnoveltyを計算していた。問題:
- CRPを実施済み → CRP付近の検査（ESR、プロカルシトニン等）まで0.5程度に割引
- 全く関係ない未実施検査でも0.55~0.48にばらつく
- **ランキングがnoveltyの微小な差で支配される**

v2の哲学: 「やったかどうか」は論理判断 → LLMに任せるべき。

## 4. 現在のNovelty実装（`_compute_novelty_llm`）

### ステップ1: LLMに実施済み検査を抽出させる

```python
sample_names = self.test_names[:80]  # ← 先頭80件のみ！
sample_text = json.dumps(sample_names, ensure_ascii=False)

prompt = f"""以下の臨床テキストから、既に実施済み・結果が判明している検査を全て列挙してください。

ルール:
- 検査名は下記リストから選ぶか、テキスト中の検査名をそのまま使う
- 「WBC 15000」→「白血球数」、「CRP 8.5」→「CRP」のように正規化
- 「心電図ST変化なし」→「12誘導心電図」、「胸部X線異常なし」→「胸部X線」
- 「CBC」→「白血球数」「赤血球数」「ヘモグロビン」「血小板数」に展開
- 結果の陽性/陰性は問わず、実施されたものを全て列挙

臨床テキスト:
{patient_text}

検査名リスト（参考）:
{sample_text}

出力: JSON配列 ["検査名1", "検査名2", ...]"""
```

LLMは JSON配列を返す。例: `["CRP", "白血球数", "体温"]`

### ステップ2: LLMの出力を検査リストにマッチング

```python
done_tests = json.loads(content)  # LLMの出力

for dt in done_tests:
    # 完全一致
    if dt in self.test_idx:          # test_idx = {"白血球数": 0, "CRP": 1, ...}
        novelty[self.test_idx[dt]] = 0.0
        continue
    # 部分一致（文字列包含）
    for tname, tidx in self.test_idx.items():
        if dt in tname or tname in dt:
            novelty[tidx] = 0.0
            break
```

### ステップ3: LLM失敗時のフォールバック

LLM呼び出しが失敗した場合のみ、embeddingギャップ検出にフォールバック:
- 患者テキストの各行をembed → 各検査名embeddingとの最大cos算出
- ソートしてギャップ検出 → 閾値以上を「実施済み」とみなす
- 閾値 < 0.55 の場合は抑制しない（精度が低いため）

## 5. 問題の詳細

### 具体例

患者テキスト:
```
・37.5度、血圧 92/60 mmHg、心拍数120回/分、呼吸数 26回/分、SpO2 89%（room air）
```

**期待**: 「バイタルサイン測定」はnovelty=0（実施済み）→ ランキングから消える
**実際**: 「バイタルサイン測定」はnovelty=1（未実施扱い）→ Part A/B/C 全て1位

### 原因1: サンプルリストの切り捨て（80件制限）

```python
sample_names = self.test_names[:80]  # 先頭80件のみLLMに見せる
```

- `test_names`は372件
- 「バイタルサイン測定」は `test_names[332]`（index=332）
- LLMは先頭80件しか見えないので、「バイタルサイン測定」という検査名を知らない
- LLMが仮に「体温測定」「血圧測定」と返しても、正式な検査名と一致しない

**影響範囲**: index 80~371の292件（全検査の78%）がサンプルに含まれない。

### 原因2: マッチングロジックの不足

LLMが「体温」「血圧」「SpO2」等の個別バイタル名を返した場合:

```python
# 部分一致チェック
dt = "体温"
tname = "バイタルサイン測定"

"体温" in "バイタルサイン測定"  → False  # 文字列として包含しない
"バイタルサイン測定" in "体温"  → False
```

個別バイタル → 「バイタルサイン測定」への論理的マッピングが存在しない。

### 原因3: 名寄せマップ（test_name_map）の不在

`test_name_map`は1,872件のエイリアス→正式名マッピングだが:
- 「体温」→「バイタルサイン測定」のマッピングがない
- 「血圧」→「バイタルサイン測定」のマッピングがない
- 「SpO2」→「バイタルサイン測定」のマッピングがない
- そもそもnoveltyのマッチングで`test_name_map`を使っていない

### 波及範囲

バイタルサインだけでなく、以下の全パターンに共通する構造的問題:

| パターン | 例 | 問題 |
|---------|---|------|
| 複合検査 → 個別結果 | バイタルサイン → 体温, 血圧, SpO2 | 論理的包含関係 |
| パネル検査 → 個別項目 | CBC → WBC, RBC, Hb, Plt | 1検査名が複数項目を含む |
| 略称 → 正式名 | ABG → 動脈血液ガス分析 | 略称が正式名に不一致 |
| index 80以降 | 全372件中292件 | サンプルリストに含まれない |

## 6. 解決の方向性（案）

### 案A: 全検査名をプロンプトに入れる
- 372件全部をJSONリストでLLMに渡す
- 問題: トークン数が増大（推定: 372件 × 平均15字 ≈ 5,580字追加）
- Gemini 3 Proなら入力コンテキストは十分大きいので問題ない可能性

### 案B: 名寄せマップをnoveltyで使う + エイリアス拡充
- マッチング時に `test_name_map` を参照
- 「体温」→「バイタルサイン測定」等のエイリアスを追加
- 問題: 全パターンを人手で列挙する必要がある

### 案C: embeddingマッチング
- LLMが返した検査名をembedding → 372検査の名前embeddingと比較
- 閾値以上（例: cos > 0.7）なら一致とみなす
- 問題: 閾値のチューニングが必要（ハイパーパラメータ回避の哲学に反する可能性）

### 案D: LLMプロンプト改修（自由列挙 + 後段マッチング）
- LLMに検査リストを渡さず「テキスト中の全ての検査・測定行為を自由に列挙せよ」
- LLMの出力をembeddingで372検査とマッチング
- 2段階: LLM抽出（自由形式）→ embeddingマッチング（意味的近傍検索）

### 案E: 検査側にもLLMフィルタ
- noveltyの責務をランキング後に移動
- 最終ランキングTop30をLLMに見せて「この中で既に実施済みのものを除外せよ」
- 問題: LLM呼び出しが増える

## 7. 制約条件

- **ハイパーパラメータなし**: VeSMedの設計哲学。閾値や手動チューニングは避ける
- **所見即所得**: LLM生成テキストの質がシステムの天井を決める
- **LLM API**: Vertex AI gemini-3-pro-preview（高品質だがレイテンシあり）
- **Embedding API**: Qwen3-Embedding-8B（dim=4096、高速）
- **日本語**: 全ての検査名・疾患名・テキストは日本語
