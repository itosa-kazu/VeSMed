# VeSMed Embedding テキスト レビュー依頼

## 背景

VeSMed（ベクトル空間医学）は、テキストembeddingで疾患候補検索と検査推薦を行うシステムです。

### embeddingの仕組み

```
sim_matrix[疾患i][検査j] = cos(disease_emb_i, test_hypothesis_emb_j)
sim_matrix_hpe[疾患i][HPE項目k] = cos(disease_emb_i, hpe_hypothesis_emb_k)
```

- **疾患側**: `findings_description`（平均18,000字の臨床記述、6セクション）をembedding
- **検査側**: `hypothesis_text`（50-100字の短文）をembedding
- **HPE側**: `hypothesis`（30-60字のキーワード列）をembedding
- embeddingモデル: Qwen3-Embedding-8B（4096次元、32Kトークン対応）

### 何が問題か

hypothesis/hypothesis_text は sim_matrix の「列側ベクトル」を決定します。
この短いテキストの質が、検索精度に直結します。

具体的な懸念:
1. **疾患名のカンニング**: hypothesisに疾患名を直接含めると、embedding空間でその疾患とだけ近くなり、症候としての汎用的マッチングが壊れる
2. **無関係な概念の混入**: 「下痢」のhypothesisに「体重減少」を含めると、体重減少の疾患にも下痢のスコアが上がる
3. **方向性の欠如**: embeddingは否定を区別できない。「陰性」「なし」等は逆効果になりうる
4. **冗長 vs 不足**: テキストが短すぎると意味が取れず、長すぎると焦点がぼける

## レビュー対象ファイル

### `hypothesis_texts_for_review.jsonl`（619行、123KB）

**HPE hypothesis（279件）** — `type: "hpe"`:
```json
{"type":"hpe","item_name":"発熱（急性：1週未満）","category":"Hx","subcategory":"ROS:全身","hypothesis":"急性発熱 数日以内 感染症 悪寒 戦慄 急性炎症"}
```
- `hypothesis`: このテキストがembeddingされ、全488疾患のfindings_descriptionとcosine類似度が計算される
- 目的: 疾患の「発熱関連」記述との類似度を適切に反映すること

**検査 hypothesis_text（340件）** — `type: "test"`:
```json
{"type":"test","test_name":"アルブミン (Alb)","hypothesis_text":"血清アルブミン低値を認めた。低アルブミン血症、低栄養、肝合成能低下、蛋白漏出を示唆する"}
```
- `hypothesis_text`: このテキストがembeddingされ、全488疾患のfindings_descriptionとcosine類似度が計算される
- 目的: 疾患の検査所見記述との類似度を適切に反映すること

### `disease_sample_for_review.jsonl`（1行、84KB） — 参考

疾患側のfindings_descriptionのサンプル（肺炎球菌性肺炎、33K字）。
hypothesis/hypothesis_textはこの長い臨床記述との類似度を計算するためのものです。

## チェックしてほしいこと

### 1. 疾患名カンニングの検出
- hypothesisに特定の疾患名が含まれていないか
- 疾患名が含まれると、その疾患だけとの類似度が不当に高くなる
- 例: NG → `"慢性関節痛 関節リウマチ 変形性関節症"` （疾患名が入っている）
- 例: OK → `"慢性関節痛 数ヶ月以上の関節痛 朝のこわばり 多関節 左右対称"` （症候のみ）

### 2. 無関係概念の混入
- 下痢のhypothesisに「体重減少」、咳嗽のhypothesisに「呼吸困難」など、別の症候が混入していないか
- hypothesisは「その症候/検査そのもの」の特徴を記述すべきで、「結果として起きる別の症状」は含めるべきでない

### 3. 否定語・方向性の問題
- embeddingは「陰性」「なし」「正常」を区別できない
- 「ANA陰性」と書いてもembeddingは「ANA」に反応する
- hypothesisに否定的な文脈が含まれていないか

### 4. テキスト品質の一貫性
- 同じカテゴリ（例: ROS:消化器）の項目間で、記述のスタイル・粒度が一貫しているか
- 一部だけ異常に短い/長いものがないか

### 5. 臨床的正確性
- 医学的に間違った記述がないか
- 検査のhypothesis_textが、その検査の異常所見を正しく反映しているか

### 6. 時間分割項目の妥当性
- 発熱、咳嗽、下痢、頭痛、関節痛は「急性」「慢性」に分割済み
- 分割の閾値（3週、2週、4週等）が臨床的に妥当か
- 他にも時間分割すべき症候がないか

## 出力フォーマット

問題のある項目を以下の形式でリストアップしてください:

```
[問題カテゴリ] item_name or test_name
  現在: "現在のhypothesis/hypothesis_text"
  問題: 具体的な問題点
  修正案: "修正後のテキスト案"
```
