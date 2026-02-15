"""
Experiment: Direct result-embedding vs disease-embedding cosine similarity
Hypothesis: We can skip test embeddings entirely and directly compare
annotated test result embeddings against disease embeddings.

Method B (proposed): result → embed → cos_sim(result_emb, disease_emb[d]) → excess → update
"""
import json
import numpy as np
import os
from openai import OpenAI
import chromadb
from config import EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL, CHROMA_DIR

# Load disease embeddings
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection("diseases")
all_data = collection.get(include=["embeddings", "metadatas"])
disease_names = [m["disease_name"] for m in all_data["metadatas"]]
disease_embs = np.array(all_data["embeddings"], dtype=np.float32)
d_norms = np.linalg.norm(disease_embs, axis=1, keepdims=True)
d_norms[d_norms == 0] = 1.0
disease_embs_n = disease_embs / d_norms
disease_idx = {n: i for i, n in enumerate(disease_names)}
print(f"Diseases: {len(disease_names)}")

client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)

# Simulate annotated test results (as the engine would produce them)
test_results = [
    ("CRP 上昇", "敗血症", ["急性心筋梗塞 (STEMI：前壁/中隔)", "慢性腎臓病 (CKD)"]),
    ("トロポニンT 上昇", "急性心筋梗塞 (STEMI：前壁/中隔)", ["敗血症", "慢性腎臓病 (CKD)"]),
    ("抗核抗体 陽性 高力価", "全身性エリテマトーデス", ["敗血症", "急性心筋梗塞 (STEMI：前壁/中隔)"]),
    ("HbA1c 上昇 10%", "糖尿病性ケトアシドーシス", ["急性心筋梗塞 (STEMI：前壁/中隔)", "全身性エリテマトーデス"]),
    ("アミラーゼ 著明上昇", "急性膵炎", ["敗血症", "糖尿病性ケトアシドーシス"]),
    ("ナトリウム 正常 異常なし", "急性心筋梗塞 (STEMI：前壁/中隔)", ["慢性腎臓病 (CKD)", "SIADH"]),
    ("白血球数 上昇 18000", "敗血症", ["全身性エリテマトーデス", "再生不良性貧血"]),
    ("Dダイマー 上昇", "肺血栓塞栓症", ["急性心筋梗塞 (STEMI：前壁/中隔)", "慢性腎臓病 (CKD)"]),
    ("血液培養 黄色ブドウ球菌 陽性", "感染性心内膜炎", ["敗血症", "急性心筋梗塞 (STEMI：前壁/中隔)"]),
    ("髄液 細胞数増加 蛋白上昇", "単純ヘルペス脳炎", ["細菌性髄膜炎", "ウイルス性髄膜炎"]),
]

# Embed all result texts
result_texts = [r[0] for r in test_results]
print(f"\nEmbedding {len(result_texts)} result texts...")
resp = client.embeddings.create(model=EMBEDDING_MODEL, input=result_texts)
result_embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
r_norms = np.linalg.norm(result_embs, axis=1, keepdims=True)
result_embs_n = result_embs / r_norms
print(f"Result embeddings shape: {result_embs_n.shape}")

# Try to load sim_matrix for comparison (optional)
sim_matrix_path = os.path.join('data', 'sim_matrix.npz')
has_sim_matrix = os.path.exists(sim_matrix_path)
if has_sim_matrix:
    sim_data = np.load(sim_matrix_path, allow_pickle=True)
    sim_matrix = sim_data['sim_matrix']
    test_names_sm = list(sim_data['test_names'])
    disease_names_sm = list(sim_data['disease_names'])
    disease_idx_sm = {n: i for i, n in enumerate(disease_names_sm)}
    test_idx_sm = {n: i for i, n in enumerate(test_names_sm)}
    print(f"sim_matrix shape: {sim_matrix.shape}")
else:
    print("sim_matrix.npz not found - skipping Method A comparison")

# For each test result, analyze direct method
print("\n" + "=" * 70)
print("=== Method B (direct result→disease cosine similarity) ===")
print("=" * 70)

for result_text, correct_disease, wrong_diseases in test_results:
    print(f"\n--- {result_text} → 正解: {correct_disease} ---")

    # Get result embedding index
    r_idx = result_texts.index(result_text)
    r_emb = result_embs_n[r_idx]

    # Method B: direct cosine similarity with all diseases
    direct_sims = disease_embs_n @ r_emb  # (N_diseases,)
    direct_mean = float(direct_sims.mean())
    direct_std = float(direct_sims.std())

    # Get disease indices
    all_check = [correct_disease] + wrong_diseases

    print(f"  Global stats: mean={direct_mean:.4f}, std={direct_std:.4f}")

    for d_name in all_check:
        d_i_direct = disease_idx.get(d_name)
        if d_i_direct is None:
            print(f"  {d_name}: NOT FOUND in disease DB")
            continue

        # Method B
        direct_sim = float(direct_sims[d_i_direct])
        direct_excess = max(0, direct_sim - direct_mean)

        label = "CORRECT" if d_name == correct_disease else "wrong"
        print(f"  [{label:7s}] {d_name}:")
        print(f"    Direct: sim={direct_sim:.4f}, excess={direct_excess:.4f}, exp(excess)={np.exp(direct_excess):.4f}")

    # Show top 10 diseases by direct excess
    direct_excess_all = np.maximum(0, direct_sims - direct_mean)
    top_indices = np.argsort(-direct_excess_all)[:10]

    # Find rank of correct disease
    sorted_indices = np.argsort(-direct_excess_all)
    correct_d_i = disease_idx.get(correct_disease)
    if correct_d_i is not None:
        correct_rank = int(np.where(sorted_indices == correct_d_i)[0][0]) + 1
    else:
        correct_rank = -1

    print(f"  正解疾患ランク: {correct_rank}/{len(disease_names)}")
    print(f"  Top 10 by direct excess:")
    for rank, idx in enumerate(top_indices):
        d_name = disease_names[idx]
        marker = " <<<" if d_name == correct_disease else ""
        print(f"    {rank+1}. {d_name}: sim={direct_sims[idx]:.4f}, excess={direct_excess_all[idx]:.4f}{marker}")

# === Discriminative power analysis ===
print("\n\n" + "=" * 70)
print("=== 弁別力比較 (Discriminative Power) ===")
print("=" * 70)
print("(正解疾患のexcess - 不正解疾患の最大excess)")
print()

gaps = []
ranks = []

for result_text, correct_disease, wrong_diseases in test_results:
    r_idx = result_texts.index(result_text)
    r_emb = result_embs_n[r_idx]

    # Method B: direct
    direct_sims = disease_embs_n @ r_emb
    direct_mean = float(direct_sims.mean())

    d_i = disease_idx.get(correct_disease)
    correct_excess_direct = max(0, float(direct_sims[d_i]) - direct_mean)

    max_wrong_excess_direct = 0
    for wd in wrong_diseases:
        d_j = disease_idx.get(wd)
        if d_j is not None:
            wrong_exc = max(0, float(direct_sims[d_j]) - direct_mean)
            max_wrong_excess_direct = max(max_wrong_excess_direct, wrong_exc)

    gap_direct = correct_excess_direct - max_wrong_excess_direct
    gaps.append(gap_direct)

    # Rank among all diseases
    direct_excess_all = np.maximum(0, direct_sims - direct_mean)
    sorted_indices = np.argsort(-direct_excess_all)
    correct_rank = int(np.where(sorted_indices == d_i)[0][0]) + 1
    ranks.append(correct_rank)

    status = "OK" if gap_direct > 0 else "FAIL"
    print(f"  [{status:4s}] {result_text}: correct_exc={correct_excess_direct:.4f}, worst_wrong={max_wrong_excess_direct:.4f}, gap={gap_direct:+.4f}, rank={correct_rank}")

print(f"\n  --- Summary ---")
print(f"  Cases with positive gap (correct > all wrong): {sum(1 for g in gaps if g > 0)}/{len(gaps)}")
print(f"  Mean gap: {np.mean(gaps):+.4f}")
print(f"  Min gap:  {np.min(gaps):+.4f}")
print(f"  Max gap:  {np.max(gaps):+.4f}")
print(f"  Mean rank of correct disease: {np.mean(ranks):.1f}")
print(f"  Median rank: {np.median(ranks):.1f}")
print(f"  Worst rank: {np.max(ranks)}")
print(f"  Rank <= 10: {sum(1 for r in ranks if r <= 10)}/{len(ranks)}")
print(f"  Rank <= 20: {sum(1 for r in ranks if r <= 20)}/{len(ranks)}")
print(f"  Rank <= 50: {sum(1 for r in ranks if r <= 50)}/{len(ranks)}")

# === Extra analysis: similarity distribution ===
print("\n\n" + "=" * 70)
print("=== 類似度分布分析 (Similarity Distribution) ===")
print("=" * 70)

for result_text, correct_disease, wrong_diseases in test_results:
    r_idx = result_texts.index(result_text)
    r_emb = result_embs_n[r_idx]
    direct_sims = disease_embs_n @ r_emb

    d_i = disease_idx.get(correct_disease)
    correct_sim = float(direct_sims[d_i])
    percentile = float((direct_sims < correct_sim).sum()) / len(direct_sims) * 100

    print(f"  {result_text}:")
    print(f"    min={direct_sims.min():.4f}, p25={np.percentile(direct_sims,25):.4f}, "
          f"median={np.median(direct_sims):.4f}, p75={np.percentile(direct_sims,75):.4f}, "
          f"max={direct_sims.max():.4f}")
    print(f"    correct_sim={correct_sim:.4f} (percentile={percentile:.1f}%)")
