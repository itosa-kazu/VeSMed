import json
import numpy as np
from openai import OpenAI
from config import EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL
import chromadb
from config import CHROMA_DIR

# Load disease embeddings from ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection("diseases")

# Get all disease embeddings
all_data = collection.get(include=["embeddings", "metadatas", "documents"])
disease_names = [m["disease_name"] for m in all_data["metadatas"]]
disease_embs = np.array(all_data["embeddings"], dtype=np.float32)
d_norms = np.linalg.norm(disease_embs, axis=1, keepdims=True)
d_norms[d_norms == 0] = 1.0
disease_embs_n = disease_embs / d_norms
print(f"Diseases: {len(disease_names)}")

# Load tests
tests = []
with open('data/tests.jsonl', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            tests.append(json.loads(line))

# Use 20 representative tests for analysis
sample_names = ['CRP', '白血球数', 'ナトリウム (Na)', 'クレアチニン (Cr)',
                'HbA1c', '12誘導心電図', '抗核抗体 (ANA)', 'Dダイマー',
                'フェリチン', '血液培養 (好気/嫌気)', 'トロポニンI',
                'プロカルシトニン (PCT)', 'AST (GOT)', 'ALT (GPT)',
                '血小板数', 'ヘモグロビン', 'アミラーゼ (血清)', 'BNP',
                '尿酸 (UA)', 'カリウム (K)']
sample = [t for t in tests if t['test_name'] in sample_names]
# Sort to match sample_names order
sample_order = {n: i for i, n in enumerate(sample_names)}
sample.sort(key=lambda t: sample_order[t['test_name']])
print(f"Sample tests: {len(sample)}")

client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)

# Embed V1 and V3 descriptions
v1_texts = [t.get('findings_description_v1', t.get('findings_description', '')) for t in sample]
v3_texts = [t['findings_description'] for t in sample]

import time
resp_v1 = client.embeddings.create(model=EMBEDDING_MODEL, input=v1_texts)
time.sleep(2)
resp_v3 = client.embeddings.create(model=EMBEDDING_MODEL, input=v3_texts)

v1_embs = np.array([d.embedding for d in resp_v1.data], dtype=np.float32)
v3_embs = np.array([d.embedding for d in resp_v3.data], dtype=np.float32)

v1_n = v1_embs / np.linalg.norm(v1_embs, axis=1, keepdims=True)
v3_n = v3_embs / np.linalg.norm(v3_embs, axis=1, keepdims=True)

# Compute partial sim_matrices
sim_v1 = disease_embs_n @ v1_n.T  # (386, 20)
sim_v3 = disease_embs_n @ v3_n.T  # (386, 20)

print(f"\nsim_matrix shape: {sim_v1.shape}")

# Analysis 1: Column statistics
print("\n=== sim_matrix列統計（各テストの全疾患に対する類似度） ===")
print(f"{'Test':<30s} {'V1_mean':>8s} {'V3_mean':>8s} {'V1_std':>8s} {'V3_std':>8s} {'V1_max':>8s} {'V3_max':>8s}")
for j, t in enumerate(sample):
    name = t['test_name'][:28]
    v1_col = sim_v1[:, j]
    v3_col = sim_v3[:, j]
    print(f"{name:<30s} {v1_col.mean():8.4f} {v3_col.mean():8.4f} "
          f"{v1_col.std():8.4f} {v3_col.std():8.4f} "
          f"{v1_col.max():8.4f} {v3_col.max():8.4f}")

# Analysis 2: Excess distribution (key for update_from_results)
print("\n=== Excess分布 (excess = max(0, sim[d][j] - col_mean[j])) ===")
col_means_v1 = sim_v1.mean(axis=0)
col_means_v3 = sim_v3.mean(axis=0)

excess_v1_all = []
excess_v3_all = []
for j in range(len(sample)):
    exc_v1 = np.maximum(0, sim_v1[:, j] - col_means_v1[j])
    exc_v3 = np.maximum(0, sim_v3[:, j] - col_means_v3[j])
    excess_v1_all.extend(exc_v1.tolist())
    excess_v3_all.extend(exc_v3.tolist())

exc_v1 = np.array(excess_v1_all)
exc_v3 = np.array(excess_v3_all)

print(f"V1 excess: mean={exc_v1.mean():.6f}, std={exc_v1.std():.6f}, max={exc_v1.max():.4f}")
print(f"V3 excess: mean={exc_v3.mean():.6f}, std={exc_v3.std():.6f}, max={exc_v3.max():.4f}")
print(f"V1 non-zero count: {(exc_v1 > 0).sum()}/{len(exc_v1)} ({(exc_v1 > 0).mean()*100:.1f}%)")
print(f"V3 non-zero count: {(exc_v3 > 0).sum()}/{len(exc_v3)} ({(exc_v3 > 0).mean()*100:.1f}%)")
print(f"V1 excess > 0.01: {(exc_v1 > 0.01).sum()} ({(exc_v1 > 0.01).mean()*100:.1f}%)")
print(f"V3 excess > 0.01: {(exc_v3 > 0.01).sum()} ({(exc_v3 > 0.01).mean()*100:.1f}%)")
print(f"V1 excess > 0.02: {(exc_v1 > 0.02).sum()} ({(exc_v1 > 0.02).mean()*100:.1f}%)")
print(f"V3 excess > 0.02: {(exc_v3 > 0.02).sum()} ({(exc_v3 > 0.02).mean()*100:.1f}%)")

# Analysis 3: For specific disease-test pairs, show the excess change
print("\n=== 具体例: 特定疾患×検査のexcess変化 ===")
pairs = [
    ('急性心筋梗塞 (STEMI：前壁/中隔)', 'トロポニンI'),
    ('急性心筋梗塞 (STEMI：前壁/中隔)', 'CRP'),
    ('敗血症', 'CRP'),
    ('敗血症', 'ナトリウム (Na)'),
    ('全身性エリテマトーデス', '抗核抗体 (ANA)'),
    ('全身性エリテマトーデス', 'CRP'),
    ('糖尿病性ケトアシドーシス', 'HbA1c'),
    ('糖尿病性ケトアシドーシス', 'ナトリウム (Na)'),
    ('急性膵炎', 'アミラーゼ (血清)'),
    ('急性膵炎', 'CRP'),
]

test_idx_map = {t['test_name']: j for j, t in enumerate(sample)}
disease_idx_map = {n: i for i, n in enumerate(disease_names)}

for d_name, t_name in pairs:
    d_i = disease_idx_map.get(d_name)
    t_j = test_idx_map.get(t_name)
    if d_i is None or t_j is None:
        print(f"  {d_name} × {t_name}: NOT FOUND (d_i={d_i}, t_j={t_j})")
        continue
    sv1 = float(sim_v1[d_i, t_j])
    sv3 = float(sim_v3[d_i, t_j])
    ev1 = max(0, sv1 - float(col_means_v1[t_j]))
    ev3 = max(0, sv3 - float(col_means_v3[t_j]))
    print(f"  {d_name} × {t_name}:")
    print(f"    V1: sim={sv1:.4f}, col_mean={col_means_v1[t_j]:.4f}, excess={ev1:.4f} -> exp(excess)={np.exp(ev1):.4f}")
    print(f"    V3: sim={sv3:.4f}, col_mean={col_means_v3[t_j]:.4f}, excess={ev3:.4f} -> exp(excess)={np.exp(ev3):.4f}")
    print(f"    差: sim {sv3-sv1:+.4f}, excess {ev3-ev1:+.4f}")

# Analysis 4: Column variance ratio (discriminative power per test)
print("\n=== テスト別弁別力（列分散） ===")
print(f"{'Test':<30s} {'V1_var':>10s} {'V3_var':>10s} {'ratio':>8s}")
for j, t in enumerate(sample):
    v1_var = sim_v1[:, j].var()
    v3_var = sim_v3[:, j].var()
    ratio = v3_var / v1_var if v1_var > 0 else 0
    print(f"{t['test_name'][:28]:<30s} {v1_var:10.6f} {v3_var:10.6f} {ratio:8.3f}")
