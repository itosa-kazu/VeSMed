"""
Option 3 検証: 初回検索固定 + sim_matrix × 極性で重み更新
旧方式（全再embed）と比較
"""
from engine import VeSMedEngine
import numpy as np

print("エンジン初期化中...")
eng = VeSMedEngine()

# ========== 極性軸の構築 ==========
normal_anchor = '検査値は基準範囲内であり正常である。異常所見を認めない。陰性。検出されず。'
abnormal_anchor = '検査値は基準範囲を逸脱し異常である。異常所見あり。陽性。上昇。低下。検出。'

resp = eng.embed_client.embeddings.create(
    model='Qwen/Qwen3-Embedding-8B',
    input=[normal_anchor, abnormal_anchor],
)
anchor_embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
norms = np.linalg.norm(anchor_embs, axis=1, keepdims=True)
anchor_embs = anchor_embs / norms
polarity_axis = anchor_embs[1] - anchor_embs[0]
polarity_axis = polarity_axis / np.linalg.norm(polarity_axis)

# ========== 患者テキストと検査結果 ==========
base_text = """47歳女性
生来健康、学校給食職員で2児の母
来院2日前までは何ともなかった。来院1日前の昼に左脇～背中のあたりが痛くなり、夜間には寝返りを打てないくらいの痛みとなった。そのころから全身、腕や太腿の内側も痛くなり、胃液の嘔吐、水っぽい便がほぼ同時に出現した。
来院当日の早朝に痛みがひどく39度まで発熱したため、本人が救急車を要請した。"""

test_results = [
    'γ-GTP正常値',
    '直接ビリルビン正常値',
    'MRCP異常なし',
]

# ========== 初回検索（症状のみ） ==========
candidates = eng.search_diseases(base_text)
candidates = eng.compute_priors(candidates)

# 初回similarityを保存
for c in candidates:
    c['sim_initial'] = c['similarity']
    c['sim_updated'] = c['similarity']  # これを更新していく

# ========== 検査結果を1つずつ処理 ==========
all_result_lines = []
for result_line in test_results:
    all_result_lines.append(result_line)

    # 行をembed
    resp = eng.embed_client.embeddings.create(
        model='Qwen/Qwen3-Embedding-8B', input=[result_line],
    )
    line_emb = np.array(resp.data[0].embedding, dtype=np.float32)
    line_emb = line_emb / np.linalg.norm(line_emb)

    # 検査名マッチ
    sims_to_names = line_emb @ eng.test_name_embs.T
    best_j = int(np.argmax(sims_to_names))
    best_name = eng.test_names[best_j]
    best_sim = float(sims_to_names[best_j])

    # 極性
    polarity = float(line_emb @ polarity_axis)

    print(f"\n--- {result_line} ---")
    print(f"  マッチ検査: {best_name} (cos={best_sim:.3f})")
    print(f"  極性: {polarity:+.4f} ({'異常' if polarity > 0 else '正常'})")

    # sim_matrixで全疾患を更新
    for c in candidates:
        d_idx = eng.disease_idx.get(c['disease_name'])
        if d_idx is not None:
            sim_dt = float(eng.sim_matrix[d_idx, best_j])
            # 方式A: polarity × sim_matrix（連続、小さい更新）
            update_a = polarity * sim_dt
            # 方式B: sign(polarity) × sim_matrix（符号のみ、大きい更新）
            update_b = (1.0 if polarity > 0 else -1.0) * sim_dt
            c['sim_A'] = c.get('sim_A', c['sim_initial']) * np.exp(update_a)
            c['sim_B'] = c.get('sim_B', c['sim_initial']) * np.exp(update_b)

    # 旧方式: 全再embed
    full_text = base_text + '\n' + '\n'.join(all_result_lines)
    old_cands = eng.search_diseases(full_text)
    old_sim = {oc['disease_name']: oc['similarity'] for oc in old_cands}

    # Top 10を表示
    # 方式Bでソート
    candidates.sort(key=lambda x: x.get('sim_B', 0), reverse=True)

    print(f"\n  {'疾患名':<20s} | {'初回':>6s} | {'旧再emb':>7s} | {'A連続':>7s} | {'B符号':>7s}")
    print("  " + "-" * 60)
    for c in candidates[:10]:
        name = c['disease_name']
        init = c['sim_initial']
        old = old_sim.get(name, 0)
        a = c.get('sim_A', 0)
        b = c.get('sim_B', 0)
        print(f"  {name:<20s} | {init:.4f} | {old:.4f} | {a:.4f} | {b:.4f}")

# ========== 最終比較 ==========
print(f"\n{'='*70}")
print("最終結果（3つの正常結果入力後）")
print(f"{'='*70}")

full_text = base_text + '\n' + '\n'.join(test_results)
old_cands = eng.search_diseases(full_text)
old_sim = {oc['disease_name']: oc['similarity'] for oc in old_cands}

# 各方式でソートしてTop10
print(f"\n--- 旧方式（全再embed）Top 10 ---")
old_sorted = sorted(old_cands, key=lambda x: x['similarity'], reverse=True)
for i, c in enumerate(old_sorted[:10]):
    print(f"  {i+1:2d}. {c['disease_name']:<25s} sim={c['similarity']:.4f}  {c.get('category','')}")

print(f"\n--- 方式A（連続: polarity×sim）Top 10 ---")
candidates.sort(key=lambda x: x.get('sim_A', 0), reverse=True)
for i, c in enumerate(candidates[:10]):
    print(f"  {i+1:2d}. {c['disease_name']:<25s} sim={c.get('sim_A',0):.4f}  {c.get('category','')}")

print(f"\n--- 方式B（符号: sign×sim）Top 10 ---")
candidates.sort(key=lambda x: x.get('sim_B', 0), reverse=True)
for i, c in enumerate(candidates[:10]):
    print(f"  {i+1:2d}. {c['disease_name']:<25s} sim={c.get('sim_B',0):.4f}  {c.get('category','')}")

print(f"\n--- 初回検索（症状のみ、参考）Top 10 ---")
candidates.sort(key=lambda x: x['sim_initial'], reverse=True)
for i, c in enumerate(candidates[:10]):
    print(f"  {i+1:2d}. {c['disease_name']:<25s} sim={c['sim_initial']:.4f}  {c.get('category','')}")
