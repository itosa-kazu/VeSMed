"""
Option 3 エビデンス検証
旧方式（全再embed）vs 新方式（症状固定 + sim_matrix更新）
2症例 × 正常/異常結果で検証
"""
from engine import VeSMedEngine
import numpy as np

print("エンジン初期化中...")
eng = VeSMedEngine()

# ========== 極性軸 ==========
normal_anchor = '検査値は基準範囲内であり正常である。異常所見を認めない。陰性。検出されず。'
abnormal_anchor = '検査値は基準範囲を逸脱し異常である。異常所見あり。陽性。上昇。低下。検出。'
resp = eng.embed_client.embeddings.create(
    model='Qwen/Qwen3-Embedding-8B', input=[normal_anchor, abnormal_anchor],
)
aembs = np.array([d.embedding for d in resp.data], dtype=np.float32)
anorms = np.linalg.norm(aembs, axis=1, keepdims=True)
aembs = aembs / anorms
polarity_axis = aembs[1] - aembs[0]
polarity_axis = polarity_axis / np.linalg.norm(polarity_axis)


def option3_update(candidates, result_lines):
    """Option 3: sim_matrix × sign(polarity) で重み更新"""
    # 初回similarityをコピー
    for c in candidates:
        c['sim_new'] = c['similarity']

    for line in result_lines:
        resp = eng.embed_client.embeddings.create(
            model='Qwen/Qwen3-Embedding-8B', input=[line],
        )
        lemb = np.array(resp.data[0].embedding, dtype=np.float32)
        lemb = lemb / np.linalg.norm(lemb)

        # 検査名マッチ
        sims = lemb @ eng.test_name_embs.T
        best_j = int(np.argmax(sims))
        best_name = eng.test_names[best_j]
        match_sim = float(sims[best_j])

        # 極性
        pol = float(lemb @ polarity_axis)
        sign = 1.0 if pol > 0 else -1.0

        print(f"    {line:<35s} → {best_name:<20s} (cos={match_sim:.3f}) pol={pol:+.4f} sign={sign:+.0f}")

        for c in candidates:
            d_idx = eng.disease_idx.get(c['disease_name'])
            if d_idx is not None:
                sim_dt = float(eng.sim_matrix[d_idx, best_j])
                c['sim_new'] *= np.exp(sign * sim_dt)

    return candidates


def find_rank(candidates, disease_name, key='similarity'):
    """特定疾患の順位を返す"""
    sorted_c = sorted(candidates, key=lambda x: x.get(key, 0), reverse=True)
    for i, c in enumerate(sorted_c):
        if c['disease_name'] == disease_name:
            return i + 1
    return -1


def show_top10(candidates, key, label):
    sorted_c = sorted(candidates, key=lambda x: x.get(key, 0), reverse=True)
    print(f"\n  --- {label} Top 10 ---")
    for i, c in enumerate(sorted_c[:10]):
        print(f"  {i+1:2d}. {c['disease_name']:<30s} {c.get(key,0):.4f}  {c.get('category','')}")


# ================================================================
# 症例1: 47F STSS疑い — 正常結果で肝胆道を除外
# ================================================================
print("\n" + "=" * 70)
print("症例1: 47F 急性腹症 → 正常結果で肝胆道を除外")
print("正解: STSS / 壊死性筋膜炎")
print("=" * 70)

symptoms1 = """47歳女性
生来健康、学校給食職員で2児の母
来院2日前までは何ともなかった。来院1日前の昼に左脇～背中のあたりが痛くなり、夜間には寝返りを打てないくらいの痛みとなった。そのころから全身、腕や太腿の内側も痛くなり、胃液の嘔吐、水っぽい便がほぼ同時に出現した。
来院当日の早朝に痛みがひどく39度まで発熱したため、本人が救急車を要請した。"""

results1 = [
    'γ-GTP正常値',
    '直接ビリルビン正常値',
    'MRCP異常なし',
]

target1a = '劇症型溶血性レンサ球菌感染症（STSS）'
target1b = '壊死性筋膜炎'
wrong1 = '総胆管結石'

# 初回検索
cands1 = eng.search_diseases(symptoms1)
cands1 = eng.compute_priors(cands1)
print(f"\n  初回検索（症状のみ）:")
print(f"    STSS: rank {find_rank(cands1, target1a)}")
print(f"    壊死性筋膜炎: rank {find_rank(cands1, target1b)}")
print(f"    総胆管結石: rank {find_rank(cands1, wrong1)}")
show_top10(cands1, 'similarity', '初回（症状のみ）')

# 旧方式
full1 = symptoms1 + '\n' + '\n'.join(results1)
old1 = eng.search_diseases(full1)
print(f"\n  旧方式（全再embed）:")
print(f"    STSS: rank {find_rank(old1, target1a)}")
print(f"    壊死性筋膜炎: rank {find_rank(old1, target1b)}")
print(f"    総胆管結石: rank {find_rank(old1, wrong1)}")
show_top10(old1, 'similarity', '旧方式（全再embed）')

# 新方式
print(f"\n  新方式（Option 3）検査結果処理:")
option3_update(cands1, results1)
print(f"\n  新方式（Option 3）:")
print(f"    STSS: rank {find_rank(cands1, target1a, 'sim_new')}")
print(f"    壊死性筋膜炎: rank {find_rank(cands1, target1b, 'sim_new')}")
print(f"    総胆管結石: rank {find_rank(cands1, wrong1, 'sim_new')}")
show_top10(cands1, 'sim_new', '新方式（Option 3）')


# ================================================================
# 症例2: 67M IE疑い — 異常結果で正解をブースト
# ================================================================
print("\n\n" + "=" * 70)
print("症例2: 67M 繰り返す発熱 → 異常結果でIEをブースト")
print("正解: 感染性心内膜炎")
print("=" * 70)

symptoms2 = """67歳の男性。繰り返す発熱を主訴に来院。
7週間前から38℃前後の発熱が出現し、市販の解熱薬で一時的に解熱するが再度発熱する。
体重減少あり。食欲低下。全身倦怠感。"""

results2 = [
    '血液培養: 黄色ブドウ球菌陽性',
    '心エコー: 大動脈弁に疣贅あり',
    'CRP上昇',
]

target2 = '感染性心内膜炎'

# 初回検索
cands2 = eng.search_diseases(symptoms2)
cands2 = eng.compute_priors(cands2)
print(f"\n  初回検索（症状のみ）:")
print(f"    感染性心内膜炎: rank {find_rank(cands2, target2)}")
show_top10(cands2, 'similarity', '初回（症状のみ）')

# 旧方式
full2 = symptoms2 + '\n' + '\n'.join(results2)
old2 = eng.search_diseases(full2)
print(f"\n  旧方式（全再embed）:")
print(f"    感染性心内膜炎: rank {find_rank(old2, target2)}")
show_top10(old2, 'similarity', '旧方式（全再embed）')

# 新方式
print(f"\n  新方式（Option 3）検査結果処理:")
option3_update(cands2, results2)
print(f"\n  新方式（Option 3）:")
print(f"    感染性心内膜炎: rank {find_rank(cands2, target2, 'sim_new')}")
show_top10(cands2, 'sim_new', '新方式（Option 3）')


# ================================================================
# サマリー
# ================================================================
print("\n\n" + "=" * 70)
print("サマリー: 正解疾患の順位比較")
print("=" * 70)
print(f"\n{'症例':<30s} | {'初回':>4s} | {'旧方式':>6s} | {'新方式':>6s}")
print("-" * 55)
print(f"{'1: STSS':<30s} | {find_rank(cands1, target1a):>4d} | {find_rank(old1, target1a):>6d} | {find_rank(cands1, target1a, 'sim_new'):>6d}")
print(f"{'1: 壊死性筋膜炎':<30s} | {find_rank(cands1, target1b):>4d} | {find_rank(old1, target1b):>6d} | {find_rank(cands1, target1b, 'sim_new'):>6d}")
print(f"{'1: 総胆管結石(誤)':<30s} | {find_rank(cands1, wrong1):>4d} | {find_rank(old1, wrong1):>6d} | {find_rank(cands1, wrong1, 'sim_new'):>6d}")
print(f"{'2: 感染性心内膜炎':<30s} | {find_rank(cands2, target2):>4d} | {find_rank(old2, target2):>6d} | {find_rank(cands2, target2, 'sim_new'):>6d}")
