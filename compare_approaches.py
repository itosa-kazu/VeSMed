"""
3つのアプローチを比較検証:
  A: sim_matrix + 列平均除去（現行）
  B: Direct embedding（結果emb × 疾患emb、sim_matrix中間層なし）
  C: Full re-embed（症状+結果を連結して再検索）
"""
from engine import VeSMedEngine
import numpy as np
import copy

print("エンジン初期化中...")
eng = VeSMedEngine()

# 疾患embeddingをChromaDBから取得（Approach B用）
print("疾患embedding取得中...")
n_diseases = eng.collection.count()
all_ids = [f"disease_{i}" for i in range(n_diseases)]
disease_embs = {}
for start in range(0, len(all_ids), 100):
    batch_ids = all_ids[start:start + 100]
    result = eng.collection.get(ids=batch_ids, include=["embeddings", "metadatas"])
    for j, mid in enumerate(result["ids"]):
        dname = result["metadatas"][j].get("disease_name", "")
        emb = np.array(result["embeddings"][j], dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        disease_embs[dname] = emb
print(f"疾患embedding: {len(disease_embs)}件")

from config import EMBEDDING_MODEL


def approach_a(cands, result_lines, symptoms):
    """A: sim_matrix + 列平均除去（現行engine実装）"""
    c = copy.deepcopy(cands)
    return eng.update_from_results(c, result_lines, symptoms=symptoms, mode="fast")


def approach_b(cands, result_lines, symptoms):
    """B: Direct embedding — 結果emb × 疾患embを直接計算。sim_matrix不使用。"""
    c = copy.deepcopy(cands)
    if not result_lines or eng.polarity_axis is None:
        return c

    # アノテーション
    annotated = eng._annotate_with_ranges(result_lines)

    # embed
    resp = eng.embed_client.embeddings.create(model=EMBEDDING_MODEL, input=annotated)
    line_embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    l_norms = np.linalg.norm(line_embs, axis=1, keepdims=True)
    l_norms[l_norms == 0] = 1.0
    line_embs = line_embs / l_norms

    for k in range(len(result_lines)):
        lemb = line_embs[k]
        pol = float(lemb @ eng.polarity_axis)
        sign = 1.0 if pol > 0 else -1.0

        # 各疾患embeddingとの直接cosine
        relevances = []
        for ci in c:
            demb = disease_embs.get(ci['disease_name'])
            if demb is not None:
                rel = float(lemb @ demb)
                relevances.append(rel)
            else:
                relevances.append(0.0)
        # 平均を超える分だけ更新
        mean_rel = np.mean(relevances) if relevances else 0.0
        for i, ci in enumerate(c):
            excess = max(0.0, relevances[i] - mean_rel)
            if excess > 0:
                ci['similarity'] *= float(np.exp(sign * excess))

    c.sort(key=lambda x: x['similarity'], reverse=True)
    return c


def approach_b_raw(cands, result_lines, symptoms):
    """B2: Direct embedding — 中心化なし（生のcos類似度をそのまま使用）"""
    c = copy.deepcopy(cands)
    if not result_lines or eng.polarity_axis is None:
        return c

    annotated = eng._annotate_with_ranges(result_lines)
    resp = eng.embed_client.embeddings.create(model=EMBEDDING_MODEL, input=annotated)
    line_embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    l_norms = np.linalg.norm(line_embs, axis=1, keepdims=True)
    l_norms[l_norms == 0] = 1.0
    line_embs = line_embs / l_norms

    for k in range(len(result_lines)):
        lemb = line_embs[k]
        pol = float(lemb @ eng.polarity_axis)
        sign = 1.0 if pol > 0 else -1.0

        for ci in c:
            demb = disease_embs.get(ci['disease_name'])
            if demb is not None:
                rel = float(lemb @ demb)
                ci['similarity'] *= float(np.exp(sign * rel))

    c.sort(key=lambda x: x['similarity'], reverse=True)
    return c


def approach_c(cands, result_lines, symptoms):
    """C: Full re-embed — 症状+結果を連結して再検索"""
    full = symptoms + '\n' + '\n'.join(result_lines)
    return eng.search_diseases(full)


def find_rank(candidates, disease_name):
    sorted_c = sorted(candidates, key=lambda x: x.get('similarity', 0), reverse=True)
    for i, c in enumerate(sorted_c):
        if c['disease_name'] == disease_name:
            return i + 1
    return -1


CASES = [
    {
        "name": "A: 定性正常（γ-GTP正常、D-Bil正常、MRCP異常なし）",
        "symptoms": "47歳女性。来院1日前から左脇～背中の痛み、全身痛、嘔吐、水様便。当日39度の発熱。",
        "results": ['γ-GTP正常値', '直接ビリルビン正常値', 'MRCP異常なし'],
        "targets": [('劇症型溶血性レンサ球菌感染症（STSS）', 'STSS', 'up')],
        "wrong": [('総胆管結石', '総胆管結石', 'down'), ('急性胆管炎', '急性胆管炎', 'down')],
    },
    {
        "name": "B: 定性異常（血培陽性、心エコー疣贅、CRP上昇）",
        "symptoms": "67歳男性。7週間前から38度前後の繰り返す発熱。体重減少、食欲低下、全身倦怠感。",
        "results": ['血液培養: 黄色ブドウ球菌陽性', '心エコー: 大動脈弁に疣贅あり', 'CRP上昇'],
        "targets": [('感染性心内膜炎', 'IE', 'up')],
        "wrong": [('成人スティル病', 'スティル病', 'down')],
    },
    {
        "name": "D: 数値高値（WBC 18000, CRP 15）",
        "symptoms": "47歳女性。来院1日前から左脇～背中の痛み、全身痛、嘔吐、水様便。当日39度の発熱。",
        "results": ['WBC 18000', 'CRP 15'],
        "targets": [('劇症型溶血性レンサ球菌感染症（STSS）', 'STSS', 'up'), ('敗血症', '敗血症', 'up')],
        "wrong": [],
    },
    {
        "name": "E: 逆方向異常（WBC 2000, 体温 35度）",
        "symptoms": "65歳男性。3日前から発熱と悪寒。本日意識レベル低下、血圧低下で搬送。",
        "results": ['WBC 2000', '体温 35度'],
        "targets": [('敗血症', '敗血症', 'up'), ('敗血症性ショック', '敗血症性ショック', 'up')],
        "wrong": [],
    },
    {
        "name": "F: 正常数値（Na 140, Hb 14）",
        "symptoms": "55歳男性。胸痛で来院。30分前から突然の胸骨後部痛。冷汗あり。",
        "results": ['Na 140', 'Hb 14'],
        "targets": [('急性心筋梗塞 (STEMI：前壁/中隔)', 'AMI', 'stay')],
        "wrong": [],
    },
    {
        "name": "G: 混合（血培陽性 + 胸部X線正常 + 尿培陰性）",
        "symptoms": "72歳男性。5日前から38度台の発熱持続。悪寒戦慄を伴う。",
        "results": ['血液培養: 黄色ブドウ球菌陽性', '胸部X線異常なし', '尿培養陰性'],
        "targets": [('感染性心内膜炎', 'IE', 'up'), ('敗血症', '敗血症', 'stay')],
        "wrong": [('肺炎球菌性肺炎', '肺炎', 'down'), ('急性腎盂腎炎', '腎盂腎炎', 'down')],
    },
    {
        "name": "H: バイタル数値（体温 39.5, 血圧 80/50, 脈拍 120）",
        "symptoms": "70歳男性。尿路感染後に高熱、意識混濁。",
        "results": ['体温 39.5', '血圧 80/50', '脈拍 120'],
        "targets": [('敗血症', '敗血症', 'up'), ('敗血症性ショック', '敗血症性ショック', 'up')],
        "wrong": [],
    },
    {
        "name": "K: 腎機能（Cr 4.5, BUN 65, K 6.2）",
        "symptoms": "60歳男性。糖尿病歴20年。1週間前から全身浮腫、尿量減少。",
        "results": ['Cr 4.5', 'BUN 65', 'K 6.2'],
        "targets": [('慢性腎臓病 (CKD)', 'CKD', 'up'), ('糖尿病性腎症', '糖尿病性腎症', 'stay')],
        "wrong": [],
    },
    {
        "name": "L: 甲状腺（TSH 0.01, FT4 5.8）",
        "symptoms": "28歳女性。動悸、手指振戦、体重減少、発汗過多。眼球突出あり。",
        "results": ['TSH 0.01', 'Free T4 5.8'],
        "targets": [('バセドウ病', 'バセドウ病', 'stay')],
        "wrong": [],
    },
    {
        "name": "N: 数値混合（トロポニンT上昇 + Na正常 + Cr正常）",
        "symptoms": "65歳男性。2時間前から胸骨後部の激痛。冷汗、嘔気あり。",
        "results": ['トロポニンT 0.5', 'Na 141', 'Cr 0.8'],
        "targets": [('急性心筋梗塞 (STEMI：前壁/中隔)', 'AMI', 'stay'), ('不安定狭心症', 'ACS', 'up')],
        "wrong": [('慢性腎臓病 (CKD)', 'CKD', 'down')],
    },
]

# 全アプローチのラベル
APPROACHES = {
    "A:sim中心化": approach_a,
    "B:直接emb中心化": approach_b,
    "B2:直接emb生": approach_b_raw,
    "C:再embed": approach_c,
}

# ヘッダー
app_names = list(APPROACHES.keys())
header = f"  {'疾患':<18s} | {'初回':>4s}"
for name in app_names:
    header += f" | {name:>12s}"
header += " | 最良"
print(f"\n{'='*120}")
print(header)
print(f"  {'-'*116}")

total_scores = {name: 0 for name in app_names}

for case in CASES:
    # 初回検索
    cands = eng.search_diseases(case["symptoms"])
    cands = eng.compute_priors(cands)

    print(f"\n  [{case['name']}]")

    all_targets = [(n, l, d, False) for n, l, d in case["targets"]] + \
                  [(n, l, d, True) for n, l, d in case["wrong"]]

    for dname, label, direction, is_wrong in all_targets:
        r_init = find_rank(cands, dname)
        r_init_s = f"{r_init:4d}" if r_init > 0 else " N/A"

        line = f"  {label:<18s} | {r_init_s}"
        best_approach = ""
        best_score = -999

        for app_name, app_fn in APPROACHES.items():
            result = app_fn(cands, case["results"], case["symptoms"])
            r = find_rank(result, dname)
            r_s = f"{r:4d}" if r > 0 else " N/A"

            # スコア計算
            if r_init <= 0 or r <= 0:
                score = 0
                mark = " "
            elif is_wrong:
                # 誤診断: 沈下がOK
                if r > r_init:
                    score = 1
                    mark = "+"
                elif r < r_init:
                    score = -1
                    mark = "-"
                else:
                    score = 0
                    mark = "="
            else:
                # 正解: direction に基づく
                if direction == "up":
                    if r < r_init:
                        score = 1
                        mark = "+"
                    elif r > r_init:
                        score = -1
                        mark = "-"
                    else:
                        score = 0
                        mark = "="
                elif direction == "stay":
                    if r <= r_init:
                        score = 1
                        mark = "+"
                    else:
                        score = -1
                        mark = "-"
                else:
                    score = 0
                    mark = "="

            total_scores[app_name] += score
            line += f" | {r_s:>10s}{mark}"

            if score > best_score:
                best_score = score
                best_approach = app_name

        line += f" | {best_approach}" if best_score > 0 else " |"
        print(line)

# サマリー
print(f"\n{'='*120}")
print("  スコアサマリー（+1=正しく変化, 0=変化なし, -1=誤った変化）")
for name, score in total_scores.items():
    print(f"    {name:<20s}: {score:+d}")
print(f"{'='*120}")
