"""
波動表現（Wave Representation）テスト
===========================================
現行方式（MEAN cosine + multiplicative update）
vs 波動方式（interference: pos_emb - neg_emb + Σ hpe_embs）
を12臨床ケースで比較検証。

検証ポイント:
1. 陰性所見追加で該当疾患のスコアが適切に下がるか
2. 陽性所見追加で該当疾患のスコアが適切に上がるか
3. 陽性+陰性の組み合わせで正しいランキング変動が起きるか
"""

import copy
import numpy as np
from engine import VeSMedEngine

print("=" * 80)
print("波動表現（Wave Representation）効果検証テスト")
print("=" * 80)

eng = VeSMedEngine()
dim = eng.disease_embs_normed.shape[1]


def get_disease_rank(candidates, disease_name):
    """候補リストでの疾患の順位とスコアを返す"""
    for i, c in enumerate(candidates):
        if c["disease_name"] == disease_name:
            return i + 1, c["similarity"]
    return None, None


def run_current_method(eng, text, hpe_findings):
    """現行方式: search → compute_priors → update_from_hpe (multiplicative)"""
    candidates = eng.search_diseases(text)
    candidates = eng.compute_priors(candidates)
    query_emb = eng._last_query_embedding.copy()
    q_norm = np.linalg.norm(query_emb)
    if q_norm > 0:
        query_emb /= q_norm

    if hpe_findings:
        candidates = eng.update_from_hpe(candidates, hpe_findings)
    return candidates, query_emb


def run_wave_method(eng, text, hpe_findings):
    """波動方式: embed → query_wave ± hpe_embs → interference similarity"""
    # 初期クエリembedding取得
    candidates = eng.search_diseases(text)
    query_wave = eng._last_query_embedding.copy()
    q_norm = np.linalg.norm(query_wave)
    if q_norm > 0:
        query_wave /= q_norm

    # HPE所見で波動修正
    if hpe_findings and eng.hpe_hyp_embs is not None:
        for f in hpe_findings:
            idx = f["index"]
            hpe_emb = eng.hpe_hyp_embs[idx]  # 正規化済み仮説embedding
            if f["polarity"] > 0:
                query_wave = query_wave + hpe_emb    # 建設的干渉
            else:
                query_wave = query_wave - hpe_emb    # 破壊的干渉

        # 再正規化
        w_norm = np.linalg.norm(query_wave)
        if w_norm > 0:
            query_wave /= w_norm

    # 干渉ベースの類似度再計算
    sims = query_wave @ eng.disease_embs_normed.T

    # 候補再構築
    idx_to_name = [""] * len(eng.disease_idx)
    for dname, idx in eng.disease_idx.items():
        idx_to_name[idx] = dname

    wave_candidates = []
    for i, sim in enumerate(sims):
        dname = idx_to_name[i]
        if not dname:
            continue
        meta = eng.disease_db.get(dname, {})
        wave_candidates.append({
            "disease_name": dname,
            "similarity": float(sim),
            "category": meta.get("category", ""),
            "urgency": meta.get("urgency", ""),
            "clinical_weight": eng.disease_2c.get(dname, {}).get("weight", 1.0),
        })

    wave_candidates.sort(key=lambda c: c["similarity"], reverse=True)
    return wave_candidates, query_wave


# ============================================================
# テストケース定義
# ============================================================
test_cases = [
    {
        "name": "急性虫垂炎 — 疝痛陰性で尿管結石を除外",
        "text": "35歳男性。右下腹部痛、発熱38.5℃、食欲不振。",
        "hpe": [
            {"item": "筋性防御", "polarity": 1},
            {"item": "疝痛・間欠的激痛", "polarity": -1},
        ],
        "expect_up": "急性虫垂炎",
        "expect_down": "尿管結石",
    },
    {
        "name": "尿管結石 — 疝痛陽性+筋性防御陰性で虫垂炎を下げる",
        "text": "45歳男性。左側腹部の激痛、突然発症、間欠的。",
        "hpe": [
            {"item": "疝痛・間欠的激痛", "polarity": 1},
            {"item": "筋性防御", "polarity": -1},
            {"item": "腰背部叩打痛", "polarity": 1},
        ],
        "expect_up": "尿管結石",
        "expect_down": "急性虫垂炎",
    },
    {
        "name": "急性胆嚢炎 — Murphy陽性 + 左下腹部痛陰性",
        "text": "55歳女性。右上腹部痛、悪心、食後に増悪。",
        "hpe": [
            {"item": "Murphy徴候", "polarity": 1},
            {"item": "左下腹部痛", "polarity": -1},
        ],
        "expect_up": "急性胆嚢炎",
        "expect_down": "大腸憩室炎",
    },
    {
        "name": "細菌性髄膜炎 — 項部硬直陽性 + 関節痛陰性",
        "text": "28歳男性。激しい頭痛、高熱39.5℃、羞明。",
        "hpe": [
            {"item": "項部硬直", "polarity": 1},
            {"item": "Kernig徴候", "polarity": 1},
            {"item": "関節痛（急性）", "polarity": -1},
        ],
        "expect_up": "細菌性髄膜炎",
        "expect_down": "関節リウマチ",
    },
    {
        "name": "肺塞栓症 — 片側下肢腫脹陽性 + 胸膜摩擦音陰性",
        "text": "60歳女性。突然の呼吸困難、胸痛、長時間のフライト後。",
        "hpe": [
            {"item": "片側下肢腫脹（深部静脈血栓）", "polarity": 1},
            {"item": "胸膜摩擦音", "polarity": -1},
        ],
        "expect_up": "肺塞栓症",
        "expect_down": "胸膜炎",
    },
    {
        "name": "陰性所見のみ — 3つの陰性で急性腹症を絞り込み",
        "text": "40歳男性。腹痛、嘔気。",
        "hpe": [
            {"item": "筋性防御", "polarity": -1},
            {"item": "反跳痛（Blumberg徴候）", "polarity": -1},
            {"item": "腸蠕動音消失", "polarity": -1},
        ],
        "expect_up": None,
        "expect_down": "急性虫垂炎",
    },
    {
        "name": "感染性心内膜炎 — 心雑音陽性 + 関節痛陰性",
        "text": "32歳男性。繰り返す発熱、全身倦怠感、体重減少。",
        "hpe": [
            {"item": "心雑音", "polarity": 1},
            {"item": "関節痛（急性）", "polarity": -1},
        ],
        "expect_up": "感染性心内膜炎",
        "expect_down": None,
    },
    {
        "name": "機械的イレウス — 腸蠕動音亢進 + 腸蠕動音消失陰性",
        "text": "70歳男性。腹部膨満、嘔吐、排便停止。開腹手術歴あり。",
        "hpe": [
            {"item": "腸蠕動音亢進/金属音", "polarity": 1},
            {"item": "腸蠕動音消失", "polarity": -1},
        ],
        "expect_up": "腸閉塞（イレウス）",
        "expect_down": None,
    },
    {
        "name": "大量の陽性所見 — 虫垂炎方向に収束",
        "text": "25歳男性。腹痛。",
        "hpe": [
            {"item": "右下腹部痛", "polarity": 1},
            {"item": "McBurney圧痛", "polarity": 1},
            {"item": "筋性防御", "polarity": 1},
            {"item": "反跳痛（Blumberg徴候）", "polarity": 1},
            {"item": "発熱（急性：1週未満）", "polarity": 1},
        ],
        "expect_up": "急性虫垂炎",
        "expect_down": None,
    },
    {
        "name": "心筋梗塞 — 胸痛陽性 + 呼吸音異常陰性",
        "text": "65歳男性。胸部絞扼感、左肩への放散痛、冷汗。",
        "hpe": [
            {"item": "胸痛（急性）", "polarity": 1},
            {"item": "呼吸音減弱", "polarity": -1},
        ],
        "expect_up": "急性心筋梗塞",
        "expect_down": "気胸",
    },
    {
        "name": "陰性のみ3件 — 広範な除外",
        "text": "50歳女性。全身倦怠感、微熱。",
        "hpe": [
            {"item": "関節痛（急性）", "polarity": -1},
            {"item": "皮疹", "polarity": -1},
            {"item": "リンパ節腫脹", "polarity": -1},
        ],
        "expect_up": None,
        "expect_down": "全身性エリテマトーデス（SLE）",
    },
    {
        "name": "混合所見5件 — 陽性3+陰性2",
        "text": "45歳女性。上腹部痛、嘔吐。",
        "hpe": [
            {"item": "心窩部圧痛", "polarity": 1},
            {"item": "Murphy徴候", "polarity": -1},
            {"item": "悪心・嘔吐", "polarity": 1},
            {"item": "下痢（急性：2週未満）", "polarity": -1},
            {"item": "発熱（急性：1週未満）", "polarity": 1},
        ],
        "expect_up": None,
        "expect_down": "急性胆嚢炎",
    },
]


# ============================================================
# テスト実行
# ============================================================
results = []

for i, tc in enumerate(test_cases):
    print(f"\n{'─' * 70}")
    print(f"Case {i+1}: {tc['name']}")
    print(f"テキスト: {tc['text']}")

    # HPE findings構築
    hpe_findings = []
    for h in tc["hpe"]:
        idx = eng.hpe_idx.get(h["item"])
        if idx is not None:
            hpe_findings.append({"item": h["item"], "index": idx, "polarity": h["polarity"]})
        else:
            print(f"  ⚠ HPE項目 '{h['item']}' が見つかりません")

    pol_str = ", ".join(f"{h['item']}({'+'if h['polarity']>0 else '-'})" for h in hpe_findings)
    print(f"HPE: {pol_str}")

    # ベースライン（HPE更新なし）
    base_cands, _ = run_current_method(eng, tc["text"], [])

    # 現行方式
    curr_cands, _ = run_current_method(eng, tc["text"], hpe_findings)

    # 波動方式
    wave_cands, _ = run_wave_method(eng, tc["text"], hpe_findings)

    # 比較
    print(f"\n  {'疾患':<25s} {'Base':>8s} {'現行':>8s} {'波動':>8s}  {'現行Δ':>8s} {'波動Δ':>8s}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8}  {'─'*8} {'─'*8}")

    check_diseases = set()
    if tc["expect_up"]:
        check_diseases.add(tc["expect_up"])
    if tc["expect_down"]:
        check_diseases.add(tc["expect_down"])
    # Top5 from each method
    for c in base_cands[:5]:
        check_diseases.add(c["disease_name"])
    for c in wave_cands[:5]:
        check_diseases.add(c["disease_name"])

    base_map = {c["disease_name"]: c["similarity"] for c in base_cands}
    curr_map = {c["disease_name"]: c["similarity"] for c in curr_cands}
    wave_map = {c["disease_name"]: c["similarity"] for c in wave_cands}

    for dname in sorted(check_diseases):
        b = base_map.get(dname, 0)
        c = curr_map.get(dname, 0)
        w = wave_map.get(dname, 0)
        d_curr = ((c - b) / b * 100) if b > 0 else 0
        d_wave = ((w - b) / b * 100) if b > 0 else 0
        marker = ""
        if dname == tc.get("expect_up"):
            marker = " ↑expect"
        elif dname == tc.get("expect_down"):
            marker = " ↓expect"
        print(f"  {dname:<25s} {b:8.4f} {c:8.4f} {w:8.4f}  {d_curr:+7.1f}% {d_wave:+7.1f}%{marker}")

    # 判定
    result = {"case": tc["name"], "current_win": False, "wave_win": False}

    if tc["expect_up"]:
        _, base_rank_up = get_disease_rank(base_cands, tc["expect_up"])
        curr_rank_up, _ = get_disease_rank(curr_cands, tc["expect_up"])
        wave_rank_up, _ = get_disease_rank(wave_cands, tc["expect_up"])
        if curr_rank_up and wave_rank_up:
            print(f"\n  ↑{tc['expect_up']}: rank base→現行={curr_rank_up}, base→波動={wave_rank_up}")

    if tc["expect_down"]:
        b_down = base_map.get(tc["expect_down"], 0)
        c_down = curr_map.get(tc["expect_down"], 0)
        w_down = wave_map.get(tc["expect_down"], 0)
        c_effect = (b_down - c_down) / b_down * 100 if b_down > 0 else 0
        w_effect = (b_down - w_down) / b_down * 100 if b_down > 0 else 0
        print(f"  ↓{tc['expect_down']}: 現行 -{c_effect:.1f}%, 波動 -{w_effect:.1f}%")
        if w_effect > c_effect:
            result["wave_win"] = True
        elif c_effect > w_effect:
            result["current_win"] = True

    results.append(result)


# ============================================================
# 総合結果
# ============================================================
print(f"\n{'=' * 80}")
print("総合結果")
print(f"{'=' * 80}")

wave_wins = sum(1 for r in results if r["wave_win"])
curr_wins = sum(1 for r in results if r["current_win"])
ties = len(results) - wave_wins - curr_wins

print(f"波動方式が優位: {wave_wins}件")
print(f"現行方式が優位: {curr_wins}件")
print(f"同等/判定不能: {ties}件")
print(f"\n合計: {len(results)}ケース")
