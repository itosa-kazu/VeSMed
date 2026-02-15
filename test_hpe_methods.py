"""
HPE更新方式の比較テスト（VeSMed核心機能）

3つの方式を比較:
  A: 現行（連続乗算）  sim *= Π exp(p_k * e_k)
  B: 平均化 (delta/N)  sim *= exp(Σ p_k*e_k / N)
  C: sqrt平均 (delta/√N) sim *= exp(Σ p_k*e_k / √N)

エンジン初期化不要 — キャッシュデータを直接読み込み。
"""
import json
import os
import copy
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ================================================================
# データ読み込み
# ================================================================

def load_data():
    hpe_cache = os.path.join(DATA_DIR, "sim_matrix_hpe.npz")
    data = np.load(hpe_cache, allow_pickle=True)
    sim_matrix_hpe = data["sim_matrix"]
    disease_names = list(data["disease_names"])
    hpe_names = list(data["hpe_names"])
    disease_idx = {name: i for i, name in enumerate(disease_names) if name}
    hpe_idx = {name: i for i, name in enumerate(hpe_names)}

    test_cache = os.path.join(DATA_DIR, "sim_matrix.npz")
    tdata = np.load(test_cache, allow_pickle=True)
    sim_matrix = tdata["sim_matrix"]
    test_names = list(tdata["test_names"])
    test_idx = {name: i for i, name in enumerate(test_names)}

    disease_2c = {}
    with open(os.path.join(DATA_DIR, "diseases.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                name = d["disease_name"]
                crit = d.get("critical_score", 0.5)
                cure = d.get("curable_score", 0.5)
                disease_2c[name] = {
                    "critical": crit,
                    "curable": cure,
                    "weight": float(np.exp(crit + cure)),
                }
            except (json.JSONDecodeError, KeyError):
                continue

    return {
        "sim_matrix_hpe": sim_matrix_hpe,
        "sim_matrix": sim_matrix,
        "disease_idx": disease_idx,
        "disease_names": disease_names,
        "hpe_names": hpe_names,
        "hpe_idx": hpe_idx,
        "test_names": test_names,
        "test_idx": test_idx,
        "disease_2c": disease_2c,
    }


# ================================================================
# HPE更新の3方式
# ================================================================

def update_multiplicative(candidates, hpe_findings, sim_matrix_hpe, disease_idx):
    """方式A: 現行（連続乗算）"""
    for f in hpe_findings:
        idx = f["index"]
        polarity = f["polarity"]
        sims = sim_matrix_hpe[:, idx]
        bg = float(sims.mean())
        for c in candidates:
            d_idx = disease_idx.get(c["disease_name"])
            if d_idx is not None:
                excess = max(0.0, float(sims[d_idx]) - bg)
                if excess > 0:
                    c["similarity"] *= float(np.exp(polarity * excess))
    candidates.sort(key=lambda c: c["similarity"], reverse=True)
    return candidates


def update_mean(candidates, hpe_findings, sim_matrix_hpe, disease_idx):
    """方式B: delta/N"""
    N = max(1, len(hpe_findings))
    for c in candidates:
        delta = 0.0
        d_idx = disease_idx.get(c["disease_name"])
        if d_idx is None:
            continue
        for f in hpe_findings:
            sims = sim_matrix_hpe[:, f["index"]]
            bg = float(sims.mean())
            excess = max(0.0, float(sims[d_idx]) - bg)
            delta += f["polarity"] * excess
        c["similarity"] *= float(np.exp(delta / N))
    candidates.sort(key=lambda c: c["similarity"], reverse=True)
    return candidates


def update_sqrt(candidates, hpe_findings, sim_matrix_hpe, disease_idx):
    """方式C: delta/√N"""
    N = max(1, len(hpe_findings))
    sqrt_N = np.sqrt(N)
    for c in candidates:
        delta = 0.0
        d_idx = disease_idx.get(c["disease_name"])
        if d_idx is None:
            continue
        for f in hpe_findings:
            sims = sim_matrix_hpe[:, f["index"]]
            bg = float(sims.mean())
            excess = max(0.0, float(sims[d_idx]) - bg)
            delta += f["polarity"] * excess
        c["similarity"] *= float(np.exp(delta / sqrt_N))
    candidates.sort(key=lambda c: c["similarity"], reverse=True)
    return candidates


# ================================================================
# 重み・分散計算（Part Aの数学）
# ================================================================

def compute_weights_and_variance(candidates, sim_matrix, disease_idx, test_names, disease_2c):
    raw_sims = np.array([c["similarity"] for c in candidates], dtype=float)
    weights = np.array([
        disease_2c.get(c["disease_name"], {}).get("weight", 1.0)
        for c in candidates
    ], dtype=float)
    sim_centered = np.maximum(0.0, raw_sims - raw_sims.mean())
    w = sim_centered * weights
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum

    n_tests = len(test_names)
    disease_rows = np.array([
        disease_idx.get(c["disease_name"], -1) for c in candidates
    ])
    valid_mask = disease_rows >= 0
    sim_sub = np.zeros((len(candidates), n_tests))
    sim_sub[valid_mask] = sim_matrix[disease_rows[valid_mask]]

    w_col = w[:, np.newaxis]
    mu = (w_col * sim_sub).sum(axis=0)
    var = (w_col * (sim_sub - mu) ** 2).sum(axis=0)
    cluster_mu = (w_col * sim_sub).sum(axis=0)

    return w, var, cluster_mu


def weight_stats(w):
    w_sorted = np.sort(w)[::-1]
    top1 = w_sorted[0] if len(w_sorted) > 0 else 0
    top3 = w_sorted[:3].sum() if len(w_sorted) >= 3 else w_sorted.sum()
    n_eff = 1.0 / (w_sorted ** 2).sum() if (w_sorted ** 2).sum() > 0 else 0
    return {"top1": float(top1), "top3": float(top3), "n_eff": float(n_eff)}


# ================================================================
# テストケース定義（正しいHPE項目名を使用）
# ================================================================

def make_finding(hpe_idx, item_name, polarity):
    if item_name not in hpe_idx:
        return None
    return {"item": item_name, "index": hpe_idx[item_name], "polarity": polarity}


def define_test_cases(hpe_idx):
    cases = []

    # ━━━ Case 1: ショック症例（実臨床再現 — 27所見） ━━━
    pos1 = [
        "発熱", "嘔気", "嘔吐", "下痢", "ショック所見", "脱水所見",
        "CRT延長", "意識変容", "急性発症（分〜時間）", "側腹部痛",
        "背部放散", "倦怠感",
    ]
    neg1 = [
        "発疹", "関節痛", "血尿", "めまい（回転性）", "便秘", "失神",
        "消化性潰瘍", "関節リウマチ", "肝疾患", "甲状腺疾患",
        "COPD", "渡航歴（熱帯/途上国）", "単関節痛", "紫斑", "痙攣",
    ]
    f1 = ([make_finding(hpe_idx, n, +1) for n in pos1] +
          [make_finding(hpe_idx, n, -1) for n in neg1])
    f1 = [f for f in f1 if f is not None]
    cases.append({
        "name": "Case 1: ショック症例（実臨床）",
        "desc": "47歳女性、敗血症疑い。多数の陽性+陰性所見",
        "findings": f1,
        "expected_top": ["敗血症性ショック", "敗血症"],
        "expected_not_top": ["偶発性低体温症"],
        "key_tests": ["血液培養", "乳酸", "プロカルシトニン（PCT）"],
        "candidates": [
            ("敗血症性ショック", 0.58), ("敗血症", 0.56),
            ("壊死性筋膜炎", 0.54), ("トキシックショック症候群", 0.53),
            ("インフルエンザ", 0.52), ("ノロウイルス腸炎", 0.51),
            ("急性膵炎", 0.49), ("急性胆嚢炎", 0.48),
            ("サルモネラ感染症", 0.47), ("尿路結石", 0.45),
            ("急性虫垂炎", 0.44), ("食中毒", 0.43),
            ("腸間膜虚血", 0.42), ("偶発性低体温症", 0.41),
            ("肺血栓塞栓症", 0.40),
        ],
    })

    # ━━━ Case 2: bvFTD（5所見 + 既往歴ノイズ） ━━━
    pos2 = ["意識変容", "高血圧"]  # 意識変容は誤抽出、高血圧は既往歴
    neg2 = ["頭痛", "痙攣", "発熱"]
    f2 = ([make_finding(hpe_idx, n, +1) for n in pos2] +
          [make_finding(hpe_idx, n, -1) for n in neg2])
    f2 = [f for f in f2 if f is not None]
    cases.append({
        "name": "Case 2: bvFTD（既往歴ノイズ）",
        "desc": "65歳男性、行動変容。高血圧+意識変容誤抽出",
        "findings": f2,
        "expected_top": ["前頭側頭型認知症"],
        "expected_not_top": ["褐色細胞腫"],
        "key_tests": ["頭部MRI"],
        "candidates": [
            ("前頭側頭型認知症", 0.55), ("うつ病", 0.52),
            ("双極性障害", 0.50), ("統合失調症", 0.49),
            ("レビー小体型認知症", 0.48), ("甲状腺機能亢進症", 0.46),
            ("褐色細胞腫", 0.45), ("クッシング症候群", 0.43),
            ("正常圧水頭症", 0.42), ("ビタミンB12欠乏症", 0.41),
        ],
    })

    # ━━━ Case 3: 軽症（発熱+咳嗽, 2所見） ━━━
    f3 = [make_finding(hpe_idx, "発熱", +1), make_finding(hpe_idx, "咳嗽", +1)]
    f3 = [f for f in f3 if f]
    cases.append({
        "name": "Case 3: 軽症（発熱+咳嗽）",
        "desc": "30歳男性。2所見のみ（最小限の信号）",
        "findings": f3,
        "expected_top": ["肺炎", "インフルエンザ"],
        "expected_not_top": [],
        "key_tests": ["胸部X線"],
        "candidates": [
            ("肺炎", 0.60), ("インフルエンザ", 0.58),
            ("急性気管支炎", 0.56), ("COVID-19", 0.54),
            ("結核", 0.52), ("百日咳", 0.50),
            ("肺血栓塞栓症", 0.48), ("心不全", 0.46),
            ("間質性肺炎", 0.44), ("肺癌", 0.42),
        ],
    })

    # ━━━ Case 4: 大量陰性（陽性なし, 13所見） ━━━
    neg4 = [
        "発熱", "咳嗽", "呼吸困難", "胸骨後部痛", "腹部膨満",
        "下痢", "嘔気", "頭痛", "めまい（回転性）", "発疹",
        "関節痛", "下肢浮腫", "体重減少",
    ]
    f4 = [make_finding(hpe_idx, n, -1) for n in neg4]
    f4 = [f for f in f4 if f]
    cases.append({
        "name": "Case 4: 陰性所見のみ（13件）",
        "desc": "多数の否定的所見、陽性なし（エッジケース）",
        "findings": f4,
        "expected_top": [],
        "expected_not_top": [],
        "key_tests": [],
        "candidates": [
            ("不安障害", 0.48), ("うつ病", 0.47),
            ("身体症状症", 0.46), ("線維筋痛症", 0.45),
            ("慢性疲労症候群", 0.44), ("甲状腺機能低下症", 0.43),
            ("糖尿病", 0.42), ("鉄欠乏性貧血", 0.41),
            ("ビタミンB12欠乏症", 0.40), ("睡眠時無呼吸症候群", 0.39),
        ],
    })

    # ━━━ Case 5: 急性腹症（12所見） ━━━
    pos5 = [
        "急性発症（分〜時間）", "嘔気", "嘔吐", "発熱",
        "筋性防御", "Blumberg徴候（反跳痛）", "心窩部圧痛",
        "ショック所見", "脱水所見",
    ]
    neg5 = ["下痢", "便秘", "血便"]
    f5 = ([make_finding(hpe_idx, n, +1) for n in pos5] +
          [make_finding(hpe_idx, n, -1) for n in neg5])
    f5 = [f for f in f5 if f]
    cases.append({
        "name": "Case 5: 急性腹症（12所見）",
        "desc": "50歳男性。腹膜刺激徴候+ショック傾向",
        "findings": f5,
        "expected_top": ["急性腹膜炎", "消化管穿孔"],
        "expected_not_top": [],
        "key_tests": ["腹部CT"],
        "candidates": [
            ("急性腹膜炎", 0.58), ("急性虫垂炎", 0.56),
            ("消化管穿孔", 0.55), ("急性膵炎", 0.54),
            ("急性胆嚢炎", 0.52), ("腸閉塞", 0.50),
            ("腸間膜虚血", 0.49), ("卵巣茎捻転", 0.47),
            ("大動脈瘤破裂", 0.45), ("尿管結石", 0.43),
        ],
    })

    # ━━━ Case 6: 頭痛のみ（1所見、N=1境界） ━━━
    f6 = [make_finding(hpe_idx, "頭痛", +1)]
    f6 = [f for f in f6 if f]
    cases.append({
        "name": "Case 6: 頭痛のみ（N=1）",
        "desc": "40歳女性。頭痛のみ。3方式の結果が一致するはず",
        "findings": f6,
        "expected_top": [],
        "expected_not_top": [],
        "key_tests": [],
        "candidates": [
            ("くも膜下出血", 0.55), ("片頭痛", 0.53),
            ("緊張型頭痛", 0.52), ("髄膜炎", 0.51),
            ("脳腫瘍", 0.49), ("側頭動脈炎", 0.47),
            ("高血圧緊急症", 0.45), ("副鼻腔炎", 0.44),
            ("緑内障", 0.42), ("一酸化炭素中毒", 0.40),
        ],
    })

    # ━━━ Case 7: 重症肺炎（大量30所見） ━━━
    pos7 = [
        "発熱", "咳嗽", "喀痰", "呼吸困難", "急性発症（分〜時間）",
        "倦怠感", "食欲低下", "ショック所見", "脱水所見",
        "CRT延長", "末梢チアノーゼ", "Crackles", "頭痛",
        "嘔気", "関節痛", "筋力低下",
    ]
    neg7 = [
        "下痢", "便秘", "血尿", "発疹", "下肢浮腫",
        "黄疸自覚", "体重減少", "紫斑", "痙攣", "言語障害",
        "片麻痺", "血便", "排尿困難/排尿時痛", "掻痒",
    ]
    f7 = ([make_finding(hpe_idx, n, +1) for n in pos7] +
          [make_finding(hpe_idx, n, -1) for n in neg7])
    f7 = [f for f in f7 if f]
    cases.append({
        "name": "Case 7: 重症肺炎（大量所見）",
        "desc": "55歳男性。重症肺炎疑い。陽性16+陰性14=30所見",
        "findings": f7,
        "expected_top": ["肺炎"],
        "expected_not_top": [],
        "key_tests": ["胸部CT", "血液培養", "喀痰培養"],
        "candidates": [
            ("肺炎", 0.62), ("COVID-19", 0.60),
            ("インフルエンザ", 0.58), ("ARDS", 0.55),
            ("肺血栓塞栓症", 0.52), ("結核", 0.50),
            ("間質性肺炎", 0.48), ("心不全", 0.46),
            ("敗血症", 0.44), ("気管支喘息", 0.42),
            ("肺癌", 0.40), ("サルコイドーシス", 0.38),
        ],
    })

    # ━━━ Case 8: ACS + 生活習慣病既往（7所見） ━━━
    pos8 = ["胸骨後部痛", "左肩・左腕放散", "持続性（>20分）",
            "高血圧", "糖尿病"]  # 高血圧・糖尿病は既往歴ノイズ
    neg8 = ["発熱", "咳嗽"]
    f8 = ([make_finding(hpe_idx, n, +1) for n in pos8] +
          [make_finding(hpe_idx, n, -1) for n in neg8])
    f8 = [f for f in f8 if f]
    cases.append({
        "name": "Case 8: ACS + 生活習慣病既往",
        "desc": "60歳男性。胸痛。高血圧+糖尿病あり（既往歴ノイズ）",
        "findings": f8,
        "expected_top": ["急性心筋梗塞"],
        "expected_not_top": ["褐色細胞腫"],
        "key_tests": ["心電図", "トロポニン"],
        "candidates": [
            ("急性心筋梗塞", 0.60), ("不安定狭心症", 0.58),
            ("急性冠症候群", 0.57), ("大動脈解離", 0.54),
            ("肺血栓塞栓症", 0.52), ("気胸", 0.49),
            ("心不全", 0.47), ("褐色細胞腫", 0.44),
            ("高血圧緊急症", 0.42), ("パニック障害", 0.40),
        ],
    })

    # ━━━ Case 9: ストレステスト（極端に多い所見 40件） ━━━
    pos9 = [
        "発熱", "咳嗽", "喀痰", "呼吸困難", "喀血",
        "胸骨後部痛", "動悸", "嘔気", "嘔吐", "下痢",
        "関節痛", "筋力低下", "頭痛", "倦怠感", "食欲低下",
        "ショック所見", "脱水所見", "CRT延長", "末梢チアノーゼ",
        "Crackles", "急性発症（分〜時間）", "意識変容",
    ]
    neg9 = [
        "便秘", "血尿", "発疹", "下肢浮腫", "黄疸自覚",
        "体重減少", "紫斑", "痙攣", "言語障害", "片麻痺",
        "血便", "排尿困難/排尿時痛", "掻痒", "月経異常",
        "朝のこわばり", "しびれ", "不眠", "嗄声",
    ]
    f9 = ([make_finding(hpe_idx, n, +1) for n in pos9] +
          [make_finding(hpe_idx, n, -1) for n in neg9])
    f9 = [f for f in f9 if f]
    cases.append({
        "name": "Case 9: ストレステスト（40所見）",
        "desc": "極端ケース。22陽性+18陰性=40所見で分散崩壊を誘発",
        "findings": f9,
        "expected_top": [],
        "expected_not_top": [],
        "key_tests": [],
        "candidates": [
            ("敗血症性ショック", 0.60), ("敗血症", 0.58),
            ("肺炎", 0.56), ("COVID-19", 0.54),
            ("ARDS", 0.52), ("インフルエンザ", 0.50),
            ("壊死性筋膜炎", 0.48), ("結核", 0.46),
            ("心不全", 0.44), ("肺血栓塞栓症", 0.42),
            ("間質性肺炎", 0.40), ("気管支喘息", 0.38),
            ("大動脈解離", 0.36), ("肺癌", 0.34),
            ("サルコイドーシス", 0.32),
        ],
    })

    # ━━━ Case 10: 対称テスト（陽性=陰性が拮抗） ━━━
    pos10 = ["発熱", "咳嗽", "喀痰", "倦怠感", "食欲低下"]
    neg10 = ["呼吸困難", "胸骨後部痛", "下痢", "嘔気", "関節痛"]
    f10 = ([make_finding(hpe_idx, n, +1) for n in pos10] +
           [make_finding(hpe_idx, n, -1) for n in neg10])
    f10 = [f for f in f10 if f]
    cases.append({
        "name": "Case 10: 対称テスト（陽性5=陰性5）",
        "desc": "陽性と陰性が拮抗。方式Aでは打ち消し合いが不均一に",
        "findings": f10,
        "expected_top": [],
        "expected_not_top": [],
        "key_tests": [],
        "candidates": [
            ("肺炎", 0.55), ("急性気管支炎", 0.53),
            ("インフルエンザ", 0.51), ("結核", 0.49),
            ("COVID-19", 0.47), ("百日咳", 0.45),
            ("間質性肺炎", 0.43), ("肺癌", 0.41),
            ("心不全", 0.39), ("気管支喘息", 0.37),
        ],
    })

    return cases


# ================================================================
# メイン比較テスト
# ================================================================

def run_comparison(data, cases):
    methods = [
        ("A:連続乗算", update_multiplicative),
        ("B:delta/N", update_mean),
        ("C:delta/√N", update_sqrt),
    ]

    print("=" * 80)
    print("  HPE更新方式 比較テスト")
    print("  A: 現行（連続乗算）  B: delta/N  C: delta/√N")
    print("=" * 80)

    all_results = []

    for case in cases:
        n_pos = sum(1 for f in case["findings"] if f["polarity"] > 0)
        n_neg = sum(1 for f in case["findings"] if f["polarity"] < 0)
        N = len(case["findings"])
        print(f"\n{'━'*80}")
        print(f"  {case['name']}")
        print(f"  {case['desc']}")
        print(f"  所見数: {N} (陽性{n_pos}, 陰性{n_neg}), √N={np.sqrt(N):.2f}")
        print(f"{'━'*80}")

        case_results = {}

        for mname, mfunc in methods:
            cands = [{"disease_name": n, "similarity": s} for n, s in case["candidates"]]
            cands = copy.deepcopy(cands)
            updated = mfunc(cands, case["findings"], data["sim_matrix_hpe"], data["disease_idx"])

            w, var, cmu = compute_weights_and_variance(
                updated, data["sim_matrix"], data["disease_idx"],
                data["test_names"], data["disease_2c"]
            )
            ws = weight_stats(w)

            sims_after = np.array([c["similarity"] for c in updated])
            sims_before = np.array([s for _, s in case["candidates"]])
            max_r = float((sims_after / sims_before).max())
            min_r = float((sims_after / sims_before).min())

            var_mean = float(var.mean())
            var_max = float(var.max())
            var_zero_n = int((var < 1e-8).sum())
            var_zero_pct = var_zero_n / len(var) * 100

            case_results[mname] = {
                "updated": updated, "w": w, "var": var, "cmu": cmu,
                "ws": ws, "max_r": max_r, "min_r": min_r,
                "var_mean": var_mean, "var_max": var_max,
                "var_zero_pct": var_zero_pct,
            }

            print(f"\n  [{mname}]")
            print(f"    sim変動: ×{min_r:.4f} ～ ×{max_r:.4f}")
            print(f"    重み: top1={ws['top1']:.3f} top3={ws['top3']:.3f} 有効N={ws['n_eff']:.1f}")
            print(f"    分散: mean={var_mean:.6f} max={var_max:.6f} zero={var_zero_pct:.1f}%")

            # 疾患ランキング
            for i, c in enumerate(updated[:5]):
                m = ""
                if any(e in c["disease_name"] for e in case.get("expected_top", [])):
                    m = " ★"
                if any(e in c["disease_name"] for e in case.get("expected_not_top", [])):
                    m = " ✗"
                print(f"      #{i+1}: {c['disease_name']:30s} sim={c['similarity']:.4f}{m}")

            # 重要検査
            if case["key_tests"]:
                var_rank = np.argsort(var)[::-1]
                for tkw in case["key_tests"]:
                    for rpos, j in enumerate(var_rank):
                        if tkw in data["test_names"][j]:
                            print(f"    → {tkw}: Part A #{rpos+1}")
                            break

        # 方式間比較
        print(f"\n  --- 比較 ---")
        for mname in ["A:連続乗算", "B:delta/N", "C:delta/√N"]:
            r = case_results[mname]
            st = "★崩壊" if r["var_zero_pct"] > 95 else \
                 "△微小" if r["var_mean"] < 1e-5 else "○正常"
            print(f"    {mname}: {st} var_mean={r['var_mean']:.6f} "
                  f"sim×{r['min_r']:.3f}~×{r['max_r']:.3f} "
                  f"top1_w={r['ws']['top1']:.3f}")

        if case["expected_top"]:
            for exp in case["expected_top"]:
                pos = {}
                for mname in ["A:連続乗算", "B:delta/N", "C:delta/√N"]:
                    for i, c in enumerate(case_results[mname]["updated"]):
                        if exp in c["disease_name"]:
                            pos[mname] = i + 1
                            break
                    else:
                        pos[mname] = "N/A"
                print(f"    {exp}: A=#{pos['A:連続乗算']} B=#{pos['B:delta/N']} C=#{pos['C:delta/√N']}")

        if case["expected_not_top"]:
            for exp in case["expected_not_top"]:
                pos = {}
                for mname in ["A:連続乗算", "B:delta/N", "C:delta/√N"]:
                    for i, c in enumerate(case_results[mname]["updated"]):
                        if exp in c["disease_name"]:
                            pos[mname] = i + 1
                            break
                    else:
                        pos[mname] = "N/A"
                print(f"    ✗{exp}: A=#{pos['A:連続乗算']} B=#{pos['B:delta/N']} C=#{pos['C:delta/√N']}")

        all_results.append({
            "name": case["name"],
            "N": N,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "results": {k: {
                "var_mean": v["var_mean"], "var_zero_pct": v["var_zero_pct"],
                "max_r": v["max_r"], "min_r": v["min_r"],
                "top1_w": v["ws"]["top1"], "n_eff": v["ws"]["n_eff"],
            } for k, v in case_results.items()},
        })

    # ================================================================
    # 総合サマリ
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  総合サマリ（10ケース × 3方式）")
    print(f"{'='*80}")

    hdr = f"  {'ケース':38s} {'N':>3s} {'方式':12s} {'var_mean':>10s} {'zero%':>6s} {'sim変動':>14s} {'top1_w':>7s} {'有効N':>6s}"
    print(hdr)
    print(f"  {'─'*len(hdr)}")
    for s in all_results:
        for mname in ["A:連続乗算", "B:delta/N", "C:delta/√N"]:
            r = s["results"][mname]
            flag = "★" if r["var_zero_pct"] > 95 else "△" if r["var_mean"] < 1e-5 else " "
            print(f"  {s['name']:38s} {s['N']:3d} {mname:12s} "
                  f"{r['var_mean']:10.6f} {r['var_zero_pct']:5.1f}% "
                  f"×{r['min_r']:.3f}~×{r['max_r']:.3f} "
                  f"{r['top1_w']:7.3f} {r['n_eff']:6.1f}{flag}")

    print(f"\n  --- 方式別集計 ---")
    for mname in ["A:連続乗算", "B:delta/N", "C:delta/√N"]:
        collapses = sum(1 for s in all_results if s["results"][mname]["var_zero_pct"] > 95)
        weak = sum(1 for s in all_results if s["results"][mname]["var_mean"] < 1e-5)
        strong = sum(1 for s in all_results if s["results"][mname]["max_r"] > 1.5)
        avg_var = np.mean([s["results"][mname]["var_mean"] for s in all_results])
        avg_top1 = np.mean([s["results"][mname]["top1_w"] for s in all_results])
        avg_neff = np.mean([s["results"][mname]["n_eff"] for s in all_results])
        print(f"    {mname}: 崩壊={collapses} 微小={weak} "
              f"平均var={avg_var:.6f} 平均top1_w={avg_top1:.3f} 平均有効N={avg_neff:.1f}")

    # N別の分析
    print(f"\n  --- 所見数Nと変動倍率の関係 ---")
    print(f"  {'N':>3s}  {'A max変動':>10s}  {'B max変動':>10s}  {'C max変動':>10s}  {'A/B比':>7s}  {'A/C比':>7s}")
    for s in sorted(all_results, key=lambda x: x["N"]):
        a = s["results"]["A:連続乗算"]
        b = s["results"]["B:delta/N"]
        c = s["results"]["C:delta/√N"]
        print(f"  {s['N']:3d}  ×{a['max_r']:9.4f}  ×{b['max_r']:9.4f}  ×{c['max_r']:9.4f}  "
              f"{a['max_r']/b['max_r']:7.2f}  {a['max_r']/c['max_r']:7.2f}")


def main():
    print("データ読み込み中...")
    data = load_data()
    print(f"  疾患: {len(data['disease_idx'])}件, HPE: {len(data['hpe_names'])}件, "
          f"検査: {len(data['test_names'])}件")

    cases = define_test_cases(data["hpe_idx"])
    for case in cases:
        n = len(case["findings"])
        np_ = sum(1 for f in case["findings"] if f["polarity"] > 0)
        nn_ = sum(1 for f in case["findings"] if f["polarity"] < 0)
        print(f"  {case['name']}: {n}所見 (pos={np_}, neg={nn_})")

    run_comparison(data, cases)


if __name__ == "__main__":
    main()
