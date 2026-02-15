"""
VeSMed パイプライン全面精査
各ステップのLLM出力を個別検証し、ハルシネーション安全弁を監査する。

テストケース:
  Case A: 47歳女性、頭痛+嘔気+バイタル+陰性所見（前回ユーザーが使用）
  Case B: 敗血症疑い（高熱、悪寒、頻脈）→ 血液培養が推薦されるべき
  Case C: 軽症（微熱のみ）→ 骨髄穿刺等の侵襲的検査が抑制されるべき
"""
import json
import sys
import time
import numpy as np

# ============================================================
# テストケース定義
# ============================================================
CASES = {
    "A": {
        "name": "47歳女性 頭痛+嘔気+バイタル",
        "text": (
            "47歳女性。3日前からの持続する頭痛と嘔気を主訴に来院。\n"
            "体温37.5℃、血圧120/80mmHg、脈拍88/分、SpO2 98%。\n"
            "項部硬直なし。Kernig徴候陰性。Murphy徴候陰性。\n"
            "CRP 2.5mg/dL、WBC 9800/μL。"
        ),
        "expect_positive": ["頭痛", "嘔気", "体温37.5"],
        "expect_negative": ["項部硬直なし", "Kernig", "Murphy"],
        "expect_results": ["CRP", "WBC"],
        "expect_diseases_high": ["片頭痛", "緊張型頭痛"],
        "expect_diseases_low": ["偶発性低体温症"],  # 37.5°Cで低体温症はありえない
        "expect_excluded": [],
        "expect_tests_recommended": ["頭部CT", "頭部MRI"],
        "expect_tests_suppressed": ["CRP"],
    },
    "B": {
        "name": "敗血症疑い: 高熱+悪寒+頻脈",
        "text": (
            "62歳男性。3日前からの39.2℃の発熱、悪寒戦慄、全身倦怠感で救急搬送。\n"
            "体温39.2℃、血圧90/55mmHg、脈拍118/分、呼吸数24/分、SpO2 94%。\n"
            "意識やや混濁。末梢冷感あり。\n"
            "WBC 18500/μL、CRP 22.3mg/dL、プロカルシトニン 8.5ng/mL。"
        ),
        "expect_positive": ["発熱", "悪寒", "全身倦怠感", "低血圧", "頻脈"],
        "expect_negative": [],
        "expect_results": ["WBC", "CRP", "プロカルシトニン"],
        "expect_diseases_high": ["敗血症", "敗血症性ショック"],
        "expect_diseases_low": ["偶発性低体温症"],
        "expect_excluded": [],
        "expect_tests_recommended": ["血液培養"],  # ★ 核心: 血液培養が推薦されるべき
        "expect_tests_suppressed": ["CRP"],
    },
    "C": {
        "name": "軽症: 微熱のみ",
        "text": (
            "28歳女性。昨日からの37.2℃の微熱を主訴に来院。\n"
            "他に症状なし。バイタルサイン正常。\n"
            "体温37.2℃、血圧115/72mmHg、脈拍76/分、SpO2 99%。"
        ),
        "expect_positive": ["微熱"],
        "expect_negative": [],
        "expect_results": [],
        "expect_diseases_high": [],
        "expect_diseases_low": [],
        "expect_excluded": [],
        "expect_tests_recommended": [],
        "expect_tests_suppressed": [],
        "expect_invasive_demoted": ["骨髄穿刺", "腰椎穿刺（髄液検査）"],
    },
}


def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def subheader(title):
    print(f"\n--- {title} ---")


def check(label, condition, detail=""):
    mark = "✓" if condition else "✗"
    msg = f"  [{mark}] {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def main():
    header("VeSMed パイプライン精査開始")

    # エンジン初期化
    print("\nエンジン初期化中...")
    t0 = time.time()
    from engine import VeSMedEngine
    eng = VeSMedEngine()
    print(f"初期化完了: {time.time()-t0:.1f}s")
    print(f"  検査数: {len(eng.test_names)}")
    print(f"  疾患数: {len(eng.disease_db)}")
    print(f"  HPE数: {len(eng.hpe_names)}")
    print(f"  sim_matrix: {eng.sim_matrix.shape if eng.sim_matrix is not None else 'None'}")

    # LLMモデル確認
    from config import VERTEX_MODEL
    print(f"  LLM: {VERTEX_MODEL}")

    total_pass = 0
    total_fail = 0

    for case_id, case in CASES.items():
        header(f"Case {case_id}: {case['name']}")
        print(f"入力テキスト:\n  {case['text'][:100]}...")

        passes = 0
        fails = 0

        # ============================================================
        # Step 0: テキスト分離（split_symptoms_results）
        # ============================================================
        subheader("Step 0: テキスト分離 (LLM)")
        t1 = time.time()
        positive_text, negative_findings, result_lines = eng.split_symptoms_results(case["text"])
        t2 = time.time()
        print(f"  所要時間: {t2-t1:.1f}s")
        print(f"  陽性テキスト: {positive_text[:120]}...")
        print(f"  陰性所見: {negative_findings}")
        print(f"  検査結果: {result_lines}")

        # 陰性所見が陽性テキストに混入していないか
        for neg_kw in case.get("expect_negative", []):
            in_neg = any(neg_kw in nf for nf in negative_findings)
            in_pos = neg_kw in positive_text and "なし" in positive_text[positive_text.index(neg_kw):positive_text.index(neg_kw)+20]
            ok = in_neg
            if ok:
                passes += 1
            else:
                fails += 1
            check(f"'{neg_kw}' → negative_findings", ok,
                  f"negative: {in_neg}, positive漏れ: {in_pos}")

        # 検査結果が正しく抽出されたか
        for res_kw in case.get("expect_results", []):
            in_res = any(res_kw in r for r in result_lines)
            if in_res:
                passes += 1
            else:
                fails += 1
            check(f"'{res_kw}' → results", in_res)

        # ============================================================
        # Step 1: 疾患検索（embedding — 知覚）
        # ============================================================
        subheader("Step 1: 疾患検索 (embedding)")
        if not positive_text.strip():
            positive_text = case["text"]

        t3 = time.time()
        candidates = eng.search_diseases(positive_text)
        candidates = eng.compute_priors(candidates)
        t4 = time.time()
        print(f"  所要時間: {t4-t3:.1f}s")
        print(f"  候補数: {len(candidates)}")
        print(f"  Top10:")
        for i, c in enumerate(candidates[:10]):
            print(f"    #{i+1}: {c['disease_name']} sim={c['similarity']:.4f} "
                  f"w={c.get('clinical_weight',0):.2f}")

        # 期待疾患が上位にあるか
        for dname in case.get("expect_diseases_high", []):
            found = False
            for i, c in enumerate(candidates[:30]):
                if dname in c["disease_name"]:
                    found = True
                    ok = i < 20
                    if ok:
                        passes += 1
                    else:
                        fails += 1
                    check(f"'{dname}' が上位20位以内", ok, f"#{i+1}")
                    break
            if not found:
                fails += 1
                check(f"'{dname}' が候補に存在", False, "not found in top30")

        # 不適切な疾患が上位にないか
        for dname in case.get("expect_diseases_low", []):
            for i, c in enumerate(candidates[:10]):
                if dname in c["disease_name"]:
                    fails += 1
                    check(f"'{dname}' がTop10にない（不適切）", False,
                          f"#{i+1} sim={c['similarity']:.4f}")
                    break
            else:
                passes += 1
                check(f"'{dname}' がTop10にない", True)

        # ============================================================
        # Step 2: LLMフィルタ（矛盾除外）
        # ============================================================
        subheader("Step 2: LLMフィルタ (矛盾除外)")
        import copy
        cands_for_filter = copy.deepcopy(candidates)

        t5 = time.time()
        filtered, exclusion_reasons = eng.filter_contradictions(
            cands_for_filter[:20], case["text"], negative_findings
        )
        t6 = time.time()
        print(f"  所要時間: {t6-t5:.1f}s")
        print(f"  除外: {len(exclusion_reasons)}件")
        for r in exclusion_reasons:
            print(f"    - {r['disease_name']}: {r.get('reason', '')}")
        print(f"  残存: {len(filtered)}件")

        # フィルタ後Top10
        print(f"  フィルタ後Top10:")
        for i, c in enumerate(filtered[:10]):
            print(f"    #{i+1}: {c['disease_name']} sim={c['similarity']:.4f}")

        # 除外理由の質チェック
        for r in exclusion_reasons:
            has_reason = bool(r.get("reason", "").strip())
            if has_reason:
                passes += 1
            else:
                fails += 1
            check(f"除外理由あり: {r['disease_name']}", has_reason,
                  r.get("reason", "なし")[:50])

        # ============================================================
        # Step 3: 検査結果による更新 (update_from_results)
        # ============================================================
        if result_lines:
            subheader("Step 3: 検査結果更新 (polarity + 反実仮想)")
            cands_copy = copy.deepcopy(filtered)
            t7 = time.time()
            updated = eng.update_from_results(cands_copy, result_lines, positive_text)
            t8 = time.time()
            print(f"  所要時間: {t8-t7:.1f}s")
            print(f"  更新後Top10:")
            for i, c in enumerate(updated[:10]):
                print(f"    #{i+1}: {c['disease_name']} sim={c['similarity']:.4f}")

            # 更新前後で候補が反映されているか
            filtered = updated  # 以降のステップは更新後を使用

        # ============================================================
        # Step 4: 統合Novelty (LLM)
        # ============================================================
        subheader("Step 4: 統合Novelty (LLM)")
        t9 = time.time()
        novelty, novelty_hpe, hpe_findings = eng.compute_all_novelty(case["text"])
        t10 = time.time()
        print(f"  所要時間: {t10-t9:.1f}s")

        n_test_suppressed = int((novelty == 0.0).sum())
        n_hpe_suppressed = int((novelty_hpe == 0.0).sum())
        print(f"  検査抑制: {n_test_suppressed}件")
        print(f"  HPE抑制: {n_hpe_suppressed}件")

        # 抑制された検査の一覧
        suppressed_tests = [eng.test_names[j] for j in range(len(eng.test_names)) if novelty[j] == 0.0]
        if suppressed_tests:
            print(f"  抑制検査一覧: {suppressed_tests}")

        # 期待される抑制の検証
        for tname in case.get("expect_tests_suppressed", []):
            found = False
            for st in suppressed_tests:
                if tname in st:
                    found = True
                    break
            if found:
                passes += 1
            else:
                fails += 1
            check(f"'{tname}' が抑制されている", found)

        # HPE所見
        if hpe_findings:
            print(f"  HPE所見:")
            for f in hpe_findings[:10]:
                print(f"    {f['item']} (polarity={f['polarity']:+d})")

        # ============================================================
        # Step 5: ランキング (Part A/B/C)
        # ============================================================
        subheader("Step 5: ランキング")

        # Part A: 鑑別推奨（分散）
        ranked_a = eng.rank_tests(filtered, novelty=novelty)
        print(f"\n  Part A: 鑑別推奨（分散ベース）Top15:")
        for i, t in enumerate(ranked_a[:15]):
            inv_idx = eng.test_idx.get(t["test_name"])
            inv_score = f" inv={eng.cos_invasive[inv_idx]:.3f}" if inv_idx is not None and eng.cos_invasive is not None else ""
            print(f"    #{i+1}: {t['test_name']}"
                  f" utility={t['utility']:.4f} score={t['score']:.4f}"
                  f" nov={t.get('novelty',1):.0f}{inv_score}")

        # Part B: Critical排除
        ranked_b = eng.rank_tests_critical(filtered, novelty=novelty)
        print(f"\n  Part B: Critical排除（最大命中）Top15:")
        for i, t in enumerate(ranked_b[:15]):
            print(f"    #{i+1}: {t['test_name']}"
                  f" utility={t['utility']:.4f} critical={t['critical_hit']:.4f}"
                  f" nov={t.get('novelty',1):.0f}"
                  f" → {t.get('hit_disease','')}")

        # Part C: 確認・同定
        ranked_c = eng.rank_tests_confirm(filtered, novelty=novelty)
        print(f"\n  Part C: 確認・同定（特異度）Top15:")
        for i, t in enumerate(ranked_c[:15]):
            print(f"    #{i+1}: {t['test_name']}"
                  f" utility={t['utility']:.4f} confirm={t['confirm_score']:.4f}"
                  f" nov={t.get('novelty',1):.0f}")

        # 期待される検査が推薦されているか
        for test_kw in case.get("expect_tests_recommended", []):
            found_a = False
            found_b = False
            found_c = False
            for i, t in enumerate(ranked_a[:20]):
                if test_kw in t["test_name"]:
                    found_a = True
                    check(f"Part A: '{test_kw}' がTop20", True, f"#{i+1}")
                    passes += 1
                    break
            if not found_a:
                # Part Aにない場合 — スコアを確認
                for t in ranked_a:
                    if test_kw in t["test_name"]:
                        rank_pos = ranked_a.index(t) + 1
                        check(f"Part A: '{test_kw}' がTop20", False,
                              f"実際: #{rank_pos} utility={t['utility']:.4f} score={t['score']:.4f}")
                        fails += 1
                        break
                else:
                    check(f"Part A: '{test_kw}' がTop20", False, "not found at all")
                    fails += 1

            for i, t in enumerate(ranked_b[:20]):
                if test_kw in t["test_name"]:
                    found_b = True
                    check(f"Part B: '{test_kw}' がTop20", True, f"#{i+1}")
                    passes += 1
                    break
            if not found_b:
                for t in ranked_b:
                    if test_kw in t["test_name"]:
                        rank_pos = ranked_b.index(t) + 1
                        check(f"Part B: '{test_kw}' がTop20", False,
                              f"実際: #{rank_pos}")
                        fails += 1
                        break

            for i, t in enumerate(ranked_c[:20]):
                if test_kw in t["test_name"]:
                    found_c = True
                    check(f"Part C: '{test_kw}' がTop20", True, f"#{i+1}")
                    passes += 1
                    break
            if not found_c:
                for t in ranked_c:
                    if test_kw in t["test_name"]:
                        rank_pos = ranked_c.index(t) + 1
                        check(f"Part C: '{test_kw}' がTop20", False,
                              f"実際: #{rank_pos}")
                        fails += 1
                        break

        # 侵襲的検査の抑制検証
        for test_kw in case.get("expect_invasive_demoted", []):
            for i, t in enumerate(ranked_a[:10]):
                if test_kw in t["test_name"]:
                    fails += 1
                    check(f"Part A: '{test_kw}' がTop10にない（侵襲的）", False,
                          f"#{i+1}")
                    break
            else:
                passes += 1
                check(f"Part A: '{test_kw}' がTop10にない", True)

        # ============================================================
        # 追加検証: utility=0.00の検査を確認
        # ============================================================
        subheader("追加検証: utility=0.00チェック")
        zero_util_a = [t for t in ranked_a[:20] if t["utility"] == 0.0 and t.get("novelty", 1) > 0]
        if zero_util_a:
            print(f"  Part A Top20にutility=0 & novelty>0の検査:")
            for t in zero_util_a:
                idx = eng.test_idx.get(t["test_name"])
                if idx is not None:
                    # sim_matrixの列を確認
                    col = eng.sim_matrix[:, idx]
                    print(f"    {t['test_name']}: score={t['score']:.6f} "
                          f"sim_col: min={col.min():.4f} max={col.max():.4f} "
                          f"std={col.std():.4f} mean={col.mean():.4f}")
                fails += 1
                check(f"utility=0.00: {t['test_name']}", False,
                      f"score={t['score']:.6f}")
        else:
            passes += 1
            check("utility=0.00の異常なし", True)

        # ============================================================
        # 追加検証: 血液培養の詳細分析（Case Bのみ）
        # ============================================================
        if case_id == "B":
            subheader("詳細分析: 血液培養")
            # sim_matrixでの血液培養の位置を確認
            bc_candidates = [t for t in eng.test_names if "血液培養" in t]
            print(f"  検査マスタ内の「血液培養」: {bc_candidates}")
            for bc_name in bc_candidates:
                idx = eng.test_idx[bc_name]
                col = eng.sim_matrix[:, idx]
                print(f"  {bc_name} (idx={idx}):")
                print(f"    sim_col stats: min={col.min():.4f} max={col.max():.4f} "
                      f"mean={col.mean():.4f} std={col.std():.4f}")

                # 敗血症との類似度
                for dname_key in ["敗血症", "敗血症性ショック", "菌血症"]:
                    for dname, didx in eng.disease_idx.items():
                        if dname_key in dname:
                            sim_val = eng.sim_matrix[didx, idx]
                            print(f"    sim({dname_key}→{bc_name}): {sim_val:.4f}")

                # novelty
                nov = novelty[idx]
                print(f"    novelty: {nov}")

                # 侵襲度
                if eng.cos_invasive is not None:
                    print(f"    cos_invasive: {eng.cos_invasive[idx]:.4f}")

                # ランキング位置
                for i, t in enumerate(ranked_a):
                    if t["test_name"] == bc_name:
                        print(f"    Part A rank: #{i+1} utility={t['utility']:.6f} "
                              f"score={t['score']:.6f}")
                        break
                for i, t in enumerate(ranked_b):
                    if t["test_name"] == bc_name:
                        print(f"    Part B rank: #{i+1} utility={t['utility']:.6f}")
                        break
                for i, t in enumerate(ranked_c):
                    if t["test_name"] == bc_name:
                        print(f"    Part C rank: #{i+1} utility={t['utility']:.6f} "
                              f"confirm={t['confirm_score']:.6f}")
                        break

        # ============================================================
        # 追加検証: 偶発性低体温症の詳細分析（候補に含まれる場合）
        # ============================================================
        subheader("詳細分析: 偶発性低体温症")
        for i, c in enumerate(candidates):
            if "低体温" in c["disease_name"]:
                print(f"  {c['disease_name']}: #{i+1} sim={c['similarity']:.4f}")
                # フィルタで除外されたか？
                still_in = any(f["disease_name"] == c["disease_name"] for f in filtered)
                excluded = any(r["disease_name"] == c["disease_name"] for r in exclusion_reasons)
                print(f"    フィルタ後残存: {still_in}")
                print(f"    除外された: {excluded}")
                if not excluded and i < 10 and "低体温" in c["disease_name"]:
                    # 発熱患者で低体温症がTop10にあるのは問題
                    if any(kw in case["text"] for kw in ["38", "39", "発熱"]):
                        fails += 1
                        check(f"発熱患者で '{c['disease_name']}' がTop10は不適切",
                              False, f"#{i+1}")
                break
        else:
            passes += 1
            check("偶発性低体温症は候補外", True)

        # ============================================================
        # 追加検証: 侵襲性ペナルティの動作確認
        # ============================================================
        subheader("侵襲性ペナルティ検証")
        # 候補疾患群のexpected_criticality
        raw_sims = np.array([c.get("similarity", 0.0) for c in filtered], dtype=float)
        weights = np.array([
            eng.disease_2c.get(c["disease_name"], {}).get("weight", 1.0)
            for c in filtered
        ], dtype=float)
        sim_centered = np.maximum(0.0, raw_sims - raw_sims.mean())
        w = sim_centered * weights
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        critical_scores = np.array([
            eng.disease_2c.get(c["disease_name"], {}).get("critical", 0.0)
            for c in filtered
        ], dtype=float)
        expected_crit = float(np.dot(w, critical_scores))
        print(f"  expected_criticality: {expected_crit:.4f}")

        if eng.cos_invasive is not None:
            top_invasive = sorted(
                [(eng.test_names[j], eng.cos_invasive[j]) for j in range(len(eng.test_names))],
                key=lambda x: x[1], reverse=True
            )[:5]
            print(f"  Top5侵襲的検査:")
            for tname, score in top_invasive:
                penalty = max(0, score - expected_crit)
                discount = np.exp(-penalty)
                print(f"    {tname}: cos_inv={score:.4f} penalty={penalty:.4f} "
                      f"discount={discount:.4f}")

        # ============================================================
        # Case小計
        # ============================================================
        subheader(f"Case {case_id} 集計")
        print(f"  PASS: {passes}  FAIL: {fails}")
        total_pass += passes
        total_fail += fails

    # ============================================================
    # 全体集計 + LLMモデル評価
    # ============================================================
    header("全体集計")
    print(f"  Total PASS: {total_pass}")
    print(f"  Total FAIL: {total_fail}")
    print(f"  合格率: {total_pass/(total_pass+total_fail)*100:.0f}%"
          if (total_pass+total_fail) > 0 else "  テストなし")

    header("LLMモデル評価サマリー")
    print(f"  使用モデル: {VERTEX_MODEL}")
    print(f"  天井 = LLM記述の質 × LLM推論の質")
    print(f"  このテストでの推論タスク:")
    print(f"    1. テキスト3分類 (split): 陽性/陰性/検査の分離精度")
    print(f"    2. 矛盾検出 (filter): 陰性所見→疾患除外の正確性")
    print(f"    3. Novelty判定: 実施済み検査の正確な特定")
    print(f"    4. HPE極性判定: 陽性/陰性の正確な判定")
    print()
    print(f"  【注意】 上記FAILが多い場合、LLMの推論品質が不足している可能性がある。")
    print(f"  gemini-3-flash-preview は高速だが、臨床推論ではpro/ultraの方が安全。")


if __name__ == "__main__":
    main()
