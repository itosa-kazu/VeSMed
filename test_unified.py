"""
VeSMed 統一テストランナー
test_cases_manual.json (30件) + eval_cases.json (21件) = 51ケースで
A方式 (sim_matrix + column-mean centering) を評価。

スコアリング:
  correct + direction="up"   → 順位↑: +1, 変化なし: 0, 順位↓: -1
  correct + direction="stay" → 変化小(±5以内): +1, 大変化: 0
  correct + direction="down" → 順位↓: +1, 変化なし: 0, 順位↑: -1
  wrong   + direction="down" → 順位↓: +1, 変化なし: 0, 順位↑: -1
"""

import json
import os
import sys
import time

# VeSMedエンジン読み込み
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import VeSMedEngine

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_manual_cases():
    """test_cases_manual.json を統一形式で読み込み"""
    path = os.path.join(DATA_DIR, "test_cases_manual.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cases = []
    for item in raw:
        expectations = []
        for c in item.get("correct", []):
            expectations.append({
                "disease": c["disease"],
                "direction": c["direction"],  # up/stay/down
                "type": "correct",
            })
        for w in item.get("wrong", []):
            expectations.append({
                "disease": w["disease"],
                "direction": w["direction"],
                "type": "wrong",
            })
        cases.append({
            "id": item["id"],
            "name": item["name"],
            "symptoms": item["symptoms"],
            "results": item["results"],
            "expectations": expectations,
        })
    return cases


def load_eval_cases():
    """eval_cases.json を統一形式で読み込み"""
    path = os.path.join(DATA_DIR, "eval_cases.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cases = []
    for item in raw:
        expectations = []
        # 正解疾患: 順位が上がるかTOP維持
        expectations.append({
            "disease": item["correct_diagnosis"],
            "direction": "up",
            "type": "correct",
        })
        # 不正解疾患: 順位が下がるべき
        for wd in item.get("wrong_diagnoses", []):
            expectations.append({
                "disease": wd,
                "direction": "down",
                "type": "wrong",
            })
        cases.append({
            "id": item["id"],
            "name": f'{item.get("specialty", "")} - {item["correct_diagnosis"]}',
            "symptoms": item["symptoms"],
            "results": item["results"],
            "expectations": expectations,
        })
    return cases


def get_rank(candidates, disease_name):
    """候補リストから疾患の順位を返す（1-indexed, 見つからなければ999）"""
    for i, c in enumerate(candidates):
        if c["disease_name"] == disease_name:
            return i + 1
    return 999


def score_expectation(rank_before, rank_after, direction):
    """
    1つの期待に対してスコアを計算。
    direction: "up" (順位を上げるべき), "down" (順位を下げるべき), "stay" (維持)
    返り値: +1 (良), 0 (中立), -1 (悪)
    """
    delta = rank_before - rank_after  # 正=順位上昇（数字が減った）

    if direction == "up":
        if delta > 0:
            return +1  # 順位上昇 → OK
        elif delta == 0:
            return 0   # 変化なし
        else:
            return -1  # 順位低下 → NG
    elif direction == "down":
        if delta < 0:
            return +1  # 順位低下 → OK（不正解が沈んだ）
        elif delta == 0:
            return 0   # 変化なし
        else:
            return -1  # 順位上昇 → NG（不正解が浮いた）
    elif direction == "stay":
        if abs(delta) <= 5:
            return +1  # 変化小 → OK
        else:
            return 0   # 大変化
    return 0


def run_test(eng, case):
    """1ケースを実行し、スコアと詳細を返す"""
    # Step 1: 症状のみで初期検索
    candidates_before = eng.search_diseases(case["symptoms"])

    # Step 2: 検査結果で重み更新（A方式: sim_matrix + centering）
    import copy
    candidates_after = copy.deepcopy(candidates_before)
    candidates_after = eng.update_from_results(
        candidates_after, case["results"], case["symptoms"], mode="fast"
    )

    # Step 3: 各期待について順位変化を評価
    results = []
    case_score = 0
    for exp in case["expectations"]:
        r_before = get_rank(candidates_before, exp["disease"])
        r_after = get_rank(candidates_after, exp["disease"])
        s = score_expectation(r_before, r_after, exp["direction"])
        case_score += s

        results.append({
            "disease": exp["disease"],
            "type": exp["type"],
            "direction": exp["direction"],
            "rank_before": r_before,
            "rank_after": r_after,
            "delta": r_before - r_after,
            "score": s,
            "status": "OK" if s >= 0 else "NG",
            "found": r_before != 999,
        })

    return case_score, results


def main():
    print("=" * 70)
    print("VeSMed 統一テストランナー")
    print("=" * 70)

    # エンジン初期化
    print("\n[初期化] エンジン起動中...")
    t0 = time.time()
    eng = VeSMedEngine()
    print(f"[初期化] {time.time()-t0:.1f}秒\n")

    # テストケース読み込み
    manual_cases = load_manual_cases()
    eval_cases = load_eval_cases()
    all_cases = manual_cases + eval_cases
    print(f"テストケース: manual={len(manual_cases)}, eval={len(eval_cases)}, 合計={len(all_cases)}")

    # 疾患名存在チェック
    print("\n[疾患名チェック]")
    missing = set()
    for case in all_cases:
        for exp in case["expectations"]:
            if exp["disease"] not in eng.disease_db:
                missing.add(exp["disease"])
    if missing:
        print(f"  ⚠ DB未登録疾患 ({len(missing)}件):")
        for m in sorted(missing):
            print(f"    - {m}")
    else:
        print("  全疾患名がDBに存在 ✓")

    # テスト実行
    print("\n" + "=" * 70)
    total_score = 0
    total_ok = 0
    total_ng = 0
    total_neutral = 0
    total_not_found = 0
    case_results = []

    for i, case in enumerate(all_cases):
        print(f"\n--- [{case['id']}] {case['name']} ---")
        score, details = run_test(eng, case)
        total_score += score
        case_results.append({"case": case, "score": score, "details": details})

        for d in details:
            if not d["found"]:
                total_not_found += 1
                mark = "N/A"
            elif d["score"] > 0:
                total_ok += 1
                mark = "OK"
            elif d["score"] < 0:
                total_ng += 1
                mark = "NG"
            else:
                total_neutral += 1
                mark = "--"

            dir_arrow = {"up": "↑", "down": "↓", "stay": "→"}.get(d["direction"], "?")
            delta_str = f"{d['delta']:+d}" if d["found"] else "N/A"
            print(f"  [{mark}] {d['type']:7s} {dir_arrow} {d['disease']}: "
                  f"{d['rank_before']}→{d['rank_after']} (Δ{delta_str})")

        print(f"  ケーススコア: {score:+d}")

    # サマリー
    total_checks = total_ok + total_ng + total_neutral + total_not_found
    print("\n" + "=" * 70)
    print("サマリー")
    print("=" * 70)
    print(f"ケース数:     {len(all_cases)}")
    print(f"チェック総数: {total_checks}")
    print(f"  OK:    {total_ok:3d} (期待通り)")
    print(f"  NG:    {total_ng:3d} (期待と逆)")
    print(f"  中立:  {total_neutral:3d} (変化なし/許容範囲)")
    if total_not_found > 0:
        print(f"  N/A:   {total_not_found:3d} (DB未登録)")
    print(f"合計スコア:   {total_score:+d}")
    print(f"正答率: {total_ok}/{total_ok+total_ng} = "
          f"{total_ok/(total_ok+total_ng)*100:.1f}%" if (total_ok+total_ng) > 0 else "N/A")

    # NGケース一覧
    ng_details = []
    for cr in case_results:
        for d in cr["details"]:
            if d["score"] < 0:
                ng_details.append({
                    "case_id": cr["case"]["id"],
                    "case_name": cr["case"]["name"],
                    **d,
                })

    if ng_details:
        print(f"\n--- NGケース詳細 ({len(ng_details)}件) ---")
        for ng in ng_details:
            dir_arrow = {"up": "↑", "down": "↓", "stay": "→"}.get(ng["direction"], "?")
            print(f"  [{ng['case_id']}] {ng['disease']} "
                  f"(期待{dir_arrow}): {ng['rank_before']}→{ng['rank_after']}")


if __name__ == "__main__":
    main()
