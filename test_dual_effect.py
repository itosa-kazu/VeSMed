"""
Dual sim_matrix 効果検証: 臨床シナリオでPart B/Cのランキングが改善したか比較。

方法:
  1. engine を Dual モード（新）で起動
  2. 各シナリオで Part A/B/C を取得
  3. sim_matrix_confirm を sim_matrix に差し替えて（旧モード）再計算
  4. 新旧の Top10 を並べて比較
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engine import VeSMedEngine

# ─── テストシナリオ ───
SCENARIOS = [
    {
        "name": "血管炎疑い（ANCA関連）",
        "input": "38歳男性。2週間前からの発熱、関節痛、下腿の紫斑。尿検査で蛋白尿と血尿。BUN/Cr上昇。",
        "expect_partC": "MPO-ANCA, PR3-ANCA が上位であるべき",
        "expect_partB": "腎生検、MPO-ANCA等が上位",
    },
    {
        "name": "甲状腺機能亢進症疑い",
        "input": "28歳女性。3ヶ月前からの動悸、体重減少（5kg）、手指振戦、発汗増加。眼球突出あり。",
        "expect_partC": "TSH, FT4, TRAb が上位であるべき",
        "expect_partB": "甲状腺クリーゼ除外のためFT4, TSH",
    },
    {
        "name": "急性心不全疑い",
        "input": "72歳男性。数日前からの労作時呼吸困難、起座呼吸、下腿浮腫。既往に高血圧。聴診で両側水泡音。",
        "expect_partC": "BNP, 心エコー が上位であるべき",
        "expect_partB": "トロポニン（ACS除外）、BNP",
    },
    {
        "name": "SLE疑い",
        "input": "25歳女性。顔面の蝶形紅斑、関節痛、口腔内潰瘍、光線過敏。血沈亢進。",
        "expect_partC": "抗核抗体、抗dsDNA が上位であるべき",
        "expect_partB": "ループス腎炎除外の検査",
    },
    {
        "name": "敗血症疑い",
        "input": "65歳男性。悪寒戦慄を伴う高熱39.5度、頻脈120、血圧90/60。尿路感染の既往あり。",
        "expect_partC": "血液培養が上位。CRPは汎用なので下位であるべき",
        "expect_partB": "血液培養、乳酸が上位",
    },
]


def run_scenario(engine, scenario):
    """1シナリオ分の検査推奨を取得（Part A/B/C）"""
    text = scenario["input"]

    # Step 0: 入力分類（LLM呼び出し回避 — テキスト全体を陽性所見として使用）
    # Step 1: embedding検索
    candidates = engine.search_diseases(text)

    # Step 2: LLMフィルタ（スキップ — embedding結果だけで比較）
    # negatives考慮もスキップ（Part B/Cの行列比較が目的）

    if not candidates:
        return None, None, None

    # novelty = 1（初回、全検査未知）
    novelty = np.ones(len(engine.test_names))

    part_a = engine.rank_tests(candidates, novelty)
    part_c = engine.rank_tests_confirm(candidates, novelty)
    part_b = engine.rank_tests_critical(candidates, novelty)

    return part_a, part_b, part_c


def print_comparison(label, new_list, old_list, top_n=10):
    """新旧ランキングを横並びで表示"""
    print(f"\n  {label}")
    print(f"  {'Rank':<5} {'新(Dual)':^40} {'旧(Single)':^40}")
    print(f"  {'─'*85}")
    for i in range(top_n):
        new_name = new_list[i]["test_name"] if i < len(new_list) else "-"
        new_util = new_list[i]["utility"] if i < len(new_list) else 0
        old_name = old_list[i]["test_name"] if i < len(old_list) else "-"
        old_util = old_list[i]["utility"] if i < len(old_list) else 0

        # 新でランクアップした検査にマーク
        marker = ""
        if i < len(new_list):
            old_rank = next((j for j, r in enumerate(old_list) if r["test_name"] == new_name), 999)
            if old_rank > i:
                marker = f" ↑{old_rank - i}"
            elif old_rank < i:
                marker = f" ↓{i - old_rank}"

        print(f"  {i+1:<5} {new_name:<32} {new_util:>7.4f}{marker:<5}  {old_name:<32} {old_util:>7.4f}")


def main():
    print("=" * 90)
    print("Dual sim_matrix 効果検証")
    print("=" * 90)

    # エンジン起動（Dualモード）
    engine = VeSMedEngine()
    print(f"\nsim_matrix: {engine.sim_matrix.shape}")
    print(f"sim_matrix_confirm: {engine.sim_matrix_confirm.shape}")
    print(f"Dual active: {engine.sim_matrix is not engine.sim_matrix_confirm}")

    for scenario in SCENARIOS:
        print(f"\n{'='*90}")
        print(f"シナリオ: {scenario['name']}")
        print(f"入力: {scenario['input']}")
        print(f"期待 Part C: {scenario['expect_partC']}")
        print(f"期待 Part B: {scenario['expect_partB']}")
        print(f"{'='*90}")

        # ─── 新（Dual）モード ───
        part_a_new, part_b_new, part_c_new = run_scenario(engine, scenario)
        if part_a_new is None:
            print("  候補疾患なし — スキップ")
            continue

        # ─── 旧（Single）モード: confirm を screen に差し替え ───
        saved_confirm = engine.sim_matrix_confirm
        engine.sim_matrix_confirm = engine.sim_matrix  # 旧動作を再現
        _, part_b_old, part_c_old = run_scenario(engine, scenario)
        engine.sim_matrix_confirm = saved_confirm  # 復元

        # ─── 比較表示 ───
        print_comparison("Part C（確認力）— Dual vs Single", part_c_new, part_c_old)
        print_comparison("Part B（致死除外）— Dual vs Single", part_b_new, part_b_old)

        # ─── 変化サマリー ───
        # Part Cで大きくランクアップした検査を抽出
        changes = []
        for i, r in enumerate(part_c_new[:20]):
            old_rank = next((j for j, o in enumerate(part_c_old) if o["test_name"] == r["test_name"]), 999)
            if old_rank - i >= 3:  # 3位以上のランクアップ
                changes.append((r["test_name"], old_rank + 1, i + 1, old_rank - i))
        if changes:
            print(f"\n  Part C 大幅ランクアップ（3位以上）:")
            for name, old_r, new_r, delta in sorted(changes, key=lambda x: -x[3]):
                print(f"    {name}: {old_r}位 → {new_r}位 (+{delta})")


if __name__ == "__main__":
    main()
