"""
VeSMed v2改善テスト: 5タスクの効果検証
"""
import sys
import time
import numpy as np

# テストケース: 47歳女性、頭痛+嘔気+バイタル+陰性所見あり
TEST_CASE = """47歳女性。3日前からの持続する頭痛と嘔気を主訴に来院。
体温37.5℃、血圧120/80mmHg、脈拍88/分、SpO2 98%。
項部硬直なし。Kernig徴候陰性。Murphy徴候陰性。
CRP 2.5mg/dL、WBC 9800/μL。"""


def main():
    print("=" * 60)
    print("VeSMed v2改善テスト")
    print("=" * 60)

    # エンジン初期化
    print("\n[1] エンジン初期化...")
    t0 = time.time()
    from engine import VeSMedEngine
    eng = VeSMedEngine()
    t1 = time.time()
    print(f"  初期化完了: {t1-t0:.1f}s")
    print(f"  検査数: {len(eng.test_names)} (パージ後、期待: ~351)")
    print(f"  HPE数: {len(eng.hpe_names)} (期待: 274)")

    # Task 3検証: マスタパージ
    print("\n[2] Task 3検証: マスタパージ")
    purged_names = ["バイタルサイン測定", "意識レベルの評価", "問診：薬剤歴"]
    for name in purged_names:
        in_list = name in eng.test_idx
        print(f"  '{name}' in test_names: {in_list} (期待: False)")

    # Task 4検証: 侵襲性スコア
    print("\n[3] Task 4検証: 侵襲性スコア")
    if eng.cos_invasive is not None:
        print(f"  cos_invasive shape: {eng.cos_invasive.shape}")
        # 侵襲的検査の例
        for tname in ["骨髄穿刺", "腰椎穿刺（髄液検査）", "CBC（白血球分画を含む）", "CRP"]:
            if tname in eng.test_idx:
                idx = eng.test_idx[tname]
                print(f"  {tname}: cos_invasive={eng.cos_invasive[idx]:.3f}")
    else:
        print("  cos_invasive: None (未計算)")

    # Task 1検証: 陽性/陰性分離
    print("\n[4] Task 1検証: 陽性/陰性分離")
    t2 = time.time()
    positive_text, negative_findings, result_lines = eng.split_symptoms_results(TEST_CASE)
    t3 = time.time()
    print(f"  分離時間: {t3-t2:.1f}s")
    print(f"  陽性テキスト: {positive_text[:80]}...")
    print(f"  陰性所見: {negative_findings}")
    print(f"  検査結果: {result_lines}")

    # 陰性所見に項部硬直なし等が含まれているか
    neg_has_rigid = any("項部硬直" in nf for nf in negative_findings)
    neg_has_kernig = any("Kernig" in nf for nf in negative_findings)
    print(f"  項部硬直なし → negative: {neg_has_rigid} (期待: True)")
    print(f"  Kernig陰性 → negative: {neg_has_kernig} (期待: True)")

    # 陽性テキストに否定所見が含まれていないか
    pos_has_rigid = "項部硬直なし" in positive_text
    print(f"  項部硬直なし → positive: {pos_has_rigid} (期待: False)")

    # 検索: 陽性テキストのみで検索
    candidates = eng.search_diseases(positive_text)
    candidates = eng.compute_priors(candidates)
    print(f"  候補疾患数: {len(candidates)}")
    # 髄膜炎のランキングを確認
    for i, c in enumerate(candidates[:20]):
        if "髄膜炎" in c["disease_name"]:
            print(f"  髄膜炎: #{i+1} sim={c['similarity']:.4f}")

    # Task 2 + Task 5検証: filter_contradictions + 統合Novelty
    print("\n[5] Task 2+5検証: フィルタ + 統合Novelty")
    t4 = time.time()

    # filter_contradictions（陰性所見付き）
    filtered, exclusion_reasons = eng.filter_contradictions(
        candidates[:20], TEST_CASE, negative_findings
    )
    t5 = time.time()
    print(f"  フィルタ時間: {t5-t4:.1f}s")
    print(f"  除外疾患: {len(exclusion_reasons)}件")
    for r in exclusion_reasons:
        print(f"    - {r['disease_name']}: {r.get('reason', '')}")

    # 統合Novelty
    t6 = time.time()
    novelty, novelty_hpe, hpe_findings = eng.compute_all_novelty(TEST_CASE)
    t7 = time.time()
    print(f"  統合Novelty時間: {t7-t6:.1f}s")
    n_suppressed = int((novelty == 0.0).sum())
    n_hpe_done = int((novelty_hpe == 0.0).sum())
    print(f"  検査抑制: {n_suppressed}件")
    print(f"  HPE聴取済み: {n_hpe_done}件")
    print(f"  HPE所見: {len(hpe_findings)}件")
    for f in hpe_findings[:5]:
        print(f"    - {f['item']} (polarity={f['polarity']:+d})")

    # バイタルサイン測定がnovelty=0でないことを確認（パージ済み）
    if "バイタルサイン測定" in eng.test_idx:
        print(f"  警告: バイタルサイン測定がまだtest_idxに存在")
    else:
        print(f"  バイタルサイン測定: パージ済み (OK)")

    # CRP, WBCがnovelty=0か確認（テキストに結果あり）
    for tname in ["CRP", "CBC（白血球分画を含む）"]:
        if tname in eng.test_idx:
            idx = eng.test_idx[tname]
            print(f"  {tname}: novelty={novelty[idx]:.0f} (期待: 0)")

    # Task 4検証: ランキングで侵襲性penalty
    print("\n[6] Task 4検証: 侵襲性penaltyの効果")
    ranked = eng.rank_tests(filtered, novelty=novelty)
    print(f"  Top10検査:")
    for i, t in enumerate(ranked[:10]):
        print(f"    #{i+1}: {t['test_name']} utility={t['utility']:.4f}")

    # 侵襲的検査のランキング位置
    for tname in ["骨髄穿刺", "腰椎穿刺（髄液検査）"]:
        for i, t in enumerate(ranked):
            if t["test_name"] == tname:
                print(f"  {tname}: #{i+1} (軽症なので高ランクでないことを期待)")
                break

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
