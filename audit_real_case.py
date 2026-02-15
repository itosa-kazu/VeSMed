"""
実臨床症例でのパイプライン精査
47歳女性、左脇～背中の痛み→全身痛→嘔吐・水様便→39度発熱→救急搬送
"""
import json
import time
import copy
import numpy as np

CASE_TEXT = """47歳女性。生来健康、学校給食職員で2児の母。
来院2日前までは何ともなかった。来院1日前の昼に左脇～背中のあたりが痛くなり、夜間には寝返りを打てないくらいの痛みとなった。そのころから全身、腕や太腿の内側も痛くなり、胃液の嘔吐、水っぽい便がほぼ同時に出現した。
来院当日の早朝に痛みがひどく39度まで発熱したため、本人が救急車を要請した。
既往歴：消化性潰瘍なし、関節リウマチなし、肝疾患なし、甲状腺疾患なし、COPDなし
渡航歴なし
右下腹部痛なし
血尿なし
めまいなし
裂けるような痛みではない
便秘なし
失神なし
発疹なし
単関節痛なし
疝痛なし
来院時：意識はややぼんやり、グッタリの外観。意識レベル低下。
体温37.5℃、血圧92/60mmHg、心拍数120回/分（整）、呼吸数26回/分、SpO2 89%（room air）"""


def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def subheader(title):
    print(f"\n--- {title} ---")


def main():
    header("実臨床症例パイプライン精査")
    print(f"入力:\n{CASE_TEXT}\n")

    # エンジン初期化
    print("エンジン初期化中...")
    t0 = time.time()
    from engine import VeSMedEngine
    eng = VeSMedEngine()
    print(f"初期化完了: {time.time()-t0:.1f}s")

    # ============================================================
    # Step 0: テキスト分離
    # ============================================================
    subheader("Step 0: テキスト3分類 (LLM)")
    t1 = time.time()
    positive_text, negative_findings, result_lines = eng.split_symptoms_results(CASE_TEXT)
    print(f"所要時間: {time.time()-t1:.1f}s")
    print(f"\n陽性テキスト:\n  {positive_text}")
    print(f"\n陰性所見 ({len(negative_findings)}件):")
    for nf in negative_findings:
        print(f"  - {nf}")
    print(f"\n検査結果 ({len(result_lines)}件):")
    for r in result_lines:
        print(f"  - {r}")

    # 臨床的に重要な陰性所見の検証
    important_negatives = [
        "消化性潰瘍", "関節リウマチ", "肝疾患", "甲状腺疾患", "COPD",
        "渡航歴", "右下腹部痛", "血尿", "めまい", "裂けるような痛み",
        "便秘", "失神", "発疹", "単関節痛", "疝痛"
    ]
    print(f"\n重要な陰性所見の分離チェック:")
    for neg in important_negatives:
        in_neg = any(neg in nf for nf in negative_findings)
        in_pos = neg in positive_text
        mark = "✓" if in_neg else ("✗" if in_pos else "?")
        print(f"  [{mark}] {neg}: neg={in_neg}, pos漏れ={in_pos}")

    # ============================================================
    # Step 1: 疾患検索 (embedding)
    # ============================================================
    subheader("Step 1: 疾患検索 (embedding — 知覚)")
    if not positive_text.strip():
        positive_text = CASE_TEXT
    t2 = time.time()
    candidates = eng.search_diseases(positive_text)
    candidates = eng.compute_priors(candidates)
    print(f"所要時間: {time.time()-t2:.1f}s")
    print(f"候補数: {len(candidates)}")
    print(f"\nTop30疾患:")
    for i, c in enumerate(candidates[:30]):
        print(f"  #{i+1:2d}: {c['disease_name']:30s} sim={c['similarity']:.4f} "
              f"w={c.get('clinical_weight',0):.2f} "
              f"urg={c.get('urgency','')}")

    # 臨床的に期待される疾患
    expected_diseases = [
        "敗血症", "敗血症性ショック", "急性膵炎", "大動脈解離",
        "腸間膜虚血", "壊死性筋膜炎",
        "トキシックショック", "A群溶連菌",
    ]
    print(f"\n期待疾患の位置:")
    for dname in expected_diseases:
        for i, c in enumerate(candidates):
            if dname in c["disease_name"]:
                print(f"  {dname}: #{i+1} sim={c['similarity']:.4f}")
                break
        else:
            print(f"  {dname}: NOT FOUND")

    # ============================================================
    # Step 2: LLMフィルタ (矛盾除外)
    # ============================================================
    subheader("Step 2: LLMフィルタ (矛盾除外)")
    cands_filter = copy.deepcopy(candidates[:20])
    t3 = time.time()
    filtered, exclusion_reasons = eng.filter_contradictions(
        cands_filter, CASE_TEXT, negative_findings
    )
    print(f"所要時間: {time.time()-t3:.1f}s")
    print(f"除外: {len(exclusion_reasons)}件")
    for r in exclusion_reasons:
        print(f"  ✗ {r['disease_name']}: {r.get('reason','')}")
    print(f"残存: {len(filtered)}件")

    print(f"\nフィルタ後Top15:")
    for i, c in enumerate(filtered[:15]):
        print(f"  #{i+1:2d}: {c['disease_name']:30s} sim={c['similarity']:.4f}")

    # 除外された疾患の妥当性チェック
    excluded_names = {r["disease_name"] for r in exclusion_reasons}
    # 大動脈解離は「裂けるような痛みではない」で除外されるべきか？
    if "大動脈解離" in excluded_names:
        print(f"\n  ★ 大動脈解離が除外された — 「裂けるような痛みではない」から妥当")
    else:
        print(f"\n  ★ 大動脈解離が除外されていない")
        for i, c in enumerate(filtered):
            if "大動脈解離" in c["disease_name"]:
                print(f"    → #{i+1}位に残存")

    # ============================================================
    # Step 3: 検査結果更新
    # ============================================================
    if result_lines:
        subheader("Step 3: 検査結果更新")
        cands_update = copy.deepcopy(filtered)
        t4 = time.time()
        updated = eng.update_from_results(cands_update, result_lines, positive_text)
        print(f"所要時間: {time.time()-t4:.1f}s")
        print(f"\n更新後Top15:")
        for i, c in enumerate(updated[:15]):
            print(f"  #{i+1:2d}: {c['disease_name']:30s} sim={c['similarity']:.4f}")
        filtered = updated
    else:
        print("\n検査結果なし → Step 3スキップ")

    # ============================================================
    # Step 3.5: HPE所見更新
    # ============================================================
    subheader("Step 3.5: HPE所見による疾患重み更新")
    # 先にnoveltyを計算してhpe_findingsを取得
    t_nov = time.time()
    novelty, novelty_hpe, hpe_findings = eng.compute_all_novelty(CASE_TEXT)
    print(f"統合Novelty所要時間: {time.time()-t_nov:.1f}s")

    print(f"\n抑制された検査 ({int((novelty==0).sum())}件):")
    for j in range(len(eng.test_names)):
        if novelty[j] == 0.0:
            print(f"  - {eng.test_names[j]}")

    print(f"\nHPE所見 ({len(hpe_findings)}件):")
    for f in hpe_findings:
        print(f"  {f['item']:25s} polarity={f['polarity']:+d}")

    if hpe_findings:
        cands_hpe = copy.deepcopy(filtered)
        cands_hpe = eng.update_from_hpe(cands_hpe, hpe_findings)
        print(f"\nHPE更新後Top15:")
        for i, c in enumerate(cands_hpe[:15]):
            print(f"  #{i+1:2d}: {c['disease_name']:30s} sim={c['similarity']:.4f}")
        filtered = cands_hpe

    # ============================================================
    # Step 4: ランキング
    # ============================================================
    subheader("Step 4: 検査ランキング")

    # 侵襲性ペナルティの基礎情報
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
    print(f"expected_criticality: {expected_crit:.4f}")

    # Part A
    ranked_a = eng.rank_tests(filtered, novelty=novelty)
    print(f"\nPart A: 鑑別推奨（分散ベース）Top20:")
    for i, t in enumerate(ranked_a[:20]):
        inv_idx = eng.test_idx.get(t["test_name"])
        inv = eng.cos_invasive[inv_idx] if inv_idx is not None and eng.cos_invasive is not None else 0
        penalty = max(0, inv - expected_crit)
        related = ", ".join(d["disease_name"] for d in t.get("details", [])[:3])
        print(f"  #{i+1:2d}: {t['test_name']:40s} "
              f"util={t['utility']:.4f} var={t['score']:.4f} "
              f"nov={t.get('novelty',1):.0f} inv={inv:.3f} pen={penalty:.3f} "
              f"← {related}")

    # Part B
    ranked_b = eng.rank_tests_critical(filtered, novelty=novelty)
    print(f"\nPart B: Critical排除（最大命中）Top20:")
    for i, t in enumerate(ranked_b[:20]):
        print(f"  #{i+1:2d}: {t['test_name']:40s} "
              f"util={t['utility']:.4f} crit={t['critical_hit']:.4f} "
              f"nov={t.get('novelty',1):.0f} → {t.get('hit_disease','')}")

    # Part C
    ranked_c = eng.rank_tests_confirm(filtered, novelty=novelty)
    print(f"\nPart C: 確認・同定（特異度）Top20:")
    for i, t in enumerate(ranked_c[:20]):
        related = ", ".join(d["disease_name"] for d in t.get("details", [])[:3])
        print(f"  #{i+1:2d}: {t['test_name']:40s} "
              f"util={t['utility']:.4f} conf={t['confirm_score']:.4f} "
              f"nov={t.get('novelty',1):.0f} ← {related}")

    # Part D
    ranked_hpe = eng.rank_hpe(filtered, novelty_hpe=novelty_hpe)
    hx_items = [r for r in ranked_hpe if r["category"] == "Hx"]
    pe_items = [r for r in ranked_hpe if r["category"] == "PE"]
    print(f"\nPart D-1: 問診推奨 Top10:")
    for i, r in enumerate(hx_items[:10]):
        print(f"  #{i+1:2d}: {r['item_name']:25s} util={r['utility']:.4f} "
              f"sub={r['subcategory']}")
    print(f"\nPart D-2: 身体診察推奨 Top10:")
    for i, r in enumerate(pe_items[:10]):
        print(f"  #{i+1:2d}: {r['item_name']:25s} util={r['utility']:.4f} "
              f"sub={r['subcategory']}")

    # ============================================================
    # 臨床的妥当性の総合評価
    # ============================================================
    header("臨床的妥当性の総合評価")

    print("""
この症例の臨床的キーポイント:
  - 急性発症の左脇～背中の痛み → 全身痛に進展
  - 嘔吐・水様便
  - 39度の発熱
  - バイタル不安定: BP 92/60, HR 120, RR 26, SpO2 89%
  - 意識レベル低下
  → ショック状態（SIRS/qSOFA陽性）
  → 多数の陰性所見（消化性潰瘍なし、RA なし、等）

臨床的に考えるべき鑑別:
  1. 敗血症/敗血症性ショック（感染巣不明）
  2. 壊死性筋膜炎（急速進行する疼痛、全身状態悪化）
  3. トキシックショック症候群（急激な発症）
  4. 急性膵炎（左脇腹～背部痛、嘔吐）
  5. 腸間膜虚血（腹痛、下痢、ショック）
  6. 劇症型A群溶連菌感染症（壊死性筋膜炎の原因）

最低限必要な検査:
  - 血液培養（必須）
  - 血液ガス（乳酸）
  - CBC、CRP、PCT
  - 肝機能、腎機能、CK
  - 凝固（DIC評価）
  - CT（感染巣検索）
""")

    # 重要検査の位置を確認
    critical_tests = [
        "血液培養", "乳酸", "血液ガス", "Dダイマー", "FDP",
        "CK", "腹部CT", "胸部CT",
        "グラム染色", "フィブリノゲン",
    ]
    print("重要検査のランキング位置:")
    for test_kw in critical_tests:
        pos_a = pos_b = pos_c = "N/A"
        for i, t in enumerate(ranked_a):
            if test_kw in t["test_name"]:
                pos_a = f"#{i+1}"
                break
        for i, t in enumerate(ranked_b):
            if test_kw in t["test_name"]:
                pos_b = f"#{i+1}"
                break
        for i, t in enumerate(ranked_c):
            if test_kw in t["test_name"]:
                pos_c = f"#{i+1}"
                break
        print(f"  {test_kw:20s}  A={pos_a:6s}  B={pos_b:6s}  C={pos_c:6s}")


if __name__ == "__main__":
    main()
