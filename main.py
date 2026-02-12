"""
VeSMed - 対話型CLI
症状 → 候補疾患 → 検査推薦 → 検査結果で重み更新（Option 3）→ ループ
"""

from engine import VeSMedEngine
from config import TOP_K_TESTS


def format_candidates(candidates, top_n=10):
    """候補疾患を表示用にフォーマット"""
    lines = []
    lines.append("=" * 60)
    lines.append("【候補疾患】")
    lines.append("-" * 60)
    for i, c in enumerate(candidates[:top_n]):
        urgency = c.get("urgency", "")
        urgency_mark = ""
        if urgency == "超緊急":
            urgency_mark = " !!!"
        elif urgency == "緊急":
            urgency_mark = " !!"
        elif urgency == "準緊急":
            urgency_mark = " !"

        w = c.get("clinical_weight", 0)
        lines.append(
            f"  {i + 1:2d}. {c['disease_name']:<30s}  "
            f"sim={c['similarity']:.3f}  "
            f"w={w:.2f}"
            f"{urgency_mark}"
        )
    lines.append("")
    return "\n".join(lines)


def format_tests(tests, top_n=10):
    """Part A: 推薦検査を表示用にフォーマット"""
    lines = []
    lines.append("【Part A: 鑑別推奨（分散ベース）】")
    lines.append("-" * 60)
    lines.append(f"  {'#':>3s}  {'検査名':<25s}  {'分散':>8s}  {'質':>6s}  {'効用':>8s}")
    lines.append("-" * 60)
    for i, t in enumerate(tests[:top_n]):
        lines.append(
            f"  {i + 1:3d}  {t['test_name']:<25s}  "
            f"{t['score']:8.4f}  "
            f"{t['quality']:6.4f}  "
            f"{t['utility']:8.4f}"
        )
    lines.append("=" * 60)
    lines.append("")
    return "\n".join(lines)


def format_confirm(tests, top_n=10):
    """Part C: 確認・同定推奨を表示用にフォーマット"""
    lines = []
    lines.append("【Part C: 確認・同定推奨（クラスタ特異度）】")
    lines.append("-" * 60)
    lines.append(f"  {'#':>3s}  {'検査名':<25s}  {'特異':>8s}  {'質':>6s}  {'効用':>8s}")
    lines.append("-" * 60)
    for i, t in enumerate(tests[:top_n]):
        lines.append(
            f"  {i + 1:3d}  {t['test_name']:<25s}  "
            f"{t['confirm_score']:8.4f}  "
            f"{t['quality']:6.4f}  "
            f"{t['utility']:8.4f}"
        )
    lines.append("=" * 60)
    lines.append("")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("  VeSMed - ベクトル空間医学統一フレームワーク")
    print("  Vector Space Medicine Unified Framework")
    print("=" * 60)
    print()

    print("エンジン初期化中...")
    try:
        engine = VeSMedEngine()
    except Exception as e:
        print(f"エラー: エンジン初期化に失敗しました: {e}")
        print("先に generate.py と index.py を実行してください。")
        return

    print(f"疾患データベース: {len(engine.disease_db)}件")
    print(f"ベクトルDB: {engine.collection.count()}件")
    print()
    print("使い方:")
    print("  症状を入力 → 候補疾患と推薦検査を表示")
    print("  検査結果は1行1件で入力（例: γ-GTP正常値、血液培養: 黄色ブドウ球菌陽性）")
    print("  「quit」で終了、「reset」で新しい患者")
    print()

    symptoms_text = ""
    result_lines = []

    while True:
        if not symptoms_text:
            print("症状を入力してください（複数行の場合は空行で確定）:")
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    if lines:
                        break
                    continue
                if line.strip().lower() == "quit":
                    print("終了します。")
                    return
                lines.append(line)
            symptoms_text = "\n".join(lines)
        else:
            print("検査結果を入力（例: γ-GTP正常値、血液培養: 黄色ブドウ球菌陽性）")
            print("  「reset」で新しい患者、「quit」で終了:")
            result_input = input("> ").strip()

            if result_input.lower() == "quit":
                print("終了します。")
                return
            if result_input.lower() == "reset":
                symptoms_text = ""
                result_lines = []
                print("\n--- 新しい患者 ---\n")
                continue
            if not result_input:
                continue

            result_lines.append(result_input)

        # Step 1: 症状で疾患検索（毎回症状から検索、検査結果はsim_matrixで更新）
        print("\nベクトル検索中...")
        candidates = engine.search_diseases(symptoms_text)
        candidates = engine.compute_priors(candidates)

        # Step 2: 検査結果で重み更新（Option 3、高速モード）
        if result_lines:
            candidates = engine.update_from_results(
                candidates, result_lines,
                symptoms=symptoms_text, mode="fast",
            )

        # 表示
        print()
        print(format_candidates(candidates))

        # Step 3: novelty + 検査ランキング
        print("情報利得計算中...")
        full_text = symptoms_text
        if result_lines:
            full_text += '\n' + '\n'.join(result_lines)
        novelty = engine.compute_novelty(full_text)
        ranked_tests = engine.rank_tests(candidates, novelty=novelty)
        confirm_tests = engine.rank_tests_confirm(candidates, novelty=novelty)

        print(format_tests(ranked_tests, top_n=TOP_K_TESTS))
        print(format_confirm(confirm_tests, top_n=TOP_K_TESTS))


if __name__ == "__main__":
    main()
