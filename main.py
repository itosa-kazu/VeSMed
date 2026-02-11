"""
VeSMed - 対話型CLI
患者情報 → 候補疾患 → 検査推薦 → 検査結果で更新 → ループ
"""

from engine import VeSMedEngine


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
    """推薦検査を表示用にフォーマット"""
    lines = []
    lines.append("【推薦検査（効用=分散×exp(質) 順）】")
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
    print("  患者情報を入力 → 候補疾患と推薦検査を表示")
    print("  検査結果は「検査名: 結果」の形式で入力")
    print("  「quit」で終了、「reset」で新しい患者")
    print()

    patient_text = ""
    done_tests = []

    while True:
        if not patient_text:
            print("患者情報を入力してください（複数行の場合は空行で確定）:")
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
            patient_text = "\n".join(lines)
        else:
            print("検査結果を入力（例: 12誘導心電図: V1-V4でST上昇）")
            print("  「reset」で新しい患者、「quit」で終了:")
            result_input = input("> ").strip()

            if result_input.lower() == "quit":
                print("終了します。")
                return
            if result_input.lower() == "reset":
                patient_text = ""
                done_tests = []
                print("\n--- 新しい患者 ---\n")
                continue
            if not result_input:
                continue

            # 検査結果を解析
            if ": " in result_input:
                test_name, result = result_input.split(": ", 1)
            elif ":" in result_input:
                test_name, result = result_input.split(":", 1)
            else:
                test_name = result_input
                result = "（詳細不明）"

            test_name = test_name.strip()
            result = result.strip()
            done_tests.append(test_name)
            patient_text = engine.update_patient_text(patient_text, test_name, result)

        # Step 1+2: ベクトル検索（生テキスト直接embed、LLM rewrite不要）
        print("\nベクトル検索中...")
        candidates = engine.search_diseases(patient_text)

        # Step 3: 先験確率計算
        candidates = engine.compute_priors(candidates)

        # 表示
        print()
        print(format_candidates(candidates))

        # Step 4: novelty計算 + 検査ランキング
        print("情報利得計算中...")
        novelty = engine.compute_novelty(patient_text)
        ranked_tests = engine.rank_tests(candidates, novelty=novelty)

        print(format_tests(ranked_tests, top_n=TOP_K_TESTS))


# config からインポート
from config import TOP_K_TESTS

if __name__ == "__main__":
    main()
