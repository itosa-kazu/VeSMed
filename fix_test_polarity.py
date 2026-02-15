"""
検査 極性分割スクリプト

「増加または減少」を含む4検査を、増加版/減少版の2項目に分割する。
hypothesis_textのみ変更（findings_descriptionは共有）。

対象:
  - 網赤血球数 → 網赤血球数（増加）/ 網赤血球数（減少）
  - 白血球数 → 白血球数（増加）/ 白血球数（減少）
  - 血小板数 → 血小板数（増加）/ 血小板数（減少）
  - 赤血球数 → 赤血球数（増加）/ 赤血球数（減少）

Usage:
    python fix_test_polarity.py              # 実行
    python fix_test_polarity.py --dry-run    # 変更内容を表示のみ
"""

import json
import sys

TESTS_JSONL = "data/tests.jsonl"

# test_name → (増加版hypothesis, 減少版hypothesis) のマッピング
# 疾患名を含まず、メカニズム・病態生理で記述
SPLITS = {
    "網赤血球数": (
        "網赤血球数の増加を認めた。末梢での赤血球破壊亢進または出血に対する骨髄の代償性造血亢進を示唆する",
        "網赤血球数の減少を認めた。骨髄での赤血球産生能低下、造血幹細胞障害を示唆する",
    ),
    "白血球数": (
        "白血球数の増加を認めた。細菌感染による骨髄動員亢進、炎症性サイトカインによる顆粒球造血促進、腫瘍性増殖を示唆する",
        "白血球数の減少を認めた。骨髄造血抑制、薬剤性骨髄障害、ウイルス感染による一過性抑制を示唆する",
    ),
    "血小板数": (
        "血小板数の増加を認めた。反応性血小板増多（炎症、鉄欠乏、脾摘後）、骨髄での巨核球系増殖を示唆する",
        "血小板数の減少を認めた。免疫性血小板破壊亢進、消費性凝固障害、骨髄造血不全を示唆する",
    ),
    "赤血球数": (
        "赤血球数の増加を認めた。骨髄での赤血球造血亢進、EPO過剰産生、相対的増加（脱水・血液濃縮）を示唆する",
        "赤血球数の減少を認めた。鉄欠乏・慢性炎症・骨髄障害による造血不全、出血・溶血による赤血球喪失を示唆する",
    ),
}


def main():
    dry_run = "--dry-run" in sys.argv

    tests = []
    with open(TESTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tests.append(json.loads(line.strip()))

    print(f"全検査: {len(tests)}件")

    new_tests = []
    split_count = 0

    for t in tests:
        name = t["test_name"]
        if name in SPLITS:
            hyp_inc, hyp_dec = SPLITS[name]

            # 増加版
            t_inc = dict(t)
            t_inc["test_name"] = f"{name}（増加）"
            t_inc["hypothesis_text"] = hyp_inc

            # 減少版
            t_dec = dict(t)
            t_dec["test_name"] = f"{name}（減少）"
            t_dec["hypothesis_text"] = hyp_dec

            new_tests.append(t_inc)
            new_tests.append(t_dec)
            split_count += 1

            if dry_run:
                print(f"\n[{name}] → 2項目に分割")
                print(f"  ↑ {t_inc['test_name']}")
                print(f"    {hyp_inc}")
                print(f"  ↓ {t_dec['test_name']}")
                print(f"    {hyp_dec}")
        else:
            new_tests.append(t)

    print(f"\n分割: {split_count}件 → {split_count * 2}件")
    print(f"新合計: {len(new_tests)}件 (旧: {len(tests)}件)")

    if not dry_run and split_count > 0:
        with open(TESTS_JSONL, "w", encoding="utf-8") as f:
            for t in new_tests:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"書き込み完了: {TESTS_JSONL}")
        print("\n次のステップ:")
        print("  del data\\sim_matrix.npz  # キャッシュ削除")
    elif dry_run:
        print("\n--dry-run: 書き込みは行いません")


if __name__ == "__main__":
    main()
