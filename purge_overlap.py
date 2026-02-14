"""
マスタパージ: tests.jsonlからHPEと重複する21件を除外。
1回実行用スクリプト。
"""
import json
import os
import shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TESTS_JSONL = os.path.join(DATA_DIR, "tests.jsonl")

# 除外対象: 問診19件 + バイタル + 意識レベル + 身体診察20件 = 41件
# 全てHPE(Part D)と重複。検査マスタにあるべきではない。
PURGE_NAMES = {
    # --- 問診 (19件) ---
    "問診：アレルギー歴",
    "問診：予防接種歴",
    "問診：体重変化・食欲",
    "問診：動物・ペット接触歴",
    "問診：喫煙歴",
    "問診：外傷歴・最近の処置",
    "問診：家族歴",
    "問診：性行為歴・STI歴",
    "問診：排尿・排便の変化",
    "問診：既往歴・手術歴",
    "問診：月経歴・産科歴",
    "問診：渡航歴",
    "問診：睡眠・精神状態",
    "問診：職業歴・環境曝露",
    "問診：薬剤歴",
    "問診：輸血歴・臓器移植歴",
    "問診：違法薬物・嗜好品",
    "問診：食事歴",
    "問診：飲酒歴",
    # --- バイタル・意識 (2件) ---
    "バイタルサイン測定",
    "意識レベルの評価",
    # --- 身体診察 (20件) ---
    "全身状態の観察",
    "頭頸部の診察",
    "口腔・咽頭の診察",
    "眼の診察",
    "心音聴診",
    "肺音聴診",
    "胸部の打診",
    "腹部の視診・聴診",
    "腹部触診",
    "直腸診",
    "皮膚の診察",
    "四肢末梢の診察",
    "関節の診察",
    "下肢の血管診察",
    "髄膜刺激徴候の検査",
    "脳神経の検査",
    "運動・感覚系の検査",
    "小脳機能・歩行の検査",
    "脊椎・背部の診察",
    "CVA叩打痛の検査",
}

# キャッシュファイル（除外後に再計算が必要）
CACHE_FILES = [
    "sim_matrix.npz",
    "test_quality.json",
    "test_name_embs.npz",
    "risk_embs.npz",
    "invasive_scores.json",
]


def main():
    # 1. バックアップ
    backup = TESTS_JSONL + ".bak_pre_purge"
    shutil.copy2(TESTS_JSONL, backup)
    print(f"バックアップ: {backup}")

    # 2. 読み込み・フィルタ
    kept = []
    purged = []
    with open(TESTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                name = t["test_name"]
                if name in PURGE_NAMES:
                    purged.append(name)
                else:
                    kept.append(line)
            except (json.JSONDecodeError, KeyError):
                kept.append(line)

    # 3. 書き出し
    with open(TESTS_JSONL, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")

    print(f"除外: {len(purged)}件")
    for name in sorted(purged):
        print(f"  - {name}")
    print(f"残存: {len(kept)}件")

    # 4. 未マッチの除外対象を警告
    unmatched = PURGE_NAMES - set(purged)
    if unmatched:
        print(f"\n警告: 以下の除外対象がtests.jsonlに見つかりませんでした:")
        for name in sorted(unmatched):
            print(f"  - {name}")

    # 5. キャッシュ削除
    deleted = []
    for fname in CACHE_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            deleted.append(fname)
    if deleted:
        print(f"\nキャッシュ削除: {', '.join(deleted)}")
    else:
        print("\n削除対象のキャッシュなし")

    print("\n完了。次回engine起動時にsim_matrix等が自動再計算されます。")


if __name__ == "__main__":
    main()
