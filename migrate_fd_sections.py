"""
VeSMed - findings_description セクション分離マイグレーション

既存の findings_description（1つの巨大テキスト）を
fd_* 個別フィールドに分割格納する。

- chunk_into_domains() でヘッダーベースの分割を実行
- fd_background, fd_typical, ... fd_pathophysiology に格納
- _gen_meta に legacy_migration として記録
- findings_description は fd_* から再組み立てして後方互換維持

Usage:
    python migrate_fd_sections.py              # 実行
    python migrate_fd_sections.py --dry-run    # 統計表示のみ
"""

import json
import os
import sys
import time
from datetime import date

from index import chunk_into_domains

DISEASES_JSONL = os.path.join(os.path.dirname(__file__), "data", "diseases.jsonl")

# ドメイン名 → fd_* フィールド名
DOMAIN_TO_FD = {
    "background":      "fd_background",
    "typical":         "fd_typical",
    "atypical":        "fd_atypical",
    "physical":        "fd_physical",
    "tests":           "fd_tests",
    "differential":    "fd_differential",
    "pathophysiology": "fd_pathophysiology",
}

FD_KEYS_ORDER = [
    "fd_background", "fd_typical", "fd_atypical",
    "fd_physical", "fd_tests", "fd_differential", "fd_pathophysiology",
]

FD_TITLES = {
    "fd_background":      "【好発背景・リスク因子・誘因】",
    "fd_typical":         "【典型来院像】",
    "fd_atypical":        "【非典型来院像・ピットフォール】",
    "fd_physical":        "【バイタルサイン・身体所見】",
    "fd_tests":           "【検査所見パターン】",
    "fd_differential":    "【鑑別キー】",
    "fd_pathophysiology": "【病態生理・発症メカニズム】",
}


def assemble_findings_description(record):
    """fd_* フィールドから findings_description を再組み立て"""
    parts = []
    for key in FD_KEYS_ORDER:
        text = record.get(key, "")
        if text:
            title = FD_TITLES[key]
            parts.append(f"{title}\n{text}")
    return "\n\n".join(parts)


def read_diseases():
    diseases = []
    with open(DISEASES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                diseases.append(json.loads(line))
    return diseases


def write_diseases(diseases):
    tmp_path = DISEASES_JSONL + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for d in diseases:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(5):
        try:
            os.replace(tmp_path, DISEASES_JSONL)
            return
        except PermissionError:
            if attempt < 4:
                time.sleep(1)
            else:
                raise


def migrate():
    dry_run = "--dry-run" in sys.argv
    diseases = read_diseases()
    today = date.today().isoformat()

    # 対象: findings_description あり & fd_background 未設定
    targets = []
    already_migrated = 0
    no_fd = 0
    for i, d in enumerate(diseases):
        if d.get("fd_background"):
            already_migrated += 1
            continue
        fd = d.get("findings_description", "")
        if not fd or len(fd) < 100:
            no_fd += 1
            continue
        targets.append((i, d))

    print(f"全疾患: {len(diseases)}件")
    print(f"  マイグレーション済み: {already_migrated}件")
    print(f"  findings_descriptionなし: {no_fd}件")
    print(f"  対象: {len(targets)}件")

    if not targets:
        print("マイグレーション対象はありません。")
        return

    # 分割統計
    domain_counts = {k: 0 for k in DOMAIN_TO_FD}
    domain_chars = {k: 0 for k in DOMAIN_TO_FD}
    chunk_fail = 0  # backgroundしか検出できなかったケース

    for i, (orig_idx, d) in enumerate(targets):
        fd = d["findings_description"]
        sections = chunk_into_domains(fd)

        if len(sections) <= 1:
            chunk_fail += 1
            if dry_run and chunk_fail <= 5:
                print(f"  警告: {d['disease_name']} - 分割失敗（{len(sections)}セクション）")

        for domain, text in sections.items():
            domain_counts[domain] += 1
            domain_chars[domain] += len(text)

        if not dry_run:
            # fd_* フィールドに書き込み
            for domain, fd_key in DOMAIN_TO_FD.items():
                if domain in sections:
                    d[fd_key] = sections[domain]

            # _gen_meta 記録
            meta = d.get("_gen_meta", {})
            for domain, fd_key in DOMAIN_TO_FD.items():
                if domain in sections:
                    meta[fd_key] = {
                        "model": "legacy_migration",
                        "date": today,
                        "chars": len(sections[domain]),
                    }
            d["_gen_meta"] = meta

            # findings_description を fd_* から再組み立て
            d["findings_description"] = assemble_findings_description(d)

            diseases[orig_idx] = d

    # 統計表示
    print(f"\n分割統計:")
    for domain in DOMAIN_TO_FD:
        fd_key = DOMAIN_TO_FD[domain]
        cnt = domain_counts[domain]
        avg = domain_chars[domain] // cnt if cnt > 0 else 0
        print(f"  {fd_key:<25s}: {cnt:>4d}件 (平均 {avg:>5d}字)")
    if chunk_fail > 0:
        print(f"\n  分割不完全: {chunk_fail}件（backgroundのみ）")

    if dry_run:
        print(f"\n--dry-run: 書き込みは行いません。")
        return

    # 書き出し
    write_diseases(diseases)
    print(f"\nマイグレーション完了: {len(targets)}件をfd_*フィールドに分割")
    print(f"次のステップ: python index.py  # ChromaDB再構築")


if __name__ == "__main__":
    migrate()
