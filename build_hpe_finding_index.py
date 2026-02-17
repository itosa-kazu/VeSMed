"""
Type F HPE逆引きインデックス構築（疾患→所見方向）

各疾患のcore_symptoms, core_signs, rare_but_specific等からType F HPE項目へマッチ。
旧build_hpe_confirm_matrix.pyの進捗データ（325/527疾患）を再利用。

API: gemini-3-pro-preview-c via new.12ai.org (RPM < 10)
"""

import json
import os
import re
import sys
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_DIR, DISEASES_JSONL

HPE_JSONL = os.path.join(DATA_DIR, "hpe_items.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "hpe_finding_reverse_index.json")
PROGRESS_FILE = os.path.join(DATA_DIR, "hpe_finding_progress.json")
OLD_PROGRESS_FILE = os.path.join(DATA_DIR, "hpe_confirm_progress.json")

LLM_API_KEY = "sk-hI5iGw1n6EuuCydhi70UNTtENTUQTFknpbeGXxadyZhhxkcR"
LLM_BASE_URL = "https://new.12ai.org/v1"
LLM_MODEL = "gemini-3-pro-preview-c"
RPM_LIMIT = 8
BATCH_SIZE = 5

TYPE_R_SUBCATS = {"薬剤歴", "嗜好/社会歴", "既往歴", "家族歴"}


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_type_f_list(hpe_items):
    """Type F項目のみの番号付きリストを構築。新しい連番を使用。"""
    lines = []
    original_to_new = {}  # original_idx → new_idx
    new_to_original = {}  # new_idx → original_idx
    new_idx = 0
    for orig_idx, item in enumerate(hpe_items):
        if item["subcategory"] not in TYPE_R_SUBCATS:
            cat = item.get("category", "")
            subcat = item.get("subcategory", "")
            name = item["item_name"]
            lines.append(f"{new_idx}: [{cat}:{subcat}] {name}")
            original_to_new[orig_idx] = new_idx
            new_to_original[new_idx] = orig_idx
            new_idx += 1
    return "\n".join(lines), original_to_new, new_to_original


def build_disease_block(disease):
    """疾患の症状・所見情報をテキストブロックに変換。"""
    dname = disease.get("disease_name", "")
    parts = [f"■ {dname}"]
    for field, label in [
        ("core_symptoms", "中核症状"),
        ("common_symptoms", "随伴症状"),
        ("core_signs", "身体所見"),
        ("rare_but_specific", "特異的所見"),
    ]:
        items = disease.get(field, [])
        if items and isinstance(items, list):
            parts.append(f"  {label}: {', '.join(str(x) for x in items)}")
        elif items:
            parts.append(f"  {label}: {items}")
    return "\n".join(parts)


SYSTEM_PROMPT = """あなたは臨床医です。各疾患について、HPE項目リストから該当番号を選択してください。

ルール:
- core_symptoms/core_signsに対応する項目は必ず含める
- rare_but_specificに該当する項目も含める
- common_symptomsに対応する項目も含める
- 確実に該当するもののみ選択（推測で追加しない）

出力形式（厳守）:
疾患名: 番号,番号,番号

例:
肺炎球菌性肺炎: 1,9,123,127,129
急性虫垂炎: 5,33,45,67

推論過程・説明・箇条書きは一切不要。上記形式のみ出力せよ。"""


def parse_response(text, disease_names, n_hpe):
    """LLM応答を {disease_name: [hpe_indices]} にパース。"""
    # thinking model: 最も「疾患名: 番号」形式の行が多いブロックを選択
    if "\n\n" in text:
        blocks = text.split("\n\n")
        best_block = text
        best_count = 0
        for block in blocks:
            count = len(re.findall(r'^.+:\s*[\d,\s]+', block, re.MULTILINE))
            if count > best_count:
                best_count = count
                best_block = block
        if best_count > 0:
            text = best_block

    result = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        line = line.replace("：", ":")
        if ":" not in line:
            continue
        colon_idx = line.index(":")
        dname_raw = line[:colon_idx].strip()
        dname_raw = re.sub(r'^[\s\-\*・■●•\d.]+', '', dname_raw).strip()
        nums_str = line[colon_idx + 1:].strip()

        if nums_str == "なし" or not nums_str:
            result[dname_raw] = []
            continue

        nums = []
        for token in re.findall(r'\d+', nums_str):
            idx = int(token)
            if 0 <= idx < n_hpe:
                nums.append(idx)
        result[dname_raw] = nums

    return result


def migrate_old_progress(hpe_items, old_progress):
    """旧build_hpe_confirm_matrix.pyの進捗データからType Fのみ抽出。"""
    type_f_indices = set(
        i for i, item in enumerate(hpe_items)
        if item["subcategory"] not in TYPE_R_SUBCATS
    )
    migrated = {}  # {disease_name: [original_hpe_indices]}
    for dname, indices in old_progress.items():
        f_indices = [i for i in indices if i in type_f_indices]
        migrated[dname] = f_indices
    return migrated


def main():
    from openai import OpenAI

    print("=" * 80)
    print("Type F HPE逆引きインデックス構築（疾患→所見方向）")
    print("=" * 80)

    # データ読み込み
    hpe_items = load_jsonl(HPE_JSONL)
    diseases = load_jsonl(DISEASES_JSONL)
    n_hpe_all = len(hpe_items)
    hpe_names = [item["item_name"] for item in hpe_items]

    # Type F項目リスト（新番号体系）
    hpe_list_text, orig_to_new, new_to_orig = build_type_f_list(hpe_items)
    n_type_f = len(new_to_orig)
    type_f_orig_indices = set(orig_to_new.keys())

    print(f"\nHPE項目: {n_hpe_all}件 (Type F: {n_type_f}件)")
    print(f"疾患マスタ: {len(diseases)}件")

    # ─── 進捗復元 ───
    # all_mappings: {disease_name: [original_hpe_indices]}
    all_mappings = {}

    # 新進捗ファイル
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            all_mappings = json.load(f)
        print(f"進捗復元: {len(all_mappings)}疾患処理済み")

    # 旧進捗からの移行
    if not all_mappings and os.path.exists(OLD_PROGRESS_FILE):
        with open(OLD_PROGRESS_FILE, "r", encoding="utf-8") as f:
            old_prog = json.load(f)
        all_mappings = migrate_old_progress(hpe_items, old_prog)
        print(f"旧進捗移行: {len(all_mappings)}疾患 (Type Fのみ抽出)")

        # 移行データ保存
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_mappings, f, ensure_ascii=False)

    # 未処理
    remaining = [d for d in diseases if d["disease_name"] not in all_mappings]
    print(f"未処理: {len(remaining)}疾患")

    if remaining:
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        min_interval = 60.0 / RPM_LIMIT
        n_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n--- LLM判定開始 ({len(remaining)}疾患, {n_batches}バッチ, RPM<{RPM_LIMIT}) ---")
        print(f"    推定時間: {n_batches * min_interval / 60:.1f}分")

        start_time = time.time()
        last_call_time = 0
        success = 0
        failed = 0

        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(remaining))
            batch = remaining[start:end]
            batch_names = [d["disease_name"] for d in batch]

            disease_blocks = "\n\n".join(build_disease_block(d) for d in batch)

            user_content = (
                f"以下の{len(batch)}疾患について、HPE項目リストから該当番号を選んでください。\n\n"
                f"【HPE項目リスト（{n_type_f}件）】\n{hpe_list_text}\n\n"
                f"【疾患】\n{disease_blocks}"
            )

            # Rate limiting
            now = time.time()
            elapsed = now - last_call_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            for attempt in range(3):
                try:
                    last_call_time = time.time()
                    response = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=0.1,
                        max_tokens=8192,
                    )
                    text = response.choices[0].message.content
                    parsed = parse_response(text, batch_names, n_type_f)

                    for dname in batch_names:
                        new_indices = None
                        if dname in parsed:
                            new_indices = parsed[dname]
                        else:
                            # fuzzy match
                            for key in parsed:
                                if dname in key or key in dname:
                                    new_indices = parsed[key]
                                    break

                        if new_indices is not None:
                            # new_idx → original_idx に変換
                            orig_indices = [new_to_orig[ni] for ni in new_indices
                                           if ni in new_to_orig]
                            all_mappings[dname] = orig_indices
                            success += 1
                        else:
                            all_mappings[dname] = []
                            failed += 1

                    break

                except Exception as e:
                    if attempt < 2:
                        print(f"  バッチ{batch_idx+1} retry {attempt+1}: {e}")
                        time.sleep(min_interval)
                    else:
                        print(f"  バッチ{batch_idx+1} FAILED: {e}")
                        for dname in batch_names:
                            all_mappings[dname] = []
                            failed += 1

            # 進捗表示 & 保存
            if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
                elapsed_total = time.time() - start_time
                print(f"  バッチ {batch_idx+1}/{n_batches} 完了 "
                      f"({success}成功, {failed}失敗, {elapsed_total:.0f}秒)")
                with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                    json.dump(all_mappings, f, ensure_ascii=False)

        total_time = time.time() - start_time
        print(f"\nLLM判定完了: {success}成功, {failed}失敗 ({total_time:.0f}秒)")

    # ─── 逆引きインデックス構築 ───
    print("\n--- 逆引きインデックス構築 ---")
    reverse_index = defaultdict(set)
    for dname, indices in all_mappings.items():
        for idx in indices:
            if 0 <= idx < n_hpe_all:
                reverse_index[hpe_names[idx]].add(dname)

    # Type F項目のみフィルタ
    type_f_names = [hpe_names[i] for i in sorted(type_f_orig_indices)]

    # 統計
    matched_f = [h for h in type_f_names if h in reverse_index]
    disease_counts = [len(reverse_index[h]) for h in matched_f]
    print(f"  逆引きマッチ: {len(matched_f)}/{n_type_f} Type F項目")
    if disease_counts:
        print(f"  対象疾患数: mean={np.mean(disease_counts):.1f}, "
              f"median={np.median(disease_counts):.0f}, "
              f"min={min(disease_counts)}, max={max(disease_counts)}")

    # Top/Bottom
    sorted_by_count = sorted(
        [(h, len(reverse_index[h])) for h in matched_f],
        key=lambda x: x[1], reverse=True
    )
    print(f"\n  疾患数Top10:")
    for hname, cnt in sorted_by_count[:10]:
        print(f"    {hname}: {cnt}疾患")
    print(f"  疾患数Bottom5:")
    for hname, cnt in sorted_by_count[-5:]:
        print(f"    {hname}: {cnt}疾患")

    # 未カバー
    uncovered = [h for h in type_f_names if h not in reverse_index]
    if uncovered:
        print(f"\n  未カバー ({len(uncovered)}件): {uncovered}")

    # ─── 保存 ───
    reverse_for_save = {
        h: sorted(list(reverse_index[h])) for h in type_f_names if h in reverse_index
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(reverse_for_save, f, ensure_ascii=False, indent=2)
    print(f"\n保存完了: {OUTPUT_FILE}")

    # 進捗ファイル削除
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)


if __name__ == "__main__":
    main()
