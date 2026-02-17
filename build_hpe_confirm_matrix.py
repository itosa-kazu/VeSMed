"""
HPE Confirm Matrix: LLM判定による逆引きインデックス構築

各疾患の症状・所見フィールド(core_symptoms, core_signs等)を279 HPE項目と照合し、
該当するHPE項目をLLMで特定。逆引きインデックス（HPE項目 → 疾患リスト）を構築し、
sim_matrix_hpe_confirm.npz を生成。

API: gemini-3-pro-preview-c (RPM < 10)
"""

import json
import os
import sys
import time
import re
import hashlib
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
    DISEASES_JSONL, DATA_DIR,
)

HPE_JSONL = os.path.join(DATA_DIR, "hpe_items.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "sim_matrix_hpe_confirm.npz")
REVERSE_INDEX_FILE = os.path.join(DATA_DIR, "hpe_confirm_reverse_index.json")
PROGRESS_FILE = os.path.join(DATA_DIR, "hpe_confirm_progress.json")

# LLM API (rate-limited)
LLM_API_KEY = "sk-hI5iGw1n6EuuCydhi70UNTtENTUQTFknpbeGXxadyZhhxkcR"
LLM_BASE_URL = "https://new.12ai.org/v1"
LLM_MODEL = "gemini-3-pro-preview-c"
RPM_LIMIT = 8
BATCH_SIZE = 5


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_hpe_list_text(hpe_items):
    """HPE項目の番号付きリストを構築。"""
    lines = []
    for i, item in enumerate(hpe_items):
        cat = item.get("category", "")
        subcat = item.get("subcategory", "")
        name = item["item_name"]
        lines.append(f"{i}: [{cat}:{subcat}] {name}")
    return "\n".join(lines)


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
- 確実に該当するもののみ選択

出力形式（厳守）:
疾患名: 番号,番号,番号

例:
肺炎球菌性肺炎: 1,9,123,127,129
急性虫垂炎: 5,33,45,67

推論過程・説明・箇条書きは一切不要。上記形式のみ出力せよ。"""


def parse_response(text, disease_names, n_hpe):
    """LLM応答を {disease_name: [hpe_indices]} にパース。"""
    result = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        # コロンの前後で分割（全角コロンも対応）
        line = line.replace("：", ":")
        if ":" not in line:
            continue
        colon_idx = line.index(":")
        dname_raw = line[:colon_idx].strip()
        # 箇条書きプレフィクス除去: ■ ・ - * 数字. など
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


def load_disease_embeddings():
    """disease_embs.npzからMEAN集約済み疾患embeddingを読み込む。"""
    embs_file = os.path.join(DATA_DIR, "disease_embs.npz")
    if not os.path.exists(embs_file):
        raise FileNotFoundError(f"{embs_file} が見つかりません。index.pyを先に実行してください。")
    data = np.load(embs_file, allow_pickle=True)
    return list(data["disease_names"]), data["disease_embs_normed"].astype(np.float32)


def batch_embed(texts, batch_size=50, max_workers=10):
    """バッチ並行embedding。"""
    client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    batches = []
    for i in range(0, len(texts), batch_size):
        batches.append((i, texts[i:i + batch_size]))

    all_embs = [None] * len(texts)

    def _embed_one(batch_info):
        start_idx, batch = batch_info
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                return start_idx, [item.embedding for item in resp.data]
            except Exception as e:
                print(f"  [embed] batch(offset={start_idx}) 失敗(試行{attempt+1}): {e}")
                time.sleep(2 ** attempt)
        return start_idx, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_embed_one, b) for b in batches]
        done = 0
        for future in as_completed(futures):
            start_idx, result = future.result()
            if result is None:
                print(f"  [embed] batch(offset={start_idx}) 完全失敗")
                return None
            for j, emb in enumerate(result):
                all_embs[start_idx + j] = emb
            done += 1
            if done % 3 == 0 or done == len(batches):
                print(f"  Embedding batch {done}/{len(batches)} done")

    return np.array(all_embs, dtype=np.float32)


def main():
    print("=" * 80)
    print("HPE Confirm Matrix: LLM判定による逆引きインデックス構築")
    print("=" * 80)

    # ─── データ読み込み ───
    hpe_items = load_jsonl(HPE_JSONL)
    diseases = load_jsonl(DISEASES_JSONL)
    n_hpe = len(hpe_items)
    hpe_names = [item["item_name"] for item in hpe_items]

    print(f"\nHPE項目: {n_hpe}件")
    print(f"疾患マスタ: {len(diseases)}件")

    # HPE項目リストテキスト（全バッチ共通）
    hpe_list_text = build_hpe_list_text(hpe_items)

    # ─── 進捗復元 ───
    all_mappings = {}  # disease_name → [hpe_indices]
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            all_mappings = json.load(f)
        print(f"進捗復元: {len(all_mappings)}疾患処理済み")

    # 未処理の疾患
    remaining = [d for d in diseases if d["disease_name"] not in all_mappings]
    print(f"未処理: {len(remaining)}疾患")

    if remaining:
        # ─── LLM判定 ───
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
                f"【HPE項目リスト】\n{hpe_list_text}\n\n"
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
                    # thinking model対策: 最後のブロックのみ使用
                    # (推論過程が含まれる場合、最終出力だけを抽出)
                    if "\n\n" in text:
                        blocks = text.split("\n\n")
                        # 「疾患名: 番号」形式の行が最も多いブロックを選択
                        best_block = text
                        best_count = 0
                        for block in blocks:
                            count = len(re.findall(r'^.+:\s*[\d,\s]+', block, re.MULTILINE))
                            if count > best_count:
                                best_count = count
                                best_block = block
                        if best_count > 0:
                            text = best_block
                    parsed = parse_response(text, batch_names, n_hpe)

                    for dname in batch_names:
                        if dname in parsed:
                            all_mappings[dname] = parsed[dname]
                            success += 1
                        else:
                            # fuzzy match
                            matched = False
                            for key in parsed:
                                if dname in key or key in dname:
                                    all_mappings[dname] = parsed[key]
                                    success += 1
                                    matched = True
                                    break
                            if not matched:
                                all_mappings[dname] = []
                                failed += 1

                    break  # success

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
                # 途中保存
                with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                    json.dump(all_mappings, f, ensure_ascii=False)

        total_time = time.time() - start_time
        print(f"\nLLM判定完了: {success}成功, {failed}失敗 ({total_time:.0f}秒)")

    # ─── 逆引きインデックス構築 ───
    print("\n--- 逆引きインデックス構築 ---")
    reverse_index = defaultdict(set)  # hpe_item_name → set of disease_names
    for dname, indices in all_mappings.items():
        for idx in indices:
            if 0 <= idx < n_hpe:
                reverse_index[hpe_names[idx]].add(dname)

    # 統計
    matched_hpe = [h for h in hpe_names if h in reverse_index]
    disease_counts = [len(reverse_index[h]) for h in matched_hpe]
    print(f"  逆引きマッチ: {len(matched_hpe)}/{n_hpe} HPE項目")
    if disease_counts:
        print(f"  対象疾患数: mean={np.mean(disease_counts):.1f}, "
              f"median={np.median(disease_counts):.0f}, "
              f"min={min(disease_counts)}, max={max(disease_counts)}")

    # Top/Bottom
    sorted_by_count = sorted(
        [(h, len(reverse_index[h])) for h in matched_hpe],
        key=lambda x: x[1], reverse=True
    )
    print(f"\n  疾患数Top10:")
    for hname, cnt in sorted_by_count[:10]:
        print(f"    {hname}: {cnt}疾患")
    print(f"  疾患数Bottom5:")
    for hname, cnt in sorted_by_count[-5:]:
        print(f"    {hname}: {cnt}疾患")

    # ─── 逆引きインデックス保存 ───
    reverse_for_save = {
        h: sorted(list(reverse_index[h])) for h in hpe_names if h in reverse_index
    }
    with open(REVERSE_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(reverse_for_save, f, ensure_ascii=False, indent=2)
    print(f"\n逆引きインデックス保存: {REVERSE_INDEX_FILE}")

    # ─── 確認用hypothesis生成 ───
    print("\n--- 確認用hypothesis生成 ---")
    confirm_texts = []
    for hname in hpe_names:
        if hname in reverse_index and len(reverse_index[hname]) > 0:
            dlist = sorted(reverse_index[hname])
            text = '、'.join(dlist)
        else:
            text = hname  # フォールバック: 項目名そのまま
        confirm_texts.append(text)

    lengths = [len(t) for t in confirm_texts]
    print(f"  hypothesis_confirm生成完了: {len(confirm_texts)}件")
    print(f"  文字数: mean={np.mean(lengths):.0f}, "
          f"median={np.median(lengths):.0f}, "
          f"min={min(lengths)}, max={max(lengths)}")

    # ─── 疾患embedding読み込み ───
    print("\n--- 疾患embedding読み込み ---")
    disease_names, disease_embs_normed = load_disease_embeddings()
    print(f"  {len(disease_names)}疾患 × {disease_embs_normed.shape[1]}次元")

    # ─── ハッシュチェック ───
    h_hash = hashlib.md5("||".join(confirm_texts).encode()).hexdigest()[:12]
    if os.path.exists(OUTPUT_FILE):
        data = np.load(OUTPUT_FILE, allow_pickle=True)
        cached_hash = str(data.get("hypothesis_hash", ""))
        if cached_hash == h_hash:
            print(f"\n既存キャッシュと一致 → スキップ")
            return
        print(f"  キャッシュ不一致 → 再計算")

    # ─── Embedding ───
    print(f"\n--- Embedding ({len(confirm_texts)}件) ---")
    confirm_embs = batch_embed(confirm_texts)
    if confirm_embs is None:
        print("ERROR: Embedding失敗")
        return

    h_norms = np.linalg.norm(confirm_embs, axis=1, keepdims=True)
    h_norms[h_norms == 0] = 1.0
    confirm_embs_normed = confirm_embs / h_norms

    # ─── sim_matrix_hpe_confirm 計算 ───
    print("\n--- sim_matrix_hpe_confirm 計算 ---")
    sim_matrix = disease_embs_normed @ confirm_embs_normed.T
    print(f"  shape: {sim_matrix.shape}")
    print(f"  mean: {sim_matrix.mean():.4f}")
    print(f"  std:  {sim_matrix.std():.4f}")

    col_vars = np.var(sim_matrix, axis=0)
    print(f"  列分散: mean={col_vars.mean():.6f}, "
          f"median={np.median(col_vars):.6f}")

    # ─── 保存 ───
    np.savez(
        OUTPUT_FILE,
        sim_matrix=sim_matrix,
        disease_names=np.array(disease_names, dtype=object),
        hpe_names=np.array(hpe_names, dtype=object),
        hypothesis_hash=np.array(h_hash),
    )
    print(f"\n保存完了: {OUTPUT_FILE}")
    print(f"  {sim_matrix.shape[0]}疾患 × {sim_matrix.shape[1]} HPE項目")

    # 進捗ファイル削除
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("進捗ファイル削除")


if __name__ == "__main__":
    main()
