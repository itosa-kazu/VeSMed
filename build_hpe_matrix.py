"""
HPE sim_matrix統合構築: screen + confirm

Screen行列: hypothesis embeddingから構築（engine.pyの自動構築と同等だが明示的再構築）
Confirm行列: 逆引きインデックス（Type R + Type F）から疾患名リストをembed

前提:
  - data/hpe_risk_reverse_index.json (build_hpe_risk_index.pyで生成)
  - data/hpe_finding_reverse_index.json (build_hpe_finding_index.pyで生成)
"""

import json
import hashlib
import os
import sys
import time
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
    DISEASES_JSONL, DATA_DIR,
)

HPE_JSONL = os.path.join(DATA_DIR, "hpe_items.jsonl")
RISK_INDEX = os.path.join(DATA_DIR, "hpe_risk_reverse_index.json")
FINDING_INDEX = os.path.join(DATA_DIR, "hpe_finding_reverse_index.json")
SCREEN_OUTPUT = os.path.join(DATA_DIR, "sim_matrix_hpe.npz")
CONFIRM_OUTPUT = os.path.join(DATA_DIR, "sim_matrix_hpe_confirm.npz")

TYPE_R_SUBCATS = {"薬剤歴", "嗜好/社会歴", "既往歴", "家族歴"}


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_disease_embeddings():
    """disease_embs.npzからMEAN集約済み疾患embeddingを読み込む。"""
    embs_file = os.path.join(DATA_DIR, "disease_embs.npz")
    if not os.path.exists(embs_file):
        raise FileNotFoundError(f"{embs_file} が見つかりません。index.pyを先に実行してください。")
    data = np.load(embs_file, allow_pickle=True)
    return list(data["disease_names"]), data["disease_embs_normed"].astype(np.float32)


def batch_embed(texts, batch_size=50, max_workers=10):
    """バッチ並行embedding。"""
    from openai import OpenAI
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


def normalize(embs):
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def main():
    print("=" * 80)
    print("HPE sim_matrix統合構築: screen + confirm")
    print("=" * 80)

    # ─── データ読み込み ───
    hpe_items = load_jsonl(HPE_JSONL)
    hpe_names = [item["item_name"] for item in hpe_items]
    hypotheses = [item["hypothesis"] for item in hpe_items]
    n_hpe = len(hpe_items)

    print(f"\nHPE項目: {n_hpe}件")

    # 逆引きインデックス
    risk_index = {}
    if os.path.exists(RISK_INDEX):
        with open(RISK_INDEX, "r", encoding="utf-8") as f:
            risk_index = json.load(f)
        print(f"Type R逆引き: {len(risk_index)}項目")
    else:
        print(f"WARNING: {RISK_INDEX} が見つかりません")

    finding_index = {}
    if os.path.exists(FINDING_INDEX):
        with open(FINDING_INDEX, "r", encoding="utf-8") as f:
            finding_index = json.load(f)
        print(f"Type F逆引き: {len(finding_index)}項目")
    else:
        print(f"WARNING: {FINDING_INDEX} が見つかりません")

    # ─── 疾患embedding ───
    print("\n--- 疾患embedding読み込み ---")
    disease_names, disease_embs = load_disease_embeddings()
    print(f"  {len(disease_names)}疾患 × {disease_embs.shape[1]}次元")

    # ─── Screen行列 ───
    print("\n" + "=" * 60)
    print("Screen行列構築")
    print("=" * 60)

    hyp_hash = hashlib.md5("||".join(hypotheses).encode()).hexdigest()[:12]
    rebuild_screen = True

    if os.path.exists(SCREEN_OUTPUT):
        data = np.load(SCREEN_OUTPUT, allow_pickle=True)
        cached_diseases = list(data.get("disease_names", []))
        cached_hpe = list(data.get("hpe_names", []))
        cached_hash = str(data.get("hypothesis_hash", ""))
        if cached_diseases == disease_names and cached_hpe == hpe_names and cached_hash == hyp_hash:
            print("  キャッシュ一致 → スキップ")
            rebuild_screen = False
            screen_matrix = data["sim_matrix"]
            hyp_embs_normed = data.get("hyp_embs")
        else:
            print("  キャッシュ不一致 → 再構築")
            if cached_hash != hyp_hash:
                print(f"    hypothesis変更検出 (old={cached_hash}, new={hyp_hash})")

    if rebuild_screen:
        print(f"\n  Embedding ({n_hpe}件)...")
        hyp_embs = batch_embed(hypotheses)
        if hyp_embs is None:
            print("ERROR: Screen embedding失敗")
            return
        hyp_embs_normed = normalize(hyp_embs)

        screen_matrix = disease_embs @ hyp_embs_normed.T
        print(f"  Screen行列: {screen_matrix.shape}")
        print(f"    mean={screen_matrix.mean():.4f}, std={screen_matrix.std():.4f}")

        np.savez(
            SCREEN_OUTPUT,
            sim_matrix=screen_matrix,
            hyp_embs=hyp_embs_normed,
            disease_names=np.array(disease_names, dtype=object),
            hpe_names=np.array(hpe_names, dtype=object),
            hypothesis_hash=np.array(hyp_hash),
        )
        print(f"  保存: {SCREEN_OUTPUT}")

    # Screen統計
    col_vars_s = np.var(screen_matrix, axis=0)
    print(f"\n  Screen列分散: mean={col_vars_s.mean():.6f}, "
          f"median={np.median(col_vars_s):.6f}")
    low_var = [(hpe_names[i], col_vars_s[i]) for i in range(n_hpe) if col_vars_s[i] < 0.0001]
    if low_var:
        print(f"  低分散項目 ({len(low_var)}件):")
        for name, v in low_var[:5]:
            print(f"    {name}: {v:.6f}")

    # ─── Confirm行列 ───
    print("\n" + "=" * 60)
    print("Confirm行列構築")
    print("=" * 60)

    # 各HPE項目のconfirm text（疾患名リスト）を構築
    confirm_texts = []
    coverage = {"risk": 0, "finding": 0, "fallback": 0}
    for item in hpe_items:
        name = item["item_name"]
        subcat = item["subcategory"]

        if subcat in TYPE_R_SUBCATS:
            # Type R: risk indexから
            dlist = risk_index.get(name, [])
            if dlist:
                confirm_texts.append("、".join(dlist))
                coverage["risk"] += 1
            else:
                confirm_texts.append(name)
                coverage["fallback"] += 1
        else:
            # Type F: finding indexから
            dlist = finding_index.get(name, [])
            if dlist:
                confirm_texts.append("、".join(dlist))
                coverage["finding"] += 1
            else:
                confirm_texts.append(name)
                coverage["fallback"] += 1

    print(f"  Confirm text構築: risk={coverage['risk']}, finding={coverage['finding']}, "
          f"fallback={coverage['fallback']}")

    lengths = [len(t) for t in confirm_texts]
    print(f"  文字数: mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}, "
          f"min={min(lengths)}, max={max(lengths)}")

    # ハッシュチェック
    c_hash = hashlib.md5("||".join(confirm_texts).encode()).hexdigest()[:12]
    rebuild_confirm = True

    if os.path.exists(CONFIRM_OUTPUT):
        data = np.load(CONFIRM_OUTPUT, allow_pickle=True)
        cached_hash = str(data.get("hypothesis_hash", ""))
        if cached_hash == c_hash:
            print("  キャッシュ一致 → スキップ")
            rebuild_confirm = False
            confirm_matrix = data["sim_matrix"]

    if rebuild_confirm:
        print(f"\n  Embedding ({n_hpe}件)...")
        confirm_embs = batch_embed(confirm_texts)
        if confirm_embs is None:
            print("ERROR: Confirm embedding失敗")
            return
        confirm_embs_normed = normalize(confirm_embs)

        confirm_matrix = disease_embs @ confirm_embs_normed.T
        print(f"  Confirm行列: {confirm_matrix.shape}")
        print(f"    mean={confirm_matrix.mean():.4f}, std={confirm_matrix.std():.4f}")

        np.savez(
            CONFIRM_OUTPUT,
            sim_matrix=confirm_matrix,
            disease_names=np.array(disease_names, dtype=object),
            hpe_names=np.array(hpe_names, dtype=object),
            hypothesis_hash=np.array(c_hash),
        )
        print(f"  保存: {CONFIRM_OUTPUT}")

    # Confirm統計
    col_vars_c = np.var(confirm_matrix, axis=0)
    print(f"\n  Confirm列分散: mean={col_vars_c.mean():.6f}, "
          f"median={np.median(col_vars_c):.6f}")

    # ─── 比較 ───
    print("\n" + "=" * 60)
    print("Screen vs Confirm比較")
    print("=" * 60)

    print(f"  Screen:  mean={screen_matrix.mean():.4f}, std={screen_matrix.std():.4f}")
    print(f"  Confirm: mean={confirm_matrix.mean():.4f}, std={confirm_matrix.std():.4f}")
    print(f"  Screen列分散:  mean={col_vars_s.mean():.6f}")
    print(f"  Confirm列分散: mean={col_vars_c.mean():.6f}")

    # Type R / Type F 別分散比較
    type_r_indices = [i for i, item in enumerate(hpe_items) if item["subcategory"] in TYPE_R_SUBCATS]
    type_f_indices = [i for i, item in enumerate(hpe_items) if item["subcategory"] not in TYPE_R_SUBCATS]

    print(f"\n  Type R ({len(type_r_indices)}件):")
    print(f"    Screen列分散:  mean={col_vars_s[type_r_indices].mean():.6f}")
    print(f"    Confirm列分散: mean={col_vars_c[type_r_indices].mean():.6f}")
    print(f"  Type F ({len(type_f_indices)}件):")
    print(f"    Screen列分散:  mean={col_vars_s[type_f_indices].mean():.6f}")
    print(f"    Confirm列分散: mean={col_vars_c[type_f_indices].mean():.6f}")

    print("\n完了")


if __name__ == "__main__":
    main()
