"""
Semantic Chunking 実験スクリプト
目的: 20K字findings_descriptionを意味チャンクに分割し、
      max-chunk類似度で疾患ランキングが改善するか検証する。

方法:
  1. 現行ChromaDB（全文1ベクトル）でベースライン取得
  2. findings_descriptionをパラグラフ分割
  3. 各チャンクをembed → 疾患ごとにmax類似度
  4. ベースライン vs チャンクのランキング比較
"""

import json
import os
import sys
import time
import numpy as np
import chromadb
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
    DISEASES_JSONL, CHROMA_DIR,
)

# ─── 設定 ───
CASE_TEXT = """65歳の男性。言動に不安を感じた妻に伴われて来院した。高血圧症で内服加療中である。
朝の散歩を日課としているが，半年前から必ず時刻通りに出かけることにこだわるようになった。
また，帰省した息子や孫を突然怒鳴りつけるなど，怒りっぽくなった。
食事は同じ内容にこだわるようになり，異なるメニューを供すると怒り出して食事の最中に席を離れてしまうことがあった。
趣味のサークルの友人から妻に電話があり，最近，サークルの運営で自分の主張を押し通そうとして困っていると相談された。
物忘れはなく，会話の不自由さはない。日常生活動作に支障はみられない。
妻は言動変化の原因になるような心当たりはないという。
診察室でも本人は受診が不満のようで，妻をなじっている。"""

# 正解疾患
CORRECT_DISEASE = "前頭側頭型認知症"

# 比較対象として必ず含める疾患
MUST_INCLUDE = ["前頭側頭型認知症", "強迫性障害"]


def load_diseases():
    """diseases.jsonlを読み込む"""
    diseases = []
    with open(DISEASES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                diseases.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return diseases


def baseline_search(query_text: str) -> list:
    """現行ChromaDB（全文1ベクトル）でベースライン検索"""
    client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query_text])
    query_emb = resp.data[0].embedding

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_collection("diseases")
    n_total = collection.count()

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n_total,
    )

    candidates = []
    for i, doc_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        similarity = 1.0 - distance
        metadata = results["metadatas"][0][i]
        candidates.append({
            "disease_name": metadata.get("disease_name", ""),
            "similarity": similarity,
        })

    # similarity降順ソート
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    return candidates, np.array(query_emb, dtype=np.float32)


def chunk_text(text: str, min_chunk_chars: int = 200, max_chunk_chars: int = 2000) -> list:
    """
    テキストをパラグラフベースでチャンク分割。
    短すぎるパラグラフは次と結合。長すぎるパラグラフは句点で分割。
    """
    # まずパラグラフ（空行区切り or 改行区切り）に分割
    paragraphs = []
    for block in text.split("\n\n"):
        for line in block.split("\n"):
            stripped = line.strip()
            if stripped:
                paragraphs.append(stripped)

    if not paragraphs:
        return [text] if text.strip() else []

    # 短いパラグラフを結合
    chunks = []
    buffer = ""
    for para in paragraphs:
        if buffer:
            buffer += "\n" + para
        else:
            buffer = para

        if len(buffer) >= min_chunk_chars:
            # 長すぎる場合は句点で分割
            if len(buffer) > max_chunk_chars:
                sentences = buffer.replace("。", "。\n").split("\n")
                sub_buf = ""
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if sub_buf and len(sub_buf) + len(sent) > max_chunk_chars:
                        chunks.append(sub_buf)
                        sub_buf = sent
                    else:
                        sub_buf = (sub_buf + sent) if sub_buf else sent
                if sub_buf:
                    chunks.append(sub_buf)
            else:
                chunks.append(buffer)
            buffer = ""

    if buffer:
        if chunks and len(buffer) < min_chunk_chars:
            chunks[-1] += "\n" + buffer
        else:
            chunks.append(buffer)

    return chunks


def embed_batch(client, texts: list, batch_size: int = 50, max_workers: int = 5) -> list:
    """テキストリストをバッチembedding（並行処理）"""
    batches = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batches.append((start, end, texts[start:end]))

    all_embs = [None] * len(texts)

    def _embed_one(batch_info):
        s, e, batch_texts = batch_info
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch_texts)
        return s, [np.array(item.embedding, dtype=np.float32) for item in resp.data]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_embed_one, b): b for b in batches}
        done = 0
        for future in as_completed(futures):
            s, embs = future.result()
            for i, emb in enumerate(embs):
                all_embs[s + i] = emb
            done += 1
            print(f"    batch {done}/{len(batches)}", flush=True)

    return [e for e in all_embs if e is not None]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """cosine類似度"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def run_experiment():
    print("=" * 70)
    print("Semantic Chunking 実験")
    print("=" * 70)

    # ─── Step 1: ベースライン ───
    print("\n[Step 1] ベースライン検索（現行: 全文1ベクトル）...")
    baseline, query_emb = baseline_search(CASE_TEXT)
    print(f"  全疾患数: {len(baseline)}")

    # ベースライン上位20 + 必須疾患の順位を表示
    baseline_ranks = {c["disease_name"]: (i + 1, c["similarity"]) for i, c in enumerate(baseline)}

    print("\n  ── ベースライン上位20 ──")
    for i, c in enumerate(baseline[:20]):
        marker = " ★" if c["disease_name"] == CORRECT_DISEASE else ""
        print(f"  {i+1:3d}. {c['disease_name']:<25s}  sim={c['similarity']:.4f}{marker}")

    for name in MUST_INCLUDE:
        rank, sim = baseline_ranks.get(name, (None, None))
        if rank and rank > 20:
            print(f"  ...{rank:3d}. {name:<25s}  sim={sim:.4f} ★")

    # ─── Step 2: 疾患データ読み込み + チャンク分割（上位30 + 必須疾患に限定）───
    print("\n[Step 2] 疾患データ読み込み + チャンク分割...")
    diseases = load_diseases()

    # ベースライン上位30の疾患名 + 必須疾患を対象にする
    top_names = set(c["disease_name"] for c in baseline[:30])
    for name in MUST_INCLUDE:
        top_names.add(name)
    print(f"  対象疾患: ベースライン上位30 + 必須疾患 = {len(top_names)}件")

    # 対象疾患のみチャンク分割
    disease_chunks = {}  # disease_name -> [chunk_text, ...]
    total_chunks = 0
    for d in diseases:
        name = d.get("disease_name", "")
        desc = d.get("findings_description", "")
        if not desc or not name:
            continue
        if name not in top_names:
            continue
        chunks = chunk_text(desc)
        disease_chunks[name] = chunks
        total_chunks += len(chunks)

    print(f"  チャンク分割済み: {len(disease_chunks)}件")
    print(f"  総チャンク数: {total_chunks}")
    print(f"  平均チャンク/疾患: {total_chunks / len(disease_chunks):.1f}")

    # FTDとOCDのチャンク状況を表示（簡潔に）
    for name in MUST_INCLUDE:
        if name in disease_chunks:
            chunks = disease_chunks[name]
            sizes = [len(c) for c in chunks]
            print(f"  {name}: {len(chunks)}チャンク (avg {sum(sizes)/len(sizes):.0f}字)")

    # ─── Step 3: 全チャンクをembed ───
    print(f"\n[Step 3] 全{total_chunks}チャンクをembedding...")
    client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)

    # フラットなリストに展開
    flat_texts = []
    flat_index = []  # (disease_name, chunk_idx)
    for name, chunks in disease_chunks.items():
        for j, chunk in enumerate(chunks):
            flat_texts.append(chunk)
            flat_index.append((name, j))

    print(f"  embedding {len(flat_texts)} chunks...", flush=True)
    start_time = time.time()
    flat_embs = embed_batch(client, flat_texts, batch_size=50)
    elapsed = time.time() - start_time
    print(f"  完了: {elapsed:.1f}秒")

    # 疾患ごとにembeddingをまとめる
    disease_chunk_embs = {}  # disease_name -> [np.array, ...]
    for (name, j), emb in zip(flat_index, flat_embs):
        if name not in disease_chunk_embs:
            disease_chunk_embs[name] = []
        disease_chunk_embs[name].append(emb)

    # ─── Step 4: Max-chunk類似度計算 ───
    print("\n[Step 4] Max-chunk類似度計算...")
    chunk_results = []
    for name, embs in disease_chunk_embs.items():
        sims = [cosine_sim(query_emb, emb) for emb in embs]
        max_sim = max(sims)
        max_idx = int(np.argmax(sims))
        chunk_results.append({
            "disease_name": name,
            "max_sim": max_sim,
            "max_chunk_idx": max_idx,
            "all_sims": sims,
        })

    chunk_results.sort(key=lambda x: x["max_sim"], reverse=True)

    # ─── Step 5: 比較 ───
    chunk_ranks = {c["disease_name"]: (i + 1, c["max_sim"]) for i, c in enumerate(chunk_results)}

    print("\n" + "=" * 70)
    print("結果比較")
    print("=" * 70)

    print(f"\n  {'疾患名':<25s}  {'ベースライン':>12s}  {'チャンク':>12s}  {'順位変動':>8s}")
    print(f"  {'─' * 25}  {'─' * 12}  {'─' * 12}  {'─' * 8}")

    # 上位20（チャンク方式）を表示
    for i, c in enumerate(chunk_results[:20]):
        name = c["disease_name"]
        b_rank, b_sim = baseline_ranks.get(name, (999, 0.0))
        c_rank = i + 1
        delta = b_rank - c_rank
        delta_str = f"+{delta}" if delta > 0 else str(delta) if delta < 0 else "="
        marker = " ★" if name == CORRECT_DISEASE else ""
        print(f"  {c_rank:3d}. {name:<25s}  {b_rank:3d} ({b_sim:.4f})  {c_rank:3d} ({c['max_sim']:.4f})  {delta_str:>5s}{marker}")

    # 必須疾患が上位20外の場合も表示
    for name in MUST_INCLUDE:
        c_rank, c_sim = chunk_ranks.get(name, (None, None))
        if c_rank and c_rank > 20:
            b_rank, b_sim = baseline_ranks.get(name, (999, 0.0))
            delta = b_rank - c_rank
            delta_str = f"+{delta}" if delta > 0 else str(delta) if delta < 0 else "="
            print(f"  ...{c_rank:3d}. {name:<25s}  {b_rank:3d} ({b_sim:.4f})  {c_rank:3d} ({c_sim:.4f})  {delta_str:>5s} ★")

    # ─── サマリー ───
    print("\n" + "=" * 70)
    print("サマリー")
    print("=" * 70)
    for name in MUST_INCLUDE:
        b_rank, b_sim = baseline_ranks.get(name, (999, 0.0))
        c_rank, c_sim = chunk_ranks.get(name, (999, 0.0))
        print(f"\n  {name}:")
        print(f"    ベースライン: {b_rank}位 (sim={b_sim:.4f})")
        print(f"    チャンク方式: {c_rank}位 (sim={c_sim:.4f})")
        print(f"    順位変動: {b_rank}位 → {c_rank}位 ({b_rank - c_rank:+d})")

        # 最マッチチャンクの内容表示
        if name in disease_chunks:
            for cr in chunk_results:
                if cr["disease_name"] == name:
                    best_idx = cr["max_chunk_idx"]
                    best_chunk = disease_chunks[name][best_idx]
                    print(f"    最マッチチャンク [{best_idx+1}/{len(disease_chunks[name])}]: {best_chunk[:100]}...")
                    # 全チャンクのsim分布
                    sims = cr["all_sims"]
                    print(f"    全チャンクsim: min={min(sims):.4f}, max={max(sims):.4f}, "
                          f"mean={np.mean(sims):.4f}, std={np.std(sims):.4f}")
                    break

    print()


if __name__ == "__main__":
    run_experiment()
