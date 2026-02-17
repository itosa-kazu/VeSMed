"""
ChromaDB → disease_embs.npz ワンショットエクスポート。
MEAN集約済み正規化疾患embeddingをNPZに保存する。
以降、engine.py / build_*.py はこのNPZのみを参照し、ChromaDB不要。
"""

import os
import numpy as np
from collections import defaultdict
import chromadb
from config import CHROMA_DIR, DATA_DIR

OUTPUT_FILE = os.path.join(DATA_DIR, "disease_embs.npz")


def main():
    print("ChromaDB → disease_embs.npz エクスポート")
    print("=" * 60)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("diseases")
    all_data = collection.get(include=["embeddings", "metadatas"])

    print(f"  チャンク総数: {len(all_data['ids'])}")

    # 疾患ごとにチャンクを集約
    disease_chunk_embs = defaultdict(list)
    for j in range(len(all_data["ids"])):
        dname = all_data["metadatas"][j].get("disease_name", "")
        if dname:
            disease_chunk_embs[dname].append(all_data["embeddings"][j])

    # MEAN集約 + 正規化
    disease_names = sorted(disease_chunk_embs.keys())
    embs_list = []
    chunk_counts = []
    for name in disease_names:
        chunks = np.array(disease_chunk_embs[name], dtype=np.float32)
        embs_list.append(chunks.mean(axis=0))
        chunk_counts.append(len(disease_chunk_embs[name]))
    disease_embs = np.array(embs_list, dtype=np.float32)

    d_norms = np.linalg.norm(disease_embs, axis=1, keepdims=True)
    d_norms[d_norms == 0] = 1.0
    disease_embs_normed = disease_embs / d_norms

    # 保存
    np.savez_compressed(
        OUTPUT_FILE,
        disease_embs_normed=disease_embs_normed,
        disease_names=np.array(disease_names, dtype=object),
    )

    # 検証
    size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print(f"  疾患数: {len(disease_names)}")
    print(f"  次元: {disease_embs_normed.shape[1]}")
    print(f"  チャンク数: mean={np.mean(chunk_counts):.1f}, min={min(chunk_counts)}, max={max(chunk_counts)}")
    print(f"  NaN: {np.isnan(disease_embs_normed).sum()}")
    print(f"  ノルム: mean={np.linalg.norm(disease_embs_normed, axis=1).mean():.6f}")
    print(f"  保存先: {OUTPUT_FILE} ({size_mb:.1f} MB)")
    print("完了")


if __name__ == "__main__":
    main()
