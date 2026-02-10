"""
VeSMed - ベクトルDB構築スクリプト
diseases.jsonlのdescription_for_embeddingをEmbeddingして
ChromaDBに格納する。
"""

import json
import os
import chromadb
from openai import OpenAI
from config import (
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
    DISEASES_JSONL, CHROMA_DIR,
)


def load_diseases(jsonl_path):
    """diseases.jsonlを読み込んでリストで返す"""
    diseases = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                diseases.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return diseases


def get_embeddings(client, texts, model=EMBEDDING_MODEL):
    """テキストのリストをEmbeddingする。バッチ対応。"""
    response = client.embeddings.create(
        model=model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def build_index():
    if not os.path.exists(DISEASES_JSONL):
        print(f"エラー: {DISEASES_JSONL} が見つかりません。先に generate.py diseases を実行してください。")
        return

    diseases = load_diseases(DISEASES_JSONL)
    if not diseases:
        print("エラー: 疾患データが0件です。")
        return

    print(f"疾患データ: {len(diseases)}件")

    # Embeddingするテキストを準備
    texts = []
    ids = []
    metadatas = []
    for i, d in enumerate(diseases):
        desc = d.get("findings_description", "")
        if not desc:
            print(f"  警告: {d.get('disease_name', '不明')} にfindings_descriptionがありません、スキップ")
            continue
        texts.append(desc)
        ids.append(f"disease_{i}")
        metadatas.append({
            "disease_name": d.get("disease_name", ""),
            "category": d.get("category", ""),
            "icd10": d.get("icd10", ""),
            "urgency": d.get("urgency", ""),
        })

    print(f"Embedding対象: {len(texts)}件")

    # Embedding取得（バッチで処理、API制限を考慮して分割）
    client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    batch_size = 20
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        print(f"  Embedding [{start + 1}-{end}/{len(texts)}] ...", end="", flush=True)
        embeddings = get_embeddings(client, batch_texts)
        all_embeddings.extend(embeddings)
        print(" OK")

    # ChromaDBに格納
    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 既存コレクションがあれば削除して再構築
    try:
        chroma_client.delete_collection("diseases")
        print("  既存コレクションを削除")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="diseases",
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=ids,
        embeddings=all_embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"\nベクトルDB構築完了: {collection.count()}件をChromaDBに格納")
    print(f"保存先: {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()
