"""
VeSMed - ベクトルDB構築スクリプト
fd_*フィールド（セクション分離格納）を直接読み取り、
各チャンクをEmbeddingしてChromaDBに格納する。

fd_*がない場合はfindings_descriptionからチャンク分割にフォールバック。

6-Domain Embedding:
  1. 背景・疫学・リスク (Context)
  2. 典型主訴・病歴 (Subjective-Typical)
  3. 非典型・ピットフォール (Subjective-Atypical)
  4. バイタル・身体所見 (Objective-Signs)
  5. 検査所見 (Objective-Tests)
  6. 病態生理 (Pathophysiology)
  ❌ 鑑別キー → Data Leakage防止のためembedding空間から排除
"""

import json
import os
import numpy as np
import chromadb
from openai import OpenAI
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import (
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
    DISEASES_JSONL, CHROMA_DIR, DATA_DIR,
)

# ─── ドメイン定義 ───
DOMAINS = [
    # (name, embed?, description)
    ("background",      True,  "背景・疫学・リスク"),
    ("typical",         True,  "典型主訴・病歴"),
    ("atypical",        True,  "非典型・ピットフォール"),
    ("physical",        True,  "バイタル・身体所見"),
    ("tests",           True,  "検査所見"),
    ("differential",    False, "鑑別キー（排除）"),
    ("pathophysiology", True,  "病態生理"),
]

EMBED_DOMAINS = [d[0] for d in DOMAINS if d[1]]  # embed対象

# fd_* フィールド → ドメイン名のマッピング
FD_TO_DOMAIN = {
    "fd_background":      "background",
    "fd_typical":         "typical",
    "fd_atypical":        "atypical",
    "fd_physical":        "physical",
    "fd_tests":           "tests",
    "fd_differential":    "differential",
    "fd_pathophysiology": "pathophysiology",
}

# embed対象の fd_* キー
EMBED_FD_KEYS = [k for k, v in FD_TO_DOMAIN.items() if v != "differential"]


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


# ─── レガシーチャンカー（フォールバック用） ───

def _is_header_line(stripped: str) -> bool:
    """ヘッダー行かどうか判定（箇条書き・インデント行を除外）"""
    if not stripped:
        return False
    if stripped.startswith("#") or stripped.startswith("■") or stripped.startswith("【"):
        return True
    if (not stripped.startswith("-") and not stripped.startswith("*")
            and not stripped.startswith(" ") and len(stripped) < 100):
        return True
    return False


def chunk_into_domains(text: str) -> dict:
    """
    findings_descriptionを7つの臨床ドメインに分割（レガシーフォールバック用）。
    fd_*フィールドが存在する場合はこの関数は呼ばれない。
    """
    lines = text.split("\n")
    boundaries = [(0, "background")]

    found = {d: False for d in ["typical", "atypical", "physical", "tests",
                                 "differential", "pathophysiology"]}

    for i, line in enumerate(lines):
        s = line.strip()
        if not s or not _is_header_line(s):
            continue

        if (not found["typical"]
                and ("主訴" in s or "典型来院像" in s or "来院像" in s)
                and "非典型" not in s and "ピットフォール" not in s):
            boundaries.append((i, "typical"))
            found["typical"] = True

        elif (not found["atypical"] and found["typical"]
              and ("非典型" in s or "ピットフォール" in s
                   or "見逃し" in s or "誤診" in s)):
            boundaries.append((i, "atypical"))
            found["atypical"] = True

        elif (not found["physical"] and found["typical"]
              and ("バイタルサイン" in s or ("身体所見" in s and len(s) < 80))):
            if not found["atypical"]:
                found["atypical"] = True
            boundaries.append((i, "physical"))
            found["physical"] = True

        elif (not found["tests"] and found["typical"]
              and ("検査所見" in s or "検査パターン" in s)):
            if not found["atypical"]:
                found["atypical"] = True
            if not found["physical"]:
                found["physical"] = True
            boundaries.append((i, "tests"))
            found["tests"] = True

        elif (not found["differential"]
              and ("鑑別キー" in s or "主要な鑑別疾患" in s or "鑑別ポイント" in s)):
            if not found["tests"]:
                found["tests"] = True
            boundaries.append((i, "differential"))
            found["differential"] = True

        elif (not found["pathophysiology"]
              and ("病態生理" in s or "発症メカニズム" in s)):
            boundaries.append((i, "pathophysiology"))
            found["pathophysiology"] = True

    sections = {}
    for idx in range(len(boundaries)):
        start = boundaries[idx][0]
        domain = boundaries[idx][1]
        end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
        section_text = "\n".join(lines[start:end]).strip()
        if section_text:
            sections[domain] = section_text

    return sections


def _get_sections_for_disease(d: dict) -> dict:
    """疾患レコードからドメイン別セクションを取得。
    fd_*フィールドがあればそれを使い、なければchunk_into_domainsにフォールバック。"""
    # fd_*フィールドチェック
    has_fd = any(d.get(k) for k in EMBED_FD_KEYS)
    if has_fd:
        sections = {}
        for fd_key, domain in FD_TO_DOMAIN.items():
            text = d.get(fd_key, "")
            if text:
                sections[domain] = text
        return sections

    # フォールバック: findings_descriptionからチャンク分割
    desc = d.get("findings_description", "")
    if not desc:
        return {}
    return chunk_into_domains(desc)


def get_embeddings(client, texts, model=EMBEDDING_MODEL):
    """テキストのリストをEmbeddingする。バッチ対応。"""
    response = client.embeddings.create(model=model, input=texts)
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

    # ドメイン分割してチャンクを準備
    texts = []
    ids = []
    metadatas = []
    disease_count = 0
    domain_stats = {d: 0 for d in EMBED_DOMAINS}
    skipped_diff = 0
    fd_direct = 0
    fd_fallback = 0

    for i, d in enumerate(diseases):
        has_fd = any(d.get(k) for k in EMBED_FD_KEYS)
        sections = _get_sections_for_disease(d)

        if not sections:
            print(f"  警告: {d.get('disease_name', '不明')} にデータがありません、スキップ")
            continue

        if has_fd:
            fd_direct += 1
        else:
            fd_fallback += 1

        disease_count += 1

        for domain in EMBED_DOMAINS:
            if domain not in sections:
                continue
            chunk = sections[domain]
            texts.append(chunk)
            ids.append(f"disease_{i}_{domain}")
            metadatas.append({
                "disease_name": d.get("disease_name", ""),
                "category": d.get("category", ""),
                "icd10": d.get("icd10", ""),
                "urgency": d.get("urgency", ""),
                "domain": domain,
            })
            domain_stats[domain] += 1

        if "differential" in sections:
            skipped_diff += 1

    print(f"Embedding対象: {disease_count}疾患 → {len(texts)}チャンク")
    print(f"  データソース: fd_*直読み {fd_direct}件 / レガシーフォールバック {fd_fallback}件")
    print(f"  ドメイン別:")
    for domain in EMBED_DOMAINS:
        label = next(d[2] for d in DOMAINS if d[0] == domain)
        print(f"    {domain:<20s} ({label}): {domain_stats[domain]}件")
    print(f"  鑑別キー排除: {skipped_diff}件")

    # Embedding取得（バッチ並行）
    client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    batch_size = 50
    max_workers = 5

    batches = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batches.append((start, end, texts[start:end]))

    all_embeddings = [None] * len(texts)

    def embed_batch(batch_info):
        s, e, batch_texts = batch_info
        embeddings = get_embeddings(client, batch_texts)
        return s, embeddings

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(embed_batch, b): b for b in batches}
        done = 0
        for future in as_completed(futures):
            s, embeddings = future.result()
            batch_info = futures[future]
            e = batch_info[1]
            for j, emb in enumerate(embeddings):
                all_embeddings[s + j] = emb
            done += 1
            if done % 10 == 0 or done == len(batches):
                print(f"  Embedding batch {done}/{len(batches)} ... OK")

    all_embeddings = [e for e in all_embeddings if e is not None]

    # ChromaDBに格納
    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        chroma_client.delete_collection("diseases")
        print("  既存コレクションを削除")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="diseases",
        metadata={"hnsw:space": "cosine"},
    )

    # バッチ追加
    add_batch = 5000
    for start in range(0, len(ids), add_batch):
        end = min(start + add_batch, len(ids))
        collection.add(
            ids=ids[start:end],
            embeddings=all_embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"\nベクトルDB構築完了: {collection.count()}チャンク ({disease_count}疾患) をChromaDBに格納")
    print(f"  = {disease_count}疾患 × 最大6ドメイン（鑑別キー排除済み）")
    print(f"保存先: {CHROMA_DIR}")

    # ─── MEAN集約NPZエクスポート ───
    print("\nMEAN集約embeddingをNPZに保存...")
    disease_chunk_embs = defaultdict(list)
    for j in range(len(ids)):
        dname = metadatas[j].get("disease_name", "")
        if dname:
            disease_chunk_embs[dname].append(all_embeddings[j])

    export_names = sorted(disease_chunk_embs.keys())
    export_embs_list = []
    for name in export_names:
        chunks = np.array(disease_chunk_embs[name], dtype=np.float32)
        export_embs_list.append(chunks.mean(axis=0))
    export_embs = np.array(export_embs_list, dtype=np.float32)

    e_norms = np.linalg.norm(export_embs, axis=1, keepdims=True)
    e_norms[e_norms == 0] = 1.0
    export_embs_normed = export_embs / e_norms

    npz_file = os.path.join(DATA_DIR, "disease_embs.npz")
    np.savez_compressed(
        npz_file,
        disease_embs_normed=export_embs_normed,
        disease_names=np.array(export_names, dtype=object),
    )
    size_mb = os.path.getsize(npz_file) / 1024 / 1024
    print(f"MEAN集約NPZ保存: {npz_file} ({len(export_names)}疾患, {size_mb:.1f}MB)")


if __name__ == "__main__":
    build_index()
