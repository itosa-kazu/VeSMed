"""Dual sim_matrix パイロット検証:
確認用hypothesis（疾患名リスト）でPart Cスコアが正しく機能するか検証。

検証シナリオ: 候補疾患 = [顕微鏡的多発血管炎, 全身性エリテマトーデス, 敗血症]
期待: MPO-ANCA > 抗dsDNA >> CRP（CRPは汎用なのでPart C下位であるべき）
"""
import json
import numpy as np
import os
import sys
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from config import (EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
                    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# パイロット検査（特異度の低い→高い順）
PILOT_TESTS = [
    "CRP",
    "白血球数（増加）",
    "胸部X線",
    "BNP",
    "トロポニンI",
    "TSH",
    "MPO-ANCA",
    "抗基底膜抗体 (抗GBM抗体)",
]

CONFIRM_PROMPT = """あなたは臨床医です。以下の検査が「異常値」を示した場合に、その検査結果によって確定診断に近づく、または強く示唆される疾患を、下記の522疾患マスタから選んでください。

検査名: {test_name}

【ルール】
1. その検査が「陽性」「異常高値」「異常低値」のいずれかを示した場合に、臨床的に意義のある疾患のみを選ぶ
2. 疾患マスタに存在する疾患名のみ使用（完全一致）
3. 関連が弱い疾患は含めない — その検査結果が診断に「直接貢献する」もののみ
4. 出力形式: 疾患名をカンマ区切りで列挙（他のテキストは不要）

【522疾患マスタ】
{disease_list}
"""


def load_disease_names():
    data = np.load(os.path.join(DATA_DIR, "sim_matrix.npz"), allow_pickle=True)
    return list(data["disease_names"])


def load_disease_embeddings():
    import chromadb
    from collections import defaultdict

    chroma_dir = os.path.join(DATA_DIR, "chroma_db")
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection("diseases")
    all_data = collection.get(include=["embeddings", "metadatas"])

    disease_chunk_embs = defaultdict(list)
    for j in range(len(all_data["ids"])):
        dname = all_data["metadatas"][j].get("disease_name", "")
        if dname:
            disease_chunk_embs[dname].append(all_data["embeddings"][j])

    disease_names = sorted(disease_chunk_embs.keys())
    embs_list = []
    for name in disease_names:
        chunks = np.array(disease_chunk_embs[name], dtype=np.float32)
        embs_list.append(chunks.mean(axis=0))
    disease_embs = np.array(embs_list, dtype=np.float32)

    d_norms = np.linalg.norm(disease_embs, axis=1, keepdims=True)
    d_norms[d_norms == 0] = 1.0
    return disease_names, disease_embs / d_norms


def load_sim_matrix():
    data = np.load(os.path.join(DATA_DIR, "sim_matrix.npz"), allow_pickle=True)
    return {
        "sim_matrix": data["sim_matrix"],
        "disease_names": list(data["disease_names"]),
        "test_names": list(data["test_names"]),
    }


def embed_text(text):
    client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    for attempt in range(3):
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
            return np.array(resp.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"  [embed] 失敗(試行{attempt+1}): {e}")
            import time; time.sleep(2 ** attempt)
    return None


def generate_disease_list(test_name, disease_names):
    """LLMで確認対象疾患リストを生成"""
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    disease_list_str = "、".join(disease_names)
    prompt = CONFIRM_PROMPT.format(test_name=test_name, disease_list=disease_list_str)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=4000,
            )
            content = resp.choices[0].message.content.strip()
            # パースして疾患マスタと照合
            # 半角カンマ、全角カンマ、改行のいずれでも分割
            import re
            raw_names = [n.strip() for n in re.split(r'[、,\n]+', content) if n.strip()]
            matched = [n for n in raw_names if n in disease_names]
            unmatched = [n for n in raw_names if n not in disease_names]
            return matched, unmatched
        except Exception as e:
            print(f"  [LLM] {test_name} 失敗(試行{attempt+1}): {e}")
            import time; time.sleep(3)
    return [], []


def compute_part_c_score(sim_col, candidate_indices, all_disease_count):
    """Part Cスコア = cluster_mu - global_mu（簡易版、均等重み）"""
    global_mu = float(np.mean(sim_col))
    if len(candidate_indices) > 0:
        cluster_mu = float(np.mean(sim_col[candidate_indices]))
    else:
        cluster_mu = global_mu
    return cluster_mu - global_mu, cluster_mu, global_mu


def main():
    print("=" * 90)
    print("Dual sim_matrix パイロット検証: 確認用hypothesisでPart Cは壊れるか？")
    print("=" * 90)

    # データ読み込み
    disease_names_master = load_disease_names()
    print(f"\n疾患マスタ: {len(disease_names_master)}件")

    print("疾患embedding読み込み中...")
    disease_names_chroma, disease_embs_normed = load_disease_embeddings()
    n_diseases = len(disease_names_chroma)
    print(f"  ChromaDB: {n_diseases}疾患")

    # 旧sim_matrix（比較用）
    sm_data = load_sim_matrix()
    sm_test_names = sm_data["test_names"]

    # ChromaDB疾患名 → index
    chroma_idx = {name: i for i, name in enumerate(disease_names_chroma)}

    # ============================================================
    # Step 1: LLMで確認対象疾患リストを生成
    # ============================================================
    print("\n--- Step 1: 確認対象疾患リスト生成 ---")
    confirm_data = {}
    for test_name in PILOT_TESTS:
        print(f"\n  [{test_name}] LLM生成中...")
        matched, unmatched = generate_disease_list(test_name, disease_names_master)
        print(f"    マッチ: {len(matched)}件, 不一致: {len(unmatched)}件")
        if unmatched:
            print(f"    不一致例: {unmatched[:5]}")

        # 確認用テキスト生成
        if matched:
            confirm_text = f"本検査が確定診断または強く示唆する対象疾患: {'、'.join(matched)}"
        else:
            confirm_text = ""
        confirm_data[test_name] = {
            "diseases": matched,
            "n_diseases": len(matched),
            "confirm_text": confirm_text,
        }
        print(f"    確認用テキスト: {len(confirm_text)}字")

    # ============================================================
    # Step 2: 確認用hypothesisをembed
    # ============================================================
    print("\n--- Step 2: Embedding ---")
    confirm_cols = {}
    for test_name in PILOT_TESTS:
        text = confirm_data[test_name]["confirm_text"]
        if not text:
            print(f"  {test_name}: スキップ（疾患リスト空）")
            continue
        emb = embed_text(text)
        if emb is None:
            print(f"  {test_name}: embedding失敗")
            continue
        emb_norm = np.linalg.norm(emb)
        if emb_norm > 0:
            emb = emb / emb_norm
        # sim_confirm列 = disease_embs @ hypothesis_emb
        sim_col = disease_embs_normed @ emb
        confirm_cols[test_name] = sim_col
        print(f"  {test_name}: done (mean_cos={np.mean(sim_col):.4f}, var={np.var(sim_col):.6f})")

    # ============================================================
    # Step 3: Part Cスコア検証（複数シナリオ）
    # ============================================================
    # 検証シナリオ: 候補疾患セット
    scenarios = [
        {
            "name": "血管炎疑い",
            "candidates": ["顕微鏡的多発血管炎", "好酸球性多発血管炎性肉芽腫症", "多発血管炎性肉芽腫症"],
            "expected_top": "MPO-ANCA",
        },
        {
            "name": "SLE疑い",
            "candidates": ["全身性エリテマトーデス", "ループス腎炎", "抗リン脂質抗体症候群"],
            "expected_top": "抗dsDNA（不在→TSHなどが上位でないこと確認）",
        },
        {
            "name": "心不全疑い",
            "candidates": ["急性心不全 (HFrEF)", "急性心不全 (HFpEF)", "慢性心不全急性増悪"],
            "expected_top": "BNP",
        },
        {
            "name": "敗血症疑い",
            "candidates": ["敗血症", "敗血症性ショック"],
            "expected_top": "CRPは下位であるべき",
        },
        {
            "name": "甲状腺疑い",
            "candidates": ["バセドウ病", "橋本病", "甲状腺クリーゼ"],
            "expected_top": "TSH",
        },
    ]

    for scenario in scenarios:
        print(f"\n{'=' * 90}")
        print(f"シナリオ: {scenario['name']}")
        print(f"候補疾患: {scenario['candidates']}")
        print(f"期待: {scenario['expected_top']}")
        print(f"{'=' * 90}")

        # 候補疾患のChromaDBインデックス
        cand_indices = [chroma_idx[d] for d in scenario["candidates"] if d in chroma_idx]

        # 各検査のPart Cスコア（確認用sim_matrix使用）
        print(f"\n  {'検査名':<30} {'n_diseases':>10} {'cluster_μ':>10} {'global_μ':>10} {'Part_C':>10} {'旧Part_C':>10}")
        print(f"  {'─' * 85}")

        scores = []
        for test_name in PILOT_TESTS:
            n_d = confirm_data[test_name]["n_diseases"]

            # 新: 確認用sim
            if test_name in confirm_cols:
                new_score, new_cl, new_gl = compute_part_c_score(
                    confirm_cols[test_name], cand_indices, n_diseases)
            else:
                new_score, new_cl, new_gl = 0, 0, 0

            # 旧: 現行sim_matrix
            if test_name in sm_test_names:
                sm_idx = sm_test_names.index(test_name)
                # sim_matrixの疾患順をChromaDB順にリマップ
                sm_disease_names = sm_data["disease_names"]
                old_col = np.zeros(n_diseases, dtype=np.float32)
                for i, dn in enumerate(sm_disease_names):
                    if dn in chroma_idx:
                        old_col[chroma_idx[dn]] = sm_data["sim_matrix"][i, sm_idx]
                old_score, old_cl, old_gl = compute_part_c_score(
                    old_col, cand_indices, n_diseases)
            else:
                old_score, old_cl, old_gl = 0, 0, 0

            scores.append((test_name, n_d, new_score, new_cl, new_gl, old_score))
            print(f"  {test_name:<30} {n_d:>10} {new_cl:>10.4f} {new_gl:>10.4f} {new_score:>+10.4f} {old_score:>+10.4f}")

        # ランキング比較
        new_ranking = sorted(scores, key=lambda x: x[2], reverse=True)
        old_ranking = sorted(scores, key=lambda x: x[5], reverse=True)
        print(f"\n  新Part Cランキング: {' > '.join(f'{s[0]}({s[2]:+.4f})' for s in new_ranking)}")
        print(f"  旧Part Cランキング: {' > '.join(f'{s[0]}({s[5]:+.4f})' for s in old_ranking)}")

    # 結果保存
    result = {}
    for test_name in PILOT_TESTS:
        result[test_name] = {
            "n_diseases": confirm_data[test_name]["n_diseases"],
            "diseases": confirm_data[test_name]["diseases"],
            "confirm_text_chars": len(confirm_data[test_name]["confirm_text"]),
        }
        if test_name in confirm_cols:
            col = confirm_cols[test_name]
            result[test_name]["mean_cos"] = float(np.mean(col))
            result[test_name]["var"] = float(np.var(col))
            result[test_name]["median_cos"] = float(np.median(col))

    output_path = os.path.join(DATA_DIR, "pilot_dual_verify.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {output_path}")


if __name__ == "__main__":
    main()
