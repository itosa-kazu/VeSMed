"""
バンドル方式 vs 現行A方式（sim_matrix + centering）の比較検証
manual 30ケースで評価。
"""

import copy
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import VeSMedEngine
from config import EMBEDDING_MODEL

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_manual_cases():
    path = os.path.join(DATA_DIR, "test_cases_manual.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cases = []
    for item in raw:
        expectations = []
        for c in item.get("correct", []):
            expectations.append({
                "disease": c["disease"],
                "direction": c["direction"],
                "type": "correct",
            })
        for w in item.get("wrong", []):
            expectations.append({
                "disease": w["disease"],
                "direction": w["direction"],
                "type": "wrong",
            })
        cases.append({
            "id": item["id"],
            "name": item["name"],
            "symptoms": item["symptoms"],
            "results": item["results"],
            "expectations": expectations,
        })
    return cases


def load_disease_embeddings(eng):
    """ChromaDBから全疾患embeddingを取得"""
    n = eng.collection.count()
    all_ids = [f"disease_{i}" for i in range(n)]
    disease_embs = {}
    batch_size = 100
    for start in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[start:start + batch_size]
        result = eng.collection.get(ids=batch_ids, include=["embeddings", "metadatas"])
        for j, mid in enumerate(result["ids"]):
            dname = result["metadatas"][j].get("disease_name", "")
            emb = np.array(result["embeddings"][j], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            disease_embs[dname] = emb
    return disease_embs


def update_bundle(eng, candidates, result_lines, disease_embs):
    """
    所見バンドル方式:
    全所見をアノテーション → 1テキストに結合 → embed → 疾患embと直接比較
    """
    # 1. アノテーション（基準範囲テーブル）
    annotated = eng._annotate_with_ranges(result_lines)

    # 2. バンドル結合
    bundle_text = "、".join(annotated)

    # 3. embed
    resp = eng.embed_client.embeddings.create(
        model=EMBEDDING_MODEL, input=[bundle_text],
    )
    bundle_emb = np.array(resp.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(bundle_emb)
    if norm > 0:
        bundle_emb = bundle_emb / norm

    # 4. 全疾患との cos 類似度
    relevance = {}
    for dname, demb in disease_embs.items():
        relevance[dname] = float(np.dot(bundle_emb, demb))

    # 5. 全疾患平均（centering）
    all_rels = list(relevance.values())
    mean_rel = np.mean(all_rels)

    # 6. 更新
    for c in candidates:
        dname = c["disease_name"]
        if dname in relevance:
            excess = relevance[dname] - mean_rel
            c["similarity"] *= float(np.exp(excess))

    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    return candidates, bundle_text, mean_rel


def update_current(eng, candidates, result_lines, symptoms):
    """現行A方式（sim_matrix + column centering）"""
    return eng.update_from_results(candidates, result_lines, symptoms, mode="fast")


def get_rank(candidates, disease_name):
    for i, c in enumerate(candidates):
        if c["disease_name"] == disease_name:
            return i + 1
    return 999


def score_expectation(rank_before, rank_after, direction):
    delta = rank_before - rank_after
    if direction == "up":
        return +1 if delta > 0 else (-1 if delta < 0 else 0)
    elif direction == "down":
        return +1 if delta < 0 else (-1 if delta > 0 else 0)
    elif direction == "stay":
        return +1 if abs(delta) <= 5 else 0
    return 0


def run_comparison(eng, cases, disease_embs):
    """両方式で全ケースを実行して比較"""

    total_a = 0
    total_b = 0
    a_wins = 0
    b_wins = 0
    ties = 0

    detail_rows = []

    for case in cases:
        # 初期検索（共通）
        candidates_init = eng.search_diseases(case["symptoms"])

        # A方式
        cands_a = copy.deepcopy(candidates_init)
        cands_a = update_current(eng, cands_a, case["results"], case["symptoms"])

        # Bundle方式
        cands_b = copy.deepcopy(candidates_init)
        cands_b, bundle_text, mean_rel = update_bundle(
            eng, cands_b, case["results"], disease_embs
        )

        # スコア計算
        case_score_a = 0
        case_score_b = 0
        case_details = []

        for exp in case["expectations"]:
            r_init = get_rank(candidates_init, exp["disease"])
            r_a = get_rank(cands_a, exp["disease"])
            r_b = get_rank(cands_b, exp["disease"])

            s_a = score_expectation(r_init, r_a, exp["direction"])
            s_b = score_expectation(r_init, r_b, exp["direction"])
            case_score_a += s_a
            case_score_b += s_b

            case_details.append({
                "disease": exp["disease"],
                "direction": exp["direction"],
                "type": exp["type"],
                "rank_init": r_init,
                "rank_a": r_a,
                "rank_b": r_b,
                "score_a": s_a,
                "score_b": s_b,
            })

        total_a += case_score_a
        total_b += case_score_b
        if case_score_a > case_score_b:
            a_wins += 1
        elif case_score_b > case_score_a:
            b_wins += 1
        else:
            ties += 1

        detail_rows.append({
            "case": case,
            "score_a": case_score_a,
            "score_b": case_score_b,
            "details": case_details,
        })

    return total_a, total_b, a_wins, b_wins, ties, detail_rows


def main():
    print("=" * 70)
    print("バンドル方式 vs 現行A方式 比較検証")
    print("=" * 70)

    # エンジン初期化
    print("\n[初期化]")
    t0 = time.time()
    eng = VeSMedEngine()
    print(f"エンジン起動: {time.time()-t0:.1f}秒")

    # 疾患embedding読込
    t1 = time.time()
    disease_embs = load_disease_embeddings(eng)
    print(f"疾患embedding: {len(disease_embs)}件 ({time.time()-t1:.1f}秒)")

    # テストケース
    cases = load_manual_cases()
    print(f"テストケース: {len(cases)}件 (manual)")

    # 比較実行
    print("\n" + "=" * 70)
    total_a, total_b, a_wins, b_wins, ties, details = run_comparison(
        eng, cases, disease_embs
    )

    # 結果出力
    print("\n" + "=" * 70)
    print("ケース別結果")
    print("=" * 70)
    for row in details:
        c = row["case"]
        sa = row["score_a"]
        sb = row["score_b"]
        winner = "A" if sa > sb else ("B" if sb > sa else "=")
        print(f"\n[{c['id']}] {c['name']} — A:{sa:+d} B:{sb:+d} [{winner}]")
        for d in row["details"]:
            dir_arrow = {"up": "↑", "down": "↓", "stay": "→"}.get(d["direction"], "?")
            mark_a = "OK" if d["score_a"] > 0 else ("NG" if d["score_a"] < 0 else "--")
            mark_b = "OK" if d["score_b"] > 0 else ("NG" if d["score_b"] < 0 else "--")
            print(f"  {d['type']:7s} {dir_arrow} {d['disease']}: "
                  f"init={d['rank_init']} → A:{d['rank_a']}[{mark_a}] B:{d['rank_b']}[{mark_b}]")

    # サマリー
    print("\n" + "=" * 70)
    print("サマリー")
    print("=" * 70)
    print(f"  A方式 (sim_matrix+centering): 合計 {total_a:+d}")
    print(f"  B方式 (所見バンドル):          合計 {total_b:+d}")
    print(f"  A勝ち: {a_wins}  B勝ち: {b_wins}  引分: {ties}")
    print(f"  差分: B - A = {total_b - total_a:+d}")

    # B方式がNGだった項目
    b_ngs = []
    for row in details:
        for d in row["details"]:
            if d["score_b"] < 0:
                b_ngs.append({"case_id": row["case"]["id"], **d})
    if b_ngs:
        print(f"\n--- B方式のNG ({len(b_ngs)}件) ---")
        for ng in b_ngs:
            dir_arrow = {"up": "↑", "down": "↓", "stay": "→"}.get(ng["direction"], "?")
            print(f"  [{ng['case_id']}] {ng['disease']} "
                  f"(期待{dir_arrow}): init={ng['rank_init']} → {ng['rank_b']}")


if __name__ == "__main__":
    main()
