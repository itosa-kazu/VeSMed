"""
MAX-chunk vs MEAN-chunk 集約方法の比較テスト

比較:
  MEAN: 疾患の全ドメインチャンクembeddingを平均 → クエリとcos類似度
  MAX:  クエリと各ドメインチャンクのcos類似度を計算 → 最大値を採用

テストケース設計:
  各テストケースに「正解疾患」(期待上位) と「クエリテキスト」を定義。
  MEAN/MAXそれぞれでの正解疾患の順位・スコアを比較する。
"""

import json
import numpy as np
from collections import defaultdict
from openai import OpenAI
from config import (
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
    CHROMA_DIR, DISEASES_JSONL,
)

# ─── テストケース定義 ───
# 各ケース: query=患者テキスト, expected=上位に来るべき疾患(部分一致)
TEST_CASES = [
    {
        "name": "急性虫垂炎（典型）",
        "query": "右下腹部痛 発熱 嘔気 McBurney圧痛 反跳痛",
        "expected": ["急性虫垂炎"],
        "domain_hint": "typical+physical",  # どのドメインが効くべきか
    },
    {
        "name": "心筋梗塞（典型）",
        "query": "胸痛 冷汗 左肩放散痛 ST上昇 トロポニン上昇",
        "expected": ["急性冠症候群", "急性心筋梗塞", "ST上昇型心筋梗塞"],
        "domain_hint": "typical+tests",
    },
    {
        "name": "糖尿病性ケトアシドーシス（検査所見中心）",
        "query": "高血糖 代謝性アシドーシス ケトン体陽性 Kussmaul呼吸 脱水",
        "expected": ["糖尿病性ケトアシドーシス", "DKA"],
        "domain_hint": "tests+physical",
    },
    {
        "name": "肺塞栓症（非典型: 失神のみ）",
        "query": "突然の失神 頻脈 低酸素血症 長時間座位後",
        "expected": ["肺塞栓", "肺血栓塞栓症"],
        "domain_hint": "atypical",
    },
    {
        "name": "SLE（多系統、背景＋検査）",
        "query": "若年女性 蝶形紅斑 関節痛 蛋白尿 抗核抗体陽性 汎血球減少",
        "expected": ["全身性エリテマトーデス", "SLE"],
        "domain_hint": "background+tests+physical",
    },
    {
        "name": "甲状腺機能亢進症（病態生理中心）",
        "query": "動悸 体重減少 手指振戦 発汗過多 甲状腺腫大 TSH低値 FT4高値",
        "expected": ["甲状腺機能亢進症", "Basedow", "バセドウ"],
        "domain_hint": "pathophysiology+tests",
    },
    {
        "name": "髄膜炎（身体所見中心）",
        "query": "発熱 頭痛 項部硬直 Kernig徴候陽性 Brudzinski徴候陽性",
        "expected": ["髄膜炎", "細菌性髄膜炎"],
        "domain_hint": "physical",
    },
    {
        "name": "大動脈解離（非典型: 背部痛）",
        "query": "突然発症の背部痛 引き裂かれるような痛み 左右血圧差 高血圧の既往",
        "expected": ["大動脈解離"],
        "domain_hint": "typical+atypical",
    },
    {
        "name": "急性膵炎（検査＋症状）",
        "query": "心窩部痛 背部放散痛 嘔吐 アミラーゼ上昇 リパーゼ上昇 飲酒歴",
        "expected": ["急性膵炎"],
        "domain_hint": "typical+tests+background",
    },
    {
        "name": "鉄欠乏性貧血（背景＋検査のみ）",
        "query": "若年女性 月経過多 倦怠感 MCV低値 フェリチン低値 血清鉄低値",
        "expected": ["鉄欠乏性貧血"],
        "domain_hint": "background+tests",
    },
    {
        "name": "ギラン・バレー（病態生理＋典型）",
        "query": "感冒後の四肢脱力 上行性麻痺 深部腱反射消失 蛋白細胞解離",
        "expected": ["ギラン・バレー", "Guillain-Barré"],
        "domain_hint": "pathophysiology+typical",
    },
    {
        "name": "アナフィラキシー（身体所見のみ）",
        "query": "蕁麻疹 呼吸困難 血圧低下 喉頭浮腫 薬剤投与直後",
        "expected": ["アナフィラキシー"],
        "domain_hint": "physical+typical",
    },
]


def load_chromadb_chunks():
    """ChromaDBから全チャンクを読み込む。"""
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("diseases")
    all_data = collection.get(include=["embeddings", "metadatas"])

    disease_chunks = defaultdict(list)  # disease_name -> [(domain, embedding)]
    for j in range(len(all_data["ids"])):
        meta = all_data["metadatas"][j]
        dname = meta.get("disease_name", "")
        domain = meta.get("domain", "unknown")
        emb = np.array(all_data["embeddings"][j], dtype=np.float32)
        if dname:
            disease_chunks[dname].append((domain, emb))

    return disease_chunks


def compute_mean_embeddings(disease_chunks):
    """MEAN集約: 全チャンクの平均embedding (正規化済み)。"""
    disease_names = sorted(disease_chunks.keys())
    embs = []
    for name in disease_names:
        chunks = np.array([e for _, e in disease_chunks[name]], dtype=np.float32)
        mean_emb = chunks.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb /= norm
        embs.append(mean_emb)
    return disease_names, np.array(embs, dtype=np.float32)


def embed_texts(texts):
    """テキストをembedding。"""
    client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # 正規化
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def rank_mean(query_emb, disease_names, disease_embs_normed):
    """MEAN方式: query_emb × disease_embs_normed → スコア順。"""
    sims = query_emb @ disease_embs_normed.T
    order = np.argsort(-sims)
    return [(disease_names[i], float(sims[i])) for i in order]


def rank_max(query_emb, disease_chunks, disease_names_sorted):
    """MAX方式: 各疾患について全チャンクとのcos類似度の最大値。"""
    scores = {}
    for dname in disease_names_sorted:
        chunks = disease_chunks[dname]
        max_sim = -1.0
        for domain, emb in chunks:
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb_n = emb / norm
            else:
                emb_n = emb
            sim = float(query_emb @ emb_n)
            if sim > max_sim:
                max_sim = sim
        scores[dname] = max_sim
    order = sorted(scores.items(), key=lambda x: -x[1])
    return order


def find_rank(ranking, expected_names):
    """
    expected_namesのいずれかに部分一致する最上位の順位とスコアを返す。
    """
    for rank, (dname, score) in enumerate(ranking, 1):
        for exp in expected_names:
            if exp in dname or dname in exp:
                return rank, dname, score
    return None, None, None


def main():
    print("=" * 80)
    print("MAX-chunk vs MEAN-chunk 比較テスト")
    print("=" * 80)

    # ChromaDB読み込み
    print("\n[1] ChromaDB読み込み...")
    disease_chunks = load_chromadb_chunks()
    n_diseases = len(disease_chunks)
    n_chunks_total = sum(len(v) for v in disease_chunks.values())
    print(f"    {n_diseases}疾患, {n_chunks_total}チャンク")

    # チャンク数の分布
    chunk_counts = [len(v) for v in disease_chunks.values()]
    print(f"    チャンク数: mean={np.mean(chunk_counts):.1f}, "
          f"min={min(chunk_counts)}, max={max(chunk_counts)}")

    # MEAN embedding計算
    print("\n[2] MEAN embedding計算...")
    disease_names, disease_embs_mean = compute_mean_embeddings(disease_chunks)
    print(f"    {disease_embs_mean.shape}")

    # クエリembedding
    print("\n[3] クエリembedding...")
    queries = [tc["query"] for tc in TEST_CASES]
    query_embs = embed_texts(queries)
    print(f"    {len(queries)}件embed完了")

    # 各テストケースで比較
    print("\n" + "=" * 80)
    print("テスト結果")
    print("=" * 80)

    mean_wins = 0
    max_wins = 0
    ties = 0
    results = []

    for i, tc in enumerate(TEST_CASES):
        q_emb = query_embs[i]

        ranking_mean = rank_mean(q_emb, disease_names, disease_embs_mean)
        ranking_max = rank_max(q_emb, disease_chunks, disease_names)

        rank_m, name_m, score_m = find_rank(ranking_mean, tc["expected"])
        rank_x, name_x, score_x = find_rank(ranking_max, tc["expected"])

        # 勝者判定
        if rank_m is not None and rank_x is not None:
            if rank_m < rank_x:
                winner = "MEAN"
                mean_wins += 1
            elif rank_x < rank_m:
                winner = "MAX"
                max_wins += 1
            else:
                winner = "TIE"
                ties += 1
        elif rank_m is not None:
            winner = "MEAN"
            mean_wins += 1
        elif rank_x is not None:
            winner = "MAX"
            max_wins += 1
        else:
            winner = "NONE"

        results.append({
            "name": tc["name"],
            "domain_hint": tc["domain_hint"],
            "mean_rank": rank_m,
            "mean_name": name_m,
            "mean_score": score_m,
            "max_rank": rank_x,
            "max_name": name_x,
            "max_score": score_x,
            "winner": winner,
        })

        print(f"\n--- [{i+1}] {tc['name']} ---")
        print(f"  クエリ: {tc['query'][:60]}...")
        print(f"  期待: {tc['expected']}")
        sm = f"{score_m:.4f}" if score_m is not None else "N/A"
        sx = f"{score_x:.4f}" if score_x is not None else "N/A"
        print(f"  MEAN: 順位={rank_m}, スコア={sm}  ({name_m})")
        print(f"  MAX:  順位={rank_x}, スコア={sx}  ({name_x})")
        print(f"  勝者: {winner}")

        # TOP5も表示
        print(f"  [MEAN TOP5]", end="")
        for r, (dn, sc) in enumerate(ranking_mean[:5], 1):
            print(f"  {r}.{dn}({sc:.4f})", end="")
        print()
        print(f"  [MAX  TOP5]", end="")
        for r, (dn, sc) in enumerate(ranking_max[:5], 1):
            print(f"  {r}.{dn}({sc:.4f})", end="")
        print()

    # サマリー
    print("\n" + "=" * 80)
    print("サマリー")
    print("=" * 80)
    print(f"  MEAN勝利: {mean_wins}")
    print(f"  MAX勝利:  {max_wins}")
    print(f"  引き分け: {ties}")
    print(f"  未検出:   {len(TEST_CASES) - mean_wins - max_wins - ties}")

    # スコア差の統計
    score_diffs = []  # positive = MAX > MEAN
    rank_diffs = []   # positive = MEAN better (lower rank)
    for r in results:
        if r["mean_score"] and r["max_score"]:
            score_diffs.append(r["max_score"] - r["mean_score"])
        if r["mean_rank"] and r["max_rank"]:
            rank_diffs.append(r["max_rank"] - r["mean_rank"])

    if score_diffs:
        print(f"\n  スコア差 (MAX - MEAN):")
        print(f"    mean={np.mean(score_diffs):.4f}, "
              f"median={np.median(score_diffs):.4f}, "
              f"min={np.min(score_diffs):.4f}, max={np.max(score_diffs):.4f}")
    if rank_diffs:
        print(f"  順位差 (MAX順位 - MEAN順位, 正=MEANが上位):")
        print(f"    mean={np.mean(rank_diffs):.1f}, "
              f"median={np.median(rank_diffs):.1f}, "
              f"min={np.min(rank_diffs)}, max={np.max(rank_diffs)}")

    # ドメイン特化分析
    print(f"\n  ドメイン別分析:")
    for r in results:
        marker = "✓" if r["winner"] in ("MEAN", "TIE") else "✗"
        if r["winner"] == "MAX":
            marker = "←MAX"
        elif r["winner"] == "MEAN":
            marker = "←MEAN"
        print(f"    {r['name']:30s}  domain={r['domain_hint']:30s}  "
              f"MEAN#{r['mean_rank']} vs MAX#{r['max_rank']}  {marker}")

    # MAX独自の追加分析: 各疾患のチャンク間分散
    print(f"\n  疾患別チャンク間類似度分散（MAXとMEANの乖離指標）:")
    variances = []
    for dname in disease_names:
        chunks = disease_chunks[dname]
        if len(chunks) < 2:
            continue
        embs = []
        for _, e in chunks:
            n = np.linalg.norm(e)
            embs.append(e / n if n > 0 else e)
        embs = np.array(embs)
        # チャンク間のcos類似度の平均
        cos_mat = embs @ embs.T
        n = len(embs)
        off_diag = [cos_mat[i][j] for i in range(n) for j in range(i+1, n)]
        if off_diag:
            variances.append((dname, 1.0 - np.mean(off_diag)))  # 1-mean_cos = 多様性
    variances.sort(key=lambda x: -x[1])
    print(f"    チャンク多様性 TOP10（1-mean_cos, 高い=ドメイン間差が大きい=MAX有利）:")
    for dn, v in variances[:10]:
        print(f"      {dn:40s}  diversity={v:.4f}")
    print(f"    チャンク多様性 BOTTOM10（低い=ドメイン間差が小さい=MEAN≈MAX）:")
    for dn, v in variances[-10:]:
        print(f"      {dn:40s}  diversity={v:.4f}")
    print(f"    全体: mean={np.mean([v for _,v in variances]):.4f}, "
          f"median={np.median([v for _,v in variances]):.4f}")


if __name__ == "__main__":
    main()
