"""
VeSMed - Qwen3 Embedding ノルム実験
────────────────────────────────────
Qwen3-Embedding-8Bの生ノルム（L2ノルム）が
重症度・マグニチュード情報をエンコードしているか検証する。

仮説: 重症側のフレーズは生ノルムが高い（or低い）パターンを持つ
"""

import os
import sys
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# プロジェクトルートの.envを読み込む
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from config import EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL


def get_raw_embeddings(client: OpenAI, texts: list[str]) -> list[np.ndarray]:
    """テキストリストをembeddingし、生のベクトル（正規化前）を返す。"""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [np.array(item.embedding, dtype=np.float64) for item in resp.data]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """コサイン類似度を計算"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def main():
    print("=" * 72)
    print("Qwen3-Embedding-8B ノルム実験: 重症度はノルムに反映されるか？")
    print("=" * 72)
    print(f"モデル: {EMBEDDING_MODEL}")
    print(f"API: {EMBEDDING_BASE_URL}")
    print()

    client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)

    # ─── 実験ペア定義 ───
    # (重症側, 軽症側, 関連疾患コンセプト)
    pairs = [
        ("CRP著明上昇",        "CRP軽度上昇",        "敗血症"),
        ("白血球著増",          "白血球軽度上昇",      "敗血症"),
        ("高熱(39度以上)",      "微熱(37度台)",        "肺炎"),
        ("激しい腹痛",          "軽い腹痛",            "急性腹症"),
        ("呼吸困難",            "軽度の息切れ",        "肺塞栓症"),
        ("意識障害",            "軽度の傾眠",          "脳卒中"),
        ("大量出血",            "少量出血",            "出血性ショック"),
        ("CRP 200",            "CRP 2.0",             "敗血症"),
        ("白血球 30000",       "白血球 12000",         "白血病"),
        ("血圧60/40",          "血圧110/70",           "ショック"),
        ("SpO2 80%",           "SpO2 96%",            "呼吸不全"),
        ("GCS 3",              "GCS 14",              "意識障害"),
    ]

    # 全テキストを一括embed（API呼び出し回数を最小化）
    all_texts = []
    for severe, mild, disease in pairs:
        all_texts.extend([severe, mild, disease])

    print(f"Embedding取得中... ({len(all_texts)}テキスト)")
    all_embs = get_raw_embeddings(client, all_texts)
    print(f"Embedding次元: {len(all_embs[0])}")
    print()

    # ─── 結果表示 ───
    print("─" * 72)
    print(f"{'ペア':<12s} │ {'重症側':^20s} │ {'軽症側':^20s} │ {'ノルム差':>8s} │ {'cos類似':>7s}")
    print(f"{'':12s} │ {'テキスト':^10s} {'ノルム':>8s} │ {'テキスト':^10s} {'ノルム':>8s} │ {'重-軽':>8s} │ {'重↔軽':>7s}")
    print("─" * 72)

    severe_norms = []
    mild_norms = []
    severe_wins = 0  # 重症側のノルムが高い回数

    for i, (severe, mild, disease) in enumerate(pairs):
        idx_base = i * 3
        emb_severe = all_embs[idx_base]
        emb_mild = all_embs[idx_base + 1]
        emb_disease = all_embs[idx_base + 2]

        norm_severe = np.linalg.norm(emb_severe)
        norm_mild = np.linalg.norm(emb_mild)
        norm_diff = norm_severe - norm_mild
        cos_pair = cosine_sim(emb_severe, emb_mild)

        severe_norms.append(norm_severe)
        mild_norms.append(norm_mild)
        if norm_severe > norm_mild:
            severe_wins += 1

        marker = "▲" if norm_severe > norm_mild else "▽"

        print(f"  {severe:<16s} │ ノルム {norm_severe:8.4f}  │  {mild:<16s} │ ノルム {norm_mild:8.4f}  │ {norm_diff:+8.4f} {marker} │ {cos_pair:.4f}")

    print("─" * 72)
    print()

    # ─── 統計サマリー ───
    print("=" * 72)
    print("■ 統計サマリー")
    print("=" * 72)
    severe_norms = np.array(severe_norms)
    mild_norms = np.array(mild_norms)

    print(f"重症側ノルム: 平均={severe_norms.mean():.4f}, 標準偏差={severe_norms.std():.4f}, "
          f"最小={severe_norms.min():.4f}, 最大={severe_norms.max():.4f}")
    print(f"軽症側ノルム: 平均={mild_norms.mean():.4f}, 標準偏差={mild_norms.std():.4f}, "
          f"最小={mild_norms.min():.4f}, 最大={mild_norms.max():.4f}")
    print(f"重症側 > 軽症側: {severe_wins}/{len(pairs)} ({severe_wins/len(pairs)*100:.1f}%)")
    print(f"ノルム差(重-軽): 平均={np.mean(severe_norms - mild_norms):+.4f}, "
          f"標準偏差={np.std(severe_norms - mild_norms):.4f}")
    print()

    # 対応のあるt検定
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(severe_norms, mild_norms)
    print(f"対応のあるt検定: t={t_stat:.4f}, p={p_value:.4f}")
    if p_value < 0.05:
        direction = "重症側が高い" if t_stat > 0 else "軽症側が高い"
        print(f"  → 有意差あり (p < 0.05): ノルムは{direction}傾向")
    else:
        print(f"  → 有意差なし (p >= 0.05): ノルムと重症度に系統的関係なし")
    print()

    # ─── 疾患コンセプトとのコサイン類似度比較 ───
    print("=" * 72)
    print("■ 疾患コンセプトとのコサイン類似度比較")
    print("  重症所見の方が関連疾患に近いか？")
    print("=" * 72)
    print(f"{'関連疾患':<16s} │ {'重症側':^20s} cos │ {'軽症側':^20s} cos │ {'差(重-軽)':>10s}")
    print("─" * 72)

    severe_closer = 0
    for i, (severe, mild, disease) in enumerate(pairs):
        idx_base = i * 3
        emb_severe = all_embs[idx_base]
        emb_mild = all_embs[idx_base + 1]
        emb_disease = all_embs[idx_base + 2]

        cos_severe_disease = cosine_sim(emb_severe, emb_disease)
        cos_mild_disease = cosine_sim(emb_mild, emb_disease)
        cos_diff = cos_severe_disease - cos_mild_disease

        if cos_severe_disease > cos_mild_disease:
            severe_closer += 1

        marker = "▲" if cos_severe_disease > cos_mild_disease else "▽"

        print(f"  {disease:<14s} │ {severe:<16s} {cos_severe_disease:.4f} │ {mild:<16s} {cos_mild_disease:.4f} │ {cos_diff:+.4f} {marker}")

    print("─" * 72)
    print(f"重症側が疾患に近い: {severe_closer}/{len(pairs)} ({severe_closer/len(pairs)*100:.1f}%)")
    print()

    # ─── 正規化前後でのコサイン類似度は同一確認 ───
    print("=" * 72)
    print("■ 補足: 正規化前後のコサイン類似度（同一性確認）")
    print("=" * 72)
    sample_a = all_embs[0]
    sample_b = all_embs[1]
    cos_raw = cosine_sim(sample_a, sample_b)

    a_normed = sample_a / np.linalg.norm(sample_a)
    b_normed = sample_b / np.linalg.norm(sample_b)
    cos_normed = float(np.dot(a_normed, b_normed))

    print(f"生ベクトル間のcos: {cos_raw:.10f}")
    print(f"正規化後のcos:     {cos_normed:.10f}")
    print(f"差: {abs(cos_raw - cos_normed):.2e} （数値誤差レベルなら正規化でノルム情報は完全に失われる）")
    print()

    # ─── ノルム分布のヒストグラム（テキスト） ───
    print("=" * 72)
    print("■ 全ベクトルのノルム分布")
    print("=" * 72)
    all_norms = np.array([np.linalg.norm(e) for e in all_embs])
    print(f"全体: 平均={all_norms.mean():.4f}, 標準偏差={all_norms.std():.4f}, "
          f"最小={all_norms.min():.4f}, 最大={all_norms.max():.4f}")
    print(f"変動係数(CV): {all_norms.std() / all_norms.mean() * 100:.2f}%")
    print()
    if all_norms.std() / all_norms.mean() < 0.01:
        print("→ ノルムの変動は極めて小さい（CV < 1%）。")
        print("  このモデルはほぼ単位球面上に射影しており、ノルムに情報はない可能性が高い。")
    elif all_norms.std() / all_norms.mean() < 0.05:
        print("→ ノルムに若干の変動あり（1% < CV < 5%）。")
        print("  テキスト長や内容に応じた微妙な変動がある可能性。")
    else:
        print("→ ノルムに有意な変動あり（CV > 5%）。")
        print("  ノルムがマグニチュード情報をエンコードしている可能性あり。")

    # ─── 結論 ───
    print()
    print("=" * 72)
    print("■ 結論")
    print("=" * 72)
    if p_value >= 0.05:
        print("ノルムと重症度の間に統計的に有意な関係は検出されなかった。")
        print("Qwen3-Embedding-8Bの生ノルムは重症度のプロキシとして使えない。")
    else:
        if t_stat > 0:
            print("重症側のノルムが統計的に有意に高い傾向が検出された。")
            print("ただし効果量と実用性は別途検討が必要。")
        else:
            print("軽症側のノルムが統計的に有意に高い傾向が検出された。")
            print("直感に反する結果であり、ノルムは重症度以外の要因を反映している可能性。")

    if severe_closer > len(pairs) * 0.7:
        print("一方、コサイン類似度は重症所見が関連疾患に近い傾向を示す。")
        print("→ 重症度情報はノルムではなく「方向」にエンコードされている。")
    print()


if __name__ == "__main__":
    main()
