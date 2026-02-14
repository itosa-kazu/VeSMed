"""
VeSMed - Web UI (Gradio)
v2アーキテクチャ: embedding知覚 + LLM判断

フロー:
  1. テキスト入力 → LLMで症状/検査結果を自動分離
  2. 症状 → embedding検索 → 候補疾患（知覚）
  3. LLM論理フィルタ → 矛盾疾患を除外（判断）
  4. 検査結果 → 極性判定 × 反実仮想で疾患重み更新
  5. Novelty → LLM二値判定
  6. 分散/命中/特異度 → 検査・問診ランキング
"""

import gradio as gr
import pandas as pd
from engine import VeSMedEngine
from config import TOP_K_TESTS

engine = None


def init_engine():
    global engine
    if engine is None:
        engine = VeSMedEngine()
    return engine


# ----------------------------------------------------------------
# コールバック
# ----------------------------------------------------------------

def analyze_patient(input_text):
    """
    v2パイプライン: embedding知覚 + LLM判断

    1. テキスト分離（LLM）
    2. 疾患検索（embedding — 知覚）
    3. 論理フィルタ（LLM — 判断: 矛盾疾患を除外）
    4. 検査結果更新（polarity + 反実仮想）
    5. Novelty（LLM二値判定）
    6. ランキング（数学 — 分散/命中/特異度）
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor
    if not input_text.strip():
        return (
            "テキストを入力してください。",
            None, None, None, None, None, None, None, None, None,
        )

    eng = init_engine()

    t0 = _time.time()

    # Step 0: 症状と検査結果を自動分離（LLM、3分類）
    positive_text, negative_findings, result_lines = eng.split_symptoms_results(input_text)
    t_split = _time.time()
    print(f"[TIMING] split: {t_split-t0:.1f}s")

    if not positive_text.strip():
        positive_text = input_text

    # Step 1: 疾患検索（embedding — 知覚、陽性所見のみ）
    candidates = eng.search_diseases(positive_text)
    t1 = _time.time()
    print(f"[TIMING] search_diseases: {t1-t_split:.1f}s")

    candidates = eng.compute_priors(candidates)

    # Step 2: 並行処理（LLMフィルタ, 検査結果更新, 統合novelty）
    full_text = input_text

    import copy
    cands_copy = copy.deepcopy(candidates)

    with ThreadPoolExecutor(max_workers=3) as executor:
        # LLM論理フィルタ: 矛盾疾患を除外（陰性所見も渡す）
        fut_filter = executor.submit(
            eng.filter_contradictions, candidates, full_text, negative_findings,
        )

        # 検査結果更新
        fut_update = executor.submit(
            eng.update_from_results, cands_copy, result_lines,
            positive_text,
        ) if result_lines else None

        # 統合Novelty: 検査novelty + HPE novelty + HPE所見抽出を1 LLMコールで
        fut_all_novelty = executor.submit(
            eng.compute_all_novelty, full_text,
        )

        # 結果回収
        novelty, novelty_hpe, hpe_findings = fut_all_novelty.result()

        filtered_candidates, exclusion_reasons = fut_filter.result()

        # 検査結果更新（フィルタ前のcopyで更新し、フィルタ後に反映）
        if fut_update:
            updated_cands = fut_update.result()
            updated_map = {c["disease_name"]: c for c in updated_cands}
            for c in filtered_candidates:
                if c["disease_name"] in updated_map:
                    c["similarity"] = updated_map[c["disease_name"]]["similarity"]

    candidates = filtered_candidates

    # HPE所見で疾患重み更新
    if hpe_findings:
        candidates = eng.update_from_hpe(candidates, hpe_findings)

    t2 = _time.time()
    print(f"[TIMING] parallel: {t2-t1:.1f}s")

    # Step 3: ランキング（数学 — 分散/命中/特異度/共通度）
    cluster_mu_tests = eng.rank_tests_cluster_mu(candidates, novelty=novelty)
    ranked_tests = eng.rank_tests(candidates, novelty=novelty)
    critical_tests = eng.rank_tests_critical(candidates, novelty=novelty)
    confirm_tests = eng.rank_tests_confirm(candidates, novelty=novelty)
    ranked_hpe = eng.rank_hpe(candidates, novelty_hpe=novelty_hpe)
    t3 = _time.time()
    print(f"[TIMING] rank: {t3-t2:.1f}s")
    print(f"[TIMING] === TOTAL: {t3-t0:.1f}s ===")

    n_results = len(result_lines)
    n_total = len(eng.disease_db)
    n_filtered = n_total - len(candidates)
    filter_info = f" / {n_filtered}疾患除外" if n_filtered > 0 else ""
    neg_info = f" / 陰性所見{len(negative_findings)}件" if negative_findings else ""
    status = f"分析完了 / {len(candidates)}疾患{filter_info}{neg_info} / 検査結果{n_results}件"

    # 除外疾患テーブル
    excluded_df = format_exclusion_df(exclusion_reasons)

    # 抑制検査テーブル（novelty=0の検査）
    suppressed_df = format_suppressed_df(eng.test_names, novelty)

    return (
        status,
        format_diseases_df(candidates),
        format_cluster_mu_df(cluster_mu_tests),
        format_tests_df(ranked_tests),
        format_critical_df(critical_tests),
        format_confirm_df(confirm_tests),
        format_hpe_df(ranked_hpe, "Hx"),
        format_hpe_df(ranked_hpe, "PE"),
        excluded_df,
        suppressed_df,
    )


def reset_session():
    return (
        "",
        "リセットしました。",
        None, None, None, None, None, None, None, None, None,
    )


# ----------------------------------------------------------------
# 表示フォーマット
# ----------------------------------------------------------------

def format_diseases_df(candidates):
    rows = []
    for i, c in enumerate(candidates[:15]):
        urgency = c.get("urgency", "")
        mark = {"超緊急": " !!!", "緊急": " !!", "準緊急": " !"}.get(urgency, "")
        rows.append({
            "#": i + 1,
            "疾患名": c["disease_name"],
            "類似度": f"{c.get('similarity', 0):.3f}",
            "重み": f"{c.get('clinical_weight', 0):.2f}",
            "緊急度": f"{urgency}{mark}",
            "診療科": c.get("category", ""),
        })
    return pd.DataFrame(rows)


def format_tests_df(ranked_tests):
    rows = []
    for i, t in enumerate(ranked_tests[:TOP_K_TESTS]):
        related = ", ".join(d["disease_name"] for d in t.get("details", [])[:3])
        if len(t.get("details", [])) > 3:
            related += f" 他{len(t['details']) - 3}件"
        rows.append({
            "#": i + 1,
            "検査名": t["test_name"],
            "効用": f"{t['utility']:.4f}",
            "分散": f"{t['score']:.4f}",
            "新規": f"{t.get('novelty', 1.0):.2f}",
            "関連疾患": related,
        })
    return pd.DataFrame(rows)


def format_critical_df(critical_tests):
    rows = []
    for i, t in enumerate(critical_tests[:TOP_K_TESTS]):
        rows.append({
            "#": i + 1,
            "検査名": t["test_name"],
            "効用": f"{t['utility']:.4f}",
            "命中": f"{t['critical_hit']:.4f}",
            "新規": f"{t.get('novelty', 1.0):.2f}",
            "排除対象": t.get("hit_disease", ""),
        })
    return pd.DataFrame(rows)


def format_confirm_df(confirm_tests):
    rows = []
    for i, t in enumerate(confirm_tests[:TOP_K_TESTS]):
        related = ", ".join(d["disease_name"] for d in t.get("details", [])[:3])
        if len(t.get("details", [])) > 3:
            related += f" 他{len(t['details']) - 3}件"
        rows.append({
            "#": i + 1,
            "検査名": t["test_name"],
            "効用": f"{t['utility']:.4f}",
            "特異": f"{t['confirm_score']:.4f}",
            "新規": f"{t.get('novelty', 1.0):.2f}",
            "関連疾患": related,
        })
    return pd.DataFrame(rows)


def format_cluster_mu_df(cluster_mu_tests):
    rows = []
    for i, t in enumerate(cluster_mu_tests[:TOP_K_TESTS]):
        related = ", ".join(d["disease_name"] for d in t.get("details", [])[:3])
        if len(t.get("details", [])) > 3:
            related += f" 他{len(t['details']) - 3}件"
        rows.append({
            "#": i + 1,
            "検査名": t["test_name"],
            "効用": f"{t['utility']:.4f}",
            "共通度": f"{t['cluster_mu']:.4f}",
            "新規": f"{t.get('novelty', 1.0):.2f}",
            "関連疾患": related,
        })
    return pd.DataFrame(rows)


def format_hpe_df(ranked_hpe, category_filter):
    """Part D: 問診(Hx)または身体診察(PE)のランキングをDataFrame化"""
    filtered = [r for r in ranked_hpe if r["category"] == category_filter]
    rows = []
    for i, r in enumerate(filtered[:TOP_K_TESTS]):
        related = ", ".join(d["disease_name"] for d in r.get("details", [])[:3])
        if len(r.get("details", [])) > 3:
            related += f" 他{len(r['details']) - 3}件"
        rows.append({
            "#": i + 1,
            "項目": r["item_name"],
            "分類": r["subcategory"],
            "効用": f"{r['utility']:.4f}",
            "分散": f"{r['score']:.4f}",
            "新規": f"{r.get('novelty', 1.0):.2f}",
            "手順": r.get("instruction", ""),
            "関連疾患": related,
        })
    return pd.DataFrame(rows)


def format_exclusion_df(exclusion_reasons):
    """除外された疾患のDataFrame"""
    if not exclusion_reasons:
        return pd.DataFrame(columns=["疾患名", "除外理由"])
    rows = []
    for r in exclusion_reasons:
        rows.append({
            "疾患名": r["disease_name"],
            "除外理由": r.get("reason", ""),
        })
    return pd.DataFrame(rows)


def format_suppressed_df(test_names, novelty):
    """novelty=0で抑制された検査のDataFrame"""
    import numpy as np
    rows = []
    for j, tname in enumerate(test_names):
        if j < len(novelty) and novelty[j] == 0.0:
            rows.append({
                "検査名": tname,
                "理由": "患者テキストから実施済みと判定",
            })
    if not rows:
        return pd.DataFrame(columns=["検査名", "理由"])
    return pd.DataFrame(rows)


# ----------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------

with gr.Blocks(
    title="VeSMed - ベクトル空間医学",
) as app:

    gr.Markdown("""
# VeSMed - ベクトル空間医学統一フレームワーク
臨床情報を自由記述 → 症状と検査結果を自動分離 → ベクトル空間で疾患検索・検査推奨
    """)

    # ===== 上段: 入力 =====
    with gr.Row():
        with gr.Column(scale=3):
            input_text = gr.Textbox(
                label="臨床情報（症状・所見・検査結果を自由に記述）",
                placeholder="例: 67歳男性。繰り返す発熱を主訴に来院。7週間前から38℃前後の発熱が出現。\n血液培養: 黄色ブドウ球菌陽性。γ-GTP正常。直接ビリルビン正常。",
                lines=6,
            )
            with gr.Row():
                analyze_btn = gr.Button("分析開始", variant="primary", scale=2)
                reset_btn = gr.Button("リセット", variant="secondary", scale=1)
        with gr.Column(scale=1):
            status_text = gr.Textbox(label="ステータス", interactive=False, lines=2)

    gr.Markdown("---")

    # ===== 中段: 結果テーブル =====
    with gr.Row():
        disease_table = gr.Dataframe(
            label="候補疾患（類似度順）",
            headers=["#", "疾患名", "類似度", "重み", "緊急度", "診療科"],
            interactive=False,
        )
    with gr.Row():
        cluster_mu_table = gr.Dataframe(
            label="Part E: 基本推奨（候補群の共通必要度）",
            headers=["#", "検査名", "効用", "共通度", "新規", "関連疾患"],
            interactive=False,
        )
    with gr.Row():
        with gr.Column(scale=1):
            test_table = gr.Dataframe(
                label="Part A: 鑑別推奨（分散ベース）",
                headers=["#", "検査名", "効用", "分散", "新規", "関連疾患"],
                interactive=False,
            )
        with gr.Column(scale=1):
            critical_table = gr.Dataframe(
                label="Part B: Critical排除推奨（最大命中）",
                headers=["#", "検査名", "効用", "命中", "新規", "排除対象"],
                interactive=False,
            )
    with gr.Row():
        confirm_table = gr.Dataframe(
            label="Part C: 確認・同定推奨（加重平均）",
            headers=["#", "検査名", "効用", "特異", "新規", "関連疾患"],
            interactive=False,
        )

    gr.Markdown("---")
    gr.Markdown("### Part D: 問診・身体診察推奨")

    with gr.Row():
        with gr.Column(scale=1):
            hpe_hx_table = gr.Dataframe(
                label="Part D-1: 問診推奨（分散ベース）",
                headers=["#", "項目", "分類", "効用", "分散", "新規", "手順", "関連疾患"],
                interactive=False,
            )
        with gr.Column(scale=1):
            hpe_pe_table = gr.Dataframe(
                label="Part D-2: 身体診察推奨（分散ベース）",
                headers=["#", "項目", "分類", "効用", "分散", "新規", "手順", "関連疾患"],
                interactive=False,
            )

    # ===== 除外・抑制情報（折りたたみ） =====
    gr.Markdown("---")
    with gr.Accordion("除外された疾患・抑制された検査", open=False):
        with gr.Row():
            with gr.Column(scale=1):
                excluded_table = gr.Dataframe(
                    label="除外疾患（LLMフィルタによる論理矛盾）",
                    headers=["疾患名", "除外理由"],
                    interactive=False,
                )
            with gr.Column(scale=1):
                suppressed_table = gr.Dataframe(
                    label="抑制検査（実施済みと判定）",
                    headers=["検査名", "理由"],
                    interactive=False,
                )

    # ===== イベント =====

    outputs = [status_text, disease_table, cluster_mu_table, test_table, critical_table,
               confirm_table, hpe_hx_table, hpe_pe_table, excluded_table, suppressed_table]

    analyze_btn.click(
        fn=analyze_patient,
        inputs=[input_text],
        outputs=outputs,
    )
    input_text.submit(
        fn=analyze_patient,
        inputs=[input_text],
        outputs=outputs,
    )

    reset_btn.click(
        fn=reset_session,
        outputs=[input_text] + outputs,
    )

    # Part D用: HPEエンジン情報表示
    if engine and engine.hpe_items:
        n_hx = sum(1 for it in engine.hpe_items if it['category'] == 'Hx')
        n_pe = sum(1 for it in engine.hpe_items if it['category'] == 'PE')
        print(f"[Web] Part D: 問診{n_hx}項目 + 身体診察{n_pe}項目 = {n_hx+n_pe}項目")


if __name__ == "__main__":
    print("エンジン初期化中...")
    init_engine()
    print(f"疾患DB: {len(engine.disease_db)}件 / ベクトルDB: {engine.collection.count()}件")
    print(f"名寄せマップ: {len(engine.test_name_map)}件")
    print("Web UI起動中...")
    app.launch(inbrowser=True, share=True)
