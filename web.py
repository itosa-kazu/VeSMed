"""
VeSMed - Web UI (Gradio)
ブラウザから操作できる対話型インターフェース

フロー:
  1. テキスト入力 → LLM/Embeddingで症状と検査結果を自動分離
  2. 症状 → ベクトル検索 → 候補疾患
  3. 検査結果 → 極性判定 × 類似度行列で疾患重み更新
  4. 全テキスト → novelty計算 → 検査ランキング
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

def analyze_patient(input_text, mode):
    """統合テキストから自動分離して分析"""
    import time as _time
    from concurrent.futures import ThreadPoolExecutor
    if not input_text.strip():
        return (
            "テキストを入力してください。",
            None, None, None, None, None, None,
        )

    eng = init_engine()
    engine_mode = "llm" if "LLM" in mode else "fast"

    t0 = _time.time()

    # Step 0: 症状と検査結果を自動分離
    symptoms_text, result_lines = eng.split_symptoms_results(
        input_text, mode=engine_mode,
    )
    t_split = _time.time()
    print(f"[TIMING] split: {t_split-t0:.1f}s")

    if not symptoms_text.strip():
        # 分離で症状が空になった場合、全テキストを症状として使う
        symptoms_text = input_text

    # Step 1: 症状で疾患検索
    candidates = eng.search_diseases(symptoms_text)
    t1 = _time.time()
    print(f"[TIMING] search_diseases: {t1-t_split:.1f}s")

    candidates = eng.compute_priors(candidates)

    # Step 2+3: 検査結果更新 と novelty計算を並行実行
    full_text = input_text  # noveltyは全テキストで計算

    import copy
    cands_copy = copy.deepcopy(candidates)

    with ThreadPoolExecutor(max_workers=4) as executor:
        fut_update = executor.submit(
            eng.update_from_results, cands_copy, result_lines,
            symptoms_text, engine_mode,
        ) if result_lines else None
        fut_novelty = executor.submit(eng.compute_novelty, full_text)
        fut_hpe_extract = executor.submit(
            eng.extract_hpe_findings, full_text,
        ) if engine_mode == "llm" else None
        fut_novelty_hpe = executor.submit(eng.compute_novelty_hpe, full_text)

        novelty = fut_novelty.result()
        hpe_findings = fut_hpe_extract.result() if fut_hpe_extract else []
        novelty_hpe_base = fut_novelty_hpe.result()
        candidates = fut_update.result() if fut_update else candidates

    # HPE所見でnovelty上書き + 疾患重み更新
    if hpe_findings:
        for f in hpe_findings:
            novelty_hpe_base[f["index"]] = 0.0
        candidates = eng.update_from_hpe(candidates, hpe_findings)
    novelty_hpe = novelty_hpe_base

    t2 = _time.time()
    print(f"[TIMING] update+novelty+hpe parallel ({engine_mode}): {t2-t1:.1f}s")

    # Step 4: 検査ランキング
    ranked_tests = eng.rank_tests(candidates, novelty=novelty)
    critical_tests = eng.rank_tests_critical(candidates, novelty=novelty)
    confirm_tests = eng.rank_tests_confirm(candidates, novelty=novelty)
    ranked_hpe = eng.rank_hpe(candidates, novelty_hpe=novelty_hpe)
    t3 = _time.time()
    print(f"[TIMING] rank_tests+critical+confirm+hpe: {t3-t2:.1f}s")
    print(f"[TIMING] === TOTAL ({engine_mode}): {t3-t0:.1f}s ===")

    n_results = len(result_lines)
    mode_label = "LLM注釈" if engine_mode == "llm" else "高速"
    status = f"分析完了（{mode_label}）/ 全{len(candidates)}疾患 / 検査結果{n_results}件自動検出"

    return (
        status,
        format_diseases_df(candidates),
        format_tests_df(ranked_tests),
        format_critical_df(critical_tests),
        format_confirm_df(confirm_tests),
        format_hpe_df(ranked_hpe, "Hx"),
        format_hpe_df(ranked_hpe, "PE"),
    )


def reset_session():
    return (
        "",
        "リセットしました。",
        None, None, None, None, None, None,
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
            mode_radio = gr.Radio(
                choices=["高速（基準範囲）", "LLM注釈（文脈考慮）"],
                value="高速（基準範囲）",
                label="結果解釈モード",
            )
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

    # ===== イベント =====

    outputs = [status_text, disease_table, test_table, critical_table, confirm_table,
               hpe_hx_table, hpe_pe_table]

    analyze_btn.click(
        fn=analyze_patient,
        inputs=[input_text, mode_radio],
        outputs=outputs,
    )
    input_text.submit(
        fn=analyze_patient,
        inputs=[input_text, mode_radio],
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
