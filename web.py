"""
VeSMed - Web UI (Gradio)
ブラウザから操作できる対話型インターフェース

フロー（Option 3: 症状と検査結果を分離）:
  1. 症状入力 → ベクトル検索 → 候補疾患（初回、固定）
  2. 検査結果入力 → sign(polarity) × sim_matrix で疾患重み更新
  3. 全テキスト（症状+結果）→ novelty計算 → 検査ランキング
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

def analyze_patient(symptoms_text, results_text, mode):
    """症状と検査結果を分離して分析（Option 3）"""
    import time as _time
    from concurrent.futures import ThreadPoolExecutor
    if not symptoms_text.strip():
        return (
            "症状を入力してください。",
            None, None, None, None, None, None,
        )

    eng = init_engine()
    # モード変換: UIラベル → engine引数
    engine_mode = "llm" if "LLM" in mode else "fast"

    t0 = _time.time()

    # Step 1: 症状で疾患検索（初回、固定）
    candidates = eng.search_diseases(symptoms_text)
    t1 = _time.time()
    print(f"[TIMING] search_diseases: {t1-t0:.1f}s")

    candidates = eng.compute_priors(candidates)

    # Step 2+3: 検査結果更新 と novelty計算を並行実行
    result_lines = [l.strip() for l in results_text.split('\n') if l.strip()]
    full_text = symptoms_text
    if result_lines:
        full_text += '\n' + '\n'.join(result_lines)

    import copy
    cands_copy = copy.deepcopy(candidates)

    with ThreadPoolExecutor(max_workers=3) as executor:
        # update_from_results, compute_novelty, compute_novelty_hpe は全て独立（並行可能）
        fut_update = executor.submit(
            eng.update_from_results, cands_copy, result_lines,
            symptoms_text, engine_mode,
        ) if result_lines else None
        fut_novelty = executor.submit(eng.compute_novelty, full_text)
        fut_novelty_hpe = executor.submit(eng.compute_novelty_hpe, full_text)

        novelty = fut_novelty.result()
        novelty_hpe = fut_novelty_hpe.result()
        candidates = fut_update.result() if fut_update else candidates

    t2 = _time.time()
    print(f"[TIMING] update+novelty+novelty_hpe parallel ({engine_mode}): {t2-t1:.1f}s")

    # Step 4: 検査ランキング + Part D（CPU演算のみ、高速）
    ranked_tests = eng.rank_tests(candidates, novelty=novelty)
    critical_tests = eng.rank_tests_critical(candidates, novelty=novelty)
    confirm_tests = eng.rank_tests_confirm(candidates, novelty=novelty)
    ranked_hpe = eng.rank_hpe(candidates, novelty_hpe=novelty_hpe)
    t3 = _time.time()
    print(f"[TIMING] rank_tests+critical+confirm+hpe: {t3-t2:.1f}s")
    print(f"[TIMING] === TOTAL ({engine_mode}): {t3-t0:.1f}s ===")

    n_results = len(result_lines)
    mode_label = "LLM注釈" if engine_mode == "llm" else "高速"
    status = f"分析完了（{mode_label}）/ 全{len(candidates)}疾患 / 検査結果{n_results}件反映"

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
        "", "",
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
症状 → ベクトル検索 → 候補疾患。検査結果 → 極性判定 × 類似度行列で疾患重み更新。
    """)

    # ===== 上段: 入力 =====
    with gr.Row():
        with gr.Column(scale=3):
            symptoms_input = gr.Textbox(
                label="症状（自由記述 — 疾患検索の基盤）",
                placeholder="例: 67歳の男性。繰り返す発熱を主訴に来院。7週間前から38℃前後の発熱が出現し、市販の解熱薬で一時的に解熱するが再度発熱する。",
                lines=4,
            )
            results_input = gr.Textbox(
                label="検査結果（1行1件 — 極性×類似度行列で疾患重み更新）",
                placeholder="例:\nγ-GTP正常値\n直接ビリルビン正常値\nMRCP異常なし\n血液培養: 黄色ブドウ球菌陽性",
                lines=4,
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
        inputs=[symptoms_input, results_input, mode_radio],
        outputs=outputs,
    )
    symptoms_input.submit(
        fn=analyze_patient,
        inputs=[symptoms_input, results_input, mode_radio],
        outputs=outputs,
    )

    reset_btn.click(
        fn=reset_session,
        outputs=[symptoms_input, results_input] + outputs,
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
