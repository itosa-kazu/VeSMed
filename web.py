"""
VeSMed - Web UI (Gradio)
ブラウザから操作できる対話型インターフェース

フロー:
  1. 患者情報入力 → ベクトル検索 → 候補疾患 + 推薦検査を表示
  2. 検査所見入力 → 患者テキスト+全履歴で再ベクトル検索 → 候補・検査を更新
  3. 繰り返し
"""

from concurrent.futures import ThreadPoolExecutor
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

def analyze_patient(patient_text, state):
    """患者情報を分析（ベクトル検索）"""
    if not patient_text.strip():
        return (
            "患者情報を入力してください。",
            None, None,
            gr.update(choices=[], value=None), "",
            state, "",
        )

    eng = init_engine()

    # rewrite_query と extract_done_tests は独立 → 並行実行で ~40秒短縮
    with ThreadPoolExecutor(max_workers=2) as ex:
        future_rewrite = ex.submit(eng.rewrite_query, patient_text)
        future_done = ex.submit(eng.extract_done_tests, patient_text)
        rewritten = future_rewrite.result()
        raw_done = future_done.result()

    candidates = eng.search_diseases(rewritten, original_text=patient_text)
    candidates = eng.compute_priors(candidates)

    # 名寄せ（candidates依存のため並行化できない）
    done_tests = eng.normalize_done_tests(raw_done, candidates)

    ranked_tests = eng.rank_tests(candidates, done_tests=done_tests)
    available_tests = eng.find_matching_tests(candidates, done_tests=done_tests)

    new_state = {
        "patient_text": patient_text,
        "candidates": candidates,
        "ranked_tests": ranked_tests,
        "done_tests": done_tests,
        "history": [],
    }

    status = f"初回分析完了 / 全{len(candidates)}疾患で計算 / 推薦検査{len(ranked_tests)}件 / 既実施{len(done_tests)}件除外"

    return (
        status,
        format_diseases_df(candidates),
        format_tests_df(ranked_tests),
        gr.update(choices=available_tests, value=None), "",
        new_state,
        "",
    )


def apply_finding(test_name, finding_text, state):
    """検査所見 → 患者テキスト+全履歴で再ベクトル検索"""
    if not state or "patient_text" not in state:
        return (
            "先に患者情報を分析してください。",
            None, None,
            gr.update(choices=[], value=None), "",
            state, "",
        )

    if not test_name:
        return (
            "検査を選択してください。",
            format_diseases_df(state["candidates"]),
            format_tests_df(state["ranked_tests"]),
            gr.update(), "",
            state,
            format_history(state.get("history", [])),
        )

    if not finding_text.strip():
        return (
            "所見を入力してください。",
            format_diseases_df(state["candidates"]),
            format_tests_df(state["ranked_tests"]),
            gr.update(), "",
            state,
            format_history(state.get("history", [])),
        )

    eng = init_engine()

    # 検査履歴を更新
    done_tests = state.get("done_tests", []) + [test_name]
    history = state.get("history", []) + [{
        "test": test_name,
        "finding": finding_text.strip(),
    }]

    # 患者テキスト + 全検査履歴を結合して再ベクトル検索
    patient_text = state["patient_text"]
    findings_text = "\n".join(
        f"【{h['test']}】{h['finding']}" for h in history
    )
    updated_text = f"{patient_text}\n\n既実施検査結果:\n{findings_text}"

    rewritten = eng.rewrite_query(updated_text)
    candidates = eng.search_diseases(rewritten, original_text=updated_text)
    candidates = eng.compute_priors(candidates)
    ranked_tests = eng.rank_tests(candidates, done_tests=done_tests)
    available_tests = eng.find_matching_tests(candidates, done_tests=done_tests)

    new_state = {
        "patient_text": patient_text,
        "candidates": candidates,
        "ranked_tests": ranked_tests,
        "done_tests": done_tests,
        "history": history,
    }

    status = f"【{test_name}】所見反映済み / 実施済み{len(done_tests)}件"

    return (
        status,
        format_diseases_df(candidates),
        format_tests_df(ranked_tests),
        gr.update(choices=available_tests, value=None), "",
        new_state,
        format_history(history),
    )


def reset_session():
    return (
        "",
        "リセットしました。",
        None, None,
        gr.update(choices=[], value=None), "",
        {},
        "",
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
            "確率": f"{c.get('prior', 0):.1%}",
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
            "質": f"{t['quality']:.4f}",
            "関連疾患": related,
        })
    return pd.DataFrame(rows)


def format_history(history):
    if not history:
        return ""
    lines = []
    for i, entry in enumerate(history):
        lines.append(f"**{i + 1}. {entry['test']}**")
        lines.append(f"  所見: {entry['finding']}")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------

with gr.Blocks(
    title="VeSMed - ベクトル空間医学",
) as app:

    gr.Markdown("""
# VeSMed - ベクトル空間医学統一フレームワーク
患者情報 → 候補疾患（ベクトル検索） → 推薦検査（情報利得） → 所見入力 → 再ベクトル検索 → 繰り返し
    """)

    state = gr.State({})

    # ===== 上段: 患者情報 =====
    with gr.Row():
        with gr.Column(scale=3):
            patient_input = gr.Textbox(
                label="患者情報（自由記述）",
                placeholder="例: 67歳の男性。繰り返す発熱を主訴に来院。7週間前から38℃前後の発熱が出現し、市販の解熱薬で一時的に解熱するが再度発熱する。",
                lines=4,
            )
            with gr.Row():
                analyze_btn = gr.Button("分析開始", variant="primary", scale=2)
                reset_btn = gr.Button("リセット", variant="secondary", scale=1)
        with gr.Column(scale=1):
            status_text = gr.Textbox(label="ステータス", interactive=False, lines=2)

    gr.Markdown("---")

    # ===== 中段: 結果テーブル =====
    with gr.Row():
        with gr.Column(scale=1):
            disease_table = gr.Dataframe(
                label="候補疾患（確率順）",
                headers=["#", "疾患名", "確率", "重み", "緊急度", "診療科"],
                interactive=False,
            )
        with gr.Column(scale=1):
            test_table = gr.Dataframe(
                label="推薦検査（効用 = 情報利得 / コスト 降順）",
                headers=["#", "検査名", "効用", "情報利得", "コスト", "関連疾患"],
                interactive=False,
            )

    gr.Markdown("---")

    # ===== 下段: 検査所見入力 =====
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 検査所見を入力")
            test_selector = gr.Dropdown(
                label="検査名（選択 or 直接入力）",
                choices=[],
                allow_custom_value=True,
                interactive=True,
            )
            finding_input = gr.Textbox(
                label="所見（自由記述）",
                placeholder="例: 疣贅あり、大動脈弁逆流あり / 異常なし / ST上昇 V1-V4",
                lines=2,
            )
            submit_btn = gr.Button("所見を反映", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 検査履歴")
            history_display = gr.Markdown("")

    # ===== イベント =====

    outputs = [status_text, disease_table, test_table, test_selector, finding_input, state, history_display]

    analyze_btn.click(
        fn=analyze_patient,
        inputs=[patient_input, state],
        outputs=outputs,
    )
    patient_input.submit(
        fn=analyze_patient,
        inputs=[patient_input, state],
        outputs=outputs,
    )

    submit_btn.click(
        fn=apply_finding,
        inputs=[test_selector, finding_input, state],
        outputs=outputs,
    )

    reset_btn.click(
        fn=reset_session,
        outputs=[patient_input] + outputs,
    )


if __name__ == "__main__":
    print("エンジン初期化中...")
    init_engine()
    print(f"疾患DB: {len(engine.disease_db)}件 / ベクトルDB: {engine.collection.count()}件")
    print(f"名寄せマップ: {len(engine.test_name_map)}件")
    print("Web UI起動中...")
    app.launch(inbrowser=True)
