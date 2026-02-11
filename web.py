"""
VeSMed - Web UI (Gradio)
ブラウザから操作できる対話型インターフェース

フロー:
  1. 患者情報入力（自由記述）→ ベクトル検索 → 候補疾患 + 推薦検査を表示
  2. 検査結果が出たら患者テキストに追記 → 再分析 → 更新
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

def analyze_patient(patient_text):
    """患者情報を分析（ベクトル検索）"""
    import time as _time
    if not patient_text.strip():
        return (
            "患者情報を入力してください。",
            None, None, None,
        )

    eng = init_engine()

    t0 = _time.time()
    # embedding検索（LLM不要: 生テキスト直接embed）
    candidates = eng.search_diseases(patient_text)
    t1 = _time.time()
    print(f"[TIMING] search_diseases: {t1-t0:.1f}s")

    candidates = eng.compute_priors(candidates)

    # novelty計算（行単位embedding → 既知情報を連続的に割引）
    novelty = eng.compute_novelty(patient_text)
    t2 = _time.time()
    print(f"[TIMING] compute_novelty: {t2-t1:.1f}s")

    ranked_tests = eng.rank_tests(candidates, novelty=novelty)
    critical_tests = eng.rank_tests_critical(candidates, novelty=novelty)
    t3 = _time.time()
    print(f"[TIMING] rank_tests+critical: {t3-t2:.1f}s")
    print(f"[TIMING] === TOTAL: {t3-t0:.1f}s ===")

    status = f"分析完了 / 全{len(candidates)}疾患で計算 / 推薦検査{len(ranked_tests)}件"

    return (
        status,
        format_diseases_df(candidates),
        format_tests_df(ranked_tests),
        format_critical_df(critical_tests),
    )


def reset_session():
    return (
        "",
        "リセットしました。",
        None, None, None,
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


# ----------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------

with gr.Blocks(
    title="VeSMed - ベクトル空間医学",
) as app:

    gr.Markdown("""
# VeSMed - ベクトル空間医学統一フレームワーク
患者情報（自由記述）→ ベクトル検索 → 候補疾患 + 推薦検査を表示。検査結果は患者テキストに追記して再分析。
    """)

    # ===== 上段: 患者情報 =====
    with gr.Row():
        with gr.Column(scale=3):
            patient_input = gr.Textbox(
                label="患者情報（自由記述 — 検査結果もここに追記）",
                placeholder="例: 67歳の男性。繰り返す発熱を主訴に来院。7週間前から38℃前後の発熱が出現し、市販の解熱薬で一時的に解熱するが再度発熱する。\n\n検査結果が出たらここに追記して再分析: 血液培養→黄色ブドウ球菌陽性、心エコー→大動脈弁に疣贅あり",
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

    # ===== イベント =====

    outputs = [status_text, disease_table, test_table, critical_table]

    analyze_btn.click(
        fn=analyze_patient,
        inputs=[patient_input],
        outputs=outputs,
    )
    patient_input.submit(
        fn=analyze_patient,
        inputs=[patient_input],
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
