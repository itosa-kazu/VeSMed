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
  7. HPE所見フィードバック → 行列演算のみで即時更新（LLM不要）
"""

import copy
import json as _json
import os

import gradio as gr
import numpy as np
import pandas as pd
from engine import VeSMedEngine
from config import TOP_K_TESTS

engine = None


def init_engine():
    global engine
    if engine is None:
        engine = VeSMedEngine()
    return engine


# HPE項目名リスト（Dropdown用、エンジン不要で読み込み）
def _load_hpe_choices():
    hx, pe = [], []
    path = os.path.join(os.path.dirname(__file__), "data", "hpe_items.jsonl")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = _json.loads(line)
                    if item["category"] == "Hx":
                        hx.append(item["item_name"])
                    else:
                        pe.append(item["item_name"])
    except Exception:
        pass
    return hx, pe

HPE_HX_CHOICES, HPE_PE_CHOICES = _load_hpe_choices()
HPE_ALL_CHOICES = HPE_HX_CHOICES + HPE_PE_CHOICES


# ----------------------------------------------------------------
# データ管理ヘルパー
# ----------------------------------------------------------------

def _load_all_names(jsonl_path, key):
    """JSONL から名前一覧を読み込む"""
    names = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    names.append(_json.loads(line).get(key, ""))
    except Exception:
        pass
    return names


_DISEASES_PATH = os.path.join(os.path.dirname(__file__), "data", "diseases.jsonl")
_TESTS_PATH = os.path.join(os.path.dirname(__file__), "data", "tests.jsonl")
_HPE_PATH = os.path.join(os.path.dirname(__file__), "data", "hpe_items.jsonl")

DISEASE_NAMES = _load_all_names(_DISEASES_PATH, "disease_name")
TEST_NAMES = _load_all_names(_TESTS_PATH, "test_name")
HPE_ITEM_NAMES = _load_all_names(_HPE_PATH, "item_name")

# 疾患のセクションフィールド
_FD_SECTIONS = [
    "fd_background", "fd_typical", "fd_atypical",
    "fd_physical", "fd_tests", "fd_differential", "fd_pathophysiology",
]
_FD_LABELS = {
    "fd_background": "背景・疫学",
    "fd_typical": "典型的所見",
    "fd_atypical": "非典型的所見",
    "fd_physical": "身体診察所見",
    "fd_tests": "検査所見",
    "fd_differential": "鑑別診断",
    "fd_pathophysiology": "病態生理",
}


def _load_jsonl_record(path, key_field, key_value):
    """JSONLから特定レコードを読み込む"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = _json.loads(line)
                    if rec.get(key_field) == key_value:
                        return rec
    except Exception:
        pass
    return {}


def _save_jsonl_record(path, key_field, key_value, updates):
    """JSONLの特定レコードを更新して書き戻す"""
    lines = []
    updated = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = _json.loads(line)
                if rec.get(key_field) == key_value:
                    rec.update(updates)
                    updated = True
                lines.append(_json.dumps(rec, ensure_ascii=False))
    if updated:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return updated


def _on_disease_select(disease_name):
    """疾患選択時のコールバック"""
    if not disease_name:
        empty = [""] * (3 + len(_FD_SECTIONS))
        return empty
    rec = _load_jsonl_record(_DISEASES_PATH, "disease_name", disease_name)
    meta = f"{rec.get('category', '')} / {rec.get('urgency', '')} / ICD-10: {rec.get('icd10', '')}"
    fd = rec.get("findings_description", "")
    dfe = rec.get("description_for_embedding", "")
    sections = [rec.get(s, "") for s in _FD_SECTIONS]
    return [meta, fd, dfe] + sections


def _save_disease(disease_name, fd, dfe, *section_values):
    """疾患テキスト保存"""
    if not disease_name:
        return "疾患を選択してください"
    updates = {"findings_description": fd, "description_for_embedding": dfe}
    for i, key in enumerate(_FD_SECTIONS):
        if i < len(section_values):
            updates[key] = section_values[i]
    ok = _save_jsonl_record(_DISEASES_PATH, "disease_name", disease_name, updates)
    if ok:
        return f"{disease_name} を保存しました。embeddingへの反映はindex.py再実行が必要です。"
    return "保存失敗: 該当疾患が見つかりません"


def _on_test_select(test_name):
    """検査選択時のコールバック"""
    if not test_name:
        return [""] * 5
    rec = _load_jsonl_record(_TESTS_PATH, "test_name", test_name)
    meta = f"{rec.get('category', '')} / {rec.get('sample_type', '')}"
    fd = rec.get("findings_description", "")
    hs = rec.get("hypothesis_screen", "")
    qd = rec.get("quality_description", "")
    ht = rec.get("hypothesis_text", "")
    return [meta, fd, hs, qd, ht]


def _save_test(test_name, fd, hs, qd, ht):
    """検査テキスト保存"""
    if not test_name:
        return "検査を選択してください"
    updates = {
        "findings_description": fd,
        "hypothesis_screen": hs,
        "quality_description": qd,
        "hypothesis_text": ht,
    }
    ok = _save_jsonl_record(_TESTS_PATH, "test_name", test_name, updates)
    if ok:
        return f"{test_name} を保存しました。embeddingへの反映はindex.py再実行が必要です。"
    return "保存失敗: 該当検査が見つかりません"


def _on_hpe_select(item_name):
    """HPE項目選択時のコールバック"""
    if not item_name:
        return [""] * 4
    rec = _load_jsonl_record(_HPE_PATH, "item_name", item_name)
    meta = f"{rec.get('category', '')} / {rec.get('subcategory', '')}"
    hypothesis = rec.get("hypothesis", "")
    instruction = rec.get("instruction", "")
    return [meta, hypothesis, instruction]


def _save_hpe(item_name, hypothesis, instruction):
    """HPE項目保存"""
    if not item_name:
        return "HPE項目を選択してください"
    updates = {"hypothesis": hypothesis, "instruction": instruction}
    ok = _save_jsonl_record(_HPE_PATH, "item_name", item_name, updates)
    if ok:
        return f"{item_name} を保存しました。sim_matrix_hpeの再構築が必要です。"
    return "保存失敗: 該当HPE項目が見つかりません"


# ----------------------------------------------------------------
# ランキング再計算（共通ロジック）
# ----------------------------------------------------------------

def _rerank(eng, candidates, novelty_tests, novelty_hpe, exclusion_reasons):
    """全Partの再ランキング。LLM不要、純粋な行列演算。"""
    cluster_mu = eng.rank_tests_cluster_mu(candidates, novelty=novelty_tests)
    ranked = eng.rank_tests(candidates, novelty=novelty_tests)
    critical = eng.rank_tests_critical(candidates, novelty=novelty_tests)
    confirm = eng.rank_tests_confirm(candidates, novelty=novelty_tests)
    hpe = eng.rank_hpe(candidates, novelty_hpe=novelty_hpe)

    return (
        format_diseases_df(candidates),
        format_hpe_df(hpe, "Hx"),
        format_hpe_df(hpe, "PE"),
        format_cluster_mu_df(cluster_mu),
        format_tests_df(ranked),
        format_critical_df(critical),
        format_confirm_df(confirm),
        format_exclusion_df(exclusion_reasons),
        format_suppressed_df(eng.test_names, novelty_tests),
    )


# ----------------------------------------------------------------
# コールバック: 初回分析（LLM使用）
# ----------------------------------------------------------------

def analyze_patient(input_text, state):
    """
    v2パイプライン: embedding知覚 + LLM判断
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor
    if not input_text.strip():
        empty = (
            "テキストを入力してください。",
            None, None, None, None, None, None, None, None, None,
            state, [], [],
        )
        return empty

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

    cands_copy = copy.deepcopy(candidates)

    with ThreadPoolExecutor(max_workers=3) as executor:
        fut_filter = executor.submit(
            eng.filter_contradictions, candidates, full_text, negative_findings,
        )
        fut_update = executor.submit(
            eng.update_from_results, cands_copy, result_lines, positive_text,
        ) if result_lines else None
        fut_all_novelty = executor.submit(eng.compute_all_novelty, full_text)

        novelty, novelty_hpe, hpe_findings = fut_all_novelty.result()
        filtered_candidates, exclusion_reasons = fut_filter.result()

        if fut_update:
            updated_cands = fut_update.result()
            updated_map = {c["disease_name"]: c for c in updated_cands}
            for c in filtered_candidates:
                if c["disease_name"] in updated_map:
                    c["similarity"] = updated_map[c["disease_name"]]["similarity"]

    candidates = filtered_candidates

    # 陰性所見 → HPE novelty橋渡し（compute_all_noveltyの見逃し補完）
    if negative_findings:
        novelty_hpe, hpe_findings = eng.patch_hpe_from_negatives(
            negative_findings, novelty_hpe, hpe_findings,
        )

    # HPE所見で疾患重み更新の前にベースを保存
    candidates_base = copy.deepcopy(candidates)

    if hpe_findings:
        candidates = eng.update_from_hpe(candidates, hpe_findings)

    t2 = _time.time()
    print(f"[TIMING] parallel: {t2-t1:.1f}s")

    # Step 3: ランキング
    tables = _rerank(eng, candidates, novelty, novelty_hpe, exclusion_reasons)
    t3 = _time.time()
    print(f"[TIMING] rank: {t3-t2:.1f}s")
    print(f"[TIMING] === TOTAL: {t3-t0:.1f}s ===")

    # ステータス
    n_results = len(result_lines)
    n_excluded = len(exclusion_reasons)
    filter_info = f" / {n_excluded}疾患除外" if n_excluded > 0 else ""
    if negative_findings:
        neg_names = "、".join(negative_findings)
        neg_info = f" / 陰性所見{len(negative_findings)}件（{neg_names}）"
    else:
        neg_info = ""
    status = f"分析完了 / {len(candidates)}疾患{filter_info}{neg_info} / 検査結果{n_results}件"

    # セッション状態保存
    auto_pos = [f["item"] for f in hpe_findings if f["polarity"] > 0]
    auto_neg = [f["item"] for f in hpe_findings if f["polarity"] < 0]

    state = {
        "input_text": input_text,
        "query_embedding": eng._last_query_embedding.tolist(),
        "candidates_base": candidates_base,
        "novelty_tests": novelty.tolist(),
        "exclusion_reasons": exclusion_reasons,
    }

    # tables = (disease, hx, pe, cluster_mu, tests, critical, confirm, excluded, suppressed)
    return (
        status,
        tables[0],  # disease
        tables[1],  # hx
        tables[2],  # pe
        tables[3],  # cluster_mu
        tables[4],  # tests
        tables[5],  # critical
        tables[6],  # confirm
        tables[7],  # excluded
        tables[8],  # suppressed
        state,
        auto_pos,
        auto_neg,
    )


# ----------------------------------------------------------------
# コールバック: HPE所見フィードバック（LLM不要）
# ----------------------------------------------------------------

def update_hpe_feedback(pos_items, neg_items, state):
    """
    波動方式: クエリembedding ± HPE仮説embedding → 干渉delta → 即時更新。
    embedding API不要（ベクトル演算のみ）。
    """
    eng = init_engine()

    if not state or "candidates_base" not in state:
        return [gr.update()] * 10 + [state, pos_items, neg_items]

    novelty_tests = np.array(state["novelty_tests"])
    exclusion_reasons = state["exclusion_reasons"]

    # HPE findings構築
    hpe_findings = []
    novelty_hpe = np.ones(len(eng.hpe_names))

    for name in (pos_items or []):
        idx = eng.hpe_idx.get(name)
        if idx is not None:
            hpe_findings.append({"item": name, "index": idx, "polarity": 1})
            novelty_hpe[idx] = 0.0

    for name in (neg_items or []):
        idx = eng.hpe_idx.get(name)
        if idx is not None:
            hpe_findings.append({"item": name, "index": idx, "polarity": -1})
            novelty_hpe[idx] = 0.0

    n_pos = sum(1 for f in hpe_findings if f["polarity"] > 0)
    n_neg = sum(1 for f in hpe_findings if f["polarity"] < 0)
    print(f"[HPEフィードバック] 陽性{n_pos}件 + 陰性{n_neg}件（波動方式）")

    # 元のクエリembeddingを復元（波動方式の基底ベクトル）
    query_emb = state.get("query_embedding")
    if query_emb is not None:
        eng._last_query_embedding = np.array(query_emb, dtype=np.float32)

    candidates = copy.deepcopy(state["candidates_base"])

    # 波動更新: query ± HPE embeddings → delta適用
    if hpe_findings:
        candidates = eng.update_from_hpe(candidates, hpe_findings)

    tables = _rerank(eng, candidates, novelty_tests, novelty_hpe, exclusion_reasons)
    status = f"HPE更新完了 / 陽性{n_pos}件 + 陰性{n_neg}件 反映（波動方式）"

    return (
        status,
        tables[0], tables[1], tables[2], tables[3],
        tables[4], tables[5], tables[6], tables[7], tables[8],
        state,
        pos_items,
        neg_items,
    )


# ----------------------------------------------------------------
# コールバック: リセット
# ----------------------------------------------------------------

def reset_session():
    return (
        "",
        "リセットしました。",
        None, None, None, None, None, None, None, None, None,
        None, [], [],
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

    session_state = gr.State(value=None)

    gr.Markdown("# VeSMed - ベクトル空間医学統一フレームワーク")

    with gr.Tabs():

        # ============================================================
        # タブ1: 臨床分析
        # ============================================================
        with gr.Tab("臨床分析"):

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
            gr.Markdown("### 問診・身体診察推奨")

            with gr.Row():
                with gr.Column(scale=1):
                    hpe_hx_table = gr.Dataframe(
                        label="問診推奨（分散ベース）",
                        headers=["#", "項目", "分類", "効用", "分散", "新規", "手順", "関連疾患"],
                        interactive=False,
                    )
                with gr.Column(scale=1):
                    hpe_pe_table = gr.Dataframe(
                        label="身体診察推奨（分散ベース）",
                        headers=["#", "項目", "分類", "効用", "分散", "新規", "手順", "関連疾患"],
                        interactive=False,
                    )

            with gr.Accordion("所見フィードバック（波動方式 — ベクトル演算のみで即時更新）", open=True):
                with gr.Row():
                    hpe_pos_dropdown = gr.Dropdown(
                        choices=HPE_ALL_CHOICES, multiselect=True, label="陽性所見（あり）",
                        info="問診・身体診察で陽性だった項目",
                    )
                    hpe_neg_dropdown = gr.Dropdown(
                        choices=HPE_ALL_CHOICES, multiselect=True, label="陰性所見（なし）",
                        info="問診・身体診察で陰性だった項目",
                    )
                hpe_update_btn = gr.Button("所見反映", variant="secondary")

            gr.Markdown("---")

            with gr.Row():
                disease_table = gr.Dataframe(
                    label="候補疾患（類似度順）",
                    headers=["#", "疾患名", "類似度", "重み", "緊急度", "診療科"],
                    interactive=False,
                )

            with gr.Row():
                cluster_mu_table = gr.Dataframe(
                    label="基本推奨（候補群の共通必要度）",
                    headers=["#", "検査名", "効用", "共通度", "新規", "関連疾患"],
                    interactive=False,
                )
            with gr.Row():
                with gr.Column(scale=1):
                    test_table = gr.Dataframe(
                        label="鑑別推奨（分散ベース）",
                        headers=["#", "検査名", "効用", "分散", "新規", "関連疾患"],
                        interactive=False,
                    )
                with gr.Column(scale=1):
                    critical_table = gr.Dataframe(
                        label="Critical排除推奨（最大命中）",
                        headers=["#", "検査名", "効用", "命中", "新規", "排除対象"],
                        interactive=False,
                    )
            with gr.Row():
                confirm_table = gr.Dataframe(
                    label="確認・同定推奨（加重平均）",
                    headers=["#", "検査名", "効用", "特異", "新規", "関連疾患"],
                    interactive=False,
                )

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

        # ============================================================
        # タブ2: データ管理
        # ============================================================
        with gr.Tab("データ管理"):

            # ----- 疾患テキスト -----
            gr.Markdown("## 疾患テキスト")
            with gr.Row():
                dm_disease_dd = gr.Dropdown(
                    choices=DISEASE_NAMES, label="疾患選択",
                    allow_custom_value=False,
                )
                dm_disease_meta = gr.Textbox(label="メタ情報", interactive=False)

            dm_disease_fd = gr.Textbox(
                label="findings_description（embedding対象）",
                lines=20, max_lines=40,
            )
            with gr.Accordion("セクション別テキスト", open=False):
                dm_fd_sections = {}
                for key in _FD_SECTIONS:
                    dm_fd_sections[key] = gr.Textbox(
                        label=_FD_LABELS[key], lines=8, max_lines=20,
                    )
            dm_disease_dfe = gr.Textbox(
                label="description_for_embedding（短縮版、未使用）",
                lines=3,
            )
            with gr.Row():
                dm_disease_save_btn = gr.Button("疾患テキスト保存", variant="primary")
                dm_disease_status = gr.Textbox(label="", interactive=False, scale=3)

            gr.Markdown("---")

            # ----- 検査テキスト -----
            gr.Markdown("## 検査テキスト")
            with gr.Row():
                dm_test_dd = gr.Dropdown(
                    choices=TEST_NAMES, label="検査選択",
                    allow_custom_value=False,
                )
                dm_test_meta = gr.Textbox(label="メタ情報", interactive=False)

            dm_test_fd = gr.Textbox(
                label="findings_description（embedding対象）",
                lines=15, max_lines=30,
            )
            dm_test_hs = gr.Textbox(
                label="hypothesis_screen（スクリーニング仮説）",
                lines=4,
            )
            dm_test_qd = gr.Textbox(
                label="quality_description（品質記述）",
                lines=5,
            )
            dm_test_ht = gr.Textbox(
                label="hypothesis_text（仮説テキスト）",
                lines=2,
            )
            with gr.Row():
                dm_test_save_btn = gr.Button("検査テキスト保存", variant="primary")
                dm_test_status = gr.Textbox(label="", interactive=False, scale=3)

            gr.Markdown("---")

            # ----- HPE項目 -----
            gr.Markdown("## HPE項目（問診・身体診察）")
            with gr.Row():
                dm_hpe_dd = gr.Dropdown(
                    choices=HPE_ITEM_NAMES, label="HPE項目選択",
                    allow_custom_value=False,
                )
                dm_hpe_meta = gr.Textbox(label="メタ情報", interactive=False)

            dm_hpe_hypothesis = gr.Textbox(
                label="hypothesis（スクリーニング仮説 — embedding対象）",
                lines=3,
            )
            dm_hpe_instruction = gr.Textbox(
                label="instruction（聴取手順）",
                lines=2,
            )
            with gr.Row():
                dm_hpe_save_btn = gr.Button("HPE項目保存", variant="primary")
                dm_hpe_status = gr.Textbox(label="", interactive=False, scale=3)

    # ===== イベント: 臨床分析タブ =====

    all_outputs = [
        status_text,
        disease_table, hpe_hx_table, hpe_pe_table,
        cluster_mu_table, test_table, critical_table, confirm_table,
        excluded_table, suppressed_table,
        session_state, hpe_pos_dropdown, hpe_neg_dropdown,
    ]

    analyze_btn.click(
        fn=analyze_patient,
        inputs=[input_text, session_state],
        outputs=all_outputs,
    )
    input_text.submit(
        fn=analyze_patient,
        inputs=[input_text, session_state],
        outputs=all_outputs,
    )

    hpe_update_btn.click(
        fn=update_hpe_feedback,
        inputs=[hpe_pos_dropdown, hpe_neg_dropdown, session_state],
        outputs=all_outputs,
    )

    reset_btn.click(
        fn=reset_session,
        outputs=[input_text] + all_outputs,
    )

    # ===== イベント: データ管理タブ =====

    dm_disease_outputs = [dm_disease_meta, dm_disease_fd, dm_disease_dfe] + \
                         [dm_fd_sections[k] for k in _FD_SECTIONS]

    dm_disease_dd.change(
        fn=_on_disease_select,
        inputs=[dm_disease_dd],
        outputs=dm_disease_outputs,
    )

    dm_disease_save_btn.click(
        fn=_save_disease,
        inputs=[dm_disease_dd, dm_disease_fd, dm_disease_dfe] +
               [dm_fd_sections[k] for k in _FD_SECTIONS],
        outputs=[dm_disease_status],
    )

    dm_test_dd.change(
        fn=_on_test_select,
        inputs=[dm_test_dd],
        outputs=[dm_test_meta, dm_test_fd, dm_test_hs, dm_test_qd, dm_test_ht],
    )

    dm_test_save_btn.click(
        fn=_save_test,
        inputs=[dm_test_dd, dm_test_fd, dm_test_hs, dm_test_qd, dm_test_ht],
        outputs=[dm_test_status],
    )

    dm_hpe_dd.change(
        fn=_on_hpe_select,
        inputs=[dm_hpe_dd],
        outputs=[dm_hpe_meta, dm_hpe_hypothesis, dm_hpe_instruction],
    )

    dm_hpe_save_btn.click(
        fn=_save_hpe,
        inputs=[dm_hpe_dd, dm_hpe_hypothesis, dm_hpe_instruction],
        outputs=[dm_hpe_status],
    )


if __name__ == "__main__":
    print("エンジン初期化中...")
    init_engine()
    print(f"疾患DB: {len(engine.disease_db)}件 / 疾患Emb: {engine.disease_embs_normed.shape[0]}件")
    print(f"名寄せマップ: {len(engine.test_name_map)}件")
    if engine.hpe_items:
        n_hx = sum(1 for it in engine.hpe_items if it['category'] == 'Hx')
        n_pe = sum(1 for it in engine.hpe_items if it['category'] == 'PE')
        print(f"HPE: 問診{n_hx}項目 + 身体診察{n_pe}項目 = {n_hx+n_pe}項目")
    print("Web UI起動中...")
    app.launch(inbrowser=True, share=True)
