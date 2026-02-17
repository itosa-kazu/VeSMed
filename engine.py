"""
VeSMed - 核心エンジン
クエリ整理 → ベクトル検索 → 情報利得計算 → 検査推薦
"""

import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from openai import OpenAI
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    LLM_FALLBACK_API_KEY, LLM_FALLBACK_BASE_URL, LLM_FALLBACK_MODEL,
    LLM_MAX_RETRIES,
    VERTEX_SA_KEY, VERTEX_PROJECT, VERTEX_LOCATION, VERTEX_MODEL,
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
    DISEASES_JSONL, TESTS_JSONL, FINDINGS_JSONL, HPE_ITEMS_JSONL,
    DATA_DIR,
)


class VeSMedEngine:
    def __init__(self):
        # Vertex AI（プライマリLLM — 直接Google API、低遅延）
        from google import genai
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = VERTEX_SA_KEY
        self.vertex_client = genai.Client(
            project=VERTEX_PROJECT,
            location=VERTEX_LOCATION,
            vertexai=True,
        )
        # フォールバック（Vertex失敗時）
        self.llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=30)
        self.llm_fallback_client = OpenAI(api_key=LLM_FALLBACK_API_KEY, base_url=LLM_FALLBACK_BASE_URL, timeout=30)
        self.embed_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)

        # 疾患embedding読込（MEAN-chunk統一、NPZから）
        self._load_disease_embeddings()

        # 検査名 名寄せマップ読み込み
        self.test_name_map = {}
        map_file = os.path.join(DATA_DIR, "test_name_map.json")
        if os.path.exists(map_file):
            with open(map_file, "r", encoding="utf-8") as f:
                self.test_name_map = json.load(f)

        # 疾患メタデータを全件メモリに読み込み（disease_name → dict）
        self.disease_db = {}
        if os.path.exists(DISEASES_JSONL):
            with open(DISEASES_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        self.disease_db[d["disease_name"]] = d
                    except (json.JSONDecodeError, KeyError):
                        continue

        # findings.jsonlを読み込み、relevant_testsにマージ
        if os.path.exists(FINDINGS_JSONL):
            with open(FINDINGS_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        fd = json.loads(line)
                        dname = fd["disease_name"]
                        if dname in self.disease_db:
                            existing = self.disease_db[dname].setdefault("relevant_tests", [])
                            for item in fd.get("history_items", []) + fd.get("exam_items", []):
                                existing.append(item)
                    except (json.JSONDecodeError, KeyError):
                        continue

        # 2Cスコア（Critical / Curable）をembeddingで計算
        self.disease_2c = {}  # disease_name → {"critical": float, "curable": float, "weight": float}
        self._compute_2c_scores()

        # 検査メタデータを全件メモリに読み込み（test_name → dict）
        self.test_db = {}
        if os.path.exists(TESTS_JSONL):
            with open(TESTS_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                        self.test_db[t["test_name"]] = t
                    except (json.JSONDecodeError, KeyError):
                        continue

        # 類似度行列（疾患×検査）をembeddingで計算
        self.sim_matrix = None    # (N_diseases, N_tests) ndarray — 鑑別用（Part A/E）
        self.sim_matrix_confirm = None  # (N_diseases, N_tests) ndarray — 確認用（Part B/C）
        # disease_embs_normed, disease_idx は _load_disease_embeddings() で設定済み
        self.test_idx = {}        # test_name → col index
        self.test_names = []      # col順の検査名リスト
        self._compute_similarity_matrix()
        self._load_confirm_matrix()

        # 検査の質（2Q: Value + Feasibility）
        self.test_quality = {}  # test_name → {"value": float, "feasibility": float}
        self._compute_test_quality()

        # 検査の侵襲性スコア（動的侵襲性バランシング用）
        self.cos_invasive = None  # (N_tests,) ndarray — 各検査の侵襲度
        self._compute_invasive_scores()

        # 検査リスクembedding（risk_description → embedding）
        self.risk_embs = {}  # test_name → np.array (4096,)
        self._compute_risk_embeddings()

        # 検査名embedding（novelty + 検査マッチ用）: 正規化済み (N_tests, dim)
        self.test_name_embs = None
        self._compute_test_name_embs()

        # 極性軸（正常←→異常の差分ベクトル、Option 3用）
        self.polarity_axis = None
        self._compute_polarity_axis()

        # 基準範囲テーブル + エイリアスマップ
        self.reference_ranges = {}  # canonical_test_name → {lower, upper, unit}
        self.range_alias = {}       # alias (lowercase) → canonical_test_name
        self._load_reference_ranges()

        # Part D: 問診・身体診察項目
        self.hpe_items = []          # [{"item_name", "category", "subcategory", "hypothesis", "instruction"}, ...]
        self.hpe_names = []          # 項目名リスト（表示用）
        self.hpe_idx = {}            # item_name → index
        self.sim_matrix_hpe = None   # (N_diseases, N_hpe_items) ndarray (screen用)
        self.sim_matrix_hpe_confirm = None  # (N_diseases, N_hpe_items) confirm行列
        self.hpe_name_embs = None    # (N_hpe_items, dim) 正規化済み
        self.hpe_hyp_embs = None     # (N_hpe_items, dim) 仮説embedding（novelty二重マッチ用）
        self._load_hpe_items()
        if self.hpe_items:
            self._compute_hpe_similarity_matrix()
            self._load_hpe_confirm_matrix()
            self._compute_hpe_name_embs()

        # 最後のクエリembedding（rank_testsでrisk_relevance計算に使用）
        self._last_query_embedding = None

    # ----------------------------------------------------------------
    # 疾患embedding読込（MEAN-chunk統一、NPZから）
    # ----------------------------------------------------------------
    def _load_disease_embeddings(self):
        """disease_embs.npzからMEAN集約済み正規化embeddingを読み込む。"""
        embs_file = os.path.join(DATA_DIR, "disease_embs.npz")
        if not os.path.exists(embs_file):
            print(f"[ERROR] {embs_file} が見つかりません。index.pyを先に実行してください。")
            self.disease_embs_normed = None
            self.disease_idx = {}
            return

        data = np.load(embs_file, allow_pickle=True)
        self.disease_embs_normed = data["disease_embs_normed"].astype(np.float32)
        disease_names = list(data["disease_names"])
        self.disease_idx = {name: i for i, name in enumerate(disease_names)}
        print(f"[疾患Emb] {self.disease_embs_normed.shape[0]}疾患の正規化embedding読込 (NPZ)")

    # ----------------------------------------------------------------
    # 2Cスコア計算（起動時に1回）
    # ----------------------------------------------------------------
    def _compute_2c_scores(self):
        """
        Critical / Curable の2アンカーテキストと各疾患のembeddingの
        余弦類似度から臨床重要度スコアを算出する。

        重み = exp(cos_critical + cos_curable)
        - Critical: 見逃しコスト（事前確率に含まれない独自情報）
        - Curable:  診断利益（治療可能性、事前確率に含まれない独自情報）
        - Common は事前確率（ベクトル検索類似度）と重複するため廃止
        - exp() により常に正値、フロア不要、softmaxと数学的に統一
        """
        anchors = {
            "critical": (
                "未治療の場合、数時間以内にバイタルサイン急速悪化（血圧低下、頻脈から徐脈へ移行、"
                "SpO2低下、呼吸停止）、意識レベル進行性低下（JCS 300、GCS 3）、"
                "ショック所見（冷汗、末梢チアノーゼ、毛細血管再充満時間延長、乏尿から無尿）、"
                "多臓器不全所見（肝酵素急上昇、凝固異常によるDIC、代謝性アシドーシス、乳酸上昇）"
                "が出現し、心停止・自発呼吸停止・瞳孔散大固定に至る。"
            ),
            "curable": (
                "治療開始後、数時間から数日で検査値の正常化（CRP低下、白血球正常化、培養陰性化、"
                "肝酵素低下、腎機能改善）、バイタルサイン安定化（解熱、血圧正常化、頻脈改善、"
                "SpO2上昇）、症状消失（疼痛消失、呼吸困難改善、意識清明化）、"
                "画像所見改善（浸潤影消退、膿瘍縮小、浮腫軽減、閉塞解除）が観察される。"
            ),
        }

        # アンカーをembedding
        anchor_texts = list(anchors.values())
        try:
            resp = self.embed_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=anchor_texts,
            )
            anchor_embs = {
                name: np.array(resp.data[i].embedding)
                for i, name in enumerate(anchors.keys())
            }
        except Exception as e:
            print(f"[2C] アンカーembedding失敗: {e}")
            return

        # disease_embs_normed（MEAN集約済み）を直接使用
        if self.disease_embs_normed is None:
            print("[2C] 疾患embeddingなし、スキップ")
            return

        # アンカーembeddingを正規化
        for name in anchor_embs:
            a = anchor_embs[name]
            norm_a = np.linalg.norm(a)
            if norm_a > 0:
                anchor_embs[name] = a / norm_a

        # disease_embs_normed × anchor_emb で一括計算
        for dname, d_idx in self.disease_idx.items():
            emb = self.disease_embs_normed[d_idx]
            scores = {}
            for anchor_name, anchor_emb in anchor_embs.items():
                scores[anchor_name] = float(np.dot(emb, anchor_emb))
            weight = math.exp(scores["critical"] + scores["curable"])
            self.disease_2c[dname] = {**scores, "weight": weight}

        print(f"[2C] {len(self.disease_2c)}疾患の2Cスコア計算完了")

    # ----------------------------------------------------------------
    # 類似度行列計算（疾患×検査、起動時に1回）
    # ----------------------------------------------------------------
    def _compute_similarity_matrix(self):
        """
        仮説embedding方式: sim_matrix[d][j] = cos(E("検査名_j 異常"), disease_emb_d)

        検査findingsの記述は不要。疾患記述に既に検査情報が含まれているため、
        「検査名 異常」という最小仮説のembeddingだけで疾患との関連性を捕捉できる。
        update_from_results（直接法）と同じ演算・同じ空間。仮想か現実かの違いだけ。
        """
        cache_file = os.path.join(DATA_DIR, "sim_matrix.npz")

        # 検査リスト構築（findings_descriptionがあるもののみ）
        self.test_names = [
            tname for tname, tdata in self.test_db.items()
            if tdata.get("findings_description")
        ]
        self.test_idx = {name: i for i, name in enumerate(self.test_names)}

        # disease_embs_normedは_load_disease_embeddings()で読込済み
        if self.disease_embs_normed is None:
            print("[sim_matrix] 疾患embeddingなし、スキップ")
            return

        disease_names = [""] * len(self.disease_idx)
        for dname, idx in self.disease_idx.items():
            disease_names[idx] = dname

        # 仮説テキスト構築: hypothesis_screen (Phase 2) > hypothesis_text (旧) > "検査名 異常"
        hypothesis_texts = []
        for tname in self.test_names:
            entry = self.test_db.get(tname, {})
            screen = entry.get("hypothesis_screen") or entry.get("hypothesis_text")
            hypothesis_texts.append(screen if screen else f"{tname} 異常")

        # 仮説テキストのハッシュ（内容変更検知）
        import hashlib
        h_hash = hashlib.md5("||".join(hypothesis_texts).encode()).hexdigest()[:12]

        # キャッシュチェック
        if os.path.exists(cache_file):
            data = np.load(cache_file, allow_pickle=True)
            cached_diseases = list(data["disease_names"])
            cached_tests = list(data["test_names"])
            is_hypothesis = bool(data.get("hypothesis_mode", False))
            cached_hash = str(data["hypothesis_hash"]) if "hypothesis_hash" in data else ""
            if (cached_diseases == disease_names and cached_tests == self.test_names
                    and is_hypothesis and cached_hash == h_hash):
                self.sim_matrix = data["sim_matrix"]
                print(f"[sim_matrix] 仮説方式キャッシュから読込 {self.sim_matrix.shape}")
                return
        hypothesis_embs = self._batch_embed(hypothesis_texts)
        if hypothesis_embs is None:
            print("[sim_matrix] 仮説embedding失敗")
            return

        # 正規化
        h_norms = np.linalg.norm(hypothesis_embs, axis=1, keepdims=True)
        h_norms[h_norms == 0] = 1.0
        hypothesis_embs_normed = hypothesis_embs / h_norms

        self.sim_matrix = self.disease_embs_normed @ hypothesis_embs_normed.T

        # キャッシュ保存
        np.savez(
            cache_file,
            sim_matrix=self.sim_matrix,
            disease_names=np.array(disease_names, dtype=object),
            test_names=np.array(self.test_names, dtype=object),
            hypothesis_mode=np.array(True),
            hypothesis_hash=np.array(h_hash),
        )
        print(f"[sim_matrix] 仮説方式 {self.sim_matrix.shape} 計算・キャッシュ完了")

    def _load_confirm_matrix(self):
        """
        Dual sim_matrix: 確認用行列 (sim_matrix_confirm) の読み込み。
        build_confirm_matrix.py で事前生成された sim_matrix_confirm.npz を読み込む。
        Part B (致死除外) と Part C (確認力) で使用。

        未生成の場合は sim_matrix（鑑別用）にフォールバック。
        """
        confirm_file = os.path.join(DATA_DIR, "sim_matrix_confirm.npz")
        if not os.path.exists(confirm_file):
            print("[sim_matrix_confirm] ファイル未生成 → sim_matrix(鑑別用)にフォールバック")
            self.sim_matrix_confirm = self.sim_matrix
            return

        data = np.load(confirm_file, allow_pickle=True)
        cached_diseases = list(data["disease_names"])
        cached_tests = list(data["test_names"])

        # 疾患・検査の順序が一致するか確認
        disease_names = sorted(self.disease_idx.keys())
        if cached_diseases == disease_names and cached_tests == self.test_names:
            self.sim_matrix_confirm = data["sim_matrix"]
            print(f"[sim_matrix_confirm] 確認用行列読込 {self.sim_matrix_confirm.shape}")
        else:
            print(f"[sim_matrix_confirm] 疾患/検査リスト不一致 → sim_matrixにフォールバック")
            print(f"  confirm: {len(cached_diseases)}疾患×{len(cached_tests)}検査")
            print(f"  screen:  {len(disease_names)}疾患×{len(self.test_names)}検査")
            self.sim_matrix_confirm = self.sim_matrix

    def _load_hpe_confirm_matrix(self):
        """
        HPE confirm行列の読み込み（build_hpe_matrix.pyで事前生成）。
        update_from_hpe()でType R項目の疾患重み更新に使用。
        未生成の場合はsim_matrix_hpe（screen用）にフォールバック。
        """
        confirm_file = os.path.join(DATA_DIR, "sim_matrix_hpe_confirm.npz")
        if not os.path.exists(confirm_file):
            print("[sim_matrix_hpe_confirm] ファイル未生成 → sim_matrix_hpeにフォールバック")
            self.sim_matrix_hpe_confirm = self.sim_matrix_hpe
            return

        data = np.load(confirm_file, allow_pickle=True)
        cached_diseases = list(data.get("disease_names", []))
        cached_hpe = list(data.get("hpe_names", []))

        disease_names = sorted(self.disease_idx.keys())
        if cached_diseases == disease_names and cached_hpe == self.hpe_names:
            self.sim_matrix_hpe_confirm = data["sim_matrix"]
            print(f"[sim_matrix_hpe_confirm] confirm行列読込 {self.sim_matrix_hpe_confirm.shape}")
        else:
            print(f"[sim_matrix_hpe_confirm] リスト不一致 → sim_matrix_hpeにフォールバック")
            self.sim_matrix_hpe_confirm = self.sim_matrix_hpe

    def _batch_embed(self, texts, batch_size=50, max_workers=40):
        """テキストリストをバッチ並行でembedding。ndarray (N, dim) を返す。"""
        batches = []
        for i in range(0, len(texts), batch_size):
            batches.append((i, texts[i:i + batch_size]))

        all_embs = [None] * len(texts)

        def _embed_one(batch_info):
            start_idx, batch = batch_info
            for attempt in range(3):
                try:
                    resp = self.embed_client.embeddings.create(
                        model=EMBEDDING_MODEL, input=batch,
                    )
                    return start_idx, [item.embedding for item in resp.data]
                except Exception as e:
                    print(f"[embed] batch(offset={start_idx}) 失敗(試行{attempt+1}): {e}")
                    time.sleep(2 ** attempt)
            return start_idx, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_embed_one, b) for b in batches]
            for future in as_completed(futures):
                start_idx, result = future.result()
                if result is None:
                    return None
                for j, emb in enumerate(result):
                    all_embs[start_idx + j] = emb

        return np.array(all_embs, dtype=np.float32)

    # ----------------------------------------------------------------
    # 検査の質推定（差分ベクトル射影、キャッシュ付き）
    # ----------------------------------------------------------------
    def _compute_test_quality(self):
        """
        検査の quality_description をembedし、差分ベクトル軸への射影で
        質スコアを推定。

        quality_axis = normalize(good_emb - bad_emb)
        quality_score = dot(test_emb_normalized, quality_axis)

        good: 致命的疾患検出・治療直結・非侵襲・安全・安価・即時
        bad:  侵襲的・禁忌多数・合併症・高額・長時間・急性期不可

        test_weight = exp(quality_score)
        """
        cache_file = os.path.join(DATA_DIR, "test_quality.json")

        # tests.jsonlからquality_descriptionを読み込み
        tests_jsonl = os.path.join(DATA_DIR, "tests.jsonl")
        test_descriptions = {}
        if os.path.exists(tests_jsonl):
            with open(tests_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        td = json.loads(line)
                        desc = td.get("quality_description", "")
                        if desc:
                            test_descriptions[td["test_name"]] = desc
                    except (json.JSONDecodeError, KeyError):
                        continue
            print(f"[2Q] {len(test_descriptions)}件のquality_description読込")

        # 全検査名を収集
        all_test_names = sorted(set(self.test_names))

        # キャッシュが存在し、全検査名をカバーし、axis形式であればロード
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            missing = set(all_test_names) - set(cached.keys())
            sample = next(iter(cached.values()), None) if cached else None
            if not missing and isinstance(sample, dict) and "axis" in sample:
                self.test_quality = cached
                print(f"[2Q] キャッシュから{len(self.test_quality)}検査の差分軸スコア読込")
                return
            print(f"[2Q] キャッシュなしまたは旧形式、再計算")

        # Good/Badアンカー（所見即所得: 観察可能な記述のみ）
        good_text = (
            "末梢静脈からの採血のみで実施可能。穿刺部の軽微な疼痛以外に身体所見の変化を生じない。"
            "検査に伴う出血・感染・臓器損傷の所見が発生しない。"
            "急性期・血行動態不安定・重症・妊婦・小児でもバイタルサインに影響なく実施できる。"
            "検査は数分で完了し、結果は15分以内に数値として判明する。"
            "陽性であれば血圧低下・SpO2低下・意識障害・ショック所見が急速に進行する疾患を検出し、"
            "陰性であればこれらの急性所見が出現する可能性を除外できる。"
            "治療開始後の検査値変化（CRP低下、白血球正常化、トロポニン低下）をモニタリングでき、"
            "解熱・血圧安定化・SpO2改善・意識清明化といった臨床所見の改善と対応する。"
        )
        bad_text = (
            "全身麻酔下でカテーテル挿入・臓器穿刺・開腹を伴い、"
            "術後に疼痛・出血・発熱・創部感染の所見が高頻度に出現する。"
            "検査に伴い血圧低下・不整脈・気胸・臓器穿孔・アナフィラキシー所見が発生しうる。"
            "血行動態不安定・凝固異常・腎機能低下の患者では合併症所見が増悪する。"
            "前処置に数時間、結果判明に数日を要し、その間に病態が進行しうる。"
            "検査値の異常を検出しても、どの疾患による異常かの鑑別には追加検査所見が必要となる。"
            "治療前後の検査値変化を反映せず、臨床所見の改善をモニタリングできない。"
        )

        # embedするテキスト: [good, bad, test1, test2, ...]
        test_list = all_test_names
        embed_texts = []
        for tname in test_list:
            desc = test_descriptions.get(tname, "")
            embed_texts.append(desc if desc else tname)
        all_texts = [good_text, bad_text] + embed_texts

        # バッチに分割して並行embedding
        batch_size = 50
        batches = []
        for i in range(0, len(all_texts), batch_size):
            batches.append((i, all_texts[i:i + batch_size]))
        total_batches = len(batches)
        all_embs = [None] * len(all_texts)

        def _embed_batch(batch_info):
            start_idx, batch = batch_info
            for attempt in range(3):
                try:
                    resp = self.embed_client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=batch,
                    )
                    return start_idx, [np.array(item.embedding) for item in resp.data]
                except Exception as e:
                    print(f"[2Q] batch (offset={start_idx}) 失敗 (試行{attempt+1}): {e}")
                    time.sleep(2 ** attempt)
            return start_idx, None

        max_workers = 40
        print(f"[2Q] {total_batches}バッチを{max_workers}並行でembedding開始")
        failed = False
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_embed_batch, b) for b in batches]
            completed = 0
            for future in as_completed(futures):
                start_idx, result = future.result()
                if result is None:
                    print(f"[2Q] batch (offset={start_idx}) が3回失敗、中断")
                    failed = True
                    break
                for j, emb in enumerate(result):
                    all_embs[start_idx + j] = emb
                completed += 1
                if completed % 5 == 0 or completed == total_batches:
                    print(f"[2Q] embedding {completed}/{total_batches} バッチ完了")

        if failed:
            return

        good_emb = all_embs[0]
        bad_emb = all_embs[1]
        test_embs = all_embs[2:]

        # 差分ベクトル軸: good - bad を正規化
        axis = good_emb - bad_emb
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 0:
            axis = axis / axis_norm

        for i, tname in enumerate(test_list):
            t_emb = test_embs[i]
            t_norm = np.linalg.norm(t_emb)
            if t_norm > 0:
                t_emb = t_emb / t_norm
            proj = float(np.dot(t_emb, axis))
            self.test_quality[tname] = {
                "axis": proj,
            }

        # キャッシュに保存
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(self.test_quality, f, ensure_ascii=False, indent=1)

        print(f"[2Q] {len(self.test_quality)}検査の差分軸スコア計算完了")

    # ----------------------------------------------------------------
    # 検査侵襲性スコア計算（動的侵襲性バランシング用）
    # ----------------------------------------------------------------
    def _compute_invasive_scores(self):
        """
        各検査のquality_descriptionと侵襲性アンカーのcos類似度を計算。

        cos_invasive_j = cos(quality_desc_emb_j, invasive_anchor_emb)

        ランキング時に:
          expected_criticality = Σ(w_i × cos_critical_i)
          penalty_j = max(0, cos_invasive_j - expected_criticality)
          utility *= exp(-penalty_j)

        軽症なのに骨髄穿刺等の侵襲的検査がランキング上位に来るのを防ぐ。
        """
        cache_file = os.path.join(DATA_DIR, "invasive_scores.json")

        if not self.test_names:
            return

        # キャッシュチェック
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_names = cached.get("test_names", [])
            if cached_names == self.test_names and "scores" in cached:
                self.cos_invasive = np.array(cached["scores"], dtype=np.float32)
                print(f"[Invasive] キャッシュから{len(self.cos_invasive)}件読込")
                return

        # 侵襲性アンカー
        invasive_anchor = (
            "全身麻酔下でカテーテル挿入・臓器穿刺・開腹を伴い、"
            "術後に疼痛・出血・発熱・創部感染の所見が高頻度に出現する。"
            "血行動態不安定・凝固異常・腎機能低下の患者では合併症所見が増悪する。"
        )

        # quality_descriptionを取得（ない場合は検査名で代用）
        texts = []
        for tname in self.test_names:
            desc = self.test_db.get(tname, {}).get("quality_description", "")
            texts.append(desc if desc else tname)

        # [anchor] + [test1, test2, ...] を一括embed
        all_texts = [invasive_anchor] + texts
        embs = self._batch_embed(all_texts)
        if embs is None:
            print("[Invasive] embedding失敗")
            self.cos_invasive = np.zeros(len(self.test_names), dtype=np.float32)
            return

        # 正規化
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs_normed = embs / norms

        anchor_emb = embs_normed[0]     # (dim,)
        test_embs = embs_normed[1:]     # (N_tests, dim)

        # cos類似度
        self.cos_invasive = test_embs @ anchor_emb  # (N_tests,)

        # キャッシュ保存
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "test_names": self.test_names,
                "scores": self.cos_invasive.tolist(),
            }, f, ensure_ascii=False)

        # 上位/下位を表示
        sorted_idx = np.argsort(self.cos_invasive)[::-1]
        top5 = [(self.test_names[i], f"{self.cos_invasive[i]:.3f}") for i in sorted_idx[:5]]
        bot5 = [(self.test_names[i], f"{self.cos_invasive[i]:.3f}") for i in sorted_idx[-5:]]
        print(f"[Invasive] {len(self.cos_invasive)}検査の侵襲度計算完了")
        print(f"  Top5(侵襲大): {top5}")
        print(f"  Bot5(侵襲小): {bot5}")

    # ----------------------------------------------------------------
    # 検査リスクembedding計算（起動時に1回）
    # ----------------------------------------------------------------
    def _compute_risk_embeddings(self):
        """
        risk_descriptionを持つ検査のembeddingを計算・キャッシュ。
        risk_relevance = cos(risk_emb, patient_emb) で患者へのリスク関連度を推定。
        """
        cache_file = os.path.join(DATA_DIR, "risk_embs.npz")

        # risk_descriptionを持つ検査を収集
        risk_tests = {}
        for tname, tdata in self.test_db.items():
            desc = tdata.get("risk_description", "")
            if desc:
                risk_tests[tname] = desc

        if not risk_tests:
            print("[Risk] risk_descriptionなし、スキップ")
            return

        risk_names = sorted(risk_tests.keys())

        # キャッシュチェック
        if os.path.exists(cache_file):
            data = np.load(cache_file, allow_pickle=True)
            cached_names = list(data["test_names"])
            if cached_names == risk_names:
                for i, name in enumerate(risk_names):
                    self.risk_embs[name] = data["embeddings"][i]
                print(f"[Risk] キャッシュから{len(self.risk_embs)}検査のリスクembedding読込")
                return

        # embedding計算
        texts = [risk_tests[n] for n in risk_names]
        embs = self._batch_embed(texts)
        if embs is None:
            print("[Risk] embedding失敗")
            return

        # 正規化して保存
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs_normed = embs / norms

        for i, name in enumerate(risk_names):
            self.risk_embs[name] = embs_normed[i]

        # キャッシュ保存
        np.savez(
            cache_file,
            embeddings=embs_normed,
            test_names=np.array(risk_names, dtype=object),
        )
        print(f"[Risk] {len(self.risk_embs)}検査のリスクembedding計算・キャッシュ完了")

    # ----------------------------------------------------------------
    # 検査名embedding計算（novelty + 検査マッチ用）
    # ----------------------------------------------------------------
    def _compute_test_name_embs(self):
        """
        検査名をembedし、novelty計算と検査名マッチに使用。
        novelty = 1 - max_line cos(line_emb, test_name_emb_j)
        患者テキストに検査名が直接言及されていれば新規性が下がる。
        """
        cache_file = os.path.join(DATA_DIR, "test_name_embs.npz")

        if not self.test_names:
            return

        # キャッシュチェック
        if os.path.exists(cache_file):
            data = np.load(cache_file, allow_pickle=True)
            cached_names = list(data["test_names"])
            if cached_names == self.test_names:
                self.test_name_embs = data["embeddings"]
                print(f"[TestNameEmb] キャッシュから{self.test_name_embs.shape[0]}件読込")
                return

        # 検査名をバッチembedding
        embs = self._batch_embed(self.test_names)
        if embs is None:
            print("[TestNameEmb] embedding失敗")
            return

        # 正規化
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.test_name_embs = embs / norms

        # キャッシュ保存
        np.savez(
            cache_file,
            embeddings=self.test_name_embs,
            test_names=np.array(self.test_names, dtype=object),
        )
        print(f"[TestNameEmb] {self.test_name_embs.shape[0]}件計算・キャッシュ完了")

    # ----------------------------------------------------------------
    # 極性軸計算（正常←→異常、Option 3用）
    # ----------------------------------------------------------------
    def _compute_polarity_axis(self):
        """
        正常/異常の差分ベクトル軸を計算。2Qと同じパターン。

        polarity_axis = normalize(abnormal_emb - normal_emb)
        polarity = dot(text_emb, polarity_axis)
          > 0: 異常（陽性、上昇、検出）
          < 0: 正常（陰性、基準範囲内、異常なし）

        定性的テキストで完璧に分離（検証済み）。
        数値テキストは未対応（Step 2: 基準範囲テーブルで解決予定）。
        """
        normal_anchor = '検査値は基準範囲内であり正常である。異常所見を認めない。陰性。検出されず。'
        abnormal_anchor = '検査値は基準範囲を逸脱し異常である。異常所見あり。陽性。上昇。低下。検出。'

        try:
            resp = self.embed_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[normal_anchor, abnormal_anchor],
            )
            embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / norms

            axis = embs[1] - embs[0]  # abnormal - normal
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 0:
                axis = axis / axis_norm

            self.polarity_axis = axis
            print(f"[Polarity] 極性軸計算完了 (dim={len(axis)})")
        except Exception as e:
            print(f"[Polarity] 極性軸計算失敗: {e}")

    # ----------------------------------------------------------------
    # Option 3: 検査結果による疾患重み更新
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # 基準範囲テーブル読込 + エイリアスマップ構築
    # ----------------------------------------------------------------
    def _load_reference_ranges(self):
        """
        reference_ranges.json を読み込み、テスト名エイリアスマップを構築。
        エイリアスマップ: 略称/日本語名 → 正規テスト名（大文字小文字不問）
        """
        ref_file = os.path.join(DATA_DIR, "reference_ranges.json")
        if not os.path.exists(ref_file):
            print("[RefRange] reference_ranges.json なし")
            return

        with open(ref_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # 数値検査のみ保持
        for name, ref in raw.items():
            if ref is not None:
                self.reference_ranges[name] = ref

        # エイリアスマップ構築
        # 1. テスト名を分解: "ナトリウム (Na)" → ["ナトリウム", "Na"]
        for name in self.reference_ranges:
            tokens = re.split(r'[（()\s/・]+', name)
            for tok in tokens:
                tok = tok.strip('）) ')
                if tok:
                    self.range_alias[tok.lower()] = name
            # フル名もエイリアスに
            self.range_alias[name.lower()] = name

        # 2. よく使われる略称を手動追加
        manual_aliases = {
            "wbc": "白血球数", "白血球": "白血球数",
            "rbc": "赤血球数", "赤血球": "赤血球数",
            "hb": "ヘモグロビン", "hgb": "ヘモグロビン",
            "ht": "ヘマトクリット", "hct": "ヘマトクリット",
            "plt": "血小板数", "血小板": "血小板数",
            "t-bil": "総ビリルビン", "tb": "総ビリルビン",
            "d-bil": "直接ビリルビン", "db": "直接ビリルビン",
            "bs": "血糖 (随時/空腹時)", "glu": "血糖 (随時/空腹時)",
            "血糖": "血糖 (随時/空腹時)", "glucose": "血糖 (随時/空腹時)",
            "pct": "プロカルシトニン (PCT)",
            "pt-inr": "PT-INR (プロトロンビン時間)",
            "inr": "PT-INR (プロトロンビン時間)",
            "aptt": "APTT (活性化部分トロンボプラスチン時間)",
            "spo2": "動脈血酸素飽和度 (SpO2)",
            "bnp": "BNP", "tsh": "TSH",
            "fe": "血清鉄 (Fe)", "鉄": "血清鉄 (Fe)",
            "esr": "赤沈 (ESR)", "赤沈": "赤沈 (ESR)",
            "hba1c": "HbA1c", "a1c": "HbA1c",
            "ldh": "LDH", "alp": "ALP",
            "ck": "CK (CPK)", "cpk": "CK (CPK)",
            "bun": "尿素窒素 (BUN)",
            "rf": "リウマトイド因子 (RF)",
            "フェリチン": "フェリチン", "ferritin": "フェリチン",
            "乳酸": "乳酸", "lactate": "乳酸",
            "アンモニア": "アンモニア", "nh3": "アンモニア",
            # バイタルサイン
            "体温": "体温", "bt": "体温",
            "血圧": "収縮期血圧", "sbp": "収縮期血圧", "bp": "収縮期血圧",
            "dbp": "拡張期血圧",
            "脈拍": "脈拍数", "心拍": "脈拍数", "心拍数": "脈拍数",
            "hr": "脈拍数", "pulse": "脈拍数",
            "呼吸数": "呼吸数", "rr": "呼吸数",
            "spo2": "SpO2", "酸素飽和度": "SpO2",
        }
        for alias, canonical in manual_aliases.items():
            if canonical in self.reference_ranges:
                self.range_alias[alias.lower()] = canonical

        print(f"[RefRange] {len(self.reference_ranges)}検査の基準範囲読込、{len(self.range_alias)}エイリアス")

    # バイタルサイン臨床用語マッピング（embedding空間でより正確にマッチ）
    VITAL_SIGN_TERMS = {
        "体温": {"上昇": "発熱", "低下": "低体温", "正常": "体温 正常 異常なし"},
        "収縮期血圧": {"上昇": "高血圧", "低下": "低血圧", "正常": "血圧 正常 異常なし"},
        "拡張期血圧": {"上昇": "拡張期高血圧", "低下": "拡張期低血圧", "正常": "血圧 正常 異常なし"},
        "脈拍数": {"上昇": "頻脈", "低下": "徐脈", "正常": "脈拍 正常 異常なし"},
        "呼吸数": {"上昇": "頻呼吸", "低下": "徐呼吸", "正常": "呼吸数 正常 異常なし"},
        "SpO2": {"上昇": "SpO2 正常 異常なし", "低下": "低酸素血症", "正常": "SpO2 正常 異常なし"},
    }

    def _annotate_with_ranges(self, result_lines: list) -> list:
        """
        基準範囲テーブルで数値結果をアノテーション（確定的、LLM不要）。

        数値抽出 + エイリアスマッチ → 基準範囲参照 → 方向語付与。
        バイタルサインは臨床用語に変換（発熱、低血圧、頻脈等）。
        """
        if not self.reference_ranges:
            return result_lines

        annotated = []
        for line in result_lines:
            # 数値抽出（カンマ区切りに対応: 14,200 → 14200）
            num_match = re.search(r'(\d[\d,]*\.?\d*)', line)
            if not num_match:
                annotated.append(line)
                continue

            value = float(num_match.group(1).replace(',', ''))
            text_part = line[:num_match.start()].strip()
            # テキスト部分が空なら行全体から数値を除いた部分
            if not text_part:
                text_part = re.sub(r'[\d.]+', '', line).strip()

            # エイリアスマッチ（最長一致）
            matched_name = None
            text_lower = text_part.lower().strip()
            # 完全一致 → 前方一致 → 部分一致
            if text_lower in self.range_alias:
                matched_name = self.range_alias[text_lower]
            else:
                # 部分一致（3文字以上のエイリアスのみ、短いエイリアスの誤マッチ防止）
                best_len = 0
                for alias, canonical in self.range_alias.items():
                    if len(alias) >= 3 and alias in text_lower and len(alias) > best_len:
                        matched_name = canonical
                        best_len = len(alias)

            if matched_name is None:
                annotated.append(line)
                continue

            ref = self.reference_ranges[matched_name]
            lower, upper = ref["lower"], ref["upper"]

            # 方向判定
            if value > upper:
                direction = "上昇"
            elif value < lower:
                direction = "低下"
            else:
                direction = "正常"

            # バイタルサインは臨床用語に変換（embedding精度向上）
            if matched_name in self.VITAL_SIGN_TERMS:
                annotation = self.VITAL_SIGN_TERMS[matched_name][direction]
            elif direction != "正常":
                annotation = f"{matched_name} {direction}"
            else:
                annotation = f"{matched_name} 正常 異常なし"
            annotated.append(annotation)

        return annotated

    def _annotate_with_llm(self, symptoms: str, result_lines: list) -> list:
        """
        LLMで検査結果を文脈付きアノテーション（患者背景を考慮）。

        症状テキスト + 結果行 → LLMが臨床的解釈を付与。
        「正常なのに異常」「薬の影響」等の文脈依存判断が可能。
        """
        if not result_lines:
            return result_lines

        results_text = "\n".join(f"- {r}" for r in result_lines)
        prompt = f"""あなたは経験豊富な臨床医です。
患者情報を踏まえ各検査結果の臨床的意味を1行で記述してください。

ルール:
- この患者にとっての解釈で判定すること
- 薬の影響で予想される値は「治療域」「薬効による」等と明記
- 異常なら「上昇」「低下」「減少」「増加」等の方向語を含める
- 正常なら「正常」「異常なし」「陰性」等を含める
- 「正常範囲内」「基準範囲内」等の表現を使わないこと
- 各項目は1つの文字列で出力（JSON配列、オブジェクト不可）

患者: {symptoms}
結果:
{results_text}

出力例: ["血圧低下（高血圧患者として相対的低血圧）", "CRP上昇（炎症反応あり）", ...]"""

        try:
            content = self._llm_call([
                {"role": "system", "content": "JSON配列のみ出力。説明不要。"},
                {"role": "user", "content": prompt},
            ])
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                if isinstance(parsed, list) and len(parsed) == len(result_lines):
                    print(f"  [LLM注釈] {len(parsed)}件変換完了")
                    return parsed
        except Exception as e:
            print(f"  [LLM注釈] 失敗、基準範囲フォールバック: {e}")

        # LLM失敗時は基準範囲でフォールバック
        return self._annotate_with_ranges(result_lines)

    # ----------------------------------------------------------------
    # ヘルパー: 検査名マッチ / エイリアス解決
    # ----------------------------------------------------------------
    def _match_test_name(self, emb):
        """embeddingから最も近い検査名とcos類似度を返す"""
        if self.test_name_embs is not None:
            sims = emb @ self.test_name_embs.T
            best_j = int(np.argmax(sims))
            return self.test_names[best_j], float(sims[best_j])
        return "?", 0.0

    def _resolve_test_alias(self, test_name):
        """検査名を基準範囲テーブルの正規名に解決。見つからなければNone"""
        lower = test_name.lower()
        if lower in self.range_alias:
            return self.range_alias[lower]
        best_len = 0
        best_name = None
        for alias, canonical in self.range_alias.items():
            if len(alias) >= 3 and alias in lower and len(alias) > best_len:
                best_name = canonical
                best_len = len(alias)
        return best_name

    # ----------------------------------------------------------------
    # Option 3: 検査結果による疾患重み更新（直接法）
    # ----------------------------------------------------------------
    def update_from_results(self, candidates: list, result_lines: list,
                            symptoms: str = "") -> list:
        """
        検査結果から疾患重みを更新（直接法）。

        sim_matrixを使わず、結果テキストのembeddingを疾患embeddingと直接比較。

        異常結果: result_emb × disease_embs → excess → exp(+excess) で増幅
        正常結果（定量）: 双方向反実仮想 "検査名 異常 上昇/低下" →
                          exp(-(excess_up + excess_down)) で抑制
        正常結果（定性）: 単方向反実仮想 "検査名 異常" → exp(-excess) で抑制
        """
        if not result_lines or self.polarity_axis is None:
            return candidates
        if self.disease_embs_normed is None:
            return candidates

        # アノテーション（LLM優先、フォールバック: 基準範囲テーブル）
        if symptoms:
            annotated = self._annotate_with_llm(symptoms, result_lines)
        else:
            annotated = self._annotate_with_ranges(result_lines)

        # アノテーション済みテキストを一括embed
        try:
            resp = self.embed_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=annotated,
            )
        except Exception as e:
            print(f"[Option3] embedding失敗: {e}")
            return candidates

        line_embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        l_norms = np.linalg.norm(line_embs, axis=1, keepdims=True)
        l_norms[l_norms == 0] = 1.0
        line_embs = line_embs / l_norms

        # 各行の極性を一括計算
        polarities = line_embs @ self.polarity_axis  # (N_lines,)

        # --- 正常結果の反実仮想テキストを構築 ---
        cf_texts = []       # 反実仮想テキスト（一括embed用）
        cf_meta = {}        # line_idx → (cf_start_offset, is_bidirectional, test_name)

        for k in range(len(result_lines)):
            if polarities[k] > 0:
                continue  # 異常: 反実仮想不要

            test_name, _ = self._match_test_name(line_embs[k])
            canonical = self._resolve_test_alias(test_name)

            cf_start = len(cf_texts)
            if canonical and canonical in self.reference_ranges:
                # 定量検査 → 双方向
                cf_texts.append(f"{test_name} 異常 上昇")
                cf_texts.append(f"{test_name} 異常 低下")
                cf_meta[k] = (cf_start, True, test_name)
            else:
                # 定性検査 → 単方向
                cf_texts.append(f"{test_name} 異常")
                cf_meta[k] = (cf_start, False, test_name)

        # 反実仮想テキストを一括embed
        cf_embs = None
        if cf_texts:
            try:
                resp = self.embed_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=cf_texts,
                )
                cf_embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
                cf_norms = np.linalg.norm(cf_embs, axis=1, keepdims=True)
                cf_norms[cf_norms == 0] = 1.0
                cf_embs = cf_embs / cf_norms
            except Exception as e:
                print(f"[Option3] 反実仮想embedding失敗: {e}")
                cf_embs = None

        # --- 各結果行を処理 ---
        for k in range(len(result_lines)):
            orig = result_lines[k]
            ann = annotated[k]
            pol = float(polarities[k])
            log_ann = f" → [{ann}]" if ann != orig else ""

            if pol > 0:
                # === 異常結果: 直接法 ===
                # result_emb × disease_embs → 関連疾患を増幅
                sims_d = line_embs[k] @ self.disease_embs_normed.T
                bg = float(sims_d.mean())

                test_name, match_sim = self._match_test_name(line_embs[k])
                print(f"  [結果] {orig}{log_ann} → {test_name} (cos={match_sim:.3f}) "
                      f"pol={pol:+.4f} → 異常（直接法）")

                for c in candidates:
                    d_idx = self.disease_idx.get(c['disease_name'])
                    if d_idx is not None:
                        excess = max(0.0, float(sims_d[d_idx]) - bg)
                        if excess > 0:
                            c['similarity'] *= float(np.exp(excess))
            else:
                # === 正常結果: 反実仮想法 ===
                if cf_embs is None or k not in cf_meta:
                    continue

                cf_start, is_bidir, test_name = cf_meta[k]
                _, match_sim = self._match_test_name(line_embs[k])

                if is_bidir:
                    # 双方向: exp(-(excess_up + excess_down))
                    cf_up = cf_embs[cf_start]
                    cf_down = cf_embs[cf_start + 1]
                    sims_up = cf_up @ self.disease_embs_normed.T
                    sims_down = cf_down @ self.disease_embs_normed.T
                    bg_up = float(sims_up.mean())
                    bg_down = float(sims_down.mean())

                    print(f"  [結果] {orig}{log_ann} → {test_name} (cos={match_sim:.3f}) "
                          f"pol={pol:+.4f} → 正常（双方向反実仮想）")

                    for c in candidates:
                        d_idx = self.disease_idx.get(c['disease_name'])
                        if d_idx is not None:
                            e_up = max(0.0, float(sims_up[d_idx]) - bg_up)
                            e_down = max(0.0, float(sims_down[d_idx]) - bg_down)
                            total = e_up + e_down
                            if total > 0:
                                c['similarity'] *= float(np.exp(-total))
                else:
                    # 単方向: exp(-excess)
                    cf_emb = cf_embs[cf_start]
                    sims_d = cf_emb @ self.disease_embs_normed.T
                    bg = float(sims_d.mean())

                    print(f"  [結果] {orig}{log_ann} → {test_name} (cos={match_sim:.3f}) "
                          f"pol={pol:+.4f} → 正常（反実仮想）")

                    for c in candidates:
                        d_idx = self.disease_idx.get(c['disease_name'])
                        if d_idx is not None:
                            excess = max(0.0, float(sims_d[d_idx]) - bg)
                            if excess > 0:
                                c['similarity'] *= float(np.exp(-excess))

        # 降順ソート
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        return candidates

    # ----------------------------------------------------------------
    # LLM呼び出し（リトライ + フォールバック）
    # ----------------------------------------------------------------
    def _llm_call(self, messages, temperature=0.1, max_tokens=65536,
                  thinking_budget=0):
        """Vertex AI → lemonapi → 12ai の3段フォールバック

        thinking_budget: 思考トークン上限（0=思考無効、抽出/注釈タスク向け）
        """
        from google.genai import types as genai_types

        # OpenAI messages → Vertex contents変換
        system_text = ""
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            elif m["role"] == "user":
                contents.append(genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=m["content"])],
                ))
            elif m["role"] == "assistant":
                contents.append(genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text=m["content"])],
                ))

        # Vertex AI（プライマリ）
        try:
            config = genai_types.GenerateContentConfig(
                system_instruction=system_text if system_text else None,
                temperature=temperature,
                max_output_tokens=max_tokens,
                thinking_config=genai_types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                ),
            )
            resp = self.vertex_client.models.generate_content(
                model=VERTEX_MODEL,
                contents=contents,
                config=config,
            )
            return resp.text.strip()
        except Exception as e:
            print(f"[LLM] Vertex AI失敗: {e}")

        # lemonapi フォールバック
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                resp = self.llm_client.chat.completions.create(
                    model="[V]gemini-3-flash-preview",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[LLM] lemonapi失敗 (試行{attempt+1}/{LLM_MAX_RETRIES+1}): {e}")
                if attempt < LLM_MAX_RETRIES:
                    time.sleep(2 ** attempt)

        # 12ai フォールバック
        print("[LLM] 12aiフォールバックに切り替え")
        try:
            resp = self.llm_fallback_client.chat.completions.create(
                model=LLM_FALLBACK_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"全APIが失敗しました: {e}")

    # ----------------------------------------------------------------
    # テキスト自動分離: 症状/所見 vs 検査結果
    # ----------------------------------------------------------------
    def split_symptoms_results(self, text: str) -> tuple:
        """
        自由記述テキストを「陽性症状」「陰性所見」「検査結果」に自動分離。

        陽性所見 → embedding検索（search_diseases）に渡す
        陰性所見 → filter_contradictions に直接渡す（embeddingは否定を区別できない）
        検査結果 → update_from_results に渡す

        Returns: (positive_text: str, negative_findings: list[str], result_lines: list[str])
        """
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return "", [], []

        return self._split_with_llm(text, lines)

    def _split_with_llm(self, full_text: str, lines: list) -> tuple:
        """LLMでテキストを陽性症状・陰性所見・検査結果に3分離"""
        prompt = f"""以下の臨床テキストを3つのカテゴリに分離してください。

カテゴリ定義:
1. positive_findings: 陽性の症状・所見（存在する所見）
   - 主訴、現病歴、既往歴、バイタルサイン、陽性の身体所見
   - 例: 「発熱38度」「頭痛あり」「心窩部圧痛」「体温37.5℃」
2. negative_findings: 陰性の所見（否定された所見）
   - 「〜なし」「〜陰性」「〜認めず」「〜否定」「正常」等
   - 例: 「項部硬直なし」「Kernig徴候陰性」「Murphy徴候陰性」「発疹認めず」
3. results: 検査結果（血液検査、画像検査、生理検査の数値・結果）
   - 例: 「WBC 12000」「CRP 8.5」「胸部X線：浸潤影あり」

ルール:
- バイタルサインの数値（体温、血圧、脈拍、SpO2等）→ positive_findings
- 判断に迷う場合はpositive_findingsに含める
- negative_findingsは個別の所見を文字列リストで返す
- 【重要】数値は必ず臨床的意味を自然言語で付与して出力せよ（例: 「BP 70/40」→「収縮期血圧70台の重篤なショック状態」、「RR 35」→「著明な頻呼吸」、「WBC 18000」→「白血球著増」、「SpO2 82%」→「重度の低酸素血症」）。元の数値も残すこと。

出力形式（JSON）:
{{"positive_findings": "陽性テキスト（改行区切り）", "negative_findings": ["陰性所見1", "陰性所見2", ...], "results": ["検査結果1", "検査結果2", ...]}}

テキスト:
{full_text}"""

        try:
            content = self._llm_call([
                {"role": "system", "content": "JSON出力のみ。説明不要。"},
                {"role": "user", "content": prompt},
            ])
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                positive = parsed.get("positive_findings", parsed.get("symptoms", ""))
                negatives = parsed.get("negative_findings", [])
                results = parsed.get("results", [])
                if isinstance(results, str):
                    results = [r.strip() for r in results.split('\n') if r.strip()]
                if isinstance(negatives, str):
                    negatives = [n.strip() for n in negatives.split('\n') if n.strip()]
                print(f"  [分離LLM] 陽性{len(positive)}字, 陰性{len(negatives)}件, 検査結果{len(results)}件")
                return positive, negatives, results
        except Exception as e:
            print(f"  [分離LLM] 失敗、embedding分離にフォールバック: {e}")

        return self._split_with_embedding(lines)

    def _split_with_embedding(self, lines: list) -> tuple:
        """Embeddingベースで検査結果行を検出（LLM不要、陰性分離なし）"""
        if not lines or self.test_name_embs is None:
            return '\n'.join(lines), [], []

        # 全行をembed
        try:
            resp = self.embed_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=lines,
            )
            line_embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            norms = np.linalg.norm(line_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            line_embs = line_embs / norms
        except Exception:
            return '\n'.join(lines), [], []

        # 各行の検査名との最大cos → 閾値で分離
        sims = line_embs @ self.test_name_embs.T  # (N_lines, N_tests)
        max_sims = sims.max(axis=1)  # (N_lines,)

        # 極性も考慮: 検査結果は具体的な値を含むので極性が明確
        has_polarity = np.abs(line_embs @ self.polarity_axis) > 0.03

        symptoms = []
        results = []
        for i, line in enumerate(lines):
            # 検査名に高い類似度 かつ 極性がある → 検査結果
            if float(max_sims[i]) > 0.65 and has_polarity[i]:
                results.append(line)
            else:
                symptoms.append(line)

        print(f"  [分離Emb] 症状{len(symptoms)}行, 検査結果{len(results)}件")
        # embeddingフォールバックでは陰性分離不可（空リスト）
        return '\n'.join(symptoms), [], results

    # ----------------------------------------------------------------
    # Step 1: Novelty計算（v2: LLM二値判定 + embeddingフォールバック）
    # ----------------------------------------------------------------
    def compute_novelty(self, patient_text: str) -> np.ndarray:
        """
        v2哲学: 知覚はembedding、判断はLLM。

        noveltyは二値: 実施済み=0, 未実施=1。
        「やったかどうか」は論理判断 → LLMに任せる。
        LLM失敗時はembeddingギャップ検出にフォールバック。
        """
        n_tests = len(self.test_names)
        if n_tests == 0 or self.test_name_embs is None:
            return np.ones(n_tests)

        return self._compute_novelty_llm(patient_text)

    def _compute_novelty_llm(self, patient_text: str) -> np.ndarray:
        """LLMに既実施検査を抽出させて二値noveltyを返す（全件提示方式）"""
        n_tests = len(self.test_names)
        novelty = np.ones(n_tests)

        # 全372件をLLMに提示（Geminiのコンテキストなら余裕）
        test_list_json = json.dumps(self.test_names, ensure_ascii=False)

        prompt = f"""以下の臨床テキストを読み、すでに実施済みの検査を特定してください。

【推論ルール】
1. 「検査結果の数値や判定が記載されている」場合のみ実施済みとする。
   - 例: 「CRP 2.5」→ CRPは実施済み
   - 例: 「WBC 8500、Hb 12.0」→ CBC関連は実施済み
2. 患者の症状の訴え（「背中が痛い」「嘔吐した」等）は検査の実施を意味しない。
3. 文字列の一致ではなく「医学的な論理包含」でマッピングすること。
4. 出力は必ず下記マスタに存在する正確な名称のみ使用すること。
5. 結果の陽性/陰性は問わない。実施されたものを全て列挙。

【臨床テキスト】
{patient_text}

【VeSMed検査マスタ（全{n_tests}件）】
{test_list_json}

出力: JSON配列 ["検査名1", "検査名2", ...]"""

        try:
            content = self._llm_call([
                {"role": "system", "content": "JSON配列のみ出力。説明不要。"},
                {"role": "user", "content": prompt},
            ])
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                done_tests = json.loads(content[start:end])
                matched = 0
                for dt in done_tests:
                    if dt in self.test_idx:
                        novelty[self.test_idx[dt]] = 0.0
                        matched += 1
                print(f"  [Novelty LLM] {len(done_tests)}件抽出, {matched}件マッチ → 二値化")
                return novelty
        except Exception as e:
            print(f"  [Novelty LLM] 失敗、ギャップ法にフォールバック: {e}")

        return self._compute_novelty_gap(patient_text)

    def _compute_novelty_gap(self, patient_text: str) -> np.ndarray:
        """embeddingギャップ検出で二値noveltyを返す（LLM不要）"""
        n_tests = len(self.test_names)
        novelty = np.ones(n_tests)

        lines = [l.strip() for l in patient_text.split('\n') if l.strip()]
        if not lines:
            return novelty

        try:
            resp = self.embed_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=lines,
            )
            line_embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            l_norms = np.linalg.norm(line_embs, axis=1, keepdims=True)
            l_norms[l_norms == 0] = 1.0
            line_embs = line_embs / l_norms
        except Exception:
            return novelty

        # 各検査の最大cos
        sims = line_embs @ self.test_name_embs.T
        max_cos = sims.max(axis=0)  # (N_tests,)

        # ギャップ検出: ソートして最大の隙間を見つける
        sorted_cos = np.sort(max_cos)[::-1]
        gaps = sorted_cos[:-1] - sorted_cos[1:]

        if len(gaps) > 0:
            # 上位20%の中で最大ギャップを探す（実施済みは少数のはず）
            search_range = max(5, len(gaps) // 5)
            best_gap_idx = int(np.argmax(gaps[:search_range]))
            threshold = (sorted_cos[best_gap_idx] + sorted_cos[best_gap_idx + 1]) / 2

            # 閾値が低すぎる場合は抑制しない（ギャップが不明瞭）
            if threshold > 0.55:
                novelty[max_cos >= threshold] = 0.0
                n_done = int((max_cos >= threshold).sum())
                print(f"  [Novelty Gap] 閾値={threshold:.3f}, {n_done}件抑制")
            else:
                print(f"  [Novelty Gap] ギャップ不明瞭(th={threshold:.3f}), 抑制なし")
        else:
            print(f"  [Novelty Gap] 検査なし")

        return novelty

    # ----------------------------------------------------------------
    # Step 2: ベクトル検索 → 候補疾患
    # ----------------------------------------------------------------
    def _embed_and_search(self, text: str) -> list:
        """
        MEAN-chunk統一: query_emb × disease_embs_normed.T で全疾患の類似度を一括計算。
        """
        resp = self.embed_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text],
        )
        query_embedding = resp.data[0].embedding
        # 最後のクエリembeddingを保持（risk_relevance計算用）
        self._last_query_embedding = np.array(query_embedding, dtype=np.float32)

        if self.disease_embs_normed is None:
            return []

        # クエリembeddingを正規化
        q_emb = self._last_query_embedding.copy()
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb /= q_norm

        # 全疾患とのcosine類似度（MEAN-chunk統一）
        sims = q_emb @ self.disease_embs_normed.T  # (N_diseases,)

        # disease_idx の逆引き
        idx_to_name = [""] * len(self.disease_idx)
        for dname, idx in self.disease_idx.items():
            idx_to_name[idx] = dname

        results = []
        for i, sim in enumerate(sims):
            dname = idx_to_name[i]
            if not dname:
                continue
            meta = self.disease_db.get(dname, {})
            results.append({
                "disease_name": dname,
                "similarity": float(sim),
                "category": meta.get("category", ""),
                "urgency": meta.get("urgency", ""),
            })

        return results

    def search_diseases(self, patient_text: str) -> list:
        """
        患者テキストで全疾患ベクトル検索（LLM rewrite不要、生テキストを直接embed）。
        返り値: [{"disease_name": str, "similarity": float, "category": str}, ...]
        """
        return self._embed_and_search(patient_text)

    # ----------------------------------------------------------------
    # Step 3: Softmax → 先験確率
    # ----------------------------------------------------------------
    def compute_priors(self, candidates: list) -> list:
        """
        候補疾患に臨床重みを付与する。
        softmax廃止: 生のcosine類似度をそのまま使う（embeddingの距離を直接信じる）。
        """
        if not candidates:
            return candidates

        for c in candidates:
            c["clinical_weight"] = self.disease_2c.get(c["disease_name"], {}).get("weight", 1.0)

        return candidates

    # ----------------------------------------------------------------
    # Step 3.5: LLM論理フィルタ（v2: 矛盾疾患を除外）
    # ----------------------------------------------------------------
    def filter_contradictions(self, candidates: list, patient_text: str,
                             negative_findings: list = None) -> tuple:
        """
        v2哲学: 判断はLLM。

        embedding検索は意味的類似性で候補を出すが、論理的矛盾を見逃す:
        - 「ANA陰性」なのにSLEが上位（embeddingは否定を区別できない）
        - 「トロポニン陰性」なのにSTEMIが上位
        - 男性なのにPID/卵巣嚢腫が上位

        LLMが論理矛盾を検出し、候補から除外する。
        embedding（知覚）の出力をLLM（判断）がフィルタする接着層。

        negative_findings: 陰性所見リスト（split_symptoms_resultsで分離された否定所見）。
        これにより「項部硬直なし」→ 髄膜炎除外 等がより確実に機能する。

        Returns: (filtered_candidates, exclusion_reasons)
          exclusion_reasons: [{"disease_name": str, "reason": str}, ...]
        """
        # 上位20疾患をLLMに提示
        top_n = min(20, len(candidates))
        disease_list = [f"- {c['disease_name']}" for c in candidates[:top_n]]

        # 陰性所見セクション
        neg_section = ""
        if negative_findings:
            neg_list = "\n".join(f"- {nf}" for nf in negative_findings)
            neg_section = f"""
【参考: 以下の陰性所見が確認されています】
{neg_list}
上記の陰性所見が疾患の「診断に必須な所見」を否定している場合のみ除外対象。
リスク因子の否定や、非典型例で欠如しうる所見の否定では除外しないこと。
"""

        prompt = f"""以下の患者テキストと候補疾患リストを見て、
患者テキストの内容と**論理的に矛盾する**疾患を指摘してください。

論理的矛盾とは（以下のいずれかに該当する場合のみ）:
- 必須検査が陰性なのにその疾患が候補（例: ANA陰性→SLE除外）
- 性別と合わない疾患（例: 男性→PID除外）
- 年齢と合わない疾患（例: 28歳→リウマチ性多発筋痛症は通常50歳以上）
- 明確に否定された所見がその疾患の**診断に必須**である場合

※ 確実な矛盾のみ除外。迷ったら除外しない。

【除外してはいけないケース（重要）】
- リスク因子の1つが否定されただけの疾患は除外禁止
  例: 飲酒なし→急性膵炎を除外してはならない（胆石・高TG・薬剤性・自己免疫性など非飲酒性の原因が多数）
  例: 喫煙なし→肺癌を除外してはならない（非喫煙者の肺癌は存在する）
- 典型的症状の一部が欠如しているだけの疾患は除外禁止（非典型例・亜型が存在する）
  例: 疝痛なし→尿管結石を除外してはならない（持続痛で来院する尿管結石は存在する）
  例: 右下腹部痛なし→急性虫垂炎を除外してはならない（後腹膜虫垂・骨盤内虫垂では背部痛や下痢で発症しうる）
  例: 咽頭痛なし→扁桃周囲膿瘍を除外してはならない（嚥下困難のみの場合がある）
- 疾患の原因が複数あるのに、1つの原因を否定しただけでは除外禁止
- 痛みの部位が典型と異なるだけでは除外禁止（関連痛・放散痛・非典型的局在は頻繁にある）

患者テキスト:
{patient_text}
{neg_section}
候補疾患:
{chr(10).join(disease_list)}

出力（JSON配列、各要素に疾患名と除外理由を含む）:
[{{"disease": "疾患名1", "reason": "除外理由1"}}, ...]
矛盾なしなら空配列 []"""

        exclusion_reasons = []
        try:
            content = self._llm_call([
                {"role": "system", "content": "JSON配列のみ出力。説明不要。"},
                {"role": "user", "content": prompt},
            ])
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                if parsed:
                    # 新形式（{disease, reason}）と旧形式（文字列）の両方に対応
                    contra_set = set()
                    for item in parsed:
                        if isinstance(item, dict):
                            dname = item.get("disease", "")
                            reason = item.get("reason", "")
                            contra_set.add(dname)
                            exclusion_reasons.append({
                                "disease_name": dname,
                                "reason": reason,
                            })
                        elif isinstance(item, str):
                            contra_set.add(item)
                            exclusion_reasons.append({
                                "disease_name": item,
                                "reason": "",
                            })
                    filtered = [c for c in candidates if c["disease_name"] not in contra_set]
                    n_removed = len(candidates) - len(filtered)
                    if n_removed > 0:
                        names = [r["disease_name"] for r in exclusion_reasons]
                        print(f"  [LLMフィルタ] {n_removed}疾患除外: {names}")
                        return filtered, exclusion_reasons
        except Exception as e:
            print(f"  [LLMフィルタ] 失敗（スキップ）: {e}")

        return candidates, []

    # ----------------------------------------------------------------
    # Step 4: 検査ランキング（prior加重分散）
    # ----------------------------------------------------------------
    def _compute_invasive_penalty(self, candidates: list, w: np.ndarray) -> np.ndarray:
        """
        動的侵襲性バランシング: 候補群の重症度に応じた侵襲ペナルティを計算。

        expected_criticality = Σ(w_i × cos_critical_i)  — 候補群の加重平均critical度
        penalty_j = max(0, cos_invasive_j - expected_criticality)

        返り値: exp(-penalty) (N_tests,) — utilityに乗算する
        """
        n_tests = len(self.test_names)
        if self.cos_invasive is None or len(self.cos_invasive) != n_tests:
            return np.ones(n_tests)

        # expected_criticality: 候補疾患群の加重平均critical度
        critical_scores = np.array([
            self.disease_2c.get(c["disease_name"], {}).get("critical", 0.0)
            for c in candidates
        ], dtype=float)
        expected_criticality = float(np.dot(w, critical_scores))

        # penalty_j = max(0, cos_invasive_j - expected_criticality)
        penalty = np.maximum(0.0, self.cos_invasive - expected_criticality)
        return np.exp(-penalty)

    def rank_tests(self, candidates: list, novelty: np.ndarray = None) -> list:
        """
        各検査について、疑い疾患群に対するcos類似度のprior加重分散を計算。
        分散が大きい = 疾患間でバラつく = 鑑別に有用な検査。

        utility = variance × novelty × invasive_discount

        novelty: v2二値判定（実施済み=0, 未実施=1）。
        """
        if self.sim_matrix is None or len(self.test_names) == 0:
            return []

        n_tests = len(self.test_names)
        if novelty is None:
            novelty = np.ones(n_tests)

        # 候補疾患の生cos類似度と2C重みを配列化
        raw_sims = np.array([c.get("similarity", 0.0) for c in candidates], dtype=float)
        weights = np.array([
            self.disease_2c.get(c["disease_name"], {}).get("weight", 1.0)
            for c in candidates
        ], dtype=float)
        # 局所重み: 平均以上の疾患のみ（遠い疾患はゼロ）
        sim_centered = np.maximum(0.0, raw_sims - raw_sims.mean())
        w = sim_centered * weights
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum

        # 候補疾患のsim_matrix行を取得 → (N_candidates, N_tests)
        disease_rows = []
        for c in candidates:
            row = self.disease_idx.get(c["disease_name"])
            disease_rows.append(row if row is not None else -1)
        disease_rows = np.array(disease_rows)

        valid_mask = disease_rows >= 0
        sim_sub = np.zeros((len(candidates), n_tests))
        sim_sub[valid_mask] = self.sim_matrix[disease_rows[valid_mask]]

        # prior加重分散を一括計算: Var_w = Σ w_i (x_i - μ)^2
        w_col = w[:, np.newaxis]  # (N_candidates, 1)
        mu = (w_col * sim_sub).sum(axis=0)  # (N_tests,)
        var = (w_col * (sim_sub - mu) ** 2).sum(axis=0)  # (N_tests,)

        # risk_relevance計算用: patient embeddingを正規化
        patient_emb = None
        if self._last_query_embedding is not None and len(self.risk_embs) > 0:
            pe = self._last_query_embedding
            pe_norm = np.linalg.norm(pe)
            if pe_norm > 0:
                patient_emb = pe / pe_norm

        # 動的侵襲性バランシング
        invasive_discount = self._compute_invasive_penalty(candidates, w)

        # 関連疾患: 各検査で重みが高い上位疾患を抽出
        # weighted_sim_sub = w * sim_sub (各疾患の検査への寄与度)
        weighted_contrib = w_col * sim_sub  # (N_candidates, N_tests)

        ranked = []
        for j, tname in enumerate(self.test_names):
            score = float(var[j])
            nov = float(novelty[j])
            inv_disc = float(invasive_discount[j])
            utility = score * nov * inv_disc

            # 関連疾患Top5（重み付き寄与度順）
            contribs = weighted_contrib[:, j]
            top_idx = np.argsort(contribs)[::-1][:5]
            details = [
                {"disease_name": candidates[int(k)]["disease_name"],
                 "contribution": round(float(contribs[k]), 6)}
                for k in top_idx if contribs[k] > 0
            ]

            ranked.append({
                "test_name": tname,
                "score": round(score, 6),
                "novelty": round(nov, 4),
                "utility": round(utility, 6),
                "details": details,
            })

        ranked.sort(key=lambda x: x["utility"], reverse=True)
        return ranked

    def rank_tests_confirm(self, candidates: list, novelty: np.ndarray = None) -> list:
        """
        Part C: 確認・同定推奨 — クラスタ特異度ランキング。
        confirm_score_j = cluster_mu_j - global_mu_j

        cluster_mu: 候補疾患群の加重平均類似度（Part Aと同じ重み）
        global_mu:  全疾患の非加重平均類似度（背景基準）

        差分 = この検査が候補群にどれだけ「特異的」か。
        CRPのような汎用検査はglobal_muも高い → 差分小 → 沈む。
        血液培養のような特異的検査はcluster_mu >> global_mu → 差分大 → 浮上。
        ハイパーパラメータなし。
        """
        # Dual sim_matrix: Part Cは確認用行列を使用
        sm = self.sim_matrix_confirm if self.sim_matrix_confirm is not None else self.sim_matrix
        if sm is None or len(self.test_names) == 0:
            return []

        n_tests = len(self.test_names)
        if novelty is None:
            novelty = np.ones(n_tests)

        # Part Aと同じ重み: 局所重み(平均以上) × 2C重み
        raw_sims = np.array([c.get("similarity", 0.0) for c in candidates], dtype=float)
        weights = np.array([
            self.disease_2c.get(c["disease_name"], {}).get("weight", 1.0)
            for c in candidates
        ], dtype=float)
        sim_centered = np.maximum(0.0, raw_sims - raw_sims.mean())
        w = sim_centered * weights
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum

        # sim_matrix_confirm行を取得
        disease_rows = []
        for c in candidates:
            row = self.disease_idx.get(c["disease_name"])
            disease_rows.append(row if row is not None else -1)
        disease_rows = np.array(disease_rows)

        valid_mask = disease_rows >= 0
        sim_sub = np.zeros((len(candidates), n_tests))
        sim_sub[valid_mask] = sm[disease_rows[valid_mask]]

        # クラスタ加重平均 vs 全疾患背景平均
        w_col = w[:, np.newaxis]
        cluster_mu = (w_col * sim_sub).sum(axis=0)  # (N_tests,)
        global_mu = sm.mean(axis=0)                   # (N_tests,)
        confirm = cluster_mu - global_mu               # クラスタ特異度

        # risk_relevance計算用
        patient_emb = None
        if self._last_query_embedding is not None and len(self.risk_embs) > 0:
            pe = self._last_query_embedding
            pe_norm = np.linalg.norm(pe)
            if pe_norm > 0:
                patient_emb = pe / pe_norm

        # 動的侵襲性バランシング
        invasive_discount = self._compute_invasive_penalty(candidates, w)

        # 関連疾患
        weighted_contrib = w_col * sim_sub

        ranked = []
        for j, tname in enumerate(self.test_names):
            score = float(confirm[j])
            nov = float(novelty[j])
            inv_disc = float(invasive_discount[j])
            utility = score * nov * inv_disc

            # 関連疾患Top5
            contribs = weighted_contrib[:, j]
            top_idx = np.argsort(contribs)[::-1][:5]
            details = [
                {"disease_name": candidates[int(k)]["disease_name"],
                 "contribution": round(float(contribs[k]), 6)}
                for k in top_idx if contribs[k] > 0
            ]

            ranked.append({
                "test_name": tname,
                "confirm_score": round(score, 6),
                "novelty": round(nov, 4),
                "utility": round(utility, 6),
                "details": details,
            })

        ranked.sort(key=lambda x: x["utility"], reverse=True)
        return ranked

    def rank_tests_cluster_mu(self, candidates: list, novelty: np.ndarray = None) -> list:
        """
        Part E: 基本推奨 — 候補群の共通必要度ランキング。
        cluster_mu_j = Σ w_i × sim_matrix[i][j]

        Part C（特異度=cluster_mu - global_mu）と違い、global_muを引かない。
        「候補疾患群に共通して関連する検査」を推薦する。
        血液培養、CBC、血液ガス等のルーチン検査が浮上。
        Part A（鑑別）、Part C（確定）と直交する第3の観点。
        """
        if self.sim_matrix is None or len(self.test_names) == 0:
            return []

        n_tests = len(self.test_names)
        if novelty is None:
            novelty = np.ones(n_tests)

        # Part A/Cと同じ重み: 局所重み(平均以上) × 2C重み
        raw_sims = np.array([c.get("similarity", 0.0) for c in candidates], dtype=float)
        weights = np.array([
            self.disease_2c.get(c["disease_name"], {}).get("weight", 1.0)
            for c in candidates
        ], dtype=float)
        sim_centered = np.maximum(0.0, raw_sims - raw_sims.mean())
        w = sim_centered * weights
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum

        # sim_matrix行を取得
        disease_rows = []
        for c in candidates:
            row = self.disease_idx.get(c["disease_name"])
            disease_rows.append(row if row is not None else -1)
        disease_rows = np.array(disease_rows)

        valid_mask = disease_rows >= 0
        sim_sub = np.zeros((len(candidates), n_tests))
        sim_sub[valid_mask] = self.sim_matrix[disease_rows[valid_mask]]

        # クラスタ加重平均（global_muを引かない = 共通必要度）
        w_col = w[:, np.newaxis]
        cluster_mu = (w_col * sim_sub).sum(axis=0)  # (N_tests,)

        # 動的侵襲性バランシング
        invasive_discount = self._compute_invasive_penalty(candidates, w)

        # 関連疾患
        weighted_contrib = w_col * sim_sub

        ranked = []
        for j, tname in enumerate(self.test_names):
            score = float(cluster_mu[j])
            nov = float(novelty[j])
            inv_disc = float(invasive_discount[j])
            utility = score * nov * inv_disc

            # 関連疾患Top5
            contribs = weighted_contrib[:, j]
            top_idx = np.argsort(contribs)[::-1][:5]
            details = [
                {"disease_name": candidates[int(k)]["disease_name"],
                 "contribution": round(float(contribs[k]), 6)}
                for k in top_idx if contribs[k] > 0
            ]

            ranked.append({
                "test_name": tname,
                "cluster_mu": round(score, 6),
                "novelty": round(nov, 4),
                "utility": round(utility, 6),
                "details": details,
            })

        ranked.sort(key=lambda x: x["utility"], reverse=True)
        return ranked

    def rank_tests_critical(self, candidates: list, novelty: np.ndarray = None) -> list:
        """
        Part B: Critical Hit — 致命疾患排除ランキング。
        critical_hit_j = max_i [ exp(cos_critical_i) * similarity_i * sim_matrix[i][j] ]

        分散ではなく最大命中: 「見逃したら死ぬ疾患を排除できる検査」を上位に。
        noveltyで既知情報を連続的に割引。
        """

        # Dual sim_matrix: Part Bは確認用行列を使用
        sm = self.sim_matrix_confirm if self.sim_matrix_confirm is not None else self.sim_matrix
        if sm is None or len(self.test_names) == 0:
            return []

        n_tests = len(self.test_names)
        if novelty is None:
            novelty = np.ones(n_tests)

        # critical_weight = exp(cos_critical) （curable除外、critical成分のみ）
        critical_w = np.array([
            math.exp(self.disease_2c.get(c["disease_name"], {}).get("critical", 0.0))
            for c in candidates
        ], dtype=float)

        # 生cosine類似度（softmax廃止、embeddingの距離を直接信じる）
        sims = np.array([c.get("similarity", 0.0) for c in candidates], dtype=float)

        # cp = critical_weight * similarity (N_candidates,)
        cp = critical_w * sims

        # sim_matrix_confirm行を取得
        disease_rows = []
        for c in candidates:
            row = self.disease_idx.get(c["disease_name"])
            disease_rows.append(row if row is not None else -1)
        disease_rows = np.array(disease_rows)

        valid_mask = disease_rows >= 0
        sim_sub = np.zeros((len(candidates), n_tests))
        sim_sub[valid_mask] = sm[disease_rows[valid_mask]]

        # weighted_sim = cp[:, None] * sim_sub → (N_candidates, N_tests)
        weighted_sim = cp[:, np.newaxis] * sim_sub

        # max over diseases
        critical_scores = weighted_sim.max(axis=0)       # (N_tests,)
        best_disease_idx = weighted_sim.argmax(axis=0)    # (N_tests,)

        # risk_relevance計算用
        patient_emb = None
        if self._last_query_embedding is not None and len(self.risk_embs) > 0:
            pe = self._last_query_embedding
            pe_norm = np.linalg.norm(pe)
            if pe_norm > 0:
                patient_emb = pe / pe_norm

        # 動的侵襲性バランシング（Part Bは局所重み×2C重みで計算）
        raw_sims_b = np.array([c.get("similarity", 0.0) for c in candidates], dtype=float)
        weights_b = np.array([
            self.disease_2c.get(c["disease_name"], {}).get("weight", 1.0)
            for c in candidates
        ], dtype=float)
        sim_centered_b = np.maximum(0.0, raw_sims_b - raw_sims_b.mean())
        w_b = sim_centered_b * weights_b
        w_b_sum = w_b.sum()
        if w_b_sum > 0:
            w_b = w_b / w_b_sum
        invasive_discount = self._compute_invasive_penalty(candidates, w_b)

        ranked = []
        for j, tname in enumerate(self.test_names):
            ch = float(critical_scores[j])
            nov = float(novelty[j])
            inv_disc = float(invasive_discount[j])
            utility = ch * nov * inv_disc

            # 最大命中疾患
            bi = int(best_disease_idx[j])
            hit_disease = candidates[bi]["disease_name"] if bi < len(candidates) else ""

            ranked.append({
                "test_name": tname,
                "critical_hit": round(ch, 6),
                "novelty": round(nov, 4),
                "utility": round(utility, 6),
                "hit_disease": hit_disease,
            })

        ranked.sort(key=lambda x: x["utility"], reverse=True)
        return ranked

    @staticmethod
    def _weighted_entropy(probs, weights):
        """重み付きエントロピー H_w = -Σ w_i × p_i × log2(p_i)"""
        mask = probs > 0
        if not np.any(mask):
            return 0.0
        return float(-np.sum(weights[mask] * probs[mask] * np.log2(probs[mask])))

    # ----------------------------------------------------------------
    # Step 5: LLMで所見を解釈 → 疾患ごとにベイズ更新
    # ----------------------------------------------------------------
    def interpret_and_update(self, candidates: list, test_name: str, finding: str) -> tuple:
        """
        検査所見をLLMで解釈し、各候補疾患について陽性/陰性+Se/Spを判定してベイズ更新。

        - 格納済みSe/Spがある検査: LLMは陽性/陰性のみ判定、Se/SpはDBから取得
        - 格納されていない所見（症状・身体所見・ROS等）: LLMがSe/Spも推定

        返り値: (updated_candidates, interpretation_dict)
        """
        normalized_name = self.test_name_map.get(test_name, test_name)

        # 各疾患のSe/Sp + purpose を取得（格納済みのもの）
        disease_se_sp = {}
        disease_purpose = {}
        for c in candidates:
            d = self.disease_db.get(c["disease_name"])
            if not d:
                continue
            for t in d.get("relevant_tests", []):
                tname = self.test_name_map.get(t["test_name"], t["test_name"])
                if tname == normalized_name:
                    se = t.get("sensitivity", [0.5, 0.5])
                    sp = t.get("specificity", [0.5, 0.5])
                    se_val = se[0] if isinstance(se, list) else se
                    sp_val = sp[0] if isinstance(sp, list) else sp
                    disease_se_sp[c["disease_name"]] = (se_val, sp_val)
                    disease_purpose[c["disease_name"]] = t.get("purpose", "")
                    break

        # Se/Spが格納されているかどうかでプロンプトを分岐
        has_stored_se_sp = len(disease_se_sp) > 0

        # 全候補疾患を対象にする（格納なしの所見でも全疾患に影響しうる）
        disease_lines = []
        for c in candidates:
            name = c["disease_name"]
            if name in disease_purpose:
                disease_lines.append(f"- {name}（検査目的: {disease_purpose[name]}）")
            else:
                disease_lines.append(f"- {name}")

        diseases_text = "\n".join(disease_lines)

        if has_stored_se_sp:
            # 格納済み検査: 陽性/陰性の判定のみ
            system_msg = (
                "あなたは経験豊富な日本の臨床医です。\n"
                "検査所見を受け取り、各候補疾患について、この所見がその疾患の\n"
                "「陽性所見（支持する）」か「陰性所見（否定する）」かを判定してください。\n"
                "出力はJSONのみ。説明文やマークダウンは不要。最初の文字は { にすること。"
            )
            user_msg = (
                f"検査名: {normalized_name}\n"
                f"所見: {finding}\n\n"
                f"候補疾患:\n{diseases_text}\n\n"
                f"各疾患について判定してください。出力形式:\n"
                f'{{"疾患名": {{"判定": "陽性" or "陰性", "理由": "簡潔な理由"}}, ...}}'
            )
        else:
            # 未格納の所見（症状・身体所見・ROS等）: 判定 + Se/Sp推定
            system_msg = (
                "あなたは経験豊富な日本の臨床医です。\n"
                "臨床所見（症状・身体所見・検査結果など）を受け取り、\n"
                "各候補疾患について以下を判定してください:\n"
                "1. この所見がその疾患を「陽性（支持する）」か「陰性（否定する）」か\n"
                "2. この所見のその疾患に対する感度(sensitivity)と特異度(specificity)の推定値\n"
                "   - 教科書・ガイドラインの一般的な値を参考に\n"
                "   - 不明な場合は臨床経験に基づく合理的な推定でよい\n"
                "出力はJSONのみ。説明文やマークダウンは不要。最初の文字は { にすること。"
            )
            user_msg = (
                f"所見: {test_name}: {finding}\n\n"
                f"候補疾患:\n{diseases_text}\n\n"
                f"各疾患について判定してください。出力形式:\n"
                f'{{"疾患名": {{"判定": "陽性" or "陰性", "se": 0.0~1.0, "sp": 0.0~1.0, "理由": "簡潔な理由"}}, ...}}'
            )

        content = self._llm_call(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        # LLM出力をパース
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
        if match:
            content = match.group(1).strip()
        start = content.find("{")
        if start > 0:
            content = content[start:]
        content = re.sub(r",\s*([}\]])", r"\1", content)

        try:
            interpretation = json.loads(content)
        except json.JSONDecodeError:
            interpretation = {c["disease_name"]: {"判定": "陽性", "理由": "解釈不能"} for c in candidates}

        # ベイズ更新（疾患ごとに陽性/陰性が異なる）
        # 原則: 検査と無関係な疾患は LR=1.0（類似度不変）
        sims = np.array([c.get("similarity", 0.0) for c in candidates])
        posteriors = np.zeros_like(sims)

        for i, c in enumerate(candidates):
            s = sims[i]
            name = c["disease_name"]
            interp = interpretation.get(name, {})
            is_positive = interp.get("判定", "陽性") == "陽性"

            if name in disease_se_sp:
                se, sp = disease_se_sp[name]
                if is_positive:
                    posteriors[i] = s * se
                else:
                    posteriors[i] = s * (1 - se)
            elif not has_stored_se_sp:
                se = float(interp.get("se", 0.5))
                sp = float(interp.get("sp", 0.5))
                if is_positive:
                    posteriors[i] = s * se
                else:
                    posteriors[i] = s * (1 - se)
            else:
                posteriors[i] = s

        # 正規化
        total = posteriors.sum()
        if total > 0:
            posteriors = posteriors / total

        updated = []
        for i, c in enumerate(candidates):
            updated.append({
                **c,
                "similarity": float(posteriors[i]),
            })
        updated.sort(key=lambda x: x["similarity"], reverse=True)

        return updated, interpretation

    def find_matching_tests(self, candidates: list, done_tests: list = None) -> list:
        """
        全329検査からdone_testsを除外した検査名リストを返す。
        """
        if done_tests is None:
            done_tests = []
        done_set = set(self.test_name_map.get(t, t) for t in done_tests)
        return sorted(t for t in self.test_names if t not in done_set)

    # ================================================================
    # Part D: 問診・身体診察推奨
    # ================================================================

    def _load_hpe_items(self):
        """hpe_items.jsonl を読み込み。"""
        if not os.path.exists(HPE_ITEMS_JSONL):
            print("[HPE] hpe_items.jsonl なし、Part Dスキップ")
            return
        with open(HPE_ITEMS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    self.hpe_items.append(item)
                except (json.JSONDecodeError, KeyError):
                    continue
        self.hpe_names = [it["item_name"] for it in self.hpe_items]
        self.hpe_idx = {name: i for i, name in enumerate(self.hpe_names)}
        print(f"[HPE] {len(self.hpe_items)}項目読込 (Hx: {sum(1 for it in self.hpe_items if it['category']=='Hx')}, PE: {sum(1 for it in self.hpe_items if it['category']=='PE')})")

    def _compute_hpe_similarity_matrix(self):
        """
        仮説embedding方式（Part A-Cと同一数学）:
        sim_matrix_hpe[d][k] = cos(E(hypothesis_k), disease_emb_d)
        """
        cache_file = os.path.join(DATA_DIR, "sim_matrix_hpe.npz")

        if self.disease_embs_normed is None:
            print("[HPE] 疾患embeddingなし、スキップ")
            return

        hypotheses = [it["hypothesis"] for it in self.hpe_items]
        n_hpe = len(hypotheses)

        # 疾患名リスト（disease_idxと同順）
        disease_names = [""] * len(self.disease_idx)
        for dname, idx in self.disease_idx.items():
            disease_names[idx] = dname

        # キャッシュチェック
        if os.path.exists(cache_file):
            data = np.load(cache_file, allow_pickle=True)
            cached_diseases = list(data["disease_names"])
            cached_hpe = list(data["hpe_names"])
            has_hyp = "hyp_embs" in data
            if cached_diseases == disease_names and cached_hpe == self.hpe_names and has_hyp:
                self.sim_matrix_hpe = data["sim_matrix"]
                self.hpe_hyp_embs = data["hyp_embs"]
                print(f"[HPE sim_matrix] キャッシュから読込 {self.sim_matrix_hpe.shape}")
                return

        # 仮説embedding
        hyp_embs = self._batch_embed(hypotheses)
        if hyp_embs is None:
            print("[HPE sim_matrix] embedding失敗")
            return

        h_norms = np.linalg.norm(hyp_embs, axis=1, keepdims=True)
        h_norms[h_norms == 0] = 1.0
        hyp_embs_normed = hyp_embs / h_norms

        self.sim_matrix_hpe = self.disease_embs_normed @ hyp_embs_normed.T
        self.hpe_hyp_embs = hyp_embs_normed  # novelty二重マッチ用

        np.savez(
            cache_file,
            sim_matrix=self.sim_matrix_hpe,
            hyp_embs=hyp_embs_normed,
            disease_names=np.array(disease_names, dtype=object),
            hpe_names=np.array(self.hpe_names, dtype=object),
        )
        print(f"[HPE sim_matrix] {self.sim_matrix_hpe.shape} 計算・キャッシュ完了")

    def _compute_hpe_name_embs(self):
        """HPE項目名をembedし、novelty計算に使用。"""
        cache_file = os.path.join(DATA_DIR, "hpe_name_embs.npz")

        if not self.hpe_names:
            return

        if os.path.exists(cache_file):
            data = np.load(cache_file, allow_pickle=True)
            cached_names = list(data["hpe_names"])
            if cached_names == self.hpe_names:
                self.hpe_name_embs = data["embeddings"]
                print(f"[HPE NameEmb] キャッシュから{self.hpe_name_embs.shape[0]}件読込")
                return

        embs = self._batch_embed(self.hpe_names)
        if embs is None:
            print("[HPE NameEmb] embedding失敗")
            return

        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.hpe_name_embs = embs / norms

        np.savez(
            cache_file,
            embeddings=self.hpe_name_embs,
            hpe_names=np.array(self.hpe_names, dtype=object),
        )
        print(f"[HPE NameEmb] {self.hpe_name_embs.shape[0]}件計算・キャッシュ完了")

    def compute_all_novelty(self, patient_text: str) -> tuple:
        """
        統合LLMコール: 検査novelty + HPE novelty + HPE所見抽出を1回で実行。
        3つの個別LLMコールを1つに統合し、APIコスト・レイテンシを削減。

        Returns: (novelty_tests: ndarray, novelty_hpe: ndarray, hpe_findings: list)
        """
        n_tests = len(self.test_names)
        n_hpe = len(self.hpe_names)

        # LLM統合プロンプト
        test_list_json = json.dumps(self.test_names, ensure_ascii=False)
        hpe_list_json = json.dumps(self.hpe_names, ensure_ascii=False)

        prompt = f"""以下の臨床テキストを読み、3つのタスクを同時に実行してください。

【タスク1: 実施済み検査の特定】
テキストに検査結果の数値・判定が明記されている検査だけを実施済みとする。
- 文字列の一致ではなく「医学的な論理包含」でマッピング
- 例: 「WBC 9800」→「CBC（白血球分画を含む）」が実施済み
- 例: 「CRP 2.5」→「CRP」が実施済み
- 出力は必ず検査マスタに存在する正確な名称のみ

【重要】タスク1の厳格ルール — 以下に該当するものは絶対に実施済みにしない:
- テキストに結果の数値・判定が書かれていない検査（推測で実施済みにしない）
- 患者の症状の訴え（「背中が痛い」「嘔吐した」→ 検査の実施を意味しない）
- 臨床状況からの推論（「救急搬送→心電図は当然」「発熱→採血は当然」→ 禁止）
- 「発熱39度」はバイタルサイン測定の根拠にはなるが、心電図・血液検査・画像検査の根拠にはならない
- 迷ったら「未実施」とする（偽陰性より偽陽性の害が大きい）

【タスク2: 聴取済みHPE項目の特定】
すでに聴取済み・確認済みの問診項目・身体診察所見を特定。
- 文字列の一致ではなく「医学的な論理包含」でマッピング
- 患者が自発的に述べた症状も「聴取済み」に含める
- 陰性所見（「〜なし」等）も聴取済み
- 同一トピックについて情報が得られていれば聴取済みとする
  例: 「機会飲酒」→「飲酒（常習）」は聴取済み（飲酒習慣について回答済み）
  例: 「タバコ吸わない」→「現在喫煙」は聴取済み（陰性所見）
  例: 「2日前から腹痛」→「腹痛（急性）」は聴取済み
- 出力は必ずHPEマスタに存在する正確な名称のみ

【タスク3: HPE所見の極性判定】
タスク2で特定した項目について、陽性(+1)か陰性(-1)かを判定。
- 「発熱」「腹痛あり」→ +1
- 「肝疾患なし」「Murphy徴候陰性」→ -1

【重要】タスク2・3の注意:
- 咳嗽・下痢・頭痛・関節痛・発熱は「急性」「慢性」に分かれています。
  テキストに期間の記載がある場合は、時間軸が一致する項目を選択してください。
  期間の記載がない場合は、どちらも選択しないでください。
  例: 「3ヶ月前からの咳」→「咳嗽（遷延・慢性：3週以上）」
  例: 「2日前からの下痢」→「下痢（急性：2週未満）」
  例: 「咳が出る」（期間不明）→ どちらも選択しない

【臨床テキスト】
{patient_text}

【検査マスタ（全{n_tests}件）】
{test_list_json}

【HPEマスタ（全{n_hpe}件）】
{hpe_list_json}

出力（JSON）:
{{"done_tests": ["検査名1", ...], "hpe_findings": [{{"item": "項目名", "polarity": 1}}, ...]}}
※ hpe_findingsには聴取済み項目を全て含め、各項目にpolarityを付与"""

        try:
            content = self._llm_call([
                {"role": "system", "content": "JSON出力のみ。説明不要。"},
                {"role": "user", "content": prompt},
            ])
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])

                # タスク1: 検査novelty
                novelty_tests = np.ones(n_tests)
                done_tests = parsed.get("done_tests", [])
                matched_tests = 0
                for dt in done_tests:
                    if dt in self.test_idx:
                        novelty_tests[self.test_idx[dt]] = 0.0
                        matched_tests += 1

                # タスク2+3: HPE novelty + 所見
                novelty_hpe = np.ones(n_hpe)
                hpe_findings = []
                for entry in parsed.get("hpe_findings", []):
                    name = entry.get("item", "")
                    pol = entry.get("polarity", 1)
                    if name in self.hpe_idx:
                        idx = self.hpe_idx[name]
                        novelty_hpe[idx] = 0.0
                        hpe_findings.append({
                            "item": name,
                            "index": idx,
                            "polarity": 1 if pol > 0 else -1,
                        })

                n_pos = sum(1 for f in hpe_findings if f["polarity"] > 0)
                n_neg = sum(1 for f in hpe_findings if f["polarity"] < 0)
                print(f"  [統合Novelty] 検査: {len(done_tests)}件抽出/{matched_tests}件マッチ, "
                      f"HPE: {len(hpe_findings)}項目(陽性{n_pos}, 陰性{n_neg})")
                return novelty_tests, novelty_hpe, hpe_findings
        except Exception as e:
            print(f"  [統合Novelty] 失敗、個別コールにフォールバック: {e}")

        # フォールバック: 個別コール
        novelty_tests = self.compute_novelty(patient_text)
        novelty_hpe = self.compute_novelty_hpe(patient_text)
        hpe_findings = self.extract_hpe_findings(patient_text)
        return novelty_tests, novelty_hpe, hpe_findings

    def extract_hpe_findings(self, patient_text: str) -> list:
        """
        LLMで患者テキストから問診/身体診察所見を極性付きで抽出。

        embeddingは否定を区別できないが、LLMは文脈を理解できる。
        用途:
          1. Part D novelty抑制（聴取済み項目の推薦を抑制）
          2. 疾患重み更新（陽性→関連疾患↑、陰性→関連疾患↓）

        返り値: [{"item": str, "index": int, "polarity": +1/-1}, ...]
        """
        if not self.hpe_names:
            return []

        item_list = ", ".join(self.hpe_names)

        prompt = f"""患者テキストで既に聴取・確認・言及済みの項目を項目リストから選び、極性を判定せよ。

極性ルール:
- 陽性所見（「発熱」「腹痛あり」「体温38.5℃」）→ +1
- 陰性所見（「肝疾患なし」「便秘なし」「Murphy徴候陰性」）→ -1

【重要】時間軸の注意:
- 咳嗽・下痢・頭痛・関節痛・発熱は「急性」「慢性」に分かれています。
  テキストに期間の記載がある場合は、時間軸が一致する項目を選択してください。
  期間の記載がない場合は、どちらも選択しないでください。
  例: 「3ヶ月前からの咳」→「咳嗽（遷延・慢性：3週以上）」
  例: 「2日前からの下痢」→「下痢（急性：2週未満）」

出力: [{{"item":"項目名","polarity":1}}, {{"item":"項目名","polarity":-1}}, ...] のJSON配列のみ。

項目リスト: {item_list}

患者テキスト: {patient_text}"""

        try:
            content = self._llm_call([
                {"role": "system", "content": "JSON配列のみ出力。説明不要。"},
                {"role": "user", "content": prompt},
            ])
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                if isinstance(parsed, list):
                    findings = []
                    for entry in parsed:
                        name = entry.get("item", "")
                        pol = entry.get("polarity", 1)
                        if name in self.hpe_idx:
                            findings.append({
                                "item": name,
                                "index": self.hpe_idx[name],
                                "polarity": 1 if pol > 0 else -1,
                            })
                    n_pos = sum(1 for f in findings if f["polarity"] > 0)
                    n_neg = sum(1 for f in findings if f["polarity"] < 0)
                    print(f"  [HPE抽出] {len(findings)}項目 (陽性{n_pos}, 陰性{n_neg})")
                    return findings
        except Exception as e:
            print(f"  [HPE抽出] LLM失敗: {e}")

        return []

    # Type R (リスク因子) サブカテゴリ: confirm行列を使用
    _HPE_TYPE_R_SUBCATS = {"既往歴", "薬剤歴", "嗜好/社会歴", "家族歴"}

    def update_from_hpe(self, candidates: list, hpe_findings: list) -> list:
        """
        問診/身体診察所見から疾患重みを更新（√N正規化方式）。

        全所見のdeltaを疾患ごとに合算し、√Nで正規化して1回だけ乗算。
        N回の指数乗算による発散を防止する。

        Type F（所見）: sim_matrix_hpe（screen行列）を使用
        Type R（リスク因子）: sim_matrix_hpe_confirm（confirm行列）を使用

        delta = Σ polarity_k × excess_k
        similarity *= exp(delta / √N)
        """
        if not hpe_findings or self.sim_matrix_hpe is None:
            return candidates

        # 1. 全所見のdeltaを疾患ごとに合算
        deltas = {}  # disease_name → float
        active_count = 0
        for f in hpe_findings:
            idx = f["index"]
            polarity = f["polarity"]

            # Type R → confirm行列, Type F → screen行列
            use_confirm = False
            if idx < len(self.hpe_items):
                subcat = self.hpe_items[idx].get("subcategory", "")
                if subcat in self._HPE_TYPE_R_SUBCATS:
                    use_confirm = True

            if use_confirm and self.sim_matrix_hpe_confirm is not None:
                sims = self.sim_matrix_hpe_confirm[:, idx]
                matrix_label = "confirm"
            else:
                sims = self.sim_matrix_hpe[:, idx]
                matrix_label = "screen"

            active_count += 1
            bg = float(sims.mean())

            for c in candidates:
                d_idx = self.disease_idx.get(c["disease_name"])
                if d_idx is not None:
                    excess = max(0.0, float(sims[d_idx]) - bg)
                    if excess > 0:
                        deltas[c["disease_name"]] = deltas.get(c["disease_name"], 0.0) + polarity * excess

            print(f"  [HPE更新] {f['item']} (pol={polarity:+d}) bg={bg:.4f} [{matrix_label}]")

        # 2. √Nで正規化して1回だけ乗算
        if active_count == 0:
            return candidates
        sqrt_n = math.sqrt(active_count)
        for c in candidates:
            delta = deltas.get(c["disease_name"], 0.0)
            if delta != 0:
                c["similarity"] *= float(np.exp(delta / sqrt_n))

        print(f"  [HPE更新] √N正規化: active={active_count}, √N={sqrt_n:.2f}")

        # 重み順にソート
        candidates.sort(key=lambda c: c["similarity"], reverse=True)
        return candidates

    def patch_hpe_from_negatives(self, negative_findings: list,
                                novelty_hpe: np.ndarray,
                                hpe_findings: list) -> tuple:
        """
        split_symptoms_resultsの陰性所見 → HPE novelty橋渡し。
        compute_all_noveltyが見逃した陰性所見をセーフティネットで補完。
        """
        import re
        if not negative_findings or not self.hpe_names:
            return novelty_hpe, hpe_findings

        already = {f["item"] for f in hpe_findings}
        patched = 0

        for neg in negative_findings:
            for idx, item in enumerate(self.hpe_items):
                name = item["item_name"]
                if name in already:
                    continue
                # "疝痛・間欠的激痛" → ["疝痛", "間欠的激痛"]
                # "発熱（急性：1週未満）" → ["発熱"]
                base = re.sub(r'[（(].+?[）)]', '', name)
                keywords = [k.strip() for k in base.split('・') if len(k.strip()) >= 2]
                for kw in keywords:
                    if kw in neg:
                        novelty_hpe[idx] = 0.0
                        hpe_findings.append({
                            "item": name, "index": idx, "polarity": -1,
                        })
                        already.add(name)
                        patched += 1
                        break

        if patched > 0:
            print(f"  [陰性→HPE橋渡し] {patched}件パッチ")
        return novelty_hpe, hpe_findings

    def compute_novelty_hpe(self, patient_text: str,
                            hpe_findings: list = None) -> np.ndarray:
        """
        Part D用novelty（v2: LLM二値判定）。
        LLM失敗時はembeddingギャップ検出にフォールバック。
        hpe_findings: LLM抽出結果があればそれで上書き。
        """
        n_hpe = len(self.hpe_names)
        if n_hpe == 0 or self.hpe_name_embs is None:
            return np.ones(n_hpe)

        novelty = self._compute_novelty_hpe_llm(patient_text)

        # LLM抽出結果で上書き（完全抑制）
        if hpe_findings:
            for f in hpe_findings:
                novelty[f["index"]] = 0.0

        return novelty

    def _compute_novelty_hpe_llm(self, patient_text: str) -> np.ndarray:
        """LLMに既聴取の問診・身体診察項目を抽出させる（全件提示方式）"""
        n_hpe = len(self.hpe_names)
        novelty = np.ones(n_hpe)

        # 全274件をLLMに提示
        hpe_list_json = json.dumps(self.hpe_names, ensure_ascii=False)

        prompt = f"""以下の臨床テキストを読み、すでに聴取済み・確認済みの問診項目・身体診察所見を特定してください。

【推論ルール】
1. 文字列の一致ではなく「医学的な論理包含」でマッピングすること。
   - 例: 患者が「2日前から腹痛」と述べている → 「腹痛」は聴取済み
   - 例: 「発熱38度」→ 「発熱」は聴取済み
   - 例: 「嘔気あり」→ 「嘔気・嘔吐」は聴取済み
   - 例: 「発疹なし」→ 「発疹・皮疹」は聴取済み（陰性所見も聴取済み）
   - 例: 「ショック所見あり」→ 「ショック徴候」は聴取済み
   - 例: 「機会飲酒」→ 「飲酒（常習）」は聴取済み（飲酒習慣について回答済み）
   - 例: 「タバコ吸わない」→ 「現在喫煙」は聴取済み（陰性所見）
2. 同一トピックについて情報が得られていれば聴取済みとする。
3. 患者が自発的に述べた症状も「聴取済み」に含める。
3. 出力は必ず下記マスタに存在する正確な名称のみ使用すること。マスタにない名称は出力しないこと。

【臨床テキスト】
{patient_text}

【VeSMed問診・身体診察マスタ（全{n_hpe}件）】
{hpe_list_json}

出力: JSON配列 ["項目名1", "項目名2", ...]"""

        try:
            content = self._llm_call([
                {"role": "system", "content": "JSON配列のみ出力。説明不要。"},
                {"role": "user", "content": prompt},
            ])
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                done_items = json.loads(content[start:end])
                matched = 0
                hpe_set = set(self.hpe_names)
                hpe_idx = {name: k for k, name in enumerate(self.hpe_names)}
                for di in done_items:
                    if di in hpe_idx:
                        novelty[hpe_idx[di]] = 0.0
                        matched += 1
                print(f"  [HPE Novelty LLM] {len(done_items)}件抽出, {matched}件マッチ → 二値化")
                return novelty
        except Exception as e:
            print(f"  [HPE Novelty LLM] 失敗、ギャップ法にフォールバック: {e}")

        return self._compute_novelty_hpe_gap(patient_text)

    def _compute_novelty_hpe_gap(self, patient_text: str) -> np.ndarray:
        """embeddingギャップ検出で二値HPE noveltyを返す"""
        n_hpe = len(self.hpe_names)
        novelty = np.ones(n_hpe)

        lines = [l.strip() for l in patient_text.split('\n') if l.strip()]
        if not lines:
            return novelty

        try:
            resp = self.embed_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=lines,
            )
            line_embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            l_norms = np.linalg.norm(line_embs, axis=1, keepdims=True)
            l_norms[l_norms == 0] = 1.0
            line_embs = line_embs / l_norms
        except Exception:
            return novelty

        sims = line_embs @ self.hpe_name_embs.T
        max_cos = sims.max(axis=0)

        # ギャップ検出
        sorted_cos = np.sort(max_cos)[::-1]
        gaps = sorted_cos[:-1] - sorted_cos[1:]

        if len(gaps) > 0:
            search_range = max(5, len(gaps) // 5)
            best_gap_idx = int(np.argmax(gaps[:search_range]))
            threshold = (sorted_cos[best_gap_idx] + sorted_cos[best_gap_idx + 1]) / 2

            if threshold > 0.55:
                novelty[max_cos >= threshold] = 0.0
                n_done = int((max_cos >= threshold).sum())
                print(f"  [HPE Novelty Gap] 閾値={threshold:.3f}, {n_done}件抑制")
            else:
                print(f"  [HPE Novelty Gap] ギャップ不明瞭(th={threshold:.3f}), 抑制なし")

        return novelty

    def rank_hpe(self, candidates: list, novelty_hpe: np.ndarray = None) -> list:
        """
        Part D: 問診・身体診察推奨ランキング。
        Part Aと同一数学（prior加重分散）。quality/risk_relevance不要（非侵襲）。

        utility = variance × novelty
        """
        if self.sim_matrix_hpe is None or len(self.hpe_names) == 0:
            return []

        n_hpe = len(self.hpe_names)
        if novelty_hpe is None:
            novelty_hpe = np.ones(n_hpe)

        # 重み: 局所重み × 2C重み（Part Aと完全同一）
        raw_sims = np.array([c.get("similarity", 0.0) for c in candidates], dtype=float)
        weights = np.array([
            self.disease_2c.get(c["disease_name"], {}).get("weight", 1.0)
            for c in candidates
        ], dtype=float)
        sim_centered = np.maximum(0.0, raw_sims - raw_sims.mean())
        w = sim_centered * weights
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum

        # sim_matrix_hpe行を取得
        disease_rows = []
        for c in candidates:
            row = self.disease_idx.get(c["disease_name"])
            disease_rows.append(row if row is not None else -1)
        disease_rows = np.array(disease_rows)

        valid_mask = disease_rows >= 0
        sim_sub = np.zeros((len(candidates), n_hpe))
        sim_sub[valid_mask] = self.sim_matrix_hpe[disease_rows[valid_mask]]

        # prior加重分散
        w_col = w[:, np.newaxis]
        mu = (w_col * sim_sub).sum(axis=0)
        var = (w_col * (sim_sub - mu) ** 2).sum(axis=0)

        # 関連疾患
        weighted_contrib = w_col * sim_sub

        ranked = []
        for k, item in enumerate(self.hpe_items):
            score = float(var[k])
            nov = float(novelty_hpe[k])
            utility = score * nov

            # 関連疾患Top5
            contribs = weighted_contrib[:, k]
            top_idx = np.argsort(contribs)[::-1][:5]
            details = [
                {"disease_name": candidates[int(i)]["disease_name"],
                 "contribution": round(float(contribs[i]), 6)}
                for i in top_idx if contribs[i] > 0
            ]

            ranked.append({
                "item_name": item["item_name"],
                "category": item["category"],
                "subcategory": item["subcategory"],
                "instruction": item.get("instruction", ""),
                "score": round(score, 6),
                "novelty": round(nov, 4),
                "utility": round(utility, 6),
                "details": details,
            })

        ranked.sort(key=lambda x: x["utility"], reverse=True)
        return ranked
