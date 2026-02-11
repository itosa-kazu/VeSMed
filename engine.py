"""
VeSMed - 核心エンジン
クエリ整理 → ベクトル検索 → 情報利得計算 → 検査推薦
"""

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import chromadb
from openai import OpenAI
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    LLM_FALLBACK_API_KEY, LLM_FALLBACK_BASE_URL, LLM_FALLBACK_MODEL,
    LLM_MAX_RETRIES,
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
    DISEASES_JSONL, TESTS_JSONL, FINDINGS_JSONL, CHROMA_DIR, DATA_DIR,
)


class VeSMedEngine:
    def __init__(self):
        self.llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=30)
        self.llm_fallback_client = OpenAI(api_key=LLM_FALLBACK_API_KEY, base_url=LLM_FALLBACK_BASE_URL, timeout=30)
        self.embed_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)

        # ChromaDB接続
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.chroma_client.get_collection("diseases")

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
        self.sim_matrix = None    # (N_diseases, N_tests) ndarray
        self.disease_idx = {}     # disease_name → row index
        self.test_idx = {}        # test_name → col index
        self.test_names = []      # col順の検査名リスト
        self._compute_similarity_matrix()

        # 検査の質（2Q: Value + Feasibility）
        self.test_quality = {}  # test_name → {"value": float, "feasibility": float}
        self._compute_test_quality()

        # 検査リスクembedding（risk_description → embedding）
        self.risk_embs = {}  # test_name → np.array (4096,)
        self._compute_risk_embeddings()

        # 検査findings embedding（novelty計算用）: 正規化済み (N_tests, dim)
        self.test_findings_embs = None
        self._load_test_findings_embs()

        # 最後のクエリembedding（rank_testsでrisk_relevance計算に使用）
        self._last_query_embedding = None

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

        # ChromaDBから全疾患のembeddingを取得
        all_ids = [f"disease_{i}" for i in range(self.collection.count())]
        batch_size = 100
        all_embeddings = {}
        for start in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[start:start + batch_size]
            result = self.collection.get(ids=batch_ids, include=["embeddings", "metadatas"])
            for j, mid in enumerate(result["ids"]):
                dname = result["metadatas"][j].get("disease_name", "")
                all_embeddings[dname] = np.array(result["embeddings"][j])

        # 余弦類似度 → exp重みを計算
        def cosine_sim(a, b):
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))

        for dname, emb in all_embeddings.items():
            scores = {}
            for anchor_name, anchor_emb in anchor_embs.items():
                scores[anchor_name] = cosine_sim(emb, anchor_emb)
            weight = math.exp(scores["critical"] + scores["curable"])
            self.disease_2c[dname] = {**scores, "weight": weight}

        print(f"[2C] {len(self.disease_2c)}疾患の2Cスコア計算完了")

    # ----------------------------------------------------------------
    # 類似度行列計算（疾患×検査、起動時に1回）
    # ----------------------------------------------------------------
    def _compute_similarity_matrix(self):
        """
        疾患findings_description × 検査findings_description のcos類似度行列を計算。
        疾患embeddingはChromaDBから、検査embeddingはAPI+キャッシュから取得。
        結果: self.sim_matrix (N_diseases, N_tests), self.disease_idx, self.test_idx
        """
        cache_file = os.path.join(DATA_DIR, "sim_matrix.npz")

        # 検査リスト構築（findings_descriptionがあるもののみ）
        self.test_names = [
            tname for tname, tdata in self.test_db.items()
            if tdata.get("findings_description")
        ]
        self.test_idx = {name: i for i, name in enumerate(self.test_names)}

        # 疾患リスト構築（ChromaDBの順序）
        n_diseases = self.collection.count()
        all_ids = [f"disease_{i}" for i in range(n_diseases)]
        disease_names = []
        disease_embs_list = []
        batch_size = 100
        for start in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[start:start + batch_size]
            result = self.collection.get(ids=batch_ids, include=["embeddings", "metadatas"])
            for j, mid in enumerate(result["ids"]):
                dname = result["metadatas"][j].get("disease_name", "")
                disease_names.append(dname)
                disease_embs_list.append(result["embeddings"][j])
        self.disease_idx = {name: i for i, name in enumerate(disease_names)}
        disease_embs = np.array(disease_embs_list, dtype=np.float32)

        # キャッシュチェック
        if os.path.exists(cache_file):
            data = np.load(cache_file, allow_pickle=True)
            cached_diseases = list(data["disease_names"])
            cached_tests = list(data["test_names"])
            if cached_diseases == disease_names and cached_tests == self.test_names:
                self.sim_matrix = data["sim_matrix"]
                # test_findings_embsもキャッシュから読み込み（存在すれば）
                if "test_findings_embs" in data:
                    self._cached_test_findings_embs = data["test_findings_embs"]
                print(f"[類似度行列] キャッシュから読込 {self.sim_matrix.shape}")
                return

        # 検査embeddingを取得（バッチ、並行）
        test_texts = [self.test_db[t]["findings_description"] for t in self.test_names]
        test_embs = self._batch_embed(test_texts)
        if test_embs is None:
            print("[類似度行列] 検査embedding失敗")
            return

        # 正規化してcos類似度 = 内積
        d_norms = np.linalg.norm(disease_embs, axis=1, keepdims=True)
        d_norms[d_norms == 0] = 1.0
        disease_embs_normed = disease_embs / d_norms

        t_norms = np.linalg.norm(test_embs, axis=1, keepdims=True)
        t_norms[t_norms == 0] = 1.0
        test_embs_normed = test_embs / t_norms

        self.sim_matrix = disease_embs_normed @ test_embs_normed.T
        self._cached_test_findings_embs = test_embs_normed  # novelty計算用に保持

        # キャッシュ保存（test_findings_embsも含む）
        np.savez(
            cache_file,
            sim_matrix=self.sim_matrix,
            test_findings_embs=test_embs_normed,
            disease_names=np.array(disease_names, dtype=object),
            test_names=np.array(self.test_names, dtype=object),
        )
        print(f"[類似度行列] {self.sim_matrix.shape} 計算・キャッシュ完了")

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

        max_workers = 10
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
    # 検査findings embedding読込（novelty計算用）
    # ----------------------------------------------------------------
    def _load_test_findings_embs(self):
        """
        sim_matrix構築時に計算されたtest findings embeddingを
        novelty計算用に (N_tests, dim) ndarray として保持する。
        """
        if hasattr(self, '_cached_test_findings_embs') and self._cached_test_findings_embs is not None:
            self.test_findings_embs = self._cached_test_findings_embs
            print(f"[TestFindingsEmb] {self.test_findings_embs.shape[0]}件読込")
        else:
            print("[TestFindingsEmb] なし（sim_matrixキャッシュ更新が必要）")

    # ----------------------------------------------------------------
    # LLM呼び出し（リトライ + フォールバック）
    # ----------------------------------------------------------------
    def _llm_call(self, messages, temperature=0.1, max_tokens=65536):
        """プライマリAPIでリトライし、失敗したらフォールバックAPIに切り替え"""
        # プライマリAPI
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
                print(f"[LLM] プライマリAPI失敗 (試行{attempt+1}/{LLM_MAX_RETRIES+1}): {e}")
                if attempt < LLM_MAX_RETRIES:
                    time.sleep(2 ** attempt)

        # フォールバックAPI
        print("[LLM] フォールバックAPIに切り替え")
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
    # Step 1: Novelty計算（患者テキストに既に含まれる情報を連続的に割引）
    # ----------------------------------------------------------------
    def compute_novelty(self, patient_text: str) -> np.ndarray:
        """
        患者テキストを行単位で分割・embeddingし、各検査のfindings_description
        embeddingとのmax cos類似度から新規性スコアを計算する。

        novelty_j = 1 - max_line cos(line_emb, test_findings_emb_j)

        返り値: (N_tests,) ndarray, 各検査の新規性スコア (0〜1)
        閾値なし、二値判定なし。全てembeddingから連続的に導出。
        """
        n_tests = len(self.test_names)
        if self.test_findings_embs is None or n_tests == 0:
            return np.ones(n_tests)

        # 患者テキストを行単位で分割（空行除外）
        lines = [l.strip() for l in patient_text.split('\n') if l.strip()]
        if not lines:
            return np.ones(n_tests)

        # 全行を一括embed
        resp = self.embed_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=lines,
        )
        line_embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        l_norms = np.linalg.norm(line_embs, axis=1, keepdims=True)
        l_norms[l_norms == 0] = 1.0
        line_embs = line_embs / l_norms

        # (N_lines, N_tests) = line_embs @ test_findings_embs.T
        sims = line_embs @ self.test_findings_embs.T

        # 各検査について全行のmax → 最も関連の深い行との類似度
        max_sims = sims.max(axis=0)  # (N_tests,)

        # novelty = 1 - max_sim（既に情報があるほどutilityを下げる）
        novelty = 1.0 - np.clip(max_sims, 0.0, 1.0)

        return novelty

    # ----------------------------------------------------------------
    # Step 2: ベクトル検索 → 候補疾患
    # ----------------------------------------------------------------
    def _embed_and_search(self, text: str) -> list:
        """単一テキストでChromaDB全件検索。内部用。"""
        resp = self.embed_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text],
        )
        query_embedding = resp.data[0].embedding
        # 最後のクエリembeddingを保持（risk_relevance計算用）
        self._last_query_embedding = np.array(query_embedding, dtype=np.float32)

        n_total = self.collection.count()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_total,
        )

        candidates = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity = 1.0 - distance
                metadata = results["metadatas"][0][i]
                candidates.append({
                    "disease_name": metadata.get("disease_name", ""),
                    "similarity": similarity,
                    "category": metadata.get("category", ""),
                    "urgency": metadata.get("urgency", ""),
                })
        return candidates

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
    # Step 4: 検査ランキング（prior加重分散）
    # ----------------------------------------------------------------
    def rank_tests(self, candidates: list, novelty: np.ndarray = None) -> list:
        """
        各検査について、疑い疾患群に対するcos類似度のprior加重分散を計算。
        分散が大きい = 疾患間でバラつく = 鑑別に有用な検査。

        utility = variance × exp(quality) × novelty × (1 - risk_relevance)

        novelty: compute_novelty()の返り値。患者テキストに既に含まれる情報を
        連続的に割引する。閾値なし、二値判定なし。
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

        # 関連疾患: 各検査で重みが高い上位疾患を抽出
        # weighted_sim_sub = w * sim_sub (各疾患の検査への寄与度)
        weighted_contrib = w_col * sim_sub  # (N_candidates, N_tests)

        ranked = []
        for j, tname in enumerate(self.test_names):
            score = float(var[j])
            q = self.test_quality.get(tname, {"axis": 0.0})
            axis_proj = q["axis"]
            nov = float(novelty[j])
            utility = score * math.exp(axis_proj) * nov

            # risk_relevance: 侵襲的検査のリスクが患者状態と関連する場合にutilityを低下
            risk_rel = 0.0
            if patient_emb is not None and tname in self.risk_embs:
                risk_rel = max(0.0, float(np.dot(self.risk_embs[tname], patient_emb)))
                utility *= (1.0 - risk_rel)

            # 関連疾患Top5（重み付き寄与度順）
            contribs = weighted_contrib[:, j]
            top_idx = np.argsort(contribs)[::-1][:5]
            details = [
                {"disease_name": candidates[int(k)]["disease_name"],
                 "contribution": round(float(contribs[k]), 6)}
                for k in top_idx if contribs[k] > 0
            ]

            tdata = self.test_db.get(tname, {})
            ranked.append({
                "test_name": tname,
                "score": round(score, 6),
                "quality": round(axis_proj, 4),
                "novelty": round(nov, 4),
                "risk_relevance": round(risk_rel, 4),
                "turnaround_minutes": tdata.get("turnaround_minutes", 60),
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
        if self.sim_matrix is None or len(self.test_names) == 0:
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

        # sim_matrix行を取得
        disease_rows = []
        for c in candidates:
            row = self.disease_idx.get(c["disease_name"])
            disease_rows.append(row if row is not None else -1)
        disease_rows = np.array(disease_rows)

        valid_mask = disease_rows >= 0
        sim_sub = np.zeros((len(candidates), n_tests))
        sim_sub[valid_mask] = self.sim_matrix[disease_rows[valid_mask]]

        # クラスタ加重平均 vs 全疾患背景平均
        w_col = w[:, np.newaxis]
        cluster_mu = (w_col * sim_sub).sum(axis=0)  # (N_tests,)
        global_mu = self.sim_matrix.mean(axis=0)      # (N_tests,)
        confirm = cluster_mu - global_mu               # クラスタ特異度

        # risk_relevance計算用
        patient_emb = None
        if self._last_query_embedding is not None and len(self.risk_embs) > 0:
            pe = self._last_query_embedding
            pe_norm = np.linalg.norm(pe)
            if pe_norm > 0:
                patient_emb = pe / pe_norm

        # 関連疾患
        weighted_contrib = w_col * sim_sub

        ranked = []
        for j, tname in enumerate(self.test_names):
            score = float(confirm[j])
            q = self.test_quality.get(tname, {"axis": 0.0})
            axis_proj = q["axis"]
            nov = float(novelty[j])
            utility = score * math.exp(axis_proj) * nov

            # risk_relevance
            risk_rel = 0.0
            if patient_emb is not None and tname in self.risk_embs:
                risk_rel = max(0.0, float(np.dot(self.risk_embs[tname], patient_emb)))
                utility *= (1.0 - risk_rel)

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
                "quality": round(axis_proj, 4),
                "novelty": round(nov, 4),
                "risk_relevance": round(risk_rel, 4),
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

        if self.sim_matrix is None or len(self.test_names) == 0:
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

        # sim_matrix行を取得
        disease_rows = []
        for c in candidates:
            row = self.disease_idx.get(c["disease_name"])
            disease_rows.append(row if row is not None else -1)
        disease_rows = np.array(disease_rows)

        valid_mask = disease_rows >= 0
        sim_sub = np.zeros((len(candidates), n_tests))
        sim_sub[valid_mask] = self.sim_matrix[disease_rows[valid_mask]]

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

        ranked = []
        for j, tname in enumerate(self.test_names):
            ch = float(critical_scores[j])
            q = self.test_quality.get(tname, {"axis": 0.0})
            axis_proj = q["axis"]
            nov = float(novelty[j])
            utility = ch * math.exp(axis_proj) * nov

            # risk_relevance
            risk_rel = 0.0
            if patient_emb is not None and tname in self.risk_embs:
                risk_rel = max(0.0, float(np.dot(self.risk_embs[tname], patient_emb)))
                utility *= (1.0 - risk_rel)

            # 最大命中疾患
            bi = int(best_disease_idx[j])
            hit_disease = candidates[bi]["disease_name"] if bi < len(candidates) else ""

            ranked.append({
                "test_name": tname,
                "critical_hit": round(ch, 6),
                "quality": round(axis_proj, 4),
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
