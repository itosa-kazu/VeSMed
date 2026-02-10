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
    TAU,
    DISEASES_JSONL, FINDINGS_JSONL, CHROMA_DIR, DATA_DIR,
)


class VeSMedEngine:
    def __init__(self):
        self.llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self.llm_fallback_client = OpenAI(api_key=LLM_FALLBACK_API_KEY, base_url=LLM_FALLBACK_BASE_URL)
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

        # 検査コスト（embedding推定）
        self.test_embed_cost = {}  # test_name → float (exp(cos_invasive))
        self._compute_test_embed_costs()

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
            "critical": "見逃すと数時間以内に死亡する、または不可逆的な臓器障害を来す致命的疾患。緊急手術や集中治療を要する。",
            "curable": "早期に適切な治療（抗菌薬、手術、特異的治療）を行えば完治または大幅な改善が期待できる治療可能な疾患。",
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
    # 検査コスト推定（embedding、キャッシュ付き）
    # ----------------------------------------------------------------
    def _compute_test_embed_costs(self):
        """
        検査の手技記述（またはフォールバックとして検査名）をembedし
        コストアンカーとの余弦類似度から侵襲度/コストを推定。
        cost = exp(cos_sim(test_description, anchor))
        結果はJSONファイルにキャッシュし、次回起動時はロードのみ。
        """
        cache_file = os.path.join(DATA_DIR, "test_embed_cost.json")

        # 全検査名を収集（名寄せ後）
        all_test_names = set()
        for d in self.disease_db.values():
            for t in d.get("relevant_tests", []):
                tname = self.test_name_map.get(t["test_name"], t["test_name"])
                all_test_names.add(tname)

        # キャッシュが存在し、全検査名をカバーしていればロード
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            missing = all_test_names - set(cached.keys())
            if not missing:
                self.test_embed_cost = {k: float(v) for k, v in cached.items()}
                print(f"[コスト] キャッシュから{len(self.test_embed_cost)}検査のコスト読込")
                return
            print(f"[コスト] キャッシュに{len(missing)}件の未計算検査あり、再計算")
        else:
            cached = {}

        # tests.jsonlから手技記述を読み込み（検査名→記述のマップ）
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
                        desc = td.get("description_for_embedding", "")
                        if desc:
                            test_descriptions[td["test_name"]] = desc
                    except (json.JSONDecodeError, KeyError):
                        continue
            print(f"[コスト] {len(test_descriptions)}件の検査手技記述を読込")

        # アンカーテキスト
        anchor_text = "大規模な設備・専門スタッフ・入院を要し、患者への身体的負担と合併症リスクが大きい高額で侵襲的な検査手技"

        # embedするテキスト: 記述があれば記述、なければ検査名
        test_list = sorted(all_test_names)
        embed_texts = []
        for tname in test_list:
            desc = test_descriptions.get(tname, "")
            embed_texts.append(desc if desc else tname)
        all_texts = [anchor_text] + embed_texts

        # バッチに分割して並行embedding（ChromaDB不使用、キャッシュJSONのみ書込）
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
                    print(f"[コスト] batch (offset={start_idx}) 失敗 (試行{attempt+1}): {e}")
                    time.sleep(2 ** attempt)
            return start_idx, None

        max_workers = 10
        print(f"[コスト] {total_batches}バッチを{max_workers}並行でembedding開始")
        failed = False
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_embed_batch, b) for b in batches]
            completed = 0
            for future in as_completed(futures):
                start_idx, result = future.result()
                if result is None:
                    print(f"[コスト] batch (offset={start_idx}) が3回失敗、中断")
                    failed = True
                    break
                for j, emb in enumerate(result):
                    all_embs[start_idx + j] = emb
                completed += 1
                if completed % 5 == 0 or completed == total_batches:
                    print(f"[コスト] embedding {completed}/{total_batches} バッチ完了")

        if failed:
            return

        anchor_emb = all_embs[0]
        test_embs = all_embs[1:]

        def cosine_sim(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        for i, tname in enumerate(test_list):
            sim = cosine_sim(test_embs[i], anchor_emb)
            self.test_embed_cost[tname] = math.exp(sim)

        # キャッシュに保存
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(self.test_embed_cost, f, ensure_ascii=False, indent=1)

        print(f"[コスト] {len(self.test_embed_cost)}検査のembeddingコスト計算完了")

    # ----------------------------------------------------------------
    # LLM呼び出し（リトライ + フォールバック）
    # ----------------------------------------------------------------
    def _llm_call(self, messages, temperature=0.1, max_tokens=65536):
        """プライマリAPIでリトライし、失敗したらフォールバックAPIに切り替え"""
        # プライマリAPI
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                resp = self.llm_client.chat.completions.create(
                    model=LLM_MODEL,
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
    # Step 1: クエリ整理（患者情報 → 標準化医学テキスト）
    # ----------------------------------------------------------------
    def rewrite_query(self, patient_text: str) -> str:
        """患者情報をLLMで標準的な医学用語に整理する"""
        return self._llm_call(
            messages=[
                {"role": "system", "content": (
                    "あなたは経験豊富な日本の臨床医です。\n"
                    "患者情報を受け取り、標準的な医学用語で簡潔に整理してください。\n"
                    "ルール：\n"
                    "- 入力情報にない情報は絶対に追加しないこと\n"
                    "- 標準的な医学用語への変換と構造化のみ行うこと\n"
                    "- 年齢、性別、主訴、現病歴、既往歴、身体所見、バイタルサインの順で整理\n"
                    "- 200-400字程度で出力\n"
                    "- 説明や前置きは不要、整理されたテキストのみ出力"
                )},
                {"role": "user", "content": patient_text},
            ],
        )

    # ----------------------------------------------------------------
    # Step 1.5: 患者テキストから既実施検査・所見を抽出
    # ----------------------------------------------------------------
    def extract_done_tests(self, patient_text: str) -> list:
        """
        患者テキストから既に実施済みの検査とその所見を抽出する。
        返り値: [{"test_name": str, "finding": str}, ...]
        """
        content = self._llm_call(
            messages=[
                {"role": "system", "content": (
                    "あなたは経験豊富な日本の臨床医です。\n"
                    "患者情報テキストから、既に実施済みの検査・画像・手技とその結果を抽出してください。\n"
                    "ルール：\n"
                    "- テキストに明示的に記載されている検査のみ抽出すること\n"
                    "- 推測で検査を追加しないこと\n"
                    "- 身体診察の所見（聴診、触診など）も含めること\n"
                    "- バイタルサイン（体温、血圧、心拍数、呼吸数、SpO2）は1つにまとめること\n"
                    "- 出力はJSONのみ。説明文やマークダウンは不要。最初の文字は [ にすること。\n"
                    "- 簡潔に。各項目のfindingは20字以内。"
                )},
                {"role": "user", "content": (
                    f"以下の患者情報から既実施の検査と所見を抽出してください。\n\n"
                    f"{patient_text}\n\n"
                    f'出力形式: [{{"test_name": "検査名", "finding": "所見"}}, ...]'
                )},
            ],
        )

        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
        if match:
            content = match.group(1).strip()
        start = content.find("[")
        if start > 0:
            content = content[start:]
        content = re.sub(r",\s*([}\]])", r"\1", content)

        # 途中で切れたJSONの修復: 未閉じの括弧を補完
        open_braces = content.count("{") - content.count("}")
        open_brackets = content.count("[") - content.count("]")
        if open_braces > 0 or open_brackets > 0:
            # 最後の不完全なオブジェクトを除去
            last_complete = content.rfind("}")
            if last_complete > 0:
                content = content[:last_complete + 1]
            content += "]" * max(0, open_brackets)

        try:
            results = json.loads(content)
            if isinstance(results, list):
                return results
        except json.JSONDecodeError:
            pass
        return []

    # ----------------------------------------------------------------
    # Step 1.6: 既実施検査の名寄せ（LLMで照合）
    # ----------------------------------------------------------------
    def normalize_done_tests(self, done_tests: list, candidates: list) -> list:
        """
        extract_done_testsの出力を、候補疾患のrelevant_testsの検査名に名寄せする。
        done_tests: [{"test_name": "血液検査(WBC)", "finding": "16000"}, ...]
        candidates: compute_priors済みの候補疾患リスト
        返り値: 名寄せ済みの検査名リスト ["白血球数", "CRP", ...]
        """
        if not done_tests or not candidates:
            return []

        # 候補疾患のrelevant_testsから検査名を収集
        db_test_names = set()
        for c in candidates:
            d = self.disease_db.get(c["disease_name"])
            if not d:
                continue
            for t in d.get("relevant_tests", []):
                db_test_names.add(t["test_name"])

        if not db_test_names:
            return [d["test_name"] for d in done_tests]

        done_str = "\n".join(f"- {d['test_name']}: {d.get('finding', '')}" for d in done_tests)
        db_str = "\n".join(f"- {n}" for n in sorted(db_test_names))

        content = self._llm_call(
            messages=[
                {"role": "system", "content": (
                    "あなたは臨床検査の専門家です。\n"
                    "【タスク】患者から抽出された既実施検査を、データベースの検査名に照合してください。\n"
                    "ルール：\n"
                    "- 既実施検査の各項目について、データベース検査名リストの中で同一または包含関係にあるものを全て列挙\n"
                    "- 例: 「血液検査(WBC)」→「白血球数」「白血球分画」\n"
                    "- 例: 「身体診察(頭頸部)」→「咽頭所見」「項部硬直」「結膜所見」\n"
                    "- 例: 「バイタルサイン」→「発熱（37.5℃以上）」「血圧測定」\n"
                    "- 該当なしの場合はスキップ\n"
                    "- 出力はJSON配列のみ。説明不要。最初の文字は [ にすること。"
                )},
                {"role": "user", "content": (
                    f"【既実施検査】\n{done_str}\n\n"
                    f"【データベース検査名リスト】\n{db_str}\n\n"
                    f'出力形式: ["検査名1", "検査名2", ...]'
                )},
            ],
        )

        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
        if match:
            content = match.group(1).strip()
        start = content.find("[")
        if start > 0:
            content = content[start:]
        content = re.sub(r",\s*\]", "]", content)

        # 途中で切れたJSONの修復
        if content.count("[") > content.count("]"):
            last_quote = content.rfind('"')
            if last_quote > 0:
                content = content[:last_quote + 1] + "]"

        try:
            results = json.loads(content)
            if isinstance(results, list):
                # DB検査名に存在するもののみ返す
                return [r for r in results if r in db_test_names]
        except json.JSONDecodeError:
            pass
        return [d["test_name"] for d in done_tests]

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

    def search_diseases(self, rewritten_text: str, original_text: str = None) -> list:
        """
        全疾患検索: 書き換え文と原文の両方で検索し、結果をマージ。
        各疾患は高い方の類似度を採用。top-k制限なし（全疾患を返す）。

        返り値: [{"disease_name": str, "similarity": float, "category": str}, ...]
        """
        # 書き換え文で全件検索
        candidates_rewritten = self._embed_and_search(rewritten_text)

        # 原文でも検索してマージ
        if original_text and original_text.strip() != rewritten_text.strip():
            candidates_original = self._embed_and_search(original_text)
        else:
            candidates_original = []

        # マージ: 疾患名をキーに、高い方の類似度を採用
        merged = {}
        for c in candidates_rewritten + candidates_original:
            name = c["disease_name"]
            if name not in merged or c["similarity"] > merged[name]["similarity"]:
                merged[name] = c

        # 類似度降順でソート（全疾患を返す）
        result = sorted(merged.values(), key=lambda x: x["similarity"], reverse=True)
        return result

    # ----------------------------------------------------------------
    # Step 3: Softmax → 先験確率
    # ----------------------------------------------------------------
    def compute_priors(self, candidates: list, tau: float = None) -> list:
        """
        候補疾患の類似度スコアをsoftmaxで先験確率に変換。
        candidatesに"prior"フィールドを追加して返す。

        生のcosine類似度にsoftmax(τ=1)を直接適用。
        正規化なし = embeddingの距離をそのまま信じる。
        """
        if tau is None:
            tau = TAU

        if not candidates:
            return candidates

        sims = np.array([c["similarity"] for c in candidates])

        # softmax with temperature（正規化なし、生のcosine類似度を直接使用）
        exp_vals = np.exp(sims / tau)
        priors = exp_vals / exp_vals.sum()

        for i, c in enumerate(candidates):
            c["prior"] = float(priors[i])
            c["clinical_weight"] = self.disease_2c.get(c["disease_name"], {}).get("weight", 1.0)

        return candidates

    # ----------------------------------------------------------------
    # Step 4: 情報利得計算 → 検査ランキング
    # ----------------------------------------------------------------
    def rank_tests(self, candidates: list, done_tests: list = None) -> list:
        """
        候補疾患のrelevant_testsを集約し、
        各検査の期待情報利得を計算してランキングする。

        返り値: [{"test_name": str, "info_gain": float, "cost_level": int,
                  "utility": float, "details": list}, ...]
        """
        if done_tests is None:
            done_tests = []
        # done_testsも名寄せして比較
        done_set = set(self.test_name_map.get(t, t) for t in done_tests)

        # 全候補疾患のrelevant_testsを集約
        # test_name → [{"disease_name", "prior", "se_low", "sp_low", ...}]
        test_disease_map = {}

        for c in candidates:
            d = self.disease_db.get(c["disease_name"])
            if not d:
                continue
            prior = c.get("prior", 0.0)

            for t in d.get("relevant_tests", []):
                # 名寄せ: 生の検査名 → 正規名
                tname = self.test_name_map.get(t["test_name"], t["test_name"])
                if tname in done_set:
                    continue

                if tname not in test_disease_map:
                    test_disease_map[tname] = {
                        "test_name": tname,
                        "cost_level": t.get("cost_level", 3),
                        "invasiveness": t.get("invasiveness", 3),
                        "turnaround_minutes": t.get("turnaround_minutes", 60),
                        "diseases": [],
                    }

                se = t.get("sensitivity", [0.5, 0.5])
                sp = t.get("specificity", [0.5, 0.5])
                test_disease_map[tname]["diseases"].append({
                    "disease_name": c["disease_name"],
                    "prior": prior,
                    "se_low": se[0] if isinstance(se, list) else se,
                    "sp_low": sp[0] if isinstance(sp, list) else sp,
                    "se_high": se[1] if isinstance(se, list) and len(se) > 1 else (se[0] if isinstance(se, list) else se),
                    "sp_high": sp[1] if isinstance(sp, list) and len(sp) > 1 else (sp[0] if isinstance(sp, list) else sp),
                    "purpose": t.get("purpose", ""),
                    "condition_notes": t.get("condition_notes", ""),
                })

        # 各検査の重み付き情報利得を計算
        # 重み = exp(cos_critical + cos_curable): 臨床重要度（見逃しコスト + 治療利益）
        # → 緊急かつ治療可能な疾患の不確実性を減らす検査が自然に高スコアになる
        clinical_weights = np.array([
            self.disease_2c.get(c["disease_name"], {}).get("weight", 1.0)
            for c in candidates
        ], dtype=float)

        ranked = []
        for tname, tdata in test_disease_map.items():
            score = self._expected_info_gain(candidates, tdata, clinical_weights)

            # コスト = exp(cos_sim(検査名, 侵襲アンカー))
            embed_cost = self.test_embed_cost.get(tname, math.exp(0.5))
            utility = score / embed_cost

            ranked.append({
                "test_name": tname,
                "info_gain": round(score, 4),
                "embed_cost": round(embed_cost, 2),
                "turnaround_minutes": tdata["turnaround_minutes"],
                "utility": round(utility, 4),
                "details": tdata["diseases"],
            })

        ranked.sort(key=lambda x: x["utility"], reverse=True)
        return ranked

    def _expected_info_gain(self, candidates, test_data, clinical_weights) -> float:
        """
        1つの検査の重み付き期待情報利得を計算する。

        重み付きエントロピー H_w(D) = -Σ w_i × p_i × log(p_i) を使用。
        w_i = exp(cos_critical + cos_curable): 2Cスコアに基づく臨床重要度。
        緊急かつ治療可能な疾患の不確実性を減らす検査ほど高スコアになる。
        """
        disease_test_info = {}
        for d in test_data["diseases"]:
            disease_test_info[d["disease_name"]] = {
                "se": d["se_low"],
                "sp": d["sp_low"],
            }

        priors = []
        se_list = []
        sp_list = []

        for c in candidates:
            p = c.get("prior", 0.0)
            priors.append(p)
            if c["disease_name"] in disease_test_info:
                se_list.append(disease_test_info[c["disease_name"]]["se"])
                sp_list.append(disease_test_info[c["disease_name"]]["sp"])
            else:
                se_list.append(0.5)
                sp_list.append(0.5)

        priors = np.array(priors)
        se_arr = np.array(se_list)
        sp_arr = np.array(sp_list)

        # 現在の重み付きエントロピー
        h_prior = self._weighted_entropy(priors, clinical_weights)

        # P(T+) = Σ_i π_i * Se_i + (1-Σ_i π_i) * FPR_avg
        total_prior = np.sum(priors)
        avg_fpr = float(np.mean(1 - sp_arr))
        p_t_pos = float(np.sum(priors * se_arr)) + max(0, 1 - total_prior) * avg_fpr
        p_t_pos = np.clip(p_t_pos, 0.001, 0.999)
        p_t_neg = 1.0 - p_t_pos

        # 事後確率: P(D_i|T+) ∝ π_i * Se_i
        posteriors_pos = priors * se_arr
        if posteriors_pos.sum() > 0:
            posteriors_pos = posteriors_pos / posteriors_pos.sum()

        # P(D_i|T-) ∝ π_i * (1-Se_i)
        posteriors_neg = priors * (1 - se_arr)
        if posteriors_neg.sum() > 0:
            posteriors_neg = posteriors_neg / posteriors_neg.sum()

        h_pos = self._weighted_entropy(posteriors_pos, clinical_weights)
        h_neg = self._weighted_entropy(posteriors_neg, clinical_weights)

        info_gain = h_prior - (p_t_pos * h_pos + p_t_neg * h_neg)
        return max(0.0, info_gain)

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
        # 原則: 検査と無関係な疾患は LR=1.0（確率不変）
        priors = np.array([c.get("prior", 0.0) for c in candidates])
        posteriors = np.zeros_like(priors)

        for i, c in enumerate(candidates):
            p = priors[i]
            name = c["disease_name"]
            interp = interpretation.get(name, {})
            is_positive = interp.get("判定", "陽性") == "陽性"

            if name in disease_se_sp:
                # この疾患にSe/Spが格納されている → 関連あり
                se, sp = disease_se_sp[name]
                if is_positive:
                    posteriors[i] = p * se
                else:
                    posteriors[i] = p * (1 - se)
            elif not has_stored_se_sp:
                # 全疾患にSe/Sp格納なし（症状・ROS等）→ LLM推定値を使用
                se = float(interp.get("se", 0.5))
                sp = float(interp.get("sp", 0.5))
                if is_positive:
                    posteriors[i] = p * se
                else:
                    posteriors[i] = p * (1 - se)
            else:
                # Se/Sp格納ありの検査だが、この疾患には無関係 → LR=1.0（確率不変）
                posteriors[i] = p

        # 正規化
        total = posteriors.sum()
        if total > 0:
            posteriors = posteriors / total

        updated = []
        for i, c in enumerate(candidates):
            updated.append({
                **c,
                "prior": float(posteriors[i]),
            })
        updated.sort(key=lambda x: x["prior"], reverse=True)

        return updated, interpretation

    def find_matching_tests(self, candidates: list, done_tests: list = None) -> list:
        """
        候補疾患のrelevant_testsから、UIで選択可能な検査名リストを返す。
        done_testsに含まれる検査は除外。
        """
        if done_tests is None:
            done_tests = []
        done_set = set(self.test_name_map.get(t, t) for t in done_tests)

        test_names = set()
        for c in candidates:
            d = self.disease_db.get(c["disease_name"])
            if not d:
                continue
            for t in d.get("relevant_tests", []):
                tname = self.test_name_map.get(t["test_name"], t["test_name"])
                if tname not in done_set:
                    test_names.add(tname)

        return sorted(test_names)
