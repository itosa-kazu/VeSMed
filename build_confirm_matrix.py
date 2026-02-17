"""
Dual sim_matrix: 確認用行列 (sim_matrix_confirm) の構築

diseases.jsonl の relevant_tests から逆引きインデックスを構築し、
各検査の確認用hypothesis（疾患名リスト）を embed して sim_matrix_confirm.npz を生成。

sim_matrix_screen  (既存 sim_matrix.npz):  Part A, Part E に使用
sim_matrix_confirm (本スクリプト生成):     Part B, Part C に使用

確認用hypothesis = "疾患A、疾患B、..."（疾患名のみ、プレフィックスなし）
→ 疾患名をembeddingに直接含めることで、Part C (cluster_mu - global_mu) の
   ターゲットアンカーを最大化する。
"""

import json
import os
import sys
import hashlib
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import time

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
    DISEASES_JSONL, TESTS_JSONL, DATA_DIR,
)

TEST_NAME_MAP_FILE = os.path.join(DATA_DIR, "test_name_map.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "sim_matrix_confirm.npz")
REVERSE_INDEX_FILE = os.path.join(DATA_DIR, "confirm_reverse_index.json")


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_test_name_map():
    if os.path.exists(TEST_NAME_MAP_FILE):
        with open(TEST_NAME_MAP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# 厳密パターンフィルタ: purpose に明示的な他疾患除外パターンを含み、
# かつ確認系キーワードを含まないエントリを除外する
_EXCLUSION_PATTERNS = ['の除外', 'を除外', 'との鑑別', 'との区別', 'の否定']
_CONFIRM_KEYWORDS = [
    '確定', '診断', '評価', '検出', '確認', '重症度',
    'モニタリング', '活動性', '測定', '程度',
]

# 手動レビュー済み除去リスト (Claude Opus 4.6 臨床判定)
# 厳密パターンでは捕捉できないが、臨床的に他疾患除外目的と判定されたエントリ
_MANUAL_EXCLUSIONS = {
    ("睡眠時無呼吸症候群", "甲状腺機能検査（TSH, FT4）"),
    ("完全房室ブロック", "心筋トロポニンT/I"),
    ("完全房室ブロック", "心臓MRI"),
    ("胃潰瘍", "腹部超音波検査"),
    ("大腸憩室炎", "下部消化管内視鏡（大腸カメラ）"),
    ("急性C型肝炎", "HBs抗原"),
    ("過敏性腸症候群", "便潜血検査(2回法)"),
    ("慢性膵炎", "IgG4"),
    ("巣状分節性糸球体硬化症", "抗核抗体(ANA)"),
    ("膜性腎症", "血清補体価（C3, C4, CH50）"),
    ("フィッシャー症候群", "テンシロンテスト（エドロホニウム試験）"),
    ("脊髄小脳変性症", "抗神経抗体（抗Yo, Hu, Ri等）"),
    ("ウイルス性髄膜炎", "血液培養"),
    ("緊張型頭痛", "頸椎レントゲン/CT"),
    ("顔面神経麻痺 (Bell麻痺)", "水痘・帯状疱疹ウイルス(VZV)抗体価・PCR"),
    ("非ホジキンリンパ腫", "HTLV-1抗体"),
    ("関節リウマチ", "抗核抗体 (ANA)"),
    ("偶発性低体温症", "甲状腺機能検査(TSH, FT3, FT4)"),
    ("偶発性低体温症", "血中コルチゾール"),
    ("偶発性低体温症", "尿中薬物スクリーニング"),
    ("乳腺炎", "プロカルシトニン"),
    ("視神経炎", "血液検査（抗AQP4抗体）"),
    ("視神経炎", "血液検査（抗MOG抗体）"),
    ("水疱性類天疱瘡", "Tzanck試験（細胞診）"),
    ("突発性難聴", "ティンパノメトリー"),
    ("メニエール病", "血液検査（甲状腺機能、梅毒、自己免疫等）"),
    ("化膿性関節炎", "単純X線（患部）"),
    ("うつ病", "頭部MRI"),
    ("百日咳", "マイコプラズマPCR"),
    ("多系統萎縮症", "ビタミンB12"),
    ("クレチン症", "抗サイログロブリン抗体"),
    ("クレチン症", "抗TPO抗体"),
    ("クレチン症", "TSH受容体抗体 (TRAb/TSAb)"),
    ("新生児呼吸窮迫症候群", "プロカルシトニン (PCT)"),
    ("抗リン脂質抗体症候群", "抗核抗体 (ANA)"),
    ("性器ヘルペス", "RPR (梅毒定性/定量)"),
    ("リチウム中毒", "血糖 (随時/空腹時)"),
    ("自閉スペクトラム症（ASD）", "脳波検査"),
    ("自閉スペクトラム症（ASD）", "染色体検査 (G分染法)"),
}


def _is_pure_exclusion(purpose: str) -> bool:
    """purposeが純粋な除外目的かどうか判定（厳密パターン）。"""
    has_excl = any(p in purpose for p in _EXCLUSION_PATTERNS)
    if not has_excl:
        return False
    has_conf = any(kw in purpose for kw in _CONFIRM_KEYWORDS)
    return not has_conf


def build_reverse_index(diseases, canonical_names, name_map):
    """
    diseases.jsonl の relevant_tests から逆引きインデックスを構築。
    test_name → set of disease_names

    厳密パターンフィルタ: purposeが明示的に他疾患除外を示すエントリは除去。
    """
    reverse = defaultdict(set)
    unmatched_tests = defaultdict(int)
    filtered_count = 0

    for d in diseases:
        dname = d.get("disease_name", "")
        if not dname:
            continue
        for rt in d.get("relevant_tests", []):
            tname_raw = rt.get("test_name", "")
            if not tname_raw:
                continue
            # 除外目的エントリをスキップ（厳密パターン + 手動レビュー）
            purpose = rt.get("purpose", "")
            if _is_pure_exclusion(purpose):
                filtered_count += 1
                continue
            if (dname, tname_raw) in _MANUAL_EXCLUSIONS:
                filtered_count += 1
                continue
            # test_name_mapで正規化
            canonical = name_map.get(tname_raw, tname_raw)
            if canonical in canonical_names:
                reverse[canonical].add(dname)
            else:
                unmatched_tests[tname_raw] += 1

    return reverse, unmatched_tests, filtered_count


def load_disease_embeddings():
    """disease_embs.npzからMEAN集約済み疾患embeddingを読み込む。"""
    embs_file = os.path.join(DATA_DIR, "disease_embs.npz")
    if not os.path.exists(embs_file):
        raise FileNotFoundError(f"{embs_file} が見つかりません。index.pyを先に実行してください。")
    data = np.load(embs_file, allow_pickle=True)
    return list(data["disease_names"]), data["disease_embs_normed"].astype(np.float32)


def batch_embed(texts, batch_size=50, max_workers=10):
    """テキストリストをバッチ並行でembedding。"""
    client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    batches = []
    for i in range(0, len(texts), batch_size):
        batches.append((i, texts[i:i + batch_size]))

    all_embs = [None] * len(texts)

    def _embed_one(batch_info):
        start_idx, batch = batch_info
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                return start_idx, [item.embedding for item in resp.data]
            except Exception as e:
                print(f"  [embed] batch(offset={start_idx}) 失敗(試行{attempt+1}): {e}")
                time.sleep(2 ** attempt)
        return start_idx, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_embed_one, b) for b in batches]
        done = 0
        for future in as_completed(futures):
            start_idx, result = future.result()
            if result is None:
                print(f"  [embed] batch(offset={start_idx}) 完全失敗")
                return None
            for j, emb in enumerate(result):
                all_embs[start_idx + j] = emb
            done += 1
            if done % 5 == 0 or done == len(batches):
                print(f"  Embedding batch {done}/{len(batches)} done")

    return np.array(all_embs, dtype=np.float32)


def main():
    print("=" * 80)
    print("Dual sim_matrix: 確認用行列 (sim_matrix_confirm) 構築")
    print("=" * 80)

    # ─── データ読み込み ───
    diseases = load_jsonl(DISEASES_JSONL)
    tests = load_jsonl(TESTS_JSONL)
    name_map = load_test_name_map()

    # findings_descriptionがある検査のみ（engine.pyと同じフィルタ）
    test_names = [t["test_name"] for t in tests if t.get("findings_description")]
    canonical_set = set(test_names)
    print(f"\n疾患マスタ: {len(diseases)}件")
    print(f"検査マスタ: {len(test_names)}件（findings_description有り）")
    print(f"表記揺れ辞書: {len(name_map)}件")

    # ─── 逆引きインデックス構築 ───
    print("\n--- 逆引きインデックス構築（除外フィルタ適用） ---")
    reverse_index, unmatched, filtered_count = build_reverse_index(
        diseases, canonical_set, name_map
    )
    print(f"  除外フィルタで除去: {filtered_count}件")

    # 統計
    matched_tests = [t for t in test_names if t in reverse_index]
    no_match_tests = [t for t in test_names if t not in reverse_index]
    disease_counts = [len(reverse_index[t]) for t in matched_tests]

    print(f"  逆引きマッチ: {len(matched_tests)}/{len(test_names)}検査")
    print(f"  未マッチ検査: {len(no_match_tests)}件")
    if disease_counts:
        print(f"  対象疾患数: mean={np.mean(disease_counts):.1f}, "
              f"median={np.median(disease_counts):.0f}, "
              f"min={min(disease_counts)}, max={max(disease_counts)}")

    # 上位/下位の確認
    sorted_by_count = sorted(
        [(t, len(reverse_index[t])) for t in matched_tests],
        key=lambda x: x[1], reverse=True
    )
    print(f"\n  疾患数Top10:")
    for tname, cnt in sorted_by_count[:10]:
        print(f"    {tname}: {cnt}疾患")
    print(f"  疾患数Bottom5:")
    for tname, cnt in sorted_by_count[-5:]:
        print(f"    {tname}: {cnt}疾患")

    if unmatched:
        top_unmatched = sorted(unmatched.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n  正規化不能な検査名Top10:")
        for tname, cnt in top_unmatched:
            print(f"    {tname}: {cnt}疾患で参照")

    # ─── 逆引きインデックス保存 ───
    reverse_for_save = {
        t: sorted(list(reverse_index[t])) for t in test_names if t in reverse_index
    }
    with open(REVERSE_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(reverse_for_save, f, ensure_ascii=False, indent=2)
    print(f"\n逆引きインデックス保存: {REVERSE_INDEX_FILE}")

    # ─── 確認用hypothesis生成 ───
    print("\n--- 確認用hypothesis生成 ---")
    confirm_texts = []
    for tname in test_names:
        if tname in reverse_index and len(reverse_index[tname]) > 0:
            dlist = sorted(reverse_index[tname])
            text = '、'.join(dlist)
        else:
            # フォールバック: screen hypothesis と同じ（hypothesis_text or "検査名 異常"）
            test_entry = next((t for t in tests if t["test_name"] == tname), None)
            custom = test_entry.get("hypothesis_text", "") if test_entry else ""
            text = custom if custom else f"{tname} 異常"
        confirm_texts.append(text)

    lengths = [len(t) for t in confirm_texts]
    print(f"  hypothesis_confirm生成完了: {len(confirm_texts)}件")
    print(f"  文字数: mean={np.mean(lengths):.0f}, "
          f"median={np.median(lengths):.0f}, "
          f"min={min(lengths)}, max={max(lengths)}")

    # ─── 疾患embedding読み込み ───
    print("\n--- 疾患embedding読み込み ---")
    disease_names, disease_embs_normed = load_disease_embeddings()
    print(f"  {len(disease_names)}疾患 × {disease_embs_normed.shape[1]}次元")

    # ─── ハッシュチェック（キャッシュ） ───
    h_hash = hashlib.md5("||".join(confirm_texts).encode()).hexdigest()[:12]
    if os.path.exists(OUTPUT_FILE):
        data = np.load(OUTPUT_FILE, allow_pickle=True)
        cached_hash = str(data.get("hypothesis_hash", ""))
        cached_diseases = list(data.get("disease_names", []))
        cached_tests = list(data.get("test_names", []))
        if (cached_hash == h_hash
                and cached_diseases == disease_names
                and cached_tests == test_names):
            print(f"\n既存キャッシュと一致 → スキップ")
            print(f"  shape: {data['sim_matrix'].shape}")
            return
        print(f"  キャッシュ不一致 → 再計算")

    # ─── Embedding ───
    print(f"\n--- Embedding ({len(confirm_texts)}件) ---")
    confirm_embs = batch_embed(confirm_texts)
    if confirm_embs is None:
        print("ERROR: Embedding失敗")
        return

    # 正規化
    h_norms = np.linalg.norm(confirm_embs, axis=1, keepdims=True)
    h_norms[h_norms == 0] = 1.0
    confirm_embs_normed = confirm_embs / h_norms

    # ─── sim_matrix_confirm 計算 ───
    print("\n--- sim_matrix_confirm 計算 ---")
    sim_matrix_confirm = disease_embs_normed @ confirm_embs_normed.T
    print(f"  shape: {sim_matrix_confirm.shape}")
    print(f"  mean: {sim_matrix_confirm.mean():.4f}")
    print(f"  std:  {sim_matrix_confirm.std():.4f}")

    # 列ごとの分散統計
    col_vars = np.var(sim_matrix_confirm, axis=0)
    print(f"  列分散: mean={col_vars.mean():.6f}, "
          f"median={np.median(col_vars):.6f}, "
          f"min={col_vars.min():.6f}, max={col_vars.max():.6f}")

    # ─── 保存 ───
    np.savez(
        OUTPUT_FILE,
        sim_matrix=sim_matrix_confirm,
        disease_names=np.array(disease_names, dtype=object),
        test_names=np.array(test_names, dtype=object),
        hypothesis_hash=np.array(h_hash),
    )
    print(f"\n保存完了: {OUTPUT_FILE}")
    print(f"  {sim_matrix_confirm.shape[0]}疾患 × {sim_matrix_confirm.shape[1]}検査")


if __name__ == "__main__":
    main()
