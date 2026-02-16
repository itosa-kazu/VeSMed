"""
Phase 2 Screen Hypothesis 全件生成

非画像検査 (~280件) に対して、Phase 2メカニズム拡張hypothesis (300-500字) を生成。
画像・内視鏡検査は旧hypothesis_textを維持。

生成結果は tests.jsonl の hypothesis_screen フィールドに保存（旧 hypothesis_text は保持）。
パイロット済み30件は pilot_extended_hypothesis.json から読み込み。
"""

import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_DIR, TESTS_JSONL

# --- API設定（Vertex AI） ---
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from openai import AsyncOpenAI

PILOT_FILE = os.path.join(DATA_DIR, "pilot_extended_hypothesis.json")

# 画像・内視鏡カテゴリ（Phase 2対象外 → 旧hypothesis維持）
IMAGING_CATEGORIES = {
    "画像検査（超音波）",
    "画像検査（CT）",
    "画像検査（MRI）",
    "画像検査（X線・造影）",
    "画像検査（核医学）",
    "内視鏡検査",
}

# --- プロンプト（パイロットと同一） ---
SYSTEM_PROMPT = """あなたは臨床検査医学の専門家です。以下の検査について、sim_matrix（疾患embeddingとの内積による類似度行列）用の「病態生理学的受容野テキスト」を生成してください。

このテキストは、embedding空間において「この検査がどのような病態生理学的領域を観測するか」を定義するベクトルの元となります。

## 入力
- 検査名と検査の詳細記述（findings_description）が提供されます

## 出力要件
**文字数**: 300〜500字（日本語）の単一の段落テキスト

**【厳守事項1: 疾患名の完全マスキング】**
特定の疾患名（例: 肺炎、SLE、心筋梗塞、クローン病、川崎病など）は**絶対に記述してはならない**。
代わりに、臓器・組織名、病態生理学的メカニズム、上位概念を用いて記述せよ。
- NG: 「肺炎、敗血症の鑑別に有用」
- OK: 「下気道の急性感染性炎症、全身性炎症反応の重症度評価」

**【厳守事項2: 極性の中立化】**
結果の方向性（上昇/低下/陽性/陰性）を断定しない。
- NG: 「CRP上昇を認めた。炎症を示唆する」
- OK: 「急性期炎症反応の有無と程度の評価、組織破壊の検出」

**【厳守事項3: 臨床コンテキストの保持】**
この検査がオーダーされる臨床状況（どんな主訴・バイタル異常・身体所見の患者に対して）、
そしてこの検査が鑑別空間をどう切り分けるか（どの病態群を区別できるか）を記述に含めよ。

**【厳守事項4: 結果のみを出力】**
説明、前置き、箇条書きは不要。テキスト本文のみを直接出力せよ。"""


def load_tests():
    tests = []
    with open(TESTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tests.append(json.loads(line))
    return tests


def save_tests(tests):
    with open(TESTS_JSONL, "w", encoding="utf-8") as f:
        for t in tests:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def clean_llm_output(text):
    text = re.sub(r"```.*?\n?", "", text).strip()
    text = re.sub(r"^#{1,4}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    return text.strip()


async def main():
    print("=" * 80)
    print("Phase 2 Screen Hypothesis 全件生成")
    print("=" * 80)

    # ─── データ読み込み ───
    tests = load_tests()
    tests_by_name = {t["test_name"]: t for t in tests}

    # パイロット済みデータ読み込み
    pilot_data = {}
    if os.path.exists(PILOT_FILE):
        with open(PILOT_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for name, entry in raw.items():
            if isinstance(entry, dict):
                pilot_data[name] = entry.get("extended_hypothesis", "")
            else:
                pilot_data[name] = entry
    print(f"パイロット済み: {len(pilot_data)}件")

    # 対象分類
    fd_tests = [t for t in tests if t.get("findings_description")]
    imaging_tests = {t["test_name"] for t in fd_tests if t.get("category", "") in IMAGING_CATEGORIES}
    already_has = {t["test_name"] for t in fd_tests if t.get("hypothesis_screen")}

    to_generate = []
    pilot_applied = 0
    imaging_kept = 0

    for t in fd_tests:
        name = t["test_name"]
        if name in imaging_tests:
            # 画像検査 → 旧hypothesis維持
            if not t.get("hypothesis_screen"):
                t["hypothesis_screen"] = t.get("hypothesis_text", f"{name} 異常")
            imaging_kept += 1
        elif name in pilot_data and pilot_data[name]:
            # パイロット済み → そのまま適用
            t["hypothesis_screen"] = pilot_data[name]
            pilot_applied += 1
        elif name in already_has:
            pass  # 既に生成済み
        else:
            to_generate.append(t)

    print(f"画像（旧維持）: {imaging_kept}件")
    print(f"パイロット適用: {pilot_applied}件")
    print(f"既存: {len(already_has)}件")
    print(f"新規生成対象: {len(to_generate)}件")

    if not to_generate:
        print("全件処理済み")
        save_tests(tests)
        return

    # ─── LLM生成 ───
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    semaphore = asyncio.Semaphore(10)
    success = 0
    failed = 0

    async def gen_one(test_entry):
        nonlocal success, failed
        name = test_entry["test_name"]
        fd = test_entry.get("findings_description", "")
        if not fd:
            return

        user_content = (
            f"検査名: {name}\n"
            f"カテゴリ: {test_entry.get('category', '')}\n\n"
            f"{fd[:8000]}"
        )

        for attempt in range(3):
            try:
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=0.3,
                        max_tokens=2048,
                    )
                    text = clean_llm_output(response.choices[0].message.content)
                    test_entry["hypothesis_screen"] = text
                    success += 1
                    if success % 20 == 0:
                        print(f"  生成完了: {success}/{len(to_generate)} ...")
                    return
            except Exception as e:
                if attempt < 2:
                    wait = (attempt + 1) * 5
                    print(f"  [{name}] retry {attempt+1}: {e}")
                    await asyncio.sleep(wait)
                else:
                    print(f"  [{name}] FAILED: {e}")
                    failed += 1

    print(f"\n--- LLM生成開始 ({len(to_generate)}件, model={LLM_MODEL}) ---")
    start = time.time()

    # バッチに分けて実行（API負荷制御）
    batch_size = 50
    for i in range(0, len(to_generate), batch_size):
        batch = to_generate[i:i + batch_size]
        tasks = [gen_one(t) for t in batch]
        await asyncio.gather(*tasks)
        elapsed = time.time() - start
        print(f"  バッチ {i//batch_size + 1}/{(len(to_generate)-1)//batch_size + 1} 完了 "
              f"({success}成功, {failed}失敗, {elapsed:.0f}秒)")

    print(f"\n生成完了: {success}成功, {failed}失敗")

    # ─── 統計 ───
    screen_lengths = []
    old_lengths = []
    for t in fd_tests:
        hs = t.get("hypothesis_screen", "")
        ho = t.get("hypothesis_text", "")
        if hs:
            screen_lengths.append(len(hs))
        if ho:
            old_lengths.append(len(ho))

    if screen_lengths:
        import numpy as np
        print(f"\nhypothesis_screen文字数:")
        print(f"  mean={np.mean(screen_lengths):.0f}, median={np.median(screen_lengths):.0f}, "
              f"min={min(screen_lengths)}, max={max(screen_lengths)}")
        print(f"hypothesis_text（旧）文字数:")
        print(f"  mean={np.mean(old_lengths):.0f}, median={np.median(old_lengths):.0f}")

    # ─── 保存 ───
    save_tests(tests)
    print(f"\ntests.jsonl保存完了 (hypothesis_screen追加、hypothesis_text保持)")


if __name__ == "__main__":
    asyncio.run(main())
