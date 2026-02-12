"""
基準範囲テーブル生成: 329検査の基準範囲をLLMで一括生成
数値で表される検査のみ。画像検査・手技・問診は null。
"""
import json
import os
import time
import asyncio
from openai import AsyncOpenAI
from config import (
    GENERATE_LLM_API_KEY, GENERATE_LLM_BASE_URL, GENERATE_LLM_MODEL,
    DATA_DIR,
)

OUTPUT_FILE = os.path.join(DATA_DIR, "reference_ranges.json")
TESTS_JSONL = os.path.join(DATA_DIR, "tests.jsonl")
MAX_CONCURRENT = 3  # RPM制限に合わせる
BATCH_SIZE = 30

client = AsyncOpenAI(
    api_key=GENERATE_LLM_API_KEY,
    base_url=GENERATE_LLM_BASE_URL,
    timeout=60,
)

SYSTEM_PROMPT = """あなたは臨床検査医学の専門家です。
検査の基準範囲（成人）を提供してください。

ルール:
- 数値で結果が出る検査のみ基準範囲を返す
- 画像検査（CT, MRI, X線, エコー等）→ null
- 手技・処置（生検, 穿刺等）→ null
- 問診・身体診察項目 → null
- 定性検査（陽性/陰性のみ）→ null
- 半定量（+/-/++/+++のみ）→ null
- 基準範囲は日本の一般的な成人基準値
- lower/upperは数値（文字列不可）
- unitは日本の臨床で使われる標準単位

出力はJSONオブジェクトのみ。説明文不要。最初の文字は { にすること。"""

USER_TEMPLATE = """以下の検査について基準範囲を返してください。

{test_list}

出力形式:
{{
  "検査名": {{
    "lower": 下限値,
    "upper": 上限値,
    "unit": "単位"
  }},
  "画像検査名": null,
  ...
}}"""


async def generate_batch(test_names: list, batch_idx: int) -> dict:
    """1バッチ分の基準範囲を生成"""
    test_list = "\n".join(f"- {name}" for name in test_names)
    user_msg = USER_TEMPLATE.format(test_list=test_list)

    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=GENERATE_LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=8192,
            )
            content = resp.choices[0].message.content.strip()

            # JSON抽出
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                content = content[start:end]

            # trailing comma修復
            import re
            content = re.sub(r",\s*([}\]])", r"\1", content)

            result = json.loads(content)
            print(f"  batch {batch_idx}: {len(result)} tests OK")
            return result

        except Exception as e:
            print(f"  batch {batch_idx} 失敗 (試行{attempt+1}): {e}")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)

    print(f"  batch {batch_idx} 3回失敗、スキップ")
    return {}


async def main():
    # 全検査名を読み込み
    test_names = []
    with open(TESTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                test_names.append(t["test_name"])
            except (json.JSONDecodeError, KeyError):
                continue

    print(f"全{len(test_names)}検査の基準範囲を生成")

    # 既存の結果をロード（途中再開用）
    existing = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"既存: {len(existing)}件")

    # 未処理のみ抽出
    remaining = [n for n in test_names if n not in existing]
    if not remaining:
        print("全件処理済み")
        return

    print(f"未処理: {len(remaining)}件")

    # バッチ分割
    batches = []
    for i in range(0, len(remaining), BATCH_SIZE):
        batches.append((i // BATCH_SIZE, remaining[i:i + BATCH_SIZE]))

    print(f"{len(batches)}バッチ、{MAX_CONCURRENT}並行で実行")

    # 並行実行
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def run_with_sem(batch_idx, names):
        async with sem:
            return await generate_batch(names, batch_idx)

    tasks = [run_with_sem(idx, names) for idx, names in batches]
    results = await asyncio.gather(*tasks)

    # マージ
    for result in results:
        existing.update(result)

    # 保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=1)

    # 統計
    n_numeric = sum(1 for v in existing.values() if v is not None)
    n_null = sum(1 for v in existing.values() if v is None)
    print(f"\n完了: {len(existing)}件 (数値: {n_numeric}, 非数値: {n_null})")


if __name__ == "__main__":
    asyncio.run(main())
