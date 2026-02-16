"""
画像・内視鏡検査の Screen Hypothesis 生成: 全所見網羅列挙

findings_description (6000-16000字) から、検出可能な全所見を網羅的に列挙。
メカニズムは書かない。所見名のカタログ。
"""

import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_DIR, TESTS_JSONL, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from openai import AsyncOpenAI

IMAGING_CATEGORIES = {
    "画像検査（超音波）",
    "画像検査（CT）",
    "画像検査（MRI）",
    "画像検査（X線・造影）",
    "画像検査（核医学）",
    "内視鏡検査",
}

SYSTEM_PROMPT = """あなたは放射線科・内視鏡科の専門医です。以下の検査について、検出可能な**全所見を網羅的に列挙**してください。

## 目的
この所見リストは、embedding空間で疾患ベクトルとのcos類似度を計算するために使われます。
所見を漏れなく列挙することで、各疾患との類似度に差が生まれ、鑑別力が向上します。

## 出力ルール

1. **所見名のみ列挙**。読点（、）区切りで1段落にまとめよ
2. **疾患名は絶対に書かない**（例: ×「肺炎の浸潤影」→ ○「浸潤影」）
3. **メカニズム説明は不要**。所見の名称・パターン名のみ
4. 解剖学的部位や分布パターンが鑑別に重要な場合は付記してよい
   - 例: 「片側葉性浸潤影」「両側びまん性すりガラス影」「心尖部壁運動異常」
5. **正常所見は含めない**。異常所見のみ
6. findings_descriptionに記載されている所見は全て含めよ。漏れは許容しない
7. 前置き・説明は不要。所見リストのみを直接出力せよ"""


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


def clean_output(text):
    text = re.sub(r"```.*?\n?", "", text).strip()
    text = re.sub(r"^#{1,4}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    # 箇条書きを読点区切りに変換
    lines = text.split("\n")
    if any(l.strip().startswith(("-", "・", "●", "•")) for l in lines):
        items = []
        for l in lines:
            l = re.sub(r"^[\s\-・●•\d.]+", "", l).strip()
            if l:
                items.append(l)
        text = "、".join(items)
    return text.strip()


async def main():
    print("=" * 80)
    print("画像・内視鏡検査 Screen Hypothesis: 全所見網羅列挙")
    print("=" * 80)

    tests = load_tests()
    targets = [t for t in tests
               if t.get("category", "") in IMAGING_CATEGORIES
               and t.get("findings_description")]
    print(f"対象: {len(targets)}件")

    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    semaphore = asyncio.Semaphore(10)
    success = 0
    failed = 0

    async def gen_one(test_entry):
        nonlocal success, failed
        name = test_entry["test_name"]
        fd = test_entry.get("findings_description", "")

        user_content = (
            f"検査名: {name}\n"
            f"カテゴリ: {test_entry.get('category', '')}\n\n"
            f"以下のfindings_descriptionに記載されている全所見を網羅的に列挙してください:\n\n"
            f"{fd[:12000]}"
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
                        temperature=0.2,
                        max_tokens=4096,
                    )
                    text = clean_output(response.choices[0].message.content)
                    old = test_entry.get("hypothesis_text", "")
                    test_entry["hypothesis_screen"] = text
                    success += 1
                    print(f"  {name}: {len(old)}字 → {len(text)}字")
                    return
            except Exception as e:
                if attempt < 2:
                    print(f"  [{name}] retry {attempt+1}: {e}")
                    await asyncio.sleep((attempt + 1) * 5)
                else:
                    print(f"  [{name}] FAILED: {e}")
                    failed += 1

    print(f"\n--- LLM生成開始 ({len(targets)}件) ---")
    start = time.time()
    tasks = [gen_one(t) for t in targets]
    await asyncio.gather(*tasks)
    elapsed = time.time() - start
    print(f"\n完了: {success}成功, {failed}失敗 ({elapsed:.0f}秒)")

    # 統計
    import numpy as np
    old_lens = [len(t.get("hypothesis_text", "")) for t in targets]
    new_lens = [len(t.get("hypothesis_screen", "")) for t in targets if t.get("hypothesis_screen")]
    print(f"\n旧hypothesis_text: mean={np.mean(old_lens):.0f}字")
    print(f"新hypothesis_screen: mean={np.mean(new_lens):.0f}字, "
          f"median={np.median(new_lens):.0f}, min={min(new_lens)}, max={max(new_lens)}")

    save_tests(tests)
    print(f"\ntests.jsonl保存完了")


if __name__ == "__main__":
    asyncio.run(main())
