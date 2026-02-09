"""
VeSMed - 検査名 名寄せスクリプト
diseases.jsonl内のrelevant_testsの検査名を正規化するマッピングを生成する。
tests.jsonlの329件を正規名リストとしてLLMに参照させ、バッチで名寄せを行う。
"""

import json
import argparse
import asyncio
import os
import threading
from openai import AsyncOpenAI
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    DISEASES_JSONL, TESTS_JSONL, DATA_DIR,
)

MAX_CONCURRENCY = 30
BATCH_SIZE = 50

MAP_FILE = os.path.join(DATA_DIR, "test_name_map.json")

SYSTEM_PROMPT = """\
あなたは日本の臨床医学に精通した専門家です。
検査名の名寄せ（正規化）を行います。

## タスク
与えられた「生の検査名リスト」の各検査名を、「正規名リスト」の中から最も適切な検査名にマッピングしてください。

## ルール
1. 生の検査名が正規名リストのいずれかと同一・同義・上位互換であれば、その正規名にマッピング
   - 例: 「D-ダイマー」→「D-dimer」（正規名リストにD-dimerがある場合）
   - 例: 「12誘導心電図」→「心電図 (ECG)」（正規名リストに心電図 (ECG)がある場合）
   - 例: 「心エコー検査」→「経胸壁心エコー」（正規名リストに経胸壁心エコーがある場合）
2. 複合検査（例: 「凝固検査(PT/APTT/Dダイマー)」）は、最も臨床的に重要な単一検査にマッピング
3. 正規名リストに適切な対応がない場合は、簡潔で標準的な検査名を自分で決めて出力
   - 同義の検査は必ず同じ名前に統一すること
4. 出力はJSONのみ。説明文やマークダウンは不要。最初の文字は { にすること。

## 出力形式
{"生の検査名1": "正規名", "生の検査名2": "正規名", ...}
"""


def load_canonical_names():
    """tests.jsonlから正規検査名リストを読み込む"""
    names = []
    if not os.path.exists(TESTS_JSONL):
        return names
    with open(TESTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                names.append(obj["test_name"])
            except (json.JSONDecodeError, KeyError):
                continue
    return sorted(set(names))


def load_raw_test_names():
    """diseases.jsonlから全ユニーク検査名を抽出する"""
    names = set()
    with open(DISEASES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                for t in d.get("relevant_tests", []):
                    names.add(t["test_name"])
            except (json.JSONDecodeError, KeyError):
                continue
    return sorted(names)


def load_existing_map():
    """既存のマッピングを読み込む"""
    if not os.path.exists(MAP_FILE):
        return {}
    with open(MAP_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_json(text):
    """LLMの出力からJSONを抽出する"""
    import re
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # { から始まる部分を抽出
    start = text.find("{")
    if start > 0:
        text = text[start:]

    text = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 不完全なJSONの修復
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        last_complete = max(text.rfind("}"), text.rfind("]"))
        if last_complete > 0:
            text = text[:last_complete + 1]
            text = re.sub(r",\s*([}\]])", r"\1", text)
            open_braces = text.count("{") - text.count("}")
            open_brackets = text.count("[") - text.count("]")
            text += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析失敗: {e}\n末尾100文字: {text[-100:]}")


# 結果の書き込みロック
_map_lock = threading.Lock()
_result_map = {}


async def normalize_batch(client, semaphore, canonical_text, batch, batch_idx, total_batches):
    """1バッチの検査名を名寄せする"""
    async with semaphore:
        label = f"[{batch_idx + 1}/{total_batches}]"
        raw_list = "\n".join(f"- {name}" for name in batch)

        user_prompt = f"""\
## 正規名リスト（この中から選ぶこと。該当なしの場合は自分で標準名を決める）
{canonical_text}

## 生の検査名リスト（これらを正規化してください）
{raw_list}

上記の生の検査名それぞれについて、正規名リストの中から最適な検査名にマッピングしてください。
出力はJSON形式のみ。"""

        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=65536,
            )
            content = response.choices[0].message.content
            mapping = extract_json(content)

            mapped = 0
            with _map_lock:
                for raw_name, canonical_name in mapping.items():
                    if raw_name in [n for n in batch]:  # バッチ内の名前のみ
                        _result_map[raw_name] = canonical_name
                        mapped += 1

            print(f"  {label} ... OK ({mapped}/{len(batch)}件マッピング)")
            return True
        except Exception as e:
            print(f"  {label} ... ERROR: {e}")
            return False


async def run_normalize(args):
    """名寄せマッピングを生成する"""
    canonical_names = load_canonical_names()
    raw_names = load_raw_test_names()

    print(f"正規名: {len(canonical_names)}件")
    print(f"生の検査名: {len(raw_names)}件")

    # 既存マッピングを読み込み
    existing = load_existing_map()
    if not args.force:
        remaining = [n for n in raw_names if n not in existing]
    else:
        remaining = raw_names

    print(f"既存マッピング: {len(existing)}件")
    print(f"処理対象: {len(remaining)}件")
    print(f"バッチサイズ: {BATCH_SIZE}  最大並行数: {args.concurrency}")

    if not remaining:
        print("すべてマッピング済みです。")
        return

    if args.dry_run:
        print(f"ドライラン: {len(remaining)}件の検査名を処理予定")
        return

    # 正規名リストのテキスト
    canonical_text = "\n".join(f"- {name}" for name in canonical_names)

    # バッチ分割
    batch_size = args.batch_size if hasattr(args, 'batch_size') else BATCH_SIZE
    batches = []
    for i in range(0, len(remaining), batch_size):
        batches.append(remaining[i:i + batch_size])

    print(f"バッチ数: {len(batches)}")

    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    semaphore = asyncio.Semaphore(args.concurrency)

    global _result_map
    _result_map = dict(existing)  # 既存マッピングを引き継ぎ

    tasks = [
        normalize_batch(client, semaphore, canonical_text, batch, i, len(batches))
        for i, batch in enumerate(batches)
    ]

    results = await asyncio.gather(*tasks)
    success = sum(1 for r in results if r)
    errors = sum(1 for r in results if not r)

    # 保存
    with open(MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(_result_map, f, ensure_ascii=False, indent=2)

    print(f"\n完了: 成功{success}バッチ, エラー{errors}バッチ")
    print(f"マッピング総数: {len(_result_map)}件")
    print(f"保存先: {MAP_FILE}")

    # 統計
    canonical_set = set(canonical_names)
    mapped_to_canonical = sum(1 for v in _result_map.values() if v in canonical_set)
    unique_targets = len(set(_result_map.values()))
    print(f"正規名リストへのマッピング: {mapped_to_canonical}件")
    print(f"ユニーク正規化名: {unique_targets}件（{len(_result_map)}件 → {unique_targets}件に集約）")


def main():
    parser = argparse.ArgumentParser(description="VeSMed 検査名 名寄せ")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY,
                        help=f"最大並行数（デフォルト: {MAX_CONCURRENCY}）")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"バッチサイズ（デフォルト: {BATCH_SIZE}）")
    parser.add_argument("--dry-run", action="store_true",
                        help="処理対象を表示するだけで実行しない")
    parser.add_argument("--force", action="store_true",
                        help="既存マッピングを無視して全件再処理")

    args = parser.parse_args()
    asyncio.run(run_normalize(args))


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
