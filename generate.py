"""
VeSMed - 疾患メタデータ生成スクリプト（並行処理版）
LLMを呼び出して疾患ごとの構造化メタデータ（Se/Sp含む）をJSONLで生成する。
断点続行対応：既に生成済みの疾患はスキップする。
最大30並行でAPIを呼び出す。
"""

import json
import argparse
import asyncio
import re
import os
import threading
from openai import AsyncOpenAI
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    DISEASE_LIST_FILE, TEST_LIST_FILE,
    DISEASES_JSONL, TESTS_JSONL, FINDINGS_JSONL, DATA_DIR,
)

MAX_CONCURRENCY = 30

SYSTEM_PROMPT = """\
あなたは日本の臨床医学に精通した専門家です。
与えられた疾患名に対して、構造化された臨床メタデータをJSON形式で生成してください。

## 最重要ルール
- 出力はJSONのみ。マークダウンのコードブロック(```)や説明文は絶対に含めないでください。
- すべて日本語で記述してください。

## description_for_embedding について（最重要フィールド）
このフィールドはベクトル検索に使われます。実際の患者が来院した時の典型的な臨床像を、
入院時病歴要約のスタイルで200-400字で記述してください。
- 好発年齢・性別
- 典型的な主訴と現病歴の経過
- 主要な身体所見
- リスク因子
を自然な文章に含めてください。教科書的定義ではなく「この疾患の患者は典型的にこう来院する」
という記述にしてください。

## relevant_tests について
- この疾患の診断に真に鑑別価値のある検査のみ列挙（通常10-30件）
- **test_nameは必ず後述の「正規検査名リスト」から選択すること**
  - リストにない特殊検査のみ自由記述可（ただし最小限に）
- sensitivity/specificityは[下界, 上界]の小数で記述（例: [0.85, 0.95]）
- 教科書・診療ガイドラインの一般的な値を参考に
- condition_notesに、Se/Spが変動する条件（年齢、重症度、時期など）を明記
"""

USER_PROMPT_TEMPLATE = """\
以下の疾患の構造化メタデータをJSON形式で生成してください。

疾患名：{disease_name}
診療科：{category}

出力するJSONスキーマ：
{{
  "disease_name": "（入力と同じ疾患名）",
  "icd10": "（ICD-10コード）",
  "category": "（入力と同じ診療科名）",
  "description_for_embedding": "（200-400字の臨床像記述）",
  "age_peak": "（好発年齢、例: '50-70代'）",
  "gender_tendency": "（性差、例: '男性に多い(3:1)' or '差なし'）",
  "urgency": "（超緊急/緊急/準緊急/通常 のいずれか）",
  "risk_factors": ["リスク因子1", "リスク因子2"],
  "core_symptoms": ["（>70%の患者に出現する症状）"],
  "common_symptoms": ["（30-70%の患者に出現する症状）"],
  "rare_but_specific": ["（<30%だが出現すれば特異的な症状）"],
  "core_signs": ["（主要な身体所見）"],
  "differential_top5": [
    {{"disease": "鑑別疾患名", "key_difference": "鑑別ポイント"}}
  ],
  "relevant_tests": [
    {{
      "test_name": "検査名",
      "purpose": "この疾患に対するこの検査の目的",
      "sensitivity": [0.0, 0.0],
      "specificity": [0.0, 0.0],
      "condition_notes": "Se/Spが変動する条件",
      "cost_level": 1,
      "invasiveness": 0,
      "turnaround_minutes": 5
    }}
  ]
}}

cost_levelは1-5の整数（1=安い、5=高い）。
invasivenessは0-5の整数（0=非侵襲、5=高侵襲）。
turnaround_minutesは結果が出るまでの目安時間（分単位の整数）。

## 正規検査名リスト（test_nameはこのリストから選ぶこと）
{canonical_tests}
"""


def parse_disease_list(filepath):
    """疾患リスト.txtをパースして疾患名と診療科のリストを返す"""
    diseases = []
    current_category = ""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("###"):
                current_category = line.lstrip("#").strip()
                if ". " in current_category:
                    current_category = current_category.split(". ", 1)[1]
            elif line.startswith("*"):
                disease_name = line.lstrip("*").strip()
                if disease_name:
                    diseases.append({
                        "name": disease_name,
                        "category": current_category,
                    })
    return diseases


def parse_test_list(filepath):
    """検査リスト.txtをパースして検査名とカテゴリのリストを返す"""
    tests = []
    current_category = ""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("###"):
                current_category = line.lstrip("#").strip()
                if ". " in current_category:
                    current_category = current_category.split(". ", 1)[1]
            elif line.startswith("*"):
                test_name = line.lstrip("*").strip()
                if test_name:
                    tests.append({
                        "name": test_name,
                        "category": current_category,
                    })
    return tests


def get_generated_names(jsonl_path, key="disease_name"):
    """既に生成済みの名前セットを返す"""
    names = set()
    if not os.path.exists(jsonl_path):
        return names
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                names.add(obj.get(key, ""))
            except json.JSONDecodeError:
                continue
    return names


def extract_json(text):
    """LLMの出力からJSONを抽出する。不完全なJSON・trailing commaにも対応。"""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    text = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
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


# ファイル書き込み用ロック（並行書き込み防止）
_write_lock = threading.Lock()


def append_jsonl(filepath, obj):
    """スレッドセーフにJSONLファイルに1行追記する"""
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with _write_lock:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line)


# ----------------------------------------------------------------
# 疾患メタデータ生成（並行版）
# ----------------------------------------------------------------

async def generate_disease_one(client, semaphore, disease, index, total, canonical_tests_text=""):
    """1つの疾患のメタデータを非同期で生成する"""
    async with semaphore:
        name = disease["name"]
        category = disease["category"]
        label = f"[{index + 1}/{total}] {category} / {name}"
        try:
            user_prompt = USER_PROMPT_TEMPLATE.format(
                disease_name=name, category=category,
                canonical_tests=canonical_tests_text,
            )
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=65536,
            )
            content = response.choices[0].message.content
            metadata = extract_json(content)
            metadata["disease_name"] = name
            metadata["category"] = category

            append_jsonl(DISEASES_JSONL, metadata)
            n_tests = len(metadata.get("relevant_tests", []))
            print(f"  {label} ... OK (検査{n_tests}件)")
            return True
        except Exception as e:
            print(f"  {label} ... ERROR: {e}")
            return False


async def cmd_diseases_async(args):
    """疾患メタデータを並行生成する"""
    diseases = parse_disease_list(DISEASE_LIST_FILE)

    if args.category:
        diseases = [d for d in diseases if args.category in d["category"]]

    generated = get_generated_names(DISEASES_JSONL, key="disease_name")
    remaining = [d for d in diseases if d["name"] not in generated]

    print(f"対象: {len(diseases)}疾患  生成済み: {len(generated)}件  残り: {len(remaining)}件")
    print(f"最大並行数: {args.concurrency}")

    if not remaining:
        print("すべて生成済みです。")
        return

    if args.dry_run:
        for d in remaining:
            print(f"  {d['category']} / {d['name']}")
        return

    # 正規検査名リストを読み込み（存在する場合）
    canonical_tests_text = ""
    if os.path.exists(TEST_LIST_FILE):
        test_list = parse_test_list(TEST_LIST_FILE)
        canonical_tests_text = "\n".join(f"- {t['name']}" for t in test_list)

    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = [
        generate_disease_one(client, semaphore, disease, i, len(remaining), canonical_tests_text)
        for i, disease in enumerate(remaining)
    ]

    results = await asyncio.gather(*tasks)
    success = sum(1 for r in results if r)
    errors = sum(1 for r in results if not r)
    print(f"\n完了: 成功{success}件, エラー{errors}件")


# ----------------------------------------------------------------
# 検査メタデータ生成（並行版）
# ----------------------------------------------------------------

TESTS_SYSTEM_PROMPT = """\
あなたは日本の臨床医学に精通した専門家です。
与えられた検査名に対して、その検査の固有属性をJSON形式で生成してください。

## 最重要ルール
- 出力はJSONのみ。思考過程、説明文、マークダウンのコードブロック(```)は絶対に含めないでください。
- 最初の文字は必ず { にしてください。
- すべて日本語で記述してください。

## description_for_embedding について（最重要フィールド）
このフィールドはベクトル検索でコスト推定に使われます。
検査の実施手順・必要な設備/スタッフ・患者への身体的負担・合併症リスクを
事実として100-200字で記述してください。
- 主観的評価（「高い」「低い」等）は不要、客観的事実のみ
- 例（冠動脈造影）: 「大腿動脈または橈骨動脈からカテーテルを挿入し冠動脈に造影剤を
  注入するX線透視下の侵襲的検査。カテーテル室と専門チームを要する。局所麻酔下で施行し
  入院が必要。出血・血腫・造影剤アレルギー・血管損傷等の合併症リスクあり。」
- 例（CRP）: 「末梢静脈から採血し、血清中のC反応性蛋白を免疫比濁法で定量する血液検査。
  外来採血室で実施可能。穿刺部の軽微な疼痛以外に身体的負担なし。」
"""

TESTS_USER_PROMPT_TEMPLATE = """\
以下の検査の固有属性をJSON形式で生成してください。

検査名：{test_name}
検査カテゴリ：{category}

出力するJSONスキーマ：
{{
  "test_name": "（入力と同じ検査名）",
  "category": "（入力と同じカテゴリ）",
  "description_for_embedding": "（100-200字の手技事実記述）",
  "turnaround_minutes": 5,
  "sample_type": "（採血/採尿/画像/生理検査/穿刺液/組織/なし など）",
  "contraindications": ["禁忌1", "禁忌2"],
  "notes": "（補足事項）"
}}

turnaround_minutesは結果が出るまでの目安時間（分単位の整数）。
"""


async def generate_test_one(client, semaphore, test, index, total):
    """1つの検査の固有属性を非同期で生成する"""
    async with semaphore:
        name = test["name"]
        category = test["category"]
        label = f"[{index + 1}/{total}] {category} / {name}"
        try:
            user_prompt = TESTS_USER_PROMPT_TEMPLATE.format(
                test_name=name, category=category,
            )
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": TESTS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=65536,
            )
            content = response.choices[0].message.content
            metadata = extract_json(content)
            metadata["test_name"] = name
            metadata["category"] = category

            append_jsonl(TESTS_JSONL, metadata)
            print(f"  {label} ... OK")
            return True
        except Exception as e:
            print(f"  {label} ... ERROR: {e}")
            return False


async def cmd_tests_async(args):
    """検査メタデータを並行生成する"""
    tests = parse_test_list(TEST_LIST_FILE)

    if args.category:
        tests = [t for t in tests if args.category in t["category"]]

    generated = get_generated_names(TESTS_JSONL, key="test_name")
    remaining = [t for t in tests if t["name"] not in generated]

    print(f"対象: {len(tests)}検査  生成済み: {len(generated)}件  残り: {len(remaining)}件")
    print(f"最大並行数: {args.concurrency}")

    if not remaining:
        print("すべて生成済みです。")
        return

    if args.dry_run:
        for t in remaining:
            print(f"  {t['category']} / {t['name']}")
        return

    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = [
        generate_test_one(client, semaphore, test, i, len(remaining))
        for i, test in enumerate(remaining)
    ]

    results = await asyncio.gather(*tasks)
    success = sum(1 for r in results if r)
    errors = sum(1 for r in results if not r)
    print(f"\n完了: 成功{success}件, エラー{errors}件")


# ----------------------------------------------------------------
# 病歴・身体所見メタデータ生成（並行版）
# ----------------------------------------------------------------

FINDINGS_SYSTEM_PROMPT = """\
あなたは日本の臨床医学に精通した専門家です。
与えられた疾患名に対して、その疾患の診断に有用な「病歴聴取項目」と「身体診察所見」を
Se/Sp付きで構造化JSON形式で生成してください。

## 最重要ルール
- 出力はJSONのみ。思考過程、説明文、マークダウンのコードブロック(```)は絶対に含めないでください。
- 最初の文字は必ず { にしてください。
- すべて日本語で記述してください。

## 生成する項目
1. **病歴聴取項目**: 問診で確認すべき症状・既往・生活歴など
   - 「〜の既往があるか」「〜の曝露歴があるか」のような具体的な質問
   - 陽性なら疾患を支持、陰性なら否定する情報
2. **身体診察所見**: 視診・触診・聴診・打診で確認すべき所見
   - 「項部硬直」「心雑音」「蝶形紅斑」のような具体的な所見

## Se/Spについて
- sensitivity/specificityは[下界, 上界]の小数で記述
- 教科書・ガイドラインの値を参考に
- 病歴項目は一般にSeは高いがSpは低い傾向
- 特異的な身体所見はSpが高い傾向
"""

FINDINGS_USER_PROMPT_TEMPLATE = """\
以下の疾患について、診断に有用な病歴聴取項目と身体診察所見をJSON形式で生成してください。

疾患名：{disease_name}
診療科：{category}

出力するJSONスキーマ：
{{
  "disease_name": "（入力と同じ疾患名）",
  "history_items": [
    {{
      "test_name": "（具体的な病歴項目。例: 'IV薬物使用歴'、'最近の歯科処置歴'）",
      "purpose": "この疾患に対するこの病歴の意義",
      "sensitivity": [0.0, 0.0],
      "specificity": [0.0, 0.0],
      "condition_notes": "Se/Spが変動する条件",
      "cost_level": 1,
      "invasiveness": 0,
      "turnaround_minutes": 1
    }}
  ],
  "exam_items": [
    {{
      "test_name": "（具体的な身体所見。例: '項部硬直'、'Osler結節'）",
      "purpose": "この疾患に対するこの所見の意義",
      "sensitivity": [0.0, 0.0],
      "specificity": [0.0, 0.0],
      "condition_notes": "Se/Spが変動する条件",
      "cost_level": 1,
      "invasiveness": 0,
      "turnaround_minutes": 1
    }}
  ]
}}

病歴項目は5-15件、身体診察所見は5-15件を目安に生成してください。
鑑別価値の高いもの（Se/Spのいずれかが高いもの）を優先してください。
"""


async def generate_finding_one(client, semaphore, disease, index, total):
    """1つの疾患の病歴・身体所見を非同期で生成する"""
    async with semaphore:
        name = disease["name"]
        category = disease["category"]
        label = f"[{index + 1}/{total}] {category} / {name}"
        try:
            user_prompt = FINDINGS_USER_PROMPT_TEMPLATE.format(
                disease_name=name, category=category,
            )
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": FINDINGS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=65536,
            )
            content = response.choices[0].message.content
            metadata = extract_json(content)
            metadata["disease_name"] = name
            metadata["category"] = category

            append_jsonl(FINDINGS_JSONL, metadata)
            n_hist = len(metadata.get("history_items", []))
            n_exam = len(metadata.get("exam_items", []))
            print(f"  {label} ... OK (病歴{n_hist}件, 診察{n_exam}件)")
            return True
        except Exception as e:
            print(f"  {label} ... ERROR: {e}")
            return False


async def cmd_findings_async(args):
    """病歴・身体所見メタデータを並行生成する"""
    diseases = parse_disease_list(DISEASE_LIST_FILE)

    if args.category:
        diseases = [d for d in diseases if args.category in d["category"]]

    generated = get_generated_names(FINDINGS_JSONL, key="disease_name")
    remaining = [d for d in diseases if d["name"] not in generated]

    print(f"対象: {len(diseases)}疾患  生成済み: {len(generated)}件  残り: {len(remaining)}件")
    print(f"最大並行数: {args.concurrency}")

    if not remaining:
        print("すべて生成済みです。")
        return

    if args.dry_run:
        for d in remaining:
            print(f"  {d['category']} / {d['name']}")
        return

    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = [
        generate_finding_one(client, semaphore, disease, i, len(remaining))
        for i, disease in enumerate(remaining)
    ]

    results = await asyncio.gather(*tasks)
    success = sum(1 for r in results if r)
    errors = sum(1 for r in results if not r)
    print(f"\n完了: 成功{success}件, エラー{errors}件")


# ----------------------------------------------------------------
# メインエントリポイント
# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VeSMed メタデータ生成")
    subparsers = parser.add_subparsers(dest="command")

    p_diseases = subparsers.add_parser("diseases", help="疾患メタデータを生成")
    p_diseases.add_argument("--category", type=str, default=None,
                            help="生成対象の診療科でフィルタ（部分一致）")
    p_diseases.add_argument("--dry-run", action="store_true",
                            help="生成対象を表示するだけで実行しない")
    p_diseases.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY,
                            help=f"最大並行数（デフォルト: {MAX_CONCURRENCY}）")

    p_tests = subparsers.add_parser("tests", help="検査メタデータを生成")
    p_tests.add_argument("--category", type=str, default=None,
                         help="生成対象のカテゴリでフィルタ（部分一致）")
    p_tests.add_argument("--dry-run", action="store_true",
                         help="生成対象を表示するだけで実行しない")
    p_tests.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY,
                         help=f"最大並行数（デフォルト: {MAX_CONCURRENCY}）")

    p_findings = subparsers.add_parser("findings", help="病歴・身体所見メタデータを生成")
    p_findings.add_argument("--category", type=str, default=None,
                            help="生成対象の診療科でフィルタ（部分一致）")
    p_findings.add_argument("--dry-run", action="store_true",
                            help="生成対象を表示するだけで実行しない")
    p_findings.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY,
                            help=f"最大並行数（デフォルト: {MAX_CONCURRENCY}）")

    args = parser.parse_args()

    if args.command == "diseases":
        asyncio.run(cmd_diseases_async(args))
    elif args.command == "tests":
        asyncio.run(cmd_tests_async(args))
    elif args.command == "findings":
        asyncio.run(cmd_findings_async(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
