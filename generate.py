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
    CLAUDE_API_KEY, CLAUDE_BASE_URL, CLAUDE_MODEL,
    DISEASE_LIST_FILE, TEST_LIST_FILE,
    DISEASES_JSONL, TESTS_JSONL, FINDINGS_JSONL, DATA_DIR,
)

# Vertex AI設定
VERTEX_SA_KEY = os.path.join(os.path.dirname(__file__), "thermal-outlet-483512-m4-8ec9647654b6.json")
VERTEX_PROJECT = "thermal-outlet-483512-m4"
VERTEX_LOCATION = "global"
VERTEX_MODEL = "gemini-3-pro-preview"

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
# 所見ベースdescription生成（疾患・検査を同一所見空間に配置）
# ----------------------------------------------------------------

DISEASE_FINDINGS_V1_PROMPT = """\
あなたは日本の臨床医学に精通した専門家です。
与えられた疾患について、患者で観察されるすべての所見を網羅的かつ詳細に列挙してください。

## ルール
- 出力はプレーンテキストのみ（JSON不要、マークダウン不要、コードブロック不要）
- 以下の5カテゴリすべてについて、該当する所見を漏れなく詳細に記述すること：
  1. 症状（主訴、随伴症状、陰性症状も含む）
  2. バイタルサイン（体温、脈拍、血圧、呼吸数、SpO2）
  3. 身体所見（視診、触診、打診、聴診）
  4. 血液・尿検査異常（具体的な検査名と異常の方向：上昇/低下/陽性等）
  5. 画像・生理検査所見（X線、CT、MRI、エコー、心電図等の具体的異常）
- 各カテゴリ内で頻度の高い所見から順に記述
- 教科書的定義や病態説明は不要、観察可能な所見のみ
- 所見ごとに括弧で補足情報（程度、条件、時期）を付記

## 出力例（急性心筋梗塞）
突然発症の胸骨後部圧迫痛（持続20分以上、ニトロ無効、左肩・左腕・顎への放散）、冷汗、嘔気嘔吐、呼吸困難、死の恐怖感、失神（下壁梗塞で迷走神経反射時）。発熱（37-38℃、24-48時間後）、頻脈（交感神経亢進）または徐脈（下壁梗塞で房室ブロック時）、血圧低下（広範囲梗塞・右室梗塞でショック）、頻呼吸、SpO2低下（肺うっ血時）。顔面蒼白、冷汗、頸静脈怒張（右心不全合併時）、III音・IV音聴取、心尖部収縮期雑音（乳頭筋機能不全による僧帽弁逆流）、肺野湿性ラ音（Killip II以上）、下腿浮腫（右心不全時）。心電図ST上昇（責任冠動脈領域の誘導）、対側誘導のST低下（mirror image）、異常Q波（数時間〜数日後）、T波陰転化。トロポニンI/T著明上昇（発症3-6時間後、ピーク12-24時間）、CK-MB上昇（発症4-8時間後）、LDH上昇（遅発性、24-48時間後）、白血球増多（12,000-15,000/μL）、CRP上昇（24-48時間後）、BNP/NT-proBNP上昇、AST上昇、血糖上昇（ストレス反応）、D-dimer軽度上昇。心エコー壁運動異常（責任領域のakinesis/hypokinesis）、EF低下、僧帽弁逆流、心嚢液貯留（Dressler症候群時）。胸部X線で肺うっ血像、心拡大、胸水（重症時）。"""

DISEASE_FINDINGS_V2_PROMPT = """\
あなたは日本の臨床医学に精通した専門家です。
与えられた疾患について、実臨床で観察されるありとあらゆるパターンを以下の構造に従って、可能な限り詳細に記述してください。
出力の長さに制限はありません。網羅性と詳細さが最も重要です。

## 最重要ルール
- 出力はプレーンテキストのみ（JSON不要、マークダウン不要、コードブロック不要）
- 「観察可能な事実」のみ記述。病態生理の説明、治療法、予後は書かない
- 「〜が起こりうる」「〜の可能性がある」ではなく「〜を認める」と断定的に記述する
- 頻度・程度は括弧で補足（例: 「発熱(38-40℃, 約90%)」）
- 検査名は正式名称を使用（略語のみ不可、例: 「CK (CPK)」「BNP」）
- 単一パターンを断定しない。病型・亜型・重症度で所見が異なる場合は全パターンを等価に列挙する
- 省略しない。「など」「等」で逃げず、知っているものを全部書く

## 記述構造（この順序で、各項目を十分に詳しく記述すること）

### 1. 好発背景・リスク因子・誘因
この疾患を発症する患者に観察される背景因子。
- 好発年齢・性別
- 基礎疾患（糖尿病、CKD、肝硬変、HIV、移植後等）
- 生活歴（飲酒、喫煙、IV薬物使用、職業曝露等）
- 薬剤歴（ステロイド、免疫抑制薬、OCP、抗凝固薬等）
- 直近のイベント（手術、長期臥床、歯科処置、旅行、動物接触等）
- 家族歴で重要なもの

### 2. 典型来院像
患者が救急/外来に来た時の最も典型的な訴えと経過。
- 主訴、発症様式（突然/緩徐/段階的）
- 症状の時間経過パターン（移動、増悪、間欠、crescendo等）
- 増悪因子・緩解因子（体位、呼吸、食事、運動等）
- 随伴症状
- この疾患に特徴的な訴え方・表現

### 3. 非典型来院像・ピットフォール
典型像と異なるすべての来院パターン。見逃しは患者の死に直結する。
- 高齢者での異なる初発症状
- 糖尿病患者での症状修飾（自律神経障害による無痛性等）
- 免疫不全・ステロイド使用下での症状マスク（発熱欠如、CRP抑制等）
- 小児での特異的な表現
- 薬剤による所見修飾（β遮断薬→頻脈マスク、抗凝固薬→出血増加/PT延長等）
- 無症候性・症状が軽微な場合
- 他疾患と誤認されやすいプレゼンテーション（下壁MI→心窩部痛→胃腸炎と誤診等）
- 病型・亜型による来院像の違い

### 4. バイタルサイン・身体所見
- バイタルサイン: 体温、脈拍、血圧、呼吸数、SpO2の具体的数値と方向
  - 病型・亜型・重症度による全バリエーションを併記
  - 軽症から重症までのスペクトラム
- 視診・触診・打診・聴診の具体的所見
- 特殊手技・誘発テスト（Murphy徴候、Kernig徴候、Jolt accentuation、Babinski反射、straight leg raise等、該当するものすべて）とその結果
- 出現頻度の高いものから順に

### 5. 検査所見パターン
- 血液検査: 具体的な検査名と異常の方向・程度（上昇/低下/陽性/陰性）、数値の目安
- 尿検査・髄液検査・その他体液検査
- 画像検査: X線、CT、MRI、エコー、血管造影等の具体的異常所見
- 生理検査: 心電図、脳波、肺機能、神経伝導等の異常パターン
- 発症からの時期による検査値の変化（例: トロポニンは発症3-6時間後から上昇、ピーク12-24時間）
- 重症度による検査値の違い
- 病型・亜型による検査パターンの違い

### 6. 鑑別キー
この疾患を類似疾患と区別する決定的な所見の組み合わせ。
- 主要な鑑別疾患それぞれに対して「Aを示唆しBを否定する所見」を具体的に列挙
- 検査値の程度差（軽度上昇 vs 著明上昇）による鑑別
- 時間経過による鑑別（急性 vs 亜急性 vs 慢性）
- 所見の組み合わせパターンによる鑑別

## 重要: 出力の詳細さについて
- 出力例と同等以上の詳細さで記述すること。出力例より短い出力は不可
- 全セクションで省略せず、知っている全パターンを書き切ること
- 簡潔さは不要。網羅性が最優先

## 出力例（急性虫垂炎）
10-30代に好発のピークを持つが、あらゆる年齢で発症する。男性がやや多い（男女比約1.2-1.4:1）。生涯発症リスクは男性8.6%、女性6.7%。糞石（成人の閉塞原因として最多、X線/CTで石灰化として描出）。小児・若年者ではウイルス感染（アデノウイルス、EBウイルス等）や細菌性腸炎に続発するリンパ濾胞過形成が閉塞原因となる。異物誤飲（種子、魚骨、バリウム残渣）。寄生虫（蟯虫、回虫、特に海外渡航歴のある患者）。虫垂腫瘍（カルチノイド、腺癌、粘液嚢胞腺腫 — 中高年の虫垂炎では鑑別に挙げる）。炎症性腸疾患（Crohn病の回盲部病変に続発）。妊娠中（特に妊娠中期以降、子宮増大による虫垂の頭側偏位で診断が遅れやすい）。家族歴あり（第一度近親者に虫垂炎の既往があるとリスク上昇）。低繊維食・高脂肪食の食生活。便秘傾向。直近のバリウム検査歴。免疫抑制状態（HIV、移植後、化学療法中）では穿孔のリスクが高く、炎症反応が乏しいまま進行する。

心窩部または臍周囲の漠然とした内臓痛で発症する。「なんとなくお腹が気持ち悪い」「みぞおちが重い」と表現されることが多く、初期には局在が不明瞭。4-12時間（典型的には6-8時間）かけて痛みが右下腹部（McBurney点付近）に移動・局在化する（visceral-somatic sequence / Kocher徴候、約50-60%に認める）。食欲不振はほぼ全例（90%以上）に認め、しばしば痛みに先行する — 「食欲がある虫垂炎は少ない」は臨床的に有用な原則。悪心（70-80%）、嘔吐（50-60%、通常は痛みの出現後に出現。痛みに先行する嘔吐は急性胃腸炎を示唆）。微熱（37.2-38.0℃）。歩行時、咳嗽時、段差を降りる時、車の振動で右下腹部痛が増悪する（腹膜刺激症状）。右股関節をやや屈曲させた姿勢で静止し、動くことを嫌がる。痛みは持続的で、消長を繰り返す疝痛パターンではない（疝痛パターンは尿管結石や腸閉塞を示唆）。排便・排ガスで一時的にやや楽になることがあるが完全には消失しない。

高齢者（65歳以上）: 痛みの訴えが軽微または非典型的で、「なんとなく調子が悪い」「食欲がない」「微熱がある」程度の主訴で来院する。腹壁の筋萎縮により筋性防御が出にくい。炎症マーカー（CRP、白血球）の上昇が遅延または軽微。結果として穿孔率が高い（高齢者の虫垂炎の40-70%が穿孔例）。せん妄や意識変容が初発症状のことがある。小児（5歳以下）: 病歴聴取が困難。不機嫌、ぐずり、嗜眠、食事拒否、歩行拒否、ジャンプを嫌がる。右下肢を屈曲させて横たわる。腹部診察に対する強い抵抗。下痢を伴う頻度が成人より高い（約30%）。穿孔率が高い（特に3歳以下では80%以上）。大網の発達が未熟なため穿孔後の被覆が不十分で汎発性腹膜炎に至りやすい。妊婦: 子宮増大に伴い虫垂が右上腹部・右側腹部に偏位する（妊娠20週以降は古典的なMcBurney点の圧痛が消失）。右上腹部痛、右側腹部痛として来院し、胆嚢炎、腎盂腎炎、HELLP症候群と誤診される。妊娠に伴う生理的白血球増多（妊娠後期で10,000-16,000/μL）のため、白血球増多が虫垂炎の指標として使いにくい。後腹膜位虫垂（retrocecal position、約15-20%）: 虫垂が盲腸の背側に位置するため、右下腹部の圧痛・筋性防御が軽微。代わりに右側腹部痛、右腰背部痛、右CVA叩打痛（腎盂腎炎と誤診）。歩行時・右股関節伸展時の痛み（psoas signが陽性になりやすい）。骨盤位虫垂（pelvic position、約30%）: 腹壁から遠く、腹部圧痛が軽微。膀胱刺激症状（頻尿、排尿時痛、残尿感）を認め、膀胱炎・尿路感染と誤診される。直腸刺激症状（しぶり腹 / tenesmus、下痢）を認め、感染性腸炎と誤診される。直腸診で右側圧痛が重要な手がかりとなる。β遮断薬内服中: 頻脈がマスクされ、穿孔・腹膜炎に至っても心拍数が上昇しない。ステロイド内服中: 発熱・CRP上昇が抑制され、白血球は見かけ上増加するが好中球左方移動は判断しにくい。腹膜刺激症状も軽微化する。免疫抑制薬内服中: 炎症反応全般が乏しく、CTでの脂肪織濃度上昇も軽微。抗菌薬先行投与例（antibiotics-modified appendicitis）: 症状と検査値が一時改善するが虫垂炎は持続し、遅発性穿孔のリスクがある。穿孔のサイン: 発症から24-36時間以上経過。一時的に痛みが軽減・消失した後（虫垂内圧の解放）、腹部全体に広がる激痛が出現。38.5℃以上の高熱、頻脈、血圧低下、全身状態の悪化。汎発性腹膜炎の所見。

単純性虫垂炎: 体温37.2-38.0℃の微熱。脈拍は80-100回/分のやや頻脈または正常。血圧正常。呼吸数正常〜軽度増加。SpO2正常。壊疽性・穿孔性虫垂炎: 38.5-40℃の高熱。頻脈（100-120回/分以上）。汎発性腹膜炎→敗血症に至ると血圧低下（収縮期90mmHg以下）、頻呼吸、SpO2低下。視診で腹式呼吸の抑制。右股関節をやや屈曲した体位。穿孔・膿瘍時は右下腹部の膨隆。触診でMcBurney点（上前腸骨棘と臍を結ぶ線上の外側1/3）の限局性圧痛。Lanz点（左右の上前腸骨棘を結ぶ線上の右側1/3）の圧痛。筋性防御（随意性→不随意性→板状硬、腹膜炎の進行を反映）。Blumberg徴候（反跳痛）。Rovsing徴候（左下腹部圧迫で右下腹部痛が放散、陽性率40-70%）。Rosenstein徴候（左側臥位で右下腹部痛が増強）。Dunphy徴候（咳で右下腹部に響く）。Psoas徴候（左側臥位で右股関節を過伸展→右下腹部〜腰部痛誘発、後腹膜位虫垂を示唆）。Obturator徴候（右股関節90度屈曲+内旋→骨盤深部痛誘発、骨盤位虫垂を示唆）。Heel drop test（爪先立ちから踵を落とすと右下腹部に響く）。直腸診で右側壁の圧痛や腫瘤触知（骨盤位虫垂やDouglas窩膿瘍を示唆）。

血液検査で白血球増多（10,000-18,000/μL、好中球優位で左方移動）。白血球数20,000/μL以上は壊疽性・穿孔性虫垂炎や膿瘍形成を示唆。CRP上昇（発症12時間以降に上昇が明瞭化、単純性では1-5mg/dL、壊疽性・穿孔性では10mg/dL以上）。発症6時間以内はCRP正常でも虫垂炎は否定できない。プロカルシトニン上昇（穿孔・敗血症合併時、0.5ng/mL以上）。血清総ビリルビン軽度上昇（1.0mg/dL以上、壊疽性・穿孔性で門脈炎を反映、穿孔の独立予測因子）。尿検査で無菌性膿尿（尿中白血球陽性だが尿培養陰性、虫垂炎症の尿管・膀胱への波及、約20-30%）。妊娠反応（尿中hCG）陰性を確認（異所性妊娠の除外、妊娠可能年齢の女性では必須）。腹部造影CT（感度94-98%、特異度93-97%）で虫垂腫大（最大外径6mm以上）、虫垂壁の肥厚（2mm以上）と造影増強（target sign）、虫垂周囲脂肪織の濃度上昇（fat stranding）、糞石（appendicolith）、Arrowhead sign、虫垂周囲の液体貯留。壊疽性では壁の造影欠損、壁内気腫。穿孔時は壁の連続性断裂、腹腔内遊離ガス、膿瘍形成。腹部超音波検査で圧迫不能の管腔構造（non-compressible tubular structure）、最大短径6mm以上、壁肥厚、カラードプラで血流増加（壊疽性では消失）、虫垂周囲の高エコー脂肪。MRI（妊婦で造影CTを避ける場合）でT2高信号の虫垂壁肥厚と周囲脂肪浮腫。腹部単純X線で石灰化した糞石（約5-10%）、Sentinel loop sign、腰椎右側弯、右腸腰筋陰影消失。

急性胃腸炎は悪心・嘔吐が腹痛に先行し水様性下痢が顕著で圧痛の局在が不明瞭。腸間膜リンパ節炎は小児で上気道炎が先行し高熱を伴い、圧痛点がMcBurney点より内側で体位変換により移動する（Klein徴候）。右側大腸憩室炎は臨床のみでの鑑別困難でCTにより虫垂正常+憩室周囲炎症を確認。尿管結石は突然の疝痛+背部〜鼠径部放散痛+血尿で腹膜刺激症状を欠く。骨盤内炎症性疾患（PID）は両側下腹部痛+帯下増量+子宮頸部移動痛（chandelier sign）。異所性妊娠破裂は無月経+性器出血+ショック+尿中hCG陽性。卵巣嚢腫茎捻転は突然発症の激痛+エコーで腫大卵巣+whirlpool sign。Crohn病（回盲部）は慢性の下痢・体重減少の病歴+画像で回腸末端壁肥厚+skip lesion。"""

TEST_FINDINGS_PROMPT = """\
あなたは日本の臨床医学に精通した専門家です。
与えられた検査について、検出・評価できるすべての所見を網羅的かつ詳細に列挙してください。

## ルール
- 出力はプレーンテキストのみ（JSON不要、マークダウン不要、コードブロック不要）
- 以下の3カテゴリすべてについて詳細に記述すること：
  1. 異常所見とその臨床的意義（検査値の方向＋示唆する疾患・病態を具体的に列挙）
  2. 正常所見が除外できる疾患・病態
  3. 偽陽性・偽陰性の条件（薬剤、時期、合併症、年齢等）
- 具体的な所見名と方向（上昇/低下/陽性/陰性等）を明記
- 異常の程度による鑑別（軽度上昇 vs 著明上昇など）も記述

## 出力例（トロポニンI/T）
トロポニン上昇：心筋壊死を示唆。著明上昇（基準値の100倍以上）は急性心筋梗塞に特徴的（STEMI/NSTEMI）。中等度上昇（10-100倍）は急性心筋炎、たこつぼ心筋症、肺塞栓症（右室負荷）。軽度上昇（1-10倍）は慢性心不全、腎不全（クリアランス低下）、敗血症（需要虚血）、心房細動（頻脈性）、心臓手術後、激しい運動後。トロポニン正常：発症6時間以降であれば心筋壊死をほぼ否定（陰性的中率99%以上）。ただし超急性期（発症3時間以内）は偽陰性あり、6-12時間後の再検が必要。高感度トロポニンでは慢性腎不全で持続的軽度上昇（偽陽性）。経時的変化（rise and fall パターン）が急性心筋障害の診断に重要。"""


TEST_QUALITY_PROMPT = """\
あなたは日本の臨床医学に精通した専門家です。
与えられた検査について、その検査の「総合的な質」を評価するための記述を生成してください。

## ルール
- 出力はプレーンテキストのみ（JSON不要、マークダウン不要、コードブロック不要）
- 以下の6つの観点すべてについて、具体的な事実を詳細に記述すること：

### 1. 実施方法と身体的負担
- どのように実施するか（採血のみ？カテーテル挿入？全身麻酔？）
- 患者が受ける苦痛の程度（穿刺痛のみ？術中疼痛？術後疼痛？）
- 必要な体位・拘束時間
- 前処置の要否（絶食、下剤、造影剤等）

### 2. 合併症リスク
- 起こりうる合併症とその頻度・重症度
- 死亡リスクの有無
- 禁忌事項

### 3. 所要時間
- 検査自体の所要時間
- 結果が判明するまでの時間（即日？数日？数週間？）
- 予約の要否、待機時間

### 4. 金銭的コスト
- おおよその保険点数または費用感（数百円？数千円？数万円？数十万円？）
- 必要な設備・人員（外来採血室で可能？専用装置が必要？専門チームが必要？）

### 5. 致命的疾患の検出能力（Critical value）
- この検査で発見・除外できる致命的疾患（見逃すと死亡する疾患）を具体的に列挙
- 各疾患に対する診断精度（高い？中程度？スクリーニング程度？）
- 緊急性の高い疾患の早期発見にどの程度貢献するか

### 6. 治療方針への貢献（Curable value）
- この検査結果が治療方針をどう変えるか（抗菌薬選択？手術適応？化学療法レジメン？）
- 治療効果のモニタリングに使えるか
- 結果によって予後がどの程度改善するか

## 出力例（トロポニンI/T）
末梢静脈から採血のみで実施可能。穿刺部の軽微な疼痛以外に身体的負担なし。前処置不要、外来採血室で看護師が実施可能。合併症は穿刺部の皮下出血程度で重篤な合併症リスクはない。高感度トロポニンは院内検査で15-30分、通常法でも1時間以内に結果判明。保険点数は約120点（約360円）、一般的な自動分析装置で測定可能。急性心筋梗塞（STEMI/NSTEMI）の診断に不可欠で、発症6時間以降の陰性的中率は99%以上。急性心筋炎、たこつぼ心筋症、肺塞栓症の鑑別にも寄与。急性大動脈解離のスクリーニングにも補助的に使用。陽性の場合、緊急心臓カテーテル検査・経皮的冠動脈インターベンション（PCI）の適応判断に直結し、再灌流療法の開始時間が予後を決定する（door-to-balloon time 90分以内が目標）。経時的測定により心筋壊死の範囲推定と治療効果判定が可能。"""


def read_jsonl(filepath):
    """JSONLファイルを読み込んでリストで返す"""
    entries = []
    if not os.path.exists(filepath):
        return entries
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def write_jsonl(filepath, entries):
    """リストをJSONLファイルに書き出す"""
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def _gen_findings_one(client, semaphore, name, category, system_prompt, index, total):
    """1エントリの findings_description を生成（Gemini用・非ストリーミング）"""
    async with semaphore:
        label = f"[{index + 1}/{total}] {name}"
        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{name}（{category}）"},
                ],
                temperature=0.2,
                max_tokens=65536,
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"```.*?\n?", "", text).strip()
            print(f"  {label} ... OK ({len(text)}字)")
            return index, text
        except Exception as e:
            print(f"  {label} ... ERROR: {e}")
            return index, None


DISEASE_SECTION_PROMPTS = [
    # Section 1
    """\
あなたは日本の臨床医学に精通した専門家です。
与えられた疾患について、「好発背景・リスク因子・誘因」のみを詳細に記述してください。
出力はプレーンテキストのみ。省略しない。知っているものを全部書く。

記述内容:
- 好発年齢・性別（男女比、年齢分布）
- 基礎疾患（糖尿病、CKD、肝硬変、HIV、移植後等）
- 生活歴（飲酒、喫煙、IV薬物使用、職業曝露等）
- 薬剤歴（ステロイド、免疫抑制薬、OCP、抗凝固薬等）
- 直近のイベント（手術、長期臥床、歯科処置、旅行、動物接触等）
- 家族歴で重要なもの
- 遺伝的素因、環境因子""",
    # Section 2
    """\
あなたは日本の臨床医学に精通した専門家です。
与えられた疾患について、「典型来院像」のみを詳細に記述してください。
出力はプレーンテキストのみ。「〜を認める」と断定的に記述する。頻度は括弧で補足。省略しない。

記述内容（すべての項目について具体的かつ詳細に記述すること）:
- 主訴: 患者が実際に使う言葉・表現を複数パターン列挙（「胸が締め付けられる」「象が乗っているよう」等）
- 発症様式: 突然/緩徐/段階的の区別、発症から来院までの典型的な時間経過
- 症状の時間経過パターン: 発症初期→数時間後→来院時の推移を時系列で記述。移動・増悪・間欠・crescendoパターンの有無
- 増悪因子・緩解因子: 体位、呼吸、食事、運動、安静、特定の動作との関係を具体的に
- 随伴症状: 各随伴症状の出現頻度（%）、出現順序、組み合わせパターン
- 重症度スペクトラム: 軽症の典型像、中等症の典型像、重症の典型像をそれぞれ記述
- 病型・亜型ごとの典型像: 亜型・病型が存在する場合、それぞれの来院パターンを個別に記述
- 受診に至るきっかけ: 何が契機で受診を決意するか（症状持続、突然の悪化、随伴症状の出現等）
- 来院時の全身状態の印象: 「苦悶様顔貌で冷汗」「比較的元気に歩いて来院」等""",
    # Section 3
    """\
あなたは日本の臨床医学に精通した専門家です。
与えられた疾患について、「非典型来院像・ピットフォール」のみを詳細に記述してください。
出力はプレーンテキストのみ。見逃しは患者の死に直結する。全パターンを書く。

記述内容:
- 高齢者での異なる初発症状
- 糖尿病患者での症状修飾（自律神経障害による無痛性等）
- 免疫不全・ステロイド使用下での症状マスク（発熱欠如、CRP抑制等）
- 小児での特異的な表現
- 薬剤による所見修飾（β遮断薬→頻脈マスク、抗凝固薬→出血増加等）
- 無症候性・症状が軽微な場合
- 他疾患と誤認されやすいプレゼンテーション
- 病型・亜型による来院像の違い""",
    # Section 4
    """\
あなたは日本の臨床医学に精通した専門家です。
与えられた疾患について、「バイタルサイン・身体所見」のみを詳細に記述してください。
出力はプレーンテキストのみ。具体的数値を含める。全バリエーションを列挙。

記述内容:
- バイタルサイン: 体温、脈拍、血圧、呼吸数、SpO2の具体的数値と方向
  - 病型・亜型・重症度による全バリエーションを併記
  - 軽症から重症までのスペクトラム
- 視診・触診・打診・聴診の具体的所見
- 特殊手技・誘発テスト（Murphy徴候、Kernig徴候、Jolt accentuation、Babinski反射等、該当するものすべて）とその結果・感度・特異度
- 出現頻度の高いものから順に""",
    # Section 5
    """\
あなたは日本の臨床医学に精通した専門家です。
与えられた疾患について、「検査所見パターン」のみを詳細に記述してください。
出力はプレーンテキストのみ。検査名は正式名称。数値の目安を含める。

記述内容:
- 血液検査: 具体的な検査名と異常の方向・程度（上昇/低下/陽性/陰性）、数値の目安
- 尿検査・髄液検査・その他体液検査
- 画像検査: X線、CT、MRI、エコー、血管造影等の具体的異常所見
- 生理検査: 心電図、脳波、肺機能、神経伝導等の異常パターン
- 発症からの時期による検査値の変化
- 重症度による検査値の違い
- 病型・亜型による検査パターンの違い""",
    # Section 6
    """\
あなたは日本の臨床医学に精通した専門家です。
与えられた疾患について、「鑑別キー」のみを詳細に記述してください。
出力はプレーンテキストのみ。具体的な所見の組み合わせで記述。

記述内容:
- 主要な鑑別疾患それぞれに対して「Aを示唆しBを否定する所見」を具体的に列挙
- 検査値の程度差（軽度上昇 vs 著明上昇）による鑑別
- 時間経過による鑑別（急性 vs 亜急性 vs 慢性）
- 所見の組み合わせパターンによる鑑別
- この疾患の確定診断に最も重要な所見""",
]

SECTION_TITLES = [
    "好発背景・リスク因子・誘因",
    "典型来院像",
    "非典型来院像・ピットフォール",
    "バイタルサイン・身体所見",
    "検査所見パターン",
    "鑑別キー",
]


# ----------------------------------------------------------------
# 検査 findings_description V3 プロンプト（6セクション並列生成）
# sim_matrixの検査側を強化：疾患記述と語彙が重なるよう設計
# ----------------------------------------------------------------

TEST_SECTION_PROMPTS = [
    # Section 1: 適応臨床像
    """\
あなたは日本の臨床医学に精通した専門家です。
与えられた検査について、「適応臨床像」のみを詳細に記述してください。
出力はプレーンテキストのみ（JSON不要、マークダウン不要、コードブロック不要、見出し記号不要）。

この検査をオーダーすべき患者像を、来院時の所見で記述する。
「〜を主訴に来院した患者」「〜を呈する患者において実施する」の形式。

記述内容:
- 主訴と症状パターン（この検査をオーダーする契機となる訴え全パターン）
- バイタルサイン異常（発熱、頻脈、低血圧、頻呼吸、SpO2低下等、該当するもの全て）
- 身体所見（視診・触診・聴診で認める所見のうち、この検査のオーダー根拠となるもの）
- 救急・緊急で必要な場合と、外来・スクリーニングで必要な場合を分けて記述
- 年齢層・性別・基礎疾患による適応の違い
- 省略しない。この検査をオーダーする全ての臨床場面を網羅する""",

    # Section 2: 疾患別異常パターン（主要群）
    """\
あなたは日本の臨床医学に精通した専門家です。
与えられた検査について、「疾患別の異常パターン（主要疾患群）」のみを詳細に記述してください。
出力はプレーンテキストのみ（JSON不要、マークダウン不要、コードブロック不要、見出し記号不要）。

この検査が臨床的に重要な役割を果たす主要疾患（15〜20件）について、
各疾患ごとに具体的な検査所見パターンを記述する。

記述ルール:
- 疾患名は正式名称を使用
- 検査値の方向（上昇/低下/陽性/陰性）と程度（具体的数値・倍率）を必ず含める
- 「基準値の○倍以上」「○○以上/以下」等の数値目安を明記
- 各疾患でこの検査結果がどの程度特異的かを記述（この所見があればほぼ確定、vs 非特異的）
- 同じ疾患内でも病型・病期・重症度による検査値の違いがあれば全て記述
- 省略しない。知っている疾患別パターンを全部書く""",

    # Section 3: 疾患別異常パターン（追加群）
    """\
あなたは日本の臨床医学に精通した専門家です。
与えられた検査について、「疾患別の異常パターン（追加疾患群）」のみを詳細に記述してください。
出力はプレーンテキストのみ（JSON不要、マークダウン不要、コードブロック不要、見出し記号不要）。

主要疾患（最も一般的な15-20疾患）以外で、この検査が異常を示す疾患を網羅的に記述する。

対象:
- 頻度は低いが見逃すと致命的な疾患
- 特殊な集団（小児、高齢者、妊婦、免疫不全）で重要な疾患
- 内分泌・代謝疾患、自己免疫疾患、遺伝性疾患での異常パターン
- 薬剤性・医原性の異常パターン
- 稀だがこの検査が唯一の手がかりとなる疾患

記述ルール:
- 各疾患について検査値の方向・程度・特異性を記述
- 数値の目安を含める
- 省略しない。まれな疾患も含めて全部書く""",

    # Section 4: 鑑別パターン
    """\
あなたは日本の臨床医学に精通した専門家です。
与えられた検査について、「鑑別パターン」のみを詳細に記述してください。
出力はプレーンテキストのみ（JSON不要、マークダウン不要、コードブロック不要、見出し記号不要）。

同じ検査異常を示す複数の疾患を、この検査の所見パターンの違いで鑑別する方法を記述する。

記述内容:
- 異常の程度差による鑑別（軽度上昇 vs 中等度 vs 著明上昇で示唆する疾患群が異なる場合）
- 経時的変化パターンによる鑑別（一過性 vs 持続性、上昇速度、ピーク到達時間の違い）
- 他の検査所見との組み合わせによる鑑別パターン
  （「本検査高値 + ○○正常 → A疾患」「本検査高値 + ○○高値 → B疾患」の形式）
- 正常値でも除外できない疾患（感度の限界）
- 検査値の乖離が診断の手がかりとなるパターン
- 省略しない。知っている鑑別パターンを全部書く""",

    # Section 5: 偽陽性・偽陰性・限界
    """\
あなたは日本の臨床医学に精通した専門家です。
与えられた検査について、「偽陽性・偽陰性・限界」のみを詳細に記述してください。
出力はプレーンテキストのみ（JSON不要、マークダウン不要、コードブロック不要、見出し記号不要）。

この検査結果が信頼できない全ての状況を網羅的に記述する。

記述内容:
- 偽陽性となる条件（疾患とは無関係に異常値を示す全てのケース）
  - 薬剤による影響（具体的薬剤名と影響の方向）
  - 生理的変動（運動後、食後、日内変動、月経周期、妊娠等）
  - 検体の問題（溶血、乳び、ビリルビン干渉、凝固、保存条件等）
- 偽陰性となる条件（疾患があるのに正常値を示す全てのケース）
  - 時期の問題（発症早期、治療開始後等）
  - 患者因子（免疫不全、高齢者、ステロイド使用中等）
  - 技術的限界
- この検査単独では診断できない状況
- 検査値が「不適切な正常」を示すパラドックス（例: 重症でも正常値）
- 省略しない。臨床的に重要な落とし穴を全部書く""",

    # Section 6: 緊急値・時間軸・モニタリング
    """\
あなたは日本の臨床医学に精通した専門家です。
与えられた検査について、「緊急値・時間軸・治療モニタリング」のみを詳細に記述してください。
出力はプレーンテキストのみ（JSON不要、マークダウン不要、コードブロック不要、見出し記号不要）。

記述内容:
- パニック値・緊急報告値（具体的な閾値と、その値が示唆する緊急病態、必要な即時対応）
- 発症からの時間経過と検査値推移
  - 発症後何時間で異常が出現するか
  - ピーク到達までの時間
  - 正常化までの時間
  - 疾患ごとの典型的な時間経過パターン
- 治療効果判定の指標としての使い方
  - 治療開始後、何時間/何日で改善が期待されるか
  - 改善が見られない場合の解釈
  - 再上昇・再燃のサイン
- 予後予測への寄与（この検査値がどの程度の値だと予後不良か）
- 連続測定（トレンド）の解釈法
- 省略しない。時間軸に関する知識を全部書く""",
]

TEST_SECTION_TITLES = [
    "適応臨床像",
    "疾患別異常パターン（主要群）",
    "疾患別異常パターン（追加群）",
    "鑑別パターン",
    "偽陽性・偽陰性・限界",
    "緊急値・時間軸・モニタリング",
]


# ----------------------------------------------------------------
# Gemini 3パス方式プロンプト（2セクション/パス、自然出力長を利用）
# ----------------------------------------------------------------

GEMINI_3PASS_PROMPTS = [
    # Pass 1: 好発背景 + 典型来院像
    """\
あなたは日本の臨床医学に精通した専門家です。
与えられた疾患について、以下の2つの項目を詳細に記述してください。
出力はプレーンテキストのみ（JSON不要、マークダウン不要、コードブロック不要）。
「〜を認める」と断定的に記述し、頻度・程度は括弧で補足する。省略しない。

## 1. 好発背景・リスク因子・誘因
- 好発年齢・性別（男女比、年齢分布のピーク）
- 基礎疾患（糖尿病、CKD、肝硬変、HIV、移植後、悪性腫瘍等）
- 生活歴（飲酒、喫煙、IV薬物使用、職業曝露、食生活等）
- 薬剤歴（ステロイド、免疫抑制薬、OCP、抗凝固薬、NSAIDs等）
- 直近のイベント（手術、長期臥床、歯科処置、旅行、動物接触等）
- 家族歴・遺伝的素因・環境因子

## 2. 典型来院像
- 主訴と発症様式（突然/緩徐/段階的）
- 症状の時間経過パターン（移動、増悪、間欠、crescendo等）
- 増悪因子・緩解因子（体位、呼吸、食事、運動等）
- 随伴症状とその出現頻度
- この疾患に特徴的な訴え方・表現
- 病型・亜型がある場合はそれぞれの典型像""",

    # Pass 2: 非典型来院像 + バイタル・身体所見
    """\
あなたは日本の臨床医学に精通した専門家です。
与えられた疾患について、以下の2つの項目を詳細に記述してください。
出力はプレーンテキストのみ（JSON不要、マークダウン不要、コードブロック不要）。
「〜を認める」と断定的に記述する。見逃しは患者の死に直結する。全パターンを書く。

## 3. 非典型来院像・ピットフォール
- 高齢者での異なる初発症状
- 糖尿病患者での症状修飾（自律神経障害による無痛性等）
- 免疫不全・ステロイド使用下での症状マスク（発熱欠如、CRP抑制等）
- 小児での特異的な表現
- 薬剤による所見修飾（β遮断薬→頻脈マスク、抗凝固薬→出血増加等）
- 無症候性・症状軽微な場合
- 他疾患と誤認されやすいプレゼンテーション
- 病型・亜型による来院像の違い

## 4. バイタルサイン・身体所見
- バイタルサイン: 体温、脈拍、血圧、呼吸数、SpO2の具体的数値と方向
  - 病型・亜型・重症度による全バリエーション（軽症〜重症のスペクトラム）
- 視診・触診・打診・聴診の具体的所見
- 特殊手技・誘発テスト（該当するものすべて）と感度・特異度
- 出現頻度の高いものから順に""",

    # Pass 3: 検査所見 + 鑑別キー
    """\
あなたは日本の臨床医学に精通した専門家です。
与えられた疾患について、以下の2つの項目を詳細に記述してください。
出力はプレーンテキストのみ（JSON不要、マークダウン不要、コードブロック不要）。
検査名は正式名称を使用。具体的な数値の目安を含める。

## 5. 検査所見パターン
- 血液検査: 具体的な検査名と異常の方向・程度、数値の目安
- 尿検査・髄液検査・その他体液検査
- 画像検査: X線、CT、MRI、エコー、血管造影等の具体的異常所見
- 生理検査: 心電図、脳波、肺機能、神経伝導等の異常パターン
- 発症からの時期による検査値の変化
- 重症度・病型・亜型による検査パターンの違い

## 6. 鑑別キー
- 主要な鑑別疾患それぞれに対して「Aを示唆しBを否定する所見」を具体的に列挙
- 検査値の程度差（軽度上昇 vs 著明上昇）による鑑別
- 時間経過による鑑別（急性 vs 亜急性 vs 慢性）
- 所見の組み合わせパターンによる鑑別
- この疾患の確定診断に最も重要な所見""",
]


async def _claude_stream_call(client, semaphore, messages, label=""):
    """Claude streaming API call with retry on 429/500/503"""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            async with semaphore:
                stream = await client.chat.completions.create(
                    model=CLAUDE_MODEL,
                    messages=messages,
                    temperature=1.0,
                    max_tokens=65536,
                    stream=True,
                )
                text = ""
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        text += chunk.choices[0].delta.content
                return text.strip()
        except Exception as e:
            err_str = str(e)
            retryable = any(code in err_str for code in ["429", "500", "503", "502"])
            if retryable and attempt < max_retries - 1:
                wait = (attempt + 1) * 30
                print(f"    {label} Retry {attempt+1}/{max_retries}, waiting {wait}s...")
                await asyncio.sleep(wait)
                continue
            raise


def _is_truncated(text):
    """Check if text appears truncated"""
    t = text.rstrip()
    return not t.endswith(("。", "）", ")", "」", "）", "％", "%"))


def _find_section_start(text, section_num):
    """Find the start position of a numbered section"""
    import re as _re
    for pat in [rf"\n{section_num}\.\s", rf"\n### {section_num}\."]:
        m = _re.search(pat, text)
        if m:
            return m.start()
    return -1


async def _gen_findings_one_claude(client, semaphore, name, category, system_prompt, index, total):
    """1エントリの findings_description を生成（Claude用・6パス並列方式）
    6セクションを並列生成し結合。各リクエストが短いため安定性が高い。"""
    label = f"[{index + 1}/{total}] {name}"

    async def gen_section(sec_idx):
        sec_label = f"{label} S{sec_idx + 1}"
        try:
            messages = [
                {"role": "system", "content": DISEASE_SECTION_PROMPTS[sec_idx]},
                {"role": "user", "content": f"{name}（{category}）"},
            ]
            text = await _claude_stream_call(client, semaphore, messages, sec_label)
            if text:
                text = re.sub(r"```.*?\n?", "", text).strip()
                return sec_idx, text
            return sec_idx, None
        except Exception as e:
            print(f"    {sec_label} ... ERROR: {e}")
            return sec_idx, None

    # 6セクションを並列で生成
    results = await asyncio.gather(*[gen_section(i) for i in range(6)])

    # 結合（セクション番号順）
    sections = [""] * 6
    failed = []
    for sec_idx, text in results:
        if text:
            sections[sec_idx] = text
        else:
            failed.append(sec_idx + 1)

    combined = "\n\n".join(s for s in sections if s)
    if not combined:
        print(f"  {label} ... ERROR: all sections failed")
        return index, None

    suffix = ""
    if failed:
        suffix = f", failed S{failed}"
    print(f"  {label} ... OK ({len(combined)}字, 6-pass{suffix})")
    return index, combined


async def _gen_findings_one_lemon_6pass(client, semaphore, name, category, system_prompt, index, total):
    """1エントリの findings_description を生成（Lemon API 6パス並列 — 疾患用プロンプト）"""
    label = f"[{index + 1}/{total}] {name}"

    async def gen_section(sec_idx):
        sec_label = f"{label} S{sec_idx + 1}"
        max_retries = 5
        for attempt in range(max_retries):
            try:
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": DISEASE_SECTION_PROMPTS[sec_idx]},
                            {"role": "user", "content": f"{name}（{category}）"},
                        ],
                        temperature=1.0,
                        max_tokens=65536,
                    )
                    text = response.choices[0].message.content or ""
                    text = text.strip()
                    text = re.sub(r"```.*?\n?", "", text).strip()
                    text = re.sub(r"^#{1,4}\s+", "", text, flags=re.MULTILINE)
                    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
                    return sec_idx, text if text else None
            except Exception as e:
                err_str = str(e)
                retryable = any(code in err_str for code in ["429", "500", "503", "502", "rate"])
                if retryable and attempt < max_retries - 1:
                    wait = (attempt + 1) * 15
                    print(f"    {sec_label} Retry {attempt+1}/{max_retries}, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                print(f"    {sec_label} ... ERROR: {e}")
                return sec_idx, None
        return sec_idx, None

    results = await asyncio.gather(*[gen_section(i) for i in range(6)])

    sections = [""] * 6
    failed = []
    for sec_idx, text in results:
        if text:
            sections[sec_idx] = text
        else:
            failed.append(sec_idx + 1)

    combined = "\n\n".join(s for s in sections if s)
    if not combined:
        print(f"  {label} ... ERROR: all sections failed")
        return index, None

    suffix = f", failed S{failed}" if failed else ""
    print(f"  {label} ... OK ({len(combined)}字, lemon-6-pass{suffix})")
    return index, combined


async def _gen_findings_one_vertex_6pass(vertex_client, semaphore, name, category, system_prompt, index, total):
    """1エントリの findings_description を生成（Vertex AI gemini-3-pro-preview 6パス並列方式）"""
    from google.genai.types import GenerateContentConfig
    label = f"[{index + 1}/{total}] {name}"

    async def gen_section(sec_idx):
        sec_label = f"{label} S{sec_idx + 1}"
        max_retries = 5
        for attempt in range(max_retries):
            try:
                async with semaphore:
                    response = await vertex_client.aio.models.generate_content(
                        model=VERTEX_MODEL,
                        contents=f"{name}（{category}）",
                        config=GenerateContentConfig(
                            system_instruction=DISEASE_SECTION_PROMPTS[sec_idx],
                            temperature=1.0,
                            max_output_tokens=65536,
                        ),
                    )
                    text = response.text if response.text else ""
                    text = text.strip()
                    text = re.sub(r"```.*?\n?", "", text).strip()
                    text = re.sub(r"^#{1,4}\s+", "", text, flags=re.MULTILINE)
                    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
                    return sec_idx, text if text else None
            except Exception as e:
                err_str = str(e)
                retryable = any(code in err_str for code in ["429", "500", "503", "502", "RESOURCE_EXHAUSTED"])
                if retryable and attempt < max_retries - 1:
                    wait = (attempt + 1) * 15
                    print(f"    {sec_label} Retry {attempt+1}/{max_retries}, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                print(f"    {sec_label} ... ERROR: {e}")
                return sec_idx, None
        return sec_idx, None

    results = await asyncio.gather(*[gen_section(i) for i in range(6)])

    sections = [""] * 6
    failed = []
    for sec_idx, text in results:
        if text:
            sections[sec_idx] = text
        else:
            failed.append(sec_idx + 1)

    combined = "\n\n".join(s for s in sections if s)
    if not combined:
        print(f"  {label} ... ERROR: all sections failed")
        return index, None

    suffix = f", failed S{failed}" if failed else ""
    print(f"  {label} ... OK ({len(combined)}字, 6-pass{suffix})")
    return index, combined


async def _gen_test_findings_one_vertex_6pass(vertex_client, semaphore, name, category, system_prompt, index, total):
    """1検査の findings_description を生成（Vertex AI 6パス並列 — 検査専用プロンプト）"""
    from google.genai.types import GenerateContentConfig
    label = f"[{index + 1}/{total}] {name}"

    async def gen_section(sec_idx):
        sec_label = f"{label} S{sec_idx + 1}"
        max_retries = 5
        for attempt in range(max_retries):
            try:
                async with semaphore:
                    response = await vertex_client.aio.models.generate_content(
                        model=VERTEX_MODEL,
                        contents=f"{name}（{category}）",
                        config=GenerateContentConfig(
                            system_instruction=TEST_SECTION_PROMPTS[sec_idx],
                            temperature=1.0,
                            max_output_tokens=65536,
                        ),
                    )
                    text = response.text if response.text else ""
                    text = text.strip()
                    text = re.sub(r"```.*?\n?", "", text).strip()
                    text = re.sub(r"^#{1,4}\s+", "", text, flags=re.MULTILINE)
                    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
                    return sec_idx, text if text else None
            except Exception as e:
                err_str = str(e)
                retryable = any(code in err_str for code in ["429", "500", "503", "502", "RESOURCE_EXHAUSTED"])
                if retryable and attempt < max_retries - 1:
                    wait = (attempt + 1) * 15
                    print(f"    {sec_label} Retry {attempt+1}/{max_retries}, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                print(f"    {sec_label} ... ERROR: {e}")
                return sec_idx, None
        return sec_idx, None

    results = await asyncio.gather(*[gen_section(i) for i in range(6)])

    sections = [""] * 6
    failed = []
    for sec_idx, text in results:
        if text:
            sections[sec_idx] = text
        else:
            failed.append(sec_idx + 1)

    combined = "\n\n".join(s for s in sections if s)
    if not combined:
        print(f"  {label} ... ERROR: all sections failed")
        return index, None

    suffix = f", failed S{failed}" if failed else ""
    print(f"  {label} ... OK ({len(combined)}字, test-6-pass{suffix})")
    return index, combined


async def _gen_test_findings_one_lemon_6pass(client, semaphore, name, category, system_prompt, index, total):
    """1検査の findings_description を生成（Lemon API 6パス並列 — 検査専用プロンプト）"""
    label = f"[{index + 1}/{total}] {name}"

    async def gen_section(sec_idx):
        sec_label = f"{label} S{sec_idx + 1}"
        max_retries = 5
        for attempt in range(max_retries):
            try:
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": TEST_SECTION_PROMPTS[sec_idx]},
                            {"role": "user", "content": f"{name}（{category}）"},
                        ],
                        temperature=1.0,
                        max_tokens=65536,
                    )
                    text = response.choices[0].message.content or ""
                    text = text.strip()
                    text = re.sub(r"```.*?\n?", "", text).strip()
                    text = re.sub(r"^#{1,4}\s+", "", text, flags=re.MULTILINE)
                    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
                    return sec_idx, text if text else None
            except Exception as e:
                err_str = str(e)
                retryable = any(code in err_str for code in ["429", "500", "503", "502", "rate"])
                if retryable and attempt < max_retries - 1:
                    wait = (attempt + 1) * 15
                    print(f"    {sec_label} Retry {attempt+1}/{max_retries}, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                print(f"    {sec_label} ... ERROR: {e}")
                return sec_idx, None
        return sec_idx, None

    results = await asyncio.gather(*[gen_section(i) for i in range(6)])

    sections = [""] * 6
    failed = []
    for sec_idx, text in results:
        if text:
            sections[sec_idx] = text
        else:
            failed.append(sec_idx + 1)

    combined = "\n\n".join(s for s in sections if s)
    if not combined:
        print(f"  {label} ... ERROR: all sections failed")
        return index, None

    suffix = f", failed S{failed}" if failed else ""
    print(f"  {label} ... OK ({len(combined)}字, lemon-6-pass{suffix})")
    return index, combined


async def _gen_findings_one_gemini_3pass(client, semaphore, name, category, system_prompt, index, total):
    """1エントリの findings_description を生成（Gemini 3パス並列方式）
    3パス×2セクション = 6セクション。各パスはGeminiの自然出力長（~4000トークン）を利用。"""
    label = f"[{index + 1}/{total}] {name}"

    async def gen_pass(pass_idx):
        pass_label = f"{label} P{pass_idx + 1}"
        max_retries = 5
        for attempt in range(max_retries):
            try:
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": GEMINI_3PASS_PROMPTS[pass_idx]},
                            {"role": "user", "content": f"{name}（{category}）"},
                        ],
                        temperature=0.2,
                        max_tokens=65536,
                    )
                    text = response.choices[0].message.content.strip()
                    text = re.sub(r"```.*?\n?", "", text).strip()
                    return pass_idx, text
            except Exception as e:
                err_str = str(e)
                retryable = any(code in err_str for code in ["429", "500", "503", "502"])
                if retryable and attempt < max_retries - 1:
                    wait = (attempt + 1) * 15
                    print(f"    {pass_label} Retry {attempt+1}/{max_retries}, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                print(f"    {pass_label} ... ERROR: {e}")
                return pass_idx, None
        return pass_idx, None

    # 3パスを並列実行
    results = await asyncio.gather(*[gen_pass(i) for i in range(3)])

    # 結合（パス番号順）
    passes = [""] * 3
    failed = []
    for pass_idx, text in results:
        if text:
            passes[pass_idx] = text
        else:
            failed.append(pass_idx + 1)

    combined = "\n\n".join(p for p in passes if p)
    if not combined:
        print(f"  {label} ... ERROR: all passes failed")
        return index, None

    suffix = f", failed P{failed}" if failed else ""
    print(f"  {label} ... OK ({len(combined)}字, 3-pass{suffix})")
    return index, combined


async def _generate_disease_findings(client, semaphore, dry_run, force=False, use_claude=False, use_gemini_3pass=False, use_vertex=False):
    """疾患の findings_description を一括生成
    --force: 全件再生成
    --vertex: Vertex AI gemini-3-pro-preview 6パス方式
    中間保存: 10件完了ごとにJSONLを書き出し"""
    diseases = read_jsonl(DISEASES_JSONL)

    if force:
        remaining = [(i, d) for i, d in enumerate(diseases)]
        print(f"疾患: {len(diseases)}件（全件再生成）")
    else:
        remaining = [(i, d) for i, d in enumerate(diseases) if not d.get("findings_description")]
        print(f"疾患: {len(diseases)}件中 {len(remaining)}件が未生成")

    if not remaining or dry_run:
        return

    if use_vertex:
        gen_func = _gen_findings_one_vertex_6pass
        model_label = f"Vertex AI {VERTEX_MODEL} 6-pass"
    elif use_gemini_3pass:
        gen_func = _gen_findings_one_gemini_3pass
        model_label = "Gemini Pro 3-pass"
    elif use_claude:
        gen_func = _gen_findings_one_claude
        model_label = "Claude Opus 4.6"
    else:
        gen_func = _gen_findings_one_lemon_6pass
        model_label = f"Lemon {LLM_MODEL} 6-pass"
    print(f"モデル: {model_label} / 並行数: {semaphore._value}")

    save_lock = asyncio.Lock()
    completed_count = [0]
    error_count = [0]

    async def gen_and_save(orig_idx, d, task_idx):
        idx, text = await gen_func(
            client, semaphore,
            d["disease_name"], d.get("category", ""),
            DISEASE_FINDINGS_V2_PROMPT, task_idx, len(remaining),
        )
        if text:
            async with save_lock:
                diseases[orig_idx]["findings_description"] = text
                completed_count[0] += 1
                if completed_count[0] % 10 == 0:
                    write_jsonl(DISEASES_JSONL, diseases)
                    print(f"  === 中間保存 {completed_count[0]}/{len(remaining)}件 (エラー{error_count[0]}件) ===")
        else:
            error_count[0] += 1
        return text is not None

    tasks = [
        gen_and_save(orig_idx, d, idx)
        for idx, (orig_idx, d) in enumerate(remaining)
    ]
    results = await asyncio.gather(*tasks)

    write_jsonl(DISEASES_JSONL, diseases)
    success = sum(1 for r in results if r)
    print(f"疾患 findings_description: {success}/{len(remaining)}件 生成完了 (エラー{error_count[0]}件)")


async def _generate_test_findings(client, semaphore, dry_run, force=False, use_vertex=False):
    """検査の findings_description を一括生成。
    use_vertex=True: Vertex AI 6パス並列（TEST_SECTION_PROMPTS使用）
    force=True: 既存のfindings_descriptionをfindings_description_v1にバックアップして上書き
    中間保存: 10件完了ごとにJSONLを書き出し"""
    tests = read_jsonl(TESTS_JSONL)

    if force:
        remaining = [(i, t) for i, t in enumerate(tests)]
        print(f"検査: {len(tests)}件（全件再生成）")
        # 既存のfindings_descriptionをv1としてバックアップ
        for _, t in remaining:
            if t.get("findings_description") and not t.get("findings_description_v1"):
                t["findings_description_v1"] = t["findings_description"]
        write_jsonl(TESTS_JSONL, tests)
        print(f"  既存findings_descriptionをfindings_description_v1にバックアップ完了")
    else:
        remaining = [(i, t) for i, t in enumerate(tests) if not t.get("findings_description")]
        print(f"検査: {len(tests)}件中 {len(remaining)}件が未生成")

    if not remaining or dry_run:
        return

    if use_vertex:
        gen_func = _gen_test_findings_one_vertex_6pass
        model_label = f"Vertex AI {VERTEX_MODEL} test-6-pass"
    else:
        gen_func = _gen_test_findings_one_lemon_6pass
        model_label = f"Lemon API {LLM_MODEL} test-6-pass"
    print(f"モデル: {model_label} / 並行数: {semaphore._value}")

    save_lock = asyncio.Lock()
    completed_count = [0]
    error_count = [0]

    async def gen_and_save(orig_idx, t, task_idx):
        idx, text = await gen_func(
            client, semaphore,
            t["test_name"], t.get("category", ""),
            TEST_FINDINGS_PROMPT, task_idx, len(remaining),
        )
        if text:
            async with save_lock:
                tests[orig_idx]["findings_description"] = text
                completed_count[0] += 1
                if completed_count[0] % 10 == 0:
                    write_jsonl(TESTS_JSONL, tests)
                    print(f"  === 中間保存 {completed_count[0]}/{len(remaining)}件 (エラー{error_count[0]}件) ===")
        else:
            error_count[0] += 1
        return text is not None

    tasks = [
        gen_and_save(orig_idx, t, idx)
        for idx, (orig_idx, t) in enumerate(remaining)
    ]
    results = await asyncio.gather(*tasks)

    write_jsonl(TESTS_JSONL, tests)
    success = sum(1 for r in results if r)
    print(f"検査 findings_description: {success}/{len(remaining)}件 生成完了 (エラー{error_count[0]}件)")


async def cmd_findings_desc_async(args):
    """疾患・検査の findings_description を並行生成（同一所見空間）"""
    force = getattr(args, 'force', False)
    use_claude = getattr(args, 'claude', False)
    use_gemini_3pass = getattr(args, 'gemini3pass', False)
    use_vertex = getattr(args, 'vertex', False)
    target = args.target

    # Vertex AIクライアントは共有（diseases/tests両方で使う可能性）
    vertex_client = None
    if use_vertex:
        from google import genai
        vertex_client = genai.Client(
            project=VERTEX_PROJECT,
            location=VERTEX_LOCATION,
            vertexai=True,
        )
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = VERTEX_SA_KEY

    coros = []
    if target in ("diseases", "all"):
        if use_vertex:
            disease_sem = asyncio.Semaphore(args.concurrency)
            coros.append(_generate_disease_findings(vertex_client, disease_sem, args.dry_run, force=force, use_vertex=True))
        elif use_claude:
            disease_client = AsyncOpenAI(api_key=CLAUDE_API_KEY, base_url=CLAUDE_BASE_URL)
            disease_sem = asyncio.Semaphore(args.concurrency)
            coros.append(_generate_disease_findings(disease_client, disease_sem, args.dry_run, force=force, use_claude=True))
        else:
            disease_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
            disease_sem = asyncio.Semaphore(args.concurrency)
            coros.append(_generate_disease_findings(disease_client, disease_sem, args.dry_run, force=force, use_gemini_3pass=use_gemini_3pass))

    if target in ("tests", "all"):
        if use_vertex:
            test_sem = asyncio.Semaphore(args.concurrency)
            coros.append(_generate_test_findings(vertex_client, test_sem, args.dry_run, force=force, use_vertex=True))
        else:
            test_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
            test_sem = asyncio.Semaphore(args.concurrency)
            coros.append(_generate_test_findings(test_client, test_sem, args.dry_run))

    await asyncio.gather(*coros)


# ----------------------------------------------------------------
# 検査quality_description生成（総合的な質の記述）
# ----------------------------------------------------------------

async def _generate_test_quality(client, semaphore, dry_run):
    """検査の quality_description を一括生成"""
    tests = read_jsonl(TESTS_JSONL)
    remaining = [(i, t) for i, t in enumerate(tests) if not t.get("quality_description")]
    print(f"検査quality: {len(tests)}件中 {len(remaining)}件が未生成")

    if not remaining or dry_run:
        return

    tasks = [
        _gen_findings_one(
            client, semaphore,
            t["test_name"], t.get("category", ""),
            TEST_QUALITY_PROMPT, idx, len(remaining),
        )
        for idx, (_, t) in enumerate(remaining)
    ]
    results = await asyncio.gather(*tasks)
    for idx, text in results:
        if text:
            orig_idx = remaining[idx][0]
            tests[orig_idx]["quality_description"] = text
    write_jsonl(TESTS_JSONL, tests)
    success = sum(1 for _, t in results if t)
    print(f"検査 quality_description: {success}/{len(remaining)}件 生成完了")


async def cmd_quality_async(args):
    """検査の quality_description を並行生成"""
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    semaphore = asyncio.Semaphore(args.concurrency)
    await _generate_test_quality(client, semaphore, args.dry_run)


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

    p_fdesc = subparsers.add_parser("findings-desc", help="所見ベースdescription生成（疾患・検査）")
    p_fdesc.add_argument("--target", choices=["diseases", "tests", "all"], default="all",
                         help="生成対象（デフォルト: all）")
    p_fdesc.add_argument("--dry-run", action="store_true",
                         help="生成対象を表示するだけで実行しない")
    p_fdesc.add_argument("--force", action="store_true",
                         help="既存のfindings_descriptionを上書き再生成")
    p_fdesc.add_argument("--claude", action="store_true",
                         help="Claude Opus 4.6 APIを使用（高品質・ストリーミング）")
    p_fdesc.add_argument("--gemini3pass", action="store_true",
                         help="Gemini Pro 3パス方式（embedding最適化、~10000字）")
    p_fdesc.add_argument("--vertex", action="store_true",
                         help="Vertex AI gemini-3-pro-preview 6パス方式")
    p_fdesc.add_argument("--concurrency", type=int, default=50,
                         help="最大並行数（デフォルト: 50、Claude使用時は10推奨）")

    p_quality = subparsers.add_parser("quality", help="検査quality_description生成")
    p_quality.add_argument("--dry-run", action="store_true",
                           help="生成対象を表示するだけで実行しない")
    p_quality.add_argument("--concurrency", type=int, default=30,
                           help="最大並行数（デフォルト: 30）")

    args = parser.parse_args()

    if args.command == "diseases":
        asyncio.run(cmd_diseases_async(args))
    elif args.command == "tests":
        asyncio.run(cmd_tests_async(args))
    elif args.command == "findings":
        asyncio.run(cmd_findings_async(args))
    elif args.command == "findings-desc":
        asyncio.run(cmd_findings_desc_async(args))
    elif args.command == "quality":
        asyncio.run(cmd_quality_async(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
