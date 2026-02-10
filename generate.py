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
# 所見ベースdescription生成（疾患・検査を同一所見空間に配置）
# ----------------------------------------------------------------

DISEASE_FINDINGS_PROMPT = """\
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
    """1エントリの findings_description を生成"""
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


async def _generate_disease_findings(client, semaphore, dry_run):
    """疾患の findings_description を一括生成"""
    diseases = read_jsonl(DISEASES_JSONL)
    remaining = [(i, d) for i, d in enumerate(diseases) if not d.get("findings_description")]
    print(f"疾患: {len(diseases)}件中 {len(remaining)}件が未生成")

    if not remaining or dry_run:
        return

    tasks = [
        _gen_findings_one(
            client, semaphore,
            d["disease_name"], d.get("category", ""),
            DISEASE_FINDINGS_PROMPT, idx, len(remaining),
        )
        for idx, (_, d) in enumerate(remaining)
    ]
    results = await asyncio.gather(*tasks)
    for idx, text in results:
        if text:
            orig_idx = remaining[idx][0]
            diseases[orig_idx]["findings_description"] = text
    write_jsonl(DISEASES_JSONL, diseases)
    success = sum(1 for _, t in results if t)
    print(f"疾患 findings_description: {success}/{len(remaining)}件 生成完了")


async def _generate_test_findings(client, semaphore, dry_run):
    """検査の findings_description を一括生成"""
    tests = read_jsonl(TESTS_JSONL)
    remaining = [(i, t) for i, t in enumerate(tests) if not t.get("findings_description")]
    print(f"検査: {len(tests)}件中 {len(remaining)}件が未生成")

    if not remaining or dry_run:
        return

    tasks = [
        _gen_findings_one(
            client, semaphore,
            t["test_name"], t.get("category", ""),
            TEST_FINDINGS_PROMPT, idx, len(remaining),
        )
        for idx, (_, t) in enumerate(remaining)
    ]
    results = await asyncio.gather(*tasks)
    for idx, text in results:
        if text:
            orig_idx = remaining[idx][0]
            tests[orig_idx]["findings_description"] = text
    write_jsonl(TESTS_JSONL, tests)
    success = sum(1 for _, t in results if t)
    print(f"検査 findings_description: {success}/{len(remaining)}件 生成完了")


async def cmd_findings_desc_async(args):
    """疾患・検査の findings_description を並行生成（同一所見空間）"""
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    semaphore = asyncio.Semaphore(args.concurrency)

    target = args.target
    coros = []
    if target in ("diseases", "all"):
        coros.append(_generate_disease_findings(client, semaphore, args.dry_run))
    if target in ("tests", "all"):
        coros.append(_generate_test_findings(client, semaphore, args.dry_run))

    # 疾患と検査を同時並行で生成
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
    p_fdesc.add_argument("--concurrency", type=int, default=50,
                         help="最大並行数（デフォルト: 50）")

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
