"""
疾患・検査のfindings_descriptionを一括改善するスクリプト

改善内容（所見即所得の原則に準拠）:
1. 頻度情報の追加: 各所見に出現頻度を付加
2. 重大性の所見化: 統計や判断ではなく、観察可能な身体変化として記述
3. 治療効果の所見化: 治療後に観察される検査値・所見の変化として記述
4. 検出困難条件: 偽陰性・偽陽性になる条件の追記
"""
import json
import asyncio
import aiohttp
import sys
import os
import time
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    LLM_FALLBACK_API_KEY, LLM_FALLBACK_BASE_URL, LLM_FALLBACK_MODEL,
    DATA_DIR,
)

# ================================================================
# プロンプト（疾患用）
# ================================================================
DISEASE_SYSTEM_PROMPT = """あなたは臨床医学の専門家です。疾患のfindings_description（所見記述）を改善してください。

■ 所見即所得の原則
このテキストはembedding検索で「疾患の所見」と「検査の所見」の類似度計算に使われます。
したがって、すべての記述は「観察・測定できる所見」でなければなりません。

■ 改善ルール

1. 頻度情報の追加（最優先）
   各所見に出現頻度を付加してください。教科書・ガイドラインの一般的な値を参考に。
   - 具体的な数値: 「胸痛（約90%に出現）」
   - 定性的表現: 「（ほぼ全例）」「（約半数）」「（10-20%）」「（稀、5%未満）」
   - 不明な場合は臨床経験に基づく合理的な推定でよい

2. 重大性を所見として記述
   ❌ 「致死率30%」「緊急手術を要する」「予後不良」
   ✅ 「未治療では数時間以内に血圧低下・意識障害が進行し、乏尿・肝酵素上昇・凝固異常が出現する」
   → 死や悪化を、観察可能な身体所見・検査値の変化として記述

3. 治療効果を観察可能な変化として記述
   ❌ 「抗菌薬で改善する」「手術で治癒する」
   ✅ 「抗菌薬投与後24-48時間でCRP低下・解熱が観察され、白血球正常化する」
   → 治療による検査値・バイタルサイン・症状の具体的変化を記述

4. 既存の良い記述は保持
   元のテキストの正確な所見記述はそのまま残し、上記を追記・補強する

■ 出力形式
改善後のfindings_descriptionテキストのみを出力。説明や前置きは不要。
文字数は1200-1800字程度（元のテキストと同程度）。"""

DISEASE_USER_TEMPLATE = """以下の疾患のfindings_descriptionを改善してください。

疾患名: {disease_name}

現在のfindings_description:
{findings_description}

上記を改善ルールに従って補強してください。改善後のテキストのみ出力。"""

# ================================================================
# プロンプト（検査用）
# ================================================================
TEST_SYSTEM_PROMPT = """あなたは臨床検査の専門家です。検査のfindings_descriptionを全面的に書き直してください。

■ 目的
このテキストはembedding検索で「疾患の所見」と「検査の所見」の類似度計算に使われます。
つまり、疾患のfindings_descriptionとの語彙・概念の重複が多いほど、類似度が高くなり、
その検査がその疾患に推薦されやすくなります。

■ 書き方のルール（最重要）

1. 「どんな臨床像の患者にこの検査をするか」を最初に書く（最優先）
   ✅ 「高熱・悪寒・ショック所見（血圧低下・頻脈・末梢冷感）を呈する患者、
      感染性心内膜炎を疑う新規心雑音と発熱の患者、カテーテル関連感染疑い、
      原因不明の発熱が持続する免疫不全患者に実施する。」
   ❌ 「菌血症の確定および起炎菌の特定に用いる。」

2. 疾患名と具体的な検査値を結びつけて記述する
   ✅ 「敗血症性ショックの患者では血液培養が80%以上で陽性となり、
      黄色ブドウ球菌が検出された場合は感染性心内膜炎の合併を示唆する。」
   ❌ 「黄色ブドウ球菌が検出された場合は有意である。」

3. 各所見に出現頻度を付加する
   ✅ 「肺炎球菌性肺炎では約60%で血液培養陽性」
   ❌ 「肺炎で陽性になることがある」

4. 値の階層別カタログ（軽度上昇/中等度上昇/高度上昇）は最小限にする
   ❌ 「0.3-1.0: ウイルス性、1.0-10.0: 細菌性、10.0以上: 重症」
   → これは教科書の参照表であり、臨床像と結びついていない

5. 治療反応を観察可能な変化として記述する
   ✅ 「適切な抗菌薬投与後48-72時間で培養陰性化が確認される」

■ 出力形式
書き直したfindings_descriptionテキストのみを出力。説明や前置きは不要。
文字数は1200-1800字程度。"""

TEST_USER_TEMPLATE = """以下の検査のfindings_descriptionを全面的に書き直してください。

検査名: {test_name}
カテゴリ: {category}

現在のfindings_description:
{findings_description}

上記の改善ルールに従って、全面的に書き直してください。書き直した後のテキストのみ出力。"""


# ================================================================
# LLM呼び出し
# ================================================================
async def llm_call(session, system_prompt, user_prompt, api_key, base_url, model, max_tokens=4096):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with session.post(
            f"{base_url}/chat/completions",
            json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data['choices'][0]['message']['content'].strip()
            elif resp.status == 429:
                await asyncio.sleep(3)
                return None
            else:
                return None
    except Exception:
        return None


async def call_with_fallback(session, system_prompt, user_prompt):
    """プライマリ3回→フォールバック3回"""
    for attempt in range(3):
        result = await llm_call(session, system_prompt, user_prompt,
                                LLM_API_KEY, LLM_BASE_URL, LLM_MODEL)
        if result:
            return result
        await asyncio.sleep(1 + attempt)

    for attempt in range(3):
        result = await llm_call(session, system_prompt, user_prompt,
                                LLM_FALLBACK_API_KEY, LLM_FALLBACK_BASE_URL, LLM_FALLBACK_MODEL)
        if result:
            return result
        await asyncio.sleep(1 + attempt)
    return None


# ================================================================
# メイン処理
# ================================================================
async def improve_diseases(session, diseases, sem, progress):
    """疾患のfindings_descriptionを改善"""
    results = {}

    async def process_one(i, d):
        async with sem:
            name = d['disease_name']
            fd = d.get('findings_description', '')
            if not fd or len(fd) < 100:
                return

            user_prompt = DISEASE_USER_TEMPLATE.format(
                disease_name=name,
                findings_description=fd,
            )
            result = await call_with_fallback(session, DISEASE_SYSTEM_PROMPT, user_prompt)
            if result and len(result) > 500:
                results[i] = result
                progress['done'] += 1
                if progress['done'] % 20 == 0:
                    print(f"  疾患: {progress['done']}/{progress['total']} 完了")
            else:
                progress['fail'] += 1
                print(f"  失敗: {name}")

    tasks = [process_one(i, d) for i, d in enumerate(diseases)]
    await asyncio.gather(*tasks)
    return results


async def improve_tests(session, tests, target_indices, sem, progress):
    """検査のfindings_descriptionを改善"""
    results = {}

    async def process_one(i):
        async with sem:
            t = tests[i]
            name = t['test_name']
            fd = t.get('findings_description', '')
            if not fd or len(fd) < 100:
                return

            user_prompt = TEST_USER_TEMPLATE.format(
                test_name=name,
                category=t.get('category', ''),
                findings_description=fd,
            )
            result = await call_with_fallback(session, TEST_SYSTEM_PROMPT, user_prompt)
            if result and len(result) > 500:
                results[i] = result
                progress['done'] += 1
                if progress['done'] % 10 == 0:
                    print(f"  検査: {progress['done']}/{progress['total']} 完了")
            else:
                progress['fail'] += 1
                print(f"  失敗: {name}")

    tasks = [process_one(i) for i in target_indices]
    await asyncio.gather(*tasks)
    return results


async def main():
    import re

    # データ読み込み
    diseases = []
    with open(os.path.join(DATA_DIR, "diseases.jsonl"), 'r', encoding='utf-8') as f:
        for line in f:
            diseases.append(json.loads(line))

    tests = []
    with open(os.path.join(DATA_DIR, "tests.jsonl"), 'r', encoding='utf-8') as f:
        for line in f:
            tests.append(json.loads(line))

    # 全検査を対象（findings_descriptionがあるもの全て）
    test_targets = [i for i, t in enumerate(tests) if t.get('findings_description', '')]

    print(f"対象: 疾患 {len(diseases)}件, 検査 {len(test_targets)}件")
    print(f"LLM: {LLM_MODEL} (primary), {LLM_FALLBACK_MODEL} (fallback)")

    sem = asyncio.Semaphore(30)  # 30並行（100+ RPM対応）

    async with aiohttp.ClientSession() as session:
        # 疾患はスキップ（改善済み）
        d_results = {}
        print(f"疾患: 改善済みのためスキップ")

        # 検査の改善（全件）
        print(f"\n=== 検査 findings_description 改善開始 ({len(test_targets)}件) ===")
        t_progress = {'done': 0, 'fail': 0, 'total': len(test_targets)}
        t_results = await improve_tests(session, tests, test_targets, sem, t_progress)
        print(f"検査完了: 成功 {len(t_results)}, 失敗 {t_progress['fail']}")

    # 結果を書き込み
    for i, new_fd in d_results.items():
        diseases[i]['findings_description'] = new_fd

    with open(os.path.join(DATA_DIR, "diseases.jsonl"), 'w', encoding='utf-8') as f:
        for d in diseases:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    for i, new_fd in t_results.items():
        tests[i]['findings_description'] = new_fd

    with open(os.path.join(DATA_DIR, "tests.jsonl"), 'w', encoding='utf-8') as f:
        for t in tests:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')

    print(f"\n書き込み完了")

    # サンプル表示
    if d_results:
        sample_i = list(d_results.keys())[0]
        d = diseases[sample_i]
        print(f"\n=== サンプル: {d['disease_name']} ===")
        print(d['findings_description'][:400])


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    # stdoutをunbufferedに
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    asyncio.run(main())
