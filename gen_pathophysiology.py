"""
VeSMed - 病態生理セクション追加スクリプト

全疾患の findings_description に第7セクション「病態生理・発症メカニズム」を追加する。
Vertex AI gemini-3-pro-preview を使用。

断点続行対応: 既に追加済みの疾患はスキップ。
10件ごとに中間保存。

Usage:
    python gen_pathophysiology.py              # 全件実行
    python gen_pathophysiology.py --dry-run    # API呼び出しなし
    python gen_pathophysiology.py --max 10     # 最初の10件のみ
"""

import json
import asyncio
import re
import os
import sys
import time

# Vertex AI設定
VERTEX_SA_KEY = os.path.join(os.path.dirname(__file__), "thermal-outlet-483512-m4-8ec9647654b6.json")
VERTEX_PROJECT = "thermal-outlet-483512-m4"
VERTEX_LOCATION = "global"
VERTEX_MODEL = "gemini-3-pro-preview"

DISEASES_JSONL = os.path.join(os.path.dirname(__file__), "data", "diseases.jsonl")

# 第7セクションのマーカー（既に追加済みかどうかの判定に使用）
PATHO_MARKER = "【病態生理・発症メカニズム】"

PATHOPHYSIOLOGY_PROMPT = """\
あなたは日本の臨床医学・病態生理学に精通した専門家です。
与えられた疾患について、「病態生理・発症メカニズム」のみを詳細に記述してください。
出力はプレーンテキストのみ。省略しない。

【重要】臨床的な症状や検査所見は「この疾患ではこの所見が見られる」ではなく、
「なぜその所見が生じるか（メカニズム）」を軸に記述すること。

記述内容:
- 発症の分子・細胞レベルのメカニズム（免疫応答、受容体異常、代謝経路の障害、遺伝子変異等）
- 臓器障害の機序（なぜその臓器が傷害されるか）
- 主要症状が生じる病態生理学的理由（例: なぜ関節痛が生じるか → 滑膜への免疫複合体沈着と補体活性化による炎症）
- 検査異常が生じるメカニズム（例: なぜCRP上昇か → IL-6による肝臓でのCRP合成亢進）
- 病型・亜型間のメカニズムの違い
- 時間経過に伴う病態の進展（急性期→慢性期の変化）
- 合併症が生じるメカニズム
- 治療標的となる経路（薬理学的メカニズムの理解に資する記述）"""

MAX_CONCURRENCY = 10


def read_diseases():
    diseases = []
    with open(DISEASES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                diseases.append(json.loads(line))
    return diseases


def write_diseases(diseases):
    """一時ファイルに書き込み→リネーム（Windows OSError対策）"""
    tmp_path = DISEASES_JSONL + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for d in diseases:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    # Windows: os.replace()がPermissionErrorになることがあるのでリトライ
    for attempt in range(5):
        try:
            if os.path.exists(DISEASES_JSONL):
                os.replace(tmp_path, DISEASES_JSONL)
            else:
                os.rename(tmp_path, DISEASES_JSONL)
            return
        except PermissionError:
            if attempt < 4:
                time.sleep(1)
            else:
                raise


async def gen_pathophysiology_one(client, semaphore, disease, index, total):
    """1疾患の病態生理セクションを生成"""
    from google.genai.types import GenerateContentConfig

    name = disease["disease_name"]
    category = disease.get("category", "")
    label = f"[{index + 1}/{total}] {name}"

    max_retries = 5
    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.aio.models.generate_content(
                    model=VERTEX_MODEL,
                    contents=f"{name}（{category}）",
                    config=GenerateContentConfig(
                        system_instruction=PATHOPHYSIOLOGY_PROMPT,
                        temperature=1.0,
                        max_output_tokens=65536,
                    ),
                )
                text = response.text if response.text else ""
                text = text.strip()
                # マークダウン記法を除去
                text = re.sub(r"```.*?\n?", "", text).strip()
                text = re.sub(r"^#{1,4}\s+", "", text, flags=re.MULTILINE)
                text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)

                if text:
                    print(f"  {label} ... OK ({len(text)}字)")
                    return index, text
                else:
                    print(f"  {label} ... EMPTY")
                    return index, None
        except Exception as e:
            err_str = str(e)
            retryable = any(code in err_str for code in ["429", "500", "503", "502", "RESOURCE_EXHAUSTED"])
            if retryable and attempt < max_retries - 1:
                wait = (attempt + 1) * 15
                print(f"    {label} Retry {attempt+1}/{max_retries}, waiting {wait}s...")
                await asyncio.sleep(wait)
                continue
            print(f"  {label} ... ERROR: {e}")
            return index, None
    return index, None


async def main():
    dry_run = "--dry-run" in sys.argv
    max_count = None
    if "--max" in sys.argv:
        idx = sys.argv.index("--max")
        if idx + 1 < len(sys.argv):
            max_count = int(sys.argv[idx + 1])

    diseases = read_diseases()
    print(f"全疾患: {len(diseases)}件")

    # 対象: findings_descriptionがあり、まだ病態生理セクションが追加されていない疾患
    targets = []
    for i, d in enumerate(diseases):
        fd = d.get("findings_description", "")
        if fd and len(fd) > 100 and PATHO_MARKER not in fd:
            targets.append((i, d))

    if max_count:
        targets = targets[:max_count]

    already_done = sum(1 for d in diseases if PATHO_MARKER in d.get("findings_description", ""))
    print(f"追加済み: {already_done}件 / 対象: {len(targets)}件")

    if not targets:
        print("追加する疾患はありません。")
        return

    if dry_run:
        print("\n--dry-run: 対象疾患一覧:")
        for i, (orig_idx, d) in enumerate(targets[:20]):
            print(f"  {i+1}. {d['disease_name']} ({len(d.get('findings_description', ''))}字)")
        if len(targets) > 20:
            print(f"  ... 他{len(targets) - 20}件")
        return

    # Vertex AIクライアント初期化
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = VERTEX_SA_KEY
    from google import genai
    client = genai.Client(
        project=VERTEX_PROJECT,
        location=VERTEX_LOCATION,
        vertexai=True,
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    t0 = time.time()

    # バッチ処理（10件ずつ、中間保存）
    batch_size = 10
    completed = 0
    failed = 0

    for batch_start in range(0, len(targets), batch_size):
        batch = targets[batch_start:batch_start + batch_size]

        coros = [
            gen_pathophysiology_one(client, semaphore, d, batch_start + j, len(targets))
            for j, (orig_idx, d) in enumerate(batch)
        ]
        results = await asyncio.gather(*coros)

        for (orig_idx, d), (_, text) in zip(batch, results):
            if text:
                # 既存のfindings_descriptionに病態生理セクションを追記
                d["findings_description"] += f"\n\n{PATHO_MARKER}\n{text}"
                diseases[orig_idx] = d
                completed += 1
            else:
                failed += 1

        # 中間保存
        write_diseases(diseases)
        elapsed = time.time() - t0
        print(f"  --- Batch saved: {completed}/{len(targets)} done, {failed} failed, {elapsed:.0f}s elapsed ---")

    elapsed = time.time() - t0
    print(f"\n完了: {completed}件追加, {failed}件失敗, {elapsed:.0f}s")
    print("次のステップ:")
    print("  del data\\sim_matrix.npz data\\sim_matrix_hpe.npz  # キャッシュ削除")


if __name__ == "__main__":
    asyncio.run(main())
