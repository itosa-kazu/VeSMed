"""
VeSMed - コモンディジーズ追加スクリプト（1回実行用）

8疾患の findings_description を Vertex AI gemini-3-pro-preview で生成し、
diseases.jsonl に追記する。

Usage:
    python gen_common_diseases.py
    python gen_common_diseases.py --dry-run   # API呼び出しなし、メタデータのみ確認
"""

import json
import asyncio
import re
import os

# Vertex AI設定
VERTEX_SA_KEY = os.path.join(os.path.dirname(__file__), "thermal-outlet-483512-m4-8ec9647654b6.json")
VERTEX_PROJECT = "thermal-outlet-483512-m4"
VERTEX_LOCATION = "global"
VERTEX_MODEL = "gemini-3-pro-preview"

DISEASES_JSONL = os.path.join(os.path.dirname(__file__), "data", "diseases.jsonl")

# generate.py から流用した6セクションプロンプト
DISEASE_SECTION_PROMPTS = [
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

# 追加する8疾患のメタデータ
COMMON_DISEASES = [
    {
        "disease_name": "感冒（急性上気道炎）",
        "icd10": "J06.9",
        "category": "呼吸器内科・外科",
        "urgency": "通常",
        "age_peak": "全年齢",
        "gender_tendency": "性差なし",
        "description_for_embedding": "鼻汁、咽頭痛、咳嗽、微熱を主症状とする上気道のウイルス感染症。ライノウイルス、コロナウイルス等が原因。通常1-2週間で自然軽快するが、二次的な細菌感染合併に注意。成人で年2-4回罹患し、最も頻度の高い急性疾患である。",
        "risk_factors": ["小児", "集団生活", "冬季", "免疫低下", "睡眠不足", "ストレス"],
        "core_symptoms": ["鼻汁", "咽頭痛", "咳嗽"],
        "common_symptoms": ["微熱", "全身倦怠感", "頭痛", "くしゃみ"],
        "rare_but_specific": [],
        "core_signs": ["鼻粘膜発赤・腫脹", "咽頭発赤", "頸部リンパ節腫脹"],
        "differential_top5": [],
        "relevant_tests": [],
    },
    {
        "disease_name": "急性気管支炎",
        "icd10": "J20.9",
        "category": "呼吸器内科・外科",
        "urgency": "通常",
        "age_peak": "全年齢",
        "gender_tendency": "性差なし",
        "description_for_embedding": "急性の咳嗽を主症状とする気管支の炎症で、多くはウイルス性上気道炎に続発する。咳嗽は1-3週間持続し、初期は乾性、後に湿性となることが多い。発熱は軽度。肺炎との鑑別が臨床上最も重要。抗菌薬は通常不要。",
        "risk_factors": ["喫煙", "大気汚染", "冬季", "上気道炎後"],
        "core_symptoms": ["咳嗽", "喀痰"],
        "common_symptoms": ["微熱", "胸部不快感", "倦怠感"],
        "rare_but_specific": [],
        "core_signs": ["散在性rhonchi", "呼吸音正常〜やや粗"],
        "differential_top5": [],
        "relevant_tests": [],
    },
    {
        "disease_name": "咳喘息",
        "icd10": "J45.9",
        "category": "呼吸器内科・外科",
        "urgency": "通常",
        "age_peak": "20-50代",
        "gender_tendency": "女性にやや多い",
        "description_for_embedding": "喘鳴を伴わない慢性咳嗽を唯一の症状とする気管支喘息の亜型。8週間以上の乾性咳嗽が典型で、夜間〜早朝に増悪する。気管支拡張薬に反応することが診断的。慢性咳嗽の原因として最も頻度が高く、約30%が典型的喘息に移行する。",
        "risk_factors": ["アトピー素因", "女性", "上気道炎後", "喫煙", "季節変動"],
        "core_symptoms": ["慢性乾性咳嗽"],
        "common_symptoms": ["夜間増悪", "会話中の咳込み", "咽頭のイガイガ感"],
        "rare_but_specific": ["気管支拡張薬で咳嗽改善"],
        "core_signs": ["聴診上wheezeなし", "呼吸音正常"],
        "differential_top5": [],
        "relevant_tests": [],
    },
    {
        "disease_name": "上気道咳嗽症候群（UACS/後鼻漏）",
        "icd10": "J31.0",
        "category": "耳鼻咽喉科",
        "urgency": "通常",
        "age_peak": "全年齢",
        "gender_tendency": "性差なし",
        "description_for_embedding": "後鼻漏による咽頭刺激が原因の慢性咳嗽。アレルギー性鼻炎、副鼻腔炎、血管運動性鼻炎に続発する。咳嗽は臥位で増悪し、咽頭の異物感・いがらっぽさを伴う。咽頭後壁の粘液付着が特徴的所見。慢性咳嗽の3大原因の1つ。",
        "risk_factors": ["アレルギー性鼻炎", "慢性副鼻腔炎", "鼻中隔弯曲", "喫煙"],
        "core_symptoms": ["慢性咳嗽", "後鼻漏感"],
        "common_symptoms": ["咽頭異物感", "頻回の咳払い", "鼻閉", "鼻汁"],
        "rare_but_specific": ["咽頭後壁の粘液付着（cobblestone appearance）"],
        "core_signs": ["咽頭後壁粘液", "鼻粘膜腫脹"],
        "differential_top5": [],
        "relevant_tests": [],
    },
    {
        "disease_name": "胃食道逆流症（GERD）",
        "icd10": "K21.0",
        "category": "消化器内科・外科",
        "urgency": "通常",
        "age_peak": "40-60代",
        "gender_tendency": "男性にやや多い",
        "description_for_embedding": "胃酸の食道への逆流により胸やけ、呑酸を主症状とする疾患。食後・臥位で増悪し、PPI投与で改善する。食道外症状として慢性咳嗽、咽頭違和感、嗄声、胸痛も引き起こし、慢性咳嗽の3大原因の1つ。肥満、飲酒、喫煙がリスク因子。",
        "risk_factors": ["肥満", "飲酒", "喫煙", "高脂肪食", "食道裂孔ヘルニア", "妊娠"],
        "core_symptoms": ["胸やけ", "呑酸"],
        "common_symptoms": ["慢性咳嗽", "咽頭違和感", "嗄声", "胸痛"],
        "rare_but_specific": ["PPI投与で症状改善"],
        "core_signs": ["心窩部圧痛（軽度）"],
        "differential_top5": [],
        "relevant_tests": [],
    },
    {
        "disease_name": "好酸球性気管支炎",
        "icd10": "J98.8",
        "category": "呼吸器内科・外科",
        "urgency": "通常",
        "age_peak": "30-50代",
        "gender_tendency": "性差なし",
        "description_for_embedding": "気道過敏性を示さない好酸球性気道炎症による慢性咳嗽。咳喘息と異なり気管支拡張薬に反応せず、誘発喀痰中の好酸球増多（>3%）で診断する。吸入ステロイドに良好に反応。慢性咳嗽の約10-30%を占める。",
        "risk_factors": ["アトピー素因", "アレルゲン曝露"],
        "core_symptoms": ["慢性乾性咳嗽"],
        "common_symptoms": ["軽度の喀痰"],
        "rare_but_specific": ["誘発喀痰好酸球>3%", "吸入ステロイドで改善"],
        "core_signs": ["聴診上wheezeなし", "呼吸機能正常"],
        "differential_top5": [],
        "relevant_tests": [],
    },
    {
        "disease_name": "ACE阻害薬誘発性咳嗽",
        "icd10": "T46.4",
        "category": "薬剤性",
        "urgency": "通常",
        "age_peak": "全年齢（ACE阻害薬服用者）",
        "gender_tendency": "女性に多い(2:1)",
        "description_for_embedding": "ACE阻害薬（エナラプリル、リシノプリル等）服用に伴う乾性咳嗽。服用開始後数日〜数ヶ月で発症し、中止後1-4週間で消失する。ブラジキニン蓄積が機序。服用者の5-35%に発生し、アジア人でより高頻度。ARBへの切り替えで改善。",
        "risk_factors": ["ACE阻害薬服用", "女性", "アジア人", "非喫煙者"],
        "core_symptoms": ["乾性咳嗽"],
        "common_symptoms": ["咽頭のイガイガ感"],
        "rare_but_specific": ["ACE阻害薬中止で咳嗽消失"],
        "core_signs": ["身体所見正常"],
        "differential_top5": [],
        "relevant_tests": [],
    },
    {
        "disease_name": "機能性ディスペプシア",
        "icd10": "K30",
        "category": "消化器内科・外科",
        "urgency": "通常",
        "age_peak": "20-50代",
        "gender_tendency": "女性にやや多い",
        "description_for_embedding": "器質的疾患がないにもかかわらず心窩部痛、心窩部灼熱感、食後のもたれ感、早期満腹感を呈する機能性疾患。Rome IV基準で診断。上部消化管内視鏡で器質的疾患を除外することが必須。H. pylori除菌、PPI、アコチアミドが治療選択肢。人口の10-20%が罹患。",
        "risk_factors": ["ストレス", "不安障害", "うつ病", "H. pylori感染", "NSAIDs使用"],
        "core_symptoms": ["心窩部痛", "食後もたれ感"],
        "common_symptoms": ["早期満腹感", "心窩部灼熱感", "嘔気", "食欲低下"],
        "rare_but_specific": [],
        "core_signs": ["心窩部圧痛（軽度）", "腹部所見乏しい"],
        "differential_top5": [],
        "relevant_tests": [],
    },
]


async def generate_findings(disease_entry):
    """1疾患の findings_description を6セクション並列生成"""
    from google import genai
    from google.genai.types import GenerateContentConfig

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = VERTEX_SA_KEY
    client = genai.Client(
        project=VERTEX_PROJECT,
        location=VERTEX_LOCATION,
        vertexai=True,
    )

    name = disease_entry["disease_name"]
    category = disease_entry["category"]

    async def gen_section(sec_idx):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await client.aio.models.generate_content(
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
                    print(f"    {name} S{sec_idx+1} Retry {attempt+1}/{max_retries}, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                print(f"    {name} S{sec_idx+1} ERROR: {e}")
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
    suffix = f", failed S{failed}" if failed else ""
    print(f"  {name} ... OK ({len(combined)}字{suffix})")
    return combined if combined else None


async def main():
    import sys
    dry_run = "--dry-run" in sys.argv

    # 既存疾患を読み込み
    existing = []
    existing_names = set()
    with open(DISEASES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                existing.append(d)
                existing_names.add(d["disease_name"])

    print(f"既存疾患: {len(existing)}件")

    # 重複チェック
    to_add = []
    for d in COMMON_DISEASES:
        if d["disease_name"] in existing_names:
            print(f"  SKIP (既存): {d['disease_name']}")
        else:
            to_add.append(d)

    if not to_add:
        print("追加する疾患はありません。")
        return

    print(f"追加対象: {len(to_add)}件")
    for d in to_add:
        print(f"  - {d['disease_name']} ({d['category']}, {d['urgency']})")

    if dry_run:
        print("\n--dry-run: API呼び出しなし。終了。")
        return

    # findings_description 生成
    print("\nfindings_description 生成中 (gemini-3-pro-preview)...")
    for d in to_add:
        findings = await generate_findings(d)
        if findings:
            d["findings_description"] = findings
        else:
            print(f"  WARNING: {d['disease_name']} の生成に失敗")
            d["findings_description"] = ""

    # diseases.jsonl に追記
    with open(DISEASES_JSONL, "a", encoding="utf-8") as f:
        for d in to_add:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n完了: {len(to_add)}疾患を diseases.jsonl に追記")
    print("次のステップ:")
    print("  python index.py          # ChromaDB再構築")
    print("  del data\\sim_matrix.npz   # sim_matrixキャッシュ削除")


if __name__ == "__main__":
    asyncio.run(main())
