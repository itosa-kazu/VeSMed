"""
問診・身体診察項目をtests.jsonlに追加するスクリプト

VeSMedの天井は記述の質で決まる。
問診・身体診察は「最良の検査」: コスト0、即時、高情報利得。
"""
import json
import asyncio
import aiohttp
import sys
import os
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    LLM_FALLBACK_API_KEY, LLM_FALLBACK_BASE_URL, LLM_FALLBACK_MODEL,
    DATA_DIR,
)

# ================================================================
# 問診・身体診察項目リスト
# ================================================================
EXAM_ITEMS = [
    # ----- 身体診察 -----
    # 全身
    {"test_name": "全身状態の観察", "category": "身体診察（視診・全身）",
     "description_for_embedding": "患者の全体的な外観・意識状態・栄養状態・脱水徴候・体位・表情・苦悶の有無を視診で評価する。ベッドサイドで数秒から数分で実施でき、器具は不要。",
     "sample_type": "非侵襲", "turnaround_minutes": 1},

    {"test_name": "バイタルサイン測定", "category": "身体診察（バイタル）",
     "description_for_embedding": "体温・血圧・脈拍・呼吸数・SpO2を測定する。体温計・血圧計・パルスオキシメーターを使用し、2-3分で完了する。",
     "sample_type": "非侵襲", "turnaround_minutes": 3},

    # 頭頸部
    {"test_name": "頭頸部の診察", "category": "身体診察（頭頸部）",
     "description_for_embedding": "頭部・顔面・頸部の視診と触診を行う。頸部リンパ節腫脹、甲状腺腫大、頸静脈怒張、頸部硬直の有無を評価する。",
     "sample_type": "非侵襲", "turnaround_minutes": 3},

    {"test_name": "口腔・咽頭の診察", "category": "身体診察（頭頸部）",
     "description_for_embedding": "舌圧子とペンライトで口腔粘膜・歯肉・舌・扁桃・咽頭後壁を視診する。口腔内乾燥、白苔、発赤、潰瘍、扁桃腫大、膿栓の有無を評価する。",
     "sample_type": "非侵襲", "turnaround_minutes": 2},

    {"test_name": "眼の診察", "category": "身体診察（頭頸部）",
     "description_for_embedding": "ペンライトで瞳孔の対光反射・左右差・眼球運動を評価し、眼瞼結膜の蒼白・黄染・充血を視診する。眼底鏡は含まない簡易診察。",
     "sample_type": "非侵襲", "turnaround_minutes": 2},

    {"test_name": "眼底検査", "category": "身体診察（頭頸部）",
     "description_for_embedding": "直像眼底鏡で眼底を観察する。乳頭浮腫、出血、白斑、動脈硬化性変化、Roth斑、増殖性変化の有無を評価する。散瞳薬を使用する場合は15-20分待機。",
     "sample_type": "非侵襲", "turnaround_minutes": 5},

    # 胸部
    {"test_name": "心音聴診", "category": "身体診察（胸部）",
     "description_for_embedding": "聴診器で心尖部・左胸骨縁・心基部の心音を聴取する。過剰心音（III音・IV音）、心雑音（収縮期・拡張期）、心膜摩擦音の有無を評価する。体位変換（左側臥位・座位前傾）で所見が変化する。",
     "sample_type": "非侵襲", "turnaround_minutes": 3},

    {"test_name": "肺音聴診", "category": "身体診察（胸部）",
     "description_for_embedding": "聴診器で両側肺野の呼吸音を前胸部・背部で聴取する。呼吸音減弱・消失、crackles（水泡音・捻髪音）、wheezes、rhonchi、胸膜摩擦音の有無と分布を評価する。",
     "sample_type": "非侵襲", "turnaround_minutes": 3},

    {"test_name": "胸部の打診", "category": "身体診察（胸部）",
     "description_for_embedding": "両側胸壁を打診し、清音・濁音・鼓音の分布を評価する。胸水貯留では濁音、気胸では鼓音と打診で鑑別できる。",
     "sample_type": "非侵襲", "turnaround_minutes": 2},

    # 腹部
    {"test_name": "腹部の視診・聴診", "category": "身体診察（腹部）",
     "description_for_embedding": "腹部の膨隆・陥凹・手術瘢痕・静脈怒張・腸蠕動の亢進を視診し、聴診器で腸蠕動音の亢進・減弱・消失、血管雑音を聴取する。",
     "sample_type": "非侵襲", "turnaround_minutes": 3},

    {"test_name": "腹部触診", "category": "身体診察（腹部）",
     "description_for_embedding": "浅い触診で圧痛部位を同定し、深い触診で腫瘤・臓器腫大・筋性防御・反跳痛を評価する。McBurney点圧痛、Murphy徴候、肝臓下縁、脾臓触知の有無を確認する。",
     "sample_type": "非侵襲", "turnaround_minutes": 5},

    {"test_name": "直腸診", "category": "身体診察（腹部）",
     "description_for_embedding": "手袋を装着し人差し指を肛門から挿入して直腸粘膜・前立腺（男性）・ダグラス窩を触診する。腫瘤触知、血液付着、前立腺肥大・硬結の有無を評価する。",
     "sample_type": "低侵襲", "turnaround_minutes": 3},

    # 皮膚
    {"test_name": "皮膚の診察", "category": "身体診察（皮膚）",
     "description_for_embedding": "全身の皮膚を視診・触診し、発疹の形態（紅斑・丘疹・水疱・膿疱・紫斑）、分布、色調変化、潰瘍、壊死、蜂窩織炎様変化、出血斑、点状出血を評価する。爪・毛髪の変化も含む。",
     "sample_type": "非侵襲", "turnaround_minutes": 5},

    # 四肢・末梢
    {"test_name": "四肢末梢の診察", "category": "身体診察（四肢）",
     "description_for_embedding": "四肢末梢の浮腫（圧痕性・非圧痕性）、チアノーゼ、冷感、毛細血管再充満時間（CRT）、動脈拍動の左右差、ばち指の有無を視診・触診で評価する。",
     "sample_type": "非侵襲", "turnaround_minutes": 3},

    {"test_name": "関節の診察", "category": "身体診察（四肢）",
     "description_for_embedding": "関節の腫脹・発赤・熱感・圧痛・可動域制限を視診・触診で評価する。変形、関節液貯留（浮球感）、クレピタスの有無を確認する。",
     "sample_type": "非侵襲", "turnaround_minutes": 5},

    {"test_name": "下肢の血管診察", "category": "身体診察（四肢）",
     "description_for_embedding": "下肢の腫脹・発赤・把握痛・Homans徴候を評価し、深部静脈血栓症を疑う所見を確認する。足背動脈・後脛骨動脈の拍動触知で末梢動脈疾患を評価する。",
     "sample_type": "非侵襲", "turnaround_minutes": 3},

    # 神経
    {"test_name": "意識レベルの評価", "category": "身体診察（神経）",
     "description_for_embedding": "GCS（Glasgow Coma Scale: 開眼E・言語V・運動M）またはJCSで意識レベルを定量評価する。呼びかけ・痛み刺激への反応を観察する。",
     "sample_type": "非侵襲", "turnaround_minutes": 2},

    {"test_name": "髄膜刺激徴候の検査", "category": "身体診察（神経）",
     "description_for_embedding": "項部硬直（受動的頸部前屈での抵抗）、Kernig徴候（股関節90度屈曲位からの膝伸展制限）、Brudzinski徴候（頸部前屈時の股関節屈曲）を検査する。jolt accentuationも確認する。",
     "sample_type": "非侵襲", "turnaround_minutes": 3},

    {"test_name": "脳神経の検査", "category": "身体診察（神経）",
     "description_for_embedding": "第I-XII脳神経を系統的に検査する。嗅覚、視力・視野、瞳孔反射、眼球運動、顔面感覚・筋力、聴力、嚥下・発声、舌の偏位を評価する。",
     "sample_type": "非侵襲", "turnaround_minutes": 10},

    {"test_name": "運動・感覚系の検査", "category": "身体診察（神経）",
     "description_for_embedding": "四肢の筋力（MMT 0-5）、筋トーヌス、深部腱反射（膝蓋腱・アキレス腱・上腕二頭筋・上腕三頭筋）、病的反射（Babinski徴候）、感覚（触覚・痛覚・温度覚・振動覚・位置覚）を検査する。",
     "sample_type": "非侵襲", "turnaround_minutes": 10},

    {"test_name": "小脳機能・歩行の検査", "category": "身体診察（神経）",
     "description_for_embedding": "指鼻指試験・踵膝試験で協調運動障害を、Romberg試験で深部感覚障害を、歩行（通常歩行・継足歩行・つま先歩行・踵歩行）で運動失調を評価する。",
     "sample_type": "非侵襲", "turnaround_minutes": 5},

    # 筋骨格
    {"test_name": "脊椎・背部の診察", "category": "身体診察（筋骨格）",
     "description_for_embedding": "脊椎の棘突起叩打痛、傍脊柱筋の圧痛、脊柱可動域、SLRテスト（下肢伸展挙上試験）、FNSテスト（大腿神経伸展試験）を評価する。",
     "sample_type": "非侵襲", "turnaround_minutes": 5},

    # 泌尿器・生殖器
    {"test_name": "CVA叩打痛の検査", "category": "身体診察（泌尿器）",
     "description_for_embedding": "背部の肋骨脊柱角（CVA: costovertebral angle）を拳で叩打し、疼痛の有無を左右で評価する。腎盂腎炎、尿路結石で陽性となる。",
     "sample_type": "非侵襲", "turnaround_minutes": 1},

    # ----- 問診 -----
    {"test_name": "問診：既往歴・手術歴", "category": "問診",
     "description_for_embedding": "過去の疾患、入院歴、手術歴、慢性疾患の有無、治療歴を聴取する。現在の症状との関連を評価する。",
     "sample_type": "問診", "turnaround_minutes": 3},

    {"test_name": "問診：家族歴", "category": "問診",
     "description_for_embedding": "血縁者の疾患歴（悪性腫瘍、心血管疾患、糖尿病、自己免疫疾患、遺伝性疾患、若年突然死）を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 2},

    {"test_name": "問診：薬剤歴", "category": "問診",
     "description_for_embedding": "処方薬、OTC薬、サプリメント、漢方薬の使用状況、最近の変更、アドヒアランスを聴取する。薬剤性の副作用の可能性を評価する。",
     "sample_type": "問診", "turnaround_minutes": 3},

    {"test_name": "問診：アレルギー歴", "category": "問診",
     "description_for_embedding": "薬物アレルギー（抗菌薬、NSAIDs、造影剤等）、食物アレルギー、環境アレルゲン、アナフィラキシーの既往を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 2},

    {"test_name": "問診：渡航歴", "category": "問診",
     "description_for_embedding": "直近6ヶ月以内の海外渡航先、滞在期間、マラリア予防内服の有無、渡航先での食事・水・虫刺されの状況を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 2},

    {"test_name": "問診：職業歴・環境曝露", "category": "問診",
     "description_for_embedding": "現在と過去の職業、粉塵（アスベスト、シリカ）・有機溶剤・重金属への曝露歴、住居環境（古い建物、カビ、換気）を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 3},

    {"test_name": "問診：喫煙歴", "category": "問診",
     "description_for_embedding": "喫煙の有無、開始年齢、1日本数、喫煙年数（pack-years）、禁煙歴、受動喫煙の程度を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 1},

    {"test_name": "問診：飲酒歴", "category": "問診",
     "description_for_embedding": "飲酒の種類・量・頻度、飲酒年数、binge drinkingの有無、離脱症状の既往、CAGE質問紙を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 2},

    {"test_name": "問診：違法薬物・嗜好品", "category": "問診",
     "description_for_embedding": "覚醒剤、大麻、コカイン、ヘロイン、MDMA、合成カンナビノイド、吸入剤（シンナー等）、処方薬乱用の使用歴と使用方法（経静脈、吸入、経口）を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 2},

    {"test_name": "問診：性行為歴・STI歴", "category": "問診",
     "description_for_embedding": "性的活動の有無、パートナーの数、コンドーム使用、男性間性行為（MSM）、性感染症（STI）の既往、HIV検査歴を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 2},

    {"test_name": "問診：月経歴・産科歴", "category": "問診",
     "description_for_embedding": "最終月経日、月経周期の規則性、月経量の変化、妊娠の可能性、妊娠・出産歴（G/P）、閉経年齢、ホルモン補充療法の使用を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 2},

    {"test_name": "問診：外傷歴・最近の処置", "category": "問診",
     "description_for_embedding": "最近の外傷（打撲、切傷、刺傷、咬傷）、手術、歯科治療、カテーテル挿入、注射、刺青、ピアスなどの侵襲的処置の有無と時期を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 2},

    {"test_name": "問診：動物・ペット接触歴", "category": "問診",
     "description_for_embedding": "飼育動物の種類（犬、猫、鳥、爬虫類、齧歯類）、野生動物との接触、咬傷・掻傷、ダニ・ノミ曝露、家畜との接触を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 2},

    {"test_name": "問診：食事歴", "category": "問診",
     "description_for_embedding": "最近の食事内容、生食（刺身、生肉、生卵、生乳）、保存状態の不良な食品、外食・弁当、同様の症状を呈する同席者の有無を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 2},

    {"test_name": "問診：予防接種歴", "category": "問診",
     "description_for_embedding": "定期予防接種の完了状況、インフルエンザワクチン、肺炎球菌ワクチン、COVID-19ワクチン、渡航前ワクチン（A型肝炎、B型肝炎、狂犬病、黄熱）の接種歴を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 2},

    {"test_name": "問診：輸血歴・臓器移植歴", "category": "問診",
     "description_for_embedding": "輸血の有無と時期、輸血反応の既往、臓器移植・骨髄移植の既往、免疫抑制剤の使用状況を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 1},

    {"test_name": "問診：体重変化・食欲", "category": "問診",
     "description_for_embedding": "意図しない体重減少・増加の有無、期間と程度（kg/月）、食欲の変化、嚥下困難、早期満腹感を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 1},

    {"test_name": "問診：睡眠・精神状態", "category": "問診",
     "description_for_embedding": "不眠、過眠、いびき・無呼吸、抑うつ気分、不安、希死念慮、幻覚・妄想、認知機能低下の自覚、日常生活動作の変化を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 3},

    {"test_name": "問診：排尿・排便の変化", "category": "問診",
     "description_for_embedding": "排尿回数・量・性状（血尿、混濁尿）の変化、排尿困難、残尿感、排便回数・性状（下痢、便秘、血便、黒色便、粘液便）の変化を聴取する。",
     "sample_type": "問診", "turnaround_minutes": 2},
]

# ================================================================
# プロンプト
# ================================================================
SYSTEM_PROMPT = """あなたは臨床医学の最高権威です。問診項目・身体診察項目のfindings_descriptionを新規作成してください。

■ 最重要原則: 所見即所得
このテキストはembedding検索で「疾患の所見記述」との類似度計算に使われます。
疾患のfindings_descriptionに含まれる語彙・概念と一致するほど、類似度が高くなり、
その検査がその疾患に推薦されます。

■ 問診・身体診察は「最良の検査」
問診・身体診察は以下の点で検査室検査より優れています：
- コストゼロ、即時実施可能
- 多くの場合、最も高い情報利得
- 鑑別診断を劇的に絞り込む力がある

■ 書き方ルール（最重要 — これがVeSMedの天井を決める）

1. 「どんな臨床像の患者にこの診察/問診を行い、何がわかるか」を最初に書く
   ✅ 「高熱・悪寒・ショック所見を呈する患者、原因不明の発熱が持続する患者、
      術後や免疫不全の患者に心音聴診を行う。新規の心雑音は感染性心内膜炎を
      強く示唆し（感度約85%）、汎収縮期雑音は僧帽弁逆流を、駆出性収縮期雑音は
      大動脈弁狭窄を、拡張期漸減雑音は大動脈弁逆流を示唆する。」

2. 疾患名と具体的な所見を直接結びつける（最重要）
   ✅ 「細菌性髄膜炎では項部硬直が約80%、Kernig徴候が約50%で陽性。
      くも膜下出血では項部硬直が約75%で出現する。」
   ❌ 「髄膜刺激徴候があれば髄膜炎を疑う」（曖昧すぎる）

3. 各所見に出現頻度を付加する（教科書・ガイドラインの一般的値）
   ✅ 「急性虫垂炎ではMcBurney点圧痛が約80%、筋性防御が約60%」
   ❌ 「虫垂炎で圧痛がある」

4. 陰性所見の意味も書く（除外に使える所見）
   ✅ 「項部硬直が完全に陰性であれば細菌性髄膜炎の可能性は大幅に低下する
     （陰性尤度比 0.3）」

5. 鑑別に直結する所見のパターンを書く
   ✅ 「右下腹部圧痛+反跳痛+筋性防御=急性虫垂炎の古典的三徴。
      左下腹部圧痛は憩室炎を示唆。心窩部から右下腹部への痛みの移動は
      虫垂炎に特徴的（約50%）。」

6. 問診項目の場合: どの回答がどの疾患を示唆するかを具体的に書く
   ✅ 「東南アジア渡航歴があり帰国後2週間以内の発熱はマラリアを第一に疑う
     （三日熱マラリアの潜伏期12-17日）。デング熱の潜伏期は4-8日。
      アフリカ渡航後の発熱ではマラリア（熱帯熱）が最も危険（致死的）。」

■ 出力形式
findings_descriptionテキストのみ出力。説明・前置き不要。
文字数は1500-2500字。問診/身体診察は対象疾患が広いため、検査室検査より長くてよい。"""

USER_TEMPLATE = """以下の{item_type}のfindings_descriptionを新規作成してください。

項目名: {test_name}
カテゴリ: {category}
手技の説明: {description_for_embedding}

上記ルールに従って、findings_descriptionを作成してください。テキストのみ出力。"""

# quality_description用プロンプト
QUALITY_SYSTEM_PROMPT = """あなたは臨床医学の専門家です。問診・身体診察項目のquality_descriptionを作成してください。

■ quality_descriptionとは
その検査（問診/身体診察）で得られる所見が、どの程度「鑑別診断を絞り込む力」を持つかを記述するものです。

■ 書き方ルール
1. この項目で得られる「特異的な所見」をリストアップする
   - 他の疾患では出にくい、その疾患に特徴的な所見
2. 「非特異的な所見」も明示する
   - 多くの疾患で共通して出現し、鑑別に寄与しにくい所見
3. 感度・特異度が高い所見を強調する

■ 出力形式
quality_descriptionテキストのみ出力。説明不要。500-800字程度。"""

QUALITY_USER_TEMPLATE = """以下の{item_type}のquality_descriptionを作成してください。

項目名: {test_name}
カテゴリ: {category}
手技の説明: {description_for_embedding}

quality_descriptionテキストのみ出力。"""


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
                text = await resp.text()
                print(f"  API error {resp.status}: {text[:100]}")
                return None
    except Exception as e:
        print(f"  Exception: {e}")
        return None


async def call_with_fallback(session, system_prompt, user_prompt):
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
async def main():
    print(f"=== 問診・身体診察項目 生成開始 ({len(EXAM_ITEMS)}件) ===")
    print(f"LLM: {LLM_MODEL} (primary), {LLM_FALLBACK_MODEL} (fallback)")

    sem = asyncio.Semaphore(20)
    results = {}

    async with aiohttp.ClientSession() as session:
        # Phase 1: findings_description生成
        print(f"\n--- Phase 1: findings_description生成 ---")
        progress = {'done': 0, 'fail': 0}

        async def gen_findings(i, item):
            async with sem:
                item_type = "問診項目" if "問診" in item["category"] else "身体診察項目"
                user_prompt = USER_TEMPLATE.format(
                    item_type=item_type,
                    test_name=item["test_name"],
                    category=item["category"],
                    description_for_embedding=item["description_for_embedding"],
                )
                result = await call_with_fallback(session, SYSTEM_PROMPT, user_prompt)
                if result and len(result) > 500:
                    results.setdefault(i, {})["findings_description"] = result
                    progress['done'] += 1
                    if progress['done'] % 10 == 0:
                        print(f"  findings: {progress['done']}/{len(EXAM_ITEMS)} 完了")
                else:
                    progress['fail'] += 1
                    print(f"  失敗(findings): {item['test_name']}")

        tasks = [gen_findings(i, item) for i, item in enumerate(EXAM_ITEMS)]
        await asyncio.gather(*tasks)
        print(f"findings完了: 成功 {progress['done']}, 失敗 {progress['fail']}")

        # Phase 2: quality_description生成
        print(f"\n--- Phase 2: quality_description生成 ---")
        q_progress = {'done': 0, 'fail': 0}

        async def gen_quality(i, item):
            async with sem:
                item_type = "問診項目" if "問診" in item["category"] else "身体診察項目"
                user_prompt = QUALITY_USER_TEMPLATE.format(
                    item_type=item_type,
                    test_name=item["test_name"],
                    category=item["category"],
                    description_for_embedding=item["description_for_embedding"],
                )
                result = await call_with_fallback(session, QUALITY_SYSTEM_PROMPT, user_prompt)
                if result and len(result) > 200:
                    results.setdefault(i, {})["quality_description"] = result
                    q_progress['done'] += 1
                    if q_progress['done'] % 10 == 0:
                        print(f"  quality: {q_progress['done']}/{len(EXAM_ITEMS)} 完了")
                else:
                    q_progress['fail'] += 1
                    print(f"  失敗(quality): {item['test_name']}")

        tasks = [gen_quality(i, item) for i, item in enumerate(EXAM_ITEMS)]
        await asyncio.gather(*tasks)
        print(f"quality完了: 成功 {q_progress['done']}, 失敗 {q_progress['fail']}")

    # tests.jsonlに追記
    existing_tests = []
    with open(os.path.join(DATA_DIR, "tests.jsonl"), 'r', encoding='utf-8') as f:
        for line in f:
            existing_tests.append(json.loads(line))

    existing_names = {t["test_name"] for t in existing_tests}
    added = 0

    for i, item in enumerate(EXAM_ITEMS):
        if item["test_name"] in existing_names:
            print(f"  スキップ（既存）: {item['test_name']}")
            continue
        if i not in results or "findings_description" not in results[i]:
            print(f"  スキップ（生成失敗）: {item['test_name']}")
            continue

        entry = {
            "test_name": item["test_name"],
            "category": item["category"],
            "description_for_embedding": item["description_for_embedding"],
            "turnaround_minutes": item["turnaround_minutes"],
            "sample_type": item["sample_type"],
            "contraindications": [],
            "notes": "",
            "findings_description": results[i]["findings_description"],
            "quality_description": results[i].get("quality_description", ""),
        }
        existing_tests.append(entry)
        added += 1

    with open(os.path.join(DATA_DIR, "tests.jsonl"), 'w', encoding='utf-8') as f:
        for t in existing_tests:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')

    print(f"\n=== 完了: {added}件追加 (合計 {len(existing_tests)}件) ===")


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    asyncio.run(main())
