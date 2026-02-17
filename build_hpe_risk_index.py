"""
Type R HPE逆引きインデックス構築（順方向: 項目→疾患）

Phase 1: テキスト検索（risk_factors + fd_background）
Phase 2: LLM検証・補完（per-item, ~80コール）

API: gemini-3-pro-preview-c via new.12ai.org (RPM < 10)
"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_DIR, DISEASES_JSONL

HPE_JSONL = os.path.join(DATA_DIR, "hpe_items.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "hpe_risk_reverse_index.json")
PROGRESS_FILE = os.path.join(DATA_DIR, "hpe_risk_progress.json")

LLM_API_KEY = "sk-hI5iGw1n6EuuCydhi70UNTtENTUQTFknpbeGXxadyZhhxkcR"
LLM_BASE_URL = "https://new.12ai.org/v1"
LLM_MODEL = "gemini-3-pro-preview-c"
RPM_LIMIT = 8

TYPE_R_SUBCATS = {"薬剤歴", "嗜好/社会歴", "既往歴", "家族歴"}

# 各項目のテキスト検索キーワード
SEARCH_KEYWORDS = {
    # 薬剤歴
    "NSAIDs": ["NSAID", "非ステロイド性抗炎症", "ロキソプロフェン", "イブプロフェン", "ジクロフェナク", "インドメタシン", "鎮痛薬"],
    "ACE阻害薬": ["ACE阻害", "エナラプリル", "リシノプリル", "ペリンドプリル", "RAAS"],
    "ARB": ["ARB", "アンジオテンシン受容体", "バルサルタン", "カンデサルタン", "テルミサルタン"],
    "β遮断薬": ["β遮断", "ベータ遮断", "メトプロロール", "ビソプロロール", "カルベジロール", "プロプラノロール"],
    "Ca拮抗薬": ["Ca拮抗", "カルシウム拮抗", "アムロジピン", "ニフェジピン", "ジルチアゼム", "ベラパミル"],
    "ループ利尿薬": ["ループ利尿", "フロセミド", "トルセミド", "利尿薬"],
    "チアジド系利尿薬": ["チアジド", "ヒドロクロロチアジド", "サイアザイド"],
    "ワルファリン": ["ワルファリン", "ワーファリン", "抗凝固"],
    "DOAC": ["DOAC", "直接経口抗凝固", "リバーロキサバン", "アピキサバン", "エドキサバン", "ダビガトラン"],
    "抗血小板薬": ["抗血小板", "アスピリン", "クロピドグレル", "プラスグレル"],
    "ステロイド": ["ステロイド", "プレドニゾロン", "デキサメタゾン", "副腎皮質", "糖質コルチコイド", "コルチコステロイド"],
    "免疫抑制薬": ["免疫抑制", "アザチオプリン", "シクロスポリン", "タクロリムス", "メトトレキサート", "MTX"],
    "経口避妊薬/HRT": ["経口避妊", "OCP", "ホルモン補充", "HRT", "エストロゲン", "ピル"],
    "スタチン": ["スタチン", "HMG-CoA", "アトルバスタチン", "ロスバスタチン"],
    "PPI": ["PPI", "プロトンポンプ阻害", "オメプラゾール", "ランソプラゾール", "エソメプラゾール"],
    "インスリン": ["インスリン"],
    "SU薬": ["SU薬", "スルホニルウレア", "グリメピリド", "グリクラジド", "グリベンクラミド"],
    "SGLT2阻害薬": ["SGLT2", "エンパグリフロジン", "ダパグリフロジン", "カナグリフロジン"],
    "GLP-1受容体作動薬": ["GLP-1", "リラグルチド", "セマグルチド", "デュラグルチド"],
    "甲状腺ホルモン": ["甲状腺ホルモン", "レボチロキシン", "チラーヂン"],
    "抗甲状腺薬": ["抗甲状腺", "チアマゾール", "プロピルチオウラシル", "MMI", "PTU", "メルカゾール"],
    "抗てんかん薬": ["抗てんかん", "バルプロ酸", "カルバマゼピン", "フェニトイン", "レベチラセタム", "ラモトリギン"],
    "抗精神病薬": ["抗精神病", "リスペリドン", "オランザピン", "クエチアピン", "ハロペリドール", "クロザピン"],
    "SSRI/SNRI": ["SSRI", "SNRI", "抗うつ", "パロキセチン", "セルトラリン", "デュロキセチン", "フルボキサミン"],
    "リチウム": ["リチウム"],
    "化学療法薬": ["化学療法", "抗がん剤", "抗癌剤", "殺細胞性", "レジメン"],
    "免疫チェックポイント阻害薬": ["チェックポイント阻害", "ニボルマブ", "ペムブロリズマブ", "アテゾリズマブ", "イピリムマブ", "irAE"],
    "生物学的製剤": ["生物学的製剤", "TNF阻害", "インフリキシマブ", "アダリムマブ", "トシリズマブ", "リツキシマブ"],
    "最近の抗菌薬使用": ["抗菌薬", "抗生物質", "抗生剤", "広域抗菌"],
    "アミオダロン": ["アミオダロン"],
    # 嗜好/社会歴
    "現在喫煙": ["喫煙", "タバコ", "smoking", "pack-year"],
    "過去喫煙": ["喫煙", "禁煙", "元喫煙"],
    "飲酒（常習）": ["飲酒", "アルコール", "エタノール", "多飲"],
    "大量飲酒/暴飲": ["大量飲酒", "暴飲", "アルコール依存", "多飲", "飲酒"],
    "コカイン/覚醒剤": ["コカイン", "覚醒剤", "アンフェタミン", "メタンフェタミン", "違法薬物"],
    "静注薬物使用(IVDU)": ["静注薬物", "IVDU", "注射薬物", "薬物注射"],
    "職業曝露": ["職業", "粉塵", "アスベスト", "化学物質", "有機溶剤", "石綿"],
    "医療従事者": ["医療従事者", "針刺し", "院内感染", "医療機関"],
    "渡航歴（熱帯/途上国）": ["渡航", "熱帯", "途上国", "海外旅行", "endemic", "流行地"],
    "長時間不動/長距離移動": ["長時間不動", "長距離", "不動", "安静臥床", "臥床"],
    "菜食/偏食": ["菜食", "偏食", "ビーガン", "栄養欠乏"],
    "生食/非加熱食品": ["生食", "生肉", "生魚", "非加熱", "刺身"],
    "集団生活": ["集団生活", "寮", "軍隊", "刑務所", "施設"],
    "ペット飼育": ["ペット", "犬", "猫", "鳥", "爬虫類", "人畜共通"],
    "性行為歴": ["性行為", "STI", "性感染", "性的接触"],
    "妊娠・授乳中": ["妊娠", "妊婦", "授乳", "産褥"],
    # 既往歴
    "高血圧": ["高血圧", "血圧上昇", "高血圧症"],
    "糖尿病": ["糖尿病", "血糖", "HbA1c", "インスリン抵抗性"],
    "脂質異常症": ["脂質異常", "高脂血", "高コレステロール", "高LDL", "動脈硬化"],
    "冠動脈疾患": ["冠動脈", "狭心症", "心筋梗塞", "虚血性心疾患"],
    "心不全": ["心不全", "心機能低下"],
    "心房細動": ["心房細動", "AF", "Af"],
    "弁膜症/人工弁": ["弁膜症", "人工弁", "弁置換"],
    "DVT/PE既往": ["DVT", "PE", "深部静脈血栓", "肺血栓塞栓", "血栓症"],
    "喘息": ["喘息", "気管支喘息"],
    "COPD": ["COPD", "慢性閉塞性肺疾患", "肺気腫"],
    "消化性潰瘍": ["消化性潰瘍", "胃潰瘍", "十二指腸潰瘍"],
    "肝疾患": ["肝疾患", "肝硬変", "肝炎", "肝障害", "B型肝炎", "C型肝炎"],
    "炎症性腸疾患": ["炎症性腸疾患", "IBD", "潰瘍性大腸炎", "クローン病"],
    "CKD": ["CKD", "慢性腎臓病", "腎機能低下", "腎不全", "透析"],
    "腎結石": ["腎結石", "尿路結石", "結石症"],
    "脳卒中": ["脳卒中", "脳梗塞", "脳出血"],
    "てんかん": ["てんかん", "痙攣", "発作"],
    "関節リウマチ": ["関節リウマチ", "RA"],
    "SLE/膠原病": ["SLE", "全身性エリテマトーデス", "膠原病", "自己免疫疾患"],
    "甲状腺疾患": ["甲状腺", "バセドウ", "橋本"],
    "悪性腫瘍": ["悪性腫瘍", "がん", "癌", "腫瘍", "悪性新生物"],
    "HIV/AIDS": ["HIV", "AIDS", "免疫不全"],
    "脾摘後": ["脾摘", "脾臓摘出", "無脾"],
    "胃切除後": ["胃切除", "胃全摘"],
    "最近の手術": ["手術", "術後"],
    "最近の入院/カテーテル": ["入院", "カテーテル"],
    "凝固異常/血栓性素因": ["凝固異常", "血栓性素因", "プロテインC", "プロテインS", "Factor V"],
    "アレルギー歴": ["アレルギー", "アナフィラキシー", "薬疹"],
    # 家族歴
    "若年心血管疾患": ["家族歴", "若年", "心筋梗塞", "心臓突然死", "冠動脈"],
    "悪性腫瘍家族歴": ["家族歴", "遺伝性", "BRCA", "Lynch", "家族性"],
    "自己免疫疾患家族歴": ["家族歴", "自己免疫", "遺伝的素因"],
    "血栓症家族歴": ["家族歴", "血栓", "凝固"],
    "突然死家族歴": ["家族歴", "突然死", "心臓突然死"],
    "脳血管疾患家族歴": ["家族歴", "脳卒中", "くも膜下出血", "脳動脈瘤"],
}


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def text_search(diseases, keywords):
    """risk_factorsとfd_backgroundをキーワード検索。"""
    matched = set()
    for d in diseases:
        dname = d["disease_name"]
        rf_text = " ".join(str(x) for x in d.get("risk_factors", []))
        bg_text = d.get("fd_background", "")
        combined = rf_text + " " + bg_text
        for kw in keywords:
            if kw in combined:
                matched.add(dname)
                break
    return sorted(matched)


PROMPTS_BY_SUBCAT = {
    "薬剤歴": (
        "この薬剤を服用中の患者において、以下のいずれかに該当する疾患を、下記の疾患リストから選んでください。\n"
        "1. この薬剤が原因・増悪因子となりうる疾患\n"
        "2. この薬剤が禁忌・相対禁忌の疾患\n"
        "3. この薬剤の副作用として発症しうる疾患\n"
        "4. この薬剤が治療薬として使われる疾患（=この薬を飲んでいるということは、この疾患の既往がある可能性）"
    ),
    "嗜好/社会歴": (
        "この生活習慣・曝露歴を持つ患者において、発症リスクが有意に上昇する疾患を、"
        "下記の疾患リストから選んでください。"
    ),
    "既往歴": (
        "この既往疾患を持つ患者において、以下のいずれかに該当する疾患を、下記の疾患リストから選んでください。\n"
        "1. この既往に直接関連して発症リスクが上がる疾患\n"
        "2. この既往の合併症として起こりうる疾患\n"
        "3. この既往の治療経過中に発症リスクが上がる疾患"
    ),
    "家族歴": (
        "この家族歴がある患者において、遺伝的・家族集積性により発症リスクが上昇する疾患を、"
        "下記の疾患リストから選んでください。"
    ),
}

SYSTEM_PROMPT = """あなたは臨床医です。指定された条件に基づいて、関連する疾患を疾患リストから選択してください。

出力形式（厳守）:
疾患名を読点（、）区切りで1行で出力。

例:
気管支喘息、COPD急性増悪、肺血栓塞栓症、褐色細胞腫

推論過程・説明・番号は一切不要。疾患名リストのみ出力せよ。"""


def parse_disease_list(text, valid_names):
    """LLM出力から疾患名リストを抽出。"""
    # thinking model: 最終ブロックを使用
    if "\n\n" in text:
        blocks = text.split("\n\n")
        for block in reversed(blocks):
            names = re.split(r"[、,\n]", block.strip())
            cleaned = [n.strip().strip("・-*■●•0123456789. ") for n in names]
            matched = [n for n in cleaned if n in valid_names]
            if len(matched) >= 2:
                return matched

    # 通常パース
    names = re.split(r"[、,\n]", text.strip())
    cleaned = [n.strip().strip("・-*■●•0123456789. ") for n in names]
    return [n for n in cleaned if n in valid_names]


def main():
    from openai import OpenAI

    print("=" * 80)
    print("Type R HPE逆引きインデックス構築（順方向: 項目→疾患）")
    print("=" * 80)

    hpe_items = load_jsonl(HPE_JSONL)
    diseases = load_jsonl(DISEASES_JSONL)
    disease_names = [d["disease_name"] for d in diseases]
    valid_set = set(disease_names)

    type_r = [item for item in hpe_items if item["subcategory"] in TYPE_R_SUBCATS]
    print(f"\nType R項目: {len(type_r)}件")
    print(f"疾患マスタ: {len(diseases)}件")

    # Phase 1: テキスト検索
    print("\n--- Phase 1: テキスト検索 ---")
    text_results = {}
    for item in type_r:
        name = item["item_name"]
        keywords = SEARCH_KEYWORDS.get(name, [name])
        matches = text_search(diseases, keywords)
        text_results[name] = matches
        print(f"  {name}: {len(matches)}件")

    total_text = sum(len(v) for v in text_results.values())
    print(f"\nテキスト検索合計: {total_text}件 (平均{total_text/len(type_r):.1f}件/項目)")

    # 進捗復元
    results = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"\n進捗復元: {len(results)}項目処理済み")

    remaining = [item for item in type_r if item["item_name"] not in results]
    print(f"未処理: {len(remaining)}項目")

    if not remaining:
        print("全項目処理済み → Phase 2スキップ")
    else:
        # Phase 2: LLM検証・補完
        print(f"\n--- Phase 2: LLM検証・補完 ({len(remaining)}項目) ---")
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        min_interval = 60.0 / RPM_LIMIT

        disease_list_text = "\n".join(f"{i}: {n}" for i, n in enumerate(disease_names))
        last_call = 0
        success = 0
        failed = 0

        for idx, item in enumerate(remaining):
            name = item["item_name"]
            subcat = item["subcategory"]
            text_matches = text_results.get(name, [])

            prompt_intro = PROMPTS_BY_SUBCAT.get(subcat, "関連する疾患を選んでください。")
            text_hint = ""
            if text_matches:
                text_hint = (
                    f"\n\n参考（テキスト検索ヒット {len(text_matches)}件）:\n"
                    + "、".join(text_matches[:30])
                    + ("\n..." if len(text_matches) > 30 else "")
                    + "\n\n上記は参考です。リストにない疾患も含め、該当する全疾患を選んでください。"
                )

            user_content = (
                f"項目: {name} ({subcat})\n\n"
                f"{prompt_intro}"
                f"{text_hint}\n\n"
                f"【疾患リスト（527件）】\n{disease_list_text}"
            )

            # Rate limiting
            now = time.time()
            wait = min_interval - (now - last_call)
            if wait > 0:
                time.sleep(wait)

            for attempt in range(3):
                try:
                    last_call = time.time()
                    response = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=0.1,
                        max_tokens=8192,
                    )
                    text = response.choices[0].message.content
                    parsed = parse_disease_list(text, valid_set)

                    # テキスト検索結果とマージ（union）
                    merged = sorted(set(parsed) | set(text_matches))
                    results[name] = merged
                    success += 1
                    print(f"  [{idx+1}/{len(remaining)}] {name}: "
                          f"text={len(text_matches)}, LLM={len(parsed)}, merged={len(merged)}")
                    break

                except Exception as e:
                    if attempt < 2:
                        print(f"  [{name}] retry {attempt+1}: {e}")
                        time.sleep(min_interval)
                    else:
                        print(f"  [{name}] FAILED: {e}")
                        results[name] = text_matches  # フォールバック
                        failed += 1

            # 10件ごとに進捗保存
            if (idx + 1) % 10 == 0 or idx == len(remaining) - 1:
                with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False)

        print(f"\nPhase 2完了: {success}成功, {failed}失敗")

    # 統計
    print("\n--- 統計 ---")
    import numpy as np
    counts = [len(v) for v in results.values()]
    print(f"  項目数: {len(results)}")
    print(f"  疾患数: mean={np.mean(counts):.1f}, median={np.median(counts):.0f}, "
          f"min={min(counts)}, max={max(counts)}")

    zero_items = [k for k, v in results.items() if len(v) == 0]
    if zero_items:
        print(f"  0疾患: {zero_items}")

    # Top/Bottom
    sorted_items = sorted(results.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"\n  Top10:")
    for name, dlist in sorted_items[:10]:
        print(f"    {name}: {len(dlist)}疾患")
    print(f"  Bottom5:")
    for name, dlist in sorted_items[-5:]:
        print(f"    {name}: {len(dlist)}疾患")

    # 保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n保存完了: {OUTPUT_FILE}")

    # 進捗ファイル削除
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)


if __name__ == "__main__":
    main()
