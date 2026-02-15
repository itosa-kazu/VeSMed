"""
B2アプローチ（直接embedding、中心化なし）の包括的検証
30ケース: 各診療科・結果タイプ・エッジケースを網羅

比較対象:
  Base: 初回検索のみ（検査結果なし）
  A:    sim_matrix + 列平均除去（現行engine実装）
  B2:   直接embedding、中心化なし
"""
from engine import VeSMedEngine
from config import EMBEDDING_MODEL
import numpy as np
import copy

print("エンジン初期化中...")
eng = VeSMedEngine()

# 疾患embeddingをChromaDBから取得（B2用）
print("疾患embedding取得中...")
n_diseases = eng.collection.count()
all_ids = [f"disease_{i}" for i in range(n_diseases)]
disease_embs = {}
for start in range(0, len(all_ids), 100):
    batch_ids = all_ids[start:start + 100]
    result = eng.collection.get(ids=batch_ids, include=["embeddings", "metadatas"])
    for j, mid in enumerate(result["ids"]):
        dname = result["metadatas"][j].get("disease_name", "")
        emb = np.array(result["embeddings"][j], dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        disease_embs[dname] = emb
print(f"疾患embedding: {len(disease_embs)}件\n")


def approach_a(cands, result_lines, symptoms):
    c = copy.deepcopy(cands)
    return eng.update_from_results(c, result_lines, symptoms=symptoms, mode="fast")


def approach_b2(cands, result_lines, symptoms):
    c = copy.deepcopy(cands)
    if not result_lines or eng.polarity_axis is None:
        return c
    annotated = eng._annotate_with_ranges(result_lines)
    resp = eng.embed_client.embeddings.create(model=EMBEDDING_MODEL, input=annotated)
    line_embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    l_norms = np.linalg.norm(line_embs, axis=1, keepdims=True)
    l_norms[l_norms == 0] = 1.0
    line_embs = line_embs / l_norms

    for k in range(len(result_lines)):
        lemb = line_embs[k]
        pol = float(lemb @ eng.polarity_axis)
        sign = 1.0 if pol > 0 else -1.0
        for ci in c:
            demb = disease_embs.get(ci['disease_name'])
            if demb is not None:
                rel = float(lemb @ demb)
                ci['similarity'] *= float(np.exp(sign * rel))
    c.sort(key=lambda x: x['similarity'], reverse=True)
    return c


def find_rank(candidates, disease_name):
    sorted_c = sorted(candidates, key=lambda x: x.get('similarity', 0), reverse=True)
    for i, c in enumerate(sorted_c):
        if c['disease_name'] == disease_name:
            return i + 1
    return -1


# ================================================================
# テストケース定義
# direction: "up"=浮上すべき, "down"=沈下すべき, "stay"=維持すべき
# ================================================================
CASES = [
    # ============================================================
    # 1. 定性的結果（既存改良）
    # ============================================================
    {
        "id": "01", "name": "定性正常: 肝胆道除外",
        "symptoms": "47歳女性。来院1日前から左脇～背中の痛み、全身痛、嘔吐、水様便。当日39度の発熱。",
        "results": ['γ-GTP正常値', '直接ビリルビン正常値', 'MRCP異常なし'],
        "targets": [
            ('劇症型溶血性レンサ球菌感染症（STSS）', 'STSS', 'up'),
        ],
        "wrong": [
            ('総胆管結石', '総胆管結石', 'down'),
            ('急性胆管炎', '急性胆管炎', 'down'),
        ],
    },
    {
        "id": "02", "name": "定性異常: IE確定",
        "symptoms": "67歳男性。7週間前から38度前後の繰り返す発熱。体重減少、食欲低下、全身倦怠感。",
        "results": ['血液培養: 黄色ブドウ球菌陽性', '心エコー: 大動脈弁に疣贅あり', 'CRP上昇'],
        "targets": [('感染性心内膜炎', 'IE', 'up')],
        "wrong": [('成人スティル病', 'スティル病', 'down')],
    },
    {
        "id": "03", "name": "半定量: 尿蛋白+++",
        "symptoms": "45歳男性。2週間前から両下肢の浮腫が進行。顔面浮腫も出現。泡立つ尿に気づいた。",
        "results": ['尿蛋白(+++)', '尿潜血(+)'],
        "targets": [('微小変化型ネフローゼ症候群', 'ネフローゼ', 'stay'), ('IgA腎症', 'IgA腎症', 'up')],
        "wrong": [('深部静脈血栓症', 'DVT', 'down')],
    },
    {
        "id": "04", "name": "混合: 血培陽性 + X線/尿培陰性",
        "symptoms": "72歳男性。5日前から38度台の発熱持続。悪寒戦慄を伴う。",
        "results": ['血液培養: 黄色ブドウ球菌陽性', '胸部X線異常なし', '尿培養陰性'],
        "targets": [('感染性心内膜炎', 'IE', 'up'), ('敗血症', '敗血症', 'stay')],
        "wrong": [('肺炎球菌性肺炎', '肺炎', 'down'), ('急性腎盂腎炎', '腎盂腎炎', 'down')],
    },

    # ============================================================
    # 2. 数値結果（基準範囲アノテーション）
    # ============================================================
    {
        "id": "05", "name": "数値高値: WBC/CRP感染",
        "symptoms": "47歳女性。来院1日前から左脇～背中の痛み、全身痛、嘔吐、水様便。当日39度の発熱。",
        "results": ['WBC 18000', 'CRP 15'],
        "targets": [('劇症型溶血性レンサ球菌感染症（STSS）', 'STSS', 'up'), ('敗血症', '敗血症', 'up')],
        "wrong": [],
    },
    {
        "id": "06", "name": "逆方向異常: WBC低下+低体温",
        "symptoms": "65歳男性。3日前から発熱と悪寒。本日意識レベル低下、血圧低下で搬送。",
        "results": ['WBC 2000', '体温 35度'],
        "targets": [('敗血症', '敗血症', 'stay'), ('敗血症性ショック', '敗血症性ショック', 'up')],
        "wrong": [],
    },
    {
        "id": "07", "name": "正常数値: Na/Hbが無関係に影響しない",
        "symptoms": "55歳男性。胸痛で来院。30分前から突然の胸骨後部痛。冷汗あり。",
        "results": ['Na 140', 'Hb 14'],
        "targets": [('急性心筋梗塞 (STEMI：前壁/中隔)', 'AMI', 'stay')],
        "wrong": [],
    },
    {
        "id": "08", "name": "数値混合: トロポニン上昇+正常",
        "symptoms": "65歳男性。2時間前から胸骨後部の激痛。冷汗、嘔気あり。",
        "results": ['トロポニンT 0.5', 'Na 141', 'Cr 0.8'],
        "targets": [('急性心筋梗塞 (STEMI：前壁/中隔)', 'AMI', 'stay')],
        "wrong": [('慢性腎臓病 (CKD)', 'CKD', 'down')],
    },

    # ============================================================
    # 3. バイタルサイン
    # ============================================================
    {
        "id": "09", "name": "バイタル数値: 発熱+低血圧+頻脈",
        "symptoms": "70歳男性。尿路感染後に高熱、意識混濁。",
        "results": ['体温 39.5', '血圧 80/50', '脈拍 120'],
        "targets": [('敗血症', '敗血症', 'up'), ('敗血症性ショック', '敗血症性ショック', 'up')],
        "wrong": [],
    },
    {
        "id": "10", "name": "バイタル定性: 低体温+低血圧+頻脈",
        "symptoms": "70歳男性。尿路感染後に高熱、意識混濁。",
        "results": ['低体温', '低血圧', '頻脈'],
        "targets": [('敗血症', '敗血症', 'up'), ('敗血症性ショック', '敗血症性ショック', 'up')],
        "wrong": [],
    },

    # ============================================================
    # 4. 肝機能・腎機能・甲状腺
    # ============================================================
    {
        "id": "11", "name": "肝機能: AST/ALT/Bil上昇",
        "symptoms": "35歳男性。1週間前から倦怠感、食欲不振。3日前から眼球黄染。",
        "results": ['AST 450', 'ALT 380', 'T-Bil 5.2'],
        "targets": [('急性B型肝炎', '急性肝炎', 'up')],
        "wrong": [],
    },
    {
        "id": "12", "name": "腎機能: Cr/BUN/K上昇",
        "symptoms": "60歳男性。糖尿病歴20年。1週間前から全身浮腫、尿量減少。",
        "results": ['Cr 4.5', 'BUN 65', 'K 6.2'],
        "targets": [('慢性腎臓病 (CKD)', 'CKD', 'up'), ('糖尿病性腎症', '糖尿病性腎症', 'stay')],
        "wrong": [],
    },
    {
        "id": "13", "name": "甲状腺機能亢進: TSH低下+FT4上昇",
        "symptoms": "28歳女性。動悸、手指振戦、体重減少、発汗過多。眼球突出あり。",
        "results": ['TSH 0.01', 'Free T4 5.8'],
        "targets": [('バセドウ病', 'バセドウ病', 'stay')],
        "wrong": [],
    },
    {
        "id": "14", "name": "甲状腺機能低下: TSH上昇+FT4低下",
        "symptoms": "55歳女性。3ヶ月前から全身倦怠感、寒がり、体重増加、便秘。顔面浮腫。",
        "results": ['TSH 15', 'Free T4 0.3'],
        "targets": [('橋本病', '橋本病', 'up')],
        "wrong": [('慢性心不全急性増悪', '心不全', 'down')],
    },

    # ============================================================
    # 5. 貧血・血液
    # ============================================================
    {
        "id": "15", "name": "鉄欠乏性貧血: Hb低下+フェリチン低下",
        "symptoms": "45歳女性。3ヶ月前から徐々に息切れ、動悸。月経過多の既往。",
        "results": ['Hb 6.5', 'フェリチン 3'],
        "targets": [('鉄欠乏性貧血', '鉄欠乏性貧血', 'up')],
        "wrong": [],
    },
    {
        "id": "16", "name": "DIC: PT延長+血小板低下+D-dimer上昇",
        "symptoms": "58歳男性。発熱と意識障害で搬送。全身に紫斑が出現。歯肉出血あり。",
        "results": ['血小板 3', 'D-dimer上昇', 'PT-INR 2.5'],
        "targets": [('播種性血管内凝固症候群 (DIC)', 'DIC', 'up')],
        "wrong": [],
    },

    # ============================================================
    # 6. 心臓系（AMI以外）
    # ============================================================
    {
        "id": "17", "name": "心不全: BNP上昇+胸部X線異常",
        "symptoms": "75歳男性。1週間前から労作時息切れが悪化。起座呼吸。両下肢浮腫。",
        "results": ['BNP 800', '胸部X線: 両側胸水あり、心拡大', 'トロポニンT正常'],
        "targets": [('慢性心不全急性増悪', '心不全', 'up')],
        "wrong": [('急性心筋梗塞 (STEMI：前壁/中隔)', 'AMI', 'down')],
    },
    {
        "id": "18", "name": "大動脈解離: D-dimer上昇+トロポニン陰性",
        "symptoms": "62歳男性。突然の激烈な背部痛。引き裂かれるような痛み。高血圧の既往。",
        "results": ['D-dimer上昇', 'トロポニンT陰性', 'CRP上昇'],
        "targets": [('急性大動脈解離 (Stanford A型)', '大動脈解離A', 'up')],
        "wrong": [('急性心筋梗塞 (STEMI：前壁/中隔)', 'AMI', 'down')],
    },
    {
        "id": "19", "name": "心膜炎: CRP上昇+トロポニン軽度上昇+心エコー心嚢液",
        "symptoms": "30歳男性。3日前から前胸部痛。前屈で軽快。1週間前に上気道炎。",
        "results": ['CRP上昇', 'トロポニンT軽度上昇', '心エコー: 心嚢液貯留あり'],
        "targets": [('急性心膜炎', '心膜炎', 'up')],
        "wrong": [],
    },

    # ============================================================
    # 7. 呼吸器・肺
    # ============================================================
    {
        "id": "20", "name": "肺塞栓: D-dimer上昇+SpO2低下+CXR正常",
        "symptoms": "45歳女性。突然の呼吸困難と胸痛。長時間フライト後。片側下肢腫脹あり。",
        "results": ['D-dimer上昇', 'SpO2 88', '胸部X線異常なし'],
        "targets": [('肺血栓塞栓症', '肺塞栓', 'up')],
        "wrong": [('肺炎球菌性肺炎', '肺炎', 'down')],
    },
    {
        "id": "21", "name": "日本語表記: 白血球/CRP/血小板",
        "symptoms": "50歳男性。3日前から高熱、咳嗽、膿性痰。呼吸困難増悪。",
        "results": ['白血球 25000', 'CRP 20', '血小板 3'],
        "targets": [('肺炎球菌性肺炎', '肺炎', 'up'), ('敗血症', '敗血症', 'up')],
        "wrong": [],
    },

    # ============================================================
    # 8. 自己免疫・膠原病
    # ============================================================
    {
        "id": "22", "name": "SLE: ANA陽性+抗dsDNA陽性+補体低下",
        "symptoms": "25歳女性。蝶形紅斑、関節痛、口腔内潰瘍。微熱が持続。",
        "results": ['抗核抗体陽性', '抗dsDNA抗体陽性', '補体低下'],
        "targets": [('全身性エリテマトーデス', 'SLE', 'up')],
        "wrong": [],
    },
    {
        "id": "23", "name": "関節リウマチ: RF陽性+抗CCP抗体陽性+CRP上昇",
        "symptoms": "50歳女性。両手の朝のこわばり1時間以上。MCP関節・PIP関節の対称性腫脹。",
        "results": ['リウマトイド因子陽性', '抗CCP抗体陽性', 'CRP上昇'],
        "targets": [('関節リウマチ', 'RA', 'up')],
        "wrong": [],
    },

    # ============================================================
    # 9. 内分泌（追加）
    # ============================================================
    {
        "id": "24", "name": "副腎クリーゼ: コルチゾール低下+Na低下+K上昇",
        "symptoms": "40歳男性。嘔吐、脱力、低血圧で搬送。色素沈着あり。ステロイド急な中止歴。",
        "results": ['Na 125', 'K 6.0', 'CRP上昇'],
        "targets": [('副腎クリーゼ', '副腎クリーゼ', 'up'), ('アジソン病', 'アジソン病', 'up')],
        "wrong": [],
    },

    # ============================================================
    # 10. 消化器
    # ============================================================
    {
        "id": "25", "name": "急性膵炎: CT膵腫大+リパーゼ/アミラーゼ上昇",
        "symptoms": "50歳男性。突然の上腹部痛、背部に放散。大量飲酒歴あり。",
        "results": ['腹部CT: 膵腫大あり', 'リパーゼ上昇', 'アミラーゼ上昇'],
        "targets": [('急性膵炎', '急性膵炎', 'stay')],
        "wrong": [('急性胆嚢炎', '急性胆嚢炎', 'down')],
    },
    {
        "id": "26", "name": "虫垂炎: WBC上昇+CRP上昇+CT虫垂腫大",
        "symptoms": "22歳男性。昨日から心窩部痛が右下腹部に移動。嘔気あり。微熱。",
        "results": ['WBC 15000', 'CRP 8', '腹部CT: 虫垂腫大あり'],
        "targets": [('急性虫垂炎', '虫垂炎', 'up')],
        "wrong": [('急性胆嚢炎', '胆嚢炎', 'down')],
    },

    # ============================================================
    # 11. 救急・重症
    # ============================================================
    {
        "id": "27", "name": "アナフィラキシー: トリプターゼ上昇+IgE上昇",
        "symptoms": "35歳男性。蜂刺傷後30分で全身蕁麻疹、呼吸困難、血圧低下。",
        "results": ['トリプターゼ上昇', 'IgE上昇', '血圧 70/40'],
        "targets": [('アナフィラキシーショック', 'アナフィラキシー', 'up')],
        "wrong": [],
    },

    # ============================================================
    # 12. 除外テスト（正常結果で順位変動なし）
    # ============================================================
    {
        "id": "28", "name": "トロポニン陰性: AMI除外",
        "symptoms": "45歳男性。胸痛で来院。2時間持続する胸骨後部痛。",
        "results": ['トロポニンT陰性', 'CRP正常'],
        "targets": [],
        "wrong": [('急性心筋梗塞 (STEMI：前壁/中隔)', 'AMI', 'down')],
    },
    {
        "id": "29", "name": "培養全陰性: 感染症除外",
        "symptoms": "40歳女性。2週間の不明熱。体重減少5kg。盗汗あり。",
        "results": ['血液培養陰性', '尿培養陰性', '喀痰培養陰性', '胸部X線異常なし'],
        "targets": [],
        "wrong": [('敗血症', '敗血症', 'down'), ('肺炎球菌性肺炎', '肺炎', 'down'), ('急性腎盂腎炎', '腎盂腎炎', 'down')],
    },
    {
        "id": "30", "name": "全正常: 安定性テスト（順位大変動なし）",
        "symptoms": "30歳男性。頭痛で来院。3日前から持続する頭痛。嘔気あり。",
        "results": ['WBC 6000', 'CRP 0.1', 'Na 140', 'K 4.0', 'Cr 0.8'],
        "targets": [('細菌性髄膜炎', '細菌性髄膜炎', 'stay')],
        "wrong": [],
    },
]


# ================================================================
# 実行
# ================================================================
print(f"{'='*100}")
print(f"  B2包括検証: {len(CASES)}ケース")
print(f"{'='*100}")

a_score = 0
b2_score = 0
a_wins = 0
b2_wins = 0
ties = 0

for case in CASES:
    cands = eng.search_diseases(case["symptoms"])
    cands = eng.compute_priors(cands)

    result_a = approach_a(cands, case["results"], case["symptoms"])
    result_b2 = approach_b2(cands, case["results"], case["symptoms"])

    all_targets = [(n, l, d, False) for n, l, d in case["targets"]] + \
                  [(n, l, d, True) for n, l, d in case["wrong"]]

    if not all_targets:
        continue

    print(f"\n  [{case['id']}] {case['name']}")

    for dname, label, direction, is_wrong in all_targets:
        r_init = find_rank(cands, dname)
        r_a = find_rank(result_a, dname)
        r_b2 = find_rank(result_b2, dname)

        def score_it(r_init, r_new, direction, is_wrong):
            if r_init <= 0 or r_new <= 0:
                return 0, "?"
            if is_wrong:
                if r_new > r_init:
                    return 1, "+"
                elif r_new < r_init:
                    return -1, "-"
                return 0, "="
            if direction == "up":
                if r_new < r_init:
                    return 1, "+"
                elif r_new > r_init:
                    return -1, "-"
                return 0, "="
            elif direction == "stay":
                if abs(r_new - r_init) <= 2:
                    return 1, "+"
                elif r_new > r_init:
                    return -1, "-"
                return 0, "="
            elif direction == "down":
                if r_new > r_init:
                    return 1, "+"
                elif r_new < r_init:
                    return -1, "-"
                return 0, "="
            return 0, "="

        sa, ma = score_it(r_init, r_a, direction, is_wrong)
        sb, mb = score_it(r_init, r_b2, direction, is_wrong)
        a_score += sa
        b2_score += sb

        # 勝敗カウント
        if sa > sb:
            a_wins += 1
        elif sb > sa:
            b2_wins += 1
        else:
            ties += 1

        r_init_s = f"{r_init:4d}" if r_init > 0 else " N/A"
        winner = ""
        if sa > sb:
            winner = "← A"
        elif sb > sa:
            winner = "← B2"

        print(f"    {label:<16s} | 初回{r_init_s} | A:{r_a:>4d}{ma} | B2:{r_b2:>4d}{mb} {winner}")


# ================================================================
# サマリー
# ================================================================
print(f"\n\n{'='*100}")
print(f"  サマリー（{len(CASES)}ケース）")
print(f"{'='*100}")
print(f"  スコア（+1=正しく変化, 0=変化なし, -1=誤った変化）:")
print(f"    A  (sim_matrix中心化):  {a_score:+d}")
print(f"    B2 (直接embedding生):   {b2_score:+d}")
print(f"")
print(f"  対戦成績（各疾患ペアで勝ち負け）:")
print(f"    A勝ち:  {a_wins}")
print(f"    B2勝ち: {b2_wins}")
print(f"    引分け: {ties}")
print(f"{'='*100}")
