"""
段階的検証: engine.update_from_results() の統合テスト
各ケースで 初回(症状のみ) vs 旧方式(全再embed) vs Option3(engine統合) を比較
"""
from engine import VeSMedEngine

print("エンジン初期化中...")
eng = VeSMedEngine()


def find_rank(candidates, disease_name, key='similarity'):
    sorted_c = sorted(candidates, key=lambda x: x.get(key, 0), reverse=True)
    for i, c in enumerate(sorted_c):
        if c['disease_name'] == disease_name:
            return i + 1
    return -1


def run_case(case_name, symptoms, results, targets, wrong_targets=None):
    """1症例を実行: 初回 vs 旧方式 vs Option3(engine統合)"""
    print(f"\n{'='*70}")
    print(f"  {case_name}")
    print(f"{'='*70}")

    # 初回（症状のみ）
    cands = eng.search_diseases(symptoms)
    cands = eng.compute_priors(cands)

    # 旧方式（全再embed）
    full = symptoms + '\n' + '\n'.join(results)
    old = eng.search_diseases(full)

    # Option 3（engine統合: update_from_results、基準範囲モード）
    import copy
    cands_opt3 = copy.deepcopy(cands)
    cands_opt3 = eng.update_from_results(cands_opt3, results, symptoms=symptoms, mode="fast")

    # 結果表示
    print(f"\n  {'疾患':<25s} | {'初回':>4s} | {'旧':>4s} | {'新':>4s} | 判定")
    print(f"  {'-'*60}")

    all_targets = targets + (wrong_targets or [])
    for name, label in all_targets:
        r_init = find_rank(cands, name)
        r_old = find_rank(old, name)
        r_new = find_rank(cands_opt3, name)

        # 判定
        is_wrong = wrong_targets and (name, label) in wrong_targets
        if is_wrong:
            if r_new > r_init:
                judge = "OK 正しく沈下"
            elif r_new < r_init:
                judge = "NG 誤って浮上"
            else:
                judge = "-- 変化なし"
        else:
            if r_new < r_init:
                judge = "OK 正しく浮上"
            elif r_new > r_init:
                judge = "NG 誤って沈下"
            else:
                judge = "-- 変化なし"

        print(f"  {label:<25s} | {r_init:>4d} | {r_old:>4d} | {r_new:>4d} | {judge}")

    return cands, old, cands_opt3


# ================================================================
# ケースA: 定性的な正常結果（Step 1で解決すべき）
# ================================================================
run_case(
    "ケースA: 定性的な正常結果（γ-GTP正常値、D-Bil正常値、MRCP異常なし）",
    """47歳女性。来院1日前から左脇～背中の痛み、全身痛、嘔吐、水様便。当日39度の発熱。""",
    ['γ-GTP正常値', '直接ビリルビン正常値', 'MRCP異常なし'],
    targets=[
        ('劇症型溶血性レンサ球菌感染症（STSS）', 'STSS(正解)'),
        ('壊死性筋膜炎', '壊死性筋膜炎(正解)'),
    ],
    wrong_targets=[
        ('総胆管結石', '総胆管結石(誤)'),
        ('急性胆管炎', '急性胆管炎(誤)'),
    ],
)

# ================================================================
# ケースB: 定性的な異常結果（Step 1で解決すべき）
# ================================================================
run_case(
    "ケースB: 定性的な異常結果（血液培養陽性、心エコー疣贅、CRP上昇）",
    """67歳男性。7週間前から38度前後の繰り返す発熱。体重減少、食欲低下、全身倦怠感。""",
    ['血液培養: 黄色ブドウ球菌陽性', '心エコー: 大動脈弁に疣贅あり', 'CRP上昇'],
    targets=[
        ('感染性心内膜炎', '感染性心内膜炎(正解)'),
        ('敗血症', '敗血症(正解寄り)'),
    ],
    wrong_targets=[
        ('成人スティル病', 'スティル病(誤寄り)'),
    ],
)

# ================================================================
# ケースC: 半定量（+/++/+++）（Step 1で解決すべき）
# ================================================================
run_case(
    "ケースC: 半定量（尿蛋白+++、尿潜血+）",
    """45歳男性。2週間前から両下肢の浮腫が進行。顔面浮腫も出現。泡立つ尿に気づいた。""",
    ['尿蛋白(+++)', '尿潜血(+)'],
    targets=[
        ('ネフローゼ症候群', 'ネフローゼ(正解)'),
        ('IgA腎症', 'IgA腎症(正解寄り)'),
    ],
    wrong_targets=[
        ('深部静脈血栓症', 'DVT(誤寄り)'),
    ],
)

# ================================================================
# ケースD: 数値のみ（Step 1では不完全、Step 2で解決すべき）
# ================================================================
run_case(
    "ケースD: 数値のみ（WBC 18000, CRP 15）— Step 1の限界を確認",
    """47歳女性。来院1日前から左脇～背中の痛み、全身痛、嘔吐、水様便。当日39度の発熱。""",
    ['WBC 18000', 'CRP 15'],
    targets=[
        ('劇症型溶血性レンサ球菌感染症（STSS）', 'STSS(正解)'),
        ('敗血症', '敗血症(正解寄り)'),
    ],
    wrong_targets=[],
)

# ================================================================
# ケースE: 数値で逆方向の異常（Step 1では失敗、Step 2で解決すべき）
# ================================================================
run_case(
    "ケースE: 逆方向異常（WBC 2000, 体温 35度）— 重症感染で低下",
    """65歳男性。3日前から発熱と悪寒。本日意識レベル低下、血圧低下で搬送。""",
    ['WBC 2000', '体温 35度'],
    targets=[
        ('敗血症', '敗血症(正解)'),
        ('敗血症性ショック', '敗血症性ショック(正解)'),
    ],
    wrong_targets=[],
)

# ================================================================
# ケースF: 数値で正常だが極性が曖昧（Step 1での挙動確認）
# ================================================================
run_case(
    "ケースF: 正常数値（Na 140, Hb 14）— 極性が正に誤判定される可能性",
    """55歳男性。胸痛で来院。30分前から突然の胸骨後部痛。冷汗あり。""",
    ['Na 140', 'Hb 14'],
    targets=[
        ('急性心筋梗塞', 'AMI(正解)'),
    ],
    wrong_targets=[],
)

# ================================================================
# ケースG: 定性的な結果の混合（正常+異常）
# ================================================================
run_case(
    "ケースG: 混合（血液培養陽性 + 胸部X線異常なし + 尿培養陰性）",
    """72歳男性。5日前から38度台の発熱持続。悪寒戦慄を伴う。""",
    ['血液培養: 黄色ブドウ球菌陽性', '胸部X線異常なし', '尿培養陰性'],
    targets=[
        ('感染性心内膜炎', 'IE(正解)'),
        ('敗血症', '敗血症(正解寄り)'),
    ],
    wrong_targets=[
        ('肺炎球菌性肺炎', '肺炎(誤寄り)'),
        ('急性腎盂腎炎', '腎盂腎炎(誤寄り)'),
    ],
)


print(f"\n\n{'='*70}")
print("Step 1 検証完了（engine統合版）")
print("OK = Step 1で正しく動作")
print("NG = Step 1では不十分（Step 2: 基準範囲テーブルで改善すべき）")
print(f"{'='*70}")
