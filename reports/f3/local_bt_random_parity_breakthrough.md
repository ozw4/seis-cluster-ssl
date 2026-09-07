# F3 Local Barlow Twins の Random parity 到達記録

- 日付: 2026-09-04
- 対象: [121 `gauss010_r221`](../../experiments/f3/facies_benchmark_v2/121_local_bt_view_region_search_v1/10_pretraining/gauss010_r221_3ep.yaml)と
  [122 nuisance/region ascent](../../experiments/f3/facies_benchmark_v2/122_local_bt_nuisance_region_ascent_v1/README.md)
- 状態: 当時の Phase C 到達点を記録した完了レポート。現在の選択処方は
  [noise/rotation search report](local_bt_rot90_asymmetric_noise_recipe.md)に集約する。
- 結論(3 点、証拠の強さを明示):
  1. **`gauss010_r221`(flip + gaussian σ0.10 + region window [2,2,1]、3 epochs)は
     large 条件で Random encoder を有意に上回る**(+0.026479、5 layout 全勝、
     t(4)=+4.54)。独立な学習 run(proj1536 版)でも再現し、プールで
     +0.022364、9/10 勝ち、t=+4.99。
  2. **この超過は処方に帰属する**。同予算の素の 3ep BT は large で −0.006225
     (t=−0.73、Random と区別できない)であり、処方の paired 増分は
     **+0.032704、5/5 勝ち、t(4)=+3.60**。
  3. **全 15 セル平均では Random と同水準**(+0.005190、t(14)=+0.66、10/15 勝ち)。
     small では依然として負(−0.019811)で、効果は size に対して単調
     (small < medium < large)。

  凍結契約(encoder / LR / optimizer / 下流 voxel decoder)は一切変更していない。

## Phase C 結果(medium 5 layouts、macro_f1 on unique validation voxels)

| layout | random | candidate | delta |
|---|---:|---:|---:|
| layout_000 | 0.528652 | 0.515345 | −0.013307 |
| layout_001 | 0.496541 | 0.484280 | −0.012261 |
| layout_002 | 0.523814 | 0.550812 | **+0.026998** |
| layout_003 | 0.534262 | 0.572258 | **+0.037996** |
| layout_004 | 0.477452 | 0.482539 | **+0.005087** |

**mean +0.008902 / median +0.005087 / wins 3/5 → gate(mean>0 かつ median>0 かつ
wins ≥ 3/5)PASS**。全セルで evaluation_voxel_count = 470,136 が一致し、
valid-token mask SHA は canonical random と同一。checkpoint audit も通過
(3 epochs / 1,875 global steps / resume_count 0)。

比較(従来の BT 系 five-way v3 canonical): medium **−0.052**。本処方は
medium 5 layout 平均 +0.0089 で符号が反転したが、上記のとおり有意差はない。

## 全 15 セル(small / medium / large × 5 layouts、canonical random 比)

| layout | small | medium | large |
|---|---:|---:|---:|
| layout_000 | −0.064083 | −0.013307 | +0.023231 |
| layout_001 | −0.030271 | −0.012261 | +0.038322 |
| layout_002 | −0.033589 | +0.026998 | +0.036107 |
| layout_003 | +0.009205 | +0.037996 | +0.005718 |
| layout_004 | +0.019682 | +0.005087 | +0.029017 |

| size | mean | median | wins | t(4) | gate |
|---|---:|---:|---:|---:|---|
| small | −0.019811 | −0.030271 | 2/5 | −1.30 | FAIL |
| medium | +0.008902 | +0.005087 | 3/5 | +0.86 | PASS |
| **large** | **+0.026479** | +0.029017 | **5/5** | **+4.54** | **PASS** |
| 全 15 セル | +0.005190 | +0.009205 | 10/15 | +0.66 | — |

従来の canonical BT five-way(small −0.061 / medium −0.052 / large −0.047)からの
改善は **+0.041 / +0.061 / +0.073** で、**全 size で BT を改善**している。

### large での独立再現と帰属

| large、5 layouts | mean | median | wins | t |
|---|---:|---:|---:|---:|
| 処方(proj384) | +0.026479 | +0.029017 | 5/5 | +4.54 |
| 処方(proj1536、独立な学習 run) | +0.018248 | +0.016190 | 4/5 | +2.63 |
| **プール(2 run × 5 layouts)** | **+0.022364** | +0.022823 | **9/10** | **+4.99** |
| 素の 3ep BT(control) | −0.006225 | −0.016551 | 2/5 | −0.73 |
| **処方の増分(paired)** | **+0.032704** | — | **5/5** | **+3.60** |

projector 幅が違えば初期化もパラメータ数も異なる別軌跡であり、その 2 本が
どちらも単独で有意。単一 run の偶然や評価パイプラインの偏りでは説明できない。

## 効果の帰属(medium 5 layouts、paired)

| 段階 | mean | median | wins | t(4) | gate |
|---|---:|---:|---:|---:|---|
| 素の 3ep BT(119 `legacy`) | −0.011172 | −0.006297 | 2/5 | −1.38 | FAIL |
| + 雑音(`gauss010`) | +0.003820 | +0.009402 | 3/5 | +0.34 | PASS |
| + 雑音 + region(`gauss010_r221`) | +0.008902 | +0.005087 | 3/5 | +0.86 | PASS |

paired 増分(layout 難易度が相殺されるため検出力が高い):

| 効果 | mean | sem | t(4) | wins |
|---|---:|---:|---:|---:|
| 雑音 | +0.014992 | 0.010618 | +1.41 | 4/5 |
| region | +0.005082 | 0.012462 | +0.41 | 4/5 |
| **処方全体(legacy → 雑音+region)** | **+0.020075** | 0.007188 | **+2.79** | **5/5** |

**処方全体の効果は t(4)=2.79(p ≈ 0.05)かつ 5 layout 全勝**であり、同一予算
(3 epochs)での素の BT に対する改善は統計的に支持される。一方、**改善後の
絶対水準が Random を上回るかは非有意**(+0.0089 ± 0.0103、t=0.86)。

したがって本実験群の結論は二段構えになる:

1. **BT を壊していた要因を特定し、その分を有意に取り戻した**(+0.0201、5/5)。
2. **取り戻した先は Random と同水準**であり、上回ったとは言えない。

## 統計的評価(重要)

| 比較 | mean | sd | sem | t(4) |
|---|---:|---:|---:|---:|
| `r221` vs random | +0.008902 | 0.023074 | 0.010319 | **+0.86** |
| 雑音のみ vs random | +0.003820 | 0.025183 | 0.011262 | +0.34 |
| region 効果(paired) | +0.005082 | 0.027866 | 0.012462 | +0.41 |
| projector 1536 効果(paired) | −0.000638 | 0.010096 | 0.004515 | −0.14 |
| window [2,2,2] 効果(paired) | −0.012790 | 0.026688 | 0.011935 | −1.07 |

layout 間のばらつき(sd 0.023)が効果量(0.009)の 2.6 倍あり、**t(4)=0.86
(p ≈ 0.44)で有意ではない**。

さらに、事前登録 gate(mean>0 かつ median>0 かつ wins ≥ 3/5)は **帰無仮説下でも
約 50% 通過する弱い基準**である(5 セルで「3 勝以上かつ中央値 > 0」は効果ゼロでも
概ね半々)。gate を事前に決めていた手続き自体は正しいが、**通過は「Random を
超えたことの証明」ではなく「超えている可能性と整合する」までの意味**しかない。

この弱い gate の解釈は、上に掲載した small / medium / large × 5 layout の
15 セル評価で解消した。medium 5 セルの通過だけを Random 超過の証明とは扱わない。

## 処方の定義

設定の正典は
[121 の YAML](../../experiments/f3/facies_benchmark_v2/121_local_bt_view_region_search_v1/10_pretraining/gauss010_r221_3ep.yaml)である。
encoder・LR・optimizer・下流 decoder 契約は比較間で不変。

## 機構(実測に基づく)

0. **効果の帰属(5 layout paired)**: 素の 3ep BT(雑音も region も無し)は
   明確に負、雑音を加えると mean +0.0038(gate PASS)、さらに region を加えると
   mean +0.0089(gate PASS、region 効果は paired で +0.0051・4/5 勝ち)。
   ただしいずれの差も有意水準には達しない。
1. **構造的不変性は有害、非構造的 nuisance 不変性は無害〜有益**
   flip / D4 / subcrop shift / token 遮蔽 / identity 化はいずれも −0.05〜−0.09
   (121・122)。加法ガウス雑音のみが下流を改善した。
2. **雑音 × region の超加法性**(layout_001 で測定): 雑音単独 +0.004、
   region 単独 −0.010 に対し、両方で +0.024(相互作用 +0.030)。region 平均が
   雑音を 1/√W 抑圧するため、「token 詳細を保ったまま雑音のみ無視する」解が
   最安になる。
3. **σ・λ は内点最適**: σ = {0.05, 0.10, 0.20} → {−0.017, −0.012, −0.022}、
   λ = {0.00125, 0.005} → {−0.017, −0.012}(いずれも layout_001)。
4. **epoch 延長は有害**だった。詳細な証拠は
   [epoch scaling report](local_bt_epoch_scaling_v1.md)に集約する。
5. **effective rank は完全に非律速**(6 回確認)。本処方の rank は 16.77 で
   Random(24.00)を大きく下回るのに下流は上回る。

## 方法論上の教訓(重要)

単一 cell(layout_001)screen の分解能は **±0.015 程度**しかなく、layout 間の
delta 幅は −0.025〜+0.038 に及ぶ。実際、

| 処方 | layout_001 | 5 layout mean | gate |
|---|---:|---:|---|
| `gauss010_r221` [2,2,1] | −0.012261 | **+0.008902** | **PASS** |
| `gauss010_r222` [2,2,2] | +0.011088 | −0.003888 | FAIL |

と **単一 cell と 5 layout で順位が逆転**した。window 掃引で [2,2,2] を「勝者」と
判定したのは winner's curse であり、5 layout paired では window 効果は
**−0.012790**(2/5 勝ち)で [2,2,1] の方が良い。**今後の arm 選抜は 5 layout で
行うこと**。単一 cell は候補の粗ふるいにのみ使う。

## 位置づけ

small / large、独立 run、region 無し対照の評価は完了し、上の表へ反映済みである。
この処方は探索途中の歴史的 milestone であり、後続の 123 によって選択処方は更新
された。後続値はここへ複製せず、冒頭の正典レポートを参照する。
