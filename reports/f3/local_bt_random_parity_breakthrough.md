# F3 Local Barlow Twins の Random parity 到達記録

- 日付: 2026-09-04
- 対象: [121 `gauss010_r221`](../../experiments/f3/facies_benchmark_v2/121_local_bt_view_region_search_v1/10_pretraining/gauss010_r221_3ep.yaml)と
  [122 nuisance/region ascent](../../experiments/f3/facies_benchmark_v2/122_local_bt_nuisance_region_ascent_v1/README.md)
- 処方定義: flip + gaussian σ0.10 + region window [2,2,1]、3 epochs。設定の正典は
  [121 の YAML](../../experiments/f3/facies_benchmark_v2/121_local_bt_view_region_search_v1/10_pretraining/gauss010_r221_3ep.yaml)。
- 指標: macro_f1 on unique validation voxels。delta = candidate − canonical Random encoder(正が良い)。
- 評価セル: small / medium / large × 5 layouts(layout_000〜layout_004)。統計単位は layout。
  t(4) は 5 layout の t、全 15 セル行の t は t(14)。
- gate 定義: mean>0 かつ median>0 かつ wins ≥ 3/5。
- 全セルで evaluation_voxel_count = 470,136。valid-token mask SHA は canonical random と同一。
- checkpoint audit: 3 epochs / 1,875 global steps / resume_count 0。
- encoder / LR / optimizer / 下流 voxel decoder: 比較間で不変。
- 現在の選択処方(後続 123 で更新)は
  [noise/rotation search report](local_bt_rot90_asymmetric_noise_recipe.md)に記録する。

## Phase C 結果(medium 5 layouts、macro_f1 on unique validation voxels)

| layout | random | candidate | delta |
|---|---:|---:|---:|
| layout_000 | 0.528652 | 0.515345 | −0.013307 |
| layout_001 | 0.496541 | 0.484280 | −0.012261 |
| layout_002 | 0.523814 | 0.550812 | **+0.026998** |
| layout_003 | 0.534262 | 0.572258 | **+0.037996** |
| layout_004 | 0.477452 | 0.482539 | **+0.005087** |

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

従来の canonical BT five-way v3(Random 比 delta)と本処方の改善幅:

| size | canonical BT five-way v3 | 本処方の改善 |
|---|---:|---:|
| small | −0.061 | +0.041 |
| medium | −0.052 | +0.061 |
| large | −0.047 | +0.073 |

### large での独立再現と帰属

| large、5 layouts | mean | median | wins | t |
|---|---:|---:|---:|---:|
| 処方(proj384) | +0.026479 | +0.029017 | 5/5 | +4.54 |
| 処方(proj1536、独立な学習 run) | +0.018248 | +0.016190 | 4/5 | +2.63 |
| **プール(2 run × 5 layouts)** | **+0.022364** | +0.022823 | **9/10** | **+4.99** |
| 素の 3ep BT(control) | −0.006225 | −0.016551 | 2/5 | −0.73 |
| **処方の増分(paired)** | **+0.032704** | — | **5/5** | **+3.60** |

## 効果の帰属(medium 5 layouts、paired)

| 段階 | mean | median | wins | t(4) | gate |
|---|---:|---:|---:|---:|---|
| 素の 3ep BT(119 `legacy`) | −0.011172 | −0.006297 | 2/5 | −1.38 | FAIL |
| + 雑音(`gauss010`) | +0.003820 | +0.009402 | 3/5 | +0.34 | PASS |
| + 雑音 + region(`gauss010_r221`) | +0.008902 | +0.005087 | 3/5 | +0.86 | PASS |

paired 増分(medium 5 layouts):

| 効果 | mean | sem | t(4) | wins |
|---|---:|---:|---:|---:|
| 雑音 | +0.014992 | 0.010618 | +1.41 | 4/5 |
| region | +0.005082 | 0.012462 | +0.41 | 4/5 |
| **処方全体(legacy → 雑音+region)** | **+0.020075** | 0.007188 | **+2.79** | **5/5** |

## 統計的評価

| 比較 | mean | sd | sem | t(4) |
|---|---:|---:|---:|---:|
| `r221` vs random | +0.008902 | 0.023074 | 0.010319 | **+0.86** |
| 雑音のみ vs random | +0.003820 | 0.025183 | 0.011262 | +0.34 |
| region 効果(paired) | +0.005082 | 0.027866 | 0.012462 | +0.41 |
| projector 1536 効果(paired) | −0.000638 | 0.010096 | 0.004515 | −0.14 |
| window [2,2,2] 効果(paired) | −0.012790 | 0.026688 | 0.011935 | −1.07 |

- 全行 medium 5 layouts、統計単位は layout(t(4))。

## 単一 cell / 補助観測

- 雑音 × region(layout_001 で測定): 雑音単独 +0.004、region 単独 −0.010、両方 +0.024(相互作用 +0.030)。
- σ 掃引(layout_001、Random 比 delta): σ = {0.05, 0.10, 0.20} → {−0.017, −0.012, −0.022}。
- λ 掃引(layout_001、Random 比 delta): λ = {0.00125, 0.005} → {−0.017, −0.012}。
- effective rank(絶対値、6 回確認): 本処方 16.77、Random 24.00。
- epoch 数の比較は [epoch scaling report](local_bt_epoch_scaling_v1.md)に記録する。

## 単一 cell vs 5 layout(`gauss010_r221` vs `gauss010_r222`、medium)

| 処方 | layout_001 | 5 layout mean | gate |
|---|---:|---:|---|
| `gauss010_r221` [2,2,1] | −0.012261 | **+0.008902** | **PASS** |
| `gauss010_r222` [2,2,2] | +0.011088 | −0.003888 | FAIL |

- 単一 cell(layout_001)と 5 layout で順位が逆転した。
- window [2,2,2] の paired 効果(5 layout): −0.012790、2/5 勝ち。採用は [2,2,1]。
