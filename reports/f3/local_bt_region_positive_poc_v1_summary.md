# F3 Local BT region-positive PoC v1 — FAIL の記録

- 日付: 2026-09-03(GPU 実行同日)
- 実験: [120 Local BT region-positive PoC](../../experiments/f3/facies_benchmark_v2/120_local_bt_region_positive_poc_v1/README.md)
- 条件: positive pair を対応 1 token(8×8×8 voxel)ではなく 2×2×1 token block の平均
  (16×16×8 voxel の領域表現)とする region-positive arm(flip view)を 3 epochs 学習し、
  canonical five-way v3 の `medium / layout_001` で Random と 119 `legacy` arm に比較した。
- 符号規約: `delta vs Random` = candidate macro_f1 − Random macro_f1(正が良い)。

## 結果(Random macro_f1 = 0.496541 / eff. rank 24.00)

| source | macro_f1 | delta vs Random | eff. rank | raw norm | screen |
|---|---:|---:|---:|---:|---|
| `local_bt_region_w221_3ep` | 0.450633 | **−0.045909** | **16.75** | 61.4 | FAIL |
| (参照)119 `legacy` 3ep | 0.460224 | −0.036318 | 20.43 | 61.2 | FAIL |

- window 効果(region − legacy、macro_f1)= −0.009591。
- gate 判定: `BT_RP_SCREEN_FAIL`、`BT_RP_PHASE_B_ELIGIBLE: False`(delta ≤ −0.02)。
  Phase B(10ep 延長)は未実施。
- checkpoint と candidate source の監査は通過。
- training_loss 推移: region 2.42 → 0.59 → 0.46;legacy 2.68 → 0.54 → 0.43。
  cross_correlation diag は 0.996 に飽和。

現在の選択処方は [noise/rotation search report](local_bt_rot90_asymmetric_noise_recipe.md) を参照。
