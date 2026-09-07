# F3 Local BT region-positive PoC v1 — FAIL の記録

- 日付: 2026-09-03(GPU 実行同日)
- 実験: [120 Local BT region-positive PoC](../../experiments/f3/facies_benchmark_v2/120_local_bt_region_positive_poc_v1/README.md)
- 仮説(ユーザー提案): positive pair を対応 1 token(8×8×8 voxel)から 2×2×1 token
  block の平均(16×16×8 voxel の領域表現)へ粗視化すれば、invariance 圧が token
  スケールの細部から低周波側へ逃げ、Random init の入力忠実構造への損傷が緩和される。
- 範囲: region-positive の 1 arm を 3 epochs 学習し、canonical five-way v3 の
  `medium / layout_001` で Random と 119 `legacy` に比較した。

## 結果(Random macro_f1 = 0.496541 / eff. rank 24.00)

| source | macro_f1 | delta vs Random | eff. rank | raw norm | screen |
|---|---:|---:|---:|---:|---|
| `local_bt_region_w221_3ep` | 0.450633 | **−0.045909** | **16.75** | 61.4 | FAIL |
| (参照)119 `legacy` 3ep | 0.460224 | −0.036318 | 20.43 | 61.2 | FAIL |

- **window 効果 = −0.009591**(region 化は無介入 legacy より悪化)。
- gate 判定: `BT_RP_SCREEN_FAIL`、`BT_RP_PHASE_B_ELIGIBLE: False`
  (delta ≤ −0.02)につき、事前登録どおり Phase B(10ep 延長)は実施しない。
- checkpoint と candidate source の監査は通過した。
- 学習は正常収束(training_loss 2.42 → 0.59 → 0.46;legacy は 2.68 → 0.54 →
  0.43。cross_correlation diag は 0.996 に飽和)。

## 主要な発見

1. **positive pair の空間サポート粗視化は損傷を緩和せず、悪化させる。** 予測
   レンジ(legacy〜parity)の下限をさらに下回った(−0.046 < −0.036)。
2. **機構: region-mean invariance は特徴の空間平滑化で自明に充足できる。**
   effective rank が legacy 20.43 → 16.75 へ大きく低下(d4td02 の 15.15 に次ぐ
   深さ)したことと整合的。block 平均を一致させる最も安価な解は隣接 token が
   表現を共有することであり、これは voxel 単位 decoder が必要とする token 間
   弁別性を直接損なう。「勾配 1/W 希釈による緩和」よりも「平滑化圧の追加」が
   支配した。
3. 当時は既存の tested axes と合わせて invariance 側を閉じたと解釈したが、
   view/noise の組み合わせは未探索だった。後続の 121–123 がこの全称的な解釈を
   覆したため、本レポートの結論は region-positive 単独の arm に限定する。

## 結論

この実験では、flip view に region mean を加えた単一 arm が Random と legacy の
いずれにも届かず、token 間弁別性をさらに損なう結果だった。これは positive pair
の定義一般、または Local Barlow Twins 一般の不可能性を示さない。後続探索と現在の
選択処方は
[noise/rotation search report](local_bt_rot90_asymmetric_noise_recipe.md)を参照する。
