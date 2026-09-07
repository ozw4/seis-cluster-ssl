# F3 Local Barlow Twins noise × rotation search v1

## 目的と範囲

実験 122 の Random-parity 候補から、noise distribution、片側だけを劣化させる
asymmetric noise、純回転と鏡映の違いを分離した探索である。粗い単一セル評価の後、
候補を canonical five-way と同じ layout / size 条件で比較し、学習期間による変化も
確認した。

## 系譜

比較元は実験 122 と `22_local_barlow_twins_v1`、下流条件は canonical five-way
v3 である。augmentation、region、epoch continuation は YAML と resolver が定義し、
config/implementation test はそれらを検証する。

## 結果

採択した処方と全条件の統計は
[`local_bt_rot90_asymmetric_noise_recipe.md`](../../../../reports/f3/local_bt_rot90_asymmetric_noise_recipe.md)、
学習期間の比較は
[`local_bt_epoch_scaling_v1.md`](../../../../reports/f3/local_bt_epoch_scaling_v1.md)
に分けて記録する。

短期 run、continuation、比較 cell は
[shared candidate runbook](../LOCAL_BT_CANDIDATE_RUNBOOK.md) の順序で実行する。
