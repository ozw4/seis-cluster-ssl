# F3 Local Barlow Twins view × region search v1

## 目的と範囲

実験 119–120 を受け、view augmentation と region-positive pooling の相互作用を
比較する探索である。構造を変える view、遮蔽、取得由来の nuisance を分け、どの
不変性が F3 の固定 voxel decoder に有用かを調べた。

## 系譜

事前学習の比較元は `22_local_barlow_twins_v1`、下流条件は canonical five-way v3
である。augmentation と region の全組合せ、学習期間、identity は YAML と
resolver が定義し、config/implementation test はそれらを検証する。

## 結果

この実験の数値、診断、判断、および解釈は
[canonical result report](../../../../reports/f3/local_bt_view_region_search_v1_summary.md)
に集約する。

短期 run と同一 arm の continuation は
[shared candidate runbook](../LOCAL_BT_CANDIDATE_RUNBOOK.md) に従う。
