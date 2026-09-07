# F3 Local Barlow Twins region-positive PoC v1

## 目的と範囲

実験 119 の anti-collapse 介入とは別に、対応 token ではなく局所領域を positive
pair とすることで、invariance の圧力を細部から逃がせるかを一候補で検証した
PoC である。encoder、optimizer、下流 decoder は比較元から変えない。

## 系譜

事前学習は `22_local_barlow_twins_v1`、比較解釈は実験 119、下流評価は canonical
five-way v3 に接続する。領域形状や学習条件は YAML と resolver が定義し、
dataset/config test はそれらを検証する。

## 結果

この実験の数値、診断、判断、および解釈は
[canonical result report](../../../../reports/f3/local_bt_region_positive_poc_v1_summary.md)
に集約する。

この PoC の実行順は [shared candidate runbook](../LOCAL_BT_CANDIDATE_RUNBOOK.md) に従う。
