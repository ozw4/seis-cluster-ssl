# F3 Local Barlow Twins anti-collapse search v1

## 目的と範囲

Local Barlow Twins が Random encoder を下回る原因を、loss strength、projector
capacity、view policy の小さな介入で切り分ける実験である。短期学習した候補を
`medium / layout_001` で比較し、representation rank の回復が下流性能の回復に
結び付くかを検証した。

## 系譜

事前学習は `22_local_barlow_twins_v1`、下流評価は canonical five-way v3 を
比較元とする。各 arm の差分、予算、identity は YAML と resolver が定義し、
config test はそれらを検証する。

## 結果

この実験の数値、診断、判断、および解釈は
[canonical result report](../../../../reports/f3/local_bt_anticollapse_search_v1_summary.md)
に集約する。

候補の実行には [shared candidate runbook](../LOCAL_BT_CANDIDATE_RUNBOOK.md) を使う。
