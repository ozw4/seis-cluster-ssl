# F3 Local Barlow Twins nuisance × region ascent v1

## 目的と範囲

実験 121 で見つかった加法雑音と region pooling の方向を、noise strength、region
shape、projector、redundancy、identity view の独立な変更に分解した探索である。
単一セルは粗い選別にだけ使い、候補の判断を複数 layout へ広げた。

## 系譜

実験 121 の有望候補を基点とし、`22_local_barlow_twins_v1` の学習契約と canonical
five-way v3 の下流契約を保つ。arm の定義と比較差分は YAML と resolver が定義し、
config test はそれらを検証する。

## 結果

この実験の数値、診断、判断、および効果の帰属は
[canonical result report](../../../../reports/f3/local_bt_random_parity_breakthrough.md)
に集約する。

config triplet の実行は [shared candidate runbook](../LOCAL_BT_CANDIDATE_RUNBOOK.md) に集約する。
