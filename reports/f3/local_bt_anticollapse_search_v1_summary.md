# F3 Local BT anti-collapse search v1 — Phase A の記録

- 日付: 2026-09-03(GPU 実行同日)
- 実験: [119 Local BT anti-collapse search](../../experiments/f3/facies_benchmark_v2/119_local_bt_anticollapse_search_v1/README.md)
- 範囲: collapse を抑える 5 arms を 3 epochs 学習し、canonical five-way v3
  の `medium / layout_001` で Random encoder と比較した。Phase B への事前登録
  gate は delta > −0.02。
- 符号: delta = arm macro_f1 − Random macro_f1(正が良い)。evaluation は単一セル
  `medium / layout_001` のみ。本表の delta は 3ep raw(continuation なし)の
  単一セル値。

## 結果(Random macro_f1 = 0.496541、effective rank = 24.00 / raw norm = 26.45)

| arm | 介入 | macro_f1 | delta vs Random | eff. rank | raw norm | screen |
|---|---|---:|---:|---:|---:|---|
| `legacy` | epochs: 3 のみ(無介入対照) | 0.460224 | **−0.036318** | 20.43 | 61.2 | FAIL |
| `lambda020` | redundancy_weight 0.005→0.02 | 0.428823 | −0.067718 | 23.25 | 52.1 | FAIL |
| `lambda080` | redundancy_weight 0.005→0.08 | 0.431601 | −0.064940 | 26.07 | 45.8 | FAIL |
| `proj1536_lambda020` | projector 384→1536 + λ0.02 | 0.453970 | −0.042571 | 30.15 | 48.4 | FAIL |
| `d4td02` | view: xy_d4_trace_drop_v1 (refl 0.5 / drop 0.02) | 0.455534 | −0.041007 | 15.15 | 77.2 | FAIL |

- gate 判定: 全 arm `BT_AC_SCREEN_FAIL`。`BT_AC_PHASE_B_ELIGIBLE: []`(delta > −0.02 該当なし)につき、事前登録どおり Phase B(10ep 延長・epoch–delta 勾配測定)は実施しない。
- checkpoint と candidate source の監査は全 arm で通過。`cross_correlation` 系 objective の loss は有限で収束。
- `xy_d4_trace_drop_v1` view は Parihaka でも負の結果だった。
- plain BT(legacy flip view)の 1ep(+25cont) / 5ep(+25cont) 既知値の正典記録:
  [base1ep/summary.md](facies_benchmark_v2/local_barlow_twins_gaussian_view_v1/base1ep/summary.md)、
  [base5ep/summary.md](facies_benchmark_v2/local_barlow_twins_gaussian_view_v1/base5ep/summary.md)
  の `local_barlow_twins_legacy_flip_base1ep` / `local_barlow_twins_legacy_flip_base5ep` 行、
  `Mean delta vs random` 列(medium 5 layout 平均、+25 continuation epochs 後)。
  本表の 3ep raw 単一セル値とはプロトコルが異なる。

現在採用している処方は
[local_bt_rot90_asymmetric_noise_recipe.md](local_bt_rot90_asymmetric_noise_recipe.md)
を参照。
