# Volve horizon checkpoint-selection comparison (45-cell subset)

Generated: 2026-09-03

Validation checkpoint selectionをMAE最小、within-2最大、within-4最大とした
3実験を、同じMAE / MAE-HMM / Random × 5 layouts × 3 supervision sizesで
比較する。各selectionは45セル、合計135 completed runsである。

## Scope and conventions

- 数値はtest primary common support上の5 layoutsの平均 ± 標本SD。
- Within-1/2/4とadjacent-order violationは百分率（%）、MAEはsamples。
- 各runのprimary supportは5 horizons × 156,583 = 782,915 observations、
  coverage 1.0、missing 0。
- `best.pt`はvalidationだけで選択され、testはこの比較・集約にのみ使用する。
- Local BT系30セル/selectionは未実行のため、本資料は正式75セルfive-way
  summaryではない。

## Artifact audit

| Selection | Checkpoint selection ID | Benchmark ID | Runs | Best epoch range | Tied optimum cells |
|---|---|---|---:|---:|---:|
| MAE | `strict_lower_validation_macro_mae_v1` | `mae_local_bt_hmm_five_way_v1` | 45/45 | 7–48 | 0 |
| within-2 | `strict_higher_validation_macro_within_2_v1` | `mae_local_bt_hmm_five_way_within2_v1` | 45/45 | 16–49 | 0 |
| within-4 | `strict_higher_validation_macro_within_4_v1` | `mae_local_bt_hmm_five_way_within4_v1` | 45/45 | 11–49 | 0 |

全135セルでbenchmark/objective identity、epoch 0–49のhistory、first optimal
epoch、metrics/best checkpointのvalidation score、および共通supportを確認した。
MAE版はlegacy artifact schemaのため、selection固有top-level fieldではなく
objective identityと旧MAE checkpoint fieldのread-time fallbackで監査した。

## MAE selection

| Model | Size | Within-1 % | Within-2 % | Within-4 % | MAE samples | Order violation % | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| MAE | small | 16.385 ± 9.045 | 30.366 ± 8.013 | 47.198 ± 2.418 | 11.811 ± 0.717 | 13.502 ± 1.660 | 16.4 ± 6.66 |
| MAE-HMM | small | 15.745 ± 5.495 | 30.955 ± 7.849 | 47.051 ± 4.125 | 11.353 ± 1.576 | 12.689 ± 5.523 | 13.6 ± 2.61 |
| Random | small | 16.058 ± 3.412 | 23.852 ± 3.916 | 32.543 ± 3.931 | 19.572 ± 1.260 | 24.786 ± 2.427 | 28.0 ± 2.55 |
| MAE | medium | 30.851 ± 8.930 | 44.425 ± 6.034 | 56.191 ± 3.360 | 9.156 ± 0.687 | 10.830 ± 2.970 | 11.2 ± 3.96 |
| MAE-HMM | medium | 30.751 ± 6.651 | 45.314 ± 4.769 | 56.711 ± 3.831 | 8.819 ± 0.897 | 9.167 ± 2.842 | 9.2 ± 1.48 |
| Random | medium | 25.935 ± 3.421 | 33.536 ± 3.055 | 41.753 ± 2.603 | 15.865 ± 0.957 | 19.201 ± 2.056 | 23.8 ± 4.15 |
| MAE | large | 49.945 ± 10.203 | 60.845 ± 7.080 | 69.736 ± 5.091 | 5.859 ± 0.614 | 6.836 ± 1.551 | 19.8 ± 9.88 |
| MAE-HMM | large | 53.115 ± 3.546 | 63.058 ± 2.956 | 71.872 ± 2.201 | 5.335 ± 0.535 | 6.813 ± 1.270 | 18.8 ± 6.61 |
| Random | large | 41.612 ± 3.695 | 49.293 ± 3.722 | 56.645 ± 3.286 | 11.473 ± 0.887 | 14.668 ± 1.162 | 40.8 ± 8.23 |

## within-2 selection

| Model | Size | Within-1 % | Within-2 % | Within-4 % | MAE samples | Order violation % | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| MAE | small | 37.927 ± 2.220 | 45.382 ± 1.715 | 51.629 ± 1.542 | 13.582 ± 1.539 | 21.184 ± 3.149 | 43.4 ± 4.45 |
| MAE-HMM | small | 36.526 ± 3.261 | 44.847 ± 2.533 | 51.574 ± 2.058 | 13.200 ± 1.269 | 20.836 ± 2.754 | 34.2 ± 1.92 |
| Random | small | 23.986 ± 2.974 | 30.767 ± 3.313 | 37.738 ± 3.242 | 21.170 ± 1.826 | 26.266 ± 3.035 | 47.8 ± 1.79 |
| MAE | medium | 46.872 ± 1.306 | 54.770 ± 1.655 | 61.361 ± 2.016 | 10.024 ± 1.258 | 13.973 ± 2.009 | 34.6 ± 6.88 |
| MAE-HMM | medium | 46.259 ± 1.424 | 54.552 ± 0.811 | 61.468 ± 0.755 | 9.721 ± 0.672 | 15.488 ± 1.871 | 29.2 ± 4.15 |
| Random | medium | 34.940 ± 1.691 | 41.960 ± 1.519 | 48.621 ± 1.333 | 16.505 ± 0.628 | 20.106 ± 1.124 | 46.6 ± 3.05 |
| MAE | large | 55.561 ± 1.999 | 64.535 ± 1.614 | 72.196 ± 1.355 | 5.768 ± 0.268 | 8.065 ± 1.496 | 26.0 ± 7.18 |
| MAE-HMM | large | 55.584 ± 1.683 | 64.934 ± 1.201 | 72.809 ± 1.084 | 5.444 ± 0.577 | 7.075 ± 1.819 | 25.2 ± 12.26 |
| Random | large | 44.078 ± 0.906 | 51.745 ± 0.775 | 58.786 ± 0.837 | 11.131 ± 0.651 | 14.350 ± 1.215 | 46.4 ± 2.51 |

## within-4 selection

| Model | Size | Within-1 % | Within-2 % | Within-4 % | MAE samples | Order violation % | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| MAE | small | 36.497 ± 3.794 | 44.823 ± 2.623 | 51.851 ± 1.727 | 12.890 ± 2.119 | 19.268 ± 4.795 | 36.2 ± 9.09 |
| MAE-HMM | small | 28.946 ± 10.035 | 41.260 ± 4.745 | 51.067 ± 1.818 | 12.180 ± 2.097 | 17.688 ± 5.991 | 26.6 ± 11.61 |
| Random | small | 24.119 ± 2.982 | 30.997 ± 3.298 | 37.970 ± 3.185 | 20.851 ± 1.362 | 26.139 ± 2.823 | 47.2 ± 1.79 |
| MAE | medium | 45.860 ± 1.265 | 54.213 ± 1.147 | 61.067 ± 1.016 | 9.913 ± 0.525 | 14.359 ± 1.281 | 31.8 ± 9.91 |
| MAE-HMM | medium | 43.563 ± 3.505 | 53.356 ± 2.000 | 61.307 ± 1.241 | 9.173 ± 1.243 | 12.538 ± 3.072 | 23.6 ± 10.04 |
| Random | medium | 34.412 ± 2.046 | 41.455 ± 1.577 | 48.226 ± 1.008 | 16.712 ± 0.299 | 20.449 ± 0.273 | 44.2 ± 3.11 |
| MAE | large | 53.488 ± 3.763 | 62.917 ± 2.756 | 71.209 ± 1.644 | 5.819 ± 0.289 | 7.413 ± 1.892 | 23.0 ± 8.15 |
| MAE-HMM | large | 55.257 ± 1.199 | 64.824 ± 1.082 | 72.923 ± 1.122 | 5.321 ± 0.531 | 6.542 ± 1.287 | 21.2 ± 3.49 |
| Random | large | 44.036 ± 0.877 | 51.666 ± 0.790 | 58.698 ± 0.938 | 11.252 ± 0.809 | 14.569 ± 1.428 | 46.6 ± 2.70 |

## Best selection by model and size

以下は5-layout平均での最良selection。Within系は最大、MAE/orderは最小を選ぶ。

| Model | Size | Within-1 best | Within-2 best | Within-4 best | MAE best | Order best |
|---|---|---|---|---|---|---|
| MAE | small | within-2 (37.927%) | within-2 (45.382%) | within-4 (51.851%) | MAE (11.811 samples) | MAE (13.502%) |
| MAE-HMM | small | within-2 (36.526%) | within-2 (44.847%) | within-2 (51.574%) | MAE (11.353 samples) | MAE (12.689%) |
| Random | small | within-4 (24.119%) | within-4 (30.997%) | within-4 (37.970%) | MAE (19.572 samples) | MAE (24.786%) |
| MAE | medium | within-2 (46.872%) | within-2 (54.770%) | within-2 (61.361%) | MAE (9.156 samples) | MAE (10.830%) |
| MAE-HMM | medium | within-2 (46.259%) | within-2 (54.552%) | within-2 (61.468%) | MAE (8.819 samples) | MAE (9.167%) |
| Random | medium | within-2 (34.940%) | within-2 (41.960%) | within-2 (48.621%) | MAE (15.865 samples) | MAE (19.201%) |
| MAE | large | within-2 (55.561%) | within-2 (64.535%) | within-2 (72.196%) | within-2 (5.768 samples) | MAE (6.836%) |
| MAE-HMM | large | within-2 (55.584%) | within-2 (64.934%) | within-4 (72.923%) | within-4 (5.321 samples) | within-4 (6.542%) |
| Random | large | within-2 (44.078%) | within-2 (51.745%) | within-2 (58.786%) | within-2 (11.131 samples) | within-2 (14.350%) |

## Pairwise selection effects across all 45 cells

Improvementは左側selectionが良いと正。Within系は`left - right`、MAE/orderは
`right - left`。W/T/Lは左側selectionの勝ち/完全同値/負けで、45セルを等重みで
数える。

| Left vs right | Metric | Mean improvement | W/T/L |
|---|---|---:|---:|
| within-2 vs MAE | Within-1 | +11.259 pp | 41/3/1 |
| within-2 vs MAE | Within-2 | +7.983 pp | 41/3/1 |
| within-2 vs MAE | Within-4 | +4.054 pp | 38/3/4 |
| within-2 vs MAE | MAE | -0.811 samples | 9/3/33 |
| within-2 vs MAE | Order violation | -3.206 pp | 9/3/33 |
| within-4 vs MAE | Within-1 | +9.531 pp | 35/8/2 |
| within-4 vs MAE | Within-2 | +7.097 pp | 34/8/3 |
| within-4 vs MAE | Within-4 | +3.847 pp | 32/8/5 |
| within-4 vs MAE | MAE | -0.541 samples | 7/8/30 |
| within-4 vs MAE | Order violation | -2.275 pp | 8/8/29 |
| within-4 vs within-2 | Within-1 | -1.728 pp | 3/24/18 |
| within-4 vs within-2 | Within-2 | -0.887 pp | 4/24/17 |
| within-4 vs within-2 | Within-4 | -0.207 pp | 9/24/12 |
| within-4 vs within-2 | MAE | +0.271 samples | 13/24/8 |
| within-4 vs within-2 | Order violation | +0.931 pp | 16/24/5 |

## Best-epoch changes

Epoch deltaは`left - right`。Earlier / same / laterも左側selectionを基準にする。

| Left vs right | Earlier / same / later | Mean epoch delta | Range |
|---|---:|---:|---:|
| within-2 vs MAE | 3/3/39 | +16.867 | -8…+36 |
| within-4 vs MAE | 3/8/34 | +13.200 | -3…+36 |
| within-4 vs within-2 | 16/24/5 | -3.667 | -21…+10 |

## Main observations

- 45セル等重み平均のWithin-1最良はwithin-2 selection（42.415%）。
- 45セル等重み平均のWithin-2最良はwithin-2 selection（50.388%）。
- 45セル等重み平均のWithin-4最良はwithin-2 selection（57.354%）。
- 45セル等重み平均のMAE最良はMAE selection（11.027 samples）。
- 45セル等重み平均のOrder violation最良はMAE selection（13.166%）。
- 9 model×size groups × 5 metricsの平均最良判定は、within-2が25件、MAEが13件、
  within-4が7件。
- validation objectiveを変えてもtest上で常に同じ方向へ改善するわけではない。
- selection間の科学的比較では、上記model×size別結果とpaired W/T/Lを優先する。
