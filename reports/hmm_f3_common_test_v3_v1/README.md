# F3 v3：共通test追加評価

2026-10-21完了。既存のvalidation-selected decoderと全体予測を固定し、**9条件・135セル**を追加test評価しました。再学習・embedding再抽出・追加推論は行っていません。停止済みの未完了実験は再開していません。

このディレクトリはレビュー専用です。公開ファイルは **README.md・comparison.csv・per_layout.csvの3点のみ**。CSVは完了済みartifactsの数値・行順を保った転載です（改行コードのみLFへ統一）。パイプライン入力には使いません。

[条件一覧](../../experiments/f3/facies_benchmark_v2/136_lithology_common_test_v3_v1/condition_inventory.csv) · [実行記録](../../experiments/f3/facies_benchmark_v2/136_lithology_common_test_v3_v1/README.md) · [既存HMMなし・v1・v2のtest結果](../hmm_f3_common_test_v1/README.md)

## 現在の最終結果：LocalBT・3e-5・50:50

2026-10-21、両seedの結果を報告した後のユーザー指定により、この条件の
**small／medium／large・全5レイアウトはdecoder seed＝42001を最終結果**とします。
[最終結果の指定・hash](../../experiments/f3/facies_benchmark_v2/137_local_bt_v3_masklr3em5_decoder_seed42001_v1/selection.yaml)
に記録しました。encoderは同じで、他モデル・他設定の結果は変更しません。

| 教師量 | mean IoU（％） | macro-F1（％） |
|---|---:|---:|
| small | 38.55±4.81 | 52.17±5.52 |
| medium | 44.69±3.59 | 58.29±4.41 |
| large | 52.73±1.45 | 66.80±1.38 |

5レイアウトの平均±標本標準偏差です。
[最終結果の集計CSV](../../artifacts/seis_ssl_cluster/surveys/f3/facies_benchmark_v2/evaluation/local_bt_v3_masklr3em5_decoder_seed42001_v1/summary/comparison.csv) ·
[最終結果の15個別値](../../artifacts/seis_ssl_cluster/surveys/f3/facies_benchmark_v2/evaluation/local_bt_v3_masklr3em5_decoder_seed42001_v1/summary/per_layout.csv)

以下の9条件表と本ディレクトリのCSVは、**元のseed＝42000の135セルの履歴**として
変更せず残しています。今後この条件を比較するときは、上記42001の15セルを参照します。
これは結果報告後の明示的な指定です。seed＝42000のv2等との比較ではdecoder seedが
異なることを明記し、同一seedの手法差や事前規定されたseed選択とは扱いません。

## 評価条件

- 前回と完全に同じ **113,979,624ボクセル**。全canonical train候補とvalidationを除いたreference-valid領域です。test maskのSHA-256完全一致を確認しました。
- 135セルの教師ラベル分割、decoder構成・学習・タイル設定は前回と一致。decoder seed=42000、50 epochs、22,000 stepsの既存validation-selected best.ptを使用し、testでcheckpointや設定を選んでいません。
- 全条件K=[6,8,10]・dist=0.2・consistency=0.1・trace-column mask=10%。encoder/head LR=1e-5、継続学習25 epochs。表のLRはmask token専用です。
- 全6クラスのtest全体の混同行列からmean IoU／macro-F1を算出。5レイアウトの算術平均±標本標準偏差（ddof=1）を報告します。レイアウト間SDでありencoder学習seed間SDではありません。
- 未ラベル振幅の事前学習は同一調査地域全体を含みます。空間bufferのないtransductive・下流ラベルholdout評価であり、完全な空間holdoutではありません。
- LocalBT/randomのLR=1e-4、25:75、75:25は完了済みdecoderがなく対象外です。Parihaka・Volveも対象外です。
- HMMなし・v1・v2の既存test結果、および全ての既存validation結果は変更していません。

## 元のseed＝42000による全9条件（履歴）

各セルは **mean IoU ± SD / macro-F1 ± SD**、単位は％です。重みはmasked:visible。全条件で5レイアウトが揃っています。

| モデル | Mask-token LR | 重み | Small：IoU / F1 | Medium：IoU / F1 | Large：IoU / F1 |
|---|---|---|---|---|---|
| MAE | 1e-5 | 50:50 | 52.13±2.25 / 64.55±3.22 | 58.20±3.74 / 70.02±4.37 | 69.21±1.71 / 80.24±1.47 |
| MAE | 3e-5 | 50:50 | 53.57±2.64 / 66.01±3.65 | 58.74±4.56 / 70.60±5.27 | 69.15±1.60 / 80.26±1.39 |
| MAE | 1e-4 | 50:50 | 53.10±3.30 / 65.52±4.10 | 57.76±3.81 / 69.63±4.88 | 69.67±1.37 / 80.68±1.20 |
| MAE | 3e-5 | 25:75 | 52.35±3.06 / 64.93±3.78 | 60.04±5.10 / 71.63±5.84 | 67.99±2.58 / 79.08±2.55 |
| MAE | 3e-5 | 75:25 | 51.91±2.16 / 64.47±2.83 | 58.98±4.14 / 70.70±4.71 | 68.44±1.49 / 79.55±1.31 |
| LocalBT | 1e-5 | 50:50 | 38.13±3.55 / 51.67±4.24 | 44.03±3.34 / 57.66±4.03 | 52.41±1.99 / 66.52±1.96 |
| LocalBT | 3e-5 | 50:50 | 39.56±2.69 / 53.35±3.20 | 43.32±2.72 / 57.02±3.45 | 52.66±1.92 / 66.66±1.69 |
| random | 1e-5 | 50:50 | 43.93±5.00 / 57.04±4.96 | 52.01±4.69 / 65.02±5.21 | 61.79±3.51 / 74.11±3.31 |
| random | 3e-5 | 50:50 | 44.55±5.01 / 57.42±5.42 | 51.86±4.46 / 64.82±4.85 | 62.57±2.73 / 75.00±2.37 |

## 元のseed＝42000：3e-5・50:50とv2正本の固定比較（履歴）

この設定は今回のtestから選び直したものではありません。v2正本は事前指定のK=[6,8,10]・dist=0.2・consistency=0.1です。差は **v3 − v2の平均値差（percentage points）** で、有意差を意味しません。

| モデル | 教師量 | Δ mean IoU | Δ macro-F1 |
|---|---|---:|---:|
| MAE | small | +1.58 | +1.24 |
| MAE | medium | +0.67 | +0.34 |
| MAE | large | +0.48 | +0.45 |
| LocalBT | small | +2.24 | +2.83 |
| LocalBT | medium | -0.70 | -0.78 |
| LocalBT | large | -0.29 | -0.32 |
| random | small | -0.51 | -0.82 |
| random | medium | +0.90 | +0.86 |
| random | large | +1.11 | +1.01 |

この元のseed＝42000集計では、MAEは全3サイズ、LocalBTはsmall、randomはmedium／largeで両指標の平均が改善しました。一方、LocalBT medium／largeとrandom smallは低下しました。したがって「全条件でv2→v3が改善」とは主張できません。この135セル集計の作成時点では正本を選び直していません。後日のLocalBT・3e-5・50:50の最終結果指定は冒頭に別記しています。

## 補足と再現性

- [27組の集計CSV](comparison.csv)
- [135セルの個別値CSV](per_layout.csv)
- [原本の全条件表](../../artifacts/seis_ssl_cluster/surveys/f3/facies_benchmark_v2/evaluation/lithology_common_test_v3_v1/summary/summary.md)
- [クラス別指標：810行](../../artifacts/seis_ssl_cluster/surveys/f3/facies_benchmark_v2/evaluation/lithology_common_test_v3_v1/summary/class_metrics.csv)
- [混同行列：4,860行](../../artifacts/seis_ssl_cluster/surveys/f3/facies_benchmark_v2/evaluation/lithology_common_test_v3_v1/summary/confusion_matrices.csv)
- [Test定義](../../artifacts/seis_ssl_cluster/surveys/f3/facies_benchmark_v2/evaluation/lithology_common_test_v3_v1/inputs/test_definition.json)
- [評価前に固定した135セルの入力・checkpoint](../../artifacts/seis_ssl_cluster/surveys/f3/facies_benchmark_v2/evaluation/lithology_common_test_v3_v1/selection_receipt.json)

関連回帰テスト58件、Ruff lint／format、パッケージ境界チェックを通過。全27組の平均・標本SDを135個別値から独立に再計算し一致を確認しました。MAE・LR=1e-5・50:50／layout_000／largeでは、別の既存汎用実装で全113,979,624ボクセルを再集計し、混同行列の完全一致を確認しました。セル公開ファイル2点、原本summary公開ファイル6点、本レビュー公開ファイル3点を確認しています。

- Run identity：`c273a0077fd9d56c9991ba8247d3b038292a29fdb1af15865e4d7ff7c1eb1dee`
- Test-mask SHA-256：`2bac34f4ebf8612d120673ae3dbda1a8720624e27a1701491059abf1c60cfe1d`
- Summary JSON SHA-256：`a331533437544353fd6ff8ecb26116462d197438e418319156894d9cdabcef1e`
- 原本comparison.csv SHA-256：`6b602c230a4a53f45162fea1d7970f8da8dfe8ed2f2b89e9e75bd43fdef2acf3`
- 原本per_layout.csv SHA-256：`e2d386f56601876d00f4fb45e679585d31f7e9c72d2db3962d49ba9e451ab2a8`
- 実行時evaluator SHA-256：`79564130e43cd314d1ec718596511425b850c63b1f2892aeaa69c75a1a68629e`
