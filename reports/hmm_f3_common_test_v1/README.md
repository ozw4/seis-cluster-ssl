# F3 HMMなし・v1・v2：共通test評価

2026-10-20完了。**23条件・345セル**を、既存のvalidation-selected best checkpointと全体予測を固定して追加評価しました。再学習・embeddings再抽出・追加推論は行っていません。

この文書はレビュー用の転載であり、パイプライン入力には使用しません。全条件の出典は[条件一覧](../../experiments/f3/facies_benchmark_v2/135_lithology_independent_test_audit_v1/condition_inventory.csv)、実行方法は[実験記録](../../experiments/f3/facies_benchmark_v2/135_lithology_independent_test_audit_v1/README.md)にあります。

本ディレクトリの公開レビュー用ファイルは **README.md・comparison.csv・per_layout.csvの3点のみ**です。CSVは完了済みproducerのレビュー出力を転載したもので、数値・行順は同一です（改行コードのみLFへ統一）。実行・再開はartifactsの原本を使用し、このディレクトリを入力にしません。

## 評価条件

- Test：reference-valid領域から、全レイアウトのcanonical train候補断面とvalidationを除外。**113,979,624ボクセル**を全条件共通で評価。
- Train／validationとtestラベルの重複はゼロ。checkpointはvalidation macro-F1、同点時IoU、さらに同点時weighted CEで選択した既存bestを使用。
- 6クラス固定。mean IoU／macro-F1はtest全体の混同行列から算出。主表は5レイアウトの算術平均±**標本標準偏差（ddof=1）**。
- Encoder事前学習は調査地域全体の未ラベル振幅を使用。空間bufferを置かない**同一調査地域内transductive・下流ラベルholdout評価**であり、完全な空間holdoutではありません。
- 図・結果を用いたパラメータ調整はしていないとのユーザー確認を記録済み。testでepoch・head・損失重み・正本を選び直していません。
- HMMなしMAEはMAE100＋MAE25、LocalBTはasymmetric LocalBT3、randomは固定初期値。HMM継続学習は25 epochs。
- v1は単一K6、dist=0.1／0.2。ここでのv2正本は事前指定済みK=[6,8,10]・dist=0.2・consistency=0.1です。
- v3・Parihaka・Volveは再評価対象外です。

## Mean IoU

| 初期モデル | 教師量 | HMMなし | v1 dist=0.1 | v1 dist=0.2 | v2正本 |
|---|---|---:|---:|---:|---:|
| MAE | small | 0.4819 ± 0.0156 | 0.5246 ± 0.0395 | 0.5226 ± 0.0248 | 0.5199 ± 0.0297 |
| MAE | medium | 0.5525 ± 0.0459 | 0.5830 ± 0.0490 | 0.5797 ± 0.0318 | 0.5808 ± 0.0393 |
| MAE | large | 0.6765 ± 0.0105 | 0.6823 ± 0.0052 | 0.6844 ± 0.0154 | 0.6867 ± 0.0133 |
| LocalBT | small | 0.3610 ± 0.0352 | 0.3705 ± 0.0293 | 0.3795 ± 0.0339 | 0.3733 ± 0.0343 |
| LocalBT | medium | 0.4222 ± 0.0345 | 0.4338 ± 0.0366 | 0.4366 ± 0.0249 | 0.4402 ± 0.0385 |
| LocalBT | large | 0.5165 ± 0.0301 | 0.5292 ± 0.0231 | 0.5354 ± 0.0276 | 0.5295 ± 0.0164 |
| random | small | 0.3301 ± 0.0133 | 0.3989 ± 0.0447 | 0.3865 ± 0.0474 | 0.4506 ± 0.0464 |
| random | medium | 0.3829 ± 0.0208 | 0.4552 ± 0.0309 | 0.4602 ± 0.0318 | 0.5097 ± 0.0536 |
| random | large | 0.4555 ± 0.0189 | 0.5522 ± 0.0314 | 0.5531 ± 0.0249 | 0.6146 ± 0.0324 |

## Macro-F1

| 初期モデル | 教師量 | HMMなし | v1 dist=0.1 | v1 dist=0.2 | v2正本 |
|---|---|---:|---:|---:|---:|
| MAE | small | 0.6183 ± 0.0278 | 0.6539 ± 0.0472 | 0.6512 ± 0.0343 | 0.6477 ± 0.0359 |
| MAE | medium | 0.6827 ± 0.0487 | 0.7014 ± 0.0560 | 0.7019 ± 0.0414 | 0.7025 ± 0.0458 |
| MAE | large | 0.7906 ± 0.0085 | 0.7971 ± 0.0045 | 0.7979 ± 0.0128 | 0.7980 ± 0.0103 |
| LocalBT | small | 0.4957 ± 0.0435 | 0.5086 ± 0.0343 | 0.5160 ± 0.0365 | 0.5053 ± 0.0434 |
| LocalBT | medium | 0.5598 ± 0.0436 | 0.5721 ± 0.0415 | 0.5739 ± 0.0312 | 0.5781 ± 0.0431 |
| LocalBT | large | 0.6594 ± 0.0286 | 0.6703 ± 0.0207 | 0.6763 ± 0.0255 | 0.6698 ± 0.0165 |
| random | small | 0.4473 ± 0.0186 | 0.5283 ± 0.0495 | 0.5192 ± 0.0499 | 0.5825 ± 0.0457 |
| random | medium | 0.5136 ± 0.0200 | 0.5896 ± 0.0341 | 0.5909 ± 0.0401 | 0.6396 ± 0.0546 |
| random | large | 0.5928 ± 0.0172 | 0.6847 ± 0.0299 | 0.6862 ± 0.0248 | 0.7399 ± 0.0292 |

## 比較上の注意

- v2正本はHMMなしに対して、3モデル×3教師量の**全9条件で平均IoU・平均F1が改善**しました。
- v1 dist=0.2からv2正本では、MAE small、LocalBT small／largeの平均IoUは低下しています。「全条件で段階的に改善」とは主張できません。
- HMMなし→v1 dist=0.2→v2正本で両指標が段階的に改善するのは**6/9条件**です。以前のvalidation集計による8/9という記述は、このtest結果にそのまま適用しません。
- 5レイアウトは5つのencoder学習seedや5つの独立した空間testではありません。voxel数を独立反復数として有意差を主張しません。

## 全条件・補足データ

この主表はHMMなし・v1両dist・事前指定のv2正本の固定比較です。v2の他の既存head構成／consistency条件を含む**全23条件**は以下に保存しています。MAEのみのK=[4,6]／[6,8]も含み、再利用された同一条件は重複計上していません。

- [全23条件の平均±標準偏差一覧](../../artifacts/seis_ssl_cluster/surveys/f3/facies_benchmark_v2/evaluation/lithology_common_test_v1/summary/summary.md)
- [全69組の集計CSV](comparison.csv)
- [345セルの個別値CSV](per_layout.csv)
- [クラス別指標CSV：2,070行](../../artifacts/seis_ssl_cluster/surveys/f3/facies_benchmark_v2/evaluation/lithology_common_test_v1/summary/class_metrics.csv)
- [混同行列CSV：12,420行](../../artifacts/seis_ssl_cluster/surveys/f3/facies_benchmark_v2/evaluation/lithology_common_test_v1/summary/confusion_matrices.csv)
- [集計JSON](../../artifacts/seis_ssl_cluster/surveys/f3/facies_benchmark_v2/evaluation/lithology_common_test_v1/summary/summary.json)
- [Test定義と入力hash](../../artifacts/seis_ssl_cluster/surveys/f3/facies_benchmark_v2/evaluation/lithology_common_test_v1/inputs/test_definition.json)
- [評価前に固定した345セルのcheckpoint識別情報](../../artifacts/seis_ssl_cluster/surveys/f3/facies_benchmark_v2/evaluation/lithology_common_test_v1/selection_receipt.json)

## 検証・再現性

- 新規・既存の関連回帰テスト54件、Ruff lint／format、パッケージ境界チェックを通過。
- 345個のbest checkpoint、入力予測・valid maskのSHA-256を確認。全セルのtest支持数は一致。
- MAE／layout_000／largeの全testボクセルを既存の汎用混同行列実装でも再集計し、完全一致を確認。
- 69組すべての平均・標本標準偏差を345件の個別値から独立に再計算して一致を確認。
- 既存validation結果・checkpoint・embeddings・予測・凍結レポートは上書きしていません。
- Run identity：`22e45c6629d606c3eb9885ea2be7471ba7dd8dcf333899e7f2621feec0234519`
- Test mask SHA-256：`2bac34f4ebf8612d120673ae3dbda1a8720624e27a1701491059abf1c60cfe1d`
- Summary JSON SHA-256：`a22b2fb563d9c7ac53446b8168ad2c5c50261559554c6ccc61b2ad9de43610f7`
- Comparison CSV SHA-256：`0e0e1037cac1dcada0c65a62ff8e8bd10c46943588894997d09ea382572e4501`
- Per-layout CSV SHA-256：`c351d5c5fcf3f65f1a051eca66ec5736496a51630a3bf366efd9ad48f8d0979b`
