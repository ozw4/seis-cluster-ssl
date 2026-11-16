# HMMあり・なしの性能比較

540 HMM条件に対して140個のHMMなし評価を対応づけた560行です。F3は共通test、Parihaka/Volveはtestです。

- [全seedの対応比較](factorial_cells.csv)
- [seed別の平均・標準偏差・勝敗数](factorial_comparison.csv)
- [HMM損失設定間の効果・交互作用](factorial_contrasts.csv)
- [Barlow蒸留比較](barlow_comparison.csv)
- [v1蒸留比較](v1_comparison.csv)

`without_hmm_primary`がHMMなし、`with_hmm_primary`がHMMあり、`primary_improvement`が改善量です。正ならHMMで改善、負なら悪化です。F3/Parihakaでは「あり−なし」、Volve MAEでは「なし−あり」としています。指標名・単位は`primary_metric`・`primary_unit`列で確認できます。

副指標も同じ列構成です。改善量のSDは対応する5レイアウトの差から求めた標本SD（ddof=1）で、個別SDの差ではありません。集計表にはHMMの勝・分・敗も示します。

`with_hmm_decoder_seed`と`without_hmm_decoder_seed`を各行に記載し、`comparison_type`で同一seed／異なるseedを区別します。Parihaka / MAE / largeは42001を`canonical`、42000を`historical`として両方掲載しています。異なるseedの差にはdecoder乱数の影響が含まれます。seedを混ぜて平均していません。

HMMなし評価を各4損失設定に再利用しています。MAEは既存の再構成継続モデル、Barlow Twinsは既存の3 epochモデル、Randomは未学習モデルです。学習予算を完全に揃えた因果実験ではなく、既存ベンチマークとの性能比較です。

この表は540 HMM条件とその対照の比較です。v1/v3・別checkpoint・archive等を含む保存済み全実験はartifactsのall_experiment_metricsカタログCSV・HTMLに収録します。

継続学習seedは42です。5レイアウトはencoder seedの5反復ではなく、ラベル量は包含関係にあります。F3は同一地域内のtransductiveなラベルholdoutです。

[実験記録](../../experiments/hmm_v2/missing_controls_completion_v1/README.md) · [出典hash](provenance.json)。公開ファイルは7点です。
