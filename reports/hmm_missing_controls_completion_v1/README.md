# HMMあり・なしの性能比較（540条件）

同じデータセット・初期表現・レイアウト・ラベル量・評価分割・decoder seed 42000で対応づけています。F3は共通test、Parihaka/Volveはtestです。

- [540行の対応比較](factorial_cells.csv)
- [108条件の平均・標準偏差・勝敗数](factorial_comparison.csv)
- [HMM損失設定間の効果・交互作用](factorial_contrasts.csv)
- [Barlow蒸留比較](barlow_comparison.csv)
- [v1蒸留比較](v1_comparison.csv)

`without_hmm_primary`がHMMなし、`with_hmm_primary`がHMMあり、`primary_improvement`が改善量です。正ならHMMで改善、負なら悪化です。F3/Parihakaでは「あり−なし」、Volve MAEでは「なし−あり」としています。指標名・単位は`primary_metric`・`primary_unit`列で確認できます。

副指標も同じ列構成です。改善量のSDは対応する5レイアウトの差から求めた標本SD（ddof=1）で、個別SDの差ではありません。108行表にはHMMの勝・分・敗も示します。

135個のHMMなし評価を各4損失設定に再利用しています。MAEは既存の再構成継続モデル、Barlow Twinsは既存の3 epochモデル、Randomは未学習モデルです。学習予算を完全に揃えた因果実験ではなく、既存ベンチマークとの性能比較です。Parihaka MAEの図用seed 42001は含めていません。

継続学習seedは42です。5レイアウトはencoder seedの5反復ではなく、ラベル量は包含関係にあります。F3は同一地域内のtransductiveなラベルholdoutです。

[実験記録](../../experiments/hmm_v2/missing_controls_completion_v1/README.md) · [出典hash](provenance.json)。公開ファイルは7点です。
