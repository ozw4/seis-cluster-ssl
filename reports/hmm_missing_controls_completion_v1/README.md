# HMMあり・なしの性能比較

保存済みHMM評価4689件を掲載した4754行です。v1・v2・v3・legacyを同じCSVに収録し、`hmm_version`・`head_ks`・モデル・評価分割・評価対象を各行に明記しています。

- [全バージョン・全seedの結果と対応比較](factorial_cells.csv)
- [seed別の平均・標準偏差・勝敗数](factorial_comparison.csv)
- [HMM損失設定間の効果・交互作用](factorial_contrasts.csv)
- [Barlow蒸留比較](barlow_comparison.csv)
- [v1蒸留比較](v1_comparison.csv)

`without_hmm_primary`がHMMなし、`with_hmm_primary`がHMMあり、`primary_improvement`が改善量です。正ならHMMで改善、負なら悪化です。F3/Parihakaでは「あり−なし」、Volve MAEでは「なし−あり」としています。指標名・単位は`primary_metric`・`primary_unit`列で確認できます。

`comparison_status=paired`は対照との評価・decoder条件が一致する比較です。`unmatched_baseline`の行もHMMの実測値を残し、HMMなし・改善量は空欄とし、理由を`comparison_note`に記載します。validationとtest、異なる評価対象を対応づけていません。F3の対照は共通test、Parihaka/Volveはtestです。

副指標も同じ列構成です。改善量のSDは対応する5レイアウトの差から求めた標本SD（ddof=1）で、個別SDの差ではありません。集計表にはn・集計状態・HMMの勝・分・敗も示します。5レイアウト未満は明示し、1件のSDは空欄です。同一レイアウトが重複する集計は平均しません。

`with_hmm_decoder_seed`と`without_hmm_decoder_seed`を各行に記載し、`comparison_type`で同一seed／異なるseedを区別します。Parihaka / MAE / largeは42001を`canonical`、42000を`historical`として両方掲載しています。異なるseedの差にはdecoder乱数の影響が含まれます。seedを混ぜて平均していません。

HMMなし評価は条件が一致するHMM評価に再利用しています。MAEは既存の再構成継続モデル、Barlow Twinsは既存の3 epochモデル、Randomは未学習モデルです。学習予算を完全に揃えた因果実験ではなく、既存ベンチマークとの性能比較です。

元のv2比較560行は保持しています。v1蒸留比較の405条件もこの主CSVに統合しています。全実験カタログにはHMM以外の手法も含む全評価・指標を収録します。

5レイアウトはencoder seedの5反復ではなく、ラベル量は包含関係にあります。F3は同一地域内のtransductiveなラベルholdoutです。

[実験記録](../../experiments/hmm_v2/missing_controls_completion_v1/README.md) · [出典hash](provenance.json)。公開ファイルは7点です。
