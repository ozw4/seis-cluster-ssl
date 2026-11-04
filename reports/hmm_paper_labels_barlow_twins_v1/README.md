# v1・v2可視化図：「Barlow Twins」表記版

2026-10-22完了。全160枚の再出力と検証が完了しています。

可視化図のモデル名だけを `LocalBT` から **Barlow Twins** に変更した版です。
対象はv1・v2それぞれF3 30枚、Parihaka 30枚、Volve 20枚、計160枚です。

- HMMなし：`Barlow Twins`
- v1：`Barlow Twins + HMM`
- v2：`Barlow Twins + HMM v2`

予測原本・checkpoint・seed・断面位置・表示範囲・配色・図の大きさは変更しません。
学習・追加推論・スコア再計算は行いません。数値の結果表・実験IDは変更しません。
各PNGについて、変更対象の2つのタイトル以外が元図とピクセル単位で一致することを検証済みです。
ユーザー指定により、元の保存先のPNGを上書きしました。旧図は削除済みでバックアップはありません。
別保存先の重複出力も整理し、図への従来のパスはそのまま利用できます。

## 修正版の保存先

| タスク | v1 | v2正本 |
|---|---|---|
| F3（各30枚） | [図・manifest・断面CSV](../../artifacts/seis_ssl_cluster/studies/hmm_v1/paper_figures/f3/layout_000/large/) | [図・manifest・断面CSV](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/) |
| Parihaka（各30枚） | [図・manifest・断面CSV](../../artifacts/seis_ssl_cluster/studies/hmm_v1/paper_figures/parihaka/layout_000/large/) | [図・manifest・断面CSV](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/) |
| Volve（各20枚） | [図・manifest・断面CSV](../../artifacts/seis_ssl_cluster/studies/hmm_v1/paper_figures/volve/layout_000/large/) | [図・manifest・断面CSV](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/) |

各保存先には元図と同名のPNGと `manifest.json`・`selected_slices.csv` のみを公開します。
正確なファイル数はv1・v2それぞれ32／32／22、合計172です。CSVは元のバイト列を保持します。
manifestには元manifestのSHA-256と表記変更の履歴を記録します。

このレビュー用ディレクトリの公開ファイルはREADME.mdの1点のみです。
パイプライン入力はartifactsから読み、reportsを入力にしません。

関連回帰テスト76件、Ruff lint／format、パッケージ境界チェックを通過。
