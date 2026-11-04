# HMM v2正本：論文用比較図

2026-10-22：以下の図リンクは可視化タイトルを **Barlow Twins** に修正した版です。
[v1・v2修正版の一覧](../hmm_paper_labels_barlow_twins_v1/README.md)。
予測と数値は不変です。元の保存先に上書きし、旧図はバックアップなしで削除しました。

2026-10-21完了。**F3 30枚・Parihaka 30枚・Volve 20枚、計80枚**を作成しました。

v1と同じlayout_000／large、断面位置・表示範囲・振幅スケール・配色・2×4配置・180×105 mm／300 dpiです。振幅、正解、MAE／Barlow Twins／randomのHMMなし・v2ありを並べています。v2は**K=[6,8,10]、dist=0.2、consistency=0.1**の事前指定正本です。

このディレクトリの公開レビュー用ファイルは **README.mdの1点のみ**。図・予測原本はartifactsに保存し、レビュー文書をパイプライン入力には使いません。

[実行定義・検証記録](../../experiments/hmm_v2/canonical_paper_sections_v1/README.md) · [v2正本の指定](../../experiments/hmm_v2/canonical_multi_source_v1/selection.yaml)

## 注意点

- encoder／decoderの再学習、設定変更、checkpoint再選択、追加スコア計算は行っていません。不足した7モデル分のみ固定decoderで追加推論しています。予測を平滑化・アンサンブル・後処理していません。
- ParihakaのHMMなしMAEは、指定済みseed=42001の正本を使用。他のdecoderは42000です。v1の旧MAE図はseed=42000なので、この基準パネルは異なります。
- 全断面の定性的な図で、F3の追加test領域だけを描いたものではありません。図の断面を新しい独立testと呼び替えていません。
- 2026-10-22にv1・v2のタイトルを修正し、元の保存先へ上書きしました。全80断面の識別情報・表示範囲・振幅スケールはv1と一致。変更しない370パネルのプロット内部はv1とピクセル単位で一致しました（Parihakaの更新MAE基準パネルを除く）。
- 各図は2125×1240 pixels。F3／Parihaka／Volveごとに、PNGにmanifest.json・selected_slices.csvを加えた正確な32／32／22ファイルを検証済みです。関連回帰テスト76件、Ruff lint／format、パッケージ境界チェックを通過。

## 全図リンク

### F3

[保存先](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large) · [出典・生成manifest](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/manifest.json) · [断面一覧CSV](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/selected_slices.csv)

#### Inline

- [130](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_layout_000_large_inline_130.png)
- [190](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_layout_000_large_inline_190.png)
- [249](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_layout_000_large_inline_249.png)
- [310](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_layout_000_large_inline_310.png)
- [370](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_layout_000_large_inline_370.png)
- [430](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_layout_000_large_inline_430.png)
- [490](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_layout_000_large_inline_490.png)
- [549](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_layout_000_large_inline_549.png)
- [610](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_layout_000_large_inline_610.png)
- [670](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_layout_000_large_inline_670.png)

#### Crossline

- [345](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_layout_000_large_crossline_345.png)
- [435](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_layout_000_large_crossline_435.png)
- [525](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_layout_000_large_crossline_525.png)
- [615](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_layout_000_large_crossline_615.png)
- [705](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_layout_000_large_crossline_705.png)
- [795](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_layout_000_large_crossline_795.png)
- [885](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_layout_000_large_crossline_885.png)
- [975](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_layout_000_large_crossline_975.png)
- [1065](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_layout_000_large_crossline_1065.png)
- [1155](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_layout_000_large_crossline_1155.png)

#### Time／水平断面

- [sample 16 / 0.08 s](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_layout_000_large_time_0p080s_sample_0016.png)
- [sample 38 / 0.19 s](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_layout_000_large_time_0p190s_sample_0038.png)
- [sample 63 / 0.315 s](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_layout_000_large_time_0p315s_sample_0063.png)
- [sample 89 / 0.445 s](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_layout_000_large_time_0p445s_sample_0089.png)
- [sample 114 / 0.57 s](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_layout_000_large_time_0p570s_sample_0114.png)
- [sample 140 / 0.7 s](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_layout_000_large_time_0p700s_sample_0140.png)
- [sample 165 / 0.825 s](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_layout_000_large_time_0p825s_sample_0165.png)
- [sample 190 / 0.95 s](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_layout_000_large_time_0p950s_sample_0190.png)
- [sample 216 / 1.08 s](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_layout_000_large_time_1p080s_sample_0216.png)
- [sample 239 / 1.195 s](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_layout_000_large_time_1p195s_sample_0239.png)

### Parihaka

[保存先](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large) · [出典・生成manifest](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/manifest.json) · [断面一覧CSV](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/selected_slices.csv)

#### Inline

- [41](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_layout_000_large_X_0041.png)
- [118](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_layout_000_large_X_0118.png)
- [196](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_layout_000_large_X_0196.png)
- [274](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_layout_000_large_X_0274.png)
- [352](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_layout_000_large_X_0352.png)
- [431](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_layout_000_large_X_0431.png)
- [509](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_layout_000_large_X_0509.png)
- [587](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_layout_000_large_X_0587.png)
- [665](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_layout_000_large_X_0665.png)
- [742](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_layout_000_large_X_0742.png)

#### Crossline

- [31](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_layout_000_large_Y_0031.png)
- [89](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_layout_000_large_Y_0089.png)
- [148](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_layout_000_large_Y_0148.png)
- [207](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_layout_000_large_Y_0207.png)
- [266](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_layout_000_large_Y_0266.png)
- [325](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_layout_000_large_Y_0325.png)
- [384](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_layout_000_large_Y_0384.png)
- [413](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_layout_000_large_Y_0413.png)
- [412](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_layout_000_large_Y_0412.png)
- [411](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_layout_000_large_Y_0411.png)

#### Time／水平断面

- [sample 334](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_layout_000_large_Z_0335.png)
- [sample 335](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_layout_000_large_Z_0336.png)
- [sample 336](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_layout_000_large_Z_0337.png)
- [sample 352](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_layout_000_large_Z_0353.png)
- [sample 452](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_layout_000_large_Z_0453.png)
- [sample 553](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_layout_000_large_Z_0554.png)
- [sample 581](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_layout_000_large_Z_0582.png)
- [sample 580](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_layout_000_large_Z_0581.png)
- [sample 579](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_layout_000_large_Z_0580.png)
- [sample 578](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_layout_000_large_Z_0579.png)

### Volve

[保存先](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large) · [出典・生成manifest](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/manifest.json) · [断面一覧CSV](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/selected_slices.csv)

#### Inline

- [9981](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_layout_000_large_inline_9981.png)
- [10021](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_layout_000_large_inline_10021.png)
- [10061](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_layout_000_large_inline_10061.png)
- [10101](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_layout_000_large_inline_10101.png)
- [10141](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_layout_000_large_inline_10141.png)
- [10181](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_layout_000_large_inline_10181.png)
- [10221](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_layout_000_large_inline_10221.png)
- [10261](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_layout_000_large_inline_10261.png)
- [10301](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_layout_000_large_inline_10301.png)
- [10341](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_layout_000_large_inline_10341.png)

#### Crossline

- [1997](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_layout_000_large_crossline_1997.png)
- [2069](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_layout_000_large_crossline_2069.png)
- [2141](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_layout_000_large_crossline_2141.png)
- [2213](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_layout_000_large_crossline_2213.png)
- [2285](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_layout_000_large_crossline_2285.png)
- [2356](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_layout_000_large_crossline_2356.png)
- [2428](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_layout_000_large_crossline_2428.png)
- [2500](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_layout_000_large_crossline_2500.png)
- [2572](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_layout_000_large_crossline_2572.png)
- [2644](../../artifacts/seis_ssl_cluster/studies/hmm_v2/canonical_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_layout_000_large_crossline_2644.png)

## 出力識別情報

- f3 manifest SHA-256：`a4c8099189c471a3a74e9d3e2d5eaee3a2de8fbee83a71c1659e43f1d1c5c48d`
- parihaka manifest SHA-256：`42ca9b60a74f67c2eec2f3e0632139fadf1efccc8ac3bc09b820d1d0ec7b43d8`
- volve manifest SHA-256：`3aed4a4c10a8f63865257659b25f606f8f7f832a453638efa9f580ba9742cbf1`
