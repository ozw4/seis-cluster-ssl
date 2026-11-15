# v2_dis0: 全断面比較図

F3 30枚・Parihaka 30枚・Volve 20枚、計80枚。

v2と同じlayout_000/large、断面位置・表示範囲・振幅スケール・配色・
2x4配置・180x105 mm/300 dpiです。蒸留係数だけを0にした
MAE・Barlow Twins・Randomを、それぞれHMMなしの親モデルと並べています。

Barlow Twinsは各データセットで25 epoch学習・特徴抽出と、図に使用する
layout_000/largeのdecoder学習を追加しました。他のレイアウト・教師量は
今回の追加実験の対象に含めていません。MAE・Randomは既存結果を再利用しました。

K=[6,8,10]・整合性0.1。元のv2モデル・図は保持しています。
ParihakaのHMMなしMAEは指定済みdecoder seed 42001、他は42000です。
全断面の定性的比較であり、新たな独立test領域を示す図ではありません。

公開ファイルはREADME.mdの1点のみ。予測配列と全図はartifactsに保存し、
このレビュー文書はパイプライン入力には使用しません。

## F3

[manifest.json](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/manifest.json)

[selected_slices.csv](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/selected_slices.csv)

### Inline

- [130.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_dis0_layout_000_large_inline_130.png)
- [190.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_dis0_layout_000_large_inline_190.png)
- [249.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_dis0_layout_000_large_inline_249.png)
- [310.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_dis0_layout_000_large_inline_310.png)
- [370.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_dis0_layout_000_large_inline_370.png)
- [430.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_dis0_layout_000_large_inline_430.png)
- [490.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_dis0_layout_000_large_inline_490.png)
- [549.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_dis0_layout_000_large_inline_549.png)
- [610.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_dis0_layout_000_large_inline_610.png)
- [670.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/inline/f3_hmm_v2_dis0_layout_000_large_inline_670.png)

### Crossline

- [345.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_dis0_layout_000_large_crossline_345.png)
- [435.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_dis0_layout_000_large_crossline_435.png)
- [525.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_dis0_layout_000_large_crossline_525.png)
- [615.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_dis0_layout_000_large_crossline_615.png)
- [705.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_dis0_layout_000_large_crossline_705.png)
- [795.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_dis0_layout_000_large_crossline_795.png)
- [885.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_dis0_layout_000_large_crossline_885.png)
- [975.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_dis0_layout_000_large_crossline_975.png)
- [1065.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_dis0_layout_000_large_crossline_1065.png)
- [1155.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/crossline/f3_hmm_v2_dis0_layout_000_large_crossline_1155.png)

### Time

- [0.08](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_dis0_layout_000_large_time_0p080s_sample_0016.png)
- [0.19](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_dis0_layout_000_large_time_0p190s_sample_0038.png)
- [0.315](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_dis0_layout_000_large_time_0p315s_sample_0063.png)
- [0.445](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_dis0_layout_000_large_time_0p445s_sample_0089.png)
- [0.5700000000000001](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_dis0_layout_000_large_time_0p570s_sample_0114.png)
- [0.7000000000000001](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_dis0_layout_000_large_time_0p700s_sample_0140.png)
- [0.8250000000000001](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_dis0_layout_000_large_time_0p825s_sample_0165.png)
- [0.9500000000000001](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_dis0_layout_000_large_time_0p950s_sample_0190.png)
- [1.08](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_dis0_layout_000_large_time_1p080s_sample_0216.png)
- [1.195](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/f3/layout_000/large/time/f3_hmm_v2_dis0_layout_000_large_time_1p195s_sample_0239.png)

## Parihaka

[manifest.json](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/manifest.json)

[selected_slices.csv](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/selected_slices.csv)

### Inline

- [41.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_dis0_layout_000_large_X_0041.png)
- [118.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_dis0_layout_000_large_X_0118.png)
- [196.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_dis0_layout_000_large_X_0196.png)
- [274.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_dis0_layout_000_large_X_0274.png)
- [352.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_dis0_layout_000_large_X_0352.png)
- [431.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_dis0_layout_000_large_X_0431.png)
- [509.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_dis0_layout_000_large_X_0509.png)
- [587.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_dis0_layout_000_large_X_0587.png)
- [665.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_dis0_layout_000_large_X_0665.png)
- [742.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/inline/parihaka_hmm_v2_dis0_layout_000_large_X_0742.png)

### Crossline

- [31.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_dis0_layout_000_large_Y_0031.png)
- [89.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_dis0_layout_000_large_Y_0089.png)
- [148.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_dis0_layout_000_large_Y_0148.png)
- [207.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_dis0_layout_000_large_Y_0207.png)
- [266.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_dis0_layout_000_large_Y_0266.png)
- [325.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_dis0_layout_000_large_Y_0325.png)
- [384.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_dis0_layout_000_large_Y_0384.png)
- [413.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_dis0_layout_000_large_Y_0413.png)
- [412.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_dis0_layout_000_large_Y_0412.png)
- [411.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/crossline/parihaka_hmm_v2_dis0_layout_000_large_Y_0411.png)

### Time

- [335.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_dis0_layout_000_large_Z_0335.png)
- [336.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_dis0_layout_000_large_Z_0336.png)
- [337.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_dis0_layout_000_large_Z_0337.png)
- [353.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_dis0_layout_000_large_Z_0353.png)
- [453.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_dis0_layout_000_large_Z_0453.png)
- [554.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_dis0_layout_000_large_Z_0554.png)
- [582.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_dis0_layout_000_large_Z_0582.png)
- [581.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_dis0_layout_000_large_Z_0581.png)
- [580.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_dis0_layout_000_large_Z_0580.png)
- [579.0](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/parihaka/layout_000/large/time/parihaka_hmm_v2_dis0_layout_000_large_Z_0579.png)

## Volve

[manifest.json](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/manifest.json)

[selected_slices.csv](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/selected_slices.csv)

### Inline

- [9981](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_dis0_layout_000_large_inline_9981.png)
- [10021](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_dis0_layout_000_large_inline_10021.png)
- [10061](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_dis0_layout_000_large_inline_10061.png)
- [10101](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_dis0_layout_000_large_inline_10101.png)
- [10141](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_dis0_layout_000_large_inline_10141.png)
- [10181](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_dis0_layout_000_large_inline_10181.png)
- [10221](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_dis0_layout_000_large_inline_10221.png)
- [10261](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_dis0_layout_000_large_inline_10261.png)
- [10301](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_dis0_layout_000_large_inline_10301.png)
- [10341](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/inline/volve_hmm_v2_dis0_layout_000_large_inline_10341.png)

### Crossline

- [1997](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_dis0_layout_000_large_crossline_1997.png)
- [2069](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_dis0_layout_000_large_crossline_2069.png)
- [2141](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_dis0_layout_000_large_crossline_2141.png)
- [2213](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_dis0_layout_000_large_crossline_2213.png)
- [2285](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_dis0_layout_000_large_crossline_2285.png)
- [2356](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_dis0_layout_000_large_crossline_2356.png)
- [2428](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_dis0_layout_000_large_crossline_2428.png)
- [2500](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_dis0_layout_000_large_crossline_2500.png)
- [2572](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_dis0_layout_000_large_crossline_2572.png)
- [2644](../../artifacts/seis_ssl_cluster/studies/hmm_v2/zero_distillation_paper_sections_v1/figures/volve/layout_000/large/crossline/volve_hmm_v2_dis0_layout_000_large_crossline_2644.png)
