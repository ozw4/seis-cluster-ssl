# F3 lithology probe comparison report

集約run数: 4

## Comparison table

| feature_kind | MODEL_TAG | BASELINE_TAG | EMBED_SPEC | LABEL_SET | PROBE_SPEC | FEATURE_SOURCE_KIND | FEATURE_SOURCE_REFERENCE_MODEL_TAG | FEATURE_SOURCE_EMBED_SPEC | FEATURE_SOURCE_DESCRIPTION | accuracy | balanced_accuracy | macro_f1 | weighted_f1 | mean_iou | class_0_f1 | class_1_f1 | class_2_f1 | class_3_f1 | class_4_f1 | class_5_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pretrained_encoder | amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1 |  | overlap_x16 | png_slices_segy_labels_v1 | linear_balanced_v1 | pretrained_encoder | 未確認 | overlap_x16 | 未確認 | 0.8872 | 0.8436 | 0.7537 | 0.8961 | 0.6509 | 0.9653 | 0.9322 | 0.9275 | 0.5332 | 0.7687 | 0.3956 |
| z_only |  | z_only_v1 | z_only_degree1 | png_slices_segy_labels_v1 | linear_balanced_v1 | z_only | amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1 | overlap_x16 | normalized token center z with polynomial degree 1 | 0.5483 | 0.5541 | 0.3671 | 0.5797 | 0.2734 | 0.8400 | 0.4046 | 0.6730 | 0.1991 | 0.0000 | 0.0859 |
| amplitude_stats |  | amplitude_stats_v1 | amplitude_stats_v1 | png_slices_segy_labels_v1 | linear_balanced_v1 | amplitude_stats | amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1 | overlap_x16 | handcrafted seismic amplitude block statistics | 0.4765 | 0.3941 | 0.3327 | 0.5231 | 0.2193 | 0.4339 | 0.4818 | 0.6551 | 0.2275 | 0.1526 | 0.0457 |
| random_encoder |  | random_encoder_amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_seed42_v1 | overlap_x16 | png_slices_segy_labels_v1 | linear_balanced_v1 | random_encoder | amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1 | overlap_x16 | same MAE architecture with random seed 42 and no pretraining | 0.8178 | 0.7944 | 0.6927 | 0.8293 | 0.5656 | 0.9497 | 0.7929 | 0.8595 | 0.4988 | 0.6809 | 0.3743 |

## Figures

- [macro_f1_comparison](figures/macro_f1_comparison.png)
- [mean_iou_comparison](figures/mean_iou_comparison.png)
- [per_class_f1_comparison](figures/per_class_f1_comparison.png)

## Interpretation

- pretrained encoderがz-onlyを上回るか: 上回る (macro F1差分 +0.3866, mean IoU差分 +0.3775)。
- pretrained encoderがamplitude-onlyを上回るか: 上回る (macro F1差分 +0.4210, mean IoU差分 +0.4316)。
- pretrained encoderがrandom encoderを上回るか: 上回る (macro F1差分 +0.0611, mean IoU差分 +0.0853)。
- class 3/5など弱いclassで改善があるか: class 3: F1差分 +0.0344、class 5: F1差分 +0.0213。
- F3 faciesが深度だけで説明できる程度: z-onlyとの差がある (macro F1差分 +0.3866) ため、深度以外の特徴が効いている可能性がある。

## Warnings

- none
