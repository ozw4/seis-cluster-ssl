# F3 token-level lithology probe report

このreportはF3 token-level lithology probeの既存artifactを統合し、pretrained model、AGC有無、probe種別の比較に使う。

## Dataset

- F3 shape: [601, 901, 255]
- classes: 6
- label source of truth: segy_label_volume
- PNG label role: train_validation_slice_selection_and_visual_qc
- train/validation slices: {"train": ["inline 250", "inline 350", "inline 450", "inline 550", "inline 650", "crossline 450", "crossline 550", "crossline 650", "crossline 850", "crossline 950", "crossline 1050", "crossline 1150"], "validation": ["inline 150", "crossline 350", "crossline 750"]}
- tokenization thresholds: {"ignore_z_border_samples": 1, "min_labeled_fraction": 0.5, "min_majority_fraction": 0.7}
- class imbalance: {"class_counts": {}, "max_to_min_positive_ratio": null, "total": 0}

| class_id | class_name | rgb |
|---:|---|---|
| 0 | Upper North Sea | [35, 92, 167] |
| 1 | Middle North Sea | [125, 180, 213] |
| 2 | Lower North Sea | [219, 241, 247] |
| 3 | Rijnland/Chalk | [254, 219, 124] |
| 4 | Scruff | [252, 120, 59] |
| 5 | Zechstein | [208, 10, 0] |

## Pretrained encoder

- MODEL_TAG: amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1
- checkpoint path: 未確認
- EMBED_SPEC: overlap_x16
- AGC有無: True
- visible loss有無: True
- mask ratio: 0.7500
- encoder fine-tuning: False

## Token dataset

- train token count: 28724
- validation token count: 7003
- class counts: {"combined": {"0": 7654, "1": 4089, "2": 19851, "3": 1814, "4": 1824, "5": 495}, "train": {"0": 6428, "1": 3286, "2": 15942, "3": 1460, "4": 1159, "5": 449}, "validation": {"0": 1226, "1": 803, "2": 3909, "3": 354, "4": 665, "5": 46}}
- dropped token ratio: 0.1717
- ambiguous token ratio: 0.0415

## Probe

- PROBE_SPEC: linear_balanced_v1
- classifier type: logistic_regression
- feature scaling: standard
- class weighting: balanced
- hyperparameters: {"batch_size": 1024, "dropout": 0.2, "early_stopping_patience": 20, "hidden_dims": [256, 128], "learning_rate": 0.001, "max_epochs": 200, "max_iter": 2000, "random_state": 42, "weight_decay": 0.0}

## Metrics

- accuracy: 0.8865
- balanced accuracy: 0.8438
- macro F1: 0.7533
- weighted F1: 0.8955
- mean IoU: 0.6501

| class_id | class_name | F1 | IoU | support |
|---:|---|---:|---:|---:|
| 0 | Upper North Sea | 0.9653 | 0.9330 | 1226 |
| 1 | Middle North Sea | 0.9311 | 0.8712 | 803 |
| 2 | Lower North Sea | 0.9267 | 0.8635 | 3909 |
| 3 | Rijnland/Chalk | 0.5349 | 0.3651 | 354 |
| 4 | Scruff | 0.7663 | 0.6211 | 665 |
| 5 | Zechstein | 0.3956 | 0.2466 | 46 |

- confusion matrix:

```text
[[1184, 27, 7, 1, 7, 0], [12, 764, 26, 0, 1, 0], [30, 47, 3440, 222, 148, 22], [1, 0, 12, 230, 62, 49], [0, 0, 30, 52, 554, 29], [0, 0, 0, 1, 9, 36]]
```

## Figures

- [confusion_matrix](figures/confusion_matrix.png)
- [per_class_f1](figures/per_class_f1.png)
- [validation_slice_inline_150](figures/validation_inline_0150_prediction.png)
- [validation_slice_crossline_350](figures/validation_crossline_0350_prediction.png)
- [validation_slice_crossline_750](figures/validation_crossline_0750_prediction.png)

## Interpretation

### 良い点

- weighted F1は0.8955で、頻出classの性能を確認できる。
- balanced accuracyは0.8438で、class imbalanceを考慮した比較指標になる。

### 失敗しているclass

- class 5 Zechstein: F1=0.3956, IoU=0.2466
- class 3 Rijnland/Chalk: F1=0.5349, IoU=0.3651

### class imbalanceの影響

- class countの最大/最小比が40.1で、minor classのF1低下に注意する。

### AGCあり/なし比較

- このrunはAGCありとして集計される。AGCあり/なしの優劣はcomparison_table.csvで同じEMBED_SPEC、LABEL_SET、PROBE_SPECを揃えて比較する。

### 次の改善候補

- comparison_table.csvでMODEL_TAG、EMBED_SPEC、PROBE_SPECごとのmacro F1とmean IoUを比較する。
- 低F1 classは教師slice追加、tokenization閾値、class weightingの影響を切り分ける。
- linear probeで頭打ちなら同じfrozen encoder上でMLP probeを比較する。

## Warnings

- none
