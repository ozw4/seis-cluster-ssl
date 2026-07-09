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

- MODEL_TAG: strat_hmm_pretext_m1_k6_topblock1_distill
- checkpoint path: 未確認
- EMBED_SPEC: overlap_x16
- AGC有無: False
- visible loss有無: 未確認
- mask ratio: 未確認
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

- accuracy: 0.8963
- balanced accuracy: 0.8310
- macro F1: 0.7586
- weighted F1: 0.9044
- mean IoU: 0.6609

| class_id | class_name | F1 | IoU | support |
|---:|---|---:|---:|---:|
| 0 | Upper North Sea | 0.9816 | 0.9639 | 1226 |
| 1 | Middle North Sea | 0.9378 | 0.8830 | 803 |
| 2 | Lower North Sea | 0.9339 | 0.8761 | 3909 |
| 3 | Rijnland/Chalk | 0.5346 | 0.3648 | 354 |
| 4 | Scruff | 0.7811 | 0.6408 | 665 |
| 5 | Zechstein | 0.3827 | 0.2366 | 46 |

- confusion matrix:

```text
[[1202, 22, 1, 0, 1, 0], [14, 762, 26, 0, 1, 0], [7, 38, 3492, 226, 132, 14], [0, 0, 16, 228, 67, 43], [0, 0, 34, 41, 562, 28], [0, 0, 0, 4, 11, 31]]
```

## Figures

- [confusion_matrix](figures/confusion_matrix.png)
- [per_class_f1](figures/per_class_f1.png)

## Interpretation

### 良い点

- weighted F1は0.9044で、頻出classの性能を確認できる。
- balanced accuracyは0.8310で、class imbalanceを考慮した比較指標になる。

### 失敗しているclass

- class 5 Zechstein: F1=0.3827, IoU=0.2366
- class 3 Rijnland/Chalk: F1=0.5346, IoU=0.3648

### class imbalanceの影響

- class countの最大/最小比が40.1で、minor classのF1低下に注意する。

### AGCあり/なし比較

- このrunはAGCなしとして集計される。AGCあり/なしの優劣はcomparison_table.csvで同じEMBED_SPEC、LABEL_SET、PROBE_SPECを揃えて比較する。

### 次の改善候補

- comparison_table.csvでMODEL_TAG、EMBED_SPEC、PROBE_SPECごとのmacro F1とmean IoUを比較する。
- 低F1 classは教師slice追加、tokenization閾値、class weightingの影響を切り分ける。
- linear probeで頭打ちなら同じfrozen encoder上でMLP probeを比較する。

## Warnings

- none
