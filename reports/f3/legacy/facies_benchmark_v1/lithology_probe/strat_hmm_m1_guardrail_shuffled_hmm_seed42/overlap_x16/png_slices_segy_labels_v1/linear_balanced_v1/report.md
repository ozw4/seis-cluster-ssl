# F3 token-level lithology probe report

## Dataset

- F3 shape: [601, 901, 255]
- classes: 6
- label source of truth: segy_label_volume
- PNG label role: train_validation_slice_selection_and_visual_qc
- train/validation slices: {"train": ["inline 250", "inline 350", "inline 450", "inline 550", "inline 650", "crossline 450", "crossline 550", "crossline 650", "crossline 850", "crossline 950", "crossline 1050", "crossline 1150"], "validation": ["inline 150", "crossline 350", "crossline 750"]}
- tokenization thresholds: {"ignore_z_border_samples": 1, "min_labeled_fraction": 0.5, "min_majority_fraction": 0.7}
- class imbalance: {"class_counts": {"0": 7654, "1": 4089, "2": 19851, "3": 1814, "4": 1824, "5": 495}, "max_to_min_positive_ratio": 40.1030303030303, "total": 35727}

| class_id | class_name | rgb |
|---:|---|---|
| 0 | Upper North Sea | [35, 92, 167] |
| 1 | Middle North Sea | [125, 180, 213] |
| 2 | Lower North Sea | [219, 241, 247] |
| 3 | Rijnland/Chalk | [254, 219, 124] |
| 4 | Scruff | [252, 120, 59] |
| 5 | Zechstein | [208, 10, 0] |

## Pretrained encoder

- MODEL_TAG: strat_hmm_m1_guardrail_shuffled_hmm_seed42
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

- accuracy: 0.8878
- balanced accuracy: 0.8416
- macro F1: 0.7549
- weighted F1: 0.8962
- mean IoU: 0.6520

| class_id | class_name | F1 | IoU | support |
|---:|---|---:|---:|---:|
| 0 | Upper North Sea | 0.9686 | 0.9391 | 1226 |
| 1 | Middle North Sea | 0.9317 | 0.8721 | 803 |
| 2 | Lower North Sea | 0.9262 | 0.8626 | 3909 |
| 3 | Rijnland/Chalk | 0.5489 | 0.3782 | 354 |
| 4 | Scruff | 0.7632 | 0.6170 | 665 |
| 5 | Zechstein | 0.3911 | 0.2431 | 46 |

- confusion matrix:

```text
[[1188, 28, 6, 0, 4, 0], [12, 764, 26, 0, 1, 0], [27, 45, 3446, 216, 150, 25], [0, 0, 12, 233, 64, 45], [0, 0, 42, 44, 551, 28], [0, 0, 0, 2, 9, 35]]
```

## Figures

- [confusion_matrix](figures/confusion_matrix.png)
- [per_class_f1](figures/per_class_f1.png)

## Warnings

- none
