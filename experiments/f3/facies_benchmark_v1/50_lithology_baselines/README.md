# F3 lithology baselines

Experiment hierarchy, artifact contract, and runbook for comparing the F3
token-level lithology probe against simple baselines. The goal is to check
whether the pretrained MAE embedding adds value beyond token position, local
amplitude statistics, or the same MAE architecture with random weights.

Source-of-truth inputs:

- Raw F3 root: `/home/dcuser/data/public_data/field/F3`
- Label source of truth: `/home/dcuser/data/public_data/field/F3/f3_labels.sgy`
  and `/workspace/artifacts/seis_ssl_cluster/registry/volumes/f3/facies_benchmark_v1/f3_facies_labels.npy`
- Artifact root: `/workspace/artifacts/seis_ssl_cluster`
- Reference pretrained token dataset:
  `/workspace/artifacts/seis_ssl_cluster/lithology/f3/facies_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/overlap_x16/png_slices_segy_labels_v1/token_dataset`

PNG labels are used only for train/validation slice selection and visual QC.
They are not the source of truth for voxel labels.

Fixed variables:

```bash
ROOT=/workspace/artifacts/seis_ssl_cluster
EXP=experiments/f3/facies_benchmark_v1

REFERENCE_MODEL_TAG=amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1
EMBED_SPEC=overlap_x16
LABEL_SET=png_slices_segy_labels_v1
PROBE_SPEC=linear_balanced_v1

Z_BASELINE_TAG=z_only_v1
AMP_BASELINE_TAG=amplitude_stats_v1
RANDOM_ENCODER_TAG=random_encoder_amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_seed42_v1
```

Each YAML is standalone and avoids inheritance, anchors, merge keys, and
symlinks. Raw YAML does not contain a top-level `stage`; the selected proc
entrypoint owns the stage identity.

Do not use `runs/` for baseline artifacts.

## Prerequisites

These stages must already be complete:

- F3 inspection
- F3 volume preparation
- pretrained encoder F3 embedding extraction
- pretrained encoder token dataset build
- pretrained encoder linear probe training

All baselines reuse the existing train/validation token split and label
selection. The label filters remain the same as the pretrained MVP:

```yaml
tokenization:
  min_labeled_fraction: 0.5
  min_majority_fraction: 0.7
  ignore_z_border_samples: 1
```

## Experiment Layout

```text
$EXP/50_lithology_baselines/
├── README.md
├── 05_build_baseline_comparison_report.yaml
├── z_only_v1/
│   ├── 01_build_baseline_token_dataset.yaml
│   ├── 02_train_linear_probe.yaml
│   └── 03_build_report.yaml
├── amplitude_stats_v1/
│   ├── 01_build_baseline_token_dataset.yaml
│   ├── 02_train_linear_probe.yaml
│   └── 03_build_report.yaml
└── random_encoder_amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_seed42_v1/
    ├── 01_create_random_checkpoint.yaml
    ├── 02_extract_embeddings.yaml
    ├── 03_build_token_dataset.yaml
    ├── 04_train_linear_probe.yaml
    └── 05_build_report.yaml
```

## Artifact Layout

Z-only and amplitude-statistics token datasets, probes, and per-run reports are
written under `lithology/f3/.../baselines`:

```text
$ROOT/lithology/f3/facies_benchmark_v1/baselines/$BASELINE_TAG/$LABEL_SET/
  token_dataset/
  probes/$PROBE_SPEC/
  reports/$PROBE_SPEC/
```

The random encoder checkpoint and embeddings are written outside `runs/`:

```text
$ROOT/pretraining/f3/facies_benchmark_v1/$RANDOM_ENCODER_TAG/random_init/mae_random_seed42.pt
$ROOT/embeddings/f3/facies_benchmark_v1/$RANDOM_ENCODER_TAG/$EMBED_SPEC/
```

The random encoder lithology token dataset, probes, and per-run reports follow
the encoder embedding hierarchy:

```text
$ROOT/lithology/f3/facies_benchmark_v1/$RANDOM_ENCODER_TAG/$EMBED_SPEC/$LABEL_SET/
  token_dataset/
  probes/$PROBE_SPEC/
  reports/$PROBE_SPEC/
```

The existing pretrained lithology hierarchy remains unchanged:

```text
$ROOT/lithology/f3/facies_benchmark_v1/$REFERENCE_MODEL_TAG/$EMBED_SPEC/$LABEL_SET/
```

## Feature Source Metadata

Every baseline token dataset metadata file and probe `metrics.json` must include
this object:

```json
{
  "feature_source": {
    "kind": "z_only | amplitude_stats | pretrained_encoder | random_encoder",
    "reference_model_tag": "...",
    "embedding_spec": "...",
    "description": "..."
  }
}
```

For z-only and amplitude-statistics baselines, `embedding_spec` records the
reference token grid spec (`overlap_x16`) whose split and label selection are
being reused. For random encoder baselines, it records the random embedding
spec.

## Runbook

Run these commands in order from the repository root.

1. Build the z-only baseline dataset.

```bash
python proc/seis_ssl_cluster/build_f3_lithology_baseline_features.py \
  --config experiments/f3/facies_benchmark_v1/50_lithology_baselines/z_only_v1/01_build_baseline_token_dataset.yaml
```

2. Train the z-only linear probe.

```bash
python proc/seis_ssl_cluster/train_f3_lithology_probe.py \
  --config experiments/f3/facies_benchmark_v1/50_lithology_baselines/z_only_v1/02_train_linear_probe.yaml
```

3. Build the amplitude-only baseline dataset.

```bash
python proc/seis_ssl_cluster/build_f3_lithology_baseline_features.py \
  --config experiments/f3/facies_benchmark_v1/50_lithology_baselines/amplitude_stats_v1/01_build_baseline_token_dataset.yaml
```

4. Train the amplitude-only linear probe.

```bash
python proc/seis_ssl_cluster/train_f3_lithology_probe.py \
  --config experiments/f3/facies_benchmark_v1/50_lithology_baselines/amplitude_stats_v1/02_train_linear_probe.yaml
```

5. Create the random encoder checkpoint.

```bash
python proc/seis_ssl_cluster/create_random_mae_checkpoint.py \
  --config experiments/f3/facies_benchmark_v1/50_lithology_baselines/random_encoder_amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_seed42_v1/01_create_random_checkpoint.yaml
```

6. Extract random encoder embeddings.

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config experiments/f3/facies_benchmark_v1/50_lithology_baselines/random_encoder_amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_seed42_v1/02_extract_embeddings.yaml \
  --device cuda \
  --skip-existing
```

7. Build the random encoder token dataset.

```bash
python proc/seis_ssl_cluster/build_f3_lithology_token_dataset.py \
  --config experiments/f3/facies_benchmark_v1/50_lithology_baselines/random_encoder_amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_seed42_v1/03_build_token_dataset.yaml
```

8. Train the random encoder linear probe.

```bash
python proc/seis_ssl_cluster/train_f3_lithology_probe.py \
  --config experiments/f3/facies_benchmark_v1/50_lithology_baselines/random_encoder_amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_seed42_v1/04_train_linear_probe.yaml
```

9. Build the pretrained-vs-baseline comparison report.

```bash
python proc/seis_ssl_cluster/build_f3_lithology_comparison_report.py \
  --config experiments/f3/facies_benchmark_v1/50_lithology_baselines/05_build_baseline_comparison_report.yaml
```

The checked-in `03_build_report.yaml` and `05_build_report.yaml` files can still
be used for per-run reports, but they are not required before the comparison
report because the comparison report reads probe `metrics.json` files directly.

## Comparison Report

The comparison command writes:

```text
$ROOT/lithology/f3/facies_benchmark_v1/reports/baseline_comparison/
  comparison_table.csv
  comparison_report.md
  figures/
```

`comparison_table.csv` contains the run identity, feature-source metadata,
overall metrics, and per-class F1 columns:

```text
feature_kind
MODEL_TAG
BASELINE_TAG
EMBED_SPEC
LABEL_SET
PROBE_SPEC
FEATURE_SOURCE_KIND
FEATURE_SOURCE_REFERENCE_MODEL_TAG
FEATURE_SOURCE_EMBED_SPEC
FEATURE_SOURCE_DESCRIPTION
accuracy
balanced_accuracy
macro_f1
weighted_f1
mean_iou
class_<ID>_f1
```

Use `macro_f1` as the primary class-balanced comparison, `mean_iou` as the
secondary segmentation-style metric, and `class_<ID>_f1` to check whether weak
classes improve. The Markdown report sorts rows by feature type and includes
figures for macro F1, mean IoU, and per-class F1. Its `Warnings` section lists
missing metrics or missing input components; do not treat an incomplete row as
evidence for or against pretraining.

## Interpretation Guide

- z-onlyが高い場合、F3 facies分類の多くが深度/層序位置で説明できる可能性がある。
- amplitude-onlyが高い場合、pretrained embeddingの価値は限定的。
- random encoderが高い場合、architectureやtokenizationだけで十分な可能性がある。
- pretrained encoderが全baselineを上回れば、事前学習の有効性を主張しやすい。

Read the final comparison as a set of controls, not as isolated scores:

- `pretrained_encoder` versus `z_only`: separates learned representation value
  from depth or stratigraphic-position signal.
- `pretrained_encoder` versus `amplitude_stats`: checks whether simple local
  amplitude statistics already explain the labels.
- `pretrained_encoder` versus `random_encoder`: checks whether the trained
  weights add value beyond architecture, patching, and tokenization.
- Per-class F1 deltas: identify whether gains are broad or limited to frequent
  classes.
