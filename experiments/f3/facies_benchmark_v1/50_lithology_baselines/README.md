# F3 lithology baselines

Experiment hierarchy and config contract for comparing the F3 token-level
lithology probe against simple baselines. This stage is for checking whether the
pretrained MAE embedding adds value beyond token position, local amplitude
statistics, or the same MAE architecture with random weights.

This directory defines the layout and metadata contract only. The z-only,
amplitude-statistics, and random-checkpoint creation implementations are out of
scope for this issue.

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

## Experiment Layout

```text
$EXP/50_lithology_baselines/
├── README.md
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

Baseline token datasets, probes, and per-run reports are written under
`lithology/f3/.../baselines`:

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

The existing pretrained lithology hierarchy remains unchanged:

```text
$ROOT/lithology/f3/facies_benchmark_v1/$REFERENCE_MODEL_TAG/$EMBED_SPEC/$LABEL_SET/
```

## Shared Split Contract

All baselines reuse the reference train/validation token split and label
selection:

```text
$ROOT/lithology/f3/facies_benchmark_v1/$REFERENCE_MODEL_TAG/$EMBED_SPEC/$LABEL_SET/token_dataset/
  splits.json
  token_dataset_metadata.json
  train_tokens.npz
  validation_tokens.npz
```

The label filters remain the same as the pretrained MVP:

```yaml
tokenization:
  min_labeled_fraction: 0.5
  min_majority_fraction: 0.7
  ignore_z_border_samples: 1
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

## Comparison Report Contract

Baseline comparison reports aggregate metrics from both the existing pretrained
run and the baseline runs:

```text
$ROOT/lithology/f3/facies_benchmark_v1/reports/baseline_comparison_v1/
  comparison_table.csv
  comparison_report.md
```

The comparison table must include:

```text
MODEL_TAG
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

## Run Order

The z-only baseline uses normalized token center `z` as its feature. Polynomial
degree defaults to `1`.

```bash
python proc/seis_ssl_cluster/build_f3_lithology_baseline_token_dataset.py \
  --config $EXP/50_lithology_baselines/$Z_BASELINE_TAG/01_build_baseline_token_dataset.yaml

python proc/seis_ssl_cluster/train_f3_lithology_probe.py \
  --config $EXP/50_lithology_baselines/$Z_BASELINE_TAG/02_train_linear_probe.yaml

python proc/seis_ssl_cluster/build_f3_lithology_report.py \
  --config $EXP/50_lithology_baselines/$Z_BASELINE_TAG/03_build_report.yaml
```

The amplitude-statistics baseline uses per-token seismic block statistics:
`mean`, `std`, `rms`, `abs_mean`, `min`, `max`, `p10`, `p50`, and `p90`.

```bash
python proc/seis_ssl_cluster/build_f3_lithology_baseline_token_dataset.py \
  --config $EXP/50_lithology_baselines/$AMP_BASELINE_TAG/01_build_baseline_token_dataset.yaml

python proc/seis_ssl_cluster/train_f3_lithology_probe.py \
  --config $EXP/50_lithology_baselines/$AMP_BASELINE_TAG/02_train_linear_probe.yaml

python proc/seis_ssl_cluster/build_f3_lithology_report.py \
  --config $EXP/50_lithology_baselines/$AMP_BASELINE_TAG/03_build_report.yaml
```

The random-encoder baseline uses the same MAE architecture as
`$REFERENCE_MODEL_TAG`, initialized with seed `42` and no pretraining:

```bash
python proc/seis_ssl_cluster/create_f3_random_mae_checkpoint.py \
  --config $EXP/50_lithology_baselines/$RANDOM_ENCODER_TAG/01_create_random_checkpoint.yaml

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config $EXP/50_lithology_baselines/$RANDOM_ENCODER_TAG/02_extract_embeddings.yaml

python proc/seis_ssl_cluster/build_f3_lithology_token_dataset.py \
  --config $EXP/50_lithology_baselines/$RANDOM_ENCODER_TAG/03_build_token_dataset.yaml

python proc/seis_ssl_cluster/train_f3_lithology_probe.py \
  --config $EXP/50_lithology_baselines/$RANDOM_ENCODER_TAG/04_train_linear_probe.yaml

python proc/seis_ssl_cluster/build_f3_lithology_report.py \
  --config $EXP/50_lithology_baselines/$RANDOM_ENCODER_TAG/05_build_report.yaml
```

`build_f3_lithology_baseline_token_dataset.py` and
`create_f3_random_mae_checkpoint.py` are reserved entrypoints for the later
implementation issues.
