# F3 token-level lithology probe

Experiment hierarchy and config contract for the F3 few-label token-level
lithology probe. This stage uses the NOPIMS-pretrained MAE encoder as a frozen
feature extractor and trains lightweight classifiers on F3 2D supervised slices.

Source-of-truth inputs:

- Raw F3 root: `/home/dcuser/data/public_data/field/F3`
- Label source of truth: `/home/dcuser/data/public_data/field/F3/f3_labels.sgy`
  and the converted label volume
- Artifact root: `/workspace/artifacts/seis_ssl_cluster`
- Frozen pretraining checkpoint:
  `/workspace/artifacts/seis_ssl_cluster/pretraining/nopims/pretrain_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full_100ep/mae_best.pt`

The configs intentionally require `mae_best.pt`; if it is absent, stop and
choose an explicit checkpoint instead of falling back silently.

PNG labels are used for train/validation slice selection and visual QC only.
They are not the source of truth for voxel labels.

If train and validation slices retain the same `token_xyz`, validation keeps
precedence and matching train rows are removed before `train_tokens.npz` is
written. The token dataset metadata records the removed row count.

Fixed variables for the MVP:

```bash
ROOT=/workspace/artifacts/seis_ssl_cluster
EXP=experiments/f3/facies_benchmark_v1

MODEL_TAG=amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1
EMBED_SPEC=overlap_x16
LABEL_SET=png_slices_segy_labels_v1
PROBE_SPEC=linear_balanced_v1
```

The canonical config directory is:

```text
$EXP/50_lithology/$MODEL_TAG/$EMBED_SPEC/$LABEL_SET/
```

Each YAML is standalone and avoids inheritance, anchors, merge keys, and
symlinks. Raw YAML does not contain a top-level `stage`; the selected proc
entrypoint owns the stage identity.

Shared top-level config contract:

```yaml
paths:
  f3_root: /home/dcuser/data/public_data/field/F3
  artifact_root: /workspace/artifacts/seis_ssl_cluster

dataset:
  name: f3_facies_benchmark
  version: facies_benchmark_v1

model:
  tag: amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1
```

## Artifact Layout

Registry artifacts:

```text
$ROOT/registry/volumes/f3/facies_benchmark_v1/
  f3_seismic.npy
  f3_facies_labels.npy
  f3_metadata.json

$ROOT/registry/manifests/f3/facies_benchmark_v1/
  f3_amplitude_manifest.json

$ROOT/registry/normalization_stats/f3/facies_benchmark_v1/
  f3_seismic.normalization_stats.json
```

Downstream artifacts:

```text
$ROOT/embeddings/f3/facies_benchmark_v1/$MODEL_TAG/$EMBED_SPEC/

$ROOT/lithology/f3/facies_benchmark_v1/$MODEL_TAG/$EMBED_SPEC/$LABEL_SET/
  token_dataset/
  probes/$PROBE_SPEC/
    probe.joblib
    scaler.joblib
  predictions/$PROBE_SPEC/
  visualizations/$PROBE_SPEC/
  reports/$PROBE_SPEC/
```

Artifact roles:

- `pretraining/` stores frozen NOPIMS MAE checkpoints, resolved configs, and
  training debug outputs. F3 lithology MVP reads this checkpoint but does not
  fine-tune the encoder.
- `embeddings/` stores extracted F3 encoder token embeddings for a fixed
  `MODEL_TAG` and `EMBED_SPEC`.
- `lithology/` stores the F3 token dataset, probe checkpoints, predictions,
  figures, and reports for a fixed `LABEL_SET` and `PROBE_SPEC`.

Do not use `runs/` for this downstream path convention.

## Runbook

Run the downstream MVP stages in order:

```bash
python proc/seis_ssl_cluster/prepare_f3_facies_volume.py \
  --config <f3-volume-registry-config>

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config <f3-embedding-config>

python proc/seis_ssl_cluster/build_f3_lithology_token_dataset.py \
  --config $EXP/50_lithology/$MODEL_TAG/$EMBED_SPEC/$LABEL_SET/01_build_token_dataset.yaml

python proc/seis_ssl_cluster/train_f3_lithology_probe.py \
  --config $EXP/50_lithology/$MODEL_TAG/$EMBED_SPEC/$LABEL_SET/02_train_linear_probe.yaml

python proc/seis_ssl_cluster/predict_f3_lithology_tokens.py \
  --config $EXP/50_lithology/$MODEL_TAG/$EMBED_SPEC/$LABEL_SET/04_predict_volume.yaml

python proc/seis_ssl_cluster/visualize_f3_lithology_predictions.py \
  --config $EXP/50_lithology/$MODEL_TAG/$EMBED_SPEC/$LABEL_SET/05_visualize_predictions.yaml

python proc/seis_ssl_cluster/build_f3_lithology_report.py \
  --config $EXP/50_lithology/$MODEL_TAG/$EMBED_SPEC/$LABEL_SET/06_build_lithology_report.yaml
```

`03_train_mlp_probe.yaml` is reserved for a lightweight MLP comparison after
the linear balanced MVP is working. Its probe spec is separate from
`linear_balanced_v1`.

## Baseline Comparison Regeneration

After changing pretrained `feature_source` metadata, rebuild the pretrained
token dataset and re-run the linear probe before regenerating the comparison.
The probe `metrics.json` must carry the same `feature_source` as
`token_dataset_metadata.json`.

```bash
python proc/seis_ssl_cluster/build_f3_lithology_token_dataset.py \
  --config experiments/f3/facies_benchmark_v1/50_lithology/<MODEL_TAG>/<EMBED_SPEC>/<LABEL_SET>/01_build_token_dataset.yaml

python proc/seis_ssl_cluster/train_f3_lithology_probe.py \
  --config experiments/f3/facies_benchmark_v1/50_lithology/<MODEL_TAG>/<EMBED_SPEC>/<LABEL_SET>/02_train_linear_probe.yaml
```

For the current F3 baseline comparison, reuse z-only, amplitude-only, or random
encoder outputs only when they already match the current train/validation token
split and label selection. Always include the xyz-coordinate baseline:

```bash
python proc/seis_ssl_cluster/build_f3_lithology_baseline_features.py \
  --config experiments/f3/facies_benchmark_v1/50_lithology_baselines/xyz_coordinates_v1/01_build_baseline_token_dataset.yaml

python proc/seis_ssl_cluster/train_f3_lithology_probe.py \
  --config experiments/f3/facies_benchmark_v1/50_lithology_baselines/xyz_coordinates_v1/02_train_linear_probe.yaml
```

Then regenerate the comparison report. The comparison config publishes selected
lightweight outputs to `results/f3/facies_benchmark_v1/baseline_comparison/`.

```bash
python proc/seis_ssl_cluster/build_f3_lithology_comparison_report.py \
  --config experiments/f3/facies_benchmark_v1/50_lithology_baselines/05_build_baseline_comparison_report.yaml

python proc/seis_ssl_cluster/validate_results_artifacts.py \
  --root results \
  --max-file-size-mb 10
```

## Figure Contract

- Use white backgrounds and fixed facies colors from the F3 inspection palette.
- Include clear legends in final figures.
- Display depth/sample axes with z increasing downward.
- Record figure inputs, slices, palette, and output paths in metadata JSON.
