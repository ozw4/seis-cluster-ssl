# F3 token-level lithology probe

This runbook defines the experiment hierarchy, artifact layout, and shared
config contract for the first F3 few-label token-level lithology probe.

## Scope

The MVP freezes a NOPIMS-pretrained MAE encoder, extracts F3 token embeddings,
builds a token dataset from sparse supervised 2D F3 slices, trains a lightweight
probe, predicts token lithology over the F3 volume, and writes publication-ready
figures and a report. Dense decoders, dense segmentation heads, and F3 encoder
fine-tuning are out of scope for this stage.

## Roots And Fixed Variables

```bash
ROOT=/workspace/artifacts/seis_ssl_cluster
EXP=experiments/f3/facies_benchmark_v1

MODEL_TAG=amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1
EMBED_SPEC=overlap_x16
LABEL_SET=png_slices_segy_labels_v1
PROBE_SPEC=linear_balanced_v1
```

- Raw data root: `/home/dcuser/data/public_data/field/F3`
- Artifact root: `/workspace/artifacts/seis_ssl_cluster`
- Config root:
  `experiments/f3/facies_benchmark_v1/50_lithology/$MODEL_TAG/$EMBED_SPEC/$LABEL_SET`
- Frozen checkpoint:
  `$ROOT/pretraining/nopims/pretrain_v1/$MODEL_TAG/full_100ep/mae_best.pt`

The checked-in configs intentionally fail if `mae_best.pt` is absent; selecting
`mae_latest.pt` must be an explicit config edit, not an implicit fallback.

Do not write this workflow under `runs/`.

## Label Contract

`f3_labels.sgy` and the converted label volume are the source of truth for
supervised lithology labels:

```text
/home/dcuser/data/public_data/field/F3/f3_labels.sgy
$ROOT/registry/volumes/f3/facies_benchmark_v1/f3_facies_labels.npy
```

PNG labels are used only to select train/validation slice locations and to
visually confirm the selected labels.

If train and validation inventory slices intersect at the same `token_xyz`,
validation keeps precedence and matching train rows are removed before writing
`train_tokens.npz`; the build metadata records the removed row count.

## Artifact Layout

```text
$ROOT/registry/volumes/f3/facies_benchmark_v1/
  f3_seismic.npy
  f3_facies_labels.npy
  f3_metadata.json

$ROOT/registry/manifests/f3/facies_benchmark_v1/
  f3_amplitude_manifest.json

$ROOT/registry/normalization_stats/f3/facies_benchmark_v1/
  f3_seismic.normalization_stats.json

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

- `pretraining/` stores the frozen NOPIMS MAE checkpoint and resolved training
  config. The F3 lithology MVP reads this checkpoint and keeps the encoder
  frozen.
- `embeddings/` stores extracted F3 token embeddings keyed by `MODEL_TAG` and
  `EMBED_SPEC`.
- `lithology/` stores downstream supervised data, probe artifacts, prediction
  volumes, figures, and reports keyed by `LABEL_SET` and `PROBE_SPEC`.

## Config Contract

Each YAML in the lithology hierarchy is standalone and uses this shared
top-level shape:

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

Stage-specific sections then define `embeddings`, `labels`, `token_dataset`,
`probe`, `predictions`, `visualizations`, or `reports` as needed. Raw YAML does
not include a top-level `stage`; the proc entrypoint owns the stage identity.

## Stages

| Order | Entrypoint | Config |
|---|---|---|
| 1 | `prepare_f3_facies_volume.py` | F3 volume registry config |
| 2 | `extract_embeddings.py` | F3 embedding extraction config |
| 3 | `build_f3_lithology_token_dataset.py` | `01_build_token_dataset.yaml` |
| 4 | `train_f3_lithology_probe.py` | `02_train_linear_probe.yaml` |
| 5 | `predict_f3_lithology_tokens.py` | `04_predict_volume.yaml` |
| 6 | `visualize_f3_lithology_predictions.py` | `05_visualize_predictions.yaml` |
| 7 | `build_f3_lithology_report.py` | `06_build_lithology_report.yaml` |

`03_train_mlp_probe.yaml` is reserved for a lightweight MLP comparison after
the linear balanced MVP is established.

## Runbook

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

## Figure Contract

- Figures use white backgrounds and fixed facies colors from the F3 inspection
  palette.
- Final figures include clear legends.
- XZ and YZ sections display sample or depth increasing downward.
- Figure metadata records source slices, palette path, prediction input path,
  output paths, DPI, and rendering settings.
