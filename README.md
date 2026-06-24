# seis-cluster-ssl

Amplitude-only 3D seismic self-supervised learning and unsupervised clustering.

`seis-cluster-ssl` learns reusable representations from unlabeled seismic amplitude volumes with a 3D masked autoencoder (MAE), extracts dense encoder embeddings, and clusters those embeddings into candidate seismic-facies domains.

The Python package name is `seis_ssl_cluster`.

## MVP scope

The current MVP is intentionally narrow:

```text
unlabeled amplitude .npy volumes
  -> survey-wise robust normalization
  -> amplitude-only 3D MAE pretraining
  -> full-volume encoder embedding extraction
  -> MiniBatchKMeans clustering
  -> token- and optional voxel-scale cluster visualization
```

The MVP does **not** use fixed seismic attributes, well labels, facies labels, a context branch, or supervised fine-tuning. Arbitrary attributes may be added later as optional clustering features without changing the amplitude-only pretraining model.

## Main features

- Explicit path-list control over training volumes
- Memory-mapped 3D NumPy input in `[x, y, z]` order
- Survey-wise percentile clipping and median/IQR normalization
- Normalization QC and clean-manifest generation
- Raw-amplitude zero-sample and zero-trace masking
- Strict 3D MAE using visible tokens only
- Voxel-validity-aware reconstruction and gradient losses
- Atomic checkpoints, deterministic resume, and non-finite diagnostics
- Optional training-time XY/XZ reconstruction visualization
- Sliding-window full-volume embedding extraction
- Deterministic global sampling, optional PCA, and MiniBatchKMeans
- Safe token-map visualization and opt-in voxel-label reconstruction

## Repository layout

```text
seis-cluster-ssl/
├── src/
│   └── seis_ssl_cluster/
│       ├── clustering/
│       ├── config/
│       ├── data/
│       ├── embedding/
│       ├── losses/
│       ├── masking/
│       ├── models/
│       ├── training/
│       ├── utils/
│       └── visualization/
├── proc/
│   ├── seis_ssl_cluster/
│   └── configs/
│       └── seis_ssl_cluster/
├── tests/
│   └── seis_ssl_cluster/
├── docs/
├── tools/
├── pyproject.toml
└── README.md
```

`proc/` contains thin command-line entrypoints. Reusable implementation code belongs under `src/seis_ssl_cluster/`.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/ozw4/seis-cluster-ssl.git
cd seis-cluster-ssl

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev,cluster,visualization]"
```

The clustering extra installs `scikit-learn` and `joblib`. The visualization extra installs `matplotlib`.

## Input data contract

Each source volume must be:

- A numeric, non-object `.npy` file
- Three-dimensional
- Stored in grid order `[x, y, z]`
- Readable with `numpy.load(path, mmap_mode="r")`

Example:

```python
import numpy as np

volume = np.load("/path/to/amplitude.npy", mmap_mode="r")
print(volume.shape)  # (X, Y, Z)
```

The repository does not redistribute NOPIMS or other seismic data. Users are responsible for data access and licensing.

## Artifact layout

Generated metadata and run products should be stored outside the raw-data directory.

Recommended layout:

```text
/workspace/artifacts/seis_ssl_cluster/
├── registry/
│   ├── splits/
│   │   └── nopims/pretrain_v1/
│   ├── manifests/
│   │   └── nopims/pretrain_v1/
│   ├── normalization_stats/
│   │   └── nopims/pretrain_v1/
│   └── qc/
│       └── nopims/pretrain_v1/
├── runs/
├── embeddings/
├── clustering/
└── visualizations/
```

The configuration validator requires generated manifest and normalization-stat paths to be absolute, under `paths.artifact_root`, and outside `paths.nopims_root`.

## Quick start

Before running a stage, edit the corresponding YAML under:

```text
proc/configs/seis_ssl_cluster/
```

At minimum, set the raw-data root where applicable, artifact root, input path-list, and stage-specific input/output paths. The full configuration guide is in [docs/configuration.md](docs/configuration.md), and the stage runbook is in [docs/seis_ssl_cluster_runbook.md](docs/seis_ssl_cluster_runbook.md).

This pipeline deliberately uses **Option 1: explicit paths in every stage YAML**. Each downstream YAML must name the upstream artifact path it consumes. Output paths are not derived automatically from dataset, version, or run IDs.

Raw user YAML is minimal. The entrypoint selects the stage, the resolver adds fixed/defaulted runtime fields and CLI overrides, and training snapshots the complete effective settings in `resolved_config.json` and checkpoints.

Configuration ownership:

| Parameter group | Source of truth |
|---|---|
| Selected amplitude volumes | explicit `train_npy_paths.txt` |
| Manifest/stat output paths | manifest-build YAML |
| Normalization sampling and clipping | normalization YAML |
| QC thresholds and clean outputs | QC YAML |
| Crop/model/mask/loss/optimizer | MAE training YAML |
| Model and zero-mask contract during extraction | checkpoint resolved config |
| Extraction window/overlap/output | extraction YAML |
| PCA/KMeans settings | clustering YAML |
| Survey/slice/voxel rendering controls | visualization YAML |
| Fixed amplitude-only contract | internal code constants |
| Complete effective run settings | `resolved_config.json` and checkpoint |

### 1. Create the training path-list

Create a plain-text file containing one `.npy` path per line.

```text
# Absolute paths are accepted.
/home/dcuser/data/NOPIMS/ENO0047389_0_00050_00400.npy
/home/dcuser/data/NOPIMS/ENO0047389_0_00050_00750.npy

# Relative paths are resolved against paths.nopims_root.
ENO0047392_0_00100_00100.npy
```

Empty lines and full-line comments are ignored. Inline comments, missing files, non-`.npy` paths, and duplicate files are rejected.

A recommended location is:

```text
/workspace/artifacts/seis_ssl_cluster/registry/splits/nopims/pretrain_v1/train_npy_paths.txt
```

### 2. Build the amplitude manifest

Dry-run:

```bash
python proc/seis_ssl_cluster/build_nopims_manifests.py \
  --config proc/configs/seis_ssl_cluster/build_nopims_manifests.yaml \
  --dry-run
```

Build:

```bash
python proc/seis_ssl_cluster/build_nopims_manifests.py \
  --config proc/configs/seis_ssl_cluster/build_nopims_manifests.yaml
```

The manifest records, for every listed volume:

- Stable survey ID
- Source amplitude path
- Shape and dtype
- `[x, y, z]` grid order
- Planned normalization-stat JSON path

This stage does not calculate normalization statistics.

### 3. Prepare survey normalization statistics

```bash
python proc/seis_ssl_cluster/prepare_nopims_normalization_stats.py \
  --config proc/configs/seis_ssl_cluster/prepare_nopims_normalization_stats.yaml
```

The default normalization contract is:

```text
clip to the 0.5 and 99.5 percentiles
center by the survey median
scale by survey IQR + epsilon
```

Statistics are sampled deterministically from each memory-mapped volume and written as one JSON file per survey.

### 4. Run normalization QC

```bash
python proc/seis_ssl_cluster/filter_manifest_by_normalization_qc.py \
  --config proc/configs/seis_ssl_cluster/filter_manifest_by_normalization_qc.yaml
```

The QC stage produces:

- A machine-readable QC report
- An excluded-survey list
- A clean path-list
- A clean manifest for pretraining

The source path-list and original manifest are not modified.

Typical rejection criteria include non-finite statistics, a very small IQR, and an excessive expected normalized range.

### 5. Run a short MAE smoke test

```bash
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config proc/configs/seis_ssl_cluster/train_amp_mae.yaml \
  --device cuda \
  --max-steps 2 \
  --output-root /workspace/artifacts/seis_ssl_cluster/runs/smoke_amp_mae
```

The training config should reference the **clean manifest** produced by the QC stage.

### 6. Run a pilot

```bash
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config proc/configs/seis_ssl_cluster/train_amp_mae.yaml \
  --device cuda \
  --max-steps 1000 \
  --output-root /workspace/artifacts/seis_ssl_cluster/runs/pilot_amp_mae_1000
```

### 7. Run full pretraining

```bash
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config proc/configs/seis_ssl_cluster/train_amp_mae.yaml \
  --device cuda \
  --output-root /workspace/artifacts/seis_ssl_cluster/runs/amp_mae_pretrain_v1
```

Resume an interrupted run with the same resolved configuration and output directory:

```bash
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config proc/configs/seis_ssl_cluster/train_amp_mae.yaml \
  --device cuda \
  --output-root /workspace/artifacts/seis_ssl_cluster/runs/amp_mae_pretrain_v1 \
  --resume /workspace/artifacts/seis_ssl_cluster/runs/amp_mae_pretrain_v1/mae_latest.pt
```

### 8. Extract full-volume embeddings

Set the trained checkpoint, clean manifest, `embeddings.output_dir`, window size, and overlap in `extract_embeddings.yaml`.

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config proc/configs/seis_ssl_cluster/extract_embeddings.yaml \
  --device cuda
```

Use `--skip-existing` to retain complete survey embeddings whose metadata matches the requested extraction run.

Embedding metadata binds each artifact to the checkpoint identity, model geometry, extraction geometry, normalization source, and zero-mask preprocessing contract.

### 9. Cluster embeddings

Configure `embeddings.input_dir`, `clustering.output_dir`, requested cluster counts, deterministic sampling, normalization, and optional PCA.

```bash
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config proc/configs/seis_ssl_cluster/cluster_embeddings.yaml
```

The MVP uses MiniBatchKMeans. Before fitting, it verifies that all embedding artifacts are mutually compatible.

Typical outputs include:

```text
preprocessor.joblib
kmeans.joblib
cluster_centers.npy
survey token-label arrays
survey label metadata
run metadata and cluster counts
```

Invalid tokens retain label `-1`.

### 10. Visualize cluster maps

```bash
python proc/seis_ssl_cluster/visualize_clusters.py \
  --config proc/configs/seis_ssl_cluster/visualize_clusters.yaml
```

The safe default renders token-level XY/XZ PNGs and summaries only.

Voxel reconstruction is opt-in because it can generate large artifacts. Select the intended surveys explicitly:

```yaml
visualization:
  survey_ids:
    - selected_survey_id
  modes: [token, voxel]
  reconstruct_voxel: true
  allow_all_surveys_for_voxel_reconstruction: false
  skip_existing_voxel_labels: true
  max_voxel_output_gib: 50.0
  allow_large_voxel_output: false
```

Slice coordinates are specified in source-volume voxel coordinates. Token views convert them internally using the patch size.

## Default MAE geometry

The raw training YAML contains only user-owned geometry and training controls:

```yaml
data:
  local_crop_size: [128, 128, 128]

model:
  patch_size: [8, 8, 8]
  encoder_dim: 384
  encoder_depth: 8
  encoder_heads: 6
  decoder_dim: 256
  decoder_depth: 4
  decoder_heads: 4

masking:
  spatial_mask_ratio: 0.75
  block_size_tokens: [2, 2, 2]

loss:
  reconstruction: huber
  huber_delta: 1.0
  gradient_weight: 0.05
  target_normalization:
    mode: none
```

The fixed amplitude-only contract is injected internally and appears in the resolved config and checkpoint, not in raw YAML:

```text
data.grid_order = [x, y, z]
data.volume_format = npy_memmap
data.input_channels = 1
data.target_channels = 1
data.use_context = false
model.name = amp_mae3d
model.in_channels = 1
model.out_channels = 1
masking.spatial_mask_mode = block
loss.valid_mask_mode = voxel
```

A `128 x 128 x 128` crop with `8 x 8 x 8` patches produces a `16 x 16 x 16` token grid containing 4096 tokens.

The encoder receives only visible tokens. Masked tokens are restored with a learned mask token before decoding.

## Loss

The training objective is:

```text
masked amplitude reconstruction loss
+ gradient_weight * valid masked gradient loss
```

Set `loss.reconstruction` in the training YAML to `huber`, `mse`, or `l1`. Huber requires `loss.huber_delta`; MSE and L1 must omit it. Both reconstruction and gradient terms exclude invalid voxels using `local_valid_mask`.

`loss.target_normalization.mode` is required. Use `mode: none` for the historical behavior. Use `mode: patch_zscore` with positive finite `eps` and `min_std` to z-score only the patchified loss target. The encoder input `x` and dataset `target` remain survey-wise normalized amplitudes; only `target_for_loss` is transformed as `(target_patch - valid_voxel_mean) / max(sqrt(valid_voxel_population_var + eps), min_std)`. Patch statistics use `local_valid_mask == true`; invalid voxels are excluded and zeroed in the normalized target. In v1, `patch_zscore` requires `loss.gradient_weight: 0.0` because the existing gradient loss operates in survey-normalized amplitude space. Debug MAE predictions are oracle-denormalized with target patch statistics before comparison and the JSON metadata records that oracle target statistics were used.

Invalid regions include source padding and configured raw-amplitude zero-sample or zero-trace influence regions.

## Starting A100 training settings

The current starting configuration is:

```yaml
train:
  batch_size: 4
  samples_per_epoch: 10000
  epochs: 100
  num_workers: 8
  shuffle: true
  lr: 3.0e-5
  weight_decay: 0.05
  amp: false
  device: cuda
  seed: 42
  grad_clip_norm: 1.0
```

These are starting values, not universal defaults. Validate stability with a smoke run and a pilot before launching a long run.

## Reproducibility and diagnostics

A training run snapshots its resolved configuration and input metadata. Checkpoints include model and optimizer state, progress state, random-number-generator state, and DataLoader generator state.

Non-finite losses or gradients stop training and write a diagnostic JSON containing the affected survey IDs, crop coordinates, tensor statistics, loss components, valid-voxel counts, and gradient information.

Optional MAE debug visualization can save XY/XZ panels containing:

```text
input
masked input
target
prediction
absolute error
valid mask
```

## Development

```bash
python -m compileall -q src proc tests
python -m ruff check .
pytest -q
```

Verify that the standalone package does not depend on the legacy package namespace:

```bash
python tools/check_seis_ssl_cluster_isolation.py
```

## Current limitations

- Pretraining uses amplitude only.
- The MVP clustering backend is MiniBatchKMeans.
- Cluster IDs are categorical and have no intrinsic geological meaning.
- Voxel maps are nearest-neighbor reconstructions of token labels.
- No supervised facies classification is included in the MVP.
- Arbitrary seismic attributes are not yet fused into clustering features.
- Multi-GPU distributed pretraining is not part of the initial MVP.

## Roadmap

Planned extensions include:

- Optional arbitrary-attribute features at clustering time
- Embedding-only vs. attribute-only vs. combined-feature ablations
- Alternative clustering backends and spatial regularization
- Supervoxel or graph-based clustering
- Cluster-to-facies evaluation on labeled external datasets
- Few-label facies adaptation

## License

No repository license file is currently included. Ensure all seismic datasets are used under their respective terms.
