# Seismic SSL Cluster Configuration

This project uses Option 1: explicit paths in every stage YAML. Each downstream
YAML names the upstream artifact path it consumes. Output paths are not derived
from dataset, version, or run IDs.

Raw user YAML is intentionally minimal. Resolvers add fixed contracts, defaults,
stage identity, and CLI overrides to the runtime config. Training writes that
complete config to `resolved_config.json`, and checkpoints store the same
resolved config under `config`.

## Configuration Ownership

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

## User YAML Shapes

Default YAML files must keep these top-level sections only:

| Stage | Default YAML | Top-level sections |
|---|---|---|
| Build manifest | `build_nopims_manifests.yaml` | `paths`, `manifest` |
| Normalization stats | `prepare_nopims_normalization_stats.yaml` | `paths`, `manifests`, `normalization` |
| Normalization QC | `filter_manifest_by_normalization_qc.yaml` | `paths`, `manifests`, `splits`, `qc` |
| MAE training | `train_amp_mae.yaml` | `paths`, `manifests`, `data`, `zero_mask`, `model`, `masking`, `loss`, `train`, `visualization` |
| Embedding extraction | `extract_embeddings.yaml` | `paths`, `manifests`, `embeddings`, `embedding` |
| Clustering | `cluster_embeddings.yaml` | `paths`, `embeddings`, `clustering` |
| Visualization | `visualize_clusters.yaml` | `paths`, `clustering`, `visualization` |

No user YAML contains a top-level `stage`; the proc entrypoint selects the stage.

## Path Contract

Every stage names its upstream inputs and downstream outputs explicitly. The
resolver validates those paths but does not derive paths from dataset names,
version strings, run IDs, or other config fields.

The `paths` mapping is stage-specific and accepts only these keys:

| Stage | Required `paths` keys |
|---|---|
| Build manifest | `nopims_root`, `artifact_root` |
| Normalization stats | `nopims_root`, `artifact_root` |
| Normalization QC | `nopims_root`, `artifact_root` |
| MAE training | `artifact_root`, `output_root` |
| Embedding extraction | `artifact_root` |
| Clustering | `artifact_root` |
| Visualization | `artifact_root` |

Generated outputs must be non-empty absolute paths under the resolved
`paths.artifact_root`. The normalized path is checked, so `..` traversal cannot
escape the artifact root. Registry stages that also have `paths.nopims_root`
reject generated outputs under the raw NOPIMS root.

Generated output fields are:

| Stage | Output fields |
|---|---|
| Build manifest | `manifest.output_dir`, `manifest.normalization_stats_dir` |
| Normalization QC | `manifests.output`, `splits.output`, `qc.output_json`, `qc.excluded_surveys` |
| MAE training | `paths.output_root` |
| Embedding extraction | `embeddings.output_dir` |
| Clustering | `clustering.output_dir` |
| Visualization | `visualization.output_dir` |

Input and handoff fields remain explicit user-visible paths. They are not
rewritten by the resolver and may point outside `artifact_root` when the stage
intentionally supports that, such as raw NOPIMS path lists or an existing
checkpoint path.

## Fixed And Checkpoint-Owned Settings

These fixed amplitude-only contract fields are not valid in raw YAML:

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
loss.reconstruction = huber
loss.valid_mask_mode = voxel
```

Embedding extraction does not repeat training sections. It loads model geometry,
masking/loss modes, and zero-mask preprocessing from the checkpoint resolved
config.

## Minimal YAML Examples

Values marked `change` are normally edited for a new dataset or run.

### Build Manifest

```yaml
paths:
  nopims_root: /home/dcuser/data/NOPIMS       # change
  artifact_root: /workspace/artifacts/seis_ssl_cluster
manifest:
  input_path_list: /workspace/artifacts/seis_ssl_cluster/registry/splits/nopims/pretrain_v1/train_npy_paths.txt  # change
  output_dir: /workspace/artifacts/seis_ssl_cluster/registry/manifests/nopims/pretrain_v1
  output_name: nopims_amplitude_manifests.json
  normalization_stats_dir: /workspace/artifacts/seis_ssl_cluster/registry/normalization_stats/nopims/pretrain_v1
```

### Normalization Stats

```yaml
paths:
  nopims_root: /home/dcuser/data/NOPIMS       # change
  artifact_root: /workspace/artifacts/seis_ssl_cluster
manifests:
  train: /workspace/artifacts/seis_ssl_cluster/registry/manifests/nopims/pretrain_v1/nopims_amplitude_manifests.json  # from build manifest
normalization:
  clipping_percentiles: [0.5, 99.5]
  epsilon: 1.0e-6
  max_samples: 1000000
  seed: 42
```

### Normalization QC

```yaml
paths:
  nopims_root: /home/dcuser/data/NOPIMS       # change
  artifact_root: /workspace/artifacts/seis_ssl_cluster
manifests:
  input: /workspace/artifacts/seis_ssl_cluster/registry/manifests/nopims/pretrain_v1/nopims_amplitude_manifests.json
  output: /workspace/artifacts/seis_ssl_cluster/registry/manifests/nopims/pretrain_v1_clean/nopims_amplitude_manifests.json
splits:
  input: /workspace/artifacts/seis_ssl_cluster/registry/splits/nopims/pretrain_v1/train_npy_paths.txt
  output: /workspace/artifacts/seis_ssl_cluster/registry/splits/nopims/pretrain_v1_clean/train_npy_paths.txt
qc:
  output_json: /workspace/artifacts/seis_ssl_cluster/registry/qc/nopims/pretrain_v1/normalization_stats_qc.json
  excluded_surveys: /workspace/artifacts/seis_ssl_cluster/registry/qc/nopims/pretrain_v1/excluded_surveys.txt
  min_iqr: 1.0e-4
  max_normalized_abs: 1.0e+6
```

### MAE Training

```yaml
paths:
  artifact_root: /workspace/artifacts/seis_ssl_cluster
  output_root: /workspace/artifacts/seis_ssl_cluster/runs/amp_mae_pretrain_v1  # change per run
manifests:
  train: /workspace/artifacts/seis_ssl_cluster/registry/manifests/nopims/pretrain_v1_clean/nopims_amplitude_manifests.json
  train_path_list: /workspace/artifacts/seis_ssl_cluster/registry/splits/nopims/pretrain_v1_clean/train_npy_paths.txt
data:
  local_crop_size: [128, 128, 128]
  min_valid_fraction: 0.1
  max_resample_attempts: 16
zero_mask:
  enabled: true
  zero_atol: 0.0
  z_sample_influence_radius: 16
  xy_trace_influence_radius: 1
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
  huber_delta: 1.0
  gradient_weight: 0.05
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
visualization:
  mae_debug:
    enabled: false
    output_dir: null
    every_steps: 1000
    every_epochs: null
    max_samples: 1
    xy_slice_index: null
    xz_slice_y_index: null
    dpi: 160
    clip_percentiles: [1.0, 99.0]
    columns: [input, masked_input, target, prediction, abs_error, valid_mask]
    panel_width: 2.6
    panel_height: 2.4
    invalid_color: lightgray
```

When `visualization.mae_debug.enabled` is true, at least one of
`every_steps` or `every_epochs` must be set to a positive integer. An explicit
`output_dir` must be an absolute path under `paths.output_root`; `null` writes
to `paths.output_root/visualizations/mae_debug`.

### Embedding Extraction

Training-owned sections are loaded from the checkpoint, not repeated here.

```yaml
paths:
  artifact_root: /workspace/artifacts/seis_ssl_cluster
manifests:
  input: /workspace/artifacts/seis_ssl_cluster/registry/manifests/nopims/pretrain_v1_clean/nopims_amplitude_manifests.json
embeddings:
  checkpoint: /workspace/artifacts/seis_ssl_cluster/runs/amp_mae_pretrain_v1/mae_latest.pt
  output_dir: /workspace/artifacts/seis_ssl_cluster/embeddings/nopims/pretrain_v1
embedding:
  window_size: [128, 128, 128]
  overlap: [64, 64, 64]
  output_dtype: float16
  batch_size: 1
  min_token_valid_fraction: 0.5
```

### Clustering

```yaml
paths:
  artifact_root: /workspace/artifacts/seis_ssl_cluster
embeddings:
  input_dir: /workspace/artifacts/seis_ssl_cluster/embeddings/nopims/pretrain_v1
clustering:
  output_dir: /workspace/artifacts/seis_ssl_cluster/clustering/nopims/pretrain_v1
  embedding_normalization: l2
  pca:
    enabled: true
    n_components: 64
    whiten: false
  sample_tokens: 1000000
  method: minibatch_kmeans
  k_values: [6, 8, 10, 12]
  minibatch_size: 8192
  seed: 42
```

### Visualization

The safe default renders token maps and summaries only. Voxel reconstruction is
opt-in and should name selected surveys.

```yaml
paths:
  artifact_root: /workspace/artifacts/seis_ssl_cluster
clustering:
  input_dir: /workspace/artifacts/seis_ssl_cluster/clustering/nopims/pretrain_v1
visualization:
  output_dir: /workspace/artifacts/seis_ssl_cluster/visualizations/clusters/nopims/pretrain_v1
  survey_ids: []
  modes: [token]
  reconstruct_voxel: false
  allow_all_surveys_for_voxel_reconstruction: false
  skip_existing_voxel_labels: true
  max_voxel_output_gib: 50.0
  allow_large_voxel_output: false
  slice_coordinate_space: voxel
  xy_slices: [750]
  xz_slices: [150]
  dpi: 160
  invalid_color: lightgray
  amplitude_underlay:
    enabled: false
    alpha: 0.35
  summaries:
    enabled: true
    include_amplitude_norm: false
```

## Migration From Older YAMLs

| Old key | New handling |
|---|---|
| `stage` | Removed; entrypoint selects stage |
| `data.grid_order` | Fixed internally |
| `data.volume_format` | Fixed internally |
| `data.input_channels` | Fixed internally |
| `data.target_channels` | Fixed internally |
| `data.use_context` | Fixed internally |
| `model.name` | Fixed internally |
| `model.in_channels` | Fixed internally |
| `model.out_channels` | Fixed internally |
| `masking.spatial_mask_mode` | Fixed internally |
| `loss.reconstruction` | Fixed internally |
| `loss.valid_mask_mode` | Fixed internally |
| `model` or `train` sections in non-training YAMLs | Removed |
| `data`, `masking`, `loss`, `train`, or `zero_mask` in extraction YAML | Loaded from checkpoint resolved config |

Stale redundant sections now fail validation instead of being silently ignored.
