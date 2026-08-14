# Seismic SSL Cluster Configuration

This project uses Option 1: explicit paths in every stage YAML. Each downstream
YAML names the upstream artifact path it consumes. Output paths are not derived
from dataset, version, or run IDs.

Raw user YAML is intentionally minimal. Resolvers add fixed contracts, defaults,
stage identity, and CLI overrides to the runtime config. Training writes that
complete config to `resolved_config.json`, and checkpoints store the same
resolved config under `config`.

Repository storage has three roles: ignored `artifacts/` holds complete run
outputs, intermediate products, and downstream inputs; tracked `reports/`
holds lightweight human-readable summaries and is never a pipeline input; and
`experiments/` holds experiment definitions and configuration.

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

## Path Configuration

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

Generated outputs must be non-empty absolute paths. Registry stages that also
have `paths.nopims_root` reject generated outputs under the raw NOPIMS root to
protect source data. Output paths are otherwise used exactly as configured;
they do not have to follow a repository-defined hierarchy or sit below
`paths.artifact_root`.

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

Recommended artifact roles under a local output root are shown below. This is
an organizational example and is not enforced by a validator:

| Directory | Contents |
|---|---|
| `pretraining/` | MAE checkpoints, resolved config, and training debug outputs |
| `embeddings/` | Extracted encoder embeddings |
| `clustering/` | KMeans models, clustering labels, and clustering metadata |
| `visualizations/` | PNGs, visualization reports, summaries, and optional voxel labels |

An example pretraining checkpoint path is
`pretraining/nopims/pretrain_v1/<MODEL_TAG>/full_100ep`; active configs may use
any explicit path that satisfies runtime input and overwrite protections.

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
  output_root: /workspace/artifacts/seis_ssl_cluster/pretraining/nopims/pretrain_v1/amp_mae_v1/full_100ep  # change per run
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
  reconstruction: huber
  huber_delta: 1.0
  gradient_weight: 0.05
  target_normalization:
    mode: none
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

`loss.reconstruction` must be `huber`, `mse`, or `l1`. Huber requires
`loss.huber_delta`; MSE and L1 must omit `loss.huber_delta`.

For a new CUDA experiment, override the training runtime without rewriting
existing experiment configs:

```yaml
train:
  num_workers: 8
  prefetch_factor: 2
  persistent_workers: true
  amp: true
  amp_dtype: auto
  stage_timing: false
```

`amp_dtype` accepts `auto`, `bfloat16`, or `float16`. On CUDA, `auto` selects
BF16 when the device supports it and otherwise selects FP16; FP16 uses a
gradient scaler. CPU training remains FP32. `prefetch_factor` is applied only
when `num_workers` is positive, and persistent workers are disabled when
`num_workers` is zero. CUDA batches use pinned memory and request non-blocking
H2D transfer only for tensors that are actually pinned. These resolved runtime
choices are recorded in `run_metadata.json`. Set `stage_timing: true` to write
`stage_timings.json` with data-wait, H2D, forward/loss, backward, and optimizer
stage summaries.

When `visualization.mae_debug.enabled` is true, at least one of
`every_steps` or `every_epochs` must be set to a positive integer. An explicit
`output_dir` must be an absolute path under `paths.output_root`; `null` writes
to `paths.output_root/visualizations/mae_debug`.

### Strat-HMM Pretext Training

The strat-HMM pretext resolver has two explicit, mutually exclusive target
schemas. Existing single-head configurations continue to use
`pseudo_targets.input_dir`, `pseudo_targets.k`, and `head.num_prototypes`; no
multi-head defaults are inserted into their resolved configuration.

Multi-head ordered-prototype pretext uses the following schema:

```yaml
pseudo_targets:
  manifest: /absolute/path/multi_head_target_manifest.json
  min_confidence: 0.0
head:
  spec: multi_resolution_ordered_prototypes_v1
  ks: [6, 8, 10]
  projection_dim: 128
  temperature: 0.1
  normalize: true
loss:
  prototype_weight: 1.0
  usage_weight: 0.005
  entropy_floor: null
  consistency_weight: 0.1
  consistency_beta: 0.1
  distillation_weight: 0.2
identity:
  model_tag: strat_hmm_multi_k6810_main_v1
  scientific_identity:
    experiment_role: multi_head_ordered_pretext
    head_spec: multi_resolution_ordered_prototypes_v1
    head_ks: [6, 8, 10]
    target_manifest_sha256: <sha256 of the manifest file>
    consistency_policy: normalized_order_smooth_l1_v1
  runtime_identity:
    device: cuda
    workers: 4
```

`head.ks` is an increasing sequence of at least two integer cardinalities, each
at least two. The manifest must be complete, hash-valid, use the supported
increasing-downward ordering, have matching K values and common valid-token
identity, and contain no boundary-weight references. The resolver validates
manifest metadata and file hashes; it does not load full pseudo-target arrays.

The manifest hash is required in scientific identity and must match the actual
manifest file. Resolved multi-head identity also records the manifest's
per-head target hashes. It also records and binds `head_projection_dim`,
`head_temperature`, `head_normalize`, all pretext loss weights and beta,
teacher/student initialization, and the effective model/data, zero-mask
preprocessing, and scientific training settings. Supplying any of those
resolved fields with a different value fails validation. Runtime identity is
limited to execution details such as device, workers, timing, cache location,
and resume path.

`prototype_weight` weights the mean head prototype loss, and `usage_weight`
weights the mean head usage loss: neither is added once per head. Likewise,
`consistency_weight` weights the mean across the three head pairs for the
K=6/8/10 experiment. A distillation-only configuration is allowed as an
explicit guardrail, but all loss weights may not be zero.

K is the pretext ordered-state cardinality, not the downstream lithology class
count. The F3 downstream `class_count` remains 6.

### Embedding Extraction

Training-owned sections are loaded from the checkpoint, not repeated here.

```yaml
paths:
  artifact_root: /workspace/artifacts/seis_ssl_cluster
manifests:
  input: /workspace/artifacts/seis_ssl_cluster/registry/manifests/nopims/pretrain_v1_clean/nopims_amplitude_manifests.json
embeddings:
  checkpoint: /workspace/artifacts/seis_ssl_cluster/pretraining/nopims/pretrain_v1/amp_mae_v1/full_100ep/mae_latest.pt
  output_dir: /workspace/artifacts/seis_ssl_cluster/embeddings/nopims/pretrain_v1
embedding:
  window_size: [128, 128, 128]
  overlap: [64, 64, 64]
  output_dtype: float16
  average_chunk_size_x: 16
  batch_size: 1
  prefetch_queue_depth: 0
  amp: false
  amp_dtype: auto
  stage_timing: false
  min_token_valid_fraction: 0.5
  preprocessing_cache:
    mode: 'off'
    chunk_size_x: 16
    reuse: true
    cleanup: false
```

Valid windows are encoded in batches of `batch_size`; an incomplete final batch
is retained. `prefetch_queue_depth: 0` uses synchronous read/preprocessing,
while a positive value bounds the producer queue by that many prepared batches.
CUDA extraction pins prepared batches and requests non-blocking H2D transfers.

Extraction remains FP32 by default. With `amp: true` on CUDA, `amp_dtype` accepts
`auto`, `bfloat16`, or `float16`; `auto` selects BF16 when supported by the
selected device and otherwise FP16. The resolved precision is saved in each
survey's metadata. Set `stage_timing: true` to write `stage_timings.json` with
read/preprocessing, queue wait, H2D, encode, D2H, cache preparation, and
merge/write summaries.

`average_chunk_size_x` bounds overlap averaging along the token-grid x axis.
The preprocessing cache modes are `off`, `memory`, and `memmap`. `memory` keeps
survey-scoped normalized amplitude and zero-mask arrays in RAM; `memmap` builds
them in `chunk_size_x` source-volume slabs under the output cache directory (or
an explicit `directory`). `reuse` permits completed fingerprint-matched memmaps
to be reopened, and `cleanup` removes the selected cache after extraction.
Interrupted memmap builds are never reused. Settings that cannot safely share
window-invariant preprocessing fall back to the uncached path and record the
reason in metadata.

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
opt-in and should name selected surveys. Set `amplitude_comparison.enabled: true`
to write side-by-side amplitude, cluster, and overlay panels for the same slices.

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
  amplitude_comparison:
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
| `loss.valid_mask_mode` | Fixed internally |
| `model` or `train` sections in non-training YAMLs | Removed |
| `data`, `masking`, `loss`, `train`, or `zero_mask` in extraction YAML | Loaded from checkpoint resolved config |

Stale redundant sections now fail validation instead of being silently ignored.

`loss.target_normalization.mode` is required. `none` preserves the existing MAE target, and `patch_zscore` normalizes only the patchified target used by reconstruction loss. Inputs and dataset targets stay in survey-wise normalized amplitude space. For `patch_zscore`, set positive finite `eps` and `min_std`; mean and population variance are computed from `local_valid_mask == true` voxels only, using `std_eff = max(sqrt(var + eps), min_std)`. `patch_zscore` is rejected when `loss.gradient_weight != 0.0` because the current gradient loss compares survey-normalized amplitude gradients.
