# Parihaka Survey-Specific 3D Amplitude MAE Pretraining

This document is the source of truth for survey-specific 3D amplitude MAE
pretraining on the Parihaka training amplitude volume. The pretraining input is
amplitude only.

## Data provenance

| Field | Contract |
| --- | --- |
| Direct distributor | [Mendeley Data](https://doi.org/10.17632/gnvyh3msrj.1) |
| Dataset title | *Parihaka + Netherlands F3 (raw volumes + labels) for seismic facies segmentation* |
| Contributor | jiang zishuo |
| Version | 1 |
| DOI | `10.17632/gnvyh3msrj.1` |
| Upstream | AIcrowd Seismic Facies Identification Challenge |
| Displayed license | CC BY 4.0 |
| Downloaded archive | `parihaka_Data.zip` |
| Upstream member used | `data_train.npz` |
| Local filename | `parihaka_data_train.npz` |
| Local modification | Filename change only |

The local file is from the AIcrowd-derived redistribution obtained through
Mendeley Data. It is not described as a file downloaded directly from AIcrowd.
Byte identity with the AIcrowd distribution and whether Mendeley transformed
the redistributed data are unverified. Acquisition date and acquirer are not
part of this contract because no verified values are available.

## Source amplitude contract

`parihaka_data_train.npz` has this fixed amplitude payload:

| Field | Value |
| --- | --- |
| NPZ key | `data` |
| ZIP member | `data.npy` |
| Logical axes | `[Z, X, Y]` |
| Shape | `[1006, 782, 590]` |
| Dtype | `float32` |
| NPY `fortran_order` | `true` |
| Element count | `464148280` |
| Finite count | `464148280` |
| Nonfinite count | `0` |
| Minimum | `-5195.5234375` |
| Maximum | `5151.71875` |
| Mean | `0.6766075433795379` |
| Population standard deviation | `390.30892519280377` |

## MAE input volume contract

The volume read directly by MAE has this identity:

| Field | Value |
| --- | --- |
| Dataset | `parihaka` |
| Version | `facies_benchmark_v1` |
| `survey_id` | `parihaka` |
| Logical axes | `[X, Y, Z]` |
| Transform | `source.transpose(1, 2, 0)` |
| Shape | `[782, 590, 1006]` |
| Dtype | `float32` |
| Storage | C-contiguous `.npy` |
| Access | `numpy.load(..., mmap_mode='r')` |

For every valid coordinate, the preparation output must satisfy
`output[x, y, z] == source[z, x, y]`. Preparation must not retain the complete
source NPZ array and complete transposed volume in RAM at the same time. Only an
output for which the full coordinate mapping, shape, dtype, finite values, and
statistics have been verified may be used.

Source and output SHA-256 values may be recorded as metadata for identity
checks. They are not content-addressed-storage, deduplication, or artifact
repository identifiers.

## Direct data paths

The data contract uses this direct directory and no additional path layer:

```text
${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/data/parihaka/facies_benchmark_v1/
  parihaka_amplitude.npy
  parihaka_amplitude_manifest.json
  parihaka_npy_paths.txt
  parihaka_amplitude.normalization_stats.json
  parihaka_prepare_metadata.json
```

`parihaka_amplitude_manifest.json` follows `SurveyManifest` and
`AmplitudeVolumeRecord` in `src/seis_ssl_cluster/data/schema.py`. It contains
exactly one survey, uses `survey_id: parihaka`, points to
`parihaka_amplitude.npy`, and fixes `grid_order` to `[x, y, z]`.
`parihaka_npy_paths.txt` contains exactly one line: the absolute path to
`parihaka_amplitude.npy`. The normalization file follows
`SurveyNormalizationStats` in `src/seis_ssl_cluster/data/normalization.py`.

## Amplitude-only boundary

`parihaka_labels_train.npz`, the NPZ key `labels`, class IDs, and label
distributions are not inputs to preparation, normalization, the MAE config,
training batches, checkpoints, run snapshots, or review results. The label file
must not be opened or hashed anywhere in this pretraining series. Label-based
channels and facies inference belong to separate downstream work.

## Model and run identity

The model tag is:

```text
amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1
```

With `MODEL_TAG` set to that exact value, the output roots are:

```text
smoke: ${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/parihaka/facies_benchmark_v1/<MODEL_TAG>/smoke_2step
full:  ${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/parihaka/facies_benchmark_v1/<MODEL_TAG>/full_100ep
```

Each output path is explicit in its config. There is no central path builder,
run-directory abstraction, or indirect path lookup in this contract.

## Full MAE training contract

The stable scientific reference is:

```text
experiments/nopims/pretrain_v1/10_pretrain/
  amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/03_full_100ep.yaml
```

The Parihaka full run fixes the following resolved values. In particular, its
explicit runtime precision contract is `amp: true` and `amp_dtype: auto`.

```yaml
data:
  local_crop_size: [128, 128, 128]
  min_valid_fraction: 0.1
  max_resample_attempts: 16
  normalized_clip_abs: 8.0
  finite_check_mode: strict
  amplitude_agc:
    enabled: true
    mode: trace_rms_z
    window_z: 65
    eps: 1.0e-3
    clip_abs: 5.0
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
  block_size_tokens: [1, 1, 1]
loss:
  reconstruction: mse
  gradient_weight: 0.0
  visible_reconstruction_weight: 0.1
  target_normalization:
    mode: patch_zscore
    eps: 1.0e-6
    min_std: 0.05
train:
  batch_size: 4
  samples_per_epoch: 10000
  epochs: 100
  num_workers: 8
  prefetch_factor: 2
  persistent_workers: true
  shuffle: true
  optimizer: AdamW
  lr: 1.0e-4
  weight_decay: 0.05
  amp: true
  amp_dtype: auto
  device: cuda
  seed: 42
  grad_clip_norm: 1.0
  checkpoint_every_steps: null
  runtime_check_mode: once
  stage_timing: false
```

Initialization is random initialization from seed 42. No NOPIMS, F3, or other
survey checkpoint initializes this model. Architecture, masking, loss, data
processing, optimizer, duration, runtime, and precision values above are fixed,
not sweep dimensions.

`latest.pt` is the checkpoint after completion of epoch 100. `best.pt` follows
the existing strictly-lower training-loss policy and is diagnostic; it is not a
model selected by downstream performance. Training uses the current generic MAE
checkpoint schema and resume contract without a Parihaka-specific schema.

## Scientific claim boundary

This run learns an unlabeled representation for the same survey from the full
Parihaka amplitude volume. Because pretraining can include amplitudes from later
within-survey evaluation regions, it is **survey-specific transductive
self-supervised pretraining**.

Completion of pretraining alone does not establish absence of label leakage,
transfer to unseen surveys, an inductive holdout result, or improved downstream
accuracy. Those claims require separate downstream designs and evidence.

## Out of scope

- Parihaka label conversion, target generation, decoder training, and downstream evaluation.
- Embedding extraction, clustering, HMM pseudo-targets, and structured pretraining.
- Sweeps over architecture, mask ratio, loss, AGC, crop, epochs, seed, or precision.
- A registry hierarchy, `ArtifactPaths`, a common publisher, a publish manifest, a PASS handoff, or a workflow state machine. None is introduced or required.
- Changes to existing NOPIMS or F3 data, checkpoints, or results.
