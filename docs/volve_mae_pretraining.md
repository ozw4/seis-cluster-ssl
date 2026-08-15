# Volve Survey-Specific Amplitude MAE Pretraining

This is the current contract for survey-specific 3D amplitude MAE pretraining
on the complete Volve canonical amplitude volume. The input is amplitude only.

## Data identity and access

| Field | Contract |
| --- | --- |
| Public root | `${SEIS_SSL_CLUSTER_VOLVE_ROOT}` |
| Default public root | `/home/dcuser/public_data/field/volve` |
| Access | Read-only |
| Canonical dataset ID | `volve_st10010_full_t_v1` |
| Survey ID | `volve_st10010` |
| Canonical amplitude SHA-256 | `8e6a66c671658b2b24b9a961652972802e2735eaad3f7166642e52064bf46567` |
| Source SEG-Y SHA-256 | `f902e2bdaa277caf93a32e5f35eae653eb8b923138db0efc1e91918ef6757b2e` |
| Logical grid | `[inline, crossline, twt]`, registered as `[x, y, z]` |
| Shape | `[401, 720, 850]` |
| Dtype | `float32` |
| Explicit source-valid mask | `valid_trace_mask.npy`, shape `[401, 720]`, dtype `bool` |

Canonical registration produces only small artifacts under:

```text
${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/data/volve/horizon_benchmark_v1/
  volve_amplitude_manifest.json
  volve_npy_paths.txt
  volve.normalization_stats.json
  volve_canonical_input_metadata.json
```

The manifest points back to the public canonical amplitude and explicit valid
trace mask. Training replaces invalid trace samples only in its in-memory crop
processing; it does not copy, overwrite, interpolate, or regenerate public
data.

## Amplitude-only boundary

Horizon bindings, horizon interpretations, fault sticks, layout definitions,
and validation or test labels are not inputs to the MAE config, dataset,
training batch, checkpoint, or run snapshot. Initialization is random from seed
42. No F3, NOPIMS, Parihaka, or other pretrained checkpoint initializes this
run.

## Model and training contract

The fixed model tag is:

```text
amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1
```

It uses `128 x 128 x 128` crops, `8 x 8 x 8` patches, spatial mask ratio
`0.75`, patch-z-score targets, MSE reconstruction, zero gradient loss weight,
visible reconstruction weight `0.1`, normalized clip magnitude `8.0`, and
trace-RMS AGC with a 65-sample Z window. The encoder has dimension 384, depth
8, and 6 heads; the decoder has dimension 256, depth 4, and 4 heads.

The full run uses batch size 4, 10,000 samples per epoch, 100 epochs, AdamW
with learning rate `1e-4` and weight decay `0.05`, CUDA AMP, and seed 42. Its
output is:

```text
${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/volve/horizon_benchmark_v1/
  amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full_100ep/
```

`latest.pt` after epoch 100 and global step 250,000 is the downstream
pretrained checkpoint. `best.pt` follows training loss and is diagnostic only;
it is not selected using downstream labels. The CPU smoke uses two optimizer
steps in a separate `smoke_2step/` directory.

The paired random encoder checkpoint uses the same architecture and seed 42,
contains no pretrained weights, and must have a different SHA-256 from the
completed pretrained checkpoint.

## Scientific claim boundary

The complete Volve amplitude survey may include regions used by later
within-survey evaluation. The supported claim is therefore **same-survey
transductive self-supervised pretraining**. Pretraining alone does not establish
an inductive holdout result, cross-survey transfer, absence of downstream label
leakage, or improved horizon accuracy.

Commands and phase ordering are maintained in the experiment
`10_pretrain/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/README.md`.
