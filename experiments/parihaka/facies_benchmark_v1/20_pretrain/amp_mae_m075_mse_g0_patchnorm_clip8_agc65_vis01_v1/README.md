# Parihaka amplitude MAE execution

Run from the repository root with the environment roots set explicitly:

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/path/to/artifacts/seis_ssl_cluster
export PARIHAKA_DATA_ROOT=/path/containing/parihaka_data_train.npz
export EXP=experiments/parihaka/facies_benchmark_v1
export PRETRAIN="$EXP/20_pretrain/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1"

python proc/seis_ssl_cluster/validate_parihaka_mae.py \
  --prepare-config "$EXP/10_prepare/01_prepare_parihaka_volume.yaml" \
  --smoke-config "$PRETRAIN/01_smoke_2step.yaml" \
  --full-config "$PRETRAIN/02_full_100ep.yaml" \
  --check inputs

python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$PRETRAIN/01_smoke_2step.yaml" --dry-run

python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$PRETRAIN/01_smoke_2step.yaml"

python proc/seis_ssl_cluster/validate_parihaka_mae.py \
  --prepare-config "$EXP/10_prepare/01_prepare_parihaka_volume.yaml" \
  --smoke-config "$PRETRAIN/01_smoke_2step.yaml" \
  --full-config "$PRETRAIN/02_full_100ep.yaml" \
  --check smoke

python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$PRETRAIN/02_full_100ep.yaml" --dry-run

python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$PRETRAIN/02_full_100ep.yaml"

python proc/seis_ssl_cluster/validate_parihaka_mae.py \
  --prepare-config "$EXP/10_prepare/01_prepare_parihaka_volume.yaml" \
  --smoke-config "$PRETRAIN/01_smoke_2step.yaml" \
  --full-config "$PRETRAIN/02_full_100ep.yaml" \
  --check full

python proc/seis_ssl_cluster/summarize_parihaka_mae.py \
  --prepare-config "$EXP/10_prepare/01_prepare_parihaka_volume.yaml" \
  --full-config "$PRETRAIN/02_full_100ep.yaml" \
  --output-dir \
    results/parihaka/facies_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1
```

The smoke and full resolved configs may differ only at these fields:

- `paths.output_root`
- `train.batch_size`
- `train.samples_per_epoch`
- `train.epochs`
- `train.num_workers`
- `train.prefetch_factor`
- `train.persistent_workers`
- `train.amp`
- `train.device`
- `train.max_steps`
- `visualization.mae_debug.enabled`

The smoke run is CPU-only, uses random initialization from seed 42, and stops
after two optimizer steps. Run the full command only after both input and smoke
validation pass. A completed full run is schema 2 at epoch 100 and global step
250000; `latest.pt` is the primary completed checkpoint and `best.pt` remains a
strictly-lower training-loss diagnostic.
