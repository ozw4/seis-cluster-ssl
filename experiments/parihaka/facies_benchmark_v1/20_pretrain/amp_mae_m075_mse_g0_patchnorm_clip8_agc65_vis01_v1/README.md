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
after two optimizer steps. The full config is validated but is not executed by
this sequence.
