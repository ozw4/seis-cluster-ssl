# Volve amplitude MAE execution

Run from the repository root with the environment roots set explicitly:

```bash
export SEIS_SSL_CLUSTER_VOLVE_ROOT=/home/dcuser/public_data/field/volve
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/path/to/artifacts/seis_ssl_cluster
export EXP=experiments/volve/horizon_benchmark_v1
export PRETRAIN="$EXP/10_pretrain/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1"

python proc/seis_ssl_cluster/prepare_volve_canonical_inputs.py --dry-run
python proc/seis_ssl_cluster/prepare_volve_canonical_inputs.py

python proc/seis_ssl_cluster/validate_volve_mae.py \
  --input-config proc/configs/seis_ssl_cluster/prepare_volve_canonical_inputs.yaml \
  --smoke-config "$PRETRAIN/01_smoke_2step.yaml" \
  --full-config "$PRETRAIN/02_full_100ep.yaml" \
  --check inputs

python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$PRETRAIN/01_smoke_2step.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$PRETRAIN/01_smoke_2step.yaml"

python proc/seis_ssl_cluster/validate_volve_mae.py \
  --input-config proc/configs/seis_ssl_cluster/prepare_volve_canonical_inputs.yaml \
  --smoke-config "$PRETRAIN/01_smoke_2step.yaml" \
  --full-config "$PRETRAIN/02_full_100ep.yaml" \
  --check smoke

python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$PRETRAIN/02_full_100ep.yaml" --dry-run
```

Run the 100-epoch command only as a separately scheduled scientific run. After
it completes, validate `--check full`, then create the paired random baseline:

```bash
python proc/seis_ssl_cluster/create_random_mae_checkpoint.py \
  --config "$PRETRAIN/03_create_random_checkpoint.yaml" --dry-run
python proc/seis_ssl_cluster/create_random_mae_checkpoint.py \
  --config "$PRETRAIN/03_create_random_checkpoint.yaml"
```

The smoke and full resolved configs may differ only in output location, runtime
resources, duration, AMP/device, `train.max_steps`, and debug visualization
enablement. `latest.pt` after epoch 100 is the downstream reference. `best.pt`
is a training-loss diagnostic and is not used for downstream selection. Each
training run snapshots the registered normalization stats and canonical input
metadata under `inputs/`; validation binds those copies and their hashes to the
current registration identity.
