# Volve amplitude MAE execution

This runbook owns only the phase ordering for canonical Volve MAE pretraining.
Data-use and claim boundaries are documented in
[`docs/volve_mae_pretraining.md`](../../../../../docs/volve_mae_pretraining.md).
The YAML and resolvers define settings, paths, checkpoint contents, and
completion criteria; validators enforce those contracts.

Start with the shared
[Volve benchmark environment](../../README.md), then select this run:

```bash
export EXP="$VOLVE_EXP"
export PRETRAIN="$EXP/10_pretrain/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1"
export FULL_RUN="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/volve/horizon_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full_100ep"
```

Prepare the canonical registration and validate it before training:

```bash
python proc/seis_ssl_cluster/prepare_volve_canonical_inputs.py \
  --only-missing --dry-run
python proc/seis_ssl_cluster/prepare_volve_canonical_inputs.py --only-missing
python proc/seis_ssl_cluster/validate_volve_mae.py \
  --input-config proc/configs/seis_ssl_cluster/prepare_volve_canonical_inputs.yaml \
  --smoke-config "$PRETRAIN/01_smoke_2step.yaml" \
  --full-config "$PRETRAIN/02_full_100ep.yaml" \
  --check inputs
```

Run and validate the isolated smoke condition:

```bash
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$PRETRAIN/01_smoke_2step.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$PRETRAIN/01_smoke_2step.yaml"
python proc/seis_ssl_cluster/validate_volve_mae.py \
  --input-config proc/configs/seis_ssl_cluster/prepare_volve_canonical_inputs.yaml \
  --smoke-config "$PRETRAIN/01_smoke_2step.yaml" \
  --full-config "$PRETRAIN/02_full_100ep.yaml" \
  --check smoke
```

Then start the full run. Resume an interrupted run only from its own rolling
checkpoint.

```bash
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$PRETRAIN/02_full_100ep.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$PRETRAIN/02_full_100ep.yaml"
```

If that live command is interrupted, resume it instead of starting another
fresh run:

```bash
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$PRETRAIN/02_full_100ep.yaml" \
  --resume "$FULL_RUN/latest.pt"
```

After completion, validate the full run and create its matched Random encoder:

```bash
python proc/seis_ssl_cluster/validate_volve_mae.py \
  --input-config proc/configs/seis_ssl_cluster/prepare_volve_canonical_inputs.yaml \
  --smoke-config "$PRETRAIN/01_smoke_2step.yaml" \
  --full-config "$PRETRAIN/02_full_100ep.yaml" \
  --check full

python proc/seis_ssl_cluster/create_random_mae_checkpoint.py \
  --config "$PRETRAIN/03_create_random_checkpoint.yaml" --dry-run
python proc/seis_ssl_cluster/create_random_mae_checkpoint.py \
  --config "$PRETRAIN/03_create_random_checkpoint.yaml"
```
