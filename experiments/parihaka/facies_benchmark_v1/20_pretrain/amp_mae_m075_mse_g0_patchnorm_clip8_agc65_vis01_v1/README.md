# Parihaka amplitude MAE execution

This runbook owns phase ordering only. Data provenance and claim boundaries are
recorded in
[`docs/parihaka_mae_pretraining.md`](../../../../../docs/parihaka_mae_pretraining.md);
the YAML and resolvers define settings and completion criteria, which the
validators enforce.

Run from the repository root with the environment roots set explicitly:

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/path/to/artifacts/seis_ssl_cluster
export PARIHAKA_DATA_ROOT=/path/containing/parihaka_data_train.npz
export EXP=experiments/parihaka/facies_benchmark_v1
export PRETRAIN="$EXP/20_pretrain/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1"

python proc/seis_ssl_cluster/prepare_parihaka_volume.py \
  --config "$EXP/10_prepare/01_prepare_parihaka_volume.yaml" --dry-run

python proc/seis_ssl_cluster/prepare_parihaka_volume.py \
  --config "$EXP/10_prepare/01_prepare_parihaka_volume.yaml"

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
    reports/parihaka/facies_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1
```
