# Volve amplitude MAE execution

Run from the repository root with the environment roots set explicitly:

```bash
export SEIS_SSL_CLUSTER_VOLVE_ROOT=/home/dcuser/public_data/field/volve
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/path/to/artifacts/seis_ssl_cluster
export EXP=experiments/volve/horizon_benchmark_v1
export PRETRAIN="$EXP/10_pretrain/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1"
export SMOKE_RUN="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/volve/horizon_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/smoke_2step"
export FULL_RUN="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/volve/horizon_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full_100ep"
export RANDOM_CHECKPOINT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/volve/horizon_benchmark_v1/random_encoder_amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_seed42_v1/random_init/mae_random_seed42.pt"

python proc/seis_ssl_cluster/prepare_volve_canonical_inputs.py \
  --only-missing

python proc/seis_ssl_cluster/validate_volve_mae.py \
  --input-config proc/configs/seis_ssl_cluster/prepare_volve_canonical_inputs.yaml \
  --smoke-config "$PRETRAIN/01_smoke_2step.yaml" \
  --full-config "$PRETRAIN/02_full_100ep.yaml" \
  --check inputs

python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$PRETRAIN/01_smoke_2step.yaml" --dry-run
if [ ! -f "$SMOKE_RUN/latest.pt" ]; then
  python proc/seis_ssl_cluster/train_amp_mae.py \
    --config "$PRETRAIN/01_smoke_2step.yaml"
fi
test -f "$SMOKE_RUN/latest.pt"

python proc/seis_ssl_cluster/validate_volve_mae.py \
  --input-config proc/configs/seis_ssl_cluster/prepare_volve_canonical_inputs.yaml \
  --smoke-config "$PRETRAIN/01_smoke_2step.yaml" \
  --full-config "$PRETRAIN/02_full_100ep.yaml" \
  --check smoke

python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$PRETRAIN/02_full_100ep.yaml" --dry-run
```

Run the following non-dry-run command only as a separately scheduled scientific
run:

```bash
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$PRETRAIN/02_full_100ep.yaml"
```

If that run is interrupted, resume it from its rolling `latest.pt` checkpoint:

```bash
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$PRETRAIN/02_full_100ep.yaml" \
  --resume "$FULL_RUN/latest.pt"
```

After epoch 100 completes, validate the run and create the paired random
baseline:

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

The random artifact is written to `$RANDOM_CHECKPOINT`. It is the seed-42
random-encoder baseline: it has the reference checkpoint's MAE architecture but
does not load its pretrained weights. Confirm its path, role, architecture, and
distinct content after creation:

```bash
python - <<'PY'
import hashlib
import os
from pathlib import Path

from seis_ssl_cluster.training.random_checkpoint import (
    load_checkpoint_metadata_without_weights,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

full_run = Path(os.environ['FULL_RUN'])
random_checkpoint = Path(os.environ['RANDOM_CHECKPOINT'])
reference_checkpoint = full_run / 'latest.pt'
reference_payload = load_checkpoint_metadata_without_weights(reference_checkpoint)
random_payload = load_checkpoint_metadata_without_weights(random_checkpoint)

assert random_checkpoint.name == 'mae_random_seed42.pt'
assert random_payload['metadata']['random_encoder_baseline'] is True
assert random_payload['metadata']['pretrained_weights_loaded'] is False
assert random_payload['metadata']['seed'] == 42
assert random_payload['config']['model'] == reference_payload['config']['model']
assert file_sha256(random_checkpoint) != file_sha256(reference_checkpoint)
print(random_checkpoint)
print('role: frozen_random; architecture: matches full_100ep/latest.pt')
PY
```

The smoke and full resolved configs may differ only in output location, runtime
resources, duration, AMP/device, `train.max_steps`, and debug visualization
enablement. `latest.pt` after epoch 100 is the downstream reference. `best.pt`
is a training-loss diagnostic and is not used for downstream selection. Each
training run snapshots the registered normalization stats and canonical input
metadata under `inputs/`; validation binds those copies and their hashes to the
current registration identity.
