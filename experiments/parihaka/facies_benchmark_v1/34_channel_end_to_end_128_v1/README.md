# Parihaka Channel end-to-end 128³ v1

This experiment is a paired end-to-end comparison between a trainable encoder
initialized from the Parihaka MAE checkpoint (`pretrained`, reported as
`finetune_pretrained`) and the same trainable encoder initialized from the
seed-42 random checkpoint (`random`, reported as `train_from_scratch`). The
encoder and decoder are both trainable. Encoder initialization is the only
condition changed within each pair.

Each supervised tile encodes a raw `128³` amplitude crop. The supervised core
remains the central `64³` voxels used by the 80³ experiment: the core is still
`8³` tokens, while the context halo alone increases to `4` tokens, or 32 voxels
on each side. Five layouts, three supervision sizes, and two initializations
give `5 × 3 × 2 = 30 jobs`.

The 80³ v1 experiment and its `channel_end_to_end` artifacts are not reused or
overwritten. This experiment writes only to `channel_end_to_end_128_v1`.

The MAE was pretrained on the full unlabeled Parihaka amplitude volume, so this
is a survey-specific, transductive evaluation. Frozen and end-to-end regimes do
not differ only by fine-tuning: frozen evaluation supplies the decoder with
offline overlap-aggregated full-volume embeddings, whereas end-to-end
evaluation encodes each raw supervised tile during training. Cross-regime score
differences therefore do not isolate encoder fine-tuning.

All training uses FP32 with AMP disabled. If the fixed 128³ condition causes a
CUDA OOM, stop before the 30-job run. Do not change AMP, crop geometry, batch
size, encoder structure, or attention behavior as a fallback.

## Environment

Use an installed development checkout and the reviewed concrete layout config.

```bash
cd /workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export EXP=experiments/parihaka/facies_benchmark_v1/34_channel_end_to_end_128_v1
export CONFIG="$EXP/01_channel_end_to_end_128.yaml"
export LAYOUT_CONFIG=experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml
export RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_end_to_end_128_v1/runs"
export CUDA_VISIBLE_DEVICES=1
```

The prepared labels, label metadata, reference embedding metadata and
valid-token mask, source amplitude and normalization statistics, and both
encoder checkpoints must already exist. This runbook does not regenerate or
modify those sources.

## 1. Run the targeted tests

```bash
pytest -q \
  tests/seis_ssl_cluster/test_parihaka_channel_end_to_end.py \
  tests/seis_ssl_cluster/test_parihaka_channel_end_to_end_results.py \
  tests/seis_ssl_cluster/test_parihaka_channel_end_to_end_128_configs.py \
  tests/seis_ssl_cluster/test_parihaka_channel_end_to_end_128_runbook.py
```

These portable tests validate both the existing 80³ contract and the new 128³
geometry without running a full-size Transformer forward.

## 2. Audit the live sources

Run this read-only audit before any CUDA job. It uses the existing public config,
job-plan, and encoder-checkpoint helpers and writes nothing.

```bash
python - <<'PY'
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.parihaka.channel_end_to_end import (
    channel_end_to_end_config_from_mapping,
    encoder_initial_state_sha256,
    inspect_channel_end_to_end_job,
)

config_path = Path(os.environ['CONFIG'])
layout_path = Path(os.environ['LAYOUT_CONFIG'])
v1_path = Path(
    'experiments/parihaka/facies_benchmark_v1/'
    '31_channel_end_to_end_v1/01_channel_end_to_end.yaml'
)
config = channel_end_to_end_config_from_mapping(load_config(config_path))
v1 = channel_end_to_end_config_from_mapping(load_config(v1_path))

for label, path in (
    ('labels', config.labels),
    ('label metadata', config.labels_metadata),
    ('pretrained checkpoint', config.pretrained_checkpoint),
    ('random checkpoint', config.random_checkpoint),
):
    if not path.is_file():
        raise FileNotFoundError(f'{label} is missing: {path}')
if not layout_path.is_file():
    raise FileNotFoundError(f'layout config is missing: {layout_path}')

plans = {
    encoder_init: inspect_channel_end_to_end_job(
        config,
        encoder_init=encoder_init,
        layout_id='layout_000',
        data_size='small',
        layout_config=layout_path,
        device='cpu',
    )
    for encoder_init in ('pretrained', 'random')
}
pretrained = plans['pretrained']
random = plans['random']
reference = pretrained.reference
for label, path in (
    ('reference metadata', reference.metadata_path),
    ('reference valid-token mask', reference.valid_tokens_path),
    ('source amplitude', reference.source_amplitude_path),
    ('normalization statistics', reference.normalization_stats_path),
):
    if not path.is_file():
        raise FileNotFoundError(f'{label} is missing: {path}')
if not isinstance(reference.preprocessing, Mapping):
    raise TypeError('reference preprocessing must be a mapping')
if not isinstance(reference.zero_mask, Mapping):
    raise TypeError('reference zero-mask settings must be a mapping')
if pretrained.model_geometry != random.model_geometry:
    raise ValueError('pretrained/random checkpoint model geometry mismatch')

pretrained_sha = encoder_initial_state_sha256(config.pretrained_checkpoint)
random_sha = encoder_initial_state_sha256(config.random_checkpoint)
if pretrained_sha != pretrained.pretrained_encoder_initial_state_sha256:
    raise ValueError('pretrained encoder initial-state SHA mismatch')
if random_sha != random.random_encoder_initial_state_sha256:
    raise ValueError('random encoder initial-state SHA mismatch')
if pretrained_sha == random_sha:
    raise ValueError('paired encoder initial states must differ')

core = config.tiles.core_size_tokens
halo = config.tiles.context_halo_tokens
if core != (8, 8, 8) or halo != (4, 4, 4):
    raise ValueError(f'unexpected tile geometry: core={core}, halo={halo}')
raw_input_shape = tuple(
    (core_axis + 2 * halo_axis) * patch_axis
    for core_axis, halo_axis, patch_axis in zip(
        core, halo, reference.patch_size_xyz, strict=True
    )
)
if raw_input_shape != (128, 128, 128):
    raise ValueError(f'raw input shape must be 128³; got {raw_input_shape}')
if config.train != v1.train:
    raise ValueError('128³ train settings differ from the 80³ v1 contract')

new_outputs = (config.runs_root, config.output_dir, config.four_way_output_dir)
v1_outputs = (v1.runs_root, v1.output_dir, v1.four_way_output_dir)
for new_path in new_outputs:
    for old_path in v1_outputs:
        if (
            new_path == old_path
            or new_path in old_path.parents
            or old_path in new_path.parents
        ):
            raise ValueError(f'output roots overlap: {new_path} and {old_path}')

print('model geometry:', pretrained.model_geometry.as_dict())
print('pretrained encoder initial-state SHA:', pretrained_sha)
print('random encoder initial-state SHA:', random_sha)
print('preprocessing:', dict(reference.preprocessing))
print('zero mask:', dict(reference.zero_mask))
print('raw input shape:', raw_input_shape)
print('live source audit: passed')
PY
```

The public job inspector also validates label identity, reference/checkpoint
roles, checkpoint digests, preprocessing, zero-mask settings, token validity,
and the minimum context required by the decoder.

## 3. Dry-run all 30 conditions

```bash
for encoder_init in pretrained random; do
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
    for size in small medium large; do
      python proc/seis_ssl_cluster/run_parihaka_channel_end_to_end.py \
        --config "$CONFIG" \
        --encoder-init "$encoder_init" \
        --layout "$layout" \
        --size "$size" \
        --layout-config "$LAYOUT_CONFIG" \
        --device cuda \
        --dry-run
    done
  done
done
```

Then use the same public plan helper to fail closed on any paired drift. For
each layout and size, selected-token SHA, actual train voxel count, class
weights, split counts, tile counts and IDs, decoder initial-state SHA,
preprocessing, runtime precision, and tile geometry must match. The
condition-specific encoder source and selected encoder initial-state SHA, the
condition label, and output path must differ.

```bash
python - <<'PY'
from __future__ import annotations

import copy
import os
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.parihaka.channel_end_to_end import (
    channel_end_to_end_config_from_mapping,
    inspect_channel_end_to_end_job,
)

config = channel_end_to_end_config_from_mapping(
    load_config(Path(os.environ['CONFIG']))
)
layout_config = Path(os.environ['LAYOUT_CONFIG'])
for layout in (f'layout_{index:03d}' for index in range(5)):
    for size in ('small', 'medium', 'large'):
        plans = {
            encoder_init: inspect_channel_end_to_end_job(
                config,
                encoder_init=encoder_init,
                layout_id=layout,
                data_size=size,
                layout_config=layout_config,
                device='cuda',
            )
            for encoder_init in ('pretrained', 'random')
        }
        pretrained = plans['pretrained']
        random = plans['random']
        pretrained_common = copy.deepcopy(dict(pretrained.benchmark_identity))
        random_common = copy.deepcopy(dict(random.benchmark_identity))
        for identity in (pretrained_common, random_common):
            identity.pop('encoder_init')
            identity.pop('encoder_source')
        if pretrained_common != random_common:
            raise ValueError(f'paired benchmark identity drift: {layout}/{size}')
        if pretrained.tile_ids != random.tile_ids:
            raise ValueError(f'paired tile ID drift: {layout}/{size}')
        if pretrained.output_dir == random.output_dir:
            raise ValueError(f'paired output paths overlap: {layout}/{size}')
        if (
            pretrained.pretrained_encoder_initial_state_sha256
            == random.random_encoder_initial_state_sha256
        ):
            raise ValueError(f'encoder initial states match: {layout}/{size}')
        print('paired dry-run identity: passed', layout, size)
PY
```

Do not start feasibility until all 30 dry-runs and all 15 paired identity checks
pass.

## 4. Run paired CUDA one-step feasibility

Run both initializations for the same `layout_000/small` condition. Each command
must finish without CUDA OOM and produce its own `latest.pt` at
`global_step == 1`.

```bash
for encoder_init in pretrained random; do
  python proc/seis_ssl_cluster/run_parihaka_channel_end_to_end.py \
    --config "$CONFIG" \
    --encoder-init "$encoder_init" \
    --layout layout_000 \
    --size small \
    --layout-config "$LAYOUT_CONFIG" \
    --device cuda \
    --max-steps 1
done
```

Monitor the selected GPU in another terminal:

```bash
watch -n 2 \
  'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'
```

After both commands finish, audit their restart checkpoints:

```bash
python - <<'PY'
from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

runs_root = Path(os.environ['RUNS_ROOT'])


def mapping(value: object, label: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f'{label} must be a mapping')
    return value


def triplet(value: object, label: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise TypeError(f'{label} must be an integer triplet')
    return tuple(value)


def require_finite(value: object, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise FloatingPointError(f'{label} contains a non-finite tensor')
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f'{label} contains a non-finite value')
    elif isinstance(value, Mapping):
        for key, child in value.items():
            require_finite(child, f'{label}.{key}')
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            require_finite(child, f'{label}[{index}]')


decoder_initial_shas: set[object] = set()
encoder_initial_shas: set[object] = set()
reference_inputs: list[object] = []
for encoder_init in ('pretrained', 'random'):
    checkpoint = (
        runs_root
        / f'encoder_init={encoder_init}'
        / 'layout=layout_000'
        / 'size=small'
        / 'latest.pt'
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f'feasibility checkpoint is missing: {checkpoint}')
    payload = mapping(
        torch.load(checkpoint, map_location='cpu', weights_only=False),
        f'{encoder_init} checkpoint',
    )
    if payload.get('global_step') != 1:
        raise ValueError(f'{encoder_init}: global_step must equal 1')
    if payload.get('completed') is not False:
        raise ValueError(f'{encoder_init}: one-step checkpoint must be resumable')
    if payload.get('scaler_state_dict') is not None:
        raise ValueError(f'{encoder_init}: AMP scaler state must be absent')

    identity = mapping(payload.get('run_identity'), f'{encoder_init} identity')
    tiles = mapping(identity.get('tiles'), f'{encoder_init} tiles')
    core = triplet(tiles.get('core_size_tokens'), f'{encoder_init} core')
    halo = triplet(tiles.get('context_halo_tokens'), f'{encoder_init} halo')
    if core != (8, 8, 8) or halo != (4, 4, 4):
        raise ValueError(f'{encoder_init}: checkpoint is not core8/halo4')
    reference = mapping(
        identity.get('reference_input'), f'{encoder_init} reference input'
    )
    patch = triplet(reference.get('patch_size'), f'{encoder_init} patch size')
    raw_input_shape = tuple(
        (core_axis + 2 * halo_axis) * patch_axis
        for core_axis, halo_axis, patch_axis in zip(core, halo, patch, strict=True)
    )
    if raw_input_shape != (128, 128, 128):
        raise ValueError(f'{encoder_init}: raw input identity is not 128³')
    runtime = mapping(identity.get('runtime'), f'{encoder_init} runtime')
    if runtime.get('amp_enabled') is not False:
        raise ValueError(f'{encoder_init}: runtime must be FP32 without AMP')

    optimizer = mapping(
        payload.get('optimizer_state_dict'), f'{encoder_init} optimizer'
    )
    groups = optimizer.get('param_groups')
    if not isinstance(groups, list):
        raise TypeError(f'{encoder_init}: optimizer groups must be a list')
    group_names = [mapping(group, 'optimizer group').get('name') for group in groups]
    if group_names != ['encoder', 'decoder']:
        raise ValueError(f'{encoder_init}: expected encoder and decoder groups')
    optimizer_state = mapping(
        optimizer.get('state'), f'{encoder_init} optimizer state'
    )
    if not optimizer_state:
        raise ValueError(f'{encoder_init}: optimizer state is empty after one step')
    require_finite(optimizer_state, f'{encoder_init} optimizer state')
    train_loss_sum = payload.get('train_loss_sum')
    train_voxels = payload.get('train_voxels')
    if not isinstance(train_loss_sum, (int, float)) or not math.isfinite(train_loss_sum):
        raise FloatingPointError(f'{encoder_init}: cumulative loss is not finite')
    if not isinstance(train_voxels, int) or train_voxels <= 0:
        raise ValueError(f'{encoder_init}: supervised voxel count must be positive')

    decoder = mapping(identity.get('decoder'), f'{encoder_init} decoder')
    encoder = mapping(identity.get('encoder_source'), f'{encoder_init} encoder')
    decoder_initial_shas.add(decoder.get('initial_state_sha256'))
    encoder_initial_shas.add(encoder.get('initial_state_sha256'))
    reference_inputs.append(reference)

if len(decoder_initial_shas) != 1:
    raise ValueError('paired decoder initial-state SHAs differ')
if len(encoder_initial_shas) != 2:
    raise ValueError('paired encoder initial-state SHAs must differ')
if reference_inputs[0] != reference_inputs[1]:
    raise ValueError('paired raw input identities differ')
print('paired CUDA one-step feasibility: passed')
PY
```

The training runner raises immediately on a non-finite loss or gradient norm;
the checkpoint audit additionally requires finite cumulative loss and finite
Adam gradient-moment state. Feasibility passes only if both checkpoints satisfy
the audit and neither run reports CUDA OOM. On OOM, do not alter the fixed
condition and do not begin the full experiment. The two feasibility jobs remain
in place and the next loop resumes them from their own `latest.pt`.

## 5. Run the restartable 30-job loop

```bash
for encoder_init in pretrained random; do
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
    for size in small medium large; do
      JOB_DIR="$RUNS_ROOT/encoder_init=$encoder_init/layout=$layout/size=$size"

      if [ -f "$JOB_DIR/metrics.json" ]; then
        echo "completed: $encoder_init/$layout/$size"
      elif [ -f "$JOB_DIR/latest.pt" ]; then
        python proc/seis_ssl_cluster/run_parihaka_channel_end_to_end.py \
          --config "$CONFIG" \
          --encoder-init "$encoder_init" \
          --layout "$layout" \
          --size "$size" \
          --layout-config "$LAYOUT_CONFIG" \
          --device cuda \
          --resume "$JOB_DIR/latest.pt"
      else
        python proc/seis_ssl_cluster/run_parihaka_channel_end_to_end.py \
          --config "$CONFIG" \
          --encoder-init "$encoder_init" \
          --layout "$layout" \
          --size "$size" \
          --layout-config "$LAYOUT_CONFIG" \
          --device cuda
      fi
    done
  done
done
```

This distinguishes a completed job (`metrics.json`), an interrupted or
one-step feasibility job (`latest.pt` only), and a fresh job (neither file).

## 6. Check progress

Count completed jobs separately for each initialization:

```bash
for encoder_init in pretrained random; do
  count=$(
    find "$RUNS_ROOT/encoder_init=$encoder_init" \
      -type f \
      -name metrics.json \
      2>/dev/null |
    wc -l
  )
  echo "$encoder_init: $count/15"
done
```

A job with `latest.pt` but without `metrics.json` is running or interrupted and
must be resumed by the loop above. Continue only when both counts are `15/15`.

## 7. Summarize all 30 metrics

First validate completeness and identity without writing summary files:

```bash
python proc/seis_ssl_cluster/summarize_parihaka_channel_end_to_end.py \
  --config "$CONFIG" \
  --dry-run
```

Expected output includes:

```text
complete_jobs: 30
```

Only after that succeeds, write the paired summary:

```bash
python proc/seis_ssl_cluster/summarize_parihaka_channel_end_to_end.py \
  --config "$CONFIG"
```

The output is:

```text
${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/channel_end_to_end_128_v1/summary/
├── comparison.csv
├── summary.json
└── summary.md
```

The primary paired delta is
`finetune_pretrained - train_from_scratch`.

## 8. Optionally write the four-way descriptive report

Run this only after the 30-job paired E2E summary is complete and only if the
existing frozen benchmark is also complete:

```bash
FROZEN_CONFIG=experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/06_channel_benchmark.yaml
python proc/seis_ssl_cluster/summarize_parihaka_channel_four_way.py \
  --config "$CONFIG" \
  --frozen-config "$FROZEN_CONFIG" \
  --dry-run
python proc/seis_ssl_cluster/summarize_parihaka_channel_four_way.py \
  --config "$CONFIG" \
  --frozen-config "$FROZEN_CONFIG"
```

This optional report is descriptive. It retains the separate paired deltas
`frozen_pretrained - frozen_random` and
`finetune_pretrained - train_from_scratch`; it does not add a cross-regime
delta. Frozen evaluation uses offline overlap-aggregated full-volume
embeddings, while E2E evaluation encodes raw tiles on the fly during supervised
training, so a cross-regime score difference is not a fine-tuning-only effect.
