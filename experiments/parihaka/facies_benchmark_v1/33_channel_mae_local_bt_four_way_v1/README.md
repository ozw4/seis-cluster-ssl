# Parihaka Channel MAE/local-BT four-way v1

This experiment compares four fixed-budget Stage 2 encoders with the same
frozen-embedding Channel decoder contract:

- `mae`
- `mae_hmm_k6`
- `local_barlow_twins`
- `local_barlow_twins_hmm_k6`

The `mae` and `mae_hmm_k6` embedding volumes and their 30 decoder jobs are
reused in place from `32_channel_ssl_hmm_four_way_v1`. Do not re-extract the MAE
embeddings, rerun the MAE decoder jobs, or copy either set of artifacts. The only
new executions are two local-BT embedding extractions and 30 decoder jobs for
the two local-BT model IDs.

All four embedding sources use the fixed-budget `full_25ep/latest.pt`
checkpoint. Extraction consumes the full-volume encoder token embeddings; the
projector and HMM head are not downstream inputs. The encoder remains frozen and
only the common binary decoder is trained. Validation selects each decoder
job's `best.pt`, and the existing common voxel-complement defines the test set.
The primary metric is `test.channel_iou`.

This is a survey-specific, transductive evaluation: SSL pretraining saw the
unlabelled Parihaka amplitude volume. It is not a new-survey transfer test. All
models use the five reviewed layouts `layout_000` through `layout_004` and the
three supervision sizes `small`, `medium`, and `large`, for `4 * 5 * 3 = 60`
metrics.

## Environment

Run from an installed development checkout. Reuse the reviewed layout file
directly; do not copy or regenerate it.

```bash
cd /workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export EXP=experiments/parihaka/facies_benchmark_v1/33_channel_mae_local_bt_four_way_v1
export PREVIOUS_EXP=experiments/parihaka/facies_benchmark_v1/32_channel_ssl_hmm_four_way_v1
export LAYOUT_CONFIG=experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml
export CUDA_VISIBLE_DEVICES=1
```

The prepared amplitude manifest, Channel labels and metadata, reviewed layouts,
the four Stage 2 checkpoints, the two MAE embedding volumes, and the existing 30
MAE metrics must already exist.

## 1. Run targeted tests

Run the config, source, runner/resume, summary, CLI, and runbook coverage:

```bash
pytest -q \
  tests/seis_ssl_cluster/test_parihaka_channel_decoder.py \
  tests/seis_ssl_cluster/test_parihaka_channel_results.py \
  tests/seis_ssl_cluster/test_parihaka_channel_mae_local_bt_four_way_configs.py \
  tests/seis_ssl_cluster/test_parihaka_channel_mae_local_bt_four_way_runbook.py

pytest -q \
  tests/seis_ssl_cluster/test_proc_dry_run.py \
  -k parihaka_channel
```

These tests do not require live checkpoints or embedding artifacts.

## 2. Audit the reused MAE artifacts

Run this read-only audit before creating any local-BT result. It checks the
three files in each existing MAE embedding source, validates 15 metrics for each
MAE model through the public result inspector, and verifies both fixed-budget
checkpoints. It writes nothing.

```bash
python - <<'PY'
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import output_paths
from seis_ssl_cluster.parihaka.channel_decoder import (
    channel_decoder_config_from_mapping,
)
from seis_ssl_cluster.parihaka.channel_results import (
    channel_summary_config_from_mapping,
    inspect_channel_model_results,
)
from seis_ssl_cluster.training import load_checkpoint

exp = Path(os.environ['EXP'])
previous_exp = Path(os.environ['PREVIOUS_EXP'])
model_ids = ('mae', 'mae_hmm_k6')
previous = channel_decoder_config_from_mapping(
    load_config(previous_exp / '05_channel_four_way.yaml')
)
summary = channel_summary_config_from_mapping(
    load_config(exp / '03_channel_mae_local_bt_four_way.yaml')
)

for model_id in model_ids:
    source = previous.models[model_id]
    paths = output_paths(source.embedding_dir, previous.survey_id)
    required = (paths.embeddings, paths.valid_tokens, paths.metadata)
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError(f'{model_id}: missing embedding artifacts: {missing}')
    checkpoint = source.expected_checkpoint
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(f'{model_id}: configured checkpoint is missing')
    payload = load_checkpoint(checkpoint, map_location='cpu')
    if payload.get('epoch') != 25:
        raise ValueError(f'{model_id}: checkpoint epoch must be 25')
    if payload.get('global_step') != 15_625:
        raise ValueError(f'{model_id}: checkpoint global_step must be 15625')
    if payload.get('amp_enabled') is not False:
        raise ValueError(f'{model_id}: checkpoint amp_enabled must be false')
    del payload

jobs = inspect_channel_model_results(summary, model_ids=model_ids)
counts = Counter(model_id for model_id, _, _ in jobs)
if counts != Counter({'mae': 15, 'mae_hmm_k6': 15}):
    raise ValueError(f'unexpected reused MAE job counts: {dict(counts)}')
print('reused MAE audit: passed (6 embedding files, 30 metrics)')
PY
```

Leave these embedding directories, checkpoints, and decoder result directories
unchanged. They remain under the existing four-way artifact roots.

## 3. Extract only the two local-BT embedding volumes

Inspect both explicit configs first:

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/01_extract_local_barlow_twins_embeddings.yaml" \
  --dry-run

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/02_extract_local_barlow_twins_hmm_k6_embeddings.yaml" \
  --dry-run
```

Confirm that each dry-run binds its own `full_25ep/latest.pt`, output directory,
window, overlap, dtype, and FP32 extraction settings. Then run the same configs
in order:

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/01_extract_local_barlow_twins_embeddings.yaml"

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/02_extract_local_barlow_twins_hmm_k6_embeddings.yaml"
```

Each output directory must contain exactly the expected producer files:

```text
parihaka.embeddings.npy
parihaka.valid_tokens.npy
parihaka.embedding_metadata.json
```

There is intentionally no MAE extraction config in this experiment.

## 4. Audit all four live embedding sources

Run this read-only audit after both local extractions. The public Channel source
inspector validates file identity and the shared frozen-embedding contract; the
remaining checks make the fixed-budget and objective identities explicit.

```bash
python - <<'PY'
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.parihaka.channel_decoder import (
    channel_decoder_config_from_mapping,
    inspect_embedding_sources,
)
from seis_ssl_cluster.training import load_checkpoint

exp = Path(os.environ['EXP'])
model_ids = (
    'mae',
    'local_barlow_twins',
    'mae_hmm_k6',
    'local_barlow_twins_hmm_k6',
)
common_keys = (
    'survey_id',
    'source_amplitude_path',
    'volume_shape_xyz',
    'model_geometry',
    'patch_size',
    'token_grid_shape',
    'window_size',
    'overlap',
    'output_dtype',
    'min_token_valid_fraction',
    'normalization_stats_path',
    'preprocessing',
    'zero_mask',
    'precision',
)


def mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f'{label} must be a mapping')
    return value


config = channel_decoder_config_from_mapping(
    load_config(exp / '03_channel_mae_local_bt_four_way.yaml')
)
geometry = inspect_embedding_sources(config)
if tuple(geometry.models) != model_ids:
    raise ValueError(f'unexpected model order: {tuple(geometry.models)}')

checkpoint_shas: set[str] = set()
embedding_signatures: set[tuple[tuple[int, ...], str]] = set()
valid_masks: dict[str, np.ndarray] = {}
payload_configs: dict[str, Mapping[str, object]] = {}
for model_id, item in geometry.models.items():
    source = config.models[model_id]
    checkpoint = source.expected_checkpoint
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(f'{model_id}: configured checkpoint is missing')
    metadata_checkpoint = Path(str(item.metadata.get('checkpoint_path')))
    if metadata_checkpoint.resolve() != checkpoint.resolve():
        raise ValueError(f'{model_id}: metadata/config checkpoint mismatch')
    actual_sha = file_sha256(checkpoint)
    if item.metadata.get('checkpoint_sha256') != actual_sha:
        raise ValueError(f'{model_id}: metadata/checkpoint SHA-256 mismatch')
    checkpoint_shas.add(actual_sha)
    embeddings = np.load(item.paths.embeddings, mmap_mode='r', allow_pickle=False)
    valid_masks[model_id] = np.load(
        item.paths.valid_tokens, mmap_mode='r', allow_pickle=False
    )
    embedding_signatures.add((tuple(embeddings.shape), str(embeddings.dtype)))
    payload = mapping(
        load_checkpoint(checkpoint, map_location='cpu'),
        f'{model_id} checkpoint payload',
    )
    if payload.get('epoch') != 25:
        raise ValueError(f'{model_id}: checkpoint epoch must be 25')
    if payload.get('global_step') != 15_625:
        raise ValueError(f'{model_id}: checkpoint global_step must be 15625')
    if payload.get('amp_enabled') is not False:
        raise ValueError(f'{model_id}: checkpoint amp_enabled must be false')
    payload_configs[model_id] = mapping(
        payload.get('config'), f'{model_id} checkpoint config'
    )
    del payload

if len(checkpoint_shas) != 4:
    raise ValueError('the four checkpoints must be distinct')
if len(embedding_signatures) != 1:
    raise ValueError('embedding shape/dtype mismatch')
reference = geometry.models['mae'].metadata
reference_valid = valid_masks['mae']
for model_id, item in geometry.models.items():
    for key in common_keys:
        if item.metadata.get(key) != reference.get(key):
            raise ValueError(f'{model_id}: common metadata {key} mismatch')
    if not np.array_equal(valid_masks[model_id], reference_valid):
        raise ValueError(f'{model_id}: valid-token mask mismatch')

mae_config = payload_configs['mae']
if mae_config.get('stage') != 'train_amp_mae' or not isinstance(
    mae_config.get('continuation'), Mapping
):
    raise ValueError('mae must be an MAE continuation')

local_objective = mapping(
    geometry.models['local_barlow_twins'].metadata.get('pretraining_objective'),
    'local_barlow_twins pretraining objective',
)
if local_objective.get('method') != 'local_barlow_twins_3d':
    raise ValueError('local_barlow_twins objective method mismatch')
if local_objective.get('local_pairs_per_crop') != 128:
    raise ValueError('local_barlow_twins local pair count must be 128')

for model_id, expected_base in (
    ('mae_hmm_k6', 'amp_mae3d'),
    ('local_barlow_twins_hmm_k6', 'local_barlow_twins_3d'),
):
    pretext = mapping(
        geometry.models[model_id].metadata.get('stratigraphy_pretext'),
        f'{model_id} stratigraphy pretext',
    )
    if pretext.get('method') != 'strat_hmm_pretext':
        raise ValueError(f'{model_id}: pretext method mismatch')
    if pretext.get('base_objective') != expected_base:
        raise ValueError(f'{model_id}: base objective mismatch')
    if pretext.get('head_num_prototypes') != 6:
        raise ValueError(f'{model_id}: prototype count mismatch')

for model_id, item in geometry.models.items():
    print(model_id, item.model_source['checkpoint_sha256'])
print('live embedding audit: passed')
PY
```

Model-specific objective and HMM metadata may differ. Geometry,
preprocessing, precision, embedding shape/dtype, and valid-token masks may not.

## 5. Preflight one common condition

Dry-run `layout_000` and `small` for all four models:

```bash
for model in mae local_barlow_twins mae_hmm_k6 local_barlow_twins_hmm_k6; do
  python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
    --config "$EXP/03_channel_mae_local_bt_four_way.yaml" \
    --model "$model" \
    --layout layout_000 \
    --size small \
    --layout-config "$LAYOUT_CONFIG" \
    --dry-run
done
```

Compare the selected inline/crossline indices, selected-token SHA-256, actual
train-voxel count, class weights, split counts, tile counts, and decoder
initial-state SHA-256. Those values must agree for all four models. Only the
checkpoint/source and output paths may differ.

## 6. Run only the 30 new local-BT jobs

Start fresh jobs without `--resume`. The execution loop deliberately contains
only the two local-BT model IDs:

```bash
for model in local_barlow_twins local_barlow_twins_hmm_k6; do
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
    for size in small medium large; do
      python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
        --config "$EXP/03_channel_mae_local_bt_four_way.yaml" \
        --model "$model" \
        --layout "$layout" \
        --size "$size" \
        --layout-config "$LAYOUT_CONFIG"
    done
  done
done
```

All jobs share the existing runs root:

```text
${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/channel_benchmark/ssl_hmm_four_way_v1/runs
```

The new model IDs create distinct result directories without altering the
existing MAE jobs. If one local job is interrupted, resume only that same job
from its own decoder `latest.pt`, for example:

```bash
python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
  --config "$EXP/03_channel_mae_local_bt_four_way.yaml" \
  --model local_barlow_twins_hmm_k6 \
  --layout layout_002 \
  --size medium \
  --layout-config "$LAYOUT_CONFIG" \
  --resume "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/ssl_hmm_four_way_v1/runs/model=local_barlow_twins_hmm_k6/layout=layout_002/size=medium/latest.pt"
```

Resume requires the same model, embedding SHA, layout, supervision, decoder,
tiles, and training identity.

## 7. Audit all 60 metrics and write the summary

The dedicated summary dry-run is the final completeness and identity audit. It
must pass before any summary file is written:

```bash
python proc/seis_ssl_cluster/summarize_parihaka_channel_mae_local_bt.py \
  --config "$EXP/03_channel_mae_local_bt_four_way.yaml" \
  --dry-run
```

The expected count is:

```text
complete_jobs: 60
```

Then write the independent MAE/local-BT summary:

```bash
python proc/seis_ssl_cluster/summarize_parihaka_channel_mae_local_bt.py \
  --config "$EXP/03_channel_mae_local_bt_four_way.yaml"
```

The summary output root is separate from the existing SSL/HMM summary:

```text
${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/channel_benchmark/mae_local_bt_four_way_v1/summary
```

It contains `comparison.csv`, `summary.json`, and `summary.md`. The runbook does
not prescribe result values or winners; interpret the four paired comparisons
only after the complete summary has been generated.
