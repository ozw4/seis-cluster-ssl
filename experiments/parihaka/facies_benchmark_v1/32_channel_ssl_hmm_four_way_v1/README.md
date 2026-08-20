# Parihaka Channel SSL/HMM four-way v1

This experiment compares four Stage 2 encoders with one frozen-embedding
Channel decoder contract. Embeddings are extracted once over the full Parihaka
volume. Every downstream job freezes the selected embedding volume and trains
only the same binary decoder.

The comparison is survey-specific and transductive: SSL pretraining saw the
unlabelled Parihaka amplitude volume. It does not measure transfer to a new
survey. Validation sections select `best.pt`; the common voxel-complement test
is evaluated once from that checkpoint.

## Experiment matrix

The model IDs are:

- `mae`
- `barlow_twins`
- `mae_hmm_k6`
- `barlow_twins_hmm_k6`

All models use the five reviewed layouts `layout_000` through `layout_004` and
the three supervision sizes `small`, `medium`, and `large`. The complete matrix
contains `4 * 5 * 3 = 60` frozen decoder jobs.

The MAE and Barlow Twins controls are 25-epoch continuations of their Stage 1
checkpoints. The HMM conditions use the matching K=6 stratigraphic pretext
continuation. All extraction configs bind `full_25ep/latest.pt`; feasibility and
`best.pt` checkpoints do not belong in this experiment.

## Environment

Run from an installed development checkout. The artifact root must be absolute.
Reuse the reviewed layout file directly; do not copy or regenerate it.

```bash
cd /workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export EXP=experiments/parihaka/facies_benchmark_v1/32_channel_ssl_hmm_four_way_v1
export LAYOUT_CONFIG=experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml
export CUDA_VISIBLE_DEVICES=1
```

The prepared amplitude manifest, Channel labels, label metadata, and four Stage
2 checkpoints must already exist. This runbook does not regenerate labels,
layouts, pseudo-targets, or pretraining checkpoints.

## Targeted tests

Run the Channel source, runner/resume, summary, config, and runbook coverage:

```bash
pytest -q \
  tests/seis_ssl_cluster/test_parihaka_channel_decoder.py \
  tests/seis_ssl_cluster/test_parihaka_channel_results.py \
  tests/seis_ssl_cluster/test_parihaka_channel_ssl_hmm_four_way_configs.py \
  tests/seis_ssl_cluster/test_parihaka_channel_ssl_hmm_four_way_runbook.py

pytest -q \
  tests/seis_ssl_cluster/test_proc_dry_run.py \
  -k parihaka_channel
```

These include the legacy `pretrained`/`random` pair. Portable tests do not need
the live checkpoints or embedding artifacts.

## Extract the four embedding volumes

Dry-run every explicit config and review its checkpoint, output directory,
window, overlap, dtype, and AMP settings:

```bash
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/01_extract_mae_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/02_extract_barlow_twins_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/03_extract_mae_hmm_k6_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/04_extract_barlow_twins_hmm_k6_embeddings.yaml" --dry-run
```

Then execute the same four configs in order. Do not construct checkpoint paths
inside a shell loop; the reviewed YAML files own those bindings.

```bash
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/01_extract_mae_embeddings.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/02_extract_barlow_twins_embeddings.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/03_extract_mae_hmm_k6_embeddings.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/04_extract_barlow_twins_hmm_k6_embeddings.yaml"
```

Each output directory must contain the three `parihaka.embeddings.npy`,
`parihaka.valid_tokens.npy`, and `parihaka.embedding_metadata.json` artifacts.

## Audit the live embedding sources

Run this one read-only auditor after extraction. It uses public Channel APIs,
then reads the actual metadata, NPY artifacts, and checkpoint payloads. It does
not use private validators and writes nothing.

```bash
python - <<'PY'
from __future__ import annotations
import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
import numpy as np
import torch
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.parihaka.channel_decoder import (
    channel_decoder_config_from_mapping,
    inspect_embedding_sources,
)
exp = Path(os.environ['EXP'])
model_ids = ('mae', 'barlow_twins', 'mae_hmm_k6', 'barlow_twins_hmm_k6')
common_keys = (
	'survey_id', 'source_amplitude_path', 'volume_shape_xyz', 'model_geometry',
	'patch_size', 'token_grid_shape', 'window_size', 'overlap', 'output_dtype',
	'min_token_valid_fraction', 'normalization_stats_path', 'preprocessing',
	'zero_mask', 'precision',
)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
def mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f'{label} must be a mapping')
    return value
config = channel_decoder_config_from_mapping(load_config(exp / '05_channel_four_way.yaml'))
geometry = inspect_embedding_sources(config)
if tuple(geometry.models) != model_ids:
    raise ValueError('unexpected model registry')
payload_configs: dict[str, Mapping[str, object]] = {}
valid_masks: dict[str, np.ndarray] = {}
embedding_signatures: set[tuple[tuple[int, ...], str]] = set()
checkpoint_shas: set[str] = set()
for model_id, embedding_input in geometry.models.items():
    source = config.models[model_id]
    checkpoint = source.expected_checkpoint
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(f'{model_id}: configured checkpoint is missing')
    metadata = embedding_input.metadata
    metadata_checkpoint = Path(str(metadata.get('checkpoint_path')))
    if metadata_checkpoint.resolve() != checkpoint.resolve():
        raise ValueError(f'{model_id}: metadata/config checkpoint mismatch')
    actual_sha = sha256(checkpoint)
    if metadata.get('checkpoint_sha256') != actual_sha:
        raise ValueError(f'{model_id}: checkpoint SHA-256 mismatch')
    checkpoint_shas.add(actual_sha)
    embeddings = np.load(embedding_input.paths.embeddings, mmap_mode='r', allow_pickle=False)
    valid = np.load(embedding_input.paths.valid_tokens, mmap_mode='r', allow_pickle=False)
    embedding_signatures.add((tuple(embeddings.shape), str(embeddings.dtype)))
    valid_masks[model_id] = valid
    payload = mapping(
        torch.load(checkpoint, map_location='cpu', weights_only=False),
        f'{model_id} checkpoint payload',
    )
    if payload.get('epoch') != 25:
        raise ValueError(
            f'{model_id}: checkpoint epoch must be 25; '
            f'got {payload.get("epoch")!r}'
        )
    if payload.get('global_step') != 15_625:
        raise ValueError(
            f'{model_id}: checkpoint global_step must be 15625; '
            f'got {payload.get("global_step")!r}'
        )
    if payload.get('amp_enabled') is not False:
        raise ValueError(f'{model_id}: checkpoint amp_enabled must be false')
    payload_configs[model_id] = mapping(payload.get('config'), f'{model_id} config')
    del payload
if len(checkpoint_shas) != 4:
    raise ValueError('the four checkpoints must be distinct')
if len(embedding_signatures) != 1:
    raise ValueError('embedding shape/dtype mismatch')
if {item.metadata.get('survey_id') for item in geometry.models.values()} != {'parihaka'}:
    raise ValueError('survey set mismatch')
reference = geometry.models['mae'].metadata
reference_valid = valid_masks['mae']
for model_id, item in geometry.models.items():
    for key in common_keys:
        if item.metadata.get(key) != reference.get(key):
            raise ValueError(f'{model_id}: common metadata {key} mismatch')
    if not np.array_equal(valid_masks[model_id], reference_valid):
        raise ValueError(f'{model_id}: valid-token mask mismatch')
mae_config = payload_configs['mae']
if mae_config.get('stage') != 'train_amp_mae' or not isinstance(mae_config.get('continuation'), Mapping):
    raise ValueError('mae must be a train_amp_mae continuation')
barlow_config = payload_configs['barlow_twins']
barlow_metadata = geometry.models['barlow_twins'].metadata
barlow_objective = mapping(
    barlow_metadata.get('pretraining_objective'),
    'barlow_twins pretraining objective',
)
if barlow_config.get('stage') != 'barlow_twins_training' or not isinstance(barlow_config.get('continuation'), Mapping):
    raise ValueError('barlow_twins must be a Barlow Twins continuation')
if barlow_objective.get('method') != 'barlow_twins_3d':
    raise ValueError('barlow_twins objective method mismatch')
for model_id, expected_base in (
    ('mae_hmm_k6', 'amp_mae3d'),
    ('barlow_twins_hmm_k6', 'barlow_twins_3d'),
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

Model-specific objective and stratigraphy metadata may differ. Geometry,
preprocessing, precision, embedding shape/dtype, and valid-token masks may not.

## Preflight one common condition

Dry-run the same layout and size for every model:

```bash
for model in mae barlow_twins mae_hmm_k6 barlow_twins_hmm_k6; do
  python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
    --config "$EXP/05_channel_four_way.yaml" \
    --model "$model" \
    --layout layout_000 \
    --size small \
    --layout-config "$LAYOUT_CONFIG" \
    --dry-run
done
```

Compare selection identity, actual train-voxel count, class weights, split
counts, tile counts, and decoder initial-state SHA-256. Only source identity,
checkpoint identity, and output path should differ.

## Run all 60 jobs

Start fresh jobs without `--resume`:

```bash
for model in mae barlow_twins mae_hmm_k6 barlow_twins_hmm_k6; do
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
    for size in small medium large; do
      python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
        --config "$EXP/05_channel_four_way.yaml" \
        --model "$model" \
        --layout "$layout" \
        --size "$size" \
        --layout-config "$LAYOUT_CONFIG"
    done
  done
done
```

Jobs write below `.../runs/model=<model>/layout=<layout>/size=<size>/`. If a job
is interrupted, resume only from that same job's `latest.pt`, for example:

```bash
python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
  --config "$EXP/05_channel_four_way.yaml" \
  --model mae_hmm_k6 \
  --layout layout_002 \
  --size medium \
  --layout-config "$LAYOUT_CONFIG" \
  --resume "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/ssl_hmm_four_way_v1/runs/model=mae_hmm_k6/layout=layout_002/size=medium/latest.pt"
```

Resume requires the same model, embedding SHA, layout, supervision, decoder,
tiles, and training identity.

## Inspect and summarize results

The summary dry-run is the result auditor and writes nothing:

```bash
python proc/seis_ssl_cluster/summarize_parihaka_channel_ssl_hmm.py \
  --config "$EXP/05_channel_four_way.yaml" \
  --dry-run
```

It requires 60 metrics files, one checkpoint identity per model, four distinct
checkpoint SHA values, common decoder seed/state, common supervision for each
layout/size, and finite test Channel IoU. Once it passes, write the summary:

```bash
python proc/seis_ssl_cluster/summarize_parihaka_channel_ssl_hmm.py \
  --config "$EXP/05_channel_four_way.yaml"
```

The main outputs are `comparison.csv`, `summary.json`, and `summary.md`.
`comparison.csv` retains all four raw IoUs and the paired HMM gains. JSON and
Markdown aggregate MAE-HMM versus MAE and Barlow-Twins-HMM versus Barlow Twins
separately by size. Significance tests and plots are outside this experiment.
