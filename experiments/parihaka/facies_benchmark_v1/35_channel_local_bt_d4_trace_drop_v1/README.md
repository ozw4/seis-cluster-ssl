# Parihaka Channel Local BT D4 + trace-drop screening v1

This experiment starts from the existing `localBT100` checkpoint and trains one
new D4 + trace-drop Local BT25 condition. The flip-only BT25 control and HMM25
condition reuse their existing checkpoints, embeddings, and Channel metrics;
they are not retrained, re-extracted, copied, or rerun. The new model ID is
`local_barlow_twins_d4_trace_drop`.

The encoder, projector, 128 local pairs, Local Barlow Twins loss, optimizer,
and 15,625-step budget are identical to the existing flip-only BT25 condition.
Only the view augmentation changes, with this exact contract:

```yaml
augmentations:
  policy: xy_d4_trace_drop_v1
  reflection_probability: 0.5
  trace_drop_probability: 0.02
```

Screening uses only `validation.channel_iou` from the five `medium` layouts.
Test IoU is not used by the gate. If the gate fails, stop without running the
other sizes or exploring more augmentations. This is a survey-specific,
transductive evaluation because SSL pretraining saw the unlabelled Parihaka
amplitude volume; it is not a new-survey transfer evaluation.

## Environment

Run from the installed development checkout:

```bash
cd /workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export SUITE=experiments/parihaka/facies_benchmark_v1/21_ssl_hmm_continuation_v1
export EXP=experiments/parihaka/facies_benchmark_v1/35_channel_local_bt_d4_trace_drop_v1
export AUG_BT_CONFIG_ROOT="$SUITE/30_stage2/local_bt100/bt_continue_d4_trace_drop"
export CHANNEL_CONFIG="$EXP/02_channel_comparison.yaml"
export LAYOUT_CONFIG=experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml
export RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/ssl_hmm_four_way_v1/runs"
export REPORT_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/local_bt_d4_trace_drop_v1/summary"
export CUDA_VISIBLE_DEVICES=1
```

The Stage 1 `localBT100` source, the control/HMM artifacts, prepared Parihaka
data, and reviewed layout config must already exist.

## 1. Run targeted tests

```bash
pytest -q \
  tests/seis_ssl_cluster/test_barlow_twins_dataset.py \
  tests/seis_ssl_cluster/test_barlow_twins_config.py \
  tests/seis_ssl_cluster/test_barlow_twins_continuation.py \
  tests/seis_ssl_cluster/test_barlow_twins_training_contract.py \
  tests/seis_ssl_cluster/test_parihaka_local_barlow_d4_trace_drop_configs.py \
  tests/seis_ssl_cluster/test_parihaka_channel_local_bt_d4_trace_drop_configs.py \
  tests/seis_ssl_cluster/test_parihaka_channel_local_bt_d4_trace_drop_runbook.py
```

These tests cover the D4 geometry and pairing contract, trace-drop validity,
legacy compatibility, dataset selection by the training runner, one-step
checkpointing, strict resume, continuation trainability, config sources, and
this runbook without requiring live artifacts.

Run the embedding metadata regression separately:

```bash
pytest -q tests/seis_ssl_cluster/test_embedding_extractor.py
```

## 2. Run the one-step GPU feasibility check

Inspect the resolved job, then run exactly one step:

```bash
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$AUG_BT_CONFIG_ROOT/01_gpu_feasibility_1step.yaml" \
  --dry-run

python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$AUG_BT_CONFIG_ROOT/01_gpu_feasibility_1step.yaml"
```

The dry-run and checkpoint must identify the Stage 1 `localBT100/latest.pt`
source, `local_barlow_twins_3d`, the exact augmentation mapping above, 128 local
pairs, and global step 1. The targeted continuation test verifies that only the
top encoder block and projector are trainable. Check that the printed loss and
gradient norm are finite, then run this read-only checkpoint audit:

```bash
python - <<'PY'
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path

from seis_ssl_cluster.training import load_checkpoint

root = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
path = root / (
	'pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/'
	'stage2/local_bt100/bt_continue_d4_trace_drop/'
	'gpu_feasibility_1step/latest.pt'
)
payload = load_checkpoint(path, map_location='cpu')
config = payload.get('config')
metrics = payload.get('metrics')
if not isinstance(config, Mapping) or not isinstance(metrics, Mapping):
	raise TypeError('checkpoint config and metrics must be mappings')
expected_source = root / (
	'pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/'
	'stage1/local_barlow_twins_v1/full_100ep/latest.pt'
)
expected_augmentation = {
	'policy': 'xy_d4_trace_drop_v1',
	'reflection_probability': 0.5,
	'trace_drop_probability': 0.02,
}
if payload.get('global_step') != 1:
	raise ValueError('feasibility checkpoint global_step must be 1')
if config.get('augmentations') != expected_augmentation:
	raise ValueError('feasibility augmentation contract mismatch')
continuation = config.get('continuation')
barlow = config.get('barlow_twins')
if not isinstance(continuation, Mapping) or not isinstance(barlow, Mapping):
	raise TypeError('continuation and barlow_twins must be mappings')
if Path(str(continuation.get('init_checkpoint'))).resolve() != expected_source:
	raise ValueError('feasibility source is not Stage 1 localBT100 latest.pt')
if continuation.get('unfreeze_top_blocks') != 1:
	raise ValueError('exactly one top encoder block must be unfrozen')
if barlow.get('method') != 'local_barlow_twins_3d':
	raise ValueError('unexpected pretraining method')
if barlow.get('local_pairs_per_crop') != 128:
	raise ValueError('local pair count must be 128')
for key in ('training_loss', 'gradient_norm'):
	value = metrics.get(key)
	if not isinstance(value, int | float) or not math.isfinite(float(value)):
		raise ValueError(f'{key} must be finite')
print('one-step feasibility audit: passed')
PY
```

The stored resolved config is also the audit record that the policy was saved
in the checkpoint.

## 3. Train the full 25-epoch continuation

Start a fresh continuation without `--resume`:

```bash
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$AUG_BT_CONFIG_ROOT/02_full_25ep.yaml" \
  --dry-run

python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$AUG_BT_CONFIG_ROOT/02_full_25ep.yaml"
```

Only after an interruption, resume this run from its own checkpoint:

```bash
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$AUG_BT_CONFIG_ROOT/02_full_25ep.yaml" \
  --resume "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/stage2/local_bt100/bt_continue_d4_trace_drop/full_25ep/latest.pt"
```

At completion, audit `latest.pt`: `epoch == 25`, `global_step == 15625`, the
continuation source is Stage 1 `localBT100/latest.pt`, and the resolved policy
is `xy_d4_trace_drop_v1` with reflection `0.5` and trace drop `0.02`.

```bash
python - <<'PY'
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from seis_ssl_cluster.training import load_checkpoint

root = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
path = root / (
	'pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/'
	'stage2/local_bt100/bt_continue_d4_trace_drop/full_25ep/latest.pt'
)
payload = load_checkpoint(path, map_location='cpu')
config = payload.get('config')
if not isinstance(config, Mapping):
	raise TypeError('checkpoint config must be a mapping')
continuation = config.get('continuation')
if not isinstance(continuation, Mapping):
	raise TypeError('checkpoint continuation must be a mapping')
expected_source = root / (
	'pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/'
	'stage1/local_barlow_twins_v1/full_100ep/latest.pt'
)
expected_augmentation = {
	'policy': 'xy_d4_trace_drop_v1',
	'reflection_probability': 0.5,
	'trace_drop_probability': 0.02,
}
if payload.get('epoch') != 25 or payload.get('global_step') != 15_625:
	raise ValueError('full checkpoint must be epoch 25, global_step 15625')
if Path(str(continuation.get('init_checkpoint'))).resolve() != expected_source:
	raise ValueError('full continuation source mismatch')
if config.get('augmentations') != expected_augmentation:
	raise ValueError('full checkpoint augmentation contract mismatch')
print('full continuation audit: passed')
PY
```

## 4. Extract and audit the augmented embedding

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/01_extract_augmented_embeddings.yaml" \
  --dry-run

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/01_extract_augmented_embeddings.yaml"
```

The output must contain `parihaka.embeddings.npy`,
`parihaka.valid_tokens.npy`, and `parihaka.embedding_metadata.json`. Run the
public `inspect_embedding_sources()` audit below. It verifies all three model
sources in `02_channel_comparison.yaml`, then checks that the candidate
checkpoint path/SHA, objective, pair count, augmentation identity, embedding
shape/dtype, valid-token mask, and preprocessing agree with their contracts.

```bash
python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.parihaka.channel_decoder import (
	channel_decoder_config_from_mapping,
	inspect_embedding_sources,
)

exp = Path(os.environ['EXP'])
config = channel_decoder_config_from_mapping(
	load_config(exp / '02_channel_comparison.yaml')
)
geometry = inspect_embedding_sources(config)
candidate_id = 'local_barlow_twins_d4_trace_drop'
control_id = 'local_barlow_twins'
candidate = geometry.models[candidate_id]
control = geometry.models[control_id]
checkpoint = config.models[candidate_id].expected_checkpoint
if checkpoint is None or Path(str(candidate.metadata['checkpoint_path'])).resolve() != checkpoint.resolve():
	raise ValueError('candidate metadata checkpoint path mismatch')
if candidate.metadata['checkpoint_sha256'] != file_sha256(checkpoint):
	raise ValueError('candidate metadata checkpoint SHA-256 mismatch')
objective = candidate.metadata.get('pretraining_objective')
expected_augmentation = {
	'policy': 'xy_d4_trace_drop_v1',
	'reflection_probability': 0.5,
	'trace_drop_probability': 0.02,
}
if not isinstance(objective, dict):
	raise TypeError('candidate pretraining_objective must be a mapping')
if objective.get('method') != 'local_barlow_twins_3d':
	raise ValueError('candidate objective method mismatch')
if objective.get('local_pairs_per_crop') != 128:
	raise ValueError('candidate local pair count mismatch')
if objective.get('augmentations') != expected_augmentation:
	raise ValueError('candidate augmentation metadata mismatch')
candidate_embeddings = np.load(candidate.paths.embeddings, mmap_mode='r', allow_pickle=False)
control_embeddings = np.load(control.paths.embeddings, mmap_mode='r', allow_pickle=False)
if (candidate_embeddings.shape, candidate_embeddings.dtype) != (
	control_embeddings.shape,
	control_embeddings.dtype,
):
	raise ValueError('candidate/control embedding shape or dtype mismatch')
candidate_valid = np.load(candidate.paths.valid_tokens, mmap_mode='r', allow_pickle=False)
control_valid = np.load(control.paths.valid_tokens, mmap_mode='r', allow_pickle=False)
if not np.array_equal(candidate_valid, control_valid):
	raise ValueError('candidate/control valid-token mask mismatch')
if candidate.metadata.get('preprocessing') != control.metadata.get('preprocessing'):
	raise ValueError('candidate/control preprocessing mismatch')
print('embedding source audit: passed')
PY
```

## 5. Preflight only the five medium candidate jobs

```bash
for layout in \
  layout_000 \
  layout_001 \
  layout_002 \
  layout_003 \
  layout_004
do
  python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
    --config "$CHANNEL_CONFIG" \
    --model local_barlow_twins_d4_trace_drop \
    --layout "$layout" \
    --size medium \
    --layout-config "$LAYOUT_CONFIG" \
    --dry-run
done
```

Compare each dry-run with the existing flip-only job for the same layout and
size. `selected_token_xyz_sha256`, actual train voxel count, class weights,
split class counts, tile counts, decoder initial-state SHA, validation/test
definition, and decoder/train/tile settings must match. Only the embedding,
checkpoint, model ID, and output directory may differ.

## 6. Run only the five medium candidate jobs

```bash
for layout in \
  layout_000 \
  layout_001 \
  layout_002 \
  layout_003 \
  layout_004
do
  python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
    --config "$CHANNEL_CONFIG" \
    --model local_barlow_twins_d4_trace_drop \
    --layout "$layout" \
    --size medium \
    --layout-config "$LAYOUT_CONFIG"
done
```

Do not rerun the existing flip-only or HMM jobs. For a restartable variant,
skip a job only when its `metrics.json` exists. Resume an interrupted job only
from that same job's
`$RUNS_ROOT/model=local_barlow_twins_d4_trace_drop/layout=<layout>/size=medium/latest.pt`
using the runner's `--resume` option.

## 7. Screen on medium validation IoU

This read-only/report script compares the 10 medium metrics, validates paired
`benchmark_identity` parity after removing only model/source identity, reads
only the validation Channel IoU, prints every layout gain and paired summary,
and writes no held-out test value. The gate is:

```text
mean validation gain >= +0.01
losses <= 1
```

```bash
python - <<'PY'
from __future__ import annotations

import json
import math
import os
import statistics
from collections.abc import Mapping
from pathlib import Path

control = 'local_barlow_twins'
candidate = 'local_barlow_twins_d4_trace_drop'
layouts = tuple(f'layout_{index:03d}' for index in range(5))
runs_root = Path(os.environ['RUNS_ROOT'])
report_root = Path(os.environ['REPORT_ROOT'])


def read_metrics(model: str, layout: str) -> Mapping[str, object]:
	path = runs_root / f'model={model}' / f'layout={layout}' / 'size=medium' / 'metrics.json'
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		raise TypeError(f'{path}: metrics payload must be a mapping')
	if payload.get('model') != model or payload.get('layout_id') != layout:
		raise ValueError(f'{path}: model/layout identity mismatch')
	if payload.get('data_size') != 'medium':
		raise ValueError(f'{path}: data_size must be medium')
	return payload


def paired_identity(payload: Mapping[str, object]) -> dict[str, object]:
	identity = payload.get('benchmark_identity')
	if not isinstance(identity, Mapping):
		raise TypeError('benchmark_identity must be a mapping')
	embedding = identity.get('embedding')
	if not isinstance(embedding, Mapping):
		raise TypeError('benchmark_identity.embedding must be a mapping')
	return {
		**{
			key: value
			for key, value in identity.items()
			if key not in {'model', 'embedding'}
		},
		'embedding': {'common_metadata': embedding.get('common_metadata')},
	}


def validation_iou(payload: Mapping[str, object]) -> float:
	validation = payload.get('validation')
	if not isinstance(validation, Mapping):
		raise TypeError('validation metrics must be a mapping')
	value = validation.get('channel_iou')
	if not isinstance(value, int | float) or not math.isfinite(float(value)):
		raise ValueError('validation Channel IoU must be finite')
	return float(value)


layout_gains: dict[str, float] = {}
for layout in layouts:
	control_metrics = read_metrics(control, layout)
	candidate_metrics = read_metrics(candidate, layout)
	if paired_identity(control_metrics) != paired_identity(candidate_metrics):
		raise ValueError(f'{layout}: downstream benchmark identity mismatch')
	layout_gains[layout] = validation_iou(candidate_metrics) - validation_iou(control_metrics)

gains = list(layout_gains.values())
paired_mean = statistics.mean(gains)
paired_median = statistics.median(gains)
sample_std = statistics.stdev(gains)
wins = sum(value > 0.0 for value in gains)
ties = sum(value == 0.0 for value in gains)
losses = sum(value < 0.0 for value in gains)
threshold = 0.01
max_losses = 1
gate_passed = paired_mean >= threshold and losses <= max_losses
result = {
	'model_control': control,
	'model_candidate': candidate,
	'metric': 'validation.channel_iou',
	'data_size': 'medium',
	'layout_gains': layout_gains,
	'paired_mean': paired_mean,
	'paired_median': paired_median,
	'sample_standard_deviation': sample_std,
	'wins': wins,
	'ties': ties,
	'losses': losses,
	'gate_threshold': threshold,
	'gate_max_losses': max_losses,
	'gate_passed': gate_passed,
}
report_root.mkdir(parents=True, exist_ok=True)
(report_root / 'screening_validation.json').write_text(
	json.dumps(result, indent=2, sort_keys=True) + '\n',
	encoding='utf-8',
)
lines = [
	'# D4 + trace-drop validation screening',
	'',
	'| layout | validation gain |',
	'|---|---:|',
]
lines.extend(f'| {layout} | {gain:+.6f} |' for layout, gain in layout_gains.items())
lines.extend(
	[
		'',
		f'- Paired mean: {paired_mean:+.6f}',
		f'- Paired median: {paired_median:+.6f}',
		f'- Sample standard deviation: {sample_std:.6f}',
		f'- Wins/ties/losses: {wins}/{ties}/{losses}',
		f'- Gate: mean >= {threshold:.2f} and losses <= {max_losses}',
		f'- Gate passed: {gate_passed}',
	]
)
(report_root / 'screening_validation.md').write_text(
	'\n'.join(lines) + '\n', encoding='utf-8'
)
for layout, gain in layout_gains.items():
	print(f'{layout}: {gain:+.6f}')
print(f'mean: {paired_mean:+.6f}')
print(f'median: {paired_median:+.6f}')
print(f'sample std: {sample_std:.6f}')
print(f'wins/ties/losses: {wins}/{ties}/{losses}')
print(f'gate_passed: {gate_passed}')
PY
```

Inspect both `$REPORT_ROOT/screening_validation.json` and
`$REPORT_ROOT/screening_validation.md` before proceeding.

## 8. Only after a human confirms that the gate passed

Open `screening_validation.json` and confirm that `gate_passed` is `true`. If it
is `false`, stop the experiment here and do not start another augmentation
search. If it is `true`, run only the candidate's remaining 10 jobs:

```bash
for layout in \
  layout_000 \
  layout_001 \
  layout_002 \
  layout_003 \
  layout_004
do
  for size in small large
  do
    python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
      --config "$CHANNEL_CONFIG" \
      --model local_barlow_twins_d4_trace_drop \
      --layout "$layout" \
      --size "$size" \
      --layout-config "$LAYOUT_CONFIG"
  done
done
```

As above, a restartable loop may skip completed `metrics.json` files and may
resume only from the interrupted job's own `latest.pt`. Do not rerun either
existing model.

## 9. After gate passage, audit 45 metrics and write the descriptive report

The following report step first calls the public
`inspect_channel_model_results()` with exactly these three conditions:

- `local_barlow_twins`
- `local_barlow_twins_d4_trace_drop`
- `local_barlow_twins_hmm_k6`

That inspector requires and fully audits all `3 * 5 * 3 = 45` metrics before
any report is written. The report then uses held-out test Channel IoU for the
final descriptive comparison only; it does not revise the screening decision.
It writes `$REPORT_ROOT/comparison.csv`, `$REPORT_ROOT/summary.json`, and
`$REPORT_ROOT/summary.md`.

```bash
python - <<'PY'
from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.parihaka.channel_results import (
	channel_summary_config_from_mapping,
	inspect_channel_model_results,
)

model_ids = (
	'local_barlow_twins',
	'local_barlow_twins_d4_trace_drop',
	'local_barlow_twins_hmm_k6',
)
layouts = tuple(f'layout_{index:03d}' for index in range(5))
sizes = ('small', 'medium', 'large')
exp = Path(os.environ['EXP'])
report_root = Path(os.environ['REPORT_ROOT'])
config = channel_summary_config_from_mapping(
	load_config(exp / '02_channel_comparison.yaml')
)
jobs = inspect_channel_model_results(config, model_ids=model_ids)
if len(jobs) != 45:
	raise ValueError(f'expected 45 audited metrics; got {len(jobs)}')


def channel_iou(model: str, layout: str, size: str) -> float:
	return float(jobs[(model, layout, size)]['test']['channel_iou'])


rows: list[dict[str, object]] = []
for size in sizes:
	for layout in layouts:
		control = channel_iou(model_ids[0], layout, size)
		augmented = channel_iou(model_ids[1], layout, size)
		hmm = channel_iou(model_ids[2], layout, size)
		rows.append(
			{
				'data_size': size,
				'layout_id': layout,
				'flip_only_channel_iou': control,
				'augmented_channel_iou': augmented,
				'hmm_channel_iou': hmm,
				'augmentation_gain': augmented - control,
				'hmm_gain': hmm - control,
				'hmm_minus_augmented': hmm - augmented,
			}
		)


def paired_summary(values: dict[str, float]) -> dict[str, object]:
	deltas = list(values.values())
	return {
		'paired_mean': statistics.mean(deltas),
		'paired_median': statistics.median(deltas),
		'sample_standard_deviation': statistics.stdev(deltas),
		'wins': sum(value > 0.0 for value in deltas),
		'ties': sum(value == 0.0 for value in deltas),
		'losses': sum(value < 0.0 for value in deltas),
		'layout_deltas': values,
	}


comparison_names = ('augmentation_gain', 'hmm_gain', 'hmm_minus_augmented')
summary = {
	'primary_metric': 'test.channel_iou',
	'purpose': 'post-gate descriptive comparison',
	'complete_jobs': len(jobs),
	'models': list(model_ids),
	'comparisons': {
		name: {
			size: paired_summary(
				{
					str(row['layout_id']): float(row[name])
					for row in rows
					if row['data_size'] == size
				}
			)
			for size in sizes
		}
		for name in comparison_names
	},
}
report_root.mkdir(parents=True, exist_ok=True)
with (report_root / 'comparison.csv').open('w', encoding='utf-8', newline='') as file_obj:
	writer = csv.DictWriter(file_obj, fieldnames=tuple(rows[0]))
	writer.writeheader()
	writer.writerows(rows)
(report_root / 'summary.json').write_text(
	json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)
lines = [
	'# D4 + trace-drop post-gate descriptive comparison',
	'',
	'Primary metric: `test.channel_iou` (not used for screening).',
]
for name in comparison_names:
	lines.extend(
		[
			'',
			f'## {name}',
			'',
			'| size | paired mean | median | sample std | wins/ties/losses |',
			'|---|---:|---:|---:|---:|',
		]
	)
	for size in sizes:
		item = summary['comparisons'][name][size]
		lines.append(
			f'| {size} | {item["paired_mean"]:+.6f} | '
			f'{item["paired_median"]:+.6f} | '
			f'{item["sample_standard_deviation"]:.6f} | '
			f'{item["wins"]}/{item["ties"]}/{item["losses"]} |'
		)
	lines.extend(['', '| size | layout | delta |', '|---|---|---:|'])
	for size in sizes:
		for layout, delta in summary['comparisons'][name][size]['layout_deltas'].items():
			lines.append(f'| {size} | {layout} | {delta:+.6f} |')
(report_root / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('full audit: passed (45 metrics)')
for path in ('comparison.csv', 'summary.json', 'summary.md'):
	print(f'output: {report_root / path}')
PY
```

These files are descriptive outputs only. They must not be fed back into the
validation gate or used as pipeline inputs.
