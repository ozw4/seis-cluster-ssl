# Parihaka HMM transition balance Phase 1

This experiment changes only the K=6 HMM cost balance between remaining in the
same state and advancing one state. It compares the following four variants in
both the MAE and trace-drop-free Local Barlow Twins branches.

| variant | same_cost | advance_cost | Δ |
|---|---:|---:|---:|
| `advance_favored_m003` | 0.03 | 0.00 | -0.03 |
| `neutral` | 0.00 | 0.00 | 0.00 |
| `persist003` | 0.00 | 0.03 | +0.03 |
| `persist010` | 0.00 | 0.10 | +0.10 |

`advance_favored_m003` is the existing H0 condition. Its checkpoints,
embeddings, and Channel metrics are reused in place and are not generated
again. The existing MAE and Local BT controls and both existing H0 models reuse
their checkpoints, embeddings, and metrics; do not retrain, re-extract, or
rerun the decoder for any of those four models. The fixed scientific contract
is:

```text
K = 6
iterations = 10
anchors = 0.25 / 0.25
expected boundaries = off
max_jump = 1
reverse forbidden
boundary_alpha = 0.0
distillation_weight = 0.2
Stage 2 budget = 25 epochs / 15,625 steps
downstream = frozen embedding decoder
screening size = medium
layouts = layout_000 ... layout_004
selection metric = validation.channel_iou
```

This is a survey-specific, transductive screening because SSL pretraining saw
the unlabelled Parihaka amplitude volume. Phase 1 computes and saves only
validation Channel IoU for new candidates: the decoder runner does not
run test inference or save test metrics in `--validation-only` mode. Phase 1
does not run `small` or `large`. Validation screening is a human-reviewed comparison,
not an automatic gate that starts a later experiment. Existing metrics files
may contain historical test results, but neither the screening code nor the
human review may inspect them; Phase 1 does not read test IoU.

## 1. Environment

Run from an installed development checkout. The artifact root and smoke root
must be absolute. Reuse the reviewed layout file directly.

```bash
set -euo pipefail
cd /workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export EXP=experiments/parihaka/facies_benchmark_v1/36_channel_hmm_transition_balance_v1
export TARGET_CONFIGS="$EXP/10_hmm_targets"
export STAGE2_CONFIGS="$EXP/20_stage2"
export EMBEDDING_CONFIGS="$EXP/30_embeddings"
export CHANNEL_CONFIG="$EXP/40_channel_transition_balance.yaml"
export FINAL_CHANNEL_CONFIG="$EXP/41_channel_transition_balance_final.yaml"
export LAYOUT_CONFIG=experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml
export CLUSTER_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/clustering/parihaka/facies_benchmark_v1/hmm_transition_balance_v1"
export PSEUDO_TARGET_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/parihaka/facies_benchmark_v1/hmm_transition_balance_v1"
export STAGE1_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/stage1"
export STAGE2_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/hmm_transition_balance_v1"
export EXISTING_RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/ssl_hmm_four_way_v1/runs"
export VALIDATION_RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/hmm_transition_balance_v1/validation_runs"
export FINAL_RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/hmm_transition_balance_v1/final_runs"
export REPORT_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/hmm_transition_balance_v1/summary"
export SMOKE_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/hmm_transition_balance_v1_smoke_1step"
export CUDA_VISIBLE_DEVICES=1
```

Do not copy, symlink, or rsync artifacts. Do not delete artifact directories.
In particular, do not use `cp`, `ln -s`, `rsync`, or `rm -rf` to make a
reused source appear under a new namespace.

## 2. Targeted tests

Run the three config contracts, the executable screening tests, and this
runbook contract before execution:

```bash
set -euo pipefail
pytest -q \
  tests/seis_ssl_cluster/test_parihaka_hmm_transition_balance_target_configs.py \
  tests/seis_ssl_cluster/test_parihaka_hmm_transition_balance_training_configs.py \
  tests/seis_ssl_cluster/test_parihaka_channel_hmm_transition_balance_configs.py \
  tests/seis_ssl_cluster/test_parihaka_channel_hmm_transition_balance_summary.py \
  tests/seis_ssl_cluster/test_parihaka_channel_hmm_transition_balance_runbook.py
```

These tests do not require live checkpoints, embeddings, pseudo-targets, or
Channel metrics.

## 3. Build the six new clustering results and pseudo-targets

The six clustering configs are explicit. Dry-run all six before executing any
of them.

```bash
set -euo pipefail
CLUSTER_CONFIGS=(
  "$TARGET_CONFIGS/mae100/neutral/01_cluster_hmm_k6.yaml"
  "$TARGET_CONFIGS/mae100/persist003/01_cluster_hmm_k6.yaml"
  "$TARGET_CONFIGS/mae100/persist010/01_cluster_hmm_k6.yaml"
  "$TARGET_CONFIGS/local_bt100/neutral/01_cluster_hmm_k6.yaml"
  "$TARGET_CONFIGS/local_bt100/persist003/01_cluster_hmm_k6.yaml"
  "$TARGET_CONFIGS/local_bt100/persist010/01_cluster_hmm_k6.yaml"
)
for config in "${CLUSTER_CONFIGS[@]}"; do
  python proc/seis_ssl_cluster/cluster_embeddings.py \
    --config "$config" \
    --dry-run
done
for config in "${CLUSTER_CONFIGS[@]}"; do
  python proc/seis_ssl_cluster/cluster_embeddings.py --config "$config"
done
```

Run the six reviewed export scripts. Each invokes
`proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py` with K=6,
confidence 1.0, boundary alpha 0.0, boundary tau 1.0, and schema version 2.

```bash
set -euo pipefail
EXPORT_SCRIPTS=(
  "$TARGET_CONFIGS/mae100/neutral/02_export_pseudo_targets.sh"
  "$TARGET_CONFIGS/mae100/persist003/02_export_pseudo_targets.sh"
  "$TARGET_CONFIGS/mae100/persist010/02_export_pseudo_targets.sh"
  "$TARGET_CONFIGS/local_bt100/neutral/02_export_pseudo_targets.sh"
  "$TARGET_CONFIGS/local_bt100/persist003/02_export_pseudo_targets.sh"
  "$TARGET_CONFIGS/local_bt100/persist010/02_export_pseudo_targets.sh"
)
for script in "${EXPORT_SCRIPTS[@]}"; do
  bash "$script"
done
```

The existing H0 clustering result and pseudo-targets are not regenerated.

## 4. Audit clustering and pseudo-target metadata

This read-only block checks the source/variant paths, K, transition and path
prior identity, and positive valid-token counts. Empty clusters and boundary
counts are diagnostics: they are printed without an arbitrary pass/fail
threshold.

```bash
set -euo pipefail
python - <<'PY'
from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

from seis_ssl_cluster.stratigraphy import (
	discover_pseudo_target_inputs,
	load_pseudo_target_metadata,
)

artifact_root = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
cluster_root = Path(os.environ['CLUSTER_ROOT'])
pseudo_target_root = Path(os.environ['PSEUDO_TARGET_ROOT'])
embedding_roots = {
	'mae100': artifact_root
	/ 'embeddings/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1'
	/ 'hmm_targets/mae100/overlap_x64',
	'local_bt100': artifact_root
	/ 'embeddings/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1'
	/ 'hmm_targets/local_bt100/overlap_x64',
}
transition_settings = {
	'neutral': {'same_cost': 0.0, 'advance_cost': 0.0},
	'persist003': {'same_cost': 0.0, 'advance_cost': 0.03},
	'persist010': {'same_cost': 0.0, 'advance_cost': 0.10},
}
conditions = tuple(
	(source, variant)
	for source in ('mae100', 'local_bt100')
	for variant in ('neutral', 'persist003', 'persist010')
)
if len(conditions) != 6:
	raise AssertionError('expected six new source/variant conditions')


def mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


for source, variant in conditions:
	cluster_dir = cluster_root / source / variant
	target_dir = pseudo_target_root / source / variant
	model_path = cluster_dir / 'models/k6/clustering_metadata.json'
	metadata = mapping(
		json.loads(model_path.read_text(encoding='utf-8')),
		f'{source}/{variant} clustering metadata',
	)
	if metadata.get('method') != 'stratigraphic_hmm_kmeans':
		raise ValueError(f'{source}/{variant}: clustering method mismatch')
	if metadata.get('k') != 6 or metadata.get('k_values') != [6]:
		raise ValueError(f'{source}/{variant}: K must be 6')

	embedding_inputs = metadata.get('embedding_inputs')
	if not isinstance(embedding_inputs, list) or not embedding_inputs:
		raise ValueError(f'{source}/{variant}: embedding inputs are missing')
	expected_embedding_root = embedding_roots[source].resolve()
	for index, value in enumerate(embedding_inputs):
		item = mapping(value, f'{source}/{variant} embedding input {index}')
		for key in ('embeddings_path', 'valid_tokens_path', 'metadata_path'):
			path = Path(str(item.get(key)))
			if path.resolve().parent != expected_embedding_root:
				raise ValueError(
					f'{source}/{variant}: {key} is outside its source root'
				)

	hmm = mapping(
		metadata.get('stratigraphic_hmm'),
		f'{source}/{variant} stratigraphic_hmm',
	)
	if hmm.get('iterations') != 10:
		raise ValueError(f'{source}/{variant}: iterations must be 10')
	transition = mapping(hmm.get('transition'), f'{source}/{variant} transition')
	expected_transition = {
		**transition_settings[variant],
		'jump_cost': 1.0,
		'reverse_cost': 1_000_000.0,
		'forbid_reverse': True,
		'max_jump': 1,
	}
	if dict(transition) != expected_transition:
		raise ValueError(f'{source}/{variant}: transition settings mismatch')
	path_prior = mapping(hmm.get('path_prior'), f'{source}/{variant} path prior')
	if path_prior.get('enabled') is not True:
		raise ValueError(f'{source}/{variant}: path prior must be enabled')
	if mapping(
		path_prior.get('initial_state'),
		f'{source}/{variant} initial anchor',
	) != {'mode': 'shallow_anchor', 'weight': 0.25}:
		raise ValueError(f'{source}/{variant}: initial anchor mismatch')
	if mapping(
		path_prior.get('terminal_state'),
		f'{source}/{variant} terminal anchor',
	) != {'mode': 'deep_anchor', 'weight': 0.25}:
		raise ValueError(f'{source}/{variant}: terminal anchor mismatch')
	expected_boundaries = mapping(
		path_prior.get('expected_boundaries'),
		f'{source}/{variant} expected boundaries',
	)
	if expected_boundaries.get('enabled') is not False:
		raise ValueError(f'{source}/{variant}: expected boundaries must be off')

	iterations = hmm.get('iteration_summaries')
	if not isinstance(iterations, list) or len(iterations) != 10:
		raise ValueError(f'{source}/{variant}: expected 10 iteration summaries')
	final_iteration = mapping(
		iterations[-1],
		f'{source}/{variant} final iteration',
	)
	if final_iteration.get('iteration') != 10:
		raise ValueError(f'{source}/{variant}: final iteration must be 10')
	cluster_counts = mapping(
		metadata.get('cluster_counts'),
		f'{source}/{variant} cluster counts',
	)
	if len(cluster_counts) != 6:
		raise ValueError(f'{source}/{variant}: cluster counts must cover K=6')
	empty_clusters = final_iteration.get('empty_clusters')
	if not isinstance(empty_clusters, list):
		raise TypeError(f'{source}/{variant}: empty_clusters must be a list')
	total_shift = final_iteration.get('total_center_shift_l2')
	if (
		not isinstance(total_shift, int | float)
		or isinstance(total_shift, bool)
		or not math.isfinite(float(total_shift))
	):
		raise ValueError(f'{source}/{variant}: final center shift must be finite')
	ordered = mapping(
		metadata.get('ordered_diagnostics'),
		f'{source}/{variant} ordered diagnostics',
	)
	aggregate = mapping(
		ordered.get('aggregate'),
		f'{source}/{variant} aggregate diagnostics',
	)
	mean_boundaries = aggregate.get('mean_boundaries_per_valid_trace')
	if (
		not isinstance(mean_boundaries, int | float)
		or isinstance(mean_boundaries, bool)
		or not math.isfinite(float(mean_boundaries))
	):
		raise ValueError(f'{source}/{variant}: boundary diagnostic must be finite')

	surveys = metadata.get('surveys')
	if not isinstance(surveys, list) or not surveys:
		raise ValueError(f'{source}/{variant}: clustering surveys are missing')
	expected_surveys = {
		str(mapping(row, f'{source}/{variant} survey').get('survey_id'))
		for row in surveys
	}
	inputs = discover_pseudo_target_inputs(target_dir, k=6)
	if not inputs:
		raise ValueError(f'{source}/{variant}: pseudo-target inputs are missing')
	if {item.survey_id for item in inputs} != expected_surveys:
		raise ValueError(f'{source}/{variant}: pseudo-target survey set mismatch')
	for item in inputs:
		if item.metadata_path.parent.resolve() != (target_dir / 'k6').resolve():
			raise ValueError(f'{source}/{variant}: pseudo-target root mismatch')
		target_metadata = mapping(
			load_pseudo_target_metadata(item),
			f'{source}/{variant}/{item.survey_id} pseudo-target metadata',
		)
		if target_metadata.get('k') != 6:
			raise ValueError(f'{source}/{variant}: pseudo-target K must be 6')
		valid_count = target_metadata.get('valid_token_count')
		if (
			not isinstance(valid_count, int)
			or isinstance(valid_count, bool)
			or valid_count <= 0
		):
			raise ValueError(f'{source}/{variant}: no valid pseudo-target token')
		provenance = mapping(
			target_metadata.get('source'),
			f'{source}/{variant} pseudo-target source',
		)
		source_dir = Path(str(provenance.get('source_clustering_output_dir')))
		if source_dir.resolve() != cluster_dir.resolve():
			raise ValueError(f'{source}/{variant}: clustering source path mismatch')
		source_label = Path(str(provenance.get('source_label_path')))
		if source_label.resolve().parent != (cluster_dir / 'labels/k6').resolve():
			raise ValueError(f'{source}/{variant}: source label path mismatch')
		boundary = mapping(
			provenance.get('boundary_weighting'),
			f'{source}/{variant} boundary weighting',
		)
		if boundary.get('alpha') != 0.0:
			raise ValueError(f'{source}/{variant}: boundary alpha must be 0.0')

	print(
		source,
		variant,
		{
			'cluster_counts': dict(cluster_counts),
			'empty_clusters': empty_clusters,
			'mean_boundaries_per_valid_trace': float(mean_boundaries),
			'total_center_shift_l2': float(total_shift),
		},
	)
print('clustering and pseudo-target audit: passed')
PY
```

## 5. Dry-run Stage 2 and optionally smoke one step

Dry-run all six full-budget configs. There are no dedicated feasibility YAMLs.

```bash
set -euo pipefail
TRAINING_CONFIGS=(
  "$STAGE2_CONFIGS/mae100/neutral/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/mae100/persist003/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/mae100/persist010/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/local_bt100/neutral/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/local_bt100/persist003/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/local_bt100/persist010/01_full_25ep.yaml"
)
for config in "${TRAINING_CONFIGS[@]}"; do
  python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
    --config "$config" \
    --dry-run
done
```

If a live one-step check is needed, use the existing CLI overrides and an
isolated smoke root. Do not add smoke YAMLs.

```bash
set -euo pipefail
SMOKE_SPECS=(
  "$STAGE2_CONFIGS/mae100/neutral/01_full_25ep.yaml|$SMOKE_ROOT/mae100/neutral"
  "$STAGE2_CONFIGS/mae100/persist003/01_full_25ep.yaml|$SMOKE_ROOT/mae100/persist003"
  "$STAGE2_CONFIGS/mae100/persist010/01_full_25ep.yaml|$SMOKE_ROOT/mae100/persist010"
  "$STAGE2_CONFIGS/local_bt100/neutral/01_full_25ep.yaml|$SMOKE_ROOT/local_bt100/neutral"
  "$STAGE2_CONFIGS/local_bt100/persist003/01_full_25ep.yaml|$SMOKE_ROOT/local_bt100/persist003"
  "$STAGE2_CONFIGS/local_bt100/persist010/01_full_25ep.yaml|$SMOKE_ROOT/local_bt100/persist010"
)
for spec in "${SMOKE_SPECS[@]}"; do
  IFS='|' read -r config output_root <<< "$spec"
  python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
    --config "$config" \
    --max-steps 1 \
    --output-root "$output_root"
done
```

## 6. Run the six full 25-epoch trainings

Start each full run without `--resume`:

```bash
set -euo pipefail
TRAINING_CONFIGS=(
  "$STAGE2_CONFIGS/mae100/neutral/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/mae100/persist003/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/mae100/persist010/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/local_bt100/neutral/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/local_bt100/persist003/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/local_bt100/persist010/01_full_25ep.yaml"
)
for config in "${TRAINING_CONFIGS[@]}"; do
  python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$config"
done
```

After an interruption, resume only from that same full run's `latest.pt`. Do
not use Stage 1, control, H0, `best.pt`, another variant, or a smoke checkpoint.

```bash
set -euo pipefail
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$STAGE2_CONFIGS/mae100/neutral/01_full_25ep.yaml" \
  --resume "$STAGE2_ROOT/mae100/neutral/full_25ep/latest.pt"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$STAGE2_CONFIGS/mae100/persist003/01_full_25ep.yaml" \
  --resume "$STAGE2_ROOT/mae100/persist003/full_25ep/latest.pt"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$STAGE2_CONFIGS/mae100/persist010/01_full_25ep.yaml" \
  --resume "$STAGE2_ROOT/mae100/persist010/full_25ep/latest.pt"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$STAGE2_CONFIGS/local_bt100/neutral/01_full_25ep.yaml" \
  --resume "$STAGE2_ROOT/local_bt100/neutral/full_25ep/latest.pt"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$STAGE2_CONFIGS/local_bt100/persist003/01_full_25ep.yaml" \
  --resume "$STAGE2_ROOT/local_bt100/persist003/full_25ep/latest.pt"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$STAGE2_CONFIGS/local_bt100/persist010/01_full_25ep.yaml" \
  --resume "$STAGE2_ROOT/local_bt100/persist010/full_25ep/latest.pt"
```

## 7. Audit the six full checkpoints

This read-only block verifies the fixed budget, teacher/student source,
pseudo-target root, K=6, distillation weight 0.2, top block 1, and FP32
training.

```bash
set -euo pipefail
python - <<'PY'
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from seis_ssl_cluster.training import load_checkpoint

stage1_root = Path(os.environ['STAGE1_ROOT'])
stage2_root = Path(os.environ['STAGE2_ROOT'])
pseudo_target_root = Path(os.environ['PSEUDO_TARGET_ROOT'])
source_checkpoints = {
	'mae100': stage1_root / 'mae/full_100ep/latest.pt',
	'local_bt100': stage1_root / 'local_barlow_twins_v1/full_100ep/latest.pt',
}
conditions = tuple(
	(source, variant)
	for source in ('mae100', 'local_bt100')
	for variant in ('neutral', 'persist003', 'persist010')
)
if len(conditions) != 6:
	raise AssertionError('expected six Stage 2 checkpoints')


def mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


for source, variant in conditions:
	checkpoint = stage2_root / source / variant / 'full_25ep/latest.pt'
	payload = mapping(
		load_checkpoint(checkpoint, map_location='cpu'),
		f'{source}/{variant} checkpoint',
	)
	if payload.get('epoch') != 25 or payload.get('global_step') != 15_625:
		raise ValueError(f'{source}/{variant}: fixed training budget mismatch')
	if payload.get('amp_enabled') is not False:
		raise ValueError(f'{source}/{variant}: training must be FP32')
	stratigraphy = mapping(
		payload.get('stratigraphy_config'),
		f'{source}/{variant} stratigraphy config',
	)
	head = mapping(stratigraphy.get('head'), f'{source}/{variant} head')
	pseudo_targets = mapping(
		stratigraphy.get('pseudo_targets'),
		f'{source}/{variant} pseudo targets',
	)
	student = mapping(stratigraphy.get('student'), f'{source}/{variant} student')
	teacher = mapping(stratigraphy.get('teacher'), f'{source}/{variant} teacher')
	loss = mapping(stratigraphy.get('loss'), f'{source}/{variant} loss')
	source_checkpoint = source_checkpoints[source].resolve()
	if head.get('num_prototypes') != 6 or pseudo_targets.get('k') != 6:
		raise ValueError(f'{source}/{variant}: K must be 6')
	if student.get('unfreeze_top_blocks') != 1:
		raise ValueError(f'{source}/{variant}: top block count must be 1')
	if loss.get('distillation_weight') != 0.2:
		raise ValueError(f'{source}/{variant}: distillation weight must be 0.2')
	if Path(str(teacher.get('checkpoint'))).resolve() != source_checkpoint:
		raise ValueError(f'{source}/{variant}: teacher source mismatch')
	if Path(str(student.get('init_checkpoint'))).resolve() != source_checkpoint:
		raise ValueError(f'{source}/{variant}: student source mismatch')
	expected_targets = (pseudo_target_root / source / variant).resolve()
	if Path(str(pseudo_targets.get('input_dir'))).resolve() != expected_targets:
		raise ValueError(f'{source}/{variant}: pseudo-target root mismatch')
	print(
		source,
		variant,
		{
			'epoch': payload['epoch'],
			'global_step': payload['global_step'],
			'teacher_student_source': str(source_checkpoint),
			'pseudo_target_root': str(expected_targets),
			'k': pseudo_targets['k'],
			'distillation_weight': loss['distillation_weight'],
			'unfreeze_top_blocks': student['unfreeze_top_blocks'],
			'amp_enabled': payload['amp_enabled'],
		},
	)
print('Stage 2 checkpoint audit: passed')
PY
```

## 8. Extract the six new embedding volumes

Dry-run every extraction config, then execute those same six configs.

```bash
set -euo pipefail
EXTRACTION_CONFIGS=(
  "$EMBEDDING_CONFIGS/01_extract_mae_hmm_k6_neutral.yaml"
  "$EMBEDDING_CONFIGS/02_extract_mae_hmm_k6_persist003.yaml"
  "$EMBEDDING_CONFIGS/03_extract_mae_hmm_k6_persist010.yaml"
  "$EMBEDDING_CONFIGS/04_extract_local_barlow_twins_hmm_k6_neutral.yaml"
  "$EMBEDDING_CONFIGS/05_extract_local_barlow_twins_hmm_k6_persist003.yaml"
  "$EMBEDDING_CONFIGS/06_extract_local_barlow_twins_hmm_k6_persist010.yaml"
)
for config in "${EXTRACTION_CONFIGS[@]}"; do
  python proc/seis_ssl_cluster/extract_embeddings.py \
    --config "$config" \
    --dry-run
done
for config in "${EXTRACTION_CONFIGS[@]}"; do
  python proc/seis_ssl_cluster/extract_embeddings.py --config "$config"
done
```

Do not re-extract `mae`, `mae_hmm_k6`, `local_barlow_twins`, or
`local_barlow_twins_hmm_k6`.

## 9. Audit all ten embedding sources

The public `inspect_embedding_sources()` API validates the configured
checkpoint path and live SHA-256, distinct checkpoint identities, shared
geometry and preprocessing, embedding shape/dtype, and valid-token mask parity.
This read-only block also fixes the expected ten-model order.

```bash
set -euo pipefail
python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.parihaka.channel_decoder import (
	channel_decoder_config_from_mapping,
	inspect_embedding_sources,
)

expected_models = (
	'mae',
	'mae_hmm_k6',
	'mae_hmm_k6_neutral',
	'mae_hmm_k6_persist003',
	'mae_hmm_k6_persist010',
	'local_barlow_twins',
	'local_barlow_twins_hmm_k6',
	'local_barlow_twins_hmm_k6_neutral',
	'local_barlow_twins_hmm_k6_persist003',
	'local_barlow_twins_hmm_k6_persist010',
)
config = channel_decoder_config_from_mapping(
	load_config(Path(os.environ['CHANNEL_CONFIG']))
)
geometry = inspect_embedding_sources(config)
if tuple(geometry.models) != expected_models:
	raise ValueError(f'unexpected model order: {tuple(geometry.models)}')

signatures: set[tuple[tuple[int, ...], str]] = set()
reference_valid: np.ndarray | None = None
for model_id, item in geometry.models.items():
	embeddings = np.load(item.paths.embeddings, mmap_mode='r', allow_pickle=False)
	valid = np.load(item.paths.valid_tokens, mmap_mode='r', allow_pickle=False)
	signatures.add((tuple(embeddings.shape), str(embeddings.dtype)))
	if reference_valid is None:
		reference_valid = valid
	elif not np.array_equal(reference_valid, valid):
		raise ValueError(f'{model_id}: valid-token mask mismatch')
	source = config.models[model_id]
	checkpoint = source.expected_checkpoint
	if checkpoint is None:
		raise ValueError(f'{model_id}: configured checkpoint is missing')
	if Path(str(item.model_source['checkpoint_path'])).resolve() != checkpoint.resolve():
		raise ValueError(f'{model_id}: checkpoint identity mismatch')
	print(
		model_id,
		item.model_source['checkpoint_sha256'],
		tuple(embeddings.shape),
		str(embeddings.dtype),
	)
if len(signatures) != 1:
	raise ValueError('embedding shape/dtype mismatch')
print(
	'geometry',
	{
		'volume_shape_xyz': geometry.volume_shape_xyz,
		'token_grid_shape_xyz': geometry.token_grid_shape_xyz,
		'patch_size_xyz': geometry.patch_size_xyz,
		'embedding_shape': geometry.embedding_shape,
		'embedding_dim': geometry.embedding_dim,
	},
)
print('ten-model embedding source audit: passed')
PY
```

## 10. Dry-run the 30 new medium decoder jobs

Only the six new candidates are execution targets. Dry-run their five reviewed
layouts at `medium` size.

```bash
set -euo pipefail
CANDIDATE_MODELS=(
  mae_hmm_k6_neutral
  mae_hmm_k6_persist003
  mae_hmm_k6_persist010
  local_barlow_twins_hmm_k6_neutral
  local_barlow_twins_hmm_k6_persist003
  local_barlow_twins_hmm_k6_persist010
)
LAYOUTS=(
  layout_000
  layout_001
  layout_002
  layout_003
  layout_004
)
for model in "${CANDIDATE_MODELS[@]}"; do
  for layout in "${LAYOUTS[@]}"; do
    python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
      --config "$CHANNEL_CONFIG" \
      --model "$model" \
      --layout "$layout" \
      --size medium \
      --layout-config "$LAYOUT_CONFIG" \
      --validation-only \
      --dry-run
  done
done
```

## 11. Run the 30 new medium decoder jobs

Run the same six models and five layouts without `--dry-run`. Keep
`--validation-only`; this is what prevents test evaluation rather than merely
hiding test metrics from the report:

```bash
set -euo pipefail
CANDIDATE_MODELS=(
  mae_hmm_k6_neutral
  mae_hmm_k6_persist003
  mae_hmm_k6_persist010
  local_barlow_twins_hmm_k6_neutral
  local_barlow_twins_hmm_k6_persist003
  local_barlow_twins_hmm_k6_persist010
)
LAYOUTS=(
  layout_000
  layout_001
  layout_002
  layout_003
  layout_004
)
for model in "${CANDIDATE_MODELS[@]}"; do
  for layout in "${LAYOUTS[@]}"; do
    job_dir="$VALIDATION_RUNS_ROOT/model=$model/layout=$layout/size=medium"
    if [[ -f "$job_dir/metrics.json" ]]; then
      echo "skip completed Channel job: model=$model layout=$layout"
      continue
    fi
    if [[ -f "$job_dir/latest.pt" ]]; then
      echo "incomplete Channel job requires explicit resume: $job_dir" >&2
      exit 1
    fi
    python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
      --config "$CHANNEL_CONFIG" \
      --model "$model" \
      --layout "$layout" \
      --size medium \
      --layout-config "$LAYOUT_CONFIG" \
      --validation-only
  done
done
```

Do not rerun decoder jobs for the MAE control, Local BT control, or either
existing H0 model. The loop skips a job only when its own `metrics.json`
exists. If it finds `latest.pt` without `metrics.json`, it prints that job and
stops instead of silently starting over or automatically resuming. Resume that
one job explicitly with the same model, layout, size, layout config, and
`--validation-only`, adding `--resume
"$VALIDATION_RUNS_ROOT/model=<model>/layout=<layout>/size=medium/latest.pt"`;
then rerun the loop. The runner accepts only that same job's `latest.pt`. Every
new candidate `metrics.json` must say `"evaluation_mode": "validation_only"`
and must not contain a top-level `test` field.

## 12. Write the validation screening summary

The experiment-local report script reads exactly 50 `medium` metrics: two
controls, two existing H0 models, and six new candidates over five layouts. It
checks paired downstream identity after removing only model/source-specific
embedding identity, reads only `validation.channel_iou`, and writes the JSON
and Markdown screening summaries. Its ranking and rejection behavior are
covered by executable tests with known metrics fixtures.

```bash
set -euo pipefail
python "$EXP/scripts/summarize_validation.py" \
  --existing-runs-root "$EXISTING_RUNS_ROOT" \
  --validation-runs-root "$VALIDATION_RUNS_ROOT" \
  --report-root "$REPORT_ROOT"
```

Inspect both `$REPORT_ROOT/screening_validation.json` and
`$REPORT_ROOT/screening_validation.md`. Phase 1 ends here. A human uses these
validation-only summaries to choose any Phase 2 condition. Reviewers must not
open the historical top-level `test` fields under `$EXISTING_RUNS_ROOT`. This
report does not start Phase 2. Do not proceed to other supervision sizes or a
held-out report.

## 13. Final test protocol after all validation phases

Do not run this section during Phase 1. After the complete multi-phase study
has frozen one final model and layout without reference to any test result, set
those IDs explicitly. The final config uses `$FINAL_RUNS_ROOT`, which is
disjoint from both historical runs and Phase 1 validation-only runs.

```bash
set -euo pipefail
export FINAL_MODEL=mae_hmm_k6_persist003
export FINAL_LAYOUT=layout_002
python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
  --config "$FINAL_CHANNEL_CONFIG" \
  --model "$FINAL_MODEL" \
  --layout "$FINAL_LAYOUT" \
  --size medium \
  --layout-config "$LAYOUT_CONFIG" \
  --dry-run
```

Record the frozen choice before continuing. Then run exactly that job in the
normal mode, deliberately omitting `--validation-only`. This retrains the
deterministic decoder in the clean final namespace, selects `best.pt` by
validation Channel IoU, and evaluates the test dataset once.

```bash
set -euo pipefail
python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
  --config "$FINAL_CHANNEL_CONFIG" \
  --model "$FINAL_MODEL" \
  --layout "$FINAL_LAYOUT" \
  --size medium \
  --layout-config "$LAYOUT_CONFIG"
```

The resulting `$FINAL_RUNS_ROOT/model=$FINAL_MODEL/layout=$FINAL_LAYOUT/size=medium/metrics.json`
must say `"evaluation_mode": "validation_and_test"` and contain the single
authorized top-level `test` result. Do not copy a validation-only checkpoint
into this namespace and do not run alternative models or layouts after seeing
that result.
