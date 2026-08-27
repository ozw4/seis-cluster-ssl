# Parihaka HMM boundary-weight Phase

This experiment keeps the selected H0 HMM transition setting fixed and changes
only the prototype-loss weight near HMM boundaries. The same comparison is run
for the MAE and trace-drop-free Local Barlow Twins branches.

| variant | boundary_alpha | boundary_tau | 扱い |
|---|---:|---:|---|
| `alpha000_tau1` | 0.0 | 1.0 | 既存H0を再利用 |
| `alpha050_tau1` | 0.5 | 1.0 | 新規 |
| `alpha100_tau1` | 1.0 | 1.0 | 新規 |

The fixed scientific contract is:

```text
K = 6
same_cost = 0.03
advance_cost = 0.0
jump_cost = 1.0
anchors = 0.25 / 0.25
expected boundaries = off
max_jump = 1
reverse forbidden
boundary_tau = 1.0
distillation_weight = 0.2
Stage 2 = 25 epochs / 15,625 steps
downstream = frozen embedding decoder
screening = medium, five layouts
layouts = layout_000 ... layout_004
metric = validation.channel_iou
```

This Phase is survey-specific, transductive, and validation-only because SSL
pretraining saw the unlabelled Parihaka amplitude volume. New candidates do not
run test inference and do not save test metrics. The existing `alpha000_tau1`
pseudo-targets, Stage 2 checkpoints, embeddings, and Channel metrics are reused
in place. The MAE and Local BT controls are also reused. None of those existing
artifacts or decoder jobs are regenerated.

## 1. Environment

Run from the development checkout. Artifact and smoke roots must be absolute.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export EXP=experiments/parihaka/facies_benchmark_v1/37_channel_hmm_boundary_weight_v1
export TARGET_CONFIGS="$EXP/10_pseudo_targets"
export STAGE2_CONFIGS="$EXP/20_stage2"
export EMBEDDING_CONFIGS="$EXP/30_embeddings"
export CHANNEL_CONFIG="$EXP/40_channel_boundary_weight.yaml"
export LAYOUT_CONFIG=experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml
export H0_CLUSTER_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/clustering/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/hmm_targets"
export EXISTING_PSEUDO_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1"
export PSEUDO_TARGET_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/parihaka/facies_benchmark_v1/hmm_boundary_weight_v1"
export STAGE1_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/stage1"
export STAGE2_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/hmm_boundary_weight_v1"
export EXISTING_RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/ssl_hmm_four_way_v1/runs"
export VALIDATION_RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/hmm_boundary_weight_v1/validation_runs"
export REPORT_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/hmm_boundary_weight_v1/summary"
export SMOKE_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/hmm_boundary_weight_v1_smoke_1step"
```

Set `CUDA_VISIBLE_DEVICES` in the operator environment when an explicit device
selection is needed.

Do not copy, symlink, synchronize, or delete artifacts to make a reused source
appear under this Phase's namespace.

## 2. Targeted tests

These tests require no live experiment artifacts.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
pytest -q \
  tests/seis_ssl_cluster/test_stratigraphy_boundary_weights.py \
  tests/seis_ssl_cluster/test_stratigraphy_export.py \
  tests/seis_ssl_cluster/test_strat_pseudo_dataset.py \
  tests/seis_ssl_cluster/test_parihaka_channel_decoder.py \
  tests/seis_ssl_cluster/test_parihaka_hmm_boundary_weight_configs.py \
  tests/seis_ssl_cluster/test_parihaka_channel_hmm_boundary_weight_configs.py \
  tests/seis_ssl_cluster/test_parihaka_channel_hmm_boundary_weight_summary.py \
  tests/seis_ssl_cluster/test_parihaka_channel_hmm_boundary_weight_runbook.py
```

## 3. Export the four new pseudo-target sets

No new HMM fit or decode is performed. Each exporter reads the selected H0
labels. Dry-run all four scripts before executing the same four scripts.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export TARGET_CONFIGS=experiments/parihaka/facies_benchmark_v1/37_channel_hmm_boundary_weight_v1/10_pseudo_targets
EXPORT_SCRIPTS=(
  "$TARGET_CONFIGS/mae100/alpha050_tau1/01_export_pseudo_targets.sh"
  "$TARGET_CONFIGS/mae100/alpha100_tau1/01_export_pseudo_targets.sh"
  "$TARGET_CONFIGS/local_bt100/alpha050_tau1/01_export_pseudo_targets.sh"
  "$TARGET_CONFIGS/local_bt100/alpha100_tau1/01_export_pseudo_targets.sh"
)
for script in "${EXPORT_SCRIPTS[@]}"; do
  bash "$script" --dry-run
done
for script in "${EXPORT_SCRIPTS[@]}"; do
  bash "$script"
done
```

Do not export `alpha000_tau1`; it is the existing H0 pseudo-target artifact.

## 4. Audit all six pseudo-target sets

This read-only audit compares the existing and two new conditions within each
source. It prints boundary and downweighted-token counts without introducing a
new acceptance threshold.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export H0_CLUSTER_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/clustering/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/hmm_targets"
export EXISTING_PSEUDO_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1"
export PSEUDO_TARGET_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/parihaka/facies_benchmark_v1/hmm_boundary_weight_v1"
python - <<'PY'
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from seis_ssl_cluster.stratigraphy import (
	boundary_weight_tokens,
	discover_pseudo_target_inputs,
	load_pseudo_target_arrays,
	load_pseudo_target_metadata,
)

h0_cluster_root = Path(os.environ['H0_CLUSTER_ROOT'])
existing_root = Path(os.environ['EXISTING_PSEUDO_ROOT'])
new_root = Path(os.environ['PSEUDO_TARGET_ROOT'])
variants = ('alpha000_tau1', 'alpha050_tau1', 'alpha100_tau1')
expected_alpha = {
	'alpha000_tau1': 0.0,
	'alpha050_tau1': 0.5,
	'alpha100_tau1': 1.0,
}


def mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


for source in ('mae100', 'local_bt100'):
	roots = {
		'alpha000_tau1': existing_root / source,
		'alpha050_tau1': new_root / source / 'alpha050_tau1',
		'alpha100_tau1': new_root / source / 'alpha100_tau1',
	}
	inputs = {
		variant: {
			item.survey_id: item
			for item in discover_pseudo_target_inputs(root, k=6)
		}
		for variant, root in roots.items()
	}
	survey_sets = {variant: set(items) for variant, items in inputs.items()}
	if len({frozenset(surveys) for surveys in survey_sets.values()}) != 1:
		raise ValueError(f'{source}: pseudo-target survey sets differ')
	expected_cluster = (h0_cluster_root / source / 'k6').resolve()
	saw_alpha050_downweight = False
	saw_alpha100_stronger = False
	for survey_id in sorted(survey_sets['alpha000_tau1']):
		arrays = {
			variant: load_pseudo_target_arrays(inputs[variant][survey_id])
			for variant in variants
		}
		metadata = {
			variant: load_pseudo_target_metadata(inputs[variant][survey_id])
			for variant in variants
		}
		base = arrays['alpha000_tau1']
		for variant in variants:
			item = arrays[variant]
			if not np.array_equal(item.labels, base.labels):
				raise ValueError(f'{source}/{survey_id}: labels differ')
			if not np.array_equal(item.confidence, base.confidence):
				raise ValueError(f'{source}/{survey_id}: confidence differs')
			if not np.array_equal(item.valid_tokens, base.valid_tokens):
				raise ValueError(f'{source}/{survey_id}: valid-token mask differs')
			meta = metadata[variant]
			if meta.get('k') != 6:
				raise ValueError(f'{source}/{survey_id}/{variant}: K must be 6')
			provenance = mapping(meta.get('source'), 'pseudo-target source')
			actual_cluster = Path(
				str(provenance.get('source_clustering_output_dir'))
			).resolve()
			if actual_cluster != expected_cluster:
				raise ValueError(
					f'{source}/{survey_id}/{variant}: H0 source mismatch'
				)
			boundary = mapping(
				provenance.get('boundary_weighting'),
				'boundary weighting',
			)
			if boundary.get('alpha') != expected_alpha[variant]:
				raise ValueError(
					f'{source}/{survey_id}/{variant}: boundary alpha mismatch'
				)
			if boundary.get('tau') != 1.0:
				raise ValueError(
					f'{source}/{survey_id}/{variant}: boundary tau must be 1.0'
				)
			valid = np.asarray(item.valid_tokens, dtype=np.bool_)
			weight = np.asarray(item.boundary_weight)
			if np.any(weight[~valid] != 0.0):
				raise ValueError(
					f'{source}/{survey_id}/{variant}: invalid weight must be zero'
				)
			if np.any((weight[valid] < 0.0) | (weight[valid] > 1.0)):
				raise ValueError(
					f'{source}/{survey_id}/{variant}: valid weight out of range'
				)
			expected_weight = boundary_weight_tokens(
				item.labels,
				item.valid_tokens,
				alpha=expected_alpha[variant],
				tau=1.0,
			)
			if not np.array_equal(weight, expected_weight):
				raise ValueError(
					f'{source}/{survey_id}/{variant}: boundary weight mismatch'
				)
		valid = np.asarray(base.valid_tokens, dtype=np.bool_)
		w000 = np.asarray(arrays['alpha000_tau1'].boundary_weight)
		w050 = np.asarray(arrays['alpha050_tau1'].boundary_weight)
		w100 = np.asarray(arrays['alpha100_tau1'].boundary_weight)
		if np.any(w000[valid] != 1.0):
			raise ValueError(f'{source}/{survey_id}: alpha000 valid weight != 1')
		if np.any(w100 > w050) or np.any(w050 > w000):
			raise ValueError(f'{source}/{survey_id}: weight ordering mismatch')
		saw_alpha050_downweight |= bool(np.any(w050[valid] < 1.0))
		saw_alpha100_stronger |= bool(np.any(w100[valid] < w050[valid]))
		adjacent_valid = valid[..., :-1] & valid[..., 1:]
		boundaries = adjacent_valid & (
			base.labels[..., :-1] != base.labels[..., 1:]
		)
		print(
			source,
			survey_id,
			{
				'boundary_count': int(np.count_nonzero(boundaries)),
				'alpha050_downweighted': int(np.count_nonzero(w050[valid] < 1.0)),
				'alpha100_downweighted': int(np.count_nonzero(w100[valid] < 1.0)),
			},
		)
	if not saw_alpha050_downweight:
		raise ValueError(f'{source}: alpha050 must downweight a valid token')
	if not saw_alpha100_stronger:
		raise ValueError(f'{source}: alpha100 must be stronger than alpha050')
print('pseudo-target audit: passed')
PY
```

## 5. Dry-run the four Stage 2 trainings

There are no dedicated feasibility YAMLs.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export STAGE2_CONFIGS=experiments/parihaka/facies_benchmark_v1/37_channel_hmm_boundary_weight_v1/20_stage2
TRAINING_CONFIGS=(
  "$STAGE2_CONFIGS/mae100/alpha050_tau1/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/mae100/alpha100_tau1/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/local_bt100/alpha050_tau1/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/local_bt100/alpha100_tau1/01_full_25ep.yaml"
)
for config in "${TRAINING_CONFIGS[@]}"; do
  python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
    --config "$config" \
    --dry-run
done
```

## 6. Optionally smoke one step

Only when a live feasibility check is needed, use CLI overrides with an
isolated output root. This is not part of the four full runs.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export STAGE2_CONFIGS=experiments/parihaka/facies_benchmark_v1/37_channel_hmm_boundary_weight_v1/20_stage2
export SMOKE_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/hmm_boundary_weight_v1_smoke_1step"
SMOKE_SPECS=(
  "$STAGE2_CONFIGS/mae100/alpha050_tau1/01_full_25ep.yaml|$SMOKE_ROOT/mae100/alpha050_tau1"
  "$STAGE2_CONFIGS/mae100/alpha100_tau1/01_full_25ep.yaml|$SMOKE_ROOT/mae100/alpha100_tau1"
  "$STAGE2_CONFIGS/local_bt100/alpha050_tau1/01_full_25ep.yaml|$SMOKE_ROOT/local_bt100/alpha050_tau1"
  "$STAGE2_CONFIGS/local_bt100/alpha100_tau1/01_full_25ep.yaml|$SMOKE_ROOT/local_bt100/alpha100_tau1"
)
for spec in "${SMOKE_SPECS[@]}"; do
  IFS='|' read -r config output_root <<< "$spec"
  python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
    --config "$config" \
    --max-steps 1 \
    --output-root "$output_root"
done
```

## 7. Run the four full 25-epoch trainings

Start each run from its configured Stage 1 teacher/student checkpoint, without
`--resume`. Never continue a new condition from the existing H0 Stage 2 run.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export STAGE2_CONFIGS=experiments/parihaka/facies_benchmark_v1/37_channel_hmm_boundary_weight_v1/20_stage2
TRAINING_CONFIGS=(
  "$STAGE2_CONFIGS/mae100/alpha050_tau1/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/mae100/alpha100_tau1/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/local_bt100/alpha050_tau1/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/local_bt100/alpha100_tau1/01_full_25ep.yaml"
)
for config in "${TRAINING_CONFIGS[@]}"; do
  python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$config"
done
```

After an interruption, resume only the same condition from its own
`full_25ep/latest.pt`; never resume automatically or from another condition.

## 8. Audit the four Stage 2 checkpoints

This read-only block verifies budget, FP32 mode, source identity, target
identity, K, distillation weight, and encoder unfreezing.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export STAGE1_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/stage1"
export STAGE2_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/hmm_boundary_weight_v1"
export PSEUDO_TARGET_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/parihaka/facies_benchmark_v1/hmm_boundary_weight_v1"
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
	for variant in ('alpha050_tau1', 'alpha100_tau1')
)


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
		raise ValueError(f'{source}/{variant}: training budget mismatch')
	if payload.get('amp_enabled') is not False:
		raise ValueError(f'{source}/{variant}: amp_enabled must be false')
	training_state = mapping(
		payload.get('training_state'),
		f'{source}/{variant} training state',
	)
	if (
		training_state.get('checkpoint_kind') != 'epoch'
		or training_state.get('batch_index') is not None
	):
		raise ValueError(f'{source}/{variant}: full epoch checkpoint required')
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
	train = mapping(stratigraphy.get('train'), f'{source}/{variant} train')
	expected_train = {
		'batch_size': 16,
		'samples_per_epoch': 10_000,
		'epochs': 25,
		'max_steps': None,
		'amp': False,
	}
	if any(train.get(key) != value for key, value in expected_train.items()):
		raise ValueError(f'{source}/{variant}: resolved train budget mismatch')
	trainability = mapping(
		payload.get('trainability_summary'),
		f'{source}/{variant} trainability summary',
	)
	trainable_names = trainability.get('trainable_names')
	if (
		not isinstance(trainable_names, list)
		or not trainable_names
		or any(
			not isinstance(name, str)
			or not name.startswith('encoder.layers.7.')
			for name in trainable_names
		)
	):
		raise ValueError(f'{source}/{variant}: top encoder block mismatch')
	for key in ('trainable_parameter_count', 'frozen_parameter_count'):
		value = trainability.get(key)
		if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
			raise ValueError(f'{source}/{variant}: invalid {key}')
	expected_source = source_checkpoints[source].resolve()
	if Path(str(teacher.get('checkpoint'))).resolve() != expected_source:
		raise ValueError(f'{source}/{variant}: teacher source mismatch')
	if Path(str(student.get('init_checkpoint'))).resolve() != expected_source:
		raise ValueError(f'{source}/{variant}: student source mismatch')
	if student.get('unfreeze_top_blocks') != 1:
		raise ValueError(f'{source}/{variant}: unfreeze_top_blocks must be 1')
	if pseudo_targets.get('k') != 6 or head.get('num_prototypes') != 6:
		raise ValueError(f'{source}/{variant}: K must be 6')
	if loss.get('distillation_weight') != 0.2:
		raise ValueError(f'{source}/{variant}: distillation weight mismatch')
	expected_targets = (pseudo_target_root / source / variant).resolve()
	if Path(str(pseudo_targets.get('input_dir'))).resolve() != expected_targets:
		raise ValueError(f'{source}/{variant}: pseudo-target root mismatch')
	print(
		source,
		variant,
		{
			'epoch': payload['epoch'],
			'global_step': payload['global_step'],
			'amp_enabled': payload['amp_enabled'],
			'checkpoint_kind': training_state['checkpoint_kind'],
			'teacher_student': str(expected_source),
			'pseudo_targets': str(expected_targets),
			'k': pseudo_targets['k'],
			'distillation_weight': loss['distillation_weight'],
			'unfreeze_top_blocks': student['unfreeze_top_blocks'],
			'trainable_parameter_count': trainability[
				'trainable_parameter_count'
			],
		},
	)
print('Stage 2 checkpoint audit: passed')
PY
```

## 9. Extract the four new embedding volumes

Dry-run all four extraction configs, then execute those same configs.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export EMBEDDING_CONFIGS=experiments/parihaka/facies_benchmark_v1/37_channel_hmm_boundary_weight_v1/30_embeddings
EXTRACTION_CONFIGS=(
  "$EMBEDDING_CONFIGS/01_extract_mae_hmm_k6_boundary_alpha050_tau1.yaml"
  "$EMBEDDING_CONFIGS/02_extract_mae_hmm_k6_boundary_alpha100_tau1.yaml"
  "$EMBEDDING_CONFIGS/03_extract_local_barlow_twins_hmm_k6_boundary_alpha050_tau1.yaml"
  "$EMBEDDING_CONFIGS/04_extract_local_barlow_twins_hmm_k6_boundary_alpha100_tau1.yaml"
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

Do not re-extract either control or either existing `alpha000_tau1` model.

## 10. Audit all eight embedding sources

The public `inspect_embedding_sources()` audit checks checkpoint identity,
volume and token geometry, preprocessing, and source metadata. The checks below
also require a single embedding shape/dtype signature and an identical
valid-token mask across all eight models.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export CHANNEL_CONFIG=experiments/parihaka/facies_benchmark_v1/37_channel_hmm_boundary_weight_v1/40_channel_boundary_weight.yaml
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
	'mae_hmm_k6_boundary_alpha050_tau1',
	'mae_hmm_k6_boundary_alpha100_tau1',
	'local_barlow_twins',
	'local_barlow_twins_hmm_k6',
	'local_barlow_twins_hmm_k6_boundary_alpha050_tau1',
	'local_barlow_twins_hmm_k6_boundary_alpha100_tau1',
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
		raise ValueError(f'{model_id}: expected checkpoint is missing')
	actual_checkpoint = Path(str(item.model_source['checkpoint_path'])).resolve()
	if actual_checkpoint != checkpoint.resolve():
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
print('eight-model embedding source audit: passed')
PY
```

## 11. Dry-run the 20 new medium decoder jobs

Only the four new models are execution targets. Each uses the five reviewed
layouts at `medium` size, for 20 validation-only jobs.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export CHANNEL_CONFIG=experiments/parihaka/facies_benchmark_v1/37_channel_hmm_boundary_weight_v1/40_channel_boundary_weight.yaml
export LAYOUT_CONFIG=experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml
CANDIDATE_MODELS=(
  mae_hmm_k6_boundary_alpha050_tau1
  mae_hmm_k6_boundary_alpha100_tau1
  local_barlow_twins_hmm_k6_boundary_alpha050_tau1
  local_barlow_twins_hmm_k6_boundary_alpha100_tau1
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

## 12. Run the 20 new medium decoder jobs

The loop skips completed jobs, but a partial job stops the Phase for explicit
operator review. It never resumes automatically.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export CHANNEL_CONFIG=experiments/parihaka/facies_benchmark_v1/37_channel_hmm_boundary_weight_v1/40_channel_boundary_weight.yaml
export LAYOUT_CONFIG=experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml
export VALIDATION_RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/hmm_boundary_weight_v1/validation_runs"
CANDIDATE_MODELS=(
  mae_hmm_k6_boundary_alpha050_tau1
  mae_hmm_k6_boundary_alpha100_tau1
  local_barlow_twins_hmm_k6_boundary_alpha050_tau1
  local_barlow_twins_hmm_k6_boundary_alpha100_tau1
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

Do not rerun the MAE control, Local BT control, or either existing
`alpha000_tau1` model. Every new `metrics.json` must have
`evaluation_mode = validation_only` and no top-level `test` field.

## 13. Write the validation screening summary

The experiment-local script reads the existing four models from the historical
runs root and the four new candidates from the validation-only root: eight
models over five layouts, or exactly 40 metrics. It reads only
`validation.channel_iou`. It verifies global downstream identity across all 40
jobs, full paired identity across models within each layout, and model-source
identity across the five layouts.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export EXP=experiments/parihaka/facies_benchmark_v1/37_channel_hmm_boundary_weight_v1
export EXISTING_RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/ssl_hmm_four_way_v1/runs"
export VALIDATION_RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/hmm_boundary_weight_v1/validation_runs"
export REPORT_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/hmm_boundary_weight_v1/summary"
python "$EXP/scripts/summarize_validation.py" \
  --existing-runs-root "$EXISTING_RUNS_ROOT" \
  --validation-runs-root "$VALIDATION_RUNS_ROOT" \
  --report-root "$REPORT_ROOT"
```

Inspect `screening_validation.json` and `screening_validation.md` under the
report root. Do not inspect historical test fields.

## 14. End the Phase

Stop after human review of the validation summary:

- If `alpha000_tau1` is selected, end the boundary-weight Phase and proceed to
  a separately prepared distillation-weight Phase.
- If `alpha050_tau1` or `alpha100_tau1` is selected, freeze that alpha; prepare
  any tau comparison as a separate Phase.
- This runbook neither implements nor executes a tau sweep.
- Do not read test results, run new-candidate test inference, or proceed to
  `small` or `large` screening.
- Do not start another Phase automatically.
