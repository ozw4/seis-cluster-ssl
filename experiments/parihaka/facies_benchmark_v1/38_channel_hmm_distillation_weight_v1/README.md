# Parihaka HMM distillation-weight validation screening

This Phase compares the fixed four-value Stage 2 distillation-weight grid on
the MAE and trace-drop-free Local Barlow Twins branches.

| variant | distillation_weight | 扱い |
|---|---:|---|
| `distill005` | 0.05 | 新規 |
| `distill010` | 0.10 | 新規 |
| `distill020` | 0.20 | 既存H0を再利用 |
| `distill040` | 0.40 | 新規 |

The scientific contract is fixed as follows:

```text
K = 6
same_cost = 0.03
advance_cost = 0.0
jump_cost = 1.0
anchors = 0.25 / 0.25
expected boundaries = off
max_jump = 1
reverse forbidden
boundary_alpha = 0.0
boundary_tau = 1.0
Stage 2 = 25 epochs / 15,625 steps
student.unfreeze_top_blocks = 1
downstream = frozen embedding decoder
screening = medium, five layouts
layouts = layout_000 ... layout_004
metric = validation.channel_iou
```

This is survey-specific, transductive, validation-only screening. New
candidates do not run test inference or store test metrics. Historical test
fields, if present, are not read. The existing H0 pseudo-targets, the existing
`distill020` checkpoints and embeddings, and the MAE and Local BT controls are
reused in place. There is no clustering, pseudo-target export, or pseudo-target
copy step in this Phase.

## 1. Environment

Run from a repository checkout with the existing Parihaka artifacts available.
Every later shell block repeats its own environment setup and can be run in a
fresh shell.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export EXP=experiments/parihaka/facies_benchmark_v1/38_channel_hmm_distillation_weight_v1
test -f "$EXP/40_channel_distillation_weight.yaml"
test -f "$EXP/scripts/summarize_validation.py"
python --version
```

## 2. Targeted tests

Run the production decoder coverage and all four Phase-specific contracts, then
the focused distillation/unfreezing coverage.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
pytest -q \
  tests/seis_ssl_cluster/test_parihaka_hmm_distillation_weight_configs.py \
  tests/seis_ssl_cluster/test_parihaka_channel_hmm_distillation_weight_configs.py \
  tests/seis_ssl_cluster/test_parihaka_channel_hmm_distillation_weight_summary.py \
  tests/seis_ssl_cluster/test_parihaka_channel_hmm_distillation_weight_runbook.py \
  tests/seis_ssl_cluster/test_parihaka_channel_decoder.py
pytest -q \
  tests/seis_ssl_cluster/test_strat_hmm_pretraining_head_only.py \
  -k "distillation or unfreeze"
```

## 3. Dry-run the six Stage 2 trainings

Only the six new positive-weight conditions are training targets. There is no
`distill020` config and no dedicated feasibility or smoke YAML.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export EXP=experiments/parihaka/facies_benchmark_v1/38_channel_hmm_distillation_weight_v1
export STAGE2_CONFIGS="$EXP/20_stage2"
TRAINING_CONFIGS=(
  "$STAGE2_CONFIGS/mae100/distill005/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/mae100/distill010/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/mae100/distill040/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/local_bt100/distill005/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/local_bt100/distill010/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/local_bt100/distill040/01_full_25ep.yaml"
)
for config in "${TRAINING_CONFIGS[@]}"; do
  python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
    --config "$config" \
    --dry-run
done
```

## 4. Optionally smoke one step

Run this only when a live feasibility check is needed. CLI overrides isolate
the one-step outputs from all full runs; no smoke YAML is added.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export EXP=experiments/parihaka/facies_benchmark_v1/38_channel_hmm_distillation_weight_v1
export STAGE2_CONFIGS="$EXP/20_stage2"
export SMOKE_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/hmm_distillation_weight_v1_smoke_1step"
SMOKE_SPECS=(
  "$STAGE2_CONFIGS/mae100/distill005/01_full_25ep.yaml|$SMOKE_ROOT/mae100/distill005"
  "$STAGE2_CONFIGS/mae100/distill010/01_full_25ep.yaml|$SMOKE_ROOT/mae100/distill010"
  "$STAGE2_CONFIGS/mae100/distill040/01_full_25ep.yaml|$SMOKE_ROOT/mae100/distill040"
  "$STAGE2_CONFIGS/local_bt100/distill005/01_full_25ep.yaml|$SMOKE_ROOT/local_bt100/distill005"
  "$STAGE2_CONFIGS/local_bt100/distill010/01_full_25ep.yaml|$SMOKE_ROOT/local_bt100/distill010"
  "$STAGE2_CONFIGS/local_bt100/distill040/01_full_25ep.yaml|$SMOKE_ROOT/local_bt100/distill040"
)
for spec in "${SMOKE_SPECS[@]}"; do
  IFS='|' read -r config output_root <<< "$spec"
  python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
    --config "$config" \
    --max-steps 1 \
    --output-root "$output_root"
done
```

## 5. Run or explicitly resume the six full 25-epoch trainings

Start each new condition from its configured source-specific Stage 1
teacher/student checkpoint. The normal full-run loop never resumes.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export EXP=experiments/parihaka/facies_benchmark_v1/38_channel_hmm_distillation_weight_v1
export STAGE2_CONFIGS="$EXP/20_stage2"
TRAINING_CONFIGS=(
  "$STAGE2_CONFIGS/mae100/distill005/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/mae100/distill010/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/mae100/distill040/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/local_bt100/distill005/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/local_bt100/distill010/01_full_25ep.yaml"
  "$STAGE2_CONFIGS/local_bt100/distill040/01_full_25ep.yaml"
)
for config in "${TRAINING_CONFIGS[@]}"; do
  python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$config"
done
```

After an interruption, set `RESUME_CONDITION` to exactly one listed condition.
The explicit maps bind that config to its own `full_25ep/latest.pt`; the block
rejects unknown conditions and never discovers or resumes a run automatically.
Do not use Stage 1, H0, a control, a smoke checkpoint, another variant, or
`best.pt` as the resume source.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export EXP=experiments/parihaka/facies_benchmark_v1/38_channel_hmm_distillation_weight_v1
export STAGE2_CONFIGS="$EXP/20_stage2"
export STAGE2_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/hmm_distillation_weight_v1"
resume_condition="${RESUME_CONDITION:?Set RESUME_CONDITION to one explicit source/variant pair}"
declare -A RESUME_CONFIGS=(
  [mae100/distill005]="$STAGE2_CONFIGS/mae100/distill005/01_full_25ep.yaml"
  [mae100/distill010]="$STAGE2_CONFIGS/mae100/distill010/01_full_25ep.yaml"
  [mae100/distill040]="$STAGE2_CONFIGS/mae100/distill040/01_full_25ep.yaml"
  [local_bt100/distill005]="$STAGE2_CONFIGS/local_bt100/distill005/01_full_25ep.yaml"
  [local_bt100/distill010]="$STAGE2_CONFIGS/local_bt100/distill010/01_full_25ep.yaml"
  [local_bt100/distill040]="$STAGE2_CONFIGS/local_bt100/distill040/01_full_25ep.yaml"
)
declare -A RESUME_CHECKPOINTS=(
  [mae100/distill005]="$STAGE2_ROOT/mae100/distill005/full_25ep/latest.pt"
  [mae100/distill010]="$STAGE2_ROOT/mae100/distill010/full_25ep/latest.pt"
  [mae100/distill040]="$STAGE2_ROOT/mae100/distill040/full_25ep/latest.pt"
  [local_bt100/distill005]="$STAGE2_ROOT/local_bt100/distill005/full_25ep/latest.pt"
  [local_bt100/distill010]="$STAGE2_ROOT/local_bt100/distill010/full_25ep/latest.pt"
  [local_bt100/distill040]="$STAGE2_ROOT/local_bt100/distill040/full_25ep/latest.pt"
)
resume_config="${RESUME_CONFIGS[$resume_condition]-}"
resume_checkpoint="${RESUME_CHECKPOINTS[$resume_condition]-}"
if [[ -z "$resume_config" || -z "$resume_checkpoint" ]]; then
  echo "unknown RESUME_CONDITION: $resume_condition" >&2
  exit 2
fi
if [[ ! -f "$resume_checkpoint" ]]; then
  echo "resume checkpoint is missing: $resume_checkpoint" >&2
  exit 1
fi
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$resume_config" \
  --resume "$resume_checkpoint"
```

## 6. Audit the six Stage 2 checkpoints

This read-only audit verifies the completed epoch checkpoint, fixed budget,
FP32 mode, source-specific teacher/student identity, reused H0 pseudo-targets,
K, variant weight, and the single unfrozen top block.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export STAGE1_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/stage1"
export STAGE2_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/hmm_distillation_weight_v1"
export PSEUDO_TARGET_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1"
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
variant_weights = {
	'distill005': 0.05,
	'distill010': 0.10,
	'distill040': 0.40,
}
conditions = tuple(
	(source, variant)
	for source in ('mae100', 'local_bt100')
	for variant in variant_weights
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
	expected_source = source_checkpoints[source].resolve()
	if Path(str(teacher.get('checkpoint'))).resolve() != expected_source:
		raise ValueError(f'{source}/{variant}: teacher source mismatch')
	if Path(str(student.get('init_checkpoint'))).resolve() != expected_source:
		raise ValueError(f'{source}/{variant}: student source mismatch')
	expected_targets = (pseudo_target_root / source).resolve()
	if Path(str(pseudo_targets.get('input_dir'))).resolve() != expected_targets:
		raise ValueError(f'{source}/{variant}: pseudo-target root mismatch')
	if pseudo_targets.get('k') != 6 or head.get('num_prototypes') != 6:
		raise ValueError(f'{source}/{variant}: K must be 6')
	if loss.get('distillation_weight') != variant_weights[variant]:
		raise ValueError(f'{source}/{variant}: distillation weight mismatch')
	if student.get('unfreeze_top_blocks') != 1:
		raise ValueError(f'{source}/{variant}: unfreeze_top_blocks must be 1')
	print(
		source,
		variant,
		{
			'epoch': payload['epoch'],
			'global_step': payload['global_step'],
			'checkpoint_kind': training_state['checkpoint_kind'],
			'teacher_student': str(expected_source),
			'pseudo_targets': str(expected_targets),
			'k': pseudo_targets['k'],
			'distillation_weight': loss['distillation_weight'],
			'unfreeze_top_blocks': student['unfreeze_top_blocks'],
			'amp_enabled': payload['amp_enabled'],
		},
	)
print('Stage 2 checkpoint audit: passed')
PY
```

## 7. Extract the six new embedding volumes

Dry-run all six extraction configs, then execute those same configs. Do not
re-extract the MAE control, Local BT control, or either existing `distill020`
HMM embedding.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export EXP=experiments/parihaka/facies_benchmark_v1/38_channel_hmm_distillation_weight_v1
export EMBEDDING_CONFIGS="$EXP/30_embeddings"
EXTRACTION_CONFIGS=(
  "$EMBEDDING_CONFIGS/01_extract_mae_hmm_k6_distill005.yaml"
  "$EMBEDDING_CONFIGS/02_extract_mae_hmm_k6_distill010.yaml"
  "$EMBEDDING_CONFIGS/03_extract_mae_hmm_k6_distill040.yaml"
  "$EMBEDDING_CONFIGS/04_extract_local_barlow_twins_hmm_k6_distill005.yaml"
  "$EMBEDDING_CONFIGS/05_extract_local_barlow_twins_hmm_k6_distill010.yaml"
  "$EMBEDDING_CONFIGS/06_extract_local_barlow_twins_hmm_k6_distill040.yaml"
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

## 8. Audit all ten embedding sources

The public `inspect_embedding_sources()` audit verifies checkpoint identity,
volume and token geometry, preprocessing, and source metadata. This read-only
block additionally requires one shape/dtype signature and an identical
valid-token mask across the two controls, two existing H0 models, and six new
models.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export CHANNEL_CONFIG=experiments/parihaka/facies_benchmark_v1/38_channel_hmm_distillation_weight_v1/40_channel_distillation_weight.yaml
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
	'mae_hmm_k6_distill005',
	'mae_hmm_k6_distill010',
	'mae_hmm_k6_distill040',
	'local_barlow_twins',
	'local_barlow_twins_hmm_k6',
	'local_barlow_twins_hmm_k6_distill005',
	'local_barlow_twins_hmm_k6_distill010',
	'local_barlow_twins_hmm_k6_distill040',
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
print('ten-model embedding source audit: passed')
PY
```

## 9. Dry-run the 30 new medium decoder jobs

Only the six new candidates are execution targets: six models by five layouts
at `medium`, for 30 validation-only jobs.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export CHANNEL_CONFIG=experiments/parihaka/facies_benchmark_v1/38_channel_hmm_distillation_weight_v1/40_channel_distillation_weight.yaml
export LAYOUT_CONFIG=experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml
CANDIDATE_MODELS=(
  mae_hmm_k6_distill005
  mae_hmm_k6_distill010
  mae_hmm_k6_distill040
  local_barlow_twins_hmm_k6_distill005
  local_barlow_twins_hmm_k6_distill010
  local_barlow_twins_hmm_k6_distill040
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

## 10. Run the 30 new medium decoder jobs

The loop skips a completed job. A job with `latest.pt` but no `metrics.json` is
printed and stops the Phase for explicit operator handling; it is never resumed
automatically. Existing controls and `distill020` models are not rerun.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export CHANNEL_CONFIG=experiments/parihaka/facies_benchmark_v1/38_channel_hmm_distillation_weight_v1/40_channel_distillation_weight.yaml
export LAYOUT_CONFIG=experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml
export VALIDATION_RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/hmm_distillation_weight_v1/validation_runs"
CANDIDATE_MODELS=(
  mae_hmm_k6_distill005
  mae_hmm_k6_distill010
  mae_hmm_k6_distill040
  local_barlow_twins_hmm_k6_distill005
  local_barlow_twins_hmm_k6_distill010
  local_barlow_twins_hmm_k6_distill040
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

Every new `metrics.json` must have `evaluation_mode = validation_only` and no
top-level `test` field.

## 11. Write the validation screening summary

The experiment-local script reads the existing four models from the historical
runs root and the six new models from the validation-only root: ten models by
five layouts, or exactly 50 metrics. It reads only
`validation.channel_iou`.

```bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/seis_ssl_cluster}"
export EXP=experiments/parihaka/facies_benchmark_v1/38_channel_hmm_distillation_weight_v1
export EXISTING_RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/ssl_hmm_four_way_v1/runs"
export VALIDATION_RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/hmm_distillation_weight_v1/validation_runs"
export REPORT_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/hmm_distillation_weight_v1/summary"
python "$EXP/scripts/summarize_validation.py" \
  --existing-runs-root "$EXISTING_RUNS_ROOT" \
  --validation-runs-root "$VALIDATION_RUNS_ROOT" \
  --report-root "$REPORT_ROOT"
```

Review `screening_validation.json`, `screening_validation.md`, and the layout
gain table manually. Do not inspect historical test fields.

## 12. End the Phase

- Freeze one of the four weights using validation results and human review.
- If the recommendation is the grid edge 0.05 or 0.40, do not automatically
  add or run another weight.
- Do not read test results or start final-test inference.
- Do not run `small`, `large`, or multi-head screening.
- Do not start another Phase automatically.

This runbook ends here and does not initiate any downstream Phase.
