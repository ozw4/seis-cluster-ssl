from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.stratigraphy import pseudo_target_paths

EXPERIMENT_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'36_channel_hmm_transition_balance_v1'
)
STAGE2_ROOT = EXPERIMENT_ROOT / '20_stage2'
H0_STAGE2_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1/30_stage2'
)
SOURCE_METHODS = {
	'mae100': 'mae',
	'local_bt100': 'local_barlow_twins_v1',
}
SOURCES = tuple(SOURCE_METHODS)
VARIANTS = ('neutral', 'persist003', 'persist010')


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	for source, source_method in SOURCE_METHODS.items():
		checkpoint = (
			root
			/ 'pretraining/parihaka/facies_benchmark_v1'
			/ 'ssl_hmm_continuation_v1/stage1'
			/ source_method
			/ 'full_100ep/latest.pt'
		)
		checkpoint.parent.mkdir(parents=True, exist_ok=True)
		checkpoint.touch()
		(
			root
			/ 'pseudo_targets/parihaka/facies_benchmark_v1'
			/ f'ssl_hmm_continuation_v1/{source}/k6'
		).mkdir(parents=True)
		for variant in VARIANTS:
			(
				root
				/ 'pseudo_targets/parihaka/facies_benchmark_v1'
				/ f'hmm_transition_balance_v1/{source}/{variant}/k6'
			).mkdir(parents=True)
	return root


@pytest.fixture
def training_configs(
	artifact_root: Path,
) -> dict[str, dict[str, dict[str, object]]]:
	del artifact_root
	return {
		source: {
			variant: resolve_strat_hmm_pretext_config(
				load_config(
					STAGE2_ROOT / source / variant / '01_full_25ep.yaml'
				)
			)
			for variant in VARIANTS
		}
		for source in SOURCES
	}


@pytest.fixture
def h0_training_configs(
	artifact_root: Path,
) -> dict[str, dict[str, object]]:
	del artifact_root
	return {
		source: resolve_strat_hmm_pretext_config(
			load_config(
				H0_STAGE2_ROOT / source / 'hmm/k6/02_full_25ep.yaml'
			)
		)
		for source in SOURCES
	}


def test_configs_resolve_with_source_stage1_and_variant_targets(
	training_configs: dict[str, dict[str, dict[str, object]]],
	artifact_root: Path,
) -> None:
	for source, source_method in SOURCE_METHODS.items():
		expected_checkpoint = (
			artifact_root
			/ 'pretraining/parihaka/facies_benchmark_v1'
			/ 'ssl_hmm_continuation_v1/stage1'
			/ source_method
			/ 'full_100ep/latest.pt'
		)
		for variant in VARIANTS:
			config = training_configs[source][variant]
			expected_targets = (
				artifact_root
				/ 'pseudo_targets/parihaka/facies_benchmark_v1'
				/ f'hmm_transition_balance_v1/{source}/{variant}'
			)
			teacher = Path(str(_section(config, 'teacher')['checkpoint']))
			student = Path(
				str(_section(config, 'student')['init_checkpoint'])
			)
			targets = Path(
				str(_section(config, 'pseudo_targets')['input_dir'])
			)

			assert config['stage'] == 'train_strat_hmm_pretext'
			assert teacher == student == expected_checkpoint
			assert targets == expected_targets
			paths = pseudo_target_paths(targets, k=6, survey_id='survey')
			assert paths.labels.parent == expected_targets / 'k6'
			assert paths.labels.parent != expected_targets / 'k6/k6'


def test_configs_differ_from_source_h0_only_by_input_and_output(
	training_configs: dict[str, dict[str, dict[str, object]]],
	h0_training_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	for source in SOURCES:
		h0 = h0_training_configs[source]
		for variant in VARIANTS:
			config = training_configs[source][variant]
			expected_output = (
				artifact_root
				/ 'pretraining/parihaka/facies_benchmark_v1'
				/ f'hmm_transition_balance_v1/{source}/{variant}/full_25ep'
			)

			assert _without_run_paths(config) == _without_run_paths(h0)
			assert Path(
				str(_section(config, 'paths')['output_root'])
			) == expected_output


def test_fixed_stage2_scientific_and_optimizer_contract(
	training_configs: dict[str, dict[str, dict[str, object]]],
) -> None:
	for source in SOURCES:
		for variant in VARIANTS:
			config = training_configs[source][variant]
			pseudo_targets = _section(config, 'pseudo_targets')
			student = _section(config, 'student')
			head = _section(config, 'head')
			loss = _section(config, 'loss')
			train = _section(config, 'train')

			assert pseudo_targets['k'] == 6
			assert pseudo_targets['min_confidence'] == 0.0
			assert student['unfreeze_top_blocks'] == 1
			assert head == {
				'num_prototypes': 6,
				'projection_dim': 128,
				'temperature': 0.1,
				'normalize': True,
			}
			assert {
				key: loss[key]
				for key in (
					'prototype_weight',
					'usage_weight',
					'entropy_floor',
					'distillation_weight',
				)
			} == {
				'prototype_weight': 1.0,
				'usage_weight': 0.005,
				'entropy_floor': None,
				'distillation_weight': 0.2,
			}
			assert train['batch_size'] == 16
			assert train['samples_per_epoch'] == 10_000
			assert train['epochs'] == 25
			assert train['lr'] == 1.0e-5
			assert train['encoder_lr'] == 1.0e-5
			assert train['weight_decay'] == 0.05
			assert train['amp'] is False
			assert train['seed'] == 42
			assert train['grad_clip_norm'] == 1.0
			assert train['max_steps'] is None
			assert train['allow_overwrite_output'] is False
			planned_global_steps = (
				int(train['epochs'])
				* int(train['samples_per_epoch'])
				// int(train['batch_size'])
			)
			assert planned_global_steps == 15_625


def test_output_roots_are_unique_and_isolated(
	training_configs: dict[str, dict[str, dict[str, object]]],
	h0_training_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	actual_outputs = {
		Path(str(_section(config, 'paths')['output_root']))
		for source_configs in training_configs.values()
		for config in source_configs.values()
	}
	expected_outputs = {
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ f'hmm_transition_balance_v1/{source}/{variant}/full_25ep'
		for source in SOURCES
		for variant in VARIANTS
	}
	stage1_outputs = {
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage1'
		/ source_method
		/ 'full_100ep'
		for source_method in SOURCE_METHODS.values()
	}
	h0_outputs = {
		Path(str(_section(config, 'paths')['output_root']))
		for config in h0_training_configs.values()
	}
	control_outputs = {
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ f'ssl_hmm_continuation_v1/stage2/{suffix}'
		for suffix in (
			'mae100/mae_continue/full_25ep',
			'local_bt100/bt_continue/full_25ep',
		)
	}
	target_generation_artifacts = {
		artifact_root
		/ 'embeddings/parihaka/facies_benchmark_v1'
		/ f'ssl_hmm_continuation_v1/hmm_targets/{source}/overlap_x64'
		for source in SOURCES
	}
	target_generation_artifacts.update(
		{
			artifact_root
			/ namespace
			/ 'parihaka/facies_benchmark_v1'
			/ f'hmm_transition_balance_v1/{source}/{variant}'
			for namespace in ('clustering', 'pseudo_targets')
			for source in SOURCES
			for variant in VARIANTS
		}
	)

	assert actual_outputs == expected_outputs
	assert len(actual_outputs) == 6
	_assert_pairwise_disjoint(actual_outputs)
	for output in actual_outputs:
		for existing in (
			stage1_outputs
			| h0_outputs
			| control_outputs
			| target_generation_artifacts
		):
			assert _paths_do_not_overlap(output, existing)


def test_initialization_never_uses_stage2_or_best_checkpoints(
	training_configs: dict[str, dict[str, dict[str, object]]],
	h0_training_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	h0_checkpoints = {
		Path(str(_section(config, 'paths')['output_root'])) / 'latest.pt'
		for config in h0_training_configs.values()
	}
	control_checkpoints = {
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ f'ssl_hmm_continuation_v1/stage2/{suffix}/latest.pt'
		for suffix in (
			'mae100/mae_continue/full_25ep',
			'local_bt100/bt_continue/full_25ep',
		)
	}

	for source_configs in training_configs.values():
		for config in source_configs.values():
			teacher = Path(str(_section(config, 'teacher')['checkpoint']))
			student = Path(
				str(_section(config, 'student')['init_checkpoint'])
			)
			for checkpoint in (teacher, student):
				assert checkpoint.name == 'latest.pt'
				assert 'stage1' in checkpoint.parts
				assert 'stage2' not in checkpoint.parts
				assert checkpoint not in h0_checkpoints
				assert checkpoint not in control_checkpoints
				assert 'best.pt' not in checkpoint.parts


def test_stage2_contains_only_six_full_configs_and_no_feasibility_yaml() -> None:
	actual_configs = {
		path.relative_to(STAGE2_ROOT)
		for path in STAGE2_ROOT.rglob('*.yaml')
	}
	expected_configs = {
		Path(source) / variant / '01_full_25ep.yaml'
		for source in SOURCES
		for variant in VARIANTS
	}

	assert actual_configs == expected_configs
	assert not tuple(STAGE2_ROOT.rglob('*feasibility*.yaml'))


def _without_run_paths(config: dict[str, object]) -> dict[str, object]:
	comparison = deepcopy(config)
	del _section(comparison, 'paths')['output_root']
	del _section(comparison, 'pseudo_targets')['input_dir']
	return comparison


def _section(config: dict[str, object], key: str) -> dict[str, object]:
	value = config[key]
	if not isinstance(value, dict):
		raise TypeError(f'{key} must be a mapping')
	return value


def _assert_pairwise_disjoint(paths: set[Path]) -> None:
	ordered_paths = sorted(paths)
	for index, left in enumerate(ordered_paths):
		for right in ordered_paths[index + 1 :]:
			assert _paths_do_not_overlap(left, right)


def _paths_do_not_overlap(left: Path, right: Path) -> bool:
	return left != right and left not in right.parents and right not in left.parents
