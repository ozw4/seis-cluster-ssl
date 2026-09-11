from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.stratigraphy import pseudo_target_paths

EXPERIMENT_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'38_channel_hmm_distillation_weight_v1'
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
VARIANT_WEIGHTS = {
	'distill005': 0.05,
	'distill010': 0.10,
	'distill040': 0.40,
}
VARIANTS = tuple(VARIANT_WEIGHTS)


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
	return root


@pytest.fixture
def stage2_configs(
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
def h0_stage2_configs(
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


def test_stage2_tree_contains_only_six_full_configs() -> None:
	expected_files = {
		Path(source) / variant / '01_full_25ep.yaml'
		for source in SOURCES
		for variant in VARIANTS
	}
	actual_files = {
		path.relative_to(STAGE2_ROOT)
		for path in STAGE2_ROOT.rglob('*')
		if path.is_file()
	}

	assert actual_files == expected_files
	assert not tuple(STAGE2_ROOT.rglob('*distill020*'))
	assert not tuple(STAGE2_ROOT.rglob('*feasibility*.yaml'))
	assert not tuple(STAGE2_ROOT.rglob('*smoke*.yaml'))


def test_experiment_has_no_new_target_generation_or_clustering() -> None:
	assert not (EXPERIMENT_ROOT / '10_pseudo_targets').exists()
	assert not tuple(EXPERIMENT_ROOT.rglob('*cluster*.yaml'))
	assert not tuple(EXPERIMENT_ROOT.rglob('*export*pseudo*'))
	assert not tuple(EXPERIMENT_ROOT.rglob('*smoke*.yaml'))


def test_configs_resolve_with_reused_h0_targets_and_source_stage1(
	stage2_configs: dict[str, dict[str, dict[str, object]]],
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
		expected_targets = (
			artifact_root
			/ 'pseudo_targets/parihaka/facies_benchmark_v1'
			/ f'ssl_hmm_continuation_v1/{source}'
		)
		for variant in VARIANTS:
			config = stage2_configs[source][variant]
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
			assert pseudo_target_paths(
				targets,
				k=6,
				survey_id='survey',
			).labels.parent == expected_targets / 'k6'


def test_configs_differ_from_source_h0_only_by_output_and_weight(
	stage2_configs: dict[str, dict[str, dict[str, object]]],
	h0_stage2_configs: dict[str, dict[str, object]],
) -> None:
	for source in SOURCES:
		h0 = h0_stage2_configs[source]
		for variant in VARIANTS:
			config = stage2_configs[source][variant]

			assert _without_sweep_fields(config) == _without_sweep_fields(h0)
			assert _section(config, 'paths')['output_root'] != _section(
				h0,
				'paths',
			)['output_root']
			assert _section(config, 'loss')['distillation_weight'] != _section(
				h0,
				'loss',
			)['distillation_weight']


def test_variants_bind_exact_positive_distillation_weights(
	stage2_configs: dict[str, dict[str, dict[str, object]]],
) -> None:
	for source in SOURCES:
		actual_weights = {
			variant: _section(stage2_configs[source][variant], 'loss')[
				'distillation_weight'
			]
			for variant in VARIANTS
		}
		assert actual_weights == VARIANT_WEIGHTS
		assert all(float(weight) > 0.0 for weight in actual_weights.values())
		assert 0.2 not in actual_weights.values()


def test_scientific_optimizer_and_training_budget_are_fixed(
	stage2_configs: dict[str, dict[str, dict[str, object]]],
) -> None:
	for source in SOURCES:
		for variant, weight in VARIANT_WEIGHTS.items():
			config = stage2_configs[source][variant]
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
				'distillation_weight': weight,
			}
			assert {
				key: train[key]
				for key in (
					'batch_size',
					'samples_per_epoch',
					'epochs',
					'num_workers',
					'shuffle',
					'lr',
					'encoder_lr',
					'weight_decay',
					'amp',
					'device',
					'seed',
					'grad_clip_norm',
					'max_steps',
					'allow_overwrite_output',
				)
			} == {
				'batch_size': 16,
				'samples_per_epoch': 10_000,
				'epochs': 25,
				'num_workers': 8,
				'shuffle': True,
				'lr': 1.0e-5,
				'encoder_lr': 1.0e-5,
				'weight_decay': 0.05,
				'amp': False,
				'device': 'cuda',
				'seed': 42,
				'grad_clip_norm': 1.0,
				'max_steps': None,
				'allow_overwrite_output': False,
			}
			planned_global_steps = (
				int(train['epochs'])
				* int(train['samples_per_epoch'])
				// int(train['batch_size'])
			)
			assert planned_global_steps == 15_625


def test_output_roots_are_unique_and_isolated(
	stage2_configs: dict[str, dict[str, dict[str, object]]],
	h0_stage2_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	outputs = {
		Path(str(_section(config, 'paths')['output_root']))
		for source_configs in stage2_configs.values()
		for config in source_configs.values()
	}
	expected_outputs = {
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ f'hmm_distillation_weight_v1/{source}/{variant}/full_25ep'
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
		for config in h0_stage2_configs.values()
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
	pseudo_target_roots = {
		artifact_root
		/ 'pseudo_targets/parihaka/facies_benchmark_v1'
		/ f'ssl_hmm_continuation_v1/{source}'
		for source in SOURCES
	}

	assert outputs == expected_outputs
	assert len(outputs) == 6
	_assert_pairwise_disjoint(outputs)
	for output in outputs:
		for existing in (
			stage1_outputs | h0_outputs | control_outputs | pseudo_target_roots
		):
			assert _paths_do_not_overlap(output, existing)


def _without_sweep_fields(config: dict[str, object]) -> dict[str, object]:
	comparison = deepcopy(config)
	del _section(comparison, 'paths')['output_root']
	del _section(comparison, 'loss')['distillation_weight']
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
