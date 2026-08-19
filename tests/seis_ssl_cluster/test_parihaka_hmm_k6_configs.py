from __future__ import annotations

from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.stratigraphy import pseudo_target_paths

SUITE_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1'
)
HMM_ROOT = SUITE_ROOT / '30_stage2'
VARIANTS = ('mae100', 'bt100')
RUNS = {
	'feasibility': '01_gpu_feasibility_1step.yaml',
	'full': '02_full_25ep.yaml',
}


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	for variant, source_method in (
		('mae100', 'mae'),
		('bt100', 'barlow_twins'),
	):
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
			/ 'ssl_hmm_continuation_v1'
			/ variant
			/ 'k6'
		).mkdir(parents=True)
	return root


@pytest.fixture
def hmm_configs(
	artifact_root: Path,
) -> dict[str, dict[str, dict[str, object]]]:
	del artifact_root
	return {
		variant: {
			run: resolve_strat_hmm_pretext_config(
				load_config(HMM_ROOT / variant / 'hmm/k6' / filename)
			)
			for run, filename in RUNS.items()
		}
		for variant in VARIANTS
	}


def test_hmm_k6_configs_share_the_paired_single_head_contract(
	hmm_configs: dict[str, dict[str, dict[str, object]]],
) -> None:
	for run in RUNS:
		mae = hmm_configs['mae100'][run]
		barlow_twins = hmm_configs['bt100'][run]
		for section in ('manifests', 'data', 'zero_mask', 'model'):
			assert mae[section] == barlow_twins[section]

		for config in (mae, barlow_twins):
			assert config['stage'] == 'train_strat_hmm_pretext'
			assert config['pseudo_targets']['k'] == 6
			assert config['pseudo_targets']['min_confidence'] == 0.0
			assert config['head'] == {
				'num_prototypes': 6,
				'projection_dim': 128,
				'temperature': 0.1,
				'normalize': True,
			}
			assert 'spec' not in config['head']
			assert 'ks' not in config['head']
			assert {
				key: config['loss'][key]
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
			assert 'consistency_weight' not in config['loss']
			assert 'consistency_beta' not in config['loss']
			assert 'consistency' not in config
			assert 'spatial_context' not in config
			assert 'pseudo_target_refresh' not in config
			assert config['student']['unfreeze_top_blocks'] == 1


def test_hmm_k6_configs_fix_the_common_optimizer_contract(
	hmm_configs: dict[str, dict[str, dict[str, object]]],
) -> None:
	for variant in VARIANTS:
		for run in RUNS:
			train = hmm_configs[variant][run]['train']
			assert train['batch_size'] == 16
			assert train['shuffle'] is True
			assert train['lr'] == 1.0e-5
			assert train['encoder_lr'] == 1.0e-5
			assert train['weight_decay'] == 0.05
			assert train['amp'] is False
			assert train['device'] == 'cuda'
			assert train['seed'] == 42
			assert train['grad_clip_norm'] == 1.0
			assert train['allow_overwrite_output'] is False


def test_hmm_k6_configs_bind_each_base_to_its_own_source_and_targets(
	hmm_configs: dict[str, dict[str, dict[str, object]]],
	artifact_root: Path,
) -> None:
	expected_sources = {
		'mae100': artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage1/mae/full_100ep/latest.pt',
		'bt100': artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage1/barlow_twins/full_100ep/latest.pt',
	}
	expected_targets = {
		variant: artifact_root
		/ 'pseudo_targets/parihaka/facies_benchmark_v1'
		/ f'ssl_hmm_continuation_v1/{variant}'
		for variant in VARIANTS
	}

	for variant in VARIANTS:
		for run in RUNS:
			config = hmm_configs[variant][run]
			teacher = Path(config['teacher']['checkpoint'])
			student = Path(config['student']['init_checkpoint'])
			targets = Path(config['pseudo_targets']['input_dir'])
			assert teacher == student == expected_sources[variant]
			assert teacher.name == 'latest.pt'
			assert targets == expected_targets[variant]
			paths = pseudo_target_paths(targets, k=6, survey_id='survey')
			assert paths.labels.parent == expected_targets[variant] / 'k6'
			assert paths.labels.parent != expected_targets[variant] / 'k6' / 'k6'
			assert paths.labels.parent.name == 'k6'
			assert paths.labels.parent.parent.name == variant


def test_hmm_k6_training_budgets_are_one_step_and_25_epochs(
	hmm_configs: dict[str, dict[str, dict[str, object]]],
) -> None:
	for variant in VARIANTS:
		feasibility = hmm_configs[variant]['feasibility']['train']
		full = hmm_configs[variant]['full']['train']

		assert feasibility['epochs'] == 1
		assert feasibility['samples_per_epoch'] == 16
		assert feasibility['num_workers'] == 0
		assert feasibility['max_steps'] == 1

		assert full['epochs'] == 25
		assert full['samples_per_epoch'] == 10_000
		assert full['num_workers'] == 8
		assert full['max_steps'] is None
		planned_global_steps = (
			full['epochs'] * full['samples_per_epoch'] // full['batch_size']
		)
		assert planned_global_steps == 15_625


def test_hmm_k6_sources_never_use_controls_or_best_checkpoints(
	hmm_configs: dict[str, dict[str, dict[str, object]]],
) -> None:
	for variant in VARIANTS:
		for run in RUNS:
			config = hmm_configs[variant][run]
			inputs = (
				config['teacher']['checkpoint'],
				config['student']['init_checkpoint'],
				config['pseudo_targets']['input_dir'],
			)
			for value in inputs:
				assert 'mae_continue/full_25ep' not in value
				assert 'bt_continue/full_25ep' not in value
				assert 'best.pt' not in value


def test_hmm_k6_output_roots_are_unique_and_isolated(
	hmm_configs: dict[str, dict[str, dict[str, object]]],
	artifact_root: Path,
) -> None:
	expected_outputs = {
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage2'
		/ variant
		/ 'hmm/k6'
		/ ('gpu_feasibility_1step' if run == 'feasibility' else 'full_25ep')
		for variant in VARIANTS
		for run in RUNS
	}
	actual_outputs = {
		Path(hmm_configs[variant][run]['paths']['output_root'])
		for variant in VARIANTS
		for run in RUNS
	}
	stage1_and_control_outputs = {
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1'
		/ stage
		for stage in (
			'stage1/mae/gpu_feasibility_1step',
			'stage1/mae/full_100ep',
			'stage1/barlow_twins/gpu_feasibility_1step',
			'stage1/barlow_twins/full_100ep',
			'stage2/mae100/mae_continue/full_25ep',
			'stage2/bt100/bt_continue/full_25ep',
		)
	}
	target_generation_outputs = {
		artifact_root
		/ namespace
		/ 'parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1'
		/ suffix
		for namespace, suffix in (
			('embeddings', 'hmm_targets/mae100/overlap_x64'),
			('embeddings', 'hmm_targets/bt100/overlap_x64'),
			('clustering', 'hmm_targets/mae100/k6'),
			('clustering', 'hmm_targets/bt100/k6'),
			('pseudo_targets', 'mae100'),
			('pseudo_targets', 'bt100'),
		)
	}

	assert actual_outputs == expected_outputs
	assert len(actual_outputs) == 4
	assert actual_outputs.isdisjoint(stage1_and_control_outputs)
	assert actual_outputs.isdisjoint(target_generation_outputs)
