from __future__ import annotations

from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config, resolve_mae_training_config

SUITE_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1'
)
STAGE1_MAE_FULL = SUITE_ROOT / '10_stage1/mae/02_full_100ep.yaml'
MAE_CONTINUATION_ROOT = SUITE_ROOT / '30_stage2/mae100/mae_continue'


@pytest.fixture
def mae_continuation_configs(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict[str, object]]:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		str(tmp_path / 'artifacts'),
	)
	return {
		'stage1_full': resolve_mae_training_config(load_config(STAGE1_MAE_FULL)),
		'feasibility': resolve_mae_training_config(
			load_config(MAE_CONTINUATION_ROOT / '01_gpu_feasibility_1step.yaml')
		),
		'full': resolve_mae_training_config(
			load_config(MAE_CONTINUATION_ROOT / '02_full_25ep.yaml')
		),
	}


def test_mae_continuation_configs_resolve_with_stage1_latest_source(
	mae_continuation_configs: dict[str, dict[str, object]],
	tmp_path: Path,
) -> None:
	expected_checkpoint = (
		tmp_path
		/ 'artifacts/pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage1/mae/full_100ep/latest.pt'
	)
	for name in ('feasibility', 'full'):
		config = mae_continuation_configs[name]
		assert config['stage'] == 'train_amp_mae'
		continuation = config['continuation']
		assert continuation['init_checkpoint'] == str(expected_checkpoint)
		assert continuation['unfreeze_top_blocks'] == 1
		assert Path(continuation['init_checkpoint']).name == 'latest.pt'


def test_mae_continuation_preserves_stage1_scientific_contract(
	mae_continuation_configs: dict[str, dict[str, object]],
) -> None:
	stage1_full = mae_continuation_configs['stage1_full']
	for name in ('feasibility', 'full'):
		config = mae_continuation_configs[name]
		for section in (
			'manifests',
			'data',
			'zero_mask',
			'model',
			'masking',
			'loss',
		):
			assert config[section] == stage1_full[section]


def test_mae_continuation_training_budgets(
	mae_continuation_configs: dict[str, dict[str, object]],
) -> None:
	feasibility_train = mae_continuation_configs['feasibility']['train']
	full_train = mae_continuation_configs['full']['train']

	common_train = {
		'batch_size': 16,
		'shuffle': True,
		'lr': 1.0e-5,
		'weight_decay': 0.05,
		'amp': False,
		'amp_dtype': 'auto',
		'device': 'cuda',
		'seed': 42,
		'grad_clip_norm': 1.0,
		'runtime_check_mode': 'once',
		'stage_timing': False,
	}
	for train in (feasibility_train, full_train):
		for key, expected in common_train.items():
			assert train[key] == expected

	assert feasibility_train['epochs'] == 1
	assert feasibility_train['samples_per_epoch'] == 16
	assert feasibility_train['num_workers'] == 0
	assert feasibility_train['prefetch_factor'] is None
	assert feasibility_train['persistent_workers'] is False
	assert feasibility_train['max_steps'] == 1
	assert (
		mae_continuation_configs['feasibility']['visualization']['mae_debug'][
			'enabled'
		]
		is False
	)

	assert full_train['epochs'] == 25
	assert full_train['samples_per_epoch'] == 10_000
	assert full_train['num_workers'] == 8
	assert full_train['prefetch_factor'] == 2
	assert full_train['persistent_workers'] is True
	assert full_train.get('max_steps') is None
	assert (
		mae_continuation_configs['full']['visualization']
		== mae_continuation_configs['stage1_full']['visualization']
	)
	planned_global_steps = (
		full_train['epochs']
		* full_train['samples_per_epoch']
		// full_train['batch_size']
	)
	assert planned_global_steps == 15_625


def test_mae_continuation_output_roots_are_unique_and_isolated(
	mae_continuation_configs: dict[str, dict[str, object]],
	tmp_path: Path,
) -> None:
	artifact_root = tmp_path / 'artifacts'
	expected_roots = {
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ (
			'ssl_hmm_continuation_v1/stage2/mae100/mae_continue/'
			'gpu_feasibility_1step'
		),
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage2/mae100/mae_continue/full_25ep',
	}
	actual_roots = {
		Path(mae_continuation_configs[name]['paths']['output_root'])
		for name in ('feasibility', 'full')
	}
	stage1_and_barlow_twins_roots = {
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ f'ssl_hmm_continuation_v1/stage1/{method}/{run}'
		for method in ('mae', 'barlow_twins')
		for run in ('gpu_feasibility_1step', 'full_100ep')
	}

	assert actual_roots == expected_roots
	assert actual_roots.isdisjoint(stage1_and_barlow_twins_roots)
