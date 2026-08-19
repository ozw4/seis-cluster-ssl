from __future__ import annotations

from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_mae_training_config,
)

SUITE_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1'
)
STAGE1_ROOT = SUITE_ROOT / '10_stage1'
STAGE1_BARLOW_TWINS_FULL = STAGE1_ROOT / 'barlow_twins/02_full_100ep.yaml'
BARLOW_CONTINUATION_ROOT = SUITE_ROOT / '30_stage2/bt100/bt_continue'
MAE_CONTINUATION_ROOT = SUITE_ROOT / '30_stage2/mae100/mae_continue'


@pytest.fixture
def barlow_continuation_configs(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict[str, object]]:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		str(tmp_path / 'artifacts'),
	)
	return {
		'stage1_full': resolve_barlow_twins_training_config(
			load_config(STAGE1_BARLOW_TWINS_FULL)
		),
		'feasibility': resolve_barlow_twins_training_config(
			load_config(
				BARLOW_CONTINUATION_ROOT / '01_gpu_feasibility_1step.yaml'
			)
		),
		'full': resolve_barlow_twins_training_config(
			load_config(BARLOW_CONTINUATION_ROOT / '02_full_25ep.yaml')
		),
	}


def test_barlow_continuation_configs_resolve_with_stage1_latest_source(
	barlow_continuation_configs: dict[str, dict[str, object]],
	tmp_path: Path,
) -> None:
	expected_checkpoint = (
		tmp_path
		/ 'artifacts/pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage1/barlow_twins/full_100ep/latest.pt'
	)
	for name in ('feasibility', 'full'):
		config = barlow_continuation_configs[name]
		assert config['stage'] == 'barlow_twins_training'
		continuation = config['continuation']
		assert continuation['init_checkpoint'] == str(expected_checkpoint)
		assert continuation['unfreeze_top_blocks'] == 1
		assert Path(continuation['init_checkpoint']).name == 'latest.pt'


def test_barlow_continuation_preserves_stage1_scientific_contract(
	barlow_continuation_configs: dict[str, dict[str, object]],
) -> None:
	stage1_full = barlow_continuation_configs['stage1_full']
	for name in ('feasibility', 'full'):
		config = barlow_continuation_configs[name]
		for section in (
			'manifests',
			'data',
			'zero_mask',
			'model',
			'augmentations',
			'barlow_twins',
		):
			assert config[section] == stage1_full[section]


def test_barlow_continuation_training_budgets(
	barlow_continuation_configs: dict[str, dict[str, object]],
) -> None:
	feasibility_train = barlow_continuation_configs['feasibility']['train']
	full_train = barlow_continuation_configs['full']['train']

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

	assert full_train['epochs'] == 25
	assert full_train['samples_per_epoch'] == 10_000
	assert full_train['num_workers'] == 8
	assert full_train['prefetch_factor'] == 2
	assert full_train['persistent_workers'] is True
	assert full_train.get('max_steps') is None
	planned_global_steps = (
		full_train['epochs']
		* full_train['samples_per_epoch']
		// full_train['batch_size']
	)
	assert planned_global_steps == 15_625


def test_barlow_continuation_output_roots_are_unique_and_isolated(
	barlow_continuation_configs: dict[str, dict[str, object]],
	tmp_path: Path,
) -> None:
	artifact_root = tmp_path / 'artifacts'
	expected_roots = {
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ (
			'ssl_hmm_continuation_v1/stage2/bt100/bt_continue/'
			'gpu_feasibility_1step'
		),
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage2/bt100/bt_continue/full_25ep',
	}
	actual_roots = {
		Path(barlow_continuation_configs[name]['paths']['output_root'])
		for name in ('feasibility', 'full')
	}

	stage1_roots = set()
	for method, resolver in (
		('mae', resolve_mae_training_config),
		('barlow_twins', resolve_barlow_twins_training_config),
	):
		for filename in ('01_gpu_feasibility_1step.yaml', '02_full_100ep.yaml'):
			config = resolver(load_config(STAGE1_ROOT / method / filename))
			stage1_roots.add(Path(config['paths']['output_root']))
	mae_continuation_roots = {
		Path(
			resolve_mae_training_config(
				load_config(MAE_CONTINUATION_ROOT / filename)
			)['paths']['output_root']
		)
		for filename in ('01_gpu_feasibility_1step.yaml', '02_full_25ep.yaml')
	}

	assert actual_roots == expected_roots
	assert len(actual_roots) == 2
	assert actual_roots.isdisjoint(stage1_roots)
	assert actual_roots.isdisjoint(mae_continuation_roots)
	assert stage1_roots.isdisjoint(mae_continuation_roots)
