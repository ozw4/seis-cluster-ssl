from __future__ import annotations

from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_mae_training_config,
)

STAGE1_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1/10_stage1'
)


@pytest.fixture
def stage1_configs(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict[str, object]]:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		str(tmp_path / 'artifacts'),
	)
	return {
		'mae_feasibility': resolve_mae_training_config(
			load_config(STAGE1_ROOT / 'mae/01_gpu_feasibility_1step.yaml')
		),
		'mae_full': resolve_mae_training_config(
			load_config(STAGE1_ROOT / 'mae/02_full_100ep.yaml')
		),
		'barlow_twins_feasibility': resolve_barlow_twins_training_config(
			load_config(
				STAGE1_ROOT
				/ 'barlow_twins/01_gpu_feasibility_1step.yaml'
			)
		),
		'barlow_twins_full': resolve_barlow_twins_training_config(
			load_config(STAGE1_ROOT / 'barlow_twins/02_full_100ep.yaml')
		),
		'local_barlow_twins_feasibility': resolve_barlow_twins_training_config(
			load_config(
				STAGE1_ROOT
				/ 'local_barlow_twins_v1/01_gpu_feasibility_1step.yaml'
			)
		),
		'local_barlow_twins_full': resolve_barlow_twins_training_config(
			load_config(
				STAGE1_ROOT / 'local_barlow_twins_v1/02_full_100ep.yaml'
			)
		),
	}


def test_all_stage1_configs_resolve(
	stage1_configs: dict[str, dict[str, object]],
) -> None:
	assert stage1_configs['mae_feasibility']['stage'] == 'train_amp_mae'
	assert stage1_configs['mae_full']['stage'] == 'train_amp_mae'
	assert (
		stage1_configs['barlow_twins_feasibility']['stage']
		== 'barlow_twins_training'
	)
	assert (
		stage1_configs['barlow_twins_full']['stage']
		== 'barlow_twins_training'
	)
	assert (
		stage1_configs['local_barlow_twins_feasibility']['stage']
		== 'barlow_twins_training'
	)
	assert (
		stage1_configs['local_barlow_twins_full']['stage']
		== 'barlow_twins_training'
	)

	for config in stage1_configs.values():
		assert Path(config['paths']['output_root']).is_absolute()


def test_full_configs_share_scientific_and_training_contract(
	stage1_configs: dict[str, dict[str, object]],
) -> None:
	mae = stage1_configs['mae_full']
	barlow_twins = stage1_configs['barlow_twins_full']

	assert mae['manifests']['train'] == barlow_twins['manifests']['train']
	assert (
		mae['manifests']['train_path_list']
		== barlow_twins['manifests']['train_path_list']
	)
	assert mae['data'] == barlow_twins['data']
	assert mae['zero_mask'] == barlow_twins['zero_mask']
	assert mae['model']['patch_size'] == barlow_twins['model']['patch_size']
	assert mae['model']['patch_size'] == [8, 8, 8]
	for key, expected in {
		'encoder_dim': 384,
		'encoder_depth': 8,
		'encoder_heads': 6,
	}.items():
		assert mae['model'][key] == barlow_twins['model'][key] == expected

	expected_train = {
		'batch_size': 16,
		'samples_per_epoch': 10_000,
		'epochs': 100,
		'num_workers': 8,
		'prefetch_factor': 2,
		'persistent_workers': True,
		'shuffle': True,
		'lr': 1.0e-4,
		'weight_decay': 0.05,
		'amp': False,
		'device': 'cuda',
		'seed': 42,
		'grad_clip_norm': 1.0,
	}
	for key, expected in expected_train.items():
		assert mae['train'][key] == barlow_twins['train'][key] == expected


def test_full_configs_preserve_method_specific_contract(
	stage1_configs: dict[str, dict[str, object]],
) -> None:
	mae = stage1_configs['mae_full']
	assert mae['masking']['spatial_mask_ratio'] == 0.75
	assert mae['loss']['reconstruction'] == 'mse'
	assert mae['loss']['gradient_weight'] == 0.0
	assert mae['loss']['visible_reconstruction_weight'] == 0.1
	assert mae['loss']['target_normalization']['mode'] == 'patch_zscore'

	barlow_twins = stage1_configs['barlow_twins_full']
	assert barlow_twins['augmentations']['horizontal_flip_probability'] == 0.5
	assert barlow_twins['barlow_twins']['projector_dim'] == 384
	assert barlow_twins['barlow_twins']['redundancy_weight'] == 0.005
	assert barlow_twins['barlow_twins']['normalization_eps'] == 1.0e-4

	local_barlow_twins = stage1_configs['local_barlow_twins_full']
	assert (
		local_barlow_twins['barlow_twins']['method']
		== 'local_barlow_twins_3d'
	)
	assert local_barlow_twins['barlow_twins']['local_pairs_per_crop'] == 128


def test_feasibility_configs_match_full_geometry_and_runtime_contract(
	stage1_configs: dict[str, dict[str, object]],
) -> None:
	for method in ('mae', 'barlow_twins', 'local_barlow_twins'):
		feasibility = stage1_configs[f'{method}_feasibility']
		full = stage1_configs[f'{method}_full']

		assert feasibility['data'] == full['data']
		assert feasibility['zero_mask'] == full['zero_mask']
		for key in (
			'patch_size',
			'encoder_dim',
			'encoder_depth',
			'encoder_heads',
		):
			assert feasibility['model'][key] == full['model'][key]

		expected_train = {
			'batch_size': 16,
			'samples_per_epoch': 16,
			'epochs': 1,
			'num_workers': 0,
			'prefetch_factor': None,
			'persistent_workers': False,
			'amp': False,
			'device': 'cuda',
			'max_steps': 1,
		}
		for key, expected in expected_train.items():
			assert feasibility['train'][key] == expected

		assert (
			feasibility['paths']['output_root']
			!= full['paths']['output_root']
		)


def test_output_roots_are_unique_and_isolated(
	stage1_configs: dict[str, dict[str, object]],
	tmp_path: Path,
) -> None:
	expected_parent = (
		tmp_path
		/ 'artifacts/pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage1'
	)
	output_roots = {
		Path(config['paths']['output_root'])
		for config in stage1_configs.values()
	}

	assert len(output_roots) == 6
	assert all(path.is_relative_to(expected_parent) for path in output_roots)
