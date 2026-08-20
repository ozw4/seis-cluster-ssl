from __future__ import annotations

from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_mae_training_config,
)

F3_STAGE1_ROOT = Path(
	'experiments/f3/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1/10_stage1'
)
PARIHAKA_STAGE1_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1/10_stage1'
)


def _load_stage1_configs(root: Path) -> dict[str, dict[str, object]]:
	return {
		'mae_feasibility': resolve_mae_training_config(
			load_config(root / 'mae/01_gpu_feasibility_1step.yaml')
		),
		'mae_full': resolve_mae_training_config(
			load_config(root / 'mae/02_full_100ep.yaml')
		),
		'barlow_twins_feasibility': resolve_barlow_twins_training_config(
			load_config(
				root / 'barlow_twins/01_gpu_feasibility_1step.yaml'
			)
		),
		'barlow_twins_full': resolve_barlow_twins_training_config(
			load_config(root / 'barlow_twins/02_full_100ep.yaml')
		),
	}


@pytest.fixture
def stage1_contracts(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict[str, dict[str, object]]]:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		str(tmp_path / 'artifacts'),
	)
	return {
		'f3': _load_stage1_configs(F3_STAGE1_ROOT),
		'parihaka': _load_stage1_configs(PARIHAKA_STAGE1_ROOT),
	}


def test_f3_stage1_configs_resolve_with_full_volume_inputs(
	stage1_contracts: dict[str, dict[str, dict[str, object]]],
	tmp_path: Path,
) -> None:
	f3 = stage1_contracts['f3']
	expected_manifest = (
		tmp_path
		/ 'artifacts/registry/manifests/f3/facies_benchmark_v1'
		/ 'f3_amplitude_manifest.json'
	)
	expected_path_list = (
		tmp_path
		/ 'artifacts/registry/splits/f3/facies_benchmark_v1'
		/ 'f3_npy_paths.txt'
	)

	for name, config in f3.items():
		expected_stage = (
			'train_amp_mae' if name.startswith('mae_') else 'barlow_twins_training'
		)
		assert config['stage'] == expected_stage
		assert config['manifests']['train'] == str(expected_manifest)
		assert config['manifests']['train_path_list'] == str(expected_path_list)
		assert 'continuation' not in config
		serialized = repr(config).lower()
		assert 'nopims' not in serialized
		assert 'train_npy_paths.txt' not in serialized


def test_f3_stage1_paired_configs_share_common_contract(
	stage1_contracts: dict[str, dict[str, dict[str, object]]],
) -> None:
	f3 = stage1_contracts['f3']
	common_train = {
		'batch_size': 16,
		'shuffle': True,
		'lr': 1.0e-4,
		'weight_decay': 0.05,
		'amp': False,
		'amp_dtype': 'auto',
		'device': 'cuda',
		'seed': 42,
		'grad_clip_norm': 1.0,
	}

	for run in ('feasibility', 'full'):
		mae = f3[f'mae_{run}']
		barlow_twins = f3[f'barlow_twins_{run}']
		for section in ('manifests', 'data', 'zero_mask', 'model'):
			assert mae[section] == barlow_twins[section]
		for key, expected in common_train.items():
			assert mae['train'][key] == barlow_twins['train'][key] == expected

	assert f3['mae_full']['data']['local_crop_size'] == [128, 128, 128]
	model = f3['mae_full']['model']
	for key, expected in {
		'patch_size': [8, 8, 8],
		'encoder_dim': 384,
		'encoder_depth': 8,
		'encoder_heads': 6,
		'decoder_dim': 256,
		'decoder_depth': 4,
		'decoder_heads': 4,
	}.items():
		assert model[key] == expected


def test_f3_stage1_preserves_method_and_parihaka_scientific_settings(
	stage1_contracts: dict[str, dict[str, dict[str, object]]],
) -> None:
	f3 = stage1_contracts['f3']
	parihaka = stage1_contracts['parihaka']
	method_sections = {
		'mae': ('data', 'zero_mask', 'model', 'masking', 'loss'),
		'barlow_twins': (
			'data',
			'zero_mask',
			'model',
			'augmentations',
			'barlow_twins',
		),
	}
	for method, sections in method_sections.items():
		for run in ('feasibility', 'full'):
			name = f'{method}_{run}'
			for section in sections:
				assert f3[name][section] == parihaka[name][section]
			assert f3[name]['train'] == parihaka[name]['train']

	mae = f3['mae_full']
	assert mae['masking']['spatial_mask_ratio'] == 0.75
	assert mae['masking']['block_size_tokens'] == [1, 1, 1]
	assert mae['loss']['reconstruction'] == 'mse'
	assert mae['loss']['gradient_weight'] == 0.0
	assert mae['loss']['visible_reconstruction_weight'] == 0.1
	assert mae['loss']['target_normalization']['mode'] == 'patch_zscore'
	assert mae['loss']['target_normalization']['eps'] == 1.0e-6
	assert mae['loss']['target_normalization']['min_std'] == 0.05
	barlow_twins = f3['barlow_twins_full']
	assert barlow_twins['augmentations']['horizontal_flip_probability'] == 0.5
	assert barlow_twins['barlow_twins']['projector_dim'] == 384
	assert barlow_twins['barlow_twins']['redundancy_weight'] == 0.005
	assert barlow_twins['barlow_twins']['normalization_eps'] == 1.0e-4


def test_f3_stage1_training_budgets(
	stage1_contracts: dict[str, dict[str, dict[str, object]]],
) -> None:
	f3 = stage1_contracts['f3']
	for method in ('mae', 'barlow_twins'):
		feasibility = f3[f'{method}_feasibility']['train']
		full = f3[f'{method}_full']['train']
		assert feasibility['batch_size'] == 16
		assert feasibility['samples_per_epoch'] == 16
		assert feasibility['epochs'] == 1
		assert feasibility['num_workers'] == 0
		assert feasibility['prefetch_factor'] is None
		assert feasibility['persistent_workers'] is False
		assert feasibility['max_steps'] == 1

		assert full['batch_size'] == 16
		assert full['samples_per_epoch'] == 10_000
		assert full['epochs'] == 100
		assert full['num_workers'] == 8
		assert full['prefetch_factor'] == 2
		assert full['persistent_workers'] is True
		assert full.get('max_steps') is None
		planned_global_steps = (
			full['epochs'] * full['samples_per_epoch'] // full['batch_size']
		)
		assert planned_global_steps == 62_500

	assert f3['mae_feasibility']['visualization']['mae_debug']['enabled'] is False
	assert f3['mae_full']['visualization']['mae_debug']['enabled'] is True


def test_f3_stage1_output_roots_are_unique(
	stage1_contracts: dict[str, dict[str, dict[str, object]]],
	tmp_path: Path,
) -> None:
	f3 = stage1_contracts['f3']
	expected_parent = (
		tmp_path
		/ 'artifacts/pretraining/f3/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage1'
	)
	output_roots = {
		Path(config['paths']['output_root']) for config in f3.values()
	}

	assert len(output_roots) == 4
	assert all(path.is_relative_to(expected_parent) for path in output_roots)
