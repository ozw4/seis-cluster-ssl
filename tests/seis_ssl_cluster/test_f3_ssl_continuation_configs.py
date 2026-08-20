from __future__ import annotations

from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_mae_training_config,
)

F3_SUITE_ROOT = Path(
	'experiments/f3/facies_benchmark_v1/21_ssl_hmm_continuation_v1'
)
PARIHAKA_SUITE_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/21_ssl_hmm_continuation_v1'
)


def _load_continuation_configs(root: Path) -> dict[str, dict[str, object]]:
	return {
		'mae_feasibility': resolve_mae_training_config(
			load_config(
				root
				/ '30_stage2/mae100/mae_continue/'
				'01_gpu_feasibility_1step.yaml'
			)
		),
		'mae_full': resolve_mae_training_config(
			load_config(
				root / '30_stage2/mae100/mae_continue/02_full_25ep.yaml'
			)
		),
		'barlow_twins_feasibility': resolve_barlow_twins_training_config(
			load_config(
				root
				/ '30_stage2/bt100/bt_continue/'
				'01_gpu_feasibility_1step.yaml'
			)
		),
		'barlow_twins_full': resolve_barlow_twins_training_config(
			load_config(
				root / '30_stage2/bt100/bt_continue/02_full_25ep.yaml'
			)
		),
	}


def _load_f3_stage1_full_configs() -> dict[str, dict[str, object]]:
	return {
		'mae': resolve_mae_training_config(
			load_config(F3_SUITE_ROOT / '10_stage1/mae/02_full_100ep.yaml')
		),
		'barlow_twins': resolve_barlow_twins_training_config(
			load_config(
				F3_SUITE_ROOT
				/ '10_stage1/barlow_twins/02_full_100ep.yaml'
			)
		),
	}


@pytest.fixture
def continuation_contracts(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict[str, dict[str, object]]]:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		str(tmp_path / 'artifacts'),
	)
	return {
		'f3': _load_continuation_configs(F3_SUITE_ROOT),
		'f3_stage1': _load_f3_stage1_full_configs(),
		'parihaka': _load_continuation_configs(PARIHAKA_SUITE_ROOT),
	}


def test_f3_continuation_configs_resolve_with_paired_stage1_latest_sources(
	continuation_contracts: dict[str, dict[str, dict[str, object]]],
	tmp_path: Path,
) -> None:
	f3 = continuation_contracts['f3']
	expected_sources = {
		'mae': (
			tmp_path
			/ 'artifacts/pretraining/f3/facies_benchmark_v1'
			/ 'ssl_hmm_continuation_v1/stage1/mae/full_100ep/latest.pt'
		),
		'barlow_twins': (
			tmp_path
			/ 'artifacts/pretraining/f3/facies_benchmark_v1'
			/ (
				'ssl_hmm_continuation_v1/stage1/barlow_twins/'
				'full_100ep/latest.pt'
			)
		),
	}

	for method in ('mae', 'barlow_twins'):
		for run in ('feasibility', 'full'):
			config = f3[f'{method}_{run}']
			expected_stage = (
				'train_amp_mae'
				if method == 'mae'
				else 'barlow_twins_training'
			)
			assert config['stage'] == expected_stage
			continuation = config['continuation']
			assert continuation['init_checkpoint'] == str(expected_sources[method])
			assert continuation['unfreeze_top_blocks'] == 1
			assert Path(continuation['init_checkpoint']).name == 'latest.pt'
			serialized = repr(continuation).lower()
			for forbidden in ('best.pt', 'nopims', 'parihaka', 'historical'):
				assert forbidden not in serialized


def test_f3_continuation_uses_full_volume_and_stage1_scientific_contract(
	continuation_contracts: dict[str, dict[str, dict[str, object]]],
	tmp_path: Path,
) -> None:
	f3 = continuation_contracts['f3']
	stage1 = continuation_contracts['f3_stage1']
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
	method_sections = {
		'mae': ('manifests', 'data', 'zero_mask', 'model', 'masking', 'loss'),
		'barlow_twins': (
			'manifests',
			'data',
			'zero_mask',
			'model',
			'augmentations',
			'barlow_twins',
		),
	}

	for method, sections in method_sections.items():
		for run in ('feasibility', 'full'):
			config = f3[f'{method}_{run}']
			assert config['manifests']['train'] == str(expected_manifest)
			assert config['manifests']['train_path_list'] == str(expected_path_list)
			assert 'train_npy_paths.txt' not in repr(config['manifests'])
			for section in sections:
				assert config[section] == stage1[method][section]


def test_f3_continuation_preserves_selected_parihaka_settings(
	continuation_contracts: dict[str, dict[str, dict[str, object]]],
) -> None:
	f3 = continuation_contracts['f3']
	parihaka = continuation_contracts['parihaka']
	method_sections = {
		'mae': ('data', 'zero_mask', 'model', 'masking', 'loss', 'visualization'),
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
			assert (
				f3[name]['continuation']['unfreeze_top_blocks']
				== parihaka[name]['continuation']['unfreeze_top_blocks']
				== 1
			)


def test_f3_continuation_training_budgets_are_paired(
	continuation_contracts: dict[str, dict[str, dict[str, object]]],
) -> None:
	f3 = continuation_contracts['f3']
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
	for method in ('mae', 'barlow_twins'):
		feasibility = f3[f'{method}_feasibility']['train']
		full = f3[f'{method}_full']['train']
		for train in (feasibility, full):
			for key, expected in common_train.items():
				assert train[key] == expected

		assert feasibility['samples_per_epoch'] == 16
		assert feasibility['epochs'] == 1
		assert feasibility['num_workers'] == 0
		assert feasibility['prefetch_factor'] is None
		assert feasibility['persistent_workers'] is False
		assert feasibility['max_steps'] == 1

		assert full['samples_per_epoch'] == 10_000
		assert full['epochs'] == 25
		assert full['num_workers'] == 8
		assert full['prefetch_factor'] == 2
		assert full['persistent_workers'] is True
		assert full.get('max_steps') is None
		planned_global_steps = (
			full['epochs'] * full['samples_per_epoch'] // full['batch_size']
		)
		assert planned_global_steps == 15_625

	assert f3['mae_feasibility']['visualization']['mae_debug']['enabled'] is False
	assert (
		f3['mae_full']['visualization']
		== continuation_contracts['f3_stage1']['mae']['visualization']
	)


def test_f3_continuation_output_roots_are_unique_and_stage1_isolated(
	continuation_contracts: dict[str, dict[str, dict[str, object]]],
	tmp_path: Path,
) -> None:
	f3 = continuation_contracts['f3']
	artifact_root = tmp_path / 'artifacts'
	expected_roots = {
		artifact_root
		/ 'pretraining/f3/facies_benchmark_v1'
		/ f'ssl_hmm_continuation_v1/stage2/{branch}/{run}'
		for branch in ('mae100/mae_continue', 'bt100/bt_continue')
		for run in ('gpu_feasibility_1step', 'full_25ep')
	}
	actual_roots = {
		Path(config['paths']['output_root']) for config in f3.values()
	}
	stage1_roots = {
		Path(config['paths']['output_root'])
		for config in continuation_contracts['f3_stage1'].values()
	}

	assert actual_roots == expected_roots
	assert len(actual_roots) == 4
	assert actual_roots.isdisjoint(stage1_roots)
