from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config, resolve_barlow_twins_training_config

SUITE_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1'
)
STAGE1_ROOT = SUITE_ROOT / '10_stage1/local_barlow_twins_v1'
STAGE2_LOCAL_ROOT = SUITE_ROOT / '30_stage2/local_bt100'
CONTROL_ROOT = STAGE2_LOCAL_ROOT / 'bt_continue'
D4_ROOT = STAGE2_LOCAL_ROOT / 'bt_continue_d4_trace_drop'
HMM_ROOT = STAGE2_LOCAL_ROOT / 'hmm/k6'
RUNS = {
	'feasibility': '01_gpu_feasibility_1step.yaml',
	'full': '02_full_25ep.yaml',
}
EXPECTED_AUGMENTATIONS = {
	'policy': 'xy_d4_trace_drop_v1',
	'reflection_probability': 0.5,
	'trace_drop_probability': 0.02,
}


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	return root


@pytest.fixture
def configs(
	artifact_root: Path,
) -> dict[str, dict[str, dict[str, object]]]:
	del artifact_root
	return {
		variant: {
			run: resolve_barlow_twins_training_config(
				load_config(root / filename)
			)
			for run, filename in RUNS.items()
		}
		for variant, root in (
			('d4', D4_ROOT),
			('control', CONTROL_ROOT),
		)
	}


def test_d4_trace_drop_configs_resolve_from_stage1_local_bt100(
	configs: dict[str, dict[str, dict[str, object]]],
	artifact_root: Path,
) -> None:
	expected_source = (
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage1/local_barlow_twins_v1'
		/ 'full_100ep/latest.pt'
	)
	for run in RUNS:
		config = configs['d4'][run]
		assert config['stage'] == 'barlow_twins_training'
		continuation = config['continuation']
		source = Path(continuation['init_checkpoint'])
		assert source == expected_source
		assert continuation['unfreeze_top_blocks'] == 1
		assert source.name == 'latest.pt'
		assert 'best.pt' not in str(source)
		assert '/stage2/' not in str(source)
		assert 'gpu_feasibility' not in str(source)
		assert '/bt_continue/' not in str(source)
		assert '/hmm/' not in str(source)


def test_d4_trace_drop_configs_have_exact_augmentation_and_local_contract(
	configs: dict[str, dict[str, dict[str, object]]],
) -> None:
	for run in RUNS:
		config = configs['d4'][run]
		assert config['augmentations'] == EXPECTED_AUGMENTATIONS
		assert 'horizontal_flip_probability' not in config['augmentations']
		assert config['barlow_twins']['method'] == 'local_barlow_twins_3d'
		assert config['barlow_twins']['local_pairs_per_crop'] == 128


def test_d4_trace_drop_differs_from_control_only_by_augmentation_and_output(
	configs: dict[str, dict[str, dict[str, object]]],
) -> None:
	for run in RUNS:
		d4 = deepcopy(configs['d4'][run])
		control = deepcopy(configs['control'][run])
		for config in (d4, control):
			config.pop('augmentations')
			config['paths'].pop('output_root')
		assert d4 == control


def test_d4_trace_drop_training_budgets(
	configs: dict[str, dict[str, dict[str, object]]],
) -> None:
	feasibility = configs['d4']['feasibility']['train']
	full = configs['d4']['full']['train']

	assert feasibility['batch_size'] == 16
	assert feasibility['samples_per_epoch'] == 16
	assert feasibility['epochs'] == 1
	assert feasibility['max_steps'] == 1

	assert full['batch_size'] == 16
	assert full['samples_per_epoch'] == 10_000
	assert full['epochs'] == 25
	assert full.get('max_steps') is None
	assert full['lr'] == 1.0e-5
	assert full['weight_decay'] == 0.05
	assert full['amp'] is False
	assert full['seed'] == 42
	planned_global_steps = (
		full['epochs'] * full['samples_per_epoch'] // full['batch_size']
	)
	assert planned_global_steps == 15_625


def test_d4_trace_drop_output_roots_are_unique_and_isolated(
	configs: dict[str, dict[str, dict[str, object]]],
	artifact_root: Path,
) -> None:
	base = (
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1'
	)
	expected = {
		base
		/ 'stage2/local_bt100/bt_continue_d4_trace_drop'
		/ ('gpu_feasibility_1step' if run == 'feasibility' else 'full_25ep')
		for run in RUNS
	}
	actual = {
		Path(configs['d4'][run]['paths']['output_root']) for run in RUNS
	}
	control_outputs = {
		Path(configs['control'][run]['paths']['output_root']) for run in RUNS
	}
	stage1_outputs = {
		Path(
			resolve_barlow_twins_training_config(
				load_config(STAGE1_ROOT / filename)
			)['paths']['output_root']
		)
		for filename in (
			'01_gpu_feasibility_1step.yaml',
			'02_full_100ep.yaml',
		)
	}
	hmm_outputs = {
		Path(load_config(HMM_ROOT / filename)['paths']['output_root'])
		for filename in RUNS.values()
	}

	assert actual == expected
	assert len(actual) == 2
	assert actual.isdisjoint(control_outputs)
	assert actual.isdisjoint(stage1_outputs)
	assert actual.isdisjoint(hmm_outputs)
