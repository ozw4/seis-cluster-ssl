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
BARLOW_VARIANTS = {
	'bt100': 'barlow_twins',
	'local_bt100': 'local_barlow_twins_v1',
}
BARLOW_CONTINUATION_ROOT = SUITE_ROOT / '30_stage2'
MAE_CONTINUATION_ROOT = SUITE_ROOT / '30_stage2/mae100/mae_continue'
RUNS = {
	'feasibility': '01_gpu_feasibility_1step.yaml',
	'full': '02_full_25ep.yaml',
}


@pytest.fixture
def barlow_continuation_configs(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict[str, dict[str, object]]]:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		str(tmp_path / 'artifacts'),
	)
	return {
		variant: {
			'stage1_full': resolve_barlow_twins_training_config(
				load_config(STAGE1_ROOT / stage1_method / '02_full_100ep.yaml')
			),
			**{
				run: resolve_barlow_twins_training_config(
					load_config(
						BARLOW_CONTINUATION_ROOT
						/ variant
						/ 'bt_continue'
						/ filename
					)
				)
				for run, filename in RUNS.items()
			},
		}
		for variant, stage1_method in BARLOW_VARIANTS.items()
	}


def test_barlow_continuation_configs_resolve_with_stage1_latest_source(
	barlow_continuation_configs: dict[str, dict[str, dict[str, object]]],
	tmp_path: Path,
) -> None:
	expected_checkpoints = {
		variant: tmp_path
		/ 'artifacts/pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage1'
		/ source_method
		/ 'full_100ep/latest.pt'
		for variant, source_method in BARLOW_VARIANTS.items()
	}
	for variant in BARLOW_VARIANTS:
		for run in RUNS:
			config = barlow_continuation_configs[variant][run]
			assert config['stage'] == 'barlow_twins_training'
			continuation = config['continuation']
			checkpoint = Path(continuation['init_checkpoint'])
			assert checkpoint == expected_checkpoints[variant]
			assert continuation['unfreeze_top_blocks'] == 1
			assert checkpoint.name == 'latest.pt'
			assert 'best.pt' not in str(checkpoint)
			assert '/stage2/' not in str(checkpoint)

	for run in RUNS:
		local = barlow_continuation_configs['local_bt100'][run]
		assert local['barlow_twins']['method'] == 'local_barlow_twins_3d'
		assert local['barlow_twins']['local_pairs_per_crop'] == 128


def test_barlow_continuation_preserves_stage1_scientific_contract(
	barlow_continuation_configs: dict[str, dict[str, dict[str, object]]],
) -> None:
	for variant in BARLOW_VARIANTS:
		stage1_full = barlow_continuation_configs[variant]['stage1_full']
		for run in RUNS:
			config = barlow_continuation_configs[variant][run]
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
	barlow_continuation_configs: dict[str, dict[str, dict[str, object]]],
) -> None:
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
	for variant in BARLOW_VARIANTS:
		feasibility_train = barlow_continuation_configs[variant]['feasibility'][
			'train'
		]
		full_train = barlow_continuation_configs[variant]['full']['train']

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

	for run in RUNS:
		assert (
			barlow_continuation_configs['local_bt100'][run]['train']
			== barlow_continuation_configs['bt100'][run]['train']
		)


def test_barlow_continuation_output_roots_are_unique_and_isolated(
	barlow_continuation_configs: dict[str, dict[str, dict[str, object]]],
	tmp_path: Path,
) -> None:
	artifact_root = tmp_path / 'artifacts'
	expected_roots = {
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage2'
		/ variant
		/ 'bt_continue'
		/ ('gpu_feasibility_1step' if run == 'feasibility' else 'full_25ep')
		for variant in BARLOW_VARIANTS
		for run in RUNS
	}
	actual_roots = {
		Path(barlow_continuation_configs[variant][run]['paths']['output_root'])
		for variant in BARLOW_VARIANTS
		for run in RUNS
	}

	stage1_roots = set()
	for method, resolver in (
		('mae', resolve_mae_training_config),
		('barlow_twins', resolve_barlow_twins_training_config),
		('local_barlow_twins_v1', resolve_barlow_twins_training_config),
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
	hmm_roots = {
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage2'
		/ variant
		/ 'hmm/k6'
		/ ('gpu_feasibility_1step' if run == 'feasibility' else 'full_25ep')
		for variant in ('mae100', *BARLOW_VARIANTS)
		for run in RUNS
	}

	assert actual_roots == expected_roots
	assert len(actual_roots) == 4
	assert actual_roots.isdisjoint(stage1_roots)
	assert actual_roots.isdisjoint(mae_continuation_roots)
	assert actual_roots.isdisjoint(hmm_roots)
	assert stage1_roots.isdisjoint(mae_continuation_roots)
