from __future__ import annotations

import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config, resolve_mae_training_config

ROOT = Path(
	'experiments/volve/horizon_benchmark_v1/'
	'31_mae_local_bt_hmm_five_way_v1/30_stage2/mae100/mae_continue'
)
STAGE1 = Path(
	'experiments/volve/horizon_benchmark_v1/10_pretrain/'
	'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/02_full_100ep.yaml'
)
CONFIGS = ('01_smoke_2step.yaml', '02_full_25ep.yaml')


@pytest.fixture(autouse=True)
def _artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts'))


def _resolved(filename: str) -> dict[str, object]:
	return resolve_mae_training_config(load_config(ROOT / filename))


@pytest.mark.parametrize('filename', CONFIGS)
def test_mae_continuation_reuses_stage1_scientific_contract(filename: str) -> None:
	continuation = load_config(ROOT / filename)
	stage1 = load_config(STAGE1)

	for section in ('manifests', 'data', 'zero_mask', 'model', 'masking', 'loss'):
		assert continuation[section] == stage1[section]
	assert continuation['continuation']['init_checkpoint'].endswith(
		'/pretraining/volve/horizon_benchmark_v1/'
		'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/'
		'full_100ep/latest.pt'
	)
	assert continuation['continuation']['unfreeze_top_blocks'] == 1
	assert 'horizon' not in repr(
		{
			'manifests': continuation['manifests'],
			'continuation': continuation['continuation'],
		}
	).lower().replace('horizon_benchmark_v1', '')


def test_full_mae_continuation_is_exact_fixed_budget() -> None:
	train = _resolved(CONFIGS[1])['train']

	assert train['epochs'] == 25
	assert train['samples_per_epoch'] == 10_000
	assert train['batch_size'] == 4
	assert train['epochs'] * train['samples_per_epoch'] // train['batch_size'] == 62_500
	assert train['lr'] == 1.0e-5
	assert train['weight_decay'] == 0.05
	assert train['seed'] == 42
	assert train.get('max_steps') is None


def test_mae_smoke_differs_only_by_output_and_two_step_limit() -> None:
	smoke = load_config(ROOT / CONFIGS[0])
	full = load_config(ROOT / CONFIGS[1])
	expected = deepcopy(full)
	expected['paths']['output_root'] = smoke['paths']['output_root']
	expected['train']['max_steps'] = 2

	assert smoke == expected
	assert smoke['paths']['output_root'].endswith('/mae100/mae_continue/smoke_2step')
	assert full['paths']['output_root'].endswith('/mae100/mae_continue/full_25ep')


def test_mae_smoke_dry_run_writes_nothing(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'dry-run-artifacts'
	environment = {**os.environ, 'SEIS_SSL_CLUSTER_ARTIFACT_ROOT': str(artifact_root)}
	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			'proc/seis_ssl_cluster/train_amp_mae.py',
			'--config',
			str(ROOT / CONFIGS[0]),
			'--dry-run',
		],
		check=True,
		capture_output=True,
		text=True,
		env=environment,
	)

	assert 'continuation.unfreeze_top_blocks: 1' in result.stdout
	assert 'execution: dry-run; training skipped' in result.stdout
	assert not artifact_root.exists()
