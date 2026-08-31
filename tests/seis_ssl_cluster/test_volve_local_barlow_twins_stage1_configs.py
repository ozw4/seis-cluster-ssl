from __future__ import annotations

import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
)

ROOT = Path(
	'experiments/volve/horizon_benchmark_v1/'
	'31_mae_local_bt_hmm_five_way_v1/10_stage1/local_barlow_twins'
)
MAE_STAGE1 = Path(
	'experiments/volve/horizon_benchmark_v1/10_pretrain/'
	'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/02_full_100ep.yaml'
)
CONFIGS = ('01_smoke_2step.yaml', '02_full_100ep.yaml')


@pytest.fixture(autouse=True)
def _artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts'))


def _resolved(filename: str) -> dict[str, object]:
	return resolve_barlow_twins_training_config(load_config(ROOT / filename))


@pytest.mark.parametrize('filename', CONFIGS)
def test_stage1_matches_volve_mae_input_and_encoder_contract(filename: str) -> None:
	local = load_config(ROOT / filename)
	mae = load_config(MAE_STAGE1)

	for section in ('manifests', 'data', 'zero_mask', 'model'):
		assert local[section] == mae[section]
	resolved = _resolved(filename)
	assert resolved['barlow_twins'] == {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 128,
		'projector_dim': 384,
		'redundancy_weight': 0.005,
		'normalization_eps': 1.0e-4,
	}
	assert resolved['augmentations'] == {'horizontal_flip_probability': 0.5}
	assert 'trace_drop' not in (ROOT / filename).read_text(encoding='utf-8').lower()


def test_full_stage1_uses_adopted_local_bt_budget() -> None:
	full = _resolved(CONFIGS[1])
	mae = load_config(MAE_STAGE1)
	train = full['train']

	assert train['epochs'] == 100
	assert train['samples_per_epoch'] == 10_000
	assert train['batch_size'] == 16
	assert train['seed'] == 42
	assert train['lr'] == 1.0e-4
	assert train['weight_decay'] == 0.05
	assert train['max_steps'] is None
	assert train['samples_per_epoch'] // train['batch_size'] * train['epochs'] == 62_500
	mae_steps = (
		mae['train']['samples_per_epoch']
		// mae['train']['batch_size']
		* mae['train']['epochs']
	)
	assert mae_steps == 250_000


def test_smoke_differs_only_by_output_and_two_step_limit() -> None:
	smoke = load_config(ROOT / CONFIGS[0])
	full = load_config(ROOT / CONFIGS[1])
	expected = deepcopy(full)
	expected['paths']['output_root'] = smoke['paths']['output_root']
	expected['train']['max_steps'] = 2

	assert smoke == expected
	assert smoke['paths']['output_root'] != full['paths']['output_root']
	assert smoke['paths']['output_root'].endswith('/stage1/local_bt/smoke_2step')
	assert full['paths']['output_root'].endswith('/stage1/local_bt/full_100ep')


def test_stage1_smoke_dry_run_writes_nothing(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'dry-run-artifacts'
	environment = {**os.environ, 'SEIS_SSL_CLUSTER_ARTIFACT_ROOT': str(artifact_root)}
	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			'proc/seis_ssl_cluster/train_amp_barlow_twins.py',
			'--config',
			str(ROOT / CONFIGS[0]),
			'--dry-run',
		],
		check=True,
		capture_output=True,
		text=True,
		env=environment,
	)

	assert 'barlow_twins.method: local_barlow_twins_3d' in result.stdout
	assert 'train.max_steps: 2' in result.stdout
	assert 'execution: dry-run; training skipped' in result.stdout
	assert not artifact_root.exists()
