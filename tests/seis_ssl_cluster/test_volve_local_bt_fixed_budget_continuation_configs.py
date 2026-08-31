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
	resolve_mae_training_config,
)

EXPERIMENT_ROOT = Path(
	'experiments/volve/horizon_benchmark_v1/31_mae_local_bt_hmm_five_way_v1'
)
ROOT = EXPERIMENT_ROOT / '30_stage2/local_bt100/local_bt_continue'
STAGE1 = EXPERIMENT_ROOT / '10_stage1/local_barlow_twins/02_full_100ep.yaml'
MAE_CONTINUATION = EXPERIMENT_ROOT / '30_stage2/mae100/mae_continue/02_full_25ep.yaml'
CONFIGS = ('01_smoke_2step.yaml', '02_full_25ep.yaml')


@pytest.fixture(autouse=True)
def _artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts'))


def _resolved(filename: str) -> dict[str, object]:
	return resolve_barlow_twins_training_config(load_config(ROOT / filename))


@pytest.mark.parametrize('filename', CONFIGS)
def test_local_bt_continuation_uses_stage1_identity_without_trace_drop(
	filename: str,
) -> None:
	continuation = load_config(ROOT / filename)
	stage1 = load_config(STAGE1)

	for section in (
		'manifests',
		'data',
		'zero_mask',
		'model',
		'augmentations',
		'barlow_twins',
	):
		assert continuation[section] == stage1[section]
	assert continuation['continuation']['init_checkpoint'].endswith(
		'/mae_local_bt_hmm_five_way_v1/stage1/local_bt/full_100ep/latest.pt'
	)
	assert continuation['continuation']['unfreeze_top_blocks'] == 1
	assert continuation['barlow_twins']['method'] == 'local_barlow_twins_3d'
	assert continuation['barlow_twins']['local_pairs_per_crop'] == 128
	assert 'trace_drop' not in (ROOT / filename).read_text(encoding='utf-8').lower()


def test_local_bt_and_mae_stage2_optimizer_budgets_match() -> None:
	local_train = _resolved(CONFIGS[1])['train']
	mae_train = resolve_mae_training_config(load_config(MAE_CONTINUATION))['train']

	for key in (
		'batch_size',
		'samples_per_epoch',
		'epochs',
		'lr',
		'weight_decay',
		'seed',
	):
		assert local_train[key] == mae_train[key]
	assert local_train['batch_size'] == 4
	assert local_train['epochs'] == 25
	assert local_train['epochs'] * local_train['samples_per_epoch'] // 4 == 62_500
	assert local_train['max_steps'] is None


def test_local_bt_smoke_differs_only_by_output_and_two_step_limit() -> None:
	smoke = load_config(ROOT / CONFIGS[0])
	full = load_config(ROOT / CONFIGS[1])
	expected = deepcopy(full)
	expected['paths']['output_root'] = smoke['paths']['output_root']
	expected['train']['max_steps'] = 2

	assert smoke == expected
	assert smoke['paths']['output_root'].endswith(
		'/local_bt100/local_bt_continue/smoke_2step'
	)
	assert full['paths']['output_root'].endswith(
		'/local_bt100/local_bt_continue/full_25ep'
	)


def test_local_bt_smoke_dry_run_writes_nothing(tmp_path: Path) -> None:
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
	assert 'continuation.unfreeze_top_blocks: 1' in result.stdout
	assert 'train.max_steps: 2' in result.stdout
	assert 'execution: dry-run; training skipped' in result.stdout
	assert not artifact_root.exists()
