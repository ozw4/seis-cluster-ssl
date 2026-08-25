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

FIVE_WAY_ROOT = Path(
	'experiments/f3/facies_benchmark_v1/110_lithology_mae_local_bt_five_way_v1'
)
LOCAL_BT_CONTINUE_ROOT = FIVE_WAY_ROOT / '10_stage2/local_bt100/local_bt_continue'
GLOBAL_BT_CONTINUE_ROOT = Path(
	'experiments/f3/facies_benchmark_v1/21_ssl_hmm_continuation_v1'
	'/30_stage2/bt100/bt_continue'
)
CONFIG_NAMES = ('01_gpu_feasibility_1step.yaml', '02_full_25ep.yaml')
TRACE_DROP_KEYS = ('policy', 'reflection_probability', 'trace_drop_probability')


@pytest.fixture(autouse=True)
def _artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		str(tmp_path / 'artifacts'),
	)


def _resolved(root: Path, filename: str) -> dict[str, object]:
	return resolve_barlow_twins_training_config(load_config(root / filename))


@pytest.mark.parametrize('filename', CONFIG_NAMES)
def test_local_bt_continue_starts_from_local_bt_100ep_checkpoint(
	filename: str,
	tmp_path: Path,
) -> None:
	config = _resolved(LOCAL_BT_CONTINUE_ROOT, filename)
	expected_source = (
		tmp_path
		/ 'artifacts/pretraining/f3/facies_benchmark_v1'
		/ 'local_barlow_twins_v1/full_100ep/latest.pt'
	)

	continuation = config['continuation']
	assert continuation['init_checkpoint'] == str(expected_source)
	assert continuation['unfreeze_top_blocks'] == 1
	serialized = repr(continuation).lower()
	for forbidden in ('best.pt', 'mae', 'ssl_hmm_continuation_v1', 'trace_drop'):
		assert forbidden not in serialized


@pytest.mark.parametrize('filename', CONFIG_NAMES)
def test_local_bt_continue_uses_local_method_without_trace_drop(
	filename: str,
) -> None:
	raw = load_config(LOCAL_BT_CONTINUE_ROOT / filename)
	config = _resolved(LOCAL_BT_CONTINUE_ROOT, filename)

	assert config['barlow_twins']['method'] == 'local_barlow_twins_3d'
	assert config['barlow_twins']['local_pairs_per_crop'] == 128
	for augmentations in (raw['augmentations'], config['augmentations']):
		for key in TRACE_DROP_KEYS:
			assert key not in augmentations


def test_local_bt_continue_full_is_fixed_budget_25ep() -> None:
	full = _resolved(LOCAL_BT_CONTINUE_ROOT, CONFIG_NAMES[1])
	train = full['train']

	assert train['epochs'] == 25
	assert train['samples_per_epoch'] == 10_000
	assert train['batch_size'] == 16
	assert train.get('max_steps') is None
	assert (
		train['epochs'] * train['samples_per_epoch'] // train['batch_size']
		== 15_625
	)
	assert train['lr'] == 1.0e-5
	assert train['weight_decay'] == 0.05
	assert train['amp'] is False
	assert train['seed'] == 42
	assert train['grad_clip_norm'] == 1.0


def test_local_bt_continue_only_changes_local_identity_and_paths() -> None:
	for filename in CONFIG_NAMES:
		local = load_config(LOCAL_BT_CONTINUE_ROOT / filename)
		standard = load_config(GLOBAL_BT_CONTINUE_ROOT / filename)

		comparison = deepcopy(local)
		comparison['paths']['output_root'] = standard['paths']['output_root']
		comparison['continuation']['init_checkpoint'] = standard['continuation'][
			'init_checkpoint'
		]
		comparison['barlow_twins'].pop('method')
		comparison['barlow_twins'].pop('local_pairs_per_crop')
		assert comparison == standard


def test_global_bt_continue_configs_are_unchanged() -> None:
	for filename, expected_leaf in zip(
		CONFIG_NAMES,
		('gpu_feasibility_1step', 'full_25ep'),
		strict=True,
	):
		standard = load_config(GLOBAL_BT_CONTINUE_ROOT / filename)
		assert 'method' not in standard['barlow_twins']
		assert 'local_pairs_per_crop' not in standard['barlow_twins']
		assert standard['paths']['output_root'].endswith(
			f'ssl_hmm_continuation_v1/stage2/bt100/bt_continue/{expected_leaf}'
		)
		assert standard['continuation']['init_checkpoint'].endswith(
			'ssl_hmm_continuation_v1/stage1/barlow_twins/full_100ep/latest.pt'
		)


def test_local_bt_continue_output_roots_are_separated(tmp_path: Path) -> None:
	feasibility = _resolved(LOCAL_BT_CONTINUE_ROOT, CONFIG_NAMES[0])
	full = _resolved(LOCAL_BT_CONTINUE_ROOT, CONFIG_NAMES[1])
	expected_prefix = (
		tmp_path
		/ 'artifacts/pretraining/f3/facies_benchmark_v1'
		/ 'mae_local_bt_five_way_v1/stage2/local_bt100/local_bt_continue'
	)

	feasibility_root = Path(feasibility['paths']['output_root'])
	full_root = Path(full['paths']['output_root'])
	assert feasibility_root == expected_prefix / 'gpu_feasibility_1step'
	assert full_root == expected_prefix / 'full_25ep'
	assert feasibility_root != full_root


def test_local_bt_continue_dry_run_creates_no_artifacts(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'dry-run-artifacts'
	environment = {**os.environ, 'SEIS_SSL_CLUSTER_ARTIFACT_ROOT': str(artifact_root)}
	for filename in CONFIG_NAMES:
		result = subprocess.run(  # noqa: S603
			[
				sys.executable,
				'proc/seis_ssl_cluster/train_amp_barlow_twins.py',
				'--config',
				str(LOCAL_BT_CONTINUE_ROOT / filename),
				'--dry-run',
			],
			check=True,
			capture_output=True,
			text=True,
			env=environment,
		)

		assert 'barlow_twins.method: local_barlow_twins_3d' in result.stdout
		assert 'barlow_twins.local_pairs_per_crop: 128' in result.stdout
		assert 'execution: dry-run; training skipped' in result.stdout
		assert not artifact_root.exists()
