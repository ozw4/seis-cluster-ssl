from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.training.random_checkpoint import (
	create_random_mae_checkpoint_from_config,
	load_checkpoint_metadata_without_weights,
	random_mae_checkpoint_config_from_mapping,
)
from tests.seis_ssl_cluster.test_random_mae_checkpoint import (
	_write_reference_checkpoint,
)

RANDOM_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/110_lithology_mae_local_bt_five_way_v1'
	'/40_random/01_create_random_checkpoint.yaml'
)
PARIHAKA_RANDOM_CONFIG = Path(
	'experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1'
	'/03_create_random_checkpoint.yaml'
)
EXPECTED_REFERENCE_SUFFIX = (
	'pretraining/f3/facies_benchmark_v1/ssl_hmm_continuation_v1'
	'/stage2/mae100/mae_continue/full_25ep/latest.pt'
)
EXPECTED_OUTPUT_SUFFIX = (
	'pretraining/f3/facies_benchmark_v1/mae_local_bt_five_way_v1'
	'/random/random_init.pt'
)


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	return root


def _place_reference_checkpoint(destination: Path) -> dict[str, torch.Tensor]:
	"""Reuse the shared tiny reference checkpoint at the F3 fixed-budget path."""
	destination.parent.mkdir(parents=True, exist_ok=True)
	source = _write_reference_checkpoint(destination.parent)
	source.replace(destination)
	payload = torch.load(destination, map_location='cpu', weights_only=False)
	return payload['model_state_dict']


def test_config_uses_f3_mae_fixed_budget_reference_and_parihaka_seed(
	artifact_root: Path,
) -> None:
	raw = load_config(RANDOM_CONFIG)
	reference_path = artifact_root / EXPECTED_REFERENCE_SUFFIX
	reference_path.parent.mkdir(parents=True, exist_ok=True)
	reference_path.touch()

	settings = random_mae_checkpoint_config_from_mapping(raw)
	assert settings.reference_checkpoint == reference_path
	assert settings.output_checkpoint == artifact_root / EXPECTED_OUTPUT_SUFFIX
	assert settings.output_checkpoint != settings.reference_checkpoint
	assert 'mae_local_bt_five_way_v1' in str(settings.output_checkpoint)

	parihaka = PARIHAKA_RANDOM_CONFIG.read_text(encoding='utf-8')
	assert 'seed: 42' in parihaka
	assert settings.seed == 42


def test_dry_run_creates_no_checkpoint(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'dry-run-artifacts'
	reference_path = artifact_root / EXPECTED_REFERENCE_SUFFIX
	reference_path.parent.mkdir(parents=True, exist_ok=True)
	reference_path.touch()
	environment = {**os.environ, 'SEIS_SSL_CLUSTER_ARTIFACT_ROOT': str(artifact_root)}
	before = {str(path) for path in artifact_root.rglob('*')}

	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			'proc/seis_ssl_cluster/create_random_mae_checkpoint.py',
			'--config',
			str(RANDOM_CONFIG),
			'--dry-run',
		],
		check=True,
		capture_output=True,
		text=True,
		env=environment,
	)

	assert 'random_checkpoint.seed: 42' in result.stdout
	assert 'execution: dry-run; checkpoint creation skipped' in result.stdout
	assert {str(path) for path in artifact_root.rglob('*')} == before
	assert not (artifact_root / EXPECTED_OUTPUT_SUFFIX).exists()


def test_random_checkpoint_from_synthetic_reference_matches_contract(
	artifact_root: Path,
) -> None:
	raw = load_config(RANDOM_CONFIG)
	reference_path = artifact_root / EXPECTED_REFERENCE_SUFFIX
	reference_state = _place_reference_checkpoint(reference_path)

	output_path = create_random_mae_checkpoint_from_config(raw)

	assert output_path == artifact_root / EXPECTED_OUTPUT_SUFFIX
	assert reference_path.is_file()
	payload_without_weights = load_checkpoint_metadata_without_weights(output_path)
	assert payload_without_weights['metadata'] == {
		'random_encoder_baseline': True,
		'reference_checkpoint': str(reference_path),
		'reference_model_tag': (
			'ssl_hmm_continuation_v1_stage2_mae100_mae_continue_full_25ep'
		),
		'seed': 42,
		'pretrained_weights_loaded': False,
	}

	payload = torch.load(output_path, map_location='cpu', weights_only=False)
	assert payload['epoch'] == 0
	assert payload['global_step'] == 0
	assert payload['optimizer_state_dict'] == {}
	random_state = payload['model_state_dict']
	assert set(random_state) == set(reference_state)
	for key, value in random_state.items():
		assert value.shape == reference_state[key].shape
	assert any(
		not torch.equal(random_state[key], reference_state[key])
		for key in random_state
	)
