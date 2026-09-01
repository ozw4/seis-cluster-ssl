"""Parity contracts for the reached F3 Local BT one-base-epoch producers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_embedding_extraction_config,
)

EXPERIMENT_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/'
	'111_local_barlow_twins_gaussian_view_v1'
)
BASE5_ROOT = EXPERIMENT_ROOT / '40_base5ep'
BASE1_ROOT = EXPERIMENT_ROOT / '50_base1ep'
ARMS = (
	(
		'local_barlow_twins_gaussian_noise_std010_base1ep',
		'gaussian_noise_std010_base1ep',
		BASE5_ROOT
		/ '10_stage1/gaussian_noise_std010_base5ep/01_screen_5ep.yaml',
		BASE1_ROOT
		/ '10_stage1/gaussian_noise_std010_base1ep/01_screen_1ep.yaml',
		BASE5_ROOT
		/ '15_stage2/gaussian_noise_std010_base5ep/01_continue_25ep.yaml',
		BASE1_ROOT
		/ '15_stage2/gaussian_noise_std010_base1ep/01_continue_25ep.yaml',
		BASE5_ROOT
		/ '20_embeddings/01_extract_gaussian_noise_std010_base5ep.yaml',
		BASE1_ROOT
		/ '20_embeddings/01_extract_gaussian_noise_std010_base1ep.yaml',
	),
	(
		'local_barlow_twins_legacy_flip_base1ep',
		'legacy_flip_base1ep',
		BASE5_ROOT / '10_stage1/legacy_flip_base5ep/01_matched_5ep.yaml',
		BASE1_ROOT / '10_stage1/legacy_flip_base1ep/01_matched_1ep.yaml',
		BASE5_ROOT
		/ '15_stage2/legacy_flip_base5ep/01_continue_25ep.yaml',
		BASE1_ROOT
		/ '15_stage2/legacy_flip_base1ep/01_continue_25ep.yaml',
		BASE5_ROOT / '20_embeddings/02_extract_legacy_flip_base5ep.yaml',
		BASE1_ROOT / '20_embeddings/02_extract_legacy_flip_base1ep.yaml',
	),
)


@pytest.fixture(autouse=True)
def _artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		str(tmp_path / 'artifacts'),
	)


@pytest.mark.parametrize(
	(
		'_model_id',
		'arm',
		'parent_base_path',
		'base_path',
		'_parent_continuation_path',
		'_continuation_path',
		'_parent_extraction_path',
		'_extraction_path',
	),
	ARMS,
)
def test_base1_changes_only_output_and_epoch(
	_model_id: str,
	arm: str,
	parent_base_path: Path,
	base_path: Path,
	_parent_continuation_path: Path,
	_continuation_path: Path,
	_parent_extraction_path: Path,
	_extraction_path: Path,
	tmp_path: Path,
) -> None:
	del (
		_model_id,
		_parent_continuation_path,
		_continuation_path,
		_parent_extraction_path,
		_extraction_path,
	)
	parent = load_config(parent_base_path)
	candidate = load_config(base_path)
	comparison = deepcopy(candidate)
	comparison['paths']['output_root'] = parent['paths']['output_root']
	comparison['train']['epochs'] = parent['train']['epochs']

	assert comparison == parent
	assert candidate['train']['epochs'] == 1
	assert 'max_steps' not in candidate['train']
	resolved = resolve_barlow_twins_training_config(candidate)
	assert Path(resolved['paths']['output_root']) == (
		tmp_path
		/ 'artifacts/pretraining/f3/facies_benchmark_v1'
		/ 'local_barlow_twins_gaussian_view_v1/base1ep/stage1'
		/ arm
		/ 'full_1ep'
	)
	assert resolved['train']['seed'] == 42
	assert (
		resolved['train']['epochs']
		* resolved['train']['samples_per_epoch']
		// resolved['train']['batch_size']
		== 625
	)


@pytest.mark.parametrize(
	(
		'_model_id',
		'arm',
		'_parent_base_path',
		'base_path',
		'parent_continuation_path',
		'continuation_path',
		'_parent_extraction_path',
		'_extraction_path',
	),
	ARMS,
)
def test_base1_continuation_is_exact_fixed_25_epoch_lineage(
	_model_id: str,
	arm: str,
	_parent_base_path: Path,
	base_path: Path,
	parent_continuation_path: Path,
	continuation_path: Path,
	_parent_extraction_path: Path,
	_extraction_path: Path,
	tmp_path: Path,
) -> None:
	del (
		_model_id,
		_parent_base_path,
		_parent_extraction_path,
		_extraction_path,
	)
	parent = load_config(parent_continuation_path)
	candidate = load_config(continuation_path)
	comparison = deepcopy(candidate)
	comparison['paths']['output_root'] = parent['paths']['output_root']
	comparison['continuation']['init_checkpoint'] = parent['continuation'][
		'init_checkpoint'
	]

	assert comparison == parent
	base = resolve_barlow_twins_training_config(load_config(base_path))
	resolved = resolve_barlow_twins_training_config(candidate)
	assert resolved['continuation'] == {
		'init_checkpoint': str(Path(base['paths']['output_root']) / 'latest.pt'),
		'unfreeze_top_blocks': 1,
	}
	assert Path(resolved['paths']['output_root']) == (
		tmp_path
		/ 'artifacts/pretraining/f3/facies_benchmark_v1'
		/ 'local_barlow_twins_gaussian_view_v1/base1ep/stage2'
		/ arm
		/ 'local_bt_continue/full_25ep'
	)
	assert resolved['train']['epochs'] == 25
	assert resolved['train']['lr'] == 1e-5
	assert resolved['continuation']['unfreeze_top_blocks'] == 1
	assert (
		resolved['train']['epochs']
		* resolved['train']['samples_per_epoch']
		// resolved['train']['batch_size']
		== 15_625
	)


@pytest.mark.parametrize(
	(
		'model_id',
		'_arm',
		'_parent_base_path',
		'_base_path',
		'_parent_continuation_path',
		'continuation_path',
		'parent_extraction_path',
		'extraction_path',
	),
	ARMS,
)
def test_base1_extraction_uses_matching_checkpoint_and_model_id(
	model_id: str,
	_arm: str,
	_parent_base_path: Path,
	_base_path: Path,
	_parent_continuation_path: Path,
	continuation_path: Path,
	parent_extraction_path: Path,
	extraction_path: Path,
	tmp_path: Path,
) -> None:
	del _arm, _parent_base_path, _base_path, _parent_continuation_path
	parent = load_config(parent_extraction_path)
	candidate = load_config(extraction_path)
	comparison = deepcopy(candidate)
	comparison['embeddings'] = parent['embeddings']

	assert comparison == parent
	continuation = resolve_barlow_twins_training_config(
		load_config(continuation_path)
	)
	resolved = resolve_embedding_extraction_config(candidate)
	assert Path(resolved['embeddings']['checkpoint']) == (
		Path(continuation['paths']['output_root']) / 'latest.pt'
	)
	assert Path(resolved['embeddings']['output_dir']) == (
		tmp_path
		/ 'artifacts/embeddings/f3/facies_benchmark_v2'
		/ 'local_barlow_twins_gaussian_view_v1/base1ep'
		/ model_id
		/ 'overlap_x64'
	)
	assert Path(resolved['embeddings']['output_dir']).parent.name == model_id


def test_base1_producer_outputs_are_pairwise_distinct() -> None:
	outputs: set[str] = set()
	config_paths: set[Path] = set()
	for (
		_model_id,
		_arm,
		_parent_base_path,
		base_path,
		_parent_continuation_path,
		continuation_path,
		_parent_extraction_path,
		extraction_path,
	) in ARMS:
		outputs.add(load_config(base_path)['paths']['output_root'])
		outputs.add(load_config(continuation_path)['paths']['output_root'])
		outputs.add(load_config(extraction_path)['embeddings']['output_dir'])
		config_paths.update((base_path, continuation_path, extraction_path))

	assert len(outputs) == 6
	assert all('/base1ep/' in output for output in outputs)
	discovered_paths = {
		path
		for stage_dir in ('10_stage1', '15_stage2', '20_embeddings')
		for path in (BASE1_ROOT / stage_dir).rglob('*.yaml')
	}
	assert discovered_paths == config_paths
