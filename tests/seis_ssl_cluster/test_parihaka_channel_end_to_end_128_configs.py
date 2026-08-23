from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.parihaka.channel_data import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.parihaka.channel_decoder import DecoderTiles
from seis_ssl_cluster.parihaka.channel_end_to_end import (
	channel_end_to_end_config_from_mapping,
)
from seis_ssl_cluster.parihaka.channel_end_to_end_results import (
	ENCODER_INITIALIZATIONS,
	channel_end_to_end_summary_config_from_mapping,
)
from seis_ssl_cluster.parihaka.channel_tiles import CHANNEL_PATCH_SIZE_VOXELS

EXPERIMENT_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'34_channel_end_to_end_128_v1'
)
REFERENCE_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'31_channel_end_to_end_v1'
)
CONFIG_NAME = '01_channel_end_to_end_128.yaml'
REFERENCE_CONFIG_NAME = '01_channel_end_to_end.yaml'
LAYOUT_CONFIG = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'30_channel_benchmark_v1/02_layouts.yaml'
)


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	return root


@pytest.fixture
def config_raw(artifact_root: Path) -> dict[str, object]:
	del artifact_root
	return load_config(EXPERIMENT_ROOT / CONFIG_NAME)


@pytest.fixture
def reference_raw(artifact_root: Path) -> dict[str, object]:
	del artifact_root
	return load_config(REFERENCE_ROOT / REFERENCE_CONFIG_NAME)


def test_128_config_resolves_for_training_and_summary(
	config_raw: Mapping[str, object],
	artifact_root: Path,
) -> None:
	config = channel_end_to_end_config_from_mapping(config_raw)
	summary = channel_end_to_end_summary_config_from_mapping(config_raw)

	assert config.tiles == DecoderTiles((8, 8, 8), (4, 4, 4))
	assert summary.core_size_tokens == config.tiles.core_size_tokens
	assert summary.context_halo_tokens == config.tiles.context_halo_tokens
	assert config.runs_root == artifact_root / 'channel_end_to_end_128_v1/runs'
	assert summary.runs_root == config.runs_root
	assert summary.output_dir == config.output_dir
	assert summary.four_way_output_dir == config.four_way_output_dir


def test_128_config_changes_only_halo_within_scientific_conditions(
	config_raw: Mapping[str, object],
	reference_raw: Mapping[str, object],
) -> None:
	assert set(config_raw) == set(reference_raw)
	for section in ('dataset', 'inputs', 'decoder', 'train'):
		assert config_raw[section] == reference_raw[section]

	tiles = _mapping(config_raw, 'tiles')
	reference_tiles = _mapping(reference_raw, 'tiles')
	assert set(tiles) == set(reference_tiles)
	assert tiles['core_size_tokens'] == reference_tiles['core_size_tokens']
	assert tiles['context_halo_tokens'] == [4, 4, 4]
	assert reference_tiles['context_halo_tokens'] == [1, 1, 1]


def test_128_config_keeps_input_decoder_and_training_contracts(
	config_raw: Mapping[str, object],
	reference_raw: Mapping[str, object],
) -> None:
	config = channel_end_to_end_config_from_mapping(config_raw)
	reference = channel_end_to_end_config_from_mapping(reference_raw)

	assert config.labels == reference.labels
	assert config.labels_metadata == reference.labels_metadata
	assert config.reference_embedding_dir == reference.reference_embedding_dir
	assert config.pretrained_checkpoint == reference.pretrained_checkpoint
	assert config.random_checkpoint == reference.random_checkpoint
	assert config.decoder == reference.decoder
	assert config.train == reference.train
	assert config.tiles.core_size_tokens == reference.tiles.core_size_tokens


def test_128_outputs_do_not_overlap_v1_outputs(
	config_raw: Mapping[str, object],
	reference_raw: Mapping[str, object],
) -> None:
	config = channel_end_to_end_config_from_mapping(config_raw)
	reference = channel_end_to_end_config_from_mapping(reference_raw)
	outputs = (
		config.runs_root,
		config.output_dir,
		config.four_way_output_dir,
	)
	reference_outputs = (
		reference.runs_root,
		reference.output_dir,
		reference.four_way_output_dir,
	)

	assert len(set(outputs)) == 3
	assert all(
		_paths_do_not_overlap(output, reference_output)
		for output in outputs
		for reference_output in reference_outputs
	)


def test_128_tile_geometry_preserves_the_supervised_core(
	config_raw: Mapping[str, object],
) -> None:
	config = channel_end_to_end_config_from_mapping(config_raw)
	core = config.tiles.core_size_tokens
	halo = config.tiles.context_halo_tokens
	patch = CHANNEL_PATCH_SIZE_VOXELS
	input_size_tokens = tuple(
		core_axis + 2 * halo_axis
		for core_axis, halo_axis in zip(core, halo, strict=True)
	)
	input_size_voxels = tuple(
		token_axis * patch_axis
		for token_axis, patch_axis in zip(input_size_tokens, patch, strict=True)
	)
	supervised_core_voxels = tuple(
		core_axis * patch_axis
		for core_axis, patch_axis in zip(core, patch, strict=True)
	)

	assert patch == (8, 8, 8)
	assert input_size_tokens == (16, 16, 16)
	assert input_size_voxels == (128, 128, 128)
	assert supervised_core_voxels == (64, 64, 64)


def test_128_config_plans_thirty_jobs_with_reviewed_layouts() -> None:
	planned_job_count = (
		len(ENCODER_INITIALIZATIONS) * len(LAYOUT_IDS) * len(DATA_SIZE_PREFIX)
	)

	assert ENCODER_INITIALIZATIONS == ('pretrained', 'random')
	assert len(LAYOUT_IDS) == 5
	assert tuple(DATA_SIZE_PREFIX) == ('small', 'medium', 'large')
	assert planned_job_count == 2 * 5 * 3 == 30
	assert LAYOUT_CONFIG.is_file()


def test_128_config_uses_full_checkpoints_not_feasibility_outputs(
	config_raw: Mapping[str, object],
) -> None:
	inputs = _mapping(config_raw, 'inputs')
	checkpoints = (
		Path(str(inputs['pretrained_checkpoint'])),
		Path(str(inputs['random_checkpoint'])),
	)

	assert checkpoints[0].name == 'latest.pt'
	assert checkpoints[1].name == 'mae_random_seed42.pt'
	for checkpoint in checkpoints:
		assert 'best.pt' not in checkpoint.parts
		assert not any('feasibility' in part for part in checkpoint.parts)


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return child


def _paths_do_not_overlap(left: Path, right: Path) -> bool:
	return left != right and left not in right.parents and right not in left.parents
