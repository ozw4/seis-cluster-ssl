from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config, resolve_embedding_extraction_config
from seis_ssl_cluster.parihaka.channel_data import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.parihaka.channel_decoder import (
	ChannelDecoderConfig,
	channel_decoder_config_from_mapping,
)

EXPERIMENT_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'32_channel_ssl_hmm_four_way_v1'
)
LEGACY_BENCHMARK_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1'
)
LAYOUT_CONFIG = LEGACY_BENCHMARK_ROOT / '02_layouts.yaml'
MODEL_CONFIGS = {
	'mae': '01_extract_mae_embeddings.yaml',
	'barlow_twins': '02_extract_barlow_twins_embeddings.yaml',
	'mae_hmm_k6': '03_extract_mae_hmm_k6_embeddings.yaml',
	'barlow_twins_hmm_k6': '04_extract_barlow_twins_hmm_k6_embeddings.yaml',
}
EXPECTED_CHECKPOINTS = {
	'mae': 'stage2/mae100/mae_continue/full_25ep/latest.pt',
	'barlow_twins': 'stage2/bt100/bt_continue/full_25ep/latest.pt',
	'mae_hmm_k6': 'stage2/mae100/hmm/k6/full_25ep/latest.pt',
	'barlow_twins_hmm_k6': 'stage2/bt100/hmm/k6/full_25ep/latest.pt',
}
EXPECTED_EMBEDDING_SETTINGS = {
	'window_size': [128, 128, 128],
	'overlap': [64, 64, 64],
	'output_dtype': 'float16',
	'batch_size': 1,
	'prefetch_queue_depth': 0,
	'amp': False,
	'amp_dtype': 'auto',
	'stage_timing': False,
	'min_token_valid_fraction': 0.5,
	'preprocessing_cache': {
		'mode': 'off',
		'chunk_size_x': 16,
		'reuse': True,
		'cleanup': False,
	},
}


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	return root


@pytest.fixture
def extraction_configs(
	artifact_root: Path,
) -> dict[str, dict[str, object]]:
	del artifact_root
	return {
		model_id: resolve_embedding_extraction_config(
			load_config(EXPERIMENT_ROOT / filename)
		)
		for model_id, filename in MODEL_CONFIGS.items()
	}


@pytest.fixture
def channel_raw(artifact_root: Path) -> dict[str, object]:
	del artifact_root
	return load_config(EXPERIMENT_ROOT / '05_channel_four_way.yaml')


@pytest.fixture
def channel_config(channel_raw: Mapping[str, object]) -> ChannelDecoderConfig:
	return channel_decoder_config_from_mapping(channel_raw)


def test_extraction_configs_resolve_to_exact_stage2_latest_checkpoints(
	extraction_configs: Mapping[str, Mapping[str, object]],
	artifact_root: Path,
) -> None:
	checkpoint_root = (
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1'
	)
	output_root = (
		artifact_root
		/ 'embeddings/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1'
		/ 'channel_four_way'
	)
	outputs: set[Path] = set()

	for model_id, config in extraction_configs.items():
		embeddings = _mapping(config, 'embeddings')
		checkpoint = Path(str(embeddings['checkpoint']))
		output_dir = Path(str(embeddings['output_dir']))

		assert config['stage'] == 'extract_embeddings'
		assert checkpoint == checkpoint_root / EXPECTED_CHECKPOINTS[model_id]
		assert checkpoint.name == 'latest.pt'
		assert 'best.pt' not in checkpoint.parts
		assert 'stage1' not in checkpoint.parts
		assert not any('feasibility' in part for part in checkpoint.parts)
		assert output_dir == output_root / model_id / 'overlap_x64'
		outputs.add(output_dir)

	assert len(outputs) == len(MODEL_CONFIGS) == 4


def test_extraction_configs_share_the_scientific_extraction_contract(
	extraction_configs: Mapping[str, Mapping[str, object]],
) -> None:
	reference = extraction_configs['mae']

	for config in extraction_configs.values():
		assert config['paths'] == reference['paths']
		assert config['manifests'] == reference['manifests']
		assert config['embedding'] == reference['embedding']
		assert config['embedding'] == EXPECTED_EMBEDDING_SETTINGS


def test_channel_config_binds_exactly_the_four_extraction_sources(
	channel_config: ChannelDecoderConfig,
	extraction_configs: Mapping[str, Mapping[str, object]],
) -> None:
	assert tuple(channel_config.models) == tuple(MODEL_CONFIGS)

	for model_id, source in channel_config.models.items():
		extraction = _mapping(extraction_configs[model_id], 'embeddings')
		assert source.embedding_dir == Path(str(extraction['output_dir']))
		assert source.expected_checkpoint == Path(str(extraction['checkpoint']))


def test_channel_config_reuses_frozen_decoder_contract_and_isolates_outputs(
	channel_raw: Mapping[str, object],
	channel_config: ChannelDecoderConfig,
	artifact_root: Path,
) -> None:
	legacy = load_config(LEGACY_BENCHMARK_ROOT / '06_channel_benchmark.yaml')
	new_inputs = _mapping(channel_raw, 'inputs')
	new_outputs = _mapping(channel_raw, 'outputs')
	legacy_outputs = _mapping(legacy, 'outputs')

	for section in ('decoder', 'tiles', 'train'):
		assert channel_raw[section] == legacy[section]
	assert new_inputs['runs_root'] == new_outputs['runs_root']
	assert channel_config.runs_root == Path(str(new_outputs['runs_root']))
	assert channel_config.runs_root == (
		artifact_root / 'channel_benchmark/ssl_hmm_four_way_v1/runs'
	)
	assert Path(str(new_outputs['output_dir'])) == (
		artifact_root / 'channel_benchmark/ssl_hmm_four_way_v1/summary'
	)
	for new_root, legacy_root in (
		(new_outputs['runs_root'], legacy_outputs['runs_root']),
		(new_outputs['output_dir'], legacy_outputs['output_dir']),
	):
		assert _paths_do_not_overlap(Path(str(new_root)), Path(str(legacy_root)))


def test_channel_config_plans_sixty_jobs_with_reviewed_layouts(
	channel_config: ChannelDecoderConfig,
) -> None:
	planned_job_count = (
		len(channel_config.models) * len(LAYOUT_IDS) * len(DATA_SIZE_PREFIX)
	)

	assert planned_job_count == 4 * 5 * 3 == 60
	assert LAYOUT_CONFIG.is_file()


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return child


def _paths_do_not_overlap(left: Path, right: Path) -> bool:
	return left != right and left not in right.parents and right not in left.parents
