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
	'33_channel_mae_local_bt_four_way_v1'
)
REFERENCE_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'32_channel_ssl_hmm_four_way_v1'
)
LAYOUT_CONFIG = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'30_channel_benchmark_v1/02_layouts.yaml'
)
LOCAL_MODEL_CONFIGS = {
	'local_barlow_twins': '01_extract_local_barlow_twins_embeddings.yaml',
	'local_barlow_twins_hmm_k6': (
		'02_extract_local_barlow_twins_hmm_k6_embeddings.yaml'
	),
}
MODEL_IDS = (
	'mae',
	'local_barlow_twins',
	'mae_hmm_k6',
	'local_barlow_twins_hmm_k6',
)
REUSED_MODEL_IDS = {'mae', 'mae_hmm_k6'}
LOCAL_MODEL_IDS = set(LOCAL_MODEL_CONFIGS)
EXPECTED_CHECKPOINTS = {
	'local_barlow_twins': (
		'stage2/local_bt100/bt_continue/full_25ep/latest.pt'
	),
	'local_barlow_twins_hmm_k6': (
		'stage2/local_bt100/hmm/k6/full_25ep/latest.pt'
	),
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
		for model_id, filename in LOCAL_MODEL_CONFIGS.items()
	}


@pytest.fixture
def reference_extraction(artifact_root: Path) -> dict[str, object]:
	del artifact_root
	return resolve_embedding_extraction_config(
		load_config(REFERENCE_ROOT / '01_extract_mae_embeddings.yaml')
	)


@pytest.fixture
def channel_raw(artifact_root: Path) -> dict[str, object]:
	del artifact_root
	return load_config(EXPERIMENT_ROOT / '03_channel_mae_local_bt_four_way.yaml')


@pytest.fixture
def reference_channel_raw(artifact_root: Path) -> dict[str, object]:
	del artifact_root
	return load_config(REFERENCE_ROOT / '05_channel_four_way.yaml')


@pytest.fixture
def channel_config(channel_raw: Mapping[str, object]) -> ChannelDecoderConfig:
	return channel_decoder_config_from_mapping(channel_raw)


def test_local_extraction_configs_resolve_to_exact_stage2_latest_checkpoints(
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
		/ 'channel_mae_local_bt_four_way_v1'
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

	assert len(outputs) == len(LOCAL_MODEL_CONFIGS) == 2


def test_local_extraction_configs_reuse_exact_mae_extraction_contract(
	extraction_configs: Mapping[str, Mapping[str, object]],
	reference_extraction: Mapping[str, object],
) -> None:
	for config in extraction_configs.values():
		assert set(config) == set(reference_extraction)
		assert config['stage'] == reference_extraction['stage']
		assert config['paths'] == reference_extraction['paths']
		assert config['manifests'] == reference_extraction['manifests']
		assert config['embedding'] == reference_extraction['embedding']
		assert set(_mapping(config, 'embeddings')) == {
			'checkpoint',
			'output_dir',
		}


def test_channel_config_binds_sources_in_required_order(
	channel_raw: Mapping[str, object],
	reference_channel_raw: Mapping[str, object],
	channel_config: ChannelDecoderConfig,
	extraction_configs: Mapping[str, Mapping[str, object]],
) -> None:
	models = _mapping(_mapping(channel_raw, 'embeddings'), 'models')
	reference_models = _mapping(
		_mapping(reference_channel_raw, 'embeddings'),
		'models',
	)

	assert tuple(models) == MODEL_IDS
	assert tuple(channel_config.models) == MODEL_IDS
	for model_id in REUSED_MODEL_IDS:
		assert models[model_id] == reference_models[model_id]
	for model_id, extraction_config in extraction_configs.items():
		extraction = _mapping(extraction_config, 'embeddings')
		assert models[model_id] == {
			'dir': extraction['output_dir'],
			'checkpoint': extraction['checkpoint'],
		}

	assert set(models) - set(reference_models) == LOCAL_MODEL_IDS
	assert set(models) & set(reference_models) == REUSED_MODEL_IDS


def test_channel_config_reuses_decoder_inputs_and_runs_but_isolates_summary(
	channel_raw: Mapping[str, object],
	reference_channel_raw: Mapping[str, object],
	channel_config: ChannelDecoderConfig,
	artifact_root: Path,
) -> None:
	inputs = _mapping(channel_raw, 'inputs')
	outputs = _mapping(channel_raw, 'outputs')
	reference_inputs = _mapping(reference_channel_raw, 'inputs')
	reference_outputs = _mapping(reference_channel_raw, 'outputs')

	assert channel_raw['dataset'] == reference_channel_raw['dataset']
	for key in ('labels_npy', 'labels_metadata_json'):
		assert inputs[key] == reference_inputs[key]
	for section in ('decoder', 'tiles', 'train'):
		assert channel_raw[section] == reference_channel_raw[section]

	assert inputs['runs_root'] == outputs['runs_root']
	assert inputs['runs_root'] == reference_inputs['runs_root']
	assert outputs['runs_root'] == reference_outputs['runs_root']
	assert channel_config.runs_root == (
		artifact_root / 'channel_benchmark/ssl_hmm_four_way_v1/runs'
	)
	new_summary = Path(str(outputs['output_dir']))
	reference_summary = Path(str(reference_outputs['output_dir']))
	assert new_summary == (
		artifact_root / 'channel_benchmark/mae_local_bt_four_way_v1/summary'
	)
	assert _paths_do_not_overlap(new_summary, reference_summary)


def test_channel_config_plans_sixty_jobs_with_reviewed_layouts(
	channel_config: ChannelDecoderConfig,
) -> None:
	planned_job_count = (
		len(channel_config.models) * len(LAYOUT_IDS) * len(DATA_SIZE_PREFIX)
	)

	assert planned_job_count == 4 * 5 * 3 == 60
	assert LAYOUT_CONFIG.is_file()


def test_experiment_has_only_the_two_local_extraction_configs() -> None:
	extraction_files = tuple(
		path.name for path in sorted(EXPERIMENT_ROOT.glob('*extract*embeddings.yaml'))
	)

	assert extraction_files == tuple(LOCAL_MODEL_CONFIGS.values())
	assert not tuple(EXPERIMENT_ROOT.glob('*extract_mae_embeddings.yaml'))


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return child


def _paths_do_not_overlap(left: Path, right: Path) -> bool:
	return left != right and left not in right.parents and right not in left.parents
