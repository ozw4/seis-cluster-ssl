from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_embedding_extraction_config,
)
from seis_ssl_cluster.parihaka.channel_data import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.parihaka.channel_decoder import (
	ChannelDecoderConfig,
	channel_decoder_config_from_mapping,
)

EXPERIMENT_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'35_channel_local_bt_d4_trace_drop_v1'
)
REFERENCE_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'33_channel_mae_local_bt_four_way_v1'
)
D4_TRAINING_CONFIG = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1/30_stage2/local_bt100/'
	'bt_continue_d4_trace_drop/02_full_25ep.yaml'
)
MODEL_IDS = (
	'local_barlow_twins',
	'local_barlow_twins_d4_trace_drop',
	'local_barlow_twins_hmm_k6',
)
REUSED_MODEL_IDS = {'local_barlow_twins', 'local_barlow_twins_hmm_k6'}
NEW_MODEL_ID = 'local_barlow_twins_d4_trace_drop'
FORBIDDEN_MODEL_IDS = {'mae', 'mae_hmm_k6', 'barlow_twins', 'random'}
EXPECTED_AUGMENTATIONS = {
	'policy': 'xy_d4_trace_drop_v1',
	'reflection_probability': 0.5,
	'trace_drop_probability': 0.02,
}


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	return root


@pytest.fixture
def extraction_config(artifact_root: Path) -> dict[str, object]:
	del artifact_root
	return resolve_embedding_extraction_config(
		load_config(EXPERIMENT_ROOT / '01_extract_augmented_embeddings.yaml')
	)


@pytest.fixture
def reference_extraction(artifact_root: Path) -> dict[str, object]:
	del artifact_root
	return resolve_embedding_extraction_config(
		load_config(
			REFERENCE_ROOT / '01_extract_local_barlow_twins_embeddings.yaml'
		)
	)


@pytest.fixture
def channel_raw(artifact_root: Path) -> dict[str, object]:
	del artifact_root
	return load_config(EXPERIMENT_ROOT / '02_channel_comparison.yaml')


@pytest.fixture
def reference_channel_raw(artifact_root: Path) -> dict[str, object]:
	del artifact_root
	return load_config(
		REFERENCE_ROOT / '03_channel_mae_local_bt_four_way.yaml'
	)


@pytest.fixture
def channel_config(channel_raw: Mapping[str, object]) -> ChannelDecoderConfig:
	return channel_decoder_config_from_mapping(channel_raw)


def test_augmented_extraction_resolves_to_full_stage2_latest_checkpoint(
	extraction_config: Mapping[str, object],
	artifact_root: Path,
) -> None:
	embeddings = _mapping(extraction_config, 'embeddings')
	checkpoint = Path(str(embeddings['checkpoint']))
	output_dir = Path(str(embeddings['output_dir']))

	assert extraction_config['stage'] == 'extract_embeddings'
	assert checkpoint == (
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage2/local_bt100'
		/ 'bt_continue_d4_trace_drop/full_25ep/latest.pt'
	)
	assert checkpoint.name == 'latest.pt'
	assert 'best.pt' not in checkpoint.parts
	assert 'stage1' not in checkpoint.parts
	assert not any('feasibility' in part for part in checkpoint.parts)
	assert output_dir == (
		artifact_root
		/ 'embeddings/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/channel_local_bt_d4_trace_drop_v1'
		/ 'local_barlow_twins_d4_trace_drop/overlap_x64'
	)


def test_augmented_extraction_reuses_exact_local_bt_extraction_contract(
	extraction_config: Mapping[str, object],
	reference_extraction: Mapping[str, object],
) -> None:
	assert set(extraction_config) == set(reference_extraction)
	assert extraction_config['stage'] == reference_extraction['stage']
	assert extraction_config['paths'] == reference_extraction['paths']
	assert extraction_config['manifests'] == reference_extraction['manifests']
	assert extraction_config['embedding'] == reference_extraction['embedding']
	assert set(_mapping(extraction_config, 'embeddings')) == {
		'checkpoint',
		'output_dir',
	}


def test_augmented_checkpoint_preserves_auditable_policy_identity(
	extraction_config: Mapping[str, object],
) -> None:
	training = resolve_barlow_twins_training_config(
		load_config(D4_TRAINING_CONFIG)
	)
	checkpoint = Path(str(_mapping(extraction_config, 'embeddings')['checkpoint']))
	training_output = Path(str(_mapping(training, 'paths')['output_root']))

	assert checkpoint == training_output / 'latest.pt'
	assert training['augmentations'] == EXPECTED_AUGMENTATIONS
	assert 'augmentations' not in extraction_config


def test_channel_config_binds_models_in_required_order(
	channel_raw: Mapping[str, object],
	channel_config: ChannelDecoderConfig,
) -> None:
	models = _mapping(_mapping(channel_raw, 'embeddings'), 'models')

	assert tuple(models) == MODEL_IDS
	assert tuple(channel_config.models) == MODEL_IDS
	assert set(models).isdisjoint(FORBIDDEN_MODEL_IDS)


def test_channel_config_reuses_controls_and_binds_augmented_extraction(
	channel_raw: Mapping[str, object],
	reference_channel_raw: Mapping[str, object],
	extraction_config: Mapping[str, object],
) -> None:
	models = _mapping(_mapping(channel_raw, 'embeddings'), 'models')
	reference_models = _mapping(
		_mapping(reference_channel_raw, 'embeddings'),
		'models',
	)
	for model_id in REUSED_MODEL_IDS:
		assert models[model_id] == reference_models[model_id]

	extraction = _mapping(extraction_config, 'embeddings')
	assert models[NEW_MODEL_ID] == {
		'dir': extraction['output_dir'],
		'checkpoint': extraction['checkpoint'],
	}
	assert set(models) - set(reference_models) == {NEW_MODEL_ID}
	assert set(models) & set(reference_models) == REUSED_MODEL_IDS


def test_channel_config_reuses_downstream_contract_and_local_runs(
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
		artifact_root / 'channel_benchmark/local_bt_d4_trace_drop_v1/summary'
	)
	assert _paths_do_not_overlap(new_summary, reference_summary)


def test_channel_config_defines_forty_five_conditions_for_one_new_model(
	channel_config: ChannelDecoderConfig,
	channel_raw: Mapping[str, object],
	reference_channel_raw: Mapping[str, object],
) -> None:
	planned_job_count = (
		len(channel_config.models) * len(LAYOUT_IDS) * len(DATA_SIZE_PREFIX)
	)
	models = _mapping(_mapping(channel_raw, 'embeddings'), 'models')
	reference_models = _mapping(
		_mapping(reference_channel_raw, 'embeddings'),
		'models',
	)

	assert planned_job_count == 3 * 5 * 3 == 45
	assert set(models) - set(reference_models) == {NEW_MODEL_ID}


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return child


def _paths_do_not_overlap(left: Path, right: Path) -> bool:
	return left != right and left not in right.parents and right not in left.parents
