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
	'36_channel_hmm_transition_balance_v1'
)
EMBEDDING_ROOT = EXPERIMENT_ROOT / '30_embeddings'
CHANNEL_CONFIG = EXPERIMENT_ROOT / '40_channel_transition_balance.yaml'
FINAL_CHANNEL_CONFIG = EXPERIMENT_ROOT / '41_channel_transition_balance_final.yaml'
MAE_EXTRACTION_REFERENCE = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'32_channel_ssl_hmm_four_way_v1/'
	'03_extract_mae_hmm_k6_embeddings.yaml'
)
LOCAL_BT_EXTRACTION_REFERENCE = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'33_channel_mae_local_bt_four_way_v1/'
	'02_extract_local_barlow_twins_hmm_k6_embeddings.yaml'
)
CHANNEL_REFERENCE = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'33_channel_mae_local_bt_four_way_v1/'
	'03_channel_mae_local_bt_four_way.yaml'
)
EXTRACTION_SPECS = {
	'mae_hmm_k6_neutral': (
		'01_extract_mae_hmm_k6_neutral.yaml',
		'mae100',
		'neutral',
		'mae',
	),
	'mae_hmm_k6_persist003': (
		'02_extract_mae_hmm_k6_persist003.yaml',
		'mae100',
		'persist003',
		'mae',
	),
	'mae_hmm_k6_persist010': (
		'03_extract_mae_hmm_k6_persist010.yaml',
		'mae100',
		'persist010',
		'mae',
	),
	'local_barlow_twins_hmm_k6_neutral': (
		'04_extract_local_barlow_twins_hmm_k6_neutral.yaml',
		'local_bt100',
		'neutral',
		'local_barlow_twins',
	),
	'local_barlow_twins_hmm_k6_persist003': (
		'05_extract_local_barlow_twins_hmm_k6_persist003.yaml',
		'local_bt100',
		'persist003',
		'local_barlow_twins',
	),
	'local_barlow_twins_hmm_k6_persist010': (
		'06_extract_local_barlow_twins_hmm_k6_persist010.yaml',
		'local_bt100',
		'persist010',
		'local_barlow_twins',
	),
}
MODEL_IDS = (
	'mae',
	'mae_hmm_k6',
	'mae_hmm_k6_neutral',
	'mae_hmm_k6_persist003',
	'mae_hmm_k6_persist010',
	'local_barlow_twins',
	'local_barlow_twins_hmm_k6',
	'local_barlow_twins_hmm_k6_neutral',
	'local_barlow_twins_hmm_k6_persist003',
	'local_barlow_twins_hmm_k6_persist010',
)
NEW_MODEL_IDS = (
	'mae_hmm_k6_neutral',
	'mae_hmm_k6_persist003',
	'mae_hmm_k6_persist010',
	'local_barlow_twins_hmm_k6_neutral',
	'local_barlow_twins_hmm_k6_persist003',
	'local_barlow_twins_hmm_k6_persist010',
)
REUSED_MODEL_IDS = (
	'mae',
	'mae_hmm_k6',
	'local_barlow_twins',
	'local_barlow_twins_hmm_k6',
)
FORBIDDEN_MODEL_IDS = {
	'barlow_twins',
	'barlow_twins_hmm_k6',
	'local_barlow_twins_d4_trace_drop',
	'random',
}
EXPECTED_PHASE1_CONDITIONS = {
	('mae_hmm_k6_neutral', 'layout_000', 'medium'),
	('mae_hmm_k6_neutral', 'layout_001', 'medium'),
	('mae_hmm_k6_neutral', 'layout_002', 'medium'),
	('mae_hmm_k6_neutral', 'layout_003', 'medium'),
	('mae_hmm_k6_neutral', 'layout_004', 'medium'),
	('mae_hmm_k6_persist003', 'layout_000', 'medium'),
	('mae_hmm_k6_persist003', 'layout_001', 'medium'),
	('mae_hmm_k6_persist003', 'layout_002', 'medium'),
	('mae_hmm_k6_persist003', 'layout_003', 'medium'),
	('mae_hmm_k6_persist003', 'layout_004', 'medium'),
	('mae_hmm_k6_persist010', 'layout_000', 'medium'),
	('mae_hmm_k6_persist010', 'layout_001', 'medium'),
	('mae_hmm_k6_persist010', 'layout_002', 'medium'),
	('mae_hmm_k6_persist010', 'layout_003', 'medium'),
	('mae_hmm_k6_persist010', 'layout_004', 'medium'),
	('local_barlow_twins_hmm_k6_neutral', 'layout_000', 'medium'),
	('local_barlow_twins_hmm_k6_neutral', 'layout_001', 'medium'),
	('local_barlow_twins_hmm_k6_neutral', 'layout_002', 'medium'),
	('local_barlow_twins_hmm_k6_neutral', 'layout_003', 'medium'),
	('local_barlow_twins_hmm_k6_neutral', 'layout_004', 'medium'),
	('local_barlow_twins_hmm_k6_persist003', 'layout_000', 'medium'),
	('local_barlow_twins_hmm_k6_persist003', 'layout_001', 'medium'),
	('local_barlow_twins_hmm_k6_persist003', 'layout_002', 'medium'),
	('local_barlow_twins_hmm_k6_persist003', 'layout_003', 'medium'),
	('local_barlow_twins_hmm_k6_persist003', 'layout_004', 'medium'),
	('local_barlow_twins_hmm_k6_persist010', 'layout_000', 'medium'),
	('local_barlow_twins_hmm_k6_persist010', 'layout_001', 'medium'),
	('local_barlow_twins_hmm_k6_persist010', 'layout_002', 'medium'),
	('local_barlow_twins_hmm_k6_persist010', 'layout_003', 'medium'),
	('local_barlow_twins_hmm_k6_persist010', 'layout_004', 'medium'),
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
			load_config(EMBEDDING_ROOT / spec[0])
		)
		for model_id, spec in EXTRACTION_SPECS.items()
	}


@pytest.fixture
def reference_extractions(
	artifact_root: Path,
) -> dict[str, dict[str, object]]:
	del artifact_root
	return {
		'mae': resolve_embedding_extraction_config(
			load_config(MAE_EXTRACTION_REFERENCE)
		),
		'local_barlow_twins': resolve_embedding_extraction_config(
			load_config(LOCAL_BT_EXTRACTION_REFERENCE)
		),
	}


@pytest.fixture
def channel_raw(artifact_root: Path) -> dict[str, object]:
	del artifact_root
	return load_config(CHANNEL_CONFIG)


@pytest.fixture
def reference_channel_raw(artifact_root: Path) -> dict[str, object]:
	del artifact_root
	return load_config(CHANNEL_REFERENCE)


@pytest.fixture
def final_channel_raw(artifact_root: Path) -> dict[str, object]:
	del artifact_root
	return load_config(FINAL_CHANNEL_CONFIG)


@pytest.fixture
def channel_config(channel_raw: Mapping[str, object]) -> ChannelDecoderConfig:
	return channel_decoder_config_from_mapping(channel_raw)


def test_exact_six_embedding_extraction_configs_are_present() -> None:
	expected_files = tuple(spec[0] for spec in EXTRACTION_SPECS.values())
	actual_files = tuple(path.name for path in sorted(EMBEDDING_ROOT.iterdir()))

	assert actual_files == expected_files
	assert CHANNEL_CONFIG.is_file()
	assert FINAL_CHANNEL_CONFIG.is_file()


def test_extractions_resolve_to_exact_full_stage2_checkpoints_and_outputs(
	extraction_configs: Mapping[str, Mapping[str, object]],
	artifact_root: Path,
) -> None:
	checkpoint_root = (
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'hmm_transition_balance_v1'
	)
	output_root = (
		artifact_root
		/ 'embeddings/parihaka/facies_benchmark_v1'
		/ 'hmm_transition_balance_v1'
	)
	checkpoints: set[Path] = set()
	outputs: set[Path] = set()

	for model_id, (_, source, variant, _) in EXTRACTION_SPECS.items():
		config = extraction_configs[model_id]
		embeddings = _mapping(config, 'embeddings')
		checkpoint = Path(str(embeddings['checkpoint']))
		output_dir = Path(str(embeddings['output_dir']))

		assert config['stage'] == 'extract_embeddings'
		assert checkpoint == checkpoint_root / source / variant / 'full_25ep/latest.pt'
		assert checkpoint.name == 'latest.pt'
		assert 'best.pt' not in checkpoint.parts
		assert 'stage1' not in checkpoint.parts
		assert not any('feasibility' in part for part in checkpoint.parts)
		assert output_dir == output_root / model_id / 'overlap_x64'
		checkpoints.add(checkpoint)
		outputs.add(output_dir)

	assert len(checkpoints) == len(outputs) == len(NEW_MODEL_IDS) == 6


def test_extractions_differ_from_source_hmm_references_only_by_destinations(
	extraction_configs: Mapping[str, Mapping[str, object]],
	reference_extractions: Mapping[str, Mapping[str, object]],
) -> None:
	for model_id, (_, _, _, reference_id) in EXTRACTION_SPECS.items():
		config = extraction_configs[model_id]
		reference = reference_extractions[reference_id]
		config_controls = {
			key: value for key, value in config.items() if key != 'embeddings'
		}
		reference_controls = {
			key: value for key, value in reference.items() if key != 'embeddings'
		}

		assert set(config) == set(reference)
		assert config_controls == reference_controls
		assert set(_mapping(config, 'embeddings')) == {
			'checkpoint',
			'output_dir',
		}


def test_channel_config_binds_models_in_required_order_without_forbidden_models(
	channel_raw: Mapping[str, object],
	channel_config: ChannelDecoderConfig,
) -> None:
	models = _mapping(_mapping(channel_raw, 'embeddings'), 'models')

	assert tuple(models) == MODEL_IDS
	assert tuple(channel_config.models) == MODEL_IDS
	assert set(models).isdisjoint(FORBIDDEN_MODEL_IDS)


def test_channel_config_reuses_controls_and_binds_candidate_extractions(
	channel_raw: Mapping[str, object],
	reference_channel_raw: Mapping[str, object],
	extraction_configs: Mapping[str, Mapping[str, object]],
) -> None:
	models = _mapping(_mapping(channel_raw, 'embeddings'), 'models')
	reference_models = _mapping(
		_mapping(reference_channel_raw, 'embeddings'),
		'models',
	)

	for model_id in REUSED_MODEL_IDS:
		assert models[model_id] == reference_models[model_id]
	for model_id in NEW_MODEL_IDS:
		extraction = _mapping(extraction_configs[model_id], 'embeddings')
		assert models[model_id] == {
			'dir': extraction['output_dir'],
			'checkpoint': extraction['checkpoint'],
		}

	assert set(models) - set(reference_models) == set(NEW_MODEL_IDS)
	assert set(models) & set(reference_models) == set(REUSED_MODEL_IDS)


def test_channel_configs_reuse_downstream_contract_and_isolate_run_modes(
	channel_raw: Mapping[str, object],
	final_channel_raw: Mapping[str, object],
	reference_channel_raw: Mapping[str, object],
	channel_config: ChannelDecoderConfig,
	artifact_root: Path,
) -> None:
	inputs = _mapping(channel_raw, 'inputs')
	outputs = _mapping(channel_raw, 'outputs')
	reference_inputs = _mapping(reference_channel_raw, 'inputs')
	reference_outputs = _mapping(reference_channel_raw, 'outputs')
	final_inputs = _mapping(final_channel_raw, 'inputs')
	final_outputs = _mapping(final_channel_raw, 'outputs')

	assert channel_raw['dataset'] == reference_channel_raw['dataset']
	assert {
		key: value for key, value in inputs.items() if key != 'runs_root'
	} == {key: value for key, value in reference_inputs.items() if key != 'runs_root'}
	for section in ('decoder', 'tiles', 'train'):
		assert channel_raw[section] == reference_channel_raw[section]
		assert final_channel_raw[section] == channel_raw[section]
	assert final_channel_raw['dataset'] == channel_raw['dataset']
	assert final_channel_raw['embeddings'] == channel_raw['embeddings']

	assert set(outputs) == set(reference_outputs)
	assert inputs['runs_root'] == outputs['runs_root']
	assert final_inputs['runs_root'] == final_outputs['runs_root']
	assert channel_config.runs_root == (
		artifact_root
		/ 'channel_benchmark/hmm_transition_balance_v1/validation_runs'
	)
	final_config = channel_decoder_config_from_mapping(final_channel_raw)
	assert final_config.runs_root == (
		artifact_root / 'channel_benchmark/hmm_transition_balance_v1/final_runs'
	)
	new_summary = Path(str(outputs['output_dir']))
	reference_summary = Path(str(reference_outputs['output_dir']))
	assert new_summary == (
		artifact_root / 'channel_benchmark/hmm_transition_balance_v1/summary'
	)
	assert _paths_do_not_overlap(new_summary, reference_summary)
	assert _paths_do_not_overlap(new_summary, channel_config.runs_root)
	assert _paths_do_not_overlap(channel_config.runs_root, final_config.runs_root)


def test_candidate_model_ids_checkpoints_and_embedding_outputs_are_unique(
	channel_raw: Mapping[str, object],
) -> None:
	models = _mapping(_mapping(channel_raw, 'embeddings'), 'models')
	new_sources = [_mapping(models, model_id) for model_id in NEW_MODEL_IDS]
	reused_sources = [_mapping(models, model_id) for model_id in REUSED_MODEL_IDS]
	new_checkpoints = {source['checkpoint'] for source in new_sources}
	new_outputs = {source['dir'] for source in new_sources}
	reused_checkpoints = {source['checkpoint'] for source in reused_sources}
	reused_outputs = {source['dir'] for source in reused_sources}

	assert len(set(NEW_MODEL_IDS)) == len(NEW_MODEL_IDS) == 6
	assert len(new_checkpoints) == len(new_outputs) == 6
	assert new_checkpoints.isdisjoint(reused_checkpoints)
	assert new_outputs.isdisjoint(reused_outputs)


def test_phase1_explicitly_selects_thirty_new_model_medium_conditions(
	channel_config: ChannelDecoderConfig,
) -> None:
	phase1_conditions = {
		(model_id, layout_id, data_size)
		for model_id in channel_config.models
		if model_id in NEW_MODEL_IDS
		for layout_id in LAYOUT_IDS
		for data_size in DATA_SIZE_PREFIX
		if data_size == 'medium'
	}
	full_config_job_count = (
		len(channel_config.models) * len(LAYOUT_IDS) * len(DATA_SIZE_PREFIX)
	)

	assert tuple(LAYOUT_IDS) == (
		'layout_000',
		'layout_001',
		'layout_002',
		'layout_003',
		'layout_004',
	)
	assert tuple(DATA_SIZE_PREFIX) == ('small', 'medium', 'large')
	assert phase1_conditions == EXPECTED_PHASE1_CONDITIONS
	assert len(phase1_conditions) == 6 * 5 == 30
	assert full_config_job_count == 10 * 5 * 3 == 150


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return child


def _paths_do_not_overlap(left: Path, right: Path) -> bool:
	return left != right and left not in right.parents and right not in left.parents
