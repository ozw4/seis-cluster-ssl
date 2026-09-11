'''Tests for the Volve horizon five-way configuration contract.'''

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.volve.horizon_five_way_config import (
	FIVE_WAY_BENCHMARK_ID,
	FIVE_WAY_HMM_K,
	FIVE_WAY_MODEL_IDS,
	FIVE_WAY_RANDOM_SEED,
	FIVE_WAY_STAGE1_EPOCHS,
	FIVE_WAY_STAGE2_EPOCHS,
	FIVE_WAY_UNFREEZE_TOP_BLOCKS,
	FIVE_WAY_WITHIN2_BENCHMARK_ID,
	FIVE_WAY_WITHIN4_BENCHMARK_ID,
	LOCAL_BARLOW_TWINS_METHOD,
	LOCAL_BARLOW_TWINS_PAIRS_PER_CROP,
	volve_horizon_five_way_config_from_mapping,
)
from seis_ssl_cluster.volve.horizon_frozen import (
	FROZEN_CONDITION_COUNT,
	enumerate_frozen_horizon_conditions,
)
from seis_ssl_cluster.volve.horizon_runner import (
	CHECKPOINT_SELECTION_VALIDATION_MAE,
	CHECKPOINT_SELECTION_VALIDATION_WITHIN_2,
	CHECKPOINT_SELECTION_VALIDATION_WITHIN_4,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
	REPOSITORY_ROOT
	/ 'experiments/volve/horizon_benchmark_v1'
	/ '31_mae_local_bt_hmm_five_way_v1'
)
DISTILL010_ROOT = EXPERIMENT_ROOT / '60_distill010'


def test_resolves_exact_mapping_in_code_order(tmp_path: Path) -> None:
	raw = _config(tmp_path)
	raw['models'] = dict(reversed(list(raw['models'].items())))
	config = volve_horizon_five_way_config_from_mapping(raw)

	assert config.model_ids == FIVE_WAY_MODEL_IDS
	assert config.model_by_id('local_barlow_twins').expected == {
		'objective': LOCAL_BARLOW_TWINS_METHOD,
		'local_pairs_per_crop': LOCAL_BARLOW_TWINS_PAIRS_PER_CROP,
		'stratigraphy_pretext': False,
	}
	assert config.train.seed == 42000
	assert config.tiles.patch_size_xyz == (8, 8, 8)
	assert FIVE_WAY_HMM_K == 6
	assert FIVE_WAY_RANDOM_SEED == 42
	assert FIVE_WAY_STAGE1_EPOCHS == 100
	assert FIVE_WAY_STAGE2_EPOCHS == 25
	assert FIVE_WAY_UNFREEZE_TOP_BLOCKS == 1
	assert config.checkpoint_selection == CHECKPOINT_SELECTION_VALIDATION_MAE
	assert config.checkpoint_selections == (CHECKPOINT_SELECTION_VALIDATION_MAE,)
	assert config.benchmark_id == FIVE_WAY_BENCHMARK_ID


def test_resolves_explicit_bundled_selections_and_benchmark_id(
	tmp_path: Path,
) -> None:
	raw = _config(tmp_path)
	raw['checkpoint_selections'] = [
		CHECKPOINT_SELECTION_VALIDATION_MAE,
		CHECKPOINT_SELECTION_VALIDATION_WITHIN_2,
		CHECKPOINT_SELECTION_VALIDATION_WITHIN_4,
	]
	raw['benchmark_id'] = 'mae_local_bt_hmm_distill010_v1'

	config = volve_horizon_five_way_config_from_mapping(raw)

	assert config.checkpoint_selections == tuple(raw['checkpoint_selections'])
	assert config.benchmark_id == 'mae_local_bt_hmm_distill010_v1'


@pytest.mark.parametrize(
	('checkpoint_selections', 'message'),
	[
		([], 'must not be empty'),
		([CHECKPOINT_SELECTION_VALIDATION_WITHIN_2], 'must include the primary'),
		(
			[
				CHECKPOINT_SELECTION_VALIDATION_MAE,
				CHECKPOINT_SELECTION_VALIDATION_MAE,
			],
			'must be unique',
		),
	],
)
def test_rejects_invalid_bundled_checkpoint_selections(
	tmp_path: Path,
	checkpoint_selections: list[str],
	message: str,
) -> None:
	raw = _config(tmp_path)
	raw['checkpoint_selections'] = checkpoint_selections

	with pytest.raises(ValueError, match=message):
		volve_horizon_five_way_config_from_mapping(raw)


def test_repository_configs_resolve_distinct_selection_and_output_identity(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str((tmp_path / 'artifacts').resolve())
	)
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_VOLVE_ROOT', str((tmp_path / 'volve').resolve())
	)
	mae = volve_horizon_five_way_config_from_mapping(
		load_config(EXPERIMENT_ROOT / '50_five_way.yaml')
	)
	within2 = volve_horizon_five_way_config_from_mapping(
		load_config(EXPERIMENT_ROOT / '51_five_way_within2.yaml')
	)
	within4 = volve_horizon_five_way_config_from_mapping(
		load_config(EXPERIMENT_ROOT / '52_five_way_within4.yaml')
	)

	assert mae.checkpoint_selection == CHECKPOINT_SELECTION_VALIDATION_MAE
	assert within2.checkpoint_selection == CHECKPOINT_SELECTION_VALIDATION_WITHIN_2
	assert within4.checkpoint_selection == CHECKPOINT_SELECTION_VALIDATION_WITHIN_4
	assert mae.benchmark_id == FIVE_WAY_BENCHMARK_ID
	assert within2.benchmark_id == FIVE_WAY_WITHIN2_BENCHMARK_ID
	assert within4.benchmark_id == FIVE_WAY_WITHIN4_BENCHMARK_ID
	assert len({mae.runs_root, within2.runs_root, within4.runs_root}) == 3
	assert len({mae.summary_root, within2.summary_root, within4.summary_root}) == 3
	assert 'mae_local_bt_hmm_five_way_within4_v1/runs' in str(
		within4.runs_root
	)
	assert 'mae_local_bt_hmm_five_way_within4_v1/summary' in str(
		within4.summary_root
	)


def test_distill010_configs_bind_weight_and_shared_checkpoint_bundle(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	artifact_root = (tmp_path / 'artifacts').resolve()
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(artifact_root))
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_VOLVE_ROOT', str((tmp_path / 'volve').resolve())
	)
	for name in (
		'30_stage2/mae100_hmm_k6_25ep.yaml',
		'30_stage2/local_bt100_hmm_k6_25ep.yaml',
	):
		stage2 = load_config(DISTILL010_ROOT / name)
		assert stage2['loss']['distillation_weight'] == 0.1
		assert stage2['train']['epochs'] == 25
		assert stage2['train']['samples_per_epoch'] == 10_000
		assert 'mae_local_bt_hmm_five_way_distill010_v1' in stage2['paths'][
			'output_root'
		]

	configs = [
		volve_horizon_five_way_config_from_mapping(
			load_config(DISTILL010_ROOT / name)
		)
		for name in (
			'50_five_way_macro_mae.yaml',
			'51_five_way_within2.yaml',
			'52_five_way_within4.yaml',
		)
	]
	assert {config.benchmark_id for config in configs} == {
		'mae_local_bt_hmm_five_way_distill010_v1'
	}
	assert len({config.runs_root for config in configs}) == 1
	assert len({config.summary_root for config in configs}) == 3
	assert all(len(config.checkpoint_selections) == 3 for config in configs)
	assert [config.checkpoint_selection for config in configs] == [
		CHECKPOINT_SELECTION_VALIDATION_MAE,
		CHECKPOINT_SELECTION_VALIDATION_WITHIN_2,
		CHECKPOINT_SELECTION_VALIDATION_WITHIN_4,
	]


def test_rejects_unknown_checkpoint_selection(tmp_path: Path) -> None:
	raw = _config(tmp_path)
	raw['checkpoint_selection'] = 'unknown'

	with pytest.raises(ValueError, match='unknown horizon checkpoint selection'):
		volve_horizon_five_way_config_from_mapping(raw)


@pytest.mark.parametrize('mode', ['missing', 'extra'])
def test_rejects_model_key_drift(tmp_path: Path, mode: str) -> None:
	raw = _config(tmp_path)
	models = raw['models']
	if mode == 'missing':
		models.pop('mae_hmm_k6')
	else:
		models['foreign'] = deepcopy(models['mae'])

	with pytest.raises(ValueError, match='exactly the five fixed model IDs'):
		volve_horizon_five_way_config_from_mapping(raw)


@pytest.mark.parametrize('field', ['checkpoint', 'embeddings_dir'])
def test_rejects_duplicate_model_paths(tmp_path: Path, field: str) -> None:
	raw = _config(tmp_path)
	raw['models']['mae_hmm_k6'][field] = raw['models']['mae'][field]

	with pytest.raises(ValueError, match='must be distinct'):
		volve_horizon_five_way_config_from_mapping(raw)


@pytest.mark.parametrize('field', ['checkpoint', 'embeddings_dir'])
def test_rejects_trace_drop_artifact_paths(tmp_path: Path, field: str) -> None:
	raw = _config(tmp_path)
	raw['models']['local_barlow_twins'][field] = str(
		tmp_path / 'artifacts' / 'trace_drop' / field
	)

	with pytest.raises(ValueError, match='trace-drop'):
		volve_horizon_five_way_config_from_mapping(raw)


def test_rejects_equal_or_non_artifact_output_roots(tmp_path: Path) -> None:
	raw = _config(tmp_path)
	raw['outputs']['summary_root'] = raw['outputs']['runs_root']
	with pytest.raises(ValueError, match='must differ'):
		volve_horizon_five_way_config_from_mapping(raw)

	raw = _config(tmp_path / 'outside')
	raw['outputs']['summary_root'] = str((tmp_path / 'elsewhere').resolve())
	with pytest.raises(ValueError, match=r'below paths\.artifact_root'):
		volve_horizon_five_way_config_from_mapping(raw)


def test_rejects_output_below_public_volve_root(tmp_path: Path) -> None:
	raw = _config(tmp_path)
	artifact_root = Path(raw['paths']['artifact_root'])
	raw['paths']['volve_root'] = str(artifact_root / 'public')
	raw['outputs']['summary_root'] = str(artifact_root / 'public' / 'summary')

	with pytest.raises(ValueError, match=r'public paths\.volve_root'):
		volve_horizon_five_way_config_from_mapping(raw)


def test_model_lookup_rejects_unknown_id(tmp_path: Path) -> None:
	config = volve_horizon_five_way_config_from_mapping(_config(tmp_path))

	assert config.model_by_id('random').model_id == 'random'
	with pytest.raises(ValueError, match='unknown Volve horizon five-way model'):
		config.model_by_id('pretrained')


def test_legacy_two_way_condition_contract_is_unchanged() -> None:
	conditions = enumerate_frozen_horizon_conditions()

	assert len(conditions) == FROZEN_CONDITION_COUNT == 30
	assert {condition[0] for condition in conditions} == {'pretrained', 'random'}


def _config(tmp_path: Path) -> dict[str, object]:
	artifact_root = (tmp_path / 'artifacts').resolve()
	public_root = (tmp_path / 'public').resolve()
	models = {
		model_id: {
			'checkpoint': str(artifact_root / 'checkpoints' / model_id / 'latest.pt'),
			'embeddings_dir': str(artifact_root / 'embeddings' / model_id),
		}
		for model_id in FIVE_WAY_MODEL_IDS
	}
	return {
		'paths': {
			'artifact_root': str(artifact_root),
			'volve_root': str(public_root),
		},
		'dataset': {'survey_id': 'volve_st10010'},
		'inputs': {
			'canonical_input_metadata': str(
				artifact_root / 'data' / 'volve_canonical_input_metadata.json'
			),
		},
		'models': models,
		'outputs': {
			'runs_root': str(artifact_root / 'five_way' / 'runs'),
			'summary_root': str(artifact_root / 'five_way' / 'summary'),
		},
		'decoder': {
			'embedding_dim': 384,
			'class_count': 5,
			'hidden_channels': [128, 64, 32],
			'upsample_factors': [[2, 2, 2]] * 3,
			'upsample_mode': 'nearest',
			'normalization': 'voxelwise_layer_norm',
		},
		'tiles': {
			'patch_size': [8, 8, 8],
			'core_size_tokens': [8, 8, 27],
			'context_halo_tokens': [1, 1, 0],
			'window_start': 552,
			'window_stop': 768,
			'min_token_valid_fraction': 1.0,
		},
		'train': {
			'epochs': 50,
			'batch_size': 1,
			'learning_rate': 1.0e-3,
			'weight_decay': 1.0e-4,
			'sampling_mode': 'all_tiles_once',
			'seed': 42000,
			'amp': True,
			'gradient_clip_norm': 1.0,
		},
	}
