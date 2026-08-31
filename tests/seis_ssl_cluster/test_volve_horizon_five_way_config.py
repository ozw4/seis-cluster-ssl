'''Tests for the Volve horizon five-way configuration contract.'''

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.volve.horizon_five_way_config import (
	FIVE_WAY_HMM_K,
	FIVE_WAY_MODEL_IDS,
	FIVE_WAY_RANDOM_SEED,
	FIVE_WAY_STAGE1_EPOCHS,
	FIVE_WAY_STAGE2_EPOCHS,
	FIVE_WAY_UNFREEZE_TOP_BLOCKS,
	LOCAL_BARLOW_TWINS_METHOD,
	LOCAL_BARLOW_TWINS_PAIRS_PER_CROP,
	volve_horizon_five_way_config_from_mapping,
)
from seis_ssl_cluster.volve.horizon_frozen import (
	FROZEN_CONDITION_COUNT,
	enumerate_frozen_horizon_conditions,
)


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
