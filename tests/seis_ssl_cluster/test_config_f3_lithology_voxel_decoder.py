from __future__ import annotations

import math
from copy import deepcopy

import pytest

from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	f3_lithology_voxel_decoder_config_from_mapping,
)


def _config(tmp_path):
	root = tmp_path / 'artifacts'
	return {
		'paths': {'artifact_root': str(root), 'f3_root': str(tmp_path / 'f3')},
		'dataset': {'name': 'tiny', 'version': 'v1'},
		'model': {'tag': 'encoder-v1', 'freeze_encoder': True},
		'embeddings': {'input_dir': str(root / 'embeddings')},
		'voxel_dataset': {'input_dir': str(root / 'voxel_supervision')},
		'decoder': {
			'spec': 'frozen_embedding_decoder_v1',
			'embedding_dim': 2,
			'class_count': 2,
			'hidden_channels': [4],
			'upsample_factors': [[1, 1, 1]],
		},
		'tiles': {'core_size_tokens': [1, 1, 1], 'context_halo_tokens': [0, 0, 0]},
		'train': {
			'epochs': 1,
			'batch_size': 1,
			'learning_rate': 0.001,
			'weight_decay': 0.0,
			'class_weight': 'balanced',
			'seed': 42,
			'num_workers': 0,
			'amp': False,
			'gradient_clip_norm': 1.0,
		},
		'outputs': {'output_dir': str(root / 'decoder')},
	}


def test_voxel_decoder_config_resolves_strict_settings(tmp_path) -> None:
	config = f3_lithology_voxel_decoder_config_from_mapping(_config(tmp_path))
	assert config.decoder.hidden_channels == (4,)
	assert config.tiles.context_halo_tokens == (0, 0, 0)
	assert config.to_dict()['decoder']['upsample_factors'] == [[1, 1, 1]]


def test_voxel_decoder_config_rejects_unknown_keys(tmp_path) -> None:
	raw = deepcopy(_config(tmp_path))
	raw['train']['early_stopping'] = True
	with pytest.raises(ValueError, match='not allowed'):
		f3_lithology_voxel_decoder_config_from_mapping(raw)


def test_voxel_decoder_config_requires_frozen_encoder(tmp_path) -> None:
	raw = deepcopy(_config(tmp_path))
	raw['model']['freeze_encoder'] = False
	with pytest.raises(ValueError, match='freeze_encoder'):
		f3_lithology_voxel_decoder_config_from_mapping(raw)


@pytest.mark.parametrize(
	('setting', 'value'),
	[
		('learning_rate', math.nan),
		('learning_rate', math.inf),
		('weight_decay', math.nan),
		('weight_decay', math.inf),
		('gradient_clip_norm', math.nan),
		('gradient_clip_norm', math.inf),
	],
)
def test_voxel_decoder_config_rejects_nonfinite_train_numbers(
	tmp_path, setting: str, value: float
) -> None:
	raw = deepcopy(_config(tmp_path))
	raw['train'][setting] = value
	with pytest.raises(ValueError, match=rf'train\.{setting}.*finite'):
		f3_lithology_voxel_decoder_config_from_mapping(raw)
