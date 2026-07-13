from __future__ import annotations

from copy import deepcopy

import pytest

from seis_ssl_cluster.config.f3_lithology_voxel_inference import (
	f3_lithology_voxel_inference_config_from_mapping,
)


def _config(tmp_path):
	root = tmp_path / 'artifacts'
	return {
		'paths': {'artifact_root': str(root), 'f3_root': str(tmp_path / 'f3')},
		'dataset': {'name': 'tiny', 'version': 'v1'},
		'model': {'tag': 'encoder-v1', 'freeze_encoder': True},
		'labels': {'class_info': str(root / 'class_info.json')},
		'embeddings': {'input_dir': str(root / 'embeddings')},
		'decoder': {'checkpoint': str(root / 'decoder' / 'best.pt')},
		'tiles': {
			'core_size_tokens': [2, 3, 4],
			'context_halo_tokens': [1, 0, 2],
		},
		'inference': {'write_probabilities': True, 'overwrite': False},
		'outputs': {'output_dir': str(root / 'predictions')},
	}


def test_voxel_inference_config_resolves_strict_settings(tmp_path) -> None:
	config = f3_lithology_voxel_inference_config_from_mapping(_config(tmp_path))

	assert config.tiles.core_size_tokens == (2, 3, 4)
	assert config.tiles.context_halo_tokens == (1, 0, 2)
	assert config.write_probabilities is True
	assert config.output_paths.predictions.name == 'f3_voxel_predictions.npy'


def test_voxel_inference_config_rejects_unknown_keys(tmp_path) -> None:
	raw = deepcopy(_config(tmp_path))
	raw['inference']['skip_existing'] = True

	with pytest.raises(ValueError, match='not allowed'):
		f3_lithology_voxel_inference_config_from_mapping(raw)


def test_voxel_inference_config_requires_frozen_encoder(tmp_path) -> None:
	raw = deepcopy(_config(tmp_path))
	raw['model']['freeze_encoder'] = False

	with pytest.raises(ValueError, match='freeze_encoder'):
		f3_lithology_voxel_inference_config_from_mapping(raw)
