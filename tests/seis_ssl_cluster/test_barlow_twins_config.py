from __future__ import annotations

from copy import deepcopy

import pytest

from seis_ssl_cluster.config import (
	resolve_barlow_twins_training_config,
	resolve_mae_training_config,
)


def test_barlow_twins_config_resolves_method_defaults() -> None:
	resolved = resolve_barlow_twins_training_config(_minimal_barlow_config())

	assert resolved['stage'] == 'barlow_twins_training'
	assert resolved['augmentations'] == {'horizontal_flip_probability': 0.5}
	assert resolved['barlow_twins'] == {
		'projector_dim': 384,
		'redundancy_weight': 0.005,
		'normalization_eps': 1.0e-4,
	}
	assert resolved['model']['decoder_depth'] == 1
	assert resolved['model']['name'] == 'amp_mae3d'
	assert 'masking' not in resolved
	assert 'loss' not in resolved
	assert 'continuation' not in resolved


def test_local_barlow_twins_config_preserves_scientific_contract() -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}

	resolved = resolve_barlow_twins_training_config(config)

	assert resolved['barlow_twins'] == {
		'projector_dim': 384,
		'redundancy_weight': 0.005,
		'normalization_eps': 1.0e-4,
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}


def test_local_barlow_twins_config_requires_pair_count() -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {'method': 'local_barlow_twins_3d'}

	with pytest.raises(
		ValueError,
		match=r'barlow_twins\.local_pairs_per_crop.*required',
	):
		resolve_barlow_twins_training_config(config)


@pytest.mark.parametrize('method', [None, 'barlow_twins_3d'])
def test_nonlocal_barlow_twins_config_rejects_pair_count(
	method: str | None,
) -> None:
	config = _minimal_barlow_config()
	barlow_twins: dict[str, object] = {'local_pairs_per_crop': 1}
	if method is not None:
		barlow_twins['method'] = method
	config['barlow_twins'] = barlow_twins

	with pytest.raises(ValueError, match=r'barlow_twins\.local_pairs_per_crop'):
		resolve_barlow_twins_training_config(config)


@pytest.mark.parametrize('local_pairs_per_crop', [0, True, 1.5, 9])
def test_local_barlow_twins_config_rejects_invalid_pair_count(
	local_pairs_per_crop: object,
) -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': local_pairs_per_crop,
	}

	with pytest.raises(ValueError, match=r'barlow_twins\.local_pairs_per_crop'):
		resolve_barlow_twins_training_config(config)


def test_barlow_twins_config_rejects_unknown_method() -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {'method': 'unknown'}

	with pytest.raises(ValueError, match=r'barlow_twins\.method'):
		resolve_barlow_twins_training_config(config)


def test_barlow_twins_config_rejects_non_string_method() -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {'method': True}

	with pytest.raises(TypeError, match=r'barlow_twins\.method must be a string'):
		resolve_barlow_twins_training_config(config)


def test_local_barlow_twins_config_does_not_mutate_raw_mapping() -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	original = deepcopy(config)

	resolve_barlow_twins_training_config(config)

	assert config == original


def test_barlow_twins_config_accepts_continuation() -> None:
	config = _minimal_barlow_config()
	config['continuation'] = {
		'init_checkpoint': '/checkpoints/barlow_twins/latest.pt',
		'unfreeze_top_blocks': 1,
	}

	resolved = resolve_barlow_twins_training_config(config)

	assert resolved['continuation'] == config['continuation']


@pytest.mark.parametrize(
	('continuation', 'message'),
	[
		(
			{
				'init_checkpoint': 'checkpoints/latest.pt',
				'unfreeze_top_blocks': 1,
			},
			r'continuation\.init_checkpoint must be an absolute path',
		),
		(
			{
				'init_checkpoint': '/checkpoints/latest.pt',
				'unfreeze_top_blocks': 1,
				'optimizer': '/checkpoints/optimizer.pt',
			},
			r'continuation key\(s\) not allowed',
		),
		(
			{'unfreeze_top_blocks': 1},
			r'continuation\.init_checkpoint is required',
		),
		(
			{
				'init_checkpoint': '/checkpoints/latest.pt',
				'unfreeze_top_blocks': 0,
			},
			r'continuation\.unfreeze_top_blocks must be a positive integer',
		),
		(
			{
				'init_checkpoint': '/checkpoints/latest.pt',
				'unfreeze_top_blocks': True,
			},
			r'continuation\.unfreeze_top_blocks must be a positive integer',
		),
		(
			{
				'init_checkpoint': '/checkpoints/latest.pt',
				'unfreeze_top_blocks': 2,
			},
			(
				r'continuation\.unfreeze_top_blocks must be less than or equal to '
				r'model\.encoder_depth \(1\)'
			),
		),
	],
)
def test_barlow_twins_config_rejects_invalid_continuation(
	continuation: dict[str, object],
	message: str,
) -> None:
	config = _minimal_barlow_config()
	config['continuation'] = continuation

	with pytest.raises(ValueError, match=message):
		resolve_barlow_twins_training_config(config)


def test_barlow_twins_config_rejects_unknown_nested_key() -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {'unknown': 1}

	with pytest.raises(ValueError, match=r'barlow_twins\.unknown'):
		resolve_barlow_twins_training_config(config)


@pytest.mark.parametrize(
	('section', 'key', 'value'),
	[
		('augmentations', 'horizontal_flip_probability', -0.1),
		('augmentations', 'horizontal_flip_probability', 1.1),
		('barlow_twins', 'projector_dim', 0),
		('barlow_twins', 'redundancy_weight', -0.1),
		('barlow_twins', 'redundancy_weight', float('inf')),
		('barlow_twins', 'normalization_eps', 0.0),
		('barlow_twins', 'normalization_eps', float('nan')),
		('train', 'batch_size', 1),
	],
)
def test_barlow_twins_config_rejects_invalid_method_values(
	section: str,
	key: str,
	value: object,
) -> None:
	config = _minimal_barlow_config()
	config[section] = {**config.get(section, {}), key: value}

	with pytest.raises((TypeError, ValueError), match=key):
		resolve_barlow_twins_training_config(config)


def test_existing_mae_config_resolution_is_unchanged() -> None:
	resolved = resolve_mae_training_config(_minimal_mae_config())

	assert resolved['stage'] == 'train_amp_mae'
	assert resolved['masking']['spatial_mask_mode'] == 'block'
	assert resolved['loss']['valid_mask_mode'] == 'voxel'
	assert resolved['loss']['visible_reconstruction_weight'] == 0.0
	assert 'augmentations' not in resolved
	assert 'barlow_twins' not in resolved


def _minimal_barlow_config() -> dict[str, object]:
	return {
		'paths': {
			'artifact_root': '/artifacts',
			'output_root': '/artifacts/pretraining/barlow_twins',
		},
		'manifests': {
			'train': '/artifacts/manifests/train.json',
			'train_path_list': '/artifacts/splits/train.txt',
		},
		'data': {'local_crop_size': [4, 4, 4]},
		'model': {
			'patch_size': [2, 2, 2],
			'encoder_dim': 12,
			'encoder_depth': 1,
			'encoder_heads': 3,
			'decoder_dim': 12,
			'decoder_depth': 1,
			'decoder_heads': 3,
		},
		'train': {'batch_size': 2, 'samples_per_epoch': 2, 'epochs': 1},
	}


def _minimal_mae_config() -> dict[str, object]:
	config = deepcopy(_minimal_barlow_config())
	config['masking'] = {
		'spatial_mask_ratio': 0.5,
		'block_size_tokens': [1, 1, 1],
	}
	config['loss'] = {
		'reconstruction': 'mse',
		'gradient_weight': 0.0,
		'target_normalization': {'mode': 'none'},
	}
	return config
