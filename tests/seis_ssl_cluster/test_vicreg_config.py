from __future__ import annotations

from copy import deepcopy

import pytest

from seis_ssl_cluster.config import (
	resolve_barlow_twins_training_config,
	resolve_vicreg_training_config,
)
from seis_ssl_cluster.config.validate import validate_config
from seis_ssl_cluster.config.vicreg import (
	resolve_vicreg_pretraining_method,
	vicreg_config_compatibility_identity,
)


def test_minimal_config_resolves_defaults_and_fixed_contract() -> None:
	resolved = resolve_vicreg_training_config(_minimal_vicreg_config())

	assert resolved['stage'] == 'vicreg_training'
	assert resolved['vicreg'] == {
		'projector_dim': 384,
		'invariance_weight': 25.0,
		'variance_weight': 25.0,
		'covariance_weight': 1.0,
		'variance_target_std': 1.0,
		'variance_eps': 1.0e-4,
		'method': 'local_vicreg_3d',
		'local_pairs_per_crop': 8,
	}
	assert resolved['augmentations'] == {'horizontal_flip_probability': 0.5}
	assert resolved['data']['grid_order'] == ['x', 'y', 'z']
	assert resolved['data']['input_channels'] == 1
	assert resolved['model']['name'] == 'amp_mae3d'
	assert resolved['model']['in_channels'] == 1
	assert resolved['model']['out_channels'] == 1
	assert resolved['zero_mask']['enabled'] is True
	assert 'continuation' not in resolved


def test_full_config_preserves_vicreg_values() -> None:
	config = _minimal_vicreg_config()
	config['vicreg'] = {
		'method': 'local_vicreg_3d',
		'local_pairs_per_crop': 4,
		'projector_dim': 96,
		'invariance_weight': 10.0,
		'variance_weight': 11.0,
		'covariance_weight': 2.0,
		'variance_target_std': 0.75,
		'variance_eps': 1.0e-3,
	}

	resolved = resolve_vicreg_training_config(config)

	assert resolved['vicreg'] == config['vicreg']
	assert resolve_vicreg_pretraining_method(resolved) == 'local_vicreg_3d'
	assert vicreg_config_compatibility_identity(resolved) == config['vicreg']


def test_stage_dispatch_resolves_vicreg_config() -> None:
	resolved = validate_config(_minimal_vicreg_config(), stage='vicreg_training')

	assert resolved['stage'] == 'vicreg_training'


@pytest.mark.parametrize('method', ['vicreg_3d', 'vicregl_3d', '', 1, True])
def test_rejects_non_local_method(method: object) -> None:
	config = _minimal_vicreg_config()
	config['vicreg'] = {
		'method': method,
		'local_pairs_per_crop': 8,
	}

	with pytest.raises((TypeError, ValueError), match=r'vicreg\.method'):
		resolve_vicreg_training_config(config)


@pytest.mark.parametrize('missing', ['method', 'local_pairs_per_crop'])
def test_rejects_missing_required_vicreg_field(missing: str) -> None:
	config = _minimal_vicreg_config()
	vicreg = config['vicreg']
	assert isinstance(vicreg, dict)
	vicreg.pop(missing)

	with pytest.raises(ValueError, match=missing):
		resolve_vicreg_training_config(config)


def test_rejects_unknown_top_level_and_section_keys() -> None:
	top_level = _minimal_vicreg_config()
	top_level['objective'] = {}
	with pytest.raises(ValueError, match='top-level section'):
		resolve_vicreg_training_config(top_level)

	nested = _minimal_vicreg_config()
	vicreg = nested['vicreg']
	assert isinstance(vicreg, dict)
	vicreg['unknown'] = 1
	with pytest.raises(ValueError, match=r'vicreg\.unknown'):
		resolve_vicreg_training_config(nested)


@pytest.mark.parametrize(
	('key', 'value'),
	[
		('local_pairs_per_crop', 0),
		('local_pairs_per_crop', True),
		('projector_dim', 0),
		('projector_dim', 1.5),
		('invariance_weight', -1.0),
		('invariance_weight', float('inf')),
		('variance_weight', True),
		('covariance_weight', float('nan')),
		('variance_target_std', 0.0),
		('variance_target_std', float('inf')),
		('variance_eps', 0.0),
		('variance_eps', True),
	],
)
def test_rejects_invalid_vicreg_values(key: str, value: object) -> None:
	config = _minimal_vicreg_config()
	vicreg = config['vicreg']
	assert isinstance(vicreg, dict)
	vicreg[key] = value

	with pytest.raises((TypeError, ValueError), match=key):
		resolve_vicreg_training_config(config)


def test_rejects_more_local_pairs_than_crop_tokens() -> None:
	config = _minimal_vicreg_config()
	vicreg = config['vicreg']
	assert isinstance(vicreg, dict)
	vicreg['local_pairs_per_crop'] = 9

	with pytest.raises(ValueError, match='crop token count'):
		resolve_vicreg_training_config(config)


@pytest.mark.parametrize('unfreeze_top_blocks', [0, 1])
def test_resolves_optional_continuation(unfreeze_top_blocks: int) -> None:
	config = _minimal_vicreg_config()
	config['continuation'] = {
		'init_checkpoint': '/checkpoints/vicreg/latest.pt',
		'unfreeze_top_blocks': unfreeze_top_blocks,
	}

	resolved = resolve_vicreg_training_config(config)

	assert resolved['continuation'] == config['continuation']


@pytest.mark.parametrize(
	('continuation', 'message'),
	[
		(
			{'init_checkpoint': '', 'unfreeze_top_blocks': 1},
			'init_checkpoint',
		),
		(
			{'init_checkpoint': 'relative.pt', 'unfreeze_top_blocks': 1},
			'absolute path',
		),
		(
			{
				'init_checkpoint': '/checkpoints/vicreg/latest.pt',
				'unfreeze_top_blocks': -1,
			},
			'unfreeze_top_blocks',
		),
		(
			{
				'init_checkpoint': '/checkpoints/vicreg/latest.pt',
				'unfreeze_top_blocks': 2,
			},
			'encoder_depth',
		),
	],
)
def test_rejects_invalid_continuation(
	continuation: dict[str, object],
	message: str,
) -> None:
	config = _minimal_vicreg_config()
	config['continuation'] = continuation

	with pytest.raises((TypeError, ValueError), match=message):
		resolve_vicreg_training_config(config)


def test_resolution_does_not_modify_source_mapping() -> None:
	config = _minimal_vicreg_config()
	expected = deepcopy(config)

	resolve_vicreg_training_config(config)

	assert config == expected


def test_common_contract_matches_local_barlow_twins() -> None:
	vicreg_raw = _minimal_vicreg_config()
	barlow_raw = deepcopy(vicreg_raw)
	barlow_raw['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	barlow_raw.pop('vicreg')

	vicreg = resolve_vicreg_training_config(vicreg_raw)
	barlow = resolve_barlow_twins_training_config(barlow_raw)

	for section in (
		'paths',
		'manifests',
		'data',
		'zero_mask',
		'model',
		'augmentations',
		'train',
	):
		assert vicreg[section] == barlow[section]
	assert barlow['stage'] == 'barlow_twins_training'
	assert barlow['barlow_twins'] == {
		'projector_dim': 384,
		'redundancy_weight': 0.005,
		'normalization_eps': 1.0e-4,
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}


def _minimal_vicreg_config() -> dict[str, object]:
	return {
		'paths': {
			'artifact_root': '/artifacts',
			'output_root': '/artifacts/pretraining/vicreg',
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
		'vicreg': {
			'method': 'local_vicreg_3d',
			'local_pairs_per_crop': 8,
		},
		'train': {'batch_size': 2, 'samples_per_epoch': 2, 'epochs': 1},
	}
