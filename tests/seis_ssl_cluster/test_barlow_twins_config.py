from __future__ import annotations

from copy import deepcopy

import pytest

from seis_ssl_cluster.config import (
	resolve_barlow_twins_training_config,
	resolve_mae_training_config,
)
from seis_ssl_cluster.config.schema import (
	DEFAULT_BARLOW_TWINS_AUGMENTATION_OPTIONS,
	HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
	HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_POLICY,
	HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_POLICY,
	IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
	OVERLAPPING_SUBCROP_XY_AUGMENTATION_POLICY,
	XY_D4_TRACE_DROP_AUGMENTATION_POLICY,
)


def _d4_augmentations() -> dict[str, object]:
	return {
		'policy': XY_D4_TRACE_DROP_AUGMENTATION_POLICY,
		'reflection_probability': 0.5,
		'trace_drop_probability': 0.02,
	}


def _gaussian_noise_augmentations() -> dict[str, object]:
	return {
		'policy': HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
		'horizontal_flip_probability': 0.5,
		'gaussian_noise_std': 0.05,
	}


def _horizontal_trace_drop_augmentations() -> dict[str, object]:
	return {
		'policy': HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_POLICY,
		'horizontal_flip_probability': 0.5,
		'trace_drop_probability': 0.1,
	}


def _zero_phase_z_filter_augmentations() -> dict[str, object]:
	return {
		'policy': HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_POLICY,
		'horizontal_flip_probability': 0.5,
		'z_filter_side_weight': 0.125,
	}


def _identity_gaussian_noise_augmentations() -> dict[str, object]:
	return {
		'policy': IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
		'gaussian_noise_std': 0.05,
	}


def _overlapping_subcrop_augmentations() -> dict[str, object]:
	return {
		'policy': OVERLAPPING_SUBCROP_XY_AUGMENTATION_POLICY,
		'horizontal_flip_probability': 0.5,
		'max_subcrop_shift_tokens': [4, 4, 0],
	}


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


def test_legacy_augmentation_resolution_is_unchanged() -> None:
	config = _minimal_barlow_config()
	config['augmentations'] = {'horizontal_flip_probability': 0.25}

	resolved = resolve_barlow_twins_training_config(config)

	assert DEFAULT_BARLOW_TWINS_AUGMENTATION_OPTIONS == {
		'horizontal_flip_probability': 0.5
	}
	assert resolved['augmentations'] == {'horizontal_flip_probability': 0.25}


def test_d4_trace_drop_augmentation_resolves_exact_mapping() -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	config['augmentations'] = _d4_augmentations()

	resolved = resolve_barlow_twins_training_config(config)

	assert resolved['augmentations'] == {
		'policy': XY_D4_TRACE_DROP_AUGMENTATION_POLICY,
		'reflection_probability': 0.5,
		'trace_drop_probability': 0.02,
	}
	assert 'horizontal_flip_probability' not in resolved['augmentations']


def test_overlapping_subcrop_augmentation_resolves_exact_mapping() -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 128,
	}
	config['augmentations'] = _overlapping_subcrop_augmentations()
	data = config['data']
	model = config['model']
	assert isinstance(data, dict)
	assert isinstance(model, dict)
	data['local_crop_size'] = [128, 128, 128]
	model['patch_size'] = [8, 8, 8]

	resolved = resolve_barlow_twins_training_config(config)

	assert resolved['augmentations'] == _overlapping_subcrop_augmentations()


@pytest.mark.parametrize(
	'augmentations',
	[
		{
			'policy': OVERLAPPING_SUBCROP_XY_AUGMENTATION_POLICY,
			'horizontal_flip_probability': 0.5,
		},
		{**_overlapping_subcrop_augmentations(), 'unknown': True},
	],
)
def test_overlapping_subcrop_rejects_nonexact_contract(
	augmentations: dict[str, object],
) -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 1,
	}
	config['augmentations'] = augmentations

	with pytest.raises((TypeError, ValueError), match='augmentations'):
		resolve_barlow_twins_training_config(config)


def test_global_barlow_rejects_overlapping_subcrop_policy() -> None:
	config = _minimal_barlow_config()
	config['augmentations'] = {
		**_overlapping_subcrop_augmentations(),
		'max_subcrop_shift_tokens': [1, 1, 0],
	}

	with pytest.raises(ValueError, match=r'requires barlow_twins\.method'):
		resolve_barlow_twins_training_config(config)


@pytest.mark.parametrize(
	('max_shift_tokens', 'message'),
	[
		([-1, 1, 0], 'nonnegative'),
		([1, 0, 1], 'Z shift'),
		([0, 0, 0], 'X or Y shift'),
		([2, 0, 0], 'less than the view token shape'),
	],
)
def test_overlapping_subcrop_rejects_invalid_shift_contract(
	max_shift_tokens: list[int],
	message: str,
) -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 1,
	}
	config['augmentations'] = {
		'policy': OVERLAPPING_SUBCROP_XY_AUGMENTATION_POLICY,
		'horizontal_flip_probability': 0.5,
		'max_subcrop_shift_tokens': max_shift_tokens,
	}

	with pytest.raises(ValueError, match=message):
		resolve_barlow_twins_training_config(config)


def test_overlapping_subcrop_rejects_insufficient_minimum_overlap() -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 3,
	}
	config['augmentations'] = {
		'policy': OVERLAPPING_SUBCROP_XY_AUGMENTATION_POLICY,
		'horizontal_flip_probability': 0.5,
		'max_subcrop_shift_tokens': [1, 1, 0],
	}

	with pytest.raises(ValueError, match='minimum overlapping token count'):
		resolve_barlow_twins_training_config(config)


def test_gaussian_noise_augmentation_resolves_exact_mapping() -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	config['augmentations'] = _gaussian_noise_augmentations()

	resolved = resolve_barlow_twins_training_config(config)

	assert resolved['augmentations'] == {
		'policy': HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
		'horizontal_flip_probability': 0.5,
		'gaussian_noise_std': 0.05,
	}


def test_horizontal_trace_drop_resolves_exact_mapping_without_square_xy() -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	config['augmentations'] = _horizontal_trace_drop_augmentations()
	data = config['data']
	model = config['model']
	assert isinstance(data, dict)
	assert isinstance(model, dict)
	data['local_crop_size'] = [4, 6, 4]
	model['patch_size'] = [1, 2, 2]

	resolved = resolve_barlow_twins_training_config(config)

	assert resolved['augmentations'] == {
		'policy': HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_POLICY,
		'horizontal_flip_probability': 0.5,
		'trace_drop_probability': 0.1,
	}


def test_zero_phase_z_filter_resolves_exact_mapping_without_square_xy() -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	config['augmentations'] = _zero_phase_z_filter_augmentations()
	data = config['data']
	model = config['model']
	assert isinstance(data, dict)
	assert isinstance(model, dict)
	data['local_crop_size'] = [4, 6, 4]
	model['patch_size'] = [1, 2, 2]

	resolved = resolve_barlow_twins_training_config(config)

	assert resolved['augmentations'] == {
		'policy': HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_POLICY,
		'horizontal_flip_probability': 0.5,
		'z_filter_side_weight': 0.125,
	}


@pytest.mark.parametrize(
	'augmentations',
	[
		{
			'policy': HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_POLICY,
			'horizontal_flip_probability': 0.5,
		},
		{
			**_zero_phase_z_filter_augmentations(),
			'trace_drop_probability': 0.02,
		},
	],
)
def test_zero_phase_z_filter_rejects_nonexact_contract(
	augmentations: dict[str, object],
) -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	config['augmentations'] = augmentations

	with pytest.raises((TypeError, ValueError), match='augmentations'):
		resolve_barlow_twins_training_config(config)


@pytest.mark.parametrize(
	('key', 'value'),
	[
		('horizontal_flip_probability', -0.1),
		('z_filter_side_weight', 0.0),
		('z_filter_side_weight', 0.5),
		('z_filter_side_weight', float('nan')),
	],
)
def test_zero_phase_z_filter_rejects_invalid_contract(
	key: str,
	value: object,
) -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	config['augmentations'] = {
		**_zero_phase_z_filter_augmentations(),
		key: value,
	}

	with pytest.raises((TypeError, ValueError), match=key):
		resolve_barlow_twins_training_config(config)


def test_zero_phase_z_filter_requires_local_barlow_twins_method() -> None:
	config = _minimal_barlow_config()
	config['augmentations'] = _zero_phase_z_filter_augmentations()

	with pytest.raises(ValueError, match=r'requires barlow_twins\.method'):
		resolve_barlow_twins_training_config(config)


@pytest.mark.parametrize(
	'augmentations',
	[
		{
			'policy': HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_POLICY,
			'horizontal_flip_probability': 0.5,
		},
		{
			**_horizontal_trace_drop_augmentations(),
			'gaussian_noise_std': 0.05,
		},
		{**_horizontal_trace_drop_augmentations(), 'unknown': True},
	],
)
def test_horizontal_trace_drop_rejects_nonexact_contract(
	augmentations: dict[str, object],
) -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	config['augmentations'] = augmentations

	with pytest.raises((TypeError, ValueError), match='augmentations'):
		resolve_barlow_twins_training_config(config)


@pytest.mark.parametrize(
	('key', 'value'),
	[
		('horizontal_flip_probability', -0.1),
		('horizontal_flip_probability', True),
		('trace_drop_probability', 1.1),
		('trace_drop_probability', float('nan')),
	],
)
def test_horizontal_trace_drop_rejects_invalid_probability(
	key: str,
	value: object,
) -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	config['augmentations'] = {
		**_horizontal_trace_drop_augmentations(),
		key: value,
	}

	with pytest.raises((TypeError, ValueError), match=key):
		resolve_barlow_twins_training_config(config)


def test_global_barlow_rejects_horizontal_trace_drop_policy() -> None:
	config = _minimal_barlow_config()
	config['augmentations'] = _horizontal_trace_drop_augmentations()

	with pytest.raises(ValueError, match=r'requires barlow_twins\.method'):
		resolve_barlow_twins_training_config(config)


@pytest.mark.parametrize(
	'augmentations',
	[
		{
			'policy': HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
			'horizontal_flip_probability': 0.5,
		},
		{
			**_gaussian_noise_augmentations(),
			'trace_drop_probability': 0.02,
		},
		{
			**_gaussian_noise_augmentations(),
			'unknown': True,
		},
	],
)
def test_gaussian_noise_augmentation_rejects_nonexact_contract(
	augmentations: dict[str, object],
) -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	config['augmentations'] = augmentations

	with pytest.raises((TypeError, ValueError), match='augmentations'):
		resolve_barlow_twins_training_config(config)


@pytest.mark.parametrize(
	('key', 'value'),
	[
		('horizontal_flip_probability', -0.1),
		('horizontal_flip_probability', 1.1),
		('horizontal_flip_probability', float('inf')),
		('horizontal_flip_probability', True),
		('gaussian_noise_std', -0.1),
		('gaussian_noise_std', float('inf')),
		('gaussian_noise_std', float('nan')),
		('gaussian_noise_std', True),
	],
)
def test_gaussian_noise_augmentation_rejects_invalid_value(
	key: str,
	value: object,
) -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	config['augmentations'] = {**_gaussian_noise_augmentations(), key: value}

	with pytest.raises((TypeError, ValueError), match=key):
		resolve_barlow_twins_training_config(config)


def test_global_barlow_rejects_gaussian_noise_policy() -> None:
	config = _minimal_barlow_config()
	config['augmentations'] = _gaussian_noise_augmentations()

	with pytest.raises(ValueError, match=r'requires barlow_twins\.method'):
		resolve_barlow_twins_training_config(config)


def test_identity_gaussian_noise_augmentation_resolves_exact_mapping() -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	config['augmentations'] = _identity_gaussian_noise_augmentations()

	resolved = resolve_barlow_twins_training_config(config)

	assert resolved['augmentations'] == {
		'policy': IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
		'gaussian_noise_std': 0.05,
	}
	assert 'horizontal_flip_probability' not in resolved['augmentations']


@pytest.mark.parametrize(
	'augmentations',
	[
		{'policy': IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_POLICY},
		{
			**_identity_gaussian_noise_augmentations(),
			'horizontal_flip_probability': 0.0,
		},
		{**_identity_gaussian_noise_augmentations(), 'unknown': True},
	],
)
def test_identity_gaussian_noise_rejects_nonexact_mapping(
	augmentations: dict[str, object],
) -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	config['augmentations'] = augmentations

	with pytest.raises((TypeError, ValueError), match='augmentations'):
		resolve_barlow_twins_training_config(config)


@pytest.mark.parametrize(
	'gaussian_noise_std',
	[0.0, -0.1, float('inf'), float('nan'), True, '0.05'],
)
def test_identity_gaussian_noise_requires_positive_finite_std(
	gaussian_noise_std: object,
) -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	config['augmentations'] = {
		**_identity_gaussian_noise_augmentations(),
		'gaussian_noise_std': gaussian_noise_std,
	}

	with pytest.raises(ValueError, match='gaussian_noise_std'):
		resolve_barlow_twins_training_config(config)


def test_global_barlow_rejects_identity_gaussian_noise_policy() -> None:
	config = _minimal_barlow_config()
	config['augmentations'] = _identity_gaussian_noise_augmentations()

	with pytest.raises(ValueError, match=r'requires barlow_twins\.method'):
		resolve_barlow_twins_training_config(config)


@pytest.mark.parametrize(
	'augmentations',
	[
		{
			**_d4_augmentations(),
			'policy': 'unknown',
		},
		{
			**_d4_augmentations(),
			'horizontal_flip_probability': 0.5,
		},
		{
			**_d4_augmentations(),
			'unknown': True,
		},
		{
			'policy': XY_D4_TRACE_DROP_AUGMENTATION_POLICY,
			'reflection_probability': 0.5,
		},
	],
)
def test_d4_trace_drop_augmentation_rejects_nonexact_contract(
	augmentations: dict[str, object],
) -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	config['augmentations'] = augmentations

	with pytest.raises((TypeError, ValueError), match='augmentations'):
		resolve_barlow_twins_training_config(config)


@pytest.mark.parametrize(
	('key', 'value'),
	[
		('reflection_probability', -0.1),
		('reflection_probability', 1.1),
		('reflection_probability', float('inf')),
		('trace_drop_probability', -0.1),
		('trace_drop_probability', 1.1),
		('trace_drop_probability', float('nan')),
		('trace_drop_probability', True),
	],
)
def test_d4_trace_drop_augmentation_rejects_invalid_probability(
	key: str,
	value: object,
) -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	config['augmentations'] = {**_d4_augmentations(), key: value}

	with pytest.raises((TypeError, ValueError), match=key):
		resolve_barlow_twins_training_config(config)


def test_global_barlow_rejects_d4_trace_drop_policy() -> None:
	config = _minimal_barlow_config()
	config['augmentations'] = _d4_augmentations()

	with pytest.raises(ValueError, match=r'requires barlow_twins\.method'):
		resolve_barlow_twins_training_config(config)


@pytest.mark.parametrize(
	('section', 'shape', 'message'),
	[
		('data', [4, 6, 4], r'data\.local_crop_size X/Y'),
		('model', [1, 2, 2], r'model\.patch_size X/Y'),
	],
)
def test_d4_trace_drop_policy_rejects_non_square_xy(
	section: str,
	shape: list[int],
	message: str,
) -> None:
	config = _minimal_barlow_config()
	config['barlow_twins'] = {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 8,
	}
	config['augmentations'] = _d4_augmentations()
	section_mapping = config[section]
	assert isinstance(section_mapping, dict)
	section_mapping['local_crop_size' if section == 'data' else 'patch_size'] = shape

	with pytest.raises(ValueError, match=message):
		resolve_barlow_twins_training_config(config)


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
