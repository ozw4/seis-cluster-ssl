"""Build amplitude encoders from supported pretraining checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch

from seis_ssl_cluster.config.barlow_twins import (
	resolve_barlow_twins_pretraining_method,
)
from seis_ssl_cluster.config.pretraining import (
	resolve_barlow_twins_training_config,
)
from seis_ssl_cluster.config.schema import (
	BARLOW_TWINS_PRETRAINING_METHOD,
	FIXED_DATA_CONTRACT,
	FIXED_MODEL_CONTRACT,
	LOCAL_BARLOW_TWINS_PRETRAINING_METHOD,
	STAGE_BARLOW_TWINS_TRAINING,
	STAGE_MAE_TRAINING,
	STAGE_STRAT_HMM_PRETEXT_TRAINING,
)
from seis_ssl_cluster.models.mae import AmplitudeMAE3D

PATCH_PROJECTION_PARAMETER_PREFIX = 'patch_projection.'
ENCODER_PARAMETER_PREFIX = 'encoder.'
AMPLITUDE_ENCODER_TRAINED_PARAMETER_PREFIXES = (
	PATCH_PROJECTION_PARAMETER_PREFIX,
	ENCODER_PARAMETER_PREFIX,
)
BARLOW_TWINS_CHECKPOINT_KIND = 'barlow_twins_pretraining'

_MAE_ALLOWED_TOP_LEVEL = frozenset(
	{
		'stage',
		'paths',
		'manifests',
		'data',
		'model',
		'continuation',
		'masking',
		'loss',
		'train',
		'zero_mask',
		'visualization',
	}
)
_MAE_REQUIRED_TOP_LEVEL = frozenset(
	{
		'stage',
		'paths',
		'manifests',
		'data',
		'model',
		'masking',
		'loss',
		'train',
		'zero_mask',
	}
)
_BARLOW_TWINS_ALLOWED_TOP_LEVEL = frozenset(
	{
		'stage',
		'paths',
		'manifests',
		'data',
		'zero_mask',
		'model',
		'continuation',
		'augmentations',
		'barlow_twins',
		'train',
	}
)
_BARLOW_TWINS_REQUIRED_TOP_LEVEL = frozenset(
	{
		'stage',
		'paths',
		'manifests',
		'data',
		'zero_mask',
		'model',
		'augmentations',
		'barlow_twins',
		'train',
	}
)


def build_model_from_config(config: Mapping[str, object]) -> AmplitudeMAE3D:
	"""Instantiate ``AmplitudeMAE3D`` from supported checkpoint geometry."""
	validate_pretraining_checkpoint_config(config)
	model = _required_mapping(config, 'model')
	_validate_model_contract(model)
	return AmplitudeMAE3D(
		in_channels=_positive_int(model.get('in_channels'), 'model.in_channels'),
		out_channels=_positive_int(model.get('out_channels'), 'model.out_channels'),
		patch_size_xyz=_positive_xyz(model.get('patch_size'), 'model.patch_size'),
		encoder_dim=_positive_int(model.get('encoder_dim'), 'model.encoder_dim'),
		encoder_depth=_positive_int(
			model.get('encoder_depth'),
			'model.encoder_depth',
		),
		encoder_heads=_positive_int(
			model.get('encoder_heads'),
			'model.encoder_heads',
		),
		decoder_dim=_positive_int(model.get('decoder_dim'), 'model.decoder_dim'),
		decoder_depth=_positive_int(
			model.get('decoder_depth'),
			'model.decoder_depth',
		),
		decoder_heads=_positive_int(
			model.get('decoder_heads'),
			'model.decoder_heads',
		),
	)


def build_model_from_checkpoint_payload(
	payload: Mapping[str, object],
) -> AmplitudeMAE3D:
	"""Instantiate and strictly load a supported bare amplitude checkpoint."""
	config = checkpoint_config_from_payload(payload)
	state_dict = model_state_dict_from_payload(payload)
	model = build_model_from_config(config)
	model.to(dtype=_checkpoint_floating_dtype(state_dict))
	model.load_state_dict(state_dict, strict=True)
	return model


def checkpoint_config_from_payload(
	payload: Mapping[str, object],
) -> Mapping[str, object]:
	"""Return a validated method-aware resolved checkpoint config."""
	value = payload.get('config')
	if not isinstance(value, Mapping):
		raise TypeError('checkpoint is missing a resolved config')
	config = cast('Mapping[str, object]', value)
	validate_pretraining_checkpoint_config(config)
	_validate_method_identity(payload, config)
	_validate_bare_model_state(payload, config)
	return config


def validate_pretraining_checkpoint_config(config: Mapping[str, object]) -> None:
	"""Validate strict top-level sections for a supported pretraining method."""
	stage = config.get('stage')
	if stage == STAGE_MAE_TRAINING:
		_validate_top_level(
			config,
			allowed=_MAE_ALLOWED_TOP_LEVEL,
			required=_MAE_REQUIRED_TOP_LEVEL,
		)
	elif stage == STAGE_BARLOW_TWINS_TRAINING:
		_validate_top_level(
			config,
			allowed=_BARLOW_TWINS_ALLOWED_TOP_LEVEL,
			required=_BARLOW_TWINS_REQUIRED_TOP_LEVEL,
		)
		_validate_resolved_barlow_twins_config(config)
	else:
		raise ValueError(
			'checkpoint config.stage must identify a supported pretraining method; '
			f'got {stage!r}'
		)
	for section in ('paths', 'manifests', 'data', 'model', 'train', 'zero_mask'):
		_required_mapping(config, section)
	_validate_model_contract(_required_mapping(config, 'model'))


def model_state_dict_from_payload(
	payload: Mapping[str, object],
) -> Mapping[str, torch.Tensor]:
	"""Return the checkpoint's bare ``AmplitudeMAE3D`` state mapping."""
	value = payload.get('model_state_dict')
	if not isinstance(value, Mapping):
		raise TypeError('checkpoint is missing model_state_dict')
	invalid_keys = sorted(repr(key) for key in value if not isinstance(key, str))
	if invalid_keys:
		raise TypeError(
			'checkpoint model_state_dict keys must be strings; '
			f'got invalid key(s): {invalid_keys!r}'
		)
	state_dict = cast('Mapping[str, torch.Tensor]', value)
	invalid_values = sorted(
		str(key)
		for key, tensor in state_dict.items()
		if not isinstance(tensor, torch.Tensor)
	)
	if invalid_values:
		raise TypeError(
			'checkpoint model_state_dict values must be tensors; '
			f'got invalid key(s): {invalid_values!r}'
		)
	return state_dict


def is_random_encoder_checkpoint(payload: Mapping[str, object]) -> bool:
	"""Return whether a payload declares the supported random baseline contract."""
	training_state = payload.get('training_state')
	metadata = payload.get('metadata')
	return (
		isinstance(training_state, Mapping)
		and training_state.get('checkpoint_kind') == 'random_init'
		and isinstance(metadata, Mapping)
		and metadata.get('random_encoder_baseline') is True
		and metadata.get('pretrained_weights_loaded') is False
	)


def _is_strat_hmm_encoder_checkpoint(payload: Mapping[str, object]) -> bool:
	stratigraphy_config = payload.get('stratigraphy_config')
	training_state = payload.get('training_state')
	return (
		isinstance(stratigraphy_config, Mapping)
		and isinstance(training_state, Mapping)
		and training_state.get('stage') == STAGE_STRAT_HMM_PRETEXT_TRAINING
	)


def _validate_top_level(
	config: Mapping[str, object],
	*,
	allowed: frozenset[str],
	required: frozenset[str],
) -> None:
	unexpected = sorted(set(config) - allowed)
	if unexpected:
		raise ValueError(
			f'checkpoint config has unsupported top-level key(s): {unexpected!r}'
		)
	missing = sorted(required - set(config))
	if missing:
		raise ValueError(
			f'checkpoint config is missing resolved section(s): {missing!r}'
		)


def _validate_resolved_barlow_twins_config(config: Mapping[str, object]) -> None:
	raw = dict(config)
	raw.pop('stage')
	for section, fixed_contract in (
		('data', FIXED_DATA_CONTRACT),
		('model', FIXED_MODEL_CONTRACT),
	):
		resolved_section = _required_mapping(config, section)
		raw[section] = {
			key: value
			for key, value in resolved_section.items()
			if key not in fixed_contract
		}
	resolved = resolve_barlow_twins_training_config(raw)
	if resolved != dict(config):
		raise ValueError(
			'Barlow Twins checkpoint config must contain the fully resolved config'
		)


def _validate_method_identity(
	payload: Mapping[str, object],
	config: Mapping[str, object],
) -> None:
	if config.get('stage') != STAGE_BARLOW_TWINS_TRAINING:
		return
	if is_random_encoder_checkpoint(payload):
		return
	if _is_strat_hmm_encoder_checkpoint(payload):
		return
	expected_method = resolve_barlow_twins_pretraining_method(config)
	if payload.get('pretraining_method') != expected_method:
		raise ValueError(
			'Barlow Twins checkpoint pretraining_method does not match config'
		)
	if payload.get('checkpoint_kind') != BARLOW_TWINS_CHECKPOINT_KIND:
		raise ValueError('Barlow Twins checkpoint checkpoint_kind is invalid')
	prefixes = payload.get('trained_parameter_prefixes')
	if not isinstance(prefixes, list | tuple) or tuple(prefixes) != (
		AMPLITUDE_ENCODER_TRAINED_PARAMETER_PREFIXES
	):
		raise ValueError(
			'Barlow Twins checkpoint trained_parameter_prefixes are invalid'
		)
	if not isinstance(payload.get('projector_state_dict'), Mapping):
		raise TypeError(
			'Barlow Twins checkpoint projector_state_dict must be a mapping'
		)


def _validate_bare_model_state(
	payload: Mapping[str, object],
	config: Mapping[str, object],
) -> None:
	state_dict = model_state_dict_from_payload(payload)
	wrapper_keys = sorted(
		key
		for key in state_dict
		if key.startswith(('backbone.', 'projector.'))
	)
	if wrapper_keys:
		method = (
			'Barlow Twins'
			if config.get('stage') == STAGE_BARLOW_TWINS_TRAINING
			else 'MAE'
		)
		raise ValueError(
			f'{method} model_state_dict must use bare AmplitudeMAE3D keys; '
			f'got wrapper key(s): {wrapper_keys!r}'
		)


def _checkpoint_floating_dtype(
	state_dict: Mapping[str, torch.Tensor],
) -> torch.dtype:
	dtypes = {
		tensor.dtype
		for tensor in state_dict.values()
		if tensor.is_floating_point()
	}
	if not dtypes:
		raise ValueError(
			'checkpoint model_state_dict does not contain floating point tensors'
		)
	if len(dtypes) != 1:
		raise ValueError(
			f'checkpoint model_state_dict has multiple floating dtypes: {dtypes!r}'
		)
	return next(iter(dtypes))


def _validate_model_contract(model: Mapping[str, object]) -> None:
	if model.get('name') != FIXED_MODEL_CONTRACT['name']:
		raise ValueError("checkpoint model.name must be 'amp_mae3d'")
	if model.get('in_channels') != FIXED_MODEL_CONTRACT['in_channels']:
		raise ValueError('checkpoint model.in_channels must be 1')
	if model.get('out_channels') != FIXED_MODEL_CONTRACT['out_channels']:
		raise ValueError('checkpoint model.out_channels must be 1')


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'checkpoint config.{key} must be a mapping')
	return cast('Mapping[str, object]', value)


def _positive_int(value: object, label: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int):
		raise TypeError(f'{label} must be an integer; got {value!r}')
	if value <= 0:
		raise ValueError(f'{label} must be positive; got {value!r}')
	return value


def _positive_xyz(value: object, label: str) -> tuple[int, int, int]:
	if not isinstance(value, list | tuple) or len(value) != 3:
		raise TypeError(f'{label} must be a length-3 sequence')
	return cast(
		'tuple[int, int, int]',
		tuple(
			_positive_int(item, f'{label}[{index}]')
			for index, item in enumerate(value)
		),
	)


__all__ = [
	'AMPLITUDE_ENCODER_TRAINED_PARAMETER_PREFIXES',
	'BARLOW_TWINS_CHECKPOINT_KIND',
	'BARLOW_TWINS_PRETRAINING_METHOD',
	'ENCODER_PARAMETER_PREFIX',
	'LOCAL_BARLOW_TWINS_PRETRAINING_METHOD',
	'PATCH_PROJECTION_PARAMETER_PREFIX',
	'build_model_from_checkpoint_payload',
	'build_model_from_config',
	'checkpoint_config_from_payload',
	'is_random_encoder_checkpoint',
	'model_state_dict_from_payload',
	'validate_pretraining_checkpoint_config',
]
