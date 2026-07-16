"""Runtime config completion and validation for amplitude MAE training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Literal, cast

from seis_ssl_cluster.config.schema import (
	DEFAULT_MAE_DATA_OPTIONS,
	DEFAULT_MAE_LOSS_OPTIONS,
	DEFAULT_MAE_TRAIN_OPTIONS,
	DEFAULT_ZERO_MASK_CONTRACT,
	EXPECTED_VALID_MASK_MODE,
	FIXED_DATA_CONTRACT,
	FIXED_LOSS_CONTRACT,
	FIXED_MASKING_CONTRACT,
	FIXED_MODEL_CONTRACT,
	STAGE_MAE_TRAINING,
	SUPPORTED_RECONSTRUCTION_LOSSES,
	SUPPORTED_RUNTIME_CHECK_MODES,
	SUPPORTED_TARGET_NORMALIZATION_MODES,
)

if TYPE_CHECKING:
	from pathlib import Path


def _complete_mae_training_config(config: Mapping[str, object]) -> dict[str, object]:
	if not isinstance(config, Mapping):
		msg = f'config must be a mapping; got {config!r}'
		raise TypeError(msg)
	resolved = deepcopy(dict(config))
	stage = resolved.get('stage', STAGE_MAE_TRAINING)
	if stage != STAGE_MAE_TRAINING:
		msg = f'config.stage must be train_amp_mae; got {stage!r}'
		raise ValueError(msg)
	resolved['stage'] = STAGE_MAE_TRAINING
	for section in ('paths', 'manifests', 'data', 'model', 'masking', 'loss', 'train'):
		_runtime_mapping(resolved, section)
	_merge_runtime_defaults(resolved, 'data', DEFAULT_MAE_DATA_OPTIONS)
	_merge_runtime_defaults(resolved, 'train', DEFAULT_MAE_TRAIN_OPTIONS)
	_validate_runtime_check_mode(_runtime_mapping(resolved, 'train'))
	_merge_runtime_defaults(resolved, 'loss', DEFAULT_MAE_LOSS_OPTIONS)
	_merge_runtime_defaults(resolved, 'zero_mask', DEFAULT_ZERO_MASK_CONTRACT)
	_validate_runtime_loss(_runtime_mapping(resolved, 'loss'))
	_merge_runtime_fixed(resolved, 'data', FIXED_DATA_CONTRACT)
	_merge_runtime_fixed(resolved, 'model', FIXED_MODEL_CONTRACT)
	_merge_runtime_fixed(resolved, 'masking', FIXED_MASKING_CONTRACT)
	_merge_runtime_fixed(resolved, 'loss', FIXED_LOSS_CONTRACT)
	return resolved


def _validate_runtime_check_mode(train: Mapping[str, object]) -> None:
	mode = train.get('runtime_check_mode')
	if mode not in SUPPORTED_RUNTIME_CHECK_MODES:
		msg = (
			'train.runtime_check_mode must be one of '
			f'{sorted(SUPPORTED_RUNTIME_CHECK_MODES)!r}; got {mode!r}'
		)
		raise ValueError(msg)


def _merge_runtime_defaults(
	config: dict[str, object],
	section: str,
	defaults: Mapping[str, object],
) -> None:
	current = config.get(section)
	if current is None:
		config[section] = deepcopy(dict(defaults))
		return
	if not isinstance(current, Mapping):
		msg = f'{section} must be a mapping'
		raise TypeError(msg)
	config[section] = {**deepcopy(dict(defaults)), **dict(current)}


def _merge_runtime_fixed(
	config: dict[str, object],
	section: str,
	fixed_values: Mapping[str, object],
) -> None:
	current = _runtime_mapping(config, section)
	for key, fixed_value in fixed_values.items():
		if key in current and current[key] != fixed_value:
			msg = (
				f'{section}.{key} is fixed by the amplitude-only training '
				f'contract; got {current[key]!r}'
			)
			raise ValueError(msg)
	config[section] = {**deepcopy(dict(fixed_values)), **dict(current)}


def _runtime_mapping(
	config: Mapping[str, object],
	section: str,
) -> Mapping[str, object]:
	value = config.get(section)
	if not isinstance(value, Mapping):
		msg = f'{section} must be a mapping'
		raise TypeError(msg)
	return value


def _int_config(parent: Mapping[str, object], key: str, default: int) -> int:
	value = parent.get(key, default)
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	if value <= 0:
		msg = f'{key} must be positive; got {value!r}'
		raise ValueError(msg)
	return value


def _nonnegative_int_config(
	parent: Mapping[str, object],
	key: str,
	default: int,
) -> int:
	value = parent.get(key, default)
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	if value < 0:
		msg = f'{key} must be nonnegative; got {value!r}'
		raise ValueError(msg)
	return value


def _optional_int_config(parent: Mapping[str, object], key: str) -> int | None:
	value = parent.get(key)
	if value is None:
		return None
	return _int_config(parent, key, 1)


def _optional_int_config_with_default(
	parent: Mapping[str, object],
	key: str,
	*,
	default: object,
) -> int | None:
	value = parent.get(key, default)
	if value is None:
		return None
	return _positive_int_value(value, key)


def _optional_any_int_config(parent: Mapping[str, object], key: str) -> int | None:
	value = parent.get(key)
	if value is None:
		return None
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{key} must be an integer or null; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_nonnegative_int_config_with_default(
	parent: Mapping[str, object],
	key: str,
	*,
	default: object,
) -> int | None:
	value = parent.get(key, default)
	if value is None:
		return None
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{key} must be an integer or null; got {value!r}'
		raise TypeError(msg)
	if value < 0:
		msg = f'{key} must be nonnegative; got {value!r}'
		raise ValueError(msg)
	return value


def _positive_int_value(value: object, key: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	if value <= 0:
		msg = f'{key} must be positive; got {value!r}'
		raise ValueError(msg)
	return value


def _float_config(parent: Mapping[str, object], key: str, default: float) -> float:
	value = parent.get(key, default)
	if not isinstance(value, float | int) or isinstance(value, bool):
		msg = f'{key} must be a float; got {value!r}'
		raise TypeError(msg)
	return float(value)


def _runtime_loss_values(
	loss_config: Mapping[str, object],
) -> tuple[
	Literal['huber', 'l1', 'mse'],
	float,
	float,
	float,
	str,
	float | None,
	float | None,
]:
	_validate_runtime_loss(loss_config)
	reconstruction = _loss_mode(loss_config['reconstruction'])
	huber_delta = (
		_required_positive_float_config(
			loss_config,
			'huber_delta',
			label='loss.huber_delta',
		)
		if reconstruction == 'huber'
		else 1.0
	)
	gradient_weight = _required_nonnegative_float_config(
		loss_config,
		'gradient_weight',
		label='loss.gradient_weight',
	)
	visible_reconstruction_weight = _required_nonnegative_float_config(
		loss_config,
		'visible_reconstruction_weight',
		label='loss.visible_reconstruction_weight',
	)
	target_normalization = _required_mapping_config(
		loss_config,
		'target_normalization',
		label='loss.target_normalization',
	)
	mode = _target_normalization_mode(target_normalization.get('mode'))
	eps = (
		_required_positive_float_config(
			target_normalization,
			'eps',
			label='loss.target_normalization.eps',
		)
		if mode == 'patch_zscore'
		else None
	)
	min_std = (
		_required_positive_float_config(
			target_normalization,
			'min_std',
			label='loss.target_normalization.min_std',
		)
		if mode == 'patch_zscore'
		else None
	)
	return (
		reconstruction,
		huber_delta,
		gradient_weight,
		visible_reconstruction_weight,
		mode,
		eps,
		min_std,
	)


def _validate_runtime_loss(loss_config: Mapping[str, object]) -> None:
	reconstruction = _required_config_value(
		loss_config,
		'reconstruction',
		label='loss.reconstruction',
	)
	_loss_mode(reconstruction)
	if reconstruction == 'huber':
		_required_positive_float_config(
			loss_config,
			'huber_delta',
			label='loss.huber_delta',
		)
	elif 'huber_delta' in loss_config:
		msg = 'loss.huber_delta must be omitted unless loss.reconstruction is huber'
		raise ValueError(msg)
	gradient_weight = _required_nonnegative_float_config(
		loss_config,
		'gradient_weight',
		label='loss.gradient_weight',
	)
	_required_nonnegative_float_config(
		loss_config,
		'visible_reconstruction_weight',
		label='loss.visible_reconstruction_weight',
	)
	target_normalization = _required_mapping_config(
		loss_config,
		'target_normalization',
		label='loss.target_normalization',
	)
	mode = _target_normalization_mode(
		_required_config_value(
			target_normalization,
			'mode',
			label='loss.target_normalization.mode',
		),
	)
	if mode == 'none':
		for key in ('eps', 'min_std'):
			if key in target_normalization:
				msg = (
					f'loss.target_normalization.{key} must be omitted '
					"when mode is 'none'"
				)
				raise ValueError(msg)
	else:
		_required_positive_float_config(
			target_normalization,
			'eps',
			label='loss.target_normalization.eps',
		)
		_required_positive_float_config(
			target_normalization,
			'min_std',
			label='loss.target_normalization.min_std',
		)
		if gradient_weight != 0.0:
			msg = (
				'loss.gradient_weight must be 0.0 when '
				"loss.target_normalization.mode is 'patch_zscore'; "
				'the current gradient loss operates in survey-normalized '
				'amplitude space'
			)
			raise ValueError(msg)
	if (
		'valid_mask_mode' in loss_config
		and loss_config.get('valid_mask_mode') != EXPECTED_VALID_MASK_MODE
	):
		msg = "loss.valid_mask_mode must be resolved internally as 'voxel'"
		raise ValueError(msg)


def _required_config_value(
	parent: Mapping[str, object],
	key: str,
	*,
	label: str,
) -> object:
	if key not in parent:
		msg = f'{label} is required'
		raise ValueError(msg)
	return parent[key]


def _required_mapping_config(
	parent: Mapping[str, object],
	key: str,
	*,
	label: str,
) -> Mapping[str, object]:
	value = _required_config_value(parent, key, label=label)
	if not isinstance(value, Mapping):
		msg = f'{label} must be a mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _target_normalization_mode(value: object) -> str:
	if value not in SUPPORTED_TARGET_NORMALIZATION_MODES:
		msg = (
			'loss.target_normalization.mode must be one of '
			f'{sorted(SUPPORTED_TARGET_NORMALIZATION_MODES)!r}; got {value!r}'
		)
		raise ValueError(msg)
	return str(value)


def _required_float_config(
	parent: Mapping[str, object],
	key: str,
	*,
	label: str,
) -> float:
	value = _required_config_value(parent, key, label=label)
	if not isinstance(value, float | int) or isinstance(value, bool):
		msg = f'{label} must be a float; got {value!r}'
		raise TypeError(msg)
	number = float(value)
	if not math.isfinite(number):
		msg = f'{label} must be finite; got {value!r}'
		raise ValueError(msg)
	return number


def _required_positive_float_config(
	parent: Mapping[str, object],
	key: str,
	*,
	label: str,
) -> float:
	value = _required_float_config(parent, key, label=label)
	if value <= 0.0:
		msg = f'{label} must be positive; got {value!r}'
		raise ValueError(msg)
	return value


def _required_nonnegative_float_config(
	parent: Mapping[str, object],
	key: str,
	*,
	label: str,
) -> float:
	value = _required_float_config(parent, key, label=label)
	if value < 0.0:
		msg = f'{label} must be nonnegative; got {value!r}'
		raise ValueError(msg)
	return value


def _positive_float_config(
	parent: Mapping[str, object],
	key: str,
	default: float,
) -> float:
	value = _float_config(parent, key, default)
	if not math.isfinite(value) or value <= 0.0:
		msg = f'{key} must be finite and positive; got {value!r}'
		raise ValueError(msg)
	return value


def _float_pair_config(
	parent: Mapping[str, object],
	key: str,
	*,
	default: tuple[float, float],
) -> tuple[float, float]:
	value = parent.get(key, default)
	if not isinstance(value, list | tuple) or len(value) != 2:
		msg = f'{key} must be a length-2 float sequence; got {value!r}'
		raise TypeError(msg)
	low, high = value
	if (
		not isinstance(low, float | int)
		or isinstance(low, bool)
		or not isinstance(high, float | int)
		or isinstance(high, bool)
	):
		msg = f'{key} must contain numeric percentile values; got {value!r}'
		raise TypeError(msg)
	resolved = (float(low), float(high))
	if not 0.0 <= resolved[0] < resolved[1] <= 100.0:
		msg = f'{key} must satisfy 0 <= low < high <= 100; got {value!r}'
		raise ValueError(msg)
	return resolved


def _optional_positive_float_config(
	parent: Mapping[str, object],
	key: str,
) -> float | None:
	value = parent.get(key)
	if value is None:
		return None
	if not isinstance(value, float | int) or isinstance(value, bool):
		msg = f'{key} must be a float; got {value!r}'
		raise TypeError(msg)
	number = float(value)
	if not math.isfinite(number) or number <= 0.0:
		msg = f'{key} must be finite and positive; got {value!r}'
		raise ValueError(msg)
	return number


def _bool_config(
	parent: Mapping[str, object],
	key: str,
	*,
	default: bool,
) -> bool:
	value = parent.get(key, default)
	if not isinstance(value, bool):
		msg = f'{key} must be a bool; got {value!r}'
		raise TypeError(msg)
	return value


def _str_config_with_default(
	parent: Mapping[str, object],
	key: str,
	default: str,
) -> str:
	value = parent.get(key, default)
	if not isinstance(value, str) or not value:
		msg = f'{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _string_tuple_config(
	parent: Mapping[str, object],
	key: str,
	*,
	default: tuple[str, ...],
) -> tuple[str, ...]:
	value = parent.get(key, default)
	if not isinstance(value, list | tuple) or not value:
		msg = f'{key} must be a non-empty string sequence; got {value!r}'
		raise TypeError(msg)
	resolved: list[str] = []
	for item in value:
		if not isinstance(item, str) or not item:
			msg = f'{key} must contain non-empty strings; got {value!r}'
			raise TypeError(msg)
		resolved.append(item)
	return tuple(resolved)


def _reject_unknown_runtime_keys(
	parent: Mapping[str, object],
	allowed: frozenset[str],
	*,
	prefix: str,
) -> None:
	unexpected = sorted(set(parent) - allowed)
	if unexpected:
		labels = [f'{prefix}.{key}' for key in unexpected]
		msg = (
			f'{prefix} key(s) not allowed: {labels!r}; '
			f'allowed keys are {sorted(allowed)!r}'
		)
		raise ValueError(msg)


def _validate_runtime_output_path_under_root(
	path: Path,
	label: str,
	*,
	root: Path,
	root_label: str,
) -> None:
	if not path.is_absolute():
		msg = f'{label} must be an absolute path; got {path}'
		raise ValueError(msg)
	if not _runtime_path_is_relative_to(path, root):
		msg = f'{label} must be under {root_label} ({root}); got {path}'
		raise ValueError(msg)


def _runtime_path_is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


def _xyz_config(parent: Mapping[str, object], key: str) -> tuple[int, int, int]:
	value = parent.get(key)
	if (
		not isinstance(value, list | tuple)
		or len(value) != 3
		or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
	):
		msg = f'{key} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = tuple(cast('tuple[int, int, int]', value))
	if any(item <= 0 for item in xyz):
		msg = f'{key} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _loss_mode(value: object) -> Literal['huber', 'l1', 'mse']:
	if value not in SUPPORTED_RECONSTRUCTION_LOSSES:
		msg = (
			'loss.reconstruction must be one of '
			f'{sorted(SUPPORTED_RECONSTRUCTION_LOSSES)!r}; got {value!r}'
		)
		raise ValueError(msg)
	return cast('Literal["huber", "l1", "mse"]', value)


__all__ = [
	'_bool_config',
	'_complete_mae_training_config',
	'_float_config',
	'_int_config',
	'_nonnegative_int_config',
	'_optional_int_config',
	'_optional_positive_float_config',
	'_runtime_loss_values',
	'_xyz_config',
]
