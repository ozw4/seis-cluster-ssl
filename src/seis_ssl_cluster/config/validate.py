"""Explicit validation for amplitude-only seismic SSL clustering configs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import TypeAlias, TypeVar

from seis_ssl_cluster.config.schema import (
	EXPECTED_GRID_ORDER,
	EXPECTED_INPUT_CHANNELS,
	EXPECTED_TARGET_CHANNELS,
	EXPECTED_USE_CONTEXT,
	EXPECTED_VOLUME_FORMAT,
	KNOWN_STAGES,
	LEGACY_ATTRIBUTE_KEY_NAMES,
	LEGACY_ATTRIBUTE_KEY_PATHS,
)

Config: TypeAlias = dict[str, object]
_T = TypeVar('_T', bound=Mapping[str, object])


def validate_config(config: _T) -> _T:
	"""Validate an amplitude-only MVP configuration and return it unchanged."""
	if not isinstance(config, Mapping):
		msg = 'config must be a mapping'
		raise TypeError(msg)

	_reject_legacy_attribute_config(config)
	stage = _validate_stage(config)

	paths = _required_mapping(config, 'paths')
	artifact_root = _validate_absolute_path(
		paths,
		'artifact_root',
		prefix='paths',
	)
	nopims_root = _validate_absolute_path(
		paths,
		'nopims_root',
		prefix='paths',
	)
	_validate_generated_metadata_paths(
		config,
		stage,
		artifact_root=artifact_root,
		nopims_root=nopims_root,
	)

	data = _required_mapping(config, 'data')
	_validate_equal(data, 'grid_order', EXPECTED_GRID_ORDER, prefix='data')
	_validate_equal(data, 'volume_format', EXPECTED_VOLUME_FORMAT, prefix='data')
	_validate_equal(data, 'input_channels', EXPECTED_INPUT_CHANNELS, prefix='data')
	_validate_equal(data, 'target_channels', EXPECTED_TARGET_CHANNELS, prefix='data')
	_validate_equal(data, 'use_context', EXPECTED_USE_CONTEXT, prefix='data')
	local_crop_size = _validate_positive_int_triplet(
		data,
		'local_crop_size',
		prefix='data',
	)

	model = _required_mapping(config, 'model')
	_validate_equal(model, 'in_channels', EXPECTED_INPUT_CHANNELS, prefix='model')
	_validate_equal(model, 'out_channels', EXPECTED_TARGET_CHANNELS, prefix='model')
	patch_size = _validate_positive_int_triplet(
		model,
		'patch_size',
		prefix='model',
	)
	_validate_divisible_crop_patch(local_crop_size, patch_size)

	masking = _required_mapping(config, 'masking')
	_validate_masking(masking)

	train = _required_mapping(config, 'train')
	_validate_train(train)

	if 'loss' in config:
		_validate_loss(_required_mapping(config, 'loss'))

	return config


def _validate_stage(config: Mapping[str, object]) -> str:
	stage = config.get('stage')
	if stage not in KNOWN_STAGES:
		msg = f'stage must be one of {sorted(KNOWN_STAGES)!r}; got {stage!r}'
		raise ValueError(msg)
	return str(stage)


def _reject_legacy_attribute_config(config: Mapping[str, object]) -> None:
	for path, key in _iter_mapping_keys(config):
		if path in LEGACY_ATTRIBUTE_KEY_PATHS or key in LEGACY_ATTRIBUTE_KEY_NAMES:
			msg = (
				f'{path} is a legacy multi-attribute config key and is not '
				'valid for the amplitude-only MVP; remove fixed-attribute '
				'configuration from this config.'
			)
			raise ValueError(msg)


def _validate_generated_metadata_paths(
	config: Mapping[str, object],
	stage: str,
	*,
	artifact_root: Path,
	nopims_root: Path,
) -> None:
	if stage == 'build_nopims_manifests':
		manifest = _required_mapping(config, 'manifest')
		_validate_artifact_output_path(
			_validate_path(manifest, 'output_dir', prefix='manifest'),
			'manifest.output_dir',
			artifact_root=artifact_root,
			nopims_root=nopims_root,
		)
		_validate_artifact_output_path(
			_validate_path(
				manifest,
				'normalization_stats_dir',
				prefix='manifest',
			),
			'manifest.normalization_stats_dir',
			artifact_root=artifact_root,
			nopims_root=nopims_root,
		)
	elif stage == 'filter_manifest_by_normalization_qc':
		manifests = _required_mapping(config, 'manifests')
		splits = _required_mapping(config, 'splits')
		qc = _required_mapping(config, 'qc')
		for parent, key, prefix in (
			(manifests, 'output', 'manifests'),
			(splits, 'output', 'splits'),
			(qc, 'output_json', 'qc'),
			(qc, 'excluded_surveys', 'qc'),
		):
			label = f'{prefix}.{key}'
			_validate_artifact_output_path(
				_validate_path(parent, key, prefix=prefix),
				label,
				artifact_root=artifact_root,
				nopims_root=nopims_root,
			)


def _validate_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	path = _validate_path(parent, key, prefix=prefix)
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _validate_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return Path(value)


def _validate_artifact_output_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
	nopims_root: Path,
) -> None:
	if not path.is_absolute():
		msg = f'{label} must be an absolute artifact-registry path; got {path}'
		raise ValueError(msg)
	if _is_relative_to(path, nopims_root):
		msg = f'{label} must not be under paths.nopims_root; got {path}'
		raise ValueError(msg)
	if not _is_relative_to(path, artifact_root):
		msg = f'{label} must be under paths.artifact_root ({artifact_root}); got {path}'
		raise ValueError(msg)


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


def _iter_mapping_keys(
	value: object,
	prefix: str = '',
) -> Sequence[tuple[str, str]]:
	if isinstance(value, Sequence) and not isinstance(value, str | bytes):
		paths: list[tuple[str, str]] = []
		for index, child in enumerate(value):
			path = f'{prefix}[{index}]' if prefix else f'[{index}]'
			paths.extend(_iter_mapping_keys(child, path))
		return paths

	if not isinstance(value, Mapping):
		return ()

	paths: list[tuple[str, str]] = []
	for key, child in value.items():
		if not isinstance(key, str):
			continue
		path = f'{prefix}.{key}' if prefix else key
		paths.append((path, key))
		paths.extend(_iter_mapping_keys(child, path))
	return paths


def _validate_masking(masking: Mapping[str, object]) -> None:
	ratio = masking.get('spatial_mask_ratio')
	if (
		not isinstance(ratio, Real)
		or isinstance(ratio, bool)
		or ratio <= 0.0
		or ratio >= 1.0
	):
		msg = 'masking.spatial_mask_ratio must be greater than 0 and less than 1'
		raise ValueError(msg)

	mode = masking.get('spatial_mask_mode')
	if mode != 'block':
		msg = "masking.spatial_mask_mode must be 'block'"
		raise ValueError(msg)

	_validate_positive_int_triplet(
		masking,
		'block_size_tokens',
		prefix='masking',
	)


def _validate_train(train: Mapping[str, object]) -> None:
	for key in ('batch_size', 'samples_per_epoch', 'epochs'):
		_validate_positive_int(train, key, prefix='train')
	if 'num_workers' in train:
		_validate_nonnegative_int(train, 'num_workers', prefix='train')
	for key in ('lr', 'weight_decay', 'grad_clip_norm'):
		if key in train:
			_validate_positive_number(train, key, prefix='train')
	_validate_bool(train, 'amp', prefix='train')
	if 'shuffle' in train:
		_validate_bool(train, 'shuffle', prefix='train')
	if 'seed' in train and not _is_int(train.get('seed')):
		msg = f'train.seed must be an integer; got {train.get("seed")!r}'
		raise ValueError(msg)


def _validate_loss(loss: Mapping[str, object]) -> None:
	if loss.get('reconstruction') != 'huber':
		msg = "loss.reconstruction must be 'huber'"
		raise ValueError(msg)
	if 'huber_delta' in loss:
		_validate_positive_number(loss, 'huber_delta', prefix='loss')
	if 'gradient_weight' in loss:
		_validate_nonnegative_number(loss, 'gradient_weight', prefix='loss')
	if loss.get('valid_mask_mode') != 'voxel':
		msg = "loss.valid_mask_mode must be 'voxel'"
		raise ValueError(msg)


def _validate_divisible_crop_patch(
	crop_size: Sequence[int],
	patch_size: Sequence[int],
) -> None:
	if any(
		crop % patch != 0
		for crop, patch in zip(crop_size, patch_size, strict=True)
	):
		msg = (
			'data.local_crop_size dimensions must be divisible by '
			'model.patch_size dimensions'
		)
		raise ValueError(msg)


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _validate_equal(
	parent: Mapping[str, object],
	key: str,
	expected: object,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if value != expected:
		msg = f'{prefix}.{key} must be {expected!r}; got {value!r}'
		raise ValueError(msg)


def _validate_positive_int_triplet(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> tuple[int, int, int]:
	value = parent.get(key)
	if (
		not isinstance(value, list)
		or len(value) != 3
		or not all(_is_int(item) and int(item) > 0 for item in value)
	):
		msg = f'{prefix}.{key} must be a list of three positive integers'
		raise ValueError(msg)
	return (int(value[0]), int(value[1]), int(value[2]))


def _validate_positive_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not _is_int(value) or int(value) <= 0:
		msg = f'{prefix}.{key} must be a positive integer; got {value!r}'
		raise ValueError(msg)


def _validate_nonnegative_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not _is_int(value) or int(value) < 0:
		msg = f'{prefix}.{key} must be a nonnegative integer; got {value!r}'
		raise ValueError(msg)


def _validate_positive_number(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not _is_number(value) or float(value) <= 0.0:
		msg = f'{prefix}.{key} must be positive; got {value!r}'
		raise ValueError(msg)


def _validate_nonnegative_number(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not _is_number(value) or float(value) < 0.0:
		msg = f'{prefix}.{key} must be nonnegative; got {value!r}'
		raise ValueError(msg)


def _validate_bool(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not isinstance(value, bool):
		msg = f'{prefix}.{key} must be a boolean; got {value!r}'
		raise TypeError(msg)


def _is_int(value: object) -> bool:
	return isinstance(value, Integral) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
	return isinstance(value, Real) and not isinstance(value, bool)
