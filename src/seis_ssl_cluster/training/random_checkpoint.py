"""Random MAE checkpoint creation for encoder baseline comparisons."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

import seis_ssl_cluster
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.training.checkpoint import capture_rng_state, load_checkpoint


@dataclass(frozen=True)
class RandomMaeCheckpointConfig:
	"""Resolved settings for a random MAE checkpoint artifact."""

	reference_checkpoint: Path
	reference_model_tag: str
	seed: int
	output_checkpoint: Path


def create_random_mae_checkpoint_from_config(
	config: Mapping[str, object],
) -> Path:
	"""Create a random MAE checkpoint from a baseline YAML mapping."""
	settings = random_mae_checkpoint_config_from_mapping(config)
	return create_random_mae_checkpoint(
		reference_checkpoint=settings.reference_checkpoint,
		reference_model_tag=settings.reference_model_tag,
		seed=settings.seed,
		output_checkpoint=settings.output_checkpoint,
	)


def create_random_mae_checkpoint(
	*,
	reference_checkpoint: str | Path,
	reference_model_tag: str,
	seed: int,
	output_checkpoint: str | Path,
) -> Path:
	"""Save a random-initialized MAE checkpoint matching ``reference_checkpoint``."""
	reference_path = Path(reference_checkpoint)
	output_path = Path(output_checkpoint)
	if not reference_path.is_file():
		msg = f'reference checkpoint does not exist: {reference_path}'
		raise FileNotFoundError(msg)
	_validate_no_runs_segment(output_path, 'random checkpoint output')
	resolved_seed = _nonnegative_int(seed, 'seed')
	resolved_reference_model_tag = _non_empty_string(
		reference_model_tag,
		'reference_model_tag',
	)

	reference_payload = load_checkpoint(reference_path, map_location='cpu')
	reference_config = _checkpoint_config(reference_payload)
	model = _build_model(reference_config, seed=resolved_seed)
	metadata = {
		'random_encoder_baseline': True,
		'reference_checkpoint': str(reference_path),
		'reference_model_tag': resolved_reference_model_tag,
		'seed': resolved_seed,
		'pretrained_weights_loaded': False,
	}
	payload = {
		'model_state_dict': {
			key: value.detach().cpu() for key, value in model.state_dict().items()
		},
		'optimizer_state_dict': {},
		'epoch': 0,
		'global_step': 0,
		'amp_enabled': False,
		'scaler_state_dict': None,
		'config': _to_plain_value(reference_config),
		'package_version': getattr(seis_ssl_cluster, '__version__', None),
		'metrics': {},
		'rng_state': {
			**capture_rng_state(),
			'random_checkpoint_seed': resolved_seed,
		},
		'training_state': {
			'schema_version': 1,
			'stage': 'create_random_mae_checkpoint',
			'checkpoint_kind': 'random_init',
			'batch_index': None,
		},
		'metadata': metadata,
	}
	return _atomic_torch_save(output_path, payload)


def random_mae_checkpoint_config_from_mapping(
	config: Mapping[str, object],
) -> RandomMaeCheckpointConfig:
	"""Resolve random-checkpoint settings from an experiment config mapping."""
	paths = _mapping(config, 'paths')
	artifact_root = _absolute_path(paths, 'artifact_root', 'paths')
	reference = _first_mapping(config, ('reference_model', 'reference'))
	random_checkpoint = _first_mapping(
		config,
		('random_checkpoint', 'random_encoder'),
	)

	reference_checkpoint = _absolute_path(
		reference,
		'checkpoint',
		'reference_model',
	)
	reference_model_tag = _non_empty_string(
		reference.get('tag', reference.get('model_tag')),
		'reference_model.tag',
	)
	output_checkpoint = _absolute_path(
		random_checkpoint,
		'output_checkpoint',
		'random_checkpoint',
	)
	_validate_under_root(output_checkpoint, artifact_root, 'output_checkpoint')
	_validate_no_runs_segment(output_checkpoint, 'output_checkpoint')
	return RandomMaeCheckpointConfig(
		reference_checkpoint=reference_checkpoint,
		reference_model_tag=reference_model_tag,
		seed=_nonnegative_int(random_checkpoint.get('seed'), 'random_checkpoint.seed'),
		output_checkpoint=output_checkpoint,
	)


def _build_model(
	checkpoint_config: Mapping[str, object],
	*,
	seed: int,
) -> AmplitudeMAE3D:
	model_config = _mapping(checkpoint_config, 'model')
	with torch.random.fork_rng(devices=[]):
		torch.manual_seed(seed)
		return AmplitudeMAE3D(
			in_channels=_positive_int(
				model_config.get('in_channels'),
				'model.in_channels',
			),
			out_channels=_positive_int(
				model_config.get('out_channels'),
				'model.out_channels',
			),
			patch_size_xyz=_xyz(model_config.get('patch_size'), 'model.patch_size'),
			encoder_dim=_positive_int(
				model_config.get('encoder_dim'),
				'model.encoder_dim',
			),
			encoder_depth=_positive_int(
				model_config.get('encoder_depth'),
				'model.encoder_depth',
			),
			encoder_heads=_positive_int(
				model_config.get('encoder_heads'),
				'model.encoder_heads',
			),
			decoder_dim=_positive_int(
				model_config.get('decoder_dim'),
				'model.decoder_dim',
			),
			decoder_depth=_positive_int(
				model_config.get('decoder_depth'),
				'model.decoder_depth',
			),
			decoder_heads=_positive_int(
				model_config.get('decoder_heads'),
				'model.decoder_heads',
			),
		)


def _checkpoint_config(payload: Mapping[str, object]) -> Mapping[str, object]:
	value = payload.get('config')
	if not isinstance(value, Mapping):
		msg = 'reference checkpoint is missing a resolved config'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _first_mapping(
	config: Mapping[str, object],
	names: tuple[str, ...],
) -> Mapping[str, object]:
	for name in names:
		value = config.get(name)
		if isinstance(value, Mapping):
			return cast('Mapping[str, object]', value)
	missing = ' or '.join(names)
	msg = f'{missing} must be a mapping'
	raise TypeError(msg)


def _mapping(config: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = config.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _absolute_path(
	config: Mapping[str, object],
	key: str,
	prefix: str,
) -> Path:
	value = _non_empty_string(config.get(key), f'{prefix}.{key}')
	path = Path(value)
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be absolute: {path}'
		raise ValueError(msg)
	return path


def _validate_under_root(path: Path, root: Path, name: str) -> None:
	try:
		path.resolve().relative_to(root.resolve())
	except ValueError as exc:
		msg = f'{name} must be under paths.artifact_root: {path}'
		raise ValueError(msg) from exc


def _validate_no_runs_segment(path: Path, name: str) -> None:
	if 'runs' in path.parts:
		msg = f'{name} must not use a runs/ artifact path: {path}'
		raise ValueError(msg)


def _non_empty_string(value: object, name: str) -> str:
	if not isinstance(value, str) or not value:
		msg = f'{name} must be a non-empty string'
		raise TypeError(msg)
	return value


def _nonnegative_int(value: object, name: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{name} must be an integer'
		raise TypeError(msg)
	if value < 0:
		msg = f'{name} must be nonnegative'
		raise ValueError(msg)
	return int(value)


def _positive_int(value: object, name: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{name} must be an integer'
		raise TypeError(msg)
	if value <= 0:
		msg = f'{name} must be positive'
		raise ValueError(msg)
	return int(value)


def _xyz(value: object, name: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, list | tuple)
		or len(value) != 3
		or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
	):
		msg = f'{name} must be a length-3 integer sequence'
		raise TypeError(msg)
	result = tuple(int(item) for item in value)
	if any(item <= 0 for item in result):
		msg = f'{name} must contain positive integers'
		raise ValueError(msg)
	return result


def _to_plain_value(value: object) -> object:
	if isinstance(value, Mapping):
		return {str(key): _to_plain_value(child) for key, child in value.items()}
	if isinstance(value, list | tuple):
		return [_to_plain_value(child) for child in value]
	if isinstance(value, Path):
		return str(value)
	return deepcopy(value)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	fd, tmp_name = tempfile.mkstemp(
		prefix=f'.{path.name}.',
		suffix='.tmp',
		dir=path.parent,
	)
	tmp_path = Path(tmp_name)
	try:
		with os.fdopen(fd, 'wb') as file_obj:
			torch.save(dict(payload), file_obj)
			file_obj.flush()
			os.fsync(file_obj.fileno())
		tmp_path.replace(path)
	finally:
		if tmp_path.exists():
			tmp_path.unlink()
	return path


__all__ = [
	'RandomMaeCheckpointConfig',
	'create_random_mae_checkpoint',
	'create_random_mae_checkpoint_from_config',
	'random_mae_checkpoint_config_from_mapping',
]
