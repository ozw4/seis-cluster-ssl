"""Weights-only initialization and trainability for MAE continuation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from seis_ssl_cluster.training.checkpoint import load_checkpoint
from seis_ssl_cluster.training.encoder_trainability import (
	freeze_all_and_unfreeze_top_encoder_blocks,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.models.mae.model import AmplitudeMAE3D


def load_mae_continuation_weights(
	model: AmplitudeMAE3D,
	checkpoint_path: str | Path,
	*,
	expected_model_config: Mapping[str, object],
) -> str:
	"""Load compatible MAE weights and return the stable source SHA-256."""
	path = Path(checkpoint_path)
	if not path.is_file():
		msg = f'MAE continuation checkpoint file does not exist: {path}'
		raise FileNotFoundError(msg)
	if not isinstance(expected_model_config, Mapping):
		msg = 'expected_model_config must be a mapping'
		raise TypeError(msg)

	source_sha256 = _file_sha256(path)
	payload = load_checkpoint(path, map_location='cpu')
	if _file_sha256(path) != source_sha256:
		raise RuntimeError(f'MAE continuation checkpoint changed while loading: {path}')
	model_state = _continuation_model_state(payload, expected_model_config)
	_validate_finite_model_state(model_state)
	try:
		model.load_state_dict(
			cast('Mapping[str, torch.Tensor]', model_state),
			strict=True,
		)
	except RuntimeError as exc:
		msg = (
			'MAE continuation checkpoint model geometry/state mismatch: '
			f'{path}: {exc}'
		)
		raise ValueError(msg) from exc
	return source_sha256


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _continuation_model_state(
	payload: object,
	expected_model_config: Mapping[str, object],
) -> Mapping[object, object]:
	if not isinstance(payload, Mapping):
		msg = 'MAE continuation checkpoint payload must be a mapping'
		raise TypeError(msg)
	model_state = payload.get('model_state_dict')
	if not isinstance(model_state, Mapping):
		msg = 'MAE continuation checkpoint model_state_dict must be a mapping'
		raise TypeError(msg)
	_validate_source_model_config(payload.get('config'), expected_model_config)
	_validate_source_training_state(payload.get('training_state'))
	return model_state


def _validate_source_model_config(
	value: object,
	expected_model_config: Mapping[str, object],
) -> None:
	checkpoint_config = value
	if not isinstance(checkpoint_config, Mapping):
		msg = 'MAE continuation checkpoint config must be a mapping'
		raise TypeError(msg)
	source_model_config = checkpoint_config.get('model')
	if not isinstance(source_model_config, Mapping):
		msg = 'MAE continuation checkpoint config.model must be a mapping'
		raise TypeError(msg)
	if source_model_config != expected_model_config:
		msg = (
			'MAE continuation checkpoint config.model does not match '
			'the current resolved model config'
		)
		raise ValueError(msg)


def _validate_source_training_state(value: object) -> None:
	training_state = value
	if not isinstance(training_state, Mapping):
		msg = 'MAE continuation checkpoint training_state must be a mapping'
		raise TypeError(msg)
	if training_state.get('stage') != 'train_amp_mae':
		msg = (
			'MAE continuation checkpoint training_state.stage must be '
			f"'train_amp_mae'; got {training_state.get('stage')!r}"
		)
		raise ValueError(msg)
	if training_state.get('checkpoint_kind') != 'epoch':
		msg = (
			'MAE continuation checkpoint training_state.checkpoint_kind must be '
			f"'epoch'; got {training_state.get('checkpoint_kind')!r}"
		)
		raise ValueError(msg)


def configure_mae_continuation_trainability(
	model: AmplitudeMAE3D,
	*,
	unfreeze_top_blocks: int,
) -> tuple[torch.nn.Parameter, ...]:
	"""Select top encoder blocks and MAE reconstruction parameters for training."""
	freeze_all_and_unfreeze_top_encoder_blocks(
		model,
		unfreeze_top_blocks=unfreeze_top_blocks,
	)

	model.mask_token.requires_grad_(requires_grad=True)
	model.encoder_to_decoder.requires_grad_(requires_grad=True)
	model.decoder.requires_grad_(requires_grad=True)
	model.prediction_head.requires_grad_(requires_grad=True)

	trainable_parameters = tuple(
		parameter for parameter in model.parameters() if parameter.requires_grad
	)
	if not trainable_parameters:
		raise RuntimeError('MAE continuation has no trainable parameters')
	return trainable_parameters


def _validate_finite_model_state(model_state: Mapping[object, object]) -> None:
	for name, value in model_state.items():
		if not isinstance(name, str):
			msg = 'MAE continuation checkpoint model_state_dict keys must be strings'
			raise TypeError(msg)
		if not isinstance(value, torch.Tensor):
			msg = (
				'MAE continuation checkpoint model_state_dict '
				f'value {name!r} must be a tensor'
			)
			raise TypeError(msg)
		if value.is_floating_point() and not bool(torch.isfinite(value).all()):
			msg = (
				'MAE continuation checkpoint model_state_dict '
				f'value {name!r} contains non-finite values'
			)
			raise ValueError(msg)


__all__ = [
	'configure_mae_continuation_trainability',
	'load_mae_continuation_weights',
]
