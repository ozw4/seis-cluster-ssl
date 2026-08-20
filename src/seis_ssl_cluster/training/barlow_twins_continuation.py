"""Weights-only initialization and trainability for Barlow Twins continuation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch

from seis_ssl_cluster.config.barlow_twins import (
	barlow_twins_config_compatibility_identity,
	resolve_barlow_twins_pretraining_method,
)
from seis_ssl_cluster.config.schema import STAGE_BARLOW_TWINS_TRAINING
from seis_ssl_cluster.models.barlow_twins import BarlowTwins3D
from seis_ssl_cluster.training.barlow_twins_checkpoint import (
	CHECKPOINT_KIND,
	TRAINED_PARAMETER_PREFIXES,
)
from seis_ssl_cluster.training.checkpoint import load_checkpoint
from seis_ssl_cluster.training.encoder_trainability import (
	freeze_all_and_unfreeze_top_encoder_blocks,
)


def load_barlow_twins_continuation_weights(
	model: BarlowTwins3D,
	checkpoint_path: str | Path,
	*,
	expected_model_config: Mapping[str, object],
	expected_barlow_twins_config: Mapping[str, object],
) -> None:
	"""Load only compatible backbone and projector weights from a checkpoint."""
	path = Path(checkpoint_path)
	if not path.is_file():
		msg = f'Barlow Twins continuation checkpoint file does not exist: {path}'
		raise FileNotFoundError(msg)
	if not isinstance(model, BarlowTwins3D):
		msg = f'model must be a BarlowTwins3D; got {type(model).__name__}'
		raise TypeError(msg)
	if not isinstance(expected_model_config, Mapping):
		raise TypeError('expected_model_config must be a mapping')
	if not isinstance(expected_barlow_twins_config, Mapping):
		raise TypeError('expected_barlow_twins_config must be a mapping')

	payload = load_checkpoint(path, map_location='cpu')
	backbone_state, projector_state = _continuation_states(
		payload,
		expected_model_config=expected_model_config,
		expected_barlow_twins_config=expected_barlow_twins_config,
	)
	_validate_finite_tensor_state(backbone_state, label='model_state_dict')
	_validate_finite_tensor_state(projector_state, label='projector_state_dict')

	try:
		model.backbone.load_state_dict(
			cast('Mapping[str, torch.Tensor]', backbone_state),
			strict=True,
		)
	except RuntimeError as exc:
		msg = (
			'Barlow Twins continuation checkpoint backbone geometry/state mismatch: '
			f'{path}: {exc}'
		)
		raise ValueError(msg) from exc
	try:
		model.projector.load_state_dict(
			cast('Mapping[str, torch.Tensor]', projector_state),
			strict=True,
		)
	except RuntimeError as exc:
		msg = (
			'Barlow Twins continuation checkpoint projector geometry/state mismatch: '
			f'{path}: {exc}'
		)
		raise ValueError(msg) from exc


def _continuation_states(
	payload: object,
	*,
	expected_model_config: Mapping[str, object],
	expected_barlow_twins_config: Mapping[str, object],
) -> tuple[Mapping[object, object], Mapping[object, object]]:
	if not isinstance(payload, Mapping):
		msg = 'Barlow Twins continuation checkpoint payload must be a mapping'
		raise TypeError(msg)
	backbone_state = payload.get('model_state_dict')
	if not isinstance(backbone_state, Mapping):
		msg = (
			'Barlow Twins continuation checkpoint model_state_dict must be a mapping'
		)
		raise TypeError(msg)
	projector_state = payload.get('projector_state_dict')
	if not isinstance(projector_state, Mapping):
		msg = (
			'Barlow Twins continuation checkpoint projector_state_dict '
			'must be a mapping'
		)
		raise TypeError(msg)

	source_method = _validate_source_config(
		payload.get('config'),
		expected_model_config=expected_model_config,
		expected_barlow_twins_config=expected_barlow_twins_config,
	)
	_validate_source_identity(payload, expected_method=source_method)
	_validate_source_training_state(payload.get('training_state'))
	return backbone_state, projector_state


def _validate_source_config(
	value: object,
	*,
	expected_model_config: Mapping[str, object],
	expected_barlow_twins_config: Mapping[str, object],
) -> str:
	if not isinstance(value, Mapping):
		raise TypeError('Barlow Twins continuation checkpoint config must be a mapping')
	source_model_config = value.get('model')
	if not isinstance(source_model_config, Mapping):
		msg = 'Barlow Twins continuation checkpoint config.model must be a mapping'
		raise TypeError(msg)
	if source_model_config != expected_model_config:
		msg = (
			'Barlow Twins continuation checkpoint config.model does not match '
			'the current resolved model config'
		)
		raise ValueError(msg)
	source_barlow_twins_config = value.get('barlow_twins')
	if not isinstance(source_barlow_twins_config, Mapping):
		msg = (
			'Barlow Twins continuation checkpoint config.barlow_twins '
			'must be a mapping'
		)
		raise TypeError(msg)
	source_method = resolve_barlow_twins_pretraining_method(value)
	expected_method = resolve_barlow_twins_pretraining_method(
		{'barlow_twins': expected_barlow_twins_config}
	)
	if source_method != expected_method:
		msg = (
			'Barlow Twins continuation checkpoint pretraining_method does not '
			'match the current resolved Barlow Twins config'
		)
		raise ValueError(msg)
	if barlow_twins_config_compatibility_identity(
		value
	) != barlow_twins_config_compatibility_identity(
		{'barlow_twins': expected_barlow_twins_config}
	):
		msg = (
			'Barlow Twins continuation checkpoint config.barlow_twins does not match '
			'the current resolved Barlow Twins config'
		)
		raise ValueError(msg)
	return source_method


def _validate_source_identity(
	payload: Mapping[object, object],
	*,
	expected_method: str,
) -> None:
	if payload.get('pretraining_method') != expected_method:
		msg = (
			'Barlow Twins continuation checkpoint pretraining_method does not '
			'match checkpoint config'
		)
		raise ValueError(msg)
	if payload.get('checkpoint_kind') != CHECKPOINT_KIND:
		msg = 'Barlow Twins continuation checkpoint checkpoint_kind is invalid'
		raise ValueError(msg)
	prefixes = payload.get('trained_parameter_prefixes')
	if (
		not isinstance(prefixes, list | tuple)
		or tuple(prefixes) != TRAINED_PARAMETER_PREFIXES
	):
		msg = (
			'Barlow Twins continuation checkpoint trained_parameter_prefixes '
			'are invalid'
		)
		raise ValueError(msg)


def _validate_source_training_state(value: object) -> None:
	if not isinstance(value, Mapping):
		msg = 'Barlow Twins continuation checkpoint training_state must be a mapping'
		raise TypeError(msg)
	if value.get('stage') != STAGE_BARLOW_TWINS_TRAINING:
		msg = (
			'Barlow Twins continuation checkpoint training_state.stage must be '
			f'{STAGE_BARLOW_TWINS_TRAINING!r}; got {value.get("stage")!r}'
		)
		raise ValueError(msg)
	if value.get('resume_boundary') != 'epoch':
		msg = (
			'Barlow Twins continuation checkpoint training_state.resume_boundary '
			f"must be 'epoch'; got {value.get('resume_boundary')!r}"
		)
		raise ValueError(msg)
	if value.get('completed_epoch') is not True:
		msg = (
			'Barlow Twins continuation checkpoint training_state.completed_epoch '
			'must be true'
		)
		raise ValueError(msg)


def _validate_finite_tensor_state(
	state: Mapping[object, object],
	*,
	label: str,
) -> None:
	for name, value in state.items():
		if not isinstance(name, str):
			msg = f'Barlow Twins continuation checkpoint {label} keys must be strings'
			raise TypeError(msg)
		if not isinstance(value, torch.Tensor):
			msg = (
				f'Barlow Twins continuation checkpoint {label} '
				f'value {name!r} must be a tensor'
			)
			raise TypeError(msg)
		if (
			value.is_floating_point() or value.is_complex()
		) and not bool(torch.isfinite(value).all()):
			msg = (
				f'Barlow Twins continuation checkpoint {label} '
				f'value {name!r} contains non-finite values'
			)
			raise ValueError(msg)


def configure_barlow_twins_continuation_trainability(
	model: BarlowTwins3D,
	*,
	unfreeze_top_blocks: int,
) -> tuple[torch.nn.Parameter, ...]:
	"""Train only top encoder blocks and the Barlow Twins projector."""
	if not isinstance(model, BarlowTwins3D):
		msg = f'model must be a BarlowTwins3D; got {type(model).__name__}'
		raise TypeError(msg)
	freeze_all_and_unfreeze_top_encoder_blocks(
		model.backbone,
		unfreeze_top_blocks=unfreeze_top_blocks,
	)
	model.projector.requires_grad_(requires_grad=True)

	trainable_parameters = tuple(
		parameter for parameter in model.parameters() if parameter.requires_grad
	)
	if not trainable_parameters:
		raise RuntimeError('Barlow Twins continuation has no trainable parameters')
	return trainable_parameters


__all__ = [
	'configure_barlow_twins_continuation_trainability',
	'load_barlow_twins_continuation_weights',
]
