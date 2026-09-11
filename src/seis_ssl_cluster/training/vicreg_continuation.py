"""Weights-only initialization and trainability for local VICReg."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch

from seis_ssl_cluster.config.schema import STAGE_VICREG_TRAINING
from seis_ssl_cluster.config.vicreg import (
	resolve_vicreg_pretraining_method,
	vicreg_config_compatibility_identity,
)
from seis_ssl_cluster.models.barlow_twins import BarlowTwins3D
from seis_ssl_cluster.training.checkpoint import load_checkpoint
from seis_ssl_cluster.training.encoder_trainability import (
	freeze_all_and_unfreeze_top_encoder_blocks,
)
from seis_ssl_cluster.training.vicreg_checkpoint import (
	CHECKPOINT_KIND,
	TRAINED_PARAMETER_PREFIXES,
)


def load_vicreg_continuation_weights(
	model: BarlowTwins3D,
	checkpoint_path: str | Path,
	*,
	expected_model_config: Mapping[str, object],
	expected_vicreg_config: Mapping[str, object],
) -> str:
	"""Load compatible VICReg weights and return the stable source SHA-256."""
	path = Path(checkpoint_path)
	if not path.is_file():
		raise FileNotFoundError(
			f'VICReg continuation checkpoint file does not exist: {path}'
		)
	if not isinstance(model, BarlowTwins3D):
		raise TypeError(
			f'model must be a BarlowTwins3D; got {type(model).__name__}'
		)
	if not isinstance(expected_model_config, Mapping):
		raise TypeError('expected_model_config must be a mapping')
	if not isinstance(expected_vicreg_config, Mapping):
		raise TypeError('expected_vicreg_config must be a mapping')

	source_sha256 = _file_sha256(path)
	payload = load_checkpoint(path, map_location='cpu')
	if _file_sha256(path) != source_sha256:
		raise RuntimeError(
			f'VICReg continuation checkpoint changed while loading: {path}'
		)
	backbone_state, projector_state = _continuation_states(
		payload,
		expected_model_config=expected_model_config,
		expected_vicreg_config=expected_vicreg_config,
	)
	_validate_finite_tensor_state(backbone_state, label='model_state_dict')
	_validate_finite_tensor_state(projector_state, label='projector_state_dict')
	try:
		model.backbone.load_state_dict(
			cast('Mapping[str, torch.Tensor]', backbone_state),
			strict=True,
		)
	except RuntimeError as exc:
		raise ValueError(
			'VICReg continuation checkpoint backbone geometry/state mismatch: '
			f'{path}: {exc}'
		) from exc
	try:
		model.projector.load_state_dict(
			cast('Mapping[str, torch.Tensor]', projector_state),
			strict=True,
		)
	except RuntimeError as exc:
		raise ValueError(
			'VICReg continuation checkpoint projector geometry/state mismatch: '
			f'{path}: {exc}'
		) from exc
	return source_sha256


def configure_vicreg_continuation_trainability(
	model: BarlowTwins3D,
	*,
	unfreeze_top_blocks: int,
) -> tuple[torch.nn.Parameter, ...]:
	"""Train only the selected top encoder blocks and the shared projector."""
	if not isinstance(model, BarlowTwins3D):
		raise TypeError(
			f'model must be a BarlowTwins3D; got {type(model).__name__}'
		)
	freeze_all_and_unfreeze_top_encoder_blocks(
		model.backbone,
		unfreeze_top_blocks=unfreeze_top_blocks,
	)
	model.projector.requires_grad_(requires_grad=True)
	parameters = tuple(
		parameter for parameter in model.parameters() if parameter.requires_grad
	)
	if not parameters:
		raise RuntimeError('VICReg continuation has no trainable parameters')
	return parameters


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as file_obj:
		for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _continuation_states(
	payload: object,
	*,
	expected_model_config: Mapping[str, object],
	expected_vicreg_config: Mapping[str, object],
) -> tuple[Mapping[object, object], Mapping[object, object]]:
	if not isinstance(payload, Mapping):
		raise TypeError('VICReg continuation checkpoint payload must be a mapping')
	backbone_state = payload.get('model_state_dict')
	if not isinstance(backbone_state, Mapping):
		raise TypeError(
			'VICReg continuation checkpoint model_state_dict must be a mapping'
		)
	projector_state = payload.get('projector_state_dict')
	if not isinstance(projector_state, Mapping):
		raise TypeError(
			'VICReg continuation checkpoint projector_state_dict must be a mapping'
		)
	source_method = _validate_source_config(
		payload.get('config'),
		expected_model_config=expected_model_config,
		expected_vicreg_config=expected_vicreg_config,
	)
	_validate_source_identity(payload, expected_method=source_method)
	_validate_source_training_state(payload.get('training_state'))
	return backbone_state, projector_state


def _validate_source_config(
	value: object,
	*,
	expected_model_config: Mapping[str, object],
	expected_vicreg_config: Mapping[str, object],
) -> str:
	if not isinstance(value, Mapping):
		raise TypeError('VICReg continuation checkpoint config must be a mapping')
	if value.get('stage') != STAGE_VICREG_TRAINING:
		raise ValueError(
			'VICReg continuation checkpoint config.stage must be '
			f'{STAGE_VICREG_TRAINING!r}; got {value.get("stage")!r}'
		)
	if value.get('model') != expected_model_config:
		raise ValueError(
			'VICReg continuation checkpoint config.model does not match '
			'the current resolved model config'
		)
	source_vicreg = value.get('vicreg')
	if not isinstance(source_vicreg, Mapping):
		raise TypeError(
			'VICReg continuation checkpoint config.vicreg must be a mapping'
		)
	source_method = resolve_vicreg_pretraining_method(value)
	expected_method = resolve_vicreg_pretraining_method(
		{'vicreg': expected_vicreg_config}
	)
	if source_method != expected_method:
		raise ValueError(
			'VICReg continuation checkpoint pretraining_method does not match '
			'the current resolved VICReg config'
		)
	if vicreg_config_compatibility_identity(
		value
	) != vicreg_config_compatibility_identity({'vicreg': expected_vicreg_config}):
		raise ValueError(
			'VICReg continuation checkpoint config.vicreg does not match '
			'the current resolved VICReg config'
		)
	return source_method


def _validate_source_identity(
	payload: Mapping[object, object],
	*,
	expected_method: str,
) -> None:
	if payload.get('pretraining_method') != expected_method:
		raise ValueError(
			'VICReg continuation checkpoint pretraining_method does not match '
			'checkpoint config'
		)
	if payload.get('checkpoint_kind') != CHECKPOINT_KIND:
		raise ValueError(
			'VICReg continuation checkpoint checkpoint_kind is invalid'
		)
	prefixes = payload.get('trained_parameter_prefixes')
	if not isinstance(prefixes, list | tuple) or tuple(prefixes) != (
		TRAINED_PARAMETER_PREFIXES
	):
		raise ValueError(
			'VICReg continuation checkpoint trained_parameter_prefixes are invalid'
		)


def _validate_source_training_state(value: object) -> None:
	if not isinstance(value, Mapping):
		raise TypeError(
			'VICReg continuation checkpoint training_state must be a mapping'
		)
	if value.get('stage') != STAGE_VICREG_TRAINING:
		raise ValueError(
			'VICReg continuation checkpoint training_state.stage must be '
			f'{STAGE_VICREG_TRAINING!r}; got {value.get("stage")!r}'
		)
	if value.get('resume_boundary') != 'epoch':
		raise ValueError(
			'VICReg continuation checkpoint training_state.resume_boundary '
			f"must be 'epoch'; got {value.get('resume_boundary')!r}"
		)
	if value.get('completed_epoch') is not True:
		raise ValueError(
			'VICReg continuation checkpoint training_state.completed_epoch '
			'must be true'
		)


def _validate_finite_tensor_state(
	state: Mapping[object, object],
	*,
	label: str,
) -> None:
	for name, value in state.items():
		if not isinstance(name, str):
			raise TypeError(
				f'VICReg continuation checkpoint {label} keys must be strings'
			)
		if not isinstance(value, torch.Tensor):
			raise TypeError(
				f'VICReg continuation checkpoint {label} value {name!r} '
				'must be a tensor'
			)
		if (value.is_floating_point() or value.is_complex()) and not bool(
			torch.isfinite(value).all()
		):
			raise ValueError(
				f'VICReg continuation checkpoint {label} value {name!r} '
				'contains non-finite values'
			)


__all__ = [
	'configure_vicreg_continuation_trainability',
	'load_vicreg_continuation_weights',
]
