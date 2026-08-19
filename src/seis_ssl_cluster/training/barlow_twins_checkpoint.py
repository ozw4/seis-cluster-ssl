"""Checkpoint and epoch-boundary resume contracts for Barlow Twins."""

from __future__ import annotations

import math
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import seis_ssl_cluster
from seis_ssl_cluster.models.amplitude_encoder_factory import (
	AMPLITUDE_ENCODER_TRAINED_PARAMETER_PREFIXES,
	BARLOW_TWINS_CHECKPOINT_KIND,
	BARLOW_TWINS_PRETRAINING_METHOD,
)
from seis_ssl_cluster.training.checkpoint import (
	capture_rng_state,
	load_checkpoint,
	restore_rng_state,
	save_checkpoint,
)

PRETRAINING_METHOD = BARLOW_TWINS_PRETRAINING_METHOD
CHECKPOINT_KIND = BARLOW_TWINS_CHECKPOINT_KIND
TRAINED_PARAMETER_PREFIXES = AMPLITUDE_ENCODER_TRAINED_PARAMETER_PREFIXES
_REQUIRED_KEYS = frozenset(
	{
		'model_state_dict',
		'projector_state_dict',
		'optimizer_state_dict',
		'epoch',
		'global_step',
		'config',
		'metrics',
		'rng_state',
		'amp_enabled',
		'scaler_state_dict',
		'training_state',
		'pretraining_method',
		'checkpoint_kind',
		'trained_parameter_prefixes',
	}
)
_COMPATIBILITY_SECTIONS = (
	'manifests',
	'data',
	'zero_mask',
	'model',
	'augmentations',
	'barlow_twins',
	'continuation',
)
_ALLOWED_TRAIN_OVERRIDES = frozenset(
	{'epochs', 'max_steps', 'device', 'allow_overwrite_output'}
)


@dataclass(frozen=True)
class BarlowTwinsResumeState:
	"""Position restored from a completed-epoch checkpoint."""

	start_epoch: int
	global_step: int


def save_barlow_twins_checkpoint(  # noqa: PLR0913
	path: str | Path,
	*,
	backbone: torch.nn.Module,
	projector: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	global_step: int,
	config: Mapping[str, object],
	metrics: Mapping[str, float],
	amp_enabled: bool,
	scaler: torch.amp.GradScaler | None,
	scaler_required: bool,
	dataset_epoch: int,
	completed_epoch: bool,
	dataloader_generator: torch.Generator | None = None,
) -> Path:
	"""Write a Barlow checkpoint with bare MAE and separate projector states."""
	rng_state = capture_rng_state()
	if dataloader_generator is not None:
		rng_state['dataloader_generator'] = dataloader_generator.get_state()
	return save_checkpoint(
		path,
		model=backbone,
		optimizer=optimizer,
		epoch=epoch,
		global_step=global_step,
		config=config,
		package_version=getattr(seis_ssl_cluster, '__version__', None),
		metrics=metrics,
		amp_enabled=amp_enabled,
		scaler=scaler,
		scaler_required=scaler_required,
		rng_state=rng_state,
		training_state={
			'schema_version': 1,
			'stage': 'barlow_twins_training',
			'resume_boundary': 'epoch',
			'dataset_epoch': dataset_epoch,
			'completed_epoch': completed_epoch,
		},
		extra_payload={
			'projector_state_dict': projector.state_dict(),
			'pretraining_method': PRETRAINING_METHOD,
			'checkpoint_kind': CHECKPOINT_KIND,
			'trained_parameter_prefixes': TRAINED_PARAMETER_PREFIXES,
		},
	)


def restore_barlow_twins_checkpoint(  # noqa: PLR0913
	payload: Mapping[str, Any],
	*,
	backbone: torch.nn.Module,
	projector: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	scaler_required: bool,
	config: Mapping[str, object],
	dataloader_generator: torch.Generator | None = None,
) -> BarlowTwinsResumeState:
	"""Validate and restore one completed-epoch Barlow Twins checkpoint."""
	_validate_payload(payload, scaler_required=scaler_required, config=config)
	backbone.load_state_dict(payload['model_state_dict'], strict=True)
	projector.load_state_dict(payload['projector_state_dict'], strict=True)
	optimizer.load_state_dict(payload['optimizer_state_dict'])
	if scaler_required:
		if scaler is None:
			raise ValueError('a GradScaler is required to resume this checkpoint')
		scaler.load_state_dict(payload['scaler_state_dict'])
	restore_rng_state(payload)
	if dataloader_generator is not None:
		rng_state = payload['rng_state']
		generator_state = rng_state.get('dataloader_generator')
		if not isinstance(generator_state, torch.Tensor):
			raise TypeError('checkpoint dataloader generator state must be a tensor')
		dataloader_generator.set_state(generator_state.cpu())
	return BarlowTwinsResumeState(
		start_epoch=int(payload['epoch']) + 1,
		global_step=int(payload['global_step']),
	)


def load_barlow_twins_checkpoint(
	path: str | Path,
	*,
	map_location: str | torch.device | None = None,
) -> dict[str, Any]:
	"""Load one Barlow Twins checkpoint payload."""
	return load_checkpoint(path, map_location=map_location)


def update_best_checkpoint(
	latest_path: Path,
	best_path: Path,
	*,
	loss: float,
	best_loss: float | None,
) -> float | None:
	"""Copy latest to best when the finite training loss improves."""
	if not math.isfinite(loss) or (best_loss is not None and loss >= best_loss):
		return best_loss
	best_path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile(
		prefix=f'.{best_path.name}.',
		suffix='.tmp',
		dir=best_path.parent,
		delete=False,
	) as file_obj:
		tmp_path = Path(file_obj.name)
	try:
		shutil.copyfile(latest_path, tmp_path)
		tmp_path.replace(best_path)
	finally:
		if tmp_path.exists():
			tmp_path.unlink()
	return loss


def _validate_payload(  # noqa: C901
	payload: Mapping[str, Any],
	*,
	scaler_required: bool,
	config: Mapping[str, object],
) -> None:
	missing = sorted(_REQUIRED_KEYS - set(payload))
	if missing:
		raise ValueError(f'Barlow Twins checkpoint is missing keys: {missing!r}')
	for key in ('model_state_dict', 'projector_state_dict', 'optimizer_state_dict'):
		if not isinstance(payload[key], Mapping):
			raise TypeError(f'checkpoint {key} must be a mapping')
	if payload['pretraining_method'] != PRETRAINING_METHOD:
		raise ValueError('checkpoint pretraining_method is not barlow_twins_3d')
	if payload['checkpoint_kind'] != CHECKPOINT_KIND:
		raise ValueError('checkpoint kind is not Barlow Twins pretraining')
	if tuple(payload['trained_parameter_prefixes']) != TRAINED_PARAMETER_PREFIXES:
		raise ValueError('checkpoint trained_parameter_prefixes do not match contract')
	training_state = payload['training_state']
	if not isinstance(training_state, Mapping):
		raise TypeError('checkpoint training_state must be a mapping')
	if training_state.get('stage') != 'barlow_twins_training':
		raise ValueError('checkpoint training stage is not barlow_twins_training')
	if training_state.get('resume_boundary') != 'epoch' or not training_state.get(
		'completed_epoch'
	):
		raise ValueError('Barlow Twins resume requires a completed epoch checkpoint')
	if scaler_required and not isinstance(payload['scaler_state_dict'], Mapping):
		raise ValueError('checkpoint is missing required GradScaler state')
	if not scaler_required and payload['scaler_state_dict'] is not None:
		raise ValueError('checkpoint scaler state is incompatible with this run')
	_validate_config_compatibility(payload['config'], config)


def _validate_config_compatibility(
	saved: object,
	current: Mapping[str, object],
) -> None:
	if not isinstance(saved, Mapping):
		raise TypeError('checkpoint config must be a mapping')
	for section in _COMPATIBILITY_SECTIONS:
		if saved.get(section) != current.get(section):
			raise ValueError(f'resume config section {section!r} does not match')
	for key, value in _mapping(saved, 'train').items():
		if key not in _ALLOWED_TRAIN_OVERRIDES and _mapping(current, 'train').get(
			key
		) != value:
			raise ValueError(f'resume train setting {key!r} does not match')


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return child


__all__ = [
	'CHECKPOINT_KIND',
	'PRETRAINING_METHOD',
	'TRAINED_PARAMETER_PREFIXES',
	'BarlowTwinsResumeState',
	'load_barlow_twins_checkpoint',
	'restore_barlow_twins_checkpoint',
	'save_barlow_twins_checkpoint',
	'update_best_checkpoint',
]
