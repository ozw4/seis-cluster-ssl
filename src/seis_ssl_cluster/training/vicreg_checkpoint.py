"""Checkpoint and epoch-boundary resume contracts for local VICReg."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

import seis_ssl_cluster
from seis_ssl_cluster.config.schema import (
	LOCAL_VICREG_PRETRAINING_METHOD,
	STAGE_VICREG_TRAINING,
)
from seis_ssl_cluster.config.vicreg import (
	resolve_vicreg_pretraining_method,
	vicreg_config_compatibility_identity,
)
from seis_ssl_cluster.models.amplitude_encoder_factory import (
	AMPLITUDE_ENCODER_TRAINED_PARAMETER_PREFIXES,
	VICREG_CHECKPOINT_KIND,
)
from seis_ssl_cluster.training.checkpoint import (
	capture_rng_state,
	load_checkpoint,
	restore_rng_state,
	save_checkpoint,
)

if TYPE_CHECKING:
	from pathlib import Path

PRETRAINING_METHOD = LOCAL_VICREG_PRETRAINING_METHOD
CHECKPOINT_KIND = VICREG_CHECKPOINT_KIND
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
	'vicreg',
	'continuation',
)
_ALLOWED_TRAIN_OVERRIDES = frozenset(
	{'epochs', 'max_steps', 'device', 'allow_overwrite_output'}
)


@dataclass(frozen=True)
class VICRegResumeState:
	"""Position restored from a completed-epoch VICReg checkpoint."""

	start_epoch: int
	global_step: int
	resume_count: int = 0


def save_vicreg_checkpoint(  # noqa: PLR0913
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
	continuation_lineage: Mapping[str, object] | None = None,
	resume_count: int = 0,
) -> Path:
	"""Write bare MAE and separate projector states for VICReg."""
	_validate_vicreg_config_stage(config, label='checkpoint')
	pretraining_method = resolve_vicreg_pretraining_method(config)
	_validate_resume_count(resume_count)
	_validate_continuation_lineage_contract(
		continuation_lineage,
		config=config,
		resume_count=resume_count,
	)
	rng_state = capture_rng_state()
	if dataloader_generator is not None:
		rng_state['dataloader_generator'] = dataloader_generator.get_state()
	extra_payload: dict[str, object] = {
		'projector_state_dict': projector.state_dict(),
		'pretraining_method': pretraining_method,
		'checkpoint_kind': CHECKPOINT_KIND,
		'trained_parameter_prefixes': TRAINED_PARAMETER_PREFIXES,
		'resume_count': resume_count,
	}
	if continuation_lineage is not None:
		extra_payload['continuation_lineage'] = dict(continuation_lineage)
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
			'stage': STAGE_VICREG_TRAINING,
			'resume_boundary': 'epoch',
			'dataset_epoch': dataset_epoch,
			'completed_epoch': completed_epoch,
		},
		extra_payload=extra_payload,
	)


def restore_vicreg_checkpoint(  # noqa: PLR0913
	payload: Mapping[str, Any],
	*,
	backbone: torch.nn.Module,
	projector: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	scaler_required: bool,
	config: Mapping[str, object],
	dataloader_generator: torch.Generator | None = None,
) -> VICRegResumeState:
	"""Validate and restore one completed-epoch VICReg checkpoint."""
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
	return VICRegResumeState(
		start_epoch=int(payload['epoch']) + 1,
		global_step=int(payload['global_step']),
		resume_count=_checkpoint_resume_count(payload) + 1,
	)


def load_vicreg_checkpoint(
	path: str | Path,
	*,
	map_location: str | torch.device | None = None,
) -> dict[str, Any]:
	"""Load one VICReg checkpoint payload."""
	return load_checkpoint(path, map_location=map_location)


def _validate_payload(  # noqa: C901, PLR0912
	payload: Mapping[str, Any],
	*,
	scaler_required: bool,
	config: Mapping[str, object],
) -> None:
	_validate_vicreg_config_stage(config, label='current')
	missing = sorted(_REQUIRED_KEYS - set(payload))
	if missing:
		raise ValueError(f'VICReg checkpoint is missing keys: {missing!r}')
	for key in ('model_state_dict', 'projector_state_dict', 'optimizer_state_dict'):
		if not isinstance(payload[key], Mapping):
			raise TypeError(f'checkpoint {key} must be a mapping')
	saved_config = payload['config']
	if not isinstance(saved_config, Mapping):
		raise TypeError('checkpoint config must be a mapping')
	_validate_vicreg_config_stage(saved_config, label='saved checkpoint')
	saved_method = resolve_vicreg_pretraining_method(saved_config)
	expected_method = resolve_vicreg_pretraining_method(config)
	if payload['pretraining_method'] != saved_method:
		raise ValueError(
			'checkpoint pretraining_method does not match checkpoint config'
		)
	if saved_method != expected_method:
		raise ValueError(
			'checkpoint pretraining_method does not match current config: '
			f'checkpoint={saved_method!r}, current={expected_method!r}'
		)
	if payload['checkpoint_kind'] != CHECKPOINT_KIND:
		raise ValueError('checkpoint kind is not VICReg pretraining')
	if tuple(payload['trained_parameter_prefixes']) != TRAINED_PARAMETER_PREFIXES:
		raise ValueError('checkpoint trained_parameter_prefixes do not match contract')
	training_state = payload['training_state']
	if not isinstance(training_state, Mapping):
		raise TypeError('checkpoint training_state must be a mapping')
	if training_state.get('stage') != STAGE_VICREG_TRAINING:
		raise ValueError('checkpoint training stage is not vicreg_training')
	if training_state.get('resume_boundary') != 'epoch' or not training_state.get(
		'completed_epoch'
	):
		raise ValueError('VICReg resume requires a completed epoch checkpoint')
	if scaler_required and not isinstance(payload['scaler_state_dict'], Mapping):
		raise ValueError('checkpoint is missing required GradScaler state')
	if not scaler_required and payload['scaler_state_dict'] is not None:
		raise ValueError('checkpoint scaler state is incompatible with this run')
	resume_count = _checkpoint_resume_count(payload)
	_validate_continuation_lineage_contract(
		payload.get('continuation_lineage'),
		config=saved_config,
		resume_count=resume_count,
	)
	_validate_config_compatibility(saved_config, config)


def _checkpoint_resume_count(payload: Mapping[str, Any]) -> int:
	value = payload.get('resume_count', 0)
	_validate_resume_count(value)
	return value


def _validate_vicreg_config_stage(
	config: Mapping[str, object],
	*,
	label: str,
) -> None:
	if config.get('stage') != STAGE_VICREG_TRAINING:
		raise ValueError(
			f'{label} config.stage must be {STAGE_VICREG_TRAINING!r}; '
			f'got {config.get("stage")!r}'
		)


def _validate_resume_count(value: object) -> None:
	if isinstance(value, bool) or not isinstance(value, int):
		raise TypeError('checkpoint resume_count must be an integer')
	if value < 0:
		raise ValueError('checkpoint resume_count must be non-negative')


def _validate_continuation_lineage_contract(
	value: object,
	*,
	config: Mapping[str, object],
	resume_count: int,
) -> None:
	continuation = config.get('continuation')
	if continuation is None:
		if value is not None:
			raise ValueError(
				'base VICReg checkpoint must not have continuation lineage'
			)
		return
	if not isinstance(continuation, Mapping):
		raise TypeError('continuation must be a mapping')
	if not isinstance(value, Mapping):
		raise TypeError('continued VICReg checkpoint is missing continuation lineage')
	if value.get('schema_version') != 1:
		raise ValueError('continuation lineage schema_version must be 1')
	if value.get('init_checkpoint') != continuation.get('init_checkpoint'):
		raise ValueError('continuation lineage init_checkpoint does not match config')
	sha256 = value.get('init_checkpoint_sha256')
	if not isinstance(sha256, str) or len(sha256) != 64 or any(
		character not in '0123456789abcdef' for character in sha256
	):
		raise ValueError('continuation lineage SHA-256 is invalid')
	lineage_resume_count = value.get('resume_count')
	_validate_resume_count(lineage_resume_count)
	if lineage_resume_count != resume_count:
		raise ValueError(
			'checkpoint resume_count does not match continuation lineage'
		)


def _validate_config_compatibility(
	saved: Mapping[str, object],
	current: Mapping[str, object],
) -> None:
	for section in _COMPATIBILITY_SECTIONS:
		if _compatibility_section(saved, section) != _compatibility_section(
			current, section
		):
			raise ValueError(f'resume config section {section!r} does not match')
	for key, value in _mapping(saved, 'train').items():
		if key not in _ALLOWED_TRAIN_OVERRIDES and _mapping(current, 'train').get(
			key
		) != value:
			raise ValueError(f'resume train setting {key!r} does not match')


def _compatibility_section(
	config: Mapping[str, object],
	section: str,
) -> object:
	if section == 'vicreg':
		return vicreg_config_compatibility_identity(config)
	return config.get(section)


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return child


__all__ = [
	'CHECKPOINT_KIND',
	'PRETRAINING_METHOD',
	'TRAINED_PARAMETER_PREFIXES',
	'VICRegResumeState',
	'load_vicreg_checkpoint',
	'restore_vicreg_checkpoint',
	'save_vicreg_checkpoint',
]
