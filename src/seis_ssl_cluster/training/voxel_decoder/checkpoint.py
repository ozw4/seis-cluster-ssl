"""Checkpoint and deterministic best-selection policy for voxel decoders."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from seis_ssl_cluster.models.voxel_decoder.spec import (
	validate_voxel_decoder_architecture_mapping,
)
from seis_ssl_cluster.training.checkpoint import capture_rng_state, restore_rng_state

BEST_SELECTION_EPSILON = 1.0e-12
# Version 5 binds the canonical decoder architecture explicitly.
CHECKPOINT_SCHEMA_VERSION = 5


def stable_model_state_sha256(model: torch.nn.Module) -> str:
	"""Hash model tensors canonically, independent of device and torch.save."""
	digest = hashlib.sha256()
	digest.update(b'seis_ssl_cluster_model_state_v1\x00')
	for name, value in sorted(model.state_dict().items()):
		if not isinstance(value, torch.Tensor):
			raise TypeError(f'model state entry {name!r} must be a tensor')
		tensor = (
			value.detach().resolve_conj().resolve_neg().to(device='cpu').contiguous()
		)
		header = json.dumps(
			{
				'name': name,
				'dtype': str(tensor.dtype),
				'shape': list(tensor.shape),
			},
			allow_nan=False,
			sort_keys=True,
			separators=(',', ':'),
		).encode()
		content = tensor.view(torch.uint8).numpy().tobytes(order='C')
		digest.update(len(header).to_bytes(8, byteorder='big'))
		digest.update(header)
		digest.update(len(content).to_bytes(8, byteorder='big'))
		digest.update(content)
	return digest.hexdigest()


def validation_is_better(
	candidate: Mapping[str, object],
	best: Mapping[str, object] | None,
	*,
	epsilon: float = BEST_SELECTION_EPSILON,
) -> bool:
	"""Apply macro-F1, mean-IoU, then weighted-CE ordering."""
	if not math.isfinite(epsilon) or epsilon < 0.0:
		raise ValueError('selection epsilon must be finite and non-negative')
	if best is None:
		return True
	comparisons = (
		(_metric(candidate, 'macro_f1'), _metric(best, 'macro_f1'), True),
		(_metric(candidate, 'mean_iou'), _metric(best, 'mean_iou'), True),
		(_weighted_ce(candidate), _weighted_ce(best), False),
	)
	for new_value, old_value, higher_is_better in comparisons:
		difference = new_value - old_value
		if abs(difference) <= epsilon:
			continue
		return difference > 0.0 if higher_is_better else difference < 0.0
	return False


def make_best_selection_state(
	*, epoch: int, validation_metrics: Mapping[str, object]
) -> dict[str, object]:
	"""Create the persisted state for the fixed selection rule."""
	return {
		'epoch': int(epoch),
		'validation_metrics': _plain_metrics(validation_metrics),
		'rule': [
			'validation.macro_f1 higher',
			'validation.mean_iou higher',
			'validation.weighted_cross_entropy lower',
		],
		'epsilon': BEST_SELECTION_EPSILON,
	}


def best_state_is_improved(
	validation_metrics: Mapping[str, object],
	best_selection_state: Mapping[str, object] | None,
) -> bool:
	"""Compare metrics against a persisted best-selection state."""
	best_metrics: Mapping[str, object] | None = None
	if best_selection_state is not None:
		value = best_selection_state.get('validation_metrics')
		if not isinstance(value, Mapping):
			raise TypeError('best_selection_state.validation_metrics must be a mapping')
		best_metrics = value
	return validation_is_better(validation_metrics, best_metrics)


def save_voxel_decoder_checkpoint(  # noqa: PLR0913
	path: str | Path,
	*,
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	global_step: int,
	resolved_config: Mapping[str, object],
	class_weights: torch.Tensor | list[float] | tuple[float, ...],
	artifact_identities: Mapping[str, object],
	tile_manifest_hashes: Mapping[str, str],
	best_selection_state: Mapping[str, object] | None,
	training_history: list[Mapping[str, object]],
	current_metrics: Mapping[str, object],
	checkpoint_kind: str,
	amp_scaler: torch.amp.GradScaler | None = None,
	batch_index: int | None = None,
	rng_state: Mapping[str, object] | None = None,
	best_checkpoint_sha256: str | None = None,
) -> Path:
	"""Atomically save the complete resumable decoder state."""
	if checkpoint_kind not in {'step', 'epoch', 'completed'}:
		raise ValueError('checkpoint_kind must be step, epoch, or completed')
	decoder_architecture = _resolved_decoder_architecture(resolved_config)
	weights = torch.as_tensor(class_weights, dtype=torch.float32).cpu().tolist()
	payload: dict[str, object] = {
		'schema_version': CHECKPOINT_SCHEMA_VERSION,
		'epoch': int(epoch),
		'global_step': int(global_step),
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'amp_scaler_state_dict': (
			None if amp_scaler is None else amp_scaler.state_dict()
		),
		'runtime_identity': _runtime_identity(model, amp_scaler),
		'best_selection_state': (
			None if best_selection_state is None else dict(best_selection_state)
		),
		'training_history': [dict(item) for item in training_history],
		'history': [dict(item) for item in training_history],
		'current_metrics': dict(current_metrics),
		'metrics': dict(current_metrics),
		'resolved_config': dict(resolved_config),
		'decoder_architecture': decoder_architecture,
		'class_weights': weights,
		'artifact_identities': dict(artifact_identities),
		'tile_manifest_hashes': dict(tile_manifest_hashes),
		'rng_states': dict(capture_rng_state() if rng_state is None else rng_state),
		'checkpoint_kind': checkpoint_kind,
		'batch_index': batch_index,
		'best_checkpoint_sha256': _optional_sha256(best_checkpoint_sha256),
	}
	payload['rng_state'] = payload['rng_states']
	checkpoint_path = Path(path)
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	fd, temporary_name = tempfile.mkstemp(
		prefix=f'.{checkpoint_path.name}.', suffix='.tmp', dir=checkpoint_path.parent
	)
	temporary = Path(temporary_name)
	try:
		with os.fdopen(fd, 'wb') as file_obj:
			torch.save(payload, file_obj)
			file_obj.flush()
			os.fsync(file_obj.fileno())
		temporary.replace(checkpoint_path)
	finally:
		if temporary.exists():
			temporary.unlink()
	return checkpoint_path


def load_voxel_decoder_checkpoint(
	path: str | Path, *, map_location: str | torch.device | None = 'cpu'
) -> dict[str, Any]:
	"""Load and minimally validate a decoder checkpoint."""
	payload = torch.load(Path(path), map_location=map_location, weights_only=False)
	if not isinstance(payload, dict):
		raise TypeError('voxel decoder checkpoint must contain a mapping')
	if payload.get('schema_version') != CHECKPOINT_SCHEMA_VERSION:
		raise ValueError('unsupported voxel decoder checkpoint schema_version')
	required = {
		'schema_version',
		'epoch',
		'global_step',
		'model_state_dict',
		'optimizer_state_dict',
		'amp_scaler_state_dict',
		'runtime_identity',
		'best_selection_state',
		'training_history',
		'current_metrics',
		'resolved_config',
		'decoder_architecture',
		'class_weights',
		'artifact_identities',
		'tile_manifest_hashes',
		'rng_states',
		'checkpoint_kind',
		'best_checkpoint_sha256',
	}
	missing = sorted(required - payload.keys())
	if missing:
		raise ValueError(f'voxel decoder checkpoint missing fields: {missing!r}')
	if payload['checkpoint_kind'] not in {'step', 'epoch', 'completed'}:
		raise ValueError('invalid voxel decoder checkpoint_kind')
	resolved_config = payload['resolved_config']
	if not isinstance(resolved_config, Mapping):
		raise TypeError('checkpoint resolved_config must be a mapping')
	resolved_architecture = _resolved_decoder_architecture(resolved_config)
	checkpoint_architecture = validate_voxel_decoder_architecture_mapping(
		payload['decoder_architecture'],
		field_prefix='checkpoint decoder_architecture',
	)
	if checkpoint_architecture != resolved_architecture:
		raise ValueError(
			'checkpoint decoder_architecture does not match resolved_config.decoder'
		)
	_optional_sha256(payload['best_checkpoint_sha256'])
	return payload


def validate_resume_identity(
	payload: Mapping[str, object],
	*,
	resolved_config: Mapping[str, object],
	class_weights: torch.Tensor | list[float] | tuple[float, ...],
	artifact_identities: Mapping[str, object],
	tile_manifest_hashes: Mapping[str, str],
) -> None:
	"""Reject resume when any data, architecture, manifest, or weight changed."""
	checkpoint_config = payload.get('resolved_config')
	if not isinstance(checkpoint_config, Mapping):
		raise TypeError('checkpoint resolved_config must be a mapping')
	checkpoint_architecture = validate_voxel_decoder_architecture_mapping(
		payload.get('decoder_architecture'),
		field_prefix='checkpoint decoder_architecture',
	)
	current_architecture = _resolved_decoder_architecture(resolved_config)
	if checkpoint_architecture != current_architecture:
		raise ValueError('resume identity mismatch: decoder architecture')
	for section in ('model', 'decoder', 'tiles', 'dataset', 'train'):
		if checkpoint_config.get(section) != resolved_config.get(section):
			raise ValueError(f'resume identity mismatch: {section}')
	if payload.get('artifact_identities') != dict(artifact_identities):
		raise ValueError('resume identity mismatch: source artifacts')
	if payload.get('tile_manifest_hashes') != dict(tile_manifest_hashes):
		raise ValueError('resume identity mismatch: tile manifests')
	expected_weights = torch.as_tensor(class_weights, dtype=torch.float32).cpu()
	actual_weights = torch.as_tensor(payload.get('class_weights'), dtype=torch.float32)
	if actual_weights.shape != expected_weights.shape or not torch.equal(
		actual_weights, expected_weights
	):
		raise ValueError('resume identity mismatch: class weights')


def _resolved_decoder_architecture(
	resolved_config: Mapping[str, object],
) -> dict[str, object]:
	return validate_voxel_decoder_architecture_mapping(
		resolved_config.get('decoder'), field_prefix='resolved_config.decoder'
	)


def restore_voxel_decoder_checkpoint(
	payload: Mapping[str, object],
	*,
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	amp_scaler: torch.amp.GradScaler | None = None,
) -> None:
	"""Restore model, optimizer, optional scaler, and all RNG streams."""
	expected_runtime = payload.get('runtime_identity')
	if not isinstance(expected_runtime, Mapping):
		raise TypeError('checkpoint runtime_identity must be a mapping')
	actual_runtime = _runtime_identity(model, amp_scaler)
	if expected_runtime != actual_runtime:
		raise ValueError(
			'resume runtime mismatch: '
			f'checkpoint={dict(expected_runtime)!r}, current={actual_runtime!r}'
		)
	model.load_state_dict(payload['model_state_dict'])  # type: ignore[arg-type]
	optimizer.load_state_dict(payload['optimizer_state_dict'])  # type: ignore[arg-type]
	scaler_state = payload.get('amp_scaler_state_dict')
	if amp_scaler is not None:
		if not isinstance(scaler_state, Mapping):
			raise ValueError('AMP resume requires amp_scaler_state_dict')
		amp_scaler.load_state_dict(dict(scaler_state))
	elif scaler_state is not None:
		raise ValueError('non-AMP resume cannot restore amp_scaler_state_dict')
	rng_states = payload.get('rng_states')
	if not isinstance(rng_states, Mapping):
		raise TypeError('checkpoint rng_states must be a mapping')
	restore_rng_state({'rng_state': rng_states})


def _metric(metrics: Mapping[str, object], name: str) -> float:
	value = metrics.get(name)
	if not isinstance(value, int | float) or isinstance(value, bool):
		raise TypeError(f'validation metric {name} must be numeric')
	metric = float(value)
	if not math.isfinite(metric):
		raise ValueError(f'validation metric {name} must be finite')
	return metric


def _weighted_ce(metrics: Mapping[str, object]) -> float:
	if 'weighted_cross_entropy' in metrics:
		return _metric(metrics, 'weighted_cross_entropy')
	return _metric(metrics, 'loss')


def _plain_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
	plain: dict[str, object] = {}
	for key, value in metrics.items():
		if isinstance(value, torch.Tensor):
			plain[key] = value.detach().cpu().tolist()
		elif hasattr(value, 'tolist'):
			plain[key] = value.tolist()  # type: ignore[union-attr]
		else:
			plain[key] = value
	return plain


def _optional_sha256(value: object) -> str | None:
	if value is None:
		return None
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise ValueError(
			'best_checkpoint_sha256 must be a lowercase SHA-256 hex digest'
		)
	return value


def _runtime_identity(
	model: torch.nn.Module,
	amp_scaler: torch.amp.GradScaler | None,
) -> dict[str, object]:
	parameter = next(model.parameters(), None)
	device = torch.device('cpu') if parameter is None else parameter.device
	return {
		'device': str(device),
		'amp_scaler': amp_scaler is not None,
	}


# Concise compatibility names for callers that operate only on decoder checkpoints.
is_better_checkpoint = validation_is_better
is_better_validation_metrics = validation_is_better
save_checkpoint = save_voxel_decoder_checkpoint
load_checkpoint = load_voxel_decoder_checkpoint


__all__ = [
	'BEST_SELECTION_EPSILON',
	'CHECKPOINT_SCHEMA_VERSION',
	'best_state_is_improved',
	'is_better_checkpoint',
	'is_better_validation_metrics',
	'load_checkpoint',
	'load_voxel_decoder_checkpoint',
	'make_best_selection_state',
	'restore_voxel_decoder_checkpoint',
	'save_checkpoint',
	'save_voxel_decoder_checkpoint',
	'stable_model_state_sha256',
	'validate_resume_identity',
	'validation_is_better',
]
