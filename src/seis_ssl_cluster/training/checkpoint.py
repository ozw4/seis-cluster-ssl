"""Checkpoint IO for amplitude MAE pretraining."""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_checkpoint(  # noqa: PLR0913
	path: str | Path,
	*,
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	config: Mapping[str, object],
	package_version: str | None = None,
	metrics: Mapping[str, float] | None = None,
	global_step: int | None = None,
	amp_enabled: bool | None = None,
	scaler: torch.amp.GradScaler | None = None,
	training_state: Mapping[str, object] | None = None,
	rng_state: Mapping[str, object] | None = None,
) -> Path:
	"""Atomically write a training checkpoint and return its path."""
	checkpoint_path = Path(path)
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	resolved_amp_enabled = False if amp_enabled is None else bool(amp_enabled)
	if resolved_amp_enabled and scaler is None:
		msg = 'scaler is required when amp_enabled is true'
		raise ValueError(msg)
	payload: dict[str, object] = {
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': int(epoch),
		'global_step': 0 if global_step is None else int(global_step),
		'amp_enabled': resolved_amp_enabled,
		'scaler_state_dict': None if scaler is None else scaler.state_dict(),
		'config': _to_plain_value(config),
		'package_version': package_version,
		'metrics': {} if metrics is None else dict(metrics),
		'rng_state': dict(capture_rng_state() if rng_state is None else rng_state),
		'training_state': (
			{} if training_state is None else _to_plain_value(training_state)
		),
	}

	fd, tmp_name = tempfile.mkstemp(
		prefix=f'.{checkpoint_path.name}.',
		suffix='.tmp',
		dir=checkpoint_path.parent,
	)
	tmp_path = Path(tmp_name)
	try:
		with os.fdopen(fd, 'wb') as file_obj:
			torch.save(payload, file_obj)
			file_obj.flush()
			os.fsync(file_obj.fileno())
		tmp_path.replace(checkpoint_path)
	finally:
		if tmp_path.exists():
			tmp_path.unlink()
	return checkpoint_path


def load_checkpoint(
	path: str | Path,
	map_location: str | torch.device | None = None,
) -> dict[str, Any]:
	"""Load a checkpoint payload from disk."""
	return torch.load(Path(path), map_location=map_location, weights_only=False)


def capture_rng_state() -> dict[str, object]:
	"""Capture Python, NumPy, and Torch RNG state for deterministic resume."""
	state: dict[str, object] = {
		'python': random.getstate(),
		'numpy': np.random.get_state(),  # noqa: NPY002
		'torch': torch.get_rng_state(),
	}
	if torch.cuda.is_available():
		state['torch_cuda'] = torch.cuda.get_rng_state_all()
	return state


def restore_rng_state(payload: Mapping[str, object]) -> None:
	"""Restore RNG state from ``payload``.

	Partial or malformed RNG state is rejected so checkpoint resume cannot
	quietly continue with nondeterministic state.
	"""
	rng_state = payload.get('rng_state')
	if not isinstance(rng_state, Mapping):
		msg = 'checkpoint rng_state must be a mapping'
		raise TypeError(msg)
	python_state = _required_rng_value(rng_state, 'python')
	if not isinstance(python_state, tuple):
		msg = 'checkpoint rng_state.python must be a tuple'
		raise TypeError(msg)

	numpy_state = _required_rng_value(rng_state, 'numpy')
	if not _is_numpy_rng_state(numpy_state):
		msg = 'checkpoint rng_state.numpy must be a NumPy RNG state tuple'
		raise TypeError(msg)

	torch_state = _required_rng_value(rng_state, 'torch')
	if not isinstance(torch_state, torch.Tensor):
		msg = 'checkpoint rng_state.torch must be a tensor'
		raise TypeError(msg)

	cuda_state = rng_state.get('torch_cuda')
	if cuda_state is not None and not _is_cuda_rng_state(cuda_state):
		msg = 'checkpoint rng_state.torch_cuda must be a list of tensors'
		raise TypeError(msg)
	random.setstate(python_state)
	np.random.set_state(numpy_state)  # noqa: NPY002
	torch.set_rng_state(torch_state.cpu())
	if torch.cuda.is_available() and isinstance(cuda_state, list):
		torch.cuda.set_rng_state_all(cuda_state)


def _required_rng_value(rng_state: Mapping[str, object], key: str) -> object:
	if key not in rng_state:
		msg = f'checkpoint rng_state is missing {key}'
		raise ValueError(msg)
	value = rng_state[key]
	if value is None:
		msg = f'checkpoint rng_state.{key} must not be null'
		raise TypeError(msg)
	return value


def _is_numpy_rng_state(value: object) -> bool:
	return (
		isinstance(value, tuple)
		and len(value) == 5
		and isinstance(value[0], str)
		and isinstance(value[1], np.ndarray)
		and isinstance(value[2], int)
		and isinstance(value[3], int)
		and isinstance(value[4], float)
	)


def _is_cuda_rng_state(value: object) -> bool:
	return isinstance(value, list) and all(
		isinstance(child, torch.Tensor) for child in value
	)


def _to_plain_value(value: object) -> object:
	if isinstance(value, Mapping):
		return {str(key): _to_plain_value(child) for key, child in value.items()}
	if isinstance(value, list | tuple):
		return [_to_plain_value(child) for child in value]
	if isinstance(value, Path):
		return str(value)
	return value


__all__ = [
	'capture_rng_state',
	'load_checkpoint',
	'restore_rng_state',
	'save_checkpoint',
]
