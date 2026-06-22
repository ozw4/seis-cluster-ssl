"""PyTorch collation helpers for amplitude MAE batches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence


def mae_collate_fn(
	samples: Sequence[Mapping[str, object]],
) -> dict[str, torch.Tensor | object]:
	"""Collate amplitude MAE samples into the training batch contract."""
	if not samples:
		msg = 'samples must contain at least one sample'
		raise ValueError(msg)

	return {
		'x': _stack_arrays(samples, 'x'),
		'target': _stack_arrays(samples, 'target'),
		'spatial_mask': _stack_arrays(samples, 'spatial_mask'),
		'visible_spatial_mask': _stack_arrays(samples, 'visible_spatial_mask'),
		'local_valid_mask': _stack_arrays(samples, 'local_valid_mask'),
		'coords': [sample.get('coords') for sample in samples],
	}


def move_batch_to_device(
	batch: Mapping[str, object],
	device: torch.device,
) -> dict[str, object]:
	"""Move tensor values in a batch to ``device`` while preserving metadata."""
	return {
		key: value.to(device) if isinstance(value, torch.Tensor) else value
		for key, value in batch.items()
	}


def _stack_arrays(
	samples: Sequence[Mapping[str, object]],
	key: str,
) -> torch.Tensor:
	arrays = [_require_array(sample, key) for sample in samples]
	first_shape = arrays[0].shape
	for array in arrays:
		if array.shape != first_shape:
			msg = (
				f'all {key!r} arrays must share shape; '
				f'got {array.shape!r}, expected {first_shape!r}'
			)
			raise ValueError(msg)
	return torch.stack([_to_tensor(array) for array in arrays], dim=0)


def _require_array(sample: Mapping[str, object], key: str) -> np.ndarray:
	try:
		value = sample[key]
	except KeyError as exc:
		msg = f'sample is missing required key {key!r}'
		raise KeyError(msg) from exc
	if not isinstance(value, np.ndarray):
		msg = f'{key} must be a NumPy array; got {type(value).__name__}'
		raise TypeError(msg)
	return value


def _to_tensor(array: np.ndarray) -> torch.Tensor:
	return torch.as_tensor(array, dtype=_torch_dtype(array))


def _torch_dtype(array: np.ndarray) -> torch.dtype:
	if np.issubdtype(array.dtype, np.floating):
		return torch.float32
	if np.issubdtype(array.dtype, np.bool_):
		return torch.bool
	if np.issubdtype(array.dtype, np.integer):
		return torch.long
	msg = f'unsupported NumPy dtype for collation: {array.dtype}'
	raise TypeError(msg)


__all__ = ['mae_collate_fn', 'move_batch_to_device']
