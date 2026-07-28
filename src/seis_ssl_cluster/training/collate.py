"""PyTorch collation helpers for amplitude MAE batches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

if TYPE_CHECKING:
	from collections.abc import Sequence


def mae_collate_fn(
	samples: Sequence[Mapping[str, object]],
) -> dict[str, torch.Tensor | object]:
	"""Collate amplitude MAE samples into the training batch contract."""
	if not samples:
		msg = 'samples must contain at least one sample'
		raise ValueError(msg)

	spatial_mask = _stack_arrays(samples, 'spatial_mask')
	return {
		'x': _stack_arrays(samples, 'x'),
		'spatial_mask': spatial_mask,
		'equal_visible_count': _has_equal_visible_count(spatial_mask),
		'local_valid_mask': _stack_arrays(samples, 'local_valid_mask'),
		'coords': [sample.get('coords') for sample in samples],
	}


def strat_pseudo_target_collate_fn(
	samples: Sequence[Mapping[str, object]],
) -> dict[str, torch.Tensor | object]:
	"""Collate token-aligned stratigraphic pseudo-target samples."""
	if not samples:
		msg = 'samples must contain at least one sample'
		raise ValueError(msg)

	return {
		'x': _stack_arrays(samples, 'x'),
		'local_valid_mask': _stack_arrays(samples, 'local_valid_mask'),
		'strat_labels': _stack_arrays(samples, 'strat_labels'),
		'strat_confidence': _stack_arrays(samples, 'strat_confidence'),
		'strat_boundary_weight': _stack_arrays(
			samples,
			'strat_boundary_weight',
		),
		'strat_valid_mask': _stack_arrays(samples, 'strat_valid_mask'),
		'coords': [sample.get('coords') for sample in samples],
	}


def strat_multi_head_target_collate_fn(
	samples: Sequence[Mapping[str, object]],
) -> dict[str, torch.Tensor | object]:
	"""Collate ordered nested stratigraphic pseudo-target heads."""
	if not samples:
		msg = 'samples must contain at least one sample'
		raise ValueError(msg)
	head_keys, fields_by_head = _multi_head_contract(samples[0])
	for sample in samples[1:]:
		other_head_keys, _ = _multi_head_contract(sample)
		if other_head_keys != head_keys:
			raise ValueError('samples must have identical multi-head target order')
	targets: dict[str, dict[str, torch.Tensor]] = {}
	for head_key in head_keys:
		targets[head_key] = {
			field: _stack_multi_head_field(samples, head_key, field)
			for field in fields_by_head[head_key]
		}
	return {
		'x': _stack_arrays(samples, 'x'),
		'local_valid_mask': _stack_arrays(samples, 'local_valid_mask'),
		'strat_multi_targets': targets,
		'coords': [sample.get('coords') for sample in samples],
	}


def strat_multi_head_posterior_collate_fn(
	samples: Sequence[Mapping[str, object]],
) -> dict[str, torch.Tensor | object]:
	"""Collate ordered nested soft posterior targets."""
	if not samples:
		raise ValueError('samples must contain at least one sample')
	head_keys, fields_by_head = _multi_head_posterior_contract(samples[0])
	for sample in samples[1:]:
		other_head_keys, _ = _multi_head_posterior_contract(sample)
		if other_head_keys != head_keys:
			raise ValueError('samples must have identical multi-head posterior order')
	posteriors: dict[str, dict[str, torch.Tensor]] = {}
	for head_key in head_keys:
		posteriors[head_key] = {
			field: _stack_multi_head_posterior_field(samples, head_key, field)
			for field in fields_by_head[head_key]
		}
	return {
		'x': _stack_arrays(samples, 'x'),
		'local_valid_mask': _stack_arrays(samples, 'local_valid_mask'),
		'strat_multi_posteriors': posteriors,
		'coords': [sample.get('coords') for sample in samples],
	}


def move_batch_to_device(
	batch: Mapping[str, object],
	device: torch.device,
	*,
	non_blocking: bool = False,
) -> dict[str, object]:
	"""Move tensor values in a batch to ``device`` while preserving metadata."""
	return cast(
		'dict[str, object]',
		_move_batch_value(batch, device, non_blocking=non_blocking),
	)


def _move_batch_value(
	value: object,
	device: torch.device,
	*,
	non_blocking: bool,
) -> object:
	if isinstance(value, torch.Tensor):
		return _move_tensor_to_device(value, device, non_blocking=non_blocking)
	if isinstance(value, Mapping):
		moved = {
			key: _move_batch_value(item, device, non_blocking=non_blocking)
			for key, item in value.items()
		}
		unchanged = all(moved[key] is item for key, item in value.items())
		return value if unchanged else moved
	if isinstance(value, list):
		moved = [
			_move_batch_value(item, device, non_blocking=non_blocking)
			for item in value
		]
		unchanged = all(left is right for left, right in zip(moved, value, strict=True))
		return value if unchanged else moved
	if isinstance(value, tuple):
		moved = tuple(
			_move_batch_value(item, device, non_blocking=non_blocking)
			for item in value
		)
		unchanged = all(left is right for left, right in zip(moved, value, strict=True))
		return value if unchanged else moved
	return value


def _move_tensor_to_device(
	value: torch.Tensor,
	device: torch.device,
	*,
	non_blocking: bool,
) -> torch.Tensor:
	use_non_blocking = (
		non_blocking and device.type == 'cuda' and value.is_pinned()
	)
	return value.to(device, non_blocking=use_non_blocking)


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


def _multi_head_contract(
	sample: Mapping[str, object],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
	targets = sample.get('strat_multi_targets')
	if not isinstance(targets, Mapping):
		raise TypeError('strat_multi_targets must be a mapping')
	head_keys = tuple(targets)
	head_ks = tuple(_multi_head_key_k(head_key) for head_key in head_keys)
	if (
		len(head_ks) < 2
		or tuple(sorted(head_ks)) != head_ks
		or len(set(head_ks)) != len(head_ks)
	):
		raise ValueError(
			'strat_multi_targets head keys must be at least two canonical '
			'ascending k{K} keys'
		)
	fields_by_head: dict[str, tuple[str, ...]] = {}
	for head_key in head_keys:
		target = targets[head_key]
		if not isinstance(target, Mapping):
			raise TypeError('strat_multi_targets must map head keys to mappings')
		fields = tuple(target)
		if fields != ('labels', 'confidence', 'boundary_weight', 'valid_mask'):
			raise ValueError(
				f'strat_multi_targets[{head_key!r}] must contain ordered fields '
				"('labels', 'confidence', 'boundary_weight', 'valid_mask')"
			)
		fields_by_head[head_key] = fields
	return head_keys, fields_by_head


def _multi_head_posterior_contract(
	sample: Mapping[str, object],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
	posteriors = sample.get('strat_multi_posteriors')
	if not isinstance(posteriors, Mapping):
		raise TypeError('strat_multi_posteriors must be a mapping')
	head_keys = tuple(posteriors)
	head_ks = tuple(_multi_head_key_k(head_key) for head_key in head_keys)
	if head_ks != (6, 8, 10):
		raise ValueError(
			'strat_multi_posteriors head keys must be canonical ascending '
			'k6, k8, k10',
		)
	fields_by_head: dict[str, tuple[str, ...]] = {}
	for head_key, k in zip(head_keys, head_ks, strict=True):
		target = posteriors[head_key]
		if not isinstance(target, Mapping):
			raise TypeError('strat_multi_posteriors must map head keys to mappings')
		fields = tuple(target)
		if fields != ('posterior', 'valid_mask'):
			raise ValueError(
				f'strat_multi_posteriors[{head_key!r}] must contain ordered '
				"fields ('posterior', 'valid_mask')",
			)
		posterior = _require_array(target, 'posterior')
		valid_mask = _require_array(target, 'valid_mask')
		if posterior.dtype != np.float32 or posterior.ndim != 4:
			raise ValueError(
				f'strat_multi_posteriors[{head_key!r}].posterior must be '
				f'float32 [X,Y,Z,{k}]',
			)
		if posterior.shape[-1] != k:
			raise ValueError(
				f'strat_multi_posteriors[{head_key!r}] posterior last dimension '
				f'must equal {k}',
			)
		if valid_mask.dtype != np.bool_ or valid_mask.shape != posterior.shape[:3]:
			raise ValueError(
				f'strat_multi_posteriors[{head_key!r}] valid_mask shape/dtype '
				'mismatch',
			)
		fields_by_head[head_key] = fields
	return head_keys, fields_by_head


def _multi_head_key_k(head_key: object) -> int:
	"""Return the K encoded by one canonical multi-head target key."""
	if not isinstance(head_key, str):
		raise TypeError('strat_multi_targets head keys must be strings')
	if not head_key.startswith('k'):
		raise ValueError(
			f'strat_multi_targets head key {head_key!r} must use canonical k{{K}}'
		)
	try:
		k = int(head_key[1:])
	except ValueError as exc:
		raise ValueError(
			f'strat_multi_targets head key {head_key!r} must use canonical k{{K}}'
		) from exc
	if k < 2 or head_key != f'k{k}':
		raise ValueError(
			f'strat_multi_targets head key {head_key!r} must use canonical k{{K}}'
		)
	return k


def _stack_multi_head_field(
	samples: Sequence[Mapping[str, object]],
	head_key: str,
	field: str,
) -> torch.Tensor:
	arrays: list[np.ndarray] = []
	for sample in samples:
		head_keys, _ = _multi_head_contract(sample)
		if head_key not in head_keys:
			raise ValueError(f'sample is missing multi-head target {head_key!r}')
		targets = sample['strat_multi_targets']
		if not isinstance(targets, Mapping):
			raise TypeError('strat_multi_targets must be a mapping')
		target = cast('Mapping[str, object]', targets[head_key])
		arrays.append(_require_array(target, field))
	first = arrays[0]
	for array in arrays[1:]:
		if array.shape != first.shape:
			raise ValueError(
				f'all {head_key!r} {field!r} arrays must share shape; got '
				f'{array.shape!r}, expected {first.shape!r}'
			)
		if array.dtype != first.dtype:
			raise TypeError(
				f'all {head_key!r} {field!r} arrays must share dtype; got '
				f'{array.dtype}, expected {first.dtype}'
			)
	return torch.stack([_to_tensor(array) for array in arrays], dim=0)


def _stack_multi_head_posterior_field(
	samples: Sequence[Mapping[str, object]],
	head_key: str,
	field: str,
) -> torch.Tensor:
	arrays: list[np.ndarray] = []
	for sample in samples:
		head_keys, _ = _multi_head_posterior_contract(sample)
		if head_key not in head_keys:
			raise ValueError(f'sample is missing multi-head posterior {head_key!r}')
		posteriors = sample['strat_multi_posteriors']
		if not isinstance(posteriors, Mapping):  # pragma: no cover - validated above
			raise TypeError('strat_multi_posteriors must be a mapping')
		target = cast('Mapping[str, object]', posteriors[head_key])
		arrays.append(_require_array(target, field))
	first = arrays[0]
	for array in arrays[1:]:
		if array.shape != first.shape:
			raise ValueError(
				f'all {head_key!r} {field!r} posterior arrays must share shape; '
				f'got {array.shape!r}, expected {first.shape!r}',
			)
		if array.dtype != first.dtype:
			raise TypeError(
				f'all {head_key!r} {field!r} posterior arrays must share dtype; '
				f'got {array.dtype}, expected {first.dtype}',
			)
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


def _has_equal_visible_count(mask: torch.Tensor) -> bool:
	visible_counts = (~mask).reshape(mask.shape[0], -1).sum(dim=1)
	return torch.equal(
		visible_counts,
		visible_counts[:1].expand_as(visible_counts),
	)


def _torch_dtype(array: np.ndarray) -> torch.dtype:
	if np.issubdtype(array.dtype, np.floating):
		return torch.float32
	if np.issubdtype(array.dtype, np.bool_):
		return torch.bool
	if np.issubdtype(array.dtype, np.integer):
		return torch.long
	msg = f'unsupported NumPy dtype for collation: {array.dtype}'
	raise TypeError(msg)


__all__ = [
	'mae_collate_fn',
	'move_batch_to_device',
	'strat_multi_head_posterior_collate_fn',
	'strat_multi_head_target_collate_fn',
	'strat_pseudo_target_collate_fn',
]
