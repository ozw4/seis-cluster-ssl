"""Deterministic in-bounds crop sampling utilities for amplitude pretraining."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.data.schema import CropRequest

if TYPE_CHECKING:
	from collections.abc import Sequence

XYZ = tuple[int, int, int]


def rng_for_sample(seed: int, epoch: int, index: int) -> np.random.Generator:
	"""Return the deterministic RNG for one epoch-aware sample index."""
	seed_value = _validate_nonnegative_int(seed, 'seed')
	epoch_value = _validate_nonnegative_int(epoch, 'epoch')
	index_value = _validate_nonnegative_int(index, 'index')
	seed_sequence = np.random.SeedSequence([seed_value, epoch_value, index_value])
	return np.random.default_rng(seed_sequence)


def select_round_robin_index(num_surveys: int, sample_index: int) -> int:
	"""Return the manifest-order survey index for ``sample_index``."""
	count = _validate_positive_int(num_surveys, 'num_surveys')
	index = _validate_nonnegative_int(sample_index, 'sample_index')
	return index % count


def sample_random_local_crop(
	shape_xyz: Sequence[int],
	local_size_xyz: Sequence[int],
	rng: np.random.Generator,
	*,
	survey_id: str = '',
) -> CropRequest:
	"""Sample a fully in-bounds local crop in `[x, y, z]` order."""
	shape = _validate_positive_xyz(shape_xyz, 'shape_xyz')
	local_size = _validate_positive_xyz(local_size_xyz, 'local_size_xyz')
	_validate_crop_fits(shape, local_size)
	start = tuple(
		int(rng.integers(0, shape_axis - size_axis + 1))
		for shape_axis, size_axis in zip(shape, local_size, strict=True)
	)
	return CropRequest(
		survey_id=survey_id,
		start_xyz=cast('XYZ', start),
		size_xyz=local_size,
	)


def expand_request_with_margin(
	request: CropRequest,
	margin_xyz: Sequence[int],
) -> tuple[CropRequest, tuple[slice, slice, slice]]:
	"""Return a margin-expanded request and slices for the center payload."""
	margin = _validate_nonnegative_xyz(margin_xyz, 'margin_xyz')
	start = _validate_xyz(request.start_xyz, 'request.start_xyz')
	size = _validate_positive_xyz(request.size_xyz, 'request.size_xyz')
	compute_start = tuple(
		start_axis - margin_axis
		for start_axis, margin_axis in zip(start, margin, strict=True)
	)
	compute_size = tuple(
		size_axis + 2 * margin_axis
		for size_axis, margin_axis in zip(size, margin, strict=True)
	)
	payload_slices = tuple(
		slice(margin_axis, margin_axis + size_axis)
		for margin_axis, size_axis in zip(margin, size, strict=True)
	)
	return (
		CropRequest(
			survey_id=request.survey_id,
			start_xyz=cast('XYZ', compute_start),
			size_xyz=cast('XYZ', compute_size),
		),
		cast('tuple[slice, slice, slice]', payload_slices),
	)


def required_zero_mask_margin_xyz(
	*,
	z_sample_influence_radius: int,
	xy_trace_influence_radius: int,
) -> XYZ:
	"""Return raw margin needed to avoid crop-boundary-biased zero masking."""
	z_radius = _validate_nonnegative_int(
		z_sample_influence_radius,
		'z_sample_influence_radius',
	)
	xy_radius = _validate_nonnegative_int(
		xy_trace_influence_radius,
		'xy_trace_influence_radius',
	)
	return (xy_radius, xy_radius, z_radius)


def validate_crop_fits(
	shape_xyz: Sequence[int],
	local_size_xyz: Sequence[int],
) -> None:
	"""Raise if ``local_size_xyz`` cannot fit fully inside ``shape_xyz``."""
	_validate_crop_fits(
		_validate_positive_xyz(shape_xyz, 'shape_xyz'),
		_validate_positive_xyz(local_size_xyz, 'local_size_xyz'),
	)


def _validate_crop_fits(shape_xyz: XYZ, local_size_xyz: XYZ) -> None:
	if all(
		shape_axis >= size_axis
		for shape_axis, size_axis in zip(shape_xyz, local_size_xyz, strict=True)
	):
		return
	msg = (
		f'local crop size {list(local_size_xyz)} does not fit inside '
		f'volume shape {list(shape_xyz)}'
	)
	raise ValueError(msg)


def _validate_xyz(value: Sequence[int], name: str) -> XYZ:
	if (
		isinstance(value, str)
		or len(value) != 3
		or not all(
			not isinstance(axis, bool) and isinstance(axis, Integral)
			for axis in value
		)
	):
		msg = f'{name} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	return cast('XYZ', tuple(int(axis) for axis in value))


def _validate_positive_xyz(value: Sequence[int], name: str) -> XYZ:
	xyz = _validate_xyz(value, name)
	if any(axis <= 0 for axis in xyz):
		msg = f'{name} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _validate_nonnegative_xyz(value: Sequence[int], name: str) -> XYZ:
	xyz = _validate_xyz(value, name)
	if any(axis < 0 for axis in xyz):
		msg = f'{name} values must be nonnegative; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _validate_positive_int(value: int, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	integer = int(value)
	if integer <= 0:
		msg = f'{name} must be positive; got {integer!r}'
		raise ValueError(msg)
	return integer


def _validate_nonnegative_int(value: int, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	integer = int(value)
	if integer < 0:
		msg = f'{name} must be nonnegative; got {integer!r}'
		raise ValueError(msg)
	return integer


__all__ = [
	'expand_request_with_margin',
	'required_zero_mask_margin_xyz',
	'rng_for_sample',
	'sample_random_local_crop',
	'select_round_robin_index',
	'validate_crop_fits',
]
