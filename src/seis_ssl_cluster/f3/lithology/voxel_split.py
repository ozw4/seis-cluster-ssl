"""Pure voxel supervision split construction for the F3 lithology benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.f3.lithology.voxel_geometry import (
	valid_tokens_to_voxel_mask,
)
from seis_ssl_cluster.f3.splits import (
	F3LineGeometry,
	F3SliceSplitRecord,
	resolve_f3_slice_array_index,
)

if TYPE_CHECKING:
	from collections.abc import Sequence

UNSUPERVISED_VOXEL_SPLIT = np.uint8(0)
TRAIN_VOXEL_SPLIT = np.uint8(1)
VALIDATION_VOXEL_SPLIT = np.uint8(2)


@dataclass(frozen=True)
class F3VoxelSupervisionSplitSummary:
	"""Counts describing selection, filtering, deduplication, and precedence."""

	raw_train_assignments: int
	raw_validation_assignments: int
	unique_train_voxels_before_precedence: int
	unique_validation_voxels: int
	cross_split_overlap_voxels: int
	train_voxels_removed_by_validation_precedence: int
	final_train_voxels: int
	final_validation_voxels: int
	invalid_label_voxels_on_selected_slices: int
	invalid_reference_voxels_on_selected_slices: int
	ignored_z_border_voxels: int


@dataclass(frozen=True)
class F3VoxelSupervisionSplit:
	"""Dense voxel split grid and its construction summary."""

	split_grid: np.ndarray
	summary: F3VoxelSupervisionSplitSummary


def build_f3_voxel_supervision_split(  # noqa: PLR0913
	records: Sequence[F3SliceSplitRecord],
	*,
	geometry: F3LineGeometry,
	label_volume: np.ndarray,
	class_ids: Sequence[int],
	valid_tokens: np.ndarray,
	patch_size_xyz: Sequence[int],
	ignore_z_border_samples: int = 1,
) -> F3VoxelSupervisionSplit:
	"""Build the canonical train/validation voxel supervision split grid."""
	labels = np.asarray(label_volume)
	_validate_inputs(
		geometry=geometry,
		labels=labels,
		class_ids=class_ids,
		valid_tokens=np.asarray(valid_tokens),
		patch_size_xyz=patch_size_xyz,
		ignore_z_border_samples=ignore_z_border_samples,
	)
	known_ids = np.asarray(tuple(int(class_id) for class_id in class_ids))
	reference_valid = valid_tokens_to_voxel_mask(
		valid_tokens,
		patch_size_xyz=patch_size_xyz,
		volume_shape_xyz=labels.shape,
	)
	label_valid = (labels >= 0) & np.isin(labels, known_ids)
	border_valid = np.ones(labels.shape, dtype=np.bool_)
	border = int(ignore_z_border_samples)
	if border:
		border_valid[:, :, :border] = False
		border_valid[:, :, -border:] = False
	eligible = label_valid & reference_valid & border_valid

	selected = np.zeros(labels.shape, dtype=np.bool_)
	train = np.zeros(labels.shape, dtype=np.bool_)
	validation = np.zeros(labels.shape, dtype=np.bool_)
	raw_train = 0
	raw_validation = 0
	for record in records:
		if record.split not in {'train', 'validation'}:
			msg = f'record split must be train or validation; got {record.split!r}'
			raise ValueError(msg)
		index = resolve_f3_slice_array_index(record, geometry)
		plane = _plane_index(record.slice_type, index)
		selected[plane] = True
		raw_count = int(np.count_nonzero(eligible[plane]))
		if record.split == 'train':
			raw_train += raw_count
			train[plane] |= eligible[plane]
		else:
			raw_validation += raw_count
			validation[plane] |= eligible[plane]

	overlap = train & validation
	final_train = train & ~validation
	if not np.any(final_train):
		msg = 'voxel supervision split must contain at least one train voxel'
		raise ValueError(msg)
	if not np.any(validation):
		msg = 'voxel supervision split must contain at least one validation voxel'
		raise ValueError(msg)

	grid = np.zeros(labels.shape, dtype=np.uint8)
	grid[final_train] = TRAIN_VOXEL_SPLIT
	grid[validation] = VALIDATION_VOXEL_SPLIT
	summary = F3VoxelSupervisionSplitSummary(
		raw_train_assignments=raw_train,
		raw_validation_assignments=raw_validation,
		unique_train_voxels_before_precedence=int(np.count_nonzero(train)),
		unique_validation_voxels=int(np.count_nonzero(validation)),
		cross_split_overlap_voxels=int(np.count_nonzero(overlap)),
		train_voxels_removed_by_validation_precedence=int(np.count_nonzero(overlap)),
		final_train_voxels=int(np.count_nonzero(final_train)),
		final_validation_voxels=int(np.count_nonzero(validation)),
		invalid_label_voxels_on_selected_slices=int(
			np.count_nonzero(selected & ~label_valid)
		),
		invalid_reference_voxels_on_selected_slices=int(
			np.count_nonzero(selected & ~reference_valid)
		),
		ignored_z_border_voxels=int(np.count_nonzero(selected & ~border_valid)),
	)
	return F3VoxelSupervisionSplit(split_grid=grid, summary=summary)


def _plane_index(slice_type: str, index: int) -> tuple[object, object, object]:
	if slice_type == 'inline':
		return index, slice(None), slice(None)
	if slice_type == 'crossline':
		return slice(None), index, slice(None)
	msg = f'slice_type must be inline or crossline; got {slice_type!r}'
	raise ValueError(msg)


def _validate_inputs(  # noqa: C901, PLR0913
	*,
	geometry: F3LineGeometry,
	labels: np.ndarray,
	class_ids: Sequence[int],
	valid_tokens: np.ndarray,
	patch_size_xyz: Sequence[int],
	ignore_z_border_samples: int,
) -> None:
	if labels.ndim != 3:
		msg = f'label_volume must be 3D [X, Y, Z]; got shape {labels.shape!r}'
		raise ValueError(msg)
	if not np.issubdtype(labels.dtype, np.integer) or labels.dtype == np.bool_:
		msg = f'label_volume dtype must be integer; got {labels.dtype}'
		raise TypeError(msg)
	if tuple(geometry.shape_xyz) != labels.shape:
		msg = (
			'geometry shape_xyz must match label_volume shape; '
			f'got {geometry.shape_xyz!r} and {labels.shape!r}'
		)
		raise ValueError(msg)
	if geometry.inline_max - geometry.inline_min + 1 != labels.shape[0]:
		msg = 'inline geometry bounds must match geometry shape_xyz'
		raise ValueError(msg)
	if geometry.crossline_max - geometry.crossline_min + 1 != labels.shape[1]:
		msg = 'crossline geometry bounds must match geometry shape_xyz'
		raise ValueError(msg)
	if len(class_ids) == 0:
		msg = 'class_ids must contain at least one known class ID'
		raise ValueError(msg)
	if any(
		not isinstance(class_id, Integral) or isinstance(class_id, bool)
		for class_id in class_ids
	):
		msg = 'class_ids must contain integers'
		raise TypeError(msg)
	if valid_tokens.ndim != 3:
		msg = f'valid_tokens must be 3D [TX, TY, TZ]; got shape {valid_tokens.shape!r}'
		raise ValueError(msg)
	if valid_tokens.dtype != np.bool_:
		msg = f'valid_tokens dtype must be bool; got {valid_tokens.dtype}'
		raise TypeError(msg)
	patch = _validated_patch_size(patch_size_xyz)
	expected_tokens = tuple(
		(size + patch_size - 1) // patch_size
		for size, patch_size in zip(labels.shape, patch, strict=True)
	)
	if valid_tokens.shape != expected_tokens:
		msg = (
			'valid_tokens shape must be the canonical token grid for label_volume '
			f'and patch_size_xyz; expected {expected_tokens!r}, '
			f'got {valid_tokens.shape!r}'
		)
		raise ValueError(msg)
	if (
		not isinstance(ignore_z_border_samples, Integral)
		or isinstance(ignore_z_border_samples, bool)
	):
		msg = 'ignore_z_border_samples must be a non-negative integer'
		raise TypeError(msg)
	if ignore_z_border_samples < 0:
		msg = 'ignore_z_border_samples must be a non-negative integer'
		raise ValueError(msg)


def _validated_patch_size(value: Sequence[int]) -> tuple[int, int, int]:
	if isinstance(value, (str, bytes)) or len(value) != 3:
		msg = 'patch_size_xyz must be a positive integer triple'
		raise ValueError(msg)
	if any(not isinstance(item, Integral) or isinstance(item, bool) for item in value):
		msg = 'patch_size_xyz must be a positive integer triple'
		raise TypeError(msg)
	result = tuple(int(item) for item in value)
	if any(item <= 0 for item in result):
		msg = f'patch_size_xyz must be a positive integer triple; got {result!r}'
		raise ValueError(msg)
	return result  # type: ignore[return-value]
