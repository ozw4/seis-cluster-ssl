from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from seis_ssl_cluster.f3.lithology.voxel_split import (
	TRAIN_VOXEL_SPLIT,
	VALIDATION_VOXEL_SPLIT,
	build_f3_voxel_supervision_split,
)
from seis_ssl_cluster.f3.splits import F3LineGeometry, F3SliceSplitRecord

SHAPE = (3, 4, 5)
GEOMETRY = F3LineGeometry(
	shape_xyz=SHAPE,
	inline_min=100,
	inline_max=102,
	crossline_min=200,
	crossline_max=203,
)


def _record(split: str, slice_type: str, index: int) -> F3SliceSplitRecord:
	return F3SliceSplitRecord(
		relative_path=f'{split}/{slice_type}_{index}.png',
		split=split,
		slice_type=slice_type,
		slice_index=index,
	)


def _build(
	records: tuple[F3SliceSplitRecord, ...],
	*,
	labels: np.ndarray | None = None,
	class_ids: tuple[int, ...] = (1, 2),
	valid_tokens: np.ndarray | None = None,
	ignore_z_border_samples: int = 0,
):
	return build_f3_voxel_supervision_split(
		records,
		geometry=GEOMETRY,
		label_volume=(
			np.ones(SHAPE, dtype=np.int16) if labels is None else labels
		),
		class_ids=class_ids,
		valid_tokens=(
			np.ones((2, 2, 2), dtype=np.bool_)
			if valid_tokens is None
			else valid_tokens
		),
		patch_size_xyz=(2, 2, 3),
		ignore_z_border_samples=ignore_z_border_samples,
	)


def test_split_resolves_lines_deduplicates_and_applies_validation_precedence() -> None:
	result = _build(
		(
			_record('train', 'inline', 100),
			_record('train', 'crossline', 201),
			_record('validation', 'inline', 102),
		)
	)
	grid = result.split_grid
	summary = result.summary

	assert grid.dtype == np.uint8
	assert np.all(grid[0, [0, 2, 3], :] == TRAIN_VOXEL_SPLIT)
	assert np.all(grid[2, :, :] == VALIDATION_VOXEL_SPLIT)
	assert summary.raw_train_assignments == 35
	assert summary.unique_train_voxels_before_precedence == 30
	assert summary.raw_validation_assignments == 20
	assert summary.unique_validation_voxels == 20
	assert summary.cross_split_overlap_voxels == 5
	assert summary.train_voxels_removed_by_validation_precedence == 5
	assert summary.final_train_voxels == np.count_nonzero(grid == TRAIN_VOXEL_SPLIT)
	assert summary.final_validation_voxels == np.count_nonzero(
		grid == VALIDATION_VOXEL_SPLIT
	)


def test_split_filters_invalid_inputs_without_mutation() -> None:
	labels = np.ones(SHAPE, dtype=np.int16)
	labels[0, 0, 2] = -1
	labels[2, 3, 2] = 99
	valid = np.ones((2, 2, 2), dtype=np.bool_)
	valid[0, 0, 0] = False
	labels_before = labels.copy()
	valid_before = valid.copy()

	result = _build(
		(
			_record('train', 'inline', 100),
			_record('validation', 'inline', 102),
		),
		labels=labels,
		valid_tokens=valid,
		ignore_z_border_samples=1,
	)

	assert result.split_grid[0, 0, 2] == 0
	assert result.split_grid[2, 3, 2] == 0
	assert not result.split_grid[:, :, [0, -1]].any()
	assert result.summary.invalid_label_voxels_on_selected_slices == 2
	assert result.summary.invalid_reference_voxels_on_selected_slices == 6
	assert result.summary.ignored_z_border_voxels == 16
	np.testing.assert_array_equal(labels, labels_before)
	np.testing.assert_array_equal(valid, valid_before)


def test_split_never_supervises_negative_labels_listed_as_class_ids() -> None:
	labels = np.ones(SHAPE, dtype=np.int16)
	labels[0, 0, 2] = -1

	result = _build(
		(
			_record('train', 'inline', 100),
			_record('validation', 'inline', 102),
		),
		labels=labels,
		class_ids=(-1, 1),
	)

	assert result.split_grid[0, 0, 2] == 0
	assert result.summary.invalid_label_voxels_on_selected_slices == 1


@pytest.mark.parametrize(
	('change', 'error'),
	[
		('geometry_shape', 'geometry shape_xyz'),
		('geometry_bounds', 'inline geometry bounds'),
		('label_shape', 'geometry shape_xyz'),
		('token_shape', 'canonical token grid'),
		('token_dtype', 'dtype must be bool'),
	],
)
def test_split_rejects_shape_dtype_and_geometry_mismatches(
	change: str,
	error: str,
) -> None:
	geometry = GEOMETRY
	labels = np.ones(SHAPE, dtype=np.int16)
	valid = np.ones((2, 2, 2), dtype=np.bool_)
	if change == 'geometry_shape':
		geometry = replace(GEOMETRY, shape_xyz=(4, 4, 5))
	elif change == 'geometry_bounds':
		geometry = replace(GEOMETRY, inline_max=103)
	elif change == 'label_shape':
		labels = np.ones((2, 4, 5), dtype=np.int16)
	elif change == 'token_shape':
		valid = np.ones((1, 2, 2), dtype=np.bool_)
	else:
		valid = valid.astype(np.uint8)

	with pytest.raises((TypeError, ValueError), match=error):
		build_f3_voxel_supervision_split(
			(_record('train', 'inline', 100), _record('validation', 'inline', 102)),
			geometry=geometry,
			label_volume=labels,
			class_ids=(1,),
			valid_tokens=valid,
			patch_size_xyz=(2, 2, 3),
			ignore_z_border_samples=0,
		)


def test_split_rejects_empty_supervised_split() -> None:
	with pytest.raises(ValueError, match='validation voxel'):
		_build((_record('train', 'inline', 100),))
