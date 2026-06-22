from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.data import NpyMemmapVolumeStore, inspect_npy_volume

if TYPE_CHECKING:
	from pathlib import Path


def _write_volume(path: Path) -> np.ndarray:
	array = np.arange(10 * 12 * 14, dtype=np.float32).reshape((10, 12, 14))
	np.save(path, array)
	return array


def test_inspect_npy_volume_reports_metadata(tmp_path: Path) -> None:
	path = tmp_path / 'volume.npy'
	array = _write_volume(path)

	info = inspect_npy_volume(path)

	assert info.path == path
	assert info.shape_xyz == array.shape
	assert info.dtype == 'float32'
	assert info.ndim == 3


def test_read_crop_in_bounds_matches_numpy_slice(tmp_path: Path) -> None:
	path = tmp_path / 'volume.npy'
	array = _write_volume(path)
	store = NpyMemmapVolumeStore()

	crop = store.read_crop(path, start_xyz=(2, 3, 4), size_xyz=(4, 5, 6))

	np.testing.assert_array_equal(crop, array[2:6, 3:8, 4:10])


def test_read_crop_with_padding_returns_crop_and_valid_mask(tmp_path: Path) -> None:
	path = tmp_path / 'volume.npy'
	array = _write_volume(path)
	store = NpyMemmapVolumeStore()

	crop, valid_mask = store.read_crop_with_padding(
		path,
		start_xyz=(-2, -1, 3),
		size_xyz=(4, 4, 5),
		pad_value=-1.0,
	)

	assert crop.shape == (4, 4, 5)
	assert valid_mask.shape == (4, 4, 5)
	assert valid_mask.dtype == np.bool_
	np.testing.assert_array_equal(crop[:2, :, :], -1.0)
	np.testing.assert_array_equal(crop[:, :1, :], -1.0)
	np.testing.assert_array_equal(crop[2:4, 1:4, :], array[0:2, 0:3, 3:8])
	assert valid_mask[2:4, 1:4, :].all()


def test_inspect_npy_volume_rejects_invalid_sources(tmp_path: Path) -> None:
	np.save(tmp_path / 'volume.npy', np.zeros((4, 5), dtype=np.float32))
	with pytest.raises(ValueError, match='3D'):
		inspect_npy_volume(tmp_path / 'volume.npy')

	np.save(tmp_path / 'objects.npy', np.empty((2, 2, 2), dtype=object))
	with pytest.raises(TypeError, match='object dtype'):
		inspect_npy_volume(tmp_path / 'objects.npy')

	text_path = tmp_path / 'volume.txt'
	text_path.write_text('not a volume', encoding='utf-8')
	with pytest.raises(ValueError, match=r'\.npy'):
		inspect_npy_volume(text_path)
