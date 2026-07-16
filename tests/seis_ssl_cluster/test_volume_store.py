from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.data import NpyMemmapVolumeStore, inspect_npy_volume, volume_store

if TYPE_CHECKING:
	from pathlib import Path


def _write_volume(path: Path) -> np.ndarray:
	array = np.arange(10 * 12 * 14, dtype=np.float32).reshape((10, 12, 14))
	np.save(path, array)
	return array


def _mapping_is_closed(array: np.ndarray) -> bool:
	mapping = getattr(array, '_mmap', None)
	return bool(mapping is not None and mapping.closed)


def _close_mapping(array: np.ndarray) -> None:
	mapping = getattr(array, '_mmap', None)
	assert mapping is not None
	mapping.close()


def _reference_padded_crop(
	array: np.ndarray,
	start_xyz: tuple[int, int, int],
	size_xyz: tuple[int, int, int],
	pad_value: float,
) -> tuple[np.ndarray, np.ndarray]:
	crop = np.full(size_xyz, pad_value, dtype=array.dtype)
	valid = np.zeros(size_xyz, dtype=bool)
	stop_xyz = tuple(
		start + size for start, size in zip(start_xyz, size_xyz, strict=True)
	)
	source_start = tuple(max(start, 0) for start in start_xyz)
	source_stop = tuple(
		min(stop, shape)
		for stop, shape in zip(stop_xyz, array.shape, strict=True)
	)
	if all(
		stop > start
		for start, stop in zip(source_start, source_stop, strict=True)
	):
		destination_start = tuple(
			source - request
			for source, request in zip(source_start, start_xyz, strict=True)
		)
		destination_stop = tuple(
			destination + stop - start
			for destination, start, stop in zip(
				destination_start,
				source_start,
				source_stop,
				strict=True,
			)
		)
		source_slices = tuple(
			slice(start, stop)
			for start, stop in zip(source_start, source_stop, strict=True)
		)
		destination_slices = tuple(
			slice(start, stop)
			for start, stop in zip(destination_start, destination_stop, strict=True)
		)
		crop[destination_slices] = array[source_slices]
		valid[destination_slices] = True
	return crop, valid


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
	assert isinstance(crop, np.memmap)
	assert not crop.flags.writeable


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


def test_in_bounds_padded_read_returns_read_only_memmap_view(tmp_path: Path) -> None:
	path = tmp_path / 'volume.npy'
	array = _write_volume(path)
	store = NpyMemmapVolumeStore()

	crop, valid_mask = store.read_crop_with_padding(path, (2, 3, 4), (4, 5, 6))

	assert isinstance(crop, np.memmap)
	assert np.shares_memory(crop, store.open(path))
	assert not crop.flags.writeable
	assert valid_mask.all()
	with pytest.raises(ValueError, match='read-only'):
		crop[0, 0, 0] = -1.0
	np.testing.assert_array_equal(crop, array[2:6, 3:8, 4:10])


@pytest.mark.parametrize(
	('start_xyz', 'size_xyz'),
	[
		((-2, 2, 3), (4, 4, 5)),
		((8, 10, 12), (4, 4, 5)),
		((-2, 10, -3), (5, 4, 6)),
		((20, 20, 20), (3, 4, 5)),
	],
)
def test_padded_reads_match_copying_reference_without_modifying_source(
	tmp_path: Path,
	start_xyz: tuple[int, int, int],
	size_xyz: tuple[int, int, int],
) -> None:
	path = tmp_path / 'volume.npy'
	array = _write_volume(path)
	original = array.copy()
	store = NpyMemmapVolumeStore()
	expected_crop, expected_valid = _reference_padded_crop(
		array,
		start_xyz,
		size_xyz,
		-7.0,
	)

	crop, valid_mask = store.read_crop_with_padding(
		path,
		start_xyz,
		size_xyz,
		pad_value=-7.0,
	)

	assert not np.shares_memory(crop, store.open(path))
	np.testing.assert_array_equal(crop, expected_crop)
	np.testing.assert_array_equal(valid_mask, expected_valid)
	np.testing.assert_array_equal(np.load(path), original)


def test_repeated_access_loads_once_and_close_reopens(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	path = tmp_path / 'volume.npy'
	_write_volume(path)
	load_calls = 0
	original_load = np.load

	def counting_load(*args: object, **kwargs: object) -> np.ndarray:
		nonlocal load_calls
		load_calls += 1
		return original_load(*args, **kwargs)

	monkeypatch.setattr(volume_store.np, 'load', counting_load)
	store = NpyMemmapVolumeStore()
	first = store.open(path)
	assert store.open(path) is first
	store.read_crop(path, (0, 0, 0), (2, 2, 2))
	store.read_crop_with_padding(path, (0, 0, 0), (2, 2, 2))
	assert load_calls == 1

	store.close()
	assert _mapping_is_closed(first)
	store.open(path)
	assert load_calls == 2
	store.close()


def test_lru_evicts_least_recently_used_volume(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	paths = tuple(tmp_path / f'volume-{index}.npy' for index in range(3))
	for index, path in enumerate(paths):
		np.save(path, np.full((2, 2, 2), index, dtype=np.float32))
	load_calls = 0
	original_load = np.load

	def counting_load(*args: object, **kwargs: object) -> np.ndarray:
		nonlocal load_calls
		load_calls += 1
		return original_load(*args, **kwargs)

	monkeypatch.setattr(volume_store.np, 'load', counting_load)
	store = NpyMemmapVolumeStore(max_open_volumes=2)
	first = store.open(paths[0])
	first_crop = store.read_crop(paths[0], (0, 0, 0), (1, 1, 1))
	store.open(paths[1])
	store.open(paths[0])
	store.open(paths[2])
	assert not _mapping_is_closed(first)
	store.open(paths[1])

	assert load_calls == 4
	assert not _mapping_is_closed(first)
	np.testing.assert_array_equal(first, 0.0)
	np.testing.assert_array_equal(first_crop, 0.0)
	store.close()
	_close_mapping(first)


def test_zero_capacity_disables_cache(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	path = tmp_path / 'volume.npy'
	_write_volume(path)
	load_calls = 0
	original_load = np.load

	def counting_load(*args: object, **kwargs: object) -> np.ndarray:
		nonlocal load_calls
		load_calls += 1
		return original_load(*args, **kwargs)

	monkeypatch.setattr(volume_store.np, 'load', counting_load)
	store = NpyMemmapVolumeStore(max_open_volumes=0)
	first = store.open(path)
	second = store.open(path)

	assert load_calls == 2
	assert first is not second
	_close_mapping(first)
	_close_mapping(second)


def test_file_replacement_invalidates_cached_mapping(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	path = tmp_path / 'volume.npy'
	np.save(path, np.zeros((2, 2, 2), dtype=np.float32))
	load_calls = 0
	original_load = np.load

	def counting_load(*args: object, **kwargs: object) -> np.ndarray:
		nonlocal load_calls
		load_calls += 1
		return original_load(*args, **kwargs)

	monkeypatch.setattr(volume_store.np, 'load', counting_load)
	store = NpyMemmapVolumeStore()
	old = store.open(path)
	replacement = tmp_path / 'replacement.npy'
	np.save(replacement, np.ones((2, 2, 2), dtype=np.float32))
	replacement.replace(path)
	new = store.open(path)

	assert load_calls == 2
	assert not _mapping_is_closed(old)
	np.testing.assert_array_equal(old, 0.0)
	np.testing.assert_array_equal(new, 1.0)
	store.close()
	_close_mapping(old)


def test_pid_change_and_pickle_reopen_without_inherited_mapping(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	path = tmp_path / 'volume.npy'
	_write_volume(path)
	load_calls = 0
	original_load = np.load

	def counting_load(*args: object, **kwargs: object) -> np.ndarray:
		nonlocal load_calls
		load_calls += 1
		return original_load(*args, **kwargs)

	monkeypatch.setattr(volume_store.np, 'load', counting_load)
	store = NpyMemmapVolumeStore()
	inherited = store.open(path)
	child_pid = os.getpid() + 1
	monkeypatch.setattr(volume_store.os, 'getpid', lambda: child_pid)
	child_mapping = store.open(path)
	assert load_calls == 2
	assert _mapping_is_closed(inherited)

	restored = pickle.loads(pickle.dumps(store))  # noqa: S301
	restored.open(path)
	assert load_calls == 3
	assert not _mapping_is_closed(child_mapping)
	store.close()
	restored.close()


def test_context_manager_closes_mapping(tmp_path: Path) -> None:
	path = tmp_path / 'volume.npy'
	_write_volume(path)
	with NpyMemmapVolumeStore() as store:
		array = store.open(path)
		assert not _mapping_is_closed(array)
	assert _mapping_is_closed(array)


def test_convenience_crop_api_reuses_process_local_store(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	path = tmp_path / 'volume.npy'
	_write_volume(path)
	load_calls = 0
	original_load = np.load

	def counting_load(*args: object, **kwargs: object) -> np.ndarray:
		nonlocal load_calls
		load_calls += 1
		return original_load(*args, **kwargs)

	monkeypatch.setattr(volume_store.np, 'load', counting_load)
	volume_store.read_crop(path, (0, 0, 0), (2, 2, 2))
	volume_store.read_crop(path, (1, 1, 1), (2, 2, 2))

	assert load_calls == 1


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
