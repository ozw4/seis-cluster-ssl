"""Memmap-backed `.npy` volume access in `[x, y, z]` grid order."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from types import TracebackType

	from typing_extensions import Self

_DEFAULT_MAX_OPEN_VOLUMES = 8
_CacheKey = tuple[Path, int, int, int, int]


@dataclass(frozen=True)
class NpyVolumeInfo:
	"""Metadata for a 3D `.npy` volume stored in `[x, y, z]` order."""

	path: Path
	shape_xyz: tuple[int, int, int]
	dtype: str
	ndim: int


def inspect_npy_volume(path: str | Path) -> NpyVolumeInfo:
	"""Inspect a numeric 3D `.npy` volume using NumPy's memmap loader."""
	volume_path = _validate_npy_path(path)
	array = _load_npy_memmap(volume_path)
	_validate_volume_array(array, volume_path)
	return NpyVolumeInfo(
		path=volume_path,
		shape_xyz=tuple(int(axis) for axis in array.shape),
		dtype=str(array.dtype),
		ndim=int(array.ndim),
	)


def open(path: str | Path) -> np.ndarray:  # noqa: A001
	"""Open and validate a numeric 3D `.npy` volume."""
	return _CONVENIENCE_STORE.open(path)


def read_crop(
	path: str | Path,
	start_xyz: tuple[int, int, int],
	size_xyz: tuple[int, int, int],
) -> np.ndarray:
	"""Read an in-bounds crop from a volume."""
	return _CONVENIENCE_STORE.read_crop(path, start_xyz, size_xyz)


def read_crop_with_padding(
	path: str | Path,
	start_xyz: tuple[int, int, int],
	size_xyz: tuple[int, int, int],
	pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
	"""Read a crop with out-of-bounds padding and a boolean valid mask."""
	return _CONVENIENCE_STORE.read_crop_with_padding(
		path,
		start_xyz,
		size_xyz,
		pad_value,
	)


class NpyMemmapVolumeStore:
	"""Read 3D `.npy` volumes and crops in `[x, y, z]` order."""

	def __init__(self, max_open_volumes: int = _DEFAULT_MAX_OPEN_VOLUMES) -> None:
		"""Initialize a store with a bounded process-local memmap cache."""
		if isinstance(max_open_volumes, bool) or not isinstance(max_open_volumes, int):
			msg = 'max_open_volumes must be a nonnegative integer'
			raise TypeError(msg)
		if max_open_volumes < 0:
			msg = 'max_open_volumes must be a nonnegative integer'
			raise ValueError(msg)
		self.max_open_volumes = max_open_volumes
		self._initialize_process_state()

	def __enter__(self) -> Self:
		"""Return this store for context-manager use."""
		self._ensure_current_process()
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_value: BaseException | None,
		traceback: TracebackType | None,
	) -> None:
		"""Close cached mappings when leaving a context."""
		self.close()

	def __getstate__(self) -> dict[str, int]:
		"""Exclude process-owned mappings and locks from serialized state."""
		return {'max_open_volumes': self.max_open_volumes}

	def __setstate__(self, state: dict[str, int]) -> None:
		"""Create empty process-local state after deserialization."""
		self.max_open_volumes = state['max_open_volumes']
		self._initialize_process_state()

	def close(self) -> None:
		"""Close all memmaps cached by this store."""
		self._ensure_current_process()
		with self._cache_lock:
			arrays = tuple(self._cache.values())
			self._cache.clear()
		for array in arrays:
			_close_memmap(array)

	def open(self, path: str | Path) -> np.ndarray:
		"""Open and validate a numeric 3D `.npy` volume."""
		volume_path = _validate_npy_path(path)
		resolved_path = volume_path.resolve(strict=True)
		self._ensure_current_process()
		if self.max_open_volumes == 0:
			return _open_validated_memmap(resolved_path)

		while True:
			key = _cache_key(resolved_path)
			with self._cache_lock:
				cached = self._cache.get(key)
				if cached is not None:
					self._cache.move_to_end(key)
					return cached

			array = _open_validated_memmap(resolved_path)
			try:
				current_key = _cache_key(resolved_path)
			except OSError:
				_close_memmap(array)
				raise
			if current_key == key:
				break
			_close_memmap(array)

		with self._cache_lock:
			cached = self._cache.get(key)
			if cached is not None:
				self._cache.move_to_end(key)
			else:
				stale_keys = tuple(
					cached_key
					for cached_key in self._cache
					if cached_key[0] == resolved_path
				)
				for stale_key in stale_keys:
					del self._cache[stale_key]
				self._cache[key] = array
				while len(self._cache) > self.max_open_volumes:
					self._cache.popitem(last=False)
		if cached is not None:
			_close_memmap(array)
			return cached
		return array

	def _initialize_process_state(self) -> None:
		self._pid = os.getpid()
		self._cache: OrderedDict[_CacheKey, np.ndarray] = OrderedDict()
		self._cache_lock = threading.Lock()

	def _ensure_current_process(self) -> None:
		pid = os.getpid()
		if pid == self._pid:
			return
		inherited_arrays = tuple(self._cache.values())
		self._cache = OrderedDict()
		self._cache_lock = threading.Lock()
		self._pid = pid
		for array in inherited_arrays:
			_close_memmap(array)

	def read_crop(
		self,
		path: str | Path,
		start_xyz: tuple[int, int, int],
		size_xyz: tuple[int, int, int],
	) -> np.ndarray:
		"""Read an in-bounds crop from a volume."""
		array = self.open(path)
		start = _validate_xyz_tuple(start_xyz, 'start_xyz')
		size = _validate_size_xyz(size_xyz)
		stop = tuple(
			start_axis + size_axis
			for start_axis, size_axis in zip(start, size, strict=True)
		)
		_validate_in_bounds(start, stop, array.shape, Path(path))
		return array[
			start[0] : stop[0],
			start[1] : stop[1],
			start[2] : stop[2],
		]

	def read_crop_with_padding(
		self,
		path: str | Path,
		start_xyz: tuple[int, int, int],
		size_xyz: tuple[int, int, int],
		pad_value: float = 0.0,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Read a crop, padding out-of-bounds regions and returning a valid mask."""
		array = self.open(path)
		start = _validate_xyz_tuple(start_xyz, 'start_xyz')
		size = _validate_size_xyz(size_xyz)
		stop = tuple(
			start_axis + size_axis
			for start_axis, size_axis in zip(start, size, strict=True)
		)
		if all(axis_start >= 0 for axis_start in start) and all(
			axis_stop <= axis_size
			for axis_stop, axis_size in zip(stop, array.shape, strict=True)
		):
			slices = tuple(
				slice(axis_start, axis_stop)
				for axis_start, axis_stop in zip(start, stop, strict=True)
			)
			return array[slices], np.ones(size, dtype=bool)

		source_start = tuple(max(axis_start, 0) for axis_start in start)
		source_stop = tuple(
			min(axis_stop, axis_size)
			for axis_stop, axis_size in zip(stop, array.shape, strict=True)
		)
		crop = np.full(size, pad_value, dtype=array.dtype)
		valid_mask = np.zeros(size, dtype=bool)

		if all(
			stop_axis > start_axis
			for start_axis, stop_axis in zip(source_start, source_stop, strict=True)
		):
			dest_start = tuple(
				source_axis_start - request_axis_start
				for source_axis_start, request_axis_start in zip(
					source_start,
					start,
					strict=True,
				)
			)
			dest_stop = tuple(
				dest_axis_start + source_axis_stop - source_axis_start
				for dest_axis_start, source_axis_start, source_axis_stop in zip(
					dest_start,
					source_start,
					source_stop,
					strict=True,
				)
			)
			source_slices = tuple(
				slice(axis_start, axis_stop)
				for axis_start, axis_stop in zip(source_start, source_stop, strict=True)
			)
			dest_slices = tuple(
				slice(axis_start, axis_stop)
				for axis_start, axis_stop in zip(dest_start, dest_stop, strict=True)
			)
			crop[dest_slices] = array[source_slices]
			valid_mask[dest_slices] = True

		return crop, valid_mask


_CONVENIENCE_STORE = NpyMemmapVolumeStore()


def _validate_npy_path(path: str | Path) -> Path:
	volume_path = Path(path)
	if volume_path.suffix != '.npy':
		msg = f'volume path must have .npy suffix: {volume_path}'
		raise ValueError(msg)
	if not volume_path.is_file():
		msg = f'volume file does not exist: {volume_path}'
		raise FileNotFoundError(msg)
	return volume_path


def _load_npy_memmap(path: Path) -> np.ndarray:
	try:
		return np.load(path, mmap_mode='r')
	except ValueError as exc:
		if 'Python objects' in str(exc) or 'allow_pickle' in str(exc):
			msg = f'volume must not use object dtype: {path}'
			raise TypeError(msg) from exc
		msg = f'failed to load .npy volume {path}: {exc}'
		raise ValueError(msg) from exc


def _open_validated_memmap(path: Path) -> np.ndarray:
	array = _load_npy_memmap(path)
	try:
		_validate_volume_array(array, path)
	except Exception:
		_close_memmap(array)
		raise
	return array


def _cache_key(path: Path) -> _CacheKey:
	stat = path.stat()
	return (path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _close_memmap(array: np.ndarray) -> None:
	mapping = getattr(array, '_mmap', None)
	if mapping is not None:
		mapping.close()


def _validate_volume_array(array: np.ndarray, path: Path) -> None:
	if array.ndim != 3:
		msg = f'volume must be a 3D [x, y, z] array; got ndim={array.ndim}: {path}'
		raise ValueError(msg)
	if array.dtype.hasobject:
		msg = f'volume must not use object dtype: {path}'
		raise TypeError(msg)
	if not np.issubdtype(array.dtype, np.number):
		msg = f'volume dtype must be numeric; got {array.dtype}: {path}'
		raise TypeError(msg)


def _validate_xyz_tuple(value: tuple[int, int, int], name: str) -> tuple[int, int, int]:
	if len(value) != 3 or not all(isinstance(axis, int) for axis in value):
		msg = f'{name} must be a length-3 integer tuple; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_size_xyz(size_xyz: tuple[int, int, int]) -> tuple[int, int, int]:
	size = _validate_xyz_tuple(size_xyz, 'size_xyz')
	if any(axis <= 0 for axis in size):
		msg = f'size_xyz values must be positive; got {size_xyz!r}'
		raise ValueError(msg)
	return size


def _validate_in_bounds(
	start: tuple[int, int, int],
	stop: tuple[int, int, int],
	shape: tuple[int, ...],
	path: Path,
) -> None:
	if any(axis_start < 0 for axis_start in start) or any(
		axis_stop > axis_size
		for axis_stop, axis_size in zip(stop, shape, strict=True)
	):
		msg = (
			f'crop is out of bounds for volume {path}: '
			f'start_xyz={start!r}, stop_xyz={stop!r}, shape_xyz={tuple(shape)!r}'
		)
		raise ValueError(msg)


__all__ = [
	'NpyMemmapVolumeStore',
	'NpyVolumeInfo',
	'inspect_npy_volume',
	'open',
	'read_crop',
	'read_crop_with_padding',
]
