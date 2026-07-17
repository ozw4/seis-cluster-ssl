"""Feature loading helpers for embedding-only clustering."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from collections.abc import Iterator, Sequence
	from types import TracebackType

	from typing_extensions import Self


_DEFAULT_MAX_OPEN_ARRAYS = 16
_CacheKey = tuple[Path, int, int, int, int]


class EmbeddingMemmapCache:
	"""Bounded process-local cache for clustering input memmaps."""

	def __init__(self, max_open_arrays: int = _DEFAULT_MAX_OPEN_ARRAYS) -> None:
		"""Initialize a bounded process-local cache."""
		if isinstance(max_open_arrays, bool) or not isinstance(max_open_arrays, int):
			msg = 'max_open_arrays must be a nonnegative integer'
			raise TypeError(msg)
		if max_open_arrays < 0:
			msg = 'max_open_arrays must be a nonnegative integer'
			raise ValueError(msg)
		self.max_open_arrays = max_open_arrays
		self._initialize_process_state()

	def __enter__(self) -> Self:
		"""Return this cache for context-manager use."""
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
		return {'max_open_arrays': self.max_open_arrays}

	def __setstate__(self, state: dict[str, int]) -> None:
		"""Create empty process-local state after deserialization."""
		self.max_open_arrays = state['max_open_arrays']
		self._initialize_process_state()

	def close(self) -> None:
		"""Release cached mappings without invalidating arrays held by callers."""
		self._ensure_current_process()
		with self._cache_lock:
			# Borrowers keep the memmap alive after cache ownership is released.
			self._cache.clear()

	def open(self, path: str | Path) -> np.ndarray:
		"""Open one `.npy` file, reusing an unchanged process-local mapping."""
		resolved_path = Path(path).resolve(strict=True)
		self._ensure_current_process()
		if self.max_open_arrays == 0:
			return _load_memmap(resolved_path)

		while True:
			key = _cache_key(resolved_path)
			with self._cache_lock:
				cached = self._cache.get(key)
				if cached is not None:
					self._cache.move_to_end(key)
					return cached
			array = _load_memmap(resolved_path)
			try:
				current_key = _cache_key(resolved_path)
			except OSError:
				_close_memmap(array)
				raise
			if current_key == key:
				break
			_close_memmap(array)

		cached = self._store(key, array)
		if cached is not None:
			_close_memmap(array)
			return cached
		return array

	def _store(
		self,
		key: _CacheKey,
		array: np.ndarray,
	) -> np.ndarray | None:
		with self._cache_lock:
			cached = self._cache.get(key)
			if cached is not None:
				self._cache.move_to_end(key)
				return cached
			stale_keys = tuple(
				cached_key for cached_key in self._cache if cached_key[0] == key[0]
			)
			for stale_key in stale_keys:
				self._cache.pop(stale_key)
			self._cache[key] = array
			while len(self._cache) > self.max_open_arrays:
				# Never close a mapping that may still be held by a caller.
				self._cache.popitem(last=False)
		return None

	def _initialize_process_state(self) -> None:
		self._pid = os.getpid()
		self._cache: OrderedDict[_CacheKey, np.ndarray] = OrderedDict()
		self._cache_lock = threading.Lock()

	def _ensure_current_process(self) -> None:
		pid = os.getpid()
		if pid == self._pid:
			return
		self._cache = OrderedDict()
		self._cache_lock = threading.Lock()
		self._pid = pid


@dataclass(frozen=True)
class EmbeddingInput:
	"""Input artifact paths for one survey's embedding grid."""

	survey_id: str
	embeddings_path: Path
	valid_tokens_path: Path
	metadata_path: Path


@dataclass(frozen=True)
class FeatureBatch:
	"""A batch of valid token features from one survey."""

	survey_id: str
	token_indices: np.ndarray
	features: np.ndarray


COMPATIBILITY_METADATA_FIELDS = (
	'checkpoint_sha256',
	'model_geometry',
	'patch_size',
	'window_size',
	'overlap',
	'min_token_valid_fraction',
	'zero_mask',
)

_MEMMAP_CACHE = EmbeddingMemmapCache()


def discover_embedding_inputs(input_dir: str | Path) -> list[EmbeddingInput]:
	"""Discover per-survey embedding artifacts in deterministic survey order."""
	root = Path(input_dir)
	if not root.is_dir():
		msg = f'embeddings.input_dir must be an existing directory: {root}'
		raise FileNotFoundError(msg)

	inputs: list[EmbeddingInput] = []
	for embeddings_path in sorted(root.glob('*.embeddings.npy')):
		survey_id = embeddings_path.name.removesuffix('.embeddings.npy')
		item = EmbeddingInput(
			survey_id=survey_id,
			embeddings_path=embeddings_path,
			valid_tokens_path=root / f'{survey_id}.valid_tokens.npy',
			metadata_path=root / f'{survey_id}.embedding_metadata.json',
		)
		_validate_embedding_input(item)
		inputs.append(item)
	if not inputs:
		msg = f'no embedding inputs found in {root}'
		raise ValueError(msg)
	return inputs


def count_valid_tokens(embedding_input: EmbeddingInput) -> int:
	"""Return the number of valid tokens in one survey."""
	valid = load_valid_tokens(embedding_input)
	return int(np.count_nonzero(valid))


def embedding_dim(embedding_input: EmbeddingInput) -> int:
	"""Return the embedding channel dimension for one survey."""
	embeddings = open_embedding_array(embedding_input)
	return int(embeddings.shape[-1])


def load_valid_tokens(
	embedding_input: EmbeddingInput,
	*,
	cache: EmbeddingMemmapCache | None = None,
) -> np.ndarray:
	"""Load a survey valid-token mask as a memory-mapped array."""
	valid = (cache or _MEMMAP_CACHE).open(embedding_input.valid_tokens_path)
	if valid.dtype != np.bool_:
		msg = (
			f'valid_tokens dtype must be bool for {embedding_input.survey_id}; '
			f'got {valid.dtype}'
		)
		raise TypeError(msg)
	if valid.ndim != 3:
		msg = (
			f'valid_tokens must be 3D for {embedding_input.survey_id}; '
			f'got shape={valid.shape!r}'
		)
		raise ValueError(msg)
	return valid


def open_embedding_array(
	embedding_input: EmbeddingInput,
	*,
	cache: EmbeddingMemmapCache | None = None,
) -> np.ndarray:
	"""Open a survey embedding grid as a memory-mapped array."""
	selected_cache = cache or _MEMMAP_CACHE
	valid = load_valid_tokens(embedding_input, cache=selected_cache)
	valid_shape = valid.shape
	embeddings = selected_cache.open(embedding_input.embeddings_path)
	if embeddings.ndim != 4:
		msg = (
			f'embeddings must be 4D for {embedding_input.survey_id}; '
			f'got shape={embeddings.shape!r}'
		)
		raise ValueError(msg)
	if embeddings.dtype.kind not in {'f', 'i', 'u'}:
		msg = (
			f'embeddings dtype must be numeric for {embedding_input.survey_id}; '
			f'got {embeddings.dtype}'
		)
		raise TypeError(msg)
	if embeddings.shape[:3] != valid_shape:
		msg = (
			f'embeddings token grid must match valid_tokens for '
			f'{embedding_input.survey_id}; got {embeddings.shape[:3]!r} and '
			f'{valid_shape!r}'
		)
		raise ValueError(msg)
	return embeddings


def valid_flat_indices(embedding_input: EmbeddingInput) -> np.ndarray:
	"""Return flattened token indices whose embedding token is valid."""
	valid = load_valid_tokens(embedding_input)
	return np.flatnonzero(valid.reshape(-1))


def extract_token_features(
	embedding_input: EmbeddingInput,
	token_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
	"""Read selected flattened token embeddings as a float32 feature matrix."""
	embeddings = open_embedding_array(embedding_input)
	flat = embeddings.reshape((-1, embeddings.shape[-1]))
	indices = np.asarray(token_indices, dtype=np.int64)
	if indices.ndim != 1:
		msg = f'token_indices must be 1D; got shape={indices.shape!r}'
		raise ValueError(msg)
	if indices.size == 0:
		return np.empty((0, embeddings.shape[-1]), dtype=np.float32)
	features = np.asarray(flat[indices], dtype=np.float32)
	_validate_finite_features(features, embedding_input.survey_id)
	return features


def iter_valid_feature_batches(
	embedding_input: EmbeddingInput,
	*,
	batch_size: int,
) -> Iterator[FeatureBatch]:
	"""Yield valid embedding features for one survey in flattened token order."""
	if batch_size <= 0:
		msg = f'batch_size must be positive; got {batch_size!r}'
		raise ValueError(msg)
	indices = valid_flat_indices(embedding_input)
	for start in range(0, indices.size, batch_size):
		batch_indices = indices[start : start + batch_size]
		yield FeatureBatch(
			survey_id=embedding_input.survey_id,
			token_indices=batch_indices,
			features=extract_token_features(embedding_input, batch_indices),
		)


def file_sha256(path: str | Path) -> str:
	"""Return the SHA-256 hex digest for a file."""
	digest = hashlib.sha256()
	with Path(path).open('rb') as file_obj:
		for block in iter(lambda: file_obj.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def close_embedding_memmap_cache() -> None:
	"""Close process-local mappings held by clustering convenience helpers."""
	_MEMMAP_CACHE.close()


def _load_memmap(path: Path) -> np.ndarray:
	return np.load(path, mmap_mode='r', allow_pickle=False)


def _cache_key(path: Path) -> _CacheKey:
	stat = path.stat()
	return (path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _close_memmap(array: np.ndarray) -> None:
	mapping = getattr(array, '_mmap', None)
	if mapping is not None:
		mapping.close()


def embedding_input_metadata(embedding_input: EmbeddingInput) -> dict[str, object]:
	"""Return deterministic metadata describing one embedding input."""
	return {
		'survey_id': embedding_input.survey_id,
		'embeddings_path': str(embedding_input.embeddings_path),
		'valid_tokens_path': str(embedding_input.valid_tokens_path),
		'metadata_path': str(embedding_input.metadata_path),
		'metadata_sha256': file_sha256(embedding_input.metadata_path),
	}


def load_embedding_metadata(embedding_input: EmbeddingInput) -> dict[str, object]:
	"""Load one survey's extraction metadata JSON."""
	try:
		payload = json.loads(embedding_input.metadata_path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		msg = (
			f'embedding metadata must be valid JSON for '
			f'{embedding_input.survey_id}: {embedding_input.metadata_path}'
		)
		raise ValueError(msg) from exc
	if not isinstance(payload, dict):
		msg = (
			f'embedding metadata must be a JSON object for '
			f'{embedding_input.survey_id}: {embedding_input.metadata_path}'
		)
		raise TypeError(msg)
	return payload


def embedding_compatibility_signature(
	embedding_input: EmbeddingInput,
) -> dict[str, object]:
	"""Return representation-defining metadata for clustering compatibility."""
	metadata = load_embedding_metadata(embedding_input)
	missing = [
		field
		for field in COMPATIBILITY_METADATA_FIELDS
		if field not in metadata
	]
	if missing:
		msg = (
			f'embedding metadata missing compatibility fields for '
			f'{embedding_input.survey_id}: {missing!r}'
		)
		raise ValueError(msg)
	return {
		**{
			field: metadata[field]
			for field in COMPATIBILITY_METADATA_FIELDS
		},
		'embedding_dim': embedding_dim(embedding_input),
	}


def validate_compatible_embedding_inputs(
	embedding_inputs: Sequence[EmbeddingInput],
) -> dict[str, object]:
	"""Require all survey embeddings to share one representation signature."""
	if not embedding_inputs:
		msg = 'at least one embedding input is required'
		raise ValueError(msg)
	signatures = [
		(item, embedding_compatibility_signature(item))
		for item in embedding_inputs
	]
	baseline_input, baseline = signatures[0]
	for item, signature in signatures[1:]:
		if signature != baseline:
			differing = [
				field
				for field, value in baseline.items()
				if signature.get(field) != value
			]
			msg = (
				'incompatible embedding artifacts for surveys '
				f'{baseline_input.survey_id!r} and {item.survey_id!r}; '
				f'differing fields: {", ".join(differing)}'
			)
			raise ValueError(msg)
	return baseline


def validate_finite_feature_batch(features: np.ndarray, survey_id: str) -> None:
	"""Raise when a feature batch contains NaN or infinity values."""
	_validate_finite_features(features, survey_id)


def _validate_embedding_input(embedding_input: EmbeddingInput) -> None:
	missing = [
		path
		for path in (
			embedding_input.embeddings_path,
			embedding_input.valid_tokens_path,
			embedding_input.metadata_path,
		)
		if not path.is_file()
	]
	if missing:
		msg = (
			f'missing embedding artifacts for {embedding_input.survey_id}: '
			f'{[str(path) for path in missing]!r}'
		)
		raise FileNotFoundError(msg)


def _validate_finite_features(features: np.ndarray, survey_id: str) -> None:
	if not np.isfinite(features).all():
		msg = f'non-finite embedding features found for survey {survey_id}'
		raise ValueError(msg)


__all__ = [
	'COMPATIBILITY_METADATA_FIELDS',
	'EmbeddingInput',
	'EmbeddingMemmapCache',
	'FeatureBatch',
	'close_embedding_memmap_cache',
	'count_valid_tokens',
	'discover_embedding_inputs',
	'embedding_compatibility_signature',
	'embedding_dim',
	'embedding_input_metadata',
	'extract_token_features',
	'file_sha256',
	'iter_valid_feature_batches',
	'load_embedding_metadata',
	'load_valid_tokens',
	'open_embedding_array',
	'valid_flat_indices',
	'validate_compatible_embedding_inputs',
	'validate_finite_feature_batch',
]
