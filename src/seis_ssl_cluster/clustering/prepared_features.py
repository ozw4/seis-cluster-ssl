"""Prepared feature storage for repeated stratigraphic HMM passes."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import joblib
import numpy as np

from seis_ssl_cluster.clustering.features import (
	EmbeddingInput,
	load_valid_tokens,
	open_embedding_array,
)

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator, Mapping
	from pathlib import Path

	from typing_extensions import Self


PREPARED_FEATURE_SCHEMA_VERSION = 1
PREPARED_FEATURE_DTYPE = np.dtype(np.float32)


@dataclass(frozen=True)
class PreparedFeatureCacheSettings:
	"""Storage and lifecycle policy for prepared HMM features."""

	chunk_size_tokens: int = 65536
	reuse: bool = True
	force_rebuild: bool = False
	cleanup: bool = False
	persist: bool = True
	directory: Path | None = None

	def validate(self) -> None:
		"""Validate the cache policy."""
		if isinstance(self.chunk_size_tokens, bool) or not isinstance(
			self.chunk_size_tokens,
			int,
		):
			raise TypeError(
				'prepared_feature_cache.chunk_size_tokens must be an integer'
			)
		if self.chunk_size_tokens <= 0:
			raise ValueError(
				'prepared_feature_cache.chunk_size_tokens must be positive',
			)
		for name in ('reuse', 'force_rebuild', 'cleanup', 'persist'):
			if not isinstance(getattr(self, name), bool):
				raise TypeError(f'prepared_feature_cache.{name} must be a boolean')
		if self.cleanup == self.persist:
			raise ValueError(
				'prepared_feature_cache.cleanup and persist must be complementary',
			)


@dataclass
class PreparedSurveyFeatures:
	"""Descriptor for one survey's prepared features."""

	embedding_input: EmbeddingInput
	_token_shape_xyz: tuple[int, int, int]
	valid_token_count: int
	feature_dim: int
	feature_mode: str
	edge_margin_tokens: tuple[int, int, int]
	fingerprint: str | None = None
	cache_path: Path | None = None
	reused: bool = False

	@property
	def token_shape_xyz(self) -> tuple[int, int, int]:
		"""Return the source token-grid shape."""
		return self._token_shape_xyz

	def features_for_flat_indices(self, flat_indices: np.ndarray) -> np.ndarray:
		"""Return prepared rows for sorted or unsorted valid flattened indices."""
		with self.open() as opened:
			return opened.features_for_flat_indices(flat_indices).copy()

	def trace_features(
		self,
		x_index: int,
		y_index: int,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Return valid z indices and prepared rows for one vertical trace."""
		with self.open() as opened:
			z_indices, rows = opened.trace_features(x_index, y_index)
			return z_indices.copy(), rows.copy()

	@contextmanager
	def open(self) -> Iterator[_OpenedPreparedSurveyFeatures]:
		"""Open at most this survey's two prepared memmaps for bounded access."""
		indices: np.ndarray | None = None
		features: np.ndarray | None = None
		try:
			if self.feature_mode == 'embedding':
				if self.cache_path is None:
					raise RuntimeError('prepared embedding cache path is missing')
				mmap_mode = None if self.valid_token_count == 0 else 'r'
				indices = np.load(
					self.cache_path / 'valid_flat_indices.npy',
					mmap_mode=mmap_mode,
				)
				features = np.load(
					self.cache_path / 'features.npy',
					mmap_mode=mmap_mode,
				)
			yield _OpenedPreparedSurveyFeatures(self, indices, features)
		finally:
			if indices is not None:
				_close_memmap(indices)
			if features is not None:
				_close_memmap(features)

	def close(self) -> None:
		"""Close this descriptor (it owns no persistent open mappings)."""


@dataclass(frozen=True)
class _OpenedPreparedSurveyFeatures:
	"""Prepared feature arrays open for one bounded survey operation."""

	descriptor: PreparedSurveyFeatures
	valid_flat_indices: np.ndarray | None
	features: np.ndarray | None

	def features_for_flat_indices(self, flat_indices: np.ndarray) -> np.ndarray:
		indices = np.asarray(flat_indices, dtype=np.int64)
		if indices.ndim != 1:
			raise ValueError(f'flat_indices must be 1D; got shape {indices.shape!r}')
		if indices.size == 0:
			return np.empty(
				(0, self.descriptor.feature_dim), dtype=PREPARED_FEATURE_DTYPE
			)
		if self.descriptor.feature_mode == 'z_coordinate':
			_validate_direct_indices(self.descriptor, indices)
			z_size = self.descriptor.token_shape_xyz[2]
			z = np.asarray(
				np.unravel_index(indices, self.descriptor.token_shape_xyz)[2]
			)
			return (
				z.astype(PREPARED_FEATURE_DTYPE) / np.float32(max(z_size - 1, 1))
			).reshape(-1, 1)
		if self.valid_flat_indices is None or self.features is None:
			raise RuntimeError('prepared embedding features are not open')
		ordinals = np.searchsorted(self.valid_flat_indices, indices)
		in_range = ordinals < self.valid_flat_indices.size
		if not np.all(in_range):
			raise ValueError(
				'flat_indices contain tokens outside the prepared valid set'
			)
		if not np.array_equal(self.valid_flat_indices[ordinals], indices):
			raise ValueError(
				'flat_indices contain tokens outside the prepared valid set'
			)
		return np.asarray(self.features[ordinals], dtype=PREPARED_FEATURE_DTYPE)

	def trace_features(
		self,
		x_index: int,
		y_index: int,
	) -> tuple[np.ndarray, np.ndarray]:
		x_count, y_count, z_count = self.descriptor.token_shape_xyz
		if not 0 <= x_index < x_count or not 0 <= y_index < y_count:
			raise IndexError('trace index is outside the token grid')
		if self.descriptor.feature_mode == 'z_coordinate':
			z_indices = _direct_trace_z_indices(self.descriptor, x_index, y_index)
			rows = (
				z_indices.astype(PREPARED_FEATURE_DTYPE)
				/ np.float32(max(z_count - 1, 1))
			).reshape(-1, 1)
			return z_indices, rows
		if self.valid_flat_indices is None or self.features is None:
			raise RuntimeError('prepared embedding features are not open')
		start = (x_index * y_count + y_index) * z_count
		stop = start + z_count
		left = int(np.searchsorted(self.valid_flat_indices, start, side='left'))
		right = int(np.searchsorted(self.valid_flat_indices, stop, side='left'))
		flat = self.valid_flat_indices[left:right]
		return (
			np.asarray(flat - start, dtype=np.int64),
			np.asarray(self.features[left:right], dtype=PREPARED_FEATURE_DTYPE),
		)


@dataclass
class PreparedFeatureStore:
	"""Prepared survey features shared by all HMM k/iteration passes."""

	surveys: tuple[PreparedSurveyFeatures, ...]
	settings: PreparedFeatureCacheSettings
	cache_root: Path | None
	feature_mode: str

	def __enter__(self) -> Self:
		"""Return this store for context-manager use."""
		return self

	def __exit__(self, *_: object) -> None:
		"""Close this store when leaving its context."""
		self.close()

	def close(self) -> None:
		"""Close mappings and apply the configured completed-cache policy."""
		for survey in self.surveys:
			survey.close()
			if self.settings.cleanup and survey.cache_path is not None:
				shutil.rmtree(survey.cache_path, ignore_errors=True)

	def by_survey_id(self) -> dict[str, PreparedSurveyFeatures]:
		"""Return prepared surveys keyed by their unique survey ids."""
		return {survey.embedding_input.survey_id: survey for survey in self.surveys}

	def to_metadata(self) -> dict[str, object]:
		"""Describe cache policy, identity, and per-survey reuse."""
		return {
			'effective_mode': 'direct'
			if self.feature_mode == 'z_coordinate'
			else 'memmap',
			'feature_mode': self.feature_mode,
			'dtype': str(PREPARED_FEATURE_DTYPE),
			'schema_version': PREPARED_FEATURE_SCHEMA_VERSION,
			'chunk_size_tokens': self.settings.chunk_size_tokens,
			'reuse': self.settings.reuse,
			'force_rebuild': self.settings.force_rebuild,
			'cleanup': self.settings.cleanup,
			'persist': self.settings.persist,
			'directory': None if self.cache_root is None else str(self.cache_root),
			'surveys': [
				{
					'survey_id': survey.embedding_input.survey_id,
					'fingerprint': survey.fingerprint,
					'valid_token_count': survey.valid_token_count,
					'feature_dim': survey.feature_dim,
					'reused': survey.reused,
					'cache_path': (
						None if survey.cache_path is None else str(survey.cache_path)
					),
				}
				for survey in self.surveys
			],
		}


def prepare_feature_store(  # noqa: PLR0913
	*,
	embedding_inputs: tuple[EmbeddingInput, ...],
	feature_dim: int,
	feature_mode: str,
	residualizer: object | None,
	preprocessor: object,
	edge_margin_tokens: tuple[int, int, int],
	settings: PreparedFeatureCacheSettings,
	default_cache_root: Path,
	prepare_batch: Callable[[EmbeddingInput, np.ndarray], np.ndarray],
) -> PreparedFeatureStore:
	"""Build or reuse prepared valid feature rows for every survey."""
	settings.validate()
	if feature_mode == 'z_coordinate':
		surveys_list: list[PreparedSurveyFeatures] = []
		for item in embedding_inputs:
			token_shape_xyz = _token_shape(item)
			valid_token_count = _count_valid_indices(
				item,
				token_shape_xyz,
				edge_margin_tokens,
				settings.chunk_size_tokens,
			)
			surveys_list.append(
				PreparedSurveyFeatures(
					item,
					token_shape_xyz,
					valid_token_count,
					1,
					feature_mode,
					edge_margin_tokens,
				)
			)
		surveys = tuple(surveys_list)
		return PreparedFeatureStore(surveys, settings, None, feature_mode)
	if feature_mode != 'embedding':
		raise ValueError(f'unsupported prepared feature mode: {feature_mode!r}')

	cache_root = settings.directory or default_cache_root
	cache_root.mkdir(parents=True, exist_ok=True)
	prepared_surveys: list[PreparedSurveyFeatures] = []
	try:
		for item in embedding_inputs:
			token_shape_xyz = _token_shape(item)
			valid_token_count = _count_valid_indices(
				item,
				token_shape_xyz,
				edge_margin_tokens,
				settings.chunk_size_tokens,
			)
			payload = _fingerprint_payload(
				item=item,
				residualizer=residualizer,
				preprocessor=preprocessor,
				feature_mode=feature_mode,
				edge_margin_tokens=edge_margin_tokens,
				feature_dim=feature_dim,
			)
			fingerprint = hashlib.sha256(
				json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(),
			).hexdigest()
			prepared_surveys.append(
				_prepare_one_survey(
					item=item,
					token_shape_xyz=token_shape_xyz,
					valid_token_count=valid_token_count,
					feature_dim=feature_dim,
					fingerprint=fingerprint,
					fingerprint_payload=payload,
					cache_root=cache_root,
					settings=settings,
					edge_margin_tokens=edge_margin_tokens,
					prepare_batch=prepare_batch,
				),
			)
	except BaseException:
		for survey in prepared_surveys:
			survey.close()
			if settings.cleanup and survey.cache_path is not None:
				shutil.rmtree(survey.cache_path, ignore_errors=True)
		raise
	return PreparedFeatureStore(
		tuple(prepared_surveys), settings, cache_root, feature_mode
	)


def _prepare_one_survey(  # noqa: PLR0913
	*,
	item: EmbeddingInput,
	token_shape_xyz: tuple[int, int, int],
	valid_token_count: int,
	feature_dim: int,
	fingerprint: str,
	fingerprint_payload: Mapping[str, object],
	cache_root: Path,
	settings: PreparedFeatureCacheSettings,
	edge_margin_tokens: tuple[int, int, int],
	prepare_batch: Callable[[EmbeddingInput, np.ndarray], np.ndarray],
) -> PreparedSurveyFeatures:
	cache_path = cache_root / fingerprint
	_remove_interrupted_builds(cache_path)
	if settings.reuse and not settings.force_rebuild:
		reused = _open_complete_cache(
			item,
			token_shape_xyz,
			cache_path,
			fingerprint,
			valid_token_count,
			feature_dim,
			edge_margin_tokens,
			settings.chunk_size_tokens,
		)
		if reused is not None:
			return reused
	if cache_path.exists():
		shutil.rmtree(cache_path)
	staging = cache_path.with_name(f'.{cache_path.name}.building-{uuid.uuid4().hex}')
	staging.mkdir()
	try:
		indices_path = staging / 'valid_flat_indices.npy'
		feature_path = staging / 'features.npy'
		if valid_token_count:
			indices = np.lib.format.open_memmap(
				indices_path,
				mode='w+',
				dtype=np.int64,
				shape=(valid_token_count,),
			)
			features = np.lib.format.open_memmap(
				feature_path,
				mode='w+',
				dtype=PREPARED_FEATURE_DTYPE,
				shape=(valid_token_count, feature_dim),
			)
			try:
				row_offset = 0
				for index_chunk in _iter_valid_index_chunks(
					item,
					token_shape_xyz,
					edge_margin_tokens,
					settings.chunk_size_tokens,
				):
					stop = row_offset + index_chunk.size
					rows = np.asarray(
						prepare_batch(item, index_chunk),
						dtype=PREPARED_FEATURE_DTYPE,
					)
					_validate_prepared_batch_shape(
						rows, index_chunk.size, feature_dim
					)
					indices[row_offset:stop] = index_chunk
					features[row_offset:stop] = rows
					row_offset = stop
				if row_offset != valid_token_count:
					raise RuntimeError(
						'valid token count changed while building prepared features'
					)
				indices.flush()
				features.flush()
			finally:
				_close_memmap(indices)
				_close_memmap(features)
		else:
			np.save(indices_path, np.empty(0, dtype=np.int64))
			np.save(
				feature_path, np.empty((0, feature_dim), dtype=PREPARED_FEATURE_DTYPE)
			)
		(staging / 'metadata.json').write_text(
			json.dumps(
				{
					'complete': True,
					'dtype': str(PREPARED_FEATURE_DTYPE),
					'feature_dim': feature_dim,
					'fingerprint': fingerprint,
					'fingerprint_payload': dict(fingerprint_payload),
					'schema_version': PREPARED_FEATURE_SCHEMA_VERSION,
					'survey_id': item.survey_id,
					'token_shape_xyz': list(token_shape_xyz),
					'valid_token_count': valid_token_count,
				},
				indent=2,
				sort_keys=True,
			)
			+ '\n',
			encoding='utf-8',
		)
		staging.replace(cache_path)
	except BaseException:
		shutil.rmtree(staging, ignore_errors=True)
		raise
	prepared = _open_complete_cache(
		item,
		token_shape_xyz,
		cache_path,
		fingerprint,
		valid_token_count,
		feature_dim,
		edge_margin_tokens,
		settings.chunk_size_tokens,
	)
	if prepared is None:
		raise RuntimeError(
			f'failed to open completed prepared feature cache: {cache_path}'
		)
	prepared.reused = False
	return prepared


def _open_complete_cache(  # noqa: PLR0913
	item: EmbeddingInput,
	token_shape_xyz: tuple[int, int, int],
	cache_path: Path,
	fingerprint: str,
	expected_valid_token_count: int,
	feature_dim: int,
	edge_margin_tokens: tuple[int, int, int],
	chunk_size_tokens: int,
) -> PreparedSurveyFeatures | None:
	indices: np.ndarray | None = None
	features: np.ndarray | None = None
	try:
		metadata = json.loads(
			(cache_path / 'metadata.json').read_text(encoding='utf-8')
		)
		if (
			metadata.get('complete') is not True
			or metadata.get('schema_version') != PREPARED_FEATURE_SCHEMA_VERSION
			or metadata.get('fingerprint') != fingerprint
			or metadata.get('dtype') != str(PREPARED_FEATURE_DTYPE)
			or metadata.get('feature_dim') != feature_dim
			or metadata.get('survey_id') != item.survey_id
			or metadata.get('token_shape_xyz') != list(token_shape_xyz)
			or metadata.get('valid_token_count') != expected_valid_token_count
		):
			return None
		mmap_mode = None if expected_valid_token_count == 0 else 'r'
		indices = np.load(cache_path / 'valid_flat_indices.npy', mmap_mode=mmap_mode)
		features = np.load(cache_path / 'features.npy', mmap_mode=mmap_mode)
	except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
		if indices is not None:
			_close_memmap(indices)
		if features is not None:
			_close_memmap(features)
		return None
	valid_cache = (
		indices.dtype == np.dtype(np.int64)
		and indices.shape == (expected_valid_token_count,)
		and _cached_indices_match(
			indices,
			item,
			token_shape_xyz,
			edge_margin_tokens,
			chunk_size_tokens,
		)
		and features.dtype == PREPARED_FEATURE_DTYPE
		and features.shape == (expected_valid_token_count, feature_dim)
	)
	_close_memmap(indices)
	_close_memmap(features)
	if not valid_cache:
		return None
	return PreparedSurveyFeatures(
		item,
		token_shape_xyz,
		expected_valid_token_count,
		feature_dim,
		'embedding',
		edge_margin_tokens,
		fingerprint=fingerprint,
		cache_path=cache_path,
		reused=True,
	)


def _fingerprint_payload(  # noqa: PLR0913
	*,
	item: EmbeddingInput,
	residualizer: object | None,
	preprocessor: object,
	feature_mode: str,
	edge_margin_tokens: tuple[int, int, int],
	feature_dim: int,
) -> dict[str, object]:
	return {
		'schema_version': PREPARED_FEATURE_SCHEMA_VERSION,
		'dtype': str(PREPARED_FEATURE_DTYPE),
		'feature_mode': feature_mode,
		'feature_dim': feature_dim,
		'edge_margin_tokens': list(edge_margin_tokens),
		'source_embedding': _path_signature(item.embeddings_path),
		'source_valid_mask': _path_signature(item.valid_tokens_path),
		'residualizer': None if residualizer is None else joblib.hash(residualizer),
		'preprocessor': joblib.hash(preprocessor),
	}


def _path_signature(path: Path) -> dict[str, object]:
	resolved = path.resolve(strict=True)
	stat = resolved.stat()
	return {
		'path': str(resolved),
		'size': stat.st_size,
		'mtime_ns': stat.st_mtime_ns,
		'ctime_ns': stat.st_ctime_ns,
		'device': stat.st_dev,
		'inode': stat.st_ino,
	}


def _token_shape(item: EmbeddingInput) -> tuple[int, int, int]:
	shape = open_embedding_array(item).shape[:3]
	return (int(shape[0]), int(shape[1]), int(shape[2]))


def _count_valid_indices(
	item: EmbeddingInput,
	token_shape_xyz: tuple[int, int, int],
	edge_margin_tokens: tuple[int, int, int],
	chunk_size_tokens: int,
) -> int:
	return sum(
		int(indices.size)
		for indices in _iter_valid_index_chunks(
			item,
			token_shape_xyz,
			edge_margin_tokens,
			chunk_size_tokens,
		)
	)


def _iter_valid_index_chunks(
	item: EmbeddingInput,
	token_shape_xyz: tuple[int, int, int],
	edge_margin_tokens: tuple[int, int, int],
	chunk_size_tokens: int,
) -> Iterator[np.ndarray]:
	valid = load_valid_tokens(item)
	if valid.shape != token_shape_xyz:
		raise ValueError(
			'valid token grid shape changed while preparing features; '
			f'got {valid.shape!r}, expected {token_shape_xyz!r}'
		)
	_validate_edge_margins(item, token_shape_xyz, edge_margin_tokens)
	flat_valid = valid.reshape(-1)
	token_count = flat_valid.size
	mx, my, mz = edge_margin_tokens
	buffer = np.empty(chunk_size_tokens, dtype=np.int64)
	buffered = 0
	for start in range(0, token_count, chunk_size_tokens):
		stop = min(start + chunk_size_tokens, token_count)
		indices = np.flatnonzero(flat_valid[start:stop]).astype(np.int64, copy=False)
		indices += start
		if indices.size and (mx or my or mz):
			x, y, z = np.unravel_index(indices, token_shape_xyz)
			inside = (
				(np.asarray(x) >= mx)
				& (np.asarray(x) < token_shape_xyz[0] - mx)
				& (np.asarray(y) >= my)
				& (np.asarray(y) < token_shape_xyz[1] - my)
				& (np.asarray(z) >= mz)
				& (np.asarray(z) < token_shape_xyz[2] - mz)
			)
			indices = indices[inside]
		index_offset = 0
		while index_offset < indices.size:
			copied = min(chunk_size_tokens - buffered, indices.size - index_offset)
			buffer[buffered : buffered + copied] = indices[
				index_offset : index_offset + copied
			]
			buffered += copied
			index_offset += copied
			if buffered == chunk_size_tokens:
				yield buffer
				buffer = np.empty(chunk_size_tokens, dtype=np.int64)
				buffered = 0
	if buffered:
		yield buffer[:buffered].copy()


def _validate_edge_margins(
	item: EmbeddingInput,
	token_shape_xyz: tuple[int, int, int],
	edge_margin_tokens: tuple[int, int, int],
) -> None:
	if len(edge_margin_tokens) != 3 or any(
		isinstance(margin, bool) or not isinstance(margin, int) or margin < 0
		for margin in edge_margin_tokens
	):
		raise ValueError(
			'edge_margin_tokens must contain three nonnegative integers; '
			f'got {edge_margin_tokens!r}'
		)
	if any(
		2 * margin >= size
		for size, margin in zip(
			token_shape_xyz, edge_margin_tokens, strict=True
		)
	):
		raise ValueError(
			f'edge_margin_tokens {edge_margin_tokens!r} leave no interior tokens '
			f'for survey {item.survey_id} with token grid shape {token_shape_xyz!r}'
		)


def _cached_indices_match(
	cached_indices: np.ndarray,
	item: EmbeddingInput,
	token_shape_xyz: tuple[int, int, int],
	edge_margin_tokens: tuple[int, int, int],
	chunk_size_tokens: int,
) -> bool:
	offset = 0
	for expected in _iter_valid_index_chunks(
		item,
		token_shape_xyz,
		edge_margin_tokens,
		chunk_size_tokens,
	):
		stop = offset + expected.size
		if stop > cached_indices.size or not np.array_equal(
			cached_indices[offset:stop], expected
		):
			return False
		offset = stop
	return offset == cached_indices.size


def _validate_direct_indices(
	survey: PreparedSurveyFeatures,
	indices: np.ndarray,
) -> None:
	shape = survey.token_shape_xyz
	token_count = int(np.prod(shape))
	if np.any(indices < 0) or np.any(indices >= token_count):
		raise ValueError('flat_indices contain tokens outside the prepared valid set')
	x, y, z = np.unravel_index(indices, shape)
	mx, my, mz = survey.edge_margin_tokens
	inside_margin = (
		(np.asarray(x) >= mx)
		& (np.asarray(x) < shape[0] - mx)
		& (np.asarray(y) >= my)
		& (np.asarray(y) < shape[1] - my)
		& (np.asarray(z) >= mz)
		& (np.asarray(z) < shape[2] - mz)
	)
	valid = load_valid_tokens(survey.embedding_input).reshape(-1)
	if not np.all(inside_margin) or not np.all(valid[indices]):
		raise ValueError('flat_indices contain tokens outside the prepared valid set')


def _direct_trace_z_indices(
	survey: PreparedSurveyFeatures,
	x_index: int,
	y_index: int,
) -> np.ndarray:
	shape = survey.token_shape_xyz
	mx, my, mz = survey.edge_margin_tokens
	if (
		x_index < mx
		or x_index >= shape[0] - mx
		or y_index < my
		or y_index >= shape[1] - my
	):
		return np.empty(0, dtype=np.int64)
	valid = load_valid_tokens(survey.embedding_input)
	inside = np.asarray(valid[x_index, y_index, mz : shape[2] - mz])
	return np.flatnonzero(inside).astype(np.int64, copy=False) + mz


def _validate_prepared_batch_shape(
	rows: np.ndarray,
	row_count: int,
	feature_dim: int,
) -> None:
	expected = (row_count, feature_dim)
	if rows.shape != expected:
		raise ValueError(
			'prepared feature batch has unexpected shape; '
			f'got {rows.shape!r}, expected {expected!r}',
		)


def _remove_interrupted_builds(cache_path: Path) -> None:
	for staging in cache_path.parent.glob(f'.{cache_path.name}.building-*'):
		shutil.rmtree(staging, ignore_errors=True)


def _close_memmap(array: np.ndarray) -> None:
	mmap = getattr(array, '_mmap', None)
	if mmap is not None:
		mmap.close()


__all__ = [
	'PREPARED_FEATURE_DTYPE',
	'PREPARED_FEATURE_SCHEMA_VERSION',
	'PreparedFeatureCacheSettings',
	'PreparedFeatureStore',
	'PreparedSurveyFeatures',
	'prepare_feature_store',
]
