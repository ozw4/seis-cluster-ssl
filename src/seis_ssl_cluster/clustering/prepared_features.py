"""Prepared feature storage for repeated stratigraphic HMM passes."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import joblib
import numpy as np

from seis_ssl_cluster.clustering.features import EmbeddingInput, open_embedding_array

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping
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
	"""Prepared valid rows and their token-grid index mapping for one survey."""

	embedding_input: EmbeddingInput
	_token_shape_xyz: tuple[int, int, int]
	valid_flat_indices: np.ndarray
	features: np.ndarray | None
	feature_dim: int
	feature_mode: str
	fingerprint: str | None = None
	cache_path: Path | None = None
	reused: bool = False

	@property
	def token_shape_xyz(self) -> tuple[int, int, int]:
		"""Return the source token-grid shape."""
		return self._token_shape_xyz

	def features_for_flat_indices(self, flat_indices: np.ndarray) -> np.ndarray:
		"""Return prepared rows for sorted or unsorted valid flattened indices."""
		indices = np.asarray(flat_indices, dtype=np.int64)
		if indices.ndim != 1:
			raise ValueError(f'flat_indices must be 1D; got shape {indices.shape!r}')
		if indices.size == 0:
			return np.empty((0, self.feature_dim), dtype=PREPARED_FEATURE_DTYPE)
		if self.feature_mode == 'z_coordinate':
			z_size = self.token_shape_xyz[2]
			z = np.asarray(np.unravel_index(indices, self.token_shape_xyz)[2])
			return (
				z.astype(PREPARED_FEATURE_DTYPE) / np.float32(max(z_size - 1, 1))
			).reshape(-1, 1)
		if self.features is None:
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
		"""Return valid z indices and prepared rows for one vertical trace."""
		x_count, y_count, z_count = self.token_shape_xyz
		if not 0 <= x_index < x_count or not 0 <= y_index < y_count:
			raise IndexError('trace index is outside the token grid')
		start = (x_index * y_count + y_index) * z_count
		stop = start + z_count
		left = int(np.searchsorted(self.valid_flat_indices, start, side='left'))
		right = int(np.searchsorted(self.valid_flat_indices, stop, side='left'))
		flat = self.valid_flat_indices[left:right]
		z_indices = np.asarray(flat - start, dtype=np.int64)
		if self.feature_mode == 'z_coordinate':
			rows = (
				z_indices.astype(PREPARED_FEATURE_DTYPE)
				/ np.float32(max(z_count - 1, 1))
			).reshape(-1, 1)
		else:
			if self.features is None:
				raise RuntimeError('prepared embedding features are not open')
			rows = np.asarray(self.features[left:right], dtype=PREPARED_FEATURE_DTYPE)
		return z_indices, rows

	def close(self) -> None:
		"""Close memory mappings owned by this prepared survey."""
		for array in (self.valid_flat_indices, self.features):
			if array is not None:
				_close_memmap(array)


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
					'valid_token_count': int(survey.valid_flat_indices.size),
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
	valid_indices_by_survey: Mapping[str, np.ndarray],
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
		surveys = tuple(
			PreparedSurveyFeatures(
				item,
				_token_shape(item),
				_validate_valid_indices(valid_indices_by_survey[item.survey_id]),
				None,
				1,
				feature_mode,
			)
			for item in embedding_inputs
		)
		return PreparedFeatureStore(surveys, settings, None, feature_mode)
	if feature_mode != 'embedding':
		raise ValueError(f'unsupported prepared feature mode: {feature_mode!r}')

	cache_root = settings.directory or default_cache_root
	cache_root.mkdir(parents=True, exist_ok=True)
	prepared_surveys: list[PreparedSurveyFeatures] = []
	try:
		for item in embedding_inputs:
			token_shape_xyz = _token_shape(item)
			valid_indices = _validate_valid_indices(
				valid_indices_by_survey[item.survey_id],
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
					valid_indices=valid_indices,
					feature_dim=feature_dim,
					fingerprint=fingerprint,
					fingerprint_payload=payload,
					cache_root=cache_root,
					settings=settings,
					prepare_batch=prepare_batch,
				),
			)
	except BaseException:
		for survey in prepared_surveys:
			survey.close()
		raise
	return PreparedFeatureStore(
		tuple(prepared_surveys), settings, cache_root, feature_mode
	)


def _prepare_one_survey(  # noqa: PLR0913
	*,
	item: EmbeddingInput,
	token_shape_xyz: tuple[int, int, int],
	valid_indices: np.ndarray,
	feature_dim: int,
	fingerprint: str,
	fingerprint_payload: Mapping[str, object],
	cache_root: Path,
	settings: PreparedFeatureCacheSettings,
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
			valid_indices,
			feature_dim,
		)
		if reused is not None:
			return reused
	if cache_path.exists():
		shutil.rmtree(cache_path)
	staging = cache_path.with_name(f'.{cache_path.name}.building-{uuid.uuid4().hex}')
	staging.mkdir()
	try:
		np.save(staging / 'valid_flat_indices.npy', valid_indices)
		feature_path = staging / 'features.npy'
		if valid_indices.size:
			features = np.lib.format.open_memmap(
				feature_path,
				mode='w+',
				dtype=PREPARED_FEATURE_DTYPE,
				shape=(valid_indices.size, feature_dim),
			)
			for start in range(0, valid_indices.size, settings.chunk_size_tokens):
				stop = min(start + settings.chunk_size_tokens, valid_indices.size)
				rows = np.asarray(
					prepare_batch(item, valid_indices[start:stop]),
					dtype=PREPARED_FEATURE_DTYPE,
				)
				_validate_prepared_batch_shape(rows, stop - start, feature_dim)
				features[start:stop] = rows
			features.flush()
			_close_memmap(features)
		else:
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
					'valid_token_count': int(valid_indices.size),
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
		valid_indices,
		feature_dim,
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
	expected_indices: np.ndarray,
	feature_dim: int,
) -> PreparedSurveyFeatures | None:
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
			or metadata.get('valid_token_count') != int(expected_indices.size)
		):
			return None
		mmap_mode = None if expected_indices.size == 0 else 'r'
		indices = np.load(cache_path / 'valid_flat_indices.npy', mmap_mode=mmap_mode)
		features = np.load(cache_path / 'features.npy', mmap_mode=mmap_mode)
	except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
		return None
	if (
		indices.dtype != np.dtype(np.int64)
		or indices.shape != expected_indices.shape
		or not np.array_equal(indices, expected_indices)
		or features.dtype != PREPARED_FEATURE_DTYPE
		or features.shape != (expected_indices.size, feature_dim)
	):
		_close_memmap(indices)
		_close_memmap(features)
		return None
	return PreparedSurveyFeatures(
		item,
		token_shape_xyz,
		indices,
		features,
		feature_dim,
		'embedding',
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


def _validate_valid_indices(value: np.ndarray) -> np.ndarray:
	indices = np.asarray(value, dtype=np.int64)
	if indices.ndim != 1:
		raise ValueError(f'valid token indices must be 1D; got {indices.shape!r}')
	if indices.size > 1 and np.any(indices[1:] <= indices[:-1]):
		raise ValueError('valid token indices must be strictly increasing')
	return indices


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
