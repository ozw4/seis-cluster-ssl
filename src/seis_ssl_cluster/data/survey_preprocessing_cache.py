"""Survey-scoped invariant preprocessing caches for window extraction."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from seis_ssl_cluster.data.normalization import (
	SurveyNormalizationStats,
	_normalize_amplitude_inplace,
)
from seis_ssl_cluster.data.volume_store import inspect_npy_volume

if TYPE_CHECKING:
	from pathlib import Path

	from typing_extensions import Self

	from seis_ssl_cluster.data.volume_store import NpyMemmapVolumeStore
	from seis_ssl_cluster.data.window_preprocessing import AmplitudePreprocessSettings

SurveyPreprocessingCacheMode = Literal['off', 'memory', 'memmap']
_CACHE_SCHEMA_VERSION = 1
_CACHE_DTYPE = np.dtype(np.float32)


@dataclass(frozen=True)
class SurveyPreprocessingCacheSettings:
	"""Storage and lifecycle settings for survey preprocessing caches."""

	mode: SurveyPreprocessingCacheMode = 'off'
	chunk_size_x: int = 16
	reuse: bool = True
	cleanup: bool = False
	directory: Path | None = None

	def validate(self) -> None:
		"""Validate cache settings."""
		if self.mode not in {'off', 'memory', 'memmap'}:
			msg = (
				'preprocessing_cache.mode must be "off", "memory", or '
				f'"memmap"; got {self.mode!r}'
			)
			raise ValueError(msg)
		if isinstance(self.chunk_size_x, bool) or not isinstance(
			self.chunk_size_x,
			int,
		):
			msg = 'preprocessing_cache.chunk_size_x must be a positive integer'
			raise TypeError(msg)
		if self.chunk_size_x <= 0:
			msg = 'preprocessing_cache.chunk_size_x must be a positive integer'
			raise ValueError(msg)
		for name in ('reuse', 'cleanup'):
			if not isinstance(getattr(self, name), bool):
				msg = f'preprocessing_cache.{name} must be a boolean'
				raise TypeError(msg)


@dataclass(frozen=True)
class SurveyPreprocessingCachePlan:
	"""Deterministic cache identity and effective execution mode."""

	requested_mode: SurveyPreprocessingCacheMode
	effective_mode: SurveyPreprocessingCacheMode
	fingerprint: str | None
	fingerprint_payload: dict[str, object] | None
	cache_root: Path | None
	reason: str | None = None

	def to_metadata(self) -> dict[str, object]:
		"""Return output metadata describing cache selection and identity."""
		metadata: dict[str, object] = {
			'requested_mode': self.requested_mode,
			'effective_mode': self.effective_mode,
			'schema_version': _CACHE_SCHEMA_VERSION,
			'dtype': str(_CACHE_DTYPE),
			'fingerprint': self.fingerprint,
		}
		if self.reason is not None:
			metadata['fallback_reason'] = self.reason
		return metadata


@dataclass
class PreparedSurveyAmplitude:
	"""Window-invariant survey data used to reproduce legacy preprocessing."""

	normalized_amplitude: np.ndarray
	zero_like_mask: np.ndarray
	mode: SurveyPreprocessingCacheMode
	fingerprint: str
	cache_path: Path | None = None
	cleanup: bool = False
	reused: bool = False

	def close(self) -> None:
		"""Close mapped arrays and apply the configured cleanup policy."""
		for array in (self.normalized_amplitude, self.zero_like_mask):
			_close_memmap(array)
		if self.cleanup and self.cache_path is not None:
			shutil.rmtree(self.cache_path, ignore_errors=True)

	def __enter__(self) -> Self:
		"""Return this prepared survey for context-manager use."""
		return self

	def __exit__(self, *_: object) -> None:
		"""Close this prepared survey when leaving a context."""
		self.close()


def plan_survey_preprocessing_cache(
	*,
	amplitude_path: Path,
	stats: SurveyNormalizationStats,
	preprocess_settings: AmplitudePreprocessSettings,
	cache_settings: SurveyPreprocessingCacheSettings,
	default_cache_root: Path,
) -> SurveyPreprocessingCachePlan:
	"""Build a cache plan without creating cache artifacts."""
	cache_settings.validate()
	if cache_settings.mode == 'off':
		return SurveyPreprocessingCachePlan('off', 'off', None, None, None)
	unsafe_reason = _cache_fallback_reason(preprocess_settings)
	if unsafe_reason is not None:
		return SurveyPreprocessingCachePlan(
			cache_settings.mode,
			'off',
			None,
			None,
			None,
			unsafe_reason,
		)
	payload = _fingerprint_payload(
		amplitude_path=amplitude_path,
		stats=stats,
		settings=preprocess_settings,
	)
	fingerprint = hashlib.sha256(
		json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(),
	).hexdigest()
	return SurveyPreprocessingCachePlan(
		cache_settings.mode,
		cache_settings.mode,
		fingerprint,
		payload,
		cache_settings.directory or default_cache_root,
	)


def prepare_survey_preprocessing_cache(  # noqa: PLR0913
	*,
	plan: SurveyPreprocessingCachePlan,
	amplitude_path: Path,
	stats: SurveyNormalizationStats,
	preprocess_settings: AmplitudePreprocessSettings,
	cache_settings: SurveyPreprocessingCacheSettings,
	store: NpyMemmapVolumeStore,
) -> PreparedSurveyAmplitude | None:
	"""Build or reuse the prepared survey arrays described by ``plan``."""
	if plan.effective_mode == 'off':
		return None
	if plan.fingerprint is None or plan.fingerprint_payload is None:
		raise RuntimeError('enabled preprocessing cache plan has no fingerprint')
	source = store.open(amplitude_path)
	shape = tuple(int(axis) for axis in source.shape)
	if plan.effective_mode == 'memory':
		normalized = np.empty(shape, dtype=_CACHE_DTYPE)
		zero_like = np.empty(shape, dtype=bool)
		_build_arrays(
			source,
			normalized,
			zero_like,
			stats=stats,
			settings=preprocess_settings,
			chunk_size_x=cache_settings.chunk_size_x,
		)
		return PreparedSurveyAmplitude(
			normalized,
			zero_like,
			'memory',
			plan.fingerprint,
		)
	if plan.cache_root is None:
		raise RuntimeError('memmap preprocessing cache plan has no cache root')
	cache_path = plan.cache_root / plan.fingerprint
	cache_path.parent.mkdir(parents=True, exist_ok=True)
	_remove_interrupted_builds(cache_path)
	if cache_settings.reuse:
		reused = _open_complete_cache(cache_path, shape, plan.fingerprint)
		if reused is not None:
			reused.cleanup = cache_settings.cleanup
			return reused
	if cache_path.exists():
		shutil.rmtree(cache_path)
	staging = cache_path.with_name(f'.{cache_path.name}.building-{uuid.uuid4().hex}')
	staging.mkdir()
	try:
		normalized = np.lib.format.open_memmap(
			staging / 'normalized_amplitude.npy',
			mode='w+',
			dtype=_CACHE_DTYPE,
			shape=shape,
		)
		zero_like = np.lib.format.open_memmap(
			staging / 'zero_like_mask.npy',
			mode='w+',
			dtype=bool,
			shape=shape,
		)
		_build_arrays(
			source,
			normalized,
			zero_like,
			stats=stats,
			settings=preprocess_settings,
			chunk_size_x=cache_settings.chunk_size_x,
		)
		normalized.flush()
		zero_like.flush()
		_close_memmap(normalized)
		_close_memmap(zero_like)
		(staging / 'metadata.json').write_text(
			json.dumps(
				{
					'complete': True,
					'dtype': str(_CACHE_DTYPE),
					'fingerprint': plan.fingerprint,
					'fingerprint_payload': plan.fingerprint_payload,
					'mode': 'memmap',
					'schema_version': _CACHE_SCHEMA_VERSION,
					'shape_xyz': list(shape),
					'zero_mask_dtype': 'bool',
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
	prepared = _open_complete_cache(cache_path, shape, plan.fingerprint)
	if prepared is None:
		raise RuntimeError(f'failed to open completed cache: {cache_path}')
	prepared.cleanup = cache_settings.cleanup
	prepared.reused = False
	return prepared


def _remove_interrupted_builds(cache_path: Path) -> None:
	pattern = f'.{cache_path.name}.building-*'
	for staging in cache_path.parent.glob(pattern):
		shutil.rmtree(staging, ignore_errors=True)


def _build_arrays(  # noqa: PLR0913
	source: np.ndarray,
	normalized: np.ndarray,
	zero_like: np.ndarray,
	*,
	stats: SurveyNormalizationStats,
	settings: AmplitudePreprocessSettings,
	chunk_size_x: int,
) -> None:
	for start_x in range(0, source.shape[0], chunk_size_x):
		stop_x = min(start_x + chunk_size_x, source.shape[0])
		chunk = np.array(source[start_x:stop_x], dtype=np.float32, copy=True)
		if settings.finite_check_mode == 'strict' and not np.isfinite(chunk).all():
			msg = 'raw amplitude contains non-finite values in valid source voxels'
			raise ValueError(msg)
		if settings.zero_mask.enabled:
			zero_like[start_x:stop_x] = (
				~np.isfinite(chunk)
				| (np.abs(chunk) <= np.float32(settings.zero_mask.zero_atol))
			)
		else:
			zero_like[start_x:stop_x] = False
		_normalize_amplitude_inplace(
			chunk,
			stats,
			normalized_clip_abs=settings.normalized_clip_abs,
		)
		if settings.finite_check_mode == 'strict' and not np.isfinite(chunk).all():
			msg = 'normalized amplitude contains non-finite values'
			raise ValueError(msg)
		normalized[start_x:stop_x] = chunk


def _open_complete_cache(
	cache_path: Path,
	shape: tuple[int, ...],
	fingerprint: str,
) -> PreparedSurveyAmplitude | None:
	try:
		metadata = json.loads(
			(cache_path / 'metadata.json').read_text(encoding='utf-8'),
		)
		if (
			metadata.get('complete') is not True
			or metadata.get('schema_version') != _CACHE_SCHEMA_VERSION
			or metadata.get('fingerprint') != fingerprint
			or metadata.get('mode') != 'memmap'
			or metadata.get('dtype') != str(_CACHE_DTYPE)
			or metadata.get('shape_xyz') != list(shape)
			or metadata.get('zero_mask_dtype') != 'bool'
		):
			return None
		normalized = np.load(cache_path / 'normalized_amplitude.npy', mmap_mode='r')
		zero_like = np.load(cache_path / 'zero_like_mask.npy', mmap_mode='r')
	except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
		return None
	if (
		normalized.shape != shape
		or normalized.dtype != _CACHE_DTYPE
		or zero_like.shape != shape
		or zero_like.dtype != np.dtype(bool)
	):
		_close_memmap(normalized)
		_close_memmap(zero_like)
		return None
	return PreparedSurveyAmplitude(
		normalized,
		zero_like,
		'memmap',
		fingerprint,
		cache_path=cache_path,
		reused=True,
	)


def _fingerprint_payload(
	*,
	amplitude_path: Path,
	stats: SurveyNormalizationStats,
	settings: AmplitudePreprocessSettings,
) -> dict[str, object]:
	resolved = amplitude_path.resolve(strict=True)
	stat = resolved.stat()
	source_info = inspect_npy_volume(resolved)
	return {
		'schema_version': _CACHE_SCHEMA_VERSION,
		'dtype': str(_CACHE_DTYPE),
		'source_signature': {
			'path': str(resolved),
			'size': stat.st_size,
			'mtime_ns': stat.st_mtime_ns,
			'device': stat.st_dev,
			'inode': stat.st_ino,
			'ctime_ns': stat.st_ctime_ns,
			'shape_xyz': list(source_info.shape_xyz),
			'dtype': source_info.dtype,
		},
		'normalization': {
			'stats': stats.to_dict(),
			'normalized_clip_abs': settings.normalized_clip_abs,
			'finite_check_mode': settings.finite_check_mode,
		},
		'zero_mask': {
			'enabled': settings.zero_mask.enabled,
			'zero_atol': settings.zero_mask.zero_atol,
			'z_sample_influence_radius': settings.zero_mask.z_sample_influence_radius,
			'xy_trace_influence_radius': settings.zero_mask.xy_trace_influence_radius,
		},
	}


def _cache_fallback_reason(settings: AmplitudePreprocessSettings) -> str | None:
	try:
		settings.zero_mask.validate()
		settings.amplitude_agc.validate()
	except (TypeError, ValueError) as exc:
		return f'unsupported preprocessing settings: {exc}'
	if settings.finite_check_mode not in {'strict', 'output_only', 'off'}:
		return f'unsupported finite_check_mode: {settings.finite_check_mode!r}'
	if settings.amplitude_agc.enabled and settings.amplitude_agc.mode != 'trace_rms_z':
		return f'unsupported window-local AGC mode: {settings.amplitude_agc.mode!r}'
	return None


def _close_memmap(array: np.ndarray) -> None:
	mapping = getattr(array, '_mmap', None)
	if mapping is not None:
		mapping.close()


__all__ = [
	'PreparedSurveyAmplitude',
	'SurveyPreprocessingCacheMode',
	'SurveyPreprocessingCachePlan',
	'SurveyPreprocessingCacheSettings',
	'plan_survey_preprocessing_cache',
	'prepare_survey_preprocessing_cache',
]
