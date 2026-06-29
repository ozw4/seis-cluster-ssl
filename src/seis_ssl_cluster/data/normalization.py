"""Survey-wise robust normalization for amplitude-only volumes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import cast

import numpy as np

from seis_ssl_cluster.data.schema import GRID_ORDER_XYZ
from seis_ssl_cluster.data.volume_store import inspect_npy_volume


@dataclass(frozen=True)
class SurveyNormalizationStats:
	"""Survey-level robust amplitude normalization statistics."""

	survey_id: str
	source_path: Path
	grid_order: tuple[str, str, str]
	clip_low_percentile: float
	clip_high_percentile: float
	clip_low: float
	clip_high: float
	median: float
	iqr: float
	eps: float = 1.0e-6

	def validate(self) -> None:
		"""Validate the fixed survey-wise normalization contract."""
		if not self.survey_id:
			msg = 'normalization stats survey_id must be non-empty'
			raise ValueError(msg)
		if self.grid_order != GRID_ORDER_XYZ:
			msg = (
				f'normalization stats grid_order must be {GRID_ORDER_XYZ!r}; '
				f'got {self.grid_order!r}'
			)
			raise ValueError(msg)
		if not 0.0 <= self.clip_low_percentile < self.clip_high_percentile <= 100.0:
			msg = (
				'normalization stats clipping percentiles must satisfy '
				'0 <= low < high <= 100; got '
				f'{self.clip_low_percentile!r}, {self.clip_high_percentile!r}'
			)
			raise ValueError(msg)
		if self.clip_low > self.clip_high:
			msg = (
				'normalization stats clip_low must be less than or equal to '
				f'clip_high; got {self.clip_low!r}, {self.clip_high!r}'
			)
			raise ValueError(msg)
		if self.iqr < 0.0:
			msg = f'normalization stats iqr must be nonnegative; got {self.iqr!r}'
			raise ValueError(msg)
		if self.eps <= 0.0:
			msg = f'normalization stats eps must be positive; got {self.eps!r}'
			raise ValueError(msg)

	def to_dict(self) -> dict[str, object]:
		"""Convert stats to a JSON-compatible dictionary."""
		return {
			'survey_id': self.survey_id,
			'source_path': str(self.source_path),
			'grid_order': list(self.grid_order),
			'clip_low_percentile': self.clip_low_percentile,
			'clip_high_percentile': self.clip_high_percentile,
			'clip_low': self.clip_low,
			'clip_high': self.clip_high,
			'median': self.median,
			'iqr': self.iqr,
			'eps': self.eps,
		}

	@classmethod
	def from_mapping(
		cls,
		data: Mapping[str, object],
		*,
		path: Path | None = None,
	) -> SurveyNormalizationStats:
		"""Build and validate stats from decoded JSON fields."""
		label = str(path) if path is not None else 'normalization stats'
		stats = cls(
			survey_id=_required_str(data, 'survey_id', label),
			source_path=Path(_required_str(data, 'source_path', label)),
			grid_order=_required_grid_order(data, 'grid_order', label),
			clip_low_percentile=_required_float(
				data,
				'clip_low_percentile',
				label,
			),
			clip_high_percentile=_required_float(
				data,
				'clip_high_percentile',
				label,
			),
			clip_low=_required_float(data, 'clip_low', label),
			clip_high=_required_float(data, 'clip_high', label),
			median=_required_float(data, 'median', label),
			iqr=_required_float(data, 'iqr', label),
			eps=_required_float(data, 'eps', label),
		)
		stats.validate()
		return stats


@dataclass(frozen=True)
class AmplitudeAgcConfig:
	"""Trace-wise local RMS AGC settings for model amplitudes."""

	enabled: bool = False
	mode: str | None = None
	window_z: int | None = None
	eps: float | None = None
	clip_abs: float | None = None

	def validate(self) -> None:
		"""Validate the configured amplitude AGC contract."""
		if not isinstance(self.enabled, bool):
			msg = f'amplitude_agc.enabled must be a boolean; got {self.enabled!r}'
			raise TypeError(msg)
		if not self.enabled:
			for key in ('mode', 'window_z', 'eps', 'clip_abs'):
				if getattr(self, key) is not None:
					msg = (
						f'amplitude_agc.{key} must be omitted when '
						'amplitude_agc.enabled is false'
					)
					raise ValueError(msg)
			return
		if self.mode != 'trace_rms_z':
			msg = (
				"amplitude_agc.mode must be 'trace_rms_z' when enabled; "
				f'got {self.mode!r}'
			)
			raise ValueError(msg)
		_validate_positive_odd_int(self.window_z, 'amplitude_agc.window_z')
		_validate_positive_finite_float(self.eps, 'amplitude_agc.eps')
		_validate_positive_finite_float(self.clip_abs, 'amplitude_agc.clip_abs')

	def to_dict(self) -> dict[str, object]:
		"""Convert AGC settings to a JSON/YAML-compatible dictionary."""
		self.validate()
		if not self.enabled:
			return {'enabled': False}
		return {
			'enabled': True,
			'mode': cast('str', self.mode),
			'window_z': cast('int', self.window_z),
			'eps': cast('float', self.eps),
			'clip_abs': cast('float', self.clip_abs),
		}

	@classmethod
	def from_mapping(
		cls,
		data: Mapping[str, object] | None,
	) -> AmplitudeAgcConfig:
		"""Build and validate AGC settings from a config mapping."""
		if data is None:
			config = cls()
			config.validate()
			return config
		if not isinstance(data, Mapping):
			msg = f'amplitude_agc config must be a mapping; got {data!r}'
			raise TypeError(msg)
		unexpected = sorted(
			set(data) - {'enabled', 'mode', 'window_z', 'eps', 'clip_abs'},
		)
		if unexpected:
			msg = f'amplitude_agc key(s) not allowed: {unexpected!r}'
			raise ValueError(msg)
		enabled = data.get('enabled')
		if not isinstance(enabled, bool):
			msg = f'amplitude_agc.enabled must be a boolean; got {enabled!r}'
			raise TypeError(msg)
		if not enabled:
			config = cls(enabled=False)
			extra = sorted(set(data) - {'enabled'})
			if extra:
				msg = (
					'amplitude_agc fields must be omitted when disabled; '
					f'got {extra!r}'
				)
				raise ValueError(msg)
			config.validate()
			return config
		config = cls(
			enabled=True,
			mode=_required_agc_value(data, 'mode'),
			window_z=_required_agc_int(data, 'window_z'),
			eps=_required_agc_float(data, 'eps'),
			clip_abs=_required_agc_float(data, 'clip_abs'),
		)
		config.validate()
		return config


def load_normalization_stats(path: str | Path) -> SurveyNormalizationStats:
	"""Load survey-level normalization statistics from a JSON file."""
	stats_path = Path(path)
	data = json.loads(stats_path.read_text(encoding='utf-8'))
	if not isinstance(data, Mapping):
		msg = f'normalization stats must be a JSON object: {stats_path}'
		raise TypeError(msg)
	return SurveyNormalizationStats.from_mapping(data, path=stats_path)


def write_normalization_stats(
	stats: SurveyNormalizationStats,
	path: str | Path,
) -> None:
	"""Write survey-level normalization statistics as deterministic JSON."""
	stats.validate()
	stats_path = Path(path)
	stats_path.parent.mkdir(parents=True, exist_ok=True)
	stats_path.write_text(
		json.dumps(stats.to_dict(), indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def normalize_amplitude(
	crop: np.ndarray,
	stats: SurveyNormalizationStats,
	*,
	normalized_clip_abs: float | None = None,
) -> np.ndarray:
	"""Clip and robust-scale an amplitude crop without changing XYZ order."""
	stats.validate()
	if (
		normalized_clip_abs is not None
		and (not np.isfinite(normalized_clip_abs) or normalized_clip_abs <= 0.0)
	):
		msg = (
			'normalized_clip_abs must be a finite positive number; '
			f'got {normalized_clip_abs!r}'
		)
		raise ValueError(msg)
	amplitude = np.asarray(crop, dtype=np.float32)
	clipped = np.clip(amplitude, stats.clip_low, stats.clip_high)
	normalized = (clipped - stats.median) / (stats.iqr + stats.eps)
	if normalized_clip_abs is not None:
		limit = np.float32(normalized_clip_abs)
		normalized = np.clip(normalized, -limit, limit)
	return normalized.astype(np.float32, copy=False)


def apply_configured_agc(
	amplitude: np.ndarray,
	valid_mask: np.ndarray,
	config: AmplitudeAgcConfig | Mapping[str, object] | None,
) -> np.ndarray:
	"""Apply configured amplitude AGC or return a float32 copy when disabled."""
	agc = (
		config
		if isinstance(config, AmplitudeAgcConfig)
		else AmplitudeAgcConfig.from_mapping(config)
	)
	agc.validate()
	if not agc.enabled:
		return np.asarray(amplitude, dtype=np.float32).copy()
	return apply_trace_rms_agc(
		amplitude,
		valid_mask,
		window_z=cast('int', agc.window_z),
		eps=cast('float', agc.eps),
		clip_abs=cast('float', agc.clip_abs),
	)


def apply_trace_rms_agc(
	amplitude: np.ndarray,
	valid_mask: np.ndarray,
	*,
	window_z: int,
	eps: float,
	clip_abs: float,
) -> np.ndarray:
	"""Apply centered trace-wise local RMS AGC along the z axis."""
	_validate_positive_odd_int(window_z, 'window_z')
	eps_float = _validate_positive_finite_float(eps, 'eps')
	clip_float = _validate_positive_finite_float(clip_abs, 'clip_abs')
	values = np.asarray(amplitude, dtype=np.float32)
	valid = np.asarray(valid_mask, dtype=bool)
	if values.ndim != 3:
		msg = f'amplitude must be 3D [X, Y, Z]; got shape={values.shape!r}'
		raise ValueError(msg)
	if valid.shape != values.shape:
		msg = (
			'valid_mask shape must match amplitude shape; '
			f'got {valid.shape!r} and {values.shape!r}'
		)
		raise ValueError(msg)
	valid_values = np.where(valid, values, np.float32(0.0))
	power = valid_values * valid_values
	local_power_sum = _moving_sum_z(power, window_z)
	local_valid_count = _moving_sum_z(valid.astype(np.float32), window_z)
	local_mean_power = local_power_sum / np.maximum(
		local_valid_count,
		np.float32(1.0),
	)
	local_rms = np.sqrt(local_mean_power + np.float32(eps_float))
	agc = valid_values / local_rms
	agc = np.clip(agc, -np.float32(clip_float), np.float32(clip_float))
	agc[~valid] = np.float32(0.0)
	return agc.astype(np.float32, copy=False)


def _moving_sum_z(values: np.ndarray, window_z: int) -> np.ndarray:
	radius = window_z // 2
	padded = np.pad(
		np.asarray(values, dtype=np.float32),
		((0, 0), (0, 0), (radius, radius)),
		mode='constant',
		constant_values=0.0,
	)
	cumulative = np.cumsum(padded, axis=2, dtype=np.float32)
	zero = np.zeros((*cumulative.shape[:2], 1), dtype=np.float32)
	cumulative = np.concatenate((zero, cumulative), axis=2)
	return cumulative[:, :, window_z:] - cumulative[:, :, :-window_z]


def compute_normalization_stats(  # noqa: PLR0913
	source_path: str | Path,
	*,
	survey_id: str,
	grid_order: Sequence[str] = GRID_ORDER_XYZ,
	clip_low_percentile: float = 0.5,
	clip_high_percentile: float = 99.5,
	max_samples: int | None = 1_000_000,
	seed: int = 42,
	eps: float = 1.0e-6,
) -> SurveyNormalizationStats:
	"""Compute robust stats from a memmap-backed `.npy` volume."""
	info = inspect_npy_volume(source_path)
	array = np.load(info.path, mmap_mode='r')
	values = _finite_values(
		_sample_voxels(
			array,
			max_samples=max_samples,
			seed=seed,
		),
		info.path,
	)
	clip_low, clip_high = np.percentile(
		values,
		[clip_low_percentile, clip_high_percentile],
	)
	q25, median, q75 = np.percentile(values, [25.0, 50.0, 75.0])
	stats = SurveyNormalizationStats(
		survey_id=survey_id,
		source_path=info.path,
		grid_order=_as_grid_order(grid_order),
		clip_low_percentile=float(clip_low_percentile),
		clip_high_percentile=float(clip_high_percentile),
		clip_low=float(clip_low),
		clip_high=float(clip_high),
		median=float(median),
		iqr=float(q75 - q25),
		eps=float(eps),
	)
	stats.validate()
	return stats


def _sample_voxels(
	array: np.ndarray,
	*,
	max_samples: int | None,
	seed: int,
) -> np.ndarray:
	total_voxels = int(array.size)
	if max_samples is None or max_samples >= total_voxels:
		return np.asarray(array).reshape(-1)
	if max_samples <= 0:
		msg = f'max_samples must be positive when provided; got {max_samples!r}'
		raise ValueError(msg)
	rng = np.random.default_rng(seed)
	indices = rng.integers(0, total_voxels, size=int(max_samples))
	return np.asarray(array).reshape(-1)[indices]


def _finite_values(values: np.ndarray, path: Path) -> np.ndarray:
	finite = np.asarray(values, dtype=np.float64)
	finite = finite[np.isfinite(finite) & (finite != 0.0)]
	if finite.size == 0:
		msg = (
			'normalization stats cannot be computed from no finite '
			f'non-zero voxels: {path}'
		)
		raise ValueError(msg)
	return finite


def _required_agc_value(data: Mapping[str, object], key: str) -> object:
	if key not in data:
		msg = f'amplitude_agc.{key} is required'
		raise ValueError(msg)
	return data[key]


def _required_agc_int(data: Mapping[str, object], key: str) -> int:
	value = _required_agc_value(data, key)
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'amplitude_agc.{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	return int(value)


def _required_agc_float(data: Mapping[str, object], key: str) -> float:
	value = _required_agc_value(data, key)
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'amplitude_agc.{key} must be a real number; got {value!r}'
		raise TypeError(msg)
	return float(value)


def _validate_positive_odd_int(value: object, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	integer = int(value)
	if integer <= 0:
		msg = f'{name} must be positive; got {integer!r}'
		raise ValueError(msg)
	if integer % 2 == 0:
		msg = f'{name} must be odd; got {integer!r}'
		raise ValueError(msg)
	return integer


def _validate_positive_finite_float(value: object, name: str) -> float:
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{name} must be a real number; got {value!r}'
		raise TypeError(msg)
	number = float(value)
	if not np.isfinite(number) or number <= 0.0:
		msg = f'{name} must be a finite positive number; got {value!r}'
		raise ValueError(msg)
	return number


def _required_str(data: Mapping[str, object], key: str, label: str) -> str:
	value = data.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{label} must define non-empty string {key!r}'
		raise TypeError(msg)
	return value


def _required_float(data: Mapping[str, object], key: str, label: str) -> float:
	value = data.get(key)
	if isinstance(value, bool) or not isinstance(value, int | float):
		msg = f'{label} must define numeric {key!r}'
		raise TypeError(msg)
	return float(value)


def _required_grid_order(
	data: Mapping[str, object],
	key: str,
	label: str,
) -> tuple[str, str, str]:
	value = data.get(key)
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str)
		or len(value) != 3
		or not all(isinstance(item, str) for item in value)
	):
		msg = f'{label} must define length-3 string sequence {key!r}'
		raise TypeError(msg)
	return _as_grid_order(cast('Sequence[str]', value))


def _as_grid_order(value: Sequence[str]) -> tuple[str, str, str]:
	if (
		isinstance(value, str)
		or len(value) != 3
		or not all(isinstance(item, str) for item in value)
	):
		msg = f'grid_order must be a length-3 string sequence; got {value!r}'
		raise TypeError(msg)
	return cast('tuple[str, str, str]', tuple(value))


__all__ = [
	'AmplitudeAgcConfig',
	'SurveyNormalizationStats',
	'apply_configured_agc',
	'apply_trace_rms_agc',
	'compute_normalization_stats',
	'load_normalization_stats',
	'normalize_amplitude',
	'write_normalization_stats',
]
