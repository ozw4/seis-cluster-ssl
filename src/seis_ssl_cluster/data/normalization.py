"""Survey-wise robust normalization for amplitude-only volumes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
) -> np.ndarray:
	"""Clip and robust-scale an amplitude crop without changing XYZ order."""
	stats.validate()
	amplitude = np.asarray(crop, dtype=np.float32)
	clipped = np.clip(amplitude, stats.clip_low, stats.clip_high)
	normalized = (clipped - stats.median) / (stats.iqr + stats.eps)
	return normalized.astype(np.float32, copy=False)


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
	'SurveyNormalizationStats',
	'compute_normalization_stats',
	'load_normalization_stats',
	'normalize_amplitude',
	'write_normalization_stats',
]
