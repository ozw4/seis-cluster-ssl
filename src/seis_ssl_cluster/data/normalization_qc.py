"""Quality-control checks for survey normalization statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
	from collections.abc import Iterable

from seis_ssl_cluster.data.normalization import (
	SurveyNormalizationStats,
	load_normalization_stats,
)
from seis_ssl_cluster.data.schema import GRID_ORDER_XYZ

QC_SCHEMA_VERSION = 1

_EXCLUDE_REASON_ORDER = (
	'missing_source',
	'missing_stats',
	'invalid_stats',
	'source_mismatch',
	'grid_order_mismatch',
	'non_finite_stats',
	'small_iqr',
	'large_norm_abs_max',
)
_FINITE_STAT_FIELDS = ('clip_low', 'clip_high', 'median', 'iqr', 'eps')
_STATS_FILENAME_SUFFIX = '.normalization_stats.json'


@dataclass(frozen=True)
class NormalizationStatsQcThresholds:
	"""Thresholds used to exclude suspect normalization statistics."""

	min_iqr: float = 1.0e-4
	max_normalized_abs: float = 1.0e6


@dataclass(frozen=True)
class NormalizationStatsQcItem:
	"""QC outcome for one survey normalization stats file."""

	survey_id: str
	stats_path: Path
	source_path: Path | None
	status: Literal['pass', 'exclude']
	exclude_reasons: tuple[str, ...]
	non_finite_fields: tuple[str, ...]
	iqr: float | None
	normalized_abs_max: float | None
	error: str | None = None


@dataclass(frozen=True)
class NormalizationStatsQcReport:
	"""QC report for a collection of survey normalization stats."""

	schema_version: int
	thresholds: NormalizationStatsQcThresholds
	items: tuple[NormalizationStatsQcItem, ...]

	def __post_init__(self) -> None:
		"""Keep report item order stable for deterministic downstream JSON."""
		items = tuple(
			sorted(self.items, key=lambda item: (item.survey_id, str(item.stats_path)))
		)
		object.__setattr__(self, 'items', items)


def evaluate_normalization_stats(  # noqa: PLR0913
	stats: SurveyNormalizationStats,
	*,
	stats_path: str | Path,
	thresholds: NormalizationStatsQcThresholds,
	expected_survey_id: str | None = None,
	expected_source_path: str | Path | None = None,
	expected_grid_order: tuple[str, str, str] = GRID_ORDER_XYZ,
) -> NormalizationStatsQcItem:
	"""Evaluate in-memory survey normalization stats against QC thresholds."""
	return _evaluate_loaded_normalization_stats(
		stats,
		stats_path=Path(stats_path),
		source_path=(
			None if expected_source_path is None else Path(expected_source_path)
		),
		survey_id=expected_survey_id,
		expected_grid_order=expected_grid_order,
		thresholds=thresholds,
	)


def evaluate_normalization_stats_file(
	stats_path: str | Path,
	*,
	survey_id: str | None = None,
	source_path: str | Path | None = None,
	grid_order: tuple[str, str, str] = GRID_ORDER_XYZ,
	thresholds: NormalizationStatsQcThresholds,
) -> NormalizationStatsQcItem:
	"""Load and evaluate a stats JSON file without raising on QC failures."""
	path = Path(stats_path)
	resolved_source_path = None if source_path is None else Path(source_path)
	fallback_survey_id = survey_id or _survey_id_from_stats_path(path)

	missing_source = (
		resolved_source_path is not None and not resolved_source_path.is_file()
	)
	missing_stats = not path.exists()
	if missing_source or missing_stats:
		reasons = []
		errors = []
		if missing_source:
			reasons.append('missing_source')
			errors.append(
				f'source amplitude file does not exist: {resolved_source_path}',
			)
		if missing_stats:
			reasons.append('missing_stats')
			errors.append(f'normalization stats file does not exist: {path}')
		return _excluded_item(
			survey_id=fallback_survey_id,
			stats_path=path,
			source_path=resolved_source_path,
			exclude_reasons=tuple(reasons),
			error='; '.join(errors),
		)

	try:
		stats = load_normalization_stats(path)
	except Exception as exc:  # noqa: BLE001 - QC reports invalid files.
		return _excluded_item(
			survey_id=fallback_survey_id,
			stats_path=path,
			source_path=resolved_source_path,
			exclude_reasons=('invalid_stats',),
			error=str(exc),
		)

	return _evaluate_loaded_normalization_stats(
		stats,
		stats_path=path,
		source_path=resolved_source_path or stats.source_path,
		survey_id=survey_id,
		expected_grid_order=grid_order,
		thresholds=thresholds,
	)


def normalization_qc_report_to_dict(
	report: NormalizationStatsQcReport,
	*,
	source_manifest_path: str | Path | None = None,
	source_split_path: str | Path | None = None,
) -> dict[str, object]:
	"""Convert a normalization QC report to a strict-JSON-compatible dict."""
	accepted_count = sum(item.status == 'pass' for item in report.items)
	excluded_count = sum(item.status == 'exclude' for item in report.items)
	counts: dict[str, int] = {
		'total': len(report.items),
		'passed': accepted_count,
		'excluded': excluded_count,
	}
	for reason in _EXCLUDE_REASON_ORDER:
		counts[reason] = sum(reason in item.exclude_reasons for item in report.items)

	excluded_surveys = [
		item.survey_id for item in report.items if item.status == 'exclude'
	]
	return {
		'schema_version': report.schema_version,
		'thresholds': {
			'min_iqr': _json_float_or_none(report.thresholds.min_iqr),
			'max_normalized_abs': _json_float_or_none(
				report.thresholds.max_normalized_abs,
			),
		},
		'entry_count': len(report.items),
		'accepted_count': accepted_count,
		'excluded_count': excluded_count,
		'counts': counts,
		'iqr_summary': _numeric_summary(item.iqr for item in report.items),
		'normalized_abs_max_summary': _numeric_summary(
			item.normalized_abs_max for item in report.items
		),
		'excluded_surveys': excluded_surveys,
		'per_survey_reason_codes': {
			item.survey_id: list(item.exclude_reasons) for item in report.items
		},
		'source_manifest_path': (
			None if source_manifest_path is None else str(source_manifest_path)
		),
		'source_split_path': (
			None if source_split_path is None else str(source_split_path)
		),
		'surveys': [_qc_item_to_dict(item) for item in report.items],
	}


def _evaluate_loaded_normalization_stats(  # noqa: PLR0913
	stats: SurveyNormalizationStats,
	*,
	stats_path: Path,
	source_path: Path | None,
	survey_id: str | None,
	expected_grid_order: tuple[str, str, str],
	thresholds: NormalizationStatsQcThresholds,
) -> NormalizationStatsQcItem:
	non_finite_fields = [
		field
		for field in _FINITE_STAT_FIELDS
		if not math.isfinite(float(getattr(stats, field)))
	]
	iqr = stats.iqr if math.isfinite(stats.iqr) else None
	denominator = stats.iqr + stats.eps
	normalized_abs_max = _compute_normalized_abs_max(stats, denominator=denominator)
	if (
		normalized_abs_max is None
		and math.isfinite(denominator)
		and denominator > 0.0
	):
		non_finite_fields.append('normalized_abs_max')

	reasons: list[str] = []
	if survey_id is not None and stats.survey_id != survey_id:
		reasons.append('source_mismatch')
	if source_path is not None and not _same_path(stats.source_path, source_path):
		reasons.append('source_mismatch')
	if stats.grid_order != expected_grid_order:
		reasons.append('grid_order_mismatch')
	if non_finite_fields or not math.isfinite(denominator) or denominator <= 0.0:
		reasons.append('non_finite_stats')
	elif stats.iqr < thresholds.min_iqr:
		reasons.append('small_iqr')

	if normalized_abs_max is None:
		if 'non_finite_stats' not in reasons:
			reasons.append('non_finite_stats')
	elif normalized_abs_max > thresholds.max_normalized_abs:
		reasons.append('large_norm_abs_max')

	exclude_reasons = tuple(
		reason for reason in _EXCLUDE_REASON_ORDER if reason in reasons
	)
	return NormalizationStatsQcItem(
		survey_id=survey_id or stats.survey_id,
		stats_path=stats_path,
		source_path=source_path,
		status='exclude' if exclude_reasons else 'pass',
		exclude_reasons=exclude_reasons,
		non_finite_fields=tuple(non_finite_fields),
		iqr=iqr,
		normalized_abs_max=normalized_abs_max,
		error=None,
	)


def _compute_normalized_abs_max(
	stats: SurveyNormalizationStats,
	*,
	denominator: float,
) -> float | None:
	if not math.isfinite(denominator) or denominator <= 0.0:
		return None
	normalized_abs_max = max(
		abs((stats.clip_low - stats.median) / denominator),
		abs((stats.clip_high - stats.median) / denominator),
	)
	if not math.isfinite(normalized_abs_max):
		return None
	return normalized_abs_max


def _excluded_item(
	*,
	survey_id: str,
	stats_path: Path,
	source_path: Path | None,
	exclude_reasons: tuple[str, ...],
	error: str,
) -> NormalizationStatsQcItem:
	return NormalizationStatsQcItem(
		survey_id=survey_id,
		stats_path=stats_path,
		source_path=source_path,
		status='exclude',
		exclude_reasons=exclude_reasons,
		non_finite_fields=(),
		iqr=None,
		normalized_abs_max=None,
		error=error,
	)


def _survey_id_from_stats_path(path: Path) -> str:
	name = path.name
	if name.endswith(_STATS_FILENAME_SUFFIX):
		return name[: -len(_STATS_FILENAME_SUFFIX)]
	return path.stem


def _qc_item_to_dict(item: NormalizationStatsQcItem) -> dict[str, object]:
	return {
		'survey_id': item.survey_id,
		'status': item.status,
		'exclude_reasons': list(item.exclude_reasons),
		'stats_path': str(item.stats_path),
		'source_path': None if item.source_path is None else str(item.source_path),
		'iqr': _json_float_or_none(item.iqr),
		'normalized_abs_max': _json_float_or_none(item.normalized_abs_max),
		'non_finite_fields': list(item.non_finite_fields),
		'error': item.error,
	}


def _numeric_summary(values: Iterable[float | None]) -> dict[str, object]:
	finite = sorted(
		float(value)
		for value in values
		if value is not None and math.isfinite(float(value))
	)
	if not finite:
		return {'count': 0, 'min': None, 'median': None, 'max': None}
	return {
		'count': len(finite),
		'min': finite[0],
		'median': float(median(finite)),
		'max': finite[-1],
	}


def _json_float_or_none(value: float | None) -> float | None:
	if value is None or not math.isfinite(value):
		return None
	return float(value)


def _same_path(left: Path, right: Path) -> bool:
	return left.resolve(strict=False) == right.resolve(strict=False)


__all__ = [
	'QC_SCHEMA_VERSION',
	'NormalizationStatsQcItem',
	'NormalizationStatsQcReport',
	'NormalizationStatsQcThresholds',
	'evaluate_normalization_stats',
	'evaluate_normalization_stats_file',
	'normalization_qc_report_to_dict',
]
