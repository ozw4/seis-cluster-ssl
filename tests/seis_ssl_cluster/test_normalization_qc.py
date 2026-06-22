from __future__ import annotations

import json
from pathlib import Path

import pytest

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	NormalizationStatsQcReport,
	NormalizationStatsQcThresholds,
	SurveyNormalizationStats,
	evaluate_normalization_stats,
	evaluate_normalization_stats_file,
	normalization_qc_report_to_dict,
	write_normalization_stats,
)


def test_evaluate_normalization_stats_passes_valid_stats() -> None:
	item = evaluate_normalization_stats(
		_stats(),
		stats_path='survey-a.normalization_stats.json',
		thresholds=NormalizationStatsQcThresholds(),
	)

	assert item.status == 'pass'
	assert item.exclude_reasons == ()
	assert item.iqr == pytest.approx(2.0)
	assert item.normalized_abs_max == pytest.approx(2.0)


def test_evaluate_normalization_stats_excludes_small_iqr_by_default() -> None:
	item = evaluate_normalization_stats(
		_stats(clip_low=-1.0e-5, clip_high=1.0e-5, median=0.0, iqr=1.0e-8),
		stats_path='survey-a.normalization_stats.json',
		thresholds=NormalizationStatsQcThresholds(),
	)

	assert item.status == 'exclude'
	assert item.exclude_reasons == ('small_iqr',)


def test_evaluate_stats_file_reports_missing_source_and_missing_stats(
	tmp_path: Path,
) -> None:
	item = evaluate_normalization_stats_file(
		tmp_path / 'missing.normalization_stats.json',
		survey_id='survey-a',
		source_path=tmp_path / 'missing.npy',
		thresholds=NormalizationStatsQcThresholds(),
	)

	assert item.status == 'exclude'
	assert item.exclude_reasons == ('missing_source', 'missing_stats')


def test_evaluate_stats_file_reports_source_mismatch(tmp_path: Path) -> None:
	source = tmp_path / 'survey-a.npy'
	source.write_bytes(b'placeholder')
	stats_path = tmp_path / 'survey-a.normalization_stats.json'
	write_normalization_stats(
		_stats(survey_id='survey-b', source_path=tmp_path / 'survey-b.npy'),
		stats_path,
	)

	item = evaluate_normalization_stats_file(
		stats_path,
		survey_id='survey-a',
		source_path=source,
		thresholds=NormalizationStatsQcThresholds(),
	)

	assert item.status == 'exclude'
	assert item.exclude_reasons == ('source_mismatch',)


def test_normalization_qc_report_to_dict_includes_required_metadata(
	tmp_path: Path,
) -> None:
	valid = evaluate_normalization_stats(
		_stats(survey_id='survey-a'),
		stats_path='survey-a.normalization_stats.json',
		thresholds=NormalizationStatsQcThresholds(),
	)
	invalid = evaluate_normalization_stats(
		_stats(survey_id='survey-b', median=float('nan')),
		stats_path='survey-b.normalization_stats.json',
		thresholds=NormalizationStatsQcThresholds(),
	)
	report = NormalizationStatsQcReport(
		schema_version=1,
		thresholds=NormalizationStatsQcThresholds(),
		items=(invalid, valid),
	)

	report_dict = normalization_qc_report_to_dict(
		report,
		source_manifest_path=tmp_path / 'manifest.json',
		source_split_path=tmp_path / 'train.txt',
	)

	json.dumps(report_dict, allow_nan=False)
	assert report_dict['entry_count'] == 2
	assert report_dict['accepted_count'] == 1
	assert report_dict['excluded_count'] == 1
	assert report_dict['source_manifest_path'] == str(tmp_path / 'manifest.json')
	assert report_dict['source_split_path'] == str(tmp_path / 'train.txt')
	assert report_dict['per_survey_reason_codes']['survey-b'] == [
		'non_finite_stats',
	]
	assert report_dict['iqr_summary']['count'] == 2
	assert report_dict['normalized_abs_max_summary']['count'] == 1


def _stats(  # noqa: PLR0913
	*,
	survey_id: str = 'survey-a',
	source_path: Path | None = None,
	clip_low: float = -2.0,
	clip_high: float = 6.0,
	median: float = 2.0,
	iqr: float = 2.0,
	eps: float = 1.0e-6,
) -> SurveyNormalizationStats:
	return SurveyNormalizationStats(
		survey_id=survey_id,
		source_path=source_path or Path(f'{survey_id}.npy'),
		grid_order=GRID_ORDER_XYZ,
		clip_low_percentile=0.5,
		clip_high_percentile=99.5,
		clip_low=clip_low,
		clip_high=clip_high,
		median=median,
		iqr=iqr,
		eps=eps,
	)
