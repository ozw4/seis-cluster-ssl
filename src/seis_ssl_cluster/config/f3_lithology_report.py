"""F3 lithology report config validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.config.common import _validate_distinct_paths
from seis_ssl_cluster.config.f3_baselines import (
	_explicit_comparison_paths,
	_validate_f3_dataset_name,
)
from seis_ssl_cluster.config.f3_lithology_common import (
	_optional_absolute_path,
	_optional_mapping,
	_required_absolute_path,
	_required_mapping,
	_validate_allowed_keys,
	_validate_output_not_under_f3_root,
)
from seis_ssl_cluster.f3 import (
	F3LithologyComparisonReportConfig,
	F3LithologyReportConfig,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


def f3_lithology_report_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyReportConfig:
	"""Validate and normalize the F3 lithology report config."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'model',
				'labels',
				'lithology',
				'probe',
				'predictions',
				'visualizations',
				'reports',
				'comparison',
				'publish',
			},
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	dataset = _required_mapping(config, 'dataset')
	_validate_f3_dataset_name(dataset)
	model = _required_mapping(config, 'model')
	labels = _required_mapping(config, 'labels')
	lithology = _required_mapping(config, 'lithology')
	probe = _required_mapping(config, 'probe')
	predictions = _optional_mapping(config, 'predictions')
	visualizations = _optional_mapping(config, 'visualizations')
	reports = _required_mapping(config, 'reports')
	metrics_json = _required_absolute_path(probe, 'metrics_json', prefix='probe')
	probe_config_json = _optional_absolute_path(
		probe,
		'probe_config_resolved_json',
		prefix='probe',
	)
	output_dir = _required_absolute_path(reports, 'output_dir', prefix='reports')
	output_markdown = _required_absolute_path(
		reports,
		'output_markdown',
		prefix='reports',
	)
	output_json = _required_absolute_path(reports, 'output_json', prefix='reports')
	prediction_metadata_json = _optional_absolute_path(
		predictions,
		'metadata_json',
		prefix='predictions',
	)
	visualization_metadata_json = _optional_absolute_path(
		visualizations,
		'metadata_json',
		prefix='visualizations',
	)
	token_dataset_metadata_json = _optional_absolute_path(
		reports,
		'token_dataset_metadata_json',
		prefix='reports',
	)
	comparison_mapping = _optional_mapping(config, 'comparison')
	comparison = (
		_embedded_comparison_config(comparison_mapping)
		if comparison_mapping
		else None
	)
	_validate_report_file_paths(
		metrics_json=metrics_json,
		probe_config_json=probe_config_json,
		token_dataset_metadata_json=token_dataset_metadata_json,
		prediction_metadata_json=prediction_metadata_json,
		visualization_metadata_json=visualization_metadata_json,
		output_markdown=output_markdown,
		output_json=output_json,
		comparison=comparison,
	)
	for label, path in _report_output_paths(
		output_dir=output_dir,
		output_markdown=output_markdown,
		output_json=output_json,
		comparison=comparison,
	):
		_validate_output_not_under_f3_root(
			path,
			label,
			f3_root=f3_root,
		)
	return F3LithologyReportConfig(
		output_dir=output_dir,
		output_markdown=output_markdown,
		output_json=output_json,
		metrics_json=metrics_json,
		probe_config_json=probe_config_json,
		token_dataset_metadata_json=token_dataset_metadata_json,
		prediction_metadata_json=prediction_metadata_json,
		visualization_metadata_json=visualization_metadata_json,
		dataset=dataset,
		model=model,
		labels=labels,
		lithology=lithology,
		probe=probe,
		comparison=comparison,
	)


def _validate_report_file_paths(  # noqa: PLR0913
	*,
	metrics_json: Path,
	probe_config_json: Path | None,
	token_dataset_metadata_json: Path | None,
	prediction_metadata_json: Path | None,
	visualization_metadata_json: Path | None,
	output_markdown: Path,
	output_json: Path,
	comparison: F3LithologyComparisonReportConfig | None,
) -> None:
	source_files = (
		(metrics_json, 'probe.metrics_json'),
		(probe_config_json, 'probe.probe_config_resolved_json'),
		(token_dataset_metadata_json, 'reports.token_dataset_metadata_json'),
		(prediction_metadata_json, 'predictions.metadata_json'),
		(visualization_metadata_json, 'visualizations.metadata_json'),
	)
	output_files = (
		(output_markdown, 'reports.output_markdown'),
		(output_json, 'reports.output_json'),
	)
	if comparison is not None:
		output_files += (
			(comparison.output_csv, 'comparison.output_csv'),
			(comparison.output_markdown, 'comparison.output_markdown'),
		)
	for output, output_label in output_files:
		for source, source_label in source_files:
			if source is not None:
				_validate_distinct_paths(output, output_label, source, source_label)
	for index, (left, left_label) in enumerate(output_files):
		for right, right_label in output_files[index + 1 :]:
			_validate_distinct_paths(left, left_label, right, right_label)


def _report_output_paths(
	*,
	output_dir: Path,
	output_markdown: Path,
	output_json: Path,
	comparison: F3LithologyComparisonReportConfig | None,
) -> tuple[tuple[str, Path], ...]:
	output_paths = (
		('reports.output_dir', output_dir),
		('reports.output_markdown', output_markdown),
		('reports.output_json', output_json),
	)
	if comparison is not None:
		output_paths += (
			('comparison.output_csv', comparison.output_csv),
			('comparison.output_markdown', comparison.output_markdown),
		)
	return output_paths


def _embedded_comparison_config(
	comparison: Mapping[str, object],
) -> F3LithologyComparisonReportConfig:
	search_root, output_csv, output_markdown = _explicit_comparison_paths(
		comparison,
	)
	return F3LithologyComparisonReportConfig(
		search_root=search_root,
		output_csv=output_csv,
		output_markdown=output_markdown,
	)


__all__ = ['f3_lithology_report_config_from_mapping']
