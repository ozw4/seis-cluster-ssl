"""F3 lithology report config validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.config.f3_lithology_common import (
	_optional_absolute_path,
	_optional_mapping,
	_optional_str,
	_required_absolute_path,
	_required_mapping,
	_validate_allowed_keys,
	_validate_artifact_path_not_f3,
)
from seis_ssl_cluster.f3 import (
	F3LithologyComparisonReportConfig,
	F3LithologyReportConfig,
)
from seis_ssl_cluster.paths import ArtifactPaths, ExperimentKey

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
	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	dataset = _required_mapping(config, 'dataset')
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
	comparison = _embedded_comparison_config(
		_optional_mapping(config, 'comparison'),
		artifact_root=artifact_root,
		dataset=dataset,
	)
	for label, path in _report_paths(
		metrics_json=metrics_json,
		probe_config_json=probe_config_json,
		token_dataset_metadata_json=token_dataset_metadata_json,
		prediction_metadata_json=prediction_metadata_json,
		visualization_metadata_json=visualization_metadata_json,
		output_dir=output_dir,
		output_markdown=output_markdown,
		output_json=output_json,
		comparison=comparison,
	):
		_validate_artifact_path_not_f3(
			path,
			label,
			artifact_root=artifact_root,
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


def _embedded_comparison_config(
	comparison: Mapping[str, object],
	*,
	artifact_root: Path,
	dataset: Mapping[str, object],
) -> F3LithologyComparisonReportConfig:
	version = _optional_str(
		dataset,
		'version',
		default='facies_benchmark_v1',
		prefix='dataset',
	)
	default_key = ExperimentKey(dataset='f3', version=version)
	default_paths = ArtifactPaths(artifact_root)
	default_search_root = default_paths.lithology_dataset(default_key)
	default_output_dir = default_paths.baseline_comparison_report(default_key)
	search_root = _optional_absolute_path(
		comparison,
		'search_root',
		prefix='comparison',
		default=default_search_root,
	)
	output_dir = _optional_absolute_path(
		comparison,
		'output_dir',
		prefix='comparison',
		default=default_output_dir,
	)
	return F3LithologyComparisonReportConfig(
		search_root=search_root,
		output_csv=_optional_absolute_path(
			comparison,
			'output_csv',
			prefix='comparison',
			default=output_dir / 'comparison_table.csv',
		),
		output_markdown=_optional_absolute_path(
			comparison,
			'output_markdown',
			prefix='comparison',
			default=output_dir / 'comparison_report.md',
		),
	)


def _report_paths(  # noqa: PLR0913
	*,
	metrics_json: Path,
	probe_config_json: Path | None,
	token_dataset_metadata_json: Path | None,
	prediction_metadata_json: Path | None,
	visualization_metadata_json: Path | None,
	output_dir: Path,
	output_markdown: Path,
	output_json: Path,
	comparison: F3LithologyComparisonReportConfig,
) -> tuple[tuple[str, Path], ...]:
	paths = [
		('probe.metrics_json', metrics_json),
		('reports.output_dir', output_dir),
		('reports.output_markdown', output_markdown),
		('reports.output_json', output_json),
		('comparison.search_root', comparison.search_root),
		('comparison.output_csv', comparison.output_csv),
		('comparison.output_markdown', comparison.output_markdown),
	]
	optional_paths = (
		('probe.probe_config_resolved_json', probe_config_json),
		('reports.token_dataset_metadata_json', token_dataset_metadata_json),
		('predictions.metadata_json', prediction_metadata_json),
		('visualizations.metadata_json', visualization_metadata_json),
	)
	paths.extend((label, path) for label, path in optional_paths if path is not None)
	return tuple(paths)


__all__ = ['f3_lithology_report_config_from_mapping']
