"""Aggregate F3 lithology probe artifacts into Markdown and CSV reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from seis_ssl_cluster.f3.lithology.report._common import (
	_mapping,
	_write_json,
	_write_text,
)
from seis_ssl_cluster.f3.lithology.report.comparison import (
	F3LithologyComparisonReportConfig,
	_comparison_payload,
	build_f3_lithology_comparison_report,
)
from seis_ssl_cluster.f3.lithology.report.figures import _figure_summary
from seis_ssl_cluster.f3.lithology.report.markdown import (
	_interpretation_summary,
	render_f3_lithology_report_markdown,
)
from seis_ssl_cluster.f3.lithology.report.metrics_loader import (
	_classes,
	_dataset_summary,
	_load_probe_token_datasets,
	_metrics_summary,
	_pretrained_summary,
	_probe_summary,
	_read_json_component,
	_read_optional_component,
	_token_dataset_metadata_path,
	_token_dataset_summary,
)
from seis_ssl_cluster.f3.lithology.report.publish import (
	F3LithologyPublishConfig,
	publish_f3_lithology_report,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


@dataclass(frozen=True)
class F3LithologyReportConfig:
	"""Input and output paths for one F3 lithology probe report."""

	output_dir: Path
	output_markdown: Path
	output_json: Path
	metrics_json: Path
	dataset: Mapping[str, object]
	model: Mapping[str, object]
	labels: Mapping[str, object]
	lithology: Mapping[str, object]
	probe: Mapping[str, object]
	probe_config_json: Path | None = None
	token_dataset_metadata_json: Path | None = None
	prediction_metadata_json: Path | None = None
	visualization_metadata_json: Path | None = None
	comparison: F3LithologyComparisonReportConfig | None = None


@dataclass(frozen=True)
class F3LithologyReportResult:
	"""Paths and payload written by one F3 lithology probe report."""

	report_markdown: Path
	report_json: Path
	payload: dict[str, object]
	comparison_csv: Path | None = None
	comparison_markdown: Path | None = None
	published_files: tuple[Path, ...] = ()


def build_f3_lithology_report(
	config: F3LithologyReportConfig,
	*,
	publish_config: F3LithologyPublishConfig | None = None,
) -> F3LithologyReportResult:
	"""Build one F3 lithology probe report and optional comparison artifacts."""
	payload = _report_payload(config)
	comparison_result = None
	if config.comparison is not None:
		comparison_result = build_f3_lithology_comparison_report(config.comparison)
		payload = dict(payload)
		payload['comparison'] = _comparison_payload(comparison_result)
	_write_json(config.output_json, payload)
	_write_text(config.output_markdown, render_f3_lithology_report_markdown(payload))
	published_files = publish_f3_lithology_report(
		config,
		publish_config,
		payload=payload,
	)
	return F3LithologyReportResult(
		report_markdown=config.output_markdown,
		report_json=config.output_json,
		payload=payload,
		comparison_csv=(
			None if comparison_result is None else comparison_result.comparison_csv
		),
		comparison_markdown=(
			None
			if comparison_result is None
			else comparison_result.comparison_markdown
		),
		published_files=published_files,
	)

def _report_payload(config: F3LithologyReportConfig) -> dict[str, object]:
	warnings: list[str] = []
	metrics = _read_json_component('metrics', config.metrics_json, warnings)
	probe_config_path = (
		config.probe_config_json
		if config.probe_config_json is not None
		else config.metrics_json.with_name('probe_config_resolved.json')
	)
	probe_config = _read_json_component(
		'probe_config_resolved',
		probe_config_path,
		warnings,
	)
	token_metadata_path = _token_dataset_metadata_path(
		config,
		_mapping(probe_config),
	)
	token_metadata = _read_optional_component(
		'token_dataset_metadata',
		token_metadata_path,
		warnings,
	)
	token_datasets, token_dataset_warnings = _load_probe_token_datasets(
		_mapping(probe_config),
	)
	warnings.extend(token_dataset_warnings)
	prediction_metadata = _read_optional_component(
		'prediction_metadata',
		config.prediction_metadata_json,
		warnings,
	)
	visualization_metadata = _read_optional_component(
		'visualization_metadata',
		config.visualization_metadata_json,
		warnings,
	)
	classes = _classes(
		_mapping(probe_config),
		_mapping(token_metadata),
		_mapping(metrics),
	)
	token_dataset = _token_dataset_summary(
		_mapping(probe_config),
		_mapping(token_metadata),
		token_datasets=token_datasets,
	)
	dataset = _dataset_summary(
		config,
		_mapping(token_metadata),
		classes,
		token_dataset,
	)
	if not _mapping(_mapping(dataset.get('class_imbalance')).get('class_counts')):
		warnings.append(
			'dataset class imbalance unavailable: '
			'no token dataset class counts were found',
		)
	pretrained = _pretrained_summary(config, _mapping(probe_config))
	probe = _probe_summary(config, _mapping(probe_config))
	metric_summary, metric_warnings = _metrics_summary(_mapping(metrics), classes)
	figures, figure_warnings = _figure_summary(
		config,
		_mapping(probe_config),
		_mapping(visualization_metadata),
	)
	warnings.extend(metric_warnings)
	warnings.extend(figure_warnings)
	interpretation = _interpretation_summary(
		pretrained=pretrained,
		token_dataset=token_dataset,
		metrics=metric_summary,
	)
	return {
		'artifact_type': 'f3_lithology_probe_report',
		'outputs': {
			'output_dir': str(config.output_dir),
			'markdown': str(config.output_markdown),
			'json': str(config.output_json),
		},
		'inputs': {
			'metrics_json': str(config.metrics_json),
			'probe_config_json': str(probe_config_path),
			'token_dataset_metadata_json': (
				None if token_metadata_path is None else str(token_metadata_path)
			),
			'prediction_metadata_json': (
				None
				if config.prediction_metadata_json is None
				else str(config.prediction_metadata_json)
			),
			'visualization_metadata_json': (
				None
				if config.visualization_metadata_json is None
				else str(config.visualization_metadata_json)
			),
		},
		'warnings': warnings,
		'dataset': dataset,
		'pretrained_encoder': pretrained,
		'token_dataset': token_dataset,
		'probe': probe,
		'metrics': metric_summary,
		'figures': figures,
		'interpretation': interpretation,
		'prediction_summary': _mapping(prediction_metadata).get('summary'),
		'comparison': None,
	}

__all__ = [
	'F3LithologyReportConfig',
	'F3LithologyReportResult',
	'build_f3_lithology_report',
]
