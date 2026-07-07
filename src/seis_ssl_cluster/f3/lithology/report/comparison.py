"""Baseline comparison report helpers for F3 lithology reports."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.f3.lithology.report._common import (
	_BASELINE_FEATURE_KINDS,
	_COMPARISON_FEATURE_KIND_ORDER,
	COMPARISON_ID_COLUMNS,
	OVERALL_METRIC_COLUMNS,
	_class_metric_sort_key,
	_embed_spec_from_config,
	_first_non_empty,
	_float_or_none,
	_mapping,
	_run_parts,
	_string_or_none,
	_write_text,
)
from seis_ssl_cluster.f3.lithology.report.figures import (
	F3LithologyComparisonFigureFontSizes,
	F3LithologyComparisonFigureSizes,
	F3LithologyComparisonFigureStyle,
	_comparison_figure_paths,
	_write_comparison_figures,
	default_f3_lithology_comparison_figure_style,
)
from seis_ssl_cluster.f3.lithology.report.markdown import (
	_best_baseline_class_f1,
	_best_comparison_row,
	_comparison_delta_sentence,
	_comparison_interpretation,
	_depth_only_sentence,
	_metric_delta,
	_render_comparison_markdown,
	_weak_class_delta_sentence,
)
from seis_ssl_cluster.f3.lithology.report.metrics_loader import (
	_load_probe_token_datasets,
	_read_json_component,
	_read_optional_json,
)
from seis_ssl_cluster.f3.lithology.report.publish import (
	F3LithologyComparisonPublishConfig,
	publish_f3_lithology_comparison_report,
)

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

	from seis_ssl_cluster.f3.lithology.token_dataset import F3LithologyTokenDataset
	from seis_ssl_cluster.results import PublishManifest


@dataclass(frozen=True)
class F3LithologyComparisonReportConfig:
	"""Input and output paths for a multi-run lithology comparison report."""

	search_root: Path
	output_csv: Path
	output_markdown: Path
	metrics_paths: tuple[Path, ...] = ()
	figure_dpi: int = 300
	figure_style: F3LithologyComparisonFigureStyle = field(
		default_factory=default_f3_lithology_comparison_figure_style,
	)


@dataclass(frozen=True)
class F3LithologyComparisonReportResult:
	"""Paths and rows written by a lithology comparison report."""

	comparison_csv: Path
	comparison_markdown: Path
	figure_paths: tuple[Path, ...]
	rows: tuple[dict[str, object], ...]
	warnings: tuple[str, ...]
	publish_manifest: PublishManifest | None = None


def build_f3_lithology_comparison_report(
	config: F3LithologyComparisonReportConfig,
	*,
	publish_config: F3LithologyComparisonPublishConfig | None = None,
) -> F3LithologyComparisonReportResult:
	"""Aggregate probe metrics into comparison CSV and Markdown reports."""
	warnings: list[str] = []
	rows: list[dict[str, object]] = []
	for metrics_path in _comparison_metrics_paths(config):
		metrics = _read_json_component(
			'comparison_metrics',
			metrics_path,
			warnings,
		)
		if metrics is None:
			continue
		warnings.extend(_comparison_metric_warnings(metrics_path, metrics))
		probe_config = _read_optional_json(metrics_path.with_name(
			'probe_config_resolved.json',
		))
		token_metadata = _read_optional_json(
			_token_metadata_path_for_metrics(metrics_path, _mapping(probe_config)),
		)
		token_datasets, token_dataset_warnings = _load_probe_token_datasets(
			_mapping(probe_config),
		)
		warnings.extend(token_dataset_warnings)
		rows.append(
			_comparison_row(
				metrics_path,
				metrics,
				probe_config,
				token_metadata,
				token_datasets,
			),
		)
	rows = sorted(
		rows,
		key=lambda row: (
			_COMPARISON_FEATURE_KIND_ORDER.get(str(row.get('feature_kind')), 99),
			str(row.get('BASELINE_TAG', '')),
			str(row.get('MODEL_TAG', '')),
			str(row.get('EMBED_SPEC', '')),
			str(row.get('LABEL_SET', '')),
			str(row.get('PROBE_SPEC', '')),
		),
	)
	fieldnames = _comparison_fieldnames(rows)
	figure_paths = _comparison_figure_paths(config.output_markdown)
	warnings.extend(
		_write_comparison_figures(
			rows,
			figure_paths,
			dpi=max(config.figure_dpi, 300),
			style=config.figure_style,
		),
	)
	_write_comparison_csv(config.output_csv, rows, fieldnames)
	_write_text(
		config.output_markdown,
		_render_comparison_markdown(rows, fieldnames, figure_paths, warnings),
	)
	publish_manifest = publish_f3_lithology_comparison_report(
		config,
		publish_config,
	)
	return F3LithologyComparisonReportResult(
		comparison_csv=config.output_csv,
		comparison_markdown=config.output_markdown,
		figure_paths=tuple(figure_paths.values()),
		rows=tuple(rows),
		warnings=tuple(warnings),
		publish_manifest=publish_manifest,
	)

def _comparison_payload(
	comparison: F3LithologyComparisonReportResult,
) -> dict[str, object]:
	return {
		'comparison_table_csv': str(comparison.comparison_csv),
		'comparison_report_markdown': str(comparison.comparison_markdown),
		'figures': [str(path) for path in comparison.figure_paths],
		'row_count': len(comparison.rows),
		'warnings': list(comparison.warnings),
	}

def _comparison_metrics_paths(
	config: F3LithologyComparisonReportConfig,
) -> tuple[Path, ...]:
	if config.metrics_paths:
		return tuple(config.metrics_paths)
	return tuple(sorted(config.search_root.glob('**/probes/*/metrics.json')))

def _comparison_row(
	metrics_path: Path,
	metrics: Mapping[str, object],
	probe_config: Mapping[str, object] | None,
	token_metadata: Mapping[str, object] | None,
	token_datasets: Mapping[str, F3LithologyTokenDataset] | None = None,
) -> dict[str, object]:
	config = _mapping(probe_config)
	model = _mapping(config.get('model'))
	labels = _mapping(config.get('labels'))
	probe = _mapping(config.get('probe'))
	path_parts = _run_parts(metrics_path)
	feature_source = _feature_source_summary(
		metrics,
		config,
		token_metadata,
		token_datasets,
	)
	model_tag = _first_non_empty(model.get('tag'), path_parts.get('MODEL_TAG'))
	feature_kind = _feature_kind(
		feature_source=feature_source,
		model_tag=model_tag,
		path_parts=path_parts,
	)
	baseline_tag = _baseline_tag(
		feature_kind=feature_kind,
		model_tag=model_tag,
		path_parts=path_parts,
		feature_source=feature_source,
	)
	embed_spec = _first_non_empty(
		_embed_spec_from_config(config),
		path_parts.get('EMBED_SPEC'),
	)
	row: dict[str, object] = {
		'feature_kind': feature_kind,
		'MODEL_TAG': '' if feature_kind in _BASELINE_FEATURE_KINDS else model_tag,
		'BASELINE_TAG': baseline_tag or '',
		'EMBED_SPEC': embed_spec,
		'LABEL_SET': _first_non_empty(labels.get('set'), path_parts.get('LABEL_SET')),
		'PROBE_SPEC': _first_non_empty(probe.get('spec'), path_parts.get(
			'PROBE_SPEC',
		)),
		'FEATURE_SOURCE_KIND': _first_non_empty(
			feature_source.get('kind'),
			feature_kind,
		),
		'FEATURE_SOURCE_REFERENCE_MODEL_TAG': _first_non_empty(
			feature_source.get('reference_model_tag'),
			'',
		),
		'FEATURE_SOURCE_EMBED_SPEC': _first_non_empty(
			feature_source.get('embedding_spec'),
			embed_spec,
		),
		'FEATURE_SOURCE_DESCRIPTION': _first_non_empty(
			feature_source.get('description'),
			'',
		),
		'_class_names': dict(_mapping(metrics.get('class_names'))),
	}
	for metric in OVERALL_METRIC_COLUMNS:
		row[metric] = _float_or_none(metrics.get(metric))
	for class_id, value in _mapping(metrics.get('per_class_f1')).items():
		row[f'class_{class_id}_f1'] = _float_or_none(value)
	return row

def _comparison_fieldnames(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
	class_columns = sorted(
		{
			key
			for row in rows
			for key in row
			if key.startswith('class_') and key.endswith('_f1')
		},
		key=_class_metric_sort_key,
	)
	return (
		*COMPARISON_ID_COLUMNS,
		*OVERALL_METRIC_COLUMNS,
		*class_columns,
	)

def _token_metadata_path_for_metrics(
	metrics_path: Path,
	probe_config: Mapping[str, object],
) -> Path:
	value = _mapping(probe_config.get('inputs')).get('token_dataset_metadata_json')
	if isinstance(value, str) and value:
		return Path(value)
	return (
		metrics_path.parent.parent.parent
		/ 'token_dataset'
		/ 'token_dataset_metadata.json'
	)

def _comparison_metric_warnings(
	metrics_path: Path,
	metrics: Mapping[str, object],
) -> list[str]:
	missing = [
		key
		for key in (*OVERALL_METRIC_COLUMNS, 'per_class_f1')
		if key not in metrics
	]
	if not missing:
		return []
	return [
		(
			'comparison metrics missing key(s): '
			f'{", ".join(missing)} ({metrics_path})'
		),
	]

def _feature_source_summary(
	metrics: Mapping[str, object],
	probe_config: Mapping[str, object],
	token_metadata: Mapping[str, object] | None,
	token_datasets: Mapping[str, F3LithologyTokenDataset] | None = None,
) -> Mapping[str, object]:
	for candidate in (
		_mapping(metrics.get('feature_source')),
		_mapping(probe_config.get('feature_source')),
		_mapping(_mapping(probe_config.get('token_dataset')).get('feature_source')),
		_mapping(_mapping(probe_config.get('embeddings')).get('feature_source')),
		_mapping(_mapping(probe_config.get('model')).get('feature_source')),
		_mapping(_mapping(token_metadata).get('feature_source')),
		_token_dataset_feature_source(token_datasets),
	):
		if candidate:
			return candidate
	return {}

def _token_dataset_feature_source(
	token_datasets: Mapping[str, F3LithologyTokenDataset] | None,
) -> Mapping[str, object]:
	if not token_datasets:
		return {}
	for split in ('train', 'validation'):
		dataset = token_datasets.get(split)
		if dataset is None:
			continue
		feature_source = _mapping(dataset.metadata.get('feature_source'))
		if feature_source:
			return feature_source
	return {}

def _feature_kind(
	*,
	feature_source: Mapping[str, object],
	model_tag: object,
	path_parts: Mapping[str, str],
) -> str:
	kind = _string_or_none(feature_source.get('kind'))
	if kind is not None:
		return kind
	baseline_tag = path_parts.get('BASELINE_TAG')
	model = _string_or_none(model_tag)
	for candidate in (baseline_tag, model):
		if candidate is None:
			continue
		if candidate.startswith('z_only'):
			return 'z_only'
		if candidate.startswith('xyz_coordinates'):
			return 'xyz_coordinates'
		if candidate.startswith('amplitude_stats'):
			return 'amplitude_stats'
		if candidate.startswith('random_encoder'):
			return 'random_encoder'
	return 'pretrained_encoder'

def _baseline_tag(
	*,
	feature_kind: str,
	model_tag: object,
	path_parts: Mapping[str, str],
	feature_source: Mapping[str, object],
) -> object:
	if feature_kind not in _BASELINE_FEATURE_KINDS:
		return None
	return _first_non_empty(
		feature_source.get('baseline_tag'),
		path_parts.get('BASELINE_TAG'),
		model_tag,
	)

def _write_comparison_csv(
	path: Path,
	rows: Sequence[Mapping[str, object]],
	fieldnames: Sequence[str],
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=fieldnames, lineterminator='\n')
		writer.writeheader()
		for row in rows:
			writer.writerow({key: row.get(key, '') for key in fieldnames})

__all__ = [
	'COMPARISON_ID_COLUMNS',
	'OVERALL_METRIC_COLUMNS',
	'F3LithologyComparisonFigureFontSizes',
	'F3LithologyComparisonFigureSizes',
	'F3LithologyComparisonFigureStyle',
	'F3LithologyComparisonPublishConfig',
	'F3LithologyComparisonReportConfig',
	'F3LithologyComparisonReportResult',
	'_baseline_tag',
	'_best_baseline_class_f1',
	'_best_comparison_row',
	'_comparison_delta_sentence',
	'_comparison_fieldnames',
	'_comparison_interpretation',
	'_comparison_metric_warnings',
	'_comparison_metrics_paths',
	'_comparison_payload',
	'_comparison_row',
	'_depth_only_sentence',
	'_feature_kind',
	'_feature_source_summary',
	'_metric_delta',
	'_render_comparison_markdown',
	'_run_parts',
	'_token_dataset_feature_source',
	'_token_metadata_path_for_metrics',
	'_weak_class_delta_sentence',
	'build_f3_lithology_comparison_report',
	'default_f3_lithology_comparison_figure_style',
	'publish_f3_lithology_comparison_report',
]
