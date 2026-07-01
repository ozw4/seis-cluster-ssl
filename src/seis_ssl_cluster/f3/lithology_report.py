"""Aggregate F3 lithology probe artifacts into Markdown and CSV reports."""

from __future__ import annotations

import csv
import json
import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.results import (
	DEFAULT_MAX_FILE_SIZE_BYTES,
	PublishItem,
	PublishManifest,
	publish_selected_results,
)

OVERALL_METRIC_COLUMNS = (
	'accuracy',
	'balanced_accuracy',
	'macro_f1',
	'weighted_f1',
	'mean_iou',
)
COMPARISON_ID_COLUMNS = (
	'feature_kind',
	'MODEL_TAG',
	'BASELINE_TAG',
	'EMBED_SPEC',
	'LABEL_SET',
	'PROBE_SPEC',
	'FEATURE_SOURCE_KIND',
	'FEATURE_SOURCE_REFERENCE_MODEL_TAG',
	'FEATURE_SOURCE_EMBED_SPEC',
	'FEATURE_SOURCE_DESCRIPTION',
)
COMPARISON_FIGURE_NAMES = (
	'macro_f1_comparison',
	'mean_iou_comparison',
	'per_class_f1_comparison',
)
_COMPARISON_FEATURE_KIND_ORDER = {
	'pretrained_encoder': 0,
	'z_only': 1,
	'amplitude_stats': 2,
	'random_encoder': 3,
}
_BASELINE_FEATURE_KINDS = frozenset(
	{
		'z_only',
		'amplitude_stats',
		'random_encoder',
	},
)
_DEFAULT_PROBE_FIGURES = (
	('confusion_matrix', Path('figures/confusion_matrix.png')),
	('per_class_f1', Path('figures/per_class_f1.png')),
)
_PUBLISH_REPORT_TARGET = Path('report.md')
_PUBLISH_JSON_TARGET = Path('report.json')
_PUBLISH_METRICS_TARGET = Path('metrics.json')
_PUBLISH_METRICS_CSV_TARGET = Path('metrics.csv')
_PUBLISH_CLASSIFICATION_REPORT_TARGET = Path('classification_report.md')
_PUBLISH_CONFUSION_MATRIX_CSV_TARGET = Path('confusion_matrix.csv')
_PUBLISH_FIGURE_DIR = Path('figures')


@dataclass(frozen=True)
class F3LithologyComparisonReportConfig:
	"""Input and output paths for a multi-run lithology comparison report."""

	search_root: Path
	output_csv: Path
	output_markdown: Path
	metrics_paths: tuple[Path, ...] = ()
	figure_dpi: int = 300


@dataclass(frozen=True)
class F3LithologyComparisonPublishConfig:
	"""Settings for publishing a lightweight F3 lithology comparison report."""

	enabled: bool = False
	output_dir: Path | None = None
	include_figures: bool = True
	max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES


@dataclass(frozen=True)
class F3LithologyComparisonReportResult:
	"""Paths and rows written by a lithology comparison report."""

	comparison_csv: Path
	comparison_markdown: Path
	figure_paths: tuple[Path, ...]
	rows: tuple[dict[str, object], ...]
	warnings: tuple[str, ...]
	publish_manifest: PublishManifest | None = None


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
class F3LithologyPublishConfig:
	"""Settings for publishing a lightweight F3 lithology probe report copy."""

	enabled: bool = False
	output_dir: Path | None = None
	include_figures: bool = True
	max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
	max_prediction_figures: int = 3


@dataclass(frozen=True)
class F3LithologyReportResult:
	"""Paths and payload written by one F3 lithology probe report."""

	report_markdown: Path
	report_json: Path
	payload: dict[str, object]
	comparison_csv: Path | None = None
	comparison_markdown: Path | None = None
	publish_manifest: PublishManifest | None = None


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
	publish_manifest = publish_f3_lithology_report(
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
		publish_manifest=publish_manifest,
	)


def publish_f3_lithology_report(
	config: F3LithologyReportConfig,
	publish_config: F3LithologyPublishConfig | None,
	*,
	payload: Mapping[str, object] | None = None,
) -> PublishManifest | None:
	"""Publish lightweight F3 lithology probe report artifacts into ``results/``."""
	if publish_config is None or not publish_config.enabled:
		return None
	if publish_config.output_dir is None:
		msg = 'publish output_dir is required when publishing is enabled'
		raise ValueError(msg)
	_validate_max_prediction_figures(publish_config.max_prediction_figures)
	if payload is None:
		payload = _read_required_json_object(config.output_json, 'publish report')
	return publish_selected_results(
		items=_publish_items_for_f3_lithology_report(
			config,
			publish_config=publish_config,
			payload=payload,
		),
		output_dir=publish_config.output_dir,
		max_file_size_bytes=publish_config.max_file_size_bytes,
	)


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
		rows.append(
			_comparison_row(metrics_path, metrics, probe_config, token_metadata),
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


def publish_f3_lithology_comparison_report(
	config: F3LithologyComparisonReportConfig,
	publish_config: F3LithologyComparisonPublishConfig | None,
) -> PublishManifest | None:
	"""Publish lightweight F3 lithology comparison artifacts into ``results/``."""
	if publish_config is None or not publish_config.enabled:
		return None
	if publish_config.output_dir is None:
		msg = 'publish output_dir is required when publishing is enabled'
		raise ValueError(msg)
	return publish_selected_results(
		items=_publish_items_for_f3_lithology_comparison_report(
			config,
			publish_config=publish_config,
		),
		output_dir=publish_config.output_dir,
		max_file_size_bytes=publish_config.max_file_size_bytes,
	)


def render_f3_lithology_report_markdown(payload: Mapping[str, object]) -> str:
	"""Render a lithology probe report payload as Japanese Markdown."""
	dataset = _mapping(payload.get('dataset'))
	pretrained = _mapping(payload.get('pretrained_encoder'))
	token_dataset = _mapping(payload.get('token_dataset'))
	probe = _mapping(payload.get('probe'))
	metrics = _mapping(payload.get('metrics'))
	figures = _sequence_of_mappings(payload.get('figures'))
	interpretation = _mapping(payload.get('interpretation'))
	warnings = _string_list(payload.get('warnings'))
	lines = [
		'# F3 token-level lithology probe report',
		'',
		'このreportはF3 token-level lithology probeの既存artifactを統合し、'
		'pretrained model、AGC有無、probe種別の比較に使う。',
		'',
		'## Dataset',
		'',
		*_render_dataset(dataset),
		'',
		'## Pretrained encoder',
		'',
		*_render_pretrained(pretrained),
		'',
		'## Token dataset',
		'',
		*_render_token_dataset(token_dataset),
		'',
		'## Probe',
		'',
		*_render_probe(probe),
		'',
		'## Metrics',
		'',
		*_render_metrics(metrics),
		'',
		'## Figures',
		'',
		*_render_figures(figures),
		'',
		'## Interpretation',
		'',
		*_render_interpretation(interpretation),
		'',
		'## Warnings',
		'',
	]
	lines.extend((f'- {warning}' for warning in warnings),)
	if not warnings:
		lines.append('- none')
	return '\n'.join(lines) + '\n'


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
	dataset = _dataset_summary(config, _mapping(token_metadata), classes)
	token_dataset = _token_dataset_summary(
		_mapping(probe_config),
		_mapping(token_metadata),
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


def _dataset_summary(
	config: F3LithologyReportConfig,
	token_metadata: Mapping[str, object],
	classes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
	geometry = _mapping(token_metadata.get('geometry'))
	summary = _mapping(token_metadata.get('summary'))
	return {
		'name': _first_non_empty(
			config.dataset.get('name'),
			_mapping(token_metadata.get('dataset')).get('name'),
		),
		'version': _first_non_empty(
			config.dataset.get('version'),
			_mapping(token_metadata.get('dataset')).get('version'),
		),
		'f3_shape': geometry.get('shape_xyz'),
		'classes': [dict(item) for item in classes],
		'train_validation_slices': _slice_summary(token_metadata),
		'tokenization_thresholds': dict(_mapping(token_metadata.get('tokenization'))),
		'class_imbalance': _class_imbalance(
			_combined_counts(
				_mapping(summary.get('train_class_counts')),
				_mapping(summary.get('validation_class_counts')),
			),
		),
		'label_source_of_truth': _first_non_empty(
			token_metadata.get('label_source_of_truth'),
			'segy_label_volume',
		),
		'png_label_role': _first_non_empty(
			config.labels.get('png_label_role'),
			token_metadata.get('png_label_role'),
		),
	}


def _pretrained_summary(
	config: F3LithologyReportConfig,
	probe_config: Mapping[str, object],
) -> dict[str, object]:
	model = _prefer_mapping(config.model, _mapping(probe_config.get('model')))
	model_tag = _string_or_none(model.get('tag'))
	return {
		'MODEL_TAG': model_tag,
		'checkpoint_path': model.get('checkpoint'),
		'EMBED_SPEC': _embed_spec(config.lithology, probe_config),
		'agc_enabled': _agc_enabled(model),
		'visible_loss_enabled': _visible_loss_enabled(model_tag),
		'mask_ratio': _mask_ratio(model_tag),
		'freeze_encoder': model.get('freeze_encoder'),
	}


def _token_dataset_summary(
	probe_config: Mapping[str, object],
	token_metadata: Mapping[str, object],
) -> dict[str, object]:
	token_summary = _mapping(token_metadata.get('summary'))
	probe_summary = _mapping(probe_config.get('summary'))
	train_counts = _prefer_mapping(
		_mapping(token_summary.get('train_class_counts')),
		_mapping(probe_summary.get('train_class_counts')),
	)
	validation_counts = _prefer_mapping(
		_mapping(token_summary.get('validation_class_counts')),
		_mapping(probe_summary.get('validation_class_counts')),
	)
	retained = _int_or_none(
		_first_non_empty(
			token_summary.get('all_labeled_tokens'),
			_sum_ints((token_summary.get('train_tokens'), token_summary.get(
				'validation_tokens',
			))),
		),
	)
	dropped = _int_or_none(token_summary.get('total_dropped_tokens'))
	ambiguous = _int_or_none(token_summary.get('total_ambiguous_tokens'))
	total = None if retained is None or dropped is None else retained + dropped
	return {
		'train_token_count': _first_non_empty(
			token_summary.get('train_tokens'),
			probe_summary.get('train_tokens'),
		),
		'validation_token_count': _first_non_empty(
			token_summary.get('validation_tokens'),
			probe_summary.get('validation_tokens'),
		),
		'class_counts': {
			'train': dict(train_counts),
			'validation': dict(validation_counts),
			'combined': _combined_counts(train_counts, validation_counts),
		},
		'total_dropped_tokens': dropped,
		'total_ambiguous_tokens': ambiguous,
		'dropped_token_ratio': _fraction_or_none(dropped, total),
		'ambiguous_token_ratio': _fraction_or_none(ambiguous, total),
		'class_imbalance': _class_imbalance(
			_combined_counts(train_counts, validation_counts),
		),
	}


def _probe_summary(
	config: F3LithologyReportConfig,
	probe_config: Mapping[str, object],
) -> dict[str, object]:
	probe = {
		**_mapping(probe_config.get('probe')),
		**config.probe,
	}
	hyperparameters = {
		key: value
		for key, value in probe.items()
		if key
		not in {
			'spec',
			'type',
			'feature_scaling',
			'class_weight',
			'output_dir',
			'metrics_json',
		}
	}
	return {
		'PROBE_SPEC': probe.get('spec'),
		'classifier_type': probe.get('type'),
		'feature_scaling': probe.get('feature_scaling'),
		'class_weighting': probe.get('class_weight'),
		'hyperparameters': hyperparameters,
		'training_summary': dict(_mapping(probe_config.get('training_summary'))),
	}


def _metrics_summary(
	metrics: Mapping[str, object],
	classes: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[str]]:
	if not metrics:
		return {
			'available': False,
			'overall': {},
			'per_class': [],
			'confusion_matrix': None,
			'missing': list(OVERALL_METRIC_COLUMNS),
		}, []
	warnings: list[str] = []
	missing = [
		key
		for key in (
			*OVERALL_METRIC_COLUMNS,
			'per_class_f1',
			'per_class_iou',
			'confusion_matrix',
		)
		if key not in metrics
	]
	if missing:
		warnings.append(f'metrics missing required key(s): {", ".join(missing)}')
	overall = {
		key: _float_or_none(metrics.get(key)) for key in OVERALL_METRIC_COLUMNS
	}
	return {
		'available': True,
		'overall': overall,
		'per_class': _per_class_metrics(metrics, classes),
		'confusion_matrix': metrics.get('confusion_matrix'),
		'missing': missing,
	}, warnings


def _figure_summary(
	config: F3LithologyReportConfig,
	probe_config: Mapping[str, object],
	visualization_metadata: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[str]]:
	warnings: list[str] = []
	figures: list[dict[str, object]] = []
	report_dir = config.output_markdown.parent
	probe_outputs = _mapping(probe_config.get('outputs'))
	for figure_type, relative in _DEFAULT_PROBE_FIGURES:
		raw_path = probe_outputs.get(f'{figure_type}_png')
		source = Path(raw_path) if isinstance(raw_path, str) else (
			config.metrics_json.parent / relative
		)
		figures.append(_figure_record(figure_type, source, report_dir, warnings))
	for item in _sequence_of_mappings(visualization_metadata.get('figures')):
		path = item.get('path')
		if not isinstance(path, str) or not path:
			continue
		figure_type = (
			f"validation_slice_{item.get('slice_type')}_{item.get('slice_index')}"
		)
		figures.append(
			_figure_record(figure_type, Path(path), report_dir, warnings),
		)
	return figures, warnings


def _interpretation_summary(
	*,
	pretrained: Mapping[str, object],
	token_dataset: Mapping[str, object],
	metrics: Mapping[str, object],
) -> dict[str, object]:
	overall = _mapping(metrics.get('overall'))
	per_class = _sequence_of_mappings(metrics.get('per_class'))
	failures = [
		item
		for item in sorted(
			per_class,
			key=lambda entry: (
				float('inf')
				if _float_or_none(entry.get('f1')) is None
				else float(entry['f1'])
			),
		)
		if _float_or_none(item.get('f1')) is not None
	][:2]
	good_points = [
		(
			'weighted F1は'
			f"{_display(overall.get('weighted_f1'))}で、頻出classの性能を確認できる。"
		),
		(
			'balanced accuracyは'
			f"{_display(overall.get('balanced_accuracy'))}で、"
			'class imbalanceを考慮した比較指標になる。'
		),
	]
	return {
		'良い点': good_points,
		'失敗しているclass': [
			(
				f"class {item.get('class_id')} {item.get('class_name')}: "
				f"F1={_display(item.get('f1'))}, IoU={_display(item.get('iou'))}"
			)
			for item in failures
		]
		or ['metricsが不足しているため特定できない。'],
		'class imbalanceの影響': _imbalance_interpretation(token_dataset),
		'AGCあり/なし比較': _agc_interpretation(pretrained),
		'次の改善候補': [
			'comparison_table.csvでMODEL_TAG、EMBED_SPEC、PROBE_SPECごとの'
			'macro F1とmean IoUを比較する。',
			'低F1 classは教師slice追加、tokenization閾値、class weightingの'
			'影響を切り分ける。',
			'linear probeで頭打ちなら同じfrozen encoder上でMLP probeを比較する。',
		],
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
) -> dict[str, object]:
	config = _mapping(probe_config)
	model = _mapping(config.get('model'))
	labels = _mapping(config.get('labels'))
	probe = _mapping(config.get('probe'))
	path_parts = _run_parts(metrics_path)
	feature_source = _feature_source_summary(metrics, config, token_metadata)
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
) -> Mapping[str, object]:
	for candidate in (
		_mapping(metrics.get('feature_source')),
		_mapping(probe_config.get('feature_source')),
		_mapping(_mapping(probe_config.get('token_dataset')).get('feature_source')),
		_mapping(_mapping(probe_config.get('embeddings')).get('feature_source')),
		_mapping(_mapping(probe_config.get('model')).get('feature_source')),
		_mapping(_mapping(token_metadata).get('feature_source')),
	):
		if candidate:
			return candidate
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


def _comparison_figure_paths(output_markdown: Path) -> dict[str, Path]:
	figures_dir = output_markdown.parent / 'figures'
	return {
		name: figures_dir / f'{name}.png'
		for name in COMPARISON_FIGURE_NAMES
	}


def _write_comparison_figures(
	rows: Sequence[Mapping[str, object]],
	figure_paths: Mapping[str, Path],
	*,
	dpi: int,
) -> list[str]:
	try:
		plt = __import__('matplotlib.pyplot', fromlist=['pyplot'])
	except ImportError as exc:
		return [f'comparison figure generation requires matplotlib: {exc}']
	for path in figure_paths.values():
		path.parent.mkdir(parents=True, exist_ok=True)
	_save_metric_comparison_bar(
		rows,
		metric='macro_f1',
		title='Macro F1 comparison',
		ylabel='Macro F1',
		output_png=figure_paths['macro_f1_comparison'],
		plt=plt,
		dpi=dpi,
	)
	_save_metric_comparison_bar(
		rows,
		metric='mean_iou',
		title='Mean IoU comparison',
		ylabel='Mean IoU',
		output_png=figure_paths['mean_iou_comparison'],
		plt=plt,
		dpi=dpi,
	)
	_save_per_class_f1_comparison(
		rows,
		output_png=figure_paths['per_class_f1_comparison'],
		plt=plt,
		dpi=dpi,
	)
	return []


def _save_metric_comparison_bar(  # noqa: PLR0913
	rows: Sequence[Mapping[str, object]],
	*,
	metric: str,
	title: str,
	ylabel: str,
	output_png: Path,
	plt: object,
	dpi: int,
) -> None:
	plot_rows = [row for row in rows if _float_or_none(row.get(metric)) is not None]
	labels = [_comparison_row_label(row) for row in plot_rows]
	values = [_float_or_none(row.get(metric)) or 0.0 for row in plot_rows]
	colors = [_comparison_row_color(row) for row in plot_rows]
	fig_width = max(6.0, 1.1 * max(len(plot_rows), 1))
	fig, axis = plt.subplots(figsize=(fig_width, 4.2), facecolor='white')
	if plot_rows:
		positions = list(range(len(plot_rows)))
		axis.bar(positions, values, color=colors, edgecolor='black', linewidth=0.6)
		axis.set_xticks(positions, labels=labels, rotation=35, ha='right')
	else:
		axis.text(0.5, 0.5, 'No metrics', ha='center', va='center')
		axis.set_xticks([])
	axis.set_title(title)
	axis.set_ylabel(ylabel)
	axis.set_ylim(0.0, 1.0)
	axis.grid(axis='y', color='#D9D9D9', linewidth=0.8)
	axis.set_axisbelow(True)
	fig.tight_layout()
	fig.savefig(output_png, dpi=dpi, facecolor='white')
	plt.close(fig)


def _save_per_class_f1_comparison(
	rows: Sequence[Mapping[str, object]],
	*,
	output_png: Path,
	plt: object,
	dpi: int,
) -> None:
	class_columns = [
		key
		for key in _comparison_fieldnames(rows)
		if key.startswith('class_') and key.endswith('_f1')
	]
	plot_rows = [
		row
		for row in rows
		if any(_float_or_none(row.get(column)) is not None for column in class_columns)
	]
	fig_width = max(8.0, 1.35 * max(len(class_columns), 1))
	fig, axis = plt.subplots(figsize=(fig_width, 4.8), facecolor='white')
	if class_columns and plot_rows:
		group_width = 0.82
		bar_width = group_width / len(plot_rows)
		for row_index, row in enumerate(plot_rows):
			positions = [
				class_index - (group_width / 2.0) + (bar_width / 2.0)
				+ row_index * bar_width
				for class_index in range(len(class_columns))
			]
			values = [
				_float_or_none(row.get(column)) or 0.0
				for column in class_columns
			]
			axis.bar(
				positions,
				values,
				width=bar_width,
				label=_comparison_row_label(row),
				color=_comparison_row_color(row),
				edgecolor='black',
				linewidth=0.45,
			)
		axis.set_xticks(
			list(range(len(class_columns))),
			labels=[_class_f1_column_label(column, rows) for column in class_columns],
			rotation=0,
		)
		axis.legend(frameon=False, fontsize=8)
	else:
		axis.text(0.5, 0.5, 'No per-class F1 metrics', ha='center', va='center')
		axis.set_xticks([])
	axis.set_title('Per-class F1 comparison')
	axis.set_ylabel('F1')
	axis.set_ylim(0.0, 1.0)
	axis.grid(axis='y', color='#D9D9D9', linewidth=0.8)
	axis.set_axisbelow(True)
	fig.tight_layout()
	fig.savefig(output_png, dpi=dpi, facecolor='white')
	plt.close(fig)


def _comparison_row_label(row: Mapping[str, object]) -> str:
	feature_kind = str(row.get('feature_kind') or '')
	if feature_kind == 'pretrained_encoder':
		return str(_first_non_empty(row.get('MODEL_TAG'), 'pretrained_encoder'))
	return str(_first_non_empty(row.get('BASELINE_TAG'), feature_kind))


def _comparison_row_color(row: Mapping[str, object]) -> str:
	return {
		'pretrained_encoder': '#2563EB',
		'z_only': '#6B7280',
		'amplitude_stats': '#D97706',
		'random_encoder': '#7C3AED',
	}.get(str(row.get('feature_kind')), '#4B5563')


def _class_f1_column_label(
	column: str,
	rows: Sequence[Mapping[str, object]],
) -> str:
	match = re.fullmatch(r'class_(\d+)_f1', column)
	if match is None:
		return column
	class_id = match.group(1)
	for row in rows:
		class_name = _mapping(row.get('_class_names')).get(class_id)
		if isinstance(class_name, str) and class_name:
			return f'{class_id}\n{class_name}'
	return f'class {class_id}'


def _write_comparison_csv(
	path: Path,
	rows: Sequence[Mapping[str, object]],
	fieldnames: Sequence[str],
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow({key: row.get(key, '') for key in fieldnames})


def _render_comparison_markdown(
	rows: Sequence[Mapping[str, object]],
	fieldnames: Sequence[str],
	figure_paths: Mapping[str, Path],
	warnings: Sequence[str],
) -> str:
	lines = [
		'# F3 lithology probe comparison report',
		'',
		f'集約run数: {len(rows)}',
		'',
		'## Comparison table',
		'',
		'| ' + ' | '.join(fieldnames) + ' |',
		'|' + '|'.join('---' for _ in fieldnames) + '|',
	]
	lines.extend(
		(
			'| '
			+ ' | '.join(_display(row.get(field, '')) for field in fieldnames)
			+ ' |'
		)
		for row in rows
	)
	lines.extend(['', '## Figures', ''])
	report_dir = (
		next(iter(figure_paths.values())).parent.parent
		if figure_paths
		else Path()
	)
	lines.extend(
		f'- [{name}]({_relative_path_for_markdown(path, report_dir)})'
		for name, path in figure_paths.items()
	)
	lines.extend(['', '## Interpretation', ''])
	lines.extend(_comparison_interpretation(rows))
	lines.extend(['', '## Warnings', ''])
	if warnings:
		lines.extend(f'- {warning}' for warning in warnings)
	else:
		lines.append('- none')
	return '\n'.join(lines) + '\n'


def _comparison_interpretation(
	rows: Sequence[Mapping[str, object]],
) -> list[str]:
	pretrained = _best_comparison_row(rows, 'pretrained_encoder', metric='macro_f1')
	z_only = _best_comparison_row(rows, 'z_only', metric='macro_f1')
	amplitude = _best_comparison_row(rows, 'amplitude_stats', metric='macro_f1')
	random_encoder = _best_comparison_row(
		rows,
		'random_encoder',
		metric='macro_f1',
	)
	return [
		(
			'- pretrained encoderがz-onlyを上回るか: '
			f'{_comparison_delta_sentence(pretrained, z_only)}'
		),
		(
			'- pretrained encoderがamplitude-onlyを上回るか: '
			f'{_comparison_delta_sentence(pretrained, amplitude)}'
		),
		(
			'- pretrained encoderがrandom encoderを上回るか: '
			f'{_comparison_delta_sentence(pretrained, random_encoder)}'
		),
		(
			'- class 3/5など弱いclassで改善があるか: '
			f'{_weak_class_delta_sentence(rows, pretrained)}'
		),
		(
			'- F3 faciesが深度だけで説明できる程度: '
			f'{_depth_only_sentence(pretrained, z_only)}'
		),
	]


def _best_comparison_row(
	rows: Sequence[Mapping[str, object]],
	feature_kind: str,
	*,
	metric: str,
) -> Mapping[str, object] | None:
	candidates = [
		row
		for row in rows
		if row.get('feature_kind') == feature_kind
		and _float_or_none(row.get(metric)) is not None
	]
	if not candidates:
		return None
	return max(candidates, key=lambda row: _float_or_none(row.get(metric)) or 0.0)


def _comparison_delta_sentence(
	pretrained: Mapping[str, object] | None,
	baseline: Mapping[str, object] | None,
) -> str:
	if pretrained is None or baseline is None:
		return '比較対象のmetricsが不足しているため未確認。'
	macro_delta = _metric_delta(pretrained, baseline, 'macro_f1')
	iou_delta = _metric_delta(pretrained, baseline, 'mean_iou')
	if macro_delta is None:
		return 'macro F1が不足しているため未確認。'
	if macro_delta > 0.0:
		relation = '上回る'
	elif macro_delta == 0.0:
		relation = '同等'
	else:
		relation = '下回る'
	iou_text = (
		'mean IoU差分 未確認'
		if iou_delta is None
		else f'mean IoU差分 {iou_delta:+.4f}'
	)
	return f'{relation} (macro F1差分 {macro_delta:+.4f}, {iou_text})。'


def _weak_class_delta_sentence(
	rows: Sequence[Mapping[str, object]],
	pretrained: Mapping[str, object] | None,
) -> str:
	if pretrained is None:
		return 'pretrained encoder metricsが不足しているため未確認。'
	class_columns = [
		column
		for column in _comparison_fieldnames(rows)
		if column.startswith('class_') and column.endswith('_f1')
	]
	priority = [
		column for column in ('class_3_f1', 'class_5_f1') if column in class_columns
	]
	targets = priority or class_columns[:2]
	if not targets:
		return 'per-class F1が不足しているため未確認。'
	parts = []
	for column in targets:
		pretrained_value = _float_or_none(pretrained.get(column))
		baseline_value = _best_baseline_class_f1(rows, column)
		if pretrained_value is None or baseline_value is None:
			continue
		class_label = column.removeprefix('class_').removesuffix('_f1')
		parts.append(
			f'class {class_label}: '
			f'F1差分 {pretrained_value - baseline_value:+.4f}',
		)
	return '、'.join(parts) + '。' if parts else '比較可能なclass別F1が不足している。'


def _best_baseline_class_f1(
	rows: Sequence[Mapping[str, object]],
	column: str,
) -> float | None:
	values = [
		value
		for row in rows
		if row.get('feature_kind') in _BASELINE_FEATURE_KINDS
		for value in (_float_or_none(row.get(column)),)
		if value is not None
	]
	return max(values) if values else None


def _depth_only_sentence(
	pretrained: Mapping[str, object] | None,
	z_only: Mapping[str, object] | None,
) -> str:
	if pretrained is None or z_only is None:
		return 'z-onlyまたはpretrained encoder metricsが不足しているため未確認。'
	pretrained_macro = _float_or_none(pretrained.get('macro_f1'))
	z_macro = _float_or_none(z_only.get('macro_f1'))
	if pretrained_macro is None or z_macro is None:
		return 'macro F1が不足しているため未確認。'
	delta = pretrained_macro - z_macro
	if delta <= 0.02:
		return (
			f'z-onlyとの差が小さい (macro F1差分 {delta:+.4f}) ため、'
			'深度で説明できる寄与が大きい。'
		)
	return (
		f'z-onlyとの差がある (macro F1差分 {delta:+.4f}) ため、'
		'深度以外の特徴が効いている可能性がある。'
	)


def _metric_delta(
	left: Mapping[str, object],
	right: Mapping[str, object],
	metric: str,
) -> float | None:
	left_value = _float_or_none(left.get(metric))
	right_value = _float_or_none(right.get(metric))
	if left_value is None or right_value is None:
		return None
	return left_value - right_value


def _render_dataset(dataset: Mapping[str, object]) -> list[str]:
	classes = _sequence_of_mappings(dataset.get('classes'))
	lines = [
		f'- F3 shape: {_display(dataset.get("f3_shape"))}',
		f'- classes: {len(classes)}',
		f'- label source of truth: {_display(dataset.get("label_source_of_truth"))}',
		f'- PNG label role: {_display(dataset.get("png_label_role"))}',
		(
			'- train/validation slices: '
			f'{_display(dataset.get("train_validation_slices"))}'
		),
		(
			'- tokenization thresholds: '
			f'{_display(dataset.get("tokenization_thresholds"))}'
		),
		f'- class imbalance: {_display(dataset.get("class_imbalance"))}',
		'',
		'| class_id | class_name | rgb |',
		'|---:|---|---|',
	]
	lines.extend(
		(
			f'| {_display(item.get("class_id"))} | '
			f'{_display(_class_name(item))} | {_display(item.get("rgb"))} |'
		)
		for item in classes
	)
	return lines


def _render_pretrained(pretrained: Mapping[str, object]) -> list[str]:
	return [
		f'- MODEL_TAG: {_display(pretrained.get("MODEL_TAG"))}',
		f'- checkpoint path: {_display(pretrained.get("checkpoint_path"))}',
		f'- EMBED_SPEC: {_display(pretrained.get("EMBED_SPEC"))}',
		f'- AGC有無: {_display(pretrained.get("agc_enabled"))}',
		f'- visible loss有無: {_display(pretrained.get("visible_loss_enabled"))}',
		f'- mask ratio: {_display(pretrained.get("mask_ratio"))}',
		(
			'- encoder fine-tuning: '
			f'{_display(pretrained.get("freeze_encoder") is not True)}'
		),
	]


def _render_token_dataset(token_dataset: Mapping[str, object]) -> list[str]:
	return [
		f'- train token count: {_display(token_dataset.get("train_token_count"))}',
		(
			'- validation token count: '
			f'{_display(token_dataset.get("validation_token_count"))}'
		),
		f'- class counts: {_display(token_dataset.get("class_counts"))}',
		(
			'- dropped token ratio: '
			f'{_display(token_dataset.get("dropped_token_ratio"))}'
		),
		(
			'- ambiguous token ratio: '
			f'{_display(token_dataset.get("ambiguous_token_ratio"))}'
		),
	]


def _render_probe(probe: Mapping[str, object]) -> list[str]:
	return [
		f'- PROBE_SPEC: {_display(probe.get("PROBE_SPEC"))}',
		f'- classifier type: {_display(probe.get("classifier_type"))}',
		f'- feature scaling: {_display(probe.get("feature_scaling"))}',
		f'- class weighting: {_display(probe.get("class_weighting"))}',
		f'- hyperparameters: {_display(probe.get("hyperparameters"))}',
	]


def _render_metrics(metrics: Mapping[str, object]) -> list[str]:
	overall = _mapping(metrics.get('overall'))
	per_class = _sequence_of_mappings(metrics.get('per_class'))
	lines = [
		f'- accuracy: {_display(overall.get("accuracy"))}',
		f'- balanced accuracy: {_display(overall.get("balanced_accuracy"))}',
		f'- macro F1: {_display(overall.get("macro_f1"))}',
		f'- weighted F1: {_display(overall.get("weighted_f1"))}',
		f'- mean IoU: {_display(overall.get("mean_iou"))}',
		'',
		'| class_id | class_name | F1 | IoU | support |',
		'|---:|---|---:|---:|---:|',
	]
	lines.extend(
		(
			f'| {_display(item.get("class_id"))} | {_display(item.get("class_name"))} '
			f'| {_display(item.get("f1"))} | {_display(item.get("iou"))} '
			f'| {_display(item.get("support"))} |'
		)
		for item in per_class
	)
	lines.extend(['', '- confusion matrix:', '', '```text'])
	matrix = metrics.get('confusion_matrix')
	lines.append(_display(matrix))
	lines.append('```')
	return lines


def _render_figures(figures: Sequence[Mapping[str, object]]) -> list[str]:
	if not figures:
		return ['- none']
	return [
		f'- [{_display(item.get("type"))}]({_display(item.get("path"))})'
		for item in figures
	]


def _render_interpretation(interpretation: Mapping[str, object]) -> list[str]:
	lines: list[str] = []
	for key in (
		'良い点',
		'失敗しているclass',
		'class imbalanceの影響',
		'AGCあり/なし比較',
		'次の改善候補',
	):
		lines.append(f'### {key}')
		lines.append('')
		value = interpretation.get(key)
		if isinstance(value, Sequence) and not isinstance(value, str | bytes):
			lines.extend(f'- {item}' for item in value)
		else:
			lines.append(f'- {_display(value)}')
		lines.append('')
	return lines[:-1]


def _read_json_component(
	name: str,
	path: Path,
	warnings: list[str],
) -> Mapping[str, object] | None:
	if not path.is_file():
		warnings.append(f'missing input report component: {name} ({path})')
		return None
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		warnings.append(
			f'invalid input report component JSON: {name} ({path}): {exc.msg}',
		)
		return None
	if not isinstance(payload, Mapping):
		warnings.append(f'input report component is not a JSON object: {name} ({path})')
		return None
	return payload


def _read_optional_component(
	name: str,
	path: Path | None,
	warnings: list[str],
) -> Mapping[str, object] | None:
	if path is None:
		return None
	return _read_json_component(name, path, warnings)


def _read_optional_json(path: Path) -> Mapping[str, object] | None:
	if not path.is_file():
		return None
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError:
		return None
	return payload if isinstance(payload, Mapping) else None


def _token_dataset_metadata_path(
	config: F3LithologyReportConfig,
	probe_config: Mapping[str, object],
) -> Path | None:
	if config.token_dataset_metadata_json is not None:
		return config.token_dataset_metadata_json
	value = _mapping(probe_config.get('inputs')).get('token_dataset_metadata_json')
	return Path(value) if isinstance(value, str) and value else None


def _classes(
	probe_config: Mapping[str, object],
	token_metadata: Mapping[str, object],
	metrics: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
	for payload in (probe_config, token_metadata):
		classes = tuple(_sequence_of_mappings(payload.get('classes')))
		if classes:
			return classes
	class_names = _mapping(metrics.get('class_names'))
	class_ids = metrics.get('class_ids')
	if isinstance(class_ids, Sequence) and not isinstance(class_ids, str | bytes):
		return tuple(
			{
				'class_id': class_id,
				'class_name': class_names.get(str(class_id), f'class_{class_id}'),
			}
			for class_id in class_ids
		)
	return ()


def _slice_summary(token_metadata: Mapping[str, object]) -> dict[str, list[str]]:
	result = {'train': [], 'validation': []}
	for item in _sequence_of_mappings(token_metadata.get('slices')):
		split = item.get('split')
		if split not in result:
			continue
		result[split].append(
			f"{item.get('slice_type')} {item.get('slice_index')}",
		)
	return result


def _per_class_metrics(
	metrics: Mapping[str, object],
	classes: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
	f1 = _mapping(metrics.get('per_class_f1'))
	iou = _mapping(metrics.get('per_class_iou'))
	precision = _mapping(metrics.get('per_class_precision'))
	recall = _mapping(metrics.get('per_class_recall'))
	support = _mapping(metrics.get('per_class_support'))
	if not classes:
		classes = tuple({'class_id': key, 'class_name': f'class_{key}'} for key in f1)
	rows = []
	for item in classes:
		class_id = item.get('class_id')
		key = str(class_id)
		rows.append(
			{
				'class_id': class_id,
				'class_name': _class_name(item),
				'precision': _float_or_none(precision.get(key)),
				'recall': _float_or_none(recall.get(key)),
				'f1': _float_or_none(f1.get(key)),
				'iou': _float_or_none(iou.get(key)),
				'support': _int_or_none(support.get(key)),
			},
		)
	return rows


def _figure_record(
	figure_type: str,
	source: Path,
	report_dir: Path,
	warnings: list[str],
) -> dict[str, object]:
	relative = _relative_path_for_markdown(source, report_dir)
	exists = source.is_file()
	if not exists:
		warnings.append(f'missing report figure: {relative}')
	return {
		'type': figure_type,
		'path': relative,
		'source_path': str(source),
		'exists': exists,
	}


def _publish_items_for_f3_lithology_report(
	config: F3LithologyReportConfig,
	*,
	publish_config: F3LithologyPublishConfig,
	payload: Mapping[str, object],
) -> tuple[PublishItem, ...]:
	figure_items, text_replacements = _publish_figure_items_and_replacements(
		config,
		publish_config=publish_config,
		payload=payload,
	)
	published_payload = _publish_report_payload(
		payload,
		text_replacements=text_replacements,
	)
	return (
		PublishItem(
			config.output_markdown,
			_PUBLISH_REPORT_TARGET,
			content_text=render_f3_lithology_report_markdown(published_payload),
		),
		PublishItem(
			config.output_json,
			_PUBLISH_JSON_TARGET,
			content_text=(
				json.dumps(published_payload, indent=2, sort_keys=True) + '\n'
			),
		),
		PublishItem(config.metrics_json, _PUBLISH_METRICS_TARGET),
		PublishItem(
			config.metrics_json.with_name('metrics.csv'),
			_PUBLISH_METRICS_CSV_TARGET,
		),
		PublishItem(
			config.metrics_json.with_name('classification_report.md'),
			_PUBLISH_CLASSIFICATION_REPORT_TARGET,
		),
		PublishItem(
			config.metrics_json.with_name('confusion_matrix.csv'),
			_PUBLISH_CONFUSION_MATRIX_CSV_TARGET,
		),
		*figure_items,
	)


def _publish_items_for_f3_lithology_comparison_report(
	config: F3LithologyComparisonReportConfig,
	*,
	publish_config: F3LithologyComparisonPublishConfig,
) -> tuple[PublishItem, ...]:
	items = [
		PublishItem(config.output_markdown, Path('comparison_report.md')),
		PublishItem(config.output_csv, Path('comparison_table.csv')),
	]
	optional_json = config.output_csv.with_suffix('.json')
	if optional_json.is_file():
		items.append(PublishItem(optional_json, Path('comparison_table.json')))
	if publish_config.include_figures:
		items.extend(
			PublishItem(
				source,
				_PUBLISH_FIGURE_DIR / source.name,
				required=False,
			)
			for source in _comparison_figure_paths(config.output_markdown).values()
		)
	return tuple(items)


def _publish_figure_items_and_replacements(
	config: F3LithologyReportConfig,
	*,
	publish_config: F3LithologyPublishConfig,
	payload: Mapping[str, object],
) -> tuple[tuple[PublishItem, ...], tuple[tuple[str, str], ...]]:
	if not publish_config.include_figures:
		return (), ()

	report_dir = config.output_markdown.parent
	items: list[PublishItem] = []
	replacements: list[tuple[str, str]] = []
	planned_targets: set[Path] = set()
	figures_by_type = {
		str(item.get('type')): item
		for item in _sequence_of_mappings(payload.get('figures'))
	}

	for figure_type, relative in _DEFAULT_PROBE_FIGURES:
		source = _source_path_from_figure(figures_by_type.get(figure_type))
		if source is None:
			source = config.metrics_json.parent / relative
		target = _PUBLISH_FIGURE_DIR / relative.name
		_append_publish_figure_item(
			items=items,
			replacements=replacements,
			planned_targets=planned_targets,
			source=source,
			target=target,
			report_dir=report_dir,
		)

	for source in _publish_prediction_figure_sources(
		config,
		max_prediction_figures=publish_config.max_prediction_figures,
	):
		_append_publish_figure_item(
			items=items,
			replacements=replacements,
			planned_targets=planned_targets,
			source=source,
			target=_PUBLISH_FIGURE_DIR / source.name,
			report_dir=report_dir,
		)

	return tuple(items), tuple(replacements)


def _append_publish_figure_item(  # noqa: PLR0913
	*,
	items: list[PublishItem],
	replacements: list[tuple[str, str]],
	planned_targets: set[Path],
	source: Path,
	target: Path,
	report_dir: Path,
) -> None:
	if target in planned_targets:
		return
	items.append(PublishItem(source, target, required=False))
	if source.is_file():
		replacements.append(
			(
				_relative_path_for_markdown(source, report_dir),
				target.as_posix(),
			),
		)
	planned_targets.add(target)


def _publish_prediction_figure_sources(
	config: F3LithologyReportConfig,
	*,
	max_prediction_figures: int,
) -> tuple[Path, ...]:
	if max_prediction_figures == 0 or config.visualization_metadata_json is None:
		return ()
	metadata = _read_optional_json(config.visualization_metadata_json)
	sources: list[Path] = []
	seen: set[Path] = set()
	for item in _sequence_of_mappings(_mapping(metadata).get('figures')):
		path = item.get('path')
		if not isinstance(path, str) or not path:
			continue
		source = Path(path)
		if source in seen or not _is_validation_prediction_figure(item, source):
			continue
		sources.append(source)
		seen.add(source)
		if len(sources) >= max_prediction_figures:
			break
	return tuple(sources)


def _is_validation_prediction_figure(
	item: Mapping[str, object],
	source: Path,
) -> bool:
	if source.suffix.lower() != '.png':
		return False
	if item.get('group') == 'validation':
		return True
	return source.name.startswith('validation_') and 'prediction' in source.name


def _source_path_from_figure(item: Mapping[str, object] | None) -> Path | None:
	if item is None:
		return None
	value = item.get('source_path')
	return Path(value) if isinstance(value, str) and value else None


def _publish_report_payload(
	payload: Mapping[str, object],
	*,
	text_replacements: Sequence[tuple[str, str]],
) -> dict[str, object]:
	published = deepcopy(dict(payload))
	published.pop('outputs', None)
	published.pop('inputs', None)
	published.pop('comparison', None)
	pretrained = dict(_mapping(published.get('pretrained_encoder')))
	pretrained.pop('checkpoint_path', None)
	published['pretrained_encoder'] = pretrained
	path_replacements = dict(text_replacements)
	published['figures'] = [
		_publish_figure_payload(item, path_replacements=path_replacements)
		for item in _sequence_of_mappings(published.get('figures'))
		if item.get('path') in path_replacements
	]
	return published


def _publish_figure_payload(
	item: Mapping[str, object],
	*,
	path_replacements: Mapping[str, str],
) -> dict[str, object]:
	figure = dict(item)
	path = figure.get('path')
	if isinstance(path, str):
		figure['path'] = path_replacements.get(path, path)
	figure.pop('source_path', None)
	return figure


def _read_required_json_object(path: Path, name: str) -> Mapping[str, object]:
	if not path.is_file():
		msg = f'required publish source does not exist: {path}'
		raise FileNotFoundError(msg)
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		msg = f'{name} JSON is invalid: {path}: {exc.msg}'
		raise ValueError(msg) from exc
	if not isinstance(payload, Mapping):
		msg = f'{name} JSON must be an object: {path}'
		raise TypeError(msg)
	return payload


def _validate_max_prediction_figures(value: int) -> None:
	if isinstance(value, bool) or not isinstance(value, int) or value < 0:
		msg = f'publish.max_prediction_figures must be non-negative; got {value!r}'
		raise ValueError(msg)


def _embed_spec(
	lithology: Mapping[str, object],
	probe_config: Mapping[str, object],
) -> str | None:
	return _first_non_empty(
		_embed_spec_from_config(probe_config),
		_embed_spec_from_lithology_root(lithology.get('root')),
	)


def _embed_spec_from_config(probe_config: Mapping[str, object]) -> str | None:
	embeddings = _mapping(probe_config.get('embeddings'))
	for key in ('spec', 'embed_spec', 'name'):
		value = embeddings.get(key)
		if isinstance(value, str) and value:
			return value
	lithology = _mapping(probe_config.get('lithology'))
	return _embed_spec_from_lithology_root(lithology.get('root'))


def _embed_spec_from_lithology_root(value: object) -> str | None:
	if not isinstance(value, str) or not value:
		return None
	parts = Path(value).parts
	if 'facies_benchmark_v1' not in parts:
		return None
	index = parts.index('facies_benchmark_v1')
	if len(parts) <= index + 2:
		return None
	return parts[index + 2]


def _agc_enabled(model: Mapping[str, object]) -> bool | None:
	agc = _mapping(model.get('amplitude_agc'))
	if isinstance(agc.get('enabled'), bool):
		return bool(agc['enabled'])
	tag = _string_or_none(model.get('tag'))
	if tag is None:
		return None
	return '_agc' in tag


def _visible_loss_enabled(model_tag: str | None) -> bool | None:
	if model_tag is None:
		return None
	match = re.search(r'_vis(\d+)', model_tag)
	if match is None:
		return None
	return int(match.group(1)) > 0


def _mask_ratio(model_tag: str | None) -> float | None:
	if model_tag is None:
		return None
	match = re.search(r'_m(\d{3})_', model_tag)
	if match is None:
		return None
	return int(match.group(1)) / 100.0


def _imbalance_interpretation(token_dataset: Mapping[str, object]) -> str:
	imbalance = _mapping(token_dataset.get('class_imbalance'))
	ratio = _float_or_none(imbalance.get('max_to_min_positive_ratio'))
	if ratio is None:
		return 'class count情報が不足しているため影響を評価できない。'
	if ratio > 5.0:
		return (
			f'class countの最大/最小比が{ratio:.3g}で、minor classのF1低下に注意する。'
		)
	return f'class countの最大/最小比は{ratio:.3g}で、極端な偏りは限定的。'


def _agc_interpretation(pretrained: Mapping[str, object]) -> str:
	agc = pretrained.get('agc_enabled')
	state = 'AGCあり' if agc is True else 'AGCなし' if agc is False else 'AGC不明'
	return (
		f'このrunは{state}として集計される。AGCあり/なしの優劣は'
		'comparison_table.csvで同じEMBED_SPEC、LABEL_SET、PROBE_SPECを揃えて比較する。'
	)


def _class_imbalance(counts: Mapping[str, int]) -> dict[str, object]:
	positive = [value for value in counts.values() if value > 0]
	total = sum(counts.values())
	return {
		'total': total,
		'class_counts': dict(counts),
		'max_to_min_positive_ratio': (
			None if not positive else max(positive) / min(positive)
		),
	}


def _combined_counts(
	left: Mapping[object, object],
	right: Mapping[object, object],
) -> dict[str, int]:
	counts: dict[str, int] = {}
	for source in (left, right):
		for key, value in source.items():
			integer = _int_or_none(value)
			if integer is None:
				continue
			counts[str(key)] = counts.get(str(key), 0) + integer
	return counts


def _run_parts(metrics_path: Path) -> dict[str, str]:
	parts = metrics_path.parts
	if 'facies_benchmark_v1' not in parts:
		return {'PROBE_SPEC': metrics_path.parent.name}
	index = parts.index('facies_benchmark_v1')
	values: dict[str, str] = {}
	if len(parts) > index + 1 and parts[index + 1] == 'baselines':
		if len(parts) > index + 2:
			values['BASELINE_TAG'] = parts[index + 2]
		if len(parts) > index + 3:
			values['LABEL_SET'] = parts[index + 3]
		probe_spec = _probe_spec_from_parts(parts)
		if probe_spec is not None:
			values['PROBE_SPEC'] = probe_spec
		return values
	if len(parts) > index + 1:
		values['MODEL_TAG'] = parts[index + 1]
	if len(parts) > index + 2:
		values['EMBED_SPEC'] = parts[index + 2]
	if len(parts) > index + 3:
		values['LABEL_SET'] = parts[index + 3]
	probe_spec = _probe_spec_from_parts(parts)
	if probe_spec is not None:
		values['PROBE_SPEC'] = probe_spec
	return values


def _probe_spec_from_parts(parts: Sequence[str]) -> str | None:
	if 'probes' not in parts:
		return None
	probe_index = parts.index('probes')
	if len(parts) <= probe_index + 1:
		return None
	return parts[probe_index + 1]


def _class_metric_sort_key(value: str) -> tuple[int, str]:
	match = re.fullmatch(r'class_(\d+)_f1', value)
	if match is None:
		return (10**9, value)
	return (int(match.group(1)), value)


def _prefer_mapping(
	preferred: Mapping[str, object],
	fallback: Mapping[str, object],
) -> Mapping[str, object]:
	return preferred if preferred else fallback


def _mapping(value: object) -> Mapping[str, object]:
	return value if isinstance(value, Mapping) else {}


def _sequence_of_mappings(value: object) -> list[Mapping[str, object]]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		return []
	return [item for item in value if isinstance(item, Mapping)]


def _string_list(value: object) -> list[str]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		return []
	return [item for item in value if isinstance(item, str)]


def _first_non_empty(*values: object) -> object:
	for value in values:
		if value not in (None, ''):
			return value
	return None


def _string_or_none(value: object) -> str | None:
	return value if isinstance(value, str) and value else None


def _float_or_none(value: object) -> float | None:
	if isinstance(value, bool):
		return None
	if isinstance(value, int | float):
		return float(value)
	return None


def _int_or_none(value: object) -> int | None:
	if isinstance(value, bool):
		return None
	if isinstance(value, int):
		return value
	if isinstance(value, float) and value.is_integer():
		return int(value)
	return None


def _sum_ints(values: Sequence[object]) -> int | None:
	total = 0
	for value in values:
		integer = _int_or_none(value)
		if integer is None:
			return None
		total += integer
	return total


def _fraction_or_none(numerator: int | None, denominator: int | None) -> float | None:
	if numerator is None or denominator is None or denominator == 0:
		return None
	return float(numerator / denominator)


def _class_name(item: Mapping[str, object]) -> object:
	return _first_non_empty(item.get('class_name'), item.get('name'))


def _display(value: object) -> str:
	if value is None:
		return '未確認'
	if isinstance(value, float):
		return f'{value:.4f}'
	if isinstance(value, list | tuple):
		return json.dumps(value, ensure_ascii=False)
	if isinstance(value, Mapping):
		return json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
	return str(value)


def _relative_path_for_markdown(path: Path, report_dir: Path) -> str:
	try:
		return os.path.relpath(path, start=report_dir)
	except ValueError:
		return path.as_posix()


def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
	json_path = Path(path)
	json_path.parent.mkdir(parents=True, exist_ok=True)
	json_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + '\n',
		encoding='utf-8',
	)


def _write_text(path: str | Path, text: str) -> None:
	text_path = Path(path)
	text_path.parent.mkdir(parents=True, exist_ok=True)
	text_path.write_text(text, encoding='utf-8')


__all__ = [
	'COMPARISON_ID_COLUMNS',
	'OVERALL_METRIC_COLUMNS',
	'F3LithologyComparisonPublishConfig',
	'F3LithologyComparisonReportConfig',
	'F3LithologyComparisonReportResult',
	'F3LithologyReportConfig',
	'F3LithologyReportResult',
	'build_f3_lithology_comparison_report',
	'build_f3_lithology_report',
	'publish_f3_lithology_comparison_report',
	'render_f3_lithology_report_markdown',
]
