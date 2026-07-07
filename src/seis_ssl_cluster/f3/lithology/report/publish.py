"""Results publishing helpers for F3 lithology reports."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.f3.lithology.report._common import (
	_DEFAULT_PROBE_FIGURES,
	_mapping,
	_relative_path_for_markdown,
	_sequence_of_mappings,
)
from seis_ssl_cluster.f3.lithology.report.figures import _comparison_figure_paths
from seis_ssl_cluster.f3.lithology.report.markdown import (
	render_f3_lithology_report_markdown,
)
from seis_ssl_cluster.f3.lithology.report.metrics_loader import (
	_read_optional_json,
	_read_required_json_object,
)
from seis_ssl_cluster.results import (
	DEFAULT_MAX_FILE_SIZE_BYTES,
	PublishItem,
	PublishManifest,
	publish_selected_results,
)

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

	from seis_ssl_cluster.f3.lithology.report._core import F3LithologyReportConfig
	from seis_ssl_cluster.f3.lithology.report.comparison import (
		F3LithologyComparisonReportConfig,
	)

_PUBLISH_REPORT_TARGET = Path('report.md')
_PUBLISH_JSON_TARGET = Path('report.json')
_PUBLISH_METRICS_TARGET = Path('metrics.json')
_PUBLISH_METRICS_CSV_TARGET = Path('metrics.csv')
_PUBLISH_CLASSIFICATION_REPORT_TARGET = Path('classification_report.md')
_PUBLISH_CONFUSION_MATRIX_CSV_TARGET = Path('confusion_matrix.csv')
_PUBLISH_FIGURE_DIR = Path('figures')

@dataclass(frozen=True)
class F3LithologyComparisonPublishConfig:
	"""Settings for publishing a lightweight F3 lithology comparison report."""

	enabled: bool = False
	output_dir: Path | None = None
	include_figures: bool = True
	max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES


@dataclass(frozen=True)
class F3LithologyPublishConfig:
	"""Settings for publishing a lightweight F3 lithology probe report copy."""

	enabled: bool = False
	output_dir: Path | None = None
	include_figures: bool = True
	max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
	max_prediction_figures: int = 3


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

def _validate_max_prediction_figures(value: int) -> None:
	if isinstance(value, bool) or not isinstance(value, int) or value < 0:
		msg = f'publish.max_prediction_figures must be non-negative; got {value!r}'
		raise ValueError(msg)

__all__ = [
	'F3LithologyComparisonPublishConfig',
	'F3LithologyPublishConfig',
	'_append_publish_figure_item',
	'_is_validation_prediction_figure',
	'_publish_figure_items_and_replacements',
	'_publish_figure_payload',
	'_publish_items_for_f3_lithology_comparison_report',
	'_publish_items_for_f3_lithology_report',
	'_publish_prediction_figure_sources',
	'_publish_report_payload',
	'_source_path_from_figure',
	'_validate_max_prediction_figures',
	'publish_f3_lithology_comparison_report',
	'publish_f3_lithology_report',
]
