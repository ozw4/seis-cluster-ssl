"""Results publishing helpers for F3 lithology reports."""

from __future__ import annotations

import json
import shutil
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

DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

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
) -> tuple[Path, ...]:
	"""Publish lightweight F3 lithology probe report artifacts into ``results/``."""
	if publish_config is None or not publish_config.enabled:
		return ()
	if publish_config.output_dir is None:
		msg = 'publish output_dir is required when publishing is enabled'
		raise ValueError(msg)
	_validate_max_prediction_figures(publish_config.max_prediction_figures)
	if payload is None:
		payload = _read_required_json_object(config.output_json, 'publish report')
	return _write_published_f3_lithology_report(
		config,
		publish_config=publish_config,
		payload=payload,
	)

def publish_f3_lithology_comparison_report(
	config: F3LithologyComparisonReportConfig,
	publish_config: F3LithologyComparisonPublishConfig | None,
) -> tuple[Path, ...]:
	"""Publish lightweight F3 lithology comparison artifacts into ``results/``."""
	if publish_config is None or not publish_config.enabled:
		return ()
	if publish_config.output_dir is None:
		msg = 'publish output_dir is required when publishing is enabled'
		raise ValueError(msg)
	return _copy_f3_lithology_comparison_files(
		config,
		publish_config=publish_config,
	)

def _write_published_f3_lithology_report(
	config: F3LithologyReportConfig,
	*,
	publish_config: F3LithologyPublishConfig,
	payload: Mapping[str, object],
) -> tuple[Path, ...]:
	figure_sources, text_replacements = _publish_figure_sources_and_replacements(
		config,
		publish_config=publish_config,
		payload=payload,
	)
	published_payload = _publish_report_payload(
		payload,
		text_replacements=text_replacements,
	)
	if publish_config.output_dir is None:
		raise ValueError('publish output_dir is required when publishing is enabled')
	output_dir = publish_config.output_dir
	files = [
		_write_published_text(
			config.output_markdown,
			output_dir / _PUBLISH_REPORT_TARGET,
			render_f3_lithology_report_markdown(published_payload),
			max_file_size_bytes=publish_config.max_file_size_bytes,
		),
		_write_published_text(
			config.output_json,
			output_dir / _PUBLISH_JSON_TARGET,
			json.dumps(published_payload, indent=2, sort_keys=True) + '\n',
			max_file_size_bytes=publish_config.max_file_size_bytes,
		),
	]
	for source, relative_target in (
		(config.metrics_json, _PUBLISH_METRICS_TARGET),
		(config.metrics_json.with_name('metrics.csv'), _PUBLISH_METRICS_CSV_TARGET),
		(
			config.metrics_json.with_name('classification_report.md'),
			_PUBLISH_CLASSIFICATION_REPORT_TARGET,
		),
		(
			config.metrics_json.with_name('confusion_matrix.csv'),
			_PUBLISH_CONFUSION_MATRIX_CSV_TARGET,
		),
	):
		files.append(
			_copy_published_file(
				source,
				output_dir / relative_target,
				max_file_size_bytes=publish_config.max_file_size_bytes,
			)
		)
	for source, relative_target in figure_sources:
		if source.is_file():
			files.append(
				_copy_published_file(
					source,
					output_dir / relative_target,
					max_file_size_bytes=publish_config.max_file_size_bytes,
				)
			)
	return tuple(files)

def _copy_f3_lithology_comparison_files(
	config: F3LithologyComparisonReportConfig,
	*,
	publish_config: F3LithologyComparisonPublishConfig,
) -> tuple[Path, ...]:
	if publish_config.output_dir is None:
		raise ValueError('publish output_dir is required when publishing is enabled')
	output_dir = publish_config.output_dir
	sources = [
		(config.output_markdown, Path('comparison_report.md')),
		(config.output_csv, Path('comparison_table.csv')),
	]
	optional_json = config.output_csv.with_suffix('.json')
	if optional_json.is_file():
		sources.append((optional_json, Path('comparison_table.json')))
	if publish_config.include_figures:
		sources.extend(
			(source, _PUBLISH_FIGURE_DIR / source.name)
			for source in _comparison_figure_paths(config.output_markdown).values()
			if source.is_file()
		)
	return tuple(
		_copy_published_file(
			source,
			output_dir / relative_target,
			max_file_size_bytes=publish_config.max_file_size_bytes,
		)
		for source, relative_target in sources
	)

def _publish_figure_sources_and_replacements(
	config: F3LithologyReportConfig,
	*,
	publish_config: F3LithologyPublishConfig,
	payload: Mapping[str, object],
) -> tuple[tuple[tuple[Path, Path], ...], tuple[tuple[str, str], ...]]:
	if not publish_config.include_figures:
		return (), ()

	report_dir = config.output_markdown.parent
	items: list[tuple[Path, Path]] = []
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
		_append_publish_figure_source(
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
		_append_publish_figure_source(
			items=items,
			replacements=replacements,
			planned_targets=planned_targets,
			source=source,
			target=_PUBLISH_FIGURE_DIR / source.name,
			report_dir=report_dir,
		)

	return tuple(items), tuple(replacements)

def _append_publish_figure_source(  # noqa: PLR0913
	*,
	items: list[tuple[Path, Path]],
	replacements: list[tuple[str, str]],
	planned_targets: set[Path],
	source: Path,
	target: Path,
	report_dir: Path,
) -> None:
	if target in planned_targets:
		return
	items.append((source, target))
	if source.is_file():
		replacements.append(
			(
				_relative_path_for_markdown(source, report_dir),
				target.as_posix(),
			),
		)
	planned_targets.add(target)

def _copy_published_file(
	source: Path, target: Path, *, max_file_size_bytes: int
) -> Path:
	if not source.is_file():
		raise FileNotFoundError(f'required publish source does not exist: {source}')
	if source.stat().st_size > max_file_size_bytes:
		raise ValueError(f'publish source exceeds max_file_size_bytes: {source}')
	target.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(source, target)
	return target

def _write_published_text(
	source: Path, target: Path, text: str, *, max_file_size_bytes: int
) -> Path:
	if not source.is_file():
		raise FileNotFoundError(f'required publish source does not exist: {source}')
	if len(text.encode('utf-8')) > max_file_size_bytes:
		raise ValueError(f'publish source exceeds max_file_size_bytes: {source}')
	target.parent.mkdir(parents=True, exist_ok=True)
	target.write_text(text, encoding='utf-8')
	return target

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
	'_append_publish_figure_source',
	'_copy_f3_lithology_comparison_files',
	'_is_validation_prediction_figure',
	'_publish_figure_payload',
	'_publish_figure_sources_and_replacements',
	'_publish_prediction_figure_sources',
	'_publish_report_payload',
	'_source_path_from_figure',
	'_validate_max_prediction_figures',
	'_write_published_f3_lithology_report',
	'publish_f3_lithology_comparison_report',
	'publish_f3_lithology_report',
]
