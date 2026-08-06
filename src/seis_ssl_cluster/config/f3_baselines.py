"""F3 lithology baseline config validation entrypoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from seis_ssl_cluster.f3 import (
	F3LithologyComparisonFigureFontSizes,
	F3LithologyComparisonFigureSizes,
	F3LithologyComparisonFigureStyle,
	F3LithologyComparisonPublishConfig,
	F3LithologyComparisonReportConfig,
	default_f3_lithology_comparison_figure_style,
)
from seis_ssl_cluster.f3.lithology.baselines import (
	f3_lithology_baseline_token_dataset_config_from_mapping,
)
from seis_ssl_cluster.training.random_checkpoint import (
	random_mae_checkpoint_config_from_mapping,
)

_DEFAULT_COMPARISON_SEARCH_ROOT = Path(
	'/workspace/artifacts/seis_ssl_cluster/lithology/f3/facies_benchmark_v1'
)
_DEFAULT_COMPARISON_OUTPUT_DIR = (
	_DEFAULT_COMPARISON_SEARCH_ROOT / 'reports' / 'baseline_comparison'
)


def f3_lithology_comparison_report_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyComparisonReportConfig:
	"""Validate and normalize the standalone F3 lithology comparison config."""
	_validate_allowed_keys(
		config,
		frozenset({'paths', 'dataset', 'comparison', 'publish'}),
		prefix='config',
	)
	comparison = _required_mapping(config, 'comparison')
	search_root = _optional_absolute_path(
		comparison,
		'search_root',
		prefix='comparison',
		default=_DEFAULT_COMPARISON_SEARCH_ROOT,
	)
	output_dir = _optional_absolute_path(
		comparison,
		'output_dir',
		prefix='comparison',
		default=_DEFAULT_COMPARISON_OUTPUT_DIR,
	)
	figure_style = _comparison_figure_style_from_mapping(comparison)
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
		metrics_paths=_metrics_paths_from_mapping(comparison),
		figure_dpi=_comparison_figure_dpi_from_mapping(comparison),
		figure_style=figure_style,
	)


def f3_lithology_comparison_publish_config_from_mapping(
	value: object,
) -> F3LithologyComparisonPublishConfig:
	"""Validate and normalize the optional F3 comparison publish config."""
	if value is None:
		return F3LithologyComparisonPublishConfig()
	if not isinstance(value, Mapping):
		msg = f'publish must be a mapping; got {value!r}'
		raise TypeError(msg)
	_validate_allowed_keys(
		value,
		frozenset({'enabled', 'output_dir', 'include_figures', 'max_file_size_mb'}),
		prefix='publish',
	)
	enabled = _optional_bool(value, 'enabled', default=False)
	include_figures = _optional_bool(value, 'include_figures', default=True)
	output_dir = _optional_path(value, 'output_dir')
	if enabled and output_dir is None:
		msg = 'publish.output_dir must be set when publish.enabled is true'
		raise ValueError(msg)
	return F3LithologyComparisonPublishConfig(
		enabled=enabled,
		output_dir=output_dir,
		include_figures=include_figures,
		max_file_size_bytes=_max_file_size_bytes(value),
	)


def _comparison_figure_dpi_from_mapping(comparison: Mapping[str, object]) -> int:
	legacy_dpi = _optional_int(
		comparison,
		'figure_dpi',
		prefix='comparison',
		default=300,
	)
	figures = _optional_mapping(comparison, 'figures')
	return _optional_positive_int(
		figures,
		'dpi',
		prefix='comparison.figures',
		default=legacy_dpi,
	)


def _comparison_figure_style_from_mapping(
	comparison: Mapping[str, object],
) -> F3LithologyComparisonFigureStyle:
	default = default_f3_lithology_comparison_figure_style()
	figures = _optional_mapping(comparison, 'figures')
	if not figures:
		return default
	_validate_allowed_keys(
		figures,
		frozenset({'dpi', 'font_sizes', 'figsize'}),
		prefix='comparison.figures',
	)
	return F3LithologyComparisonFigureStyle(
		font_sizes=_comparison_figure_font_sizes_from_mapping(
			_optional_mapping(figures, 'font_sizes'),
			default=default.font_sizes,
		),
		figsize=_comparison_figure_sizes_from_mapping(
			_optional_mapping(figures, 'figsize'),
			default=default.figsize,
		),
	)


def _comparison_figure_font_sizes_from_mapping(
	value: Mapping[str, object],
	*,
	default: F3LithologyComparisonFigureFontSizes,
) -> F3LithologyComparisonFigureFontSizes:
	_validate_allowed_keys(
		value,
		frozenset({'title', 'axis_label', 'tick', 'legend', 'bar_label'}),
		prefix='comparison.figures.font_sizes',
	)
	return F3LithologyComparisonFigureFontSizes(
		title=_optional_positive_int(
			value,
			'title',
			prefix='comparison.figures.font_sizes',
			default=default.title,
		),
		axis_label=_optional_positive_int(
			value,
			'axis_label',
			prefix='comparison.figures.font_sizes',
			default=default.axis_label,
		),
		tick=_optional_positive_int(
			value,
			'tick',
			prefix='comparison.figures.font_sizes',
			default=default.tick,
		),
		legend=_optional_positive_int(
			value,
			'legend',
			prefix='comparison.figures.font_sizes',
			default=default.legend,
		),
		bar_label=_optional_positive_int(
			value,
			'bar_label',
			prefix='comparison.figures.font_sizes',
			default=default.bar_label,
		),
	)


def _comparison_figure_sizes_from_mapping(
	value: Mapping[str, object],
	*,
	default: F3LithologyComparisonFigureSizes,
) -> F3LithologyComparisonFigureSizes:
	_validate_allowed_keys(
		value,
		frozenset({'metric', 'per_class'}),
		prefix='comparison.figures.figsize',
	)
	return F3LithologyComparisonFigureSizes(
		metric=_optional_figsize(
			value,
			'metric',
			prefix='comparison.figures.figsize',
			default=default.metric,
		),
		per_class=_optional_figsize(
			value,
			'per_class',
			prefix='comparison.figures.figsize',
			default=default.per_class,
		),
	)


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, Any]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, Any]:
	value = parent.get(key)
	if value is None:
		return {}
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_allowed_keys(
	mapping: Mapping[str, object],
	allowed: frozenset[str],
	*,
	prefix: str,
) -> None:
	unknown = sorted(set(mapping) - allowed)
	if unknown:
		msg = f'{prefix} has unsupported key(s): {", ".join(unknown)}'
		raise ValueError(msg)


def _optional_absolute_path(
	mapping: Mapping[str, object],
	key: str,
	*,
	prefix: str,
	default: Path,
) -> Path:
	value = mapping.get(key)
	if value is None:
		return default
	return _absolute_path(value, label=f'{prefix}.{key}')


def _absolute_path(value: object, *, label: str) -> Path:
	if not isinstance(value, str) or not value:
		msg = f'{label} must be a non-empty string path; got {value!r}'
		raise TypeError(msg)
	path = Path(value)
	if not path.is_absolute():
		msg = f'{label} must be an absolute path: {path}'
		raise ValueError(msg)
	return path


def _optional_int(
	mapping: Mapping[str, object],
	key: str,
	*,
	prefix: str,
	default: int,
) -> int:
	value = mapping.get(key)
	if value is None:
		return default
	if not isinstance(value, int):
		msg = f'{prefix}.{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_positive_int(
	mapping: Mapping[str, object],
	key: str,
	*,
	prefix: str,
	default: int,
) -> int:
	value = _optional_int(mapping, key, prefix=prefix, default=default)
	if value <= 0:
		msg = f'{prefix}.{key} must be positive; got {value!r}'
		raise ValueError(msg)
	return value


def _optional_figsize(
	mapping: Mapping[str, object],
	key: str,
	*,
	prefix: str,
	default: tuple[float, float],
) -> tuple[float, float]:
	value = mapping.get(key)
	if value is None:
		return default
	if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
		msg = f'{prefix}.{key} must be a two-item numeric sequence; got {value!r}'
		raise TypeError(msg)
	return (
		_positive_float(value[0], label=f'{prefix}.{key}[0]'),
		_positive_float(value[1], label=f'{prefix}.{key}[1]'),
	)


def _positive_float(value: object, *, label: str) -> float:
	if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
		msg = f'{label} must be positive; got {value!r}'
		raise ValueError(msg)
	return float(value)


def _optional_bool(
	mapping: Mapping[str, object],
	key: str,
	*,
	default: bool,
) -> bool:
	value = mapping.get(key, default)
	if not isinstance(value, bool):
		msg = f'publish.{key} must be a boolean; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_path(mapping: Mapping[str, object], key: str) -> Path | None:
	value = mapping.get(key)
	if value is None:
		return None
	if not isinstance(value, str) or not value:
		msg = f'publish.{key} must be a non-empty string path; got {value!r}'
		raise TypeError(msg)
	return Path(value)


def _max_file_size_bytes(mapping: Mapping[str, object]) -> int:
	value = mapping.get('max_file_size_mb', 10)
	if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
		msg = f'publish.max_file_size_mb must be positive; got {value!r}'
		raise ValueError(msg)
	return int(value * 1024 * 1024)


def _metrics_paths_from_mapping(mapping: Mapping[str, object]) -> tuple[Path, ...]:
	value = mapping.get('metrics_json')
	if value is None:
		return ()
	if not isinstance(value, Sequence) or isinstance(value, str):
		msg = f'comparison.metrics_json must be a sequence; got {value!r}'
		raise TypeError(msg)
	return tuple(
		_absolute_path(item, label='comparison.metrics_json')
		for item in value
	)


__all__ = [
	'f3_lithology_baseline_token_dataset_config_from_mapping',
	'f3_lithology_comparison_publish_config_from_mapping',
	'f3_lithology_comparison_report_config_from_mapping',
	'random_mae_checkpoint_config_from_mapping',
]
