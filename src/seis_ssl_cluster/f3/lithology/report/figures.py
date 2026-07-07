"""Figure helpers for F3 lithology reports."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.f3.lithology.report._common import (
	_DEFAULT_PROBE_FIGURES,
	COMPARISON_FIGURE_NAMES,
	_class_metric_sort_key,
	_first_non_empty,
	_float_or_none,
	_mapping,
	_relative_path_for_markdown,
	_sequence_of_mappings,
)

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

	from seis_ssl_cluster.f3.lithology.report._core import F3LithologyReportConfig


@dataclass(frozen=True)
class F3LithologyComparisonFigureFontSizes:
	"""Font sizes for baseline comparison figures."""

	title: int = 10
	axis_label: int = 9
	tick: int = 8
	legend: int = 8
	bar_label: int = 7


@dataclass(frozen=True)
class F3LithologyComparisonFigureSizes:
	"""Figure sizes for baseline comparison figures."""

	metric: tuple[float, float] = (6.5, 3.6)
	per_class: tuple[float, float] = (8.0, 4.2)


@dataclass(frozen=True)
class F3LithologyComparisonFigureStyle:
	"""Style defaults for F3 baseline comparison figures."""

	font_sizes: F3LithologyComparisonFigureFontSizes = field(
		default_factory=F3LithologyComparisonFigureFontSizes,
	)
	figsize: F3LithologyComparisonFigureSizes = field(
		default_factory=F3LithologyComparisonFigureSizes,
	)


def default_f3_lithology_comparison_figure_style() -> F3LithologyComparisonFigureStyle:
	"""Return the default publication-oriented comparison figure style."""
	return F3LithologyComparisonFigureStyle()
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
	style: F3LithologyComparisonFigureStyle,
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
		title='Macro F1',
		ylabel='Macro F1',
		output_png=figure_paths['macro_f1_comparison'],
		plt=plt,
		dpi=dpi,
		style=style,
	)
	_save_metric_comparison_bar(
		rows,
		metric='mean_iou',
		title='Mean IoU',
		ylabel='Mean IoU',
		output_png=figure_paths['mean_iou_comparison'],
		plt=plt,
		dpi=dpi,
		style=style,
	)
	_save_per_class_f1_comparison(
		rows,
		output_png=figure_paths['per_class_f1_comparison'],
		plt=plt,
		dpi=dpi,
		style=style,
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
	style: F3LithologyComparisonFigureStyle,
) -> None:
	plot_rows = [row for row in rows if _float_or_none(row.get(metric)) is not None]
	labels = [_comparison_row_label(row) for row in plot_rows]
	values = [_float_or_none(row.get(metric)) or 0.0 for row in plot_rows]
	colors = [_comparison_row_color(row) for row in plot_rows]
	fig_width = max(style.figsize.metric[0], 1.1 * max(len(plot_rows), 1))
	figsize = (fig_width, style.figsize.metric[1])
	fig, axis = plt.subplots(figsize=figsize, facecolor='white')
	if plot_rows:
		positions = list(range(len(plot_rows)))
		axis.bar(positions, values, color=colors, edgecolor='black', linewidth=0.6)
		axis.set_xticks(positions, labels=labels, rotation=35, ha='right')
	else:
		axis.text(0.5, 0.5, 'No metrics', ha='center', va='center')
		axis.set_xticks([])
	axis.set_title(title, fontsize=style.font_sizes.title, pad=6)
	axis.set_xlabel('Feature source', fontsize=style.font_sizes.axis_label)
	axis.set_ylabel(ylabel, fontsize=style.font_sizes.axis_label)
	axis.set_ylim(0.0, 1.0)
	axis.tick_params(axis='both', labelsize=style.font_sizes.tick)
	axis.grid(axis='y', color='#D9D9D9', linewidth=0.7)
	axis.set_axisbelow(True)
	axis.spines['top'].set_visible(False)
	axis.spines['right'].set_visible(False)
	fig.tight_layout()
	fig.savefig(output_png, dpi=dpi, facecolor='white', bbox_inches='tight')
	plt.close(fig)

def _save_per_class_f1_comparison(
	rows: Sequence[Mapping[str, object]],
	*,
	output_png: Path,
	plt: object,
	dpi: int,
	style: F3LithologyComparisonFigureStyle,
) -> None:
	class_columns = [
		key
		for key in sorted(
		{
			key
			for row in rows
			for key in row
			if key.startswith('class_') and key.endswith('_f1')
		},
		key=_class_metric_sort_key,
	)
		if key.startswith('class_') and key.endswith('_f1')
	]
	plot_rows = [
		row
		for row in rows
		if any(_float_or_none(row.get(column)) is not None for column in class_columns)
	]
	fig_width = max(
		style.figsize.per_class[0],
		1.05 * max(len(class_columns), 1) + 0.35 * max(len(plot_rows), 1),
	)
	figsize = (fig_width, style.figsize.per_class[1])
	fig, axis = plt.subplots(figsize=figsize, facecolor='white')
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
			rotation=45,
			ha='right',
		)
		axis.legend(
			frameon=False,
			fontsize=style.font_sizes.legend,
			loc='upper left',
			bbox_to_anchor=(1.01, 1.0),
			borderaxespad=0.0,
		)
	else:
		axis.text(0.5, 0.5, 'No per-class F1 metrics', ha='center', va='center')
		axis.set_xticks([])
	axis.set_title('Per-class F1', fontsize=style.font_sizes.title, pad=6)
	axis.set_xlabel('Class', fontsize=style.font_sizes.axis_label)
	axis.set_ylabel('F1', fontsize=style.font_sizes.axis_label)
	axis.set_ylim(0.0, 1.0)
	axis.tick_params(axis='both', labelsize=style.font_sizes.tick)
	axis.grid(axis='y', color='#D9D9D9', linewidth=0.7)
	axis.set_axisbelow(True)
	axis.spines['top'].set_visible(False)
	axis.spines['right'].set_visible(False)
	fig.tight_layout()
	fig.savefig(output_png, dpi=dpi, facecolor='white', bbox_inches='tight')
	plt.close(fig)

def _comparison_row_label(row: Mapping[str, object]) -> str:
	feature_kind = str(row.get('feature_kind') or '')
	display_names = {
		'pretrained_encoder': 'Pretrained',
		'z_only': 'Z only',
		'xyz_coordinates': 'XYZ only',
		'amplitude_stats': 'Amplitude stats',
		'random_encoder': 'Random encoder',
	}
	return display_names.get(
		feature_kind,
		_short_plot_label(
			_first_non_empty(
				feature_kind,
				row.get('BASELINE_TAG'),
				row.get('MODEL_TAG'),
				'unknown',
			),
		),
	)

def _comparison_row_color(row: Mapping[str, object]) -> str:
	return {
		'pretrained_encoder': '#2563EB',
		'z_only': '#6B7280',
		'xyz_coordinates': '#059669',
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
			return f'{class_id}\n{_wrap_plot_label(class_name, width=14)}'
	return f'class {class_id}'

def _short_plot_label(label: object) -> str:
	text = str(label or '').strip().replace('_', ' ')
	return text if len(text) <= 18 else f'{text[:15]}...'

def _wrap_plot_label(label: str, *, width: int) -> str:
	lines = textwrap.wrap(label, width=width, break_long_words=False)
	if not lines:
		return label
	if len(lines) <= 2:
		return '\n'.join(lines)
	return '\n'.join((*lines[:2], '...'))

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

__all__ = [
	'COMPARISON_FIGURE_NAMES',
	'F3LithologyComparisonFigureFontSizes',
	'F3LithologyComparisonFigureSizes',
	'F3LithologyComparisonFigureStyle',
	'_class_f1_column_label',
	'_comparison_figure_paths',
	'_comparison_row_color',
	'_comparison_row_label',
	'_figure_record',
	'_figure_summary',
	'_save_metric_comparison_bar',
	'_save_per_class_f1_comparison',
	'_short_plot_label',
	'_wrap_plot_label',
	'_write_comparison_figures',
	'default_f3_lithology_comparison_figure_style',
]
