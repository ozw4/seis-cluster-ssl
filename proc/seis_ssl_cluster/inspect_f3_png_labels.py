"""Inspect F3 facies benchmark PNG labels by RGB class definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import (
	load_config,
	resolve_f3_facies_inspection_config,
)
from seis_ssl_cluster.config.schema import STAGE_F3_PNG_LABELS
from seis_ssl_cluster.f3 import (
	F3PngLabelOutputConfig,
	inspect_f3_png_labels,
	write_f3_png_label_inspection_outputs,
)
from seis_ssl_cluster.utils.cli import parse_config_args

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '00_inspection'
	/ '03_inspect_png_labels.yaml'
)


def main() -> None:
	"""Inspect F3 PNG labels and write class-distribution artifacts."""
	args = parse_config_args(
		'Inspect F3 facies benchmark PNG labels.',
		DEFAULT_CONFIG,
	)
	config = resolve_f3_facies_inspection_config(
		load_config(args.config),
		stage=STAGE_F3_PNG_LABELS,
	)
	paths = _required_mapping(config, 'paths')
	inspection = _required_mapping(config, 'inspection')
	f3_root = Path(_required_str(paths, 'f3_root'))
	candidate_extensions = _string_sequence(
		inspection.get('candidate_extensions', ['.png']),
		'inspection.candidate_extensions',
	)
	allow_unknown_colors = _optional_bool(
		inspection.get('allow_unknown_colors', False),
		'inspection.allow_unknown_colors',
	)
	outputs = _output_config(inspection)

	if args.dry_run:
		_print_summary(
			config=config,
			f3_root=f3_root,
			outputs=outputs,
			candidate_extensions=candidate_extensions,
			allow_unknown_colors=allow_unknown_colors,
		)
		print('execution: dry-run; F3 PNG label inspection skipped')
		return

	result = inspect_f3_png_labels(
		f3_root,
		candidate_extensions=candidate_extensions,
		allow_unknown_colors=allow_unknown_colors,
	)
	write_f3_png_label_inspection_outputs(result, outputs)
	print(f'f3_png_labels.file_count: {len(result.files)}')
	print(f'f3_png_labels.total_pixels: {result.total_pixel_count()}')
	print(
		'f3_png_labels.unknown_pixels: '
		f'{result.total_unknown_pixel_count()}',
	)
	print(f'wrote PNG label inventory: {outputs.inventory_csv}')
	print(f'wrote PNG label class counts: {outputs.class_counts_csv}')
	print(f'wrote PNG label summary: {outputs.summary_json}')
	print(f'wrote PNG label markdown: {outputs.summary_markdown}')


def _output_config(inspection: Mapping[str, object]) -> F3PngLabelOutputConfig:
	return F3PngLabelOutputConfig(
		inventory_csv=Path(_required_str(inspection, 'inventory_csv')),
		class_counts_csv=Path(_required_str(inspection, 'class_counts_csv')),
		summary_json=Path(_required_str(inspection, 'summary_json')),
		summary_markdown=Path(_required_str(inspection, 'summary_markdown')),
		class_distribution_train_png=Path(
			_required_str(inspection, 'class_distribution_train_png'),
		),
		class_distribution_validation_png=Path(
			_required_str(inspection, 'class_distribution_validation_png'),
		),
		class_distribution_per_slice_png=Path(
			_required_str(inspection, 'class_distribution_per_slice_png'),
		),
		dpi=_optional_positive_int(inspection.get('figure_dpi', 300), 'figure_dpi'),
	)


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, Any]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _required_str(parent: Mapping[str, object], key: str) -> str:
	value = parent.get(key)
	if not isinstance(value, str):
		msg = f'{key} must be a string; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_bool(value: object, label: str) -> bool:
	if not isinstance(value, bool):
		msg = f'{label} must be a boolean; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_positive_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'inspection.{label} must be a positive integer; got {value!r}'
		raise ValueError(msg)
	return value


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'{label} must be a sequence of strings; got {value!r}'
		raise TypeError(msg)
	values = tuple(value)
	if not all(isinstance(item, str) for item in values):
		msg = f'{label} must be a sequence of strings; got {value!r}'
		raise TypeError(msg)
	return values


def _print_summary(
	*,
	config: Mapping[str, object],
	f3_root: Path,
	outputs: F3PngLabelOutputConfig,
	candidate_extensions: Sequence[str],
	allow_unknown_colors: bool,
) -> None:
	paths = _required_mapping(config, 'paths')
	output_root = _required_mapping(config, 'outputs')
	print(f'stage: {config.get("stage")}')
	print(f'paths.f3_root: {f3_root}')
	print(f'paths.artifact_root: {paths.get("artifact_root")}')
	print(f'outputs.inspection_dir: {output_root.get("inspection_dir")}')
	print(f'inspection.candidate_extensions: {", ".join(candidate_extensions)}')
	print(f'inspection.allow_unknown_colors: {allow_unknown_colors}')
	print(f'inspection.inventory_csv: {outputs.inventory_csv}')
	print(f'inspection.class_counts_csv: {outputs.class_counts_csv}')
	print(f'inspection.summary_json: {outputs.summary_json}')
	print(f'inspection.summary_markdown: {outputs.summary_markdown}')
	print(
		'inspection.class_distribution_train_png: '
		f'{outputs.class_distribution_train_png}',
	)
	print(
		'inspection.class_distribution_validation_png: '
		f'{outputs.class_distribution_validation_png}',
	)
	print(
		'inspection.class_distribution_per_slice_png: '
		f'{outputs.class_distribution_per_slice_png}',
	)
	print(f'inspection.figure_dpi: {outputs.dpi}')


if __name__ == '__main__':
	main()
