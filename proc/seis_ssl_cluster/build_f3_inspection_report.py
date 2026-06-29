"""Build the consolidated F3 facies benchmark inspection report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import (
	load_config,
	resolve_f3_facies_inspection_config,
)
from seis_ssl_cluster.config.schema import STAGE_F3_INSPECTION_REPORT
from seis_ssl_cluster.f3 import (
	F3InspectionReportConfig,
	build_f3_inspection_report,
)
from seis_ssl_cluster.utils.cli import parse_config_args

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '00_inspection'
	/ '07_build_inspection_report.yaml'
)


def main() -> None:
	"""Build the consolidated F3 inspection Markdown and JSON reports."""
	args = parse_config_args(
		'Build the consolidated F3 facies benchmark inspection report.',
		DEFAULT_CONFIG,
	)
	config = resolve_f3_facies_inspection_config(
		load_config(args.config),
		stage=STAGE_F3_INSPECTION_REPORT,
	)
	paths = _required_mapping(config, 'paths')
	output_root = _required_mapping(config, 'outputs')
	inspection = _required_mapping(config, 'inspection')
	report_config = _report_config(output_root, inspection)

	if args.dry_run:
		_print_summary(
			config=config,
			f3_root=Path(_required_str(paths, 'f3_root')),
			report_config=report_config,
		)
		print('execution: dry-run; F3 inspection report skipped')
		return

	result = build_f3_inspection_report(report_config)
	readiness = _required_mapping(result.payload, 'downstream_readiness')
	warnings = result.payload.get('warnings', [])
	warning_count = len(warnings) if isinstance(warnings, Sequence) else 0
	print(f'f3_inspection_report.readiness: {readiness.get("status")}')
	print(f'f3_inspection_report.warning_count: {warning_count}')
	print(f'wrote F3 inspection report Markdown: {result.report_markdown}')
	print(f'wrote F3 inspection report JSON: {result.report_json}')


def _report_config(
	outputs: Mapping[str, object],
	inspection: Mapping[str, object],
) -> F3InspectionReportConfig:
	return F3InspectionReportConfig(
		inspection_dir=Path(_required_str(outputs, 'inspection_dir')),
		file_inventory_json=Path(_required_str(inspection, 'file_inventory_json')),
		class_info_json=Path(_required_str(inspection, 'class_info_json')),
		segy_geometry_json=Path(_required_str(inspection, 'segy_geometry_json')),
		seismic_amplitude_stats_json=Path(
			_required_str(inspection, 'seismic_amplitude_stats_json'),
		),
		label_unique_values_json=Path(
			_required_str(inspection, 'label_unique_values_json'),
		),
		png_label_summary_json=Path(
			_required_str(inspection, 'png_label_summary_json'),
		),
		png_label_inventory_json=Path(
			_required_str(inspection, 'png_label_inventory_json'),
		),
		quicklook_metadata_json=Path(
			_required_str(inspection, 'quicklook_metadata_json'),
		),
		label_consistency_json=Path(
			_required_str(inspection, 'label_consistency_json'),
		),
		tokenization_preview_json=Path(
			_required_str(inspection, 'tokenization_preview_json'),
		),
		output_markdown=Path(_required_str(inspection, 'output_markdown')),
		output_json=Path(_required_str(inspection, 'output_json')),
		figure_paths=tuple(
			Path(path)
			for path in _string_sequence(
				inspection.get('figure_paths', []),
				'inspection.figure_paths',
			)
		)
		or F3InspectionReportConfig.figure_paths,
	)


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, Any]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _required_str(parent: Mapping[str, object], key: str) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'{label} must be a sequence of strings; got {value!r}'
		raise TypeError(msg)
	values = tuple(value)
	if not all(isinstance(item, str) and item for item in values):
		msg = f'{label} must contain strings; got {value!r}'
		raise TypeError(msg)
	return values


def _print_summary(
	*,
	config: Mapping[str, object],
	f3_root: Path,
	report_config: F3InspectionReportConfig,
) -> None:
	output_root = _required_mapping(config, 'outputs')
	print(f'stage: {config.get("stage")}')
	print(f'paths.f3_root: {f3_root}')
	print(f'outputs.inspection_dir: {output_root.get("inspection_dir")}')
	print(f'inspection.file_inventory_json: {report_config.file_inventory_json}')
	print(f'inspection.class_info_json: {report_config.class_info_json}')
	print(f'inspection.segy_geometry_json: {report_config.segy_geometry_json}')
	print(
		'inspection.seismic_amplitude_stats_json: '
		f'{report_config.seismic_amplitude_stats_json}',
	)
	print(
		'inspection.label_unique_values_json: '
		f'{report_config.label_unique_values_json}',
	)
	print(
		'inspection.png_label_summary_json: '
		f'{report_config.png_label_summary_json}',
	)
	print(
		'inspection.png_label_inventory_json: '
		f'{report_config.png_label_inventory_json}',
	)
	print(
		'inspection.quicklook_metadata_json: '
		f'{report_config.quicklook_metadata_json}',
	)
	print(
		'inspection.label_consistency_json: '
		f'{report_config.label_consistency_json}',
	)
	print(
		'inspection.tokenization_preview_json: '
		f'{report_config.tokenization_preview_json}',
	)
	print(f'inspection.output_markdown: {report_config.output_markdown}')
	print(f'inspection.output_json: {report_config.output_json}')
	print(
		'inspection.figure_paths: '
		f'{", ".join(path.as_posix() for path in report_config.figure_paths)}',
	)


if __name__ == '__main__':
	main()
