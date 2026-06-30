"""Check F3 PNG teacher labels against the dense SEGY label volume."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import (
	load_config,
	resolve_f3_facies_inspection_config,
)
from seis_ssl_cluster.config.schema import STAGE_F3_LABEL_CONSISTENCY
from seis_ssl_cluster.f3 import (
	F3LabelConsistencyFigureConfig,
	F3LabelConsistencyOutputConfig,
	check_f3_label_consistency,
	inspect_f3_png_labels,
	inspect_f3_segy_files,
	write_f3_label_consistency_outputs,
)
from seis_ssl_cluster.utils.cli import parse_config_args

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '00_inspection'
	/ '05_check_label_consistency.yaml'
)


def main() -> None:
	"""Check F3 PNG labels against dense SEGY labels and write QC artifacts."""
	args = parse_config_args(
		'Check F3 PNG teacher labels against dense SEGY label slices.',
		DEFAULT_CONFIG,
	)
	config = resolve_f3_facies_inspection_config(
		load_config(args.config),
		stage=STAGE_F3_LABEL_CONSISTENCY,
	)
	paths = _required_mapping(config, 'paths')
	inspection = _required_mapping(config, 'inspection')
	f3_root = Path(_required_str(paths, 'f3_root'))
	outputs = _output_config(inspection)
	figure = _figure_config(_required_mapping(inspection, 'figure'))
	consistency = _required_mapping(inspection, 'consistency')
	max_mismatch_rate = _optional_fraction(
		consistency.get('max_mismatch_rate', 0.001),
		'inspection.consistency.max_mismatch_rate',
	)
	ignore_border_samples_z = _optional_nonnegative_int(
		consistency.get('ignore_border_samples_z', 0),
		'inspection.consistency.ignore_border_samples_z',
	)
	segy_extensions = _string_sequence(
		inspection.get('candidate_extensions', ['.segy', '.sgy']),
		'inspection.candidate_extensions',
	)
	png_extensions = _string_sequence(
		inspection.get('png_candidate_extensions', ['.png']),
		'inspection.png_candidate_extensions',
	)

	if args.dry_run:
		_print_summary(
			config=config,
			f3_root=f3_root,
			outputs=outputs,
			figure=figure,
			segy_extensions=segy_extensions,
			png_extensions=png_extensions,
			max_mismatch_rate=max_mismatch_rate,
			ignore_border_samples_z=ignore_border_samples_z,
		)
		print('execution: dry-run; F3 label consistency check skipped')
		return

	segy = inspect_f3_segy_files(
		f3_root,
		candidate_extensions=segy_extensions,
	)
	png_labels = inspect_f3_png_labels(
		f3_root,
		candidate_extensions=png_extensions,
		allow_unknown_colors=True,
	)
	report = check_f3_label_consistency(
		segy,
		png_labels,
		max_mismatch_rate=max_mismatch_rate,
		ignore_border_samples_z=ignore_border_samples_z,
	)
	result = write_f3_label_consistency_outputs(report, outputs, figure)
	print(f'f3_label_consistency.png_count: {len(report.records)}')
	print(f'f3_label_consistency.passed: {report.passed}')
	print(
		'f3_label_consistency.ignore_border_samples_z: '
		f'{report.ignore_border_samples_z}',
	)
	print(
		'f3_label_consistency.max_mismatch_rate: '
		f'{report.max_observed_mismatch_rate()}',
	)
	print(
		'f3_label_consistency.total_mismatch_pixels: '
		f'{report.total_mismatch_pixel_count()}',
	)
	print(f'wrote label consistency JSON: {result.metadata_json}')
	print(f'wrote label consistency CSV: {result.report_csv}')
	print(f'wrote label consistency Markdown: {result.report_markdown}')
	print(f'wrote label consistency quicklook dir: {outputs.consistency_dir}')
	if not report.passed:
		print(
			'f3_label_consistency.threshold_exceeded: '
			f'max_mismatch_rate={report.max_mismatch_rate}',
		)
		raise SystemExit(1)


def _output_config(
	inspection: Mapping[str, object],
) -> F3LabelConsistencyOutputConfig:
	return F3LabelConsistencyOutputConfig(
		consistency_dir=Path(_required_str(inspection, 'consistency_dir')),
		output_json=Path(_required_str(inspection, 'output_json')),
		output_csv=Path(_required_str(inspection, 'output_csv')),
		report_path=Path(_required_str(inspection, 'report_path')),
	)


def _figure_config(
	figure: Mapping[str, object],
) -> F3LabelConsistencyFigureConfig:
	return F3LabelConsistencyFigureConfig(
		dpi=_optional_positive_int(figure.get('dpi', 300), 'inspection.figure.dpi'),
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


def _optional_positive_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{label} must be a positive integer; got {value!r}'
		raise ValueError(msg)
	return value


def _optional_nonnegative_int(value: object, label: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int) or value < 0:
		msg = f'{label} must be a nonnegative integer; got {value!r}'
		raise ValueError(msg)
	return value


def _optional_fraction(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'{label} must be a number in [0, 1]; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not 0.0 <= fraction <= 1.0:
		msg = f'{label} must be in [0, 1]; got {value!r}'
		raise ValueError(msg)
	return fraction


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'{label} must be a sequence of strings; got {value!r}'
		raise TypeError(msg)
	values = tuple(value)
	if not values or not all(isinstance(item, str) and item for item in values):
		msg = f'{label} must contain non-empty strings; got {value!r}'
		raise TypeError(msg)
	return values


def _print_summary(  # noqa: PLR0913
	*,
	config: Mapping[str, object],
	f3_root: Path,
	outputs: F3LabelConsistencyOutputConfig,
	figure: F3LabelConsistencyFigureConfig,
	segy_extensions: Sequence[str],
	png_extensions: Sequence[str],
	max_mismatch_rate: float,
	ignore_border_samples_z: int,
) -> None:
	paths = _required_mapping(config, 'paths')
	output_root = _required_mapping(config, 'outputs')
	inspection = _required_mapping(config, 'inspection')
	print(f'stage: {config.get("stage")}')
	print(f'paths.f3_root: {f3_root}')
	print(f'paths.artifact_root: {paths.get("artifact_root")}')
	print(f'outputs.inspection_dir: {output_root.get("inspection_dir")}')
	print(f'inspection.segy_geometry_json: {inspection.get("segy_geometry_json")}')
	print(
		'inspection.png_label_inventory_json: '
		f'{inspection.get("png_label_inventory_json")}',
	)
	print(f'inspection.consistency_dir: {outputs.consistency_dir}')
	print(f'inspection.output_json: {outputs.output_json}')
	print(f'inspection.output_csv: {outputs.output_csv}')
	print(f'inspection.report_path: {outputs.report_path}')
	print(f'inspection.candidate_extensions: {", ".join(segy_extensions)}')
	print(f'inspection.png_candidate_extensions: {", ".join(png_extensions)}')
	print(
		'inspection.consistency.max_mismatch_rate: '
		f'{max_mismatch_rate}',
	)
	print(
		'inspection.consistency.ignore_border_samples_z: '
		f'{ignore_border_samples_z}',
	)
	print(f'inspection.figure.dpi: {figure.dpi}')


if __name__ == '__main__':
	main()
