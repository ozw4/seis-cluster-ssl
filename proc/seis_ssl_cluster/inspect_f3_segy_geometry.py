"""Inspect F3 facies benchmark SEGY geometry, axes, and statistics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import (
	load_config,
	resolve_f3_facies_inspection_config,
)
from seis_ssl_cluster.config.schema import STAGE_F3_SEGY_GEOMETRY
from seis_ssl_cluster.f3 import (
	F3SegyInspectionOutputConfig,
	inspect_f3_segy_files,
	write_f3_segy_inspection_outputs,
)
from seis_ssl_cluster.utils.cli import parse_config_args

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '00_inspection'
	/ '02_inspect_segy_geometry.yaml'
)


def main() -> None:
	"""Inspect F3 SEGY files and write geometry/statistics artifacts."""
	args = parse_config_args(
		'Inspect F3 facies benchmark SEGY geometry and statistics.',
		DEFAULT_CONFIG,
	)
	config = resolve_f3_facies_inspection_config(
		load_config(args.config),
		stage=STAGE_F3_SEGY_GEOMETRY,
	)
	paths = _required_mapping(config, 'paths')
	inspection = _required_mapping(config, 'inspection')
	f3_root = Path(_required_str(paths, 'f3_root'))
	outputs = _output_config(inspection)
	candidate_extensions = _string_sequence(
		inspection.get('candidate_extensions', ['.segy', '.sgy']),
		'inspection.candidate_extensions',
	)

	if args.dry_run:
		_print_summary(
			config=config,
			f3_root=f3_root,
			outputs=outputs,
			candidate_extensions=candidate_extensions,
		)
		print('execution: dry-run; F3 SEGY inspection skipped')
		return

	result = inspect_f3_segy_files(
		f3_root,
		candidate_extensions=candidate_extensions,
	)
	write_f3_segy_inspection_outputs(result, outputs)
	print(f'f3_segy.seismic_shape: {result.seismic.geometry.cube_shape}')
	print(f'f3_segy.label_shape: {result.label.geometry.cube_shape}')
	print(
		'f3_segy.shape_matches: '
		f'{result.seismic.geometry.cube_shape == result.label.geometry.cube_shape}',
	)
	print(
		'f3_segy.label_unique_values: '
		f'{result.label_unique_values["unique_values"]}',
	)
	print(f'wrote SEGY metadata: {outputs.metadata_json}')
	print(f'wrote SEGY geometry: {outputs.geometry_json}')
	print(f'wrote SEGY summary: {outputs.summary_markdown}')
	print(
		'wrote seismic amplitude stats: '
		f'{outputs.seismic_amplitude_stats_json}',
	)
	print(f'wrote label unique values: {outputs.label_unique_values_json}')


def _output_config(
	inspection: Mapping[str, object],
) -> F3SegyInspectionOutputConfig:
	_required_str(inspection, 'segy_dir')
	return F3SegyInspectionOutputConfig(
		metadata_json=Path(_required_str(inspection, 'metadata_json')),
		summary_markdown=Path(_required_str(inspection, 'summary_markdown')),
		seismic_amplitude_stats_json=Path(
			_required_str(inspection, 'seismic_amplitude_stats_json'),
		),
		label_unique_values_json=Path(
			_required_str(inspection, 'label_unique_values_json'),
		),
		geometry_json=Path(_required_str(inspection, 'output_json')),
		geometry_csv=Path(_required_str(inspection, 'output_csv')),
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
	outputs: F3SegyInspectionOutputConfig,
	candidate_extensions: Sequence[str],
) -> None:
	paths = _required_mapping(config, 'paths')
	output_root = _required_mapping(config, 'outputs')
	inspection = _required_mapping(config, 'inspection')
	print(f'stage: {config.get("stage")}')
	print(f'paths.f3_root: {f3_root}')
	print(f'paths.artifact_root: {paths.get("artifact_root")}')
	print(f'outputs.inspection_dir: {output_root.get("inspection_dir")}')
	print(f'inspection.inventory_json: {inspection.get("inventory_json")}')
	print(f'inspection.segy_dir: {inspection.get("segy_dir")}')
	print(
		'inspection.candidate_extensions: '
		f'{", ".join(candidate_extensions)}',
	)
	print(f'inspection.output_json: {outputs.geometry_json}')
	print(f'inspection.output_csv: {outputs.geometry_csv}')
	print(f'inspection.metadata_json: {outputs.metadata_json}')
	print(f'inspection.summary_markdown: {outputs.summary_markdown}')
	print(
		'inspection.seismic_amplitude_stats_json: '
		f'{outputs.seismic_amplitude_stats_json}',
	)
	print(
		'inspection.label_unique_values_json: '
		f'{outputs.label_unique_values_json}',
	)


if __name__ == '__main__':
	main()
