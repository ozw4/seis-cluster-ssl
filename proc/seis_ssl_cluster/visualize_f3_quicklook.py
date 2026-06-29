"""Create F3 facies benchmark seismic, label, and overlay quicklook figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import (
	load_config,
	resolve_f3_facies_inspection_config,
)
from seis_ssl_cluster.config.schema import STAGE_F3_QUICKLOOK
from seis_ssl_cluster.f3 import (
	F3QuicklookFigureConfig,
	F3QuicklookOutputConfig,
	inspect_f3_png_labels,
	inspect_f3_segy_files,
	write_f3_quicklook_outputs,
)
from seis_ssl_cluster.utils.cli import parse_config_args

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '00_inspection'
	/ '04_make_quicklook_figures.yaml'
)


def main() -> None:
	"""Write F3 quicklook figures and JSON metadata sidecars."""
	args = parse_config_args(
		'Create F3 facies benchmark quicklook figures.',
		DEFAULT_CONFIG,
	)
	config = resolve_f3_facies_inspection_config(
		load_config(args.config),
		stage=STAGE_F3_QUICKLOOK,
	)
	paths = _required_mapping(config, 'paths')
	inspection = _required_mapping(config, 'inspection')
	f3_root = Path(_required_str(paths, 'f3_root'))
	outputs = _output_config(inspection)
	figure = _figure_config(_required_mapping(inspection, 'figure'))
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
		)
		print('execution: dry-run; F3 quicklook visualization skipped')
		return

	segy = inspect_f3_segy_files(
		f3_root,
		candidate_extensions=segy_extensions,
	)
	png_labels = inspect_f3_png_labels(
		f3_root,
		candidate_extensions=png_extensions,
	)
	result = write_f3_quicklook_outputs(
		segy,
		png_labels,
		outputs,
		figure,
	)
	print(f'f3_quicklook.png_count: {len(result.png_paths)}')
	print(f'f3_quicklook.sidecar_count: {len(result.sidecar_paths)}')
	print(f'f3_quicklook.seismic_shape: {segy.seismic.geometry.cube_shape}')
	print(f'f3_quicklook.label_shape: {segy.label.geometry.cube_shape}')
	print(f'wrote F3 quicklook metadata: {result.metadata_json}')
	print(f'wrote F3 quicklook directory: {outputs.quicklook_dir}')


def _output_config(inspection: Mapping[str, object]) -> F3QuicklookOutputConfig:
	return F3QuicklookOutputConfig(
		quicklook_dir=Path(_required_str(inspection, 'quicklook_dir')),
		seismic_dir=Path(_required_str(inspection, 'seismic_dir')),
		labels_dir=Path(_required_str(inspection, 'labels_dir')),
		overlays_dir=Path(_required_str(inspection, 'overlays_dir')),
		metadata_json=Path(_required_str(inspection, 'metadata_json')),
	)


def _figure_config(figure: Mapping[str, object]) -> F3QuicklookFigureConfig:
	return F3QuicklookFigureConfig(
		dpi=_optional_positive_int(figure.get('dpi', 300), 'inspection.figure.dpi'),
		seismic_cmap=_optional_str(
			figure.get('seismic_cmap', 'gray'),
			'inspection.figure.seismic_cmap',
		),
		amplitude_clip_percentiles=_float_pair(
			figure.get('clip_percentiles', [1.0, 99.0]),
			'inspection.figure.clip_percentiles',
		),
		overlay_alpha=_optional_fraction(
			figure.get('overlay_alpha', 0.45),
			'inspection.figure.overlay_alpha',
		),
		xz_yz_origin=_optional_str(
			figure.get('xz_yz_origin', 'upper'),
			'inspection.figure.xz_yz_origin',
		),
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


def _optional_str(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		msg = f'{label} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_positive_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{label} must be a positive integer; got {value!r}'
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


def _float_pair(value: object, label: str) -> tuple[float, float]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'{label} must be a two-item sequence; got {value!r}'
		raise TypeError(msg)
	values = tuple(value)
	if len(values) != 2 or not all(
		isinstance(item, int | float) and not isinstance(item, bool) for item in values
	):
		msg = f'{label} must be a two-item numeric sequence; got {value!r}'
		raise TypeError(msg)
	low, high = float(values[0]), float(values[1])
	if not 0.0 <= low < high <= 100.0:
		msg = f'{label} must satisfy 0 <= low < high <= 100; got {value!r}'
		raise ValueError(msg)
	return low, high


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
	outputs: F3QuicklookOutputConfig,
	figure: F3QuicklookFigureConfig,
	segy_extensions: Sequence[str],
	png_extensions: Sequence[str],
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
	print(f'inspection.palette_json: {inspection.get("palette_json")}')
	print(f'inspection.quicklook_dir: {outputs.quicklook_dir}')
	print(f'inspection.seismic_dir: {outputs.seismic_dir}')
	print(f'inspection.labels_dir: {outputs.labels_dir}')
	print(f'inspection.overlays_dir: {outputs.overlays_dir}')
	print(f'inspection.metadata_json: {outputs.metadata_json}')
	print(f'inspection.candidate_extensions: {", ".join(segy_extensions)}')
	print(f'inspection.png_candidate_extensions: {", ".join(png_extensions)}')
	print(f'inspection.figure.dpi: {figure.dpi}')
	print(f'inspection.figure.seismic_cmap: {figure.seismic_cmap}')
	print(
		'inspection.figure.clip_percentiles: '
		f'{list(figure.amplitude_clip_percentiles)}',
	)
	print(f'inspection.figure.overlay_alpha: {figure.overlay_alpha}')
	print(f'inspection.figure.xz_yz_origin: {figure.xz_yz_origin}')


if __name__ == '__main__':
	main()
