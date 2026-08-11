# ruff: noqa: CPY001
"""Thin entrypoint for preparing the Parihaka amplitude volume."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import argparse

from seis_ssl_cluster.cli import (
	add_overwrite_argument,
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.parihaka import (
	ParihakaPrepareVolumeConfig,
	inspect_parihaka_preparation,
	parihaka_prepare_volume_config_from_mapping,
	prepare_parihaka_volume,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'parihaka'
	/ 'facies_benchmark_v1'
	/ '10_prepare'
	/ '01_prepare_parihaka_volume.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the Parihaka preparation CLI parser."""
	parser = build_config_parser(
		'Prepare the amplitude-only Parihaka XYZ NPY and direct metadata.',
		default_config=DEFAULT_CONFIG,
		dry_run_help=(
			'Validate roots, output state, ZIP inventory, and NPY header without '
			'writing or extracting the large member.'
		),
	)
	add_overwrite_argument(
		parser,
		help_text='Replace one existing, complete, hash-consistent output set.',
	)
	return parser


def main() -> None:
	"""Validate or run bounded-memory Parihaka amplitude preparation."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=parihaka_prepare_volume_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		inspection = inspect_parihaka_preparation(config, overwrite=args.overwrite)
		_print_dry_run(config, inspection.shape_zxy)
		return
	result = prepare_parihaka_volume(config, overwrite=args.overwrite)
	print(f'parihaka_prepare.source_npz: {config.inputs.amplitude_npz}')
	print(f'parihaka_prepare.source_sha256: {result.source_sha256}')
	print(f'parihaka_prepare.source_statistics: {result.source_statistics.to_dict()}')
	print(f'parihaka_prepare.amplitude_npy: {result.amplitude_npy}')
	print(f'parihaka_prepare.output_sha256: {result.output_sha256}')
	print(f'parihaka_prepare.shape_xyz: {result.shape_xyz}')
	print(f'parihaka_prepare.dtype: {result.dtype}')
	print(f'parihaka_prepare.order: {result.order}')
	print(f'parihaka_prepare.output_statistics: {result.output_statistics.to_dict()}')
	print(f'parihaka_prepare.manifest: {result.manifest}')
	print(f'parihaka_prepare.path_list: {result.path_list}')
	print(f'parihaka_prepare.normalization_stats: {result.normalization_stats}')
	print(f'parihaka_prepare.metadata: {result.metadata}')


def _print_dry_run(
	config: ParihakaPrepareVolumeConfig,
	shape_zxy: tuple[int, int, int],
) -> None:
	shape_xyz = (shape_zxy[1], shape_zxy[2], shape_zxy[0])
	print(f'paths.parihaka_root: {config.paths.parihaka_root}')
	print(f'paths.artifact_root: {config.paths.artifact_root}')
	print(f'inputs.amplitude_npz: {config.inputs.amplitude_npz}')
	print(f'source.shape_zxy: {shape_zxy}')
	print(f'source.dtype: {config.source.dtype}')
	print(f'source.fortran_order: {str(config.source.fortran_order).lower()}')
	print(f'conversion.transpose_axes: {config.conversion.transpose_axes}')
	print(f'conversion.chunk_size_z: {config.conversion.chunk_size_z}')
	print(f'destination.shape_xyz: {shape_xyz}')
	for path in config.outputs.files():
		print(f'planned_output: {path}')
	print('parihaka_prepare.execution: dry-run; no files written')


if __name__ == '__main__':
	main()
