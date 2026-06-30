"""Thin entrypoint for preparing F3 facies benchmark registry volumes."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3 import (
	F3PrepareVolumeConfig,
	f3_prepare_volume_config_from_mapping,
	prepare_f3_facies_volume,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '10_prepare'
	/ '01_prepare_f3_volume.yaml'
)


def main() -> None:
	"""Prepare F3 facies benchmark NPY volumes or print a dry-run summary."""
	parser = ArgumentParser(description='Prepare F3 facies benchmark NPY volumes.')
	parser.add_argument(
		'--config',
		type=Path,
		default=DEFAULT_CONFIG,
		help='Path to a YAML configuration file.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate the config and print a run summary without writing outputs.',
	)
	parser.add_argument(
		'--overwrite',
		action='store_true',
		help='Replace existing F3 preparation outputs.',
	)
	args = parser.parse_args()

	config = f3_prepare_volume_config_from_mapping(load_config(args.config))
	if args.dry_run:
		_print_prepare_summary(config)
		print('f3_prepare.execution: dry-run; preparation skipped')
		return

	result = prepare_f3_facies_volume(config, overwrite=args.overwrite)
	print(f'f3_prepare.seismic_npy: {result.seismic_npy}')
	print(f'f3_prepare.label_npy: {result.label_npy}')
	print(f'f3_prepare.metadata_path: {result.metadata_path}')
	print(f'f3_prepare.manifest_path: {result.manifest_path}')
	print(f'f3_prepare.split_path: {result.split_path}')
	print(f'f3_prepare.normalization_stats_path: {result.normalization_stats_path}')
	print(f'f3_prepare.shape_xyz: {result.shape_xyz}')
	print(f'f3_prepare.label_dtype: {result.label_dtype}')


def _print_prepare_summary(config: F3PrepareVolumeConfig) -> None:
	print(f'paths.f3_root: {config.paths.f3_root}')
	print(f'paths.artifact_root: {config.paths.artifact_root}')
	print(f'inputs.seismic_segy: {config.inputs.seismic_segy}')
	print(f'inputs.label_segy: {config.inputs.label_segy}')
	print(f'inputs.class_info: {config.inputs.class_info}')
	print(f'inputs.inspection_report: {config.inputs.inspection_report}')
	print(f'dataset.survey_id: {config.dataset.survey_id}')
	print(f'outputs.seismic_npy: {config.outputs.seismic_npy}')
	print(f'outputs.label_npy: {config.outputs.label_npy}')
	print(f'outputs.metadata_path: {config.outputs.metadata_path}')
	print(f'outputs.manifest_path: {config.outputs.manifest_path}')
	print(f'outputs.split_path: {config.outputs.split_path}')
	print(
		'outputs.normalization_stats_path: '
		f'{config.outputs.normalization_stats_path}',
	)
	print(
		'normalization.clipping_percentiles: '
		f'[{config.normalization.clip_low_percentile}, '
		f'{config.normalization.clip_high_percentile}]',
	)
	print(f'normalization.max_samples: {config.normalization.max_samples}')
	print(f'normalization.seed: {config.normalization.seed}')


if __name__ == '__main__':
	main()
