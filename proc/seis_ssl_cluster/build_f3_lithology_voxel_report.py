"""Build and optionally publish the common F3 voxel lithology report."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import argparse

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_report import (
	F3LithologyVoxelReportConfig,
	f3_lithology_voxel_report_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_report import (
	build_f3_lithology_voxel_report,
	inspect_f3_lithology_voxel_report,
)

STAGE = 'build_f3_lithology_voxel_report'
DEFAULT_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/voxel_lithology/14_build_report.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the report CLI parser."""
	return build_config_parser(
		'Build F3 voxel lithology figures and report.',
		default_config=DEFAULT_CONFIG,
		dry_run_help='Validate inputs and print the resolved report plan.',
	)


def main() -> None:
	"""Build the report or print a no-write dry-run summary."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=f3_lithology_voxel_report_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		inspect_f3_lithology_voxel_report(config)
		_print_summary(config)
		print('execution: dry-run; F3 voxel lithology report skipped')
		return
	result = build_f3_lithology_voxel_report(config)
	print(f'f3_lithology_voxel_report.markdown: {result.report_markdown}')
	print(f'f3_lithology_voxel_report.json: {result.report_json}')
	print(f'f3_lithology_voxel_report.figure_count: {len(result.figure_paths)}')
	if result.publish_manifest is not None:
		print(f'published F3 voxel report: {result.publish_manifest.output_dir}')
		print(f'wrote publish manifest: {result.publish_manifest.manifest_path}')


def _print_summary(config: F3LithologyVoxelReportConfig) -> None:
	print(f'stage: {STAGE}')
	print(f'voxel_predictions.input_dir: {config.prediction_input_dir}')
	print(f'voxel_dataset.input_dir: {config.voxel_dataset_input_dir}')
	print(f'evaluation.input_dir: {config.evaluation_input_dir}')
	print(f'outputs.output_dir: {config.output_dir}')
	print(f'report.include_confidence: {config.figure.include_confidence}')
	print(f'publish.enabled: {config.publish.enabled}')
	if config.publish.output_dir is not None:
		print(f'publish.output_dir: {config.publish.output_dir}')


if __name__ == '__main__':
	main()
