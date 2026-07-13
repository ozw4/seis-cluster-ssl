"""CLI for the F3 original-split voxel benchmark summary."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_results import (
	f3_lithology_voxel_results_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_results import (
	F3LithologyVoxelResultsConfig,
	summarize_f3_lithology_voxel_results,
	validate_f3_lithology_voxel_results_inputs,
)

if TYPE_CHECKING:
	import argparse
	from collections.abc import Mapping

STAGE = 'summarize_f3_lithology_voxel_results'
DEFAULT_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/90_f3_voxel_results/'
	'01_summarize_original_split.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the summary CLI parser."""
	return build_config_parser(
		'Consolidate the six original-split F3 voxel evaluations.',
		default_config=DEFAULT_CONFIG,
		dry_run_help=(
			'Validate configuration and print planned outputs without writing.'
		),
	)


def _config_from_mapping(
	raw_config: Mapping[str, object], *, config_path: Path
) -> F3LithologyVoxelResultsConfig:
	return resolve_config_for_cli(
		raw_config,
		resolver=f3_lithology_voxel_results_config_from_mapping,
		config_path=config_path,
	)


def _print_summary(config: F3LithologyVoxelResultsConfig) -> None:
	print(f'stage: {STAGE}')
	for run in config.runs:
		print(f'runs.{run.key}: {run.input_dir}')
	print(f'outputs.summary_json: {config.output_dir / "voxel_results_summary.json"}')
	print(f'outputs.summary_markdown: {config.output_dir / "voxel_results_summary.md"}')
	print(f'publish.enabled: {config.publish.enabled}')


def main() -> None:
	"""Run the original-split voxel summary."""
	parser = build_parser()
	args = parser.parse_args()
	raw = load_config_for_cli(args.config, loader=load_config)
	config = _config_from_mapping(raw, config_path=args.config)
	if args.dry_run:
		validate_f3_lithology_voxel_results_inputs(config)
		_print_summary(config)
		print('execution: dry-run; F3 voxel result summary skipped')
		return
	result = summarize_f3_lithology_voxel_results(config)
	print(f'f3_lithology_voxel_results.summary_json: {result.summary_json}')
	print(f'f3_lithology_voxel_results.summary_markdown: {result.summary_markdown}')
	print(f'f3_lithology_voxel_results.decoder_value: {result.decoder_value}')
	print(f'f3_lithology_voxel_results.m2a_vs_m1_voxel: {result.m2a_vs_m1_voxel}')
	if result.publish_manifest is not None:
		print(f'wrote publish manifest: {result.publish_manifest.manifest_path}')


if __name__ == '__main__':
	main()
