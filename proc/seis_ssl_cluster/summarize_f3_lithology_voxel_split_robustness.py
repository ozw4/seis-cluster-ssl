"""Summarize paired M1/M2-A voxel robustness with split as the unit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_robustness import (
	f3_lithology_voxel_split_summary_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_robustness import (
	inspect_f3_lithology_voxel_split_robustness,
	summarize_f3_lithology_voxel_split_robustness,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the split-level summary parser."""
	return build_config_parser(
		'Summarize paired F3 voxel split robustness.',
		config_required=True,
		dry_run_help='Validate all run identities and print the summary plan.',
	)


def main() -> None:
	"""Resolve and optionally write the split-level summary."""
	args = build_parser().parse_args()
	raw = load_config_for_cli(parse_config_path(args), loader=load_config)
	config = f3_lithology_voxel_split_summary_config_from_mapping(raw)
	if args.dry_run:
		inspection = inspect_f3_lithology_voxel_split_robustness(config)
		print('stage: summarize_f3_lithology_voxel_split_robustness')
		print(f'inputs.v0_run_manifest: {config.v0_run_manifest}')
		print(f'inputs.v1_run_manifest: {config.v1_run_manifest}')
		print(f'inputs.original_summary_dir: {config.original_summary_dir}')
		print(f'outputs.root: {config.suite_root / "reports"}')
		print(f'publish.enabled: {config.publish.enabled}')
		print(f'publish.output_dir: {config.publish.output_dir}')
		print('statistical_unit: split')
		print(f'provisional_status: {inspection.status}')
		print('execution: dry-run')
		return
	result = summarize_f3_lithology_voxel_split_robustness(config)
	print(f'voxel_split_robustness.summary: {result.summary_json}')
	print(f'voxel_split_robustness.status: {result.status}')
	if result.published_files:
		print(f'voxel_split_robustness.published: {result.published_files[0].parent}')


if __name__ == '__main__':
	main()
