# ruff: noqa: CPY001
"""Write per-section Parihaka Channel label statistics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import argparse

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.parihaka.channel_data import (
	channel_inspection_config_from_mapping,
	inspect_prepared_labels,
	write_section_statistics,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'parihaka/facies_benchmark_v1/30_channel_benchmark_v1'
	/ '01_prepare_channel_labels.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the inspection parser."""
	return build_config_parser(
		'Inspect every Parihaka X and Y section without selecting lines.',
		default_config=DEFAULT_CONFIG,
	)


def main() -> None:
	"""Validate or write section statistics."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = channel_inspection_config_from_mapping(raw)
	labels = inspect_prepared_labels(config.labels)
	if args.dry_run:
		print(f'labels: {config.labels}')
		print(f'shape_xyz: {labels.shape}')
		print(f'planned_output: {config.output_csv}')
		print('execution: dry-run; no files written')
		return
	row_count = write_section_statistics(config)
	print(f'section_counts: {config.output_csv}')
	print(f'row_count: {row_count}')


if __name__ == '__main__':
	main()
