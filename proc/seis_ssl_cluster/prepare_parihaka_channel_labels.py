# ruff: noqa: CPY001
"""Prepare Parihaka Channel labels in XYZ order."""

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
	channel_label_config_from_mapping,
	inspect_source_labels,
	prepare_channel_labels,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'parihaka/facies_benchmark_v1/30_channel_benchmark_v1'
	/ '01_prepare_channel_labels.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the label preparation parser."""
	return build_config_parser(
		'Prepare Parihaka Channel labels as a C-contiguous XYZ NPY.',
		default_config=DEFAULT_CONFIG,
	)


def main() -> None:
	"""Validate or prepare the label volume."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = channel_label_config_from_mapping(raw)
	inspection = inspect_source_labels(config)
	if args.dry_run:
		for key, value in inspection.items():
			print(f'{key}: {value}')
		print(f'planned_output: {config.output_labels}')
		print(f'planned_output: {config.output_metadata}')
		print('execution: dry-run; no files written')
		return
	labels, metadata = prepare_channel_labels(config)
	print(f'labels: {labels}')
	print(f'metadata: {metadata}')


if __name__ == '__main__':
	main()
