"""Build the Parihaka Channel four-condition descriptive table."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.cli import load_config_for_cli
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.parihaka.channel_end_to_end_results import (
	channel_end_to_end_summary_config_from_mapping,
	inspect_channel_four_way_results,
	summarize_channel_four_way,
)
from seis_ssl_cluster.parihaka.channel_results import (
	channel_summary_config_from_mapping,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'parihaka/facies_benchmark_v1/31_channel_end_to_end_v1'
	/ '01_channel_end_to_end.yaml'
)
DEFAULT_FROZEN_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'parihaka/facies_benchmark_v1/30_channel_benchmark_v1'
	/ '06_channel_benchmark.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the four-condition summary parser."""
	parser = argparse.ArgumentParser(
		description='Compare paired frozen and end-to-end Parihaka Channel jobs.'
	)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	parser.add_argument(
		'--frozen-config', type=Path, default=DEFAULT_FROZEN_CONFIG
	)
	parser.add_argument('--dry-run', action='store_true')
	return parser


def main() -> None:
	"""Validate both 30-job sets or write the descriptive table."""
	args = build_parser().parse_args()
	end_raw = load_config_for_cli(args.config, loader=load_config)
	frozen_raw = load_config_for_cli(args.frozen_config, loader=load_config)
	end_config = channel_end_to_end_summary_config_from_mapping(end_raw)
	frozen_config = channel_summary_config_from_mapping(frozen_raw)
	end_jobs, frozen_jobs = inspect_channel_four_way_results(
		end_config, frozen_config
	)
	if args.dry_run:
		print(f'end_to_end_complete_jobs: {len(end_jobs)}')
		print(f'frozen_complete_jobs: {len(frozen_jobs)}')
		print(f'output_dir: {end_config.four_way_output_dir}')
		print('execution: dry-run; no files written')
		return
	for path in summarize_channel_four_way(end_config, frozen_config):
		print(f'output: {path}')


if __name__ == '__main__':
	main()
