"""Summarize the complete paired Parihaka Channel end-to-end benchmark."""

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
from seis_ssl_cluster.parihaka.channel_end_to_end_results import (
	channel_end_to_end_summary_config_from_mapping,
	inspect_channel_end_to_end_results,
	summarize_channel_end_to_end,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'parihaka/facies_benchmark_v1/31_channel_end_to_end_v1'
	/ '01_channel_end_to_end.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the end-to-end summary parser."""
	return build_config_parser(
		'Summarize all 30 paired Parihaka Channel end-to-end jobs.',
		default_config=DEFAULT_CONFIG,
	)


def main() -> None:
	"""Validate completeness or write three direct summary files."""
	args = build_parser().parse_args()
	raw = load_config_for_cli(parse_config_path(args), loader=load_config)
	config = channel_end_to_end_summary_config_from_mapping(raw)
	jobs = inspect_channel_end_to_end_results(config)
	if args.dry_run:
		print(f'complete_jobs: {len(jobs)}')
		print(f'output_dir: {config.output_dir}')
		print('execution: dry-run; no files written')
		return
	for path in summarize_channel_end_to_end(config):
		print(f'output: {path}')


if __name__ == '__main__':
	main()
