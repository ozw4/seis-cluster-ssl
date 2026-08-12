# ruff: noqa: CPY001
"""Summarize the complete paired Parihaka Channel benchmark."""

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
from seis_ssl_cluster.parihaka.channel_results import (
	channel_summary_config_from_mapping,
	inspect_channel_benchmark_results,
	summarize_channel_benchmark,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'parihaka/facies_benchmark_v1/30_channel_benchmark_v1'
	/ '06_channel_benchmark.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the summary parser."""
	return build_config_parser(
		'Summarize all 30 paired Parihaka Channel jobs.', default_config=DEFAULT_CONFIG
	)


def main() -> None:
	"""Validate completeness or write three direct summary files."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = channel_summary_config_from_mapping(raw)
	jobs = inspect_channel_benchmark_results(config)
	if args.dry_run:
		print(f'complete_jobs: {len(jobs)}')
		print(f'output_dir: {config.output_dir}')
		print('execution: dry-run; no files written')
		return
	for path in summarize_channel_benchmark(config):
		print(f'output: {path}')


if __name__ == '__main__':
	main()
