"""Summarize the four-model Parihaka Channel SSL/HMM benchmark."""

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
	CHANNEL_SSL_HMM_MODEL_IDS,
	channel_summary_config_from_mapping,
	inspect_channel_model_results,
	summarize_channel_ssl_hmm_four_way,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'parihaka/facies_benchmark_v1/32_channel_ssl_hmm_four_way_v1'
	/ '05_channel_four_way.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the SSL/HMM four-way summary parser."""
	return build_config_parser(
		'Summarize all 60 Parihaka Channel SSL/HMM jobs.',
		default_config=DEFAULT_CONFIG,
	)


def main() -> None:
	"""Validate completeness or write the four-way summary files."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = channel_summary_config_from_mapping(raw)
	jobs = inspect_channel_model_results(
		config, model_ids=CHANNEL_SSL_HMM_MODEL_IDS
	)
	if args.dry_run:
		print(f'complete_jobs: {len(jobs)}')
		print(f'models: {", ".join(CHANNEL_SSL_HMM_MODEL_IDS)}')
		print(f'output_dir: {config.output_dir}')
		print('execution: dry-run; no files written')
		return
	for path in summarize_channel_ssl_hmm_four_way(config):
		print(f'output: {path}')


if __name__ == '__main__':
	main()
