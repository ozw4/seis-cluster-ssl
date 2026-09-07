"""Audit or summarize a transferred Parihaka Channel representation recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.cli import load_config_for_cli, resolve_config_for_cli
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.parihaka.channel_recipe_transfer_results import (
	channel_recipe_transfer_summary_config_from_mapping,
	inspect_channel_recipe_transfer_results,
	summarize_channel_recipe_transfer,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments/parihaka/facies_benchmark_v1'
	/ '39_channel_local_bt_rot90_asym_g060_v1/30_downstream'
	/ '01_channel_candidate.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the completeness-audit and summary parser."""
	parser = argparse.ArgumentParser(
		description=(
			'Audit or summarize a transferred Parihaka Channel recipe against '
			'the current learned model and legacy random baseline.'
		)
	)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	parser.add_argument(
		'--check-only',
		'--dry-run',
		dest='check_only',
		action='store_true',
		help='Audit every required job without writing summary files.',
	)
	return parser


def main() -> None:
	"""Run the read-only audit or write the three summary files."""
	args = build_parser().parse_args()
	raw = load_config_for_cli(args.config, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=channel_recipe_transfer_summary_config_from_mapping,
		config_path=args.config,
	)
	if args.check_only:
		report = inspect_channel_recipe_transfer_results(config)
		print(f'candidate_model_id: {report["candidate_model_id"]}')
		print(f'complete_learned_jobs: {report["complete_learned_jobs"]}')
		print(f'complete_legacy_jobs: {report["complete_legacy_jobs"]}')
		print(f'compared_cells: {report["compared_cells"]}')
		print('execution: check-only; summary files skipped')
		return
	for path in summarize_channel_recipe_transfer(config):
		print(f'output: {path}')


if __name__ == '__main__':
	main()
