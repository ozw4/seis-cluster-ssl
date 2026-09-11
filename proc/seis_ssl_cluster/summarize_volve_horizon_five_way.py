'''Audit or summarize the 75 Volve horizon five-way decoder jobs.'''

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.cli import load_config_for_cli, resolve_config_for_cli
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.volve.horizon_five_way_config import (
	volve_horizon_five_way_config_from_mapping,
)
from seis_ssl_cluster.volve.horizon_five_way_results import (
	inspect_volve_horizon_five_way_results,
	summarize_volve_horizon_five_way,
)


def build_parser() -> argparse.ArgumentParser:
	'''Build the completeness-audit and summary parser.'''
	parser = argparse.ArgumentParser(
		description='Audit or summarize all 75 Volve horizon five-way jobs.'
	)
	parser.add_argument(
		'--config',
		type=Path,
		required=True,
		help='Path to the canonical Volve five-way comparison YAML.',
	)
	parser.add_argument(
		'--check-only',
		action='store_true',
		help='Audit all 75 jobs without writing summary files.',
	)
	return parser


def main() -> None:
	'''Run the read-only audit or atomically write the five summaries.'''
	args = build_parser().parse_args()
	raw = load_config_for_cli(args.config, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=volve_horizon_five_way_config_from_mapping,
		config_path=args.config,
	)
	if args.check_only:
		report = inspect_volve_horizon_five_way_results(config)
		print(f'complete_jobs: {report["complete_jobs"]}')
		print(f'model_order: {", ".join(config.model_ids)}')
		print('execution: check-only; summary files skipped')
		return
	result = summarize_volve_horizon_five_way(config)
	print(f'complete_jobs: {result["complete_jobs"]}')
	for output in result['outputs']:
		print(f'output: {output}')


if __name__ == '__main__':
	main()
