'''Audit or summarize one Volve horizon recipe-arm comparison.'''

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.cli import load_config_for_cli, resolve_config_for_cli
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.volve.horizon_recipe_arm import (
	volve_horizon_recipe_arm_config_from_mapping,
)
from seis_ssl_cluster.volve.horizon_recipe_arm_results import (
	inspect_volve_horizon_recipe_arm_results,
	summarize_volve_horizon_recipe_arm,
)


def build_parser() -> argparse.ArgumentParser:
	'''Build the completeness-audit and summary parser.'''
	parser = argparse.ArgumentParser(
		description='Audit or summarize one Volve horizon recipe-arm comparison.'
	)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument(
		'--check-only',
		action='store_true',
		help='Audit every configured cell without writing summary files.',
	)
	return parser


def main() -> None:
	'''Run the read-only audit or atomically write the paired summary.'''
	args = build_parser().parse_args()
	raw = load_config_for_cli(args.config, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=volve_horizon_recipe_arm_config_from_mapping,
		config_path=args.config,
	)
	if args.check_only:
		report = inspect_volve_horizon_recipe_arm_results(config)
		print(f'arm_id: {report["arm_id"]}')
		print(f'complete_cells: {report["complete_cells"]}')
		print(f'expected_cells: {report["expected_cells"]}')
		print('execution: check-only; summary files skipped')
		return
	result = summarize_volve_horizon_recipe_arm(config)
	summary = result['summary']
	if not isinstance(summary, dict):
		raise TypeError('recipe-arm summary payload must be a mapping')
	print(f'arm_id: {result["arm_id"]}')
	print(f'complete_cells: {result["complete_cells"]}')
	print(f'random_minus_arm_mean: {summary["mean"]:.6f}')
	print(
		'arm_better_cells: '
		f'{summary["arm_better_count"]}/{summary["n"]}'
	)
	t_statistic = summary['t_statistic']
	if t_statistic is not None:
		print(f't_statistic: {float(t_statistic):.4f}')
	for output in result['outputs']:
		print(f'output: {output}')


if __name__ == '__main__':
	main()
