'''Inspect Volve binding-v2 horizons and construct the paired split plans.'''

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import argparse

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.volve.horizon_data import (
	load_volve_horizon_data,
	resolve_volve_horizon_inspection_config,
	write_section_statistics_csv,
)
from seis_ssl_cluster.volve.horizon_layouts import (
	HorizonSplitPlan,
	build_all_horizon_split_plans,
	load_volve_horizon_layouts,
	write_plans_metadata,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[1]
	/ 'configs'
	/ 'seis_ssl_cluster'
	/ 'inspect_volve_horizon_sections.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	'''Build the binding-v2 horizon inspection parser.'''
	return build_config_parser(
		'Validate Volve horizons and construct 15 paired physical-section plans.',
		default_config=DEFAULT_CONFIG,
		dry_run_help='Validate inputs and print all plans without writing outputs.',
	)


def main() -> None:
	'''Validate inputs, display all plans, and optionally write small artifacts.'''
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=resolve_volve_horizon_inspection_config,
		config_path=config_path,
	)
	data = load_volve_horizon_data(config.volve_root)
	layouts = load_volve_horizon_layouts(config.layout_config, data)
	plans = build_all_horizon_split_plans(data, layouts)
	_print_plans(plans)
	if args.dry_run:
		print('execution: dry-run; no files written')
		return
	row_count = write_section_statistics_csv(
		data, config.section_statistics_csv
	)
	write_plans_metadata(plans, config.split_plans_json)
	print(f'section_statistics_csv: {config.section_statistics_csv}')
	print(f'section_row_count: {row_count}')
	print(f'split_plans_json: {config.split_plans_json}')
	print('execution: wrote')


def _print_plans(plans: tuple[HorizonSplitPlan, ...]) -> None:
	for plan in plans:
		identity = plan.identity()
		lines = identity['selected_physical_lines']
		counts = identity['per_horizon_counts']['train']
		print(
			f'{plan.layout_id}/{plan.data_size}: '
			f'inline={lines["inline"]} crossline={lines["crossline"]} '
			f'train={counts} identity={plan.scientific_identity_sha256}'
		)
	first = plans[0]
	print(f'condition_count: {len(plans)}')
	print(
		'twt_window: '
		f'[{first.twt_window.start_index}, '
		f'{first.twt_window.stop_index_exclusive})'
	)


if __name__ == '__main__':
	main()
