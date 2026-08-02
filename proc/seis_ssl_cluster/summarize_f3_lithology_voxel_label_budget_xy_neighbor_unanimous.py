"""Summarize the unanimous XY-neighbour original-split screening result."""

from __future__ import annotations

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
)
from seis_ssl_cluster.config import (
	f3_lithology_voxel_label_budget_xy_neighbor_unanimous as unanimous_config,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_xy_neighbor_unanimous_results as unanimous_results,
)


def build_parser() -> object:
	"""Build the unanimous aggregate-stage parser."""
	return build_config_parser(
		'Summarize XY-neighbour-unanimous original-split screening.',
		config_required=True,
		dry_run_help='Live-revalidate and reaggregate without writing.',
	)


def main() -> None:
	"""Validate or publish the paired unanimous summary."""
	args = build_parser().parse_args()
	raw = load_config_for_cli(parse_config_path(args), loader=load_config)
	if set(raw) == {'run_config'}:
		raw = load_config(raw['run_config'])
	resolver = (
		unanimous_config
		.f3_lithology_voxel_label_budget_xy_neighbor_unanimous_config_from_mapping
	)
	config = resolver(raw)
	if args.dry_run:
		inspection = (
			unanimous_results.inspect_f3_lithology_voxel_label_budget_xy_neighbor_unanimous_results(
				config
			)
		)
		print(f'overall_status: {inspection["decisions"]["overall_status"]}')
		print('execution: dry-run; summary and publish skipped')
		return
	summarize = (
		unanimous_results
		.summarize_f3_lithology_voxel_label_budget_xy_neighbor_unanimous
	)
	result = summarize(config)
	print(f'xy_neighbor_unanimous_results.summary: {result["summary_json"]}')
	print(
		'xy_neighbor_unanimous_results.status: '
		f'{result["decisions"]["overall_status"]}'
	)


if __name__ == '__main__':
	main()
