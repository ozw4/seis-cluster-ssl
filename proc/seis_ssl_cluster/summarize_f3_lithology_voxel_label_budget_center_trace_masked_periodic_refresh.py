"""Summarize the periodic-refresh original-split screening result."""
# ruff: noqa: CPY001, E501

from __future__ import annotations

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
)
from seis_ssl_cluster.config import (
	f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh as periodic_config,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_center_trace_masked_periodic_refresh_results as periodic_results,
)


def build_parser() -> object:
	"""Build the periodic-refresh aggregate-stage parser."""
	return build_config_parser(
		'Summarize periodic-refresh original-split screening.',
		config_required=True,
		dry_run_help='Live-revalidate and reaggregate without writing.',
	)


def main() -> None:
	"""Validate or publish the paired periodic-refresh summary."""
	args = build_parser().parse_args()
	raw = load_config_for_cli(parse_config_path(args), loader=load_config)
	if set(raw) == {'run_config'}:
		raw = load_config(raw['run_config'])
	config = periodic_config.config_from_mapping(raw)
	if args.dry_run:
		inspection = periodic_results.inspect_results(config)
		print(f'overall_status: {inspection["decisions"]["overall_status"]}')
		print('execution: dry-run; summary and publish skipped')
		return
	result = periodic_results.summarize_results(config)
	print(f'periodic_refresh_original_results.summary: {result["summary_json"]}')
	print(
		'periodic_refresh_original_results.status: '
		f'{result["decisions"]["overall_status"]}'
	)


if __name__ == '__main__':
	main()
