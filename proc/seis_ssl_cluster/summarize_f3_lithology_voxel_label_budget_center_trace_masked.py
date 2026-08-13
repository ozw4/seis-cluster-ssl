"""Summarize the center-trace masked original-split screening result."""

from __future__ import annotations

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
)
from seis_ssl_cluster.config import (
	f3_lithology_voxel_label_budget_center_trace_masked as center_config,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_center_trace_masked_results as center_results,
)


def build_parser() -> object:
	"""Build the center-trace masked aggregate-stage parser."""
	return build_config_parser(
		'Summarize center-trace masked original-split screening.',
		config_required=True,
		dry_run_help='Live-revalidate and reaggregate without writing.',
	)


def main() -> None:
	"""Validate or publish the paired center-trace masked summary."""
	args = build_parser().parse_args()
	raw = load_config_for_cli(parse_config_path(args), loader=load_config)
	if set(raw) == {'run_config'}:
		raw = load_config(raw['run_config'])
	resolver = center_config.config_from_mapping
	config = resolver(raw)
	if args.dry_run:
		inspect_results = center_results.inspect_results
		inspection = inspect_results(config)
		print(f'overall_status: {inspection["decisions"]["overall_status"]}')
		print('execution: dry-run; summary and publish skipped')
		return
	result = (
		center_results.summarize_f3_lithology_voxel_label_budget_center_trace_masked(
			config
		)
	)
	print(f'center_trace_masked_results.summary: {result["summary_json"]}')
	print(
		f'center_trace_masked_results.status: {result["decisions"]["overall_status"]}'
	)


if __name__ == '__main__':
	main()
