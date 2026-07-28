"""Summarize and publish the M5-U original-split screening result."""

from __future__ import annotations

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_soft_posterior import (
	f3_lithology_voxel_label_budget_soft_posterior_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_soft_posterior_results import (
	inspect_f3_lithology_voxel_label_budget_soft_posterior_results,
	summarize_f3_lithology_voxel_label_budget_soft_posterior,
)


def build_parser() -> object:
	"""Build the M5-U aggregate-stage parser."""
	return build_config_parser(
		'Summarize M5-U soft-posterior screening.',
		config_required=True,
		dry_run_help='Live-revalidate and reaggregate without writing.',
	)


def main() -> None:
	"""Validate or publish the M5-U original-split summary."""
	args = build_parser().parse_args()
	raw = load_config_for_cli(parse_config_path(args), loader=load_config)
	if set(raw) == {'run_config'}:
		raw = load_config(raw['run_config'])
	config = f3_lithology_voxel_label_budget_soft_posterior_config_from_mapping(raw)
	if args.dry_run:
		inspection = inspect_f3_lithology_voxel_label_budget_soft_posterior_results(
			config
		)
		print(f'overall_status: {inspection["decisions"]["overall_status"]}')
		print('execution: dry-run; summary and publish skipped')
		return
	result = summarize_f3_lithology_voxel_label_budget_soft_posterior(config)
	print(f'soft_posterior_results.summary: {result["summary_json"]}')
	print(f'soft_posterior_results.status: {result["decisions"]["overall_status"]}')


if __name__ == '__main__':
	main()
