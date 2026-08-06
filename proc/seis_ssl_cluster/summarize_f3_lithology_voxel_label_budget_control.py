"""Summarize the paired current-code K=6 voxel label-budget control."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_control import (
	f3_lithology_voxel_label_budget_control_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_control import (
	inspect_f3_lithology_voxel_label_budget_control_results,
	summarize_f3_lithology_voxel_label_budget_control,
	validate_f3_lithology_voxel_label_budget_control_summary_preflight,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the current-K6 control summary CLI."""
	return build_config_parser(
		'Summarize the current-code K=6 paired voxel label-budget control.',
		config_required=True,
		dry_run_help=(
			'Validate 15 current jobs, immutable MAE/M1 references, paired '
			'identities, and M1-MAE published parity without writing.'
		),
	)


def main() -> None:
	"""Inspect or write all lightweight current-K6 control outputs."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	config = resolve_config_for_cli(
		load_config_for_cli(config_path, loader=load_config),
		resolver=f3_lithology_voxel_label_budget_control_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		inspection = inspect_f3_lithology_voxel_label_budget_control_results(config)
		validate_f3_lithology_voxel_label_budget_control_summary_preflight(config)
		print('stage: summarize_f3_lithology_voxel_label_budget_control')
		print('current_candidate_jobs: 15/15')
		print('historical_reference_jobs: 30/30 (MAE/M1)')
		print('paired_identity_mismatches: 0')
		print('uncovered_validation_voxels: 0')
		print(
			'historical_m1_mae_parity: '
			f"{inspection.historical_m1_mae_parity['status']}"
		)
		print(f"readiness_status: {inspection.readiness['status']}")
		print('execution: dry-run; summary and publish skipped')
		return
	result = summarize_f3_lithology_voxel_label_budget_control(config)
	print(f'current_k6_control.summary_json: {result.summary_json}')
	print(f'current_k6_control.summary_markdown: {result.summary_markdown}')
	print(f'current_k6_control.handoff: {result.handoff_markdown}')
	print(f"current_k6_control.status: {result.readiness['status']}")
	if result.published_files:
		print(f'current_k6_control.published_file_count: {len(result.published_files)}')


if __name__ == '__main__':
	main()
