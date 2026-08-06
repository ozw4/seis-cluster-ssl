"""CLI for the F3 original-split low-label voxel summary."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_results import (
	F3VoxelLabelBudgetResultsConfig,
	f3_lithology_voxel_label_budget_results_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import (
	inspect_f3_lithology_voxel_label_budget_results,
	summarize_f3_lithology_voxel_label_budget_results,
)

if TYPE_CHECKING:
	import argparse
	from collections.abc import Mapping

STAGE = 'summarize_f3_lithology_voxel_label_budget'
DEFAULT_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/91_f3_voxel_label_budget_v1/'
	'03_summarize_voxel_label_budget.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the summary CLI parser."""
	return build_config_parser(
		'Aggregate the complete 45-job F3 low-label voxel benchmark.',
		default_config=DEFAULT_CONFIG,
		dry_run_help=(
			'Validate all manifests, pair identities, and metrics without writing.'
		),
	)


def _config_from_mapping(
	raw_config: Mapping[str, object], *, config_path: Path
) -> F3VoxelLabelBudgetResultsConfig:
	return resolve_config_for_cli(
		raw_config,
		resolver=f3_lithology_voxel_label_budget_results_config_from_mapping,
		config_path=config_path,
	)


def _print_inspection(
	config: F3VoxelLabelBudgetResultsConfig,
	*,
	job_count: int,
	paired_count: int,
	decisions: Mapping[str, object],
) -> None:
	print(f'stage: {STAGE}')
	print(f'suite.root: {config.suite_root}')
	print(f'suite.dataset_manifest: {config.dataset_manifest}')
	print(f'suite.run_manifest: {config.run_manifest}')
	print(f'complete jobs: {job_count}/45')
	print(f'paired conditions: {paired_count}/15')
	print('paired identity mismatches: 0')
	print(f'outputs.reports_dir: {config.reports_dir}')
	print(f'publish.enabled: {config.publish.enabled}')
	print(f'scientific_decisions: {decisions}')


def main() -> None:
	"""Run or validate the low-label voxel result summary."""
	parser = build_parser()
	args = parser.parse_args()
	raw = load_config_for_cli(args.config, loader=load_config)
	config = _config_from_mapping(raw, config_path=args.config)
	if args.dry_run:
		inspection = inspect_f3_lithology_voxel_label_budget_results(config)
		_print_inspection(
			config,
			job_count=len(inspection.job_metrics),
			paired_count=len(inspection.paired_metrics),
			decisions=inspection.decisions,
		)
		print('execution: dry-run; summary and publish skipped')
		return
	result = summarize_f3_lithology_voxel_label_budget_results(config)
	print(f'voxel_label_budget.summary_json: {result.summary_json}')
	print(f'voxel_label_budget.summary_markdown: {result.summary_markdown}')
	print(f'voxel_label_budget.scientific_decisions: {result.decisions}')
	if result.published_files:
		print(f'voxel_label_budget.published_file_count: {len(result.published_files)}')


if __name__ == '__main__':
	main()
