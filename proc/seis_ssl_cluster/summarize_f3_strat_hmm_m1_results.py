"""CLI for F3 strat-HMM milestone-1 result consolidation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology.m1_results import (
	F3StratHMMM1ResultsConfig,
	consolidate_f3_strat_hmm_m1_results,
	f3_strat_hmm_m1_results_config_from_mapping,
)

if TYPE_CHECKING:
	import argparse
	from collections.abc import Mapping

STAGE = 'summarize_f3_strat_hmm_m1_results'
DEFAULT_CONFIG = (
	Path('experiments')
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '82_strat_hmm_m1_results'
	/ '01_summarize_m1_results.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for M1 result consolidation."""
	return build_config_parser(
		'Consolidate F3 strat-HMM milestone-1 result artifacts.',
		default_config=DEFAULT_CONFIG,
		dry_run_help='Validate inputs and print planned outputs without writing.',
	)


def main() -> None:
	"""Run M1 result consolidation."""
	parser = build_parser()
	args = parser.parse_args()
	raw_config = load_config_for_cli(args.config, loader=load_config)
	config = _config_from_mapping(raw_config, config_path=args.config)
	if args.dry_run:
		_print_summary(config)
		print('execution: dry-run; M1 result consolidation skipped')
		return
	result = consolidate_f3_strat_hmm_m1_results(config)
	print(f'f3_strat_hmm_m1_results.summary_json: {result.summary_json}')
	print(f'f3_strat_hmm_m1_results.summary_markdown: {result.summary_markdown}')
	for table_path in result.table_paths:
		print(f'f3_strat_hmm_m1_results.table: {table_path}')
	for figure_path in result.figure_paths:
		print(f'f3_strat_hmm_m1_results.figure: {figure_path}')
	print(f'f3_strat_hmm_m1_results.warning_count: {len(result.warnings)}')
	if result.publish_manifest is not None:
		print(
			'published F3 strat-HMM M1 results: '
			f'{result.publish_manifest.output_dir}',
		)
		print(f'wrote publish manifest: {result.publish_manifest.manifest_path}')


def _config_from_mapping(
	raw_config: Mapping[str, object],
	*,
	config_path: Path,
) -> F3StratHMMM1ResultsConfig:
	return resolve_config_for_cli(
		raw_config,
		resolver=f3_strat_hmm_m1_results_config_from_mapping,
		config_path=config_path,
	)


def _print_summary(config: F3StratHMMM1ResultsConfig) -> None:
	print(f'stage: {STAGE}')
	print(f'inputs.baseline_comparison_csv: {config.baseline_comparison_csv}')
	print(f'inputs.label_budget_suite_root: {config.label_budget_suite_root}')
	print(f'inputs.split_index_suite_root: {config.split_index_suite_root}')
	print(f'models.baseline: {config.baseline_model}')
	print(f'models.candidate: {config.candidate_model}')
	print(f'outputs.summary_json: {config.output_dir / "m1_results_summary.json"}')
	print(f'outputs.summary_markdown: {config.output_dir / "m1_results_summary.md"}')
	print(f'outputs.tables_dir: {config.output_dir / "tables"}')
	print(f'outputs.figures_dir: {config.output_dir / "figures"}')
	print(f'publish.enabled: {config.publish.enabled}')
	if config.publish.output_dir is not None:
		print(f'publish.output_dir: {config.publish.output_dir}')
	print(f'publish.include_figures: {config.publish.include_figures}')
	print(f'publish.max_file_size_bytes: {config.publish.max_file_size_bytes}')


if __name__ == '__main__':
	main()
