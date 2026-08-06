"""CLI for F3 strat-HMM M2-A versus M1 result consolidation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology.m2_results import (
	F3StratHMMM2ResultsConfig,
	consolidate_f3_strat_hmm_m2_results,
	f3_strat_hmm_m2_results_config_from_mapping,
)

if TYPE_CHECKING:
	import argparse
	from collections.abc import Mapping

STAGE = 'summarize_f3_strat_hmm_m2_results'
DEFAULT_CONFIG = (
	Path('experiments/f3/facies_benchmark_v1/86_strat_hmm_m2a_results')
	/ '01_summarize_m2a_results.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the M2-A consolidation CLI parser."""
	return build_config_parser(
		'Consolidate F3 strat-HMM M2-A versus M1 result artifacts.',
		default_config=DEFAULT_CONFIG,
		dry_run_help='Validate inputs and print planned outputs without writing.',
	)


def _config_from_mapping(
	raw_config: Mapping[str, object], *, config_path: Path
) -> F3StratHMMM2ResultsConfig:
	return resolve_config_for_cli(
		raw_config,
		resolver=f3_strat_hmm_m2_results_config_from_mapping,
		config_path=config_path,
	)


def _print_summary(config: F3StratHMMM2ResultsConfig) -> None:
	print(f'stage: {STAGE}')
	for key in (
		'baseline_comparison_csv',
		'm1_metrics_json',
		'm2a_metrics_json',
		'label_budget_suite_root',
		'split_index_suite_root',
		'class_info_json',
	):
		print(f'inputs.{key}: {getattr(config, key)}')
	print(f'inputs.monitored_class_ids: {list(config.monitored_class_ids)}')
	print(f'models.baseline: {config.baseline_model}')
	print(f'models.candidate: {config.candidate_model}')
	print(f'outputs.summary_json: {config.output_dir / "m2a_results_summary.json"}')
	print(f'outputs.summary_markdown: {config.output_dir / "m2a_results_summary.md"}')
	print(f'publish.enabled: {config.publish.enabled}')


def main() -> None:
	"""Run M2-A result consolidation."""
	parser = build_parser()
	args = parser.parse_args()
	raw = load_config_for_cli(args.config, loader=load_config)
	config = _config_from_mapping(raw, config_path=args.config)
	if args.dry_run:
		_print_summary(config)
		print('execution: dry-run; M2-A result consolidation skipped')
		return
	result = consolidate_f3_strat_hmm_m2_results(config)
	print(f'f3_strat_hmm_m2_results.summary_json: {result.summary_json}')
	print(f'f3_strat_hmm_m2_results.summary_markdown: {result.summary_markdown}')
	print(f'f3_strat_hmm_m2_results.decision: {result.decision}')
	if result.published_files:
		print(
			'published F3 strat-HMM M2-A results: '
			f'{result.published_files[0].parent}'
		)


if __name__ == '__main__':
	main()
