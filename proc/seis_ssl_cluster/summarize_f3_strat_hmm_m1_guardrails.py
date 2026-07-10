"""CLI for the F3 strat-HMM milestone-1 guardrail summary."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import build_config_parser, load_config_for_cli
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology.guardrails import (
	f3_guardrail_summary_config_from_mapping,
	summarize_f3_strat_hmm_m1_guardrails,
)

if TYPE_CHECKING:
	import argparse

DEFAULT_CONFIG = (
	Path('experiments')
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '83_strat_hmm_m1_guardrails'
	/ '08_summarize_guardrails.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the guardrail-summary CLI parser."""
	return build_config_parser(
		'Consolidate F3 strat-HMM milestone-1 guardrail results.',
		default_config=DEFAULT_CONFIG,
		dry_run_help='Validate the contract and print planned outputs.',
	)


def main() -> None:
	"""Run guardrail result consolidation."""
	args = build_parser().parse_args()
	raw = load_config_for_cli(args.config, loader=load_config)
	config = f3_guardrail_summary_config_from_mapping(raw)
	print(f'suite.name: {config.suite_name}')
	print(f'suite.strict: {config.strict}')
	print(f'outputs.summary_json: {config.output_dir / "guardrail_summary.json"}')
	print(f'outputs.summary_markdown: {config.output_dir / "guardrail_summary.md"}')
	if args.dry_run:
		print('execution: dry-run; guardrail summary skipped')
		return
	result = summarize_f3_strat_hmm_m1_guardrails(config)
	print(f'guardrails.summary_json: {result.summary_json}')
	print(f'guardrails.summary_markdown: {result.summary_markdown}')
	print(f'guardrails.pending_roles: {list(result.pending_roles)}')


if __name__ == '__main__':
	main()
