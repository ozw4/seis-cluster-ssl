"""CLI for deterministic shuffled-HMM pseudo-target construction."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	add_overwrite_argument,
	build_config_parser,
	load_config_for_cli,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology.guardrails import (
	f3_shuffled_hmm_target_config_from_mapping,
)
from seis_ssl_cluster.stratigraphy.shuffle_targets import (
	plan_shuffled_hmm_pseudo_targets,
	shuffle_strat_hmm_pseudo_targets,
)

if TYPE_CHECKING:
	import argparse

DEFAULT_CONFIG = (
	Path('experiments')
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '83_strat_hmm_m1_guardrails'
	/ '03_build_shuffled_hmm_pseudo_targets.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the shuffled pseudo-target CLI parser."""
	parser = build_config_parser(
		'Build deterministic shuffled strat-HMM pseudo-targets.',
		default_config=DEFAULT_CONFIG,
		dry_run_help='Validate inputs and print planned outputs without writing.',
	)
	add_overwrite_argument(parser, help_text='Replace existing shuffled outputs.')
	return parser


def main() -> None:
	"""Validate, plan, or build shuffled pseudo-target artifacts."""
	args = build_parser().parse_args()
	raw = load_config_for_cli(args.config, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_shuffled_hmm_target_config_from_mapping,
		config_path=args.config,
	)
	overwrite = config.overwrite or args.overwrite
	planned = plan_shuffled_hmm_pseudo_targets(
		config.source_root,
		config.output_root,
		k=config.k,
		overwrite=overwrite,
	)
	print(f'shuffle.mode: {config.shuffle_scope}')
	print(f'shuffle.seed: {config.seed}')
	for paths in planned:
		print(f'planned_output: {paths.metadata}')
	if args.dry_run:
		print('execution: dry-run; pseudo-target shuffle skipped')
		return
	results = shuffle_strat_hmm_pseudo_targets(
		config.source_root,
		config.output_root,
		k=config.k,
		seed=config.seed,
		mode=config.shuffle_scope,
		overwrite=overwrite,
	)
	for result in results:
		print(f'pseudo_target: {result.paths.metadata}')


if __name__ == '__main__':
	main()
