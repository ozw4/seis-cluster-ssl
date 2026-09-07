"""Summarize one F3 lithology candidate against all five canonical models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	f3_lithology_candidate_config_from_mapping,
	inspect_f3_lithology_candidate_five_way,
	load_f3_lithology_candidate_canonical_config,
	summarize_f3_lithology_candidate_five_way,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for the candidate-versus-five-way summary."""
	return build_config_parser(
		'Summarize one F3 lithology candidate against the 75 five-way jobs.',
		config_help=(
			'Path to the minimal candidate YAML; outputs.five_way_summary_root '
			'names the summary directory.'
		),
		dry_run_help=(
			'Audit the 90 candidate and five-way evaluations read-only and '
			'report completeness without writing summary files.'
		),
	)


def main() -> None:
	"""Write the five-file summary or print the dry-run completeness audit."""
	parser = build_parser()
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_lithology_candidate_config_from_mapping,
		config_path=config_path,
	)
	canonical_config = load_f3_lithology_candidate_canonical_config(config)
	if args.dry_run:
		report = inspect_f3_lithology_candidate_five_way(config, canonical_config)
		print(f'candidate_id: {report["candidate_id"]}')
		print(f'complete_jobs: {report["complete_jobs"]}')
		print(f'models: {", ".join(report["models"])}')
		print(f'comparisons: {", ".join(report["comparisons"])}')
		print('execution: dry-run; summary files skipped')
		return

	result = summarize_f3_lithology_candidate_five_way(config, canonical_config)
	print(f'candidate_id: {result["candidate_id"]}')
	print(f'complete_jobs: {result["complete_jobs"]}')
	for output in result['outputs']:
		print(f'output: {output}')


if __name__ == '__main__':
	main()
