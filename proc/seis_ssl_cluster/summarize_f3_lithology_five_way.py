"""Summarize the 75-job F3 lithology five-way benchmark matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_five_way import (
	f3_lithology_five_way_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.five_way_results import (
	inspect_f3_lithology_five_way_results,
	summarize_f3_lithology_five_way,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for the five-way summary."""
	return build_config_parser(
		'Summarize the 75 F3 lithology five-way decoder jobs.',
		config_help='Path to the canonical five-way comparison YAML.',
		dry_run_help=(
			'Audit all 75 evaluations read-only and report completeness '
			'without writing summary files.'
		),
	)


def main() -> None:
	"""Write the five-way summary or print the dry-run completeness audit."""
	parser = build_parser()
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_lithology_five_way_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		report = inspect_f3_lithology_five_way_results(config)
		print(f'complete_jobs: {report["complete_jobs"]}')
		print(f'model_order: {", ".join(config.model_ids)}')
		print('execution: dry-run; summary files skipped')
		return

	result = summarize_f3_lithology_five_way(config)
	print(f'complete_jobs: {result["complete_jobs"]}')
	for output in result['outputs']:
		print(f'output: {output}')


if __name__ == '__main__':
	main()
