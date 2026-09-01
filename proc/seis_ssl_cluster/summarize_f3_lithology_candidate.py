"""Summarize one F3 lithology candidate against canonical random metrics."""

from __future__ import annotations

import argparse

from seis_ssl_cluster.cli import (
	add_config_argument,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	f3_lithology_candidate_config_from_mapping,
	load_f3_lithology_candidate_canonical_config,
	summarize_f3_lithology_candidate,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for a completed candidate summary."""
	parser = argparse.ArgumentParser(
		description='Summarize one F3 lithology candidate against random.'
	)
	add_config_argument(
		parser,
		help_text='Path to the minimal candidate YAML.',
	)
	return parser


def main() -> None:
	"""Write comparison.csv, summary.json, and summary.md."""
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
	result = summarize_f3_lithology_candidate(config, canonical_config)
	print(f'candidate_id: {result["candidate_id"]}')
	print(f'complete_jobs: {result["complete_jobs"]}')
	for output in result['outputs']:
		print(f'output: {output}')


if __name__ == '__main__':
	main()
