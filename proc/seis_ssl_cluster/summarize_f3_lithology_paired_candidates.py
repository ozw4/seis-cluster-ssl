"""Audit or summarize an exact F3 lithology paired-candidate comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.cli import load_config_for_cli, resolve_config_for_cli
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology.paired_candidate_results import (
	f3_paired_candidate_summary_config_from_mapping,
	inspect_f3_paired_candidate_results,
	summarize_f3_paired_candidate_results,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments/f3/facies_benchmark_v2'
	/ '123_local_bt_noise_rotation_search_v1/80_hmm_summary'
	/ '01_hmm_k6_25ep_vs_rot90_asym_g060_3ep.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the paired completeness-audit and summary parser."""
	parser = argparse.ArgumentParser(
		description=(
			'Audit or summarize two F3 lithology candidates over the exact '
			'canonical-v3 15-cell matrix.'
		)
	)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	parser.add_argument(
		'--check-only',
		'--dry-run',
		dest='check_only',
		action='store_true',
		help='Audit both sources and all 30 jobs without writing summary files.',
	)
	return parser


def main() -> None:
	"""Run the read-only audit or write the exact three-file summary."""
	args = build_parser().parse_args()
	raw = load_config_for_cli(args.config, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_paired_candidate_summary_config_from_mapping,
		config_path=args.config,
	)
	if args.check_only:
		report = inspect_f3_paired_candidate_results(config)
		print(f'candidate_id: {report["candidate_id"]}')
		print(f'control_id: {report["control_id"]}')
		print(f'complete_candidate_jobs: {report["complete_candidate_jobs"]}')
		print(f'complete_control_jobs: {report["complete_control_jobs"]}')
		print(f'compared_cells: {report["compared_cells"]}')
		print('execution: check-only; summary files skipped')
		return
	for path in summarize_f3_paired_candidate_results(config):
		print(f'output: {path}')


if __name__ == '__main__':
	main()
