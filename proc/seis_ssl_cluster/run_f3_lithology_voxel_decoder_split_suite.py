"""Run paired M1/M2-A voxel decoders across existing splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.cli import load_config_for_cli, parse_config_path
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_robustness import (
	f3_lithology_voxel_decoder_split_suite_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_robustness import (
	run_f3_lithology_voxel_decoder_split_suite,
	voxel_decoder_split_jobs,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the resumable V1 split-suite parser."""
	parser = argparse.ArgumentParser(description='Run paired F3 V1 decoder split jobs.')
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--only-missing', action='store_true')
	parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
	return parser


def main() -> None:
	"""Resolve and run the decoder matrix."""
	args = build_parser().parse_args()
	raw = load_config_for_cli(parse_config_path(args), loader=load_config)
	config = f3_lithology_voxel_decoder_split_suite_config_from_mapping(raw)
	jobs = voxel_decoder_split_jobs(config)
	if args.dry_run:
		print('stage: run_f3_lithology_voxel_decoder_split_suite')
		print(f'only_missing: {args.only_missing}')
		print(f'job_count: {len(jobs)}')
		for job in jobs:
			print(
				f'- {job.split_id} {job.model_role} {job.model_tag} '
				f'-> {job.output_root}'
			)
		print('execution: dry-run')
		return
	result = run_f3_lithology_voxel_decoder_split_suite(
		config, only_missing=args.only_missing, device=args.device
	)
	print(f'voxel_decoder_split_suite.manifest: {result.manifest_json}')
	print(f'voxel_decoder_split_suite.row_count: {len(result.rows)}')


if __name__ == '__main__':
	main()
