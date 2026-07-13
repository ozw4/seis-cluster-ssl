"""Run paired M1/M2-A V0 voxel projection jobs across existing splits."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_robustness import (
	f3_lithology_voxel_v0_split_suite_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_robustness import (
	run_f3_lithology_voxel_v0_split_suite,
	voxel_v0_split_jobs,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the generic V0 split-suite parser."""
	parser = build_config_parser(
		'Run paired F3 V0 token projections across existing splits.',
		config_required=True,
		dry_run_help='Validate identities and print the job matrix.',
	)
	parser.add_argument('--only-missing', action='store_true')
	return parser


def main() -> None:
	"""Resolve and run the V0 matrix."""
	args = build_parser().parse_args()
	raw = load_config_for_cli(parse_config_path(args), loader=load_config)
	config = f3_lithology_voxel_v0_split_suite_config_from_mapping(raw)
	jobs = voxel_v0_split_jobs(config)
	if args.dry_run:
		_print_jobs(jobs, only_missing=args.only_missing)
		return
	result = run_f3_lithology_voxel_v0_split_suite(
		config, only_missing=args.only_missing
	)
	print(f'voxel_v0_split_suite.manifest: {result.manifest_json}')
	print(f'voxel_v0_split_suite.row_count: {len(result.rows)}')


def _print_jobs(jobs: object, *, only_missing: bool) -> None:
	rows = tuple(jobs)  # type: ignore[arg-type]
	print('stage: run_f3_lithology_voxel_v0_split_suite')
	print(f'only_missing: {only_missing}')
	print(f'job_count: {len(rows)}')
	for job in rows:
		print(f'- {job.split_id} {job.model_role} {job.model_tag} -> {job.output_root}')
	print('execution: dry-run')


if __name__ == '__main__':
	main()
