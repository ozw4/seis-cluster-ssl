"""Build split-specific F3 voxel supervision from an existing inventory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_robustness import (
	f3_lithology_voxel_split_dataset_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_robustness import (
	build_f3_lithology_voxel_split_datasets,
	voxel_split_dataset_jobs,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the manifest-driven split supervision parser."""
	parser = build_config_parser(
		'Build F3 voxel supervision for an existing split inventory.',
		config_required=True,
		dry_run_help='Validate and print the planned split matrix.',
	)
	parser.add_argument('--only-missing', action='store_true')
	return parser


def main() -> None:
	"""Resolve and run the split supervision suite."""
	args = build_parser().parse_args()
	raw = load_config_for_cli(parse_config_path(args), loader=load_config)
	config = f3_lithology_voxel_split_dataset_config_from_mapping(raw)
	if args.dry_run:
		_print_jobs(voxel_split_dataset_jobs(config), only_missing=args.only_missing)
		return
	result = build_f3_lithology_voxel_split_datasets(
		config, only_missing=args.only_missing
	)
	print(f'voxel_split_datasets.manifest: {result.manifest_json}')
	print(f'voxel_split_datasets.row_count: {len(result.rows)}')


def _print_jobs(jobs: object, *, only_missing: bool) -> None:
	rows = tuple(jobs)  # type: ignore[arg-type]
	print('stage: build_f3_lithology_voxel_split_datasets')
	print(f'only_missing: {only_missing}')
	print(f'job_count: {len(rows)}')
	for job in rows:
		print(f'- {job.split_id} shared -> {job.output_root}')
	print('execution: dry-run')


if __name__ == '__main__':
	main()
