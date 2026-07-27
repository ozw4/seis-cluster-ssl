"""Build the M4 selected-multi-head six-split low-label datasets."""
# ruff: noqa: D103, E501

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_split import (
	f3_lithology_voxel_label_budget_split_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_split import (
	build_f3_lithology_voxel_label_budget_split_datasets,
	inspect_f3_lithology_voxel_label_budget_split_datasets,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the strict config, dry-run, and only-missing parser."""
	parser = build_config_parser(
		'Build M4 selected multi-head six-split low-label datasets.',
		config_required=True,
		dry_run_help='Validate the 12 dataset matrix without writing.',
	)
	parser.add_argument('--only-missing', action='store_true')
	return parser


def main() -> None:
	args = build_parser().parse_args()
	path = parse_config_path(args)
	config = resolve_config_for_cli(load_config_for_cli(path, loader=load_config), resolver=f3_lithology_voxel_label_budget_split_config_from_mapping, config_path=path)
	if args.dry_run:
		inspection = inspect_f3_lithology_voxel_label_budget_split_datasets(config)
		print(f'condition_count: {len(inspection.conditions)}')
		for item in inspection.conditions:
			print(f'{item.split_id}/{item.budget_id}: {item.row["train_voxel_count"]} train voxels')
		return
	manifest, rows = build_f3_lithology_voxel_label_budget_split_datasets(config, only_missing=args.only_missing)
	print(f'low_label_split_dataset_manifest: {manifest}')
	print(f'condition_count: {len(rows)}')


if __name__ == '__main__':
	main()
