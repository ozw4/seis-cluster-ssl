"""Build the canonical F3 voxel supervision dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_dataset import (
	F3LithologyVoxelDatasetConfig,
	f3_lithology_voxel_dataset_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_dataset import (
	F3LithologyVoxelDatasetInspection,
	build_f3_lithology_voxel_dataset,
	inspect_f3_lithology_voxel_dataset,
)

if TYPE_CHECKING:
	import argparse

STAGE = 'build_f3_lithology_voxel_dataset'
DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v2'
	/ '10_prepare'
	/ '03_build_voxel_supervision.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Return the config/dry-run parser."""
	return build_config_parser(
		'Build the F3 voxel supervision dataset.',
		default_config=DEFAULT_CONFIG,
		dry_run_help='Validate all sources and geometry without writing outputs.',
	)


def main() -> None:
	"""Run the builder or its fully validating dry-run."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_lithology_voxel_dataset_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		inspection = inspect_f3_lithology_voxel_dataset(config)
		_print_summary(config, inspection)
		print('execution: dry-run')
		return
	result = build_f3_lithology_voxel_dataset(config)
	print(f'voxel_dataset.train_voxel_count: {result.train_voxel_count}')
	print(f'voxel_dataset.validation_voxel_count: {result.validation_voxel_count}')
	print(f'voxel_dataset.split_grid: {result.split_grid}')
	print(f'voxel_dataset.metadata_json: {result.metadata_json}')
	print(f'voxel_dataset.class_counts_csv: {result.class_counts_csv}')
	print(f'voxel_dataset.split_manifest_json: {result.split_manifest_json}')
	print(f'voxel_dataset.summary_markdown: {result.summary_markdown}')


def _print_summary(
	config: F3LithologyVoxelDatasetConfig,
	inspection: F3LithologyVoxelDatasetInspection,
) -> None:
	print(f'stage: {STAGE}')
	print(f'dataset: {dict(config.dataset)}')
	print(f'labels.source_label_volume: {config.source_label_volume}')
	print(f'labels.png_label_inventory: {config.png_label_inventory}')
	print(f'labels.class_info: {config.class_info}')
	print(f'reference_embedding.valid_tokens: {config.reference_valid_tokens}')
	print(f'reference_embedding.patch_size: {inspection.patch_size_xyz}')
	print(f'reference_embedding.token_grid_shape: {inspection.token_grid_shape_xyz}')
	print(f'reference_embedding.volume_shape_xyz: {inspection.volume_shape_xyz}')
	print(
		'voxel_dataset.train_voxel_count: '
		f'{inspection.split.summary.final_train_voxels}'
	)
	print(
		'voxel_dataset.validation_voxel_count: '
		f'{inspection.split.summary.final_validation_voxels}'
	)
	print(f'voxel_dataset.output_dir: {config.output_dir}')
	print(
		'voxel_dataset.split_policy: PNG inventory; validation precedence; '
		'no random split'
	)


if __name__ == '__main__':
	main()
