"""Project an F3 token-prediction artifact to nearest-repeat voxels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_projection import (
	F3LithologyVoxelProjectionConfig,
	f3_lithology_voxel_projection_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_projection import (
	F3VoxelProjectionResult,
	project_f3_lithology_tokens_to_voxels,
)

if TYPE_CHECKING:
	import argparse

STAGE = 'project_f3_lithology_tokens_to_voxels'


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for V0 nearest voxel projection."""
	return build_config_parser(
		'Project F3 lithology token predictions to nearest-repeat voxels.',
		dry_run_help='Validate inputs and print a summary without writing outputs.',
	)


def main() -> None:
	"""Resolve the config, summarize it, and optionally write the projection."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=f3_lithology_voxel_projection_config_from_mapping,
		config_path=config_path,
	)
	_print_summary(config)
	if args.dry_run:
		print('execution: dry-run')
		return
	result = project_f3_lithology_tokens_to_voxels(
		config.source.input_dir,
		config.output_dir,
		write_probabilities=config.write_probabilities,
		overwrite=config.overwrite,
	)
	_print_result(result)


def _print_summary(config: F3LithologyVoxelProjectionConfig) -> None:
	print(f'stage: {STAGE}')
	print(f'dataset.name: {config.dataset["name"]}')
	print(f'dataset.version: {config.dataset["version"]}')
	print(f'model.tag: {config.model["tag"]}')
	print(f'token_predictions.predictions: {config.source.predictions}')
	print(f'token_predictions.probabilities: {config.source.probabilities}')
	print(f'token_predictions.valid_tokens: {config.source.valid_tokens}')
	print(f'token_predictions.metadata_json: {config.source.metadata_json}')
	print(f'source.token_grid_shape_xyz: {config.source.token_grid_shape_xyz}')
	print(f'source.patch_size_xyz: {config.source.patch_size_xyz}')
	print(f'source.volume_shape_xyz: {config.source.volume_shape_xyz}')
	print(f'voxel_projection.mode: {config.mode}')
	print(
		'voxel_projection.write_probabilities: '
		f'{config.write_probabilities}'
	)
	_print_output_paths(config)


def _print_output_paths(config: F3LithologyVoxelProjectionConfig) -> None:
	paths = config.output_paths
	print(f'voxel_projection.output_dir: {paths.output_dir}')
	print(f'outputs.predictions: {paths.predictions}')
	print(f'outputs.confidence: {paths.confidence}')
	print(f'outputs.valid_mask: {paths.valid_mask}')
	if config.write_probabilities:
		print(f'outputs.probabilities: {paths.probabilities}')
	print(f'outputs.metadata: {paths.metadata}')


def _print_result(result: F3VoxelProjectionResult) -> None:
	# Kept separate so main remains a thin procedure entrypoint.
	paths = result.paths
	print(f'output.predictions: {paths.predictions}')
	print(f'output.confidence: {paths.confidence}')
	print(f'output.valid_mask: {paths.valid_mask}')
	if result.probabilities_written:
		print(f'output.probabilities: {paths.probabilities}')
	print(f'output.metadata: {paths.metadata}')
	print(f'valid_voxel_count: {result.valid_voxel_count}')
	print(f'invalid_voxel_count: {result.invalid_voxel_count}')
	print('execution: complete')


if __name__ == '__main__':
	main()
