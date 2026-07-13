"""Train the V1 F3 lithology decoder from precomputed embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.cli import (
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	F3LithologyVoxelDecoderConfig,
	f3_lithology_voxel_decoder_config_from_mapping,
)
from seis_ssl_cluster.training.voxel_decoder.runner import (
	inspect_f3_lithology_voxel_decoder,
	run_f3_lithology_voxel_decoder,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the voxel-decoder training CLI parser."""
	parser = argparse.ArgumentParser(
		description='Train the V1 F3 voxel decoder from frozen embeddings.'
	)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument(
		'--dry-run', action='store_true', help='Print geometry and output plan only.'
	)
	parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
	parser.add_argument('--max-steps', type=int)
	parser.add_argument('--resume', type=Path)
	return parser


def main() -> None:
	"""Resolve the config and dispatch dry-run or training."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=f3_lithology_voxel_decoder_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		_print_dry_run(config)
		return
	result = run_f3_lithology_voxel_decoder(
		config,
		device=args.device,
		max_steps=args.max_steps,
		resume=args.resume,
	)
	print(f'voxel_decoder.latest_checkpoint: {result.latest_checkpoint}')
	print(f'voxel_decoder.best_checkpoint: {result.best_checkpoint}')
	print(f'voxel_decoder.history_csv: {result.history_csv}')
	print(f'voxel_decoder.global_step: {result.global_step}')
	print(f'voxel_decoder.completed: {result.completed}')


def _print_dry_run(config: F3LithologyVoxelDecoderConfig) -> None:
	plan = inspect_f3_lithology_voxel_decoder(config)
	print('stage: train_f3_lithology_voxel_decoder')
	print(f'model.tag: {config.model["tag"]}')
	print(f'model.freeze_encoder: {config.model["freeze_encoder"]}')
	print(f'embeddings.array: {plan.embeddings}')
	print(f'embeddings.valid_tokens: {plan.valid_tokens}')
	print(f'embeddings.metadata: {plan.embedding_metadata}')
	print(f'geometry.patch_size_xyz: {list(plan.patch_size_xyz)}')
	print(f'geometry.token_grid_shape_xyz: {list(plan.token_grid_shape_xyz)}')
	print(f'geometry.volume_shape_xyz: {list(plan.volume_shape_xyz)}')
	print(f'voxel_dataset.metadata: {plan.voxel_metadata}')
	print(f'voxel_dataset.split_grid: {plan.split_grid}')
	print(f'decoder.spec: {config.decoder.spec}')
	print(f'decoder.class_count: {config.decoder.class_count}')
	print(f'outputs.output_dir: {config.output_dir}')
	for name in (
		'latest.pt',
		'best.pt',
		'resolved_config.json',
		'run_metadata.json',
		'history.csv',
		'train_tile_manifest.json',
		'validation_tile_manifest.json',
	):
		print(f'outputs.planned: {config.output_dir / name}')
	print('execution: dry-run; array loading and voxel decoder training skipped')


if __name__ == '__main__':
	main()
