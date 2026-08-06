"""Run context-halo chunked inference with a trained F3 voxel decoder."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.cli import (
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_inference import (
	F3LithologyVoxelInferenceConfig,
	f3_lithology_voxel_inference_config_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_decoder_inference import (
	VoxelDecoderInferenceResult,
	inspect_f3_lithology_voxel_inference,
	predict_f3_lithology_voxels,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	CONFIDENCE_NAME,
	METADATA_NAME,
	PREDICTIONS_NAME,
	PROBABILITIES_NAME,
	VALID_MASK_NAME,
)

STAGE = 'predict_f3_lithology_voxels'


def build_parser() -> argparse.ArgumentParser:
	"""Build the voxel-decoder inference CLI parser."""
	parser = argparse.ArgumentParser(
		description='Predict F3 lithology voxels in context-halo chunks.'
	)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument(
		'--dry-run', action='store_true', help='Validate and print the run plan only.'
	)
	parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
	probabilities = parser.add_mutually_exclusive_group()
	probabilities.add_argument(
		'--write-probabilities',
		dest='write_probabilities',
		action='store_true',
		help='Write the optional float16 class-probability volume.',
	)
	probabilities.add_argument(
		'--no-write-probabilities',
		dest='write_probabilities',
		action='store_false',
		help='Do not write the optional class-probability volume.',
	)
	parser.set_defaults(write_probabilities=None)
	parser.add_argument(
		'--overwrite',
		action='store_true',
		help='Atomically replace an existing output artifact.',
	)
	return parser


def main() -> None:
	"""Resolve the job, print its plan, and optionally run inference."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=f3_lithology_voxel_inference_config_from_mapping,
		config_path=config_path,
	)
	_print_plan(config, write_probabilities=args.write_probabilities)
	if args.dry_run:
		print('execution: dry-run; voxel decoder inference skipped')
		return
	result = predict_f3_lithology_voxels(
		config,
		device=args.device,
		write_probabilities=args.write_probabilities,
		overwrite=True if args.overwrite else None,
	)
	_print_result(result)


def _print_plan(
	config: F3LithologyVoxelInferenceConfig,
	*,
	write_probabilities: bool | None,
) -> None:
	plan = inspect_f3_lithology_voxel_inference(config)
	probabilities = (
		config.write_probabilities
		if write_probabilities is None
		else write_probabilities
	)
	print(f'stage: {STAGE}')
	print(f'model.tag: {config.model["tag"]}')
	print(f'decoder.checkpoint: {plan.checkpoint}')
	print(f'decoder.checkpoint_sha256: {file_sha256(plan.checkpoint)}')
	print(f'decoder.spec: {plan.decoder_spec["spec"]}')
	print(f'decoder.upsample_mode: {plan.decoder_spec["upsample_mode"]}')
	print(f'decoder.normalization: {plan.decoder_spec["normalization"]}')
	print(f'embeddings.array: {plan.embeddings}')
	print(f'embeddings.valid_tokens: {plan.valid_tokens}')
	print(f'embeddings.metadata: {plan.embedding_metadata}')
	print(f'geometry.patch_size_xyz: {list(plan.patch_size_xyz)}')
	print(f'geometry.token_grid_shape_xyz: {list(plan.token_grid_shape_xyz)}')
	print(f'geometry.volume_shape_xyz: {list(plan.volume_shape_xyz)}')
	print(f'tiles.core_size_tokens: {list(config.tiles.core_size_tokens)}')
	print(
		'tiles.context_halo_tokens: '
		f'{list(config.tiles.context_halo_tokens)}'
	)
	print(f'inference.write_probabilities: {probabilities}')
	print(f'outputs.output_dir: {config.output_dir}')
	print(f'outputs.predictions: {config.output_dir / PREDICTIONS_NAME}')
	print(f'outputs.confidence: {config.output_dir / CONFIDENCE_NAME}')
	print(f'outputs.valid_mask: {config.output_dir / VALID_MASK_NAME}')
	if probabilities:
		print(f'outputs.probabilities: {config.output_dir / PROBABILITIES_NAME}')
	print(f'outputs.metadata: {config.output_dir / METADATA_NAME}')


def _print_result(result: VoxelDecoderInferenceResult) -> None:
	print(f'output.predictions: {result.output_dir / PREDICTIONS_NAME}')
	print(f'output.confidence: {result.output_dir / CONFIDENCE_NAME}')
	print(f'output.valid_mask: {result.output_dir / VALID_MASK_NAME}')
	if result.probabilities_written:
		print(f'output.probabilities: {result.output_dir / PROBABILITIES_NAME}')
	print(f'output.metadata: {result.output_dir / METADATA_NAME}')
	print(f'core_tile_count: {result.tile_count}')
	print(f'valid_voxel_count: {result.valid_voxel_count}')
	print(f'invalid_voxel_count: {result.invalid_voxel_count}')
	print('execution: complete')


if __name__ == '__main__':
	main()
