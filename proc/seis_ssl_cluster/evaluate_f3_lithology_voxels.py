"""Evaluate a common V0/V1 F3 voxel prediction artifact."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_evaluation import (
	F3LithologyVoxelEvaluationConfig,
	f3_lithology_voxel_evaluation_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_evaluation import (
	evaluate_f3_lithology_voxels,
	inspect_f3_lithology_voxel_evaluation,
)

if TYPE_CHECKING:
	import argparse

STAGE = 'evaluate_f3_lithology_voxels'


def build_parser() -> argparse.ArgumentParser:
	"""Build the config-driven evaluator parser."""
	return build_config_parser(
		'Evaluate common F3 V0/V1 voxel prediction artifacts.',
		dry_run_help='Validate identities and print the plan without writing outputs.',
	)


def main() -> None:
	"""Validate the run plan and optionally write numeric evaluation artifacts."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_lithology_voxel_evaluation_config_from_mapping,
		config_path=config_path,
	)
	inspection = inspect_f3_lithology_voxel_evaluation(config)
	_print_plan(config, validation_voxel_count=inspection.validation_voxel_count)
	if args.dry_run:
		print('execution: dry-run; evaluation outputs skipped')
		return
	result = evaluate_f3_lithology_voxels(config)
	print(f'validation_voxel_count: {result.validation_voxel_count}')
	print(f'outputs.output_dir: {result.output_dir}')
	print('execution: complete')


def _print_plan(
	config: F3LithologyVoxelEvaluationConfig, *, validation_voxel_count: int
) -> None:
	print(f'stage: {STAGE}')
	print(f'voxel_predictions.input_dir: {config.prediction_input_dir}')
	print(f'voxel_dataset.input_dir: {config.voxel_dataset_input_dir}')
	print(f'labels.source_label_volume: {config.source_label_volume}')
	print(f'evaluation.validation_voxel_count: {validation_voxel_count}')
	print(f'evaluation.monitored_class_ids: {list(config.monitored_class_ids)}')
	print(f'evaluation.boundary_tolerances: {list(config.boundary_tolerances)}')
	print(
		'evaluation.boundary_region_radii: '
		f'{list(config.boundary_region_radii)}'
	)
	print(f'outputs.output_dir: {config.output_dir}')


if __name__ == '__main__':
	main()
