"""Build common F3 section-layout voxel supervision datasets."""

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
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout_dataset import (
	F3SectionLayoutDatasetConfig,
	f3_lithology_voxel_section_layout_dataset_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout import (
	build_f3_lithology_voxel_section_layout_datasets,
	inspect_f3_lithology_voxel_section_layout_datasets,
)

if TYPE_CHECKING:
	import argparse

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '109_f3_voxel_section_layout_v1'
	/ '03_build_section_layout_datasets.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the strict dry-run and recovery parser."""
	parser = build_config_parser(
		'Build model-independent F3 section-layout voxel supervision datasets.',
		default_config=DEFAULT_CONFIG,
		dry_run_help='Validate sources and print the exact 15-condition plan.',
	)
	parser.add_argument(
		'--only-missing',
		action='store_true',
		help='Reuse only exact complete conditions; reject stale or partial output.',
	)
	parser.add_argument(
		'--quarantine-invalid',
		action='store_true',
		help=(
			'With --only-missing, move invalid conditions to timestamped siblings '
			'before rebuilding.'
		),
	)
	return parser


def main() -> None:
	"""Resolve config, inspect for dry-run, or build the common datasets."""
	args = build_parser().parse_args()
	if args.quarantine_invalid and not args.only_missing:
		build_parser().error('--quarantine-invalid requires --only-missing')
	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_lithology_voxel_section_layout_dataset_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		inspection = inspect_f3_lithology_voxel_section_layout_datasets(config)
		_print_plan(
			config,
			inspection.conditions,
			only_missing=bool(args.only_missing),
			quarantine_invalid=bool(args.quarantine_invalid),
		)
		return
	result = build_f3_lithology_voxel_section_layout_datasets(
		config,
		only_missing=bool(args.only_missing),
		quarantine_invalid=bool(args.quarantine_invalid),
	)
	actions: dict[str, int] = {}
	for row in result.rows:
		action = str(row['action'])
		actions[action] = actions.get(action, 0) + 1
	print(f'section_layout_datasets.manifest: {result.manifest_json}')
	print(f'section_layout_datasets.condition_count: {len(result.rows)}')
	print(f'section_layout_datasets.actions: {actions}')
	print(f'section_layout_datasets.quarantine_count: {len(result.quarantines)}')


def _print_plan(
	config: F3SectionLayoutDatasetConfig,
	conditions: object,
	*,
	only_missing: bool,
	quarantine_invalid: bool,
) -> None:
	rows = tuple(conditions)  # type: ignore[arg-type]
	print('stage: build_f3_lithology_voxel_section_layout_datasets')
	print(f'output_root: {config.output_root}')
	print(f'condition_count: {len(rows)}')
	print(f'only_missing: {only_missing}')
	print(f'quarantine_invalid: {quarantine_invalid}')
	for condition in rows:
		print(
			f'- {condition.layout_id}/{condition.data_size} '
			f'train_voxels={condition.actual_train_voxel_count} -> '
			f'{condition.output_dir}'
		)
	print('execution: dry-run; no writes or quarantines')


if __name__ == '__main__':
	main()
