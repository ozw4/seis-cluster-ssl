"""Build encoder-independent F3 low-label voxel supervision datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget import (
	F3VoxelLabelBudgetDatasetConfig,
	f3_lithology_voxel_label_budget_dataset_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget import (
	VoxelLabelBudgetDatasetInspection,
	build_f3_lithology_voxel_label_budget_datasets,
	inspect_f3_lithology_voxel_label_budget_datasets,
)

if TYPE_CHECKING:
	import argparse

STAGE = 'build_f3_lithology_voxel_label_budget_datasets'


def build_parser() -> argparse.ArgumentParser:
	"""Build the strict config, dry-run, and only-missing parser."""
	parser = build_config_parser(
		'Build F3 original-split low-label voxel supervision datasets.',
		config_required=True,
		dry_run_help=(
			'Validate both token suites and all derived grids without writing.'
		),
	)
	parser.add_argument(
		'--only-missing',
		action='store_true',
		help=(
			'Reuse fully validated conditions and quarantine invalid or partial '
			'conditions before rebuilding them.'
		),
	)
	return parser


def main() -> None:
	"""Resolve the config and inspect or build all requested conditions."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_lithology_voxel_label_budget_dataset_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		inspection = inspect_f3_lithology_voxel_label_budget_datasets(config)
		_print_dry_run(config, inspection, only_missing=bool(args.only_missing))
		return
	result = build_f3_lithology_voxel_label_budget_datasets(
		config,
		only_missing=bool(args.only_missing),
	)
	actions: dict[str, int] = {}
	for row in result.rows:
		action = str(row['action'])
		actions[action] = actions.get(action, 0) + 1
	print(f'voxel_label_budget.manifest: {result.manifest_json}')
	print(f'voxel_label_budget.condition_count: {len(result.rows)}')
	print(f'voxel_label_budget.actions: {actions}')
	print(f'voxel_label_budget.quarantine_count: {len(result.quarantines)}')
	for path in result.quarantines:
		print(f'voxel_label_budget.quarantine: {path}')


def _print_dry_run(
	config: F3VoxelLabelBudgetDatasetConfig,
	inspection: VoxelLabelBudgetDatasetInspection,
	*,
	only_missing: bool,
) -> None:
	print(f'stage: {STAGE}')
	print(f'suite.name: {config.suite_name}')
	print(f'suite.output_root: {config.output_root}')
	print(f'budgets: {list(config.budgets)}')
	print(f'subsample_seeds: {list(config.subsample_seeds)}')
	print(f'models: {dict(config.models)}')
	print(f'condition_count: {len(inspection.conditions)}')
	print(
		'common_validation_voxel_count: '
		f'{inspection.conditions[0].validation_voxel_count}'
	)
	print(f'only_missing: {only_missing}')
	for condition in inspection.conditions:
		print(
			f'- {condition.budget_id} seed={condition.subsample_seed} '
			f'train_voxels={condition.train_voxel_count} -> '
			f'{condition.output_dir}'
		)
	print('execution: dry-run; no artifacts written')


if __name__ == '__main__':
	main()
