"""Build paired F3 lithology label-budget token datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_robustness import (
	f3_lithology_label_budget_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.robustness import (
	F3LabelBudgetConfig,
	build_f3_lithology_label_budget_datasets,
	label_budget_dry_run_summary,
)

if TYPE_CHECKING:
	import argparse

STAGE = 'build_f3_lithology_label_budget_datasets'


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for paired label-budget token datasets."""
	return build_config_parser(
		'Build paired F3 lithology label-budget token datasets.',
		config_required=True,
		dry_run_help=(
			'Validate the config and print a run summary without writing outputs.'
		),
	)


def main() -> None:
	"""Build paired label-budget datasets or print a dry-run summary."""
	parser = build_parser()
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=f3_lithology_label_budget_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		_print_dry_run_summary(config)
		print('execution: dry-run; label-budget datasets skipped')
		return

	result = build_f3_lithology_label_budget_datasets(config)
	print(f'f3_lithology_label_budget.suite_manifest: {result.suite_manifest_json}')
	print(f'f3_lithology_label_budget.dataset_count: {len(result.dataset_roots)}')


def _print_dry_run_summary(config: F3LabelBudgetConfig) -> None:
	print(f'stage: {STAGE}')
	for key, value in label_budget_dry_run_summary(config).items():
		print(f'{key}: {value}')


if __name__ == '__main__':
	main()
