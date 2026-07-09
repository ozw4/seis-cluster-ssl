"""Build paired F3 lithology token datasets for split/index inventories."""

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
	f3_lithology_split_sweep_dataset_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.robustness import (
	F3SplitSweepDatasetConfig,
	build_f3_lithology_split_sweep_datasets,
	split_sweep_dataset_dry_run_summary,
)

if TYPE_CHECKING:
	import argparse

STAGE = 'build_f3_lithology_split_sweep_datasets'


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for split/index paired token datasets."""
	parser = build_config_parser(
		'Build paired F3 lithology token datasets for split/index inventories.',
		config_required=True,
		dry_run_help=(
			'Validate the config and print a run summary without writing outputs.'
		),
	)
	parser.add_argument(
		'--only-missing',
		action='store_true',
		help='Skip complete split/model token datasets and build missing outputs.',
	)
	return parser


def main() -> None:
	"""Build split/index token datasets or print a dry-run summary."""
	parser = build_parser()
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=f3_lithology_split_sweep_dataset_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		_print_dry_run_summary(config, only_missing=args.only_missing)
		print('execution: dry-run; split-sweep token datasets skipped')
		return

	result = build_f3_lithology_split_sweep_datasets(
		config,
		only_missing=args.only_missing,
	)
	print(f'f3_lithology_split_sweep_datasets.manifest: {result.manifest_json}')
	print(
		'f3_lithology_split_sweep_datasets.dataset_count: '
		f'{len(result.dataset_roots)}',
	)


def _print_dry_run_summary(
	config: F3SplitSweepDatasetConfig,
	*,
	only_missing: bool,
) -> None:
	print(f'stage: {STAGE}')
	for key, value in split_sweep_dataset_dry_run_summary(
		config,
		only_missing=only_missing,
	).items():
		print(f'{key}: {value}')


if __name__ == '__main__':
	main()
