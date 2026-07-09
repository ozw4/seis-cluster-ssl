"""Generate alternative F3 lithology split/index inventories."""

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
	f3_lithology_split_inventory_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.robustness import (
	F3SplitInventoryConfig,
	build_f3_lithology_split_inventories,
	split_inventory_dry_run_summary,
)

if TYPE_CHECKING:
	import argparse

STAGE = 'generate_f3_lithology_split_inventories'


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for F3 split inventory generation."""
	return build_config_parser(
		'Generate alternative F3 lithology split/index inventories.',
		config_required=True,
		dry_run_help=(
			'Validate the config and print a run summary without writing outputs.'
		),
	)


def main() -> None:
	"""Generate split inventories or print a dry-run summary."""
	parser = build_parser()
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=f3_lithology_split_inventory_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		_print_dry_run_summary(config)
		print('execution: dry-run; split inventories skipped')
		return

	result = build_f3_lithology_split_inventories(config)
	print(f'f3_lithology_split_inventories.manifest: {result.manifest_json}')
	print(f'f3_lithology_split_inventories.split_count: {len(result.inventory_paths)}')


def _print_dry_run_summary(config: F3SplitInventoryConfig) -> None:
	print(f'stage: {STAGE}')
	for key, value in split_inventory_dry_run_summary(config).items():
		print(f'{key}: {value}')


if __name__ == '__main__':
	main()
