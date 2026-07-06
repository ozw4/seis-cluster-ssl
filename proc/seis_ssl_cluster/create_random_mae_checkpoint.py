"""Create a random-initialized MAE checkpoint from a reference architecture."""

from __future__ import annotations

import argparse

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.training.random_checkpoint import (
	create_random_mae_checkpoint_from_config,
	random_mae_checkpoint_config_from_mapping,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for random MAE checkpoint creation."""
	return build_config_parser(
		'Create a random-initialized MAE checkpoint baseline.',
		config_help='Path to a random checkpoint YAML configuration file.',
		dry_run_help='Validate the config and print a run summary without writing.',
	)


def main() -> None:
	"""Create a random MAE checkpoint or print a dry-run summary."""
	parser = build_parser()
	args = parser.parse_args()

	config_path = parse_config_path(args)
	config = load_config_for_cli(config_path, loader=load_config)
	settings = resolve_config_for_cli(
		config,
		resolver=random_mae_checkpoint_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		print(f'reference.checkpoint: {settings.reference_checkpoint}')
		print(f'reference.model_tag: {settings.reference_model_tag}')
		print(f'random_checkpoint.seed: {settings.seed}')
		print(f'random_checkpoint.output_checkpoint: {settings.output_checkpoint}')
		print('execution: dry-run; checkpoint creation skipped')
		return

	checkpoint_path = create_random_mae_checkpoint_from_config(config)
	print(f'checkpoint: {checkpoint_path}')


if __name__ == '__main__':
	main()
