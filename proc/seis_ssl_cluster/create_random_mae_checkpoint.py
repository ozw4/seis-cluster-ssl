"""Create a random-initialized MAE checkpoint from a reference architecture."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.training.random_checkpoint import (
	create_random_mae_checkpoint_from_config,
	random_mae_checkpoint_config_from_mapping,
)


def main() -> None:
	"""Create a random MAE checkpoint or print a dry-run summary."""
	parser = ArgumentParser(
		description='Create a random-initialized MAE checkpoint baseline.',
	)
	parser.add_argument(
		'--config',
		type=Path,
		required=True,
		help='Path to a random checkpoint YAML configuration file.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate the config and print a run summary without writing.',
	)
	args = parser.parse_args()

	config = load_config(args.config)
	settings = random_mae_checkpoint_config_from_mapping(config)
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
