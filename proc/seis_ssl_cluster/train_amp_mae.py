"""Thin entrypoint for amplitude-only MAE training."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.cli import (
	add_device_argument,
	add_path_argument,
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import (
	load_config,
	resolve_mae_training_config,
)
from seis_ssl_cluster.training.mae import run_mae_pretraining
from seis_ssl_cluster.utils.cli import print_config_summary

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[1]
	/ 'configs'
	/ 'seis_ssl_cluster'
	/ 'train_amp_mae.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for amplitude-only MAE training."""
	parser = build_config_parser(
		'Train an amplitude-only MAE model.',
		default_config=DEFAULT_CONFIG,
	)
	add_device_argument(parser, help_text='Training device override.')
	parser.add_argument(
		'--max-steps',
		type=int,
		help='Stop after N optimizer steps for smoke runs.',
	)
	add_path_argument(
		parser,
		'--output-root',
		help_text=(
		'Override paths.output_root for checkpoints and run snapshots; '
			'must be an explicit absolute output path.'
		),
	)
	add_path_argument(
		parser,
		'--resume',
		help_text='Resume amplitude MAE pretraining from a checkpoint.',
	)
	return parser


def main() -> None:
	"""Run amplitude-only MAE pretraining or print a dry-run summary."""
	parser = build_parser()
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	_apply_cli_overrides(
		raw_config,
		device=args.device,
		max_steps=args.max_steps,
		output_root=args.output_root,
	)
	config = resolve_config_for_cli(
		raw_config,
		resolver=resolve_mae_training_config,
		config_path=config_path,
	)
	if args.resume is not None and not args.resume.is_file():
		raise FileNotFoundError(f'resume checkpoint does not exist: {args.resume}')
	if args.dry_run:
		print_config_summary(config)
		if args.resume is not None:
			print(f'resume: {args.resume}')
		print('execution: dry-run; training skipped')
		return

	checkpoint_path = run_mae_pretraining(config, resume=args.resume)
	print(f'checkpoint: {checkpoint_path}')


def _apply_cli_overrides(
	config: dict[str, object],
	*,
	device: str | None,
	max_steps: int | None,
	output_root: Path | None,
) -> None:
	if device is not None or max_steps is not None:
		train = _section(config, 'train')
	if output_root is not None:
		paths = _section(config, 'paths')
	if device is not None:
		train['device'] = device
	if max_steps is not None:
		train['max_steps'] = max_steps
	if output_root is not None:
		paths['output_root'] = str(output_root)


def _section(config: dict[str, object], key: str) -> dict[str, object]:
	value = config[key]
	if not isinstance(value, dict):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


if __name__ == '__main__':
	main()
