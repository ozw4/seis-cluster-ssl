"""Thin entrypoint for amplitude-only MAE training."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / 'src'
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from seis_ssl_cluster.config import (  # noqa: E402
	load_config,
	resolve_mae_training_config,
)
from seis_ssl_cluster.training.mae import run_mae_pretraining  # noqa: E402
from seis_ssl_cluster.utils.cli import print_config_summary  # noqa: E402

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[1]
	/ 'configs'
	/ 'seis_ssl_cluster'
	/ 'train_amp_mae.yaml'
)


def main() -> None:
	"""Run amplitude-only MAE pretraining or print a dry-run summary."""
	parser = ArgumentParser(description='Train an amplitude-only MAE model.')
	parser.add_argument(
		'--config',
		type=Path,
		default=DEFAULT_CONFIG,
		help='Path to a YAML configuration file.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate the config and print a run summary without executing.',
	)
	parser.add_argument(
		'--device',
		choices=('auto', 'cpu', 'cuda'),
		help='Training device override.',
	)
	parser.add_argument(
		'--max-steps',
		type=int,
		help='Stop after N optimizer steps for smoke runs.',
	)
	parser.add_argument(
		'--output-root',
		type=Path,
		help='Override paths.output_root for checkpoints and run snapshots.',
	)
	parser.add_argument(
		'--resume',
		type=Path,
		help='Resume amplitude MAE pretraining from a checkpoint.',
	)
	args = parser.parse_args()

	raw_config = load_config(args.config)
	_apply_cli_overrides(
		raw_config,
		device=args.device,
		max_steps=args.max_steps,
		output_root=args.output_root,
	)
	config = resolve_mae_training_config(raw_config)
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
