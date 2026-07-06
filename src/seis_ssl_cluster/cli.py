"""Generic helpers for procedure-script command line interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
	import argparse
	from collections.abc import Callable, Mapping

T = TypeVar('T')


def add_config_argument(
	parser: argparse.ArgumentParser,
	*,
	default: str | Path | None = None,
	required: bool | None = None,
) -> None:
	"""Add the standard YAML config path argument."""
	config_required = default is None if required is None else required
	parser.add_argument(
		'--config',
		type=Path,
		default=Path(default) if default is not None else None,
		required=config_required,
		help='Path to a YAML configuration file.',
	)


def parse_config_path(args: argparse.Namespace) -> Path:
	"""Return the parsed standard config path."""
	config_path = getattr(args, 'config', None)
	if config_path is None:
		msg = 'config path is required'
		raise ValueError(msg)
	return Path(config_path)


def load_config_for_cli(
	config_path: Path,
	*,
	loader: Callable[[Path], Mapping[str, object]],
) -> Mapping[str, object]:
	"""Load a config through the provided repository config loader."""
	if not config_path.exists():
		msg = f'config file does not exist: {config_path}'
		raise FileNotFoundError(msg)
	if not config_path.is_file():
		msg = f'config path is not a file: {config_path}'
		raise FileNotFoundError(msg)
	return loader(config_path)


def resolve_config_for_cli(
	raw_config: Mapping[str, object],
	*,
	resolver: Callable[[Mapping[str, object]], T],
	config_path: Path,
) -> T:
	"""Resolve a raw config and include the source path in resolver failures."""
	try:
		return resolver(raw_config)
	except Exception as exc:
		msg = f'{exc} (config: {config_path})'
		raise type(exc)(msg) from exc


def print_cli_summary(title: str, items: Mapping[str, object]) -> None:
	"""Print a compact title and key/value run summary."""
	print(title)
	for key, value in items.items():
		print(f'{key}: {_format_cli_value(value)}')


def add_device_argument(parser: argparse.ArgumentParser) -> None:
	"""Add the common device override flag."""
	parser.add_argument(
		'--device',
		choices=('auto', 'cpu', 'cuda'),
		help='Device override.',
	)


def add_dry_run_argument(parser: argparse.ArgumentParser) -> None:
	"""Add the common dry-run flag."""
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate the config and print a run summary without executing.',
	)


def add_skip_existing_argument(parser: argparse.ArgumentParser) -> None:
	"""Add the common skip-existing flag."""
	parser.add_argument(
		'--skip-existing',
		action='store_true',
		help='Skip outputs that already exist.',
	)


def add_overwrite_argument(parser: argparse.ArgumentParser) -> None:
	"""Add the common overwrite flag."""
	parser.add_argument(
		'--overwrite',
		action='store_true',
		help='Replace existing outputs.',
	)


def _format_cli_value(value: object) -> str:
	if isinstance(value, bool):
		return str(value).lower()
	if isinstance(value, list):
		return ', '.join(str(item) for item in value)
	if value is None:
		return 'null'
	return str(value)


__all__ = [
	'add_config_argument',
	'add_device_argument',
	'add_dry_run_argument',
	'add_overwrite_argument',
	'add_skip_existing_argument',
	'load_config_for_cli',
	'parse_config_path',
	'print_cli_summary',
	'resolve_config_for_cli',
]
