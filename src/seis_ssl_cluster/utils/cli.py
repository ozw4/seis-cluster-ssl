"""Shared helpers for thin seismic SSL clustering procedure entrypoints."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import load_config, validate_config


def parse_config_args(
	description: str,
	default_config: str | Path | None = None,
) -> argparse.Namespace:
	"""Parse common config and dry-run arguments."""
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument(
		'--config',
		type=Path,
		default=Path(default_config) if default_config is not None else None,
		required=default_config is None,
		help='Path to a YAML configuration file.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate the config and print a run summary without executing.',
	)
	return parser.parse_args()


def print_config_summary(cfg: Mapping[str, Any]) -> None:
	"""Print a compact summary of validated amplitude-only config values."""
	paths = _mapping(cfg.get('paths'))
	data = _mapping(cfg.get('data'))
	model = _mapping(cfg.get('model'))
	masking = _mapping(cfg.get('masking'))
	train = _mapping(cfg.get('train'))

	rows: list[tuple[str, Any]] = [
		('stage', cfg.get('stage')),
		('paths.nopims_root', paths.get('nopims_root')),
		('paths.artifact_root', paths.get('artifact_root')),
		('data.grid_order', data.get('grid_order')),
		('data.volume_format', data.get('volume_format')),
		('data.local_crop_size', data.get('local_crop_size')),
		('data.input_channels', data.get('input_channels')),
		('data.target_channels', data.get('target_channels')),
		('data.use_context', data.get('use_context')),
		('model.name', model.get('name')),
		('model.patch_size', model.get('patch_size')),
		('masking.spatial_mask_ratio', masking.get('spatial_mask_ratio')),
		('masking.spatial_mask_mode', masking.get('spatial_mask_mode')),
		('masking.block_size_tokens', masking.get('block_size_tokens')),
		('train.batch_size', train.get('batch_size')),
		('train.epochs', train.get('epochs')),
		('train.device', train.get('device')),
	]

	for key, value in rows:
		print(f'{key}: {_format_value(value)}')


def run_pending_entrypoint(
	description: str,
	default_config: str | Path | None = None,
) -> None:
	"""Validate config, print a summary, and report pending execution."""
	args = parse_config_args(description, default_config)
	config = validate_config(load_config(args.config))
	print_config_summary(config)
	if args.dry_run:
		print('execution: dry-run; implementation pending')
		return

	message = (
		f'execution pending for stage {config.get("stage")!r}; '
		'use --dry-run to validate configuration only'
	)
	raise SystemExit(message)


def _mapping(value: object) -> Mapping[str, Any]:
	if isinstance(value, Mapping):
		return value
	return {}


def _format_value(value: object) -> str:
	if isinstance(value, bool):
		return str(value).lower()
	if isinstance(value, list):
		return ', '.join(str(item) for item in value)
	if value is None:
		return 'null'
	return str(value)


__all__ = [
	'parse_config_args',
	'print_config_summary',
	'run_pending_entrypoint',
]
