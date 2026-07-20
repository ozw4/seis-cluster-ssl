"""YAML IO for amplitude-only seismic SSL clustering configs."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import yaml

from seis_ssl_cluster.config.schema import KNOWN_STAGES
from seis_ssl_cluster.config.validate import validate_config

_ENVIRONMENT_VARIABLE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')


def load_config(path: str | Path) -> dict[str, object]:
	"""Load a YAML config file as a raw mapping."""
	config_path = Path(path)
	with config_path.open(encoding='utf-8') as file_obj:
		loaded = yaml.safe_load(file_obj)

	if not isinstance(loaded, dict):
		msg = f'config file must contain a mapping: {config_path}'
		raise TypeError(msg)

	return _expand_environment_variables(loaded)


def _expand_environment_variables(value: object) -> object:
	"""Expand required ``${NAME}`` values in a loaded YAML structure."""
	if isinstance(value, dict):
		return {
			key: _expand_environment_variables(child)
			for key, child in value.items()
		}
	if isinstance(value, list):
		return [_expand_environment_variables(child) for child in value]
	if not isinstance(value, str):
		return value

	def replace(match: re.Match[str]) -> str:
		name = match.group(1)
		try:
			return os.environ[name]
		except KeyError as exc:
			msg = f'config environment variable is required: {name}'
			raise ValueError(msg) from exc

	return _ENVIRONMENT_VARIABLE.sub(replace, value)


def main() -> None:
	"""Load, validate, and print a compact JSON summary for one config file."""
	parser = argparse.ArgumentParser(
		description='Validate a SeisSSLCluster amplitude-only config YAML file.',
	)
	parser.add_argument('config_path', type=Path)
	parser.add_argument(
		'--stage',
		required=True,
		choices=sorted(KNOWN_STAGES),
		help='Pipeline stage selected by the caller.',
	)
	args = parser.parse_args()

	config = validate_config(load_config(args.config_path), stage=args.stage)
	summary = {
		'stage': config.get('stage'),
		'paths': config.get('paths', {}),
		'data': config.get('data', {}),
		'model': config.get('model', {}),
	}
	print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
	main()
