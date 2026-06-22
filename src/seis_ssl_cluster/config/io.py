"""YAML IO for amplitude-only seismic SSL clustering configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from seis_ssl_cluster.config.schema import DEFAULT_ARTIFACT_ROOT, DEFAULT_NOPIMS_ROOT
from seis_ssl_cluster.config.validate import validate_config


def load_config(path: str | Path) -> dict[str, object]:
	"""Load a YAML config file and apply path defaults."""
	config_path = Path(path)
	with config_path.open(encoding='utf-8') as file_obj:
		loaded = yaml.safe_load(file_obj)

	if not isinstance(loaded, dict):
		msg = f'config file must contain a mapping: {config_path}'
		raise TypeError(msg)

	paths = loaded.setdefault('paths', {})
	if not isinstance(paths, dict):
		msg = 'paths must be a mapping'
		raise TypeError(msg)
	paths.setdefault('nopims_root', DEFAULT_NOPIMS_ROOT)
	paths.setdefault('artifact_root', DEFAULT_ARTIFACT_ROOT)

	return loaded


def main() -> None:
	"""Load, validate, and print a compact JSON summary for one config file."""
	parser = argparse.ArgumentParser(
		description='Validate a SeisSSLCluster amplitude-only config YAML file.',
	)
	parser.add_argument('config_path', type=Path)
	args = parser.parse_args()

	config = validate_config(load_config(args.config_path))
	summary = {
		'stage': config.get('stage'),
		'paths': config.get('paths', {}),
		'data': config.get('data', {}),
		'model': config.get('model', {}),
	}
	print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
	main()
