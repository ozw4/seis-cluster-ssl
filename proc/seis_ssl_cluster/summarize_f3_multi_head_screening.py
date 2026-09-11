"""Audit or summarize Multi-Head screening against historical canonical HMM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology.multi_head_screening_results import (
	inspect_multi_head_screening,
	screening_config_from_mapping,
	summarize_multi_head_screening,
)


def main() -> None:
	"""Require live evidence even in read-only check mode."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument(
		'--check-only', '--dry-run', dest='check_only', action='store_true'
	)
	args = parser.parse_args()
	config = screening_config_from_mapping(load_config(args.config))
	if args.check_only:
		print(
			json.dumps(inspect_multi_head_screening(config), allow_nan=False, indent=2)
		)
	else:
		for path in summarize_multi_head_screening(config):
			print(f'output: {path}')


if __name__ == '__main__':
	main()
