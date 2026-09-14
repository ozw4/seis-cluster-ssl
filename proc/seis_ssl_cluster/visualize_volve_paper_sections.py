"""Create real-coordinate Volve paper sections without training or scoring."""

from __future__ import annotations

import argparse
import json

from seis_ssl_cluster.volve.paper_sections import run


def build_parser() -> argparse.ArgumentParser:
	"""Build the command-line parser without reading artifacts."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', required=True)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--overwrite', action='store_true')
	return parser


def main() -> None:
	"""Parse the one visualization configuration and optional read-only preflight."""
	args = build_parser().parse_args()
	print(
		json.dumps(
			run(args.config, dry_run=args.dry_run, overwrite=args.overwrite), indent=2
		)
	)


if __name__ == '__main__':
	main()
