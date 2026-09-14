"""Export frozen Parihaka paper-comparison predictions without evaluation."""

import argparse
import json
from pathlib import Path

from seis_ssl_cluster.parihaka.paper_prediction import load_config, run


def build_parser() -> argparse.ArgumentParser:
	"""Build the command-line parser without reading artifacts."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--dry-run', action='store_true')
	return parser


def main() -> None:
	"""Run explicit-config preflight or prediction export."""
	args = build_parser().parse_args()
	print(
		json.dumps(
			run(load_config(args.config.resolve()), dry_run=args.dry_run), indent=2
		)
	)


if __name__ == '__main__':
	main()
