"""Publish the lightweight unanimous XY-neighbour pretraining review."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.f3.xy_neighbor_unanimous_results import (
	load_f3_xy_neighbor_unanimous_review_config,
	publish_f3_xy_neighbor_unanimous_review,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the standalone unanimous review publisher parser."""
	parser = argparse.ArgumentParser(
		description='Publish F3 unanimous XY-neighbour pretraining review.'
	)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--dry-run', action='store_true')
	return parser


def main() -> int:
	"""Validate the immutable lineage and publish lightweight review files."""
	args = build_parser().parse_args()
	result = publish_f3_xy_neighbor_unanimous_review(
		load_f3_xy_neighbor_unanimous_review_config(args.config),
		dry_run=args.dry_run,
	)
	print(f'summary JSON: {result.summary_json}')
	print(f'summary Markdown: {result.summary_markdown}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
