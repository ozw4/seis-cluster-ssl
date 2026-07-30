"""Publish lightweight F3 XY-neighbour consensus review artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.f3.xy_neighbor_consensus_results import (
	load_f3_xy_neighbor_consensus_review_config,
	publish_f3_xy_neighbor_consensus_review,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the config-driven successor review publisher parser."""
	parser = argparse.ArgumentParser(
		description='Publish F3 XY-neighbour consensus target and pretraining review.'
	)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--dry-run', action='store_true')
	return parser


def main() -> int:
	"""Publish (or plan) lightweight source-only review artifacts."""
	args = build_parser().parse_args()
	result = publish_f3_xy_neighbor_consensus_review(
		load_f3_xy_neighbor_consensus_review_config(args.config), dry_run=args.dry_run
	)
	for path in (result.summary_json, result.summary_markdown):
		print(f'output: {path}')
	if result.publish_manifest is not None:
		print(f'publish_manifest: {result.publish_manifest.manifest_path}')
	if args.dry_run:
		print('execution: dry-run; review artifacts were not written')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
