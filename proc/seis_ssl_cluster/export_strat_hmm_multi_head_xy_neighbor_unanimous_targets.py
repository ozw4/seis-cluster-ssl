"""Export immutable source-label XY-neighbour unanimous hard targets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.stratigraphy.xy_neighbor_unanimous_targets import (
	export_multi_head_xy_neighbor_unanimous_targets,
	resolve_multi_head_xy_neighbor_unanimous_target_export_config,
)

if TYPE_CHECKING:
	from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
	"""Build the fixed-policy source-label unanimous exporter parser."""
	parser = argparse.ArgumentParser(
		description=(
			'Export immutable K=6/8/10 source-label XY-neighbour unanimous hard '
			'targets.'
		)
	)
	parser.add_argument('--config', required=True, type=Path)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--only-missing', action='store_true')
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	"""Plan or atomically publish the complete source-only target bundle."""
	args = build_parser().parse_args(argv)
	config = resolve_multi_head_xy_neighbor_unanimous_target_export_config(
		load_config(args.config)
	)
	plans = export_multi_head_xy_neighbor_unanimous_targets(
		config,
		dry_run=args.dry_run,
		only_missing=args.only_missing,
	)
	for plan in plans:
		print(f'k={plan.k} planned action: {plan.action}')
		if plan.reason:
			print(f'k={plan.k} detail: {plan.reason}')
	if args.dry_run:
		print('execution: dry-run; no arrays or manifests written')
	else:
		print(f'XY-neighbour-unanimous target manifest: {config.handoff_manifest}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
