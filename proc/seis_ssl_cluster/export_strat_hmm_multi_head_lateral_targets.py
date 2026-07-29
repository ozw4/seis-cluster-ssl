"""Export immutable one-step XY lateral hard targets from frozen HMM sources."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.stratigraphy import (
	export_multi_head_lateral_targets,
	resolve_multi_head_lateral_target_export_config,
)

if TYPE_CHECKING:
	from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
	"""Build the fixed-policy lateral target exporter parser."""
	parser = argparse.ArgumentParser(
		description='Export immutable K=6/8/10 one-step lateral HMM targets.'
	)
	parser.add_argument('--config', required=True, type=Path)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--only-missing', action='store_true')
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	"""Plan or atomically publish the complete lateral target bundle."""
	args = build_parser().parse_args(argv)
	config = resolve_multi_head_lateral_target_export_config(load_config(args.config))
	plans = export_multi_head_lateral_targets(
		config, dry_run=args.dry_run, only_missing=args.only_missing
	)
	if args.dry_run:
		affinity_scale = plans[0].affinity_scale
		if affinity_scale is None:
			raise RuntimeError('lateral dry-run did not resolve affinity scale')
		print(f'resolved affinity scale: {affinity_scale:.17g}')
	for plan in plans:
		print(f'k={plan.k} planned action: {plan.action}')
		if args.dry_run:
			emission_gap_scale = plan.emission_gap_scale
			if emission_gap_scale is None:
				raise RuntimeError(
					'lateral dry-run did not resolve emission-gap scale'
				)
			print(
				f'k={plan.k} resolved emission-gap scale: '
				f'{emission_gap_scale:.17g}'
			)
		if plan.reason:
			print(f'k={plan.k} detail: {plan.reason}')
	if args.dry_run:
		print('execution: dry-run; no arrays or manifests written')
	else:
		print(f'lateral target manifest: {config.handoff_manifest}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
