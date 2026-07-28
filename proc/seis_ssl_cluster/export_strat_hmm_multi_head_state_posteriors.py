"""Export immutable K=6/8/10 state posterior artifacts from frozen HMMs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.stratigraphy import (
	export_multi_head_state_posteriors,
	resolve_multi_head_state_posterior_export_config,
)

if TYPE_CHECKING:
	from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
	"""Build the frozen posterior export command-line parser."""
	parser = argparse.ArgumentParser(
		description='Export frozen K=6/8/10 HMM state posteriors.',
	)
	parser.add_argument('--config', required=True, type=Path)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--only-missing', action='store_true')
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	"""Plan or publish a complete immutable posterior manifest."""
	args = build_parser().parse_args(argv)
	config = resolve_multi_head_state_posterior_export_config(load_config(args.config))
	plans = export_multi_head_state_posteriors(
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
		print(f'posterior manifest: {config.handoff_manifest}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
