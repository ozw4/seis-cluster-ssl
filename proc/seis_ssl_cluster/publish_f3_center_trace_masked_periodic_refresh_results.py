"""Publish lightweight review artifacts for experiment 107."""
# ruff: noqa: CPY001

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.f3.center_trace_masked_periodic_refresh_results import (
	load_f3_center_trace_masked_periodic_refresh_review_config,
	publish_f3_center_trace_masked_periodic_refresh_review,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the config-driven periodic-refresh review publisher parser."""
	parser = argparse.ArgumentParser(
		description=(
			'Publish F3 periodic center-trace masked pretraining review artifacts.'
		),
	)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--quarantine-invalid', action='store_true')
	return parser


def main() -> int:
	"""Publish or plan the lightweight periodic-refresh review tree."""
	args = build_parser().parse_args()
	result = publish_f3_center_trace_masked_periodic_refresh_review(
		load_f3_center_trace_masked_periodic_refresh_review_config(args.config),
		dry_run=args.dry_run,
		quarantine_invalid=args.quarantine_invalid,
	)
	for path in (
		result.summary_json,
		result.summary_markdown,
		result.refresh_events,
		result.generation_summary,
		result.checkpoint_summary,
		result.pretraining_handoff,
	):
		print(f'output: {path}')
	if args.dry_run:
		print('execution: dry-run; review artifacts were not written')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
