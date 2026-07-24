"""Export the strict schema-v1 K=6/8/10 HMM pseudo-target bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.stratigraphy import (
	export_multi_head_pseudo_targets,
	resolve_multi_head_pseudo_target_export_config,
)

if TYPE_CHECKING:
	from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
	"""Build the strict multi-head export CLI parser."""
	parser = argparse.ArgumentParser(
		description='Export K=6/8/10 schema-v1 HMM pseudo-targets from one config.',
	)
	parser.add_argument('--config', required=True, type=Path)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--only-missing', action='store_true')
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	"""Validate, plan, and optionally publish the complete target bundle."""
	args = build_parser().parse_args(argv)
	config = resolve_multi_head_pseudo_target_export_config(load_config(args.config))
	plans = export_multi_head_pseudo_targets(
		config,
		dry_run=args.dry_run,
		only_missing=args.only_missing,
	)
	print(f'source clustering root: {config.clustering_output_dir}')
	print(
		f'source clustering config: {config.clustering_config} '
		f'sha256={file_sha256(config.clustering_config)}'
	)
	print(f'source embedding root: {config.source_embedding_dir}')
	print(
		'schema/boundary/confidence policy: '
		f'schema={config.schema_version} boundary_weight=absent '
		f'confidence={config.confidence}',
	)
	for plan in plans:
		print(f'k={plan.k} source labels:')
		for label in plan.source_labels:
			print(f'  {label} sha256={file_sha256(label)}')
		print(f'k={plan.k} output root: {config.pseudo_target_root / f"k{plan.k}"}')
		print(f'k={plan.k} planned action: {plan.action}')
		if plan.reason:
			print(f'k={plan.k} detail: {plan.reason}')
	if args.dry_run:
		print('execution: dry-run; no arrays or manifests written')
	else:
		print(f'handoff manifest: {config.handoff_manifest}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
