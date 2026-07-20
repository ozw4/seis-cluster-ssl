"""Export stratigraphic HMM clustering labels as pseudo-target artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	add_dry_run_argument,
	add_overwrite_argument,
)
from seis_ssl_cluster.stratigraphy import (
	ExportedPseudoTargetResult,
	export_hmm_cluster_labels_as_pseudo_targets,
	prepare_hmm_cluster_label_pseudo_target_exports,
)

if TYPE_CHECKING:
	from collections.abc import Iterable, Sequence


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for stratigraphic HMM pseudo-target export."""
	parser = argparse.ArgumentParser(
		description='Export stratigraphic HMM cluster labels as pseudo-targets.',
	)
	parser.add_argument(
		'--clustering-output-dir',
		type=Path,
		required=True,
		help='Directory containing labels/k{k} stratigraphic HMM outputs.',
	)
	parser.add_argument(
		'--pseudo-target-root',
		type=Path,
		required=True,
		help='Root directory for exported pseudo-target artifacts.',
	)
	parser.add_argument('--k', type=int, required=True, help='Cluster count to export.')
	parser.add_argument(
		'--confidence',
		type=float,
		default=1.0,
		help='Constant confidence assigned to valid tokens.',
	)
	parser.add_argument(
		'--boundary-alpha',
		type=float,
		default=0.0,
		help='Boundary downweighting strength in [0, 1].',
	)
	parser.add_argument(
		'--boundary-tau',
		type=float,
		default=1.0,
		help='Positive exponential boundary-distance scale.',
	)
	parser.add_argument(
		'--schema-version',
		type=int,
		default=2,
		choices=(1, 2),
		help='Pseudo-target schema version (K=8/10 bootstrap exports use 1).',
	)
	parser.add_argument(
		'--no-boundary-weight',
		action='store_true',
		help='Omit the boundary-weight artifact; required for schema v1.',
	)
	add_overwrite_argument(parser)
	add_dry_run_argument(
		parser,
		help_text='Validate inputs and print a summary without writing arrays.',
	)
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	"""Export pseudo-target artifacts or print a dry-run summary."""
	parser = build_parser()
	args = parser.parse_args(argv)
	if args.dry_run:
		results = prepare_hmm_cluster_label_pseudo_target_exports(
			clustering_output_dir=args.clustering_output_dir,
			pseudo_target_root=args.pseudo_target_root,
			k=args.k,
			confidence=args.confidence,
			boundary_alpha=args.boundary_alpha,
			boundary_tau=args.boundary_tau,
			overwrite=args.overwrite,
			schema_version=args.schema_version,
			write_boundary_weight=not args.no_boundary_weight,
		)
		_print_summary(
			results,
			dry_run=True,
			boundary_alpha=args.boundary_alpha,
			boundary_tau=args.boundary_tau,
		)
		return 0

	results = export_hmm_cluster_labels_as_pseudo_targets(
		clustering_output_dir=args.clustering_output_dir,
		pseudo_target_root=args.pseudo_target_root,
		k=args.k,
		confidence=args.confidence,
		boundary_alpha=args.boundary_alpha,
		boundary_tau=args.boundary_tau,
		overwrite=args.overwrite,
		schema_version=args.schema_version,
		write_boundary_weight=not args.no_boundary_weight,
	)
	_print_summary(
		results,
		dry_run=False,
		boundary_alpha=args.boundary_alpha,
		boundary_tau=args.boundary_tau,
	)
	return 0


def _print_summary(
	results: Iterable[ExportedPseudoTargetResult],
	*,
	dry_run: bool,
	boundary_alpha: float,
	boundary_tau: float,
) -> None:
	result_list = list(results)
	status = 'dry-run; no files written' if dry_run else 'written'
	print(f'pseudo_target_exports: {len(result_list)}')
	print(f'execution: {status}')
	print(f'boundary_weighting: alpha={boundary_alpha} tau={boundary_tau}')
	for item in result_list:
		print(
			f'{item.survey_id}: valid_tokens={item.valid_token_count} '
			f'labels={item.labels_path} '
			f'boundary_weight={item.boundary_weight_path}',
		)


if __name__ == '__main__':
	raise SystemExit(main())
