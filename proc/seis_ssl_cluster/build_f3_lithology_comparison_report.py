"""Build F3 lithology pretrained-vs-baseline comparison reports."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from seis_ssl_cluster.f3 import (
	F3LithologyComparisonReportConfig,
	build_f3_lithology_comparison_report,
)

STAGE = 'build_f3_lithology_comparison_report'
DEFAULT_SEARCH_ROOT = (
	Path('/workspace')
	/ 'artifacts'
	/ 'seis_ssl_cluster'
	/ 'lithology'
	/ 'f3'
	/ 'facies_benchmark_v1'
)
DEFAULT_OUTPUT_DIR = DEFAULT_SEARCH_ROOT / 'reports' / 'baseline_comparison'


def main() -> None:
	"""Build an F3 lithology comparison report or print a dry-run summary."""
	parser = ArgumentParser(
		description='Build an F3 lithology pretrained-vs-baseline comparison report.',
	)
	parser.add_argument(
		'--search-root',
		type=Path,
		default=DEFAULT_SEARCH_ROOT,
		help='Artifact tree to search for probe metrics.json files.',
	)
	parser.add_argument(
		'--output-dir',
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help='Directory for comparison_table.csv, comparison_report.md, and figures.',
	)
	parser.add_argument(
		'--output-csv',
		type=Path,
		default=None,
		help='Explicit comparison table output path.',
	)
	parser.add_argument(
		'--output-markdown',
		type=Path,
		default=None,
		help='Explicit Markdown report output path.',
	)
	parser.add_argument(
		'--metrics-json',
		type=Path,
		action='append',
		default=[],
		help='Explicit metrics.json path. May be passed multiple times.',
	)
	parser.add_argument(
		'--figure-dpi',
		type=int,
		default=300,
		help='Figure DPI. Values below 300 are raised to 300.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Print the resolved outputs without writing reports.',
	)
	args = parser.parse_args()

	config = F3LithologyComparisonReportConfig(
		search_root=args.search_root,
		output_csv=args.output_csv or args.output_dir / 'comparison_table.csv',
		output_markdown=(
			args.output_markdown or args.output_dir / 'comparison_report.md'
		),
		metrics_paths=tuple(args.metrics_json),
		figure_dpi=args.figure_dpi,
	)
	if args.dry_run:
		_print_summary(config)
		print('execution: dry-run; F3 lithology comparison report skipped')
		return

	result = build_f3_lithology_comparison_report(config)
	print(f'f3_lithology_comparison_report.warning_count: {len(result.warnings)}')
	print(f'f3_lithology_comparison_report.rows: {len(result.rows)}')
	print(f'f3_lithology_comparison_report.csv: {result.comparison_csv}')
	print(f'f3_lithology_comparison_report.markdown: {result.comparison_markdown}')
	for path in result.figure_paths:
		print(f'f3_lithology_comparison_report.figure: {path}')


def _print_summary(config: F3LithologyComparisonReportConfig) -> None:
	print(f'stage: {STAGE}')
	print(f'comparison.search_root: {config.search_root}')
	print(f'comparison.output_csv: {config.output_csv}')
	print(f'comparison.output_markdown: {config.output_markdown}')
	print(f'comparison.figure_dpi: {config.figure_dpi}')
	if config.metrics_paths:
		for path in config.metrics_paths:
			print(f'comparison.metrics_json: {path}')
	else:
		print('comparison.metrics_json: discovered from search_root')


if __name__ == '__main__':
	main()
