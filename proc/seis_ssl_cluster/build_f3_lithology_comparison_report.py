"""Build F3 lithology pretrained-vs-baseline comparison reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from argparse import Namespace
	from collections.abc import Mapping

from seis_ssl_cluster.cli import (
	add_append_path_argument,
	add_path_argument,
	build_config_parser,
	load_config_for_cli,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_baselines import (
	f3_lithology_comparison_publish_config_from_mapping,
	f3_lithology_comparison_report_config_from_mapping,
)
from seis_ssl_cluster.f3 import (
	F3LithologyComparisonPublishConfig,
	F3LithologyComparisonReportConfig,
	build_f3_lithology_comparison_report,
)

STAGE = 'build_f3_lithology_comparison_report'
DEFAULT_SEARCH_ROOT = Path(
	'/workspace/artifacts/seis_ssl_cluster/lithology/f3/facies_benchmark_v1'
)
DEFAULT_OUTPUT_DIR = DEFAULT_SEARCH_ROOT / 'reports' / 'baseline_comparison'


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for F3 lithology comparison reports."""
	parser = build_config_parser(
		'Build an F3 lithology pretrained-vs-baseline comparison report.',
		config_required=False,
		dry_run_help='Print the resolved outputs without writing reports.',
	)
	add_path_argument(
		parser,
		'--search-root',
		default=None,
		help_text='Artifact tree to search for probe metrics.json files.',
	)
	add_path_argument(
		parser,
		'--output-dir',
		default=None,
		help_text=(
			'Directory for comparison_table.csv, comparison_report.md, and figures.'
		),
	)
	add_path_argument(
		parser,
		'--output-csv',
		default=None,
		help_text='Explicit comparison table output path.',
	)
	add_path_argument(
		parser,
		'--output-markdown',
		default=None,
		help_text='Explicit Markdown report output path.',
	)
	add_append_path_argument(
		parser,
		'--metrics-json',
		help_text='Explicit metrics.json path. May be passed multiple times.',
	)
	parser.add_argument(
		'--figure-dpi',
		type=int,
		default=None,
		help='Figure DPI. Values below 300 are raised to 300.',
	)
	return parser


def main() -> None:
	"""Build an F3 lithology comparison report or print a dry-run summary."""
	parser = build_parser()
	args = parser.parse_args()

	raw_config = (
		load_config_for_cli(args.config, loader=load_config)
		if args.config is not None
		else None
	)
	config = _config_from_args(args, raw_config=raw_config, config_path=args.config)
	publish_config = f3_lithology_comparison_publish_config_from_mapping(
		None if raw_config is None else raw_config.get('publish'),
	)
	if args.dry_run:
		_print_summary(config, publish_config=publish_config)
		print('execution: dry-run; F3 lithology comparison report skipped')
		return

	result = build_f3_lithology_comparison_report(
		config,
		publish_config=publish_config,
	)
	print(f'f3_lithology_comparison_report.warning_count: {len(result.warnings)}')
	print(f'f3_lithology_comparison_report.rows: {len(result.rows)}')
	print(f'f3_lithology_comparison_report.csv: {result.comparison_csv}')
	print(f'f3_lithology_comparison_report.markdown: {result.comparison_markdown}')
	for path in result.figure_paths:
		print(f'f3_lithology_comparison_report.figure: {path}')
	if result.publish_manifest is not None:
		print(
			'published F3 lithology comparison report: '
			f'{result.publish_manifest.output_dir}',
		)
		print(f'wrote publish manifest: {result.publish_manifest.manifest_path}')


def _print_summary(
	config: F3LithologyComparisonReportConfig,
	*,
	publish_config: F3LithologyComparisonPublishConfig,
) -> None:
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
	print(f'publish.enabled: {publish_config.enabled}')
	if publish_config.output_dir is not None:
		print(f'publish.output_dir: {publish_config.output_dir}')
	print(f'publish.include_figures: {publish_config.include_figures}')
	print(
		'publish.max_file_size_bytes: '
		f'{publish_config.max_file_size_bytes}',
	)


def _config_from_args(
	args: Namespace,
	*,
	raw_config: Mapping[str, object] | None,
	config_path: Path | None,
) -> F3LithologyComparisonReportConfig:
	if args.config is not None:
		if raw_config is None:
			msg = 'raw_config is required when args.config is set'
			raise ValueError(msg)
		if config_path is None:
			msg = 'config_path is required when args.config is set'
			raise ValueError(msg)
		config = resolve_config_for_cli(
			raw_config,
			resolver=f3_lithology_comparison_report_config_from_mapping,
			config_path=config_path,
		)
	else:
		output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
		config = F3LithologyComparisonReportConfig(
			search_root=args.search_root or DEFAULT_SEARCH_ROOT,
			output_csv=args.output_csv or output_dir / 'comparison_table.csv',
			output_markdown=(
				args.output_markdown or output_dir / 'comparison_report.md'
			),
			metrics_paths=tuple(args.metrics_json),
			figure_dpi=args.figure_dpi or 300,
		)
	if args.config is None:
		return config
	return _config_with_overrides(
		config,
		search_root=args.search_root,
		output_dir=args.output_dir,
		output_csv=args.output_csv,
		output_markdown=args.output_markdown,
		metrics_paths=tuple(args.metrics_json),
		figure_dpi=args.figure_dpi,
	)


def _config_with_overrides(  # noqa: PLR0913
	config: F3LithologyComparisonReportConfig,
	*,
	search_root: Path | None,
	output_dir: Path | None,
	output_csv: Path | None,
	output_markdown: Path | None,
	metrics_paths: tuple[Path, ...],
	figure_dpi: int | None,
) -> F3LithologyComparisonReportConfig:
	resolved_output_dir = output_dir or config.output_markdown.parent
	return F3LithologyComparisonReportConfig(
		search_root=search_root or config.search_root,
		output_csv=output_csv or (
			resolved_output_dir / 'comparison_table.csv'
			if output_dir is not None
			else config.output_csv
		),
		output_markdown=output_markdown or (
			resolved_output_dir / 'comparison_report.md'
			if output_dir is not None
			else config.output_markdown
		),
		metrics_paths=metrics_paths or config.metrics_paths,
		figure_dpi=figure_dpi or config.figure_dpi,
		figure_style=config.figure_style,
	)


if __name__ == '__main__':
	main()
