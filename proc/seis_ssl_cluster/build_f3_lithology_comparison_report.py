# ruff: noqa: CPY001
"""Build F3 lithology pretrained-vs-baseline comparison reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import argparse
	from argparse import Namespace
	from collections.abc import Mapping
	from pathlib import Path

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
	if args.config is None:
		if args.search_root is None:
			parser.error('--search-root is required when --config is not provided')
		if args.output_dir is None and (
			args.output_csv is None or args.output_markdown is None
		):
			parser.error(
				'--output-dir or both --output-csv and --output-markdown are '
				'required when --config is not provided',
			)

	raw_config = (
		load_config_for_cli(args.config, loader=load_config)
		if args.config is not None
		else None
	)
	try:
		config = _config_from_args(
			args,
			raw_config=raw_config,
			config_path=args.config,
		)
	except (TypeError, ValueError) as exc:
		parser.error(str(exc))
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
		resolver_input = dict(raw_config)
	else:
		resolver_input = {}
	comparison = dict(resolver_input.get('comparison', {}))
	for key, value in (
		('search_root', args.search_root),
		('output_dir', args.output_dir),
		('output_csv', args.output_csv),
		('output_markdown', args.output_markdown),
		('figure_dpi', args.figure_dpi),
	):
		if value is not None:
			comparison[key] = value if key == 'figure_dpi' else str(value)
	if args.metrics_json:
		comparison['metrics_json'] = [str(path) for path in args.metrics_json]
	resolver_input['comparison'] = comparison
	if config_path is None:
		return f3_lithology_comparison_report_config_from_mapping(resolver_input)
	return resolve_config_for_cli(
		resolver_input,
		resolver=f3_lithology_comparison_report_config_from_mapping,
		config_path=config_path,
	)


if __name__ == '__main__':
	main()
