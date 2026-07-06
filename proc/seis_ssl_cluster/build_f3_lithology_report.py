"""Build F3 token-level lithology probe reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology import (
	f3_lithology_publish_config_from_mapping,
	f3_lithology_report_config_from_mapping,
)
from seis_ssl_cluster.f3 import (
	F3LithologyPublishConfig,
	F3LithologyReportConfig,
	build_f3_lithology_report,
)

STAGE = 'build_f3_lithology_report'
DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '50_lithology'
	/ 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	/ 'overlap_x16'
	/ 'png_slices_segy_labels_v1'
	/ '06_build_lithology_report.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for F3 lithology probe reports."""
	return build_config_parser(
		'Build an F3 lithology probe report.',
		default_config=DEFAULT_CONFIG,
		dry_run_help=(
			'Validate the config and print a run summary without writing reports.'
		),
	)


def main() -> None:
	"""Build an F3 lithology probe report or print a dry-run summary."""
	parser = build_parser()
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=f3_lithology_report_config_from_mapping,
		config_path=config_path,
	)
	publish_config = f3_lithology_publish_config_from_mapping(
		raw_config.get('publish'),
	)
	if args.dry_run:
		_print_summary(config, publish_config=publish_config)
		print('execution: dry-run; F3 lithology report skipped')
		return

	result = build_f3_lithology_report(config, publish_config=publish_config)
	warnings = result.payload.get('warnings', [])
	warning_count = len(warnings) if isinstance(warnings, Sequence) else 0
	print(f'f3_lithology_report.warning_count: {warning_count}')
	print(f'f3_lithology_report.markdown: {result.report_markdown}')
	print(f'f3_lithology_report.json: {result.report_json}')
	if result.comparison_csv is not None:
		print(f'f3_lithology_report.comparison_csv: {result.comparison_csv}')
	if result.comparison_markdown is not None:
		print(
			'f3_lithology_report.comparison_markdown: '
			f'{result.comparison_markdown}',
		)
	if result.publish_manifest is not None:
		print(f'published F3 lithology report: {result.publish_manifest.output_dir}')
		print(f'wrote publish manifest: {result.publish_manifest.manifest_path}')


def _print_summary(
	config: F3LithologyReportConfig,
	*,
	publish_config: F3LithologyPublishConfig,
) -> None:
	print(f'stage: {STAGE}')
	print(f'model.tag: {config.model.get("tag")}')
	print(f'model.checkpoint: {config.model.get("checkpoint")}')
	print(f'lithology.root: {config.lithology.get("root")}')
	print(f'probe.spec: {config.probe.get("spec")}')
	print(f'probe.metrics_json: {config.metrics_json}')
	print(f'probe.probe_config_resolved_json: {config.probe_config_json}')
	print(f'reports.output_dir: {config.output_dir}')
	print(f'reports.output_markdown: {config.output_markdown}')
	print(f'reports.output_json: {config.output_json}')
	print(f'predictions.metadata_json: {config.prediction_metadata_json}')
	print(f'visualizations.metadata_json: {config.visualization_metadata_json}')
	if config.comparison is not None:
		print(f'comparison.search_root: {config.comparison.search_root}')
		print(f'comparison.output_csv: {config.comparison.output_csv}')
		print(f'comparison.output_markdown: {config.comparison.output_markdown}')
	print(f'publish.enabled: {publish_config.enabled}')
	if publish_config.output_dir is not None:
		print(f'publish.output_dir: {publish_config.output_dir}')
	print(f'publish.include_figures: {publish_config.include_figures}')
	print(
		'publish.max_file_size_bytes: '
		f'{publish_config.max_file_size_bytes}',
	)
	print(
		'publish.max_prediction_figures: '
		f'{publish_config.max_prediction_figures}',
	)


if __name__ == '__main__':
	main()
