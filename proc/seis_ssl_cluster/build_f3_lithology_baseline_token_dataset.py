"""Build F3 lithology baseline token datasets from reference token splits."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3 import (
	F3LithologyBaselineTokenDatasetConfig,
	build_f3_lithology_baseline_token_dataset,
	f3_lithology_baseline_token_dataset_config_from_mapping,
)

STAGE = 'build_f3_lithology_baseline_token_dataset'
DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '50_lithology_baselines'
	/ 'z_only_v1'
	/ '01_build_baseline_token_dataset.yaml'
)


def main() -> None:
	"""Build an F3 lithology baseline token dataset or print a dry-run summary."""
	parser = ArgumentParser(
		description='Build F3 lithology baseline token datasets.',
	)
	parser.add_argument(
		'--config',
		type=Path,
		default=DEFAULT_CONFIG,
		help='Path to a YAML configuration file.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate the config and print a run summary without writing outputs.',
	)
	args = parser.parse_args()

	raw_config = load_config(args.config)
	config = f3_lithology_baseline_token_dataset_config_from_mapping(raw_config)
	if args.dry_run:
		_print_summary(config)
		print('execution: dry-run; F3 lithology baseline token dataset build skipped')
		return

	result = build_f3_lithology_baseline_token_dataset(config)
	print(f'f3_lithology_baseline_token_dataset.train_tokens: {result.train_npz}')
	print(
		'f3_lithology_baseline_token_dataset.validation_tokens: '
		f'{result.validation_npz}',
	)
	print(
		'f3_lithology_baseline_token_dataset.metadata_json: '
		f'{result.metadata_json}',
	)
	print(
		'f3_lithology_baseline_token_dataset.feature_summary_json: '
		f'{result.feature_summary_json}',
	)
	print(
		'f3_lithology_baseline_token_dataset.feature_summary_markdown: '
		f'{result.feature_summary_markdown}',
	)
	if result.split_manifest_json is not None:
		print(
			'f3_lithology_baseline_token_dataset.split_manifest: '
			f'{result.split_manifest_json}',
		)
	if result.class_counts_csv is not None:
		print(
			'f3_lithology_baseline_token_dataset.class_counts_csv: '
			f'{result.class_counts_csv}',
		)
	if result.summary_markdown is not None:
		print(
			'f3_lithology_baseline_token_dataset.summary_markdown: '
			f'{result.summary_markdown}',
		)
	print(
		'f3_lithology_baseline_token_dataset.train_token_count: '
		f'{result.train_token_count}',
	)
	print(
		'f3_lithology_baseline_token_dataset.validation_token_count: '
		f'{result.validation_token_count}',
	)
	print(f'f3_lithology_baseline_token_dataset.feature_dim: {result.feature_dim}')


def _print_summary(config: F3LithologyBaselineTokenDatasetConfig) -> None:
	print(f'stage: {STAGE}')
	print(f'reference_token_dataset.train_tokens: {config.reference.train_tokens}')
	print(
		'reference_token_dataset.validation_tokens: '
		f'{config.reference.validation_tokens}',
	)
	print(f'reference_token_dataset.metadata_json: {config.reference.metadata_json}')
	print(f'reference_token_dataset.split_manifest: {config.reference.split_manifest}')
	print(f'baseline.kind: {config.features.kind}')
	print(f'token_dataset.output_dir: {config.outputs.output_dir}')
	print(f'token_dataset.metadata_json: {config.outputs.metadata_json}')
	print(
		'token_dataset.feature_summary_json: '
		f'{config.outputs.feature_summary_json}',
	)
	print(
		'token_dataset.feature_summary_markdown: '
		f'{config.outputs.feature_summary_markdown}',
	)
	if config.outputs.split_manifest_json is not None:
		print(f'token_dataset.split_manifest: {config.outputs.split_manifest_json}')
	if config.outputs.class_counts_csv is not None:
		print(f'token_dataset.class_counts_csv: {config.outputs.class_counts_csv}')
	if config.outputs.summary_markdown is not None:
		print(f'token_dataset.summary_markdown: {config.outputs.summary_markdown}')
	if config.feature_source is not None:
		print(f'token_dataset.feature_source: {dict(config.feature_source)}')


if __name__ == '__main__':
	main()
