"""Build F3 token-level lithology datasets from supervised slice locations."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology import (
	f3_lithology_token_dataset_config_from_mapping,
)
from seis_ssl_cluster.f3 import (
	F3LithologyTokenDatasetConfig,
	build_f3_lithology_token_dataset,
)

STAGE = 'build_f3_lithology_token_dataset'
DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '50_lithology'
	/ 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	/ 'overlap_x16'
	/ 'png_slices_segy_labels_v1'
	/ '01_build_token_dataset.yaml'
)


def main() -> None:
	"""Build F3 lithology token datasets or print a dry-run summary."""
	parser = ArgumentParser(description='Build F3 lithology token datasets.')
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
	config = f3_lithology_token_dataset_config_from_mapping(raw_config)
	if args.dry_run:
		_print_summary(config)
		print('execution: dry-run; F3 lithology token dataset build skipped')
		return

	result = build_f3_lithology_token_dataset(config)
	print(f'f3_lithology_token_dataset.train_tokens: {result.train_npz}')
	print(f'f3_lithology_token_dataset.validation_tokens: {result.validation_npz}')
	print(f'f3_lithology_token_dataset.all_labeled_tokens: {result.all_labeled_npz}')
	print(f'f3_lithology_token_dataset.metadata_json: {result.metadata_json}')
	print(f'f3_lithology_token_dataset.class_counts_csv: {result.class_counts_csv}')
	print(f'f3_lithology_token_dataset.summary_markdown: {result.summary_markdown}')
	print(f'f3_lithology_token_dataset.split_manifest: {result.split_manifest_json}')
	print(f'f3_lithology_token_dataset.quicklook_count: {len(result.quicklook_paths)}')
	print(f'f3_lithology_token_dataset.train_token_count: {result.train_token_count}')
	print(
		'f3_lithology_token_dataset.validation_token_count: '
		f'{result.validation_token_count}',
	)


def _print_summary(config: F3LithologyTokenDatasetConfig) -> None:
	print(f'stage: {STAGE}')
	print(f'embeddings.input_dir: {config.inputs.embeddings_dir}')
	print(f'labels.source_label_volume: {config.inputs.label_volume}')
	print(f'labels.source_label_segy: {config.inputs.source_label_segy}')
	print(f'labels.png_label_inventory: {config.inputs.png_label_inventory}')
	print(f'labels.class_info: {config.inputs.class_info}')
	print(f'labels.segy_geometry_json: {config.inputs.segy_geometry_json}')
	print(f'registry.seismic_volume: {config.inputs.seismic_volume}')
	print(f'registry.metadata_json: {config.inputs.volume_metadata_json}')
	print('token_dataset.patch_size_source: embedding metadata')
	print(
		'token_dataset.tokenization.min_labeled_fraction: '
		f'{config.policy.min_labeled_fraction}',
	)
	print(
		'token_dataset.tokenization.min_majority_fraction: '
		f'{config.policy.min_majority_fraction}',
	)
	print(
		'token_dataset.tokenization.ignore_z_border_samples: '
		f'{config.policy.ignore_z_border_samples}',
	)
	print(f'token_dataset.output_dir: {config.outputs.output_dir}')
	print(f'token_dataset.metadata_json: {config.outputs.metadata_json}')
	print(f'token_dataset.class_counts_csv: {config.outputs.class_counts_csv}')
	print(f'token_dataset.summary_markdown: {config.outputs.summary_markdown}')
	print(f'token_dataset.split_manifest: {config.outputs.split_manifest_json}')
	print(f'token_dataset.quicklook_dir: {config.outputs.quicklook_dir}')
	print(f'token_dataset.figure.dpi: {config.figure_dpi}')
	if config.feature_source is not None:
		print(f'token_dataset.feature_source: {dict(config.feature_source)}')
	if config.reference_token_dataset is not None:
		reference = config.reference_token_dataset
		print('token_dataset.split_source: reference_token_dataset')
		print(f'token_dataset.reference.train_tokens: {reference.train_tokens}')
		print(
			'token_dataset.reference.validation_tokens: '
			f'{reference.validation_tokens}',
		)
		print(f'token_dataset.reference.metadata_json: {reference.metadata_json}')
		print(
			'token_dataset.reference.split_manifest: '
			f'{reference.split_manifest_json}',
		)


if __name__ == '__main__':
	main()
