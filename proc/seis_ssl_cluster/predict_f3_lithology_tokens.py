"""Apply a trained F3 lithology probe to the full token embedding volume."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology import (
	f3_lithology_prediction_config_from_mapping,
)
from seis_ssl_cluster.f3 import (
	F3LithologyPredictionConfig,
	predict_f3_lithology_tokens,
)

STAGE = 'predict_f3_lithology_tokens'
DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '50_lithology'
	/ 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	/ 'overlap_x16'
	/ 'png_slices_segy_labels_v1'
	/ '04_predict_volume.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for F3 lithology token prediction."""
	return build_config_parser(
		'Predict F3 lithology classes for all tokens.',
		default_config=DEFAULT_CONFIG,
		dry_run_help=(
			'Validate the config and print a run summary without writing outputs.'
		),
	)


def main() -> None:
	"""Run full F3 token prediction or print a dry-run summary."""
	parser = build_parser()
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=f3_lithology_prediction_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		_print_summary(config)
		print('execution: dry-run; F3 lithology token prediction skipped')
		return

	result = predict_f3_lithology_tokens(config)
	print(f'f3_lithology_prediction.token_predictions: {result.token_predictions}')
	print(f'f3_lithology_prediction.probability_volume: {result.probability_volume}')
	print(f'f3_lithology_prediction.valid_token_grid: {result.valid_token_grid}')
	print(f'f3_lithology_prediction.metadata_json: {result.metadata_json}')
	print(
		'f3_lithology_prediction.validation_slice_metrics_csv: '
		f'{result.validation_slice_metrics_csv}',
	)
	print(f'f3_lithology_prediction.valid_token_count: {result.valid_token_count}')
	print(f'f3_lithology_prediction.invalid_token_count: {result.invalid_token_count}')
	print(
		'f3_lithology_prediction.validation_slice_count: '
		f'{result.validation_slice_count}',
	)


def _print_summary(config: F3LithologyPredictionConfig) -> None:
	print(f'stage: {STAGE}')
	print(f'embeddings.input_dir: {config.inputs.embeddings_dir}')
	print(f'probe.probe_joblib: {config.inputs.probe_joblib}')
	print(f'probe.scaler_joblib: {config.inputs.scaler_joblib}')
	print(f'labels.source_label_volume: {config.inputs.label_volume}')
	print(f'labels.class_info: {config.inputs.class_info}')
	print(f'labels.png_label_inventory: {config.inputs.png_label_inventory}')
	print(f'labels.segy_geometry_json: {config.inputs.segy_geometry_json}')
	print(f'labels.source_label_segy: {config.inputs.source_label_segy}')
	print(f'model.tag: {config.model.get("tag")}')
	print(f'model.freeze_encoder: {config.model.get("freeze_encoder")}')
	print(f'probe.spec: {config.probe.get("spec")}')
	print(f'predictions.batch_size: {config.batch_size}')
	print(f'predictions.output_dir: {config.outputs.output_dir}')
	print(f'predictions.token_predictions: {config.outputs.token_predictions}')
	print(f'predictions.probability_volume: {config.outputs.probability_volume}')
	print(f'predictions.valid_token_grid: {config.outputs.valid_token_grid}')
	print(f'predictions.metadata_json: {config.outputs.metadata_json}')
	print(
		'predictions.validation_slice_metrics_csv: '
		f'{config.outputs.validation_slice_metrics_csv}',
	)
	print(f'classes.count: {len(config.classes)}')


if __name__ == '__main__':
	main()
