"""Train and evaluate an F3 token-level lithology probe."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology import (
	f3_lithology_probe_config_from_mapping,
)
from seis_ssl_cluster.f3 import (
	F3LithologyProbeConfig,
	train_and_evaluate_f3_lithology_probe,
)

STAGE = 'train_f3_lithology_probe'
DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '50_lithology'
	/ 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	/ 'overlap_x16'
	/ 'png_slices_segy_labels_v1'
	/ '02_train_linear_probe.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for F3 token-level lithology probe training."""
	return build_config_parser(
		'Train an F3 token-level lithology probe.',
		default_config=DEFAULT_CONFIG,
		dry_run_help='Validate the config and print a run summary without training.',
	)


def main() -> None:
	"""Train an F3 lithology probe or print a dry-run summary."""
	parser = build_parser()
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=f3_lithology_probe_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		_print_summary(config)
		print('execution: dry-run; F3 lithology probe training skipped')
		return

	result = train_and_evaluate_f3_lithology_probe(config)
	print(f'f3_lithology_probe.probe_joblib: {result.probe_joblib}')
	print(f'f3_lithology_probe.scaler_joblib: {result.scaler_joblib}')
	print(f'f3_lithology_probe.config_json: {result.config_json}')
	print(f'f3_lithology_probe.metrics_json: {result.metrics_json}')
	print(f'f3_lithology_probe.metrics_csv: {result.metrics_csv}')
	print(f'f3_lithology_probe.confusion_matrix_csv: {result.confusion_matrix_csv}')
	print(
		'f3_lithology_probe.classification_report_md: '
		f'{result.classification_report_md}',
	)
	print(f'f3_lithology_probe.confusion_matrix_png: {result.confusion_matrix_png}')
	print(f'f3_lithology_probe.per_class_f1_png: {result.per_class_f1_png}')
	print(f'f3_lithology_probe.train_token_count: {result.train_token_count}')
	print(
		f'f3_lithology_probe.validation_token_count: {result.validation_token_count}',
	)


def _print_summary(config: F3LithologyProbeConfig) -> None:
	print(f'stage: {STAGE}')
	print(f'token_dataset.train_tokens: {config.inputs.train_tokens}')
	print(f'token_dataset.validation_tokens: {config.inputs.validation_tokens}')
	print(f'token_dataset.metadata_json: {config.inputs.token_dataset_metadata_json}')
	feature_source = config.token_dataset.get('feature_source')
	if isinstance(feature_source, Mapping):
		print(f'token_dataset.feature_source: {dict(feature_source)}')
	print(f'labels.class_info: {config.inputs.class_info}')
	print(f'model.tag: {config.model.get("tag")}')
	print(f'model.checkpoint: {config.model.get("checkpoint")}')
	print(f'model.freeze_encoder: {config.model.get("freeze_encoder")}')
	print(f'probe.spec: {config.probe.spec}')
	print(f'probe.type: {config.probe.probe_type}')
	print(f'probe.feature_scaling: {config.probe.feature_scaling}')
	print(f'probe.class_weight: {config.probe.class_weight}')
	print(f'probe.random_state: {config.probe.random_state}')
	if config.probe.probe_type == 'logistic_regression':
		print(f'probe.max_iter: {config.probe.max_iter}')
	else:
		print(f'probe.hidden_dims: {list(config.probe.hidden_dims)}')
		print(f'probe.dropout: {config.probe.dropout}')
		print(f'probe.max_epochs: {config.probe.max_epochs}')
		print(
			f'probe.early_stopping_patience: {config.probe.early_stopping_patience}',
		)
		print(f'probe.batch_size: {config.probe.batch_size}')
		print(f'probe.learning_rate: {config.probe.learning_rate}')
		print(f'probe.weight_decay: {config.probe.weight_decay}')
	print(f'evaluation.metrics: {list(config.evaluation_metrics)}')
	print(f'evaluation.figure.dpi: {config.figure_dpi}')
	print(f'probe.output_dir: {config.outputs.output_dir}')
	print(f'probe.probe_joblib: {config.outputs.probe_joblib}')
	print(f'probe.scaler_joblib: {config.outputs.scaler_joblib}')
	print(f'probe.probe_config_resolved_json: {config.outputs.config_json}')
	print(f'probe.metrics_json: {config.outputs.metrics_json}')
	print(f'probe.metrics_csv: {config.outputs.metrics_csv}')
	print(f'probe.confusion_matrix_csv: {config.outputs.confusion_matrix_csv}')
	print(f'probe.classification_report_md: {config.outputs.classification_report_md}')
	print(f'probe.confusion_matrix_png: {config.outputs.confusion_matrix_png}')
	print(f'probe.per_class_f1_png: {config.outputs.per_class_f1_png}')
	print(f'classes.count: {len(config.classes)}')


if __name__ == '__main__':
	main()
