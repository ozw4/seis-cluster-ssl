"""Apply a trained F3 lithology probe to the full token embedding volume."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3 import (
	F3LithologyPredictionConfig,
	F3LithologyPredictionInputs,
	F3LithologyPredictionOutputs,
	F3LithologyTokenPolicy,
	predict_f3_lithology_tokens,
	read_f3_lithology_prediction_classes,
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


def main() -> None:
	"""Run full F3 token prediction or print a dry-run summary."""
	parser = ArgumentParser(description='Predict F3 lithology classes for all tokens.')
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
	config = f3_lithology_prediction_config_from_mapping(raw_config)
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


def f3_lithology_prediction_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyPredictionConfig:
	"""Validate and normalize the F3 lithology prediction config."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'model',
				'embeddings',
				'labels',
				'lithology',
				'probe',
				'predictions',
			},
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	dataset = _required_mapping(config, 'dataset')
	model = _required_mapping(config, 'model')
	_validate_frozen_encoder(model)
	embeddings = _required_mapping(config, 'embeddings')
	labels = _required_mapping(config, 'labels')
	lithology = _required_mapping(config, 'lithology')
	probe = _required_mapping(config, 'probe')
	predictions = _required_mapping(config, 'predictions')
	inputs = F3LithologyPredictionInputs(
		embeddings_dir=_required_absolute_path(
			embeddings,
			'input_dir',
			prefix='embeddings',
		),
		probe_joblib=_required_absolute_path(
			probe,
			'probe_joblib',
			prefix='probe',
		),
		scaler_joblib=_required_absolute_path(
			probe,
			'scaler_joblib',
			prefix='probe',
		),
		label_volume=_required_absolute_path(
			labels,
			'source_label_volume',
			prefix='labels',
		),
		class_info=_required_absolute_path(labels, 'class_info', prefix='labels'),
		png_label_inventory=_required_absolute_path(
			labels,
			'png_label_inventory',
			prefix='labels',
		),
		segy_geometry_json=_required_absolute_path(
			labels,
			'segy_geometry_json',
			prefix='labels',
		),
		source_label_segy=_optional_absolute_path(
			labels,
			'source_label_segy',
			prefix='labels',
		),
	)
	outputs = _prediction_outputs_from_mapping(predictions)
	for label, path in _all_paths(inputs, outputs):
		_validate_path(path, label, artifact_root=artifact_root, f3_root=f3_root)
	return F3LithologyPredictionConfig(
		inputs=inputs,
		outputs=outputs,
		classes=read_f3_lithology_prediction_classes(inputs.class_info),
		token_policy=_token_policy_from_mapping(predictions),
		dataset=dataset,
		model=model,
		embeddings=embeddings,
		labels=labels,
		lithology=lithology,
		probe=probe,
		batch_size=_optional_positive_int(
			predictions.get('batch_size', 4096),
			'predictions.batch_size',
		),
	)


def _prediction_outputs_from_mapping(
	predictions: Mapping[str, object],
) -> F3LithologyPredictionOutputs:
	output_dir = _required_absolute_path(
		predictions,
		'output_dir',
		prefix='predictions',
	)
	return F3LithologyPredictionOutputs(
		output_dir=output_dir,
		token_predictions=_required_absolute_path(
			predictions,
			'token_predictions',
			prefix='predictions',
		),
		probability_volume=_required_absolute_path(
			predictions,
			'probability_volume',
			prefix='predictions',
		),
		valid_token_grid=_required_absolute_path(
			predictions,
			'valid_token_grid',
			prefix='predictions',
		),
		metadata_json=_required_absolute_path(
			predictions,
			'metadata_json',
			prefix='predictions',
		),
		validation_slice_metrics_csv=_required_absolute_path(
			predictions,
			'validation_slice_metrics_csv',
			prefix='predictions',
		),
	)


def _token_policy_from_mapping(
	predictions: Mapping[str, object],
) -> F3LithologyTokenPolicy:
	tokenization = predictions.get('tokenization')
	if tokenization is None:
		return F3LithologyTokenPolicy()
	if not isinstance(tokenization, Mapping):
		msg = f'predictions.tokenization must be a mapping; got {tokenization!r}'
		raise TypeError(msg)
	return F3LithologyTokenPolicy(
		min_labeled_fraction=_required_fraction(
			tokenization,
			'min_labeled_fraction',
			prefix='predictions.tokenization',
		),
		min_majority_fraction=_required_fraction(
			tokenization,
			'min_majority_fraction',
			prefix='predictions.tokenization',
		),
		ignore_z_border_samples=_required_nonnegative_int(
			tokenization,
			'ignore_z_border_samples',
			prefix='predictions.tokenization',
		),
	)


def _all_paths(
	inputs: F3LithologyPredictionInputs,
	outputs: F3LithologyPredictionOutputs,
) -> tuple[tuple[str, Path], ...]:
	paths: list[tuple[str, Path]] = [
		('embeddings.input_dir', inputs.embeddings_dir),
		('probe.probe_joblib', inputs.probe_joblib),
		('probe.scaler_joblib', inputs.scaler_joblib),
		('labels.source_label_volume', inputs.label_volume),
		('labels.class_info', inputs.class_info),
		('labels.png_label_inventory', inputs.png_label_inventory),
		('labels.segy_geometry_json', inputs.segy_geometry_json),
		('predictions.output_dir', outputs.output_dir),
		('predictions.token_predictions', outputs.token_predictions),
		('predictions.probability_volume', outputs.probability_volume),
		('predictions.valid_token_grid', outputs.valid_token_grid),
		('predictions.metadata_json', outputs.metadata_json),
		(
			'predictions.validation_slice_metrics_csv',
			outputs.validation_slice_metrics_csv,
		),
	]
	if inputs.source_label_segy is not None:
		paths.append(('labels.source_label_segy', inputs.source_label_segy))
	return tuple(paths)


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


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, Any]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_allowed_keys(
	parent: Mapping[str, object],
	allowed: frozenset[str],
	*,
	prefix: str,
) -> None:
	unexpected = sorted(set(parent) - allowed)
	if unexpected:
		msg = (
			f'{prefix} key(s) not allowed: {unexpected!r}; '
			f'allowed keys are {sorted(allowed)!r}'
		)
		raise ValueError(msg)


def _required_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	path = Path(_required_str(parent, key, prefix=prefix))
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _optional_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path | None:
	value = parent.get(key)
	if value is None:
		return None
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	path = Path(value)
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _required_str(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_positive_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{label} must be a positive integer; got {value!r}'
		raise ValueError(msg)
	return value


def _required_fraction(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> float:
	value = parent.get(key)
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'{prefix}.{key} must be a number in [0, 1]; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not 0.0 <= fraction <= 1.0:
		msg = f'{prefix}.{key} must be in [0, 1]; got {value!r}'
		raise ValueError(msg)
	return fraction


def _required_nonnegative_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> int:
	value = parent.get(key)
	if not isinstance(value, int) or isinstance(value, bool) or value < 0:
		msg = f'{prefix}.{key} must be a nonnegative integer; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_frozen_encoder(model: Mapping[str, object]) -> None:
	if model.get('freeze_encoder') is not True:
		msg = 'model.freeze_encoder must be true for F3 lithology prediction'
		raise ValueError(msg)


def _validate_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
	f3_root: Path,
) -> None:
	if 'runs' in path.parts:
		msg = f'{label} must not use runs/ paths; got {path}'
		raise ValueError(msg)
	if label == 'labels.source_label_segy':
		if not _is_relative_to(path, f3_root):
			msg = f'{label} must be under paths.f3_root ({f3_root}); got {path}'
			raise ValueError(msg)
		return
	if not _is_relative_to(path, artifact_root):
		msg = f'{label} must be under paths.artifact_root ({artifact_root}); got {path}'
		raise ValueError(msg)


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


if __name__ == '__main__':
	main()
