"""F3 lithology prediction config validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from seis_ssl_cluster.config.f3_lithology_common import (
	_optional_absolute_path,
	_optional_mapping,
	_optional_positive_int,
	_required_absolute_path,
	_required_fraction,
	_required_mapping,
	_required_nonnegative_int,
	_validate_allowed_keys,
	_validate_artifact_or_f3_source_path,
	_validate_frozen_encoder,
)
from seis_ssl_cluster.f3 import (
	F3LithologyPredictionConfig,
	F3LithologyPredictionInputs,
	F3LithologyPredictionOutputs,
	read_f3_lithology_prediction_classes,
)
from seis_ssl_cluster.f3.lithology.tokens import F3LithologyTokenPolicy

if TYPE_CHECKING:
	from pathlib import Path


def f3_lithology_prediction_config_from_mapping(
	config: Mapping[str, object],
	*,
	load_classes: bool = True,
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
				'token_dataset',
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
	_validate_frozen_encoder(model, stage='F3 lithology prediction')
	embeddings = _required_mapping(config, 'embeddings')
	labels = _required_mapping(config, 'labels')
	lithology = _required_mapping(config, 'lithology')
	token_dataset = _optional_mapping(config, 'token_dataset')
	probe = _required_mapping(config, 'probe')
	predictions = _required_mapping(config, 'predictions')
	lithology_root = _optional_absolute_path(lithology, 'root', prefix='lithology')
	token_dataset_dir = _optional_absolute_path(
		token_dataset,
		'input_dir',
		prefix='token_dataset',
		default=(
			None if lithology_root is None else lithology_root / 'token_dataset'
		),
	)
	validation_tokens = _optional_absolute_path(
		token_dataset,
		'validation_tokens',
		prefix='token_dataset',
		default=(
			None
			if token_dataset_dir is None
			else token_dataset_dir / 'validation_tokens.npz'
		),
	)
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
		validation_tokens=validation_tokens,
	)
	outputs = _prediction_outputs_from_mapping(predictions)
	for label, path in _prediction_paths(inputs, outputs):
		_validate_artifact_or_f3_source_path(
			path,
			label,
			artifact_root=artifact_root,
			f3_root=f3_root,
		)
	return F3LithologyPredictionConfig(
		inputs=inputs,
		outputs=outputs,
		classes=(
			read_f3_lithology_prediction_classes(inputs.class_info)
			if load_classes
			else None
		),
		token_policy=_prediction_token_policy_from_mapping(predictions),
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


def _prediction_token_policy_from_mapping(
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


def _prediction_paths(
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
	if inputs.validation_tokens is not None:
		paths.append(('token_dataset.validation_tokens', inputs.validation_tokens))
	return tuple(paths)


__all__ = ['f3_lithology_prediction_config_from_mapping']
