"""F3 lithology probe config validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from seis_ssl_cluster.config.f3_lithology_common import (
	_hidden_dims,
	_optional_absolute_path,
	_optional_fraction,
	_optional_int,
	_optional_mapping,
	_optional_nonnegative_float,
	_optional_nullable_str,
	_optional_positive_float,
	_optional_positive_int,
	_optional_str,
	_required_absolute_path,
	_required_mapping,
	_required_str,
	_string_item,
	_validate_allowed_keys,
	_validate_frozen_encoder,
	_validate_output_not_under_f3_root,
)
from seis_ssl_cluster.f3 import (
	DEFAULT_EVALUATION_METRICS,
	F3LithologyProbeConfig,
	F3LithologyProbeInputs,
	F3LithologyProbeOutputs,
	F3LithologyProbeSettings,
)
from seis_ssl_cluster.f3.lithology.tokens import read_f3_lithology_class_info


def f3_lithology_probe_config_from_mapping(
	config: Mapping[str, object],
	*,
	load_classes: bool = True,
) -> F3LithologyProbeConfig:
	"""Validate and normalize the F3 lithology probe config."""
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
				'evaluation',
			},
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	_required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	dataset = _required_mapping(config, 'dataset')
	model = _required_mapping(config, 'model')
	_validate_frozen_encoder(model, stage='F3 lithology probe training')
	embeddings = _required_mapping(config, 'embeddings')
	labels = _required_mapping(config, 'labels')
	lithology = _required_mapping(config, 'lithology')
	token_dataset = _required_mapping(config, 'token_dataset')
	probe = _required_mapping(config, 'probe')
	evaluation = _optional_mapping(config, 'evaluation')
	token_dataset_dir = _required_absolute_path(
		token_dataset,
		'input_dir',
		prefix='token_dataset',
	)
	class_info_path = _required_absolute_path(labels, 'class_info', prefix='labels')
	outputs = F3LithologyProbeOutputs(
		output_dir=_required_absolute_path(probe, 'output_dir', prefix='probe'),
	)
	token_dataset_metadata_json = _optional_absolute_path(
		token_dataset,
		'metadata_json',
		prefix='token_dataset',
	)
	for label, path in (
		('token_dataset.input_dir', token_dataset_dir),
		('labels.class_info', class_info_path),
		('probe.output_dir', outputs.output_dir),
		('probe.probe_joblib', outputs.probe_joblib),
		('probe.scaler_joblib', outputs.scaler_joblib),
		('probe.probe_config_resolved_json', outputs.config_json),
		('probe.metrics_json', outputs.metrics_json),
		('probe.metrics_csv', outputs.metrics_csv),
		('probe.confusion_matrix_csv', outputs.confusion_matrix_csv),
		('probe.classification_report_md', outputs.classification_report_md),
		('probe.confusion_matrix_png', outputs.confusion_matrix_png),
		('probe.per_class_f1_png', outputs.per_class_f1_png),
	):
		_validate_output_not_under_f3_root(
			path,
			label,
			f3_root=f3_root,
		)
	if token_dataset_metadata_json is not None:
		_validate_output_not_under_f3_root(
			token_dataset_metadata_json,
			'token_dataset.metadata_json',
			f3_root=f3_root,
		)
	return F3LithologyProbeConfig(
		inputs=F3LithologyProbeInputs(
			train_tokens=token_dataset_dir / 'train_tokens.npz',
			validation_tokens=token_dataset_dir / 'validation_tokens.npz',
			class_info=class_info_path,
			token_dataset_metadata_json=token_dataset_metadata_json,
		),
		outputs=outputs,
		classes=(
			read_f3_lithology_class_info(class_info_path) if load_classes else None
		),
		probe=_probe_settings_from_mapping(probe),
		dataset=dataset,
		model=model,
		embeddings=embeddings,
		labels=labels,
		token_dataset=token_dataset,
		lithology=lithology,
		evaluation_metrics=_evaluation_metrics(evaluation),
		figure_dpi=_probe_figure_dpi(evaluation),
	)


def _probe_settings_from_mapping(
	probe: Mapping[str, object],
) -> F3LithologyProbeSettings:
	_validate_allowed_keys(
		probe,
		frozenset(
			{
				'spec',
				'type',
				'feature_scaling',
				'class_weight',
				'max_iter',
				'random_state',
				'hidden_dims',
				'dropout',
				'max_epochs',
				'early_stopping_patience',
				'batch_size',
				'learning_rate',
				'weight_decay',
				'output_dir',
			},
		),
		prefix='probe',
	)
	return F3LithologyProbeSettings(
		spec=_required_str(probe, 'spec', prefix='probe'),
		probe_type=_required_str(probe, 'type', prefix='probe'),
		feature_scaling=_optional_str(
			probe,
			'feature_scaling',
			default='standard',
			prefix='probe',
		),
		class_weight=_optional_nullable_str(
			probe,
			'class_weight',
			default='balanced',
			prefix='probe',
		),
		max_iter=_optional_positive_int(probe.get('max_iter', 2000), 'probe.max_iter'),
		hidden_dims=_hidden_dims(probe.get('hidden_dims', (256, 128))),
		dropout=_optional_fraction(probe.get('dropout', 0.2), 'probe.dropout'),
		max_epochs=_optional_positive_int(
			probe.get('max_epochs', 200),
			'probe.max_epochs',
		),
		early_stopping_patience=_optional_positive_int(
			probe.get('early_stopping_patience', 20),
			'probe.early_stopping_patience',
		),
		batch_size=_optional_positive_int(
			probe.get('batch_size', 1024),
			'probe.batch_size',
		),
		learning_rate=_optional_positive_float(
			probe.get('learning_rate', 1.0e-3),
			'probe.learning_rate',
		),
		weight_decay=_optional_nonnegative_float(
			probe.get('weight_decay', 0.0),
			'probe.weight_decay',
		),
		random_state=_optional_int(
			probe.get('random_state', 42),
			'probe.random_state',
		),
	)


def _evaluation_metrics(evaluation: Mapping[str, object]) -> tuple[str, ...]:
	value = evaluation.get('metrics', DEFAULT_EVALUATION_METRICS)
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'evaluation.metrics must be a list of metric names; got {value!r}'
		raise TypeError(msg)
	metrics = tuple(_string_item(item, 'evaluation.metrics') for item in value)
	if not metrics:
		msg = 'evaluation.metrics must contain at least one metric name'
		raise ValueError(msg)
	return metrics


def _probe_figure_dpi(evaluation: Mapping[str, object]) -> int:
	figure = evaluation.get('figure')
	if figure is None:
		return 300
	if not isinstance(figure, Mapping):
		msg = f'evaluation.figure must be a mapping; got {figure!r}'
		raise TypeError(msg)
	return _optional_positive_int(figure.get('dpi', 300), 'evaluation.figure.dpi')


__all__ = ['f3_lithology_probe_config_from_mapping']
