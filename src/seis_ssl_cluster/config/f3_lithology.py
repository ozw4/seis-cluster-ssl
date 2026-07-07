"""F3 lithology config validation entrypoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from seis_ssl_cluster.config import f3_lithology_common as _f3_lithology_common
from seis_ssl_cluster.config.f3_lithology_common import (
	_hidden_dims,
	_int_tuple,
	_max_file_size_bytes,
	_optional_absolute_path,
	_optional_bool_value,
	_optional_fraction,
	_optional_int,
	_optional_mapping,
	_optional_non_negative_int,
	_optional_nonnegative_float,
	_optional_nullable_str,
	_optional_path,
	_optional_positive_float,
	_optional_positive_int,
	_optional_str,
	_percentiles,
	_publish_optional_bool,
	_required_absolute_path,
	_required_fraction,
	_required_mapping,
	_required_nonnegative_int,
	_required_str,
	_string_item,
	_validate_allowed_keys,
	_validate_artifact_or_f3_source_path,
	_validate_artifact_path_not_f3,
	_validate_frozen_encoder,
)
from seis_ssl_cluster.f3 import (
	DEFAULT_EVALUATION_METRICS,
	F3LithologyComparisonReportConfig,
	F3LithologyPredictionConfig,
	F3LithologyPredictionInputs,
	F3LithologyPredictionOutputs,
	F3LithologyProbeConfig,
	F3LithologyProbeInputs,
	F3LithologyProbeOutputs,
	F3LithologyProbeSettings,
	F3LithologyPublishConfig,
	F3LithologyReportConfig,
	F3LithologyTokenDatasetConfig,
	F3LithologyTokenDatasetInputs,
	F3LithologyTokenDatasetOutputs,
	F3LithologyTokenPolicy,
	F3LithologyVisualizationConfig,
	F3LithologyVisualizationFigureConfig,
	F3LithologyVisualizationInputs,
	F3LithologyVisualizationOutputs,
	F3ReferenceTokenDataset,
	read_f3_lithology_class_info,
	read_f3_lithology_prediction_classes,
	read_f3_lithology_visualization_classes,
)
from seis_ssl_cluster.f3.prepare_volume import f3_prepare_volume_config_from_mapping
from seis_ssl_cluster.paths import ArtifactPaths, ExperimentKey

if TYPE_CHECKING:
	from pathlib import Path

_is_relative_to = _f3_lithology_common._is_relative_to  # noqa: SLF001


def f3_lithology_token_dataset_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyTokenDatasetConfig:
	"""Validate and normalize the F3 lithology token dataset config."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'model',
				'embeddings',
				'labels',
				'registry',
				'lithology',
				'token_dataset',
				'feature_source',
			},
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	dataset = _required_mapping(config, 'dataset')
	model = _required_mapping(config, 'model')
	embeddings = _required_mapping(config, 'embeddings')
	labels = _required_mapping(config, 'labels')
	registry = _required_mapping(config, 'registry')
	token_dataset = _required_mapping(config, 'token_dataset')
	outputs = _token_dataset_outputs_from_mapping(token_dataset)
	for label, path in _token_dataset_output_paths(outputs):
		_validate_artifact_path_not_f3(
			path,
			label,
			artifact_root=artifact_root,
			f3_root=f3_root,
		)
	inputs = F3LithologyTokenDatasetInputs(
		embeddings_dir=_required_absolute_path(
			embeddings,
			'input_dir',
			prefix='embeddings',
		),
		label_volume=_required_absolute_path(
			labels,
			'source_label_volume',
			prefix='labels',
		),
		seismic_volume=_required_absolute_path(
			registry,
			'seismic_volume',
			prefix='registry',
		),
		png_label_inventory=_required_absolute_path(
			labels,
			'png_label_inventory',
			prefix='labels',
		),
		class_info=_required_absolute_path(labels, 'class_info', prefix='labels'),
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
		volume_metadata_json=_optional_absolute_path(
			registry,
			'metadata_json',
			prefix='registry',
		),
	)
	policy = _token_dataset_policy_from_mapping(
		_required_mapping(token_dataset, 'tokenization'),
	)
	reference_token_dataset = _reference_token_dataset_from_mapping(token_dataset)
	feature_source = _feature_source(
		config,
		token_dataset,
		reference_token_dataset=reference_token_dataset,
	)
	return F3LithologyTokenDatasetConfig(
		inputs=inputs,
		outputs=outputs,
		policy=policy,
		dataset=dataset,
		model=model,
		figure_dpi=_token_dataset_figure_dpi(token_dataset),
		feature_source=feature_source,
		reference_token_dataset=reference_token_dataset,
	)


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
	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
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
	for label, path in _probe_artifact_paths(
		token_dataset_dir=token_dataset_dir,
		class_info_path=class_info_path,
		token_dataset_metadata_json=token_dataset_metadata_json,
		outputs=outputs,
	):
		_validate_artifact_path_not_f3(
			path,
			label,
			artifact_root=artifact_root,
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


def f3_lithology_visualization_config_from_mapping(
	config: Mapping[str, object],
	*,
	load_classes: bool = True,
) -> F3LithologyVisualizationConfig:
	"""Validate and normalize the F3 lithology visualization config."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'model',
				'labels',
				'registry',
				'lithology',
				'probe',
				'predictions',
				'visualizations',
			},
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	dataset = _required_mapping(config, 'dataset')
	model = _required_mapping(config, 'model')
	_validate_frozen_encoder(model, stage='F3 lithology visualization')
	labels = _required_mapping(config, 'labels')
	registry = _required_mapping(config, 'registry')
	lithology = _required_mapping(config, 'lithology')
	probe = _required_mapping(config, 'probe')
	predictions = _required_mapping(config, 'predictions')
	visualizations = _required_mapping(config, 'visualizations')
	inputs = F3LithologyVisualizationInputs(
		seismic_volume=_required_absolute_path(
			registry,
			'seismic_volume',
			prefix='registry',
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
		prediction_metadata_json=_required_absolute_path(
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
	outputs = F3LithologyVisualizationOutputs(
		output_dir=_required_absolute_path(
			visualizations,
			'output_dir',
			prefix='visualizations',
		),
		metadata_json=_required_absolute_path(
			visualizations,
			'metadata_json',
			prefix='visualizations',
		),
		selected_slices_dir=_required_absolute_path(
			visualizations,
			'selected_slices_dir',
			prefix='visualizations',
		),
	)
	for label, path in _visualization_paths(inputs, outputs, labels):
		_validate_artifact_or_f3_source_path(
			path,
			label,
			artifact_root=artifact_root,
			f3_root=f3_root,
		)
	return F3LithologyVisualizationConfig(
		inputs=inputs,
		outputs=outputs,
		classes=(
			read_f3_lithology_visualization_classes(inputs.class_info)
			if load_classes
			else None
		),
		dataset=dataset,
		model=model,
		labels=labels,
		lithology=lithology,
		probe=probe,
		predictions=predictions,
		selected_slices=_selected_slices_from_mapping(
			_required_mapping(visualizations, 'slices'),
		),
		figure=_figure_config_from_mapping(
			_optional_mapping(visualizations, 'figure'),
		),
	)


def f3_lithology_report_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyReportConfig:
	"""Validate and normalize the F3 lithology report config."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'model',
				'labels',
				'lithology',
				'probe',
				'predictions',
				'visualizations',
				'reports',
				'comparison',
				'publish',
			},
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	dataset = _required_mapping(config, 'dataset')
	model = _required_mapping(config, 'model')
	labels = _required_mapping(config, 'labels')
	lithology = _required_mapping(config, 'lithology')
	probe = _required_mapping(config, 'probe')
	predictions = _optional_mapping(config, 'predictions')
	visualizations = _optional_mapping(config, 'visualizations')
	reports = _required_mapping(config, 'reports')
	metrics_json = _required_absolute_path(probe, 'metrics_json', prefix='probe')
	probe_config_json = _optional_absolute_path(
		probe,
		'probe_config_resolved_json',
		prefix='probe',
	)
	output_dir = _required_absolute_path(reports, 'output_dir', prefix='reports')
	output_markdown = _required_absolute_path(
		reports,
		'output_markdown',
		prefix='reports',
	)
	output_json = _required_absolute_path(reports, 'output_json', prefix='reports')
	prediction_metadata_json = _optional_absolute_path(
		predictions,
		'metadata_json',
		prefix='predictions',
	)
	visualization_metadata_json = _optional_absolute_path(
		visualizations,
		'metadata_json',
		prefix='visualizations',
	)
	token_dataset_metadata_json = _optional_absolute_path(
		reports,
		'token_dataset_metadata_json',
		prefix='reports',
	)
	comparison = _embedded_comparison_config(
		_optional_mapping(config, 'comparison'),
		artifact_root=artifact_root,
		dataset=dataset,
	)
	for label, path in _report_paths(
		metrics_json=metrics_json,
		probe_config_json=probe_config_json,
		token_dataset_metadata_json=token_dataset_metadata_json,
		prediction_metadata_json=prediction_metadata_json,
		visualization_metadata_json=visualization_metadata_json,
		output_dir=output_dir,
		output_markdown=output_markdown,
		output_json=output_json,
		comparison=comparison,
	):
		_validate_artifact_path_not_f3(
			path,
			label,
			artifact_root=artifact_root,
			f3_root=f3_root,
		)
	return F3LithologyReportConfig(
		output_dir=output_dir,
		output_markdown=output_markdown,
		output_json=output_json,
		metrics_json=metrics_json,
		probe_config_json=probe_config_json,
		token_dataset_metadata_json=token_dataset_metadata_json,
		prediction_metadata_json=prediction_metadata_json,
		visualization_metadata_json=visualization_metadata_json,
		dataset=dataset,
		model=model,
		labels=labels,
		lithology=lithology,
		probe=probe,
		comparison=comparison,
	)


def f3_lithology_publish_config_from_mapping(
	value: object,
) -> F3LithologyPublishConfig:
	"""Validate and normalize the optional F3 lithology publish config."""
	if value is None:
		return F3LithologyPublishConfig()
	if not isinstance(value, Mapping):
		msg = f'publish must be a mapping; got {value!r}'
		raise TypeError(msg)
	enabled = _publish_optional_bool(value, 'enabled', default=False)
	include_figures = _publish_optional_bool(value, 'include_figures', default=True)
	output_dir = _optional_path(value, 'output_dir')
	if enabled and output_dir is None:
		msg = 'publish.output_dir must be set when publish.enabled is true'
		raise ValueError(msg)
	return F3LithologyPublishConfig(
		enabled=enabled,
		output_dir=output_dir,
		include_figures=include_figures,
		max_file_size_bytes=_max_file_size_bytes(value),
		max_prediction_figures=_optional_non_negative_int(
			value,
			'max_prediction_figures',
			default=3,
		),
	)


def _token_dataset_outputs_from_mapping(
	token_dataset: Mapping[str, object],
) -> F3LithologyTokenDatasetOutputs:
	return F3LithologyTokenDatasetOutputs(
		output_dir=_required_absolute_path(
			token_dataset,
			'output_dir',
			prefix='token_dataset',
		),
		metadata_json=_required_absolute_path(
			token_dataset,
			'metadata_json',
			prefix='token_dataset',
		),
		class_counts_csv=_required_absolute_path(
			token_dataset,
			'class_counts_csv',
			prefix='token_dataset',
		),
		summary_markdown=_required_absolute_path(
			token_dataset,
			'summary_markdown',
			prefix='token_dataset',
		),
		split_manifest_json=_required_absolute_path(
			token_dataset,
			'split_manifest',
			prefix='token_dataset',
		),
		quicklook_dir=_required_absolute_path(
			token_dataset,
			'quicklook_dir',
			prefix='token_dataset',
		),
	)


def _token_dataset_policy_from_mapping(
	policy: Mapping[str, object],
) -> F3LithologyTokenPolicy:
	for key in ('patch_size', 'patch_size_xyz'):
		if key in policy:
			msg = (
				'token_dataset.tokenization must not override patch size; '
				'patch size is read from embedding metadata'
			)
			raise ValueError(msg)
	return F3LithologyTokenPolicy(
		min_labeled_fraction=_required_fraction(
			policy,
			'min_labeled_fraction',
			prefix='token_dataset.tokenization',
		),
		min_majority_fraction=_required_fraction(
			policy,
			'min_majority_fraction',
			prefix='token_dataset.tokenization',
		),
		ignore_z_border_samples=_required_nonnegative_int(
			policy,
			'ignore_z_border_samples',
			prefix='token_dataset.tokenization',
		),
	)


def _token_dataset_figure_dpi(token_dataset: Mapping[str, object]) -> int:
	figure = token_dataset.get('figure')
	if figure is None:
		return 300
	if not isinstance(figure, Mapping):
		msg = f'token_dataset.figure must be a mapping; got {figure!r}'
		raise TypeError(msg)
	return _optional_positive_int(
		figure.get('dpi', 300),
		'token_dataset.figure.dpi',
	)


def _feature_source(
	config: Mapping[str, object],
	token_dataset: Mapping[str, object],
	*,
	reference_token_dataset: F3ReferenceTokenDataset | None,
) -> Mapping[str, object] | None:
	top_level = config.get('feature_source')
	nested = token_dataset.get('feature_source')
	if top_level is not None and nested is not None and top_level != nested:
		msg = 'config.feature_source and token_dataset.feature_source must match'
		raise ValueError(msg)
	value = top_level if top_level is not None else nested
	if value is None:
		return None
	if not isinstance(value, Mapping):
		msg = f'feature_source must be a mapping; got {value!r}'
		raise TypeError(msg)
	feature_source = {
		'kind': _feature_source_str(value, 'kind'),
		'reference_model_tag': _feature_source_str(value, 'reference_model_tag'),
		'embedding_spec': _feature_source_str(value, 'embedding_spec'),
		'description': _feature_source_str(value, 'description'),
	}
	if (
		reference_token_dataset is None
		and feature_source['kind'] != 'pretrained_encoder'
	):
		msg = (
			'feature_source.kind must be "pretrained_encoder" for pretrained '
			f'token datasets; got {feature_source["kind"]!r}'
		)
		raise ValueError(msg)
	return feature_source


def _feature_source_str(value: Mapping[str, object], key: str) -> str:
	item = value.get(key)
	if not isinstance(item, str) or not item:
		msg = f'feature_source.{key} must be a non-empty string; got {item!r}'
		raise TypeError(msg)
	return item


def _reference_token_dataset_from_mapping(
	token_dataset: Mapping[str, object],
) -> F3ReferenceTokenDataset | None:
	value = token_dataset.get('reference_token_dataset')
	if value is None:
		return None
	if not isinstance(value, Mapping):
		msg = f'token_dataset.reference_token_dataset must be a mapping; got {value!r}'
		raise TypeError(msg)
	root = _optional_absolute_path(
		value,
		'root',
		prefix='token_dataset.reference_token_dataset',
	)
	train_tokens = _optional_absolute_path(
		value,
		'train_tokens',
		prefix='token_dataset.reference_token_dataset',
	)
	validation_tokens = _optional_absolute_path(
		value,
		'validation_tokens',
		prefix='token_dataset.reference_token_dataset',
	)
	metadata_json = _optional_absolute_path(
		value,
		'metadata_json',
		prefix='token_dataset.reference_token_dataset',
	)
	split_manifest_json = _optional_absolute_path(
		value,
		'split_manifest_json',
		prefix='token_dataset.reference_token_dataset',
	)
	split_manifest = _optional_absolute_path(
		value,
		'split_manifest',
		prefix='token_dataset.reference_token_dataset',
	)
	if root is not None:
		train_tokens = train_tokens or root / 'train_tokens.npz'
		validation_tokens = validation_tokens or root / 'validation_tokens.npz'
		metadata_json = metadata_json or root / 'token_dataset_metadata.json'
		split_manifest_json = (
			split_manifest_json or split_manifest or root / 'splits.json'
		)
	else:
		split_manifest_json = split_manifest_json or split_manifest
	if train_tokens is None or validation_tokens is None or metadata_json is None:
		msg = (
			'token_dataset.reference_token_dataset requires root or explicit '
			'train_tokens, validation_tokens, and metadata_json paths'
		)
		raise KeyError(msg)
	return F3ReferenceTokenDataset(
		train_tokens=train_tokens,
		validation_tokens=validation_tokens,
		metadata_json=metadata_json,
		split_manifest_json=split_manifest_json,
		root=root,
	)


def _token_dataset_output_paths(
	outputs: F3LithologyTokenDatasetOutputs,
) -> tuple[tuple[str, Path], ...]:
	return (
		('token_dataset.output_dir', outputs.output_dir),
		('token_dataset.metadata_json', outputs.metadata_json),
		('token_dataset.class_counts_csv', outputs.class_counts_csv),
		('token_dataset.summary_markdown', outputs.summary_markdown),
		('token_dataset.split_manifest', outputs.split_manifest_json),
		('token_dataset.quicklook_dir', outputs.quicklook_dir),
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


def _probe_artifact_paths(
	*,
	token_dataset_dir: Path,
	class_info_path: Path,
	token_dataset_metadata_json: Path | None,
	outputs: F3LithologyProbeOutputs,
) -> tuple[tuple[str, Path], ...]:
	paths: list[tuple[str, Path]] = [
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
	]
	if token_dataset_metadata_json is not None:
		paths.append(('token_dataset.metadata_json', token_dataset_metadata_json))
	return tuple(paths)


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


def _selected_slices_from_mapping(
	slices: Mapping[str, object],
) -> dict[str, tuple[int, ...]]:
	_validate_allowed_keys(
		slices,
		frozenset({'inline', 'crossline', 'z'}),
		prefix='visualizations.slices',
	)
	return {
		key: _int_tuple(slices.get(key, ()), f'visualizations.slices.{key}')
		for key in ('inline', 'crossline', 'z')
	}


def _figure_config_from_mapping(
	figure: Mapping[str, object],
) -> F3LithologyVisualizationFigureConfig:
	output_formats = figure.get('output_formats', ['png'])
	if output_formats != ['png']:
		msg = 'visualizations.figure.output_formats must be ["png"]'
		raise ValueError(msg)
	return F3LithologyVisualizationFigureConfig(
		dpi=_optional_positive_int(
			figure.get('dpi', 300),
			'visualizations.figure.dpi',
		),
		background=_optional_str(
			figure,
			'background',
			default='white',
			prefix='visualizations.figure',
		),
		z_axis=_optional_str(
			figure,
			'z_axis',
			default='down',
			prefix='visualizations.figure',
		),
		include_legend=_optional_bool_value(
			figure.get('include_legend', True),
			'visualizations.figure.include_legend',
		),
		include_confidence=_optional_bool_value(
			figure.get('include_confidence', False),
			'visualizations.figure.include_confidence',
		),
		amplitude_clip_percentiles=_percentiles(
			figure.get('amplitude_clip_percentiles', (1.0, 99.0)),
		),
	)


def _visualization_paths(
	inputs: F3LithologyVisualizationInputs,
	outputs: F3LithologyVisualizationOutputs,
	labels: Mapping[str, object],
) -> tuple[tuple[str, Path], ...]:
	paths: list[tuple[str, Path]] = [
		('registry.seismic_volume', inputs.seismic_volume),
		('labels.source_label_volume', inputs.label_volume),
		('labels.class_info', inputs.class_info),
		('labels.png_label_inventory', inputs.png_label_inventory),
		('labels.segy_geometry_json', inputs.segy_geometry_json),
		('predictions.token_predictions', inputs.token_predictions),
		('predictions.probability_volume', inputs.probability_volume),
		('predictions.metadata_json', inputs.prediction_metadata_json),
		(
			'predictions.validation_slice_metrics_csv',
			inputs.validation_slice_metrics_csv,
		),
		('visualizations.output_dir', outputs.output_dir),
		('visualizations.metadata_json', outputs.metadata_json),
		('visualizations.selected_slices_dir', outputs.selected_slices_dir),
	]
	source_label = _optional_absolute_path(
		labels,
		'source_label_segy',
		prefix='labels',
	)
	if source_label is not None:
		paths.append(('labels.source_label_segy', source_label))
	return tuple(paths)


def _embedded_comparison_config(
	comparison: Mapping[str, object],
	*,
	artifact_root: Path,
	dataset: Mapping[str, object],
) -> F3LithologyComparisonReportConfig:
	version = _optional_str(
		dataset,
		'version',
		default='facies_benchmark_v1',
		prefix='dataset',
	)
	default_key = ExperimentKey(dataset='f3', version=version)
	default_paths = ArtifactPaths(artifact_root)
	default_search_root = default_paths.lithology_dataset(default_key)
	default_output_dir = default_paths.baseline_comparison_report(default_key)
	search_root = _optional_absolute_path(
		comparison,
		'search_root',
		prefix='comparison',
		default=default_search_root,
	)
	output_dir = _optional_absolute_path(
		comparison,
		'output_dir',
		prefix='comparison',
		default=default_output_dir,
	)
	return F3LithologyComparisonReportConfig(
		search_root=search_root,
		output_csv=_optional_absolute_path(
			comparison,
			'output_csv',
			prefix='comparison',
			default=output_dir / 'comparison_table.csv',
		),
		output_markdown=_optional_absolute_path(
			comparison,
			'output_markdown',
			prefix='comparison',
			default=output_dir / 'comparison_report.md',
		),
	)


def _report_paths(  # noqa: PLR0913
	*,
	metrics_json: Path,
	probe_config_json: Path | None,
	token_dataset_metadata_json: Path | None,
	prediction_metadata_json: Path | None,
	visualization_metadata_json: Path | None,
	output_dir: Path,
	output_markdown: Path,
	output_json: Path,
	comparison: F3LithologyComparisonReportConfig,
) -> tuple[tuple[str, Path], ...]:
	paths = [
		('probe.metrics_json', metrics_json),
		('reports.output_dir', output_dir),
		('reports.output_markdown', output_markdown),
		('reports.output_json', output_json),
		('comparison.search_root', comparison.search_root),
		('comparison.output_csv', comparison.output_csv),
		('comparison.output_markdown', comparison.output_markdown),
	]
	optional_paths = (
		('probe.probe_config_resolved_json', probe_config_json),
		('reports.token_dataset_metadata_json', token_dataset_metadata_json),
		('predictions.metadata_json', prediction_metadata_json),
		('visualizations.metadata_json', visualization_metadata_json),
	)
	paths.extend((label, path) for label, path in optional_paths if path is not None)
	return tuple(paths)


__all__ = [
	'f3_lithology_prediction_config_from_mapping',
	'f3_lithology_probe_config_from_mapping',
	'f3_lithology_publish_config_from_mapping',
	'f3_lithology_report_config_from_mapping',
	'f3_lithology_token_dataset_config_from_mapping',
	'f3_lithology_visualization_config_from_mapping',
	'f3_prepare_volume_config_from_mapping',
]
