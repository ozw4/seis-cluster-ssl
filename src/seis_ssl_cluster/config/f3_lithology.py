"""F3 lithology config validation entrypoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from seis_ssl_cluster.config import f3_lithology_common as _f3_lithology_common
from seis_ssl_cluster.config.f3_lithology_common import (
	_int_tuple,
	_max_file_size_bytes,
	_optional_absolute_path,
	_optional_bool_value,
	_optional_mapping,
	_optional_non_negative_int,
	_optional_path,
	_optional_positive_int,
	_optional_str,
	_percentiles,
	_publish_optional_bool,
	_required_absolute_path,
	_required_fraction,
	_required_mapping,
	_required_nonnegative_int,
	_validate_allowed_keys,
	_validate_artifact_or_f3_source_path,
	_validate_artifact_path_not_f3,
	_validate_frozen_encoder,
)
from seis_ssl_cluster.config.f3_lithology_probe import (
	f3_lithology_probe_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_token_dataset import (
	f3_lithology_token_dataset_config_from_mapping,
)
from seis_ssl_cluster.f3 import (
	F3LithologyComparisonReportConfig,
	F3LithologyPredictionConfig,
	F3LithologyPredictionInputs,
	F3LithologyPredictionOutputs,
	F3LithologyPublishConfig,
	F3LithologyReportConfig,
	F3LithologyTokenPolicy,
	F3LithologyVisualizationConfig,
	F3LithologyVisualizationFigureConfig,
	F3LithologyVisualizationInputs,
	F3LithologyVisualizationOutputs,
	read_f3_lithology_prediction_classes,
	read_f3_lithology_visualization_classes,
)
from seis_ssl_cluster.f3.prepare_volume import f3_prepare_volume_config_from_mapping
from seis_ssl_cluster.paths import ArtifactPaths, ExperimentKey

if TYPE_CHECKING:
	from pathlib import Path

_is_relative_to = _f3_lithology_common._is_relative_to  # noqa: SLF001


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
