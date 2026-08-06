"""F3 lithology visualization config validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.config.f3_lithology_common import (
	_int_tuple,
	_optional_absolute_path,
	_optional_bool_value,
	_optional_mapping,
	_optional_positive_int,
	_optional_str,
	_percentiles,
	_required_absolute_path,
	_required_mapping,
	_validate_allowed_keys,
	_validate_artifact_path_not_f3,
	_validate_frozen_encoder,
)
from seis_ssl_cluster.f3.lithology.visualization import (
	F3LithologyVisualizationConfig,
	F3LithologyVisualizationFigureConfig,
	F3LithologyVisualizationInputs,
	F3LithologyVisualizationOutputs,
	read_f3_lithology_visualization_classes,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


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
		if label.startswith('visualizations.'):
			_validate_artifact_path_not_f3(
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


__all__ = ['f3_lithology_visualization_config_from_mapping']
