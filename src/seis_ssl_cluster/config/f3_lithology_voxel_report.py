"""Strict configuration for F3 voxel report generation and publish."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_required_str,
	_validate_allowed_keys,
	_validate_output_not_under_f3_root,
)
from seis_ssl_cluster.f3.lithology.voxel_report import (
	F3LithologyVoxelPublishConfig,
	F3LithologyVoxelReportConfig,
)
from seis_ssl_cluster.f3.lithology.voxel_visualization import (
	F3LithologyVoxelFigureConfig,
)

DEFAULT_REPORTS_ROOT = Path('reports')
_DEFAULT_PUBLISH_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


def f3_lithology_voxel_report_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyVoxelReportConfig:
	"""Validate and resolve the common V0/V1 report config."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'labels',
				'voxel_predictions',
				'voxel_dataset',
				'evaluation',
				'report',
				'outputs',
				'publish',
			}
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	dataset = _required_mapping(config, 'dataset')
	labels = _required_mapping(config, 'labels')
	predictions = _required_mapping(config, 'voxel_predictions')
	voxel_dataset = _required_mapping(config, 'voxel_dataset')
	evaluation = _required_mapping(config, 'evaluation')
	report = _required_mapping(config, 'report')
	outputs = _required_mapping(config, 'outputs')
	publish = config.get('publish', {})
	if not isinstance(publish, Mapping):
		raise TypeError('publish must be a mapping')
	_validate_allowed_keys(
		paths,
		frozenset({'artifact_root', 'f3_root', 'reports_root'}),
		prefix='paths',
	)
	_validate_allowed_keys(dataset, frozenset({'name', 'version'}), prefix='dataset')
	_validate_allowed_keys(
		labels,
		frozenset(
			{
				'seismic_volume',
				'source_label_volume',
				'class_info',
				'png_label_inventory',
				'segy_geometry_json',
			}
		),
		prefix='labels',
	)
	_validate_allowed_keys(
		predictions, frozenset({'input_dir'}), prefix='voxel_predictions'
	)
	_validate_allowed_keys(
		voxel_dataset, frozenset({'input_dir'}), prefix='voxel_dataset'
	)
	_validate_allowed_keys(evaluation, frozenset({'input_dir'}), prefix='evaluation')
	_validate_allowed_keys(
		report,
		frozenset(
			{
				'selected_slices',
				'dpi',
				'include_confidence',
				'amplitude_clip_percentiles',
			}
		),
		prefix='report',
	)
	_validate_allowed_keys(
		outputs, frozenset({'output_dir', 'overwrite'}), prefix='outputs'
	)
	_validate_allowed_keys(
		publish,
		frozenset(
			{
				'enabled',
				'output_dir',
				'max_file_size_mb',
				'overwrite',
			}
		),
		prefix='publish',
	)

	_required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	resolved = {
		'labels.seismic_volume': _required_absolute_path(
			labels, 'seismic_volume', prefix='labels'
		),
		'labels.source_label_volume': _required_absolute_path(
			labels, 'source_label_volume', prefix='labels'
		),
		'labels.class_info': _required_absolute_path(
			labels, 'class_info', prefix='labels'
		),
		'labels.png_label_inventory': _required_absolute_path(
			labels, 'png_label_inventory', prefix='labels'
		),
		'labels.segy_geometry_json': _required_absolute_path(
			labels, 'segy_geometry_json', prefix='labels'
		),
		'voxel_predictions.input_dir': _required_absolute_path(
			predictions, 'input_dir', prefix='voxel_predictions'
		),
		'voxel_dataset.input_dir': _required_absolute_path(
			voxel_dataset, 'input_dir', prefix='voxel_dataset'
		),
		'evaluation.input_dir': _required_absolute_path(
			evaluation, 'input_dir', prefix='evaluation'
		),
	}
	output_dir = _required_absolute_path(outputs, 'output_dir', prefix='outputs')
	_validate_output_not_under_f3_root(
		output_dir,
		'outputs.output_dir',
		f3_root=f3_root,
	)
	for label, source in resolved.items():
		if _paths_overlap(output_dir, source):
			raise ValueError(f'outputs.output_dir must not overlap {label}')
	overwrite = outputs.get('overwrite', False)
	if not isinstance(overwrite, bool):
		raise TypeError('outputs.overwrite must be boolean')
	if output_dir.exists() and not overwrite:
		raise FileExistsError(f'refusing to overwrite existing output: {output_dir}')

	return F3LithologyVoxelReportConfig(
		prediction_input_dir=resolved['voxel_predictions.input_dir'],
		voxel_dataset_input_dir=resolved['voxel_dataset.input_dir'],
		evaluation_input_dir=resolved['evaluation.input_dir'],
		seismic_volume=resolved['labels.seismic_volume'],
		label_volume=resolved['labels.source_label_volume'],
		class_info=resolved['labels.class_info'],
		png_label_inventory=resolved['labels.png_label_inventory'],
		segy_geometry_json=resolved['labels.segy_geometry_json'],
		output_dir=output_dir,
		dataset={
			'name': _required_str(dataset, 'name', prefix='dataset'),
			'version': _required_str(dataset, 'version', prefix='dataset'),
		},
		selected_slices=_selected_slices(report.get('selected_slices')),
		figure=_figure_config(report),
		publish=_publish_config(publish, paths=paths),
		overwrite=overwrite,
	)


def _selected_slices(value: object) -> Mapping[str, tuple[int, ...]]:
	if value is None:
		return {}
	if not isinstance(value, Mapping):
		raise TypeError('report.selected_slices must be a mapping')
	_validate_allowed_keys(
		value, frozenset({'inline', 'crossline'}), prefix='report.selected_slices'
	)
	result = {}
	for key in ('inline', 'crossline'):
		indices = value.get(key, ())
		if not isinstance(indices, Sequence) or isinstance(indices, str | bytes):
			raise TypeError(f'report.selected_slices.{key} must be a sequence')
		if any(not isinstance(item, int) or isinstance(item, bool) for item in indices):
			raise TypeError(f'report.selected_slices.{key} must contain integers')
		if len(set(indices)) != len(indices):
			raise ValueError(
				f'report.selected_slices.{key} must not contain duplicates'
			)
		if indices:
			result[key] = tuple(indices)
	return result


def _figure_config(report: Mapping[str, object]) -> F3LithologyVoxelFigureConfig:
	dpi = report.get('dpi', 150)
	include_confidence = report.get('include_confidence', False)
	percentiles = report.get('amplitude_clip_percentiles', (1.0, 99.0))
	if not isinstance(percentiles, Sequence) or isinstance(percentiles, str | bytes):
		raise TypeError('report.amplitude_clip_percentiles must be a sequence')
	if len(percentiles) != 2 or any(
		not isinstance(value, int | float) or isinstance(value, bool)
		for value in percentiles
	):
		raise ValueError('report.amplitude_clip_percentiles must contain two numbers')
	return F3LithologyVoxelFigureConfig(
		dpi=dpi,  # type: ignore[arg-type]
		include_confidence=include_confidence,  # type: ignore[arg-type]
		amplitude_clip_percentiles=(float(percentiles[0]), float(percentiles[1])),
	)


def _publish_config(
	publish: Mapping[str, object], *, paths: Mapping[str, object]
) -> F3LithologyVoxelPublishConfig:
	enabled = publish.get('enabled', False)
	overwrite = publish.get('overwrite', True)
	if not isinstance(enabled, bool) or not isinstance(overwrite, bool):
		raise TypeError('publish.enabled and publish.overwrite must be boolean')
	output_value = publish.get('output_dir')
	if output_value is not None and (
		not isinstance(output_value, str) or not output_value
	):
		raise TypeError('publish.output_dir must be a non-empty path string')
	reports_value = paths.get('reports_root')
	if reports_value is not None and (
		not isinstance(reports_value, str) or not reports_value
	):
		raise TypeError('paths.reports_root must be a non-empty path string')
	max_mb = publish.get(
		'max_file_size_mb', _DEFAULT_PUBLISH_MAX_FILE_SIZE_BYTES / (1024 * 1024)
	)
	if not isinstance(max_mb, int | float) or isinstance(max_mb, bool) or max_mb <= 0:
		raise ValueError('publish.max_file_size_mb must be positive')
	return F3LithologyVoxelPublishConfig(
		enabled=enabled,
		output_dir=None if output_value is None else Path(output_value),
		reports_root=(
			DEFAULT_REPORTS_ROOT if reports_value is None else Path(reports_value)
		),
		max_file_size_bytes=int(float(max_mb) * 1024 * 1024),
		overwrite=overwrite,
	)


def _paths_overlap(first: Path, second: Path) -> bool:
	left = first.resolve(strict=False)
	right = second.resolve(strict=False)
	return left == right or left in right.parents or right in left.parents


__all__ = [
	'F3LithologyVoxelPublishConfig',
	'F3LithologyVoxelReportConfig',
	'f3_lithology_voxel_report_config_from_mapping',
]
