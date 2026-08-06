"""Strict configuration for common F3 voxel-prediction evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_required_str,
	_validate_allowed_keys,
	_validate_artifact_path_not_f3,
)

if TYPE_CHECKING:
	from pathlib import Path

DEFAULT_BOUNDARY_TOLERANCES = (1, 2, 4, 8)
DEFAULT_BOUNDARY_REGION_RADII = (1, 2, 4, 8)
DEFAULT_MONITORED_CLASS_IDS = (3, 5)


@dataclass(frozen=True)
class F3LithologyVoxelEvaluationConfig:
	"""Resolved inputs, metric policy, and outputs for one evaluation."""

	artifact_root: Path
	f3_root: Path
	dataset: Mapping[str, str]
	prediction_input_dir: Path
	voxel_dataset_input_dir: Path
	source_label_volume: Path
	source_label_segy: Path
	png_label_inventory: Path
	segy_geometry_json: Path
	class_info: Path
	output_dir: Path
	monitored_class_ids: tuple[int, ...]
	boundary_tolerances: tuple[int, ...]
	boundary_region_radii: tuple[int, ...]
	chunk_size_x: int
	overwrite: bool


def f3_lithology_voxel_evaluation_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyVoxelEvaluationConfig:
	"""Strictly validate and resolve one common V0/V1 evaluation config."""
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
				'outputs',
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
	outputs = _required_mapping(config, 'outputs')
	_validate_allowed_keys(
		paths, frozenset({'artifact_root', 'f3_root'}), prefix='paths'
	)
	_validate_allowed_keys(dataset, frozenset({'name', 'version'}), prefix='dataset')
	_validate_allowed_keys(
		labels,
		frozenset(
			{
				'source_label_volume',
				'source_label_segy',
				'png_label_inventory',
				'segy_geometry_json',
				'class_info',
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
	_validate_allowed_keys(
		evaluation,
		frozenset(
			{
				'monitored_class_ids',
				'boundary_tolerances',
				'boundary_region_radii',
				'chunk_size_x',
			}
		),
		prefix='evaluation',
	)
	_validate_allowed_keys(
		outputs, frozenset({'output_dir', 'overwrite'}), prefix='outputs'
	)

	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	resolved = {
		'voxel_predictions.input_dir': _required_absolute_path(
			predictions, 'input_dir', prefix='voxel_predictions'
		),
		'voxel_dataset.input_dir': _required_absolute_path(
			voxel_dataset, 'input_dir', prefix='voxel_dataset'
		),
		**{
			f'labels.{key}': _required_absolute_path(labels, key, prefix='labels')
			for key in (
				'source_label_volume',
				'source_label_segy',
				'png_label_inventory',
				'segy_geometry_json',
				'class_info',
			)
		},
	}
	output_dir = _required_absolute_path(outputs, 'output_dir', prefix='outputs')
	_validate_artifact_path_not_f3(
		output_dir,
		'outputs.output_dir',
		artifact_root=artifact_root,
		f3_root=f3_root,
	)
	for label, source in resolved.items():
		if _paths_overlap(output_dir, source):
			raise ValueError(f'outputs.output_dir must not overlap {label}')
	overwrite = outputs.get('overwrite', False)
	if not isinstance(overwrite, bool):
		raise TypeError(f'outputs.overwrite must be a boolean; got {overwrite!r}')
	if output_dir.exists() and not overwrite:
		raise FileExistsError(f'refusing to overwrite existing output: {output_dir}')

	monitored = _integer_sequence(
		evaluation.get('monitored_class_ids', DEFAULT_MONITORED_CLASS_IDS),
		'evaluation.monitored_class_ids',
		allow_empty=True,
	)
	tolerances = _integer_sequence(
		evaluation.get('boundary_tolerances', DEFAULT_BOUNDARY_TOLERANCES),
		'evaluation.boundary_tolerances',
	)
	radii = _integer_sequence(
		evaluation.get('boundary_region_radii', DEFAULT_BOUNDARY_REGION_RADII),
		'evaluation.boundary_region_radii',
	)
	chunk_size_x = evaluation.get('chunk_size_x', 8)
	if (
		not isinstance(chunk_size_x, int)
		or isinstance(chunk_size_x, bool)
		or chunk_size_x <= 0
	):
		raise ValueError('evaluation.chunk_size_x must be a positive integer')
	return F3LithologyVoxelEvaluationConfig(
		artifact_root=artifact_root,
		f3_root=f3_root,
		dataset={
			'name': _required_str(dataset, 'name', prefix='dataset'),
			'version': _required_str(dataset, 'version', prefix='dataset'),
		},
		prediction_input_dir=resolved['voxel_predictions.input_dir'],
		voxel_dataset_input_dir=resolved['voxel_dataset.input_dir'],
		source_label_volume=resolved['labels.source_label_volume'],
		source_label_segy=resolved['labels.source_label_segy'],
		png_label_inventory=resolved['labels.png_label_inventory'],
		segy_geometry_json=resolved['labels.segy_geometry_json'],
		class_info=resolved['labels.class_info'],
		output_dir=output_dir,
		monitored_class_ids=monitored,
		boundary_tolerances=tolerances,
		boundary_region_radii=radii,
		chunk_size_x=chunk_size_x,
		overwrite=overwrite,
	)


def _integer_sequence(
	value: object, label: str, *, allow_empty: bool = False
) -> tuple[int, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be a sequence of integers')
	result = tuple(value)
	if not result and not allow_empty:
		raise ValueError(f'{label} must not be empty')
	if any(not isinstance(item, int) or isinstance(item, bool) for item in result):
		raise TypeError(f'{label} must contain integers')
	if any(item < 0 for item in result):
		raise ValueError(f'{label} must contain non-negative integers')
	if len(set(result)) != len(result):
		raise ValueError(f'{label} must not contain duplicates')
	return result


def _paths_overlap(first: Path, second: Path) -> bool:
	left = first.resolve(strict=False)
	right = second.resolve(strict=False)
	return left == right or left in right.parents or right in left.parents


__all__ = [
	'DEFAULT_BOUNDARY_REGION_RADII',
	'DEFAULT_BOUNDARY_TOLERANCES',
	'DEFAULT_MONITORED_CLASS_IDS',
	'F3LithologyVoxelEvaluationConfig',
	'f3_lithology_voxel_evaluation_config_from_mapping',
]
