"""Configuration for the encoder-independent F3 voxel supervision artifact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_required_nonnegative_int,
	_required_str,
	_validate_allowed_keys,
	_validate_output_not_under_f3_root,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


@dataclass(frozen=True)
class F3LithologyVoxelDatasetConfig:
	"""Resolved inputs and policy for one voxel supervision artifact."""

	artifact_root: Path
	f3_root: Path
	dataset: Mapping[str, str]
	source_label_volume: Path
	source_label_segy: Path
	png_label_inventory: Path
	class_info: Path
	segy_geometry_json: Path
	reference_metadata_json: Path
	reference_valid_tokens: Path
	output_dir: Path
	ignore_z_border_samples: int
	overwrite: bool


def f3_lithology_voxel_dataset_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyVoxelDatasetConfig:
	"""Strictly validate and resolve an F3 voxel supervision config."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'labels',
				'reference_embedding',
				'voxel_dataset',
				'outputs',
			}
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	dataset = _required_mapping(config, 'dataset')
	labels = _required_mapping(config, 'labels')
	reference = _required_mapping(config, 'reference_embedding')
	voxel = _required_mapping(config, 'voxel_dataset')
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
				'class_info',
				'segy_geometry_json',
			}
		),
		prefix='labels',
	)
	_validate_allowed_keys(
		reference,
		frozenset({'metadata_json', 'valid_tokens'}),
		prefix='reference_embedding',
	)
	_validate_allowed_keys(
		voxel,
		frozenset({'output_dir', 'ignore_z_border_samples'}),
		prefix='voxel_dataset',
	)
	_validate_allowed_keys(outputs, frozenset({'overwrite'}), prefix='outputs')

	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	resolved_paths = {
		'labels.source_label_volume': _required_absolute_path(
			labels, 'source_label_volume', prefix='labels'
		),
		'labels.source_label_segy': _required_absolute_path(
			labels, 'source_label_segy', prefix='labels'
		),
		'labels.png_label_inventory': _required_absolute_path(
			labels, 'png_label_inventory', prefix='labels'
		),
		'labels.class_info': _required_absolute_path(
			labels, 'class_info', prefix='labels'
		),
		'labels.segy_geometry_json': _required_absolute_path(
			labels, 'segy_geometry_json', prefix='labels'
		),
		'reference_embedding.metadata_json': _required_absolute_path(
			reference, 'metadata_json', prefix='reference_embedding'
		),
		'reference_embedding.valid_tokens': _required_absolute_path(
			reference, 'valid_tokens', prefix='reference_embedding'
		),
	}
	output_dir = _required_absolute_path(voxel, 'output_dir', prefix='voxel_dataset')
	_validate_output_not_under_f3_root(
		output_dir,
		'voxel_dataset.output_dir',
		f3_root=f3_root,
	)
	overwrite = outputs.get('overwrite')
	if not isinstance(overwrite, bool):
		raise TypeError(f'outputs.overwrite must be a boolean; got {overwrite!r}')
	return F3LithologyVoxelDatasetConfig(
		artifact_root=artifact_root,
		f3_root=f3_root,
		dataset={
			'name': _required_str(dataset, 'name', prefix='dataset'),
			'version': _required_str(dataset, 'version', prefix='dataset'),
		},
		source_label_volume=resolved_paths['labels.source_label_volume'],
		source_label_segy=resolved_paths['labels.source_label_segy'],
		png_label_inventory=resolved_paths['labels.png_label_inventory'],
		class_info=resolved_paths['labels.class_info'],
		segy_geometry_json=resolved_paths['labels.segy_geometry_json'],
		reference_metadata_json=resolved_paths['reference_embedding.metadata_json'],
		reference_valid_tokens=resolved_paths['reference_embedding.valid_tokens'],
		output_dir=output_dir,
		ignore_z_border_samples=_required_nonnegative_int(
			voxel, 'ignore_z_border_samples', prefix='voxel_dataset'
		),
		overwrite=overwrite,
	)


__all__ = [
	'F3LithologyVoxelDatasetConfig',
	'f3_lithology_voxel_dataset_config_from_mapping',
]
