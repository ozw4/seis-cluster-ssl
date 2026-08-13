"""Strict configuration for F3 section-layout voxel supervision datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_validate_allowed_keys,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


@dataclass(frozen=True)
class F3SectionLayoutDatasetConfig:
	"""Source artifacts and destination for the model-independent builder."""

	section_layout_contract: Path
	canonical_voxel_dataset: Path
	source_label_volume: Path
	png_label_inventory: Path
	segy_geometry_json: Path
	class_info: Path
	reference_valid_tokens: Path
	output_root: Path


def f3_lithology_voxel_section_layout_dataset_config_from_mapping(
	config: Mapping[str, object],
) -> F3SectionLayoutDatasetConfig:
	"""Resolve builder paths and reject every unknown config key."""
	_validate_allowed_keys(config, frozenset({'inputs', 'outputs'}), prefix='config')
	inputs = _required_mapping(config, 'inputs')
	outputs = _required_mapping(config, 'outputs')
	input_names = (
		'section_layout_contract',
		'canonical_voxel_dataset',
		'source_label_volume',
		'png_label_inventory',
		'segy_geometry_json',
		'class_info',
		'reference_valid_tokens',
	)
	_validate_allowed_keys(inputs, frozenset(input_names), prefix='inputs')
	if set(inputs) != set(input_names):
		missing = sorted(set(input_names) - set(inputs))
		raise ValueError(f'inputs must define every source path; missing={missing!r}')
	_validate_allowed_keys(outputs, frozenset({'output_root'}), prefix='outputs')
	if set(outputs) != {'output_root'}:
		raise ValueError('outputs must define exactly output_root')
	resolved = {
		name: _required_absolute_path(inputs, name, prefix='inputs')
		for name in input_names
	}
	return F3SectionLayoutDatasetConfig(
		section_layout_contract=resolved['section_layout_contract'],
		canonical_voxel_dataset=resolved['canonical_voxel_dataset'],
		source_label_volume=resolved['source_label_volume'],
		png_label_inventory=resolved['png_label_inventory'],
		segy_geometry_json=resolved['segy_geometry_json'],
		class_info=resolved['class_info'],
		reference_valid_tokens=resolved['reference_valid_tokens'],
		output_root=_required_absolute_path(outputs, 'output_root', prefix='outputs'),
	)


__all__ = [
	'F3SectionLayoutDatasetConfig',
	'f3_lithology_voxel_section_layout_dataset_config_from_mapping',
]
