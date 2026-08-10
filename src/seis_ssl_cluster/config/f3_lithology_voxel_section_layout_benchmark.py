"""Closed configuration for the F3 section-layout decoder benchmark."""
# ruff: noqa: CPY001

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pathlib import Path

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_required_str,
	_validate_allowed_keys,
)
from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	VoxelDecoderSpec,
	VoxelDecoderTileSettings,
	VoxelDecoderTrainSettings,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DECODER_SEED,
	FIXED_DECODER_CONTRACT,
)


@dataclass(frozen=True)
class F3SectionLayoutBenchmarkConfig:
	"""Resolved sources and the one fixed decoder policy for the generic runner."""

	artifact_root: Path
	f3_root: Path
	labels: Mapping[str, Path]
	dataset: Mapping[str, str]
	model_roster: Path
	dataset_manifest: Path
	decoder: VoxelDecoderSpec
	tiles: VoxelDecoderTileSettings
	train: VoxelDecoderTrainSettings
	write_probabilities: bool
	evaluation: Mapping[str, object]
	benchmark_root: Path
	smoke_root: Path


def f3_lithology_voxel_section_layout_benchmark_config_from_mapping(
	config: Mapping[str, object],
) -> F3SectionLayoutBenchmarkConfig:
	"""Resolve the runner mapping and reject every unregistered setting."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'references',
				'decoder',
				'tiles',
				'train',
				'inference',
				'evaluation',
				'outputs',
			}
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	dataset = _required_mapping(config, 'dataset')
	references = _required_mapping(config, 'references')
	decoder = _required_mapping(config, 'decoder')
	tiles = _required_mapping(config, 'tiles')
	train = _required_mapping(config, 'train')
	inference = _required_mapping(config, 'inference')
	evaluation = _required_mapping(config, 'evaluation')
	outputs = _required_mapping(config, 'outputs')
	_validate_allowed_keys(
		paths,
		frozenset(
			{
				'artifact_root',
				'f3_root',
				'source_label_volume',
				'source_label_segy',
				'png_label_inventory',
				'segy_geometry_json',
				'class_info',
			}
		),
		prefix='paths',
	)
	_validate_allowed_keys(dataset, frozenset({'name', 'version'}), prefix='dataset')
	_validate_allowed_keys(
		references,
		frozenset({'model_roster', 'section_layout_dataset_manifest'}),
		prefix='references',
	)
	_validate_allowed_keys(
		decoder,
		frozenset(FIXED_DECODER_CONTRACT)
		- {
			'epochs',
			'batch_size',
			'learning_rate',
			'weight_decay',
			'class_weight',
			'sampling_mode',
			'steps_per_epoch',
			'amp',
			'gradient_clip_norm',
			'write_probabilities',
			'seed',
		},
		prefix='decoder',
	)
	_validate_allowed_keys(
		tiles, frozenset({'core_size_tokens', 'context_halo_tokens'}), prefix='tiles'
	)
	_validate_allowed_keys(
		train,
		frozenset(
			{
				'epochs',
				'batch_size',
				'learning_rate',
				'weight_decay',
				'class_weight',
				'sampling_mode',
				'steps_per_epoch',
				'seed',
				'num_workers',
				'amp',
				'gradient_clip_norm',
			}
		),
		prefix='train',
	)
	_validate_allowed_keys(
		inference, frozenset({'write_probabilities'}), prefix='inference'
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
		outputs, frozenset({'benchmark_root', 'smoke_root'}), prefix='outputs'
	)
	_validate_fixed_mapping(decoder, 'decoder', keys=tuple(decoder))
	_validate_fixed_mapping(
		train,
		'train',
		keys=(
			'epochs',
			'batch_size',
			'learning_rate',
			'weight_decay',
			'class_weight',
			'sampling_mode',
			'steps_per_epoch',
			'seed',
			'amp',
			'gradient_clip_norm',
		),
	)
	_validate_fixed_mapping(inference, 'inference', keys=('write_probabilities',))
	num_workers = _integer(train.get('num_workers'), 'train.num_workers', minimum=0)
	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	benchmark_root = _required_absolute_path(
		outputs, 'benchmark_root', prefix='outputs'
	)
	smoke_root = _required_absolute_path(outputs, 'smoke_root', prefix='outputs')
	if (
		benchmark_root == smoke_root
		or benchmark_root in smoke_root.parents
		or smoke_root in benchmark_root.parents
	):
		raise ValueError('outputs.benchmark_root and smoke_root must be disjoint')
	return F3SectionLayoutBenchmarkConfig(
		artifact_root=artifact_root,
		f3_root=f3_root,
		labels={
			key: _required_absolute_path(paths, key, prefix='paths')
			for key in (
				'source_label_volume',
				'source_label_segy',
				'png_label_inventory',
				'segy_geometry_json',
				'class_info',
			)
		},
		dataset={
			'name': _required_str(dataset, 'name', prefix='dataset'),
			'version': _required_str(dataset, 'version', prefix='dataset'),
		},
		model_roster=_required_absolute_path(
			references, 'model_roster', prefix='references'
		),
		dataset_manifest=_required_absolute_path(
			references, 'section_layout_dataset_manifest', prefix='references'
		),
		decoder=VoxelDecoderSpec(
			spec=str(decoder['spec']),
			embedding_dim=int(decoder['embedding_dim']),
			class_count=int(decoder['class_count']),
			hidden_channels=tuple(
				int(value)
				for value in _sequence(
					decoder['hidden_channels'], 'decoder.hidden_channels'
				)
			),
			upsample_factors=tuple(
				tuple(
					int(item)
					for item in _sequence(value, 'decoder.upsample_factors item')
				)
				for value in _sequence(
					decoder['upsample_factors'], 'decoder.upsample_factors'
				)
			),
			upsample_mode=str(decoder['upsample_mode']),
			normalization=str(decoder['normalization']),
		),
		tiles=VoxelDecoderTileSettings(
			core_size_tokens=_triplet(
				tiles.get('core_size_tokens'), 'tiles.core_size_tokens', minimum=1
			),
			context_halo_tokens=_triplet(
				tiles.get('context_halo_tokens'), 'tiles.context_halo_tokens', minimum=0
			),
		),
		train=VoxelDecoderTrainSettings(
			epochs=50,
			batch_size=1,
			learning_rate=0.001,
			weight_decay=0.0001,
			class_weight='balanced',
			seed=DECODER_SEED,
			num_workers=num_workers,
			amp=True,
			gradient_clip_norm=1.0,
			sampling_mode='uniform_tiles_with_replacement',
			steps_per_epoch=440,
		),
		write_probabilities=False,
		evaluation={
			'monitored_class_ids': list(
				_integer_sequence(
					evaluation.get('monitored_class_ids'),
					'evaluation.monitored_class_ids',
					minimum=0,
				)
			),
			'boundary_tolerances': list(
				_integer_sequence(
					evaluation.get('boundary_tolerances'),
					'evaluation.boundary_tolerances',
					minimum=1,
				)
			),
			'boundary_region_radii': list(
				_integer_sequence(
					evaluation.get('boundary_region_radii'),
					'evaluation.boundary_region_radii',
					minimum=1,
				)
			),
			'chunk_size_x': _integer(
				evaluation.get('chunk_size_x'), 'evaluation.chunk_size_x', minimum=1
			),
		},
		benchmark_root=benchmark_root,
		smoke_root=smoke_root,
	)


resolve_f3_lithology_voxel_section_layout_benchmark_config = (
	f3_lithology_voxel_section_layout_benchmark_config_from_mapping
)


def _validate_fixed_mapping(
	value: Mapping[str, object], prefix: str, *, keys: tuple[str, ...]
) -> None:
	for key in keys:
		expected = FIXED_DECODER_CONTRACT[key]
		actual = value.get(key)
		if isinstance(expected, tuple):
			actual = _plain_tuple(actual)
		if not _same_fixed_value(actual, expected):
			raise ValueError(f'{prefix}.{key} must be exactly {expected!r}')


def _plain_tuple(value: object) -> object:
	if isinstance(value, Sequence) and not isinstance(value, str | bytes):
		return tuple(_plain_tuple(item) for item in value)
	return value


def _same_fixed_value(actual: object, expected: object) -> bool:
	if isinstance(expected, tuple):
		return (
			isinstance(actual, tuple)
			and len(actual) == len(expected)
			and all(
				_same_fixed_value(left, right)
				for left, right in zip(actual, expected, strict=True)
			)
		)
	if isinstance(expected, bool):
		return isinstance(actual, bool) and actual is expected
	if isinstance(expected, int):
		return (
			isinstance(actual, int)
			and not isinstance(actual, bool)
			and actual == expected
		)
	if isinstance(expected, float):
		return isinstance(actual, float) and actual == expected
	return type(actual) is type(expected) and actual == expected


def _sequence(value: object, label: str) -> tuple[object, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be a sequence')
	return tuple(value)


def _triplet(value: object, label: str, *, minimum: int) -> tuple[int, int, int]:
	items = _integer_sequence(value, label, minimum=minimum)
	if len(items) != 3:
		raise ValueError(f'{label} must contain exactly three integers')
	return items  # type: ignore[return-value]


def _integer_sequence(value: object, label: str, *, minimum: int) -> tuple[int, ...]:
	items = _sequence(value, label)
	return tuple(_integer(item, f'{label} item', minimum=minimum) for item in items)


def _integer(value: object, label: str, *, minimum: int) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
		raise ValueError(f'{label} must be an integer >= {minimum}')
	return value


__all__ = [
	'F3SectionLayoutBenchmarkConfig',
	'f3_lithology_voxel_section_layout_benchmark_config_from_mapping',
	'resolve_f3_lithology_voxel_section_layout_benchmark_config',
]
