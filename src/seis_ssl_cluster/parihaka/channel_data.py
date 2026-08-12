# ruff: noqa: CPY001
"""Labels and explicit section layouts for the Parihaka Channel benchmark."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from seis_ssl_cluster.config import load_config

SOURCE_SHAPE_ZXY = (1006, 782, 590)
PREPARED_SHAPE_XYZ = (782, 590, 1006)
LABEL_DTYPE = np.dtype(np.int8)
CLASS_IDS = (1, 2, 3, 4, 5, 6)
CHANNEL_CLASS_ID = 5
LAYOUT_IDS = tuple(f'layout_{index:03d}' for index in range(5))
DATA_SIZE_PREFIX = {'small': 1, 'medium': 2, 'large': 4}


@dataclass(frozen=True)
class ChannelLabelConfig:
	"""Resolved label preparation inputs and outputs."""

	source_npz: Path
	output_labels: Path
	output_metadata: Path
	array_key: str = 'labels'
	chunk_size_z: int = 8


@dataclass(frozen=True)
class ChannelInspectionConfig:
	"""Resolved section-statistics inputs and output."""

	labels: Path
	output_csv: Path


@dataclass(frozen=True)
class SectionLines:
	"""Ordered inline and crossline indices."""

	inline: tuple[int, ...]
	crossline: tuple[int, ...]


@dataclass(frozen=True)
class ChannelLayouts:
	"""Five training layouts and shared held-out sections."""

	validation: SectionLines
	test: SectionLines
	layouts: Mapping[str, SectionLines]


def channel_label_config_from_mapping(
	config: Mapping[str, object],
) -> ChannelLabelConfig:
	"""Resolve the small label-preparation config."""
	inputs = _mapping(config, 'inputs')
	outputs = _mapping(config, 'outputs')
	conversion = _mapping(config, 'conversion')
	key = inputs.get('array_key', 'labels')
	if not isinstance(key, str) or not key:
		raise ValueError('inputs.array_key must be a non-empty string')
	chunk_size = conversion.get('chunk_size_z', 8)
	if (
		not isinstance(chunk_size, int)
		or isinstance(chunk_size, bool)
		or chunk_size <= 0
	):
		raise ValueError('conversion.chunk_size_z must be a positive integer')
	return ChannelLabelConfig(
		source_npz=_path(inputs, 'labels_npz', 'inputs'),
		output_labels=_path(outputs, 'labels_npy', 'outputs'),
		output_metadata=_path(outputs, 'metadata_json', 'outputs'),
		array_key=key,
		chunk_size_z=chunk_size,
	)


def channel_inspection_config_from_mapping(
	config: Mapping[str, object],
) -> ChannelInspectionConfig:
	"""Resolve the section-inspection config."""
	inputs = _mapping(config, 'inputs')
	outputs = _mapping(config, 'outputs')
	return ChannelInspectionConfig(
		labels=_path(inputs, 'labels_npy', 'inputs'),
		output_csv=_path(outputs, 'section_counts_csv', 'outputs'),
	)


def inspect_source_labels(config: ChannelLabelConfig) -> dict[str, object]:
	"""Validate source identity and values without writing outputs."""
	if not config.source_npz.is_file():
		raise FileNotFoundError(f'missing Parihaka label NPZ: {config.source_npz}')
	with np.load(config.source_npz, allow_pickle=False) as archive:
		if set(archive.files) != {config.array_key}:
			raise ValueError(
				f'label NPZ must contain only key {config.array_key!r}; '
				f'got {archive.files!r}'
			)
		labels = archive[config.array_key]
		_validate_source_array(labels)
		class_ids = _chunked_class_ids(labels, config.chunk_size_z)
	if class_ids != CLASS_IDS:
		raise ValueError(f'label class IDs must be {CLASS_IDS!r}; got {class_ids!r}')
	return {
		'key': config.array_key,
		'source_axes': ['Z', 'X', 'Y'],
		'source_shape': list(SOURCE_SHAPE_ZXY),
		'dtype': LABEL_DTYPE.name,
		'class_ids': list(class_ids),
		'destination_axes': ['X', 'Y', 'Z'],
		'destination_shape': list(PREPARED_SHAPE_XYZ),
		'transpose_axes': [1, 2, 0],
		'chunk_size_z': config.chunk_size_z,
	}


def prepare_channel_labels(config: ChannelLabelConfig) -> tuple[Path, Path]:
	"""Validate and transpose the label volume into a C-contiguous XYZ NPY."""
	inspection = inspect_source_labels(config)
	for path in (config.output_labels, config.output_metadata):
		if path.exists():
			raise FileExistsError(f'channel label output already exists: {path}')
	config.output_labels.parent.mkdir(parents=True, exist_ok=True)
	config.output_metadata.parent.mkdir(parents=True, exist_ok=True)
	staged_labels = config.output_labels.with_name(f'.{config.output_labels.name}.tmp')
	staged_metadata = config.output_metadata.with_name(
		f'.{config.output_metadata.name}.tmp'
	)
	try:
		with np.load(config.source_npz, allow_pickle=False) as archive:
			source = archive[config.array_key]
			output = np.lib.format.open_memmap(
				staged_labels,
				mode='w+',
				dtype=LABEL_DTYPE,
				shape=PREPARED_SHAPE_XYZ,
				fortran_order=False,
			)
			for z_start in range(0, SOURCE_SHAPE_ZXY[0], config.chunk_size_z):
				z_stop = min(z_start + config.chunk_size_z, SOURCE_SHAPE_ZXY[0])
				output[:, :, z_start:z_stop] = source[z_start:z_stop].transpose(1, 2, 0)
			output.flush()
			_verify_coordinates(source, output)
			del output
		metadata = {
			'schema_version': 1,
			'artifact_type': 'parihaka_channel_labels',
			**inspection,
			'output_labels': str(config.output_labels),
			'order': 'C',
			'channel_definition': {
				'positive_class_id': CHANNEL_CLASS_ID,
				'negative_class_ids': [1, 2, 3, 4, 6],
			},
		}
		staged_metadata.write_text(
			json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8'
		)
		staged_labels.replace(config.output_labels)
		staged_metadata.replace(config.output_metadata)
	except BaseException:
		for path in (staged_labels, staged_metadata):
			if path.exists():
				path.unlink()
		raise
	return config.output_labels, config.output_metadata


def inspect_prepared_labels(path: str | Path) -> np.ndarray:
	"""Open and validate the prepared XYZ label array as a memory map."""
	labels = np.load(Path(path), mmap_mode='r', allow_pickle=False)
	if labels.shape != PREPARED_SHAPE_XYZ:
		raise ValueError(
			f'prepared labels shape must be {PREPARED_SHAPE_XYZ!r}; '
			f'got {labels.shape!r}'
		)
	if labels.dtype != LABEL_DTYPE:
		raise TypeError(
			f'prepared labels dtype must be {LABEL_DTYPE.name}; got {labels.dtype}'
		)
	return labels


def write_section_statistics(config: ChannelInspectionConfig) -> int:
	"""Write Channel counts for every X and Y section without selecting lines."""
	labels = inspect_prepared_labels(config.labels)
	config.output_csv.parent.mkdir(parents=True, exist_ok=True)
	rows = section_statistics(labels)
	with config.output_csv.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=tuple(rows[0]))
		writer.writeheader()
		writer.writerows(rows)
	return len(rows)


def section_statistics(labels: np.ndarray) -> list[dict[str, object]]:
	"""Return per-X and per-Y Channel statistics in coordinate order."""
	if labels.ndim != 3:
		raise ValueError('labels must be a 3D XYZ array')
	rows: list[dict[str, object]] = []
	for orientation, count in (('x', labels.shape[0]), ('y', labels.shape[1])):
		for index in range(count):
			section = labels[index, :, :] if orientation == 'x' else labels[:, index, :]
			channel = int(np.count_nonzero(section == CHANNEL_CLASS_ID))
			non_channel = int(section.size - channel)
			rows.append(
				{
					'orientation': orientation,
					'section_index': index,
					'channel_voxel_count': channel,
					'non_channel_voxel_count': non_channel,
					'channel_fraction': channel / section.size,
				}
			)
	return rows


def load_channel_layouts(
	path: str | Path, volume_shape: Sequence[int]
) -> ChannelLayouts:
	"""Load and validate five explicit, ordered section layouts."""
	raw = load_config(path)
	mapping = _mapping(raw, 'axis_mapping')
	if mapping != {'inline': 'x', 'crossline': 'y'}:
		raise ValueError(
			"axis_mapping must explicitly be {'inline': 'x', 'crossline': 'y'}"
		)
	validation = _section_lines(_mapping(raw, 'validation'), 'validation', volume_shape)
	test = _section_lines(_mapping(raw, 'test'), 'test', volume_shape)
	layout_values = _mapping(raw, 'layouts')
	if set(layout_values) != set(LAYOUT_IDS):
		raise ValueError(f'layouts must contain exactly {LAYOUT_IDS!r}')
	layouts = {
		layout_id: _section_lines(
			_mapping(layout_values, layout_id), layout_id, volume_shape, expected=4
		)
		for layout_id in LAYOUT_IDS
	}
	_validate_disjoint_held_out(validation, test)
	for layout_id, lines in layouts.items():
		_validate_training_disjoint(lines, validation, test, layout_id)
	return ChannelLayouts(validation=validation, test=test, layouts=layouts)


def selected_training_lines(
	layouts: ChannelLayouts, layout_id: str, data_size: str
) -> SectionLines:
	"""Return the ordered 1+1, 2+2, or 4+4 training prefix."""
	if layout_id not in layouts.layouts:
		raise ValueError(f'unknown layout_id: {layout_id!r}')
	try:
		prefix = DATA_SIZE_PREFIX[data_size]
	except KeyError as error:
		raise ValueError(f'unknown data size: {data_size!r}') from error
	lines = layouts.layouts[layout_id]
	return SectionLines(lines.inline[:prefix], lines.crossline[:prefix])


def split_mask_for_crop(  # noqa: PLR0913
	*,
	shape: Sequence[int],
	start_xyz: Sequence[int],
	train: SectionLines,
	validation: SectionLines,
	test: SectionLines,
	split: str,
) -> np.ndarray:
	"""Build one crop mask using test > validation > train > ignore priority."""
	if len(shape) != 3 or len(start_xyz) != 3:
		raise ValueError('shape and start_xyz must contain three values')
	x = np.arange(start_xyz[0], start_xyz[0] + shape[0])[:, None, None]
	y = np.arange(start_xyz[1], start_xyz[1] + shape[1])[None, :, None]
	test_mask = np.isin(x, test.inline) | np.isin(y, test.crossline)
	validation_mask = (~test_mask) & (
		np.isin(x, validation.inline) | np.isin(y, validation.crossline)
	)
	train_mask = (
		(~test_mask)
		& (~validation_mask)
		& (np.isin(x, train.inline) | np.isin(y, train.crossline))
	)
	if split == 'test':
		return np.broadcast_to(test_mask, tuple(shape)).copy()
	if split == 'validation':
		return np.broadcast_to(validation_mask, tuple(shape)).copy()
	if split == 'train':
		return np.broadcast_to(train_mask, tuple(shape)).copy()
	raise ValueError("split must be 'train', 'validation', or 'test'")


def _validate_source_array(labels: np.ndarray) -> None:
	if labels.shape != SOURCE_SHAPE_ZXY:
		raise ValueError(
			f'source labels shape must be {SOURCE_SHAPE_ZXY!r}; got {labels.shape!r}'
		)
	if labels.dtype != LABEL_DTYPE:
		raise TypeError(
			f'source labels dtype must be {LABEL_DTYPE.name}; got {labels.dtype}'
		)


def _chunked_class_ids(labels: np.ndarray, chunk_size_z: int) -> tuple[int, ...]:
	values: set[int] = set()
	for start in range(0, labels.shape[0], chunk_size_z):
		values.update(
			int(item) for item in np.unique(labels[start : start + chunk_size_z])
		)
	return tuple(sorted(values))


def _verify_coordinates(source: np.ndarray, output: np.ndarray) -> None:
	z_size, x_size, y_size = source.shape
	coordinates = (
		(0, 0, 0),
		(z_size - 1, x_size - 1, y_size - 1),
		(z_size // 2, x_size // 2, y_size // 2),
		(min(17, z_size - 1), min(29, x_size - 1), min(41, y_size - 1)),
	)
	for z, x, y in coordinates:
		if source[z, x, y] != output[x, y, z]:
			raise RuntimeError(f'label coordinate verification failed at {(z, x, y)!r}')
	if not output.flags.c_contiguous:
		raise RuntimeError('prepared label output must be C-contiguous')


def _section_lines(
	value: Mapping[str, object],
	label: str,
	volume_shape: Sequence[int],
	*,
	expected: int | None = None,
) -> SectionLines:
	inline = _indices(value.get('inline'), f'{label}.inline', int(volume_shape[0]))
	crossline = _indices(
		value.get('crossline'), f'{label}.crossline', int(volume_shape[1])
	)
	if expected is not None and (len(inline) != expected or len(crossline) != expected):
		raise ValueError(
			f'{label} must contain exactly {expected} inline and crossline indices'
		)
	return SectionLines(inline, crossline)


def _indices(value: object, label: str, bound: int) -> tuple[int, ...]:
	if not isinstance(value, list) or not value:
		raise TypeError(f'{label} must be a non-empty ordered list')
	if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
		raise TypeError(f'{label} must contain integers')
	items = cast('tuple[int, ...]', tuple(value))
	if len(set(items)) != len(items):
		raise ValueError(f'{label} must not contain duplicates')
	if any(item < 0 or item >= bound for item in items):
		raise ValueError(f'{label} contains an out-of-range index for size {bound}')
	return items


def _validate_disjoint_held_out(validation: SectionLines, test: SectionLines) -> None:
	for orientation in ('inline', 'crossline'):
		if set(getattr(validation, orientation)) & set(getattr(test, orientation)):
			raise ValueError(f'validation and test {orientation} line numbers overlap')


def _validate_training_disjoint(
	train: SectionLines, validation: SectionLines, test: SectionLines, layout_id: str
) -> None:
	for orientation in ('inline', 'crossline'):
		held_out = set(getattr(validation, orientation)) | set(
			getattr(test, orientation)
		)
		if set(getattr(train, orientation)) & held_out:
			raise ValueError(
				f'{layout_id} training and held-out {orientation} lines overlap'
			)


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return child


def _path(value: Mapping[str, object], key: str, prefix: str) -> Path:
	item = value.get(key)
	if not isinstance(item, str) or not item:
		raise ValueError(f'{prefix}.{key} must be a non-empty path')
	path = Path(item)
	if not path.is_absolute():
		raise ValueError(f'{prefix}.{key} must be absolute')
	return path


__all__ = [
	'CHANNEL_CLASS_ID',
	'CLASS_IDS',
	'DATA_SIZE_PREFIX',
	'LAYOUT_IDS',
	'PREPARED_SHAPE_XYZ',
	'SOURCE_SHAPE_ZXY',
	'ChannelInspectionConfig',
	'ChannelLabelConfig',
	'ChannelLayouts',
	'SectionLines',
	'channel_inspection_config_from_mapping',
	'channel_label_config_from_mapping',
	'inspect_prepared_labels',
	'inspect_source_labels',
	'load_channel_layouts',
	'prepare_channel_labels',
	'section_statistics',
	'selected_training_lines',
	'split_mask_for_crop',
	'write_section_statistics',
]
