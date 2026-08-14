"""Labels and explicit section layouts for the Parihaka Channel benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
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
CHANNEL_AXIS_MAPPING = {'inline': 'x', 'crossline': 'y'}
LAYOUT_IDS = tuple(f'layout_{index:03d}' for index in range(5))
DATA_SIZE_PREFIX = {'small': 1, 'medium': 2, 'large': 4}
CHANNEL_TEST_MODE = 'voxel_complement_of_all_large_training_and_validation_planes'
CHANNEL_SELECTION_SEMANTICS = 'stable_hash_partial_section_token_footprints_v1'


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
class ChannelTrainingSelection:
	"""Strict target-count settings shared by both Channel benchmarks."""

	semantics: str
	allowed_relative_error: float
	target_train_voxel_counts: Mapping[str, int]


@dataclass(frozen=True)
class ChannelSelectionResult:
	"""One deterministic partial-section training selection."""

	layout_id: str
	data_size: str
	train_lines: SectionLines
	semantics: str
	target_train_voxel_count: int
	actual_train_voxel_count: int
	count_error: int
	relative_count_error: float
	selected_token_xyz: tuple[tuple[int, int, int], ...]
	selected_token_xyz_sha256: str
	per_line_contributions: Mapping[str, int]
	class_counts: tuple[int, int]

	def identity(self) -> dict[str, object]:
		"""Return the selection fields carried by run identities."""
		return {
			'semantics': self.semantics,
			'target_train_voxel_count': self.target_train_voxel_count,
			'actual_train_voxel_count': self.actual_train_voxel_count,
			'count_error': self.count_error,
			'relative_count_error': self.relative_count_error,
			'selected_token_xyz': [list(item) for item in self.selected_token_xyz],
			'selected_token_xyz_sha256': self.selected_token_xyz_sha256,
			'per_line_contributions': dict(self.per_line_contributions),
		}


@dataclass(frozen=True)
class ChannelLayouts:
	"""Five training layouts and shared validation sections."""

	training_selection: ChannelTrainingSelection
	validation: SectionLines
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
	outputs = _mapping(config, 'outputs')
	return ChannelInspectionConfig(
		labels=_path(outputs, 'labels_npy', 'outputs'),
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
			'schema_version': 2,
			'artifact_type': 'parihaka_channel_labels',
			**inspection,
			'output_labels': str(config.output_labels),
			'order': 'C',
			'channel_definition': {
				'positive_class_id': CHANNEL_CLASS_ID,
				'negative_class_ids': [1, 2, 3, 4, 6],
			},
			'prepared_label_identity': {
				'labels_sha256': _file_sha256(staged_labels),
				'source_npz_path': str(config.source_npz),
				'source_key': config.array_key,
				'shape': list(PREPARED_SHAPE_XYZ),
				'dtype': LABEL_DTYPE.name,
				'class_definition': {
					'positive_class_id': CHANNEL_CLASS_ID,
					'negative_class_ids': [1, 2, 3, 4, 6],
				},
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


def inspect_prepared_label_identity(
	labels_path: str | Path, metadata_path: str | Path
) -> dict[str, object]:
	"""Validate prepared labels against their recorded content identity."""
	labels_path = Path(labels_path)
	metadata_path = Path(metadata_path)
	labels = np.load(labels_path, mmap_mode='r', allow_pickle=False)
	if labels.ndim != 3:
		raise ValueError('prepared labels must be a 3D XYZ array')
	metadata = _read_label_metadata(metadata_path)
	if metadata.get('artifact_type') != 'parihaka_channel_labels':
		raise ValueError('prepared label metadata artifact_type mismatch')
	if metadata.get('output_labels') != str(labels_path):
		raise ValueError('prepared label metadata output_labels path mismatch')
	identity = metadata.get('prepared_label_identity')
	if not isinstance(identity, Mapping):
		raise TypeError('prepared label metadata identity must be a mapping')
	_validate_prepared_label_identity(identity, labels_path, labels)
	return dict(identity)


def _read_label_metadata(path: Path) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(f'missing prepared label metadata: {path}')
	try:
		metadata = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(f'invalid prepared label metadata JSON: {path}') from exc
	if not isinstance(metadata, Mapping):
		raise TypeError('prepared label metadata must be a mapping')
	return metadata


def _validate_prepared_label_identity(
	identity: Mapping[str, object], labels_path: Path, labels: np.ndarray
) -> None:
	expected_keys = {
		'labels_sha256',
		'source_npz_path',
		'source_key',
		'shape',
		'dtype',
		'class_definition',
	}
	if set(identity) != expected_keys:
		raise ValueError(
			'prepared label metadata identity must contain exactly '
			f'{sorted(expected_keys)!r}'
		)
	sha256 = identity.get('labels_sha256')
	if (
		not isinstance(sha256, str)
		or len(sha256) != 64
		or any(character not in '0123456789abcdef' for character in sha256)
	):
		raise ValueError('prepared label metadata labels_sha256 is invalid')
	if _file_sha256(labels_path) != sha256:
		raise ValueError('prepared label SHA-256 does not match labels_npy')
	if identity.get('shape') != list(labels.shape):
		raise ValueError('prepared label metadata shape mismatch')
	if identity.get('dtype') != labels.dtype.name:
		raise ValueError('prepared label metadata dtype mismatch')
	for key in ('source_npz_path', 'source_key'):
		value = identity.get(key)
		if not isinstance(value, str) or not value:
			raise TypeError(f'prepared label metadata {key} must be non-empty')
	expected_classes = {
		'positive_class_id': CHANNEL_CLASS_ID,
		'negative_class_ids': [1, 2, 3, 4, 6],
	}
	if identity.get('class_definition') != expected_classes:
		raise ValueError('prepared label metadata class_definition mismatch')


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
	expected_keys = {'training_selection', 'validation', 'layouts'}
	if set(raw) != expected_keys:
		raise ValueError(
			f'layout config must contain exactly {sorted(expected_keys)!r}; '
			'inline is fixed to X and crossline is fixed to Y'
		)
	training_selection = _training_selection(_mapping(raw, 'training_selection'))
	validation = _section_lines(_mapping(raw, 'validation'), 'validation', volume_shape)
	layout_values = _mapping(raw, 'layouts')
	if set(layout_values) != set(LAYOUT_IDS):
		raise ValueError(f'layouts must contain exactly {LAYOUT_IDS!r}')
	layouts = {
		layout_id: _section_lines(
			_mapping(layout_values, layout_id), layout_id, volume_shape, expected=4
		)
		for layout_id in LAYOUT_IDS
	}
	for layout_id, lines in layouts.items():
		_validate_training_disjoint(lines, validation, layout_id)
	_validate_unique_training_selections(layouts)
	return ChannelLayouts(
		training_selection=training_selection,
		validation=validation,
		layouts=layouts,
	)


def select_channel_training(  # noqa: C901, PLR0912, PLR0915
	layouts: ChannelLayouts,
	layout_id: str,
	valid_tokens: np.ndarray,
	labels: np.ndarray,
	patch_size_xyz: Sequence[int],
) -> tuple[ChannelSelectionResult, ...]:
	"""Select nested partial-section token footprints for one layout."""
	if layout_id not in layouts.layouts:
		raise ValueError(f'unknown layout_id: {layout_id!r}')
	valid = np.asarray(valid_tokens)
	label_array = np.asarray(labels)
	patch = _positive_triplet(patch_size_xyz, 'patch_size_xyz')
	if valid.ndim != 3 or valid.dtype != np.bool_:
		raise TypeError('valid_tokens must be a 3D bool array')
	if label_array.ndim != 3:
		raise ValueError('labels must be a 3D array')
	expected_grid = tuple(
		(size + patch[axis] - 1) // patch[axis]
		for axis, size in enumerate(label_array.shape)
	)
	if valid.shape != expected_grid:
		raise ValueError('valid-token shape is incompatible with labels and patch size')
	selected_tokens: set[tuple[int, int, int]] = set()
	previous_voxels: set[int] = set()
	results: list[ChannelSelectionResult] = []
	settings = layouts.training_selection
	for data_size in DATA_SIZE_PREFIX:
		lines = selected_training_lines(layouts, layout_id, data_size)
		footprints = _candidate_token_footprints(
			valid, label_array, lines, layouts.validation, patch
		)
		by_xyz = {item[0]: item for item in footprints}
		if not selected_tokens <= set(by_xyz):
			raise AssertionError('nested active lines lost a selected token')
		ordered = sorted(
			footprints,
			key=lambda item: (
				stable_channel_token_sha256(layout_id, item[0]),
				item[0],
			),
		)
		covered = {
			line_key
			for xyz in selected_tokens
			for line_key in by_xyz[xyz][2]
		}
		for line_key in _ordered_line_keys(lines):
			if line_key in covered:
				continue
			footprint = next(
				(
					item
					for item in ordered
					if item[0] not in selected_tokens and line_key in item[2]
				),
				None,
			)
			if footprint is None:
				raise ValueError(
					f'{layout_id}/{data_size} active line {line_key!r} '
					'contributes no teacher voxels'
				)
			selected_tokens.add(footprint[0])
			covered.update(footprint[2])
		target = settings.target_train_voxel_counts[data_size]
		count = sum(len(by_xyz[xyz][1]) for xyz in selected_tokens)
		if count < target:
			for footprint in ordered:
				if footprint[0] in selected_tokens:
					continue
				after = count + len(footprint[1])
				if after < target:
					selected_tokens.add(footprint[0])
					count = after
					continue
				if abs(after - target) < abs(count - target):
					selected_tokens.add(footprint[0])
					count = after
				break
		selected_footprints = tuple(by_xyz[xyz] for xyz in sorted(selected_tokens))
		selected_voxels = {
			flat for _, indices, _ in selected_footprints for flat in indices
		}
		if not previous_voxels <= selected_voxels:
			raise AssertionError('nested selection lost teacher voxels')
		previous_voxels = selected_voxels
		actual = len(selected_voxels)
		error = actual - target
		relative_error = abs(error) / target
		if relative_error > settings.allowed_relative_error + 1e-15:
			raise ValueError(
				f'{layout_id}/{data_size} target relative error '
				f'{relative_error:.6g} exceeds '
				f'{settings.allowed_relative_error:.6g}'
			)
		contributions = _per_line_contributions(selected_footprints, lines)
		if any(value <= 0 for value in contributions.values()):
			raise ValueError(
				f'{layout_id}/{data_size} active line contributes no teacher voxels'
			)
		flat_labels = label_array.reshape(-1)
		selected_indices = np.asarray(sorted(selected_voxels), dtype=np.int64)
		selected_labels = flat_labels[selected_indices]
		channel = int(np.count_nonzero(selected_labels == CHANNEL_CLASS_ID))
		class_counts = (actual - channel, channel)
		if any(value <= 0 for value in class_counts):
			raise ValueError(
				'train sections must contain both Channel and non-Channel voxels '
				f'after selection ({layout_id}/{data_size})'
			)
		coordinates = tuple(sorted(selected_tokens))
		results.append(
			ChannelSelectionResult(
				layout_id=layout_id,
				data_size=data_size,
				train_lines=lines,
				semantics=settings.semantics,
				target_train_voxel_count=target,
				actual_train_voxel_count=actual,
				count_error=error,
				relative_count_error=relative_error,
				selected_token_xyz=coordinates,
				selected_token_xyz_sha256=selected_token_xyz_sha256(coordinates),
				per_line_contributions=contributions,
				class_counts=class_counts,
			)
		)
	return tuple(results)


def selected_token_mask(
	selected_token_xyz: Sequence[Sequence[int]],
	token_grid_shape_xyz: Sequence[int],
) -> np.ndarray:
	"""Reconstruct a token-grid bool mask without materializing voxel artifacts."""
	shape = _positive_triplet(token_grid_shape_xyz, 'token_grid_shape_xyz')
	mask = np.zeros(shape, dtype=np.bool_)
	seen: set[tuple[int, int, int]] = set()
	for raw in selected_token_xyz:
		xyz = _nonnegative_triplet(raw, 'selected_token_xyz entry')
		if xyz in seen:
			raise ValueError('selected_token_xyz must not contain duplicates')
		if any(xyz[axis] >= shape[axis] for axis in range(3)):
			raise ValueError('selected token coordinate is outside the token grid')
		seen.add(xyz)
		mask[xyz] = True
	return mask


def stable_channel_token_sha256(
	layout_id: str, token_xyz: Sequence[int]
) -> str:
	"""Return the specified stable ordering digest for one token coordinate."""
	if layout_id not in LAYOUT_IDS:
		raise ValueError(f'unknown layout_id: {layout_id!r}')
	xyz = _nonnegative_triplet(token_xyz, 'token_xyz')
	text = (
		f'{layout_id}|{xyz[0]},{xyz[1]},{xyz[2]}|'
		f'{CHANNEL_SELECTION_SEMANTICS}'
	)
	return hashlib.sha256(text.encode()).hexdigest()


def selected_token_xyz_sha256(
	selected_token_xyz: Sequence[Sequence[int]],
) -> str:
	"""Return a stable identity digest for an ordered token-coordinate list."""
	coordinates = tuple(
		_nonnegative_triplet(item, 'selected_token_xyz entry')
		for item in selected_token_xyz
	)
	if len(coordinates) != len(set(coordinates)):
		raise ValueError('selected_token_xyz must not contain duplicates')
	encoded = json.dumps(coordinates, separators=(',', ':')).encode()
	return hashlib.sha256(encoded).hexdigest()


def common_reserved_training_lines(layouts: ChannelLayouts) -> SectionLines:
	"""Return the sorted union of every layout's large training candidates."""
	return SectionLines(
		inline=tuple(
			sorted(
				{
					index
					for lines in layouts.layouts.values()
					for index in lines.inline
				}
			)
		),
		crossline=tuple(
			sorted(
				{
					index
					for lines in layouts.layouts.values()
					for index in lines.crossline
				}
			)
		),
	)


def channel_test_definition(reserved_training: SectionLines) -> dict[str, object]:
	"""Return the fixed common-test identity stored by both benchmark regimes."""
	return {
		'mode': CHANNEL_TEST_MODE,
		'reserved_large_inline': list(reserved_training.inline),
		'reserved_large_crossline': list(reserved_training.crossline),
	}


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
	reserved_training: SectionLines,
	split: str,
) -> np.ndarray:
	"""Build one crop mask from train, validation, and common reserved planes."""
	if len(shape) != 3 or len(start_xyz) != 3:
		raise ValueError('shape and start_xyz must contain three values')
	x = np.arange(start_xyz[0], start_xyz[0] + shape[0])[:, None, None]
	y = np.arange(start_xyz[1], start_xyz[1] + shape[1])[None, :, None]
	validation_mask = np.isin(x, validation.inline) | np.isin(
		y, validation.crossline
	)
	reserved_mask = np.isin(x, reserved_training.inline) | np.isin(
		y, reserved_training.crossline
	)
	selected_train_mask = np.isin(x, train.inline) | np.isin(y, train.crossline)
	train_mask = selected_train_mask & ~validation_mask
	test_mask = ~reserved_mask & ~validation_mask
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


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as file_obj:
		for block in iter(lambda: file_obj.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


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


def _validate_training_disjoint(
	train: SectionLines, validation: SectionLines, layout_id: str
) -> None:
	for orientation in ('inline', 'crossline'):
		if set(getattr(train, orientation)) & set(getattr(validation, orientation)):
			raise ValueError(
				f'{layout_id} training and validation {orientation} lines overlap'
			)


def _validate_unique_training_selections(
	layouts: Mapping[str, SectionLines],
) -> None:
	for data_size, prefix in DATA_SIZE_PREFIX.items():
		seen: dict[tuple[frozenset[int], frozenset[int]], str] = {}
		for layout_id in LAYOUT_IDS:
			lines = layouts[layout_id]
			identity = (
				frozenset(lines.inline[:prefix]),
				frozenset(lines.crossline[:prefix]),
			)
			if duplicate := seen.get(identity):
				raise ValueError(
					f'{data_size} training section sets must be unique across layouts; '
					f'{duplicate} and {layout_id} select the same sections'
				)
			seen[identity] = layout_id


def _training_selection(value: Mapping[str, object]) -> ChannelTrainingSelection:
	expected = {'semantics', 'allowed_relative_error', 'target_train_voxel_counts'}
	if set(value) != expected:
		raise ValueError(
			'training_selection must contain exactly '
			f'{sorted(expected)!r}'
		)
	semantics = value.get('semantics')
	if semantics != CHANNEL_SELECTION_SEMANTICS:
		raise ValueError(
			f'training_selection.semantics must equal '
			f'{CHANNEL_SELECTION_SEMANTICS!r}'
		)
	tolerance = value.get('allowed_relative_error')
	if (
		not isinstance(tolerance, int | float)
		or isinstance(tolerance, bool)
		or not math.isfinite(tolerance)
		or not 0 < tolerance <= 0.1
	):
		raise ValueError(
			'training_selection.allowed_relative_error must be finite and in (0, 0.1]'
		)
	targets = _mapping(value, 'target_train_voxel_counts')
	if set(targets) != set(DATA_SIZE_PREFIX):
		raise ValueError(
			'training_selection.target_train_voxel_counts must define exactly '
			"small, medium, and large"
		)
	resolved: dict[str, int] = {}
	for data_size in DATA_SIZE_PREFIX:
		target = targets.get(data_size)
		if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
			raise ValueError(
				'training_selection.target_train_voxel_counts.'
				f'{data_size} must be a positive integer'
			)
		resolved[data_size] = target
	if not resolved['small'] < resolved['medium'] < resolved['large']:
		raise ValueError(
			'training selection targets must satisfy small < medium < large'
		)
	return ChannelTrainingSelection(
		semantics=CHANNEL_SELECTION_SEMANTICS,
		allowed_relative_error=float(tolerance),
		target_train_voxel_counts=resolved,
	)


def _active_section_mask(shape: Sequence[int], lines: SectionLines) -> np.ndarray:
	x = np.arange(shape[0])[:, None, None]
	y = np.arange(shape[1])[None, :, None]
	return np.broadcast_to(
		np.isin(x, lines.inline) | np.isin(y, lines.crossline), tuple(shape)
	).copy()


def _candidate_footprints(
	candidate: np.ndarray,
	lines: SectionLines,
	patch: tuple[int, int, int],
) -> tuple[
	tuple[
		tuple[int, int, int],
		tuple[int, ...],
		Mapping[tuple[str, int], tuple[int, ...]],
	],
	...,
]:
	voxel_xyz = np.argwhere(candidate)
	if not voxel_xyz.size:
		return ()
	token_xyz = np.unique(voxel_xyz // np.asarray(patch, dtype=np.int64), axis=0)
	line_keys = _ordered_line_keys(lines)
	result = []
	for row in token_xyz:
		coordinate = tuple(int(item) for item in row)
		start = tuple(coordinate[axis] * patch[axis] for axis in range(3))
		stop = tuple(
			min(start[axis] + patch[axis], candidate.shape[axis])
			for axis in range(3)
		)
		block = tuple(slice(start[axis], stop[axis]) for axis in range(3))
		local = np.argwhere(candidate[block])
		global_xyz = local + np.asarray(start, dtype=np.int64)
		flat_array = np.ravel_multi_index(global_xyz.T, candidate.shape)
		order = np.argsort(flat_array)
		global_xyz = global_xyz[order]
		flat = tuple(int(value) for value in flat_array[order])
		owned: dict[tuple[str, int], list[int]] = {}
		for xyz, flat_index in zip(global_xyz, flat, strict=True):
			x, y = int(xyz[0]), int(xyz[1])
			owner = next(
				key
				for key in line_keys
				if (key[0] == 'inline' and x == key[1])
				or (key[0] == 'crossline' and y == key[1])
			)
			owned.setdefault(owner, []).append(flat_index)
		result.append(
			(coordinate, flat, {key: tuple(values) for key, values in owned.items()})
		)
	return tuple(result)


def _candidate_token_footprints(
	valid_tokens: np.ndarray,
	labels: np.ndarray,
	lines: SectionLines,
	validation: SectionLines,
	patch: tuple[int, int, int],
) -> tuple[
	tuple[
		tuple[int, int, int],
		tuple[int, ...],
		Mapping[tuple[str, int], tuple[int, ...]],
	],
	...,
]:
	"""Build only active-plane footprints without full-volume bool masks."""
	inline_tokens = tuple(sorted({value // patch[0] for value in lines.inline}))
	crossline_tokens = tuple(
		sorted({value // patch[1] for value in lines.crossline})
	)
	tx = np.arange(valid_tokens.shape[0])[:, None, None]
	ty = np.arange(valid_tokens.shape[1])[None, :, None]
	candidate_tokens = valid_tokens & (
		np.isin(tx, inline_tokens) | np.isin(ty, crossline_tokens)
	)
	line_keys = _ordered_line_keys(lines)
	classes = np.asarray(CLASS_IDS, dtype=labels.dtype)
	result = []
	for row in np.argwhere(candidate_tokens):
		coordinate = tuple(int(item) for item in row)
		start = tuple(coordinate[axis] * patch[axis] for axis in range(3))
		stop = tuple(
			min(start[axis] + patch[axis], labels.shape[axis])
			for axis in range(3)
		)
		x_values = np.arange(start[0], stop[0])[:, None, None]
		y_values = np.arange(start[1], stop[1])[None, :, None]
		active = np.isin(x_values, lines.inline) | np.isin(
			y_values, lines.crossline
		)
		held_out = np.isin(x_values, validation.inline) | np.isin(
			y_values, validation.crossline
		)
		block = tuple(slice(start[axis], stop[axis]) for axis in range(3))
		candidate = active & ~held_out & np.isin(labels[block], classes)
		local = np.argwhere(candidate)
		if not local.size:
			continue
		global_xyz = local + np.asarray(start, dtype=np.int64)
		flat_array = np.ravel_multi_index(global_xyz.T, labels.shape)
		order = np.argsort(flat_array)
		global_xyz = global_xyz[order]
		flat = tuple(int(value) for value in flat_array[order])
		owned: dict[tuple[str, int], list[int]] = {}
		for xyz, flat_index in zip(global_xyz, flat, strict=True):
			x_value, y_value = int(xyz[0]), int(xyz[1])
			owner = next(
				key
				for key in line_keys
				if (key[0] == 'inline' and x_value == key[1])
				or (key[0] == 'crossline' and y_value == key[1])
			)
			owned.setdefault(owner, []).append(flat_index)
		result.append(
			(coordinate, flat, {key: tuple(values) for key, values in owned.items()})
		)
	return tuple(result)


def _ordered_line_keys(lines: SectionLines) -> tuple[tuple[str, int], ...]:
	return tuple(('inline', value) for value in lines.inline) + tuple(
		('crossline', value) for value in lines.crossline
	)


def _per_line_contributions(
	selected: Sequence[
		tuple[
			tuple[int, int, int],
			tuple[int, ...],
			Mapping[tuple[str, int], tuple[int, ...]],
		]
	],
	lines: SectionLines,
) -> dict[str, int]:
	result = {f'{kind}:{index}': 0 for kind, index in _ordered_line_keys(lines)}
	for _, _, owned in selected:
		for (kind, index), indices in owned.items():
			result[f'{kind}:{index}'] += len(indices)
	return result


def _expand_token_mask(
	mask: np.ndarray,
	patch: Sequence[int],
	shape: Sequence[int],
) -> np.ndarray:
	result = np.asarray(mask, dtype=np.bool_)
	for axis, repeats in enumerate(patch):
		result = np.repeat(result, repeats, axis=axis)
	return result[tuple(slice(0, size) for size in shape)]


def _positive_triplet(value: Sequence[int], label: str) -> tuple[int, int, int]:
	if len(value) != 3 or any(
		not isinstance(item, int) or isinstance(item, bool) or item <= 0
		for item in value
	):
		raise ValueError(f'{label} must be a positive integer triple')
	return (int(value[0]), int(value[1]), int(value[2]))


def _nonnegative_triplet(
	value: Sequence[int], label: str
) -> tuple[int, int, int]:
	if len(value) != 3 or any(
		not isinstance(item, int | np.integer)
		or isinstance(item, bool | np.bool_)
		or int(item) < 0
		for item in value
	):
		raise ValueError(f'{label} must be a nonnegative integer triple')
	return (int(value[0]), int(value[1]), int(value[2]))


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
	'CHANNEL_AXIS_MAPPING',
	'CHANNEL_CLASS_ID',
	'CHANNEL_SELECTION_SEMANTICS',
	'CHANNEL_TEST_MODE',
	'CLASS_IDS',
	'DATA_SIZE_PREFIX',
	'LAYOUT_IDS',
	'PREPARED_SHAPE_XYZ',
	'SOURCE_SHAPE_ZXY',
	'ChannelInspectionConfig',
	'ChannelLabelConfig',
	'ChannelLayouts',
	'ChannelSelectionResult',
	'ChannelTrainingSelection',
	'SectionLines',
	'channel_inspection_config_from_mapping',
	'channel_label_config_from_mapping',
	'channel_test_definition',
	'common_reserved_training_lines',
	'inspect_prepared_label_identity',
	'inspect_prepared_labels',
	'inspect_source_labels',
	'load_channel_layouts',
	'prepare_channel_labels',
	'section_statistics',
	'select_channel_training',
	'selected_token_mask',
	'selected_token_xyz_sha256',
	'selected_training_lines',
	'split_mask_for_crop',
	'stable_channel_token_sha256',
	'write_section_statistics',
]
