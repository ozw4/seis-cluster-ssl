"""Build token-level F3 lithology datasets from dense label volumes."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.embedding.sliding_window import token_grid_shape_xyz
from seis_ssl_cluster.f3.labels import (
	F3ClassInfo,
	parse_class_info_payload,
)
from seis_ssl_cluster.f3.splits import (
	F3LineGeometry,
	F3SliceSplitRecord,
	f3_slice_split_manifest,
	load_f3_slice_split_records,
	read_f3_line_geometry,
	resolve_f3_slice_array_index,
)
from seis_ssl_cluster.f3.tokenization import (
	F3TokenizationConfig,
	F3TokenizationSliceResult,
	tokenize_label_slice,
)
from seis_ssl_cluster.f3.visualization import class_id_image_to_rgb

if TYPE_CHECKING:
	from numpy.typing import NDArray

XYZ = tuple[int, int, int]

CLASS_COUNTS_FIELDNAMES = (
	'split',
	'class_id',
	'class_name',
	'count',
	'fraction',
)

_INVALID_LABEL_RGB = (226, 226, 226)
_DROPPED_RGB = (213, 94, 0)


@dataclass(frozen=True)
class F3LithologyTokenPolicy:
	"""Thresholds for converting dense labels to token labels."""

	min_labeled_fraction: float = 0.5
	min_majority_fraction: float = 0.7
	ignore_z_border_samples: int = 1

	def __post_init__(self) -> None:
		"""Validate label aggregation thresholds."""
		_validate_fraction(self.min_labeled_fraction, 'min_labeled_fraction')
		_validate_fraction(self.min_majority_fraction, 'min_majority_fraction')
		if (
			not isinstance(self.ignore_z_border_samples, int)
			or isinstance(self.ignore_z_border_samples, bool)
			or self.ignore_z_border_samples < 0
		):
			msg = (
				'ignore_z_border_samples must be a nonnegative integer; '
				f'got {self.ignore_z_border_samples!r}'
			)
			raise ValueError(msg)

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable policy record."""
		return {
			'min_labeled_fraction': self.min_labeled_fraction,
			'min_majority_fraction': self.min_majority_fraction,
			'ignore_z_border_samples': self.ignore_z_border_samples,
		}


@dataclass(frozen=True)
class F3LithologyTokenDatasetInputs:
	"""Input artifact paths for F3 lithology token dataset construction."""

	embeddings_dir: Path
	label_volume: Path
	seismic_volume: Path
	png_label_inventory: Path
	class_info: Path
	segy_geometry_json: Path
	source_label_segy: Path | None = None
	volume_metadata_json: Path | None = None


@dataclass(frozen=True)
class F3LithologyTokenDatasetOutputs:
	"""Output artifact paths for F3 lithology token datasets."""

	output_dir: Path
	metadata_json: Path
	class_counts_csv: Path
	summary_markdown: Path
	split_manifest_json: Path
	quicklook_dir: Path

	@property
	def train_npz(self) -> Path:
		"""Return train split token dataset path."""
		return self.output_dir / 'train_tokens.npz'

	@property
	def validation_npz(self) -> Path:
		"""Return validation split token dataset path."""
		return self.output_dir / 'validation_tokens.npz'

	@property
	def all_labeled_npz(self) -> Path:
		"""Return all labeled tokens dataset path."""
		return self.output_dir / 'all_labeled_tokens.npz'


@dataclass(frozen=True)
class F3LithologyTokenDatasetConfig:
	"""Complete F3 lithology token dataset build configuration."""

	inputs: F3LithologyTokenDatasetInputs
	outputs: F3LithologyTokenDatasetOutputs
	policy: F3LithologyTokenPolicy
	dataset: Mapping[str, object]
	model: Mapping[str, object]
	figure_dpi: int = 300


@dataclass(frozen=True)
class F3EmbeddingArtifact:
	"""Loaded embedding, valid-token, and metadata artifacts for one survey."""

	survey_id: str
	embeddings_path: Path
	valid_tokens_path: Path
	metadata_path: Path
	embeddings: NDArray[np.generic]
	valid_tokens: NDArray[np.bool_]
	metadata: Mapping[str, object]
	patch_size_xyz: XYZ
	token_grid_shape_xyz: XYZ
	embedding_dim: int


@dataclass(frozen=True)
class F3SliceTokenization:
	"""Token labelization result for one supervised F3 slice."""

	record: F3SliceSplitRecord
	array_index: int
	tokenization: F3TokenizationSliceResult
	usable_mask: NDArray[np.bool_]
	invalid_embedding_mask: NDArray[np.bool_]

	@property
	def retained_tokens(self) -> int:
		"""Return tokens written to the supervised dataset."""
		return int(np.count_nonzero(self.usable_mask))

	@property
	def dropped_tokens(self) -> int:
		"""Return tokens not written to the supervised dataset."""
		return int(self.tokenization.total_tokens - self.retained_tokens)

	@property
	def invalid_embedding_token_count(self) -> int:
		"""Return label-retained tokens dropped by the embedding valid mask."""
		return int(np.count_nonzero(self.invalid_embedding_mask))

	def to_summary_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable per-slice summary."""
		return {
			'relative_path': self.record.relative_path,
			'split': self.record.split,
			'slice_type': self.record.slice_type,
			'slice_index': self.record.slice_index,
			'array_index': self.array_index,
			'token_grid_shape': [
				int(axis) for axis in self.tokenization.retained_mask.shape
			],
			'total_tokens': self.tokenization.total_tokens,
			'retained_tokens': self.retained_tokens,
			'dropped_tokens': self.dropped_tokens,
			'ambiguous_token_count': self.tokenization.ambiguous_token_count,
			'empty_token_count': self.tokenization.empty_token_count,
			'invalid_embedding_token_count': self.invalid_embedding_token_count,
			'class_counts_retained': _class_counts_from_mask(
				self.tokenization.majority_class_ids,
				self.usable_mask,
			),
		}


@dataclass(frozen=True)
class F3TokenArrays:
	"""Flat arrays saved into one token dataset NPZ."""

	features: NDArray[np.float32]
	labels: NDArray[np.int64]
	survey_id: NDArray[np.str_]
	split: NDArray[np.str_]
	slice_type: NDArray[np.str_]
	slice_index: NDArray[np.int64]
	token_xyz: NDArray[np.int64]
	voxel_center_xyz: NDArray[np.float32]
	majority_fraction: NDArray[np.float32]
	labeled_fraction: NDArray[np.float32]

	@property
	def count(self) -> int:
		"""Return number of tokens."""
		return int(self.labels.shape[0])


@dataclass(frozen=True)
class F3SplitTokenOverlapResolution:
	"""Record how duplicate token coordinates across supervised splits were handled."""

	strategy: str
	unique_cross_split_duplicate_tokens: int
	removed_token_rows_by_split: Mapping[str, int]

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable overlap-resolution record."""
		return {
			'strategy': self.strategy,
			'unique_cross_split_duplicate_tokens': (
				self.unique_cross_split_duplicate_tokens
			),
			'removed_token_rows_by_split': dict(self.removed_token_rows_by_split),
		}


@dataclass(frozen=True)
class F3LithologyTokenDatasetResult:
	"""Paths and counts written by the F3 lithology token dataset builder."""

	train_npz: Path
	validation_npz: Path
	all_labeled_npz: Path
	metadata_json: Path
	class_counts_csv: Path
	summary_markdown: Path
	split_manifest_json: Path
	quicklook_paths: tuple[Path, ...]
	train_token_count: int
	validation_token_count: int


def build_f3_lithology_token_dataset(
	config: F3LithologyTokenDatasetConfig,
) -> F3LithologyTokenDatasetResult:
	"""Build train/validation/all token-level lithology dataset artifacts."""
	classes = read_f3_lithology_class_info(config.inputs.class_info)
	slice_records = load_f3_slice_split_records(config.inputs.png_label_inventory)
	geometry = read_f3_line_geometry(config.inputs.segy_geometry_json)
	embedding = _single_embedding_artifact(config.inputs.embeddings_dir)
	label_volume = np.load(config.inputs.label_volume, mmap_mode='r')
	seismic_volume = np.load(config.inputs.seismic_volume, mmap_mode='r')
	_validate_inputs(
		label_volume=label_volume,
		seismic_volume=seismic_volume,
		geometry=geometry,
		embedding=embedding,
	)

	slice_results = [
		tokenize_f3_lithology_slice(
			record,
			label_volume=label_volume,
			valid_tokens=embedding.valid_tokens,
			geometry=geometry,
			patch_size_xyz=embedding.patch_size_xyz,
			policy=config.policy,
			classes=classes,
		)
		for record in slice_records
	]
	slice_results, split_overlap_resolution = _resolve_cross_split_token_overlaps(
		slice_results,
	)
	arrays_by_split = {
		split: _concatenate_token_arrays(
			[
				_token_arrays_for_slice(
					result,
					embedding=embedding,
					volume_shape_xyz=geometry.shape_xyz,
				)
				for result in slice_results
				if result.record.split == split
			],
			embedding_dim=embedding.embedding_dim,
		)
		for split in ('train', 'validation')
	}
	all_arrays = _concatenate_token_arrays(
		list(arrays_by_split.values()),
		embedding_dim=embedding.embedding_dim,
	)

	_write_outputs(
		config,
		classes=classes,
		geometry=geometry,
		embedding=embedding,
		slice_records=slice_records,
		slice_results=slice_results,
		split_overlap_resolution=split_overlap_resolution,
		arrays_by_split=arrays_by_split,
		all_arrays=all_arrays,
	)
	quicklook_paths = write_f3_lithology_token_quicklooks(
		slice_results,
		seismic_volume=seismic_volume,
		label_volume=label_volume,
		classes=classes,
		output_dir=config.outputs.quicklook_dir,
		dpi=config.figure_dpi,
	)
	return F3LithologyTokenDatasetResult(
		train_npz=config.outputs.train_npz,
		validation_npz=config.outputs.validation_npz,
		all_labeled_npz=config.outputs.all_labeled_npz,
		metadata_json=config.outputs.metadata_json,
		class_counts_csv=config.outputs.class_counts_csv,
		summary_markdown=config.outputs.summary_markdown,
		split_manifest_json=config.outputs.split_manifest_json,
		quicklook_paths=quicklook_paths,
		train_token_count=arrays_by_split['train'].count,
		validation_token_count=arrays_by_split['validation'].count,
	)


def tokenize_f3_lithology_slice(  # noqa: PLR0913
	record: F3SliceSplitRecord,
	*,
	label_volume: NDArray[np.generic],
	valid_tokens: NDArray[np.bool_],
	geometry: F3LineGeometry,
	patch_size_xyz: Sequence[int],
	policy: F3LithologyTokenPolicy,
	classes: Sequence[F3ClassInfo] = (),
) -> F3SliceTokenization:
	"""Aggregate a dense F3 label slice into retained token labels."""
	patch = _positive_xyz(patch_size_xyz, 'patch_size_xyz')
	array_index = resolve_f3_slice_array_index(record, geometry)
	label_slice = _label_slice(label_volume, record, array_index)
	class_id_map = _apply_z_border_ignore(
		label_slice,
		ignore_z_border_samples=policy.ignore_z_border_samples,
	)
	tokenization = tokenize_label_slice(
		class_id_map,
		slice_type=record.slice_type,
		slice_index=record.slice_index,
		array_index=array_index,
		config=F3TokenizationConfig(
			patch_size_xyz=patch,
			min_labeled_fraction=policy.min_labeled_fraction,
			min_majority_fraction=policy.min_majority_fraction,
		),
		classes=classes,
	)
	valid_plane = _valid_token_plane(
		valid_tokens,
		record=record,
		tokenization=tokenization,
	)
	usable_mask = tokenization.retained_mask & valid_plane
	return F3SliceTokenization(
		record=record,
		array_index=array_index,
		tokenization=tokenization,
		usable_mask=usable_mask,
		invalid_embedding_mask=tokenization.retained_mask & ~valid_plane,
	)


def load_f3_embedding_artifacts(
	input_dir: str | Path,
) -> tuple[F3EmbeddingArtifact, ...]:
	"""Load all embedding volumes under an F3 embedding output directory."""
	root = Path(input_dir)
	embedding_paths = sorted(root.glob('*.embeddings.npy'))
	if not embedding_paths:
		msg = f'no embedding arrays found under {root}'
		raise FileNotFoundError(msg)
	return tuple(_load_embedding_artifact(path) for path in embedding_paths)


def read_f3_lithology_class_info(path: str | Path) -> tuple[F3ClassInfo, ...]:
	"""Read class-info from raw F3 class_info or inspection palette JSON."""
	class_info_path = Path(path)
	with class_info_path.open(encoding='utf-8') as file_obj:
		payload = json.load(file_obj)
	if not isinstance(payload, Mapping):
		msg = f'class_info JSON must contain an object: {class_info_path}'
		raise TypeError(msg)
	classes = payload.get('classes')
	if classes is None:
		return parse_class_info_payload(payload, source=class_info_path)
	if not isinstance(classes, Sequence) or isinstance(classes, str | bytes):
		msg = f'{class_info_path} classes must be a list'
		raise TypeError(msg)
	return tuple(
		F3ClassInfo(
			class_id=_mapping_int(item, 'class_id', source=class_info_path),
			class_name=_mapping_str(
				item,
				'class_name',
				fallback_key='name',
				source=class_info_path,
			),
			rgb=_mapping_rgb(item, source=class_info_path),
		)
		for item in classes
	)


def write_f3_lithology_token_quicklooks(  # noqa: PLR0913
	slice_results: Sequence[F3SliceTokenization],
	*,
	seismic_volume: NDArray[np.generic],
	label_volume: NDArray[np.generic],
	classes: Sequence[F3ClassInfo],
	output_dir: str | Path,
	dpi: int = 300,
) -> tuple[Path, ...]:
	"""Write token dataset quicklook panels for all supervised slices."""
	if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi <= 0:
		msg = f'dpi must be a positive integer; got {dpi!r}'
		raise ValueError(msg)
	root = Path(output_dir)
	root.mkdir(parents=True, exist_ok=True)
	paths: list[Path] = []
	for result in slice_results:
		path = root / (
			f'{result.record.split}_{result.record.slice_type}_'
			f'{result.record.slice_index:04d}_token_labels.png'
		)
		_save_token_quicklook(
			result,
			seismic_volume=seismic_volume,
			label_volume=label_volume,
			classes=classes,
			path=path,
			dpi=dpi,
		)
		paths.append(path)
	return tuple(paths)


def _single_embedding_artifact(input_dir: Path) -> F3EmbeddingArtifact:
	artifacts = load_f3_embedding_artifacts(input_dir)
	if len(artifacts) != 1:
		msg = (
			'F3 lithology token dataset expects exactly one embedding volume; '
			f'got {len(artifacts)} under {input_dir}'
		)
		raise ValueError(msg)
	return artifacts[0]


def _load_embedding_artifact(path: Path) -> F3EmbeddingArtifact:
	survey_id = path.name.removesuffix('.embeddings.npy')
	valid_tokens_path = path.with_name(f'{survey_id}.valid_tokens.npy')
	metadata_path = path.with_name(f'{survey_id}.embedding_metadata.json')
	if not valid_tokens_path.is_file():
		msg = f'missing valid-token mask for embedding volume: {valid_tokens_path}'
		raise FileNotFoundError(msg)
	if not metadata_path.is_file():
		msg = f'missing embedding metadata for embedding volume: {metadata_path}'
		raise FileNotFoundError(msg)
	with metadata_path.open(encoding='utf-8') as file_obj:
		metadata = json.load(file_obj)
	if not isinstance(metadata, Mapping):
		msg = f'embedding metadata must contain an object: {metadata_path}'
		raise TypeError(msg)
	embeddings = np.load(path, mmap_mode='r')
	valid_tokens = np.asarray(np.load(valid_tokens_path, mmap_mode='r'), dtype=np.bool_)
	if embeddings.ndim != 4:
		msg = f'embeddings must be 4D [tx, ty, tz, dim]; got {embeddings.shape!r}'
		raise ValueError(msg)
	if valid_tokens.shape != embeddings.shape[:3]:
		msg = (
			'valid token mask shape must match embedding token grid; '
			f'got {valid_tokens.shape!r}, expected={embeddings.shape[:3]!r}'
		)
		raise ValueError(msg)
	patch_size = _positive_xyz(metadata.get('patch_size'), 'embedding patch_size')
	token_grid = _positive_xyz(
		metadata.get('token_grid_shape'),
		'embedding token_grid_shape',
	)
	if token_grid != tuple(int(axis) for axis in embeddings.shape[:3]):
		msg = (
			'embedding metadata token_grid_shape does not match array shape; '
			f'metadata={token_grid!r}, array={embeddings.shape[:3]!r}'
		)
		raise ValueError(msg)
	return F3EmbeddingArtifact(
		survey_id=survey_id,
		embeddings_path=path,
		valid_tokens_path=valid_tokens_path,
		metadata_path=metadata_path,
		embeddings=embeddings,
		valid_tokens=valid_tokens,
		metadata=metadata,
		patch_size_xyz=patch_size,
		token_grid_shape_xyz=token_grid,
		embedding_dim=int(embeddings.shape[3]),
	)


def _validate_inputs(
	*,
	label_volume: NDArray[np.generic],
	seismic_volume: NDArray[np.generic],
	geometry: F3LineGeometry,
	embedding: F3EmbeddingArtifact,
) -> None:
	if label_volume.ndim != 3:
		msg = f'label volume must be 3D XYZ; got {label_volume.shape!r}'
		raise ValueError(msg)
	if seismic_volume.shape != label_volume.shape:
		msg = (
			'seismic and label volumes must have matching XYZ shapes; '
			f'seismic={seismic_volume.shape!r}, label={label_volume.shape!r}'
		)
		raise ValueError(msg)
	if tuple(int(axis) for axis in label_volume.shape) != geometry.shape_xyz:
		msg = (
			'F3 geometry shape does not match label volume; '
			f'geometry={geometry.shape_xyz!r}, label={label_volume.shape!r}'
		)
		raise ValueError(msg)
	expected_grid = token_grid_shape_xyz(geometry.shape_xyz, embedding.patch_size_xyz)
	if expected_grid != embedding.token_grid_shape_xyz:
		msg = (
			'embedding token grid does not match label volume and patch size; '
			f'expected={expected_grid!r}, got={embedding.token_grid_shape_xyz!r}'
		)
		raise ValueError(msg)
	metadata_shape = embedding.metadata.get('volume_shape_xyz')
	if metadata_shape is not None:
		volume_shape = _positive_xyz(metadata_shape, 'embedding volume_shape_xyz')
		if volume_shape != geometry.shape_xyz:
			msg = (
				'embedding metadata volume_shape_xyz does not match label volume; '
				f'metadata={volume_shape!r}, label={geometry.shape_xyz!r}'
			)
			raise ValueError(msg)


def _label_slice(
	label_volume: NDArray[np.generic],
	record: F3SliceSplitRecord,
	array_index: int,
) -> NDArray[np.generic]:
	if record.slice_type == 'inline':
		return np.asarray(label_volume[array_index, :, :])
	if record.slice_type == 'crossline':
		return np.asarray(label_volume[:, array_index, :])
	msg = f'slice_type must be inline or crossline; got {record.slice_type!r}'
	raise ValueError(msg)


def _apply_z_border_ignore(
	label_slice: NDArray[np.generic],
	*,
	ignore_z_border_samples: int,
) -> NDArray[np.int32]:
	array = np.asarray(label_slice)
	if array.ndim != 2:
		msg = f'label slice must be 2D; got {array.shape!r}'
		raise ValueError(msg)
	class_ids = _normalize_label_values(array)
	if ignore_z_border_samples == 0:
		return class_ids
	if ignore_z_border_samples * 2 >= class_ids.shape[1]:
		msg = (
			'ignore_z_border_samples is too large for z/sample axis; '
			f'ignore_z_border_samples={ignore_z_border_samples}, '
			f'z_size={class_ids.shape[1]}'
		)
		raise ValueError(msg)
	class_ids[:, :ignore_z_border_samples] = -1
	class_ids[:, -ignore_z_border_samples:] = -1
	return class_ids


def _normalize_label_values(values: NDArray[np.generic]) -> NDArray[np.int32]:
	array = np.asarray(values)
	if not np.issubdtype(array.dtype, np.number):
		msg = f'label values must be numeric; got {array.dtype}'
		raise TypeError(msg)
	finite = np.isfinite(array)
	rounded = np.rint(array)
	if not np.array_equal(array[finite], rounded[finite]):
		msg = 'label volume must contain integer-like class ids'
		raise ValueError(msg)
	labels = np.full(array.shape, -1, dtype=np.int32)
	labels[finite] = rounded[finite].astype(np.int32)
	labels[labels < 0] = -1
	return labels


def _valid_token_plane(
	valid_tokens: NDArray[np.bool_],
	*,
	record: F3SliceSplitRecord,
	tokenization: F3TokenizationSliceResult,
) -> NDArray[np.bool_]:
	fixed = tokenization.plane.fixed_token_index
	if record.slice_type == 'inline':
		plane = valid_tokens[fixed, :, :]
	elif record.slice_type == 'crossline':
		plane = valid_tokens[:, fixed, :]
	else:
		msg = f'slice_type must be inline or crossline; got {record.slice_type!r}'
		raise ValueError(msg)
	if plane.shape != tokenization.retained_mask.shape:
		msg = (
			'valid-token plane shape does not match tokenized label plane; '
			f'valid={plane.shape!r}, labels={tokenization.retained_mask.shape!r}'
		)
		raise ValueError(msg)
	return np.asarray(plane, dtype=np.bool_)


def _token_arrays_for_slice(
	result: F3SliceTokenization,
	*,
	embedding: F3EmbeddingArtifact,
	volume_shape_xyz: XYZ,
) -> F3TokenArrays:
	indices = np.argwhere(result.usable_mask)
	token_xyz = _token_xyz_from_plane_indices(result, indices)
	features = _features_for_tokens(embedding.embeddings, token_xyz)
	labels = result.tokenization.majority_class_ids[result.usable_mask].astype(
		np.int64,
		copy=False,
	)
	count = int(labels.shape[0])
	return F3TokenArrays(
		features=features,
		labels=labels,
		survey_id=_string_array(embedding.survey_id, count),
		split=_string_array(result.record.split, count),
		slice_type=_string_array(result.record.slice_type, count),
		slice_index=np.full(count, result.record.slice_index, dtype=np.int64),
		token_xyz=token_xyz.astype(np.int64, copy=False),
		voxel_center_xyz=_voxel_centers(
			token_xyz,
			patch_size_xyz=embedding.patch_size_xyz,
			volume_shape_xyz=volume_shape_xyz,
		),
		majority_fraction=result.tokenization.majority_fraction[
			result.usable_mask
		].astype(np.float32, copy=False),
		labeled_fraction=result.tokenization.labeled_fraction[
			result.usable_mask
		].astype(np.float32, copy=False),
	)


def _token_xyz_from_plane_indices(
	result: F3SliceTokenization,
	indices: NDArray[np.int64],
) -> NDArray[np.int64]:
	if indices.size == 0:
		return np.empty((0, 3), dtype=np.int64)
	fixed = result.tokenization.plane.fixed_token_index
	row_tokens = indices[:, 0]
	column_tokens = indices[:, 1]
	if result.record.slice_type == 'inline':
		return np.column_stack(
			(
				np.full(indices.shape[0], fixed, dtype=np.int64),
				row_tokens,
				column_tokens,
			),
		)
	if result.record.slice_type == 'crossline':
		return np.column_stack(
			(
				row_tokens,
				np.full(indices.shape[0], fixed, dtype=np.int64),
				column_tokens,
			),
		)
	msg = f'slice_type must be inline or crossline; got {result.record.slice_type!r}'
	raise ValueError(msg)


def _features_for_tokens(
	embeddings: NDArray[np.generic],
	token_xyz: NDArray[np.int64],
) -> NDArray[np.float32]:
	if token_xyz.size == 0:
		return np.empty((0, int(embeddings.shape[3])), dtype=np.float32)
	return np.asarray(
		embeddings[token_xyz[:, 0], token_xyz[:, 1], token_xyz[:, 2]],
		dtype=np.float32,
	)


def _voxel_centers(
	token_xyz: NDArray[np.int64],
	*,
	patch_size_xyz: XYZ,
	volume_shape_xyz: XYZ,
) -> NDArray[np.float32]:
	if token_xyz.size == 0:
		return np.empty((0, 3), dtype=np.float32)
	centers = np.empty(token_xyz.shape, dtype=np.float32)
	for axis, (patch, size) in enumerate(
		zip(patch_size_xyz, volume_shape_xyz, strict=True),
	):
		start = token_xyz[:, axis] * patch
		stop = np.minimum(start + patch, size)
		centers[:, axis] = (start + stop - 1) / 2.0
	return centers


def _concatenate_token_arrays(
	arrays: Sequence[F3TokenArrays],
	*,
	embedding_dim: int,
) -> F3TokenArrays:
	if not arrays:
		return _empty_token_arrays(embedding_dim)
	nonempty = [array for array in arrays if array.count > 0]
	if not nonempty:
		return _empty_token_arrays(embedding_dim)
	return F3TokenArrays(
		features=np.concatenate([array.features for array in nonempty], axis=0),
		labels=np.concatenate([array.labels for array in nonempty], axis=0),
		survey_id=np.concatenate([array.survey_id for array in nonempty], axis=0),
		split=np.concatenate([array.split for array in nonempty], axis=0),
		slice_type=np.concatenate([array.slice_type for array in nonempty], axis=0),
		slice_index=np.concatenate([array.slice_index for array in nonempty], axis=0),
		token_xyz=np.concatenate([array.token_xyz for array in nonempty], axis=0),
		voxel_center_xyz=np.concatenate(
			[array.voxel_center_xyz for array in nonempty],
			axis=0,
		),
		majority_fraction=np.concatenate(
			[array.majority_fraction for array in nonempty],
			axis=0,
		),
		labeled_fraction=np.concatenate(
			[array.labeled_fraction for array in nonempty],
			axis=0,
		),
	)


def _empty_token_arrays(embedding_dim: int) -> F3TokenArrays:
	return F3TokenArrays(
		features=np.empty((0, embedding_dim), dtype=np.float32),
		labels=np.empty((0,), dtype=np.int64),
		survey_id=np.empty((0,), dtype='<U1'),
		split=np.empty((0,), dtype='<U1'),
		slice_type=np.empty((0,), dtype='<U1'),
		slice_index=np.empty((0,), dtype=np.int64),
		token_xyz=np.empty((0, 3), dtype=np.int64),
		voxel_center_xyz=np.empty((0, 3), dtype=np.float32),
		majority_fraction=np.empty((0,), dtype=np.float32),
		labeled_fraction=np.empty((0,), dtype=np.float32),
	)


def _resolve_cross_split_token_overlaps(
	slice_results: Sequence[F3SliceTokenization],
) -> tuple[tuple[F3SliceTokenization, ...], F3SplitTokenOverlapResolution]:
	validation_tokens = _retained_token_set(slice_results, split='validation')
	train_tokens = _retained_token_set(slice_results, split='train')
	overlap_tokens = train_tokens & validation_tokens
	removed_by_split = {'train': 0, 'validation': 0}
	strategy = 'validation_holdout_priority_exclude_duplicate_token_xyz_from_train'
	if not overlap_tokens:
		return (
			tuple(slice_results),
			F3SplitTokenOverlapResolution(
				strategy=strategy,
				unique_cross_split_duplicate_tokens=0,
				removed_token_rows_by_split=removed_by_split,
			),
		)

	adjusted: list[F3SliceTokenization] = []
	for result in slice_results:
		if result.record.split != 'train' or not np.any(result.usable_mask):
			adjusted.append(result)
			continue
		retained_indices = np.argwhere(result.usable_mask)
		token_xyz = _token_xyz_from_plane_indices(result, retained_indices)
		drop_mask = np.fromiter(
			(_token_key(xyz) in overlap_tokens for xyz in token_xyz),
			dtype=np.bool_,
			count=token_xyz.shape[0],
		)
		if not np.any(drop_mask):
			adjusted.append(result)
			continue
		updated_usable_mask = np.array(result.usable_mask, dtype=np.bool_, copy=True)
		drop_indices = retained_indices[drop_mask]
		updated_usable_mask[drop_indices[:, 0], drop_indices[:, 1]] = False
		removed_by_split['train'] += int(np.count_nonzero(drop_mask))
		adjusted.append(
			F3SliceTokenization(
				record=result.record,
				array_index=result.array_index,
				tokenization=result.tokenization,
				usable_mask=updated_usable_mask,
				invalid_embedding_mask=result.invalid_embedding_mask,
			),
		)

	adjusted_results = tuple(adjusted)
	_validate_no_cross_split_token_overlap(adjusted_results)
	return (
		adjusted_results,
		F3SplitTokenOverlapResolution(
			strategy=strategy,
			unique_cross_split_duplicate_tokens=len(overlap_tokens),
			removed_token_rows_by_split=removed_by_split,
		),
	)


def _retained_token_set(
	slice_results: Sequence[F3SliceTokenization],
	*,
	split: str,
) -> set[tuple[int, int, int]]:
	tokens: set[tuple[int, int, int]] = set()
	for result in slice_results:
		if result.record.split != split or not np.any(result.usable_mask):
			continue
		indices = np.argwhere(result.usable_mask)
		token_xyz = _token_xyz_from_plane_indices(result, indices)
		tokens.update(_token_key(xyz) for xyz in token_xyz)
	return tokens


def _validate_no_cross_split_token_overlap(
	slice_results: Sequence[F3SliceTokenization],
) -> None:
	overlap = _retained_token_set(
		slice_results,
		split='train',
	) & _retained_token_set(slice_results, split='validation')
	if overlap:
		msg = (
			'train and validation token datasets must be disjoint by token_xyz; '
			f'found {len(overlap)} overlapping token coordinates'
		)
		raise ValueError(msg)


def _token_key(token_xyz: NDArray[np.integer]) -> tuple[int, int, int]:
	return tuple(int(axis) for axis in token_xyz)


def _string_array(value: str, count: int) -> NDArray[np.str_]:
	dtype = f'<U{max(1, len(value))}'
	return np.full(count, value, dtype=dtype)


def _write_outputs(  # noqa: PLR0913
	config: F3LithologyTokenDatasetConfig,
	*,
	classes: Sequence[F3ClassInfo],
	geometry: F3LineGeometry,
	embedding: F3EmbeddingArtifact,
	slice_records: Sequence[F3SliceSplitRecord],
	slice_results: Sequence[F3SliceTokenization],
	split_overlap_resolution: F3SplitTokenOverlapResolution,
	arrays_by_split: Mapping[str, F3TokenArrays],
	all_arrays: F3TokenArrays,
) -> None:
	outputs = config.outputs
	outputs.output_dir.mkdir(parents=True, exist_ok=True)
	outputs.metadata_json.parent.mkdir(parents=True, exist_ok=True)
	outputs.class_counts_csv.parent.mkdir(parents=True, exist_ok=True)
	outputs.summary_markdown.parent.mkdir(parents=True, exist_ok=True)
	outputs.split_manifest_json.parent.mkdir(parents=True, exist_ok=True)
	_save_npz(outputs.train_npz, arrays_by_split['train'])
	_save_npz(outputs.validation_npz, arrays_by_split['validation'])
	_save_npz(outputs.all_labeled_npz, all_arrays)
	_write_json(outputs.split_manifest_json, f3_slice_split_manifest(slice_records))
	_write_class_counts_csv(outputs.class_counts_csv, classes, arrays_by_split)
	_write_text(
		outputs.summary_markdown,
		_render_summary_markdown(
			arrays_by_split,
			all_arrays=all_arrays,
			slice_results=slice_results,
			split_overlap_resolution=split_overlap_resolution,
		),
	)
	_write_json(
		outputs.metadata_json,
		_metadata_payload(
			config,
			classes=classes,
			geometry=geometry,
			embedding=embedding,
			slice_results=slice_results,
			split_overlap_resolution=split_overlap_resolution,
			arrays_by_split=arrays_by_split,
			all_arrays=all_arrays,
		),
	)


def _save_npz(path: Path, arrays: F3TokenArrays) -> None:
	np.savez_compressed(
		path,
		features=arrays.features,
		labels=arrays.labels,
		survey_id=arrays.survey_id,
		split=arrays.split,
		slice_type=arrays.slice_type,
		slice_index=arrays.slice_index,
		token_xyz=arrays.token_xyz,
		voxel_center_xyz=arrays.voxel_center_xyz,
		majority_fraction=arrays.majority_fraction,
		labeled_fraction=arrays.labeled_fraction,
	)


def _write_class_counts_csv(
	path: Path,
	classes: Sequence[F3ClassInfo],
	arrays_by_split: Mapping[str, F3TokenArrays],
) -> None:
	rows: list[dict[str, object]] = []
	for split in ('train', 'validation'):
		rows.extend(
			_class_count_rows(split, classes, arrays_by_split[split].labels),
		)
	all_labels = (
		np.concatenate(
			[arrays_by_split['train'].labels, arrays_by_split['validation'].labels],
		)
		if arrays_by_split
		else np.empty((0,), dtype=np.int64)
	)
	rows.extend(_class_count_rows('all_labeled', classes, all_labels))
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=CLASS_COUNTS_FIELDNAMES)
		writer.writeheader()
		writer.writerows(rows)


def _class_count_rows(
	split: str,
	classes: Sequence[F3ClassInfo],
	labels: NDArray[np.int64],
) -> list[dict[str, object]]:
	counts = Counter(int(label) for label in labels)
	total = int(labels.shape[0])
	return [
		{
			'split': split,
			'class_id': class_info.class_id,
			'class_name': class_info.class_name,
			'count': int(counts.get(class_info.class_id, 0)),
			'fraction': _fraction(int(counts.get(class_info.class_id, 0)), total),
		}
		for class_info in classes
	]


def _metadata_payload(  # noqa: PLR0913
	config: F3LithologyTokenDatasetConfig,
	*,
	classes: Sequence[F3ClassInfo],
	geometry: F3LineGeometry,
	embedding: F3EmbeddingArtifact,
	slice_results: Sequence[F3SliceTokenization],
	split_overlap_resolution: F3SplitTokenOverlapResolution,
	arrays_by_split: Mapping[str, F3TokenArrays],
	all_arrays: F3TokenArrays,
) -> dict[str, object]:
	return {
		'artifact_type': 'f3_lithology_token_dataset',
		'dataset': dict(config.dataset),
		'model': dict(config.model),
		'label_source_of_truth': 'segy_label_volume',
		'png_label_role': 'train_validation_slice_selection_and_visual_qc',
		'split_strategy': 'png_label_inventory_slice_split_no_random_token_split',
		'cross_split_token_overlap_resolution': (
			split_overlap_resolution.to_dict()
		),
		'no_random_split': True,
		'inputs': {
			'embeddings_dir': str(config.inputs.embeddings_dir),
			'label_volume': str(config.inputs.label_volume),
			'seismic_volume': str(config.inputs.seismic_volume),
			'png_label_inventory': str(config.inputs.png_label_inventory),
			'class_info': str(config.inputs.class_info),
			'segy_geometry_json': str(config.inputs.segy_geometry_json),
			'source_label_segy': (
				None
				if config.inputs.source_label_segy is None
				else str(config.inputs.source_label_segy)
			),
			'volume_metadata_json': (
				None
				if config.inputs.volume_metadata_json is None
				else str(config.inputs.volume_metadata_json)
			),
		},
		'embedding': {
			'survey_id': embedding.survey_id,
			'embeddings_path': str(embedding.embeddings_path),
			'valid_tokens_path': str(embedding.valid_tokens_path),
			'metadata_path': str(embedding.metadata_path),
			'patch_size_xyz': list(embedding.patch_size_xyz),
			'token_grid_shape_xyz': list(embedding.token_grid_shape_xyz),
			'embedding_dim': embedding.embedding_dim,
		},
		'geometry': geometry.to_dict(),
		'tokenization': config.policy.to_dict(),
		'classes': [class_info.to_dict() for class_info in classes],
		'outputs': {
			'train_tokens': str(config.outputs.train_npz),
			'validation_tokens': str(config.outputs.validation_npz),
			'all_labeled_tokens': str(config.outputs.all_labeled_npz),
			'metadata_json': str(config.outputs.metadata_json),
			'class_counts_csv': str(config.outputs.class_counts_csv),
			'summary_markdown': str(config.outputs.summary_markdown),
			'split_manifest_json': str(config.outputs.split_manifest_json),
			'quicklook_dir': str(config.outputs.quicklook_dir),
		},
		'summary': {
			'train_tokens': arrays_by_split['train'].count,
			'validation_tokens': arrays_by_split['validation'].count,
			'all_labeled_tokens': all_arrays.count,
			'slice_count': len(slice_results),
			'total_dropped_tokens': int(
				sum(result.dropped_tokens for result in slice_results),
			),
			'total_ambiguous_tokens': int(
				sum(
					result.tokenization.ambiguous_token_count
					for result in slice_results
				),
			),
			'total_empty_tokens': int(
				sum(result.tokenization.empty_token_count for result in slice_results),
			),
			'total_invalid_embedding_tokens': int(
				sum(result.invalid_embedding_token_count for result in slice_results),
			),
			'cross_split_duplicate_token_xyz': (
				split_overlap_resolution.unique_cross_split_duplicate_tokens
			),
			'cross_split_duplicate_rows_removed_from_train': int(
				split_overlap_resolution.removed_token_rows_by_split.get('train', 0),
			),
		},
		'slices': [result.to_summary_dict() for result in slice_results],
	}


def _render_summary_markdown(
	arrays_by_split: Mapping[str, F3TokenArrays],
	*,
	all_arrays: F3TokenArrays,
	slice_results: Sequence[F3SliceTokenization],
	split_overlap_resolution: F3SplitTokenOverlapResolution,
) -> str:
	lines = [
		'# F3 lithology token dataset',
		'',
		f'- train tokens: {arrays_by_split["train"].count}',
		f'- validation tokens: {arrays_by_split["validation"].count}',
		f'- all labeled tokens: {all_arrays.count}',
		f'- supervised slices: {len(slice_results)}',
		'- split strategy: png_label_inventory slice split; no random token split',
		(
			'- cross-split duplicate token_xyz: '
			f'{split_overlap_resolution.unique_cross_split_duplicate_tokens}'
		),
		(
			'- train token rows removed for validation holdout: '
			f'{split_overlap_resolution.removed_token_rows_by_split.get("train", 0)}'
		),
		'',
		'## Per-slice tokenization',
		'',
		(
			'| split | slice | total | retained | dropped | ambiguous | '
			'empty | invalid_embedding |'
		),
		'|---|---|---:|---:|---:|---:|---:|---:|',
	]
	lines.extend(
		f'| {result.record.split} | '
		f'{result.record.slice_type} {result.record.slice_index} | '
		f'{result.tokenization.total_tokens} | {result.retained_tokens} | '
		f'{result.dropped_tokens} | '
		f'{result.tokenization.ambiguous_token_count} | '
		f'{result.tokenization.empty_token_count} | '
		f'{result.invalid_embedding_token_count} |'
		for result in slice_results
	)
	return '\n'.join(lines) + '\n'


def _save_token_quicklook(  # noqa: PLR0913
	result: F3SliceTokenization,
	*,
	seismic_volume: NDArray[np.generic],
	label_volume: NDArray[np.generic],
	classes: Sequence[F3ClassInfo],
	path: Path,
	dpi: int,
) -> None:
	plt = _matplotlib_pyplot()
	seismic = _display_slice(seismic_volume, result)
	dense_label = _display_slice(label_volume, result)
	retained = _expanded_token_values(result).T
	dropped = _expanded_token_mask(~result.usable_mask, result).T
	clip_low, clip_high = _amplitude_clip(seismic)
	fig, axes = plt.subplots(1, 4, figsize=(13.6, 4.0), dpi=dpi, sharey=True)
	axes[0].imshow(
		seismic,
		cmap='gray',
		vmin=clip_low,
		vmax=clip_high,
		origin='upper',
		aspect='auto',
	)
	axes[0].set_title('seismic')
	axes[1].imshow(
		class_id_image_to_rgb(dense_label, classes, invalid_rgb=_INVALID_LABEL_RGB),
		origin='upper',
		aspect='auto',
		interpolation='nearest',
	)
	axes[1].set_title('dense label')
	axes[2].imshow(
		class_id_image_to_rgb(retained, classes, invalid_rgb=_INVALID_LABEL_RGB),
		origin='upper',
		aspect='auto',
		interpolation='nearest',
	)
	axes[2].set_title('retained token labels')
	axes[3].imshow(
		_binary_mask_rgb(dropped, true_rgb=_DROPPED_RGB),
		origin='upper',
		aspect='auto',
		interpolation='nearest',
	)
	axes[3].set_title('dropped/ambiguous')
	for ax in axes:
		_configure_axes(ax, result)
	fig.legend(
		handles=_class_legend_handles(classes),
		loc='center right',
		bbox_to_anchor=(0.995, 0.5),
		frameon=False,
		fontsize=6,
		title='facies',
		title_fontsize=7,
	)
	fig.suptitle(
		f'{result.record.split} {result.record.slice_type} '
		f'{result.record.slice_index}',
		fontsize=10,
	)
	fig.tight_layout(rect=(0.0, 0.0, 0.90, 0.94))
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(path, facecolor='white', bbox_inches='tight')
	plt.close(fig)


def _display_slice(
	volume: NDArray[np.generic],
	result: F3SliceTokenization,
) -> NDArray[np.generic]:
	array = np.asarray(volume)
	if result.record.slice_type == 'inline':
		return np.asarray(array[result.array_index, :, :]).T
	if result.record.slice_type == 'crossline':
		return np.asarray(array[:, result.array_index, :]).T
	msg = f'slice_type must be inline or crossline; got {result.record.slice_type!r}'
	raise ValueError(msg)


def _expanded_token_values(result: F3SliceTokenization) -> NDArray[np.int32]:
	values = np.full(
		result.tokenization.majority_class_ids.shape,
		-1,
		dtype=np.int32,
	)
	values[result.usable_mask] = result.tokenization.majority_class_ids[
		result.usable_mask
	]
	return _expand_token_plane(values, result)


def _expanded_token_mask(
	mask: NDArray[np.bool_],
	result: F3SliceTokenization,
) -> NDArray[np.bool_]:
	return _expand_token_plane(np.asarray(mask, dtype=np.bool_), result)


def _expand_token_plane(
	token_values: NDArray[np.generic],
	result: F3SliceTokenization,
) -> NDArray[np.generic]:
	output = np.zeros(result.tokenization.class_id_map.shape, dtype=token_values.dtype)
	if np.issubdtype(token_values.dtype, np.integer):
		output[...] = -1
	plane = result.tokenization.plane
	for row_token in range(token_values.shape[0]):
		row_start = row_token * plane.row_patch_size
		row_stop = min(row_start + plane.row_patch_size, output.shape[0])
		for column_token in range(token_values.shape[1]):
			column_start = column_token * plane.column_patch_size
			column_stop = min(
				column_start + plane.column_patch_size,
				output.shape[1],
			)
			output[row_start:row_stop, column_start:column_stop] = token_values[
				row_token,
				column_token,
			]
	return output


def _configure_axes(ax: object, result: F3SliceTokenization) -> None:
	if result.record.slice_type == 'inline':
		ax.set_xlabel('crossline index')
	else:
		ax.set_xlabel('inline index')
	ax.set_ylabel('sample/time index down')
	ax.tick_params(labelsize=6)


def _amplitude_clip(values: NDArray[np.generic]) -> tuple[float, float]:
	array = np.asarray(values, dtype=np.float32)
	finite = array[np.isfinite(array)]
	if finite.size == 0:
		return 0.0, 1.0
	low, high = np.percentile(finite, [1.0, 99.0])
	if float(low) == float(high):
		return float(low) - 1.0, float(high) + 1.0
	return float(low), float(high)


def _binary_mask_rgb(
	mask: NDArray[np.bool_],
	*,
	true_rgb: tuple[int, int, int],
) -> NDArray[np.uint8]:
	array = np.asarray(mask, dtype=np.bool_)
	rgb = np.full((*array.shape, 3), 255, dtype=np.uint8)
	rgb[array] = true_rgb
	return rgb


def _class_legend_handles(classes: Sequence[F3ClassInfo]) -> list[object]:
	patches = _matplotlib_patches()
	return [
		patches.Patch(
			facecolor=class_info.hex_color,
			edgecolor='none',
			label=f'{class_info.class_id}: {class_info.class_name}',
		)
		for class_info in classes
	]


def _matplotlib_pyplot() -> object:
	try:
		return __import__('matplotlib.pyplot', fromlist=['pyplot'])
	except ImportError as exc:
		msg = (
			'F3 lithology token quicklook requires matplotlib; '
			'install seis-cluster-ssl[visualization].'
		)
		raise ImportError(msg) from exc


def _matplotlib_patches() -> object:
	try:
		return __import__('matplotlib.patches', fromlist=['patches'])
	except ImportError as exc:
		msg = (
			'F3 lithology token quicklook requires matplotlib; '
			'install seis-cluster-ssl[visualization].'
		)
		raise ImportError(msg) from exc


def _class_counts_from_mask(
	majority_class_ids: NDArray[np.int32],
	mask: NDArray[np.bool_],
) -> dict[str, int]:
	counts = Counter(int(value) for value in majority_class_ids[mask])
	return {str(class_id): int(count) for class_id, count in sorted(counts.items())}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _write_text(path: Path, text: str) -> None:
	path.write_text(text, encoding='utf-8')


def _mapping_int(
	value: object,
	key: str,
	*,
	source: Path,
) -> int:
	item = _mapping(value, source=source)
	raw = item.get(key)
	if isinstance(raw, bool):
		msg = f'{source} {key} must be an integer; got {raw!r}'
		raise TypeError(msg)
	try:
		return int(cast('object', raw))
	except (TypeError, ValueError) as exc:
		msg = f'{source} {key} must be an integer; got {raw!r}'
		raise ValueError(msg) from exc


def _mapping_str(
	value: object,
	key: str,
	*,
	fallback_key: str,
	source: Path,
) -> str:
	item = _mapping(value, source=source)
	raw = item.get(key)
	if raw is None:
		raw = item.get(fallback_key)
	if not isinstance(raw, str) or not raw:
		msg = f'{source} {key} must be a non-empty string; got {raw!r}'
		raise TypeError(msg)
	return raw


def _mapping_rgb(value: object, *, source: Path) -> tuple[int, int, int]:
	item = _mapping(value, source=source)
	raw = item.get('rgb')
	if raw is None:
		raw = item.get('color')
	if (
		not isinstance(raw, Sequence)
		or isinstance(raw, str | bytes)
		or len(raw) != 3
	):
		msg = f'{source} class RGB must contain three channels; got {raw!r}'
		raise ValueError(msg)
	channels = tuple(raw)
	if not all(
		isinstance(channel, int) and not isinstance(channel, bool)
		for channel in channels
	):
		msg = f'{source} class RGB channels must be integers; got {raw!r}'
		raise TypeError(msg)
	if any(channel < 0 or channel > 255 for channel in channels):
		msg = f'{source} class RGB channels must be in [0, 255]; got {raw!r}'
		raise ValueError(msg)
	return cast('tuple[int, int, int]', channels)


def _mapping(value: object, *, source: Path) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		msg = f'{source} class entries must be objects; got {value!r}'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _positive_xyz(value: object, label: str) -> XYZ:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 3
	):
		msg = f'{label} must be a length-3 sequence; got {value!r}'
		raise TypeError(msg)
	values = tuple(value)
	if not all(isinstance(axis, int) and not isinstance(axis, bool) for axis in values):
		msg = f'{label} values must be integers; got {value!r}'
		raise TypeError(msg)
	xyz = cast('XYZ', values)
	if any(axis <= 0 for axis in xyz):
		msg = f'{label} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _validate_fraction(value: object, label: str) -> None:
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'{label} must be a number in [0, 1]; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not 0.0 <= fraction <= 1.0:
		msg = f'{label} must be in [0, 1]; got {value!r}'
		raise ValueError(msg)


def _fraction(numerator: int, denominator: int) -> float:
	if denominator == 0:
		return 0.0
	return float(numerator / denominator)


__all__ = [
	'CLASS_COUNTS_FIELDNAMES',
	'F3EmbeddingArtifact',
	'F3LithologyTokenDatasetConfig',
	'F3LithologyTokenDatasetInputs',
	'F3LithologyTokenDatasetOutputs',
	'F3LithologyTokenDatasetResult',
	'F3LithologyTokenPolicy',
	'F3SliceTokenization',
	'F3TokenArrays',
	'build_f3_lithology_token_dataset',
	'load_f3_embedding_artifacts',
	'read_f3_lithology_class_info',
	'tokenize_f3_lithology_slice',
	'write_f3_lithology_token_quicklooks',
]
