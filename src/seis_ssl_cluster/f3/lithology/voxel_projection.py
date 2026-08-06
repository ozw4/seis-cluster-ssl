"""Project full-volume F3 token predictions to their nearest voxels."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.embedding.sliding_window import token_grid_shape_xyz
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_geometry import (
	project_token_grid_nearest,
	valid_tokens_to_voxel_mask,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	ARTIFACT_TYPE,
	INVALID_CONFIDENCE_VALUE,
	INVALID_PREDICTION_CLASS_ID,
	METADATA_NAME,
	SCHEMA_VERSION,
	F3VoxelPredictionArrays,
	commit_f3_voxel_prediction_artifact,
	create_f3_voxel_prediction_staging_dir,
	open_f3_voxel_prediction_memmaps,
	validate_f3_voxel_prediction_arrays,
	write_f3_voxel_prediction_metadata,
)

if TYPE_CHECKING:
	from numpy.typing import NDArray

SOURCE_PREDICTIONS_NAME = 'f3_token_predictions.npy'
SOURCE_PROBABILITIES_NAME = 'f3_token_probabilities.npy'
SOURCE_VALID_GRID_NAME = 'f3_valid_token_grid.npy'
SOURCE_METADATA_NAME = 'prediction_metadata.json'


@dataclass(frozen=True)
class F3VoxelProjectionResult:
	"""Committed nearest-projection artifact paths and voxel counts."""

	output_dir: Path
	volume_shape_xyz: tuple[int, int, int]
	valid_voxel_count: int
	invalid_voxel_count: int
	probabilities_written: bool


@dataclass(frozen=True)
class F3VoxelProjectionSourceInfo:
	"""Validated identity and geometry of one token-prediction artifact."""

	input_dir: Path
	predictions: Path
	probabilities: Path
	valid_tokens: Path
	metadata_json: Path
	metadata: Mapping[str, object]
	volume_shape_xyz: tuple[int, int, int]
	patch_size_xyz: tuple[int, int, int]
	token_grid_shape_xyz: tuple[int, int, int]
	class_probability_order: tuple[int, ...]
	model_tag: str


@dataclass(frozen=True)
class _TokenPredictionArtifact:
	root: Path
	metadata_path: Path
	metadata: Mapping[str, object]
	predictions_path: Path
	probabilities_path: Path
	valid_tokens_path: Path
	predictions: NDArray[np.int16]
	probabilities: NDArray[np.float32]
	valid_tokens: NDArray[np.bool_]
	volume_shape_xyz: tuple[int, int, int]
	patch_size_xyz: tuple[int, int, int]
	token_grid_shape_xyz: tuple[int, int, int]
	class_probability_order: tuple[int, ...]
	classes: tuple[Mapping[str, object], ...]
	model_tag: str


def inspect_f3_lithology_token_projection_source(
	token_prediction_dir: str | Path,
) -> F3VoxelProjectionSourceInfo:
	"""Validate a token artifact and return its projection-relevant identity."""
	source = _load_and_validate_token_artifact(Path(token_prediction_dir))
	return F3VoxelProjectionSourceInfo(
		input_dir=source.root,
		predictions=source.predictions_path,
		probabilities=source.probabilities_path,
		valid_tokens=source.valid_tokens_path,
		metadata_json=source.metadata_path,
		metadata=source.metadata,
		volume_shape_xyz=source.volume_shape_xyz,
		patch_size_xyz=source.patch_size_xyz,
		token_grid_shape_xyz=source.token_grid_shape_xyz,
		class_probability_order=source.class_probability_order,
		model_tag=source.model_tag,
	)


def project_f3_lithology_tokens_to_voxels(
	token_prediction_dir: str | Path,
	output_dir: str | Path,
	*,
	write_probabilities: bool = False,
	token_axis_chunk_size: int = 1,
	overwrite: bool = False,
) -> F3VoxelProjectionResult:
	"""Write the canonical nearest-repeat voxel projection of a token artifact."""
	if not isinstance(write_probabilities, bool):
		raise TypeError('write_probabilities must be bool')
	if not isinstance(overwrite, bool):
		raise TypeError('overwrite must be bool')
	if (
		not isinstance(token_axis_chunk_size, int)
		or isinstance(token_axis_chunk_size, bool)
		or token_axis_chunk_size <= 0
	):
		raise ValueError('token_axis_chunk_size must be a positive integer')

	source = _load_and_validate_token_artifact(Path(token_prediction_dir))
	output_root = Path(output_dir)
	resolved_source = source.root.resolve(strict=True)
	resolved_output = output_root.resolve(strict=False)
	if resolved_source.is_relative_to(resolved_output):
		raise ValueError(
			'output_dir must differ from token_prediction_dir and must not contain '
			'token_prediction_dir, to preserve the source token artifact'
		)
	if resolved_output.is_relative_to(resolved_source):
		raise ValueError(
			'output_dir must not be inside token_prediction_dir, to preserve the '
			'source token artifact'
		)
	staging = create_f3_voxel_prediction_staging_dir(
		output_root, overwrite=overwrite
	)
	try:
		arrays = open_f3_voxel_prediction_memmaps(
			staging,
			volume_shape_xyz=source.volume_shape_xyz,
			class_count=len(source.class_probability_order),
			include_probabilities=write_probabilities,
		)
		_write_projection_chunks(
			source,
			arrays=arrays,
			token_axis_chunk_size=token_axis_chunk_size,
		)
		for array in (
			arrays.predictions,
			arrays.confidence,
			arrays.valid_mask,
			arrays.probabilities,
		):
			if isinstance(array, np.memmap):
				array.flush()
		summary = validate_f3_voxel_prediction_arrays(
			arrays,
			volume_shape_xyz=source.volume_shape_xyz,
			class_probability_order=source.class_probability_order,
		)
		metadata = _projection_metadata(
			source,
			output_dir=output_root,
			write_probabilities=write_probabilities,
			summary=summary,
		)
		write_f3_voxel_prediction_metadata(staging / METADATA_NAME, metadata)
		committed_output_dir = commit_f3_voxel_prediction_artifact(
			staging, output_root, overwrite=overwrite
		)
	except BaseException:
		shutil.rmtree(staging, ignore_errors=True)
		raise
	return F3VoxelProjectionResult(
		output_dir=committed_output_dir,
		volume_shape_xyz=source.volume_shape_xyz,
		valid_voxel_count=cast('int', summary['valid_voxel_count']),
		invalid_voxel_count=cast('int', summary['invalid_voxel_count']),
		probabilities_written=write_probabilities,
	)


def _load_and_validate_token_artifact(root: Path) -> _TokenPredictionArtifact:
	paths = {
		'predictions': root / SOURCE_PREDICTIONS_NAME,
		'probabilities': root / SOURCE_PROBABILITIES_NAME,
		'valid_tokens': root / SOURCE_VALID_GRID_NAME,
		'metadata': root / SOURCE_METADATA_NAME,
	}
	missing = [path.name for path in paths.values() if not path.is_file()]
	if missing:
		raise FileNotFoundError(
			'incomplete token prediction artifact; missing: ' + ', '.join(missing)
		)
	with paths['metadata'].open(encoding='utf-8') as file_obj:
		metadata_value = json.load(file_obj, parse_constant=_reject_json_constant)
	if not isinstance(metadata_value, Mapping):
		raise TypeError('token prediction metadata must contain a JSON object')
	metadata = cast('Mapping[str, object]', metadata_value)
	if metadata.get('artifact_type') != 'f3_lithology_token_predictions':
		raise ValueError(
			"token prediction metadata artifact_type must be "
			"'f3_lithology_token_predictions'"
		)
	_validate_token_metadata_output_binding(metadata, paths=paths)

	geometry = _mapping(metadata.get('geometry'), 'geometry')
	embedding = _mapping(metadata.get('embedding'), 'embedding')
	model = _mapping(metadata.get('model'), 'model')
	_mapping(metadata.get('probe'), 'probe')
	_mapping(metadata.get('embeddings'), 'embeddings')
	volume_shape = _positive_triplet(geometry.get('shape_xyz'), 'geometry.shape_xyz')
	patch_size = _positive_triplet(
		embedding.get('patch_size_xyz'), 'embedding.patch_size_xyz'
	)
	token_shape = _positive_triplet(
		embedding.get('token_grid_shape_xyz'),
		'embedding.token_grid_shape_xyz',
	)
	expected_grid = token_grid_shape_xyz(volume_shape, patch_size)
	if token_shape != expected_grid:
		raise ValueError(
			'token prediction metadata geometry is inconsistent; '
			f'expected token grid {expected_grid!r}, got {token_shape!r}'
		)
	class_order = _class_order(metadata.get('class_probability_order'))
	classes = _classes(metadata.get('classes'), class_order=class_order)
	model_tag = model.get('tag')
	if not isinstance(model_tag, str) or not model_tag:
		raise TypeError(
			'token prediction metadata model.tag must be a non-empty string'
		)
	if metadata.get('invalid_prediction_class_id') != INVALID_PREDICTION_CLASS_ID:
		raise ValueError('token prediction invalid_prediction_class_id must be -1')
	if metadata.get('invalid_probability_value') != 'nan':
		raise ValueError("token prediction invalid_probability_value must be 'nan'")

	predictions = np.load(paths['predictions'], mmap_mode='r', allow_pickle=False)
	probabilities = np.load(paths['probabilities'], mmap_mode='r', allow_pickle=False)
	valid_tokens = np.load(paths['valid_tokens'], mmap_mode='r', allow_pickle=False)
	_validate_token_arrays(
		predictions,
		probabilities,
		valid_tokens,
		token_shape=token_shape,
		class_order=class_order,
	)
	_validate_token_summary(
		metadata,
		predictions=predictions,
		probabilities=probabilities,
		valid_tokens=valid_tokens,
	)
	return _TokenPredictionArtifact(
		root=root,
		metadata_path=paths['metadata'],
		metadata=metadata,
		predictions_path=paths['predictions'],
		probabilities_path=paths['probabilities'],
		valid_tokens_path=paths['valid_tokens'],
		predictions=predictions,
		probabilities=probabilities,
		valid_tokens=valid_tokens,
		volume_shape_xyz=volume_shape,
		patch_size_xyz=patch_size,
		token_grid_shape_xyz=token_shape,
		class_probability_order=class_order,
		classes=classes,
		model_tag=model_tag,
	)


def _validate_token_arrays(  # noqa: C901
	predictions: np.ndarray,
	probabilities: np.ndarray,
	valid_tokens: np.ndarray,
	*,
	token_shape: tuple[int, int, int],
	class_order: tuple[int, ...],
) -> None:
	for label, array, dtype, shape in (
		('predictions', predictions, np.dtype(np.int16), token_shape),
		(
			'probabilities',
			probabilities,
			np.dtype(np.float32),
			(*token_shape, len(class_order)),
		),
		('valid token grid', valid_tokens, np.dtype(np.bool_), token_shape),
	):
		if array.dtype != dtype:
			raise TypeError(f'token {label} dtype must be {dtype}; got {array.dtype}')
		if tuple(array.shape) != shape:
			raise ValueError(
				f'token {label} shape must be {shape!r}; got {array.shape!r}'
			)
	valid = np.asarray(valid_tokens).ravel()
	flat_predictions = predictions.reshape(-1)
	flat_probabilities = probabilities.reshape((-1, len(class_order)))
	for start in range(0, valid.size, 1_000_000):
		stop = min(start + 1_000_000, valid.size)
		mask = valid[start:stop]
		chunk_predictions = np.asarray(flat_predictions[start:stop])
		chunk_probabilities = np.asarray(flat_probabilities[start:stop])
		if np.any(chunk_predictions[~mask] != INVALID_PREDICTION_CLASS_ID):
			raise ValueError('invalid tokens must use prediction class ID -1')
		if np.any(~np.isnan(chunk_probabilities[~mask])):
			raise ValueError('invalid token probability rows must contain only NaN')
		if not np.any(mask):
			continue
		valid_predictions = chunk_predictions[mask]
		valid_probabilities = chunk_probabilities[mask]
		if not np.all(np.isin(valid_predictions, class_order)):
			raise ValueError('valid tokens contain an unknown prediction class ID')
		if not np.all(np.isfinite(valid_probabilities)) or np.any(
			valid_probabilities < 0.0
		):
			raise ValueError(
				'valid token probabilities must be finite and non-negative'
			)
		if not np.allclose(
			valid_probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6
		):
			raise ValueError('valid token probability rows must sum to 1')
		argmax_predictions = np.asarray(class_order, dtype=np.int16)[
			np.argmax(valid_probabilities, axis=1)
		]
		if not np.array_equal(valid_predictions, argmax_predictions):
			raise ValueError(
				'token hard predictions must match probability argmax in class order'
			)


def _validate_token_summary(
	metadata: Mapping[str, object],
	*,
	predictions: np.ndarray,
	probabilities: np.ndarray,
	valid_tokens: np.ndarray,
) -> None:
	summary = _mapping(metadata.get('summary'), 'summary')
	expected = {
		'token_grid_shape_xyz': list(predictions.shape),
		'probability_grid_shape': list(probabilities.shape),
		'valid_token_count': int(np.count_nonzero(valid_tokens)),
		'invalid_token_count': int(valid_tokens.size - np.count_nonzero(valid_tokens)),
	}
	for key, value in expected.items():
		if summary.get(key) != value:
			raise ValueError(
				f'token prediction metadata summary {key} does not match arrays; '
				f'metadata={summary.get(key)!r}, arrays={value!r}'
			)


def _write_projection_chunks(
	source: _TokenPredictionArtifact,
	*,
	arrays: F3VoxelPredictionArrays,
	token_axis_chunk_size: int,
) -> None:
	for token_start in range(
		0, source.token_grid_shape_xyz[0], token_axis_chunk_size
	):
		token_stop = min(
			token_start + token_axis_chunk_size, source.token_grid_shape_xyz[0]
		)
		voxel_start = token_start * source.patch_size_xyz[0]
		voxel_stop = min(
			token_stop * source.patch_size_xyz[0], source.volume_shape_xyz[0]
		)
		if voxel_start >= voxel_stop:
			break
		local_shape = (
			voxel_stop - voxel_start,
			source.volume_shape_xyz[1],
			source.volume_shape_xyz[2],
		)
		token_slice = slice(token_start, token_stop)
		valid = valid_tokens_to_voxel_mask(
			source.valid_tokens[token_slice],
			patch_size_xyz=source.patch_size_xyz,
			volume_shape_xyz=local_shape,
		)
		predictions = project_token_grid_nearest(
			source.predictions[token_slice],
			patch_size_xyz=source.patch_size_xyz,
			volume_shape_xyz=local_shape,
		)
		probability_tokens = np.asarray(source.probabilities[token_slice])
		confidence = project_token_grid_nearest(
			probability_tokens.max(axis=-1),
			patch_size_xyz=source.patch_size_xyz,
			volume_shape_xyz=local_shape,
		)
		voxel_slice = slice(voxel_start, voxel_stop)
		arrays.valid_mask[voxel_slice] = valid
		arrays.predictions[voxel_slice] = np.where(
			valid, predictions, INVALID_PREDICTION_CLASS_ID
		).astype(np.int16, copy=False)
		arrays.confidence[voxel_slice] = np.where(
			valid, confidence, np.nan
		).astype(np.float16, copy=False)
		if arrays.probabilities is not None:
			projected_probabilities = project_token_grid_nearest(
				probability_tokens,
				patch_size_xyz=source.patch_size_xyz,
				volume_shape_xyz=local_shape,
			)
			arrays.probabilities[voxel_slice] = np.where(
				valid[..., None], projected_probabilities, np.nan
			).astype(np.float16, copy=False)


def _projection_metadata(
	source: _TokenPredictionArtifact,
	*,
	output_dir: Path,
	write_probabilities: bool,
	summary: Mapping[str, object],
) -> dict[str, object]:
	resolved_output_dir = output_dir.resolve(strict=False)
	output_files = {
		'predictions': str(resolved_output_dir / 'f3_voxel_predictions.npy'),
		'confidence': str(resolved_output_dir / 'f3_voxel_confidence.npy'),
		'valid_mask': str(resolved_output_dir / 'f3_valid_voxel_mask.npy'),
	}
	if write_probabilities:
		output_files['probabilities'] = str(
			resolved_output_dir / 'f3_voxel_probabilities.npy'
		)
	source_files = {
		'token_predictions': _identity(source.predictions_path),
		'token_probabilities': _identity(source.probabilities_path),
		'valid_token_grid': _identity(source.valid_tokens_path),
		'prediction_metadata': _identity(source.metadata_path),
	}
	return {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'prediction_kind': 'token_projection_nearest',
		'prediction_semantics': (
			'nearest repetition of token predictions; not a learned voxel prediction'
		),
		'model_tag': source.model_tag,
		'class_probability_order': list(source.class_probability_order),
		'classes': [dict(item) for item in source.classes],
		'volume_shape_xyz': list(source.volume_shape_xyz),
		'patch_size_xyz': list(source.patch_size_xyz),
		'invalid_prediction_class_id': INVALID_PREDICTION_CLASS_ID,
		'invalid_confidence_value': INVALID_CONFIDENCE_VALUE,
		'inputs': {
			'token_prediction_dir': str(source.root),
			**{name: value['path'] for name, value in source_files.items()},
		},
		'source_identity': {
			'token_artifact_files': source_files,
			'probe_spec': dict(_mapping(source.metadata.get('probe'), 'probe')),
			'embedding_identity': {
				'config': dict(
					_mapping(source.metadata.get('embeddings'), 'embeddings')
				),
				'artifact': dict(
					_mapping(source.metadata.get('embedding'), 'embedding')
				),
			},
			'model_identity': dict(_mapping(source.metadata.get('model'), 'model')),
		},
		'outputs': output_files,
		'summary': dict(summary),
	}


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'token prediction metadata {label} must be a mapping')
	return cast('Mapping[str, object]', value)


def _validate_token_metadata_output_binding(
	metadata: Mapping[str, object], *, paths: Mapping[str, Path]
) -> None:
	outputs = _mapping(metadata.get('outputs'), 'outputs')
	for metadata_key, path_key in (
		('token_predictions', 'predictions'),
		('probability_volume', 'probabilities'),
		('valid_token_grid', 'valid_tokens'),
		('metadata_json', 'metadata'),
	):
		value = outputs.get(metadata_key)
		if not isinstance(value, str) or not value:
			raise TypeError(
				f'token prediction metadata outputs.{metadata_key} '
				'must be a non-empty path string'
			)
		if Path(value).resolve(strict=False) != paths[path_key].resolve(strict=False):
			raise ValueError(
				f'token prediction metadata outputs.{metadata_key} does not '
				f'identify {paths[path_key]}'
			)


def _positive_triplet(value: object, label: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 3
	):
		raise TypeError(f'token prediction metadata {label} must be a positive triple')
	if any(
		not isinstance(item, int) or isinstance(item, bool) or item <= 0
		for item in value
	):
		raise ValueError(
			f'token prediction metadata {label} must be a positive integer triple'
		)
	return cast('tuple[int, int, int]', tuple(value))


def _class_order(value: object) -> tuple[int, ...]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or not value
	):
		raise TypeError(
			'token prediction metadata class_probability_order must be non-empty'
		)
	if any(
		not isinstance(item, int)
		or isinstance(item, bool)
		or item == INVALID_PREDICTION_CLASS_ID
		for item in value
	):
		raise ValueError('token prediction class order contains an invalid class ID')
	result = tuple(value)
	if len(set(result)) != len(result):
		raise ValueError('token prediction class order must not contain duplicates')
	return cast('tuple[int, ...]', result)


def _classes(
	value: object, *, class_order: tuple[int, ...]
) -> tuple[Mapping[str, object], ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError('token prediction metadata classes must be a sequence')
	if any(not isinstance(item, Mapping) for item in value):
		raise TypeError('token prediction metadata classes entries must be mappings')
	classes = cast('tuple[Mapping[str, object], ...]', tuple(value))
	class_ids = tuple(item.get('class_id') for item in classes)
	if class_ids != class_order:
		raise ValueError(
			'token prediction classes order must match class_probability_order'
		)
	return classes


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _reject_json_constant(value: str) -> None:
	raise ValueError(f'non-standard JSON constant: {value}')


__all__ = [
	'F3VoxelProjectionResult',
	'F3VoxelProjectionSourceInfo',
	'inspect_f3_lithology_token_projection_source',
	'project_f3_lithology_tokens_to_voxels',
]
