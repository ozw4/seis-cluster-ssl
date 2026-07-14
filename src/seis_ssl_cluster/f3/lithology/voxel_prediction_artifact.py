"""Shared on-disk contract for F3 voxel lithology predictions."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.models.voxel_decoder.spec import (
	validate_voxel_decoder_architecture_mapping,
)

if TYPE_CHECKING:
	from numpy.typing import NDArray

ARTIFACT_TYPE = 'f3_lithology_voxel_predictions'
SCHEMA_VERSION = 1
PREDICTION_KINDS = frozenset(
	{'token_projection_nearest', 'frozen_embedding_decoder'}
)
PREDICTIONS_NAME = 'f3_voxel_predictions.npy'
CONFIDENCE_NAME = 'f3_voxel_confidence.npy'
VALID_MASK_NAME = 'f3_valid_voxel_mask.npy'
PROBABILITIES_NAME = 'f3_voxel_probabilities.npy'
METADATA_NAME = 'prediction_metadata.json'
INVALID_PREDICTION_CLASS_ID = -1
INVALID_CONFIDENCE_VALUE = 'nan'
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2


@dataclass(frozen=True)
class F3VoxelPredictionArtifactPaths:
	"""Deterministic paths belonging to one voxel-prediction run."""

	output_dir: Path
	predictions: Path
	confidence: Path
	valid_mask: Path
	probabilities: Path
	metadata: Path


@dataclass(frozen=True)
class F3VoxelPredictionArrays:
	"""Writable or loaded arrays in a voxel-prediction artifact."""

	predictions: NDArray[np.int16]
	confidence: NDArray[np.float16]
	valid_mask: NDArray[np.bool_]
	probabilities: NDArray[np.float16] | None


@dataclass(frozen=True)
class F3VoxelPredictionArtifact:
	"""A completely validated voxel-prediction artifact."""

	paths: F3VoxelPredictionArtifactPaths
	metadata: Mapping[str, object]
	arrays: F3VoxelPredictionArrays


def f3_voxel_prediction_artifact_paths(
	output_dir: str | Path,
) -> F3VoxelPredictionArtifactPaths:
	"""Return the fixed artifact paths below ``output_dir``."""
	root = Path(output_dir)
	return F3VoxelPredictionArtifactPaths(
		output_dir=root,
		predictions=root / PREDICTIONS_NAME,
		confidence=root / CONFIDENCE_NAME,
		valid_mask=root / VALID_MASK_NAME,
		probabilities=root / PROBABILITIES_NAME,
		metadata=root / METADATA_NAME,
	)


def discover_f3_voxel_probability_path(
	paths_or_dir: F3VoxelPredictionArtifactPaths | str | Path,
) -> Path | None:
	"""Return the optional probability path when it exists."""
	paths = _coerce_paths(paths_or_dir)
	return paths.probabilities if paths.probabilities.is_file() else None


def create_f3_voxel_prediction_staging_paths(
	output_dir: str | Path,
	*,
	overwrite: bool = False,
) -> F3VoxelPredictionArtifactPaths:
	"""Create a same-filesystem staging directory for an artifact run."""
	target = Path(output_dir)
	if target.exists() and not overwrite:
		raise FileExistsError(f'refusing to overwrite existing output: {target}')
	target.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(
		tempfile.mkdtemp(prefix=f'.{target.name}.staging-', dir=target.parent)
	)
	return f3_voxel_prediction_artifact_paths(staging)


def open_f3_voxel_prediction_memmaps(
	paths_or_dir: F3VoxelPredictionArtifactPaths | str | Path,
	*,
	volume_shape_xyz: Sequence[int],
	class_count: int,
	include_probabilities: bool = False,
) -> F3VoxelPredictionArrays:
	"""Create initialized ``.npy`` memmaps for a producer to fill in chunks."""
	paths = _coerce_paths(paths_or_dir)
	shape = _positive_triplet(volume_shape_xyz, 'volume_shape_xyz')
	if (
		not isinstance(class_count, int)
		or isinstance(class_count, bool)
		or class_count <= 0
	):
		raise ValueError(f'class_count must be a positive integer; got {class_count!r}')
	paths.output_dir.mkdir(parents=True, exist_ok=True)
	for path in (paths.predictions, paths.confidence, paths.valid_mask):
		if path.exists():
			raise FileExistsError(f'refusing to overwrite existing output: {path}')
	if paths.probabilities.exists():
		raise FileExistsError(
			f'refusing to overwrite existing output: {paths.probabilities}'
		)
	predictions = np.lib.format.open_memmap(
		paths.predictions, mode='w+', dtype=np.int16, shape=shape
	)
	confidence = np.lib.format.open_memmap(
		paths.confidence, mode='w+', dtype=np.float16, shape=shape
	)
	valid_mask = np.lib.format.open_memmap(
		paths.valid_mask, mode='w+', dtype=np.bool_, shape=shape
	)
	probabilities = (
		np.lib.format.open_memmap(
			paths.probabilities,
			mode='w+',
			dtype=np.float16,
			shape=(*shape, class_count),
		)
		if include_probabilities
		else None
	)
	predictions[...] = INVALID_PREDICTION_CLASS_ID
	confidence[...] = np.nan
	valid_mask[...] = False
	if probabilities is not None:
		probabilities[...] = np.nan
	return F3VoxelPredictionArrays(
		predictions=predictions,
		confidence=confidence,
		valid_mask=valid_mask,
		probabilities=probabilities,
	)


def write_f3_voxel_prediction_metadata(
	path: str | Path, metadata: Mapping[str, object]
) -> None:
	"""Atomically write metadata as strict standard JSON."""
	metadata_path = Path(path)
	metadata_path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = metadata_path.with_name(f'.{metadata_path.name}.tmp')
	try:
		tmp_path.write_text(
			json.dumps(
				metadata, allow_nan=False, indent=2, sort_keys=True
			)
			+ '\n',
			encoding='utf-8',
		)
		tmp_path.replace(metadata_path)
	except BaseException:
		tmp_path.unlink(missing_ok=True)
		raise


def read_f3_voxel_prediction_metadata(
	path: str | Path,
) -> Mapping[str, object]:
	"""Read a voxel-prediction metadata standard-JSON object."""
	with Path(path).open(encoding='utf-8') as file_obj:
		payload = json.load(file_obj, parse_constant=_reject_json_constant)
	if not isinstance(payload, Mapping):
		raise TypeError('voxel prediction metadata must contain a JSON object')
	return cast('Mapping[str, object]', payload)


def validate_f3_voxel_prediction_arrays(  # noqa: C901, PLR0912
	arrays: F3VoxelPredictionArrays,
	*,
	volume_shape_xyz: Sequence[int],
	class_probability_order: Sequence[int],
	chunk_voxels: int = 1_000_000,
) -> dict[str, object]:
	"""Validate array dtypes and values without copying whole memmaps to RAM."""
	shape = _positive_triplet(volume_shape_xyz, 'volume_shape_xyz')
	class_ids = _class_id_order(class_probability_order)
	if (
		not isinstance(chunk_voxels, int)
		or isinstance(chunk_voxels, bool)
		or chunk_voxels <= 0
	):
		raise ValueError(f'chunk_voxels must be positive; got {chunk_voxels!r}')
	_validate_array_layout(arrays, shape=shape, class_count=len(class_ids))

	flat_predictions = arrays.predictions.reshape(-1)
	flat_confidence = arrays.confidence.reshape(-1)
	flat_mask = arrays.valid_mask.reshape(-1)
	flat_probabilities = (
		None
		if arrays.probabilities is None
		else arrays.probabilities.reshape((-1, len(class_ids)))
	)
	class_counts = dict.fromkeys(class_ids, 0)
	valid_count = 0
	for start in range(0, flat_mask.size, chunk_voxels):
		stop = min(start + chunk_voxels, flat_mask.size)
		mask = flat_mask[start:stop]
		predictions = flat_predictions[start:stop]
		confidence = flat_confidence[start:stop]
		valid = np.flatnonzero(mask)
		invalid = np.flatnonzero(~mask)
		if invalid.size and not np.all(
			predictions[invalid] == INVALID_PREDICTION_CLASS_ID
		):
			raise ValueError('invalid voxels must use prediction class ID -1')
		if invalid.size and not np.all(np.isnan(confidence[invalid])):
			raise ValueError('invalid voxels must use NaN confidence')
		if valid.size:
			valid_predictions = predictions[valid]
			if not np.all(np.isin(valid_predictions, class_ids)):
				raise ValueError('valid voxels contain an unknown prediction class ID')
			valid_confidence = confidence[valid]
			if not np.all(np.isfinite(valid_confidence)) or not np.all(
				(valid_confidence >= 0.0) & (valid_confidence <= 1.0)
			):
				raise ValueError('valid voxel confidence must be finite and in [0, 1]')
			for class_id in class_ids:
				class_counts[class_id] += int(
					np.count_nonzero(valid_predictions == class_id)
				)
		valid_count += int(valid.size)
		if flat_probabilities is not None:
			probabilities = flat_probabilities[start:stop]
			if invalid.size and not np.all(np.isnan(probabilities[invalid])):
				raise ValueError('invalid voxel probability rows must contain only NaN')
			if valid.size:
				valid_probabilities = probabilities[valid]
				if not np.all(np.isfinite(valid_probabilities)) or np.any(
					valid_probabilities < 0.0
				):
					raise ValueError(
						'valid voxel probabilities must be finite and non-negative'
					)
				if not np.allclose(
					valid_probabilities.sum(axis=1), 1.0, rtol=1e-3, atol=1e-3
				):
					raise ValueError('valid voxel probability rows must sum to 1')
				if not np.allclose(
					valid_confidence,
					valid_probabilities.max(axis=1),
					rtol=1e-3,
					atol=1e-3,
				):
					raise ValueError(
						'confidence must equal the maximum class probability'
					)
	invalid_count = int(flat_mask.size - valid_count)
	return {
		'valid_voxel_count': valid_count,
		'invalid_voxel_count': invalid_count,
		'class_prediction_counts': {
			str(class_id): class_counts[class_id] for class_id in class_ids
		},
	}


def validate_f3_voxel_prediction_artifact(
	paths_or_dir: F3VoxelPredictionArtifactPaths | str | Path,
	*,
	mmap_mode: str | None = 'r',
) -> F3VoxelPredictionArtifact:
	"""Load and validate one complete artifact, rejecting partial output."""
	paths = _coerce_paths(paths_or_dir)
	return _validate_f3_voxel_prediction_artifact(
		paths,
		metadata_paths=paths,
		mmap_mode=mmap_mode,
	)


def _validate_f3_voxel_prediction_artifact(
	paths: F3VoxelPredictionArtifactPaths,
	*,
	metadata_paths: F3VoxelPredictionArtifactPaths,
	mmap_mode: str | None,
) -> F3VoxelPredictionArtifact:
	required = (paths.predictions, paths.confidence, paths.valid_mask, paths.metadata)
	missing = [path.name for path in required if not path.is_file()]
	if missing:
		raise FileNotFoundError(
			f'incomplete voxel prediction artifact; missing: {", ".join(missing)}'
		)
	metadata = read_f3_voxel_prediction_metadata(paths.metadata)
	shape, class_ids = _validate_metadata(metadata)
	_validate_metadata_output_binding(
		metadata,
		paths=metadata_paths,
		probabilities_present=paths.probabilities.is_file(),
	)
	arrays = F3VoxelPredictionArrays(
		predictions=np.load(paths.predictions, mmap_mode=mmap_mode, allow_pickle=False),
		confidence=np.load(paths.confidence, mmap_mode=mmap_mode, allow_pickle=False),
		valid_mask=np.load(paths.valid_mask, mmap_mode=mmap_mode, allow_pickle=False),
		probabilities=(
			None
			if discover_f3_voxel_probability_path(paths) is None
			else np.load(paths.probabilities, mmap_mode=mmap_mode, allow_pickle=False)
		),
	)
	summary = validate_f3_voxel_prediction_arrays(
		arrays,
		volume_shape_xyz=shape,
		class_probability_order=class_ids,
	)
	metadata_summary = cast('Mapping[str, object]', metadata['summary'])
	if metadata_summary != summary:
		raise ValueError(
			'artifact summary does not match voxel prediction arrays; '
			f'metadata={metadata_summary!r}, arrays={summary!r}'
		)
	return F3VoxelPredictionArtifact(paths=paths, metadata=metadata, arrays=arrays)


def commit_f3_voxel_prediction_artifact(
	staging_paths_or_dir: F3VoxelPredictionArtifactPaths | str | Path,
	output_dir: str | Path,
	*,
	overwrite: bool = False,
) -> F3VoxelPredictionArtifactPaths:
	"""Validate and transactionally publish a same-filesystem staging directory.

	An atomic directory exchange is used for overwrite when the platform and
	filesystem support it. Otherwise, the previous artifact is kept at a unique
	backup path until promotion of the validated staging directory succeeds.
	"""
	staging = _coerce_paths(staging_paths_or_dir)
	target = Path(output_dir)
	if staging.output_dir.resolve() == target.resolve():
		raise ValueError('staging and output directories must be different')
	if staging.output_dir.parent.resolve() != target.parent.resolve():
		raise ValueError('staging and output directories must have the same parent')
	_validate_f3_voxel_prediction_artifact(
		staging,
		metadata_paths=f3_voxel_prediction_artifact_paths(target),
		mmap_mode='r',
	)
	if target.exists() and not overwrite:
		raise FileExistsError(f'refusing to overwrite existing output: {target}')
	if not target.exists():
		_rename_directory_no_replace(staging.output_dir, target)
		return f3_voxel_prediction_artifact_paths(target)
	try:
		_exchange_directories(staging.output_dir, target)
	except NotImplementedError:
		_replace_directory_portably(staging.output_dir, target)
	else:
		shutil.rmtree(staging.output_dir)
	return f3_voxel_prediction_artifact_paths(target)


def _replace_directory_portably(source: Path, target: Path) -> None:
	"""Replace ``target`` while retaining it until ``source`` is promoted."""
	backup = Path(
		tempfile.mkdtemp(prefix=f'.{target.name}.backup-', dir=target.parent)
	)
	try:
		target.rename(backup)
	except Exception:
		backup.rmdir()
		raise
	try:
		_promote_staging_directory(source, target)
	except Exception as promotion_error:
		try:
			backup.rename(target)
		except Exception as rollback_error:
			raise RuntimeError(
				f'failed to promote staging directory {source} to {target}: '
				f'{promotion_error}; rollback also failed; the previous artifact '
				f'remains at {backup}'
			) from rollback_error
		raise
	shutil.rmtree(backup)


def _promote_staging_directory(source: Path, target: Path) -> None:
	"""Rename validated staging into the now-vacant target path."""
	source.rename(target)


def _rename_directory_no_replace(source: Path, target: Path) -> None:
	"""Atomically publish a directory only when the target is still absent."""
	try:
		_rename_directories(
			source,
			target,
			flags=_RENAME_NOREPLACE,
			operation='no-replace publish',
			flag_name='RENAME_NOREPLACE',
		)
	except OSError as error:
		if error.errno == errno.EEXIST:
			raise FileExistsError(
				f'refusing to overwrite existing output: {target}'
			) from error
		raise


def _exchange_directories(source: Path, target: Path) -> None:
	"""Atomically exchange two same-filesystem directory entries."""
	_rename_directories(
		source,
		target,
		flags=_RENAME_EXCHANGE,
		operation='overwrite',
		flag_name='RENAME_EXCHANGE',
	)


def _rename_directories(
	source: Path,
	target: Path,
	*,
	flags: int,
	operation: str,
	flag_name: str,
) -> None:
	"""Invoke Linux renameat2 with the requested atomic directory semantics."""
	libc = ctypes.CDLL(None, use_errno=True)
	try:
		renameat2 = libc.renameat2
	except AttributeError as error:
		raise NotImplementedError(
			f'atomic directory {operation} requires renameat2({flag_name})'
		) from error
	renameat2.argtypes = (
		ctypes.c_int,
		ctypes.c_char_p,
		ctypes.c_int,
		ctypes.c_char_p,
		ctypes.c_uint,
	)
	renameat2.restype = ctypes.c_int
	result = renameat2(
		_AT_FDCWD,
		os.fsencode(source),
		_AT_FDCWD,
		os.fsencode(target),
		flags,
	)
	if result != 0:
		error_number = ctypes.get_errno()
		if error_number in {
			errno.EINVAL,
			errno.ENOSYS,
			errno.EOPNOTSUPP,
		}:
			raise NotImplementedError(
				f'atomic directory {operation} is not supported by this platform '
				'or filesystem'
			) from OSError(error_number, os.strerror(error_number))
		raise OSError(
			error_number,
			os.strerror(error_number),
			f'{source} <-> {target}',
		)


def _coerce_paths(
	value: F3VoxelPredictionArtifactPaths | str | Path,
) -> F3VoxelPredictionArtifactPaths:
	if isinstance(value, F3VoxelPredictionArtifactPaths):
		return value
	return f3_voxel_prediction_artifact_paths(value)


def _reject_json_constant(value: str) -> None:
	raise ValueError(f'non-standard JSON constant: {value}')


def _positive_triplet(value: Sequence[int], label: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 3
	):
		raise TypeError(f'{label} must be a positive integer triple')
	if any(
		not isinstance(item, int | np.integer)
		or isinstance(item, bool | np.bool_)
		or int(item) <= 0
		for item in value
	):
		raise ValueError(f'{label} must be a positive integer triple')
	return cast('tuple[int, int, int]', tuple(int(item) for item in value))


def _class_id_order(value: Sequence[int]) -> tuple[int, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
		raise TypeError('class_probability_order must be a non-empty integer sequence')
	if any(
		not isinstance(item, int | np.integer)
		or isinstance(item, bool | np.bool_)
		or int(item) == INVALID_PREDICTION_CLASS_ID
		for item in value
	):
		raise ValueError(
			'class_probability_order must contain integer class IDs other than -1'
		)
	result = tuple(int(item) for item in value)
	if len(set(result)) != len(result):
		raise ValueError('class_probability_order must not contain duplicates')
	return result


def _validate_array_layout(
	arrays: F3VoxelPredictionArrays,
	*,
	shape: tuple[int, int, int],
	class_count: int,
) -> None:
	expected = (
		('predictions', arrays.predictions, np.dtype(np.int16), shape),
		('confidence', arrays.confidence, np.dtype(np.float16), shape),
		('valid_mask', arrays.valid_mask, np.dtype(np.bool_), shape),
	)
	for label, array, dtype, expected_shape in expected:
		if array.dtype != dtype:
			raise TypeError(f'{label} dtype must be {dtype}; got {array.dtype}')
		if tuple(array.shape) != expected_shape:
			raise ValueError(
				f'{label} shape must be {expected_shape!r}; got {array.shape!r}'
			)
	if arrays.probabilities is not None:
		if arrays.probabilities.dtype != np.dtype(np.float16):
			raise TypeError(
				'probabilities dtype must be float16; '
				f'got {arrays.probabilities.dtype}'
			)
		expected_probability_shape = (*shape, class_count)
		if tuple(arrays.probabilities.shape) != expected_probability_shape:
			raise ValueError(
				'probabilities shape must be '
				f'{expected_probability_shape!r}; got {arrays.probabilities.shape!r}'
			)


def _validate_metadata(  # noqa: C901, PLR0912
	metadata: Mapping[str, object],
) -> tuple[tuple[int, int, int], tuple[int, ...]]:
	required = {
		'artifact_type',
		'schema_version',
		'prediction_kind',
		'model_tag',
		'class_probability_order',
		'classes',
		'volume_shape_xyz',
		'patch_size_xyz',
		'invalid_prediction_class_id',
		'invalid_confidence_value',
		'inputs',
		'source_identity',
		'outputs',
		'summary',
	}
	missing = sorted(required - metadata.keys())
	if missing:
		raise ValueError(f'voxel prediction metadata is missing keys: {missing!r}')
	if metadata['artifact_type'] != ARTIFACT_TYPE:
		raise ValueError(f'artifact_type must be {ARTIFACT_TYPE!r}')
	if metadata['schema_version'] != SCHEMA_VERSION:
		raise ValueError(f'schema_version must be {SCHEMA_VERSION}')
	if metadata['prediction_kind'] not in PREDICTION_KINDS:
		raise ValueError(
			f'unsupported prediction_kind: {metadata["prediction_kind"]!r}'
		)
	if metadata['prediction_kind'] == 'frozen_embedding_decoder':
		if 'decoder_architecture' not in metadata:
			raise ValueError(
				'voxel prediction metadata is missing keys: '
				"['decoder_architecture']"
			)
		validate_voxel_decoder_architecture_mapping(
			metadata['decoder_architecture'],
			field_prefix='decoder_architecture',
		)
	if not isinstance(metadata['model_tag'], str) or not metadata['model_tag']:
		raise TypeError('model_tag must be a non-empty string')
	shape_value = metadata['volume_shape_xyz']
	patch_value = metadata['patch_size_xyz']
	order_value = metadata['class_probability_order']
	if not isinstance(shape_value, Sequence):
		raise TypeError('volume_shape_xyz must be a positive integer triple')
	if not isinstance(patch_value, Sequence):
		raise TypeError('patch_size_xyz must be a positive integer triple')
	if not isinstance(order_value, Sequence):
		raise TypeError('class_probability_order must be an integer sequence')
	shape = _positive_triplet(shape_value, 'volume_shape_xyz')
	_positive_triplet(patch_value, 'patch_size_xyz')
	class_ids = _class_id_order(order_value)
	if metadata['invalid_prediction_class_id'] != INVALID_PREDICTION_CLASS_ID:
		raise ValueError('invalid_prediction_class_id must be -1')
	if metadata['invalid_confidence_value'] != INVALID_CONFIDENCE_VALUE:
		raise ValueError("invalid_confidence_value must be the string 'nan'")
	for label in ('inputs', 'source_identity', 'outputs', 'summary'):
		if not isinstance(metadata[label], Mapping):
			raise TypeError(f'{label} must be a mapping')
	classes = metadata['classes']
	if not isinstance(classes, Sequence) or isinstance(classes, str | bytes):
		raise TypeError('classes must be a sequence')
	try:
		metadata_class_ids = tuple(_metadata_class_id(item) for item in classes)
	except (KeyError, TypeError, ValueError) as error:
		msg = 'every classes entry must contain an integer class_id'
		raise TypeError(msg) from error
	if metadata_class_ids != class_ids:
		raise ValueError(
			'classes order must match class_probability_order; '
			f'classes={metadata_class_ids!r}, order={class_ids!r}'
		)
	return shape, class_ids


def _validate_metadata_output_binding(
	metadata: Mapping[str, object],
	*,
	paths: F3VoxelPredictionArtifactPaths,
	probabilities_present: bool,
) -> None:
	outputs = cast('Mapping[str, object]', metadata['outputs'])
	expected = {
		'predictions': paths.predictions,
		'confidence': paths.confidence,
		'valid_mask': paths.valid_mask,
	}
	for key, expected_path in expected.items():
		_validate_metadata_output_path(
			outputs,
			key=key,
			expected_path=expected_path,
			output_dir=paths.output_dir,
		)
	probabilities_declared = 'probabilities' in outputs
	if probabilities_declared and not probabilities_present:
		raise FileNotFoundError(
			'incomplete voxel prediction artifact; metadata declares '
			f'{PROBABILITIES_NAME}, but the file is missing'
		)
	if probabilities_present and not probabilities_declared:
		raise ValueError(
			'incomplete voxel prediction metadata; outputs.probabilities is '
			f'required when {PROBABILITIES_NAME} exists'
		)
	if probabilities_declared:
		_validate_metadata_output_path(
			outputs,
			key='probabilities',
			expected_path=paths.probabilities,
			output_dir=paths.output_dir,
		)


def _validate_metadata_output_path(
	outputs: Mapping[str, object],
	*,
	key: str,
	expected_path: Path,
	output_dir: Path,
) -> None:
	value = outputs.get(key)
	if not isinstance(value, str) or not value:
		raise TypeError(
			f'voxel prediction metadata outputs.{key} must be a non-empty path string'
		)
	declared_path = Path(value)
	if not declared_path.is_absolute():
		declared_path = output_dir / declared_path
	if declared_path.resolve(strict=False) != expected_path.resolve(strict=False):
		raise ValueError(
			f'voxel prediction metadata outputs.{key} does not identify '
			f'{expected_path}'
		)


def _metadata_class_id(value: object) -> int:
	if not isinstance(value, Mapping):
		raise TypeError
	class_id = value['class_id']
	if not isinstance(class_id, int) or isinstance(class_id, bool):
		raise TypeError
	return class_id


__all__ = [
	'ARTIFACT_TYPE',
	'CONFIDENCE_NAME',
	'INVALID_CONFIDENCE_VALUE',
	'INVALID_PREDICTION_CLASS_ID',
	'METADATA_NAME',
	'PREDICTIONS_NAME',
	'PREDICTION_KINDS',
	'PROBABILITIES_NAME',
	'SCHEMA_VERSION',
	'VALID_MASK_NAME',
	'F3VoxelPredictionArrays',
	'F3VoxelPredictionArtifact',
	'F3VoxelPredictionArtifactPaths',
	'commit_f3_voxel_prediction_artifact',
	'create_f3_voxel_prediction_staging_paths',
	'discover_f3_voxel_probability_path',
	'f3_voxel_prediction_artifact_paths',
	'open_f3_voxel_prediction_memmaps',
	'read_f3_voxel_prediction_metadata',
	'validate_f3_voxel_prediction_arrays',
	'validate_f3_voxel_prediction_artifact',
	'write_f3_voxel_prediction_metadata',
]
