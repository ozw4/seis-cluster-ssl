"""Reconstruct cluster label maps from token grids."""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

XYZ = tuple[int, int, int]


@dataclass(frozen=True)
class ReconstructedLabels:
	"""Paths for one reconstructed cluster-label output."""

	survey_id: str
	token_labels_path: Path
	voxel_labels_path: Path | None
	token_shape_xyz: XYZ
	voxel_shape_xyz: XYZ | None
	skipped_existing_voxel_labels: bool = False


def reconstruct_voxel_labels(
	token_labels: np.ndarray,
	*,
	patch_size_xyz: Sequence[int],
	volume_shape_xyz: Sequence[int],
	output_path: str | Path | None = None,
) -> np.ndarray:
	"""Nearest-neighbor upsample token labels to the clipped voxel grid."""
	labels = _validate_token_labels(token_labels)
	patch = _validate_positive_xyz(patch_size_xyz, 'patch_size_xyz')
	volume_shape = _validate_positive_xyz(volume_shape_xyz, 'volume_shape_xyz')
	padded_shape = cast(
		'XYZ',
		tuple(
			token_axis * patch_axis
			for token_axis, patch_axis in zip(labels.shape, patch, strict=True)
		),
	)
	if any(
		volume_axis > padded_axis
		for volume_axis, padded_axis in zip(volume_shape, padded_shape, strict=True)
	):
		msg = (
			'volume_shape_xyz is larger than the token grid implied by '
			f'patch_size_xyz; got volume_shape_xyz={volume_shape!r}, '
			f'token_shape_xyz={labels.shape!r}, patch_size_xyz={patch!r}'
		)
		raise ValueError(msg)

	if output_path is None:
		voxels = np.empty(volume_shape, dtype=np.int32)
	else:
		path = Path(output_path)
		path.parent.mkdir(parents=True, exist_ok=True)
		voxels = np.lib.format.open_memmap(
			path,
			mode='w+',
			dtype=np.int32,
			shape=volume_shape,
		)
	token_limits = cast(
		'XYZ',
		tuple(
			_min_token_count(volume_axis, patch_axis)
			for volume_axis, patch_axis in zip(volume_shape, patch, strict=True)
		),
	)
	for token_x in range(token_limits[0]):
		x_start = token_x * patch[0]
		x_stop = min(x_start + patch[0], volume_shape[0])
		for token_y in range(token_limits[1]):
			y_start = token_y * patch[1]
			y_stop = min(y_start + patch[1], volume_shape[1])
			for token_z in range(token_limits[2]):
				z_start = token_z * patch[2]
				z_stop = min(z_start + patch[2], volume_shape[2])
				voxels[x_start:x_stop, y_start:y_stop, z_start:z_stop] = np.int32(
					labels[token_x, token_y, token_z],
				)
	if hasattr(voxels, 'flush'):
		voxels.flush()
	return voxels


def reconstruct_labels_for_survey(
	token_labels_path: str | Path,
	*,
	metadata_path: str | Path | None = None,
	output_dir: str | Path | None = None,
	write_voxel_labels: bool = True,
	skip_existing_voxel_labels: bool = False,
) -> ReconstructedLabels:
	"""Load one token-label artifact and optionally write voxel labels."""
	token_path = Path(token_labels_path)
	survey_id = token_path.name.removesuffix('.cluster_labels_token.npy')
	labels = _validate_token_labels(np.load(token_path, mmap_mode='r'))
	metadata = _load_metadata(metadata_path)
	patch = _metadata_xyz(metadata, 'patch_size', default=(1, 1, 1))
	volume_shape = resolve_volume_shape_xyz(metadata, labels.shape, patch)
	voxel_path = None
	skipped_existing = False
	if write_voxel_labels:
		root = token_path.parent if output_dir is None else Path(output_dir)
		voxel_path = root / f'{survey_id}.cluster_labels_voxel.npy'
		if skip_existing_voxel_labels and voxel_path.is_file():
			_validate_existing_voxel_labels(
				voxel_path,
				volume_shape_xyz=volume_shape,
			)
			skipped_existing = True
		else:
			reconstruct_voxel_labels(
				labels,
				patch_size_xyz=patch,
				volume_shape_xyz=volume_shape,
				output_path=voxel_path,
			)
	return ReconstructedLabels(
		survey_id=survey_id,
		token_labels_path=token_path,
		voxel_labels_path=voxel_path,
		token_shape_xyz=cast('XYZ', tuple(int(axis) for axis in labels.shape)),
		voxel_shape_xyz=volume_shape if write_voxel_labels else None,
		skipped_existing_voxel_labels=skipped_existing,
	)


def resolve_volume_shape_xyz(
	metadata: Mapping[str, object],
	token_shape_xyz: Sequence[int],
	patch_size_xyz: Sequence[int],
) -> XYZ:
	"""Resolve original voxel shape from metadata or source volume inspection."""
	for key in ('volume_shape_xyz', 'volume_shape', 'shape_xyz'):
		if key in metadata:
			return _validate_positive_xyz(
				cast('Sequence[int]', metadata[key]),
				key,
			)
	source_path = metadata.get('source_amplitude_path')
	if isinstance(source_path, str) and source_path:
		path = Path(source_path)
		if path.is_file():
			array = np.load(path, mmap_mode='r')
			if array.ndim == 3:
				return cast('XYZ', tuple(int(axis) for axis in array.shape))
	patch = _validate_positive_xyz(patch_size_xyz, 'patch_size_xyz')
	token_shape = _validate_positive_xyz(token_shape_xyz, 'token_shape_xyz')
	return cast(
		'XYZ',
		tuple(
			token_axis * patch_axis
			for token_axis, patch_axis in zip(token_shape, patch, strict=True)
		),
	)


def _load_metadata(path: str | Path | None) -> dict[str, object]:
	if path is None:
		return {}
	metadata_path = Path(path)
	if not metadata_path.is_file():
		return {}
	payload = json.loads(metadata_path.read_text(encoding='utf-8'))
	if not isinstance(payload, dict):
		msg = f'label metadata must be a JSON object: {metadata_path}'
		raise TypeError(msg)
	embedding_input = payload.get('embedding_input')
	if isinstance(embedding_input, dict):
		nested_path = embedding_input.get('metadata_path')
		if isinstance(nested_path, str) and Path(nested_path).is_file():
			nested = json.loads(Path(nested_path).read_text(encoding='utf-8'))
			if isinstance(nested, dict):
				return {**nested, **payload}
	return payload


def _metadata_xyz(
	metadata: Mapping[str, object],
	key: str,
	*,
	default: XYZ,
) -> XYZ:
	value = metadata.get(key, default)
	return _validate_positive_xyz(cast('Sequence[int]', value), key)


def _validate_token_labels(labels: np.ndarray) -> np.ndarray:
	array = np.asarray(labels)
	if array.ndim != 3:
		msg = f'token labels must be 3D; got shape={array.shape!r}'
		raise ValueError(msg)
	if array.dtype.kind not in {'i', 'u'}:
		msg = f'token labels must use an integer dtype; got {array.dtype}'
		raise TypeError(msg)
	return array


def _validate_existing_voxel_labels(
	path: Path,
	*,
	volume_shape_xyz: XYZ,
) -> None:
	array = np.load(path, mmap_mode='r')
	if array.shape != volume_shape_xyz or array.dtype != np.dtype(np.int32):
		msg = (
			'incompatible existing voxel labels; expected '
			f'shape={volume_shape_xyz!r} and dtype=int32, got '
			f'shape={array.shape!r} and dtype={array.dtype} at {path}'
		)
		raise ValueError(msg)


def _validate_positive_xyz(value: Sequence[int], name: str) -> XYZ:
	if (
		isinstance(value, str)
		or len(value) != 3
		or not all(
			not isinstance(axis, bool) and isinstance(axis, Integral)
			for axis in value
		)
	):
		msg = f'{name} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = cast('XYZ', tuple(int(axis) for axis in value))
	if any(axis <= 0 for axis in xyz):
		msg = f'{name} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _min_token_count(volume_axis: int, patch_axis: int) -> int:
	return (volume_axis + patch_axis - 1) // patch_axis


__all__ = [
	'ReconstructedLabels',
	'reconstruct_labels_for_survey',
	'reconstruct_voxel_labels',
	'resolve_volume_shape_xyz',
]
