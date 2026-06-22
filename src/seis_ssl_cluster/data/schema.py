"""Typed data contracts for amplitude-only survey manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np

GRID_ORDER_XYZ: tuple[str, str, str] = ('x', 'y', 'z')
TensorLike: TypeAlias = object


@dataclass(frozen=True)
class AmplitudeVolumeRecord:
	"""Manifest record for one amplitude volume."""

	survey_id: str
	path: Path
	shape_xyz: tuple[int, int, int]
	dtype: str
	grid_order: tuple[str, str, str]
	normalization_stats_path: Path

	def validate(self) -> None:
		"""Validate amplitude volume metadata."""
		if not self.survey_id:
			msg = 'amplitude.survey_id must be non-empty'
			raise ValueError(msg)
		if self.path.suffix != '.npy':
			msg = f'amplitude.path must point to a .npy file: {self.path}'
			raise ValueError(msg)
		_validate_shape_xyz(self.shape_xyz, 'amplitude.shape_xyz')
		_validate_grid_order(self.grid_order, 'amplitude.grid_order')
		_validate_numeric_dtype(self.dtype, 'amplitude.dtype')
		if self.normalization_stats_path.suffix != '.json':
			msg = (
				'amplitude.normalization_stats_path must point to a .json file: '
				f'{self.normalization_stats_path}'
			)
			raise ValueError(msg)
		if not self.normalization_stats_path.is_absolute():
			msg = (
				'amplitude.normalization_stats_path must be an absolute '
				'artifact-registry path; got '
				f'{self.normalization_stats_path}'
			)
			raise ValueError(msg)


@dataclass(frozen=True)
class SurveyManifest:
	"""Amplitude-only manifest for one survey."""

	survey_id: str
	root: Path
	amplitude: AmplitudeVolumeRecord

	def validate(self) -> None:
		"""Validate manifest and nested amplitude metadata."""
		if not self.survey_id:
			msg = 'survey_id must be non-empty'
			raise ValueError(msg)
		self.amplitude.validate()
		if self.amplitude.survey_id != self.survey_id:
			msg = (
				f'amplitude survey_id {self.amplitude.survey_id!r} does not match '
				f'manifest survey_id {self.survey_id!r}'
			)
			raise ValueError(msg)


@dataclass(frozen=True)
class CropRequest:
	"""Spatial request for a local crop."""

	survey_id: str
	start_xyz: tuple[int, int, int]
	size_xyz: tuple[int, int, int]


def survey_manifest_to_dict(manifest: SurveyManifest) -> dict[str, object]:
	"""Convert a survey manifest to a JSON-compatible dictionary."""
	manifest.validate()
	return {
		'survey_id': manifest.survey_id,
		'root': str(manifest.root),
		'amplitude': _amplitude_record_to_dict(manifest.amplitude),
	}


def survey_manifest_from_dict(data: Mapping[str, object]) -> SurveyManifest:
	"""Build a survey manifest from a dictionary loaded from JSON."""
	manifest = SurveyManifest(
		survey_id=_require_str(data, 'survey_id'),
		root=Path(_require_str(data, 'root')),
		amplitude=_amplitude_record_from_dict(
			_require_nested_mapping(data.get('amplitude'), 'amplitude'),
		),
	)
	manifest.validate()
	return manifest


def write_manifest_json(manifests: Sequence[SurveyManifest], path: Path) -> None:
	"""Write survey manifests to a deterministic JSON file."""
	payload = [survey_manifest_to_dict(manifest) for manifest in manifests]
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def read_manifest_json(path: Path) -> list[SurveyManifest]:
	"""Read survey manifests from a JSON file."""
	data = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(data, list):
		msg = f'manifest JSON must contain a list; got {type(data).__name__}'
		raise TypeError(msg)
	return [
		survey_manifest_from_dict(_require_nested_mapping(item, 'manifest'))
		for item in data
	]


def _amplitude_record_to_dict(record: AmplitudeVolumeRecord) -> dict[str, object]:
	return {
		'survey_id': record.survey_id,
		'path': str(record.path),
		'shape_xyz': list(record.shape_xyz),
		'dtype': record.dtype,
		'grid_order': list(record.grid_order),
		'normalization_stats_path': str(record.normalization_stats_path),
	}


def _amplitude_record_from_dict(
	data: Mapping[str, object],
) -> AmplitudeVolumeRecord:
	record = AmplitudeVolumeRecord(
		survey_id=_require_str(data, 'survey_id'),
		path=Path(_require_str(data, 'path')),
		shape_xyz=_require_int_tuple3(data, 'shape_xyz'),
		dtype=_require_str(data, 'dtype'),
		grid_order=_require_str_tuple3(data, 'grid_order'),
		normalization_stats_path=Path(_require_str(data, 'normalization_stats_path')),
	)
	record.validate()
	return record


def _require_nested_mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		msg = f'{label!r} must be a mapping; got {type(value).__name__}'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _require_str(data: Mapping[str, object], key: str) -> str:
	value = data[key]
	if not isinstance(value, str):
		msg = f'{key!r} must be a string; got {type(value).__name__}'
		raise TypeError(msg)
	return value


def _require_int_tuple3(
	data: Mapping[str, object],
	key: str,
) -> tuple[int, int, int]:
	value = data[key]
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str)
		or len(value) != 3
		or not all(isinstance(item, int) for item in value)
	):
		msg = f'{key!r} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = cast('tuple[int, int, int]', tuple(value))
	_validate_shape_xyz(xyz, key)
	return xyz


def _require_str_tuple3(
	data: Mapping[str, object],
	key: str,
) -> tuple[str, str, str]:
	value = data[key]
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str)
		or len(value) != 3
		or not all(isinstance(item, str) for item in value)
	):
		msg = f'{key!r} must be a length-3 string sequence; got {value!r}'
		raise TypeError(msg)
	grid_order = cast('tuple[str, str, str]', tuple(value))
	_validate_grid_order(grid_order, key)
	return grid_order


def _validate_shape_xyz(shape_xyz: tuple[int, int, int], label: str) -> None:
	if any(axis <= 0 for axis in shape_xyz):
		msg = f'{label} values must be positive; got {shape_xyz!r}'
		raise ValueError(msg)


def _validate_grid_order(grid_order: tuple[str, str, str], label: str) -> None:
	if grid_order != GRID_ORDER_XYZ:
		msg = f'{label} must be {GRID_ORDER_XYZ!r}; got {grid_order!r}'
		raise ValueError(msg)


def _validate_numeric_dtype(dtype: str, label: str) -> None:
	try:
		np_dtype = np.dtype(dtype)
	except TypeError as exc:
		msg = f'{label} must be a NumPy dtype string; got {dtype!r}'
		raise TypeError(msg) from exc
	if np_dtype.hasobject or not np.issubdtype(np_dtype, np.number):
		msg = f'{label} must be numeric; got {dtype!r}'
		raise TypeError(msg)


__all__ = [
	'GRID_ORDER_XYZ',
	'AmplitudeVolumeRecord',
	'CropRequest',
	'SurveyManifest',
	'TensorLike',
	'read_manifest_json',
	'survey_manifest_from_dict',
	'survey_manifest_to_dict',
	'write_manifest_json',
]
