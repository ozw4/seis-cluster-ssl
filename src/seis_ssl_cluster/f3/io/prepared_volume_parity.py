"""Read-only parity gate between two prepared F3 facies registry versions.

Reusing encoder checkpoints trained on one prepared version (v1) with the
amplitude of another (v2) assumes both preparations produced the same data.
This module checks that assumption on the live artifacts instead of on the
config values: NPY bytes, shape/dtype/grid order, the semantic normalization
fields and the class order must all match. Output paths and the dataset
version are expected to differ and are not compared.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from seis_ssl_cluster.data.normalization import load_normalization_stats
from seis_ssl_cluster.data.volume_store import inspect_npy_volume
from seis_ssl_cluster.embedding.writer import file_sha256

if TYPE_CHECKING:
	from pathlib import Path

	from seis_ssl_cluster.f3.io.prepare_volume import F3PrepareVolumeConfig

# Normalization fields that carry meaning for checkpoint reuse. ``max_samples``
# and ``seed`` are not recorded in the stats JSON, so they come from the config.
NORMALIZATION_PARITY_FIELDS = (
	'clip_low_percentile',
	'clip_high_percentile',
	'eps',
	'max_samples',
	'seed',
	'clip_low',
	'clip_high',
	'median',
	'iqr',
)


@dataclass(frozen=True)
class F3PreparedVolumeIdentity:
	"""Semantic identity of one prepared F3 facies registry version."""

	dataset_version: str
	seismic_npy: Path
	seismic_sha256: str
	seismic_shape_xyz: tuple[int, int, int]
	seismic_dtype: str
	label_npy: Path
	label_sha256: str
	label_shape_xyz: tuple[int, int, int]
	label_dtype: str
	grid_order: tuple[str, str, str]
	normalization: Mapping[str, object]
	class_order: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class F3PreparedVolumeParity:
	"""Outcome of comparing a candidate preparation against a reference."""

	reference: F3PreparedVolumeIdentity
	candidate: F3PreparedVolumeIdentity
	mismatches: tuple[str, ...]

	@property
	def passed(self) -> bool:
		"""Return ``True`` when every compared field matches."""
		return not self.mismatches


def inspect_f3_prepared_volume_identity(
	config: F3PrepareVolumeConfig,
) -> F3PreparedVolumeIdentity:
	"""Read the semantic identity of the prepared volumes named by ``config``."""
	outputs = config.outputs
	seismic = inspect_npy_volume(outputs.seismic_npy)
	label = inspect_npy_volume(outputs.label_npy)
	metadata = _read_json_mapping(outputs.metadata_path, label='F3 metadata JSON')
	stats = load_normalization_stats(outputs.normalization_stats_path)
	if stats.source_path != outputs.seismic_npy:
		msg = (
			'normalization stats source_path does not name the prepared seismic '
			f'volume: {stats.source_path} != {outputs.seismic_npy}'
		)
		raise ValueError(msg)
	volumes = _required_mapping(metadata, 'volumes', label=outputs.metadata_path)
	_validate_metadata_volume(
		_required_mapping(volumes, 'seismic', label=outputs.metadata_path),
		shape_xyz=seismic.shape_xyz,
		dtype=seismic.dtype,
		label=f'{outputs.metadata_path} volumes.seismic',
	)
	_validate_metadata_volume(
		_required_mapping(volumes, 'label', label=outputs.metadata_path),
		shape_xyz=label.shape_xyz,
		dtype=label.dtype,
		label=f'{outputs.metadata_path} volumes.label',
	)
	grid_order = _string_triplet(
		metadata.get('grid_order'),
		label=f'{outputs.metadata_path} grid_order',
	)
	if grid_order != stats.grid_order:
		msg = (
			f'metadata grid_order {grid_order!r} differs from normalization stats '
			f'grid_order {stats.grid_order!r}: {outputs.metadata_path}'
		)
		raise ValueError(msg)
	classes = metadata.get('facies_classes')
	if not isinstance(classes, list) or not classes:
		msg = (
			'F3 metadata facies_classes must be a non-empty list: '
			f'{outputs.metadata_path}'
		)
		raise ValueError(msg)
	normalization = {
		'clip_low_percentile': stats.clip_low_percentile,
		'clip_high_percentile': stats.clip_high_percentile,
		'eps': stats.eps,
		'max_samples': config.normalization.max_samples,
		'seed': config.normalization.seed,
		'clip_low': stats.clip_low,
		'clip_high': stats.clip_high,
		'median': stats.median,
		'iqr': stats.iqr,
	}
	_validate_stats_match_config(
		normalization, config, label=str(outputs.normalization_stats_path)
	)
	return F3PreparedVolumeIdentity(
		dataset_version=config.dataset.version,
		seismic_npy=outputs.seismic_npy,
		seismic_sha256=file_sha256(outputs.seismic_npy),
		seismic_shape_xyz=seismic.shape_xyz,
		seismic_dtype=seismic.dtype,
		label_npy=outputs.label_npy,
		label_sha256=file_sha256(outputs.label_npy),
		label_shape_xyz=label.shape_xyz,
		label_dtype=label.dtype,
		grid_order=grid_order,
		normalization=normalization,
		class_order=tuple(
			_class_entry(entry, path=outputs.metadata_path) for entry in classes
		),
	)


def check_f3_prepared_volume_parity(
	reference: F3PrepareVolumeConfig,
	candidate: F3PrepareVolumeConfig,
) -> F3PreparedVolumeParity:
	"""Compare the live prepared volumes of ``candidate`` against ``reference``.

	The comparison is exact: SHA-256 of both NPY files, shape, dtype, grid
	order, the semantic normalization fields and the class order. Nothing is
	written.
	"""
	reference_identity = inspect_f3_prepared_volume_identity(reference)
	candidate_identity = inspect_f3_prepared_volume_identity(candidate)
	mismatches = [
		f'{field}: {getattr(reference_identity, field)!r} != '
		f'{getattr(candidate_identity, field)!r}'
		for field in (
			'seismic_sha256',
			'seismic_shape_xyz',
			'seismic_dtype',
			'label_sha256',
			'label_shape_xyz',
			'label_dtype',
			'grid_order',
			'class_order',
		)
		if getattr(reference_identity, field) != getattr(candidate_identity, field)
	]
	mismatches.extend(
		f'normalization.{field}: {reference_identity.normalization[field]!r} != '
		f'{candidate_identity.normalization[field]!r}'
		for field in NORMALIZATION_PARITY_FIELDS
		if reference_identity.normalization[field]
		!= candidate_identity.normalization[field]
	)
	return F3PreparedVolumeParity(
		reference=reference_identity,
		candidate=candidate_identity,
		mismatches=tuple(mismatches),
	)


def _validate_stats_match_config(
	normalization: Mapping[str, object],
	config: F3PrepareVolumeConfig,
	*,
	label: str,
) -> None:
	expected = {
		'clip_low_percentile': config.normalization.clip_low_percentile,
		'clip_high_percentile': config.normalization.clip_high_percentile,
		'eps': config.normalization.eps,
	}
	for key, value in expected.items():
		if normalization[key] != value:
			msg = (
				f'normalization stats {key} {normalization[key]!r} differs from the '
				f'prepare config value {value!r}: {label}'
			)
			raise ValueError(msg)


def _validate_metadata_volume(
	entry: Mapping[str, object],
	*,
	shape_xyz: tuple[int, ...],
	dtype: str,
	label: str,
) -> None:
	recorded_shape = entry.get('shape_xyz')
	recorded_dtype = entry.get('dtype')
	if not isinstance(recorded_shape, list) or tuple(recorded_shape) != shape_xyz:
		msg = (
			f'{label} shape_xyz {recorded_shape!r} differs from NPY shape {shape_xyz!r}'
		)
		raise ValueError(msg)
	if recorded_dtype != dtype:
		msg = f'{label} dtype {recorded_dtype!r} differs from NPY dtype {dtype!r}'
		raise ValueError(msg)


def _class_entry(entry: object, *, path: Path) -> Mapping[str, object]:
	if not isinstance(entry, Mapping) or 'class_id' not in entry:
		msg = f'F3 metadata facies_classes entries must map class_id: {path}'
		raise ValueError(msg)
	return {
		key: tuple(value) if isinstance(value, list) else value
		for key, value in sorted(entry.items())
	}


def _string_triplet(value: object, *, label: str) -> tuple[str, str, str]:
	if (
		not isinstance(value, list)
		or len(value) != 3
		or not all(isinstance(item, str) for item in value)
	):
		msg = f'{label} must be a list of three axis names; got {value!r}'
		raise ValueError(msg)
	return (value[0], value[1], value[2])


def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, object]:
	if not path.is_file():
		msg = f'{label} does not exist: {path}'
		raise FileNotFoundError(msg)
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		msg = f'{label} must be a JSON object: {path}'
		raise TypeError(msg)
	return payload


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
	*,
	label: Path,
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'F3 metadata {key} must be a mapping: {label}'
		raise TypeError(msg)
	return value


__all__ = [
	'NORMALIZATION_PARITY_FIELDS',
	'F3PreparedVolumeIdentity',
	'F3PreparedVolumeParity',
	'check_f3_prepared_volume_parity',
	'inspect_f3_prepared_volume_identity',
]
