from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.volve import (
	VOLVE_CANONICAL_RELATIVE_ROOT,
	VolveCanonicalIdentity,
	VolveCanonicalInputConfig,
	prepare_volve_canonical_inputs,
)

if TYPE_CHECKING:
	from pathlib import Path


def write_synthetic_volve_canonical_root(
	tmp_path: Path,
) -> VolveCanonicalIdentity:
	canonical_root = (
		tmp_path / 'public' / 'volve' / VOLVE_CANONICAL_RELATIVE_ROOT
	)
	canonical_root.mkdir(parents=True)
	shape_xyz = (2, 3, 4)
	amplitude = np.arange(np.prod(shape_xyz), dtype=np.float32).reshape(shape_xyz)
	valid_mask = np.ones(shape_xyz[:2], dtype=bool)
	valid_mask[1, 2] = False
	amplitude[~valid_mask] = np.nan
	amplitude_path = canonical_root / 'amplitude.npy'
	np.save(amplitude_path, amplitude)
	np.save(canonical_root / 'valid_trace_mask.npy', valid_mask)
	np.save(canonical_root / 'inline_values.npy', np.array([100, 101]))
	np.save(canonical_root / 'crossline_values.npy', np.array([200, 201, 202]))
	np.save(
		canonical_root / 'time_ms.npy',
		np.array([4.0, 8.0, 12.0, 16.0], dtype=np.float32),
	)
	stats_path = canonical_root / 'normalization_stats.json'
	_write_json(
		stats_path,
		{
			'schema_version': 1,
			'status': 'PASS',
			'deterministic_sample': {
				'policy': 'synthetic deterministic sample',
				'value_quantiles': {
					'p1': -4.0,
					'p25': -1.0,
					'p50': 0.25,
					'p75': 1.0,
					'p99': 5.0,
				},
			},
		},
	)
	parity_path = canonical_root / 'trace_parity.json'
	_write_json(
		parity_path,
		{
			'schema_version': 1,
			'status': 'PASS',
			'all_exact': True,
			'full_volume_checks': {
				'valid_finite_voxel_count': 5 * shape_xyz[2],
				'missing_nan_voxel_count': shape_xyz[2],
			},
		},
	)
	identity = VolveCanonicalIdentity(
		dataset_id='synthetic_volve',
		survey_id='synthetic_survey',
		shape_xyz=shape_xyz,
		dtype='float32',
		inline_min=100,
		inline_max=101,
		crossline_min=200,
		crossline_max=202,
		first_twt_ms=4.0,
		sample_interval_ms=4.0,
		valid_trace_count=5,
		missing_trace_count=1,
		source_segy_sha256='a' * 64,
		amplitude_sha256=_sha256(amplitude_path),
		amplitude_size_bytes=amplitude_path.stat().st_size,
	)
	artifacts = {
		name: _artifact_record(canonical_root / name)
		for name in (
			'amplitude.npy',
			'valid_trace_mask.npy',
			'inline_values.npy',
			'crossline_values.npy',
			'time_ms.npy',
			'normalization_stats.json',
			'trace_parity.json',
		)
	}
	_write_json(
		canonical_root / 'canonical_volume_manifest.json',
		{
			'schema_version': 1,
			'status': 'PASS',
			'dataset_id': identity.dataset_id,
			'shape': list(shape_xyz),
			'axis_order': ['inline', 'crossline', 'twt'],
			'dtype': 'float32',
			'geometry': {
				'valid_trace_count': 5,
				'missing_trace_count': 1,
			},
			'sources': {'segy': {'sha256': identity.source_segy_sha256}},
			'artifacts': artifacts,
		},
	)
	return identity


def synthetic_volve_canonical_config(tmp_path: Path) -> VolveCanonicalInputConfig:
	identity = write_synthetic_volve_canonical_root(tmp_path)
	return VolveCanonicalInputConfig(
		volve_root=(tmp_path / 'public' / 'volve').resolve(),
		artifact_root=(tmp_path / 'artifacts').resolve(),
		identity=identity,
	)


def write_synthetic_volve_registration(
	tmp_path: Path,
) -> VolveCanonicalInputConfig:
	config = synthetic_volve_canonical_config(tmp_path)
	prepare_volve_canonical_inputs(config)
	return config


def _artifact_record(path: Path) -> dict[str, object]:
	return {
		'relative_path': path.name,
		'size_bytes': path.stat().st_size,
		'sha256': _sha256(path),
	}


def _sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)
