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
from seis_ssl_cluster.volve.horizon_data import (
	BINDING_RELATIVE_ROOT,
	CANONICAL_RELATIVE_ROOT,
	HORIZON_NAMES,
	VISUAL_QC_RELATIVE_ROOT,
	VolveHorizonData,
	VolveHorizonGeometry,
	VolveHorizonPaths,
	array_sha256,
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


def write_synthetic_volve_horizon_root(
	tmp_path: Path,
) -> tuple[Path, VolveHorizonGeometry]:
	root = (tmp_path / 'public' / 'volve_horizons').resolve()
	canonical = root / CANONICAL_RELATIVE_ROOT
	binding = root / BINDING_RELATIVE_ROOT
	visual_qc = root / VISUAL_QC_RELATIVE_ROOT
	canonical.mkdir(parents=True)
	binding.mkdir(parents=True)
	visual_qc.mkdir(parents=True)
	geometry = VolveHorizonGeometry(
		shape_xyz=(6, 7, 800),
		inline_min=100,
		inline_max=105,
		crossline_min=200,
		crossline_max=206,
		first_twt_ms=4.0,
		sample_interval_ms=4.0,
	)
	inline = np.arange(100, 106, dtype=np.int32)
	crossline = np.arange(200, 207, dtype=np.int32)
	time_ms = 4.0 + 4.0 * np.arange(800, dtype=np.float32)
	valid = np.ones((6, 7), dtype=np.bool_)
	valid[0, 0] = False
	np.save(canonical / 'inline_values.npy', inline)
	np.save(canonical / 'crossline_values.npy', crossline)
	np.save(canonical / 'time_ms.npy', time_ms)
	np.save(canonical / 'valid_trace_mask.npy', valid)

	shape = (len(HORIZON_NAMES), 6, 7)
	bound = np.broadcast_to(valid, shape).copy()
	bound[0, 1, 1] = False
	bound[4, 2, 2] = False
	source = bound.copy()
	base_samples = np.asarray([572, 610, 650, 700, 743], dtype=np.int16)
	sample_index = np.broadcast_to(base_samples[:, None, None], shape).copy()
	sample_index[~bound] = -1
	sample_float = sample_index.astype(np.float32)
	twt_ms = 4.0 + 4.0 * sample_float
	residual = np.zeros(shape, dtype=np.float32)
	for array in (sample_float, twt_ms, residual):
		array[~bound] = np.nan
	common = np.all(bound, axis=0)
	continuous_order = common.copy()
	sample_order = common.copy()
	binding_npz = binding / 'volve_horizon_binding_v2.npz'
	np.savez_compressed(
		binding_npz,
		horizon_names=np.asarray(HORIZON_NAMES),
		twt_ms=twt_ms,
		sample_float=sample_float,
		sample_index=sample_index,
		xy_residual_m=residual,
		source_present_mask=source,
		bound_valid_mask=bound,
		common_bound_mask=common,
		continuous_strict_order_mask=continuous_order,
		sample_strict_order_mask=sample_order,
	)
	per_horizon = {
		name: {'bound_valid_count': int(np.count_nonzero(bound[index]))}
		for index, name in enumerate(HORIZON_NAMES)
	}
	horizon_summary = {
		'artifact': str(binding_npz),
		'artifact_sha256': _sha256(binding_npz),
		'horizon_order': list(HORIZON_NAMES),
		'per_horizon': per_horizon,
		'common_bound_count': int(np.count_nonzero(common)),
		'continuous_strict_order_count': int(
			np.count_nonzero(continuous_order)
		),
		'sample_strict_order_count': int(np.count_nonzero(sample_order)),
		'continuous_order_excluded_count': 0,
		'sample_order_excluded_count': 0,
		'common_bound_mask_sha256': array_sha256(common),
		'sample_strict_order_mask_sha256': array_sha256(sample_order),
		'adjacent_order': {},
		'acceptance_checks': {'synthetic_horizons_pass': True},
		'diagnostic_checks': {},
	}
	_write_json(binding / 'volve_horizon_binding_summary_v2.json', horizon_summary)
	grid_summary = {
		'schema_version': 2,
		'artifact_type': 'volve_st10010_twt_interpretation_binding',
		'status': 'PASS',
		'acceptance_checks': {
			'trace_grid': {'synthetic_trace_grid_pass': True},
			'horizons': {'synthetic_horizons_pass': True},
			'faults': {'synthetic_faults_pass': True},
		},
		'horizons': horizon_summary,
		'trace_grid': {
			'valid_trace_mask_sha256': array_sha256(valid),
		},
	}
	grid_path = binding / 'volve_grid_binding_summary_v2.json'
	_write_json(grid_path, grid_summary)
	_write_json(
		visual_qc / 'manual_review.json',
		{
			'schema_version': 1,
			'artifact_type': 'volve_binding_visual_qc_manual_review',
			'status': 'PASS',
			'horizon_visual_qc': 'PASS',
			'fault_visual_qc': 'PASS',
			'source_binding_summary_sha256': _sha256(grid_path),
		},
	)
	return root, geometry


def write_synthetic_frozen_horizon_data(
	tmp_path: Path,
	*,
	shape_xy: tuple[int, int] = (24, 24),
) -> VolveHorizonData:
	'''Write horizon inputs with room for the fixed reserved test lines.'''
	if shape_xy[0] < 24 or shape_xy[1] < 24:
		raise ValueError('synthetic frozen horizon shape must be at least 24 by 24')
	root = (tmp_path / 'public' / 'frozen_volve').resolve()
	canonical = root / CANONICAL_RELATIVE_ROOT
	binding = root / BINDING_RELATIVE_ROOT
	visual_qc = root / VISUAL_QC_RELATIVE_ROOT
	canonical.mkdir(parents=True)
	binding.mkdir(parents=True)
	visual_qc.mkdir(parents=True)
	inline = np.arange(100, 100 + shape_xy[0], dtype=np.int32)
	crossline = np.arange(200, 200 + shape_xy[1], dtype=np.int32)
	time_ms = 4.0 + 4.0 * np.arange(800, dtype=np.float32)
	valid = np.ones(shape_xy, dtype=np.bool_)
	valid[0, 0] = False
	paths = VolveHorizonPaths(
		binding_npz=binding / 'volve_horizon_binding_v2.npz',
		horizon_summary=binding / 'volve_horizon_binding_summary_v2.json',
		grid_summary=binding / 'volve_grid_binding_summary_v2.json',
		manual_review=visual_qc / 'manual_review.json',
		inline_values=canonical / 'inline_values.npy',
		crossline_values=canonical / 'crossline_values.npy',
		time_ms=canonical / 'time_ms.npy',
		valid_trace_mask=canonical / 'valid_trace_mask.npy',
	)
	for path, value in (
		(paths.inline_values, inline),
		(paths.crossline_values, crossline),
		(paths.time_ms, time_ms),
		(paths.valid_trace_mask, valid),
	):
		np.save(path, value)
	shape = (len(HORIZON_NAMES), *shape_xy)
	bound = np.broadcast_to(valid, shape).copy()
	base = np.asarray([572.25, 610.5, 650.0, 700.75, 743.25], dtype=np.float32)
	sample_float = np.broadcast_to(base[:, None, None], shape).copy()
	sample_float[~bound] = np.nan
	sample_index = np.full(shape, -1, dtype=np.int16)
	sample_index[bound] = np.floor(sample_float[bound]).astype(np.int16)
	twt_ms = 4.0 + 4.0 * sample_float
	residual = np.zeros(shape, dtype=np.float32)
	residual[~bound] = np.nan
	common = np.all(bound, axis=0)
	for path in (
		paths.binding_npz,
		paths.horizon_summary,
		paths.grid_summary,
		paths.manual_review,
	):
		path.write_bytes(path.name.encode())
	return VolveHorizonData(
		paths=paths,
		inline_values=np.load(paths.inline_values, mmap_mode='r'),
		crossline_values=np.load(paths.crossline_values, mmap_mode='r'),
		time_ms=np.load(paths.time_ms, mmap_mode='r'),
		valid_trace_mask=np.load(paths.valid_trace_mask, mmap_mode='r'),
		horizon_names=HORIZON_NAMES,
		twt_ms=twt_ms,
		sample_float=sample_float,
		sample_index=sample_index,
		xy_residual_m=residual,
		source_present_mask=bound.copy(),
		bound_valid_mask=bound,
		common_bound_mask=common,
		continuous_strict_order_mask=common.copy(),
		sample_strict_order_mask=common.copy(),
		binding_npz_sha256=_sha256(paths.binding_npz),
		horizon_summary_sha256=_sha256(paths.horizon_summary),
		grid_summary_sha256=_sha256(paths.grid_summary),
		manual_review_sha256=_sha256(paths.manual_review),
	)


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
