# ruff: noqa: CPY001
"""Fail-closed validation for Parihaka MAE inputs and training artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np

from seis_ssl_cluster.config import load_config, resolve_mae_training_config
from seis_ssl_cluster.data.normalization import (
	compute_normalization_stats,
	load_normalization_stats,
)
from seis_ssl_cluster.data.schema import GRID_ORDER_XYZ, read_manifest_json
from seis_ssl_cluster.data.volume_store import inspect_npy_volume
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.parihaka.prepare_volume import (
	ArrayStatistics,
	ParihakaPrepareVolumeConfig,
	inspect_parihaka_preparation,
	parihaka_prepare_volume_config_from_mapping,
)
from seis_ssl_cluster.training.mae_checkpoint import (
	MaeCheckpointInspection,
	inspect_mae_checkpoint,
)

MODEL_TAG = 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
SMOKE_DIR_NAME = 'smoke_2step'
FULL_DIR_NAME = 'full_100ep'
_STAT_ABS_TOL = 1.0e-6
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_SMOKE_FULL_ALLOWED_DIFFERENCES = frozenset(
	{
		'paths.output_root',
		'train.batch_size',
		'train.samples_per_epoch',
		'train.epochs',
		'train.num_workers',
		'train.prefetch_factor',
		'train.persistent_workers',
		'train.amp',
		'train.device',
		'train.max_steps',
		'visualization.mae_debug.enabled',
	},
)
_PARIHAKA_FULL_CONTRACT: Mapping[str, object] = {
	'data': {
		'amplitude_agc': {
			'clip_abs': 5.0,
			'enabled': True,
			'eps': 1.0e-3,
			'mode': 'trace_rms_z',
			'window_z': 65,
		},
		'finite_check_mode': 'strict',
		'grid_order': ['x', 'y', 'z'],
		'input_channels': 1,
		'local_crop_size': [128, 128, 128],
		'max_resample_attempts': 16,
		'min_valid_fraction': 0.1,
		'normalized_clip_abs': 8.0,
		'target_channels': 1,
		'use_context': False,
		'volume_format': 'npy_memmap',
	},
	'zero_mask': {
		'enabled': True,
		'xy_trace_influence_radius': 1,
		'z_sample_influence_radius': 16,
		'zero_atol': 0.0,
	},
	'model': {
		'decoder_depth': 4,
		'decoder_dim': 256,
		'decoder_heads': 4,
		'encoder_depth': 8,
		'encoder_dim': 384,
		'encoder_heads': 6,
		'in_channels': 1,
		'name': 'amp_mae3d',
		'out_channels': 1,
		'patch_size': [8, 8, 8],
	},
	'masking': {
		'block_size_tokens': [1, 1, 1],
		'spatial_mask_mode': 'block',
		'spatial_mask_ratio': 0.75,
	},
	'loss': {
		'gradient_weight': 0.0,
		'reconstruction': 'mse',
		'target_normalization': {
			'eps': 1.0e-6,
			'min_std': 0.05,
			'mode': 'patch_zscore',
		},
		'valid_mask_mode': 'voxel',
		'visible_reconstruction_weight': 0.1,
	},
	'train': {
		'amp': True,
		'amp_dtype': 'auto',
		'batch_size': 4,
		'device': 'cuda',
		'epochs': 100,
		'grad_clip_norm': 1.0,
		'lr': 1.0e-4,
		'num_workers': 8,
		'persistent_workers': True,
		'prefetch_factor': 2,
		'runtime_check_mode': 'once',
		'samples_per_epoch': 10_000,
		'seed': 42,
		'shuffle': True,
		'stage_timing': False,
		'weight_decay': 0.05,
	},
	'visualization': {
		'mae_debug': {
			'clip_percentiles': [1.0, 99.0],
			'columns': [
				'input',
				'masked_input',
				'target',
				'prediction',
				'abs_error',
				'valid_mask',
			],
			'dpi': 160,
			'enabled': True,
			'every_epochs': None,
			'every_steps': 1_000,
			'invalid_color': 'lightgray',
			'max_samples': 1,
			'panel_height': 2.4,
			'panel_width': 2.6,
			'xy_slice_index': None,
			'xz_slice_y_index': None,
		},
	},
}


@dataclass(frozen=True)
class ParihakaMaeValidationResult:
	"""Small validation result suitable for stdout or optional JSON."""

	check: str
	status: str
	prepare_config: Path
	smoke_config: Path
	full_config: Path
	source_npz: Path
	source_sha256: str
	prepared_npy: Path
	prepared_sha256: str
	smoke_output_root: Path
	full_output_root: Path
	latest_checkpoint: Path | None = None
	latest_sha256: str | None = None
	best_checkpoint: Path | None = None
	best_sha256: str | None = None
	checkpoint_schema_version: int | None = None
	checkpoint_epoch: int | None = None
	checkpoint_global_step: int | None = None
	resolved_precision: str | None = None
	finite_metric_min: float | None = None
	finite_metric_max: float | None = None
	best_checkpoint_epoch: int | None = None
	best_checkpoint_global_step: int | None = None
	best_metric_key: str | None = None
	best_metric_value: float | None = None
	scaler_present: bool | None = None
	latest_metrics: tuple[tuple[str, float], ...] = ()
	best_metrics: tuple[tuple[str, float], ...] = ()

	def to_dict(self) -> dict[str, object]:
		"""Return the intentionally small report payload."""
		return {
			'check': self.check,
			'status': self.status,
			'prepare_config': str(self.prepare_config),
			'smoke_config': str(self.smoke_config),
			'full_config': str(self.full_config),
			'source': {
				'path': str(self.source_npz),
				'sha256': self.source_sha256,
			},
			'prepared_npy': {
				'path': str(self.prepared_npy),
				'sha256': self.prepared_sha256,
			},
			'smoke_output_root': str(self.smoke_output_root),
			'full_output_root': str(self.full_output_root),
			'latest_checkpoint': _optional_checkpoint_reference(
				self.latest_checkpoint,
				self.latest_sha256,
			),
			'best_checkpoint': _optional_checkpoint_reference(
				self.best_checkpoint,
				self.best_sha256,
			),
			'checkpoint': {
				'schema_version': self.checkpoint_schema_version,
				'epoch': self.checkpoint_epoch,
				'global_step': self.checkpoint_global_step,
				'resolved_precision': self.resolved_precision,
				'finite_metric_min': self.finite_metric_min,
				'finite_metric_max': self.finite_metric_max,
				'best_epoch': self.best_checkpoint_epoch,
				'best_global_step': self.best_checkpoint_global_step,
				'best_metric_key': self.best_metric_key,
				'best_metric_value': self.best_metric_value,
				'scaler_present': self.scaler_present,
			},
		}


@dataclass(frozen=True)
class _InputValidation:
	prepare: ParihakaPrepareVolumeConfig
	smoke: dict[str, object]
	full: dict[str, object]
	source_sha256: str
	prepared_sha256: str


def validate_parihaka_mae(
	*,
	prepare_config_path: str | Path,
	smoke_config_path: str | Path,
	full_config_path: str | Path,
	check: str,
) -> ParihakaMaeValidationResult:
	"""Validate Parihaka input identity and one closed training check."""
	if check not in {'inputs', 'smoke', 'full'}:
		msg = f'check must be inputs, smoke, or full; got {check!r}'
		raise ValueError(msg)
	prepare_path = _required_file(prepare_config_path, 'prepare config')
	smoke_path = _required_file(smoke_config_path, 'smoke config')
	full_path = _required_file(full_config_path, 'full config')
	prepare = parihaka_prepare_volume_config_from_mapping(load_config(prepare_path))
	smoke_raw = load_config(smoke_path)
	full_raw = load_config(full_path)
	inputs = validate_parihaka_mae_inputs_from_configs(
		prepare=prepare,
		smoke_raw=smoke_raw,
		full_raw=full_raw,
	)
	base = {
		'check': check,
		'status': 'pass',
		'prepare_config': prepare_path,
		'smoke_config': smoke_path,
		'full_config': full_path,
		'source_npz': prepare.inputs.amplitude_npz,
		'source_sha256': inputs.source_sha256,
		'prepared_npy': prepare.outputs.amplitude_npy,
		'prepared_sha256': inputs.prepared_sha256,
		'smoke_output_root': Path(
			cast('Mapping[str, object]', inputs.smoke['paths'])['output_root']
		),
		'full_output_root': Path(
			cast('Mapping[str, object]', inputs.full['paths'])['output_root']
		),
	}
	if check == 'inputs':
		return ParihakaMaeValidationResult(**base)
	if check == 'smoke':
		return _validate_smoke(inputs, base=base)
	return _validate_full(inputs, base=base)


def validate_parihaka_mae_inputs_from_configs(
	*,
	prepare: ParihakaPrepareVolumeConfig,
	smoke_raw: Mapping[str, object],
	full_raw: Mapping[str, object],
) -> _InputValidation:
	"""Validate already-loaded configs, enabling bounded synthetic fixtures."""
	_reject_label_contract(smoke_raw, label='smoke config')
	_reject_label_contract(full_raw, label='full config')
	inspect_parihaka_preparation(prepare, overwrite=True)
	metadata = _read_json_mapping(prepare.outputs.metadata, 'preparation metadata')
	_reject_label_contract(metadata, label='preparation metadata')
	_validate_metadata_identity(prepare, metadata)
	source_sha256 = _file_sha256(prepare.inputs.amplitude_npz)
	_validate_source_live_identity(prepare, metadata, source_sha256=source_sha256)
	prepared_sha256 = _validate_prepared_live_identity(prepare, metadata)
	_validate_manifest_path_list_and_stats(prepare, metadata)
	smoke = resolve_mae_training_config(smoke_raw)
	full = resolve_mae_training_config(full_raw)
	_validate_training_configs(prepare, smoke, full)
	return _InputValidation(
		prepare=prepare,
		smoke=smoke,
		full=full,
		source_sha256=source_sha256,
		prepared_sha256=prepared_sha256,
	)


def write_parihaka_mae_validation_report(
	result: ParihakaMaeValidationResult,
	path: str | Path,
) -> Path:
	"""Write one explicitly requested small JSON report."""
	report_path = Path(path)
	if report_path.exists() and not report_path.is_file():
		msg = f'JSON report path exists but is not a file: {report_path}'
		raise ValueError(msg)
	report_path.parent.mkdir(parents=True, exist_ok=True)
	report_path.write_text(
		json.dumps(result.to_dict(), indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)
	return report_path


def _validate_metadata_identity(
	prepare: ParihakaPrepareVolumeConfig,
	metadata: Mapping[str, object],
) -> None:
	_expected_equal(
		metadata.get('artifact_type'),
		'parihaka_amplitude_preparation',
		'metadata.artifact_type',
	)
	_expected_equal(metadata.get('schema_version'), 1, 'metadata.schema_version')
	_expected_equal(metadata.get('status'), 'complete', 'metadata.status')
	dataset = _mapping(metadata, 'dataset', 'metadata')
	for key in ('name', 'version', 'survey_id'):
		_expected_equal(
			dataset.get(key), getattr(prepare.dataset, key), f'metadata.dataset.{key}'
		)
	conversion = _mapping(metadata, 'conversion', 'metadata')
	_expected_equal(
		conversion.get('axis_mapping'), 'ZXY -> XYZ', 'metadata.conversion.axis_mapping'
	)
	_expected_equal(
		conversion.get('transpose_axes'),
		[1, 2, 0],
		'metadata.conversion.transpose_axes',
	)
	_expected_equal(
		conversion.get('verification'),
		'full_chunkwise_bitwise',
		'metadata.conversion.verification',
	)
	_expected_equal(
		conversion.get('chunk_size_z'),
		prepare.conversion.chunk_size_z,
		'metadata.conversion.chunk_size_z',
	)
	_validate_source_config_metadata(prepare, metadata)


def _validate_source_config_metadata(
	prepare: ParihakaPrepareVolumeConfig,
	metadata: Mapping[str, object],
) -> None:
	provenance = _mapping(metadata, 'provenance', 'metadata')
	source_config = prepare.source
	for key in (
		'direct_distributor',
		'dataset_title',
		'contributor',
		'version',
		'doi',
		'upstream',
		'displayed_license',
		'archive_name',
		'upstream_member',
		'local_filename',
		'local_modification',
		'aicrowd_byte_identity',
		'redistribution_transformation',
		'acquisition_date',
		'acquisition_by',
	):
		_expected_equal(
			provenance.get(key),
			getattr(source_config, key),
			f'metadata.provenance.{key}',
		)
	source = _mapping(metadata, 'source', 'metadata')
	for key, expected in (
		('array_key', source_config.array_key),
		('member_name', source_config.member_name),
		('shape_zxy', list(source_config.shape_zxy)),
		('dtype', source_config.dtype),
		('fortran_order', source_config.fortran_order),
	):
		_expected_equal(source.get(key), expected, f'metadata.source.{key}')
	header = _mapping(source, 'npy_header', 'metadata.source')
	for key, expected in (
		('shape', list(source_config.shape_zxy)),
		('dtype', source_config.dtype),
		('fortran_order', source_config.fortran_order),
	):
		_expected_equal(header.get(key), expected, f'metadata.source.npy_header.{key}')
	_validate_statistics_mapping(
		_mapping(source, 'statistics', 'metadata.source'),
		source_config.expected_statistics,
		label='metadata.source.statistics',
	)


def _validate_source_live_identity(
	prepare: ParihakaPrepareVolumeConfig,
	metadata: Mapping[str, object],
	*,
	source_sha256: str,
) -> None:
	source = _mapping(metadata, 'source', 'metadata')
	path = prepare.inputs.amplitude_npz
	_expected_equal(source.get('npz_path'), str(path), 'metadata.source.npz_path')
	_expected_equal(
		source.get('npz_sha256'), source_sha256, 'metadata.source.npz_sha256'
	)
	_expected_equal(
		source.get('npz_size_bytes'),
		path.stat().st_size,
		'metadata.source.npz_size_bytes',
	)


def _validate_prepared_live_identity(
	prepare: ParihakaPrepareVolumeConfig,
	metadata: Mapping[str, object],
) -> str:
	path = prepare.outputs.amplitude_npy
	info = inspect_npy_volume(path)
	expected_shape = (
		prepare.source.shape_zxy[1],
		prepare.source.shape_zxy[2],
		prepare.source.shape_zxy[0],
	)
	_expected_equal(info.shape_xyz, expected_shape, 'prepared NPY shape')
	_expected_equal(info.dtype, prepare.source.dtype, 'prepared NPY dtype')
	array = np.load(path, mmap_mode='r', allow_pickle=False)
	if not isinstance(array, np.memmap) or not array.flags.c_contiguous:
		msg = f'prepared NPY must be a C-contiguous mmap-readable array: {path}'
		raise ValueError(msg)
	digest = _file_sha256(path)
	outputs = _mapping(metadata, 'outputs', 'metadata')
	record = _mapping(outputs, 'amplitude_npy', 'metadata.outputs')
	for key, expected in (
		('path', str(path)),
		('sha256', digest),
		('size_bytes', path.stat().st_size),
		('shape_xyz', list(expected_shape)),
		('dtype', prepare.source.dtype),
		('order', 'C'),
	):
		_expected_equal(
			record.get(key), expected, f'metadata.outputs.amplitude_npy.{key}'
		)
	live_statistics = _stream_npy_statistics(array)
	_validate_statistics_mapping(
		_mapping(record, 'statistics', 'metadata.outputs.amplitude_npy'),
		live_statistics,
		label='metadata.outputs.amplitude_npy.statistics',
	)
	return digest


def _validate_manifest_path_list_and_stats(
	prepare: ParihakaPrepareVolumeConfig,
	metadata: Mapping[str, object],
) -> None:
	outputs = _mapping(metadata, 'outputs', 'metadata')
	paths = {
		'manifest': prepare.outputs.manifest,
		'path_list': prepare.outputs.path_list,
		'normalization_stats': prepare.outputs.normalization_stats,
	}
	for key, path in paths.items():
		record = _mapping(outputs, key, 'metadata.outputs')
		_expected_equal(record.get('path'), str(path), f'metadata.outputs.{key}.path')
		_expected_equal(
			record.get('sha256'), _file_sha256(path), f'metadata.outputs.{key}.sha256'
		)
	manifests = read_manifest_json(prepare.outputs.manifest)
	if len(manifests) != 1:
		msg = f'Parihaka manifest must contain exactly one survey; got {len(manifests)}'
		raise ValueError(msg)
	manifest = manifests[0]
	expected_shape = (
		prepare.source.shape_zxy[1],
		prepare.source.shape_zxy[2],
		prepare.source.shape_zxy[0],
	)
	for actual, expected, label in (
		(manifest.survey_id, prepare.dataset.survey_id, 'manifest survey_id'),
		(
			manifest.amplitude.survey_id,
			prepare.dataset.survey_id,
			'manifest amplitude survey_id',
		),
		(
			manifest.amplitude.path,
			prepare.outputs.amplitude_npy,
			'manifest amplitude path',
		),
		(manifest.amplitude.shape_xyz, expected_shape, 'manifest amplitude shape'),
		(manifest.amplitude.dtype, prepare.source.dtype, 'manifest amplitude dtype'),
		(manifest.amplitude.grid_order, GRID_ORDER_XYZ, 'manifest grid order'),
		(
			manifest.amplitude.normalization_stats_path,
			prepare.outputs.normalization_stats,
			'manifest normalization stats path',
		),
	):
		_expected_equal(actual, expected, label)
	lines = prepare.outputs.path_list.read_text(encoding='utf-8').splitlines()
	_expected_equal(lines, [str(prepare.outputs.amplitude_npy)], 'path list')
	if not prepare.outputs.path_list.read_bytes().endswith(b'\n'):
		raise ValueError(
			f'path list must end in a newline: {prepare.outputs.path_list}'
		)
	stats = load_normalization_stats(prepare.outputs.normalization_stats)
	for actual, expected, label in (
		(stats.survey_id, prepare.dataset.survey_id, 'normalization survey_id'),
		(stats.source_path, prepare.outputs.amplitude_npy, 'normalization source path'),
		(stats.grid_order, GRID_ORDER_XYZ, 'normalization grid order'),
		(
			stats.clip_low_percentile,
			prepare.normalization.clip_low_percentile,
			'normalization low percentile',
		),
		(
			stats.clip_high_percentile,
			prepare.normalization.clip_high_percentile,
			'normalization high percentile',
		),
		(stats.eps, prepare.normalization.eps, 'normalization epsilon'),
	):
		_expected_equal(actual, expected, label)
	recomputed = compute_normalization_stats(
		prepare.outputs.amplitude_npy,
		survey_id=prepare.dataset.survey_id,
		grid_order=GRID_ORDER_XYZ,
		clip_low_percentile=prepare.normalization.clip_low_percentile,
		clip_high_percentile=prepare.normalization.clip_high_percentile,
		max_samples=prepare.normalization.max_samples,
		seed=prepare.normalization.seed,
		eps=prepare.normalization.eps,
	)
	_expected_equal(
		stats.to_dict(),
		recomputed.to_dict(),
		'normalization stats/config reproducibility',
	)


def _validate_training_configs(
	prepare: ParihakaPrepareVolumeConfig,
	smoke: Mapping[str, object],
	full: Mapping[str, object],
) -> None:
	artifact_root = prepare.paths.artifact_root
	base = (
		artifact_root / 'pretraining' / 'parihaka' / prepare.dataset.version / MODEL_TAG
	)
	expected_smoke = base / SMOKE_DIR_NAME
	expected_full = base / FULL_DIR_NAME
	for config, output, label in (
		(smoke, expected_smoke, 'smoke'),
		(full, expected_full, 'full'),
	):
		paths = _mapping(config, 'paths', f'{label} config')
		manifests = _mapping(config, 'manifests', f'{label} config')
		_expected_equal(
			paths.get('artifact_root'),
			str(artifact_root),
			f'{label} paths.artifact_root',
		)
		_expected_equal(
			paths.get('output_root'), str(output), f'{label} paths.output_root'
		)
		_expected_equal(
			manifests.get('train'),
			str(prepare.outputs.manifest),
			f'{label} manifests.train',
		)
		_expected_equal(
			manifests.get('train_path_list'),
			str(prepare.outputs.path_list),
			f'{label} manifests.train_path_list',
		)
	_validate_disjoint_run_roots(expected_smoke, expected_full)
	differences = _differing_paths(smoke, full)
	unexpected = differences - _SMOKE_FULL_ALLOWED_DIFFERENCES
	if unexpected:
		msg = (
			'smoke/full config differences are outside the closed allowlist; '
			f'unexpected={sorted(unexpected)!r}'
		)
		raise ValueError(msg)
	_validate_full_contract(full)
	_validate_smoke_contract(smoke)
	_reject_initial_checkpoint(full)


def _validate_full_contract(full: Mapping[str, object]) -> None:
	for section, expected in _PARIHAKA_FULL_CONTRACT.items():
		_expected_equal(
			full.get(section),
			expected,
			f'full {section} Parihaka contract',
		)


def _validate_smoke_contract(smoke: Mapping[str, object]) -> None:
	train = _mapping(smoke, 'train', 'smoke config')
	for key, expected in (
		('amp', False),
		('amp_dtype', 'auto'),
		('device', 'cpu'),
		('seed', 42),
		('epochs', 1),
		('samples_per_epoch', 2),
		('max_steps', 2),
	):
		_expected_equal(train.get(key), expected, f'smoke train.{key}')


def _reject_initial_checkpoint(config: Mapping[str, object]) -> None:
	for path, key, _value in _walk_mapping(config):
		if key in {'teacher', 'init_checkpoint', 'teacher_checkpoint'}:
			msg = f'full config must use seeded random initialization; found {path}'
			raise ValueError(msg)


def _validate_disjoint_run_roots(smoke_root: Path, full_root: Path) -> None:
	resolved_smoke = smoke_root.resolve(strict=False)
	resolved_full = full_root.resolve(strict=False)
	if (
		resolved_smoke == resolved_full
		or _is_relative_to(resolved_smoke, resolved_full)
		or _is_relative_to(resolved_full, resolved_smoke)
	):
		msg = (
			'smoke and full output roots must be distinct and disjoint: '
			f'{smoke_root}, {full_root}'
		)
		raise ValueError(msg)


def _validate_smoke(
	inputs: _InputValidation,
	*,
	base: Mapping[str, object],
) -> ParihakaMaeValidationResult:
	smoke_root = Path(
		cast('Mapping[str, object]', inputs.smoke['paths'])['output_root']
	)
	full_root = Path(cast('Mapping[str, object]', inputs.full['paths'])['output_root'])
	_validate_disjoint_run_roots(smoke_root, full_root)
	latest_path = smoke_root / 'latest.pt'
	best_path = smoke_root / 'best.pt'
	latest = _validate_checkpoint(
		latest_path,
		inputs.smoke,
		label='latest smoke',
		expected_epoch=1,
		expected_global_step=2,
		expected_precision='float32',
		expected_amp_enabled=False,
		expected_scaler_enabled=False,
	)
	best = _validate_checkpoint(
		best_path,
		inputs.smoke,
		label='best smoke',
		expected_epoch=1,
		expected_global_step=2,
		expected_precision='float32',
		expected_amp_enabled=False,
		expected_scaler_enabled=False,
	)
	_validate_run_snapshots(
		inputs,
		smoke_root,
		resolved_config=inputs.smoke,
		expected_precision={
			'amp_requested': False,
			'amp_dtype_requested': 'auto',
			'resolved_dtype': 'float32',
			'amp_enabled': False,
			'grad_scaler_enabled': False,
		},
		label='smoke',
	)
	metric_key = best.best_metric_key
	best_metric = best.best_metric_value
	if metric_key is None or best_metric is None or not math.isfinite(best_metric):
		raise ValueError('best checkpoint must contain a finite existing best metric')
	metric_values = [value for _, value in latest.metrics + best.metrics]
	return ParihakaMaeValidationResult(
		**base,
		latest_checkpoint=latest_path,
		latest_sha256=_file_sha256(latest_path),
		best_checkpoint=best_path,
		best_sha256=_file_sha256(best_path),
		checkpoint_schema_version=2,
		checkpoint_epoch=1,
		checkpoint_global_step=2,
		resolved_precision='float32',
		finite_metric_min=min(metric_values),
		finite_metric_max=max(metric_values),
		best_checkpoint_epoch=1,
		best_checkpoint_global_step=2,
		best_metric_key=metric_key,
		best_metric_value=best_metric,
		scaler_present=False,
		latest_metrics=latest.metrics,
		best_metrics=best.metrics,
	)


def _validate_full(
	inputs: _InputValidation,
	*,
	base: Mapping[str, object],
) -> ParihakaMaeValidationResult:
	full_root = Path(cast('Mapping[str, object]', inputs.full['paths'])['output_root'])
	smoke_root = Path(
		cast('Mapping[str, object]', inputs.smoke['paths'])['output_root']
	)
	_validate_disjoint_run_roots(smoke_root, full_root)
	precision = _load_full_precision_contract(full_root, inputs.full)
	resolved_precision = cast('str', precision['resolved_dtype'])
	amp_enabled = cast('bool', precision['amp_enabled'])
	scaler_enabled = cast('bool', precision['grad_scaler_enabled'])
	latest_path = full_root / 'latest.pt'
	best_path = full_root / 'best.pt'
	latest = _validate_checkpoint(
		latest_path,
		inputs.full,
		label='latest full',
		expected_epoch=100,
		expected_global_step=250_000,
		expected_precision=resolved_precision,
		expected_amp_enabled=amp_enabled,
		expected_scaler_enabled=scaler_enabled,
	)
	best = _validate_checkpoint(
		best_path,
		inputs.full,
		label='best full',
		expected_epoch=None,
		expected_global_step=None,
		expected_precision=resolved_precision,
		expected_amp_enabled=amp_enabled,
		expected_scaler_enabled=scaler_enabled,
	)
	best_epoch = best.epoch
	if not 1 <= best_epoch <= 100:
		msg = f'best full checkpoint epoch must be in [1, 100]; got {best_epoch}'
		raise ValueError(msg)
	steps_per_epoch = _full_steps_per_epoch(inputs.full)
	best_global_step = best_epoch * steps_per_epoch
	_expected_equal(
		best.global_step,
		best_global_step,
		'best full checkpoint global_step',
	)
	_validate_run_snapshots(
		inputs,
		full_root,
		resolved_config=inputs.full,
		expected_precision=precision,
		label='full',
	)
	best_metric_key = best.best_metric_key
	best_metric = best.best_metric_value
	if best_metric_key is None or best_metric is None or not math.isfinite(best_metric):
		raise ValueError('best full checkpoint must contain a finite best metric')
	latest_metric = dict(latest.metrics).get(best_metric_key)
	if latest_metric is None:
		msg = f'latest full checkpoint metric {best_metric_key} must be finite'
		raise ValueError(msg)
	if best_metric > latest_metric:
		msg = (
			'best full checkpoint metric must be no greater than latest: '
			f'{best_metric_key} best={best_metric}, latest={latest_metric}'
		)
		raise ValueError(msg)
	metric_values = [value for _, value in latest.metrics + best.metrics]
	return ParihakaMaeValidationResult(
		**base,
		latest_checkpoint=latest_path,
		latest_sha256=_file_sha256(latest_path),
		best_checkpoint=best_path,
		best_sha256=_file_sha256(best_path),
		checkpoint_schema_version=2,
		checkpoint_epoch=100,
		checkpoint_global_step=250_000,
		resolved_precision=resolved_precision,
		finite_metric_min=min(metric_values),
		finite_metric_max=max(metric_values),
		best_checkpoint_epoch=best_epoch,
		best_checkpoint_global_step=best_global_step,
		best_metric_key=best_metric_key,
		best_metric_value=best_metric,
		scaler_present=bool(precision['grad_scaler_enabled']),
		latest_metrics=latest.metrics,
		best_metrics=best.metrics,
	)


def _load_full_precision_contract(
	full_root: Path,
	full: Mapping[str, object],
) -> dict[str, object]:
	run_metadata = _read_json_mapping(
		full_root / 'run_metadata.json', 'full run metadata'
	)
	precision = _mapping(run_metadata, 'precision', 'full run metadata')
	expected_keys = {
		'amp_requested',
		'amp_dtype_requested',
		'resolved_dtype',
		'amp_enabled',
		'grad_scaler_enabled',
	}
	_expected_equal(set(precision), expected_keys, 'full run metadata precision fields')
	for key in ('amp_requested', 'amp_enabled', 'grad_scaler_enabled'):
		if not isinstance(precision.get(key), bool):
			raise TypeError(f'full run metadata precision.{key} must be a boolean')
	requested_dtype = precision.get('amp_dtype_requested')
	if requested_dtype not in {'auto', 'bfloat16', 'float16'}:
		msg = (
			'full run metadata precision.amp_dtype_requested must be auto, '
			'bfloat16, or float16'
		)
		raise ValueError(msg)
	resolved_dtype = precision.get('resolved_dtype')
	if resolved_dtype not in {'float32', 'bfloat16', 'float16'}:
		msg = (
			'full run metadata precision.resolved_dtype must be float32, '
			'bfloat16, or float16'
		)
		raise ValueError(msg)
	train = _mapping(full, 'train', 'full config')
	_expected_equal(
		precision.get('amp_requested'), train.get('amp'), 'full requested AMP'
	)
	_expected_equal(
		precision.get('amp_dtype_requested'),
		train.get('amp_dtype'),
		'full requested AMP dtype',
	)
	amp_enabled = cast('bool', precision['amp_enabled'])
	scaler_enabled = cast('bool', precision['grad_scaler_enabled'])
	amp_requested = cast('bool', precision['amp_requested'])
	if not amp_requested or not amp_enabled or resolved_dtype == 'float32':
		raise ValueError(
			'full run metadata must describe enabled CUDA AMP with a reduced dtype'
		)
	_expected_equal(
		scaler_enabled,
		resolved_dtype == 'float16',
		'full run metadata precision scaler contract',
	)
	return dict(precision)


def _full_steps_per_epoch(full: Mapping[str, object]) -> int:
	train = _mapping(full, 'train', 'full config')
	samples = _required_int(train.get('samples_per_epoch'), 'full samples_per_epoch')
	batch_size = _required_int(train.get('batch_size'), 'full batch_size')
	if samples % batch_size:
		raise ValueError('full samples_per_epoch must be divisible by batch_size')
	steps = samples // batch_size
	_expected_equal(steps, 2_500, 'full steps per epoch')
	return steps


def _validate_checkpoint(  # noqa: PLR0913
	path: Path,
	resolved_config: Mapping[str, object],
	*,
	label: str,
	expected_epoch: int | None,
	expected_global_step: int | None,
	expected_precision: str,
	expected_amp_enabled: bool,
	expected_scaler_enabled: bool,
) -> MaeCheckpointInspection:
	if not path.is_file():
		raise FileNotFoundError(f'{label} checkpoint does not exist: {path}')
	inspection = inspect_mae_checkpoint(
		path,
		resolved_config=resolved_config,
		model=_build_mae_model(
			_mapping(resolved_config, 'model', f'{label} config')
		),
		resolved_precision=expected_precision,
		amp_enabled=expected_amp_enabled,
		scaler_present=expected_scaler_enabled,
	)
	for actual, expected, field in (
		(inspection.schema_version, 2, 'training_state.schema_version'),
		(inspection.stage, 'train_amp_mae', 'training_state.stage'),
		(
			inspection.checkpoint_kind,
			'epoch',
			'training_state.checkpoint_kind',
		),
		(inspection.batch_index, None, 'training_state.batch_index'),
		(
			inspection.resolved_precision,
			expected_precision,
			'training_state.resolved_precision',
		),
		(inspection.amp_enabled, expected_amp_enabled, 'amp_enabled'),
		(inspection.scaler_present, expected_scaler_enabled, 'scaler_state_dict'),
	):
		_expected_equal(actual, expected, f'{label} checkpoint {field}')
	if expected_epoch is not None:
		_expected_equal(inspection.epoch, expected_epoch, f'{label} checkpoint epoch')
	if expected_global_step is not None:
		_expected_equal(
			inspection.global_step,
			expected_global_step,
			f'{label} checkpoint global_step',
		)
	return inspection


def _validate_run_snapshots(
	inputs: _InputValidation,
	output_root: Path,
	*,
	resolved_config: Mapping[str, object],
	expected_precision: Mapping[str, object],
	label: str,
) -> Mapping[str, object]:
	resolved_path = output_root / 'resolved_config.json'
	manifest_snapshot = output_root / 'manifest.json'
	path_list_snapshot = output_root / 'inputs' / inputs.prepare.outputs.path_list.name
	run_metadata_path = output_root / 'run_metadata.json'
	for path in (
		resolved_path,
		manifest_snapshot,
		path_list_snapshot,
		run_metadata_path,
	):
		if not path.is_file():
			raise FileNotFoundError(f'{label} run snapshot does not exist: {path}')
	expected_resolved = (
		json.dumps(resolved_config, indent=2, sort_keys=True, allow_nan=False) + '\n'
	).encode()
	_expected_equal(
		resolved_path.read_bytes(), expected_resolved, 'resolved config snapshot bytes'
	)
	_expected_equal(
		manifest_snapshot.read_bytes(),
		inputs.prepare.outputs.manifest.read_bytes(),
		'manifest snapshot bytes',
	)
	_expected_equal(
		path_list_snapshot.read_bytes(),
		inputs.prepare.outputs.path_list.read_bytes(),
		'path-list snapshot bytes',
	)
	run_metadata = _read_json_mapping(run_metadata_path, f'{label} run metadata')
	_expected_equal(
		run_metadata.get('runtime_check_mode'),
		'once',
		f'{label} run metadata runtime_check_mode',
	)
	_expected_equal(
		run_metadata.get('precision'),
		dict(expected_precision),
		f'{label} run metadata precision',
	)
	return run_metadata


def _build_mae_model(model: Mapping[str, object]) -> AmplitudeMAE3D:
	return AmplitudeMAE3D(
		in_channels=int(model['in_channels']),
		out_channels=int(model['out_channels']),
		patch_size_xyz=cast('tuple[int, int, int]', tuple(model['patch_size'])),
		encoder_dim=int(model['encoder_dim']),
		encoder_depth=int(model['encoder_depth']),
		encoder_heads=int(model['encoder_heads']),
		decoder_dim=int(model['decoder_dim']),
		decoder_depth=int(model['decoder_depth']),
		decoder_heads=int(model['decoder_heads']),
		runtime_check_mode='once',
	)


def _stream_npy_statistics(array: np.ndarray) -> ArrayStatistics:
	element_count = finite_count = 0
	minimum = math.inf
	maximum = -math.inf
	total = total_squares = 0.0
	for start in range(0, array.shape[0], 8):
		chunk = np.asarray(array[start : start + 8])
		element_count += int(chunk.size)
		finite = np.isfinite(chunk)
		count = int(np.count_nonzero(finite))
		finite_count += count
		if count == 0:
			continue
		values = chunk if count == chunk.size else chunk[finite]
		minimum = min(minimum, float(np.min(values)))
		maximum = max(maximum, float(np.max(values)))
		total += float(np.sum(values, dtype=np.float64))
		total_squares += float(
			np.sum(np.square(values, dtype=np.float64), dtype=np.float64),
		)
	mean = total / finite_count if finite_count else math.nan
	variance = total_squares / finite_count - mean * mean if finite_count else math.nan
	return ArrayStatistics(
		element_count=element_count,
		finite_count=finite_count,
		nonfinite_count=element_count - finite_count,
		minimum=minimum,
		maximum=maximum,
		mean=mean,
		population_std=math.sqrt(max(0.0, variance)),
	)


def _validate_statistics_mapping(
	actual: Mapping[str, object],
	expected: ArrayStatistics,
	*,
	label: str,
) -> None:
	for key, expected_value in expected.to_dict().items():
		actual_value = actual.get(key)
		if key in {'mean', 'population_std'}:
			if (
				isinstance(actual_value, bool)
				or not isinstance(actual_value, int | float)
				or not math.isclose(
					float(actual_value),
					float(expected_value),
					rel_tol=0.0,
					abs_tol=_STAT_ABS_TOL,
				)
			):
				raise ValueError(
					_statistic_mismatch(label, key, expected_value, actual_value)
				)
		elif actual_value != expected_value:
			raise ValueError(
				_statistic_mismatch(label, key, expected_value, actual_value),
			)


def _statistic_mismatch(
	label: str,
	key: str,
	expected: object,
	actual: object,
) -> str:
	return f'{label}.{key} mismatch: expected {expected!r}, got {actual!r}'


def _differing_paths(left: object, right: object, prefix: str = '') -> frozenset[str]:
	if isinstance(left, Mapping) and isinstance(right, Mapping):
		differences: set[str] = set()
		for key in set(left) | set(right):
			path = f'{prefix}.{key}' if prefix else str(key)
			if key not in left or key not in right:
				differences.add(path)
			else:
				differences.update(_differing_paths(left[key], right[key], path))
		return frozenset(differences)
	return frozenset({prefix}) if left != right else frozenset()


def _reject_label_contract(value: object, *, label: str) -> None:
	for path, key, child in _walk_mapping(value):
		normalized = key.lower()
		if (
			normalized
			in {'label', 'labels', 'class', 'classes', 'class_id', 'class_count'}
			or normalized.startswith(('label_', 'class_'))
			or normalized.endswith('_label')
		):
			raise ValueError(f'{label} contains forbidden label/class field: {path}')
		if isinstance(child, str) and key != 'dataset_title':
			child_lower = child.lower()
			if ('/' in child or '\\' in child) and 'label' in Path(child_lower).name:
				raise ValueError(f'{label} contains forbidden label path: {path}')


def _walk_mapping(value: object, prefix: str = '') -> Sequence[tuple[str, str, object]]:
	items: list[tuple[str, str, object]] = []
	if not isinstance(value, Mapping):
		return items
	for key, child in value.items():
		key_string = str(key)
		path = f'{prefix}.{key_string}' if prefix else key_string
		items.append((path, key_string, child))
		items.extend(_walk_mapping(child, path))
	return items


def _mapping(
	parent: Mapping[str, object], key: str, label: str
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'{label}.{key} must be a mapping')
	return cast('Mapping[str, object]', value)


def _expected_equal(actual: object, expected: object, label: str) -> None:
	if actual != expected:
		raise ValueError(f'{label} mismatch: expected {expected!r}, got {actual!r}')


def _required_int(value: object, label: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int):
		raise TypeError(f'{label} must be an integer')
	return value


def _read_json_mapping(path: Path, label: str) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(f'{label} does not exist: {path}')
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(f'{label} is invalid JSON: {path}: {exc}') from exc
	if not isinstance(payload, Mapping):
		raise TypeError(f'{label} must contain a JSON object: {path}')
	return cast('Mapping[str, object]', payload)


def _required_file(path: str | Path, label: str) -> Path:
	resolved = Path(path)
	if not resolved.is_file():
		raise FileNotFoundError(f'{label} does not exist: {resolved}')
	return resolved


def _file_sha256(path: Path) -> str:
	digest = sha256()
	with path.open('rb') as file_obj:
		for chunk in iter(lambda: file_obj.read(_HASH_CHUNK_BYTES), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.relative_to(root)
	except ValueError:
		return False
	return True


def _optional_checkpoint_reference(
	path: Path | None,
	digest: str | None,
) -> dict[str, str] | None:
	if path is None:
		return None
	return {'path': str(path), 'sha256': cast('str', digest)}


__all__ = [
	'FULL_DIR_NAME',
	'MODEL_TAG',
	'SMOKE_DIR_NAME',
	'ParihakaMaeValidationResult',
	'validate_parihaka_mae',
	'validate_parihaka_mae_inputs_from_configs',
	'write_parihaka_mae_validation_report',
]
