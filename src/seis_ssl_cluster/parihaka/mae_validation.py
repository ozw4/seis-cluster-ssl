# ruff: noqa: CPY001
"""Fail-closed validation for Parihaka MAE inputs and CPU smoke artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
import torch

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
from seis_ssl_cluster.training.checkpoint import load_checkpoint
from seis_ssl_cluster.training.mae_checkpoint import (
	_best_metric_from_metrics,
	_validate_resume_payload,
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
_STABLE_NOPIMS_FULL = (
	Path(__file__).resolve().parents[3]
	/ 'experiments'
	/ 'nopims'
	/ 'pretrain_v1'
	/ '10_pretrain'
	/ MODEL_TAG
	/ '03_full_100ep.yaml'
)


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
	"""Validate Parihaka input identity and, optionally, the smoke run."""
	if check not in {'inputs', 'smoke'}:
		msg = f'check must be inputs or smoke; got {check!r}'
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
	return _validate_smoke(inputs, base=base)


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
	if differences != _SMOKE_FULL_ALLOWED_DIFFERENCES:
		msg = (
			'smoke/full config differences must equal the closed allowlist; '
			f'expected={sorted(_SMOKE_FULL_ALLOWED_DIFFERENCES)!r}, '
			f'got={sorted(differences)!r}'
		)
		raise ValueError(msg)
	_validate_full_contract(full)
	_validate_smoke_contract(smoke)
	_validate_nopims_scientific_parity(full)
	_reject_initial_checkpoint(full)


def _validate_full_contract(full: Mapping[str, object]) -> None:
	train = _mapping(full, 'train', 'full config')
	for key, expected in (
		('amp', True),
		('amp_dtype', 'auto'),
		('device', 'cuda'),
		('seed', 42),
		('epochs', 100),
		('samples_per_epoch', 10_000),
	):
		_expected_equal(train.get(key), expected, f'full train.{key}')


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


def _validate_nopims_scientific_parity(full: Mapping[str, object]) -> None:
	stable = resolve_mae_training_config(load_config(_STABLE_NOPIMS_FULL))
	for section in ('data', 'zero_mask', 'model', 'masking', 'loss', 'visualization'):
		_expected_equal(
			full.get(section), stable.get(section), f'full {section} stable parity'
		)
	full_train = dict(_mapping(full, 'train', 'full config'))
	stable_train = dict(_mapping(stable, 'train', 'stable NOPIMS config'))
	stable_train['amp'] = True
	_expected_equal(full_train, stable_train, 'full train stable parity')


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
	latest = _validate_checkpoint(latest_path, inputs.smoke, label='latest')
	best = _validate_checkpoint(best_path, inputs.smoke, label='best')
	_validate_run_snapshots(inputs, smoke_root)
	best_metrics = _mapping(best, 'metrics', 'best checkpoint')
	metric_key, best_metric = _best_metric_from_metrics(
		cast('Mapping[str, float]', best_metrics)
	)
	if metric_key is None or best_metric is None or not math.isfinite(best_metric):
		raise ValueError('best checkpoint must contain a finite existing best metric')
	metric_values = _finite_metric_values(latest, 'latest') + _finite_metric_values(
		best, 'best'
	)
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
	)


def _validate_checkpoint(
	path: Path,
	resolved_smoke: Mapping[str, object],
	*,
	label: str,
) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(f'{label} smoke checkpoint does not exist: {path}')
	payload = cast('Mapping[str, object]', load_checkpoint(path, map_location='cpu'))
	_validate_resume_payload(
		payload,
		amp_enabled=False,
		scaler_required=False,
		resolved_precision='float32',
	)
	training_state = _mapping(payload, 'training_state', f'{label} checkpoint')
	for actual, expected, field in (
		(training_state.get('schema_version'), 2, 'training_state.schema_version'),
		(training_state.get('stage'), 'train_amp_mae', 'training_state.stage'),
		(
			training_state.get('checkpoint_kind'),
			'epoch',
			'training_state.checkpoint_kind',
		),
		(training_state.get('batch_index'), None, 'training_state.batch_index'),
		(
			training_state.get('resolved_precision'),
			'float32',
			'training_state.resolved_precision',
		),
		(payload.get('epoch'), 1, 'epoch'),
		(payload.get('global_step'), 2, 'global_step'),
		(payload.get('amp_enabled'), False, 'amp_enabled'),
	):
		_expected_equal(actual, expected, f'{label} checkpoint {field}')
	_expected_equal(payload.get('config'), resolved_smoke, f'{label} checkpoint config')
	model = _build_mae_model(_mapping(resolved_smoke, 'model', 'smoke config'))
	state = _mapping(payload, 'model_state_dict', f'{label} checkpoint')
	try:
		model.load_state_dict(state, strict=True)
	except RuntimeError as exc:
		msg = f'{label} checkpoint model geometry/state mismatch: {path}: {exc}'
		raise ValueError(msg) from exc
	_require_finite_tree(state, f'{label} checkpoint model_state_dict')
	_require_finite_tree(
		_mapping(payload, 'optimizer_state_dict', f'{label} checkpoint'),
		f'{label} checkpoint optimizer_state_dict',
	)
	_finite_metric_values(payload, label)
	return payload


def _validate_run_snapshots(inputs: _InputValidation, smoke_root: Path) -> None:
	resolved_path = smoke_root / 'resolved_config.json'
	manifest_snapshot = smoke_root / 'manifest.json'
	path_list_snapshot = smoke_root / 'inputs' / inputs.prepare.outputs.path_list.name
	run_metadata_path = smoke_root / 'run_metadata.json'
	for path in (
		resolved_path,
		manifest_snapshot,
		path_list_snapshot,
		run_metadata_path,
	):
		if not path.is_file():
			raise FileNotFoundError(f'smoke run snapshot does not exist: {path}')
	expected_resolved = (
		json.dumps(inputs.smoke, indent=2, sort_keys=True, allow_nan=False) + '\n'
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
	run_metadata = _read_json_mapping(run_metadata_path, 'smoke run metadata')
	_expected_equal(
		run_metadata.get('runtime_check_mode'),
		'once',
		'run metadata runtime_check_mode',
	)
	_expected_equal(
		run_metadata.get('precision'),
		{
			'amp_requested': False,
			'amp_dtype_requested': 'auto',
			'resolved_dtype': 'float32',
			'amp_enabled': False,
			'grad_scaler_enabled': False,
		},
		'run metadata precision',
	)


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


def _finite_metric_values(payload: Mapping[str, object], label: str) -> list[float]:
	metrics = _mapping(payload, 'metrics', f'{label} checkpoint')
	values: list[float] = []
	for key, value in metrics.items():
		if isinstance(value, bool) or not isinstance(value, int | float):
			raise TypeError(f'{label} checkpoint metric {key} must be numeric')
		floating = float(value)
		if not math.isfinite(floating):
			raise ValueError(f'{label} checkpoint metric {key} must be finite')
		values.append(floating)
	if not values:
		raise ValueError(f'{label} checkpoint metrics must not be empty')
	return values


def _require_finite_tree(value: object, label: str) -> None:
	if isinstance(value, torch.Tensor):
		if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(
			torch.isfinite(value).all(),
		):
			raise ValueError(f'{label} contains a nonfinite tensor')
		return
	if isinstance(value, Mapping):
		for key, child in value.items():
			_require_finite_tree(child, f'{label}.{key}')
		return
	if isinstance(value, Sequence) and not isinstance(value, str | bytes):
		for index, child in enumerate(value):
			_require_finite_tree(child, f'{label}[{index}]')


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
