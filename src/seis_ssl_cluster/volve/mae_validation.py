'''Fail-closed validation for Volve survey-specific MAE pretraining.'''

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import torch

from seis_ssl_cluster.config import load_config, resolve_mae_training_config
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.training.mae_checkpoint import (
	MaeCheckpointInspection,
	inspect_mae_checkpoint,
)
from seis_ssl_cluster.volve.canonical_inputs import (
	VOLVE_CANONICAL_DATASET_ID,
	VolveCanonicalInputConfig,
	resolve_volve_canonical_input_config,
	validate_volve_canonical_input_registration,
)

MODEL_TAG = 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
SMOKE_DIR_NAME = 'smoke_2step'
FULL_DIR_NAME = 'full_100ep'
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
_FULL_CONTRACT: Mapping[str, object] = {
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
class VolveMaeValidationResult:
	'''Small validation result suitable for stdout or explicit JSON.'''

	check: str
	status: str
	input_config: Path
	smoke_config: Path
	full_config: Path
	canonical_dataset_id: str
	scientific_identity_sha256: str
	manifest: Path
	valid_mask: Path
	smoke_output_root: Path
	full_output_root: Path
	latest_checkpoint: Path | None = None
	latest_sha256: str | None = None
	checkpoint_schema_version: int | None = None
	checkpoint_epoch: int | None = None
	checkpoint_global_step: int | None = None
	resolved_precision: str | None = None
	finite_metric_min: float | None = None
	finite_metric_max: float | None = None

	def to_dict(self) -> dict[str, object]:
		'''Return the intentionally small report payload.'''
		return {
			'check': self.check,
			'status': self.status,
			'input_config': str(self.input_config),
			'smoke_config': str(self.smoke_config),
			'full_config': str(self.full_config),
			'canonical_dataset_id': self.canonical_dataset_id,
			'scientific_identity_sha256': self.scientific_identity_sha256,
			'manifest': str(self.manifest),
			'valid_mask': str(self.valid_mask),
			'smoke_output_root': str(self.smoke_output_root),
			'full_output_root': str(self.full_output_root),
			'latest_checkpoint': _optional_checkpoint_reference(
				self.latest_checkpoint, self.latest_sha256
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
class VolveMaeInputValidation:
	'''Resolved, validated input and training contracts.'''

	input_config: VolveCanonicalInputConfig
	smoke: dict[str, object]
	full: dict[str, object]
	scientific_identity_sha256: str
	manifest_path: Path
	path_list_path: Path
	valid_mask_path: Path
	normalization_stats_path: Path
	canonical_input_metadata_path: Path


def validate_volve_mae(
	*,
	input_config_path: str | Path,
	smoke_config_path: str | Path,
	full_config_path: str | Path,
	check: str,
) -> VolveMaeValidationResult:
	'''Validate Volve inputs and one closed training phase.'''
	if check not in {'inputs', 'smoke', 'full'}:
		raise ValueError(f'check must be inputs, smoke, or full; got {check!r}')
	input_path = _required_file(input_config_path, 'input config')
	smoke_path = _required_file(smoke_config_path, 'smoke config')
	full_path = _required_file(full_config_path, 'full config')
	input_config = resolve_volve_canonical_input_config(load_config(input_path))
	inputs = validate_volve_mae_inputs_from_configs(
		input_config=input_config,
		smoke_raw=load_config(smoke_path),
		full_raw=load_config(full_path),
	)
	base = {
		'check': check,
		'status': 'pass',
		'input_config': input_path,
		'smoke_config': smoke_path,
		'full_config': full_path,
		'canonical_dataset_id': VOLVE_CANONICAL_DATASET_ID,
		'scientific_identity_sha256': inputs.scientific_identity_sha256,
		'manifest': inputs.manifest_path,
		'valid_mask': inputs.valid_mask_path,
		'smoke_output_root': _output_root(inputs.smoke),
		'full_output_root': _output_root(inputs.full),
	}
	if check == 'inputs':
		return VolveMaeValidationResult(**base)
	if check == 'smoke':
		return _validate_smoke(inputs, base=base)
	return _validate_full(inputs, base=base)


def validate_volve_mae_inputs_from_configs(
	*,
	input_config: VolveCanonicalInputConfig,
	smoke_raw: Mapping[str, object],
	full_raw: Mapping[str, object],
) -> VolveMaeInputValidation:
	'''Validate registration identity and already-loaded MAE configs.'''
	_reject_supervision_contract(smoke_raw, label='smoke config')
	_reject_supervision_contract(full_raw, label='full config')
	registration = validate_volve_canonical_input_registration(input_config)
	manifest = registration.manifest
	mask_relative = manifest.amplitude.valid_mask_path
	if mask_relative is None:
		raise ValueError('Volve amplitude manifest requires explicit valid_mask_path')
	valid_mask = _resolve_manifest_path(manifest.root, mask_relative)
	if not valid_mask.is_file():
		raise FileNotFoundError(
			f'Volve explicit valid mask does not exist: {valid_mask}'
		)
	metadata = _read_json_mapping(
		input_config.paths.metadata_path,
		'Volve canonical input metadata',
	)
	_reject_supervision_contract(metadata, label='Volve input metadata')
	_expected_equal(
		metadata.get('scientific_identity_sha256'),
		registration.scientific_identity_sha256,
		'metadata scientific identity SHA-256',
	)
	smoke = resolve_mae_training_config(smoke_raw)
	full = resolve_mae_training_config(full_raw)
	_validate_training_configs(input_config, smoke, full)
	return VolveMaeInputValidation(
		input_config=input_config,
		smoke=smoke,
		full=full,
		scientific_identity_sha256=registration.scientific_identity_sha256,
		manifest_path=input_config.paths.manifest_path,
		path_list_path=input_config.paths.path_list_path,
		valid_mask_path=valid_mask,
		normalization_stats_path=input_config.paths.normalization_stats_path,
		canonical_input_metadata_path=input_config.paths.metadata_path,
	)


def write_volve_mae_validation_report(
	result: VolveMaeValidationResult,
	path: str | Path,
) -> Path:
	'''Write one explicitly requested small JSON report.'''
	report_path = Path(path)
	if report_path.exists() and not report_path.is_file():
		raise ValueError(f'JSON report path exists but is not a file: {report_path}')
	report_path.parent.mkdir(parents=True, exist_ok=True)
	report_path.write_text(
		json.dumps(result.to_dict(), indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)
	return report_path


def _validate_training_configs(
	input_config: VolveCanonicalInputConfig,
	smoke: Mapping[str, object],
	full: Mapping[str, object],
) -> None:
	artifact_root = input_config.artifact_root
	base = artifact_root / 'pretraining' / 'volve' / 'horizon_benchmark_v1' / MODEL_TAG
	for config, expected_output, label in (
		(smoke, base / SMOKE_DIR_NAME, 'smoke'),
		(full, base / FULL_DIR_NAME, 'full'),
	):
		paths = _mapping(config, 'paths', f'{label} config')
		manifests = _mapping(config, 'manifests', f'{label} config')
		_expected_equal(
			paths.get('artifact_root'), str(artifact_root), f'{label} artifact root'
		)
		_expected_equal(
			paths.get('output_root'), str(expected_output), f'{label} output root'
		)
		_expected_equal(
			manifests.get('train'),
			str(input_config.paths.manifest_path),
			f'{label} train manifest',
		)
		_expected_equal(
			manifests.get('train_path_list'),
			str(input_config.paths.path_list_path),
			f'{label} train path list',
		)
		_expected_equal(
			manifests.get('canonical_input_metadata'),
			str(input_config.paths.metadata_path),
			f'{label} canonical input metadata',
		)
		_validate_output_outside_public_root(
			Path(cast('str', paths['output_root'])), input_config.volve_root
		)
	_validate_disjoint_run_roots(_output_root(smoke), _output_root(full))
	differences = _differing_paths(smoke, full)
	unexpected = differences - _SMOKE_FULL_ALLOWED_DIFFERENCES
	if unexpected:
		raise ValueError(
			'smoke/full config differences are outside the closed allowlist; '
			f'unexpected={sorted(unexpected)!r}'
		)
	for section, expected in _FULL_CONTRACT.items():
		_expected_equal(full.get(section), expected, f'full {section} Volve contract')
	train = _mapping(smoke, 'train', 'smoke config')
	for key, expected in (
		('amp', False),
		('amp_dtype', 'auto'),
		('batch_size', 1),
		('device', 'cpu'),
		('epochs', 1),
		('max_steps', 2),
		('samples_per_epoch', 2),
		('seed', 42),
	):
		_expected_equal(train.get(key), expected, f'smoke train.{key}')
	_reject_initial_checkpoint(smoke)
	_reject_initial_checkpoint(full)


def _validate_smoke(
	inputs: VolveMaeInputValidation,
	*,
	base: Mapping[str, object],
) -> VolveMaeValidationResult:
	root = _output_root(inputs.smoke)
	precision = {
		'amp_requested': False,
		'amp_dtype_requested': 'auto',
		'resolved_dtype': 'float32',
		'amp_enabled': False,
		'grad_scaler_enabled': False,
	}
	_validate_run_snapshots(inputs, root, inputs.smoke, precision, label='smoke')
	model = _build_mae_model(_mapping(inputs.smoke, 'model', 'smoke config'))
	latest_path = root / 'latest.pt'
	latest = _inspect_checkpoint(
		latest_path,
		inputs.smoke,
		model=model,
		label='latest smoke',
		expected_epoch=1,
		expected_global_step=2,
		expected_precision='float32',
		expected_amp_enabled=False,
		expected_scaler_enabled=False,
	)
	_validate_checkpoint_forward(model)
	metrics = [value for _, value in latest.metrics]
	return VolveMaeValidationResult(
		**base,
		latest_checkpoint=latest_path,
		latest_sha256=_file_sha256(latest_path),
		checkpoint_schema_version=latest.schema_version,
		checkpoint_epoch=latest.epoch,
		checkpoint_global_step=latest.global_step,
		resolved_precision=latest.resolved_precision,
		finite_metric_min=min(metrics),
		finite_metric_max=max(metrics),
	)


def _validate_full(
	inputs: VolveMaeInputValidation,
	*,
	base: Mapping[str, object],
) -> VolveMaeValidationResult:
	root = _output_root(inputs.full)
	precision = _load_full_precision_contract(root, inputs.full)
	resolved_precision = cast('str', precision['resolved_dtype'])
	amp_enabled = cast('bool', precision['amp_enabled'])
	scaler_enabled = cast('bool', precision['grad_scaler_enabled'])
	_validate_run_snapshots(inputs, root, inputs.full, precision, label='full')
	model = _build_mae_model(_mapping(inputs.full, 'model', 'full config'))
	latest_path = root / 'latest.pt'
	latest = _inspect_checkpoint(
		latest_path,
		inputs.full,
		model=model,
		label='latest full',
		expected_epoch=100,
		expected_global_step=250_000,
		expected_precision=resolved_precision,
		expected_amp_enabled=amp_enabled,
		expected_scaler_enabled=scaler_enabled,
	)
	metrics = [value for _, value in latest.metrics]
	return VolveMaeValidationResult(
		**base,
		latest_checkpoint=latest_path,
		latest_sha256=_file_sha256(latest_path),
		checkpoint_schema_version=latest.schema_version,
		checkpoint_epoch=latest.epoch,
		checkpoint_global_step=latest.global_step,
		resolved_precision=latest.resolved_precision,
		finite_metric_min=min(metrics),
		finite_metric_max=max(metrics),
	)


def _inspect_checkpoint(  # noqa: PLR0913
	path: Path,
	config: Mapping[str, object],
	*,
	model: AmplitudeMAE3D,
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
		resolved_config=config,
		model=model,
		resolved_precision=expected_precision,
		amp_enabled=expected_amp_enabled,
		scaler_present=expected_scaler_enabled,
	)
	for actual, expected, field in (
		(inspection.schema_version, 2, 'schema_version'),
		(inspection.stage, 'train_amp_mae', 'stage'),
		(inspection.checkpoint_kind, 'epoch', 'checkpoint_kind'),
		(inspection.batch_index, None, 'batch_index'),
		(inspection.resolved_precision, expected_precision, 'resolved_precision'),
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


def _validate_checkpoint_forward(model: AmplitudeMAE3D) -> None:
	patch = model.patch_size_xyz
	x = torch.zeros((1, model.in_channels, *patch), dtype=torch.float32)
	valid = torch.ones((1, *patch), dtype=torch.bool)
	model.eval()
	with torch.inference_mode():
		encoded = model.encode_tokens(x, valid_mask=valid)['tokens']
	if not isinstance(encoded, torch.Tensor) or not bool(torch.isfinite(encoded).all()):
		raise ValueError('latest smoke checkpoint forward produced non-finite tokens')


def _validate_run_snapshots(
	inputs: VolveMaeInputValidation,
	output_root: Path,
	config: Mapping[str, object],
	expected_precision: Mapping[str, object],
	*,
	label: str,
) -> None:
	resolved_path = output_root / 'resolved_config.json'
	manifest_snapshot = output_root / 'manifest.json'
	path_list_snapshot = output_root / 'inputs' / inputs.path_list_path.name
	normalization_snapshot = (
		output_root / 'inputs' / inputs.normalization_stats_path.name
	)
	canonical_metadata_snapshot = (
		output_root / 'inputs' / inputs.canonical_input_metadata_path.name
	)
	metadata_path = output_root / 'run_metadata.json'
	for path in (
		resolved_path,
		manifest_snapshot,
		path_list_snapshot,
		normalization_snapshot,
		canonical_metadata_snapshot,
		metadata_path,
	):
		if not path.is_file():
			raise FileNotFoundError(f'{label} run snapshot does not exist: {path}')
	expected_resolved = (
		json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + '\n'
	).encode()
	_expected_equal(
		resolved_path.read_bytes(), expected_resolved, f'{label} config snapshot'
	)
	_expected_equal(
		manifest_snapshot.read_bytes(),
		inputs.manifest_path.read_bytes(),
		f'{label} manifest snapshot',
	)
	_expected_equal(
		path_list_snapshot.read_bytes(),
		inputs.path_list_path.read_bytes(),
		f'{label} path-list snapshot',
	)
	_expected_equal(
		normalization_snapshot.read_bytes(),
		inputs.normalization_stats_path.read_bytes(),
		f'{label} normalization stats snapshot',
	)
	_expected_equal(
		canonical_metadata_snapshot.read_bytes(),
		inputs.canonical_input_metadata_path.read_bytes(),
		f'{label} canonical input metadata snapshot',
	)
	snapshot_metadata = _read_json_mapping(
		canonical_metadata_snapshot,
		f'{label} canonical input metadata snapshot',
	)
	_expected_equal(
		snapshot_metadata.get('scientific_identity_sha256'),
		inputs.scientific_identity_sha256,
		f'{label} snapshot scientific identity SHA-256',
	)
	_reject_supervision_contract(
		_read_json_value(resolved_path, f'{label} resolved config'),
		label=f'{label} resolved config',
	)
	_reject_supervision_contract(
		_read_json_value(manifest_snapshot, f'{label} manifest snapshot'),
		label=f'{label} manifest snapshot',
	)
	metadata = _read_json_mapping(metadata_path, f'{label} run metadata')
	_expected_equal(
		metadata.get('runtime_check_mode'), 'once', f'{label} runtime mode'
	)
	_expected_equal(
		metadata.get('precision'), dict(expected_precision), f'{label} precision'
	)
	for key, expected in (
		('input_scientific_identity_sha256', inputs.scientific_identity_sha256),
		('normalization_stats_sha256', _file_sha256(normalization_snapshot)),
		(
			'canonical_input_metadata_sha256',
			_file_sha256(canonical_metadata_snapshot),
		),
	):
		_expected_equal(metadata.get(key), expected, f'{label} run metadata {key}')
	_reject_supervision_contract(metadata, label=f'{label} run metadata')


def _load_full_precision_contract(
	root: Path,
	full: Mapping[str, object],
) -> dict[str, object]:
	metadata = _read_json_mapping(root / 'run_metadata.json', 'full run metadata')
	precision = _mapping(metadata, 'precision', 'full run metadata')
	_expected_equal(
		set(precision),
		{
			'amp_requested',
			'amp_dtype_requested',
			'resolved_dtype',
			'amp_enabled',
			'grad_scaler_enabled',
		},
		'full precision fields',
	)
	train = _mapping(full, 'train', 'full config')
	_expected_equal(
		precision.get('amp_requested'), train.get('amp'), 'full requested AMP'
	)
	_expected_equal(
		precision.get('amp_dtype_requested'),
		train.get('amp_dtype'),
		'full requested AMP dtype',
	)
	resolved = precision.get('resolved_dtype')
	if (
		resolved not in {'bfloat16', 'float16'}
		or precision.get('amp_enabled') is not True
	):
		raise ValueError('full run must record enabled CUDA AMP with a reduced dtype')
	expected_scaler = resolved == 'float16'
	_expected_equal(
		precision.get('grad_scaler_enabled'), expected_scaler, 'full scaler contract'
	)
	return dict(precision)


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


def _reject_supervision_contract(value: object, *, label: str) -> None:
	for path, key, child in _walk_mapping(value):
		normalized = key.lower()
		parts = set(normalized.replace('-', '_').split('_'))
		if parts & {'label', 'labels', 'fault', 'faults', 'layout', 'layouts'}:
			raise ValueError(f'{label} contains forbidden supervision field: {path}')
		if 'horizon' in parts or 'horizons' in parts:
			raise ValueError(f'{label} contains forbidden horizon field: {path}')
		if isinstance(child, str) and ('/' in child or '\\' in child):
			name = Path(child).name.lower()
			if any(token in name for token in ('horizon', 'fault', 'layout', 'label')):
				raise ValueError(f'{label} contains forbidden supervision path: {path}')


def _reject_initial_checkpoint(config: Mapping[str, object]) -> None:
	for path, key, _child in _walk_mapping(config):
		if key in {'checkpoint', 'init_checkpoint', 'teacher', 'teacher_checkpoint'}:
			raise ValueError(
				f'Volve MAE must use seed-42 random initialization; found {path}'
			)


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


def _validate_disjoint_run_roots(smoke: Path, full: Path) -> None:
	if (
		smoke.resolve(strict=False) == full.resolve(strict=False)
		or _is_relative_to(smoke.resolve(strict=False), full.resolve(strict=False))
		or _is_relative_to(full.resolve(strict=False), smoke.resolve(strict=False))
	):
		raise ValueError(
			f'smoke and full output roots must be disjoint: {smoke}, {full}'
		)


def _validate_output_outside_public_root(output: Path, public_root: Path) -> None:
	if _is_relative_to(
		output.resolve(strict=False), public_root.resolve(strict=False)
	):
		raise ValueError(
			f'MAE output must not be under read-only public data: {output}'
		)


def _resolve_manifest_path(root: Path, value: Path) -> Path:
	return value if value.is_absolute() else root / value


def _output_root(config: Mapping[str, object]) -> Path:
	paths = _mapping(config, 'paths', 'config')
	return Path(cast('str', paths['output_root']))


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


def _read_json_value(path: Path, label: str) -> object:
	if not path.is_file():
		raise FileNotFoundError(f'{label} does not exist: {path}')
	try:
		return json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(f'{label} is invalid JSON: {path}') from exc


def _read_json_mapping(path: Path, label: str) -> Mapping[str, object]:
	payload = _read_json_value(path, label)
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
		for chunk in iter(lambda: file_obj.read(8 * 1024 * 1024), b''):
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
	'VolveMaeInputValidation',
	'VolveMaeValidationResult',
	'validate_volve_mae',
	'validate_volve_mae_inputs_from_configs',
	'write_volve_mae_validation_report',
]
