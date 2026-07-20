"""Runtime helpers for stratigraphic HMM pretext training."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

import seis_ssl_cluster
from seis_ssl_cluster.data import ZeroMaskConfig
from seis_ssl_cluster.stratigraphy import discover_pseudo_target_inputs
from seis_ssl_cluster.training.checkpoint import capture_rng_state, load_checkpoint

if TYPE_CHECKING:
	from seis_ssl_cluster.training.strat_hmm.state import TrainabilitySummary


def _rng_state_for_step_checkpoint(
	*,
	dataloader: torch.utils.data.DataLoader,
	epoch_start_dataloader_rng_state: torch.Tensor,
	batch_index: int,
) -> dict[str, object]:
	if batch_index >= len(dataloader) - 1:
		return _rng_state_with_dataloader(dataloader)
	return _rng_state_with_dataloader(
		dataloader,
		dataloader_generator_state=epoch_start_dataloader_rng_state,
	)


def _rng_state_with_dataloader(
	dataloader: torch.utils.data.DataLoader,
	*,
	dataloader_generator_state: torch.Tensor | None = None,
) -> dict[str, object]:
	rng_state = capture_rng_state()
	rng_state['dataloader_generator'] = (
		_dataloader_generator_state(dataloader)
		if dataloader_generator_state is None
		else dataloader_generator_state.clone()
	)
	return rng_state


def _dataloader_generator_state(
	dataloader: torch.utils.data.DataLoader,
) -> torch.Tensor:
	generator = getattr(dataloader, 'generator', None)
	if not isinstance(generator, torch.Generator):
		msg = 'strat HMM dataloader must expose a torch.Generator for resume'
		raise TypeError(msg)
	return generator.get_state().clone()


def _restore_dataloader_generator_state(
	*,
	payload: Mapping[str, object],
	dataloader: torch.utils.data.DataLoader,
) -> None:
	rng_state = payload['rng_state']
	if not isinstance(rng_state, Mapping):
		msg = 'resume checkpoint rng_state must be a mapping'
		raise TypeError(msg)
	generator_state = rng_state['dataloader_generator']
	if not isinstance(generator_state, torch.Tensor):
		msg = 'resume checkpoint rng_state.dataloader_generator must be a tensor'
		raise TypeError(msg)
	generator = getattr(dataloader, 'generator', None)
	if not isinstance(generator, torch.Generator):
		msg = 'strat HMM dataloader must expose a torch.Generator for resume'
		raise TypeError(msg)
	generator.set_state(generator_state.cpu())


def _load_existing_best_score(output_root: Path) -> float | None:
	best_path = output_root / 'best.pt'
	if not best_path.is_file():
		return None
	payload = load_checkpoint(best_path, map_location='cpu')
	metrics = payload.get('metrics')
	if not isinstance(metrics, Mapping):
		return None
	loss = metrics.get('loss')
	if isinstance(loss, int | float) and not isinstance(loss, bool):
		score = float(loss)
		if math.isfinite(score):
			return score
	return None


def _zero_mask_from_config(config: Mapping[str, object]) -> ZeroMaskConfig:
	value = _mapping(config, 'zero_mask')
	zero_mask = ZeroMaskConfig(
		enabled=_bool_config(value, 'enabled', default=True),
		zero_atol=_float_config(value, 'zero_atol', 0.0),
		z_sample_influence_radius=_int_config(
			value,
			'z_sample_influence_radius',
			16,
		),
		xy_trace_influence_radius=_int_config(
			value,
			'xy_trace_influence_radius',
			1,
		),
	)
	zero_mask.validate()
	return zero_mask


def _snapshot_run_inputs(
	*,
	output_root: Path,
	config: Mapping[str, object],
	control_identity: Mapping[str, object] | None = None,
	overwrite: bool = False,
) -> None:
	_write_json(
		output_root / 'resolved_config.json',
		_to_json_safe(config),
		overwrite=overwrite,
	)
	_write_json(
		output_root / 'run_metadata.json',
		_run_metadata_payload(control_identity=control_identity),
		overwrite=overwrite,
	)


def _write_run_metadata(
	*,
	output_root: Path,
	trainability_summary: TrainabilitySummary,
	control_identity: Mapping[str, object] | None = None,
	overwrite: bool,
) -> None:
	payload = _run_metadata_payload(control_identity=control_identity)
	payload['trainability_summary'] = _trainability_summary_payload(
		trainability_summary,
	)
	_write_json(output_root / 'run_metadata.json', payload, overwrite=overwrite)


def _run_metadata_payload(
	*, control_identity: Mapping[str, object] | None = None
) -> dict[str, object]:
	payload: dict[str, object] = {
		'created_at_utc': datetime.now(timezone.utc).isoformat(),
		'git_commit': _git_commit(),
		'package_version': getattr(seis_ssl_cluster, '__version__', None),
	}
	if control_identity is not None:
		payload['control_identity'] = _to_json_safe(control_identity)
	return payload


def _strat_hmm_control_identity(
	config: Mapping[str, object],
) -> dict[str, object] | None:
	"""Build immutable provenance for an explicitly identified control run."""
	identity = config.get('identity')
	if identity is None:
		return None
	if not isinstance(identity, Mapping):
		raise TypeError('identity must be a mapping')
	model_tag = _non_empty_string(identity.get('model_tag'), 'identity.model_tag')
	teacher = _mapping(config, 'teacher')
	student = _mapping(config, 'student')
	pseudo_targets = _mapping(config, 'pseudo_targets')
	teacher_path = _path_config(teacher, 'checkpoint')
	student_path = Path(
		student.get('init_checkpoint') or str(teacher_path)
	)
	pseudo_root = _path_config(pseudo_targets, 'input_dir')
	pseudo_inputs = discover_pseudo_target_inputs(
		pseudo_root,
		k=_int_config(pseudo_targets, 'k', 1),
	)
	if not pseudo_inputs:
		raise ValueError('pseudo-target identity requires at least one survey input')
	pseudo_identity: list[dict[str, object]] = []
	for item in pseudo_inputs:
		entry: dict[str, object] = {
			'survey_id': item.survey_id,
			'labels': _file_identity(item.labels_path),
			'confidence': _file_identity(item.confidence_path),
			'valid_tokens': _file_identity(item.valid_tokens_path),
			'metadata': _file_identity(item.metadata_path),
			'boundary_weight_present': item.boundary_weight_path is not None,
		}
		if item.boundary_weight_path is not None:
			entry['boundary_weight'] = _file_identity(item.boundary_weight_path)
		pseudo_identity.append(entry)
	return {
		'schema_version': 1,
		'model_tag': model_tag,
		'scientific_identity': _to_json_safe(
			identity.get('scientific_identity', {}),
		),
		'runtime_identity': {
			**_to_mapping(identity.get('runtime_identity', {}), 'runtime_identity'),
			'git_commit': _git_commit(),
			'git_status_short': _git_status_short(),
			'git_diff_sha256': _git_diff_sha256(),
			'finite_check_mode': _mapping(config, 'data').get(
				'finite_check_mode', 'strict'
			),
		},
		'resolved_training_config_sha256': _canonical_sha256(config),
		'input_identities': {
			'teacher_checkpoint': _file_identity(teacher_path),
			'student_init_checkpoint': _file_identity(student_path),
			'pseudo_targets': pseudo_identity,
		},
	}


def _to_mapping(value: object, label: str) -> dict[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'identity.{label} must be a mapping')
	return {str(key): _to_json_safe(child) for key, child in value.items()}


def _file_identity(path: Path) -> dict[str, str]:
	if not path.is_file():
		raise FileNotFoundError(path)
	return {'path': str(path), 'sha256': _file_sha256(path)}


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, object]) -> str:
	payload = json.dumps(
		_to_json_safe(value), sort_keys=True, separators=(',', ':'), allow_nan=False
	).encode('utf-8')
	return hashlib.sha256(payload).hexdigest()


def _git_status_short() -> list[str]:
	git = shutil.which('git')
	if git is None:
		return []
	try:
		output = subprocess.check_output(  # noqa: S603
			[git, 'status', '--short'],
			cwd=Path(__file__).resolve().parents[3],
			text=True,
			stderr=subprocess.DEVNULL,
		)
	except (OSError, subprocess.CalledProcessError):
		return []
	return output.splitlines()


def _git_diff_sha256() -> str | None:
	git = shutil.which('git')
	if git is None:
		return None
	try:
		output = subprocess.check_output(  # noqa: S603
			[git, 'diff', '--binary', 'HEAD'],
			cwd=Path(__file__).resolve().parents[3],
			stderr=subprocess.DEVNULL,
		)
	except (OSError, subprocess.CalledProcessError):
		return None
	return hashlib.sha256(output).hexdigest()


def _write_json(path: Path, payload: object, *, overwrite: bool = False) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	if path.exists() and not overwrite:
		return
	text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
	path.write_text(f'{text}\n', encoding='utf-8')


def _git_commit() -> str | None:
	git = shutil.which('git')
	if git is None:
		return None
	try:
		return subprocess.check_output(  # noqa: S603
			[git, 'rev-parse', 'HEAD'],
			cwd=Path(__file__).resolve().parents[3],
			text=True,
			stderr=subprocess.DEVNULL,
		).strip()
	except (OSError, subprocess.CalledProcessError):
		return None


def _resolve_device(train_config: Mapping[str, object]) -> torch.device:
	device_name = train_config.get('device')
	if device_name is None or device_name == 'auto':
		return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if not isinstance(device_name, str):
		msg = f'train.device must be a string; got {device_name!r}'
		raise TypeError(msg)
	if device_name not in {'cpu', 'cuda'}:
		msg = 'train.device must be "auto", "cpu", or "cuda"'
		raise ValueError(msg)
	device = torch.device(device_name)
	if device.type == 'cuda' and not torch.cuda.is_available():
		msg = 'train.device requested CUDA, but CUDA is not available'
		raise ValueError(msg)
	return device


def _required_tensor(batch: Mapping[str, object], key: str) -> torch.Tensor:
	value = batch.get(key)
	if not isinstance(value, torch.Tensor):
		msg = f'batch key {key!r} must be a tensor'
		raise TypeError(msg)
	return value


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _non_empty_string(value: object, name: str) -> str:
	if not isinstance(value, str) or not value:
		msg = f'{name} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _path_config(config: Mapping[str, object], key: str) -> Path:
	return Path(_non_empty_string(config.get(key), key))


def _int_config(config: Mapping[str, object], key: str, default: int) -> int:
	value = config.get(key, default)
	if isinstance(value, bool) or not isinstance(value, int):
		msg = f'{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	return int(value)


def _optional_int_config(config: Mapping[str, object], key: str) -> int | None:
	value = config.get(key)
	if value is None:
		return None
	if isinstance(value, bool) or not isinstance(value, int):
		msg = f'{key} must be an integer or None; got {value!r}'
		raise TypeError(msg)
	return int(value)


def _optional_positive_int_config(
	config: Mapping[str, object],
	key: str,
) -> int | None:
	value = _optional_int_config(config, key)
	if value is not None and value <= 0:
		msg = f'{key} must be a positive integer or None; got {value!r}'
		raise ValueError(msg)
	return value


def _float_config(config: Mapping[str, object], key: str, default: float) -> float:
	value = config.get(key, default)
	if isinstance(value, bool) or not isinstance(value, int | float):
		msg = f'{key} must be a float; got {value!r}'
		raise TypeError(msg)
	result = float(value)
	if not math.isfinite(result):
		msg = f'{key} must be finite; got {value!r}'
		raise ValueError(msg)
	return result


def _optional_float_config(
	config: Mapping[str, object],
	key: str,
) -> float | None:
	value = config.get(key)
	if value is None:
		return None
	return _float_config(config, key, 0.0)


def _bool_config(
	config: Mapping[str, object],
	key: str,
	*,
	default: bool,
) -> bool:
	value = config.get(key, default)
	if not isinstance(value, bool):
		msg = f'{key} must be a boolean; got {value!r}'
		raise TypeError(msg)
	return value


def _xyz_config(
	config: Mapping[str, object],
	key: str,
) -> tuple[int, int, int]:
	value = config.get(key)
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str)
		or len(value) != 3
		or any(isinstance(axis, bool) or not isinstance(axis, int) for axis in value)
	):
		msg = f'{key} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = tuple(int(axis) for axis in value)
	if any(axis <= 0 for axis in xyz):
		msg = f'{key} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return cast('tuple[int, int, int]', xyz)


def _trainability_summary_payload(
	summary: TrainabilitySummary,
) -> dict[str, object]:
	return {
		'trainable_parameter_count': summary.trainable_parameter_count,
		'frozen_parameter_count': summary.frozen_parameter_count,
		'trainable_names': list(summary.trainable_names),
	}


def _to_json_safe(value: object) -> object:
	if isinstance(value, Mapping):
		return {str(key): _to_json_safe(child) for key, child in value.items()}
	if isinstance(value, list | tuple):
		return [_to_json_safe(child) for child in value]
	if isinstance(value, Path):
		return str(value)
	return value


__all__ = [
	'_bool_config',
	'_dataloader_generator_state',
	'_float_config',
	'_int_config',
	'_load_existing_best_score',
	'_mapping',
	'_non_empty_string',
	'_optional_float_config',
	'_optional_int_config',
	'_optional_positive_int_config',
	'_path_config',
	'_required_tensor',
	'_resolve_device',
	'_restore_dataloader_generator_state',
	'_rng_state_for_step_checkpoint',
	'_rng_state_with_dataloader',
	'_snapshot_run_inputs',
	'_strat_hmm_control_identity',
	'_to_json_safe',
	'_trainability_summary_payload',
	'_write_run_metadata',
	'_xyz_config',
	'_zero_mask_from_config',
]
