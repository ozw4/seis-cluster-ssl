'''Shared training lifecycle for the Volve five-horizon benchmarks.'''

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import nn

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.volve.horizon_data import HORIZON_NAMES
from seis_ssl_cluster.volve.horizon_metrics import compute_horizon_metrics

if TYPE_CHECKING:
	from typing import Any

	from torch.utils.data import Dataset

LATEST_NAME = 'latest.pt'
BEST_NAME = 'best.pt'
METRICS_NAME = 'metrics.json'
HISTORY_NAME = 'history.json'
CHECKPOINT_SELECTION_VALIDATION_MAE = (
	'strict_lower_validation_macro_mae_v1'
)
CHECKPOINT_SELECTION_VALIDATION_WITHIN_2 = (
	'strict_higher_validation_macro_within_2_v1'
)
CHECKPOINT_SELECTION_IDS = frozenset(
	{
		CHECKPOINT_SELECTION_VALIDATION_MAE,
		CHECKPOINT_SELECTION_VALIDATION_WITHIN_2,
	}
)


@dataclass(frozen=True)
class HorizonRunnerSettings:
	'''Execution settings shared by frozen and end-to-end horizon jobs.'''

	epochs: int
	seed: int
	amp_on_cuda: bool
	gradient_clip_norm: float
	checkpoint_selection: str = CHECKPOINT_SELECTION_VALIDATION_MAE


@dataclass(frozen=True)
class HorizonRuntimeContext:
	'''Resolved precision and optimizer-step state passed to regime callables.'''

	device: torch.device
	amp_enabled: bool
	scaler: torch.amp.GradScaler | None
	gradient_clip_norm: float


BuildModelAndOptimizer = Callable[
	[torch.device], tuple[nn.Module, torch.optim.Optimizer]
]
TrainOneItem = Callable[
	[
		nn.Module,
		Mapping[str, object],
		torch.optim.Optimizer,
		HorizonRuntimeContext,
	],
	float,
]
PredictOneItem = Callable[
	[nn.Module, Mapping[str, object], HorizonRuntimeContext], torch.Tensor
]


def run_horizon_training_job(  # noqa: C901, PLR0912, PLR0913, PLR0915
	*,
	output_dir: Path,
	run_identity: Mapping[str, object],
	settings: HorizonRunnerSettings,
	datasets: Mapping[str, Dataset[dict[str, Any]]],
	expected_counts: Mapping[str, Sequence[int]],
	metrics_metadata: Mapping[str, object],
	build_model_and_optimizer: BuildModelAndOptimizer,
	train_one_item: TrainOneItem,
	predict_one_item: PredictOneItem,
	device: str = 'auto',
	max_steps: int | None = None,
	resume: str | Path | None = None,
) -> Path | None:
	'''Run the lifecycle shared by the frozen and end-to-end regimes.'''
	_validate_runner_inputs(
		settings=settings,
		datasets=datasets,
		expected_counts=expected_counts,
		max_steps=max_steps,
	)
	resume_path = None if resume is None else Path(resume)
	_validate_output(output_dir, resume_path)
	run_device = resolve_horizon_device(device)
	runtime_precision = horizon_runtime_precision_identity(
		run_device, amp_requested=settings.amp_on_cuda
	)
	runtime_run_identity = {
		**run_identity,
		'runtime_precision': runtime_precision,
	}
	_configure_determinism()
	_seed_everything(settings.seed)
	model, optimizer = build_model_and_optimizer(run_device)
	if not any(parameter.requires_grad for parameter in model.parameters()):
		raise ValueError('horizon model must have trainable parameters')
	amp_enabled = bool(runtime_precision['amp_enabled'])
	scaler = torch.amp.GradScaler('cuda', enabled=True) if amp_enabled else None
	runtime = HorizonRuntimeContext(
		device=run_device,
		amp_enabled=amp_enabled,
		scaler=scaler,
		gradient_clip_norm=settings.gradient_clip_norm,
	)
	history: list[dict[str, object]] = []
	best_epoch: int | None = None
	best_score = initial_best_validation_score(settings.checkpoint_selection)
	global_step = 0
	start_epoch = 0
	start_position = 0
	train_loss_sum = 0.0
	train_item_count = 0
	if resume_path is not None:
		payload = torch.load(resume_path, map_location=run_device, weights_only=False)
		validate_horizon_resume_runtime(
			payload,
			expected=runtime_precision,
			scaler=scaler,
		)
		if payload.get('run_identity') != runtime_run_identity:
			raise ValueError('resume checkpoint does not match this horizon job')
		if payload.get('completed') is True:
			raise ValueError('completed horizon job cannot be resumed')
		model.load_state_dict(_state_dict(payload))
		optimizer.load_state_dict(_mapping(payload, 'optimizer_state_dict'))
		if scaler is not None:
			scaler.load_state_dict(_mapping(payload, 'scaler_state_dict'))
		history = [
			dict(cast('Mapping[str, object]', row))
			for row in _sequence(payload.get('history'), 'history')
		]
		best_epoch = _optional_int(payload.get('best_epoch'), 'best_epoch')
		best_score = _resume_best_validation_score(
			payload,
			selection=settings.checkpoint_selection,
		)
		global_step = _nonnegative_int(payload.get('global_step'), 'global_step')
		start_epoch = _nonnegative_int(payload.get('epoch'), 'epoch')
		start_position = _nonnegative_int(
			payload.get('next_position'), 'next_position'
		)
		train_loss_sum = _finite_number(
			payload.get('train_loss_sum'), 'train_loss_sum'
		)
		train_item_count = _nonnegative_int(
			payload.get('train_item_count', payload.get('train_tile_count')),
			'train_item_count',
		)
	output_dir.mkdir(parents=True, exist_ok=True)
	for epoch in range(start_epoch, settings.epochs):
		order = deterministic_tile_order(
			len(datasets['train']), settings.seed, epoch
		)
		for position in range(start_position, len(order)):
			if max_steps is not None and global_step >= max_steps:
				_save_latest(
					output_dir,
					model,
					optimizer,
					scaler,
					history=history,
					best_epoch=best_epoch,
					checkpoint_selection=settings.checkpoint_selection,
					best_score=best_score,
					global_step=global_step,
					epoch=epoch,
					next_position=position,
					train_loss_sum=train_loss_sum,
					train_item_count=train_item_count,
					completed=False,
					runtime_precision=runtime_precision,
					run_identity=runtime_run_identity,
				)
				_write_json(output_dir / HISTORY_NAME, history)
				return None
			loss_value = train_one_item(
				model,
				datasets['train'][order[position]],
				optimizer,
				runtime,
			)
			if not math.isfinite(loss_value):
				raise FloatingPointError('horizon training loss is non-finite')
			train_loss_sum += loss_value
			train_item_count += 1
			global_step += 1
			if (
				max_steps is not None
				and global_step >= max_steps
				and position + 1 < len(order)
			):
				_save_latest(
					output_dir,
					model,
					optimizer,
					scaler,
					history=history,
					best_epoch=best_epoch,
					checkpoint_selection=settings.checkpoint_selection,
					best_score=best_score,
					global_step=global_step,
					epoch=epoch,
					next_position=position + 1,
					train_loss_sum=train_loss_sum,
					train_item_count=train_item_count,
					completed=False,
					runtime_precision=runtime_precision,
					run_identity=runtime_run_identity,
				)
				_write_json(output_dir / HISTORY_NAME, history)
				return None
		validation = evaluate_horizon_dataset(
			model,
			datasets['validation'],
			runtime,
			predict_one_item=predict_one_item,
			expected_counts=expected_counts['validation'],
			expected_primary_counts=expected_counts['validation'],
		)
		validation_mae = _required_metric(
			validation['secondary'], 'macro_mae_samples'
		)
		validation_within_2 = _required_metric(
			validation['secondary'], 'macro_within_2_samples'
		)
		history.append(
			{
				'epoch': epoch,
				'global_step': global_step,
				'train_macro_cross_entropy': (
					train_loss_sum / train_item_count
				),
				'validation_macro_mae_samples': validation_mae,
				'validation_macro_within_2_samples': validation_within_2,
			}
		)
		candidate_score = validation_checkpoint_score(
			cast('Mapping[str, object]', validation['secondary']),
			settings.checkpoint_selection,
		)
		if validation_score_improved(
			candidate_score,
			best_score,
			settings.checkpoint_selection,
		):
			best_score = candidate_score
			best_epoch = epoch
			_save_checkpoint(
				output_dir / BEST_NAME,
				model=model,
				optimizer=optimizer,
				scaler=scaler,
				payload={
					'run_identity': runtime_run_identity,
					'runtime_precision': runtime_precision,
					'checkpoint_selection': settings.checkpoint_selection,
					'best_validation_score': best_score,
					'epoch': epoch,
					'global_step': global_step,
					'validation': validation['secondary'],
				},
			)
		_save_latest(
			output_dir,
			model,
			optimizer,
			scaler,
			history=history,
			best_epoch=best_epoch,
			checkpoint_selection=settings.checkpoint_selection,
			best_score=best_score,
			global_step=global_step,
			epoch=epoch + 1,
			next_position=0,
			train_loss_sum=0.0,
			train_item_count=0,
			completed=False,
			runtime_precision=runtime_precision,
			run_identity=runtime_run_identity,
		)
		_write_json(output_dir / HISTORY_NAME, history)
		start_position = 0
		train_loss_sum = 0.0
		train_item_count = 0
		if max_steps is not None and global_step >= max_steps:
			return None
	if best_epoch is None:
		raise RuntimeError('training completed without a best checkpoint')
	best_path = output_dir / BEST_NAME
	best = torch.load(best_path, map_location=run_device, weights_only=False)
	if best.get('run_identity') != runtime_run_identity:
		raise ValueError('best checkpoint identity changed before test evaluation')
	if best.get('runtime_precision') != runtime_precision:
		raise ValueError(
			'best checkpoint runtime precision changed before test evaluation'
		)
	model.load_state_dict(_state_dict(best))
	test = evaluate_horizon_dataset(
		model,
		datasets['test'],
		runtime,
		predict_one_item=predict_one_item,
		expected_counts=expected_counts['test'],
		expected_primary_counts=expected_counts['test_primary'],
	)
	metrics_payload = {
		**metrics_metadata,
		'benchmark_identity': runtime_run_identity,
		'runtime_precision': runtime_precision,
		'best_epoch': best_epoch,
		'checkpoint_selection': settings.checkpoint_selection,
		'best_validation_score': best_score,
		'best_checkpoint': {
			'path': str(best_path),
			'sha256': file_sha256(best_path),
		},
		'validation': best['validation'],
		'test': {
			'primary_common': test['primary'],
			'secondary_per_horizon': test['secondary'],
			'evaluation_pass_count': 1,
		},
	}
	metrics_path = output_dir / METRICS_NAME
	_write_json(metrics_path, metrics_payload)
	_save_latest(
		output_dir,
		model,
		optimizer,
		scaler,
		history=history,
		best_epoch=best_epoch,
		checkpoint_selection=settings.checkpoint_selection,
		best_score=best_score,
		global_step=global_step,
		epoch=settings.epochs,
		next_position=0,
		train_loss_sum=0.0,
		train_item_count=0,
		completed=True,
		runtime_precision=runtime_precision,
		run_identity=runtime_run_identity,
	)
	return metrics_path


def evaluate_horizon_dataset(  # noqa: PLR0913
	model: nn.Module,
	dataset: Dataset[dict[str, Any]],
	runtime: HorizonRuntimeContext,
	*,
	predict_one_item: PredictOneItem,
	expected_counts: Sequence[int],
	expected_primary_counts: Sequence[int],
) -> dict[str, object]:
	'''Evaluate one split with exact-once secondary and primary coverage.'''
	model.eval()
	predictions: list[np.ndarray] = []
	targets: list[np.ndarray] = []
	secondary_masks: list[np.ndarray] = []
	primary_masks: list[np.ndarray] = []
	with torch.inference_mode():
		for index in range(len(dataset)):
			item = dataset[index]
			prediction = predict_one_item(model, item, runtime)
			predictions.append(prediction.detach().cpu().numpy())
			targets.append(
				_tensor(item, 'target_sample_float').unsqueeze(0).numpy()
			)
			secondary_masks.append(
				(
					_tensor(item, 'supervision_mask')
					& _tensor(item, 'output_valid_mask').unsqueeze(0)
				)
				.unsqueeze(0)
				.numpy()
			)
			primary_masks.append(
				(
					_tensor(item, 'primary_evaluation_mask')
					& _tensor(item, 'output_valid_mask').unsqueeze(0)
				)
				.unsqueeze(0)
				.numpy()
			)
	predicted = np.concatenate(predictions)
	target = np.concatenate(targets)
	secondary_mask = np.concatenate(secondary_masks)
	primary_mask = np.concatenate(primary_masks)
	_validate_evaluation_counts(
		secondary_mask,
		expected_counts,
		label='evaluation',
	)
	_validate_evaluation_counts(
		primary_mask,
		expected_primary_counts,
		label='primary evaluation',
	)
	return {
		'primary': compute_horizon_metrics(predicted, target, primary_mask),
		'secondary': compute_horizon_metrics(predicted, target, secondary_mask),
	}


def backward_and_step_horizon_optimizer(
	*,
	loss: torch.Tensor,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	gradient_clip_norm: float,
) -> torch.Tensor:
	'''Backpropagate, clip gradients, and delegate AMP overflow to GradScaler.'''
	if scaler is None:
		loss.backward()
	else:
		scaler.scale(loss).backward()
		scaler.unscale_(optimizer)
	gradient_norm = torch.nn.utils.clip_grad_norm_(
		model.parameters(),
		max_norm=gradient_clip_norm,
		error_if_nonfinite=False,
	)
	gradient_is_finite = bool(torch.isfinite(gradient_norm).item())
	if scaler is None:
		if not gradient_is_finite:
			raise FloatingPointError(
				'non-finite Volve horizon gradient norm'
			)
		optimizer.step()
	else:
		# GradScaler must run so it can skip an overflowed update and lower its scale.
		scaler.step(optimizer)
		scaler.update()
	return gradient_norm.detach()

def deterministic_tile_order(tile_count: int, seed: int, epoch: int) -> tuple[int, ...]:
	'''Return the all-items-once order for one epoch.'''
	if tile_count <= 0:
		raise ValueError('tile_count must be positive')
	if epoch < 0:
		raise ValueError('epoch must be non-negative')
	generator = torch.Generator().manual_seed(seed + epoch)
	return tuple(
		int(index) for index in torch.randperm(tile_count, generator=generator)
	)


def initial_best_validation_score(selection: str) -> float:
	'''Return the sentinel score for one supported checkpoint policy.'''
	if selection == CHECKPOINT_SELECTION_VALIDATION_MAE:
		return math.inf
	if selection == CHECKPOINT_SELECTION_VALIDATION_WITHIN_2:
		return -math.inf
	raise ValueError(f'unknown horizon checkpoint selection: {selection!r}')


def validation_checkpoint_score(
	validation_metrics: Mapping[str, object],
	selection: str,
) -> float:
	'''Read the finite validation score selected by one supported policy.'''
	if selection == CHECKPOINT_SELECTION_VALIDATION_MAE:
		key = 'macro_mae_samples'
	elif selection == CHECKPOINT_SELECTION_VALIDATION_WITHIN_2:
		key = 'macro_within_2_samples'
	else:
		raise ValueError(f'unknown horizon checkpoint selection: {selection!r}')
	return _required_metric(validation_metrics, key)


def validation_score_improved(
	candidate: float,
	best: float,
	selection: str,
) -> bool:
	'''Compare finite scores strictly in the configured policy direction.'''
	if not math.isfinite(candidate):
		raise ValueError('validation checkpoint score candidate must be finite')
	_validate_best_validation_score(best, selection=selection)
	if selection == CHECKPOINT_SELECTION_VALIDATION_MAE:
		return candidate < best
	return candidate > best


def validation_mae_improved(candidate: float, best: float) -> bool:
	'''Compatibility wrapper for strict validation macro MAE selection.'''
	return validation_score_improved(
		candidate,
		best,
		CHECKPOINT_SELECTION_VALIDATION_MAE,
	)


def resolve_horizon_device(value: str) -> torch.device:
	'''Resolve the requested CPU/CUDA device.'''
	if value not in {'auto', 'cpu', 'cuda'}:
		raise ValueError('device must be auto, cpu, or cuda')
	if value == 'auto':
		return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if value == 'cuda' and not torch.cuda.is_available():
		raise RuntimeError('CUDA was requested but is not available')
	return torch.device(value)


def horizon_runtime_precision_identity(
	device: torch.device, *, amp_requested: bool
) -> dict[str, object]:
	'''Describe the resolved precision mode included in resume identity.'''
	amp_enabled = amp_requested and device.type == 'cuda'
	return {
		'device_type': device.type,
		'amp_enabled': amp_enabled,
		'autocast_dtype': 'float16' if amp_enabled else None,
		'scaler_required': amp_enabled,
	}


def validate_horizon_resume_runtime(
	payload: Mapping[str, object],
	*,
	expected: Mapping[str, object],
	scaler: torch.amp.GradScaler | None,
) -> None:
	'''Reject precision changes and missing or unexpected scaler state.'''
	if payload.get('runtime_precision') != expected:
		raise ValueError('resume checkpoint runtime precision does not match this run')
	scaler_required = expected.get('scaler_required') is True
	scaler_state = payload.get('scaler_state_dict')
	if scaler_required:
		if scaler is None:
			raise ValueError('runtime precision requires a GradScaler')
		if not isinstance(scaler_state, Mapping) or not scaler_state:
			raise ValueError('resume checkpoint is missing required GradScaler state')
	elif scaler_state is not None:
		raise ValueError('resume checkpoint has unexpected GradScaler state')


def horizon_autocast(
	device: torch.device, *, enabled: bool
) -> AbstractContextManager[None]:
	'''Return autocast only for the resolved AMP execution mode.'''
	return torch.autocast(device_type=device.type) if enabled else nullcontext()


def _validate_runner_inputs(  # noqa: C901
	*,
	settings: HorizonRunnerSettings,
	datasets: Mapping[str, Dataset[dict[str, Any]]],
	expected_counts: Mapping[str, Sequence[int]],
	max_steps: int | None,
) -> None:
	initial_best_validation_score(settings.checkpoint_selection)
	if settings.epochs <= 0:
		raise ValueError('epochs must be positive')
	if settings.seed < 0:
		raise ValueError('seed must be non-negative')
	if (
		not math.isfinite(settings.gradient_clip_norm)
		or settings.gradient_clip_norm <= 0
	):
		raise ValueError('gradient_clip_norm must be positive and finite')
	if max_steps is not None and (
		not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0
	):
		raise ValueError('max_steps must be a positive integer')
	if set(datasets) != {'train', 'validation', 'test'}:
		raise ValueError('datasets must contain train, validation, and test')
	if set(expected_counts) != {'train', 'validation', 'test', 'test_primary'}:
		raise ValueError('expected_counts must contain all horizon split count sets')
	for split, dataset in datasets.items():
		if len(dataset) == 0:
			raise ValueError(f'{split} dataset must not be empty')
	for split, counts in expected_counts.items():
		if len(counts) != len(HORIZON_NAMES) or any(count <= 0 for count in counts):
			raise ValueError(f'{split} counts must be positive for all horizons')


def _validate_evaluation_counts(
	mask: np.ndarray,
	expected_counts: Sequence[int],
	*,
	label: str,
) -> None:
	actual_counts = tuple(
		int(np.count_nonzero(mask[:, index]))
		for index in range(len(HORIZON_NAMES))
	)
	if actual_counts != tuple(expected_counts):
		raise RuntimeError(
			f'{label} tiles do not provide exact-once model-valid coverage; '
			f'expected {tuple(expected_counts)!r}, got {actual_counts!r}'
		)


def _save_latest(  # noqa: PLR0913
	output_dir: Path,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	*,
	history: Sequence[Mapping[str, object]],
	best_epoch: int | None,
	checkpoint_selection: str,
	best_score: float,
	global_step: int,
	epoch: int,
	next_position: int,
	train_loss_sum: float,
	train_item_count: int,
	completed: bool,
	runtime_precision: Mapping[str, object],
	run_identity: Mapping[str, object],
) -> None:
	_save_checkpoint(
		output_dir / LATEST_NAME,
		model=model,
		optimizer=optimizer,
		scaler=scaler,
		payload={
			'run_identity': run_identity,
			'runtime_precision': runtime_precision,
			'history': list(history),
			'best_epoch': best_epoch,
			'checkpoint_selection': checkpoint_selection,
			'best_validation_score': best_score,
			'global_step': global_step,
			'epoch': epoch,
			'next_position': next_position,
			'train_loss_sum': train_loss_sum,
			'train_item_count': train_item_count,
			# Retain the 006 checkpoint field while 007 adopts the shared name.
			'train_tile_count': train_item_count,
			'completed': completed,
		},
	)


def _resume_best_validation_score(
	payload: Mapping[str, object],
	*,
	selection: str,
) -> float:
	checkpoint_selection = payload.get('checkpoint_selection')
	if checkpoint_selection is not None and checkpoint_selection != selection:
		raise ValueError('resume checkpoint selection does not match this run')
	if 'best_validation_score' in payload:
		best = _number(
			payload.get('best_validation_score'),
			'best_validation_score',
		)
	elif selection == CHECKPOINT_SELECTION_VALIDATION_MAE:
		best = _number(
			payload.get('best_validation_macro_mae_samples', math.inf),
			'best_validation_macro_mae_samples',
		)
	else:
		raise ValueError(
			'within-2 checkpoint selection cannot resume a legacy MAE-only state'
		)
	_validate_best_validation_score(best, selection=selection)
	return best


def _validate_best_validation_score(best: float, *, selection: str) -> None:
	initial = initial_best_validation_score(selection)
	if math.isnan(best) or (math.isinf(best) and best != initial):
		raise ValueError('best validation checkpoint score is invalid')


def _save_checkpoint(
	path: Path,
	*,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	payload: Mapping[str, object],
) -> None:
	full = {
		**payload,
		'model_state_dict': {
			name: value.detach().cpu() for name, value in model.state_dict().items()
		},
		'optimizer_state_dict': optimizer.state_dict(),
		'scaler_state_dict': None if scaler is None else scaler.state_dict(),
	}
	temporary = path.with_name(f'.{path.name}.tmp')
	torch.save(full, temporary)
	temporary.replace(path)


def _validate_output(output_dir: Path, resume: Path | None) -> None:
	if resume is None:
		if output_dir.exists() and any(output_dir.iterdir()):
			raise FileExistsError(f'horizon job output is non-empty: {output_dir}')
		return
	if not resume.is_file() or resume.name != LATEST_NAME:
		raise FileNotFoundError(f'resume must identify an existing {LATEST_NAME}')
	if resume.parent.resolve() != output_dir.resolve():
		raise ValueError('resume checkpoint must be in this job output directory')


def _configure_determinism() -> None:
	os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
	torch.use_deterministic_algorithms(mode=True)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def _seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)  # noqa: NPY002
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f'.{path.name}.tmp')
	temporary.write_text(
		json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	temporary.replace(path)


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return child


def _state_dict(payload: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
	value = payload.get('model_state_dict')
	if not isinstance(value, Mapping):
		raise TypeError('checkpoint model_state_dict must be a mapping')
	return cast('Mapping[str, torch.Tensor]', value)


def _tensor(value: Mapping[str, object], key: str) -> torch.Tensor:
	item = value.get(key)
	if not isinstance(item, torch.Tensor):
		raise TypeError(f'{key} must be a tensor')
	return item


def _sequence(value: object, label: str) -> Sequence[object]:
	if not isinstance(value, list):
		raise TypeError(f'{label} must be a list')
	return value


def _optional_int(value: object, label: str) -> int | None:
	return None if value is None else _nonnegative_int(value, label)


def _nonnegative_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		raise TypeError(f'{label} must be an integer')
	if value < 0:
		raise ValueError(f'{label} must be non-negative')
	return value


def _finite_number(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		raise TypeError(f'{label} must be numeric')
	result = float(value)
	if not math.isfinite(result):
		raise ValueError(f'{label} must be finite')
	return result


def _number(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		raise TypeError(f'{label} must be numeric')
	return float(value)


def _required_metric(metrics: object, key: str) -> float:
	if not isinstance(metrics, Mapping):
		raise TypeError('metrics must be a mapping')
	value = metrics.get(key)
	if (
		not isinstance(value, int | float)
		or isinstance(value, bool)
		or not math.isfinite(float(value))
	):
		raise ValueError(f'metric {key} must be finite')
	return float(value)


__all__ = [
	'BEST_NAME',
	'CHECKPOINT_SELECTION_IDS',
	'CHECKPOINT_SELECTION_VALIDATION_MAE',
	'CHECKPOINT_SELECTION_VALIDATION_WITHIN_2',
	'HISTORY_NAME',
	'LATEST_NAME',
	'METRICS_NAME',
	'HorizonRunnerSettings',
	'HorizonRuntimeContext',
	'backward_and_step_horizon_optimizer',
	'deterministic_tile_order',
	'evaluate_horizon_dataset',
	'horizon_autocast',
	'horizon_runtime_precision_identity',
	'initial_best_validation_score',
	'resolve_horizon_device',
	'run_horizon_training_job',
	'validate_horizon_resume_runtime',
	'validation_checkpoint_score',
	'validation_mae_improved',
	'validation_score_improved',
]
