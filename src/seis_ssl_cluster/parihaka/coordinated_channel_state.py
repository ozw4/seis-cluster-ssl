"""Read-only, deep completion/resume evidence for coordinated Channel cells."""

# ruff: noqa: SLF001 - Reuse the existing decoder's scientific contract unchanged.

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch

from seis_ssl_cluster.parihaka import channel_completion as completion
from seis_ssl_cluster.parihaka import channel_decoder as decoder

if TYPE_CHECKING:
	from pathlib import Path

LATEST_KEYS = frozenset(
	{
		'run_identity',
		'evaluation_mode',
		'history',
		'best_epoch',
		'best_iou',
		'global_step',
		'epoch',
		'next_position',
		'train_confusion',
		'train_loss_sum',
		'train_voxels',
		'completed',
		'model_state_dict',
		'optimizer_state_dict',
		'scaler_state_dict',
	}
)
BEST_KEYS = frozenset(
	{
		'run_identity',
		'epoch',
		'validation',
		'model_state_dict',
		'optimizer_state_dict',
		'scaler_state_dict',
	}
)


def _mapping(value: object) -> Mapping:
	if not isinstance(value, Mapping):
		raise TypeError('Channel checkpoint evidence must be a mapping')
	return value


def _integer(value: object) -> int:
	if type(value) is not int:
		raise TypeError('Channel counters must be integers, not booleans')
	return value


def _number(value: object) -> float:
	if isinstance(value, bool) or not isinstance(value, int | float):
		raise TypeError('Channel metrics must be numeric')
	if not math.isfinite(value):
		raise ValueError('Channel metrics must be finite')
	return float(value)


def _tensor(value: object, expected: torch.Tensor) -> torch.Tensor:
	if (
		not isinstance(value, torch.Tensor)
		or value.shape != expected.shape
		or value.dtype != expected.dtype
		or value.layout != torch.strided
		or not bool(torch.isfinite(value).all())
	):
		raise ValueError('Channel tensor shape, dtype, layout, or finite check failed')
	return value


def _validate_weights_and_optimizer(
	payload: Mapping,
	model: torch.nn.Module,
	optimizer: torch.optim.AdamW,
	step: int,
) -> None:
	state = _mapping(payload.get('model_state_dict'))
	expected_state = model.state_dict()
	if set(state) != set(expected_state):
		raise ValueError('Channel model state keys differ from the exact decoder')
	for name, expected in expected_state.items():
		_tensor(state[name], expected)
	if payload.get('scaler_state_dict') is not None:
		raise ValueError('canonical Channel decoder requires AMP disabled')
	stored = _mapping(payload.get('optimizer_state_dict'))
	expected_optimizer = optimizer.state_dict()
	if (
		set(stored) != {'state', 'param_groups'}
		or stored['param_groups'] != expected_optimizer['param_groups']
	):
		raise ValueError('Channel AdamW groups, parameter order, or settings differ')
	moments = _mapping(stored['state'])
	parameters = tuple(model.parameters())
	if set(moments) != set(range(len(parameters))) or any(
		type(index) is not int for index in moments
	):
		raise ValueError('Channel AdamW must retain every parameter state')
	for index, parameter in enumerate(parameters):
		entry = _mapping(moments[index])
		if set(entry) != {'step', 'exp_avg', 'exp_avg_sq'}:
			raise ValueError('Channel AdamW state fields differ')
		saved_step = _tensor(entry['step'], torch.tensor(0.0))
		if saved_step.item() != step:
			raise ValueError('Channel AdamW step differs from the saved budget')
		_tensor(entry['exp_avg'], parameter)
		variance = _tensor(entry['exp_avg_sq'], parameter)
		if bool((variance < 0).any()):
			raise ValueError('Channel AdamW variance cannot be negative')


def _validate_history(latest: Mapping, path: Path, epoch: int, tiles: int) -> list:
	history = latest.get('history')
	if not isinstance(history, list) or len(history) != epoch:
		raise ValueError('Channel partial history length differs from completed epochs')
	for index, value in enumerate(history):
		row = _mapping(value)
		if set(row) != set(completion.HISTORY_FIELDS):
			raise ValueError('Channel history has unexpected fields')
		if (
			_integer(row['epoch']) != index
			or _integer(row['global_step']) != (index + 1) * tiles
		):
			raise ValueError('Channel history epoch/step sequence is inconsistent')
		for field, number in row.items():
			measured = _number(number)
			if measured < 0 or (('iou' in field or 'f1' in field) and measured > 1):
				raise ValueError('Channel history metrics are out of range')
	completion._validate_csv_history(history, path)
	return history


def _confusion(value: object) -> np.ndarray:
	if (
		not isinstance(value, list)
		or len(value) != 2
		or any(not isinstance(row, list) or len(row) != 2 for row in value)
		or any(type(item) is not int or item < 0 for row in value for item in row)
	):
		raise ValueError('Channel confusion must contain nonnegative integer counts')
	return np.asarray(value, dtype=np.int64)


def _validate_accumulators(latest: Mapping, position: int) -> None:
	confusion = _confusion(latest.get('train_confusion'))
	voxels = _integer(latest.get('train_voxels'))
	loss = _number(latest.get('train_loss_sum'))
	if voxels != int(confusion.sum()) or loss < 0:
		raise ValueError('Channel partial accumulators are inconsistent')
	if position == 0:
		if voxels != 0 or loss != 0:
			raise ValueError('Channel epoch boundary must have empty accumulators')
	elif voxels <= 0:
		raise ValueError('Channel intra-epoch partial requires supervised voxels')


def _validate_best(
	best: Mapping,
	latest: Mapping,
	history: list,
	plan: decoder.ChannelDecoderPlan,
) -> None:
	selected = max(history, key=lambda row: row['validation_channel_iou'])
	validation = _mapping(best.get('validation'))
	if set(validation) != {
		*completion.PUBLIC_METRIC_NAMES,
		'loss',
		'confusion_matrix',
		'supervised_voxel_count',
	}:
		raise ValueError('Channel best validation fields differ')
	confusion = _confusion(validation['confusion_matrix'])
	if _integer(validation['supervised_voxel_count']) != int(confusion.sum()) or int(
		confusion.sum()
	) != sum(plan.split_counts['validation']):
		raise ValueError('Channel best validation voxel count differs from the split')
	if tuple(int(value) for value in confusion.sum(axis=1)) != tuple(
		plan.split_counts['validation']
	):
		raise ValueError('Channel best validation class counts differ from the split')
	recomputed = decoder.channel_metrics(confusion)
	for name in completion.PUBLIC_METRIC_NAMES:
		if _number(validation[name]) != recomputed[name]:
			raise ValueError(
				'Channel best validation differs from its confusion matrix'
			)
	completion._validate_selection(
		{
			'best_epoch': latest['best_epoch'],
			'validation': {
				key: validation[key] for key in completion.PUBLIC_METRIC_NAMES
			},
		},
		latest,
		best,
		selected,
	)


def _validate_metrics(metrics: Mapping, latest: Mapping, identity: Mapping) -> None:
	if (
		metrics.get('benchmark_identity') != identity
		or metrics.get('evaluation_mode') != 'validation_and_test'
		or 'evaluation_provenance' in metrics
		or metrics.get('model') != identity['model']
		or metrics.get('layout_id') != identity['layout_id']
		or metrics.get('data_size') != identity['data_size']
		or _integer(metrics.get('best_epoch')) != latest['best_epoch']
	):
		raise ValueError('Channel metrics differ from the fresh canonical identity')
	for split in ('validation', 'test'):
		values = _mapping(metrics.get(split))
		if set(values) != set(completion.PUBLIC_METRIC_NAMES):
			raise ValueError('Channel result requires all five public metrics')
		if any(not 0 <= _number(value) <= 1 for value in values.values()):
			raise ValueError('Channel public metrics must be in [0, 1]')


def inspect_channel_state(  # noqa: C901, PLR0912
	plan: decoder.ChannelDecoderPlan, identity: Mapping
) -> str:
	"""Return fresh/partial/complete only after exact identity and deep state checks.

	The caller owns path safety, a held cell lock, stable input hashes, and the
	input receipt. A completed checkpoint is never turned into a resumable one.
	"""
	root = plan.output_dir
	if not root.exists() or not any(root.iterdir()):
		return 'fresh'
	files = {path.name for path in root.iterdir()}
	if not {'latest.pt', 'history.csv'} <= files or files - {
		'latest.pt',
		'best.pt',
		'history.csv',
		'metrics.json',
	}:
		raise ValueError('Channel output has missing or foreign files')
	latest = _mapping(
		torch.load(root / 'latest.pt', map_location='cpu', weights_only=False)
	)
	if set(latest) != LATEST_KEYS or latest['run_identity'] != identity:
		raise ValueError('Channel latest schema or fresh identity differs')
	if latest['evaluation_mode'] != 'validation_and_test':
		raise ValueError('canonical Channel cells require validation and test')
	if type(latest['completed']) is not bool:
		raise TypeError('Channel completed flag must be a boolean')
	epoch = _integer(latest['epoch'])
	step = _integer(latest['global_step'])
	position = _integer(latest['next_position'])
	tiles = plan.tile_counts['train']
	if (
		not 0 <= epoch <= 50
		or not 0 <= position < tiles
		or not 1 <= step <= 50 * tiles
		or step != epoch * tiles + position
		or (epoch == 50 and position != 0)
	):
		raise ValueError('Channel epoch/position/step budget is inconsistent')
	history = _validate_history(latest, root / 'history.csv', epoch, tiles)
	_validate_accumulators(latest, position)
	with torch.device('cpu'):
		model = decoder._make_decoder(plan.config.decoder, plan.geometry.patch_size_xyz)
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=plan.config.train.learning_rate,
		weight_decay=plan.config.train.weight_decay,
	)
	_validate_weights_and_optimizer(latest, model, optimizer, step)
	best = None
	if epoch:
		best = _mapping(
			torch.load(root / 'best.pt', map_location='cpu', weights_only=False)
		)
		if set(best) != BEST_KEYS or best['run_identity'] != identity:
			raise ValueError('Channel best schema or fresh identity differs')
		_validate_best(best, latest, history, plan)
		_validate_weights_and_optimizer(
			best,
			model,
			optimizer,
			(_integer(best['epoch']) + 1) * tiles,
		)
	elif (
		'best.pt' in files
		or latest['best_epoch'] is not None
		or latest['best_iou'] != -1.0
	):
		raise ValueError('Channel first-epoch partial must not have a selected best')
	if 'metrics.json' in files:
		if epoch != 50 or best is None:
			raise ValueError('Channel metrics cannot precede the full training budget')
		metrics = _mapping(
			json.loads((root / 'metrics.json').read_text(encoding='utf-8'))
		)
		_validate_metrics(metrics, latest, identity)
		completion._validate_selection(
			metrics,
			latest,
			best,
			max(history, key=lambda row: row['validation_channel_iou']),
		)
	if latest['completed']:
		completion.inspect_completed_channel_job(root)
		if best is None or any(
			not torch.equal(value, best['model_state_dict'][key])
			for key, value in latest['model_state_dict'].items()
		):
			raise ValueError(
				'completed Channel model must equal the selected best model'
			)
		return 'complete'
	return 'partial'
