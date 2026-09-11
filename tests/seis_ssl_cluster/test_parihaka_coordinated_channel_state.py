"""CPU-only deep state checks for future coordinated canonical Channel cells."""

# ruff: noqa: SLF001 - Exercise exact checkpoint schemas and runner resume semantics.

from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from seis_ssl_cluster.parihaka import channel_decoder as decoder
from seis_ssl_cluster.parihaka import coordinated_channel_state as audit
from seis_ssl_cluster.parihaka.channel_data import SectionLines


def state_fixture(tmp_path, *, epoch=1, position=0, completed=False):
	architecture = decoder.DecoderArchitecture(
		2,
		2,
		(2,),
		((2, 2, 2),),
		'nearest',
		'voxelwise_layer_norm',
	)
	training = decoder.DecoderTrain(
		50,
		1,
		0.001,
		0.0001,
		'balanced',
		'all_tiles_once',
		42000,
		amp=False,
		gradient_clip_norm=1.0,
	)
	plan = SimpleNamespace(
		output_dir=tmp_path / 'cell',
		config=SimpleNamespace(decoder=architecture, train=training),
		geometry=SimpleNamespace(patch_size_xyz=(2, 2, 2)),
		tile_counts={'train': 3, 'validation': 1, 'test': 1},
		split_counts={'validation': (5, 5)},
	)
	identity = {
		'model': 'test',
		'layout_id': 'layout_000',
		'data_size': 'small',
		'training': {'epochs': 50, 'batch_size': 1, 'sampling_mode': 'all_tiles_once'},
		'tile_counts': plan.tile_counts,
	}
	model = decoder._make_decoder(architecture, (2, 2, 2))
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
	step = epoch * 3 + position
	optimizer_state = optimizer.state_dict()
	optimizer_state['state'] = {
		index: {
			'step': torch.tensor(float(step)),
			'exp_avg': torch.zeros_like(parameter),
			'exp_avg_sq': torch.zeros_like(parameter),
		}
		for index, parameter in enumerate(model.parameters())
	}
	validation = {
		**decoder.channel_metrics(np.array([[4, 1], [1, 4]], dtype=np.int64)),
		'loss': 0.5,
		'supervised_voxel_count': 10,
	}
	history = [
		{
			'epoch': index,
			'global_step': (index + 1) * 3,
			'train_loss': 0.6,
			'train_channel_iou': 0.5,
			'validation_loss': 0.5,
			'validation_channel_iou': validation['channel_iou'],
			'validation_channel_f1': validation['channel_f1'],
		}
		for index in range(epoch)
	]
	latest = {
		'run_identity': identity,
		'evaluation_mode': 'validation_and_test',
		'history': history,
		'best_epoch': 0 if epoch else None,
		'best_iou': validation['channel_iou'] if epoch else -1.0,
		'global_step': step,
		'epoch': epoch,
		'next_position': position,
		'train_confusion': [[1, 0], [0, 1]] if position else [[0, 0], [0, 0]],
		'train_loss_sum': 0.5 if position else 0.0,
		'train_voxels': 2 if position else 0,
		'completed': completed,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer_state,
		'scaler_state_dict': None,
	}
	best = {
		'run_identity': identity,
		'epoch': 0,
		'validation': validation,
		'model_state_dict': copy.deepcopy(model.state_dict()),
		'optimizer_state_dict': copy.deepcopy(optimizer_state),
		'scaler_state_dict': None,
	}
	for entry in best['optimizer_state_dict']['state'].values():
		entry['step'] = torch.tensor(3.0)
	metrics = {
		'model': 'test',
		'layout_id': 'layout_000',
		'data_size': 'small',
		'benchmark_identity': identity,
		'evaluation_mode': 'validation_and_test',
		'best_epoch': 0,
		'validation': decoder._public_metrics(validation),
		'test': decoder._public_metrics(validation),
	}
	plan.output_dir.mkdir()
	write_state(plan, latest, best if epoch else None, metrics if completed else None)
	return plan, identity, latest, best, metrics


def write_state(plan, latest, best=None, metrics=None):
	torch.save(latest, plan.output_dir / 'latest.pt')
	decoder._write_history(plan.output_dir / 'history.csv', latest['history'])
	if best is not None:
		torch.save(best, plan.output_dir / 'best.pt')
	if metrics is not None:
		(plan.output_dir / 'metrics.json').write_text(json.dumps(metrics))


@pytest.mark.parametrize(
	('epoch', 'position', 'completed', 'expected'),
	[
		(0, 1, False, 'partial'),
		(1, 0, False, 'partial'),
		(1, 2, False, 'partial'),
		(50, 0, False, 'partial'),
		(50, 0, True, 'complete'),
	],
)
def test_valid_partial_and_complete_states(
	tmp_path, epoch, position, completed, expected
):
	plan, identity, _, _, _ = state_fixture(
		tmp_path, epoch=epoch, position=position, completed=completed
	)
	assert audit.inspect_channel_state(plan, identity) == expected


def test_metrics_before_completed_latest_is_valid_zero_step_partial(tmp_path):
	plan, identity, latest, best, metrics = state_fixture(tmp_path, epoch=50)
	write_state(plan, latest, best, metrics)
	assert audit.inspect_channel_state(plan, identity) == 'partial'


def test_self_consistent_validation_with_wrong_class_counts_is_rejected(tmp_path):
	plan, identity, latest, best, metrics = state_fixture(
		tmp_path,
		epoch=50,
		completed=True,
	)
	validation = {
		**decoder.channel_metrics(np.array([[9, 0], [0, 1]], dtype=np.int64)),
		'loss': 0.5,
		'supervised_voxel_count': 10,
	}
	best['validation'] = validation
	latest['best_iou'] = validation['channel_iou']
	for row in latest['history']:
		row['validation_channel_iou'] = validation['channel_iou']
		row['validation_channel_f1'] = validation['channel_f1']
	metrics['validation'] = decoder._public_metrics(validation)
	write_state(plan, latest, best, metrics)
	with pytest.raises(ValueError, match='class counts differ'):
		audit.inspect_channel_state(plan, identity)


@pytest.mark.parametrize(
	'mutation',
	[
		'false_completed',
		'float_epoch',
		'bool_step',
		'over_budget',
		'position',
		'history_length',
		'history_step',
		'history_nan',
		'bad_best',
		'early_metrics',
		'confusion_negative',
		'voxel_mismatch',
		'boundary_accumulator',
		'loss_nan',
		'weights_missing',
		'weights_shape',
		'weights_dtype',
		'weights_nan',
		'optimizer_missing',
		'optimizer_order',
		'optimizer_lr',
		'optimizer_dtype',
		'optimizer_shape',
		'optimizer_step',
		'optimizer_nan',
		'optimizer_variance',
		'scaler',
		'mode',
		'identity',
		'extra_key',
	],
)
def test_partial_corruption_fails_closed(tmp_path, mutation):  # noqa: C901, PLR0912, PLR0915
	plan, identity, latest, best, metrics = state_fixture(tmp_path)
	parameter = next(iter(latest['model_state_dict']))
	optimizer = latest['optimizer_state_dict']
	if mutation == 'false_completed':
		latest['completed'] = True
	elif mutation == 'float_epoch':
		latest['epoch'] = 1.0
	elif mutation == 'bool_step':
		latest['global_step'] = True
	elif mutation == 'over_budget':
		latest['global_step'] = 151
	elif mutation == 'position':
		latest['next_position'] = 3
	elif mutation == 'history_length':
		latest['history'] = []
	elif mutation == 'history_step':
		latest['history'][0]['global_step'] = 2
	elif mutation == 'history_nan':
		latest['history'][0]['train_loss'] = float('nan')
	elif mutation == 'bad_best':
		best['epoch'] = 1
	elif mutation == 'early_metrics':
		(plan.output_dir / 'metrics.json').write_text(json.dumps(metrics))
	elif mutation == 'confusion_negative':
		latest['train_confusion'][0][0] = -1
	elif mutation == 'voxel_mismatch':
		latest['train_voxels'] = 1
	elif mutation == 'boundary_accumulator':
		latest['train_loss_sum'] = 0.1
	elif mutation == 'loss_nan':
		latest['train_loss_sum'] = float('nan')
	elif mutation == 'weights_missing':
		latest['model_state_dict'].pop(parameter)
	elif mutation == 'weights_shape':
		latest['model_state_dict'][parameter] = torch.ones(1)
	elif mutation == 'weights_dtype':
		latest['model_state_dict'][parameter] = latest['model_state_dict'][
			parameter
		].double()
	elif mutation == 'weights_nan':
		latest['model_state_dict'][parameter].fill_(float('nan'))
	elif mutation == 'optimizer_missing':
		optimizer['state'].pop(0)
	elif mutation == 'optimizer_order':
		optimizer['param_groups'][0]['params'].reverse()
	elif mutation == 'optimizer_lr':
		optimizer['param_groups'][0]['lr'] = 0.01
	elif mutation == 'optimizer_dtype':
		optimizer['state'][0]['exp_avg'] = optimizer['state'][0]['exp_avg'].double()
	elif mutation == 'optimizer_shape':
		optimizer['state'][0]['exp_avg'] = torch.ones(1)
	elif mutation == 'optimizer_step':
		optimizer['state'][0]['step'] = torch.tensor(2.0)
	elif mutation == 'optimizer_nan':
		optimizer['state'][0]['exp_avg'].fill_(float('nan'))
	elif mutation == 'optimizer_variance':
		optimizer['state'][0]['exp_avg_sq'].fill_(-1)
	elif mutation == 'scaler':
		latest['scaler_state_dict'] = {}
	elif mutation == 'mode':
		latest['evaluation_mode'] = 'validation_only'
	elif mutation == 'identity':
		latest['run_identity'] = {**identity, 'data_size': 'large'}
	elif mutation == 'extra_key':
		latest['foreign'] = True
	write_state(plan, latest, best)
	with pytest.raises((ValueError, TypeError, FileNotFoundError)):
		audit.inspect_channel_state(plan, identity)


@pytest.mark.parametrize(
	'mutation',
	['csv', 'foreign', 'best_weights', 'best_metrics', 'test_nan', 'missing_metrics'],
)
def test_completed_false_evidence_fails_closed(tmp_path, mutation):
	plan, identity, _latest, best, metrics = state_fixture(
		tmp_path, epoch=50, completed=True
	)
	if mutation == 'csv':
		(plan.output_dir / 'history.csv').write_text('epoch\n0\n')
	elif mutation == 'foreign':
		(plan.output_dir / 'foreign.txt').write_text('foreign')
	elif mutation == 'best_weights':
		key = next(iter(best['model_state_dict']))
		best['model_state_dict'][key].add_(1)
		torch.save(best, plan.output_dir / 'best.pt')
	elif mutation == 'best_metrics':
		best['validation']['confusion_matrix'] = [[5, 0], [0, 5]]
		torch.save(best, plan.output_dir / 'best.pt')
	elif mutation == 'test_nan':
		metrics['test']['channel_iou'] = float('nan')
		(plan.output_dir / 'metrics.json').write_text(json.dumps(metrics))
	else:
		(plan.output_dir / 'metrics.json').unlink()
	with pytest.raises((ValueError, TypeError, FileNotFoundError)):
		audit.inspect_channel_state(plan, identity)


class _TinyTiles(torch.utils.data.Dataset):
	def __init__(self, **kwargs):
		self.split = kwargs['split']

	def __len__(self):
		return 3 if self.split == 'train' else 1

	def __getitem__(self, index):
		labels = torch.arange(8).reshape(2, 2, 2) % 2
		return {
			'embeddings': torch.tensor([0.5 + index, -0.5 - index]).reshape(2, 1, 1, 1),
			'token_valid_mask': torch.ones((1, 1, 1), dtype=torch.bool),
			'labels': labels,
			'supervision_mask': torch.ones_like(labels, dtype=torch.bool),
			'core_mask': torch.ones_like(labels, dtype=torch.bool),
		}


def test_dropout_free_real_runner_resume_preserves_cpu_trajectory(
	tmp_path, monkeypatch
):
	plan, identity, _, _, _ = state_fixture(tmp_path)
	plan.config.train = replace(plan.config.train, epochs=2)
	plan.config.labels = tmp_path / 'unused.npy'
	plan.config.tiles = decoder.DecoderTiles((1, 1, 1), (0, 0, 0))
	plan.model, plan.layout_id, plan.data_size = 'test', 'layout_000', 'small'
	plan.train_lines = SectionLines((0,), (0,))
	plan.layouts = SimpleNamespace(validation=SectionLines((1,), (1,)))
	plan.reserved_training_lines = SectionLines((0,), (0,))
	plan.class_weights = (1.0, 1.0)
	plan.split_counts = {'train': (12, 12), 'validation': (4, 4), 'test': (4, 4)}
	plan.selection = SimpleNamespace(selected_token_xyz=((0, 0, 0),))
	plan.geometry.token_grid_shape_xyz = (1, 1, 1)
	plan.geometry.models = {
		'test': SimpleNamespace(
			paths=SimpleNamespace(embeddings='unused', valid_tokens='unused')
		)
	}
	monkeypatch.setattr(decoder, 'ChannelTileDataset', _TinyTiles)
	monkeypatch.setattr(decoder, '_run_identity', lambda _: identity)
	model = decoder._make_decoder(plan.config.decoder, (2, 2, 2))
	assert not any(
		isinstance(layer, torch.nn.modules.dropout._DropoutNd)
		for layer in model.modules()
	)
	plan.output_dir = tmp_path / 'uninterrupted'
	decoder.run_channel_decoder_job(plan, device='cpu')
	full = torch.load(
		plan.output_dir / 'latest.pt', map_location='cpu', weights_only=False
	)
	plan.output_dir = tmp_path / 'resumed'
	assert decoder.run_channel_decoder_job(plan, device='cpu', max_steps=2) is None
	decoder.run_channel_decoder_job(
		plan, device='cpu', resume=plan.output_dir / 'latest.pt'
	)
	resumed = torch.load(
		plan.output_dir / 'latest.pt', map_location='cpu', weights_only=False
	)
	assert resumed['history'] == full['history']
	assert resumed['global_step'] == full['global_step'] == 6
	assert resumed['best_epoch'] == full['best_epoch']
	for name, tensor in full['model_state_dict'].items():
		assert torch.equal(tensor, resumed['model_state_dict'][name])
	for index, values in full['optimizer_state_dict']['state'].items():
		for name, tensor in values.items():
			assert torch.equal(
				tensor, resumed['optimizer_state_dict']['state'][index][name]
			)
