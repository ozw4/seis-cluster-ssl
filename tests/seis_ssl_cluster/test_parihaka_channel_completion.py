"""Full-budget and test-only provenance checks for Parihaka decoder results."""

from __future__ import annotations

import csv
import json
import shutil
from typing import TYPE_CHECKING

import pytest
import torch

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.parihaka.channel_completion import (
	HISTORY_FIELDS,
	PUBLIC_METRIC_NAMES,
	inspect_completed_channel_job,
)

if TYPE_CHECKING:
	from pathlib import Path


def _write_json(path: Path, value: object) -> None:
	path.write_text(json.dumps(value), encoding='utf-8')


def _write_history(path: Path, rows: list[dict[str, object]]) -> None:
	with path.open('w', newline='', encoding='utf-8') as stream:
		writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
		writer.writeheader()
		writer.writerows(rows)


def _make_job(job_dir: Path, *, mode: str = 'normal') -> Path:
	job_dir.mkdir(parents=True)
	identity = {
		'model': 'mae',
		'layout_id': 'layout_000',
		'data_size': 'medium',
		'training': {'epochs': 50, 'batch_size': 1, 'sampling_mode': 'all_tiles_once'},
		'tile_counts': {'train': 2},
	}
	history = [
		{
			'epoch': index,
			'global_step': (index + 1) * 2,
			'train_loss': 0.2,
			'train_channel_iou': 0.3,
			'validation_loss': 0.4,
			'validation_channel_iou': 0.8 if index in (7, 17) else 0.2,
			'validation_channel_f1': 0.5,
		}
		for index in range(50)
	]
	validation = dict.fromkeys(PUBLIC_METRIC_NAMES, 0.5)
	validation['channel_iou'] = 0.8
	metrics = {
		'benchmark_identity': identity,
		'best_epoch': 7,
		'validation': validation,
		'test': dict(validation),
	}
	latest = {
		'run_identity': identity,
		'history': history,
		'best_epoch': 7,
		'best_iou': 0.8,
		'global_step': 100,
		'epoch': 50,
		'next_position': 0,
		'completed': True,
		'model_state_dict': {'weight': torch.ones(1)},
	}
	best = {
		'run_identity': identity,
		'epoch': 7,
		'validation': {**validation, 'loss': 0.4, 'confusion_matrix': [[1, 2], [3, 4]]},
		'model_state_dict': {'weight': torch.zeros(1)},
	}
	if mode != 'legacy':
		metrics['evaluation_mode'] = 'validation_and_test'
		latest['evaluation_mode'] = 'validation_and_test'
	torch.save(latest, job_dir / 'latest.pt')
	torch.save(best, job_dir / 'best.pt')
	_write_history(job_dir / 'history.csv', history)
	_write_json(job_dir / 'metrics.json', metrics)
	return job_dir


def _make_test_reuse(tmp_path: Path) -> tuple[Path, Path]:
	source = _make_job(tmp_path / 'validation_source')
	latest = torch.load(source / 'latest.pt', weights_only=False)
	latest['evaluation_mode'] = 'validation_only'
	torch.save(latest, source / 'latest.pt')
	metrics = json.loads((source / 'metrics.json').read_text())
	metrics['evaluation_mode'] = 'validation_only'
	test = metrics.pop('test')
	_write_json(source / 'metrics.json', metrics)
	output = tmp_path / 'test_evaluation'
	shutil.copytree(source, output)
	metrics.update(
		{
			'evaluation_mode': 'validation_and_test',
			'test': test,
			'evaluation_provenance': {
				'method': 'completed_validation_selected_checkpoint_test_only',
				'additional_optimizer_steps': 0,
				'source_dir': str(source),
				'source_latest_sha256': file_sha256(source / 'latest.pt'),
				'source_best_sha256': file_sha256(source / 'best.pt'),
				'source_metrics_sha256': file_sha256(source / 'metrics.json'),
			},
		}
	)
	_write_json(output / 'metrics.json', metrics)
	return output, source


@pytest.mark.parametrize('mode', ['normal', 'legacy'])
def test_accepts_full_budget_and_first_validation_maximum(
	tmp_path: Path, mode: str
) -> None:
	job = _make_job(tmp_path / 'job', mode=mode)
	assert inspect_completed_channel_job(job) == {
		'epochs': 50,
		'global_step': 100,
		'best_epoch': 7,
		'test_only_reuse': False,
	}


def test_accepts_exact_zero_step_test_reuse(tmp_path: Path) -> None:
	job, source = _make_test_reuse(tmp_path)
	before = {path.name: file_sha256(path) for path in source.iterdir()}
	assert inspect_completed_channel_job(job)['test_only_reuse'] is True
	assert {path.name: file_sha256(path) for path in source.iterdir()} == before


@pytest.mark.parametrize(
	('field', 'value'),
	[('epochs', 49), ('batch_size', 2), ('sampling_mode', 'random')],
)
def test_rejects_a_different_declared_training_budget(
	tmp_path: Path,
	field: str,
	value: object,
) -> None:
	job = _make_job(tmp_path / 'job')
	metrics = json.loads((job / 'metrics.json').read_text())
	metrics['benchmark_identity']['training'][field] = value
	_write_json(job / 'metrics.json', metrics)
	for filename in ('latest.pt', 'best.pt'):
		payload = torch.load(job / filename, weights_only=False)
		payload['run_identity']['training'][field] = value
		torch.save(payload, job / filename)
	with pytest.raises(ValueError, match='50 epochs, batch size 1, and all_tiles_once'):
		inspect_completed_channel_job(job)


def test_rejects_consistent_selection_of_a_later_tied_maximum(tmp_path: Path) -> None:
	job = _make_job(tmp_path / 'job')
	metrics = json.loads((job / 'metrics.json').read_text())
	metrics['best_epoch'] = 17
	_write_json(job / 'metrics.json', metrics)
	for filename, field in (('latest.pt', 'best_epoch'), ('best.pt', 'epoch')):
		payload = torch.load(job / filename, weights_only=False)
		payload[field] = 17
		torch.save(payload, job / filename)
	with pytest.raises(ValueError, match='first maximum validation IoU'):
		inspect_completed_channel_job(job)


@pytest.mark.parametrize(
	('filename', 'field', 'value', 'message'),
	[
		('latest.pt', 'completed', False, 'full decoder budget'),
		('latest.pt', 'epoch', 49, 'full decoder budget'),
		('latest.pt', 'global_step', 99, 'full decoder budget'),
		('latest.pt', 'next_position', 1, 'full decoder budget'),
		('latest.pt', 'run_identity', {}, 'identity differs'),
		('best.pt', 'run_identity', {}, 'identity differs'),
		('latest.pt', 'best_epoch', 17, 'first maximum'),
		('best.pt', 'epoch', 17, 'first maximum'),
		('latest.pt', 'best_iou', 0.7, 'best IoU differs'),
		('latest.pt', 'evaluation_mode', 'validation_only', 'test-only provenance'),
		('latest.pt', 'evaluation_mode', 'other', 'mode is inconsistent'),
	],
)
def test_rejects_incomplete_or_mismatched_checkpoints(
	tmp_path: Path,
	filename: str,
	field: str,
	value: object,
	message: str,
) -> None:
	job = _make_job(tmp_path / 'job')
	payload = torch.load(job / filename, weights_only=False)
	payload[field] = value
	torch.save(payload, job / filename)
	with pytest.raises((ValueError, TypeError), match=message):
		inspect_completed_channel_job(job)


@pytest.mark.parametrize(
	'change', ['short', 'epoch', 'step', 'nan', 'csv', 'csv_short']
)
def test_rejects_incomplete_or_different_histories(tmp_path: Path, change: str) -> None:
	job = _make_job(tmp_path / 'job')
	latest = torch.load(job / 'latest.pt', weights_only=False)
	history = latest['history']
	if change == 'short':
		history.pop()
	elif change == 'epoch':
		history[3]['epoch'] = 2
	elif change == 'step':
		history[3]['global_step'] = 7
	elif change == 'nan':
		history[3]['train_loss'] = float('nan')
	elif change == 'csv':
		history[3]['train_loss'] = 0.123
		_write_history(job / 'history.csv', history)
	else:
		_write_history(job / 'history.csv', history[:-1])
	if not change.startswith('csv'):
		torch.save(latest, job / 'latest.pt')
	with pytest.raises(ValueError, match='history'):
		inspect_completed_channel_job(job)


@pytest.mark.parametrize(
	'change', ['best_epoch', 'validation', 'test_nan', 'test_missing', 'mode']
)
def test_rejects_invalid_metrics(tmp_path: Path, change: str) -> None:
	job = _make_job(tmp_path / 'job')
	metrics = json.loads((job / 'metrics.json').read_text())
	if change == 'best_epoch':
		metrics['best_epoch'] = 17
	elif change == 'validation':
		metrics['validation']['channel_recall'] = 0.3
	elif change == 'test_nan':
		metrics['test']['channel_iou'] = float('nan')
	elif change == 'test_missing':
		del metrics['test']
	else:
		metrics['evaluation_mode'] = 'validation_only'
	_write_json(job / 'metrics.json', metrics)
	with pytest.raises((ValueError, TypeError)):
		inspect_completed_channel_job(job)


@pytest.mark.parametrize('name', ['loss', 'channel_iou', 'channel_f1'])
def test_rejects_best_validation_different_from_selected_history(
	tmp_path: Path,
	name: str,
) -> None:
	job = _make_job(tmp_path / 'job')
	best = torch.load(job / 'best.pt', weights_only=False)
	best['validation'][name] = 0.123
	torch.save(best, job / 'best.pt')
	with pytest.raises(ValueError, match='validation'):
		inspect_completed_channel_job(job)


@pytest.mark.parametrize(
	'change',
	['steps', 'method', 'source_best', 'copy_best', 'source_metrics', 'validation'],
)
def test_rejects_unproven_test_only_reuse(tmp_path: Path, change: str) -> None:
	job, source = _make_test_reuse(tmp_path)
	metrics = json.loads((job / 'metrics.json').read_text())
	if change == 'steps':
		metrics['evaluation_provenance']['additional_optimizer_steps'] = 1
	elif change == 'method':
		metrics['evaluation_provenance']['method'] = 'unchecked'
	elif change in {'source_best', 'copy_best'}:
		path = (source if change == 'source_best' else job) / 'best.pt'
		best = torch.load(path, weights_only=False)
		best['model_state_dict']['weight'].fill_(42)
		torch.save(best, path)
	elif change == 'source_metrics':
		source_metrics = json.loads((source / 'metrics.json').read_text())
		source_metrics['validation']['channel_recall'] = 0.3
		_write_json(source / 'metrics.json', source_metrics)
	else:
		source_metrics = json.loads((source / 'metrics.json').read_text())
		source_metrics['validation']['channel_recall'] = 0.3
		_write_json(source / 'metrics.json', source_metrics)
		metrics['evaluation_provenance']['source_metrics_sha256'] = file_sha256(
			source / 'metrics.json'
		)
	_write_json(job / 'metrics.json', metrics)
	with pytest.raises(ValueError, match='test-only'):
		inspect_completed_channel_job(job)
