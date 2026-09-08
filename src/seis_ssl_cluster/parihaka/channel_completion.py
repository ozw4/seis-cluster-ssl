"""Completed-decoder evidence for the Parihaka pretraining comparison."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch

from seis_ssl_cluster.embedding.writer import file_sha256

PUBLIC_METRIC_NAMES = (
	'channel_iou',
	'channel_f1',
	'channel_precision',
	'channel_recall',
	'balanced_accuracy',
)
HISTORY_FIELDS = (
	'epoch',
	'global_step',
	'train_loss',
	'train_channel_iou',
	'validation_loss',
	'validation_channel_iou',
	'validation_channel_f1',
)
COMPLETED_EPOCHS = 50


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _integer(value: object, label: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int):
		raise TypeError(f'{label} must be an integer')
	return value


def _number(value: object, label: str) -> float:
	if isinstance(value, bool) or not isinstance(value, int | float):
		raise TypeError(f'{label} must be numeric')
	if not math.isfinite(value):
		raise ValueError(f'{label} must be finite')
	return float(value)


def _read_json(path: Path) -> Mapping[str, object]:
	return _mapping(json.loads(path.read_text(encoding='utf-8')), str(path))


def _validate_budget(
	latest: Mapping[str, object],
	identity: Mapping[str, object],
) -> int:
	training = _mapping(identity.get('training'), 'benchmark training')
	if (
		_integer(training.get('epochs'), 'training epochs') != COMPLETED_EPOCHS
		or _integer(training.get('batch_size'), 'training batch size') != 1
		or training.get('sampling_mode') != 'all_tiles_once'
	):
		raise ValueError('expected 50 epochs, batch size 1, and all_tiles_once')
	tiles = _mapping(identity.get('tile_counts'), 'benchmark tile counts')
	steps_per_epoch = _integer(tiles.get('train'), 'training tile count')
	if steps_per_epoch <= 0:
		raise ValueError('training tile count must be positive')
	if (
		latest.get('completed') is not True
		or _integer(latest.get('epoch'), 'latest epoch') != COMPLETED_EPOCHS
		or _integer(latest.get('next_position'), 'latest next position') != 0
		or _integer(latest.get('global_step'), 'latest global step')
		!= COMPLETED_EPOCHS * steps_per_epoch
	):
		raise ValueError('latest checkpoint does not prove the full decoder budget')
	return steps_per_epoch


def _validate_history(
	latest: Mapping[str, object],
	history_path: Path,
	steps_per_epoch: int,
) -> Mapping[str, object]:
	history = latest.get('history')
	if not isinstance(history, list) or len(history) != COMPLETED_EPOCHS:
		raise ValueError('decoder history must contain exactly 50 rows')
	rows = [_mapping(row, 'decoder history row') for row in history]
	for index, row in enumerate(rows):
		if set(row) != set(HISTORY_FIELDS):
			raise ValueError('decoder history row has unexpected fields')
		if (
			_integer(row['epoch'], 'history epoch') != index
			or _integer(row['global_step'], 'history global step')
			!= (index + 1) * steps_per_epoch
		):
			raise ValueError('decoder history epoch or step sequence is incomplete')
		for field, value in row.items():
			_number(value, f'history {field}')
	_validate_csv_history(rows, history_path)
	return max(rows, key=lambda row: _number(row['validation_channel_iou'], 'IoU'))


def _validate_csv_history(
	history: Sequence[Mapping[str, object]],
	path: Path,
) -> None:
	with path.open(encoding='utf-8', newline='') as stream:
		reader = csv.DictReader(stream)
		if (
			reader.fieldnames is None
			or len(reader.fieldnames) != len(HISTORY_FIELDS)
			or set(reader.fieldnames) != set(HISTORY_FIELDS)
		):
			raise ValueError('history CSV columns do not match checkpoint history')
		rows = list(reader)
	if len(rows) != len(history):
		raise ValueError('history CSV row count does not match checkpoint history')
	for expected, actual in zip(history, rows, strict=True):
		if set(actual) != set(HISTORY_FIELDS):
			raise ValueError('history CSV row has unexpected fields')
		for field, value in expected.items():
			try:
				matches = Decimal(actual[field]) == Decimal(str(value))
			except (InvalidOperation, TypeError, ValueError) as exc:
				raise ValueError(f'history CSV {field} is not numeric') from exc
			if not matches:
				raise ValueError(f'history CSV {field} differs from checkpoint history')


def _validate_selection(
	metrics: Mapping[str, object],
	latest: Mapping[str, object],
	best: Mapping[str, object],
	selected: Mapping[str, object],
) -> int:
	epoch = _integer(selected['epoch'], 'selected epoch')
	if (
		_integer(best.get('epoch'), 'best epoch') != epoch
		or _integer(latest.get('best_epoch'), 'latest best epoch') != epoch
		or _integer(metrics.get('best_epoch'), 'metrics best epoch') != epoch
	):
		raise ValueError('decoder best epoch must be the first maximum validation IoU')
	validation = _mapping(best.get('validation'), 'best validation')
	public = _mapping(metrics.get('validation'), 'metrics validation')
	if set(public) != set(PUBLIC_METRIC_NAMES):
		raise ValueError('metrics validation must contain all five public metrics')
	for name in PUBLIC_METRIC_NAMES:
		if _number(public[name], name) != _number(validation.get(name), name):
			raise ValueError(f'best and metrics validation {name} differ')
	for name in ('loss', 'channel_iou', 'channel_f1'):
		if _number(validation.get(name), name) != selected[f'validation_{name}']:
			raise ValueError(f'best validation {name} differs from selected history')
	if _number(latest.get('best_iou'), 'latest best IoU') != validation['channel_iou']:
		raise ValueError('latest best IoU differs from selected validation')
	return epoch


def _validate_test_reuse(job_dir: Path, metrics: Mapping[str, object]) -> None:
	provenance = _mapping(metrics.get('evaluation_provenance'), 'test-only provenance')
	if (
		provenance.get('method') != 'completed_validation_selected_checkpoint_test_only'
		or _integer(provenance.get('additional_optimizer_steps'), 'additional steps')
		!= 0
	):
		raise ValueError('test-only provenance must prove zero-step checkpoint reuse')
	source = Path(str(provenance.get('source_dir', '')))
	if not source.is_absolute() or source.resolve() == job_dir.resolve():
		raise ValueError(
			'test-only source must be a separate absolute artifact directory'
		)
	for name, field in (
		('latest.pt', 'source_latest_sha256'),
		('best.pt', 'source_best_sha256'),
	):
		if (
			file_sha256(source / name) != provenance.get(field)
			or file_sha256(job_dir / name) != provenance[field]
		):
			raise ValueError(f'test-only {name} source/copy checksum mismatch')
	if file_sha256(source / 'history.csv') != file_sha256(job_dir / 'history.csv'):
		raise ValueError('test-only history source/copy checksum mismatch')
	source_metrics = source / 'metrics.json'
	if file_sha256(source_metrics) != provenance.get('source_metrics_sha256'):
		raise ValueError('test-only source metrics checksum mismatch')
	expected_source = dict(metrics)
	del expected_source['test']
	del expected_source['evaluation_provenance']
	expected_source['evaluation_mode'] = 'validation_only'
	if _read_json(source_metrics) != expected_source:
		raise ValueError('test-only metrics do not preserve the validation source')


def inspect_completed_channel_job(
	job_dir: Path,
	*,
	metrics: Mapping[str, object] | None = None,
) -> dict[str, object]:
	"""Require full decoder training, validation selection, and held-out test evidence.

	Legacy completed jobs may omit evaluation-mode fields. Test-only exports may
	retain validation-only training checkpoints when live, checksummed provenance
	proves exact reuse of the original completed validation run.
	"""
	if metrics is None:
		metrics = _read_json(job_dir / 'metrics.json')
	latest = _mapping(
		torch.load(job_dir / 'latest.pt', map_location='cpu', weights_only=False),
		'latest checkpoint',
	)
	best = _mapping(
		torch.load(job_dir / 'best.pt', map_location='cpu', weights_only=False),
		'best checkpoint',
	)
	identity = _mapping(metrics.get('benchmark_identity'), 'metrics benchmark identity')
	if latest.get('run_identity') != identity or best.get('run_identity') != identity:
		raise ValueError('decoder checkpoint identity differs from metrics identity')
	steps_per_epoch = _validate_budget(latest, identity)
	selected = _validate_history(latest, job_dir / 'history.csv', steps_per_epoch)
	best_epoch = _validate_selection(metrics, latest, best, selected)
	test = _mapping(metrics.get('test'), 'test metrics')
	if set(test) != set(PUBLIC_METRIC_NAMES):
		raise ValueError('test metrics must contain all five public metrics')
	for name, value in test.items():
		_number(value, f'test {name}')
	if metrics.get('evaluation_mode', 'validation_and_test') != 'validation_and_test':
		raise ValueError('completed result must include held-out test evaluation')
	mode = latest.get('evaluation_mode', 'validation_and_test')
	if mode == 'validation_only':
		_validate_test_reuse(job_dir, metrics)
	elif mode != 'validation_and_test' or 'evaluation_provenance' in metrics:
		raise ValueError('decoder checkpoint evaluation mode is inconsistent')
	return {
		'epochs': COMPLETED_EPOCHS,
		'global_step': latest['global_step'],
		'best_epoch': best_epoch,
		'test_only_reuse': mode == 'validation_only',
	}
