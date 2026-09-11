"""Bind a live F3 control to the frozen HMM_v1 Condition 2b without reading reports."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_FREEZE_DATE = '2026-09-11'
_CONDITION_IDS = ('1', '2a', '2b', '3', '4a', '4b', '5', '6a', '6b')
_DATASETS = ('f3', 'parihaka', 'volve')
_CELL_IDENTITIES = {
	(f'layout_{index:03d}', size)
	for index in range(5)
	for size in ('small', 'medium', 'large')
}
_RECORD_NAMES = ('manifest.json', 'cells.csv', 'summary.json')
_REQUIRED_RECEIPT_KEYS = {
	'control_id',
	'condition_id',
	'checkpoint_sha256',
	'f3_cell_count',
	'canonical_control_cells_sha256',
}
_COMPLETENESS = {
	'expected_cells': 405,
	'loaded_cells': 405,
	'missing_cells': 0,
	'duplicate_cell_identities': 0,
	'unreadable_results': 0,
	'non_finite_selected_metrics': 0,
	'cells_per_condition_per_dataset': 15,
}


def _sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
	value = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(value, dict):
		raise TypeError(f'expected a JSON object: {path}')
	return value


def _validate_sha256(value: object, label: str) -> None:
	if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
		raise ValueError(f'{label} must be a lowercase SHA-256 digest')


def _metric(value: object, label: str, *, bounded: bool = True) -> float:
	if isinstance(value, bool) or not isinstance(value, (int, float)):
		raise TypeError(f'{label} must be a numeric metric')
	number = float(value)
	if not math.isfinite(number) or number < 0.0 or (bounded and number > 1.0):
		raise ValueError(f'{label} must be finite and within its metric range')
	return number


def canonical_control_cells_sha256(cells: Sequence[Mapping[str, Any]]) -> str:
	"""Hash all 15 exact control cells, sorted by layout/size with unrounded floats."""
	identities = [(cell['layout_id'], cell['data_size']) for cell in cells]
	if len(identities) != 15 or set(identities) != _CELL_IDENTITIES:
		raise ValueError(
			'frozen control requires exactly 15 distinct layout/data_size cells'
		)
	canonical = [
		[
			cell['layout_id'],
			cell['data_size'],
			_metric(cell['mean_iou'], 'mean_iou'),
			_metric(cell['macro_f1'], 'macro_f1'),
		]
		for cell in sorted(
			cells, key=lambda cell: (cell['layout_id'], cell['data_size'])
		)
	]
	payload = json.dumps(canonical, separators=(',', ':'), allow_nan=False).encode(
		'utf-8'
	)
	return hashlib.sha256(payload).hexdigest()


def _validate_receipt(receipt: Mapping[str, Any]) -> None:
	optional = {'schema_version', 'freeze_date', 'freeze_record_sha256'}
	if not receipt.keys() >= _REQUIRED_RECEIPT_KEYS or receipt.keys() - (
		_REQUIRED_RECEIPT_KEYS | optional
	):
		raise ValueError('invalid frozen control receipt fields')
	if receipt['control_id'] != 'mae_hmm_k6' or receipt['condition_id'] != '2b':
		raise ValueError('frozen control receipt must bind mae_hmm_k6 to Condition 2b')
	if type(receipt['f3_cell_count']) is not int or receipt['f3_cell_count'] != 15:
		raise ValueError('frozen control receipt must contain f3_cell_count=15')
	for field in ('checkpoint_sha256', 'canonical_control_cells_sha256'):
		_validate_sha256(receipt[field], field)
	if 'schema_version' in receipt and (
		type(receipt['schema_version']) is not int or receipt['schema_version'] != 1
	):
		raise ValueError('unsupported frozen control receipt schema_version')
	if receipt.get('freeze_date', _FREEZE_DATE) != _FREEZE_DATE:
		raise ValueError('frozen control receipt must refer to the 2026-09-11 freeze')
	if 'freeze_record_sha256' in receipt:
		records = receipt['freeze_record_sha256']
		if not isinstance(records, Mapping) or set(records) != set(_RECORD_NAMES):
			raise ValueError(
				'receipt must identify all three authoritative freeze records'
			)
		for name, digest in records.items():
			_validate_sha256(digest, name)


def load_control_freeze_receipt(path: Path) -> dict[str, Any]:
	"""Read the small audit receipt without reading reports or live artifacts."""
	receipt = _read_json(path)
	_validate_receipt(receipt)
	return receipt


def verify_control_freeze_receipt(
	receipt: Mapping[str, Any],
	checkpoint_sha256: str,
	cells: Sequence[Mapping[str, Any]],
) -> None:
	"""Reject any checkpoint or control-cell difference from frozen Condition 2b."""
	_validate_receipt(receipt)
	_validate_sha256(checkpoint_sha256, 'live control checkpoint_sha256')
	if checkpoint_sha256 != receipt['checkpoint_sha256']:
		raise ValueError(
			'live control checkpoint SHA-256 differs from frozen HMM_v1 Condition 2b'
		)
	if (
		canonical_control_cells_sha256(cells)
		!= receipt['canonical_control_cells_sha256']
	):
		raise ValueError('live control cells differ from frozen HMM_v1 Condition 2b')


def _verify_freeze_records(freeze_root: Path, manifest: Mapping[str, Any]) -> None:
	if any(
		manifest.get(key) != value
		for key, value in {
			'schema_version': 1,
			'version': 'HMM_v1',
			'status': 'frozen',
			'freeze_date': _FREEZE_DATE,
			'head_mode': 'single',
			**_COMPLETENESS,
		}.items()
	):
		raise ValueError('manifest must be the complete 2026-09-11 HMM_v1 freeze')
	entries = manifest['canonical_results']
	expected = {'reports/hmm_v1/cells.csv', 'reports/hmm_v1/summary.json'}
	if len(entries) != 2 or {entry['path'] for entry in entries} != expected:
		raise ValueError(
			'manifest must identify both authoritative freeze result files'
		)
	for entry in entries:
		path = freeze_root / Path(entry['path']).name
		if (
			path.stat().st_size != entry['size_bytes']
			or _sha256(path) != entry['sha256']
		):
			raise ValueError(f'canonical freeze result checksum mismatch: {path.name}')


def _verify_summary(
	cells: Sequence[Mapping[str, str]], summary: Mapping[str, Any]
) -> None:
	expected = {
		(dataset, condition, layout, size)
		for dataset in _DATASETS
		for condition in _CONDITION_IDS
		for layout, size in _CELL_IDENTITIES
	}
	identities = [
		(row['dataset'], row['condition_id'], row['layout_id'], row['data_size'])
		for row in cells
	]
	if len(cells) != 405 or set(identities) != expected:
		raise ValueError('freeze must contain all 405 distinct expected cells')
	if (
		summary.get('version') != 'HMM_v1'
		or summary.get('schema_version') != 1
		or summary.get('completeness') != _COMPLETENESS
		or set(summary['datasets']) != set(_DATASETS)
	):
		raise ValueError('freeze summary completeness disagrees with the manifest')
	for dataset in _DATASETS:
		dataset_summary = summary['datasets'][dataset]
		if set(dataset_summary['conditions']) != set(_CONDITION_IDS):
			raise ValueError('freeze summary must include all nine conditions')
		for condition in _CONDITION_IDS:
			selected = [
				row
				for row in cells
				if (row['dataset'], row['condition_id']) == (dataset, condition)
			]
			for row in selected:
				if any(
					row[key] != dataset_summary[key]
					for key in ('task', 'primary_metric', 'secondary_metric')
				):
					raise ValueError(
						'freeze summary metric definitions disagree with cells.csv'
					)
			primary = [
				_metric(
					float(row['primary_value']),
					'primary_value',
					bounded=dataset != 'volve',
				)
				for row in selected
			]
			secondary = [
				_metric(float(row['secondary_value']), 'secondary_value')
				for row in selected
			]
			expected_stats = {
				'cells': 15,
				'mean': statistics.fmean(primary),
				'population_sd': statistics.pstdev(primary),
				'secondary_mean': statistics.fmean(secondary),
			}
			if any(
				dataset_summary['conditions'][condition].get(key) != value
				for key, value in expected_stats.items()
			):
				raise ValueError('freeze summary statistics disagree with cells.csv')


def build_control_freeze_receipt(freeze_root: Path) -> dict[str, Any]:
	"""Derive an audit-only receipt once from the three authoritative freeze records."""
	manifest = _read_json(freeze_root / 'manifest.json')
	_verify_freeze_records(freeze_root, manifest)
	with (freeze_root / 'cells.csv').open(encoding='utf-8', newline='') as stream:
		cells = list(csv.DictReader(stream))
	summary = _read_json(freeze_root / 'summary.json')
	_verify_summary(cells, summary)
	if summary['experiment_snapshot'] != manifest['experiment_snapshot']:
		raise ValueError('freeze summary snapshot disagrees with the manifest')
	selected = [
		{
			'layout_id': row['layout_id'],
			'data_size': row['data_size'],
			'mean_iou': float(row['primary_value']),
			'macro_f1': float(row['secondary_value']),
		}
		for row in cells
		if row['dataset'] == 'f3' and row['condition_id'] == '2b'
	]
	if (
		summary['datasets']['f3']['primary_metric']
		!= 'mean_iou_unique_validation_voxels'
		or summary['datasets']['f3']['secondary_metric']
		!= 'macro_f1_unique_validation_voxels'
	):
		raise ValueError('frozen F3 control requires Mean IoU and Macro F1')
	checkpoints = [
		entry
		for entry in manifest['checkpoints']
		if entry['dataset'] == 'f3' and entry['condition_id'] == '2b'
	]
	if (
		len(checkpoints) != 1
		or checkpoints[0]['checkpoint_id'] != 'mae_hmm_k6'
		or checkpoints[0]['present_at_freeze'] is not True
	):
		raise ValueError(
			'freeze manifest must identify one present F3 Condition 2b checkpoint'
		)
	receipt = {
		'schema_version': 1,
		'freeze_date': manifest['freeze_date'],
		'control_id': checkpoints[0]['checkpoint_id'],
		'condition_id': checkpoints[0]['condition_id'],
		'checkpoint_sha256': checkpoints[0]['sha256'],
		'f3_cell_count': len(selected),
		'canonical_control_cells_sha256': canonical_control_cells_sha256(selected),
		'freeze_record_sha256': {
			name: _sha256(freeze_root / name) for name in _RECORD_NAMES
		},
	}
	_validate_receipt(receipt)
	return receipt
