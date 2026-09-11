"""Portable frozen-control receipt generation and exact parity contracts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from seis_ssl_cluster.f3.lithology.hmm_v1_freeze_receipt import (
	build_control_freeze_receipt,
	canonical_control_cells_sha256,
	load_control_freeze_receipt,
	verify_control_freeze_receipt,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FREEZE_ROOT = REPOSITORY_ROOT / 'reports/hmm_v1'
RECEIPT_PATH = (
	REPOSITORY_ROOT
	/ 'experiments/f3/facies_benchmark_v2/127_hmm_v2_multi_head_screening_v1'
	/ '60_summary/hmm_v1_condition_2b_receipt.json'
)
CLI_PATH = REPOSITORY_ROOT / 'tools/build_f3_hmm_v1_control_receipt.py'


@pytest.fixture
def frozen_cells() -> list[dict[str, Any]]:
	with (FREEZE_ROOT / 'cells.csv').open(encoding='utf-8', newline='') as stream:
		return [
			{
				'layout_id': row['layout_id'],
				'data_size': row['data_size'],
				'mean_iou': float(row['primary_value']),
				'macro_f1': float(row['secondary_value']),
			}
			for row in csv.DictReader(stream)
			if row['dataset'] == 'f3' and row['condition_id'] == '2b'
		]


@pytest.fixture
def copied_freeze(tmp_path: Path) -> Path:
	for name in ('manifest.json', 'cells.csv', 'summary.json'):
		shutil.copyfile(FREEZE_ROOT / name, tmp_path / name)
	return tmp_path


def _refresh_record_hash(freeze_root: Path, name: str) -> None:
	manifest_path = freeze_root / 'manifest.json'
	manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
	for entry in manifest['canonical_results']:
		if Path(entry['path']).name == name:
			path = freeze_root / name
			entry.update(
				size_bytes=path.stat().st_size,
				sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
			)
	manifest_path.write_text(json.dumps(manifest), encoding='utf-8')


def test_committed_receipt_reproduces_exact_freeze_without_artifacts(
	frozen_cells: list[dict[str, Any]],
) -> None:
	receipt = build_control_freeze_receipt(FREEZE_ROOT)
	assert receipt == load_control_freeze_receipt(RECEIPT_PATH)
	assert receipt['control_id'] == 'mae_hmm_k6'
	assert receipt['condition_id'] == '2b'
	assert receipt['f3_cell_count'] == 15
	assert receipt['freeze_date'] == '2026-09-11'
	assert set(receipt['freeze_record_sha256']) == {
		'manifest.json',
		'cells.csv',
		'summary.json',
	}
	verify_control_freeze_receipt(receipt, receipt['checkpoint_sha256'], frozen_cells)


def test_canonical_cells_preserve_full_precision_and_ignore_row_order(
	frozen_cells: list[dict[str, Any]],
) -> None:
	expected = [
		[row['layout_id'], row['data_size'], row['mean_iou'], row['macro_f1']]
		for row in sorted(
			frozen_cells, key=lambda row: (row['layout_id'], row['data_size'])
		)
	]
	expected_digest = hashlib.sha256(
		json.dumps(expected, separators=(',', ':'), allow_nan=False).encode()
	).hexdigest()
	assert (
		canonical_control_cells_sha256(list(reversed(frozen_cells))) == expected_digest
	)
	frozen_cells[0]['mean_iou'] = math.nextafter(frozen_cells[0]['mean_iou'], math.inf)
	assert canonical_control_cells_sha256(frozen_cells) != expected_digest


@pytest.mark.parametrize(
	'mutation', ['mean_iou', 'macro_f1', 'checkpoint', 'missing', 'duplicate']
)
def test_live_control_drift_fails_closed(
	frozen_cells: list[dict[str, Any]],
	mutation: str,
) -> None:
	receipt = load_control_freeze_receipt(RECEIPT_PATH)
	checkpoint = receipt['checkpoint_sha256']
	if mutation in {'mean_iou', 'macro_f1'}:
		frozen_cells[0][mutation] += 0.001
	elif mutation == 'checkpoint':
		checkpoint = '0' * 64
	elif mutation == 'missing':
		frozen_cells.pop()
	else:
		frozen_cells[-1] = frozen_cells[0]
	with pytest.raises(ValueError, match=r'frozen|15 distinct'):
		verify_control_freeze_receipt(receipt, checkpoint, frozen_cells)


@pytest.mark.parametrize('value', [True, '0.5', math.nan, math.inf, -0.1, 1.1])
@pytest.mark.parametrize('metric', ['mean_iou', 'macro_f1'])
def test_canonical_cells_reject_invalid_metric(
	frozen_cells: list[dict[str, Any]],
	metric: str,
	value: object,
) -> None:
	frozen_cells[0][metric] = value
	with pytest.raises((TypeError, ValueError), match='metric'):
		canonical_control_cells_sha256(frozen_cells)


def test_runtime_verification_reads_only_receipt(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	frozen_cells: list[dict[str, Any]],
) -> None:
	receipt = build_control_freeze_receipt(FREEZE_ROOT)
	path = tmp_path / 'receipt.json'
	path.write_text(json.dumps(receipt), encoding='utf-8')
	original_read = Path.read_text

	def receipt_only_read(self: Path, *args: Any, **kwargs: Any) -> str:
		assert self == path
		return original_read(self, *args, **kwargs)

	monkeypatch.setattr(Path, 'read_text', receipt_only_read)
	verify_control_freeze_receipt(
		load_control_freeze_receipt(path), receipt['checkpoint_sha256'], frozen_cells
	)


@pytest.mark.parametrize(
	('field', 'value'),
	[
		('control_id', 'another_control'),
		('condition_id', '2a'),
		('checkpoint_sha256', 'not-a-digest'),
		('canonical_control_cells_sha256', 'A' * 64),
		('f3_cell_count', 14),
		('f3_cell_count', 15.0),
		('freeze_date', '2026-09-12'),
		('schema_version', True),
		('freeze_record_sha256', {}),
	],
)
def test_receipt_contract_rejects_malformed_fields(
	tmp_path: Path,
	field: str,
	value: object,
) -> None:
	receipt = load_control_freeze_receipt(RECEIPT_PATH)
	receipt[field] = value
	path = tmp_path / 'receipt.json'
	path.write_text(json.dumps(receipt), encoding='utf-8')
	with pytest.raises(ValueError, match=r'receipt|SHA-256'):
		load_control_freeze_receipt(path)


def test_generator_rejects_record_checksum_drift(copied_freeze: Path) -> None:
	with (copied_freeze / 'cells.csv').open('a', encoding='utf-8') as stream:
		stream.write('\n')
	with pytest.raises(ValueError, match='checksum mismatch'):
		build_control_freeze_receipt(copied_freeze)


def test_generator_rejects_incomplete_405_cell_freeze(copied_freeze: Path) -> None:
	path = copied_freeze / 'cells.csv'
	path.write_text(
		'\n'.join(path.read_text(encoding='utf-8').splitlines()[:-1]) + '\n',
		encoding='utf-8',
	)
	_refresh_record_hash(copied_freeze, 'cells.csv')
	with pytest.raises(ValueError, match='405 distinct'):
		build_control_freeze_receipt(copied_freeze)


def test_generator_rejects_summary_inconsistency(copied_freeze: Path) -> None:
	path = copied_freeze / 'summary.json'
	summary = json.loads(path.read_text(encoding='utf-8'))
	summary['datasets']['f3']['conditions']['2b']['mean'] += 0.001
	path.write_text(json.dumps(summary), encoding='utf-8')
	_refresh_record_hash(copied_freeze, 'summary.json')
	with pytest.raises(ValueError, match='statistics disagree'):
		build_control_freeze_receipt(copied_freeze)


@pytest.mark.parametrize(
	'mutation',
	['freeze_date', 'completeness', 'checkpoint_missing', 'checkpoint_duplicate'],
)
def test_generator_rejects_manifest_inconsistency(
	copied_freeze: Path,
	mutation: str,
) -> None:
	path = copied_freeze / 'manifest.json'
	manifest = json.loads(path.read_text(encoding='utf-8'))
	if mutation == 'freeze_date':
		manifest['freeze_date'] = '2026-09-12'
	elif mutation == 'completeness':
		manifest['loaded_cells'] = 404
	else:
		entry = next(
			row
			for row in manifest['checkpoints']
			if row['dataset'] == 'f3' and row['condition_id'] == '2b'
		)
		if mutation == 'checkpoint_missing':
			manifest['checkpoints'].remove(entry)
		else:
			manifest['checkpoints'].append(entry)
	path.write_text(json.dumps(manifest), encoding='utf-8')
	with pytest.raises(ValueError, match='freeze'):
		build_control_freeze_receipt(copied_freeze)


def test_cli_emits_only_receipt_and_checks_without_overwrite(tmp_path: Path) -> None:
	output = tmp_path / 'receipt.json'
	command = [
		sys.executable,
		str(CLI_PATH),
		'--freeze-root',
		str(FREEZE_ROOT),
		'--output',
		str(output),
	]
	subprocess.run(command, check=True, capture_output=True, text=True)  # noqa: S603
	assert list(tmp_path.iterdir()) == [output]
	assert load_control_freeze_receipt(output) == load_control_freeze_receipt(
		RECEIPT_PATH
	)
	snapshot = output.read_bytes(), output.stat().st_mtime_ns
	checked = subprocess.run(  # noqa: S603
		[*command, '--check'], check=True, capture_output=True, text=True
	)
	assert json.loads(checked.stdout) == {'status': 'ok', 'f3_cell_count': 15}
	assert (output.read_bytes(), output.stat().st_mtime_ns) == snapshot
	refused = subprocess.run(command, check=False, capture_output=True, text=True)  # noqa: S603
	assert refused.returncode != 0
	assert 'FileExistsError' in refused.stderr
	assert (output.read_bytes(), output.stat().st_mtime_ns) == snapshot


def test_cli_check_rejects_edited_receipt(tmp_path: Path) -> None:
	receipt = load_control_freeze_receipt(RECEIPT_PATH)
	receipt['checkpoint_sha256'] = '0' * 64
	output = tmp_path / 'receipt.json'
	output.write_text(json.dumps(receipt), encoding='utf-8')
	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			str(CLI_PATH),
			'--freeze-root',
			str(FREEZE_ROOT),
			'--output',
			str(output),
			'--check',
		],
		check=False,
		capture_output=True,
		text=True,
	)
	assert result.returncode != 0
	assert 'differs from the authoritative' in result.stderr
