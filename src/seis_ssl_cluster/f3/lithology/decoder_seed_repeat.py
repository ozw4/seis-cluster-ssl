"""Isolated F3 LocalBT masked-v3 decoder seed repeat; frozen encoder unchanged."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.io.labels import F3ClassInfo
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	audit_f3_lithology_candidate_source,
	f3_lithology_candidate_config_from_mapping,
	load_f3_lithology_candidate_canonical_config,
	resolve_f3_lithology_candidate_job,
)
from seis_ssl_cluster.f3.lithology.common_test import (
	LAYOUTS,
	SIZES,
	TestPlan,
	evaluate_cell,
	inspect_test_cell,
	load_cached_test_cell,
	publish_summary,
)
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	F3FiveWayJob,
	decoder_config_for_job,
	inspect_f3_lithology_five_way_job,
	run_f3_lithology_frozen_encoder_job,
)
from seis_ssl_cluster.training.voxel_decoder.checkpoint import (
	load_voxel_decoder_checkpoint,
)

NAMESPACE = 'local_bt_v3_masklr3em5_decoder_seed42001_v1'
CONDITION = 'v3__local_bt__k6810__masklr3e-5__mw0.5__vw0.5'
NEW_CONDITION = CONDITION + '__decoder42001'


@dataclass(frozen=True)
class SeedRepeat:
	"""Preflight-pinned source, fifteen jobs and unchanged test definition."""

	config: dict[str, Any]
	jobs: tuple[F3FiveWayJob, ...]
	condition: dict[str, str]
	test_plan: TestPlan
	mask: Path
	identity: str


def _read(path: Path) -> dict[str, Any]:
	return json.loads(path.read_text(encoding='utf-8'))


def _verify(record: dict[str, str]) -> Path:
	path = Path(record['path'])
	if 'reports' in path.resolve().parts or file_sha256(path) != record['sha256']:
		raise ValueError(f'input hash drift or forbidden report input: {path}')
	return path


def _write(path: Path, payload: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp = path.with_name('.' + path.name + '.tmp')
	tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
	tmp.replace(path)


def repeat_job(baseline: F3FiveWayJob, runs_root: Path) -> F3FiveWayJob:
	"""Change only seed and output, preserving source and supervision identities."""
	if baseline.decoder_seed != 42000:
		raise ValueError('baseline decoder seed must be 42000')
	output = (
		runs_root
		/ f'model={baseline.model.model_id}'
		/ f'layout={baseline.layout_id}'
		/ f'size={baseline.data_size}'
	)
	if output.resolve() == baseline.output_dir.resolve():
		raise ValueError('seed repeat must not overwrite the baseline')
	return replace(baseline, output_dir=output, decoder_seed=42001)


def load_repeat(path: Path) -> SeedRepeat:  # noqa: C901, PLR0912
	"""Audit frozen source identities and all fifteen seed-only config changes."""
	c = load_config(path)
	if (
		c['experiment_id'] != NAMESPACE
		or c['baseline_condition_id'] != CONDITION
		or c['baseline_decoder_seed'] != 42000
		or c['decoder_seed'] != 42001
		or c['layouts'] != list(LAYOUTS)
		or c['sizes'] != list(SIZES)
	):
		raise ValueError('fixed F3 LocalBT v3 seed-repeat scope changed')
	artifact = Path(c['artifact_root'])
	for key, prefix in (
		('runs_root', 'surveys/f3/facies_benchmark_v2/benchmark'),
		('test_root', 'surveys/f3/facies_benchmark_v2/evaluation'),
		('operations_root', 'operations/f3'),
	):
		expected = artifact / prefix / NAMESPACE
		if key == 'runs_root':
			expected /= 'runs'
		if Path(c[key]).resolve() != expected.resolve():
			raise ValueError(f'{key} must use the separate seed-repeat namespace')
	for record in c['inputs'].values():
		_verify(record)
	receipt = _read(Path(c['inputs']['baseline_receipt']['path']))
	baseline_cells = [x for x in receipt['cells'] if x['condition_id'] == CONDITION]
	if len(baseline_cells) != 15 or {
		(x['layout_id'], x['data_size']) for x in baseline_cells
	} != {(layout, size) for layout in LAYOUTS for size in SIZES}:
		raise ValueError('baseline must contain all fifteen paired cells')
	condition = baseline_cells[0]['condition']
	candidate = f3_lithology_candidate_config_from_mapping(
		load_config(Path(c['inputs']['candidate_config']['path']))
	)
	canonical = load_f3_lithology_candidate_canonical_config(candidate)
	source = audit_f3_lithology_candidate_source(candidate, canonical)
	if (
		source['checkpoint_sha256'] != condition['checkpoint_sha256']
		or source['embedding_metadata_sha256'] != condition['embedding_metadata_sha256']
		or candidate.candidate_id != condition['historical_condition_id']
	):
		raise ValueError('repeat must preserve the original v3 encoder and embeddings')
	jobs = []
	for cell in baseline_cells:
		old = _read(_verify(cell['resolved_decoder_config']))
		baseline = resolve_f3_lithology_candidate_job(
			candidate, canonical, layout=cell['layout_id'], size=cell['data_size']
		)
		if decoder_config_for_job(baseline).to_dict() != old:
			raise ValueError(
				'baseline differs from the original fixed decoder contract'
			)
		_verify(cell['decoder_checkpoint'])
		_verify(cell['supervision_grid'])
		job = repeat_job(baseline, Path(c['runs_root']))
		inspect_f3_lithology_five_way_job(job)
		jobs.append(job)
	mask = Path(c['inputs']['test_mask']['path'])
	test_inputs = receipt['config']['inputs']
	for name in ('labels', 'class_info', 'canonical_grid', 'reference_valid_tokens'):
		_verify(test_inputs[name])
	mask_data = np.load(mask, mmap_mode='r')
	if mask_data.dtype != np.bool_ or int(mask_data.sum()) != 113979624:
		raise ValueError('agreed test mask changed')
	for cell in baseline_cells:
		grid = np.load(cell['supervision_grid']['path'], mmap_mode='r')
		for start in range(0, mask_data.shape[0], 8):
			if np.any(mask_data[start : start + 8] & (grid[start : start + 8] != 0)):
				raise ValueError('test overlaps decoder train/validation')
	identity_payload = {
		'definition_sha256': file_sha256(path),
		'definition': c,
		'baseline_cells': baseline_cells,
		'source': source,
		'decoder_configs': [decoder_config_for_job(j).to_dict() for j in jobs],
	}
	identity = hashlib.sha256(
		json.dumps(identity_payload, sort_keys=True).encode()
	).hexdigest()
	classes = tuple(
		F3ClassInfo(x['class_id'], x['class_name'], tuple(x['rgb']))
		for x in _read(Path(test_inputs['class_info']['path']))['classes']
	)
	config = dict(
		receipt['config'],
		output_root=c['test_root'],
		expected_cells=15,
		expected_conditions=1,
	)
	plan = TestPlan(config, classes, (), (), identity_payload, identity)
	return SeedRepeat(
		c, tuple(jobs), dict(condition, audit_id=NEW_CONDITION), plan, mask, identity
	)


def _wait_for_gpu(operations: Path, status: dict[str, Any]) -> None:
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required; refusing CPU training fallback')
	while True:
		free, _total = torch.cuda.mem_get_info(0)
		if free >= 20 * 1024**3:
			return
		_write(
			operations / 'status.json',
			dict(status, status='waiting_for_gpu', free_bytes=free),
		)
		time.sleep(30)


def _run_job(job: F3FiveWayJob) -> None:
	latest = job.decoder_dir / 'latest.pt'
	resume = None
	if latest.exists():
		payload = load_voxel_decoder_checkpoint(latest, map_location='cpu')
		if payload['resolved_config'] != decoder_config_for_job(job).to_dict():
			raise ValueError('existing decoder belongs to a different seed/config')
		if payload['checkpoint_kind'] != 'completed':
			resume = latest
	if not job.metrics_path.exists():
		run_f3_lithology_frozen_encoder_job(job, device='cuda', resume=resume)
	if (
		_read(job.decoder_dir / 'resolved_config.json')
		!= decoder_config_for_job(job).to_dict()
	):
		raise ValueError('completed decoder config changed')


def run_repeat(repeat: SeedRepeat) -> None:
	"""Train fifteen cells serially, select on validation, score the fixed test."""
	operations = Path(repeat.config['operations_root'])
	operations.mkdir(parents=True, exist_ok=True)
	test_root = Path(repeat.config['test_root'])
	test_root.mkdir(parents=True, exist_ok=True)
	with (operations / '.run.lock').open('a') as lock:
		fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
		selection = test_root / 'selection_receipt.json'
		if selection.exists():
			if _read(selection)['run_identity'] != repeat.identity:
				raise ValueError('resume experiment/input identity drift')
		else:
			_write(
				selection, dict(repeat.test_plan.receipt, run_identity=repeat.identity)
			)
		rows, results = [], []
		status: dict[str, Any] = {
			'pid': os.getpid(),
			'run_identity': repeat.identity,
			'completed_cells': 0,
			'total_cells': 15,
			'decoder_seed': 42001,
		}
		try:
			for job in repeat.jobs:
				status.update(
					layout=job.layout_id, size=job.data_size, status='training'
				)
				_write(operations / 'status.json', status)
				_wait_for_gpu(operations, status)
				_write(operations / 'status.json', status)
				_run_job(job)
				status['status'] = 'test_evaluation'
				_write(operations / 'status.json', status)
				cell = inspect_test_cell(
					job.output_dir,
					repeat.condition,
					job.layout_id,
					job.data_size,
					decoder_seed=42001,
				)
				destination = (
					test_root / 'cells' / NEW_CONDITION / job.layout_id / job.data_size
				)
				if destination.exists():
					metrics = load_cached_test_cell(
						repeat.test_plan, cell, destination, repeat.mask
					)
				else:
					metrics = evaluate_cell(
						repeat.test_plan, cell, repeat.mask, destination
					)
				results.append(metrics)
				row = {
					k: repeat.condition[k]
					for k in (
						'version',
						'source',
						'head_ks',
						'distillation_weight',
						'consistency_weight',
						'mask_token_lr',
						'masked_prototype_weight',
						'visible_prototype_weight',
					)
				}
				row.update(
					condition_id=NEW_CONDITION,
					layout_id=job.layout_id,
					data_size=job.data_size,
					**{
						k: metrics[k]
						for k in (
							'mean_iou',
							'macro_f1',
							'accuracy',
							'balanced_accuracy',
							'evaluation_voxel_count',
						)
					},
				)
				rows.append(row)
				status['completed_cells'] = len(rows)
				_write(operations / 'status.json', status)
				print(json.dumps(status), flush=True)
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
			if not (test_root / 'summary').exists():
				publish_summary(test_root, rows, results, repeat.identity)
			status['status'] = 'complete'
			_write(operations / 'status.json', status)
		except BaseException as exc:
			_write(
				operations / 'status.json',
				dict(status, status='failed', error=str(exc)),
			)
			raise


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI without reading artifacts or starting execution."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', type=Path, required=True)
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument('--dry-run', action='store_true')
	mode.add_argument('--execute', action='store_true')
	return parser


def main() -> None:
	"""Read-only by default; require explicit execution of the fixed repeat."""
	args = build_parser().parse_args()
	repeat = load_repeat(args.config)
	print(
		json.dumps(
			{
				'mode': 'execute' if args.execute else 'dry_run',
				'cells': len(repeat.jobs),
				'run_identity': repeat.identity,
				'encoder': 'fixed',
				'baseline_decoder_seed': 42000,
				'decoder_seed': 42001,
			}
		),
		flush=True,
	)
	if args.execute:
		run_repeat(repeat)
