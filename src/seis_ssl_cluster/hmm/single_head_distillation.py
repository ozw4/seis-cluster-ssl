"""Run the separate zero-distillation extension of frozen single-head HMM v1."""

# ruff: noqa: PLC0415, S603
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.hmm.multi_source_cell import audit_completed_cell
from seis_ssl_cluster.hmm.multi_source_survey import _metrics
from seis_ssl_cluster.training.checkpoint import load_checkpoint
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	validate_stratigraphy_checkpoint_payload,
)
from seis_ssl_cluster.training.strat_hmm_source_audit import _validate_frozen_state

if TYPE_CHECKING:
	from seis_ssl_cluster.volve.horizon_recipe_arm import VolveHorizonRecipeArmConfig

STUDY = 'hmm_v1_k6_zero_distillation_ablation_v1'
MATRIX = Path('experiments/hmm_v1/zero_distillation_ablation_v1/matrix.yaml')
SURVEYS = ('f3', 'parihaka', 'volve')
SOURCES = ('mae', 'local_bt', 'random')
LAYOUTS = tuple(f'layout_{i:03d}' for i in range(5))
SIZES = ('small', 'medium', 'large')
OPT_IN = 'single_head_zero_distillation_ablation'
PUBLISHED_FILES = frozenset(
	{'cells.csv', 'comparison.csv', 'summary.json', 'summary.md'}
)


def _read_rows(path: Path) -> list[dict[str, str]]:
	with path.open(newline='') as stream:
		return list(csv.DictReader(stream))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
	with path.open('w', newline='') as stream:
		writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
		writer.writeheader()
		writer.writerows(rows)


def validate_matrix(path: Path = MATRIX) -> dict[str, Any]:  # noqa: C901, PLR0912
	"""Reject missing arms, changed baselines and changes beyond the ablation."""
	matrix = yaml.safe_load(path.read_text())
	arms = matrix['arms']
	identities = [(a['survey'], a['source']) for a in arms]
	if (
		matrix['study'] != STUDY
		or len(identities) != 9
		or set(identities) != {(s, f) for s in SURVEYS for f in SOURCES}
	):
		raise ValueError('all nine prespecified survey/source arms are required')
	for ref in [matrix['controls'], *[r for a in arms for r in a['configs'].values()]]:
		if 'reports' in Path(ref['path']).parts:
			raise ValueError('reports cannot be pipeline inputs')
		if file_sha256(ref['path']) != ref['sha256']:
			raise ValueError(f'bound input hash drift: {ref["path"]}')
	for arm in arms:
		full = yaml.safe_load(Path(arm['configs']['training']['path']).read_text())
		base = yaml.safe_load(
			Path(arm['configs']['baseline_training']['path']).read_text()
		)
		normalized = deepcopy(full)
		normalized['paths'] = base['paths']
		if 'identity' in base:
			normalized['identity'] = base['identity']
		else:
			normalized.pop('identity', None)
		if (
			full['loss']['distillation_weight'] != 0
			or full['train'].get(OPT_IN) is not True
		):
			raise ValueError(
				'zero distillation and explicit single-head opt-in required'
			)
		if 'spec' in full['head'] or full['head']['num_prototypes'] != 6:
			raise ValueError('single-head K6 configuration required')
		normalized['loss']['distillation_weight'] = base['loss']['distillation_weight']
		del normalized['train'][OPT_IN]
		if normalized != base:
			raise ValueError(
				'non-distillation training settings differ from v1 baseline'
			)
		smoke = yaml.safe_load(Path(arm['configs']['smoke']['path']).read_text())
		expected = deepcopy(full)
		expected['paths']['output_root'] = full['paths']['output_root'].replace(
			'/full_25ep', '/smoke_1step'
		)
		expected['train'].update(epochs=1, max_steps=1)
		if smoke != expected:
			raise ValueError('smoke differs beyond output and duration')
		for kind in ('embeddings', 'downstream'):
			candidate = yaml.safe_load(Path(arm['configs'][kind]['path']).read_text())
			reference_key = (
				'embedding_reference'
				if kind == 'embeddings'
				else 'downstream_reference'
			)
			original = yaml.safe_load(
				Path(arm['configs'][reference_key]['path']).read_text()
			)
			if kind == 'embeddings':
				for field in ('checkpoint', 'output_dir'):
					candidate['embeddings'][field] = original['embeddings'][field]
			else:
				candidate['outputs'] = original['outputs']
				if arm['survey'] == 'f3':
					candidate['candidate'] = original['candidate']
				elif arm['survey'] == 'parihaka':
					candidate['embeddings'] = original['embeddings']
					candidate['inputs']['runs_root'] = original['inputs']['runs_root']
				else:
					candidate['arm'].pop('single_head_training_config')
					for field in (
						'arm_id',
						'checkpoint',
						'embeddings_dir',
						'hmm',
						'multi_head_training_config',
					):
						if field in original['arm']:
							candidate['arm'][field] = original['arm'][field]
			if candidate != original:
				raise ValueError(
					f'non-lineage {kind} settings differ from their reference'
				)
	return matrix


def audit_single_head_source(path: Path) -> dict[str, Any]:
	"""Require the complete final source, unchanged frozen tensors and zero loss."""
	config = resolve_strat_hmm_pretext_config(load_config(path))
	train = config['train']
	if 'spec' in config['head'] or train.get(OPT_IN) is not True:
		raise ValueError('source must be the opt-in single-head ablation')
	checkpoint = Path(config['paths']['output_root']) / 'latest.pt'
	before = file_sha256(checkpoint)
	payload = load_checkpoint(checkpoint, map_location='cpu')
	validate_stratigraphy_checkpoint_payload(payload, expected_config=config)
	if payload.get('stratigraphy_config') != config:
		raise ValueError('checkpoint configuration differs from declared recipe')
	steps = train['epochs'] * math.ceil(
		train['samples_per_epoch'] / train['batch_size']
	)
	state = payload['training_state']
	if (
		train['max_steps'] is not None
		or train['epochs'] != 25
		or payload['epoch'] != 25
		or payload['global_step'] != steps
		or state['checkpoint_kind'] != 'epoch'
		or state.get('batch_index') is not None
	):
		raise ValueError('source is not a completed 25-epoch continuation')
	if (
		config['loss']['distillation_weight'] != 0
		or payload['metrics']['loss_distillation'] != 0
	):
		raise ValueError('zero-distillation source has a nonzero distillation loss')
	_validate_frozen_state(payload, config)
	if file_sha256(checkpoint) != before:
		raise ValueError('checkpoint changed during source validation')
	return {
		'checkpoint': str(checkpoint),
		'checkpoint_sha256': before,
		'parent_checkpoint': config['student']['init_checkpoint'],
		'parent_checkpoint_sha256': file_sha256(config['student']['init_checkpoint']),
		'epoch': 25,
		'global_step': steps,
		'distillation_weight': 0.0,
	}


def audit_volve_single_head_source(
	config: VolveHorizonRecipeArmConfig,
) -> dict[str, object]:
	"""Connect the explicit single-head extension to the existing Volve runner."""
	from seis_ssl_cluster.volve.horizon_five_way_sources import _validate_hmm_checkpoint
	from seis_ssl_cluster.volve.horizon_recipe_arm import as_five_way_config

	evidence = audit_single_head_source(config.single_head_training_config)
	if Path(evidence['checkpoint']).resolve() != config.arm_checkpoint.resolve():
		raise ValueError('single-head source checkpoint differs from Volve arm')
	payload = load_checkpoint(config.arm_checkpoint, map_location='cpu')
	_validate_hmm_checkpoint(
		as_five_way_config(config).models[0], payload, base_config=payload['config']
	)
	return {
		**evidence,
		'model_id': config.arm_id,
		'role': 'recipe_arm',
		'objective': as_five_way_config(config).models[0].expected['objective'],
	}


def _operations() -> Path:
	return (
		Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
		/ 'operations/hmm_v1/zero_distillation_ablation_v1'
	)


def _run(command: list[str], label: str, device: str, *, dry_run: bool) -> None:
	"""Record exact commands and keep each child log separate."""
	print(json.dumps({'label': label, 'gpu': device, 'command': command}), flush=True)
	if dry_run:
		return
	root = _operations() / 'execution'
	root.mkdir(parents=True, exist_ok=True)
	stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
	log = root / f'{stamp}_{label}.log'
	env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED='1')
	with log.open('x') as stream:
		result = subprocess.run(
			command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
		)
	with (root / f'{stamp}_{label}.json').open('x') as stream:
		json.dump(
			{
				'command': command,
				'gpu': device,
				'returncode': result.returncode,
				'log': str(log),
				'finished_utc': datetime.now(timezone.utc).isoformat(),
			},
			stream,
			indent=2,
		)
	if result.returncode:
		raise RuntimeError(f'{label} failed with status {result.returncode}; see {log}')


def _cell_path(arm: dict[str, Any], layout: str, size: str) -> Path:
	raw = load_config(arm['configs']['downstream']['path'])
	return (
		Path(raw['outputs']['runs_root'])
		/ f'model={arm["candidate_id"]}'
		/ f'layout={layout}'
		/ f'size={size}'
	)


def run_arm(arm: dict[str, Any], phase: str, device: str, *, dry_run: bool) -> None:
	"""Prevent conflicting direct and whole-study writers for the same arm."""
	root = _operations() / 'arm_locks'
	root.mkdir(parents=True, exist_ok=True)
	with (root / f'{phase}_{arm["survey"]}_{arm["source"]}.lock').open('a') as lock:
		fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		_run_arm(arm, phase, device, dry_run=dry_run)


def _run_arm(arm: dict[str, Any], phase: str, device: str, *, dry_run: bool) -> None:  # noqa: C901
	"""Execute one fixed arm, resuming only its own interrupted checkpoints."""
	cfg = arm['configs']
	label = f'{phase}_{arm["survey"]}_{arm["source"]}'
	if phase in ('smoke', 'full'):
		path = cfg['smoke' if phase == 'smoke' else 'training']['path']
		raw = load_config(path)
		latest = Path(raw['paths']['output_root']) / 'latest.pt'
		command = [
			sys.executable,
			'proc/seis_ssl_cluster/train_strat_hmm_pretext.py',
			'--config',
			path,
		]
		if latest.exists() and not dry_run:
			payload = load_checkpoint(latest, map_location='cpu')
			if (
				payload['epoch'] == raw['train']['epochs']
				and payload['training_state']['checkpoint_kind'] == 'epoch'
			):
				if phase == 'full':
					audit_single_head_source(Path(path))
				return
			if phase == 'smoke':
				raise ValueError('partial smoke requires inspection')
			command += ['--resume', str(latest)]
		_run(command, label, device, dry_run=dry_run)
		if phase == 'full' and not dry_run:
			audit_single_head_source(Path(path))
	elif phase == 'embeddings':
		if not dry_run:
			audit_single_head_source(Path(cfg['training']['path']))
		command = [
			sys.executable,
			'proc/seis_ssl_cluster/extract_embeddings.py',
			'--config',
			cfg['embeddings']['path'],
			'--device',
			'cuda',
			'--skip-existing',
		]
		_run(command, label, device, dry_run=dry_run)
	else:
		for size in SIZES:
			for layout in LAYOUTS:
				command = [
					sys.executable,
					'-m',
					'seis_ssl_cluster.hmm.multi_source_cell',
					'--survey',
					arm['survey'],
					'--config',
					cfg['downstream']['path'],
					'--candidate',
					arm['candidate_id'],
					'--layout',
					layout,
					'--size',
					size,
				]
				cell = _cell_path(arm, layout, size)
				latest = cell / (
					'decoder/latest.pt' if arm['survey'] == 'f3' else 'latest.pt'
				)
				if latest.exists():
					command += ['--resume', str(latest)]
				_run(command, f'{label}_{layout}_{size}', device, dry_run=dry_run)


def comparison_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Validate the complete 405-cell grid and report paired within-budget effects."""
	keys = [
		(r['survey'], r['source'], float(r['weight']), r['layout_id'], r['data_size'])
		for r in cells
	]
	expected = {
		(s, f, w, l, z)
		for s in SURVEYS
		for f in SOURCES
		for w in (0.0, 0.1, 0.2)
		for l in LAYOUTS
		for z in SIZES
	}
	if len(keys) != len(expected) or set(keys) != expected:
		raise ValueError('exactly 405 unique cells are required')
	if not all(
		math.isfinite(float(r[k])) for r in cells for k in ('primary', 'secondary')
	):
		raise ValueError('nonfinite result')
	lookup = dict(zip(keys, cells, strict=True))
	rows = []
	for s in SURVEYS:
		for f in SOURCES:
			for z in SIZES:
				for w in (0.0, 0.1, 0.2):
					group = [lookup[s, f, w, l, z] for l in LAYOUTS]
					row = {
						'survey': s,
						'source': f,
						'data_size': z,
						'distillation_weight': w,
						'evaluation_split': 'validation' if s == 'f3' else 'test',
						'layouts': 5,
					}
					for field in ('primary', 'secondary'):
						vals = [float(r[field]) for r in group]
						row[field + '_mean'], row[field + '_sample_sd'] = (
							statistics.mean(vals),
							statistics.stdev(vals),
						)
					for reference in (0.1, 0.2):
						diffs = [
							float(lookup[s, f, reference, l, z]['primary'])
							- float(lookup[s, f, w, l, z]['primary'])
							if s == 'volve'
							else float(lookup[s, f, w, l, z]['primary'])
							- float(lookup[s, f, reference, l, z]['primary'])
							for l in LAYOUTS
						]
						row[f'improvement_vs_{reference}_mean'] = statistics.mean(diffs)
						row[f'improvement_vs_{reference}_sample_sd'] = statistics.stdev(
							diffs
						)
					rows.append(row)
	return rows


def publish_summary(output: Path, cells: list[dict[str, Any]]) -> None:
	"""Publish exactly the four owned result files, refusing existing outputs."""
	rows = comparison_rows(cells)
	output.mkdir(parents=True, exist_ok=False)
	_write_rows(output / 'cells.csv', cells)
	_write_rows(output / 'comparison.csv', rows)
	(output / 'summary.json').write_text(
		json.dumps(
			{
				'study': STUDY,
				'status': 'complete',
				'cells': cells,
				'rows': rows,
				'aggregation': (
					'mean and sample SD over five paired layouts; one encoder seed'
				),
				'published_files': sorted(PUBLISHED_FILES),
			},
			indent=2,
		)
		+ '\n'
	)
	lines = [
		'# Single-head K6 distillation 0 / 0.1 / 0.2',
		'',
		(
			'F3: original validation; Parihaka/Volve: test. '
			'Positive improvement favors the current weight.'
		),
		'',
		(
			'| Survey | Source | Budget | Weight | Primary mean ± sample SD | '
			'Improvement vs 0.2 |'
		),
		'| --- | --- | --- | ---: | ---: | ---: |',
	]
	lines.extend(
		f'| {r["survey"]} | {r["source"]} | {r["data_size"]} | '
		f'{r["distillation_weight"]} | {r["primary_mean"]:.6f} ± '
		f'{r["primary_sample_sd"]:.6f} | {r["improvement_vs_0.2_mean"]:+.6f} |'
		for r in rows
	)
	(output / 'summary.md').write_text('\n'.join(lines) + '\n')
	if {p.name for p in output.iterdir()} != PUBLISHED_FILES:
		raise ValueError('published file set differs from explicit ownership')


def summarize(matrix: dict[str, Any]) -> None:
	"""Combine all new cells with the immutable 270-cell v1 control receipt."""
	cells = [
		{
			**r,
			'weight': float(r['weight']),
			'primary': float(r['primary']),
			'secondary': float(r['secondary']),
		}
		for r in _read_rows(Path(matrix['controls']['path']))
	]
	for arm in matrix['arms']:
		audit_single_head_source(Path(arm['configs']['training']['path']))
		raw = load_config(arm['configs']['downstream']['path'])
		for layout in LAYOUTS:
			for size in SIZES:
				audit_completed_cell(
					arm['survey'], raw, arm['candidate_id'], layout, size
				)
				path = _cell_path(arm, layout, size) / (
					'evaluation/metrics.json'
					if arm['survey'] == 'f3'
					else 'metrics.json'
				)
				primary, secondary, _ = _metrics(path, arm['survey'])
				cells.append(
					{
						'survey': arm['survey'],
						'source': arm['source'],
						'weight': 0.0,
						'layout_id': layout,
						'data_size': size,
						'primary': primary,
						'secondary': secondary,
					}
				)
	root = (
		Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
		/ 'studies/hmm_v1/zero_distillation_ablation_v1/summary'
	)
	publish_summary(root, cells)
	print(f'complete: {root}', flush=True)


def main() -> None:
	"""Validate, plan, or execute the prespecified extension without editing v1."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		'--phase',
		choices=(
			'validate',
			'smoke',
			'full',
			'embeddings',
			'downstream',
			'summary',
			'all',
		),
		required=True,
	)
	parser.add_argument('--matrix', type=Path, default=MATRIX)
	parser.add_argument('--survey', choices=SURVEYS)
	parser.add_argument('--source', choices=SOURCES)
	parser.add_argument('--gpus', nargs='+', default=['0'])
	parser.add_argument('--dry-run', action='store_true')
	args = parser.parse_args()
	matrix = validate_matrix(args.matrix)
	if args.phase == 'validate':
		print(
			'valid: nine zero-weight continuations, '
			'135 new downstream cells, 270 reused controls'
		)
		return
	if args.phase == 'summary':
		summarize(matrix)
		return
	arms = [
		a
		for a in matrix['arms']
		if (not args.survey or a['survey'] == args.survey)
		and (not args.source or a['source'] == args.source)
	]
	phases = (
		('full', 'embeddings', 'downstream') if args.phase == 'all' else (args.phase,)
	)
	root = _operations()
	root.mkdir(parents=True, exist_ok=True)
	with (
		root / f'{args.phase}_{args.survey or "all"}_{args.source or "all"}.lock'
	).open('a') as lock:
		fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		for phase in phases:

			def worker(index: int, current_phase: str = phase) -> None:
				for arm in arms[index :: len(args.gpus)]:
					run_arm(arm, current_phase, args.gpus[index], dry_run=args.dry_run)

			with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
				list(pool.map(worker, range(len(args.gpus))))
		if (
			args.phase == 'all'
			and not args.survey
			and not args.source
			and not args.dry_run
		):
			summarize(matrix)


if __name__ == '__main__':
	main()
