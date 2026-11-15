"""Fixed MAE HMM v2 distillation study using the existing survey runners."""

# ruff: noqa: PLC0415, S603
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shlex
import statistics
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.hmm.multi_source_survey import _metrics
from seis_ssl_cluster.training.strat_hmm_source_audit import audit_multi_head_source

STUDY = 'hmm_v2_k6810_distillation_sensitivity_v1'
MATRIX = Path('experiments/hmm_v2/distillation_sensitivity_v1/matrix.yaml')
SURVEYS = ('f3', 'parihaka', 'volve')
WEIGHTS = ('0.0', '0.1', '0.2', '0.4')
LAYOUTS = tuple(f'layout_{i:03d}' for i in range(5))
SIZES = ('small', 'medium', 'large')
PUBLISHED_FILES = frozenset(
	{
		'cells.csv',
		'comparison.csv',
		'summary.json',
		'summary.md',
		'sensitivity.png',
		'feature_diagnostics.csv',
		'f3_validation.csv',
	}
)


def read_rows(path: Path) -> list[dict[str, str]]:
	"""Read a CSV with its original field precision."""
	with path.open(newline='') as stream:
		return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
	"""Write a finite, explicit review table."""
	with path.open('w', newline='') as stream:
		writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
		writer.writeheader()
		writer.writerows(rows)


def validate_matrix(matrix_path: Path = MATRIX) -> dict[str, Any]:  # noqa: C901, PLR0912
	"""Bind every definition and require only the prespecified scientific delta."""
	matrix = load_config(matrix_path)
	weights = (
		('0.0', '0.2') if matrix.get('source') in ('random', 'local_bt') else WEIGHTS
	)
	if matrix['weights'] != [float(w) for w in weights] or set(
		matrix['surveys']
	) != set(SURVEYS):
		raise ValueError('fixed distillation study scope changed')
	for arms in matrix['surveys'].values():
		if set(arms) != set(weights):
			raise ValueError('all prespecified coefficients are required')
		baseline = arms['0.2']
		for weight, arm in arms.items():
			for ref in arm['configs'].values():
				if file_sha256(ref['path']) != ref['sha256']:
					raise ValueError(f'configuration hash drift: {ref["path"]}')
			if weight == '0.2':
				continue
			full = yaml.safe_load(Path(arm['configs']['training']['path']).read_text())
			smoke = yaml.safe_load(Path(arm['configs']['smoke']['path']).read_text())
			expected_smoke = deepcopy(full)
			expected_smoke['paths']['output_root'] = full['paths'][
				'output_root'
			].replace('/full_25ep', '/smoke_1step')
			expected_smoke['train'].update(epochs=1, max_steps=1)
			if smoke != expected_smoke:
				raise ValueError(
					'smoke must retain the full recipe except duration/output'
				)
			for kind in ('training', 'embeddings', 'downstream'):
				path = arm['configs'][kind]['path']
				base_path = baseline['configs'][kind]['path']
				candidate = yaml.safe_load(Path(path).read_text())
				original = yaml.safe_load(Path(base_path).read_text())
				normalized = json.loads(
					json.dumps(candidate)
					.replace(matrix['study'], 'hmm_v2_k6810_cons010_multi_source_v1')
					.replace(arm['candidate_id'], baseline['candidate_id'])
				)
				if kind == 'training':
					if (
						candidate['loss']['distillation_weight'] != float(weight)
						or candidate['train'].get('feature_diagnostics_samples') != 16
					):
						raise ValueError('distillation/diagnostics definition mismatch')
					normalized['loss']['distillation_weight'] = 0.2
					del normalized['train']['feature_diagnostics_samples']
				if kind == 'downstream' and 'arm' in original:
					normalized['arm']['multi_head_training_config'] = original['arm'][
						'multi_head_training_config'
					]
				if normalized != original:
					raise ValueError(f'non-distillation configuration delta: {path}')
	return matrix


def _artifact_root() -> Path:
	return Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])


def _study_root(matrix: dict[str, Any]) -> Path:
	return (
		_artifact_root()
		/ 'studies/hmm_v2'
		/ matrix.get('analysis_id', 'distillation_sensitivity_v1')
	)


def _operations_root(matrix: dict[str, Any]) -> Path:
	return (
		_artifact_root()
		/ 'operations/hmm_v2'
		/ matrix.get('analysis_id', 'distillation_sensitivity_v1')
	)


def _cell_path(survey: str, arm: dict[str, Any], layout: str, size: str) -> Path:
	raw = load_config(arm['configs']['downstream']['path'])
	root = (
		Path(raw['outputs']['runs_root'])
		/ f'model={arm["candidate_id"]}'
		/ f'layout={layout}'
		/ f'size={size}'
	)
	return root / ('evaluation/metrics.json' if survey == 'f3' else 'metrics.json')


def _run(command: list[str], log_root: Path, label: str) -> None:
	"""Keep commands, interruption/resume identities, times and exit status."""
	log_root.mkdir(parents=True, exist_ok=True)
	environment = os.environ.copy()
	if label.startswith(('medium_', 'remaining_')):
		environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
	event = {
		'time': datetime.now(timezone.utc).isoformat(),
		'label': label,
		'command': command,
		'event': 'start',
		'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
		'cpu_threads': {
			key: environment.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')
		},
	}
	if '--resume' in command:
		checkpoint = Path(command[command.index('--resume') + 1])
		event['resume_checkpoint_sha256'] = file_sha256(checkpoint)
	with (log_root / 'events.jsonl').open('a') as stream:
		stream.write(json.dumps(event) + '\n')
	print(shlex.join(command), flush=True)
	with (log_root / f'{label}.lock').open('a') as lock:
		fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		with (log_root / f'{label}.log').open('a') as log:
			result = subprocess.run(
				command,
				stdout=log,
				stderr=subprocess.STDOUT,
				check=False,
				env=environment,
			)
	event.update(
		time=datetime.now(timezone.utc).isoformat(),
		event='end',
		returncode=result.returncode,
	)
	with (log_root / 'events.jsonl').open('a') as stream:
		stream.write(json.dumps(event) + '\n')
	result.check_returncode()


def execute_phase(matrix: dict[str, Any], args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915
	"""Execute one study phase, retaining the all-condition medium-first gate."""
	logs = _operations_root(matrix) / 'execution'
	if args.phase == 'remaining':
		for survey, arms in matrix['surveys'].items():
			for arm in arms.values():
				if arm['execution'] == 'new' and any(
					not _cell_path(survey, arm, layout, 'medium').is_file()
					for layout in LAYOUTS
				):
					raise ValueError(
						f'all {matrix["new_continuations"] * 5} medium cells must '
						'finish before any small/large cell'
					)
	for survey, arms in matrix['surveys'].items():
		if args.survey and survey != args.survey:
			continue
		for weight, arm in arms.items():
			if arm['execution'] != 'new' or (
				args.weight is not None and float(weight) != args.weight
			):
				continue
			cfg = arm['configs']
			label = f'{args.phase}_{survey}_{weight}'
			if args.phase in ('smoke', 'full'):
				recipe = cfg['smoke' if args.phase == 'smoke' else 'training']['path']
				raw = load_config(recipe)
				latest = Path(raw['paths']['output_root']) / 'latest.pt'
				if args.phase == 'full' and latest.exists():
					epochs_path = latest.parent / 'multi_head_epoch_metrics.csv'
					if (
						epochs_path.exists()
						and int(read_rows(epochs_path)[-1]['epoch']) == 25
					):
						if matrix.get('retrospective_audits', True):
							audit_multi_head_source(Path(recipe))
						continue
					if not args.resume:
						raise ValueError('partial training requires --resume')
				command = [
					sys.executable,
					'proc/seis_ssl_cluster/train_strat_hmm_pretext.py',
					'--config',
					recipe,
				]
				if args.resume and latest.exists():
					command += ['--resume', str(latest)]
				if args.dry_run:
					command += ['--dry-run']
				_run(command, logs, label)
			elif args.phase == 'embeddings':
				if matrix.get('retrospective_audits', True):
					audit_multi_head_source(Path(cfg['training']['path']))
				command = [
					sys.executable,
					'proc/seis_ssl_cluster/extract_embeddings.py',
					'--config',
					cfg['embeddings']['path'],
					'--device',
					'cuda',
					'--skip-existing',
				]
				_run([*command, '--dry-run'], logs, label + '_dry_run')
				if not args.dry_run:
					_run(command, logs, label)
			else:
				sizes = ('medium',) if args.phase == 'medium' else ('small', 'large')
				for size in sizes:
					for layout in LAYOUTS:
						if (
							not matrix.get('retrospective_audits', True)
							and _cell_path(survey, arm, layout, size).is_file()
						):
							continue
						command = [
							sys.executable,
							'-m',
							'seis_ssl_cluster.hmm.multi_source_cell',
							'--survey',
							survey,
							'--config',
							cfg['downstream']['path'],
							'--candidate',
							arm['candidate_id'],
							'--layout',
							layout,
							'--size',
							size,
						]
						if args.resume:
							cp = _cell_path(survey, arm, layout, size).parent
							cp = (
								cp.parent / 'decoder/latest.pt'
								if survey == 'f3'
								else cp / 'latest.pt'
							)
							if cp.exists():
								command += ['--resume', str(cp)]
						if args.dry_run:
							command += ['--dry-run']
						elif size == 'medium' and layout == 'layout_000':
							_run([*command, '--dry-run'], logs, label + '_dry_run')
						_run(command, logs, f'{label}_{layout}_{size}')


def prepare_test(matrix: dict[str, Any]) -> Path:
	"""Inventory the fixed F3 selected decoders before common-test evaluation."""
	from seis_ssl_cluster.f3.lithology.common_test import DISTILLATION_NAMESPACE

	output = _study_root(matrix) / 'test_definition'
	output.mkdir(parents=True, exist_ok=False)
	rows = []
	source = matrix.get('source', 'mae')
	for weight, arm in matrix['surveys']['f3'].items():
		raw = load_config(arm['configs']['downstream']['path'])
		cp = Path(raw['candidate']['checkpoint'])
		embedding = Path(raw['candidate']['embeddings_dir'])
		meta = next(embedding.glob('*.embedding_metadata.json'))
		rows.append(
			{
				'audit_id': f'v2__{source}__k6810__dist{weight}__cons010',
				'version': 'v2',
				'source': source,
				'head_ks': '6|8|10',
				'distillation_weight': weight,
				'consistency_weight': '0.1',
				'historical_condition_id': arm['candidate_id'],
				'layouts': '|'.join(LAYOUTS),
				'sizes': '|'.join(SIZES),
				'existing_cells': 15,
				'checkpoint_sha256': file_sha256(cp),
				'embedding_metadata_sha256': file_sha256(meta),
				'checkpoint_artifact_relative': str(cp.relative_to(_artifact_root())),
				'embeddings_artifact_relative': str(
					embedding.relative_to(_artifact_root())
				),
				'historical_runs_artifact_relative': str(
					(
						Path(raw['outputs']['runs_root'])
						/ f'model={arm["candidate_id"]}'
					).relative_to(_artifact_root())
				),
			}
		)
	inventory = output / 'condition_inventory.csv'
	write_rows(inventory, rows)
	config = load_config(
		'experiments/f3/facies_benchmark_v2/136_lithology_common_test_v3_v1/evaluation.yaml'
	)
	config.update(
		experiment_id=matrix.get('test_namespace', DISTILLATION_NAMESPACE),
		output_root=str(
			_artifact_root()
			/ 'surveys/f3/facies_benchmark_v2/evaluation'
			/ matrix.get('test_namespace', DISTILLATION_NAMESPACE)
		),
		expected_conditions=len(matrix['weights']),
		expected_cells=len(matrix['weights']) * 15,
	)
	config['inputs']['inventory'] = {
		'path': str(inventory),
		'sha256': file_sha256(inventory),
	}
	path = output / 'evaluate.yaml'
	path.write_text(yaml.safe_dump(config, sort_keys=False))
	return path


def comparison_rows(
	cells: list[dict[str, Any]], *, weights: tuple[str, ...] = WEIGHTS
) -> list[dict[str, Any]]:
	"""Compute five-layout sample SD and paired improvements within each budget."""
	expected = {
		(s, float(w), l, z)
		for s in SURVEYS
		for w in weights
		for l in LAYOUTS
		for z in SIZES
	}
	actual = [
		(c['survey'], c['distillation_weight'], c['layout_id'], c['data_size'])
		for c in cells
	]
	if len(actual) != len(expected) or set(actual) != expected:
		raise ValueError(f'exactly {len(expected)} unique test cells are required')
	rows = []
	for survey in SURVEYS:
		for size in SIZES:
			for weight in weights:
				group = [
					c
					for c in cells
					if (c['survey'], c['data_size'], c['distillation_weight'])
					== (survey, size, float(weight))
				]
				row = {
					'survey': survey,
					'data_size': size,
					'distillation_weight': float(weight),
					'layout_count': 5,
					'evaluation_split': 'common_test' if survey == 'f3' else 'test',
				}
				for field in ('primary', 'secondary', 'primary_improvement_vs_020'):
					row[field + '_mean'] = statistics.mean(c[field] for c in group)
					row[field + '_sample_sd'] = statistics.stdev(
						c[field] for c in group
					)
				rows.append(row)
	return rows


def collect_summary(matrix: dict[str, Any]) -> dict[str, Any]:
	"""Aggregate runner outputs using the already recorded experiment lineage."""
	from seis_ssl_cluster.f3.lithology.common_test import DISTILLATION_NAMESPACE

	test = read_rows(
		_artifact_root()
		/ 'surveys/f3/facies_benchmark_v2/evaluation'
		/ matrix.get('test_namespace', DISTILLATION_NAMESPACE)
		/ 'summary/per_layout.csv'
	)
	previous_test = read_rows(
		_artifact_root()
		/ 'surveys/f3/facies_benchmark_v2/evaluation'
		/ 'lithology_common_test_v1/summary/per_layout.csv'
	)
	for row in test:
		if float(row['distillation_weight']) != 0.2:
			continue
		previous = next(
			r
			for r in previous_test
			if (
				r['version'],
				r['source'],
				r['head_ks'],
				r['distillation_weight'],
				r['consistency_weight'],
				r['layout_id'],
				r['data_size'],
			)
			== (
				'v2',
				matrix.get('source', 'mae'),
				'6|8|10',
				'0.2',
				'0.1',
				row['layout_id'],
				row['data_size'],
			)
		)
		if any(
			float(row[key]) != float(previous[key]) for key in ('mean_iou', 'macro_f1')
		):
			raise ValueError('rescored F3 0.2 differs from original common-test result')
	cells, evidence, validation = [], [], []
	for survey, arms in matrix['surveys'].items():
		baseline = arms['0.2']
		for weight, arm in arms.items():
			for layout in LAYOUTS:
				for size in SIZES:
					values = [
						_metrics(_cell_path(survey, a, layout, size), survey)
						for a in (arm, baseline)
					]
					primary, secondary = values[0][:2]
					control = values[1][0]
					if survey == 'f3':
						validation.append(
							{
								'survey': survey,
								'distillation_weight': float(weight),
								'layout_id': layout,
								'data_size': size,
								'validation_mean_iou': primary,
								'validation_macro_f1': secondary,
							}
						)
						selected = [
							next(
								r
								for r in test
								if (
									float(r['distillation_weight']),
									r['layout_id'],
									r['data_size'],
								)
								== (float(w), layout, size)
							)
							for w in (weight, '0.2')
						]
						primary, secondary, control = (
							float(selected[0]['mean_iou']),
							float(selected[0]['macro_f1']),
							float(selected[1]['mean_iou']),
						)
					cells.append(
						{
							'survey': survey,
							'distillation_weight': float(weight),
							'layout_id': layout,
							'data_size': size,
							'primary': primary,
							'secondary': secondary,
							'primary_improvement_vs_020': (
								control - primary
								if survey == 'volve'
								else primary - control
							),
							'reused_encoder': weight == '0.2',
						}
					)
					evidence.append(
						{
							'survey': survey,
							'weight': weight,
							'layout': layout,
							'size': size,
							'candidate_id': arm['candidate_id'],
							'configs': arm['configs'],
							'metrics_paths': [
								str(_cell_path(survey, a, layout, size))
								for a in (arm, baseline)
							],
							'metrics_sha256': [v[2] for v in values],
						}
					)
	operations = _operations_root(matrix)
	weights = tuple(str(w) for w in matrix.get('weights', [0.0, 0.1, 0.2, 0.4]))
	return {
		'study': matrix.get('study', STUDY),
		'source': matrix.get('source', 'mae'),
		'weights': weights,
		'status': 'complete',
		'cells': cells,
		'rows': comparison_rows(cells, weights=weights),
		'validation': validation,
		'evidence': evidence,
		'aggregation_policy': (
			'reuse recorded lineage; no retrospective experiment audits'
		),
		'recorded_evidence': [
			{'path': str(operations / name), 'sha256': file_sha256(operations / name)}
			for name in ('preflight.json', 'experiments_complete.json')
		],
		'feature_diagnostics': collect_diagnostics(matrix),
	}


def collect_diagnostics(matrix: dict[str, Any]) -> list[dict[str, Any]]:
	"""Keep fixed-crop distances separate from changing training-crop losses."""
	rows = []
	for survey, arms in matrix['surveys'].items():
		crop_hash = None
		for weight, arm in arms.items():
			config = load_config(arm['configs']['training']['path'])
			training = Path(config['paths']['output_root'])
			diagnostic = (
				_study_root(matrix) / 'baseline_diagnostics' / survey
				if weight == '0.2'
				else training
			)
			fixed = {
				int(r['epoch']): r
				for r in read_rows(diagnostic / 'feature_diagnostics.csv')
			}
			expected = {0, 25} if weight == '0.2' else set(range(26))
			if set(fixed) != expected:
				raise ValueError('missing fixed-crop feature diagnostics')
			hashes = {r['crop_sha256'] for r in fixed.values()}
			if len(hashes) != 1 or (crop_hash is not None and hashes != {crop_hash}):
				raise ValueError('diagnostic crops differ across coefficients')
			crop_hash = next(iter(hashes))
			losses = {
				int(r['epoch']): r
				for r in read_rows(training / 'multi_head_epoch_metrics.csv')
			}
			if set(losses) != set(range(1, 26)):
				raise ValueError('missing training loss diagnostics')
			for epoch in range(26):
				row = {
					'survey': survey,
					'distillation_weight': float(weight),
					'epoch': epoch,
					'fixed_crop_teacher_cosine_distance': float(
						fixed[epoch]['teacher_cosine_distance']
					)
					if epoch in fixed
					else '',
					'fixed_crop_sha256': crop_hash,
				}
				for name in ('prototype', 'usage', 'consistency', 'distillation'):
					value = float(losses[epoch][f'loss_{name}']) if epoch else ''
					row[f'loss_{name}_raw'] = value
					row[f'loss_{name}_weighted'] = (
						value * config['loss'][f'{name}_weight'] if epoch else ''
					)
				for head in (6, 8, 10):
					key = f'prototype_usage_entropy_k{head}'
					row[key] = float(losses[epoch][key]) if epoch else ''
				for pair in ('k6_k8', 'k6_k10', 'k8_k10'):
					key = f'loss_consistency_{pair}'
					row[key] = float(losses[epoch][key]) if epoch else ''
				rows.append(row)
	return rows


def publish_summary(output: Path, payload: dict[str, Any]) -> None:
	"""Publish the exact seven review files to artifacts."""
	import matplotlib.pyplot as plt

	weights = tuple(payload.get('weights', WEIGHTS))
	comparison_rows(payload['cells'], weights=weights)
	if 'reports' in output.resolve().parts:
		raise ValueError('pipeline output must be in artifacts')
	output.mkdir(parents=True, exist_ok=False)
	write_rows(output / 'cells.csv', payload['cells'])
	write_rows(output / 'comparison.csv', payload['rows'])
	write_rows(output / 'feature_diagnostics.csv', payload['feature_diagnostics'])
	write_rows(output / 'f3_validation.csv', payload['validation'])
	(output / 'summary.json').write_text(
		json.dumps(payload, indent=2, allow_nan=False) + '\n'
	)
	lines = [
		'# Fixed-seed HMM v2 distillation sensitivity',
		'',
		(
			'Five layouts share one encoder per coefficient and survey. '
			'Budgets are nested. Sample SD uses ddof=1. '
			'Test results do not select coefficients or checkpoints.'
		),
		'',
		'| Survey | Budget | Weight | Primary mean ± sample SD | Improvement vs 0.2 |',
		'| --- | --- | ---: | ---: | ---: |',
	]
	lines.extend(
		f'| {row["survey"]} | {row["data_size"]} | {row["distillation_weight"]} | '
		f'{row["primary_mean"]:.6f} ± {row["primary_sample_sd"]:.6f} | '
		f'{row["primary_improvement_vs_020_mean"]:+.6f} |'
		for row in payload['rows']
	)
	(output / 'summary.md').write_text('\n'.join(lines) + '\n')
	fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
	for i, survey in enumerate(SURVEYS):
		for j, size in enumerate(SIZES):
			rows = [
				r
				for r in payload['rows']
				if (r['survey'], r['data_size']) == (survey, size)
			]
			ax = axes[i, j]
			ax.errorbar(
				[r['distillation_weight'] for r in rows],
				[r['primary_mean'] for r in rows],
				yerr=[r['primary_sample_sd'] for r in rows],
				marker='o',
				capsize=3,
			)
			ax.set(
				title=f'{survey} / {size}',
				xlabel='Distillation weight',
				ylabel={
					'f3': 'Common-test mean IoU',
					'parihaka': 'Test Channel IoU',
					'volve': 'Test macro MAE [sample]',
				}[survey],
				xticks=[float(w) for w in weights],
			)
	fig.savefig(output / 'sensitivity.png', dpi=150)
	plt.close(fig)


def main() -> None:
	"""Run the fixed study's existing training, extraction and evaluation routes."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--matrix', type=Path, default=MATRIX)
	parser.add_argument(
		'--phase',
		choices=(
			'validate',
			'smoke',
			'full',
			'embeddings',
			'medium',
			'remaining',
			'test',
			'summary',
		),
		required=True,
	)
	parser.add_argument('--survey', choices=SURVEYS)
	parser.add_argument('--weight', type=float, choices=(0.0, 0.1, 0.4))
	parser.add_argument('--resume', action='store_true')
	parser.add_argument('--dry-run', action='store_true')
	args = parser.parse_args()
	matrix = validate_matrix(args.matrix)
	if args.phase == 'validate':
		print(
			f'{matrix["new_continuations"]} new continuations, '
			f'{matrix["new_downstream_cells"]} new decoder cells, '
			f'{matrix["final_cells"]} final cells; definitions validated'
		)
	elif args.phase == 'test':
		from seis_ssl_cluster.f3.lithology.common_test import (
			inspect_test_plan,
			run_test,
		)

		path = _study_root(matrix) / 'test_definition/evaluate.yaml'
		if not path.exists():
			path = prepare_test(matrix)
		plan = inspect_test_plan(path)
		if args.dry_run:
			print(f'common-test audit passed: {len(plan.cells)} cells; scoring skipped')
		else:
			run_test(plan, resume=args.resume)
	elif args.phase == 'summary':
		payload = collect_summary(matrix)
		if not args.dry_run:
			publish_summary(
				_study_root(matrix) / 'summary',
				payload,
			)
	else:
		execute_phase(matrix, args)


if __name__ == '__main__':
	main()
