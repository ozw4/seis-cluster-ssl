"""Complete missing Barlow zero-distillation cells and aligned F3 v1 test scores."""

# ruff: noqa: C901, PLR0912, PLC0415
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology.common_test import (
	BARLOW_ZERO_DISTILLATION_NAMESPACE,
	V1_ZERO_DISTILLATION_NAMESPACE,
	inspect_test_plan,
	run_test,
)
from seis_ssl_cluster.hmm.distillation_sensitivity import (
	LAYOUTS,
	SIZES,
	_cell_path,
	_run,
	read_rows,
	write_rows,
)
from seis_ssl_cluster.hmm.distillation_sensitivity import (
	validate_matrix as validate_barlow,
)
from seis_ssl_cluster.hmm.multi_source_survey import _metrics

MATRIX = Path('experiments/hmm_v2/missing_controls_completion_v1/matrix.yaml')
STUDY = 'missing_controls_completion_v1'
PUBLISHED_FILES = frozenset(
	{
		'barlow_cells.csv',
		'barlow_comparison.csv',
		'v1_cells.csv',
		'v1_comparison.csv',
		'factorial_cells.csv',
		'factorial_comparison.csv',
		'factorial_contrasts.csv',
		'provenance.json',
		'summary.json',
		'summary.md',
	}
)
REVIEW_COPIES = frozenset(
	{
		'factorial_cells.csv',
		'factorial_comparison.csv',
		'factorial_contrasts.csv',
		'barlow_comparison.csv',
		'v1_comparison.csv',
	}
)
REVIEW_FILES = REVIEW_COPIES | {'README.md', 'provenance.json'}


def root() -> Path:
	"""Locate canonical artifacts through the authorized execution environment."""
	return Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])


def operations() -> Path:
	"""Keep execution state separate from immutable scientific outputs."""
	return root() / 'operations/hmm_v2' / STUDY


def study_root() -> Path:
	"""Return the artifact-side study owner."""
	return root() / 'studies/hmm_v2' / STUDY


def save_status(status: str, **values: object) -> None:
	"""Atomically expose phase, progress and failures for later status requests."""
	operations().mkdir(parents=True, exist_ok=True)
	path = operations() / 'status.json'
	temporary = path.with_suffix('.tmp')
	temporary.write_text(
		json.dumps(
			dict(status=status, utc=datetime.now(timezone.utc).isoformat(), **values),
			indent=2,
		)
		+ '\n'
	)
	temporary.replace(path)


def validate() -> dict[str, Any]:
	"""Bind fixed inputs without using test outcomes to select missing cells."""
	matrix = load_config(MATRIX)
	if matrix['study'] != STUDY or matrix['new_decoder_cells'] != 42:
		raise ValueError('fixed completion scope changed')
	for ref in matrix['inputs'].values():
		if (
			file_sha256(ref['path']) != ref['sha256']
			or 'reports' in Path(ref['path']).parts
		):
			raise ValueError(f'input identity drift: {ref["path"]}')
	barlow = validate_barlow(Path(matrix['inputs']['barlow_matrix']['path']))
	if barlow['scope'] != 'paper figures: layout_000/large only':
		raise ValueError('expected the incomplete figure-only Barlow source study')
	return matrix


def prepare_test(kind: str) -> Path:
	"""Freeze inventories of all selected decoders before common-test scoring."""
	matrix = validate()
	barlow = load_config(matrix['inputs']['barlow_matrix']['path'])
	v1 = load_config(matrix['inputs']['v1_matrix']['path'])
	if kind == 'v1':
		arms = [(a['source'], '0.0', a) for a in v1['arms'] if a['survey'] == 'f3']
		namespace = V1_ZERO_DISTILLATION_NAMESPACE
	else:
		arms = [('local_bt', w, a) for w, a in barlow['surveys']['f3'].items()]
		namespace = BARLOW_ZERO_DISTILLATION_NAMESPACE
	rows = []
	for source, weight, arm in arms:
		cfg = load_config(arm['configs']['downstream']['path'])
		cp = Path(cfg['candidate']['checkpoint'])
		embedding = Path(cfg['candidate']['embeddings_dir'])
		metadata = next(embedding.glob('*.embedding_metadata.json'))
		version, heads, consistency = (
			('v1', '6', '0') if kind == 'v1' else ('v2', '6|8|10', '0.1')
		)
		rows.append(
			{
				'audit_id': f'{version}__{source}__dist{weight}__cons{consistency}',
				'version': version,
				'source': source,
				'head_ks': heads,
				'distillation_weight': weight,
				'consistency_weight': consistency,
				'historical_condition_id': arm['candidate_id'],
				'layouts': '|'.join(LAYOUTS),
				'sizes': '|'.join(SIZES),
				'existing_cells': 15,
				'checkpoint_sha256': file_sha256(cp),
				'embedding_metadata_sha256': file_sha256(metadata),
				'checkpoint_artifact_relative': str(cp.relative_to(root())),
				'embeddings_artifact_relative': str(embedding.relative_to(root())),
				'historical_runs_artifact_relative': str(
					(
						Path(cfg['outputs']['runs_root'])
						/ f'model={arm["candidate_id"]}'
					).relative_to(root())
				),
			}
		)
	output = study_root() / 'test_definitions' / kind
	output.mkdir(parents=True, exist_ok=True)
	inventory = output / 'condition_inventory.csv'
	if not inventory.exists():
		write_rows(inventory, rows)
	elif read_rows(inventory) != [{k: str(v) for k, v in row.items()} for row in rows]:
		raise ValueError('existing test inventory drift')
	config = load_config(matrix['inputs']['test_template']['path'])
	config.update(
		experiment_id=namespace,
		output_root=str(
			root() / 'surveys/f3/facies_benchmark_v2/evaluation' / namespace
		),
		expected_conditions=len(rows),
		expected_cells=len(rows) * 15,
	)
	config['inputs']['inventory'] = {
		'path': str(inventory),
		'sha256': file_sha256(inventory),
	}
	path = output / 'evaluation.yaml'
	if path.exists():
		if load_config(path) != config:
			raise ValueError('existing test definition drift')
	else:
		path.write_text(yaml.safe_dump(config, sort_keys=False))
	return path


def score_test(kind: str) -> None:
	"""Reuse the fixed mask and validation-selected predictions with strict resume."""
	path = prepare_test(kind)
	plan = inspect_test_plan(path)
	run_test(plan, resume=Path(plan.config['output_root']).exists())


def decoder_command(
	survey: str, arm: dict[str, Any], layout: str, size: str
) -> list[str]:
	"""Use the original source-specific recipe and only this cell's checkpoint."""
	command = [
		sys.executable,
		'-m',
		'seis_ssl_cluster.hmm.multi_source_cell',
		'--survey',
		survey,
		'--config',
		arm['configs']['downstream']['path'],
		'--candidate',
		arm['candidate_id'],
		'--layout',
		layout,
		'--size',
		size,
	]
	cell = _cell_path(survey, arm, layout, size)
	checkpoint = (
		cell.parent.parent / 'decoder/latest.pt'
		if survey == 'f3'
		else cell.parent / 'latest.pt'
	)
	if checkpoint.exists() and not cell.exists():
		command += ['--resume', str(checkpoint)]
	return command


def plan_decoders(matrix: dict[str, Any]) -> list[dict[str, Any]]:
	"""List the full fixed matrix, including the three immutable reused cells."""
	barlow = load_config(matrix['inputs']['barlow_matrix']['path'])
	return [
		{
			'survey': survey,
			'layout': layout,
			'size': size,
			'complete': _cell_path(survey, arms['0.0'], layout, size).exists(),
			'command': decoder_command(survey, arms['0.0'], layout, size),
		}
		for size in ('medium', 'small', 'large')
		for survey, arms in barlow['surveys'].items()
		for layout in LAYOUTS
	]


def means(cells: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
	"""Require all five distinct layouts in every prespecified condition."""
	groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
	for cell in cells:
		groups[tuple(cell[k] for k in keys)].append(cell)
	rows = []
	for _, group in sorted(groups.items()):
		if len(group) != 5 or {r['layout_id'] for r in group} != set(LAYOUTS):
			raise ValueError('each condition requires exactly five paired layouts')
		row = {k: group[0][k] for k in keys}
		row['n'] = 5
		for metric in ('primary', 'secondary'):
			vals = [float(r[metric]) for r in group]
			if not all(math.isfinite(v) for v in vals):
				raise ValueError('nonfinite metric')
			row[metric + '_mean'], row[metric + '_sample_sd'] = (
				statistics.mean(vals),
				statistics.stdev(vals),
			)
		rows.append(row)
	return rows


def collect_results(matrix: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
	"""Keep all conditions; align F3 controls on the exact existing common test."""
	barlow = load_config(matrix['inputs']['barlow_matrix']['path'])
	f3_base = read_rows(
		root()
		/ 'surveys/f3/facies_benchmark_v2/evaluation/lithology_common_test_v1'
		/ 'summary/per_layout.csv'
	)
	v1_new = read_rows(
		root()
		/ 'surveys/f3/facies_benchmark_v2/evaluation'
		/ V1_ZERO_DISTILLATION_NAMESPACE
		/ 'summary/per_layout.csv'
	)
	bt_test = read_rows(
		root()
		/ 'surveys/f3/facies_benchmark_v2/evaluation'
		/ BARLOW_ZERO_DISTILLATION_NAMESPACE
		/ 'summary/per_layout.csv'
	)
	bt_cells = []
	for survey, arms in barlow['surveys'].items():
		for weight, arm in arms.items():
			for layout in LAYOUTS:
				for size in SIZES:
					path = _cell_path(survey, arm, layout, size)
					primary, secondary, _digest = _metrics(path, survey)
					if survey == 'f3':
						r = next(
							r
							for r in bt_test
							if r['distillation_weight'] == weight
							and r['layout_id'] == layout
							and r['data_size'] == size
						)
						primary, secondary = float(r['mean_iou']), float(r['macro_f1'])
					bt_cells.append(
						{
							'survey': survey,
							'source': 'local_bt',
							'distillation_weight': float(weight),
							'consistency_weight': 0.1,
							'layout_id': layout,
							'data_size': size,
							'primary': primary,
							'secondary': secondary,
							'evaluation_split': 'common_test'
							if survey == 'f3'
							else 'test',
						}
					)
	for r in bt_test:
		if float(r['distillation_weight']) == 0.2:
			old = next(
				x
				for x in f3_base
				if x['version'] == 'v2'
				and x['source'] == 'local_bt'
				and x['head_ks'] == '6|8|10'
				and x['consistency_weight'] == '0.1'
				and x['layout_id'] == r['layout_id']
				and x['data_size'] == r['data_size']
			)
			if any(float(old[k]) != float(r[k]) for k in ('mean_iou', 'macro_f1')):
				raise ValueError('reused 0.2 common-test control changed')
	v1_cells = read_rows(Path(matrix['inputs']['v1_cells']['path']))
	for r in v1_cells:
		r['distillation_weight'] = float(r.pop('weight'))
		r['primary'], r['secondary'] = float(r['primary']), float(r['secondary'])
		r['evaluation_split'] = 'common_test' if r['survey'] == 'f3' else 'test'
		if r['survey'] == 'f3':
			table = v1_new if r['distillation_weight'] == 0 else f3_base
			item = next(
				x
				for x in table
				if x['version'] == 'v1'
				and x['source'] == r['source']
				and float(x['distillation_weight']) == r['distillation_weight']
				and x['layout_id'] == r['layout_id']
				and x['data_size'] == r['data_size']
			)
			r['primary'], r['secondary'] = (
				float(item['mean_iou']),
				float(item['macro_f1']),
			)
	factor = []
	for name, cons in (
		('k6810_multi_source_evaluation_v1', 0.0),
		('k6810_cons010_multi_source_evaluation_v1', 0.1),
	):
		payload = json.loads(
			(root() / 'studies/hmm_v2' / name / 'summary/summary.json').read_text()
		)
		for c in payload['cells']:
			r = {
				'survey': c['survey'],
				'source': c['source_family'],
				'distillation_weight': 0.2,
				'consistency_weight': cons,
				'layout_id': c['layout_id'],
				'data_size': c['data_size'],
				'primary': c['primary'],
				'secondary': c['secondary'],
				'evaluation_split': 'common_test' if c['survey'] == 'f3' else 'test',
			}
			if r['survey'] == 'f3':
				item = next(
					x
					for x in f3_base
					if x['version'] == 'v2'
					and x['source'] == r['source']
					and x['head_ks'] == '6|8|10'
					and float(x['consistency_weight']) == cons
					and x['layout_id'] == r['layout_id']
					and x['data_size'] == r['data_size']
				)
				r['primary'], r['secondary'] = (
					float(item['mean_iou']),
					float(item['macro_f1']),
				)
			factor.append(r)
	factor.extend(
		dict(
			r,
			distillation_weight=0.0,
			consistency_weight=0.0,
			primary=float(r['primary']),
			secondary=float(r['secondary']),
		)
		for r in read_rows(root() / 'studies/hmm_v2/v2_dis0_con0_v1/summary/cells.csv')
	)
	for source, name in (
		('mae', 'distillation_sensitivity_v1'),
		('random', 'random_distillation_ablation_v1'),
	):
		factor.extend(
			{k: r[k] for k in ('survey', 'layout_id', 'data_size')}
			| {
				'source': source,
				'distillation_weight': 0.0,
				'consistency_weight': 0.1,
				'primary': float(r['primary']),
				'secondary': float(r['secondary']),
				'evaluation_split': 'common_test' if r['survey'] == 'f3' else 'test',
			}
			for r in read_rows(root() / 'studies/hmm_v2' / name / 'summary/cells.csv')
			if float(r['distillation_weight']) == 0
		)
	factor.extend(r for r in bt_cells if r['distillation_weight'] == 0)
	if (
		len(factor) != 540
		or len(
			{
				(
					r['survey'],
					r['source'],
					r['distillation_weight'],
					r['consistency_weight'],
					r['layout_id'],
					r['data_size'],
				)
				for r in factor
			}
		)
		!= 540
	):
		raise ValueError('complete unique 540-cell factorial grid required')
	keys = (
		'survey',
		'source',
		'data_size',
		'distillation_weight',
		'consistency_weight',
		'evaluation_split',
	)
	return {
		'barlow_cells': bt_cells,
		'barlow_comparison': means(bt_cells, keys),
		'v1_cells': v1_cells,
		'v1_comparison': means(
			v1_cells, tuple(k for k in keys if k != 'consistency_weight')
		),
		'factorial_cells': factor,
		'factorial_comparison': means(factor, keys),
		'factorial_contrasts': contrasts(factor),
	}


def contrasts(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Report within-layout effects and interaction in each survey's original units."""
	lookup = {
		(
			r['survey'],
			r['source'],
			r['data_size'],
			r['layout_id'],
			r['distillation_weight'],
			r['consistency_weight'],
		): float(r['primary'])
		for r in cells
	}
	rows = []
	for survey in ('f3', 'parihaka', 'volve'):
		for source in ('mae', 'local_bt', 'random'):
			for size in SIZES:
				by_contrast = defaultdict(list)
				for layout in LAYOUTS:
					v = {
						(d, c): lookup[survey, source, size, layout, d, c]
						for d in (0.0, 0.2)
						for c in (0.0, 0.1)
					}
					sign = -1 if survey == 'volve' else 1
					e0 = sign * (v[0.0, 0.0] - v[0.2, 0.0])
					e1 = sign * (v[0.0, 0.1] - v[0.2, 0.1])
					for label, value in {
						'remove_distillation_at_cons000': e0,
						'remove_distillation_at_cons010': e1,
						'add_consistency_at_dist000': sign
						* (v[0.0, 0.1] - v[0.0, 0.0]),
						'add_consistency_at_dist020': sign
						* (v[0.2, 0.1] - v[0.2, 0.0]),
						'interaction': e1 - e0,
					}.items():
						by_contrast[label].append(value)
				for label, values in by_contrast.items():
					rows.append(
						{
							'survey': survey,
							'source': source,
							'data_size': size,
							'contrast': label,
							'n': 5,
							'paired_effect_mean': statistics.mean(values),
							'paired_effect_sample_sd': statistics.stdev(values),
						}
					)
	return rows


def publish(matrix: dict[str, Any]) -> None:
	"""Atomically publish the exact owned set with all planned outcomes retained."""
	output = study_root() / 'summary'
	if output.exists():
		if {p.name for p in output.iterdir()} != PUBLISHED_FILES:
			raise ValueError('incomplete prior publication')
		return
	results = collect_results(matrix)
	output.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(
		prefix='.summary-', dir=output.parent
	) as temporary:
		stage = Path(temporary)
		for name, rows in results.items():
			write_rows(stage / (name + '.csv'), rows)
		(stage / 'provenance.json').write_text(
			json.dumps(
				{
					'matrix_sha256': file_sha256(MATRIX),
					'inputs': matrix['inputs'],
					'test_definitions': {
						k: file_sha256(prepare_test(k)) for k in ('v1', 'barlow')
					},
				},
				indent=2,
			)
			+ '\n'
		)
		(stage / 'summary.json').write_text(
			json.dumps(
				{
					'status': 'complete',
					'new_decoder_cells': 42,
					'reused_barlow_cells': 3,
					'v1_common_test_cells': 45,
					'barlow_common_test_cells': 30,
					'factorial_cells': 540,
					'v1_aligned_cells': 405,
					'published_files': sorted(PUBLISHED_FILES),
					'limitations': (
						'one encoder continuation seed; layouts are not encoder seeds; '
						'F3 transductive label holdout'
					),
				},
				indent=2,
			)
			+ '\n'
		)
		lines = [
			'# Missing-control completion',
			'',
			(
				'All prespecified cells are retained. F3 uses common test; '
				'other surveys use test.'
			),
			(
				'Positive contrast favors the named change; '
				'interaction is a difference of effects.'
			),
			(
				'Five paired layouts, sample SD (ddof=1), seed 42; '
				'nested budgets are not independent.'
			),
			'',
			'## Barlow Twins: distillation 0 vs 0.2',
			'',
			'| Survey | Budget | Distillation | Primary mean ± sample SD |',
			'| --- | --- | ---: | ---: |',
		]
		lines += [
			f'| {r["survey"]} | {r["data_size"]} | {r["distillation_weight"]} | '
			f'{r["primary_mean"]:.6f} ± {r["primary_sample_sd"]:.6f} |'
			for r in results['barlow_comparison']
		]
		lines += [
			'',
			(
				'The full 2x2 matrix is in factorial_cells.csv '
				'and factorial_comparison.csv.'
			),
			'Paired changes and their interaction are in factorial_contrasts.csv.',
			('Aligned v1 comparisons are in v1_cells.csv and v1_comparison.csv.'),
		]
		(stage / 'summary.md').write_text('\n'.join(lines) + '\n')
		if {p.name for p in stage.iterdir()} != PUBLISHED_FILES:
			raise ValueError('unexpected publication files')
		stage.replace(output)


def publish_review(summary: Path, output: Path) -> None:
	"""Curate the exact small review set without using reports as pipeline inputs."""
	if 'reports' in summary.resolve().parts:
		raise ValueError('review sources must be artifact originals')
	if {p.name for p in summary.iterdir()} != PUBLISHED_FILES:
		raise ValueError('review requires the complete artifact summary')
	metadata = json.loads((summary / 'summary.json').read_text())
	if metadata.get('status') != 'complete' or metadata.get('factorial_cells') != 540:
		raise ValueError('review requires the completed 540-cell comparison')
	if output.exists():
		raise FileExistsError(output)
	lines = [
		'# HMM distillation and consistency: completed controls',
		'',
		'All 42 missing decoder trainings and 75 F3 common-test evaluations completed.',
		'The 540-cell matrix retains both distillation weights (0/0.2) and both',
		'consistency weights (0/0.1) for every survey and initialization.',
		'',
		'- [All 540 cells](factorial_cells.csv)',
		'- [108 condition means and sample SDs](factorial_comparison.csv)',
		'- [135 paired effects and interactions](factorial_contrasts.csv)',
		'- [Barlow distillation comparison](barlow_comparison.csv)',
		'- [Aligned v1 comparison](v1_comparison.csv)',
		'',
		'F3 uses the common test; Parihaka and Volve use test. Metrics remain in',
		'their original units. Positive paired effects favor the named change.',
		'Sample SD uses five paired layouts (ddof=1). Nested budgets are not',
		'independent repetitions. There is one continuation seed (42), and F3 is',
		'a same-survey transductive label holdout.',
		'',
		'[Experiment and recovery record]',
		'(../../experiments/hmm_v2/missing_controls_completion_v1/README.md).',
		'[Source hashes](provenance.json). The exact review set contains seven files.',
	]
	# Keep the Markdown link on one line.
	lines[-3:-1] = [lines[-3] + lines[-2]]
	output.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(prefix='.review-', dir=output.parent) as tmp:
		stage = Path(tmp)
		for name in REVIEW_COPIES:
			(stage / name).write_bytes((summary / name).read_bytes())
		(stage / 'README.md').write_text('\n'.join(lines) + '\n')
		(stage / 'provenance.json').write_text(
			json.dumps(
				{
					'artifact_summary': str(
						summary.resolve().relative_to(root().resolve())
					),
					'summary_sha256': file_sha256(summary / 'summary.json'),
					'lineage_sha256': file_sha256(summary / 'provenance.json'),
					'source_file_sha256': {
						name: file_sha256(summary / name)
						for name in sorted(REVIEW_COPIES)
					},
					'published_files': sorted(REVIEW_FILES),
				},
				indent=2,
			)
			+ '\n'
		)
		if {p.name for p in stage.iterdir()} != REVIEW_FILES:
			raise ValueError('unexpected review publication files')
		stage.rename(output)


def run(gpu: str) -> None:
	"""Run CPU test scoring alongside the fixed medium-first decoder completion."""
	from seis_ssl_cluster.hmm.v2_dis0_con0 import gpu_available

	matrix = validate()
	logs = operations() / 'execution'
	logs.mkdir(parents=True, exist_ok=True)
	with (operations() / 'study.lock').open('a') as lock:
		fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		save_status(
			'running', phase='v1-test and Barlow decoders', pid=os.getpid(), gpu=gpu
		)
		with (logs / 'v1_test.log').open('a') as test_log:
			cpu = subprocess.Popen(
				[
					sys.executable,
					'-m',
					'seis_ssl_cluster.hmm.missing_controls',
					'--phase',
					'v1-test',
				],
				stdout=test_log,
				stderr=subprocess.STDOUT,
			)
			try:
				for i, job in enumerate(plan_decoders(matrix), 1):
					if cpu.poll() not in (None, 0):
						raise RuntimeError(  # noqa: TRY301
							'owned v1 test process failed; see v1_test.log'
						)
					if not job['complete']:
						while not gpu_available(gpu, 12 * 1024):
							save_status('queued', phase='decoder', gpu=gpu, cell=job)
							time.sleep(30)
					prefix = 'medium' if job['size'] == 'medium' else 'remaining'
					label = f'{prefix}_{job["survey"]}_{job["layout"]}_{job["size"]}'
					save_status(
						'running',
						phase='decoder',
						cell_index=i,
						total_cells=45,
						cell=job,
						gpu=gpu,
					)
					if not job['complete']:
						_run([*job['command'], '--dry-run'], logs, label + '_dry_run')
					_run(job['command'], logs, label)
				if cpu.wait() != 0:
					raise RuntimeError('v1 test process failed')  # noqa: TRY301
				score_test('barlow')
				publish(matrix)
			except BaseException as error:
				# Let an already-running owned CPU evaluation retain its results.
				save_status(
					'failed',
					error_type=type(error).__name__,
					error=str(error),
					cpu_test_pid=cpu.pid,
				)
				raise
		save_status('complete', new_decoder_cells=42, factorial_cells=540, v1_cells=405)
		(operations() / 'complete.json').write_text(
			(operations() / 'status.json').read_text()
		)


def main() -> None:
	"""Plan, validate or execute only the fixed missing-control extension."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		'--phase',
		choices=(
			'plan',
			'dry-run',
			'run',
			'v1-test',
			'barlow-test',
			'summary',
			'review',
		),
		default='plan',
	)
	parser.add_argument('--gpu', default='0')
	args = parser.parse_args()
	matrix = validate()
	if args.phase == 'plan':
		print(json.dumps(plan_decoders(matrix), indent=2))
	elif args.phase == 'dry-run':
		jobs = plan_decoders(matrix)
		for survey in ('f3', 'parihaka', 'volve'):
			survey_jobs = [j for j in jobs if j['survey'] == survey]
			job = next((j for j in survey_jobs if not j['complete']), survey_jobs[0])
			_run([*job['command'], '--dry-run'], operations() / 'preflight', survey)
		inspect_test_plan(prepare_test('v1'))
	elif args.phase == 'run':
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
		run(args.gpu)
	elif args.phase == 'summary':
		publish(matrix)
	elif args.phase == 'review':
		publish_review(
			study_root() / 'summary', Path('reports/hmm_missing_controls_completion_v1')
		)
	else:
		score_test(args.phase.removesuffix('-test'))


if __name__ == '__main__':
	main()
