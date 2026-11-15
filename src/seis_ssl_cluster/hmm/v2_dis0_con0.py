"""Fixed K6810 experiment with both distillation and consistency set to zero."""

# ruff: noqa: PLC0415, S603
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.hmm.distillation_sensitivity import (
	_cell_path,
	execute_phase,
	read_rows,
	write_rows,
)
from seis_ssl_cluster.hmm.multi_source_survey import _metrics

STUDY = 'v2_dis0_con0_v1'
LEGACY_STUDY = 'vis_dis0_con0_v1'
MATRIX = Path('experiments/hmm_v2/v2_dis0_con0_v1/matrix.yaml')
SURVEYS = ('f3', 'parihaka', 'volve')
SOURCES = ('mae', 'local_bt', 'random')
LAYOUTS = tuple(f'layout_{i:03d}' for i in range(5))
SIZES = ('small', 'medium', 'large')
SUMMARY_FILES = frozenset(
	{
		'cells.csv',
		'comparison.csv',
		'f3_validation.csv',
		'training_diagnostics.csv',
		'summary.json',
		'summary.md',
	}
)
REVIEW_FILES = frozenset(
	{'README.md', 'cells.csv', 'comparison.csv', 'provenance.json'}
)


def _root() -> Path:
	return Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])


def _operations() -> Path:
	return _root() / 'operations/hmm_v2' / STUDY


def _study_root() -> Path:
	return _root() / 'studies/hmm_v2' / STUDY


def _smoke_output(arm: dict[str, Any], training_output: str) -> str:
	output = training_output.replace('/full_25ep', '/smoke_1step')
	if 'legacy_smoke_output_root' not in arm:
		return output
	legacy_output = output.replace(STUDY, LEGACY_STUDY)
	if arm['survey'] != 'volve' or arm['legacy_smoke_output_root'] != legacy_output:
		raise ValueError('legacy smoke must retain its original Volve output')
	return legacy_output


def validate_inherited(arm: dict[str, Any]) -> None:
	"""Allow only consistency zero and the new output/model identities."""
	for kind in ('training', 'embeddings', 'downstream'):
		candidate = yaml.safe_load(Path(arm['configs'][kind]['path']).read_text())
		original = yaml.safe_load(
			Path(arm['reference_configs'][kind]['path']).read_text()
		)
		normalized = json.loads(
			json.dumps(candidate)
			.replace(STUDY, arm['reference_study'])
			.replace(arm['candidate_id'], arm['reference_candidate_id'])
		)
		if kind == 'training':
			if (
				candidate['head']['ks'] != [6, 8, 10]
				or candidate['loss']['distillation_weight'] != 0.0
				or candidate['loss']['consistency_weight'] != 0.0
				or candidate['identity']['scientific_identity']['variant'] != 'nocons'
				or candidate['train'].get('feature_diagnostics_samples') != 16
			):
				raise ValueError(
					'expected K6810/distillation0/consistency0 with fixed diagnostics'
				)
			normalized['loss']['consistency_weight'] = original['loss'][
				'consistency_weight'
			]
			normalized['identity']['scientific_identity'] = original['identity'][
				'scientific_identity'
			]
			# Only the variant and its coefficient may change in this identity.
			expected = deepcopy(original['identity']['scientific_identity'])
			expected['variant'] = 'nocons'
			if 'consistency_weight' in expected:
				expected['consistency_weight'] = 0.0
			if candidate['identity']['scientific_identity'] != expected:
				raise ValueError('scientific source identity changed')
		if kind == 'downstream' and 'arm' in original:
			if candidate['arm']['multi_head_training_config'] != (
				'${SEIS_SSL_CLUSTER_WORKSPACE}/' + arm['configs']['training']['path']
			):
				raise ValueError('downstream must reference its own training recipe')
			normalized['arm']['multi_head_training_config'] = original['arm'][
				'multi_head_training_config'
			]
		if normalized != original:
			raise ValueError(
				f'non-consistency recipe change: {arm["survey"]}/{arm["source"]}/{kind}'
			)
	full = yaml.safe_load(Path(arm['configs']['training']['path']).read_text())
	smoke = yaml.safe_load(Path(arm['configs']['smoke']['path']).read_text())
	expected = deepcopy(full)
	expected['paths']['output_root'] = _smoke_output(arm, full['paths']['output_root'])
	expected['train'].update(epochs=1, max_steps=1)
	if smoke != expected:
		raise ValueError('smoke differs from full recipe beyond duration/output')


def validate_matrix(path: Path = MATRIX) -> dict[str, Any]:
	"""Require the complete fixed nine-arm experiment and all SHA-bound recipes."""
	matrix = yaml.safe_load(path.read_text())
	if (
		matrix['study'] != STUDY
		or matrix['display_name'] != 'v2_dis0_con0'
		or matrix['head_ks'] != [6, 8, 10]
		or matrix['distillation_weight'] != 0.0
		or matrix['consistency_weight'] != 0.0
		or matrix['seed'] != 42
		or matrix['epochs'] != 25
		or matrix['layouts'] != list(LAYOUTS)
		or matrix['sizes'] != list(SIZES)
		or matrix['new_continuations'] != 9
		or matrix['new_downstream_cells'] != 135
		or matrix['test_namespace'] != 'lithology_common_test_v2_dis0_con0_v1'
		or {(a['survey'], a['source']) for a in matrix['arms']}
		!= {(s, f) for s in SURVEYS for f in SOURCES}
		or len(matrix['arms']) != 9
	):
		raise ValueError('fixed v2_dis0_con0 experiment scope changed')
	for arm in matrix['arms']:
		for record in [
			*arm['configs'].values(),
			*arm['reference_configs'].values(),
			arm['historical_training'],
		]:
			path = Path(record['path'])
			if (
				'reports' in path.resolve().parts
				or file_sha256(path) != record['sha256']
			):
				raise ValueError(f'configuration hash drift or report input: {path}')
		validate_inherited(arm)
	return matrix


def validate_smoke(arm: dict[str, Any]) -> dict[str, Any]:
	"""Check actual zero constraint contributions and historical initialization."""
	import torch

	config = load_config(arm['configs']['smoke']['path'])
	root = Path(config['paths']['output_root'])
	payload = torch.load(root / 'latest.pt', map_location='cpu', weights_only=False)
	metrics = payload['metrics']
	if (
		payload['global_step'] != 1
		or metrics['loss_distillation_weighted'] != 0.0
		or metrics['loss_consistency_weighted'] != 0.0
		or metrics['loss_prototype_weighted'] <= 0.0
	):
		raise ValueError(
			'smoke must learn prototypes with both constraints contributing zero'
		)
	metadata = json.loads((root / 'run_metadata.json').read_text())
	historical = load_config(arm['historical_training']['path'])
	previous = json.loads(
		(Path(historical['paths']['output_root']) / 'run_metadata.json').read_text()
	)
	if (
		metadata['control_identity']['initial_parameter_sha256']
		!= previous['control_identity']['initial_parameter_sha256']
	):
		raise ValueError('initial student/head parameters differ from historical v2')
	return {
		'survey': arm['survey'],
		'source': arm['source'],
		'initial_parameters_match_historical_v2': True,
		'weighted_distillation': 0.0,
		'weighted_consistency': 0.0,
		'weighted_prototype': metrics['loss_prototype_weighted'],
	}


def run_arm(arm: dict[str, Any], phase: str, *, dry_run: bool = False) -> None:
	"""Use existing runners and exact-arm resume without changing shared recipes."""
	reused_smoke = False
	if phase == 'smoke' and not dry_run:
		config = load_config(arm['configs']['smoke']['path'])
		if (Path(config['paths']['output_root']) / 'latest.pt').exists():
			reused_smoke = True
	mini = {
		'analysis_id': f'{STUDY}/{arm["source"]}',
		'new_continuations': 1,
		'retrospective_audits': False,
		'surveys': {arm['survey']: {'0.0': dict(arm, execution='new')}},
	}
	if not reused_smoke:
		execute_phase(
			mini,
			SimpleNamespace(
				phase=phase,
				survey=arm['survey'],
				weight=0.0,
				resume=True,
				dry_run=dry_run,
			),
		)
	if phase == 'smoke' and not dry_run:
		result = validate_smoke(arm)
		result['reused'] = reused_smoke
		out = _operations() / 'smoke_validation'
		out.mkdir(parents=True, exist_ok=True)
		(out / f'{arm["survey"]}_{arm["source"]}.json').write_text(
			json.dumps(result, indent=2) + '\n'
		)


def gpu_available(device: str, required_mib: int) -> bool:
	"""Admit work only with an idle GPU and enough space for the original batch."""
	output = subprocess.check_output(
		[
			shutil.which('nvidia-smi') or '/usr/bin/nvidia-smi',
			f'--id={device}',
			'--query-gpu=memory.free,utilization.gpu',
			'--format=csv,noheader,nounits',
		],
		text=True,
	)
	free, utilization = (int(v.strip()) for v in output.strip().split(','))
	return free >= required_mib and utilization <= 10


def run_phase(matrix: dict[str, Any], phase: str, devices: list[str]) -> None:  # noqa: C901
	"""Queue a phase without stopping other jobs or polling their experiment state."""
	jobs = list(matrix['arms'])
	if phase == 'smoke':
		pending = []
		for arm in jobs:
			config = load_config(arm['configs']['smoke']['path'])
			if (Path(config['paths']['output_root']) / 'latest.pt').is_file():
				run_arm(arm, 'smoke')
				print(f'Reused smoke: {arm["survey"]}/{arm["source"]}', flush=True)
			else:
				pending.append(arm)
		jobs = pending
	mutex, stopped = threading.Lock(), threading.Event()

	def worker(device: str) -> None:  # noqa: C901
		try:
			with (_operations() / f'gpu_{device}.lock').open('a') as lease:
				fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
				while not stopped.is_set():
					with mutex:
						if not jobs:
							return
					chosen = None
					for threshold in (38 * 1024, 12 * 1024):
						if not gpu_available(device, threshold):
							continue
						time.sleep(5)
						if not gpu_available(device, threshold):
							continue
						with mutex:
							for index, arm in enumerate(jobs):
								required = (
									12 * 1024
									if phase in ('medium', 'remaining')
									or arm['survey'] == 'volve'
									else 38 * 1024
								)
								if required <= threshold:
									chosen = jobs.pop(index)
									break
						if chosen is not None:
							break
					if chosen is None:
						stopped.wait(30)
						continue
					command = [
						sys.executable,
						'-m',
						'seis_ssl_cluster.hmm.v2_dis0_con0',
						'--phase',
						phase,
						'--survey',
						chosen['survey'],
						'--source',
						chosen['source'],
					]
					environment = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
					print(
						f'{phase}: {chosen["survey"]}/{chosen["source"]} '
						f'on GPU {device}',
						flush=True,
					)
					subprocess.run(command, env=environment, check=True)
		except BaseException:
			stopped.set()
			raise

	with ThreadPoolExecutor(max_workers=len(devices)) as executor:
		futures = [executor.submit(worker, d) for d in devices]
		for future in futures:
			future.result()


def prepare_test(matrix: dict[str, Any]) -> Path:
	"""Bind all 45 selected F3 decoders to the original common-test mask."""
	output = _study_root() / 'test_definition'
	if (output / 'evaluate.yaml').exists():
		return output / 'evaluate.yaml'
	rows = []
	for arm in matrix['arms']:
		if arm['survey'] != 'f3':
			continue
		raw = load_config(arm['configs']['downstream']['path'])
		cp = Path(raw['candidate']['checkpoint'])
		embedding = Path(raw['candidate']['embeddings_dir'])
		meta = next(embedding.glob('*.embedding_metadata.json'))
		rows.append(
			{
				'audit_id': f'v2_dis0_con0__{arm["source"]}',
				'version': 'v2',
				'source': arm['source'],
				'head_ks': '6|8|10',
				'distillation_weight': '0.0',
				'consistency_weight': '0',
				'historical_condition_id': arm['candidate_id'],
				'layouts': '|'.join(LAYOUTS),
				'sizes': '|'.join(SIZES),
				'existing_cells': 15,
				'checkpoint_sha256': file_sha256(cp),
				'embedding_metadata_sha256': file_sha256(meta),
				'checkpoint_artifact_relative': str(cp.relative_to(_root())),
				'embeddings_artifact_relative': str(embedding.relative_to(_root())),
				'historical_runs_artifact_relative': str(
					(
						Path(raw['outputs']['runs_root'])
						/ f'model={arm["candidate_id"]}'
					).relative_to(_root())
				),
			}
		)
	output.mkdir(parents=True, exist_ok=False)
	inventory = output / 'condition_inventory.csv'
	write_rows(inventory, rows)
	config = load_config(
		'experiments/f3/facies_benchmark_v2/136_lithology_common_test_v3_v1/evaluation.yaml'
	)
	config.update(
		experiment_id=matrix['test_namespace'],
		expected_conditions=3,
		expected_cells=45,
		output_root=str(
			_root()
			/ 'surveys/f3/facies_benchmark_v2/evaluation'
			/ matrix['test_namespace']
		),
	)
	config['inputs']['inventory'] = {
		'path': str(inventory),
		'sha256': file_sha256(inventory),
	}
	path = output / 'evaluate.yaml'
	path.write_text(yaml.safe_dump(config, sort_keys=False))
	return path


def collect_summary(matrix: dict[str, Any]) -> dict[str, Any]:
	"""Keep F3 common-test values separate from checkpoint-selection validation."""
	test = read_rows(
		_root()
		/ 'surveys/f3/facies_benchmark_v2/evaluation'
		/ matrix['test_namespace']
		/ 'summary/per_layout.csv'
	)
	test_index = {(r['source'], r['layout_id'], r['data_size']): r for r in test}
	if len(test_index) != 45:
		raise ValueError('all 45 F3 common-test cells are required')
	cells, validation, diagnostics, evidence = [], [], [], []
	for arm in matrix['arms']:
		survey, source = arm['survey'], arm['source']
		cfg = load_config(arm['configs']['training']['path'])
		epochs = read_rows(
			Path(cfg['paths']['output_root']) / 'multi_head_epoch_metrics.csv'
		)
		if [int(r['epoch']) for r in epochs] != list(range(1, 26)) or any(
			float(r[k]) != 0.0
			for r in epochs
			for k in ('loss_distillation_weighted', 'loss_consistency_weighted')
		):
			raise ValueError(
				'expected 25 epochs with both constraint contributions zero'
			)
		diagnostics.extend(dict(survey=survey, source=source, **r) for r in epochs)
		for layout in LAYOUTS:
			for size in SIZES:
				path = _cell_path(survey, arm, layout, size)
				primary, secondary, digest = _metrics(path, survey)
				if survey == 'f3':
					validation.append(
						{
							'source': source,
							'layout_id': layout,
							'data_size': size,
							'validation_mean_iou': primary,
							'validation_macro_f1': secondary,
						}
					)
					row = test_index[(source, layout, size)]
					primary, secondary = float(row['mean_iou']), float(row['macro_f1'])
				cells.append(
					{
						'survey': survey,
						'source': source,
						'layout_id': layout,
						'data_size': size,
						'primary': primary,
						'secondary': secondary,
						'evaluation_split': 'common_test' if survey == 'f3' else 'test',
					}
				)
				evidence.append(
					{
						'survey': survey,
						'source': source,
						'layout_id': layout,
						'data_size': size,
						'metrics_path': str(path),
						'metrics_sha256': digest,
					}
				)
	rows = aggregate_cells(cells)
	return {
		'study': STUDY,
		'display_name': 'v2_dis0_con0',
		'status': 'complete',
		'head_ks': [6, 8, 10],
		'distillation_weight': 0.0,
		'consistency_weight': 0.0,
		'cells': cells,
		'rows': rows,
		'validation': validation,
		'diagnostics': diagnostics,
		'evidence': evidence,
		'matrix_sha256': file_sha256(MATRIX),
	}


def aggregate_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Require the full grid and report each survey/source/budget independently."""
	expected = {
		(s, f, l, z) for s in SURVEYS for f in SOURCES for l in LAYOUTS for z in SIZES
	}
	if (
		len(cells) != 135
		or {(r['survey'], r['source'], r['layout_id'], r['data_size']) for r in cells}
		!= expected
	):
		raise ValueError('expected 135 unique complete cells')
	if any(
		not math.isfinite(float(r[k])) for r in cells for k in ('primary', 'secondary')
	):
		raise ValueError('summary metrics must be finite')
	rows = []
	for survey in SURVEYS:
		for source in SOURCES:
			for size in SIZES:
				group = [
					r
					for r in cells
					if (r['survey'], r['source'], r['data_size'])
					== (survey, source, size)
				]
				row = {
					'survey': survey,
					'source': source,
					'data_size': size,
					'layout_count': 5,
				}
				for key in ('primary', 'secondary'):
					values = [float(r[key]) for r in group]
					row[key + '_mean'] = statistics.mean(values)
					row[key + '_sample_sd'] = statistics.stdev(values)
				rows.append(row)
	return rows


def publish_summary(root: Path, payload: dict[str, Any]) -> None:
	"""Publish exactly six artifact-side files, atomically and without overwrite."""
	if root.exists():
		raise FileExistsError(root)
	if payload['rows'] != aggregate_cells(payload['cells']):
		raise ValueError('summary rows disagree with cell metrics')
	root.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(prefix='.summary-', dir=root.parent) as tmp:
		stage = Path(tmp)
		for filename, key in (
			('cells.csv', 'cells'),
			('comparison.csv', 'rows'),
			('f3_validation.csv', 'validation'),
			('training_diagnostics.csv', 'diagnostics'),
		):
			write_rows(stage / filename, payload[key])
		(stage / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
		lines = [
			'# v2_dis0_con0',
			'',
			'K=[6,8,10]; distillation=0; consistency=0.',
			'',
			(
				'Five-layout means and sample SD; continuation seed 42; '
				'one encoder run per source.'
			),
			(
				'F3 common-test Mean IoU; Parihaka test Channel IoU; '
				'Volve test MAE [sample].'
			),
			'',
			'| Survey | Source | Budget | Primary mean | Sample SD |',
			'|---|---|---|---:|---:|',
		]
		lines += [
			f'| {r["survey"]} | {r["source"]} | {r["data_size"]} | '
			f'{r["primary_mean"]:.6f} | {r["primary_sample_sd"]:.6f} |'
			for r in payload['rows']
		]
		(stage / 'summary.md').write_text('\n'.join(lines) + '\n')
		if {p.name for p in stage.iterdir()} != SUMMARY_FILES:
			raise ValueError('unexpected summary publication files')
		stage.rename(root)


def publish_review(summary: Path, root: Path) -> None:
	"""Curate exactly four small review files; reports never feed the experiment."""
	if root.exists():
		raise FileExistsError(root)
	if {p.name for p in summary.iterdir()} != SUMMARY_FILES:
		raise ValueError('incomplete artifact summary')
	root.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(prefix='.review-', dir=root.parent) as tmp:
		stage = Path(tmp)
		for filename in ('cells.csv', 'comparison.csv'):
			(stage / filename).write_bytes((summary / filename).read_bytes())
		(stage / 'README.md').write_text(
			(summary / 'summary.md').read_text()
			+ '\nAll nine 25-epoch continuations and 135 decoder cells are complete.\n'
			'No statistical-significance claim; '
			'layouts are not encoder-seed repetitions.\n'
			'Raw predictions, embeddings and checkpoints remain in artifacts.\n'
		)
		(stage / 'provenance.json').write_text(
			json.dumps(
				{
					'study': STUDY,
					'artifact_summary': str(summary.relative_to(_root())),
					'summary_sha256': file_sha256(summary / 'summary.json'),
					'published_files': sorted(REVIEW_FILES),
				},
				indent=2,
			)
			+ '\n'
		)
		if {p.name for p in stage.iterdir()} != REVIEW_FILES:
			raise ValueError('unexpected review publication files')
		stage.rename(root)


def run_study(matrix: dict[str, Any], devices: list[str]) -> None:
	"""Run the authorized full experiment, leaving existing jobs untouched."""
	from seis_ssl_cluster.f3.lithology.common_test import inspect_test_plan, run_test

	_operations().mkdir(parents=True, exist_ok=True)
	with (_operations() / 'study.lock').open('a') as lock:
		fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		for phase in ('smoke', 'full', 'embeddings', 'medium', 'remaining'):
			print(f'Queued phase: {phase}; {len(matrix["arms"])} arms', flush=True)
			run_phase(matrix, phase, devices)
		path = prepare_test(matrix)
		run_test(inspect_test_plan(path), resume=True)
		summary = _study_root() / 'summary'
		if not summary.exists():
			publish_summary(summary, collect_summary(matrix))
		review = Path(os.environ['SEIS_SSL_CLUSTER_WORKSPACE']) / 'reports' / STUDY
		if not review.exists():
			publish_review(summary, review)
		(_operations() / 'complete.json').write_text(
			json.dumps(
				{
					'status': 'complete',
					'utc': datetime.now(timezone.utc).isoformat(),
					'new_encoder_runs': 9,
					'new_decoder_cells': 135,
					'f3_common_test_cells': 45,
					'matrix_sha256': file_sha256(MATRIX),
				},
				indent=2,
			)
			+ '\n'
		)


def main() -> None:
	"""Validate without GPUs, dry-run all recipes, or execute the fixed study."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		'--phase',
		choices=(
			'validate',
			'dry-run',
			'run',
			'smoke',
			'full',
			'embeddings',
			'medium',
			'remaining',
		),
		required=True,
	)
	parser.add_argument('--survey', choices=SURVEYS)
	parser.add_argument('--source', choices=SOURCES)
	parser.add_argument('--gpus', nargs='+', default=['0', '1'])
	args = parser.parse_args()
	matrix = validate_matrix()
	if args.phase == 'validate':
		print(
			'v2_dis0_con0: nine K6810/dist0/cons0 arms; '
			'36 bound YAMLs; 135 decoder cells'
		)
	elif args.phase == 'dry-run':
		for arm in matrix['arms']:
			for phase in ('smoke', 'full'):
				run_arm(arm, phase, dry_run=True)
	elif args.phase == 'run':
		if len(set(args.gpus)) != len(args.gpus) or any(
			not d.isdigit() for d in args.gpus
		):
			raise ValueError('GPU IDs must be distinct integers')
		run_study(matrix, args.gpus)
	else:
		if args.survey is None or args.source is None:
			parser.error('an individual phase requires --survey and --source')
		arm = next(
			a
			for a in matrix['arms']
			if (a['survey'], a['source']) == (args.survey, args.source)
		)
		run_arm(arm, args.phase)


if __name__ == '__main__':
	main()
