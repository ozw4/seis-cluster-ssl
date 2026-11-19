"""Execute and audit the matched unordered k-means study."""

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
from pathlib import Path
from typing import Any

from seis_ssl_cluster.clustering.kmeans_definition import (
	LAYOUTS,
	SIZES,
	SOURCES,
	STUDY,
	SURVEYS,
	read_yaml,
	validate_study,
)
from seis_ssl_cluster.clustering.kmeans_targets import (
	build_kmeans_targets,
	file_reference,
	load_unordered_target_manifest,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.hmm.multi_source_cell import audit_completed_cell
from seis_ssl_cluster.hmm.multi_source_survey import (
	_check_directories,
	_metrics,
	_paired_identity,
	_paths,
)
from seis_ssl_cluster.training.strat_hmm_source_audit import audit_multi_head_source

MODULE = 'seis_ssl_cluster.clustering.kmeans_study'
STAGES = ('targets', 'smoke', 'full', 'audit', 'embeddings', 'downstream', 'summary')
PUBLISHED_FILES = frozenset({'comparison.csv', 'summary.json', 'summary.md'})


def validate_cells(cells: list[dict[str, Any]]) -> None:
	"""Require every one of the 135 distinct matched decoder cells."""
	expected = {
		(s, f, l, z) for s in SURVEYS for f in SOURCES for l in LAYOUTS for z in SIZES
	}
	actual = [
		(c['survey'], c['source_family'], c['layout_id'], c['data_size']) for c in cells
	]
	if len(actual) != len(expected) or set(actual) != expected:
		raise ValueError('comparison requires exactly 135 unique paired cells')


def comparison_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Summarize five paired layouts separately for every survey/family/budget."""
	validate_cells(cells)
	rows = []
	for survey in SURVEYS:
		for family in SOURCES:
			for size in SIZES:
				selected = [
					c
					for c in cells
					if (c['survey'], c['source_family'], c['data_size'])
					== (survey, family, size)
				]
				row = {
					'survey': survey,
					'source_family': family,
					'data_size': size,
					'layout_count': 5,
					'evaluation_split': 'validation' if survey == 'f3' else 'test',
				}
				for metric in ('primary', 'secondary'):
					candidate = [c[metric] for c in selected]
					control = [c['control_' + metric] for c in selected]
					differences = [
						a - b for a, b in zip(candidate, control, strict=True)
					]
					for name, values in (
						('kmeans', candidate),
						('hmm_v2', control),
						('difference', differences),
					):
						row[f'{metric}_{name}_mean'] = statistics.mean(values)
						row[f'{metric}_{name}_sample_sd'] = statistics.stdev(values)
				rows.append(row)
	return rows


def collect_summary(matrix_path: Path, workspace: Path) -> dict[str, Any]:
	"""Audit both encoders and all paired decoder outputs before publishing."""
	matrix = validate_study(matrix_path, workspace)
	cells, evidence = [], []
	for survey in SURVEYS:
		for family in SOURCES:
			arm = matrix['surveys'][survey][family]
			if matrix['study'] == 'kmeans_k6810_dis0_con0_v1':
				from seis_ssl_cluster.clustering.kmeans_zero import (
					validate_full_history,
				)

				validate_full_history(arm, workspace)
			lineage = [
				audit_multi_head_source(workspace / path)
				for path in (
					arm['configs']['training'],
					arm['references']['training']['path'],
				)
			]
			candidate, control = (
				load_config(workspace / path)
				for path in (
					arm['configs']['downstream'],
					arm['references']['downstream']['path'],
				)
			)
			ids = (arm['candidate_id'], arm['control_id'])
			paths = [
				_paths(raw, survey, cid)
				for raw, cid in zip((candidate, control), ids, strict=True)
			]
			for (root, checkpoint), cid, source in zip(
				paths, ids, lineage, strict=True
			):
				_check_directories(root, cid)
				if source['checkpoint'] != file_reference(checkpoint):
					raise ValueError('summary encoder lineage mismatch')
			for layout in LAYOUTS:
				for size in SIZES:
					audits = [
						audit_completed_cell(survey, raw, cid, layout, size)
						for raw, cid in zip((candidate, control), ids, strict=True)
					]
					_paired_identity(survey, *audits)
					metrics = []
					for (root, _), cid in zip(paths, ids, strict=True):
						name = (
							'evaluation/metrics.json'
							if survey == 'f3'
							else 'metrics.json'
						)
						path = (
							root
							/ f'model={cid}'
							/ f'layout={layout}'
							/ f'size={size}'
							/ name
						)
						primary, secondary, digest = _metrics(path, survey)
						metrics.append(
							(primary, secondary, {'path': str(path), 'sha256': digest})
						)
					cells.append(
						{
							'survey': survey,
							'source_family': family,
							'candidate_id': ids[0],
							'control_id': ids[1],
							'layout_id': layout,
							'data_size': size,
							'primary': metrics[0][0],
							'secondary': metrics[0][1],
							'control_primary': metrics[1][0],
							'control_secondary': metrics[1][1],
							'candidate_metrics': metrics[0][2],
							'control_metrics': metrics[1][2],
						}
					)
					evidence.append(
						{
							'survey': survey,
							'source_family': family,
							'layout_id': layout,
							'data_size': size,
							'sources': lineage,
							'paired_cells': audits,
						}
					)
	rows = comparison_rows(cells)
	payload = {
		'schema_version': 1,
		'study': matrix['study'],
		'status': 'complete',
		'matrix': file_reference(matrix_path),
		'cells': cells,
		'rows': rows,
		'evidence': evidence,
		'metrics': {
			'f3': {
				'primary': 'mean_iou',
				'secondary': 'macro_f1',
				'primary_direction': 'higher',
			},
			'parihaka': {
				'primary': 'channel_iou',
				'secondary': 'channel_f1',
				'primary_direction': 'higher',
			},
			'volve': {
				'primary': 'macro_mae_samples',
				'secondary': 'macro_within_2_samples',
				'primary_direction': 'lower',
			},
		},
		'interpretation': (
			'Joint ablation of HMM target generation and ordered '
			'cross-head consistency; raw differences are kmeans minus HMM v2.'
		),
	}
	if matrix['study'] == 'kmeans_k6810_dis0_con0_v1':
		from seis_ssl_cluster.clustering.kmeans_zero import finalize_payload

		payload = finalize_payload(payload)
	return payload


def publish_summary(output: Path, payload: dict[str, Any]) -> None:
	"""Publish exactly the owned review files, refusing existing summary directories."""
	if 'reports' in output.resolve().parts:
		raise ValueError('study writes artifacts only')
	validate_cells(payload['cells'])
	output.mkdir(parents=True, exist_ok=False)
	(output / 'summary.json').write_text(
		json.dumps(payload, indent=2, allow_nan=False) + '\n'
	)
	with (output / 'comparison.csv').open('w', newline='') as stream:
		writer = csv.DictWriter(stream, fieldnames=list(payload['rows'][0]))
		writer.writeheader()
		writer.writerows(payload['rows'])
	if payload.get('study') == 'kmeans_k6810_dis0_con0_v1':
		from seis_ssl_cluster.hmm.distillation_sensitivity import write_rows

		write_rows(output / 'cells.csv', payload['cells'])
	lines = [
		'# Unordered k-means versus HMM v2: ' + payload.get('study', STUDY),
		'',
		payload['interpretation'],
		'',
		'| Survey | Source | Size | k-means | HMM v2 | Difference ± sample SD |',
		'| --- | --- | --- | ---: | ---: | ---: |',
	]
	lines.extend(
		f'| {row["survey"]} | {row["source_family"]} | {row["data_size"]} | '
		f'{row["primary_kmeans_mean"]:.6f} | {row["primary_hmm_v2_mean"]:.6f} | '
		f'{row["primary_difference_mean"]:.6f} ± '
		f'{row["primary_difference_sample_sd"]:.6f} |'
		for row in payload['rows']
	)
	(output / 'summary.md').write_text('\n'.join(lines) + '\n')


def command_plan(  # noqa: C901, PLR0912
	matrix_path: Path, workspace: Path, args: argparse.Namespace
) -> list[tuple[str, list[str]]]:
	"""Enumerate stages without reading execution artifacts or creating outputs."""
	matrix = validate_study(matrix_path, workspace)
	first, last = STAGES.index(args.first), STAGES.index(args.last)
	if first > last:
		raise ValueError('stage range is reversed')
	if (args.layout or args.size) and (args.first, args.last) != (
		'downstream',
		'downstream',
	):
		raise ValueError('cell selection requires downstream only')
	if args.resume and (
		not args.candidate
		or first != last
		or args.first not in {'full', 'downstream'}
		or (args.first == 'downstream' and not (args.layout and args.size))
	):
		raise ValueError(
			'resume requires one candidate and one training stage or decoder cell'
		)
	if args.candidate and not args.survey:
		raise ValueError('candidate selection requires survey')
	if (args.survey or args.candidate) and last == STAGES.index('summary'):
		raise ValueError('summary requires the complete three-survey matrix')
	commands = []
	for stage in STAGES[first : last + 1]:
		if stage == 'summary':
			commands.append(
				(
					stage,
					[
						sys.executable,
						'-m',
						MODULE,
						'--matrix',
						str(matrix_path),
						'--summarize',
					],
				)
			)
			continue
		for survey in (args.survey,) if args.survey else SURVEYS:
			arms = matrix['surveys'][survey]
			selected = [
				a
				for a in arms.values()
				if not args.candidate or a['candidate_id'] == args.candidate
			]
			if not selected:
				raise ValueError('unknown candidate')
			for arm in selected:
				cid = arm['candidate_id']
				configs = {
					name: workspace / path for name, path in arm['configs'].items()
				}
				if stage == 'targets':
					cmd = [
						sys.executable,
						'-m',
						MODULE,
						'--target-config',
						str(configs['targets']),
					]
				elif stage in {'smoke', 'full'}:
					path = configs['training' if stage == 'full' else 'smoke']
					cmd = [
						sys.executable,
						'proc/seis_ssl_cluster/train_strat_hmm_pretext.py',
						'--config',
						str(path),
					]
					if args.resume:
						cmd += [
							'--resume',
							os.path.expandvars(read_yaml(path)['paths']['output_root'])
							+ '/latest.pt',
						]
				elif stage == 'audit':
					cmd = [
						sys.executable,
						'-m',
						MODULE,
						'--audit-config',
						str(configs['training']),
					]
				elif stage == 'embeddings':
					cmd = [
						sys.executable,
						'proc/seis_ssl_cluster/extract_embeddings.py',
						'--config',
						str(configs['embeddings']),
						'--device',
						'cuda',
					]
				else:
					for layout in (args.layout,) if args.layout else LAYOUTS:
						for size in (args.size,) if args.size else SIZES:
							cmd = [
								sys.executable,
								'-m',
								'seis_ssl_cluster.hmm.multi_source_cell',
								'--survey',
								survey,
								'--config',
								str(configs['downstream']),
								'--candidate',
								cid,
								'--layout',
								layout,
								'--size',
								size,
							]
							if args.resume:
								root = os.path.expandvars(
									read_yaml(configs['downstream'])['outputs'][
										'runs_root'
									]
								)
								suffix = (
									'decoder/latest.pt'
									if survey == 'f3'
									else 'latest.pt'
								)
								cmd += [
									'--resume',
									f'{root}/model={cid}/layout={layout}/size={size}/{suffix}',
								]
							commands.append((stage, cmd))
					continue
				commands.append((stage, cmd))
	return commands


def run_command(stage: str, command: list[str], *, dry_run: bool, log: Path) -> None:
	"""Bind live target hashes and retain stage logs for explicit recovery."""
	environment = os.environ.copy()
	if stage in {'smoke', 'full'}:
		path = Path(command[command.index('--config') + 1])
		raw = read_yaml(path)
		manifest = Path(os.path.expandvars(raw['pseudo_targets']['manifest']))
		load_unordered_target_manifest(manifest)
		environment['SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256'] = (
			file_reference(manifest)['sha256']
		)
		output = Path(os.path.expandvars(raw['paths']['output_root']))
		if (
			stage == 'full'
			and (output / 'latest.pt').exists()
			and '--resume' not in command
		):
			audit_multi_head_source(path)
			print('reused audited completed training:', output)
			return
	if stage in {'embeddings', 'downstream'}:
		config_path = Path(command[command.index('--config') + 1])
		experiment = config_path.parent.parent
		cid = config_path.stem
		audit_multi_head_source(
			experiment / '30_pretraining' / cid / '02_full_25ep.yaml'
		)
		if stage == 'embeddings':
			output = Path(load_config(config_path)['embeddings']['output_dir'])
			if output.exists() and not dry_run:
				raise FileExistsError(
					f'embeddings already exist; start from downstream: {output}'
				)
	if dry_run:
		command = [*command, '--dry-run']
	with log.open('x') as stream:
		stream.write(shlex.join(command) + '\n')
		stream.flush()
		subprocess.run(
			command,
			check=True,
			env=environment,
			stdout=stream,
			stderr=subprocess.STDOUT,
		)


def build_parser() -> argparse.ArgumentParser:
	"""Build the study CLI without inspecting data or execution state."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		'--matrix',
		type=Path,
		default=Path('experiments/kmeans') / STUDY / 'matrix.yaml',
	)
	parser.add_argument('--target-config', type=Path)
	parser.add_argument('--audit-config', type=Path)
	parser.add_argument('--summarize', action='store_true')
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument('--plan', action='store_true')
	mode.add_argument('--dry-run', action='store_true')
	mode.add_argument('--execute', action='store_true')
	parser.add_argument('--survey', choices=SURVEYS)
	parser.add_argument('--candidate')
	parser.add_argument('--from', dest='first', choices=STAGES, default=STAGES[0])
	parser.add_argument('--to', dest='last', choices=STAGES, default=STAGES[-1])
	parser.add_argument('--layout', choices=LAYOUTS)
	parser.add_argument('--size', choices=SIZES)
	parser.add_argument('--resume', action='store_true')
	return parser


def main() -> None:  # noqa: C901, PLR0912, PLR0915
	"""Plan, validate, or execute the study and its individual stages."""
	args = build_parser().parse_args()
	workspace = Path(__file__).resolve().parents[3]
	os.environ.setdefault('SEIS_SSL_CLUSTER_WORKSPACE', str(workspace))
	if args.target_config:
		config = load_config(args.target_config)
		if args.dry_run:
			from seis_ssl_cluster.clustering.features import (
				discover_embedding_inputs,
			)
			from seis_ssl_cluster.clustering.kmeans import (
				clustering_settings_from_config,
			)

			clustering_settings_from_config(config)
			discover_embedding_inputs(config['embeddings']['input_dir'])
			print(
				json.dumps(
					{'status': 'validated', 'output_root': config['output_root']}
				)
			)
		else:
			print(build_kmeans_targets(config, args.target_config))
		return
	if args.audit_config:
		if args.dry_run:
			print('planned source audit:', args.audit_config)
		else:
			print(json.dumps(audit_multi_head_source(args.audit_config), indent=2))
		return
	if args.summarize:
		matrix = validate_study(args.matrix, workspace)
		if matrix['study'] == 'kmeans_k6810_dis0_con0_v1':
			from seis_ssl_cluster.clustering.kmeans_zero import prepare_test

			if not args.dry_run:
				from seis_ssl_cluster.f3.lithology.common_test import (
					inspect_test_plan,
					run_test,
				)

				run_test(inspect_test_plan(prepare_test(matrix)), resume=True)
		payload = collect_summary(args.matrix, workspace)
		if not args.dry_run:
			output = (
				Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
				/ 'studies'
				/ 'kmeans'
				/ matrix['study']
				/ 'summary'
			)
			publish_summary(output, payload)
			if matrix['study'] == 'kmeans_k6810_dis0_con0_v1':
				from seis_ssl_cluster.clustering.kmeans_zero import publish_review

				publish_review(output, workspace / 'reports' / matrix['study'])
				from seis_ssl_cluster.clustering.kmeans_zero import refresh_catalog

				refresh_catalog(workspace)
		return
	commands = command_plan(args.matrix, workspace, args)
	if args.plan or not (args.execute or args.dry_run):
		for stage, command in commands:
			print(stage + ': ' + shlex.join(command))
		return
	if validate_study(args.matrix, workspace)['study'] == 'kmeans_k6810_dis0_con0_v1':
		from seis_ssl_cluster.clustering.kmeans_zero import validate_live_inputs

		validate_live_inputs(validate_study(args.matrix, workspace), workspace)
	if args.dry_run:
		# Live preflight is read-only, including its logs.
		for stage, command in commands:
			if stage in {'smoke', 'full'}:
				path = Path(command[command.index('--config') + 1])
				raw = read_yaml(path)
				env = os.environ.copy()
				env['SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256'] = (
					file_reference(
						Path(os.path.expandvars(raw['pseudo_targets']['manifest']))
					)['sha256']
				)
				subprocess.run([*command, '--dry-run'], check=True, env=env)
			else:
				subprocess.run([*command, '--dry-run'], check=True)
		return
	from datetime import datetime, timezone

	matrix = validate_study(args.matrix, workspace)
	root = (
		Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
		/ 'operations'
		/ 'kmeans'
		/ matrix['study']
	)
	root.mkdir(parents=True, exist_ok=True)
	with (root / 'study.lock').open('a') as lock:
		fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		logs = root / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
		logs.mkdir()
		for i, (stage, command) in enumerate(commands):
			print(f'{i + 1}/{len(commands)}: {stage}', flush=True)
			from seis_ssl_cluster.clustering.kmeans_zero import execute_recorded

			if matrix['study'] == 'kmeans_k6810_dis0_con0_v1':
				execute_recorded(
					matrix,
					workspace,
					root,
					i,
					len(commands),
					stage,
					command,
					logs / f'{i:03d}_{stage}.log',
				)
			else:
				run_command(
					stage, command, dry_run=False, log=logs / f'{i:03d}_{stage}.log'
				)
		if matrix['study'] == 'kmeans_k6810_dis0_con0_v1':
			(root / 'status.json').write_text(
				json.dumps(
					{
						'status': 'complete'
						if args.last == 'summary'
						else 'complete_requested_stages',
						'first_stage': args.first,
						'last_stage': args.last,
						'pid': os.getpid(),
					}
				)
				+ '\n'
			)


if __name__ == '__main__':
	main()
