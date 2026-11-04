"""Isolated MAE/no-HMM large decoder-seed repeat; canonical contracts stay fixed."""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import io
import json
import os
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.parihaka.channel_completion import (
	PUBLIC_METRIC_NAMES,
	inspect_completed_channel_job,
)
from seis_ssl_cluster.parihaka.channel_decoder import (
	ChannelDecoderConfig,
	ChannelDecoderPlan,
	channel_decoder_config_from_mapping,
	channel_decoder_run_identity,
	decoder_initial_state_sha256,
	inspect_channel_decoder_job,
	run_channel_decoder_job,
)

NAMESPACE = 'mae_no_hmm_large_decoder_seed42001_v1'
EXPERIMENT_ID = '47_' + NAMESPACE
LAYOUTS = tuple(f'layout_{index:03d}' for index in range(5))
PUBLISHED_FILES = ('comparison.csv', 'summary.json', 'summary.md')


@dataclass(frozen=True)
class SeedRepeat:
	"""Source-bound seed repeat with distinct execution and review destinations."""

	definition: dict[str, Any]
	baseline: ChannelDecoderConfig
	candidate: ChannelDecoderConfig
	layout_config: Path
	summary_root: Path
	operations_root: Path


def load_repeat(path: Path) -> SeedRepeat:  # noqa: C901
	"""Resolve only MAE/large/seed42001, without relaxing the canonical resolver."""
	raw = load_config(path)
	expected = {
		'schema_version': 1,
		'experiment_id': EXPERIMENT_ID,
		'model': 'mae',
		'data_size': 'large',
		'layouts': list(LAYOUTS),
		'baseline_decoder_seed': 42000,
		'decoder_seed': 42001,
	}
	path_keys = {
		'baseline_config',
		'layout_config',
		'runs_root',
		'summary_root',
		'operations_root',
	}
	hash_keys = {
		'baseline_config_sha256',
		'layout_config_sha256',
		'checkpoint_sha256',
		'baseline_metrics_sha256',
	}
	if set(raw) != set(expected) | path_keys | hash_keys or any(
		raw.get(key) != value or type(raw.get(key)) is not type(value)
		for key, value in expected.items()
	):
		raise ValueError('seed-repeat definition must retain the fixed MAE/large scope')
	for key in path_keys:
		if (
			not isinstance(raw[key], str)
			or not Path(raw[key]).is_absolute()
			or 'reports' in Path(raw[key]).resolve().parts
		):
			raise ValueError(f'{key} must be an absolute non-report path')
	for key in ('baseline_config', 'layout_config'):
		if file_sha256(Path(raw[key])) != raw[key + '_sha256']:
			raise ValueError(f'{key} definition hash drift')
	if set(raw['baseline_metrics_sha256']) != set(LAYOUTS):
		raise ValueError('baseline must bind all five layouts')
	artifact = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT']).resolve()
	benchmark = artifact / 'surveys/parihaka/facies_benchmark_v1/benchmark' / NAMESPACE
	for key, target in (
		('runs_root', benchmark / 'runs'),
		('summary_root', benchmark / 'summary'),
		('operations_root', artifact / 'operations/parihaka' / NAMESPACE),
	):
		if Path(raw[key]).resolve() != target:
			raise ValueError(f'{key} must use the isolated seed-repeat namespace')
	baseline = channel_decoder_config_from_mapping(
		load_config(Path(raw['baseline_config']))
	)
	baseline = replace(baseline, models={'mae': baseline.models['mae']})
	if baseline.survey_id != 'parihaka' or baseline.train.seed != 42000:
		raise ValueError('baseline must retain canonical Parihaka decoder settings')
	if (
		baseline.models['mae'].expected_checkpoint is None
		or file_sha256(baseline.models['mae'].expected_checkpoint)
		!= raw['checkpoint_sha256']
	):
		raise ValueError('frozen MAE checkpoint hash drift')
	candidate = replace(
		baseline,
		train=replace(baseline.train, seed=42001),
		runs_root=Path(raw['runs_root']),
	)
	return SeedRepeat(
		raw,
		baseline,
		candidate,
		Path(raw['layout_config']),
		Path(raw['summary_root']),
		Path(raw['operations_root']),
	)


def inspect_repeat_plan(repeat: SeedRepeat, layout: str) -> ChannelDecoderPlan:
	"""Preflight unchanged frozen inputs and the same layout's canonical result."""
	if layout not in LAYOUTS:
		raise ValueError('unknown seed-repeat layout')
	plan = inspect_channel_decoder_job(
		repeat.candidate,
		model='mae',
		layout_id=layout,
		data_size='large',
		layout_config=repeat.layout_config,
	)
	_frozen_baseline(repeat, plan)
	return plan


def _frozen_baseline(repeat: SeedRepeat, plan: ChannelDecoderPlan) -> dict[str, Any]:
	path = (
		repeat.baseline.runs_root
		/ f'model=mae/layout={plan.layout_id}/size=large/metrics.json'
	)
	if (
		file_sha256(path)
		!= repeat.definition['baseline_metrics_sha256'][plan.layout_id]
	):
		raise ValueError('canonical baseline metrics hash drift')
	metrics = json.loads(path.read_text(encoding='utf-8'))
	identity = copy.deepcopy(channel_decoder_run_identity(plan))
	if (
		identity['embedding']['checkpoint_sha256']
		!= repeat.definition['checkpoint_sha256']
	):
		raise ValueError('seed repeat changed the frozen encoder')
	if plan.geometry.models['mae'].metadata.get('stratigraphy_pretext') is not None:
		raise ValueError('seed repeat requires the non-HMM MAE source')
	identity['training']['seed'] = 42000
	identity['decoder_initial_state_sha256'] = decoder_initial_state_sha256(
		plan.config.decoder,
		plan.geometry.patch_size_xyz,
		42000,
	)
	if metrics['benchmark_identity'] != identity:
		raise ValueError(
			'seed repeat changed supervision, geometry, source, or decoder settings'
		)
	inspect_completed_channel_job(path.parent, metrics=metrics)
	return metrics


def audit_repeat_cell(repeat: SeedRepeat, plan: ChannelDecoderPlan) -> dict[str, Any]:
	"""Require a completed own-seed result, with unchanged baseline provenance."""
	_frozen_baseline(repeat, plan)
	path = plan.output_dir / 'metrics.json'
	metrics = json.loads(path.read_text(encoding='utf-8'))
	if metrics['benchmark_identity'] != channel_decoder_run_identity(plan):
		raise ValueError('completed result differs from the seed-repeat plan')
	inspect_completed_channel_job(plan.output_dir, metrics=metrics)
	return metrics


def inspect_summary(repeat: SeedRepeat) -> dict[str, Any]:
	"""Compare all five completed layouts; report every public held-out metric."""
	rows = []
	for layout in LAYOUTS:
		plan = inspect_repeat_plan(repeat, layout)
		original = _frozen_baseline(repeat, plan)
		candidate = audit_repeat_cell(repeat, plan)
		rows.append(
			{
				'layout_id': layout,
				'data_size': 'large',
				'baseline_metrics_sha256': repeat.definition['baseline_metrics_sha256'][
					layout
				],
				'repeat_metrics_sha256': file_sha256(plan.output_dir / 'metrics.json'),
				'baseline': original['test'],
				'repeat': candidate['test'],
			}
		)
	aggregates = {}
	for metric in PUBLIC_METRIC_NAMES:
		values = {}
		for arm in ('baseline', 'repeat'):
			ordered = sorted(float(row[arm][metric]) for row in rows)
			values[arm] = {
				'mean': statistics.fmean(ordered),
				'median': statistics.median(ordered),
				'trimmed_mean': statistics.fmean(ordered[1:-1]),
				'sample_sd': statistics.stdev(ordered),
			}
		values['paired_mean_delta'] = statistics.fmean(
			float(row['repeat'][metric]) - float(row['baseline'][metric])
			for row in rows
		)
		aggregates[metric] = values
	return {
		'schema_version': 1,
		'status': 'complete_requested_scope',
		'experiment_id': EXPERIMENT_ID,
		'model': 'mae',
		'data_size': 'large',
		'baseline_decoder_seed': 42000,
		'decoder_seed': 42001,
		'cell_count': 5,
		'checkpoint_sha256': repeat.definition['checkpoint_sha256'],
		'rows': rows,
		'aggregates': aggregates,
		'interpretation': (
			'Two decoder seeds, same frozen encoder; not two pretraining seeds. '
			'Preserve both; do not select by test.'
		),
	}


def publication_contents(payload: dict[str, Any]) -> dict[str, str]:
	"""Return the exact three-file review set, excluding raw execution outputs."""
	stream = io.StringIO()
	fields = ['layout_id', 'data_size'] + [
		f'{arm}_{metric}'
		for metric in PUBLIC_METRIC_NAMES
		for arm in ('baseline', 'repeat', 'delta')
	]
	writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
	writer.writeheader()
	for row in payload['rows']:
		flat = {key: row[key] for key in ('layout_id', 'data_size')}
		for metric in PUBLIC_METRIC_NAMES:
			old, new = row['baseline'][metric], row['repeat'][metric]
			flat.update(
				{
					f'baseline_{metric}': old,
					f'repeat_{metric}': new,
					f'delta_{metric}': new - old,
				}
			)
		writer.writerow(flat)
	lines = [
		'# Parihaka MAE/no-HMM large: decoder seed repeat',
		'',
		(
			'Same encoder, embeddings, five layouts, 50 epochs '
			'and validation selection. '
			'Decoder seed 42000 versus 42001.'
		),
		'',
		(
			'| Metric | Original mean | New mean | Original median | New median '
			'| Original trimmed mean | New trimmed mean |'
		),
		'| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
	]
	for metric, values in payload['aggregates'].items():
		cells = [
			f'{values[arm][statistic]:.6f}'
			for statistic in ('mean', 'median', 'trimmed_mean')
			for arm in ('baseline', 'repeat')
		]
		lines.append('| ' + ' | '.join([metric, *cells]) + ' |')
	lines.extend(['', payload['interpretation'], ''])
	return {
		'comparison.csv': stream.getvalue(),
		'summary.json': json.dumps(payload, indent=2, sort_keys=True) + '\n',
		'summary.md': '\n'.join(lines),
	}


def publish_summary(repeat: SeedRepeat) -> tuple[Path, ...]:
	"""Publish five audited cells atomically; reject incomplete/conflicting output."""
	contents = publication_contents(inspect_summary(repeat))
	output = repeat.summary_root
	if output.exists():
		if {path.name for path in output.iterdir()} != set(PUBLISHED_FILES) or any(
			not (output / name).is_file()
			or (output / name).read_text(encoding='utf-8') != content
			for name, content in contents.items()
		):
			raise ValueError(
				'existing seed-repeat publication conflicts with audited results'
			)
	else:
		output.parent.mkdir(parents=True, exist_ok=True)
		staging = Path(
			tempfile.mkdtemp(prefix='.seed-repeat-summary-', dir=output.parent)
		)
		try:
			for name, content in contents.items():
				(staging / name).write_text(content, encoding='utf-8')
			staging.replace(output)
		finally:
			if staging.exists():
				shutil.rmtree(staging)
	return tuple(output / name for name in PUBLISHED_FILES)


def _write_status(repeat: SeedRepeat, **values: object) -> None:
	payload = {
		'experiment_id': EXPERIMENT_ID,
		'pid': os.getpid(),
		'decoder_seed': 42001,
		'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
		'updated_at_utc': datetime.now(timezone.utc).isoformat(),
		**values,
	}
	path = repeat.operations_root / 'status.json'
	temporary = path.with_suffix('.json.tmp')
	temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
	temporary.replace(path)


def _wait_for_gpu(
	repeat: SeedRepeat, device: str, layout: str, completed: list[str]
) -> None:
	"""Admit one decoder with 20 GiB free, without changing or killing other jobs."""
	if device == 'cpu' or (device == 'auto' and not torch.cuda.is_available()):
		return
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA was requested but is unavailable')
	while (free := torch.cuda.mem_get_info()[0]) < 20 * 1024**3:
		_write_status(
			repeat,
			status='waiting_for_gpu_memory',
			current_layout=layout,
			completed_layouts=completed,
			free_gpu_bytes=free,
		)
		print(f'WAIT {layout}: GPU free={free / 1024**3:.1f} GiB', flush=True)
		time.sleep(30)


def build_parser() -> argparse.ArgumentParser:
	"""Expose explicit planning, isolated execution and complete-only publication."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--layout', choices=LAYOUTS)
	parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cuda')
	parser.add_argument('--max-steps', type=int)
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument('--dry-run', action='store_true')
	mode.add_argument('--execute', action='store_true')
	mode.add_argument('--publish-summary', action='store_true')
	return parser


def main() -> None:
	"""Run five large MAE decoder cells serially and preserve canonical artifacts."""
	args = build_parser().parse_args()
	if args.max_steps is not None and (not args.execute or args.layout is None):
		raise ValueError('--max-steps requires --execute and one explicit layout')
	repeat = load_repeat(args.config)
	layouts = (args.layout,) if args.layout else LAYOUTS
	if not args.execute and not args.publish_summary:
		for layout in layouts:
			plan = inspect_repeat_plan(repeat, layout)
			print(
				json.dumps(
					{
						'layout': layout,
						'size': 'large',
						'seed': plan.config.train.seed,
						'tile_counts': dict(plan.tile_counts),
						'output': str(plan.output_dir),
						'decoder_initial_state_sha256': channel_decoder_run_identity(
							plan
						)['decoder_initial_state_sha256'],
					}
				),
				flush=True,
			)
		print('execution: dry-run; no files written', flush=True)
		return
	repeat.operations_root.mkdir(parents=True, exist_ok=True)
	with (repeat.operations_root / 'experiment.lock').open('a') as lock:
		fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		completed = []
		try:
			if args.publish_summary:
				publish_summary(repeat)
			else:
				for layout in layouts:
					_write_status(
						repeat,
						status='running',
						phase='preflight',
						current_layout=layout,
						completed_layouts=completed,
					)
					plan = inspect_repeat_plan(repeat, layout)
					if not (plan.output_dir / 'metrics.json').exists():
						latest = plan.output_dir / 'latest.pt'
						_wait_for_gpu(repeat, args.device, layout, completed)
						_write_status(
							repeat,
							status='running',
							phase='training_and_evaluation',
							current_layout=layout,
							completed_layouts=completed,
						)
						print(
							f'RUN {layout}: seed=42001; resume={latest.exists()}',
							flush=True,
						)
						result = run_channel_decoder_job(
							plan,
							device=args.device,
							max_steps=args.max_steps,
							resume=latest if latest.exists() else None,
						)
						if result is None:
							_write_status(
								repeat,
								status='partial',
								current_layout=layout,
								completed_layouts=completed,
							)
							return
					audit_repeat_cell(repeat, plan)
					completed.append(layout)
					print(f'COMPLETE {layout}', flush=True)
				if len(layouts) == len(LAYOUTS):
					_write_status(
						repeat,
						status='running',
						phase='summary',
						completed_layouts=completed,
					)
					publish_summary(repeat)
			_write_status(
				repeat,
				status='complete_requested_scope'
				if args.publish_summary or len(layouts) == len(LAYOUTS)
				else 'complete_layout',
				completed_layouts=list(LAYOUTS) if args.publish_summary else completed,
			)
		except (Exception, KeyboardInterrupt) as exc:
			_write_status(
				repeat, status='failed', completed_layouts=completed, error=str(exc)
			)
			raise


if __name__ == '__main__':
	main()
