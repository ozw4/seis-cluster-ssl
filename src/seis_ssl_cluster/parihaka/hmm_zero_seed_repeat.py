"""Audited decoder-seed repeat for Parihaka HMM v2 with both constraints zero."""

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

NAMESPACE = 'hmm_v2_dis0_con0_decoder_seed42001_v1'
EXPERIMENT_ID = '54_' + NAMESPACE
LAYOUTS = tuple(f'layout_{index:03d}' for index in range(5))
SIZES = ('medium', 'large')
SOURCES = ('mae', 'local_bt', 'random')
PUBLISHED_FILES = ('cells.csv', 'comparison.csv', 'provenance.json')


@dataclass(frozen=True)
class Repeat:
	"""Frozen source definitions and isolated destinations for thirty cells."""

	definition: dict[str, Any]
	baselines: dict[str, ChannelDecoderConfig]
	candidates: dict[str, ChannelDecoderConfig]
	layout_config: Path
	summary_root: Path
	operations_root: Path


def load_repeat(path: Path) -> Repeat:  # noqa: C901
	"""Bind all source hashes and change only decoder seed and output location."""
	raw = load_config(path)
	expected = {
		'schema_version': 1,
		'experiment_id': EXPERIMENT_ID,
		'layouts': list(LAYOUTS),
		'sizes': list(SIZES),
		'baseline_decoder_seed': 42000,
		'decoder_seed': 42001,
		'cell_count': 30,
	}
	if any(
		raw.get(key) != value or type(raw.get(key)) is not type(value)
		for key, value in expected.items()
	) or [arm['source'] for arm in raw['arms']] != list(SOURCES):
		raise ValueError('repeat must retain all thirty zero/zero HMM v2 cells')
	artifact = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT']).resolve()
	benchmark = artifact / 'surveys/parihaka/facies_benchmark_v1/benchmark' / NAMESPACE
	for key, target in (
		('runs_root', benchmark / 'runs'),
		('summary_root', benchmark / 'summary'),
		('operations_root', artifact / 'operations/parihaka' / NAMESPACE),
	):
		if Path(raw[key]).resolve() != target:
			raise ValueError(f'{key} must use the isolated repeat namespace')
	layout = Path(raw['layout_config'])
	if file_sha256(layout) != raw['layout_config_sha256']:
		raise ValueError('layout definition hash drift')
	baselines, candidates = {}, {}
	for arm in raw['arms']:
		for key in ('baseline_config', 'training_config'):
			path = Path(arm[key])
			if 'reports' in path.resolve().parts:
				raise ValueError('reports cannot be repeat inputs')
			if file_sha256(path) != arm[key + '_sha256']:
				raise ValueError(f'{key} definition hash drift')
		training = load_config(Path(arm['training_config']))
		if (
			training['head']['ks'] != [6, 8, 10]
			or training['loss']['distillation_weight'] != 0.0
			or training['loss']['consistency_weight'] != 0.0
			or training['train']['seed'] != 42
			or training['identity']['model_tag'] != arm['model']
		):
			raise ValueError('encoder must retain the original zero/zero HMM v2')
		baseline = channel_decoder_config_from_mapping(
			load_config(Path(arm['baseline_config']))
		)
		if baseline.survey_id != 'parihaka' or set(baseline.models) != {arm['model']}:
			raise ValueError('baseline survey/model drift')
		checkpoint = baseline.models[arm['model']].expected_checkpoint
		if checkpoint is None or file_sha256(checkpoint) != arm['checkpoint_sha256']:
			raise ValueError('frozen HMM encoder checkpoint hash drift')
		if set(arm['baseline_metrics_sha256']) != {
			f'{layout}/{size}' for layout in LAYOUTS for size in SIZES
		}:
			raise ValueError('baseline must bind both sizes for all five layouts')
		baselines[arm['source']] = baseline
		candidates[arm['source']] = replace(
			baseline,
			train=replace(baseline.train, seed=42001),
			runs_root=Path(raw['runs_root']),
		)
	return Repeat(
		raw,
		baselines,
		candidates,
		layout,
		Path(raw['summary_root']),
		Path(raw['operations_root']),
	)


def baseline_metrics(repeat: Repeat, plan: ChannelDecoderPlan) -> dict[str, Any]:
	"""Check the source result and exact seed-normalized benchmark identity."""
	arm = next(arm for arm in repeat.definition['arms'] if arm['model'] == plan.model)
	path = repeat.baselines[arm['source']].runs_root / (
		f'model={plan.model}/layout={plan.layout_id}/size={plan.data_size}/metrics.json'
	)
	if (
		file_sha256(path)
		!= arm['baseline_metrics_sha256'][f'{plan.layout_id}/{plan.data_size}']
	):
		raise ValueError('baseline metrics hash drift')
	metrics = json.loads(path.read_text(encoding='utf-8'))
	identity = copy.deepcopy(channel_decoder_run_identity(plan))
	pretext = identity['embedding']['model_source'].get('stratigraphy_pretext', {})
	if (
		identity['embedding']['checkpoint_sha256'] != arm['checkpoint_sha256']
		or pretext.get('head_ks') != [6, 8, 10]
		or pretext.get('distillation_weight') != 0.0
		or pretext.get('consistency_weight') != 0.0
	):
		raise ValueError('repeat requires the original zero/zero HMM encoder')
	identity['training']['seed'] = 42000
	identity['decoder_initial_state_sha256'] = decoder_initial_state_sha256(
		plan.config.decoder,
		plan.geometry.patch_size_xyz,
		42000,
	)
	if metrics['benchmark_identity'] != identity:
		raise ValueError('repeat changed settings other than decoder seed')
	inspect_completed_channel_job(path.parent, metrics=metrics)
	return metrics


def inspect_plan(
	repeat: Repeat, source: str, layout: str, size: str
) -> ChannelDecoderPlan:
	"""Inspect one supported cell before allowing any training output."""
	if source not in SOURCES or layout not in LAYOUTS or size not in SIZES:
		raise ValueError('unsupported repeat cell')
	config = repeat.candidates[source]
	plan = inspect_channel_decoder_job(
		config,
		model=next(iter(config.models)),
		layout_id=layout,
		data_size=size,
		layout_config=repeat.layout_config,
	)
	baseline_metrics(repeat, plan)
	return plan


def audit_cell(repeat: Repeat, plan: ChannelDecoderPlan) -> dict[str, Any]:
	"""Require the complete own-seed fifty-epoch validation-selected result."""
	baseline_metrics(repeat, plan)
	metrics = json.loads((plan.output_dir / 'metrics.json').read_text(encoding='utf-8'))
	if metrics['benchmark_identity'] != channel_decoder_run_identity(plan):
		raise ValueError('completed repeat has a different benchmark identity')
	inspect_completed_channel_job(plan.output_dir, metrics=metrics)
	return metrics


def publication_contents(rows: list[dict[str, Any]]) -> dict[str, str]:
	"""Publish all paired metrics and six groups without selecting a seed."""
	keys = {(row['source'], row['layout_id'], row['data_size']) for row in rows}
	if len(rows) != 30 or keys != {
		(source, layout, size)
		for source in SOURCES
		for layout in LAYOUTS
		for size in SIZES
	}:
		raise ValueError('summary requires all thirty unique completed cells')
	groups = []
	for source in SOURCES:
		for size in SIZES:
			selected = [
				r for r in rows if r['source'] == source and r['data_size'] == size
			]
			group = {'source': source, 'data_size': size, 'n_layouts': 5}
			for metric in PUBLIC_METRIC_NAMES:
				for arm in ('baseline', 'repeat', 'delta'):
					values = [r[f'{arm}_{metric}'] for r in selected]
					group[f'{arm}_{metric}_mean'] = statistics.mean(values)
					group[f'{arm}_{metric}_std'] = statistics.stdev(values)
			groups.append(group)
	contents = {}
	for name, records in (('cells.csv', rows), ('comparison.csv', groups)):
		stream = io.StringIO()
		writer = csv.DictWriter(
			stream, fieldnames=list(records[0]), lineterminator='\n'
		)
		writer.writeheader()
		writer.writerows(records)
		contents[name] = stream.getvalue()
	contents['provenance.json'] = (
		json.dumps(
			{
				'experiment_id': EXPERIMENT_ID,
				'cell_count': 30,
				'baseline_decoder_seed': 42000,
				'decoder_seed': 42001,
				'encoder_seed': 42,
				'head_ks': [6, 8, 10],
				'distillation_weight': 0.0,
				'consistency_weight': 0.0,
				'epochs': 50,
				'selection': 'first maximum validation Channel IoU',
				'published_files': list(PUBLISHED_FILES),
				'interpretation': (
					'Two decoder seeds with the same frozen encoder; retain both.'
				),
			},
			indent=2,
		)
		+ '\n'
	)
	return contents


def publish_summary(repeat: Repeat) -> None:
	"""Audit all thirty results and atomically publish only the owned three files."""
	rows = []
	for source in SOURCES:
		for size in SIZES:
			for layout in LAYOUTS:
				plan = inspect_plan(repeat, source, layout, size)
				original, candidate = (
					baseline_metrics(repeat, plan),
					audit_cell(repeat, plan),
				)
				row = {
					'source': source,
					'model': plan.model,
					'layout_id': layout,
					'data_size': size,
					'baseline_decoder_seed': 42000,
					'decoder_seed': 42001,
					'baseline_best_epoch': original['best_epoch'],
					'repeat_best_epoch': candidate['best_epoch'],
					'repeat_metrics_sha256': file_sha256(
						plan.output_dir / 'metrics.json'
					),
				}
				for metric in PUBLIC_METRIC_NAMES:
					old, new = original['test'][metric], candidate['test'][metric]
					row.update(
						{
							f'baseline_{metric}': old,
							f'repeat_{metric}': new,
							f'delta_{metric}': new - old,
						}
					)
				rows.append(row)
	contents = publication_contents(rows)
	output = repeat.summary_root
	if output.exists():
		if {p.name for p in output.iterdir()} != set(PUBLISHED_FILES) or any(
			(output / name).read_text(encoding='utf-8') != content
			for name, content in contents.items()
		):
			raise ValueError('existing publication conflicts with audited repeat')
		return
	output.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(tempfile.mkdtemp(prefix='.hmm-seed-summary-', dir=output.parent))
	try:
		for name, content in contents.items():
			(staging / name).write_text(content, encoding='utf-8')
		staging.replace(output)
	finally:
		if staging.exists():
			shutil.rmtree(staging)


def write_status(repeat: Repeat, **values: object) -> None:
	"""Atomically record restartable execution state outside the review output."""
	payload = {
		'experiment_id': EXPERIMENT_ID,
		'pid': os.getpid(),
		'decoder_seed': 42001,
		'total_cells': 30,
		'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
		'updated_at_utc': datetime.now(timezone.utc).isoformat(),
		**values,
	}
	path = repeat.operations_root / 'status.json'
	temporary = path.with_suffix('.json.tmp')
	temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
	temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
	"""Expose isolated execution and read-only planning through the standard CLI."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument('--dry-run', action='store_true')
	mode.add_argument('--execute', action='store_true')
	mode.add_argument('--publish-summary', action='store_true')
	return parser


def main() -> None:  # noqa: C901
	"""Preflight or execute all thirty cells serially, resuming only their own state."""
	args = build_parser().parse_args()
	repeat = load_repeat(args.config)
	if not args.execute and not args.publish_summary:
		for source in SOURCES:
			for size in SIZES:
				for layout in LAYOUTS:
					plan = inspect_plan(repeat, source, layout, size)
					print(
						json.dumps(
							{
								'source': source,
								'layout': layout,
								'size': size,
								'seed': plan.config.train.seed,
								'tiles': dict(plan.tile_counts),
								'output': str(plan.output_dir),
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
			if not args.publish_summary:
				for source in SOURCES:
					for size in SIZES:
						for layout in LAYOUTS:
							cell = f'{source}/{layout}/{size}'
							write_status(
								repeat,
								status='running',
								phase='preflight',
								current_cell=cell,
								completed_cells=completed,
							)
							plan = inspect_plan(repeat, source, layout, size)
							if not (plan.output_dir / 'metrics.json').exists():
								if args.device == 'cuda':
									while torch.cuda.mem_get_info()[0] < 20 * 1024**3:
										write_status(
											repeat,
											status='waiting_for_gpu_memory',
											current_cell=cell,
											completed_cells=completed,
										)
										time.sleep(30)
								latest = plan.output_dir / 'latest.pt'
								write_status(
									repeat,
									status='running',
									phase='training_and_evaluation',
									current_cell=cell,
									completed_cells=completed,
								)
								print(
									f'RUN {cell}: seed=42001; resume={latest.exists()}',
									flush=True,
								)
								run_channel_decoder_job(
									plan,
									device=args.device,
									resume=latest if latest.exists() else None,
								)
							audit_cell(repeat, plan)
							completed.append(cell)
							print(f'COMPLETE {cell} ({len(completed)}/30)', flush=True)
			write_status(
				repeat, status='running', phase='summary', completed_cells=completed
			)
			publish_summary(repeat)
			write_status(repeat, status='complete_requested_scope', completed_count=30)
		except (Exception, KeyboardInterrupt) as exc:
			write_status(
				repeat, status='failed', completed_cells=completed, error=str(exc)
			)
			raise


if __name__ == '__main__':
	main()
