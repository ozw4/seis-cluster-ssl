"""Audit and summarize the nine-arm Parihaka pretraining comparison."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.parihaka.channel_completion import inspect_completed_channel_job
from seis_ssl_cluster.parihaka.channel_data import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.parihaka.channel_results import (
	ChannelSummaryConfig,
	inspect_channel_model_results,
)

OUTPUT_NAMES = ('comparison.csv', 'summary.json', 'summary.md')
EXPECTED_MODEL_IDS = (
	'mae',
	'mae_hmm_k6_distill010',
	'mae_hmm_k6',
	'local_barlow_twins_rot90_asym_g060_3ep',
	'local_barlow_twins_rot90_asym_g060_3ep_hmm_k6_distill010',
	'local_barlow_twins_rot90_asym_g060_3ep_hmm_k6_distill020',
	'random',
	'random_init_self_target_hmm_k6_distill010',
	'random_init_self_target_hmm_k6_distill020',
)


def _verify_recorded_hashes(value: object) -> None:
	if isinstance(value, Mapping):
		if 'path' in value and 'sha256' in value:
			path = Path(str(value['path']))
			if file_sha256(path) != value['sha256']:
				raise ValueError(f'live training input checksum mismatch: {path}')
		else:
			for nested in value.values():
				_verify_recorded_hashes(nested)
	elif isinstance(value, list):
		for nested in value:
			_verify_recorded_hashes(nested)


def checkpoint_complete(checkpoint: Path, train_config: Path) -> bool:
	"""Check the full HMM budget and resolved scientific configuration."""
	if not checkpoint.is_file():
		return False
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	expected = resolve_strat_hmm_pretext_config(load_config(train_config))
	stored = payload['stratigraphy_config']
	for section in (
		'data',
		'zero_mask',
		'manifests',
		'model',
		'pseudo_targets',
		'teacher',
		'student',
		'head',
		'loss',
	):
		if stored[section] != expected[section]:
			raise ValueError(f'{checkpoint}: mismatched {section} configuration')
	for field in (
		'epochs',
		'batch_size',
		'samples_per_epoch',
		'lr',
		'encoder_lr',
		'seed',
		'amp',
		'weight_decay',
		'grad_clip_norm',
		'shuffle',
		'max_steps',
	):
		if stored['train'][field] != expected['train'][field]:
			raise ValueError(f'{checkpoint}: mismatched train.{field}')
	if 'identity' in expected and stored.get('identity') != expected['identity']:
		raise ValueError(f'{checkpoint}: mismatched scientific identity')
	if 'identity' in expected:
		control = payload['control_identity']
		for field in ('model_tag', 'scientific_identity'):
			if control[field] != expected['identity'][field]:
				raise ValueError(f'{checkpoint}: mismatched control identity {field}')
		_verify_recorded_hashes(control['input_identities'])
	epochs = expected['train']['epochs']
	steps = epochs * (
		expected['train']['samples_per_epoch'] // expected['train']['batch_size']
	)
	training_state = _mapping(payload.get('training_state', {}))
	complete = (
		payload['epoch'] == epochs
		and payload['global_step'] == steps
		and training_state.get('stage') == 'train_strat_hmm_pretext'
		and training_state.get('checkpoint_kind') == 'epoch'
	)
	print(
		json.dumps(
			{
				'checkpoint': str(checkpoint),
				'epoch': payload['epoch'],
				'global_step': payload['global_step'],
				'complete': complete,
			}
		)
	)
	return complete


def _mapping(value: object) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('comparison configuration section must be a mapping')
	return value


def _common_identity(payload: Mapping[str, object]) -> dict[str, object]:
	identity = _mapping(payload['benchmark_identity'])
	embedding = _mapping(identity['embedding'])
	metadata = _mapping(embedding['common_metadata'])
	return {
		**{
			key: value
			for key, value in identity.items()
			if key not in {'model', 'embedding'}
		},
		'embedding': {
			'common_metadata': {
				key: value
				for key, value in metadata.items()
				if key != 'pretraining_objective'
			},
		},
	}


def _inspect_checkpoint(entry: Mapping[str, object]) -> dict[str, object]:
	checkpoint = Path(str(entry['checkpoint']))
	if 'train_config' in entry and not checkpoint_complete(
		checkpoint, Path(str(entry['train_config']))
	):
		raise ValueError(f'{checkpoint}: incomplete HMM training')
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	for field in ('epoch', 'global_step'):
		if payload[field] != entry[field]:
			raise ValueError(f'{checkpoint}: checkpoint {field} is incomplete')
	if 'distillation_weight' in entry:
		stratigraphy = payload['stratigraphy_config']
		if stratigraphy['loss']['distillation_weight'] != entry['distillation_weight']:
			raise ValueError(f'{checkpoint}: checkpoint distillation weight mismatch')
		if (
			stratigraphy['teacher']['checkpoint']
			!= stratigraphy['student']['init_checkpoint']
		):
			raise ValueError(f'{checkpoint}: teacher must match student initialization')
		if stratigraphy['pseudo_targets']['k'] != 6:
			raise ValueError(f'{checkpoint}: expected K=6 pseudo targets')
	return {
		'sha256': file_sha256(checkpoint),
		'epoch': payload['epoch'],
		'global_step': payload['global_step'],
	}


def inspect_comparison(
	config: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
	"""Require nine complete model arms with matching downstream identities."""
	models = _mapping(config['models'])
	if set(models) != set(EXPECTED_MODEL_IDS):
		raise ValueError(
			'the requested comparison requires exactly nine specified arms'
		)
	if any(_mapping(entry)['model_id'] != key for key, entry in models.items()):
		raise ValueError('comparison arm keys and model IDs must match')
	output_dir = Path(str(config['output_dir']))
	rows: list[dict[str, object]] = []
	identities: dict[tuple[str, str], dict[str, object]] = {}
	checkpoints: dict[str, dict[str, object]] = {}
	for arm_id, raw in models.items():
		entry = _mapping(raw)
		model_id = str(entry['model_id'])
		checkpoints[arm_id] = _inspect_checkpoint(entry)
		checkpoint_hash = checkpoints[arm_id]['sha256']
		jobs = inspect_channel_model_results(
			ChannelSummaryConfig(Path(str(entry['runs_root'])), output_dir),
			model_ids=('pretrained', 'random') if model_id == 'random' else (model_id,),
		)
		for layout_id in LAYOUT_IDS:
			for size in DATA_SIZE_PREFIX:
				metrics = jobs[(model_id, layout_id, size)]
				inspect_completed_channel_job(
					Path(str(entry['runs_root']))
					/ f'model={model_id}'
					/ f'layout={layout_id}'
					/ f'size={size}',
					metrics=metrics,
				)
				benchmark = _mapping(metrics['benchmark_identity'])
				embedding = _mapping(benchmark['embedding'])
				if embedding['checkpoint_sha256'] != checkpoint_hash:
					raise ValueError(
						f'{arm_id}/{layout_id}/{size}: stale checkpoint metrics'
					)
				identity = _common_identity(metrics)
				key = (layout_id, size)
				if identities.setdefault(key, identity) != identity:
					raise ValueError(
						f'{arm_id}/{layout_id}/{size}: downstream identity mismatch'
					)
				test = _mapping(metrics['test'])
				rows.append(
					{
						'arm_id': arm_id,
						'model_id': model_id,
						'layout_id': layout_id,
						'data_size': size,
						'test_channel_iou': test['channel_iou'],
					}
				)
	return rows, checkpoints


def summarize_comparison(config: Mapping[str, object]) -> tuple[Path, Path, Path]:
	"""Write the exact three-file result set after a full 135-cell audit."""
	rows, checkpoints = inspect_comparison(config)
	output_dir = Path(str(config['output_dir']))
	if output_dir.exists():
		raise FileExistsError(f'refusing to overwrite comparison: {output_dir}')
	models = _mapping(config['models'])
	means = {
		arm_id: {
			size: statistics.fmean(
				float(cast('float', row['test_channel_iou']))
				for row in rows
				if row['arm_id'] == arm_id
				and (size == 'all_15' or row['data_size'] == size)
			)
			for size in (*DATA_SIZE_PREFIX, 'all_15')
		}
		for arm_id in models
	}
	payload = {
		'schema_version': 1,
		'survey': 'parihaka',
		'complete_arms': len(models),
		'complete_cells': len(rows),
		'primary_metric': 'test.channel_iou',
		'downstream_identity_parity': True,
		'checkpoints': checkpoints,
		'model_means': means,
		'comparison': rows,
	}
	stream = io.StringIO(newline='')
	writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
	writer.writeheader()
	writer.writerows(rows)
	markdown = [
		'# Parihaka nine-arm pretraining comparison',
		'',
		'All 135 held-out test cells passed checkpoint and benchmark identity audits.',
		'',
		'| Arm | Small | Medium | Large | All 15 |',
		'|---|---:|---:|---:|---:|',
	]
	for arm_id, values in means.items():
		markdown.append(
			f'| {arm_id} | '
			+ ' | '.join(
				f'{values[size]:.6f}' for size in (*DATA_SIZE_PREFIX, 'all_15')
			)
			+ ' |'
		)
	contents = (
		stream.getvalue(),
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		'\n'.join(markdown) + '\n',
	)
	output_dir.mkdir(parents=True)
	paths = tuple(output_dir / name for name in OUTPUT_NAMES)
	for path, content in zip(paths, contents, strict=True):
		path.write_text(content, encoding='utf-8')
	return cast('tuple[Path, Path, Path]', paths)


def main() -> None:
	"""Run a focused checkpoint audit or the comparison summary."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--checkpoint-complete', type=Path)
	parser.add_argument('--train-config', type=Path)
	parser.add_argument('--summary-config', type=Path)
	parser.add_argument('--check-only', action='store_true')
	args = parser.parse_args()
	if args.checkpoint_complete is not None:
		if args.train_config is None:
			parser.error('--train-config is required for a checkpoint audit')
		complete = checkpoint_complete(args.checkpoint_complete, args.train_config)
		raise SystemExit(0 if complete else 1)
	if args.summary_config is not None:
		config = load_config(args.summary_config)
		if args.check_only:
			rows, checkpoints = inspect_comparison(config)
			print(
				json.dumps(
					{'complete_cells': len(rows), 'complete_arms': len(checkpoints)}
				)
			)
		else:
			for path in summarize_comparison(config):
				print(f'output: {path}')
		return
	parser.error('Select a checkpoint audit or summary')


if __name__ == '__main__':
	main()
