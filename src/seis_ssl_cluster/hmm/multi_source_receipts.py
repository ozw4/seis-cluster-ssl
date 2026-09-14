"""Freeze matching controls and the selected F3 arm without runtime report inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology.hmm_v1_freeze_receipt import (
	build_control_freeze_receipt,
	canonical_control_cells_sha256,
)

SURVEYS = ('f3', 'parihaka', 'volve')
SOURCE_FAMILIES = ('mae', 'local_bt', 'random')
DATA_SIZES = ('small', 'medium', 'large')
CELL_IDENTITIES = {(f'layout_{i:03d}', size) for i in range(5) for size in DATA_SIZES}
CANDIDATE_IDS = {
	'mae': 'mae100_hmm_v2_mh_k6810_distill020',
	'local_bt': 'local_bt_rot90_asym_g060_3ep_hmm_v2_mh_k6810_distill020',
	'random': 'random_init_hmm_v2_mh_k6810_distill020',
}
CONDITIONS = {'mae': '2b', 'local_bt': '4b', 'random': '6b'}
METRICS = {
	'f3': {
		'task': 'lithology facies classification',
		'evaluation_split': 'validation',
		'primary_metric': 'mean_iou_unique_validation_voxels',
		'primary_direction': 'higher',
		'secondary_metric': 'macro_f1_unique_validation_voxels',
	},
	'parihaka': {
		'task': 'channel facies classification',
		'evaluation_split': 'test',
		'primary_metric': 'test_channel_iou',
		'primary_direction': 'higher',
		'secondary_metric': 'test_channel_f1',
	},
	'volve': {
		'task': 'five-horizon estimation',
		'evaluation_split': 'test',
		'primary_metric': 'test_macro_mae_samples',
		'primary_direction': 'lower',
		'secondary_metric': 'test_macro_within_2_samples',
	},
}


def file_sha256(path: Path) -> str:
	"""Hash files in bounded memory, including external embeddings."""
	hasher = hashlib.sha256()
	with path.open('rb') as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b''):
			hasher.update(block)
	return hasher.hexdigest()


def object_sha256(value: object) -> str:
	"""Digest canonical JSON without rounding numeric evidence."""
	return hashlib.sha256(
		json.dumps(
			value, sort_keys=True, separators=(',', ':'), allow_nan=False
		).encode()
	).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
	"""Read an object, rejecting non-object JSON."""
	value = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(value, dict):
		raise TypeError(f'expected JSON object: {path}')
	return value


def _digest(value: object) -> None:
	if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
		raise ValueError('expected lowercase SHA-256 digest')


def artifact_path(value: str, artifact_root: Path) -> Path:
	"""Resolve a portable artifact identity without permitting namespace escape."""
	path = Path(value)
	if path.is_absolute() or '..' in path.parts or 'reports' in path.parts:
		raise ValueError('artifact identity must be relative and outside reports')
	resolved = (artifact_root / path).resolve()
	if not resolved.is_relative_to(artifact_root.resolve()):
		raise ValueError('artifact identity escapes artifact root')
	return resolved


def canonical_cells_sha256(survey: str, cells: list[dict[str, Any]]) -> str:
	"""Validate and hash fifteen normalized primary/secondary metric cells."""
	if survey not in SURVEYS:
		raise ValueError('unknown survey')
	ids = [(row['layout_id'], row['data_size']) for row in cells]
	if len(ids) != 15 or set(ids) != CELL_IDENTITIES:
		raise ValueError('expected exactly 15 distinct layout/data_size cells')
	for row in cells:
		for metric in ('primary', 'secondary'):
			value = row[metric]
			if (
				type(value) not in (float, int)
				or not math.isfinite(value)
				or value < 0
				or ((survey != 'volve' or metric == 'secondary') and value > 1)
			):
				raise ValueError('invalid finite metric range')
	# F3 retains the established Condition 2b digest representation.
	if survey == 'f3':
		return canonical_control_cells_sha256(
			[{**r, 'mean_iou': r['primary'], 'macro_f1': r['secondary']} for r in cells]
		)
	return object_sha256(
		[
			[r['layout_id'], r['data_size'], float(r['primary']), float(r['secondary'])]
			for r in sorted(cells, key=lambda r: (r['layout_id'], r['data_size']))
		]
	)


def validate_fixed_training_config(config: dict[str, Any]) -> None:
	"""Enforce the selected scientific recipe while retaining family batch/workers."""
	expected = {
		'head': {
			'spec': 'multi_resolution_ordered_prototypes_v1',
			'ks': [6, 8, 10],
			'projection_dim': 128,
			'temperature': 0.1,
			'normalize': True,
		},
		'loss': {
			'prototype_weight': 1.0,
			'usage_weight': 0.005,
			'distillation_weight': 0.2,
			'consistency_weight': 0.0,
			'consistency_beta': 0.1,
		},
		'student': {'unfreeze_top_blocks': 1},
		'pseudo_targets': {'target_representation': 'hard_viterbi_labels_v1'},
		'train': {
			'epochs': 25,
			'samples_per_epoch': 10000,
			'seed': 42,
			'amp': False,
			'lr': 1e-5,
			'encoder_lr': 1e-5,
			'allow_overwrite_output': False,
			'max_steps': None,
		},
	}
	for section, fields in expected.items():
		if not isinstance(config.get(section), dict) or any(
			config[section].get(key) != value for key, value in fields.items()
		):
			raise ValueError(f'fixed K6810 training contract drift: {section}')


def load_matrix(path: Path) -> dict[str, Any]:
	"""Require the complete fixed matrix; reject result-dependent switches."""
	matrix = load_config(path)
	if set(matrix) != {
		'schema_version',
		'selected_head_ks',
		'distillation_weight',
		'consistency_weight',
		'surveys',
	} or any(
		matrix.get(k) != v
		for k, v in {
			'schema_version': 1,
			'selected_head_ks': [6, 8, 10],
			'distillation_weight': 0.2,
			'consistency_weight': 0.0,
		}.items()
	):
		raise ValueError('matrix must preserve the fixed K6810 scientific contract')
	if set(matrix['surveys']) != set(SURVEYS):
		raise ValueError('matrix must contain exactly three surveys')
	for survey, entry in matrix['surveys'].items():
		if (
			set(entry) != {'evaluation_split', 'arms'}
			or entry['evaluation_split'] != METRICS[survey]['evaluation_split']
		):
			raise ValueError('invalid survey evaluation contract')
		if set(entry['arms']) != set(SOURCE_FAMILIES):
			raise ValueError('matrix must contain exactly three source families')
		for family, arm in entry['arms'].items():
			if arm != {
				'candidate_id': CANDIDATE_IDS[family],
				'execution': 'reuse' if (survey, family) == ('f3', 'mae') else 'new',
			}:
				raise ValueError('matrix arm identity or execution drift')
	return matrix


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
	return {**payload, 'receipt_sha256': object_sha256(payload)}


def _check_seal(receipt: dict[str, Any]) -> None:
	if receipt.get('receipt_sha256') != object_sha256(
		{k: v for k, v in receipt.items() if k != 'receipt_sha256'}
	):
		raise ValueError('receipt content digest drift')


def _validate_control(entry: dict[str, Any]) -> None:
	survey, family = entry['survey'], entry['source_family']
	if (
		survey not in SURVEYS
		or family not in SOURCE_FAMILIES
		or entry['condition_id'] != CONDITIONS[family]
	):
		raise ValueError('matching control family drift')
	if any(entry.get(key) != value for key, value in METRICS[survey].items()):
		raise ValueError('matching control metric contract drift')
	_digest(entry['checkpoint_sha256'])
	artifact_path(entry['checkpoint_path'], Path('/artifact_identity'))
	if (
		not entry['checkpoint_id']
		or canonical_cells_sha256(survey, entry['cells'])
		!= entry['canonical_cells_sha256']
	):
		raise ValueError('matching control cells drift')


def load_control_receipts(path: Path) -> dict[str, Any]:
	"""Load nine frozen controls without opening reports or live artifacts."""
	receipt = read_json(path)
	_check_seal(receipt)
	entries = receipt['controls']
	if (
		receipt.get('schema_version') != 1
		or len(entries) != 9
		or {(e['survey'], e['source_family']) for e in entries}
		!= {(s, f) for s in SURVEYS for f in SOURCE_FAMILIES}
	):
		raise ValueError('expected exactly nine matching controls')
	for entry in entries:
		_validate_control(entry)
	return receipt


def control_receipt_for(
	receipt: dict[str, Any], survey: str, source_family: str
) -> dict[str, Any]:
	"""Select the one matching Single-Head source family."""
	entries = [
		e
		for e in receipt['controls']
		if (e['survey'], e['source_family']) == (survey, source_family)
	]
	if len(entries) != 1:
		raise ValueError('missing or duplicate matching control')
	_validate_control(entries[0])
	return entries[0]


def verify_control(
	entry: dict[str, Any], checkpoint_sha256: str, cells: list[dict[str, Any]]
) -> None:
	"""Compare live audited checkpoint and metrics against the frozen control."""
	_validate_control(entry)
	if checkpoint_sha256 != entry['checkpoint_sha256']:
		raise ValueError('control checkpoint hash drift')
	if (
		canonical_cells_sha256(entry['survey'], cells)
		!= entry['canonical_cells_sha256']
	):
		raise ValueError('control cell metric drift')


def build_control_receipts(freeze_root: Path) -> dict[str, Any]:
	"""Explicitly derive nine controls after validating all authoritative records."""
	# The public legacy builder verifies all 405 cells and both result file hashes.
	legacy = build_control_freeze_receipt(freeze_root)
	manifest = read_json(freeze_root / 'manifest.json')
	summary = read_json(freeze_root / 'summary.json')
	with (freeze_root / 'cells.csv').open(encoding='utf-8', newline='') as stream:
		rows = list(csv.DictReader(stream))
	entries = []
	for survey in SURVEYS:
		for family, condition in CONDITIONS.items():
			checkpoints = [
				e
				for e in manifest['checkpoints']
				if (e['dataset'], e['condition_id']) == (survey, condition)
			]
			if len(checkpoints) != 1 or checkpoints[0]['present_at_freeze'] is not True:
				raise ValueError('missing frozen checkpoint identity')
			checkpoint = checkpoints[0]
			metrics = summary['datasets'][survey]
			if (
				any(
					metrics[k] != METRICS[survey][k]
					for k in ('task', 'primary_metric', 'secondary_metric')
				)
				or metrics['direction'] != METRICS[survey]['primary_direction']
			):
				raise ValueError('frozen survey metric contract drift')
			cells = [
				{
					'layout_id': r['layout_id'],
					'data_size': r['data_size'],
					'primary': float(r['primary_value']),
					'secondary': float(r['secondary_value']),
				}
				for r in rows
				if (r['dataset'], r['condition_id']) == (survey, condition)
			]
			entries.append(
				{
					'survey': survey,
					'source_family': family,
					'condition_id': condition,
					**METRICS[survey],
					'checkpoint_id': checkpoint['checkpoint_id'],
					'checkpoint_path': checkpoint['path'],
					'checkpoint_sha256': checkpoint['sha256'],
					'cells': cells,
					'canonical_cells_sha256': canonical_cells_sha256(survey, cells),
				}
			)
	return _seal(
		{
			'schema_version': 1,
			'freeze_date': legacy['freeze_date'],
			'freeze_record_sha256': legacy['freeze_record_sha256'],
			'controls': entries,
		}
	)


def _validate_reuse(receipt: dict[str, Any]) -> None:
	_check_seal(receipt)
	if any(
		receipt.get(k) != v
		for k, v in {
			'schema_version': 1,
			'candidate_id': CANDIDATE_IDS['mae'],
			'head_ks': [6, 8, 10],
			'distillation_weight': 0.2,
			'consistency_weight': 0.0,
		}.items()
	):
		raise ValueError('F3 reuse scientific contract drift')
	if (
		canonical_cells_sha256('f3', receipt['cells'])
		!= receipt['canonical_cells_sha256']
	):
		raise ValueError('F3 reuse cell drift')
	for name in (
		'checkpoint',
		'embeddings',
		'valid_tokens',
		'embedding_metadata',
		'source_summary',
		'resolved_training_config',
	):
		identity = receipt[name]
		_digest(identity['sha256'])
		artifact_path(identity['path'], Path('/artifact_identity'))
	for cell in receipt['cells']:
		_digest(cell['metrics_sha256'])
		artifact_path(cell['metrics_path'], Path('/artifact_identity'))


def load_reuse_receipt(path: Path) -> dict[str, Any]:
	"""Read selected-arm evidence without executing any part of experiment 127."""
	receipt = read_json(path)
	_validate_reuse(receipt)
	return receipt


def build_reuse_receipt(source_summary: Path, artifact_root: Path) -> dict[str, Any]:
	"""Extract selected-arm evidence from an already completed paired summary."""
	summary = read_json(source_summary)
	selected = [
		e for e in summary['candidates'] if e['candidate_id'] == CANDIDATE_IDS['mae']
	]
	if (
		summary['status'] != 'complete'
		or len(selected) != 1
		or selected[0]['head_ks'] != [6, 8, 10]
	):
		raise ValueError('expected complete selected F3 K6810 summary')
	source = selected[0]['sources']['candidate']
	lineage = source['semantic_lineage']
	if (
		lineage['status'] != 'complete'
		or lineage['head_ks'] != [6, 8, 10]
		or lineage['epoch'] != 25
		or lineage['model_tag'] != CANDIDATE_IDS['mae']
		or lineage['stratigraphy_checkpoint']['consistency_weight'] != 0
	):
		raise ValueError('selected source audit drift')
	if lineage['checkpoint'] != {
		'path': source['checkpoint_path'],
		'sha256': source['checkpoint_sha256'],
	}:
		raise ValueError('checkpoint and source audit disagree')

	resolved_path = Path(source['checkpoint_path']).parent / 'resolved_config.json'
	resolved = read_json(resolved_path)
	validate_fixed_training_config(resolved)
	from seis_ssl_cluster.training.random_checkpoint import (  # noqa: PLC0415
		load_checkpoint_metadata_without_weights,
	)

	checkpoint_metadata = load_checkpoint_metadata_without_weights(
		Path(source['checkpoint_path'])
	)
	if checkpoint_metadata.get('stratigraphy_config') != resolved:
		raise ValueError('resolved training config differs from checkpoint')
	if (
		resolved['head']['ks'] != [6, 8, 10]
		or resolved['loss']['distillation_weight'] != 0.2
		or resolved['loss']['consistency_weight'] != 0.0
		or resolved['train']['epochs'] != 25
		or resolved['train']['seed'] != 42
	):
		raise ValueError('selected resolved training contract drift')

	def relative(value: str | Path) -> str:
		return str(Path(value).resolve().relative_to(artifact_root.resolve()))

	cells = [
		{
			'layout_id': r['layout_id'],
			'data_size': r['data_size'],
			'primary': r['candidate_mean_iou'],
			'secondary': r['candidate_macro_f1'],
			'metrics_path': relative(r['candidate_metrics_path']),
			'metrics_sha256': r['candidate_metrics_sha256'],
		}
		for r in summary['comparison']
		if r['candidate_id'] == CANDIDATE_IDS['mae']
	]
	payload = {
		'schema_version': 1,
		'candidate_id': CANDIDATE_IDS['mae'],
		'head_ks': [6, 8, 10],
		'distillation_weight': 0.2,
		'consistency_weight': 0.0,
		'cells': cells,
		'canonical_cells_sha256': canonical_cells_sha256('f3', cells),
		'source_summary': {
			'path': relative(source_summary),
			'sha256': file_sha256(source_summary),
		},
	}
	for name, prefix in (
		('checkpoint', 'checkpoint'),
		('embeddings', 'embeddings'),
		('valid_tokens', 'valid_tokens'),
		('embedding_metadata', 'embedding_metadata'),
	):
		payload[name] = {
			'path': relative(source[f'{prefix}_path']),
			'sha256': source[f'{prefix}_sha256'],
		}
	payload['resolved_training_config'] = {
		'path': relative(resolved_path),
		'sha256': file_sha256(resolved_path),
	}
	receipt = _seal(payload)
	verify_reuse_receipt(receipt, artifact_root)
	return receipt


def verify_reuse_receipt(
	receipt: dict[str, Any], artifact_root: Path
) -> list[dict[str, Any]]:
	"""Read-only hash audit of the selected source, summary, and 15 metrics files."""
	_validate_reuse(receipt)
	for name in (
		'checkpoint',
		'embeddings',
		'valid_tokens',
		'embedding_metadata',
		'source_summary',
		'resolved_training_config',
	):
		identity = receipt[name]
		if (
			file_sha256(artifact_path(identity['path'], artifact_root))
			!= identity['sha256']
		):
			raise ValueError(f'F3 reuse {name} hash drift')
	for cell in receipt['cells']:
		path = artifact_path(cell['metrics_path'], artifact_root)
		if file_sha256(path) != cell['metrics_sha256']:
			raise ValueError('F3 reuse cell metrics hash drift')
		metrics = read_json(path)
		if (
			metrics['mean_iou'] != cell['primary']
			or metrics['macro_f1'] != cell['secondary']
		):
			raise ValueError('F3 reuse metrics differ from receipt')
	return receipt['cells']


def main() -> None:
	"""Build receipts only on explicit request, or validate existing files read-only."""
	parser = argparse.ArgumentParser(description=__doc__)
	mode = parser.add_mutually_exclusive_group(required=True)
	mode.add_argument('--build-receipts', action='store_true')
	mode.add_argument('--check', action='store_true')
	parser.add_argument('--matrix', type=Path, required=True)
	parser.add_argument('--controls', type=Path, required=True)
	parser.add_argument('--reuse', type=Path, required=True)
	parser.add_argument('--freeze-root', type=Path)
	parser.add_argument('--source-summary', type=Path)
	parser.add_argument('--artifact-root', type=Path)
	args = parser.parse_args()
	load_matrix(args.matrix)
	if args.build_receipts:
		if not all((args.freeze_root, args.source_summary, args.artifact_root)):
			parser.error(
				'build requires --freeze-root, --source-summary and --artifact-root'
			)
		if args.controls.exists() or args.reuse.exists():
			raise FileExistsError('receipt output already exists')
		controls = build_control_receipts(args.freeze_root)
		reuse = build_reuse_receipt(args.source_summary, args.artifact_root)
		for path, value in ((args.controls, controls), (args.reuse, reuse)):
			with path.open('x', encoding='utf-8') as stream:
				stream.write(
					json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
				)
	else:
		load_control_receipts(args.controls)
		receipt = load_reuse_receipt(args.reuse)
		if args.artifact_root:
			verify_reuse_receipt(receipt, args.artifact_root)
	print('K6810 matrix and receipts validated')


if __name__ == '__main__':
	main()
