"""Explicit common-test evaluation of frozen F3 decoder predictions only."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.io.labels import F3ClassInfo
from seis_ssl_cluster.f3.lithology.metrics import (
	lithology_metrics_from_confusion_matrix,
)

if TYPE_CHECKING:
	from _hashlib import HASH
	from collections.abc import Mapping, Sequence

NAMESPACE = 'lithology_common_test_v1'
V3_NAMESPACE = 'lithology_common_test_v3_v1'
DISTILLATION_NAMESPACE = 'lithology_common_test_distillation_sensitivity_v1'
RANDOM_DISTILLATION_NAMESPACE = 'lithology_common_test_random_distillation_v1'
ZERO_CONSTRAINT_NAMESPACE = 'lithology_common_test_v2_dis0_con0_v1'
V1_ZERO_DISTILLATION_NAMESPACE = 'lithology_common_test_v1_zero_distillation_v1'
BARLOW_ZERO_DISTILLATION_NAMESPACE = 'lithology_common_test_barlow_zero_distillation_v1'
CONDITION_KEYS = (
	'version',
	'source',
	'head_ks',
	'distillation_weight',
	'consistency_weight',
)
V3_KEYS = ('mask_token_lr', 'masked_prototype_weight', 'visible_prototype_weight')
LAYOUTS = tuple(f'layout_{i:03d}' for i in range(5))
SIZES = ('small', 'medium', 'large')
SUMMARY_FILES = frozenset(
	{
		'comparison.csv',
		'per_layout.csv',
		'class_metrics.csv',
		'confusion_matrices.csv',
		'summary.json',
		'summary.md',
	}
)
CELL_FILES = frozenset({'metrics.json', 'evidence.json'})


@dataclass(frozen=True)
class TestCell:
	"""One immutable selected decoder and its existing full-volume prediction."""

	condition: dict[str, str]
	layout: str
	size: str
	root: Path
	predictions: Path
	valid_mask: Path
	evidence: dict[str, Any]


@dataclass(frozen=True)
class TestPlan:
	"""Audited, pre-score selection receipt and fixed test inputs."""

	config: dict[str, Any]
	classes: tuple[F3ClassInfo, ...]
	cells: tuple[TestCell, ...]
	supervision_grids: tuple[Path, ...]
	receipt: dict[str, Any]
	identity: str


def _json(path: Path) -> dict[str, Any]:
	with path.open(encoding='utf-8') as stream:
		value = json.load(stream)
	if not isinstance(value, dict):
		raise TypeError(f'expected JSON mapping: {path}')
	return value


def _digest(value: object) -> str:
	return hashlib.sha256(
		json.dumps(value, sort_keys=True, allow_nan=False).encode()
	).hexdigest()


def _bound_file(record: Mapping[str, Any]) -> Path:
	path = Path(record['path'])
	if not path.is_file() or file_sha256(path) != record['sha256']:
		raise ValueError(f'input hash drift: {path}')
	return path


def _artifact_path(root: Path, relative: str) -> Path:
	path = root / relative
	if Path(relative).is_absolute() or not path.resolve().is_relative_to(
		root.resolve()
	):
		raise ValueError(f'expected artifact-relative path: {relative}')
	return path


def _expected_conditions(namespace: str = NAMESPACE) -> set[tuple[str, ...]]:  # noqa: C901, PLR0911, PLR0912
	if namespace == V1_ZERO_DISTILLATION_NAMESPACE:
		return {
			('v1', source, '6', '0.0', '0') for source in ('mae', 'local_bt', 'random')
		}
	if namespace == BARLOW_ZERO_DISTILLATION_NAMESPACE:
		return {
			('v2', 'local_bt', '6|8|10', weight, '0.1') for weight in ('0.0', '0.2')
		}
	if namespace == ZERO_CONSTRAINT_NAMESPACE:
		return {
			('v2', source, '6|8|10', '0.0', '0')
			for source in ('mae', 'local_bt', 'random')
		}
	if namespace == RANDOM_DISTILLATION_NAMESPACE:
		return {('v2', 'random', '6|8|10', weight, '0.1') for weight in ('0.0', '0.2')}
	if namespace == DISTILLATION_NAMESPACE:
		return {
			('v2', 'mae', '6|8|10', weight, '0.1')
			for weight in ('0.0', '0.1', '0.2', '0.4')
		}
	if namespace == V3_NAMESPACE:
		conditions = {
			('v3', source, '6|8|10', '0.2', '0.1', lr, '0.5', '0.5')
			for source in ('mae', 'local_bt', 'random')
			for lr in ('1e-5', '3e-5')
		}
		conditions.add(('v3', 'mae', '6|8|10', '0.2', '0.1', '1e-4', '0.5', '0.5'))
		for masked, visible in (('0.25', '0.75'), ('0.75', '0.25')):
			conditions.add(
				('v3', 'mae', '6|8|10', '0.2', '0.1', '3e-5', masked, visible)
			)
		return conditions
	if namespace != NAMESPACE:
		raise ValueError('unsupported test experiment')
	conditions = set()
	for source in ('mae', 'local_bt', 'random'):
		conditions.add(('none', source, '', '', ''))
		for dist in ('0.1', '0.2'):
			conditions.add(('v1', source, '6', dist, '0'))
		for heads in ('4|6|8', '6|8|10'):
			for cons in ('0', '0.1'):
				conditions.add(('v2', source, heads, '0.2', cons))
	for heads in ('4|6', '6|8'):
		conditions.add(('v2', 'mae', heads, '0.2', '0'))
	return conditions


def _scope(config: Mapping[str, Any]) -> tuple[tuple[str, ...], set[tuple[str, ...]]]:
	namespace = config['experiment_id']
	expected = _expected_conditions(namespace)
	if config['expected_conditions'] != len(expected) or config[
		'expected_cells'
	] != len(expected) * len(LAYOUTS) * len(SIZES):
		raise ValueError('fixed existing-condition scope/count mismatch')
	keys = CONDITION_KEYS + (V3_KEYS if namespace == V3_NAMESPACE else ())
	if (
		namespace
		in (
			V3_NAMESPACE,
			DISTILLATION_NAMESPACE,
			RANDOM_DISTILLATION_NAMESPACE,
			ZERO_CONSTRAINT_NAMESPACE,
			V1_ZERO_DISTILLATION_NAMESPACE,
			BARLOW_ZERO_DISTILLATION_NAMESPACE,
		)
		and 'reference_test_mask' not in config['inputs']
	):
		raise ValueError(
			'test extension requires the previously agreed reference test mask'
		)
	return keys, expected


def _condition_keys(condition: Mapping[str, Any]) -> tuple[str, ...]:
	return CONDITION_KEYS + (V3_KEYS if condition['version'] == 'v3' else ())


def _inspect_v3_parameters(condition: Mapping[str, str], workspace: Path) -> None:
	path = _artifact_path(workspace, condition['pretraining_config_workspace_relative'])
	_bound_file({'path': str(path), 'sha256': condition['pretraining_config_sha256']})
	config = load_config(path)
	context, train = config['spatial_context'], config['train']
	if (
		config['head']['ks'] != [6, 8, 10]
		or config['loss']['distillation_weight'] != 0.2
		or config['loss']['consistency_weight'] != 0.1
		or context['column_fraction'] != 0.1
		or train['epochs'] != 25
		or train['seed'] != 42
		or float(train['lr']) != 1e-5
		or float(train['encoder_lr']) != 1e-5
		or float(train.get('mask_token_lr', train['lr']))
		!= float(condition['mask_token_lr'])
		or any(
			context[k] != float(condition[k])
			for k in ('masked_prototype_weight', 'visible_prototype_weight')
		)
	):
		raise ValueError(
			'v3 inventory parameters differ from the pinned training recipe'
		)


def inspect_test_plan(config_path: Path) -> TestPlan:  # noqa: C901, PLR0912
	"""Read-only audit of every source before any test score is computed."""
	config = load_config(config_path)
	keys, expected_conditions = _scope(config)
	if config['mask_policy'] != 'reference_valid_and_canonical_split_zero_v1':
		raise ValueError('unsupported test experiment or mask policy')
	if config['class_ids'] != list(range(6)) or config['summary_ddof'] != 1:
		raise ValueError('fixed six-class metrics and sample SD are required')
	artifact = Path(config['artifact_root'])
	output = Path(config['output_root'])
	expected = (
		artifact / 'surveys/f3/facies_benchmark_v2/evaluation' / config['experiment_id']
	)
	if output.resolve() != expected.resolve():
		raise ValueError('output must use the separate common-test namespace')
	for name, record in config['inputs'].items():
		_bound_file(record)
		if 'reports' in Path(record['path']).resolve().parts:
			raise ValueError(f'report is not a pipeline input: {name}')
	inventory = Path(config['inputs']['inventory']['path'])
	with inventory.open(newline='', encoding='utf-8') as stream:
		conditions = list(csv.DictReader(stream))
	if (
		len(conditions) != len(expected_conditions)
		or {tuple(row[k] for k in keys) for row in conditions} != expected_conditions
	):
		raise ValueError(
			'inventory must contain exactly the fixed existing F3 conditions'
		)
	if len({row['audit_id'] for row in conditions}) != len(conditions):
		raise ValueError('duplicate condition identity')
	class_data = _json(Path(config['inputs']['class_info']['path']))['classes']
	classes = tuple(
		F3ClassInfo(c['class_id'], c['class_name'], tuple(c['rgb'])) for c in class_data
	)
	if [c.class_id for c in classes] != config['class_ids']:
		raise ValueError('class-info order differs from fixed test policy')
	cells = []
	grids: dict[str, str] = {}
	for condition in conditions:
		if condition['version'] == 'v3':
			_inspect_v3_parameters(condition, Path(config['workspace_root']))
		cp = _artifact_path(artifact, condition['checkpoint_artifact_relative'])
		_bound_file({'path': str(cp), 'sha256': condition['checkpoint_sha256']})
		embedding = _artifact_path(artifact, condition['embeddings_artifact_relative'])
		emeta = next(embedding.glob('*.embedding_metadata.json'))
		_bound_file(
			{'path': str(emeta), 'sha256': condition['embedding_metadata_sha256']}
		)
		if _json(emeta)['checkpoint_sha256'] != condition['checkpoint_sha256']:
			raise ValueError('encoder and embedding source differ')
		root = _artifact_path(artifact, condition['historical_runs_artifact_relative'])
		for layout in LAYOUTS:
			for size in SIZES:
				run = root / f'layout={layout}/size={size}'
				cell = inspect_test_cell(run, condition, layout, size)
				grid = cell.evidence['supervision_grid']
				grids[grid['path']] = grid['sha256']
				cells.append(cell)
	for path, sha in grids.items():
		_bound_file({'path': path, 'sha256': sha})
	if len(grids) != 15:
		raise ValueError('expected one shared supervision grid per layout/budget')
	receipt = {
		'config_sha256': file_sha256(config_path),
		'config': config,
		'checkpoint_selection': (
			'validation.macro_f1 / mean_iou / weighted_CE; never test'
		),
		'claim': 'same-survey transductive label holdout, not spatial holdout',
		'cells': [cell.evidence for cell in cells],
	}
	return TestPlan(
		config,
		classes,
		tuple(cells),
		tuple(Path(p) for p in grids),
		receipt,
		_digest(receipt),
	)


def inspect_test_cell(  # noqa: C901
	run: Path,
	condition: dict[str, str],
	layout: str,
	size: str,
	*,
	decoder_seed: int = 42000,
) -> TestCell:
	"""Audit an explicitly seeded validation-selected decoder before test scoring."""
	metadata_path = run / 'prediction/prediction_metadata.json'
	metadata = _json(metadata_path)
	identity = metadata['source_identity']
	for name in ('decoder_checkpoint', 'resolved_decoder_config'):
		_bound_file(identity[name])
	if (
		Path(identity['decoder_checkpoint']['path']).resolve()
		!= (run / 'decoder/best.pt').resolve()
	):
		raise ValueError('prediction does not use its own selected best checkpoint')
	config = _json(Path(identity['resolved_decoder_config']['path']))
	emeta_path = _bound_file(
		{
			'path': metadata['inputs']['embedding_metadata'],
			'sha256': condition['embedding_metadata_sha256'],
		}
	)
	emeta = _json(emeta_path)
	if (
		emeta['checkpoint_sha256'] != condition['checkpoint_sha256']
		or Path(config['embeddings']['input_dir']).resolve()
		!= emeta_path.parent.resolve()
		or Path(config['embeddings']['checkpoint_path']).resolve()
		!= Path(emeta['checkpoint_path']).resolve()
	):
		raise ValueError('prediction source is not the inventoried encoder')
	if config['train']['epochs'] != 50 or config['train']['seed'] != decoder_seed:
		raise ValueError('decoder training contract changed')
	for split in ('train', 'validation'):
		_bound_file(identity['tile_manifests'][split])
	grid = identity['artifact_identities']['voxel_split_grid']
	with (run / 'decoder/history.csv').open(newline='', encoding='utf-8') as stream:
		history = list(csv.DictReader(stream))
	if len(history) != 50 or int(history[-1]['global_step']) != 22000:
		raise ValueError('incomplete frozen decoder history')
	evaluation_path = run / 'evaluation/evaluation_metadata.json'
	evaluation = _json(evaluation_path)
	if evaluation['aggregation']['primary_unit'] != 'unique_validation_voxel':
		raise ValueError('historical checkpoint must have validation-only evaluation')
	if (
		metadata['coverage']['missing_write_count'] != 0
		or not metadata['coverage']['exact_once']
	):
		raise ValueError('incomplete prediction coverage')
	pred = Path(metadata['outputs']['predictions'])
	valid = Path(metadata['outputs']['valid_mask'])
	for path in (pred, valid):
		if (
			path.parent.resolve() != metadata_path.parent.resolve()
			or not path.is_file()
		):
			raise ValueError('prediction output path mismatch')
	evidence = {
		'condition_id': condition['audit_id'],
		'layout_id': layout,
		'data_size': size,
		'decoder_checkpoint': identity['decoder_checkpoint'],
		'prediction_metadata': {
			'path': str(metadata_path),
			'sha256': file_sha256(metadata_path),
		},
		'evaluation_metadata': {
			'path': str(evaluation_path),
			'sha256': file_sha256(evaluation_path),
		},
		'predictions': {
			'path': str(pred),
			'sha256': evaluation['inputs']['voxel_predictions']['sha256'],
		},
		'valid_mask': {
			'path': str(valid),
			'sha256': evaluation['inputs']['voxel_valid_mask']['sha256'],
		},
		'supervision_grid': grid,
		'resolved_decoder_config': identity['resolved_decoder_config'],
		'tile_manifests': identity['tile_manifests'],
		'history_sha256': file_sha256(run / 'decoder/history.csv'),
		'condition': condition,
	}
	return TestCell(condition, layout, size, run, pred, valid, evidence)


def common_test_mask(reference_valid: np.ndarray, canonical: np.ndarray) -> np.ndarray:
	"""Exclude the complete canonical train pool and validation, not one layout."""
	if reference_valid.dtype != np.bool_ or canonical.shape != reference_valid.shape:
		raise ValueError('reference-valid mask must be bool and match canonical shape')
	if not np.isin(canonical, [0, 1, 2]).all():
		raise ValueError('unknown canonical split code')
	return reference_valid & (canonical == 0)


def masked_confusion(
	labels: np.ndarray, predictions: np.ndarray, test: np.ndarray, valid: np.ndarray
) -> np.ndarray:
	"""Fast six-class accumulation with explicit coverage and label failures."""
	if not (labels.shape == predictions.shape == test.shape == valid.shape):
		raise ValueError('test, labels, predictions and coverage shapes must agree')
	if test.dtype != np.bool_ or valid.dtype != np.bool_:
		raise ValueError('test and coverage masks must be boolean')
	if np.any(test & ~valid):
		raise ValueError('test voxel missing prediction coverage')
	true, pred = labels[test], predictions[test]
	for values in (true, pred):
		if (
			not np.issubdtype(values.dtype, np.integer)
			or np.any(values < 0)
			or np.any(values >= 6)
		):
			raise ValueError('unknown or non-integer class on fixed test mask')
	codes = true.astype(np.int16) * 6 + pred.astype(np.int16)
	return np.bincount(codes, minlength=36).reshape(6, 6).astype(np.int64)


def _write_json(path: Path, value: object) -> None:
	path.write_text(
		json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _atomic_json(path: Path, value: object) -> None:
	tmp = path.with_name('.' + path.name + '.tmp')
	_write_json(tmp, value)
	tmp.replace(path)


def _build_test_inputs(plan: TestPlan, root: Path) -> Path:
	inputs = plan.config['inputs']
	canonical = np.load(inputs['canonical_grid']['path'], mmap_mode='r')
	valid_tokens = np.load(inputs['reference_valid_tokens']['path'], mmap_mode='r')
	shape = tuple(plan.config['volume_shape_xyz'])
	if canonical.shape != shape or valid_tokens.dtype != np.bool_:
		raise ValueError('canonical/reference geometry mismatch')
	grids = [np.load(p, mmap_mode='r') for p in plan.supervision_grids]
	target = root / 'inputs'
	with tempfile.TemporaryDirectory(prefix='.inputs-', dir=root) as tmp:
		stage = Path(tmp)
		mask = np.lib.format.open_memmap(
			stage / 'test_mask.npy', mode='w+', dtype=bool, shape=shape
		)
		for start in range(0, shape[0], 8):
			stop = min(start + 8, shape[0])
			valid = np.repeat(
				np.repeat(
					np.repeat(valid_tokens[start // 8 : start // 8 + 1], 8, axis=0),
					8,
					axis=1,
				),
				8,
				axis=2,
			)[: stop - start, : shape[1], : shape[2]]
			chunk = common_test_mask(valid, canonical[start:stop])
			if any(np.any(chunk & (g[start:stop] != 0)) for g in grids):
				raise ValueError('test overlaps decoder train/validation labels')
			mask[start:stop] = chunk
		count = int(mask.sum())
		if count != plan.config['expected_test_voxels']:
			raise ValueError(f'test mask count drift: {count}')
		mask.flush()
		del mask
		if (
			'reference_test_mask' in inputs
			and file_sha256(stage / 'test_mask.npy')
			!= inputs['reference_test_mask']['sha256']
		):
			raise ValueError('test mask differs from the previously agreed reference')
		_write_json(
			stage / 'test_definition.json',
			{
				'run_identity': plan.identity,
				'split': 'test',
				'voxel_count': count,
				'mask_policy': plan.config['mask_policy'],
				'inputs': inputs,
				'mask_sha256': file_sha256(stage / 'test_mask.npy'),
				'train_validation_overlap': 0,
				'spatial_buffer': None,
			},
		)
		if target.exists():
			raise FileExistsError('refusing to replace test inputs')
		stage.rename(target)
	return target / 'test_mask.npy'


def _array_digest(path: Path, array: np.memmap) -> HASH:
	digest = hashlib.sha256()
	with path.open('rb') as stream:
		digest.update(stream.read(array.offset))
	return digest


def evaluate_cell(
	plan: TestPlan, cell: TestCell, mask_path: Path, destination: Path
) -> dict[str, Any]:
	"""Evaluate one preselected checkpoint; never train, select, or mask failures."""
	if destination.exists():
		raise FileExistsError(f'refusing to overwrite test cell: {destination}')
	inputs = plan.config['inputs']
	labels = np.load(inputs['labels']['path'], mmap_mode='r')
	mask = np.load(mask_path, mmap_mode='r')
	pred = np.load(cell.predictions, mmap_mode='r')
	valid = np.load(cell.valid_mask, mmap_mode='r')
	if labels.shape != mask.shape or pred.dtype != np.int16 or valid.dtype != np.bool_:
		raise ValueError('prediction/label dtype or shape mismatch')
	pred_hash, valid_hash = (
		_array_digest(cell.predictions, pred),
		_array_digest(cell.valid_mask, valid),
	)
	matrix = np.zeros((6, 6), dtype=np.int64)
	for start in range(0, mask.shape[0], plan.config['chunk_size_x']):
		slab = slice(start, start + plan.config['chunk_size_x'])
		pred_hash.update(memoryview(pred[slab]).cast('B'))
		valid_hash.update(memoryview(valid[slab]).cast('B'))
		matrix += masked_confusion(labels[slab], pred[slab], mask[slab], valid[slab])
	for name, digest in (('predictions', pred_hash), ('valid_mask', valid_hash)):
		if digest.hexdigest() != cell.evidence[name]['sha256']:
			raise ValueError(f'{name} hash drift while evaluating fixed test')
	_bound_file(cell.evidence['decoder_checkpoint'])
	if int(matrix.sum()) != plan.config['expected_test_voxels']:
		raise ValueError('test count mismatch')
	metrics = lithology_metrics_from_confusion_matrix(matrix, plan.classes)
	metrics.update(
		split='test',
		aggregation_unit='unique_test_voxel',
		evaluation_voxel_count=int(matrix.sum()),
	)
	evidence = dict(
		cell.evidence,
		run_identity=plan.identity,
		test_mask_sha256=file_sha256(mask_path),
	)
	destination.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(prefix='.cell-', dir=destination.parent) as tmp:
		stage = Path(tmp)
		_write_json(stage / 'metrics.json', metrics)
		_write_json(stage / 'evidence.json', evidence)
		stage.rename(destination)
	return metrics


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Compute mean and sample SD for exactly five unique layouts per group."""
	groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		groups[row['condition_id'], row['data_size']].append(row)
	result = []
	for (_condition, size), values in groups.items():
		if len(values) != 5 or {v['layout_id'] for v in values} != set(LAYOUTS):
			raise ValueError('summary requires exactly five distinct layouts')
		entry = {k: values[0][k] for k in ('condition_id', *_condition_keys(values[0]))}
		entry.update(data_size=size, layouts=5)
		for metric in ('mean_iou', 'macro_f1', 'accuracy', 'balanced_accuracy'):
			data = [float(v[metric]) for v in values]
			entry[metric + '_mean'] = statistics.mean(data)
			entry[metric + '_sample_sd'] = statistics.stdev(data)
		result.append(entry)
	return result


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
	with path.open('w', newline='', encoding='utf-8') as stream:
		writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


def publish_summary(
	root: Path, rows: list[dict[str, Any]], metrics: list[dict[str, Any]], identity: str
) -> None:
	"""Publish exactly the six review files; existing results are immutable."""
	target = root / 'summary'
	if target.exists():
		raise FileExistsError('refusing to overwrite test summary')
	aggregates = aggregate_rows(rows)
	class_rows, confusion_rows = [], []
	for row, cell in zip(rows, metrics, strict=True):
		base = {k: row[k] for k in ('condition_id', 'layout_id', 'data_size')}
		for cls in range(6):
			class_rows.append(
				dict(
					base,
					class_id=cls,
					**{
						k: cell['per_class_' + k][str(cls)]
						for k in ('precision', 'recall', 'f1', 'iou', 'support')
					},
				)
			)
			confusion_rows.extend(
				dict(
					base,
					true_class_id=cls,
					predicted_class_id=pred,
					count=cell['confusion_matrix'][cls][pred],
				)
				for pred in range(6)
			)
	with tempfile.TemporaryDirectory(prefix='.summary-', dir=root) as tmp:
		stage = Path(tmp)
		for name, data in [
			('comparison.csv', aggregates),
			('per_layout.csv', rows),
			('class_metrics.csv', class_rows),
			('confusion_matrices.csv', confusion_rows),
		]:
			_write_csv(stage / name, data)
		_write_json(
			stage / 'summary.json',
			{
				'status': 'complete',
				'run_identity': identity,
				'split': 'test',
				'cell_count': len(rows),
				'condition_count': len({r['condition_id'] for r in rows}),
				'aggregation': (
					'arithmetic mean and sample SD (ddof=1) across five layouts'
				),
				'claim': 'same-survey transductive label holdout; no spatial buffer',
				'rows': aggregates,
				'published_files': sorted(SUMMARY_FILES),
			},
		)
		lines = [
			'# F3 common-test evaluation',
			'',
			'Mean ± sample SD across five layouts. Each cell: mean IoU / macro-F1.',
			'',
			'| Condition | Small | Medium | Large |',
			'|---|---|---|---|',
		]
		by_condition: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
		for row in aggregates:
			by_condition[row['condition_id']][row['data_size']] = row
		for condition, sizes in by_condition.items():
			values = []
			for size in SIZES:
				r = sizes[size]
				values.append(
					f'{r["mean_iou_mean"]:.4f} ± {r["mean_iou_sample_sd"]:.4f} / '
					f'{r["macro_f1_mean"]:.4f} ± {r["macro_f1_sample_sd"]:.4f}'
				)
			lines.append('| ' + ' | '.join([condition, *values]) + ' |')
		(stage / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
		if {p.name for p in stage.iterdir()} != SUMMARY_FILES:
			raise ValueError('unexpected summary publication set')
		stage.rename(target)


def load_cached_test_cell(
	plan: TestPlan, cell: TestCell, output: Path, mask_path: Path
) -> dict[str, Any]:
	"""Verify a completed test cell against its frozen identities and confusion."""
	if {p.name for p in output.iterdir()} != CELL_FILES:
		raise ValueError('incomplete cached test cell')
	expected = dict(
		cell.evidence,
		run_identity=plan.identity,
		test_mask_sha256=file_sha256(mask_path),
	)
	if _json(output / 'evidence.json') != expected:
		raise ValueError('cached test cell identity drift')
	metrics = _json(output / 'metrics.json')
	matrix = np.asarray(metrics['confusion_matrix'], dtype=np.int64)
	if int(matrix.sum()) != plan.config['expected_test_voxels']:
		raise ValueError('cached test count mismatch')
	recomputed = lithology_metrics_from_confusion_matrix(matrix, plan.classes)
	recomputed.update(
		split='test',
		aggregation_unit='unique_test_voxel',
		evaluation_voxel_count=int(matrix.sum()),
	)
	if metrics != recomputed:
		raise ValueError('cached metrics disagree with confusion matrix')
	for name in ('predictions', 'valid_mask'):
		_bound_file(cell.evidence[name])
	return metrics


def run_test(plan: TestPlan, *, resume: bool = False) -> Path:
	"""Freeze identities, evaluate serially on CPU, and preserve all old outputs."""
	root = Path(plan.config['output_root'])
	if root.exists() and not resume:
		raise FileExistsError(
			'output exists; only explicit identity-checked --resume is allowed'
		)
	root.mkdir(parents=True, exist_ok=True)
	with (root / '.run.lock').open('a') as lock:
		fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
		receipt_path = root / 'selection_receipt.json'
		if receipt_path.exists():
			if _json(receipt_path)['run_identity'] != plan.identity:
				raise ValueError('resume selection/input identity drift')
		else:
			_write_json(
				receipt_path,
				dict(
					plan.receipt,
					run_identity=plan.identity,
					frozen_at_utc=datetime.now(timezone.utc).isoformat(),
				),
			)
		mask_path = root / 'inputs/test_mask.npy'
		if mask_path.exists():
			definition = _json(mask_path.parent / 'test_definition.json')
			if (
				definition['run_identity'] != plan.identity
				or file_sha256(mask_path) != definition['mask_sha256']
			):
				raise ValueError('resume test-mask drift')
		else:
			mask_path = _build_test_inputs(plan, root)
		rows, results = [], []
		try:
			for index, cell in enumerate(plan.cells, 1):
				output = (
					root
					/ 'cells'
					/ cell.condition['audit_id']
					/ cell.layout
					/ cell.size
				)
				if output.exists():
					metrics = load_cached_test_cell(plan, cell, output, mask_path)
				else:
					metrics = evaluate_cell(plan, cell, mask_path, output)
				row = {k: cell.condition[k] for k in _condition_keys(cell.condition)}
				row.update(
					condition_id=cell.condition['audit_id'],
					layout_id=cell.layout,
					data_size=cell.size,
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
				results.append(metrics)
				_atomic_json(
					root / 'status.json',
					{
						'status': 'running',
						'completed_cells': index,
						'total_cells': len(plan.cells),
						'run_identity': plan.identity,
					},
				)
				print(
					json.dumps(
						{
							'completed': index,
							'total': len(plan.cells),
							'cell': str(output.relative_to(root)),
						}
					),
					flush=True,
				)
			if not (root / 'summary').exists():
				publish_summary(root, rows, results, plan.identity)
			_atomic_json(
				root / 'status.json',
				{
					'status': 'complete',
					'completed_cells': len(rows),
					'total_cells': len(plan.cells),
					'run_identity': plan.identity,
				},
			)
		except BaseException:
			_atomic_json(
				root / 'status.json',
				{
					'status': 'failed',
					'completed_cells': len(rows),
					'total_cells': len(plan.cells),
					'run_identity': plan.identity,
				},
			)
			raise
	return root / 'summary'


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI without reading artifacts or starting execution."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', type=Path, required=True)
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument('--dry-run', action='store_true')
	mode.add_argument('--execute', action='store_true')
	parser.add_argument('--resume', action='store_true')
	return parser


def main(argv: Sequence[str] | None = None) -> None:
	"""Default to a read-only preflight; scoring requires --execute."""
	parser = build_parser()
	args = parser.parse_args(argv)
	if args.resume and not args.execute:
		parser.error('--resume requires --execute')
	plan = inspect_test_plan(args.config)
	print(
		json.dumps(
			{
				'mode': 'execute' if args.execute else 'dry_run',
				'cells': len(plan.cells),
				'conditions': plan.config['expected_conditions'],
				'run_identity': plan.identity,
				'test_voxels': plan.config['expected_test_voxels'],
				'output_root': plan.config['output_root'],
			}
		),
		flush=True,
	)
	if args.execute:
		run_test(plan, resume=args.resume)


if __name__ == '__main__':
	main()
