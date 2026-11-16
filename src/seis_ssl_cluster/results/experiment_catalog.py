"""Export all retained F3, Parihaka and Volve evaluations without running models."""

# ruff: noqa: C901, PLR0912, PLR0915
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SURVEYS = ('f3', 'parihaka', 'volve')
FIXED_FILES = frozenset(
	{
		'experiment_metrics.csv',
		'metrics_long.csv',
		'layout_summary.csv',
		'published_statistics.csv',
		'experiment_catalog.csv',
		'source_files.csv',
		'metric_dictionary.csv',
		'index.html',
		'validation.json',
		'manifest.json',
	}
)
CONTEXT_FIELDS = (
	'record_id',
	'survey',
	'experiment',
	'model',
	'initialization',
	'protocol',
	'hmm_version',
	'head_type',
	'head_ks',
	'distillation_weight',
	'consistency_weight',
	'encoder_seed',
	'decoder_seed',
	'data_version',
	'label_budget',
	'replicate_id',
	'replicate_kind',
	'split',
	'support',
	'checkpoint_selection',
	'evaluation_variant',
	'evaluation_kind',
	'status',
	'archived',
	'duplicate_of',
	'source_cohort',
	'definition',
	'condition_config',
	'encoder_checkpoint',
	'source_file',
	'source_id',
	'source_sha256',
	'best_epoch',
	'condition_inference',
)
MAIN_METRICS = (
	'mean_iou',
	'macro_f1',
	'accuracy',
	'balanced_accuracy',
	'weighted_f1',
	'channel_iou',
	'channel_f1',
	'channel_precision',
	'channel_recall',
	'macro_mae_samples',
	'macro_within_2_samples',
	'macro.within_1',
	'macro.within_4',
)
SKIP_KEYS = frozenset(
	{
		'evidence',
		'inputs',
		'definitions',
		'matrix',
		'config',
		'resolved_config',
		'benchmark_identity',
		'source_summaries',
		'control_receipt',
		'selection_receipt',
		'input_identities',
		'recorded_evidence',
		'run_identity',
		'source',
		'sources',
		'feature_source',
		'metadata',
		'coordinates',
		'outputs',
		'training',
		'train',
		'normalization',
		'preprocessing',
		'decoder',
		'optimizer',
		'model_geometry',
		'class_ids',
		'class_names',
		'class_weights',
		'selected_inline_indices',
		'selected_crossline_indices',
		'supervision',
		'tile_counts',
		'rows_missing',
		'cells',
		'feature_diagnostics',
		'diagnostics',
		'validation_inputs',
	}
)
ID_KEYS = frozenset(
	{
		'schema_version',
		'epoch',
		'global_step',
		'best_epoch',
		'class_id',
		'horizon_id',
		'layout_id',
		'data_size',
		'distillation_weight',
		'consistency_weight',
		'seed',
		'array_index',
		'slice_index',
		'sample_index',
		'x',
		'y',
		'z',
		'order',
	}
)
METRIC_TOKEN = re.compile(
	r'iou|f1|accuracy|precision|recall|mae|error|within|coverage|boundary|support|'
	r'confusion|loss|entropy|distance|violation|primary|secondary|improvement|delta|'
	r'count|mean|median|std|stdev|sample_sd|population_sd|fraction|wins|ties|losses|metric|score|rmse|auc|dice',
	re.IGNORECASE,
)


def sha256(path: Path) -> str:
	"""Hash a source without opening any model or prediction array."""
	h = hashlib.sha256()
	with path.open('rb') as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b''):
			h.update(chunk)
	return h.hexdigest()


def portable(value: str, workspace: Path) -> str:
	"""Keep repository identities portable in the exported files."""
	return value.replace(str(workspace) + '/', '')


def flatten_metrics(value: object, prefix: str = '') -> dict[str, float | None]:
	"""Preserve scalar, class, horizon and matrix values, excluding provenance."""
	out: dict[str, float | None] = {}
	if isinstance(value, dict):
		for key, child in value.items():
			if key in SKIP_KEYS or key in ID_KEYS or key.endswith('_sha256'):
				continue
			name = f'{prefix}.{key}' if prefix else str(key)
			out.update(flatten_metrics(child, name))
	elif isinstance(value, list):
		for index, child in enumerate(value):
			out.update(flatten_metrics(child, f'{prefix}[{index}]'))
	elif (
		not isinstance(value, bool)
		and (isinstance(value, (float, int)) or value is None)
		and METRIC_TOKEN.search(prefix)
	):
		out[prefix] = (
			float(value) if value is not None and math.isfinite(value) else None
		)
	return out


def metric_info(name: str) -> tuple[str, str]:
	"""Use explicit units; never mix samples, milliseconds or IoU fractions."""
	low = name.lower()
	if (
		'count' in low
		or 'support' in low
		or ('confusion_matrix' in low and 'normalized' not in low)
	):
		return 'count', 'descriptive'
	if 'row_normalized' in low or any(
		x in low
		for x in (
			'iou',
			'accuracy',
			'precision',
			'recall',
			'f1',
			'within',
			'fraction',
			'rate',
		)
	):
		return 'fraction', 'lower' if any(
			x in low for x in ('miss_rate', 'violation')
		) else 'higher'
	if '_ms' in low:
		return 'ms', 'lower'
	if any(x in low for x in ('mae', 'error', 'median_ae')):
		return (
			'samples'
			if 'sample' in low or 'boundary_position' in low
			else 'source-defined',
			'lower',
		)
	if 'loss' in low:
		return 'source-defined', 'lower'
	return 'source-defined', 'descriptive'


def _json(path: Path) -> dict[str, Any]:
	value = json.loads(path.read_text())
	return value if isinstance(value, dict) else {'items': value}


def _experiment_root(path: Path, workspace: Path) -> str:
	parts = path.relative_to(workspace).parts
	count = 4 if len(parts) > 3 and parts[1] in SURVEYS else 3
	return '/'.join(parts[:count])


class DefinitionIndex:
	"""Map source output directories and checkpoints back to versioned recipes."""

	def __init__(self, workspace: Path, artifact_root: Path) -> None:
		"""Index versioned output owners and training checkpoint definitions."""
		self.workspace = workspace
		self.artifact_root = artifact_root
		self.prefixes: list[tuple[str, str, str, dict[str, Any]]] = []
		self.catalog: dict[str, dict[str, Any]] = {}
		self.checkpoints: dict[str, tuple[str, dict[str, Any]]] = {}
		for path in sorted((workspace / 'experiments').rglob('*.yaml')):
			try:
				raw = yaml.load(
					path.read_text(),
					Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader),  # noqa: S506
				)
			except (ValueError, yaml.YAMLError, OSError):
				continue
			if not isinstance(raw, dict):
				continue
			root = _experiment_root(path, workspace)
			relative = str(path.relative_to(workspace))
			if any(x in path.parts for x in SURVEYS):
				survey = next(x for x in SURVEYS if x in path.parts)
				entry = self.catalog.setdefault(
					root, {'survey': survey, 'definition': root, 'config_count': 0}
				)
				entry['config_count'] += 1
			for section, keys in (
				('outputs', ('runs_root', 'output_dir', 'summary_root', 'output_root')),
				('paths', ('output_root',)),
				('embeddings', ('output_dir',)),
				('clustering', ('output_dir',)),
			):
				mapping = raw.get(section, {})
				if not isinstance(mapping, dict):
					continue
				for key in keys:
					value = mapping.get(key)
					if not isinstance(value, str):
						continue
					value = value.replace(
						'${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}', str(artifact_root)
					).replace('${SEIS_SSL_CLUSTER_WORKSPACE}', str(workspace))
					if '${' in value or not Path(value).is_absolute():
						continue
					resolved = str(Path(value).resolve())
					self.prefixes.append((resolved, root, relative, raw))
					if section == 'paths' and key == 'output_root':
						for checkpoint in ('latest.pt', 'best.pt'):
							self.checkpoints[
								str((Path(resolved) / checkpoint).resolve())
							] = (relative, raw)
		self.prefixes.sort(key=lambda x: len(x[0]), reverse=True)

	def match(self, path: Path) -> tuple[str, str]:
		"""Choose the longest owned output prefix, never an input reference."""
		value = str(path.resolve())
		for prefix, experiment, definition, _ in self.prefixes:
			if value == prefix or value.startswith(prefix + '/'):
				return experiment, definition
		return '', ''

	def condition(self, checkpoint: str) -> tuple[str, dict[str, Any]]:
		"""Prefer the recorded training config over a versioned definition."""
		if not checkpoint:
			return '', {}
		path = Path(checkpoint)
		if not path.is_absolute():
			path = self.workspace / path
		resolved = path.resolve()
		actual = resolved.parent / 'resolved_config.json'
		if actual.is_file() and actual.stat().st_size < 1024 * 1024:
			try:
				return portable(str(actual), self.workspace), _json(actual)
			except (ValueError, OSError):
				pass
		return self.checkpoints.get(str(resolved), ('', {}))


def _path_fields(path: Path) -> dict[str, str]:
	parts = path.parts
	fields = {
		key: next((p.split('=', 1)[1] for p in parts if p.startswith(key + '=')), '')
		for key in (
			'model',
			'layout',
			'size',
			'budget',
			'subsample_seed',
			'split_index',
			'encoder_init',
		)
	}
	if not fields['layout']:
		fields['layout'] = next(
			(p for p in parts if re.fullmatch(r'layout_\d+', p)), ''
		)
	if not fields['size']:
		fields['size'] = next(
			(p for p in reversed(parts) if p in ('small', 'medium', 'large')), ''
		)
	fields['budget'] = fields['budget'] or next(
		(p for p in parts if re.fullmatch(r'cap\d+|full', p)), ''
	)
	fields['split_index'] = fields['split_index'] or next(
		(p for p in parts if re.fullmatch(r'split_\d+', p)), ''
	)
	return fields


def _safe_sidecar(path: Path) -> dict[str, Any]:
	if path.is_file() and path.stat().st_size < 1024 * 1024:
		try:
			return _json(path)
		except (ValueError, OSError):
			pass
	return {}


def _family(text: str) -> str:
	low = text.lower()
	for tokens, name in [
		(('vicreg',), 'VICReg'),
		(('barlow', 'local_bt', 'localbt'), 'Barlow Twins'),
		(('random',), 'Random'),
		(('xyz_coordinates',), 'XYZ coordinates'),
		(('raw_patch', 'raw_amplitude'), 'Raw amplitude'),
		(('mae',), 'MAE'),
	]:
		if any(token in low for token in tokens):
			return name
	return 'unknown'


def _infer_tokens(text: str) -> dict[str, Any]:
	"""Expose name-derived settings as inferred rather than inventing defaults."""
	out: dict[str, Any] = {}
	for pattern, key, scale in [
		(r'distill(\d{3})', 'distillation_weight', 100),
		(r'cons(\d{3})', 'consistency_weight', 100),
	]:
		match = re.search(pattern, text)
		if match:
			out[key] = int(match[1]) / scale
	if 'k6810' in text:
		out['head_ks'] = '6|8|10'
	elif 'k468' in text:
		out['head_ks'] = '4|6|8'
	elif re.search(r'[_/]k6(?:[_/]|$)', text):
		out['head_ks'] = '6'
	return out


def _initialization(
	model: str,
	embedding: dict[str, Any],
	checkpoint: str | None,
	actual: dict[str, Any],
) -> str:
	"""Prefer an explicit model name over shared parent directory names."""
	family = _family(str(model))
	if family == 'unknown':
		family = _family(
			str(embedding.get('model_source', {})) + ' ' + str(checkpoint or '')
		)
	if family == 'unknown':
		family = _family(str(actual.get('student', {})))
	return family


def context(
	path: Path, payload: dict[str, Any], index: DefinitionIndex
) -> dict[str, Any]:
	"""Recover experiment, split-independent identity and explicit loss settings."""
	workspace = index.workspace
	relative = portable(str(path), workspace)
	parts = path.parts
	survey = next((s for s in SURVEYS if s in parts), '')
	if not survey:
		survey = next(
			(
				s
				for s in SURVEYS
				if re.search(rf'(^|[/_]){s}([/_]|$)', relative, re.IGNORECASE)
			),
			'',
		)
	fields = _path_fields(path)
	evidence = _safe_sidecar(path.with_name('evidence.json'))
	evaluation_metadata = _safe_sidecar(path.with_name('evaluation_metadata.json'))
	condition = evidence.get('condition', {})
	identity = payload.get('benchmark_identity', {})
	embedding = identity.get('embedding', {})
	decoder = {}
	for candidate in (
		path.parent.parent / 'decoder/resolved_config.json',
		path.parent / 'resolved_config.json',
		path.parent / 'probe_config_resolved.json',
	):
		decoder = _safe_sidecar(candidate)
		if decoder:
			break
	if not decoder and isinstance(evidence.get('resolved_decoder_config'), dict):
		decoder = _safe_sidecar(
			Path(evidence['resolved_decoder_config'].get('path', '/nonexistent'))
		)
	model = (
		payload.get('model')
		or payload.get('model_tag')
		or fields['model']
		or fields['encoder_init']
		or evaluation_metadata.get('model_tag')
		or decoder.get('model', {}).get('tag')
	)
	if not isinstance(model, str):
		model = model.get('tag', '') if isinstance(model, dict) else ''
	model = (
		model
		or condition.get('historical_condition_id')
		or evidence.get('condition_id')
	)
	model = model or payload.get('feature_source', {}).get('reference_model_tag', '')
	if not model:
		for token in (
			'lithology',
			'lithology_benchmark',
			'channel_benchmark',
			'horizon',
		):
			if token in parts and parts.index(token) + 1 < len(parts):
				model = parts[parts.index(token) + 1]
				break
	model = model or path.parent.name
	checkpoint = embedding.get('checkpoint_path') or decoder.get('embeddings', {}).get(
		'checkpoint_path'
	)
	checkpoint = checkpoint or identity.get('encoder_source', {}).get('checkpoint_path')
	if not checkpoint:
		embedding_dir = decoder.get('embeddings', {}).get('input_dir', '')
		if embedding_dir:
			for metadata_path in sorted(
				Path(embedding_dir).glob('*.embedding_metadata.json')
			):
				checkpoint = _safe_sidecar(metadata_path).get('checkpoint_path')
				if checkpoint:
					break
	if not checkpoint and condition.get('checkpoint_artifact_relative'):
		checkpoint = str(
			index.artifact_root / condition['checkpoint_artifact_relative']
		)
	condition_path, actual = index.condition(str(checkpoint or ''))
	head = actual.get('head', {})
	loss = actual.get('loss', {})
	model_text = f'{model} {checkpoint or ""}'
	inferred = _infer_tokens(model_text)
	ks = head.get('ks') or (
		[head['num_prototypes']] if 'num_prototypes' in head else []
	)
	ks_text = (
		'|'.join(str(k) for k in ks)
		or str(condition.get('head_ks', ''))
		or inferred.get('head_ks', '')
	)
	protocol = 'none'
	if 'kmeans' in model_text.lower():
		protocol = 'k-means'
	elif (
		'hmm' in model.lower()
		or head
		or any(
			bool(embedding.get('model_source', {}).get(key))
			for key in ('stratigraphy_pretext', 'multi_head', 'hmm_k', 'pseudo_targets')
		)
	) and condition.get('version') != 'none':
		protocol = next(
			(
				f'HMM v{v}'
				for v in (3, 2, 1)
				if re.search(rf'hmm[_ ]v{v}(?:[_/ ]|$)', model_text, re.IGNORECASE)
			),
			'',
		)
		protocol = protocol or (
			'multi-head (legacy)'
			if len(ks) > 1 or 'mh_' in model_text
			else 'single-head K6 / legacy HMM'
		)
	# Early v1 model tags predate the literal "hmm_v1" spelling. The recorded
	# single-head training schema and evaluation manifest identify that version.
	if protocol not in ('none', 'k-means'):
		if condition.get('version') in ('v1', 'v2', 'v3'):
			protocol = f'HMM {condition["version"]}'
		elif (
			protocol == 'single-head K6 / legacy HMM'
			and actual.get('stage') == 'train_strat_hmm_pretext'
			and head.get('num_prototypes')
			and not head.get('ks')
		):
			protocol = 'HMM v1'
	experiment, definition = index.match(path)
	cohort_parts = list(parts)
	cut = next(
		(
			i
			for i, p in enumerate(parts)
			if p in ('runs', 'validation_runs', 'jobs', 'cells', 'probes')
		),
		len(parts) - 1,
	)
	cohort = portable(str(Path(*cohort_parts[:cut])), workspace)
	if not experiment:
		experiment = cohort
	replicate = (
		payload.get('layout_id') or evidence.get('layout_id') or fields['layout']
	)
	kind = 'layout' if replicate else ''
	for field, label in [
		('subsample_seed', 'label_subsample'),
		('split_index', 'split_index'),
	]:
		if not replicate and fields[field]:
			replicate, kind = f'{field}={fields[field]}', label
	training = identity.get('training', {}) or decoder.get('train', {})
	seed = training.get(
		'seed', identity.get('decoder', {}).get('initialization_seed', '')
	)
	variant = payload.get('checkpoint_selection', '')
	if path.name.startswith('metrics_'):
		variant = path.stem.removeprefix('metrics_')
	for token in ('voxel_evaluations', 'voxel_reports', 'probes'):
		if token in parts:
			variant = f'{token}/{path.parent.name}'
	archived = 'archive' in parts or any(
		x in relative.lower()
		for x in ('fp16_overflow', 'fp32_partial', 'historical_preflight')
	)
	smoke = any(
		x in relative.lower()
		for x in ('smoke', 'feasibility', '/fixture', '/integration', '/sandbox')
	)
	return {
		'survey': survey or str(payload.get('survey', 'unknown')),
		'experiment': experiment,
		'model': str(model),
		'initialization': _initialization(model, embedding, checkpoint, actual),
		'protocol': protocol,
		'hmm_version': protocol.removeprefix('HMM ')
		if protocol.startswith('HMM v')
		else 'legacy'
		if 'legacy' in protocol
		else '',
		'head_type': 'multi' if '|' in ks_text else 'single' if ks_text else '',
		'head_ks': ks_text,
		'distillation_weight': loss.get(
			'distillation_weight',
			condition.get(
				'distillation_weight', inferred.get('distillation_weight', '')
			),
		),
		'consistency_weight': loss.get(
			'consistency_weight',
			condition.get('consistency_weight', inferred.get('consistency_weight', '')),
		),
		'encoder_seed': actual.get('train', {}).get('seed', ''),
		'decoder_seed': seed,
		'data_version': next(
			(
				p
				for p in parts
				if p.startswith(('facies_benchmark_', 'horizon_benchmark_'))
			),
			'',
		),
		'label_budget': payload.get('data_size')
		or evidence.get('data_size')
		or fields['size']
		or fields['budget'],
		'replicate_id': str(replicate or ''),
		'replicate_kind': kind,
		'checkpoint_selection': str(variant),
		'evaluation_variant': path.stem,
		'evaluation_kind': 'token_probe'
		if 'probes' in parts
		else 'end_to_end'
		if identity.get('encoder_source') or 'end_to_end' in relative
		else 'frozen_decoder',
		'status': 'smoke/feasibility' if smoke else 'saved_result',
		'archived': archived,
		'duplicate_of': '',
		'source_cohort': cohort,
		'definition': definition,
		'condition_config': condition_path,
		'encoder_checkpoint': portable(str(checkpoint or ''), workspace),
		'source_file': relative,
		'source_id': hashlib.sha256(relative.encode()).hexdigest()[:16],
		'source_sha256': '',
		'best_epoch': payload.get('best_epoch', ''),
		'condition_inference': 'recorded_training_config'
		if actual
		else 'condition_manifest'
		if condition
		else 'name_inference_or_unknown',
	}


def evaluation_scopes(
	payload: dict[str, Any], path: Path
) -> list[tuple[str, str, dict[str, Any]]]:
	"""Keep validation, common test and per-horizon support definitions separate."""
	result = []
	for split in ('validation', 'test'):
		data = payload.get(split)
		if not isinstance(data, dict):
			continue
		if split == 'test' and 'primary_common' in data:
			result.extend(
				(split, scope, data[scope])
				for scope in ('primary_common', 'secondary_per_horizon')
				if isinstance(data.get(scope), dict)
			)
		else:
			result.append(
				(
					split,
					'validation_labels' if split == 'validation' else 'test_labels',
					data,
				)
			)
	if not result and any(
		k in payload
		for k in (
			'mean_iou',
			'macro_f1',
			'accuracy',
			'channel_iou',
			'macro_mae_samples',
		)
	):
		unit = str(payload.get('aggregation_unit', ''))
		split = payload.get('split', '')
		if 'common_test' in str(path) or ('test' in unit and not split):
			split = 'common_test'
		elif 'validation' in unit or '/validation/' in str(path):
			split = 'validation'
		elif 'probes' in path.parts:
			probe = _safe_sidecar(path.with_name('probe_config_resolved.json'))
			if probe.get('inputs', {}).get('validation_tokens'):
				split, unit = 'validation', 'validation_tokens'
		result.append((str(split or 'unknown'), unit or 'source-defined', payload))
	return result


def discover(artifact_root: Path, output: Path) -> list[Path]:
	"""Scan canonical payloads once; do not follow directory compatibility aliases."""
	paths: list[Path] = []
	seen: set[Path] = set()
	for root in [
		*(artifact_root / 'surveys' / s for s in SURVEYS),
		artifact_root / 'archive',
		artifact_root / 'studies',
	]:
		for directory, dirs, files in os.walk(root, followlinks=False):
			dirs[:] = [
				d
				for d in dirs
				if not d.startswith('.')
				and not (Path(directory) / d).resolve().is_relative_to(output.resolve())
				and not d.startswith(
					('all_experiment_metrics_', 'results_figure_candidates_')
				)
			]
			for name in files:
				path = Path(directory) / name
				if path.suffix not in ('.json', '.csv'):
					continue
				if path.suffix == '.json' and not any(
					x in name.lower()
					for x in (
						'metrics',
						'summary',
						'report',
						'comparison',
						'screening_validation',
					)
				):
					continue
				resolved = path.resolve()
				if (
					resolved in seen
					or not resolved.is_file()
					or not resolved.is_relative_to(artifact_root.resolve())
				):
					continue
				seen.add(resolved)
				paths.append(resolved)
	return sorted(paths)


def _write_csv(
	path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | None = None
) -> None:
	fieldnames = fields or tuple(dict.fromkeys(k for row in rows for k in row))
	with path.open('w', encoding='utf-8-sig', newline='') as stream:
		writer = csv.DictWriter(
			stream, fieldnames=fieldnames, extrasaction='ignore', lineterminator='\n'
		)
		writer.writeheader()
		for row in rows:
			writer.writerow(
				{
					k: json.dumps(v, ensure_ascii=False, separators=(',', ':'))
					if isinstance(v, (dict, list))
					else v
					for k, v in row.items()
				}
			)


def _main_value(metrics: dict[str, Any], survey: str) -> tuple[str, object]:
	name = {
		'f3': 'mean_iou',
		'parihaka': 'channel_iou',
		'volve': 'macro_mae_samples',
	}.get(survey, '')
	return name, metrics.get(name)


def summarize_layouts(
	records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	"""Compute descriptive SD within an exact cohort, budget, split and setting."""
	group_fields = (
		'survey',
		'experiment',
		'model',
		'source_cohort',
		'initialization',
		'protocol',
		'head_ks',
		'distillation_weight',
		'consistency_weight',
		'encoder_seed',
		'decoder_seed',
		'data_version',
		'label_budget',
		'split',
		'support',
		'checkpoint_selection',
		'evaluation_variant',
		'evaluation_kind',
		'status',
		'archived',
		'replicate_kind',
	)
	groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
	for record in records:
		groups[tuple(str(record.get(k, '')) for k in group_fields)].append(record)
	long_rows, wide_rows = [], []
	for number, (_, group) in enumerate(sorted(groups.items()), 1):
		base = {k: group[0].get(k, '') for k in group_fields}
		base['group_id'] = f'G{number:05d}'
		base['source_record_count'] = len(group)
		wide = dict(base)
		for metric in sorted(
			{
				k
				for r in group
				for k in r['_metrics']
				if '[' not in k
				and not k.startswith(('per_class', 'confusion', 'per_horizon'))
			}
		):
			by_replicate: dict[str, list[float]] = defaultdict(list)
			for r in group:
				value = r['_metrics'].get(metric)
				if value is not None:
					by_replicate[r['replicate_id'] or r['record_id']].append(value)
			conflict = any(len(set(v)) > 1 for v in by_replicate.values())
			vals = [v[0] for v in by_replicate.values()]
			row = {
				**base,
				'metric': metric,
				'n': len(vals),
				'mean': statistics.mean(vals) if vals and not conflict else None,
				'sample_sd': statistics.stdev(vals)
				if len(vals) > 1 and not conflict
				else None,
				'aggregation_status': 'conflicting_duplicate_replicates'
				if conflict
				else 'descriptive',
				'unit': metric_info(metric)[0],
				'direction': metric_info(metric)[1],
			}
			long_rows.append(row)
			if metric in MAIN_METRICS:
				wide[metric] = row['mean']
				wide[metric + '_sd'] = row['sample_sd']
				wide['n'] = row['n']
		wide_rows.append(wide)
	return long_rows, wide_rows


def build_catalog(
	workspace: Path, artifact_root: Path, output: Path, *, dry_run: bool = False
) -> dict[str, Any]:
	"""Export endpoint metrics, summaries, detailed CSVs and an offline HTML report."""
	workspace, artifact_root, output = (
		workspace.resolve(),
		artifact_root.resolve(),
		output.resolve(),
	)
	paths = discover(artifact_root, output)
	if dry_run:
		return {
			'status': 'dry_run',
			'candidate_files': len(paths),
			'candidate_bytes': sum(p.stat().st_size for p in paths),
		}
	if output.exists():
		raise FileExistsError(output)
	output.mkdir(parents=True)
	(output / 'source_tables').mkdir()
	index = DefinitionIndex(workspace, artifact_root)
	records, sources, reported = [], [], []
	duplicate_keys: dict[tuple, str] = {}
	copied_tables: dict[str, str] = {}
	for position, path in enumerate(paths, 1):
		if position % 500 == 0:
			print(
				f'files {position}/{len(paths)}; evaluations {len(records)}', flush=True
			)
		before = path.stat()
		relative = portable(str(path), workspace)
		source_id = hashlib.sha256(relative.encode()).hexdigest()[:16]
		source = {
			'source_id': source_id,
			'path': relative,
			'bytes': before.st_size,
			'sha256': '',
			'category': '',
			'status': '',
			'record_count': 0,
			'bundle_table': '',
		}
		sources.append(source)
		try:
			if path.suffix == '.csv':
				with path.open(encoding='utf-8-sig', newline='') as stream:
					reader = csv.reader(stream)
					header = next(reader, [])
				if not METRIC_TOKEN.search(' '.join(header)) and not any(
					token in path.name for token in ('metrics', 'confusion_matrix')
				):
					source.update(
						category='non_metric_table', status='excluded_with_reason'
					)
					continue
				source['sha256'] = sha256(path)
				source['category'] = (
					'training_diagnostics_csv'
					if 'epoch_metrics' in path.name
					else 'detailed_or_published_csv'
				)
				bundle = copied_tables.get(source['sha256'])
				if not bundle:
					bundle = f'source_tables/{source_id}_{path.name}'
					(output / bundle).write_text(
						portable(path.read_text(encoding='utf-8-sig'), workspace),
						encoding='utf-8-sig',
					)
					copied_tables[source['sha256']] = bundle
				source.update(status='included_as_source_table', bundle_table=bundle)
				continue
			source['sha256'] = sha256(path)
			payload = _json(path)
			base = context(path, payload, index)
			base['source_sha256'] = source['sha256']
			is_endpoint = path.name == 'metrics.json' or path.stem.startswith(
				'metrics_'
			)
			if is_endpoint:
				for split, support, data in evaluation_scopes(payload, path):
					metrics = flatten_metrics(data)
					if not metrics:
						continue
					record = {
						**base,
						'split': split,
						'support': support,
						'record_id': f'R{len(records) + 1:06d}',
						'_metrics': metrics,
					}
					record.update((k, metrics.get(k)) for k in MAIN_METRICS)
					record['primary_metric'], record['primary_value'] = _main_value(
						metrics, record['survey']
					)
					# Identical metric files within the same run and support.
					key = (str(path.parent), split, support, source['sha256'])
					if key in duplicate_keys:
						record['duplicate_of'] = duplicate_keys[key]
					else:
						duplicate_keys[key] = record['record_id']
					records.append(record)
					source['record_count'] += 1
				source.update(
					category='endpoint_metrics',
					status='parsed'
					if source['record_count']
					else 'no_recognized_evaluation_scope',
				)
			else:
				metrics = flatten_metrics(payload)
				for metric, value in metrics.items():
					reported.append(
						{
							'survey': base['survey'],
							'experiment': base['experiment'],
							'source_file': relative,
							'source_id': source_id,
							'metric_path': metric,
							'value': value,
							'unit': metric_info(metric)[0],
							'direction': metric_info(metric)[1],
						}
					)
				source.update(
					category='published_or_auxiliary_json',
					status='parsed',
					record_count=len(metrics),
				)
			if (
				path.stat().st_size != before.st_size
				or path.stat().st_mtime_ns != before.st_mtime_ns
			):
				source['status'] = 'changed_during_snapshot'
		except (
			OSError,
			ValueError,
			TypeError,
			KeyError,
			AttributeError,
			RecursionError,
		) as error:
			source.update(
				status='parse_error', error=f'{type(error).__name__}: {error}'
			)
	# Detailed trace, slice and region CSVs retain their original numeric strings.
	long_rows = []
	for record in records:
		for metric, value in record['_metrics'].items():
			unit, direction = metric_info(metric)
			long_rows.append(
				{
					'record_id': record['record_id'],
					'survey': record['survey'],
					'experiment': record['experiment'],
					'model': record['model'],
					'split': record['split'],
					'support': record['support'],
					'label_budget': record['label_budget'],
					'replicate_id': record['replicate_id'],
					'metric': metric,
					'value': value,
					'unit': unit,
					'direction': direction,
					'source_id': record['source_id'],
					'finite': value is not None,
				}
			)
	aggregates, wide = summarize_layouts([r for r in records if not r['duplicate_of']])
	for entry in index.catalog.values():
		matching = [r for r in records if r['experiment'] == entry['definition']]
		entry['evaluation_rows'] = len(matching)
		entry['models_with_metrics'] = len({r['model'] for r in matching})
		entry['status'] = (
			'saved_metrics_available' if matching else 'no_endpoint_metrics_matched'
		)
		entry['readme'] = (
			entry['definition'] + '/README.md'
			if (workspace / entry['definition'] / 'README.md').exists()
			else ''
		)
	dictionary = {}
	for row in [*long_rows, *reported]:
		metric = row.get('metric', row.get('metric_path', ''))
		entry = dictionary.setdefault(
			metric,
			{
				'metric': metric,
				'unit': row['unit'],
				'direction': row['direction'],
				'observations': 0,
				'missing_nonfinite': 0,
			},
		)
		entry['observations'] += 1
		entry['missing_nonfinite'] += row['value'] is None
	_write_csv(
		output / 'experiment_metrics.csv',
		records,
		(*CONTEXT_FIELDS, *MAIN_METRICS, 'primary_metric', 'primary_value'),
	)
	_write_csv(output / 'metrics_long.csv', long_rows)
	_write_csv(output / 'layout_summary.csv', aggregates)
	_write_csv(output / 'published_statistics.csv', reported)
	_write_csv(output / 'experiment_catalog.csv', list(index.catalog.values()))
	_write_csv(output / 'source_files.csv', sources)
	_write_csv(output / 'metric_dictionary.csv', list(dictionary.values()))
	validation = {
		'status': 'exported',
		'generated_utc': datetime.now(timezone.utc).isoformat(),
		'endpoint_rows': len(records),
		'metric_values': len(long_rows),
		'layout_summary_rows': len(aggregates),
		'published_statistic_values': len(reported),
		'source_files': len(sources),
		'by_survey': dict(Counter(r['survey'] for r in records)),
		'definition_catalog': len(index.catalog),
		'source_statuses': dict(Counter(s['status'] for s in sources)),
		'copied_csv_files': len(copied_tables),
		'missing_nonfinite_values': sum(r['value'] is None for r in long_rows),
		'new_training_or_evaluation': False,
		'independence': 'layouts/subsamples are not independent encoder seeds',
		'csv_numeric_values_preserved': True,
		'metric_identity': 'exact original metric paths retained',
		'scope': (
			'retained canonical survey outputs, archives and cross-survey '
			'studies; directory aliases not followed; reports not consumed'
		),
	}
	(output / 'validation.json').write_text(
		json.dumps(validation, ensure_ascii=False, indent=2)
	)
	data = {
		'records': records,
		'groups': wide,
		'definitions': list(index.catalog.values()),
		'sources': sources,
		'validation': validation,
		'main_metrics': MAIN_METRICS,
	}
	template = Path(__file__).with_name('experiment_catalog.html').read_text()
	payload = (
		json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
		.replace('<', '\\u003c')
		.replace('>', '\\u003e')
		.replace('&', '\\u0026')
	)
	(output / 'index.html').write_text(template.replace('__CATALOG_DATA__', payload))
	files = sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file())
	expected = FIXED_FILES | {
		str(p.relative_to(output)) for p in (output / 'source_tables').glob('*.csv')
	}
	if set(files) != expected - {'manifest.json'}:
		raise ValueError('published file set differs from exact ownership')
	manifest = {
		'published_files': sorted(expected),
		'sha256': {f: sha256(output / f) for f in files},
		'self_hash_policy': 'manifest.json is listed without a self hash',
	}
	(output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
	return validation


def build_parser() -> argparse.ArgumentParser:
	"""Expose the standard thin-CLI parser without reading experiment outputs."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--workspace', type=Path, default=Path.cwd())
	parser.add_argument('--artifact-root', type=Path)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--zip', action='store_true')
	return parser


def main() -> None:
	"""Write a portable CSV/HTML bundle from saved evaluations only."""
	args = build_parser().parse_args()
	root = args.artifact_root or args.workspace / 'artifacts/seis_ssl_cluster'
	result = build_catalog(args.workspace, root, args.output, dry_run=args.dry_run)
	print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
	if args.zip and not args.dry_run:
		manifest = json.loads((args.output / 'manifest.json').read_text())
		archive = args.output.with_suffix('.zip')
		with zipfile.ZipFile(
			archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6
		) as stream:
			for path in manifest['published_files']:
				stream.write(args.output / path, f'{args.output.name}/{path}')
		with zipfile.ZipFile(archive) as stream:
			if stream.testzip() is not None:
				raise ValueError('archive integrity failed')
		print(f'archive: {archive}', flush=True)


if __name__ == '__main__':
	main()
