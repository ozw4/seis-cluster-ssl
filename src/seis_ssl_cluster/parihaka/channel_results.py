"""Paired results for the Parihaka Channel benchmark."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.parihaka.channel_data import (
	CHANNEL_AXIS_MAPPING,
	DATA_SIZE_PREFIX,
	LAYOUT_IDS,
)
from seis_ssl_cluster.parihaka.channel_decoder import (
	CHANNEL_PRETRAINED_CHECKPOINT_SUFFIX,
	CHANNEL_PRETRAINED_MODEL_TAG,
	CHANNEL_RANDOM_ENCODER_SEED,
)

MODELS = ('pretrained', 'random')
OUTPUT_NAMES = ('comparison.csv', 'summary.json', 'summary.md')

_BENCHMARK_IDENTITY_KEYS = {
	'model',
	'layout_id',
	'data_size',
	'embedding',
	'decoder_initial_state_sha256',
	'label_path',
	'label_metadata_path',
	'prepared_label_identity',
	'train_lines',
	'validation',
	'test',
	'geometry',
	'class_weights',
	'decoder',
	'training',
	'tiles',
	'split_class_counts',
	'tile_counts',
}
_EMBEDDING_COMMON_METADATA_KEYS = {
	'survey_id',
	'source_amplitude_path',
	'volume_shape_xyz',
	'model_geometry',
	'patch_size',
	'token_grid_shape',
	'window_size',
	'overlap',
	'output_dtype',
	'min_token_valid_fraction',
	'normalization_stats_path',
	'preprocessing',
	'zero_mask',
	'precision',
	'pretraining_objective',
}
_GLOBAL_IDENTITY_KEYS = (
	'label_path',
	'label_metadata_path',
	'prepared_label_identity',
	'geometry',
	'decoder',
	'decoder_initial_state_sha256',
	'training',
	'tiles',
)
_PRETRAINED_MODEL_SOURCE_KEYS = {
	'role',
	'checkpoint_path',
	'checkpoint_sha256',
	'model_tag',
}
_RANDOM_MODEL_SOURCE_KEYS = {
	'role',
	'checkpoint_path',
	'checkpoint_sha256',
	'random_encoder_baseline',
	'pretrained_weights_loaded',
	'seed',
	'checkpoint_kind',
	'reference_checkpoint',
	'reference_checkpoint_sha256',
	'reference_model_tag',
}


@dataclass(frozen=True)
class ChannelSummaryConfig:
	"""Resolved benchmark result paths."""

	runs_root: Path
	output_dir: Path


def channel_summary_config_from_mapping(
	config: Mapping[str, object],
) -> ChannelSummaryConfig:
	"""Resolve the direct summary configuration."""
	inputs = _mapping(config, 'inputs')
	outputs = _mapping(config, 'outputs')
	return ChannelSummaryConfig(
		runs_root=_absolute_path(inputs, 'runs_root', 'inputs'),
		output_dir=_absolute_path(outputs, 'output_dir', 'outputs'),
	)


def inspect_channel_benchmark_results(
	config: ChannelSummaryConfig,
) -> dict[tuple[str, str, str], Mapping[str, object]]:
	"""Require and validate all 30 metrics files."""
	rows: dict[tuple[str, str, str], Mapping[str, object]] = {}
	missing: list[Path] = []
	for model in MODELS:
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZE_PREFIX:
				path = _metrics_path(config.runs_root, model, layout_id, data_size)
				if not path.is_file():
					missing.append(path)
					continue
				payload = _read_json(path)
				_validate_identity(payload, model, layout_id, data_size, path)
				_metric(payload, 'test', 'channel_iou', path)
				_validate_supervision(payload, path)
				_class_weights(payload, path)
				_validate_benchmark_identity(
					payload, model, layout_id, data_size, path
				)
				rows[(model, layout_id, data_size)] = payload
	if missing:
		raise FileNotFoundError(
			f'Parihaka Channel summary requires all 30 jobs; missing {len(missing)}: '
			+ ', '.join(str(path) for path in missing)
		)
	if len(rows) != 30:
		raise ValueError(f'expected 30 unique benchmark conditions; got {len(rows)}')
	_validate_supervision_parity(rows, config.runs_root)
	return rows


def summarize_channel_benchmark(
	config: ChannelSummaryConfig,
) -> tuple[Path, Path, Path]:
	"""Write the paired Channel-IoU comparison and size aggregates."""
	jobs = inspect_channel_benchmark_results(config)
	if config.output_dir.exists() and any(
		(config.output_dir / name).exists() for name in OUTPUT_NAMES
	):
		raise FileExistsError(
			f'channel summary outputs already exist: {config.output_dir}'
		)
	comparison: list[dict[str, object]] = []
	for data_size in DATA_SIZE_PREFIX:
		for layout_id in LAYOUT_IDS:
			pretrained = _metric(
				jobs[('pretrained', layout_id, data_size)],
				'test',
				'channel_iou',
				_metrics_path(config.runs_root, 'pretrained', layout_id, data_size),
			)
			random = _metric(
				jobs[('random', layout_id, data_size)],
				'test',
				'channel_iou',
				_metrics_path(config.runs_root, 'random', layout_id, data_size),
			)
			comparison.append(
				{
					'data_size': data_size,
					'layout_id': layout_id,
					'pretrained_channel_iou': pretrained,
					'random_channel_iou': random,
					'delta_channel_iou': pretrained - random,
				}
			)
	aggregates: dict[str, object] = {}
	for data_size in DATA_SIZE_PREFIX:
		selected = [row for row in comparison if row['data_size'] == data_size]
		deltas = [float(row['delta_channel_iou']) for row in selected]
		aggregates[data_size] = {
			'paired_mean': statistics.fmean(deltas),
			'paired_median': statistics.median(deltas),
			'sample_standard_deviation': statistics.stdev(deltas),
			'pretrained_wins': sum(delta > 0 for delta in deltas),
			'ties': sum(delta == 0 for delta in deltas),
			'pretrained_losses': sum(delta < 0 for delta in deltas),
			'layout_deltas': {
				str(row['layout_id']): row['delta_channel_iou'] for row in selected
			},
		}
	payload = {
		'schema_version': 1,
		'primary_metric': 'test.channel_iou',
		'job_count': len(jobs),
		'comparison': comparison,
		'by_size': aggregates,
	}
	config.output_dir.mkdir(parents=True, exist_ok=True)
	comparison_path = config.output_dir / OUTPUT_NAMES[0]
	with comparison_path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=tuple(comparison[0]))
		writer.writeheader()
		writer.writerows(comparison)
	json_path = config.output_dir / OUTPUT_NAMES[1]
	json_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)
	markdown_path = config.output_dir / OUTPUT_NAMES[2]
	markdown_path.write_text(_markdown(aggregates, comparison), encoding='utf-8')
	return comparison_path, json_path, markdown_path


def _markdown(
	aggregates: Mapping[str, object], comparison: list[dict[str, object]]
) -> str:
	lines = [
		'# Parihaka Channel benchmark',
		'',
		'Primary metric: test Channel IoU. Deltas are pretrained minus random.',
		'',
		'| size | paired mean | paired median | sample std | wins/ties/losses |',
		'|---|---:|---:|---:|---:|',
	]
	for data_size in DATA_SIZE_PREFIX:
		row = _mapping(aggregates, data_size)
		lines.append(
			f'| {data_size} | {float(row["paired_mean"]):.6f} | '
			f'{float(row["paired_median"]):.6f} | '
			f'{float(row["sample_standard_deviation"]):.6f} | '
			f'{row["pretrained_wins"]}/{row["ties"]}/{row["pretrained_losses"]} |'
		)
	lines.extend(
		[
			'',
			'| size | layout | pretrained | random | delta |',
			'|---|---|---:|---:|---:|',
		]
	)
	lines.extend(
		f'| {row["data_size"]} | {row["layout_id"]} | '
		f'{float(row["pretrained_channel_iou"]):.6f} | '
		f'{float(row["random_channel_iou"]):.6f} | '
		f'{float(row["delta_channel_iou"]):.6f} |'
		for row in comparison
	)
	return '\n'.join(lines) + '\n'


def _metrics_path(root: Path, model: str, layout: str, size: str) -> Path:
	return (
		root / f'model={model}' / f'layout={layout}' / f'size={size}' / 'metrics.json'
	)


def _validate_identity(
	payload: Mapping[str, object], model: str, layout: str, size: str, path: Path
) -> None:
	expected = {'model': model, 'layout_id': layout, 'data_size': size}
	for key, value in expected.items():
		if payload.get(key) != value:
			raise ValueError(f'{path} has incorrect {key}: {payload.get(key)!r}')


def _metric(payload: Mapping[str, object], split: str, key: str, path: Path) -> float:
	value = payload.get(split)
	if not isinstance(value, Mapping):
		raise TypeError(f'{path} {split} must be a mapping')
	metric = value.get(key)
	if not isinstance(metric, int | float) or isinstance(metric, bool):
		raise TypeError(f'{path} {split}.{key} must be numeric')
	return float(metric)


def _validate_supervision(payload: Mapping[str, object], path: Path) -> None:
	supervision = payload.get('supervision')
	if not isinstance(supervision, Mapping):
		raise TypeError(f'{path} supervision must be a mapping')
	expected_keys = {
		'axis_mapping',
		'train_inline',
		'train_crossline',
		'validation_inline',
		'validation_crossline',
		'test_inline',
		'test_crossline',
		'split_class_counts',
	}
	if set(supervision) != expected_keys:
		raise ValueError(
			f'{path} supervision must contain exactly {sorted(expected_keys)!r}'
		)
	if supervision.get('axis_mapping') != CHANNEL_AXIS_MAPPING:
		raise ValueError(
			f"{path} supervision.axis_mapping must be "
			f'{CHANNEL_AXIS_MAPPING!r}'
		)
	for key in (
		'train_inline',
		'train_crossline',
		'validation_inline',
		'validation_crossline',
		'test_inline',
		'test_crossline',
	):
		_indices(supervision.get(key), f'{path} supervision.{key}')
	counts = supervision.get('split_class_counts')
	if not isinstance(counts, Mapping):
		raise TypeError(f'{path} supervision.split_class_counts must be a mapping')
	if set(counts) != {'train', 'validation', 'test'}:
		raise ValueError(
			f'{path} supervision.split_class_counts must contain exactly '
			"'train', 'validation', and 'test'"
		)
	for split in ('train', 'validation', 'test'):
		_class_counts(
			counts.get(split), f'{path} supervision.split_class_counts.{split}'
		)


def _validate_supervision_parity(
	rows: Mapping[tuple[str, str, str], Mapping[str, object]], runs_root: Path
) -> None:
	_validate_common_held_out(rows, runs_root)
	_validate_pairs(rows, runs_root)
	_validate_nested_training(rows)
	_validate_unique_layout_training(rows)
	for key, payload in rows.items():
		path = _metrics_path(runs_root, *key)
		_validate_identity_redundancy(
			payload, _benchmark_identity(payload, path), path
		)
	_validate_benchmark_identity_parity(rows, runs_root)


def _validate_benchmark_identity(
	payload: Mapping[str, object],
	model: str,
	layout_id: str,
	data_size: str,
	path: Path,
) -> None:
	identity = _benchmark_identity(payload, path)
	if set(identity) != _BENCHMARK_IDENTITY_KEYS:
		raise ValueError(
			f'{path} benchmark_identity must contain exactly '
			f'{sorted(_BENCHMARK_IDENTITY_KEYS)!r}'
		)
	for key, expected in (
		('model', model),
		('layout_id', layout_id),
		('data_size', data_size),
	):
		if identity.get(key) != expected:
			raise ValueError(
				f'{path} benchmark_identity.{key} must equal {expected!r}'
			)
	embedding = _identity_mapping(identity, 'embedding', path)
	if set(embedding) != {
		'checkpoint_path',
		'checkpoint_sha256',
		'model_source',
		'common_metadata',
	}:
		raise ValueError(f'{path} benchmark_identity.embedding has invalid fields')
	checkpoint_path = embedding.get('checkpoint_path')
	if not isinstance(checkpoint_path, str) or not checkpoint_path:
		raise TypeError(
			f'{path} benchmark_identity.embedding.checkpoint_path must be non-empty'
		)
	_validate_sha256(
		embedding.get('checkpoint_sha256'),
		f'{path} benchmark_identity.embedding.checkpoint_sha256',
	)
	_validate_model_source(embedding, model, path)
	common_metadata = _identity_mapping(
		embedding, 'common_metadata', path, prefix='benchmark_identity.embedding'
	)
	if set(common_metadata) != _EMBEDDING_COMMON_METADATA_KEYS:
		raise ValueError(
			f'{path} benchmark_identity.embedding.common_metadata has invalid fields'
		)
	_validate_sha256(
		identity.get('decoder_initial_state_sha256'),
		f'{path} benchmark_identity.decoder_initial_state_sha256',
	)
	_validate_label_identity_paths(identity, path)
	_validate_prepared_label_identity(identity, path)
	for key, expected_keys in (
		('train_lines', {'inline', 'crossline'}),
		('validation', {'inline', 'crossline'}),
		('test', {'inline', 'crossline'}),
		(
			'geometry',
			{
				'embedding_shape',
				'volume_shape_xyz',
				'token_grid_shape_xyz',
				'patch_size_xyz',
			},
		),
		(
			'decoder',
			{
				'spec',
				'embedding_dim',
				'class_count',
				'hidden_channels',
				'upsample_factors',
				'upsample_mode',
				'normalization',
			},
		),
		(
			'training',
			{
				'epochs',
				'batch_size',
				'learning_rate',
				'weight_decay',
				'class_weight',
				'sampling_mode',
				'seed',
				'amp',
				'gradient_clip_norm',
			},
		),
		('tiles', {'core_size_tokens', 'context_halo_tokens'}),
		('split_class_counts', {'train', 'validation', 'test'}),
		('tile_counts', {'train', 'validation', 'test'}),
	):
		value = _identity_mapping(identity, key, path)
		if set(value) != expected_keys:
			raise ValueError(
				f'{path} benchmark_identity.{key} must contain exactly '
				f'{sorted(expected_keys)!r}'
			)


def _validate_prepared_label_identity(
	identity: Mapping[str, object], path: Path
) -> None:
	prepared_label = _identity_mapping(identity, 'prepared_label_identity', path)
	expected_label_keys = {
		'labels_sha256',
		'source_npz_path',
		'source_key',
		'shape',
		'dtype',
		'class_definition',
	}
	if set(prepared_label) != expected_label_keys:
		raise ValueError(
			f'{path} benchmark_identity.prepared_label_identity has invalid fields'
		)
	_validate_sha256(
		prepared_label.get('labels_sha256'),
		f'{path} benchmark_identity.prepared_label_identity.labels_sha256',
	)
	for key in ('source_npz_path', 'source_key', 'dtype'):
		value = prepared_label.get(key)
		if not isinstance(value, str) or not value:
			raise TypeError(
				f'{path} benchmark_identity.prepared_label_identity.{key} '
				'must be non-empty'
			)
	shape = prepared_label.get('shape')
	if (
		not isinstance(shape, list)
		or len(shape) != 3
		or any(
			not isinstance(item, int) or isinstance(item, bool) or item <= 0
			for item in shape
		)
	):
		raise TypeError(
			f'{path} benchmark_identity.prepared_label_identity.shape is invalid'
		)
	if prepared_label.get('dtype') != 'int8':
		raise ValueError(
			f'{path} benchmark_identity.prepared_label_identity.dtype must be int8'
		)
	if prepared_label.get('class_definition') != {
		'positive_class_id': 5,
		'negative_class_ids': [1, 2, 3, 4, 6],
	}:
		raise ValueError(
			f'{path} benchmark_identity.prepared_label_identity.class_definition '
			'is invalid'
		)


def _validate_label_identity_paths(
	identity: Mapping[str, object], path: Path
) -> None:
	for key in ('label_path', 'label_metadata_path'):
		value = identity.get(key)
		if not isinstance(value, str) or not value:
			raise TypeError(f'{path} benchmark_identity.{key} must be non-empty')


def _validate_model_source(  # noqa: C901
	embedding: Mapping[str, object], model: str, path: Path
) -> None:
	source = _identity_mapping(
		embedding, 'model_source', path, prefix='benchmark_identity.embedding'
	)
	expected_keys = (
		_PRETRAINED_MODEL_SOURCE_KEYS
		if model == 'pretrained'
		else _RANDOM_MODEL_SOURCE_KEYS
	)
	if set(source) != expected_keys:
		raise ValueError(
			f'{path} benchmark_identity.embedding.model_source has invalid fields '
			f'for {model}'
		)
	if source.get('role') != model:
		raise ValueError(
			f'{path} benchmark_identity embedding model-source role must be {model}'
		)
	if source.get('checkpoint_path') != embedding.get('checkpoint_path'):
		raise ValueError(
			f'{path} model-source checkpoint_path does not match embedding identity'
		)
	if source.get('checkpoint_sha256') != embedding.get('checkpoint_sha256'):
		raise ValueError(
			f'{path} model-source checkpoint_sha256 does not match embedding identity'
		)
	if model == 'pretrained':
		checkpoint_path = Path(str(source['checkpoint_path']))
		if tuple(
			checkpoint_path.parts[-len(CHANNEL_PRETRAINED_CHECKPOINT_SUFFIX) :]
		) != CHANNEL_PRETRAINED_CHECKPOINT_SUFFIX:
			raise ValueError(
				f'{path} pretrained model-source checkpoint is not the expected '
				'Parihaka full_100ep/latest.pt'
			)
		if source.get('model_tag') != CHANNEL_PRETRAINED_MODEL_TAG:
			raise ValueError(f'{path} pretrained model-source model_tag mismatch')
		return
	_validate_sha256(
		source.get('reference_checkpoint_sha256'),
		f'{path} random model-source reference_checkpoint_sha256',
	)
	if source.get('random_encoder_baseline') is not True:
		raise ValueError(
			f'{path} random model-source random_encoder_baseline mismatch'
		)
	if source.get('pretrained_weights_loaded') is not False:
		raise ValueError(
			f'{path} random model-source pretrained_weights_loaded mismatch'
		)
	expected = {
		'seed': CHANNEL_RANDOM_ENCODER_SEED,
		'checkpoint_kind': 'random_init',
		'reference_model_tag': CHANNEL_PRETRAINED_MODEL_TAG,
	}
	for key, value in expected.items():
		if source.get(key) != value:
			raise ValueError(f'{path} random model-source {key} mismatch')
	if not isinstance(source.get('reference_checkpoint'), str) or not source.get(
		'reference_checkpoint'
	):
		raise TypeError(
			f'{path} random model-source reference_checkpoint must be non-empty'
		)


def _validate_identity_redundancy(
	payload: Mapping[str, object], identity: Mapping[str, object], path: Path
) -> None:
	supervision = _supervision(payload)
	for identity_key, prefix in (
		('train_lines', 'train'),
		('validation', 'validation'),
		('test', 'test'),
	):
		lines = _identity_mapping(identity, identity_key, path)
		for orientation in ('inline', 'crossline'):
			if lines.get(orientation) != supervision.get(f'{prefix}_{orientation}'):
				raise ValueError(
					f'{path} benchmark_identity.{identity_key}.{orientation} '
					'does not match supervision'
				)
	if _class_weights(identity, path) != _class_weights(payload, path):
		raise ValueError(
			f'{path} benchmark_identity.class_weights does not match metrics'
		)
	identity_counts = _identity_mapping(identity, 'split_class_counts', path)
	if identity_counts != supervision.get('split_class_counts'):
		raise ValueError(
			f'{path} benchmark_identity.split_class_counts does not match '
			'supervision'
		)
	prepared_label = _identity_mapping(identity, 'prepared_label_identity', path)
	geometry = _identity_mapping(identity, 'geometry', path)
	if prepared_label.get('shape') != geometry.get('volume_shape_xyz'):
		raise ValueError(
			f'{path} prepared label shape does not match benchmark geometry'
		)
	tile_counts = _identity_mapping(identity, 'tile_counts', path)
	if any(
		not isinstance(tile_counts.get(split), int)
		or isinstance(tile_counts.get(split), bool)
		or int(tile_counts[split]) <= 0
		for split in ('train', 'validation', 'test')
	):
		raise TypeError(
			f'{path} benchmark_identity.tile_counts must contain positive integers'
		)


def _validate_benchmark_identity_parity(  # noqa: C901
	rows: Mapping[tuple[str, str, str], Mapping[str, object]], runs_root: Path
) -> None:
	first_key = (MODELS[0], LAYOUT_IDS[0], next(iter(DATA_SIZE_PREFIX)))
	common = _benchmark_identity(rows[first_key], _metrics_path(runs_root, *first_key))
	common_metadata = _embedding_identity(common)['common_metadata']
	for key, payload in rows.items():
		path = _metrics_path(runs_root, *key)
		identity = _benchmark_identity(payload, path)
		for field in _GLOBAL_IDENTITY_KEYS:
			if identity[field] != common[field]:
				raise ValueError(
					f'{path} benchmark_identity.{field} does not match all 30 jobs'
				)
		if _embedding_identity(identity)['common_metadata'] != common_metadata:
			raise ValueError(
				f'{path} embedding common metadata does not match all 30 jobs'
			)
	model_checkpoints: dict[str, tuple[object, object, object]] = {}
	for model in MODELS:
		first_model_key = (model, LAYOUT_IDS[0], next(iter(DATA_SIZE_PREFIX)))
		first_embedding = _embedding_identity(
			_benchmark_identity(
				rows[first_model_key], _metrics_path(runs_root, *first_model_key)
			)
		)
		checkpoint = (
			first_embedding['checkpoint_path'],
			first_embedding['checkpoint_sha256'],
			first_embedding['model_source'],
		)
		model_checkpoints[model] = checkpoint
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZE_PREFIX:
				key = (model, layout_id, data_size)
				embedding = _embedding_identity(
					_benchmark_identity(rows[key], _metrics_path(runs_root, *key))
				)
				if (
					embedding['checkpoint_path'],
					embedding['checkpoint_sha256'],
					embedding['model_source'],
				) != checkpoint:
					raise ValueError(
						f'{_metrics_path(runs_root, *key)} {model} embedding '
						'checkpoint does not match its other 15 jobs'
					)
	if model_checkpoints['pretrained'][1] == model_checkpoints['random'][1]:
		raise ValueError('pretrained and random checkpoint SHA-256 must differ')
	pretrained_source = model_checkpoints['pretrained'][2]
	random_source = model_checkpoints['random'][2]
	if not isinstance(pretrained_source, Mapping) or not isinstance(
		random_source, Mapping
	):
		raise TypeError('benchmark model-source identity must be a mapping')
	if random_source['reference_checkpoint'] != pretrained_source['checkpoint_path']:
		raise ValueError(
			'random model-source reference checkpoint does not match pretrained source'
		)
	if (
		random_source['reference_checkpoint_sha256']
		!= pretrained_source['checkpoint_sha256']
	):
		raise ValueError(
			'random model-source reference checkpoint SHA-256 does not match '
			'pretrained source'
		)


def _benchmark_identity(
	payload: Mapping[str, object], path: Path
) -> Mapping[str, object]:
	identity = payload.get('benchmark_identity')
	if not isinstance(identity, Mapping):
		raise TypeError(f'{path} benchmark_identity must be a mapping')
	return identity


def _embedding_identity(identity: Mapping[str, object]) -> Mapping[str, object]:
	value = identity.get('embedding')
	if not isinstance(value, Mapping):
		raise TypeError('benchmark_identity.embedding must be a mapping')
	return value


def _identity_mapping(
	value: Mapping[str, object],
	key: str,
	path: Path,
	*,
	prefix: str = 'benchmark_identity',
) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{path} {prefix}.{key} must be a mapping')
	return child


def _validate_sha256(value: object, label: str) -> None:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdefABCDEF' for character in value)
	):
		raise TypeError(f'{label} must be a SHA-256 hex digest')


def _validate_common_held_out(
	rows: Mapping[tuple[str, str, str], Mapping[str, object]], runs_root: Path
) -> None:
	first_key = (MODELS[0], LAYOUT_IDS[0], next(iter(DATA_SIZE_PREFIX)))
	common_supervision = _supervision(rows[first_key])
	common_held_out = _held_out_identity(common_supervision)
	for key, payload in rows.items():
		if _held_out_identity(_supervision(payload)) != common_held_out:
			raise ValueError(
				f'{_metrics_path(runs_root, *key)} validation/test supervision '
				'does not match all 30 jobs'
			)


def _validate_pairs(
	rows: Mapping[tuple[str, str, str], Mapping[str, object]], runs_root: Path
) -> None:
	for layout_id in LAYOUT_IDS:
		for data_size in DATA_SIZE_PREFIX:
			pretrained = rows[('pretrained', layout_id, data_size)]
			random = rows[('random', layout_id, data_size)]
			pretrained_path = _metrics_path(
				runs_root, 'pretrained', layout_id, data_size
			)
			random_path = _metrics_path(runs_root, 'random', layout_id, data_size)
			if _supervision(pretrained) != _supervision(random):
				raise ValueError(
					f'{layout_id}/{data_size} pretrained/random supervision mismatch'
				)
			pretrained_weights = _class_weights(
				pretrained,
				pretrained_path,
			)
			random_weights = _class_weights(
				random,
				random_path,
			)
			if pretrained_weights != random_weights:
				raise ValueError(
					f'{layout_id}/{data_size} pretrained/random class_weights mismatch'
				)
			if _paired_benchmark_identity(
				_benchmark_identity(pretrained, pretrained_path)
			) != _paired_benchmark_identity(
				_benchmark_identity(random, random_path)
			):
				raise ValueError(
					f'{layout_id}/{data_size} pretrained/random benchmark identity '
					'mismatch outside model-specific checkpoint'
				)


def _paired_benchmark_identity(
	identity: Mapping[str, object],
) -> dict[str, object]:
	return {
		**{
			key: value
			for key, value in identity.items()
			if key not in {'model', 'embedding'}
		},
		'embedding': {
			'common_metadata': _embedding_identity(identity)['common_metadata']
		},
	}


def _validate_nested_training(
	rows: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> None:
	for model in MODELS:
		for layout_id in LAYOUT_IDS:
			by_size = {
				data_size: _supervision(rows[(model, layout_id, data_size)])
				for data_size in DATA_SIZE_PREFIX
			}
			for orientation in ('inline', 'crossline'):
				key = f'train_{orientation}'
				large: tuple[int, ...] | None = None
				for data_size, prefix in DATA_SIZE_PREFIX.items():
					current = _indices(
						by_size[data_size].get(key),
						f'{model}/{layout_id}/{data_size} supervision.{key}',
					)
					if len(current) != prefix:
						raise ValueError(
							f'{model}/{layout_id}/{data_size} {key} must contain '
							f'exactly {prefix} indices'
						)
					if data_size == 'large':
						large = current
				if large is None:
					raise RuntimeError('large Channel supervision is unavailable')
				for data_size, prefix in DATA_SIZE_PREFIX.items():
					current = _indices(
						by_size[data_size].get(key),
						f'{model}/{layout_id}/{data_size} supervision.{key}',
					)
					if current != large[:prefix]:
						raise ValueError(
							f'{model}/{layout_id} {key} is not nested in '
							'small/medium/large prefix order'
						)


def _validate_unique_layout_training(
	rows: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> None:
	for model in MODELS:
		for data_size in DATA_SIZE_PREFIX:
			seen: dict[tuple[frozenset[int], frozenset[int]], str] = {}
			for layout_id in LAYOUT_IDS:
				supervision = _supervision(rows[(model, layout_id, data_size)])
				identity = (
					frozenset(
						_indices(
							supervision.get('train_inline'),
							f'{model}/{layout_id}/{data_size} '
							'supervision.train_inline',
						)
					),
					frozenset(
						_indices(
							supervision.get('train_crossline'),
							f'{model}/{layout_id}/{data_size} '
							'supervision.train_crossline',
						)
					),
				)
				if duplicate := seen.get(identity):
					raise ValueError(
						f'{model}/{data_size} training section sets must be unique '
						f'across layouts; {duplicate} and {layout_id} select the '
						'same sections'
					)
				seen[identity] = layout_id


def _supervision(payload: Mapping[str, object]) -> Mapping[str, object]:
	value = payload.get('supervision')
	if not isinstance(value, Mapping):
		raise TypeError('supervision must be a mapping')
	return value


def _held_out_identity(supervision: Mapping[str, object]) -> tuple[object, ...]:
	counts = supervision.get('split_class_counts')
	if not isinstance(counts, Mapping):
		raise TypeError('supervision.split_class_counts must be a mapping')
	return (
		supervision.get('axis_mapping'),
		supervision.get('validation_inline'),
		supervision.get('validation_crossline'),
		supervision.get('test_inline'),
		supervision.get('test_crossline'),
		counts.get('validation'),
		counts.get('test'),
	)


def _indices(value: object, label: str) -> tuple[int, ...]:
	if not isinstance(value, list) or not value:
		raise TypeError(f'{label} must be a non-empty list')
	if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
		raise TypeError(f'{label} must contain integers')
	items = tuple(value)
	if len(set(items)) != len(items):
		raise ValueError(f'{label} must not contain duplicates')
	return items


def _class_counts(value: object, label: str) -> tuple[int, int]:
	if (
		not isinstance(value, list)
		or len(value) != 2
		or any(
			not isinstance(item, int) or isinstance(item, bool) or item < 0
			for item in value
		)
	):
		raise TypeError(f'{label} must contain two non-negative integers')
	return value[0], value[1]


def _class_weights(payload: Mapping[str, object], path: Path) -> tuple[float, float]:
	value = payload.get('class_weights')
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 2
		or any(
			not isinstance(item, int | float)
			or isinstance(item, bool)
			or not math.isfinite(float(item))
			for item in value
		)
	):
		raise TypeError(f'{path} class_weights must contain two finite numbers')
	return float(value[0]), float(value[1])


def _read_json(path: Path) -> Mapping[str, object]:
	value = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(value, Mapping):
		raise TypeError(f'metrics must contain an object: {path}')
	return value


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return child


def _absolute_path(value: Mapping[str, object], key: str, prefix: str) -> Path:
	item = value.get(key)
	if not isinstance(item, str) or not item:
		raise ValueError(f'{prefix}.{key} must be a non-empty path')
	path = Path(item)
	if not path.is_absolute():
		raise ValueError(f'{prefix}.{key} must be absolute')
	return path


__all__ = [
	'ChannelSummaryConfig',
	'channel_summary_config_from_mapping',
	'inspect_channel_benchmark_results',
	'summarize_channel_benchmark',
]
