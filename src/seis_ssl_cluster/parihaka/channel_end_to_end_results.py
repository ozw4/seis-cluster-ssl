"""Paired summaries for Parihaka Channel end-to-end training."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.parihaka.channel_checkpoints import (
	CHANNEL_PRETRAINED_MODEL_TAG,
	CHANNEL_RANDOM_ENCODER_SEED,
)
from seis_ssl_cluster.parihaka.channel_data import (
	CHANNEL_AXIS_MAPPING,
	CHANNEL_TEST_MODE,
	DATA_SIZE_PREFIX,
	LAYOUT_IDS,
)
from seis_ssl_cluster.parihaka.channel_results import (
	ChannelSummaryConfig,
	inspect_channel_benchmark_results,
)

ENCODER_INITIALIZATIONS = ('pretrained', 'random')
END_TO_END_OUTPUT_NAMES = ('comparison.csv', 'summary.json', 'summary.md')
FOUR_WAY_OUTPUT_NAMES = (
	'four_way_comparison.csv',
	'four_way_summary.json',
	'four_way_summary.md',
)

_IDENTITY_KEYS = {
	'encoder_init',
	'layout_id',
	'data_size',
	'encoder_source',
	'encoder_initial_states',
	'reference_input',
	'labels',
	'supervision',
	'decoder',
	'optimizer',
	'training',
	'tiles',
	'runtime',
}
_REFERENCE_INPUT_KEYS = {
	'amplitude_path',
	'normalization_stats_path',
	'reference_metadata_path',
	'reference_metadata_sha256',
	'reference_valid_tokens_path',
	'reference_valid_tokens_sha256',
	'preprocessing',
	'zero_mask',
	'min_token_valid_fraction',
	'patch_size',
	'volume_shape',
	'token_grid_shape',
}
_SUPERVISION_KEYS = {
	'train_lines',
	'validation_lines',
	'test_definition',
	'split_class_counts',
	'tile_counts',
	'class_weights',
}


@dataclass(frozen=True)
class ChannelEndToEndSummaryConfig:
	"""Resolved end-to-end run and report paths."""

	runs_root: Path
	output_dir: Path
	four_way_output_dir: Path


def channel_end_to_end_summary_config_from_mapping(
	config: Mapping[str, object],
) -> ChannelEndToEndSummaryConfig:
	"""Resolve summary paths from the one-job end-to-end config."""
	outputs = _mapping(config, 'outputs', 'config')
	return ChannelEndToEndSummaryConfig(
		runs_root=_absolute_path(outputs, 'runs_root', 'outputs'),
		output_dir=_absolute_path(outputs, 'output_dir', 'outputs'),
		four_way_output_dir=_absolute_path(
			outputs, 'four_way_output_dir', 'outputs'
		),
	)


def inspect_channel_end_to_end_results(
	config: ChannelEndToEndSummaryConfig,
) -> dict[tuple[str, str, str], Mapping[str, object]]:
	"""Require and fail-closed validate all 30 end-to-end metrics files."""
	jobs: dict[tuple[str, str, str], Mapping[str, object]] = {}
	missing: list[Path] = []
	for encoder_init in ENCODER_INITIALIZATIONS:
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZE_PREFIX:
				path = _end_to_end_metrics_path(
					config.runs_root, encoder_init, layout_id, data_size
				)
				if not path.is_file():
					missing.append(path)
					continue
				payload = _read_json(path)
				_validate_end_to_end_job(
					payload, encoder_init, layout_id, data_size, path
				)
				key = (encoder_init, layout_id, data_size)
				if key in jobs:
					raise ValueError(f'duplicate end-to-end condition: {key!r}')
				jobs[key] = payload
	if missing:
		raise FileNotFoundError(
			'Parihaka Channel end-to-end summary requires all 30 jobs; '
			f'missing {len(missing)}: '
			+ ', '.join(str(path) for path in missing)
		)
	if len(jobs) != 30:
		raise ValueError(f'expected 30 unique end-to-end conditions; got {len(jobs)}')
	_validate_end_to_end_collection(jobs, config.runs_root)
	return jobs


def summarize_channel_end_to_end(
	config: ChannelEndToEndSummaryConfig,
) -> tuple[Path, Path, Path]:
	"""Write the paired end-to-end pretraining comparison."""
	jobs = inspect_channel_end_to_end_results(config)
	_require_outputs_absent(config.output_dir, END_TO_END_OUTPUT_NAMES)
	comparison = _end_to_end_comparison(jobs, config.runs_root)
	aggregates = _paired_aggregates(comparison, 'end_to_end_pretraining_delta')
	payload = {
		'schema_version': 1,
		'primary_metric': 'test.channel_iou',
		'paired_comparison': 'finetune_pretrained - train_from_scratch',
		'job_count': len(jobs),
		'comparison': comparison,
		'by_size': aggregates,
	}
	config.output_dir.mkdir(parents=True, exist_ok=True)
	comparison_path = config.output_dir / END_TO_END_OUTPUT_NAMES[0]
	_write_csv(comparison_path, comparison)
	json_path = config.output_dir / END_TO_END_OUTPUT_NAMES[1]
	_write_json(json_path, payload)
	markdown_path = config.output_dir / END_TO_END_OUTPUT_NAMES[2]
	markdown_path.write_text(
		_end_to_end_markdown(aggregates, comparison), encoding='utf-8'
	)
	return comparison_path, json_path, markdown_path


def summarize_channel_four_way(
	end_to_end_config: ChannelEndToEndSummaryConfig,
	frozen_config: ChannelSummaryConfig,
) -> tuple[Path, Path, Path]:
	"""Write a descriptive table of the two distinct paired experiments."""
	end_to_end, frozen = inspect_channel_four_way_results(
		end_to_end_config, frozen_config
	)
	_require_outputs_absent(
		end_to_end_config.four_way_output_dir, FOUR_WAY_OUTPUT_NAMES
	)
	comparison: list[dict[str, object]] = []
	for data_size in DATA_SIZE_PREFIX:
		for layout_id in LAYOUT_IDS:
			frozen_pretrained = _test_iou(
				frozen[('pretrained', layout_id, data_size)],
				Path(f'frozen/pretrained/{layout_id}/{data_size}'),
			)
			frozen_random = _test_iou(
				frozen[('random', layout_id, data_size)],
				Path(f'frozen/random/{layout_id}/{data_size}'),
			)
			finetune = _test_iou(
				end_to_end[('pretrained', layout_id, data_size)],
				Path(f'end-to-end/pretrained/{layout_id}/{data_size}'),
			)
			scratch = _test_iou(
				end_to_end[('random', layout_id, data_size)],
				Path(f'end-to-end/random/{layout_id}/{data_size}'),
			)
			comparison.append(
				{
					'data_size': data_size,
					'layout_id': layout_id,
					'frozen_pretrained_channel_iou': frozen_pretrained,
					'frozen_random_channel_iou': frozen_random,
					'finetune_pretrained_channel_iou': finetune,
					'train_from_scratch_channel_iou': scratch,
					'frozen_representation_delta': (
						frozen_pretrained - frozen_random
					),
					'end_to_end_pretraining_delta': finetune - scratch,
				}
			)
	payload = {
		'schema_version': 1,
		'primary_metric': 'test.channel_iou',
		'job_count': {'frozen': len(frozen), 'end_to_end': len(end_to_end)},
		'comparison': comparison,
		'by_size': {
			'frozen_representation_delta': _paired_aggregates(
				comparison, 'frozen_representation_delta'
			),
			'end_to_end_pretraining_delta': _paired_aggregates(
				comparison, 'end_to_end_pretraining_delta'
			),
		},
		'claim_boundary': (
			'Cross-regime score differences do not isolate encoder fine-tuning '
			'because the encoder input context differs.'
		),
	}
	output_dir = end_to_end_config.four_way_output_dir
	output_dir.mkdir(parents=True, exist_ok=True)
	csv_path = output_dir / FOUR_WAY_OUTPUT_NAMES[0]
	_write_csv(csv_path, comparison)
	json_path = output_dir / FOUR_WAY_OUTPUT_NAMES[1]
	_write_json(json_path, payload)
	markdown_path = output_dir / FOUR_WAY_OUTPUT_NAMES[2]
	markdown_path.write_text(_four_way_markdown(comparison), encoding='utf-8')
	return csv_path, json_path, markdown_path


def inspect_channel_four_way_results(
	end_to_end_config: ChannelEndToEndSummaryConfig,
	frozen_config: ChannelSummaryConfig,
) -> tuple[
	dict[tuple[str, str, str], Mapping[str, object]],
	dict[tuple[str, str, str], Mapping[str, object]],
]:
	"""Validate both complete job sets and their shared supervision identity."""
	end_to_end = inspect_channel_end_to_end_results(end_to_end_config)
	frozen = inspect_channel_benchmark_results(frozen_config)
	_validate_four_way_pairing(end_to_end, frozen)
	return end_to_end, frozen


def _validate_end_to_end_job(  # noqa: C901, PLR0912
	payload: Mapping[str, object],
	encoder_init: str,
	layout_id: str,
	data_size: str,
	path: Path,
) -> None:
	expected = {
		'encoder_init': encoder_init,
		'condition_name': (
			'finetune_pretrained'
			if encoder_init == 'pretrained'
			else 'train_from_scratch'
		),
		'layout_id': layout_id,
		'data_size': data_size,
	}
	for key, value in expected.items():
		if payload.get(key) != value:
			raise ValueError(f'{path} has incorrect {key}: {payload.get(key)!r}')
	_test_iou(payload, path)
	identity = _mapping(payload, 'benchmark_identity', str(path))
	if set(identity) != _IDENTITY_KEYS:
		raise ValueError(f'{path} benchmark_identity has invalid fields')
	for key in ('encoder_init', 'layout_id', 'data_size'):
		if identity.get(key) != expected[key]:
			raise ValueError(f'{path} benchmark_identity.{key} mismatch')
	reference = _mapping(identity, 'reference_input', f'{path} benchmark_identity')
	if set(reference) != _REFERENCE_INPUT_KEYS:
		raise ValueError(f'{path} reference_input has invalid fields')
	for key in ('reference_metadata_sha256', 'reference_valid_tokens_sha256'):
		_validate_sha256(reference.get(key), f'{path} reference_input.{key}')
	labels = _mapping(identity, 'labels', f'{path} benchmark_identity')
	if set(labels) != {'path', 'metadata_path', 'prepared_label_identity'}:
		raise ValueError(f'{path} labels identity has invalid fields')
	prepared = _mapping(labels, 'prepared_label_identity', f'{path} labels')
	_validate_sha256(prepared.get('labels_sha256'), f'{path} labels_sha256')
	supervision = _mapping(identity, 'supervision', f'{path} benchmark_identity')
	if set(supervision) != _SUPERVISION_KEYS:
		raise ValueError(f'{path} benchmark supervision has invalid fields')
	_validate_identity_supervision(supervision, path)
	decoder = _mapping(identity, 'decoder', f'{path} benchmark_identity')
	if set(decoder) != {'architecture', 'initial_state_sha256'}:
		raise ValueError(f'{path} decoder identity has invalid fields')
	_validate_sha256(decoder.get('initial_state_sha256'), f'{path} decoder SHA')
	_validate_common_identity_components(identity, path)
	source = _mapping(identity, 'encoder_source', f'{path} benchmark_identity')
	_validate_encoder_source(source, encoder_init, path)
	initial_states = _mapping(
		identity, 'encoder_initial_states', f'{path} benchmark_identity'
	)
	if set(initial_states) != {'pretrained_sha256', 'random_sha256'}:
		raise ValueError(f'{path} encoder_initial_states has invalid fields')
	for key in ('pretrained_sha256', 'random_sha256'):
		_validate_sha256(initial_states.get(key), f'{path} {key}')
	if initial_states['pretrained_sha256'] == initial_states['random_sha256']:
		raise ValueError(f'{path} pretrained/random encoder initial SHA must differ')
	selected = (
		initial_states['pretrained_sha256']
		if encoder_init == 'pretrained'
		else initial_states['random_sha256']
	)
	if source.get('initial_state_sha256') != selected:
		raise ValueError(f'{path} selected encoder initial state SHA mismatch')
	_validate_metrics_redundancy(payload, identity, path)


def _validate_encoder_source(  # noqa: C901
	source: Mapping[str, object], encoder_init: str, path: Path
) -> None:
	role_fields = (
		{'role', 'checkpoint_path', 'checkpoint_sha256', 'model_tag'}
		if encoder_init == 'pretrained'
		else {
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
	)
	expected_fields = role_fields | {
		'model_geometry',
		'parameter_dtype',
		'trainable_modules',
		'initial_state_sha256',
	}
	if set(source) != expected_fields:
		raise ValueError(f'{path} encoder_source has invalid fields for {encoder_init}')
	if source.get('role') != encoder_init:
		raise ValueError(f'{path} encoder role must be {encoder_init}')
	_validate_sha256(source.get('checkpoint_sha256'), f'{path} checkpoint SHA')
	_validate_sha256(source.get('initial_state_sha256'), f'{path} encoder state SHA')
	if source.get('parameter_dtype') != 'float32':
		raise ValueError(f'{path} encoder parameter dtype must be float32')
	if source.get('trainable_modules') != ['patch_projection', 'encoder']:
		raise ValueError(f'{path} encoder trainable modules are invalid')
	checkpoint_path = source.get('checkpoint_path')
	if not isinstance(checkpoint_path, str) or not checkpoint_path:
		raise TypeError(f'{path} encoder checkpoint path must be non-empty')
	geometry = source.get('model_geometry')
	if not isinstance(geometry, Mapping):
		raise TypeError(f'{path} encoder model geometry must be a mapping')
	if encoder_init == 'pretrained' and source.get('model_tag') != (
		CHANNEL_PRETRAINED_MODEL_TAG
	):
		raise ValueError(f'{path} pretrained encoder model tag mismatch')
	if encoder_init == 'random':
		expected = {
			'random_encoder_baseline': True,
			'pretrained_weights_loaded': False,
			'seed': CHANNEL_RANDOM_ENCODER_SEED,
			'checkpoint_kind': 'random_init',
			'reference_model_tag': CHANNEL_PRETRAINED_MODEL_TAG,
		}
		for key, value in expected.items():
			if source.get(key) != value:
				raise ValueError(f'{path} random encoder source {key} mismatch')
		_validate_sha256(
			source.get('reference_checkpoint_sha256'),
			f'{path} random reference checkpoint SHA',
		)


def _validate_common_identity_components(
	identity: Mapping[str, object], path: Path
) -> None:
	expected_fields = {
		'optimizer': {
			'encoder_learning_rate',
			'decoder_learning_rate',
			'weight_decay',
			'parameter_group_names',
		},
		'training': {
			'epochs',
			'batch_size',
			'sampling_mode',
			'seed',
			'gradient_clip_norm',
		},
		'tiles': {'core_size_tokens', 'context_halo_tokens'},
		'runtime': {
			'resolved_device_type',
			'amp_enabled',
			'autocast_dtype',
			'grad_scaler_enabled',
		},
	}
	for field, keys in expected_fields.items():
		value = _mapping(identity, field, f'{path} benchmark_identity')
		if set(value) != keys:
			raise ValueError(f'{path} benchmark_identity.{field} has invalid fields')
	optimizer = _mapping(identity, 'optimizer', str(path))
	if optimizer.get('parameter_group_names') != ['encoder', 'decoder']:
		raise ValueError(f'{path} optimizer parameter groups are invalid')
	tiles = _mapping(identity, 'tiles', str(path))
	if tiles != {
		'core_size_tokens': [8, 8, 8],
		'context_halo_tokens': [1, 1, 1],
	}:
		raise ValueError(f'{path} tile geometry is invalid')


def _validate_identity_supervision(
	supervision: Mapping[str, object], path: Path
) -> None:
	for key in ('train_lines', 'validation_lines'):
		lines = _mapping(supervision, key, f'{path} supervision')
		if set(lines) != {'inline', 'crossline'}:
			raise ValueError(f'{path} supervision.{key} has invalid fields')
		for orientation in ('inline', 'crossline'):
			_indices(lines.get(orientation), f'{path} supervision.{key}.{orientation}')
	_validate_test_definition(
		supervision.get('test_definition'), f'{path} supervision.test_definition'
	)
	counts = _mapping(supervision, 'split_class_counts', f'{path} supervision')
	tiles = _mapping(supervision, 'tile_counts', f'{path} supervision')
	if set(counts) != {'train', 'validation', 'test'} or set(tiles) != {
		'train',
		'validation',
		'test',
	}:
		raise ValueError(f'{path} split/tile counts have invalid fields')
	for split in ('train', 'validation', 'test'):
		_class_counts(counts.get(split), f'{path} {split} class counts')
		if (
			not isinstance(tiles.get(split), int)
			or isinstance(tiles.get(split), bool)
			or int(tiles[split]) <= 0
		):
			raise TypeError(f'{path} {split} tile count must be positive')
	_class_weights(supervision.get('class_weights'), path)


def _validate_metrics_redundancy(
	payload: Mapping[str, object], identity: Mapping[str, object], path: Path
) -> None:
	metric_supervision = _mapping(payload, 'supervision', str(path))
	expected_metric_keys = {
		'axis_mapping',
		'train_inline',
		'train_crossline',
		'validation_inline',
		'validation_crossline',
		'test_definition',
		'split_class_counts',
		'tile_counts',
	}
	if set(metric_supervision) != expected_metric_keys:
		raise ValueError(f'{path} supervision has invalid fields')
	if metric_supervision.get('axis_mapping') != CHANNEL_AXIS_MAPPING:
		raise ValueError(f'{path} supervision axis mapping is invalid')
	identity_supervision = _mapping(
		identity, 'supervision', f'{path} benchmark_identity'
	)
	line_names = {
		'train_lines': 'train',
		'validation_lines': 'validation',
	}
	for identity_key, metric_prefix in line_names.items():
		lines = _mapping(identity_supervision, identity_key, f'{path} supervision')
		for orientation in ('inline', 'crossline'):
			if lines.get(orientation) != metric_supervision.get(
				f'{metric_prefix}_{orientation}'
			):
				raise ValueError(f'{path} supervision line identity mismatch')
	if identity_supervision.get('test_definition') != metric_supervision.get(
		'test_definition'
	):
		raise ValueError(f'{path} supervision test definition identity mismatch')
	for key in ('split_class_counts', 'tile_counts'):
		if identity_supervision.get(key) != metric_supervision.get(key):
			raise ValueError(f'{path} supervision {key} identity mismatch')
	metric_weights = _class_weights(payload.get('class_weights'), path)
	identity_weights = _class_weights(identity_supervision.get('class_weights'), path)
	if metric_weights != identity_weights:
		raise ValueError(f'{path} class weight identity mismatch')


def _validate_end_to_end_collection(
	jobs: Mapping[tuple[str, str, str], Mapping[str, object]], runs_root: Path
) -> None:
	first_key = ('pretrained', LAYOUT_IDS[0], next(iter(DATA_SIZE_PREFIX)))
	common = _identity(jobs[first_key])
	for key, payload in jobs.items():
		identity = _identity(payload)
		path = _end_to_end_metrics_path(runs_root, *key)
		for field in (
			'encoder_initial_states',
			'reference_input',
			'labels',
			'decoder',
			'optimizer',
			'training',
			'tiles',
			'runtime',
		):
			if identity[field] != common[field]:
				raise ValueError(f'{path} benchmark_identity.{field} drift')
		if _mapping(identity, 'supervision', str(path)).get(
			'validation_lines'
		) != _mapping(common, 'supervision', 'common').get('validation_lines'):
			raise ValueError(f'{path} validation supervision drift')
		current_supervision = _mapping(identity, 'supervision', str(path))
		common_supervision = _mapping(common, 'supervision', 'common')
		if current_supervision.get('test_definition') != common_supervision.get(
			'test_definition'
		):
			raise ValueError(f'{path} test supervision drift')
		current_counts = _mapping(
			current_supervision, 'split_class_counts', f'{path} supervision'
		)
		common_counts = _mapping(
			common_supervision, 'split_class_counts', 'common supervision'
		)
		if tuple(current_counts.get(split) for split in ('validation', 'test')) != (
			tuple(common_counts.get(split) for split in ('validation', 'test'))
		):
			raise ValueError(f'{path} validation/test class count drift')
		current_tiles = _mapping(
			current_supervision, 'tile_counts', f'{path} supervision'
		)
		common_tiles = _mapping(
			common_supervision, 'tile_counts', 'common supervision'
		)
		if tuple(current_tiles.get(split) for split in ('validation', 'test')) != (
			tuple(common_tiles.get(split) for split in ('validation', 'test'))
		):
			raise ValueError(f'{path} validation/test tile count drift')
	_validate_encoder_conditions(jobs, runs_root)
	_validate_end_to_end_pairs(jobs)
	_validate_nested_and_unique_layouts(jobs)


def _validate_encoder_conditions(
	jobs: Mapping[tuple[str, str, str], Mapping[str, object]], runs_root: Path
) -> None:
	sources: dict[str, Mapping[str, object]] = {}
	for encoder_init in ENCODER_INITIALIZATIONS:
		first_key = (encoder_init, LAYOUT_IDS[0], next(iter(DATA_SIZE_PREFIX)))
		source = _mapping(_identity(jobs[first_key]), 'encoder_source', 'identity')
		sources[encoder_init] = source
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZE_PREFIX:
				key = (encoder_init, layout_id, data_size)
				current = _mapping(
					_identity(jobs[key]), 'encoder_source', 'identity'
				)
				if current != source:
					raise ValueError(
						f'{_end_to_end_metrics_path(runs_root, *key)} encoder source '
						'does not match its other 15 jobs'
					)
	pretrained = sources['pretrained']
	random = sources['random']
	if pretrained['checkpoint_sha256'] == random['checkpoint_sha256']:
		raise ValueError('pretrained and random checkpoint SHA-256 must differ')
	for field in ('model_geometry', 'trainable_modules', 'parameter_dtype'):
		if pretrained[field] != random[field]:
			raise ValueError(f'pretrained/random encoder {field} mismatch')
	if random.get('reference_checkpoint') != pretrained.get('checkpoint_path'):
		raise ValueError('random reference checkpoint does not match pretrained source')
	if random.get('reference_checkpoint_sha256') != pretrained.get(
		'checkpoint_sha256'
	):
		raise ValueError(
			'random reference checkpoint SHA does not match pretrained source'
		)


def _validate_end_to_end_pairs(
	jobs: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> None:
	for layout_id in LAYOUT_IDS:
		for data_size in DATA_SIZE_PREFIX:
			pretrained = jobs[('pretrained', layout_id, data_size)]
			random = jobs[('random', layout_id, data_size)]
			pre_identity = _identity(pretrained)
			random_identity = _identity(random)
			pre_supervision = _mapping(pre_identity, 'supervision', 'identity')
			random_supervision = _mapping(random_identity, 'supervision', 'identity')
			if pre_supervision != random_supervision:
				raise ValueError(
					f'{layout_id}/{data_size} end-to-end supervision mismatch'
				)
			if pre_identity['decoder'] != random_identity['decoder']:
				raise ValueError(
					f'{layout_id}/{data_size} decoder initial state mismatch'
				)
			if _tile_order_contract(pre_identity) != _tile_order_contract(
				random_identity
			):
				raise ValueError(f'{layout_id}/{data_size} tile order mismatch')


def _validate_nested_and_unique_layouts(
	jobs: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> None:
	for encoder_init in ENCODER_INITIALIZATIONS:
		for layout_id in LAYOUT_IDS:
			for orientation in ('inline', 'crossline'):
				large_lines = _train_lines(
					jobs[(encoder_init, layout_id, 'large')], orientation
				)
				for data_size, prefix in DATA_SIZE_PREFIX.items():
					current = _train_lines(
						jobs[(encoder_init, layout_id, data_size)], orientation
					)
					if len(current) != prefix or current != large_lines[:prefix]:
						raise ValueError(
							f'{encoder_init}/{layout_id} {orientation} training '
							'lines are not nested'
						)
		for data_size in DATA_SIZE_PREFIX:
			seen: set[tuple[frozenset[int], frozenset[int]]] = set()
			for layout_id in LAYOUT_IDS:
				key = (
					frozenset(
						_train_lines(
							jobs[(encoder_init, layout_id, data_size)], 'inline'
						)
					),
					frozenset(
						_train_lines(
							jobs[(encoder_init, layout_id, data_size)], 'crossline'
						)
					),
				)
				if key in seen:
					raise ValueError(
						f'{encoder_init}/{data_size} layout training sections '
						'must be unique'
					)
				seen.add(key)


def _validate_four_way_pairing(
	end_to_end: Mapping[tuple[str, str, str], Mapping[str, object]],
	frozen: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> None:
	for layout_id in LAYOUT_IDS:
		for data_size in DATA_SIZE_PREFIX:
			frozen_payload = frozen[('pretrained', layout_id, data_size)]
			end_payload = end_to_end[('pretrained', layout_id, data_size)]
			frozen_identity = _identity(frozen_payload)
			end_identity = _identity(end_payload)
			frozen_label = frozen_identity.get('prepared_label_identity')
			end_label = _mapping(end_identity, 'labels', 'end-to-end identity').get(
				'prepared_label_identity'
			)
			if frozen_label != end_label:
				raise ValueError(f'{layout_id}/{data_size} label identity mismatch')
			if _normalized_frozen_supervision(frozen_payload, frozen_identity) != (
				_normalized_end_to_end_supervision(end_payload, end_identity)
			):
				raise ValueError(
					f'{layout_id}/{data_size} four-way supervision mismatch'
				)
			frozen_decoder = _mapping(frozen_identity, 'decoder', 'frozen identity')
			end_decoder = _mapping(
				_mapping(end_identity, 'decoder', 'end-to-end identity'),
				'architecture',
				'end-to-end decoder',
			)
			if frozen_decoder.get('class_count') != end_decoder.get('class_count'):
				raise ValueError(f'{layout_id}/{data_size} class count mismatch')
			if _class_weights(frozen_payload.get('class_weights'), Path('frozen')) != (
				_class_weights(end_payload.get('class_weights'), Path('end-to-end'))
			):
				raise ValueError(f'{layout_id}/{data_size} class weight mismatch')


def _normalized_frozen_supervision(
	payload: Mapping[str, object], identity: Mapping[str, object]
) -> dict[str, object]:
	supervision = _mapping(payload, 'supervision', 'frozen metrics')
	return {
		'axis_mapping': supervision.get('axis_mapping'),
		'train_inline': supervision.get('train_inline'),
		'train_crossline': supervision.get('train_crossline'),
		'validation_inline': supervision.get('validation_inline'),
		'validation_crossline': supervision.get('validation_crossline'),
		'test_definition': supervision.get('test_definition'),
		'split_class_counts': supervision.get('split_class_counts'),
		'tile_counts': identity.get('tile_counts'),
	}


def _normalized_end_to_end_supervision(
	payload: Mapping[str, object], identity: Mapping[str, object]
) -> dict[str, object]:
	supervision = _mapping(payload, 'supervision', 'end-to-end metrics')
	identity_supervision = _mapping(identity, 'supervision', 'end-to-end identity')
	return {
		'axis_mapping': supervision.get('axis_mapping'),
		'train_inline': supervision.get('train_inline'),
		'train_crossline': supervision.get('train_crossline'),
		'validation_inline': supervision.get('validation_inline'),
		'validation_crossline': supervision.get('validation_crossline'),
		'test_definition': supervision.get('test_definition'),
		'split_class_counts': supervision.get('split_class_counts'),
		'tile_counts': identity_supervision.get('tile_counts'),
	}


def _end_to_end_comparison(
	jobs: Mapping[tuple[str, str, str], Mapping[str, object]], runs_root: Path
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	for data_size in DATA_SIZE_PREFIX:
		for layout_id in LAYOUT_IDS:
			finetune = _test_iou(
				jobs[('pretrained', layout_id, data_size)],
				_end_to_end_metrics_path(
					runs_root, 'pretrained', layout_id, data_size
				),
			)
			scratch = _test_iou(
				jobs[('random', layout_id, data_size)],
				_end_to_end_metrics_path(
					runs_root, 'random', layout_id, data_size
				),
			)
			rows.append(
				{
					'data_size': data_size,
					'layout_id': layout_id,
					'finetune_pretrained_channel_iou': finetune,
					'train_from_scratch_channel_iou': scratch,
					'end_to_end_pretraining_delta': finetune - scratch,
				}
			)
	return rows


def _paired_aggregates(
	comparison: Sequence[Mapping[str, object]], delta_key: str
) -> dict[str, object]:
	result: dict[str, object] = {}
	for data_size in DATA_SIZE_PREFIX:
		rows = [row for row in comparison if row.get('data_size') == data_size]
		deltas = [float(row[delta_key]) for row in rows]
		result[data_size] = {
			'paired_mean': statistics.fmean(deltas),
			'paired_median': statistics.median(deltas),
			'sample_standard_deviation': statistics.stdev(deltas),
			'pretrained_wins': sum(delta > 0 for delta in deltas),
			'ties': sum(delta == 0 for delta in deltas),
			'pretrained_losses': sum(delta < 0 for delta in deltas),
			'layout_deltas': {
				str(row['layout_id']): float(row[delta_key]) for row in rows
			},
		}
	return result


def _end_to_end_markdown(
	aggregates: Mapping[str, object], comparison: Sequence[Mapping[str, object]]
) -> str:
	lines = [
		'# Parihaka Channel end-to-end initialization comparison',
		'',
		(
			'Primary metric: test Channel IoU. Deltas are finetune pretrained '
			'minus train from scratch.'
		),
		'',
		'| size | paired mean | paired median | sample std | wins/ties/losses |',
		'|---|---:|---:|---:|---:|',
	]
	for data_size in DATA_SIZE_PREFIX:
		row = _mapping(aggregates, data_size, 'aggregates')
		lines.append(
			f'| {data_size} | {float(row["paired_mean"]):.6f} | '
			f'{float(row["paired_median"]):.6f} | '
			f'{float(row["sample_standard_deviation"]):.6f} | '
			f'{row["pretrained_wins"]}/{row["ties"]}/'
			f'{row["pretrained_losses"]} |'
		)
	lines.extend(
		[
			'',
			'| size | layout | finetune pretrained | train from scratch | delta |',
			'|---|---|---:|---:|---:|',
		]
	)
	lines.extend(
		f'| {row["data_size"]} | {row["layout_id"]} | '
		f'{float(row["finetune_pretrained_channel_iou"]):.6f} | '
		f'{float(row["train_from_scratch_channel_iou"]):.6f} | '
		f'{float(row["end_to_end_pretraining_delta"]):.6f} |'
		for row in comparison
	)
	return '\n'.join(lines) + '\n'


def _four_way_markdown(comparison: Sequence[Mapping[str, object]]) -> str:
	lines = [
		'# Parihaka Channel four-condition comparison',
		'',
		(
			'Frozen representation delta and end-to-end pretraining delta answer '
			'different scientific questions. Cross-regime score differences do not '
			'isolate encoder fine-tuning because the encoder input context differs.'
		),
		'',
		(
			'Frozen jobs use 128^3 overlap embeddings; end-to-end jobs use an 80^3 '
			'raw-amplitude encoder crop. This is a descriptive table of two paired '
			'experiments.'
		),
		'',
		(
			'| size | layout | frozen pretrained | frozen random | finetune '
			'pretrained | train from scratch | frozen delta | end-to-end delta |'
		),
		'|---|---|---:|---:|---:|---:|---:|---:|',
	]
	lines.extend(
		f'| {row["data_size"]} | {row["layout_id"]} | '
		f'{float(row["frozen_pretrained_channel_iou"]):.6f} | '
		f'{float(row["frozen_random_channel_iou"]):.6f} | '
		f'{float(row["finetune_pretrained_channel_iou"]):.6f} | '
		f'{float(row["train_from_scratch_channel_iou"]):.6f} | '
		f'{float(row["frozen_representation_delta"]):.6f} | '
		f'{float(row["end_to_end_pretraining_delta"]):.6f} |'
		for row in comparison
	)
	return '\n'.join(lines) + '\n'


def _train_lines(
	payload: Mapping[str, object], orientation: str
) -> tuple[int, ...]:
	identity = _identity(payload)
	supervision = _mapping(identity, 'supervision', 'benchmark_identity')
	lines = _mapping(supervision, 'train_lines', 'benchmark_identity.supervision')
	return _indices(lines.get(orientation), f'train_lines.{orientation}')


def _tile_order_contract(identity: Mapping[str, object]) -> tuple[object, object]:
	training = _mapping(identity, 'training', 'benchmark_identity')
	return training.get('sampling_mode'), training.get('seed')


def _identity(payload: Mapping[str, object]) -> Mapping[str, object]:
	return _mapping(payload, 'benchmark_identity', 'metrics')


def _end_to_end_metrics_path(
	root: Path, encoder_init: str, layout_id: str, data_size: str
) -> Path:
	return (
		root
		/ f'encoder_init={encoder_init}'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
		/ 'metrics.json'
	)


def _test_iou(payload: Mapping[str, object], path: Path) -> float:
	test = _mapping(payload, 'test', str(path))
	value = test.get('channel_iou')
	if (
		not isinstance(value, int | float)
		or isinstance(value, bool)
		or not math.isfinite(float(value))
	):
		raise TypeError(f'{path} test.channel_iou must be finite and numeric')
	return float(value)


def _class_weights(value: object, path: Path) -> tuple[float, float]:
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
		raise TypeError(f'{path} class weights must contain two finite numbers')
	return float(value[0]), float(value[1])


def _class_counts(value: object, label: str) -> tuple[int, int]:
	if (
		not isinstance(value, list)
		or len(value) != 2
		or any(
			not isinstance(item, int) or isinstance(item, bool) or item <= 0
			for item in value
		)
	):
		raise TypeError(f'{label} must contain two positive integers')
	return value[0], value[1]


def _validate_test_definition(value: object, label: str) -> None:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	expected = {'mode', 'reserved_large_inline', 'reserved_large_crossline'}
	if set(value) != expected:
		raise ValueError(f'{label} has invalid fields')
	if value.get('mode') != CHANNEL_TEST_MODE:
		raise ValueError(f'{label}.mode must be {CHANNEL_TEST_MODE!r}')
	for key in ('reserved_large_inline', 'reserved_large_crossline'):
		indices = _indices(value.get(key), f'{label}.{key}')
		if indices != tuple(sorted(indices)):
			raise ValueError(f'{label}.{key} must be sorted')


def _indices(value: object, label: str) -> tuple[int, ...]:
	if not isinstance(value, list) or not value:
		raise TypeError(f'{label} must be a non-empty list')
	if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
		raise TypeError(f'{label} must contain integers')
	items = tuple(value)
	if len(set(items)) != len(items):
		raise ValueError(f'{label} must not contain duplicates')
	return items


def _validate_sha256(value: object, label: str) -> None:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise TypeError(f'{label} must be a lowercase SHA-256 digest')


def _require_outputs_absent(output_dir: Path, names: Sequence[str]) -> None:
	existing = [output_dir / name for name in names if (output_dir / name).exists()]
	if existing:
		raise FileExistsError(
			f'Channel summary outputs already exist: {", ".join(map(str, existing))}'
		)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=tuple(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)


def _read_json(path: Path) -> Mapping[str, object]:
	value = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(value, Mapping):
		raise TypeError(f'metrics must contain an object: {path}')
	return value


def _mapping(
	value: Mapping[str, object], key: str, prefix: str
) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{prefix}.{key} must be a mapping')
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
	'ENCODER_INITIALIZATIONS',
	'END_TO_END_OUTPUT_NAMES',
	'FOUR_WAY_OUTPUT_NAMES',
	'ChannelEndToEndSummaryConfig',
	'channel_end_to_end_summary_config_from_mapping',
	'inspect_channel_end_to_end_results',
	'inspect_channel_four_way_results',
	'summarize_channel_end_to_end',
	'summarize_channel_four_way',
]
