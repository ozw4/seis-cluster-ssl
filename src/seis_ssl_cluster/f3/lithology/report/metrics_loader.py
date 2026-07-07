"""Metrics and metadata loading helpers for F3 lithology reports."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.f3.lithology.report._common import (
	OVERALL_METRIC_COLUMNS,
	_agc_enabled,
	_class_imbalance,
	_class_name,
	_combined_counts,
	_embed_spec,
	_first_non_empty,
	_float_or_none,
	_fraction_or_none,
	_int_or_none,
	_mapping,
	_mask_ratio,
	_prefer_mapping,
	_sequence_of_mappings,
	_string_or_none,
	_sum_ints,
	_visible_loss_enabled,
)
from seis_ssl_cluster.f3.lithology.token_dataset import (
	F3LithologyTokenDataset,
	load_f3_lithology_token_dataset,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.f3.lithology.report._core import F3LithologyReportConfig


def _dataset_summary(
	config: F3LithologyReportConfig,
	token_metadata: Mapping[str, object],
	classes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
	geometry = _mapping(token_metadata.get('geometry'))
	summary = _mapping(token_metadata.get('summary'))
	return {
		'name': _first_non_empty(
			config.dataset.get('name'),
			_mapping(token_metadata.get('dataset')).get('name'),
		),
		'version': _first_non_empty(
			config.dataset.get('version'),
			_mapping(token_metadata.get('dataset')).get('version'),
		),
		'f3_shape': geometry.get('shape_xyz'),
		'classes': [dict(item) for item in classes],
		'train_validation_slices': _slice_summary(token_metadata),
		'tokenization_thresholds': dict(_mapping(token_metadata.get('tokenization'))),
		'class_imbalance': _class_imbalance(
			_combined_counts(
				_mapping(summary.get('train_class_counts')),
				_mapping(summary.get('validation_class_counts')),
			),
		),
		'label_source_of_truth': _first_non_empty(
			token_metadata.get('label_source_of_truth'),
			'segy_label_volume',
		),
		'png_label_role': _first_non_empty(
			config.labels.get('png_label_role'),
			token_metadata.get('png_label_role'),
		),
	}

def _pretrained_summary(
	config: F3LithologyReportConfig,
	probe_config: Mapping[str, object],
) -> dict[str, object]:
	model = _prefer_mapping(config.model, _mapping(probe_config.get('model')))
	model_tag = _string_or_none(model.get('tag'))
	return {
		'MODEL_TAG': model_tag,
		'checkpoint_path': model.get('checkpoint'),
		'EMBED_SPEC': _embed_spec(config.lithology, probe_config),
		'agc_enabled': _agc_enabled(model),
		'visible_loss_enabled': _visible_loss_enabled(model_tag),
		'mask_ratio': _mask_ratio(model_tag),
		'freeze_encoder': model.get('freeze_encoder'),
	}

def _token_dataset_summary(
	probe_config: Mapping[str, object],
	token_metadata: Mapping[str, object],
	*,
	token_datasets: Mapping[str, F3LithologyTokenDataset] | None = None,
) -> dict[str, object]:
	token_summary = _mapping(token_metadata.get('summary'))
	probe_summary = _mapping(probe_config.get('summary'))
	dataset_summary = _loaded_token_dataset_summary(token_datasets)
	train_counts = _prefer_mapping(
		_mapping(dataset_summary.get('train_class_counts')),
		_mapping(token_summary.get('train_class_counts')),
		_mapping(probe_summary.get('train_class_counts')),
	)
	validation_counts = _prefer_mapping(
		_mapping(dataset_summary.get('validation_class_counts')),
		_mapping(token_summary.get('validation_class_counts')),
		_mapping(probe_summary.get('validation_class_counts')),
	)
	retained = _int_or_none(
		_first_non_empty(
			dataset_summary.get('all_labeled_tokens'),
			token_summary.get('all_labeled_tokens'),
			_sum_ints((token_summary.get('train_tokens'), token_summary.get(
				'validation_tokens',
			))),
		),
	)
	dropped = _int_or_none(token_summary.get('total_dropped_tokens'))
	ambiguous = _int_or_none(token_summary.get('total_ambiguous_tokens'))
	total = None if retained is None or dropped is None else retained + dropped
	return {
		'train_token_count': _first_non_empty(
			dataset_summary.get('train_tokens'),
			token_summary.get('train_tokens'),
			probe_summary.get('train_tokens'),
		),
		'validation_token_count': _first_non_empty(
			dataset_summary.get('validation_tokens'),
			token_summary.get('validation_tokens'),
			probe_summary.get('validation_tokens'),
		),
		'class_counts': {
			'train': dict(train_counts),
			'validation': dict(validation_counts),
			'combined': _combined_counts(train_counts, validation_counts),
		},
		'total_dropped_tokens': dropped,
		'total_ambiguous_tokens': ambiguous,
		'dropped_token_ratio': _fraction_or_none(dropped, total),
		'ambiguous_token_ratio': _fraction_or_none(ambiguous, total),
		'class_imbalance': _class_imbalance(
			_combined_counts(train_counts, validation_counts),
		),
	}

def _load_probe_token_datasets(
	probe_config: Mapping[str, object],
) -> tuple[dict[str, F3LithologyTokenDataset], list[str]]:
	paths = _probe_token_dataset_paths(probe_config)
	if not paths:
		return {}, []
	datasets: dict[str, F3LithologyTokenDataset] = {}
	warnings: list[str] = []
	for split, path in paths.items():
		try:
			datasets[split] = load_f3_lithology_token_dataset(path)
		except (OSError, KeyError, TypeError, ValueError) as exc:  # noqa: PERF203
			warnings.append(
				f'unable to load {split} token dataset component: {path} ({exc})',
			)
	return datasets, warnings

def _probe_token_dataset_paths(
	probe_config: Mapping[str, object],
) -> dict[str, Path]:
	inputs = _mapping(probe_config.get('inputs'))
	paths: dict[str, Path] = {}
	for split, key in (
		('train', 'train_tokens'),
		('validation', 'validation_tokens'),
	):
		value = inputs.get(key)
		if isinstance(value, str) and value:
			paths[split] = Path(value)
	return paths

def _loaded_token_dataset_summary(
	token_datasets: Mapping[str, F3LithologyTokenDataset] | None,
) -> dict[str, object]:
	if not token_datasets:
		return {}
	train = token_datasets.get('train')
	validation = token_datasets.get('validation')
	summary: dict[str, object] = {}
	if train is not None:
		summary['train_tokens'] = train.count
		summary['train_class_counts'] = _token_dataset_class_counts(train)
	if validation is not None:
		summary['validation_tokens'] = validation.count
		summary['validation_class_counts'] = _token_dataset_class_counts(validation)
	if train is not None and validation is not None:
		summary['all_labeled_tokens'] = train.count + validation.count
	return summary

def _token_dataset_class_counts(
	dataset: F3LithologyTokenDataset,
) -> dict[int, int]:
	return {
		int(class_id): int(count)
		for class_id, count in sorted(
			Counter(int(label) for label in dataset.labels).items(),
		)
	}

def _probe_summary(
	config: F3LithologyReportConfig,
	probe_config: Mapping[str, object],
) -> dict[str, object]:
	probe = {
		**_mapping(probe_config.get('probe')),
		**config.probe,
	}
	hyperparameters = {
		key: value
		for key, value in probe.items()
		if key
		not in {
			'spec',
			'type',
			'feature_scaling',
			'class_weight',
			'output_dir',
			'metrics_json',
		}
	}
	return {
		'PROBE_SPEC': probe.get('spec'),
		'classifier_type': probe.get('type'),
		'feature_scaling': probe.get('feature_scaling'),
		'class_weighting': probe.get('class_weight'),
		'hyperparameters': hyperparameters,
		'training_summary': dict(_mapping(probe_config.get('training_summary'))),
	}

def _metrics_summary(
	metrics: Mapping[str, object],
	classes: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[str]]:
	if not metrics:
		return {
			'available': False,
			'overall': {},
			'per_class': [],
			'confusion_matrix': None,
			'missing': list(OVERALL_METRIC_COLUMNS),
		}, []
	warnings: list[str] = []
	missing = [
		key
		for key in (
			*OVERALL_METRIC_COLUMNS,
			'per_class_f1',
			'per_class_iou',
			'confusion_matrix',
		)
		if key not in metrics
	]
	if missing:
		warnings.append(f'metrics missing required key(s): {", ".join(missing)}')
	overall = {
		key: _float_or_none(metrics.get(key)) for key in OVERALL_METRIC_COLUMNS
	}
	return {
		'available': True,
		'overall': overall,
		'per_class': _per_class_metrics(metrics, classes),
		'confusion_matrix': metrics.get('confusion_matrix'),
		'missing': missing,
	}, warnings

def _read_json_component(
	name: str,
	path: Path,
	warnings: list[str],
) -> Mapping[str, object] | None:
	if not path.is_file():
		warnings.append(f'missing input report component: {name} ({path})')
		return None
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		warnings.append(
			f'invalid input report component JSON: {name} ({path}): {exc.msg}',
		)
		return None
	if not isinstance(payload, Mapping):
		warnings.append(f'input report component is not a JSON object: {name} ({path})')
		return None
	return payload

def _read_optional_component(
	name: str,
	path: Path | None,
	warnings: list[str],
) -> Mapping[str, object] | None:
	if path is None:
		return None
	return _read_json_component(name, path, warnings)

def _read_optional_json(path: Path) -> Mapping[str, object] | None:
	if not path.is_file():
		return None
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError:
		return None
	return payload if isinstance(payload, Mapping) else None

def _token_dataset_metadata_path(
	config: F3LithologyReportConfig,
	probe_config: Mapping[str, object],
) -> Path | None:
	if config.token_dataset_metadata_json is not None:
		return config.token_dataset_metadata_json
	value = _mapping(probe_config.get('inputs')).get('token_dataset_metadata_json')
	return Path(value) if isinstance(value, str) and value else None

def _classes(
	probe_config: Mapping[str, object],
	token_metadata: Mapping[str, object],
	metrics: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
	for payload in (probe_config, token_metadata):
		classes = tuple(_sequence_of_mappings(payload.get('classes')))
		if classes:
			return classes
	class_names = _mapping(metrics.get('class_names'))
	class_ids = metrics.get('class_ids')
	if isinstance(class_ids, Sequence) and not isinstance(class_ids, str | bytes):
		return tuple(
			{
				'class_id': class_id,
				'class_name': class_names.get(str(class_id), f'class_{class_id}'),
			}
			for class_id in class_ids
		)
	return ()

def _slice_summary(token_metadata: Mapping[str, object]) -> dict[str, list[str]]:
	result = {'train': [], 'validation': []}
	for item in _sequence_of_mappings(token_metadata.get('slices')):
		split = item.get('split')
		if split not in result:
			continue
		result[split].append(
			f"{item.get('slice_type')} {item.get('slice_index')}",
		)
	return result

def _per_class_metrics(
	metrics: Mapping[str, object],
	classes: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
	f1 = _mapping(metrics.get('per_class_f1'))
	iou = _mapping(metrics.get('per_class_iou'))
	precision = _mapping(metrics.get('per_class_precision'))
	recall = _mapping(metrics.get('per_class_recall'))
	support = _mapping(metrics.get('per_class_support'))
	if not classes:
		classes = tuple({'class_id': key, 'class_name': f'class_{key}'} for key in f1)
	rows = []
	for item in classes:
		class_id = item.get('class_id')
		key = str(class_id)
		rows.append(
			{
				'class_id': class_id,
				'class_name': _class_name(item),
				'precision': _float_or_none(precision.get(key)),
				'recall': _float_or_none(recall.get(key)),
				'f1': _float_or_none(f1.get(key)),
				'iou': _float_or_none(iou.get(key)),
				'support': _int_or_none(support.get(key)),
			},
		)
	return rows

def _read_required_json_object(path: Path, name: str) -> Mapping[str, object]:
	if not path.is_file():
		msg = f'required publish source does not exist: {path}'
		raise FileNotFoundError(msg)
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		msg = f'{name} JSON is invalid: {path}: {exc.msg}'
		raise ValueError(msg) from exc
	if not isinstance(payload, Mapping):
		msg = f'{name} JSON must be an object: {path}'
		raise TypeError(msg)
	return payload

__all__ = [
	'OVERALL_METRIC_COLUMNS',
	'_classes',
	'_dataset_summary',
	'_load_probe_token_datasets',
	'_loaded_token_dataset_summary',
	'_metrics_summary',
	'_per_class_metrics',
	'_pretrained_summary',
	'_probe_summary',
	'_probe_token_dataset_paths',
	'_read_json_component',
	'_read_optional_component',
	'_read_optional_json',
	'_read_required_json_object',
	'_token_dataset_class_counts',
	'_token_dataset_metadata_path',
	'_token_dataset_summary',
]
