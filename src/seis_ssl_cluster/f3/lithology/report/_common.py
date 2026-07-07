"""Shared helpers for F3 lithology report modules."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

OVERALL_METRIC_COLUMNS = (
	'accuracy',
	'balanced_accuracy',
	'macro_f1',
	'weighted_f1',
	'mean_iou',
)
COMPARISON_ID_COLUMNS = (
	'feature_kind',
	'MODEL_TAG',
	'BASELINE_TAG',
	'EMBED_SPEC',
	'LABEL_SET',
	'PROBE_SPEC',
	'FEATURE_SOURCE_KIND',
	'FEATURE_SOURCE_REFERENCE_MODEL_TAG',
	'FEATURE_SOURCE_EMBED_SPEC',
	'FEATURE_SOURCE_DESCRIPTION',
)
COMPARISON_FIGURE_NAMES = (
	'macro_f1_comparison',
	'mean_iou_comparison',
	'per_class_f1_comparison',
)
_COMPARISON_FEATURE_KIND_ORDER = {
	'pretrained_encoder': 0,
	'z_only': 1,
	'xyz_coordinates': 2,
	'amplitude_stats': 3,
	'random_encoder': 4,
}
_BASELINE_FEATURE_KINDS = frozenset(
	{
		'z_only',
		'xyz_coordinates',
		'amplitude_stats',
		'random_encoder',
	},
)
_DEFAULT_PROBE_FIGURES = (
	('confusion_matrix', Path('figures/confusion_matrix.png')),
	('per_class_f1', Path('figures/per_class_f1.png')),
)

def _embed_spec(
	lithology: Mapping[str, object],
	probe_config: Mapping[str, object],
) -> str | None:
	return _first_non_empty(
		_embed_spec_from_config(probe_config),
		_embed_spec_from_lithology_root(lithology.get('root')),
	)

def _embed_spec_from_config(probe_config: Mapping[str, object]) -> str | None:
	embeddings = _mapping(probe_config.get('embeddings'))
	for key in ('spec', 'embed_spec', 'name'):
		value = embeddings.get(key)
		if isinstance(value, str) and value:
			return value
	lithology = _mapping(probe_config.get('lithology'))
	return _embed_spec_from_lithology_root(lithology.get('root'))

def _embed_spec_from_lithology_root(value: object) -> str | None:
	if not isinstance(value, str) or not value:
		return None
	parts = Path(value).parts
	if 'facies_benchmark_v1' not in parts:
		return None
	index = parts.index('facies_benchmark_v1')
	if len(parts) <= index + 2:
		return None
	return parts[index + 2]

def _agc_enabled(model: Mapping[str, object]) -> bool | None:
	agc = _mapping(model.get('amplitude_agc'))
	if isinstance(agc.get('enabled'), bool):
		return bool(agc['enabled'])
	tag = _string_or_none(model.get('tag'))
	if tag is None:
		return None
	return '_agc' in tag

def _visible_loss_enabled(model_tag: str | None) -> bool | None:
	if model_tag is None:
		return None
	match = re.search(r'_vis(\d+)', model_tag)
	if match is None:
		return None
	return int(match.group(1)) > 0

def _mask_ratio(model_tag: str | None) -> float | None:
	if model_tag is None:
		return None
	match = re.search(r'_m(\d{3})_', model_tag)
	if match is None:
		return None
	return int(match.group(1)) / 100.0

def _class_imbalance(counts: Mapping[str, int]) -> dict[str, object]:
	positive = [value for value in counts.values() if value > 0]
	total = sum(counts.values())
	return {
		'total': total,
		'class_counts': dict(counts),
		'max_to_min_positive_ratio': (
			None if not positive else max(positive) / min(positive)
		),
	}

def _combined_counts(
	left: Mapping[object, object],
	right: Mapping[object, object],
) -> dict[str, int]:
	counts: dict[str, int] = {}
	for source in (left, right):
		for key, value in source.items():
			integer = _int_or_none(value)
			if integer is None:
				continue
			counts[str(key)] = counts.get(str(key), 0) + integer
	return counts

def _run_parts(metrics_path: Path) -> dict[str, str]:
	parts = metrics_path.parts
	if 'facies_benchmark_v1' not in parts:
		return {'PROBE_SPEC': metrics_path.parent.name}
	index = parts.index('facies_benchmark_v1')
	values: dict[str, str] = {}
	if len(parts) > index + 1 and parts[index + 1] == 'baselines':
		if len(parts) > index + 2:
			values['BASELINE_TAG'] = parts[index + 2]
		if len(parts) > index + 3:
			values['LABEL_SET'] = parts[index + 3]
		probe_spec = _probe_spec_from_parts(parts)
		if probe_spec is not None:
			values['PROBE_SPEC'] = probe_spec
		return values
	if len(parts) > index + 1:
		values['MODEL_TAG'] = parts[index + 1]
	if len(parts) > index + 2:
		values['EMBED_SPEC'] = parts[index + 2]
	if len(parts) > index + 3:
		values['LABEL_SET'] = parts[index + 3]
	probe_spec = _probe_spec_from_parts(parts)
	if probe_spec is not None:
		values['PROBE_SPEC'] = probe_spec
	return values

def _probe_spec_from_parts(parts: Sequence[str]) -> str | None:
	if 'probes' not in parts:
		return None
	probe_index = parts.index('probes')
	if len(parts) <= probe_index + 1:
		return None
	return parts[probe_index + 1]

def _class_metric_sort_key(value: str) -> tuple[int, str]:
	match = re.fullmatch(r'class_(\d+)_f1', value)
	if match is None:
		return (10**9, value)
	return (int(match.group(1)), value)

def _prefer_mapping(
	preferred: Mapping[str, object],
	*fallbacks: Mapping[str, object],
) -> Mapping[str, object]:
	if preferred:
		return preferred
	for fallback in fallbacks:
		if fallback:
			return fallback
	return {}

def _mapping(value: object) -> Mapping[str, object]:
	return value if isinstance(value, Mapping) else {}

def _sequence_of_mappings(value: object) -> list[Mapping[str, object]]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		return []
	return [item for item in value if isinstance(item, Mapping)]

def _string_list(value: object) -> list[str]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		return []
	return [item for item in value if isinstance(item, str)]

def _first_non_empty(*values: object) -> object:
	for value in values:
		if value not in (None, ''):
			return value
	return None

def _string_or_none(value: object) -> str | None:
	return value if isinstance(value, str) and value else None

def _float_or_none(value: object) -> float | None:
	if isinstance(value, bool):
		return None
	if isinstance(value, int | float):
		return float(value)
	return None

def _int_or_none(value: object) -> int | None:
	if isinstance(value, bool):
		return None
	if isinstance(value, int):
		return value
	if isinstance(value, float) and value.is_integer():
		return int(value)
	return None

def _sum_ints(values: Sequence[object]) -> int | None:
	total = 0
	for value in values:
		integer = _int_or_none(value)
		if integer is None:
			return None
		total += integer
	return total

def _fraction_or_none(numerator: int | None, denominator: int | None) -> float | None:
	if numerator is None or denominator is None or denominator == 0:
		return None
	return float(numerator / denominator)

def _class_name(item: Mapping[str, object]) -> object:
	return _first_non_empty(item.get('class_name'), item.get('name'))

def _display(value: object) -> str:
	if value is None:
		return '未確認'
	if isinstance(value, float):
		return f'{value:.4f}'
	if isinstance(value, list | tuple):
		return json.dumps(value, ensure_ascii=False)
	if isinstance(value, Mapping):
		return json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
	return str(value)

def _relative_path_for_markdown(path: Path, report_dir: Path) -> str:
	try:
		return os.path.relpath(path, start=report_dir)
	except ValueError:
		return path.as_posix()

def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
	json_path = Path(path)
	json_path.parent.mkdir(parents=True, exist_ok=True)
	json_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + '\n',
		encoding='utf-8',
	)

def _write_text(path: str | Path, text: str) -> None:
	text_path = Path(path)
	text_path.parent.mkdir(parents=True, exist_ok=True)
	text_path.write_text(text, encoding='utf-8')

__all__ = [
	'COMPARISON_FIGURE_NAMES',
	'COMPARISON_ID_COLUMNS',
	'OVERALL_METRIC_COLUMNS',
	'_BASELINE_FEATURE_KINDS',
	'_COMPARISON_FEATURE_KIND_ORDER',
	'_DEFAULT_PROBE_FIGURES',
]
