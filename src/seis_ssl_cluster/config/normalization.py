"""Normalization registry config validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, TypeVar

from seis_ssl_cluster.config.base import _resolve_base
from seis_ssl_cluster.config.common import (
	_is_int,
	_is_number,
	_required_mapping,
	_validate_distinct_paths,
	_validate_non_empty_path,
	_validate_output_path,
	_validate_path,
	_validate_positive_int,
	_validate_positive_number,
	_validate_required_key,
)
from seis_ssl_cluster.config.schema import (
	STAGE_NORMALIZATION_QC,
	STAGE_NORMALIZATION_STATS,
)

Config: TypeAlias = dict[str, object]
_T = TypeVar('_T', bound=Mapping[str, object])

_FIXED_DISABLED_NORMALIZATION_KEYS = frozenset(
	{
		'smooth_time_depth_trend_correction',
		'trace_wise_agc',
		'patch_wise_zscore',
	},
)


def resolve_normalization_stats_config(config: _T) -> Config:
	"""Validate and resolve raw config for normalization-stat preparation."""
	resolved, _paths = _resolve_base(config, STAGE_NORMALIZATION_STATS)
	manifests = _required_mapping(resolved, 'manifests')
	_validate_non_empty_path(manifests, 'train', prefix='manifests')
	normalization = _required_mapping(resolved, 'normalization')
	_validate_normalization(normalization)
	return resolved


def resolve_normalization_qc_config(config: _T) -> Config:
	"""Validate and resolve raw config for normalization QC filtering."""
	resolved, paths = _resolve_base(config, STAGE_NORMALIZATION_QC)
	manifests = _required_mapping(resolved, 'manifests')
	splits = _required_mapping(resolved, 'splits')
	qc = _required_mapping(resolved, 'qc')
	source_files = (
		(
			_validate_non_empty_path(manifests, 'input', prefix='manifests'),
			'manifests.input',
		),
		(
			_validate_non_empty_path(splits, 'input', prefix='splits'),
			'splits.input',
		),
	)
	output_files = []
	for parent, key, prefix in (
		(manifests, 'output', 'manifests'),
		(splits, 'output', 'splits'),
		(qc, 'output_json', 'qc'),
		(qc, 'excluded_surveys', 'qc'),
	):
		label = f'{prefix}.{key}'
		path = _validate_path(parent, key, prefix=prefix)
		output_files.append((path, label))
	for output, output_label in output_files:
		for source, source_label in source_files:
			_validate_distinct_paths(
				output,
				output_label,
				source,
				source_label,
			)
	for index, (left, left_label) in enumerate(output_files):
		for right, right_label in output_files[index + 1 :]:
			_validate_distinct_paths(left, left_label, right, right_label)
	for path, label in output_files:
		_validate_output_path(
			path,
			label,
			input_root=paths.nopims_root,
			input_root_label='paths.nopims_root',
		)
	for key in ('min_iqr', 'max_normalized_abs'):
		_validate_required_key(qc, key, prefix='qc')
		_validate_positive_number(qc, key, prefix='qc')
	return resolved


def _validate_normalization(normalization: Mapping[str, object]) -> None:
	for key in ('clipping_percentiles', 'epsilon', 'max_samples', 'seed'):
		_validate_required_key(normalization, key, prefix='normalization')
	value = normalization.get('clipping_percentiles')
	if (
		not isinstance(value, list)
		or len(value) != 2
		or not all(_is_number(item) for item in value)
		or float(value[0]) >= float(value[1])
	):
		msg = 'normalization.clipping_percentiles must be two increasing numbers'
		raise ValueError(msg)
	_validate_positive_number(normalization, 'epsilon', prefix='normalization')
	_validate_positive_int(normalization, 'max_samples', prefix='normalization')
	if not _is_int(normalization.get('seed')):
		msg = (
			'normalization.seed must be an integer; '
			f'got {normalization.get("seed")!r}'
		)
		raise ValueError(msg)
	for key in sorted(_FIXED_DISABLED_NORMALIZATION_KEYS):
		if key in normalization:
			msg = (
				f'normalization.{key} is fixed disabled by the amplitude-only '
				'implementation contract and must be removed from raw YAML.'
			)
			raise ValueError(msg)


__all__ = [
	'resolve_normalization_qc_config',
	'resolve_normalization_stats_config',
]
