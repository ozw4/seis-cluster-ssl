"""Validation and resolution for cluster visualization configs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, TypeVar

from seis_ssl_cluster.config.base import _resolve_base
from seis_ssl_cluster.config.common import (
	_required_child_mapping,
	_required_mapping,
	_validate_allowed_keys,
	_validate_bool,
	_validate_fraction,
	_validate_non_empty_str,
	_validate_nonnegative_finite_number,
	_validate_nonnegative_int_list,
	_validate_output_path,
	_validate_path,
	_validate_positive_int,
	_validate_required_keys,
)
from seis_ssl_cluster.config.schema import STAGE_CLUSTER_VISUALIZATION

Config: TypeAlias = dict[str, object]
_T = TypeVar('_T', bound=Mapping[str, object])

_VISUALIZATION_CLUSTERING_KEYS = frozenset({'input_dir'})
_VISUALIZATION_KEYS = frozenset(
	{
		'output_dir',
		'survey_ids',
		'modes',
		'reconstruct_voxel',
		'allow_all_surveys_for_voxel_reconstruction',
		'skip_existing_voxel_labels',
		'max_voxel_output_gib',
		'allow_large_voxel_output',
		'slice_coordinate_space',
		'xy_slices',
		'xz_slices',
		'dpi',
		'invalid_color',
		'amplitude_underlay',
		'amplitude_comparison',
		'summaries',
	},
)
_VISUALIZATION_REQUIRED_KEYS = _VISUALIZATION_KEYS
_VISUALIZATION_UNDERLAY_KEYS = frozenset({'enabled', 'alpha'})
_VISUALIZATION_COMPARISON_KEYS = frozenset({'enabled', 'alpha'})
_VISUALIZATION_SUMMARY_KEYS = frozenset({'enabled', 'include_amplitude_norm'})


def resolve_cluster_visualization_config(config: _T) -> Config:
	"""Validate and resolve raw config for cluster visualization."""
	resolved, paths = _resolve_base(
		config,
		STAGE_CLUSTER_VISUALIZATION,
		require_nopims_root=False,
	)
	clustering = _required_mapping(resolved, 'clustering')
	visualization = _required_mapping(resolved, 'visualization')
	_validate_allowed_keys(
		clustering,
		_VISUALIZATION_CLUSTERING_KEYS,
		prefix='clustering',
	)
	_validate_allowed_keys(
		visualization,
		_VISUALIZATION_KEYS,
		prefix='visualization',
	)
	_validate_required_keys(
		visualization,
		_VISUALIZATION_REQUIRED_KEYS,
		prefix='visualization',
	)
	input_dir = _validate_path(clustering, 'input_dir', prefix='clustering')
	_validate_output_path(
		input_dir,
		'clustering.input_dir',
		input_root=paths.nopims_root,
		input_root_label='paths.nopims_root',
	)
	output_dir = _validate_path(
		visualization,
		'output_dir',
		prefix='visualization',
	)
	_validate_output_path(
		output_dir,
		'visualization.output_dir',
		input_root=paths.nopims_root,
		input_root_label='paths.nopims_root',
	)
	_validate_survey_id_list(visualization)
	_validate_visualization_modes(visualization)
	_validate_bool(visualization, 'reconstruct_voxel', prefix='visualization')
	_validate_bool(
		visualization,
		'allow_all_surveys_for_voxel_reconstruction',
		prefix='visualization',
	)
	_validate_bool(
		visualization,
		'skip_existing_voxel_labels',
		prefix='visualization',
	)
	_validate_nonnegative_finite_number(
		visualization,
		'max_voxel_output_gib',
		prefix='visualization',
	)
	_validate_bool(
		visualization,
		'allow_large_voxel_output',
		prefix='visualization',
	)
	_validate_slice_coordinate_space(visualization)
	_validate_nonnegative_int_list(visualization, 'xy_slices', prefix='visualization')
	_validate_nonnegative_int_list(visualization, 'xz_slices', prefix='visualization')
	_validate_positive_int(visualization, 'dpi', prefix='visualization')
	_validate_non_empty_str(visualization, 'invalid_color', prefix='visualization')
	underlay = _required_child_mapping(
		visualization,
		'amplitude_underlay',
		prefix='visualization',
	)
	_validate_allowed_keys(
		underlay,
		_VISUALIZATION_UNDERLAY_KEYS,
		prefix='visualization.amplitude_underlay',
	)
	_validate_required_keys(
		underlay,
		_VISUALIZATION_UNDERLAY_KEYS,
		prefix='visualization.amplitude_underlay',
	)
	_validate_bool(underlay, 'enabled', prefix='visualization.amplitude_underlay')
	_validate_fraction(underlay, 'alpha', prefix='visualization.amplitude_underlay')
	comparison = _required_child_mapping(
		visualization,
		'amplitude_comparison',
		prefix='visualization',
	)
	_validate_allowed_keys(
		comparison,
		_VISUALIZATION_COMPARISON_KEYS,
		prefix='visualization.amplitude_comparison',
	)
	_validate_required_keys(
		comparison,
		_VISUALIZATION_COMPARISON_KEYS,
		prefix='visualization.amplitude_comparison',
	)
	_validate_bool(
		comparison,
		'enabled',
		prefix='visualization.amplitude_comparison',
	)
	_validate_fraction(
		comparison,
		'alpha',
		prefix='visualization.amplitude_comparison',
	)
	summaries = _required_child_mapping(
		visualization,
		'summaries',
		prefix='visualization',
	)
	_validate_allowed_keys(
		summaries,
		_VISUALIZATION_SUMMARY_KEYS,
		prefix='visualization.summaries',
	)
	_validate_required_keys(
		summaries,
		_VISUALIZATION_SUMMARY_KEYS,
		prefix='visualization.summaries',
	)
	_validate_bool(summaries, 'enabled', prefix='visualization.summaries')
	_validate_bool(
		summaries,
		'include_amplitude_norm',
		prefix='visualization.summaries',
	)
	return resolved


def _validate_survey_id_list(visualization: Mapping[str, object]) -> None:
	value = visualization.get('survey_ids')
	if not isinstance(value, list) or any(
		not isinstance(item, str) or not item
		for item in value
	):
		msg = 'visualization.survey_ids must be a list of non-empty strings'
		raise ValueError(msg)


def _validate_visualization_modes(visualization: Mapping[str, object]) -> None:
	value = visualization.get('modes')
	if (
		not isinstance(value, list)
		or not value
		or any(not isinstance(item, str) for item in value)
	):
		msg = 'visualization.modes must be a non-empty list of strings'
		raise ValueError(msg)
	unknown = sorted(set(value) - {'token', 'voxel'})
	if unknown:
		msg = f'visualization.modes contains unsupported mode(s): {unknown!r}'
		raise ValueError(msg)
	if len(set(value)) != len(value):
		msg = f'visualization.modes must not contain duplicates; got {value!r}'
		raise ValueError(msg)


def _validate_slice_coordinate_space(visualization: Mapping[str, object]) -> None:
	value = visualization.get('slice_coordinate_space')
	if value != 'voxel':
		msg = (
			'visualization.slice_coordinate_space must be "voxel"; '
			f'got {value!r}'
		)
		raise ValueError(msg)


__all__ = ['resolve_cluster_visualization_config']
