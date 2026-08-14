"""F3 facies benchmark inspection config validation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TypeAlias, TypeVar

from seis_ssl_cluster.config.base import (
	_reject_legacy_attribute_config,
	_reject_stage_key,
	_ResolvedPaths,
)
from seis_ssl_cluster.config.common import (
	_validate_absolute_path,
	_validate_allowed_keys,
	_validate_mapping,
	_validate_output_path,
	_validate_path,
	_validate_required_keys,
)
from seis_ssl_cluster.config.schema import (
	F3_FACIES_DATASET_NAME,
	F3_FACIES_DATASET_VERSION,
	F3_FACIES_INSPECTION_STAGES,
)

Config: TypeAlias = dict[str, object]
_T = TypeVar('_T', bound=Mapping[str, object])

_F3_FACIES_INSPECTION_TOP_LEVEL = frozenset(
	{'paths', 'outputs', 'dataset', 'inspection'},
)
_F3_FACIES_INSPECTION_REQUIRED_TOP_LEVEL = frozenset(
	{'paths', 'outputs', 'dataset', 'inspection'},
)
_F3_FACIES_INSPECTION_PATH_KEYS = frozenset({'f3_root', 'artifact_root'})
_F3_FACIES_INSPECTION_OUTPUT_KEYS = frozenset({'inspection_dir'})
_F3_FACIES_INSPECTION_DATASET_KEYS = frozenset({'name', 'version'})
_F3_FACIES_INSPECTION_PATH_KEY_SUFFIXES = (
	'_dir',
	'_json',
	'_csv',
	'_markdown',
	'_png',
	'_path',
)


def resolve_f3_facies_inspection_config(config: _T, *, stage: str) -> Config:
	"""Validate and resolve a raw config for an F3 facies inspection entrypoint."""
	if stage not in F3_FACIES_INSPECTION_STAGES:
		msg = (
			f'stage must be one of {sorted(F3_FACIES_INSPECTION_STAGES)!r}; '
			f'got {stage!r}'
		)
		raise ValueError(msg)
	_validate_mapping(config)
	_reject_legacy_attribute_config(config)
	_reject_stage_key(config)
	_validate_f3_facies_inspection_top_level_sections(config, stage)

	resolved = deepcopy(dict(config))
	resolved['stage'] = stage
	paths = _validate_f3_facies_inspection_paths(
		_required_mapping(resolved, 'paths'),
	)
	_validate_f3_facies_inspection_outputs(
		_required_mapping(resolved, 'outputs'),
		paths=paths,
	)
	_validate_f3_facies_dataset(_required_mapping(resolved, 'dataset'))
	inspection = _required_mapping(resolved, 'inspection')
	if not inspection:
		msg = 'inspection must contain stage-specific settings'
		raise ValueError(msg)
	_validate_f3_facies_inspection_artifact_paths(
		inspection,
		input_root=paths.f3_root,
	)
	return resolved


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_f3_facies_inspection_top_level_sections(
	config: Mapping[str, object],
	stage: str,
) -> None:
	keys = set(config)
	unexpected = sorted(keys - _F3_FACIES_INSPECTION_TOP_LEVEL)
	if unexpected:
		msg = (
			f'top-level section(s) not allowed for {stage}: {unexpected!r}; '
			'allowed sections are '
			f'{sorted(_F3_FACIES_INSPECTION_TOP_LEVEL)!r}'
		)
		raise ValueError(msg)
	missing = sorted(_F3_FACIES_INSPECTION_REQUIRED_TOP_LEVEL - keys)
	if missing:
		msg = f'missing required top-level section(s) for {stage}: {missing!r}'
		raise ValueError(msg)


def _validate_f3_facies_inspection_paths(
	paths: Mapping[str, object],
) -> _ResolvedPaths:
	_validate_allowed_keys(
		paths,
		_F3_FACIES_INSPECTION_PATH_KEYS,
		prefix='paths',
	)
	return _ResolvedPaths(
		f3_root=_validate_absolute_path(paths, 'f3_root', prefix='paths'),
		artifact_root=_validate_absolute_path(paths, 'artifact_root', prefix='paths'),
	)


def _validate_f3_facies_inspection_outputs(
	outputs: Mapping[str, object],
	*,
	paths: _ResolvedPaths,
) -> None:
	_validate_allowed_keys(
		outputs,
		_F3_FACIES_INSPECTION_OUTPUT_KEYS,
		prefix='outputs',
	)
	_validate_required_keys(
		outputs,
		_F3_FACIES_INSPECTION_OUTPUT_KEYS,
		prefix='outputs',
	)
	inspection_dir = _validate_path(outputs, 'inspection_dir', prefix='outputs')
	_validate_output_path(
		inspection_dir,
		'outputs.inspection_dir',
		input_root=paths.f3_root,
		input_root_label='paths.f3_root',
	)


def _validate_f3_facies_inspection_artifact_paths(
	inspection: Mapping[str, object],
	*,
	input_root: Path,
	prefix: str = 'inspection',
) -> None:
	for key, value in inspection.items():
		label = f'{prefix}.{key}'
		if isinstance(value, Mapping):
			_validate_f3_facies_inspection_artifact_paths(
				value,
				input_root=input_root,
				prefix=label,
			)
			continue
		if not _is_f3_facies_inspection_path_key(key):
			continue
		if not isinstance(value, str) or not value:
			msg = f'{label} must be a non-empty string; got {value!r}'
			raise TypeError(msg)
		_validate_output_path(
			Path(value),
			label,
			input_root=input_root,
			input_root_label='paths.f3_root',
		)


def _is_f3_facies_inspection_path_key(key: str) -> bool:
	return key.endswith(_F3_FACIES_INSPECTION_PATH_KEY_SUFFIXES)


def _validate_f3_facies_dataset(dataset: Mapping[str, object]) -> None:
	_validate_allowed_keys(
		dataset,
		_F3_FACIES_INSPECTION_DATASET_KEYS,
		prefix='dataset',
	)
	_validate_required_keys(
		dataset,
		_F3_FACIES_INSPECTION_DATASET_KEYS,
		prefix='dataset',
	)
	if dataset.get('name') != F3_FACIES_DATASET_NAME:
		msg = (
			f'dataset.name must be {F3_FACIES_DATASET_NAME!r}; '
			f'got {dataset.get("name")!r}'
		)
		raise ValueError(msg)
	if dataset.get('version') != F3_FACIES_DATASET_VERSION:
		msg = (
			f'dataset.version must be {F3_FACIES_DATASET_VERSION!r}; '
			f'got {dataset.get("version")!r}'
		)
		raise ValueError(msg)


__all__ = ['resolve_f3_facies_inspection_config']
