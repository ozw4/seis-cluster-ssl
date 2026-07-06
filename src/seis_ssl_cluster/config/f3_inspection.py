"""F3 facies benchmark inspection config validation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TypeAlias, TypeVar

from seis_ssl_cluster.config.artifact_path_validation import (
	_validate_artifact_output_path,
)
from seis_ssl_cluster.config.base import (
	_reject_legacy_attribute_config,
	_reject_stage_key,
	_ResolvedPaths,
)
from seis_ssl_cluster.config.common import (
	_validate_absolute_path,
	_validate_allowed_keys,
	_validate_bool,
	_validate_mapping,
	_validate_path,
	_validate_path_under_root,
	_validate_positive_finite_number,
	_validate_required_keys,
)
from seis_ssl_cluster.config.schema import (
	F3_FACIES_DATASET_NAME,
	F3_FACIES_DATASET_VERSION,
	F3_FACIES_INSPECTION_ARTIFACT_SUBDIR,
	F3_FACIES_INSPECTION_STAGES,
)

Config: TypeAlias = dict[str, object]
_T = TypeVar('_T', bound=Mapping[str, object])

_F3_FACIES_INSPECTION_TOP_LEVEL = frozenset(
	{'paths', 'outputs', 'dataset', 'inspection', 'publish'},
)
_F3_FACIES_INSPECTION_REQUIRED_TOP_LEVEL = frozenset(
	{'paths', 'outputs', 'dataset', 'inspection'},
)
_F3_FACIES_INSPECTION_PATH_KEYS = frozenset({'f3_root', 'artifact_root'})
_F3_FACIES_INSPECTION_OUTPUT_KEYS = frozenset({'inspection_dir'})
_F3_FACIES_INSPECTION_DATASET_KEYS = frozenset({'name', 'version'})
_F3_FACIES_INSPECTION_PUBLISH_KEYS = frozenset(
	{'enabled', 'output_dir', 'include_figures', 'max_file_size_mb'},
)
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
	inspection_dir = _validate_f3_facies_inspection_outputs(
		_required_mapping(resolved, 'outputs'),
		paths=paths,
	)
	_validate_f3_facies_dataset(_required_mapping(resolved, 'dataset'))
	if 'publish' in resolved:
		_validate_f3_facies_inspection_publish(
			_required_mapping(resolved, 'publish'),
		)
	inspection = _required_mapping(resolved, 'inspection')
	if not inspection:
		msg = 'inspection must contain stage-specific settings'
		raise ValueError(msg)
	_validate_f3_facies_inspection_artifact_paths(
		inspection,
		inspection_dir=inspection_dir,
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
) -> Path:
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
	_validate_artifact_output_path(
		inspection_dir,
		'outputs.inspection_dir',
		artifact_root=paths.artifact_root,
		nopims_root=paths.f3_root,
		raw_root_label='paths.f3_root',
	)
	expected_relative = Path(F3_FACIES_INSPECTION_ARTIFACT_SUBDIR)
	actual_relative = inspection_dir.resolve(strict=False).relative_to(
		paths.artifact_root.resolve(strict=False),
	)
	if actual_relative != expected_relative:
		msg = (
			'outputs.inspection_dir must be paths.artifact_root / '
			f'{F3_FACIES_INSPECTION_ARTIFACT_SUBDIR!r}; got {inspection_dir}'
		)
		raise ValueError(msg)
	return inspection_dir


def _validate_f3_facies_inspection_artifact_paths(
	inspection: Mapping[str, object],
	*,
	inspection_dir: Path,
	prefix: str = 'inspection',
) -> None:
	for key, value in inspection.items():
		label = f'{prefix}.{key}'
		if isinstance(value, Mapping):
			_validate_f3_facies_inspection_artifact_paths(
				value,
				inspection_dir=inspection_dir,
				prefix=label,
			)
			continue
		if not _is_f3_facies_inspection_path_key(key):
			continue
		if not isinstance(value, str) or not value:
			msg = f'{label} must be a non-empty string; got {value!r}'
			raise TypeError(msg)
		_validate_path_under_root(
			Path(value),
			label,
			root=inspection_dir,
			root_label='outputs.inspection_dir',
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


def _validate_f3_facies_inspection_publish(
	publish: Mapping[str, object],
) -> None:
	_validate_allowed_keys(
		publish,
		_F3_FACIES_INSPECTION_PUBLISH_KEYS,
		prefix='publish',
	)
	if 'enabled' in publish:
		_validate_bool(publish, 'enabled', prefix='publish')
	if 'include_figures' in publish:
		_validate_bool(publish, 'include_figures', prefix='publish')
	if 'output_dir' in publish:
		_validate_path(publish, 'output_dir', prefix='publish')
	if publish.get('enabled') is True and 'output_dir' not in publish:
		msg = 'publish.output_dir is required when publish.enabled is true'
		raise ValueError(msg)
	if 'max_file_size_mb' in publish:
		_validate_positive_finite_number(
			publish,
			'max_file_size_mb',
			prefix='publish',
		)


__all__ = ['resolve_f3_facies_inspection_config']
