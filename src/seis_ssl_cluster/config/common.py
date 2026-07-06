"""Common primitive validation helpers for config resolvers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Integral, Real
from pathlib import Path
from typing import TypeAlias

from seis_ssl_cluster.config.schema import (
	LEGACY_ATTRIBUTE_KEY_NAMES,
	LEGACY_ATTRIBUTE_KEY_PATHS,
	STAGE_BUILD_MANIFESTS,
	STAGE_CLUSTER_VISUALIZATION,
	STAGE_CLUSTERING,
	STAGE_EMBEDDING_EXTRACTION,
	STAGE_MAE_TRAINING,
	STAGE_NORMALIZATION_QC,
	STAGE_NORMALIZATION_STATS,
	STAGE_PATH_KEYS,
)
from seis_ssl_cluster.paths import ArtifactPaths, ExperimentKey, reject_runs_path

Config: TypeAlias = dict[str, object]

_ALLOWED_TOP_LEVEL: dict[str, frozenset[str]] = {
	STAGE_BUILD_MANIFESTS: frozenset({'paths', 'manifest'}),
	STAGE_NORMALIZATION_STATS: frozenset(
		{'paths', 'manifests', 'normalization'},
	),
	STAGE_NORMALIZATION_QC: frozenset({'paths', 'manifests', 'splits', 'qc'}),
	STAGE_MAE_TRAINING: frozenset(
		{
			'paths',
			'manifests',
			'data',
			'zero_mask',
			'model',
			'masking',
			'loss',
			'train',
			'visualization',
		},
	),
	STAGE_EMBEDDING_EXTRACTION: frozenset(
		{'paths', 'manifests', 'embeddings', 'embedding'},
	),
	STAGE_CLUSTERING: frozenset({'paths', 'embeddings', 'clustering'}),
	STAGE_CLUSTER_VISUALIZATION: frozenset(
		{'paths', 'clustering', 'visualization'},
	),
}

_REQUIRED_TOP_LEVEL: dict[str, frozenset[str]] = {
	STAGE_BUILD_MANIFESTS: frozenset({'paths', 'manifest'}),
	STAGE_NORMALIZATION_STATS: frozenset(
		{'paths', 'manifests', 'normalization'},
	),
	STAGE_NORMALIZATION_QC: frozenset({'paths', 'manifests', 'splits', 'qc'}),
	STAGE_MAE_TRAINING: frozenset(
		{'paths', 'manifests', 'data', 'model', 'masking', 'loss', 'train'},
	),
	STAGE_EMBEDDING_EXTRACTION: frozenset(
		{'paths', 'manifests', 'embeddings', 'embedding'},
	),
	STAGE_CLUSTERING: frozenset({'paths', 'embeddings', 'clustering'}),
	STAGE_CLUSTER_VISUALIZATION: frozenset(
		{'paths', 'clustering', 'visualization'},
	),
}

_NOPIMS_DATASET = 'nopims'
_NOPIMS_PRETRAIN_VERSION = 'pretrain_v1'


class _ResolvedPaths:
	def __init__(
		self,
		*,
		artifact_root: Path,
		nopims_root: Path | None = None,
		f3_root: Path | None = None,
	) -> None:
		self.nopims_root = nopims_root
		self.f3_root = f3_root
		self.artifact_root = artifact_root


def _resolve_base(
	config: Mapping[str, object],
	stage: str,
	*,
	require_nopims_root: bool = True,
) -> tuple[Config, _ResolvedPaths]:
	_validate_mapping(config)
	_reject_legacy_attribute_config(config)
	_reject_stage_key(config)
	_validate_top_level_sections(config, stage)
	resolved = deepcopy(dict(config))
	resolved['stage'] = stage
	paths = _validate_paths(
		_required_mapping(resolved, 'paths'),
		require_nopims_root=require_nopims_root,
		allowed_keys=STAGE_PATH_KEYS[stage],
	)
	return resolved, paths


def _reject_stage_key(config: Mapping[str, object]) -> None:
	if 'stage' in config:
		msg = (
			'stage is selected by the entrypoint; remove the top-level '
			'stage key from this YAML and choose the proc script instead.'
		)
		raise ValueError(msg)


def _validate_top_level_sections(config: Mapping[str, object], stage: str) -> None:
	allowed = _ALLOWED_TOP_LEVEL[stage]
	required = _REQUIRED_TOP_LEVEL[stage]
	keys = set(config)
	unexpected = sorted(keys - allowed)
	if unexpected:
		msg = (
			f'top-level section(s) not allowed for {stage}: {unexpected!r}; '
			f'allowed sections are {sorted(allowed)!r}'
		)
		raise ValueError(msg)
	missing = sorted(required - keys)
	if missing:
		msg = f'missing required top-level section(s) for {stage}: {missing!r}'
		raise ValueError(msg)


def _reject_legacy_attribute_config(config: Mapping[str, object]) -> None:
	for path, key in _iter_mapping_keys(config):
		if path in LEGACY_ATTRIBUTE_KEY_PATHS or key in LEGACY_ATTRIBUTE_KEY_NAMES:
			msg = (
				f'{path} is a legacy multi-attribute config key and is not '
				'valid for the amplitude-only MVP; remove fixed-attribute '
				'configuration from this config.'
			)
			raise ValueError(msg)


def _validate_paths(
	paths: Mapping[str, object],
	*,
	require_nopims_root: bool,
	allowed_keys: frozenset[str] | None,
) -> _ResolvedPaths:
	if allowed_keys is not None:
		_validate_allowed_keys(paths, allowed_keys, prefix='paths')
	nopims_root: Path | None = None
	if require_nopims_root or 'nopims_root' in paths:
		nopims_root = _validate_absolute_path(paths, 'nopims_root', prefix='paths')
	if 'output_root' in paths:
		_validate_non_empty_path(paths, 'output_root', prefix='paths')
	return _ResolvedPaths(
		artifact_root=_validate_absolute_path(paths, 'artifact_root', prefix='paths'),
		nopims_root=nopims_root,
	)


def _validate_artifact_output_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
	nopims_root: Path | None,
	raw_root_label: str = 'paths.nopims_root',
) -> None:
	if not path.is_absolute():
		msg = f'{label} must be an absolute artifact-registry path; got {path}'
		raise ValueError(msg)
	reject_runs_path(path, label=label)
	if nopims_root is not None and _is_relative_to(path, nopims_root):
		msg = f'{label} must not be under {raw_root_label}; got {path}'
		raise ValueError(msg)
	if not _is_relative_to(path, artifact_root):
		msg = f'{label} must be under paths.artifact_root ({artifact_root}); got {path}'
		raise ValueError(msg)


def _validate_nopims_checkpoint_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
) -> None:
	reject_runs_path(path, label=label)
	_validate_nopims_pretraining_path(
		path.parent,
		f'{label} parent',
		artifact_root=artifact_root,
	)


def _validate_nopims_pretraining_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
) -> None:
	relative = _artifact_relative_path(path, artifact_root)
	if relative is None:
		return
	parts = relative.parts
	expected = (
		'pretraining/nopims/pretrain_v1/<MODEL_TAG>/<RUN_SPEC>'
	)
	if not _is_nopims_artifact_path(parts, ('pretraining',)):
		return
	if len(parts) != 5 or parts[2] != _NOPIMS_PRETRAIN_VERSION:
		_raise_nopims_artifact_path_error(label, path, expected)
	key = ExperimentKey(
		dataset=parts[1],
		version=parts[2],
		model_tag=parts[3],
		run_spec=parts[4],
	)
	_validate_artifact_path_matches(
		path,
		ArtifactPaths(artifact_root).pretraining(key),
		label=label,
		expected=expected,
	)


def _validate_nopims_embedding_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
) -> None:
	relative = _artifact_relative_path(path, artifact_root)
	if relative is None:
		return
	parts = relative.parts
	expected = (
		'embeddings/nopims/pretrain_v1/'
		'<MODEL_TAG>/<SUBSET>/<EMBED_SPEC>'
	)
	if not _is_nopims_artifact_path(parts, ('embeddings',)):
		return
	if len(parts) != 6 or parts[2] != _NOPIMS_PRETRAIN_VERSION:
		_raise_nopims_artifact_path_error(label, path, expected)
	key = ExperimentKey(
		dataset=parts[1],
		version=parts[2],
		model_tag=parts[3],
		subset=parts[4],
		embed_spec=parts[5],
	)
	_validate_artifact_path_matches(
		path,
		ArtifactPaths(artifact_root).embeddings(key),
		label=label,
		expected=expected,
	)


def _validate_nopims_clustering_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
) -> None:
	relative = _artifact_relative_path(path, artifact_root)
	if relative is None:
		return
	parts = relative.parts
	expected = (
		'clustering/nopims/pretrain_v1/'
		'<MODEL_TAG>/<SUBSET>/<EMBED_SPEC>/<CLUSTER_SPEC>'
	)
	if not _is_nopims_artifact_path(parts, ('clustering',)):
		return
	if len(parts) != 7 or parts[2] != _NOPIMS_PRETRAIN_VERSION:
		_raise_nopims_artifact_path_error(label, path, expected)
	key = ExperimentKey(
		dataset=parts[1],
		version=parts[2],
		model_tag=parts[3],
		subset=parts[4],
		embed_spec=parts[5],
		cluster_spec=parts[6],
	)
	_validate_artifact_path_matches(
		path,
		ArtifactPaths(artifact_root).clustering(key),
		label=label,
		expected=expected,
	)


def _validate_nopims_cluster_visualization_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
) -> None:
	relative = _artifact_relative_path(path, artifact_root)
	if relative is None:
		return
	parts = relative.parts
	expected = (
		'visualizations/clusters/nopims/pretrain_v1/'
		'<MODEL_TAG>/<SUBSET>/<EMBED_SPEC>/<CLUSTER_SPEC>/<VIZ_SPEC>'
	)
	if not _is_nopims_artifact_path(parts, ('visualizations', 'clusters')):
		return
	if len(parts) != 9 or parts[3] != _NOPIMS_PRETRAIN_VERSION:
		_raise_nopims_artifact_path_error(label, path, expected)
	key = ExperimentKey(
		dataset=parts[2],
		version=parts[3],
		model_tag=parts[4],
		subset=parts[5],
		embed_spec=parts[6],
		cluster_spec=parts[7],
		viz_spec=parts[8],
	)
	_validate_artifact_path_matches(
		path,
		ArtifactPaths(artifact_root).cluster_visualization(key),
		label=label,
		expected=expected,
	)


def _artifact_relative_path(path: Path, artifact_root: Path) -> Path | None:
	try:
		return path.resolve(strict=False).relative_to(
			artifact_root.resolve(strict=False),
		)
	except ValueError:
		return None


def _is_nopims_artifact_path(
	parts: tuple[str, ...],
	stage_prefix: tuple[str, ...],
) -> bool:
	prefix_len = len(stage_prefix)
	if len(parts) <= prefix_len:
		return False
	return (
		parts[:prefix_len] == stage_prefix
		and parts[prefix_len] == _NOPIMS_DATASET
	)


def _validate_artifact_path_matches(
	path: Path,
	expected_path: Path,
	*,
	label: str,
	expected: str,
) -> None:
	if path.resolve(strict=False) != expected_path.resolve(strict=False):
		_raise_nopims_artifact_path_error(label, path, expected)


def _raise_nopims_artifact_path_error(
	label: str,
	path: Path,
	expected: str,
) -> None:
	msg = f'{label} must follow ArtifactPaths {expected}; got {path}'
	raise ValueError(msg)


def _validate_mapping(config: Mapping[str, object]) -> None:
	if not isinstance(config, Mapping):
		msg = 'config must be a mapping'
		raise TypeError(msg)


def _validate_allowed_keys(
	parent: Mapping[str, object],
	allowed: frozenset[str],
	*,
	prefix: str,
) -> None:
	unexpected = sorted(set(parent) - allowed)
	if unexpected:
		labels = [f'{prefix}.{key}' for key in unexpected]
		msg = (
			f'{prefix} key(s) not allowed: {labels!r}; '
			f'allowed keys are {sorted(allowed)!r}'
		)
		raise ValueError(msg)


def _validate_required_keys(
	parent: Mapping[str, object],
	keys: frozenset[str],
	*,
	prefix: str,
) -> None:
	for key in sorted(keys):
		_validate_required_key(parent, key, prefix=prefix)


def _validate_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	path = _validate_path(parent, key, prefix=prefix)
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _validate_non_empty_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	return _validate_path(parent, key, prefix=prefix)


def _validate_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return Path(value)


def _validate_optional_output_path_under_root(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
	root: Path,
	root_label: str,
) -> None:
	value = parent.get(key)
	if value is None:
		return
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string or null; got {value!r}'
		raise TypeError(msg)
	_validate_path_under_root(
		Path(value),
		f'{prefix}.{key}',
		root=root,
		root_label=root_label,
	)


def _validate_path_under_root(
	path: Path,
	label: str,
	*,
	root: Path,
	root_label: str,
) -> None:
	if not path.is_absolute():
		msg = f'{label} must be an absolute path; got {path}'
		raise ValueError(msg)
	if not _is_relative_to(path, root):
		msg = f'{label} must be under {root_label} ({root}); got {path}'
		raise ValueError(msg)


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


def _iter_mapping_keys(
	value: object,
	prefix: str = '',
) -> Sequence[tuple[str, str]]:
	if isinstance(value, Sequence) and not isinstance(value, str | bytes):
		paths: list[tuple[str, str]] = []
		for index, child in enumerate(value):
			path = f'{prefix}[{index}]' if prefix else f'[{index}]'
			paths.extend(_iter_mapping_keys(child, path))
		return paths

	if not isinstance(value, Mapping):
		return ()

	paths: list[tuple[str, str]] = []
	for key, child in value.items():
		if not isinstance(key, str):
			continue
		path = f'{prefix}.{key}' if prefix else key
		paths.append((path, key))
		paths.extend(_iter_mapping_keys(child, path))
	return paths


def _merge_section_defaults(
	config: dict[str, object],
	section: str,
	defaults: Mapping[str, object],
) -> None:
	current = config.get(section)
	if current is None:
		config[section] = deepcopy(dict(defaults))
		return
	if not isinstance(current, dict):
		msg = f'{section} must be a mapping'
		raise TypeError(msg)
	config[section] = {**deepcopy(dict(defaults)), **current}


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _required_child_mapping(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{prefix}.{key} must be a mapping'
		raise TypeError(msg)
	return value


def _validate_non_empty_str(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)


def _validate_positive_int_triplet(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> tuple[int, int, int]:
	value = parent.get(key)
	if (
		not isinstance(value, list)
		or len(value) != 3
		or not all(_is_int(item) and int(item) > 0 for item in value)
	):
		msg = f'{prefix}.{key} must be a list of three positive integers'
		raise ValueError(msg)
	return (int(value[0]), int(value[1]), int(value[2]))


def _validate_nonnegative_int_triplet(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> tuple[int, int, int]:
	value = parent.get(key)
	if (
		not isinstance(value, list)
		or len(value) != 3
		or not all(_is_int(item) and int(item) >= 0 for item in value)
	):
		msg = f'{prefix}.{key} must be a list of three nonnegative integers'
		raise ValueError(msg)
	return (int(value[0]), int(value[1]), int(value[2]))


def _validate_positive_int_list(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if (
		not isinstance(value, list)
		or not value
		or not all(_is_int(item) and int(item) > 0 for item in value)
	):
		msg = f'{prefix}.{key} must be a non-empty list of positive integers'
		raise ValueError(msg)


def _validate_unique_positive_int_list(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	_validate_positive_int_list(parent, key, prefix=prefix)
	value = parent.get(key)
	if not isinstance(value, list):
		msg = f'{prefix}.{key} must be a non-empty list of positive integers'
		raise TypeError(msg)
	values = [int(item) for item in value]
	if len(set(values)) != len(values):
		msg = f'{prefix}.{key} must not contain duplicates; got {values!r}'
		raise ValueError(msg)


def _validate_nonnegative_int_list(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if (
		not isinstance(value, list)
		or any(not _is_int(item) or int(item) < 0 for item in value)
	):
		msg = f'{prefix}.{key} must be a list of nonnegative integers'
		raise ValueError(msg)


def _validate_positive_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not _is_int(value) or int(value) <= 0:
		msg = f'{prefix}.{key} must be a positive integer; got {value!r}'
		raise ValueError(msg)


def _validate_required_key(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	if key not in parent:
		msg = f'{prefix}.{key} is required'
		raise ValueError(msg)


def _validate_nonnegative_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not _is_int(value) or int(value) < 0:
		msg = f'{prefix}.{key} must be a nonnegative integer; got {value!r}'
		raise ValueError(msg)


def _validate_optional_positive_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	if key not in parent or parent.get(key) is None:
		return
	_validate_positive_int(parent, key, prefix=prefix)


def _validate_optional_nonnegative_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	if key not in parent or parent.get(key) is None:
		return
	_validate_nonnegative_int(parent, key, prefix=prefix)


def _validate_positive_number(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not _is_number(value) or float(value) <= 0.0:
		msg = f'{prefix}.{key} must be positive; got {value!r}'
		raise ValueError(msg)


def _validate_nonnegative_number(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not _is_number(value) or float(value) < 0.0:
		msg = f'{prefix}.{key} must be nonnegative; got {value!r}'
		raise ValueError(msg)


def _validate_positive_finite_number(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if (
		not _is_number(value)
		or float(value) <= 0.0
		or not math.isfinite(float(value))
	):
		msg = f'{prefix}.{key} must be a finite positive number; got {value!r}'
		raise ValueError(msg)


def _validate_nonnegative_finite_number(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if (
		not _is_number(value)
		or float(value) < 0.0
		or not math.isfinite(float(value))
	):
		msg = f'{prefix}.{key} must be a nonnegative finite number; got {value!r}'
		raise ValueError(msg)


def _validate_optional_fraction(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	if key in parent:
		_validate_fraction(parent, key, prefix=prefix)


def _validate_fraction(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if (
		not _is_number(value)
		or float(value) < 0.0
		or float(value) > 1.0
		or not math.isfinite(float(value))
	):
		msg = f'{prefix}.{key} must be between 0 and 1; got {value!r}'
		raise ValueError(msg)


def _validate_bool(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not isinstance(value, bool):
		msg = f'{prefix}.{key} must be a boolean; got {value!r}'
		raise TypeError(msg)


def _is_int(value: object) -> bool:
	return isinstance(value, Integral) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
	return isinstance(value, Real) and not isinstance(value, bool)
