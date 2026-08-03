"""Stage-aware base validation helpers for config resolvers."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, TypeAlias

from seis_ssl_cluster.config.common import (
	_iter_mapping_keys,
	_required_mapping,
	_validate_absolute_path,
	_validate_allowed_keys,
	_validate_mapping,
	_validate_non_empty_path,
)
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
	STAGE_STRAT_HMM_PRETEXT_TRAINING,
	STAGE_STRAT_HMM_PSEUDO_TARGETS,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

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
	STAGE_STRAT_HMM_PRETEXT_TRAINING: frozenset(
		{
			'paths',
			'identity',
			'manifests',
			'data',
			'zero_mask',
			'model',
			'pseudo_targets',
			'teacher',
			'student',
			'head',
			'loss',
			'train',
			'spatial_context',
		},
	),
	STAGE_STRAT_HMM_PSEUDO_TARGETS: frozenset(
		{
			'paths',
			'manifests',
			'checkpoint',
			'model',
			'inference',
			'hmm',
			'outputs',
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
	STAGE_STRAT_HMM_PRETEXT_TRAINING: frozenset(
		{
			'paths',
			'manifests',
			'data',
			'model',
			'pseudo_targets',
			'teacher',
			'student',
			'head',
			'loss',
			'train',
		},
	),
	STAGE_STRAT_HMM_PSEUDO_TARGETS: frozenset(
		{
			'paths',
			'manifests',
			'checkpoint',
			'model',
			'inference',
			'hmm',
			'outputs',
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
