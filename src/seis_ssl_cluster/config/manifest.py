"""Manifest-build config validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, TypeVar

from seis_ssl_cluster.config.artifact_path_validation import (
	_validate_artifact_output_path,
)
from seis_ssl_cluster.config.base import _resolve_base
from seis_ssl_cluster.config.common import (
	_required_mapping,
	_validate_non_empty_path,
	_validate_non_empty_str,
	_validate_path,
)
from seis_ssl_cluster.config.schema import STAGE_BUILD_MANIFESTS

Config: TypeAlias = dict[str, object]
_T = TypeVar('_T', bound=Mapping[str, object])


def resolve_manifest_build_config(config: _T) -> Config:
	"""Validate and resolve raw config for the manifest-build entrypoint."""
	resolved, paths = _resolve_base(config, STAGE_BUILD_MANIFESTS)
	manifest = _required_mapping(resolved, 'manifest')
	_validate_non_empty_path(manifest, 'input_path_list', prefix='manifest')
	_validate_artifact_output_path(
		_validate_path(manifest, 'output_dir', prefix='manifest'),
		'manifest.output_dir',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
	)
	_validate_artifact_output_path(
		_validate_path(manifest, 'normalization_stats_dir', prefix='manifest'),
		'manifest.normalization_stats_dir',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
	)
	_validate_non_empty_str(manifest, 'output_name', prefix='manifest')
	return resolved


__all__ = ['resolve_manifest_build_config']
