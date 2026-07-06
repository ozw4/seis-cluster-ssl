"""Validation and resolution for embedding clustering configs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, TypeVar

from seis_ssl_cluster.config.artifact_paths import (
	_validate_artifact_output_path,
	_validate_nopims_clustering_path,
	_validate_nopims_embedding_path,
)
from seis_ssl_cluster.config.base import _resolve_base
from seis_ssl_cluster.config.common import (
	_is_int,
	_required_child_mapping,
	_required_mapping,
	_validate_allowed_keys,
	_validate_bool,
	_validate_path,
	_validate_positive_int,
	_validate_required_key,
	_validate_required_keys,
	_validate_unique_positive_int_list,
)
from seis_ssl_cluster.config.schema import STAGE_CLUSTERING

Config: TypeAlias = dict[str, object]
_T = TypeVar('_T', bound=Mapping[str, object])

_CLUSTERING_EMBEDDINGS_KEYS = frozenset({'input_dir'})
_CLUSTERING_KEYS = frozenset(
	{
		'output_dir',
		'embedding_normalization',
		'residualization',
		'pca',
		'sample_tokens',
		'method',
		'k_values',
		'minibatch_size',
		'prediction_batch_size',
		'seed',
	},
)
_CLUSTERING_REQUIRED_KEYS = frozenset(
	{
		'output_dir',
		'embedding_normalization',
		'residualization',
		'pca',
		'sample_tokens',
		'method',
		'k_values',
		'minibatch_size',
		'seed',
	},
)
_CLUSTERING_RESIDUALIZATION_KEYS = frozenset(
	{
		'enabled',
		'mode',
		'group_by',
		'add_global_mean_back',
		'min_group_count',
	},
)
_CLUSTERING_RESIDUALIZATION_ENABLED_REQUIRED_KEYS = (
	_CLUSTERING_RESIDUALIZATION_KEYS
)
_CLUSTERING_PCA_KEYS = frozenset({'enabled', 'n_components', 'whiten'})


def resolve_clustering_config(config: _T) -> Config:
	"""Validate and resolve raw config for embedding clustering."""
	resolved, paths = _resolve_base(
		config,
		STAGE_CLUSTERING,
		require_nopims_root=False,
	)
	embeddings = _required_mapping(resolved, 'embeddings')
	clustering = _required_mapping(resolved, 'clustering')
	if 'residualization' not in clustering:
		resolved['clustering']['residualization'] = {'enabled': False}
		clustering = _required_mapping(resolved, 'clustering')
	_validate_allowed_keys(
		embeddings,
		_CLUSTERING_EMBEDDINGS_KEYS,
		prefix='embeddings',
	)
	_validate_allowed_keys(clustering, _CLUSTERING_KEYS, prefix='clustering')
	_validate_required_keys(
		clustering,
		_CLUSTERING_REQUIRED_KEYS,
		prefix='clustering',
	)
	input_dir = _validate_path(embeddings, 'input_dir', prefix='embeddings')
	_validate_artifact_output_path(
		input_dir,
		'embeddings.input_dir',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
	)
	_validate_nopims_embedding_path(
		input_dir,
		'embeddings.input_dir',
		artifact_root=paths.artifact_root,
	)
	output_dir = _validate_path(clustering, 'output_dir', prefix='clustering')
	_validate_artifact_output_path(
		output_dir,
		'clustering.output_dir',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
	)
	_validate_nopims_clustering_path(
		output_dir,
		'clustering.output_dir',
		artifact_root=paths.artifact_root,
	)
	_validate_clustering_normalization(clustering)
	residualization = _required_child_mapping(
		clustering,
		'residualization',
		prefix='clustering',
	)
	_validate_clustering_residualization(residualization)
	pca = _required_child_mapping(clustering, 'pca', prefix='clustering')
	_validate_allowed_keys(pca, _CLUSTERING_PCA_KEYS, prefix='clustering.pca')
	_validate_required_keys(
		pca,
		_CLUSTERING_PCA_KEYS,
		prefix='clustering.pca',
	)
	_validate_bool(pca, 'enabled', prefix='clustering.pca')
	_validate_positive_int(pca, 'n_components', prefix='clustering.pca')
	_validate_bool(pca, 'whiten', prefix='clustering.pca')
	_validate_positive_int(clustering, 'sample_tokens', prefix='clustering')
	_validate_clustering_method(clustering)
	_validate_unique_positive_int_list(
		clustering,
		'k_values',
		prefix='clustering',
	)
	_validate_positive_int(clustering, 'minibatch_size', prefix='clustering')
	if 'prediction_batch_size' in clustering:
		_validate_positive_int(
			clustering,
			'prediction_batch_size',
			prefix='clustering',
		)
	if not _is_int(clustering.get('seed')):
		msg = f'clustering.seed must be an integer; got {clustering.get("seed")!r}'
		raise ValueError(msg)
	return resolved


def _validate_clustering_residualization(
	residualization: Mapping[str, object],
) -> None:
	_validate_allowed_keys(
		residualization,
		_CLUSTERING_RESIDUALIZATION_KEYS,
		prefix='clustering.residualization',
	)
	_validate_required_key(
		residualization,
		'enabled',
		prefix='clustering.residualization',
	)
	_validate_bool(
		residualization,
		'enabled',
		prefix='clustering.residualization',
	)
	if not residualization['enabled']:
		return
	_validate_required_keys(
		residualization,
		_CLUSTERING_RESIDUALIZATION_ENABLED_REQUIRED_KEYS,
		prefix='clustering.residualization',
	)
	if residualization['mode'] != 'local_token_position':
		msg = (
			"clustering.residualization.mode must be 'local_token_position'; "
			f'got {residualization["mode"]!r}'
		)
		raise ValueError(msg)
	if residualization['group_by'] not in {'token_phase', 'local_token_position'}:
		msg = (
			'clustering.residualization.group_by must be '
			"'token_phase' or 'local_token_position'; "
			f'got {residualization["group_by"]!r}'
		)
		raise ValueError(msg)
	_validate_bool(
		residualization,
		'add_global_mean_back',
		prefix='clustering.residualization',
	)
	_validate_positive_int(
		residualization,
		'min_group_count',
		prefix='clustering.residualization',
	)


def _validate_clustering_normalization(clustering: Mapping[str, object]) -> None:
	value = clustering.get('embedding_normalization')
	if value not in {'l2', 'none'}:
		msg = (
			'clustering.embedding_normalization must be "l2" or "none"; '
			f'got {value!r}'
		)
		raise ValueError(msg)


def _validate_clustering_method(clustering: Mapping[str, object]) -> None:
	value = clustering.get('method')
	if value != 'minibatch_kmeans':
		msg = 'clustering.method must be "minibatch_kmeans"'
		raise ValueError(msg)


__all__ = ['resolve_clustering_config']
