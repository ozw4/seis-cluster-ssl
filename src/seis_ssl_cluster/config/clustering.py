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
	_validate_nonnegative_finite_number,
	_validate_optional_positive_int,
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
		'stratigraphic_hmm',
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
_CLUSTERING_RESIDUALIZATION_ENABLED_REQUIRED_KEYS = _CLUSTERING_RESIDUALIZATION_KEYS
_CLUSTERING_PCA_KEYS = frozenset({'enabled', 'n_components', 'whiten'})
_CLUSTERING_METHODS = frozenset(
	{
		'minibatch_kmeans',
		'stratigraphic_hmm_kmeans',
	},
)
_STRATIGRAPHIC_HMM_KEYS = frozenset(
	{
		'emission_source',
		'iterations',
		'z_axis',
		'z_direction',
		'transition',
		'init',
		'update',
	},
)
_STRATIGRAPHIC_HMM_REQUIRED_KEYS = frozenset(
	{
		'iterations',
		'z_axis',
		'z_direction',
		'transition',
		'init',
		'update',
	},
)
_STRATIGRAPHIC_HMM_EMISSION_SOURCES = frozenset({'embedding', 'z_coordinate'})
_STRATIGRAPHIC_HMM_TRANSITION_KEYS = frozenset(
	{
		'same_cost',
		'advance_cost',
		'jump_cost',
		'reverse_cost',
		'forbid_reverse',
		'max_jump',
	},
)
_STRATIGRAPHIC_HMM_INIT_KEYS = frozenset({'order_by'})
_STRATIGRAPHIC_HMM_UPDATE_KEYS = frozenset({'empty_cluster_policy'})


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
	_validate_stratigraphic_hmm(clustering)
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
			f'clustering.embedding_normalization must be "l2" or "none"; got {value!r}'
		)
		raise ValueError(msg)


def _validate_clustering_method(clustering: Mapping[str, object]) -> None:
	value = clustering.get('method')
	if value not in _CLUSTERING_METHODS:
		msg = (
			'clustering.method must be "minibatch_kmeans" or '
			f'"stratigraphic_hmm_kmeans"; got {value!r}'
		)
		raise ValueError(msg)


def _validate_stratigraphic_hmm(clustering: Mapping[str, object]) -> None:
	if clustering.get('method') != 'stratigraphic_hmm_kmeans':
		return
	if 'stratigraphic_hmm' not in clustering:
		msg = (
			'clustering.stratigraphic_hmm is required when clustering.method is '
			"'stratigraphic_hmm_kmeans'"
		)
		raise ValueError(msg)
	hmm = _required_child_mapping(
		clustering,
		'stratigraphic_hmm',
		prefix='clustering',
	)
	_validate_allowed_keys(
		hmm,
		_STRATIGRAPHIC_HMM_KEYS,
		prefix='clustering.stratigraphic_hmm',
	)
	_validate_required_keys(
		hmm,
		_STRATIGRAPHIC_HMM_REQUIRED_KEYS,
		prefix='clustering.stratigraphic_hmm',
	)
	_validate_stratigraphic_hmm_emission_source(hmm)
	_validate_positive_int(
		hmm,
		'iterations',
		prefix='clustering.stratigraphic_hmm',
	)
	_validate_stratigraphic_hmm_z_axis(hmm)
	_validate_stratigraphic_hmm_z_direction(hmm)
	_validate_stratigraphic_hmm_transition(hmm)
	_validate_stratigraphic_hmm_init(hmm)
	_validate_stratigraphic_hmm_update(hmm)


def _validate_stratigraphic_hmm_emission_source(
	hmm: Mapping[str, object],
) -> None:
	value = hmm.get('emission_source', 'embedding')
	if value not in _STRATIGRAPHIC_HMM_EMISSION_SOURCES:
		msg = (
			'clustering.stratigraphic_hmm.emission_source must be '
			"'embedding' or 'z_coordinate'; "
			f'got {value!r}'
		)
		raise ValueError(msg)


def _validate_stratigraphic_hmm_z_axis(hmm: Mapping[str, object]) -> None:
	value = hmm.get('z_axis')
	if not _is_int(value) or int(value) != 2:
		msg = (
			'clustering.stratigraphic_hmm.z_axis must currently be integer 2; '
			f'got {value!r}'
		)
		raise ValueError(msg)


def _validate_stratigraphic_hmm_z_direction(
	hmm: Mapping[str, object],
) -> None:
	value = hmm.get('z_direction')
	if value != 'increasing_downward':
		msg = (
			'clustering.stratigraphic_hmm.z_direction must currently be '
			f"'increasing_downward'; got {value!r}"
		)
		raise ValueError(msg)


def _validate_stratigraphic_hmm_transition(
	hmm: Mapping[str, object],
) -> None:
	transition = _required_child_mapping(
		hmm,
		'transition',
		prefix='clustering.stratigraphic_hmm',
	)
	_validate_allowed_keys(
		transition,
		_STRATIGRAPHIC_HMM_TRANSITION_KEYS,
		prefix='clustering.stratigraphic_hmm.transition',
	)
	_validate_required_keys(
		transition,
		_STRATIGRAPHIC_HMM_TRANSITION_KEYS,
		prefix='clustering.stratigraphic_hmm.transition',
	)
	for key in ('same_cost', 'advance_cost', 'jump_cost', 'reverse_cost'):
		_validate_nonnegative_finite_number(
			transition,
			key,
			prefix='clustering.stratigraphic_hmm.transition',
		)
	_validate_bool(
		transition,
		'forbid_reverse',
		prefix='clustering.stratigraphic_hmm.transition',
	)
	_validate_optional_positive_int(
		transition,
		'max_jump',
		prefix='clustering.stratigraphic_hmm.transition',
	)


def _validate_stratigraphic_hmm_init(hmm: Mapping[str, object]) -> None:
	init = _required_child_mapping(
		hmm,
		'init',
		prefix='clustering.stratigraphic_hmm',
	)
	_validate_allowed_keys(
		init,
		_STRATIGRAPHIC_HMM_INIT_KEYS,
		prefix='clustering.stratigraphic_hmm.init',
	)
	_validate_required_keys(
		init,
		_STRATIGRAPHIC_HMM_INIT_KEYS,
		prefix='clustering.stratigraphic_hmm.init',
	)
	value = init.get('order_by')
	if value != 'mean_z':
		msg = (
			"clustering.stratigraphic_hmm.init.order_by must be 'mean_z'; "
			f'got {value!r}'
		)
		raise ValueError(msg)


def _validate_stratigraphic_hmm_update(hmm: Mapping[str, object]) -> None:
	update = _required_child_mapping(
		hmm,
		'update',
		prefix='clustering.stratigraphic_hmm',
	)
	_validate_allowed_keys(
		update,
		_STRATIGRAPHIC_HMM_UPDATE_KEYS,
		prefix='clustering.stratigraphic_hmm.update',
	)
	_validate_required_keys(
		update,
		_STRATIGRAPHIC_HMM_UPDATE_KEYS,
		prefix='clustering.stratigraphic_hmm.update',
	)
	value = update.get('empty_cluster_policy')
	if value != 'keep_previous':
		msg = (
			'clustering.stratigraphic_hmm.update.empty_cluster_policy must be '
			f"'keep_previous'; got {value!r}"
		)
		raise ValueError(msg)


__all__ = ['resolve_clustering_config']
