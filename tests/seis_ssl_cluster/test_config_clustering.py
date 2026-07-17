from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.clustering import resolve_clustering_config
from seis_ssl_cluster.config.validate import (
	resolve_clustering_config as validate_resolve_clustering_config,
)

CONFIG_DIR = Path('proc/configs/seis_ssl_cluster')


def test_clustering_config_resolves_from_stage_module() -> None:
	resolved = resolve_clustering_config(_minimal_clustering_config())

	assert resolved['stage'] == 'cluster_embeddings'
	assert resolved['clustering']['method'] == 'minibatch_kmeans'
	assert resolved['clustering']['k_values'] == [4, 6, 8]


def test_stratigraphic_hmm_clustering_config_resolves() -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['method'] = 'stratigraphic_hmm_kmeans'
	cfg['clustering']['stratigraphic_hmm'] = _stratigraphic_hmm_config()

	resolved = resolve_clustering_config(cfg)

	assert resolved['clustering']['method'] == 'stratigraphic_hmm_kmeans'
	assert resolved['clustering']['stratigraphic_hmm']['z_axis'] == 2
	assert 'edge_margin_tokens' not in resolved['clustering']['stratigraphic_hmm']
	assert 'path_prior' not in resolved['clustering']['stratigraphic_hmm']


def test_stratigraphic_hmm_clustering_config_accepts_edge_margin_tokens() -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['method'] = 'stratigraphic_hmm_kmeans'
	cfg['clustering']['stratigraphic_hmm'] = _stratigraphic_hmm_config()
	cfg['clustering']['stratigraphic_hmm']['edge_margin_tokens'] = [8, 8, 0]

	resolved = resolve_clustering_config(cfg)

	assert resolved['clustering']['stratigraphic_hmm']['edge_margin_tokens'] == [
		8,
		8,
		0,
	]


def test_stratigraphic_hmm_clustering_config_accepts_prepared_feature_cache() -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['method'] = 'stratigraphic_hmm_kmeans'
	hmm = _stratigraphic_hmm_config()
	hmm['prepared_feature_cache'] = {
		'chunk_size_tokens': 128,
		'reuse': True,
		'force_rebuild': False,
		'cleanup': True,
		'persist': False,
		'directory': 'artifacts/prepared',
	}
	cfg['clustering']['stratigraphic_hmm'] = hmm

	resolved = resolve_clustering_config(cfg)

	assert (
		resolved['clustering']['stratigraphic_hmm']['prepared_feature_cache']
		== hmm['prepared_feature_cache']
	)


@pytest.mark.parametrize(
	'cache',
	[
		{'chunk_size_tokens': 0},
		{'reuse': 'true'},
		{'cleanup': True, 'persist': True},
		{'directory': ''},
		{'unknown': True},
	],
)
def test_stratigraphic_hmm_clustering_config_rejects_invalid_prepared_cache(
	cache: object,
) -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['method'] = 'stratigraphic_hmm_kmeans'
	hmm = _stratigraphic_hmm_config()
	hmm['prepared_feature_cache'] = cache
	cfg['clustering']['stratigraphic_hmm'] = hmm

	with pytest.raises((TypeError, ValueError), match='prepared_feature_cache'):
		resolve_clustering_config(cfg)


@pytest.mark.parametrize(
	'value',
	[
		[8, 8],
		[8, -1, 0],
		[8, True, 0],
		[8, 2.5, 0],
	],
)
def test_stratigraphic_hmm_clustering_config_rejects_invalid_edge_margin_tokens(
	value: object,
) -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['method'] = 'stratigraphic_hmm_kmeans'
	cfg['clustering']['stratigraphic_hmm'] = _stratigraphic_hmm_config()
	cfg['clustering']['stratigraphic_hmm']['edge_margin_tokens'] = value

	with pytest.raises(ValueError, match='edge_margin_tokens'):
		resolve_clustering_config(cfg)


def test_stratigraphic_hmm_clustering_config_accepts_enabled_path_prior() -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['method'] = 'stratigraphic_hmm_kmeans'
	cfg['clustering']['stratigraphic_hmm'] = _stratigraphic_hmm_config()
	cfg['clustering']['stratigraphic_hmm']['path_prior'] = _path_prior_config()

	resolved = resolve_clustering_config(cfg)

	path_prior = resolved['clustering']['stratigraphic_hmm']['path_prior']
	assert path_prior['enabled'] is True
	assert path_prior['initial_state']['mode'] == 'shallow_anchor'


def test_stratigraphic_hmm_clustering_config_accepts_disabled_path_prior() -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['method'] = 'stratigraphic_hmm_kmeans'
	cfg['clustering']['stratigraphic_hmm'] = _stratigraphic_hmm_config()
	cfg['clustering']['stratigraphic_hmm']['path_prior'] = {'enabled': False}

	resolved = resolve_clustering_config(cfg)

	assert resolved['clustering']['stratigraphic_hmm']['path_prior'] == {
		'enabled': False,
	}


def test_hmm_config_accepts_zero_expected_boundary_target() -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['method'] = 'stratigraphic_hmm_kmeans'
	hmm = _stratigraphic_hmm_config()
	path_prior = _path_prior_config()
	path_prior['expected_boundaries'] = {
		'enabled': True,
		'target': 0,
		'weight': 0.1,
	}
	hmm['path_prior'] = path_prior
	cfg['clustering']['stratigraphic_hmm'] = hmm

	resolved = resolve_clustering_config(cfg)

	assert (
		resolved['clustering']['stratigraphic_hmm']['path_prior'][
			'expected_boundaries'
		]['target']
		== 0
	)


@pytest.mark.parametrize(
	('path', 'value', 'message'),
	[
		(('initial_state', 'extra'), 1, 'extra'),
		(('initial_state', 'mode'), 'deep_anchor', 'initial_state.mode'),
		(('terminal_state', 'mode'), 'shallow_anchor', 'terminal_state.mode'),
		(('terminal_state', 'weight'), -0.1, 'terminal_state.weight'),
		(('expected_boundaries', 'target'), -1, 'expected_boundaries.target'),
	],
)
def test_stratigraphic_hmm_clustering_config_rejects_invalid_path_prior(
	path: tuple[str, ...],
	value: object,
	message: str,
) -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['method'] = 'stratigraphic_hmm_kmeans'
	hmm = _stratigraphic_hmm_config()
	path_prior = _path_prior_config()
	_set_nested(path_prior, path, value)
	hmm['path_prior'] = path_prior
	cfg['clustering']['stratigraphic_hmm'] = hmm

	with pytest.raises(ValueError, match=message):
		resolve_clustering_config(cfg)


def test_stratigraphic_hmm_clustering_config_allows_default_emission_source() -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['method'] = 'stratigraphic_hmm_kmeans'
	cfg['clustering']['stratigraphic_hmm'] = _stratigraphic_hmm_config()
	cfg['clustering']['stratigraphic_hmm'].pop('emission_source')

	resolved = resolve_clustering_config(cfg)

	assert 'emission_source' not in resolved['clustering']['stratigraphic_hmm']


def test_stratigraphic_hmm_clustering_config_accepts_z_coordinate_emission() -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['method'] = 'stratigraphic_hmm_kmeans'
	cfg['clustering']['stratigraphic_hmm'] = _stratigraphic_hmm_config()
	cfg['clustering']['stratigraphic_hmm']['emission_source'] = 'z_coordinate'

	resolved = resolve_clustering_config(cfg)

	assert (
		resolved['clustering']['stratigraphic_hmm']['emission_source'] == 'z_coordinate'
	)


def test_stratigraphic_hmm_clustering_config_requires_hmm_mapping() -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['method'] = 'stratigraphic_hmm_kmeans'

	with pytest.raises(ValueError, match=r'clustering\.stratigraphic_hmm'):
		resolve_clustering_config(cfg)


@pytest.mark.parametrize(
	('path', 'value', 'message'),
	[
		(('iterations',), 0, 'iterations'),
		(('z_axis',), 1, 'z_axis'),
		(('transition', 'reverse_cost'), -1.0, 'reverse_cost'),
		(('transition', 'max_jump'), 0, 'max_jump'),
		(('init', 'order_by'), 'depth', 'order_by'),
		(('emission_source',), 'depth', 'emission_source'),
	],
)
def test_stratigraphic_hmm_clustering_config_validates_nested_values(
	path: tuple[str, ...],
	value: object,
	message: str,
) -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['method'] = 'stratigraphic_hmm_kmeans'
	hmm = _stratigraphic_hmm_config()
	_set_nested(hmm, path, value)
	cfg['clustering']['stratigraphic_hmm'] = hmm

	with pytest.raises(ValueError, match=message):
		resolve_clustering_config(cfg)


def test_stratigraphic_hmm_clustering_config_rejects_nested_unknown_key() -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['method'] = 'stratigraphic_hmm_kmeans'
	hmm = _stratigraphic_hmm_config()
	transition = hmm['transition']
	assert isinstance(transition, dict)
	transition['extra'] = 1
	cfg['clustering']['stratigraphic_hmm'] = hmm

	with pytest.raises(
		ValueError,
		match=r'clustering\.stratigraphic_hmm\.transition\.extra',
	):
		resolve_clustering_config(cfg)


def test_minibatch_kmeans_clustering_config_does_not_require_hmm_mapping() -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering'].pop('stratigraphic_hmm', None)

	resolved = resolve_clustering_config(cfg)

	assert resolved['clustering']['method'] == 'minibatch_kmeans'
	assert 'stratigraphic_hmm' not in resolved['clustering']


def test_clustering_config_validate_module_reexports_stage_resolver() -> None:
	assert validate_resolve_clustering_config is resolve_clustering_config


@pytest.mark.parametrize(
	('field', 'value', 'error'),
	[
		('n_components', 0, ValueError),
		('whiten', 'false', TypeError),
	],
)
def test_clustering_config_validates_pca(
	field: str,
	value: object,
	error: type[Exception],
) -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['pca'][field] = value

	with pytest.raises(error, match=field):
		resolve_clustering_config(cfg)


@pytest.mark.parametrize(
	('mode', 'group_by', 'message'),
	[
		('token_phase', 'token_phase', 'mode'),
		('local_token_position', 'survey_id', 'group_by'),
	],
)
def test_clustering_config_validates_residualization_local_token_contract(
	mode: str,
	group_by: str,
	message: str,
) -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['residualization'] = {
		'enabled': True,
		'mode': mode,
		'group_by': group_by,
		'add_global_mean_back': True,
		'min_group_count': 32,
	}

	with pytest.raises(ValueError, match=message):
		resolve_clustering_config(cfg)


def test_clustering_output_dir_accepts_canonical_nopims_path() -> None:
	cfg = _minimal_clustering_config()

	resolved = resolve_clustering_config(cfg)

	assert resolved['clustering']['output_dir'].endswith(
		'/model_a/ten_surveys/overlap_x16/k4_6_8_pca16',
	)


def test_clustering_output_dir_enforces_canonical_nopims_shape() -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['output_dir'] = (
		'/artifacts/clustering/nopims/pretrain_v1/model_a/ten_surveys/overlap_x16'
	)

	with pytest.raises(ValueError, match=r'<CLUSTER_SPEC>'):
		resolve_clustering_config(cfg)


def test_active_default_nopims_clustering_yaml_resolves() -> None:
	resolve_clustering_config(load_config(CONFIG_DIR / 'cluster_embeddings.yaml'))


def _minimal_clustering_config() -> dict[str, object]:
	return deepcopy(
		{
			'paths': {'artifact_root': '/artifacts'},
			'embeddings': {
				'input_dir': (
					'/artifacts/embeddings/nopims/pretrain_v1/'
					'model_a/ten_surveys/overlap_x16'
				),
			},
			'clustering': {
				'output_dir': (
					'/artifacts/clustering/nopims/pretrain_v1/'
					'model_a/ten_surveys/overlap_x16/k4_6_8_pca16'
				),
				'embedding_normalization': 'l2',
				'residualization': {'enabled': False},
				'pca': {'enabled': True, 'n_components': 16, 'whiten': False},
				'sample_tokens': 100000,
				'method': 'minibatch_kmeans',
				'k_values': [4, 6, 8],
				'minibatch_size': 8192,
				'seed': 42,
			},
		},
	)


def _stratigraphic_hmm_config() -> dict[str, object]:
	return {
		'emission_source': 'embedding',
		'iterations': 10,
		'z_axis': 2,
		'z_direction': 'increasing_downward',
		'transition': {
			'same_cost': 0.0,
			'advance_cost': 0.25,
			'jump_cost': 1.0,
			'reverse_cost': 1000000.0,
			'forbid_reverse': True,
			'max_jump': None,
		},
		'init': {'order_by': 'mean_z'},
		'update': {'empty_cluster_policy': 'keep_previous'},
	}


def _path_prior_config() -> dict[str, object]:
	return {
		'enabled': True,
		'initial_state': {
			'mode': 'shallow_anchor',
			'weight': 0.5,
		},
		'terminal_state': {
			'mode': 'deep_anchor',
			'weight': 0.5,
		},
		'expected_boundaries': {
			'enabled': False,
			'target': 'auto_k_minus_1',
			'weight': 0.1,
		},
	}


def _set_nested(
	parent: dict[str, object],
	path: tuple[str, ...],
	value: object,
) -> None:
	current = parent
	for key in path[:-1]:
		current = current[key]
		assert isinstance(current, dict)
	current[path[-1]] = value
