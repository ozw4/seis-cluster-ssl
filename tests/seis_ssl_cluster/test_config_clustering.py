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
