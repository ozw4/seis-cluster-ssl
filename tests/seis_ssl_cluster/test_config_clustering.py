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
