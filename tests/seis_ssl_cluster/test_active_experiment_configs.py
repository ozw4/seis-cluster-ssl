from __future__ import annotations

from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_cluster_visualization_config,
	resolve_clustering_config,
	resolve_embedding_extraction_config,
)
from seis_ssl_cluster.paths import DEFAULT_ARTIFACT_ROOT, ArtifactPaths, ExperimentKey

NOPIMS_ROOT = Path('experiments/nopims/pretrain_v1')
NOPIMS_EMBEDDING_CONFIGS = sorted((NOPIMS_ROOT / '20_embedding').rglob('*.yaml'))
NOPIMS_CLUSTERING_CONFIGS = sorted((NOPIMS_ROOT / '30_clustering').rglob('*.yaml'))
NOPIMS_VISUALIZATION_CONFIGS = sorted(
	(NOPIMS_ROOT / '40_visualization').rglob('*.yaml'),
)


@pytest.mark.parametrize('config_path', NOPIMS_EMBEDDING_CONFIGS)
def test_active_nopims_embedding_configs_resolve(config_path: Path) -> None:
	resolve_embedding_extraction_config(load_config(config_path))


@pytest.mark.parametrize('config_path', NOPIMS_CLUSTERING_CONFIGS)
def test_active_nopims_clustering_configs_resolve(config_path: Path) -> None:
	resolve_clustering_config(load_config(config_path))


@pytest.mark.parametrize('config_path', NOPIMS_VISUALIZATION_CONFIGS)
def test_active_nopims_cluster_visualization_configs_resolve(
	config_path: Path,
) -> None:
	resolve_cluster_visualization_config(load_config(config_path))


def test_active_nopims_overlap_x16_paths_match_artifact_paths_contract() -> None:
	model_tag = 'amp_mae_m075_mse_g0_patchnorm_clip8_vis01_v1'
	paths = ArtifactPaths(DEFAULT_ARTIFACT_ROOT)
	key = ExperimentKey(
		dataset='nopims',
		version='pretrain_v1',
		model_tag=model_tag,
		subset='ten_surveys',
		embed_spec='overlap_x16',
		cluster_spec='k6_8_whiten',
		viz_spec='voxel_cmp_xy750_xz150',
	)

	embedding = load_config(
		NOPIMS_ROOT
		/ '20_embedding'
		/ model_tag
		/ '01_ten_surveys_overlap_x16.yaml',
	)
	clustering = load_config(
		NOPIMS_ROOT
		/ '30_clustering'
		/ model_tag
		/ '01_ten_surveys_overlap_x16_k6_8_whiten.yaml',
	)
	visualization = load_config(
		NOPIMS_ROOT
		/ '40_visualization'
		/ model_tag
		/ '01_ten_surveys_overlap_x16_whiten.yaml',
	)

	assert Path(embedding['embeddings']['output_dir']) == paths.embeddings(key)
	assert Path(clustering['embeddings']['input_dir']) == paths.embeddings(key)
	assert Path(clustering['clustering']['output_dir']) == paths.clustering(key)
	assert Path(visualization['clustering']['input_dir']) == paths.clustering(key)
	assert (
		Path(visualization['visualization']['output_dir'])
		== paths.cluster_visualization(key)
	)
