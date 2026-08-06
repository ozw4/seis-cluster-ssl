from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.cluster_visualization import (
	resolve_cluster_visualization_config,
)
from seis_ssl_cluster.config.validate import (
	resolve_cluster_visualization_config as validate_resolve_cluster_visualization_config,
)

CONFIG_DIR = Path('proc/configs/seis_ssl_cluster')


def test_cluster_visualization_config_resolves_from_stage_module() -> None:
	resolved = resolve_cluster_visualization_config(_minimal_visualization_config())

	assert resolved['stage'] == 'visualize_clusters'
	assert resolved['visualization']['modes'] == ['token', 'voxel']
	assert resolved['visualization']['summaries']['enabled'] is True


def test_cluster_visualization_validate_module_reexports_stage_resolver() -> None:
	assert (
		validate_resolve_cluster_visualization_config
		is resolve_cluster_visualization_config
	)


@pytest.mark.parametrize(
	('comparison', 'error', 'message'),
	[
		({'enabled': True}, ValueError, 'alpha'),
		({'enabled': 'true', 'alpha': 0.35}, TypeError, 'enabled'),
		({'enabled': False, 'alpha': 1.5}, ValueError, 'alpha'),
	],
)
def test_cluster_visualization_config_validates_amplitude_comparison_contract(
	comparison: dict[str, object],
	error: type[Exception],
	message: str,
) -> None:
	cfg = _minimal_visualization_config()
	cfg['visualization']['amplitude_comparison'] = comparison

	with pytest.raises(error, match=message):
		resolve_cluster_visualization_config(cfg)


def test_cluster_visualization_config_accepts_disabled_amplitude_comparison() -> None:
	cfg = _minimal_visualization_config()
	cfg['visualization']['amplitude_comparison'] = {
		'enabled': False,
		'alpha': 0.35,
	}

	resolved = resolve_cluster_visualization_config(cfg)

	assert resolved['visualization']['amplitude_comparison'] == {
		'enabled': False,
		'alpha': 0.35,
	}


def test_cluster_visualization_output_dir_accepts_canonical_nopims_path() -> None:
	cfg = _minimal_visualization_config()

	resolved = resolve_cluster_visualization_config(cfg)

	assert resolved['visualization']['output_dir'].endswith(
		'/model_a/ten_surveys/overlap_x16/k4_6_8_pca16/voxel_cmp_xy750_xz150',
	)


def test_cluster_visualization_output_dir_preserves_explicit_path() -> None:
	cfg = _minimal_visualization_config()
	explicit_output = (
		'/artifacts/visualizations/clusters/nopims/pretrain_v1/'
		'model_a/ten_surveys/overlap_x16/k4_6_8_pca16'
	)
	cfg['visualization']['output_dir'] = explicit_output

	resolved = resolve_cluster_visualization_config(cfg)

	assert resolved['visualization']['output_dir'] == explicit_output


@pytest.mark.parametrize(
	'relationship',
	['equal', 'output_inside_input', 'input_inside_output'],
)
def test_cluster_visualization_rejects_overlapping_input_and_output_directories(
	relationship: str,
) -> None:
	cfg = _minimal_visualization_config()
	if relationship == 'equal':
		cfg['visualization']['output_dir'] = cfg['clustering']['input_dir']
	elif relationship == 'output_inside_input':
		cfg['visualization']['output_dir'] = (
			f"{cfg['clustering']['input_dir']}/plots"
		)
	else:
		cfg['clustering']['input_dir'] = (
			f"{cfg['visualization']['output_dir']}/clusters"
		)

	with pytest.raises(ValueError, match='must be disjoint directories'):
		resolve_cluster_visualization_config(cfg)


def test_cluster_visualization_preserves_disjoint_explicit_paths_with_runs() -> None:
	cfg = _minimal_visualization_config()
	input_dir = '/external/runs/clustering/../clustering/model_a'
	output_dir = '/external/runs/visualizations/model_a'
	cfg['clustering']['input_dir'] = input_dir
	cfg['visualization']['output_dir'] = output_dir

	resolved = resolve_cluster_visualization_config(cfg)

	assert resolved['clustering']['input_dir'] == input_dir
	assert resolved['visualization']['output_dir'] == output_dir


def test_active_default_nopims_cluster_visualization_yaml_resolves() -> None:
	resolve_cluster_visualization_config(
		load_config(CONFIG_DIR / 'visualize_clusters.yaml'),
	)


def _minimal_visualization_config() -> dict[str, object]:
	return deepcopy(
		{
			'paths': {'artifact_root': '/artifacts'},
			'clustering': {
				'input_dir': (
					'/artifacts/clustering/nopims/pretrain_v1/'
					'model_a/ten_surveys/overlap_x16/k4_6_8_pca16'
				),
			},
			'visualization': {
				'output_dir': (
					'/artifacts/visualizations/clusters/nopims/pretrain_v1/'
					'model_a/ten_surveys/overlap_x16/k4_6_8_pca16/'
					'voxel_cmp_xy750_xz150'
				),
				'survey_ids': [],
				'modes': ['token', 'voxel'],
				'reconstruct_voxel': True,
				'allow_all_surveys_for_voxel_reconstruction': False,
				'skip_existing_voxel_labels': True,
				'max_voxel_output_gib': 50.0,
				'allow_large_voxel_output': False,
				'slice_coordinate_space': 'voxel',
				'xy_slices': [750],
				'xz_slices': [150],
				'dpi': 160,
				'invalid_color': 'lightgray',
				'amplitude_underlay': {'enabled': True, 'alpha': 0.35},
				'amplitude_comparison': {'enabled': True, 'alpha': 0.35},
				'summaries': {'enabled': True, 'include_amplitude_norm': False},
			},
		},
	)
