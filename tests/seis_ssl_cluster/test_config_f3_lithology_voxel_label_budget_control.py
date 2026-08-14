"""Focused contracts for the current-code K=6 low-label control config."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_control import (
	CURRENT_K6_MODEL_ID,
	CURRENT_K6_MODEL_TAG,
	EXPECTED_COMPARISONS,
	f3_lithology_voxel_label_budget_control_config_from_mapping,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_resolves_fixed_current_k6_control_contract(tmp_path: Path) -> None:
	config = f3_lithology_voxel_label_budget_control_config_from_mapping(
		_mapping(tmp_path)
	)

	assert config.candidate.model_id == CURRENT_K6_MODEL_ID
	assert config.candidate.model_tag == CURRENT_K6_MODEL_TAG
	assert config.job_count == 15
	assert config.decoder_seed(4) == 42004
	assert config.comparisons == EXPECTED_COMPARISONS
	assert config.model_by_role == {CURRENT_K6_MODEL_ID: config.candidate}
	assert config.reports_dir == config.output_root / 'reports'
	assert config.reports_root == tmp_path / 'reports'
	assert config.publish.reports_root == config.reports_root
	assert config.to_dict()['paths']['reports_root'] == str(config.reports_root)
	assert config.to_dict()['candidate'] == {
		'model_id': CURRENT_K6_MODEL_ID,
		'model_tag': CURRENT_K6_MODEL_TAG,
		'embeddings_dir': str(config.candidate.embeddings_dir),
	}


@pytest.mark.parametrize(
	('section', 'key'),
	[
		(None, 'unexpected'),
		('paths', 'unexpected'),
		('references', 'unexpected'),
		('candidate', 'unexpected'),
		('train', 'unexpected'),
		('decision', 'unexpected'),
		('outputs', 'unexpected'),
		('publish', 'unexpected'),
	],
)
def test_rejects_unknown_keys(
	tmp_path: Path, section: str | None, key: str
) -> None:
	raw = _mapping(tmp_path)
	target = raw if section is None else raw[section]
	assert isinstance(target, dict)
	target[key] = True

	with pytest.raises(ValueError, match='not allowed'):
		f3_lithology_voxel_label_budget_control_config_from_mapping(raw)


@pytest.mark.parametrize(
	('section', 'key', 'value', 'match'),
	[
		('candidate', 'model_id', 'm1', 'candidate.model_id'),
		('candidate', 'model_tag', 'wrong-tag', 'candidate.model_tag'),
		('references', 'mae_model_id', 'other', 'references.mae_model_id'),
		('seed_policy', 'base_seed', 1, 'seed policy'),
		('train', 'steps_per_epoch', 439, 'train settings'),
		('evaluation', 'chunk_size_x', 4, 'evaluation settings'),
		('outputs', 'overwrite', True, 'outputs.overwrite'),
		('decision', 'drift_absolute_mean_delta', 0.02, 'drift_absolute'),
		('decision', 'major_degradation_delta', -0.04, 'exactly -0.05'),
	],
)
def test_rejects_scientific_contract_drift(
	tmp_path: Path, section: str, key: str, value: object, match: str
) -> None:
	raw = _mapping(tmp_path)
	target = raw[section]
	assert isinstance(target, dict)
	target[key] = value

	with pytest.raises(ValueError, match=match):
		f3_lithology_voxel_label_budget_control_config_from_mapping(raw)


def test_preserves_explicit_artifact_and_publish_paths(
	tmp_path: Path,
) -> None:
	raw = _mapping(tmp_path)
	candidate = raw['candidate']
	assert isinstance(candidate, dict)
	candidate['embeddings_dir'] = str(tmp_path / 'outside-artifacts')
	resolved = f3_lithology_voxel_label_budget_control_config_from_mapping(raw)
	assert resolved.candidate.embeddings_dir == tmp_path / 'outside-artifacts'

	raw = _mapping(tmp_path)
	publish = raw['publish']
	assert isinstance(publish, dict)
	publish['output_dir'] = str(tmp_path / 'outside-results')
	resolved = f3_lithology_voxel_label_budget_control_config_from_mapping(raw)
	assert resolved.publish.output_dir == tmp_path / 'outside-results'


def test_decoder_seed_rejects_unconfigured_subsample_seed(tmp_path: Path) -> None:
	config = f3_lithology_voxel_label_budget_control_config_from_mapping(
		_mapping(tmp_path)
	)

	with pytest.raises(ValueError, match='unknown configured'):
		config.decoder_seed(6)


def _mapping(tmp_path: Path) -> dict[str, object]:
	artifact_root = tmp_path / 'artifacts'
	f3_root = tmp_path / 'f3'
	reports_root = tmp_path / 'reports'
	return {
		'paths': {
			'artifact_root': str(artifact_root),
			'f3_root': str(f3_root),
			'reports_root': str(reports_root),
		},
		'dataset': {
			'name': 'f3_facies_benchmark',
			'version': 'facies_benchmark_v1',
		},
		'references': {
			'dataset_manifest': str(
				artifact_root
				/ 'lithology/f3/facies_benchmark_v1/voxel_label_budget_v1/'
				'original_split/voxel_label_budget_dataset_manifest.json'
			),
			'historical_run_manifest': str(
				artifact_root
				/ 'lithology/f3/facies_benchmark_v1/voxel_label_budget_v1/'
				'original_split/voxel_label_budget_run_manifest.json'
			),
			'mae_model_id': 'mae',
			'historical_m1_model_id': 'm1',
		},
		'candidate': {
			'model_id': CURRENT_K6_MODEL_ID,
			'model_tag': CURRENT_K6_MODEL_TAG,
			'embeddings_dir': str(
				artifact_root
				/ 'embeddings/f3/facies_benchmark_v1'
				/ CURRENT_K6_MODEL_TAG
				/ 'overlap_x16'
			),
		},
		'budgets': ['cap25', 'cap50', 'cap100'],
		'subsample_seeds': [0, 1, 2, 3, 4],
		'seed_policy': {'base_seed': 42000, 'add_subsample_seed': True},
		'labels': {
			'seismic_volume': str(artifact_root / 'registry/volumes/f3/f3.npy'),
			'source_label_volume': str(
				artifact_root / 'registry/volumes/f3/labels.npy'
			),
			'source_label_segy': str(f3_root / 'f3_labels.sgy'),
			'png_label_inventory': str(artifact_root / 'inspection/f3/labels.csv'),
			'segy_geometry_json': str(artifact_root / 'inspection/f3/geometry.json'),
			'class_info': str(artifact_root / 'inspection/f3/class_info.json'),
		},
		'decoder': {
			'spec': 'frozen_embedding_decoder_nearest_voxel_ln_v1',
			'embedding_dim': 384,
			'class_count': 6,
			'hidden_channels': [128, 64, 32],
			'upsample_factors': [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
			'upsample_mode': 'nearest',
			'normalization': 'voxelwise_layer_norm',
		},
		'tiles': {
			'core_size_tokens': [8, 8, 8],
			'context_halo_tokens': [1, 1, 1],
		},
		'train': {
			'epochs': 50,
			'batch_size': 1,
			'learning_rate': 0.001,
			'weight_decay': 0.0001,
			'class_weight': 'balanced',
			'sampling_mode': 'uniform_tiles_with_replacement',
			'steps_per_epoch': 440,
			'num_workers': 0,
			'amp': True,
			'gradient_clip_norm': 1.0,
		},
		'inference': {'write_probabilities': False},
		'evaluation': {
			'monitored_class_ids': [3, 5],
			'boundary_tolerances': [2, 4],
			'boundary_region_radii': [2, 4],
			'chunk_size_x': 8,
		},
		'report': {
			'selected_slices': {'inline': [], 'crossline': []},
			'dpi': 150,
			'include_confidence': False,
			'amplitude_clip_percentiles': [1.0, 99.0],
		},
		'comparisons': [list(pair) for pair in EXPECTED_COMPARISONS],
		'decision': {
			'minimum_positive_budgets': 2,
			'minimum_primary_wins': 4,
			'drift_absolute_mean_delta': 0.01,
			'drift_budget_count': 2,
			'monitored_class_ids': [3, 5],
			'major_degradation_delta': -0.05,
			'systematic_degradation_budget_count': 2,
		},
		'outputs': {
			'output_root': str(
				artifact_root
				/ 'lithology/f3/facies_benchmark_v1/'
				'voxel_label_budget_current_k6_control_v1/original_split'
			),
			'overwrite': False,
		},
		'publish': {
			'enabled': True,
			'output_dir': str(
				reports_root
				/ 'f3/facies_benchmark_v1/'
				'strat_hmm_m1_current_k6_control_v1'
			),
			'max_file_size_mb': 10,
			'overwrite': True,
		},
	}
