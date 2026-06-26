from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.clustering import run_embedding_clustering

if TYPE_CHECKING:
	from pathlib import Path


def test_run_embedding_clustering_writes_deterministic_labels_for_multiple_k(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	first_output = tmp_path / 'clusters-a'
	second_output = tmp_path / 'clusters-b'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.array(
			[
				[[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]],
				[[[0.0, 1.0, 0.0], [0.0, 0.9, 0.1]]],
			],
			dtype=np.float32,
		),
		valid=np.array([[[True, True]], [[True, False]]]),
	)
	_write_embedding_artifacts(
		input_dir,
		'survey_b',
		embeddings=np.array(
			[
				[[[0.0, 0.0, 1.0], [0.1, 0.0, 0.9]]],
				[[[1.0, 0.1, 0.0], [0.0, 1.0, 0.1]]],
			],
			dtype=np.float32,
		),
		valid=np.array([[[True, True]], [[True, True]]]),
	)

	first = run_embedding_clustering(_config(input_dir, first_output))
	second = run_embedding_clustering(_config(input_dir, second_output))

	assert [result.k for result in first.results] == [2, 3]
	assert [result.k for result in second.results] == [2, 3]
	assert first.sample.sample_count == 7
	assert first.sample.total_valid_count == 7
	for output_dir in (first_output, second_output):
		for k in (2, 3):
			assert (output_dir / 'models' / f'k{k}' / 'preprocessor.joblib').is_file()
			assert (output_dir / 'models' / f'k{k}' / 'kmeans.joblib').is_file()
			assert (output_dir / 'models' / f'k{k}' / 'cluster_centers.npy').is_file()
			metadata = json.loads(
				(output_dir / 'models' / f'k{k}' / 'clustering_metadata.json')
				.read_text(encoding='utf-8'),
			)
			assert metadata['sample']['count'] == 7
			assert metadata['invalid_token_count'] == 1
			assert metadata['embedding_compatibility_signature']['embedding_dim'] == 3
			assert [
				item['survey_id']
				for item in metadata['embedding_inputs']
			] == ['survey_a', 'survey_b']
			assert all(
				item['metadata_path'].endswith('.embedding_metadata.json')
				and len(item['metadata_sha256']) == 64
				for item in metadata['embedding_inputs']
			)

	survey_a_first = np.load(
		first_output / 'labels' / 'k2' / 'survey_a.cluster_labels_token.npy',
	)
	survey_a_second = np.load(
		second_output / 'labels' / 'k2' / 'survey_a.cluster_labels_token.npy',
	)
	np.testing.assert_array_equal(survey_a_first, survey_a_second)
	assert survey_a_first.shape == (2, 1, 2)
	assert survey_a_first[1, 0, 1] == -1
	assert np.all(survey_a_first[np.array([[[True, True]], [[True, False]]])] >= 0)

	for result in first.results:
		assert sum(result.cluster_counts.values()) == 7
		assert result.invalid_token_count == 1


def test_run_embedding_clustering_rejects_different_checkpoint_hashes(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.ones((1, 1, 2, 3), dtype=np.float32),
		valid=np.ones((1, 1, 2), dtype=np.bool_),
	)
	_write_embedding_artifacts(
		input_dir,
		'survey_b',
		embeddings=np.ones((1, 1, 2, 3), dtype=np.float32),
		valid=np.ones((1, 1, 2), dtype=np.bool_),
		metadata_updates={'checkpoint_sha256': 'checkpoint-b'},
	)

	with pytest.raises(
		ValueError,
		match=r"survey_a.*survey_b.*checkpoint_sha256",
	):
		run_embedding_clustering(_config(input_dir, tmp_path / 'clusters'))


def test_run_embedding_clustering_rejects_different_model_geometry(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.ones((1, 1, 2, 3), dtype=np.float32),
		valid=np.ones((1, 1, 2), dtype=np.bool_),
	)
	_write_embedding_artifacts(
		input_dir,
		'survey_b',
		embeddings=np.ones((1, 1, 2, 3), dtype=np.float32),
		valid=np.ones((1, 1, 2), dtype=np.bool_),
		metadata_updates={
			'model_geometry': {
				'name': 'amp_mae3d',
				'encoder_dim': 3,
				'encoder_depth': 2,
				'encoder_heads': 1,
			},
			'patch_size': [1, 2, 2],
		},
	)

	with pytest.raises(
		ValueError,
		match=r"survey_a.*survey_b.*model_geometry.*patch_size",
	):
		run_embedding_clustering(_config(input_dir, tmp_path / 'clusters'))


def test_run_embedding_clustering_rejects_different_extraction_contract(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.ones((1, 1, 2, 3), dtype=np.float32),
		valid=np.ones((1, 1, 2), dtype=np.bool_),
	)
	_write_embedding_artifacts(
		input_dir,
		'survey_b',
		embeddings=np.ones((1, 1, 2, 3), dtype=np.float32),
		valid=np.ones((1, 1, 2), dtype=np.bool_),
		metadata_updates={
			'window_size': [6, 4, 4],
			'overlap': [3, 2, 2],
			'zero_mask': {
				'enabled': False,
				'zero_atol': 0.0,
				'z_sample_influence_radius': 1,
				'xy_trace_influence_radius': 1,
			},
		},
	)

	with pytest.raises(
		ValueError,
		match=r"survey_a.*survey_b.*window_size.*overlap.*zero_mask",
	):
		run_embedding_clustering(_config(input_dir, tmp_path / 'clusters'))


def test_run_embedding_clustering_rejects_duplicate_k_values(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	input_dir.mkdir()
	config = _config(input_dir, tmp_path / 'clusters')
	config['clustering']['k_values'] = [2, 2]

	with pytest.raises(ValueError, match=r'k_values.*duplicates'):
		run_embedding_clustering(config)


def test_run_embedding_clustering_reports_non_finite_feature_survey(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.ones((1, 1, 2, 3), dtype=np.float32),
		valid=np.ones((1, 1, 2), dtype=np.bool_),
	)
	_write_embedding_artifacts(
		input_dir,
		'survey_bad',
		embeddings=np.array([[[[np.inf, 1.0, 1.0], [1.0, 1.0, 1.0]]]]),
		valid=np.ones((1, 1, 2), dtype=np.bool_),
	)

	with pytest.raises(ValueError, match=r'non-finite.*survey_bad'):
		run_embedding_clustering(_config(input_dir, tmp_path / 'clusters'))


def test_run_embedding_clustering_applies_residualization_before_pca_and_kmeans(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	output_dir = tmp_path / 'clusters'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.array(
			[
				[[[10.0, 0.0], [11.0, 1.0]], [[0.0, 10.0], [1.0, 11.0]]],
				[[[12.0, 0.0], [13.0, 1.0]], [[2.0, 10.0], [3.0, 11.0]]],
			],
			dtype=np.float32,
		),
		valid=np.ones((2, 2, 2), dtype=np.bool_),
	)
	config = _config(input_dir, output_dir)
	config['clustering']['embedding_normalization'] = 'none'
	config['clustering']['sample_tokens'] = 8
	config['clustering']['k_values'] = [2]
	config['clustering']['pca'] = {
		'enabled': True,
		'n_components': 2,
		'whiten': False,
	}
	config['clustering']['residualization'] = {
		'enabled': True,
		'mode': 'local_token_position',
		'group_by': 'token_phase',
		'add_global_mean_back': True,
		'min_group_count': 1,
	}

	result = run_embedding_clustering(config)

	assert result.results[0].k == 2
	assert (output_dir / 'models' / 'residualizer.npz').is_file()
	metadata = json.loads(
		(output_dir / 'models' / 'k2' / 'clustering_metadata.json').read_text(
			encoding='utf-8',
		),
	)
	assert metadata['residualization']['enabled'] is True
	assert metadata['residualization']['group_by'] == 'token_phase'
	assert metadata['pca']['enabled'] is True
	labels = np.load(output_dir / 'labels' / 'k2' / 'survey_a.cluster_labels_token.npy')
	assert labels.shape == (2, 2, 2)
	assert np.all(labels >= 0)


def _config(input_dir: Path, output_dir: Path) -> dict[str, object]:
	return {
		'embeddings': {'input_dir': str(input_dir)},
		'clustering': {
			'output_dir': str(output_dir),
			'embedding_normalization': 'l2',
			'residualization': {
				'enabled': False,
			},
			'pca': {
				'enabled': True,
				'n_components': 2,
				'whiten': False,
			},
			'sample_tokens': 100,
			'method': 'minibatch_kmeans',
			'k_values': [2, 3],
			'minibatch_size': 4,
			'seed': 42,
		},
	}


def _write_embedding_artifacts(
	root: Path,
	survey_id: str,
	*,
	embeddings: np.ndarray,
	valid: np.ndarray,
	metadata_updates: dict[str, object] | None = None,
) -> None:
	np.save(root / f'{survey_id}.embeddings.npy', embeddings)
	np.save(root / f'{survey_id}.valid_tokens.npy', valid.astype(np.bool_))
	metadata = _embedding_metadata(survey_id, embeddings.shape[:3])
	if metadata_updates:
		metadata.update(metadata_updates)
	(root / f'{survey_id}.embedding_metadata.json').write_text(
		json.dumps(metadata) + '\n',
		encoding='utf-8',
	)


def _embedding_metadata(
	survey_id: str,
	token_grid_shape: tuple[int, int, int],
) -> dict[str, object]:
	return {
		'survey_id': survey_id,
		'source_amplitude_path': f'{survey_id}.npy',
		'checkpoint_path': 'checkpoint.pt',
		'checkpoint_sha256': 'checkpoint-a',
		'model_geometry': {
			'name': 'amp_mae3d',
			'encoder_dim': 3,
			'encoder_depth': 1,
			'encoder_heads': 1,
		},
		'patch_size': [2, 2, 2],
		'token_grid_shape': list(token_grid_shape),
		'window_size': [4, 4, 4],
		'overlap': [2, 2, 2],
		'normalization_stats_path': f'{survey_id}.normalization_stats.json',
		'output_dtype': 'float32',
		'min_token_valid_fraction': 0.5,
		'zero_mask': {
			'enabled': True,
			'zero_atol': 0.0,
			'z_sample_influence_radius': 1,
			'xy_trace_influence_radius': 1,
		},
	}
