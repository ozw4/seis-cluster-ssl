from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.clustering.features import (
	discover_embedding_inputs,
	extract_token_features,
)
from seis_ssl_cluster.clustering.kmeans import (
	PCASettings,
	ResidualizationSettings,
	apply_residualizer_to_sample,
	fit_preprocessor,
	fit_residualizer,
)
from seis_ssl_cluster.clustering.residualization import (
	residualization_keys_for_flat_indices,
)
from seis_ssl_cluster.clustering.stratigraphic_hmm import (
	initialize_ordered_centers,
	prepare_feature_batch_for_indices,
	sample_token_z_coordinates,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_sample_token_z_coordinates_preserves_sample_row_order(tmp_path: Path) -> None:
	input_dir = tmp_path / 'embeddings'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_b',
		embeddings=np.ones((1, 2, 4, 2), dtype=np.float32),
		valid=np.ones((1, 2, 4), dtype=np.bool_),
	)
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.ones((2, 1, 4, 2), dtype=np.float32),
		valid=np.ones((2, 1, 4), dtype=np.bool_),
	)
	embedding_inputs = tuple(discover_embedding_inputs(input_dir))
	per_survey = {
		'survey_a': np.array([3, 4, 7], dtype=np.int64),
		'survey_b': np.array([6, 1], dtype=np.int64),
	}

	z = sample_token_z_coordinates(embedding_inputs, per_survey)

	np.testing.assert_array_equal(z, np.array([3, 0, 3, 2, 1], dtype=np.int32))


def test_initialize_ordered_centers_orders_by_mean_z() -> None:
	features = np.array(
		[
			[-5.0, 0.0],
			[-4.8, 0.1],
			[5.0, 0.0],
			[5.2, -0.1],
		],
		dtype=np.float32,
	)
	sample_z = np.array([1, 2, 8, 9], dtype=np.int32)

	centers = initialize_ordered_centers(
		features,
		sample_z,
		k=2,
		batch_size=4,
		seed=42,
	)

	assert centers.dtype == np.float32
	assert centers.shape == (2, 2)
	assert centers[0, 0] < 0.0
	assert centers[1, 0] > 0.0


def test_prepare_feature_batch_for_indices_matches_kmeans_preprocessing(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	input_dir.mkdir()
	embeddings = np.array(
		[
			[[[10.0, 0.0], [11.0, 1.0]], [[0.0, 10.0], [1.0, 11.0]]],
			[[[12.0, 0.0], [13.0, 1.0]], [[2.0, 10.0], [3.0, 11.0]]],
		],
		dtype=np.float32,
	)
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=embeddings,
		valid=np.ones((2, 2, 2), dtype=np.bool_),
	)
	embedding_input = discover_embedding_inputs(input_dir)[0]
	training_indices = np.arange(8, dtype=np.int64)
	training_features = extract_token_features(embedding_input, training_indices)
	per_survey = {'survey_a': training_indices}
	residualizer = fit_residualizer(
		training_features,
		embedding_inputs=(embedding_input,),
		per_survey_token_indices=per_survey,
		settings=ResidualizationSettings(
			enabled=True,
			mode='local_token_position',
			group_by='token_phase',
			add_global_mean_back=True,
			min_group_count=1,
		),
	)
	training_input_features = apply_residualizer_to_sample(
		training_features,
		embedding_inputs=(embedding_input,),
		per_survey_token_indices=per_survey,
		residualizer=residualizer,
	)
	preprocessor = fit_preprocessor(
		training_input_features,
		normalization='none',
		pca=PCASettings(enabled=True, n_components=2, whiten=False),
		seed=7,
	)
	batch_indices = np.array([0, 3, 6], dtype=np.int64)

	batch = prepare_feature_batch_for_indices(
		embedding_input,
		batch_indices,
		residualizer=residualizer,
		preprocessor=preprocessor,
	)
	expected_features = residualizer.transform(
		extract_token_features(embedding_input, batch_indices),
		residualization_keys_for_flat_indices(
			embedding_input,
			batch_indices,
			group_by='token_phase',
		),
	)
	expected = np.asarray(preprocessor.transform(expected_features), dtype=np.float32)

	assert batch.shape == (3, 2)
	assert np.all(np.isfinite(batch))
	np.testing.assert_allclose(batch, expected, rtol=1e-6, atol=1e-6)
	empty = prepare_feature_batch_for_indices(
		embedding_input,
		np.empty(0, dtype=np.int64),
		residualizer=residualizer,
		preprocessor=preprocessor,
	)
	assert empty.shape == (0, 2)
	assert empty.dtype == np.float32


def test_initialize_ordered_centers_places_empty_clusters_last_deterministic() -> None:
	features = np.array(
		[
			[0.0, 0.0],
			[0.0, 0.0],
			[10.0, 0.0],
		],
		dtype=np.float32,
	)
	sample_z = np.array([1, 2, 9], dtype=np.int32)

	first = initialize_ordered_centers(
		features,
		sample_z,
		k=3,
		batch_size=3,
		seed=0,
	)
	second = initialize_ordered_centers(
		features,
		sample_z,
		k=3,
		batch_size=3,
		seed=0,
	)

	np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
	np.testing.assert_allclose(
		first,
		np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0]], dtype=np.float32),
		rtol=1e-6,
		atol=1e-6,
	)


def _write_embedding_artifacts(
	root: Path,
	survey_id: str,
	*,
	embeddings: np.ndarray,
	valid: np.ndarray,
) -> None:
	np.save(root / f'{survey_id}.embeddings.npy', embeddings)
	np.save(root / f'{survey_id}.valid_tokens.npy', valid.astype(np.bool_))
	(root / f'{survey_id}.embedding_metadata.json').write_text(
		json.dumps(_embedding_metadata(survey_id, embeddings.shape[:3])) + '\n',
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
			'encoder_dim': 2,
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
