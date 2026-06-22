from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.clustering.features import (
	discover_embedding_inputs,
	extract_token_features,
	iter_valid_feature_batches,
	valid_flat_indices,
	validate_compatible_embedding_inputs,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_discover_embedding_inputs_and_extract_valid_features(tmp_path: Path) -> None:
	_write_embedding_artifacts(
		tmp_path,
		'survey_b',
		embeddings=np.arange(12, dtype=np.float32).reshape(2, 2, 1, 3),
		valid=np.array([[[True], [False]], [[True], [True]]]),
	)
	_write_embedding_artifacts(
		tmp_path,
		'survey_a',
		embeddings=np.arange(100, 112, dtype=np.float32).reshape(2, 2, 1, 3),
		valid=np.array([[[False], [True]], [[False], [True]]]),
	)

	inputs = discover_embedding_inputs(tmp_path)

	assert [item.survey_id for item in inputs] == ['survey_a', 'survey_b']
	indices = valid_flat_indices(inputs[1])
	assert indices.tolist() == [0, 2, 3]
	features = extract_token_features(inputs[1], indices)
	np.testing.assert_array_equal(
		features,
		np.array(
			[
				[0.0, 1.0, 2.0],
				[6.0, 7.0, 8.0],
				[9.0, 10.0, 11.0],
			],
			dtype=np.float32,
		),
	)

	batches = list(iter_valid_feature_batches(inputs[1], batch_size=2))

	assert [batch.token_indices.tolist() for batch in batches] == [[0, 2], [3]]
	assert [batch.features.shape for batch in batches] == [(2, 3), (1, 3)]


def test_discover_embedding_inputs_requires_complete_artifacts(
	tmp_path: Path,
) -> None:
	np.save(tmp_path / 'survey_a.embeddings.npy', np.zeros((1, 1, 1, 2)))

	with pytest.raises(FileNotFoundError, match='missing embedding artifacts'):
		discover_embedding_inputs(tmp_path)


def test_embedding_compatibility_allows_survey_specific_grid_and_paths(
	tmp_path: Path,
) -> None:
	_write_embedding_artifacts(
		tmp_path,
		'survey_a',
		embeddings=np.ones((1, 1, 1, 2), dtype=np.float32),
		valid=np.ones((1, 1, 1), dtype=np.bool_),
	)
	_write_embedding_artifacts(
		tmp_path,
		'survey_b',
		embeddings=np.ones((2, 1, 1, 2), dtype=np.float32),
		valid=np.ones((2, 1, 1), dtype=np.bool_),
	)
	inputs = discover_embedding_inputs(tmp_path)

	signature = validate_compatible_embedding_inputs(inputs)

	assert signature['embedding_dim'] == 2
	assert signature['checkpoint_sha256'] == 'checkpoint-a'


def test_extract_token_features_reports_non_finite_survey(
	tmp_path: Path,
) -> None:
	_write_embedding_artifacts(
		tmp_path,
		'survey_bad',
		embeddings=np.array([[[[np.nan, 1.0]]]], dtype=np.float32),
		valid=np.ones((1, 1, 1), dtype=np.bool_),
	)
	embedding_input = discover_embedding_inputs(tmp_path)[0]

	with pytest.raises(ValueError, match='survey_bad'):
		extract_token_features(embedding_input, [0])


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
