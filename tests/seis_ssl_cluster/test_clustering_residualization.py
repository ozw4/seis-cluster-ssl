from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.clustering.features import discover_embedding_inputs, valid_flat_indices
from seis_ssl_cluster.clustering.residualization import (
	fit_local_token_position_residualizer,
	read_residualizer_npz,
	residualization_keys_for_flat_indices,
	token_phase_keys_for_grid,
	write_residualizer_npz,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_token_phase_key_generation_uses_stride_tokens_and_valid_mask() -> None:
	valid = np.zeros((5, 7, 9), dtype=np.bool_)
	valid[0, 0, 0] = True
	valid[1, 2, 3] = True
	valid[4, 6, 8] = True

	keys = token_phase_keys_for_grid(
		(5, 7, 9),
		patch_size_xyz=(2, 3, 4),
		window_size_xyz=(8, 12, 16),
		overlap_xyz=(4, 6, 8),
		valid_mask=valid,
	)

	np.testing.assert_array_equal(
		keys,
		np.array(
			[
				[0, 0, 0],
				[1, 0, 1],
				[0, 0, 0],
			],
			dtype=np.int64,
		),
	)


def test_residualizer_removes_group_mean_with_global_mean_added_back() -> None:
	group_keys = np.array(
		[[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]],
		dtype=np.int64,
	)
	embeddings = np.array(
		[
			[11.0, 1.0],
			[13.0, 3.0],
			[-3.0, 7.0],
			[-1.0, 9.0],
		],
		dtype=np.float32,
	)

	residualizer = fit_local_token_position_residualizer(
		embeddings,
		group_keys,
		group_by='token_phase',
		add_global_mean_back=True,
		min_group_count=1,
	)
	transformed = residualizer.transform(embeddings, group_keys)
	global_mean = embeddings.mean(axis=0)

	for key in np.unique(group_keys, axis=0):
		group = np.all(group_keys == key, axis=1)
		np.testing.assert_allclose(transformed[group].mean(axis=0), global_mean)


def test_residualizer_without_global_mean_back_centers_groups_at_zero() -> None:
	group_keys = np.array(
		[[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]],
		dtype=np.int64,
	)
	embeddings = np.array(
		[[2.0, 3.0], [4.0, 5.0], [10.0, 20.0], [12.0, 22.0]],
		dtype=np.float32,
	)

	residualizer = fit_local_token_position_residualizer(
		embeddings,
		group_keys,
		group_by='token_phase',
		add_global_mean_back=False,
		min_group_count=1,
	)
	transformed = residualizer.transform(embeddings, group_keys)

	for key in np.unique(group_keys, axis=0):
		group = np.all(group_keys == key, axis=1)
		np.testing.assert_allclose(transformed[group].mean(axis=0), 0.0)


def test_min_group_count_uses_global_mean_as_untrusted_group_mean() -> None:
	group_keys = np.array(
		[[0, 0, 0], [0, 0, 0], [1, 0, 0]],
		dtype=np.int64,
	)
	embeddings = np.array(
		[[0.0, 0.0], [2.0, 2.0], [100.0, 100.0]],
		dtype=np.float32,
	)

	residualizer = fit_local_token_position_residualizer(
		embeddings,
		group_keys,
		group_by='token_phase',
		add_global_mean_back=True,
		min_group_count=2,
	)

	np.testing.assert_array_equal(
		residualizer.group_means[(1, 0, 0)],
		residualizer.global_mean,
	)
	np.testing.assert_array_equal(
		residualizer.transform(embeddings, group_keys)[2],
		embeddings[2],
	)


def test_valid_mask_false_tokens_are_excluded_from_generated_keys(tmp_path: Path) -> None:
	_write_embedding_artifacts(
		tmp_path,
		'survey_a',
		embeddings=np.ones((2, 2, 2, 2), dtype=np.float32),
		valid=np.array(
			[
				[[True, False], [False, True]],
				[[True, False], [False, False]],
			],
		),
	)
	embedding_input = discover_embedding_inputs(tmp_path)[0]
	indices = valid_flat_indices(embedding_input)

	keys = residualization_keys_for_flat_indices(
		embedding_input,
		indices,
		group_by='token_phase',
	)

	assert keys.shape == (3, 3)
	np.testing.assert_array_equal(keys, np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0]]))


def test_residualizer_npz_round_trip_preserves_transform(tmp_path: Path) -> None:
	group_keys = np.array(
		[[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]],
		dtype=np.int64,
	)
	embeddings = np.arange(8, dtype=np.float32).reshape(4, 2)
	residualizer = fit_local_token_position_residualizer(
		embeddings,
		group_keys,
		group_by='token_phase',
		add_global_mean_back=True,
		min_group_count=1,
	)

	path = tmp_path / 'residualizer.npz'
	write_residualizer_npz(path, residualizer)
	loaded = read_residualizer_npz(path)

	np.testing.assert_allclose(
		loaded.transform(embeddings, group_keys),
		residualizer.transform(embeddings, group_keys),
	)


def test_local_token_position_grouping_requires_exact_metadata(tmp_path: Path) -> None:
	_write_embedding_artifacts(
		tmp_path,
		'survey_a',
		embeddings=np.ones((1, 1, 1, 2), dtype=np.float32),
		valid=np.ones((1, 1, 1), dtype=np.bool_),
	)
	embedding_input = discover_embedding_inputs(tmp_path)[0]

	with pytest.raises(ValueError, match='per-token local position metadata'):
		residualization_keys_for_flat_indices(
			embedding_input,
			np.array([0], dtype=np.int64),
			group_by='local_token_position',
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
		json.dumps(
			{
				'survey_id': survey_id,
				'source_amplitude_path': f'{survey_id}.npy',
				'checkpoint_path': 'checkpoint.pt',
				'checkpoint_sha256': 'checkpoint-a',
				'model_geometry': {
					'name': 'amp_mae3d',
					'encoder_dim': int(embeddings.shape[-1]),
					'encoder_depth': 1,
					'encoder_heads': 1,
				},
				'patch_size': [1, 1, 1],
				'token_grid_shape': list(embeddings.shape[:3]),
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
			},
		)
		+ '\n',
		encoding='utf-8',
	)
