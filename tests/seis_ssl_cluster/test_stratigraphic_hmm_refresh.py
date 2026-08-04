# ruff: noqa: CPY001

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.clustering import stratigraphic_hmm
from seis_ssl_cluster.clustering.features import EmbeddingInput, extract_token_features
from seis_ssl_cluster.clustering.prepared_features import (
	PreparedFeatureCacheSettings,
	PreparedFeatureStore,
	prepare_feature_store,
)
from seis_ssl_cluster.clustering.stratigraphic_hmm_refresh import (
	run_warm_start_ordered_hmm_refresh,
)

if TYPE_CHECKING:
	from pathlib import Path


class _IdentityPreprocessor:
	def transform(self, features: np.ndarray) -> np.ndarray:
		return features


def test_warm_start_refresh_runs_exact_updates_and_final_decode(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	store = _make_store(tmp_path, np.array([0.0, 1.0, 10.0, 11.0]))
	previous_centers = np.array([[0.0], [10.0]], dtype=np.float32)
	previous_bytes = previous_centers.tobytes()
	decode_calls = 0
	update_calls: list[tuple[np.ndarray, str]] = []
	original_decode = stratigraphic_hmm.decode_prepared_survey_ordered_labels
	original_update = stratigraphic_hmm.update_centers_from_prepared_labels

	def record_decode(*args: object, **kwargs: object) -> np.ndarray:
		nonlocal decode_calls
		decode_calls += 1
		return original_decode(*args, **kwargs)

	def record_update(
		prepared_features: PreparedFeatureStore,
		labels_by_survey: object,
		**kwargs: object,
	) -> tuple[np.ndarray, dict[str, object]]:
		update_calls.append(
			(np.asarray(kwargs['centers']).copy(), kwargs['empty_cluster_policy'])
		)
		return original_update(prepared_features, labels_by_survey, **kwargs)

	monkeypatch.setattr(
		stratigraphic_hmm,
		'decode_prepared_survey_ordered_labels',
		record_decode,
	)
	monkeypatch.setattr(
		stratigraphic_hmm,
		'update_centers_from_prepared_labels',
		record_update,
	)
	monkeypatch.setattr(
		stratigraphic_hmm,
		'initialize_ordered_centers',
		lambda *_args, **_kwargs: (_ for _ in ()).throw(
			AssertionError('warm-start refresh must not initialize centers')
		),
	)

	result = run_warm_start_ordered_hmm_refresh(
		store,
		previous_centers,
		transition_costs=np.zeros((2, 2), dtype=np.float32),
		initial_state_costs=np.zeros(2, dtype=np.float32),
		terminal_state_costs=np.zeros(2, dtype=np.float32),
		expected_boundaries=None,
		iterations=2,
	)

	assert decode_calls == 3
	assert len(update_calls) == 2
	assert all(policy == 'keep_previous' for _, policy in update_calls)
	np.testing.assert_allclose(
		result.centers,
		np.array([[0.5], [10.5]], dtype=np.float32),
	)
	np.testing.assert_array_equal(
		result.labels_by_survey['survey_a'],
		np.array([[[0, 0, 1, 1]]], dtype=np.int32),
	)
	assert result.iteration_diagnostics[0].cluster_counts == {0: 2, 1: 2}
	assert result.iteration_diagnostics[0].center_shift_l2 == [0.5, 0.5]
	assert result.iteration_diagnostics[1].center_shift_l2 == [0.0, 0.0]
	assert result.final_state_counts == {0: 2, 1: 2}
	assert result.final_transition_counts == {
		'same': 2,
		'forward': 1,
		'reverse': 0,
		'jump': 0,
	}
	assert result.final_state_mean_z == {0: 0.5, 1: 2.5}
	assert previous_centers.tobytes() == previous_bytes
	store.close()


def test_warm_start_refresh_preserves_center_row_identity_when_depth_order_crosses(
	tmp_path: Path,
) -> None:
	store = _make_store(tmp_path, np.array([9.0, 0.0, 0.0, 0.0]))
	result = run_warm_start_ordered_hmm_refresh(
		store,
		np.array([[0.0], [10.0]], dtype=np.float32),
		transition_costs=np.zeros((2, 2), dtype=np.float32),
		initial_state_costs=np.zeros(2, dtype=np.float32),
		terminal_state_costs=np.zeros(2, dtype=np.float32),
		expected_boundaries=None,
		iterations=1,
	)

	np.testing.assert_allclose(
		result.centers,
		np.array([[0.0], [9.0]], dtype=np.float32),
	)
	assert result.final_state_mean_z == {0: 2.0, 1: 0.0}
	store.close()


def test_warm_start_refresh_validates_expected_boundaries_for_long_trace(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	store = _make_store(tmp_path, np.array([0.0, 1.0, 2.0, 3.0]))
	seen_trace_lengths: list[int] = []
	original_resolver = stratigraphic_hmm._resolve_expected_boundary_count  # noqa: SLF001

	def record_resolver(
		settings: object,
		*,
		k: int,
		valid_trace_length: int,
	) -> int | None:
		seen_trace_lengths.append(valid_trace_length)
		return original_resolver(
			settings,
			k=k,
			valid_trace_length=valid_trace_length,
		)

	monkeypatch.setattr(
		stratigraphic_hmm,
		'_resolve_expected_boundary_count',
		record_resolver,
	)
	result = run_warm_start_ordered_hmm_refresh(
		store,
		np.arange(4, dtype=np.float32).reshape(4, 1),
		transition_costs=np.zeros((4, 4), dtype=np.float32),
		initial_state_costs=np.zeros(4, dtype=np.float32),
		terminal_state_costs=np.zeros(4, dtype=np.float32),
		expected_boundaries=stratigraphic_hmm.HMMExpectedBoundariesSettings(
			enabled=True,
			target=3,
			weight=1.0,
		),
		iterations=1,
	)

	assert result.centers.shape == (4, 1)
	assert seen_trace_lengths == [4, 4, 4]
	store.close()


def test_warm_start_refresh_rejects_empty_state_and_preserves_store(
	tmp_path: Path,
) -> None:
	store = _make_store(tmp_path, np.array([0.0, 0.0]))
	cache_path = store.surveys[0].cache_path
	assert cache_path is not None
	feature_path = cache_path / 'features.npy'
	feature_bytes = feature_path.read_bytes()
	center_bytes = np.array([[0.0], [10.0]], dtype=np.float32).tobytes()

	with pytest.raises(ValueError, match='empty HMM state'):
		run_warm_start_ordered_hmm_refresh(
			store,
			np.array([[0.0], [10.0]], dtype=np.float32),
			transition_costs=np.zeros((2, 2), dtype=np.float32),
			initial_state_costs=np.zeros(2, dtype=np.float32),
			terminal_state_costs=np.zeros(2, dtype=np.float32),
			expected_boundaries=None,
			iterations=1,
		)

	assert feature_path.read_bytes() == feature_bytes
	assert np.array([[0.0], [10.0]], dtype=np.float32).tobytes() == center_bytes
	store.close()


def test_warm_start_refresh_rejects_nonfinite_prepared_features(
	tmp_path: Path,
) -> None:
	store = _make_store(tmp_path, np.array([0.0, 1.0]))
	cache_path = store.surveys[0].cache_path
	assert cache_path is not None
	feature_path = cache_path / 'features.npy'
	features = np.load(feature_path).copy()
	features[0, 0] = np.nan
	np.save(feature_path, features)
	corrupt_bytes = feature_path.read_bytes()

	with pytest.raises(ValueError, match=r'prepared features.*finite'):
		run_warm_start_ordered_hmm_refresh(
			store,
			np.array([[0.0], [1.0]], dtype=np.float32),
			transition_costs=np.zeros((2, 2), dtype=np.float32),
			initial_state_costs=np.zeros(2, dtype=np.float32),
			terminal_state_costs=np.zeros(2, dtype=np.float32),
			expected_boundaries=None,
			iterations=1,
		)

	assert feature_path.read_bytes() == corrupt_bytes
	store.close()


@pytest.mark.parametrize(
	('bad_labels', 'match'),
	[
		(np.array([[[2, 2]]], dtype=np.int32), 'out of range'),
		(np.array([[[0]]], dtype=np.int32), 'label grid shape'),
	],
)
def test_warm_start_refresh_rejects_invalid_decoder_labels(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	bad_labels: np.ndarray,
	match: str,
) -> None:
	store = _make_store(tmp_path, np.array([0.0, 1.0]))

	def return_invalid_labels(*_args: object, **_kwargs: object) -> np.ndarray:
		return bad_labels

	monkeypatch.setattr(
		stratigraphic_hmm,
		'decode_prepared_survey_ordered_labels',
		return_invalid_labels,
	)
	with pytest.raises(ValueError, match=match):
		run_warm_start_ordered_hmm_refresh(
			store,
			np.array([[0.0], [1.0]], dtype=np.float32),
			transition_costs=np.zeros((2, 2), dtype=np.float32),
			initial_state_costs=np.zeros(2, dtype=np.float32),
			terminal_state_costs=np.zeros(2, dtype=np.float32),
			expected_boundaries=None,
			iterations=1,
		)
	store.close()


@pytest.mark.parametrize(
	('centers', 'iterations', 'match'),
	[
		(np.array([[np.nan]], dtype=np.float32), 1, 'centers'),
		(np.array([[0.0], [1.0]], dtype=np.float32), 0, 'iterations'),
		(np.array([[0.0], [1.0]], dtype=np.float32), 1, 'transition_costs'),
	],
)
def test_warm_start_refresh_rejects_invalid_inputs(
	tmp_path: Path,
	centers: np.ndarray,
	iterations: int,
	match: str,
) -> None:
	store = _make_store(tmp_path, np.array([0.0, 1.0]))
	transition_costs = np.zeros((2, 2), dtype=np.float32)
	if match == 'transition_costs':
		transition_costs = np.zeros((1, 1), dtype=np.float32)
	with pytest.raises((TypeError, ValueError), match=match):
		run_warm_start_ordered_hmm_refresh(
			store,
			centers,
			transition_costs=transition_costs,
			initial_state_costs=np.zeros(2, dtype=np.float32),
			terminal_state_costs=np.zeros(2, dtype=np.float32),
			expected_boundaries=None,
			iterations=iterations,
		)
	store.close()


def _make_store(tmp_path: Path, values: np.ndarray) -> PreparedFeatureStore:
	shape = (1, 1, values.size, 1)
	embeddings_path = tmp_path / 'survey_a.embeddings.npy'
	valid_tokens_path = tmp_path / 'survey_a.valid_tokens.npy'
	metadata_path = tmp_path / 'survey_a.embedding_metadata.json'
	np.save(embeddings_path, values.astype(np.float32).reshape(shape))
	np.save(valid_tokens_path, np.ones(shape[:3], dtype=np.bool_))
	metadata_path.write_text(
		json.dumps(
			{
				'token_grid_shape': list(shape[:3]),
				'patch_size': [1, 1, 1],
				'window_size': [3, 3, 3],
				'overlap': [1, 1, 1],
			}
		)
		+ '\n',
		encoding='utf-8',
	)
	item = EmbeddingInput(
		'survey_a',
		embeddings_path,
		valid_tokens_path,
		metadata_path,
	)
	return prepare_feature_store(
		embedding_inputs=(item,),
		feature_dim=1,
		feature_mode='embedding',
		residualizer=None,
		preprocessor=_IdentityPreprocessor(),
		edge_margin_tokens=(0, 0, 0),
		settings=PreparedFeatureCacheSettings(directory=tmp_path / 'prepared'),
		default_cache_root=tmp_path / 'unused',
		prepare_batch=extract_token_features,
	)
