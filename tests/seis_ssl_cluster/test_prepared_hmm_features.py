from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import pytest

import seis_ssl_cluster.clustering.prepared_features as prepared_features_module
from seis_ssl_cluster.clustering.features import EmbeddingInput, extract_token_features
from seis_ssl_cluster.clustering.prepared_features import (
	PreparedFeatureCacheSettings,
	PreparedSurveyFeatures,
	prepare_feature_store,
)
from seis_ssl_cluster.clustering.stratigraphic_hmm import (
	HMMTransitionSettings,
	build_ordered_transition_costs,
	decode_prepared_survey_ordered_labels,
	decode_survey_ordered_labels,
	update_centers_from_labels,
	update_centers_from_prepared_labels,
)

if TYPE_CHECKING:
	from pathlib import Path


class _IdentityPreprocessor:
	def __init__(self, version: int = 0) -> None:
		self.version = version

	def transform(self, features: np.ndarray) -> np.ndarray:
		return features


def test_prepared_decode_update_and_objective_match_on_the_fly_reference(
	tmp_path: Path,
) -> None:
	item = _write_input(tmp_path, 'survey_a', shape=(1, 2, 4, 1))
	valid_indices = np.array([0, 1, 3, 4, 6, 7], dtype=np.int64)
	valid = np.zeros(8, dtype=np.bool_)
	valid[valid_indices] = True
	np.save(item.valid_tokens_path, valid.reshape((1, 2, 4)))
	prepare_calls = 0

	def prepare_batch(source: EmbeddingInput, indices: np.ndarray) -> np.ndarray:
		nonlocal prepare_calls
		prepare_calls += 1
		return extract_token_features(source, indices)

	store = prepare_feature_store(
		embedding_inputs=(item,),
		feature_dim=1,
		feature_mode='embedding',
		residualizer=None,
		preprocessor=_IdentityPreprocessor(),
		edge_margin_tokens=(0, 0, 0),
		settings=PreparedFeatureCacheSettings(
			chunk_size_tokens=2,
			directory=tmp_path / 'prepared',
		),
		default_cache_root=tmp_path / 'unused',
		prepare_batch=prepare_batch,
	)
	assert prepare_calls == 3
	centers = np.array([[1.0], [6.0]], dtype=np.float32)
	transitions = build_ordered_transition_costs(
		2,
		HMMTransitionSettings(
			same_cost=0.0,
			advance_cost=0.0,
			jump_cost=1.0,
			reverse_cost=10.0,
			forbid_reverse=True,
			max_jump=1,
		),
	)
	prepared_labels = decode_prepared_survey_ordered_labels(
		store.surveys[0],
		centers=centers,
		transition_costs=transitions,
		initial_state_costs=None,
		terminal_state_costs=None,
		expected_boundaries=None,
	)
	reference_labels = decode_survey_ordered_labels(
		item,
		centers=centers,
		residualizer=None,
		preprocessor=_IdentityPreprocessor(),
		transition_costs=transitions,
		emission_source='embedding',
	)
	np.testing.assert_array_equal(prepared_labels, reference_labels)
	prepared_centers, prepared_summary = update_centers_from_prepared_labels(
		store,
		{item.survey_id: prepared_labels},
		centers=centers,
		prediction_batch_size=2,
		empty_cluster_policy='keep_previous',
	)
	reference_centers, reference_summary = update_centers_from_labels(
		(item,),
		{item.survey_id: reference_labels},
		centers=centers,
		residualizer=None,
		preprocessor=_IdentityPreprocessor(),
		prediction_batch_size=2,
		empty_cluster_policy='keep_previous',
	)
	np.testing.assert_allclose(prepared_centers, reference_centers)
	assert prepared_summary == reference_summary
	prepared_objective = _squared_feature_objective(
		store.surveys[0].features_for_flat_indices(valid_indices),
		prepared_labels.reshape(-1)[valid_indices],
		prepared_centers,
	)
	reference_objective = _squared_feature_objective(
		extract_token_features(item, valid_indices),
		reference_labels.reshape(-1)[valid_indices],
		reference_centers,
	)
	assert prepared_objective == reference_objective
	assert prepare_calls == 3
	store.close()


def test_prepared_feature_store_opens_only_one_survey_at_a_time(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	items = (
		_write_input(tmp_path, 'survey_a', shape=(1, 1, 2, 1)),
		_write_input(tmp_path, 'survey_b', shape=(1, 1, 2, 1)),
	)
	store = prepare_feature_store(
		embedding_inputs=items,
		feature_dim=1,
		feature_mode='embedding',
		residualizer=None,
		preprocessor=_IdentityPreprocessor(),
		edge_margin_tokens=(0, 0, 0),
		settings=PreparedFeatureCacheSettings(directory=tmp_path / 'prepared'),
		default_cache_root=tmp_path / 'unused',
		prepare_batch=extract_token_features,
	)
	assert all(
		not any(isinstance(value, np.ndarray) for value in vars(survey).values())
		for survey in store.surveys
	)
	original_open = PreparedSurveyFeatures.open
	active = 0
	max_active = 0

	@contextmanager
	def tracked_open(survey: PreparedSurveyFeatures):
		nonlocal active, max_active
		active += 1
		max_active = max(max_active, active)
		try:
			with original_open(survey) as opened:
				yield opened
		finally:
			active -= 1

	monkeypatch.setattr(PreparedSurveyFeatures, 'open', tracked_open)
	update_centers_from_prepared_labels(
		store,
		{
			item.survey_id: np.zeros((1, 1, 2), dtype=np.int32)
			for item in items
		},
		centers=np.zeros((1, 1), dtype=np.float32),
		prediction_batch_size=1,
		empty_cluster_policy='keep_previous',
	)

	assert active == 0
	assert max_active == 1
	store.close()


def test_prepared_feature_store_build_reuse_force_and_index_mapping(
	tmp_path: Path,
) -> None:
	item = _write_input(
		tmp_path,
		'survey_a',
		shape=(2, 2, 3, 2),
	)
	valid_indices = np.array([0, 2, 3, 5, 7, 8, 11], dtype=np.int64)
	valid = np.zeros(12, dtype=np.bool_)
	valid[valid_indices] = True
	np.save(item.valid_tokens_path, valid.reshape((2, 2, 3)))
	cache_root = tmp_path / 'prepared'
	calls: list[np.ndarray] = []

	def prepare_batch(source: EmbeddingInput, indices: np.ndarray) -> np.ndarray:
		calls.append(indices.copy())
		return extract_token_features(source, indices) + 1.0

	settings = PreparedFeatureCacheSettings(
		chunk_size_tokens=3,
		directory=cache_root,
	)
	store = prepare_feature_store(
		embedding_inputs=(item,),
		feature_dim=2,
		feature_mode='embedding',
		residualizer=None,
		preprocessor=_IdentityPreprocessor(),
		edge_margin_tokens=(0, 0, 0),
		settings=settings,
		default_cache_root=tmp_path / 'unused',
		prepare_batch=prepare_batch,
	)
	prepared = store.surveys[0]
	assert [len(indices) for indices in calls] == [3, 3, 1]
	assert not prepared.reused
	assert prepared.cache_path is not None
	assert (prepared.cache_path / 'metadata.json').is_file()

	requested = np.array([11, 0, 7], dtype=np.int64)
	np.testing.assert_allclose(
		prepared.features_for_flat_indices(requested),
		extract_token_features(item, requested) + 1.0,
	)
	z_indices, trace = prepared.trace_features(0, 1)
	np.testing.assert_array_equal(z_indices, np.array([0, 2]))
	np.testing.assert_allclose(
		trace,
		extract_token_features(item, np.array([3, 5])) + 1.0,
	)
	fingerprint = prepared.fingerprint
	store.close()

	def unexpected_prepare(_source: EmbeddingInput, _indices: np.ndarray) -> np.ndarray:
		raise AssertionError('complete cache should have been reused')

	reused = prepare_feature_store(
		embedding_inputs=(item,),
		feature_dim=2,
		feature_mode='embedding',
		residualizer=None,
		preprocessor=_IdentityPreprocessor(),
		edge_margin_tokens=(0, 0, 0),
		settings=settings,
		default_cache_root=tmp_path / 'unused',
		prepare_batch=unexpected_prepare,
	)
	assert reused.surveys[0].reused
	assert reused.surveys[0].fingerprint == fingerprint
	reused.close()

	calls.clear()
	forced = prepare_feature_store(
		embedding_inputs=(item,),
		feature_dim=2,
		feature_mode='embedding',
		residualizer=None,
		preprocessor=_IdentityPreprocessor(),
		edge_margin_tokens=(0, 0, 0),
		settings=PreparedFeatureCacheSettings(
			chunk_size_tokens=4,
			force_rebuild=True,
			directory=cache_root,
		),
		default_cache_root=tmp_path / 'unused',
		prepare_batch=prepare_batch,
	)
	assert [len(indices) for indices in calls] == [4, 3]
	assert not forced.surveys[0].reused
	forced.close()


def test_prepared_feature_store_partial_cleanup_fingerprint_and_cleanup_policy(
	tmp_path: Path,
) -> None:
	item = _write_input(tmp_path, 'survey_a', shape=(1, 1, 3, 1))
	np.save(
		item.valid_tokens_path,
		np.array([True, False, True], dtype=np.bool_).reshape((1, 1, 3)),
	)
	cache_root = tmp_path / 'prepared'
	settings = PreparedFeatureCacheSettings(directory=cache_root)
	kwargs = {
		'embedding_inputs': (item,),
		'feature_dim': 1,
		'feature_mode': 'embedding',
		'residualizer': None,
		'preprocessor': _IdentityPreprocessor(),
		'settings': settings,
		'default_cache_root': tmp_path / 'unused',
		'prepare_batch': extract_token_features,
	}
	first = prepare_feature_store(edge_margin_tokens=(0, 0, 0), **kwargs)
	first_path = first.surveys[0].cache_path
	assert first_path is not None
	first.close()
	interrupted = cache_root / f'.{first_path.name}.building-interrupted'
	interrupted.mkdir()

	second = prepare_feature_store(edge_margin_tokens=(0, 0, 0), **kwargs)
	assert not interrupted.exists()
	second.close()
	margin_changed = prepare_feature_store(edge_margin_tokens=(0, 0, 1), **kwargs)
	assert margin_changed.surveys[0].fingerprint != first_path.name
	margin_changed.close()

	embedding_replacement = tmp_path / 'embedding-replacement.npy'
	np.save(embedding_replacement, np.full((1, 1, 3, 1), 4.0, dtype=np.float32))
	embedding_replacement.replace(item.embeddings_path)
	embedding_changed = prepare_feature_store(edge_margin_tokens=(0, 0, 0), **kwargs)
	embedding_fingerprint = embedding_changed.surveys[0].fingerprint
	assert embedding_fingerprint != first_path.name
	assert not embedding_changed.surveys[0].reused
	embedding_changed.close()

	mask_replacement = tmp_path / 'mask-replacement.npy'
	np.save(mask_replacement, np.ones((1, 1, 3), dtype=np.bool_))
	mask_replacement.replace(item.valid_tokens_path)
	mask_changed = prepare_feature_store(edge_margin_tokens=(0, 0, 0), **kwargs)
	mask_fingerprint = mask_changed.surveys[0].fingerprint
	assert mask_fingerprint != embedding_fingerprint
	assert not mask_changed.surveys[0].reused
	mask_changed.close()

	residualizer_changed = prepare_feature_store(
		edge_margin_tokens=(0, 0, 0),
		**{**kwargs, 'residualizer': {'version': 1}},
	)
	assert residualizer_changed.surveys[0].fingerprint != mask_fingerprint
	assert not residualizer_changed.surveys[0].reused
	residualizer_changed.close()

	preprocessor_changed = prepare_feature_store(
		edge_margin_tokens=(0, 0, 0),
		**{**kwargs, 'preprocessor': _IdentityPreprocessor(version=1)},
	)
	assert preprocessor_changed.surveys[0].fingerprint != mask_fingerprint
	assert not preprocessor_changed.surveys[0].reused
	preprocessor_changed.close()

	cleanup = prepare_feature_store(
		edge_margin_tokens=(0, 0, 0),
		**{
			**kwargs,
			'settings': PreparedFeatureCacheSettings(
				cleanup=True,
				persist=False,
				force_rebuild=True,
				directory=cache_root,
			),
		},
	)
	cleanup_path = cleanup.surveys[0].cache_path
	assert cleanup_path is not None
	assert cleanup_path.exists()
	cleanup.close()
	assert not cleanup_path.exists()


def test_prepared_feature_store_cleans_completed_surveys_after_later_failure(
	tmp_path: Path,
) -> None:
	first = _write_input(tmp_path, 'survey_a', shape=(1, 1, 2, 1))
	second = _write_input(tmp_path, 'survey_b', shape=(1, 1, 2, 1))
	cache_root = tmp_path / 'prepared'

	def prepare_batch(item: EmbeddingInput, indices: np.ndarray) -> np.ndarray:
		if item.survey_id == 'survey_b':
			raise RuntimeError('injected second-survey failure')
		return extract_token_features(item, indices)

	with pytest.raises(RuntimeError, match='second-survey failure'):
		prepare_feature_store(
			embedding_inputs=(first, second),
			feature_dim=1,
			feature_mode='embedding',
			residualizer=None,
			preprocessor=_IdentityPreprocessor(),
			edge_margin_tokens=(0, 0, 0),
			settings=PreparedFeatureCacheSettings(
				cleanup=True,
				persist=False,
				directory=cache_root,
			),
			default_cache_root=tmp_path / 'unused',
			prepare_batch=prepare_batch,
		)

	assert cache_root.is_dir()
	assert not any(cache_root.iterdir())


def test_prepared_feature_store_reports_configured_cleanup_failure(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	item = _write_input(tmp_path, 'survey_a', shape=(1, 1, 2, 1))
	store = prepare_feature_store(
		embedding_inputs=(item,),
		feature_dim=1,
		feature_mode='embedding',
		residualizer=None,
		preprocessor=_IdentityPreprocessor(),
		edge_margin_tokens=(0, 0, 0),
		settings=PreparedFeatureCacheSettings(
			cleanup=True,
			persist=False,
			directory=tmp_path / 'prepared',
		),
		default_cache_root=tmp_path / 'unused',
		prepare_batch=extract_token_features,
	)
	cache_path = store.surveys[0].cache_path
	assert cache_path is not None
	original_rmtree = prepared_features_module.shutil.rmtree

	def fail_configured_cleanup(path: Path) -> None:
		if path == cache_path:
			raise OSError('injected prepared-cache cleanup failure')
		original_rmtree(path)

	monkeypatch.setattr(
		prepared_features_module.shutil,
		'rmtree',
		fail_configured_cleanup,
	)
	with pytest.raises(OSError, match='prepared-cache cleanup failure'):
		store.close()
	assert cache_path.exists()
	original_rmtree(cache_path)


def test_prepared_feature_store_zero_valid_and_z_coordinate_fast_path(
	tmp_path: Path,
) -> None:
	item = _write_input(tmp_path, 'survey_a', shape=(1, 1, 4, 2))
	cache_root = tmp_path / 'prepared'
	np.save(item.valid_tokens_path, np.zeros((1, 1, 4), dtype=np.bool_))
	zero = prepare_feature_store(
		embedding_inputs=(item,),
		feature_dim=2,
		feature_mode='embedding',
		residualizer=None,
		preprocessor=_IdentityPreprocessor(),
		edge_margin_tokens=(0, 0, 0),
		settings=PreparedFeatureCacheSettings(directory=cache_root),
		default_cache_root=tmp_path / 'unused',
		prepare_batch=lambda *_: (_ for _ in ()).throw(AssertionError()),
	)
	assert zero.surveys[0].valid_token_count == 0
	with zero.surveys[0].open() as opened:
		assert opened.features is not None
		assert opened.features.shape == (0, 2)
	zero.close()

	valid = np.array([True, False, True, True], dtype=np.bool_)
	np.save(item.valid_tokens_path, valid.reshape((1, 1, 4)))
	direct_root = tmp_path / 'direct-must-not-exist'
	direct = prepare_feature_store(
		embedding_inputs=(item,),
		feature_dim=1,
		feature_mode='z_coordinate',
		residualizer=None,
		preprocessor=_IdentityPreprocessor(),
		edge_margin_tokens=(0, 0, 0),
		settings=PreparedFeatureCacheSettings(directory=direct_root),
		default_cache_root=tmp_path / 'unused',
		prepare_batch=lambda *_: (_ for _ in ()).throw(AssertionError()),
	)
	assert not direct_root.exists()
	assert direct.to_metadata()['effective_mode'] == 'direct'
	np.testing.assert_allclose(
		direct.surveys[0].features_for_flat_indices(np.array([0, 3])),
		np.array([[0.0], [1.0]], dtype=np.float32),
	)
	with pytest.raises(ValueError, match='prepared valid set'):
		direct.surveys[0].features_for_flat_indices(np.array([1]))
	direct.close()


def _write_input(
	root: Path,
	survey_id: str,
	*,
	shape: tuple[int, int, int, int],
) -> EmbeddingInput:
	embeddings_path = root / f'{survey_id}.embeddings.npy'
	valid_path = root / f'{survey_id}.valid_tokens.npy'
	metadata_path = root / f'{survey_id}.embedding_metadata.json'
	np.save(embeddings_path, np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
	np.save(valid_path, np.ones(shape[:3], dtype=np.bool_))
	metadata_path.write_text('{}\n', encoding='utf-8')
	return EmbeddingInput(survey_id, embeddings_path, valid_path, metadata_path)


def _squared_feature_objective(
	features: np.ndarray,
	labels: np.ndarray,
	centers: np.ndarray,
) -> float:
	deltas = features - centers[np.asarray(labels, dtype=np.int64)]
	return float(np.sum(deltas * deltas, dtype=np.float64))
