from __future__ import annotations

import json
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pytest

from seis_ssl_cluster.clustering import run_embedding_clustering, stratigraphic_hmm
from seis_ssl_cluster.clustering.features import (
	EmbeddingInput,
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
	LocalTokenPositionResidualizer,
	residualization_keys_for_flat_indices,
)
from seis_ssl_cluster.clustering.stratigraphic_hmm import (
	decode_survey_ordered_labels,
	initialize_ordered_centers,
	normalized_z_features_for_indices,
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


def test_normalized_z_features_for_indices_uses_token_grid_shape(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.ones((2, 1, 4, 3), dtype=np.float32),
		valid=np.ones((2, 1, 4), dtype=np.bool_),
	)
	embedding_input = discover_embedding_inputs(input_dir)[0]

	features = normalized_z_features_for_indices(
		embedding_input,
		np.array([0, 1, 3, 4, 7], dtype=np.int64),
	)

	assert features.dtype == np.float32
	assert features.shape == (5, 1)
	np.testing.assert_allclose(
		features[:, 0],
		np.array([0.0, 1.0 / 3.0, 1.0, 0.0, 1.0], dtype=np.float32),
	)


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


def test_prepare_feature_batch_supports_legacy_residualizer_artifact(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	input_dir.mkdir()
	embeddings = np.array([[[[10.0, 20.0], [11.0, 21.0]]]], dtype=np.float32)
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=embeddings,
		valid=np.ones((1, 1, 2), dtype=np.bool_),
	)
	embedding_input = discover_embedding_inputs(input_dir)[0]
	legacy = LocalTokenPositionResidualizer(
		mode='local_token_position',
		group_by='token_phase',
		add_global_mean_back=True,
		min_group_count=1,
		means=np.array([[1.0, 2.0]], dtype=np.float32),
		counts=np.array([2], dtype=np.int64),
		group_shape=None,
		fallback_mean=np.array([5.0, 6.0], dtype=np.float32),
		legacy_group_keys=np.array([[0, 0, 0]], dtype=np.int64),
	)
	preprocessor = fit_preprocessor(
		embeddings.reshape((-1, 2)),
		normalization='none',
		pca=PCASettings(enabled=False, n_components=2, whiten=False),
		seed=7,
	)

	batch = prepare_feature_batch_for_indices(
		embedding_input,
		np.array([0, 1], dtype=np.int64),
		residualizer=legacy,
		preprocessor=preprocessor,
	)

	np.testing.assert_allclose(batch, embeddings.reshape((-1, 2)) + 4.0)


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


def test_run_embedding_clustering_stratigraphic_hmm_writes_artifacts(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	output_dir = tmp_path / 'clusters'
	input_dir.mkdir()
	base_trace = np.array(
		[
			[0.0, 0.0],
			[0.2, 0.0],
			[4.0, 0.0],
			[4.2, 0.0],
			[8.0, 0.0],
		],
		dtype=np.float32,
	)
	embeddings_a = np.empty((2, 2, 5, 2), dtype=np.float32)
	embeddings_b = np.empty((1, 2, 5, 2), dtype=np.float32)
	for x_index in range(embeddings_a.shape[0]):
		for y_index in range(embeddings_a.shape[1]):
			embeddings_a[x_index, y_index] = base_trace + (0.05 * x_index)
	for y_index in range(embeddings_b.shape[1]):
		embeddings_b[0, y_index] = base_trace + (0.03 * y_index)
	valid_a = np.ones((2, 2, 5), dtype=np.bool_)
	valid_a[0, 1, 2] = False
	valid_b = np.ones((1, 2, 5), dtype=np.bool_)
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=embeddings_a,
		valid=valid_a,
	)
	_write_embedding_artifacts(
		input_dir,
		'survey_b',
		embeddings=embeddings_b,
		valid=valid_b,
	)

	config = _hmm_config(input_dir, output_dir)
	config['clustering']['stage_timing'] = True
	result = run_embedding_clustering(config)

	assert [item.k for item in result.results] == [3]
	k_dir = output_dir / 'models' / 'k3'
	assert (k_dir / 'preprocessor.joblib').is_file()
	assert (k_dir / 'hmm_model.joblib').is_file()
	assert (k_dir / 'cluster_centers.npy').is_file()
	assert (k_dir / 'clustering_metadata.json').is_file()
	assert not (k_dir / 'kmeans.joblib').exists()
	timings = json.loads((output_dir / 'stage_timings.json').read_text())
	assert set(timings['stages']) == {
		'center_accumulation',
		'center_finalize',
		'emission',
		'viterbi',
	}
	metadata = json.loads((k_dir / 'clustering_metadata.json').read_text())
	assert metadata['method'] == 'stratigraphic_hmm_kmeans'
	assert metadata['emission_source'] == 'embedding'
	assert metadata['stratigraphic_hmm']['emission_source'] == 'embedding'
	prepared_cache = metadata['stratigraphic_hmm']['prepared_feature_cache']
	assert prepared_cache['effective_mode'] == 'memmap'
	assert len(prepared_cache['surveys'][0]['fingerprint']) == 64
	assert metadata['stratigraphic_hmm']['iteration_summaries']
	assert metadata['stratigraphic_hmm']['init'] == {'order_by': 'mean_z'}
	assert (
		metadata['ordered_diagnostics']['aggregate']['reverse_transition_rate'] == 0.0
	)
	assert (
		metadata['ordered_diagnostics']['per_survey']['survey_a'][
			'reverse_transition_rate'
		]
		== 0.0
	)

	labels_a = np.load(
		output_dir / 'labels' / 'k3' / 'survey_a.cluster_labels_token.npy',
	)
	assert labels_a.dtype == np.int32
	assert labels_a.shape == (2, 2, 5)
	assert labels_a[0, 1, 2] == -1
	for x_index in range(labels_a.shape[0]):
		for y_index in range(labels_a.shape[1]):
			trace = labels_a[x_index, y_index]
			valid_trace = trace[trace >= 0]
			assert np.all(np.diff(valid_trace) >= 0)

	label_metadata = json.loads(
		(
			output_dir / 'labels' / 'k3' / 'survey_a.cluster_label_metadata.json'
		).read_text(),
	)
	assert label_metadata['method'] == 'stratigraphic_hmm_kmeans'
	assert label_metadata['emission_source'] == 'embedding'
	assert label_metadata['invalid_token_count'] == 1
	assert label_metadata['ordered_diagnostics']['reverse_transition_rate'] == 0.0
	assert '0_to_1' in label_metadata['ordered_boundary_summary']
	assert sum(label_metadata['cluster_counts'].values()) == int(
		np.count_nonzero(valid_a),
	)


def test_stratigraphic_hmm_closes_prepared_features_on_failure(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	input_dir = tmp_path / 'embeddings'
	output_dir = tmp_path / 'clusters'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.arange(10, dtype=np.float32).reshape(1, 1, 5, 2),
		valid=np.ones((1, 1, 5), dtype=np.bool_),
	)
	config = _hmm_config(input_dir, output_dir)
	config['clustering']['stratigraphic_hmm']['prepared_feature_cache'] = {
		'cleanup': True,
		'persist': False,
	}
	closed = False
	original_close = stratigraphic_hmm.PreparedFeatureStore.close

	def record_close(store: stratigraphic_hmm.PreparedFeatureStore) -> None:
		nonlocal closed
		closed = True
		original_close(store)

	def fail_decode(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
		del args, kwargs
		raise RuntimeError('decode failed')

	monkeypatch.setattr(
		stratigraphic_hmm.PreparedFeatureStore,
		'close',
		record_close,
	)
	monkeypatch.setattr(stratigraphic_hmm, '_decode_all_surveys', fail_decode)

	with pytest.raises(RuntimeError, match='decode failed'):
		run_embedding_clustering(config)

	assert closed
	assert list((output_dir / 'prepared_features').iterdir()) == []


def test_stratigraphic_hmm_multi_k_prepares_each_feature_token_once(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""The shared prepared-feature pass must not repeat for each K value."""
	input_dir = tmp_path / 'embeddings'
	output_dir = tmp_path / 'clusters'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.stack(
			(
				np.arange(12, dtype=np.float32),
				np.arange(12, dtype=np.float32) ** 2,
			),
			axis=-1,
		).reshape(1, 1, 12, 2),
		valid=np.ones((1, 1, 12), dtype=np.bool_),
	)
	config = _hmm_config(input_dir, output_dir)
	config['clustering']['k_values'] = [6, 8, 10]
	config['clustering']['stratigraphic_hmm']['iterations'] = 1
	config['clustering']['stratigraphic_hmm']['prepared_feature_cache'] = {
		'chunk_size_tokens': 4,
		'reuse': False,
		'force_rebuild': False,
		'cleanup': True,
		'persist': False,
	}
	prepared_indices: list[np.ndarray] = []
	original_prepare = stratigraphic_hmm.prepare_feature_batch_for_indices

	def record_prepare(
		embedding_input: EmbeddingInput,
		flat_indices: np.ndarray,
		**kwargs: object,
	) -> np.ndarray:
		prepared_indices.append(flat_indices.copy())
		return original_prepare(
			embedding_input,
			flat_indices,
			**kwargs,
		)

	monkeypatch.setattr(
		stratigraphic_hmm,
		'prepare_feature_batch_for_indices',
		record_prepare,
	)

	result = run_embedding_clustering(config)

	assert [item.k for item in result.results] == [6, 8, 10]
	assert len(prepared_indices) == 3
	np.testing.assert_array_equal(
		np.concatenate(prepared_indices),
		np.arange(12, dtype=np.int64),
	)


def test_stratigraphic_hmm_saved_labels_decode_from_saved_centers(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	output_dir = tmp_path / 'clusters'
	input_dir.mkdir()
	embeddings = np.array(
		[
			[
				[
					[-0.9456348, 0.0],
					[-2.0896657, 0.0],
					[-5.166128, 0.0],
					[-1.9232596, 0.0],
				],
			],
		],
		dtype=np.float32,
	)
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=embeddings,
		valid=np.ones((1, 1, 4), dtype=np.bool_),
	)
	config = _hmm_config(input_dir, output_dir)
	config['clustering']['sample_tokens'] = 4
	config['clustering']['minibatch_size'] = 4
	hmm_config = config['clustering']['stratigraphic_hmm']
	hmm_config['iterations'] = 1
	hmm_config['transition']['advance_cost'] = 2.0
	hmm_config['transition']['jump_cost'] = 5.0

	run_embedding_clustering(config)

	k_dir = output_dir / 'models' / 'k3'
	labels = np.load(
		output_dir / 'labels' / 'k3' / 'survey_a.cluster_labels_token.npy',
	)
	centers = np.load(k_dir / 'cluster_centers.npy')
	preprocessor = joblib.load(k_dir / 'preprocessor.joblib')
	hmm_model = joblib.load(k_dir / 'hmm_model.joblib')
	embedding_input = discover_embedding_inputs(input_dir)[0]
	decoded = decode_survey_ordered_labels(
		embedding_input,
		centers=centers,
		residualizer=None,
		preprocessor=preprocessor,
		transition_costs=hmm_model['transition_costs'],
		emission_source=hmm_model['emission_source'],
	)

	assert len(hmm_model['iteration_summaries']) == 1
	assert hmm_model['iteration_summaries'][0]['total_center_shift_l2'] > 0.0
	np.testing.assert_allclose(centers, hmm_model['centers'], rtol=0.0, atol=0.0)
	np.testing.assert_array_equal(labels, decoded)


def test_stratigraphic_hmm_metadata_is_strict_json_safe(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	output_dir = tmp_path / 'clusters'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.zeros((2, 2, 6, 2), dtype=np.float32),
		valid=np.ones((2, 2, 6), dtype=np.bool_),
	)

	run_embedding_clustering(
		_hmm_config(input_dir, output_dir, emission_source='z_coordinate'),
	)

	k_dir = output_dir / 'models' / 'k3'
	metadata_paths = [
		k_dir / 'clustering_metadata.json',
		output_dir / 'labels' / 'k3' / 'survey_a.cluster_label_metadata.json',
	]
	for metadata_path in metadata_paths:
		text = metadata_path.read_text(encoding='utf-8')
		assert 'Infinity' not in text
		assert '-Infinity' not in text
		assert 'NaN' not in text
		json.loads(text)

	metadata = json.loads((k_dir / 'clustering_metadata.json').read_text())
	assert metadata['stratigraphic_hmm']['transition_costs'][1][0] is None
	model = joblib.load(k_dir / 'hmm_model.joblib')
	assert np.isinf(model['transition_costs'][1, 0])


def test_run_embedding_clustering_stratigraphic_hmm_path_prior_metadata(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	output_dir = tmp_path / 'clusters'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.zeros((2, 2, 6, 2), dtype=np.float32),
		valid=np.ones((2, 2, 6), dtype=np.bool_),
	)
	config = _hmm_config(input_dir, output_dir, emission_source='z_coordinate')
	config['clustering']['stratigraphic_hmm']['path_prior'] = {
		'enabled': True,
		'initial_state': {'mode': 'shallow_anchor', 'weight': 0.5},
		'terminal_state': {'mode': 'deep_anchor', 'weight': 0.5},
		'expected_boundaries': {
			'enabled': False,
			'target': 'auto_k_minus_1',
			'weight': 0.1,
		},
	}

	run_embedding_clustering(config)

	metadata = json.loads(
		(output_dir / 'models' / 'k3' / 'clustering_metadata.json').read_text(),
	)
	path_prior = metadata['stratigraphic_hmm']['path_prior']
	assert path_prior['enabled'] is True
	assert path_prior['initial_state'] == {'mode': 'shallow_anchor', 'weight': 0.5}
	assert path_prior['terminal_state'] == {'mode': 'deep_anchor', 'weight': 0.5}
	assert path_prior['expected_boundaries'] == {
		'enabled': False,
		'target': 'auto_k_minus_1',
		'weight': 0.1,
		'target_resolution': 'per_trace_min_target_valid_length_minus_one',
	}
	np.testing.assert_allclose(path_prior['initial_state_costs'], [0.0, 0.25, 0.5])
	np.testing.assert_allclose(path_prior['terminal_state_costs'], [0.5, 0.25, 0.0])
	assert np.all(np.isfinite(path_prior['initial_state_costs']))
	assert np.all(np.isfinite(path_prior['terminal_state_costs']))
	assert (
		metadata['ordered_diagnostics']['aggregate']['reverse_transition_rate'] == 0.0
	)


def test_run_embedding_clustering_expected_boundary_prior_increases_boundaries(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	no_prior_output = tmp_path / 'clusters_no_prior'
	prior_output = tmp_path / 'clusters_prior'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.zeros((2, 2, 6, 2), dtype=np.float32),
		valid=np.ones((2, 2, 6), dtype=np.bool_),
	)
	no_prior_config = _hmm_config(
		input_dir,
		no_prior_output,
		emission_source='z_coordinate',
	)
	prior_config = _hmm_config(
		input_dir,
		prior_output,
		emission_source='z_coordinate',
	)
	for config in (no_prior_config, prior_config):
		config['clustering']['sample_tokens'] = 24
		config['clustering']['minibatch_size'] = 24
		hmm_config = config['clustering']['stratigraphic_hmm']
		hmm_config['iterations'] = 1
		hmm_config['transition']['advance_cost'] = 5.0
		hmm_config['transition']['jump_cost'] = 10.0
	prior_config['clustering']['stratigraphic_hmm']['path_prior'] = {
		'enabled': True,
		'initial_state': {'mode': 'none', 'weight': 0.0},
		'terminal_state': {'mode': 'none', 'weight': 0.0},
		'expected_boundaries': {
			'enabled': True,
			'target': 'auto_k_minus_1',
			'weight': 50.0,
		},
	}

	run_embedding_clustering(no_prior_config)
	run_embedding_clustering(prior_config)

	no_prior_metadata = json.loads(
		(no_prior_output / 'models' / 'k3' / 'clustering_metadata.json').read_text(),
	)
	prior_metadata = json.loads(
		(prior_output / 'models' / 'k3' / 'clustering_metadata.json').read_text(),
	)
	no_prior_mean = no_prior_metadata['ordered_diagnostics']['aggregate'][
		'mean_boundaries_per_valid_trace'
	]
	prior_mean = prior_metadata['ordered_diagnostics']['aggregate'][
		'mean_boundaries_per_valid_trace'
	]

	assert prior_mean > no_prior_mean
	assert (
		prior_metadata['ordered_diagnostics']['aggregate']['reverse_transition_rate']
		== 0.0
	)
	assert prior_metadata['stratigraphic_hmm']['path_prior']['expected_boundaries'] == {
		'enabled': True,
		'target': 'auto_k_minus_1',
		'weight': 50.0,
		'target_resolution': 'per_trace_min_target_valid_length_minus_one',
	}


def test_run_embedding_clustering_disabled_path_prior_gates_expected_boundaries(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	no_prior_output = tmp_path / 'clusters_no_prior'
	disabled_prior_output = tmp_path / 'clusters_disabled_prior'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.zeros((2, 2, 6, 2), dtype=np.float32),
		valid=np.ones((2, 2, 6), dtype=np.bool_),
	)
	no_prior_config = _hmm_config(
		input_dir,
		no_prior_output,
		emission_source='z_coordinate',
	)
	disabled_prior_config = _hmm_config(
		input_dir,
		disabled_prior_output,
		emission_source='z_coordinate',
	)
	for config in (no_prior_config, disabled_prior_config):
		config['clustering']['sample_tokens'] = 24
		config['clustering']['minibatch_size'] = 24
		hmm_config = config['clustering']['stratigraphic_hmm']
		hmm_config['iterations'] = 1
		hmm_config['transition']['advance_cost'] = 5.0
		hmm_config['transition']['jump_cost'] = 10.0
	disabled_prior_config['clustering']['stratigraphic_hmm']['path_prior'] = {
		'enabled': False,
		'initial_state': {'mode': 'none', 'weight': 0.0},
		'terminal_state': {'mode': 'none', 'weight': 0.0},
		'expected_boundaries': {
			'enabled': True,
			'target': 'auto_k_minus_1',
			'weight': 50.0,
		},
	}

	run_embedding_clustering(no_prior_config)
	run_embedding_clustering(disabled_prior_config)

	no_prior_labels = np.load(
		no_prior_output / 'labels' / 'k3' / 'survey_a.cluster_labels_token.npy',
	)
	disabled_prior_labels = np.load(
		disabled_prior_output
		/ 'labels'
		/ 'k3'
		/ 'survey_a.cluster_labels_token.npy',
	)
	metadata = json.loads(
		(
			disabled_prior_output / 'models' / 'k3' / 'clustering_metadata.json'
		).read_text(),
	)

	np.testing.assert_array_equal(disabled_prior_labels, no_prior_labels)
	assert metadata['stratigraphic_hmm']['path_prior']['enabled'] is False
	assert (
		metadata['stratigraphic_hmm']['path_prior']['expected_boundaries'][
			'enabled'
		]
		is True
	)


def test_run_embedding_clustering_stratigraphic_hmm_zonly_writes_metadata(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	output_dir = tmp_path / 'clusters'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.zeros((2, 2, 6, 2), dtype=np.float32),
		valid=np.ones((2, 2, 6), dtype=np.bool_),
	)

	result = run_embedding_clustering(
		_hmm_config(input_dir, output_dir, emission_source='z_coordinate'),
	)

	assert [item.k for item in result.results] == [3]
	k_dir = output_dir / 'models' / 'k3'
	assert (k_dir / 'preprocessor.joblib').is_file()
	assert (k_dir / 'hmm_model.joblib').is_file()
	metadata = json.loads((k_dir / 'clustering_metadata.json').read_text())
	assert metadata['emission_source'] == 'z_coordinate'
	assert (
		metadata['emission_features']['embedding_features_used_for_emissions'] is False
	)
	assert metadata['emission_features']['embedding_artifacts_used_for'] == [
		'token_grid_shape',
		'validity_masks',
	]
	assert metadata['stratigraphic_hmm']['emission_source'] == 'z_coordinate'
	assert (
		metadata['stratigraphic_hmm']['prepared_feature_cache']['effective_mode']
		== 'direct'
	)
	assert metadata['normalization'] == 'none'
	assert metadata['pca']['enabled'] is False

	label_metadata = json.loads(
		(
			output_dir / 'labels' / 'k3' / 'survey_a.cluster_label_metadata.json'
		).read_text(),
	)
	assert label_metadata['emission_source'] == 'z_coordinate'


def test_stratigraphic_hmm_zonly_labels_are_monotone_when_reverse_forbidden(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	output_dir = tmp_path / 'clusters'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.zeros((2, 2, 6, 2), dtype=np.float32),
		valid=np.ones((2, 2, 6), dtype=np.bool_),
	)

	run_embedding_clustering(
		_hmm_config(input_dir, output_dir, emission_source='z_coordinate'),
	)

	labels = np.load(
		output_dir / 'labels' / 'k3' / 'survey_a.cluster_labels_token.npy',
	)
	for x_index in range(labels.shape[0]):
		for y_index in range(labels.shape[1]):
			trace = labels[x_index, y_index]
			valid_trace = trace[trace >= 0]
			assert np.all(np.diff(valid_trace) >= 0)


def test_stratigraphic_hmm_edge_margin_excludes_embedding_tokens(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	output_dir = tmp_path / 'clusters'
	input_dir.mkdir()
	embeddings = _edge_margin_embedding_grid((6, 5, 4), dim=2)
	valid = np.ones((6, 5, 4), dtype=np.bool_)
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=embeddings,
		valid=valid,
	)
	config = _hmm_config(
		input_dir,
		output_dir,
		edge_margin_tokens=[1, 1, 0],
	)
	config['clustering']['sample_tokens'] = 1000

	result = run_embedding_clustering(config)

	labels = np.load(
		output_dir / 'labels' / 'k3' / 'survey_a.cluster_labels_token.npy',
	)
	interior = np.zeros(valid.shape, dtype=np.bool_)
	interior[1:-1, 1:-1, :] = True
	assert np.all(labels[~interior] == -1)
	assert np.all(labels[interior & valid] >= 0)
	interior_valid_count = int(np.count_nonzero(interior & valid))
	assert result.sample.total_valid_count == interior_valid_count
	assert result.sample.sample_count == interior_valid_count

	model_metadata = json.loads(
		(output_dir / 'models' / 'k3' / 'clustering_metadata.json').read_text(),
	)
	assert model_metadata['sample']['total_valid_count'] == interior_valid_count
	assert model_metadata['sample']['count'] == interior_valid_count
	assert model_metadata['stratigraphic_hmm']['edge_margin_tokens'] == [1, 1, 0]
	assert (
		model_metadata['stratigraphic_hmm'][
			'edge_margin_excluded_valid_token_count'
		]
		== int(np.count_nonzero(valid & ~interior))
	)
	assert sum(model_metadata['cluster_counts'].values()) == interior_valid_count
	label_metadata = json.loads(
		(
			output_dir / 'labels' / 'k3' / 'survey_a.cluster_label_metadata.json'
		).read_text(),
	)
	assert label_metadata['valid_token_count'] == interior_valid_count
	assert sum(label_metadata['cluster_counts'].values()) == interior_valid_count


def test_stratigraphic_hmm_edge_margin_excludes_z_coordinate_tokens(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	output_dir = tmp_path / 'clusters'
	input_dir.mkdir()
	valid = np.ones((6, 5, 4), dtype=np.bool_)
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=_edge_margin_embedding_grid(valid.shape, dim=2),
		valid=valid,
	)
	config = _hmm_config(
		input_dir,
		output_dir,
		emission_source='z_coordinate',
		edge_margin_tokens=[1, 1, 0],
	)
	config['clustering']['sample_tokens'] = 1000

	run_embedding_clustering(config)

	labels = np.load(
		output_dir / 'labels' / 'k3' / 'survey_a.cluster_labels_token.npy',
	)
	interior = np.zeros(valid.shape, dtype=np.bool_)
	interior[1:-1, 1:-1, :] = True
	assert np.all(labels[~interior] == -1)
	assert np.all(labels[interior] >= 0)
	for x_index in range(1, labels.shape[0] - 1):
		for y_index in range(1, labels.shape[1] - 1):
			valid_trace = labels[x_index, y_index, :]
			assert np.all(np.diff(valid_trace) >= 0)


def test_stratigraphic_hmm_edge_margin_rejects_empty_interior(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	output_dir = tmp_path / 'clusters'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=_edge_margin_embedding_grid((6, 5, 4), dim=2),
		valid=np.ones((6, 5, 4), dtype=np.bool_),
	)

	with pytest.raises(
		ValueError,
		match=(
			r'edge_margin_tokens \[3,0,0\] leave no interior tokens '
			r'for survey survey_a with token grid shape \(6, 5, 4\)'
		),
	):
		run_embedding_clustering(
			_hmm_config(
				input_dir,
				output_dir,
				edge_margin_tokens=[3, 0, 0],
			),
		)


def _hmm_config(
	input_dir: Path,
	output_dir: Path,
	*,
	emission_source: str = 'embedding',
	edge_margin_tokens: list[int] | None = None,
) -> dict[str, object]:
	config: dict[str, object] = {
		'embeddings': {'input_dir': str(input_dir)},
		'clustering': {
			'output_dir': str(output_dir),
			'embedding_normalization': 'none',
			'residualization': {'enabled': False},
			'pca': {
				'enabled': False,
				'n_components': 2,
				'whiten': False,
			},
			'sample_tokens': 100,
			'method': 'stratigraphic_hmm_kmeans',
			'k_values': [3],
			'minibatch_size': 8,
			'prediction_batch_size': 3,
			'seed': 7,
			'stratigraphic_hmm': {
				'emission_source': emission_source,
				'iterations': 2,
				'z_axis': 2,
				'z_direction': 'increasing_downward',
				'transition': {
					'same_cost': 0.0,
					'advance_cost': 0.01,
					'jump_cost': 0.02,
					'reverse_cost': 1000000.0,
					'forbid_reverse': True,
					'max_jump': None,
				},
				'init': {'order_by': 'mean_z'},
				'update': {'empty_cluster_policy': 'keep_previous'},
			},
		},
	}
	if edge_margin_tokens is not None:
		config['clustering']['stratigraphic_hmm']['edge_margin_tokens'] = (
			edge_margin_tokens
		)
	return config


def _edge_margin_embedding_grid(
	shape: tuple[int, int, int],
	*,
	dim: int,
) -> np.ndarray:
	embeddings = np.empty((*shape, dim), dtype=np.float32)
	for x_index in range(shape[0]):
		for y_index in range(shape[1]):
			for z_index in range(shape[2]):
				embeddings[x_index, y_index, z_index, 0] = float(z_index * 4)
				embeddings[x_index, y_index, z_index, 1:] = float(
					x_index + y_index
				)
	return embeddings


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
