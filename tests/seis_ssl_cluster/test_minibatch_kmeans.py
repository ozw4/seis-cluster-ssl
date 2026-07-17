from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

import seis_ssl_cluster.clustering.kmeans as kmeans_module
from seis_ssl_cluster.clustering import run_embedding_clustering
from seis_ssl_cluster.clustering.features import (
	EmbeddingInput,
	discover_embedding_inputs,
	embedding_input_metadata,
)
from seis_ssl_cluster.clustering.kmeans import apply_residualizer_to_sample
from seis_ssl_cluster.clustering.residualization import LocalTokenPositionResidualizer
from seis_ssl_cluster.clustering.writer import (
	write_labels_for_k,
	write_labels_for_models,
)
from seis_ssl_cluster.utils import StageTimer

if TYPE_CHECKING:
	from pathlib import Path


def test_disabled_residualization_skips_group_ids_and_copy(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	features = np.ones((2, 3), dtype=np.float64)

	def fail_group_ids() -> None:
		raise AssertionError('group IDs must not be generated')

	monkeypatch.setattr(
		kmeans_module,
		'sample_residualization_group_ids',
		fail_group_ids,
	)
	actual = apply_residualizer_to_sample(
		features,
		embedding_inputs=(),
		per_survey_token_indices={},
		residualizer=None,
	)

	assert actual is features


def test_label_writer_applies_legacy_residualizer_with_coordinate_keys(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.array([[[[10.0, 20.0], [11.0, 21.0]]]], dtype=np.float32),
		valid=np.ones((1, 1, 2), dtype=np.bool_),
	)
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

	results = write_labels_for_k(
		output_dir=tmp_path / 'clusters',
		k=1,
		embedding_inputs=discover_embedding_inputs(input_dir),
		residualizer=legacy,
		preprocessor=_IdentityPreprocessor(),
		kmeans=_ZeroKMeans(),
		prediction_batch_size=1,
		label_metadata={},
	)

	assert results[0].cluster_counts == {0: 2}
	labels = np.load(results[0].labels_path)
	np.testing.assert_array_equal(labels, np.zeros((1, 1, 2), dtype=np.int32))


def test_multi_model_label_writer_matches_single_passes_and_transforms_once(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.arange(18, dtype=np.float32).reshape(1, 2, 3, 3),
		valid=np.array([[[True, False, True], [True, True, True]]]),
	)
	inputs = discover_embedding_inputs(input_dir)
	multi_preprocessor = _CountingPreprocessor()
	timer = StageTimer(enabled=True)
	multi = write_labels_for_models(
		output_dir=tmp_path / 'multi',
		kmeans_by_k={3: _ModuloKMeans(3), 1: _ModuloKMeans(1)},
		embedding_inputs=inputs,
		residualizer=None,
		preprocessor=multi_preprocessor,
		prediction_batch_size=2,
		label_metadata={'source': 'test'},
		timer=timer,
	)

	assert list(multi) == [3, 1]
	assert multi_preprocessor.call_count == 3
	assert set(timer.to_dict()['stages']) == {
		'feature_read',
		'predict_k1',
		'predict_k3',
		'preprocess',
		'write',
	}
	for k in (3, 1):
		reference_labels, reference_counts, reference_metadata = (
			_reference_labels_for_model(
				inputs[0],
				k=k,
				prediction_batch_size=2,
				label_metadata={'source': 'test'},
			)
		)
		np.testing.assert_array_equal(
			np.load(multi[k][0].labels_path),
			reference_labels,
		)
		assert multi[k][0].cluster_counts == reference_counts
		multi_metadata = json.loads(
			multi[k][0].metadata_path.read_text(encoding='utf-8'),
		)
		multi_metadata.pop('label_path')
		assert multi_metadata == reference_metadata


def test_multi_model_label_writer_removes_partial_outputs_on_predict_failure(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.ones((1, 1, 3, 2), dtype=np.float32),
		valid=np.ones((1, 1, 3), dtype=np.bool_),
	)
	output_dir = tmp_path / 'clusters'

	with pytest.raises(RuntimeError, match='injected predict failure'):
		write_labels_for_models(
			output_dir=output_dir,
			kmeans_by_k={1: _ZeroKMeans(), 2: _FailingKMeans()},
			embedding_inputs=discover_embedding_inputs(input_dir),
			residualizer=None,
			preprocessor=_IdentityPreprocessor(),
			prediction_batch_size=2,
			label_metadata={},
		)

	assert not list(output_dir.rglob('*.npy'))
	assert not list(output_dir.rglob('*.json'))
	assert not list(output_dir.rglob('*.partial'))


def test_multi_model_label_writer_rolls_back_publication_failure(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	input_dir = tmp_path / 'embeddings'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.ones((1, 1, 2, 1), dtype=np.float32),
		valid=np.ones((1, 1, 2), dtype=np.bool_),
	)
	output_dir = tmp_path / 'clusters'
	labels_dir = output_dir / 'labels' / 'k1'
	labels_dir.mkdir(parents=True)
	labels_path = labels_dir / 'survey_a.cluster_labels_token.npy'
	metadata_path = labels_dir / 'survey_a.cluster_label_metadata.json'
	old_labels = np.full((1, 1, 2), 7, dtype=np.int32)
	np.save(labels_path, old_labels)
	metadata_path.write_text('{"old": true}\n', encoding='utf-8')
	path_type = type(tmp_path)
	original_replace = path_type.replace
	publication_count = 0

	def fail_second_publication(source: Path, target: Path) -> Path:
		nonlocal publication_count
		if source.name.endswith('.partial'):
			publication_count += 1
			if publication_count == 2:
				raise OSError('injected publication failure')
		return original_replace(source, target)

	monkeypatch.setattr(path_type, 'replace', fail_second_publication)
	with pytest.raises(OSError, match='injected publication failure'):
		write_labels_for_models(
			output_dir=output_dir,
			kmeans_by_k={1: _ZeroKMeans()},
			embedding_inputs=discover_embedding_inputs(input_dir),
			residualizer=None,
			preprocessor=_IdentityPreprocessor(),
			prediction_batch_size=2,
			label_metadata={},
		)

	np.testing.assert_array_equal(np.load(labels_path), old_labels)
	assert metadata_path.read_text(encoding='utf-8') == '{"old": true}\n'
	assert not list(output_dir.rglob('*.partial'))
	assert not list(output_dir.rglob('*.backup'))


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
	second_config = _config(input_dir, second_output)
	second_config['clustering']['stage_timing'] = True
	second = run_embedding_clustering(second_config)

	assert [result.k for result in first.results] == [2, 3]
	assert [result.k for result in second.results] == [2, 3]
	assert first.sample.sample_count == 7
	assert first.sample.total_valid_count == 7
	assert not (first_output / 'stage_timings.json').exists()
	assert (second_output / 'stage_timings.json').is_file()
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


def test_run_embedding_clustering_dispatches_stratigraphic_hmm_backend(
	tmp_path: Path,
) -> None:
	config = _config(tmp_path / 'embeddings', tmp_path / 'clusters')
	config['clustering']['method'] = 'stratigraphic_hmm_kmeans'
	config['clustering']['stratigraphic_hmm'] = _stratigraphic_hmm_config()

	with pytest.raises(FileNotFoundError, match=r'embeddings\.input_dir'):
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


def _stratigraphic_hmm_config() -> dict[str, object]:
	return {
		'iterations': 10,
		'z_axis': 2,
		'z_direction': 'increasing_downward',
		'transition': {
			'same_cost': 0.0,
			'advance_cost': 0.25,
			'jump_cost': 1.0,
			'reverse_cost': 1000000.0,
			'forbid_reverse': True,
			'max_jump': None,
		},
		'init': {'order_by': 'mean_z'},
		'update': {'empty_cluster_policy': 'keep_previous'},
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


class _IdentityPreprocessor:
	@staticmethod
	def transform(features: np.ndarray) -> np.ndarray:
		return features


class _ZeroKMeans:
	@staticmethod
	def predict(features: np.ndarray) -> np.ndarray:
		return np.zeros(features.shape[0], dtype=np.int32)


class _ModuloKMeans:
	def __init__(self, k: int) -> None:
		self.k = k

	def predict(self, features: np.ndarray) -> np.ndarray:
		return np.arange(features.shape[0], dtype=np.int32) % self.k


class _FailingKMeans:
	@staticmethod
	def predict(features: np.ndarray) -> np.ndarray:
		del features
		msg = 'injected predict failure'
		raise RuntimeError(msg)


class _CountingPreprocessor:
	def __init__(self) -> None:
		self.call_count = 0

	def transform(self, features: np.ndarray) -> np.ndarray:
		self.call_count += 1
		return features


def _reference_labels_for_model(
	item: EmbeddingInput,
	*,
	k: int,
	prediction_batch_size: int,
	label_metadata: dict[str, object],
) -> tuple[np.ndarray, dict[int, int], dict[str, object]]:
	"""Run an independent single-model, multi-pass label reference."""
	embeddings = np.load(item.embeddings_path)
	valid = np.load(item.valid_tokens_path)
	indices = np.flatnonzero(valid.reshape(-1))
	labels = np.full(embeddings.shape[:3], -1, dtype=np.int32)
	flat_embeddings = embeddings.reshape((-1, embeddings.shape[-1]))
	flat_labels = labels.reshape(-1)
	model = _ModuloKMeans(k)
	preprocessor = _IdentityPreprocessor()
	for start in range(0, indices.size, prediction_batch_size):
		batch_indices = indices[start : start + prediction_batch_size]
		features = np.asarray(flat_embeddings[batch_indices], dtype=np.float32)
		flat_labels[batch_indices] = model.predict(preprocessor.transform(features))
	counts_array = np.bincount(flat_labels[indices], minlength=k)
	counts = {
		label: int(count)
		for label, count in enumerate(counts_array)
	}
	metadata = {
		**label_metadata,
		'k': k,
		'survey_id': item.survey_id,
		'embedding_input': embedding_input_metadata(item),
		'token_grid_shape': list(embeddings.shape[:3]),
		'embedding_dim': int(embeddings.shape[-1]),
		'valid_token_count': int(indices.size),
		'invalid_token_count': int(valid.size - indices.size),
		'cluster_counts': {str(label): count for label, count in counts.items()},
	}
	return labels, counts, metadata
