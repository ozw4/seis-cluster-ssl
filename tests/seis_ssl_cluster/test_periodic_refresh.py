# ruff: noqa: PLR0913

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.stratigraphy import periodic_refresh
from seis_ssl_cluster.stratigraphy.multi_head import build_multi_head_target_manifest
from seis_ssl_cluster.stratigraphy.periodic_refresh import (
	HardTargetPolicy,
	HashedArtifactReference,
	InitialHMMArtifact,
	InitialPeriodicRefreshConfig,
	PeriodicRefreshConfig,
	PreviousCenterArtifact,
	load_periodic_refresh_generation,
	produce_initial_periodic_refresh_generation,
	produce_periodic_refresh_generation,
	quarantine_periodic_refresh_generation,
)
from seis_ssl_cluster.stratigraphy.targets import write_pseudo_target

if TYPE_CHECKING:
	from pathlib import Path

CANONICAL_KS = (6, 8, 10)


def test_periodic_refresh_publishes_valid_generation_and_reuses_unchanged_output(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	initial_dir = config.previous_generation_manifest.path.parent
	assert [path.name for path in initial_dir.iterdir()] == ['refresh_generation.json']
	result = produce_periodic_refresh_generation(config)
	manifest_mtime = result.manifest_path.stat().st_mtime_ns

	assert result.reused is False
	assert result.generation_dir.name == 'refresh_0001_epoch002'
	payload = load_periodic_refresh_generation(result.manifest_path)
	assert payload['status'] == 'COMPLETE'
	assert payload['generation_index'] == 1
	assert payload['refresh_after_epoch'] == 2
	assert payload['initial_hard_target_policy'] == {
		'confidence_mode': 'constant',
		'confidence': 1.0,
		'boundary_weight_mode': 'absent',
	}
	assert set(payload['centers']) == {str(k) for k in CANONICAL_KS}
	assert set(payload['final_labels']) == {str(k) for k in CANONICAL_KS}
	assert set(payload['per_k_targets']) == {str(k) for k in CANONICAL_KS}
	assert (result.generation_dir / 'prepared_features').is_dir()
	assert (
		result.generation_dir / 'pseudo_targets' / 'multi_head_target_manifest.json'
	).is_file()

	reused = produce_periodic_refresh_generation(config)
	assert reused.reused is True
	assert reused.manifest_sha256 == result.manifest_sha256
	assert reused.manifest_path.stat().st_mtime_ns == manifest_mtime


def test_periodic_refresh_binds_fixed_edge_margin_to_common_mask(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path, edge_margin=(0, 0, 1))
	result = produce_periodic_refresh_generation(config)
	labels = np.load(
		result.generation_dir
		/ 'hmm'
		/ 'k6'
		/ 'final_labels'
		/ 'survey.cluster_labels_token.npy',
		allow_pickle=False,
	)
	assert np.all(labels[:, :, 0] == -1)
	assert np.all(labels[:, :, -1] == -1)
	assert np.all(labels[:, :, 1:-1] >= 0)


def test_periodic_refresh_diagnostics_use_common_valid_mask(tmp_path: Path) -> None:
	config = _write_fixture(tmp_path, edge_margin=(0, 0, 1))
	result = produce_periodic_refresh_generation(config)
	diagnostics = json.loads(
		(result.generation_dir / 'refresh_diagnostics.json').read_text(
			encoding='utf-8'
		)
	)
	for k in CANONICAL_KS:
		assert diagnostics['per_k'][str(k)]['valid_token_count'] == 10


def test_periodic_refresh_validation_failure_leaves_output_unpublished(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _write_fixture(tmp_path)
	original_validator = periodic_refresh.validate_periodic_refresh_generation
	validation_calls = 0

	def fail_on_publication_validation(path: str | Path, **kwargs: object) -> None:
		nonlocal validation_calls
		validation_calls += 1
		if validation_calls == 2:
			assert not config.output_generation_dir.exists()
			raise RuntimeError('publication validation failure')
		original_validator(path, **kwargs)

	monkeypatch.setattr(
		periodic_refresh,
		'validate_periodic_refresh_generation',
		fail_on_publication_validation,
	)
	with pytest.raises(RuntimeError, match='publication validation failure'):
		produce_periodic_refresh_generation(config)
	assert validation_calls == 2
	assert not config.output_generation_dir.exists()
	assert not config.output_generation_dir.is_symlink()


def test_periodic_refresh_rejects_tampered_generation_until_quarantined(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	result = produce_periodic_refresh_generation(config)
	content_path = result.generation_dir / 'refresh_diagnostics.json'
	content_path.write_text(
		content_path.read_text(encoding='utf-8') + ' ', encoding='utf-8'
	)

	with pytest.raises(ValueError, match='stale or invalid'):
		produce_periodic_refresh_generation(config)

	quarantined = quarantine_periodic_refresh_generation(result.generation_dir)
	assert quarantined.is_dir()
	assert not result.generation_dir.exists()


def test_periodic_refresh_requires_immediate_previous_generation(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	stale_config = replace(
		config,
		generation_index=2,
		refresh_after_epoch=5,
		output_generation_dir=tmp_path / 'generations' / 'refresh_0002_epoch005',
	)
	with pytest.raises(ValueError, match='exactly one less'):
		produce_periodic_refresh_generation(stale_config)
	assert not stale_config.output_generation_dir.exists()


def test_periodic_refresh_rejects_unscheduled_epoch(tmp_path: Path) -> None:
	config = _write_fixture(tmp_path)
	unscheduled_config = replace(
		config,
		refresh_after_epoch=3,
		output_generation_dir=tmp_path / 'generations' / 'refresh_0001_epoch003',
	)
	with pytest.raises(ValueError, match='exact refresh schedule'):
		produce_periodic_refresh_generation(unscheduled_config)


def test_periodic_refresh_requires_current_student_embedding_semantics(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	descriptor_path = config.current_embedding_descriptor.path
	descriptor = json.loads(descriptor_path.read_text(encoding='utf-8'))
	descriptor['embedding_semantics'] = 'masked_student_embedding_v1'
	descriptor_path.write_text(
		json.dumps(descriptor, sort_keys=True) + '\n', encoding='utf-8'
	)
	invalid_config = replace(
		config,
		current_embedding_descriptor=_reference(descriptor_path),
	)
	with pytest.raises(ValueError, match='embedding semantics'):
		produce_periodic_refresh_generation(invalid_config)


def test_periodic_refresh_requires_embedding_hmm_emission_source(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	artifact = config.initial_hmm_artifacts[0]
	model = joblib.load(artifact.hmm_model.path)
	model['emission_source'] = 'z_coordinate'
	joblib.dump(model, artifact.hmm_model.path)
	invalid_artifact = replace(artifact, hmm_model=_reference(artifact.hmm_model.path))
	invalid_config = replace(
		config,
		initial_hmm_artifacts=(invalid_artifact, *config.initial_hmm_artifacts[1:]),
	)
	with pytest.raises(ValueError, match='emission_source must be embedding'):
		periodic_refresh._load_fixed_hmm(  # noqa: SLF001
			invalid_config.initial_hmm_artifacts[0]
		)


def test_periodic_refresh_persists_arbitrary_descriptor_name(tmp_path: Path) -> None:
	config = _write_fixture(tmp_path)
	original = config.current_embedding_descriptor.path
	custom = original.with_name('student_embedding_descriptor.json')
	original.replace(custom)
	config = replace(config, current_embedding_descriptor=_reference(custom))
	result = produce_periodic_refresh_generation(config)
	payload = load_periodic_refresh_generation(result.manifest_path)
	assert payload['embeddings']['descriptor']['path'].endswith(custom.name)
	assert (result.generation_dir / 'embeddings' / custom.name).is_file()


def test_periodic_refresh_rejects_foreign_persisted_center(tmp_path: Path) -> None:
	config = _write_fixture(tmp_path)
	result = produce_periodic_refresh_generation(config)
	foreign = tmp_path / 'foreign_centers.npy'
	shutil.copyfile(
		result.generation_dir / 'hmm' / 'k6' / 'centers_after.npy', foreign
	)
	payload = json.loads(result.manifest_path.read_text(encoding='utf-8'))
	payload['centers']['6']['after'] = {
		'path': str(foreign),
		'sha256': file_sha256(foreign),
	}
	result.manifest_path.write_text(
		json.dumps(payload, sort_keys=True) + '\n', encoding='utf-8'
	)
	with pytest.raises(ValueError, match='escapes the generation root'):
		load_periodic_refresh_generation(result.manifest_path)


def test_periodic_refresh_binds_declared_previous_centers_to_previous_generation(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	result = produce_periodic_refresh_generation(config)
	foreign = tmp_path / 'foreign_previous_centers.npy'
	np.save(foreign, np.zeros((6, 1), dtype=np.float32), allow_pickle=False)
	payload = json.loads(result.manifest_path.read_text(encoding='utf-8'))
	foreign_reference = {
		'path': str(foreign),
		'sha256': file_sha256(foreign),
	}
	payload['previous_centers']['6'] = foreign_reference
	payload['request_identity']['previous_centers']['6'] = foreign_reference
	result.manifest_path.write_text(
		json.dumps(payload, sort_keys=True) + '\n', encoding='utf-8'
	)
	with pytest.raises(ValueError, match='previous k=6 center identity'):
		load_periodic_refresh_generation(result.manifest_path)


def test_periodic_refresh_binds_centers_before_to_previous_generation(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	result = produce_periodic_refresh_generation(config)
	before_path = result.generation_dir / 'hmm' / 'k6' / 'centers_before.npy'
	np.save(
		before_path,
		np.full((6, 1), -123.0, dtype=np.float32),
		allow_pickle=False,
	)
	payload = json.loads(result.manifest_path.read_text(encoding='utf-8'))
	payload['centers']['6']['before']['sha256'] = file_sha256(before_path)
	for entry in payload['content_files']:
		if entry['path'] == 'hmm/k6/centers_before.npy':
			entry['sha256'] = file_sha256(before_path)
	payload['generation_content_sha256'] = hashlib.sha256(
		json.dumps(
			payload['content_files'],
			sort_keys=True,
			separators=(',', ':'),
			allow_nan=False,
		).encode()
	).hexdigest()
	result.manifest_path.write_text(
		json.dumps(payload, sort_keys=True) + '\n', encoding='utf-8'
	)
	with pytest.raises(ValueError, match=r'centers\.before does not match'):
		load_periodic_refresh_generation(result.manifest_path)


def test_periodic_refresh_preserves_source_confidence_and_boundary_arrays(
	tmp_path: Path,
) -> None:
	policy = HardTargetPolicy(
		confidence_mode='source_array',
		boundary_weight_mode='source_array',
	)
	config = _write_fixture(tmp_path, target_policy=policy)
	result = produce_periodic_refresh_generation(config)
	payload = load_periodic_refresh_generation(result.manifest_path)
	assert payload['initial_hard_target_policy'] == {
		'confidence_mode': 'source_array',
		'confidence': 1.0,
		'boundary_weight_mode': 'source_array',
	}
	target_entry = payload['per_k_targets']['6']['surveys']['survey']
	confidence = np.load(target_entry['confidence']['path'], allow_pickle=False)
	boundary = np.load(
		target_entry['boundary_weight']['path'], allow_pickle=False
	)
	valid = confidence > 0
	assert np.unique(confidence[valid]).size > 1
	assert np.all(boundary[valid] == 0.25)


def _write_fixture(
	tmp_path: Path,
	*,
	edge_margin: tuple[int, int, int] = (0, 0, 0),
	target_policy: HardTargetPolicy | None = None,
) -> PeriodicRefreshConfig:
	if target_policy is None:
		target_policy = HardTargetPolicy()
	embedding_root = tmp_path / 'current_embeddings'
	embedding_root.mkdir()
	token_shape = (1, 1, 12)
	embeddings = np.arange(12, dtype=np.float32).reshape((*token_shape, 1))
	valid_tokens = np.ones(token_shape, dtype=np.bool_)
	np.save(embedding_root / 'survey.embeddings.npy', embeddings, allow_pickle=False)
	np.save(
		embedding_root / 'survey.valid_tokens.npy', valid_tokens, allow_pickle=False
	)
	(embedding_root / 'survey.embedding_metadata.json').write_text(
		json.dumps(
			{
				'survey_id': 'survey',
				'source_amplitude_path': 'amplitude.npy',
				'checkpoint_path': 'checkpoint.pt',
				'checkpoint_sha256': 'checkpoint',
				'model_geometry': {'name': 'fixture'},
				'patch_size': [1, 1, 1],
				'token_grid_shape': list(token_shape),
				'window_size': [1, 1, 1],
				'overlap': [0, 0, 0],
				'normalization_stats_path': 'stats.json',
				'output_dtype': 'float32',
				'min_token_valid_fraction': 1.0,
				'zero_mask': {},
			}
		),
		encoding='utf-8',
	)
	state_hash = 'a' * 64
	descriptor = _write_embedding_descriptor(
		embedding_root, token_shape=token_shape, state_hash=state_hash
	)
	initial_manifest = _write_initial_targets(
		tmp_path,
		embedding_root,
		valid_tokens=valid_tokens,
		token_shape=token_shape,
		edge_margin=edge_margin,
		target_policy=target_policy,
	)
	clustering_config = tmp_path / 'initial_clustering.yaml'
	clustering_config.write_text('clustering: {}\n', encoding='utf-8')
	source_embedding_metadata = embedding_root / 'survey.embedding_metadata.json'
	artifacts, previous_centers = _write_hmm_artifacts(
		tmp_path, embeddings=embeddings, edge_margin=edge_margin
	)
	initial_result = produce_initial_periodic_refresh_generation(
		InitialPeriodicRefreshConfig(
			initial_hard_target_manifest=_reference(initial_manifest),
			initial_hmm_artifacts=artifacts,
			clustering_config=_reference(clustering_config),
			source_embedding_metadata=_reference(source_embedding_metadata),
			output_generation_dir=tmp_path / 'generations' / 'refresh_0000_initial',
			target_policy=target_policy,
		)
	)
	return PeriodicRefreshConfig(
		generation_index=1,
		refresh_after_epoch=2,
		source_student_state_sha256=state_hash,
		previous_generation_manifest=_reference(initial_result.manifest_path),
		current_embedding_descriptor=_reference(descriptor),
		initial_hard_target_manifest=_reference(initial_manifest),
		initial_hmm_artifacts=artifacts,
		clustering_config=_reference(clustering_config),
		source_embedding_metadata=_reference(source_embedding_metadata),
		previous_centers=previous_centers,
		output_generation_dir=tmp_path / 'generations' / 'refresh_0001_epoch002',
		target_policy=target_policy,
		prediction_batch_size=4,
	)


def _write_embedding_descriptor(
	root: Path,
	*,
	token_shape: tuple[int, int, int],
	state_hash: str,
) -> Path:
	embeddings_path = root / 'survey.embeddings.npy'
	valid_path = root / 'survey.valid_tokens.npy'
	metadata_path = root / 'survey.embedding_metadata.json'
	descriptor = {
		'artifact_type': 'embedding_refresh_extraction',
		'schema_version': 1,
		'status': 'COMPLETE',
		'completion_status': 'COMPLETE',
		'embedding_semantics': 'current_student_unmasked_eval_full_survey_v1',
		'source_student_state_sha256': state_hash,
		'outputs': {
			'survey': {
				'embeddings': {
					'path': embeddings_path.name,
					'sha256': file_sha256(embeddings_path),
					'shape': [*token_shape, 1],
					'dtype': 'float32',
				},
				'valid_tokens': {
					'path': valid_path.name,
					'sha256': file_sha256(valid_path),
					'shape': list(token_shape),
					'dtype': 'bool',
				},
				'metadata': {
					'path': metadata_path.name,
					'sha256': file_sha256(metadata_path),
				},
			}
		},
	}
	descriptor_path = root / 'refresh_extraction_descriptor.json'
	descriptor_path.write_text(
		json.dumps(descriptor, sort_keys=True) + '\n', encoding='utf-8'
	)
	return descriptor_path


def _write_initial_targets(
	tmp_path: Path,
	embedding_root: Path,
	*,
	valid_tokens: np.ndarray,
	token_shape: tuple[int, int, int],
	edge_margin: tuple[int, int, int],
	target_policy: HardTargetPolicy,
) -> Path:
	target_root = tmp_path / 'initial_targets'
	replay_root = tmp_path / 'initial_k6_replay'
	decoded_root = tmp_path / 'initial_decoded_labels'
	replay_decoded_root = tmp_path / 'initial_replay_decoded_labels'
	for root in (target_root, replay_root, decoded_root, replay_decoded_root):
		root.mkdir()
	for k in CANONICAL_KS:
		(decoded_root / f'k{k}').mkdir()
		(replay_decoded_root / f'k{k}').mkdir()
		label_offsets = edge_margin[-1] or 0
		labels = np.minimum(
			np.maximum(np.arange(token_shape[-1]) - label_offsets, 0), k - 1
		).astype(np.int32)
		labels = np.broadcast_to(labels, token_shape).copy()
		target_valid_tokens = valid_tokens.copy()
		for axis, margin in enumerate(edge_margin):
			if margin:
				slices = [slice(None)] * 3
				slices[axis] = slice(0, margin)
				target_valid_tokens[tuple(slices)] = False
				slices[axis] = slice(-margin, None)
				target_valid_tokens[tuple(slices)] = False
		labels[~target_valid_tokens] = -1
		confidence = np.zeros(token_shape, dtype=np.float32)
		if target_policy.confidence_mode == 'constant':
			confidence[target_valid_tokens] = np.float32(target_policy.confidence)
		else:
			confidence[target_valid_tokens] = np.linspace(
				0.25,
				0.75,
				int(np.count_nonzero(target_valid_tokens)),
				dtype=np.float32,
			)
		boundary_weight = (
			None
			if target_policy.boundary_weight_mode == 'absent'
			else np.where(target_valid_tokens, 0.25, 0.0).astype(np.float32)
		)
		decoded_path = decoded_root / f'k{k}' / 'survey.cluster_labels_token.npy'
		replay_decoded_path = (
			replay_decoded_root / f'k{k}' / 'survey.cluster_labels_token.npy'
		)
		np.save(decoded_path, labels, allow_pickle=False)
		np.save(replay_decoded_path, labels, allow_pickle=False)
		write_pseudo_target(
			target_root,
			k=k,
			survey_id='survey',
			labels=labels,
			confidence=confidence,
			valid_tokens=target_valid_tokens,
			boundary_weight=boundary_weight,
			metadata={'source_label_path': str(decoded_path)},
			schema_version=2 if boundary_weight is not None else 1,
			write_boundary_weight=boundary_weight is not None,
		)
		write_pseudo_target(
			replay_root,
			k=k,
			survey_id='survey',
			labels=labels,
			confidence=confidence,
			valid_tokens=target_valid_tokens,
			boundary_weight=boundary_weight,
			metadata={'source_label_path': str(replay_decoded_path)},
			schema_version=2 if boundary_weight is not None else 1,
			write_boundary_weight=boundary_weight is not None,
		)
	manifest_path = tmp_path / 'initial_target_manifest.json'
	build_multi_head_target_manifest(
		manifest_path=manifest_path,
		source_embedding_dir=embedding_root,
		head_roots=dict.fromkeys(CANONICAL_KS, target_root),
		replay_k6_root=replay_root,
	)
	return manifest_path


def _write_hmm_artifacts(
	tmp_path: Path,
	*,
	embeddings: np.ndarray,
	edge_margin: tuple[int, int, int],
) -> tuple[tuple[InitialHMMArtifact, ...], tuple[PreviousCenterArtifact, ...]]:
	preprocessor = StandardScaler().fit(embeddings.reshape(-1, 1))
	preprocessor_path = tmp_path / 'preprocessor.joblib'
	joblib.dump(preprocessor, preprocessor_path)
	transition = {
		'same_cost': 0.0,
		'advance_cost': 0.0,
		'jump_cost': 0.0,
		'reverse_cost': 0.0,
		'forbid_reverse': False,
		'max_jump': None,
	}
	path_prior = {
		'enabled': False,
		'initial_state': {'mode': 'none', 'weight': 0.0},
		'terminal_state': {'mode': 'none', 'weight': 0.0},
		'expected_boundaries': {
			'enabled': False,
			'target': 'auto_k_minus_1',
			'weight': 0.0,
		},
	}
	artifacts: list[InitialHMMArtifact] = []
	previous_centers: list[PreviousCenterArtifact] = []
	for k in CANONICAL_KS:
		model_root = tmp_path / 'initial_models' / f'k{k}'
		model_root.mkdir(parents=True)
		centers = preprocessor.transform(
			(np.arange(k, dtype=np.float32) + edge_margin[-1]).reshape(-1, 1)
		).astype(np.float32)
		centers_path = model_root / 'cluster_centers.npy'
		model_path = model_root / 'hmm_model.joblib'
		metadata_path = model_root / 'clustering_metadata.json'
		np.save(centers_path, centers, allow_pickle=False)
		joblib.dump(
			{
				'emission_source': 'embedding',
				'centers': centers,
				'transition_settings': transition,
			'edge_margin_tokens': list(edge_margin),
				'path_prior': path_prior,
				'transition_costs': np.zeros((k, k), dtype=np.float32),
				'initial_state_costs': np.zeros(k, dtype=np.float32),
				'terminal_state_costs': np.zeros(k, dtype=np.float32),
			},
			model_path,
		)
		prepared_identity = {
			'chunk_size_tokens': 4,
			'reuse': False,
			'force_rebuild': False,
			'cleanup': False,
			'persist': True,
			'directory': str(tmp_path / 'unused_external_cache'),
		}
		metadata_path.write_text(
			json.dumps(
				{
					'k': k,
					'stratigraphic_hmm': {
						'emission_source': 'embedding',
						'z_axis': 2,
						'z_direction': 'increasing_downward',
						'init': {'order_by': 'depth'},
						'update': {'empty_cluster_policy': 'keep_previous'},
						'edge_margin_tokens': list(edge_margin),
						'transition': transition,
						'transition_costs': [[0.0] * k for _ in range(k)],
						'path_prior': {
							**path_prior,
							'initial_state_costs': [0.0] * k,
							'terminal_state_costs': [0.0] * k,
						},
						'prepared_feature_cache': prepared_identity,
					},
				}
			)
			+ '\n',
			encoding='utf-8',
		)
		artifacts.append(
			InitialHMMArtifact(
				k=k,
				centers=_reference(centers_path),
				hmm_model=_reference(model_path),
				preprocessor=_reference(preprocessor_path),
				metadata=_reference(metadata_path),
			)
		)
		previous_centers.append(
			PreviousCenterArtifact(k=k, centers=_reference(centers_path))
		)
	return tuple(artifacts), tuple(previous_centers)


def _reference(path: Path) -> HashedArtifactReference:
	return HashedArtifactReference(path=path, sha256=file_sha256(path))
