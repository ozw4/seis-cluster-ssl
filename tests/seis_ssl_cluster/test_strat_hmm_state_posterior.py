"""Focused contracts for immutable HMM state-posterior publication."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.clustering.features import (
	EmbeddingInput,
	embedding_input_metadata,
	file_sha256,
)
from seis_ssl_cluster.clustering.stratigraphic_hmm import (
	forward_backward_state_posteriors,
)
from seis_ssl_cluster.stratigraphy import state_posterior

if TYPE_CHECKING:
	from pathlib import Path


def test_export_survey_writes_exact_posterior_in_full_token_grid(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""The bounded writer emits the same marginals as the posterior core."""
	labels_path, valid_path, embedding = _tiny_survey(tmp_path)
	monkeypatch.setattr(
		state_posterior, '_source_label_path', lambda _source: labels_path
	)
	monkeypatch.setattr(
		state_posterior,
		'prepare_feature_batch_for_indices',
		lambda *_args, **_kwargs: np.array([[0.0], [1.0]], dtype=np.float32),
	)
	stats = state_posterior._PosteriorStats(2)
	output_root = tmp_path / 'out'
	output_root.mkdir()
	entry, diagnostics = state_posterior._export_survey(
		output_root,
		embedding,
		{'valid_tokens': state_posterior._reference(valid_path)},
		_tiny_model(),
		statistics=(stats, state_posterior._PosteriorStats(2)),
	)

	posterior = np.load(entry['posterior']['path'], allow_pickle=False)
	expected = forward_backward_state_posteriors(
		np.array([[0.0, 1.0], [1.0, 0.0]]),
		_tiny_model()['transition_costs'],
		initial_state_costs=_tiny_model()['initial_costs'],
		terminal_state_costs=_tiny_model()['terminal_costs'],
	).posterior.astype(np.float32)
	np.testing.assert_allclose(posterior[0, 0], expected)
	assert posterior.dtype == np.float32
	assert entry['posterior']['shape'] == [1, 1, 2, 2]
	assert entry['posterior']['dtype'] == 'float32'
	assert entry['valid_tokens']['shape'] == [1, 1, 2]
	assert entry['valid_tokens']['dtype'] == 'bool'
	assert np.array_equal(np.load(entry['valid_tokens']['path']), [[[True, True]]])
	assert diagnostics['posterior_argmax_viterbi_mismatch_rate'] == 0.0


def test_export_survey_rejects_viterbi_replay_mismatch(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A changed frozen label cannot publish a posterior artifact."""
	labels_path, valid_path, embedding = _tiny_survey(tmp_path, labels=(0, 0))
	monkeypatch.setattr(
		state_posterior, '_source_label_path', lambda _source: labels_path
	)
	monkeypatch.setattr(
		state_posterior,
		'prepare_feature_batch_for_indices',
		lambda *_args, **_kwargs: np.array([[0.0], [1.0]], dtype=np.float32),
	)
	output_root = tmp_path / 'out'
	output_root.mkdir()

	with pytest.raises(ValueError, match='Viterbi replay'):
		state_posterior._export_survey(
			output_root,
			embedding,
			{'valid_tokens': state_posterior._reference(valid_path)},
			_tiny_model(),
			statistics=(
				state_posterior._PosteriorStats(2),
				state_posterior._PosteriorStats(2),
			),
		)


def test_export_survey_uses_shared_frozen_hmm_replay(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Posterior and lateral export share the one side-effect-free replay core."""
	labels_path, valid_path, embedding = _tiny_survey(tmp_path)
	monkeypatch.setattr(
		state_posterior, '_source_label_path', lambda _source: labels_path
	)
	called = False

	def replay(*_args, **_kwargs):
		nonlocal called
		called = True
		return np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0, 1])

	monkeypatch.setattr(state_posterior, 'replay_frozen_hmm_trace', replay)
	output_root = tmp_path / 'out'
	output_root.mkdir()
	state_posterior._export_survey(
		output_root,
		embedding,
		{'valid_tokens': state_posterior._reference(valid_path)},
		_tiny_model(),
		statistics=(
			state_posterior._PosteriorStats(2),
			state_posterior._PosteriorStats(2),
		),
	)
	assert called


def test_export_survey_rejects_embedding_valid_mask_mismatch(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Embedding masks must exactly match the hard target before replay."""
	labels_path, valid_path, embedding = _tiny_survey(tmp_path)
	embedding_valid_path = tmp_path / 'embedding.valid_tokens.npy'
	np.save(embedding_valid_path, np.array([[[True, False]]], dtype=np.bool_))
	embedding = EmbeddingInput(
		survey_id=embedding.survey_id,
		embeddings_path=embedding.embeddings_path,
		valid_tokens_path=embedding_valid_path,
		metadata_path=embedding.metadata_path,
	)
	monkeypatch.setattr(
		state_posterior, '_source_label_path', lambda _source: labels_path
	)
	output_root = tmp_path / 'out'
	output_root.mkdir()

	with pytest.raises(ValueError, match='source embedding valid mask does not cover'):
		state_posterior._export_survey(
			output_root,
			embedding,
			{'valid_tokens': state_posterior._reference(valid_path)},
			_tiny_model(),
			statistics=(
				state_posterior._PosteriorStats(2),
				state_posterior._PosteriorStats(2),
			),
		)


def test_export_survey_accepts_hard_target_subset_of_embedding_mask(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Edge-excluded hard targets may use a strict subset of source embeddings."""
	labels_path, valid_path, embedding = _tiny_survey(tmp_path, labels=(0, -1))
	embedding_valid_path = tmp_path / 'embedding.valid_tokens.npy'
	np.save(embedding_valid_path, np.array([[[True, True]]], dtype=np.bool_))
	embedding = EmbeddingInput(
		survey_id=embedding.survey_id,
		embeddings_path=embedding.embeddings_path,
		valid_tokens_path=embedding_valid_path,
		metadata_path=embedding.metadata_path,
	)
	monkeypatch.setattr(
		state_posterior, '_source_label_path', lambda _source: labels_path
	)
	monkeypatch.setattr(
		state_posterior,
		'prepare_feature_batch_for_indices',
		lambda *_args, **_kwargs: np.array([[0.0]], dtype=np.float32),
	)
	output_root = tmp_path / 'out'
	output_root.mkdir()

	entry, _ = state_posterior._export_survey(
		output_root,
		embedding,
		{'valid_tokens': state_posterior._reference(valid_path)},
		_tiny_model(),
		statistics=(
			state_posterior._PosteriorStats(2),
			state_posterior._PosteriorStats(2),
		),
	)

	assert np.array_equal(
		np.load(entry['valid_tokens']['path'], allow_pickle=False),
		np.array([[[True, False]]]),
	)


@pytest.mark.parametrize(
	('posterior', 'valid'),
	[
		(np.array([[[[np.nan, 0.0]]]], dtype=np.float32), np.array([[[True]]])),
		(np.array([[[[-0.1, 1.1]]]], dtype=np.float32), np.array([[[True]]])),
		(np.array([[[[0.4, 0.4]]]], dtype=np.float32), np.array([[[True]]])),
		(np.array([[[[1.0, 0.0]]]], dtype=np.float32), np.array([[[False]]])),
	],
)
def test_validator_rejects_invalid_posterior_rows(
	posterior: np.ndarray,
	valid: np.ndarray,
) -> None:
	with pytest.raises(ValueError, match=r'finite|non-negative|sum|zero'):
		state_posterior._validate_posterior_array(posterior, valid, k=2)


def test_hashed_provenance_rejects_input_drift(tmp_path: Path) -> None:
	path = tmp_path / 'input.bin'
	path.write_bytes(b'frozen')
	reference = {'path': str(path), 'sha256': file_sha256(path)}
	path.write_bytes(b'drifted')

	with pytest.raises(ValueError, match='hash mismatch'):
		state_posterior._hashed_path(reference, 'input')


def test_source_label_path_accepts_legacy_metadata_with_hashed_label_reference(
	tmp_path: Path,
) -> None:
	"""Legacy source labels remain hash-verified through their manifest entry."""
	labels = tmp_path / 'labels.npy'
	np.save(labels, np.array([[[0]]], dtype=np.int32))
	metadata = tmp_path / 'metadata.json'
	metadata.write_text(
		json.dumps({'source': {'source_label_path': str(labels)}}),
		encoding='utf-8',
	)

	assert state_posterior._source_label_path(
		{
			'metadata': state_posterior._reference(metadata),
			'labels': state_posterior._reference(labels),
		}
	) == labels


def test_source_label_path_rejects_metadata_without_cryptographic_binding(
	tmp_path: Path,
) -> None:
	"""A source label cannot bypass digest verification."""
	labels = tmp_path / 'labels.npy'
	np.save(labels, np.array([[[0]]], dtype=np.int32))
	metadata = tmp_path / 'metadata.json'
	metadata.write_text(
		json.dumps({'source': {'source_label_path': str(labels)}}),
		encoding='utf-8',
	)

	with pytest.raises(TypeError, match='hard target labels'):
		state_posterior._source_label_path(
			{'metadata': state_posterior._reference(metadata)}
		)


def test_load_model_rejects_missing_emission_source(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Frozen posterior replay cannot infer the historical feature basis."""
	config = _config(tmp_path)
	model_dir = config.clustering_output_dir / 'models' / 'k6'
	model_dir.mkdir(parents=True)
	for filename in (
		'preprocessor.joblib',
		'hmm_model.joblib',
		'clustering_metadata.json',
	):
		(model_dir / filename).write_bytes(b'frozen')
	np.save(model_dir / 'cluster_centers.npy', np.zeros((6, 1), dtype=np.float32))
	monkeypatch.setattr(state_posterior.joblib, 'load', lambda _path: {})

	with pytest.raises(ValueError, match='must record a valid emission_source'):
		state_posterior._load_model(config, k=6)


@pytest.mark.parametrize(
	'artifact',
	['embedding', 'valid_tokens', 'metadata'],
)
def test_source_embedding_identity_rejects_all_recorded_artifact_drift(
	tmp_path: Path,
	artifact: str,
) -> None:
	root = tmp_path / 'embeddings'
	root.mkdir()
	embedding = EmbeddingInput(
		survey_id='survey',
		embeddings_path=root / 'survey.embeddings.npy',
		valid_tokens_path=root / 'survey.valid_tokens.npy',
		metadata_path=root / 'survey.embedding_metadata.json',
	)
	np.save(embedding.embeddings_path, np.zeros((1, 1, 1, 2), dtype=np.float32))
	np.save(embedding.valid_tokens_path, np.ones((1, 1, 1), dtype=np.bool_))
	embedding.metadata_path.write_text('{"survey_id": "survey"}\n', encoding='utf-8')
	source = {
		'surveys': {
			'survey': {
				'metadata': embedding_input_metadata(embedding),
				'embedding_path': str(embedding.embeddings_path),
				'embedding_sha256': file_sha256(embedding.embeddings_path),
				'metadata_path': str(embedding.metadata_path),
				'metadata_sha256': file_sha256(embedding.metadata_path),
				'valid_tokens_path': str(embedding.valid_tokens_path),
				'valid_tokens_sha256': file_sha256(embedding.valid_tokens_path),
			}
		}
	}
	if artifact == 'embedding':
		np.save(embedding.embeddings_path, np.ones((1, 1, 1, 2), dtype=np.float32))
	elif artifact == 'valid_tokens':
		np.save(embedding.valid_tokens_path, np.zeros((1, 1, 1), dtype=np.bool_))
	else:
		embedding.metadata_path.write_text('{"survey_id": "drift"}\n', encoding='utf-8')

	with pytest.raises(ValueError, match=f'source embedding {artifact} identity'):
		state_posterior._validate_source_embedding_identity(
			source, {'survey': embedding}
		)


def test_manifest_validation_rehashes_live_source_embedding_inputs(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	root = tmp_path / 'embeddings'
	root.mkdir()
	embedding = EmbeddingInput(
		survey_id='survey',
		embeddings_path=root / 'survey.embeddings.npy',
		valid_tokens_path=root / 'survey.valid_tokens.npy',
		metadata_path=root / 'survey.embedding_metadata.json',
	)
	np.save(embedding.embeddings_path, np.zeros((1, 1, 1, 2), dtype=np.float32))
	np.save(embedding.valid_tokens_path, np.ones((1, 1, 1), dtype=np.bool_))
	embedding.metadata_path.write_text('{"survey_id": "survey"}\n', encoding='utf-8')
	source_embedding = {
		'input_dir': str(root),
		'surveys': {
			'survey': {
				'metadata': embedding_input_metadata(embedding),
				'embedding_path': str(embedding.embeddings_path),
				'embedding_sha256': file_sha256(embedding.embeddings_path),
				'metadata_path': str(embedding.metadata_path),
				'metadata_sha256': file_sha256(embedding.metadata_path),
				'valid_tokens_path': str(embedding.valid_tokens_path),
				'valid_tokens_sha256': file_sha256(embedding.valid_tokens_path),
			}
		},
	}
	hard_manifest = tmp_path / 'hard_manifest.json'
	hard_manifest.write_text('{}\n', encoding='utf-8')
	source = {'source_embedding': source_embedding}
	payload = {
		'artifact_type': state_posterior.ARTIFACT_TYPE,
		'schema_version': state_posterior.SCHEMA_VERSION,
		'posterior_semantics': state_posterior.POSTERIOR_SEMANTICS,
		'head_ks': list(state_posterior.CANONICAL_KS),
		'cost_temperature': 1.0,
		'source_hard_manifest': state_posterior._reference(hard_manifest),
		'source_hard_export_handoff': {},
		'source_embedding': source_embedding,
		'heads': {str(k): {} for k in state_posterior.CANONICAL_KS},
	}
	monkeypatch.setattr(
		state_posterior,
		'load_multi_head_target_manifest',
		lambda _path, **_kwargs: source,
	)
	monkeypatch.setattr(
		state_posterior, '_validate_source_manifest', lambda _source: None
	)
	monkeypatch.setattr(
		state_posterior, '_validate_manifest_hard_source_anchor',
		lambda _payload, _source, *, validate_array_semantics: (
			{} if validate_array_semantics else pytest.fail('expected full validation')
		),
	)
	monkeypatch.setattr(
		state_posterior, '_validate_manifest_heads', lambda *_args, **_kwargs: None
	)
	np.save(embedding.embeddings_path, np.ones((1, 1, 1, 2), dtype=np.float32))
	posterior_manifest = tmp_path / 'posterior_manifest.json'
	posterior_manifest.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='source embedding embedding identity'):
		state_posterior.load_multi_head_state_posterior_manifest(posterior_manifest)


def test_hard_source_anchor_rejects_replaced_frozen_model(
	tmp_path: Path,
) -> None:
	"""Hard-label replay cannot substitute a different frozen model snapshot."""
	config, source, models = _anchored_hard_source(tmp_path)

	state_posterior._validate_hard_source_model_anchor(
		source, config=config, models=models
	)

	centers = tmp_path / 'clustering' / 'models' / 'k6' / 'cluster_centers.npy'
	centers.write_bytes(b'replaced center bytes')
	models[6]['frozen_identity'] = _frozen_identity(
		tmp_path / 'clustering', k=6
	)
	with pytest.raises(ValueError, match='hard source centers hash mismatch'):
		state_posterior._validate_hard_source_model_anchor(
			source, config=config, models=models
		)


def test_hard_source_anchor_rejects_replaced_frozen_config(
	tmp_path: Path,
) -> None:
	config, source, models = _anchored_hard_source(tmp_path)
	assert config.clustering_config is not None
	config.clustering_config.write_text('frozen: false\n', encoding='utf-8')

	with pytest.raises(ValueError, match='hard source clustering config hash mismatch'):
		state_posterior._validate_hard_source_model_anchor(
			source, config=config, models=models
		)


def test_hard_source_anchor_reconstructs_historical_handoff_model_identities(
	tmp_path: Path,
) -> None:
	config, source, models = _anchored_hard_source(
		tmp_path, include_model_artifacts=False
	)

	state_posterior._validate_hard_source_model_anchor(
		source, config=config, models=models
	)


def test_dry_run_plans_without_creating_outputs(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _config(tmp_path)
	monkeypatch.setattr(
		state_posterior,
		'load_multi_head_target_manifest',
		lambda _path: {'head_ks': list(state_posterior.CANONICAL_KS)},
	)
	monkeypatch.setattr(state_posterior, '_validate_frozen_inputs', lambda *_args: None)

	plans = state_posterior.export_multi_head_state_posteriors(config, dry_run=True)

	assert [plan.action for plan in plans] == ['NEW', 'NEW', 'NEW']
	assert not config.posterior_root.exists()


def test_all_k_export_reuses_complete_outputs_with_only_missing(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A complete immutable bundle is reused byte-for-byte on resume."""
	config, source, models, embedding = _export_fixture(tmp_path)
	monkeypatch.setattr(
		state_posterior,
		'load_multi_head_target_manifest',
		lambda _path, **_kwargs: source,
	)
	monkeypatch.setattr(state_posterior, '_validate_frozen_inputs', lambda *_args: None)
	monkeypatch.setattr(
		state_posterior,
		'_load_model',
		lambda _config, *, k: models[k],
	)
	monkeypatch.setattr(
		state_posterior,
		'discover_embedding_inputs',
		lambda _root: [embedding],
	)
	monkeypatch.setattr(
		state_posterior,
		'prepare_feature_batch_for_indices',
		lambda *_args, **_kwargs: np.array([[0.0], [1.0]], dtype=np.float32),
	)
	monkeypatch.setattr(
		state_posterior,
		'_manifest',
		lambda _config: {'publication': 'complete'},
	)
	monkeypatch.setattr(
		state_posterior,
		'load_multi_head_state_posterior_manifest',
		lambda _path: {'publication': 'complete'},
	)

	first = state_posterior.export_multi_head_state_posteriors(config)
	assert [plan.action for plan in first] == ['NEW', 'NEW', 'NEW']
	before = {
		path.relative_to(config.posterior_root): (
			path.stat().st_mtime_ns,
			path.read_bytes(),
		)
		for path in config.posterior_root.rglob('*')
		if path.is_file()
	}

	second = state_posterior.export_multi_head_state_posteriors(
		config,
		only_missing=True,
	)
	assert [plan.action for plan in second] == ['REUSE', 'REUSE', 'REUSE']
	after = {
		path.relative_to(config.posterior_root): (
			path.stat().st_mtime_ns,
			path.read_bytes(),
		)
		for path in config.posterior_root.rglob('*')
		if path.is_file()
	}
	assert after == before

	(config.posterior_root / 'k6' / 'unreferenced.bin').write_bytes(b'unknown')
	plans = state_posterior.plan_multi_head_state_posterior_exports(
		config,
		only_missing=True,
	)
	assert [plan.action for plan in plans] == ['QUARANTINE', 'REUSE', 'REUSE']
	assert plans[0].reason is not None
	assert 'files differ from metadata' in plans[0].reason


def test_export_quarantines_stale_handoff_before_republishing(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A stale owned handoff is preserved instead of atomically overwritten."""
	config, source, models, embedding = _export_fixture(tmp_path)
	monkeypatch.setattr(
		state_posterior,
		'load_multi_head_target_manifest',
		lambda _path, **_kwargs: source,
	)
	monkeypatch.setattr(state_posterior, '_validate_frozen_inputs', lambda *_args: None)
	monkeypatch.setattr(
		state_posterior,
		'_load_model',
		lambda _config, *, k: models[k],
	)
	monkeypatch.setattr(
		state_posterior,
		'discover_embedding_inputs',
		lambda _root: [embedding],
	)
	monkeypatch.setattr(
		state_posterior,
		'prepare_feature_batch_for_indices',
		lambda *_args, **_kwargs: np.array([[0.0], [1.0]], dtype=np.float32),
	)
	monkeypatch.setattr(
		state_posterior,
		'_manifest',
		lambda _config: {'publication': 'complete'},
	)
	monkeypatch.setattr(
		state_posterior,
		'load_multi_head_state_posterior_manifest',
		lambda _path: {'publication': 'complete'},
	)

	state_posterior.export_multi_head_state_posteriors(config)
	config.handoff_manifest.write_bytes(b'stale handoff bytes')
	monkeypatch.setattr(
		state_posterior,
		'load_multi_head_state_posterior_manifest',
		lambda _path: (_ for _ in ()).throw(ValueError('stale handoff')),
	)

	plans = state_posterior.export_multi_head_state_posteriors(
		config,
		only_missing=True,
	)

	assert [plan.action for plan in plans] == ['REUSE', 'REUSE', 'REUSE']
	quarantines = list(config.posterior_root.glob('handoff.json.quarantine-*'))
	assert len(quarantines) == 1
	assert quarantines[0].read_bytes() == b'stale handoff bytes'
	assert json.loads(config.handoff_manifest.read_text(encoding='utf-8')) == {
		'publication': 'complete'
	}


def test_manifest_loader_rejects_cross_k_valid_mask_mismatch(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A mask change in one completed head cannot pass manifest validation."""
	config, source, models, embedding = _export_fixture(tmp_path)
	monkeypatch.setattr(
		state_posterior,
		'_load_model',
		lambda _config, *, k: models[k],
	)
	monkeypatch.setattr(
		state_posterior,
		'discover_embedding_inputs',
		lambda _root: [embedding],
	)
	monkeypatch.setattr(
		state_posterior,
		'prepare_feature_batch_for_indices',
		lambda *_args, **_kwargs: np.array([[0.0], [1.0]], dtype=np.float32),
	)
	for k in state_posterior.CANONICAL_KS:
		state_posterior._export_head(config, source, k=k)

	head_path = config.posterior_root / 'k8' / 'head_metadata.json'
	head = json.loads(head_path.read_text(encoding='utf-8'))
	valid_path = config.posterior_root / 'k8' / 'survey.valid_tokens.npy'
	posterior_path = config.posterior_root / 'k8' / 'survey.state_posterior.npy'
	valid = np.load(valid_path)
	posterior = np.load(posterior_path)
	valid[0, 0, 1] = False
	posterior[0, 0, 1] = 0.0
	np.save(valid_path, valid, allow_pickle=False)
	np.save(posterior_path, posterior, allow_pickle=False)
	head['surveys']['survey']['valid_tokens'] = state_posterior._array_reference(
		valid_path,
		shape=valid.shape,
		dtype=valid.dtype,
	)
	head['surveys']['survey']['posterior'] = state_posterior._array_reference(
		posterior_path,
		shape=posterior.shape,
		dtype=posterior.dtype,
	)
	head_path.write_text(json.dumps(head), encoding='utf-8')
	config.source_hard_manifest.write_text('{}\n', encoding='utf-8')
	source['source_embedding'] = {}
	monkeypatch.setattr(
		state_posterior,
		'load_multi_head_target_manifest',
		lambda _path, **_kwargs: source,
	)
	monkeypatch.setattr(
		state_posterior,
		'_validate_source_manifest',
		lambda _source: None,
	)
	monkeypatch.setattr(
		state_posterior,
		'_validate_manifest_hard_source_anchor',
		lambda _payload, _source, *, validate_array_semantics: (
			{
				str(k): models[k]['identity']
				for k in state_posterior.CANONICAL_KS
			}
			if validate_array_semantics
			else pytest.fail('expected full validation')
		),
	)
	monkeypatch.setattr(
		state_posterior,
		'_validate_manifest_source_embedding_identity',
		lambda _source_embedding: None,
	)
	posterior_manifest = tmp_path / 'posterior_manifest.json'
	posterior_manifest.write_text(
		json.dumps(
			{
				'artifact_type': state_posterior.ARTIFACT_TYPE,
				'schema_version': state_posterior.SCHEMA_VERSION,
				'posterior_semantics': state_posterior.POSTERIOR_SEMANTICS,
				'head_ks': list(state_posterior.CANONICAL_KS),
				'cost_temperature': 1.0,
				'source_hard_manifest': state_posterior._reference(
					config.source_hard_manifest
				),
				'source_hard_export_handoff': {},
				'source_embedding': {},
				'heads': {
					str(k): json.loads(
						(
							config.posterior_root / f'k{k}' / 'head_metadata.json'
						).read_text(
							encoding='utf-8'
						)
					)
					for k in state_posterior.CANONICAL_KS
				},
			}
		),
		encoding='utf-8',
	)

	with pytest.raises(ValueError, match='valid mask differs'):
		state_posterior.load_multi_head_state_posterior_manifest(posterior_manifest)


def test_plan_quarantines_invalid_owned_output_without_only_missing(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _config(tmp_path)
	(config.posterior_root / 'k6').mkdir(parents=True)
	monkeypatch.setattr(
		state_posterior,
		'load_multi_head_target_manifest',
		lambda _path: {'head_ks': list(state_posterior.CANONICAL_KS)},
	)
	monkeypatch.setattr(state_posterior, '_validate_frozen_inputs', lambda *_args: None)
	monkeypatch.setattr(
		state_posterior,
		'_validate_complete_head',
		lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError('partial')),
	)

	plans = state_posterior.plan_multi_head_state_posterior_exports(
		config, only_missing=False
	)

	assert [plan.action for plan in plans] == ['QUARANTINE', 'NEW', 'NEW']


def test_plan_quarantines_malformed_head_metadata(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Missing owned head fields make the output partial, not fatal to planning."""
	config = _config(tmp_path)
	head_path = config.posterior_root / 'k6' / 'head_metadata.json'
	head_path.parent.mkdir(parents=True)
	head_path.write_text('{}\n', encoding='utf-8')
	monkeypatch.setattr(
		state_posterior,
		'load_multi_head_target_manifest',
		lambda _path: {'head_ks': list(state_posterior.CANONICAL_KS)},
	)
	monkeypatch.setattr(state_posterior, '_validate_frozen_inputs', lambda *_args: None)

	plans = state_posterior.plan_multi_head_state_posterior_exports(
		config, only_missing=True
	)

	assert [plan.action for plan in plans] == ['QUARANTINE', 'NEW', 'NEW']
	assert plans[0].reason is not None
	assert 'manifest keys mismatch' in plans[0].reason


def test_diagnostics_validation_requires_all_posterior_metrics(tmp_path: Path) -> None:
	metrics = state_posterior._PosteriorStats(6).finish()
	diagnostics_json = tmp_path / 'diagnostics.json'
	diagnostics_csv = tmp_path / 'diagnostics.csv'
	broken = dict(metrics)
	broken.pop('expected_normalized_order')
	payload = {
		'per_survey': {'survey': broken},
		'aggregate': {**metrics, 'survey_count': 1},
	}
	diagnostics_json.write_text(json.dumps(payload), encoding='utf-8')
	diagnostics_csv.write_text('scope,survey_id,metric,value\n', encoding='utf-8')
	diagnostics = {
		**payload,
		'json': state_posterior._reference(diagnostics_json),
		'csv': state_posterior._reference(diagnostics_csv),
	}

	with pytest.raises(ValueError, match='manifest keys mismatch'):
		state_posterior._validate_diagnostics(diagnostics, k=6)


def test_quarantine_preserves_partial_owned_output(tmp_path: Path) -> None:
	partial = tmp_path / 'k6'
	partial.mkdir()
	(partial / 'old.bin').write_bytes(b'old artifact bytes')

	state_posterior._quarantine(partial)

	quarantines = list(tmp_path.glob('k6.quarantine-*'))
	assert len(quarantines) == 1
	assert (quarantines[0] / 'old.bin').read_bytes() == b'old artifact bytes'


def test_export_head_cleans_temporary_directory_after_producer_failure(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _config(tmp_path)
	model = _tiny_model()
	monkeypatch.setattr(state_posterior, '_load_model', lambda *_args, **_kwargs: model)
	monkeypatch.setattr(
		state_posterior,
		'discover_embedding_inputs',
		lambda _root: [_tiny_survey(tmp_path)[2]],
	)
	def fail_export_survey(*_args: object, **_kwargs: object) -> None:
		raise RuntimeError('producer failed')

	monkeypatch.setattr(state_posterior, '_export_survey', fail_export_survey)
	source = {
		'heads': {
			'6': {'surveys': {'survey': {}}},
		},
	}

	with pytest.raises(RuntimeError, match='producer failed'):
		state_posterior._export_head(config, source, k=6)

	assert not (config.posterior_root / 'k6').exists()
	assert not list(config.posterior_root.glob('.k6.posterior.*'))


def test_export_keeps_earlier_heads_staged_when_later_head_fails(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A replay/export failure must not publish any newly generated head."""
	config, source, models, embedding = _export_fixture(tmp_path)
	monkeypatch.setattr(
		state_posterior,
		'load_multi_head_target_manifest',
		lambda _path: source,
	)
	monkeypatch.setattr(state_posterior, '_validate_frozen_inputs', lambda *_args: None)
	monkeypatch.setattr(
		state_posterior,
		'_load_model',
		lambda _config, *, k: models[k],
	)
	monkeypatch.setattr(
		state_posterior,
		'discover_embedding_inputs',
		lambda _root: [embedding],
	)
	monkeypatch.setattr(
		state_posterior,
		'prepare_feature_batch_for_indices',
		lambda *_args, **_kwargs: np.array([[0.0], [1.0]], dtype=np.float32),
	)
	real_export_head = state_posterior._export_head

	def fail_k8(*args: object, **kwargs: object) -> None:
		if kwargs['k'] == 8:
			raise ValueError('Viterbi replay differs from frozen hard labels')
		real_export_head(*args, **kwargs)

	monkeypatch.setattr(state_posterior, '_export_head', fail_k8)

	with pytest.raises(ValueError, match='Viterbi replay'):
		state_posterior.export_multi_head_state_posteriors(config)

	assert not any((config.posterior_root / f'k{k}').exists() for k in (6, 8, 10))
	assert not config.handoff_manifest.exists()
	assert not list(config.posterior_root.glob('.posterior.bundle.*'))


def test_posterior_stats_uses_fixed_memory_histograms() -> None:
	stats = state_posterior._PosteriorStats(2)
	posterior = np.tile(np.array([[0.25, 0.75]]), (10_000, 1))
	labels = np.ones(10_000, dtype=np.int64)
	stats.trace(labels.size)
	stats.add(posterior, labels)

	assert stats.entropy.counts.shape == (4096,)
	assert stats.entropy.count == 10_000
	assert not hasattr(stats.entropy, 'values')


def _tiny_survey(
	tmp_path: Path,
	*,
	labels: tuple[int, int] = (0, 1),
) -> tuple[Path, Path, EmbeddingInput]:
	labels_path = tmp_path / 'labels.npy'
	valid_path = tmp_path / 'valid.npy'
	np.save(labels_path, np.asarray(labels, dtype=np.int64).reshape(1, 1, 2))
	np.save(
		valid_path,
		np.asarray(labels, dtype=np.int64).reshape(1, 1, 2) >= 0,
	)
	return (
		labels_path,
		valid_path,
		EmbeddingInput(
			survey_id='survey',
			embeddings_path=tmp_path / 'embeddings.npy',
			valid_tokens_path=valid_path,
			metadata_path=tmp_path / 'embedding_metadata.json',
		),
	)


def _tiny_model() -> dict[str, object]:
	return {
		'centers': np.array([[0.0], [1.0]], dtype=np.float32),
		'residualizer': None,
		'preprocessor': object(),
		'emission_source': 'embedding',
		'hmm': {},
		'transition_costs': np.array([[0.0, 0.0], [20.0, 0.0]]),
		'initial_costs': np.array([0.0, 2.0]),
		'terminal_costs': np.zeros(2),
		'identity': {},
	}


def _config(tmp_path: Path) -> state_posterior.MultiHeadStatePosteriorExportConfig:
	return state_posterior.MultiHeadStatePosteriorExportConfig(
		source_hard_manifest=tmp_path / 'hard_manifest.json',
		clustering_output_dir=tmp_path / 'clustering',
		source_embedding_dir=tmp_path / 'embeddings',
		posterior_root=tmp_path / 'posterior',
		clustering_config=None,
		handoff_manifest=tmp_path / 'posterior' / 'handoff.json',
	)


def _export_fixture(
	tmp_path: Path,
) -> tuple[
	state_posterior.MultiHeadStatePosteriorExportConfig,
	dict[str, object],
	dict[int, dict[str, object]],
	EmbeddingInput,
]:
	"""Create the smallest frozen source that can drive every exporter stage."""
	config = _config(tmp_path)
	valid_path = tmp_path / 'hard_valid_tokens.npy'
	np.save(valid_path, np.ones((1, 1, 2), dtype=np.bool_))
	heads: dict[str, object] = {}
	models: dict[int, dict[str, object]] = {}
	for k in state_posterior.CANONICAL_KS:
		labels_path = tmp_path / f'labels_k{k}.npy'
		np.save(labels_path, np.array([[[0, 1]]], dtype=np.int32))
		target_metadata = tmp_path / f'target_k{k}.metadata.json'
		target_metadata.write_text(
			json.dumps(
				{
					'source': {
						'source_label_path': str(labels_path),
						'source_label_sha256': file_sha256(labels_path),
					}
				}
			),
			encoding='utf-8',
		)
		heads[str(k)] = {
			'surveys': {
				'survey': {
					'metadata': state_posterior._reference(target_metadata),
					'valid_tokens': state_posterior._reference(valid_path),
				}
			}
		}
		model_dir = tmp_path / 'models' / f'k{k}'
		model_dir.mkdir(parents=True)
		identity = {}
		for name, filename in {
			'preprocessor': 'preprocessor.joblib',
			'hmm_model': 'hmm_model.joblib',
			'centers': 'cluster_centers.npy',
			'metadata': 'clustering_metadata.json',
		}.items():
			path = model_dir / filename
			path.write_bytes(name.encode())
			identity[name] = state_posterior._reference(path)
		centers = np.full((k, 1), 20.0, dtype=np.float32)
		centers[0, 0] = 0.0
		centers[1, 0] = 1.0
		models[k] = {
			'centers': centers,
			'residualizer': None,
			'preprocessor': object(),
			'emission_source': 'embedding',
			'hmm': {},
			'transition_costs': np.zeros((k, k), dtype=np.float32),
			'initial_costs': np.zeros(k, dtype=np.float32),
			'terminal_costs': np.zeros(k, dtype=np.float32),
			'identity': identity,
		}
	embedding = EmbeddingInput(
		survey_id='survey',
		embeddings_path=tmp_path / 'embeddings.npy',
		valid_tokens_path=valid_path,
		metadata_path=tmp_path / 'embedding_metadata.json',
	)
	return config, {'head_ks': [6, 8, 10], 'heads': heads}, models, embedding


def _anchored_hard_source(
	tmp_path: Path,
	*,
	include_model_artifacts: bool = True,
) -> tuple[
	state_posterior.MultiHeadStatePosteriorExportConfig,
	dict[str, object],
	dict[int, dict[str, object]],
]:
	clustering = tmp_path / 'clustering'
	config_path = tmp_path / 'frozen_config.yaml'
	config_path.write_text('frozen: true\n', encoding='utf-8')
	target_root = tmp_path / 'hard_targets'
	heads: dict[str, object] = {}
	metadata_hashes: dict[str, str] = {}
	model_artifacts: dict[str, dict[str, object]] = {}
	label_hashes: dict[str, dict[str, str]] = {}
	models: dict[int, dict[str, object]] = {}
	for k in state_posterior.CANONICAL_KS:
		label = clustering / 'labels' / f'k{k}' / 'survey.cluster_labels_token.npy'
		label.parent.mkdir(parents=True, exist_ok=True)
		np.save(label, np.array([[[0]]], dtype=np.int32))
		model_metadata = clustering / 'models' / f'k{k}' / 'clustering_metadata.json'
		model_metadata.parent.mkdir(parents=True, exist_ok=True)
		model_metadata.write_text(f'{{"k": {k}}}\n', encoding='utf-8')
		for name, content in (
			('preprocessor.joblib', b'preprocessor'),
			('hmm_model.joblib', b'hmm'),
		):
			(model_metadata.parent / name).write_bytes(content)
		centers = model_metadata.parent / 'cluster_centers.npy'
		centers.write_bytes(f'centers-{k}'.encode())
		target_metadata = target_root / f'k{k}' / 'survey.metadata.json'
		target_metadata.parent.mkdir(parents=True, exist_ok=True)
		target_metadata.write_text(
			json.dumps(
				{
					'source': {
						'source_label_path': str(label),
						'source_label_sha256': file_sha256(label),
					}
				}
			),
			encoding='utf-8',
		)
		heads[str(k)] = {
			'pseudo_target_root': str(target_root / f'k{k}'),
			'surveys': {
				'survey': {'metadata': state_posterior._reference(target_metadata)}
			},
		}
		metadata_hashes[str(k)] = file_sha256(model_metadata)
		label_hashes[str(k)] = {label.name: file_sha256(label)}
		model_artifacts[str(k)] = _frozen_identity(clustering, k=k)
		models[k] = {'frozen_identity': dict(model_artifacts[str(k)])}
	clustering_handoff: dict[str, object] = {
		'path': str(clustering),
		'config_path': str(config_path),
		'config_sha256': file_sha256(config_path),
		'metadata_sha256': metadata_hashes,
		'labels': label_hashes,
	}
	if include_model_artifacts:
		clustering_handoff['model_artifacts'] = {
			str(k): {**identity, 'residualizer': None}
			for k, identity in model_artifacts.items()
		}
	handoff = {
		'artifact_type': 'strat_hmm_multi_head_pseudo_target_export_handoff',
		'schema_version': 2,
		'completion_status': 'COMPLETE',
		'pseudo_target_root': str(target_root),
		'clustering': clustering_handoff,
		'heads': {
			str(k): {'pseudo_target_root': str(target_root / f'k{k}')}
			for k in state_posterior.CANONICAL_KS
		},
	}
	(target_root / 'multi_head_pseudo_target_export_handoff.json').write_text(
		json.dumps(handoff), encoding='utf-8'
	)
	return (
		state_posterior.MultiHeadStatePosteriorExportConfig(
			source_hard_manifest=tmp_path / 'hard_manifest.json',
			clustering_output_dir=clustering,
			source_embedding_dir=tmp_path / 'embeddings',
			posterior_root=tmp_path / 'posterior',
			clustering_config=config_path,
			handoff_manifest=tmp_path / 'posterior' / 'handoff.json',
		),
		{'heads': heads},
		models,
	)


def _frozen_identity(root: Path, *, k: int) -> dict[str, object]:
	model_dir = root / 'models' / f'k{k}'
	return {
		name: state_posterior._reference(model_dir / filename)
		for name, filename in {
			'preprocessor': 'preprocessor.joblib',
			'hmm_model': 'hmm_model.joblib',
			'centers': 'cluster_centers.npy',
			'metadata': 'clustering_metadata.json',
		}.items()
	}
