# ruff: noqa: CPY001, PLR0913

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pytest
import yaml

import seis_ssl_cluster.training.strat_hmm_checkpoint as checkpoint_module
from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.clustering.kmeans import PCASettings, fit_preprocessor
from seis_ssl_cluster.config.pretraining import (
	CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT,
	PERIODIC_REFRESH_CHECKPOINT_SELECTION_POLICY,
	PERIODIC_REFRESH_EXPERIMENT_ROLE,
	PERIODIC_REFRESH_MODEL_TAG,
	PERIODIC_REFRESH_VARIANT,
	_multi_head_target_hashes,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	_validated_periodic_checkpoint_selection,
	inspect_stratigraphy_checkpoint,
)
from tests.seis_ssl_cluster.test_config_strat_hmm_multi_head import (
	_multi_head_config,
)


def _periodic_config(tmp_path: Path) -> dict[str, object]:
	config = _multi_head_config(tmp_path)
	output_root = Path(config['paths']['output_root'])  # type: ignore[index]
	initial_root = tmp_path / 'initial_hmm'
	common_root = initial_root / 'common'
	common_root.mkdir(parents=True)
	clustering_config = {
		'embedding_normalization': 'none',
		'residualization': {'enabled': False},
		'pca': {'enabled': False, 'n_components': 2, 'whiten': False},
		'method': 'stratigraphic_hmm_kmeans',
		'k_values': [6, 8, 10],
		'stratigraphic_hmm': {
			'emission_source': 'embedding',
			'z_axis': 2,
			'z_direction': 'increasing_downward',
			'init': {'order_by': 'mean_z'},
			'update': {'empty_cluster_policy': 'keep_previous'},
		},
	}
	(common_root / 'clustering.yaml').write_text(
		yaml.safe_dump({'clustering': clustering_config}, sort_keys=False),
		encoding='utf-8',
	)
	preprocessor = fit_preprocessor(
		np.zeros((2, 3), dtype=np.float32),
		normalization='none',
		pca=PCASettings(enabled=False, n_components=2, whiten=False),
		seed=0,
	)
	joblib.dump(preprocessor, common_root / 'preprocessor.joblib')
	# The source metadata and the explicit embedding paths are immutable artifacts
	# already recorded by the target manifest.
	source_metadata = (
		Path(config['pseudo_targets']['manifest']).parent  # type: ignore[index]
		/ 'embeddings'
		/ 'survey.embedding_metadata.json'
	)
	assert source_metadata.is_file()
	source_input = {
		'survey_id': 'survey',
		'embeddings_path': str(source_metadata.with_name('survey.embeddings.npy')),
		'valid_tokens_path': str(source_metadata.with_name('survey.valid_tokens.npy')),
		'metadata_path': str(source_metadata),
		'metadata_sha256': file_sha256(source_metadata),
	}
	source_payload = json.loads(source_metadata.read_text(encoding='utf-8'))
	compatibility = {
		key: source_payload[key]
		for key in (
			'model_geometry',
			'patch_size',
			'window_size',
			'overlap',
			'min_token_valid_fraction',
			'zero_mask',
		)
	}
	compatibility['embedding_dim'] = 3
	model_identity = {
		'embedding_inputs': [source_input],
		'embedding_compatibility_signature': compatibility,
		'normalization': 'none',
		'residualization': {'enabled': False},
		'pca': {
			'enabled': False,
			'n_components': 2,
			'effective_n_components': None,
			'whiten': False,
		},
		'stratigraphic_hmm': {
			'emission_source': 'embedding',
			'z_axis': 2,
			'z_direction': 'increasing_downward',
			'init': {'order_by': 'mean_z'},
			'update': {'empty_cluster_policy': 'keep_previous'},
			'edge_margin_tokens': [0, 0, 0],
			'prepared_feature_cache': {
				'feature_mode': 'embedding',
				'dtype': 'float32',
				'schema_version': 1,
				'surveys': [{'survey_id': 'survey', 'feature_dim': 3}],
			},
		},
	}
	initial_heads: dict[str, dict[str, str]] = {}
	for k in (6, 8, 10):
		head_root = initial_root / f'k{k}'
		head_root.mkdir()
		model_metadata = head_root / 'model_metadata.json'
		model_metadata.write_text(
			json.dumps({'k': k, **model_identity}), encoding='utf-8'
		)
		centers = head_root / 'centers.npy'
		np.save(centers, np.zeros((k, 3), dtype=np.float32), allow_pickle=False)
		hmm_model = head_root / 'hmm_model.joblib'
		joblib.dump(
			{
				'emission_source': 'embedding',
				'centers': np.zeros((k, 3), dtype=np.float32),
			},
			hmm_model,
		)
		initial_heads[str(k)] = {
			'model_metadata': str(model_metadata),
			'hmm_model': str(hmm_model),
			'centers': str(centers),
		}

	config['spatial_context'] = deepcopy(CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT)
	config['student']['unfreeze_top_blocks'] = 1  # type: ignore[index]
	config['train']['epochs'] = 25  # type: ignore[index]
	config['identity']['model_tag'] = PERIODIC_REFRESH_MODEL_TAG  # type: ignore[index]
	manifest_path = Path(config['pseudo_targets']['manifest'])  # type: ignore[index]
	manifest = load_multi_head_target_manifest(manifest_path)
	scientific = config['identity']['scientific_identity']  # type: ignore[index]
	scientific.update(
		{
			'experiment_role': PERIODIC_REFRESH_EXPERIMENT_ROLE,
			'variant': PERIODIC_REFRESH_VARIANT,
			'head_spec': 'multi_resolution_ordered_prototypes_v1',
			'head_ks': [6, 8, 10],
			'target_representation': 'hard_viterbi_labels_v1',
			'target_manifest_sha256': file_sha256(manifest_path),
			'target_head_hashes': _multi_head_target_hashes(manifest),
			'objective_semantics': CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT['objective'],
			'mask_semantics': CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT['mask_semantics'],
			'column_fraction': 0.10,
			'selection_policy': CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT[
				'selection_policy'
			],
			'replacement': CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT['replacement'],
			'replacement_initialization': CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT[
				'replacement_initialization'
			],
			'rng_policy': CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT['rng_policy'],
			'masked_prototype_weight': 0.50,
			'visible_prototype_weight': 0.50,
			'distillation_scope': CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT[
				'distillation_scope'
			],
			'supervised_loss': 'structured_hmm_center_trace_masked_hard_v1',
			'consistency_policy': 'disabled_for_center_trace_masked_v1',
		}
	)
	config['pseudo_target_refresh'] = {
		'enabled': True,
		'semantics': 'periodic_student_hmm_center_refresh_v1',
		'generation_root': str(output_root / 'target_refresh'),
		'refresh_after_epochs': [2, 5, 8, 11, 14, 17, 20],
		'hmm_iterations_per_refresh': 2,
		'embedding_source': 'current_student',
		'embedding_mode': 'unmasked_eval_full_survey',
		'center_initialization': 'previous_generation',
		'center_update': 'full_mean',
		'preprocessing_policy': 'freeze_initial',
		'target_replacement': 'atomic_next_epoch',
		'empty_cluster_policy': 'error',
		'checkpoint_selection': 'final_completed_epoch',
		'initial_hmm_artifacts': {
			'common': {
				'clustering_config': str(common_root / 'clustering.yaml'),
				'preprocessor': str(common_root / 'preprocessor.joblib'),
				'residualizer': None,
				'source_embedding_metadata': str(source_metadata),
			},
			'heads': initial_heads,
		},
	}
	return config


def _periodic_event(
	sequence: int,
	*,
	epoch: int,
	global_step: int,
	kind: str,
	batch_index: int | None,
	phase: str,
	selected: bool = False,
) -> dict[str, object]:
	return {
		'sequence': sequence,
		'epoch': epoch,
		'global_step': global_step,
		'checkpoint_kind': kind,
		'batch_index': batch_index,
		'refresh_phase': phase,
		'selected': selected,
	}


def test_periodic_config_resolves_with_full_scientific_identity(
	tmp_path: Path,
) -> None:
	resolved = resolve_strat_hmm_pretext_config(_periodic_config(tmp_path))

	scientific = resolved['identity']['scientific_identity']  # type: ignore[index]
	assert scientific['model_role'] == 'mh_ctmask010_refresh3ep_hmm2_nocons'  # type: ignore[index]
	assert scientific['refresh_after_epochs'] == [2, 5, 8, 11, 14, 17, 20]  # type: ignore[index]
	assert scientific['preprocessing_policy'] == (  # type: ignore[index]
		'freeze_initial_residualizer_pca_v1'
	)
	assert scientific['target_activation_policy'] == (  # type: ignore[index]
		'atomic_next_epoch_activation_v1'
	)


def test_periodic_config_requires_scientific_identity(tmp_path: Path) -> None:
	config = _periodic_config(tmp_path)
	del config['identity']

	with pytest.raises(ValueError, match='top-level scientific identity'):
		resolve_strat_hmm_pretext_config(config)


def test_periodic_config_requires_initial_hmm_model_artifacts(tmp_path: Path) -> None:
	config = _periodic_config(tmp_path)
	head = config['pseudo_target_refresh']['initial_hmm_artifacts']['heads']['6']  # type: ignore[index]
	del head['hmm_model']

	with pytest.raises(ValueError, match='hmm_model is required'):
		resolve_strat_hmm_pretext_config(config)


def test_periodic_initial_artifacts_bind_centers_to_feature_dimension(
	tmp_path: Path,
) -> None:
	config = _periodic_config(tmp_path)
	heads = config['pseudo_target_refresh']['initial_hmm_artifacts']['heads']  # type: ignore[index]
	for key, head in heads.items():
		centers_path = Path(head['centers'])
		np.save(
			centers_path,
			np.zeros((int(key), 4), dtype=np.float32),
			allow_pickle=False,
		)

	with pytest.raises(ValueError, match='centers feature dimension'):
		resolve_strat_hmm_pretext_config(config)


def test_periodic_initial_artifacts_bind_shared_preprocessor_identity(
	tmp_path: Path,
) -> None:
	config = _periodic_config(tmp_path)
	preprocessor_path = Path(
		config['pseudo_target_refresh']['initial_hmm_artifacts']['common'][
			'preprocessor'
		]  # type: ignore[index]
	)
	preprocessor = fit_preprocessor(
		np.zeros((2, 3), dtype=np.float32),
		normalization='l2',
		pca=PCASettings(enabled=False, n_components=2, whiten=False),
		seed=0,
	)
	joblib.dump(preprocessor, preprocessor_path)

	with pytest.raises(ValueError, match='preprocessor normalization identity'):
		resolve_strat_hmm_pretext_config(config)


def test_periodic_initial_artifacts_bind_common_ordering_identity(
	tmp_path: Path,
) -> None:
	config = _periodic_config(tmp_path)
	clustering_path = Path(
		config['pseudo_target_refresh']['initial_hmm_artifacts']['common'][
			'clustering_config'
		]  # type: ignore[index]
	)
	clustering_config = yaml.safe_load(clustering_path.read_text(encoding='utf-8'))
	clustering_config['clustering']['stratigraphic_hmm']['init']['order_by'] = 'depth'
	clustering_path.write_text(
		yaml.safe_dump(clustering_config, sort_keys=False),
		encoding='utf-8',
	)

	with pytest.raises(ValueError, match=r'ordering init\.order_by identity'):
		resolve_strat_hmm_pretext_config(config)


@pytest.mark.parametrize(
	('field', 'value', 'match'),
	[
		('refresh_after_epochs', [2, 5, 8, 8, 14, 17, 20], 'refresh_after_epochs'),
		('hmm_iterations_per_refresh', 3, 'hmm_iterations_per_refresh'),
		('embedding_mode', 'masked', 'embedding_mode'),
	]
)
def test_periodic_config_rejects_schedule_and_policy_drift(
	tmp_path: Path,
	field: str,
	value: object,
	match: str,
) -> None:
	config = _periodic_config(tmp_path)
	config['pseudo_target_refresh'][field] = value  # type: ignore[index]

	with pytest.raises(ValueError, match=match):
		resolve_strat_hmm_pretext_config(config)


def test_periodic_selection_ignores_loss_and_only_selects_final_epoch() -> None:
	selection = {
		'schema_version': 1,
		'policy': PERIODIC_REFRESH_CHECKPOINT_SELECTION_POLICY,
		'events': [
			_periodic_event(
				0,
				epoch=24,
				global_step=100,
				kind='step',
				batch_index=3,
				phase='training',
			),
			_periodic_event(
				1,
				epoch=24,
				global_step=100,
				kind='epoch',
				batch_index=None,
				phase='refresh_required',
			),
		],
		'selected': None,
	}
	validated = _validated_periodic_checkpoint_selection(selection, train_epochs=25)
	assert validated['selected'] is None
	assert all('loss' not in event for event in validated['events'])

	bad = deepcopy(selection)
	bad['events'][1]['selected'] = True  # type: ignore[index]
	bad['selected'] = _periodic_event(  # type: ignore[index]
		1,
		epoch=24,
		global_step=100,
		kind='epoch',
		batch_index=None,
		phase='refresh_required',
		selected=True,
	)
	with pytest.raises(ValueError, match='only completed epoch 25'):
		_validated_periodic_checkpoint_selection(bad, train_epochs=25)

	final = deepcopy(selection)
	final['events'].append(  # type: ignore[index]
		_periodic_event(
			2,
			epoch=25,
			global_step=110,
			kind='epoch',
			batch_index=None,
			phase='training',
			selected=True,
		)
	)
	final['selected'] = _periodic_event(  # type: ignore[index]
		2,
		epoch=25,
		global_step=110,
		kind='epoch',
		batch_index=None,
		phase='training',
		selected=True,
	)
	assert _validated_periodic_checkpoint_selection(final, train_epochs=25)[
		'selected'
	] is not None


def test_schema8_inspection_reports_active_generation(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(
		checkpoint_module,
		'validate_stratigraphy_checkpoint_payload',
		lambda *_: None,
	)
	payload = {
		'stratigraphy_checkpoint': {
			'schema_version': 8,
			'target_representation': 'hard_viterbi_labels_v1',
			'objective_semantics': 'center_trace_masked_hmm_path_reconstruction_v1',
			'checkpoint_selection_policy': PERIODIC_REFRESH_CHECKPOINT_SELECTION_POLICY,
		},
		'model_state_dict': {},
		'stratigraphy_state_dict': {},
		'optimizer_state_dict': {},
		'target_refresh_state': {'active_generation_id': 'refresh_0000_initial'},
	}

	inspection = inspect_stratigraphy_checkpoint(payload)
	assert inspection['active_generation_id'] == 'refresh_0000_initial'
	assert inspection['best_selection_metric'] is None


def test_schema8_epoch_checkpoint_cannot_skip_refresh_boundary() -> None:
	state = {
		'refresh_phase': 'training',
		'last_completed_refresh_epoch': 0,
		'next_scheduled_refresh_epoch': 2,
	}
	for epoch in (2, 3):
		with pytest.raises(ValueError, match='scheduled refresh'):
			checkpoint_module._validate_periodic_checkpoint_phase(  # noqa: SLF001
				epoch=epoch,
				checkpoint_kind='epoch',
				state=state,
			)


def test_schema8_refresh_chain_requires_prior_manifest_link(tmp_path: Path) -> None:
	manifest_hashes = ('0' * 64, '1' * 64)
	generation_content_hashes = ('2' * 64, '3' * 64)
	generations = [
		{
			'generation_index': 0,
			'generation_id': 'refresh_0000_initial',
			'refresh_after_epoch': 0,
			'previous_generation_manifest_sha256': None,
			'source_student_state_sha256': None,
			'manifest_path': str(tmp_path / 'g0' / 'manifest.json'),
			'manifest_sha256': manifest_hashes[0],
			'generation_content_sha256': generation_content_hashes[0],
		},
		{
			'generation_index': 1,
			'generation_id': 'refresh_0001_epoch002',
			'refresh_after_epoch': 2,
			'previous_generation_manifest_sha256': 'f' * 64,
			'source_student_state_sha256': '4' * 64,
			'manifest_path': str(tmp_path / 'g1' / 'manifest.json'),
			'manifest_sha256': manifest_hashes[1],
			'generation_content_sha256': generation_content_hashes[1],
		},
	]
	chain = {
		'schema_version': 1,
		'semantics': 'periodic_student_hmm_refresh_chain_v1',
		'refresh_after_epochs': [2, 5, 8, 11, 14, 17, 20],
		'fixed_preprocessing_hmm_identity_sha256': '5' * 64,
		'generations': generations,
	}
	chain_path = tmp_path / 'periodic_refresh_chain.json'
	chain_path.write_text(json.dumps(chain), encoding='utf-8')
	loaded_payloads = [
		{
			'refresh_after_epoch': 0,
			'previous_generation_manifest': None,
			'source_student_state_sha256': None,
		},
		{
			'refresh_after_epoch': 2,
			'previous_generation_manifest': {'sha256': 'f' * 64},
			'source_student_state_sha256': '4' * 64,
		},
	]

	with pytest.raises(ValueError, match='chain is disconnected'):
		checkpoint_module._validate_periodic_refresh_chain(  # noqa: SLF001
			chain_path,
			chain_sha256=file_sha256(chain_path),
			generations=generations,
			loaded_payloads=loaded_payloads,
			expected_config=None,
		)
