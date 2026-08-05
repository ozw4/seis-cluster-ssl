# ruff: noqa: CPY001, PLR0913

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import torch
import yaml

import seis_ssl_cluster.training.strat_hmm.runner as runner_module
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
from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	SurveyManifest,
	SurveyNormalizationStats,
	write_manifest_json,
	write_normalization_stats,
)
from seis_ssl_cluster.embedding import REFRESH_EXTRACTION_DESCRIPTOR_NAME
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.stratigraphy.periodic_refresh import (
	HardTargetPolicy,
	HashedArtifactReference,
)
from seis_ssl_cluster.training.checkpoint import capture_rng_state, load_checkpoint
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	_validated_periodic_checkpoint_selection,
	inspect_stratigraphy_checkpoint,
)
from tests.seis_ssl_cluster.test_config_strat_hmm_multi_head import (
	_multi_head_config,
)
from tests.seis_ssl_cluster.test_random_mae_checkpoint import (
	_reference_checkpoint_config,
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


def test_periodic_runner_uses_exact_refresh_schedule() -> None:
	assert [
		epoch
		for epoch in range(1, 26)
		if runner_module._periodic_scheduled_epoch(epoch)  # noqa: SLF001
	] == [2, 5, 8, 11, 14, 17, 20]
	assert not runner_module._periodic_scheduled_epoch(25)  # noqa: SLF001


def test_periodic_runner_refresh_boundary_resume_matches_uninterrupted(  # noqa: C901, PLR0915
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Compare a refresh-boundary resume with uninterrupted training."""
	config = _periodic_config(tmp_path)
	initial_target = Path(config['pseudo_targets']['manifest'])  # type: ignore[index]
	amplitude_path = tmp_path / 'amplitude.npy'
	np.save(
		amplitude_path,
		np.arange(48, dtype=np.float32).reshape(2, 2, 12),
		allow_pickle=False,
	)
	stats_path = tmp_path / 'normalization_stats.json'
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id='survey',
			source_path=amplitude_path,
			grid_order=GRID_ORDER_XYZ,
			clip_low_percentile=0.0,
			clip_high_percentile=100.0,
			clip_low=-100.0,
			clip_high=100.0,
			median=0.0,
			iqr=1.0,
		),
		stats_path,
	)
	train_manifest = tmp_path / 'train_manifest.json'
	write_manifest_json(
		[
			SurveyManifest(
				survey_id='survey',
				root=tmp_path,
				amplitude=AmplitudeVolumeRecord(
					survey_id='survey',
					path=amplitude_path,
					shape_xyz=(2, 2, 12),
					dtype='float32',
					grid_order=GRID_ORDER_XYZ,
					normalization_stats_path=stats_path,
				),
			),
		],
		train_manifest,
	)
	config['manifests']['train'] = str(train_manifest)  # type: ignore[index]
	config['manifests']['train_path_list'] = str(tmp_path / 'train_paths.txt')  # type: ignore[index]
	Path(config['manifests']['train_path_list']).write_text(  # type: ignore[index]
		f'{amplitude_path}\n', encoding='utf-8'
	)
	config['data'].update(  # type: ignore[index]
		{
			'local_crop_size': [2, 2, 12],
			'min_valid_fraction': 0.0,
			'max_resample_attempts': 4,
			'normalized_clip_abs': 100.0,
			'amplitude_agc': {'enabled': False},
			'finite_check_mode': 'strict',
		}
	)
	config['model'].update(  # type: ignore[index]
		{
			'patch_size': [1, 1, 1],
			'encoder_dim': 3,
			'encoder_depth': 1,
			'encoder_heads': 1,
			'decoder_dim': 3,
			'decoder_depth': 1,
			'decoder_heads': 1,
		}
	)
	config['train'].update(  # type: ignore[index]
		{
			'batch_size': 1,
			'samples_per_epoch': 1,
			'epochs': 25,
			'max_steps': 3,
			'num_workers': 0,
			'shuffle': False,
			'device': 'cpu',
			'seed': 7,
			'lr': 1.0e-4,
			'encoder_lr': 1.0e-4,
			'weight_decay': 0.0,
			'amp': False,
			'grad_clip_norm': 1.0,
		}
	)
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
	initial_heads = config['pseudo_target_refresh']['initial_hmm_artifacts'][  # type: ignore[index]
		'heads'
	]
	for raw_k, raw_head in initial_heads.items():  # type: ignore[union-attr]
		k = int(raw_k)
		head = raw_head  # type: ignore[assignment]
		hmm_path = Path(head['hmm_model'])  # type: ignore[index]
		joblib.dump(
			{
				'emission_source': 'embedding',
				'centers': np.zeros((k, 3), dtype=np.float32),
				'transition_settings': transition,
				'edge_margin_tokens': [0, 0, 0],
				'path_prior': path_prior,
				'transition_costs': np.zeros((k, k), dtype=np.float32),
				'initial_state_costs': np.zeros(k, dtype=np.float32),
				'terminal_state_costs': np.zeros(k, dtype=np.float32),
			},
			hmm_path,
		)
		metadata_path = Path(head['model_metadata'])  # type: ignore[index]
		metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
		stratigraphic_hmm = metadata['stratigraphic_hmm']
		stratigraphic_hmm.update(
			{
				'transition': transition,
				'transition_costs': [[0.0] * k for _ in range(k)],
				'path_prior': {
					**path_prior,
					'initial_state_costs': [0.0] * k,
					'terminal_state_costs': [0.0] * k,
				},
			}
		)
		stratigraphic_hmm['prepared_feature_cache'].update(
			{
				'chunk_size_tokens': 4,
				'reuse': False,
				'force_rebuild': False,
				'cleanup': False,
				'persist': True,
				'directory': str(tmp_path / 'external_prepared_cache'),
			}
		)
		metadata_path.write_text(
			json.dumps(metadata, sort_keys=True) + '\n', encoding='utf-8'
		)

	checkpoint_config = _reference_checkpoint_config(tmp_path)
	checkpoint_config['paths']['output_root'] = str(tmp_path / 'teacher_run')  # type: ignore[index]
	checkpoint_config['manifests']['train'] = str(train_manifest)  # type: ignore[index]
	checkpoint_config['manifests']['train_path_list'] = str(  # type: ignore[index]
		tmp_path / 'train_paths.txt'
	)
	checkpoint_config['data'].update(  # type: ignore[index]
		{
			'local_crop_size': [2, 2, 12],
			'min_valid_fraction': 0.0,
			'normalized_clip_abs': 100.0,
			'amplitude_agc': {'enabled': False},
		}
	)
	checkpoint_config['model'].update(  # type: ignore[index]
		{
			'patch_size': [1, 1, 1],
			'encoder_dim': 3,
			'encoder_depth': 1,
			'encoder_heads': 1,
			'decoder_dim': 3,
			'decoder_depth': 1,
			'decoder_heads': 1,
		}
	)
	teacher = AmplitudeMAE3D(
		in_channels=1,
		out_channels=1,
		patch_size_xyz=(1, 1, 1),
		encoder_dim=3,
		encoder_depth=1,
		encoder_heads=1,
		decoder_dim=3,
		decoder_depth=1,
		decoder_heads=1,
	)
	with torch.inference_mode():
		feature_rows = []
		for value in range(48):
			window = torch.tensor([[[[[float(value)]]]]])
			encoded = teacher.encode_tokens(
				window, valid_mask=torch.ones((1, 1, 1, 1), dtype=torch.bool)
			)
			feature_rows.append(encoded['tokens'].reshape(-1, 3)[0].numpy())
	feature_rows_array = np.asarray(feature_rows, dtype=np.float32)
	for raw_k, raw_head in initial_heads.items():  # type: ignore[union-attr]
		k = int(raw_k)
		head = raw_head  # type: ignore[assignment]
		centers = feature_rows_array[np.linspace(0, 47, k, dtype=int)]
		centers_path = Path(head['centers'])  # type: ignore[index]
		np.save(centers_path, centers, allow_pickle=False)
		hmm_path = Path(head['hmm_model'])  # type: ignore[index]
		hmm_model = joblib.load(hmm_path)
		hmm_model['centers'] = centers
		joblib.dump(hmm_model, hmm_path)
	teacher_path = tmp_path / 'teacher.pt'
	torch.save(
		{'model_state_dict': teacher.state_dict(), 'config': checkpoint_config},
		teacher_path,
	)
	config['teacher'] = {'checkpoint': str(teacher_path)}
	config['student'] = {'unfreeze_top_blocks': 1}
	resolved_config = resolve_strat_hmm_pretext_config(config)

	build_calls: list[Path | None] = []
	original_build = runner_module._build_strat_hmm_dataset_and_dataloader  # noqa: SLF001

	def record_build(**kwargs: object) -> tuple[object, torch.utils.data.DataLoader]:
		target_override = kwargs.get('target_manifest_override')
		build_calls.append(
			None if target_override is None else Path(str(target_override))
		)
		return original_build(**kwargs)  # type: ignore[arg-type]

	monkeypatch.setattr(
		runner_module, '_build_strat_hmm_dataset_and_dataloader', record_build
	)
	extraction_calls: list[Path] = []
	original_extract = runner_module.extract_embeddings_from_loaded_model

	def record_extract(*args: object, **kwargs: object) -> object:
		extraction_calls.append(Path(str(args[2])))
		return original_extract(*args, **kwargs)  # type: ignore[arg-type]

	monkeypatch.setattr(
		runner_module, 'extract_embeddings_from_loaded_model', record_extract
	)
	latest_path = runner_module.run_strat_hmm_pretext_training(resolved_config)
	assert latest_path == (
		Path(str(resolved_config['paths']['output_root'])) / 'latest.pt'  # type: ignore[index]
	)
	checkpoint_path = latest_path
	assert checkpoint_path.name == 'latest.pt'
	payload_checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
	inspection = inspect_stratigraphy_checkpoint(payload_checkpoint)
	assert inspection['stratigraphy_checkpoint']['schema_version'] == 8  # type: ignore[index]
	assert payload_checkpoint['epoch'] == 3
	assert payload_checkpoint['global_step'] == 3
	assert payload_checkpoint['target_refresh_state']['active_generation_id'] == (  # type: ignore[index]
		'refresh_0001_epoch002'
	)
	assert not checkpoint_path.with_name('selected.pt').exists()
	assert build_calls == [
		Path(str(initial_target)).resolve(),
		(
			Path(str(resolved_config['pseudo_target_refresh']['generation_root']))  # type: ignore[index]
			/ 'generations'
			/ 'refresh_0001_epoch002'
			/ 'pseudo_targets'
			/ 'multi_head_target_manifest.json'
		).resolve(),
	]
	assert len(extraction_calls) == 1
	manifest_path = Path(
		str(payload_checkpoint['target_refresh_state']['active_generation_manifest_path'])  # type: ignore[index]
	)
	payload = runner_module.load_periodic_refresh_generation(manifest_path)
	assert payload['status'] == 'COMPLETE'
	assert payload['iterations'] == 2
	diagnostics = json.loads(
		Path(str(payload['refresh_diagnostics']['path'])).read_text(  # type: ignore[index]
			encoding='utf-8'
		)
	)
	assert len(diagnostics['per_k']['6']['iterations']) == 2
	assert Path(str(payload['embeddings']['descriptor']['path'])).is_file()  # type: ignore[index]
	refresh_root = Path(
		str(resolved_config['pseudo_target_refresh']['generation_root'])  # type: ignore[index]
	)
	chain = json.loads(
		(refresh_root / 'periodic_refresh_chain.json').read_text(
			encoding='utf-8'
		)
	)
	assert [item['generation_id'] for item in chain['generations']] == [  # type: ignore[index]
		'refresh_0000_initial',
		'refresh_0001_epoch002',
	]
	events = [
		json.loads(line)
		for line in (
			Path(str(resolved_config['paths']['output_root']))  # type: ignore[index]
			/ 'target_refresh_events.jsonl'
		)
		.read_text(encoding='utf-8')
		.splitlines()
	]
	assert any(
		item['event_type'] == 'refresh' and item['status'] == 'complete'
		for item in events
	)

	# Resume from the completed epoch-2 refresh checkpoint and compare it with
	# the uninterrupted epoch-3 run.  Both executions use the production
	# extraction, generation activation, and dataloader rebuild paths above.
	resumed_config = deepcopy(resolved_config)
	resumed_output_root = tmp_path / 'resumed'
	resumed_generation_root = resumed_output_root / 'target_refresh'
	resumed_config['paths']['output_root'] = str(resumed_output_root)  # type: ignore[index]
	resumed_config['pseudo_target_refresh']['generation_root'] = str(  # type: ignore[index]
		resumed_generation_root
	)
	resumed_config['identity']['scientific_identity']['generation_root'] = str(  # type: ignore[index]
		resumed_generation_root
	)
	resumed_config['train']['max_steps'] = 2  # type: ignore[index]
	partial_path = runner_module.run_strat_hmm_pretext_training(resumed_config)
	partial_payload = load_checkpoint(partial_path, map_location='cpu')
	assert partial_payload['epoch'] == 2
	assert partial_payload['training_state']['checkpoint_kind'] == 'refresh'  # type: ignore[index]
	assert partial_payload['target_refresh_state']['refresh_phase'] == (  # type: ignore[index]
		'refresh_complete'
	)

	resumed_config['train']['max_steps'] = 3  # type: ignore[index]
	resumed_path = runner_module.run_strat_hmm_pretext_training(
		resumed_config, resume=partial_path
	)
	resumed_payload = load_checkpoint(resumed_path, map_location='cpu')

	def assert_checkpoint_values_match(left: object, right: object) -> None:
		if isinstance(left, torch.Tensor):
			assert isinstance(right, torch.Tensor)
			assert torch.equal(left, right)
		elif isinstance(left, dict):
			assert isinstance(right, dict)
			assert left.keys() == right.keys()
			for key in left:
				assert_checkpoint_values_match(left[key], right[key])
		elif isinstance(left, list):
			assert isinstance(right, list)
			assert len(left) == len(right)
			for left_item, right_item in zip(left, right, strict=True):
				assert_checkpoint_values_match(left_item, right_item)
		elif isinstance(left, tuple):
			assert isinstance(right, tuple)
			assert len(left) == len(right)
			for left_item, right_item in zip(left, right, strict=True):
				assert_checkpoint_values_match(left_item, right_item)
		elif isinstance(left, np.ndarray):
			assert isinstance(right, np.ndarray)
			assert np.array_equal(left, right)
		else:
			assert left == right

	for key in (
		'model_state_dict',
		'stratigraphy_state_dict',
		'spatial_context_state_dict',
		'optimizer_state_dict',
		'scaler_state_dict',
		'rng_state',
		'metrics',
		'checkpoint_selection',
	):
		assert_checkpoint_values_match(payload_checkpoint[key], resumed_payload[key])
	assert resumed_payload['epoch'] == payload_checkpoint['epoch'] == 3
	assert resumed_payload['global_step'] == payload_checkpoint['global_step'] == 3
	for key in (
		'active_generation_id',
		'active_generation_index',
		'last_completed_refresh_epoch',
		'refresh_phase',
	):
		assert resumed_payload['target_refresh_state'][key] == (  # type: ignore[index]
			payload_checkpoint['target_refresh_state'][key]  # type: ignore[index]
		)

	full_generation = runner_module.load_periodic_refresh_generation(
		Path(
			str(payload_checkpoint['target_refresh_state']['active_generation_manifest_path'])  # type: ignore[index]
		)
	)
	resumed_generation = runner_module.load_periodic_refresh_generation(
		Path(
			str(resumed_payload['target_refresh_state']['active_generation_manifest_path'])  # type: ignore[index]
		)
	)
	assert full_generation['generation_id'] == resumed_generation['generation_id']
	assert full_generation['source_student_state_sha256'] == (
		resumed_generation['source_student_state_sha256']
	)

	full_target = load_multi_head_target_manifest(
		full_generation['canonical_multi_head_target_manifest']['path']  # type: ignore[index]
	)
	resumed_target = load_multi_head_target_manifest(
		resumed_generation['canonical_multi_head_target_manifest']['path']  # type: ignore[index]
	)
	for raw_k in ('6', '8', '10'):
		full_head = full_target['heads'][raw_k]['surveys']['survey']  # type: ignore[index]
		resumed_head = resumed_target['heads'][raw_k]['surveys']['survey']  # type: ignore[index]
		for artifact_name in ('labels', 'confidence', 'valid_tokens'):
			assert full_head[artifact_name]['sha256'] == (  # type: ignore[index]
				resumed_head[artifact_name]['sha256']  # type: ignore[index]
			)
		assert full_generation['centers'][raw_k]['after']['sha256'] == (  # type: ignore[index]
			resumed_generation['centers'][raw_k]['after']['sha256']  # type: ignore[index]
		)


def test_periodic_recovered_epoch_metrics_ignore_checkpoint_metadata() -> None:
	assert runner_module._periodic_epoch_metrics_from_checkpoint(  # noqa: SLF001
		{
			'metrics': {
				'loss': 0.25,
				'eligible_xy_column_count': 4.0,
				'trainable_parameter_count': 2.0,
				'frozen_parameter_count': 3.0,
				'amp_enabled': 0.0,
			}
		}
	) == {
		'loss': 0.25,
		'eligible_xy_column_count': 4.0,
		}


def test_periodic_runner_validates_complete_generation_before_producer(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	output_root = tmp_path / 'refresh_0001_epoch002'
	output_root.mkdir()
	manifest_path = output_root / 'refresh_generation.json'
	manifest_path.write_text('{}', encoding='utf-8')
	load_calls: list[tuple[Path, object]] = []

	monkeypatch.setattr(
		runner_module,
		'_periodic_request_identity',
		lambda _config: {'request': 'expected'},
	)

	def record_load(path: Path, **kwargs: object) -> dict[str, object]:
		load_calls.append((path, kwargs['expected_identity']))
		return {}

	monkeypatch.setattr(
		runner_module, 'load_periodic_refresh_generation', record_load
	)
	monkeypatch.setattr(
		runner_module,
		'produce_periodic_refresh_generation',
		lambda _config: pytest.fail('complete generation must be reused'),
	)

	result = runner_module._load_or_produce_periodic_refresh_generation(  # noqa: SLF001
		SimpleNamespace(output_generation_dir=output_root)
	)

	assert result == manifest_path.resolve()
	assert load_calls == [(manifest_path, {'request': 'expected'})]


def test_periodic_runner_pointer_rollback_rejects_foreign_state(
	tmp_path: Path,
) -> None:
	pointer_path = tmp_path / 'active_target_generation.json'
	old_pointer = {
		'manifest_path': str(tmp_path / 'old.json'),
		'manifest_sha256': '0' * 64,
	}
	new_pointer = {
		'manifest_path': str(tmp_path / 'new.json'),
		'manifest_sha256': '1' * 64,
	}
	pointer_path.write_text(
		json.dumps(
			{
				'manifest_path': str(tmp_path / 'foreign.json'),
				'manifest_sha256': '2' * 64,
			}
		),
		encoding='utf-8',
	)

	with pytest.raises(RuntimeError, match='foreign active target pointer'):
		runner_module._rollback_periodic_refresh_pointer(  # noqa: SLF001
			pointer_path=pointer_path,
			old_pointer=old_pointer,
			new_pointer=new_pointer,
		)


def test_periodic_runner_rolls_back_pointer_on_post_activation_failure(  # noqa: PLR0915
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	refresh_root = tmp_path / 'refresh'
	pointer_path = refresh_root / 'active_target_generation.json'
	pointer_path.parent.mkdir(parents=True)
	old_manifest = tmp_path / 'old' / 'refresh_generation.json'
	old_manifest.parent.mkdir()
	old_manifest.write_text('old', encoding='utf-8')
	new_manifest = refresh_root / 'generations' / 'refresh_0001_epoch002' / (
		'refresh_generation.json'
	)
	new_manifest.parent.mkdir(parents=True)
	new_manifest.write_text('new', encoding='utf-8')
	old_target = tmp_path / 'initial-target.json'
	old_target.write_text('target', encoding='utf-8')
	old_pointer = {
		'manifest_path': str(old_manifest.resolve()),
		'manifest_sha256': '0' * 64,
	}
	new_pointer = {
		'manifest_path': str(new_manifest.resolve()),
		'manifest_sha256': file_sha256(new_manifest),
	}
	runner_module._json_write_atomic(pointer_path, old_pointer)  # noqa: SLF001

	student = torch.nn.Linear(1, 1)
	teacher = torch.nn.Linear(1, 1)
	heads = torch.nn.Linear(1, 1)
	replacement = torch.nn.Linear(1, 1)
	optimizer = torch.optim.AdamW(student.parameters(), lr=1.0e-3)
	components = SimpleNamespace(
		student=student,
		teacher=teacher,
		heads=heads,
		replacement_token=replacement,
		optimizer=optimizer,
		mae_checkpoint_config={},
		trainability_summary=SimpleNamespace(
			trainable_parameter_count=2,
			frozen_parameter_count=0,
			trainable_names=('weight', 'bias'),
		),
	)
	student.eval()
	teacher.train()
	heads.eval()
	replacement.train()
	flags_before = tuple(
		module.training
		for module in (student, teacher, heads, replacement)
	)
	rng_before = capture_rng_state()
	events: list[str] = []
	metadata = tmp_path / 'source-metadata.json'
	clustering = tmp_path / 'clustering.yaml'
	for path in (metadata, clustering):
		path.write_text('{}', encoding='utf-8')
	artifact = lambda path: HashedArtifactReference(  # noqa: E731
		path=path,
		sha256=file_sha256(path),
	)
	initial_target = artifact(old_target)

	monkeypatch.setattr(
		runner_module,
		'_periodic_initial_artifacts',
		lambda _config: ((), artifact(clustering), artifact(metadata), initial_target),
	)
	monkeypatch.setattr(runner_module, '_periodic_previous_centers', lambda _path: ())
	monkeypatch.setattr(
		runner_module,
		'_periodic_target_policy',
		lambda _path: HardTargetPolicy(),
	)
	monkeypatch.setattr(
		runner_module,
		'_periodic_embedding_extraction_config',
		lambda **_kwargs: {},
	)

	def extract(_student, _config, output_dir, *_args, **_kwargs):
		Path(output_dir).mkdir(parents=True, exist_ok=True)
		(Path(output_dir) / REFRESH_EXTRACTION_DESCRIPTOR_NAME).write_text(
			'{}', encoding='utf-8'
		)

	monkeypatch.setattr(
		runner_module, 'extract_embeddings_from_loaded_model', extract
	)
	monkeypatch.setattr(
		runner_module,
		'_load_or_produce_periodic_refresh_generation',
		lambda _config: new_manifest,
	)

	def activate(**_kwargs):
		runner_module._json_write_atomic(pointer_path, new_pointer)  # noqa: SLF001
		return {
			'active_generation_id': 'refresh_0001_epoch002',
			'active_generation_manifest_path': str(new_manifest),
			'active_generation_manifest_sha256': new_pointer['manifest_sha256'],
			'active_generation_content_sha256': '3' * 64,
			'active_target_manifest_path': str(old_target),
			'active_target_manifest_sha256': file_sha256(old_target),
		}

	monkeypatch.setattr(runner_module, '_activate_periodic_generation', activate)
	monkeypatch.setattr(
		runner_module,
		'_build_strat_hmm_dataset_and_dataloader',
		lambda **_kwargs: (
			[],
			torch.utils.data.DataLoader(
				[0], generator=torch.Generator().manual_seed(17)
			),
		),
	)
	monkeypatch.setattr(
		runner_module,
		'load_periodic_refresh_generation',
		lambda _path, **_kwargs: {},
	)

	def append_event(_output_root, payload):
		events.append(str(payload['status']))
		if payload['status'] == 'complete':
			raise OSError('event log failure')

	monkeypatch.setattr(runner_module, '_append_target_refresh_event', append_event)
	old_loader = torch.utils.data.DataLoader(
		[0], generator=torch.Generator().manual_seed(23)
	)

	with pytest.raises(OSError, match='event log failure'):
		runner_module._perform_periodic_refresh(  # noqa: SLF001
			config={'pseudo_target_refresh': {'generation_root': str(refresh_root)}},
			output_root=tmp_path / 'output',
			manifests=[],
			dataset_kwargs={'seed': 23},
			components=components,
			device=torch.device('cpu'),
			dataloader=old_loader,
			state={
				'active_generation_index': 0,
				'active_generation_id': 'refresh_0000_initial',
				'active_generation_manifest_path': str(old_manifest),
				'active_generation_manifest_sha256': '0' * 64,
			},
			refresh_epoch=2,
			global_step=4,
		)

	assert runner_module._periodic_pointer(pointer_path) == old_pointer  # noqa: SLF001
	assert events == ['start', 'complete', 'failure']
	assert tuple(
		module.training for module in (student, teacher, heads, replacement)
	) == flags_before
	assert runner_module._rng_state_hash(capture_rng_state()) == (  # noqa: SLF001
		runner_module._rng_state_hash(rng_before)  # noqa: SLF001
	)


def test_periodic_event_append_quarantines_partial_trailing_record(
	tmp_path: Path,
) -> None:
	output_root = tmp_path / 'output'
	output_root.mkdir()
	event_path = output_root / 'target_refresh_events.jsonl'
	first = {'event_type': 'generation', 'status': 'complete'}
	partial = b'{"event_type":"refresh","status":"start"'
	event_path.write_bytes(
		json.dumps(first, sort_keys=True).encode() + b'\n' + partial
	)
	second = {'event_type': 'generation', 'status': 'recovered'}

	runner_module._append_target_refresh_event(output_root, second)  # noqa: SLF001

	assert [
		json.loads(line)
		for line in event_path.read_text(encoding='utf-8').splitlines()
	] == [first, second]
	quarantined = list(output_root.glob('target_refresh_events.jsonl.quarantine.*'))
	assert len(quarantined) == 1
	assert quarantined[0].read_bytes() == partial


def test_periodic_event_append_separates_unterminated_valid_record(
	tmp_path: Path,
) -> None:
	output_root = tmp_path / 'output'
	output_root.mkdir()
	event_path = output_root / 'target_refresh_events.jsonl'
	first = {'event_type': 'generation', 'status': 'complete'}
	event_path.write_text(json.dumps(first), encoding='utf-8')
	second = {'event_type': 'generation', 'status': 'next'}

	runner_module._append_target_refresh_event(output_root, second)  # noqa: SLF001

	assert [
		json.loads(line)
		for line in event_path.read_text(encoding='utf-8').splitlines()
	] == [first, second]


def test_periodic_resume_reconstructs_checkpoint_event_idempotently(
	tmp_path: Path,
) -> None:
	student = torch.nn.Linear(1, 1)
	optimizer = torch.optim.AdamW(student.parameters(), lr=1.0e-3)
	components = SimpleNamespace(student=student, optimizer=optimizer)
	state = {
		'active_generation_id': 'refresh_0001_epoch002',
		'active_generation_manifest_sha256': '1' * 64,
		'active_generation_content_sha256': '2' * 64,
		'active_target_manifest_sha256': '3' * 64,
		'source_student_state_sha256': '4' * 64,
		'refresh_phase': 'refresh_required',
	}
	payload = {
		'training_state': {'checkpoint_kind': 'epoch'},
		'epoch': 5,
		'global_step': 10,
	}

	for _ in range(2):
		runner_module._recover_periodic_checkpoint_event(  # noqa: SLF001
			output_root=tmp_path / 'output',
			payload=payload,
			state=state,
			components=components,
		)

	events = [
		json.loads(line)
		for line in (
			tmp_path / 'output' / 'target_refresh_events.jsonl'
		).read_text(encoding='utf-8').splitlines()
	]
	assert len(events) == 1
	assert events[0]['checkpoint_kind'] == 'epoch'
	assert events[0]['epoch'] == 5
	assert events[0]['refresh_phase'] == 'refresh_required'
