'''Tests for Volve five-way checkpoint and embedding preflight.'''

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
import yaml

from proc.seis_ssl_cluster import audit_volve_horizon_five_way_sources as cli
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.volve.horizon_five_way_config import (
	FIVE_WAY_MODEL_IDS,
	LOCAL_BARLOW_TWINS_METHOD,
	volve_horizon_five_way_config_from_mapping,
)
from seis_ssl_cluster.volve.horizon_five_way_sources import (
	FIVE_WAY_LOCAL_BT_STAGE1_GLOBAL_STEPS,
	FIVE_WAY_STAGE1_GLOBAL_STEPS,
	FIVE_WAY_STAGE2_GLOBAL_STEPS,
	audit_volve_horizon_five_way_sources,
	inspect_volve_horizon_five_way_embedding_suite,
	plan_volve_horizon_five_way_embeddings,
	plan_volve_horizon_five_way_sources,
)


def test_source_plan_is_static_when_artifacts_are_missing(tmp_path: Path) -> None:
	config = volve_horizon_five_way_config_from_mapping(_config_mapping(tmp_path))

	sources = plan_volve_horizon_five_way_sources(config)
	embeddings = plan_volve_horizon_five_way_embeddings(config)

	assert tuple(row['model_id'] for row in sources) == FIVE_WAY_MODEL_IDS
	assert tuple(row['model_id'] for row in embeddings) == FIVE_WAY_MODEL_IDS
	assert not config.artifact_root.exists()


def test_checkpoint_audit_passes_and_reports_fixed_budgets(tmp_path: Path) -> None:
	universe = _write_universe(tmp_path, embeddings=False)
	report = audit_volve_horizon_five_way_sources(universe['config'])
	sources = cast('list[dict[str, object]]', report['sources'])

	assert report['model_order'] == list(FIVE_WAY_MODEL_IDS)
	assert sources[0]['stage_2'] == {
		'epochs': 25,
		'global_steps': FIVE_WAY_STAGE2_GLOBAL_STEPS,
		'unfreeze_top_blocks': 1,
	}
	assert sources[0]['parent_checkpoint_sha256'] == sources[1][
		'parent_checkpoint_sha256'
	]
	assert sources[2]['parent_checkpoint_sha256'] == sources[3][
		'parent_checkpoint_sha256'
	]
	assert sources[0]['parent_checkpoint_sha256'] != sources[2][
		'parent_checkpoint_sha256'
	]
	assert sources[-1]['stage_2'] is None


def test_checkpoint_audit_rejects_hmm_k_and_pseudo_source_swap(
	tmp_path: Path,
) -> None:
	universe = _write_universe(tmp_path / 'k', embeddings=False)
	payload = _load(universe['checkpoints']['mae_hmm_k6'])
	payload['stratigraphy_config']['head']['num_prototypes'] = 5
	_save(universe['checkpoints']['mae_hmm_k6'], payload)
	with pytest.raises(ValueError, match='prototype count'):
		audit_volve_horizon_five_way_sources(universe['config'])

	universe = _write_universe(tmp_path / 'swap', embeddings=False)
	metadata_path = universe['pseudo_metadata']['mae_hmm_k6']
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['source'] = {
		'checkpoint_path': str(universe['parents']['local']),
		'checkpoint_sha256': file_sha256(universe['parents']['local']),
	}
	_write_json(metadata_path, metadata)
	with pytest.raises(ValueError, match='does not match its stage-1 parent'):
		audit_volve_horizon_five_way_sources(universe['config'])


def test_checkpoint_audit_rejects_local_identity_and_trace_drop(
	tmp_path: Path,
) -> None:
	universe = _write_universe(tmp_path / 'method', embeddings=False)
	payload = _load(universe['checkpoints']['local_barlow_twins'])
	payload['config']['barlow_twins']['local_pairs_per_crop'] = 64
	_save(universe['checkpoints']['local_barlow_twins'], payload)
	with pytest.raises(ValueError, match='local_pairs_per_crop'):
		audit_volve_horizon_five_way_sources(universe['config'])

	universe = _write_universe(tmp_path / 'trace', embeddings=False)
	payload = _load(universe['checkpoints']['local_barlow_twins'])
	payload['config']['augmentations']['trace_drop_probability'] = 0.1
	_save(universe['checkpoints']['local_barlow_twins'], payload)
	with pytest.raises(ValueError, match='must be disabled'):
		audit_volve_horizon_five_way_sources(universe['config'])


def test_checkpoint_audit_rejects_random_budget_and_parent_sha(
	tmp_path: Path,
) -> None:
	universe = _write_universe(tmp_path / 'random', embeddings=False)
	payload = _load(universe['checkpoints']['random'])
	payload['metadata']['seed'] = 7
	_save(universe['checkpoints']['random'], payload)
	with pytest.raises(ValueError, match=r'metadata.seed'):
		audit_volve_horizon_five_way_sources(universe['config'])

	universe = _write_universe(tmp_path / 'budget', embeddings=False)
	payload = _load(universe['checkpoints']['mae'])
	payload['global_step'] -= 1
	_save(universe['checkpoints']['mae'], payload)
	with pytest.raises(ValueError, match='global_step'):
		audit_volve_horizon_five_way_sources(universe['config'])

	universe = _write_universe(tmp_path / 'sha', embeddings=False)
	payload = _load(universe['checkpoints']['mae'])
	payload['continuation_lineage']['init_checkpoint_sha256'] = 'f' * 64
	_save(universe['checkpoints']['mae'], payload)
	with pytest.raises(ValueError, match='does not match the parent file'):
		audit_volve_horizon_five_way_sources(universe['config'])


def test_checkpoint_audit_rejects_tampered_pseudo_target_file(
	tmp_path: Path,
) -> None:
	universe = _write_universe(tmp_path, embeddings=False)
	payload = _load(universe['checkpoints']['mae_hmm_k6'])
	identities = payload['control_identity']['input_identities']['pseudo_targets']
	labels_path = Path(identities[0]['labels']['path'])
	np.save(labels_path, np.ones((1, 1, 1), dtype=np.int16))

	with pytest.raises(ValueError, match='SHA-256 differs from its live file'):
		audit_volve_horizon_five_way_sources(universe['config'])


def test_checkpoint_audit_rejects_tampered_pseudo_source_chain(
	tmp_path: Path,
) -> None:
	universe = _write_universe(tmp_path, embeddings=False)
	pseudo_metadata = json.loads(
		universe['pseudo_metadata']['mae_hmm_k6'].read_text(encoding='utf-8')
	)
	cluster_metadata_path = Path(pseudo_metadata['source']['source_metadata_path'])
	cluster_metadata = json.loads(cluster_metadata_path.read_text(encoding='utf-8'))
	embedding_metadata_path = Path(
		cluster_metadata['embedding_input']['metadata_path']
	)
	with embedding_metadata_path.open('a', encoding='utf-8') as stream:
		stream.write('\n')

	with pytest.raises(ValueError, match='source embedding metadata SHA-256'):
		audit_volve_horizon_five_way_sources(universe['config'])


def test_checkpoint_audit_rejects_wrong_pseudo_target_survey(
	tmp_path: Path,
) -> None:
	universe = _write_universe(tmp_path, embeddings=False)
	metadata_path = universe['pseudo_metadata']['mae_hmm_k6']
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['survey_id'] = 'wrong_survey'
	_write_json(metadata_path, metadata)

	with pytest.raises(ValueError, match='pseudo target survey_id'):
		audit_volve_horizon_five_way_sources(universe['config'])


def test_embedding_suite_accepts_distinct_objectives_on_shared_support(
	tmp_path: Path,
) -> None:
	universe = _write_universe(tmp_path, embeddings=True)
	suite = inspect_volve_horizon_five_way_embedding_suite(universe['config'])

	assert tuple(suite.sources) == FIVE_WAY_MODEL_IDS
	assert suite.volume_shape_xyz == (16, 16, 800)
	assert suite.token_grid_shape_xyz == (2, 2, 100)
	assert suite.embedding_shape == (2, 2, 100, 384)
	assert suite.embedding_dim == 384
	assert suite.model_valid_lateral_mask.shape == (16, 16)
	assert len(suite.valid_tokens_sha256) == 64
	assert suite.sources['mae'].metadata['pretraining_objective'] != (
		suite.sources['local_barlow_twins'].metadata['pretraining_objective']
	)


def test_embedding_suite_rejects_wrong_array_embedding_dimension(
	tmp_path: Path,
) -> None:
	universe = _write_universe(tmp_path, embeddings=True)
	model = universe['config'].model_by_id('mae_hmm_k6')
	paths = output_paths(model.embeddings_dir, universe['config'].survey_id)
	np.save(paths.embeddings, np.zeros((2, 2, 100, 383), dtype=np.float16))

	with pytest.raises(ValueError, match='embedding array shape must equal'):
		inspect_volve_horizon_five_way_embedding_suite(universe['config'])


def test_embedding_suite_rejects_out_of_order_supplied_source_audit(
	tmp_path: Path,
) -> None:
	universe = _write_universe(tmp_path, embeddings=True)
	report = audit_volve_horizon_five_way_sources(universe['config'])
	report['sources'] = list(reversed(report['sources']))

	with pytest.raises(ValueError, match='fixed model order'):
		inspect_volve_horizon_five_way_embedding_suite(
			universe['config'],
			source_audit=report,
		)


@pytest.mark.parametrize(
	('field', 'replacement', 'message'),
	[
		('token_grid_shape', [2, 2, 99], 'token_grid_shape'),
		('window_size', [64, 128, 128], 'window_size'),
		('overlap', [32, 64, 64], 'overlap'),
		('precision', {'amp_enabled': False}, 'precision'),
	],
)
def test_embedding_suite_rejects_shared_metadata_drift(
	tmp_path: Path,
	field: str,
	replacement: object,
	message: str,
) -> None:
	universe = _write_universe(tmp_path, embeddings=True)
	path = universe['embedding_metadata']['mae_hmm_k6']
	metadata = json.loads(path.read_text(encoding='utf-8'))
	metadata[field] = replacement
	_write_json(path, metadata)

	with pytest.raises((TypeError, ValueError), match=message):
		inspect_volve_horizon_five_way_embedding_suite(universe['config'])


def test_embedding_suite_rejects_mask_and_checkpoint_identity_drift(
	tmp_path: Path,
) -> None:
	universe = _write_universe(tmp_path / 'mask', embeddings=True)
	paths = output_paths(
		universe['config'].model_by_id('mae_hmm_k6').embeddings_dir,
		universe['config'].survey_id,
	)
	mask = np.load(paths.valid_tokens)
	mask[0, 0, 0] = False
	np.save(paths.valid_tokens, mask)
	with pytest.raises(ValueError, match='valid-token mask differs'):
		inspect_volve_horizon_five_way_embedding_suite(universe['config'])

	universe = _write_universe(tmp_path / 'checkpoint', embeddings=True)
	path = universe['embedding_metadata']['mae_hmm_k6']
	metadata = json.loads(path.read_text(encoding='utf-8'))
	metadata['checkpoint_sha256'] = '0' * 64
	_write_json(path, metadata)
	with pytest.raises(ValueError, match='checkpoint SHA-256'):
		inspect_volve_horizon_five_way_embedding_suite(universe['config'])


def test_embedding_suite_rejects_canonical_input_content_drift(
	tmp_path: Path,
) -> None:
	universe = _write_universe(tmp_path, embeddings=True)
	metadata_path = universe['embedding_metadata']['mae']
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	amplitude_path = Path(metadata['source_amplitude_path'])
	np.save(amplitude_path, np.ones((1,), dtype=np.float32))

	with pytest.raises(ValueError, match='canonical_amplitude_sha256'):
		inspect_volve_horizon_five_way_embedding_suite(universe['config'])


def test_audit_cli_dry_run_needs_no_artifacts_and_writes_nothing(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	raw = _config_mapping(tmp_path)
	config_path = tmp_path / 'five_way.yaml'
	config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')
	before = {path.relative_to(tmp_path) for path in tmp_path.rglob('*')}
	monkeypatch.setattr(
		'sys.argv',
		['audit', '--config', str(config_path), '--dry-run'],
	)
	cli.main()
	after = {path.relative_to(tmp_path) for path in tmp_path.rglob('*')}
	payload = json.loads(capsys.readouterr().out)

	assert payload['execution'] == 'dry-run'
	assert payload['model_order'] == list(FIVE_WAY_MODEL_IDS)
	assert before == after


def _write_universe(tmp_path: Path, *, embeddings: bool) -> dict[str, object]:
	raw = _config_mapping(tmp_path)
	checkpoint_paths = {
		model_id: Path(raw['models'][model_id]['checkpoint'])
		for model_id in FIVE_WAY_MODEL_IDS
	}
	parent_root = (tmp_path / 'artifacts' / 'parents').resolve()
	mae_parent = parent_root / 'mae100.pt'
	local_parent = parent_root / 'local_bt100.pt'
	mae_parent_payload = _stage1_payload(local=False)
	local_parent_payload = _stage1_payload(local=True)
	_save(mae_parent, mae_parent_payload)
	_save(local_parent, local_parent_payload)
	mae_parent_sha = file_sha256(mae_parent)
	local_parent_sha = file_sha256(local_parent)
	pseudo_paths: dict[str, Path] = {}

	for model_id in FIVE_WAY_MODEL_IDS:
		path = checkpoint_paths[model_id]
		if model_id == 'random':
			payload = {
				'config': deepcopy(mae_parent_payload['config']),
				'epoch': 0,
				'global_step': 0,
				'metadata': {
					'random_encoder_baseline': True,
					'pretrained_weights_loaded': False,
					'seed': 42,
					'reference_checkpoint': str(mae_parent),
					'reference_checkpoint_sha256': mae_parent_sha,
				},
				'training_state': {
					'stage': 'create_random_mae_checkpoint',
					'checkpoint_kind': 'random_init',
				},
			}
		elif model_id in {'mae', 'local_barlow_twins'}:
			local = model_id == 'local_barlow_twins'
			parent = local_parent if local else mae_parent
			parent_sha = local_parent_sha if local else mae_parent_sha
			config = deepcopy(
				local_parent_payload['config']
				if local
				else mae_parent_payload['config']
			)
			config['train'] = {
				'epochs': 25,
				'samples_per_epoch': 10_000,
				'batch_size': 4,
			}
			config['continuation'] = {
				'init_checkpoint': str(parent),
				'unfreeze_top_blocks': 1,
			}
			payload = {
				'config': config,
				'epoch': 25,
				'global_step': FIVE_WAY_STAGE2_GLOBAL_STEPS,
				'continuation_lineage': {
					'schema_version': 1,
					'init_checkpoint': str(parent),
					'init_checkpoint_sha256': parent_sha,
					'resume_count': 0,
				},
				'training_state': (
					{
						'stage': 'barlow_twins_training',
						'completed_epoch': True,
					}
					if local
					else {'stage': 'train_amp_mae', 'checkpoint_kind': 'epoch'}
				),
			}
		else:
			local = model_id == 'local_barlow_twins_hmm_k6'
			parent = local_parent if local else mae_parent
			parent_sha = local_parent_sha if local else mae_parent_sha
			parent_payload = local_parent_payload if local else mae_parent_payload
			pseudo_dir = (tmp_path / 'artifacts' / 'pseudo' / model_id).resolve()
			pseudo_path = pseudo_dir / 'volve_st10010.pseudo_target_metadata.json'
			labels_path = pseudo_dir / 'volve_st10010.pseudo_labels.npy'
			confidence_path = pseudo_dir / 'volve_st10010.pseudo_confidence.npy'
			valid_tokens_path = pseudo_dir / 'volve_st10010.valid_tokens.npy'
			stage1_embedding_metadata = (
				tmp_path
				/ 'artifacts'
				/ 'pseudo_sources'
				/ model_id
				/ 'volve_st10010.embedding_metadata.json'
			).resolve()
			_write_json(
				stage1_embedding_metadata,
				{
					'checkpoint_path': str(parent),
					'checkpoint_sha256': parent_sha,
				},
			)
			cluster_metadata = stage1_embedding_metadata.with_name(
				'volve_st10010.cluster_label_metadata.json'
			)
			_write_json(
				cluster_metadata,
				{
					'embedding_input': {
						'metadata_path': str(stage1_embedding_metadata),
						'metadata_sha256': file_sha256(stage1_embedding_metadata),
					}
				},
			)
			_write_json(
				pseudo_path,
				{
					'artifact_type': 'strat_hmm_pseudo_target',
					'schema_version': 2,
					'k': 6,
					'survey_id': 'volve_st10010',
					'source': {
						'source_metadata_path': str(cluster_metadata),
						'source_metadata_sha256': file_sha256(cluster_metadata),
					},
				},
			)
			pseudo_dir.mkdir(parents=True, exist_ok=True)
			np.save(labels_path, np.zeros((1, 1, 1), dtype=np.int16))
			np.save(confidence_path, np.ones((1, 1, 1), dtype=np.float32))
			np.save(valid_tokens_path, np.ones((1, 1, 1), dtype=np.bool_))
			pseudo_paths[model_id] = pseudo_path
			stratigraphy = {
				'head': {'num_prototypes': 6},
				'teacher': {'checkpoint': str(parent)},
				'student': {
					'init_checkpoint': str(parent),
					'unfreeze_top_blocks': 1,
				},
				'pseudo_targets': {'input_dir': str(pseudo_dir), 'k': 6},
				'train': {
					'epochs': 25,
					'samples_per_epoch': 10_000,
					'batch_size': 4,
				},
			}
			payload = {
				'config': deepcopy(parent_payload['config']),
				'stratigraphy_config': stratigraphy,
				'epoch': 25,
				'global_step': FIVE_WAY_STAGE2_GLOBAL_STEPS,
				'training_state': {
					'stage': 'train_strat_hmm_pretext',
					'checkpoint_kind': 'epoch',
				},
				'control_identity': {
					'input_identities': {
						'teacher_checkpoint': {
							'path': str(parent),
							'sha256': parent_sha,
						},
						'student_init_checkpoint': {
							'path': str(parent),
							'sha256': parent_sha,
						},
						'pseudo_targets': [
							{
								'survey_id': 'volve_st10010',
								'labels': _file_identity(labels_path),
								'confidence': _file_identity(confidence_path),
								'valid_tokens': _file_identity(valid_tokens_path),
								'metadata': _file_identity(pseudo_path),
								'boundary_weight_present': False,
							}
						],
					}
				},
			}
		_save(path, payload)

	config = volve_horizon_five_way_config_from_mapping(raw)
	universe: dict[str, object] = {
		'config': config,
		'checkpoints': checkpoint_paths,
		'parents': {'mae': mae_parent, 'local': local_parent},
		'pseudo_metadata': pseudo_paths,
	}
	if embeddings:
		universe.update(_write_embedding_suite(config, tmp_path))
	return universe


def _stage1_payload(*, local: bool) -> dict[str, object]:
	config: dict[str, object] = {
		'stage': 'barlow_twins_training' if local else 'train_amp_mae',
		'model': {
			'in_channels': 1,
			'patch_size': [8, 8, 8],
			'encoder_dim': 384,
			'encoder_depth': 8,
			'encoder_heads': 6,
		},
		'train': {
			'epochs': 100,
			'samples_per_epoch': 10_000,
			'batch_size': 16 if local else 4,
		},
	}
	if local:
		config['barlow_twins'] = {
			'method': LOCAL_BARLOW_TWINS_METHOD,
			'local_pairs_per_crop': 128,
		}
		config['augmentations'] = {
			'policy': 'horizontal_flip_gaussian_noise_v1',
			'horizontal_flip_probability': 0.5,
			'gaussian_noise_std': 0.05,
		}
	return {
		'config': config,
		'epoch': 100,
		'global_step': (
			FIVE_WAY_LOCAL_BT_STAGE1_GLOBAL_STEPS
			if local
			else FIVE_WAY_STAGE1_GLOBAL_STEPS
		),
		'training_state': {
			'stage': 'barlow_twins_training' if local else 'train_amp_mae',
			'checkpoint_kind': 'epoch',
		},
	}


def _write_embedding_suite(
	config: object,
	tmp_path: Path,
) -> dict[str, object]:
	volume_shape = (16, 16, 800)
	token_grid = (2, 2, 100)
	valid_mask_path = (tmp_path / 'public' / 'valid_trace_mask.npy').resolve()
	valid_mask_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(valid_mask_path, np.ones(volume_shape[:2], dtype=np.bool_))
	amplitude_path = (tmp_path / 'public' / 'amplitude.npy').resolve()
	np.save(amplitude_path, np.zeros((1,), dtype=np.float32))
	normalization_path = (tmp_path / 'artifacts' / 'normalization.json').resolve()
	_write_json(normalization_path, {'mean': 0.0, 'std': 1.0})
	canonical_identity = {
		'survey_id': 'volve_st10010',
		'shape_xyz': list(volume_shape),
		'canonical_amplitude_sha256': file_sha256(amplitude_path),
		'valid_trace_mask_sha256': file_sha256(valid_mask_path),
		'inline_values_sha256': '1' * 64,
		'crossline_values_sha256': '2' * 64,
		'time_axis_sha256': '3' * 64,
		'canonical_normalization_stats_sha256': file_sha256(normalization_path),
	}
	canonical_path = cast(
		'Path',
		config.canonical_input_metadata,
	)
	_write_json(
		canonical_path,
		{
			'artifact_type': 'volve_canonical_input_registration',
			'status': 'PASS',
			'scientific_identity': canonical_identity,
			'scientific_identity_sha256': _json_sha256(canonical_identity),
			'provenance': {
				'amplitude': {'path': str(amplitude_path)},
				'public_inputs': {'valid_trace_mask.npy': str(valid_mask_path)},
			},
			'outputs': {'normalization_stats': str(normalization_path)},
		},
	)
	common = {
		'survey_id': 'volve_st10010',
		'source_amplitude_path': str(amplitude_path),
		'source_valid_mask_path': str(valid_mask_path),
		'volume_shape_xyz': list(volume_shape),
		'model_geometry': {
			'in_channels': 1,
			'patch_size': [8, 8, 8],
			'encoder_dim': 384,
			'encoder_depth': 8,
			'encoder_heads': 6,
		},
		'patch_size': [8, 8, 8],
		'token_grid_shape': list(token_grid),
		'window_size': [128, 128, 128],
		'overlap': [64, 64, 64],
		'output_dtype': 'float16',
		'precision': {
			'amp_requested': True,
			'amp_dtype_requested': 'auto',
			'resolved_dtype': 'float16',
			'amp_enabled': True,
		},
		'min_token_valid_fraction': 1.0,
		'normalization_stats_path': str(normalization_path),
		'normalized_clip_abs': 8.0,
		'amplitude_agc': {'enabled': True, 'mode': 'trace_rms_z'},
		'finite_check_mode': 'strict',
		'preprocessing': {'normalized_clip_abs': 8.0},
		'preprocessing_cache': {'requested_mode': 'off', 'effective_mode': 'off'},
		'zero_mask': {'enabled': True},
	}
	metadata_paths: dict[str, Path] = {}
	valid_tokens = np.ones(token_grid, dtype=np.bool_)
	for model_id in FIVE_WAY_MODEL_IDS:
		model = config.model_by_id(model_id)
		paths = output_paths(model.embeddings_dir, config.survey_id)
		paths.embeddings.parent.mkdir(parents=True, exist_ok=True)
		np.save(paths.embeddings, np.zeros((*token_grid, 384), dtype=np.float16))
		np.save(paths.valid_tokens, valid_tokens)
		metadata = {
			**common,
			'checkpoint_path': str(model.checkpoint),
			'checkpoint_sha256': file_sha256(model.checkpoint),
			'pretraining_objective': (
				{
					'method': LOCAL_BARLOW_TWINS_METHOD,
					'local_pairs_per_crop': 128,
				}
				if model_id.startswith('local_barlow_twins')
				else {'reconstruction': 'mse'}
			),
		}
		if model_id.startswith('local_barlow_twins'):
			metadata['pretraining_method'] = LOCAL_BARLOW_TWINS_METHOD
		if model_id.endswith('_hmm_k6'):
			metadata['stratigraphy_pretext'] = {
				'method': 'strat_hmm_pretext',
				'base_objective': (
					LOCAL_BARLOW_TWINS_METHOD
					if model_id.startswith('local_barlow_twins')
					else 'amp_mae3d'
				),
				'head_num_prototypes': 6,
				'unfreeze_top_blocks': 1,
				'pseudo_target_input_dir': str(
					tmp_path / 'artifacts' / 'pseudo' / model_id
				),
			}
		_write_json(paths.metadata, metadata)
		metadata_paths[model_id] = paths.metadata
	return {'embedding_metadata': metadata_paths}


def _config_mapping(tmp_path: Path) -> dict[str, object]:
	artifact_root = (tmp_path / 'artifacts').resolve()
	return {
		'paths': {
			'artifact_root': str(artifact_root),
			'volve_root': str((tmp_path / 'public').resolve()),
		},
		'dataset': {'survey_id': 'volve_st10010'},
		'inputs': {
			'canonical_input_metadata': str(
				artifact_root / 'canonical_input_metadata.json'
			)
		},
		'models': {
			model_id: {
				'checkpoint': str(
					artifact_root / 'checkpoints' / model_id / 'latest.pt'
				),
				'embeddings_dir': str(artifact_root / 'embeddings' / model_id),
			}
			for model_id in FIVE_WAY_MODEL_IDS
		},
		'outputs': {
			'runs_root': str(artifact_root / 'runs'),
			'summary_root': str(artifact_root / 'summary'),
		},
		'decoder': {
			'embedding_dim': 384,
			'class_count': 5,
			'hidden_channels': [128, 64, 32],
			'upsample_factors': [[2, 2, 2]] * 3,
			'upsample_mode': 'nearest',
			'normalization': 'voxelwise_layer_norm',
		},
		'tiles': {
			'patch_size': [8, 8, 8],
			'core_size_tokens': [8, 8, 27],
			'context_halo_tokens': [1, 1, 0],
			'window_start': 552,
			'window_stop': 768,
			'min_token_valid_fraction': 1.0,
		},
		'train': {
			'epochs': 50,
			'batch_size': 1,
			'learning_rate': 1.0e-3,
			'weight_decay': 1.0e-4,
			'sampling_mode': 'all_tiles_once',
			'seed': 42000,
			'amp': True,
			'gradient_clip_norm': 1.0,
		},
	}


def _save(path: Path, payload: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(payload, path)


def _load(path: Path) -> dict[str, object]:
	return torch.load(path, map_location='cpu', weights_only=False)


def _write_json(path: Path, payload: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _json_sha256(payload: object) -> str:
	return hashlib.sha256(
		json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
	).hexdigest()


def _file_identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}
