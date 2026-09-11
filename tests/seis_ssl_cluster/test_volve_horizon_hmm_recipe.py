"""Check HMM continuation lineage without weakening the frozen Volve contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.volve.horizon_recipe_arm import (
	as_five_way_config,
	audit_volve_horizon_recipe_arm_sources,
	volve_horizon_recipe_arm_config_from_mapping,
)
from tests.seis_ssl_cluster.helpers_volve_five_way import (
	_hmm_payload,
	load_checkpoint,
	save_checkpoint,
	write_json,
)
from tests.seis_ssl_cluster.helpers_volve_recipe_arm import (
	arm_checkpoint_payload,
	baseline_checkpoint_payload,
	recipe_arm_config_mapping,
)


def _write_universe(tmp_path: Path, *, random: bool = False) -> dict[str, Any]:
	raw = recipe_arm_config_mapping(tmp_path)
	arm = raw['arm']
	assert isinstance(arm, dict)
	parent = tmp_path / 'parent.pt'
	parent_payload = (
		baseline_checkpoint_payload() if random else arm_checkpoint_payload()
	)
	if random:
		parent_payload['config'].update({'stage': 'train_amp_mae'})
		arm['recipe'] = {'method': 'random_encoder', 'seed': 42}
	save_checkpoint(parent, parent_payload)
	parent_sha = file_sha256(parent)
	clustering = tmp_path / 'clustering'
	clustering_config = tmp_path / 'clustering.json'
	pseudo_dir = tmp_path / 'targets'
	pseudo = pseudo_dir / 'k6/volve_st10010.pseudo_target_metadata.json'
	labels = pseudo_dir / 'labels.npy'
	confidence = pseudo_dir / 'confidence.npy'
	valid = pseudo_dir / 'valid.npy'
	write_json(
		pseudo,
		{
			'survey_id': 'volve_st10010',
			'k': 6,
			'source': {
				'checkpoint_path': str(parent),
				'checkpoint_sha256': parent_sha,
				'source_clustering_output_dir': str(clustering),
			},
		},
	)
	for path in (labels, confidence, valid):
		path.write_bytes(b'fixture')
	cluster_recipe = {
		'output_dir': str(clustering),
		'pca': {'enabled': True, 'n_components': 64, 'whiten': False},
		'residualization': {'enabled': True, 'mode': 'local_token_position'},
		'stratigraphic_hmm': {
			'iterations': 10,
			'path_prior': {'expected_boundaries': {'enabled': False}},
		},
	}
	write_json(
		clustering_config,
		{
			'embeddings': {'input_dir': str(tmp_path / 'embeddings')},
			'clustering': cluster_recipe,
		},
	)
	cluster_metadata = deepcopy(cluster_recipe)
	cluster_metadata['stratigraphic_hmm']['path_prior']['expected_boundaries'][
		'weight'
	] = 0.0
	cluster_metadata.update(
		{
			'method': 'stratigraphic_hmm_kmeans',
			'k': 6,
			'normalization': 'l2',
			'random_seed': 42,
			'embedding_compatibility_signature': {'checkpoint_sha256': parent_sha},
			'embedding_inputs': [_write_source_embedding(tmp_path, parent_sha)],
		}
	)
	write_json(clustering / 'models/k6/clustering_metadata.json', cluster_metadata)
	arm['hmm'] = {
		'init_checkpoint': str(parent),
		'source_embeddings_dir': str(tmp_path / 'embeddings'),
		'pseudo_targets_dir': str(pseudo_dir),
		'clustering_config': str(clustering_config),
		'distillation_weight': 0.1,
	}
	payload = _hmm_payload(
		parent=parent,
		parent_sha=parent_sha,
		parent_payload=parent_payload,
		pseudo_dir=pseudo_dir,
		pseudo_path=pseudo,
		labels_path=labels,
		confidence_path=confidence,
		valid_tokens_path=valid,
	)
	stratigraphy = payload['stratigraphy_config']
	stratigraphy['head'].update(
		{'projection_dim': 128, 'temperature': 0.1, 'normalize': True}
	)
	stratigraphy['loss'] = {
		'prototype_weight': 1.0,
		'usage_weight': 0.005,
		'entropy_floor': None,
		'distillation_weight': 0.1,
	}
	stratigraphy['train'].update(
		{
			'lr': 1.0e-5,
			'encoder_lr': 1.0e-5,
			'weight_decay': 0.05,
			'amp': False,
			'seed': 42,
			'shuffle': True,
			'grad_clip_norm': 1.0,
		}
	)
	save_checkpoint(Path(arm['checkpoint']), payload)
	config = volve_horizon_recipe_arm_config_from_mapping(raw)
	return {'raw': raw, 'config': config, 'parent': parent, 'labels': labels}


def _write_source_embedding(tmp_path: Path, parent_sha: str) -> dict[str, str]:
	directory = tmp_path / 'embeddings'
	path = directory / 'volve_st10010.embedding_metadata.json'
	write_json(
		path,
		{
			'survey_id': 'volve_st10010',
			'checkpoint_sha256': parent_sha,
			'window_size': [128, 128, 128],
			'overlap': [64, 64, 64],
			'output_dtype': 'float16',
			'min_token_valid_fraction': 1.0,
			'precision': {'amp_enabled': True, 'resolved_dtype': 'bfloat16'},
		},
	)
	return {
		'metadata_path': str(path),
		'metadata_sha256': file_sha256(path),
		'embeddings_path': str(directory / 'volve_st10010.embeddings.npy'),
		'valid_tokens_path': str(directory / 'volve_st10010.valid_tokens.npy'),
	}


@pytest.mark.parametrize('random', [False, True])
def test_hmm_recipe_audit_accepts_complete_self_target_lineage(
	tmp_path: Path, *, random: bool
) -> None:
	universe = _write_universe(tmp_path, random=random)
	config = universe['config']
	result = audit_volve_horizon_recipe_arm_sources(config, model_ids=(config.arm_id,))
	source = result['sources'][0]
	assert source['stage_2']['global_steps'] == 62500
	assert source['hmm']['distillation_weight'] == 0.1
	assert source['parent_checkpoint'] == str(universe['parent'])
	assert as_five_way_config(config).models[0].expected['stratigraphy_pretext'] is True
	assert config.train.epochs == 50
	assert config.train.seed == 42000


@pytest.mark.parametrize(
	('field', 'value', 'message'),
	[
		('epoch', 24, 'epoch must equal 25'),
		('global_step', 62499, 'global_step must equal 62500'),
		('distillation_weight', 0.2, 'distillation_weight'),
		('amp', True, 'train.amp'),
		('teacher', '/another/parent.pt', 'teacher and student parents differ'),
	],
)
def test_hmm_recipe_audit_rejects_wrong_budget_loss_or_parent(
	tmp_path: Path, field: str, value: object, message: str
) -> None:
	config = _write_universe(tmp_path)['config']
	payload = load_checkpoint(config.arm_checkpoint)
	if field in ('epoch', 'global_step'):
		payload[field] = value
	elif field == 'teacher':
		payload['stratigraphy_config']['teacher']['checkpoint'] = value
	else:
		section = 'loss' if field == 'distillation_weight' else 'train'
		payload['stratigraphy_config'][section][field] = value
	save_checkpoint(config.arm_checkpoint, payload)
	with pytest.raises(ValueError, match=message):
		audit_volve_horizon_recipe_arm_sources(config, model_ids=(config.arm_id,))


def test_hmm_recipe_audit_rejects_changed_target_bytes(tmp_path: Path) -> None:
	universe = _write_universe(tmp_path)
	universe['labels'].write_bytes(b'changed')
	config = universe['config']
	with pytest.raises(ValueError, match='SHA-256'):
		audit_volve_horizon_recipe_arm_sources(config, model_ids=(config.arm_id,))


def test_hmm_recipe_config_rejects_unknown_distillation_weight(tmp_path: Path) -> None:
	raw = _write_universe(tmp_path)['raw']
	raw['arm']['hmm']['distillation_weight'] = 0.3
	with pytest.raises(ValueError, match=r'0\.1 or 0\.2'):
		volve_horizon_recipe_arm_config_from_mapping(raw)


def test_hmm_recipe_audit_rejects_changed_source_embedding_metadata(
	tmp_path: Path,
) -> None:
	config = _write_universe(tmp_path)['config']
	path = tmp_path / 'embeddings/volve_st10010.embedding_metadata.json'
	path.write_text('{}', encoding='utf-8')
	with pytest.raises(ValueError, match='source embedding metadata SHA-256 mismatch'):
		audit_volve_horizon_recipe_arm_sources(config, model_ids=(config.arm_id,))
