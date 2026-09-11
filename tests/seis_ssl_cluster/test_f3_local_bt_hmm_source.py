from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.local_bt_hmm_source import (
	HMM_MODEL_TAG,
	validate_f3_local_bt_candidate_source,
	validate_f3_local_bt_pseudo_target_science,
)
from seis_ssl_cluster.stratigraphy import write_pseudo_target
from seis_ssl_cluster.training.checkpoint import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
	REPO_ROOT
	/ 'experiments/f3/facies_benchmark_v2/123_local_bt_noise_rotation_search_v1'
)
BASE_CONFIG = EXPERIMENT_ROOT / '10_pretraining/rot90_asym_g060_3ep.yaml'
HMM_CONFIG = (
	EXPERIMENT_ROOT / '50_hmm_pretraining/rot90_asym_g060_3ep/hmm/k6/02_full_25ep.yaml'
)
SURVEY_ID = 'f3_facies_benchmark'
_BLOCK_SUFFIXES = (
	'norm1.weight',
	'norm1.bias',
	'attention.in_proj_weight',
	'attention.in_proj_bias',
	'attention.out_proj.weight',
	'attention.out_proj.bias',
	'norm2.weight',
	'norm2.bias',
	'mlp.0.weight',
	'mlp.0.bias',
	'mlp.3.weight',
	'mlp.3.bias',
)


@dataclass(frozen=True)
class _SourceFixture:
	base_config: dict[str, object]
	hmm_config: dict[str, object]
	control_checkpoint: Path
	candidate_checkpoint: Path
	control_metadata: Path
	candidate_metadata: Path
	pseudo_metadata: Path
	source_labels: Path
	cluster_label_metadata: Path
	clustering_metadata: Path
	target_embedding_metadata: Path


@pytest.fixture
def source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _SourceFixture:
	artifact_root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(artifact_root))
	base_config = resolve_barlow_twins_training_config(load_config(BASE_CONFIG))
	control_checkpoint = (
		Path(str(base_config['paths']['output_root'])) / 'latest.pt'  # type: ignore[index]
	)
	control_checkpoint.parent.mkdir(parents=True)
	base_state = _base_model_state()
	torch.save(
		{
			'config': base_config,
			'epoch': 3,
			'global_step': 1_875,
			'pretraining_method': 'local_barlow_twins_3d',
			'checkpoint_kind': 'barlow_twins_pretraining',
			'model_state_dict': base_state,
			'projector_state_dict': {'projection.weight': torch.ones(1)},
			'training_state': {
				'schema_version': 1,
				'stage': 'barlow_twins_training',
				'resume_boundary': 'epoch',
				'dataset_epoch': 2,
				'completed_epoch': True,
			},
		},
		control_checkpoint,
	)
	pseudo_root = (
		artifact_root / 'pseudo_targets/f3/facies_benchmark_v1/'
		'local_bt_noise_rotation_search_v1/rot90_asym_g060_3ep'
	)
	target_embedding_root = (
		artifact_root
		/ 'embeddings/f3/facies_benchmark_v1/local_bt_noise_rotation_search_v1/'
		'hmm_targets/rot90_asym_g060_3ep/overlap_x64'
	)
	target_embedding_root.mkdir(parents=True)
	target_embeddings = target_embedding_root / f'{SURVEY_ID}.embeddings.npy'
	target_valid_tokens = target_embedding_root / f'{SURVEY_ID}.valid_tokens.npy'
	np.save(target_embeddings, np.zeros((2, 2, 2, 384), dtype=np.float16))
	np.save(target_valid_tokens, np.ones((2, 2, 2), dtype=np.bool_))
	target_embedding_metadata = (
		target_embedding_root / f'{SURVEY_ID}.embedding_metadata.json'
	)
	_write_metadata(
		target_embedding_metadata,
		checkpoint=control_checkpoint,
		base_config=base_config,
		include_positive_window=True,
		pretext=None,
	)
	embedding_input = {
		'survey_id': SURVEY_ID,
		'embeddings_path': str(target_embeddings),
		'valid_tokens_path': str(target_valid_tokens),
		'metadata_path': str(target_embedding_metadata),
		'metadata_sha256': file_sha256(target_embedding_metadata),
	}
	clustering_root = (
		artifact_root
		/ 'clustering/f3/facies_benchmark_v1/local_bt_noise_rotation_search_v1/'
		'hmm_targets/rot90_asym_g060_3ep/k6'
	)
	label_root = clustering_root / 'labels/k6'
	label_root.mkdir(parents=True)
	source_labels = label_root / f'{SURVEY_ID}.cluster_labels_token.npy'
	labels = np.zeros((2, 2, 2), dtype=np.int32)
	np.save(source_labels, labels)
	cluster_label_metadata = label_root / f'{SURVEY_ID}.cluster_label_metadata.json'
	label_metadata = _cluster_recipe_metadata(
		embedding_input=embedding_input,
		checkpoint=control_checkpoint,
	)
	label_metadata.update(
		{
			'embedding_input': deepcopy(embedding_input),
			'embedding_inputs': [deepcopy(embedding_input)],
			'label_path': str(source_labels),
			'survey_id': SURVEY_ID,
		}
	)
	_write_json(cluster_label_metadata, label_metadata)
	clustering_metadata = clustering_root / 'models/k6/clustering_metadata.json'
	clustering_metadata.parent.mkdir(parents=True)
	model_metadata = _cluster_recipe_metadata(
		embedding_input=embedding_input,
		checkpoint=control_checkpoint,
	)
	model_metadata.update(
		{
			'embedding_inputs': [deepcopy(embedding_input)],
			'stratigraphic_hmm': _hmm_recipe_metadata(),
			'surveys': [
				{
					'survey_id': SURVEY_ID,
					'label_path': str(source_labels),
					'label_metadata_path': str(cluster_label_metadata),
				}
			],
		}
	)
	_write_json(clustering_metadata, model_metadata)
	pseudo_paths = write_pseudo_target(
		pseudo_root,
		k=6,
		survey_id=SURVEY_ID,
		labels=labels,
		confidence=np.ones((2, 2, 2), dtype=np.float32),
		valid_tokens=np.ones((2, 2, 2), dtype=np.bool_),
		boundary_weight=np.ones((2, 2, 2), dtype=np.float32),
		metadata={
			'source_method': 'stratigraphic_hmm_kmeans',
			'export_confidence': 1.0,
			'boundary_weighting': {
				'method': 'transition_distance_exponential',
				'alpha': 0.0,
				'tau': 1.0,
				'adjacent_transition_distance': 0,
				'invalid_gap_crossing': False,
			},
			'source_clustering_output_dir': str(clustering_root),
			'source_label_path': str(source_labels),
			'source_label_sha256': file_sha256(source_labels),
			'source_metadata_path': str(cluster_label_metadata),
			'source_metadata_sha256': file_sha256(cluster_label_metadata),
		},
	)
	hmm_config = resolve_strat_hmm_pretext_config(load_config(HMM_CONFIG))
	student_names = tuple(f'encoder.layers.7.{name}' for name in _BLOCK_SUFFIXES)
	control_identity = {
		'schema_version': 1,
		'model_tag': HMM_MODEL_TAG,
		'scientific_identity': {},
		'runtime_identity': {
			'git_commit': None,
			'git_status_short': '',
			'git_diff_sha256': hashlib.sha256(b'').hexdigest(),
			'finite_check_mode': 'strict',
		},
		'resolved_training_config_sha256': _canonical_sha256(hmm_config),
		'input_identities': {
			'teacher_checkpoint': _file_identity(control_checkpoint),
			'student_init_checkpoint': _file_identity(control_checkpoint),
			'pseudo_targets': [
				{
					'survey_id': SURVEY_ID,
					'labels': _file_identity(pseudo_paths.labels),
					'confidence': _file_identity(pseudo_paths.confidence),
					'valid_tokens': _file_identity(pseudo_paths.valid_tokens),
					'metadata': _file_identity(pseudo_paths.metadata),
					'boundary_weight_present': True,
					'boundary_weight': _file_identity(pseudo_paths.boundary_weight),
				}
			],
		},
		'initial_parameter_sha256': {
			'student_trainable': {
				name: _parameter_sha256(name, base_state[name])
				for name in student_names
			},
			'prototype_head': {
				'prototypes': hashlib.sha256(b'prototypes').hexdigest(),
				'projection.weight': hashlib.sha256(b'projection.weight').hexdigest(),
				'projection.bias': hashlib.sha256(b'projection.bias').hexdigest(),
			},
		},
		'initial_state_sha256': {
			'student': _state_dict_sha256(base_state),
			'head': hashlib.sha256(b'head').hexdigest(),
		},
	}
	candidate_checkpoint = (
		Path(str(hmm_config['paths']['output_root'])) / 'latest.pt'  # type: ignore[index]
	)
	candidate_checkpoint.parent.mkdir(parents=True)
	candidate_base_config = deepcopy(base_config)
	candidate_base_config['paths']['output_root'] = str(  # type: ignore[index]
		hmm_config['paths']['output_root']  # type: ignore[index]
	)
	torch.save(
		{
			'config': candidate_base_config,
			'stratigraphy_config': hmm_config,
			'control_identity': control_identity,
			'epoch': 25,
			'global_step': 15_625,
			'model_state_dict': base_state,
			'stratigraphy_state_dict': {'prototypes': torch.ones(6, 128)},
			'trainability_summary': {
				'trainable_parameter_count': 12,
				'frozen_parameter_count': 1,
				'trainable_names': list(student_names),
			},
			'training_state': {
				'schema_version': 1,
				'stage': 'train_strat_hmm_pretext',
				'checkpoint_kind': 'epoch',
				'batch_index': None,
			},
		},
		candidate_checkpoint,
	)
	control_metadata = tmp_path / 'control.embedding_metadata.json'
	candidate_metadata = tmp_path / 'candidate.embedding_metadata.json'
	_write_metadata(
		control_metadata,
		checkpoint=control_checkpoint,
		base_config=base_config,
		include_positive_window=False,
		pretext=None,
	)
	_write_metadata(
		candidate_metadata,
		checkpoint=candidate_checkpoint,
		base_config=base_config,
		include_positive_window=True,
		pretext={
			'method': 'strat_hmm_pretext',
			'base_objective': 'local_barlow_twins_3d',
			'head_num_prototypes': 6,
			'unfreeze_top_blocks': 1,
			'distillation_weight': 0.2,
			'pseudo_target_input_dir': str(pseudo_root),
			'model_tag': HMM_MODEL_TAG,
			'control_identity_sha256': _canonical_sha256(control_identity),
		},
	)
	return _SourceFixture(
		base_config=base_config,
		hmm_config=hmm_config,
		control_checkpoint=control_checkpoint,
		candidate_checkpoint=candidate_checkpoint,
		control_metadata=control_metadata,
		candidate_metadata=candidate_metadata,
		pseudo_metadata=pseudo_paths.metadata,
		source_labels=source_labels,
		cluster_label_metadata=cluster_label_metadata,
		clustering_metadata=clustering_metadata,
		target_embedding_metadata=target_embedding_metadata,
	)


def test_completed_control_and_hmm_candidate_pass_strict_source_audit(
	source_fixture: _SourceFixture,
) -> None:
	control = _validate(source_fixture, role='control')
	candidate = _validate(source_fixture, role='candidate')

	assert control['checkpoint']['epoch'] == 3  # type: ignore[index]
	assert control['role'] == 'control'
	assert candidate['checkpoint']['global_step'] == 15_625  # type: ignore[index]
	assert candidate['model_tag'] == HMM_MODEL_TAG
	assert candidate['stratigraphy_training_config']['resolved_sha256'] == (  # type: ignore[index]
		_canonical_sha256(source_fixture.hmm_config)
	)
	science = candidate['pseudo_target_science'][0]  # type: ignore[index]
	assert science['pseudo_target_metadata'] == _file_identity(
		source_fixture.pseudo_metadata
	)
	assert science['cluster_label_metadata'] == _file_identity(
		source_fixture.cluster_label_metadata
	)
	assert science['clustering_metadata'] == _file_identity(
		source_fixture.clustering_metadata
	)
	assert science['target_embedding_metadata'] == _file_identity(
		source_fixture.target_embedding_metadata
	)

	standalone = validate_f3_local_bt_pseudo_target_science(
		base_training_config=BASE_CONFIG,
		stratigraphy_training_config=HMM_CONFIG,
	)
	assert standalone == candidate['pseudo_target_science']


def test_candidate_checkpoint_base_config_uses_hmm_output_root(
	source_fixture: _SourceFixture,
) -> None:
	payload = load_checkpoint(source_fixture.candidate_checkpoint, map_location='cpu')
	assert (
		payload['config']['paths']['output_root']
		== (  # type: ignore[index]
			source_fixture.hmm_config['paths']['output_root']  # type: ignore[index]
		)
	)
	_validate(source_fixture, role='candidate')


@pytest.mark.parametrize(
	('keys', 'value'),
	[
		(('paths', 'output_root'), '/not-the-hmm-output'),
		(('augmentations', 'gaussian_noise_std'), 0.5),
	],
)
def test_candidate_rejects_checkpoint_base_config_drift(
	source_fixture: _SourceFixture,
	keys: tuple[str, ...],
	value: object,
) -> None:
	payload = load_checkpoint(source_fixture.candidate_checkpoint, map_location='cpu')
	_set_nested(payload['config'], keys, value)  # type: ignore[arg-type]
	_rewrite_checkpoint(
		source_fixture.candidate_checkpoint,
		source_fixture.candidate_metadata,
		payload,
	)
	with pytest.raises(ValueError, match='checkpoint output root'):
		_validate(source_fixture, role='candidate')


def test_control_rejects_checkpoint_base_config_output_root_drift(
	source_fixture: _SourceFixture,
) -> None:
	payload = load_checkpoint(source_fixture.control_checkpoint, map_location='cpu')
	payload['config']['paths']['output_root'] = '/not-the-control-output'
	_rewrite_checkpoint(
		source_fixture.control_checkpoint,
		source_fixture.control_metadata,
		payload,
	)
	with pytest.raises(ValueError, match='checkpoint output root'):
		_validate(source_fixture, role='control')


@pytest.mark.parametrize(
	('role', 'epoch', 'global_step', 'stratigraphy'),
	[
		('control', 2, 1_250, None),
		('candidate', 24, 15_000, HMM_CONFIG),
		('control', 3, 1_875, HMM_CONFIG),
		('candidate', 25, 15_625, None),
	],
)
def test_declared_expectation_cannot_weaken_fixed_comparison(
	source_fixture: _SourceFixture,
	role: str,
	epoch: int,
	global_step: int,
	stratigraphy: Path | None,
) -> None:
	checkpoint = (
		source_fixture.control_checkpoint
		if role == 'control'
		else source_fixture.candidate_checkpoint
	)
	metadata = (
		source_fixture.control_metadata
		if role == 'control'
		else source_fixture.candidate_metadata
	)
	with pytest.raises(ValueError, match=r'expectation|must not declare|requires'):
		validate_f3_local_bt_candidate_source(
			checkpoint,
			metadata,
			role=role,  # type: ignore[arg-type]
			base_training_config=BASE_CONFIG,
			stratigraphy_training_config=stratigraphy,
			epoch=epoch,
			global_step=global_step,
		)


def test_control_rejects_incomplete_epoch_and_stratigraphy_lineage(
	source_fixture: _SourceFixture,
) -> None:
	payload = load_checkpoint(source_fixture.control_checkpoint, map_location='cpu')
	payload['training_state']['completed_epoch'] = False
	_rewrite_checkpoint(
		source_fixture.control_checkpoint, source_fixture.control_metadata, payload
	)
	with pytest.raises(ValueError, match='completed LocalBT epoch'):
		_validate(source_fixture, role='control')

	payload['training_state']['completed_epoch'] = True
	payload['stratigraphy_config'] = {}
	_rewrite_checkpoint(
		source_fixture.control_checkpoint, source_fixture.control_metadata, payload
	)
	with pytest.raises(ValueError, match='must not contain stratigraphy lineage'):
		_validate(source_fixture, role='control')


def test_candidate_rejects_incomplete_epoch_and_hmm_config_drift(
	source_fixture: _SourceFixture,
) -> None:
	payload = load_checkpoint(source_fixture.candidate_checkpoint, map_location='cpu')
	payload['training_state']['checkpoint_kind'] = 'step'
	_rewrite_checkpoint(
		source_fixture.candidate_checkpoint,
		source_fixture.candidate_metadata,
		payload,
	)
	with pytest.raises(ValueError, match='completed HMM epoch'):
		_validate(source_fixture, role='candidate')

	payload['training_state']['checkpoint_kind'] = 'epoch'
	payload['stratigraphy_config']['loss']['distillation_weight'] = 0.1
	_rewrite_checkpoint(
		source_fixture.candidate_checkpoint,
		source_fixture.candidate_metadata,
		payload,
	)
	with pytest.raises(ValueError, match='does not exactly match the resolved HMM'):
		_validate(source_fixture, role='candidate')


@pytest.mark.parametrize(
	('mutation', 'message'),
	[
		('resolved_config', 'resolved training config SHA'),
		('teacher_sha', 'teacher checkpoint SHA'),
		('initial_student_hash', 'initial student parameter'),
		('head_hash_set', 'prototype head parameter hash set'),
	],
)
def test_candidate_rejects_control_identity_drift(
	source_fixture: _SourceFixture, mutation: str, message: str
) -> None:
	payload = load_checkpoint(source_fixture.candidate_checkpoint, map_location='cpu')
	identity = payload['control_identity']
	if mutation == 'resolved_config':
		identity['resolved_training_config_sha256'] = hashlib.sha256(
			b'wrong'
		).hexdigest()
	elif mutation == 'teacher_sha':
		identity['input_identities']['teacher_checkpoint']['sha256'] = hashlib.sha256(
			b'wrong'
		).hexdigest()
	elif mutation == 'initial_student_hash':
		name = next(iter(identity['initial_parameter_sha256']['student_trainable']))
		identity['initial_parameter_sha256']['student_trainable'][name] = (
			hashlib.sha256(b'wrong').hexdigest()
		)
	else:
		identity['initial_parameter_sha256']['prototype_head'].pop('prototypes')
	_rewrite_checkpoint(
		source_fixture.candidate_checkpoint,
		source_fixture.candidate_metadata,
		payload,
	)
	with pytest.raises(ValueError, match=message):
		_validate(source_fixture, role='candidate')


def test_candidate_embedding_requires_positive_window_and_exact_hmm_identity(
	source_fixture: _SourceFixture,
) -> None:
	metadata = json.loads(source_fixture.candidate_metadata.read_text())
	metadata['pretraining_objective'].pop('positive_window_tokens')
	_write_json(source_fixture.candidate_metadata, metadata)
	with pytest.raises(ValueError, match='positive-window contract'):
		_validate(source_fixture, role='candidate')

	metadata['pretraining_objective']['positive_window_tokens'] = [2, 2, 1]
	metadata['stratigraphy_pretext']['distillation_weight'] = 0.1
	_write_json(source_fixture.candidate_metadata, metadata)
	with pytest.raises(ValueError, match='stratigraphy_pretext identity'):
		_validate(source_fixture, role='candidate')


def test_control_embedding_rejects_hmm_identity(
	source_fixture: _SourceFixture,
) -> None:
	metadata = json.loads(source_fixture.control_metadata.read_text())
	metadata['stratigraphy_pretext'] = {'method': 'strat_hmm_pretext'}
	_write_json(source_fixture.control_metadata, metadata)
	with pytest.raises(ValueError, match='must not declare stratigraphy_pretext'):
		_validate(source_fixture, role='control')


@pytest.mark.parametrize(
	('keys', 'value', 'message'),
	[
		(('schema_version',), 1, 'schema_version'),
		(('source', 'export_confidence'), 0.9, 'export_confidence'),
		(('source', 'boundary_weighting', 'alpha'), 0.1, 'alpha'),
		(('source', 'boundary_weighting', 'tau'), 2.0, 'tau'),
		(
			('source', 'boundary_weighting', 'method'),
			'linear',
			'boundary weighting.method',
		),
	],
)
def test_candidate_rejects_pseudo_target_science_drift(
	source_fixture: _SourceFixture,
	keys: tuple[str, ...],
	value: object,
	message: str,
) -> None:
	metadata = json.loads(source_fixture.pseudo_metadata.read_text())
	_set_nested(metadata, keys, value)
	_rewrite_pseudo_metadata(source_fixture, metadata)
	with pytest.raises(ValueError, match=message):
		_validate(source_fixture, role='candidate')


@pytest.mark.parametrize(
	('keys', 'value', 'message'),
	[
		(('method',), 'kmeans', 'method'),
		(('k',), 5, r'\.k'),
		(('normalization',), 'none', 'normalization'),
		(('random_seed',), 43, 'random_seed'),
		(('emission_source',), 'position', 'emission_source'),
		(('residualization', 'enabled'), False, 'residualization.enabled'),
		(
			('residualization', 'mode'),
			'global',
			'residualization.mode',
		),
		(
			('residualization', 'group_by'),
			'none',
			'residualization.group_by',
		),
		(
			('residualization', 'add_global_mean_back'),
			False,
			'residualization.add_global_mean_back',
		),
		(
			('residualization', 'min_group_count'),
			31,
			'residualization.min_group_count',
		),
		(('pca', 'n_components'), 32, 'pca.n_components'),
		(('pca', 'whiten'), True, 'pca.whiten'),
	],
)
def test_candidate_rejects_cluster_label_recipe_drift(
	source_fixture: _SourceFixture,
	keys: tuple[str, ...],
	value: object,
	message: str,
) -> None:
	metadata = json.loads(source_fixture.cluster_label_metadata.read_text())
	_set_nested(metadata, keys, value)
	_rewrite_cluster_label_metadata(source_fixture, metadata)
	with pytest.raises(ValueError, match=message):
		_validate(source_fixture, role='candidate')


@pytest.mark.parametrize(
	('keys', 'value', 'message'),
	[
		(('iterations',), 9, 'iterations'),
		(('z_axis',), 1, 'z_axis'),
		(('z_direction',), 'decreasing_downward', 'z_direction'),
		(('edge_margin_tokens',), [7, 8, 0], 'edge_margin_tokens'),
		(('transition', 'same_cost'), 0.04, 'transition.same_cost'),
		(('transition', 'advance_cost'), 0.1, 'transition.advance_cost'),
		(('transition', 'jump_cost'), 2.0, 'transition.jump_cost'),
		(('transition', 'reverse_cost'), 10.0, 'transition.reverse_cost'),
		(('transition', 'forbid_reverse'), False, 'transition.forbid_reverse'),
		(('transition', 'max_jump'), 2, 'transition.max_jump'),
		(
			('path_prior', 'initial_state', 'weight'),
			0.1,
			'initial_state.weight',
		),
		(
			('path_prior', 'terminal_state', 'weight'),
			0.1,
			'terminal_state.weight',
		),
		(
			('path_prior', 'expected_boundaries', 'enabled'),
			True,
			'expected_boundaries.enabled',
		),
		(('init', 'order_by'), 'random', 'init.order_by'),
		(
			('update', 'empty_cluster_policy'),
			'error',
			'update.empty_cluster_policy',
		),
	],
)
def test_candidate_rejects_hmm_path_recipe_drift(
	source_fixture: _SourceFixture,
	keys: tuple[str, ...],
	value: object,
	message: str,
) -> None:
	metadata = json.loads(source_fixture.clustering_metadata.read_text())
	_set_nested(metadata['stratigraphic_hmm'], keys, value)
	_write_json(source_fixture.clustering_metadata, metadata)
	with pytest.raises(ValueError, match=message):
		_validate(source_fixture, role='candidate')


def test_candidate_rejects_source_label_and_metadata_sha_drift(
	source_fixture: _SourceFixture,
) -> None:
	metadata = json.loads(source_fixture.pseudo_metadata.read_text())
	metadata['source']['source_label_sha256'] = hashlib.sha256(b'wrong').hexdigest()
	_rewrite_pseudo_metadata(source_fixture, metadata)
	with pytest.raises(ValueError, match='source label SHA'):
		_validate(source_fixture, role='candidate')


def test_candidate_rejects_target_embedding_binding_drift(
	source_fixture: _SourceFixture,
) -> None:
	metadata = json.loads(source_fixture.target_embedding_metadata.read_text())
	metadata['checkpoint_sha256'] = hashlib.sha256(b'wrong').hexdigest()
	_rewrite_target_embedding_metadata(source_fixture, metadata)
	with pytest.raises(ValueError, match='base checkpoint SHA'):
		_validate(source_fixture, role='candidate')


@pytest.mark.parametrize('field', ['embeddings_path', 'valid_tokens_path'])
def test_candidate_rejects_embedding_input_artifact_path_drift(
	source_fixture: _SourceFixture, field: str
) -> None:
	metadata = json.loads(source_fixture.cluster_label_metadata.read_text())
	metadata['embedding_input'][field] = str(source_fixture.target_embedding_metadata)
	metadata['embedding_inputs'][0][field] = str(
		source_fixture.target_embedding_metadata
	)
	_rewrite_cluster_label_metadata(source_fixture, metadata)
	with pytest.raises(ValueError, match=field):
		_validate(source_fixture, role='candidate')


def test_candidate_rejects_pseudo_valid_mask_not_derived_from_source_labels(
	source_fixture: _SourceFixture,
) -> None:
	valid_path = source_fixture.pseudo_metadata.with_name(
		f'{SURVEY_ID}.valid_tokens.npy'
	)
	valid = np.ones((2, 2, 2), dtype=np.bool_)
	valid[0, 0, 0] = False
	np.save(valid_path, valid)
	payload = load_checkpoint(source_fixture.candidate_checkpoint, map_location='cpu')
	identities = payload['control_identity']['input_identities']['pseudo_targets']
	identities[0]['valid_tokens'] = _file_identity(valid_path)
	_rewrite_checkpoint(
		source_fixture.candidate_checkpoint,
		source_fixture.candidate_metadata,
		payload,
	)
	with pytest.raises(ValueError, match=r'valid_tokens must equal.*labels >= 0'):
		_validate(source_fixture, role='candidate')


def test_standalone_science_audit_rejects_config_drifted_current_base(
	source_fixture: _SourceFixture,
) -> None:
	payload = load_checkpoint(source_fixture.control_checkpoint, map_location='cpu')
	payload['config']['augmentations']['gaussian_noise_std'] = 0.5
	torch.save(payload, source_fixture.control_checkpoint)
	with pytest.raises(ValueError, match='resolved base training config'):
		validate_f3_local_bt_pseudo_target_science(
			base_training_config=BASE_CONFIG,
			stratigraphy_training_config=HMM_CONFIG,
		)


def _validate(fixture: _SourceFixture, *, role: str) -> dict[str, object]:
	is_control = role == 'control'
	return validate_f3_local_bt_candidate_source(
		fixture.control_checkpoint if is_control else fixture.candidate_checkpoint,
		fixture.control_metadata if is_control else fixture.candidate_metadata,
		role=role,  # type: ignore[arg-type]
		base_training_config=BASE_CONFIG,
		stratigraphy_training_config=None if is_control else HMM_CONFIG,
		epoch=3 if is_control else 25,
		global_step=1_875 if is_control else 15_625,
	)


def _base_model_state() -> dict[str, torch.Tensor]:
	return {
		f'encoder.layers.7.{suffix}': torch.tensor([float(index)])
		for index, suffix in enumerate(_BLOCK_SUFFIXES, start=1)
	}


def _cluster_recipe_metadata(
	*, embedding_input: dict[str, object], checkpoint: Path
) -> dict[str, object]:
	return {
		'method': 'stratigraphic_hmm_kmeans',
		'k': 6,
		'k_values': [6],
		'normalization': 'l2',
		'random_seed': 42,
		'emission_source': 'embedding',
		'emission_features': {
			'source': 'embedding',
			'embedding_features_used_for_emissions': True,
			'embedding_artifacts_used_for': [
				'token_grid_shape',
				'validity_masks',
				'embedding_features',
			],
		},
		'embedding_compatibility_signature': {
			'checkpoint_sha256': file_sha256(checkpoint),
		},
		'residualization': {
			'enabled': True,
			'mode': 'local_token_position',
			'group_by': 'token_phase',
			'add_global_mean_back': True,
			'min_group_count': 32,
		},
		'pca': {
			'enabled': True,
			'n_components': 64,
			'effective_n_components': 64,
			'whiten': False,
		},
		'embedding_inputs': [deepcopy(embedding_input)],
	}


def _hmm_recipe_metadata() -> dict[str, object]:
	return {
		'iterations': 10,
		'z_axis': 2,
		'z_direction': 'increasing_downward',
		'edge_margin_tokens': [8, 8, 0],
		'emission_source': 'embedding',
		'transition': {
			'same_cost': 0.03,
			'advance_cost': 0.0,
			'jump_cost': 1.0,
			'reverse_cost': 1.0e6,
			'forbid_reverse': True,
			'max_jump': 1,
		},
		'path_prior': {
			'enabled': True,
			'initial_state': {'mode': 'shallow_anchor', 'weight': 0.25},
			'terminal_state': {'mode': 'deep_anchor', 'weight': 0.25},
			'expected_boundaries': {'enabled': False},
		},
		'init': {'order_by': 'mean_z'},
		'update': {'empty_cluster_policy': 'keep_previous'},
	}


def _write_metadata(
	path: Path,
	*,
	checkpoint: Path,
	base_config: dict[str, object],
	include_positive_window: bool,
	pretext: dict[str, object] | None,
) -> None:
	model = deepcopy(base_config['model'])
	objective = deepcopy(base_config['barlow_twins'])
	objective['augmentations'] = deepcopy(base_config['augmentations'])
	if not include_positive_window:
		objective.pop('positive_window_tokens')
	data = base_config['data']
	metadata: dict[str, object] = {
		'checkpoint_path': str(checkpoint),
		'checkpoint_sha256': file_sha256(checkpoint),
		'pretraining_method': 'local_barlow_twins_3d',
		'model_geometry': model,
		'patch_size': model['patch_size'],
		'pretraining_objective': objective,
		'normalized_clip_abs': data['normalized_clip_abs'],
		'finite_check_mode': data['finite_check_mode'],
		'amplitude_agc': deepcopy(data['amplitude_agc']),
		'zero_mask': deepcopy(base_config['zero_mask']),
		'preprocessing': {
			'normalized_clip_abs': data['normalized_clip_abs'],
			'finite_check_mode': data['finite_check_mode'],
			'amplitude_agc': deepcopy(data['amplitude_agc']),
		},
	}
	if pretext is not None:
		metadata['stratigraphy_pretext'] = pretext
	_write_json(path, metadata)


def _rewrite_checkpoint(
	path: Path, metadata_path: Path, payload: dict[str, object]
) -> None:
	torch.save(payload, path)
	metadata = json.loads(metadata_path.read_text())
	metadata['checkpoint_sha256'] = file_sha256(path)
	_write_json(metadata_path, metadata)


def _rewrite_pseudo_metadata(
	fixture: _SourceFixture, metadata: dict[str, object]
) -> None:
	_write_json(fixture.pseudo_metadata, metadata)
	payload = load_checkpoint(fixture.candidate_checkpoint, map_location='cpu')
	identities = payload['control_identity']['input_identities']['pseudo_targets']
	identities[0]['metadata'] = _file_identity(fixture.pseudo_metadata)
	_rewrite_checkpoint(
		fixture.candidate_checkpoint,
		fixture.candidate_metadata,
		payload,
	)


def _rewrite_cluster_label_metadata(
	fixture: _SourceFixture, metadata: dict[str, object]
) -> None:
	_write_json(fixture.cluster_label_metadata, metadata)
	pseudo = json.loads(fixture.pseudo_metadata.read_text())
	pseudo['source']['source_metadata_sha256'] = file_sha256(
		fixture.cluster_label_metadata
	)
	_rewrite_pseudo_metadata(fixture, pseudo)


def _rewrite_target_embedding_metadata(
	fixture: _SourceFixture, metadata: dict[str, object]
) -> None:
	_write_json(fixture.target_embedding_metadata, metadata)
	digest = file_sha256(fixture.target_embedding_metadata)
	label_metadata = json.loads(fixture.cluster_label_metadata.read_text())
	label_metadata['embedding_input']['metadata_sha256'] = digest
	label_metadata['embedding_inputs'][0]['metadata_sha256'] = digest
	_rewrite_cluster_label_metadata(fixture, label_metadata)
	model_metadata = json.loads(fixture.clustering_metadata.read_text())
	model_metadata['embedding_inputs'][0]['metadata_sha256'] = digest
	_write_json(fixture.clustering_metadata, model_metadata)


def _set_nested(value: dict[str, object], keys: tuple[str, ...], item: object) -> None:
	parent = value
	for key in keys[:-1]:
		child = parent[key]
		assert isinstance(child, dict)
		parent = child
	parent[keys[-1]] = item


def _file_identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _parameter_sha256(name: str, parameter: torch.Tensor) -> str:
	value = parameter.detach().cpu().contiguous()
	digest = hashlib.sha256()
	digest.update(name.encode())
	digest.update(str(value.dtype).encode())
	digest.update(str(tuple(value.shape)).encode())
	digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
	return digest.hexdigest()


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
	digest = hashlib.sha256()
	for name in sorted(state_dict):
		value = state_dict[name].detach().cpu().contiguous()
		digest.update(name.encode())
		digest.update(str(value.dtype).encode())
		digest.update(str(tuple(value.shape)).encode())
		digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
	return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
	return hashlib.sha256(
		json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
	).hexdigest()


def _write_json(path: Path, value: object) -> None:
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
