from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.f3.lithology.hmm_final_source_audit import (
	audit_f3_hmm_final_source,
	audit_f3_hmm_final_sources,
)
from seis_ssl_cluster.stratigraphy import write_pseudo_target
from seis_ssl_cluster.training.strat_hmm.runtime import _strat_hmm_control_identity

EXPERIMENT = Path(
	'experiments/f3/facies_benchmark_v2/126_pretraining_comparison_completion_v1'
)
ARMS = (
	'mae100_hmm_k6_distill010',
	'local_bt_rot90_asym_g060_3ep_hmm_k6_distill010',
	'random_self_target_hmm_k6_distill020',
)


def _write_targets(root: Path, survey: str = 'f3_facies_benchmark') -> None:
	write_pseudo_target(
		root,
		k=6,
		survey_id=survey,
		labels=np.zeros((2, 2, 2), dtype=np.int32),
		confidence=np.ones((2, 2, 2), dtype=np.float32),
		valid_tokens=np.ones((2, 2, 2), dtype=np.bool_),
		boundary_weight=np.ones((2, 2, 2), dtype=np.float32),
		metadata={'fixture': 'final_hmm_source_audit'},
	)


def _make_source(
	root: Path, monkeypatch: pytest.MonkeyPatch, arm: str = ARMS[0]
) -> tuple[Path, Path, dict]:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	config_path = EXPERIMENT / '10_pretraining' / arm / '02_full_25ep.yaml'
	raw = load_config(config_path)
	parent = Path(raw['teacher']['checkpoint'])
	parent.parent.mkdir(parents=True, exist_ok=True)
	parent.write_bytes(b'parent checkpoint fixture')
	_write_targets(Path(raw['pseudo_targets']['input_dir']))
	config = resolve_strat_hmm_pretext_config(raw)
	checkpoint = Path(config['paths']['output_root']) / 'latest.pt'
	checkpoint.parent.mkdir(parents=True, exist_ok=True)
	# Use the training producer for provenance, independently of the audit helper.
	payload = {
		'epoch': 25,
		'global_step': 15_625,
		'amp_enabled': False,
		'stratigraphy_config': config,
		'control_identity': _strat_hmm_control_identity(config),
		'training_state': {
			'schema_version': 1,
			'stage': 'train_strat_hmm_pretext',
			'checkpoint_kind': 'epoch',
			'batch_index': None,
		},
		'model_state_dict': {'weight': torch.ones(1)},
	}
	torch.save(payload, checkpoint)
	return config_path, checkpoint, payload


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict]:
	return _make_source(tmp_path, monkeypatch)


def test_all_three_final_sources_pass_without_materializing_weights(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	configs = [_make_source(tmp_path, monkeypatch, arm)[0] for arm in ARMS]

	def reject_tensor_load(*_args: object, **_kwargs: object) -> None:
		raise AssertionError('audit must not materialize checkpoint tensor storage')

	monkeypatch.setattr(torch, 'load', reject_tensor_load)
	result = audit_f3_hmm_final_sources(configs)
	assert result['source_count'] == 3
	assert result['status'] == 'complete'
	assert {
		row['scientific_identity']['distillation_weight'] for row in result['sources']
	} == {0.1, 0.2}


@pytest.mark.parametrize(
	('keys', 'value', 'message'),
	[
		(('training_state', 'checkpoint_kind'), 'step', 'checkpoint_kind'),
		(('training_state', 'stage'), 'train_amp_mae', 'training_state.stage'),
		(('training_state', 'batch_index'), 624, 'partial batch_index'),
		(('epoch',), 24, 'budget is incomplete'),
		(('global_step',), 15_624, 'budget is incomplete'),
		(
			('stratigraphy_config', 'data', 'amplitude_agc', 'window_z'),
			33,
			'resolved configuration differs',
		),
		(
			('stratigraphy_config', 'train', 'weight_decay'),
			0.1,
			'resolved configuration differs',
		),
		(
			('control_identity', 'resolved_training_config_sha256'),
			'0' * 64,
			'configuration SHA mismatch',
		),
		(('control_identity', 'model_tag'), 'another_arm', 'model_tag mismatch'),
		(
			('control_identity', 'scientific_identity', 'distillation_weight'),
			0.9,
			'scientific_identity mismatch',
		),
		(('amp_enabled',), True, 'AMP state'),
	],
)
def test_rejects_incomplete_or_mismatched_checkpoint(
	source: tuple[Path, Path, dict],
	keys: tuple[str, ...],
	value: object,
	message: str,
) -> None:
	config, checkpoint, payload = source
	section = payload
	for key in keys[:-1]:
		section = section[key]
	section[keys[-1]] = value
	torch.save(payload, checkpoint)
	with pytest.raises(ValueError, match=message):
		audit_f3_hmm_final_source(config)


@pytest.mark.parametrize(
	'file_role',
	[
		'teacher_checkpoint',
		'labels',
		'confidence',
		'valid_tokens',
		'metadata',
		'boundary_weight',
	],
)
def test_rejects_live_parent_or_target_byte_drift(
	source: tuple[Path, Path, dict], file_role: str
) -> None:
	config, _, payload = source
	inputs = payload['control_identity']['input_identities']
	identity = (
		inputs['teacher_checkpoint']
		if file_role == 'teacher_checkpoint'
		else inputs['pseudo_targets'][0][file_role]
	)
	path = Path(identity['path'])
	# Whitespace keeps JSON valid while changing the recorded byte identity.
	path.write_bytes(path.read_bytes() + b'\n')
	with pytest.raises(ValueError, match='live parent/target input identities'):
		audit_f3_hmm_final_source(config)


def test_rejects_unrecorded_live_target_survey(source: tuple[Path, Path, dict]) -> None:
	config, _, payload = source
	root = Path(payload['stratigraphy_config']['pseudo_targets']['input_dir'])
	_write_targets(root, survey='unrecorded_survey')
	with pytest.raises(ValueError, match='live parent/target input identities'):
		audit_f3_hmm_final_source(config)


def test_rejects_missing_training_state(source: tuple[Path, Path, dict]) -> None:
	config, checkpoint, payload = source
	del payload['training_state']
	torch.save(payload, checkpoint)
	with pytest.raises(TypeError, match='training_state must be a mapping'):
		audit_f3_hmm_final_source(config)


def test_rejects_smoke_recipe(source: tuple[Path, Path, dict]) -> None:
	config, _, _ = source
	with pytest.raises(ValueError, match='full F3 25-epoch'):
		audit_f3_hmm_final_source(config.with_name('01_smoke.yaml'))


def test_rejects_empty_or_duplicate_sources(source: tuple[Path, Path, dict]) -> None:
	config, _, _ = source
	with pytest.raises(ValueError, match='at least one'):
		audit_f3_hmm_final_sources([])
	with pytest.raises(ValueError, match='distinct checkpoints'):
		audit_f3_hmm_final_sources([config, config])
