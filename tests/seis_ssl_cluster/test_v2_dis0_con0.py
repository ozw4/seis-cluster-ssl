"""Contracts for the fixed multi-head zero-distillation/zero-consistency study."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from seis_ssl_cluster.f3.lithology.common_test import (
	ZERO_CONSTRAINT_NAMESPACE,
	_expected_conditions,
	_scope,
)
from seis_ssl_cluster.hmm import v2_dis0_con0 as subject
from seis_ssl_cluster.training.strat_hmm.components import build_strat_hmm_components
from seis_ssl_cluster.training.strat_hmm.encoding import UnmaskedEncoderPair
from seis_ssl_cluster.training.strat_hmm.losses import (
	compute_strat_hmm_multi_head_losses,
)
from tests.seis_ssl_cluster.test_hmm_distillation_sensitivity import _runtime_config
from tests.seis_ssl_cluster.test_strat_hmm_pretraining_head_only import (
	_single_batch_dataloader,
)


def test_all_nine_arms_inherit_their_own_dis0_parent():
	matrix = subject.validate_matrix()
	assert len(matrix['arms']) == 9
	for arm in matrix['arms']:
		assert len(arm['configs']) == 4
		assert 'distill000_cons000' in arm['candidate_id']
	for survey in subject.SURVEYS:
		arm = next(a for a in matrix['arms'] if a['survey'] == survey)
		root = Path(arm['configs']['training']['path']).parents[2]
		assert len(list(root.rglob('*.yaml'))) == 12


def test_volve_downstream_cannot_point_to_a_different_training_definition(tmp_path):
	arm = deepcopy(
		next(a for a in subject.validate_matrix()['arms'] if a['survey'] == 'volve')
	)
	raw = yaml.safe_load(Path(arm['configs']['downstream']['path']).read_text())
	raw['arm']['multi_head_training_config'] = (
		'${SEIS_SSL_CLUSTER_WORKSPACE}/wrong/full.yaml'
	)
	path = tmp_path / 'downstream.yaml'
	path.write_text(yaml.safe_dump(raw))
	arm['configs']['downstream']['path'] = str(path)
	with pytest.raises(ValueError, match='own training recipe'):
		subject.validate_inherited(arm)


def test_legacy_smoke_binding_rejects_another_output_destination():
	arm = deepcopy(
		next(a for a in subject.validate_matrix()['arms'] if a['survey'] == 'volve')
	)
	arm['legacy_smoke_output_root'] += '_different'
	with pytest.raises(ValueError, match='original Volve output'):
		subject.validate_inherited(arm)


def test_completed_smoke_is_reused_without_gpu_admission_or_checkpoint_rewrite(
	tmp_path, monkeypatch
):
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	operations = root / 'operations/hmm_v2' / subject.STUDY
	operations.mkdir(parents=True)
	legacy = root / 'vis_dis0_con0_v1/smoke_1step'
	historical = tmp_path / 'historical'
	legacy.mkdir(parents=True)
	historical.mkdir()
	metadata = {
		'control_identity': {
			'initial_parameter_sha256': {'student': 'same', 'heads': 'same'}
		}
	}
	for path in (legacy, historical):
		(path / 'run_metadata.json').write_text(json.dumps(metadata))
	payload = {
		'global_step': 1,
		'metrics': {
			'loss_distillation_weighted': 0.0,
			'loss_consistency_weighted': 0.0,
			'loss_prototype_weighted': 1.0,
		},
	}
	checkpoint = legacy / 'latest.pt'
	torch.save(payload, checkpoint)
	before = checkpoint.read_bytes()
	arm = {
		'survey': 'volve',
		'source': 'mae',
		'configs': {'smoke': {'path': 'smoke.yaml'}},
		'historical_training': {'path': 'historical.yaml'},
	}
	monkeypatch.setattr(
		subject,
		'load_config',
		lambda path: {
			'paths': {
				'output_root': str(legacy if path == 'smoke.yaml' else historical)
			}
		},
	)

	def must_not_execute(*_args, **_kwargs):
		raise AssertionError(
			'completed CPU validation must not start training or wait for a GPU'
		)

	monkeypatch.setattr(subject, 'gpu_available', must_not_execute)
	monkeypatch.setattr(subject, 'execute_phase', must_not_execute)
	subject.run_phase({'arms': [arm]}, 'smoke', ['0'])
	proof = json.loads((operations / 'smoke_validation/volve_mae.json').read_text())
	assert proof['reused'] is True
	assert proof['initial_parameters_match_historical_v2'] is True
	assert checkpoint.read_bytes() == before
	payload['metrics']['loss_consistency_weighted'] = 0.1
	torch.save(payload, checkpoint)
	with pytest.raises(ValueError, match='both constraints contributing zero'):
		subject.run_phase({'arms': [arm]}, 'smoke', ['0'])


@pytest.mark.parametrize('kind', ['learning_rate', 'target', 'coefficient', 'smoke'])
def test_unrelated_recipe_changes_and_incorrect_coefficients_rejected(tmp_path, kind):
	arm = deepcopy(subject.validate_matrix()['arms'][0])
	key = 'smoke' if kind == 'smoke' else 'training'
	raw = yaml.safe_load(Path(arm['configs'][key]['path']).read_text())
	if kind == 'learning_rate':
		raw['train']['lr'] = 0.5
	elif kind == 'target':
		raw['pseudo_targets']['manifest'] = 'other-target.json'
	elif kind == 'coefficient':
		raw['loss']['consistency_weight'] = 0.1
	else:
		raw['train']['batch_size'] = 1
	path = tmp_path / 'changed.yaml'
	path.write_text(yaml.safe_dump(raw))
	arm['configs'][key]['path'] = str(path)
	with pytest.raises(ValueError, match=r'recipe change|expected K6810|smoke differs'):
		subject.validate_inherited(arm)


def test_both_zero_constraints_leave_only_prototype_and_usage_gradients(tmp_path):
	config = _runtime_config(tmp_path, 0.0)
	config['loss']['consistency_weight'] = 0.0
	torch.manual_seed(42)
	c = build_strat_hmm_components(config, device='cpu')
	batch = next(iter(_single_batch_dataloader(config)))
	target = {
		k.removeprefix('strat_'): v for k, v in batch.items() if k.startswith('strat_')
	}
	batch['strat_multi_targets'] = {f'k{k}': deepcopy(target) for k in (6, 8, 10)}
	params = [p for p in c.student.parameters() if p.requires_grad]
	with torch.no_grad():
		for p in params:
			p.add_(0.01 * torch.randn_like(p))
	encoded, teacher = UnmaskedEncoderPair(
		c.student, c.teacher, use_teacher=True
	).encode(batch['x'], valid_mask=batch['local_valid_mask'])
	losses = compute_strat_hmm_multi_head_losses(
		heads=c.heads,
		encoded=encoded,
		teacher_encoded=teacher,
		batch=batch,
		loss_config=config['loss'],
		pseudo_target_config=config['pseudo_targets'],
	)
	assert losses['loss_distillation'].item() > 0
	assert losses['loss_consistency'].item() > 0
	objective = (
		losses['loss_prototype'] + config['loss']['usage_weight'] * losses['loss_usage']
	)
	actual = torch.autograd.grad(losses['loss'], params, retain_graph=True)
	expected = torch.autograd.grad(objective, params, retain_graph=True)
	torch.testing.assert_close(actual, expected, rtol=0, atol=0)
	before = [p.detach().clone() for p in params]
	losses['loss'].backward()
	c.optimizer.step()
	assert any(not torch.equal(a, b) for a, b in zip(before, params, strict=True))
	assert all(p.grad is None for p in c.teacher.parameters())


def test_common_test_extension_has_only_three_fixed_zero_constraint_conditions():
	conditions = _expected_conditions(ZERO_CONSTRAINT_NAMESPACE)
	assert conditions == {('v2', s, '6|8|10', '0.0', '0') for s in subject.SOURCES}
	config = {
		'experiment_id': ZERO_CONSTRAINT_NAMESPACE,
		'expected_conditions': 3,
		'expected_cells': 45,
		'inputs': {},
	}
	with pytest.raises(ValueError, match='reference test mask'):
		_scope(config)
	config['inputs']['reference_test_mask'] = {
		'path': 'original-mask',
		'sha256': 'bound',
	}
	assert _scope(config)[1] == conditions


@pytest.mark.parametrize(
	('free', 'utilization', 'expected'),
	[(50000, 100, False), (10000, 0, False), (50000, 0, True)],
)
def test_gpu_admission_preserves_busy_jobs(monkeypatch, free, utilization, expected):
	monkeypatch.setattr(
		subject.subprocess,
		'check_output',
		lambda *_a, **_kw: f'{free}, {utilization}\n',
	)
	assert subject.gpu_available('0', 38000) is expected


def _cells():
	return [
		{
			'survey': s,
			'source': f,
			'layout_id': l,
			'data_size': z,
			'primary': 0.5 + 0.01 * i,
			'secondary': 0.7,
			'evaluation_split': 'common_test' if s == 'f3' else 'test',
		}
		for s in subject.SURVEYS
		for f in subject.SOURCES
		for i, l in enumerate(subject.LAYOUTS)
		for z in subject.SIZES
	]


def test_complete_grid_and_exact_summary_and_review_publication(tmp_path, monkeypatch):
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts'))
	cells = _cells()
	rows = subject.aggregate_cells(cells)
	assert len(rows) == 27
	assert rows[0]['primary_mean'] == pytest.approx(0.52)
	assert rows[0]['primary_sample_sd'] == pytest.approx(0.0158113883)
	payload = {
		'cells': cells,
		'rows': rows,
		'validation': [{'source': 'mae', 'mean_iou': 0.5}],
		'diagnostics': [
			{
				'epoch': 1,
				'loss_distillation_weighted': 0.0,
				'loss_consistency_weighted': 0.0,
			}
		],
	}
	root = tmp_path / 'artifacts/studies/summary'
	subject.publish_summary(root, payload)
	assert {p.name for p in root.iterdir()} == subject.SUMMARY_FILES
	assert json.loads((root / 'summary.json').read_text())['cells'] == cells
	review = tmp_path / 'reports/v2_dis0_con0_v1'
	subject.publish_review(root, review)
	assert {p.name for p in review.iterdir()} == subject.REVIEW_FILES
	assert (review / 'cells.csv').read_bytes() == (root / 'cells.csv').read_bytes()
	with pytest.raises(FileExistsError):
		subject.publish_summary(root, payload)
	with pytest.raises(FileExistsError):
		subject.publish_review(root, review)
	for invalid in (cells[:-1], [*cells[:-1], cells[0]]):
		with pytest.raises(ValueError, match='135 unique'):
			subject.aggregate_cells(invalid)
	bad = deepcopy(cells)
	bad[0]['primary'] = float('nan')
	with pytest.raises(ValueError, match='finite'):
		subject.aggregate_cells(bad)
