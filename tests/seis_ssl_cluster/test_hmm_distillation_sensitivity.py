"""Regression contracts for the opt-in fixed-target distillation sensitivity."""

import csv
import json
import statistics
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import seis_ssl_cluster.hmm.distillation_sensitivity as study
from seis_ssl_cluster.config.pretraining import resolve_strat_hmm_pretext_config
from seis_ssl_cluster.f3.lithology.common_test import (
	DISTILLATION_NAMESPACE,
	RANDOM_DISTILLATION_NAMESPACE,
	_expected_conditions,
	_scope,
)
from seis_ssl_cluster.hmm.distillation_auxiliary import (
	PUBLISHED_FILES as AUX_FILES,
)
from seis_ssl_cluster.hmm.distillation_auxiliary import (
	_pair,
	publish_auxiliary,
)
from seis_ssl_cluster.hmm.distillation_review import (
	CURATED_COPIES,
	RANDOM_CURATED_COPIES,
	curate_review,
)
from seis_ssl_cluster.hmm.distillation_sensitivity import (
	LAYOUTS,
	PUBLISHED_FILES,
	SIZES,
	SURVEYS,
	WEIGHTS,
	comparison_rows,
	publish_summary,
	validate_matrix,
)
from seis_ssl_cluster.training.strat_hmm.components import build_strat_hmm_components
from seis_ssl_cluster.training.strat_hmm.encoding import UnmaskedEncoderPair
from seis_ssl_cluster.training.strat_hmm.epoch import (
	train_strat_hmm_multi_head_one_epoch,
)
from seis_ssl_cluster.training.strat_hmm.feature_diagnostics import (
	FixedCropFeatureDiagnostics,
)
from seis_ssl_cluster.training.strat_hmm.losses import (
	compute_strat_hmm_multi_head_losses,
)
from tests.seis_ssl_cluster.test_config_strat_hmm_multi_head import _multi_head_config
from tests.seis_ssl_cluster.test_strat_hmm_pretraining_head_only import (
	_resolved_config,
	_single_batch_dataloader,
)


def _config(tmp_path, weight, *, diagnostics=True):
	raw = _multi_head_config(tmp_path)
	raw['student']['unfreeze_top_blocks'] = 1
	raw['loss']['distillation_weight'] = weight
	if diagnostics:
		raw['train']['feature_diagnostics_samples'] = 2
	return raw


@pytest.mark.parametrize('weight', [-0.1, float('nan'), float('inf')])
def test_negative_or_nonfinite_distillation_rejected(tmp_path, weight):
	raw = _config(tmp_path, weight)
	with pytest.raises(ValueError, match='distillation_weight'):
		resolve_strat_hmm_pretext_config(raw)
	with pytest.raises(ValueError, match='finite'):
		build_strat_hmm_components(raw, device='cpu')


def test_opt_in_retains_existing_zero_weight_guard(tmp_path):
	raw = _config(tmp_path, 0.0, diagnostics=False)
	with pytest.raises(ValueError, match='must be positive'):
		resolve_strat_hmm_pretext_config(raw)
	raw['train']['feature_diagnostics_samples'] = 2
	config = resolve_strat_hmm_pretext_config(raw)
	assert config['student']['unfreeze_top_blocks'] == 1


def test_initialization_and_rng_match_positive_reference(tmp_path):
	raw = _runtime_config(tmp_path, 0.2)
	states = []
	for weight, diagnostics in [
		(0.2, False),
		(0.0, True),
		(0.1, True),
		(0.2, True),
		(0.4, True),
	]:
		candidate = deepcopy(raw)
		candidate['loss']['distillation_weight'] = weight
		if not diagnostics:
			del candidate['train']['feature_diagnostics_samples']
		torch.manual_seed(42)
		components = build_strat_hmm_components(candidate, device='cpu')
		states.append(
			(
				components.student.state_dict(),
				components.heads.state_dict(),
				torch.get_rng_state(),
			)
		)
		assert components.teacher is not None
		assert not any(p.requires_grad for p in components.teacher.parameters())
	for state in states[1:]:
		torch.testing.assert_close(state, states[0], rtol=0, atol=0)


def test_zero_weight_updates_only_top_block_and_has_no_distillation_gradient(tmp_path):
	config = _runtime_config(tmp_path, 0.0)
	torch.manual_seed(42)
	c = build_strat_hmm_components(config, device='cpu')
	# Align the three targets to the existing tiny amplitude crop.
	batch = next(iter(_single_batch_dataloader(config)))
	target = {
		k: batch[k]
		for k in ('labels', 'confidence', 'valid_mask', 'boundary_weight')
		if k in batch
	}
	# The collator uses strat_ prefixed names.
	if not target:
		target = {
			k.removeprefix('strat_'): v
			for k, v in batch.items()
			if k.startswith('strat_')
		}
	batch['strat_multi_targets'] = {f'k{k}': deepcopy(target) for k in (6, 8, 10)}
	before = {k: v.clone() for k, v in c.student.state_dict().items()}
	with torch.no_grad():
		for p in c.student.parameters():
			if p.requires_grad:
				p.add_(0.01 * torch.randn_like(p))
	encoded, reference = UnmaskedEncoderPair(
		c.student, c.teacher, use_teacher=True
	).encode(batch['x'], valid_mask=batch['local_valid_mask'])
	losses = compute_strat_hmm_multi_head_losses(
		heads=c.heads,
		encoded=encoded,
		teacher_encoded=reference,
		batch=batch,
		loss_config=config['loss'],
		pseudo_target_config=config['pseudo_targets'],
	)
	assert losses['loss_distillation'].item() > 0
	params = [p for p in c.student.parameters() if p.requires_grad]
	zero_grad = torch.autograd.grad(
		0.0 * losses['loss_distillation'], params, retain_graph=True, allow_unused=True
	)
	assert all(g is None or torch.count_nonzero(g) == 0 for g in zero_grad)
	objective = (
		losses['loss_prototype']
		+ config['loss']['usage_weight'] * losses['loss_usage']
		+ config['loss']['consistency_weight'] * losses['loss_consistency']
	)
	actual_grad = torch.autograd.grad(losses['loss'], params, retain_graph=True)
	expected_grad = torch.autograd.grad(objective, params, retain_graph=True)
	torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
	pre_step = [p.clone() for p in params]
	losses['loss'].backward()
	c.optimizer.step()
	assert any(not torch.equal(a, b) for a, b in zip(pre_step, params, strict=True))
	trainable = {name for name, p in c.student.named_parameters() if p.requires_grad}
	assert trainable
	assert all(name.startswith('encoder.layers.1.') for name in trainable)
	assert all(
		torch.equal(before[name], value)
		for name, value in c.student.state_dict().items()
		if name not in trainable
	)
	assert all(p.grad is None for p in c.teacher.parameters())


def _runtime_config(tmp_path, weight):
	config = _resolved_config(
		tmp_path, encoder_depth=2, unfreeze_top_blocks=1, distillation_weight=0.2
	)
	config['head'] = {
		'spec': 'multi_resolution_ordered_prototypes_v1',
		'ks': [6, 8, 10],
		'projection_dim': 4,
		'temperature': 0.1,
		'normalize': True,
	}
	config['loss'].update(
		distillation_weight=weight, consistency_weight=0.1, consistency_beta=0.1
	)
	config['train']['feature_diagnostics_samples'] = 2
	return config


def _summary_cells(weights=WEIGHTS):

	return [
		{
			'survey': s,
			'distillation_weight': float(w),
			'layout_id': l,
			'data_size': z,
			'primary': 0.5 + 0.01 * i,
			'secondary': 0.7,
			'primary_improvement_vs_020': 0.0,
		}
		for s in SURVEYS
		for w in weights
		for i, l in enumerate(LAYOUTS)
		for z in SIZES
	]


@pytest.mark.parametrize('weights', [WEIGHTS, ('0.0', '0.2')])
def test_complete_scope_sample_sd_and_exact_published_files(tmp_path, weights):

	cells = _summary_cells(weights)
	rows = comparison_rows(cells, weights=weights)
	assert len(rows) == len(weights) * 9
	assert rows[0]['primary_sample_sd'] == pytest.approx(
		statistics.stdev([0.5, 0.51, 0.52, 0.53, 0.54])
	)
	output = tmp_path / 'summary'
	publish_summary(
		output,
		{
			'weights': weights,
			'cells': cells,
			'rows': rows,
			'feature_diagnostics': [{'epoch': 0, 'distance': 0.0}],
			'validation': [{'validation_mean_iou': 0.5}],
		},
	)
	assert {p.name for p in output.iterdir()} == PUBLISHED_FILES
	with pytest.raises(ValueError, match=f'{len(weights) * 45} unique'):
		comparison_rows(cells[:-1], weights=weights)
	with pytest.raises(ValueError, match=f'{len(weights) * 45} unique'):
		comparison_rows([*cells[:-1], cells[0]], weights=weights)


@pytest.mark.parametrize(
	('source', 'weights'), [('mae', WEIGHTS), ('random', ('0.0', '0.2'))]
)
def test_summary_reuses_recorded_lineage_and_keeps_validation_separate(  # noqa: C901
	tmp_path, monkeypatch, source, weights
):
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	matrix = {
		'source': source,
		'weights': [float(w) for w in weights],
		'surveys': {s: {} for s in SURVEYS},
	}
	namespace = DISTILLATION_NAMESPACE
	if source == 'random':
		namespace = RANDOM_DISTILLATION_NAMESPACE
		matrix.update(
			analysis_id='random_distillation_ablation_v1', test_namespace=namespace
		)
	test_rows = []
	for survey in SURVEYS:
		for weight in weights:
			arm = {'candidate_id': weight, 'configs': {}}
			matrix['surveys'][survey][weight] = arm
			for layout in LAYOUTS:
				for size in SIZES:
					primary = 0.5 + float(weight) * 0.1
					path = tmp_path / survey / weight / layout / f'{size}.json'
					path.parent.mkdir(parents=True, exist_ok=True)
					metrics = {
						'f3': {'mean_iou': 0.1, 'macro_f1': 0.2},
						'parihaka': {
							'test': {'channel_iou': primary, 'channel_f1': 0.7}
						},
						'volve': {
							'test': {
								'primary_common': {
									'macro_mae_samples': primary,
									'macro_within_2_samples': 0.7,
								}
							}
						},
					}[survey]
					path.write_text(json.dumps(metrics))
					if survey == 'f3':
						test_rows.append(
							{
								'version': 'v2',
								'source': source,
								'head_ks': '6|8|10',
								'distillation_weight': weight,
								'consistency_weight': '0.1',
								'layout_id': layout,
								'data_size': size,
								'mean_iou': primary,
								'macro_f1': 0.7,
							}
						)
	for output_namespace, rows in (
		(namespace, test_rows),
		(
			'lithology_common_test_v1',
			[r for r in test_rows if r['distillation_weight'] == '0.2'],
		),
	):
		path = (
			tmp_path
			/ 'surveys/f3/facies_benchmark_v2/evaluation'
			/ output_namespace
			/ 'summary/per_layout.csv'
		)
		path.parent.mkdir(parents=True)
		study.write_rows(path, rows)
	operations = (
		tmp_path
		/ 'operations/hmm_v2'
		/ matrix.get('analysis_id', 'distillation_sensitivity_v1')
	)
	operations.mkdir(parents=True)
	for name in ('preflight.json', 'experiments_complete.json'):
		(operations / name).write_text('{"status": "complete"}')

	def forbid_audit(*_args, **_kwargs):
		pytest.fail(
			'summary must reuse recorded lineage without a new experiment audit'
		)

	monkeypatch.setattr(study, 'audit_multi_head_source', forbid_audit)
	monkeypatch.setattr(
		'seis_ssl_cluster.hmm.multi_source_cell.audit_completed_cell', forbid_audit
	)
	monkeypatch.setattr(study, 'collect_diagnostics', lambda _matrix: [])
	monkeypatch.setattr(
		study,
		'_cell_path',
		lambda s, a, l, z: tmp_path / s / a['candidate_id'] / l / f'{z}.json',
	)
	payload = study.collect_summary(matrix)
	assert len(payload['cells']) == len(weights) * 45
	assert len(payload['validation']) == len(weights) * 15
	assert all(r['validation_mean_iou'] == 0.1 for r in payload['validation'])
	for row in payload['cells']:
		if row['distillation_weight'] == 0:
			expected = 0.02 if row['survey'] == 'volve' else -0.02
			assert row['primary_improvement_vs_020'] == pytest.approx(expected)
	assert len(payload['recorded_evidence']) == 2
	assert len(payload['rows']) == len(weights) * 9


def test_random_manifest_and_common_test_keep_fixed_endpoints():
	matrix = validate_matrix(
		Path('experiments/hmm_v2/random_distillation_ablation_v1/matrix.yaml')
	)
	assert matrix['source'] == 'random'
	assert matrix['weights'] == [0.0, 0.2]
	assert matrix['new_continuations'] == 3
	assert matrix['new_downstream_cells'] == 45
	assert matrix['final_cells'] == 90
	assert matrix['retrospective_audits'] is False
	for arms in matrix['surveys'].values():
		assert arms['0.2']['execution'] == 'reuse'
		config = yaml.safe_load(
			Path(arms['0.0']['configs']['training']['path']).read_text()
		)
		assert config['student']['init_checkpoint'] == config['teacher']['checkpoint']
		assert 'random' in config['teacher']['checkpoint']
		assert config['student']['unfreeze_top_blocks'] == 1
		assert config['train']['epochs'] == 25
		assert config['train']['seed'] == 42
		assert config['loss']['distillation_weight'] == 0
	assert _expected_conditions(RANDOM_DISTILLATION_NAMESPACE) == {
		('v2', 'random', '6|8|10', w, '0.1') for w in ('0.0', '0.2')
	}
	config = {
		'experiment_id': RANDOM_DISTILLATION_NAMESPACE,
		'expected_conditions': 2,
		'expected_cells': 30,
		'inputs': {},
	}
	with pytest.raises(ValueError, match='reference test mask'):
		_scope(config)
	config['inputs']['reference_test_mask'] = {}
	assert len(_scope(config)[1]) == 2


def test_fixed_f3_test_scope_and_reference_mask():

	expected = _expected_conditions(DISTILLATION_NAMESPACE)
	assert len(expected) == 4
	assert {r[3] for r in expected} == {'0.0', '0.1', '0.2', '0.4'}
	config = {
		'experiment_id': DISTILLATION_NAMESPACE,
		'expected_conditions': 4,
		'expected_cells': 60,
		'inputs': {},
	}
	with pytest.raises(ValueError, match='reference test mask'):
		_scope(config)
	config['inputs']['reference_test_mask'] = {}
	assert _scope(config)[1] == expected


def test_fixed_crop_diagnostics_preserve_rng_modes_teacher_and_resume(tmp_path):

	config = _runtime_config(tmp_path, 0.0)
	components = build_strat_hmm_components(config, device='cpu')
	batch = next(iter(_single_batch_dataloader(config)))

	class Dataset:
		epoch = 7

		def __len__(self):
			return 1

		def set_epoch(self, epoch):
			self.epoch = epoch

		def __getitem__(self, _index):
			torch.rand(3)
			np.random.random(3)  # noqa: NPY002 - exercise the captured global RNG
			target = {
				k.removeprefix('strat_'): v[0].numpy()
				for k, v in batch.items()
				if k.startswith('strat_')
			}
			return {
				'x': batch['x'][0].numpy(),
				'local_valid_mask': batch['local_valid_mask'][0].numpy(),
				'strat_multi_targets': {f'k{k}': target for k in (6, 8, 10)},
			}

	dataset = Dataset()
	torch_state = torch.get_rng_state().clone()
	numpy_state = np.random.get_state()  # noqa: NPY002
	diagnostics = FixedCropFeatureDiagnostics(dataset, samples=2, output_root=tmp_path)
	assert dataset.epoch == 7
	assert torch.equal(torch_state, torch.get_rng_state())
	np.testing.assert_equal(numpy_state, np.random.get_state())  # noqa: NPY002
	components.student.train()
	flags = [m.training for m in components.student.modules()]
	diagnostics.record(components.student, components.teacher, epoch=0, global_step=0)
	with torch.no_grad():
		for p in components.student.parameters():
			if p.requires_grad:
				p.add_(0.1 * torch.randn_like(p))
	diagnostics.record(components.student, components.teacher, epoch=1, global_step=1)
	diagnostics.record(components.student, components.teacher, epoch=1, global_step=1)
	with (tmp_path / 'feature_diagnostics.csv').open() as stream:
		rows = list(csv.DictReader(stream))
	assert [r['epoch'] for r in rows] == ['0', '1']
	assert float(rows[1]['teacher_cosine_distance']) > float(
		rows[0]['teacher_cosine_distance']
	)
	assert [m.training for m in components.student.modules()] == flags
	assert all(p.grad is None for p in components.teacher.parameters())


def test_auxiliary_improvement_direction_and_published_files(tmp_path):
	iou = _pair(
		'parihaka',
		'mae',
		'6',
		'medium',
		'layout_000',
		0.3,
		0.4,
		comparison='v1_dist010_vs020',
	)
	mae = _pair(
		'volve',
		'mae',
		'6',
		'medium',
		'layout_000',
		5.0,
		4.0,
		comparison='v1_dist010_vs020',
	)
	assert iou['improvement'] == pytest.approx(0.1)
	assert mae['improvement'] == 1.0
	publish_auxiliary(
		tmp_path / 'aux',
		{
			'v1': [iou],
			'v2': [mae],
			'rows': [iou],
			'missing': [],
			'interpretation': 'fixed comparisons',
		},
	)
	assert {p.name for p in (tmp_path / 'aux').iterdir()} == AUX_FILES


def test_prespecified_manifest_preserves_all_controls():

	matrix = validate_matrix()
	assert matrix['new_continuations'] == 9
	assert matrix['new_downstream_cells'] == 135
	assert matrix['final_cells'] == 180
	paths = set()
	for survey, arms in matrix['surveys'].items():
		assert set(arms) == {'0.0', '0.1', '0.2', '0.4'}
		assert arms['0.2']['execution'] == 'reuse'
		for weight, arm in arms.items():
			config = yaml.safe_load(
				Path(arm['configs']['training']['path']).read_text()
			)
			assert (
				config['teacher']['checkpoint'] == config['student']['init_checkpoint']
			)
			assert config['teacher']['checkpoint'].endswith('/full_100ep/latest.pt')
			assert config['student']['unfreeze_top_blocks'] == 1
			assert config['head']['ks'] == [6, 8, 10]
			assert config['train']['epochs'] == 25
			assert config['train']['samples_per_epoch'] == 10000
			assert config['train']['seed'] == 42
			assert config['train']['num_workers'] == 8
			assert config['train']['batch_size'] == (4 if survey == 'volve' else 16)
			assert config['loss']['distillation_weight'] == float(weight)
			assert config['loss']['prototype_weight'] == 1.0
			assert config['loss']['usage_weight'] == 0.005
			assert config['loss']['consistency_weight'] == 0.1
			if weight != '0.2':
				paths.update(c['path'] for c in arm['configs'].values())
	assert len(paths) == 36


@pytest.mark.parametrize('source', ['mae', 'random'])
def test_review_publication_owns_exact_file_set(tmp_path, source):
	weights = ('0.0', '0.2') if source == 'random' else WEIGHTS
	cells = _summary_cells(weights)
	study = tmp_path / 'artifacts'
	copies = RANDOM_CURATED_COPIES if source == 'random' else CURATED_COPIES
	for relative in copies.values():
		path = study / relative
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text('review fixture')
	payload = {
		'source': source,
		'weights': weights,
		'status': 'complete',
		'cells': cells,
		'rows': comparison_rows(cells, weights=weights),
		'feature_diagnostics': [
			{
				'survey': s,
				'distillation_weight': float(w),
				'epoch': 25,
				'fixed_crop_teacher_cosine_distance': 0.01,
			}
			for s in SURVEYS
			for w in weights
		],
	}
	(study / 'summary/summary.json').write_text(json.dumps(payload))
	output = tmp_path / 'reports'
	curate_review(study, output)
	expected = {
		'README.md',
		'provenance.json',
		'cells.csv',
		'comparison.csv',
		'feature_diagnostics.csv',
		'sensitivity.png',
		'auxiliary_v1_pairs.csv',
		'auxiliary_v2_consistency_pairs.csv',
		'auxiliary_comparison.csv',
		'f3_validation.csv',
	}
	if source == 'random':
		expected -= {
			'auxiliary_v1_pairs.csv',
			'auxiliary_v2_consistency_pairs.csv',
			'auxiliary_comparison.csv',
		}
	assert {p.name for p in output.iterdir()} == expected


def test_remaining_gate_requires_all_medium_conditions(tmp_path, monkeypatch):
	matrix = validate_matrix()
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))

	def medium_path(survey, arm, layout, size):
		assert size == 'medium'
		path = tmp_path / f'{survey}_{arm["candidate_id"]}_{layout}'
		if not (
			survey == 'volve'
			and 'distill040' in arm['candidate_id']
			and layout == 'layout_004'
		):
			path.touch()
		return path

	monkeypatch.setattr(study, '_cell_path', medium_path)
	with pytest.raises(ValueError, match='all 45 medium cells'):
		study.execute_phase(
			matrix, SimpleNamespace(phase='remaining', survey='f3', weight=0.0)
		)


def test_positive_weight_retains_missing_teacher_failure(tmp_path):
	config = _runtime_config(tmp_path, 0.2)
	c = build_strat_hmm_components(config, device='cpu')
	batch = next(iter(_single_batch_dataloader(config)))
	with pytest.raises(
		ValueError,
		match=r'teacher is required when loss\.distillation_weight is positive',
	):
		train_strat_hmm_multi_head_one_epoch(
			student=c.student,
			teacher=None,
			heads=c.heads,
			dataloader=[batch],
			optimizer=c.optimizer,
			device=torch.device('cpu'),
			epoch=1,
			loss_config=config['loss'],
			pseudo_target_config=config['pseudo_targets'],
		)
