"""Contracts for zero/zero unordered targets, matching HMM and complete publication."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster.clustering import kmeans_study
from seis_ssl_cluster.clustering import kmeans_zero as zero
from seis_ssl_cluster.clustering.kmeans_definition import read_yaml, validate_study
from seis_ssl_cluster.clustering.kmeans_study import (
	command_plan,
	comparison_rows,
	publish_summary,
)
from seis_ssl_cluster.config import resolve_strat_hmm_pretext_config
from seis_ssl_cluster.f3.lithology.common_test import _expected_conditions, _scope
from seis_ssl_cluster.training.checkpoint import load_checkpoint
from seis_ssl_cluster.training.strat_hmm.runner import run_strat_hmm_pretext_training
from tests.seis_ssl_cluster.test_kmeans_study import _cells
from tests.seis_ssl_cluster.test_unordered_kmeans_targets import (
	_targets,
	_training_config,
)

ROOT = Path(__file__).resolve().parents[2]


def test_all_arms_start_from_hmm_parents_with_zero_constraints():
	matrix = validate_study(ROOT / zero.MATRIX, ROOT)
	args = argparse.Namespace(
		first='targets',
		last='summary',
		survey=None,
		candidate=None,
		layout=None,
		size=None,
		resume=False,
	)
	commands = command_plan(ROOT / zero.MATRIX, ROOT, args)
	assert sum(stage == 'smoke' for stage, _ in commands) == 9
	assert sum(stage == 'full' for stage, _ in commands) == 9
	assert sum(stage == 'downstream' for stage, _ in commands) == 135
	for arms in matrix['surveys'].values():
		for arm in arms.values():
			cfg = read_yaml(ROOT / arm['configs']['training'])
			old = read_yaml(ROOT / arm['references']['training']['path'])
			assert cfg['student'] == old['student']
			assert cfg['teacher'] == old['teacher']
			assert cfg['loss'] == old['loss']
			assert cfg['train'] == old['train']
			assert cfg['loss']['distillation_weight'] == 0.0
			assert cfg['loss']['consistency_weight'] == 0.0
			assert (
				cfg['pseudo_targets']['target_representation']
				== 'unordered_kmeans_labels_v1'
			)
			assert zero.STUDY in cfg['paths']['output_root']
			assert (
				cfg['student']['init_checkpoint']
				!= cfg['paths']['output_root'] + '/latest.pt'
			)


@pytest.mark.parametrize('corruption', ['missing', 'hash', 'loss', 'output'])
def test_definition_rejects_missing_arms_and_unmatched_recipes(tmp_path, corruption):
	matrix = copy.deepcopy(read_yaml(ROOT / zero.MATRIX))
	arm = matrix['surveys']['f3']['mae']
	if corruption == 'missing':
		del matrix['surveys']['volve']['mae']
	elif corruption == 'hash':
		arm['references']['training']['sha256'] = '0' * 64
	else:
		cfg = read_yaml(ROOT / arm['configs']['training'])
		if corruption == 'loss':
			cfg['loss']['distillation_weight'] = 0.2
		else:
			cfg['paths']['output_root'] = '/unrelated/output'
		path = tmp_path / 'training.yaml'
		path.write_text(yaml.safe_dump(cfg))
		arm['configs']['training'] = str(path)
	with pytest.raises(ValueError, match=r'all three|hash drift|definition drift'):
		zero.validate_definition(matrix, ROOT)


def test_unordered_zero_training_diagnostics_and_existing_guard(tmp_path):
	manifest, _ = _targets(tmp_path, shape=(2, 2, 2), margins=(0, 0, 0))
	raw = _training_config(tmp_path, manifest)
	raw['loss']['distillation_weight'] = 0.0
	with pytest.raises(ValueError, match='must be positive'):
		resolve_strat_hmm_pretext_config(raw)
	raw['train']['feature_diagnostics_samples'] = 2
	config = resolve_strat_hmm_pretext_config(raw)
	path = run_strat_hmm_pretext_training(config)
	payload = load_checkpoint(path, map_location='cpu')
	assert payload['stratigraphy_checkpoint']['schema_version'] == 9
	assert payload['metrics']['loss_distillation_weighted'] == 0.0
	assert payload['metrics']['loss_consistency_weighted'] == 0.0
	assert payload['metrics']['loss_prototype_weighted'] > 0.0
	assert (path.parent / 'feature_diagnostics.csv').is_file()


def test_kmeans_test_scope_requires_shared_reference_mask():
	assert _expected_conditions(zero.TEST_NAMESPACE) == {
		('kmeans', source, '6|8|10', '0.0', '0') for source in zero.SOURCES
	}
	with pytest.raises(ValueError, match='reference test mask'):
		_scope(
			{
				'experiment_id': zero.TEST_NAMESPACE,
				'expected_conditions': 3,
				'expected_cells': 45,
				'inputs': {},
			}
		)


def test_exact_artifact_and_review_publication(tmp_path):
	cells = _cells()
	payload = {
		'study': zero.STUDY,
		'cells': cells,
		'rows': comparison_rows(cells),
		'interpretation': 'fixture',
	}
	summary = tmp_path / 'artifacts/summary'
	publish_summary(summary, payload)
	assert {p.name for p in summary.iterdir()} == zero.SUMMARY_FILES
	review = tmp_path / 'reports/review'
	zero.publish_review(summary, review)
	assert {p.name for p in review.iterdir()} == zero.REVIEW_FILES
	assert json.loads((review / 'provenance.json').read_text())['cells'] == 135
	with pytest.raises(FileExistsError):
		zero.publish_review(summary, review)


def test_execution_failure_stops_admission_and_is_recorded(tmp_path, monkeypatch):
	def fail(*_a, **_kw):
		raise RuntimeError('owned training failed')

	monkeypatch.setattr(kmeans_study, 'run_command', fail)
	with pytest.raises(RuntimeError, match='owned training failed'):
		zero.execute_recorded(
			{'surveys': {}},
			ROOT,
			tmp_path,
			0,
			9,
			'audit',
			['python', '--config', 'owned.yaml'],
			tmp_path / 'run.log',
		)
	state = json.loads((tmp_path / 'status.json').read_text())
	assert state['status'] == 'failed'
	assert state['stage'] == 'audit'


def test_full_training_requires_all_smoke_checks(tmp_path, monkeypatch):
	called = []
	monkeypatch.setattr(zero, 'wait_for_gpu', lambda *_a: None)
	monkeypatch.setattr(
		kmeans_study, 'run_command', lambda *_a, **_kw: called.append(True)
	)

	def missing_smoke(*_a):
		raise ValueError('missing verified smoke')

	monkeypatch.setattr(zero, 'validate_smoke', missing_smoke)
	with pytest.raises(ValueError, match='missing verified smoke'):
		zero.execute_recorded(
			{'surveys': {'f3': {'mae': {}}}},
			ROOT,
			tmp_path,
			0,
			9,
			'full',
			['python', '--config', 'owned.yaml'],
			tmp_path / 'run.log',
		)
	assert called == []


@pytest.mark.parametrize('corruption', ['none', 'mask', 'count'])
def test_comparison_uses_common_test_and_preserves_error_direction(
	tmp_path, monkeypatch, corruption
):
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	base = tmp_path / 'surveys/f3/facies_benchmark_v2/evaluation'
	for i, namespace in enumerate(
		(zero.TEST_NAMESPACE, 'lithology_common_test_v2_dis0_con0_v1')
	):
		root = base / namespace
		(root / 'inputs').mkdir(parents=True)
		(root / 'summary').mkdir()
		definition = {
			'mask_sha256': 'shared',
			'voxel_count': 113979624,
			'split': 'test',
		}
		if i == 0 and corruption == 'mask':
			definition['mask_sha256'] = 'different'
		if i == 0 and corruption == 'count':
			definition['voxel_count'] = 1
		(root / 'inputs/test_definition.json').write_text(json.dumps(definition))
		rows = [
			{
				'source': f,
				'layout_id': l,
				'data_size': s,
				'mean_iou': 0.3 + i / 10,
				'macro_f1': 0.4 + i / 10,
			}
			for f in zero.SOURCES
			for l in zero.LAYOUTS
			for s in zero.SIZES
		]
		zero.write_rows(root / 'summary/per_layout.csv', rows)
	payload = {'cells': _cells()}
	if corruption != 'none':
		with pytest.raises(ValueError, match='identical common-test masks'):
			zero.finalize_payload(payload)
		return
	result = zero.finalize_payload(payload)
	assert len(result['rows']) == 27
	for cell in result['cells']:
		assert cell['decoder_seed'] == 42000
		if cell['survey'] == 'f3':
			assert cell['evaluation_split'] == 'common_test'
			assert cell['primary'] == pytest.approx(0.3)
			assert cell['control_primary'] == pytest.approx(0.4)
			assert cell['primary_improvement'] == pytest.approx(-0.1)
		elif cell['survey'] == 'volve':
			assert cell['primary_improvement'] == -cell['primary_difference']


@pytest.mark.parametrize('corruption', ['none', 'distillation', 'initial'])
def test_smoke_requires_matching_parameters_and_zero_constraints(
	tmp_path, monkeypatch, corruption
):
	import torch  # noqa: PLC0415

	arm = {
		'configs': {'smoke': 'smoke.yaml'},
		'references': {'training': {'path': 'hmm.yaml'}},
		'candidate_id': 'candidate',
	}
	for name in ('smoke', 'hmm'):
		root = tmp_path / name
		root.mkdir()
		(tmp_path / f'{name}.yaml').write_text(
			yaml.safe_dump({'paths': {'output_root': str(root)}})
		)
		(root / 'run_metadata.json').write_text(
			json.dumps({'control_identity': {'initial_parameter_sha256': 'same'}})
		)
	payload = {
		'global_step': 1,
		'metrics': {
			'loss_prototype_weighted': 1.0,
			'loss_distillation_weighted': 0.0,
			'loss_consistency_weighted': 0.0,
		},
	}
	if corruption == 'initial':
		(tmp_path / 'smoke/run_metadata.json').write_text(
			json.dumps({'control_identity': {'initial_parameter_sha256': 'changed'}})
		)
	elif corruption == 'distillation':
		payload['metrics']['loss_distillation_weighted'] = 0.1
	monkeypatch.setattr(torch, 'load', lambda *_a, **_kw: payload)
	if corruption == 'none':
		assert zero.validate_smoke(arm, tmp_path)['initial_parameters_match']
	else:
		with pytest.raises(ValueError, match=r'losses|initial parameters'):
			zero.validate_smoke(arm, tmp_path)


@pytest.mark.parametrize('changed_metric', [False, True])
def test_catalog_refresh_preserves_previous_metrics_and_backup(
	tmp_path, monkeypatch, changed_metric
):
	import zipfile  # noqa: PLC0415

	from seis_ssl_cluster.results import experiment_catalog  # noqa: PLC0415

	artifact = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(artifact))
	parent = artifact / 'studies/all_experiment_metrics_20261115_v1'
	old = parent / 'catalog'
	old.mkdir(parents=True)
	row = {
		'source_file': 'retained/metrics.json',
		'split': 'test',
		'support': 'same',
		'checkpoint_selection': '',
		'channel_iou': '0.5',
	}
	zero.write_rows(old / 'experiment_metrics.csv', [row])
	(parent / 'catalog.zip').write_bytes(b'previous bundle')

	def build(_workspace, _artifact, output):
		output.mkdir()
		new_row = {**row, 'channel_iou': '0.6' if changed_metric else '0.5'}
		zero.write_rows(output / 'experiment_metrics.csv', [new_row])
		(output / 'index.html').write_text('new HTML')
		(output / 'manifest.json').write_text(
			json.dumps(
				{
					'published_files': [
						'experiment_metrics.csv',
						'index.html',
						'manifest.json',
					],
					'sha256': {p.name: zero.file_sha256(p) for p in output.iterdir()},
				}
			)
		)

	monkeypatch.setattr(experiment_catalog, 'build_catalog', build)
	if changed_metric:
		with pytest.raises(ValueError, match='omitted or changed'):
			zero.refresh_catalog(tmp_path)
		assert (parent / 'catalog.zip').read_bytes() == b'previous bundle'
		assert zero.read_rows(old / 'experiment_metrics.csv')[0]['channel_iou'] == '0.5'
	else:
		zero.refresh_catalog(tmp_path)
		backups = list(
			(artifact / 'operations/kmeans' / zero.STUDY).glob('previous_catalog_*')
		)
		assert len(backups) == 1
		assert (backups[0] / 'catalog.zip').read_bytes() == b'previous bundle'
		with zipfile.ZipFile(parent / 'catalog.zip') as stream:
			assert set(stream.namelist()) == {
				'catalog/experiment_metrics.csv',
				'catalog/index.html',
				'catalog/manifest.json',
			}
			assert stream.testzip() is None
