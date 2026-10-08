"""Survey-bound LR3 inheritance, candidate scoping and exact publication."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster.hmm import trace_mask_lr3 as followup
from seis_ssl_cluster.hmm.multi_source_receipts import CELL_IDENTITIES
from seis_ssl_cluster.hmm.trace_mask_definition import definition_path, read_yaml

WORKSPACE = Path(__file__).resolve().parents[2]
ROOTS = (
	WORKSPACE
	/ 'experiments/parihaka/facies_benchmark_v1'
	/ followup.EXPERIMENTS['parihaka'],
	WORKSPACE
	/ 'experiments/volve/horizon_benchmark_v1'
	/ followup.EXPERIMENTS['volve'],
)


def _args(**values):
	return argparse.Namespace(
		**{'first': 'smoke', 'last': 'summary', 'candidate': None, **values}
	)


@pytest.mark.parametrize('root', ROOTS)
def test_exact_definition_and_explicit_candidate_plan(root):
	definition = followup.validate_definition(root)
	assert len(list(root.rglob('*.yaml'))) == 16
	commands = followup.command_plan(root, _args())
	assert len(commands) == 58
	assert sum(stage == 'downstream' for stage, _command in commands) == 45
	for cell in definition['cells']:
		full = read_yaml(definition_path(cell['train_full_config'], root))
		assert full['train']['mask_token_lr'] == 3.0e-5
		assert full['train']['lr'] == full['train']['encoder_lr'] == 1.0e-5
		assert full['train']['epochs'] == 25
		assert full['train']['seed'] == 42
		assert full['spatial_context']['masked_prototype_weight'] == 0.5
		assert full['spatial_context']['visible_prototype_weight'] == 0.5
		assert full['loss']['consistency_weight'] == 0.1
		plan = followup.command_plan(root, _args(candidate=cell['candidate_id']))
		assert len(plan) == 20
		assert sum(stage == 'downstream' for stage, _command in plan) == 15
		assert '--candidate' in plan[-1][1]
		assert cell['candidate_id'] in plan[-1][1]
		assert all(str(root) in ' '.join(command) for _stage, command in plan)
		assert [
			stage
			for stage, _command in followup.command_plan(
				root, _args(first='summary', candidate=cell['candidate_id'])
			)
		] == ['audit', 'summary']


@pytest.mark.parametrize(
	'values', [{'first': 'full', 'last': 'smoke'}, {'candidate': 'unknown'}]
)
def test_invalid_stage_or_candidate_is_rejected(values):
	with pytest.raises(ValueError, match=r'reversed|unknown'):
		followup.command_plan(ROOTS[0], _args(**values))


@pytest.fixture(params=ROOTS)
def copied_definition(request, tmp_path, monkeypatch):
	root = request.param
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(tmp_path))
	destination = tmp_path / root.relative_to(WORKSPACE)
	shutil.copytree(root, destination)
	survey = read_yaml(root / 'execution.yaml')['survey']
	parent_root = root.parent / followup.PARENTS[survey]
	shutil.copytree(parent_root, tmp_path / parent_root.relative_to(WORKSPACE))
	parent = read_yaml(parent_root / 'execution.yaml')
	paths = [definition_path(parent['canonical_selection']['path'], parent_root)]
	for cell in parent['cells']:
		paths.extend(
			definition_path(cell[field], parent_root)
			for field in (
				'baseline_train_config',
				'baseline_embedding_config',
				'baseline_downstream_config',
			)
		)
	for path in paths:
		target = tmp_path / path.relative_to(WORKSPACE)
		target.parent.mkdir(parents=True, exist_ok=True)
		shutil.copyfile(path, target)
	return destination


@pytest.mark.parametrize(
	('section', 'field', 'value'),
	[
		('train', 'mask_token_lr', 1.0e-5),
		('train', 'encoder_lr', 3.0e-5),
		('train', 'lr', 3.0e-5),
		('train', 'seed', 43),
		('train', 'epochs', 24),
		('loss', 'consistency_weight', 0.0),
		('spatial_context', 'masked_prototype_weight', 0.75),
		('spatial_context', 'column_fraction', 0.2),
		('student', 'init_checkpoint', 'wrong-parent.pt'),
	],
)
def test_only_mask_learning_rate_changes(copied_definition, section, field, value):
	definition = followup.validate_definition(copied_definition)
	path = definition_path(
		definition['cells'][0]['train_full_config'], copied_definition
	)
	raw = read_yaml(path)
	raw[section][field] = value
	path.write_text(yaml.safe_dump(raw))
	with pytest.raises(ValueError, match='other than mask-token LR'):
		followup.validate_definition(copied_definition)


@pytest.mark.parametrize(
	'change', ['parent_hash', 'baseline', 'target', 'selection', 'count', 'candidate']
)
def test_frozen_provenance_and_closed_matrix(copied_definition, change):
	path = copied_definition / 'execution.yaml'
	raw = read_yaml(path)
	if change == 'parent_hash':
		raw['parent_definition_sha256'] = '0' * 64
	elif change == 'baseline':
		raw['cells'][0]['baseline_checkpoint_sha256'] = '0' * 64
	elif change == 'target':
		raw['cells'][0]['target_manifest_sha256'] = '0' * 64
	elif change == 'selection':
		raw['canonical_selection']['sha256'] = '0' * 64
	elif change == 'count':
		raw['cells'].pop()
	else:
		raw['cells'][0]['candidate_id'] = 'unknown'
	path.write_text(yaml.safe_dump(raw))
	with pytest.raises(
		ValueError, match=r'identity drift|binding drift|source families'
	):
		followup.validate_definition(copied_definition)


@pytest.mark.parametrize(
	('field', 'section', 'key'),
	[
		('train_smoke_config', 'train', 'seed'),
		('embedding_config', 'embeddings', 'checkpoint'),
		('downstream_config', 'outputs', 'runs_root'),
	],
)
def test_stage_handoff_drift_is_rejected(copied_definition, field, section, key):
	definition = followup.validate_definition(copied_definition)
	path = definition_path(definition['cells'][0][field], copied_definition)
	raw = read_yaml(path)
	raw[section][key] = 'wrong'
	path.write_text(yaml.safe_dump(raw))
	with pytest.raises(ValueError, match=r'inheritance drift|owned survey'):
		followup.validate_definition(copied_definition)


@pytest.fixture(params=ROOTS)
def completed(request, monkeypatch):
	root = request.param
	survey = followup.validate_definition(root)['survey']
	monkeypatch.setattr(followup, 'validate_canonical_evidence', lambda *_args: None)

	def audited(_experiment, observed_survey, cell):
		assert observed_survey == survey
		primary, control = (6.0, 7.0) if survey == 'volve' else (0.6, 0.5)
		return [
			{
				'candidate_id': cell['candidate_id'],
				'baseline_id': 'canonical-v02',
				'layout_id': layout,
				'data_size': size,
				'primary': primary,
				'control_primary': control,
				'secondary': 0.7,
				'control_secondary': 0.6,
			}
			for layout, size in CELL_IDENTITIES
		], {'audited': True}

	monkeypatch.setattr(followup, '_audited_arm', audited)
	return root, survey


def test_candidate_summary_is_not_a_complete_three_source_matrix(completed):
	root, survey = completed
	payload = followup.inspect_summary(root, followup.CANDIDATE_IDS['mae'])
	assert payload['status'] == 'complete_candidate'
	assert len(payload['cells']) == 15
	assert len(payload['rows']) == 3
	assert all(row['cell_count'] == 5 for row in payload['rows'])
	assert all(
		row['primary_improvement'] == pytest.approx(1.0 if survey == 'volve' else 0.1)
		for row in payload['rows']
	)
	full = followup.inspect_summary(root)
	assert full['status'] == 'complete'
	assert len(full['cells']) == 45
	assert len(full['rows']) == 9


@pytest.mark.parametrize('candidate', [None, followup.CANDIDATE_IDS['mae']])
def test_exact_publication_and_recovery(completed, candidate, tmp_path, monkeypatch):
	root, _survey = completed
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	paths = followup.summarize(root, candidate)
	assert {path.name for path in paths[0].parent.iterdir()} == set(
		followup.PUBLISHED_FILES
	)
	assert followup.summarize(root, candidate) == paths
	payload = json.loads((paths[0].parent / 'summary.json').read_text())
	assert len(payload['cells']) == (45 if candidate is None else 15)
	assert 'Mask-token LR=3e-5' in (paths[0].parent / 'summary.md').read_text()
	if candidate:
		assert paths[0].parent.name == candidate
	(paths[0].parent / 'extra.txt').touch()
	with pytest.raises(ValueError, match='publication drift'):
		followup.summarize(root, candidate)


def test_incomplete_candidate_cannot_publish(completed, tmp_path, monkeypatch):
	root, _survey = completed
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))

	def incomplete(*_args):
		return [], {}

	monkeypatch.setattr(followup, '_audited_arm', incomplete)
	with pytest.raises(ValueError, match='15 complete'):
		followup.summarize(root, followup.CANDIDATE_IDS['mae'])
	assert not list(tmp_path.rglob('summary.json'))
