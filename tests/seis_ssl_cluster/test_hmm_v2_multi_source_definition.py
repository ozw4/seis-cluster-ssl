"""Production preflight rejects source-family drift before any live work."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.hmm import multi_source_driver as driver
from seis_ssl_cluster.hmm import multi_source_survey as summary
from seis_ssl_cluster.hmm.multi_source_definition import (
	validate_multi_source_experiment_definition,
)
from tests.seis_ssl_cluster.test_hmm_v2_multi_source_driver import _args

WORKSPACE = Path(__file__).resolve().parents[2]
SURVEYS = ('f3', 'parihaka', 'volve')
MATRIX = Path('experiments/hmm_v2/k6810_multi_source_evaluation_v1/matrix.yaml')


def read(path):
	return yaml.safe_load(path.read_text())


def write(path, value):
	path.write_text(yaml.safe_dump(value, sort_keys=False))


@pytest.fixture(params=SURVEYS)
def experiment(request, tmp_path, monkeypatch):
	survey = request.param
	original = next(
		WORKSPACE.glob(f'experiments/{survey}/*/*hmm_v2_k6810_multi_source_v1')
	)
	workspace = tmp_path / 'workspace with spaces'
	root = workspace / original.relative_to(WORKSPACE)
	shutil.copytree(original, root)
	paths = {MATRIX}
	for arm in read(original / 'execution.yaml')['arms'].values():
		paths.update(
			Path(arm[f'{s}_reference']) for s in ('clustering', 'training', 'embedding')
		)
	for relative in paths:
		path = workspace / relative
		path.parent.mkdir(parents=True, exist_ok=True)
		shutil.copyfile(WORKSPACE / relative, path)
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(workspace))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(workspace / 'artifacts'))
	return root


def training_path(root, family='random'):
	arm = read(root / 'execution.yaml')['arms'][family]
	return root / '30_pretraining' / arm['candidate_id'] / '02_full_25ep.yaml'


def snapshot(root):
	return {
		str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()
	}


def test_preflight_validates_without_environment_or_artifacts(experiment, monkeypatch):
	for variable in (
		'SEIS_SSL_CLUSTER_WORKSPACE',
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256',
		'SEIS_SSL_CLUSTER_VOLVE_ROOT',
	):
		monkeypatch.delenv(variable, raising=False)
	before = snapshot(experiment.parents[3])
	validate_multi_source_experiment_definition(experiment)
	assert snapshot(experiment.parents[3]) == before


@pytest.mark.parametrize(
	('section', 'field', 'value'),
	[
		('teacher', 'checkpoint', 'wrong-mae-parent.pt'),
		('student', 'init_checkpoint', 'wrong-mae-parent.pt'),
		('data', 'agc', {'enabled': False}),
		('zero_mask', 'enabled', False),
		('model', 'encoder_depth', 7),
		('manifests', 'train', 'another-survey.json'),
		('train', 'batch_size', 9),
		('train', 'num_workers', 17),
	],
)
def test_preflight_rejects_training_source_drift(experiment, section, field, value):
	path = training_path(experiment)
	config = read(path)
	config[section][field] = value
	write(path, config)
	with pytest.raises(ValueError, match='inheritance drift'):
		validate_multi_source_experiment_definition(experiment)


@pytest.mark.parametrize('stage', ['cluster', 'replay', 'embedding', 'smoke'])
def test_preflight_rejects_other_stage_drift(experiment, stage):
	arm = read(experiment / 'execution.yaml')['arms']['random']
	cid, family = arm['candidate_id'], arm['target_family']
	if stage in ('cluster', 'replay'):
		name = (
			'01_cluster_hmm_k8k10.yaml'
			if stage == 'cluster'
			else '02_replay_hmm_k6.yaml'
		)
		path = experiment / '10_targets' / family / name
		config = read(path)
		config['clustering']['transition'] = 'changed'
	elif stage == 'embedding':
		path = experiment / '40_embeddings' / f'{cid}.yaml'
		config = read(path)
		config['embedding']['overlap'] = [0, 0, 0]
	else:
		path = training_path(experiment).with_name('01_gpu_feasibility_1step.yaml')
		config = read(path)
		config['train']['num_workers'] = 17
	write(path, config)
	with pytest.raises(ValueError, match='inheritance drift'):
		validate_multi_source_experiment_definition(experiment)


@pytest.mark.parametrize(
	'change',
	[
		'schema',
		'bool_schema',
		'survey',
		'arms',
		'candidate',
		'target',
		'missing_reference',
		'absolute_reference',
		'escape_reference',
	],
)
def test_preflight_rejects_invalid_execution(experiment, change):
	path = experiment / 'execution.yaml'
	config = read(path)
	arm = config['arms']['random']
	if change == 'schema':
		config['schema_version'] = 2
	elif change == 'bool_schema':
		config['schema_version'] = True
	elif change == 'survey':
		config['survey'] = 'unknown'
	elif change == 'arms':
		config['arms'].pop('random')
	elif change == 'candidate':
		arm['candidate_id'] = 'mae'
	elif change == 'target':
		arm['target_family'] = 'mae100'
	elif change == 'missing_reference':
		arm['training_reference'] = f'experiments/{config["survey"]}/missing.yaml'
	elif change == 'absolute_reference':
		arm['training_reference'] = str(
			experiment.parents[3] / arm['training_reference']
		)
	else:
		arm['training_reference'] = '../reference.yaml'
	write(path, config)
	with pytest.raises(
		(ValueError, FileNotFoundError), match=r'execution|No such file'
	):
		validate_multi_source_experiment_definition(experiment)


@pytest.mark.parametrize('stage', ['driver', 'summary'])
def test_both_entrypoints_fail_before_live_work(experiment, monkeypatch, stage):
	path = training_path(experiment)
	config = read(path)
	config['student']['init_checkpoint'] = 'wrong-parent.pt'
	write(path, config)
	before = snapshot(experiment.parents[3])
	if stage == 'driver':
		argv = ['driver', '--experiment', str(experiment), '--execute']
		entrypoint = driver.main
	else:
		argv = [
			'summary',
			'--config',
			str(experiment / '60_summary/01_paired_comparison.yaml'),
		]
		entrypoint = summary.main
	monkeypatch.setattr(sys, 'argv', argv)
	with pytest.raises(ValueError, match='inheritance drift'):
		entrypoint()
	assert snapshot(experiment.parents[3]) == before


@pytest.mark.parametrize('field', ['training_config', 'downstream_config'])
def test_summary_cannot_audit_unvalidated_alternate_configs(experiment, field):
	config = load_config(experiment / '60_summary/01_paired_comparison.yaml')
	config['arms']['random'][field] = str(experiment / 'alternate.yaml')
	with pytest.raises(ValueError, match='summary inputs differ'):
		summary.inspect_survey(config)
	assert not (experiment.parents[3] / 'artifacts').exists()


def test_validated_definition_retains_exact_plan(experiment):
	plan = driver.command_plan(experiment, _args())
	count = 2 if read(experiment / 'execution.yaml')['survey'] == 'f3' else 3
	assert sum(stage == 'full' for stage, _ in plan) == count
	assert sum(stage == 'downstream' for stage, _ in plan) == count * 15


def test_preflight_rejects_shared_target_and_replay_outputs(experiment):
	arm = read(experiment / 'execution.yaml')['arms']['random']
	root = experiment / '10_targets' / arm['target_family']
	path = root / '02_replay_hmm_k6.yaml'
	config = read(path)
	config['clustering']['output_dir'] = read(root / '01_cluster_hmm_k8k10.yaml')[
		'clustering'
	]['output_dir']
	write(path, config)
	with pytest.raises(ValueError, match='require separate outputs'):
		validate_multi_source_experiment_definition(experiment)


@pytest.mark.parametrize('field', ['manifests', 'paths', 'embeddings'])
def test_embedding_inheritance_preserves_inputs_and_checkpoint(experiment, field):
	arm = read(experiment / 'execution.yaml')['arms']['random']
	path = experiment / '40_embeddings' / f'{arm["candidate_id"]}.yaml'
	config = read(path)
	key = {'manifests': 'input', 'paths': 'artifact_root', 'embeddings': 'checkpoint'}[
		field
	]
	config[field][key] = 'wrong-source'
	write(path, config)
	with pytest.raises(ValueError, match='inheritance drift'):
		validate_multi_source_experiment_definition(experiment)
