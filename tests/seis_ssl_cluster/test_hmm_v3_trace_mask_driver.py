"""Exact v02 inheritance, safe command planning and v3 restart behavior."""

from __future__ import annotations

import argparse
import copy
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster import training
from seis_ssl_cluster.hmm import trace_mask_driver as driver
from seis_ssl_cluster.hmm.trace_mask_definition import (
	CANDIDATE_IDS,
	definition_path,
	read_yaml,
	validate_trace_mask_definition,
)
from seis_ssl_cluster.training import strat_hmm_checkpoint, strat_hmm_source_audit

WORKSPACE = Path(__file__).resolve().parents[2]
ROOTS = tuple(
	sorted(WORKSPACE.glob('experiments/*/*/*hmm_v3_k6810_trace_mask_multi_source_v1'))
)


def _args(**values):
	return argparse.Namespace(
		**{
			'first': 'smoke',
			'last': 'summary',
			'candidate': None,
			'layout': None,
			'size': None,
			**values,
		}
	)


@pytest.mark.parametrize('root', ROOTS, ids=lambda path: path.parts[-3])
def test_three_new_arms_and_exact_45_cell_plan(root):
	plan = driver.command_plan(root, _args())
	assert len(plan) == 58
	assert sum(stage == 'smoke' for stage, _ in plan) == 3
	assert sum(stage == 'full' for stage, _ in plan) == 3
	assert sum(stage == 'downstream' for stage, _ in plan) == 45
	assert sum(stage == 'audit' for stage, _ in plan) == 3
	assert max(
		index for index, (stage, _) in enumerate(plan) if stage == 'audit'
	) < next(index for index, (stage, _) in enumerate(plan) if stage == 'embeddings')
	assert not any('hmm_v2_mh' in ' '.join(command) for _, command in plan)
	assert [
		stage for stage, _ in driver.command_plan(root, _args(first='summary'))
	] == ['audit'] * 3 + ['summary']


def test_candidate_and_cell_narrowing_preserve_source_audit():
	candidate = CANDIDATE_IDS['mae']
	plan = driver.command_plan(
		ROOTS[0],
		_args(
			first='downstream',
			last='downstream',
			candidate=candidate,
			layout='layout_002',
			size='small',
		),
	)
	assert [stage for stage, _ in plan] == ['audit', 'downstream']
	assert candidate in plan[-1][1]
	assert 'layout_002' in plan[-1][1]
	assert 'small' in plan[-1][1]


@pytest.mark.parametrize(
	'values',
	[
		{'first': 'full', 'last': 'smoke'},
		{'layout': 'layout_000'},
		{'candidate': 'unknown'},
		{'candidate': CANDIDATE_IDS['mae']},
	],
)
def test_invalid_or_partial_summary_plan_is_rejected(values):
	with pytest.raises(
		ValueError, match=r'reversed|downstream only|unknown v3|all three'
	):
		driver.command_plan(ROOTS[0], _args(**values))


@pytest.fixture
def copied_definition(tmp_path):
	root = ROOTS[0]
	destination = tmp_path / root.relative_to(WORKSPACE)
	shutil.copytree(root, destination)
	definition = read_yaml(root / 'execution.yaml')
	paths = [definition_path(definition['canonical_selection']['path'], root)]
	for cell in definition['cells']:
		paths.extend(
			definition_path(cell[field], root)
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
	'change',
	[
		'loss',
		'seed',
		'mask',
		'parent',
		'decoder',
		'embedding',
		'baseline',
		'tag',
		'manifest',
		'selection',
	],
)
def test_definition_rejects_scientific_and_baseline_drift(copied_definition, change):
	root = copied_definition
	definition = read_yaml(root / 'execution.yaml')
	cell = definition['cells'][0]
	path = definition_path(cell['train_full_config'], root)
	raw = read_yaml(path)
	if change == 'loss':
		raw['loss']['consistency_weight'] = 0.0
	elif change == 'seed':
		raw['train']['seed'] = 43
	elif change == 'mask':
		raw['spatial_context']['column_fraction'] = 0.2
	elif change == 'parent':
		raw['student']['init_checkpoint'] = raw['paths']['output_root'] + '/latest.pt'
	elif change == 'decoder':
		path = definition_path(cell['downstream_config'], root)
		raw = read_yaml(path)
		raw['benchmark']['canonical_config'] += '.changed'
	elif change == 'embedding':
		path = definition_path(cell['embedding_config'], root)
		raw = read_yaml(path)
		raw['embeddings']['checkpoint'] += '.changed'
	elif change == 'baseline':
		path = definition_path(cell['baseline_train_config'], root)
		raw = read_yaml(path)
		raw['train']['seed'] = 43
	elif change == 'tag':
		raw['identity']['model_tag'] = 'foreign_model'
	elif change == 'manifest':
		raw['identity']['scientific_identity']['target_manifest_sha256'] = '0' * 64
	else:
		path = definition_path(definition['canonical_selection']['path'], root)
		raw = read_yaml(path)
		raw['selected_experiment'] = 'k6810_multi_source_evaluation_v1'
	path.write_text(yaml.safe_dump(raw), encoding='utf-8')
	with pytest.raises(ValueError, match=r'drift|inheritance'):
		validate_trace_mask_definition(root)


@pytest.fixture
def restart(tmp_path, monkeypatch):
	config = {
		'paths': {'output_root': str(tmp_path)},
		'train': {
			'epochs': 25,
			'samples_per_epoch': 10,
			'batch_size': 4,
			'max_steps': None,
		},
	}
	(tmp_path / 'latest.pt').touch()
	payload = {
		'stratigraphy_config': copy.deepcopy(config),
		'epoch': 4,
		'global_step': 12,
	}
	monkeypatch.setattr(driver, 'load_config', lambda _: config)
	monkeypatch.setattr(driver, 'resolve_strat_hmm_pretext_config', lambda raw: raw)
	monkeypatch.setattr(training, 'load_checkpoint', lambda *_args, **_kwargs: payload)
	monkeypatch.setattr(
		strat_hmm_checkpoint,
		'validate_stratigraphy_checkpoint_payload',
		lambda *_args, **_kwargs: None,
	)
	monkeypatch.setattr(
		strat_hmm_checkpoint,
		'inspect_stratigraphy_checkpoint',
		lambda *_args, **_kwargs: {'resume_compatibility': {'compatible': True}},
	)
	audits = []
	monkeypatch.setattr(
		strat_hmm_source_audit,
		'audit_multi_head_source',
		audits.append,
	)
	return config, payload, audits


def test_restart_resumes_only_owned_partial_checkpoint(restart, tmp_path):
	_config, _payload, audits = restart
	assert driver.training_restart(tmp_path / 'config.yaml') == (
		False,
		tmp_path / 'latest.pt',
	)
	assert not audits


def test_restart_complete_full_requires_source_audit(restart, tmp_path):
	_config, payload, audits = restart
	payload.update(epoch=25, global_step=75)
	assert driver.training_restart(tmp_path / 'config.yaml') == (
		True,
		tmp_path / 'latest.pt',
	)
	assert audits == [tmp_path / 'config.yaml']


def test_restart_rejects_foreign_configuration(restart, tmp_path):
	_config, payload, _audits = restart
	payload['stratigraphy_config']['paths']['output_root'] = str(tmp_path / 'foreign')
	with pytest.raises(ValueError, match='identity drift'):
		driver.training_restart(tmp_path / 'config.yaml')


def test_command_dispatch_preserves_spaces_and_stops_failure(tmp_path):
	log = tmp_path / 'log with spaces.txt'
	driver.run_command(
		'audit',
		[
			driver.sys.executable,
			'-c',
			'import sys; print(sys.argv[1])',
			'path with spaces',
		],
		dry_run=False,
		log=log,
	)
	assert log.read_text().splitlines()[-1] == 'path with spaces'
	with pytest.raises(subprocess.CalledProcessError):
		driver.run_command(
			'audit',
			[driver.sys.executable, '-c', 'raise SystemExit(7)'],
			dry_run=False,
			log=None,
		)


def test_training_dispatch_auto_resumes_owned_latest(tmp_path, monkeypatch):
	manifest = tmp_path / 'manifest.json'
	manifest.write_text('{}', encoding='utf-8')
	config = tmp_path / 'train.yaml'
	config.write_text(
		yaml.safe_dump({'pseudo_targets': {'manifest': str(manifest)}}),
		encoding='utf-8',
	)
	latest = tmp_path / 'latest.pt'
	monkeypatch.setattr(driver, 'training_restart', lambda _: (False, latest))
	calls = []
	monkeypatch.setattr(
		driver.subprocess, 'run', lambda command, **_kwargs: calls.append(command)
	)
	driver.run_command(
		'full', ['python', 'train', '--config', str(config)], dry_run=False, log=None
	)
	assert calls[0][-2:] == ['--resume', str(latest)]
	monkeypatch.setattr(driver, 'training_restart', lambda _: (True, latest))
	driver.run_command(
		'full', ['python', 'train', '--config', str(config)], dry_run=False, log=None
	)
	assert len(calls) == 1
