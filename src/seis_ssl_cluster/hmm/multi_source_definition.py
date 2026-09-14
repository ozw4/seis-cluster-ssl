"""Read-only validation of source-family inheritance in fixed K6810 definitions."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from seis_ssl_cluster.hmm.multi_source_receipts import (
	SURVEYS,
	load_matrix,
	validate_fixed_training_config,
)

TARGET_FAMILIES = {'mae': 'mae100', 'local_bt': 'local_bt3', 'random': 'random'}


def _read(path: Path) -> dict[str, Any]:
	value = yaml.safe_load(path.read_text(encoding='utf-8'))
	if not isinstance(value, dict):
		raise TypeError(f'{path}: expected a YAML mapping')
	return value


def _compare(actual: dict[str, Any], expected: dict[str, Any], path: Path) -> None:
	if actual != expected:
		changed = sorted(
			k
			for k in actual.keys() | expected.keys()
			if k not in actual or k not in expected or actual[k] != expected[k]
		)
		raise ValueError(f'{path}: source reference inheritance drift in {changed}')


def _output(
	actual: dict[str, Any], reference: dict[str, Any], section: str, key: str
) -> None:
	value = actual[section][key]
	if not isinstance(value, str) or not value or value == reference[section][key]:
		raise ValueError(f'{section}.{key} must use a separate nonempty output path')


def _reference(workspace: Path, survey: str, value: object) -> dict[str, Any]:
	if not isinstance(value, str) or not value:
		raise ValueError('execution reference must be a repository-relative file')
	path = Path(value)
	if (
		path.is_absolute()
		or '..' in path.parts
		or not (workspace / path)
		.resolve()
		.is_relative_to(workspace / 'experiments' / survey)
	):
		raise ValueError(
			'execution reference must stay in its survey experiment namespace'
		)
	return _read(workspace / path)


def _training(root: Path, cid: str, reference: dict[str, Any]) -> dict[str, Any]:
	path = root / '30_pretraining' / cid / '02_full_25ep.yaml'
	full = _read(path)
	validate_fixed_training_config(full)
	_output(full, reference, 'paths', 'output_root')
	expected = copy.deepcopy(reference)
	expected['paths']['output_root'] = full['paths']['output_root']
	for section in ('pseudo_targets', 'head', 'loss', 'identity'):
		expected[section] = full[section]
	_compare(full, expected, path)
	smoke_path = path.with_name('01_gpu_feasibility_1step.yaml')
	smoke = _read(smoke_path)
	_output(smoke, full, 'paths', 'output_root')
	_output(smoke, reference, 'paths', 'output_root')
	expected = copy.deepcopy(full)
	expected['paths']['output_root'] = smoke['paths']['output_root']
	expected['train'].update(epochs=1, max_steps=1)
	_compare(smoke, expected, smoke_path)
	return full


def _targets(
	root: Path, arm: dict[str, Any], cluster: dict[str, Any], train: dict[str, Any]
) -> None:
	outputs = set()
	for name, ks in (
		('01_cluster_hmm_k8k10.yaml', [8, 10]),
		('02_replay_hmm_k6.yaml', [6]),
	):
		path = root / '10_targets' / arm['target_family'] / name
		actual = _read(path)
		_output(actual, cluster, 'clustering', 'output_dir')
		output = actual['clustering']['output_dir']
		if output in outputs:
			raise ValueError('target and replay clustering require separate outputs')
		outputs.add(output)
		expected = copy.deepcopy(cluster)
		expected['clustering'].update(
			k_values=ks, output_dir=actual['clustering']['output_dir']
		)
		_compare(actual, expected, path)
	path = root / '20_manifests' / f'{arm["candidate_id"]}.yaml'
	manifest = _read(path)
	if (
		list(manifest['head_roots']) != [6, 8, 10]
		or manifest['head_roots'][6] != train['pseudo_targets']['input_dir']
		or manifest['head_roots'][6] == manifest['replay_k6_root']
		or manifest['source_embedding_dir'] != cluster['embeddings']['input_dir']
	):
		raise ValueError(f'{path}: manifest source reference inheritance drift')


def validate_multi_source_experiment_definition(experiment_root: Path) -> None:
	"""Reject scientific or family drift before any commands or live artifact reads.

	Compare raw YAML so preflight needs neither environment expansion nor the
	future target manifests produced by this experiment.
	"""
	root = experiment_root.resolve()
	definition = _read(root / 'execution.yaml')
	survey = definition.get('survey')
	if (
		set(definition) != {'schema_version', 'survey', 'arms'}
		or type(definition.get('schema_version')) is not int
		or definition['schema_version'] != 1
		or survey not in SURVEYS
		or root.parents[1].name != survey
		or root.parents[2].name != 'experiments'
	):
		raise ValueError('invalid execution schema_version or survey namespace')
	workspace = root.parents[3]
	matrix = load_matrix(
		workspace / 'experiments/hmm_v2/k6810_multi_source_evaluation_v1/matrix.yaml'
	)
	expected_arms = {
		f: a
		for f, a in matrix['surveys'][survey]['arms'].items()
		if a['execution'] == 'new'
	}
	arms = definition['arms']
	if not isinstance(arms, dict) or set(arms) != set(expected_arms):
		raise ValueError('execution arms differ from the fixed new-arm matrix')
	for family, arm in arms.items():
		if (
			not isinstance(arm, dict)
			or set(arm)
			!= {
				'candidate_id',
				'target_family',
				'clustering_reference',
				'training_reference',
				'embedding_reference',
			}
			or arm['candidate_id'] != expected_arms[family]['candidate_id']
			or arm['target_family'] != TARGET_FAMILIES[family]
		):
			raise ValueError(
				'execution candidate_id or target_family differs from matrix'
			)
		references = {
			stage: _reference(workspace, survey, arm[f'{stage}_reference'])
			for stage in ('clustering', 'training', 'embedding')
		}
		cid = arm['candidate_id']
		full = _training(root, cid, references['training'])
		_targets(root, arm, references['clustering'], references['training'])
		path = root / '40_embeddings' / f'{cid}.yaml'
		embedding = _read(path)
		reference = references['embedding']
		_output(embedding, reference, 'embeddings', 'output_dir')
		expected = copy.deepcopy(reference)
		expected['embeddings']['checkpoint'] = (
			full['paths']['output_root'] + '/latest.pt'
		)
		expected['embeddings']['output_dir'] = embedding['embeddings']['output_dir']
		_compare(embedding, expected, path)
