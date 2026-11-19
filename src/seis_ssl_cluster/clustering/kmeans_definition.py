"""Matched nine-arm definitions for the unordered k-means ablation."""

# ruff: noqa: PLC0415

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import yaml

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.clustering.kmeans_targets import (
	CONSISTENCY_POLICY,
	TARGET_REPRESENTATION,
)

if TYPE_CHECKING:
	from pathlib import Path

STUDY = 'kmeans_k6810_multi_source_v1'
SOURCES = ('mae', 'local_bt', 'random')
SURVEYS = ('f3', 'parihaka', 'volve')
SIZES = ('small', 'medium', 'large')
LAYOUTS = tuple(f'layout_{i:03d}' for i in range(5))


def read_yaml(path: Path) -> dict[str, Any]:
	"""Read a versioned definition without environment expansion."""
	if 'reports' in path.resolve().parts:
		raise ValueError('reports cannot be pipeline inputs')
	return yaml.safe_load(path.read_text())


def _replace(value: Any, replacements: dict[str, str]) -> Any:  # noqa: ANN401
	if isinstance(value, dict):
		return {
			_replace(k, replacements): _replace(v, replacements)
			for k, v in value.items()
		}
	if isinstance(value, list):
		return [_replace(v, replacements) for v in value]
	if isinstance(value, str):
		for old, new in replacements.items():
			value = value.replace(old, new)
	return value


def arm_configs(
	workspace: Path, survey: str, arm: dict[str, Any]
) -> dict[str, dict[str, Any]]:
	"""Derive the only permitted changes from bound canonical HMM v2 configs."""
	baseline = {}
	for stage, ref in arm['references'].items():
		path = workspace / ref['path']
		if file_sha256(path) != ref['sha256']:
			raise ValueError(f'canonical reference hash drift: {path}')
		baseline[stage] = read_yaml(path)
	cid = arm['candidate_id']
	control = arm['control_id']
	if cid != control.replace('hmm_v2', 'kmeans').removesuffix('_cons010'):
		raise ValueError('k-means candidate identity drift')
	replacements = {
		# Replace the experiment path before changing its namespace substring.
		arm['reference_experiment']: arm['experiment'],
		control: cid,
		'hmm_v2_k6810_cons010_multi_source_v1': STUDY,
	}
	result = {
		stage: _replace(copy.deepcopy(raw), replacements)
		for stage, raw in baseline.items()
		if stage != 'targets'
	}
	series = 'facies_benchmark_v1' if survey != 'volve' else 'horizon_benchmark_v1'
	target_root = (
		'${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/surveys/'
		+ survey
		+ '/'
		+ series
		+ '/pseudo_targets/'
		+ STUDY
		+ '/'
		+ cid
	)
	full = result['training']
	full['pseudo_targets'] = {
		'manifest': target_root + '/multi_head_target_manifest.json',
		'target_representation': TARGET_REPRESENTATION,
		'min_confidence': 0.0,
	}
	full['loss']['consistency_weight'] = 0.0
	full['identity']['scientific_identity'] = {
		'experiment_role': 'multi_head_unordered_kmeans_pretext',
		'variant': 'kmeans_nocons',
		'head_spec': full['head']['spec'],
		'head_ks': [6, 8, 10],
		'target_representation': TARGET_REPRESENTATION,
		'target_manifest_sha256': (
			'${SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256}'
		),
		'consistency_policy': CONSISTENCY_POLICY,
	}
	smoke = copy.deepcopy(full)
	smoke['paths']['output_root'] = full['paths']['output_root'].replace(
		'/full_25ep', '/smoke_1step'
	)
	smoke['train'].update(epochs=1, max_steps=1)
	result['smoke'] = smoke
	targets = copy.deepcopy(baseline['targets'])
	targets['output_root'] = target_root
	targets['edge_margin_tokens'] = targets['clustering'].pop('stratigraphic_hmm')[
		'edge_margin_tokens'
	]
	targets['clustering'].update(
		method='minibatch_kmeans',
		k_values=[6, 8, 10],
		output_dir=target_root + '/clustering',
	)
	result['targets'] = targets
	return result


def validate_study(matrix_path: Path, workspace: Path) -> dict[str, Any]:  # noqa: C901
	"""Reject missing arms, scientific drift, and candidate/output collisions."""
	matrix = read_yaml(matrix_path)
	if matrix.get('study') == 'kmeans_k6810_dis0_con0_v1':
		from seis_ssl_cluster.clustering.kmeans_zero import (
			validate_definition,
		)

		return validate_definition(matrix, workspace)
	if (
		matrix.get('schema_version') != 1
		or matrix.get('study') != STUDY
		or set(matrix['surveys']) != set(SURVEYS)
	):
		raise ValueError('invalid k-means study matrix')
	outputs = set()
	for survey in SURVEYS:
		arms = matrix['surveys'][survey]
		if set(arms) != set(SOURCES):
			raise ValueError('each survey requires exactly three source families')
		for arm in arms.values():
			expected = arm_configs(workspace, survey, arm)
			if set(arm['configs']) != set(expected):
				raise ValueError('k-means stage set mismatch')
			for stage, value in expected.items():
				path = workspace / arm['configs'][stage]
				if read_yaml(path) != value:
					raise ValueError(f'k-means definition drift: {path}')
			for stage in ('training', 'smoke'):
				output = expected[stage]['paths']['output_root']
				if output in outputs:
					raise ValueError('k-means output collision')
				outputs.add(output)
	return matrix
