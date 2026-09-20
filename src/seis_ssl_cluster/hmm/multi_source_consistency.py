"""Exact continuation definitions for the fixed consistency-weight ablation."""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Any

import yaml

from seis_ssl_cluster.hmm.multi_source_protocol import (
	experiment_protocol,
	protocol_for_heads,
)
from seis_ssl_cluster.hmm.multi_source_receipts import (
	file_sha256,
	validate_fixed_training_config,
)

if TYPE_CHECKING:
	from pathlib import Path

BASELINE_NUMBERS = {
	'f3': {'k468': 129, 'k6810': 128},
	'parihaka': {'k468': 42, 'k6810': 41},
	'volve': {'k468': 35, 'k6810': 34},
}


def baseline_experiment(root: Path, family: str) -> Path:
	"""Return the fixed zero-consistency source, including the F3 screening arm."""
	protocol = experiment_protocol(root)
	survey = root.parents[1].name
	if (survey, family) == ('f3', 'mae'):
		return root.parent / '127_hmm_v2_multi_head_screening_v1'
	number = BASELINE_NUMBERS[survey][protocol.tag]
	return root.parent / f'{number}_hmm_v2_{protocol.tag}_multi_source_v1'


def baseline_paths(root: Path, family: str) -> dict[str, Path]:
	"""Identify the three inherited scientific definitions."""
	protocol = experiment_protocol(root)
	cid = protocol_for_heads(protocol.head_ks).candidate_ids[family]
	baseline = baseline_experiment(root, family)
	return {
		'training': baseline / '30_pretraining' / cid / '02_full_25ep.yaml',
		'embeddings': baseline / '40_embeddings' / f'{cid}.yaml',
		'downstream': baseline / '50_downstream' / f'{cid}.yaml',
	}


def _replace(value: object, replacements: dict[str, str]) -> object:
	if isinstance(value, dict):
		return {
			_replace(k, replacements): _replace(v, replacements)
			for k, v in value.items()
		}
	if isinstance(value, list):
		return [_replace(v, replacements) for v in value]
	return replacements.get(value, value) if isinstance(value, str) else value


def consistency_arm_configs(
	root: Path, family: str, binding: dict[str, Any]
) -> dict[str, dict[str, Any]]:
	"""Derive configs with only weight, identity and owned outputs changed.

	The versioned binding freezes baseline config bytes and the existing target
	manifest. No clustering, target export or completed model reuse is permitted.
	"""
	protocol = experiment_protocol(root)
	if protocol.consistency_weight != 0.1:
		raise ValueError('expected a consistency 0.1 experiment')
	workspace = root.parents[3]
	paths = baseline_paths(root, family)
	cid = protocol.candidate_ids[family]
	if (
		set(binding)
		!= {'candidate_id', 'baseline_configs_sha256', 'target_manifest_sha256'}
		or binding['candidate_id'] != cid
		or set(binding['baseline_configs_sha256']) != set(paths)
		or not isinstance(binding['target_manifest_sha256'], str)
		or re.fullmatch('[0-9a-f]{64}', binding['target_manifest_sha256']) is None
	):
		raise ValueError('consistency baseline binding contract drift')
	baseline = {}
	for stage, path in paths.items():
		if file_sha256(path) != binding['baseline_configs_sha256'][stage]:
			raise ValueError(f'consistency baseline config hash drift: {path}')
		baseline[stage] = yaml.safe_load(path.read_text())
	validate_fixed_training_config(
		baseline['training'], protocol_for_heads(protocol.head_ks)
	)
	survey, series = root.parts[-3:-1]
	artifact = '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/surveys'
	training_series = 'facies_benchmark_v1' if survey == 'f3' else series
	output = (
		f'{artifact}/{survey}/{training_series}/pretraining/{protocol.namespace}/{cid}'
	)
	embedding = (
		f'{artifact}/{survey}/{series}/embeddings/'
		f'{protocol.namespace}/{cid}/overlap_x64'
	)
	benchmark = f'{artifact}/{survey}/{series}/benchmark/{protocol.namespace}'
	full = copy.deepcopy(baseline['training'])
	full['paths']['output_root'] = output + '/full_25ep'
	full['loss']['consistency_weight'] = 0.1
	full['identity']['model_tag'] = cid
	full['identity']['scientific_identity'].update(
		variant='cons010', target_manifest_sha256=binding['target_manifest_sha256']
	)
	smoke = copy.deepcopy(full)
	smoke['paths']['output_root'] = output + '/gpu_feasibility_1step'
	smoke['train'].update(epochs=1, max_steps=1)
	training_file = f'30_pretraining/{cid}/02_full_25ep.yaml'
	training_path = '${SEIS_SSL_CLUSTER_WORKSPACE}/' + str(
		root.relative_to(workspace) / training_file
	)
	embeddings = copy.deepcopy(baseline['embeddings'])
	embeddings['embeddings'].update(
		checkpoint=output + '/full_25ep/latest.pt', output_dir=embedding
	)
	down = baseline['downstream']
	replacements = {
		baseline['training']['identity']['model_tag']: cid,
		baseline['embeddings']['embeddings']['checkpoint']: output
		+ '/full_25ep/latest.pt',
		baseline['embeddings']['embeddings']['output_dir']: embedding,
		down['outputs']['runs_root']: benchmark + '/runs',
		'${SEIS_SSL_CLUSTER_WORKSPACE}/'
		+ str(paths['training'].relative_to(workspace)): training_path,
	}
	for field in ('summary_root', 'output_dir'):
		if field in down['outputs']:
			replacements[down['outputs'][field]] = benchmark + '/summary/' + cid
	return {
		training_file: full,
		f'30_pretraining/{cid}/01_gpu_feasibility_1step.yaml': smoke,
		f'30_pretraining/{cid}/03_audit.yaml': {'training_configs': [training_path]},
		f'40_embeddings/{cid}.yaml': embeddings,
		f'50_downstream/{cid}.yaml': _replace(down, replacements),
	}


def validate_consistency_definition(root: Path, definition: dict[str, Any]) -> None:
	"""Reject output overlap, stale targets, source drift and decoder changes."""
	for family, binding in definition['arms'].items():
		for relative, expected in consistency_arm_configs(
			root, family, binding
		).items():
			path = root / relative
			if yaml.safe_load(path.read_text()) != expected:
				raise ValueError(f'{path}: consistency baseline inheritance drift')
	for stage in ('10_targets', '20_manifests'):
		if (root / stage).exists():
			raise ValueError('consistency study must retain existing targets read-only')
