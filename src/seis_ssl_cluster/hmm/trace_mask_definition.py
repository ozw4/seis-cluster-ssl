"""Frozen inheritance contracts for the K6810 trace-mask successor to v02."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from seis_ssl_cluster.hmm.multi_source_protocol import K6810_CONS010
from seis_ssl_cluster.hmm.multi_source_receipts import (
	SOURCE_FAMILIES,
	SURVEYS,
	file_sha256,
	validate_fixed_training_config,
)

NAMESPACE = 'hmm_v3_k6810_trace_mask_multi_source_v1'
CANDIDATE_IDS = {
	family: f'{source}_hmm_v3_mh_k6810_ctmask010_distill020_cons010'
	for family, source in (
		('mae', 'mae100'),
		('local_bt', 'local_bt_rot90_asym_g060_3ep'),
		('random', 'random_init'),
	)
}
MASK_CONTEXT = {
	'objective': 'center_trace_masked_hmm_path_reconstruction_v1',
	'mask_semantics': 'xy_token_column_full_z_v1',
	'column_fraction': 0.1,
	'selection_policy': 'supervised_valid_xy_columns_round_half_up_leave_one_v1',
	'replacement': 'learned_encoder_mask_token_v1',
	'replacement_initialization': 'normal_std_0p02_train_seed_salted_v1',
	'rng_policy': 'stateless_step_seed_v1',
	'masked_prototype_weight': 0.5,
	'visible_prototype_weight': 0.5,
	'distillation_scope': 'visible_only_v1',
}


def read_yaml(path: Path) -> dict[str, Any]:
	"""Read a YAML object and reject report inputs."""
	if 'reports' in path.resolve().parts:
		raise ValueError('reports cannot be pipeline inputs')
	value = yaml.safe_load(path.read_text(encoding='utf-8'))
	if not isinstance(value, dict):
		raise TypeError(f'expected YAML mapping: {path}')
	return value


def definition_path(value: str, experiment: Path) -> Path:
	"""Resolve versioned references without requiring live artifact variables."""
	workspace = experiment.resolve().parents[3]
	value = value.replace('${SEIS_SSL_CLUSTER_WORKSPACE}', str(workspace))
	path = Path(os.path.expandvars(value))
	if not path.is_absolute():
		path = workspace / path
	if 'reports' in path.resolve().parts:
		raise ValueError('reports cannot be pipeline inputs')
	return path


def replace_values(value: Any, replacements: dict[str, str]) -> Any:  # noqa: ANN401
	"""Replace exact identity and path values in inherited survey definitions."""
	if isinstance(value, dict):
		return {
			replace_values(key, replacements): replace_values(item, replacements)
			for key, item in value.items()
		}
	if isinstance(value, list):
		return [replace_values(item, replacements) for item in value]
	return replacements.get(value, value) if isinstance(value, str) else value


def _owned_output(value: str, survey: str, candidate: str | None = None) -> None:
	parts = Path(value).parts
	if 'reports' in parts or NAMESPACE not in parts or survey not in parts:
		raise ValueError('v3 output must use its owned survey artifact namespace')
	if candidate is not None and candidate not in parts:
		raise ValueError('v3 output candidate identity drift')


def _training_identity(full: dict[str, Any], cell: dict[str, Any]) -> None:
	scientific = full['identity']['scientific_identity']
	expected = {
		'experiment_role': 'multi_head_center_trace_masked_hard_pretext',
		'variant': 'ctmask010_cons010',
		'head_spec': full['head']['spec'],
		'head_ks': [6, 8, 10],
		'target_representation': 'hard_viterbi_labels_v1',
		'target_manifest_sha256': cell['target_manifest_sha256'],
		'objective_semantics': MASK_CONTEXT['objective'],
		**{k: v for k, v in MASK_CONTEXT.items() if k != 'objective'},
		'supervised_loss': 'structured_hmm_center_trace_masked_hard_v1',
		'consistency_policy': 'normalized_order_smooth_l1_v1',
	}
	if any(scientific.get(k) != v for k, v in expected.items()):
		raise ValueError('v3 scientific mask identity drift')
	if set(scientific.get('target_head_hashes', {})) != {'6', '8', '10'}:
		raise ValueError('v3 requires explicitly bound per-head target hashes')
	for field in (
		'prototype_weight',
		'usage_weight',
		'consistency_weight',
		'distillation_weight',
	):
		if field in scientific and scientific[field] != full['loss'][field]:
			raise ValueError('v3 scientific loss identity drift')
	if 'train' in scientific and any(
		value != full['train'].get(key) for key, value in scientific['train'].items()
	):
		raise ValueError('v3 scientific training identity drift')


def _validate_cell(experiment: Path, survey: str, cell: dict[str, Any]) -> None:  # noqa: C901, PLR0912
	family, candidate = cell['source'], cell['candidate_id']
	if candidate != CANDIDATE_IDS[family] or cell['survey'] != survey:
		raise ValueError('v3 candidate or survey identity drift')
	baseline = {}
	baseline_number = {'f3': 131, 'parihaka': 44, 'volve': 37}[survey]
	baseline_root = (
		experiment.parent / f'{baseline_number}_hmm_v2_k6810_cons010_multi_source_v1'
	)
	baseline_id = K6810_CONS010.candidate_ids[family]
	for stage, field in (
		('train', 'baseline_train_config'),
		('embedding', 'baseline_embedding_config'),
		('downstream', 'baseline_downstream_config'),
	):
		path = definition_path(cell[field], experiment)
		relative = {
			'train': f'30_pretraining/{baseline_id}/02_full_25ep.yaml',
			'embedding': f'40_embeddings/{baseline_id}.yaml',
			'downstream': f'50_downstream/{baseline_id}.yaml',
		}[stage]
		if path.resolve() != (baseline_root / relative).resolve():
			raise ValueError(
				'v3 baseline must identify the canonical v02 source family'
			)
		if file_sha256(path) != cell['baseline_config_sha256'][stage]:
			raise ValueError(f'v02 baseline config hash drift: {path}')
		baseline[stage] = read_yaml(path)
	validate_fixed_training_config(baseline['train'], K6810_CONS010)
	full = read_yaml(definition_path(cell['train_full_config'], experiment))
	if {
		key: value
		for key, value in full.items()
		if key not in ('paths', 'identity', 'spatial_context')
	} != {
		key: value
		for key, value in baseline['train'].items()
		if key not in ('paths', 'identity')
	}:
		raise ValueError('v3 training differs from canonical v02 inheritance')
	if (
		full['spatial_context'] != MASK_CONTEXT
		or full['identity']['model_tag'] != candidate
	):
		raise ValueError('v3 mask or model identity drift')
	if (
		full['identity']['runtime_identity']
		!= baseline['train']['identity']['runtime_identity']
	):
		raise ValueError('v3 runtime inheritance drift')
	_training_identity(full, cell)
	output = full['paths']['output_root']
	_owned_output(output, survey, candidate)
	if full['paths']['artifact_root'] != baseline['train']['paths']['artifact_root']:
		raise ValueError('v3 artifact root inheritance drift')
	smoke = read_yaml(definition_path(cell['train_smoke_config'], experiment))
	expected = copy.deepcopy(full)
	expected['paths']['output_root'] = (
		output.rsplit('/', 1)[0] + '/gpu_feasibility_1step'
	)
	expected['train'].update(epochs=1, max_steps=1)
	if 'train' in expected['identity']['scientific_identity']:
		expected['identity']['scientific_identity']['train'].update(
			epochs=1, max_steps=1
		)
	if smoke != expected:
		raise ValueError('v3 smoke inheritance drift')
	audit = read_yaml(definition_path(cell['audit_config'], experiment))
	if audit != {'training_configs': [cell['train_full_config']]}:
		raise ValueError('v3 source audit binding drift')
	embedding = read_yaml(definition_path(cell['embedding_config'], experiment))
	expected = copy.deepcopy(baseline['embedding'])
	expected['embeddings'].update(
		checkpoint=output + '/latest.pt',
		output_dir=embedding['embeddings']['output_dir'],
	)
	_owned_output(embedding['embeddings']['output_dir'], survey, candidate)
	if embedding != expected:
		raise ValueError('v3 embedding inheritance drift')
	downstream = read_yaml(definition_path(cell['downstream_config'], experiment))
	replacements = {
		baseline['train']['identity']['model_tag']: candidate,
		baseline['train']['paths']['output_root'] + '/latest.pt': output + '/latest.pt',
		baseline['embedding']['embeddings']['output_dir']: embedding['embeddings'][
			'output_dir'
		],
		cell['baseline_train_config']: cell['train_full_config'],
	}
	for key, value in downstream['outputs'].items():
		_owned_output(value, survey)
		replacements[baseline['downstream']['outputs'][key]] = value
	if downstream != replace_values(baseline['downstream'], replacements):
		raise ValueError('v3 downstream decoder inheritance drift')


def validate_trace_mask_definition(experiment: Path) -> dict[str, Any]:
	"""Require all three new arms with byte-bound canonical v02 definitions."""
	definition = read_yaml(experiment / 'execution.yaml')
	if (
		definition.get('schema_version') != 1
		or definition.get('survey') not in SURVEYS
		or definition.get('head_ks') != [6, 8, 10]
		or len(definition.get('cells', [])) != 3
		or {cell['source'] for cell in definition['cells']} != set(SOURCE_FAMILIES)
	):
		raise ValueError('v3 requires one survey and all three K6810 source families')
	if (
		definition['experiment_id'] != experiment.name
		or definition_path(definition['experiment_root'], experiment).resolve()
		!= experiment.resolve()
	):
		raise ValueError('v3 experiment root identity drift')
	selection = definition['canonical_selection']
	path = definition_path(selection['path'], experiment)
	if file_sha256(path) != selection['sha256']:
		raise ValueError('canonical v02 selection config hash drift')
	canonical = read_yaml(path)
	if canonical['selected_experiment'] != 'k6810_cons010_multi_source_evaluation_v1':
		raise ValueError('v3 must inherit canonical v02 K6810 consistency 0.1')
	_owned_output(definition['summary_root'], definition['survey'])
	for cell in definition['cells']:
		_validate_cell(experiment, definition['survey'], cell)
	return definition
