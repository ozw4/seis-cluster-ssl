"""Publish lightweight review evidence for the experiment-104 handoff."""
# ruff: noqa: C901, E501, ISC004, PLR0911, PLR0912, TRY300

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.center_trace_masked_pretraining_validation import (
	load_f3_center_trace_masked_pretraining_handoff,
)
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	checkpoint_selection_sha256,
	selected_checkpoint_selection_event,
	validate_stratigraphy_checkpoint_payload,
)

_ARTIFACT_ROOT_PLACEHOLDER = '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}'
_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1'
_VARIANT = 'ctmask010_nocons'
_REVIEW_TYPE = 'f3_center_trace_masked_pretraining_review'
_HEAD_KS = (6, 8, 10)
_EXPECTED_EMBEDDING_SHAPE = (76, 113, 32, 384)
_EXPECTED_VALID_TOKEN_SHAPE = (76, 113, 32)

SUMMARY_JSON = 'center_trace_masked_pretraining_summary.json'
SUMMARY_MARKDOWN = 'center_trace_masked_pretraining_summary.md'
TRAINING_DIAGNOSTICS_CSV = 'center_trace_masked_training_diagnostics.csv'
CHECKPOINT_SELECTION_SUMMARY_JSON = (
	'center_trace_masked_checkpoint_selection_summary.json'
)
PRETRAINING_HANDOFF_JSON = 'center_trace_masked_pretraining_handoff.json'
EMBEDDING_EXECUTION_JSON = 'embedding_extraction_execution.json'
OUTPUT_NAMES = (
	SUMMARY_JSON,
	SUMMARY_MARKDOWN,
	TRAINING_DIAGNOSTICS_CSV,
	CHECKPOINT_SELECTION_SUMMARY_JSON,
	PRETRAINING_HANDOFF_JSON,
)
_DIAGNOSTIC_FIELDS = (
	'loss',
	'loss_prototype',
	'loss_prototype_masked',
	'loss_prototype_visible',
	'loss_usage',
	'loss_distillation',
	'loss_consistency_contribution',
	'masked_supervised_token_fraction',
	'visible_supervised_token_fraction',
	'valid_distillation_token_fraction',
	'eligible_xy_column_count',
	'selected_xy_column_count',
	*tuple(
		f'{prefix}_k{k}'
		for prefix in (
			'loss_prototype_masked',
			'loss_prototype_visible',
			'loss_usage',
			'target_usage_entropy',
			'prototype_usage_entropy',
			'masked_top1_accuracy',
		)
		for k in _HEAD_KS
	),
)


@dataclass(frozen=True)
class F3CenterTraceMaskedPretrainingReviewConfig:
	"""Source and output paths for the center-trace review publication."""

	artifact_root: Path
	workspace_root: Path
	pretraining_handoff: Path
	output_dir: Path


@dataclass(frozen=True)
class F3CenterTraceMaskedPretrainingReviewResult:
	"""Review output paths."""

	output_dir: Path
	summary_json: Path
	summary_markdown: Path
	training_diagnostics: Path
	checkpoint_selection_summary: Path
	pretraining_handoff: Path


_CONFIG_KEYS = frozenset(
	{'artifact_root', 'workspace_root', 'pretraining_handoff', 'output_dir'}
)


def f3_center_trace_masked_pretraining_review_config_from_mapping(
	config: Mapping[str, object],
) -> F3CenterTraceMaskedPretrainingReviewConfig:
	"""Resolve the closed review-publication configuration."""
	if not isinstance(config, Mapping):
		raise TypeError('center-trace review config must be a mapping')
	unknown, missing = set(config) - _CONFIG_KEYS, _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(f'unknown center-trace review keys: {sorted(unknown)!r}')
	if missing:
		raise ValueError(f'missing center-trace review keys: {sorted(missing)!r}')

	def path(key: str) -> Path:
		value = config[key]
		if not isinstance(value, str) or not value:
			raise TypeError(f'{key} must be a non-empty path string')
		return Path(value).resolve()

	result = F3CenterTraceMaskedPretrainingReviewConfig(
		artifact_root=path('artifact_root'),
		workspace_root=path('workspace_root'),
		pretraining_handoff=path('pretraining_handoff'),
		output_dir=path('output_dir'),
	)
	if not result.artifact_root.is_dir() or not result.workspace_root.is_dir():
		raise FileNotFoundError(
			'artifact_root and workspace_root must be existing directories'
		)
	if not result.pretraining_handoff.is_file():
		raise FileNotFoundError(
			f'pretraining_handoff is missing: {result.pretraining_handoff}'
		)
	return result


def publish_f3_center_trace_masked_pretraining_review(
	config: F3CenterTraceMaskedPretrainingReviewConfig,
	*,
	dry_run: bool = False,
) -> F3CenterTraceMaskedPretrainingReviewResult:
	"""Validate live PASS evidence and publish lightweight review files."""
	handoff = load_f3_center_trace_masked_pretraining_handoff(
		config.pretraining_handoff
	)
	live = _inspect_live_evidence(handoff=handoff)
	portable = _mapping(
		_portable_value(
			_review_evidence(config, handoff=handoff, live=live), config=config
		),
		'portable review evidence',
	)
	result = F3CenterTraceMaskedPretrainingReviewResult(
		output_dir=config.output_dir,
		summary_json=config.output_dir / SUMMARY_JSON,
		summary_markdown=config.output_dir / SUMMARY_MARKDOWN,
		training_diagnostics=config.output_dir / TRAINING_DIAGNOSTICS_CSV,
		checkpoint_selection_summary=(
			config.output_dir / CHECKPOINT_SELECTION_SUMMARY_JSON
		),
		pretraining_handoff=config.output_dir / PRETRAINING_HANDOFF_JSON,
	)
	if dry_run:
		return result
	_publish_review(config, portable=portable, live=live, handoff=handoff)
	return result


def _inspect_live_evidence(
	*,
	handoff: Mapping[str, object],
) -> dict[str, object]:
	"""Revalidate every live file named by the complete handoff."""
	targets = _mapping(handoff['targets'], 'handoff targets')
	_validate_source_references(targets)
	checkpoint = _mapping(handoff['checkpoint'], 'handoff checkpoint')
	selected_path = _artifact_file(
		checkpoint['path'], label='selected checkpoint'
	)
	latest_path = _artifact_file(
		checkpoint['latest_path'], label='latest checkpoint'
	)
	if file_sha256(selected_path) != checkpoint['sha256']:
		raise ValueError('selected checkpoint SHA-256 does not match handoff')
	if file_sha256(latest_path) != checkpoint['latest_sha256']:
		raise ValueError('latest checkpoint SHA-256 does not match handoff')
	selected = _torch_mapping(selected_path, 'selected checkpoint')
	latest = _torch_mapping(latest_path, 'latest checkpoint')
	_validate_checkpoint_lineage(
		handoff=handoff,
		checkpoint=checkpoint,
		selected=selected,
		latest=latest,
	)
	checkpoint_root = selected_path.parent
	selection_path = checkpoint_root / 'checkpoint_selection_summary.json'
	selection_payload = _json_mapping(selection_path, 'checkpoint selection summary')
	if selection_payload != latest.get('checkpoint_selection'):
		raise ValueError('checkpoint selection summary is stale')
	if checkpoint_selection_sha256(selection_payload) != checkpoint[
		'selection_history_sha256'
	]:
		raise ValueError('checkpoint selection summary hash does not match handoff')
	diagnostics_reference = _mapping(
		handoff['training_diagnostics'], 'handoff training diagnostics'
	)
	diagnostics_path = _validate_live_reference(
		diagnostics_reference,
		label='handoff training diagnostics',
	)
	expected_diagnostics_path = checkpoint_root / 'multi_head_epoch_metrics.csv'
	if diagnostics_path != expected_diagnostics_path.resolve():
		raise ValueError(
			'center-trace training diagnostics path does not bind selected checkpoint'
		)
	diagnostics = _read_diagnostics(diagnostics_path)
	if len(diagnostics) != 25 or [row['epoch'] for row in diagnostics] != list(
		range(1, 26)
	):
		raise ValueError('center-trace training diagnostics must cover epochs 1-25')
	if (
		latest.get('epoch') != 25
		or latest.get('global_step') != 25600
		or diagnostics[-1]['global_step'] != 25600
	):
		raise ValueError('center-trace full run completion identity is invalid')
	if diagnostics[-1]['global_step'] != latest['global_step']:
		raise ValueError('center-trace diagnostics do not bind latest checkpoint')
	embedding = _embedding_evidence(
		handoff=handoff, selected=selected_path, selected_payload=selected
	)
	training_execution_path = checkpoint_root / 'run_metadata.json'
	training_execution_payload = _json_mapping(
		training_execution_path, 'training run metadata'
	)
	training_execution = _execution_counts(
		training_execution_payload.get('execution_counts'),
		keys=('fresh', 'resume'),
		label='training execution counts',
	)
	embedding_execution_path = (
		Path(str(embedding['root'])) / EMBEDDING_EXECUTION_JSON
	)
	embedding_execution_payload = _json_mapping(
		embedding_execution_path, 'embedding execution summary'
	)
	if (
		embedding_execution_payload.get('artifact_type')
		!= 'embedding_extraction_execution'
		or embedding_execution_payload.get('schema_version') != 1
	):
		raise ValueError('embedding execution summary identity mismatch')
	embedding_execution = _execution_counts(
		embedding_execution_payload,
		keys=('fresh', 'reuse'),
		label='embedding execution counts',
	)
	if sum(training_execution.values()) != 1:
		raise ValueError('training execution counts must describe one invocation')
	if embedding_execution_payload.get('survey_count') != embedding['survey_count']:
		raise ValueError('embedding execution summary survey count is stale')
	if sum(embedding_execution.values()) != embedding['survey_count']:
		raise ValueError('embedding execution counts do not cover all surveys')
	return {
		'checkpoint_root': checkpoint_root,
		'selected': selected,
		'latest': latest,
		'selection': selection_payload,
		'selection_path': selection_path,
		'diagnostics': diagnostics,
		'diagnostics_path': diagnostics_path,
		'diagnostics_reference': dict(diagnostics_reference),
		'training_execution': training_execution,
		'training_execution_path': training_execution_path,
		'embedding_execution': embedding_execution,
		'embedding_execution_path': embedding_execution_path,
		'embedding': embedding,
	}


def _validate_source_references(
	targets: Mapping[str, object],
) -> None:
	"""Fail closed when a handoff input has drifted since validation."""
	for key in ('target_manifest', 'hard_baseline_config', 'hard_baseline_handoff'):
		_validate_live_reference(targets[key], label=f'handoff {key}')
	inputs = _mapping(targets['real_data_inputs'], 'handoff real-data inputs')
	for key in (
		'train_manifest',
		'train_path_list',
		'teacher_checkpoint',
		'student_init_checkpoint',
	):
		_validate_live_reference(inputs[key], label=f'handoff inputs.{key}')
	for index, survey_value in enumerate(inputs['surveys']):
		survey = _mapping(survey_value, f'handoff input survey {index}')
		for key in ('amplitude', 'normalization_stats'):
			_validate_live_reference(
				survey[key],
				label=f'handoff input survey {index}.{key}',
			)


def _validate_checkpoint_lineage(
	*,
	handoff: Mapping[str, object],
	checkpoint: Mapping[str, object],
	selected: Mapping[str, object],
	latest: Mapping[str, object],
) -> None:
	"""Bind schema-7 payloads and the selected event to the PASS handoff."""
	targets = _mapping(handoff['targets'], 'handoff targets')
	target_manifest = _mapping(targets['target_manifest'], 'handoff target manifest')
	for label, payload in (
		('selected checkpoint', selected),
		('latest checkpoint', latest),
	):
		validate_stratigraphy_checkpoint_payload(payload)
		identity = _mapping(payload['stratigraphy_checkpoint'], f'{label} identity')
		if identity.get('schema_version') != 7 or identity.get('model_tag') != _MODEL_TAG:
			raise ValueError(f'{label} schema/model identity mismatch')
		if identity.get('scientific_identity_sha256') != targets[
			'scientific_identity_sha256'
		]:
			raise ValueError(f'{label} scientific identity mismatch')
		if identity.get('target_manifest_sha256') != target_manifest['sha256']:
			raise ValueError(f'{label} target manifest mismatch')
		if identity.get('schema_version') != checkpoint['schema_version']:
			raise ValueError(f'{label} checkpoint schema mismatch')
	selected_identity = _mapping(
		selected['stratigraphy_checkpoint'], 'selected checkpoint identity'
	)
	metrics = _mapping(selected['metrics'], 'selected checkpoint metrics')
	selection = _mapping(latest['checkpoint_selection'], 'latest checkpoint selection')
	event = selected_checkpoint_selection_event(selection)
	if selected.get('epoch') != checkpoint['selected_epoch']:
		raise ValueError('selected checkpoint epoch does not match handoff')
	if selected.get('global_step') != checkpoint['selected_global_step']:
		raise ValueError('selected checkpoint step does not match handoff')
	if metrics.get('loss') != checkpoint['selected_loss']:
		raise ValueError('selected checkpoint loss does not match handoff')
	if _mapping(selected['training_state'], 'selected training state').get(
		'checkpoint_kind'
	) != checkpoint['selected_checkpoint_kind']:
		raise ValueError('selected checkpoint kind does not match handoff')
	if event.get('epoch') != selected.get('epoch') or event.get('global_step') != selected.get(
		'global_step'
	):
		raise ValueError('selected checkpoint event is not bound to its payload')
	if event.get('loss') != metrics.get('loss'):
		raise ValueError('selected checkpoint event loss is not bound to its payload')
	if selected_identity.get('initial_spatial_context_state_sha256') != targets[
		'initial_spatial_context_state_sha256'
	]:
		raise ValueError('selected replacement-token initialization is stale')
	if selected_identity.get('optimizer_group_identity') != checkpoint[
		'optimizer_group_identity'
	]:
		raise ValueError('selected optimizer group identity is stale')
	if selected.get('trainability_summary') != checkpoint['trainability_summary']:
		raise ValueError('selected trainability summary is stale')


def _embedding_evidence(
	*,
	handoff: Mapping[str, object],
	selected: Path,
	selected_payload: Mapping[str, object],
) -> dict[str, object]:
	"""Verify the selected checkpoint's complete unmasked embedding output."""
	embedding = _mapping(handoff['embedding'], 'handoff embedding')
	root = _artifact_file(
		embedding['root'], label='handoff embedding root'
	)
	if not root.is_dir():
		raise FileNotFoundError(f'handoff embedding root is missing: {root}')
	files = output_paths(root, 'f3_facies_benchmark')
	for path, key in (
		(files.metadata, 'metadata_sha256'),
		(files.embeddings, 'embeddings_sha256'),
		(files.valid_tokens, 'valid_tokens_sha256'),
	):
		if not path.is_file() or file_sha256(path) != embedding[key]:
			raise ValueError(f'handoff embedding {key} does not match live bytes')
	for path, key in (
		(files.metadata, 'metadata_path'),
		(files.embeddings, 'embeddings_path'),
		(files.valid_tokens, 'valid_tokens_path'),
	):
		if Path(str(embedding[key])).resolve() != path.resolve():
			raise ValueError(f'handoff embedding {key} path is stale')
	metadata = _json_mapping(files.metadata, 'embedding metadata')
	if Path(str(metadata.get('checkpoint_path'))).resolve() != selected.resolve():
		raise ValueError('embedding metadata does not bind selected checkpoint')
	if metadata.get('checkpoint_sha256') != file_sha256(selected):
		raise ValueError('embedding metadata checkpoint hash is stale')
	selected_identity = _mapping(
		selected_payload['stratigraphy_checkpoint'], 'selected checkpoint identity'
	)
	stratigraphy = _mapping(
		metadata.get('stratigraphy_pretext'), 'embedding stratigraphy identity'
	)
	for key in (
		'model_tag',
		'target_representation',
		'objective_semantics',
		'mask_semantics',
		'column_fraction',
		'selection_policy',
		'replacement',
		'replacement_initialization',
		'rng_policy',
		'masked_prototype_weight',
		'visible_prototype_weight',
		'distillation_scope',
		'supervised_loss',
		'consistency_policy',
		'scientific_identity_sha256',
	):
		if stratigraphy.get(key) != selected_identity.get(key):
			raise ValueError(f'embedding stratigraphy identity mismatch: {key}')
	if stratigraphy.get('checkpoint_stratigraphy_state_sha256') != selected_identity.get(
		'stratigraphy_state_sha256'
	):
		raise ValueError('embedding checkpoint state identity mismatch')
	if stratigraphy.get('checkpoint_spatial_context_state_sha256') != selected_identity.get(
		'spatial_context_state_sha256'
	):
		raise ValueError('embedding replacement-token state identity mismatch')
	if stratigraphy.get('target_manifest_sha256') != selected_identity.get(
		'target_manifest_sha256'
	):
		raise ValueError('embedding target manifest identity mismatch')
	if stratigraphy.get('per_head_target_sha256') != selected_identity.get(
		'per_head_targets'
	):
		raise ValueError('embedding target head identity mismatch')
	embeddings = np.load(files.embeddings, mmap_mode='r', allow_pickle=False)
	valid = np.load(files.valid_tokens, mmap_mode='r', allow_pickle=False)
	valid_count = int(valid.sum())
	if (
		tuple(embeddings.shape) != _EXPECTED_EMBEDDING_SHAPE
		or embeddings.dtype != np.float16
		or tuple(valid.shape) != _EXPECTED_VALID_TOKEN_SHAPE
		or valid.dtype != np.bool_
		or valid_count != int(embedding['finite_valid_count'])
		or not valid_count
		or not np.isfinite(embeddings[valid]).all()
	):
		raise ValueError('center-trace embedding shape/dtype/finite identity mismatch')
	canonical = _mapping(
		embedding['canonical_valid_token_identities'],
		'canonical valid-token identities',
	)
	for role, value in canonical.items():
		item = _mapping(value, f'canonical valid-token identity {role}')
		path = _artifact_file(
			item['path'], label=f'canonical {role} mask'
		)
		if file_sha256(path) != item['sha256']:
			raise ValueError(f'canonical {role} valid-token hash is stale')
		if item['sha256'] != embedding['valid_tokens_sha256']:
			raise ValueError('canonical valid-token mask parity failed')
	return {
		**dict(embedding),
		'root': root,
		'metadata_path': files.metadata,
		'embeddings_path': files.embeddings,
		'valid_tokens_path': files.valid_tokens,
		'actual_finite_valid_count': valid_count,
		'survey_count': 1,
	}


def _review_evidence(
	config: F3CenterTraceMaskedPretrainingReviewConfig,
	*,
	handoff: Mapping[str, object],
	live: Mapping[str, object],
) -> dict[str, object]:
	targets = _mapping(handoff['targets'], 'handoff targets')
	checkpoint = _mapping(handoff['checkpoint'], 'handoff checkpoint')
	selected = _mapping(live['selected'], 'live selected checkpoint')
	selected_identity = _mapping(
		selected['stratigraphy_checkpoint'], 'selected checkpoint identity'
	)
	diagnostics = live['diagnostics']
	ranges = {
		field: _metric_range(diagnostics, field) for field in _DIAGNOSTIC_FIELDS
	}
	return {
		'artifact_type': _REVIEW_TYPE,
		'schema_version': 1,
		'status': 'PASS',
		'execution': {
			'git_sha': _mapping(handoff['execution']['after'], 'execution after')[
				'git_commit'
			],
			'dirty_status': _mapping(handoff['execution']['after'], 'execution after')[
				'git_status_short'
			],
			'before': handoff['execution']['before'],
			'after': handoff['execution']['after'],
		},
		'baseline_candidate_fixed_field_parity': targets[
			'hard_baseline_config_parity'
		],
		'hard_target_identity': {
			'target_manifest': targets['target_manifest'],
			'per_head_target_hashes': targets['per_head_target_hashes'],
			'target_representation': targets['target_representation'],
			'boundary_weight_semantics': 'valid_token_indicator_v1',
		},
		'objective_mask_loss_identity': {
			key: targets[key]
			for key in (
				'experiment_role',
				'variant',
				'objective_semantics',
				'mask_semantics',
				'column_fraction',
				'selection_policy',
				'replacement',
				'replacement_initialization',
				'rng_policy',
				'masked_prototype_weight',
				'visible_prototype_weight',
				'distillation_scope',
				'supervised_loss',
				'consistency_policy',
			)
		},
		'full_run': {
			'epoch_count': len(diagnostics),
			'final_epoch': diagnostics[-1]['epoch'],
			'final_global_step': diagnostics[-1]['global_step'],
			'expected_epochs': 25,
			'expected_global_step': 25600,
		},
		'selected_checkpoint': {
			'path': checkpoint['path'],
			'sha256': checkpoint['sha256'],
			'kind': checkpoint['selected_checkpoint_kind'],
			'epoch': checkpoint['selected_epoch'],
			'global_step': checkpoint['selected_global_step'],
			'loss': checkpoint['selected_loss'],
			'latest_path': checkpoint['latest_path'],
			'latest_sha256': checkpoint['latest_sha256'],
			'selection_history_sha256': checkpoint['selection_history_sha256'],
			'selection_event': selected_checkpoint_selection_event(
				_mapping(live['selection'], 'live checkpoint selection')
			),
		},
		'replacement_token': {
			'initial_sha256': targets['initial_spatial_context_state_sha256'],
			'current_sha256': selected_identity['spatial_context_state_sha256'],
			'identity': targets['replacement'],
		},
		'optimizer_trainability': {
			'optimizer_group_identity': checkpoint['optimizer_group_identity'],
			'trainability_summary': checkpoint['trainability_summary'],
			'trainability_summary_sha256': checkpoint['trainability_summary_sha256'],
		},
		'training_metrics': {
			'ranges': ranges,
			'rows': len(diagnostics),
			'consistency_contribution_exact_zero': all(
				row['loss_consistency_contribution'] == 0.0 for row in diagnostics
			),
		},
		'embedding': {
			key: live['embedding'][key]
			for key in (
				'root',
				'metadata_path',
				'metadata_sha256',
				'embeddings_path',
				'embeddings_sha256',
				'valid_tokens_path',
				'valid_tokens_sha256',
				'embeddings_shape',
				'embeddings_dtype',
				'valid_tokens_shape',
				'valid_tokens_dtype',
				'finite_valid_count',
			)
		},
		'execution_counts': {
			'training': live['training_execution'],
			'embedding': live['embedding_execution'],
		},
		'canonical_valid_mask_parity': live['embedding'][
			'canonical_valid_token_identities'
		],
		'checkpoint_selection_source': {
			'path': live['selection_path'],
			'sha256': file_sha256(live['selection_path']),
		},
		'training_diagnostics_source': {
			'path': live['diagnostics_path'],
			'sha256': live['diagnostics_reference']['sha256'],
		},
		'execution_counts_source': {
			'training': {
				'path': live['training_execution_path'],
				'sha256': file_sha256(live['training_execution_path']),
			},
			'embedding': {
				'path': live['embedding_execution_path'],
				'sha256': file_sha256(live['embedding_execution_path']),
			},
		},
		'pass_handoff': {
			'path': config.pretraining_handoff,
			'sha256': file_sha256(config.pretraining_handoff),
			'status': handoff['status'],
		},
		'downstream_screening': {
			'ready': True,
			'executed': False,
			'original_split_gate': 'NOT_EXECUTED',
			'six_split': 'NOT_EXECUTED',
		},
		'scientific_superiority_conclusion': 'NOT_CONCLUDED',
	}


def render_f3_center_trace_masked_pretraining_review_markdown(
	evidence: Mapping[str, object],
) -> str:
	"""Render a compact review without raw arrays or downstream metrics."""
	full = _mapping(evidence['full_run'], 'full run')
	checkpoint = _mapping(evidence['selected_checkpoint'], 'selected checkpoint')
	embedding = _mapping(evidence['embedding'], 'embedding')
	execution = _mapping(evidence['execution'], 'execution')
	execution_counts = _mapping(evidence['execution_counts'], 'execution counts')
	training_counts = _mapping(execution_counts['training'], 'training counts')
	embedding_counts = _mapping(execution_counts['embedding'], 'embedding counts')
	lines = [
		'# F3 center-trace masked pretraining review',
		'',
		f'- Status: `{evidence["status"]}`',
		f'- Model tag: `{_MODEL_TAG}`',
		f'- Variant: `{_VARIANT}`',
		f'- Execution Git SHA: `{execution["git_sha"]}`',
		f'- Execution dirty status: `{execution["dirty_status"]}`',
		f'- Full run: epoch `{full["final_epoch"]}` / global step `{full["final_global_step"]}`',
		f'- Selected checkpoint: `{checkpoint["kind"]}` epoch `{checkpoint["epoch"]}` step `{checkpoint["global_step"]}` loss `{float(checkpoint["loss"]):.8g}`',
		f'- Selected checkpoint SHA-256: `{checkpoint["sha256"]}`',
		f'- Embedding shape/dtype: `{embedding["embeddings_shape"]}` / `{embedding["embeddings_dtype"]}`',
		f'- Valid-token count: `{embedding["finite_valid_count"]}`',
		f'- Execution counts: training fresh `{training_counts["fresh"]}` / resume `{training_counts["resume"]}`; embedding fresh `{embedding_counts["fresh"]}` / reuse `{embedding_counts["reuse"]}`',
		f'- PASS handoff SHA-256: `{_mapping(evidence["pass_handoff"], "PASS handoff")["sha256"]}`',
		'',
		'## Fixed scientific identity',
		'',
		'The hard K=6/8/10 target identity, center-trace mask semantics, '
		'0.50/0.50 masked-visible objective, visible-only distillation, learned '
		'replacement token, and disabled consistency policy were validated from '
		'the live PASS handoff.',
		'',
		'## Training diagnostics',
		'',
		'| Metric | Minimum | Maximum |',
		'| --- | ---: | ---: |',
	]
	ranges = _mapping(evidence['training_metrics'], 'training metrics')['ranges']
	for field in (
		'loss',
		'loss_prototype_masked_k6',
		'loss_prototype_visible_k6',
		'loss_prototype_masked_k8',
		'loss_prototype_visible_k8',
		'loss_prototype_masked_k10',
		'loss_prototype_visible_k10',
		'masked_top1_accuracy_k6',
		'masked_top1_accuracy_k8',
		'masked_top1_accuracy_k10',
		'masked_supervised_token_fraction',
		'visible_supervised_token_fraction',
		'eligible_xy_column_count',
		'selected_xy_column_count',
	):
		item = _mapping(ranges[field], f'metric range {field}')
		lines.append(
			f'| `{field}` | {float(item["min"]):.8g} | {float(item["max"]):.8g} |'
		)
	lines.extend(
		[
			'',
			'## Downstream status',
			'',
			'Downstream decoder evaluation, the original-split gate, and six-split '
			'screening were not executed here. The validated PASS handoff is ready '
			'for that authorized downstream screening.',
			'',
			'Scientific superiority is not concluded by this pretraining review.',
		]
	)
	return '\n'.join(lines) + '\n'


def _publish_review(
	config: F3CenterTraceMaskedPretrainingReviewConfig,
	*,
	portable: Mapping[str, object],
	live: Mapping[str, object],
	handoff: Mapping[str, object],
) -> None:
	"""Write the fixed lightweight review tree."""
	_validate_existing_output_dir(config)
	selection = _mapping(live['selection'], 'live selection')
	config.output_dir.mkdir(parents=True, exist_ok=True)
	(config.output_dir / SUMMARY_JSON).write_text(
		_json_text(portable), encoding='utf-8'
	)
	(config.output_dir / SUMMARY_MARKDOWN).write_text(
		render_f3_center_trace_masked_pretraining_review_markdown(portable),
		encoding='utf-8',
	)
	(config.output_dir / TRAINING_DIAGNOSTICS_CSV).write_text(
		_diagnostics_csv_text(live['diagnostics']), encoding='utf-8'
	)
	(config.output_dir / CHECKPOINT_SELECTION_SUMMARY_JSON).write_text(
		_json_text(selection), encoding='utf-8'
	)
	(config.output_dir / PRETRAINING_HANDOFF_JSON).write_text(
		_json_text(_portable_value(handoff, config=config)), encoding='utf-8'
	)


def _validate_existing_output_dir(
	config: F3CenterTraceMaskedPretrainingReviewConfig,
) -> None:
	"""Reject foreign or heavy files instead of deleting them in place."""
	if not config.output_dir.exists():
		return
	if not config.output_dir.is_dir():
		raise ValueError(f'center-trace output_dir is not a directory: {config.output_dir}')
	allowed = set(OUTPUT_NAMES)
	for path in config.output_dir.rglob('*'):
		if path.is_symlink():
			raise ValueError(f'center-trace output must not contain symlinks: {path}')
		if path.is_dir():
			continue
		if path.relative_to(config.output_dir).as_posix() not in allowed:
			raise ValueError(f'center-trace output contains unallowlisted file: {path}')


def _read_diagnostics(path: Path) -> list[dict[str, float | int]]:
	if not path.is_file():
		raise FileNotFoundError(f'center-trace training diagnostics are missing: {path}')
	with path.open(newline='', encoding='utf-8') as handle:
		rows = list(csv.DictReader(handle))
	if not rows or not {'epoch', 'global_step', *_DIAGNOSTIC_FIELDS} <= set(rows[0]):
		raise ValueError('center-trace training diagnostics schema is incomplete')
	parsed: list[dict[str, float | int]] = []
	for row in rows:
		try:
			item: dict[str, float | int] = {
				'epoch': int(row['epoch']),
				'global_step': int(row['global_step']),
				**{field: float(row[field]) for field in _DIAGNOSTIC_FIELDS},
			}
		except (KeyError, TypeError, ValueError) as error:
			raise ValueError(
				'center-trace training diagnostics contain invalid values'
			) from error
		if any(
			not math.isfinite(float(value))
			for key, value in item.items()
			if key not in {'epoch', 'global_step'}
		):
			raise ValueError('center-trace training diagnostics contain non-finite values')
		if item['loss_consistency_contribution'] != 0.0:
			raise ValueError('center-trace training diagnostics consistency is nonzero')
		if item['masked_supervised_token_fraction'] <= 0.0 or item[
			'visible_supervised_token_fraction'
		] <= 0.0:
			raise ValueError('center-trace diagnostics have an empty supervised branch')
		if item['selected_xy_column_count'] <= 0.0 or item[
			'eligible_xy_column_count'
		] <= 0.0:
			raise ValueError('center-trace diagnostics have empty XY selection')
		parsed.append(item)
	return parsed


def _execution_counts(
	value: object,
	*,
	keys: tuple[str, str],
	label: str,
) -> dict[str, int]:
	counts = _mapping(value, label)
	result: dict[str, int] = {}
	for key in keys:
		item = counts.get(key)
		if isinstance(item, bool) or not isinstance(item, int) or item < 0:
			raise ValueError(f'{label}.{key} must be a nonnegative integer')
		result[key] = item
	return result


def _diagnostics_csv_text(rows: object) -> str:
	if not isinstance(rows, list):
		raise TypeError('training diagnostics must be a list')
	stream = io.StringIO(newline='')
	writer = csv.DictWriter(
		stream,
		fieldnames=('epoch', 'global_step', *_DIAGNOSTIC_FIELDS),
		lineterminator='\n',
	)
	writer.writeheader()
	writer.writerows(rows)
	return stream.getvalue()


def _metric_range(rows: object, field: str) -> dict[str, float]:
	if not isinstance(rows, list) or not rows:
		raise ValueError(f'metric range {field} has no rows')
	values = [float(_mapping(row, 'diagnostic row')[field]) for row in rows]
	return {'min': min(values), 'max': max(values)}


def _validate_live_reference(
	value: object,
	*,
	label: str,
) -> Path:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must contain path and sha256')
	path = _artifact_or_workspace_file(value.get('path'), label=label)
	digest = value.get('sha256')
	if file_sha256(path) != digest:
		raise ValueError(f'{label} SHA-256 does not match live bytes')
	return path


def _artifact_or_workspace_file(
	value: object,
	*,
	label: str,
) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} path is missing')
	path = Path(value).resolve()
	if not path.is_file():
		raise FileNotFoundError(f'{label} is missing: {path}')
	return path


def _artifact_file(value: object, *, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} path is missing')
	path = Path(value).resolve()
	if not path.exists():
		raise FileNotFoundError(f'{label} is missing: {path}')
	return path


def _torch_mapping(path: Path, label: str) -> Mapping[str, object]:
	payload = torch.load(path, map_location='cpu', weights_only=False)
	return _mapping(payload, label)


def _json_mapping(path: Path, label: str) -> Mapping[str, object]:
	try:
		return _mapping(json.loads(path.read_text(encoding='utf-8')), label)
	except (OSError, json.JSONDecodeError) as error:
		raise ValueError(f'{label} is not valid JSON: {path}') from error


def _json_text(value: object) -> str:
	return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'


def _portable_value(
	value: object,
	*,
	config: F3CenterTraceMaskedPretrainingReviewConfig,
) -> object:
	if isinstance(value, Mapping):
		return {
			str(key): _portable_value(item, config=config)
			for key, item in value.items()
		}
	if isinstance(value, list | tuple):
		return [_portable_value(item, config=config) for item in value]
	if isinstance(value, Path):
		return _portable_path(str(value), config=config)
	if isinstance(value, str):
		return _portable_path(value, config=config)
	return value


def _portable_path(
	value: str,
	*,
	config: F3CenterTraceMaskedPretrainingReviewConfig,
) -> str:
	artifact = str(config.artifact_root.resolve())
	workspace = str(config.workspace_root.resolve())
	if value == artifact:
		return _ARTIFACT_ROOT_PLACEHOLDER
	if value.startswith(f'{artifact}/'):
		return f'{_ARTIFACT_ROOT_PLACEHOLDER}{value[len(artifact):]}'
	if value == workspace:
		return '.'
	if value.startswith(f'{workspace}/'):
		return value[len(workspace) + 1 :]
	return value.replace(
		f'{artifact}/', f'{_ARTIFACT_ROOT_PLACEHOLDER}/'
	).replace(f'{workspace}/', '')


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


__all__ = [
	'CHECKPOINT_SELECTION_SUMMARY_JSON',
	'EMBEDDING_EXECUTION_JSON',
	'OUTPUT_NAMES',
	'PRETRAINING_HANDOFF_JSON',
	'SUMMARY_JSON',
	'SUMMARY_MARKDOWN',
	'TRAINING_DIAGNOSTICS_CSV',
	'F3CenterTraceMaskedPretrainingReviewConfig',
	'F3CenterTraceMaskedPretrainingReviewResult',
	'f3_center_trace_masked_pretraining_review_config_from_mapping',
	'publish_f3_center_trace_masked_pretraining_review',
	'render_f3_center_trace_masked_pretraining_review_markdown',
]
