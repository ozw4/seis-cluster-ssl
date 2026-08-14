# ruff: noqa: TRY301
"""Validation helpers for the current-code F3 single-head K=6 control."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.stratigraphy import discover_pseudo_target_inputs
from seis_ssl_cluster.training.checkpoint import load_checkpoint
from seis_ssl_cluster.training.strat_hmm.components import (
	build_strat_hmm_head_only_components,
)

MODEL_TAG = 'strat_hmm_pretext_m1_current_k6_topblock1_distill_v1'
HISTORICAL_M1_TAG = 'strat_hmm_pretext_m1_k6_topblock1_distill'
MAE_TAG = 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
MIGRATION_ROOT = Path('reports/f3/facies_benchmark_v1/performance_migration_validation')
HISTORICAL_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/'
	'03_train_single_head_topblock_distill_full.yaml'
)
HISTORICAL_PRETRAIN_ROOT = Path(
	'artifacts/seis_ssl_cluster/pretraining/f3/facies_benchmark_v1/'
	'strat_hmm_pretext_m1_k6_topblock1_distill'
)
HISTORICAL_EMBEDDING_ROOT = Path(
	'artifacts/seis_ssl_cluster/embeddings/f3/facies_benchmark_v1/'
	'strat_hmm_pretext_m1_k6_topblock1_distill/overlap_x16'
)
HISTORICAL_TOKEN_METRICS = Path(
	'reports/f3/facies_benchmark_v1/lithology_probe/'
	'strat_hmm_pretext_m1_k6_topblock1_distill/overlap_x16/'
	'png_slices_segy_labels_v1/linear_balanced_v1/metrics.json'
)
MAE_TOKEN_METRICS = Path(
	'reports/f3/facies_benchmark_v1/lithology_probe/'
	'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/overlap_x16/'
	'png_slices_segy_labels_v1/linear_balanced_v1/metrics.json'
)


def write_control_preflight(config_path: Path) -> tuple[Path, Path]:
	"""Validate immutable inputs and write the control input manifest."""
	config = _resolved_pretraining_config(config_path)
	output_root = _path(_section(config, 'paths'), 'output_root')
	identity = _section(config, 'identity')
	if identity.get('model_tag') != MODEL_TAG:
		raise ValueError('current control model tag mismatch')
	migration = _migration_evidence()
	inputs = _input_identities(config, config_path=config_path)
	parity = _historical_scientific_parity(config)
	payload = {
		'artifact_type': 'f3_current_k6_control_input_manifest',
		'schema_version': 1,
		'created_at_utc': datetime.now(timezone.utc).isoformat(),
		'model_tag': MODEL_TAG,
		'current_git': _git_identity(),
		'migration': migration,
		'inputs': inputs,
		'historical_scientific_parity': parity,
		'finite_check_mode_exception': {
			'historical_reconstructed': 'off',
			'current_explicit': _section(config, 'data').get('finite_check_mode'),
			'classification': 'runtime_check_path_only',
			'source_volume_nonfinite_count': 0,
		},
	}
	if not parity['pass']:
		raise ValueError('historical M1 scientific parity failed')
	preflight = output_root / 'preflight'
	json_path = preflight / 'control_input_manifest.json'
	markdown_path = preflight / 'control_input_manifest.md'
	_write_json(json_path, payload)
	markdown_path.parent.mkdir(parents=True, exist_ok=True)
	markdown_path.write_text(_render_preflight_markdown(payload), encoding='utf-8')
	return json_path, markdown_path


def validate_current_k6_checkpoint(
	*,
	config_path: Path,
	reports_dir: Path,
) -> Path:
	"""Validate completion, provenance, freeze, and optimizer contracts."""
	config = _resolved_pretraining_config(config_path)
	paths = _section(config, 'paths')
	output_root = _path(paths, 'output_root')
	latest_path = output_root / 'latest.pt'
	best_path = output_root / 'best.pt'
	result_path = reports_dir / 'checkpoint_validation.json'
	checks: dict[str, object] = {}
	try:
		latest = load_checkpoint(latest_path, map_location='cpu')
		best = load_checkpoint(best_path, map_location='cpu')
		checks['latest'] = _checkpoint_completion(latest, expected_step=25600)
		checks['best'] = _best_checkpoint_validation(best, latest)
		checks['provenance'] = _checkpoint_provenance_validation(
			latest,
			config=config,
		)
		checks['freeze_contract'] = _freeze_contract_validation(
			latest,
			config=config,
		)
		checks['optimizer'] = _optimizer_contract_validation(latest)
		checks['historical_comparison'] = _historical_checkpoint_comparison(
			latest,
			best,
		)
		checks['resolved_config_exists'] = (
			output_root / 'resolved_config.json'
		).is_file()
		checks['run_metadata_exists'] = (output_root / 'run_metadata.json').is_file()
		if not _checkpoint_checks_pass(checks):
			raise ValueError('one or more checkpoint contract checks failed')
		payload: dict[str, object] = {
			'artifact_type': 'f3_current_k6_checkpoint_validation',
			'schema_version': 1,
			'status': 'PASS',
			'model_tag': MODEL_TAG,
			'best_selection_criterion': 'lowest finite metrics.loss',
			'latest_path': str(latest_path),
			'best_path': str(best_path),
			'latest_sha256': file_sha256(latest_path),
			'best_sha256': file_sha256(best_path),
			'checks': checks,
		}
	except Exception as error:
		payload = {
			'artifact_type': 'f3_current_k6_checkpoint_validation',
			'schema_version': 1,
			'status': 'FAIL',
			'model_tag': MODEL_TAG,
			'error': f'{type(error).__name__}: {error}',
			'checks': checks,
		}
		_write_json(result_path, payload)
		raise
	_write_json(result_path, payload)
	return result_path


def validate_current_k6_embeddings(  # noqa: C901
	*,
	embeddings_dir: Path,
	checkpoint_path: Path,
	reports_dir: Path,
) -> Path:
	"""Validate current embeddings and calculate historical diagnostics."""
	result_path = reports_dir / 'embedding_validation.json'
	try:
		current = output_paths(embeddings_dir, 'f3_facies_benchmark')
		historical = output_paths(HISTORICAL_EMBEDDING_ROOT, 'f3_facies_benchmark')
		metadata = _read_json(current.metadata)
		if metadata.get('checkpoint_sha256') != file_sha256(checkpoint_path):
			raise ValueError('embedding checkpoint SHA-256 mismatch')
		if (
			Path(str(metadata.get('checkpoint_path'))).resolve()
			!= checkpoint_path.resolve()
		):
			raise ValueError('embedding checkpoint path mismatch')
		stratigraphy = _mapping(
			metadata.get('stratigraphy_pretext'), 'embedding stratigraphy_pretext'
		)
		if stratigraphy.get('model_tag') != MODEL_TAG:
			raise ValueError('embedding model tag is absent or mismatched')
		checkpoint_payload = load_checkpoint(checkpoint_path, map_location='cpu')
		control_identity = _mapping(
			checkpoint_payload.get('control_identity'), 'checkpoint control_identity'
		)
		if stratigraphy.get('control_identity_sha256') != _control_identity_sha256(
			control_identity
		):
			raise ValueError('embedding control identity SHA-256 mismatch')
		current_embeddings = np.load(current.embeddings, mmap_mode='r')
		current_valid = np.load(current.valid_tokens, mmap_mode='r')
		historical_embeddings = np.load(historical.embeddings, mmap_mode='r')
		historical_valid = np.load(historical.valid_tokens, mmap_mode='r')
		if current_embeddings.shape != (76, 113, 32, 384):
			raise ValueError(
				f'current embedding shape mismatch: {current_embeddings.shape!r}'
			)
		if current_embeddings.dtype != np.float16:
			raise TypeError('current embeddings must be float16')
		if current_valid.shape != (76, 113, 32) or current_valid.dtype != np.bool_:
			raise ValueError('current valid-token array contract mismatch')
		valid = np.asarray(current_valid, dtype=bool)
		if not np.array_equal(valid, np.asarray(historical_valid, dtype=bool)):
			raise ValueError('current/historical valid-token masks differ')
		finite_count = _nonfinite_valid_embedding_count(current_embeddings, valid)
		if finite_count != 0:
			raise ValueError(f'non-finite current valid embeddings: {finite_count}')
		diagnostic = _embedding_diagnostic(
			current_embeddings,
			historical_embeddings,
			valid,
		)
		payload = {
			'artifact_type': 'f3_current_k6_embedding_validation',
			'schema_version': 1,
			'status': 'PASS',
			'model_tag': MODEL_TAG,
			'embeddings': {
				'path': str(current.embeddings),
				'shape': list(current_embeddings.shape),
				'dtype': str(current_embeddings.dtype),
				'sha256': file_sha256(current.embeddings),
			},
			'valid_tokens': {
				'path': str(current.valid_tokens),
				'shape': list(current_valid.shape),
				'dtype': str(current_valid.dtype),
				'sha256': file_sha256(current.valid_tokens),
				'valid_count': int(valid.sum()),
			},
			'nonfinite_valid_embedding_count': finite_count,
			'checkpoint': {
				'path': str(checkpoint_path),
				'sha256': file_sha256(checkpoint_path),
			},
			'historical_diagnostic': diagnostic,
		}
	except Exception as error:
		payload = {
			'artifact_type': 'f3_current_k6_embedding_validation',
			'schema_version': 1,
			'status': 'FAIL',
			'model_tag': MODEL_TAG,
			'error': f'{type(error).__name__}: {error}',
		}
		_write_json(result_path, payload)
		raise
	_write_json(result_path, payload)
	return result_path


def write_token_probe_comparison(
	*, current_metrics_path: Path, output_path: Path
) -> Path:
	"""Write the full-label token sanity comparison against M1 and MAE."""
	current = _read_json(current_metrics_path)
	m1 = _read_json(HISTORICAL_TOKEN_METRICS)
	mae = _read_json(MAE_TOKEN_METRICS)
	metrics = (
		'accuracy',
		'balanced_accuracy',
		'macro_f1',
		'weighted_f1',
		'mean_iou',
		*(
			f'class_{class_id}_{metric}'
			for class_id in range(6)
			for metric in ('f1', 'iou')
		),
	)
	rows = []
	for metric in metrics:
		current_value = _token_metric(current, metric)
		m1_value = _token_metric(m1, metric)
		mae_value = _token_metric(mae, metric)
		rows.append(
			{
				'metric': metric,
				'current_k6': current_value,
				'historical_m1': m1_value,
				'mae': mae_value,
				'current_minus_historical_m1': current_value - m1_value,
				'current_minus_mae': current_value - mae_value,
			}
		)
	_write_csv(output_path, rows)
	return output_path


def _migration_evidence() -> dict[str, object]:
	decision_path = MIGRATION_ROOT / 'performance_migration_decision.json'
	decision = _read_json(decision_path)
	if decision.get('status') not in {'PASS_REUSE_EXISTING', 'PASS_WITH_NUMERIC_DRIFT'}:
		raise ValueError(f'migration status blocks control: {decision.get("status")!r}')
	if decision.get('required_rerun_scope') != (
		'no historical rerun; add a future current-code K=6 control'
	):
		raise ValueError('migration decision rerun scope mismatch')
	pseudo = _read_json(MIGRATION_ROOT / 'pseudo_target_parity.json')
	hmm = _read_json(MIGRATION_ROOT / 'hmm_parity.json')
	probe = _read_json(MIGRATION_ROOT / 'probe_parity.json')
	embedding = _read_json(MIGRATION_ROOT / 'embedding_parity.json')
	if not (
		pseudo.get('labels', {}).get('exact')
		and pseudo.get('confidence', {}).get('exact')
		and pseudo.get('valid_tokens', {}).get('exact')
		and pseudo.get('confidence', {}).get('threshold_crossing_count') == 0
	):
		raise ValueError('pseudo-target parity is not exact')
	hmm_labels = _mapping(hmm.get('labels'), 'HMM label parity')
	if not (
		hmm_labels.get('decoded_labels_exact')
		and hmm_labels.get('valid_token_mask_exact')
		and _mapping(hmm.get('centers'), 'HMM center parity')
		.get('comparison', {})
		.get('allclose')
	):
		raise ValueError('K=6 HMM labels or valid tokens are not exact')
	if not _probe_exact(probe):
		raise ValueError('linear probe parity is not exact')
	cache = _mapping(embedding.get('comparisons'), 'embedding parity comparisons').get(
		'B_current_cache_off_vs_C_current_memmap_cache'
	)
	if not (
		isinstance(cache, Mapping)
		and cache.get('status') == 'EXACT'
		and cache.get('embedding_array_equal')
		and cache.get('valid_token_mask_exact')
	):
		raise ValueError('current cache-off vs memmap extraction is not exact')
	return {
		'decision': {
			'path': str(decision_path),
			'sha256': file_sha256(decision_path),
			'status': decision.get('status'),
			'required_rerun_scope': decision.get('required_rerun_scope'),
			'decision_current_git_sha': decision.get('current_git_sha'),
		},
		'parity': {
			'pseudo_target_exact': True,
			'hmm_labels_valid_exact': True,
			'probe_exact': True,
			'current_cache_off_vs_memmap_exact': True,
		},
	}


def _input_identities(
	config: Mapping[str, object], *, config_path: Path
) -> dict[str, object]:
	pseudo = _section(config, 'pseudo_targets')
	teacher = _path(_section(config, 'teacher'), 'checkpoint')
	student = _path(_section(config, 'student'), 'init_checkpoint')
	inputs = discover_pseudo_target_inputs(
		_path(pseudo, 'input_dir'), k=int(pseudo['k'])
	)
	if len(inputs) != 1:
		raise ValueError('current control requires exactly one F3 pseudo-target input')
	target = inputs[0]
	if target.boundary_weight_path is not None:
		raise ValueError(
			'current K=6 control pseudo target must have no boundary weight'
		)
	metadata = _read_json(target.metadata_path)
	if metadata.get('k') != 6 or metadata.get('schema_version') != 1:
		raise ValueError('current K=6 target metadata contract mismatch')
	manifest = _path(_section(config, 'manifests'), 'train')
	normalization = Path(
		'/workspace/artifacts/seis_ssl_cluster/registry/normalization_stats/f3/'
		'facies_benchmark_v1/f3_seismic.normalization_stats.json'
	)
	return {
		'training_config': _identity(config_path),
		'resolved_training_config_sha256': _canonical_sha256(config),
		'teacher_checkpoint': _identity(teacher),
		'student_init_checkpoint': _identity(student),
		'pseudo_target_root': str(_path(pseudo, 'input_dir')),
		'pseudo_target': {
			'labels': _identity(target.labels_path),
			'confidence': _identity(target.confidence_path),
			'valid_tokens': _identity(target.valid_tokens_path),
			'metadata': _identity(target.metadata_path),
			'schema_version': metadata.get('schema_version'),
			'k': metadata.get('k'),
			'boundary_weight_present': False,
		},
		'f3_manifest': _identity(manifest),
		'normalization_stats': _identity(normalization),
		'historical_m1': {
			'config': _identity(HISTORICAL_CONFIG),
			'best_checkpoint': _identity(HISTORICAL_PRETRAIN_ROOT / 'best.pt'),
			'latest_checkpoint': _identity(HISTORICAL_PRETRAIN_ROOT / 'latest.pt'),
		},
	}


def _historical_scientific_parity(config: Mapping[str, object]) -> dict[str, object]:
	historical = load_config(HISTORICAL_CONFIG)
	current_view = _scientific_view(config)
	historical_view = _scientific_view(historical)
	mismatches = {
		key: {'current': current_view.get(key), 'historical': historical_view.get(key)}
		for key in sorted(set(current_view) | set(historical_view))
		if current_view.get(key) != historical_view.get(key)
	}
	return {
		'pass': not mismatches,
		'fields_compared': sorted(current_view),
		'mismatches': mismatches,
		'allowed_runtime_difference': {
			'data.finite_check_mode': {'historical': 'off', 'current': 'strict'}
		},
	}


def _scientific_view(config: Mapping[str, object]) -> dict[str, object]:
	data = _section(config, 'data')
	return {
		'data': {
			key: data.get(key)
			for key in (
				'local_crop_size',
				'min_valid_fraction',
				'max_resample_attempts',
				'normalized_clip_abs',
				'amplitude_agc',
			)
		},
		'zero_mask': dict(_section(config, 'zero_mask')),
		'model': {
			key: _section(config, 'model').get(key)
			for key in (
				'patch_size',
				'encoder_dim',
				'encoder_depth',
				'encoder_heads',
				'decoder_dim',
				'decoder_depth',
				'decoder_heads',
			)
		},
		'pseudo_targets': dict(_section(config, 'pseudo_targets')),
		'teacher': dict(_section(config, 'teacher')),
		'student': dict(_section(config, 'student')),
		'head': dict(_section(config, 'head')),
		'loss': {
			key: _section(config, 'loss').get(key)
			for key in (
				'prototype_weight',
				'usage_weight',
				'entropy_floor',
				'distillation_weight',
			)
		},
		'train': {
			key: _section(config, 'train').get(key)
			for key in (
				'batch_size',
				'samples_per_epoch',
				'epochs',
				'num_workers',
				'shuffle',
				'lr',
				'encoder_lr',
				'weight_decay',
				'amp',
				'device',
				'seed',
				'grad_clip_norm',
				'checkpoint_every_steps',
				'max_steps',
				'allow_overwrite_output',
			)
		},
	}


def _checkpoint_completion(
	payload: Mapping[str, object], *, expected_step: int
) -> bool:
	if payload.get('epoch') != 25 or payload.get('global_step') != expected_step:
		return False
	state = _mapping(payload.get('training_state'), 'training_state')
	if state.get('checkpoint_kind') != 'epoch':
		return False
	return _finite_metrics(_mapping(payload.get('metrics'), 'checkpoint metrics'))


def _checkpoint_checks_pass(checks: Mapping[str, object]) -> bool:
	"""Evaluate scalar and structured checkpoint contract checks."""
	for name, value in checks.items():
		if name == 'historical_comparison':
			continue
		if name in {'freeze_contract', 'optimizer'}:
			if not _mapping(value, name).get('pass'):
				return False
		elif value is not True:
			return False
	return True


def _best_checkpoint_validation(
	best: Mapping[str, object], latest: Mapping[str, object]
) -> bool:
	if not _finite_metrics(_mapping(best.get('metrics'), 'best metrics')):
		return False
	state = _mapping(best.get('training_state'), 'best training_state')
	if state.get('checkpoint_kind') not in {'epoch', 'step'}:
		return False
	best_loss = float(_mapping(best.get('metrics'), 'best metrics')['loss'])
	latest_loss = float(_mapping(latest.get('metrics'), 'latest metrics')['loss'])
	return best_loss <= latest_loss or math.isclose(best_loss, latest_loss)


def _checkpoint_provenance_validation(  # noqa: PLR0911
	payload: Mapping[str, object], *, config: Mapping[str, object]
) -> bool:
	identity = _mapping(payload.get('control_identity'), 'checkpoint control_identity')
	if identity.get('model_tag') != MODEL_TAG:
		return False
	if identity.get('resolved_training_config_sha256') != _canonical_sha256(config):
		return False
	config_identity = _section(config, 'identity')
	if identity.get('scientific_identity') != config_identity.get(
		'scientific_identity', {}
	):
		return False
	runtime = _mapping(identity.get('runtime_identity'), 'runtime identity')
	if runtime.get('finite_check_mode') != 'strict':
		return False
	if not isinstance(runtime.get('git_commit'), str) or not isinstance(
		runtime.get('git_diff_sha256'), str
	):
		return False
	inputs = _mapping(identity.get('input_identities'), 'checkpoint input_identities')
	teacher = _mapping(inputs.get('teacher_checkpoint'), 'teacher checkpoint identity')
	student = _mapping(inputs.get('student_init_checkpoint'), 'student init identity')
	teacher_path = _path(_section(config, 'teacher'), 'checkpoint')
	student_path = _path(_section(config, 'student'), 'init_checkpoint')
	if (
		teacher != _identity(teacher_path)
		or student != _identity(student_path)
		or teacher.get('sha256') != student.get('sha256')
	):
		return False
	pseudo = inputs.get('pseudo_targets')
	if (
		not isinstance(pseudo, Sequence)
		or isinstance(pseudo, str | bytes)
		or len(pseudo) != 1
	):
		return False
	target = _mapping(pseudo[0], 'pseudo target identity')
	pseudo_config = _section(config, 'pseudo_targets')
	resolved_targets = discover_pseudo_target_inputs(
		_path(pseudo_config, 'input_dir'), k=int(pseudo_config['k'])
	)
	if len(resolved_targets) != 1:
		return False
	expected = resolved_targets[0]
	return (
		target.get('survey_id') == expected.survey_id
		and target.get('labels') == _identity(expected.labels_path)
		and target.get('confidence') == _identity(expected.confidence_path)
		and target.get('valid_tokens') == _identity(expected.valid_tokens_path)
		and target.get('metadata') == _identity(expected.metadata_path)
		and target.get('boundary_weight_present') is False
		and expected.boundary_weight_path is None
	)


def _freeze_contract_validation(
	payload: Mapping[str, object], *, config: Mapping[str, object]
) -> dict[str, object]:
	model_state = _tensor_mapping(payload.get('model_state_dict'), 'model_state_dict')
	init_path = _path(_section(config, 'student'), 'init_checkpoint')
	init = _tensor_mapping(
		load_checkpoint(init_path, map_location='cpu').get('model_state_dict'),
		'initial model_state_dict',
	)
	depth = int(_section(config, 'model')['encoder_depth'])
	top_prefix = f'encoder.layers.{depth - 1}.'
	frozen_exact = all(
		torch.equal(value, init[name])
		for name, value in model_state.items()
		if not name.startswith(top_prefix)
	)
	identity = _mapping(payload.get('control_identity'), 'control_identity')
	initial_hashes = _mapping(
		identity.get('initial_parameter_sha256'), 'initial_parameter_sha256'
	)
	top_initial = _mapping(initial_hashes.get('student_trainable'), 'student hashes')
	head_initial = _mapping(initial_hashes.get('prototype_head'), 'head hashes')
	current_top = {
		name: _tensor_hash(name, value)
		for name, value in model_state.items()
		if name in top_initial
	}
	head_state = _tensor_mapping(
		payload.get('stratigraphy_state_dict'), 'stratigraphy_state_dict'
	)
	current_head = {
		name: _tensor_hash(name, value) for name, value in head_state.items()
	}
	top_changed = any(
		current_top.get(name) != expected for name, expected in top_initial.items()
	)
	head_changed = any(
		current_head.get(name) != expected for name, expected in head_initial.items()
	)
	components = build_strat_hmm_head_only_components(config, device='cpu')
	teacher_frozen = components.teacher is not None and all(
		not parameter.requires_grad for parameter in components.teacher.parameters()
	)
	return {
		'pass': frozen_exact and top_changed and head_changed and teacher_frozen,
		'teacher_all_frozen': teacher_frozen,
		'student_non_top_bitwise_init': frozen_exact,
		'student_top_parameter_changed': top_changed,
		'prototype_head_parameter_changed': head_changed,
		'top_block_parameter_delta_norm': _delta_norm(model_state, init, top_prefix),
		'prototype_head_parameter_norm': _state_norm(head_state),
	}


def _optimizer_contract_validation(payload: Mapping[str, object]) -> dict[str, object]:
	optimizer = _mapping(payload.get('optimizer_state_dict'), 'optimizer_state_dict')
	groups = optimizer.get('param_groups')
	if not isinstance(groups, Sequence) or isinstance(groups, str | bytes):
		raise TypeError('optimizer param_groups must be a sequence')
	by_name = {
		str(_mapping(group, 'optimizer group').get('name')): _mapping(
			group, 'optimizer group'
		)
		for group in groups
	}
	head = by_name.get('head', {})
	encoder = by_name.get('encoder', {})
	head_lr = float(head.get('lr', math.nan))
	encoder_lr = float(encoder.get('lr', math.nan))
	passed = (
		set(by_name) == {'head', 'encoder'}
		and math.isclose(head_lr, 3.0e-4)
		and math.isclose(encoder_lr, 1.0e-5)
		and bool(head.get('params'))
		and bool(encoder.get('params'))
	)
	return {
		'pass': passed,
		'groups': {
			name: {
				'lr': group.get('lr'),
				'parameter_count': len(group.get('params', [])),
			}
			for name, group in by_name.items()
		},
		'frozen_blocks_absent_from_optimizer': passed,
		'teacher_absent_from_optimizer': passed,
	}


def _historical_checkpoint_comparison(
	latest: Mapping[str, object], best: Mapping[str, object]
) -> dict[str, object]:
	historical_latest = load_checkpoint(
		HISTORICAL_PRETRAIN_ROOT / 'latest.pt', map_location='cpu'
	)
	historical_best = load_checkpoint(
		HISTORICAL_PRETRAIN_ROOT / 'best.pt', map_location='cpu'
	)
	return {
		'current_best_epoch': best.get('epoch'),
		'historical_best_epoch': historical_best.get('epoch'),
		'current_best_loss': _mapping(best.get('metrics'), 'metrics').get('loss'),
		'historical_best_loss': _mapping(historical_best.get('metrics'), 'metrics').get(
			'loss'
		),
		'current_latest_loss': _mapping(latest.get('metrics'), 'metrics').get('loss'),
		'historical_latest_loss': _mapping(
			historical_latest.get('metrics'), 'metrics'
		).get('loss'),
		'current_latest_prototype_loss': _mapping(latest.get('metrics'), 'metrics').get(
			'loss_prototype'
		),
		'historical_latest_prototype_loss': _mapping(
			historical_latest.get('metrics'), 'metrics'
		).get('loss_prototype'),
		'current_latest_distillation_loss': _mapping(
			latest.get('metrics'), 'metrics'
		).get('loss_distillation'),
		'historical_latest_distillation_loss': _mapping(
			historical_latest.get('metrics'), 'metrics'
		).get('loss_distillation'),
	}


def _embedding_diagnostic(
	current: np.ndarray, historical: np.ndarray, valid: np.ndarray
) -> dict[str, float | bool]:
	if current.shape != historical.shape or current.dtype != historical.dtype:
		raise ValueError('historical/current embedding shape or dtype mismatch')
	max_abs = 0.0
	abs_sum = 0.0
	count = 0
	cosine_sum = 0.0
	token_count = 0
	for start in range(0, current.shape[0], 4):
		stop = min(current.shape[0], start + 4)
		mask = valid[start:stop]
		left = np.asarray(current[start:stop][mask], dtype=np.float32)
		right = np.asarray(historical[start:stop][mask], dtype=np.float32)
		delta = np.abs(left - right)
		max_abs = max(max_abs, float(delta.max(initial=0.0)))
		abs_sum += float(delta.sum(dtype=np.float64))
		count += int(delta.size)
		dot = np.sum(left * right, axis=1, dtype=np.float64)
		norm = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
		cosine_sum += float(
			np.divide(dot, norm, out=np.ones_like(dot), where=norm > 0).sum()
		)
		token_count += int(left.shape[0])
	return {
		'valid_token_mask_exact': True,
		'max_absolute_delta': max_abs,
		'mean_absolute_delta': abs_sum / count,
		'mean_per_token_cosine_similarity': cosine_sum / token_count,
	}


def _nonfinite_valid_embedding_count(values: np.ndarray, valid: np.ndarray) -> int:
	count = 0
	for start in range(0, values.shape[0], 4):
		stop = min(values.shape[0], start + 4)
		count += int((~np.isfinite(values[start:stop][valid[start:stop]])).sum())
	return count


def _delta_norm(
	current: Mapping[str, torch.Tensor], init: Mapping[str, torch.Tensor], prefix: str
) -> float:
	return math.sqrt(
		sum(
			float((value.float() - init[name].float()).square().sum().item())
			for name, value in current.items()
			if name.startswith(prefix)
		)
	)


def _state_norm(state: Mapping[str, torch.Tensor]) -> float:
	return math.sqrt(
		sum(float(value.float().square().sum().item()) for value in state.values())
	)


def _tensor_hash(name: str, value: torch.Tensor) -> str:
	contiguous = value.detach().cpu().contiguous()
	digest = hashlib.sha256()
	digest.update(name.encode('utf-8'))
	digest.update(str(contiguous.dtype).encode('utf-8'))
	digest.update(str(tuple(contiguous.shape)).encode('utf-8'))
	digest.update(contiguous.view(torch.uint8).numpy().tobytes())
	return digest.hexdigest()


def _finite_metrics(metrics: Mapping[str, object]) -> bool:
	return bool(metrics) and all(
		not isinstance(value, bool)
		and isinstance(value, int | float)
		and math.isfinite(float(value))
		for value in metrics.values()
	)


def _token_metric(payload: Mapping[str, object], name: str) -> float:
	if name.startswith('class_'):
		_, class_id, metric = name.split('_', 2)
		values = _mapping(payload.get(f'per_class_{metric}'), f'per_class_{metric}')
		return float(values[class_id])
	return float(payload[name])


def _probe_exact(payload: Mapping[str, object]) -> bool:
	parity = _mapping(payload.get('parity'), 'probe parity')
	return all(
		_mapping(value, 'probe parity comparison').get('prediction_exact')
		and _mapping(value, 'probe parity comparison').get('confusion_matrix_exact')
		and _mapping(value, 'probe parity comparison').get('primary_metrics_exact')
		and _mapping(value, 'probe parity comparison').get('true_labels_exact')
		and _mapping(value, 'probe parity comparison').get(
			'validation_coordinates_exact'
		)
		for value in parity.values()
	)


def _nested_bool(payload: Mapping[str, object], keys: tuple[str, ...]) -> bool:
	value: object = payload
	for key in keys:
		if not isinstance(value, Mapping):
			return False
		value = value.get(key)
	return value is True


def _git_identity() -> dict[str, object]:
	root = Path(__file__).resolve().parents[3]
	status = _git_output(root, 'status', '--short')
	diff = _git_bytes(root, 'diff', '--binary', 'HEAD')
	return {
		'head': _git_output(root, 'rev-parse', 'HEAD').strip(),
		'dirty_status': status.splitlines(),
		'git_diff_sha256': hashlib.sha256(diff).hexdigest(),
	}


def _git_output(root: Path, *args: str) -> str:
	git = shutil.which('git')
	if git is None:
		raise RuntimeError('git is required for control provenance')
	return subprocess.check_output(  # noqa: S603
		[git, *args], cwd=root, text=True
	).strip()


def _git_bytes(root: Path, *args: str) -> bytes:
	git = shutil.which('git')
	if git is None:
		raise RuntimeError('git is required for control provenance')
	return subprocess.check_output([git, *args], cwd=root)  # noqa: S603


def _resolved_pretraining_config(config_path: Path) -> dict[str, object]:
	return resolve_strat_hmm_pretext_config(load_config(config_path))


def _identity(path: Path) -> dict[str, str]:
	if not path.is_file():
		raise FileNotFoundError(path)
	return {'path': str(path), 'sha256': file_sha256(path)}


def _canonical_sha256(value: Mapping[str, object]) -> str:
	encoded = json.dumps(
		value, default=str, sort_keys=True, separators=(',', ':')
	).encode()
	return hashlib.sha256(encoded).hexdigest()


def _control_identity_sha256(value: Mapping[str, object]) -> str:
	encoded = json.dumps(
		value, sort_keys=True, separators=(',', ':'), allow_nan=False
	).encode('utf-8')
	return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
	with path.open(encoding='utf-8') as handle:
		payload = json.load(handle)
	if not isinstance(payload, Mapping):
		raise TypeError(f'expected JSON object: {path}')
	return cast('Mapping[str, object]', payload)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	if not rows:
		raise ValueError('cannot write empty CSV')
	with path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


def _render_preflight_markdown(payload: Mapping[str, object]) -> str:
	migration = _mapping(payload.get('migration'), 'migration')
	decision = _mapping(migration.get('decision'), 'migration decision')
	inputs = _mapping(payload.get('inputs'), 'inputs')
	git = _mapping(payload.get('current_git'), 'git')
	config_identity = _mapping(inputs.get('training_config'), 'config')
	teacher_identity = _mapping(inputs.get('teacher_checkpoint'), 'teacher')
	student_identity = _mapping(inputs.get('student_init_checkpoint'), 'student')
	lines = [
		'# Current-code single-head K=6 control input manifest',
		'',
		f'- Model tag: `{payload["model_tag"]}`',
		f'- Migration status: `{decision["status"]}`',
		f'- Required rerun scope: `{decision["required_rerun_scope"]}`',
		f'- Current git SHA: `{git["head"]}`',
		f'- Training config SHA-256: `{config_identity["sha256"]}`',
		f'- Teacher SHA-256: `{teacher_identity["sha256"]}`',
		f'- Student init SHA-256: `{student_identity["sha256"]}`',
		'',
		'`finite_check_mode: strict` is an explicit current-code check-path '
		'setting. Historical M1 was reconstructed as `off`; the finite F3 source '
		'volume means this '
		'does not change supplied inputs, targets, or loss semantics.',
		'',
	]
	return '\n'.join(lines)


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return cast('Mapping[str, object]', value)


def _section(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	return _mapping(parent.get(key), key)


def _path(mapping: Mapping[str, object], key: str) -> Path:
	value = mapping.get(key)
	if not isinstance(value, str) or not value:
		raise TypeError(f'{key} must be a non-empty path string')
	return Path(value)


def _tensor_mapping(value: object, label: str) -> Mapping[str, torch.Tensor]:
	mapping = _mapping(value, label)
	if not all(
		isinstance(key, str) and isinstance(item, torch.Tensor)
		for key, item in mapping.items()
	):
		raise TypeError(f'{label} must contain tensors')
	return cast('Mapping[str, torch.Tensor]', mapping)


__all__ = [
	'HISTORICAL_M1_TAG',
	'MAE_TAG',
	'MODEL_TAG',
	'validate_current_k6_checkpoint',
	'validate_current_k6_embeddings',
	'write_control_preflight',
	'write_token_probe_comparison',
]
