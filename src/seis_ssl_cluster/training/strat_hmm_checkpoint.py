"""Checkpoint helpers for stratigraphic HMM pretext training."""


from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

import seis_ssl_cluster
from seis_ssl_cluster.config.pretraining import (
	CENTER_TRACE_CONSISTENCY_POLICY,
	CENTER_TRACE_EXPERIMENT_ROLE,
	CENTER_TRACE_MODEL_TAG,
	CENTER_TRACE_REPLACEMENT_INITIALIZATION,
	CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT,
	CENTER_TRACE_SUPERVISED_LOSS,
	CENTER_TRACE_VARIANT,
	PERIODIC_REFRESH_CENTER_UPDATE_SEMANTICS,
	PERIODIC_REFRESH_CHECKPOINT_SELECTION_POLICY,
	PERIODIC_REFRESH_EMBEDDING_SEMANTICS,
	PERIODIC_REFRESH_EXPERIMENT_ROLE,
	PERIODIC_REFRESH_MODEL_ROLE,
	PERIODIC_REFRESH_MODEL_TAG,
	PERIODIC_REFRESH_PREPROCESSING_POLICY,
	PERIODIC_REFRESH_SCHEDULE,
	PERIODIC_REFRESH_SCHEDULE_SEMANTICS,
	PERIODIC_REFRESH_SEMANTICS,
	PERIODIC_REFRESH_TARGET_ACTIVATION_POLICY,
	PERIODIC_REFRESH_TARGET_REPRESENTATION,
	PERIODIC_REFRESH_VARIANT,
	_is_periodic_refresh_config,
	_validate_periodic_refresh_config,
)
from seis_ssl_cluster.stratigraphy.lateral_targets import (
	load_multi_head_lateral_target_manifest,
)
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.stratigraphy.periodic_refresh import (
	load_periodic_refresh_generation,
)
from seis_ssl_cluster.stratigraphy.prototypes import (
	MultiResolutionOrderedPrototypeHeads,
	OrderedPrototypeHead,
)
from seis_ssl_cluster.stratigraphy.state_posterior import (
	load_multi_head_state_posterior_manifest,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus_targets import (
	load_multi_head_xy_neighbor_consensus_target_manifest,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_unanimous_targets import (
	load_multi_head_xy_neighbor_unanimous_target_manifest,
)
from seis_ssl_cluster.training.checkpoint import capture_rng_state, load_checkpoint


@dataclass(frozen=True)
class StratRollingCheckpointResult:
	"""Result of a rolling strat HMM checkpoint write."""

	latest_path: Path
	best_path: Path | None
	best_score: float | None
	best_updated: bool
	checkpoint_selection: Mapping[str, object] | None = None
	selected_path: Path | None = None


_CHECKPOINT_SELECTION_SCHEMA_VERSION = 1
_CHECKPOINT_SELECTION_CRITERION = 'metrics.loss'
_CHECKPOINT_SELECTION_POLICY = 'strictly_lower_loss_v1'
_CHECKPOINT_SELECTION_TRANSACTION_SCHEMA_VERSION = 1
_CHECKPOINT_SELECTION_TRANSACTION_NAME = '.checkpoint_selection_transaction.json'

_XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION = 'xy_neighbor_consensus_hard_labels_v1'
_XY_NEIGHBOR_CONSENSUS_TARGET_SEMANTICS = (
	'xy_neighbor_consensus_hard_label_smoothing_v1'
)
_XY_NEIGHBOR_CONSENSUS_CONSISTENCY_POLICY = 'disabled_for_xy_neighbor_consensus_v1'
_XY_NEIGHBOR_CONSENSUS_EXPERIMENT_ROLE = (
	'multi_head_ordered_xy_neighbor_consensus_hard_pretext'
)
_XY_NEIGHBOR_UNANIMOUS_TARGET_REPRESENTATION = 'xy_neighbor_unanimous_hard_labels_v1'
_XY_NEIGHBOR_UNANIMOUS_TARGET_SEMANTICS = 'xy_neighbor_unanimous_outlier_correction_v1'
_XY_NEIGHBOR_UNANIMOUS_CONSISTENCY_POLICY = 'disabled_for_xy_neighbor_unanimous_v1'
_XY_NEIGHBOR_UNANIMOUS_EXPERIMENT_ROLE = (
	'multi_head_ordered_xy_neighbor_unanimous_hard_pretext'
)

_CENTER_TRACE_TARGET_REPRESENTATION = 'hard_viterbi_labels_v1'
_CENTER_TRACE_HEAD_SPEC = 'multi_resolution_ordered_prototypes_v1'
_CENTER_TRACE_HEAD_KS = (6, 8, 10)
_CENTER_TRACE_CHECKPOINT_IDENTITY_FIELDS = frozenset(
	{
		'schema_version',
		'head_spec',
		'head_ks',
		'target_representation',
		'target_manifest_sha256',
		'target_manifest',
		'per_head_targets',
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
		'prototype_weight',
		'usage_weight',
		'consistency_weight',
		'consistency_beta',
		'distillation_weight',
		'model_tag',
		'output_root',
		'scientific_identity_sha256',
		'student_state_sha256',
		'stratigraphy_state_sha256',
		'spatial_context_state_sha256',
		'initial_spatial_context_state_sha256',
		'optimizer_group_identity',
		'teacher_checkpoint_sha256',
		'student_init_checkpoint_sha256',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
	}
)

_PERIODIC_REFRESH_CHECKPOINT_IDENTITY_FIELDS = (
	(
		_CENTER_TRACE_CHECKPOINT_IDENTITY_FIELDS
		- frozenset({'target_manifest_sha256', 'target_manifest', 'per_head_targets'})
	)
	| frozenset(
		{
			'model_role',
			'initial_hard_target_manifest_sha256',
			'initial_hard_target_manifest',
			'initial_per_head_targets',
			'initial_hmm_artifacts',
			'target_refresh_semantics',
			'refresh_schedule_semantics',
			'refresh_after_epochs',
			'hmm_iterations_per_refresh',
			'embedding_source',
			'embedding_mode',
			'refresh_embedding_semantics',
			'center_initialization',
			'center_update',
			'center_update_semantics',
			'preprocessing_policy',
			'target_activation_policy',
			'empty_state_policy',
			'checkpoint_selection_policy',
			'fixed_preprocessor_sha256',
			'fixed_residualizer_sha256',
			'fixed_clustering_config_sha256',
			'source_embedding_metadata_sha256',
			'source_valid_token_hashes',
			'feature_dimension',
			'generation_root',
			'target_refresh_state_sha256',
		}
	)
)

_PERIODIC_REFRESH_STATE_SCHEMA_VERSION = 1
_PERIODIC_REFRESH_STATE_KEYS = frozenset(
	{
		'schema_version',
		'active_generation_index',
		'active_generation_id',
		'active_generation_manifest_path',
		'active_generation_manifest_sha256',
		'active_generation_content_sha256',
		'active_target_manifest_path',
		'active_target_manifest_sha256',
		'periodic_refresh_chain_path',
		'periodic_refresh_chain_sha256',
		'last_completed_refresh_epoch',
		'next_scheduled_refresh_epoch',
		'refresh_phase',
		'source_student_state_sha256',
		'fixed_preprocessing_hmm_identity_sha256',
		'generations',
	}
)
_PERIODIC_REFRESH_GENERATION_KEYS = frozenset(
	{
		'generation_index',
		'generation_id',
		'manifest_path',
		'manifest_sha256',
		'generation_content_sha256',
	}
)
_PERIODIC_REFRESH_CHAIN_KEYS = frozenset(
	{
		'schema_version',
		'semantics',
		'refresh_after_epochs',
		'fixed_preprocessing_hmm_identity_sha256',
		'generations',
	}
)
_PERIODIC_REFRESH_CHAIN_GENERATION_KEYS = frozenset(
	{
		'generation_index',
		'generation_id',
		'refresh_after_epoch',
		'previous_generation_manifest_sha256',
		'source_student_state_sha256',
		'manifest_path',
		'manifest_sha256',
		'generation_content_sha256',
	}
)
_PERIODIC_REFRESH_CHAIN_SEMANTICS = 'periodic_student_hmm_refresh_chain_v1'

_XY_NEIGHBOR_CONSENSUS_CHECKPOINT_IDENTITY_FIELDS = frozenset(
	{
		'schema_version',
		'head_spec',
		'head_ks',
		'target_representation',
		'target_semantics',
		'xy_neighbor_consensus_target_manifest_sha256',
		'xy_neighbor_consensus_target_manifest',
		'per_head_xy_neighbor_consensus_targets',
		'source_hard_manifest_sha256',
		'xy_neighbor_consensus_smoothing',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
		'model_tag',
		'output_root',
		'scientific_identity_sha256',
		'stratigraphy_state_sha256',
		'optimizer_group_identity',
		'teacher_checkpoint_sha256',
		'student_init_checkpoint_sha256',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
	}
)

_XY_NEIGHBOR_CONSENSUS_SCIENTIFIC_IDENTITY_FIELDS = frozenset(
	{
		'experiment_role',
		'variant',
		'head_spec',
		'head_ks',
		'head_projection_dim',
		'head_temperature',
		'head_normalize',
		'target_representation',
		'target_semantics',
		'xy_neighbor_consensus_target_manifest_sha256',
		'xy_neighbor_consensus_target_head_hashes',
		'source_hard_manifest_sha256',
		'xy_neighbor_consensus_smoothing',
		'supervised_loss',
		'consistency_policy',
		'prototype_weight',
		'usage_weight',
		'consistency_weight',
		'consistency_beta',
		'distillation_weight',
		'teacher_checkpoint',
		'student_init_checkpoint',
		'student_unfreeze_top_blocks',
		'model',
		'data',
		'zero_mask',
		'train',
	}
)

_XY_NEIGHBOR_UNANIMOUS_CHECKPOINT_IDENTITY_FIELDS = frozenset(
	{
		'schema_version',
		'head_spec',
		'head_ks',
		'target_representation',
		'target_semantics',
		'xy_neighbor_unanimous_target_manifest_sha256',
		'xy_neighbor_unanimous_target_manifest',
		'per_head_xy_neighbor_unanimous_targets',
		'source_hard_manifest_sha256',
		'xy_neighbor_unanimous_smoothing',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
		'model_tag',
		'output_root',
		'scientific_identity_sha256',
		'stratigraphy_state_sha256',
		'optimizer_group_identity',
		'teacher_checkpoint_sha256',
		'student_init_checkpoint_sha256',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
	}
)

_XY_NEIGHBOR_UNANIMOUS_SCIENTIFIC_IDENTITY_FIELDS = frozenset(
	{
		'experiment_role',
		'variant',
		'head_spec',
		'head_ks',
		'head_projection_dim',
		'head_temperature',
		'head_normalize',
		'target_representation',
		'target_semantics',
		'xy_neighbor_unanimous_target_manifest_sha256',
		'xy_neighbor_unanimous_target_head_hashes',
		'source_hard_manifest_sha256',
		'xy_neighbor_unanimous_smoothing',
		'supervised_loss',
		'consistency_policy',
		'prototype_weight',
		'usage_weight',
		'consistency_weight',
		'consistency_beta',
		'distillation_weight',
		'teacher_checkpoint',
		'student_init_checkpoint',
		'student_unfreeze_top_blocks',
		'model',
		'data',
		'zero_mask',
		'train',
	}
)


def save_strat_hmm_rolling_checkpoint(  # noqa: PLR0913
	checkpoint_dir: str | Path,
	*,
	student: torch.nn.Module,
	head: torch.nn.Module,
	spatial_context: torch.nn.Module | None = None,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	mae_config: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	metrics: Mapping[str, float],
	global_step: int,
	checkpoint_kind: Literal['step', 'epoch', 'refresh'],
	batch_index: int | None,
	amp_enabled: bool = False,
	scaler: torch.amp.GradScaler | None = None,
	rng_state: Mapping[str, object] | None = None,
	best_score: float | None = None,
	checkpoint_selection: Mapping[str, object] | None = None,
	trainability_summary: Mapping[str, object] | None = None,
	control_identity: Mapping[str, object] | None = None,
	target_refresh_state: Mapping[str, object] | None = None,
	epoch_metrics_state: Mapping[str, object] | None = None,
) -> StratRollingCheckpointResult:
	"""Write rolling ``latest.pt`` and update ``best.pt`` on lower loss."""
	checkpoint_root = Path(checkpoint_dir)
	checkpoint_root.mkdir(parents=True, exist_ok=True)
	if _is_periodic_refresh_config(stratigraphy_config):
		return _save_periodic_refresh_rolling_checkpoint(
			checkpoint_root,
			student=student,
			head=head,
			spatial_context=spatial_context,
			optimizer=optimizer,
			epoch=epoch,
			mae_config=mae_config,
			stratigraphy_config=stratigraphy_config,
			metrics=metrics,
			global_step=global_step,
			checkpoint_kind=checkpoint_kind,
			batch_index=batch_index,
			amp_enabled=amp_enabled,
			scaler=scaler,
			rng_state=rng_state,
			best_score=best_score,
			checkpoint_selection=checkpoint_selection,
			trainability_summary=trainability_summary,
			control_identity=control_identity,
			target_refresh_state=target_refresh_state,
			epoch_metrics_state=epoch_metrics_state,
		)
	is_multi_head = _is_multi_head_config(stratigraphy_config)
	if is_multi_head:
		recover_strat_hmm_rolling_checkpoint(checkpoint_root)
	selection_best_score = (
		float(selected_checkpoint_selection_event(checkpoint_selection)['loss'])
		if checkpoint_selection is not None
		else None
	)
	if is_multi_head and best_score != selection_best_score:
		raise ValueError('checkpoint selection best_score does not match history')
	selection = (
		_update_checkpoint_selection(
			checkpoint_selection,
			epoch=epoch,
			global_step=global_step,
			checkpoint_kind=checkpoint_kind,
			batch_index=batch_index,
			loss=_required_loss_score(metrics),
		)
		if is_multi_head
		else None
	)
	score = _loss_score(metrics)
	best_updated = (
		bool(selection['events'][-1]['best_updated'])
		if selection is not None
		else _is_improved(score, best_score)
	)
	if selection is not None and best_updated:
		_write_checkpoint_selection_transaction(checkpoint_root, selection)
	latest_path = save_strat_hmm_checkpoint(
		checkpoint_root / 'latest.pt',
		student=student,
		head=head,
		spatial_context=spatial_context,
		optimizer=optimizer,
		epoch=epoch,
		mae_config=mae_config,
		stratigraphy_config=stratigraphy_config,
		metrics=metrics,
		global_step=global_step,
		amp_enabled=amp_enabled,
		scaler=scaler,
		checkpoint_kind=checkpoint_kind,
		batch_index=batch_index,
		rng_state=rng_state,
		trainability_summary=trainability_summary,
		control_identity=control_identity,
		checkpoint_selection=selection,
	)
	resolved_best_score = (
		float(selection['selected']['loss']) if selection is not None else best_score
	)
	best_path = checkpoint_root / 'best.pt'
	if best_updated:
		_copy_checkpoint_atomic(latest_path, best_path)
		resolved_best_score = score
	if selection is not None:
		_write_checkpoint_selection_reports(checkpoint_root, selection)
	if selection is not None and best_updated:
		_checkpoint_selection_transaction_path(checkpoint_root).unlink()
	return StratRollingCheckpointResult(
		latest_path=latest_path,
		best_path=best_path,
		best_score=resolved_best_score,
		best_updated=best_updated,
		checkpoint_selection=selection,
	)


def _save_periodic_refresh_rolling_checkpoint(  # noqa: PLR0913
	checkpoint_root: Path,
	*,
	student: torch.nn.Module,
	head: torch.nn.Module,
	spatial_context: torch.nn.Module | None,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	mae_config: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	metrics: Mapping[str, float],
	global_step: int,
	checkpoint_kind: Literal['step', 'epoch', 'refresh'],
	batch_index: int | None,
	amp_enabled: bool,
	scaler: torch.amp.GradScaler | None,
	rng_state: Mapping[str, object] | None,
	best_score: float | None,
	checkpoint_selection: Mapping[str, object] | None,
	trainability_summary: Mapping[str, object] | None,
	control_identity: Mapping[str, object] | None,
	target_refresh_state: Mapping[str, object] | None,
	epoch_metrics_state: Mapping[str, object] | None,
) -> StratRollingCheckpointResult:
	"""Write periodic rolling state without loss-based model selection."""
	if best_score is not None:
		raise ValueError('periodic refresh checkpoints do not accept best_score')
	best_path = checkpoint_root / 'best.pt'
	if best_path.exists() or best_path.is_symlink():
		raise ValueError(
			'periodic refresh checkpoint root must not contain best.pt'
		)
	if target_refresh_state is None:
		raise ValueError('periodic refresh rolling checkpoint requires refresh state')
	if checkpoint_kind == 'step' and epoch_metrics_state is None:
		raise ValueError('periodic step checkpoint requires epoch metrics state')
	refresh_state = _validated_target_refresh_state(
		target_refresh_state, expected_config=stratigraphy_config
	)
	selection = _periodic_checkpoint_selection_for_payload(
		selection=checkpoint_selection,
		epoch=epoch,
		global_step=global_step,
		checkpoint_kind=checkpoint_kind,
		batch_index=batch_index,
		target_refresh_state=refresh_state,
		train_epochs=_periodic_train_epochs(stratigraphy_config),
	)
	latest_path = save_strat_hmm_checkpoint(
		checkpoint_root / 'latest.pt',
		student=student,
		head=head,
		spatial_context=spatial_context,
		optimizer=optimizer,
		epoch=epoch,
		mae_config=mae_config,
		stratigraphy_config=stratigraphy_config,
		metrics=metrics,
		global_step=global_step,
		amp_enabled=amp_enabled,
		scaler=scaler,
		checkpoint_kind=checkpoint_kind,
		batch_index=batch_index,
		rng_state=rng_state,
		trainability_summary=trainability_summary,
		control_identity=control_identity,
		checkpoint_selection=selection,
		target_refresh_state=refresh_state,
		epoch_metrics_state=epoch_metrics_state,
	)
	selected_path: Path | None = None
	if selection['selected'] is not None:
		selected_path = checkpoint_root / 'selected.pt'
		_copy_checkpoint_atomic(latest_path, selected_path)
	_atomic_json(
		checkpoint_root / 'checkpoint_selection_summary.json',
		selection,
	)
	return StratRollingCheckpointResult(
		latest_path=latest_path,
		best_path=None,
		best_score=None,
		best_updated=False,
		checkpoint_selection=selection,
		selected_path=selected_path,
	)


def recover_strat_hmm_rolling_checkpoint(checkpoint_dir: str | Path) -> None:
	"""Finish an interrupted multi-head ``latest.pt``/``best.pt`` update.

	A transaction record is only left behind while an improved latest checkpoint
	is waiting to be copied to ``best.pt``.  Recovery verifies that the latest
	payload is that recorded candidate before repairing the derived best file.
	"""
	checkpoint_root = Path(checkpoint_dir)
	transaction_path = _checkpoint_selection_transaction_path(checkpoint_root)
	if not transaction_path.is_file():
		return
	transaction = _load_checkpoint_selection_transaction(transaction_path)
	latest_path = checkpoint_root / 'latest.pt'
	if not latest_path.is_file():
		transaction_path.unlink()
		return
	latest = load_checkpoint(latest_path, map_location='cpu')
	selection = _validated_checkpoint_selection(latest.get('checkpoint_selection'))
	if checkpoint_selection_sha256(selection) != transaction['selection_sha256']:
		transaction_path.unlink()
		return
	selected = selected_checkpoint_selection_event(selection)
	if selected != transaction['selected']:
		raise ValueError('checkpoint selection transaction selected event mismatch')
	events = selection['events']
	if not isinstance(events, list) or not isinstance(events[-1], Mapping):
		raise TypeError('checkpoint selection transaction events are invalid')
	last = events[-1]
	if (
		_selection_event_identity_from_event(last) != selected
		or last.get('best_updated') is not True
	):
		raise ValueError('checkpoint selection transaction does not select latest')
	_validate_checkpoint_selection_payload_binding(latest, selection)
	best_path = checkpoint_root / 'best.pt'
	if not _checkpoint_payload_matches_selected_event(best_path, selected):
		_copy_checkpoint_atomic(latest_path, best_path)
	_write_checkpoint_selection_reports(checkpoint_root, selection)
	transaction_path.unlink()


def save_strat_hmm_checkpoint(  # noqa: C901, PLR0913
	path: str | Path,
	*,
	student: torch.nn.Module,
	head: torch.nn.Module,
	spatial_context: torch.nn.Module | None = None,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	mae_config: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	metrics: Mapping[str, float],
	global_step: int,
	checkpoint_kind: Literal['step', 'epoch', 'refresh'],
	batch_index: int | None,
	amp_enabled: bool = False,
	scaler: torch.amp.GradScaler | None = None,
	rng_state: Mapping[str, object] | None = None,
	trainability_summary: Mapping[str, object] | None = None,
	control_identity: Mapping[str, object] | None = None,
	checkpoint_selection: Mapping[str, object] | None = None,
	target_refresh_state: Mapping[str, object] | None = None,
	epoch_metrics_state: Mapping[str, object] | None = None,
) -> Path:
	"""Atomically save an extraction-compatible strat HMM checkpoint."""
	checkpoint_path = Path(path)
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	model_state_dict = _state_dict_cpu(student)
	stratigraphy_state_dict = _state_dict_cpu(head)
	spatial_context_state_dict = (
		None if spatial_context is None else _state_dict_cpu(spatial_context)
	)
	_validate_checkpoint_inputs(
		model_state_dict=model_state_dict,
		stratigraphy_state_dict=stratigraphy_state_dict,
		optimizer=optimizer,
		stratigraphy_config=stratigraphy_config,
		student=student,
		head=head,
		spatial_context=spatial_context,
		spatial_context_state_dict=spatial_context_state_dict,
	)
	payload = {
		'model_state_dict': model_state_dict,
		'stratigraphy_state_dict': stratigraphy_state_dict,
		'stratigraphy_config': _to_plain_value(stratigraphy_config),
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': int(epoch),
		'global_step': int(global_step),
		'amp_enabled': bool(amp_enabled),
		'scaler_state_dict': None if scaler is None else scaler.state_dict(),
		'config': _to_plain_value(mae_config),
		'package_version': getattr(seis_ssl_cluster, '__version__', None),
		'metrics': dict(metrics),
		'trainability_summary': (
			{}
			if trainability_summary is None
			else _to_plain_value(trainability_summary)
		),
		'rng_state': dict(capture_rng_state() if rng_state is None else rng_state),
		'training_state': {
			'schema_version': 1,
			'stage': 'train_strat_hmm_pretext',
			'checkpoint_kind': checkpoint_kind,
			'batch_index': batch_index,
		},
	}
	if epoch_metrics_state is not None:
		payload['epoch_metrics_state'] = _validated_epoch_metrics_state(
			epoch_metrics_state
		)
	elif _is_periodic_refresh_config(stratigraphy_config) and checkpoint_kind == 'step':
		raise ValueError('schema-8 step checkpoint requires epoch metrics state')
	if control_identity is not None:
		payload['control_identity'] = _to_plain_value(control_identity)
	if _is_periodic_refresh_config(stratigraphy_config):
		if target_refresh_state is None:
			raise ValueError('schema-8 checkpoint requires target_refresh_state')
		payload['target_refresh_state'] = _validated_target_refresh_state(
			target_refresh_state,
			expected_config=stratigraphy_config,
		)
	elif target_refresh_state is not None:
		raise ValueError(
			'target_refresh_state is only valid for periodic-refresh checkpoints'
		)
	if spatial_context_state_dict is not None:
		payload['spatial_context_state_dict'] = spatial_context_state_dict
		payload['spatial_context_state_sha256'] = _state_sha256(
			spatial_context_state_dict
		)
		initial_states = _required_mapping(
			control_identity or {}, 'initial_state_sha256'
		)
		payload['initial_spatial_context_state_sha256'] = _required_sha256(
			initial_states.get('spatial_context'),
			'initial_state_sha256.spatial_context',
		)
	if _is_periodic_refresh_config(stratigraphy_config):
		if checkpoint_selection is None:
			checkpoint_selection = _periodic_checkpoint_selection_for_payload(
				epoch=epoch,
				global_step=global_step,
				checkpoint_kind=checkpoint_kind,
				batch_index=batch_index,
				target_refresh_state=payload['target_refresh_state'],
				train_epochs=_periodic_train_epochs(stratigraphy_config),
			)
		payload['checkpoint_selection'] = _validated_periodic_checkpoint_selection(
			checkpoint_selection,
			train_epochs=_periodic_train_epochs(stratigraphy_config),
		)
		payload['stratigraphy_checkpoint'] = _periodic_refresh_checkpoint_identity(
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
			spatial_context=spatial_context,
			spatial_context_state_dict=spatial_context_state_dict,
			control_identity=control_identity,
			optimizer=optimizer,
			student=student,
			head=head,
			target_refresh_state=payload['target_refresh_state'],
		)
		_validate_periodic_training_state(
			payload, train_epochs=_periodic_train_epochs(stratigraphy_config)
		)
		_validate_periodic_checkpoint_selection_payload_binding(
			payload, payload['checkpoint_selection']
		)
	elif _is_multi_head_config(stratigraphy_config):
		if checkpoint_selection is None:
			checkpoint_selection = _update_checkpoint_selection(
				None,
				epoch=epoch,
				global_step=global_step,
				checkpoint_kind=checkpoint_kind,
				batch_index=batch_index,
				loss=_required_loss_score(metrics),
			)
		payload['checkpoint_selection'] = _validated_checkpoint_selection(
			checkpoint_selection
		)
		payload['stratigraphy_checkpoint'] = _multi_head_checkpoint_identity(
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
			control_identity=control_identity,
			optimizer=optimizer,
			student=student,
			head=head,
			spatial_context=spatial_context,
			spatial_context_state_dict=spatial_context_state_dict,
		)
	return _atomic_torch_save(checkpoint_path, payload)


def validate_stratigraphy_checkpoint_payload(  # noqa: C901, PLR0912, PLR0913, PLR0915
	payload: Mapping[str, object],
	*,
	expected_config: Mapping[str, object] | None = None,
	expected_optimizer: torch.optim.Optimizer | None = None,
	expected_student: torch.nn.Module | None = None,
	expected_head: torch.nn.Module | None = None,
	expected_spatial_context: torch.nn.Module | None = None,
) -> None:
	"""Validate the versioned multi-head identity before loading any state."""
	identity = payload.get('stratigraphy_checkpoint')
	config = payload.get('stratigraphy_config')
	if identity is None:
		if isinstance(config, Mapping) and _is_multi_head_config(config):
			raise ValueError('multi-head checkpoint is missing versioned identity')
		return
	if not isinstance(identity, Mapping):
		raise TypeError('checkpoint stratigraphy_checkpoint must be a mapping')
	if identity.get('schema_version') not in {2, 3, 4, 5, 6, 7, 8}:
		raise ValueError('unsupported stratigraphy checkpoint schema_version')
	if identity.get('head_spec') != 'multi_resolution_ordered_prototypes_v1':
		raise ValueError('unsupported stratigraphy multi-head head_spec')
	state = payload.get('stratigraphy_state_dict')
	if not isinstance(config, Mapping) or not isinstance(state, Mapping):
		raise TypeError('multi-head checkpoint requires config and head state mappings')
	expected_schema_version = (
		8
		if _is_periodic_refresh_config(config)
		else 7
		if _is_center_trace_config(config)
		else 6
		if _is_xy_neighbor_unanimous_multi_head_config(config)
		else 5
		if _is_xy_neighbor_consensus_multi_head_config(config)
		else (
			4
			if _is_lateral_multi_head_config(config)
			else 3
			if _is_soft_multi_head_config(config)
			else 2
		)
	)
	aux_payload_keys = {
		'spatial_context_state_dict',
		'spatial_context_state_sha256',
		'initial_spatial_context_state_sha256',
	}
	unsupported_aux_payload_keys = sorted(
		{
			str(key)
			for key in payload
			if str(key).startswith('spatial_context')
			and str(key) not in aux_payload_keys
		}
	)
	if unsupported_aux_payload_keys:
		raise ValueError(
			'checkpoint has unsupported spatial_context field(s): '
			f'{unsupported_aux_payload_keys!r}'
		)
	if expected_schema_version not in {7, 8}:
		aux_fields = sorted(set(payload) & aux_payload_keys)
		if aux_fields:
			raise ValueError(
				'spatial_context fields are only valid for schema-7/schema-8 '
				'checkpoints: '
				f'{aux_fields!r}'
			)
		identity_aux_fields = sorted(
			{
				str(key)
				for key in identity
				if str(key).startswith('spatial_context')
			}
		)
		if identity_aux_fields:
			raise ValueError(
				'schema-2-to-6 checkpoint identity must not contain spatial_context '
				f'field(s): {identity_aux_fields!r}'
			)
	if identity.get('schema_version') != expected_schema_version:
		raise ValueError(
			'multi-head checkpoint schema_version does not match target representation'
		)
	_validate_multi_head_identity(
		identity=identity,
		stratigraphy_config=config,
		stratigraphy_state_dict=state,
	)
	if expected_schema_version == 8:
		target_refresh_state = payload.get('target_refresh_state')
		if not isinstance(target_refresh_state, Mapping):
			raise ValueError('schema-8 checkpoint is missing target_refresh_state')
		validated_refresh_state = _validated_target_refresh_state(
			target_refresh_state,
			expected_config=config,
			recovery_source_student_state_sha256=(
				identity.get('student_state_sha256')
				if isinstance(identity.get('student_state_sha256'), str)
				else None
			),
		)
		if identity.get('target_refresh_state_sha256') != _canonical_sha256(
			validated_refresh_state
		):
			raise ValueError(
				'schema-8 target_refresh_state hash does not match identity'
			)
		_validate_periodic_training_state(
			payload, train_epochs=_periodic_train_epochs(config)
		)
		selection = _validated_periodic_checkpoint_selection(
			payload.get('checkpoint_selection'),
			train_epochs=_periodic_train_epochs(config),
		)
		_validate_periodic_checkpoint_selection_payload_binding(payload, selection)
	elif 'target_refresh_state' in payload:
		raise ValueError(
			'target_refresh_state is only valid for schema-8 checkpoints'
		)
	if expected_schema_version in {7, 8}:
		model_state = payload.get('model_state_dict')
		if not isinstance(model_state, Mapping):
			raise TypeError(
				f'schema-{expected_schema_version} model_state_dict must be a mapping'
			)
		if any(not isinstance(value, torch.Tensor) for value in model_state.values()):
			raise TypeError(
				f'schema-{expected_schema_version} model_state_dict values must '
				'be tensors'
			)
		student_state_sha256 = _required_sha256(
			identity.get('student_state_sha256'),
			'checkpoint student_state_sha256',
		)
		if student_state_sha256 != _state_sha256(model_state):
			raise ValueError('checkpoint student state SHA-256 mismatch')
		spatial_state = payload.get('spatial_context_state_dict')
		if not isinstance(spatial_state, Mapping):
			raise ValueError(
				f'schema-{expected_schema_version} checkpoint is missing '
				'spatial_context_state_dict'
			)
		_validate_finite_state_dict(
			{
				str(key): value
				for key, value in spatial_state.items()
				if isinstance(value, torch.Tensor)
			},
			label='spatial_context_state_dict',
		)
		if any(not isinstance(value, torch.Tensor) for value in spatial_state.values()):
			raise TypeError('spatial_context_state_dict values must be tensors')
		if set(spatial_state) != {'replacement_token'}:
			raise ValueError(
				f'schema-{expected_schema_version} spatial_context_state_dict must '
				'contain only replacement_token'
			)
		replacement_state = spatial_state['replacement_token']
		model_config = _required_mapping(config, 'model')
		model_state = payload.get('model_state_dict')
		if not isinstance(model_state, Mapping):
			raise TypeError(
				f'schema-{expected_schema_version} model_state_dict must be a mapping'
			)
		model_dtypes = {
			value.dtype
			for value in model_state.values()
			if isinstance(value, torch.Tensor) and value.is_floating_point()
		}
		if not model_dtypes or replacement_state.dtype not in model_dtypes:
			raise TypeError(
				f'schema-{expected_schema_version} replacement_token dtype does not '
				'match model state dtype'
			)
		if (
			not replacement_state.is_floating_point()
			or replacement_state.ndim != 1
			or replacement_state.shape[0] != model_config.get('encoder_dim')
		):
			raise ValueError(
				f'schema-{expected_schema_version} replacement_token shape or dtype '
				'does not match model encoder'
			)
		spatial_sha256 = _required_sha256(
			payload.get('spatial_context_state_sha256'),
			'checkpoint spatial_context_state_sha256',
		)
		if spatial_sha256 != _state_sha256(spatial_state):
			raise ValueError('checkpoint spatial_context state SHA-256 mismatch')
		initial_spatial_sha256 = _required_sha256(
			payload.get('initial_spatial_context_state_sha256'),
			'checkpoint initial_spatial_context_state_sha256',
		)
		identity_spatial_sha256 = _required_sha256(
			identity.get('spatial_context_state_sha256'),
			'checkpoint identity spatial_context_state_sha256',
		)
		if identity_spatial_sha256 != spatial_sha256:
			raise ValueError(
				'checkpoint spatial_context state hash does not match identity'
			)
		if initial_spatial_sha256 != identity.get(
			'initial_spatial_context_state_sha256'
		):
			raise ValueError(
				'checkpoint initial spatial_context state hash does not match identity'
			)
		if expected_spatial_context is not None:
			expected_state = expected_spatial_context.state_dict()
			if set(expected_state) != set(spatial_state):
				raise ValueError(
					f'schema-{expected_schema_version} spatial_context state keys do '
					'not match current module'
				)
			for key, expected_value in expected_state.items():
				actual_value = spatial_state[key]
				if (
					actual_value.shape != expected_value.shape
					or actual_value.dtype != expected_value.dtype
				):
					raise ValueError(
					f'schema-{expected_schema_version} spatial_context state '
					'geometry or dtype does not match current module'
				)
	elif expected_spatial_context is not None:
		raise ValueError(
			'spatial_context is only valid for schema-7/schema-8 checkpoints'
		)
	if expected_schema_version == 6:
		_validate_xy_neighbor_unanimous_control_identity(
			identity=identity,
			control_identity=payload.get('control_identity'),
		)
	if expected_schema_version != 8:
		selection = _validated_checkpoint_selection(payload.get('checkpoint_selection'))
		_validate_checkpoint_selection_payload_binding(payload, selection)
	if not _optimizer_state_group_identity_matches(
		payload.get('optimizer_state_dict'),
		identity.get('optimizer_group_identity'),
	):
		raise ValueError(
			'checkpoint optimizer state groups do not match optimizer group identity'
		)
	if expected_config is not None:
		_validate_expected_multi_head_identity(
			identity,
			expected_config,
			expected_student=expected_student,
			expected_head=expected_head,
			expected_spatial_context=expected_spatial_context,
		)
	if expected_optimizer is not None:
		_validate_expected_optimizer_group_identity(
			identity=identity,
			optimizer_state_dict=payload.get('optimizer_state_dict'),
			expected_optimizer=expected_optimizer,
			expected_student=expected_student,
			expected_head=expected_head,
			expected_spatial_context=expected_spatial_context,
		)


def inspect_stratigraphy_checkpoint(
	payload: Mapping[str, object],
	*,
	expected_config: Mapping[str, object] | None = None,
	expected_optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
	"""Return a machine-readable checkpoint identity and state summary."""
	validate_stratigraphy_checkpoint_payload(payload)
	result: dict[str, object] = {
		'stratigraphy_checkpoint': _to_plain_value(
			payload.get('stratigraphy_checkpoint', {'schema_version': 1}),
		),
		'model_state': _state_summary(payload.get('model_state_dict')),
		'stratigraphy_state': _state_summary(payload.get('stratigraphy_state_dict')),
		'optimizer_groups': _optimizer_group_summary(
			payload.get('optimizer_state_dict')
		),
		'best_selection_metric': 'metrics.loss',
		'embedding_compatible': isinstance(payload.get('model_state_dict'), Mapping),
		'resume_compatibility': _resume_compatibility_result(
			payload,
			expected_config=expected_config,
			expected_optimizer=expected_optimizer,
		),
	}
	identity = payload.get('stratigraphy_checkpoint')
	if isinstance(identity, Mapping) and identity.get('schema_version') == 8:
		target_refresh_state = payload.get('target_refresh_state')
		if not isinstance(target_refresh_state, Mapping):
			raise TypeError('schema-8 target_refresh_state must be a mapping')
		result.update(
			{
				'best_selection_metric': None,
				'representation': identity['target_representation'],
				'objective': identity['objective_semantics'],
				'selection_policy': identity['checkpoint_selection_policy'],
				'active_generation_id': target_refresh_state['active_generation_id'],
			}
		)
	elif isinstance(identity, Mapping) and identity.get('schema_version') == 7:
		result.update(
			{
				'representation': identity['target_representation'],
				'objective': identity['objective_semantics'],
				'mask_policy': {
					'objective': identity['objective_semantics'],
					'mask_semantics': identity['mask_semantics'],
					'column_fraction': identity['column_fraction'],
					'selection_policy': identity['selection_policy'],
					'replacement': identity['replacement'],
					'replacement_initialization': identity[
						'replacement_initialization'
					],
					'rng_policy': identity['rng_policy'],
				},
				'auxiliary_state_sha256': identity['spatial_context_state_sha256'],
				'initial_auxiliary_state_sha256': identity[
					'initial_spatial_context_state_sha256'
				],
			}
		)
	return result


def _resume_compatibility_result(
	payload: Mapping[str, object],
	*,
	expected_config: Mapping[str, object] | None,
	expected_optimizer: torch.optim.Optimizer | None,
) -> dict[str, object]:
	"""Report resume compatibility without loading checkpoint state."""
	if expected_config is None:
		return {
			'checked': False,
			'compatible': None,
			'reason': 'expected_config was not provided',
		}
	try:
		_validate_resume_compatibility(
			payload,
			expected_config,
			expected_optimizer,
		)
	except (TypeError, ValueError) as exc:
		return {'checked': True, 'compatible': False, 'reason': str(exc)}
	return {'checked': True, 'compatible': True, 'reason': None}


def _validate_resume_compatibility(
	payload: Mapping[str, object],
	expected_config: Mapping[str, object],
	expected_optimizer: torch.optim.Optimizer | None,
) -> None:
	"""Apply the same compatibility checks used before resume state loading."""
	from seis_ssl_cluster.training.strat_hmm.resume import (  # noqa: PLC0415
		_first_compatibility_mismatch,
		_strat_resume_compatibility_view,
		_validate_stratigraphy_checkpoint_mode,
	)

	checkpoint_config = payload.get('stratigraphy_config')
	if not isinstance(checkpoint_config, Mapping):
		raise TypeError('resume checkpoint stratigraphy_config must be a mapping')
	checkpoint_view = _strat_resume_compatibility_view(checkpoint_config)
	expected_view = _strat_resume_compatibility_view(expected_config)
	if checkpoint_view != expected_view:
		label = _first_compatibility_mismatch(checkpoint_view, expected_view)
		msg = (
			'resume checkpoint stratigraphy_config is incompatible with '
			f'current resolved config at {label}'
		)
		raise ValueError(msg)
	_validate_stratigraphy_checkpoint_mode(
		payload,
		expected_config,
		optimizer=expected_optimizer,
	)


def _state_dict_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
	return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _is_multi_head_config(config: Mapping[str, object]) -> bool:
	head = config.get('head')
	return isinstance(head, Mapping) and 'spec' in head


def _is_center_trace_config(config: Mapping[str, object]) -> bool:
	return isinstance(config.get('spatial_context'), Mapping)


def _validate_center_trace_optimizer_layout(  # noqa: C901
	*,
	optimizer: torch.optim.Optimizer,
	stratigraphy_config: Mapping[str, object],
	student: torch.nn.Module,
	head: torch.nn.Module,
	spatial_context: torch.nn.Module | None,
) -> None:
	if spatial_context is None:
		raise ValueError('center-trace optimizer validation requires spatial_context')
	groups = optimizer.param_groups
	if [group.get('name') for group in groups] != [
		'head',
		'encoder',
		'spatial_context',
	]:
		raise ValueError(
			'center-trace optimizer requires exactly head, encoder, and '
			'spatial_context parameter groups'
		)
	train = _required_mapping(stratigraphy_config, 'train')
	expected_lrs = (
		_required_positive_finite_number(train.get('lr'), 'train.lr'),
		_required_positive_finite_number(train.get('encoder_lr'), 'train.encoder_lr'),
		_required_positive_finite_number(train.get('lr'), 'train.lr'),
	)
	for group, expected_lr in zip(groups, expected_lrs, strict=True):
		actual_lr = _required_positive_finite_number(
			group.get('lr'), f'optimizer {group.get("name")!r} group lr'
		)
		if actual_lr != expected_lr:
			raise ValueError(
				'center-trace optimizer group learning rate does not match train config'
			)
	modules = (student, head, spatial_context)
	known_parameters = {
		id(parameter): parameter
		for module in modules
		for parameter in module.parameters()
	}
	if len(known_parameters) != sum(
		1 for module in modules for _parameter in module.parameters()
	):
		raise ValueError('center-trace modules must not share parameters')
	trainable_parameters = tuple(
		parameter for module in modules for parameter in module.parameters()
		if parameter.requires_grad
	)
	group_parameters = tuple(
		parameter for group in groups for parameter in group['params']
	)
	group_ids = tuple(id(parameter) for parameter in group_parameters)
	if len(group_ids) != len(set(group_ids)):
		raise ValueError('center-trace optimizer groups contain duplicate parameters')
	if set(group_ids) != {id(parameter) for parameter in trainable_parameters}:
		raise ValueError(
			'center-trace optimizer groups must exactly match trainable parameters'
		)
	for parameter_id in group_ids:
		if parameter_id not in known_parameters:
			raise ValueError(
				'center-trace optimizer contains a parameter outside the '
				'student/head/spatial_context modules'
			)
	if {id(parameter) for parameter in groups[0]['params']} != {
		id(parameter) for parameter in head.parameters() if parameter.requires_grad
	}:
		raise ValueError('center-trace optimizer head group identity mismatch')
	if {id(parameter) for parameter in groups[1]['params']} != {
		id(parameter)
		for parameter in student.parameters()
		if parameter.requires_grad
	}:
		raise ValueError('center-trace optimizer encoder group identity mismatch')
	if {id(parameter) for parameter in groups[2]['params']} != {
		id(parameter)
		for parameter in spatial_context.parameters()
		if parameter.requires_grad
	}:
		raise ValueError(
			'center-trace optimizer spatial_context group identity mismatch'
		)


def _validate_center_trace_spatial_context_state(
	config: Mapping[str, object],
	spatial_context: torch.nn.Module,
	state: Mapping[str, torch.Tensor],
) -> None:
	"""Validate the independent schema-7 replacement-token state."""
	expected_state = spatial_context.state_dict()
	if set(state) != set(expected_state):
		raise ValueError(
			'center-trace spatial_context state keys do not match replacement token'
		)
	for key, expected in expected_state.items():
		actual = state[key]
		if not isinstance(actual, torch.Tensor):
			raise TypeError(f'spatial_context_state_dict.{key} must be a tensor')
		if actual.shape != expected.shape:
			raise ValueError(
				f'spatial_context state shape mismatch for {key}: '
				f'{tuple(actual.shape)!r} != {tuple(expected.shape)!r}'
			)
		if actual.dtype != expected.dtype:
			raise TypeError(
				f'spatial_context state dtype mismatch for {key}: '
				f'{actual.dtype} != {expected.dtype}'
			)
	_validate_finite_state_dict(state, label='spatial_context_state_dict')
	identity = _required_mapping(config, 'identity')
	scientific = _required_mapping(identity, 'scientific_identity')
	if not all(
		isinstance(value, torch.Tensor) and value.is_floating_point()
		for value in state.values()
	):
		raise TypeError('center-trace spatial_context state must be floating tensors')
	if scientific.get('replacement_initialization') != (
		CENTER_TRACE_REPLACEMENT_INITIALIZATION
	):
		raise ValueError('center-trace replacement initialization policy mismatch')


def _validate_checkpoint_inputs(  # noqa: PLR0913
	*,
	model_state_dict: Mapping[str, torch.Tensor],
	stratigraphy_state_dict: Mapping[str, torch.Tensor],
	optimizer: torch.optim.Optimizer,
	stratigraphy_config: Mapping[str, object],
	student: torch.nn.Module,
	head: torch.nn.Module,
	spatial_context: torch.nn.Module | None,
	spatial_context_state_dict: Mapping[str, torch.Tensor] | None,
) -> None:
	_validate_finite_state_dict(model_state_dict, label='model_state_dict')
	_validate_finite_state_dict(
		stratigraphy_state_dict, label='stratigraphy_state_dict'
	)
	if _is_multi_head_config(stratigraphy_config):
		_validate_multi_head_config_and_state(
			stratigraphy_config, stratigraphy_state_dict
		)
		_validate_multi_head_module_and_state(
			stratigraphy_config,
			head,
			stratigraphy_state_dict,
		)
		_validate_multi_head_optimizer_layout(
			optimizer=optimizer,
			stratigraphy_config=stratigraphy_config,
			student=student,
			head=head,
			spatial_context=spatial_context,
		)
		if _is_center_trace_config(stratigraphy_config):
			if spatial_context is None or spatial_context_state_dict is None:
				raise ValueError(
					'schema-7 center-trace checkpoint requires spatial_context state'
				)
			_validate_center_trace_spatial_context_state(
				stratigraphy_config,
				spatial_context,
				spatial_context_state_dict,
			)
		elif spatial_context is not None or spatial_context_state_dict is not None:
			raise ValueError(
				'spatial_context state is only valid for schema-7 center-trace '
				'checkpoints'
			)
	elif spatial_context is not None or spatial_context_state_dict is not None:
		raise ValueError(
			'spatial_context state requires a multi-head center-trace checkpoint'
		)
	_validate_finite_optimizer_state(optimizer.state_dict())
	for group in optimizer.param_groups:
		if not group.get('params'):
			raise ValueError('optimizer parameter group must not be empty')


def _validate_multi_head_optimizer_layout(  # noqa: C901
	*,
	optimizer: torch.optim.Optimizer,
	stratigraphy_config: Mapping[str, object],
	student: torch.nn.Module,
	head: torch.nn.Module,
	spatial_context: torch.nn.Module | None,
) -> None:
	"""Require the fixed multi-head encoder/head optimizer partition."""
	if _is_center_trace_config(stratigraphy_config):
		_validate_center_trace_optimizer_layout(
			optimizer=optimizer,
			stratigraphy_config=stratigraphy_config,
			student=student,
			head=head,
			spatial_context=spatial_context,
		)
		return
	groups = optimizer.param_groups
	if len(groups) != 2 or [group.get('name') for group in groups] != [
		'head',
		'encoder',
	]:
		raise ValueError(
			'multi-head optimizer requires exactly head and encoder parameter groups'
		)
	train = _required_mapping(stratigraphy_config, 'train')
	expected_lrs = (
		_required_positive_finite_number(train.get('lr'), 'train.lr'),
		_required_positive_finite_number(train.get('encoder_lr'), 'train.encoder_lr'),
	)
	for group, expected_lr in zip(groups, expected_lrs, strict=True):
		actual_lr = _required_positive_finite_number(
			group.get('lr'), f'multi-head optimizer {group.get("name")!r} group lr'
		)
		if actual_lr != expected_lr:
			raise ValueError(
				'multi-head optimizer group learning rate does not match train config'
			)

	head_parameters = tuple(head.parameters())
	student_parameters = tuple(student.parameters())
	if not all(parameter.requires_grad for parameter in head_parameters):
		raise ValueError('all multi-head head parameters must be trainable')
	trainable_student_parameters = tuple(
		parameter for parameter in student_parameters if parameter.requires_grad
	)
	if not trainable_student_parameters:
		raise ValueError('multi-head optimizer requires trainable encoder parameters')

	known_parameters = {
		id(parameter): parameter
		for parameter in (*head_parameters, *student_parameters)
	}
	group_parameters = tuple(
		parameter for group in groups for parameter in group['params']
	)
	group_ids = tuple(id(parameter) for parameter in group_parameters)
	if len(group_ids) != len(set(group_ids)):
		raise ValueError('multi-head optimizer groups contain duplicate parameters')
	if any(parameter_id not in known_parameters for parameter_id in group_ids):
		raise ValueError(
			'multi-head optimizer contains a parameter outside the student/head modules'
		)
	if any(
		not known_parameters[parameter_id].requires_grad for parameter_id in group_ids
	):
		raise ValueError('multi-head optimizer contains frozen parameters')

	head_ids = {id(parameter) for parameter in head_parameters}
	encoder_ids = {id(parameter) for parameter in trainable_student_parameters}
	if {id(parameter) for parameter in groups[0]['params']} != head_ids:
		raise ValueError(
			'multi-head optimizer head group must contain every head parameter '
			'exactly once'
		)
	if {id(parameter) for parameter in groups[1]['params']} != encoder_ids:
		raise ValueError(
			'multi-head optimizer encoder group must contain every trainable encoder '
			'parameter exactly once'
		)


def _validate_finite_optimizer_state(state: Mapping[str, object]) -> None:
	for value in _walk_values(state):
		if (
			isinstance(value, torch.Tensor)
			and value.is_floating_point()
			and not torch.isfinite(value).all()
		):
			raise ValueError('optimizer state contains non-finite values')


def _walk_values(value: object) -> Iterator[object]:
	if isinstance(value, Mapping):
		for child in value.values():
			yield from _walk_values(child)
	elif isinstance(value, list | tuple):
		for child in value:
			yield from _walk_values(child)
	else:
		yield value


def _validate_finite_state_dict(
	state: Mapping[str, torch.Tensor], *, label: str
) -> None:
	for key, value in state.items():
		if not isinstance(value, torch.Tensor):
			raise TypeError(f'{label}.{key} must be a tensor')
		if value.is_floating_point() and not torch.isfinite(value).all():
			raise ValueError(f'{label}.{key} contains non-finite values')


def _validate_multi_head_config_and_state(
	config: Mapping[str, object], state: Mapping[str, torch.Tensor]
) -> None:
	head = config.get('head')
	if not isinstance(head, Mapping):
		raise TypeError('multi-head stratigraphy_config.head must be a mapping')
	if head.get('spec') != 'multi_resolution_ordered_prototypes_v1':
		raise ValueError('multi-head checkpoint head spec mismatch')
	ks = _head_ks(head.get('ks'))
	prototype_keys = {f'heads.k{k}.prototypes' for k in ks}
	projection_keys = {
		f'heads.k{k}.projection.{parameter}'
		for k in ks
		for parameter in ('weight', 'bias')
	}
	state_keys = set(state)
	if 'projection_dim' in head:
		expected_keys = (
			prototype_keys | projection_keys
			if head.get('projection_dim') is not None
			else prototype_keys
		)
		valid_state_keys = {frozenset(expected_keys)}
	else:
		valid_state_keys = {
			frozenset(prototype_keys),
			frozenset(prototype_keys | projection_keys),
		}
	if frozenset(state_keys) not in valid_state_keys:
		raise ValueError('multi-head checkpoint head state keys do not match head.ks')
	_validate_multi_head_state_shapes(head, state)


def _validate_multi_head_module_and_state(
	config: Mapping[str, object],
	module: torch.nn.Module,
	state: Mapping[str, torch.Tensor],
) -> None:
	"""Require the saved module to implement the configured multi-head contract."""
	if type(module) is not MultiResolutionOrderedPrototypeHeads:
		raise TypeError(
			'multi-head checkpoint head module type must be '
			'MultiResolutionOrderedPrototypeHeads'
		)
	head = _required_mapping(config, 'head')
	ks = _head_ks(head.get('ks'))
	if module.head_ks != ks:
		raise ValueError(
			'multi-head checkpoint head module K order does not match head.ks'
		)
	expected_head_keys = tuple(f'k{k}' for k in ks)
	if tuple(module.heads) != expected_head_keys:
		raise ValueError('multi-head checkpoint head module keys do not match head.ks')
	for k in ks:
		per_head = module.heads[f'k{k}']
		if type(per_head) is not OrderedPrototypeHead:
			raise TypeError(
				'multi-head checkpoint per-head module type must be '
				'OrderedPrototypeHead'
			)
		if per_head.num_prototypes != k:
			raise ValueError(
				'multi-head checkpoint per-head prototype count does not match head.ks'
			)
		if per_head.feature_dim != module.feature_dim:
			raise ValueError(
				'multi-head checkpoint per-head feature dimension does not match '
				'module feature dimension'
			)
	_validate_multi_head_module_scientific_identity(config, module)
	_validate_multi_head_state_shapes(head, state)


def _validate_multi_head_module_scientific_identity(
	config: Mapping[str, object],
	module: MultiResolutionOrderedPrototypeHeads,
) -> None:
	"""Bind non-state-dict head settings and loss weights to provenance."""
	identity = config.get('identity')
	if not isinstance(identity, Mapping):
		return
	scientific = identity.get('scientific_identity')
	if not isinstance(scientific, Mapping):
		return
	head = _required_mapping(config, 'head')
	for scientific_key, config_key, module_key in (
		('head_temperature', 'temperature', 'temperature'),
		('head_normalize', 'normalize', 'normalize'),
	):
		if scientific_key not in scientific:
			continue
		expected = scientific[scientific_key]
		if head.get(config_key) != expected:
			raise ValueError(
				f'multi-head checkpoint head.{config_key} does not match '
				'scientific identity'
			)
		if any(
			getattr(per_head, module_key) != expected
			for per_head in module.heads.values()
		):
			raise ValueError(
				f'multi-head checkpoint module {module_key} does not match '
				'scientific identity'
			)

	for key in (
		'prototype_weight',
		'usage_weight',
		'consistency_weight',
		'consistency_beta',
		'distillation_weight',
	):
		if key not in scientific:
			continue
		loss = _required_mapping(config, 'loss')
		if loss.get(key) != scientific[key]:
			raise ValueError(
				f'multi-head checkpoint loss.{key} does not match scientific identity'
			)


def _validate_multi_head_state_shapes(
	head: Mapping[str, object], state: Mapping[str, torch.Tensor]
) -> None:
	"""Validate that every saved head tensor has its declared prototype shape."""
	ks = _head_ks(head.get('ks'))
	projection_dim: int | None = None
	feature_dim: int | None = None
	for k in ks:
		prototypes = state[f'heads.k{k}.prototypes']
		if prototypes.ndim != 2 or prototypes.shape[0] != k:
			raise ValueError(
				'multi-head checkpoint prototype tensor shape does not match head.ks'
			)
		current_projection_dim = prototypes.shape[1]
		if projection_dim is None:
			projection_dim = current_projection_dim
		elif current_projection_dim != projection_dim:
			raise ValueError(
				'multi-head checkpoint prototype tensor shapes must share projection '
				'dimension'
			)
		configured_projection_dim = head.get('projection_dim')
		if (
			configured_projection_dim is not None
			and current_projection_dim != configured_projection_dim
		):
			raise ValueError(
				'multi-head checkpoint prototype tensor shape does not match '
				'head.projection_dim'
			)
		projection_weight_key = f'heads.k{k}.projection.weight'
		if projection_weight_key not in state:
			continue
		weight = state[projection_weight_key]
		bias = state[f'heads.k{k}.projection.bias']
		if (
			weight.ndim != 2
			or bias.ndim != 1
			or weight.shape[0] != current_projection_dim
			or bias.shape[0] != current_projection_dim
		):
			raise ValueError(
				'multi-head checkpoint projection tensor shapes do not match '
				'prototype shape'
			)
		current_feature_dim = weight.shape[1]
		if feature_dim is None:
			feature_dim = current_feature_dim
		elif current_feature_dim != feature_dim:
			raise ValueError(
				'multi-head checkpoint projection tensor shapes must share feature '
				'dimension'
			)


def _multi_head_checkpoint_identity(  # noqa: PLR0913
	*,
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, torch.Tensor],
	control_identity: Mapping[str, object] | None,
	optimizer: torch.optim.Optimizer,
	student: torch.nn.Module,
	head: torch.nn.Module,
	spatial_context: torch.nn.Module | None = None,
	spatial_context_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, object]:
	head_config = _required_mapping(stratigraphy_config, 'head')
	pseudo_targets = _required_mapping(stratigraphy_config, 'pseudo_targets')
	identity = _required_mapping(stratigraphy_config, 'identity')
	scientific = _required_mapping(identity, 'scientific_identity')
	if _is_center_trace_config(stratigraphy_config):
		return _center_trace_checkpoint_identity(
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
			spatial_context=spatial_context,
			spatial_context_state_dict=spatial_context_state_dict,
			control_identity=control_identity,
			optimizer=optimizer,
			student=student,
			head=head,
		)
	if _is_soft_multi_head_config(stratigraphy_config):
		return _soft_multi_head_checkpoint_identity(
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
			control_identity=control_identity,
			optimizer=optimizer,
			student=student,
			head=head,
		)
	if _is_lateral_multi_head_config(stratigraphy_config):
		return _lateral_multi_head_checkpoint_identity(
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
			control_identity=control_identity,
			optimizer=optimizer,
			student=student,
			head=head,
		)
	if _is_xy_neighbor_unanimous_multi_head_config(stratigraphy_config):
		return _xy_neighbor_unanimous_multi_head_checkpoint_identity(
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
			control_identity=control_identity,
			optimizer=optimizer,
			student=student,
			head=head,
		)
	if _is_xy_neighbor_consensus_multi_head_config(stratigraphy_config):
		return _xy_neighbor_consensus_multi_head_checkpoint_identity(
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
			control_identity=control_identity,
			optimizer=optimizer,
			student=student,
			head=head,
		)
	manifest_path = _required_string(
		pseudo_targets.get('manifest'), 'pseudo_targets.manifest'
	)
	manifest_sha256 = _file_sha256(Path(manifest_path))
	if scientific.get('target_manifest_sha256') != manifest_sha256:
		raise ValueError('target manifest SHA-256 does not match scientific identity')
	per_head_targets = _manifest_per_head_target_hashes(Path(manifest_path))
	if scientific.get('target_head_hashes') != per_head_targets:
		raise ValueError(
			'scientific identity per-head target hashes do not match target manifest'
		)
	result: dict[str, object] = {
		'schema_version': 2,
		'head_spec': head_config['spec'],
		'head_ks': list(_head_ks(head_config.get('ks'))),
		'target_manifest': {'path': manifest_path, 'sha256': manifest_sha256},
		'per_head_targets': per_head_targets,
		'consistency_policy': scientific.get('consistency_policy'),
		'consistency_weight': scientific.get('consistency_weight'),
		'consistency_beta': scientific.get('consistency_beta'),
		'model_tag': identity.get('model_tag'),
		'output_root': _required_mapping(stratigraphy_config, 'paths').get(
			'output_root'
		),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'stratigraphy_state_sha256': _state_sha256(stratigraphy_state_dict),
		'optimizer_group_identity': _optimizer_group_identity(
			optimizer,
			parameter_names=_stratigraphy_parameter_names(student, head),
		),
	}
	if not isinstance(control_identity, Mapping):
		raise TypeError('multi-head checkpoint requires control identity')
	inputs = _required_mapping(control_identity, 'input_identities')
	result['teacher_checkpoint_sha256'] = _required_sha256(
		_required_mapping(inputs, 'teacher_checkpoint').get('sha256'),
		'input_identities.teacher_checkpoint.sha256',
	)
	result['student_init_checkpoint_sha256'] = _required_sha256(
		_required_mapping(inputs, 'student_init_checkpoint').get('sha256'),
		'input_identities.student_init_checkpoint.sha256',
	)
	initial_states = control_identity.get('initial_state_sha256')
	if not isinstance(initial_states, Mapping):
		raise TypeError('multi-head checkpoint requires initial state hashes')
	result['initial_student_state_sha256'] = _required_sha256(
		initial_states.get('student'), 'initial_state_sha256.student'
	)
	result['initial_head_state_sha256'] = _required_sha256(
		initial_states.get('head'), 'initial_state_sha256.head'
	)
	return result


def _center_trace_checkpoint_identity(  # noqa: PLR0913
	*,
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, torch.Tensor],
	spatial_context: torch.nn.Module | None,
	spatial_context_state_dict: Mapping[str, torch.Tensor] | None,
	control_identity: Mapping[str, object] | None,
	optimizer: torch.optim.Optimizer,
	student: torch.nn.Module,
	head: torch.nn.Module,
) -> dict[str, object]:
	"""Build the closed schema-7 center-trace checkpoint identity."""
	if spatial_context is None or spatial_context_state_dict is None:
		raise ValueError('schema-7 checkpoint requires spatial_context state')
	if not isinstance(control_identity, Mapping):
		raise TypeError('schema-7 checkpoint requires control identity')
	head_config = _required_mapping(stratigraphy_config, 'head')
	if head_config.get('spec') != _CENTER_TRACE_HEAD_SPEC or tuple(
		head_config.get('ks', ())
	) != _CENTER_TRACE_HEAD_KS:
		raise ValueError('schema-7 checkpoint requires head K=(6, 8, 10)')
	spatial = _required_mapping(stratigraphy_config, 'spatial_context')
	for key, expected in CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT.items():
		if spatial.get(key) != expected:
			raise ValueError(f'schema-7 spatial_context.{key} contract mismatch')
	pseudo_targets = _required_mapping(stratigraphy_config, 'pseudo_targets')
	manifest_path = _required_string(
		pseudo_targets.get('manifest'), 'pseudo_targets.manifest'
	)
	manifest_sha256 = _file_sha256(Path(manifest_path))
	per_head_targets = _manifest_per_head_target_hashes(Path(manifest_path))
	identity = _required_mapping(stratigraphy_config, 'identity')
	scientific = _required_mapping(identity, 'scientific_identity')
	if scientific.get('target_manifest_sha256') != manifest_sha256:
		raise ValueError('schema-7 target manifest SHA-256 mismatch')
	if scientific.get('target_head_hashes') != per_head_targets:
		raise ValueError('schema-7 per-head target hashes do not match manifest')
	inputs = _required_mapping(control_identity, 'input_identities')
	initial_states = _required_mapping(control_identity, 'initial_state_sha256')
	spatial_state_sha256 = _state_sha256(spatial_context_state_dict)
	return {
		'schema_version': 7,
		'head_spec': _CENTER_TRACE_HEAD_SPEC,
		'head_ks': list(_CENTER_TRACE_HEAD_KS),
		'target_representation': _CENTER_TRACE_TARGET_REPRESENTATION,
		'target_manifest_sha256': manifest_sha256,
		'target_manifest': {
			'path': manifest_path,
			'sha256': manifest_sha256,
		},
		'per_head_targets': per_head_targets,
		'objective_semantics': scientific['objective_semantics'],
		'mask_semantics': scientific['mask_semantics'],
		'column_fraction': scientific['column_fraction'],
		'selection_policy': scientific['selection_policy'],
		'replacement': scientific['replacement'],
		'replacement_initialization': scientific['replacement_initialization'],
		'rng_policy': scientific['rng_policy'],
		'masked_prototype_weight': scientific['masked_prototype_weight'],
		'visible_prototype_weight': scientific['visible_prototype_weight'],
		'distillation_scope': scientific['distillation_scope'],
		'supervised_loss': scientific['supervised_loss'],
		'consistency_policy': scientific['consistency_policy'],
		'prototype_weight': scientific['prototype_weight'],
		'usage_weight': scientific['usage_weight'],
		'consistency_weight': scientific['consistency_weight'],
		'consistency_beta': scientific['consistency_beta'],
		'distillation_weight': scientific['distillation_weight'],
		'model_tag': identity.get('model_tag'),
		'output_root': _required_mapping(stratigraphy_config, 'paths').get(
			'output_root'
		),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'student_state_sha256': _state_sha256(_state_dict_cpu(student)),
		'stratigraphy_state_sha256': _state_sha256(stratigraphy_state_dict),
		'spatial_context_state_sha256': spatial_state_sha256,
		'initial_spatial_context_state_sha256': _required_sha256(
			initial_states.get('spatial_context'),
			'initial_state_sha256.spatial_context',
		),
		'optimizer_group_identity': _optimizer_group_identity(
			optimizer,
			parameter_names=_stratigraphy_parameter_names(
				student, head, spatial_context=spatial_context
			),
		),
		'teacher_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'teacher_checkpoint').get('sha256'),
			'input_identities.teacher_checkpoint.sha256',
		),
		'student_init_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'student_init_checkpoint').get('sha256'),
			'input_identities.student_init_checkpoint.sha256',
		),
		'initial_student_state_sha256': _required_sha256(
			initial_states.get('student'), 'initial_state_sha256.student'
		),
		'initial_head_state_sha256': _required_sha256(
			initial_states.get('head'), 'initial_state_sha256.head'
		),
	}


def _periodic_refresh_checkpoint_identity(  # noqa: PLR0913
	*,
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, torch.Tensor],
	spatial_context: torch.nn.Module | None,
	spatial_context_state_dict: Mapping[str, torch.Tensor] | None,
	control_identity: Mapping[str, object] | None,
	optimizer: torch.optim.Optimizer,
	student: torch.nn.Module,
	head: torch.nn.Module,
	target_refresh_state: Mapping[str, object],
) -> dict[str, object]:
	"""Build the independent schema-8 periodic-refresh checkpoint identity."""
	if spatial_context is None or spatial_context_state_dict is None:
		raise ValueError('schema-8 checkpoint requires spatial_context state')
	if not isinstance(control_identity, Mapping):
		raise TypeError('schema-8 checkpoint requires control identity')
	head_config = _required_mapping(stratigraphy_config, 'head')
	if head_config.get('spec') != 'multi_resolution_ordered_prototypes_v1' or tuple(
		head_config.get('ks', ())
	) != (6, 8, 10):
		raise ValueError('schema-8 checkpoint requires head K=(6, 8, 10)')
	spatial = _required_mapping(stratigraphy_config, 'spatial_context')
	for key, expected in CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT.items():
		if spatial.get(key) != expected:
			raise ValueError(f'schema-8 spatial_context.{key} contract mismatch')
	pseudo_targets = _required_mapping(stratigraphy_config, 'pseudo_targets')
	initial_manifest_path = Path(
		_required_string(pseudo_targets.get('manifest'), 'pseudo_targets.manifest')
	)
	initial_manifest_sha256 = _file_sha256(initial_manifest_path)
	per_head_targets = _manifest_per_head_target_hashes(initial_manifest_path)
	identity = _required_mapping(stratigraphy_config, 'identity')
	scientific = _required_mapping(identity, 'scientific_identity')
	if scientific.get('target_manifest_sha256') != initial_manifest_sha256:
		raise ValueError('schema-8 initial target manifest SHA-256 mismatch')
	if scientific.get('target_head_hashes') != per_head_targets:
		raise ValueError(
			'schema-8 initial per-head target hashes do not match manifest'
		)
	inputs = _required_mapping(control_identity, 'input_identities')
	initial_states = _required_mapping(control_identity, 'initial_state_sha256')
	return {
		'schema_version': 8,
		'head_spec': 'multi_resolution_ordered_prototypes_v1',
		'head_ks': [6, 8, 10],
		'target_representation': PERIODIC_REFRESH_TARGET_REPRESENTATION,
		'initial_hard_target_manifest_sha256': initial_manifest_sha256,
		'initial_hard_target_manifest': {
			'path': str(initial_manifest_path),
			'sha256': initial_manifest_sha256,
		},
		'initial_per_head_targets': per_head_targets,
		'objective_semantics': scientific['objective_semantics'],
		'mask_semantics': scientific['mask_semantics'],
		'column_fraction': scientific['column_fraction'],
		'selection_policy': scientific['selection_policy'],
		'replacement': scientific['replacement'],
		'replacement_initialization': scientific['replacement_initialization'],
		'rng_policy': scientific['rng_policy'],
		'masked_prototype_weight': scientific['masked_prototype_weight'],
		'visible_prototype_weight': scientific['visible_prototype_weight'],
		'distillation_scope': scientific['distillation_scope'],
		'supervised_loss': scientific['supervised_loss'],
		'consistency_policy': scientific['consistency_policy'],
		'prototype_weight': scientific['prototype_weight'],
		'usage_weight': scientific['usage_weight'],
		'consistency_weight': scientific['consistency_weight'],
		'consistency_beta': scientific['consistency_beta'],
		'distillation_weight': scientific['distillation_weight'],
		'model_role': scientific['model_role'],
		'target_refresh_semantics': scientific['target_refresh_semantics'],
		'refresh_schedule_semantics': scientific['refresh_schedule_semantics'],
		'refresh_after_epochs': scientific['refresh_after_epochs'],
		'hmm_iterations_per_refresh': scientific['hmm_iterations_per_refresh'],
		'embedding_source': scientific['embedding_source'],
		'embedding_mode': scientific['embedding_mode'],
		'refresh_embedding_semantics': scientific['refresh_embedding_semantics'],
		'center_initialization': scientific['center_initialization'],
		'center_update': scientific['center_update'],
		'center_update_semantics': scientific['center_update_semantics'],
		'preprocessing_policy': scientific['preprocessing_policy'],
		'target_activation_policy': scientific['target_activation_policy'],
		'empty_state_policy': scientific['empty_state_policy'],
		'checkpoint_selection_policy': scientific['checkpoint_selection_policy'],
		'initial_hmm_artifacts': scientific['initial_hmm_artifacts'],
		'fixed_preprocessor_sha256': scientific['fixed_preprocessor_sha256'],
		'fixed_residualizer_sha256': scientific['fixed_residualizer_sha256'],
		'fixed_clustering_config_sha256': scientific[
			'fixed_clustering_config_sha256'
		],
		'source_embedding_metadata_sha256': scientific[
			'source_embedding_metadata_sha256'
		],
		'source_valid_token_hashes': scientific['source_valid_token_hashes'],
		'feature_dimension': scientific['feature_dimension'],
		'generation_root': scientific['generation_root'],
		'target_refresh_state_sha256': _canonical_sha256(target_refresh_state),
		'model_tag': identity.get('model_tag'),
		'output_root': _required_mapping(stratigraphy_config, 'paths').get(
			'output_root'
		),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'student_state_sha256': _state_sha256(_state_dict_cpu(student)),
		'stratigraphy_state_sha256': _state_sha256(stratigraphy_state_dict),
		'spatial_context_state_sha256': _state_sha256(spatial_context_state_dict),
		'initial_spatial_context_state_sha256': _required_sha256(
			initial_states.get('spatial_context'),
			'initial_state_sha256.spatial_context',
		),
		'optimizer_group_identity': _optimizer_group_identity(
			optimizer,
			parameter_names=_stratigraphy_parameter_names(
				student, head, spatial_context=spatial_context
			),
		),
		'teacher_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'teacher_checkpoint').get('sha256'),
			'input_identities.teacher_checkpoint.sha256',
		),
		'student_init_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'student_init_checkpoint').get('sha256'),
			'input_identities.student_init_checkpoint.sha256',
		),
		'initial_student_state_sha256': _required_sha256(
			initial_states.get('student'), 'initial_state_sha256.student'
		),
		'initial_head_state_sha256': _required_sha256(
			initial_states.get('head'), 'initial_state_sha256.head'
		),
	}


def _periodic_train_epochs(config: Mapping[str, object]) -> int:
	train = _required_mapping(config, 'train')
	epochs = train.get('epochs')
	if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs != 25:
		raise ValueError('periodic refresh requires train.epochs == 25')
	return epochs


def _periodic_fixed_preprocessing_identity_sha256(
	scientific: Mapping[str, object],
) -> str:
	keys = (
		'initial_hmm_artifacts',
		'fixed_preprocessor_sha256',
		'fixed_residualizer_sha256',
		'fixed_clustering_config_sha256',
		'source_embedding_metadata_sha256',
		'source_valid_token_hashes',
		'feature_dimension',
	)
	return _canonical_sha256({key: scientific.get(key) for key in keys})


def _validated_target_refresh_state(  # noqa: C901, PLR0912, PLR0915
	value: object,
	*,
	expected_config: Mapping[str, object] | None = None,
	recovery_source_student_state_sha256: str | None = None,
) -> dict[str, object]:
	"""Validate the complete, externally published periodic refresh state."""
	if not isinstance(value, Mapping):
		raise TypeError('target_refresh_state must be a mapping')
	unknown = sorted(set(value) - _PERIODIC_REFRESH_STATE_KEYS)
	missing = sorted(_PERIODIC_REFRESH_STATE_KEYS - set(value))
	if unknown:
		raise ValueError(
			f'target_refresh_state has unsupported field(s): {unknown!r}'
		)
	if missing:
		raise ValueError(f'target_refresh_state is missing field(s): {missing!r}')
	if value.get('schema_version') != _PERIODIC_REFRESH_STATE_SCHEMA_VERSION:
		raise ValueError('unsupported target_refresh_state schema_version')

	active_index = _nonnegative_int_value(
		value.get('active_generation_index'), 'active_generation_index'
	)
	active_id = _required_string(
		value.get('active_generation_id'), 'active_generation_id'
	)
	active_manifest_path = _absolute_state_path(
		value.get('active_generation_manifest_path'),
		'active_generation_manifest_path',
	)
	active_manifest_sha256 = _required_sha256(
		value.get('active_generation_manifest_sha256'),
		'active_generation_manifest_sha256',
	)
	active_content_sha256 = _required_sha256(
		value.get('active_generation_content_sha256'),
		'active_generation_content_sha256',
	)
	active_target_path = _absolute_state_path(
		value.get('active_target_manifest_path'), 'active_target_manifest_path'
	)
	active_target_sha256 = _required_sha256(
		value.get('active_target_manifest_sha256'),
		'active_target_manifest_sha256',
	)
	chain_path = _absolute_state_path(
		value.get('periodic_refresh_chain_path'), 'periodic_refresh_chain_path'
	)
	chain_sha256 = _required_sha256(
		value.get('periodic_refresh_chain_sha256'),
		'periodic_refresh_chain_sha256',
	)
	last_refresh = _nonnegative_int_value(
		value.get('last_completed_refresh_epoch'),
		'last_completed_refresh_epoch',
	)
	next_refresh = value.get('next_scheduled_refresh_epoch')
	if next_refresh is not None:
		next_refresh = _positive_int_value(
			next_refresh, 'next_scheduled_refresh_epoch'
		)
	phase = value.get('refresh_phase')
	if phase not in {'training', 'refresh_required', 'refresh_complete'}:
		raise ValueError(
			'target_refresh_state.refresh_phase must be training, refresh_required, '
			'or refresh_complete'
		)
	source_hash = value.get('source_student_state_sha256')
	if source_hash is not None:
		source_hash = _required_sha256(
			source_hash, 'source_student_state_sha256'
		)
	fixed_identity_hash = _required_sha256(
		value.get('fixed_preprocessing_hmm_identity_sha256'),
		'fixed_preprocessing_hmm_identity_sha256',
	)

	raw_generations = value.get('generations')
	if not isinstance(raw_generations, list) or not raw_generations:
		raise ValueError('target_refresh_state.generations must be a non-empty list')
	if len(raw_generations) != active_index + 1:
		raise ValueError(
		'target_refresh_state.generations must end at active_generation_index'
	)
	generations: list[dict[str, object]] = []
	loaded_payloads: list[Mapping[str, object]] = []
	for expected_index, raw_generation in enumerate(raw_generations):
		if not isinstance(raw_generation, Mapping):
			raise TypeError('target_refresh_state generation must be a mapping')
		if set(raw_generation) != _PERIODIC_REFRESH_GENERATION_KEYS:
			raise ValueError(
				'target_refresh_state generation fields are not closed'
			)
		generation_index = _nonnegative_int_value(
				raw_generation.get('generation_index'),
				'generation.generation_index',
			)
		if generation_index != expected_index:
			raise ValueError(
				'target_refresh_state generation indices must be contiguous from zero'
			)
		generation_id = _required_string(
				raw_generation.get('generation_id'), 'generation.generation_id'
		)
		manifest_path = _absolute_state_path(
				raw_generation.get('manifest_path'), 'generation.manifest_path'
			)
		manifest_sha256 = _required_sha256(
				raw_generation.get('manifest_sha256'), 'generation.manifest_sha256'
			)
		content_sha256 = _required_sha256(
				raw_generation.get('generation_content_sha256'),
			'generation.generation_content_sha256',
		)
		if not manifest_path.is_file():
			raise FileNotFoundError(
				f'target_refresh_state generation manifest is missing: {manifest_path}'
			)
		if _file_sha256(manifest_path) != manifest_sha256:
			raise ValueError(
				'target_refresh_state generation manifest hash mismatch: '
				f'{manifest_path}'
			)
		try:
			generation_payload = load_periodic_refresh_generation(manifest_path)
		except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
			raise ValueError(
				f'target_refresh_state generation is incomplete or corrupt: '
				f'{manifest_path}: {exc}'
			) from exc
		if generation_payload.get('generation_index') != generation_index:
			raise ValueError(
				'target_refresh_state generation index does not match manifest'
			)
		if generation_payload.get('generation_id') != generation_id:
			raise ValueError(
				'target_refresh_state generation id does not match manifest'
			)
		if generation_payload.get('generation_content_sha256') != content_sha256:
			raise ValueError(
				'target_refresh_state generation content hash does not match manifest'
			)
		generations.append(
			{
				'generation_index': generation_index,
				'generation_id': generation_id,
				'manifest_path': str(manifest_path),
				'manifest_sha256': manifest_sha256,
				'generation_content_sha256': content_sha256,
			}
		)
		loaded_payloads.append(generation_payload)

	last_generation = generations[-1]
	if active_index != last_generation['generation_index']:
		raise ValueError('active generation index does not match generation list')
	if active_id != last_generation['generation_id']:
		raise ValueError('active generation id does not match generation list')
	if active_manifest_path != Path(str(last_generation['manifest_path'])):
		raise ValueError(
			'active generation manifest path does not match generation list'
		)
	if active_manifest_sha256 != last_generation['manifest_sha256']:
		raise ValueError(
			'active generation manifest hash does not match generation list'
		)
	if active_content_sha256 != last_generation['generation_content_sha256']:
		raise ValueError(
		'active generation content hash does not match generation list'
	)

	active_payload = loaded_payloads[-1]
	canonical_target = active_payload.get('canonical_multi_head_target_manifest')
	if not isinstance(canonical_target, Mapping):
		raise TypeError(
			'active generation canonical_multi_head_target_manifest must be a mapping'
		)
	canonical_target_path = _absolute_state_path(
		canonical_target.get('path'),
		'active generation canonical target path',
	)
	canonical_target_sha256 = _required_sha256(
		canonical_target.get('sha256'),
		'active generation canonical target hash',
	)
	if active_target_path != canonical_target_path:
		raise ValueError('active target manifest path does not match active generation')
	if active_target_sha256 != canonical_target_sha256:
		raise ValueError('active target manifest hash does not match active generation')
	if not active_target_path.is_file():
		raise FileNotFoundError(
		f'active target manifest is missing: {active_target_path}'
	)
	if _file_sha256(active_target_path) != active_target_sha256:
		raise ValueError('active target manifest SHA-256 mismatch')
	try:
		target_payload = load_multi_head_target_manifest(active_target_path)
	except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
		raise ValueError(
			f'active target manifest is incomplete or corrupt: {active_target_path}'
		) from exc
	if tuple(target_payload.get('head_ks', ())) != (6, 8, 10):
		raise ValueError('active target manifest head_ks must be [6, 8, 10]')

	_refresh_root = _periodic_refresh_root_from_config(expected_config)
	chain_generations = generations
	chain_payloads = loaded_payloads
	validated_chain_sha256 = chain_sha256
	if _refresh_root is not None:
		_expected_periodic_state_paths(
			refresh_root=_refresh_root,
			active_manifest_path=active_manifest_path,
			active_target_path=active_target_path,
			chain_path=chain_path,
			generations=generations,
		)
		pointer_path = _refresh_root / 'active_target_generation.json'
		if not pointer_path.is_file():
			raise FileNotFoundError(
				f'active target generation pointer is missing: {pointer_path}'
			)
		pointer = _load_json_mapping(pointer_path, 'active target generation pointer')
		if set(pointer) != {'manifest_path', 'manifest_sha256'}:
			raise ValueError('active target generation pointer fields are not closed')
		actual_chain_sha256 = _file_sha256(chain_path)
		if actual_chain_sha256 != chain_sha256:
			(
				extra_generation,
				extra_payload,
				pointer_acceptance,
			) = _recover_periodic_refresh_extension(
				chain_path=chain_path,
				pointer_path=pointer_path,
				refresh_root=_refresh_root,
				generations=generations,
				loaded_payloads=loaded_payloads,
				expected_config=expected_config,
				expected_source_student_state_sha256=(
					recovery_source_student_state_sha256
				),
			)
			chain_generations = [*generations, extra_generation]
			chain_payloads = [*loaded_payloads, extra_payload]
			validated_chain_sha256 = actual_chain_sha256
		else:
			pointer_acceptance = {
				'manifest_path': str(active_manifest_path),
				'manifest_sha256': active_manifest_sha256,
			}
		if dict(pointer) != pointer_acceptance:
			raise ValueError(
				'active target generation pointer does not match checkpoint state'
			)

	_validate_periodic_refresh_chain(
		chain_path,
		chain_sha256=validated_chain_sha256,
		generations=chain_generations,
		loaded_payloads=chain_payloads,
		expected_config=expected_config,
	)

	if expected_config is not None:
		_validate_periodic_refresh_state_against_config(
			state={
				'active_generation_index': active_index,
				'active_generation_id': active_id,
				'active_target_manifest_path': str(active_target_path),
				'active_target_manifest_sha256': active_target_sha256,
				'last_completed_refresh_epoch': last_refresh,
				'next_scheduled_refresh_epoch': next_refresh,
				'refresh_phase': phase,
				'source_student_state_sha256': source_hash,
				'fixed_preprocessing_hmm_identity_sha256': fixed_identity_hash,
			},
			loaded_payloads=loaded_payloads,
			expected_config=expected_config,
		)

	return {
		'schema_version': _PERIODIC_REFRESH_STATE_SCHEMA_VERSION,
		'active_generation_index': active_index,
		'active_generation_id': active_id,
		'active_generation_manifest_path': str(active_manifest_path),
		'active_generation_manifest_sha256': active_manifest_sha256,
		'active_generation_content_sha256': active_content_sha256,
		'active_target_manifest_path': str(active_target_path),
		'active_target_manifest_sha256': active_target_sha256,
		'periodic_refresh_chain_path': str(chain_path),
		'periodic_refresh_chain_sha256': chain_sha256,
		'last_completed_refresh_epoch': last_refresh,
		'next_scheduled_refresh_epoch': next_refresh,
		'refresh_phase': phase,
		'source_student_state_sha256': source_hash,
		'fixed_preprocessing_hmm_identity_sha256': fixed_identity_hash,
		'generations': generations,
	}


def _recover_periodic_refresh_extension(  # noqa: C901, PLR0912, PLR0913, PLR0915
	*,
	chain_path: Path,
	pointer_path: Path,
	refresh_root: Path,
	generations: list[dict[str, object]],
	loaded_payloads: list[Mapping[str, object]],
	expected_config: Mapping[str, object],
	expected_source_student_state_sha256: str | None,
) -> tuple[dict[str, object], Mapping[str, object], dict[str, str]]:
	"""Accept only the one generation published before its refresh checkpoint."""
	chain = _load_json_mapping(chain_path, 'periodic refresh chain')
	if set(chain) != _PERIODIC_REFRESH_CHAIN_KEYS:
		raise ValueError('periodic refresh chain fields are not closed')
	if chain.get('schema_version') != 1 or chain.get(
		'semantics'
	) != _PERIODIC_REFRESH_CHAIN_SEMANTICS:
		raise ValueError('periodic refresh chain identity is invalid')
	if tuple(chain.get('refresh_after_epochs', ())) != PERIODIC_REFRESH_SCHEDULE:
		raise ValueError('periodic refresh chain schedule mismatch')
	scientific = _required_mapping(
		_required_mapping(expected_config, 'identity'), 'scientific_identity'
	)
	if chain.get('fixed_preprocessing_hmm_identity_sha256') != (
		_periodic_fixed_preprocessing_identity_sha256(scientific)
	):
		raise ValueError('periodic refresh chain fixed preprocessing identity drift')
	chain_generations = chain.get('generations')
	if not isinstance(chain_generations, list) or len(chain_generations) != len(
		generations
	) + 1:
		raise ValueError('periodic refresh chain has no single recoverable extension')
	for sequence, (generation, payload) in enumerate(
		zip(generations, loaded_payloads, strict=True)
	):
		raw = chain_generations[sequence]
		if not isinstance(raw, Mapping):
			raise TypeError('periodic refresh chain generation must be a mapping')
		if set(raw) != _PERIODIC_REFRESH_CHAIN_GENERATION_KEYS:
			raise ValueError('periodic refresh chain generation fields are not closed')
		previous = payload.get('previous_generation_manifest')
		previous_hash = (
			None
			if previous is None
			else _required_sha256(
				_required_mapping(payload, 'previous_generation_manifest').get(
					'sha256'
				),
				'previous generation manifest sha256',
			)
		)
		expected = {
			'generation_index': generation['generation_index'],
			'generation_id': generation['generation_id'],
			'refresh_after_epoch': payload['refresh_after_epoch'],
			'previous_generation_manifest_sha256': previous_hash,
			'source_student_state_sha256': payload.get(
				'source_student_state_sha256'
			),
			'manifest_path': generation['manifest_path'],
			'manifest_sha256': generation['manifest_sha256'],
			'generation_content_sha256': generation['generation_content_sha256'],
		}
		if dict(raw) != expected:
			raise ValueError('periodic refresh chain prefix does not match state')
	extra_raw = chain_generations[-1]
	if not isinstance(extra_raw, Mapping) or set(extra_raw) != (
		_PERIODIC_REFRESH_CHAIN_GENERATION_KEYS
	):
		raise ValueError('periodic refresh chain extension fields are invalid')
	extra_index = len(generations)
	if extra_index > len(PERIODIC_REFRESH_SCHEDULE):
		raise ValueError('periodic refresh chain extension is beyond the schedule')
	extra_path = _absolute_state_path(
		extra_raw.get('manifest_path'), 'periodic refresh extension manifest path'
	)
	expected_id = (
		f'refresh_{extra_index:04d}_epoch'
		f'{PERIODIC_REFRESH_SCHEDULE[extra_index - 1]:03d}'
	)
	if extra_raw.get('generation_index') != extra_index or extra_raw.get(
		'generation_id'
	) != expected_id or extra_raw.get('refresh_after_epoch') != (
		PERIODIC_REFRESH_SCHEDULE[extra_index - 1]
	):
		raise ValueError('periodic refresh chain extension schedule mismatch')
	try:
		extra_path.resolve().relative_to((refresh_root / 'generations').resolve())
	except ValueError as exc:
		raise ValueError(
			'periodic refresh chain extension is outside generation_root'
		) from exc
	if not extra_path.is_file():
		raise FileNotFoundError(extra_path)
	extra_hash = _required_sha256(
		extra_raw.get('manifest_sha256'), 'periodic extension manifest hash'
	)
	if _file_sha256(extra_path) != extra_hash:
		raise ValueError('periodic refresh extension manifest hash mismatch')
	try:
		extra_payload = load_periodic_refresh_generation(extra_path)
	except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
		raise ValueError('periodic refresh extension is incomplete or corrupt') from exc
	if extra_payload.get('generation_index') != extra_index or extra_payload.get(
		'generation_id'
	) != expected_id:
		raise ValueError('periodic refresh extension manifest identity mismatch')
	if extra_payload.get('refresh_after_epoch') != extra_raw['refresh_after_epoch']:
		raise ValueError('periodic refresh extension epoch mismatch')
	if extra_payload.get('generation_content_sha256') != extra_raw[
		'generation_content_sha256'
	]:
		raise ValueError('periodic refresh extension content hash mismatch')
	previous = _required_mapping(
		extra_payload,
		'previous_generation_manifest',
	)
	if previous.get('sha256') != generations[-1]['manifest_sha256'] or Path(
		str(previous.get('path'))
	).resolve() != Path(str(generations[-1]['manifest_path'])).resolve():
		raise ValueError('periodic refresh extension previous manifest mismatch')
	extra_source = _required_sha256(
		extra_payload.get('source_student_state_sha256'),
		'periodic refresh extension source student state hash',
	)
	if (
		expected_source_student_state_sha256 is not None
		and extra_source != expected_source_student_state_sha256
	):
		raise ValueError('periodic refresh extension source student state mismatch')
	new_record = dict(extra_raw)
	new_record['manifest_path'] = str(extra_path)
	new_record['manifest_sha256'] = extra_hash
	pointer = _load_json_mapping(pointer_path, 'active target generation pointer')
	if set(pointer) != {'manifest_path', 'manifest_sha256'}:
		raise ValueError('active target generation pointer fields are not closed')
	accepted = (
		{
			'manifest_path': str(Path(str(generations[-1]['manifest_path'])).resolve()),
			'manifest_sha256': generations[-1]['manifest_sha256'],
		},
		{
			'manifest_path': str(extra_path),
			'manifest_sha256': extra_hash,
		},
	)
	pointer_value = {
		'manifest_path': pointer.get('manifest_path'),
		'manifest_sha256': pointer.get('manifest_sha256'),
	}
	if pointer_value not in accepted:
		raise ValueError('active target pointer is foreign or out of chain order')
	return new_record, extra_payload, pointer_value


def _periodic_refresh_root_from_config(
	config: Mapping[str, object] | None,
) -> Path | None:
	if config is None:
		return None
	identity = _required_mapping(config, 'identity')
	scientific = _required_mapping(identity, 'scientific_identity')
	return Path(_required_string(scientific.get('generation_root'), 'generation_root'))


def _expected_periodic_state_paths(
	*,
	refresh_root: Path,
	active_manifest_path: Path,
	active_target_path: Path,
	chain_path: Path,
	generations: list[dict[str, object]],
) -> None:
	refresh_root = refresh_root.resolve()
	for label, path, expected in (
		('active generation manifest', active_manifest_path, None),
		(
			'periodic refresh chain',
			chain_path,
			refresh_root / 'periodic_refresh_chain.json',
		),
	):
		try:
			path.resolve().relative_to(refresh_root)
		except ValueError as exc:
			raise ValueError(f'{label} must be under generation_root') from exc
		if expected is not None and path.resolve() != expected.resolve():
			raise ValueError(f'{label} path does not match generation_root layout')
	if int(generations[-1]['generation_index']) > 0:
		try:
			active_target_path.resolve().relative_to(refresh_root)
		except ValueError as exc:
			raise ValueError(
				'active target manifest must be under generation_root'
			) from exc
	for generation in generations:
		path = Path(str(generation['manifest_path']))
		try:
			path.resolve().relative_to(refresh_root / 'generations')
		except ValueError as exc:
			raise ValueError(
				'generation manifest must be under generation_root/generations'
			) from exc


def _validate_periodic_refresh_chain(  # noqa: C901, PLR0912
	path: Path,
	*,
	chain_sha256: str,
	generations: list[dict[str, object]],
	loaded_payloads: list[Mapping[str, object]],
	expected_config: Mapping[str, object] | None,
) -> None:
	if not path.is_file():
		raise FileNotFoundError(f'periodic refresh chain is missing: {path}')
	if _file_sha256(path) != chain_sha256:
		raise ValueError('periodic refresh chain SHA-256 mismatch')
	chain = _load_json_mapping(path, 'periodic refresh chain')
	if set(chain) != _PERIODIC_REFRESH_CHAIN_KEYS:
		raise ValueError('periodic refresh chain fields are not closed')
	if chain.get('schema_version') != 1:
		raise ValueError('periodic refresh chain schema_version is invalid')
	if chain.get('semantics') != _PERIODIC_REFRESH_CHAIN_SEMANTICS:
		raise ValueError('periodic refresh chain semantics are invalid')
	if tuple(chain.get('refresh_after_epochs', ())) != PERIODIC_REFRESH_SCHEDULE:
		raise ValueError('periodic refresh chain schedule mismatch')
	chain_fixed_hash = _required_sha256(
		chain.get('fixed_preprocessing_hmm_identity_sha256'),
		'periodic refresh chain fixed preprocessing identity hash',
	)
	chain_generations = chain.get('generations')
	if not isinstance(chain_generations, list):
		raise TypeError('periodic refresh chain generations must be a list')
	if len(chain_generations) != len(generations) or len(
		loaded_payloads
	) != len(generations):
		raise ValueError('periodic refresh chain generations do not match state')
	expected_generations: list[dict[str, object]] = []
	for sequence, (generation, payload) in enumerate(
		zip(generations, loaded_payloads, strict=True)
	):
		previous = payload.get('previous_generation_manifest')
		if previous is None:
			previous_hash = None
		elif isinstance(previous, Mapping):
			previous_hash = _required_sha256(
				previous.get('sha256'),
				'periodic refresh previous generation manifest hash',
			)
		else:
			raise TypeError(
				'periodic refresh previous generation manifest must be a '
				'mapping or null'
			)
		expected_previous_hash = (
			None
			if sequence == 0
			else generations[sequence - 1]['manifest_sha256']
		)
		if previous_hash != expected_previous_hash:
			raise ValueError(
				'periodic refresh generation chain is disconnected from the prior '
				'generation manifest'
			)
		source_hash = payload.get('source_student_state_sha256')
		if source_hash is not None:
			source_hash = _required_sha256(
				source_hash,
				'periodic refresh chain source student state hash',
			)
		expected_generations.append(
			{
				'generation_index': generation['generation_index'],
				'generation_id': generation['generation_id'],
				'refresh_after_epoch': payload.get('refresh_after_epoch'),
				'previous_generation_manifest_sha256': previous_hash,
				'source_student_state_sha256': source_hash,
				'manifest_path': generation['manifest_path'],
				'manifest_sha256': generation['manifest_sha256'],
				'generation_content_sha256': generation[
					'generation_content_sha256'
				],
			}
		)
	for sequence, raw_generation in enumerate(chain_generations):
		if not isinstance(raw_generation, Mapping):
			raise TypeError('periodic refresh chain generation must be a mapping')
		if set(raw_generation) != _PERIODIC_REFRESH_CHAIN_GENERATION_KEYS:
			raise ValueError(
				'periodic refresh chain generation fields are not closed'
			)
		if dict(raw_generation) != expected_generations[sequence]:
			raise ValueError(
				'periodic refresh chain generation lineage does not match manifests'
			)
	if expected_config is not None:
		scientific = _required_mapping(
			_required_mapping(expected_config, 'identity'), 'scientific_identity'
		)
		expected_hash = _periodic_fixed_preprocessing_identity_sha256(scientific)
		if chain_fixed_hash != expected_hash:
			raise ValueError(
				'periodic refresh chain fixed preprocessing identity drift'
			)


def _validate_periodic_refresh_state_against_config(  # noqa: C901, PLR0912, PLR0915
	*,
	state: Mapping[str, object],
	loaded_payloads: list[Mapping[str, object]],
	expected_config: Mapping[str, object],
) -> None:
	scientific = _required_mapping(
		_required_mapping(expected_config, 'identity'), 'scientific_identity'
	)
	schedule = tuple(PERIODIC_REFRESH_SCHEDULE)
	active_index = int(state['active_generation_index'])
	if active_index < 0 or active_index > len(schedule):
		raise ValueError(
			'active_generation_index exceeds the periodic refresh schedule'
		)
	last_refresh = int(state['last_completed_refresh_epoch'])
	next_refresh = state['next_scheduled_refresh_epoch']
	phase = state['refresh_phase']
	expected_target_sha256 = _required_sha256(
		scientific.get('initial_hard_target_manifest_sha256'),
		'identity.scientific_identity.initial_hard_target_manifest_sha256',
	)
	initial_target_path = Path(
		_required_string(
			_required_mapping(expected_config, 'pseudo_targets').get('manifest'),
			'pseudo_targets.manifest',
		)
	)
	if active_index == 0 and Path(
		str(state['active_target_manifest_path'])
	) != initial_target_path:
		raise ValueError('initial active target manifest path does not match config')
	if active_index == 0 and state['source_student_state_sha256'] is not None:
		raise ValueError('initial active generation cannot bind a student state hash')
	if active_index > 0 and state['source_student_state_sha256'] is None:
		raise ValueError('refreshed active generation requires a student state hash')
	expected_fixed_hash = _periodic_fixed_preprocessing_identity_sha256(scientific)
	if state['fixed_preprocessing_hmm_identity_sha256'] != expected_fixed_hash:
		raise ValueError('target refresh fixed preprocessing/HMM identity mismatch')
	if last_refresh not in ({0} | set(schedule)):
		raise ValueError('last_completed_refresh_epoch is not on the exact schedule')
	if next_refresh is not None and next_refresh not in schedule:
		raise ValueError('next_scheduled_refresh_epoch is not on the exact schedule')
	expected_index = 0 if last_refresh == 0 else schedule.index(last_refresh) + 1
	if phase == 'refresh_required':
		if next_refresh is None or next_refresh <= last_refresh:
			raise ValueError(
				'refresh_required state does not identify the next refresh'
			)
		expected_index = schedule.index(next_refresh)
	elif phase == 'refresh_complete' and last_refresh == 0:
		raise ValueError('refresh_complete state must identify a completed refresh')
	if active_index != expected_index:
		raise ValueError('active generation index does not match refresh phase')
	expected_next = next(
		(epoch for epoch in schedule if epoch > last_refresh), None
	)
	if next_refresh != expected_next:
		raise ValueError(
			'next scheduled refresh does not follow completed refresh epoch'
		)
	if phase == 'refresh_required' and next_refresh != expected_next:
		raise ValueError('refresh_required state schedule mismatch')
	for index, payload in enumerate(loaded_payloads):
		expected_id = (
			'refresh_0000_initial'
			if index == 0
			else f'refresh_{index:04d}_epoch{schedule[index - 1]:03d}'
		)
		expected_epoch = 0 if index == 0 else schedule[index - 1]
		if payload.get('generation_id') != expected_id or payload.get(
			'refresh_after_epoch'
		) != expected_epoch:
			raise ValueError(
				f'generation {index} does not match the exact periodic schedule'
			)
		initial_target = payload.get('initial_hard_target_manifest')
		if not isinstance(initial_target, Mapping):
			raise TypeError('generation initial target reference must be a mapping')
		if (
			initial_target.get('path') != str(initial_target_path)
			or initial_target.get('sha256') != expected_target_sha256
		):
			raise ValueError(
				f'generation {index} initial target manifest lineage mismatch'
			)
		fixed_identity = payload.get('fixed_preprocessing_hmm_identity')
		if not isinstance(fixed_identity, Mapping):
			raise TypeError(
				f'generation {index} fixed preprocessing identity must be a mapping'
			)
		artifacts = fixed_identity.get('artifacts')
		if not isinstance(artifacts, Mapping):
			raise TypeError(f'generation {index} fixed HMM artifacts must be a mapping')
		configured_artifacts = _required_mapping(
			scientific, 'initial_hmm_artifacts'
		)
		configured_common = _required_mapping(configured_artifacts, 'common')
		configured_heads = _required_mapping(configured_artifacts, 'heads')
		for k in ('6', '8', '10'):
			artifact = artifacts.get(k)
			if not isinstance(artifact, Mapping):
				raise TypeError(f'generation {index} initial HMM k={k} is invalid')
			configured_head = _required_mapping(configured_heads, k)
			for artifact_key, configured_key in (
				('centers', 'centers'),
				('hmm_model', 'hmm_model'),
				('metadata', 'model_metadata'),
			):
				artifact_ref = _required_mapping(
					artifact, artifact_key
				)
				configured_ref = _required_mapping(
					configured_head, configured_key
				)
				if (
					artifact_ref.get('path') != configured_ref.get('path')
					or artifact_ref.get('sha256') != configured_ref.get('sha256')
				):
					raise ValueError(
						f'generation {index} initial HMM {k} {artifact_key} drift'
					)
			for artifact_key, configured_key in (
				('preprocessor', 'preprocessor'),
				('residualizer', 'residualizer'),
			):
				artifact_value = artifact.get(artifact_key)
				configured_value = configured_common.get(configured_key)
				if artifact_value is None or configured_value is None:
					if artifact_value is not configured_value:
						raise ValueError(
							f'generation {index} initial HMM {k} {artifact_key} drift'
						)
					continue
				if not isinstance(artifact_value, Mapping) or not isinstance(
					configured_value, Mapping
				):
					raise TypeError(
						f'generation {index} initial HMM {k} {artifact_key} is invalid'
					)
				if (
					artifact_value.get('path') != configured_value.get('path')
					or artifact_value.get('sha256') != configured_value.get('sha256')
				):
					raise ValueError(
						f'generation {index} initial HMM {k} {artifact_key} drift'
					)
			for artifact_key in ('clustering_config', 'source_embedding_metadata'):
				artifact_ref = fixed_identity.get(artifact_key)
				configured_ref = configured_common.get(artifact_key)
				if not isinstance(artifact_ref, Mapping) or not isinstance(
					configured_ref, Mapping
				):
					raise TypeError(
						f'generation {index} fixed {artifact_key} is invalid'
					)
				if dict(artifact_ref) != dict(configured_ref):
					raise ValueError(
						f'generation {index} fixed {artifact_key} drift'
					)
	if active_index > 0:
		active_source = loaded_payloads[-1].get('source_student_state_sha256')
		if active_source != state['source_student_state_sha256']:
			raise ValueError('active generation source student state hash mismatch')


def _load_json_mapping(path: Path, label: str) -> dict[str, object]:
	try:
		value = json.loads(path.read_text(encoding='utf-8'))
	except (OSError, json.JSONDecodeError) as exc:
		raise ValueError(f'{label} must be valid JSON: {path}') from exc
	if not isinstance(value, dict):
		raise TypeError(f'{label} must be a JSON object')
	return value


def _absolute_state_path(value: object, label: str) -> Path:
	path = Path(_required_string(value, label))
	if not path.is_absolute():
		raise ValueError(f'{label} must be an absolute path')
	return path


def _nonnegative_int_value(value: object, label: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int) or value < 0:
		raise ValueError(f'{label} must be a nonnegative integer')
	return value


def _positive_int_value(value: object, label: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int) or value < 1:
		raise ValueError(f'{label} must be a positive integer')
	return value


def _soft_multi_head_checkpoint_identity(  # noqa: PLR0913
	*,
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, torch.Tensor],
	control_identity: Mapping[str, object] | None,
	optimizer: torch.optim.Optimizer,
	student: torch.nn.Module,
	head: torch.nn.Module,
) -> dict[str, object]:
	"""Build schema-v3 identity for an immutable soft-posterior run."""
	head_config = _required_mapping(stratigraphy_config, 'head')
	pseudo_targets = _required_mapping(stratigraphy_config, 'pseudo_targets')
	identity = _required_mapping(stratigraphy_config, 'identity')
	scientific = _required_mapping(identity, 'scientific_identity')
	manifest_path = _required_string(
		pseudo_targets.get('manifest'), 'pseudo_targets.manifest'
	)
	manifest_sha256 = _file_sha256(Path(manifest_path))
	per_head_posteriors = _posterior_per_head_hashes(Path(manifest_path))
	if scientific.get('posterior_manifest_sha256') != manifest_sha256:
		raise ValueError(
			'posterior manifest SHA-256 does not match scientific identity'
		)
	if scientific.get('posterior_head_hashes') != per_head_posteriors:
		raise ValueError('scientific identity posterior hashes do not match manifest')
	if not isinstance(control_identity, Mapping):
		raise TypeError('multi-head checkpoint requires control identity')
	inputs = _required_mapping(control_identity, 'input_identities')
	initial_states = control_identity.get('initial_state_sha256')
	if not isinstance(initial_states, Mapping):
		raise TypeError('multi-head checkpoint requires initial state hashes')
	return {
		'schema_version': 3,
		'head_spec': head_config['spec'],
		'head_ks': list(_head_ks(head_config.get('ks'))),
		'target_representation': scientific['target_representation'],
		'posterior_semantics': scientific['posterior_semantics'],
		'posterior_cost_temperature': scientific['posterior_cost_temperature'],
		'posterior_manifest_sha256': manifest_sha256,
		'posterior_manifest': {'path': manifest_path, 'sha256': manifest_sha256},
		'per_head_posteriors': per_head_posteriors,
		'consistency_policy': scientific['consistency_policy'],
		'consistency_weight': scientific['consistency_weight'],
		'consistency_beta': scientific['consistency_beta'],
		'model_tag': identity.get('model_tag'),
		'output_root': _required_mapping(stratigraphy_config, 'paths').get(
			'output_root'
		),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'stratigraphy_state_sha256': _state_sha256(stratigraphy_state_dict),
		'optimizer_group_identity': _optimizer_group_identity(
			optimizer,
			parameter_names=_stratigraphy_parameter_names(student, head),
		),
		'teacher_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'teacher_checkpoint').get('sha256'),
			'input_identities.teacher_checkpoint.sha256',
		),
		'student_init_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'student_init_checkpoint').get('sha256'),
			'input_identities.student_init_checkpoint.sha256',
		),
		'initial_student_state_sha256': _required_sha256(
			initial_states.get('student'), 'initial_state_sha256.student'
		),
		'initial_head_state_sha256': _required_sha256(
			initial_states.get('head'), 'initial_state_sha256.head'
		),
	}


def _lateral_multi_head_checkpoint_identity(  # noqa: PLR0913
	*,
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, torch.Tensor],
	control_identity: Mapping[str, object] | None,
	optimizer: torch.optim.Optimizer,
	student: torch.nn.Module,
	head: torch.nn.Module,
) -> dict[str, object]:
	"""Build schema-v4 identity for immutable M5-LS hard supervision."""
	head_config = _required_mapping(stratigraphy_config, 'head')
	pseudo_targets = _required_mapping(stratigraphy_config, 'pseudo_targets')
	identity = _required_mapping(stratigraphy_config, 'identity')
	scientific = _required_mapping(identity, 'scientific_identity')
	manifest_path = _required_string(
		pseudo_targets.get('manifest'), 'pseudo_targets.manifest'
	)
	manifest_sha256 = _file_sha256(Path(manifest_path))
	per_head_targets = _lateral_per_head_target_hashes(Path(manifest_path))
	if scientific.get('lateral_target_manifest_sha256') != manifest_sha256:
		raise ValueError('lateral manifest SHA-256 does not match scientific identity')
	if scientific.get('lateral_target_head_hashes') != per_head_targets:
		raise ValueError('lateral target hashes do not match scientific identity')
	if not isinstance(control_identity, Mapping):
		raise TypeError('multi-head checkpoint requires control identity')
	inputs = _required_mapping(control_identity, 'input_identities')
	initial_states = control_identity.get('initial_state_sha256')
	if not isinstance(initial_states, Mapping):
		raise TypeError('multi-head checkpoint requires initial state hashes')
	return {
		'schema_version': 4,
		'head_spec': head_config['spec'],
		'head_ks': list(_head_ks(head_config.get('ks'))),
		'target_representation': scientific['target_representation'],
		'target_semantics': scientific['target_semantics'],
		'lateral_target_manifest_sha256': manifest_sha256,
		'lateral_target_manifest': {'path': manifest_path, 'sha256': manifest_sha256},
		'per_head_lateral_targets': per_head_targets,
		'source_hard_manifest_sha256': scientific['source_hard_manifest_sha256'],
		'source_posterior_manifest_sha256': scientific[
			'source_posterior_manifest_sha256'
		],
		'lateral_smoothing': scientific['lateral_smoothing'],
		'consistency_policy': scientific['consistency_policy'],
		'consistency_weight': scientific['consistency_weight'],
		'consistency_beta': scientific['consistency_beta'],
		'model_tag': identity.get('model_tag'),
		'output_root': _required_mapping(stratigraphy_config, 'paths').get(
			'output_root'
		),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'stratigraphy_state_sha256': _state_sha256(stratigraphy_state_dict),
		'optimizer_group_identity': _optimizer_group_identity(
			optimizer,
			parameter_names=_stratigraphy_parameter_names(student, head),
		),
		'teacher_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'teacher_checkpoint').get('sha256'),
			'input_identities.teacher_checkpoint.sha256',
		),
		'student_init_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'student_init_checkpoint').get('sha256'),
			'input_identities.student_init_checkpoint.sha256',
		),
		'initial_student_state_sha256': _required_sha256(
			initial_states.get('student'), 'initial_state_sha256.student'
		),
		'initial_head_state_sha256': _required_sha256(
			initial_states.get('head'), 'initial_state_sha256.head'
		),
	}


def _xy_neighbor_consensus_multi_head_checkpoint_identity(  # noqa: PLR0913
	*,
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, torch.Tensor],
	control_identity: Mapping[str, object] | None,
	optimizer: torch.optim.Optimizer,
	student: torch.nn.Module,
	head: torch.nn.Module,
) -> dict[str, object]:
	"""Build schema-v5 identity for source-hard XY consensus supervision."""
	head_config = _required_mapping(stratigraphy_config, 'head')
	pseudo_targets = _required_mapping(stratigraphy_config, 'pseudo_targets')
	identity = _required_mapping(stratigraphy_config, 'identity')
	scientific = _required_mapping(identity, 'scientific_identity')
	manifest_path = _required_string(
		pseudo_targets.get('manifest'), 'pseudo_targets.manifest'
	)
	manifest_sha256 = _file_sha256(Path(manifest_path))
	(
		per_head_targets,
		source_hard_manifest_sha256,
		smoothing,
	) = _xy_neighbor_consensus_target_provenance(Path(manifest_path))
	if (
		scientific.get('xy_neighbor_consensus_target_manifest_sha256')
		!= manifest_sha256
	):
		raise ValueError(
			'XY neighbor consensus manifest SHA-256 does not match scientific identity'
		)
	if scientific.get('xy_neighbor_consensus_target_head_hashes') != per_head_targets:
		raise ValueError(
			'XY neighbor consensus target hashes do not match scientific identity'
		)
	if scientific.get('source_hard_manifest_sha256') != source_hard_manifest_sha256:
		raise ValueError(
			'XY neighbor consensus source hard manifest does not match scientific '
			'identity'
		)
	if _to_plain_value(scientific.get('xy_neighbor_consensus_smoothing')) != smoothing:
		raise ValueError(
			'XY neighbor consensus smoothing policy does not match scientific identity'
		)
	if not isinstance(control_identity, Mapping):
		raise TypeError('multi-head checkpoint requires control identity')
	inputs = _required_mapping(control_identity, 'input_identities')
	initial_states = control_identity.get('initial_state_sha256')
	if not isinstance(initial_states, Mapping):
		raise TypeError('multi-head checkpoint requires initial state hashes')
	return {
		'schema_version': 5,
		'head_spec': head_config['spec'],
		'head_ks': list(_head_ks(head_config.get('ks'))),
		'target_representation': scientific['target_representation'],
		'target_semantics': scientific['target_semantics'],
		'xy_neighbor_consensus_target_manifest_sha256': manifest_sha256,
		'xy_neighbor_consensus_target_manifest': {
			'path': manifest_path,
			'sha256': manifest_sha256,
		},
		'per_head_xy_neighbor_consensus_targets': per_head_targets,
		'source_hard_manifest_sha256': source_hard_manifest_sha256,
		'xy_neighbor_consensus_smoothing': smoothing,
		'consistency_policy': scientific['consistency_policy'],
		'consistency_weight': scientific['consistency_weight'],
		'consistency_beta': scientific['consistency_beta'],
		'model_tag': identity.get('model_tag'),
		'output_root': _required_mapping(stratigraphy_config, 'paths').get(
			'output_root'
		),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'stratigraphy_state_sha256': _state_sha256(stratigraphy_state_dict),
		'optimizer_group_identity': _optimizer_group_identity(
			optimizer,
			parameter_names=_stratigraphy_parameter_names(student, head),
		),
		'teacher_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'teacher_checkpoint').get('sha256'),
			'input_identities.teacher_checkpoint.sha256',
		),
		'student_init_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'student_init_checkpoint').get('sha256'),
			'input_identities.student_init_checkpoint.sha256',
		),
		'initial_student_state_sha256': _required_sha256(
			initial_states.get('student'), 'initial_state_sha256.student'
		),
		'initial_head_state_sha256': _required_sha256(
			initial_states.get('head'), 'initial_state_sha256.head'
		),
	}


def _xy_neighbor_unanimous_multi_head_checkpoint_identity(  # noqa: PLR0913
	*,
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, torch.Tensor],
	control_identity: Mapping[str, object] | None,
	optimizer: torch.optim.Optimizer,
	student: torch.nn.Module,
	head: torch.nn.Module,
) -> dict[str, object]:
	"""Build schema-v6 identity for source-hard XY unanimous supervision."""
	head_config = _required_mapping(stratigraphy_config, 'head')
	pseudo_targets = _required_mapping(stratigraphy_config, 'pseudo_targets')
	identity = _required_mapping(stratigraphy_config, 'identity')
	scientific = _required_mapping(identity, 'scientific_identity')
	manifest_path = _required_string(
		pseudo_targets.get('manifest'), 'pseudo_targets.manifest'
	)
	manifest_sha256 = _file_sha256(Path(manifest_path))
	(
		per_head_targets,
		source_hard_manifest_sha256,
		smoothing,
	) = _xy_neighbor_unanimous_target_provenance(Path(manifest_path))
	if (
		scientific.get('xy_neighbor_unanimous_target_manifest_sha256')
		!= manifest_sha256
	):
		raise ValueError(
			'XY neighbor unanimous manifest SHA-256 does not match scientific identity'
		)
	if scientific.get('xy_neighbor_unanimous_target_head_hashes') != per_head_targets:
		raise ValueError(
			'XY neighbor unanimous target hashes do not match scientific identity'
		)
	if scientific.get('source_hard_manifest_sha256') != source_hard_manifest_sha256:
		raise ValueError(
			'XY neighbor unanimous source hard manifest does not match scientific '
			'identity'
		)
	if _to_plain_value(scientific.get('xy_neighbor_unanimous_smoothing')) != smoothing:
		raise ValueError(
			'XY neighbor unanimous smoothing policy does not match scientific identity'
		)
	if not isinstance(control_identity, Mapping):
		raise TypeError('multi-head checkpoint requires control identity')
	inputs = _required_mapping(control_identity, 'input_identities')
	initial_states = control_identity.get('initial_state_sha256')
	if not isinstance(initial_states, Mapping):
		raise TypeError('multi-head checkpoint requires initial state hashes')
	return {
		'schema_version': 6,
		'head_spec': head_config['spec'],
		'head_ks': list(_head_ks(head_config.get('ks'))),
		'target_representation': scientific['target_representation'],
		'target_semantics': scientific['target_semantics'],
		'xy_neighbor_unanimous_target_manifest_sha256': manifest_sha256,
		'xy_neighbor_unanimous_target_manifest': {
			'path': manifest_path,
			'sha256': manifest_sha256,
		},
		'per_head_xy_neighbor_unanimous_targets': per_head_targets,
		'source_hard_manifest_sha256': source_hard_manifest_sha256,
		'xy_neighbor_unanimous_smoothing': smoothing,
		'consistency_policy': scientific['consistency_policy'],
		'consistency_weight': scientific['consistency_weight'],
		'consistency_beta': scientific['consistency_beta'],
		'model_tag': identity.get('model_tag'),
		'output_root': _required_mapping(stratigraphy_config, 'paths').get(
			'output_root'
		),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'stratigraphy_state_sha256': _state_sha256(stratigraphy_state_dict),
		'optimizer_group_identity': _optimizer_group_identity(
			optimizer,
			parameter_names=_stratigraphy_parameter_names(student, head),
		),
		'teacher_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'teacher_checkpoint').get('sha256'),
			'input_identities.teacher_checkpoint.sha256',
		),
		'student_init_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'student_init_checkpoint').get('sha256'),
			'input_identities.student_init_checkpoint.sha256',
		),
		'initial_student_state_sha256': _required_sha256(
			initial_states.get('student'), 'initial_state_sha256.student'
		),
		'initial_head_state_sha256': _required_sha256(
			initial_states.get('head'), 'initial_state_sha256.head'
		),
	}


def _validate_multi_head_identity(  # noqa: C901, PLR0912
	*,
	identity: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, object],
) -> None:
	if _is_periodic_refresh_config(stratigraphy_config):
		_validate_periodic_refresh_identity(
			identity=identity,
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
		)
		return
	if _is_center_trace_config(stratigraphy_config):
		_validate_center_trace_identity(
			identity=identity,
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
		)
		return
	if _is_soft_multi_head_config(stratigraphy_config):
		_validate_soft_multi_head_identity(
			identity=identity,
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
		)
		return
	if _is_lateral_multi_head_config(stratigraphy_config):
		_validate_lateral_multi_head_identity(
			identity=identity,
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
		)
		return
	if _is_xy_neighbor_unanimous_multi_head_config(stratigraphy_config):
		_validate_xy_neighbor_unanimous_multi_head_identity(
			identity=identity,
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
		)
		return
	if _is_xy_neighbor_consensus_multi_head_config(stratigraphy_config):
		_validate_xy_neighbor_consensus_multi_head_identity(
			identity=identity,
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
		)
		return
	_validate_multi_head_config_and_state(
		stratigraphy_config,
		{
			str(key): value
			for key, value in stratigraphy_state_dict.items()
			if isinstance(value, torch.Tensor)
		},
	)
	head = _required_mapping(stratigraphy_config, 'head')
	config_identity = _required_mapping(stratigraphy_config, 'identity')
	paths = _required_mapping(stratigraphy_config, 'paths')
	scientific = _required_mapping(config_identity, 'scientific_identity')
	_reject_xy_neighbor_consensus_fields_from_legacy_identity(identity, scientific)
	if list(_head_ks(head.get('ks'))) != identity.get('head_ks'):
		raise ValueError('checkpoint head_ks does not match stratigraphy config')
	if identity.get('head_spec') != head.get('spec'):
		raise ValueError('checkpoint head_spec does not match stratigraphy config')
	for key, expected in (
		('model_tag', config_identity.get('model_tag')),
		('output_root', paths.get('output_root')),
	):
		if identity.get(key) != expected:
			raise ValueError(f'checkpoint {key} does not match stratigraphy config')
	if identity.get('stratigraphy_state_sha256') != _state_sha256(
		stratigraphy_state_dict
	):
		raise ValueError('checkpoint stratigraphy state SHA-256 mismatch')
	_required_sha256(
		identity.get('initial_student_state_sha256'),
		'checkpoint initial_student_state_sha256',
	)
	_required_sha256(
		identity.get('initial_head_state_sha256'),
		'checkpoint initial_head_state_sha256',
	)
	_required_sha256(
		identity.get('teacher_checkpoint_sha256'),
		'checkpoint teacher_checkpoint_sha256',
	)
	_required_sha256(
		identity.get('student_init_checkpoint_sha256'),
		'checkpoint student_init_checkpoint_sha256',
	)
	scientific = _validate_multi_head_target_manifest_identity(
		identity=identity,
		stratigraphy_config=stratigraphy_config,
	)
	if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
		scientific
	):
		raise ValueError('checkpoint scientific identity SHA-256 mismatch')
	for key in (
		'target_head_hashes',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
	):
		checkpoint_key = 'per_head_targets' if key == 'target_head_hashes' else key
		if identity.get(checkpoint_key) != scientific.get(key):
			raise ValueError(
				f'checkpoint {checkpoint_key} does not match scientific identity'
			)


def _validate_periodic_refresh_identity(  # noqa: C901, PLR0912
	*,
	identity: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, object],
) -> None:
	"""Validate the closed schema-8 periodic-refresh identity."""
	if identity.get('schema_version') != 8:
		raise ValueError('periodic refresh checkpoint requires schema_version 8')
	unknown = sorted(set(identity) - _PERIODIC_REFRESH_CHECKPOINT_IDENTITY_FIELDS)
	if unknown:
		raise ValueError(
		f'schema-8 checkpoint identity has unsupported field(s): {unknown!r}'
	)
	_validate_multi_head_config_and_state(
		stratigraphy_config,
		{
			str(key): value
			for key, value in stratigraphy_state_dict.items()
			if isinstance(value, torch.Tensor)
		},
	)
	head = _required_mapping(stratigraphy_config, 'head')
	config_identity = _required_mapping(stratigraphy_config, 'identity')
	scientific = _required_mapping(config_identity, 'scientific_identity')
	paths = _required_mapping(stratigraphy_config, 'paths')
	if identity.get('head_spec') != 'multi_resolution_ordered_prototypes_v1':
		raise ValueError('schema-8 checkpoint head_spec is invalid')
	if identity.get('head_ks') != [6, 8, 10]:
		raise ValueError('schema-8 checkpoint head_ks must be [6, 8, 10]')
	if head.get('spec') != identity.get('head_spec') or list(
		head.get('ks', ())
	) != [6, 8, 10]:
		raise ValueError('schema-8 checkpoint head config is not [6, 8, 10]')
	for key, expected in (
		('model_tag', config_identity.get('model_tag')),
		('output_root', paths.get('output_root')),
	):
		if identity.get(key) != expected:
			raise ValueError(f'schema-8 checkpoint {key} does not match config')
	if identity.get('target_representation') != PERIODIC_REFRESH_TARGET_REPRESENTATION:
		raise ValueError(
			'schema-8 checkpoint target representation is not hard Viterbi'
		)
	manifest = identity.get('initial_hard_target_manifest')
	if not isinstance(manifest, Mapping):
		raise TypeError('schema-8 initial_hard_target_manifest must be a mapping')
	manifest_path = _absolute_state_path(
		manifest.get('path'), 'schema-8 initial target manifest path'
	)
	manifest_sha256 = _required_sha256(
		manifest.get('sha256'), 'schema-8 initial target manifest hash'
	)
	if not manifest_path.is_file() or _file_sha256(manifest_path) != manifest_sha256:
		raise ValueError('schema-8 initial target manifest is missing or hash-drifted')
	if identity.get('initial_hard_target_manifest_sha256') != manifest_sha256:
		raise ValueError('schema-8 initial target manifest hash fields disagree')
	if identity.get('initial_per_head_targets') != _manifest_per_head_target_hashes(
		manifest_path
	):
		raise ValueError(
			'schema-8 initial per-head target hashes do not match manifest'
		)
	pseudo_targets = _required_mapping(stratigraphy_config, 'pseudo_targets')
	if manifest_path != Path(
		_required_string(pseudo_targets.get('manifest'), 'pseudo_targets.manifest')
	):
		raise ValueError('schema-8 initial target manifest path does not match config')
	for key in _PERIODIC_REFRESH_CHECKPOINT_IDENTITY_FIELDS:
		if key in {
			'schema_version',
			'head_spec',
			'head_ks',
			'initial_hard_target_manifest_sha256',
			'initial_hard_target_manifest',
			'initial_per_head_targets',
			'optimizer_group_identity',
			'teacher_checkpoint_sha256',
			'student_init_checkpoint_sha256',
			'initial_student_state_sha256',
			'initial_head_state_sha256',
			'spatial_context_state_sha256',
			'initial_spatial_context_state_sha256',
			'stratigraphy_state_sha256',
			'student_state_sha256',
			'target_refresh_state_sha256',
			'model_tag',
			'output_root',
			'scientific_identity_sha256',
		}:
			continue
		if identity.get(key) != scientific.get(key):
			raise ValueError(
				f'schema-8 checkpoint {key} does not match scientific identity'
			)
	if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
		scientific
	):
		raise ValueError('schema-8 checkpoint scientific identity SHA-256 mismatch')
	_validate_periodic_refresh_config_identity_values(
		stratigraphy_config, scientific
	)
	for key in (
		'student_state_sha256',
		'stratigraphy_state_sha256',
		'spatial_context_state_sha256',
		'initial_spatial_context_state_sha256',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'target_refresh_state_sha256',
		'teacher_checkpoint_sha256',
		'student_init_checkpoint_sha256',
	):
		_required_sha256(identity.get(key), f'schema-8 checkpoint {key}')


def _validate_periodic_refresh_config_identity_values(  # noqa: C901, PLR0912
	config: Mapping[str, object], scientific: Mapping[str, object]
) -> None:
	identity = _required_mapping(config, 'identity')
	paths = _required_mapping(config, 'paths')
	periodic = _validate_periodic_refresh_config(
		_required_mapping(config, 'pseudo_target_refresh'),
		output_root=Path(
			_required_string(paths.get('output_root'), 'paths.output_root')
		),
		train=_required_mapping(config, 'train'),
		pseudo_targets=_required_mapping(config, 'pseudo_targets'),
		head=_required_mapping(config, 'head'),
		multi_head=True,
	)
	if periodic is None:
		raise ValueError('schema-8 periodic refresh config cannot be disabled')
	if identity.get('model_tag') != PERIODIC_REFRESH_MODEL_TAG:
		raise ValueError('periodic refresh model tag mismatch')
	checks = {
		'experiment_role': PERIODIC_REFRESH_EXPERIMENT_ROLE,
		'variant': PERIODIC_REFRESH_VARIANT,
		'model_role': PERIODIC_REFRESH_MODEL_ROLE,
		'head_spec': 'multi_resolution_ordered_prototypes_v1',
		'head_ks': [6, 8, 10],
		'target_representation': PERIODIC_REFRESH_TARGET_REPRESENTATION,
		'target_refresh_semantics': PERIODIC_REFRESH_SEMANTICS,
		'refresh_schedule_semantics': PERIODIC_REFRESH_SCHEDULE_SEMANTICS,
		'refresh_after_epochs': list(PERIODIC_REFRESH_SCHEDULE),
		'hmm_iterations_per_refresh': 2,
		'embedding_source': 'current_student',
		'embedding_mode': 'unmasked_eval_full_survey',
		'refresh_embedding_semantics': PERIODIC_REFRESH_EMBEDDING_SEMANTICS,
		'center_initialization': 'previous_generation',
		'center_update': 'full_mean',
		'center_update_semantics': PERIODIC_REFRESH_CENTER_UPDATE_SEMANTICS,
		'preprocessing_policy': PERIODIC_REFRESH_PREPROCESSING_POLICY,
		'target_activation_policy': PERIODIC_REFRESH_TARGET_ACTIVATION_POLICY,
		'empty_state_policy': 'error',
		'checkpoint_selection_policy': PERIODIC_REFRESH_CHECKPOINT_SELECTION_POLICY,
		'initial_hard_target_manifest_sha256': periodic[
			'initial_hard_target_manifest'
		]['sha256'],
	}
	for key, expected in checks.items():
		if scientific.get(key) != expected:
			raise ValueError(f'periodic refresh scientific identity mismatch: {key}')
	for key in (
		'generation_root',
		'initial_hmm_artifacts',
		'fixed_preprocessor_sha256',
		'fixed_residualizer_sha256',
		'fixed_clustering_config_sha256',
		'source_embedding_metadata_sha256',
		'source_valid_token_hashes',
		'feature_dimension',
	):
		if scientific.get(key) != periodic.get(key):
			raise ValueError(f'periodic refresh scientific identity mismatch: {key}')
	if scientific.get('output_root') not in {None, paths.get('output_root')}:
		# output_root is a checkpoint identity field, not scientific; retain a
		# defensive check for hand-authored scientific identities.
		raise ValueError('periodic refresh scientific output root mismatch')
	if _required_mapping(config, 'train').get('epochs') != 25:
		raise ValueError('periodic refresh requires train.epochs == 25')
	if _required_mapping(config, 'student').get('unfreeze_top_blocks') != 1:
		raise ValueError('periodic refresh requires one unfrozen student block')
	spatial = _required_mapping(config, 'spatial_context')
	for key, expected in CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT.items():
		if spatial.get(key) != expected or scientific.get(
			'objective_semantics' if key == 'objective' else key
		) != expected:
			raise ValueError(f'periodic refresh spatial identity mismatch: {key}')
	loss = _required_mapping(config, 'loss')
	for key, expected in (
		('prototype_weight', 1.0),
		('usage_weight', 0.005),
		('consistency_weight', 0.0),
		('consistency_beta', 0.1),
		('distillation_weight', 0.2),
	):
		if loss.get(key) != expected or scientific.get(key) != expected:
			raise ValueError(f'periodic refresh loss identity mismatch: {key}')


def _validate_center_trace_identity(  # noqa: C901, PLR0912
	*,
	identity: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, object],
) -> None:
	"""Validate schema-7 identity without accepting another target route."""
	if identity.get('schema_version') != 7:
		raise ValueError('center-trace masked checkpoint requires schema_version 7')
	unknown = sorted(set(identity) - _CENTER_TRACE_CHECKPOINT_IDENTITY_FIELDS)
	if unknown:
		raise ValueError(
			f'schema-7 checkpoint identity has unsupported field(s): {unknown!r}'
		)
	_validate_multi_head_config_and_state(
		stratigraphy_config,
		{
			str(key): value
			for key, value in stratigraphy_state_dict.items()
			if isinstance(value, torch.Tensor)
		},
	)
	head = _required_mapping(stratigraphy_config, 'head')
	config_identity = _required_mapping(stratigraphy_config, 'identity')
	scientific = _required_mapping(config_identity, 'scientific_identity')
	paths = _required_mapping(stratigraphy_config, 'paths')
	if identity.get('head_spec') != _CENTER_TRACE_HEAD_SPEC:
		raise ValueError('schema-7 checkpoint head_spec is not center-trace multi-head')
	if identity.get('head_ks') != list(_CENTER_TRACE_HEAD_KS):
		raise ValueError('schema-7 checkpoint head_ks must be [6, 8, 10]')
	if head.get('spec') != _CENTER_TRACE_HEAD_SPEC or list(head.get('ks', ())) != list(
		_CENTER_TRACE_HEAD_KS
	):
		raise ValueError('schema-7 checkpoint head config is not [6, 8, 10]')
	for key, expected in (
		('model_tag', config_identity.get('model_tag')),
		('output_root', paths.get('output_root')),
	):
		if identity.get(key) != expected:
			raise ValueError(f'checkpoint {key} does not match stratigraphy config')
	if identity.get('stratigraphy_state_sha256') != _state_sha256(
		{
			str(key): value
			for key, value in stratigraphy_state_dict.items()
			if isinstance(value, torch.Tensor)
		}
	):
		raise ValueError('checkpoint stratigraphy state SHA-256 mismatch')
	pseudo_targets = _required_mapping(stratigraphy_config, 'pseudo_targets')
	target = identity.get('target_manifest')
	if not isinstance(target, Mapping):
		raise TypeError('schema-7 checkpoint target_manifest must be a mapping')
	target_path = Path(
		_required_string(target.get('path'), 'checkpoint target_manifest.path')
	)
	target_sha256 = _required_sha256(
		target.get('sha256'), 'checkpoint target_manifest.sha256'
	)
	if target_sha256 != _file_sha256(target_path):
		raise ValueError('checkpoint target manifest SHA-256 mismatch')
	if identity.get('target_manifest_sha256') != target_sha256:
		raise ValueError('checkpoint target manifest hash fields disagree')
	if identity.get('per_head_targets') != _manifest_per_head_target_hashes(
		target_path
	):
		raise ValueError(
			'checkpoint per-head target hashes do not match target manifest'
		)
	if _required_string(
		pseudo_targets.get('manifest'), 'pseudo_targets.manifest'
	) != str(target_path):
		raise ValueError('schema-7 target manifest path does not match config')
	for key in _CENTER_TRACE_CHECKPOINT_IDENTITY_FIELDS:
		if key in {
			'schema_version',
			'head_spec',
			'head_ks',
			'target_manifest_sha256',
			'target_manifest',
			'per_head_targets',
			'optimizer_group_identity',
			'teacher_checkpoint_sha256',
			'student_init_checkpoint_sha256',
			'initial_student_state_sha256',
			'initial_head_state_sha256',
			'spatial_context_state_sha256',
			'initial_spatial_context_state_sha256',
			'stratigraphy_state_sha256',
			'model_tag',
			'output_root',
			'scientific_identity_sha256',
			'student_state_sha256',
		}:
			continue
		if identity.get(key) != scientific.get(key):
			raise ValueError(
				f'checkpoint {key} does not match center-trace scientific identity'
			)
	if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
		scientific
	):
		raise ValueError('checkpoint scientific identity SHA-256 mismatch')
	for key in (
		'student_state_sha256',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'initial_spatial_context_state_sha256',
		'teacher_checkpoint_sha256',
		'student_init_checkpoint_sha256',
		'spatial_context_state_sha256',
	):
		_required_sha256(identity.get(key), f'checkpoint {key}')
	_validate_center_trace_config_identity_values(
		stratigraphy_config, scientific
	)


def _validate_center_trace_config_identity_values(  # noqa: C901
	config: Mapping[str, object], scientific: Mapping[str, object]
) -> None:
	spatial = _required_mapping(config, 'spatial_context')
	for key, expected in CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT.items():
		if spatial.get(key) != expected or scientific.get(
			'objective_semantics' if key == 'objective' else key
		) != expected:
			raise ValueError(f'schema-7 center-trace {key} identity mismatch')
	if scientific.get('experiment_role') != CENTER_TRACE_EXPERIMENT_ROLE:
		raise ValueError('schema-7 center-trace experiment role mismatch')
	if scientific.get('variant') != CENTER_TRACE_VARIANT:
		raise ValueError('schema-7 center-trace variant mismatch')
	if scientific.get('target_representation') != _CENTER_TRACE_TARGET_REPRESENTATION:
		raise ValueError('schema-7 center-trace target representation mismatch')
	if scientific.get('supervised_loss') != CENTER_TRACE_SUPERVISED_LOSS:
		raise ValueError('schema-7 center-trace supervised loss mismatch')
	if scientific.get('consistency_policy') != CENTER_TRACE_CONSISTENCY_POLICY:
		raise ValueError('schema-7 center-trace consistency policy mismatch')
	if scientific.get('student_unfreeze_top_blocks') != 1:
		raise ValueError('schema-7 center-trace unfreeze depth must be 1')
	if scientific.get('head_ks') != list(_CENTER_TRACE_HEAD_KS):
		raise ValueError('schema-7 center-trace head K identity mismatch')
	if _required_mapping(config, 'identity').get('model_tag') != CENTER_TRACE_MODEL_TAG:
		raise ValueError('schema-7 center-trace model tag mismatch')
	loss = _required_mapping(config, 'loss')
	for key, expected in (
		('prototype_weight', 1.0),
		('usage_weight', 0.005),
		('consistency_weight', 0.0),
		('consistency_beta', 0.1),
		('distillation_weight', 0.2),
	):
		if loss.get(key) != expected or scientific.get(key) != expected:
			raise ValueError(f'schema-7 center-trace {key} identity mismatch')


def _validate_multi_head_target_manifest_identity(
	*,
	identity: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
) -> Mapping[str, object]:
	"""Bind the checkpoint manifest to its scientific identity."""
	target = identity.get('target_manifest')
	if not isinstance(target, Mapping):
		raise TypeError('checkpoint target_manifest must be a mapping')
	path = _required_string(target.get('path'), 'checkpoint target_manifest.path')
	manifest_sha256 = _required_sha256(
		target.get('sha256'), 'checkpoint target_manifest.sha256'
	)
	if manifest_sha256 != _file_sha256(Path(path)):
		raise ValueError('checkpoint target manifest SHA-256 mismatch')
	if identity.get('per_head_targets') != _manifest_per_head_target_hashes(Path(path)):
		raise ValueError(
			'checkpoint per-head target hashes do not match target manifest'
		)
	scientific = _required_mapping(
		_required_mapping(stratigraphy_config, 'identity'), 'scientific_identity'
	)
	if manifest_sha256 != _required_sha256(
		scientific.get('target_manifest_sha256'),
		'scientific identity target_manifest_sha256',
	):
		raise ValueError(
			'checkpoint target manifest SHA-256 does not match scientific identity'
		)
	return scientific


def _validate_expected_multi_head_identity(
	identity: Mapping[str, object],
	config: Mapping[str, object],
	*,
	expected_student: torch.nn.Module | None = None,
	expected_head: torch.nn.Module | None = None,
	expected_spatial_context: torch.nn.Module | None = None,
) -> None:
	if _is_periodic_refresh_config(config):
		_validate_expected_periodic_refresh_identity(
			identity,
			config,
			expected_student=expected_student,
			expected_head=expected_head,
			expected_spatial_context=expected_spatial_context,
		)
		return
	if _is_center_trace_config(config):
		_validate_expected_center_trace_identity(
			identity,
			config,
			expected_student=expected_student,
			expected_head=expected_head,
			expected_spatial_context=expected_spatial_context,
		)
		return
	if _is_soft_multi_head_config(config):
		_validate_expected_soft_multi_head_identity(identity, config)
		return
	if _is_lateral_multi_head_config(config):
		_validate_expected_lateral_multi_head_identity(identity, config)
		return
	if _is_xy_neighbor_unanimous_multi_head_config(config):
		_validate_expected_xy_neighbor_unanimous_multi_head_identity(identity, config)
		return
	if _is_xy_neighbor_consensus_multi_head_config(config):
		_validate_expected_xy_neighbor_consensus_multi_head_identity(identity, config)
		return
	if identity.get('target_representation') not in {None, 'hard_viterbi_labels_v1'}:
		raise ValueError(
			'checkpoint multi-head identity is incompatible at target_representation'
		)
	head = _required_mapping(config, 'head')
	config_identity = _required_mapping(config, 'identity')
	scientific = _required_mapping(config_identity, 'scientific_identity')
	teacher = _required_mapping(config, 'teacher')
	student = _required_mapping(config, 'student')
	teacher_path = Path(
		_required_string(teacher.get('checkpoint'), 'teacher.checkpoint')
	)
	student_path = Path(
		student.get('init_checkpoint')
		or _required_string(teacher.get('checkpoint'), 'teacher.checkpoint')
	)
	checks = {
		'head_spec': head.get('spec'),
		'head_ks': list(_head_ks(head.get('ks'))),
		'per_head_targets': scientific.get('target_head_hashes'),
		'consistency_policy': scientific.get('consistency_policy'),
		'consistency_weight': scientific.get('consistency_weight'),
		'consistency_beta': scientific.get('consistency_beta'),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'model_tag': config_identity.get('model_tag'),
		'output_root': _required_mapping(config, 'paths').get('output_root'),
		'teacher_checkpoint_sha256': _file_sha256(teacher_path),
		'student_init_checkpoint_sha256': _file_sha256(student_path),
	}
	for key, expected in checks.items():
		if identity.get(key) != expected:
			raise ValueError(f'checkpoint multi-head identity is incompatible at {key}')


def _validate_expected_periodic_refresh_identity(  # noqa: C901
	identity: Mapping[str, object],
	config: Mapping[str, object],
	*,
	expected_student: torch.nn.Module | None,
	expected_head: torch.nn.Module | None,
	expected_spatial_context: torch.nn.Module | None,
) -> None:
	"""Compare schema-8 identity against the current periodic config."""
	scientific = _required_mapping(
		_required_mapping(config, 'identity'), 'scientific_identity'
	)
	teacher = _required_mapping(config, 'teacher')
	student = _required_mapping(config, 'student')
	pseudo_targets = _required_mapping(config, 'pseudo_targets')
	manifest_path = Path(
		_required_string(pseudo_targets.get('manifest'), 'pseudo_targets.manifest')
	)
	checks = {
		'schema_version': 8,
		'head_spec': 'multi_resolution_ordered_prototypes_v1',
		'head_ks': [6, 8, 10],
		'target_representation': PERIODIC_REFRESH_TARGET_REPRESENTATION,
		'initial_hard_target_manifest_sha256': scientific.get(
			'initial_hard_target_manifest_sha256'
		),
		'initial_per_head_targets': scientific.get('target_head_hashes'),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'model_tag': _required_mapping(config, 'identity').get('model_tag'),
		'output_root': _required_mapping(config, 'paths').get('output_root'),
		'teacher_checkpoint_sha256': _file_sha256(
			Path(_required_string(teacher.get('checkpoint'), 'teacher.checkpoint'))
		),
		'student_init_checkpoint_sha256': _file_sha256(
			Path(
				student.get('init_checkpoint')
				or _required_string(teacher.get('checkpoint'), 'teacher.checkpoint')
			)
		),
	}
	if checks['initial_hard_target_manifest_sha256'] != _file_sha256(manifest_path):
		raise ValueError('schema-8 expected target manifest hash does not match file')
	for key, expected in checks.items():
		if identity.get(key) != expected:
			raise ValueError(f'schema-8 checkpoint identity is incompatible at {key}')
	expected_initial_manifest = {
		'path': str(manifest_path),
		'sha256': checks['initial_hard_target_manifest_sha256'],
	}
	if identity.get('initial_hard_target_manifest') != expected_initial_manifest:
		raise ValueError(
		'schema-8 checkpoint identity is incompatible at '
		'initial_hard_target_manifest'
	)
	for key in _PERIODIC_REFRESH_CHECKPOINT_IDENTITY_FIELDS:
		if key in {
			'schema_version',
			'head_spec',
			'head_ks',
			'initial_hard_target_manifest_sha256',
			'initial_hard_target_manifest',
			'initial_per_head_targets',
			'optimizer_group_identity',
			'student_state_sha256',
			'stratigraphy_state_sha256',
			'spatial_context_state_sha256',
			'initial_spatial_context_state_sha256',
			'initial_student_state_sha256',
			'initial_head_state_sha256',
			'teacher_checkpoint_sha256',
			'student_init_checkpoint_sha256',
			'target_refresh_state_sha256',
			'model_tag',
			'output_root',
			'scientific_identity_sha256',
		}:
			continue
		if identity.get(key) != scientific.get(key):
			raise ValueError(
				f'schema-8 checkpoint identity is incompatible at {key}'
			)
	if expected_student is not None and identity.get(
		'initial_student_state_sha256'
	) != _state_sha256(expected_student.state_dict()):
		raise ValueError('schema-8 initial student state hash is incompatible')
	if expected_head is not None and identity.get(
		'initial_head_state_sha256'
	) != _state_sha256(expected_head.state_dict()):
		raise ValueError('schema-8 initial head state hash is incompatible')
	if expected_spatial_context is not None and identity.get(
		'initial_spatial_context_state_sha256'
	) != _state_sha256(expected_spatial_context.state_dict()):
		raise ValueError('schema-8 initial spatial context state hash is incompatible')


def _validate_expected_center_trace_identity(
	identity: Mapping[str, object],
	config: Mapping[str, object],
	*,
	expected_student: torch.nn.Module | None,
	expected_head: torch.nn.Module | None,
	expected_spatial_context: torch.nn.Module | None,
) -> None:
	"""Compare schema-7 identity against the current resolved center config."""
	scientific = _required_mapping(
		_required_mapping(config, 'identity'), 'scientific_identity'
	)
	teacher = _required_mapping(config, 'teacher')
	student = _required_mapping(config, 'student')
	pseudo_targets = _required_mapping(config, 'pseudo_targets')
	checks = {
		'schema_version': 7,
		'head_spec': _CENTER_TRACE_HEAD_SPEC,
		'head_ks': list(_CENTER_TRACE_HEAD_KS),
		'target_representation': _CENTER_TRACE_TARGET_REPRESENTATION,
		'target_manifest_sha256': scientific.get('target_manifest_sha256'),
		'per_head_targets': scientific.get('target_head_hashes'),
		'objective_semantics': scientific.get('objective_semantics'),
		'mask_semantics': scientific.get('mask_semantics'),
		'column_fraction': scientific.get('column_fraction'),
		'selection_policy': scientific.get('selection_policy'),
		'replacement': scientific.get('replacement'),
		'replacement_initialization': scientific.get('replacement_initialization'),
		'rng_policy': scientific.get('rng_policy'),
		'masked_prototype_weight': scientific.get('masked_prototype_weight'),
		'visible_prototype_weight': scientific.get('visible_prototype_weight'),
		'distillation_scope': scientific.get('distillation_scope'),
		'supervised_loss': scientific.get('supervised_loss'),
		'consistency_policy': scientific.get('consistency_policy'),
		'prototype_weight': scientific.get('prototype_weight'),
		'usage_weight': scientific.get('usage_weight'),
		'consistency_weight': scientific.get('consistency_weight'),
		'consistency_beta': scientific.get('consistency_beta'),
		'distillation_weight': scientific.get('distillation_weight'),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'model_tag': _required_mapping(config, 'identity').get('model_tag'),
		'output_root': _required_mapping(config, 'paths').get('output_root'),
		'teacher_checkpoint_sha256': _file_sha256(
			Path(_required_string(teacher.get('checkpoint'), 'teacher.checkpoint'))
		),
		'student_init_checkpoint_sha256': _file_sha256(
			Path(
				student.get('init_checkpoint')
				or _required_string(teacher.get('checkpoint'), 'teacher.checkpoint')
			)
		),
	}
	manifest_path = Path(
		_required_string(pseudo_targets.get('manifest'), 'pseudo_targets.manifest')
	)
	if checks['target_manifest_sha256'] != _file_sha256(manifest_path):
		raise ValueError('schema-7 expected target manifest hash does not match file')
	for key, expected in checks.items():
		if identity.get(key) != expected:
			raise ValueError(
				f'schema-7 checkpoint identity is incompatible at {key}'
			)
	if expected_student is not None and identity.get(
		'initial_student_state_sha256'
	) != _state_sha256(
			expected_student.state_dict()
		):
			raise ValueError('schema-7 initial student state hash is incompatible')
	if expected_head is not None and identity.get(
		'initial_head_state_sha256'
	) != _state_sha256(
			expected_head.state_dict()
		):
			raise ValueError('schema-7 initial head state hash is incompatible')
	if expected_spatial_context is not None:
		initial_spatial_hash = _state_sha256(expected_spatial_context.state_dict())
		if identity.get('initial_spatial_context_state_sha256') != initial_spatial_hash:
			raise ValueError(
				'schema-7 initial spatial_context state hash is incompatible'
			)


def _validate_expected_optimizer_group_identity(  # noqa: PLR0913
	*,
	identity: Mapping[str, object],
	optimizer_state_dict: object,
	expected_optimizer: torch.optim.Optimizer,
	expected_student: torch.nn.Module | None,
	expected_head: torch.nn.Module | None,
	expected_spatial_context: torch.nn.Module | None = None,
) -> None:
	if expected_student is None or expected_head is None:
		expected = _optimizer_group_summary(expected_optimizer.state_dict())
	else:
		expected = _optimizer_group_identity(
			expected_optimizer,
			parameter_names=_stratigraphy_parameter_names(
				expected_student,
				expected_head,
				spatial_context=expected_spatial_context,
			),
		)
	checkpoint_identity = identity.get('optimizer_group_identity')
	if expected_student is not None and expected_head is not None:
		if checkpoint_identity != expected:
			raise ValueError(
				'checkpoint optimizer group identity is incompatible with current '
				'optimizer'
			)
	elif not _optimizer_group_counts_match(checkpoint_identity, expected):
		raise ValueError(
			'checkpoint optimizer group identity is incompatible with current optimizer'
		)
	if not _optimizer_state_group_identity_matches(
		optimizer_state_dict, checkpoint_identity
	):
		raise ValueError(
			'checkpoint optimizer state groups do not match optimizer group identity'
		)


def _head_ks(value: object) -> tuple[int, ...]:
	if (
		not isinstance(value, list | tuple)
		or not value
		or any(isinstance(k, bool) or not isinstance(k, int) for k in value)
	):
		raise TypeError('multi-head checkpoint head.ks must be non-empty integers')
	return tuple(value)


def _manifest_per_head_target_hashes(
	manifest_path: Path,
) -> dict[str, dict[str, dict[str, str]]]:
	"""Load target references and return their manifest-bound per-head hashes."""
	manifest = load_multi_head_target_manifest(
		manifest_path,
		validate_array_semantics=False,
	)
	head_ks = _head_ks(manifest.get('head_ks'))
	heads = _required_mapping(manifest, 'heads')
	result: dict[str, dict[str, dict[str, str]]] = {}
	for k in head_ks:
		head = _required_mapping(heads, str(k))
		surveys = _required_mapping(head, 'surveys')
		result[str(k)] = {}
		for survey_id, entry in surveys.items():
			if not isinstance(entry, Mapping):
				raise TypeError(
					f'manifest heads.{k}.surveys.{survey_id} must be a mapping'
				)
			target = entry
			result[str(k)][str(survey_id)] = {
				name: _required_string(
					_required_mapping(target, name).get('sha256'),
					f'manifest heads.{k}.surveys.{survey_id}.{name}.sha256',
				)
				for name in ('labels', 'confidence', 'valid_tokens', 'metadata')
			}
	return result


def _is_soft_multi_head_config(config: Mapping[str, object]) -> bool:
	pseudo_targets = config.get('pseudo_targets')
	return (
		isinstance(pseudo_targets, Mapping)
		and pseudo_targets.get('target_representation')
		== 'ordered_path_state_posterior_v1'
	)


def _is_lateral_multi_head_config(config: Mapping[str, object]) -> bool:
	pseudo_targets = config.get('pseudo_targets')
	return (
		isinstance(pseudo_targets, Mapping)
		and pseudo_targets.get('target_representation')
		== 'lateral_mean_field_hard_labels_v1'
	)


def _is_xy_neighbor_consensus_multi_head_config(
	config: Mapping[str, object],
) -> bool:
	pseudo_targets = config.get('pseudo_targets')
	return (
		isinstance(pseudo_targets, Mapping)
		and pseudo_targets.get('target_representation')
		== _XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION
	)


def _is_xy_neighbor_unanimous_multi_head_config(
	config: Mapping[str, object],
) -> bool:
	pseudo_targets = config.get('pseudo_targets')
	return (
		isinstance(pseudo_targets, Mapping)
		and pseudo_targets.get('target_representation')
		== _XY_NEIGHBOR_UNANIMOUS_TARGET_REPRESENTATION
	)


def _posterior_per_head_hashes(
	manifest_path: Path,
) -> dict[str, dict[str, dict[str, str]]]:
	manifest = load_multi_head_state_posterior_manifest(
		manifest_path,
		validate_array_semantics=False,
	)
	heads = _required_mapping(manifest, 'heads')
	result: dict[str, dict[str, dict[str, str]]] = {}
	for k in _head_ks(manifest.get('head_ks')):
		head = _required_mapping(heads, str(k))
		surveys = _required_mapping(head, 'surveys')
		result[str(k)] = {}
		for survey_id, entry in surveys.items():
			if not isinstance(entry, Mapping):
				raise TypeError(f'posterior manifest head {k} survey must be a mapping')
			result[str(k)][str(survey_id)] = {
				name: _required_string(
					_required_mapping(entry, name).get('sha256'),
					f'posterior manifest heads.{k}.surveys.{survey_id}.{name}.sha256',
				)
				for name in ('posterior', 'valid_tokens', 'metadata')
			}
	return result


def _lateral_per_head_target_hashes(
	manifest_path: Path,
) -> dict[str, dict[str, dict[str, str]]]:
	manifest = load_multi_head_lateral_target_manifest(
		manifest_path,
		validate_array_semantics=False,
	)
	return _per_head_hard_target_hashes(manifest)


def _xy_neighbor_consensus_per_head_target_hashes(
	manifest_path: Path,
) -> dict[str, dict[str, dict[str, str]]]:
	return _xy_neighbor_consensus_target_provenance(manifest_path)[0]


def _xy_neighbor_consensus_target_provenance(
	manifest_path: Path,
) -> tuple[
	dict[str, dict[str, dict[str, str]]],
	str,
	Mapping[str, object],
]:
	"""Read the source-hard-only identity from a strict consensus manifest."""
	manifest = load_multi_head_xy_neighbor_consensus_target_manifest(
		manifest_path,
		validate_array_semantics=False,
	)
	source_hard = _required_mapping(manifest, 'source_hard_manifest')
	source_hard_manifest_sha256 = _required_sha256(
		source_hard.get('sha256'),
		'XY neighbor consensus source_hard_manifest.sha256',
	)
	smoothing = _required_mapping(manifest, 'smoothing')
	return (
		_per_head_hard_target_hashes(manifest),
		source_hard_manifest_sha256,
		_to_plain_value(smoothing),
	)


def _xy_neighbor_unanimous_per_head_target_hashes(
	manifest_path: Path,
) -> dict[str, dict[str, dict[str, str]]]:
	return _xy_neighbor_unanimous_target_provenance(manifest_path)[0]


def _xy_neighbor_unanimous_target_provenance(
	manifest_path: Path,
) -> tuple[
	dict[str, dict[str, dict[str, str]]],
	str,
	Mapping[str, object],
]:
	"""Read source-hard-only identity from a strict unanimous manifest."""
	manifest = load_multi_head_xy_neighbor_unanimous_target_manifest(
		manifest_path,
		validate_array_semantics=False,
	)
	source_hard = _required_mapping(manifest, 'source_hard_manifest')
	source_hard_manifest_sha256 = _required_sha256(
		source_hard.get('sha256'),
		'XY neighbor unanimous source_hard_manifest.sha256',
	)
	smoothing = _required_mapping(manifest, 'smoothing')
	return (
		_per_head_hard_target_hashes(manifest),
		source_hard_manifest_sha256,
		_to_plain_value(smoothing),
	)


def _per_head_hard_target_hashes(
	manifest: Mapping[str, object],
) -> dict[str, dict[str, dict[str, str]]]:
	heads = _required_mapping(manifest, 'heads')
	result: dict[str, dict[str, dict[str, str]]] = {}
	for k in _head_ks(manifest.get('head_ks')):
		head = _required_mapping(heads, str(k))
		surveys = _required_mapping(head, 'surveys')
		result[str(k)] = {}
		for survey_id, entry in surveys.items():
			if not isinstance(entry, Mapping):
				raise TypeError(f'lateral manifest head {k} survey must be a mapping')
			target = entry
			result[str(k)][str(survey_id)] = {
				name: _required_string(
					_required_mapping(target, name).get('sha256'),
					f'lateral manifest heads.{k}.surveys.{survey_id}.{name}.sha256',
				)
				for name in ('labels', 'confidence', 'valid_tokens', 'metadata')
			}
	return result


def _validate_soft_multi_head_identity(  # noqa: C901
	*,
	identity: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, object],
) -> None:
	"""Validate the schema-v3 soft identity without accepting hard artifacts."""
	if identity.get('schema_version') != 3:
		raise ValueError('soft posterior checkpoint requires schema_version 3')
	_validate_multi_head_config_and_state(
		stratigraphy_config,
		{
			str(key): value
			for key, value in stratigraphy_state_dict.items()
			if isinstance(value, torch.Tensor)
		},
	)
	head = _required_mapping(stratigraphy_config, 'head')
	config_identity = _required_mapping(stratigraphy_config, 'identity')
	paths = _required_mapping(stratigraphy_config, 'paths')
	scientific = _required_mapping(config_identity, 'scientific_identity')
	_reject_xy_neighbor_consensus_fields_from_legacy_identity(identity, scientific)
	if identity.get('head_spec') != head.get('spec') or identity.get('head_ks') != list(
		_head_ks(head.get('ks'))
	):
		raise ValueError(
			'soft checkpoint head identity does not match stratigraphy config'
		)
	if identity.get('model_tag') != config_identity.get('model_tag') or identity.get(
		'output_root'
	) != paths.get('output_root'):
		raise ValueError(
			'soft checkpoint model identity does not match stratigraphy config'
		)
	if identity.get('stratigraphy_state_sha256') != _state_sha256(
		stratigraphy_state_dict
	):
		raise ValueError('checkpoint stratigraphy state SHA-256 mismatch')
	posterior = identity.get('posterior_manifest')
	if not isinstance(posterior, Mapping):
		raise TypeError('soft checkpoint posterior_manifest must be a mapping')
	path = Path(
		_required_string(posterior.get('path'), 'checkpoint posterior_manifest.path')
	)
	sha256 = _required_sha256(
		posterior.get('sha256'), 'checkpoint posterior_manifest.sha256'
	)
	if sha256 != _file_sha256(path):
		raise ValueError('checkpoint posterior manifest SHA-256 mismatch')
	if identity.get('per_head_posteriors') != _posterior_per_head_hashes(path):
		raise ValueError('checkpoint posterior hashes do not match posterior manifest')
	for key in (
		'target_representation',
		'posterior_semantics',
		'posterior_cost_temperature',
		'posterior_manifest_sha256',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
	):
		if identity.get(key) != scientific.get(key):
			raise ValueError(f'checkpoint {key} does not match scientific identity')
	if identity.get('per_head_posteriors') != scientific.get('posterior_head_hashes'):
		raise ValueError('checkpoint posterior hashes do not match scientific identity')
	if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
		scientific
	):
		raise ValueError('checkpoint scientific identity SHA-256 mismatch')
	for key in (
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'teacher_checkpoint_sha256',
		'student_init_checkpoint_sha256',
	):
		_required_sha256(identity.get(key), f'checkpoint {key}')


def _validate_expected_soft_multi_head_identity(
	identity: Mapping[str, object], config: Mapping[str, object]
) -> None:
	scientific = _required_mapping(
		_required_mapping(config, 'identity'), 'scientific_identity'
	)
	teacher = _required_mapping(config, 'teacher')
	student = _required_mapping(config, 'student')
	checks = {
		'target_representation': scientific.get('target_representation'),
		'posterior_semantics': scientific.get('posterior_semantics'),
		'posterior_cost_temperature': scientific.get('posterior_cost_temperature'),
		'posterior_manifest_sha256': scientific.get('posterior_manifest_sha256'),
		'per_head_posteriors': scientific.get('posterior_head_hashes'),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'model_tag': _required_mapping(config, 'identity').get('model_tag'),
		'output_root': _required_mapping(config, 'paths').get('output_root'),
		'teacher_checkpoint_sha256': _file_sha256(
			Path(_required_string(teacher.get('checkpoint'), 'teacher.checkpoint'))
		),
		'student_init_checkpoint_sha256': _file_sha256(
			Path(
				student.get('init_checkpoint')
				or _required_string(teacher.get('checkpoint'), 'teacher.checkpoint')
			)
		),
	}
	for key, expected in checks.items():
		if identity.get(key) != expected:
			raise ValueError(
				f'checkpoint soft multi-head identity is incompatible at {key}'
			)


def _validate_lateral_multi_head_identity(  # noqa: C901
	*,
	identity: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, object],
) -> None:
	"""Validate schema-v4 M5-LS identity before any model state is restored."""
	if identity.get('schema_version') != 4:
		raise ValueError('lateral hard checkpoint requires schema_version 4')
	_validate_multi_head_config_and_state(
		stratigraphy_config,
		{
			str(key): value
			for key, value in stratigraphy_state_dict.items()
			if isinstance(value, torch.Tensor)
		},
	)
	head = _required_mapping(stratigraphy_config, 'head')
	config_identity = _required_mapping(stratigraphy_config, 'identity')
	paths = _required_mapping(stratigraphy_config, 'paths')
	scientific = _required_mapping(config_identity, 'scientific_identity')
	_reject_xy_neighbor_consensus_fields_from_legacy_identity(identity, scientific)
	if identity.get('head_spec') != head.get('spec') or identity.get('head_ks') != list(
		_head_ks(head.get('ks'))
	):
		raise ValueError('lateral checkpoint head identity does not match config')
	if identity.get('model_tag') != config_identity.get('model_tag') or identity.get(
		'output_root'
	) != paths.get('output_root'):
		raise ValueError('lateral checkpoint model identity does not match config')
	if identity.get('stratigraphy_state_sha256') != _state_sha256(
		stratigraphy_state_dict
	):
		raise ValueError('checkpoint stratigraphy state SHA-256 mismatch')
	target = identity.get('lateral_target_manifest')
	if not isinstance(target, Mapping):
		raise TypeError('lateral checkpoint target manifest must be a mapping')
	path = Path(
		_required_string(target.get('path'), 'checkpoint lateral_target_manifest.path')
	)
	sha256 = _required_sha256(
		target.get('sha256'), 'checkpoint lateral_target_manifest.sha256'
	)
	if sha256 != _file_sha256(path):
		raise ValueError('checkpoint lateral target manifest SHA-256 mismatch')
	if identity.get('per_head_lateral_targets') != _lateral_per_head_target_hashes(
		path
	):
		raise ValueError('checkpoint lateral target hashes do not match manifest')
	for key in (
		'target_representation',
		'target_semantics',
		'lateral_target_manifest_sha256',
		'source_hard_manifest_sha256',
		'source_posterior_manifest_sha256',
		'lateral_smoothing',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
	):
		if identity.get(key) != scientific.get(key):
			raise ValueError(f'checkpoint {key} does not match scientific identity')
	if identity.get('per_head_lateral_targets') != scientific.get(
		'lateral_target_head_hashes'
	):
		raise ValueError('checkpoint lateral hashes do not match scientific identity')
	if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
		scientific
	):
		raise ValueError('checkpoint scientific identity SHA-256 mismatch')
	for key in (
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'teacher_checkpoint_sha256',
		'student_init_checkpoint_sha256',
	):
		_required_sha256(identity.get(key), f'checkpoint {key}')


def _validate_expected_lateral_multi_head_identity(
	identity: Mapping[str, object], config: Mapping[str, object]
) -> None:
	scientific = _required_mapping(
		_required_mapping(config, 'identity'), 'scientific_identity'
	)
	teacher = _required_mapping(config, 'teacher')
	student = _required_mapping(config, 'student')
	checks = {
		'target_representation': scientific.get('target_representation'),
		'target_semantics': scientific.get('target_semantics'),
		'lateral_target_manifest_sha256': scientific.get(
			'lateral_target_manifest_sha256'
		),
		'per_head_lateral_targets': scientific.get('lateral_target_head_hashes'),
		'source_hard_manifest_sha256': scientific.get('source_hard_manifest_sha256'),
		'source_posterior_manifest_sha256': scientific.get(
			'source_posterior_manifest_sha256'
		),
		'lateral_smoothing': scientific.get('lateral_smoothing'),
		'consistency_policy': scientific.get('consistency_policy'),
		'consistency_weight': scientific.get('consistency_weight'),
		'consistency_beta': scientific.get('consistency_beta'),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'model_tag': _required_mapping(config, 'identity').get('model_tag'),
		'output_root': _required_mapping(config, 'paths').get('output_root'),
		'teacher_checkpoint_sha256': _file_sha256(
			Path(_required_string(teacher.get('checkpoint'), 'teacher.checkpoint'))
		),
		'student_init_checkpoint_sha256': _file_sha256(
			Path(
				student.get('init_checkpoint')
				or _required_string(teacher.get('checkpoint'), 'teacher.checkpoint')
			)
		),
	}
	for key, expected in checks.items():
		if identity.get(key) != expected:
			raise ValueError(
				f'checkpoint lateral multi-head identity is incompatible at {key}'
			)


def _validate_xy_neighbor_consensus_multi_head_identity(  # noqa: C901, PLR0912
	*,
	identity: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, object],
) -> None:
	"""Validate schema-v5 XY-consensus identity before restoring state."""
	if identity.get('schema_version') != 5:
		raise ValueError('XY neighbor consensus checkpoint requires schema_version 5')
	_validate_multi_head_config_and_state(
		stratigraphy_config,
		{
			str(key): value
			for key, value in stratigraphy_state_dict.items()
			if isinstance(value, torch.Tensor)
		},
	)
	head = _required_mapping(stratigraphy_config, 'head')
	config_identity = _required_mapping(stratigraphy_config, 'identity')
	paths = _required_mapping(stratigraphy_config, 'paths')
	scientific = _required_mapping(config_identity, 'scientific_identity')
	_reject_xy_neighbor_consensus_legacy_fields(identity, scientific)
	if identity.get('head_spec') != head.get('spec') or identity.get('head_ks') != list(
		_head_ks(head.get('ks'))
	):
		raise ValueError(
			'XY neighbor consensus checkpoint head identity does not match config'
		)
	if identity.get('model_tag') != config_identity.get('model_tag') or identity.get(
		'output_root'
	) != paths.get('output_root'):
		raise ValueError(
			'XY neighbor consensus checkpoint model identity does not match config'
		)
	if identity.get('stratigraphy_state_sha256') != _state_sha256(
		stratigraphy_state_dict
	):
		raise ValueError('checkpoint stratigraphy state SHA-256 mismatch')
	target = identity.get('xy_neighbor_consensus_target_manifest')
	if not isinstance(target, Mapping):
		raise TypeError(
			'XY neighbor consensus checkpoint target manifest must be a mapping'
		)
	path = Path(
		_required_string(
			target.get('path'),
			'checkpoint xy_neighbor_consensus_target_manifest.path',
		)
	)
	sha256 = _required_sha256(
		target.get('sha256'),
		'checkpoint xy_neighbor_consensus_target_manifest.sha256',
	)
	if sha256 != _file_sha256(path):
		raise ValueError('checkpoint XY neighbor consensus manifest SHA-256 mismatch')
	(
		per_head_targets,
		source_hard_manifest_sha256,
		smoothing,
	) = _xy_neighbor_consensus_target_provenance(path)
	if identity.get('per_head_xy_neighbor_consensus_targets') != per_head_targets:
		raise ValueError(
			'checkpoint XY neighbor consensus target hashes do not match manifest'
		)
	for key, expected in (
		('target_representation', scientific.get('target_representation')),
		('target_semantics', scientific.get('target_semantics')),
		(
			'xy_neighbor_consensus_target_manifest_sha256',
			scientific.get('xy_neighbor_consensus_target_manifest_sha256'),
		),
		('source_hard_manifest_sha256', scientific.get('source_hard_manifest_sha256')),
		(
			'xy_neighbor_consensus_smoothing',
			_to_plain_value(scientific.get('xy_neighbor_consensus_smoothing')),
		),
		('consistency_policy', scientific.get('consistency_policy')),
		('consistency_weight', scientific.get('consistency_weight')),
		('consistency_beta', scientific.get('consistency_beta')),
	):
		if identity.get(key) != expected:
			raise ValueError(f'checkpoint {key} does not match scientific identity')
	if identity.get('xy_neighbor_consensus_target_manifest_sha256') != sha256:
		raise ValueError(
			'checkpoint XY neighbor consensus manifest hash does not match target '
			'manifest'
		)
	if identity.get('per_head_xy_neighbor_consensus_targets') != scientific.get(
		'xy_neighbor_consensus_target_head_hashes'
	):
		raise ValueError(
			'checkpoint XY neighbor consensus hashes do not match scientific identity'
		)
	if identity.get('source_hard_manifest_sha256') != source_hard_manifest_sha256:
		raise ValueError(
			'checkpoint XY neighbor consensus source hard manifest does not match '
			'target manifest'
		)
	if identity.get('xy_neighbor_consensus_smoothing') != smoothing:
		raise ValueError(
			'checkpoint XY neighbor consensus smoothing does not match target manifest'
		)
	if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
		scientific
	):
		raise ValueError('checkpoint scientific identity SHA-256 mismatch')
	for key in (
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'teacher_checkpoint_sha256',
		'student_init_checkpoint_sha256',
	):
		_required_sha256(identity.get(key), f'checkpoint {key}')


def _validate_expected_xy_neighbor_consensus_multi_head_identity(
	identity: Mapping[str, object], config: Mapping[str, object]
) -> None:
	"""Reject cross-resume and stale source-hard consensus provenance."""
	scientific = _required_mapping(
		_required_mapping(config, 'identity'), 'scientific_identity'
	)
	teacher = _required_mapping(config, 'teacher')
	student = _required_mapping(config, 'student')
	checks = {
		'target_representation': scientific.get('target_representation'),
		'target_semantics': scientific.get('target_semantics'),
		'xy_neighbor_consensus_target_manifest_sha256': scientific.get(
			'xy_neighbor_consensus_target_manifest_sha256'
		),
		'per_head_xy_neighbor_consensus_targets': scientific.get(
			'xy_neighbor_consensus_target_head_hashes'
		),
		'source_hard_manifest_sha256': scientific.get('source_hard_manifest_sha256'),
		'xy_neighbor_consensus_smoothing': _to_plain_value(
			scientific.get('xy_neighbor_consensus_smoothing')
		),
		'consistency_policy': scientific.get('consistency_policy'),
		'consistency_weight': scientific.get('consistency_weight'),
		'consistency_beta': scientific.get('consistency_beta'),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'model_tag': _required_mapping(config, 'identity').get('model_tag'),
		'output_root': _required_mapping(config, 'paths').get('output_root'),
		'teacher_checkpoint_sha256': _file_sha256(
			Path(_required_string(teacher.get('checkpoint'), 'teacher.checkpoint'))
		),
		'student_init_checkpoint_sha256': _file_sha256(
			Path(
				student.get('init_checkpoint')
				or _required_string(teacher.get('checkpoint'), 'teacher.checkpoint')
			)
		),
	}
	for key, expected in checks.items():
		if identity.get(key) != expected:
			raise ValueError(
				'checkpoint XY neighbor consensus multi-head identity is '
				f'incompatible at {key}'
			)


def _validate_xy_neighbor_unanimous_multi_head_identity(  # noqa: C901, PLR0912
	*,
	identity: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, object],
) -> None:
	"""Validate schema-v6 XY-unanimous identity before restoring state."""
	if identity.get('schema_version') != 6:
		raise ValueError('XY neighbor unanimous checkpoint requires schema_version 6')
	_validate_multi_head_config_and_state(
		stratigraphy_config,
		{
			str(key): value
			for key, value in stratigraphy_state_dict.items()
			if isinstance(value, torch.Tensor)
		},
	)
	head = _required_mapping(stratigraphy_config, 'head')
	config_identity = _required_mapping(stratigraphy_config, 'identity')
	paths = _required_mapping(stratigraphy_config, 'paths')
	scientific = _required_mapping(config_identity, 'scientific_identity')
	_reject_xy_neighbor_unanimous_legacy_fields(identity, scientific)
	if (
		identity.get('target_representation')
		!= _XY_NEIGHBOR_UNANIMOUS_TARGET_REPRESENTATION
		or identity.get('target_semantics') != _XY_NEIGHBOR_UNANIMOUS_TARGET_SEMANTICS
		or identity.get('consistency_policy')
		!= _XY_NEIGHBOR_UNANIMOUS_CONSISTENCY_POLICY
		or identity.get('model_tag')
		!= 'strat_hmm_pretext_mh_k6810_xyunanim1_nocons_topblock1_distill_v1'
	):
		raise ValueError('checkpoint does not carry the fixed XY unanimous identity')
	for key, expected in (
		('experiment_role', _XY_NEIGHBOR_UNANIMOUS_EXPERIMENT_ROLE),
		('variant', 'xyunanim1_nocons'),
		('head_spec', 'multi_resolution_ordered_prototypes_v1'),
		('head_ks', [6, 8, 10]),
		('target_representation', _XY_NEIGHBOR_UNANIMOUS_TARGET_REPRESENTATION),
		('target_semantics', _XY_NEIGHBOR_UNANIMOUS_TARGET_SEMANTICS),
		('supervised_loss', 'structured_hmm_hard_categorical_v1'),
		('consistency_policy', _XY_NEIGHBOR_UNANIMOUS_CONSISTENCY_POLICY),
		('prototype_weight', 1.0),
		('usage_weight', 0.005),
		('consistency_weight', 0.0),
		('consistency_beta', 0.1),
		('distillation_weight', 0.2),
		('student_unfreeze_top_blocks', 1),
	):
		if scientific.get(key) != expected:
			raise ValueError(
				'checkpoint scientific identity does not carry fixed XY unanimous '
				f'{key}'
			)
	if identity.get('head_spec') != head.get('spec') or identity.get('head_ks') != list(
		_head_ks(head.get('ks'))
	):
		raise ValueError(
			'XY neighbor unanimous checkpoint head identity does not match config'
		)
	if identity.get('model_tag') != config_identity.get('model_tag') or identity.get(
		'output_root'
	) != paths.get('output_root'):
		raise ValueError(
			'XY neighbor unanimous checkpoint model identity does not match config'
		)
	if identity.get('stratigraphy_state_sha256') != _state_sha256(
		stratigraphy_state_dict
	):
		raise ValueError('checkpoint stratigraphy state SHA-256 mismatch')
	target = identity.get('xy_neighbor_unanimous_target_manifest')
	if not isinstance(target, Mapping):
		raise TypeError(
			'XY neighbor unanimous checkpoint target manifest must be a mapping'
		)
	path = Path(
		_required_string(
			target.get('path'),
			'checkpoint xy_neighbor_unanimous_target_manifest.path',
		)
	)
	sha256 = _required_sha256(
		target.get('sha256'),
		'checkpoint xy_neighbor_unanimous_target_manifest.sha256',
	)
	if sha256 != _file_sha256(path):
		raise ValueError('checkpoint XY neighbor unanimous manifest SHA-256 mismatch')
	(
		per_head_targets,
		source_hard_manifest_sha256,
		smoothing,
	) = _xy_neighbor_unanimous_target_provenance(path)
	if identity.get('per_head_xy_neighbor_unanimous_targets') != per_head_targets:
		raise ValueError(
			'checkpoint XY neighbor unanimous target hashes do not match manifest'
		)
	for key, expected in (
		('target_representation', scientific.get('target_representation')),
		('target_semantics', scientific.get('target_semantics')),
		(
			'xy_neighbor_unanimous_target_manifest_sha256',
			scientific.get('xy_neighbor_unanimous_target_manifest_sha256'),
		),
		('source_hard_manifest_sha256', scientific.get('source_hard_manifest_sha256')),
		(
			'xy_neighbor_unanimous_smoothing',
			_to_plain_value(scientific.get('xy_neighbor_unanimous_smoothing')),
		),
		('consistency_policy', scientific.get('consistency_policy')),
		('consistency_weight', scientific.get('consistency_weight')),
		('consistency_beta', scientific.get('consistency_beta')),
	):
		if identity.get(key) != expected:
			raise ValueError(f'checkpoint {key} does not match scientific identity')
	if identity.get('xy_neighbor_unanimous_target_manifest_sha256') != sha256:
		raise ValueError(
			'checkpoint XY neighbor unanimous manifest hash does not match target '
			'manifest'
		)
	if identity.get('per_head_xy_neighbor_unanimous_targets') != scientific.get(
		'xy_neighbor_unanimous_target_head_hashes'
	):
		raise ValueError(
			'checkpoint XY neighbor unanimous hashes do not match scientific identity'
		)
	if identity.get('source_hard_manifest_sha256') != source_hard_manifest_sha256:
		raise ValueError(
			'checkpoint XY neighbor unanimous source hard manifest does not match '
			'target manifest'
		)
	if identity.get('xy_neighbor_unanimous_smoothing') != smoothing:
		raise ValueError(
			'checkpoint XY neighbor unanimous smoothing does not match target manifest'
		)
	if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
		scientific
	):
		raise ValueError('checkpoint scientific identity SHA-256 mismatch')
	for key in (
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'teacher_checkpoint_sha256',
		'student_init_checkpoint_sha256',
	):
		_required_sha256(identity.get(key), f'checkpoint {key}')


def _validate_xy_neighbor_unanimous_control_identity(  # noqa: C901
	*,
	identity: Mapping[str, object],
	control_identity: object,
) -> None:
	"""Bind schema-6 provenance hashes to the saved pre-optimization control."""

	def mapping(value: object, label: str) -> Mapping[str, object]:
		if not isinstance(value, Mapping):
			raise TypeError(f'{label} must be a mapping')
		return value

	control = mapping(control_identity, 'checkpoint control_identity')
	if control.get('schema_version') != 2:
		raise ValueError('XY neighbor unanimous checkpoint control schema is invalid')
	if control.get('model_tag') != identity.get('model_tag'):
		raise ValueError('XY neighbor unanimous control model tag differs')
	scientific = mapping(
		control.get('scientific_identity'),
		'checkpoint control scientific identity',
	)
	if scientific_identity_sha256(scientific) != identity.get(
		'scientific_identity_sha256'
	):
		raise ValueError('XY neighbor unanimous control scientific identity differs')
	inputs = mapping(
		control.get('input_identities'), 'checkpoint control input identities'
	)
	target_input = mapping(
		inputs.get('target_manifest'), 'checkpoint control target manifest'
	)
	target_identity = mapping(
		identity.get('xy_neighbor_unanimous_target_manifest'),
		'checkpoint unanimous target manifest',
	)
	if Path(
		_required_string(
			target_input.get('path'), 'checkpoint control target manifest.path'
		)
	).resolve() != Path(
		_required_string(
			target_identity.get('path'), 'checkpoint unanimous target manifest.path'
		)
	).resolve() or _required_sha256(
		target_input.get('sha256'), 'checkpoint control target manifest.sha256'
	) != _required_sha256(
		target_identity.get('sha256'),
		'checkpoint unanimous target manifest.sha256',
	):
		raise ValueError('XY neighbor unanimous control target manifest differs')
	for input_name, identity_name in (
		('teacher_checkpoint', 'teacher_checkpoint_sha256'),
		('student_init_checkpoint', 'student_init_checkpoint_sha256'),
	):
		entry = mapping(inputs.get(input_name), f'checkpoint control {input_name}')
		if _required_sha256(
			entry.get('sha256'), f'checkpoint control {input_name}.sha256'
		) != _required_sha256(
			identity.get(identity_name), f'checkpoint {identity_name}'
		):
			raise ValueError(
				f'XY neighbor unanimous control {input_name} provenance differs'
			)
	initial = mapping(
		control.get('initial_state_sha256'), 'checkpoint control initial state'
	)
	for control_name, identity_name in (
		('student', 'initial_student_state_sha256'),
		('head', 'initial_head_state_sha256'),
	):
		if _required_sha256(
			initial.get(control_name),
			f'checkpoint control initial state.{control_name}',
		) != _required_sha256(
			identity.get(identity_name), f'checkpoint {identity_name}'
		):
			raise ValueError(
				f'XY neighbor unanimous control initial {control_name} hash differs'
			)


def _validate_expected_xy_neighbor_unanimous_multi_head_identity(
	identity: Mapping[str, object], config: Mapping[str, object]
) -> None:
	"""Reject cross-resume and stale source-hard unanimous provenance."""
	scientific = _required_mapping(
		_required_mapping(config, 'identity'), 'scientific_identity'
	)
	teacher = _required_mapping(config, 'teacher')
	student = _required_mapping(config, 'student')
	checks = {
		'target_representation': scientific.get('target_representation'),
		'target_semantics': scientific.get('target_semantics'),
		'xy_neighbor_unanimous_target_manifest_sha256': scientific.get(
			'xy_neighbor_unanimous_target_manifest_sha256'
		),
		'per_head_xy_neighbor_unanimous_targets': scientific.get(
			'xy_neighbor_unanimous_target_head_hashes'
		),
		'source_hard_manifest_sha256': scientific.get('source_hard_manifest_sha256'),
		'xy_neighbor_unanimous_smoothing': _to_plain_value(
			scientific.get('xy_neighbor_unanimous_smoothing')
		),
		'consistency_policy': scientific.get('consistency_policy'),
		'consistency_weight': scientific.get('consistency_weight'),
		'consistency_beta': scientific.get('consistency_beta'),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'model_tag': _required_mapping(config, 'identity').get('model_tag'),
		'output_root': _required_mapping(config, 'paths').get('output_root'),
		'teacher_checkpoint_sha256': _file_sha256(
			Path(_required_string(teacher.get('checkpoint'), 'teacher.checkpoint'))
		),
		'student_init_checkpoint_sha256': _file_sha256(
			Path(
				student.get('init_checkpoint')
				or _required_string(teacher.get('checkpoint'), 'teacher.checkpoint')
			)
		),
	}
	for key, expected in checks.items():
		if identity.get(key) != expected:
			raise ValueError(
				'checkpoint XY neighbor unanimous multi-head identity is '
				f'incompatible at {key}'
			)


def _reject_xy_neighbor_consensus_legacy_fields(
	identity: Mapping[str, object], scientific: Mapping[str, object]
) -> None:
	"""Reject every non-v5 field, including posterior and M5-LS carry-over."""
	unknown_identity = sorted(
		set(identity) - _XY_NEIGHBOR_CONSENSUS_CHECKPOINT_IDENTITY_FIELDS
	)
	if unknown_identity:
		raise ValueError(
			'XY neighbor consensus checkpoint must not contain legacy or unknown '
			f'identity fields: {unknown_identity!r}'
		)
	unknown_scientific = sorted(
		set(scientific) - _XY_NEIGHBOR_CONSENSUS_SCIENTIFIC_IDENTITY_FIELDS
	)
	if unknown_scientific:
		raise ValueError(
			'XY neighbor consensus scientific identity must not contain legacy or '
			f'unknown fields: {unknown_scientific!r}'
		)


def _reject_xy_neighbor_unanimous_legacy_fields(
	identity: Mapping[str, object], scientific: Mapping[str, object]
) -> None:
	"""Reject every non-v6 field, including the 3-of-4 predecessor."""
	unknown_identity = sorted(
		set(identity) - _XY_NEIGHBOR_UNANIMOUS_CHECKPOINT_IDENTITY_FIELDS
	)
	if unknown_identity:
		raise ValueError(
			'XY neighbor unanimous checkpoint must not contain legacy or unknown '
			f'identity fields: {unknown_identity!r}'
		)
	unknown_scientific = sorted(
		set(scientific) - _XY_NEIGHBOR_UNANIMOUS_SCIENTIFIC_IDENTITY_FIELDS
	)
	if unknown_scientific:
		raise ValueError(
			'XY neighbor unanimous scientific identity must not contain legacy or '
			f'unknown fields: {unknown_scientific!r}'
		)


def _reject_xy_neighbor_consensus_fields_from_legacy_identity(
	identity: Mapping[str, object], scientific: Mapping[str, object]
) -> None:
	"""Keep schema-v2 through v4 identities disjoint from the XY successor."""
	forbidden: list[str] = []
	for prefix, values in (
		('checkpoint', identity),
		('scientific_identity', scientific),
	):
		forbidden.extend(
			f'{prefix}.{key}'
			for key in values
			if isinstance(key, str)
			and key.startswith(('xy_neighbor_consensus_', 'xy_neighbor_unanimous_'))
		)
		forbidden.extend(
			f'{prefix}.{key}'
			for key, expected in (
				('target_representation', _XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION),
				('target_semantics', _XY_NEIGHBOR_CONSENSUS_TARGET_SEMANTICS),
				('consistency_policy', _XY_NEIGHBOR_CONSENSUS_CONSISTENCY_POLICY),
				('experiment_role', _XY_NEIGHBOR_CONSENSUS_EXPERIMENT_ROLE),
				('target_representation', _XY_NEIGHBOR_UNANIMOUS_TARGET_REPRESENTATION),
				('target_semantics', _XY_NEIGHBOR_UNANIMOUS_TARGET_SEMANTICS),
				('consistency_policy', _XY_NEIGHBOR_UNANIMOUS_CONSISTENCY_POLICY),
				('experiment_role', _XY_NEIGHBOR_UNANIMOUS_EXPERIMENT_ROLE),
			)
			if values.get(key) == expected
		)
	if forbidden:
		raise ValueError(
			'legacy multi-head checkpoint identity must not contain XY neighbor '
			f'consensus fields: {sorted(forbidden)!r}'
		)


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return value


def _required_string(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty string')
	return value


def _required_sha256(value: object, label: str) -> str:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise ValueError(f'{label} must be a lowercase SHA-256 digest')
	return value


def _required_positive_finite_number(value: object, label: str) -> float:
	if (
		isinstance(value, bool)
		or not isinstance(value, int | float)
		or not math.isfinite(float(value))
		or value <= 0.0
	):
		raise ValueError(f'{label} must be a positive finite number')
	return float(value)


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as file_obj:
		for block in iter(lambda: file_obj.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _state_sha256(state: Mapping[str, object]) -> str:
	digest = hashlib.sha256()
	for key in sorted(state):
		value = state[key]
		if not isinstance(value, torch.Tensor):
			raise TypeError(f'state value {key!r} must be a tensor')
		cpu = value.detach().cpu().contiguous()
		digest.update(key.encode('utf-8'))
		digest.update(str(cpu.dtype).encode('utf-8'))
		digest.update(str(tuple(cpu.shape)).encode('utf-8'))
		digest.update(cpu.view(torch.uint8).numpy().tobytes())
	return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
	return hashlib.sha256(
		json.dumps(
			_to_plain_value(value),
			sort_keys=True,
			separators=(',', ':'),
			allow_nan=False,
		).encode('utf-8')
	).hexdigest()


def scientific_identity_sha256(scientific_identity: Mapping[str, object]) -> str:
	"""Return the canonical checksum recorded for a scientific identity."""
	return _canonical_sha256(scientific_identity)


def _optimizer_group_identity(
	optimizer: torch.optim.Optimizer,
	*,
	parameter_names: Mapping[int, str],
) -> list[dict[str, object]]:
	result: list[dict[str, object]] = []
	for group in optimizer.param_groups:
		parameters = group['params']
		try:
			names = [parameter_names[id(parameter)] for parameter in parameters]
		except KeyError as exc:
			raise ValueError(
				'optimizer contains a parameter outside the student/head modules'
			) from exc
		if len(names) != len(set(names)):
			raise ValueError('optimizer parameter group contains duplicate parameters')
		result.append(
			{
				'name': group.get('name'),
				'parameter_names': names,
				'lr': _required_positive_finite_number(
					group.get('lr'),
					f'optimizer {group.get("name")!r} group lr',
				),
			}
		)
	return result


def _stratigraphy_parameter_names(
	student: torch.nn.Module,
	head: torch.nn.Module,
	*,
	spatial_context: torch.nn.Module | None = None,
) -> dict[int, str]:
	result: dict[int, str] = {}
	modules: tuple[tuple[str, torch.nn.Module], ...] = (
		('student', student),
		('head', head),
	)
	if spatial_context is not None:
		modules += (('spatial_context', spatial_context),)
	for prefix, module in modules:
		for name, parameter in module.named_parameters():
			parameter_id = id(parameter)
			if parameter_id in result:
				raise ValueError(
					'student and head modules must not share trainable parameters'
				)
			result[parameter_id] = f'{prefix}.{name}'
	return result


def _optimizer_state_group_identity_matches(value: object, identity: object) -> bool:
	if not isinstance(value, Mapping):
		return False
	groups = value.get('param_groups')
	if not isinstance(groups, list) or not isinstance(identity, list):
		return False
	if len(groups) != len(identity):
		return False
	next_parameter_id = 0
	for group, expected in zip(groups, identity, strict=True):
		if not isinstance(group, Mapping) or not isinstance(expected, Mapping):
			return False
		parameters = group.get('params')
		names = expected.get('parameter_names')
		if (
			group.get('name') != expected.get('name')
			or not isinstance(parameters, list)
			or not isinstance(names, list)
			or not all(isinstance(name, str) for name in names)
			or group.get('lr') != expected.get('lr')
			or len(parameters) != len(names)
			or parameters
			!= list(range(next_parameter_id, next_parameter_id + len(parameters)))
		):
			return False
		next_parameter_id += len(parameters)
	return True


def _optimizer_group_counts_match(identity: object, summary: object) -> bool:
	if not isinstance(identity, list) or not isinstance(summary, list):
		return False
	if len(identity) != len(summary):
		return False
	return all(
		isinstance(recorded, Mapping)
		and isinstance(expected, Mapping)
		and recorded.get('name') == expected.get('name')
		and recorded.get('lr') == expected.get('lr')
		and len(recorded.get('parameter_names', [])) == len(expected.get('params', []))
		for recorded, expected in zip(identity, summary, strict=True)
	)


def _optimizer_group_summary(value: object) -> object:
	if not isinstance(value, Mapping):
		return None
	groups = value.get('param_groups')
	return groups if isinstance(groups, list) else None


def _state_summary(value: object) -> dict[str, object] | None:
	if not isinstance(value, Mapping):
		return None
	state = {
		str(key): tensor
		for key, tensor in value.items()
		if isinstance(tensor, torch.Tensor)
	}
	return {
		'keys': sorted(state),
		'shapes': {key: list(tensor.shape) for key, tensor in state.items()},
		'sha256': _state_sha256(state),
	}


_PERIODIC_SELECTION_SCHEMA_VERSION = 1
_PERIODIC_SELECTION_KEYS = frozenset(
	{'schema_version', 'policy', 'events', 'selected'}
)
_PERIODIC_SELECTION_EVENT_KEYS = frozenset(
	{
		'sequence',
		'epoch',
		'global_step',
		'checkpoint_kind',
		'batch_index',
		'refresh_phase',
		'selected',
	}
)


def _periodic_checkpoint_selection_for_payload(  # noqa: PLR0913
	*,
	selection: Mapping[str, object] | None = None,
	epoch: int,
	global_step: int,
	checkpoint_kind: Literal['step', 'epoch', 'refresh'],
	batch_index: int | None,
	target_refresh_state: Mapping[str, object],
	train_epochs: int,
) -> dict[str, object]:
	if selection is None:
		events: list[Mapping[str, object]] = []
	else:
		validated = _validated_periodic_checkpoint_selection(
			selection, train_epochs=train_epochs
		)
		events = validated['events']
	if not isinstance(events, list):
		raise TypeError('periodic checkpoint selection events must be a list')
	state = _validated_target_refresh_state(target_refresh_state)
	event = {
		'sequence': len(events),
		'epoch': epoch,
		'global_step': global_step,
		'checkpoint_kind': checkpoint_kind,
		'batch_index': batch_index,
		'refresh_phase': state['refresh_phase'],
		'selected': False,
	}
	if _periodic_event_is_final(event, state, train_epochs):
		event['selected'] = True
	if any(
		_periodic_event_identity(existing) == _periodic_event_identity(event)
		for existing in events
	):
		raise ValueError(
			'periodic checkpoint selection history contains a duplicate event'
		)
	combined = [dict(existing) for existing in events]
	combined.append(event)
	selected = _periodic_event_identity(event) if event['selected'] else None
	return _validated_periodic_checkpoint_selection(
		{
			'schema_version': _PERIODIC_SELECTION_SCHEMA_VERSION,
			'policy': PERIODIC_REFRESH_CHECKPOINT_SELECTION_POLICY,
			'events': combined,
			'selected': selected,
		},
		train_epochs=train_epochs,
	)


def _validated_periodic_checkpoint_selection(  # noqa: C901, PLR0912
	value: object,
	*,
	train_epochs: int,
) -> dict[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('periodic checkpoint_selection must be a mapping')
	if set(value) != _PERIODIC_SELECTION_KEYS:
		raise ValueError('periodic checkpoint_selection fields are not closed')
	if value.get('schema_version') != _PERIODIC_SELECTION_SCHEMA_VERSION:
		raise ValueError('unsupported periodic checkpoint selection schema_version')
	if value.get('policy') != PERIODIC_REFRESH_CHECKPOINT_SELECTION_POLICY:
		raise ValueError('periodic checkpoint selection policy is invalid')
	if (
		isinstance(train_epochs, bool)
		or not isinstance(train_epochs, int)
		or train_epochs != 25
	):
		raise ValueError('periodic checkpoint selection requires train_epochs == 25')
	raw_events = value.get('events')
	if not isinstance(raw_events, list) or not raw_events:
		raise ValueError('periodic checkpoint selection events must be non-empty')
	events: list[dict[str, object]] = []
	previous: Mapping[str, object] | None = None
	selected_event: dict[str, object] | None = None
	for sequence, raw_event in enumerate(raw_events):
		event = _validated_periodic_selection_event(raw_event, sequence)
		if previous is not None:
			_periodic_selection_event_order(previous, event)
		if event['selected']:
			if selected_event is not None:
				raise ValueError(
					'periodic checkpoint selection has multiple selected events'
				)
			if event['checkpoint_kind'] != 'epoch' or event['epoch'] != train_epochs:
				raise ValueError('only completed epoch 25 may be selected periodically')
			if event['refresh_phase'] != 'training':
				raise ValueError(
					'refresh-required state cannot be selected periodically'
				)
			selected_event = _periodic_event_identity(event)
		previous = event
		events.append(event)
	selected = value.get('selected')
	if selected_event is None:
		if selected is not None:
			raise ValueError('periodic checkpoint selection selected must be null')
	else:
		if _periodic_event_identity(selected) != selected_event:
			raise ValueError(
				'periodic checkpoint selection selected event is inconsistent'
			)
		if not events[-1]['selected']:
			raise ValueError('periodic selected event must be the latest event')
	return {
		'schema_version': _PERIODIC_SELECTION_SCHEMA_VERSION,
		'policy': PERIODIC_REFRESH_CHECKPOINT_SELECTION_POLICY,
		'events': events,
		'selected': selected_event,
	}


def _validated_periodic_selection_event(
	value: object, sequence: int
) -> dict[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('periodic checkpoint selection event must be a mapping')
	if set(value) != _PERIODIC_SELECTION_EVENT_KEYS:
		raise ValueError('periodic checkpoint selection event fields are not closed')
	if value.get('sequence') != sequence:
		raise ValueError('periodic checkpoint selection sequence is not contiguous')
	epoch = _nonnegative_int_value(value.get('epoch'), 'periodic selection epoch')
	global_step = _nonnegative_int_value(
		value.get('global_step'), 'periodic selection global_step'
	)
	kind = value.get('checkpoint_kind')
	if kind not in {'step', 'epoch', 'refresh'}:
		raise ValueError('periodic checkpoint selection kind is invalid')
	batch_index = value.get('batch_index')
	if kind == 'step':
		batch_index = _nonnegative_int_value(
			batch_index, 'periodic step batch_index'
		)
	elif batch_index is not None:
		raise ValueError(
			'periodic epoch and refresh selection batch_index must be null'
		)
	phase = value.get('refresh_phase')
	if phase not in {'training', 'refresh_required', 'refresh_complete'}:
		raise ValueError('periodic selection refresh_phase is invalid')
	if kind == 'refresh' and phase != 'refresh_complete':
		raise ValueError('periodic refresh checkpoint must have refresh_complete phase')
	if not isinstance(value.get('selected'), bool):
		raise TypeError('periodic selection selected must be a boolean')
	return {
		'sequence': sequence,
		'epoch': epoch,
		'global_step': global_step,
		'checkpoint_kind': kind,
		'batch_index': batch_index,
		'refresh_phase': phase,
		'selected': value['selected'],
	}


def _periodic_event_identity(value: Mapping[str, object]) -> dict[str, object]:
	return {
		'sequence': value['sequence'],
		'epoch': value['epoch'],
		'global_step': value['global_step'],
		'checkpoint_kind': value['checkpoint_kind'],
		'batch_index': value['batch_index'],
		'refresh_phase': value['refresh_phase'],
	}


def _periodic_selection_event_order(
	previous: Mapping[str, object], event: Mapping[str, object]
) -> None:
	if (
		event['epoch'] < previous['epoch']
		or event['global_step'] < previous['global_step']
	):
		raise ValueError('periodic checkpoint selection events are not chronological')
	if event['global_step'] != previous['global_step']:
		return
	if event['epoch'] != previous['epoch']:
		raise ValueError('periodic selection repeated global_step changed epoch')
	ranks = {'step': 0, 'epoch': 1, 'refresh': 2}
	if ranks[str(event['checkpoint_kind'])] <= ranks[str(previous['checkpoint_kind'])]:
		raise ValueError('periodic selection boundary event order is invalid')


def _periodic_event_is_final(
	event: Mapping[str, object],
	state: Mapping[str, object],
	train_epochs: int,
) -> bool:
	return (
		event['checkpoint_kind'] == 'epoch'
		and event['epoch'] == train_epochs
		and state.get('refresh_phase') == 'training'
		and state.get('next_scheduled_refresh_epoch') is None
	)


def _validate_periodic_training_state(  # noqa: C901
	payload: Mapping[str, object], *, train_epochs: int
) -> None:
	training_state = payload.get('training_state')
	if not isinstance(training_state, Mapping):
		raise TypeError('schema-8 training_state must be a mapping')
	if set(training_state) != {
		'schema_version',
		'stage',
		'checkpoint_kind',
		'batch_index',
	}:
		raise ValueError('schema-8 training_state fields are not closed')
	if training_state.get('schema_version') != 1:
		raise ValueError('schema-8 training_state schema_version is invalid')
	if training_state.get('stage') != 'train_strat_hmm_pretext':
		raise ValueError('schema-8 training_state stage is invalid')
	kind = training_state.get('checkpoint_kind')
	if kind not in {'step', 'epoch', 'refresh'}:
		raise ValueError('schema-8 checkpoint kind is invalid')
	batch_index = training_state.get('batch_index')
	if kind == 'step':
		_nonnegative_int_value(batch_index, 'schema-8 step batch_index')
		epoch_metrics_state = _validated_epoch_metrics_state(
			payload.get('epoch_metrics_state')
		)
		if epoch_metrics_state['batch_count'] != int(batch_index) + 1:
			raise ValueError(
				'schema-8 epoch metrics state batch count does not match batch_index'
			)
	elif batch_index is not None:
		raise ValueError('schema-8 epoch/refresh batch_index must be null')
	checkpoint_config = payload.get('stratigraphy_config')
	state = _validated_target_refresh_state(
		payload.get('target_refresh_state'),
		expected_config=(
			checkpoint_config if isinstance(checkpoint_config, Mapping) else None
		),
	)
	epoch = _nonnegative_int_value(payload.get('epoch'), 'schema-8 epoch')
	if epoch > train_epochs:
		raise ValueError('schema-8 checkpoint epoch exceeds train.epochs')
	_validate_periodic_checkpoint_phase(
		epoch=epoch,
		checkpoint_kind=kind,
		state=state,
	)
	if kind == 'refresh' and state['refresh_phase'] != 'refresh_complete':
		raise ValueError(
			'schema-8 refresh checkpoint requires refresh_complete state'
		)


def _validated_epoch_metrics_state(value: object) -> dict[str, object]:
	"""Validate the exact running metric accumulator stored in step checkpoints."""
	if not isinstance(value, Mapping):
		raise TypeError('schema-8 epoch metrics state must be a mapping')
	if set(value) != {'schema_version', 'batch_count', 'totals'}:
		raise ValueError('schema-8 epoch metrics state fields are not closed')
	if value.get('schema_version') != 1:
		raise ValueError('schema-8 epoch metrics state schema_version is invalid')
	batch_count = _nonnegative_int_value(
		value.get('batch_count'), 'schema-8 epoch metrics batch_count'
	)
	if batch_count == 0:
		raise ValueError('schema-8 epoch metrics batch_count must be positive')
	totals = value.get('totals')
	if not isinstance(totals, Mapping) or not totals:
		raise ValueError('schema-8 epoch metrics totals must be non-empty')
	validated_totals: dict[str, float] = {}
	for key, raw_value in totals.items():
		if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
			raise TypeError('schema-8 epoch metric totals must be numeric')
		metric = float(raw_value)
		if not math.isfinite(metric):
			raise ValueError('schema-8 epoch metric totals must be finite')
		validated_totals[str(key)] = metric
	return {
		'schema_version': 1,
		'batch_count': batch_count,
		'totals': validated_totals,
	}


def _validate_periodic_checkpoint_phase(  # noqa: C901
	*,
	epoch: int,
	checkpoint_kind: object,
	state: Mapping[str, object],
) -> None:
	"""Bind refresh phase to the completed/partial training epoch boundary."""
	phase = state['refresh_phase']
	next_refresh = state['next_scheduled_refresh_epoch']
	last_refresh = state['last_completed_refresh_epoch']
	if checkpoint_kind == 'step':
		if phase != 'training':
			raise ValueError(
				'schema-8 step checkpoint must remain in training phase '
				'before a boundary'
			)
		if next_refresh is not None and epoch > next_refresh:
			raise ValueError(
				'schema-8 step checkpoint epoch skips a scheduled refresh boundary'
			)
		return
	if checkpoint_kind == 'refresh':
		if phase != 'refresh_complete':
			raise ValueError(
				'schema-8 refresh checkpoint must have refresh_complete phase'
			)
		if epoch not in PERIODIC_REFRESH_SCHEDULE or last_refresh != epoch:
			raise ValueError(
				'schema-8 refresh checkpoint epoch must be the completed scheduled '
				'refresh epoch'
			)
		return
	if checkpoint_kind != 'epoch':
		raise ValueError('schema-8 checkpoint kind is invalid')
	if phase == 'refresh_required':
		if epoch not in PERIODIC_REFRESH_SCHEDULE or next_refresh != epoch:
			raise ValueError(
				'schema-8 refresh_required epoch must identify the next '
				'scheduled refresh'
			)
		return
	if phase == 'refresh_complete':
		raise ValueError(
			'schema-8 refresh_complete state requires a refresh checkpoint'
		)
	if phase != 'training':
		raise ValueError('schema-8 checkpoint refresh phase is invalid')
	if next_refresh is not None and epoch >= next_refresh:
		raise ValueError(
			'schema-8 epoch checkpoint cannot remain training at a scheduled '
			'refresh or after it'
		)


def _validate_periodic_checkpoint_selection_payload_binding(  # noqa: C901, PLR0912
	payload: Mapping[str, object], selection: Mapping[str, object]
) -> None:
	events = selection['events']
	if not isinstance(events, list) or not events:
		raise ValueError('periodic checkpoint selection events must be non-empty')
	last = events[-1]
	if not isinstance(last, Mapping):
		raise TypeError('periodic checkpoint selection final event must be a mapping')
	training_state = payload.get('training_state')
	if not isinstance(training_state, Mapping):
		raise TypeError('periodic checkpoint training_state must be a mapping')
	epoch = _nonnegative_int_value(payload.get('epoch'), 'payload epoch')
	global_step = _nonnegative_int_value(
		payload.get('global_step'), 'payload global_step'
	)
	kind = training_state.get('checkpoint_kind')
	if kind not in {'step', 'epoch', 'refresh'}:
		raise ValueError('periodic checkpoint kind must be step, epoch, or refresh')
	batch_index = training_state.get('batch_index')
	if kind == 'step':
		batch_index = _nonnegative_int_value(batch_index, 'payload batch_index')
	elif batch_index is not None:
		raise ValueError('periodic epoch and refresh batch_index must be null')
	checkpoint_config = payload.get('stratigraphy_config')
	state = _validated_target_refresh_state(
		payload.get('target_refresh_state'),
		expected_config=(
			checkpoint_config if isinstance(checkpoint_config, Mapping) else None
		),
	)
	if (
		epoch != last['epoch']
		or global_step != last['global_step']
		or kind != last['checkpoint_kind']
		or batch_index != last['batch_index']
		or state['refresh_phase'] != last['refresh_phase']
	):
		raise ValueError(
			'periodic checkpoint payload does not match final selection event'
		)
	if kind == 'refresh':
		if len(events) < 2 or not isinstance(events[-2], Mapping):
			raise ValueError('schema-8 refresh checkpoint must follow an epoch event')
		previous = events[-2]
		if (
			previous.get('checkpoint_kind') != 'epoch'
			or previous.get('epoch') != epoch
			or previous.get('global_step') != global_step
			or previous.get('refresh_phase') != 'refresh_required'
		):
			raise ValueError(
				'schema-8 refresh checkpoint must follow a scheduled refresh boundary '
				'without advancing global_step or epoch'
			)
		if state['last_completed_refresh_epoch'] != epoch:
			raise ValueError(
			'schema-8 refresh checkpoint must complete the checkpoint epoch'
			)
	selected = selection.get('selected')
	eligible = _periodic_event_is_final(
		last,
		state,
		train_epochs=25,
	)
	if eligible and selected != _periodic_event_identity(last):
		raise ValueError('completed epoch 25 must be the periodic selected event')
	if selected is not None and not eligible:
		raise ValueError('periodic selected event is not a completed final epoch')


def checkpoint_selection_sha256(selection: Mapping[str, object]) -> str:
	"""Return the stable digest used to bind public selection evidence."""
	canonical = _validated_checkpoint_selection(selection)
	encoded = json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()
	return hashlib.sha256(encoded).hexdigest()


def selected_checkpoint_selection_event(
	selection: Mapping[str, object],
) -> Mapping[str, object]:
	"""Return a validated copy of the selected rolling-checkpoint event."""
	return _validated_checkpoint_selection(selection)['selected']


def _update_checkpoint_selection(  # noqa: PLR0913
	selection: Mapping[str, object] | None,
	*,
	epoch: int,
	global_step: int,
	checkpoint_kind: Literal['step', 'epoch'],
	batch_index: int | None,
	loss: float,
) -> dict[str, object]:
	if selection is None:
		events: list[dict[str, object]] = []
		best_score: float | None = None
	else:
		validated = _validated_checkpoint_selection(selection)
		events = [dict(event) for event in validated['events']]
		selected = validated['selected']
		best_score = float(selected['loss'])
	event = {
		'sequence': len(events),
		'epoch': int(epoch),
		'global_step': int(global_step),
		'checkpoint_kind': checkpoint_kind,
		'batch_index': batch_index,
		'loss': loss,
		'previous_best_score': best_score,
		'best_updated': _is_improved(loss, best_score),
		'best_score_after': loss if _is_improved(loss, best_score) else best_score,
	}
	if any(
		_selection_event_key(existing) == _selection_event_key(event)
		for existing in events
	):
		raise ValueError('checkpoint selection history contains a duplicate event')
	events.append(event)
	selected = event if event['best_updated'] else selected
	return _validated_checkpoint_selection(
		{
			'schema_version': _CHECKPOINT_SELECTION_SCHEMA_VERSION,
			'criterion': _CHECKPOINT_SELECTION_CRITERION,
			'improvement_policy': _CHECKPOINT_SELECTION_POLICY,
			'events': events,
			'selected': _selection_event_identity_from_event(selected),
		}
	)


def _validated_checkpoint_selection(  # noqa: C901, PLR0912
	value: object,
) -> dict[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('multi-head checkpoint_selection must be a mapping')
	if value.get('schema_version') != _CHECKPOINT_SELECTION_SCHEMA_VERSION:
		raise ValueError('unsupported checkpoint selection schema_version')
	if value.get('criterion') != _CHECKPOINT_SELECTION_CRITERION:
		raise ValueError('checkpoint selection criterion must be metrics.loss')
	if value.get('improvement_policy') != _CHECKPOINT_SELECTION_POLICY:
		raise ValueError('unsupported checkpoint selection improvement policy')
	raw_events = value.get('events')
	if not isinstance(raw_events, list) or not raw_events:
		raise ValueError('checkpoint selection events must be a non-empty list')
	events: list[dict[str, object]] = []
	best_score: float | None = None
	previous_event: Mapping[str, object] | None = None
	keys: set[tuple[int, int, str, int | None]] = set()
	selected: dict[str, object] | None = None
	for sequence, raw_event in enumerate(raw_events):
		if not isinstance(raw_event, Mapping):
			raise TypeError('checkpoint selection event must be a mapping')
		event = _validated_selection_event(raw_event, sequence)
		key = _selection_event_key(event)
		if key in keys:
			raise ValueError('checkpoint selection history contains a duplicate event')
		keys.add(key)
		if previous_event is not None:
			if event['epoch'] < previous_event['epoch']:
				raise ValueError('checkpoint selection event epochs must not regress')
			_selection_event_order(previous_event, event)
		previous_event = event
		if event['previous_best_score'] != best_score:
			raise ValueError('checkpoint selection previous_best_score is inconsistent')
		updated = _is_improved(float(event['loss']), best_score)
		if event['best_updated'] != updated:
			raise ValueError('checkpoint selection best_updated is inconsistent')
		best_score = float(event['loss']) if updated else best_score
		if event['best_score_after'] != best_score:
			raise ValueError('checkpoint selection best_score_after is inconsistent')
		if updated:
			selected = _selection_event_identity_from_event(event)
		events.append(event)
	if selected is None:
		raise ValueError('checkpoint selection history has no selected event')
	if _selection_event_identity(value.get('selected')) != selected:
		raise ValueError('checkpoint selection selected event is inconsistent')
	return {
		'schema_version': _CHECKPOINT_SELECTION_SCHEMA_VERSION,
		'criterion': _CHECKPOINT_SELECTION_CRITERION,
		'improvement_policy': _CHECKPOINT_SELECTION_POLICY,
		'events': events,
		'selected': selected,
	}


def _validate_checkpoint_selection_payload_binding(
	payload: Mapping[str, object], selection: Mapping[str, object]
) -> None:
	"""Require a checkpoint to identify the final event in its history."""
	events = selection['events']
	if not isinstance(events, list) or not events:
		raise ValueError('checkpoint selection events must be a non-empty list')
	last = events[-1]
	if not isinstance(last, Mapping):
		raise TypeError('checkpoint selection final event must be a mapping')
	training_state = payload.get('training_state')
	metrics = payload.get('metrics')
	if not isinstance(training_state, Mapping) or not isinstance(metrics, Mapping):
		raise TypeError(
			'checkpoint payload training_state and metrics must be mappings '
			'for checkpoint selection'
		)
	epoch, global_step = payload.get('epoch'), payload.get('global_step')
	if any(
		isinstance(value, bool) or not isinstance(value, int)
		for value in (epoch, global_step)
	):
		raise TypeError('checkpoint payload epoch/global_step must be integers')
	checkpoint_kind = training_state.get('checkpoint_kind')
	batch_index = training_state.get('batch_index')
	if checkpoint_kind not in {'step', 'epoch'}:
		raise ValueError('checkpoint payload checkpoint_kind must be step or epoch')
	if checkpoint_kind == 'step' and (
		isinstance(batch_index, bool)
		or not isinstance(batch_index, int)
		or batch_index < 0
	):
		raise TypeError(
			'checkpoint payload step batch_index must be a nonnegative integer'
		)
	if checkpoint_kind == 'epoch' and batch_index is not None:
		raise ValueError('checkpoint payload epoch batch_index must be null')
	loss = _finite_selection_number(metrics.get('loss'), 'payload metrics.loss')
	if (
		epoch != last['epoch']
		or global_step != last['global_step']
		or checkpoint_kind != last['checkpoint_kind']
		or batch_index != last['batch_index']
		or loss != last['loss']
	):
		raise ValueError(
			'checkpoint payload does not match final checkpoint selection event'
		)


def _validated_selection_event(
	value: Mapping[str, object], sequence: int
) -> dict[str, object]:
	if value.get('sequence') != sequence:
		raise ValueError('checkpoint selection event sequence is not contiguous')
	epoch, global_step = value.get('epoch'), value.get('global_step')
	if any(
		isinstance(item, bool) or not isinstance(item, int)
		for item in (epoch, global_step)
	):
		raise TypeError('checkpoint selection epoch/global_step must be integers')
	if epoch < 0 or global_step < 0:
		raise ValueError('checkpoint selection counters must be nonnegative')
	kind = value.get('checkpoint_kind')
	if kind not in {'step', 'epoch'}:
		raise ValueError('checkpoint selection checkpoint_kind must be step or epoch')
	batch_index = value.get('batch_index')
	if kind == 'step' and (
		isinstance(batch_index, bool)
		or not isinstance(batch_index, int)
		or batch_index < 0
	):
		raise TypeError('checkpoint selection step batch_index must be nonnegative')
	if kind == 'epoch' and batch_index is not None:
		raise ValueError('checkpoint selection epoch batch_index must be null')
	loss = _finite_selection_number(value.get('loss'), 'loss')
	previous = _nullable_selection_number(
		value.get('previous_best_score'), 'previous_best_score'
	)
	after = _nullable_selection_number(
		value.get('best_score_after'), 'best_score_after'
	)
	if not isinstance(value.get('best_updated'), bool):
		raise TypeError('checkpoint selection best_updated must be a boolean')
	return {
		'sequence': sequence,
		'epoch': epoch,
		'global_step': global_step,
		'checkpoint_kind': kind,
		'batch_index': batch_index,
		'loss': loss,
		'previous_best_score': previous,
		'best_updated': value['best_updated'],
		'best_score_after': after,
	}


def _finite_selection_number(value: object, label: str) -> float:
	if (
		isinstance(value, bool)
		or not isinstance(value, int | float)
		or not math.isfinite(float(value))
	):
		raise ValueError(f'checkpoint selection {label} must be finite')
	return float(value)


def _nullable_selection_number(value: object, label: str) -> float | None:
	return None if value is None else _finite_selection_number(value, label)


def _selection_event_key(
	event: Mapping[str, object],
) -> tuple[int, int, str, int | None]:
	return (
		int(event['epoch']),
		int(event['global_step']),
		str(event['checkpoint_kind']),
		event['batch_index'] if isinstance(event['batch_index'], int) else None,
	)


def _selection_event_order(
	previous: Mapping[str, object], event: Mapping[str, object]
) -> None:
	"""Require chronological checkpoints and the sole valid repeated-step pair."""
	previous_step = int(previous['global_step'])
	event_step = int(event['global_step'])
	if event_step < previous_step:
		raise ValueError('checkpoint selection events are not chronological')
	if event_step > previous_step:
		return
	if (
		event['epoch'] != previous['epoch']
		or previous['checkpoint_kind'] != 'step'
		or event['checkpoint_kind'] != 'epoch'
	):
		raise ValueError(
			'checkpoint selection repeated global_step must be a '
			'same-epoch step-to-epoch pair'
		)


def _selection_event_identity(value: object) -> dict[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('checkpoint selection selected event must be a mapping')
	keys = (
		'sequence',
		'epoch',
		'global_step',
		'checkpoint_kind',
		'batch_index',
		'loss',
	)
	if set(value) != set(keys):
		raise ValueError('checkpoint selection selected event identity is invalid')
	sequence, epoch, global_step = (
		value['sequence'],
		value['epoch'],
		value['global_step'],
	)
	if any(
		isinstance(item, bool) or not isinstance(item, int)
		for item in (sequence, epoch, global_step)
	):
		raise TypeError(
			'checkpoint selection selected event sequence/epoch/global_step '
			'must be integers'
		)
	if any(item < 0 for item in (sequence, epoch, global_step)):
		raise ValueError(
			'checkpoint selection selected event counters must be nonnegative'
		)
	kind = value['checkpoint_kind']
	if kind not in {'step', 'epoch'}:
		raise ValueError(
			'checkpoint selection selected event checkpoint_kind must be step or epoch'
		)
	batch_index = value['batch_index']
	if kind == 'step' and (
		isinstance(batch_index, bool)
		or not isinstance(batch_index, int)
		or batch_index < 0
	):
		raise TypeError(
			'checkpoint selection selected step batch_index must be a '
			'nonnegative integer'
		)
	if kind == 'epoch' and batch_index is not None:
		raise ValueError('checkpoint selection selected epoch batch_index must be null')
	return {
		'sequence': sequence,
		'epoch': epoch,
		'global_step': global_step,
		'checkpoint_kind': kind,
		'batch_index': batch_index,
		'loss': _finite_selection_number(value['loss'], 'selected event loss'),
	}


def _selection_event_identity_from_event(
	value: Mapping[str, object],
) -> dict[str, object]:
	keys = (
		'sequence',
		'epoch',
		'global_step',
		'checkpoint_kind',
		'batch_index',
		'loss',
	)
	return _selection_event_identity({key: value[key] for key in keys})


def _checkpoint_selection_transaction_path(checkpoint_root: Path) -> Path:
	return checkpoint_root / _CHECKPOINT_SELECTION_TRANSACTION_NAME


def _write_checkpoint_selection_transaction(
	checkpoint_root: Path, selection: Mapping[str, object]
) -> None:
	canonical = _validated_checkpoint_selection(selection)
	events = canonical['events']
	if not isinstance(events, list) or not isinstance(events[-1], Mapping):
		raise TypeError('checkpoint selection transaction events are invalid')
	selected = selected_checkpoint_selection_event(canonical)
	last = events[-1]
	if (
		_selection_event_identity_from_event(last) != selected
		or last.get('best_updated') is not True
	):
		raise ValueError('checkpoint selection transaction does not select latest')
	_atomic_json(
		_checkpoint_selection_transaction_path(checkpoint_root),
		{
			'schema_version': _CHECKPOINT_SELECTION_TRANSACTION_SCHEMA_VERSION,
			'selection_sha256': checkpoint_selection_sha256(canonical),
			'selected': selected,
		},
	)


def _load_checkpoint_selection_transaction(path: Path) -> dict[str, object]:
	try:
		value = json.loads(path.read_text(encoding='utf-8'))
	except (OSError, json.JSONDecodeError) as error:
		raise ValueError('checkpoint selection transaction is unreadable') from error
	if not isinstance(value, Mapping):
		raise TypeError('checkpoint selection transaction must be a mapping')
	if value.get('schema_version') != _CHECKPOINT_SELECTION_TRANSACTION_SCHEMA_VERSION:
		raise ValueError('unsupported checkpoint selection transaction schema_version')
	digest = value.get('selection_sha256')
	if not isinstance(digest, str) or len(digest) != 64:
		raise TypeError('checkpoint selection transaction digest is invalid')
	if any(character not in '0123456789abcdef' for character in digest):
		raise ValueError('checkpoint selection transaction digest is invalid')
	return {
		'selection_sha256': digest,
		'selected': _selection_event_identity(value.get('selected')),
	}


def _checkpoint_payload_matches_selected_event(
	path: Path, selected: Mapping[str, object]
) -> bool:
	if not path.is_file():
		return False
	try:
		payload = load_checkpoint(path, map_location='cpu')
		selection = _validated_checkpoint_selection(payload.get('checkpoint_selection'))
		_validate_checkpoint_selection_payload_binding(payload, selection)
		events = selection['events']
		return (
			isinstance(events, list)
			and isinstance(events[-1], Mapping)
			and selected_checkpoint_selection_event(selection) == selected
			and _selection_event_identity_from_event(events[-1]) == selected
		)
	except (OSError, RuntimeError, TypeError, ValueError):
		return False


def _required_loss_score(metrics: Mapping[str, float]) -> float:
	score = _loss_score(metrics)
	if score is None:
		raise ValueError('checkpoint selection requires finite metrics.loss')
	return score


def _write_checkpoint_selection_reports(
	checkpoint_root: Path, selection: Mapping[str, object]
) -> None:
	canonical = _validated_checkpoint_selection(selection)
	_atomic_json(checkpoint_root / 'checkpoint_selection_summary.json', canonical)
	fieldnames = [
		'sequence',
		'epoch',
		'global_step',
		'checkpoint_kind',
		'batch_index',
		'loss',
		'previous_best_score',
		'best_updated',
		'best_score_after',
	]
	fd, name = tempfile.mkstemp(
		prefix='.checkpoint_selection_history.', suffix='.tmp', dir=checkpoint_root
	)
	temporary = Path(name)
	try:
		with os.fdopen(fd, 'w', newline='', encoding='utf-8') as handle:
			writer = csv.DictWriter(handle, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(canonical['events'])
			handle.flush()
			os.fsync(handle.fileno())
		temporary.replace(checkpoint_root / 'checkpoint_selection_history.csv')
	finally:
		if temporary.exists():
			temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
	fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
	temporary = Path(name)
	try:
		with os.fdopen(fd, 'w', encoding='utf-8') as handle:
			json.dump(value, handle, sort_keys=True, indent=2)
			handle.write('\n')
			handle.flush()
			os.fsync(handle.fileno())
		temporary.replace(path)
	finally:
		if temporary.exists():
			temporary.unlink()


def _loss_score(metrics: Mapping[str, float]) -> float | None:
	value = metrics.get('loss')
	if isinstance(value, int | float) and not isinstance(value, bool):
		score = float(value)
		if math.isfinite(score):
			return score
	return None


def _is_improved(score: float | None, best_score: float | None) -> bool:
	if score is None:
		return False
	if best_score is None:
		return True
	return score < best_score


def _copy_checkpoint_atomic(source: Path, target: Path) -> None:
	target.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = target.with_suffix('.pt.tmp')
	shutil.copy2(source, tmp_path)
	tmp_path.replace(target)


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> Path:
	fd, tmp_name = tempfile.mkstemp(
		prefix=f'.{path.name}.',
		suffix='.tmp',
		dir=path.parent,
	)
	tmp_path = Path(tmp_name)
	try:
		with os.fdopen(fd, 'wb') as file_obj:
			torch.save(dict(payload), file_obj)
			file_obj.flush()
			os.fsync(file_obj.fileno())
		tmp_path.replace(path)
	finally:
		if tmp_path.exists():
			tmp_path.unlink()
	return path


def _to_plain_value(value: object) -> object:
	if isinstance(value, Mapping):
		return {str(key): _to_plain_value(child) for key, child in value.items()}
	if isinstance(value, list | tuple):
		return [_to_plain_value(child) for child in value]
	if isinstance(value, Path):
		return str(value)
	return value


__all__ = [
	'StratRollingCheckpointResult',
	'recover_strat_hmm_rolling_checkpoint',
	'save_strat_hmm_checkpoint',
	'save_strat_hmm_rolling_checkpoint',
]
