"""Runner for stratigraphic HMM pretext training."""



from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.config.pretraining import (
	PERIODIC_REFRESH_CHECKPOINT_SELECTION_POLICY,
	PERIODIC_REFRESH_PREPROCESSING_POLICY,
	PERIODIC_REFRESH_SCHEDULE,
	PERIODIC_REFRESH_SCHEDULE_SEMANTICS,
	_is_periodic_refresh_config,
)
from seis_ssl_cluster.data import (
	NopimsStratMultiHeadPosteriorDataset,
	NopimsStratMultiHeadTargetDataset,
	NopimsStratPseudoTargetDataset,
	load_strat_multi_head_lateral_target_manifest_adapter,
	load_strat_multi_head_xy_neighbor_consensus_target_manifest_adapter,
	load_strat_multi_head_xy_neighbor_unanimous_target_manifest_adapter,
	read_manifest_json,
)
from seis_ssl_cluster.embedding import (
	REFRESH_EXTRACTION_DESCRIPTOR_NAME,
	extract_embeddings_from_loaded_model,
)
from seis_ssl_cluster.stratigraphy import (
	discover_pseudo_target_inputs,
)
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.stratigraphy.periodic_refresh import (
	CANONICAL_KS,
	HardTargetPolicy,
	HashedArtifactReference,
	InitialHMMArtifact,
	InitialPeriodicRefreshConfig,
	PeriodicRefreshConfig,
	PreviousCenterArtifact,
	load_periodic_refresh_generation,
	produce_initial_periodic_refresh_generation,
	produce_periodic_refresh_generation,
	quarantine_periodic_refresh_generation,
)
from seis_ssl_cluster.stratigraphy.periodic_refresh import (
	_initial_request_identity as _periodic_initial_request_identity,
)
from seis_ssl_cluster.stratigraphy.periodic_refresh import (
	_request_identity as _periodic_request_identity,
)
from seis_ssl_cluster.stratigraphy.targets import (
	StratPseudoTargetInput,
	load_pseudo_target_arrays,
)
from seis_ssl_cluster.training.checkpoint import (
	capture_rng_state,
	load_checkpoint,
	restore_rng_state,
)
from seis_ssl_cluster.training.dataloaders import (
	build_strat_multi_head_posterior_dataloader,
	build_strat_multi_head_target_dataloader,
	build_strat_pseudo_target_dataloader,
)
from seis_ssl_cluster.training.mae import prepare_run_directory
from seis_ssl_cluster.training.strat_hmm.components import (
	_trainability_metrics,
	build_strat_hmm_components,
)
from seis_ssl_cluster.training.strat_hmm.epoch import (
	train_strat_hmm_center_trace_masked_one_epoch,
	train_strat_hmm_head_only_one_epoch,
	train_strat_hmm_multi_head_one_epoch,
)
from seis_ssl_cluster.training.strat_hmm.resume import (
	restore_strat_hmm_training_checkpoint,
)
from seis_ssl_cluster.training.strat_hmm.runtime import (
	_bool_config,
	_dataloader_generator_state,
	_float_config,
	_int_config,
	_load_existing_best_score,
	_mapping,
	_optional_float_config,
	_optional_int_config,
	_optional_positive_int_config,
	_path_config,
	_resolve_device,
	_restore_dataloader_generator_state,
	_rng_state_for_step_checkpoint,
	_rng_state_with_dataloader,
	_snapshot_run_inputs,
	_strat_hmm_control_identity,
	_trainability_summary_payload,
	_write_run_metadata,
	_xyz_config,
	_zero_mask_from_config,
)
from seis_ssl_cluster.training.strat_hmm.state import (
	StratHmmCenterTraceMaskedComponents,
	StratHmmMultiHeadComponents,
	StratHmmResumeState,
	StratHmmTrainingState,
)
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	_periodic_fixed_preprocessing_identity_sha256,
	recover_strat_hmm_rolling_checkpoint,
	save_strat_hmm_rolling_checkpoint,
	selected_checkpoint_selection_event,
)

if TYPE_CHECKING:
	from collections.abc import Iterable

	from seis_ssl_cluster.data.window_preprocessing import FiniteCheckMode


def run_strat_hmm_pretext_training(  # noqa: C901, PLR0912, PLR0915
	config: Mapping[str, object],
	*,
	resume: str | Path | None = None,
	quarantine_invalid: bool = False,
) -> Path:
	"""Run strat HMM pretext training from ``config``.

	Both single-head and multi-head runs use the same rolling checkpoint contract.
	For periodic refresh runs, ``quarantine_invalid`` explicitly moves an owned
	partial or foreign generation aside before retrying it; the default remains
	fail-closed.
	"""
	train_config = _mapping(config, 'train')
	paths_config = _mapping(config, 'paths')
	data_config = _mapping(config, 'data')
	model_config = _mapping(config, 'model')
	pseudo_config = _mapping(config, 'pseudo_targets')
	is_multi_head = 'spec' in _mapping(config, 'head')
	is_periodic_refresh = _is_periodic_refresh_config(config)
	target_representation = pseudo_config.get(
		'target_representation', 'hard_viterbi_labels_v1'
	)
	device = _resolve_device(train_config)
	seed = _int_config(train_config, 'seed', 42)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed_all(seed)

	output_root = _path_config(paths_config, 'output_root')
	control_identity = _strat_hmm_control_identity(config)
	allow_overwrite = _bool_config(
		train_config,
		'allow_overwrite_output',
		default=False,
	)
	prepare_run_directory(
		output_root=output_root,
		resume=resume,
		allow_overwrite=allow_overwrite or _preflight_only_output_root(output_root),
	)
	execution_counts = {
		'fresh': int(resume is None),
		'resume': int(resume is not None),
	}
	_snapshot_run_inputs(
		output_root=output_root,
		config=config,
		control_identity=control_identity,
		execution_counts=execution_counts,
		overwrite=(allow_overwrite and resume is None),
	)

	manifests = read_manifest_json(_path_config(_mapping(config, 'manifests'), 'train'))
	dataset_kwargs = {
		'local_crop_size_xyz': _xyz_config(data_config, 'local_crop_size'),
		'patch_size_xyz': _xyz_config(model_config, 'patch_size'),
		'seed': seed,
		'samples_per_epoch': _int_config(train_config, 'samples_per_epoch', 1),
		'zero_mask': _zero_mask_from_config(config),
		'min_valid_fraction': _float_config(data_config, 'min_valid_fraction', 0.0),
		'max_resample_attempts': _int_config(data_config, 'max_resample_attempts', 16),
		'normalized_clip_abs': _optional_float_config(
			data_config, 'normalized_clip_abs'
		),
		'amplitude_agc': data_config.get('amplitude_agc'),
		'finite_check_mode': cast(
			'FiniteCheckMode', data_config.get('finite_check_mode', 'strict')
		),
		'min_confidence': _float_config(pseudo_config, 'min_confidence', 0.0),
	}
	periodic_resume_payload: Mapping[str, object] | None = None
	periodic_state: dict[str, object] | None = None
	if is_periodic_refresh:
		if resume is None:
			periodic_state = _initialize_periodic_refresh_state(
				config=config,
				output_root=output_root,
				quarantine_invalid=quarantine_invalid,
			)
		else:
			periodic_resume_payload = load_checkpoint(resume, map_location=device)
			raw_state = periodic_resume_payload.get('target_refresh_state')
			if not isinstance(raw_state, Mapping):
				raise ValueError(
					'periodic refresh resume checkpoint is missing target_refresh_state'
				)
			periodic_state = dict(raw_state)
	target_manifest_override = (
		None
		if periodic_state is None
		else Path(str(periodic_state['active_target_manifest_path']))
	)
	dataset, dataloader = _build_strat_hmm_dataset_and_dataloader(
		config=config,
		manifests=manifests,
		dataset_kwargs=dataset_kwargs,
		device=device,
		target_manifest_override=target_manifest_override,
	)
	components = build_strat_hmm_components(config, device=device)
	is_center_trace = isinstance(components, StratHmmCenterTraceMaskedComponents)
	identity_head = (
		components.heads
		if is_multi_head
		else components.head
	)
	if control_identity is not None:
		control_identity = _with_initial_parameter_identities(
			control_identity,
			student=components.student,
			head=identity_head,
			replacement_token=(
				components.replacement_token if is_center_trace else None
			),
		)
	_write_run_metadata(
		output_root=output_root,
		trainability_summary=components.trainability_summary,
		control_identity=control_identity,
		execution_counts=execution_counts,
		overwrite=True,
	)
	amp_enabled = (
		_bool_config(train_config, 'amp', default=False)
		and device.type == 'cuda'
		and torch.cuda.is_available()
	)
	scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled) if amp_enabled else None
	resume_state = StratHmmResumeState(start_epoch=1, global_step=0, skip_batches=0)
	resume_checkpoint_kind: object | None = None
	resume_epoch_metric_totals: dict[str, float] | None = None
	resume_epoch_batch_count = 0
	recovered_completed_epoch = False
	if resume is not None:
		if is_multi_head and not is_periodic_refresh:
			recover_strat_hmm_rolling_checkpoint(Path(resume).parent)
		payload = periodic_resume_payload or load_checkpoint(
			resume, map_location=device
		)
		resume_checkpoint_kind = _mapping(
			payload, 'training_state'
		).get('checkpoint_kind')
		resume_state = restore_strat_hmm_training_checkpoint(
			payload=payload,
			student=components.student,
			head=(
				components.heads
				if is_multi_head
				else components.head
			),
			spatial_context=(
				components.replacement_token if is_center_trace else None
			),
			optimizer=components.optimizer,
			scaler=scaler,
			amp_enabled=amp_enabled,
			config=config,
		)
		if is_periodic_refresh and resume_checkpoint_kind == 'step':
			epoch_metrics_state = _mapping(
				payload, 'epoch_metrics_state'
			)
			raw_totals = _mapping(
				epoch_metrics_state, 'totals'
			)
			resume_epoch_metric_totals = {
				str(key): float(value) for key, value in raw_totals.items()
			}
			resume_epoch_batch_count = _int_config(
				epoch_metrics_state, 'batch_count', 0
			)
			if resume_epoch_batch_count != resume_state.skip_batches:
				raise ValueError(
					'periodic epoch metrics state does not match resumed batch position'
				)
		if is_periodic_refresh:
			if not isinstance(resume_state.target_refresh_state, Mapping):
				raise ValueError(
					'periodic refresh resume state is missing target_refresh_state'
				)
			if resume_checkpoint_kind in {'epoch', 'refresh'}:
				_recover_periodic_checkpoint_event(
					output_root=output_root,
					payload=payload,
					state=resume_state.target_refresh_state,
					components=components,
				)
			periodic_state = dict(resume_state.target_refresh_state)
			if resume_checkpoint_kind == 'refresh':
				periodic_state = _periodic_state_with_phase(
					periodic_state, phase='training'
				)
				resume_state = StratHmmResumeState(
					start_epoch=resume_state.start_epoch,
					global_step=resume_state.global_step,
					skip_batches=resume_state.skip_batches,
					refresh_phase='training',
					refresh_required=False,
					target_refresh_state=periodic_state,
				)
		_restore_dataloader_generator_state(payload=payload, dataloader=dataloader)
		if (
			resume_checkpoint_kind == 'step'
			and resume_state.skip_batches >= len(dataloader)
		):
			if is_periodic_refresh:
				if not isinstance(periodic_state, Mapping):
					raise ValueError('periodic refresh state was not initialized')
				recovered_completed_epoch = True
				periodic_state = _periodic_state_with_phase(
					periodic_state,
					phase=(
						'refresh_required'
						if _periodic_scheduled_epoch(resume_state.start_epoch)
						else 'training'
					),
				)
			resume_state = StratHmmResumeState(
				start_epoch=resume_state.start_epoch + 1,
				global_step=resume_state.global_step,
				skip_batches=0,
				refresh_phase=(
					periodic_state.get('refresh_phase')
					if isinstance(periodic_state, Mapping)
					else resume_state.refresh_phase
				),
				refresh_required=(
					bool(
						isinstance(periodic_state, Mapping)
						and periodic_state.get('refresh_phase') == 'refresh_required'
					)
					if is_periodic_refresh
					else resume_state.refresh_required
				),
				target_refresh_state=(
					periodic_state
					if is_periodic_refresh
					else resume_state.target_refresh_state
				),
			)
	epochs = _int_config(train_config, 'epochs', 1)
	max_steps = _optional_int_config(train_config, 'max_steps')
	checkpoint_every_steps = _optional_positive_int_config(
		train_config,
		'checkpoint_every_steps',
	)
	grad_clip_norm = _optional_float_config(train_config, 'grad_clip_norm')

	state = StratHmmTrainingState(
		epoch=resume_state.start_epoch - 1,
		global_step=resume_state.global_step,
		metrics={},
		last_batch_index=-1,
		completed_epoch=True,
	)
	checkpoint_path: Path | None = None
	best_score = (
		_load_existing_best_score(output_root)
		if resume is not None and not _is_periodic_refresh_config(config)
		else None
	)
	checkpoint_selection: Mapping[str, object] | None = None
	if resume is not None and is_multi_head:
		checkpoint_selection = payload.get('checkpoint_selection')
		if not isinstance(checkpoint_selection, Mapping):
			raise TypeError('multi-head resume checkpoint is missing selection history')
		if not is_periodic_refresh:
			best_score = float(
				selected_checkpoint_selection_event(checkpoint_selection)['loss']
			)
	if recovered_completed_epoch:
		if not isinstance(periodic_state, Mapping):
			raise ValueError('periodic refresh state was not initialized')
		checkpoint_state = dict(periodic_state)
		result = _save_periodic_checkpoint(
			output_root=output_root,
			components=components,
			config=config,
			metrics=_checkpoint_metrics(payload),
			global_step=resume_state.global_step,
			epoch=resume_state.start_epoch - 1,
			checkpoint_kind='epoch',
			batch_index=None,
			dataloader=dataloader,
			amp_enabled=amp_enabled,
			scaler=scaler,
			control_identity=control_identity,
			checkpoint_selection=checkpoint_selection,
			target_refresh_state=checkpoint_state,
		)
		checkpoint_path = result.latest_path
		checkpoint_selection = result.checkpoint_selection
		_append_target_refresh_event(
			output_root,
			{
				'event_type': 'checkpoint',
				'status': 'complete',
				'checkpoint_kind': 'epoch',
				'epoch': resume_state.start_epoch - 1,
				'global_step_before': resume_state.global_step,
				'global_step_after': resume_state.global_step,
				'active_generation_id': checkpoint_state['active_generation_id'],
				'active_generation_manifest_sha256': checkpoint_state[
					'active_generation_manifest_sha256'
				],
				'active_generation_content_sha256': checkpoint_state[
					'active_generation_content_sha256'
				],
				'active_target_manifest_sha256': checkpoint_state[
					'active_target_manifest_sha256'
				],
				'source_student_state_sha256': checkpoint_state[
					'source_student_state_sha256'
				],
				'student_state_sha256': _state_dict_sha256(
					components.student.state_dict()
				),
				'optimizer_state_sha256': _optimizer_state_sha256(
					components.optimizer
				),
				'refresh_phase': checkpoint_state['refresh_phase'],
				'recovered_from_completed_step': True,
			},
		)
		_append_multi_head_epoch_metrics(
			output_root=output_root,
			epoch=resume_state.start_epoch - 1,
			global_step=resume_state.global_step,
			metrics=_periodic_epoch_metrics_from_checkpoint(payload),
		)
	if is_periodic_refresh:
		if not isinstance(components, StratHmmCenterTraceMaskedComponents):
			raise TypeError('periodic refresh requires center-trace components')
		if not isinstance(periodic_state, Mapping):
			raise ValueError('periodic refresh state was not initialized')
		if resume_state.refresh_required:
			refresh_epoch = _periodic_refresh_epoch_from_state(periodic_state)
			dataset, dataloader, periodic_state = _perform_periodic_refresh(
				config=config,
				output_root=output_root,
				manifests=manifests,
				dataset_kwargs=dataset_kwargs,
				components=components,
				device=device,
				dataloader=dataloader,
				state=periodic_state,
				refresh_epoch=refresh_epoch,
				global_step=resume_state.global_step,
				quarantine_invalid=quarantine_invalid,
			)
			refresh_result = _save_periodic_checkpoint(
				output_root=output_root,
				components=components,
				config=config,
				metrics=_checkpoint_metrics(payload),
				global_step=resume_state.global_step,
				epoch=refresh_epoch,
				checkpoint_kind='refresh',
				batch_index=None,
				dataloader=dataloader,
				amp_enabled=amp_enabled,
				scaler=scaler,
				control_identity=control_identity,
				checkpoint_selection=checkpoint_selection,
				target_refresh_state=periodic_state,
			)
			checkpoint_path = refresh_result.latest_path
			checkpoint_selection = refresh_result.checkpoint_selection
			_append_target_refresh_event(
				output_root,
				{
					'event_type': 'checkpoint',
					'status': 'complete',
					'checkpoint_kind': 'refresh',
					'epoch': refresh_epoch,
					'global_step_before': resume_state.global_step,
					'global_step_after': resume_state.global_step,
					'active_generation_id': periodic_state['active_generation_id'],
					'active_generation_manifest_sha256': periodic_state[
						'active_generation_manifest_sha256'
					],
					'active_generation_content_sha256': periodic_state[
						'active_generation_content_sha256'
					],
					'active_target_manifest_sha256': periodic_state[
						'active_target_manifest_sha256'
					],
					'source_student_state_sha256': periodic_state[
						'source_student_state_sha256'
					],
					'student_state_sha256': _state_dict_sha256(
						components.student.state_dict()
					),
					'optimizer_state_sha256': _optimizer_state_sha256(
						components.optimizer
					),
					'refresh_phase': periodic_state['refresh_phase'],
				},
			)
			periodic_state = _periodic_state_with_phase(
				periodic_state, phase='training'
			)
			resume_state = StratHmmResumeState(
				start_epoch=resume_state.start_epoch,
				global_step=resume_state.global_step,
				skip_batches=0,
			)
	for epoch in range(resume_state.start_epoch, epochs + 1):
		set_epoch = getattr(dataset, 'set_epoch', None)
		if callable(set_epoch):
			set_epoch(epoch - 1)
		epoch_start_dataloader_rng_state = _dataloader_generator_state(dataloader)
		remaining_steps = None
		if max_steps is not None:
			remaining_steps = max_steps - state.global_step
			if remaining_steps <= 0:
				break
		skip_batches = (
			resume_state.skip_batches if epoch == resume_state.start_epoch else 0
		)

		if is_periodic_refresh:
			if not isinstance(components, StratHmmCenterTraceMaskedComponents):
				raise TypeError('periodic refresh requires center-trace components')
			if not isinstance(periodic_state, Mapping):
				raise ValueError('periodic refresh state was not initialized')

			def save_periodic_step_checkpoint(
				step_state: StratHmmTrainingState,
				epoch_start_rng_state: torch.Tensor = epoch_start_dataloader_rng_state,
			) -> None:
				nonlocal checkpoint_path, checkpoint_selection
				if (
					checkpoint_every_steps is None
					or step_state.global_step % checkpoint_every_steps != 0
				):
					return
				if (
					step_state.epoch_metric_totals is None
					or step_state.epoch_batch_count <= 0
				):
					raise ValueError(
						'periodic step checkpoint is missing epoch metric totals'
					)
				checkpoint_metrics = step_state.metrics
				if step_state.completed_epoch:
					if step_state.epoch_metrics is None:
						raise ValueError(
							'completed periodic step checkpoint is missing '
							'epoch metrics'
						)
					checkpoint_metrics = step_state.epoch_metrics
				result = _save_periodic_checkpoint(
					output_root=output_root,
					components=components,
					config=config,
					metrics={
						**checkpoint_metrics,
						**_trainability_metrics(components.trainability_summary),
						**(
							{'amp_enabled': float(amp_enabled)}
							if step_state.completed_epoch
							else {}
						),
					},
					global_step=step_state.global_step,
					epoch=step_state.epoch,
					checkpoint_kind='step',
					batch_index=step_state.last_batch_index,
					dataloader=dataloader,  # noqa: B023
					epoch_start_dataloader_rng_state=epoch_start_rng_state,
					amp_enabled=amp_enabled,
					scaler=scaler,
					control_identity=control_identity,
					checkpoint_selection=checkpoint_selection,
					target_refresh_state=periodic_state,  # noqa: B023
					epoch_metrics_state={
						'schema_version': 1,
						'batch_count': step_state.epoch_batch_count,
						'totals': step_state.epoch_metric_totals,
					},
				)
				checkpoint_path = result.latest_path
				checkpoint_selection = result.checkpoint_selection
				_append_target_refresh_event(
					output_root,
					{
						'event_type': 'checkpoint',
						'status': 'complete',
						'checkpoint_kind': 'step',
						'epoch': step_state.epoch,
						'global_step_before': step_state.global_step - 1,
						'global_step_after': step_state.global_step,
						'active_generation_id': periodic_state[  # noqa: B023
							'active_generation_id'
						],
						'active_generation_manifest_sha256': periodic_state[  # noqa: B023
							'active_generation_manifest_sha256'
						],
						'active_generation_content_sha256': periodic_state[  # noqa: B023
							'active_generation_content_sha256'
						],
						'active_target_manifest_sha256': periodic_state[  # noqa: B023
							'active_target_manifest_sha256'
						],
						'source_student_state_sha256': periodic_state[  # noqa: B023
							'source_student_state_sha256'
						],
						'student_state_sha256': _state_dict_sha256(
							components.student.state_dict()
						),
						'optimizer_state_sha256': _optimizer_state_sha256(
							components.optimizer
						),
					},
				)

			state = train_strat_hmm_center_trace_masked_one_epoch(
				student=components.student,
				teacher=components.teacher,
				heads=components.heads,
				replacement_token=components.replacement_token,
				dataloader=dataloader,
				optimizer=components.optimizer,
				device=device,
				epoch=epoch,
				loss_config=_mapping(config, 'loss'),
				pseudo_target_config=pseudo_config,
				training_seed=seed,
				column_fraction=float(
					_mapping(config, 'spatial_context')['column_fraction']
				),
				amp_enabled=amp_enabled,
				scaler=scaler,
				global_step=(
					resume_state.global_step
					if epoch == resume_state.start_epoch
					else state.global_step
				),
				max_steps=remaining_steps,
				grad_clip_norm=grad_clip_norm,
				skip_batches=skip_batches,
				step_callback=save_periodic_step_checkpoint,
				initial_epoch_metric_totals=(
					resume_epoch_metric_totals
					if epoch == resume_state.start_epoch
					else None
				),
				initial_epoch_batch_count=(
					resume_epoch_batch_count
					if epoch == resume_state.start_epoch
					else 0
				),
			)
			trainability_metrics = _trainability_metrics(
				components.trainability_summary
			)
			metrics = {
				**state.metrics,
				**trainability_metrics,
				'amp_enabled': float(amp_enabled),
			}
			if state.completed_epoch:
				scheduled = _periodic_scheduled_epoch(epoch)
				checkpoint_state = _periodic_state_with_phase(
					periodic_state,
					phase='refresh_required' if scheduled else 'training',
				)
				result = _save_periodic_checkpoint(
					output_root=output_root,
					components=components,
					config=config,
					metrics=metrics,
					global_step=state.global_step,
					epoch=epoch,
					checkpoint_kind='epoch',
					batch_index=None,
					dataloader=dataloader,
					amp_enabled=amp_enabled,
					scaler=scaler,
					control_identity=control_identity,
					checkpoint_selection=checkpoint_selection,
					target_refresh_state=checkpoint_state,
				)
				checkpoint_path = result.latest_path
				checkpoint_selection = result.checkpoint_selection
				_append_target_refresh_event(
					output_root,
					{
						'event_type': 'checkpoint',
						'status': 'complete',
						'checkpoint_kind': 'epoch',
						'epoch': epoch,
						'global_step_before': state.global_step,
						'global_step_after': state.global_step,
						'active_generation_id': checkpoint_state[
							'active_generation_id'
						],
						'active_generation_manifest_sha256': checkpoint_state[
							'active_generation_manifest_sha256'
						],
						'active_generation_content_sha256': checkpoint_state[
							'active_generation_content_sha256'
						],
						'active_target_manifest_sha256': checkpoint_state[
							'active_target_manifest_sha256'
						],
						'source_student_state_sha256': checkpoint_state[
							'source_student_state_sha256'
						],
						'student_state_sha256': _state_dict_sha256(
							components.student.state_dict()
						),
						'optimizer_state_sha256': _optimizer_state_sha256(
							components.optimizer
						),
						'refresh_phase': checkpoint_state['refresh_phase'],
					},
				)
				if scheduled:
					dataset, dataloader, periodic_state = _perform_periodic_refresh(
						config=config,
						output_root=output_root,
						manifests=manifests,
						dataset_kwargs=dataset_kwargs,
						components=components,
						device=device,
						dataloader=dataloader,
						state=periodic_state,
						refresh_epoch=epoch,
						global_step=state.global_step,
						quarantine_invalid=quarantine_invalid,
					)
					result = _save_periodic_checkpoint(
						output_root=output_root,
						components=components,
						config=config,
						metrics=metrics,
						global_step=state.global_step,
						epoch=epoch,
						checkpoint_kind='refresh',
						batch_index=None,
						dataloader=dataloader,
						amp_enabled=amp_enabled,
						scaler=scaler,
						control_identity=control_identity,
						checkpoint_selection=checkpoint_selection,
						target_refresh_state=periodic_state,
					)
					checkpoint_path = result.latest_path
					checkpoint_selection = result.checkpoint_selection
					_append_target_refresh_event(
						output_root,
						{
							'event_type': 'checkpoint',
							'status': 'complete',
							'checkpoint_kind': 'refresh',
							'epoch': epoch,
							'global_step_before': state.global_step,
							'global_step_after': state.global_step,
							'active_generation_id': periodic_state[
								'active_generation_id'
							],
							'active_generation_manifest_sha256': periodic_state[
								'active_generation_manifest_sha256'
							],
							'active_generation_content_sha256': periodic_state[
								'active_generation_content_sha256'
							],
							'active_target_manifest_sha256': periodic_state[
								'active_target_manifest_sha256'
							],
							'source_student_state_sha256': periodic_state[
								'source_student_state_sha256'
							],
							'student_state_sha256': _state_dict_sha256(
								components.student.state_dict()
							),
							'optimizer_state_sha256': _optimizer_state_sha256(
								components.optimizer
							),
							'refresh_phase': periodic_state['refresh_phase'],
						},
					)
					periodic_state = _periodic_state_with_phase(
						periodic_state, phase='training'
					)
			else:
				checkpoint_state = _periodic_state_with_phase(
					periodic_state, phase='training'
				)
				if (
					state.epoch_metric_totals is None
					or state.epoch_batch_count <= 0
				):
					raise ValueError(
						'periodic step checkpoint is missing epoch metric totals'
					)
				result = _save_periodic_checkpoint(
					output_root=output_root,
					components=components,
					config=config,
					metrics=metrics,
					global_step=state.global_step,
					epoch=epoch,
					checkpoint_kind='step',
					batch_index=state.last_batch_index,
					dataloader=dataloader,
					epoch_start_dataloader_rng_state=epoch_start_dataloader_rng_state,
					amp_enabled=amp_enabled,
					scaler=scaler,
					control_identity=control_identity,
					checkpoint_selection=checkpoint_selection,
					target_refresh_state=checkpoint_state,
					epoch_metrics_state={
						'schema_version': 1,
						'batch_count': state.epoch_batch_count,
						'totals': state.epoch_metric_totals,
					},
				)
				checkpoint_path = result.latest_path
				checkpoint_selection = result.checkpoint_selection
				periodic_state = checkpoint_state
			if state.completed_epoch:
				_append_multi_head_epoch_metrics(
					output_root=output_root,
					epoch=epoch,
					global_step=state.global_step,
					metrics=state.metrics,
				)
			if max_steps is not None and state.global_step >= max_steps:
				break
			continue

		if is_center_trace:

			def save_center_trace_step_checkpoint(
				step_state: StratHmmTrainingState,
				epoch_start_rng_state: torch.Tensor = epoch_start_dataloader_rng_state,
			) -> None:
				nonlocal best_score, checkpoint_path, checkpoint_selection
				if (
					checkpoint_every_steps is None
					or step_state.global_step % checkpoint_every_steps != 0
				):
					return
				result = save_strat_hmm_rolling_checkpoint(
					output_root,
					student=components.student,
					head=components.heads,
					spatial_context=components.replacement_token,
					optimizer=components.optimizer,
					epoch=step_state.epoch,
					mae_config=components.mae_checkpoint_config,
					stratigraphy_config=config,
					metrics={
						**step_state.metrics,
						**_trainability_metrics(components.trainability_summary),
					},
					global_step=step_state.global_step,
					checkpoint_kind='step',
					batch_index=step_state.last_batch_index,
					amp_enabled=amp_enabled,
					scaler=scaler,
					rng_state=_rng_state_for_step_checkpoint(
						dataloader=dataloader,  # noqa: B023
						epoch_start_dataloader_rng_state=epoch_start_rng_state,
						batch_index=step_state.last_batch_index,
					),
					trainability_summary=_trainability_summary_payload(
						components.trainability_summary
					),
					control_identity=control_identity,
					best_score=best_score,
					checkpoint_selection=checkpoint_selection,
				)
				best_score = result.best_score
				checkpoint_selection = result.checkpoint_selection
				checkpoint_path = result.latest_path

			state = train_strat_hmm_center_trace_masked_one_epoch(
				student=components.student,
				teacher=components.teacher,
				heads=components.heads,
				replacement_token=components.replacement_token,
				dataloader=dataloader,
				optimizer=components.optimizer,
				device=device,
				epoch=epoch,
				loss_config=_mapping(config, 'loss'),
				pseudo_target_config=pseudo_config,
				training_seed=seed,
				column_fraction=float(
					_mapping(config, 'spatial_context')['column_fraction']
				),
				amp_enabled=amp_enabled,
				scaler=scaler,
				global_step=state.global_step,
				max_steps=remaining_steps,
				grad_clip_norm=grad_clip_norm,
				skip_batches=skip_batches,
				step_callback=save_center_trace_step_checkpoint,
			)
		elif isinstance(components, StratHmmMultiHeadComponents):

			def save_multi_head_step_checkpoint(
				step_state: StratHmmTrainingState,
				epoch_start_rng_state: torch.Tensor = epoch_start_dataloader_rng_state,
			) -> None:
				nonlocal best_score, checkpoint_path, checkpoint_selection
				if (
					checkpoint_every_steps is None
					or step_state.global_step % checkpoint_every_steps != 0
				):
					return
				result = save_strat_hmm_rolling_checkpoint(
					output_root,
					student=components.student,
					head=components.heads,
					optimizer=components.optimizer,
					epoch=step_state.epoch,
					mae_config=components.mae_checkpoint_config,
					stratigraphy_config=config,
					metrics={
						**step_state.metrics,
						**_trainability_metrics(components.trainability_summary),
					},
					global_step=step_state.global_step,
					checkpoint_kind='step',
					batch_index=step_state.last_batch_index,
					amp_enabled=amp_enabled,
					scaler=scaler,
					rng_state=_rng_state_for_step_checkpoint(
						dataloader=dataloader,  # noqa: B023
						epoch_start_dataloader_rng_state=epoch_start_rng_state,
						batch_index=step_state.last_batch_index,
					),
					trainability_summary=_trainability_summary_payload(
						components.trainability_summary
					),
					control_identity=control_identity,
					best_score=best_score,
					checkpoint_selection=checkpoint_selection,
				)
				best_score = result.best_score
				checkpoint_selection = result.checkpoint_selection
				checkpoint_path = result.latest_path

			state = train_strat_hmm_multi_head_one_epoch(
				student=components.student,
				teacher=components.teacher,
				heads=components.heads,
				dataloader=dataloader,
				optimizer=components.optimizer,
				device=device,
				epoch=epoch,
				loss_config=_mapping(config, 'loss'),
				pseudo_target_config=pseudo_config,
				target_representation=str(target_representation),
				amp_enabled=amp_enabled,
				scaler=scaler,
				global_step=state.global_step,
				max_steps=remaining_steps,
				grad_clip_norm=grad_clip_norm,
				skip_batches=skip_batches,
				step_callback=save_multi_head_step_checkpoint,
			)
		else:

			def save_step_checkpoint(
				step_state: StratHmmTrainingState,
				epoch_start_rng_state: torch.Tensor = epoch_start_dataloader_rng_state,
			) -> None:
				nonlocal best_score, checkpoint_path
				if (
					checkpoint_every_steps is None
					or step_state.global_step % checkpoint_every_steps != 0
				):
					return
				result = save_strat_hmm_rolling_checkpoint(
					output_root,
					student=components.student,
					head=components.head,
					optimizer=components.optimizer,
					epoch=step_state.epoch,
					mae_config=components.mae_checkpoint_config,
					stratigraphy_config=config,
					metrics={
						**step_state.metrics,
						**_trainability_metrics(components.trainability_summary),
					},
					global_step=step_state.global_step,
					checkpoint_kind='step',
					batch_index=step_state.last_batch_index,
					amp_enabled=amp_enabled,
					scaler=scaler,
					rng_state=_rng_state_for_step_checkpoint(
						dataloader=dataloader,  # noqa: B023
						epoch_start_dataloader_rng_state=epoch_start_rng_state,
						batch_index=step_state.last_batch_index,
					),
					trainability_summary=_trainability_summary_payload(
						components.trainability_summary,
					),
					control_identity=control_identity,
					best_score=best_score,
				)
				best_score = result.best_score
				checkpoint_path = result.latest_path

			state = train_strat_hmm_head_only_one_epoch(
				student=components.student,
				teacher=components.teacher,
				head=components.head,
				dataloader=dataloader,
				optimizer=components.optimizer,
				device=device,
				epoch=epoch,
				loss_config=_mapping(config, 'loss'),
				pseudo_target_config=pseudo_config,
				amp_enabled=amp_enabled,
				scaler=scaler,
				global_step=state.global_step,
				max_steps=remaining_steps,
				grad_clip_norm=grad_clip_norm,
				skip_batches=skip_batches,
				step_callback=save_step_checkpoint,
			)
		trainability_metrics = _trainability_metrics(components.trainability_summary)
		checkpoint_kind: Literal['step', 'epoch'] = (
			'epoch' if state.completed_epoch else 'step'
		)
		result = save_strat_hmm_rolling_checkpoint(
			output_root,
			student=components.student,
			head=(
				components.heads
				if is_multi_head
				else components.head
			),
			spatial_context=(
				components.replacement_token if is_center_trace else None
			),
			optimizer=components.optimizer,
			epoch=epoch,
			mae_config=components.mae_checkpoint_config,
			stratigraphy_config=config,
			metrics={
				**state.metrics,
				**trainability_metrics,
				'amp_enabled': float(amp_enabled),
			},
			global_step=state.global_step,
			checkpoint_kind=checkpoint_kind,
			batch_index=None if state.completed_epoch else state.last_batch_index,
			amp_enabled=amp_enabled,
			scaler=scaler,
			rng_state=(
				_rng_state_with_dataloader(dataloader)
				if state.completed_epoch
				else _rng_state_for_step_checkpoint(
					dataloader=dataloader,
					epoch_start_dataloader_rng_state=epoch_start_dataloader_rng_state,
					batch_index=state.last_batch_index,
				)
			),
			trainability_summary=_trainability_summary_payload(
				components.trainability_summary,
			),
			control_identity=control_identity,
			best_score=best_score,
			checkpoint_selection=checkpoint_selection,
		)
		best_score = result.best_score
		checkpoint_selection = result.checkpoint_selection
		checkpoint_path = result.latest_path
		if is_multi_head and state.completed_epoch:
			_append_multi_head_epoch_metrics(
				output_root=output_root,
				epoch=epoch,
				global_step=state.global_step,
				metrics=state.metrics,
			)
		if max_steps is not None and state.global_step >= max_steps:
			break

	if checkpoint_path is None:
		msg = 'no strat HMM pretext training steps were run'
		raise ValueError(msg)
	return checkpoint_path


def _build_strat_hmm_dataset_and_dataloader(
	*,
	config: Mapping[str, object],
	manifests: object,
	dataset_kwargs: Mapping[str, object],
	device: torch.device,
	target_manifest_override: Path | None = None,
) -> tuple[object, torch.utils.data.DataLoader]:
	"""Build the configured target route with a fresh deterministic loader."""
	train_config = _mapping(config, 'train')
	pseudo_config = _mapping(config, 'pseudo_targets')
	is_multi_head = 'spec' in _mapping(config, 'head')
	target_representation = pseudo_config.get(
		'target_representation', 'hard_viterbi_labels_v1'
	)
	if target_manifest_override is not None and (
		not is_multi_head or target_representation != 'hard_viterbi_labels_v1'
	):
		raise ValueError('periodic refresh requires the hard multi-head target route')
	loader_kwargs = {
		'batch_size': _int_config(train_config, 'batch_size', 1),
		'num_workers': _int_config(train_config, 'num_workers', 0),
		'shuffle': _bool_config(train_config, 'shuffle', default=True),
		'seed': int(dataset_kwargs['seed']),
		'device': device,
	}
	if is_multi_head:
		if target_representation == 'ordered_path_state_posterior_v1':
			if target_manifest_override is not None:
				raise ValueError('periodic refresh cannot use posterior targets')
			posterior_kwargs = {
				key: value
				for key, value in dataset_kwargs.items()
				if key != 'min_confidence'
			}
			dataset = NopimsStratMultiHeadPosteriorDataset(
				manifests,
				_path_config(pseudo_config, 'manifest'),
				**posterior_kwargs,
			)
			return dataset, build_strat_multi_head_posterior_dataloader(
				dataset, **loader_kwargs
			)
		if target_representation not in {
			'hard_viterbi_labels_v1',
			'lateral_mean_field_hard_labels_v1',
			'xy_neighbor_consensus_hard_labels_v1',
			'xy_neighbor_unanimous_hard_labels_v1',
		}:
			raise ValueError(
				'unsupported multi-head target representation: '
				f'{target_representation!r}'
			)
		target_manifest: object = (
			target_manifest_override
			if target_manifest_override is not None
			else _path_config(pseudo_config, 'manifest')
		)
		if target_manifest_override is None:
			if target_representation == 'lateral_mean_field_hard_labels_v1':
				target_manifest = (
					load_strat_multi_head_lateral_target_manifest_adapter(
						target_manifest
					).target_manifest
				)
			elif target_representation == 'xy_neighbor_consensus_hard_labels_v1':
				target_manifest = (
					load_strat_multi_head_xy_neighbor_consensus_target_manifest_adapter(
						target_manifest
					).target_manifest
				)
			elif target_representation == 'xy_neighbor_unanimous_hard_labels_v1':
				target_manifest = (
					load_strat_multi_head_xy_neighbor_unanimous_target_manifest_adapter(
						target_manifest
					).target_manifest
				)
		dataset = NopimsStratMultiHeadTargetDataset(
			manifests, target_manifest, **dataset_kwargs
		)
		return dataset, build_strat_multi_head_target_dataloader(
			dataset, **loader_kwargs
		)
	pseudo_inputs = discover_pseudo_target_inputs(
			_path_config(pseudo_config, 'input_dir'),
			k=_int_config(pseudo_config, 'k', 1),
		)
	dataset = NopimsStratPseudoTargetDataset(
		manifests, pseudo_inputs, **dataset_kwargs
	)
	return dataset, build_strat_pseudo_target_dataloader(dataset, **loader_kwargs)


def _artifact_reference(path: str | Path) -> HashedArtifactReference:
	resolved = Path(path).resolve()
	if not resolved.is_file():
		raise FileNotFoundError(resolved)
	return HashedArtifactReference(path=resolved, sha256=file_sha256(resolved))


def _periodic_refresh_mapping(
	config: Mapping[str, object],
) -> Mapping[str, object]:
	value = config.get('pseudo_target_refresh')
	if not isinstance(value, Mapping):
		raise TypeError('pseudo_target_refresh must be a mapping')
	return value


def _periodic_nested_mapping(value: object, name: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{name} must be a mapping')
	return value


def _periodic_initial_artifacts(
	config: Mapping[str, object],
) -> tuple[
	tuple[InitialHMMArtifact, ...],
	HashedArtifactReference,
	HashedArtifactReference,
	HashedArtifactReference,
	]:
	refresh = _periodic_refresh_mapping(config)
	initial = _mapping(refresh, 'initial_hmm_artifacts')
	common = _mapping(initial, 'common')
	clustering_config = _artifact_reference(
		_path_config(common, 'clustering_config')
	)
	preprocessor = _artifact_reference(_path_config(common, 'preprocessor'))
	residualizer_value = common.get('residualizer')
	residualizer = (
		None
		if residualizer_value is None
		else _artifact_reference(Path(str(residualizer_value)))
	)
	source_embedding_metadata = _artifact_reference(
		_path_config(common, 'source_embedding_metadata')
	)
	heads = _mapping(initial, 'heads')
	artifacts: list[InitialHMMArtifact] = []
	for k in CANONICAL_KS:
		head = _mapping(heads, str(k))
		artifacts.append(
			InitialHMMArtifact(
				k=k,
				centers=_artifact_reference(_path_config(head, 'centers')),
				hmm_model=_artifact_reference(_path_config(head, 'hmm_model')),
				preprocessor=preprocessor,
				metadata=_artifact_reference(_path_config(head, 'model_metadata')),
				residualizer=residualizer,
			)
		)
	return (
		tuple(artifacts),
		clustering_config,
		source_embedding_metadata,
		_artifact_reference(
			_path_config(_mapping(config, 'pseudo_targets'), 'manifest')
		),
	)


def _periodic_target_policy(
	manifest_path: Path,
) -> HardTargetPolicy:
	payload = load_multi_head_target_manifest(manifest_path)
	heads = _periodic_nested_mapping(
		payload.get('heads'), 'target manifest heads'
	)
	confidence_values: list[float] = []
	boundary_modes: set[bool] = set()
	for k in CANONICAL_KS:
		head = _periodic_nested_mapping(
			heads.get(str(k)), f'target manifest k={k}'
		)
		surveys = _periodic_nested_mapping(
			head.get('surveys'), f'target manifest k={k} surveys'
		)
		for survey_id, raw_entry in surveys.items():
			entry = _periodic_nested_mapping(
				raw_entry, f'target manifest k={k} {survey_id}'
			)
			refs = {
				name: _periodic_nested_mapping(
					entry.get(name), f'{k} {survey_id} {name}'
				)
				for name in ('labels', 'confidence', 'valid_tokens', 'metadata')
			}
			boundary = entry.get('boundary_weight')
			boundary_ref = (
				None
				if boundary is None
				else Path(
					str(
						_periodic_nested_mapping(
							boundary, 'boundary_weight'
						)['path']
					)
				)
			)
			item = StratPseudoTargetInput(
				survey_id=str(survey_id),
				k=k,
				labels_path=Path(str(refs['labels']['path'])),
				confidence_path=Path(str(refs['confidence']['path'])),
				valid_tokens_path=Path(str(refs['valid_tokens']['path'])),
				boundary_weight_path=boundary_ref,
				metadata_path=Path(str(refs['metadata']['path'])),
			)
			arrays = load_pseudo_target_arrays(item, mmap_mode='r')
			valid = np.asarray(arrays.valid_tokens, dtype=np.bool_)
			values = np.asarray(arrays.confidence)[valid]
			if values.size == 0:
				raise ValueError(
					f'periodic target policy cannot infer confidence for {survey_id}'
				)
			confidence_values.extend(float(value) for value in values)
			boundary_modes.add(boundary is not None)
	if len(boundary_modes) != 1:
		raise ValueError('periodic target boundary-weight policy is inconsistent')
	constant = bool(confidence_values) and all(
		value == confidence_values[0] for value in confidence_values
	)
	return HardTargetPolicy(
		confidence_mode='constant' if constant else 'source_array',
		confidence=confidence_values[0] if constant else 1.0,
		boundary_weight_mode=(
			'source_array' if boundary_modes.pop() else 'absent'
		),
	)


def _periodic_initial_config(
	config: Mapping[str, object],
) -> tuple[InitialPeriodicRefreshConfig, Path]:
	refresh = _periodic_refresh_mapping(config)
	artifacts, clustering_config, source_metadata, initial_manifest = (
		_periodic_initial_artifacts(config)
	)
	refresh_root = _path_config(refresh, 'generation_root').resolve()
	return (
		InitialPeriodicRefreshConfig(
			initial_hard_target_manifest=initial_manifest,
			initial_hmm_artifacts=artifacts,
			clustering_config=clustering_config,
			source_embedding_metadata=source_metadata,
			output_generation_dir=(
				refresh_root / 'generations' / 'refresh_0000_initial'
			),
			target_policy=_periodic_target_policy(initial_manifest.path),
		),
		refresh_root,
	)


def _json_write_atomic(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fd, temporary_name = tempfile.mkstemp(
		prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent
	)
	temporary = Path(temporary_name)
	try:
		with os.fdopen(fd, 'w', encoding='utf-8') as handle:
			json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
			handle.write('\n')
			handle.flush()
			os.fsync(handle.fileno())
		temporary.replace(path)
	finally:
		if temporary.exists():
			temporary.unlink()


def _append_target_refresh_event(  # noqa: C901
	output_root: Path,
	payload: Mapping[str, object],
) -> None:
	path = output_root / 'target_refresh_events.jsonl'
	path.parent.mkdir(parents=True, exist_ok=True)
	existing_events = (
		_read_target_refresh_events(path) if path.is_file() else []
	)
	if payload.get('event_type') == 'refresh' and payload.get('status') in {
		'start',
		'complete',
	}:
		identity_fields = (
			'event_type',
			'status',
			'refresh_epoch',
			'generation_index',
			'generation_id',
			'source_student_state_sha256',
			'student_state_sha256',
			'optimizer_state_sha256',
		)
		if payload.get('status') == 'complete':
			identity_fields += (
				'output_generation_manifest_path',
				'output_generation_manifest_sha256',
				'active_target_manifest_path',
				'active_target_manifest_sha256',
			)
		for existing in existing_events:
			if all(
				existing.get(key) == payload.get(key)
				for key in ('event_type', 'status', 'refresh_epoch')
			) and not all(
					existing.get(key) == payload.get(key)
				for key in identity_fields
			):
					raise ValueError(
						'conflicting periodic refresh lifecycle event already exists'
					)
			if all(existing.get(key) == payload.get(key) for key in identity_fields):
				return
	checkpoint_identity = (
		'event_type',
		'status',
		'checkpoint_kind',
		'epoch',
		'global_step_after',
	)
	if (
		payload.get('event_type') == 'checkpoint'
		and payload.get('status') == 'complete'
	):
		for existing in existing_events:
			if all(
				existing.get(key) == payload.get(key) for key in checkpoint_identity
			):
				if dict(existing) != dict(payload):
					raise ValueError(
						'conflicting periodic checkpoint event already exists'
					)
				return
	needs_separator = False
	if path.is_file():
		existing_bytes = path.read_bytes()
		needs_separator = bool(existing_bytes) and not existing_bytes.endswith(b'\n')
	with path.open('a', encoding='utf-8') as handle:
		if needs_separator:
			handle.write('\n')
		handle.write(json.dumps(dict(payload), sort_keys=True, allow_nan=False))
		handle.write('\n')
		handle.flush()
		os.fsync(handle.fileno())


def _read_target_refresh_events(path: Path) -> list[Mapping[str, object]]:
	"""Read events and quarantine a malformed trailing append fragment."""
	raw = path.read_bytes()
	chunks = raw.splitlines(keepends=True)
	last_content_index = next(
		(
			index
			for index in range(len(chunks) - 1, -1, -1)
			if chunks[index].strip()
		),
		None,
	)
	events: list[Mapping[str, object]] = []
	offset = 0
	for index, chunk in enumerate(chunks):
		start = offset
		offset += len(chunk)
		if not chunk.strip():
			continue
		try:
			value = json.loads(chunk.decode('utf-8'))
		except (UnicodeDecodeError, json.JSONDecodeError) as exc:
			if index != last_content_index:
				raise ValueError(
					f'periodic refresh event is invalid JSON at line {index + 1}'
				) from exc
			_quarantine_target_refresh_event_fragment(path, raw[start:])
			break
		if not isinstance(value, Mapping):
			raise TypeError('periodic refresh event must be a mapping')
		events.append(value)
	return events


def _quarantine_target_refresh_event_fragment(path: Path, fragment: bytes) -> None:
	fd, _ = tempfile.mkstemp(
		prefix=f'{path.name}.quarantine.', dir=path.parent
	)
	with os.fdopen(fd, 'wb') as handle:
		handle.write(fragment)
		handle.flush()
		os.fsync(handle.fileno())
	with path.open('r+b') as handle:
		handle.truncate(path.stat().st_size - len(fragment))
		handle.flush()
		os.fsync(handle.fileno())


def _recover_periodic_checkpoint_event(
	*,
	output_root: Path,
	payload: Mapping[str, object],
	state: Mapping[str, object],
	components: StratHmmCenterTraceMaskedComponents,
) -> None:
	"""Reconcile checkpoint evidence before resuming a periodic refresh."""
	training_state = _mapping(payload, 'training_state')
	kind = training_state.get('checkpoint_kind')
	if kind not in {'epoch', 'refresh'}:
		return
	epoch = payload.get('epoch')
	global_step = payload.get('global_step')
	if (
		isinstance(epoch, bool)
		or not isinstance(epoch, int)
		or isinstance(global_step, bool)
		or not isinstance(global_step, int)
	):
		raise TypeError('periodic checkpoint event counters must be integers')
	_append_target_refresh_event(
		output_root,
		{
			'event_type': 'checkpoint',
			'status': 'complete',
			'checkpoint_kind': kind,
			'epoch': epoch,
			'global_step_before': global_step,
			'global_step_after': global_step,
			'active_generation_id': state['active_generation_id'],
			'active_generation_manifest_sha256': state[
				'active_generation_manifest_sha256'
			],
			'active_generation_content_sha256': state[
				'active_generation_content_sha256'
			],
			'active_target_manifest_sha256': state['active_target_manifest_sha256'],
			'source_student_state_sha256': state['source_student_state_sha256'],
			'student_state_sha256': _state_dict_sha256(
				components.student.state_dict()
			),
			'optimizer_state_sha256': _optimizer_state_sha256(
				components.optimizer
			),
			'refresh_phase': state['refresh_phase'],
		},
	)


def _periodic_generation_record(manifest_path: Path) -> dict[str, object]:
	payload = load_periodic_refresh_generation(manifest_path)
	previous = payload.get('previous_generation_manifest')
	previous_hash = (
		None
		if previous is None
		else str(
			_periodic_nested_mapping(
				previous, 'previous_generation_manifest'
			)['sha256']
		)
	)
	source_hash = payload.get('source_student_state_sha256')
	return {
		'generation_index': int(payload['generation_index']),
		'generation_id': str(payload['generation_id']),
		'refresh_after_epoch': int(payload['refresh_after_epoch']),
		'previous_generation_manifest_sha256': previous_hash,
		'source_student_state_sha256': (
			None if source_hash is None else str(source_hash)
		),
		'manifest_path': str(manifest_path.resolve()),
		'manifest_sha256': file_sha256(manifest_path),
		'generation_content_sha256': str(payload['generation_content_sha256']),
	}


def _periodic_chain_payload(
	*,
	fixed_identity_hash: str,
	generations: list[dict[str, object]],
) -> dict[str, object]:
	return {
		'schema_version': 1,
		'semantics': 'periodic_student_hmm_refresh_chain_v1',
		'refresh_after_epochs': list(PERIODIC_REFRESH_SCHEDULE),
		'fixed_preprocessing_hmm_identity_sha256': fixed_identity_hash,
		'generations': generations,
	}


def _periodic_chain_from_state(
	state: Mapping[str, object],
) -> list[dict[str, object]]:
	raw = state.get('generations')
	if not isinstance(raw, list):
		raise TypeError('periodic refresh state generations must be a list')
	result: list[dict[str, object]] = []
	for item in raw:
		if not isinstance(item, Mapping):
			raise TypeError('periodic refresh state generation must be a mapping')
		manifest_path = Path(str(item.get('manifest_path'))).resolve()
		record = _periodic_generation_record(manifest_path)
		for key in (
			'generation_index',
			'generation_id',
			'manifest_path',
			'manifest_sha256',
			'generation_content_sha256',
		):
			if item.get(key) != record[key]:
				raise ValueError(
					f'periodic refresh state generation {key} does not match manifest'
				)
		result.append(record)
	return result


def _periodic_state_generation_record(
	record: Mapping[str, object],
) -> dict[str, object]:
	return {
		'generation_index': record['generation_index'],
		'generation_id': record['generation_id'],
		'manifest_path': record['manifest_path'],
		'manifest_sha256': record['manifest_sha256'],
		'generation_content_sha256': record['generation_content_sha256'],
	}


def _periodic_fixed_identity_hash(config: Mapping[str, object]) -> str:
	identity = _mapping(config, 'identity')
	scientific = _mapping(identity, 'scientific_identity')
	return _periodic_fixed_preprocessing_identity_sha256(scientific)


def _read_periodic_chain(path: Path) -> dict[str, object]:
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except (OSError, json.JSONDecodeError) as exc:
		raise ValueError(f'periodic refresh chain must be valid JSON: {path}') from exc
	if not isinstance(payload, dict):
		raise TypeError('periodic refresh chain must be a JSON object')
	if set(payload) != {
		'schema_version',
		'semantics',
		'refresh_after_epochs',
		'fixed_preprocessing_hmm_identity_sha256',
		'generations',
	}:
		raise ValueError('periodic refresh chain fields are not closed')
	if payload['schema_version'] != 1 or payload['semantics'] != (
		'periodic_student_hmm_refresh_chain_v1'
	):
		raise ValueError('periodic refresh chain identity is invalid')
	if tuple(payload['refresh_after_epochs']) != PERIODIC_REFRESH_SCHEDULE:
		raise ValueError('periodic refresh chain schedule mismatch')
	if not isinstance(payload['generations'], list):
		raise TypeError('periodic refresh chain generations must be a list')
	return payload


def _periodic_pointer(path: Path) -> dict[str, str]:
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except (OSError, json.JSONDecodeError) as exc:
		raise ValueError(
			f'active target generation pointer must be valid JSON: {path}'
		) from exc
	if not isinstance(payload, dict) or set(payload) != {
		'manifest_path',
		'manifest_sha256',
	}:
		raise ValueError('active target generation pointer fields are not closed')
	if not isinstance(payload['manifest_path'], str) or not isinstance(
		payload['manifest_sha256'], str
	):
		raise TypeError('active target generation pointer values must be strings')
	return {
		'manifest_path': payload['manifest_path'],
		'manifest_sha256': payload['manifest_sha256'],
	}


def _rollback_periodic_refresh_pointer(
	*,
	pointer_path: Path,
	old_pointer: Mapping[str, str],
	new_pointer: Mapping[str, str],
) -> None:
	"""Restore the pre-refresh pointer, rejecting ambiguous public state."""
	if not pointer_path.is_file():
		raise RuntimeError(
			'periodic refresh rollback cannot verify the active target pointer'
		)
	current = _periodic_pointer(pointer_path)
	if current == dict(old_pointer):
		return
	if current != dict(new_pointer):
		raise RuntimeError(
			'periodic refresh rollback found a foreign active target pointer'
		)
	_json_write_atomic(pointer_path, dict(old_pointer))
	if _periodic_pointer(pointer_path) != dict(old_pointer):
		raise RuntimeError(
		'periodic refresh rollback did not restore the active target pointer'
	)


def _periodic_state_for_generation(  # noqa: PLR0913
	*,
	refresh_root: Path,
	config: Mapping[str, object],
	generations: list[dict[str, object]],
	active_manifest: Path,
	phase: str,
	last_completed_refresh_epoch: int,
	source_student_state_sha256: str | None,
) -> dict[str, object]:
	active = generations[-1]
	active_payload = load_periodic_refresh_generation(active_manifest)
	target = _periodic_nested_mapping(
		active_payload.get('canonical_multi_head_target_manifest'),
		'active canonical target reference',
	)
	next_epoch = next(
		(
			epoch
			for epoch in PERIODIC_REFRESH_SCHEDULE
			if epoch > last_completed_refresh_epoch
		),
		None,
	)
	chain_path = refresh_root / 'periodic_refresh_chain.json'
	return {
		'schema_version': 1,
		'active_generation_index': int(active['generation_index']),
		'active_generation_id': str(active['generation_id']),
		'active_generation_manifest_path': str(active_manifest.resolve()),
		'active_generation_manifest_sha256': str(active['manifest_sha256']),
		'active_generation_content_sha256': str(
			active['generation_content_sha256']
		),
		'active_target_manifest_path': str(Path(str(target['path'])).resolve()),
		'active_target_manifest_sha256': str(target['sha256']),
		'periodic_refresh_chain_path': str(chain_path.resolve()),
		'periodic_refresh_chain_sha256': file_sha256(chain_path),
		'last_completed_refresh_epoch': last_completed_refresh_epoch,
		'next_scheduled_refresh_epoch': next_epoch,
		'refresh_phase': phase,
		'source_student_state_sha256': source_student_state_sha256,
		'fixed_preprocessing_hmm_identity_sha256': _periodic_fixed_identity_hash(
			config
		),
		'generations': [
			_periodic_state_generation_record(record) for record in generations
		],
	}


def _periodic_state_with_phase(
	state: Mapping[str, object],
	*,
	phase: str,
) -> dict[str, object]:
	updated = dict(state)
	updated['refresh_phase'] = phase
	return updated


def _initialize_periodic_refresh_state(  # noqa: C901, PLR0912, PLR0915
	*,
	config: Mapping[str, object],
	output_root: Path,
	quarantine_invalid: bool = False,
) -> dict[str, object]:
	initial_config, refresh_root = _periodic_initial_config(config)
	_validate_periodic_refresh_runtime_ownership(
		initial_config=initial_config,
		refresh_root=refresh_root,
		output_root=output_root,
	)
	initial_manifest = (
		initial_config.output_generation_dir / 'refresh_generation.json'
	).resolve()
	expected_identity = _periodic_initial_request_identity(initial_config)
	if initial_config.output_generation_dir.exists() or (
		initial_config.output_generation_dir.is_symlink()
	):
		# Validate a complete generation before asking the producer to do any
		# work.  A partial or foreign directory remains fail-closed unless the
		# caller explicitly requested its recoverable quarantine.
		try:
			load_periodic_refresh_generation(
				initial_manifest,
				expected_identity=expected_identity,
			)
		except (OSError, TypeError, ValueError, KeyError):
			if not quarantine_invalid:
				raise
			quarantine_periodic_refresh_generation(
				initial_config.output_generation_dir
			)
			result = produce_initial_periodic_refresh_generation(initial_config)
			initial_manifest = result.manifest_path.resolve()
	else:
		result = produce_initial_periodic_refresh_generation(initial_config)
		initial_manifest = result.manifest_path.resolve()
	load_periodic_refresh_generation(
		initial_manifest,
		expected_identity=expected_identity,
	)
	initial_record = _periodic_generation_record(initial_manifest)
	chain_path = refresh_root / 'periodic_refresh_chain.json'
	fixed_hash = _periodic_fixed_identity_hash(config)
	expected_chain = _periodic_chain_payload(
		fixed_identity_hash=fixed_hash, generations=[initial_record]
	)
	if chain_path.is_file():
		try:
			chain_matches = _read_periodic_chain(chain_path) == expected_chain
		except (OSError, TypeError, ValueError, KeyError):
			if not quarantine_invalid:
				raise
			quarantine_periodic_refresh_generation(chain_path)
			chain_matches = False
		if not chain_matches:
			if chain_path.exists() and not quarantine_invalid:
				raise ValueError('existing periodic refresh chain is foreign or stale')
			if chain_path.exists():
				quarantine_periodic_refresh_generation(chain_path)
			_json_write_atomic(chain_path, expected_chain)
	elif chain_path.exists() or chain_path.is_symlink():
		if not quarantine_invalid:
			raise ValueError('existing periodic refresh chain is foreign or stale')
		quarantine_periodic_refresh_generation(chain_path)
		_json_write_atomic(chain_path, expected_chain)
	else:
		_json_write_atomic(chain_path, expected_chain)
	pointer_path = refresh_root / 'active_target_generation.json'
	initial_payload = load_periodic_refresh_generation(initial_manifest)
	initial_target = _periodic_nested_mapping(
		initial_payload.get('canonical_multi_head_target_manifest'),
		'initial canonical target reference',
	)
	expected_pointer = {
		'manifest_path': str(initial_manifest),
		'manifest_sha256': initial_record['manifest_sha256'],
	}
	if pointer_path.is_file():
		try:
			pointer_matches = _periodic_pointer(pointer_path) == expected_pointer
		except (OSError, TypeError, ValueError, KeyError):
			if not quarantine_invalid:
				raise
			quarantine_periodic_refresh_generation(pointer_path)
			pointer_matches = False
		if not pointer_matches:
			if pointer_path.exists() and not quarantine_invalid:
				raise ValueError('existing active target pointer is foreign or stale')
			if pointer_path.exists():
				quarantine_periodic_refresh_generation(pointer_path)
			_json_write_atomic(pointer_path, expected_pointer)
	elif pointer_path.exists() or pointer_path.is_symlink():
		if not quarantine_invalid:
			raise ValueError('existing active target pointer is foreign or stale')
		quarantine_periodic_refresh_generation(pointer_path)
		_json_write_atomic(pointer_path, expected_pointer)
	else:
		_json_write_atomic(pointer_path, expected_pointer)
	state = _periodic_state_for_generation(
		refresh_root=refresh_root,
		config=config,
		generations=[initial_record],
		active_manifest=initial_manifest,
		phase='training',
		last_completed_refresh_epoch=0,
		source_student_state_sha256=None,
	)
	if Path(str(initial_target['path'])).resolve() != Path(
		state['active_target_manifest_path']
	).resolve():
		raise ValueError('initial periodic target binding is inconsistent')
	_append_target_refresh_event(
		output_root,
		{
			'event_type': 'generation',
			'status': 'complete',
			'phase': 'initial_bind',
			'generation_index': 0,
			'generation_id': initial_record['generation_id'],
			'output_generation_manifest_path': str(initial_manifest),
			'output_generation_manifest_sha256': initial_record['manifest_sha256'],
			'active_target_manifest_path': state['active_target_manifest_path'],
			'active_target_manifest_sha256': state['active_target_manifest_sha256'],
			'global_step_before': 0,
			'global_step_after': 0,
		},
	)
	return state


def _validate_periodic_refresh_runtime_ownership(
	*,
	initial_config: InitialPeriodicRefreshConfig,
	refresh_root: Path,
	output_root: Path,
) -> None:
	resolved_refresh_root = refresh_root.resolve(strict=False)
	resolved_output_root = output_root.resolve(strict=False)
	try:
		relative = resolved_refresh_root.relative_to(resolved_output_root)
	except ValueError as exc:
		raise ValueError(
			'pseudo_target_refresh.generation_root must be a strict child of '
			'paths.output_root'
		) from exc
	if relative == Path():
		raise ValueError(
			'pseudo_target_refresh.generation_root must be a strict child of '
			'paths.output_root'
		)

	source_artifacts = [
		initial_config.initial_hard_target_manifest,
		initial_config.clustering_config,
		initial_config.source_embedding_metadata,
	]
	for artifact in initial_config.initial_hmm_artifacts:
		source_artifacts.extend(
			(
				artifact.centers,
				artifact.hmm_model,
				artifact.preprocessor,
				artifact.metadata,
			)
		)
		if artifact.residualizer is not None:
			source_artifacts.append(artifact.residualizer)
	for source_artifact in source_artifacts:
		try:
			source_artifact.path.resolve(strict=False).relative_to(
				resolved_refresh_root
			)
		except ValueError:
			continue
		raise ValueError(
			'pseudo_target_refresh initial source artifacts must be outside '
			'generation_root'
		)


def _periodic_embedding_extraction_config(
	*,
	config: Mapping[str, object],
	output_dir: Path,
	source_embedding_metadata: HashedArtifactReference,
) -> dict[str, object]:
	metadata = json.loads(
		source_embedding_metadata.path.read_text(encoding='utf-8')
	)
	if not isinstance(metadata, Mapping):
		raise TypeError('source embedding metadata must be a JSON object')
	for key in ('window_size', 'overlap', 'output_dtype', 'min_token_valid_fraction'):
		if key not in metadata:
			raise ValueError(f'source embedding metadata is missing {key!r}')
	manifest_config = _mapping(config, 'manifests')
	manifest_input = _path_config(manifest_config, 'train')
	standard_manifests: dict[str, object] = {'input': str(manifest_input)}
	for key in ('path_list', 'train_path_list', 'input_path_list'):
		value = manifest_config.get(key)
		if value is not None:
			standard_manifests[key] = value
	embedding: dict[str, object] = {
		'window_size': metadata['window_size'],
		'overlap': metadata['overlap'],
		'output_dtype': metadata['output_dtype'],
		'average_chunk_size_x': metadata.get('average_chunk_size_x', 16),
		'batch_size': 1,
		'prefetch_queue_depth': 0,
		'amp': False,
		'amp_dtype': 'auto',
		'stage_timing': False,
		'min_token_valid_fraction': metadata['min_token_valid_fraction'],
		'preprocessing_cache': {
			'mode': 'off',
			'chunk_size_x': 16,
			'reuse': True,
			'cleanup': False,
		},
	}
	return {
		'manifests': standard_manifests,
		'embeddings': {
			'checkpoint': str(output_dir / 'unused_checkpoint.pt'),
			'output_dir': str(output_dir),
		},
		'embedding': embedding,
	}


def _module_training_flags(
	modules: tuple[torch.nn.Module, ...],
) -> tuple[tuple[torch.nn.Module, bool], ...]:
	return tuple((module, bool(module.training)) for module in modules)


def _restore_module_training_flags(
	flags: tuple[tuple[torch.nn.Module, bool], ...],
) -> None:
	for module, training in flags:
		module.train(training)


def _rng_state_hash(state: Mapping[str, object]) -> str:
	digest = hashlib.sha256()

	def update(value: object) -> None:
		if isinstance(value, Mapping):
			digest.update(b'mapping')
			for key in sorted(value, key=str):
				update(str(key))
				update(value[key])
			return
		if isinstance(value, torch.Tensor):
			tensor = value.detach().cpu().contiguous()
			digest.update(b'tensor')
			digest.update(str(tensor.dtype).encode('utf-8'))
			digest.update(repr(tuple(tensor.shape)).encode('ascii'))
			digest.update(tensor.numpy().tobytes())
			return
		if isinstance(value, np.ndarray):
			digest.update(b'ndarray')
			digest.update(str(value.dtype).encode('utf-8'))
			digest.update(repr(tuple(value.shape)).encode('ascii'))
			digest.update(np.ascontiguousarray(value).tobytes())
			return
		if isinstance(value, tuple | list):
			digest.update(b'tuple' if isinstance(value, tuple) else b'list')
			for item in value:
				update(item)
			return
		digest.update(type(value).__name__.encode('utf-8'))
		digest.update(repr(value).encode('utf-8'))

	update(state)
	return digest.hexdigest()


def _restore_captured_rng(state: Mapping[str, object]) -> None:
	restore_rng_state({'rng_state': state})


def _shutdown_strat_hmm_dataloader(
	dataloader: torch.utils.data.DataLoader,
) -> None:
	iterator = getattr(dataloader, '_iterator', None)
	shutdown = getattr(iterator, '_shutdown_workers', None)
	if callable(shutdown):
		shutdown()
	dataloader._iterator = None  # noqa: SLF001


def _set_dataloader_generator_state(
	dataloader: torch.utils.data.DataLoader,
	state: torch.Tensor,
) -> None:
	generator = getattr(dataloader, 'generator', None)
	if not isinstance(generator, torch.Generator):
		raise TypeError('periodic refresh dataloader must expose a torch.Generator')
	generator.set_state(state.cpu())


def _checkpoint_metrics(payload: Mapping[str, object]) -> dict[str, float]:
	metrics = payload.get('metrics', {})
	if not isinstance(metrics, Mapping):
		raise TypeError('checkpoint metrics must be a mapping')
	result: dict[str, float] = {}
	for key, value in metrics.items():
		if isinstance(value, bool) or not isinstance(value, (int, float)):
			continue
		if not math.isfinite(float(value)):
			raise ValueError('checkpoint metrics must be finite')
		result[str(key)] = float(value)
	return result


def _periodic_epoch_metrics_from_checkpoint(
	payload: Mapping[str, object],
) -> dict[str, float]:
	"""Recover epoch diagnostics without checkpoint-only training metadata."""
	metrics = _checkpoint_metrics(payload)
	for key in ('amp_enabled', 'trainable_parameter_count', 'frozen_parameter_count'):
		metrics.pop(key, None)
	return metrics


def _periodic_refresh_epoch_from_state(state: Mapping[str, object]) -> int:
	if state.get('refresh_phase') != 'refresh_required':
		raise ValueError('periodic refresh state does not require a refresh')
	epoch = state.get('next_scheduled_refresh_epoch')
	if isinstance(epoch, bool) or not isinstance(epoch, int):
		raise TypeError('periodic refresh state has no next scheduled epoch')
	if epoch not in PERIODIC_REFRESH_SCHEDULE:
		raise ValueError('periodic refresh state has an invalid scheduled epoch')
	return epoch


def _periodic_scheduled_epoch(epoch: int) -> bool:
	return epoch in PERIODIC_REFRESH_SCHEDULE


def _periodic_previous_centers(
	manifest_path: Path,
) -> tuple[PreviousCenterArtifact, ...]:
	payload = load_periodic_refresh_generation(manifest_path)
	centers = _periodic_nested_mapping(
		payload.get('centers'), 'previous generation centers'
	)
	result: list[PreviousCenterArtifact] = []
	for k in CANONICAL_KS:
		entry = _periodic_nested_mapping(
			centers.get(str(k)), f'previous generation k={k} centers'
		)
		after = _periodic_nested_mapping(
			entry.get('after'), f'previous generation k={k} after centers'
		)
		result.append(
			PreviousCenterArtifact(
				k=k,
				centers=HashedArtifactReference(
					path=Path(str(after['path'])),
					sha256=str(after['sha256']),
				),
			)
		)
	return tuple(result)


def _periodic_generation_lineage_is_exact(
	*,
	chain: Mapping[str, object],
	config: Mapping[str, object],
	expected_generations: list[dict[str, object]],
) -> None:
	if chain.get('fixed_preprocessing_hmm_identity_sha256') != (
		_periodic_fixed_identity_hash(config)
	):
		raise ValueError('periodic refresh chain fixed preprocessing identity drift')
	actual = chain.get('generations')
	if actual != expected_generations:
		raise ValueError('periodic refresh chain lineage does not match state')


def _activate_periodic_generation(  # noqa: C901, PLR0912, PLR0913
	*,
	config: Mapping[str, object],
	state: Mapping[str, object],
	refresh_root: Path,
	refresh_epoch: int,
	result_manifest: Path,
	source_student_state_sha256: str,
) -> dict[str, object]:
	old_generations = _periodic_chain_from_state(state)
	if not old_generations:
		raise ValueError('periodic refresh state has no active generation')
	old_active = old_generations[-1]
	old_index = int(old_active['generation_index'])
	generation_index = old_index + 1
	if generation_index > len(PERIODIC_REFRESH_SCHEDULE):
		raise ValueError('periodic refresh produced too many generations')
	if refresh_epoch != PERIODIC_REFRESH_SCHEDULE[generation_index - 1]:
		raise ValueError('periodic refresh epoch does not match generation index')
	new_record = _periodic_generation_record(result_manifest)
	if new_record['generation_index'] != generation_index:
		raise ValueError('periodic refresh generation index does not match state')
	active_payload = load_periodic_refresh_generation(result_manifest)
	if active_payload.get('source_student_state_sha256') != source_student_state_sha256:
		raise ValueError('periodic generation source student state hash mismatch')
	chain_path = refresh_root / 'periodic_refresh_chain.json'
	if not chain_path.is_file():
		raise FileNotFoundError(chain_path)
	chain = _read_periodic_chain(chain_path)
	actual_generations = chain['generations']
	if not isinstance(actual_generations, list):
		raise TypeError('periodic refresh chain generations must be a list')
	pointer_path = refresh_root / 'active_target_generation.json'
	if not pointer_path.is_file():
		raise FileNotFoundError(pointer_path)
	pointer = _periodic_pointer(pointer_path)
	old_pointer = {
		'manifest_path': str(Path(str(old_active['manifest_path'])).resolve()),
		'manifest_sha256': str(old_active['manifest_sha256']),
	}
	new_pointer = {
		'manifest_path': str(result_manifest.resolve()),
		'manifest_sha256': str(new_record['manifest_sha256']),
	}
	if len(actual_generations) == len(old_generations):
		if pointer != old_pointer:
			raise ValueError(
				'active target pointer advanced before the refresh chain was appended'
			)
		_periodic_generation_lineage_is_exact(
			chain=chain, config=config, expected_generations=old_generations
		)
		_json_write_atomic(
			chain_path,
			_periodic_chain_payload(
				fixed_identity_hash=_periodic_fixed_identity_hash(config),
				generations=[*old_generations, new_record],
			),
		)
	elif len(actual_generations) == len(old_generations) + 1:
		if pointer not in (old_pointer, new_pointer):
			raise ValueError('active target pointer is foreign or out of chain order')
		_periodic_generation_lineage_is_exact(
			chain=chain,
			config=config,
			expected_generations=[*old_generations, new_record],
		)
	else:
		raise ValueError(
			'periodic refresh chain contains an unexpected future generation'
		)
	if pointer == old_pointer:
		_json_write_atomic(pointer_path, new_pointer)
	elif pointer != new_pointer:
		raise ValueError('active target pointer is foreign or out of chain order')
	return _periodic_state_for_generation(
		refresh_root=refresh_root,
		config=config,
		generations=[*old_generations, new_record],
		active_manifest=result_manifest,
		phase='refresh_complete',
		last_completed_refresh_epoch=refresh_epoch,
		source_student_state_sha256=source_student_state_sha256,
	)


def _load_or_produce_periodic_refresh_generation(
	config: PeriodicRefreshConfig,
	*,
	quarantine_invalid: bool = False,
) -> Path:
	"""Validate/reuse a generation, with explicit recovery for owned invalid output."""
	expected_identity = _periodic_request_identity(config)
	output_root = Path(config.output_generation_dir)
	manifest_path = output_root / 'refresh_generation.json'
	if output_root.exists() or output_root.is_symlink():
		try:
			load_periodic_refresh_generation(
				manifest_path,
				expected_identity=expected_identity,
			)
		except (OSError, TypeError, ValueError, KeyError):
			if not quarantine_invalid:
				raise
			quarantine_periodic_refresh_generation(output_root)
		else:
			return manifest_path.resolve()
	result = produce_periodic_refresh_generation(config)
	manifest_path = result.manifest_path.resolve()
	load_periodic_refresh_generation(
		manifest_path,
		expected_identity=expected_identity,
	)
	return manifest_path


def _perform_periodic_refresh(  # noqa: C901, PLR0912, PLR0913, PLR0915
	*,
	config: Mapping[str, object],
	output_root: Path,
	manifests: object,
	dataset_kwargs: Mapping[str, object],
	components: StratHmmCenterTraceMaskedComponents,
	device: torch.device,
	dataloader: torch.utils.data.DataLoader,
	state: Mapping[str, object],
	refresh_epoch: int,
	global_step: int,
	quarantine_invalid: bool = False,
) -> tuple[object, torch.utils.data.DataLoader, dict[str, object]]:
	refresh = _periodic_refresh_mapping(config)
	refresh_root = _path_config(refresh, 'generation_root').resolve()
	old_manifest = Path(str(state['active_generation_manifest_path'])).resolve()
	generation_index = int(state['active_generation_index']) + 1
	generation_id = (
		f'refresh_{generation_index:04d}_epoch{refresh_epoch:03d}'
	)
	generation_dir = refresh_root / 'generations' / generation_id
	embedding_dir = refresh_root / 'embeddings' / generation_id
	initial_artifacts, clustering_config, source_metadata, initial_target = (
		_periodic_initial_artifacts(config)
	)
	previous_centers = _periodic_previous_centers(old_manifest)
	student_hash = _state_dict_sha256(components.student.state_dict())
	if components.teacher is None:
		raise TypeError('periodic refresh requires a teacher module')
	pointer_path = refresh_root / 'active_target_generation.json'
	if not pointer_path.is_file():
		raise FileNotFoundError(pointer_path)
	old_pointer = {
		'manifest_path': str(old_manifest),
		'manifest_sha256': str(state['active_generation_manifest_sha256']),
	}
	pointer_before = _periodic_pointer(pointer_path)
	old_loader_rng = _dataloader_generator_state(dataloader)
	_shutdown_strat_hmm_dataloader(dataloader)
	modules = (
		components.student,
		components.teacher,
		components.heads,
		components.replacement_token,
	)
	training_flags = _module_training_flags(modules)
	rng_before = capture_rng_state()
	activation_manifest: Path | None = None
	new_pointer: dict[str, str] | None = None
	new_dataset: object | None = None
	new_dataloader: torch.utils.data.DataLoader | None = None
	new_state: dict[str, object] | None = None
	try:
		_append_target_refresh_event(
			output_root,
			{
				'event_type': 'refresh',
				'status': 'start',
				'generation_index': generation_index,
				'generation_id': generation_id,
				'refresh_epoch': refresh_epoch,
				'source_student_state_sha256': student_hash,
				'student_state_sha256': student_hash,
				'optimizer_state_sha256': _optimizer_state_sha256(
					components.optimizer
				),
				'active_generation_id_before': state['active_generation_id'],
				'global_step_before': global_step,
				'rng_before_sha256': _rng_state_hash(rng_before),
			},
		)
		extraction_config = _periodic_embedding_extraction_config(
			config=config,
			output_dir=embedding_dir,
			source_embedding_metadata=source_metadata,
		)
		extract_embeddings_from_loaded_model(
			components.student,
			extraction_config,
			embedding_dir,
			student_hash,
			checkpoint_config=components.mae_checkpoint_config,
			reuse=True,
			device=device,
		)
		if _state_dict_sha256(components.student.state_dict()) != student_hash:
			raise RuntimeError(  # noqa: TRY301
				'periodic embedding extraction changed student state'
			)
		descriptor = embedding_dir / REFRESH_EXTRACTION_DESCRIPTOR_NAME
		generation_config = PeriodicRefreshConfig(
			generation_index=generation_index,
			refresh_after_epoch=refresh_epoch,
			source_student_state_sha256=student_hash,
			previous_generation_manifest=_artifact_reference(old_manifest),
			current_embedding_descriptor=_artifact_reference(descriptor),
			initial_hard_target_manifest=initial_target,
			initial_hmm_artifacts=initial_artifacts,
			clustering_config=clustering_config,
			source_embedding_metadata=source_metadata,
			previous_centers=previous_centers,
			output_generation_dir=generation_dir,
			target_policy=_periodic_target_policy(initial_target.path),
			iterations=2,
		)
		if quarantine_invalid:
			activation_manifest = _load_or_produce_periodic_refresh_generation(
				generation_config,
				quarantine_invalid=True,
			)
		else:
			activation_manifest = _load_or_produce_periodic_refresh_generation(
				generation_config
			)
		new_pointer = {
			'manifest_path': str(activation_manifest),
			'manifest_sha256': file_sha256(activation_manifest),
		}
		if pointer_before not in (old_pointer, new_pointer):
			raise ValueError(  # noqa: TRY301
				'active target pointer is foreign or out of refresh order'
			)
		new_state = _activate_periodic_generation(
			config=config,
			state=state,
			refresh_root=refresh_root,
			refresh_epoch=refresh_epoch,
			result_manifest=activation_manifest,
			source_student_state_sha256=student_hash,
		)
		new_dataset, new_dataloader = _build_strat_hmm_dataset_and_dataloader(
			config=config,
			manifests=manifests,
			dataset_kwargs=dataset_kwargs,
			device=device,
			target_manifest_override=Path(
				str(new_state['active_target_manifest_path'])
			),
		)
		_set_dataloader_generator_state(new_dataloader, old_loader_rng)
		_restore_captured_rng(rng_before)
		_restore_module_training_flags(training_flags)
		rng_after = capture_rng_state()
		if _rng_state_hash(rng_after) != _rng_state_hash(rng_before):
			raise RuntimeError(  # noqa: TRY301
				'periodic refresh changed training RNG state'
			)
		if _module_training_flags(modules) != training_flags:
			raise RuntimeError(  # noqa: TRY301
				'periodic refresh changed module training flags'
			)
		new_payload = load_periodic_refresh_generation(
			Path(str(new_state['active_generation_manifest_path']))
		)
		diagnostics_ref = new_payload.get('refresh_diagnostics')
		diagnostics = None
		if isinstance(diagnostics_ref, Mapping):
			diagnostics_path = Path(str(diagnostics_ref['path']))
			if diagnostics_path.is_file():
				diagnostics = json.loads(
					diagnostics_path.read_text(encoding='utf-8')
				)
		_append_target_refresh_event(
			output_root,
			{
				'event_type': 'refresh',
				'status': 'complete',
				'generation_index': generation_index,
				'generation_id': new_state['active_generation_id'],
				'refresh_epoch': refresh_epoch,
				'source_student_state_sha256': student_hash,
				'student_state_sha256': _state_dict_sha256(
					components.student.state_dict()
				),
				'optimizer_state_sha256': _optimizer_state_sha256(
					components.optimizer
				),
				'output_generation_manifest_path': new_state[
					'active_generation_manifest_path'
				],
				'output_generation_manifest_sha256': new_state[
					'active_generation_manifest_sha256'
				],
				'active_target_manifest_path': new_state[
					'active_target_manifest_path'
				],
				'active_target_manifest_sha256': new_state[
					'active_target_manifest_sha256'
				],
				'global_step_before': global_step,
				'global_step_after': global_step,
				'rng_before_sha256': _rng_state_hash(rng_before),
				'rng_after_restore_sha256': _rng_state_hash(rng_after),
				'diagnostics': diagnostics,
			},
		)
	except BaseException as exc:
		cleanup_errors: list[BaseException] = []
		if new_dataloader is not None:
			try:
				_shutdown_strat_hmm_dataloader(new_dataloader)
			except BaseException as cleanup_exc:  # noqa: BLE001
				cleanup_errors.append(cleanup_exc)
		if pointer_before == old_pointer and new_pointer is not None:
			try:
				_rollback_periodic_refresh_pointer(
					pointer_path=pointer_path,
					old_pointer=old_pointer,
					new_pointer=new_pointer,
				)
			except BaseException as cleanup_exc:  # noqa: BLE001
				cleanup_errors.append(cleanup_exc)
		try:
			_restore_captured_rng(rng_before)
		except BaseException as cleanup_exc:  # noqa: BLE001
			cleanup_errors.append(cleanup_exc)
		try:
			_restore_module_training_flags(training_flags)
		except BaseException as cleanup_exc:  # noqa: BLE001
			cleanup_errors.append(cleanup_exc)
		try:
			_append_target_refresh_event(
				output_root,
				{
					'event_type': 'refresh',
					'status': 'failure',
					'generation_index': generation_index,
					'generation_id': generation_id,
					'refresh_epoch': refresh_epoch,
					'source_student_state_sha256': student_hash,
					'student_state_sha256': _state_dict_sha256(
						components.student.state_dict()
					),
					'optimizer_state_sha256': _optimizer_state_sha256(
						components.optimizer
					),
					'active_generation_id_after': (
						generation_id
						if pointer_before != old_pointer
						else state['active_generation_id']
					),
					'global_step_before': global_step,
					'rng_before_sha256': _rng_state_hash(rng_before),
					'rng_after_restore_sha256': _rng_state_hash(
						capture_rng_state()
					),
					'error_type': type(exc).__name__,
					'error': str(exc),
				},
			)
		except BaseException as cleanup_exc:  # noqa: BLE001
			cleanup_errors.append(cleanup_exc)
		if cleanup_errors:
			details = '; '.join(
				f'{type(error).__name__}: {error}' for error in cleanup_errors
			)
			raise RuntimeError(
				f'periodic refresh failed and recovery was incomplete: {details}'
			) from exc
		raise
	if new_dataset is None or new_dataloader is None or new_state is None:
		raise RuntimeError('periodic refresh completed without a new target loader')
	return new_dataset, new_dataloader, new_state


def _save_periodic_checkpoint(  # noqa: PLR0913
	*,
	output_root: Path,
	components: StratHmmCenterTraceMaskedComponents,
	config: Mapping[str, object],
	metrics: Mapping[str, float],
	global_step: int,
	epoch: int,
	checkpoint_kind: Literal['step', 'epoch', 'refresh'],
	batch_index: int | None,
	dataloader: torch.utils.data.DataLoader,
	amp_enabled: bool,
	scaler: torch.amp.GradScaler | None,
	control_identity: Mapping[str, object] | None,
	checkpoint_selection: Mapping[str, object] | None,
	target_refresh_state: Mapping[str, object],
	epoch_metrics_state: Mapping[str, object] | None = None,
	epoch_start_dataloader_rng_state: torch.Tensor | None = None,
) -> object:
	if checkpoint_kind == 'step':
		if epoch_start_dataloader_rng_state is None or batch_index is None:
			raise ValueError('periodic step checkpoint requires batch position state')
		rng_state = _rng_state_for_step_checkpoint(
			dataloader=dataloader,
			epoch_start_dataloader_rng_state=epoch_start_dataloader_rng_state,
			batch_index=batch_index,
		)
	else:
		rng_state = _rng_state_with_dataloader(dataloader)
	return save_strat_hmm_rolling_checkpoint(
		output_root,
		student=components.student,
		head=components.heads,
		spatial_context=components.replacement_token,
		optimizer=components.optimizer,
		epoch=epoch,
		mae_config=components.mae_checkpoint_config,
		stratigraphy_config=config,
		metrics=dict(metrics),
		global_step=global_step,
		checkpoint_kind=checkpoint_kind,
		batch_index=batch_index,
		amp_enabled=amp_enabled,
		scaler=scaler,
		rng_state=rng_state,
		trainability_summary=_trainability_summary_payload(
			components.trainability_summary
		),
		control_identity=control_identity,
		best_score=None,
		checkpoint_selection=checkpoint_selection,
		target_refresh_state=target_refresh_state,
		epoch_metrics_state=epoch_metrics_state,
	)


def inspect_strat_hmm_pretext_plan(
	config: Mapping[str, object],
) -> dict[str, object]:
	"""Validate the closed center-trace action plan without mutating state."""
	if not isinstance(config.get('spatial_context'), Mapping):
		raise TypeError('center-trace dry-run requires spatial_context')
	train_config = _mapping(config, 'train')
	data_config = _mapping(config, 'data')
	model_config = _mapping(config, 'model')
	pseudo_config = _mapping(config, 'pseudo_targets')
	is_periodic_refresh = _is_periodic_refresh_config(config)
	device = _resolve_device(train_config)
	seed = _int_config(train_config, 'seed', 42)
	manifests = read_manifest_json(_path_config(_mapping(config, 'manifests'), 'train'))
	dataset_kwargs = {
		'local_crop_size_xyz': _xyz_config(data_config, 'local_crop_size'),
		'patch_size_xyz': _xyz_config(model_config, 'patch_size'),
		'seed': seed,
		'samples_per_epoch': _int_config(train_config, 'samples_per_epoch', 1),
		'zero_mask': _zero_mask_from_config(config),
		'min_valid_fraction': _float_config(data_config, 'min_valid_fraction', 0.0),
		'max_resample_attempts': _int_config(data_config, 'max_resample_attempts', 16),
		'normalized_clip_abs': _optional_float_config(
			data_config, 'normalized_clip_abs'
		),
		'amplitude_agc': data_config.get('amplitude_agc'),
		'finite_check_mode': cast(
			'FiniteCheckMode', data_config.get('finite_check_mode', 'strict')
		),
		'min_confidence': _float_config(pseudo_config, 'min_confidence', 0.0),
	}
	periodic_initial: InitialPeriodicRefreshConfig | None = None
	periodic_root: Path | None = None
	if is_periodic_refresh:
		periodic_initial, periodic_root = _periodic_initial_config(config)
	dataset, dataloader = _build_strat_hmm_dataset_and_dataloader(
		config=config,
		manifests=manifests,
		dataset_kwargs=dataset_kwargs,
		device=device,
		target_manifest_override=(
			None
			if periodic_initial is None
			else periodic_initial.initial_hard_target_manifest.path
		),
	)
	# Component construction checks the route and optimizer partition.  Isolate
	# its random initialization so inspection has no observable RNG effect.
	with torch.random.fork_rng(
		devices=[device] if device.type == 'cuda' and torch.cuda.is_available() else []
	):
		components = build_strat_hmm_components(config, device=device)
	if not isinstance(components, StratHmmCenterTraceMaskedComponents):
		raise TypeError('center-trace dry-run did not dispatch center components')
	control_identity = _strat_hmm_control_identity(config)
	if is_periodic_refresh:
		if periodic_initial is None or periodic_root is None:
			raise AssertionError('periodic dry-run initial contract was not built')
		initial_artifacts = {
			str(artifact.k): {
				'centers': {
					'path': str(artifact.centers.path),
					'sha256': artifact.centers.sha256,
				},
				'hmm_model': {
					'path': str(artifact.hmm_model.path),
					'sha256': artifact.hmm_model.sha256,
				},
				'preprocessor': {
					'path': str(artifact.preprocessor.path),
					'sha256': artifact.preprocessor.sha256,
				},
				'metadata': {
					'path': str(artifact.metadata.path),
					'sha256': artifact.metadata.sha256,
				},
			}
			for artifact in periodic_initial.initial_hmm_artifacts
		}
		return {
			'route': 'center_trace_masked_periodic_hmm_refresh',
			'target_representation': 'hard_viterbi_labels_v1',
			'dataset': type(dataset).__name__,
			'dataloader': type(dataloader).__name__,
			'components': type(components).__name__,
			'epoch': 'train_strat_hmm_center_trace_masked_one_epoch',
			'refresh_epochs': list(PERIODIC_REFRESH_SCHEDULE),
			'refresh_schedule': list(PERIODIC_REFRESH_SCHEDULE),
			'refresh_schedule_semantics': PERIODIC_REFRESH_SCHEDULE_SEMANTICS,
			'refresh_count': len(PERIODIC_REFRESH_SCHEDULE),
			'generation_directories': [
				str(
					periodic_root
					/ 'generations'
					/ (
						'refresh_0000_initial'
						if index == 0
						else f'refresh_{index:04d}_epoch{epoch:03d}'
					)
				)
				for index, epoch in enumerate((0, *PERIODIC_REFRESH_SCHEDULE))
			],
			'initial_target': {
				'path': str(periodic_initial.initial_hard_target_manifest.path),
				'sha256': periodic_initial.initial_hard_target_manifest.sha256,
			},
			'initial_hmm_artifacts': initial_artifacts,
			'initial_hmm_identities': initial_artifacts,
			'fixed_preprocessing_policy': PERIODIC_REFRESH_PREPROCESSING_POLICY,
			'fixed_preprocessing_hmm_identity_sha256': _periodic_fixed_identity_hash(
				config
			),
			'expected_generation_count': len(PERIODIC_REFRESH_SCHEDULE) + 1,
			'generation_count': len(PERIODIC_REFRESH_SCHEDULE) + 1,
			'checkpoint_schema': 8,
			'checkpoint_selection': PERIODIC_REFRESH_CHECKPOINT_SELECTION_POLICY,
			'checkpoint_selection_policy': PERIODIC_REFRESH_CHECKPOINT_SELECTION_POLICY,
			'model_tag': _mapping(config, 'identity').get('model_tag'),
			'control_identity_validated': control_identity is not None,
			'samples_per_epoch': len(dataset),
			'outputs_written': False,
		}
	return {
		'route': 'center_trace_masked_hard_multi_head',
		'target_representation': 'hard_viterbi_labels_v1',
		'dataset': type(dataset).__name__,
		'dataloader': type(dataloader).__name__,
		'components': type(components).__name__,
		'epoch': 'train_strat_hmm_center_trace_masked_one_epoch',
		'checkpoint_schema': 7,
		'model_tag': _mapping(config, 'identity').get('model_tag'),
		'control_identity_validated': control_identity is not None,
		'samples_per_epoch': len(dataset),
		'outputs_written': False,
	}


def _append_multi_head_epoch_metrics(
	*,
	output_root: Path,
	epoch: int,
	global_step: int,
	metrics: Mapping[str, float],
) -> None:
	"""Append finite multi-head epoch diagnostics without duplicating resumes."""
	if not all(math.isfinite(float(value)) for value in metrics.values()):
		raise ValueError('multi-head epoch metrics must be finite')
	path = output_root / 'multi_head_epoch_metrics.csv'
	fieldnames = ['epoch', 'global_step', *sorted(metrics)]
	if path.is_file():
		with path.open(newline='', encoding='utf-8') as handle:
			rows = list(csv.DictReader(handle))
		if rows and list(rows[0]) != fieldnames:
			raise ValueError('multi-head epoch metrics schema changed during resume')
		if any(int(row['epoch']) == epoch for row in rows):
			return
	else:
		rows = []
	row = {
		'epoch': str(epoch),
		'global_step': str(global_step),
		**{key: str(value) for key, value in metrics.items()},
	}
	with path.open('a', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		if not rows:
			writer.writeheader()
		writer.writerow(row)


def _preflight_only_output_root(output_root: Path) -> bool:
	"""Permit a validated control preflight without permitting stale outputs."""
	if not output_root.is_dir():
		return False
	entries = tuple(output_root.iterdir())
	return len(entries) == 1 and entries[0].name == 'preflight' and entries[0].is_dir()


def _with_initial_parameter_identities(
	control_identity: Mapping[str, object],
	*,
	student: torch.nn.Module,
	head: torch.nn.Module,
	replacement_token: torch.nn.Module | None = None,
) -> dict[str, object]:
	"""Record pre-optimization state hashes for control freeze validation."""
	result = dict(control_identity)
	result['initial_parameter_sha256'] = {
		'student_trainable': _parameter_sha256(
			(name, parameter)
			for name, parameter in student.named_parameters()
			if parameter.requires_grad
		),
		'prototype_head': _parameter_sha256(head.named_parameters()),
	}
	result['initial_state_sha256'] = {
		'student': _state_dict_sha256(student.state_dict()),
		'head': _state_dict_sha256(head.state_dict()),
	}
	if replacement_token is not None:
		result['initial_state_sha256']['spatial_context'] = _state_dict_sha256(
			replacement_token.state_dict()
		)
	return result


def _parameter_sha256(
	parameters: Iterable[tuple[str, torch.Tensor]],
) -> dict[str, str]:
	"""Hash parameter names, shapes, dtypes, and raw tensor bytes."""
	result: dict[str, str] = {}
	for name, parameter in parameters:
		value = parameter.detach().cpu().contiguous()
		digest = hashlib.sha256()
		digest.update(name.encode('utf-8'))
		digest.update(str(value.dtype).encode('utf-8'))
		digest.update(str(tuple(value.shape)).encode('utf-8'))
		digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
		result[name] = digest.hexdigest()
	return result


def _state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
	"""Hash a complete initial module state, including frozen tensors."""
	digest = hashlib.sha256()
	for name in sorted(state_dict):
		value = state_dict[name].detach().cpu().contiguous()
		digest.update(name.encode('utf-8'))
		digest.update(str(value.dtype).encode('utf-8'))
		digest.update(str(tuple(value.shape)).encode('utf-8'))
		digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
	return digest.hexdigest()


def _optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
	"""Hash optimizer state deterministically for refresh-continuity events."""
	digest = hashlib.sha256()
	_update_optimizer_hash(digest, optimizer.state_dict())
	return digest.hexdigest()


def _update_optimizer_hash(digest: hashlib._Hash, value: object) -> None:
	if isinstance(value, torch.Tensor):
		tensor = value.detach().cpu().contiguous()
		digest.update(b'tensor')
		digest.update(str(tensor.dtype).encode('utf-8'))
		digest.update(str(tuple(tensor.shape)).encode('utf-8'))
		digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
		return
	if isinstance(value, Mapping):
		digest.update(b'mapping')
		for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
			_update_optimizer_hash(digest, key)
			_update_optimizer_hash(digest, value[key])
		return
	if isinstance(value, list | tuple):
		digest.update(b'list' if isinstance(value, list) else b'tuple')
		for child in value:
			_update_optimizer_hash(digest, child)
		return
	digest.update(type(value).__name__.encode('utf-8'))
	digest.update(repr(value).encode('utf-8'))


__all__ = ['inspect_strat_hmm_pretext_plan', 'run_strat_hmm_pretext_training']
