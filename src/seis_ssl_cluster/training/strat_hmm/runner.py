"""Runner for stratigraphic HMM pretext training."""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import torch

from seis_ssl_cluster.data import (
	NopimsStratMultiHeadPosteriorDataset,
	NopimsStratMultiHeadTargetDataset,
	NopimsStratPseudoTargetDataset,
	load_strat_multi_head_lateral_target_manifest_adapter,
	load_strat_multi_head_xy_neighbor_consensus_target_manifest_adapter,
	load_strat_multi_head_xy_neighbor_unanimous_target_manifest_adapter,
	read_manifest_json,
)
from seis_ssl_cluster.stratigraphy import (
	discover_pseudo_target_inputs,
)
from seis_ssl_cluster.training.checkpoint import load_checkpoint
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
) -> Path:
	"""Run strat HMM pretext training from ``config``.

	Both single-head and multi-head runs use the same rolling checkpoint contract.
	"""
	train_config = _mapping(config, 'train')
	paths_config = _mapping(config, 'paths')
	data_config = _mapping(config, 'data')
	model_config = _mapping(config, 'model')
	pseudo_config = _mapping(config, 'pseudo_targets')
	is_multi_head = 'spec' in _mapping(config, 'head')
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
	if is_multi_head:
		if target_representation == 'ordered_path_state_posterior_v1':
			posterior_dataset_kwargs = {
				key: value
				for key, value in dataset_kwargs.items()
				if key != 'min_confidence'
			}
			dataset = NopimsStratMultiHeadPosteriorDataset(
				manifests,
				_path_config(pseudo_config, 'manifest'),
				**posterior_dataset_kwargs,
			)
			dataloader = build_strat_multi_head_posterior_dataloader(
				dataset,
				batch_size=_int_config(train_config, 'batch_size', 1),
				num_workers=_int_config(train_config, 'num_workers', 0),
				shuffle=_bool_config(train_config, 'shuffle', default=True),
				seed=seed,
				device=device,
			)
		elif target_representation in {
			'hard_viterbi_labels_v1',
			'lateral_mean_field_hard_labels_v1',
			'xy_neighbor_consensus_hard_labels_v1',
			'xy_neighbor_unanimous_hard_labels_v1',
		}:
			target_manifest: object = _path_config(pseudo_config, 'manifest')
			if target_representation == 'lateral_mean_field_hard_labels_v1':
				target_manifest = load_strat_multi_head_lateral_target_manifest_adapter(
					target_manifest
				).target_manifest
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
				manifests,
				target_manifest,
				**dataset_kwargs,
			)
			dataloader = build_strat_multi_head_target_dataloader(
				dataset,
				batch_size=_int_config(train_config, 'batch_size', 1),
				num_workers=_int_config(train_config, 'num_workers', 0),
				shuffle=_bool_config(train_config, 'shuffle', default=True),
				seed=seed,
				device=device,
			)
		else:
			raise ValueError(
				'unsupported multi-head target representation: '
				f'{target_representation!r}'
			)
	else:
		pseudo_inputs = discover_pseudo_target_inputs(
			_path_config(pseudo_config, 'input_dir'),
			k=_int_config(pseudo_config, 'k', 1),
		)
		dataset = NopimsStratPseudoTargetDataset(
			manifests, pseudo_inputs, **dataset_kwargs
		)
		dataloader = build_strat_pseudo_target_dataloader(
			dataset,
			batch_size=_int_config(train_config, 'batch_size', 1),
			num_workers=_int_config(train_config, 'num_workers', 0),
			shuffle=_bool_config(train_config, 'shuffle', default=True),
			seed=seed,
			device=device,
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
	if resume is not None:
		if is_multi_head:
			recover_strat_hmm_rolling_checkpoint(Path(resume).parent)
		payload = load_checkpoint(resume, map_location=device)
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
		_restore_dataloader_generator_state(payload=payload, dataloader=dataloader)
		if resume_state.skip_batches >= len(dataloader):
			resume_state = StratHmmResumeState(
				start_epoch=resume_state.start_epoch + 1,
				global_step=resume_state.global_step,
				skip_batches=0,
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
	best_score = _load_existing_best_score(output_root) if resume is not None else None
	checkpoint_selection: Mapping[str, object] | None = None
	if resume is not None and is_multi_head:
		checkpoint_selection = load_checkpoint(resume, map_location='cpu').get(
			'checkpoint_selection'
		)
		if not isinstance(checkpoint_selection, Mapping):
			raise TypeError('multi-head resume checkpoint is missing selection history')
		best_score = float(
			selected_checkpoint_selection_event(checkpoint_selection)['loss']
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
						dataloader=dataloader,
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
						dataloader=dataloader,
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
						dataloader=dataloader,
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
	device = _resolve_device(train_config)
	seed = _int_config(train_config, 'seed', 42)
	manifests = read_manifest_json(_path_config(_mapping(config, 'manifests'), 'train'))
	dataset = NopimsStratMultiHeadTargetDataset(
		manifests,
		_path_config(pseudo_config, 'manifest'),
		local_crop_size_xyz=_xyz_config(data_config, 'local_crop_size'),
		patch_size_xyz=_xyz_config(model_config, 'patch_size'),
		seed=seed,
		samples_per_epoch=_int_config(train_config, 'samples_per_epoch', 1),
		zero_mask=_zero_mask_from_config(config),
		min_valid_fraction=_float_config(data_config, 'min_valid_fraction', 0.0),
		max_resample_attempts=_int_config(data_config, 'max_resample_attempts', 16),
		normalized_clip_abs=_optional_float_config(
			data_config, 'normalized_clip_abs'
		),
		amplitude_agc=data_config.get('amplitude_agc'),
		finite_check_mode=cast(
			'FiniteCheckMode', data_config.get('finite_check_mode', 'strict')
		),
		min_confidence=_float_config(pseudo_config, 'min_confidence', 0.0),
	)
	dataloader = build_strat_multi_head_target_dataloader(
		dataset,
		batch_size=_int_config(train_config, 'batch_size', 1),
		num_workers=_int_config(train_config, 'num_workers', 0),
		shuffle=_bool_config(train_config, 'shuffle', default=True),
		seed=seed,
		device=device,
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
		digest.update(value.view(torch.uint8).numpy().tobytes())
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
		digest.update(value.view(torch.uint8).numpy().tobytes())
	return digest.hexdigest()


__all__ = ['inspect_strat_hmm_pretext_plan', 'run_strat_hmm_pretext_training']
