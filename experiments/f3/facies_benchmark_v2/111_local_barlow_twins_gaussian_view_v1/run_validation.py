# ruff: noqa: INP001
"""Run one validation-only F3 job for a Gaussian-view Barlow candidate.

The canonical five-way source audit intentionally remains unchanged: it owns
the published five-source, 100+25-epoch comparison. This screen audits both a
candidate base checkpoint and its fixed 25-epoch top-block continuation, then
imports the canonical runner's mapping helpers so the downstream decoder,
inference, and validation evaluator follow the same v3 benchmark path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import cast

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_five_way import (
	F3FiveWayConfig,
	F3FiveWayModelSource,
	f3_lithology_five_way_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	f3_lithology_voxel_decoder_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_evaluation import (
	f3_lithology_voxel_evaluation_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_inference import (
	f3_lithology_voxel_inference_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology import (
	five_way_results,
	five_way_runner,
	five_way_sources,
)
from seis_ssl_cluster.f3.lithology.voxel_decoder_inference import (
	predict_f3_lithology_voxels,
)
from seis_ssl_cluster.f3.lithology.voxel_evaluation import (
	evaluate_f3_lithology_voxels,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout import (
	validate_f3_lithology_voxel_section_layout_condition,
)
from seis_ssl_cluster.training.random_checkpoint import (
	load_checkpoint_metadata_without_weights,
)
from seis_ssl_cluster.training.voxel_decoder.runner import (
	run_f3_lithology_voxel_decoder,
)

HORIZONTAL_VIEW_POLICY = 'horizontal_flip_gaussian_noise_v1'
IDENTITY_VIEW_POLICY = 'identity_gaussian_noise_v1'
LOCAL_BARLOW_METHOD = 'local_barlow_twins_3d'
LOCAL_PAIRS_PER_CROP = 128
HORIZONTAL_FLIP_PROBABILITY = 0.5
VALIDATION_AGGREGATION_UNIT = 'unique_validation_voxel'
AUDIT_NAME = 'candidate_source_audit.json'
CHECKPOINT_AUDIT_SCHEMA_VERSION = 3
PROTOCOL_LOCK_SCHEMA_VERSION = 1
PROTOCOL_LOCK_TYPE = 'f3_local_barlow_twins_gaussian25_protocol_v1'
SELECTION_LOCK_SCHEMA_VERSION = 3
SELECTION_LOCK_TYPE = 'f3_local_barlow_twins_gaussian25_selection_v1'
FINAL_RESULT_SCHEMA_VERSION = 2
FINAL_RESULT_TYPE = 'f3_local_barlow_twins_gaussian25_final_validation_v1'
FORCED_STD005_ID = 'local_barlow_twins_gaussian_noise_std005'
FORCED_STD010_ID = 'local_barlow_twins_gaussian_noise_std010'
IDENTITY_STD010_ID = 'local_barlow_twins_identity_gaussian_noise_std010'
EXPECTED_CANDIDATES = {
	FORCED_STD005_ID: (HORIZONTAL_VIEW_POLICY, 0.05),
	FORCED_STD010_ID: (HORIZONTAL_VIEW_POLICY, 0.10),
	IDENTITY_STD010_ID: (IDENTITY_VIEW_POLICY, 0.10),
}
CANDIDATE_TIE_PRIORITY = (
	FORCED_STD005_ID,
	FORCED_STD010_ID,
	IDENTITY_STD010_ID,
)
LEGACY_CONTROL_ID = 'local_barlow_twins_legacy_flip_25ep'
EXPECTED_CONTROLS = {LEGACY_CONTROL_ID}
EXPECTED_CANONICAL_SUMMARY = 'f3_lithology_mae_local_bt_five_way_v3'
EXPECTED_CANONICAL_CONFIG_RELATIVE_PATH = Path(
	'experiments/f3/facies_benchmark_v2/'
	'110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml'
)
EXPECTED_CANONICAL_CONFIG_SHA256 = (
	'285b0233ff82fe83808f82e929b611f570a67f01fa983ef191dda23d1858061b'
)
BASE_PRETRAINING_EPOCHS = 25
EXPECTED_BASE_STEPS = 15_625
EXPECTED_REFERENCE_BASE_RELATIVE_PATH = Path(
	'pretraining/f3/facies_benchmark_v1/'
	'local_barlow_twins_v1/full_100ep/latest.pt'
)
EXPECTED_REFERENCE_BASE_SHA256 = (
	'84550ed658166e8e6a40cd664e2e9ffbeab0c12d6917006abb417cd25e228ac0'
)
EXPECTED_REFERENCE_FINAL_RELATIVE_PATH = Path(
	'pretraining/f3/facies_benchmark_v1/'
	'mae_local_bt_five_way_v1/stage2/local_bt100/'
	'local_bt_continue/full_25ep/latest.pt'
)
EXPECTED_PRETRAINING_MANIFEST_RELATIVE_PATH = Path(
	'registry/manifests/f3/facies_benchmark_v1/f3_amplitude_manifest.json'
)
EXPECTED_PRETRAINING_PATH_LIST_RELATIVE_PATH = Path(
	'registry/splits/f3/facies_benchmark_v1/f3_npy_paths.txt'
)
EXPECTED_REFERENCE_FINAL_SHA256 = (
	'1c5312244f290dbfdcf2688ffa9fa8b5c64452ade162d5335be1bb8a0e256291'
)
EXPECTED_RANDOM_CHECKPOINT_SHA256 = (
	'6548d52446e7d6b9b57acd2bd39a8389a76bc5df55b52a9eda0472eb182a438c'
)
EXPECTED_CANONICAL_COMPARISON_SHA256 = (
	'b135122a7db2b6b359817096ac546f99d4e4fac1ee003a99ce7289c0445cf913'
)
EXPECTED_PRETRAINING_MANIFEST_SHA256 = (
	'c5dbc3a66a5c2eed0ec5df8745f8bf5a461b1e2e66156700091f1a751bdc0ef5'
)
EXPECTED_PRETRAINING_PATH_LIST_SHA256 = (
	'b52fd5e0c57edb2d2158be12b94046b554b5e6e13ba17008321bcdbe0ae2acb1'
)
CONTINUATION_EPOCHS = 25
EXPECTED_CONTINUATION_STEPS = 15_625
CONTINUATION_UNFREEZE_TOP_BLOCKS = 1
EXPECTED_COMPLETED_TRAINING_STATE = {
	'schema_version': 1,
	'stage': 'barlow_twins_training',
	'resume_boundary': 'epoch',
	'dataset_epoch': 24,
	'completed_epoch': True,
}
EXPECTED_TRAINED_PARAMETER_PREFIXES = ['patch_projection.', 'encoder.']
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class CandidateSource:
	"""One fixed base/continuation lineage admitted to validation."""

	candidate_id: str
	base_checkpoint: Path
	final_checkpoint: Path
	embeddings_dir: Path
	view_policy: str | None
	gaussian_noise_std: float | None
	base_pretraining_epochs: int
	continuation_epochs: int
	selectable: bool


@dataclass(frozen=True)
class ValidationSettings:
	"""Strict experiment-local source and output settings."""

	canonical_five_way_config: Path
	canonical_five_way_config_sha256: str
	reference_base_checkpoint: Path
	reference_base_checkpoint_sha256: str
	reference_final_checkpoint: Path
	reference_final_checkpoint_sha256: str
	random_checkpoint_sha256: str
	canonical_comparison_sha256: str
	pretraining_manifest_sha256: str
	pretraining_path_list_sha256: str
	candidates: tuple[CandidateSource, ...]
	controls: tuple[CandidateSource, ...]
	runs_root: Path
	protocol_lock: Path
	selection_lock: Path
	final_result: Path

	def candidate_by_id(self, candidate_id: str) -> CandidateSource:
		"""Return one configured candidate by its fixed ID."""
		for candidate in self.candidates:
			if candidate.candidate_id == candidate_id:
				return candidate
		raise ValueError(
			f'unknown candidate: {candidate_id!r}; '
			f'expected one of {sorted(EXPECTED_CANDIDATES)!r}'
		)

	def source_by_id(self, source_id: str) -> CandidateSource:
		"""Return one configured selectable candidate or attribution control."""
		for source in (*self.candidates, *self.controls):
			if source.candidate_id == source_id:
				return source
		raise ValueError(
			f'unknown validation source: {source_id!r}; expected one of '
			f'{sorted(set(EXPECTED_CANDIDATES) | EXPECTED_CONTROLS)!r}'
		)


def validation_settings_from_mapping(  # noqa: C901, PLR0912, PLR0915
	config: Mapping[str, object],
) -> ValidationSettings:
	"""Resolve the deliberately small candidate-validation configuration."""
	_require_exact_keys(
		config, {'benchmark', 'candidates', 'controls', 'outputs'}, 'config'
	)
	benchmark = _mapping(config, 'benchmark')
	outputs = _mapping(config, 'outputs')
	_require_exact_keys(
		benchmark,
		{
			'canonical_five_way_config',
			'canonical_five_way_config_sha256',
			'reference_base_checkpoint',
			'reference_base_checkpoint_sha256',
			'reference_final_checkpoint',
			'reference_final_checkpoint_sha256',
			'random_checkpoint_sha256',
			'canonical_comparison_sha256',
			'pretraining_manifest_sha256',
			'pretraining_path_list_sha256',
		},
		'benchmark',
	)
	_require_exact_keys(
		outputs,
		{'runs_root', 'protocol_lock', 'selection_lock', 'final_result'},
		'outputs',
	)
	candidate_values = config['candidates']
	if not isinstance(candidate_values, Sequence) or isinstance(
		candidate_values, str | bytes
	):
		raise TypeError('candidates must be a list')
	candidates = tuple(
		_candidate_from_mapping(value, index=index)
		for index, value in enumerate(candidate_values)
	)
	if len(candidates) != len(EXPECTED_CANDIDATES) or {
		candidate.candidate_id for candidate in candidates
	} != set(EXPECTED_CANDIDATES):
		raise ValueError(
			'candidates must define exactly '
			f'{sorted(EXPECTED_CANDIDATES)!r}'
		)
	control_values = config['controls']
	if not isinstance(control_values, Sequence) or isinstance(
		control_values, str | bytes
	):
		raise TypeError('controls must be a list')
	controls = tuple(
		_control_from_mapping(value, index=index)
		for index, value in enumerate(control_values)
	)
	if len(controls) != len(EXPECTED_CONTROLS) or {
		control.candidate_id for control in controls
	} != EXPECTED_CONTROLS:
		raise ValueError(
			'controls must define exactly '
			f'{sorted(EXPECTED_CONTROLS)!r}'
		)
	all_ids = [source.candidate_id for source in (*candidates, *controls)]
	if len(all_ids) != len(set(all_ids)):
		raise ValueError('candidate and control IDs must be unique')
	all_base_checkpoints = [
		source.base_checkpoint for source in (*candidates, *controls)
	]
	if len(all_base_checkpoints) != len(set(all_base_checkpoints)):
		raise ValueError('candidate and control base checkpoint paths must be unique')
	all_final_checkpoints = [
		source.final_checkpoint for source in (*candidates, *controls)
	]
	if len(all_final_checkpoints) != len(set(all_final_checkpoints)):
		raise ValueError('candidate and control final checkpoint paths must be unique')
	if set(all_base_checkpoints) & set(all_final_checkpoints):
		raise ValueError('base and final checkpoint paths must not overlap')
	all_embeddings = [source.embeddings_dir for source in (*candidates, *controls)]
	if len(all_embeddings) != len(set(all_embeddings)):
		raise ValueError('candidate and control embedding paths must be unique')
	for source in (*candidates, *controls):
		if source.embeddings_dir.parent.name != source.candidate_id:
			raise ValueError(
				f'{source.candidate_id} embeddings_dir must use its full model ID'
			)
	reference_base_sha256 = _sha256_value(
		benchmark['reference_base_checkpoint_sha256'],
		'benchmark.reference_base_checkpoint_sha256',
	)
	if reference_base_sha256 != EXPECTED_REFERENCE_BASE_SHA256:
		raise ValueError(
			'benchmark.reference_base_checkpoint_sha256 must equal the pinned '
			'canonical Local Barlow base checkpoint SHA-256'
		)
	reference_final_sha256 = _sha256_value(
		benchmark['reference_final_checkpoint_sha256'],
		'benchmark.reference_final_checkpoint_sha256',
	)
	if reference_final_sha256 != EXPECTED_REFERENCE_FINAL_SHA256:
		raise ValueError(
			'benchmark.reference_final_checkpoint_sha256 must equal the pinned '
			'canonical Local Barlow continuation checkpoint SHA-256'
		)
	random_checkpoint_sha256 = _pinned_sha256(
		benchmark['random_checkpoint_sha256'],
		label='benchmark.random_checkpoint_sha256',
		expected=EXPECTED_RANDOM_CHECKPOINT_SHA256,
	)
	canonical_comparison_sha256 = _pinned_sha256(
		benchmark['canonical_comparison_sha256'],
		label='benchmark.canonical_comparison_sha256',
		expected=EXPECTED_CANONICAL_COMPARISON_SHA256,
	)
	pretraining_manifest_sha256 = _pinned_sha256(
		benchmark['pretraining_manifest_sha256'],
		label='benchmark.pretraining_manifest_sha256',
		expected=EXPECTED_PRETRAINING_MANIFEST_SHA256,
	)
	pretraining_path_list_sha256 = _pinned_sha256(
		benchmark['pretraining_path_list_sha256'],
		label='benchmark.pretraining_path_list_sha256',
		expected=EXPECTED_PRETRAINING_PATH_LIST_SHA256,
	)
	canonical_five_way_config_sha256 = _pinned_sha256(
		benchmark['canonical_five_way_config_sha256'],
		label='benchmark.canonical_five_way_config_sha256',
		expected=EXPECTED_CANONICAL_CONFIG_SHA256,
	)
	runs_root = _absolute_path(outputs['runs_root'], 'outputs.runs_root')
	protocol_lock = _absolute_path(outputs['protocol_lock'], 'outputs.protocol_lock')
	if protocol_lock != runs_root.parent / 'gaussian25_protocol_lock.json':
		raise ValueError(
			'outputs.protocol_lock must be gaussian25_protocol_lock.json next to '
			'the validation runs directory'
		)
	selection_lock = _absolute_path(
		outputs['selection_lock'], 'outputs.selection_lock'
	)
	if selection_lock != runs_root.parent / 'gaussian25_selection_lock.json':
		raise ValueError(
			'outputs.selection_lock must be gaussian25_selection_lock.json next to '
			'the validation runs directory'
		)
	final_result = _absolute_path(outputs['final_result'], 'outputs.final_result')
	if final_result != runs_root.parent / 'gaussian25_final_result.json':
		raise ValueError(
			'outputs.final_result must be gaussian25_final_result.json next to '
			'the validation runs directory'
		)
	if len({protocol_lock, selection_lock, final_result}) != 3:
		raise ValueError('protocol, selection, and final-result paths must be distinct')
	return ValidationSettings(
		canonical_five_way_config=_absolute_path(
			benchmark['canonical_five_way_config'],
			'benchmark.canonical_five_way_config',
		),
		canonical_five_way_config_sha256=canonical_five_way_config_sha256,
		reference_base_checkpoint=_absolute_path(
			benchmark['reference_base_checkpoint'],
			'benchmark.reference_base_checkpoint',
		),
		reference_base_checkpoint_sha256=reference_base_sha256,
		reference_final_checkpoint=_absolute_path(
			benchmark['reference_final_checkpoint'],
			'benchmark.reference_final_checkpoint',
		),
		reference_final_checkpoint_sha256=reference_final_sha256,
		random_checkpoint_sha256=random_checkpoint_sha256,
		canonical_comparison_sha256=canonical_comparison_sha256,
		pretraining_manifest_sha256=pretraining_manifest_sha256,
		pretraining_path_list_sha256=pretraining_path_list_sha256,
		candidates=candidates,
		controls=controls,
		runs_root=runs_root,
		protocol_lock=protocol_lock,
		selection_lock=selection_lock,
		final_result=final_result,
	)


def audit_candidate_base_checkpoint(
	*,
	candidate: CandidateSource,
	canonical_config: F3FiveWayConfig,
	reference_base_checkpoint: Path,
) -> dict[str, object]:
	"""Audit one completed base before it initializes continuation."""
	_validate_reference_base_checkpoint(
		canonical_config=canonical_config,
		reference_base_checkpoint=reference_base_checkpoint,
		verify_file=True,
	)
	if not candidate.base_checkpoint.is_file():
		raise FileNotFoundError(
			f'missing candidate base checkpoint: {candidate.base_checkpoint}'
		)
	payload = load_checkpoint_metadata_without_weights(candidate.base_checkpoint)
	reference_payload = load_checkpoint_metadata_without_weights(
		reference_base_checkpoint
	)
	expected_augmentations = _expected_augmentations(candidate)
	_validate_candidate_base_checkpoint(
		candidate,
		payload=payload,
		reference_payload=reference_payload,
		expected_augmentations=expected_augmentations,
	)
	return {
		'audit_schema_version': CHECKPOINT_AUDIT_SCHEMA_VERSION,
		'audit_type': 'pre_continuation_base_checkpoint_only',
		'candidate_id': candidate.candidate_id,
		'base_checkpoint': str(candidate.base_checkpoint),
		'base_checkpoint_sha256': file_sha256(candidate.base_checkpoint),
		'base_pretraining_epochs': candidate.base_pretraining_epochs,
		'base_global_step': EXPECTED_BASE_STEPS,
		'augmentations': expected_augmentations,
		'reference_base_checkpoint': str(reference_base_checkpoint),
		'reference_base_checkpoint_sha256': EXPECTED_REFERENCE_BASE_SHA256,
		'base_parity_exceptions': _base_parity_exceptions(candidate),
		'final_checkpoint_required': False,
		'embeddings_required': False,
		'passed': True,
	}


def audit_candidate_checkpoints(
	*,
	candidate: CandidateSource,
	canonical_config: F3FiveWayConfig,
	reference_base_checkpoint: Path,
	reference_final_checkpoint: Path,
) -> dict[str, object]:
	"""Audit a completed base/continuation lineage before extraction."""
	base_audit = audit_candidate_base_checkpoint(
		candidate=candidate,
		canonical_config=canonical_config,
		reference_base_checkpoint=reference_base_checkpoint,
	)
	_validate_reference_final_checkpoint(
		canonical_config=canonical_config,
		reference_final_checkpoint=reference_final_checkpoint,
		verify_file=True,
	)
	if not candidate.final_checkpoint.is_file():
		raise FileNotFoundError(
			f'missing candidate final checkpoint: {candidate.final_checkpoint}'
		)
	final_payload = load_checkpoint_metadata_without_weights(candidate.final_checkpoint)
	reference_final_payload = load_checkpoint_metadata_without_weights(
		reference_final_checkpoint
	)
	expected_augmentations = _expected_augmentations(candidate)
	_validate_candidate_final_checkpoint(
		candidate,
		payload=final_payload,
		reference_payload=reference_final_payload,
		expected_augmentations=expected_augmentations,
	)
	return {
		'audit_schema_version': CHECKPOINT_AUDIT_SCHEMA_VERSION,
		'audit_type': 'pre_extraction_base_and_continuation_checkpoint_only',
		'candidate_id': candidate.candidate_id,
		'base_checkpoint': str(candidate.base_checkpoint),
		'base_checkpoint_sha256': base_audit['base_checkpoint_sha256'],
		'final_checkpoint': str(candidate.final_checkpoint),
		'final_checkpoint_sha256': file_sha256(candidate.final_checkpoint),
		'continuation_init_checkpoint_sha256': base_audit[
			'base_checkpoint_sha256'
		],
		'base_pretraining_epochs': candidate.base_pretraining_epochs,
		'base_global_step': EXPECTED_BASE_STEPS,
		'continuation_epochs': candidate.continuation_epochs,
		'continuation_global_step': EXPECTED_CONTINUATION_STEPS,
		'augmentations': expected_augmentations,
		'reference_base_checkpoint': str(reference_base_checkpoint),
		'reference_base_checkpoint_sha256': EXPECTED_REFERENCE_BASE_SHA256,
		'reference_final_checkpoint': str(reference_final_checkpoint),
		'reference_final_checkpoint_sha256': EXPECTED_REFERENCE_FINAL_SHA256,
		'base_parity_exceptions': _base_parity_exceptions(candidate),
		'final_parity_exceptions': _final_parity_exceptions(),
		'embeddings_required': False,
		'passed': True,
	}


def audit_candidate_source(  # noqa: PLR0913
	*,
	candidate: CandidateSource,
	candidate_model: F3FiveWayModelSource,
	canonical_config: F3FiveWayConfig,
	reference_base_checkpoint: Path,
	reference_final_checkpoint: Path,
	protocol_lock_identity: Mapping[str, object],
) -> dict[str, object]:
	"""Audit candidate identity without applying the canonical budget policy."""
	checkpoint_audit = audit_candidate_checkpoints(
		candidate=candidate,
		canonical_config=canonical_config,
		reference_base_checkpoint=reference_base_checkpoint,
		reference_final_checkpoint=reference_final_checkpoint,
	)
	survey_id = canonical_config.dataset['name']
	candidate_files = output_paths(candidate.embeddings_dir, survey_id)
	random_files = output_paths(
		canonical_config.model_by_id('random').embeddings_dir, survey_id
	)
	for role, path in (
		('candidate base checkpoint', candidate.base_checkpoint),
		('candidate final checkpoint', candidate.final_checkpoint),
		('reference base checkpoint', reference_base_checkpoint),
		('reference final checkpoint', reference_final_checkpoint),
		('candidate embeddings', candidate_files.embeddings),
		('candidate valid-token mask', candidate_files.valid_tokens),
		('candidate embedding metadata', candidate_files.metadata),
		('random valid-token mask', random_files.valid_tokens),
		('random embedding metadata', random_files.metadata),
	):
		if not path.is_file():
			raise FileNotFoundError(f'missing {role}: {path}')
	candidate_metadata = _read_json(candidate_files.metadata)
	random_metadata = _read_json(random_files.metadata)
	five_way_sources._validate_extraction_contract(  # noqa: SLF001
		candidate.candidate_id, candidate_metadata
	)
	five_way_sources._validate_shared_identity(  # noqa: SLF001
		candidate.candidate_id, candidate_metadata, random_metadata
	)
	token_grid_shape = five_way_sources._token_grid_shape(  # noqa: SLF001
		candidate.candidate_id, candidate_metadata
	)
	five_way_sources._validate_arrays(  # noqa: SLF001
		candidate.candidate_id, candidate_files, token_grid_shape
	)
	if not five_way_sources._masks_identical(  # noqa: SLF001
		candidate_files.valid_tokens, random_files.valid_tokens
	):
		raise ValueError(
			f'{candidate.candidate_id} valid-token mask is not byte-identical '
			'to the canonical random source'
		)
	final_checkpoint_sha256 = five_way_sources._validate_checkpoint_identity(  # noqa: SLF001
		candidate_model, candidate_metadata
	)
	five_way_sources._validate_objective_identity(  # noqa: SLF001
		candidate_model, candidate_metadata
	)
	expected_augmentations = _expected_augmentations(candidate)
	objective = _mapping(candidate_metadata, 'pretraining_objective')
	if candidate.selectable:
		if objective.get('augmentations') != expected_augmentations:
			raise ValueError(
				f'{candidate.candidate_id} embedding objective must record exact '
				f'candidate augmentations {expected_augmentations!r}'
			)
	elif 'augmentations' in objective:
		raise ValueError(
			f'{candidate.candidate_id} legacy embedding objective must not '
			'introduce a named augmentation policy'
		)
	if final_checkpoint_sha256 != checkpoint_audit['final_checkpoint_sha256']:
		raise ValueError('embedding metadata and live final checkpoint SHA disagree')
	return {
		'audit_schema_version': CHECKPOINT_AUDIT_SCHEMA_VERSION,
		'candidate_id': candidate.candidate_id,
		'validation_only': True,
		'selection_eligible': candidate.selectable,
		'protocol_lock': dict(protocol_lock_identity),
		'base_checkpoint': str(candidate.base_checkpoint),
		'base_checkpoint_sha256': checkpoint_audit['base_checkpoint_sha256'],
		'final_checkpoint': str(candidate.final_checkpoint),
		'final_checkpoint_sha256': final_checkpoint_sha256,
		'continuation_init_checkpoint_sha256': checkpoint_audit[
			'continuation_init_checkpoint_sha256'
		],
		'embeddings_dir': str(candidate.embeddings_dir),
		'embedding_metadata': str(candidate_files.metadata),
		'base_pretraining_epochs': candidate.base_pretraining_epochs,
		'continuation_epochs': candidate.continuation_epochs,
		'augmentations': expected_augmentations,
		'reference_base_checkpoint': str(reference_base_checkpoint),
		'reference_base_checkpoint_sha256': file_sha256(reference_base_checkpoint),
		'reference_final_checkpoint': str(reference_final_checkpoint),
		'reference_final_checkpoint_sha256': file_sha256(
			reference_final_checkpoint
		),
		'base_parity_exceptions': _base_parity_exceptions(candidate),
		'final_parity_exceptions': _final_parity_exceptions(),
		'fixed_downstream_summary_name': canonical_config.summary_name,
		'fixed_section_layout_dataset_root': str(
			canonical_config.section_layout_dataset_root
		),
		'token_grid_shape': list(token_grid_shape),
		'valid_token_mask': 'byte_identical_to_canonical_random',
		'evaluation_split': 'validation',
		'evaluation_aggregation_unit': VALIDATION_AGGREGATION_UNIT,
	}


def _validate_candidate_base_checkpoint(  # noqa: C901, PLR0912
	candidate: CandidateSource,
	*,
	payload: Mapping[str, object],
	reference_payload: Mapping[str, object],
	expected_augmentations: Mapping[str, object],
) -> None:
	if payload.get('checkpoint_kind') != 'barlow_twins_pretraining':
		raise ValueError('candidate base checkpoint kind must be Barlow Twins')
	if payload.get('pretraining_method') != LOCAL_BARLOW_METHOD:
		raise ValueError('candidate base checkpoint method must be local Barlow')
	if candidate.base_pretraining_epochs != BASE_PRETRAINING_EPOCHS:
		raise ValueError('screening base must declare exactly 25 epochs')
	if payload.get('epoch') != BASE_PRETRAINING_EPOCHS:
		raise ValueError('candidate base checkpoint must end at epoch 25')
	training_state = _mapping(payload, 'training_state')
	if dict(training_state) != EXPECTED_COMPLETED_TRAINING_STATE:
		raise ValueError('candidate base must record the exact completed epoch state')
	if payload.get('amp_enabled') is not False:
		raise ValueError('candidate base must record amp_enabled=false')
	if payload.get('scaler_state_dict') is not None:
		raise ValueError('candidate base scaler_state_dict must be null')
	if payload.get('trained_parameter_prefixes') != (
		EXPECTED_TRAINED_PARAMETER_PREFIXES
	):
		raise ValueError(
			'candidate base trained_parameter_prefixes differ from canonical'
		)
	candidate_config = _mapping(payload, 'config')
	reference_config = _mapping(reference_payload, 'config')
	paths = _mapping(candidate_config, 'paths')
	if paths.get('output_root') != str(candidate.base_checkpoint.parent):
		raise ValueError('candidate base config output_root must own its checkpoint')
	augmentations = _mapping(candidate_config, 'augmentations')
	if dict(augmentations) != dict(expected_augmentations):
		raise ValueError('candidate base records unexpected augmentations')
	barlow = _mapping(candidate_config, 'barlow_twins')
	if (
		barlow.get('method') != LOCAL_BARLOW_METHOD
		or barlow.get('local_pairs_per_crop') != LOCAL_PAIRS_PER_CROP
	):
		raise ValueError('candidate base changes the Local Barlow objective')
	train = _mapping(candidate_config, 'train')
	if train.get('epochs') != BASE_PRETRAINING_EPOCHS:
		raise ValueError('candidate base train.epochs must equal 25')
	samples_per_epoch = _positive_int(
		train.get('samples_per_epoch'), 'candidate base train.samples_per_epoch'
	)
	batch_size = _positive_int(
		train.get('batch_size'), 'candidate base train.batch_size'
	)
	if samples_per_epoch % batch_size:
		raise ValueError('candidate base samples_per_epoch must divide by batch_size')
	expected_steps = BASE_PRETRAINING_EPOCHS * samples_per_epoch // batch_size
	if expected_steps != EXPECTED_BASE_STEPS:
		raise ValueError('candidate base sampling no longer yields 15625 steps')
	if payload.get('global_step') != EXPECTED_BASE_STEPS:
		raise ValueError(
			f'candidate base global_step must equal {EXPECTED_BASE_STEPS}; '
			f'got {payload.get("global_step")!r}'
		)
	if _base_parity_projection(
		candidate_config,
		allow_augmentation_difference=candidate.selectable,
	) != _base_parity_projection(
		reference_config,
		allow_augmentation_difference=candidate.selectable,
	):
		allowed = (
			'augmentation, duration, or output root'
			if candidate.selectable
			else 'duration or output root'
		)
		raise ValueError(
			'candidate base config differs from the canonical Local Barlow base '
			f'outside {allowed}'
		)


def _base_parity_projection(
	config: Mapping[str, object], *, allow_augmentation_difference: bool
) -> dict[str, object]:
	projected = deepcopy(dict(config))
	if allow_augmentation_difference:
		projected.pop('augmentations', None)
	paths = cast('dict[str, object]', projected.get('paths'))
	paths.pop('output_root', None)
	train = cast('dict[str, object]', projected.get('train'))
	train.pop('epochs', None)
	return projected


def _validate_candidate_final_checkpoint(  # noqa: C901, PLR0912, PLR0915
	candidate: CandidateSource,
	*,
	payload: Mapping[str, object],
	reference_payload: Mapping[str, object],
	expected_augmentations: Mapping[str, object],
) -> None:
	if payload.get('checkpoint_kind') != 'barlow_twins_pretraining':
		raise ValueError('candidate final checkpoint kind must be Barlow Twins')
	if payload.get('pretraining_method') != LOCAL_BARLOW_METHOD:
		raise ValueError('candidate final checkpoint method must be local Barlow')
	if candidate.continuation_epochs != CONTINUATION_EPOCHS:
		raise ValueError('candidate continuation must declare exactly 25 epochs')
	if payload.get('epoch') != CONTINUATION_EPOCHS:
		raise ValueError('candidate final checkpoint must end at epoch 25')
	training_state = _mapping(payload, 'training_state')
	if dict(training_state) != EXPECTED_COMPLETED_TRAINING_STATE:
		raise ValueError(
			'candidate continuation must record the exact completed epoch state'
		)
	if payload.get('amp_enabled') is not False:
		raise ValueError('candidate continuation must record amp_enabled=false')
	if payload.get('scaler_state_dict') is not None:
		raise ValueError('candidate continuation scaler_state_dict must be null')
	if payload.get('trained_parameter_prefixes') != (
		EXPECTED_TRAINED_PARAMETER_PREFIXES
	):
		raise ValueError(
			'candidate continuation trained_parameter_prefixes differ from canonical'
		)
	candidate_config = _mapping(payload, 'config')
	reference_config = _mapping(reference_payload, 'config')
	paths = _mapping(candidate_config, 'paths')
	if paths.get('output_root') != str(candidate.final_checkpoint.parent):
		raise ValueError('candidate final config output_root must own its checkpoint')
	augmentations = _mapping(candidate_config, 'augmentations')
	if dict(augmentations) != dict(expected_augmentations):
		raise ValueError('candidate final records unexpected augmentations')
	barlow = _mapping(candidate_config, 'barlow_twins')
	if (
		barlow.get('method') != LOCAL_BARLOW_METHOD
		or barlow.get('local_pairs_per_crop') != LOCAL_PAIRS_PER_CROP
	):
		raise ValueError('candidate continuation changes the Local Barlow objective')
	continuation = _mapping(candidate_config, 'continuation')
	_require_exact_keys(
		continuation,
		{'init_checkpoint', 'unfreeze_top_blocks'},
		'candidate continuation',
	)
	if continuation.get('init_checkpoint') != str(candidate.base_checkpoint):
		raise ValueError(
			'candidate continuation.init_checkpoint must equal its exact '
			'base checkpoint'
		)
	if continuation.get('unfreeze_top_blocks') != CONTINUATION_UNFREEZE_TOP_BLOCKS:
		raise ValueError('candidate continuation must unfreeze exactly one top block')
	lineage = _mapping(payload, 'continuation_lineage')
	_require_exact_keys(
		lineage,
		{'schema_version', 'init_checkpoint', 'init_checkpoint_sha256', 'resume_count'},
		'candidate continuation lineage',
	)
	if lineage.get('schema_version') != 1:
		raise ValueError('candidate continuation lineage schema_version must equal 1')
	if lineage.get('init_checkpoint') != str(candidate.base_checkpoint):
		raise ValueError(
			'candidate continuation lineage init_checkpoint must equal its exact base'
		)
	lineage_sha256 = _sha256_value(
		lineage.get('init_checkpoint_sha256'),
		'candidate continuation lineage init_checkpoint_sha256',
	)
	if lineage_sha256 != file_sha256(candidate.base_checkpoint):
		raise ValueError(
			'candidate continuation lineage init checkpoint SHA-256 is stale'
		)
	if lineage.get('resume_count') != 0:
		raise ValueError('candidate continuation lineage resume_count must equal 0')
	train = _mapping(candidate_config, 'train')
	if train.get('epochs') != CONTINUATION_EPOCHS:
		raise ValueError('candidate continuation train.epochs must equal 25')
	samples_per_epoch = _positive_int(
		train.get('samples_per_epoch'), 'candidate continuation train.samples_per_epoch'
	)
	batch_size = _positive_int(
		train.get('batch_size'), 'candidate continuation train.batch_size'
	)
	if samples_per_epoch % batch_size:
		raise ValueError('candidate continuation sampling must divide by batch size')
	expected_steps = CONTINUATION_EPOCHS * samples_per_epoch // batch_size
	if expected_steps != EXPECTED_CONTINUATION_STEPS:
		raise ValueError('candidate continuation no longer yields 15625 steps')
	if payload.get('global_step') != EXPECTED_CONTINUATION_STEPS:
		raise ValueError(
			'candidate continuation global_step must equal '
			f'{EXPECTED_CONTINUATION_STEPS}; got {payload.get("global_step")!r}'
		)
	if _final_parity_projection(candidate_config) != _final_parity_projection(
		reference_config
	):
		raise ValueError(
			'candidate continuation config differs from canonical top-1 continuation '
			'outside augmentation, output root, or continuation.init_checkpoint'
		)


def _final_parity_projection(config: Mapping[str, object]) -> dict[str, object]:
	projected = deepcopy(dict(config))
	projected.pop('augmentations', None)
	paths = cast('dict[str, object]', projected.get('paths'))
	paths.pop('output_root', None)
	continuation = cast('dict[str, object]', projected.get('continuation'))
	continuation.pop('init_checkpoint', None)
	return projected


def _base_parity_exceptions(candidate: CandidateSource) -> list[str]:
	if candidate.selectable:
		return ['augmentations', 'train.epochs', 'paths.output_root']
	return ['train.epochs', 'paths.output_root']


def _final_parity_exceptions() -> list[str]:
	return ['augmentations', 'paths.output_root', 'continuation.init_checkpoint']


def _expected_augmentations(candidate: CandidateSource) -> dict[str, object]:
	if candidate.gaussian_noise_std is None:
		return {
			'horizontal_flip_probability': HORIZONTAL_FLIP_PROBABILITY,
		}
	if candidate.view_policy == HORIZONTAL_VIEW_POLICY:
		return {
			'policy': HORIZONTAL_VIEW_POLICY,
			'horizontal_flip_probability': HORIZONTAL_FLIP_PROBABILITY,
			'gaussian_noise_std': candidate.gaussian_noise_std,
		}
	if candidate.view_policy == IDENTITY_VIEW_POLICY:
		return {
			'policy': IDENTITY_VIEW_POLICY,
			'gaussian_noise_std': candidate.gaussian_noise_std,
		}
	raise ValueError(f'unsupported candidate view policy: {candidate.view_policy!r}')


def _candidate_config(
	canonical: F3FiveWayConfig,
	*,
	candidate: CandidateSource,
	runs_root: Path,
) -> F3FiveWayConfig:
	base = canonical.model_by_id('local_barlow_twins')
	model = replace(
		base,
		model_id=candidate.candidate_id,
		checkpoint=candidate.final_checkpoint,
		embeddings_dir=candidate.embeddings_dir,
	)
	return replace(
		canonical,
		models=tuple(
			model if source.model_id == 'local_barlow_twins' else source
			for source in canonical.models
		),
		runs_root=runs_root,
		summary_root=runs_root.parent / 'unused_summary',
		summary_name='local_barlow_twins_gaussian_view_v1_validation',
	)


def create_gaussian25_protocol_lock(
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	*,
	created_at_utc: str | None = None,
	git_head: str | None = None,
) -> dict[str, object]:
	"""Seal the completed bases and fixed protocol before stage 2 or validation."""
	if settings.protocol_lock.exists() or settings.protocol_lock.is_symlink():
		raise FileExistsError(
			'protocol lock already exists; refusing to overwrite: '
			f'{settings.protocol_lock}'
		)
	_reject_pre_protocol_evidence(settings)
	benchmark_provenance = _validate_benchmark_provenance(
		settings, canonical, verify_files=True
	)
	repository_state = _git_repository_state()
	resolved_git_head = cast('str', repository_state['git_head'])
	if git_head is not None and git_head != resolved_git_head:
		raise ValueError('protocol lock Git HEAD must equal the live repository')
	base_inputs = _collect_protocol_base_inputs(settings, canonical)
	payload = _protocol_lock_payload(
		base_inputs=base_inputs,
		created_at_utc=created_at_utc or _utc_timestamp(),
		git_head=resolved_git_head,
		repository_state=repository_state,
		benchmark_provenance=benchmark_provenance,
	)
	_write_exclusive_json(settings.protocol_lock, payload)
	return payload


def validate_gaussian25_protocol_lock(
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
) -> Mapping[str, object]:
	"""Revalidate the immutable pre-validation protocol and its four bases."""
	if settings.protocol_lock.is_symlink():
		raise ValueError(
			f'protocol lock must not be a symlink: {settings.protocol_lock}'
		)
	if not settings.protocol_lock.is_file():
		raise FileNotFoundError(
			'Gaussian25 protocol lock is required before this operation: '
			f'{settings.protocol_lock}'
		)
	stored = _read_json(settings.protocol_lock)
	_require_exact_keys(
		stored,
		{
			'base_checkpoint_inputs',
			'base_pretraining_epochs',
			'benchmark_provenance',
			'continuation_epochs',
			'created_at_utc',
			'git_head',
			'protocol_lock_type',
			'repository_state',
			'schema_version',
			'stage_boundary',
			'validation_only',
		},
		'protocol lock',
	)
	created_at_utc = stored['created_at_utc']
	if not isinstance(created_at_utc, str):
		raise TypeError('protocol lock created_at_utc must be a string')
	_validate_utc_timestamp(created_at_utc)
	git_head = _sha1_value(stored['git_head'], 'protocol lock git_head')
	repository_state = _git_repository_state()
	if repository_state.get('git_head') != git_head:
		raise ValueError('protocol lock Git HEAD differs from the live repository')
	benchmark_provenance = _validate_benchmark_provenance(
		settings, canonical, verify_files=True
	)
	base_inputs = _collect_protocol_base_inputs(settings, canonical)
	expected = _protocol_lock_payload(
		base_inputs=base_inputs,
		created_at_utc=created_at_utc,
		git_head=git_head,
		repository_state=repository_state,
		benchmark_provenance=benchmark_provenance,
	)
	if dict(stored) != expected:
		raise ValueError(
			'protocol lock does not match the frozen repository, benchmark, and bases'
		)
	return stored


def _collect_protocol_base_inputs(
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
) -> list[dict[str, object]]:
	return [
		audit_candidate_base_checkpoint(
			candidate=settings.source_by_id(source_id),
			canonical_config=canonical,
			reference_base_checkpoint=settings.reference_base_checkpoint,
		)
		for source_id in (*CANDIDATE_TIE_PRIORITY, LEGACY_CONTROL_ID)
	]


def _protocol_lock_payload(
	*,
	base_inputs: Sequence[Mapping[str, object]],
	created_at_utc: str,
	git_head: str,
	repository_state: Mapping[str, object],
	benchmark_provenance: Mapping[str, object],
) -> dict[str, object]:
	_validate_utc_timestamp(created_at_utc)
	_sha1_value(git_head, 'git_head')
	if [row.get('candidate_id') for row in base_inputs] != [
		*CANDIDATE_TIE_PRIORITY,
		LEGACY_CONTROL_ID,
	]:
		raise ValueError('protocol lock must bind the four base checkpoints in order')
	return {
		'schema_version': PROTOCOL_LOCK_SCHEMA_VERSION,
		'protocol_lock_type': PROTOCOL_LOCK_TYPE,
		'validation_only': True,
		'stage_boundary': 'completed_bases_before_continuation',
		'base_pretraining_epochs': BASE_PRETRAINING_EPOCHS,
		'continuation_epochs': CONTINUATION_EPOCHS,
		'base_checkpoint_inputs': [dict(row) for row in base_inputs],
		'created_at_utc': created_at_utc,
		'git_head': git_head,
		'repository_state': dict(repository_state),
		'benchmark_provenance': dict(benchmark_provenance),
	}


def _protocol_lock_identity(
	settings: ValidationSettings,
	protocol_lock: Mapping[str, object],
) -> dict[str, str]:
	if protocol_lock.get('protocol_lock_type') != PROTOCOL_LOCK_TYPE:
		raise ValueError('protocol lock type is invalid')
	return {
		'path': str(settings.protocol_lock),
		'sha256': file_sha256(settings.protocol_lock),
	}


def _reject_pre_protocol_evidence(settings: ValidationSettings) -> None:
	for label, path in (
		('selection lock', settings.selection_lock),
		('final result', settings.final_result),
		('validation runs', settings.runs_root),
	):
		if _path_contains_evidence(path):
			raise ValueError(
				f'cannot create protocol lock after {label} evidence exists: {path}'
			)
	for source in (*settings.candidates, *settings.controls):
		for label, path in (
			('continuation output', source.final_checkpoint.parent),
			('candidate embeddings', source.embeddings_dir),
		):
			if _path_contains_evidence(path):
				raise ValueError(
					'cannot create protocol lock after '
					f'{source.candidate_id} {label} evidence exists: {path}'
				)


def _path_contains_evidence(path: Path) -> bool:
	if path.is_symlink() or path.is_file():
		return True
	if path.is_dir():
		return next(path.iterdir(), None) is not None
	return path.exists()


def create_gaussian25_selection_lock(
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	*,
	created_at_utc: str | None = None,
	git_head: str | None = None,
) -> dict[str, object]:
	"""Exclusively lock the best of exactly three 25-epoch view candidates."""
	if settings.selection_lock.exists() or settings.selection_lock.is_symlink():
		raise FileExistsError(
			f'selection lock already exists; refusing to overwrite: '
			f'{settings.selection_lock}'
		)
	protocol_lock = validate_gaussian25_protocol_lock(settings, canonical)
	protocol_lock_identity = _protocol_lock_identity(settings, protocol_lock)
	_reject_legacy_results_before_selection_lock(settings, canonical)
	evidence = _collect_selection_evidence(
		settings,
		canonical,
		protocol_lock_identity=protocol_lock_identity,
		verify_live_checkpoints=True,
	)
	repository_state = _mapping(protocol_lock, 'repository_state')
	resolved_git_head = cast('str', protocol_lock['git_head'])
	if git_head is not None and git_head != resolved_git_head:
		raise ValueError('selection lock Git HEAD must equal the protocol lock')
	benchmark_provenance = _mapping(protocol_lock, 'benchmark_provenance')
	payload = _selection_lock_payload(
		evidence,
		created_at_utc=created_at_utc or _utc_timestamp(),
		git_head=resolved_git_head,
		repository_state=repository_state,
		benchmark_provenance=benchmark_provenance,
		protocol_lock_identity=protocol_lock_identity,
	)
	_write_exclusive_json(settings.selection_lock, payload)
	return payload


def validate_gaussian25_selection_lock(
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
) -> Mapping[str, object]:
	"""Validate the immutable lock and every medium input it names."""
	protocol_lock = validate_gaussian25_protocol_lock(settings, canonical)
	protocol_lock_identity = _protocol_lock_identity(settings, protocol_lock)
	if settings.selection_lock.is_symlink():
		raise ValueError(
			f'selection lock must not be a symlink: {settings.selection_lock}'
		)
	if not settings.selection_lock.is_file():
		raise FileNotFoundError(
			'Gaussian25 selection lock is required before this validation job: '
			f'{settings.selection_lock}'
		)
	stored = _read_json(settings.selection_lock)
	_require_exact_keys(
		stored,
		{
			'candidate_means',
			'created_at_utc',
			'data_size',
			'evaluation_aggregation_unit',
			'fixed_strength_geometry_contrast',
			'git_head',
			'repository_state',
			'benchmark_provenance',
			'protocol_lock',
			'inputs',
			'layout_ids',
			'base_pretraining_epochs',
			'continuation_epochs',
			'schema_version',
			'selected_candidate_id',
			'selected_gaussian_noise_std',
			'selected_view_policy',
			'selection_lock_type',
			'selection_metric',
			'tie_priority',
			'tie_rule',
			'validation_only',
		},
		'selection lock',
	)
	created_at_utc = stored['created_at_utc']
	if not isinstance(created_at_utc, str):
		raise TypeError('selection lock created_at_utc must be a string')
	_validate_utc_timestamp(created_at_utc)
	git_head = _sha1_value(stored['git_head'], 'selection lock git_head')
	repository_state = _mapping(protocol_lock, 'repository_state')
	if repository_state.get('git_head') != git_head:
		raise ValueError('selection lock Git HEAD differs from the protocol lock')
	benchmark_provenance = _mapping(protocol_lock, 'benchmark_provenance')
	evidence = _collect_selection_evidence(
		settings,
		canonical,
		protocol_lock_identity=protocol_lock_identity,
		verify_live_checkpoints=False,
	)
	expected = _selection_lock_payload(
		evidence,
		created_at_utc=created_at_utc,
		git_head=git_head,
		repository_state=repository_state,
		benchmark_provenance=benchmark_provenance,
		protocol_lock_identity=protocol_lock_identity,
	)
	if dict(stored) != expected:
		raise ValueError(
			'selection lock does not match the fixed Gaussian25 inputs and rule'
		)
	return stored


def enforce_validation_order(
	*,
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	candidate: CandidateSource,
	data_size: str,
) -> Mapping[str, object] | None:
	"""Enforce selection and medium-gate ordering before resolving a job."""
	validate_gaussian25_protocol_lock(settings, canonical)
	if data_size == 'medium' and candidate.selectable:
		return None
	selection_lock = validate_gaussian25_selection_lock(settings, canonical)
	if data_size == 'medium':
		if candidate.candidate_id != LEGACY_CONTROL_ID:
			raise ValueError('only the legacy control may run medium after the lock')
		return {
			'selection_lock': selection_lock,
			'medium_gate': None,
		}
	medium_gate = _validate_medium_random_gate(
		settings=settings,
		canonical=canonical,
		selection_lock=selection_lock,
	)
	selected_id = cast('str', selection_lock['selected_candidate_id'])
	allowed = _post_lock_allowed_ids(selected_id)
	if candidate.candidate_id not in allowed:
		raise ValueError(
			f'{candidate.candidate_id} is not allowed after the Gaussian25 lock; '
			f'expected one of {sorted(allowed)!r}'
		)
	return {
		'selection_lock': selection_lock,
		'medium_gate': medium_gate,
	}


def _validation_order_provenance(
	settings: ValidationSettings,
	order_evidence: Mapping[str, object],
) -> dict[str, object]:
	selection_lock = _mapping(order_evidence, 'selection_lock')
	identity: dict[str, object] = {
		'path': str(settings.selection_lock),
		'sha256': file_sha256(settings.selection_lock),
		'selected_candidate_id': selection_lock['selected_candidate_id'],
	}
	protocol_identity = {
		'path': str(settings.protocol_lock),
		'sha256': file_sha256(settings.protocol_lock),
	}
	medium_gate = order_evidence.get('medium_gate')
	if medium_gate is None:
		return {
			'protocol_lock': protocol_identity,
			'selection_lock': identity,
			'medium_gate': None,
		}
	if not isinstance(medium_gate, Mapping):
		raise TypeError('medium gate order evidence must be a mapping')
	inputs = medium_gate.get('inputs')
	if not isinstance(inputs, list):
		raise TypeError('medium gate inputs must be a list')
	random_inputs = [
		{
			'layout_id': row.get('layout_id'),
			'metrics_path': row.get('metrics_path'),
			'metrics_sha256': row.get('metrics_sha256'),
		}
		for row in inputs
		if isinstance(row, Mapping) and row.get('candidate_id') == 'random'
	]
	if len(random_inputs) != len(LAYOUT_IDS) or {
		row['layout_id'] for row in random_inputs
	} != set(LAYOUT_IDS):
		raise ValueError('medium gate must bind exactly five random metric inputs')
	return {
		'protocol_lock': protocol_identity,
		'selection_lock': identity,
		'medium_gate': {
			'locked_candidate_id': medium_gate['locked_candidate_id'],
			'locked_candidate_wins_over_random': medium_gate[
				'locked_candidate_wins_over_random'
			],
			'legacy_wins_over_random': medium_gate['legacy_wins_over_random'],
			'random_metric_inputs': random_inputs,
		},
	}


def _post_lock_allowed_ids(selected_id: str) -> set[str]:
	if selected_id not in EXPECTED_CANDIDATES:
		raise ValueError(f'unsupported locked candidate ID: {selected_id!r}')
	allowed = {selected_id, LEGACY_CONTROL_ID}
	if selected_id == IDENTITY_STD010_ID:
		allowed.add(FORCED_STD010_ID)
	return allowed


def _collect_selection_evidence(
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	*,
	protocol_lock_identity: Mapping[str, object],
	verify_live_checkpoints: bool,
) -> dict[str, object]:
	_validate_reference_checkpoints(
		canonical_config=canonical,
		reference_base_checkpoint=settings.reference_base_checkpoint,
		reference_final_checkpoint=settings.reference_final_checkpoint,
		verify_files=verify_live_checkpoints,
	)
	base_checkpoint_shas: dict[str, str | None] = {}
	final_checkpoint_shas: dict[str, str | None] = {}
	expected_source_audits: dict[str, Mapping[str, object] | None] = {}
	for candidate_id in CANDIDATE_TIE_PRIORITY:
		candidate = settings.candidate_by_id(candidate_id)
		base_checkpoint_shas[candidate_id] = (
			file_sha256(candidate.base_checkpoint)
			if verify_live_checkpoints
			else None
		)
		final_checkpoint_shas[candidate_id] = (
			file_sha256(candidate.final_checkpoint)
			if verify_live_checkpoints
			else None
		)
		if verify_live_checkpoints:
			candidate_config = _candidate_config(
				canonical, candidate=candidate, runs_root=settings.runs_root
			)
			model = candidate_config.model_by_id(candidate_id)
			expected_source_audits[candidate_id] = audit_candidate_source(
				candidate=candidate,
				candidate_model=model,
				canonical_config=canonical,
				reference_base_checkpoint=settings.reference_base_checkpoint,
				reference_final_checkpoint=settings.reference_final_checkpoint,
				protocol_lock_identity=protocol_lock_identity,
			)
		else:
			expected_source_audits[candidate_id] = None
	rows: list[dict[str, object]] = []
	scores: dict[str, dict[str, float]] = {}
	for candidate_id in CANDIDATE_TIE_PRIORITY:
		candidate = settings.candidate_by_id(candidate_id)
		candidate_config = _candidate_config(
			canonical, candidate=candidate, runs_root=settings.runs_root
		)
		by_layout: dict[str, float] = {}
		for layout_id in LAYOUT_IDS:
			job = five_way_runner.resolve_f3_lithology_five_way_job(
				candidate_config,
				model=candidate_id,
				layout=layout_id,
				size='medium',
			)
			row = _read_candidate_job_evidence(
				job=job,
				candidate=candidate,
				canonical=canonical,
				reference_base_checkpoint=settings.reference_base_checkpoint,
				reference_final_checkpoint=settings.reference_final_checkpoint,
				expected_base_checkpoint_sha256=base_checkpoint_shas[candidate_id],
				expected_final_checkpoint_sha256=final_checkpoint_shas[candidate_id],
				expected_source_audit=expected_source_audits[candidate_id],
				expected_protocol_lock=protocol_lock_identity,
				verify_evaluation_identity=verify_live_checkpoints,
			)
			rows.append(row)
			by_layout[layout_id] = cast('float', row['macro_f1'])
		scores[candidate_id] = by_layout
	selection = _rank_candidate_scores(scores)
	return {
		'inputs': rows,
		**selection,
	}


def _rank_candidate_scores(
	scores: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
	if set(scores) != set(CANDIDATE_TIE_PRIORITY):
		raise ValueError('selection scores must define exactly three candidates')
	for candidate_id in CANDIDATE_TIE_PRIORITY:
		if set(scores[candidate_id]) != set(LAYOUT_IDS):
			raise ValueError(
				f'{candidate_id} selection scores must define exactly five layouts'
			)
	means = {
		candidate_id: fmean(scores[candidate_id][layout] for layout in LAYOUT_IDS)
		for candidate_id in CANDIDATE_TIE_PRIORITY
	}
	selected_id = max(
		CANDIDATE_TIE_PRIORITY, key=lambda candidate_id: means[candidate_id]
	)
	geometry_deltas = {
		layout_id: (
			scores[IDENTITY_STD010_ID][layout_id]
			- scores[FORCED_STD010_ID][layout_id]
		)
		for layout_id in LAYOUT_IDS
	}
	return {
		'candidate_means': means,
		'selected_candidate_id': selected_id,
		'fixed_strength_geometry_contrast': {
			'contrast_id': 'identity_std010_minus_forced_flip_std010',
			'identity_candidate_id': IDENTITY_STD010_ID,
			'forced_flip_candidate_id': FORCED_STD010_ID,
			'gaussian_noise_std': 0.10,
			'per_layout_macro_f1_delta': geometry_deltas,
			'mean_macro_f1_delta': fmean(geometry_deltas.values()),
		},
	}


def _selection_lock_payload(  # noqa: PLR0913
	evidence: Mapping[str, object],
	*,
	created_at_utc: str,
	git_head: str,
	repository_state: Mapping[str, object],
	benchmark_provenance: Mapping[str, object],
	protocol_lock_identity: Mapping[str, object],
) -> dict[str, object]:
	_validate_utc_timestamp(created_at_utc)
	_sha1_value(git_head, 'git_head')
	selected_id = cast('str', evidence['selected_candidate_id'])
	view_policy, noise_std = EXPECTED_CANDIDATES[selected_id]
	return {
		'schema_version': SELECTION_LOCK_SCHEMA_VERSION,
		'selection_lock_type': SELECTION_LOCK_TYPE,
		'validation_only': True,
		'base_pretraining_epochs': BASE_PRETRAINING_EPOCHS,
		'continuation_epochs': CONTINUATION_EPOCHS,
		'data_size': 'medium',
		'layout_ids': list(LAYOUT_IDS),
		'selection_metric': 'macro_f1',
		'evaluation_aggregation_unit': VALIDATION_AGGREGATION_UNIT,
		'tie_rule': 'exact_mean_tie_uses_priority_order',
		'tie_priority': list(CANDIDATE_TIE_PRIORITY),
		'candidate_means': evidence['candidate_means'],
		'selected_candidate_id': selected_id,
		'selected_view_policy': view_policy,
		'selected_gaussian_noise_std': noise_std,
		'fixed_strength_geometry_contrast': evidence[
			'fixed_strength_geometry_contrast'
		],
		'inputs': evidence['inputs'],
		'created_at_utc': created_at_utc,
		'git_head': git_head,
		'protocol_lock': dict(protocol_lock_identity),
		'repository_state': dict(repository_state),
		'benchmark_provenance': dict(benchmark_provenance),
	}


def _read_candidate_job_evidence(  # noqa: PLR0913
	*,
	job: five_way_runner.F3FiveWayJob,
	candidate: CandidateSource,
	canonical: F3FiveWayConfig,
	reference_base_checkpoint: Path,
	reference_final_checkpoint: Path,
	expected_base_checkpoint_sha256: str | None,
	expected_final_checkpoint_sha256: str | None,
	expected_source_audit: Mapping[str, object] | None,
	expected_protocol_lock: Mapping[str, object],
	verify_evaluation_identity: bool,
	expected_validation_order: Mapping[str, object] | None = None,
) -> dict[str, object]:
	metrics, metrics_sha256 = _read_json_with_sha(job.metrics_path)
	audit_path = job.output_dir / AUDIT_NAME
	audit, audit_sha256 = _read_json_with_sha(audit_path)
	_validate_job_source_audit(
		audit=audit,
		job=job,
		candidate=candidate,
		canonical=canonical,
		reference_base_checkpoint=reference_base_checkpoint,
		reference_final_checkpoint=reference_final_checkpoint,
		expected_base_checkpoint_sha256=expected_base_checkpoint_sha256,
		expected_final_checkpoint_sha256=expected_final_checkpoint_sha256,
		expected_source_audit=expected_source_audit,
		expected_protocol_lock=expected_protocol_lock,
		expected_validation_order=expected_validation_order,
	)
	if verify_evaluation_identity:
		_validate_job_evaluation_identity(
			job=job, metrics_sha256=metrics_sha256
		)
	macro_f1 = _macro_f1(metrics, job.metrics_path)
	return {
		'candidate_id': candidate.candidate_id,
		'layout_id': job.layout_id,
		'data_size': job.data_size,
		'macro_f1': macro_f1,
		'base_checkpoint_sha256': _sha256_value(
			audit.get('base_checkpoint_sha256'),
			f'{audit_path} base_checkpoint_sha256',
		),
		'continuation_init_checkpoint_sha256': _sha256_value(
			audit.get('continuation_init_checkpoint_sha256'),
			f'{audit_path} continuation_init_checkpoint_sha256',
		),
		'final_checkpoint_sha256': _sha256_value(
			audit.get('final_checkpoint_sha256'),
			f'{audit_path} final_checkpoint_sha256',
		),
		'metrics_path': str(job.metrics_path),
		'metrics_sha256': metrics_sha256,
		'candidate_audit_path': str(audit_path),
		'candidate_audit_sha256': audit_sha256,
	}


def _validate_job_source_audit(  # noqa: C901, PLR0912, PLR0913
	*,
	audit: Mapping[str, object],
	job: five_way_runner.F3FiveWayJob,
	candidate: CandidateSource,
	canonical: F3FiveWayConfig,
	reference_base_checkpoint: Path,
	reference_final_checkpoint: Path,
	expected_base_checkpoint_sha256: str | None,
	expected_final_checkpoint_sha256: str | None,
	expected_source_audit: Mapping[str, object] | None,
	expected_protocol_lock: Mapping[str, object],
	expected_validation_order: Mapping[str, object] | None = None,
) -> None:
	if expected_source_audit is not None:
		source_payload = dict(audit)
		for key in (
			'layout_id',
			'data_size',
			'metrics_path',
			'validation_order_provenance',
		):
			source_payload.pop(key, None)
		if source_payload != dict(expected_source_audit):
			raise ValueError(
				f'{job.output_dir} candidate audit differs from the live source audit'
			)
	actual_validation_order = audit.get('validation_order_provenance')
	if expected_validation_order is None:
		if 'validation_order_provenance' in audit:
			raise ValueError(
				f'{job.output_dir} unexpectedly records post-lock order provenance'
			)
	elif actual_validation_order != expected_validation_order:
		raise ValueError(
			f'{job.output_dir} validation order provenance is stale or incomplete'
		)
	if job.model.checkpoint != candidate.final_checkpoint:
		raise ValueError(f'{job.output_dir} must evaluate the final checkpoint')
	expected_values: Mapping[str, object] = {
		'audit_schema_version': CHECKPOINT_AUDIT_SCHEMA_VERSION,
		'candidate_id': candidate.candidate_id,
		'validation_only': True,
		'selection_eligible': candidate.selectable,
		'protocol_lock': dict(expected_protocol_lock),
		'base_checkpoint': str(candidate.base_checkpoint),
		'final_checkpoint': str(candidate.final_checkpoint),
		'embeddings_dir': str(candidate.embeddings_dir),
		'base_pretraining_epochs': BASE_PRETRAINING_EPOCHS,
		'continuation_epochs': CONTINUATION_EPOCHS,
		'augmentations': _expected_augmentations(candidate),
		'reference_base_checkpoint': str(reference_base_checkpoint),
		'reference_base_checkpoint_sha256': EXPECTED_REFERENCE_BASE_SHA256,
		'reference_final_checkpoint': str(reference_final_checkpoint),
		'reference_final_checkpoint_sha256': EXPECTED_REFERENCE_FINAL_SHA256,
		'base_parity_exceptions': _base_parity_exceptions(candidate),
		'final_parity_exceptions': _final_parity_exceptions(),
		'fixed_downstream_summary_name': canonical.summary_name,
		'fixed_section_layout_dataset_root': str(
			canonical.section_layout_dataset_root
		),
		'valid_token_mask': 'byte_identical_to_canonical_random',
		'evaluation_split': 'validation',
		'evaluation_aggregation_unit': VALIDATION_AGGREGATION_UNIT,
		'layout_id': job.layout_id,
		'data_size': job.data_size,
		'metrics_path': str(job.metrics_path),
	}
	for key, expected in expected_values.items():
		if audit.get(key) != expected:
			raise ValueError(
				f'{job.output_dir} candidate audit {key} must equal {expected!r}'
			)
	base_checkpoint_sha256 = _sha256_value(
		audit.get('base_checkpoint_sha256'),
		f'{job.output_dir} candidate audit base_checkpoint_sha256',
	)
	continuation_init_checkpoint_sha256 = _sha256_value(
		audit.get('continuation_init_checkpoint_sha256'),
		f'{job.output_dir} candidate audit continuation init checkpoint SHA-256',
	)
	if continuation_init_checkpoint_sha256 != base_checkpoint_sha256:
		raise ValueError(
			f'{job.output_dir} continuation lineage SHA-256 differs from the base'
		)
	if (
		expected_base_checkpoint_sha256 is not None
		and base_checkpoint_sha256 != expected_base_checkpoint_sha256
	):
		raise ValueError(
			f'{job.output_dir} candidate audit base checkpoint SHA-256 is stale'
		)
	final_checkpoint_sha256 = _sha256_value(
		audit.get('final_checkpoint_sha256'),
		f'{job.output_dir} candidate audit final_checkpoint_sha256',
	)
	if (
		expected_final_checkpoint_sha256 is not None
		and final_checkpoint_sha256 != expected_final_checkpoint_sha256
	):
		raise ValueError(
			f'{job.output_dir} candidate audit final checkpoint SHA-256 is stale'
		)
	embedding_metadata = output_paths(
		candidate.embeddings_dir, canonical.dataset['name']
	).metadata
	if audit.get('embedding_metadata') != str(embedding_metadata):
		raise ValueError(
			f'{job.output_dir} candidate audit embedding metadata path changed'
		)
	token_grid_shape = audit.get('token_grid_shape')
	if (
		not isinstance(token_grid_shape, list)
		or len(token_grid_shape) != 3
		or any(
			not isinstance(item, int) or isinstance(item, bool) or item <= 0
			for item in token_grid_shape
		)
	):
		raise ValueError(
			f'{job.output_dir} candidate audit token_grid_shape is invalid'
		)


def _validate_job_evaluation_identity(  # noqa: C901
	*,
	job: five_way_runner.F3FiveWayJob,
	metrics_sha256: str,
) -> None:
	decoder_config = f3_lithology_voxel_decoder_config_from_mapping(
		five_way_runner._decoder_config_mapping(job)  # noqa: SLF001
	)
	if not five_way_runner._decoder_is_completed(  # noqa: SLF001
		job, decoder_config
	):
		raise ValueError(f'{job.output_dir} decoder is not a completed fixed job')
	metadata_path = job.evaluation_dir / 'evaluation_metadata.json'
	metadata = _read_json(metadata_path)
	if metadata.get('dataset') != dict(job.config.dataset):
		raise ValueError(f'{metadata_path} dataset identity changed')
	if metadata.get('model_tag') != job.model.model_id:
		raise ValueError(f'{metadata_path} model identity changed')
	aggregation = _mapping(metadata, 'aggregation')
	if aggregation.get('primary_unit') != VALIDATION_AGGREGATION_UNIT:
		raise ValueError(f'{metadata_path} is not validation unique-voxel output')
	policy = _mapping(metadata, 'policy')
	for key, expected in five_way_runner.FIVE_WAY_EVALUATION_POLICY.items():
		normalized = list(expected) if isinstance(expected, tuple) else expected
		if policy.get(key) != normalized:
			raise ValueError(f'{metadata_path} evaluation policy {key} changed')
	outputs = _mapping(metadata, 'outputs')
	metrics_identity = _mapping(outputs, 'metrics.json')
	if metrics_identity.get('path') != str(job.metrics_path):
		raise ValueError(f'{metadata_path} metrics path identity changed')
	if metrics_identity.get('sha256') != metrics_sha256:
		raise ValueError(f'{job.metrics_path} changed after evaluation')
	source_identity = five_way_results._job_source_identity(  # noqa: SLF001
		label=f'{job.model.model_id}/{job.layout_id}/{job.data_size}',
		model=job.model,
		survey_id=job.config.dataset['name'],
		job_dir=job.output_dir,
		evaluation_metadata=metadata,
	)
	best_checkpoint = job.decoder_dir / five_way_runner.BEST_CHECKPOINT_NAME
	if file_sha256(best_checkpoint) != source_identity['decoder_checkpoint_sha256']:
		raise ValueError(
			f'{best_checkpoint} changed after full-volume prediction'
		)
	inspection = five_way_runner.inspect_f3_lithology_five_way_job(job)
	run_metadata = _read_json(job.decoder_dir / 'run_metadata.json')
	if run_metadata.get('initial_model_state_sha256') != inspection.get(
		'decoder_initial_state_sha256'
	):
		raise ValueError(
			f'{job.output_dir} decoder initial-state identity changed'
		)


def _reject_legacy_results_before_selection_lock(
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
) -> None:
	for source in (*settings.candidates, *settings.controls):
		candidate_config = _candidate_config(
			canonical, candidate=source, runs_root=settings.runs_root
		)
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZES:
				if source.selectable and data_size == 'medium':
					continue
				job = five_way_runner.resolve_f3_lithology_five_way_job(
					candidate_config,
					model=source.candidate_id,
					layout=layout_id,
					size=data_size,
				)
				for path in (job.metrics_path, job.output_dir / AUDIT_NAME):
					if path.exists() or path.is_symlink():
						raise ValueError(
							'validation was materialized out of preregistered order '
							f'before Gaussian25 selection was locked: {path}'
						)


def _validate_medium_random_gate(
	*,
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	selection_lock: Mapping[str, object],
	require_open: bool = True,
) -> dict[str, object]:
	protocol_lock = validate_gaussian25_protocol_lock(settings, canonical)
	protocol_lock_identity = _protocol_lock_identity(settings, protocol_lock)
	selected_id = cast('str', selection_lock['selected_candidate_id'])
	selected = settings.candidate_by_id(selected_id)
	legacy = settings.source_by_id(LEGACY_CONTROL_ID)
	scores: dict[str, dict[str, float]] = {
		selected_id: {},
		LEGACY_CONTROL_ID: {},
		'random': {},
	}
	inputs: list[dict[str, object]] = []
	live_source_audits: dict[str, Mapping[str, object]] = {}
	for source in (selected, legacy):
		candidate_config = _candidate_config(
			canonical, candidate=source, runs_root=settings.runs_root
		)
		live_source_audits[source.candidate_id] = audit_candidate_source(
			candidate=source,
			candidate_model=candidate_config.model_by_id(source.candidate_id),
			canonical_config=canonical,
			reference_base_checkpoint=settings.reference_base_checkpoint,
			reference_final_checkpoint=settings.reference_final_checkpoint,
			protocol_lock_identity=protocol_lock_identity,
		)
	for source in (selected, legacy):
		candidate_config = _candidate_config(
			canonical, candidate=source, runs_root=settings.runs_root
		)
		for layout_id in LAYOUT_IDS:
			job = five_way_runner.resolve_f3_lithology_five_way_job(
				candidate_config,
				model=source.candidate_id,
				layout=layout_id,
				size='medium',
			)
			row = _read_candidate_job_evidence(
				job=job,
				candidate=source,
				canonical=canonical,
				reference_base_checkpoint=settings.reference_base_checkpoint,
				reference_final_checkpoint=settings.reference_final_checkpoint,
				expected_base_checkpoint_sha256=cast(
					'str',
					live_source_audits[source.candidate_id][
						'base_checkpoint_sha256'
					],
				),
				expected_final_checkpoint_sha256=cast(
					'str',
					live_source_audits[source.candidate_id][
						'final_checkpoint_sha256'
					],
				),
				expected_source_audit=live_source_audits[source.candidate_id],
				expected_protocol_lock=protocol_lock_identity,
				verify_evaluation_identity=True,
				expected_validation_order=(
					None
					if source.selectable
					else _validation_order_provenance(
						settings,
						{
							'selection_lock': selection_lock,
							'medium_gate': None,
						},
					)
				),
			)
			inputs.append(row)
			scores[source.candidate_id][layout_id] = cast(
				'float', row['macro_f1']
			)
	for layout_id in LAYOUT_IDS:
		random_row = _read_random_job_evidence(
			canonical, layout_id=layout_id, data_size='medium'
		)
		inputs.append(random_row)
		scores['random'][layout_id] = cast('float', random_row['macro_f1'])
	wins = _medium_5of5_wins(scores, selected_id=selected_id)
	gate_open = any(wins.values())
	if require_open and not gate_open:
		raise ValueError(
			'small/large validation is forbidden because neither the locked '
			'candidate nor legacy has 5/5 positive medium deltas over random'
		)
	return {
		'gate_open': gate_open,
		'layout_ids': list(LAYOUT_IDS),
		'locked_candidate_id': selected_id,
		'locked_candidate_wins_over_random': wins[selected_id],
		'legacy_wins_over_random': wins[LEGACY_CONTROL_ID],
		'locked_candidate_deltas': {
			layout_id: scores[selected_id][layout_id] - scores['random'][layout_id]
			for layout_id in LAYOUT_IDS
		},
		'legacy_deltas': {
			layout_id: (
				scores[LEGACY_CONTROL_ID][layout_id]
				- scores['random'][layout_id]
			)
			for layout_id in LAYOUT_IDS
		},
		'inputs': inputs,
	}


def _medium_5of5_wins(
	scores: Mapping[str, Mapping[str, float]], *, selected_id: str
) -> dict[str, bool]:
	expected_ids = {selected_id, LEGACY_CONTROL_ID, 'random'}
	if set(scores) != expected_ids:
		raise ValueError('medium gate scores have unexpected source IDs')
	for source_id in expected_ids:
		if set(scores[source_id]) != set(LAYOUT_IDS):
			raise ValueError(
				f'{source_id} medium gate must define all five paired layouts'
			)
	return {
		source_id: all(
			scores[source_id][layout_id] > scores['random'][layout_id]
			for layout_id in LAYOUT_IDS
		)
		for source_id in (selected_id, LEGACY_CONTROL_ID)
	}


def _read_random_job_evidence(
	canonical: F3FiveWayConfig, *, layout_id: str, data_size: str
) -> dict[str, object]:
	job = five_way_runner.resolve_f3_lithology_five_way_job(
		canonical, model='random', layout=layout_id, size=data_size
	)
	metrics, metrics_sha256 = _read_json_with_sha(job.metrics_path)
	_validate_job_evaluation_identity(job=job, metrics_sha256=metrics_sha256)
	macro_f1 = _macro_f1(metrics, job.metrics_path)
	_validate_random_comparison_row(canonical, job=job, macro_f1=macro_f1)
	return {
		'candidate_id': 'random',
		'layout_id': layout_id,
		'data_size': data_size,
		'macro_f1': macro_f1,
		'checkpoint_path': str(job.model.checkpoint),
		'checkpoint_sha256': EXPECTED_RANDOM_CHECKPOINT_SHA256,
		'metrics_path': str(job.metrics_path),
		'metrics_sha256': metrics_sha256,
		'canonical_comparison': str(canonical.summary_root / 'comparison.csv'),
	}


def _validate_random_comparison_row(
	canonical: F3FiveWayConfig,
	*,
	job: five_way_runner.F3FiveWayJob,
	macro_f1: float,
) -> None:
	"""Require the live random cell to equal its pinned comparison-ledger row."""
	comparison = canonical.summary_root / 'comparison.csv'
	if not comparison.is_file():
		raise FileNotFoundError(f'missing canonical comparison.csv: {comparison}')
	with comparison.open(encoding='utf-8', newline='') as stream:
		reader = csv.DictReader(stream)
		if reader.fieldnames != list(five_way_results.COMPARISON_FIELDNAMES):
			raise ValueError('canonical comparison.csv has unexpected columns')
		matches = [
			row
			for row in reader
			if row['model_id'] == 'random'
			and row['layout_id'] == job.layout_id
			and row['data_size'] == job.data_size
		]
	if len(matches) != 1:
		raise ValueError(
			'canonical comparison.csv must contain exactly one matching random row '
			f'for {job.layout_id}/{job.data_size}; found {len(matches)}'
		)
	row = matches[0]
	if row['checkpoint_path'] != str(job.model.checkpoint):
		raise ValueError('canonical random comparison checkpoint path drifted')
	if row['encoder_checkpoint_sha256'] != EXPECTED_RANDOM_CHECKPOINT_SHA256:
		raise ValueError('canonical random comparison checkpoint SHA-256 drifted')
	if row['metrics_path'] != str(job.metrics_path):
		raise ValueError('canonical random comparison metrics path drifted')
	try:
		comparison_macro_f1 = float(row['macro_f1'])
	except ValueError as error:
		raise ValueError(
			'canonical random comparison macro_f1 must be numeric'
		) from error
	if not math.isfinite(comparison_macro_f1) or comparison_macro_f1 != macro_f1:
		raise ValueError(
			'live random macro_f1 does not equal the pinned canonical comparison row'
		)


def create_gaussian25_final_result(
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	*,
	created_at_utc: str | None = None,
) -> dict[str, object]:
	"""Audit the exact reached 25-epoch branch and publish its immutable result."""
	if settings.final_result.exists() or settings.final_result.is_symlink():
		raise FileExistsError(
			'final result already exists; refusing to overwrite: '
			f'{settings.final_result}'
		)
	protocol_lock = validate_gaussian25_protocol_lock(settings, canonical)
	protocol_lock_identity = _protocol_lock_identity(settings, protocol_lock)
	selection_lock = validate_gaussian25_selection_lock(settings, canonical)
	selection_lock_sha256 = file_sha256(settings.selection_lock)
	benchmark_provenance = _validate_benchmark_provenance(
		settings, canonical, verify_files=True
	)
	five_way_sources.audit_f3_lithology_five_way_sources(canonical)
	medium_gate = _validate_medium_random_gate(
		settings=settings,
		canonical=canonical,
		selection_lock=selection_lock,
		require_open=False,
	)
	selected_id = cast('str', selection_lock['selected_candidate_id'])
	gate_open = cast('bool', medium_gate['gate_open'])
	expected_cells = _expected_gaussian25_cells(
		selected_id=selected_id, medium_gate_open=gate_open
	)
	_validate_exact_candidate_cell_set(
		settings=settings,
		canonical=canonical,
		expected_cells=expected_cells,
	)
	post_lock_order = _validation_order_provenance(
		settings,
		{
			'selection_lock': selection_lock,
			'medium_gate': medium_gate,
		},
	)
	lock_only_order = _validation_order_provenance(
		settings,
		{
			'selection_lock': selection_lock,
			'medium_gate': None,
		},
	)
	source_ids = sorted({source_id for source_id, _, _ in expected_cells})
	live_source_audits: dict[str, Mapping[str, object]] = {}
	for source_id in source_ids:
		source = settings.source_by_id(source_id)
		candidate_config = _candidate_config(
			canonical, candidate=source, runs_root=settings.runs_root
		)
		live_source_audits[source_id] = audit_candidate_source(
			candidate=source,
			candidate_model=candidate_config.model_by_id(source_id),
			canonical_config=canonical,
			reference_base_checkpoint=settings.reference_base_checkpoint,
			reference_final_checkpoint=settings.reference_final_checkpoint,
			protocol_lock_identity=protocol_lock_identity,
		)
	candidate_inputs: list[dict[str, object]] = []
	for source_id, layout_id, data_size in sorted(expected_cells):
		source = settings.source_by_id(source_id)
		candidate_config = _candidate_config(
			canonical, candidate=source, runs_root=settings.runs_root
		)
		job = five_way_runner.resolve_f3_lithology_five_way_job(
			candidate_config,
			model=source_id,
			layout=layout_id,
			size=data_size,
		)
		expected_order: Mapping[str, object] | None
		if source.selectable and data_size == 'medium':
			expected_order = None
		elif data_size == 'medium':
			expected_order = lock_only_order
		else:
			expected_order = post_lock_order
		candidate_inputs.append(
			_read_candidate_job_evidence(
				job=job,
				candidate=source,
				canonical=canonical,
				reference_base_checkpoint=settings.reference_base_checkpoint,
				reference_final_checkpoint=settings.reference_final_checkpoint,
				expected_base_checkpoint_sha256=cast(
					'str', live_source_audits[source_id]['base_checkpoint_sha256']
				),
				expected_final_checkpoint_sha256=cast(
					'str', live_source_audits[source_id]['final_checkpoint_sha256']
				),
				expected_source_audit=live_source_audits[source_id],
				expected_protocol_lock=protocol_lock_identity,
				verify_evaluation_identity=True,
				expected_validation_order=expected_order,
			)
		)
	reached_sizes = DATA_SIZES if gate_open else ('medium',)
	random_inputs = [
		_read_random_job_evidence(
			canonical, layout_id=layout_id, data_size=data_size
		)
		for data_size in reached_sizes
		for layout_id in LAYOUT_IDS
	]
	_assert_unique_evidence_rows(candidate_inputs, include_candidate=True)
	_assert_unique_evidence_rows(random_inputs, include_candidate=False)
	scores = _scores_from_evidence((*candidate_inputs, *random_inputs))
	selected_result = _arm_random_result(
		scores, arm_id=selected_id, medium_gate_open=gate_open
	)
	legacy_result = _arm_random_result(
		scores, arm_id=LEGACY_CONTROL_ID, medium_gate_open=gate_open
	)
	attribution = _paired_arm_contrast(
		scores,
		left_id=selected_id,
		right_id=LEGACY_CONTROL_ID,
		contrast_id='selected_gaussian_minus_matched_legacy',
		full_branch_reached=gate_open,
	)
	geometry = _geometry_result(
		scores,
		selected_id=selected_id,
		medium_gate_open=gate_open,
	)
	selected_passed = cast('bool', selected_result['wins_all_15_over_random'])
	legacy_passed = cast('bool', legacy_result['wins_all_15_over_random'])
	attribution_passed = cast('bool', attribution['wins_all_15'])
	winner_id = _choose_gaussian25_winner(
		selected_id=selected_id,
		selected_passed=selected_passed,
		legacy_passed=legacy_passed,
		attribution_passed=attribution_passed,
	)
	passed = winner_id is not None
	payload = {
		'schema_version': FINAL_RESULT_SCHEMA_VERSION,
		'final_result_type': FINAL_RESULT_TYPE,
		'validation_only': True,
		'base_pretraining_epochs': BASE_PRETRAINING_EPOCHS,
		'continuation_epochs': CONTINUATION_EPOCHS,
		'protocol_lock': protocol_lock_identity,
		'selection_lock': {
			'path': str(settings.selection_lock),
			'sha256': selection_lock_sha256,
			'selected_candidate_id': selected_id,
		},
		'benchmark_provenance': benchmark_provenance,
		'repository_state': selection_lock['repository_state'],
		'medium_gate': {
			'gate_open': gate_open,
			'locked_candidate_wins_over_random': medium_gate[
				'locked_candidate_wins_over_random'
			],
			'legacy_wins_over_random': medium_gate['legacy_wins_over_random'],
		},
		'exact_expected_candidate_cells': [
			{
				'candidate_id': source_id,
				'layout_id': layout_id,
				'data_size': data_size,
			}
			for source_id, layout_id, data_size in sorted(expected_cells)
		],
		'candidate_inputs': candidate_inputs,
		'random_inputs': random_inputs,
		'arm_results': {
			selected_id: selected_result,
			LEGACY_CONTROL_ID: legacy_result,
		},
		'gaussian_attribution': attribution,
		'identity_vs_forced_geometry': geometry,
		'passed': passed,
		'winner_candidate_id': winner_id,
		'authorizes_next_base_duration': not passed,
		'failure_stage': (
			None if passed else ('final_15of15' if gate_open else 'medium_5of5')
		),
		'created_at_utc': created_at_utc or _utc_timestamp(),
	}
	_validate_utc_timestamp(cast('str', payload['created_at_utc']))
	_write_exclusive_json(settings.final_result, payload)
	return payload


def _choose_gaussian25_winner(
	*,
	selected_id: str,
	selected_passed: bool,
	legacy_passed: bool,
	attribution_passed: bool,
) -> str | None:
	if selected_id not in EXPECTED_CANDIDATES:
		raise ValueError(f'unsupported selected candidate: {selected_id!r}')
	if selected_passed and legacy_passed:
		return selected_id if attribution_passed else LEGACY_CONTROL_ID
	if selected_passed:
		return selected_id
	if legacy_passed:
		return LEGACY_CONTROL_ID
	return None


def _expected_gaussian25_cells(
	*, selected_id: str, medium_gate_open: bool
) -> set[tuple[str, str, str]]:
	if selected_id not in EXPECTED_CANDIDATES:
		raise ValueError(f'unsupported selected candidate: {selected_id!r}')
	expected = {
		(candidate_id, layout_id, 'medium')
		for candidate_id in CANDIDATE_TIE_PRIORITY
		for layout_id in LAYOUT_IDS
	}
	expected.update(
		(LEGACY_CONTROL_ID, layout_id, 'medium') for layout_id in LAYOUT_IDS
	)
	if not medium_gate_open:
		return expected
	post_lock_ids = {selected_id, LEGACY_CONTROL_ID}
	if selected_id == IDENTITY_STD010_ID:
		post_lock_ids.add(FORCED_STD010_ID)
	expected.update(
		(source_id, layout_id, data_size)
		for source_id in post_lock_ids
		for layout_id in LAYOUT_IDS
		for data_size in ('small', 'large')
	)
	return expected


def _validate_exact_candidate_cell_set(
	*,
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	expected_cells: set[tuple[str, str, str]],
) -> None:
	all_source_ids = set(EXPECTED_CANDIDATES) | EXPECTED_CONTROLS
	for source_id in sorted(all_source_ids):
		source = settings.source_by_id(source_id)
		candidate_config = _candidate_config(
			canonical, candidate=source, runs_root=settings.runs_root
		)
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZES:
				job = five_way_runner.resolve_f3_lithology_five_way_job(
					candidate_config,
					model=source_id,
					layout=layout_id,
					size=data_size,
				)
				paths = (job.metrics_path, job.output_dir / AUDIT_NAME)
				expected = (source_id, layout_id, data_size) in expected_cells
				for path in paths:
					if path.is_symlink():
						raise ValueError(
							f'validation evidence must not be a symlink: {path}'
						)
					if expected and not path.is_file():
						raise FileNotFoundError(
							f'missing expected validation cell evidence: {path}'
						)
					if not expected and path.exists():
						raise ValueError(
							'validation cell exists outside the exact reached set: '
							f'{path}'
						)


def _assert_unique_evidence_rows(
	rows: Sequence[Mapping[str, object]], *, include_candidate: bool
) -> None:
	keys: list[tuple[object, ...]] = []
	for row in rows:
		key = (row.get('layout_id'), row.get('data_size'))
		if include_candidate:
			key = (row.get('candidate_id'), *key)
		keys.append(key)
	if len(keys) != len(set(keys)):
		raise ValueError('validation evidence contains duplicated cell identities')


def _scores_from_evidence(
	rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[tuple[str, str], float]]:
	result: dict[str, dict[tuple[str, str], float]] = {}
	for row in rows:
		source_id = row.get('candidate_id')
		layout_id = row.get('layout_id')
		data_size = row.get('data_size')
		if not all(
			isinstance(value, str) for value in (source_id, layout_id, data_size)
		):
			raise TypeError('validation evidence cell identity must contain strings')
		value = row.get('macro_f1')
		if not isinstance(value, int | float) or isinstance(value, bool):
			raise TypeError('validation evidence macro_f1 must be numeric')
		key = (cast('str', layout_id), cast('str', data_size))
		by_cell = result.setdefault(cast('str', source_id), {})
		if key in by_cell:
			raise ValueError('validation evidence duplicates a source cell')
		by_cell[key] = float(value)
	return result


def _arm_random_result(
	scores: Mapping[str, Mapping[tuple[str, str], float]],
	*,
	arm_id: str,
	medium_gate_open: bool,
) -> dict[str, object]:
	expected_cells = {
		(layout_id, data_size)
		for layout_id in LAYOUT_IDS
		for data_size in (DATA_SIZES if medium_gate_open else ('medium',))
	}
	if set(scores.get(arm_id, {})) != expected_cells:
		raise ValueError(f'{arm_id} does not define the exact reached cell set')
	if set(scores.get('random', {})) != expected_cells:
		raise ValueError('random baseline does not define the exact reached cell set')
	deltas = {
		f'{layout_id}/{data_size}': (
			scores[arm_id][(layout_id, data_size)]
			- scores['random'][(layout_id, data_size)]
		)
		for layout_id, data_size in sorted(expected_cells)
	}
	return {
		'evaluated_cell_count': len(expected_cells),
		'paired_macro_f1_deltas_over_random': deltas,
		'positive_delta_count': sum(delta > 0.0 for delta in deltas.values()),
		'wins_all_15_over_random': (
			medium_gate_open
			and len(deltas) == 15
			and all(delta > 0.0 for delta in deltas.values())
		),
	}


def _paired_arm_contrast(
	scores: Mapping[str, Mapping[tuple[str, str], float]],
	*,
	left_id: str,
	right_id: str,
	contrast_id: str,
	full_branch_reached: bool,
) -> dict[str, object]:
	left_cells = set(scores.get(left_id, {}))
	right_cells = set(scores.get(right_id, {}))
	if left_cells != right_cells:
		raise ValueError(f'{contrast_id} arms do not have identical reached cells')
	deltas = {
		f'{layout_id}/{data_size}': (
			scores[left_id][(layout_id, data_size)]
			- scores[right_id][(layout_id, data_size)]
		)
		for layout_id, data_size in sorted(left_cells)
	}
	return {
		'contrast_id': contrast_id,
		'left_candidate_id': left_id,
		'right_candidate_id': right_id,
		'evaluated_cell_count': len(deltas),
		'paired_macro_f1_deltas': deltas,
		'positive_delta_count': sum(delta > 0.0 for delta in deltas.values()),
		'wins_all_15': (
			full_branch_reached
			and len(deltas) == 15
			and all(delta > 0.0 for delta in deltas.values())
		),
	}


def _geometry_result(
	scores: Mapping[str, Mapping[tuple[str, str], float]],
	*,
	selected_id: str,
	medium_gate_open: bool,
) -> dict[str, object]:
	full_geometry_required = selected_id == IDENTITY_STD010_ID and medium_gate_open
	required_cells = {
		(layout_id, data_size)
		for layout_id in LAYOUT_IDS
		for data_size in (DATA_SIZES if full_geometry_required else ('medium',))
	}
	for source_id in (IDENTITY_STD010_ID, FORCED_STD010_ID):
		if not required_cells.issubset(set(scores.get(source_id, {}))):
			raise ValueError(
				'identity/forced geometry evidence is incomplete for the reached branch'
			)
	deltas = {
		f'{layout_id}/{data_size}': (
			scores[IDENTITY_STD010_ID][(layout_id, data_size)]
			- scores[FORCED_STD010_ID][(layout_id, data_size)]
		)
		for layout_id, data_size in sorted(required_cells)
	}
	return {
		'contrast_id': 'identity_std010_minus_forced_flip_std010',
		'left_candidate_id': IDENTITY_STD010_ID,
		'right_candidate_id': FORCED_STD010_ID,
		'evaluated_cell_count': len(deltas),
		'paired_macro_f1_deltas': deltas,
		'positive_delta_count': sum(delta > 0.0 for delta in deltas.values()),
		'wins_all_15': (
			full_geometry_required and all(delta > 0.0 for delta in deltas.values())
		),
		'full_15_cell_geometry_required': full_geometry_required,
		'required_cell_count': len(required_cells),
		'complete': True,
	}


def _run_job(
	job: five_way_runner.F3FiveWayJob,
	*,
	audit: Mapping[str, object],
	device: str,
	resume: Path | None,
) -> dict[str, object]:
	if job.metrics_path.is_file():
		raise FileExistsError(
			f'validation job already completed: {job.metrics_path}'
		)
	validate_f3_lithology_voxel_section_layout_condition(job.condition_dir)
	job_audit = dict(audit)
	job_audit.update(
		{
			'layout_id': job.layout_id,
			'data_size': job.data_size,
			'metrics_path': str(job.metrics_path),
		}
	)
	_write_stable_json(job.output_dir / AUDIT_NAME, job_audit)
	decoder_config = f3_lithology_voxel_decoder_config_from_mapping(
		five_way_runner._decoder_config_mapping(job)  # noqa: SLF001
	)
	decoder_completed = five_way_runner._decoder_is_completed(  # noqa: SLF001
		job, decoder_config
	)
	if decoder_completed:
		if resume is not None:
			raise FileExistsError('decoder is already complete; omit --resume')
	else:
		if resume is None and five_way_runner._decoder_dir_is_occupied(  # noqa: SLF001
			job
		):
			latest = job.decoder_dir / five_way_runner.LATEST_CHECKPOINT_NAME
			raise FileExistsError(
				f'interrupted decoder run; resume with --resume {latest}'
			)
		training = run_f3_lithology_voxel_decoder(
			decoder_config, device=device, resume=resume
		)
		if not training.completed:
			return {
				'completed': False,
				'latest_checkpoint': str(training.latest_checkpoint),
				'protocol_lock': audit['protocol_lock'],
			}
	if not (
		job.prediction_dir / five_way_runner.PREDICTION_METADATA_NAME
	).is_file():
		inference_config = f3_lithology_voxel_inference_config_from_mapping(
			five_way_runner._inference_config_mapping(job)  # noqa: SLF001
		)
		predict_f3_lithology_voxels(inference_config, device=device)
	evaluation_config = f3_lithology_voxel_evaluation_config_from_mapping(
		five_way_runner._evaluation_config_mapping(job)  # noqa: SLF001
	)
	evaluate_f3_lithology_voxels(evaluation_config)
	metrics = _read_json(job.metrics_path)
	if metrics.get('aggregation_unit') != VALIDATION_AGGREGATION_UNIT:
		raise ValueError('candidate evaluation did not use unique validation voxels')
	return {
		'completed': True,
		'validation_only': True,
		'candidate_id': job.model.model_id,
		'selection_eligible': bool(audit['selection_eligible']),
		'protocol_lock': audit['protocol_lock'],
		'layout_id': job.layout_id,
		'data_size': job.data_size,
		'macro_f1': metrics.get('macro_f1'),
		'metrics_path': str(job.metrics_path),
	}


def _write_stable_json(path: Path, payload: Mapping[str, object]) -> None:
	encoded = json.dumps(payload, indent=2, sort_keys=True) + '\n'
	if path.is_file():
		if path.read_text(encoding='utf-8') != encoded:
			raise ValueError(f'existing source audit differs: {path}')
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(encoded, encoding='utf-8')


def _write_exclusive_json(path: Path, payload: Mapping[str, object]) -> None:
	"""Atomically publish one JSON artifact without replacing any path."""
	encoded = (json.dumps(payload, indent=2, sort_keys=True) + '\n').encode()
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary: Path | None = None
	try:
		with tempfile.NamedTemporaryFile(
			mode='wb',
			dir=path.parent,
			prefix=f'.{path.name}.',
			suffix='.tmp',
			delete=False,
		) as stream:
			temporary = Path(stream.name)
			stream.write(encoded)
			stream.flush()
			os.fsync(stream.fileno())
		if temporary is None:
			raise RuntimeError('failed to create selection-lock temporary file')
		try:
			os.link(temporary, path)
		except FileExistsError as error:
			raise FileExistsError(
				f'refusing to overwrite immutable selection lock: {path}'
			) from error
	finally:
		if temporary is not None:
			temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Mapping[str, object]:
	value = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(value, Mapping):
		raise TypeError(f'{path} must contain a JSON object')
	return value


def _read_json_with_sha(path: Path) -> tuple[Mapping[str, object], str]:
	try:
		content = path.read_bytes()
	except FileNotFoundError as error:
		raise FileNotFoundError(
			f'missing required validation artifact: {path}'
		) from error
	try:
		value = json.loads(content)
	except json.JSONDecodeError as error:
		raise ValueError(f'{path} must contain valid JSON') from error
	if not isinstance(value, Mapping):
		raise TypeError(f'{path} must contain a JSON object')
	return value, hashlib.sha256(content).hexdigest()


def _macro_f1(metrics: Mapping[str, object], path: Path) -> float:
	if metrics.get('aggregation_unit') != VALIDATION_AGGREGATION_UNIT:
		raise ValueError(
			f'{path} aggregation_unit must equal {VALIDATION_AGGREGATION_UNIT!r}'
		)
	value = metrics.get('macro_f1')
	if not isinstance(value, int | float) or isinstance(value, bool):
		raise ValueError(f'{path} macro_f1 must be numeric')  # noqa: TRY004
	result = float(value)
	if not math.isfinite(result) or not 0.0 <= result <= 1.0:
		raise ValueError(f'{path} macro_f1 must be finite and within [0, 1]')
	return result


def _sha256_value(value: object, label: str) -> str:
	if not isinstance(value, str) or len(value) != 64:
		raise ValueError(f'{label} must be a SHA-256 digest')
	try:
		bytes.fromhex(value)
	except ValueError as error:
		raise ValueError(f'{label} must be a SHA-256 digest') from error
	return value


def _pinned_sha256(value: object, *, label: str, expected: str) -> str:
	resolved = _sha256_value(value, label)
	if resolved != expected:
		raise ValueError(f'{label} must equal the pinned canonical SHA-256')
	return resolved


def _sha1_value(value: object, label: str) -> str:
	if not isinstance(value, str) or len(value) != 40:
		raise ValueError(f'{label} must be a 40-character Git commit')
	try:
		bytes.fromhex(value)
	except ValueError as error:
		raise ValueError(f'{label} must be a 40-character Git commit') from error
	return value


def _utc_timestamp() -> str:
	return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _validate_utc_timestamp(value: str) -> None:
	if not value.endswith('Z'):
		raise ValueError('selection lock created_at_utc must end in Z')
	try:
		parsed = datetime.fromisoformat(value.removesuffix('Z') + '+00:00')
	except ValueError as error:
		raise ValueError('selection lock created_at_utc is invalid') from error
	if parsed.tzinfo != timezone.utc:
		raise ValueError('selection lock created_at_utc must be UTC')


def _git_head() -> str:
	completed = subprocess.run(
		('git', 'rev-parse', 'HEAD'),  # noqa: S607
		cwd=REPOSITORY_ROOT,
		check=True,
		capture_output=True,
		text=True,
	)
	return _sha1_value(completed.stdout.strip(), 'current Git HEAD')


def _git_repository_state() -> dict[str, object]:  # noqa: C901, PLR0912
	"""Describe HEAD plus every dirty file that can affect this experiment."""
	completed = subprocess.run(
		(  # noqa: S607
			'git',
			'status',
			'--porcelain=v1',
			'-z',
			'--untracked-files=all',
		),
		cwd=REPOSITORY_ROOT,
		check=True,
		capture_output=True,
	)
	raw_status = completed.stdout
	status_by_path: dict[str, str] = {}
	experiment_root = Path(__file__).resolve().parent
	experiment_relative = experiment_root.relative_to(REPOSITORY_ROOT)
	records = raw_status.split(b'\0')
	index = 0
	while index < len(records):
		record = records[index]
		index += 1
		if not record:
			continue
		if len(record) < 4 or record[2:3] != b' ':
			raise ValueError('unexpected Git porcelain status record')
		status = record[:2].decode('ascii')
		path = record[3:].decode('utf-8')
		if 'R' in status or 'C' in status:
			if index >= len(records) or not records[index]:
				raise ValueError('incomplete Git rename/copy status record')
			old_path = records[index].decode('utf-8')
			index += 1
			if any(
				_is_relevant_provenance_path(Path(value), experiment_relative)
				for value in (old_path, path)
			):
				raise ValueError(
					'Git rename/copy state is unsupported while creating validation '
					f'provenance: {old_path} -> {path}'
				)
			continue
		status_by_path[path] = status
	relevant_paths = {
		path
		for path in status_by_path
		if _is_relevant_provenance_path(Path(path), experiment_relative)
	}
	for path in experiment_root.rglob('*'):
		if (
			path.is_file()
			and not path.is_symlink()
			and '__pycache__' not in path.parts
			and path.suffix != '.pyc'
		):
			relevant_paths.add(str(path.relative_to(REPOSITORY_ROOT)))
	relevant_status = [
		{'path': path, 'git_status': status_by_path[path]}
		for path in sorted(relevant_paths & set(status_by_path))
	]
	relevant_status_bytes = json.dumps(
		relevant_status, separators=(',', ':'), sort_keys=True
	).encode()
	inventory: list[dict[str, object]] = []
	for relative_value in sorted(relevant_paths):
		relative = Path(relative_value)
		path = REPOSITORY_ROOT / relative
		status = status_by_path.get(relative_value, '  ')
		if path.is_symlink():
			raise ValueError(f'provenance file must not be a symlink: {relative_value}')
		if path.is_file():
			sha256: str | None = file_sha256(path)
			state = 'file'
		elif 'D' in status:
			sha256 = None
			state = 'deleted'
		else:
			raise FileNotFoundError(
				f'relevant Git status path is not a regular file: {relative_value}'
			)
		inventory.append(
			{
				'path': relative.as_posix(),
				'git_status': status,
				'state': state,
				'sha256': sha256,
			}
		)
	return {
		'git_head': _git_head(),
		'git_dirty': bool(relevant_status),
		'relevant_git_status_sha256': hashlib.sha256(
			relevant_status_bytes
		).hexdigest(),
		'relevant_file_inventory': inventory,
	}


def _is_relevant_provenance_path(path: Path, experiment_relative: Path) -> bool:
	return (
		path.is_relative_to(experiment_relative)
		or path.is_relative_to(Path('src/seis_ssl_cluster'))
		or path.is_relative_to(Path('proc/seis_ssl_cluster'))
		or path.is_relative_to(Path('tests/seis_ssl_cluster'))
	)


def _candidate_from_mapping(value: object, *, index: int) -> CandidateSource:
	if not isinstance(value, Mapping):
		raise TypeError(f'candidates[{index}] must be a mapping')
	_require_exact_keys(
		value,
		{
			'candidate_id',
			'base_checkpoint',
			'final_checkpoint',
			'embeddings_dir',
			'view_policy',
			'gaussian_noise_std',
			'base_pretraining_epochs',
			'continuation_epochs',
		},
		f'candidates[{index}]',
	)
	candidate_id = value['candidate_id']
	if not isinstance(candidate_id, str) or candidate_id not in EXPECTED_CANDIDATES:
		raise ValueError(f'candidates[{index}].candidate_id is unsupported')
	view_policy = value['view_policy']
	expected_policy, expected_noise_std = EXPECTED_CANDIDATES[candidate_id]
	if view_policy != expected_policy:
		raise ValueError(
			f'candidates[{index}].view_policy does not match candidate ID'
		)
	noise_std = value['gaussian_noise_std']
	if not isinstance(noise_std, int | float) or isinstance(noise_std, bool):
		raise TypeError(f'candidates[{index}].gaussian_noise_std must be numeric')
	if float(noise_std) != expected_noise_std:
		raise ValueError(
			f'candidates[{index}].gaussian_noise_std does not match candidate ID'
		)
	base_epochs = _positive_int(
		value['base_pretraining_epochs'],
		f'candidates[{index}].base_pretraining_epochs',
	)
	if base_epochs != BASE_PRETRAINING_EPOCHS:
		raise ValueError('initial candidate bases must use exactly 25 epochs')
	continuation_epochs = _positive_int(
		value['continuation_epochs'],
		f'candidates[{index}].continuation_epochs',
	)
	if continuation_epochs != CONTINUATION_EPOCHS:
		raise ValueError('candidate continuations must use exactly 25 epochs')
	return CandidateSource(
		candidate_id=candidate_id,
		base_checkpoint=_absolute_path(
			value['base_checkpoint'], f'candidates[{index}].base_checkpoint'
		),
		final_checkpoint=_absolute_path(
			value['final_checkpoint'], f'candidates[{index}].final_checkpoint'
		),
		embeddings_dir=_absolute_path(
			value['embeddings_dir'], f'candidates[{index}].embeddings_dir'
		),
		view_policy=cast('str', view_policy),
		gaussian_noise_std=float(noise_std),
		base_pretraining_epochs=base_epochs,
		continuation_epochs=continuation_epochs,
		selectable=True,
	)


def _control_from_mapping(value: object, *, index: int) -> CandidateSource:
	if not isinstance(value, Mapping):
		raise TypeError(f'controls[{index}] must be a mapping')
	_require_exact_keys(
		value,
		{
			'candidate_id',
			'base_checkpoint',
			'final_checkpoint',
			'embeddings_dir',
			'base_pretraining_epochs',
			'continuation_epochs',
			'selectable',
		},
		f'controls[{index}]',
	)
	candidate_id = value['candidate_id']
	if candidate_id != LEGACY_CONTROL_ID:
		raise ValueError(f'controls[{index}].candidate_id is unsupported')
	if value['selectable'] is not False:
		raise ValueError(f'controls[{index}].selectable must be false')
	base_epochs = _positive_int(
		value['base_pretraining_epochs'],
		f'controls[{index}].base_pretraining_epochs',
	)
	if base_epochs != BASE_PRETRAINING_EPOCHS:
		raise ValueError('matched-duration control base must use exactly 25 epochs')
	continuation_epochs = _positive_int(
		value['continuation_epochs'],
		f'controls[{index}].continuation_epochs',
	)
	if continuation_epochs != CONTINUATION_EPOCHS:
		raise ValueError('control continuation must use exactly 25 epochs')
	return CandidateSource(
		candidate_id=candidate_id,
		base_checkpoint=_absolute_path(
			value['base_checkpoint'], f'controls[{index}].base_checkpoint'
		),
		final_checkpoint=_absolute_path(
			value['final_checkpoint'], f'controls[{index}].final_checkpoint'
		),
		embeddings_dir=_absolute_path(
			value['embeddings_dir'], f'controls[{index}].embeddings_dir'
		),
		view_policy=None,
		gaussian_noise_std=None,
		base_pretraining_epochs=base_epochs,
		continuation_epochs=continuation_epochs,
		selectable=False,
	)


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return child


def _require_exact_keys(
	value: Mapping[str, object], expected: set[str], label: str
) -> None:
	if set(value) != expected:
		raise ValueError(
			f'{label} must define exactly {sorted(expected)!r}; '
			f'got {sorted(value)!r}'
		)


def _absolute_path(value: object, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a path string')
	path = Path(value)
	if not path.is_absolute():
		raise ValueError(f'{label} must be absolute')
	return path


def _positive_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		raise ValueError(f'{label} must be a positive integer')
	return value


def _canonical_config(settings: ValidationSettings) -> F3FiveWayConfig:
	expected_path = REPOSITORY_ROOT / EXPECTED_CANONICAL_CONFIG_RELATIVE_PATH
	if settings.canonical_five_way_config != expected_path:
		raise ValueError(
			'benchmark.canonical_five_way_config must point to the v3 comparison'
		)
	_validate_pinned_file(
		settings.canonical_five_way_config,
		expected_sha256=settings.canonical_five_way_config_sha256,
		label='canonical five-way config',
	)
	canonical = f3_lithology_five_way_config_from_mapping(
		load_config(settings.canonical_five_way_config)
	)
	if canonical.summary_name != EXPECTED_CANONICAL_SUMMARY:
		raise ValueError('canonical comparison summary identity is not v3')
	_validate_reference_checkpoints(
		canonical_config=canonical,
		reference_base_checkpoint=settings.reference_base_checkpoint,
		reference_final_checkpoint=settings.reference_final_checkpoint,
		verify_files=False,
	)
	return canonical


def _validate_benchmark_provenance(
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	*,
	verify_files: bool,
) -> dict[str, object]:
	"""Bind every reused baseline and pretraining input to a pinned digest."""
	random_checkpoint = canonical.model_by_id('random').checkpoint
	comparison = canonical.summary_root / 'comparison.csv'
	expected_manifest = (
		canonical.artifact_root / EXPECTED_PRETRAINING_MANIFEST_RELATIVE_PATH
	)
	expected_path_list = (
		canonical.artifact_root / EXPECTED_PRETRAINING_PATH_LIST_RELATIVE_PATH
	)
	if not verify_files:
		return {
			'canonical_five_way_config': str(settings.canonical_five_way_config),
			'canonical_five_way_config_sha256': (
				settings.canonical_five_way_config_sha256
			),
			'reference_base_checkpoint': str(settings.reference_base_checkpoint),
			'reference_base_checkpoint_sha256': (
				settings.reference_base_checkpoint_sha256
			),
			'reference_final_checkpoint': str(settings.reference_final_checkpoint),
			'reference_final_checkpoint_sha256': (
				settings.reference_final_checkpoint_sha256
			),
			'random_checkpoint': str(random_checkpoint),
			'random_checkpoint_sha256': settings.random_checkpoint_sha256,
			'canonical_comparison': str(comparison),
			'canonical_comparison_sha256': settings.canonical_comparison_sha256,
			'pretraining_manifest': str(expected_manifest),
			'pretraining_manifest_sha256': settings.pretraining_manifest_sha256,
			'pretraining_path_list': str(expected_path_list),
			'pretraining_path_list_sha256': settings.pretraining_path_list_sha256,
		}
	_validate_pinned_file(
		settings.canonical_five_way_config,
		expected_sha256=settings.canonical_five_way_config_sha256,
		label='canonical five-way config',
	)
	_validate_pinned_file(
		settings.reference_base_checkpoint,
		expected_sha256=settings.reference_base_checkpoint_sha256,
		label='canonical reference base checkpoint',
	)
	_validate_pinned_file(
		settings.reference_final_checkpoint,
		expected_sha256=settings.reference_final_checkpoint_sha256,
		label='canonical reference final checkpoint',
	)
	_validate_pinned_file(
		random_checkpoint,
		expected_sha256=settings.random_checkpoint_sha256,
		label='canonical random checkpoint',
	)
	_validate_pinned_file(
		comparison,
		expected_sha256=settings.canonical_comparison_sha256,
		label='canonical comparison.csv',
	)
	_validate_pinned_file(
		expected_manifest,
		expected_sha256=settings.pretraining_manifest_sha256,
		label='pretraining manifest',
	)
	_validate_pinned_file(
		expected_path_list,
		expected_sha256=settings.pretraining_path_list_sha256,
		label='pretraining path list',
	)
	reference_payload = load_checkpoint_metadata_without_weights(
		settings.reference_base_checkpoint
	)
	reference_config = _mapping(reference_payload, 'config')
	reference_manifests = _mapping(reference_config, 'manifests')
	for key, expected in (
		('train', expected_manifest),
		('train_path_list', expected_path_list),
	):
		value = reference_manifests.get(key)
		if not isinstance(value, str) or Path(value).resolve() != expected.resolve():
			raise ValueError(
				f'canonical reference base manifests.{key} does not resolve to '
				f'the pinned F3 pretraining input: {expected}'
			)
	return {
		'canonical_five_way_config': str(settings.canonical_five_way_config),
		'canonical_five_way_config_sha256': file_sha256(
			settings.canonical_five_way_config
		),
		'reference_base_checkpoint': str(settings.reference_base_checkpoint),
		'reference_base_checkpoint_sha256': file_sha256(
			settings.reference_base_checkpoint
		),
		'reference_final_checkpoint': str(settings.reference_final_checkpoint),
		'reference_final_checkpoint_sha256': file_sha256(
			settings.reference_final_checkpoint
		),
		'random_checkpoint': str(random_checkpoint),
		'random_checkpoint_sha256': file_sha256(random_checkpoint),
		'canonical_comparison': str(comparison),
		'canonical_comparison_sha256': file_sha256(comparison),
		'pretraining_manifest': str(expected_manifest),
		'pretraining_manifest_sha256': file_sha256(expected_manifest),
		'pretraining_path_list': str(expected_path_list),
		'pretraining_path_list_sha256': file_sha256(expected_path_list),
	}


def _validate_pinned_file(
	path: Path, *, expected_sha256: str, label: str
) -> None:
	if path.is_symlink():
		raise ValueError(f'{label} must not be a symlink: {path}')
	if not path.is_file():
		raise FileNotFoundError(f'missing {label}: {path}')
	if file_sha256(path) != expected_sha256:
		raise ValueError(f'{label} SHA-256 does not match the pinned identity')


def _validate_reference_checkpoints(
	*,
	canonical_config: F3FiveWayConfig,
	reference_base_checkpoint: Path,
	reference_final_checkpoint: Path,
	verify_files: bool,
) -> None:
	_validate_reference_base_checkpoint(
		canonical_config=canonical_config,
		reference_base_checkpoint=reference_base_checkpoint,
		verify_file=verify_files,
	)
	_validate_reference_final_checkpoint(
		canonical_config=canonical_config,
		reference_final_checkpoint=reference_final_checkpoint,
		verify_file=verify_files,
	)


def _validate_reference_base_checkpoint(
	*,
	canonical_config: F3FiveWayConfig,
	reference_base_checkpoint: Path,
	verify_file: bool,
) -> None:
	expected_base = (
		canonical_config.artifact_root / EXPECTED_REFERENCE_BASE_RELATIVE_PATH
	)
	if reference_base_checkpoint.resolve() != expected_base.resolve():
		raise ValueError(
			'benchmark.reference_base_checkpoint must use the pinned canonical '
			f'Local Barlow base path: {expected_base}'
		)
	if not verify_file:
		return
	if not reference_base_checkpoint.is_file():
		raise FileNotFoundError(
			f'missing reference base checkpoint: {reference_base_checkpoint}'
		)
	if file_sha256(reference_base_checkpoint) != EXPECTED_REFERENCE_BASE_SHA256:
		raise ValueError(
			'reference base checkpoint SHA-256 does not match the pinned '
			'canonical Local Barlow checkpoint'
		)


def _validate_reference_final_checkpoint(
	*,
	canonical_config: F3FiveWayConfig,
	reference_final_checkpoint: Path,
	verify_file: bool,
) -> None:
	expected_final = (
		canonical_config.artifact_root / EXPECTED_REFERENCE_FINAL_RELATIVE_PATH
	)
	if reference_final_checkpoint.resolve() != expected_final.resolve():
		raise ValueError(
			'benchmark.reference_final_checkpoint must use the pinned canonical '
			f'Local Barlow continuation path: {expected_final}'
		)
	canonical_final = canonical_config.model_by_id('local_barlow_twins').checkpoint
	if canonical_final.resolve() != expected_final.resolve():
		raise ValueError(
			'canonical v3 Local Barlow source is not the pinned continuation checkpoint'
		)
	if not verify_file:
		return
	if not reference_final_checkpoint.is_file():
		raise FileNotFoundError(
			f'missing reference final checkpoint: {reference_final_checkpoint}'
		)
	if file_sha256(reference_final_checkpoint) != EXPECTED_REFERENCE_FINAL_SHA256:
		raise ValueError(
			'reference final checkpoint SHA-256 does not match the pinned '
			'canonical Local Barlow checkpoint'
		)


def build_parser() -> argparse.ArgumentParser:
	"""Build the one-cell validation CLI."""
	parser = argparse.ArgumentParser(
		description='Run one validation-only Gaussian-view candidate job.'
	)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument(
		'--candidate',
		choices=sorted(set(EXPECTED_CANDIDATES) | EXPECTED_CONTROLS),
	)
	parser.add_argument('--layout', choices=LAYOUT_IDS)
	parser.add_argument('--size', choices=DATA_SIZES)
	parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
	parser.add_argument('--resume', type=Path)
	parser.add_argument('--dry-run', action='store_true')
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument('--audit-base-checkpoint-only', action='store_true')
	mode.add_argument('--audit-checkpoint-only', action='store_true')
	mode.add_argument('--create-protocol-lock', action='store_true')
	mode.add_argument('--create-selection-lock', action='store_true')
	mode.add_argument('--create-final-result', action='store_true')
	return parser


def _validate_cli_arguments(  # noqa: C901, PLR0912
	args: argparse.Namespace,
) -> None:
	if args.audit_base_checkpoint_only or args.audit_checkpoint_only:
		if args.candidate is None:
			raise ValueError('checkpoint audit modes require --candidate')
		if any(value is not None for value in (args.layout, args.size, args.resume)):
			raise ValueError(
				'checkpoint audit modes do not accept --layout, --size, or --resume'
			)
		if args.dry_run:
			raise ValueError('checkpoint audit modes are already read-only')
		return
	if args.create_protocol_lock:
		if any(
			value is not None
			for value in (args.candidate, args.layout, args.size, args.resume)
		):
			raise ValueError(
				'--create-protocol-lock does not accept job-selection arguments'
			)
		if args.dry_run:
			raise ValueError('--create-protocol-lock does not support --dry-run')
		return
	if args.create_selection_lock:
		if any(
			value is not None
			for value in (args.candidate, args.layout, args.size, args.resume)
		):
			raise ValueError(
				'--create-selection-lock does not accept job-selection arguments'
			)
		if args.dry_run:
			raise ValueError('--create-selection-lock does not support --dry-run')
		return
	if args.create_final_result:
		if any(
			value is not None
			for value in (args.candidate, args.layout, args.size, args.resume)
		):
			raise ValueError(
				'--create-final-result does not accept job-selection arguments'
			)
		if args.dry_run:
			raise ValueError('--create-final-result does not support --dry-run')
		return
	missing = [
		flag
		for flag, value in (
			('--candidate', args.candidate),
			('--layout', args.layout),
			('--size', args.size),
		)
		if value is None
	]
	if missing:
		raise ValueError(f'validation jobs require {", ".join(missing)}')


def main() -> None:  # noqa: C901, PLR0912
	"""Resolve, audit, and execute one candidate validation job."""
	args = build_parser().parse_args()
	_validate_cli_arguments(args)
	settings = validation_settings_from_mapping(load_config(args.config))
	canonical = _canonical_config(settings)
	_validate_benchmark_provenance(settings, canonical, verify_files=True)
	if args.audit_base_checkpoint_only:
		if settings.protocol_lock.exists() or settings.protocol_lock.is_symlink():
			raise ValueError(
				'--audit-base-checkpoint-only is the sole pre-protocol operation; '
				'the protocol lock already exists'
			)
		candidate = settings.source_by_id(args.candidate)
		result = audit_candidate_base_checkpoint(
			candidate=candidate,
			canonical_config=canonical,
			reference_base_checkpoint=settings.reference_base_checkpoint,
		)
		for key, value in result.items():
			print(f'{key}: {value}')
		return
	if args.create_protocol_lock:
		result = create_gaussian25_protocol_lock(settings, canonical)
		print(f'base_checkpoint_inputs: {result["base_checkpoint_inputs"]}')
		print(f'protocol_lock: {settings.protocol_lock}')
		return
	protocol_lock = validate_gaussian25_protocol_lock(settings, canonical)
	protocol_lock_identity = _protocol_lock_identity(settings, protocol_lock)
	if args.audit_checkpoint_only:
		candidate = settings.source_by_id(args.candidate)
		result = audit_candidate_checkpoints(
			candidate=candidate,
			canonical_config=canonical,
			reference_base_checkpoint=settings.reference_base_checkpoint,
			reference_final_checkpoint=settings.reference_final_checkpoint,
		)
		result = {**result, 'protocol_lock': protocol_lock_identity}
		for key, value in result.items():
			print(f'{key}: {value}')
		return
	if args.create_selection_lock:
		five_way_sources.audit_f3_lithology_five_way_sources(canonical)
		result = create_gaussian25_selection_lock(settings, canonical)
		for key in (
			'selected_candidate_id',
			'selected_view_policy',
			'selected_gaussian_noise_std',
			'candidate_means',
			'fixed_strength_geometry_contrast',
		):
			print(f'{key}: {result[key]}')
		print(f'selection_lock: {settings.selection_lock}')
		return
	if args.create_final_result:
		result = create_gaussian25_final_result(settings, canonical)
		for key in (
			'passed',
			'winner_candidate_id',
			'authorizes_next_base_duration',
			'failure_stage',
			'medium_gate',
			'gaussian_attribution',
		):
			print(f'{key}: {result[key]}')
		print(f'final_result: {settings.final_result}')
		return
	candidate = settings.source_by_id(args.candidate)
	order_evidence = enforce_validation_order(
		settings=settings,
		canonical=canonical,
		candidate=candidate,
		data_size=args.size,
	)
	candidate_config = _candidate_config(
		canonical, candidate=candidate, runs_root=settings.runs_root
	)
	job = five_way_runner.resolve_f3_lithology_five_way_job(
		candidate_config,
		model=candidate.candidate_id,
		layout=args.layout,
		size=args.size,
	)
	if args.resume is not None and args.resume != (
		job.decoder_dir / five_way_runner.LATEST_CHECKPOINT_NAME
	):
		raise ValueError(
			'--resume must be this job decoder latest.pt: '
			f'{job.decoder_dir / five_way_runner.LATEST_CHECKPOINT_NAME}'
		)
	if args.dry_run:
		summary = five_way_runner.inspect_f3_lithology_five_way_job(job)
		summary.update(
			{
				'candidate_source_audit': 'required before execution',
				'evaluation_split': 'validation',
				'evaluation_aggregation_unit': VALIDATION_AGGREGATION_UNIT,
				'selection_eligible': candidate.selectable,
				'protocol_lock': protocol_lock_identity,
				'execution': 'dry-run; no files written',
			}
		)
		for key, value in summary.items():
			print(f'{key}: {value}')
		return
	five_way_sources.audit_f3_lithology_five_way_sources(canonical)
	audit = audit_candidate_source(
		candidate=candidate,
		candidate_model=job.model,
		canonical_config=canonical,
		reference_base_checkpoint=settings.reference_base_checkpoint,
		reference_final_checkpoint=settings.reference_final_checkpoint,
		protocol_lock_identity=protocol_lock_identity,
	)
	if order_evidence is not None:
		audit = {
			**audit,
			'validation_order_provenance': _validation_order_provenance(
				settings, order_evidence
			),
		}
	result = _run_job(
		job, audit=audit, device=args.device, resume=args.resume
	)
	for key, value in result.items():
		print(f'{key}: {value}')


if __name__ == '__main__':
	main()
