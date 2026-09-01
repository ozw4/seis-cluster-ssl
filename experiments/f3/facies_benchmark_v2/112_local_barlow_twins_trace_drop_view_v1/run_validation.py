# ruff: noqa: INP001, SLF001, TC001
"""Run the preregistered trace-drop, validation-only F3 comparison.

The experiment has one fixed candidate. Its protocol is sealed after the
fresh one-epoch base and before continuation, and freezes all 15 canonical
random cells before any candidate metric exists. The failed Gaussian base-1
result supplies medium-only attribution controls; it never selects or gates
the trace-drop candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_five_way import (
	F3FiveWayConfig,
	F3FiveWayModelSource,
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
from seis_ssl_cluster.training.random_checkpoint import (
	load_checkpoint_metadata_without_weights,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
GAUSSIAN25_RUNNER = (
	EXPERIMENT_ROOT.parent
	/ '111_local_barlow_twins_gaussian_view_v1/run_validation.py'
)
GAUSSIAN25_RUNNER_SHA256 = (
	'a704e64a2da59c85a4e82e318bb3156c2cf74825ad50b4fb7463bd8dd2c1bccd'
)

CANDIDATE_ID = 'local_barlow_twins_horizontal_trace_drop_p001_base1ep'
GAUSSIAN_CONTROL_ID = 'local_barlow_twins_gaussian_noise_std010_base1ep'
LEGACY_CONTROL_ID = 'local_barlow_twins_legacy_flip_base1ep'
RANDOM_ID = 'random'
TRACE_DROP_POLICY = 'horizontal_flip_trace_drop_v1'
EXPECTED_AUGMENTATIONS = {
	'policy': TRACE_DROP_POLICY,
	'horizontal_flip_probability': 0.5,
	'trace_drop_probability': 0.01,
}
FALLBACK_TRACE_DROP_PROBABILITY = 0.02
LOCAL_BARLOW_METHOD = 'local_barlow_twins_3d'
LOCAL_PAIRS_PER_CROP = 128
BASE_PRETRAINING_EPOCHS = 1
EXPECTED_BASE_STEPS = 625
CONTINUATION_EPOCHS = 25
EXPECTED_CONTINUATION_STEPS = 15_625
CONTINUATION_UNFREEZE_TOP_BLOCKS = 1
EXPECTED_TRAINED_PARAMETER_PREFIXES = ['patch_projection.', 'encoder.']
EXPECTED_BASE_TRAINING_STATE = {
	'schema_version': 1,
	'stage': 'barlow_twins_training',
	'resume_boundary': 'epoch',
	'dataset_epoch': 0,
	'completed_epoch': True,
}
EXPECTED_CONTINUATION_TRAINING_STATE = {
	'schema_version': 1,
	'stage': 'barlow_twins_training',
	'resume_boundary': 'epoch',
	'dataset_epoch': 24,
	'completed_epoch': True,
}
EXPECTED_BENCHMARK_SHA256 = {
	'canonical_five_way_config_sha256': (
		'285b0233ff82fe83808f82e929b611f570a67f01fa983ef191dda23d1858061b'
	),
	'reference_base_checkpoint_sha256': (
		'84550ed658166e8e6a40cd664e2e9ffbeab0c12d6917006abb417cd25e228ac0'
	),
	'reference_final_checkpoint_sha256': (
		'1c5312244f290dbfdcf2688ffa9fa8b5c64452ade162d5335be1bb8a0e256291'
	),
	'random_checkpoint_sha256': (
		'6548d52446e7d6b9b57acd2bd39a8389a76bc5df55b52a9eda0472eb182a438c'
	),
	'canonical_comparison_sha256': (
		'b135122a7db2b6b359817096ac546f99d4e4fac1ee003a99ce7289c0445cf913'
	),
	'pretraining_manifest_sha256': (
		'c5dbc3a66a5c2eed0ec5df8745f8bf5a461b1e2e66156700091f1a751bdc0ef5'
	),
	'pretraining_path_list_sha256': (
		'b52fd5e0c57edb2d2158be12b94046b554b5e6e13ba17008321bcdbe0ae2acb1'
	),
}

PARENT_FINAL_RESULT_SHA256 = (
	'6ab9cb2ae8bf89eb5dea9ab34244b60f41499f44cc046995e41ec0350854e1fa'
)
PARENT_FINAL_RESULT_TYPE = 'f3_local_barlow_twins_gaussian_base1_final_validation_v1'
PARENT_FINAL_RESULT_RELATIVE_PATH = Path(
	'f3_lithology_benchmark/local_barlow_twins_gaussian_view_v1/'
	'base1ep/validation/gaussian_base1_final_result.json'
)
EXPECTED_BASE_RELATIVE_PATH = Path(
	'pretraining/f3/facies_benchmark_v1/local_barlow_twins_trace_drop_view_v1/'
	'base1ep/stage1/horizontal_trace_drop_p001_base1ep/full_1ep/latest.pt'
)
EXPECTED_FINAL_RELATIVE_PATH = Path(
	'pretraining/f3/facies_benchmark_v1/local_barlow_twins_trace_drop_view_v1/'
	'base1ep/stage2/horizontal_trace_drop_p001_base1ep/'
	'local_bt_continue/full_25ep/latest.pt'
)
EXPECTED_EMBEDDING_RELATIVE_PATH = Path(
	'embeddings/f3/facies_benchmark_v2/local_barlow_twins_trace_drop_view_v1/'
	'base1ep/local_barlow_twins_horizontal_trace_drop_p001_base1ep/overlap_x64'
)
VALIDATION_RELATIVE_ROOT = Path(
	'f3_lithology_benchmark/local_barlow_twins_trace_drop_view_v1/'
	'base1ep/validation'
)

CHECKPOINT_AUDIT_SCHEMA_VERSION = 1
PROTOCOL_LOCK_SCHEMA_VERSION = 1
PROTOCOL_LOCK_TYPE = (
	'f3_local_barlow_twins_horizontal_trace_drop_p001_protocol_v1'
)
FINAL_RESULT_SCHEMA_VERSION = 1
FINAL_RESULT_TYPE = (
	'f3_local_barlow_twins_horizontal_trace_drop_p001_final_validation_v1'
)
VALIDATION_AGGREGATION_UNIT = 'unique_validation_voxel'
AUDIT_NAME = 'candidate_source_audit.json'


def _load_gaussian25_namespace() -> dict[str, object]:
	"""Load only stable, generic benchmark helpers from the frozen runner."""
	if GAUSSIAN25_RUNNER.is_symlink() or not GAUSSIAN25_RUNNER.is_file():
		raise FileNotFoundError(
			f'missing frozen Gaussian25 runner: {GAUSSIAN25_RUNNER}'
		)
	if file_sha256(GAUSSIAN25_RUNNER) != GAUSSIAN25_RUNNER_SHA256:
		raise ValueError('frozen Gaussian25 runner SHA-256 changed')
	namespace = runpy.run_path(str(GAUSSIAN25_RUNNER))
	required = {
		'_arm_random_result',
		'_candidate_config',
		'_canonical_config',
		'_git_repository_state',
		'_read_random_job_evidence',
		'_run_job',
		'_validate_benchmark_provenance',
		'_validate_job_evaluation_identity',
		'_validate_reference_base_checkpoint',
		'_validate_reference_final_checkpoint',
	}
	missing = required - set(namespace)
	if missing:
		raise RuntimeError(
			f'frozen Gaussian25 helper API is incomplete: {sorted(missing)!r}'
		)
	return namespace


_GAUSSIAN25 = _load_gaussian25_namespace()


def _shared(name: str) -> Any:  # noqa: ANN401
	return cast('Any', _GAUSSIAN25[name])


@dataclass(frozen=True)
class CandidateSource:
	"""The one preregistered trace-drop source."""

	candidate_id: str
	role: str
	base_checkpoint: Path
	final_checkpoint: Path
	embeddings_dir: Path
	augmentations: Mapping[str, object]
	base_pretraining_epochs: int
	continuation_epochs: int
	view_policy: str = TRACE_DROP_POLICY
	gaussian_noise_std: float | None = None
	selectable: bool = True


@dataclass(frozen=True)
class ValidationSettings:
	"""Pinned parent, benchmark, candidate, and isolated outputs."""

	parent_final_result: Path
	parent_final_result_sha256: str
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
	candidate: CandidateSource
	runs_root: Path
	protocol_lock: Path
	final_result: Path

	@property
	def candidates(self) -> tuple[CandidateSource, ...]:
		"""Compatibility view for the frozen generic config helper."""
		return (self.candidate,)

	@property
	def controls(self) -> tuple[CandidateSource, ...]:
		"""No live attribution control is trained in this experiment."""
		return ()

	def source_by_id(self, source_id: str) -> CandidateSource:
		"""Return the sole candidate by its fixed ID."""
		if source_id != CANDIDATE_ID:
			raise ValueError(f'unknown trace-drop candidate: {source_id!r}')
		return self.candidate


def validation_settings_from_mapping(
	config: Mapping[str, object],
) -> ValidationSettings:
	"""Resolve the exact one-candidate configuration without defaults."""
	_require_exact_keys(
		config, {'parent', 'benchmark', 'candidate', 'outputs'}, 'config'
	)
	parent = _mapping(config, 'parent')
	benchmark = _mapping(config, 'benchmark')
	candidate_value = _mapping(config, 'candidate')
	outputs = _mapping(config, 'outputs')
	_require_exact_keys(parent, {'final_result', 'final_result_sha256'}, 'parent')
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
		candidate_value,
		{
			'candidate_id',
			'role',
			'base_checkpoint',
			'final_checkpoint',
			'embeddings_dir',
			'augmentations',
			'base_pretraining_epochs',
			'continuation_epochs',
		},
		'candidate',
	)
	_require_exact_keys(
		outputs, {'runs_root', 'protocol_lock', 'final_result'}, 'outputs'
	)
	if candidate_value.get('candidate_id') != CANDIDATE_ID:
		raise ValueError('candidate.candidate_id is not the preregistered p=.01 arm')
	if candidate_value.get('role') != 'preregistered_trace_drop_view':
		raise ValueError(
			'candidate.role must identify the preregistered trace-drop view'
		)
	augmentations = _mapping(candidate_value, 'augmentations')
	if not _type_sensitive_equal(augmentations, EXPECTED_AUGMENTATIONS):
		raise ValueError(
			'candidate augmentations differ from the preregistered mapping'
		)
	if not _type_sensitive_equal(
		candidate_value.get('base_pretraining_epochs'), BASE_PRETRAINING_EPOCHS
	):
		raise ValueError('candidate base duration must equal one epoch')
	if not _type_sensitive_equal(
		candidate_value.get('continuation_epochs'), CONTINUATION_EPOCHS
	):
		raise ValueError('candidate continuation duration must equal 25 epochs')
	parent_sha = _sha256_value(parent.get('final_result_sha256'), 'parent SHA-256')
	if parent_sha != PARENT_FINAL_RESULT_SHA256:
		raise ValueError('parent final-result SHA-256 is not the pinned base1 result')
	for key, expected in EXPECTED_BENCHMARK_SHA256.items():
		if _sha256_value(benchmark.get(key), f'benchmark.{key}') != expected:
			raise ValueError(f'benchmark.{key} is not the pinned canonical digest')
	candidate = CandidateSource(
		candidate_id=CANDIDATE_ID,
		role='preregistered_trace_drop_view',
		base_checkpoint=_absolute_path(
			candidate_value.get('base_checkpoint'), 'candidate.base_checkpoint'
		),
		final_checkpoint=_absolute_path(
			candidate_value.get('final_checkpoint'), 'candidate.final_checkpoint'
		),
		embeddings_dir=_absolute_path(
			candidate_value.get('embeddings_dir'), 'candidate.embeddings_dir'
		),
		augmentations=dict(EXPECTED_AUGMENTATIONS),
		base_pretraining_epochs=BASE_PRETRAINING_EPOCHS,
		continuation_epochs=CONTINUATION_EPOCHS,
	)
	return ValidationSettings(
		parent_final_result=_absolute_path(
			parent.get('final_result'), 'parent.final_result'
		),
		parent_final_result_sha256=parent_sha,
		canonical_five_way_config=_absolute_path(
			benchmark.get('canonical_five_way_config'),
			'benchmark.canonical_five_way_config',
		),
		canonical_five_way_config_sha256=cast(
			'str', benchmark['canonical_five_way_config_sha256']
		),
		reference_base_checkpoint=_absolute_path(
			benchmark.get('reference_base_checkpoint'),
			'benchmark.reference_base_checkpoint',
		),
		reference_base_checkpoint_sha256=cast(
			'str', benchmark['reference_base_checkpoint_sha256']
		),
		reference_final_checkpoint=_absolute_path(
			benchmark.get('reference_final_checkpoint'),
			'benchmark.reference_final_checkpoint',
		),
		reference_final_checkpoint_sha256=cast(
			'str', benchmark['reference_final_checkpoint_sha256']
		),
		random_checkpoint_sha256=cast('str', benchmark['random_checkpoint_sha256']),
		canonical_comparison_sha256=cast(
			'str', benchmark['canonical_comparison_sha256']
		),
		pretraining_manifest_sha256=cast(
			'str', benchmark['pretraining_manifest_sha256']
		),
		pretraining_path_list_sha256=cast(
			'str', benchmark['pretraining_path_list_sha256']
		),
		candidate=candidate,
		runs_root=_absolute_path(outputs.get('runs_root'), 'outputs.runs_root'),
		protocol_lock=_absolute_path(
			outputs.get('protocol_lock'), 'outputs.protocol_lock'
		),
		final_result=_absolute_path(
			outputs.get('final_result'), 'outputs.final_result'
		),
	)


def _canonical_config(settings: ValidationSettings) -> F3FiveWayConfig:
	canonical = cast('F3FiveWayConfig', _shared('_canonical_config')(settings))
	_validate_experiment_paths(settings, canonical)
	return canonical


def _validate_experiment_paths(
	settings: ValidationSettings, canonical: F3FiveWayConfig
) -> None:
	artifact_root = canonical.artifact_root
	if settings.parent_final_result != (
		artifact_root / PARENT_FINAL_RESULT_RELATIVE_PATH
	):
		raise ValueError('parent result path is not the pinned base1 result')
	if settings.candidate.base_checkpoint != (
		artifact_root / EXPECTED_BASE_RELATIVE_PATH
	):
		raise ValueError('candidate base checkpoint path changed')
	if settings.candidate.final_checkpoint != (
		artifact_root / EXPECTED_FINAL_RELATIVE_PATH
	):
		raise ValueError('candidate final checkpoint path changed')
	if settings.candidate.embeddings_dir != (
		artifact_root / EXPECTED_EMBEDDING_RELATIVE_PATH
	):
		raise ValueError('candidate embedding path changed')
	validation_root = artifact_root / VALIDATION_RELATIVE_ROOT
	if settings.runs_root != validation_root / 'runs':
		raise ValueError('runs_root is outside the isolated trace-drop namespace')
	if settings.protocol_lock != validation_root / 'trace_drop_p001_protocol_lock.json':
		raise ValueError('protocol lock path changed')
	if settings.final_result != validation_root / 'trace_drop_p001_final_result.json':
		raise ValueError('final result path changed')


def validate_parent_result(  # noqa: C901
	settings: ValidationSettings, canonical: F3FiveWayConfig
) -> dict[str, object]:
	"""Pin the failed base1 result and replay its medium-only control evidence."""
	parent, parent_sha = _read_hashed_json(
		settings.parent_final_result, label='parent Gaussian base1 final result'
	)
	if parent_sha != settings.parent_final_result_sha256:
		raise ValueError('parent Gaussian base1 final-result SHA-256 changed')
	expected_values: Mapping[str, object] = {
		'schema_version': 1,
		'final_result_type': PARENT_FINAL_RESULT_TYPE,
		'validation_only': True,
		'base_pretraining_epochs': 1,
		'continuation_epochs': 25,
		'passed': False,
		'winner_candidate_id': None,
		'authorizes_next_base_duration': False,
		'authorized_next_base_pretraining_epochs': None,
		'failure_stage': 'medium_5of5',
	}
	for key, expected in expected_values.items():
		if not _type_sensitive_equal(parent.get(key), expected):
			raise ValueError(f'parent {key} must equal {expected!r}')
	gate = _mapping(parent, 'medium_gate')
	if not _type_sensitive_equal(
		gate,
		{
			'gate_open': False,
			'selected_wins_over_random': False,
			'legacy_wins_over_random': False,
		},
	):
		raise ValueError('parent medium gate changed')
	controls = _mapping_rows(parent.get('candidate_inputs'), 'parent candidate_inputs')
	parent_random = _mapping_rows(parent.get('random_inputs'), 'parent random_inputs')
	expected_control_cells = {
		(source_id, layout_id, 'medium')
		for source_id in (GAUSSIAN_CONTROL_ID, LEGACY_CONTROL_ID)
		for layout_id in LAYOUT_IDS
	}
	if {_cell_key(row) for row in controls} != expected_control_cells:
		raise ValueError('parent does not contain the exact ten frozen control cells')
	_validate_frozen_control_inputs(controls)
	expected_random_cells = {
		(RANDOM_ID, layout_id, 'medium') for layout_id in LAYOUT_IDS
	}
	if {_cell_key(row) for row in parent_random} != expected_random_cells:
		raise ValueError('parent does not contain the exact five medium random cells')
	fresh_random = [
		cast(
			'dict[str, object]',
			_shared('_read_random_job_evidence')(
				canonical, layout_id=layout_id, data_size='medium'
			),
		)
		for layout_id in LAYOUT_IDS
	]
	if not _type_sensitive_equal(parent_random, fresh_random):
		raise ValueError('parent random rows differ from fresh canonical evidence')
	scores = _scores_from_evidence((*controls, *parent_random))
	arm_results = _mapping(parent, 'arm_results')
	for source_id in (GAUSSIAN_CONTROL_ID, LEGACY_CONTROL_ID):
		recomputed = cast(
			'dict[str, object]',
			_shared('_arm_random_result')(
				scores, arm_id=source_id, medium_gate_open=False
			),
		)
		if not _type_sensitive_equal(arm_results.get(source_id), recomputed):
			raise ValueError(f'parent {source_id} result does not recompute')
	recomputed_attribution = _medium_control_contrast(
		scores,
		left_id=GAUSSIAN_CONTROL_ID,
		right_id=LEGACY_CONTROL_ID,
		contrast_id='selected_gaussian_minus_matched_legacy',
	)
	parent_attribution = _mapping(parent, 'gaussian_attribution')
	parent_projection = dict(parent_attribution)
	parent_projection['wins_all_5'] = parent_projection.pop('wins_all_15')
	recomputed_parent_projection = dict(recomputed_attribution)
	recomputed_parent_projection.pop('data_size')
	if not _type_sensitive_equal(parent_projection, recomputed_parent_projection):
		raise ValueError('parent Gaussian-minus-legacy attribution does not recompute')
	return {
		'path': str(settings.parent_final_result),
		'sha256': parent_sha,
		'schema_version': 1,
		'final_result_type': PARENT_FINAL_RESULT_TYPE,
		'passed': False,
		'failure_stage': 'medium_5of5',
		'frozen_medium_control_ids': [GAUSSIAN_CONTROL_ID, LEGACY_CONTROL_ID],
		'frozen_medium_control_inputs': [dict(row) for row in controls],
		'frozen_parent_random_inputs': [dict(row) for row in parent_random],
		'recomputed_parent_arm_results': {
			source_id: dict(cast('Mapping[str, object]', arm_results[source_id]))
			for source_id in (GAUSSIAN_CONTROL_ID, LEGACY_CONTROL_ID)
		},
		'recomputed_parent_gaussian_attribution': dict(parent_attribution),
		'fresh_random_type_sensitive_match': True,
	}


def _validate_frozen_control_inputs(
	controls: Sequence[Mapping[str, object]],
) -> None:
	"""Revalidate every parent control metric and audit by exact digest."""
	for row in controls:
		metrics_path = _absolute_path(
			row.get('metrics_path'), 'parent control metrics_path'
		)
		metrics, metrics_sha = _read_hashed_json(
			metrics_path, label='parent control metrics'
		)
		if metrics_sha != _sha256_value(
			row.get('metrics_sha256'), 'parent control metrics_sha256'
		):
			raise ValueError('parent control metrics SHA-256 changed')
		_macro_f1(metrics, metrics_path)
		if not _type_sensitive_equal(metrics.get('macro_f1'), row.get('macro_f1')):
			raise ValueError('parent control macro_f1 differs from its metrics file')
		audit_path = _absolute_path(
			row.get('candidate_audit_path'), 'parent control candidate_audit_path'
		)
		_, audit_sha = _read_hashed_json(
			audit_path, label='parent control candidate audit'
		)
		if audit_sha != _sha256_value(
			row.get('candidate_audit_sha256'),
			'parent control candidate_audit_sha256',
		):
			raise ValueError('parent control candidate-audit SHA-256 changed')


def audit_candidate_base_checkpoint(
	*,
	candidate: CandidateSource,
	canonical_config: F3FiveWayConfig,
	reference_base_checkpoint: Path,
) -> dict[str, object]:
	"""Audit the fresh one-epoch base before the protocol is sealed."""
	_shared('_validate_reference_base_checkpoint')(
		canonical_config=canonical_config,
		reference_base_checkpoint=reference_base_checkpoint,
		verify_file=True,
	)
	_validate_regular_file(candidate.base_checkpoint, label='candidate base checkpoint')
	payload = load_checkpoint_metadata_without_weights(candidate.base_checkpoint)
	reference = load_checkpoint_metadata_without_weights(reference_base_checkpoint)
	_validate_candidate_base_checkpoint(candidate, payload=payload, reference=reference)
	return {
		'audit_schema_version': CHECKPOINT_AUDIT_SCHEMA_VERSION,
		'audit_type': 'trace_drop_base1_pre_continuation_checkpoint_only',
		'candidate_id': CANDIDATE_ID,
		'base_checkpoint': str(candidate.base_checkpoint),
		'base_checkpoint_sha256': file_sha256(candidate.base_checkpoint),
		'base_pretraining_epochs': BASE_PRETRAINING_EPOCHS,
		'base_global_step': EXPECTED_BASE_STEPS,
		'base_resume_count': 0,
		'augmentations': dict(EXPECTED_AUGMENTATIONS),
		'reference_base_checkpoint': str(reference_base_checkpoint),
		'reference_base_checkpoint_sha256': file_sha256(reference_base_checkpoint),
		'base_parity_exceptions': [
			'augmentations',
			'train.epochs',
			'paths.output_root',
		],
		'final_checkpoint_required': False,
		'embeddings_required': False,
		'passed': True,
	}


def _validate_candidate_base_checkpoint(
	candidate: CandidateSource,
	*,
	payload: Mapping[str, object],
	reference: Mapping[str, object],
) -> None:
	if payload.get('continuation_lineage') is not None:
		raise ValueError('candidate base must not record continuation lineage')
	_common_checkpoint_identity(
		payload,
		epochs=BASE_PRETRAINING_EPOCHS,
		global_step=EXPECTED_BASE_STEPS,
		training_state=EXPECTED_BASE_TRAINING_STATE,
		label='candidate base',
	)
	config = _mapping(payload, 'config')
	reference_config = _mapping(reference, 'config')
	if _mapping(config, 'paths').get('output_root') != str(
		candidate.base_checkpoint.parent
	):
		raise ValueError('candidate base output_root must own its checkpoint')
	if not _type_sensitive_equal(
		_mapping(config, 'augmentations'), EXPECTED_AUGMENTATIONS
	):
		raise ValueError('candidate base records unexpected augmentations')
	_validate_barlow_objective(config, label='candidate base')
	_validate_training_steps(
		config,
		epochs=BASE_PRETRAINING_EPOCHS,
		expected_steps=EXPECTED_BASE_STEPS,
		label='candidate base',
	)
	if not _type_sensitive_equal(
		_base_parity_projection(config), _base_parity_projection(reference_config)
	):
		raise ValueError(
			'candidate base differs from canonical outside augmentation, duration, '
			'and output root'
		)


def audit_candidate_checkpoints(
	*,
	candidate: CandidateSource,
	canonical_config: F3FiveWayConfig,
	reference_base_checkpoint: Path,
	reference_final_checkpoint: Path,
) -> dict[str, object]:
	"""Audit the base plus its fresh fixed 25-epoch continuation."""
	base = audit_candidate_base_checkpoint(
		candidate=candidate,
		canonical_config=canonical_config,
		reference_base_checkpoint=reference_base_checkpoint,
	)
	_shared('_validate_reference_final_checkpoint')(
		canonical_config=canonical_config,
		reference_final_checkpoint=reference_final_checkpoint,
		verify_file=True,
	)
	_validate_regular_file(
		candidate.final_checkpoint, label='candidate final checkpoint'
	)
	payload = load_checkpoint_metadata_without_weights(candidate.final_checkpoint)
	reference = load_checkpoint_metadata_without_weights(reference_final_checkpoint)
	_validate_candidate_final_checkpoint(
		candidate, payload=payload, reference=reference
	)
	return {
		'audit_schema_version': CHECKPOINT_AUDIT_SCHEMA_VERSION,
		'audit_type': 'trace_drop_base1_pre_extraction_lineage_checkpoint_only',
		'candidate_id': CANDIDATE_ID,
		'base_checkpoint': str(candidate.base_checkpoint),
		'base_checkpoint_sha256': base['base_checkpoint_sha256'],
		'base_pretraining_epochs': BASE_PRETRAINING_EPOCHS,
		'base_global_step': EXPECTED_BASE_STEPS,
		'base_resume_count': 0,
		'final_checkpoint': str(candidate.final_checkpoint),
		'final_checkpoint_sha256': file_sha256(candidate.final_checkpoint),
		'continuation_init_checkpoint_sha256': base['base_checkpoint_sha256'],
		'continuation_epochs': CONTINUATION_EPOCHS,
		'continuation_global_step': EXPECTED_CONTINUATION_STEPS,
		'continuation_resume_count': 0,
		'continuation_lineage_resume_count': 0,
		'augmentations': dict(EXPECTED_AUGMENTATIONS),
		'reference_base_checkpoint': str(reference_base_checkpoint),
		'reference_base_checkpoint_sha256': file_sha256(reference_base_checkpoint),
		'reference_final_checkpoint': str(reference_final_checkpoint),
		'reference_final_checkpoint_sha256': file_sha256(reference_final_checkpoint),
		'base_parity_exceptions': [
			'augmentations',
			'train.epochs',
			'paths.output_root',
		],
		'final_parity_exceptions': [
			'augmentations',
			'paths.output_root',
			'continuation.init_checkpoint',
		],
		'embeddings_required': False,
		'passed': True,
	}


def _validate_candidate_final_checkpoint(
	candidate: CandidateSource,
	*,
	payload: Mapping[str, object],
	reference: Mapping[str, object],
) -> None:
	_common_checkpoint_identity(
		payload,
		epochs=CONTINUATION_EPOCHS,
		global_step=EXPECTED_CONTINUATION_STEPS,
		training_state=EXPECTED_CONTINUATION_TRAINING_STATE,
		label='candidate continuation',
	)
	config = _mapping(payload, 'config')
	reference_config = _mapping(reference, 'config')
	if _mapping(config, 'paths').get('output_root') != str(
		candidate.final_checkpoint.parent
	):
		raise ValueError('candidate continuation output_root must own its checkpoint')
	if not _type_sensitive_equal(
		_mapping(config, 'augmentations'), EXPECTED_AUGMENTATIONS
	):
		raise ValueError('candidate continuation records unexpected augmentations')
	_validate_barlow_objective(config, label='candidate continuation')
	continuation = _mapping(config, 'continuation')
	_require_exact_keys(
		continuation,
		{'init_checkpoint', 'unfreeze_top_blocks'},
		'candidate continuation config',
	)
	if continuation.get('init_checkpoint') != str(candidate.base_checkpoint):
		raise ValueError('continuation must initialize from its exact fresh base')
	if not _type_sensitive_equal(
		continuation.get('unfreeze_top_blocks'), CONTINUATION_UNFREEZE_TOP_BLOCKS
	):
		raise ValueError('continuation must unfreeze exactly one top block')
	lineage = _mapping(payload, 'continuation_lineage')
	_require_exact_keys(
		lineage,
		{'schema_version', 'init_checkpoint', 'init_checkpoint_sha256', 'resume_count'},
		'candidate continuation lineage',
	)
	if not _type_sensitive_equal(lineage.get('schema_version'), 1):
		raise ValueError('continuation lineage schema_version must equal 1')
	if lineage.get('init_checkpoint') != str(candidate.base_checkpoint):
		raise ValueError('continuation lineage base path changed')
	if lineage.get('init_checkpoint_sha256') != file_sha256(candidate.base_checkpoint):
		raise ValueError('continuation lineage base SHA-256 changed')
	if not _type_sensitive_equal(lineage.get('resume_count'), 0):
		raise ValueError('continuation lineage resume_count must equal 0')
	_validate_training_steps(
		config,
		epochs=CONTINUATION_EPOCHS,
		expected_steps=EXPECTED_CONTINUATION_STEPS,
		label='candidate continuation',
	)
	if not _type_sensitive_equal(
		_final_parity_projection(config), _final_parity_projection(reference_config)
	):
		raise ValueError(
			'candidate continuation differs from canonical outside augmentation, '
			'output root, and init checkpoint'
		)


def _common_checkpoint_identity(
	payload: Mapping[str, object],
	*,
	epochs: int,
	global_step: int,
	training_state: Mapping[str, object],
	label: str,
) -> None:
	if payload.get('checkpoint_kind') != 'barlow_twins_pretraining':
		raise ValueError(f'{label} kind must be Barlow Twins')
	if payload.get('pretraining_method') != LOCAL_BARLOW_METHOD:
		raise ValueError(f'{label} method must be Local Barlow')
	if not _type_sensitive_equal(payload.get('epoch'), epochs):
		raise ValueError(f'{label} epoch must equal {epochs}')
	if not _type_sensitive_equal(payload.get('global_step'), global_step):
		raise ValueError(f'{label} global_step must equal {global_step}')
	if not _type_sensitive_equal(_mapping(payload, 'training_state'), training_state):
		raise ValueError(f'{label} completed training_state changed')
	if not _type_sensitive_equal(payload.get('resume_count'), 0):
		raise ValueError(f'{label} resume_count must equal 0')
	if payload.get('amp_enabled') is not False:
		raise ValueError(f'{label} amp_enabled must be false')
	if payload.get('scaler_state_dict') is not None:
		raise ValueError(f'{label} scaler_state_dict must be null')
	if not _type_sensitive_equal(
		payload.get('trained_parameter_prefixes'), EXPECTED_TRAINED_PARAMETER_PREFIXES
	):
		raise ValueError(f'{label} trained parameter prefixes changed')


def _validate_barlow_objective(config: Mapping[str, object], *, label: str) -> None:
	barlow = _mapping(config, 'barlow_twins')
	if barlow.get('method') != LOCAL_BARLOW_METHOD or not _type_sensitive_equal(
		barlow.get('local_pairs_per_crop'), LOCAL_PAIRS_PER_CROP
	):
		raise ValueError(f'{label} changes the Local Barlow objective')


def _validate_training_steps(
	config: Mapping[str, object], *, epochs: int, expected_steps: int, label: str
) -> None:
	train = _mapping(config, 'train')
	if not _type_sensitive_equal(train.get('epochs'), epochs):
		raise ValueError(f'{label} train.epochs must equal {epochs}')
	samples = _positive_int(train.get('samples_per_epoch'), f'{label} samples')
	batch = _positive_int(train.get('batch_size'), f'{label} batch size')
	if samples % batch or epochs * samples // batch != expected_steps:
		raise ValueError(f'{label} no longer yields {expected_steps} steps')


def _base_parity_projection(config: Mapping[str, object]) -> dict[str, object]:
	projected = deepcopy(dict(config))
	projected.pop('augmentations', None)
	cast('dict[str, object]', projected['paths']).pop('output_root', None)
	cast('dict[str, object]', projected['train']).pop('epochs', None)
	return projected


def _final_parity_projection(config: Mapping[str, object]) -> dict[str, object]:
	projected = deepcopy(dict(config))
	projected.pop('augmentations', None)
	cast('dict[str, object]', projected['paths']).pop('output_root', None)
	cast('dict[str, object]', projected['continuation']).pop('init_checkpoint', None)
	return projected


def create_trace_drop_protocol_lock(
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	*,
	created_at_utc: str | None = None,
	git_head: str | None = None,
) -> dict[str, object]:
	"""Seal parent, fresh base, and all random cells before continuation."""
	if settings.protocol_lock.exists() or settings.protocol_lock.is_symlink():
		raise FileExistsError(f'protocol lock already exists: {settings.protocol_lock}')
	_reject_pre_protocol_evidence(settings, canonical)
	parent = validate_parent_result(settings, canonical)
	benchmark = cast(
		'Mapping[str, object]',
		_shared('_validate_benchmark_provenance')(
			settings, canonical, verify_files=True
		),
	)
	repository = _experiment_repository_state()
	resolved_git_head = _string(repository.get('git_head'), 'repository git_head')
	if git_head is not None and git_head != resolved_git_head:
		raise ValueError('protocol Git HEAD must equal the live repository')
	base = audit_candidate_base_checkpoint(
		candidate=settings.candidate,
		canonical_config=canonical,
		reference_base_checkpoint=settings.reference_base_checkpoint,
	)
	frozen_random = _collect_all_random_inputs(canonical)
	payload = _protocol_lock_payload(
		parent_result=parent,
		base_checkpoint_input=base,
		frozen_random_inputs=frozen_random,
		benchmark_provenance=benchmark,
		repository_state=repository,
		created_at_utc=created_at_utc or _utc_timestamp(),
		git_head=resolved_git_head,
	)
	_reject_pre_protocol_evidence(settings, canonical)
	_write_exclusive_json(settings.protocol_lock, payload)
	return payload


def validate_trace_drop_protocol_lock(
	settings: ValidationSettings, canonical: F3FiveWayConfig
) -> Mapping[str, object]:
	"""Replay every immutable protocol input against live evidence."""
	stored = _read_regular_json(
		settings.protocol_lock, label='trace-drop protocol lock'
	)
	created = _string(stored.get('created_at_utc'), 'protocol created_at_utc')
	_validate_utc_timestamp(created)
	git_head = _sha1_value(stored.get('git_head'), 'protocol git_head')
	expected = _protocol_lock_payload(
		parent_result=validate_parent_result(settings, canonical),
		base_checkpoint_input=audit_candidate_base_checkpoint(
			candidate=settings.candidate,
			canonical_config=canonical,
			reference_base_checkpoint=settings.reference_base_checkpoint,
		),
		frozen_random_inputs=_collect_all_random_inputs(canonical),
		benchmark_provenance=cast(
			'Mapping[str, object]',
			_shared('_validate_benchmark_provenance')(
				settings, canonical, verify_files=True
			),
		),
		repository_state=_experiment_repository_state(),
		created_at_utc=created,
		git_head=git_head,
	)
	if not _type_sensitive_equal(stored, expected):
		raise ValueError('trace-drop protocol lock differs from live frozen inputs')
	return stored


def _protocol_lock_payload(  # noqa: PLR0913
	*,
	parent_result: Mapping[str, object],
	base_checkpoint_input: Mapping[str, object],
	frozen_random_inputs: Sequence[Mapping[str, object]],
	benchmark_provenance: Mapping[str, object],
	repository_state: Mapping[str, object],
	created_at_utc: str,
	git_head: str,
) -> dict[str, object]:
	_validate_utc_timestamp(created_at_utc)
	_sha1_value(git_head, 'protocol git_head')
	_validate_all_random_cells(frozen_random_inputs)
	if base_checkpoint_input.get('candidate_id') != CANDIDATE_ID:
		raise ValueError('protocol base input must be the preregistered candidate')
	return {
		'schema_version': PROTOCOL_LOCK_SCHEMA_VERSION,
		'protocol_lock_type': PROTOCOL_LOCK_TYPE,
		'validation_only': True,
		'stage_boundary': 'completed_fresh_base1_before_continuation',
		'base_pretraining_epochs': BASE_PRETRAINING_EPOCHS,
		'continuation_epochs': CONTINUATION_EPOCHS,
		'candidate_id': CANDIDATE_ID,
		'preregistered_augmentations': dict(EXPECTED_AUGMENTATIONS),
		'selection_basis': 'fixed_unlabeled_16_crop_strength_match_v1',
		'candidate_validation_metric_inputs': [],
		'medium_gate_contract': {
			'data_size': 'medium',
			'layout_ids': list(LAYOUT_IDS),
			'comparison_source_id': RANDOM_ID,
			'criterion': 'strict_positive_delta_over_random_all_5_medium_layouts',
			'required_positive_delta_count': 5,
		},
		'success_contract': {
			'data_sizes': list(DATA_SIZES),
			'layout_ids': list(LAYOUT_IDS),
			'comparison_source_id': RANDOM_ID,
			'criterion': 'strict_positive_delta_over_random_all_15_validation_cells',
			'required_positive_delta_count': 15,
		},
		'parent_result': dict(parent_result),
		'base_checkpoint_input': dict(base_checkpoint_input),
		'frozen_random_inputs': [dict(row) for row in frozen_random_inputs],
		'benchmark_provenance': dict(benchmark_provenance),
		'repository_state': dict(repository_state),
		'created_at_utc': created_at_utc,
		'git_head': git_head,
	}


def _collect_all_random_inputs(canonical: F3FiveWayConfig) -> list[dict[str, object]]:
	rows = [
		cast(
			'dict[str, object]',
			_shared('_read_random_job_evidence')(
				canonical, layout_id=layout_id, data_size=data_size
			),
		)
		for data_size in DATA_SIZES
		for layout_id in LAYOUT_IDS
	]
	_validate_all_random_cells(rows)
	return rows


def _validate_all_random_cells(rows: Sequence[Mapping[str, object]]) -> None:
	expected = {
		(RANDOM_ID, layout_id, data_size)
		for data_size in DATA_SIZES
		for layout_id in LAYOUT_IDS
	}
	actual = {_cell_key(row) for row in rows}
	if len(rows) != 15 or len(actual) != 15 or actual != expected:
		raise ValueError('protocol must freeze exactly all 15 canonical random cells')


def _reject_pre_protocol_evidence(
	settings: ValidationSettings, canonical: F3FiveWayConfig
) -> None:
	for label, path in (
		('final result', settings.final_result),
		('validation runs', settings.runs_root),
		('continuation output', settings.candidate.final_checkpoint.parent),
		('embedding output', settings.candidate.embeddings_dir),
	):
		if _path_contains_evidence(path):
			raise ValueError(f'cannot lock protocol after {label} exists: {path}')
	config = _candidate_config(
		canonical, candidate=settings.candidate, runs_root=settings.runs_root
	)
	for layout_id in LAYOUT_IDS:
		for data_size in DATA_SIZES:
			job = five_way_runner.resolve_f3_lithology_five_way_job(
				config,
				model=CANDIDATE_ID,
				layout=layout_id,
				size=data_size,
			)
			if _path_contains_evidence(job.metrics_path):
				raise ValueError('candidate metric exists before protocol lock')


def _path_contains_evidence(path: Path) -> bool:
	if path.is_symlink() or path.is_file():
		return True
	if path.is_dir():
		return next(path.iterdir(), None) is not None
	return path.exists()


def _experiment_repository_state() -> dict[str, object]:
	"""Extend the frozen helper's core inventory with this experiment tree."""
	state = dict(cast('Mapping[str, object]', _shared('_git_repository_state')()))
	inventory = [
		{
			'path': path.relative_to(REPOSITORY_ROOT).as_posix(),
			'sha256': file_sha256(path),
		}
		for path in sorted(EXPERIMENT_ROOT.rglob('*'))
		if (
			path.is_file()
			and not path.is_symlink()
			and '__pycache__' not in path.parts
			and path.suffix != '.pyc'
		)
	]
	state['trace_drop_experiment_file_inventory'] = inventory
	return state


def _protocol_identity(
	settings: ValidationSettings, protocol: Mapping[str, object]
) -> dict[str, str]:
	if protocol.get('protocol_lock_type') != PROTOCOL_LOCK_TYPE:
		raise ValueError('trace-drop protocol lock type changed')
	return {
		'path': str(settings.protocol_lock),
		'sha256': file_sha256(settings.protocol_lock),
	}


def _candidate_config(
	canonical: F3FiveWayConfig,
	*,
	candidate: CandidateSource,
	runs_root: Path,
) -> F3FiveWayConfig:
	return cast(
		'F3FiveWayConfig',
		_shared('_candidate_config')(
			canonical, candidate=candidate, runs_root=runs_root
		),
	)


def audit_candidate_source(  # noqa: PLR0913
	*,
	candidate: CandidateSource,
	candidate_model: F3FiveWayModelSource,
	canonical_config: F3FiveWayConfig,
	reference_base_checkpoint: Path,
	reference_final_checkpoint: Path,
	protocol_lock_identity: Mapping[str, object],
) -> dict[str, object]:
	"""Audit checkpoint and extraction lineage without the five-way source audit."""
	checkpoint = audit_candidate_checkpoints(
		candidate=candidate,
		canonical_config=canonical_config,
		reference_base_checkpoint=reference_base_checkpoint,
		reference_final_checkpoint=reference_final_checkpoint,
	)
	survey_id = canonical_config.dataset['name']
	files = output_paths(candidate.embeddings_dir, survey_id)
	random_files = output_paths(
		canonical_config.model_by_id(RANDOM_ID).embeddings_dir, survey_id
	)
	for label, path in (
		('candidate embeddings', files.embeddings),
		('candidate valid-token mask', files.valid_tokens),
		('candidate embedding metadata', files.metadata),
		('random valid-token mask', random_files.valid_tokens),
		('random embedding metadata', random_files.metadata),
	):
		_validate_regular_file(path, label=label)
	metadata = _read_regular_json(files.metadata, label='candidate metadata')
	random_metadata = _read_regular_json(random_files.metadata, label='random metadata')
	five_way_sources._validate_extraction_contract(CANDIDATE_ID, metadata)
	five_way_sources._validate_shared_identity(CANDIDATE_ID, metadata, random_metadata)
	token_grid_shape = five_way_sources._token_grid_shape(CANDIDATE_ID, metadata)
	five_way_sources._validate_arrays(CANDIDATE_ID, files, token_grid_shape)
	if not five_way_sources._masks_identical(
		files.valid_tokens, random_files.valid_tokens
	):
		raise ValueError('candidate valid-token mask differs from random')
	final_sha = five_way_sources._validate_checkpoint_identity(
		candidate_model, metadata
	)
	five_way_sources._validate_objective_identity(candidate_model, metadata)
	objective = _mapping(metadata, 'pretraining_objective')
	if not _type_sensitive_equal(
		objective.get('augmentations'), EXPECTED_AUGMENTATIONS
	):
		raise ValueError('embedding objective augmentation mapping changed')
	if final_sha != checkpoint['final_checkpoint_sha256']:
		raise ValueError('embedding metadata and final checkpoint SHA-256 disagree')
	return {
		'audit_schema_version': CHECKPOINT_AUDIT_SCHEMA_VERSION,
		'audit_type': 'trace_drop_base1_validation_source_lineage',
		'candidate_id': CANDIDATE_ID,
		'source_role': candidate.role,
		'validation_only': True,
		'selection_eligible': True,
		'protocol_lock': dict(protocol_lock_identity),
		'base_checkpoint': str(candidate.base_checkpoint),
		'base_checkpoint_sha256': checkpoint['base_checkpoint_sha256'],
		'base_pretraining_epochs': BASE_PRETRAINING_EPOCHS,
		'base_global_step': EXPECTED_BASE_STEPS,
		'base_resume_count': 0,
		'final_checkpoint': str(candidate.final_checkpoint),
		'final_checkpoint_sha256': final_sha,
		'continuation_init_checkpoint_sha256': checkpoint[
			'continuation_init_checkpoint_sha256'
		],
		'continuation_epochs': CONTINUATION_EPOCHS,
		'continuation_global_step': EXPECTED_CONTINUATION_STEPS,
		'continuation_resume_count': 0,
		'continuation_lineage_resume_count': 0,
		'embeddings_dir': str(candidate.embeddings_dir),
		'embeddings_path': str(files.embeddings),
		'embeddings_sha256': file_sha256(files.embeddings),
		'embedding_metadata': str(files.metadata),
		'embedding_metadata_sha256': file_sha256(files.metadata),
		'valid_tokens_path': str(files.valid_tokens),
		'valid_tokens_sha256': file_sha256(files.valid_tokens),
		'augmentations': dict(EXPECTED_AUGMENTATIONS),
		'reference_base_checkpoint': str(reference_base_checkpoint),
		'reference_base_checkpoint_sha256': checkpoint[
			'reference_base_checkpoint_sha256'
		],
		'reference_final_checkpoint': str(reference_final_checkpoint),
		'reference_final_checkpoint_sha256': checkpoint[
			'reference_final_checkpoint_sha256'
		],
		'base_parity_exceptions': checkpoint['base_parity_exceptions'],
		'final_parity_exceptions': checkpoint['final_parity_exceptions'],
		'fixed_downstream_summary_name': canonical_config.summary_name,
		'fixed_section_layout_dataset_root': str(
			canonical_config.section_layout_dataset_root
		),
		'token_grid_shape': list(token_grid_shape),
		'valid_token_mask': 'byte_identical_to_canonical_random',
		'evaluation_split': 'validation',
		'evaluation_aggregation_unit': VALIDATION_AGGREGATION_UNIT,
	}


def enforce_validation_order(
	*, settings: ValidationSettings, canonical: F3FiveWayConfig, data_size: str
) -> dict[str, object]:
	"""Require the protocol for medium and the strict 5/5 gate thereafter."""
	protocol = validate_trace_drop_protocol_lock(settings, canonical)
	if data_size == 'medium':
		return {'protocol_lock': protocol, 'medium_gate': None}
	gate = _validate_medium_random_gate(
		settings=settings, canonical=canonical, protocol=protocol, require_open=True
	)
	return {'protocol_lock': protocol, 'medium_gate': gate}


def _validate_medium_random_gate(
	*,
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	protocol: Mapping[str, object],
	require_open: bool,
) -> dict[str, object]:
	protocol_identity = _protocol_identity(settings, protocol)
	config = _candidate_config(
		canonical, candidate=settings.candidate, runs_root=settings.runs_root
	)
	source_audit = audit_candidate_source(
		candidate=settings.candidate,
		candidate_model=config.model_by_id(CANDIDATE_ID),
		canonical_config=canonical,
		reference_base_checkpoint=settings.reference_base_checkpoint,
		reference_final_checkpoint=settings.reference_final_checkpoint,
		protocol_lock_identity=protocol_identity,
	)
	medium_order = {
		'protocol_lock': protocol_identity,
		'medium_gate': None,
	}
	candidate_inputs: list[dict[str, object]] = []
	for layout_id in LAYOUT_IDS:
		job = five_way_runner.resolve_f3_lithology_five_way_job(
			config, model=CANDIDATE_ID, layout=layout_id, size='medium'
		)
		candidate_inputs.append(
			_read_candidate_job_evidence(
				job=job,
				candidate=settings.candidate,
				expected_source_audit=source_audit,
				expected_validation_order=medium_order,
				verify_evaluation_identity=True,
			)
		)
	random_inputs = [
		dict(row)
		for row in _mapping_rows(
			protocol.get('frozen_random_inputs'), 'protocol frozen_random_inputs'
		)
		if row.get('data_size') == 'medium'
	]
	if len(random_inputs) != 5:
		raise ValueError('protocol must supply five frozen medium random cells')
	result = _medium_gate_result(candidate_inputs, random_inputs)
	gate_open = cast('bool', result['gate_open'])
	if require_open and not gate_open:
		raise ValueError(
			'small/large validation is forbidden because trace drop did not beat '
			'random on all five medium layouts'
		)
	return result


def _medium_gate_result(
	candidate_inputs: Sequence[Mapping[str, object]],
	random_inputs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
	"""Apply the strict 5/5 gate to exact paired medium evidence."""
	scores = _scores_from_evidence((*candidate_inputs, *random_inputs))
	expected_cells = {(layout_id, 'medium') for layout_id in LAYOUT_IDS}
	if set(scores.get(CANDIDATE_ID, {})) != expected_cells:
		raise ValueError('medium gate candidate inputs are not exactly five layouts')
	if set(scores.get(RANDOM_ID, {})) != expected_cells:
		raise ValueError('medium gate random inputs are not exactly five layouts')
	deltas = {
		layout_id: (
			scores[CANDIDATE_ID][(layout_id, 'medium')]
			- scores[RANDOM_ID][(layout_id, 'medium')]
		)
		for layout_id in LAYOUT_IDS
	}
	positive_count = sum(delta > 0.0 for delta in deltas.values())
	gate_open = positive_count == 5
	return {
		'gate_open': gate_open,
		'criterion': 'strict_positive_delta_over_random_all_5_medium_layouts',
		'candidate_id': CANDIDATE_ID,
		'layout_ids': list(LAYOUT_IDS),
		'required_positive_delta_count': 5,
		'positive_delta_count': positive_count,
		'wins_all_5_over_random': gate_open,
		'paired_macro_f1_deltas_over_random': deltas,
		'inputs': [*candidate_inputs, *random_inputs],
	}


def _validation_order_provenance(
	settings: ValidationSettings,
	protocol: Mapping[str, object],
	gate: Mapping[str, object] | None,
) -> dict[str, object]:
	result: dict[str, object] = {
		'protocol_lock': _protocol_identity(settings, protocol),
		'medium_gate': None,
	}
	if gate is None:
		return result
	inputs = _mapping_rows(gate.get('inputs'), 'medium gate inputs')
	if len(inputs) != 10:
		raise ValueError('medium gate must bind five candidate and five random cells')
	result['medium_gate'] = {
		'gate_open': gate.get('gate_open'),
		'positive_delta_count': gate.get('positive_delta_count'),
		'wins_all_5_over_random': gate.get('wins_all_5_over_random'),
		'inputs': [
			{
				'candidate_id': row.get('candidate_id'),
				'layout_id': row.get('layout_id'),
				'data_size': row.get('data_size'),
				'metrics_path': row.get('metrics_path'),
				'metrics_sha256': row.get('metrics_sha256'),
			}
			for row in inputs
		],
	}
	return result


def _read_candidate_job_evidence(
	*,
	job: five_way_runner.F3FiveWayJob,
	candidate: CandidateSource,
	expected_source_audit: Mapping[str, object],
	expected_validation_order: Mapping[str, object],
	verify_evaluation_identity: bool,
) -> dict[str, object]:
	metrics, metrics_sha = _read_hashed_json(job.metrics_path, label='metrics')
	audit_path = job.output_dir / AUDIT_NAME
	audit, audit_sha = _read_hashed_json(audit_path, label='candidate source audit')
	source_payload = dict(audit)
	for key in (
		'layout_id',
		'data_size',
		'metrics_path',
		'validation_order_provenance',
	):
		source_payload.pop(key, None)
	if not _type_sensitive_equal(source_payload, expected_source_audit):
		raise ValueError(f'{job.output_dir} source audit differs from live lineage')
	if audit.get('layout_id') != job.layout_id:
		raise ValueError('candidate audit layout identity changed')
	if audit.get('data_size') != job.data_size:
		raise ValueError('candidate audit data-size identity changed')
	if audit.get('metrics_path') != str(job.metrics_path):
		raise ValueError('candidate audit metrics path changed')
	if not _type_sensitive_equal(
		audit.get('validation_order_provenance'), expected_validation_order
	):
		raise ValueError('candidate audit validation-order provenance changed')
	if verify_evaluation_identity:
		_shared('_validate_job_evaluation_identity')(
			job=job, metrics_sha256=metrics_sha
		)
		evaluation_metadata = _read_regular_json(
			job.evaluation_dir / 'evaluation_metadata.json',
			label='evaluation metadata',
		)
		prediction_identity = five_way_results._job_source_identity(
			label=f'{candidate.candidate_id}/{job.layout_id}/{job.data_size}',
			model=job.model,
			survey_id=job.config.dataset['name'],
			job_dir=job.output_dir,
			evaluation_metadata=evaluation_metadata,
		)
		for key in (
			'embeddings_sha256',
			'embedding_metadata_sha256',
			'valid_tokens_sha256',
		):
			if prediction_identity.get(key) != audit.get(key):
				raise ValueError(
					f'{job.output_dir} prediction {key} differs from audit'
				)
	return {
		'candidate_id': CANDIDATE_ID,
		'layout_id': job.layout_id,
		'data_size': job.data_size,
		'macro_f1': _macro_f1(metrics, job.metrics_path),
		'base_checkpoint_sha256': audit['base_checkpoint_sha256'],
		'continuation_init_checkpoint_sha256': audit[
			'continuation_init_checkpoint_sha256'
		],
		'final_checkpoint_sha256': audit['final_checkpoint_sha256'],
		'embeddings_sha256': audit['embeddings_sha256'],
		'embedding_metadata_sha256': audit['embedding_metadata_sha256'],
		'valid_tokens_sha256': audit['valid_tokens_sha256'],
		'metrics_path': str(job.metrics_path),
		'metrics_sha256': metrics_sha,
		'candidate_audit_path': str(audit_path),
		'candidate_audit_sha256': audit_sha,
	}


def create_trace_drop_final_result(
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	*,
	created_at_utc: str | None = None,
) -> dict[str, object]:
	"""Replay the exact reached branch and exclusively publish its decision."""
	if settings.final_result.exists() or settings.final_result.is_symlink():
		raise FileExistsError(f'final result already exists: {settings.final_result}')
	protocol = validate_trace_drop_protocol_lock(settings, canonical)
	protocol_identity = _protocol_identity(settings, protocol)
	gate = _validate_medium_random_gate(
		settings=settings, canonical=canonical, protocol=protocol, require_open=False
	)
	gate_open = cast('bool', gate['gate_open'])
	expected_cells = _expected_candidate_cells(medium_gate_open=gate_open)
	_validate_exact_candidate_cell_set(
		settings=settings, canonical=canonical, expected_cells=expected_cells
	)
	config = _candidate_config(
		canonical, candidate=settings.candidate, runs_root=settings.runs_root
	)
	source_audit = audit_candidate_source(
		candidate=settings.candidate,
		candidate_model=config.model_by_id(CANDIDATE_ID),
		canonical_config=canonical,
		reference_base_checkpoint=settings.reference_base_checkpoint,
		reference_final_checkpoint=settings.reference_final_checkpoint,
		protocol_lock_identity=protocol_identity,
	)
	medium_order = _validation_order_provenance(settings, protocol, None)
	post_gate_order = _validation_order_provenance(settings, protocol, gate)
	candidate_inputs: list[dict[str, object]] = []
	for _, layout_id, data_size in sorted(expected_cells):
		job = five_way_runner.resolve_f3_lithology_five_way_job(
			config, model=CANDIDATE_ID, layout=layout_id, size=data_size
		)
		candidate_inputs.append(
			_read_candidate_job_evidence(
				job=job,
				candidate=settings.candidate,
				expected_source_audit=source_audit,
				expected_validation_order=(
					medium_order if data_size == 'medium' else post_gate_order
				),
				verify_evaluation_identity=True,
			)
		)
	frozen_random = _mapping_rows(
		protocol.get('frozen_random_inputs'), 'protocol frozen_random_inputs'
	)
	reached_sizes = set(DATA_SIZES if gate_open else ('medium',))
	random_inputs = [
		dict(row) for row in frozen_random if row.get('data_size') in reached_sizes
	]
	_assert_unique_evidence_rows(candidate_inputs, include_candidate=True)
	_assert_unique_evidence_rows(random_inputs, include_candidate=False)
	scores = _scores_from_evidence((*candidate_inputs, *random_inputs))
	arm_result = cast(
		'dict[str, object]',
		_shared('_arm_random_result')(
			scores, arm_id=CANDIDATE_ID, medium_gate_open=gate_open
		),
	)
	parent = _mapping(protocol, 'parent_result')
	frozen_controls = _mapping_rows(
		parent.get('frozen_medium_control_inputs'),
		'protocol parent frozen controls',
	)
	attribution_scores = _scores_from_evidence((*candidate_inputs, *frozen_controls))
	attribution = {
		'trace_minus_gaussian': _medium_control_contrast(
			attribution_scores,
			left_id=CANDIDATE_ID,
			right_id=GAUSSIAN_CONTROL_ID,
			contrast_id='trace_drop_p001_minus_gaussian_std010_base1',
		),
		'trace_minus_legacy': _medium_control_contrast(
			attribution_scores,
			left_id=CANDIDATE_ID,
			right_id=LEGACY_CONTROL_ID,
			contrast_id='trace_drop_p001_minus_legacy_flip_base1',
		),
	}
	decision = _terminal_decision(arm_result=arm_result, gate_open=gate_open)
	payload = {
		'schema_version': FINAL_RESULT_SCHEMA_VERSION,
		'final_result_type': FINAL_RESULT_TYPE,
		'validation_only': True,
		'base_pretraining_epochs': BASE_PRETRAINING_EPOCHS,
		'continuation_epochs': CONTINUATION_EPOCHS,
		'candidate_id': CANDIDATE_ID,
		'augmentations': dict(EXPECTED_AUGMENTATIONS),
		'parent_result': dict(parent),
		'protocol_lock': protocol_identity,
		'benchmark_provenance': dict(_mapping(protocol, 'benchmark_provenance')),
		'repository_state': dict(_mapping(protocol, 'repository_state')),
		'medium_gate': dict(gate),
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
		'frozen_medium_control_inputs': [dict(row) for row in frozen_controls],
		'frozen_control_provenance': {
			'parent_path': parent['path'],
			'parent_sha256': parent['sha256'],
			'control_candidate_ids': [GAUSSIAN_CONTROL_ID, LEGACY_CONTROL_ID],
			'data_size': 'medium',
			'evaluated_cell_count': 10,
			'fresh_parent_random_type_sensitive_match': parent[
				'fresh_random_type_sensitive_match'
			],
		},
		'arm_results': {CANDIDATE_ID: arm_result},
		'trace_drop_attribution': attribution,
		**decision,
		'created_at_utc': created_at_utc or _utc_timestamp(),
	}
	_validate_utc_timestamp(cast('str', payload['created_at_utc']))
	_write_exclusive_json(settings.final_result, payload)
	return payload


def _terminal_decision(
	*, arm_result: Mapping[str, object], gate_open: bool
) -> dict[str, object]:
	"""Resolve success and the separately documented p=.02 authorization."""
	passed = arm_result.get('wins_all_15_over_random')
	if not isinstance(passed, bool):
		raise TypeError('arm result wins_all_15_over_random must be a bool')
	return {
		'passed': passed,
		'winner_candidate_id': CANDIDATE_ID if passed else None,
		'failure_stage': (
			None if passed else ('final_15of15' if gate_open else 'medium_5of5')
		),
		'authorizes_trace_drop_p002_followup': not passed,
		'authorized_trace_drop_probability': (
			None if passed else FALLBACK_TRACE_DROP_PROBABILITY
		),
	}


def _expected_candidate_cells(
	*, medium_gate_open: bool
) -> set[tuple[str, str, str]]:
	sizes = DATA_SIZES if medium_gate_open else ('medium',)
	return {
		(CANDIDATE_ID, layout_id, data_size)
		for layout_id in LAYOUT_IDS
		for data_size in sizes
	}


def _validate_exact_candidate_cell_set(
	*,
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	expected_cells: set[tuple[str, str, str]],
) -> None:
	config = _candidate_config(
		canonical, candidate=settings.candidate, runs_root=settings.runs_root
	)
	for layout_id in LAYOUT_IDS:
		for data_size in DATA_SIZES:
			job = five_way_runner.resolve_f3_lithology_five_way_job(
				config, model=CANDIDATE_ID, layout=layout_id, size=data_size
			)
			expected = (CANDIDATE_ID, layout_id, data_size) in expected_cells
			for path in (job.metrics_path, job.output_dir / AUDIT_NAME):
				if path.is_symlink():
					raise ValueError(f'validation evidence is a symlink: {path}')
				if expected and not path.is_file():
					raise FileNotFoundError(
						f'missing expected trace-drop validation evidence: {path}'
					)
				if not expected and path.exists():
					raise ValueError(
						f'validation evidence exists outside reached set: {path}'
					)


def _medium_control_contrast(
	scores: Mapping[str, Mapping[tuple[str, str], float]],
	*,
	left_id: str,
	right_id: str,
	contrast_id: str,
) -> dict[str, object]:
	expected_cells = {(layout_id, 'medium') for layout_id in LAYOUT_IDS}
	if set(scores.get(left_id, {})) & expected_cells != expected_cells:
		raise ValueError(f'{contrast_id} left arm lacks five medium cells')
	if set(scores.get(right_id, {})) != expected_cells:
		raise ValueError(
			f'{contrast_id} right control is not exactly five medium cells'
		)
	deltas = {
		f'{layout_id}/medium': (
			scores[left_id][(layout_id, 'medium')]
			- scores[right_id][(layout_id, 'medium')]
		)
		for layout_id in LAYOUT_IDS
	}
	positive_count = sum(delta > 0.0 for delta in deltas.values())
	return {
		'contrast_id': contrast_id,
		'left_candidate_id': left_id,
		'right_candidate_id': right_id,
		'data_size': 'medium',
		'evaluated_cell_count': 5,
		'paired_macro_f1_deltas': deltas,
		'positive_delta_count': positive_count,
		'wins_all_5': positive_count == 5,
	}


def _scores_from_evidence(
	rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[tuple[str, str], float]]:
	result: dict[str, dict[tuple[str, str], float]] = {}
	for row in rows:
		source_id, layout_id, data_size = _cell_key(row)
		value = row.get('macro_f1')
		if not isinstance(value, int | float) or isinstance(value, bool):
			raise TypeError('evidence macro_f1 must be numeric')
		cell = (layout_id, data_size)
		if cell in result.setdefault(source_id, {}):
			raise ValueError('validation evidence duplicates a source cell')
		result[source_id][cell] = float(value)
	return result


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
		raise ValueError('validation evidence contains duplicate cells')


def _run_job(
	job: five_way_runner.F3FiveWayJob,
	*,
	audit: Mapping[str, object],
	device: str,
	resume: Path | None,
) -> dict[str, object]:
	return cast(
		'dict[str, object]',
		_shared('_run_job')(job, audit=audit, device=device, resume=resume),
	)


def _read_regular_json(path: Path, *, label: str) -> Mapping[str, object]:
	value, _ = _read_hashed_json(path, label=label)
	return value


def _read_hashed_json(path: Path, *, label: str) -> tuple[Mapping[str, object], str]:
	_validate_regular_file(path, label=label)
	raw = path.read_bytes()
	try:
		value = json.loads(raw)
	except json.JSONDecodeError as error:
		raise ValueError(f'{label} is not valid JSON: {path}') from error
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must contain a JSON object: {path}')
	return value, hashlib.sha256(raw).hexdigest()


def _validate_regular_file(path: Path, *, label: str) -> None:
	if path.is_symlink():
		raise ValueError(f'{label} must not be a symlink: {path}')
	if not path.is_file():
		raise FileNotFoundError(f'missing {label}: {path}')


def _write_exclusive_json(path: Path, payload: Mapping[str, object]) -> None:
	"""Atomically publish immutable JSON without replacing an existing path."""
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
		try:
			os.link(cast('Path', temporary), path)
		except FileExistsError as error:
			raise FileExistsError(
				f'refusing to overwrite immutable file: {path}'
			) from error
	finally:
		if temporary is not None:
			temporary.unlink(missing_ok=True)


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return child


def _mapping_rows(value: object, label: str) -> list[Mapping[str, object]]:
	if not isinstance(value, list):
		raise TypeError(f'{label} must be a list')
	rows: list[Mapping[str, object]] = []
	for index, row in enumerate(value):
		if not isinstance(row, Mapping):
			raise TypeError(f'{label}[{index}] must be a mapping')
		rows.append(row)
	return rows


def _cell_key(row: Mapping[str, object]) -> tuple[str, str, str]:
	values = (row.get('candidate_id'), row.get('layout_id'), row.get('data_size'))
	if not all(isinstance(value, str) for value in values):
		raise TypeError('evidence cell identity must contain strings')
	return cast('tuple[str, str, str]', values)


def _type_sensitive_equal(actual: object, expected: object) -> bool:
	"""Compare scientific evidence without bool/int/float coercion."""
	if isinstance(expected, Mapping):
		if not isinstance(actual, Mapping) or set(actual) != set(expected):
			return False
		return all(
			_type_sensitive_equal(actual[key], expected[key]) for key in expected
		)
	if isinstance(expected, list):
		return (
			isinstance(actual, list)
			and len(actual) == len(expected)
			and all(
				_type_sensitive_equal(actual_value, expected_value)
				for actual_value, expected_value in zip(actual, expected, strict=True)
			)
		)
	if isinstance(expected, tuple):
		return (
			isinstance(actual, tuple)
			and len(actual) == len(expected)
			and all(
				_type_sensitive_equal(actual_value, expected_value)
				for actual_value, expected_value in zip(actual, expected, strict=True)
			)
		)
	return type(actual) is type(expected) and actual == expected


def _require_exact_keys(
	value: Mapping[str, object], expected: set[str], label: str
) -> None:
	if set(value) != expected:
		raise ValueError(
			f'{label} must define exactly {sorted(expected)!r}; got {sorted(value)!r}'
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


def _sha256_value(value: object, label: str) -> str:
	if not isinstance(value, str) or len(value) != 64:
		raise ValueError(f'{label} must be a SHA-256 digest')
	try:
		bytes.fromhex(value)
	except ValueError as error:
		raise ValueError(f'{label} must be a SHA-256 digest') from error
	return value


def _sha1_value(value: object, label: str) -> str:
	if not isinstance(value, str) or len(value) != 40:
		raise ValueError(f'{label} must be a Git commit')
	try:
		bytes.fromhex(value)
	except ValueError as error:
		raise ValueError(f'{label} must be a Git commit') from error
	return value


def _string(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty string')
	return value


def _macro_f1(metrics: Mapping[str, object], path: Path) -> float:
	if metrics.get('aggregation_unit') != VALIDATION_AGGREGATION_UNIT:
		raise ValueError(f'{path} does not use unique validation voxels')
	value = metrics.get('macro_f1')
	if not isinstance(value, int | float) or isinstance(value, bool):
		raise TypeError(f'{path} macro_f1 must be numeric')
	result = float(value)
	if not math.isfinite(result) or not 0.0 <= result <= 1.0:
		raise ValueError(f'{path} macro_f1 must be finite and within [0, 1]')
	return result


def _utc_timestamp() -> str:
	return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _validate_utc_timestamp(value: str) -> None:
	if not value.endswith('Z'):
		raise ValueError('created_at_utc must end in Z')
	try:
		parsed = datetime.fromisoformat(value.removesuffix('Z') + '+00:00')
	except ValueError as error:
		raise ValueError('created_at_utc is invalid') from error
	if parsed.tzinfo != timezone.utc:
		raise ValueError('created_at_utc must be UTC')


def build_parser() -> argparse.ArgumentParser:
	"""Build the one-candidate validation CLI."""
	parser = argparse.ArgumentParser(
		description='Run one validation-only trace-drop candidate job.'
	)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--candidate', choices=(CANDIDATE_ID,))
	parser.add_argument('--layout', choices=LAYOUT_IDS)
	parser.add_argument('--size', choices=DATA_SIZES)
	parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
	parser.add_argument('--resume', type=Path)
	parser.add_argument('--dry-run', action='store_true')
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument('--audit-parent-only', action='store_true')
	mode.add_argument('--audit-base-checkpoint-only', action='store_true')
	mode.add_argument('--create-protocol-lock', action='store_true')
	mode.add_argument('--audit-checkpoint-only', action='store_true')
	mode.add_argument('--create-final-result', action='store_true')
	return parser


def _validate_cli_arguments(args: argparse.Namespace) -> None:  # noqa: C901
	if args.audit_parent_only:
		if any(
			value is not None
			for value in (args.candidate, args.layout, args.size, args.resume)
		):
			raise ValueError('--audit-parent-only does not accept job arguments')
		if args.dry_run:
			raise ValueError('--audit-parent-only is already read-only')
		return
	if args.audit_base_checkpoint_only or args.audit_checkpoint_only:
		if args.candidate != CANDIDATE_ID:
			raise ValueError('checkpoint audit requires the fixed --candidate')
		if any(value is not None for value in (args.layout, args.size, args.resume)):
			raise ValueError('checkpoint audit does not accept layout, size, or resume')
		if args.dry_run:
			raise ValueError('checkpoint audit is already read-only')
		return
	if args.create_protocol_lock or args.create_final_result:
		if any(
			value is not None
			for value in (args.candidate, args.layout, args.size, args.resume)
		):
			raise ValueError('lock/result creation does not accept job arguments')
		if args.dry_run:
			raise ValueError('lock/result creation does not support --dry-run')
		return
	if args.candidate != CANDIDATE_ID or args.layout is None or args.size is None:
		raise ValueError('validation jobs require candidate, layout, and size')


def _print_mapping(payload: Mapping[str, object]) -> None:
	for key, value in payload.items():
		print(f'{key}: {value}')


def main() -> None:
	"""Audit, lock, or run exactly one validation-only operation."""
	args = build_parser().parse_args()
	_validate_cli_arguments(args)
	settings = validation_settings_from_mapping(load_config(args.config))
	canonical = _canonical_config(settings)
	_shared('_validate_benchmark_provenance')(settings, canonical, verify_files=True)
	if args.audit_parent_only:
		_print_mapping(validate_parent_result(settings, canonical))
		return
	if args.audit_base_checkpoint_only:
		if settings.protocol_lock.exists() or settings.protocol_lock.is_symlink():
			raise ValueError('base-only audit is forbidden after protocol lock')
		_print_mapping(
			audit_candidate_base_checkpoint(
				candidate=settings.candidate,
				canonical_config=canonical,
				reference_base_checkpoint=settings.reference_base_checkpoint,
			)
		)
		return
	if args.create_protocol_lock:
		_print_mapping(create_trace_drop_protocol_lock(settings, canonical))
		return
	protocol = validate_trace_drop_protocol_lock(settings, canonical)
	protocol_identity = _protocol_identity(settings, protocol)
	if args.audit_checkpoint_only:
		_print_mapping(
			{
				**audit_candidate_checkpoints(
					candidate=settings.candidate,
					canonical_config=canonical,
					reference_base_checkpoint=settings.reference_base_checkpoint,
					reference_final_checkpoint=settings.reference_final_checkpoint,
				),
				'protocol_lock': protocol_identity,
			}
		)
		return
	if args.create_final_result:
		_print_mapping(create_trace_drop_final_result(settings, canonical))
		return
	order = enforce_validation_order(
		settings=settings, canonical=canonical, data_size=args.size
	)
	config = _candidate_config(
		canonical, candidate=settings.candidate, runs_root=settings.runs_root
	)
	job = five_way_runner.resolve_f3_lithology_five_way_job(
		config, model=CANDIDATE_ID, layout=args.layout, size=args.size
	)
	expected_resume = job.decoder_dir / five_way_runner.LATEST_CHECKPOINT_NAME
	if args.resume is not None and args.resume != expected_resume:
		raise ValueError(f'--resume must equal this job latest.pt: {expected_resume}')
	if args.dry_run:
		summary = five_way_runner.inspect_f3_lithology_five_way_job(job)
		summary.update(
			{
				'candidate_source_audit': 'required before execution',
				'evaluation_split': 'validation',
				'evaluation_aggregation_unit': VALIDATION_AGGREGATION_UNIT,
				'protocol_lock': protocol_identity,
				'execution': 'dry-run; no files written',
			}
		)
		_print_mapping(summary)
		return
	audit = audit_candidate_source(
		candidate=settings.candidate,
		candidate_model=job.model,
		canonical_config=canonical,
		reference_base_checkpoint=settings.reference_base_checkpoint,
		reference_final_checkpoint=settings.reference_final_checkpoint,
		protocol_lock_identity=protocol_identity,
	)
	audit = {
		**audit,
		'validation_order_provenance': _validation_order_provenance(
			settings,
			protocol,
			cast('Mapping[str, object] | None', order.get('medium_gate')),
		),
	}
	_print_mapping(_run_job(job, audit=audit, device=args.device, resume=args.resume))


if __name__ == '__main__':
	main()
