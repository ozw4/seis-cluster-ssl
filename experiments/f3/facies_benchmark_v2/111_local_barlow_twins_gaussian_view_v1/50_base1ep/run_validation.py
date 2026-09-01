# ruff: noqa: INP001, SLF001, TC001
"""Run the reached one-base-epoch, validation-only F3 comparison.

This duration branch is authorized only by the immutable failed base-5
result. It inherits that result's selected Gaussian view without tuning, pairs
it with a fresh matched legacy-view control, and uses the unchanged canonical
F3 decoder/evaluator. The local locks and result are deliberately independent
of the base-5 branch.
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

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
GAUSSIAN25_RUNNER = EXPERIMENT_ROOT / 'run_validation.py'
GAUSSIAN25_RUNNER_SHA256 = (
	'a704e64a2da59c85a4e82e318bb3156c2cf74825ad50b4fb7463bd8dd2c1bccd'
)
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
	'5b1e9c8778892d1226179c92e7d9fd788cb13329e05e5663ed3dd8ddb1e18d43'
)
PARENT_FINAL_RESULT_SCHEMA_VERSION = 1
PARENT_FINAL_RESULT_TYPE = 'f3_local_barlow_twins_gaussian_base5_final_validation_v1'
PARENT_SELECTION_SCHEMA_VERSION = 1
PARENT_SELECTION_TYPE = 'f3_local_barlow_twins_gaussian_base5_selection_v1'
PARENT_SELECTION_SHA256 = (
	'41cef83f3f87eb2d5afc8159cb1fbc91a9287855a20172f4ebe7c55374c9a502'
)
GRANDPARENT_SELECTED_ID = 'local_barlow_twins_gaussian_noise_std010'
PARENT_SELECTED_ID = 'local_barlow_twins_gaussian_noise_std010_base5ep'

SELECTED_ID = 'local_barlow_twins_gaussian_noise_std010_base1ep'
LEGACY_ID = 'local_barlow_twins_legacy_flip_base1ep'
EXPECTED_SOURCE_IDS = (SELECTED_ID, LEGACY_ID)
HORIZONTAL_VIEW_POLICY = 'horizontal_flip_gaussian_noise_v1'
GAUSSIAN_NOISE_STD = 0.10
HORIZONTAL_FLIP_PROBABILITY = 0.5
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

CHECKPOINT_AUDIT_SCHEMA_VERSION = 1
PROTOCOL_LOCK_SCHEMA_VERSION = 1
PROTOCOL_LOCK_TYPE = 'f3_local_barlow_twins_gaussian_base1_protocol_v1'
SELECTION_LOCK_SCHEMA_VERSION = 1
SELECTION_LOCK_TYPE = 'f3_local_barlow_twins_gaussian_base1_selection_v1'
FINAL_RESULT_SCHEMA_VERSION = 1
FINAL_RESULT_TYPE = 'f3_local_barlow_twins_gaussian_base1_final_validation_v1'
VALIDATION_AGGREGATION_UNIT = 'unique_validation_voxel'
AUDIT_NAME = 'candidate_source_audit.json'

PARENT_FINAL_RESULT_RELATIVE_PATH = Path(
	'f3_lithology_benchmark/local_barlow_twins_gaussian_view_v1/'
	'base5ep/validation/gaussian_base5_final_result.json'
)
PARENT_SELECTION_RELATIVE_PATH = Path(
	'f3_lithology_benchmark/local_barlow_twins_gaussian_view_v1/'
	'base5ep/validation/gaussian_base5_selection_lock.json'
)
BRANCH_VALIDATION_RELATIVE_ROOT = Path(
	'f3_lithology_benchmark/local_barlow_twins_gaussian_view_v1/base1ep/validation'
)
EXPECTED_BASE_RELATIVE_PATHS = {
	SELECTED_ID: Path(
		'pretraining/f3/facies_benchmark_v1/'
		'local_barlow_twins_gaussian_view_v1/base1ep/stage1/'
		'gaussian_noise_std010_base1ep/full_1ep/latest.pt'
	),
	LEGACY_ID: Path(
		'pretraining/f3/facies_benchmark_v1/'
		'local_barlow_twins_gaussian_view_v1/base1ep/stage1/'
		'legacy_flip_base1ep/full_1ep/latest.pt'
	),
}
EXPECTED_FINAL_RELATIVE_PATHS = {
	SELECTED_ID: Path(
		'pretraining/f3/facies_benchmark_v1/'
		'local_barlow_twins_gaussian_view_v1/base1ep/stage2/'
		'gaussian_noise_std010_base1ep/local_bt_continue/full_25ep/latest.pt'
	),
	LEGACY_ID: Path(
		'pretraining/f3/facies_benchmark_v1/'
		'local_barlow_twins_gaussian_view_v1/base1ep/stage2/'
		'legacy_flip_base1ep/local_bt_continue/full_25ep/latest.pt'
	),
}
EXPECTED_EMBEDDING_RELATIVE_PATHS = {
	SELECTED_ID: Path(
		'embeddings/f3/facies_benchmark_v2/'
		'local_barlow_twins_gaussian_view_v1/base1ep/'
		'local_barlow_twins_gaussian_noise_std010_base1ep/overlap_x64'
	),
	LEGACY_ID: Path(
		'embeddings/f3/facies_benchmark_v2/'
		'local_barlow_twins_gaussian_view_v1/base1ep/'
		'local_barlow_twins_legacy_flip_base1ep/overlap_x64'
	),
}


def _load_gaussian25_namespace() -> dict[str, object]:
	"""Load only frozen shared benchmark/job helpers from the 25-epoch runner."""
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
		'_paired_arm_contrast',
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
	"""One fixed fresh-base/fixed-continuation validation arm."""

	candidate_id: str
	role: str
	parent_candidate_id: str | None
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
	"""Pinned parent, benchmark, two arms, and branch-local outputs."""

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
	sources: tuple[CandidateSource, ...]
	runs_root: Path
	protocol_lock: Path
	selection_lock: Path
	final_result: Path

	@property
	def candidates(self) -> tuple[CandidateSource, ...]:
		"""Compatibility view for fixed shared benchmark helpers."""
		return tuple(source for source in self.sources if source.selectable)

	@property
	def controls(self) -> tuple[CandidateSource, ...]:
		"""Compatibility view for fixed shared benchmark helpers."""
		return tuple(source for source in self.sources if not source.selectable)

	def source_by_id(self, source_id: str) -> CandidateSource:
		"""Return one of the two fixed duration-branch sources."""
		for source in self.sources:
			if source.candidate_id == source_id:
				return source
		raise ValueError(
			f'unknown base1 source {source_id!r}; expected {EXPECTED_SOURCE_IDS!r}'
		)


def validation_settings_from_mapping(  # noqa: C901
	config: Mapping[str, object],
) -> ValidationSettings:
	"""Resolve the strict, duration-specific validation configuration."""
	_require_exact_keys(config, {'parent', 'benchmark', 'sources', 'outputs'}, 'config')
	parent = _mapping(config, 'parent')
	benchmark = _mapping(config, 'benchmark')
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
		outputs,
		{'runs_root', 'protocol_lock', 'selection_lock', 'final_result'},
		'outputs',
	)
	values = config['sources']
	if not isinstance(values, Sequence) or isinstance(values, str | bytes):
		raise TypeError('sources must be a list')
	sources = tuple(
		_source_from_mapping(value, index=index) for index, value in enumerate(values)
	)
	if tuple(source.candidate_id for source in sources) != EXPECTED_SOURCE_IDS:
		raise ValueError(
			'sources must define selected Gaussian then matched legacy in fixed order'
		)
	for attribute in ('base_checkpoint', 'final_checkpoint', 'embeddings_dir'):
		paths = [cast('Path', getattr(source, attribute)) for source in sources]
		if len(paths) != len(set(paths)):
			raise ValueError(f'source {attribute} paths must be distinct')
	parent_sha = _sha256_value(
		parent['final_result_sha256'], 'parent.final_result_sha256'
	)
	if parent_sha != PARENT_FINAL_RESULT_SHA256:
		raise ValueError('parent.final_result_sha256 is not the pinned failed result')
	for key, expected_sha in EXPECTED_BENCHMARK_SHA256.items():
		actual_sha = _sha256_value(benchmark[key], f'benchmark.{key}')
		if actual_sha != expected_sha:
			raise ValueError(f'benchmark.{key} is not the pinned canonical digest')
	settings = ValidationSettings(
		parent_final_result=_absolute_path(
			parent['final_result'], 'parent.final_result'
		),
		parent_final_result_sha256=parent_sha,
		canonical_five_way_config=_absolute_path(
			benchmark['canonical_five_way_config'],
			'benchmark.canonical_five_way_config',
		),
		canonical_five_way_config_sha256=_sha256_value(
			benchmark['canonical_five_way_config_sha256'],
			'benchmark.canonical_five_way_config_sha256',
		),
		reference_base_checkpoint=_absolute_path(
			benchmark['reference_base_checkpoint'],
			'benchmark.reference_base_checkpoint',
		),
		reference_base_checkpoint_sha256=_sha256_value(
			benchmark['reference_base_checkpoint_sha256'],
			'benchmark.reference_base_checkpoint_sha256',
		),
		reference_final_checkpoint=_absolute_path(
			benchmark['reference_final_checkpoint'],
			'benchmark.reference_final_checkpoint',
		),
		reference_final_checkpoint_sha256=_sha256_value(
			benchmark['reference_final_checkpoint_sha256'],
			'benchmark.reference_final_checkpoint_sha256',
		),
		random_checkpoint_sha256=_sha256_value(
			benchmark['random_checkpoint_sha256'],
			'benchmark.random_checkpoint_sha256',
		),
		canonical_comparison_sha256=_sha256_value(
			benchmark['canonical_comparison_sha256'],
			'benchmark.canonical_comparison_sha256',
		),
		pretraining_manifest_sha256=_sha256_value(
			benchmark['pretraining_manifest_sha256'],
			'benchmark.pretraining_manifest_sha256',
		),
		pretraining_path_list_sha256=_sha256_value(
			benchmark['pretraining_path_list_sha256'],
			'benchmark.pretraining_path_list_sha256',
		),
		sources=sources,
		runs_root=_absolute_path(outputs['runs_root'], 'outputs.runs_root'),
		protocol_lock=_absolute_path(outputs['protocol_lock'], 'outputs.protocol_lock'),
		selection_lock=_absolute_path(
			outputs['selection_lock'], 'outputs.selection_lock'
		),
		final_result=_absolute_path(outputs['final_result'], 'outputs.final_result'),
	)
	if settings.protocol_lock != (
		settings.runs_root.parent / 'gaussian_base1_protocol_lock.json'
	):
		raise ValueError('protocol lock must be gaussian_base1_protocol_lock.json')
	if settings.selection_lock != (
		settings.runs_root.parent / 'gaussian_base1_selection_lock.json'
	):
		raise ValueError('selection lock must be gaussian_base1_selection_lock.json')
	if settings.final_result != (
		settings.runs_root.parent / 'gaussian_base1_final_result.json'
	):
		raise ValueError('final result must be gaussian_base1_final_result.json')
	return settings


def _source_from_mapping(value: object, *, index: int) -> CandidateSource:
	if not isinstance(value, Mapping):
		raise TypeError(f'sources[{index}] must be a mapping')
	_require_exact_keys(
		value,
		{
			'candidate_id',
			'role',
			'parent_candidate_id',
			'base_checkpoint',
			'final_checkpoint',
			'embeddings_dir',
			'view_policy',
			'gaussian_noise_std',
			'base_pretraining_epochs',
			'continuation_epochs',
			'selectable',
		},
		f'sources[{index}]',
	)
	candidate_id = value['candidate_id']
	if candidate_id not in EXPECTED_SOURCE_IDS:
		raise ValueError(f'sources[{index}].candidate_id is unsupported')
	selected = candidate_id == SELECTED_ID
	expected = {
		'role': 'inherited_selected_view' if selected else 'matched_legacy_control',
		'parent_candidate_id': PARENT_SELECTED_ID if selected else None,
		'view_policy': HORIZONTAL_VIEW_POLICY if selected else None,
		'gaussian_noise_std': GAUSSIAN_NOISE_STD if selected else None,
		'base_pretraining_epochs': BASE_PRETRAINING_EPOCHS,
		'continuation_epochs': CONTINUATION_EPOCHS,
		'selectable': selected,
	}
	for key, expected_value in expected.items():
		actual = value[key]
		if key == 'gaussian_noise_std' and actual is not None:
			if not isinstance(actual, int | float) or isinstance(actual, bool):
				raise TypeError(f'sources[{index}].{key} must be numeric or null')
			actual = float(actual)
		if not _type_sensitive_equal(actual, expected_value):
			raise ValueError(f'sources[{index}].{key} must equal {expected_value!r}')
	return CandidateSource(
		candidate_id=cast('str', candidate_id),
		role=cast('str', value['role']),
		parent_candidate_id=cast('str | None', value['parent_candidate_id']),
		base_checkpoint=_absolute_path(
			value['base_checkpoint'], f'sources[{index}].base_checkpoint'
		),
		final_checkpoint=_absolute_path(
			value['final_checkpoint'], f'sources[{index}].final_checkpoint'
		),
		embeddings_dir=_absolute_path(
			value['embeddings_dir'], f'sources[{index}].embeddings_dir'
		),
		view_policy=cast('str | None', value['view_policy']),
		gaussian_noise_std=(GAUSSIAN_NOISE_STD if selected else None),
		base_pretraining_epochs=BASE_PRETRAINING_EPOCHS,
		continuation_epochs=CONTINUATION_EPOCHS,
		selectable=selected,
	)


def _canonical_config(settings: ValidationSettings) -> F3FiveWayConfig:
	canonical = cast('F3FiveWayConfig', _shared('_canonical_config')(settings))
	_validate_branch_paths(settings, canonical)
	return canonical


def _validate_branch_paths(
	settings: ValidationSettings, canonical: F3FiveWayConfig
) -> None:
	artifact_root = canonical.artifact_root
	if (
		settings.parent_final_result
		!= artifact_root / PARENT_FINAL_RESULT_RELATIVE_PATH
	):
		raise ValueError('parent result path is not the pinned base5 result')
	validation_root = artifact_root / BRANCH_VALIDATION_RELATIVE_ROOT
	if settings.runs_root != validation_root / 'runs':
		raise ValueError('runs_root must use the isolated base1ep validation root')
	for source in settings.sources:
		if source.base_checkpoint != (
			artifact_root / EXPECTED_BASE_RELATIVE_PATHS[source.candidate_id]
		):
			raise ValueError(f'{source.candidate_id} base checkpoint path changed')
		if source.final_checkpoint != (
			artifact_root / EXPECTED_FINAL_RELATIVE_PATHS[source.candidate_id]
		):
			raise ValueError(f'{source.candidate_id} final checkpoint path changed')
		if source.embeddings_dir != (
			artifact_root / EXPECTED_EMBEDDING_RELATIVE_PATHS[source.candidate_id]
		):
			raise ValueError(f'{source.candidate_id} embedding path changed')


def validate_parent_result(  # noqa: C901
	settings: ValidationSettings, canonical: F3FiveWayConfig
) -> dict[str, object]:
	"""Audit the immutable failed parent and its selected-view payload."""
	expected_result = canonical.artifact_root / PARENT_FINAL_RESULT_RELATIVE_PATH
	if settings.parent_final_result != expected_result:
		raise ValueError(
			'configured parent result is not the canonical base5 result'
		)
	parent, parent_sha = _read_hashed_json(
		settings.parent_final_result, label='parent base5 final result'
	)
	if parent_sha != settings.parent_final_result_sha256:
		raise ValueError('parent base5 final-result SHA-256 changed')
	required_values: Mapping[str, object] = {
		'schema_version': PARENT_FINAL_RESULT_SCHEMA_VERSION,
		'final_result_type': PARENT_FINAL_RESULT_TYPE,
		'validation_only': True,
		'base_pretraining_epochs': 5,
		'continuation_epochs': 25,
		'passed': False,
		'winner_candidate_id': None,
		'authorizes_next_base_duration': True,
		'authorized_next_base_pretraining_epochs': 1,
		'failure_stage': 'medium_5of5',
	}
	for key, expected in required_values.items():
		if not _type_sensitive_equal(parent.get(key), expected):
			raise ValueError(
				f'parent base5 final result {key} must equal {expected!r}'
			)
	parent_selection = _mapping(parent, 'selection_lock')
	_require_exact_keys(
		parent_selection,
		{
			'path',
			'sha256',
			'parent_selected_candidate_id',
			'selected_candidate_id',
		},
		'parent final-result selection_lock',
	)
	if (
		parent_selection.get('parent_selected_candidate_id')
		!= GRANDPARENT_SELECTED_ID
	):
		raise ValueError('parent result changed the original std010 selection')
	if parent_selection.get('selected_candidate_id') != PARENT_SELECTED_ID:
		raise ValueError('parent result did not select the pinned std010 view')
	selection_path_value = parent_selection.get('path')
	if not isinstance(selection_path_value, str):
		raise TypeError('parent selection-lock path must be a string')
	selection_path = Path(selection_path_value)
	expected_selection = canonical.artifact_root / PARENT_SELECTION_RELATIVE_PATH
	if selection_path != expected_selection:
		raise ValueError('parent selection-lock path changed')
	selection, selection_sha = _read_hashed_json(
		selection_path, label='parent base5 selection lock'
	)
	embedded_selection_sha = _sha256_value(
		parent_selection.get('sha256'), 'parent selection-lock SHA-256'
	)
	if embedded_selection_sha != PARENT_SELECTION_SHA256:
		raise ValueError('parent selection-lock identity is not the pinned base5 lock')
	if selection_sha != embedded_selection_sha:
		raise ValueError('parent selection-lock SHA-256 changed')
	selection_values: Mapping[str, object] = {
		'schema_version': PARENT_SELECTION_SCHEMA_VERSION,
		'selection_lock_type': PARENT_SELECTION_TYPE,
		'validation_only': True,
		'base_pretraining_epochs': 5,
		'continuation_epochs': 25,
		'parent_selected_candidate_id': GRANDPARENT_SELECTED_ID,
		'selected_candidate_id': PARENT_SELECTED_ID,
		'selected_view_policy': HORIZONTAL_VIEW_POLICY,
		'selected_gaussian_noise_std': GAUSSIAN_NOISE_STD,
		'selection_basis': 'inherited_failed_gaussian25_selection_no_base5_metrics',
		'base5_metric_inputs': [],
	}
	for key, expected in selection_values.items():
		if not _type_sensitive_equal(selection.get(key), expected):
			raise ValueError(
				f'parent base5 selection lock {key} must equal {expected!r}'
			)
	return {
		'path': str(settings.parent_final_result),
		'sha256': parent_sha,
		'schema_version': PARENT_FINAL_RESULT_SCHEMA_VERSION,
		'final_result_type': PARENT_FINAL_RESULT_TYPE,
		'passed': False,
		'authorizes_next_base_duration': True,
		'authorized_next_base_pretraining_epochs': 1,
		'parent_selection_lock': {
			'path': str(selection_path),
			'sha256': selection_sha,
			'parent_selected_candidate_id': GRANDPARENT_SELECTED_ID,
			'selected_candidate_id': PARENT_SELECTED_ID,
			'selected_view_policy': HORIZONTAL_VIEW_POLICY,
			'selected_gaussian_noise_std': GAUSSIAN_NOISE_STD,
		},
	}


def _expected_augmentations(source: CandidateSource) -> dict[str, object]:
	if source.candidate_id == SELECTED_ID:
		return {
			'policy': HORIZONTAL_VIEW_POLICY,
			'horizontal_flip_probability': HORIZONTAL_FLIP_PROBABILITY,
			'gaussian_noise_std': GAUSSIAN_NOISE_STD,
		}
	if source.candidate_id == LEGACY_ID:
		return {'horizontal_flip_probability': HORIZONTAL_FLIP_PROBABILITY}
	raise ValueError(f'unsupported source: {source.candidate_id!r}')


def audit_candidate_base_checkpoint(
	*,
	candidate: CandidateSource,
	canonical_config: F3FiveWayConfig,
	reference_base_checkpoint: Path,
) -> dict[str, object]:
	"""Audit one fresh one-epoch base before either continuation starts."""
	_shared('_validate_reference_base_checkpoint')(
		canonical_config=canonical_config,
		reference_base_checkpoint=reference_base_checkpoint,
		verify_file=True,
	)
	if candidate.base_checkpoint.is_symlink():
		raise ValueError('candidate base checkpoint must not be a symlink')
	if not candidate.base_checkpoint.is_file():
		raise FileNotFoundError(
			f'missing candidate base checkpoint: {candidate.base_checkpoint}'
		)
	payload = load_checkpoint_metadata_without_weights(candidate.base_checkpoint)
	reference = load_checkpoint_metadata_without_weights(reference_base_checkpoint)
	_validate_candidate_base_checkpoint(candidate, payload=payload, reference=reference)
	return {
		'audit_schema_version': CHECKPOINT_AUDIT_SCHEMA_VERSION,
		'audit_type': 'base1_pre_continuation_checkpoint_only',
		'candidate_id': candidate.candidate_id,
		'base_checkpoint': str(candidate.base_checkpoint),
		'base_checkpoint_sha256': file_sha256(candidate.base_checkpoint),
		'base_pretraining_epochs': BASE_PRETRAINING_EPOCHS,
		'base_global_step': EXPECTED_BASE_STEPS,
		'base_resume_count': 0,
		'augmentations': _expected_augmentations(candidate),
		'reference_base_checkpoint': str(reference_base_checkpoint),
		'reference_base_checkpoint_sha256': file_sha256(reference_base_checkpoint),
		'base_parity_exceptions': _base_parity_exceptions(candidate),
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
	paths = _mapping(config, 'paths')
	if paths.get('output_root') != str(candidate.base_checkpoint.parent):
		raise ValueError('candidate base output_root must own its checkpoint')
	if not _type_sensitive_equal(
		_mapping(config, 'augmentations'), _expected_augmentations(candidate)
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
		_base_parity_projection(
			config, allow_augmentation_difference=candidate.selectable
		),
		_base_parity_projection(
			reference_config, allow_augmentation_difference=candidate.selectable
		),
	):
		raise ValueError(
			'candidate base differs from canonical outside duration, output root, '
			'and the selected arm augmentation'
		)


def audit_candidate_checkpoints(
	*,
	candidate: CandidateSource,
	canonical_config: F3FiveWayConfig,
	reference_base_checkpoint: Path,
	reference_final_checkpoint: Path,
) -> dict[str, object]:
	"""Audit a fresh one-epoch base and exact fresh 25-epoch continuation."""
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
	if candidate.final_checkpoint.is_symlink():
		raise ValueError('candidate final checkpoint must not be a symlink')
	if not candidate.final_checkpoint.is_file():
		raise FileNotFoundError(
			f'missing candidate final checkpoint: {candidate.final_checkpoint}'
		)
	payload = load_checkpoint_metadata_without_weights(candidate.final_checkpoint)
	reference = load_checkpoint_metadata_without_weights(reference_final_checkpoint)
	_validate_candidate_final_checkpoint(
		candidate, payload=payload, reference=reference
	)
	return {
		'audit_schema_version': CHECKPOINT_AUDIT_SCHEMA_VERSION,
		'audit_type': 'base1_pre_extraction_lineage_checkpoint_only',
		'candidate_id': candidate.candidate_id,
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
		'augmentations': _expected_augmentations(candidate),
		'reference_base_checkpoint': str(reference_base_checkpoint),
		'reference_base_checkpoint_sha256': file_sha256(reference_base_checkpoint),
		'reference_final_checkpoint': str(reference_final_checkpoint),
		'reference_final_checkpoint_sha256': file_sha256(reference_final_checkpoint),
		'base_parity_exceptions': _base_parity_exceptions(candidate),
		'final_parity_exceptions': _final_parity_exceptions(),
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
	paths = _mapping(config, 'paths')
	if paths.get('output_root') != str(candidate.final_checkpoint.parent):
		raise ValueError('candidate continuation output_root must own its checkpoint')
	if not _type_sensitive_equal(
		_mapping(config, 'augmentations'), _expected_augmentations(candidate)
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
		raise ValueError('continuation must initialize from its exact base path')
	unfreeze_top_blocks = continuation.get('unfreeze_top_blocks')
	if (
		not isinstance(unfreeze_top_blocks, int)
		or isinstance(unfreeze_top_blocks, bool)
		or unfreeze_top_blocks != CONTINUATION_UNFREEZE_TOP_BLOCKS
	):
		raise ValueError('continuation must unfreeze exactly one top block')
	lineage = _mapping(payload, 'continuation_lineage')
	_require_exact_keys(
		lineage,
		{'schema_version', 'init_checkpoint', 'init_checkpoint_sha256', 'resume_count'},
		'candidate continuation lineage',
	)
	lineage_schema_version = lineage.get('schema_version')
	if (
		not isinstance(lineage_schema_version, int)
		or isinstance(lineage_schema_version, bool)
		or lineage_schema_version != 1
	):
		raise ValueError('continuation lineage schema_version must equal 1')
	if lineage.get('init_checkpoint') != str(candidate.base_checkpoint):
		raise ValueError('continuation lineage base path changed')
	if lineage.get('init_checkpoint_sha256') != file_sha256(candidate.base_checkpoint):
		raise ValueError('continuation lineage base SHA-256 changed')
	lineage_resume_count = lineage.get('resume_count')
	if (
		not isinstance(lineage_resume_count, int)
		or isinstance(lineage_resume_count, bool)
		or lineage_resume_count != 0
	):
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
		raise ValueError(f'{label} method must be local Barlow')
	if not _type_sensitive_equal(payload.get('epoch'), epochs):
		raise ValueError(f'{label} epoch must equal {epochs}')
	if not _type_sensitive_equal(payload.get('global_step'), global_step):
		raise ValueError(f'{label} global_step must equal {global_step}')
	if not _type_sensitive_equal(_mapping(payload, 'training_state'), training_state):
		raise ValueError(f'{label} completed training_state changed')
	resume_count = payload.get('resume_count')
	if (
		not isinstance(resume_count, int)
		or isinstance(resume_count, bool)
		or resume_count != 0
	):
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
	if (
		barlow.get('method') != LOCAL_BARLOW_METHOD
		or not _type_sensitive_equal(
			barlow.get('local_pairs_per_crop'), LOCAL_PAIRS_PER_CROP
		)
	):
		raise ValueError(f'{label} changes the Local Barlow objective')


def _validate_training_steps(
	config: Mapping[str, object],
	*,
	epochs: int,
	expected_steps: int,
	label: str,
) -> None:
	train = _mapping(config, 'train')
	if not _type_sensitive_equal(train.get('epochs'), epochs):
		raise ValueError(f'{label} train.epochs must equal {epochs}')
	samples = _positive_int(train.get('samples_per_epoch'), f'{label} samples')
	batch = _positive_int(train.get('batch_size'), f'{label} batch size')
	if samples % batch or epochs * samples // batch != expected_steps:
		raise ValueError(f'{label} no longer yields {expected_steps} steps')


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


def create_base1_protocol_lock(
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	*,
	created_at_utc: str | None = None,
	git_head: str | None = None,
) -> dict[str, object]:
	"""Seal parent authorization and both fresh bases before continuation."""
	if settings.protocol_lock.exists() or settings.protocol_lock.is_symlink():
		raise FileExistsError(f'protocol lock already exists: {settings.protocol_lock}')
	_reject_pre_protocol_evidence(settings)
	parent = validate_parent_result(settings, canonical)
	benchmark = cast(
		'Mapping[str, object]',
		_shared('_validate_benchmark_provenance')(
			settings, canonical, verify_files=True
		),
	)
	repository = cast('Mapping[str, object]', _shared('_git_repository_state')())
	resolved_git_head = cast('str', repository['git_head'])
	if git_head is not None and git_head != resolved_git_head:
		raise ValueError('protocol Git HEAD must equal the live repository')
	bases = _collect_base_audits(settings, canonical)
	payload = _protocol_lock_payload(
		parent_result=parent,
		base_inputs=bases,
		benchmark_provenance=benchmark,
		repository_state=repository,
		created_at_utc=created_at_utc or _utc_timestamp(),
		git_head=resolved_git_head,
	)
	_reject_pre_protocol_evidence(settings)
	_write_exclusive_json(settings.protocol_lock, payload)
	return payload


def validate_base1_protocol_lock(
	settings: ValidationSettings, canonical: F3FiveWayConfig
) -> Mapping[str, object]:
	"""Replay the immutable base1 protocol against all live frozen inputs."""
	stored = _read_regular_json(settings.protocol_lock, label='base1 protocol lock')
	created = _string(stored.get('created_at_utc'), 'protocol created_at_utc')
	_validate_utc_timestamp(created)
	git_head = _sha1_value(stored.get('git_head'), 'protocol git_head')
	parent = validate_parent_result(settings, canonical)
	benchmark = cast(
		'Mapping[str, object]',
		_shared('_validate_benchmark_provenance')(
			settings, canonical, verify_files=True
		),
	)
	repository = cast('Mapping[str, object]', _shared('_git_repository_state')())
	expected = _protocol_lock_payload(
		parent_result=parent,
		base_inputs=_collect_base_audits(settings, canonical),
		benchmark_provenance=benchmark,
		repository_state=repository,
		created_at_utc=created,
		git_head=git_head,
	)
	if not _type_sensitive_equal(stored, expected):
		raise ValueError('base1 protocol lock differs from live frozen inputs')
	return stored


def _collect_base_audits(
	settings: ValidationSettings, canonical: F3FiveWayConfig
) -> list[dict[str, object]]:
	return [
		audit_candidate_base_checkpoint(
			candidate=settings.source_by_id(source_id),
			canonical_config=canonical,
			reference_base_checkpoint=settings.reference_base_checkpoint,
		)
		for source_id in EXPECTED_SOURCE_IDS
	]


def _protocol_lock_payload(  # noqa: PLR0913
	*,
	parent_result: Mapping[str, object],
	base_inputs: Sequence[Mapping[str, object]],
	benchmark_provenance: Mapping[str, object],
	repository_state: Mapping[str, object],
	created_at_utc: str,
	git_head: str,
) -> dict[str, object]:
	_validate_utc_timestamp(created_at_utc)
	_sha1_value(git_head, 'protocol git_head')
	if [row.get('candidate_id') for row in base_inputs] != list(EXPECTED_SOURCE_IDS):
		raise ValueError('protocol must bind both base1 arms in fixed order')
	return {
		'schema_version': PROTOCOL_LOCK_SCHEMA_VERSION,
		'protocol_lock_type': PROTOCOL_LOCK_TYPE,
		'validation_only': True,
		'stage_boundary': 'completed_fresh_base1_before_continuation',
		'base_pretraining_epochs': BASE_PRETRAINING_EPOCHS,
		'continuation_epochs': CONTINUATION_EPOCHS,
		'parent_result': dict(parent_result),
		'base_checkpoint_inputs': [dict(row) for row in base_inputs],
		'benchmark_provenance': dict(benchmark_provenance),
		'repository_state': dict(repository_state),
		'created_at_utc': created_at_utc,
		'git_head': git_head,
	}


def _reject_pre_protocol_evidence(settings: ValidationSettings) -> None:
	for label, path in (
		('selection lock', settings.selection_lock),
		('final result', settings.final_result),
		('validation runs', settings.runs_root),
	):
		if _path_contains_evidence(path):
			raise ValueError(f'cannot lock protocol after {label} exists: {path}')
	for source in settings.sources:
		for label, path in (
			('continuation output', source.final_checkpoint.parent),
			('embedding output', source.embeddings_dir),
		):
			if _path_contains_evidence(path):
				raise ValueError(
					f'cannot lock protocol after {source.candidate_id} {label}: {path}'
				)


def _reject_pre_selection_evidence(settings: ValidationSettings) -> None:
	if _path_contains_evidence(settings.final_result) or _path_contains_evidence(
		settings.runs_root
	):
		raise ValueError('selection lock must precede all base1 validation evidence')
	for source in settings.sources:
		for path in (source.final_checkpoint.parent, source.embeddings_dir):
			if _path_contains_evidence(path):
				raise ValueError(
					'selection lock must be created immediately after protocol and '
					f'before later evidence: {path}'
				)


def _path_contains_evidence(path: Path) -> bool:
	if path.is_symlink() or path.is_file():
		return True
	if path.is_dir():
		return next(path.iterdir(), None) is not None
	return path.exists()


def _protocol_identity(
	settings: ValidationSettings, protocol: Mapping[str, object]
) -> dict[str, str]:
	if protocol.get('protocol_lock_type') != PROTOCOL_LOCK_TYPE:
		raise ValueError('base1 protocol lock type changed')
	return {
		'path': str(settings.protocol_lock),
		'sha256': file_sha256(settings.protocol_lock),
	}


def create_base1_selection_lock(
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	*,
	created_at_utc: str | None = None,
) -> dict[str, object]:
	"""Inherit the parent choice before any continuation or validation evidence."""
	if settings.selection_lock.exists() or settings.selection_lock.is_symlink():
		raise FileExistsError(
			f'selection lock already exists: {settings.selection_lock}'
		)
	protocol = validate_base1_protocol_lock(settings, canonical)
	_reject_pre_selection_evidence(settings)
	payload = _selection_lock_payload(
		parent_result=validate_parent_result(settings, canonical),
		protocol_lock=_protocol_identity(settings, protocol),
		repository_state=_mapping(protocol, 'repository_state'),
		benchmark_provenance=_mapping(protocol, 'benchmark_provenance'),
		created_at_utc=created_at_utc or _utc_timestamp(),
		git_head=_string(protocol.get('git_head'), 'protocol git_head'),
	)
	_reject_pre_selection_evidence(settings)
	_write_exclusive_json(settings.selection_lock, payload)
	return payload


def validate_base1_selection_lock(
	settings: ValidationSettings, canonical: F3FiveWayConfig
) -> Mapping[str, object]:
	"""Replay the inherited selection without reading any base1 metric."""
	protocol = validate_base1_protocol_lock(settings, canonical)
	stored = _read_regular_json(settings.selection_lock, label='base1 selection lock')
	created = _string(stored.get('created_at_utc'), 'selection created_at_utc')
	_validate_utc_timestamp(created)
	expected = _selection_lock_payload(
		parent_result=validate_parent_result(settings, canonical),
		protocol_lock=_protocol_identity(settings, protocol),
		repository_state=_mapping(protocol, 'repository_state'),
		benchmark_provenance=_mapping(protocol, 'benchmark_provenance'),
		created_at_utc=created,
		git_head=_string(protocol.get('git_head'), 'protocol git_head'),
	)
	if not _type_sensitive_equal(stored, expected):
		raise ValueError('base1 selection lock differs from inherited parent choice')
	return stored


def _selection_lock_payload(  # noqa: PLR0913
	*,
	parent_result: Mapping[str, object],
	protocol_lock: Mapping[str, object],
	repository_state: Mapping[str, object],
	benchmark_provenance: Mapping[str, object],
	created_at_utc: str,
	git_head: str,
) -> dict[str, object]:
	_validate_utc_timestamp(created_at_utc)
	_sha1_value(git_head, 'selection git_head')
	parent_selection = _mapping(parent_result, 'parent_selection_lock')
	if parent_selection.get('selected_candidate_id') != PARENT_SELECTED_ID:
		raise ValueError('selection payload parent choice changed')
	return {
		'schema_version': SELECTION_LOCK_SCHEMA_VERSION,
		'selection_lock_type': SELECTION_LOCK_TYPE,
		'validation_only': True,
		'selection_basis': 'inherited_failed_base5_selection_no_base1_metrics',
		'base1_metric_inputs': [],
		'base_pretraining_epochs': BASE_PRETRAINING_EPOCHS,
		'continuation_epochs': CONTINUATION_EPOCHS,
		'parent_result': dict(parent_result),
		'protocol_lock': dict(protocol_lock),
		'parent_selected_candidate_id': PARENT_SELECTED_ID,
		'selected_candidate_id': SELECTED_ID,
		'selected_view_policy': HORIZONTAL_VIEW_POLICY,
		'selected_gaussian_noise_std': GAUSSIAN_NOISE_STD,
		'matched_legacy_candidate_id': LEGACY_ID,
		'candidate_id_mapping': {
			PARENT_SELECTED_ID: SELECTED_ID,
		},
		'repository_state': dict(repository_state),
		'benchmark_provenance': dict(benchmark_provenance),
		'created_at_utc': created_at_utc,
		'git_head': git_head,
	}


def _selection_identity(
	settings: ValidationSettings, selection: Mapping[str, object]
) -> dict[str, object]:
	if selection.get('selection_lock_type') != SELECTION_LOCK_TYPE:
		raise ValueError('base1 selection lock type changed')
	return {
		'path': str(settings.selection_lock),
		'sha256': file_sha256(settings.selection_lock),
		'parent_selected_candidate_id': PARENT_SELECTED_ID,
		'selected_candidate_id': SELECTED_ID,
	}


def audit_candidate_source(  # noqa: PLR0913
	*,
	candidate: CandidateSource,
	candidate_model: F3FiveWayModelSource,
	canonical_config: F3FiveWayConfig,
	reference_base_checkpoint: Path,
	reference_final_checkpoint: Path,
	protocol_lock_identity: Mapping[str, object],
	selection_lock_identity: Mapping[str, object],
) -> dict[str, object]:
	"""Audit the complete checkpoint/extraction lineage for one base1 arm."""
	checkpoint = audit_candidate_checkpoints(
		candidate=candidate,
		canonical_config=canonical_config,
		reference_base_checkpoint=reference_base_checkpoint,
		reference_final_checkpoint=reference_final_checkpoint,
	)
	survey_id = canonical_config.dataset['name']
	files = output_paths(candidate.embeddings_dir, survey_id)
	random_files = output_paths(
		canonical_config.model_by_id('random').embeddings_dir, survey_id
	)
	for label, path in (
		('candidate embeddings', files.embeddings),
		('candidate valid-token mask', files.valid_tokens),
		('candidate embedding metadata', files.metadata),
		('random valid-token mask', random_files.valid_tokens),
		('random embedding metadata', random_files.metadata),
	):
		if path.is_symlink():
			raise ValueError(f'{label} must not be a symlink: {path}')
		if not path.is_file():
			raise FileNotFoundError(f'missing {label}: {path}')
	metadata = _read_regular_json(files.metadata, label='candidate metadata')
	random_metadata = _read_regular_json(random_files.metadata, label='random metadata')
	five_way_sources._validate_extraction_contract(candidate.candidate_id, metadata)
	five_way_sources._validate_shared_identity(
		candidate.candidate_id, metadata, random_metadata
	)
	token_grid_shape = five_way_sources._token_grid_shape(
		candidate.candidate_id, metadata
	)
	five_way_sources._validate_arrays(candidate.candidate_id, files, token_grid_shape)
	if not five_way_sources._masks_identical(
		files.valid_tokens, random_files.valid_tokens
	):
		raise ValueError(
			f'{candidate.candidate_id} valid-token mask differs from random'
		)
	final_sha = five_way_sources._validate_checkpoint_identity(
		candidate_model, metadata
	)
	five_way_sources._validate_objective_identity(candidate_model, metadata)
	objective = _mapping(metadata, 'pretraining_objective')
	expected_augmentations = _expected_augmentations(candidate)
	if candidate.selectable:
		if not _type_sensitive_equal(
			objective.get('augmentations'), expected_augmentations
		):
			raise ValueError('selected embedding objective augmentation changed')
	elif 'augmentations' in objective:
		raise ValueError('legacy embedding objective introduced a named policy')
	if final_sha != checkpoint['final_checkpoint_sha256']:
		raise ValueError('embedding metadata and final checkpoint SHA-256 disagree')
	embeddings_sha256 = file_sha256(files.embeddings)
	embedding_metadata_sha256 = file_sha256(files.metadata)
	valid_tokens_sha256 = file_sha256(files.valid_tokens)
	return {
		'audit_schema_version': CHECKPOINT_AUDIT_SCHEMA_VERSION,
		'audit_type': 'base1_validation_source_lineage',
		'candidate_id': candidate.candidate_id,
		'source_role': candidate.role,
		'validation_only': True,
		'selection_eligible': candidate.selectable,
		'parent_candidate_id': candidate.parent_candidate_id,
		'protocol_lock': dict(protocol_lock_identity),
		'selection_lock': dict(selection_lock_identity),
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
		'embeddings_sha256': embeddings_sha256,
		'embedding_metadata': str(files.metadata),
		'embedding_metadata_sha256': embedding_metadata_sha256,
		'valid_tokens_path': str(files.valid_tokens),
		'valid_tokens_sha256': valid_tokens_sha256,
		'augmentations': expected_augmentations,
		'reference_base_checkpoint': str(reference_base_checkpoint),
		'reference_base_checkpoint_sha256': checkpoint[
			'reference_base_checkpoint_sha256'
		],
		'reference_final_checkpoint': str(reference_final_checkpoint),
		'reference_final_checkpoint_sha256': checkpoint[
			'reference_final_checkpoint_sha256'
		],
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


def enforce_validation_order(
	*,
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	candidate: CandidateSource,
	data_size: str,
) -> dict[str, object]:
	"""Require inherited selection for medium and a strict 5/5 gate thereafter."""
	if candidate.candidate_id not in EXPECTED_SOURCE_IDS:
		raise ValueError('only the two fixed base1 arms are admissible')
	protocol = validate_base1_protocol_lock(settings, canonical)
	selection = validate_base1_selection_lock(settings, canonical)
	if data_size == 'medium':
		return {
			'protocol_lock': protocol,
			'selection_lock': selection,
			'medium_gate': None,
		}
	gate = _validate_medium_random_gate(
		settings=settings,
		canonical=canonical,
		selection_lock=selection,
		require_open=True,
	)
	return {
		'protocol_lock': protocol,
		'selection_lock': selection,
		'medium_gate': gate,
	}


def _validation_order_provenance(
	settings: ValidationSettings,
	order: Mapping[str, object],
) -> dict[str, object]:
	protocol = _mapping(order, 'protocol_lock')
	selection = _mapping(order, 'selection_lock')
	result: dict[str, object] = {
		'protocol_lock': _protocol_identity(settings, protocol),
		'selection_lock': _selection_identity(settings, selection),
		'medium_gate': None,
	}
	gate = order.get('medium_gate')
	if gate is None:
		return result
	if not isinstance(gate, Mapping):
		raise TypeError('medium gate provenance must be a mapping')
	inputs = gate.get('inputs')
	if not isinstance(inputs, list):
		raise TypeError('medium gate inputs must be a list')
	identities = [
		{
			'candidate_id': row.get('candidate_id'),
			'layout_id': row.get('layout_id'),
			'data_size': row.get('data_size'),
			'metrics_path': row.get('metrics_path'),
			'metrics_sha256': row.get('metrics_sha256'),
		}
		for row in inputs
		if isinstance(row, Mapping)
	]
	if len(identities) != 15:
		raise ValueError('base1 medium gate must bind ten arm and five random cells')
	result['medium_gate'] = {
		'gate_open': gate.get('gate_open'),
		'selected_wins_over_random': gate.get('selected_wins_over_random'),
		'legacy_wins_over_random': gate.get('legacy_wins_over_random'),
		'inputs': identities,
	}
	return result


def _validate_medium_random_gate(
	*,
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	selection_lock: Mapping[str, object],
	require_open: bool = True,
) -> dict[str, object]:
	protocol = validate_base1_protocol_lock(settings, canonical)
	protocol_identity = _protocol_identity(settings, protocol)
	selection_identity = _selection_identity(settings, selection_lock)
	scores: dict[str, dict[str, float]] = {
		SELECTED_ID: {},
		LEGACY_ID: {},
		'random': {},
	}
	inputs: list[dict[str, object]] = []
	live_audits: dict[str, Mapping[str, object]] = {}
	medium_order = _validation_order_provenance(
		settings,
		{
			'protocol_lock': protocol,
			'selection_lock': selection_lock,
			'medium_gate': None,
		},
	)
	for source in settings.sources:
		config = _candidate_config(
			canonical, candidate=source, runs_root=settings.runs_root
		)
		live_audits[source.candidate_id] = audit_candidate_source(
			candidate=source,
			candidate_model=config.model_by_id(source.candidate_id),
			canonical_config=canonical,
			reference_base_checkpoint=settings.reference_base_checkpoint,
			reference_final_checkpoint=settings.reference_final_checkpoint,
			protocol_lock_identity=protocol_identity,
			selection_lock_identity=selection_identity,
		)
	for source in settings.sources:
		config = _candidate_config(
			canonical, candidate=source, runs_root=settings.runs_root
		)
		for layout_id in LAYOUT_IDS:
			job = five_way_runner.resolve_f3_lithology_five_way_job(
				config,
				model=source.candidate_id,
				layout=layout_id,
				size='medium',
			)
			row = _read_candidate_job_evidence(
				job=job,
				candidate=source,
				expected_source_audit=live_audits[source.candidate_id],
				expected_validation_order=medium_order,
				verify_evaluation_identity=True,
			)
			inputs.append(row)
			scores[source.candidate_id][layout_id] = cast('float', row['macro_f1'])
	for layout_id in LAYOUT_IDS:
		row = cast(
			'dict[str, object]',
			_shared('_read_random_job_evidence')(
				canonical, layout_id=layout_id, data_size='medium'
			),
		)
		inputs.append(row)
		scores['random'][layout_id] = cast('float', row['macro_f1'])
	wins = _medium_5of5_wins(scores)
	gate_open = any(wins.values())
	if require_open and not gate_open:
		raise ValueError(
			'small/large validation is forbidden because neither base1 arm has '
			'five positive medium deltas over random'
		)
	return {
		'gate_open': gate_open,
		'layout_ids': list(LAYOUT_IDS),
		'selected_candidate_id': SELECTED_ID,
		'legacy_candidate_id': LEGACY_ID,
		'selected_wins_over_random': wins[SELECTED_ID],
		'legacy_wins_over_random': wins[LEGACY_ID],
		'selected_deltas': {
			layout: scores[SELECTED_ID][layout] - scores['random'][layout]
			for layout in LAYOUT_IDS
		},
		'legacy_deltas': {
			layout: scores[LEGACY_ID][layout] - scores['random'][layout]
			for layout in LAYOUT_IDS
		},
		'inputs': inputs,
	}


def _medium_5of5_wins(
	scores: Mapping[str, Mapping[str, float]],
) -> dict[str, bool]:
	if set(scores) != {SELECTED_ID, LEGACY_ID, 'random'}:
		raise ValueError('medium scores must define exactly both arms and random')
	for source_id, values in scores.items():
		if set(values) != set(LAYOUT_IDS):
			raise ValueError(f'{source_id} must define all five medium layouts')
	return {
		source_id: all(
			scores[source_id][layout] > scores['random'][layout]
			for layout in LAYOUT_IDS
		)
		for source_id in EXPECTED_SOURCE_IDS
	}


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
					f'{job.output_dir} prediction {key} differs from source audit'
				)
	macro_f1 = _macro_f1(metrics, job.metrics_path)
	return {
		'candidate_id': candidate.candidate_id,
		'layout_id': job.layout_id,
		'data_size': job.data_size,
		'macro_f1': macro_f1,
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


def create_base1_final_result(
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	*,
	created_at_utc: str | None = None,
) -> dict[str, object]:
	"""Replay the exact reached branch and exclusively publish its decision."""
	if settings.final_result.exists() or settings.final_result.is_symlink():
		raise FileExistsError(f'final result already exists: {settings.final_result}')
	protocol = validate_base1_protocol_lock(settings, canonical)
	selection = validate_base1_selection_lock(settings, canonical)
	protocol_identity = _protocol_identity(settings, protocol)
	selection_identity = _selection_identity(settings, selection)
	parent = validate_parent_result(settings, canonical)
	benchmark = cast(
		'Mapping[str, object]',
		_shared('_validate_benchmark_provenance')(
			settings, canonical, verify_files=True
		),
	)
	five_way_sources.audit_f3_lithology_five_way_sources(canonical)
	gate = _validate_medium_random_gate(
		settings=settings,
		canonical=canonical,
		selection_lock=selection,
		require_open=False,
	)
	gate_open = cast('bool', gate['gate_open'])
	expected_cells = _expected_base1_cells(medium_gate_open=gate_open)
	_validate_exact_candidate_cell_set(
		settings=settings, canonical=canonical, expected_cells=expected_cells
	)
	medium_order = _validation_order_provenance(
		settings,
		{
			'protocol_lock': protocol,
			'selection_lock': selection,
			'medium_gate': None,
		},
	)
	post_gate_order = _validation_order_provenance(
		settings,
		{
			'protocol_lock': protocol,
			'selection_lock': selection,
			'medium_gate': gate,
		},
	)
	live_audits: dict[str, Mapping[str, object]] = {}
	for source in settings.sources:
		config = _candidate_config(
			canonical, candidate=source, runs_root=settings.runs_root
		)
		live_audits[source.candidate_id] = audit_candidate_source(
			candidate=source,
			candidate_model=config.model_by_id(source.candidate_id),
			canonical_config=canonical,
			reference_base_checkpoint=settings.reference_base_checkpoint,
			reference_final_checkpoint=settings.reference_final_checkpoint,
			protocol_lock_identity=protocol_identity,
			selection_lock_identity=selection_identity,
		)
	candidate_inputs: list[dict[str, object]] = []
	for source_id, layout_id, data_size in sorted(expected_cells):
		source = settings.source_by_id(source_id)
		config = _candidate_config(
			canonical, candidate=source, runs_root=settings.runs_root
		)
		job = five_way_runner.resolve_f3_lithology_five_way_job(
			config, model=source_id, layout=layout_id, size=data_size
		)
		candidate_inputs.append(
			_read_candidate_job_evidence(
				job=job,
				candidate=source,
				expected_source_audit=live_audits[source_id],
				expected_validation_order=(
					medium_order if data_size == 'medium' else post_gate_order
				),
				verify_evaluation_identity=True,
			)
		)
	reached_sizes = DATA_SIZES if gate_open else ('medium',)
	random_inputs = [
		cast(
			'dict[str, object]',
			_shared('_read_random_job_evidence')(
				canonical, layout_id=layout_id, data_size=data_size
			),
		)
		for data_size in reached_sizes
		for layout_id in LAYOUT_IDS
	]
	_assert_unique_evidence_rows(candidate_inputs, include_candidate=True)
	_assert_unique_evidence_rows(random_inputs, include_candidate=False)
	scores = _scores_from_evidence((*candidate_inputs, *random_inputs))
	selected_result = _arm_random_result(
		scores, arm_id=SELECTED_ID, medium_gate_open=gate_open
	)
	legacy_result = _arm_random_result(
		scores, arm_id=LEGACY_ID, medium_gate_open=gate_open
	)
	attribution = _paired_arm_contrast(
		scores,
		left_id=SELECTED_ID,
		right_id=LEGACY_ID,
		full_branch_reached=gate_open,
	)
	winner = _choose_base1_winner(
		selected_passed=cast('bool', selected_result['wins_all_15_over_random']),
		legacy_passed=cast('bool', legacy_result['wins_all_15_over_random']),
		attribution_passed=cast('bool', attribution['wins_all_15']),
	)
	passed = winner is not None
	payload = {
		'schema_version': FINAL_RESULT_SCHEMA_VERSION,
		'final_result_type': FINAL_RESULT_TYPE,
		'validation_only': True,
		'base_pretraining_epochs': BASE_PRETRAINING_EPOCHS,
		'continuation_epochs': CONTINUATION_EPOCHS,
		'parent_result': parent,
		'protocol_lock': protocol_identity,
		'selection_lock': selection_identity,
		'benchmark_provenance': benchmark,
		'repository_state': selection['repository_state'],
		'medium_gate': {
			'gate_open': gate_open,
			'selected_wins_over_random': gate['selected_wins_over_random'],
			'legacy_wins_over_random': gate['legacy_wins_over_random'],
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
			SELECTED_ID: selected_result,
			LEGACY_ID: legacy_result,
		},
		'gaussian_attribution': attribution,
		'passed': passed,
		'winner_candidate_id': winner,
		'authorizes_next_base_duration': False,
		'authorized_next_base_pretraining_epochs': None,
		'failure_stage': (
			None if passed else ('final_15of15' if gate_open else 'medium_5of5')
		),
		'created_at_utc': created_at_utc or _utc_timestamp(),
	}
	_validate_utc_timestamp(cast('str', payload['created_at_utc']))
	_write_exclusive_json(settings.final_result, payload)
	return payload


def _expected_base1_cells(*, medium_gate_open: bool) -> set[tuple[str, str, str]]:
	sizes = DATA_SIZES if medium_gate_open else ('medium',)
	return {
		(source_id, layout_id, data_size)
		for source_id in EXPECTED_SOURCE_IDS
		for layout_id in LAYOUT_IDS
		for data_size in sizes
	}


def _validate_exact_candidate_cell_set(
	*,
	settings: ValidationSettings,
	canonical: F3FiveWayConfig,
	expected_cells: set[tuple[str, str, str]],
) -> None:
	for source in settings.sources:
		config = _candidate_config(
			canonical, candidate=source, runs_root=settings.runs_root
		)
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZES:
				job = five_way_runner.resolve_f3_lithology_five_way_job(
					config,
					model=source.candidate_id,
					layout=layout_id,
					size=data_size,
				)
				expected = (
					source.candidate_id,
					layout_id,
					data_size,
				) in expected_cells
				for path in (job.metrics_path, job.output_dir / AUDIT_NAME):
					if path.is_symlink():
						raise ValueError(f'validation evidence is a symlink: {path}')
					if expected and not path.is_file():
						raise FileNotFoundError(
							f'missing expected base1 validation evidence: {path}'
						)
					if not expected and path.exists():
						raise ValueError(
							'base1 validation evidence exists outside reached set: '
							f'{path}'
						)


def _scores_from_evidence(
	rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[tuple[str, str], float]]:
	result: dict[str, dict[tuple[str, str], float]] = {}
	for row in rows:
		source_id = _string(row.get('candidate_id'), 'evidence candidate_id')
		layout_id = _string(row.get('layout_id'), 'evidence layout_id')
		data_size = _string(row.get('data_size'), 'evidence data_size')
		value = row.get('macro_f1')
		if not isinstance(value, int | float) or isinstance(value, bool):
			raise TypeError('evidence macro_f1 must be numeric')
		cell = (layout_id, data_size)
		if cell in result.setdefault(source_id, {}):
			raise ValueError('validation evidence duplicates a source cell')
		result[source_id][cell] = float(value)
	return result


def _arm_random_result(
	scores: Mapping[str, Mapping[tuple[str, str], float]],
	*,
	arm_id: str,
	medium_gate_open: bool,
) -> dict[str, object]:
	return cast(
		'dict[str, object]',
		_shared('_arm_random_result')(
			scores, arm_id=arm_id, medium_gate_open=medium_gate_open
		),
	)


def _paired_arm_contrast(
	scores: Mapping[str, Mapping[tuple[str, str], float]],
	*,
	left_id: str,
	right_id: str,
	full_branch_reached: bool,
) -> dict[str, object]:
	return cast(
		'dict[str, object]',
		_shared('_paired_arm_contrast')(
			scores,
			left_id=left_id,
			right_id=right_id,
			contrast_id='selected_gaussian_minus_matched_legacy',
			full_branch_reached=full_branch_reached,
		),
	)


def _choose_base1_winner(
	*,
	selected_passed: bool,
	legacy_passed: bool,
	attribution_passed: bool,
) -> str | None:
	if selected_passed and legacy_passed:
		return SELECTED_ID if attribution_passed else LEGACY_ID
	if selected_passed:
		return SELECTED_ID
	if legacy_passed:
		return LEGACY_ID
	return None


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
	if path.is_symlink():
		raise ValueError(f'{label} must not be a symlink: {path}')
	if not path.is_file():
		raise FileNotFoundError(f'missing {label}: {path}')
	raw = path.read_bytes()
	try:
		value = json.loads(raw)
	except json.JSONDecodeError as error:
		raise ValueError(f'{label} is not valid JSON: {path}') from error
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must contain a JSON object: {path}')
	return value, hashlib.sha256(raw).hexdigest()


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


def _type_sensitive_equal(actual: object, expected: object) -> bool:
	"""Compare replayed scientific evidence without numeric type coercion."""
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
	"""Build the duration-specific validation CLI."""
	parser = argparse.ArgumentParser(
		description='Run one validation-only one-base-epoch F3 job.'
	)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--candidate', choices=EXPECTED_SOURCE_IDS)
	parser.add_argument('--layout', choices=LAYOUT_IDS)
	parser.add_argument('--size', choices=DATA_SIZES)
	parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
	parser.add_argument('--resume', type=Path)
	parser.add_argument('--dry-run', action='store_true')
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument('--audit-parent-only', action='store_true')
	mode.add_argument('--audit-base-checkpoint-only', action='store_true')
	mode.add_argument('--create-protocol-lock', action='store_true')
	mode.add_argument('--create-selection-lock', action='store_true')
	mode.add_argument('--audit-checkpoint-only', action='store_true')
	mode.add_argument('--create-final-result', action='store_true')
	return parser


def _validate_cli_arguments(args: argparse.Namespace) -> None:  # noqa: C901
	checkpoint_mode = args.audit_base_checkpoint_only or args.audit_checkpoint_only
	if checkpoint_mode:
		if args.candidate is None:
			raise ValueError('checkpoint audit modes require --candidate')
		if any(value is not None for value in (args.layout, args.size, args.resume)):
			raise ValueError(
				'checkpoint audit modes do not accept --layout, --size, or --resume'
			)
		if args.dry_run:
			raise ValueError('checkpoint audit modes are already read-only')
		return
	if args.audit_parent_only:
		if any(
			value is not None
			for value in (args.candidate, args.layout, args.size, args.resume)
		):
			raise ValueError('--audit-parent-only does not accept job arguments')
		if args.dry_run:
			raise ValueError('--audit-parent-only is already read-only')
		return
	if (
		args.create_protocol_lock
		or args.create_selection_lock
		or args.create_final_result
	):
		if any(
			value is not None
			for value in (args.candidate, args.layout, args.size, args.resume)
		):
			raise ValueError('lock/result creation modes do not accept job arguments')
		if args.dry_run:
			raise ValueError('lock/result creation modes do not support --dry-run')
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


def _print_mapping(payload: Mapping[str, object]) -> None:
	for key, value in payload.items():
		print(f'{key}: {value}')


def main() -> None:  # noqa: PLR0911
	"""Audit, lock, or run exactly one validation-only branch operation."""
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
				candidate=settings.source_by_id(args.candidate),
				canonical_config=canonical,
				reference_base_checkpoint=settings.reference_base_checkpoint,
			)
		)
		return
	if args.create_protocol_lock:
		_print_mapping(create_base1_protocol_lock(settings, canonical))
		return
	if args.create_selection_lock:
		_print_mapping(create_base1_selection_lock(settings, canonical))
		return
	protocol = validate_base1_protocol_lock(settings, canonical)
	selection = validate_base1_selection_lock(settings, canonical)
	protocol_identity = _protocol_identity(settings, protocol)
	selection_identity = _selection_identity(settings, selection)
	if args.audit_checkpoint_only:
		result = audit_candidate_checkpoints(
			candidate=settings.source_by_id(args.candidate),
			canonical_config=canonical,
			reference_base_checkpoint=settings.reference_base_checkpoint,
			reference_final_checkpoint=settings.reference_final_checkpoint,
		)
		_print_mapping(
			{
				**result,
				'protocol_lock': protocol_identity,
				'selection_lock': selection_identity,
			}
		)
		return
	if args.create_final_result:
		_print_mapping(create_base1_final_result(settings, canonical))
		return
	candidate = settings.source_by_id(args.candidate)
	order = enforce_validation_order(
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
				'selection_lock': selection_identity,
				'execution': 'dry-run; no files written',
			}
		)
		_print_mapping(summary)
		return
	five_way_sources.audit_f3_lithology_five_way_sources(canonical)
	audit = audit_candidate_source(
		candidate=candidate,
		candidate_model=job.model,
		canonical_config=canonical,
		reference_base_checkpoint=settings.reference_base_checkpoint,
		reference_final_checkpoint=settings.reference_final_checkpoint,
		protocol_lock_identity=protocol_identity,
		selection_lock_identity=selection_identity,
	)
	audit = {
		**audit,
		'validation_order_provenance': _validation_order_provenance(settings, order),
	}
	_print_mapping(_run_job(job, audit=audit, device=args.device, resume=args.resume))


if __name__ == '__main__':
	main()
