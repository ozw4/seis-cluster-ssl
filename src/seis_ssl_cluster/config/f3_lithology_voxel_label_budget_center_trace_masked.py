"""Closed configuration for center-trace masked original-split screening."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
	F3VoxelLabelBudgetMultiHeadConfig,
	config_from_mapping_for_candidates,
	f3_lithology_voxel_label_budget_multi_head_config_from_mapping,
)

CENTER_TRACE_MASKED_MODEL_ID = 'mh_ctmask010_nocons'
CENTER_TRACE_MASKED_MODEL_TAG = (
	'strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1'
)
EXPECTED_CANDIDATES = ((CENTER_TRACE_MASKED_MODEL_ID, CENTER_TRACE_MASKED_MODEL_TAG),)
_CONFIG_KEYS = frozenset({'multi_head', 'hard_multi_head_config', 'screening_audit'})


@dataclass(frozen=True)
class F3VoxelLabelBudgetCenterTraceMaskedConfig:
	"""One closed 15-job center-trace masked decoder screen."""

	multi_head: F3VoxelLabelBudgetMultiHeadConfig
	hard_multi_head_config: Path
	screening_audit: Path
	screening_audit_payload: Mapping[str, object] | None = None

	def __getattr__(self, name: str) -> object:
		"""Delegate common decoder settings to the isolated matrix config."""
		return getattr(self.multi_head, name)

	@property
	def base(self) -> object:
		"""Expose the shared control configuration used by result helpers."""
		return self.multi_head.base

	@property
	def candidates(self) -> tuple[object, ...]:
		"""Expose only the center-trace masked candidate."""
		return self.multi_head.candidates

	@property
	def run_manifest_name(self) -> str:
		"""Return the candidate-owned manifest filename."""
		return 'center_trace_masked_job_manifest.json'

	@property
	def run_manifest_type(self) -> str:
		"""Return the isolated candidate manifest artifact type."""
		return 'f3_lithology_voxel_label_budget_center_trace_masked'


def f3_lithology_voxel_label_budget_center_trace_masked_config_from_mapping(
	config: Mapping[str, object],
) -> F3VoxelLabelBudgetCenterTraceMaskedConfig:
	"""Resolve the exact center-trace masked 3-by-5 candidate matrix."""
	if not isinstance(config, Mapping):
		raise TypeError('center-trace masked config must be a mapping')
	unknown = set(config) - _CONFIG_KEYS
	missing = _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(
			f'unknown center-trace masked config keys: {sorted(unknown)!r}'
		)
	if missing:
		raise ValueError(
			f'missing center-trace masked config keys: {sorted(missing)!r}'
		)
	multi_head_raw = config['multi_head']
	if not isinstance(multi_head_raw, Mapping):
		raise TypeError('multi_head must be a mapping')
	multi_head = config_from_mapping_for_candidates(
		multi_head_raw,
		expected_candidates=EXPECTED_CANDIDATES,
	)
	if multi_head.job_count != 15:
		raise ValueError('center-trace masked screen must contain exactly 15 jobs')
	if (
		len(multi_head.candidates) != 1
		or multi_head.candidates[0].model_id != CENTER_TRACE_MASKED_MODEL_ID
		or multi_head.candidates[0].model_tag != CENTER_TRACE_MASKED_MODEL_TAG
	):
		raise ValueError('center-trace masked candidate identity mismatch')
	_validate_fixed_candidate_paths(multi_head)
	hard_multi_head_config = _required_existing_path(
		config['hard_multi_head_config'], 'hard_multi_head_config'
	)
	hard = f3_lithology_voxel_label_budget_multi_head_config_from_mapping(
		load_config(hard_multi_head_config)
	)
	_validate_hard_decoder_contract(multi_head, hard)
	screening_audit = _required_existing_path(
		config['screening_audit'], 'screening_audit'
	)
	if screening_audit != _expected_screening_audit_path(multi_head.artifact_root):
		raise ValueError(
			'screening_audit must use the canonical center-trace preflight path'
		)
	audit_payload = _validate_screening_audit_binding(screening_audit, multi_head)
	return F3VoxelLabelBudgetCenterTraceMaskedConfig(
		multi_head=multi_head,
		hard_multi_head_config=hard_multi_head_config,
		screening_audit=screening_audit,
		screening_audit_payload=audit_payload,
	)


def load_f3_lithology_voxel_label_budget_center_trace_masked_config(
	path: str | Path,
) -> F3VoxelLabelBudgetCenterTraceMaskedConfig:
	"""Load and resolve a center-trace masked decoder YAML file."""
	return f3_lithology_voxel_label_budget_center_trace_masked_config_from_mapping(
		load_config(path)
	)


def _validate_fixed_candidate_paths(
	multi_head: F3VoxelLabelBudgetMultiHeadConfig,
) -> None:
	candidate = multi_head.candidates[0]
	root = multi_head.artifact_root
	expected_embeddings = (
		root
		/ 'embeddings/f3/facies_benchmark_v1'
		/ CENTER_TRACE_MASKED_MODEL_TAG
		/ 'overlap_x16'
	)
	expected_handoff = (
		root
		/ 'pretraining/f3/facies_benchmark_v1'
		/ CENTER_TRACE_MASKED_MODEL_TAG
		/ 'preflight/center_trace_masked_handoff.json'
	)
	expected_output = (
		root
		/ 'lithology/f3/facies_benchmark_v1'
		/ 'voxel_label_budget_center_trace_masked_k6810_v1/original_split'
	)
	if candidate.embeddings_dir != expected_embeddings:
		raise ValueError(
			'center-trace masked embeddings_dir must use the canonical artifact'
		)
	if candidate.pretraining_handoff != expected_handoff:
		raise ValueError(
			'center-trace masked pretraining_handoff must use the canonical artifact'
		)
	if multi_head.output_root != expected_output:
		raise ValueError('center-trace masked output root must be candidate-owned')


def _expected_screening_audit_path(artifact_root: Path) -> Path:
	return (
		artifact_root
		/ 'lithology/f3/facies_benchmark_v1'
		/ 'voxel_label_budget_center_trace_masked_k6810_v1'
		/ 'original_split/preflight/center_trace_masked_screening_audit.json'
	)


def validate_f3_center_trace_masked_screening_audit(
	config: F3VoxelLabelBudgetCenterTraceMaskedConfig,
) -> Mapping[str, object]:
	"""Live-revalidate the immutable audit and its candidate binding."""
	return _validate_screening_audit_binding(config.screening_audit, config.multi_head)


def _required_existing_path(value: object, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty path string')
	path = Path(value).resolve()
	if not path.is_file():
		raise FileNotFoundError(path)
	return path


def _validate_hard_decoder_contract(
	candidate: F3VoxelLabelBudgetMultiHeadConfig,
	hard: F3VoxelLabelBudgetMultiHeadConfig,
) -> None:
	"""Keep all decoder, data, and evaluation settings paired to the hard run."""
	for name in (
		'dataset_manifest',
		'multi_head_target_manifest',
		'original_run_manifest',
		'current_k6_run_manifest',
	):
		if getattr(candidate, name) != getattr(hard, name):
			raise ValueError(f'center-trace masked decoder contract mismatch: {name}')
	if (
		candidate.references.mae_model_id != hard.references.mae_model_id
		or candidate.references.current_k6_model_id
		!= hard.references.current_k6_model_id
	):
		raise ValueError(
			'center-trace masked decoder contract mismatch: reference model IDs'
		)
	for name in (
		'artifact_root',
		'f3_root',
		'reports_root',
		'dataset',
		'budgets',
		'subsample_seeds',
		'base_seed',
		'add_subsample_seed',
		'labels',
		'decoder',
		'tiles',
		'train',
		'write_probabilities',
		'evaluation',
		'overwrite',
	):
		if getattr(candidate.base, name) != getattr(hard.base, name):
			raise ValueError(f'center-trace masked decoder contract mismatch: {name}')


def _validate_screening_audit_binding(
	path: Path,
	multi_head: F3VoxelLabelBudgetMultiHeadConfig,
) -> Mapping[str, object]:
	"""Require a PASS audit for the exact live center-trace artifacts."""
	audit = importlib.import_module(
		'seis_ssl_cluster.f3.center_trace_masked_screening_audit'
	)
	payload = audit.load_f3_center_trace_masked_screening_audit(path)
	candidate = multi_head.candidates[0]
	audit.validate_f3_center_trace_masked_screening_audit_binding(
		payload,
		model_id=candidate.model_id,
		model_tag=candidate.model_tag,
		pretraining_handoff=candidate.pretraining_handoff,
		embeddings_dir=candidate.embeddings_dir,
	)
	return payload


config_from_mapping = (
	f3_lithology_voxel_label_budget_center_trace_masked_config_from_mapping
)


__all__ = [
	'CENTER_TRACE_MASKED_MODEL_ID',
	'CENTER_TRACE_MASKED_MODEL_TAG',
	'EXPECTED_CANDIDATES',
	'F3VoxelLabelBudgetCenterTraceMaskedConfig',
	'config_from_mapping',
	'f3_lithology_voxel_label_budget_center_trace_masked_config_from_mapping',
	'load_f3_lithology_voxel_label_budget_center_trace_masked_config',
	'validate_f3_center_trace_masked_screening_audit',
]
