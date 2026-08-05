"""Closed configuration for the periodic-refresh original-split screen."""
# ruff: noqa: C901, CPY001, E501

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_center_trace_masked import (
	_validate_hard_decoder_contract,
)
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
	F3VoxelLabelBudgetMultiHeadConfig,
	config_from_mapping_for_candidates,
	f3_lithology_voxel_label_budget_multi_head_config_from_mapping,
)
from seis_ssl_cluster.paths import ensure_under_root

PERIODIC_REFRESH_MODEL_ID = 'mh_ctmask010_refresh3ep_hmm2_nocons'
PERIODIC_REFRESH_MODEL_TAG = (
	'strat_hmm_pretext_mh_k6810_ctmask010_refresh3ep_hmm2_nocons_topblock1_distill_v1'
)
PERIODIC_REFRESH_VARIANT = 'ctmask010_refresh3ep_hmm2_nocons'
EXPECTED_CANDIDATES = ((PERIODIC_REFRESH_MODEL_ID, PERIODIC_REFRESH_MODEL_TAG),)
_CONFIG_KEYS = frozenset(
	{
		'multi_head',
		'hard_multi_head_config',
		'center_trace_masked_run_manifest',
		'periodic_refresh_handoff',
		'screening_audit',
	}
)


@dataclass(frozen=True)
class F3VoxelLabelBudgetCenterTraceMaskedPeriodicRefreshConfig:
	"""One closed 15-job periodic-refresh decoder screen."""

	multi_head: F3VoxelLabelBudgetMultiHeadConfig
	hard_multi_head_config: Path
	center_trace_masked_run_manifest: Path
	periodic_refresh_handoff: Path
	screening_audit: Path
	screening_audit_payload: Mapping[str, object] | None = None

	def __getattr__(self, name: str) -> object:
		"""Delegate the fixed decoder settings to the isolated matrix config."""
		return getattr(self.multi_head, name)

	@property
	def base(self) -> object:
		"""Expose the shared decoder configuration used by stage helpers."""
		return self.multi_head.base

	@property
	def candidates(self) -> tuple[object, ...]:
		"""Expose only the periodic-refresh candidate."""
		return self.multi_head.candidates

	@property
	def run_manifest_name(self) -> str:
		"""Return the periodic candidate-owned manifest filename."""
		return 'periodic_refresh_original_job_manifest.json'

	@property
	def run_manifest_type(self) -> str:
		"""Return the isolated periodic candidate manifest type."""
		return 'f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh'


def f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh_config_from_mapping(
	config: Mapping[str, object],
) -> F3VoxelLabelBudgetCenterTraceMaskedPeriodicRefreshConfig:
	"""Resolve the exact periodic-refresh 3-by-5 candidate matrix."""
	if not isinstance(config, Mapping):
		raise TypeError('periodic-refresh config must be a mapping')
	unknown = set(config) - _CONFIG_KEYS
	missing = _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(f'unknown periodic-refresh config keys: {sorted(unknown)!r}')
	if missing:
		raise ValueError(f'missing periodic-refresh config keys: {sorted(missing)!r}')
	multi_head_raw = config['multi_head']
	if not isinstance(multi_head_raw, Mapping):
		raise TypeError('multi_head must be a mapping')
	multi_head = config_from_mapping_for_candidates(
		multi_head_raw,
		expected_candidates=EXPECTED_CANDIDATES,
	)
	if multi_head.job_count != 15:
		raise ValueError('periodic-refresh screen must contain exactly 15 jobs')
	if (
		len(multi_head.candidates) != 1
		or multi_head.candidates[0].model_id != PERIODIC_REFRESH_MODEL_ID
		or multi_head.candidates[0].model_tag != PERIODIC_REFRESH_MODEL_TAG
	):
		raise ValueError('periodic-refresh candidate identity mismatch')
	_validate_fixed_candidate_paths(multi_head)
	hard_multi_head_config = _required_existing_path(
		config['hard_multi_head_config'], 'hard_multi_head_config'
	)
	hard = f3_lithology_voxel_label_budget_multi_head_config_from_mapping(
		load_config(hard_multi_head_config)
	)
	_validate_hard_decoder_contract(multi_head, hard)
	center_trace_masked_run_manifest = _required_existing_path(
		config['center_trace_masked_run_manifest'],
		'center_trace_masked_run_manifest',
	)
	if center_trace_masked_run_manifest.name != 'center_trace_masked_job_manifest.json':
		raise ValueError('center-trace reference manifest name is not canonical')
	expected_center_manifest = (
		multi_head.artifact_root
		/ 'lithology/f3/facies_benchmark_v1'
		/ 'voxel_label_budget_center_trace_masked_k6810_v1'
		/ 'original_split/reports/center_trace_masked_job_manifest.json'
	)
	if center_trace_masked_run_manifest != expected_center_manifest:
		raise ValueError('center-trace reference manifest path is not canonical')
	ensure_under_root(
		center_trace_masked_run_manifest,
		root=multi_head.artifact_root,
		label='center_trace_masked_run_manifest',
	)
	periodic_refresh_handoff = _required_existing_path(
		config['periodic_refresh_handoff'], 'periodic_refresh_handoff'
	)
	if periodic_refresh_handoff.name != 'periodic_refresh_handoff.json':
		raise ValueError('periodic_refresh_handoff name is not canonical')
	if periodic_refresh_handoff != multi_head.candidates[0].pretraining_handoff:
		raise ValueError('periodic-refresh handoff is not the candidate handoff')
	ensure_under_root(
		periodic_refresh_handoff,
		root=multi_head.artifact_root,
		label='periodic_refresh_handoff',
	)
	screening_audit = _required_existing_path(
		config['screening_audit'], 'screening_audit'
	)
	if screening_audit != _expected_screening_audit_path(multi_head.artifact_root):
		raise ValueError(
			'screening_audit must use the canonical periodic-refresh preflight path'
		)
	ensure_under_root(
		screening_audit,
		root=multi_head.artifact_root,
		label='screening_audit',
	)
	audit_payload = _validate_screening_audit_binding(screening_audit, multi_head)
	ensure_under_root(
		multi_head.output_root,
		root=multi_head.artifact_root,
		label='outputs.output_root',
	)
	return F3VoxelLabelBudgetCenterTraceMaskedPeriodicRefreshConfig(
		multi_head=multi_head,
		hard_multi_head_config=hard_multi_head_config,
		center_trace_masked_run_manifest=center_trace_masked_run_manifest,
		periodic_refresh_handoff=periodic_refresh_handoff,
		screening_audit=screening_audit,
		screening_audit_payload=audit_payload,
	)


def load_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh_config(
	path: str | Path,
) -> F3VoxelLabelBudgetCenterTraceMaskedPeriodicRefreshConfig:
	"""Load and resolve a periodic-refresh decoder YAML file."""
	return f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh_config_from_mapping(
		load_config(path)
	)


def _validate_fixed_candidate_paths(
	multi_head: F3VoxelLabelBudgetMultiHeadConfig,
) -> None:
	"""Keep periodic artifacts separate from fixed center-trace artifacts."""
	candidate = multi_head.candidates[0]
	root = multi_head.artifact_root
	expected_embeddings = (
		root
		/ 'embeddings/f3/facies_benchmark_v1'
		/ PERIODIC_REFRESH_MODEL_TAG
		/ 'overlap_x16'
	)
	expected_handoff = (
		root
		/ 'pretraining/f3/facies_benchmark_v1'
		/ PERIODIC_REFRESH_MODEL_TAG
		/ 'preflight/periodic_refresh_handoff.json'
	)
	expected_output = (
		root
		/ 'lithology/f3/facies_benchmark_v1'
		/ 'voxel_label_budget_center_trace_masked_periodic_refresh_k6810_v1'
		/ 'original_split'
	)
	if candidate.embeddings_dir != expected_embeddings:
		raise ValueError(
			'periodic-refresh embeddings_dir must use the canonical artifact'
		)
	if candidate.pretraining_handoff != expected_handoff:
		raise ValueError(
			'periodic-refresh pretraining_handoff must use the canonical artifact'
		)
	if multi_head.output_root != expected_output:
		raise ValueError('periodic-refresh output root must be candidate-owned')
	if candidate.pretraining_handoff.name != 'periodic_refresh_handoff.json':
		raise ValueError('periodic-refresh handoff name is not canonical')


def _expected_screening_audit_path(artifact_root: Path) -> Path:
	return (
		artifact_root
		/ 'lithology/f3/facies_benchmark_v1'
		/ 'voxel_label_budget_center_trace_masked_periodic_refresh_k6810_v1'
		/ 'original_split/preflight/periodic_refresh_screening_audit.json'
	)


def validate_f3_center_trace_masked_periodic_refresh_screening_audit(
	config: F3VoxelLabelBudgetCenterTraceMaskedPeriodicRefreshConfig,
) -> Mapping[str, object]:
	"""Live-revalidate the immutable periodic audit and candidate binding."""
	return _validate_screening_audit_binding(config.screening_audit, config.multi_head)


def _required_existing_path(value: object, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty path string')
	path = Path(value).resolve()
	if not path.is_file():
		raise FileNotFoundError(path)
	return path


def _validate_screening_audit_binding(
	path: Path,
	multi_head: F3VoxelLabelBudgetMultiHeadConfig,
) -> Mapping[str, object]:
	"""Require a PASS audit bound to the periodic candidate artifacts."""
	audit = importlib.import_module(
		'seis_ssl_cluster.f3.center_trace_masked_periodic_refresh_screening_audit'
	)
	payload = audit.load_f3_center_trace_masked_periodic_refresh_screening_audit(path)
	candidate = multi_head.candidates[0]
	audit.validate_f3_center_trace_masked_periodic_refresh_screening_audit_binding(
		payload,
		model_id=candidate.model_id,
		model_tag=candidate.model_tag,
		pretraining_handoff=candidate.pretraining_handoff,
		embeddings_dir=candidate.embeddings_dir,
	)
	return payload


config_from_mapping = f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh_config_from_mapping


__all__ = [
	'EXPECTED_CANDIDATES',
	'PERIODIC_REFRESH_MODEL_ID',
	'PERIODIC_REFRESH_MODEL_TAG',
	'PERIODIC_REFRESH_VARIANT',
	'F3VoxelLabelBudgetCenterTraceMaskedPeriodicRefreshConfig',
	'config_from_mapping',
	'f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh_config_from_mapping',
	'load_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh_config',
	'validate_f3_center_trace_masked_periodic_refresh_screening_audit',
]
