"""Closed XY-neighbour-consensus configuration for original-split screening."""

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

XY_MODEL_ID = 'mh_xycons1_nocons'
XY_MODEL_TAG = (
	'strat_hmm_pretext_mh_k6810_xycons1_nocons_topblock1_distill_v1'
)
EXPECTED_CANDIDATES = ((XY_MODEL_ID, XY_MODEL_TAG),)
_CONFIG_KEYS = frozenset(
	{'multi_head', 'hard_multi_head_config', 'screening_audit'}
)


@dataclass(frozen=True)
class F3VoxelLabelBudgetXYNeighborConsensusConfig:
	"""One schema-v5 XY candidate with a frozen hard-decoder contract."""

	multi_head: F3VoxelLabelBudgetMultiHeadConfig
	hard_multi_head_config: Path
	screening_audit: Path

	def __getattr__(self, name: str) -> object:
		"""Delegate common decoder settings to the shared multi-head config."""
		return getattr(self.multi_head, name)

	@property
	def base(self) -> object:
		"""Expose the shared control configuration expected by job helpers."""
		return self.multi_head.base

	@property
	def candidates(self) -> tuple[object, ...]:
		"""Expose the one canonical XY-neighbour consensus candidate."""
		return self.multi_head.candidates

	@property
	def run_manifest_name(self) -> str:
		"""Return the candidate-owned decoder manifest name."""
		return 'xy_neighbor_consensus_job_manifest.json'

	@property
	def run_manifest_type(self) -> str:
		"""Return the distinct artifact type for this candidate matrix."""
		return 'f3_lithology_voxel_label_budget_xy_neighbor_consensus'


def f3_lithology_voxel_label_budget_xy_neighbor_consensus_config_from_mapping(
	config: Mapping[str, object],
) -> F3VoxelLabelBudgetXYNeighborConsensusConfig:
	"""Resolve the closed 3-budget by 5-seed XY-consensus screening matrix."""
	if not isinstance(config, Mapping):
		raise TypeError('XY-neighbour-consensus config must be a mapping')
	unknown = set(config) - _CONFIG_KEYS
	missing = _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(
			'unknown XY-neighbour-consensus config keys: '
			f'{sorted(unknown)!r}'
		)
	if missing:
		raise ValueError(
			'missing XY-neighbour-consensus config keys: '
			f'{sorted(missing)!r}'
		)
	multi_head_raw = config['multi_head']
	if not isinstance(multi_head_raw, Mapping):
		raise TypeError('multi_head must be a mapping')
	multi_head = config_from_mapping_for_candidates(
		multi_head_raw,
		expected_candidates=EXPECTED_CANDIDATES,
	)
	if multi_head.job_count != 15:
		raise ValueError('XY-neighbour-consensus screen must contain exactly 15 jobs')
	if (
		len(multi_head.candidates) != 1
		or multi_head.candidates[0].model_id != XY_MODEL_ID
		or multi_head.candidates[0].model_tag != XY_MODEL_TAG
	):
		raise ValueError('XY-neighbour-consensus candidate identity mismatch')
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
			'screening_audit must use the canonical candidate preflight path'
		)
	_validate_screening_audit_binding(screening_audit, multi_head)
	return F3VoxelLabelBudgetXYNeighborConsensusConfig(
		multi_head=multi_head,
		hard_multi_head_config=hard_multi_head_config,
		screening_audit=screening_audit,
	)


def _validate_fixed_candidate_paths(
	multi_head: F3VoxelLabelBudgetMultiHeadConfig,
) -> None:
	"""Keep the closed schema-v5 candidate from being redirected to another run."""
	candidate = multi_head.candidates[0]
	root = multi_head.artifact_root
	expected_embeddings = (
		root
		/ 'embeddings/f3/facies_benchmark_v1'
		/ XY_MODEL_TAG
		/ 'overlap_x16'
	)
	expected_handoff = (
		root
		/ 'pretraining/f3/facies_benchmark_v1'
		/ XY_MODEL_TAG
		/ 'preflight/xy_neighbor_consensus_handoff.json'
	)
	expected_output = (
		root
		/ 'lithology/f3/facies_benchmark_v1'
		/ 'voxel_label_budget_xy_neighbor_consensus_k6810_v1/original_split'
	)
	if candidate.embeddings_dir != expected_embeddings:
		raise ValueError('XY-neighbour-consensus embeddings_dir must be canonical')
	if candidate.pretraining_handoff != expected_handoff:
		raise ValueError(
			'XY-neighbour-consensus pretraining_handoff must be canonical'
		)
	if multi_head.output_root != expected_output:
		raise ValueError('XY-neighbour-consensus output root must be candidate-owned')


def _expected_screening_audit_path(artifact_root: Path) -> Path:
	return (
		artifact_root
		/ 'lithology/f3/facies_benchmark_v1'
		/ 'voxel_label_budget_xy_neighbor_consensus_k6810_v1'
		/ 'original_split/preflight/xy_neighbor_consensus_screening_audit.json'
	)


def validate_f3_xy_neighbor_consensus_screening_audit(
	config: F3VoxelLabelBudgetXYNeighborConsensusConfig,
) -> Mapping[str, object]:
	"""Live-revalidate the immutable pre-screen audit and candidate binding."""
	return _validate_screening_audit_binding(config.screening_audit, config.multi_head)


def _required_existing_path(value: object, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty path string')
	path = Path(value).resolve()
	if not path.is_file():
		raise FileNotFoundError(path)
	return path


def _validate_hard_decoder_contract(
	xy: F3VoxelLabelBudgetMultiHeadConfig,
	hard: F3VoxelLabelBudgetMultiHeadConfig,
) -> None:
	"""Keep all original-split decoder behavior byte-for-byte paired to M4."""
	for name in (
		'dataset_manifest',
		'multi_head_target_manifest',
		'original_run_manifest',
		'current_k6_run_manifest',
	):
		if getattr(xy, name) != getattr(hard, name):
			raise ValueError(
				'XY-neighbour-consensus decoder/training contract mismatch: '
				f'{name}'
			)
	if (
		xy.references.mae_model_id != hard.references.mae_model_id
		or xy.references.current_k6_model_id != hard.references.current_k6_model_id
	):
		raise ValueError(
			'XY-neighbour-consensus decoder/training contract mismatch: '
			'reference model IDs'
		)
	for name in (
		'artifact_root',
		'f3_root',
		'results_root',
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
		if getattr(xy.base, name) != getattr(hard.base, name):
			raise ValueError(
				'XY-neighbour-consensus decoder/training contract mismatch: '
				f'{name}'
			)


def _validate_screening_audit_binding(
	path: Path,
	multi_head: F3VoxelLabelBudgetMultiHeadConfig,
) -> Mapping[str, object]:
	"""Require the PASS audit to name the live schema-v5 candidate artifacts."""
	# Load lazily so the audit module remains independently usable while this
	# closed configuration is being resolved by command-line entry points.
	audit = importlib.import_module(
		'seis_ssl_cluster.f3.lithology.xy_neighbor_consensus_screening_audit'
	)
	loader = audit.load_f3_xy_neighbor_consensus_screening_audit
	payload = loader(path)
	candidate = multi_head.candidates[0]
	validator = audit.validate_f3_xy_neighbor_consensus_screening_audit_binding
	validator(
		payload,
		model_id=candidate.model_id,
		model_tag=candidate.model_tag,
		pretraining_handoff=candidate.pretraining_handoff,
		embeddings_dir=candidate.embeddings_dir,
	)
	return payload


__all__ = [
	'EXPECTED_CANDIDATES',
	'XY_MODEL_ID',
	'XY_MODEL_TAG',
	'F3VoxelLabelBudgetXYNeighborConsensusConfig',
	'f3_lithology_voxel_label_budget_xy_neighbor_consensus_config_from_mapping',
	'validate_f3_xy_neighbor_consensus_screening_audit',
]
