"""Schema-v6 XY-neighbour-unanimous decoder runner wrapper."""

from __future__ import annotations

from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
	F3VoxelLabelBudgetMultiHeadCandidate,
	f3_lithology_voxel_label_budget_multi_head_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_multi_head import (
	F3VoxelLabelBudgetMultiHeadInspection,
	F3VoxelLabelBudgetMultiHeadRunResult,
	inspect_f3_lithology_voxel_label_budget_multi_head,
	load_f3_lithology_voxel_label_budget_multi_head_rows,
	multi_head_run_manifest_path,
	run_f3_lithology_voxel_label_budget_multi_head,
)

_MODEL_ID = 'mh_xyunanim1_nocons'
_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_xyunanim1_nocons_topblock1_distill_v1'


def inspect_f3_lithology_voxel_label_budget_xy_neighbor_unanimous(
	config: object, **filters: object
) -> F3VoxelLabelBudgetMultiHeadInspection:
	"""Inspect the closed 3-budget by 5-seed unanimous candidate matrix."""
	_validate_hard_reference_rows(config)
	return inspect_f3_lithology_voxel_label_budget_multi_head(config, **filters)


def run_f3_lithology_voxel_label_budget_xy_neighbor_unanimous(
	config: object, **kwargs: object
) -> F3VoxelLabelBudgetMultiHeadRunResult:
	"""Run, reuse, resume, or quarantine only unanimous-candidate-owned jobs."""
	_validate_hard_reference_rows(config)
	return run_f3_lithology_voxel_label_budget_multi_head(config, **kwargs)


def load_f3_lithology_voxel_label_budget_xy_neighbor_unanimous_rows(
	config: object,
) -> tuple[object, ...]:
	"""Live-revalidate all fifteen completed schema-v6 candidate rows."""
	return load_f3_lithology_voxel_label_budget_multi_head_rows(config)


def xy_neighbor_unanimous_run_manifest_path(config: object) -> Path:
	"""Return the candidate-owned schema-v6 job manifest path."""
	return multi_head_run_manifest_path(config)


def _validate_hard_reference_rows(config: object) -> None:
	"""Admit the primary hard baseline only after live row revalidation."""
	_validate_unanimous_candidate_matrix(config)
	hard_path = getattr(config, 'hard_multi_head_config', None)
	if not isinstance(hard_path, Path):
		raise TypeError('XY-neighbour-unanimous hard_multi_head_config is missing')
	hard = f3_lithology_voxel_label_budget_multi_head_config_from_mapping(
		load_config(hard_path)
	)
	rows = load_f3_lithology_voxel_label_budget_multi_head_rows(hard)
	primary = [row for row in rows if row.get('model_role') == 'mh_nocons']
	expected = len(config.budgets) * len(config.subsample_seeds)
	if len(primary) != expected:
		raise ValueError('hard mh_nocons reference matrix is incomplete')


def _validate_unanimous_candidate_matrix(config: object) -> None:
	"""Keep every public wrapper call limited to the fifteen new candidate jobs."""
	candidates = getattr(config, 'candidates', None)
	if not isinstance(candidates, tuple) or len(candidates) != 1:
		raise ValueError('unanimous runner requires exactly one candidate')
	candidate = candidates[0]
	if not isinstance(candidate, F3VoxelLabelBudgetMultiHeadCandidate) or (
		candidate.model_id != _MODEL_ID or candidate.model_tag != _MODEL_TAG
	):
		raise ValueError('unanimous runner candidate identity mismatch')
	budgets = getattr(config, 'budgets', None)
	seeds = getattr(config, 'subsample_seeds', None)
	if not isinstance(budgets, tuple) or not isinstance(seeds, tuple):
		raise TypeError('unanimous runner budgets/seeds are missing')
	if len(budgets) * len(seeds) != 15:
		raise ValueError('unanimous runner requires exactly fifteen jobs')


__all__ = [
	'inspect_f3_lithology_voxel_label_budget_xy_neighbor_unanimous',
	'load_f3_lithology_voxel_label_budget_xy_neighbor_unanimous_rows',
	'run_f3_lithology_voxel_label_budget_xy_neighbor_unanimous',
	'xy_neighbor_unanimous_run_manifest_path',
]
