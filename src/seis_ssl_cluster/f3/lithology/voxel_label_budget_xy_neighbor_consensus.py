"""Schema-v5 XY-neighbour-consensus decoder runner wrapper."""

from __future__ import annotations

from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
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


def inspect_f3_lithology_voxel_label_budget_xy_neighbor_consensus(
	config: object, **filters: object
) -> F3VoxelLabelBudgetMultiHeadInspection:
	"""Inspect the closed 3-budget by 5-seed XY-consensus candidate matrix."""
	_validate_hard_reference_rows(config)
	return inspect_f3_lithology_voxel_label_budget_multi_head(config, **filters)


def run_f3_lithology_voxel_label_budget_xy_neighbor_consensus(
	config: object, **kwargs: object
) -> F3VoxelLabelBudgetMultiHeadRunResult:
	"""Run, reuse, resume, or quarantine only XY-consensus-owned jobs."""
	_validate_hard_reference_rows(config)
	return run_f3_lithology_voxel_label_budget_multi_head(config, **kwargs)


def load_f3_lithology_voxel_label_budget_xy_neighbor_consensus_rows(
	config: object,
) -> tuple[object, ...]:
	"""Live-revalidate all fifteen completed schema-v5 candidate rows."""
	return load_f3_lithology_voxel_label_budget_multi_head_rows(config)


def xy_neighbor_consensus_run_manifest_path(config: object) -> Path:
	"""Return the candidate-owned schema-v5 job manifest path."""
	return multi_head_run_manifest_path(config)


def _validate_hard_reference_rows(config: object) -> None:
	"""Admit the primary hard baseline only after live row revalidation."""
	hard_path = getattr(config, 'hard_multi_head_config', None)
	if not isinstance(hard_path, Path):
		raise TypeError('XY-consensus config hard_multi_head_config is missing')
	hard = f3_lithology_voxel_label_budget_multi_head_config_from_mapping(
		load_config(hard_path)
	)
	rows = load_f3_lithology_voxel_label_budget_multi_head_rows(hard)
	primary = [row for row in rows if row.get('model_role') == 'mh_nocons']
	expected = len(config.budgets) * len(config.subsample_seeds)
	if len(primary) != expected:
		raise ValueError('hard mh_nocons reference matrix is incomplete')


__all__ = [
	'inspect_f3_lithology_voxel_label_budget_xy_neighbor_consensus',
	'load_f3_lithology_voxel_label_budget_xy_neighbor_consensus_rows',
	'run_f3_lithology_voxel_label_budget_xy_neighbor_consensus',
	'xy_neighbor_consensus_run_manifest_path',
]
