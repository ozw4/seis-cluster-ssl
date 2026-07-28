"""M5-U one-candidate runner backed by the shared decoder job machinery."""
# ruff: noqa: TC003

from __future__ import annotations

from pathlib import Path

from seis_ssl_cluster.f3.lithology.voxel_label_budget_multi_head import (
	F3VoxelLabelBudgetMultiHeadInspection,
	F3VoxelLabelBudgetMultiHeadRunResult,
	inspect_f3_lithology_voxel_label_budget_multi_head,
	load_f3_lithology_voxel_label_budget_multi_head_rows,
	multi_head_run_manifest_path,
	run_f3_lithology_voxel_label_budget_multi_head,
)


def inspect_f3_lithology_voxel_label_budget_soft_posterior(
	config: object, **filters: object
) -> F3VoxelLabelBudgetMultiHeadInspection:
	"""Inspect the isolated 3-budget by 5-seed M5-U job matrix."""
	return inspect_f3_lithology_voxel_label_budget_multi_head(config, **filters)


def run_f3_lithology_voxel_label_budget_soft_posterior(
	config: object, **kwargs: object
) -> F3VoxelLabelBudgetMultiHeadRunResult:
	"""Run, reuse, resume, or quarantine only M5-U-owned decoder jobs."""
	return run_f3_lithology_voxel_label_budget_multi_head(config, **kwargs)


def load_f3_lithology_voxel_label_budget_soft_posterior_rows(
	config: object,
) -> tuple[object, ...]:
	"""Live-revalidate all fifteen completed M5-U rows."""
	return load_f3_lithology_voxel_label_budget_multi_head_rows(config)


def soft_posterior_run_manifest_path(config: object) -> Path:
	"""Return the M5-U candidate-owned manifest path."""
	return multi_head_run_manifest_path(config)
