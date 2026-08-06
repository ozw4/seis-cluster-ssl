"""Closed M5-U configuration for the original-split soft-posterior screen."""
# ruff: noqa: D102, D105, TC003

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
	F3VoxelLabelBudgetMultiHeadConfig,
	config_from_mapping_for_candidates,
	f3_lithology_voxel_label_budget_multi_head_config_from_mapping,
)

SOFT_MODEL_ID = 'mh_soft_nocons'
SOFT_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_soft_nocons_topblock1_distill_v1'
EXPECTED_CANDIDATES = ((SOFT_MODEL_ID, SOFT_MODEL_TAG),)


@dataclass(frozen=True)
class F3VoxelLabelBudgetSoftPosteriorConfig:
	"""One M5-U candidate while preserving the shared decoder contract."""

	multi_head: F3VoxelLabelBudgetMultiHeadConfig
	hard_multi_head_config: Path

	def __getattr__(self, name: str) -> object:
		return getattr(self.multi_head, name)

	@property
	def base(self) -> object:
		"""Expose the shared control configuration expected by the job helper."""
		return self.multi_head.base

	@property
	def candidates(self) -> tuple[object, ...]:
		return self.multi_head.candidates

	@property
	def run_manifest_name(self) -> str:
		return 'soft_posterior_job_manifest.json'

	@property
	def run_manifest_type(self) -> str:
		return 'f3_lithology_voxel_label_budget_soft_posterior'


def f3_lithology_voxel_label_budget_soft_posterior_config_from_mapping(
	config: Mapping[str, object],
) -> F3VoxelLabelBudgetSoftPosteriorConfig:
	"""Resolve only the single preregistered M5-U candidate and 15-job matrix."""
	unknown = set(config) - {
		'paths',
		'dataset',
		'references',
		'candidates',
		'budgets',
		'subsample_seeds',
		'seed_policy',
		'labels',
		'decoder',
		'tiles',
		'train',
		'inference',
		'evaluation',
		'outputs',
		'hard_multi_head_config',
	}
	if unknown:
		raise ValueError(f'unknown M5-U config keys: {sorted(unknown)!r}')
	value = config.get('hard_multi_head_config')
	if not isinstance(value, str) or not value:
		raise TypeError('hard_multi_head_config must be a non-empty path string')
	hard_multi_head_config = Path(value).resolve()
	if not hard_multi_head_config.is_file():
		raise FileNotFoundError(hard_multi_head_config)
	base = config_from_mapping_for_candidates(
		{key: item for key, item in config.items() if key != 'hard_multi_head_config'},
		expected_candidates=EXPECTED_CANDIDATES,
	)
	hard_config = f3_lithology_voxel_label_budget_multi_head_config_from_mapping(
		load_config(hard_multi_head_config)
	)
	_validate_hard_decoder_contract(base, hard_config)
	if base.job_count != 15:
		raise ValueError('M5-U soft-posterior screen must contain exactly 15 jobs')
	if base.candidates[0].model_id != SOFT_MODEL_ID:
		raise ValueError('M5-U candidate identity mismatch')
	return F3VoxelLabelBudgetSoftPosteriorConfig(
		multi_head=base, hard_multi_head_config=hard_multi_head_config
	)


def _validate_hard_decoder_contract(
	soft: F3VoxelLabelBudgetMultiHeadConfig,
	hard: F3VoxelLabelBudgetMultiHeadConfig,
) -> None:
	"""Require M5-U to retain the original-split decoder scientific contract."""
	for name in (
		'dataset_manifest',
		'multi_head_target_manifest',
		'original_run_manifest',
		'current_k6_run_manifest',
	):
		if getattr(soft, name) != getattr(hard, name):
			raise ValueError(f'M5-U decoder/training contract mismatch: {name}')
	if (
		soft.references.mae_model_id != hard.references.mae_model_id
		or soft.references.current_k6_model_id
		!= hard.references.current_k6_model_id
	):
		raise ValueError('M5-U decoder/training contract mismatch: reference model IDs')
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
		if getattr(soft.base, name) != getattr(hard.base, name):
			raise ValueError(f'M5-U decoder/training contract mismatch: {name}')


__all__ = [
	'EXPECTED_CANDIDATES',
	'SOFT_MODEL_ID',
	'SOFT_MODEL_TAG',
	'F3VoxelLabelBudgetSoftPosteriorConfig',
	'f3_lithology_voxel_label_budget_soft_posterior_config_from_mapping',
]
