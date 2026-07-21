"""Strict configuration for the paired multi-head low-label voxel matrix."""
# ruff: noqa: TC003

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_required_str,
	_validate_allowed_keys,
)
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_control import (
	CANONICAL_BASE_SEED,
	CURRENT_K6_MODEL_ID,
	CURRENT_K6_MODEL_TAG,
	F3VoxelLabelBudgetControlConfig,
	f3_lithology_voxel_label_budget_control_config_from_mapping,
)
from seis_ssl_cluster.paths import ensure_under_root

EXPECTED_CANDIDATES = (
	('mh_nocons', 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'),
	('mh_cons010', 'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1'),
)


@dataclass(frozen=True)
class F3VoxelLabelBudgetMultiHeadCandidate:
	"""One frozen multi-head encoder admitted to the matrix."""

	model_id: str
	model_tag: str
	embeddings_dir: Path
	pretraining_handoff: Path


@dataclass(frozen=True)
class F3VoxelLabelBudgetMultiHeadReferences:
	"""Read-only paired source manifests."""

	dataset_manifest: Path
	multi_head_target_manifest: Path
	original_run_manifest: Path
	current_k6_run_manifest: Path
	mae_model_id: str
	current_k6_model_id: str
	historical_m1_model_id: str | None


@dataclass(frozen=True)
class F3VoxelLabelBudgetMultiHeadConfig:
	"""Common decoder contract plus exactly the two candidate encoders."""

	base: F3VoxelLabelBudgetControlConfig
	references: F3VoxelLabelBudgetMultiHeadReferences
	candidates: tuple[F3VoxelLabelBudgetMultiHeadCandidate, ...]

	def __getattr__(self, name: str) -> object:
		"""Delegate fixed decoder settings to the established control config."""
		return getattr(self.base, name)

	@property
	def dataset_manifest(self) -> Path:
		"""Return the immutable shared dataset manifest."""
		return self.references.dataset_manifest

	@property
	def original_run_manifest(self) -> Path:
		"""Return the immutable MAE/historical-M1 run reference."""
		return self.references.original_run_manifest

	@property
	def multi_head_target_manifest(self) -> Path:
		"""Return the exact K=6/8/10 pseudo-target manifest expected upstream."""
		return self.references.multi_head_target_manifest

	@property
	def current_k6_run_manifest(self) -> Path:
		"""Return the immutable current-K6 control manifest."""
		return self.references.current_k6_run_manifest

	@property
	def reports_dir(self) -> Path:
		"""Return this issue's owned report directory."""
		return self.base.output_root / 'reports'

	@property
	def job_count(self) -> int:
		"""Return the exact candidate/budget/seed matrix size."""
		return (
			len(self.candidates)
			* len(self.base.budgets)
			* len(self.base.subsample_seeds)
		)

	def decoder_seed(self, subsample_seed: int) -> int:
		"""Return the paired decoder seed for a configured subsample seed."""
		return self.base.decoder_seed(subsample_seed)


def f3_lithology_voxel_label_budget_multi_head_config_from_mapping(  # noqa: C901
	config: Mapping[str, object],
) -> F3VoxelLabelBudgetMultiHeadConfig:
	"""Resolve the generic candidate list without widening the control config."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
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
			}
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	references = _required_mapping(config, 'references')
	candidates = config.get('candidates')
	if not isinstance(candidates, list) or not candidates:
		raise TypeError('candidates must be a non-empty list')
	_validate_allowed_keys(
		references,
		frozenset(
			{
				'dataset_manifest',
				'multi_head_target_manifest',
				'original_run_manifest',
				'current_k6_run_manifest',
				'mae_model_id',
				'current_k6_model_id',
				'historical_m1_model_id',
			}
		),
		prefix='references',
	)
	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	resolved_candidates: list[F3VoxelLabelBudgetMultiHeadCandidate] = []
	for index, value in enumerate(candidates):
		candidate = _required_mapping({'candidate': value}, 'candidate')
		_validate_allowed_keys(
			candidate,
			frozenset(
				{'model_id', 'model_tag', 'embeddings_dir', 'pretraining_handoff'}
			),
			prefix=f'candidates[{index}]',
		)
		item = F3VoxelLabelBudgetMultiHeadCandidate(
			model_id=_required_str(
				candidate, 'model_id', prefix=f'candidates[{index}]'
			),
			model_tag=_required_str(
				candidate, 'model_tag', prefix=f'candidates[{index}]'
			),
			embeddings_dir=_required_absolute_path(
				candidate, 'embeddings_dir', prefix=f'candidates[{index}]'
			),
			pretraining_handoff=_required_absolute_path(
				candidate, 'pretraining_handoff', prefix=f'candidates[{index}]'
			),
		)
		ensure_under_root(
			item.embeddings_dir,
			root=artifact_root,
			label=f'candidates[{index}].embeddings_dir',
		)
		ensure_under_root(
			item.pretraining_handoff,
			root=artifact_root,
			label=f'candidates[{index}].pretraining_handoff',
		)
		resolved_candidates.append(item)
	if (
		tuple((item.model_id, item.model_tag) for item in resolved_candidates)
		!= EXPECTED_CANDIDATES
	):
		raise ValueError('candidates must be the canonical nocons then cons010 pair')
	if len({item.embeddings_dir for item in resolved_candidates}) != len(
		resolved_candidates
	):
		raise ValueError('candidate embedding paths must not be duplicated')
	if len({item.pretraining_handoff for item in resolved_candidates}) != len(
		resolved_candidates
	):
		raise ValueError('candidate handoff paths must not be duplicated')
	refs = F3VoxelLabelBudgetMultiHeadReferences(
		dataset_manifest=_required_absolute_path(
			references, 'dataset_manifest', prefix='references'
		),
		multi_head_target_manifest=_required_absolute_path(
			references, 'multi_head_target_manifest', prefix='references'
		),
		original_run_manifest=_required_absolute_path(
			references, 'original_run_manifest', prefix='references'
		),
		current_k6_run_manifest=_required_absolute_path(
			references, 'current_k6_run_manifest', prefix='references'
		),
		mae_model_id=_required_str(references, 'mae_model_id', prefix='references'),
		current_k6_model_id=_required_str(
			references, 'current_k6_model_id', prefix='references'
		),
		historical_m1_model_id=(
			_required_str(
				references,
				'historical_m1_model_id',
				prefix='references',
			)
			if 'historical_m1_model_id' in references
			else None
		),
	)
	for label, path in (
		('references.dataset_manifest', refs.dataset_manifest),
		('references.multi_head_target_manifest', refs.multi_head_target_manifest),
		('references.original_run_manifest', refs.original_run_manifest),
		('references.current_k6_run_manifest', refs.current_k6_run_manifest),
	):
		ensure_under_root(path, root=artifact_root, label=label)
	if (refs.mae_model_id, refs.current_k6_model_id) != (
		'mae',
		CURRENT_K6_MODEL_ID,
	):
		raise ValueError('reference model IDs must be mae, m1_current_k6')
	if (
		refs.historical_m1_model_id is not None
		and refs.historical_m1_model_id != 'm1'
	):
		raise ValueError('historical_m1_model_id must be m1')
	reference_ids = {refs.mae_model_id, refs.current_k6_model_id}
	if refs.historical_m1_model_id is not None:
		reference_ids.add(refs.historical_m1_model_id)
	if any(item.model_id in reference_ids for item in resolved_candidates):
		raise ValueError('candidate IDs must not collide with reference IDs')

	# The established resolver remains the single authority for the common fixed
	# decoder contract. The multi-head runner validates MAE separately;
	# historical M1 is an optional report-only source and must not gate candidate
	# execution.
	base_raw = dict(config)
	base_raw.pop('candidates')
	base_raw['references'] = {
		'dataset_manifest': str(refs.dataset_manifest),
	}
	base_raw['candidate'] = {
		'model_id': CURRENT_K6_MODEL_ID,
		'model_tag': CURRENT_K6_MODEL_TAG,
		'embeddings_dir': str(
			artifact_root
			/ 'embeddings/f3/facies_benchmark_v1'
			/ CURRENT_K6_MODEL_TAG
			/ 'overlap_x16'
		),
	}
	base_raw['report'] = {
		'selected_slices': {'inline': [], 'crossline': []},
		'dpi': 150,
		'include_confidence': False,
		'amplitude_clip_percentiles': [1.0, 99.0],
	}
	base_raw['comparisons'] = [
		[CURRENT_K6_MODEL_ID, 'm1'],
		[CURRENT_K6_MODEL_ID, 'mae'],
		['m1', 'mae'],
	]
	base_raw['decision'] = {
		'minimum_positive_budgets': 2,
		'minimum_primary_wins': 4,
		'drift_absolute_mean_delta': 0.01,
		'drift_budget_count': 2,
		'monitored_class_ids': [3, 5],
		'major_degradation_delta': -0.05,
		'systematic_degradation_budget_count': 2,
	}
	base_raw['publish'] = {
		'enabled': False,
		'output_dir': str(
			_required_absolute_path(paths, 'results_root', prefix='paths')
			/ 'f3/facies_benchmark_v1/multi_head_unused_publish'
		),
		'max_file_size_mb': 10,
		'overwrite': False,
	}
	base = f3_lithology_voxel_label_budget_control_config_from_mapping(
		base_raw, validate_pairing_reference=False
	)
	if base.base_seed != CANONICAL_BASE_SEED:
		raise ValueError('seed policy must use base_seed 42000')
	return F3VoxelLabelBudgetMultiHeadConfig(
		base=base, references=refs, candidates=tuple(resolved_candidates)
	)
__all__ = [
	'EXPECTED_CANDIDATES',
	'F3VoxelLabelBudgetMultiHeadCandidate',
	'F3VoxelLabelBudgetMultiHeadConfig',
	'F3VoxelLabelBudgetMultiHeadReferences',
	'f3_lithology_voxel_label_budget_multi_head_config_from_mapping',
]
