"""Strict contract for the M4 multi-head six-split low-label suite."""
# ruff: noqa: D102, E501, TC003

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_validate_allowed_keys,
)
from seis_ssl_cluster.paths import ensure_under_root

SPLIT_IDS = tuple(f'split_{index:03d}' for index in range(6))
BUDGETS = ('cap25', 'cap50')
MODEL_TAGS = {
	'mae': 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
	'm1_current_k6': 'strat_hmm_pretext_m1_current_k6_topblock1_distill_v1',
	'mh_nocons': 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
}


@dataclass(frozen=True)
class F3VoxelLabelBudgetSplitConfig:
	"""Immutable paths and the intentionally fixed confirmatory matrix."""

	artifact_root: Path
	results_root: Path
	output_root: Path
	split_inventory_manifest: Path
	split_dataset_manifest: Path
	voxel_dataset_manifest: Path
	original_dataset_manifest: Path
	multi_head_decisions: Path
	multi_head_handoff: Path
	embeddings: Mapping[str, Path]
	split_ids: tuple[str, ...]
	budgets: tuple[str, ...]
	label_subset_seed: int
	decoder_seed: int

	@property
	def models(self) -> tuple[str, ...]:
		return tuple(MODEL_TAGS)

	@property
	def job_count(self) -> int:
		return len(self.split_ids) * len(self.budgets) * len(self.models)


def f3_lithology_voxel_label_budget_split_config_from_mapping(
	config: Mapping[str, object],
) -> F3VoxelLabelBudgetSplitConfig:
	"""Resolve the isolated suite and reject every tuning or selection knob."""
	_validate_allowed_keys(config, frozenset({'paths', 'inputs', 'matrix'}), prefix='config')
	paths = _required_mapping(config, 'paths')
	inputs = _required_mapping(config, 'inputs')
	matrix = _required_mapping(config, 'matrix')
	_validate_allowed_keys(paths, frozenset({'artifact_root', 'results_root', 'output_root'}), prefix='paths')
	_validate_allowed_keys(inputs, frozenset({
		'split_inventory_manifest', 'split_dataset_manifest', 'voxel_dataset_manifest',
		'original_dataset_manifest', 'multi_head_decisions', 'multi_head_handoff', 'embeddings',
	}), prefix='inputs')
	_validate_allowed_keys(matrix, frozenset({'split_ids', 'budgets', 'per_class_caps', 'label_subset_seed', 'decoder_seed', 'models'}), prefix='matrix')
	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	result = F3VoxelLabelBudgetSplitConfig(
		artifact_root=artifact_root,
		results_root=_required_absolute_path(paths, 'results_root', prefix='paths'),
		output_root=_required_absolute_path(paths, 'output_root', prefix='paths'),
		split_inventory_manifest=_required_absolute_path(inputs, 'split_inventory_manifest', prefix='inputs'),
		split_dataset_manifest=_required_absolute_path(inputs, 'split_dataset_manifest', prefix='inputs'),
		voxel_dataset_manifest=_required_absolute_path(inputs, 'voxel_dataset_manifest', prefix='inputs'),
		original_dataset_manifest=_required_absolute_path(inputs, 'original_dataset_manifest', prefix='inputs'),
		multi_head_decisions=_required_absolute_path(inputs, 'multi_head_decisions', prefix='inputs'),
		multi_head_handoff=_required_absolute_path(inputs, 'multi_head_handoff', prefix='inputs'),
		embeddings=_embedding_paths(inputs, artifact_root),
		split_ids=_strings(matrix.get('split_ids'), 'matrix.split_ids'),
		budgets=_strings(matrix.get('budgets'), 'matrix.budgets'),
		label_subset_seed=_integer(matrix.get('label_subset_seed'), 'matrix.label_subset_seed'),
		decoder_seed=_integer(matrix.get('decoder_seed'), 'matrix.decoder_seed'),
	)
	for label, path in ((name, getattr(result, name)) for name in (
		'output_root', 'split_inventory_manifest', 'split_dataset_manifest', 'voxel_dataset_manifest',
		'original_dataset_manifest', 'multi_head_handoff', *(),
	)):
		ensure_under_root(path, root=artifact_root, label=label)
	if result.split_ids != SPLIT_IDS or len(set(result.split_ids)) != len(SPLIT_IDS):
		raise ValueError('matrix.split_ids must be the canonical six unique split IDs')
	if result.budgets != BUDGETS or matrix.get('per_class_caps') != [25, 50]:
		raise ValueError('matrix budgets/per_class_caps must be cap25/cap50 and 25/50')
	if result.label_subset_seed != 0 or result.decoder_seed != 42000:
		raise ValueError('matrix seeds must be label_subset_seed=0 and decoder_seed=42000')
	if _strings(matrix.get('models'), 'matrix.models') != tuple(MODEL_TAGS):
		raise ValueError('matrix.models must be mae, m1_current_k6, mh_nocons')
	return result


def _embedding_paths(inputs: Mapping[str, object], artifact_root: Path) -> Mapping[str, Path]:
	raw = _required_mapping(inputs, 'embeddings')
	_validate_allowed_keys(raw, frozenset(MODEL_TAGS), prefix='inputs.embeddings')
	paths = {name: _required_absolute_path(raw, name, prefix='inputs.embeddings') for name in MODEL_TAGS}
	for name, path in paths.items():
		ensure_under_root(path, root=artifact_root, label=f'inputs.embeddings.{name}')
	return paths


def _strings(value: object, label: str) -> tuple[str, ...]:
	if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
		raise TypeError(f'{label} must be a list of strings')
	return tuple(value)


def _integer(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		raise TypeError(f'{label} must be an integer')
	return value


__all__ = ['BUDGETS', 'MODEL_TAGS', 'SPLIT_IDS', 'F3VoxelLabelBudgetSplitConfig', 'f3_lithology_voxel_label_budget_split_config_from_mapping']
