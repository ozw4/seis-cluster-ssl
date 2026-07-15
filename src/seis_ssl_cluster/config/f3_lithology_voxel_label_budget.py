"""Strict configurations for the original-split voxel label-budget benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_required_str,
	_validate_allowed_keys,
)

if TYPE_CHECKING:
	from pathlib import Path


@dataclass(frozen=True)
class F3VoxelLabelBudgetDatasetConfig:
	"""Inputs and outputs for the encoder-independent low-label voxel grids."""

	artifact_root: Path
	suite_name: str
	output_root: Path
	common_voxel_dataset: Path
	mae_m1_label_budget_manifest: Path
	m1_m2a_label_budget_manifest: Path
	models: Mapping[str, str]
	budgets: tuple[str, ...]
	subsample_seeds: tuple[int, ...]
	patch_size_xyz: tuple[int, int, int]
	require_all_classes: bool
	overwrite: bool

	def __post_init__(self) -> None:
		"""Validate output placement for direct dataclass construction."""
		_require_under_artifact_root(
			self.output_root, self.artifact_root, 'suite.output_root'
		)


def f3_lithology_voxel_label_budget_dataset_config_from_mapping(
	config: Mapping[str, object],
) -> F3VoxelLabelBudgetDatasetConfig:
	"""Resolve a low-label voxel dataset builder mapping with unknown-key rejection."""
	_validate_allowed_keys(
		config,
		frozenset({'paths', 'suite', 'inputs', 'models', 'label_budget', 'outputs'}),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	suite = _required_mapping(config, 'suite')
	inputs = _required_mapping(config, 'inputs')
	models = _required_mapping(config, 'models')
	label_budget = _required_mapping(config, 'label_budget')
	outputs = _required_mapping(config, 'outputs')
	_validate_allowed_keys(paths, frozenset({'artifact_root'}), prefix='paths')
	_validate_allowed_keys(suite, frozenset({'name', 'output_root'}), prefix='suite')
	_validate_allowed_keys(
		inputs,
		frozenset(
			{
				'common_voxel_dataset',
				'mae_m1_label_budget_manifest',
				'm1_m2a_label_budget_manifest',
			}
		),
		prefix='inputs',
	)
	_validate_allowed_keys(models, frozenset({'mae', 'm1', 'm2a'}), prefix='models')
	_validate_allowed_keys(
		label_budget,
		frozenset(
			{'budgets', 'subsample_seeds', 'patch_size_xyz', 'require_all_classes'}
		),
		prefix='label_budget',
	)
	_validate_allowed_keys(outputs, frozenset({'overwrite'}), prefix='outputs')
	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	output_root = _required_absolute_path(suite, 'output_root', prefix='suite')
	resolved_inputs = {
		key: _required_absolute_path(inputs, key, prefix='inputs')
		for key in (
			'common_voxel_dataset',
			'mae_m1_label_budget_manifest',
			'm1_m2a_label_budget_manifest',
		)
	}
	for label, path in (
		('suite.output_root', output_root),
		*((f'inputs.{key}', value) for key, value in resolved_inputs.items()),
	):
		_require_under_artifact_root(path, artifact_root, label)
	model_tags = {
		role: _required_str(models, role, prefix='models')
		for role in ('mae', 'm1', 'm2a')
	}
	if len(set(model_tags.values())) != 3:
		raise ValueError('models.mae, models.m1, and models.m2a must be distinct')
	require_all_classes = label_budget.get('require_all_classes')
	if not isinstance(require_all_classes, bool):
		raise TypeError('label_budget.require_all_classes must be boolean')
	overwrite = outputs.get('overwrite')
	if not isinstance(overwrite, bool):
		raise TypeError('outputs.overwrite must be boolean')
	return F3VoxelLabelBudgetDatasetConfig(
		artifact_root=artifact_root,
		suite_name=_required_str(suite, 'name', prefix='suite'),
		output_root=output_root,
		common_voxel_dataset=resolved_inputs['common_voxel_dataset'],
		mae_m1_label_budget_manifest=resolved_inputs[
			'mae_m1_label_budget_manifest'
		],
		m1_m2a_label_budget_manifest=resolved_inputs[
			'm1_m2a_label_budget_manifest'
		],
		models=model_tags,
		budgets=_budget_ids(label_budget.get('budgets')),
		subsample_seeds=_nonnegative_ints(
			label_budget.get('subsample_seeds'), 'label_budget.subsample_seeds'
		),
		patch_size_xyz=_positive_triplet(
			label_budget.get('patch_size_xyz'), 'label_budget.patch_size_xyz'
		),
		require_all_classes=require_all_classes,
		overwrite=overwrite,
	)


def _budget_ids(value: object) -> tuple[str, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
		raise TypeError('label_budget.budgets must be a non-empty list')
	result: list[str] = []
	for item in value:
		if not isinstance(item, str) or not item.startswith('cap'):
			raise ValueError(
				'label_budget.budgets entries must have form cap<positive-int>'
			)
		try:
			cap = int(item[3:])
		except ValueError as error:
			raise ValueError(
				'label_budget.budgets entries must have form cap<positive-int>'
			) from error
		if cap <= 0 or item != f'cap{cap}':
			raise ValueError(
				'label_budget.budgets entries must be canonical positive caps'
			)
		result.append(item)
	if len(set(result)) != len(result):
		raise ValueError('label_budget.budgets must not contain duplicates')
	return tuple(result)


def _nonnegative_ints(value: object, label: str) -> tuple[int, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
		raise TypeError(f'{label} must be a non-empty list')
	if any(
		not isinstance(item, int) or isinstance(item, bool) or item < 0
		for item in value
	):
		raise ValueError(f'{label} must contain non-negative integers')
	result = tuple(int(item) for item in value)
	if len(set(result)) != len(result):
		raise ValueError(f'{label} must not contain duplicates')
	return result


def _positive_triplet(value: object, label: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 3
	):
		raise TypeError(f'{label} must be an integer triple')
	items = tuple(value)
	if any(
		not isinstance(item, int) or isinstance(item, bool) or item <= 0
		for item in items
	):
		raise ValueError(f'{label} must contain positive integers')
	return (int(items[0]), int(items[1]), int(items[2]))


def _require_under_artifact_root(path: Path, root: Path, label: str) -> None:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError as error:
		raise ValueError(f'{label} must be under paths.artifact_root') from error


__all__ = [
	'F3VoxelLabelBudgetDatasetConfig',
	'f3_lithology_voxel_label_budget_dataset_config_from_mapping',
]
