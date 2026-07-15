from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from seis_ssl_cluster.config.f3_lithology_voxel_label_budget import (
	f3_lithology_voxel_label_budget_dataset_config_from_mapping,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_resolves_strict_voxel_label_budget_config(tmp_path: Path) -> None:
	raw = _mapping(tmp_path)

	config = f3_lithology_voxel_label_budget_dataset_config_from_mapping(raw)

	assert config.suite_name == 'fixture_suite'
	assert config.budgets == ('cap25', 'cap50', 'cap100')
	assert config.subsample_seeds == (0, 1, 2, 3, 4)
	assert config.patch_size_xyz == (8, 8, 8)
	assert config.models == {'mae': 'mae_tag', 'm1': 'm1_tag', 'm2a': 'm2a_tag'}


@pytest.mark.parametrize(
	('section', 'key'),
	[
		(None, 'unexpected'),
		('paths', 'unexpected'),
		('suite', 'unexpected'),
		('inputs', 'unexpected'),
		('models', 'unexpected'),
		('label_budget', 'unexpected'),
		('outputs', 'unexpected'),
	],
)
def test_rejects_unknown_keys(
	tmp_path: Path, section: str | None, key: str
) -> None:
	raw = _mapping(tmp_path)
	target = raw if section is None else raw[section]
	assert isinstance(target, dict)
	target[key] = True

	with pytest.raises(ValueError, match='not allowed'):
		f3_lithology_voxel_label_budget_dataset_config_from_mapping(raw)


@pytest.mark.parametrize(
	'budgets',
	[[], ['25'], ['cap0'], ['cap025'], ['cap25', 'cap25']],
)
def test_rejects_invalid_budget_ids(tmp_path: Path, budgets: list[str]) -> None:
	raw = _mapping(tmp_path)
	label_budget = raw['label_budget']
	assert isinstance(label_budget, dict)
	label_budget['budgets'] = budgets

	with pytest.raises((TypeError, ValueError), match='budgets'):
		f3_lithology_voxel_label_budget_dataset_config_from_mapping(raw)


@pytest.mark.parametrize(
	('key', 'value'),
	[
		('subsample_seeds', [0, 0]),
		('subsample_seeds', [-1]),
		('patch_size_xyz', [8, 8]),
		('patch_size_xyz', [8, 0, 8]),
		('require_all_classes', 1),
	],
)
def test_rejects_invalid_label_budget_fields(
	tmp_path: Path, key: str, value: object
) -> None:
	raw = _mapping(tmp_path)
	label_budget = raw['label_budget']
	assert isinstance(label_budget, dict)
	label_budget[key] = value

	with pytest.raises((TypeError, ValueError), match=key):
		f3_lithology_voxel_label_budget_dataset_config_from_mapping(raw)


def test_rejects_duplicate_models_and_paths_outside_artifact_root(
	tmp_path: Path,
) -> None:
	raw = _mapping(tmp_path)
	models = raw['models']
	assert isinstance(models, dict)
	models['m2a'] = 'm1_tag'
	with pytest.raises(ValueError, match='must be distinct'):
		f3_lithology_voxel_label_budget_dataset_config_from_mapping(raw)

	raw = _mapping(tmp_path)
	suite = raw['suite']
	assert isinstance(suite, dict)
	suite['output_root'] = str(tmp_path / 'outside')
	with pytest.raises(ValueError, match=r'under paths\.artifact_root'):
		f3_lithology_voxel_label_budget_dataset_config_from_mapping(raw)


def _mapping(tmp_path: Path) -> dict[str, object]:
	artifact_root = tmp_path / 'artifacts'
	return {
		'paths': {'artifact_root': str(artifact_root)},
		'suite': {
			'name': 'fixture_suite',
			'output_root': str(artifact_root / 'low_label'),
		},
		'inputs': {
			'common_voxel_dataset': str(artifact_root / 'common'),
			'mae_m1_label_budget_manifest': str(
				artifact_root / 'label_budget_m1' / 'suite_manifest.json'
			),
			'm1_m2a_label_budget_manifest': str(
				artifact_root / 'label_budget_m2a' / 'suite_manifest.json'
			),
		},
		'models': {'mae': 'mae_tag', 'm1': 'm1_tag', 'm2a': 'm2a_tag'},
		'label_budget': {
			'budgets': ['cap25', 'cap50', 'cap100'],
			'subsample_seeds': [0, 1, 2, 3, 4],
			'patch_size_xyz': [8, 8, 8],
			'require_all_classes': True,
		},
		'outputs': {'overwrite': False},
	}
