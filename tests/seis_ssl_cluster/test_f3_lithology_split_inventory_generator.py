from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from seis_ssl_cluster.config.f3_lithology_robustness import (
	f3_lithology_split_inventory_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.robustness import (
	build_f3_lithology_split_inventories,
)
from seis_ssl_cluster.f3.splits import load_f3_slice_split_records

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / 'proc' / 'seis_ssl_cluster' / (
	'generate_f3_lithology_split_inventories.py'
)


def test_base_split_copied_as_split_000(tmp_path: Path) -> None:
	config = _config(tmp_path)
	source_rows = _read_rows(config.inputs.base_png_label_inventory)

	build_f3_lithology_split_inventories(config)

	output_rows = _read_rows(
		config.output_root / 'split_inventories/split_000/png_label_inventory.csv',
	)
	assert [row['split'] for row in output_rows] == [
		row['split'] for row in source_rows
	]
	assert load_f3_slice_split_records(
		config.output_root / 'split_inventories/split_000/png_label_inventory.csv',
	)


def test_generated_split_has_configured_validation_count(tmp_path: Path) -> None:
	config = _config(tmp_path)

	build_f3_lithology_split_inventories(config)

	records = load_f3_slice_split_records(
		config.output_root / 'split_inventories/split_001/png_label_inventory.csv',
	)
	assert sum(record.split == 'validation' for record in records) == 2


def test_non_split_fields_are_preserved(tmp_path: Path) -> None:
	config = _config(tmp_path)
	source_rows = _read_rows(config.inputs.base_png_label_inventory)

	build_f3_lithology_split_inventories(config)

	output_rows = _read_rows(
		config.output_root / 'split_inventories/split_001/png_label_inventory.csv',
	)
	for source, output in zip(source_rows, output_rows, strict=True):
		source_without_split = {
			key: value for key, value in source.items() if key != 'split'
		}
		output_without_split = {
			key: value for key, value in output.items() if key != 'split'
		}
		assert output_without_split == source_without_split


def test_class_support_constraints_are_enforced(tmp_path: Path) -> None:
	config = _config(
		tmp_path,
		min_validation_tokens_per_class={'default': 1, '1': 3},
	)

	build_f3_lithology_split_inventories(config)

	metadata = json.loads(
		(
			config.output_root / 'split_inventories/split_001/split_metadata.json'
		).read_text(encoding='utf-8'),
	)
	counts = metadata['validation_class_counts_estimated']
	assert counts['0'] >= 1
	assert counts['1'] >= 3


def test_impossible_constraints_fail_with_useful_error(tmp_path: Path) -> None:
	config = _config(tmp_path, min_validation_tokens_per_class={'default': 100})

	with pytest.raises(ValueError, match=r'could not generate.*constraints'):
		build_f3_lithology_split_inventories(config)


def test_overwrite_false_rejects_existing_outputs(tmp_path: Path) -> None:
	config = _config(tmp_path)
	build_f3_lithology_split_inventories(config)

	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		build_f3_lithology_split_inventories(config)


def test_cli_dry_run_prints_expected_summary(tmp_path: Path) -> None:
	paths = _write_fixture(tmp_path)
	config_path = tmp_path / 'split_inventory.yaml'
	config_path.write_text(_config_yaml(paths, tmp_path / 'out'), encoding='utf-8')
	env = os.environ.copy()
	env['PYTHONPATH'] = os.pathsep.join(
		(str(REPO_ROOT / 'src'), env.get('PYTHONPATH', '')),
	)

	completed = subprocess.run(  # noqa: S603
		[sys.executable, str(CLI), '--config', str(config_path), '--dry-run'],
		cwd=REPO_ROOT,
		env=env,
		text=True,
		capture_output=True,
		check=True,
		timeout=30,
	)

	assert 'base inventory path:' in completed.stdout
	assert 'number of source slices: 6' in completed.stdout
	assert 'execution: dry-run; split inventories skipped' in completed.stdout
	assert not (tmp_path / 'out').exists()


def _config(
	tmp_path: Path,
	*,
	min_validation_tokens_per_class: dict[str, int] | None = None,
):
	paths = _write_fixture(tmp_path)
	return f3_lithology_split_inventory_config_from_mapping(
		{
			'paths': {'artifact_root': str(tmp_path)},
			'inputs': {
				'base_png_label_inventory': str(paths['inventory']),
				'source_label_volume': str(paths['label_volume']),
				'segy_geometry_json': str(paths['geometry']),
				'class_info': str(paths['class_info']),
				'reference_embedding_metadata': str(paths['embedding_metadata']),
			},
			'split_sweep': {
				'name': 'split_index_fixture',
				'output_root': str(tmp_path / 'out'),
				'split_ids': ['split_000', 'split_001'],
				'random_seeds': [0, 1],
				'validation_slice_count': 2,
				'require_validation_all_classes': True,
				'min_validation_tokens_per_class': (
					min_validation_tokens_per_class or {'default': 1}
				),
				'include_base_split_as_split_000': True,
			},
			'tokenization': {
				'min_labeled_fraction': 0.5,
				'min_majority_fraction': 0.5,
				'ignore_z_border_samples': 0,
				'patch_size': [1, 1, 1],
			},
			'outputs': {'overwrite': False},
		},
	)


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
	inventory = tmp_path / 'png_label_inventory.csv'
	if not inventory.exists():
		_write_inventory(inventory)
	label_volume = tmp_path / 'f3_facies_labels.npy'
	if not label_volume.exists():
		np.save(label_volume, _label_volume())
	geometry = tmp_path / 'segy_geometry.json'
	if not geometry.exists():
		geometry.write_text(
			json.dumps(
				{
					'cube_shape': [6, 2, 1],
					'iline_min': 100,
					'iline_max': 105,
					'xline_min': 200,
					'xline_max': 201,
				},
			),
			encoding='utf-8',
		)
	class_info = tmp_path / 'class_info.json'
	if not class_info.exists():
		class_info.write_text(
			json.dumps(
				{
					'classes': [
						{'class_id': 0, 'class_name': 'zero', 'rgb': [0, 0, 0]},
						{'class_id': 1, 'class_name': 'one', 'rgb': [255, 255, 255]},
					],
				},
			),
			encoding='utf-8',
		)
	embedding_metadata = tmp_path / 'embedding_metadata.json'
	if not embedding_metadata.exists():
		embedding_metadata.write_text(
			json.dumps({'token_grid_shape': [6, 2, 1]}) + '\n',
			encoding='utf-8',
		)
	return {
		'inventory': inventory,
		'label_volume': label_volume,
		'geometry': geometry,
		'class_info': class_info,
		'embedding_metadata': embedding_metadata,
	}


def _write_inventory(path: Path) -> None:
	rows = [
		{
			'relative_path': f'labels/inline_{100 + offset}.png',
			'absolute_path': f'/fixture/inline_{100 + offset}.png',
			'split': 'validation' if offset in {0, 1} else 'train',
			'slice_type': 'inline',
			'slice_index': str(100 + offset),
			'note': f'row-{offset}',
		}
		for offset in range(6)
	]
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=tuple(rows[0].keys()))
		writer.writeheader()
		writer.writerows(rows)


def _label_volume() -> np.ndarray:
	volume = np.zeros((6, 2, 1), dtype=np.int32)
	volume[0, :, :] = 0
	volume[1, :, :] = 1
	volume[2, :, :] = np.asarray([[0], [1]], dtype=np.int32)
	volume[3, :, :] = 0
	volume[4, :, :] = 1
	volume[5, :, :] = np.asarray([[0], [1]], dtype=np.int32)
	return volume


def _read_rows(path: Path) -> list[dict[str, str]]:
	with path.open(encoding='utf-8', newline='') as file_obj:
		return list(csv.DictReader(file_obj))


def _config_yaml(paths: dict[str, Path], output_root: Path) -> str:
	return f"""
paths:
  artifact_root: {paths['inventory'].parent}
inputs:
  base_png_label_inventory: {paths['inventory']}
  source_label_volume: {paths['label_volume']}
  segy_geometry_json: {paths['geometry']}
  class_info: {paths['class_info']}
  reference_embedding_metadata: {paths['embedding_metadata']}
split_sweep:
  name: split_index_fixture
  output_root: {output_root}
  split_ids: [split_000, split_001]
  random_seeds: [0, 1]
  validation_slice_count: 2
  require_validation_all_classes: true
  min_validation_tokens_per_class:
    default: 1
  include_base_split_as_split_000: true
tokenization:
  min_labeled_fraction: 0.5
  min_majority_fraction: 0.5
  ignore_z_border_samples: 0
  patch_size: [1, 1, 1]
outputs:
  overwrite: false
"""
