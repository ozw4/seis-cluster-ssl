from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster.f3 import (
	CATEGORY_CLASS_INFO,
	CATEGORY_LABEL_PNG,
	CATEGORY_LABEL_SEGY,
	CATEGORY_SEISMIC_SEGY,
	F3InventoryOutputConfig,
	extract_label_split,
	parse_label_png_name,
	read_class_info,
	rgb_to_hex,
	scan_f3_file_inventory,
	write_f3_file_inventory_outputs,
)
from tests.helpers import run_python_proc


def test_f3_file_inventory_parses_class_info_and_label_pngs(
	tmp_path: Path,
) -> None:
	f3_root = _make_f3_fixture(tmp_path)

	inventory = scan_f3_file_inventory(f3_root)

	assert inventory.category_counts()[CATEGORY_SEISMIC_SEGY] == 1
	assert inventory.category_counts()[CATEGORY_LABEL_SEGY] == 1
	assert inventory.category_counts()[CATEGORY_CLASS_INFO] == 1
	assert inventory.category_counts()[CATEGORY_LABEL_PNG] == 2
	assert inventory.split_counts() == {'train': 1, 'validation': 1}
	assert inventory.classes[0].class_id == 0
	assert inventory.classes[0].class_name == 'Zero is a valid class'
	assert inventory.classes[0].hex_color == '#010203'

	png_records = {
		record.relative_path: record
		for record in inventory.label_png_files()
	}
	train_record = png_records['interpretation/train/0001_labels_inline_0250.PNG']
	assert train_record.split == 'train'
	assert train_record.slice_type == 'inline'
	assert train_record.slice_index == 250
	validation_record = png_records[
		'interpretation/validation/0008_labels_crossline_0450.png'
	]
	assert validation_record.split == 'validation'
	assert validation_record.slice_type == 'crossline'
	assert validation_record.slice_index == 450


def test_f3_class_info_and_label_helpers_are_stable(tmp_path: Path) -> None:
	f3_root = _make_f3_fixture(tmp_path)

	classes = read_class_info(f3_root / 'interpretation' / 'class_info.JSON')
	parts = parse_label_png_name('0008_labels_crossline_0450.png')

	assert [(item.class_id, item.class_name) for item in classes] == [
		(0, 'Zero is a valid class'),
		(1, 'Class one'),
	]
	assert rgb_to_hex((35, 92, 167)) == '#235CA7'
	assert parts.slice_type == 'crossline'
	assert parts.slice_index == 450
	assert extract_label_split(Path('interpretation/train/example.png')) == 'train'
	assert (
		extract_label_split(Path('interpretation/validation/example.png'))
		== 'validation'
	)


def test_f3_inventory_writes_required_outputs(tmp_path: Path) -> None:
	f3_root = _make_f3_fixture(tmp_path)
	inventory = scan_f3_file_inventory(f3_root)
	output_dir = tmp_path / 'artifacts' / 'inspection' / 'f3' / 'facies_benchmark_v1'
	outputs = F3InventoryOutputConfig(
		file_inventory_json=output_dir / 'inventory' / 'file_inventory.json',
		file_inventory_csv=output_dir / 'inventory' / 'file_inventory.csv',
		file_inventory_markdown=output_dir / 'inventory' / 'file_inventory.md',
		class_info_json=output_dir / 'inventory' / 'class_info.json',
		label_png_inventory_csv=(
			output_dir / 'inventory' / 'label_png_inventory.csv'
		),
	)

	write_f3_file_inventory_outputs(inventory, outputs)

	file_inventory = json.loads(
		outputs.file_inventory_json.read_text(encoding='utf-8'),
	)
	class_info = json.loads(outputs.class_info_json.read_text(encoding='utf-8'))
	summary = outputs.file_inventory_markdown.read_text(encoding='utf-8')
	with outputs.label_png_inventory_csv.open(encoding='utf-8', newline='') as file_obj:
		label_rows = list(csv.DictReader(file_obj))

	assert file_inventory['category_counts'][CATEGORY_LABEL_PNG] == 2
	assert class_info['classes'][0]['class_id'] == 0
	assert class_info['classes'][0]['class_name'] == 'Zero is a valid class'
	assert 'SEGYファイル' in summary
	assert 'class_infoのclass一覧' in summary
	assert 'train PNG枚数: 1' in summary
	assert 'validation PNG枚数: 1' in summary
	assert len(label_rows) == 2
	assert label_rows[0]['category'] == CATEGORY_LABEL_PNG


def test_f3_inventory_missing_class_info_has_clear_error(tmp_path: Path) -> None:
	f3_root = tmp_path / 'F3'
	f3_root.mkdir()
	(f3_root / 'f3_seismic.sgy').write_bytes(b'segy')

	with pytest.raises(FileNotFoundError, match=r'class_info\.json'):
		scan_f3_file_inventory(f3_root)


def test_f3_inventory_missing_root_has_clear_error(tmp_path: Path) -> None:
	with pytest.raises(FileNotFoundError, match='F3 root directory'):
		scan_f3_file_inventory(tmp_path / 'missing_F3')


def test_inspect_f3_files_proc_writes_inventory_outputs(tmp_path: Path) -> None:
	f3_root = _make_f3_fixture(tmp_path)
	artifact_root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	inspection_dir = artifact_root / 'inspection' / 'f3' / 'facies_benchmark_v1'
	inventory_dir = inspection_dir / 'inventory'
	config = {
		'paths': {
			'f3_root': str(f3_root),
			'artifact_root': str(artifact_root),
		},
		'outputs': {'inspection_dir': str(inspection_dir)},
		'dataset': {
			'name': 'f3_facies_benchmark',
			'version': 'facies_benchmark_v1',
		},
		'inspection': {
			'inventory_dir': str(inventory_dir),
			'output_json': str(inventory_dir / 'file_inventory.json'),
			'output_csv': str(inventory_dir / 'file_inventory.csv'),
			'output_markdown': str(inventory_dir / 'file_inventory.md'),
			'class_info_json': str(inventory_dir / 'class_info.json'),
			'label_png_inventory_csv': str(
				inventory_dir / 'label_png_inventory.csv',
			),
			'include_globs': ['**/*'],
			'exclude_globs': ['**/.DS_Store'],
			'hash_files': False,
		},
	}
	config_path = tmp_path / 'inspect_f3_files.yaml'
	config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/inspect_f3_files.py'),
		'--config',
		config_path,
	)

	assert result.returncode == 0, result.stderr
	assert (inventory_dir / 'file_inventory.json').is_file()
	assert (inventory_dir / 'file_inventory.md').is_file()
	assert (inventory_dir / 'class_info.json').is_file()
	assert (inventory_dir / 'label_png_inventory.csv').is_file()
	assert 'f3_inventory.file_count: 5' in result.stdout
	assert 'f3_inventory.label_png_count: 2' in result.stdout


def _make_f3_fixture(tmp_path: Path) -> Path:
	f3_root = tmp_path / 'F3'
	(f3_root / 'interpretation' / 'train').mkdir(parents=True)
	(f3_root / 'interpretation' / 'validation').mkdir(parents=True)
	(f3_root / 'f3_seismic.SGY').write_bytes(b'seismic')
	(f3_root / 'f3_labels.sgy').write_bytes(b'labels')
	(f3_root / 'interpretation' / 'class_info.JSON').write_text(
		json.dumps(
			{
				'0': {'name': 'Zero is a valid class', 'color': [1, 2, 3]},
				'1': {'name': 'Class one', 'color': [35, 92, 167]},
			},
		),
		encoding='utf-8',
	)
	(
		f3_root / 'interpretation' / 'train' / '0001_labels_inline_0250.PNG'
	).write_bytes(b'png')
	(
		f3_root
		/ 'interpretation'
		/ 'validation'
		/ '0008_labels_crossline_0450.png'
	).write_bytes(b'png')
	return f3_root
