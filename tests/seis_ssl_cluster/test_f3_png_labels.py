from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.f3 import (
	F3ClassInfo,
	F3PngLabelOutputConfig,
	count_class_pixels,
	extract_label_split,
	inspect_f3_png_labels,
	parse_label_png_name,
	png_label_inspection_to_dict,
	rgb_to_class_id_map,
	write_f3_png_label_inspection_outputs,
)
from tests.helpers import run_python_proc


def test_rgb_to_class_id_map_matches_class_info_and_preserves_class_zero() -> None:
	classes = _classes()
	image = np.asarray(
		[
			[(1, 2, 3), (35, 92, 167)],
			[(1, 2, 3), (9, 8, 7)],
		],
		dtype=np.uint8,
	)

	result = rgb_to_class_id_map(image, classes)
	counts = count_class_pixels(result.class_id_map, classes)

	assert result.class_id_map.tolist() == [[0, 1], [0, 2]]
	assert [(item.class_id, item.pixel_count) for item in counts] == [
		(0, 2),
		(1, 1),
		(2, 1),
	]
	assert result.unknown_pixel_count == 0


def test_rgb_to_class_id_map_detects_unknown_colors() -> None:
	classes = _classes()
	image = np.asarray(
		[
			[(1, 2, 3), (255, 0, 255)],
			[(255, 0, 255), (35, 92, 167)],
		],
		dtype=np.uint8,
	)

	with pytest.raises(ValueError, match='absent from class_info'):
		rgb_to_class_id_map(image, classes)

	result = rgb_to_class_id_map(image, classes, allow_unknown_colors=True)

	assert result.class_id_map.tolist() == [[0, -1], [-1, 1]]
	assert result.unknown_pixel_count == 2
	assert result.unknown_colors[0].rgb == (255, 0, 255)
	assert result.unknown_colors[0].hex_color == '#FF00FF'


def test_png_filename_metadata_parses_split_type_and_index() -> None:
	parts = parse_label_png_name('0008_labels_crossline_0450.png')

	assert extract_label_split(Path('interpretation/train/example.png')) == 'train'
	assert parts.slice_type == 'crossline'
	assert parts.slice_index == 450


def test_inspect_f3_png_labels_counts_per_slice_and_unknowns(
	tmp_path: Path,
) -> None:
	f3_root = _make_f3_png_fixture(tmp_path, include_unknown=True)

	result = inspect_f3_png_labels(f3_root, allow_unknown_colors=True)
	summary = png_label_inspection_to_dict(result)
	records = {item.relative_path: item for item in result.files}
	train = records['interpretation/train/0001_labels_inline_0250.PNG']
	validation = records[
		'interpretation/validation/0008_labels_crossline_0450.png'
	]

	assert len(result.files) == 2
	assert train.split == 'train'
	assert train.slice_type == 'inline'
	assert train.slice_index == 250
	assert train.class_count_by_id() == {0: 2, 1: 1, 2: 1}
	assert validation.split == 'validation'
	assert validation.slice_type == 'crossline'
	assert validation.slice_index == 450
	assert validation.class_count_by_id() == {0: 1, 1: 1, 2: 1}
	assert validation.unknown_pixel_count == 1
	assert result.total_unknown_pixel_count() == 1
	assert result.unknown_colors()[0].rgb == (255, 0, 255)
	assert summary['splits']['train']['total_pixels'] == 4
	assert summary['total_unknown_pixels'] == 1


def test_f3_png_label_outputs_and_figures_are_written(tmp_path: Path) -> None:
	f3_root = _make_f3_png_fixture(tmp_path)
	result = inspect_f3_png_labels(f3_root)
	outputs = _outputs(tmp_path / 'inspection')

	write_f3_png_label_inspection_outputs(result, outputs)

	with outputs.inventory_csv.open(encoding='utf-8', newline='') as file_obj:
		inventory_rows = list(csv.DictReader(file_obj))
	with outputs.class_counts_csv.open(encoding='utf-8', newline='') as file_obj:
		count_rows = list(csv.DictReader(file_obj))
	summary = json.loads(outputs.summary_json.read_text(encoding='utf-8'))
	markdown = outputs.summary_markdown.read_text(encoding='utf-8')

	assert len(inventory_rows) == 2
	assert inventory_rows[0]['split'] == 'train'
	assert any(
		row['scope'] == 'per_png_file'
		and row['class_id'] == '0'
		and row['pixel_count'] == '2'
		for row in count_rows
	)
	assert summary['total_pixels'] == 8
	assert 'Overall class distribution' in markdown
	assert outputs.class_distribution_train_png.is_file()
	assert outputs.class_distribution_validation_png.is_file()
	assert outputs.class_distribution_per_slice_png.is_file()


def test_inspect_f3_png_labels_proc_writes_outputs(tmp_path: Path) -> None:
	f3_root = _make_f3_png_fixture(tmp_path)
	artifact_root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	inspection_dir = artifact_root / 'inspection' / 'f3' / 'facies_benchmark_v1'
	labels_dir = inspection_dir / 'labels'
	stats_dir = inspection_dir / 'stats'
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
			'candidate_extensions': ['.png'],
			'allow_unknown_colors': False,
			'inventory_csv': str(labels_dir / 'png_label_inventory.csv'),
			'class_counts_csv': str(labels_dir / 'png_label_class_counts.csv'),
			'summary_json': str(labels_dir / 'png_label_summary.json'),
			'summary_markdown': str(labels_dir / 'png_label_summary.md'),
			'class_distribution_train_png': str(
				stats_dir / 'class_distribution_train.png',
			),
			'class_distribution_validation_png': str(
				stats_dir / 'class_distribution_validation.png',
			),
			'class_distribution_per_slice_png': str(
				stats_dir / 'class_distribution_per_slice.png',
			),
			'figure_dpi': 90,
		},
	}
	config_path = tmp_path / 'inspect_f3_png_labels.yaml'
	config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/inspect_f3_png_labels.py'),
		'--config',
		config_path,
		extra_env={'MPLCONFIGDIR': str(tmp_path / 'matplotlib')},
	)

	assert result.returncode == 0, result.stderr
	assert (labels_dir / 'png_label_inventory.csv').is_file()
	assert (labels_dir / 'png_label_class_counts.csv').is_file()
	assert (labels_dir / 'png_label_summary.json').is_file()
	assert (labels_dir / 'png_label_summary.md').is_file()
	assert (stats_dir / 'class_distribution_train.png').is_file()
	assert 'f3_png_labels.file_count: 2' in result.stdout
	assert 'f3_png_labels.unknown_pixels: 0' in result.stdout


def _classes() -> tuple[F3ClassInfo, ...]:
	return (
		F3ClassInfo(class_id=0, class_name='Zero is valid', rgb=(1, 2, 3)),
		F3ClassInfo(class_id=1, class_name='Class one', rgb=(35, 92, 167)),
		F3ClassInfo(class_id=2, class_name='Class two', rgb=(9, 8, 7)),
	)


def _make_f3_png_fixture(
	tmp_path: Path,
	*,
	include_unknown: bool = False,
) -> Path:
	f3_root = tmp_path / 'F3'
	(f3_root / 'interpretation' / 'train').mkdir(parents=True)
	(f3_root / 'interpretation' / 'validation').mkdir(parents=True)
	(f3_root / 'interpretation' / 'class_info.json').write_text(
		json.dumps(
			{
				'0': {'name': 'Zero is valid', 'color': [1, 2, 3]},
				'1': {'name': 'Class one', 'color': [35, 92, 167]},
				'2': {'name': 'Class two', 'color': [9, 8, 7]},
			},
		),
		encoding='utf-8',
	)
	_write_png(
		f3_root / 'interpretation' / 'train' / '0001_labels_inline_0250.PNG',
		np.asarray(
			[
				[(1, 2, 3), (35, 92, 167)],
				[(1, 2, 3), (9, 8, 7)],
			],
			dtype=np.uint8,
		),
	)
	unknown_or_class = (255, 0, 255) if include_unknown else (1, 2, 3)
	_write_png(
		f3_root
		/ 'interpretation'
		/ 'validation'
		/ '0008_labels_crossline_0450.png',
		np.asarray(
			[
				[(1, 2, 3), (35, 92, 167)],
				[unknown_or_class, (9, 8, 7)],
			],
			dtype=np.uint8,
		),
	)
	return f3_root


def _write_png(path: Path, image: np.ndarray) -> None:
	image_module = pytest.importorskip('matplotlib.image')
	path.parent.mkdir(parents=True, exist_ok=True)
	image_module.imsave(path, image)


def _outputs(root: Path) -> F3PngLabelOutputConfig:
	return F3PngLabelOutputConfig(
		inventory_csv=root / 'labels' / 'png_label_inventory.csv',
		class_counts_csv=root / 'labels' / 'png_label_class_counts.csv',
		summary_json=root / 'labels' / 'png_label_summary.json',
		summary_markdown=root / 'labels' / 'png_label_summary.md',
		class_distribution_train_png=(
			root / 'stats' / 'class_distribution_train.png'
		),
		class_distribution_validation_png=(
			root / 'stats' / 'class_distribution_validation.png'
		),
		class_distribution_per_slice_png=(
			root / 'stats' / 'class_distribution_per_slice.png'
		),
		dpi=90,
	)
