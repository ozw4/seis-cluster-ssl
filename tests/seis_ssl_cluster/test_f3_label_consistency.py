from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.f3 import (
	F3ClassInfo,
	F3LabelConsistencyFigureConfig,
	F3LabelConsistencyOutputConfig,
	F3SegyFileInspection,
	F3SegyGeometry,
	F3SegyInspection,
	calculate_label_unique_values,
	calculate_seismic_amplitude_stats,
	check_f3_label_consistency,
	class_id_image_to_rgb,
	extract_teacher_label_slice,
	inspect_f3_png_labels,
	label_consistency_report_to_dict,
	write_f3_label_consistency_outputs,
)
from tests.helpers import run_python_proc


def test_label_consistency_detects_exact_png_segy_match(tmp_path: Path) -> None:
	labels = _label_cube()
	f3_root = _make_png_fixture(
		tmp_path,
		inline_ids=labels[2, :, :],
		crossline_ids=labels[:, 4, :],
	)
	report = check_f3_label_consistency(
		_segy_inspection(f3_root, labels),
		inspect_f3_png_labels(f3_root),
	)
	summary = label_consistency_report_to_dict(report)
	records = {record.output_prefix: record for record in report.records}

	assert report.passed is True
	assert summary['passed'] is True
	assert records['train_inline_0102'].matched_pixel_count == 15
	assert records['train_inline_0102'].mismatch_pixel_count == 0
	assert records['train_inline_0102'].mismatch_rate == 0.0
	assert records['train_inline_0102'].orientation == 'none'
	assert records['train_inline_0102'].segy_line_mapping.array_index == 2
	assert records['validation_crossline_0304'].segy_line_mapping.array_index == 4


def test_label_consistency_detects_one_pixel_mismatch(tmp_path: Path) -> None:
	labels = _label_cube()
	inline_png = labels[2, :, :].copy()
	inline_png[0, 0] = 1 if inline_png[0, 0] != 1 else 2
	f3_root = _make_png_fixture(tmp_path, inline_ids=inline_png)

	report = check_f3_label_consistency(
		_segy_inspection(f3_root, labels),
		inspect_f3_png_labels(f3_root),
		max_mismatch_rate=0.001,
	)
	record = report.records[0]

	assert report.passed is False
	assert record.matched_pixel_count == 14
	assert record.mismatch_pixel_count == 1
	assert record.mismatch_rate == pytest.approx(1 / 15)
	assert record.exceeds_threshold is True


def test_extract_teacher_label_slice_supports_inline_and_crossline() -> None:
	labels = _label_cube()
	geometry = _geometry(
		Path('f3_labels.sgy'),
		role='label',
		shape=labels.shape,
		iline_min=100,
		xline_min=300,
	)

	inline = extract_teacher_label_slice(
		labels,
		geometry,
		slice_type='inline',
		slice_index=102,
	)
	crossline = extract_teacher_label_slice(
		labels,
		geometry,
		slice_type='crossline',
		slice_index=304,
	)

	assert inline.resolved_line.array_index == 2
	assert np.array_equal(inline.values, labels[2, :, :])
	assert crossline.resolved_line.array_index == 4
	assert np.array_equal(crossline.values, labels[:, 4, :])


def test_label_consistency_reports_transpose_orientation(tmp_path: Path) -> None:
	labels = _label_cube()
	f3_root = _make_png_fixture(
		tmp_path,
		inline_ids=labels[2, :, :].T,
	)

	report = check_f3_label_consistency(
		_segy_inspection(f3_root, labels),
		inspect_f3_png_labels(f3_root),
	)
	record = report.records[0]

	assert report.passed is True
	assert record.png_label_shape == (3, 5)
	assert record.segy_slice_shape == (5, 3)
	assert record.orientation == 'transpose_png_to_segy'
	assert record.alignment is not None
	assert record.alignment.transpose is True
	assert record.mismatch_pixel_count == 0


def test_label_consistency_reports_unknown_png_color(tmp_path: Path) -> None:
	labels = _label_cube()
	inline_rgb = class_id_image_to_rgb(labels[2, :, :], _classes())
	inline_rgb[0, 0] = (255, 0, 255)
	f3_root = _make_png_fixture(tmp_path, inline_rgb=inline_rgb)

	report = check_f3_label_consistency(
		_segy_inspection(f3_root, labels),
		inspect_f3_png_labels(f3_root, allow_unknown_colors=True),
	)
	record = report.records[0]

	assert record.unknown_png_pixel_count == 1
	assert record.mismatch_pixel_count == 1
	assert record.mismatch_rate == pytest.approx(1 / 15)


def test_label_consistency_reports_unexpected_segy_label_values(
	tmp_path: Path,
) -> None:
	labels = _label_cube()
	segy_labels = labels.copy()
	segy_labels[2, 0, 0] = 99
	f3_root = _make_png_fixture(tmp_path, inline_ids=labels[2, :, :])

	report = check_f3_label_consistency(
		_segy_inspection(f3_root, segy_labels),
		inspect_f3_png_labels(f3_root),
	)
	record = report.records[0]

	assert record.unexpected_segy_label_values == (99,)
	assert record.mismatch_pixel_count == 1
	assert report.passed is False


def test_label_consistency_outputs_reports_and_qc_figures(tmp_path: Path) -> None:
	pytest.importorskip('matplotlib.pyplot')
	labels = _label_cube()
	f3_root = _make_png_fixture(tmp_path, inline_ids=labels[2, :, :])
	report = check_f3_label_consistency(
		_segy_inspection(f3_root, labels),
		inspect_f3_png_labels(f3_root),
	)
	outputs = F3LabelConsistencyOutputConfig(
		consistency_dir=tmp_path / 'quicklook' / 'consistency',
		output_json=tmp_path / 'labels' / 'label_consistency_report.json',
		output_csv=tmp_path / 'labels' / 'label_consistency_report.csv',
		report_path=tmp_path / 'labels' / 'label_consistency_report.md',
	)

	result = write_f3_label_consistency_outputs(
		report,
		outputs,
		F3LabelConsistencyFigureConfig(dpi=40),
	)

	with outputs.output_csv.open(encoding='utf-8', newline='') as file_obj:
		rows = list(csv.DictReader(file_obj))
	summary = json.loads(outputs.output_json.read_text(encoding='utf-8'))
	per_slice = json.loads(
		(
			outputs.consistency_dir / 'train_inline_0102_consistency.json'
		).read_text(encoding='utf-8'),
	)
	markdown = outputs.report_path.read_text(encoding='utf-8')

	assert len(rows) == 1
	assert rows[0]['mismatch_pixel_count'] == '0'
	assert summary['png_label_file_count'] == 1
	assert per_slice['result']['orientation'] == 'none'
	assert 'Per-slice results' in markdown
	assert outputs.consistency_dir.joinpath(
		'train_inline_0102_png_label.png',
	).is_file()
	assert outputs.consistency_dir.joinpath(
		'train_inline_0102_segy_label.png',
	).is_file()
	assert outputs.consistency_dir.joinpath(
		'train_inline_0102_mismatch.png',
	).is_file()
	assert result.metadata_json == outputs.output_json
	assert len(result.figure_paths) == 3


def test_check_f3_label_consistency_proc_dry_run(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	inspection_dir = artifact_root / 'inspection' / 'f3' / 'facies_benchmark_v1'
	config = {
		'paths': {
			'f3_root': str(tmp_path / 'F3'),
			'artifact_root': str(artifact_root),
		},
		'outputs': {'inspection_dir': str(inspection_dir)},
		'dataset': {
			'name': 'f3_facies_benchmark',
			'version': 'facies_benchmark_v1',
		},
		'inspection': {
			'segy_geometry_json': str(inspection_dir / 'segy' / 'segy_geometry.json'),
			'png_label_inventory_json': str(
				inspection_dir / 'labels' / 'png_label_inventory.json',
			),
			'consistency_dir': str(inspection_dir / 'quicklook' / 'consistency'),
			'output_json': str(
				inspection_dir / 'labels' / 'label_consistency_report.json',
			),
			'output_csv': str(
				inspection_dir / 'labels' / 'label_consistency_report.csv',
			),
			'report_path': str(
				inspection_dir / 'labels' / 'label_consistency_report.md',
			),
			'consistency': {'max_mismatch_rate': 0.001},
			'figure': {'dpi': 40, 'background': 'white', 'output_formats': ['png']},
		},
	}
	config_path = tmp_path / 'check_f3_label_consistency.yaml'
	config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/check_f3_label_consistency.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert 'stage: check_f3_label_consistency' in result.stdout
	assert 'inspection.consistency.max_mismatch_rate: 0.001' in result.stdout
	assert 'execution: dry-run; F3 label consistency check skipped' in result.stdout


def _classes() -> tuple[F3ClassInfo, ...]:
	return (
		F3ClassInfo(class_id=0, class_name='Class zero', rgb=(1, 2, 3)),
		F3ClassInfo(class_id=1, class_name='Class one', rgb=(35, 92, 167)),
		F3ClassInfo(class_id=2, class_name='Class two', rgb=(9, 8, 7)),
	)


def _label_cube() -> np.ndarray:
	return np.asarray(
		[
			[[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2], [1, 2, 0]],
			[[1, 2, 0], [2, 0, 1], [0, 1, 2], [1, 2, 0], [2, 0, 1]],
			[[2, 0, 1], [0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2]],
			[[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2], [1, 2, 0]],
		],
		dtype=np.int32,
	)


def _make_png_fixture(
	tmp_path: Path,
	*,
	inline_ids: np.ndarray | None = None,
	inline_rgb: np.ndarray | None = None,
	crossline_ids: np.ndarray | None = None,
) -> Path:
	f3_root = tmp_path / 'F3'
	(f3_root / 'interpretation' / 'train').mkdir(parents=True)
	(f3_root / 'interpretation' / 'validation').mkdir(parents=True)
	(f3_root / 'interpretation' / 'class_info.json').write_text(
		json.dumps(
			{
				'0': {'name': 'Class zero', 'color': [1, 2, 3]},
				'1': {'name': 'Class one', 'color': [35, 92, 167]},
				'2': {'name': 'Class two', 'color': [9, 8, 7]},
			},
		),
		encoding='utf-8',
	)
	if inline_ids is not None and inline_rgb is not None:
		msg = 'pass only one of inline_ids or inline_rgb'
		raise ValueError(msg)
	if inline_ids is not None:
		inline_rgb = class_id_image_to_rgb(inline_ids, _classes())
	if inline_rgb is not None:
		_write_png(
			f3_root / 'interpretation' / 'train' / '0001_labels_inline_0102.png',
			inline_rgb,
		)
	if crossline_ids is not None:
		_write_png(
			f3_root
			/ 'interpretation'
			/ 'validation'
			/ '0002_labels_crossline_0304.png',
			class_id_image_to_rgb(crossline_ids, _classes()),
		)
	return f3_root


def _segy_inspection(f3_root: Path, labels: np.ndarray) -> F3SegyInspection:
	classes = _classes()
	seismic = np.arange(labels.size, dtype=np.float32).reshape(labels.shape)
	seismic_geometry = _geometry(
		f3_root / 'f3_seismic.sgy',
		role='seismic',
		shape=labels.shape,
		iline_min=100,
		xline_min=300,
	)
	label_geometry = _geometry(
		f3_root / 'f3_labels.sgy',
		role='label',
		shape=labels.shape,
		iline_min=100,
		xline_min=300,
	)
	return F3SegyInspection(
		f3_root=f3_root,
		class_info_path=f3_root / 'interpretation' / 'class_info.json',
		classes=classes,
		seismic=F3SegyFileInspection(
			geometry=seismic_geometry,
			cube=seismic,
		),
		label=F3SegyFileInspection(
			geometry=label_geometry,
			cube=labels,
		),
		seismic_amplitude_stats=calculate_seismic_amplitude_stats(seismic),
		label_unique_values=calculate_label_unique_values(labels, classes),
	)


def _geometry(
	path: Path,
	*,
	role: str,
	shape: tuple[int, int, int],
	iline_min: int,
	xline_min: int,
) -> F3SegyGeometry:
	return F3SegyGeometry(
		role=role,
		path=path,
		file_size=0,
		iline_count=shape[0],
		xline_count=shape[1],
		sample_count=shape[2],
		iline_min=iline_min,
		iline_max=iline_min + shape[0] - 1,
		xline_min=xline_min,
		xline_max=xline_min + shape[1] - 1,
		sample_min=0,
		sample_max=shape[2] - 1,
		cube_shape=shape,
		dtype='float32',
	)


def _write_png(path: Path, image: np.ndarray) -> None:
	image_module = pytest.importorskip('matplotlib.image')
	path.parent.mkdir(parents=True, exist_ok=True)
	image_module.imsave(path, image)
