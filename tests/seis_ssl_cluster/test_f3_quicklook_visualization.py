from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.f3 import (
	F3ClassInfo,
	F3QuicklookFigureConfig,
	F3QuicklookOutputConfig,
	F3SegyFileInspection,
	F3SegyGeometry,
	F3SegyInspection,
	align_png_label_to_seismic_slice,
	calculate_label_unique_values,
	calculate_seismic_amplitude_stats,
	class_id_image_to_rgb,
	facies_legend_labels,
	inspect_f3_png_labels,
	make_orthogonal_display_slice,
	make_teacher_seismic_display_slice,
	write_f3_quicklook_outputs,
)
from tests.helpers import run_python_proc


def test_orthogonal_slices_use_expected_shapes_and_origins() -> None:
	cube = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

	xy = make_orthogonal_display_slice(cube, 'xy')
	xz = make_orthogonal_display_slice(cube, 'xz')
	yz = make_orthogonal_display_slice(cube, 'yz')

	assert xy.image.shape == (3, 2)
	assert xy.origin == 'lower'
	assert np.array_equal(xy.image, cube[:, :, 2].T)
	assert xz.image.shape == (4, 2)
	assert xz.origin == 'upper'
	assert np.array_equal(xz.image, cube[:, 1, :].T)
	assert yz.image.shape == (4, 3)
	assert yz.origin == 'upper'
	assert np.array_equal(yz.image, cube[1, :, :].T)


def test_teacher_slice_mapping_uses_contiguous_segy_line_coordinates() -> None:
	cube = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
	geometry = _geometry(
		Path('f3_seismic.sgy'),
		role='seismic',
		shape=cube.shape,
		iline_min=100,
		xline_min=300,
	)

	inline_slice, inline_mapping = make_teacher_seismic_display_slice(
		cube,
		geometry,
		slice_type='inline',
		slice_index=102,
	)
	crossline_slice, crossline_mapping = make_teacher_seismic_display_slice(
		cube,
		geometry,
		slice_type='crossline',
		slice_index=304,
	)

	assert inline_mapping.array_index == 2
	assert inline_mapping.resolution == 'contiguous_coordinate'
	assert inline_slice.image.shape == (3, 5)
	assert inline_slice.horizontal_axis == 'crossline index'
	assert np.array_equal(inline_slice.image, cube[2, :, :].T)
	assert crossline_mapping.array_index == 4
	assert crossline_slice.image.shape == (3, 4)
	assert crossline_slice.horizontal_axis == 'inline index'
	assert np.array_equal(crossline_slice.image, cube[:, 4, :].T)


def test_png_label_shape_alignment_records_none_and_transpose() -> None:
	label = np.zeros((3, 5, 3), dtype=np.uint8)
	transposed_label = np.zeros((5, 3, 3), dtype=np.uint8)

	none = align_png_label_to_seismic_slice(label, seismic_shape=(3, 5))
	transposed = align_png_label_to_seismic_slice(
		transposed_label,
		seismic_shape=(3, 5),
	)

	assert none.transform == 'none'
	assert none.rgb.shape == (3, 5, 3)
	assert transposed.transform == 'transpose'
	assert transposed.rgb.shape == (3, 5, 3)
	with pytest.raises(ValueError, match='does not match'):
		align_png_label_to_seismic_slice(label, seismic_shape=(4, 5))


def test_label_rgb_and_legend_use_class_info_colors() -> None:
	classes = _classes()
	label_slice = np.asarray([[0, 1], [2, 99]], dtype=np.int32)

	rgb = class_id_image_to_rgb(label_slice, classes)

	assert rgb[0, 0].tolist() == [1, 2, 3]
	assert rgb[0, 1].tolist() == [35, 92, 167]
	assert rgb[1, 0].tolist() == [9, 8, 7]
	assert rgb[1, 1].tolist() == [226, 226, 226]
	assert facies_legend_labels(classes) == (
		'0: Class zero',
		'1: Class one',
		'2: Class two',
	)


def test_f3_quicklook_outputs_pngs_and_json_sidecars(tmp_path: Path) -> None:
	pytest.importorskip('matplotlib.pyplot')
	f3_root = _make_png_fixture(tmp_path)
	png_labels = inspect_f3_png_labels(f3_root)
	segy = _segy_inspection(f3_root)
	outputs = F3QuicklookOutputConfig(
		quicklook_dir=tmp_path / 'quicklook',
		seismic_dir=tmp_path / 'quicklook' / 'seismic',
		labels_dir=tmp_path / 'quicklook' / 'labels',
		overlays_dir=tmp_path / 'quicklook' / 'overlays',
		metadata_json=tmp_path / 'stats' / 'quicklook_metadata.json',
	)

	result = write_f3_quicklook_outputs(
		segy,
		png_labels,
		outputs,
		F3QuicklookFigureConfig(dpi=40),
	)

	expected_paths = {
		outputs.seismic_dir / 'seismic_xy_z_mid.png',
		outputs.seismic_dir / 'seismic_xz_y_mid.png',
		outputs.seismic_dir / 'seismic_yz_x_mid.png',
		outputs.labels_dir / 'label_xy_z_mid.png',
		outputs.labels_dir / 'label_xz_y_mid.png',
		outputs.labels_dir / 'label_yz_x_mid.png',
		outputs.labels_dir / 'label_slices_train_contact_sheet.png',
		outputs.labels_dir / 'label_slices_validation_contact_sheet.png',
		outputs.overlays_dir / 'train_inline_0102_overlay.png',
		outputs.overlays_dir / 'validation_crossline_0304_overlay.png',
	}
	assert set(result.png_paths) == expected_paths
	for path in expected_paths:
		assert path.is_file()
		assert path.with_suffix('.json').is_file()
	overlay_metadata = json.loads(
		(outputs.overlays_dir / 'train_inline_0102_overlay.json').read_text(
			encoding='utf-8',
		),
	)
	transpose_metadata = json.loads(
		(outputs.overlays_dir / 'validation_crossline_0304_overlay.json').read_text(
			encoding='utf-8'
		),
	)
	contact_metadata = json.loads(
		(outputs.labels_dir / 'label_slices_train_contact_sheet.json').read_text(
			encoding='utf-8',
		),
	)
	summary = json.loads(outputs.metadata_json.read_text(encoding='utf-8'))

	assert isinstance(contact_metadata['source_png_labels'][0], str)
	assert contact_metadata['source_png_labels'][0].endswith('labels_inline_0102.png')
	assert isinstance(overlay_metadata['source_png_label'], str)
	assert isinstance(overlay_metadata['source_png_label_relative_path'], str)
	assert overlay_metadata['source_png_label'].endswith('labels_inline_0102.png')
	assert overlay_metadata['origin'] == 'upper'
	assert overlay_metadata['label_shape_alignment']['transform'] == 'none'
	assert overlay_metadata['segy_line_mapping']['array_index'] == 2
	assert transpose_metadata['label_shape_alignment']['transform'] == 'transpose'
	assert summary['png_label_file_count'] == 2
	assert len(summary['outputs']) == len(expected_paths)


def test_visualize_f3_quicklook_proc_dry_run(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	inspection_dir = artifact_root / 'inspection' / 'f3' / 'facies_benchmark_v1'
	quicklook_dir = inspection_dir / 'quicklook'
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
			'palette_json': str(inspection_dir / 'labels' / 'facies_palette.json'),
			'quicklook_dir': str(quicklook_dir),
			'seismic_dir': str(quicklook_dir / 'seismic'),
			'labels_dir': str(quicklook_dir / 'labels'),
			'overlays_dir': str(quicklook_dir / 'overlays'),
			'metadata_json': str(inspection_dir / 'stats' / 'quicklook_metadata.json'),
			'figure': {
				'dpi': 40,
				'seismic_cmap': 'gray',
				'clip_percentiles': [1.0, 99.0],
				'overlay_alpha': 0.45,
				'xz_yz_origin': 'upper',
			},
		},
	}
	config_path = tmp_path / 'visualize_f3_quicklook.yaml'
	config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/visualize_f3_quicklook.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert 'stage: visualize_f3_quicklook' in result.stdout
	assert 'inspection.figure.overlay_alpha: 0.45' in result.stdout
	assert 'execution: dry-run; F3 quicklook visualization skipped' in result.stdout


def _classes() -> tuple[F3ClassInfo, ...]:
	return (
		F3ClassInfo(class_id=0, class_name='Class zero', rgb=(1, 2, 3)),
		F3ClassInfo(class_id=1, class_name='Class one', rgb=(35, 92, 167)),
		F3ClassInfo(class_id=2, class_name='Class two', rgb=(9, 8, 7)),
	)


def _make_png_fixture(tmp_path: Path) -> Path:
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
	inline_labels = np.asarray(
		[
			[0, 1, 2, 0, 1],
			[1, 2, 0, 1, 2],
			[2, 0, 1, 2, 0],
		],
		dtype=np.int32,
	)
	crossline_labels = np.asarray(
		[
			[0, 1, 2, 0],
			[1, 2, 0, 1],
			[2, 0, 1, 2],
		],
		dtype=np.int32,
	)
	_write_png(
		f3_root / 'interpretation' / 'train' / '0001_labels_inline_0102.png',
		class_id_image_to_rgb(inline_labels, _classes()),
	)
	_write_png(
		f3_root / 'interpretation' / 'validation' / '0002_labels_crossline_0304.png',
		np.transpose(class_id_image_to_rgb(crossline_labels, _classes()), (1, 0, 2)),
	)
	return f3_root


def _segy_inspection(f3_root: Path) -> F3SegyInspection:
	classes = _classes()
	seismic = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
	labels = np.asarray(
		[
			[[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2], [1, 2, 0]],
			[[1, 2, 0], [2, 0, 1], [0, 1, 2], [1, 2, 0], [2, 0, 1]],
			[[2, 0, 1], [0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2]],
			[[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2], [1, 2, 0]],
		],
		dtype=np.float32,
	)
	seismic_geometry = _geometry(
		f3_root / 'f3_seismic.sgy',
		role='seismic',
		shape=seismic.shape,
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
