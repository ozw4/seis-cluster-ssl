from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.f3 import (
	F3ClassInfo,
	F3TokenizationConfig,
	F3TokenizationFigureConfig,
	F3TokenizationOutputConfig,
	class_id_image_to_rgb,
	inspect_f3_png_labels,
	load_f3_label_consistency_alignments,
	token_plane_spec,
	tokenize_label_slice,
	write_f3_tokenization_preview_outputs,
)
from tests.helpers import run_python_proc


def test_simple_2d_label_slice_majority_votes_tokens() -> None:
	labels = np.asarray(
		[
			[0, 0, 1, 1],
			[0, 1, 1, 1],
			[2, 2, 0, 0],
			[2, 2, 0, 1],
		],
		dtype=np.int32,
	)

	result = tokenize_label_slice(
		labels,
		slice_type='inline',
		slice_index=10,
		config=F3TokenizationConfig(
			patch_size_xyz=(4, 2, 2),
			min_labeled_fraction=1.0,
			min_majority_fraction=0.5,
		),
		classes=_classes(),
	)

	assert result.majority_class_ids.tolist() == [[0, 1], [2, 0]]
	assert result.retained_mask.tolist() == [[True, True], [True, True]]
	assert result.total_tokens == 4
	assert result.retained_tokens == 4
	assert result.class_counts_retained == {0: 2, 1: 1, 2: 1}
	assert result.plane.fixed_axis == 'x'
	assert result.plane.fixed_token_index == 2
	assert result.plane.row_axis == 'y'
	assert result.plane.column_axis == 'z'


def test_majority_fraction_condition_drops_mixed_token() -> None:
	labels = np.asarray([[0, 1], [1, 0]], dtype=np.int32)

	result = tokenize_label_slice(
		labels,
		slice_type='inline',
		slice_index=0,
		config=F3TokenizationConfig(
			patch_size_xyz=(2, 2, 2),
			min_labeled_fraction=1.0,
			min_majority_fraction=0.75,
		),
	)

	assert result.majority_class_ids.tolist() == [[0]]
	assert result.majority_fraction.tolist() == [[0.5]]
	assert result.retained_mask.tolist() == [[False]]
	assert result.ambiguous_token_count == 1
	assert result.empty_token_count == 0


def test_labeled_fraction_condition_drops_sparse_token() -> None:
	labels = np.asarray([[0, -1], [-1, -1]], dtype=np.int32)

	result = tokenize_label_slice(
		labels,
		slice_type='inline',
		slice_index=0,
		config=F3TokenizationConfig(
			patch_size_xyz=(2, 2, 2),
			min_labeled_fraction=0.5,
			min_majority_fraction=1.0,
		),
	)

	assert result.labeled_fraction.tolist() == [[0.25]]
	assert result.retained_mask.tolist() == [[False]]
	assert result.ambiguous_token_count == 1
	assert result.empty_token_count == 0


def test_empty_token_is_counted_separately_from_ambiguous() -> None:
	result = tokenize_label_slice(
		np.full((2, 2), -1, dtype=np.int32),
		slice_type='inline',
		slice_index=0,
		config=F3TokenizationConfig(
			patch_size_xyz=(2, 2, 2),
			min_labeled_fraction=0.5,
			min_majority_fraction=0.7,
		),
	)

	assert result.retained_mask.tolist() == [[False]]
	assert result.ambiguous_token_count == 0
	assert result.empty_token_count == 1


def test_class_zero_is_valid_and_retained() -> None:
	result = tokenize_label_slice(
		np.zeros((2, 2), dtype=np.int32),
		slice_type='inline',
		slice_index=0,
		config=F3TokenizationConfig(
			patch_size_xyz=(2, 2, 2),
			min_labeled_fraction=1.0,
			min_majority_fraction=1.0,
		),
		classes=_classes(),
	)

	assert result.retained_tokens == 1
	assert result.class_counts_retained == {0: 1, 1: 0, 2: 0}


def test_inline_and_crossline_fixed_axis_mapping() -> None:
	inline = token_plane_spec(
		slice_type='inline',
		slice_index=16,
		patch_size_xyz=(8, 4, 2),
	)
	offset_inline = token_plane_spec(
		slice_type='inline',
		slice_index=250,
		array_index=2,
		patch_size_xyz=(8, 4, 2),
	)
	crossline = token_plane_spec(
		slice_type='crossline',
		slice_index=10,
		patch_size_xyz=(8, 4, 2),
	)
	offset_crossline = token_plane_spec(
		slice_type='crossline',
		slice_index=450,
		array_index=10,
		patch_size_xyz=(8, 4, 2),
	)

	assert inline.fixed_axis == 'x'
	assert inline.fixed_token_index == 2
	assert inline.row_axis == 'y'
	assert inline.row_patch_size == 4
	assert offset_inline.slice_index == 250
	assert offset_inline.array_index == 2
	assert offset_inline.fixed_token_index == 0
	assert crossline.fixed_axis == 'y'
	assert crossline.fixed_token_index == 2
	assert crossline.row_axis == 'x'
	assert crossline.row_patch_size == 8
	assert offset_crossline.slice_index == 450
	assert offset_crossline.array_index == 10
	assert offset_crossline.fixed_token_index == 2


def test_tokenization_preview_outputs_figures_and_summaries(tmp_path: Path) -> None:
	pytest.importorskip('matplotlib.pyplot')
	f3_root = _make_png_fixture(tmp_path)
	png_labels = inspect_f3_png_labels(f3_root)
	alignments_path = _write_label_consistency_json(
		tmp_path,
		relative_paths=[item.relative_path for item in png_labels.files],
	)
	outputs = F3TokenizationOutputConfig(
		tokenization_dir=tmp_path / 'quicklook' / 'tokenization',
		metadata_json=tmp_path / 'stats' / 'tokenization_preview.json',
		summary_csv=tmp_path / 'stats' / 'tokenization_summary.csv',
		summary_markdown=tmp_path / 'stats' / 'tokenization_summary.md',
	)

	result = write_f3_tokenization_preview_outputs(
		png_labels,
		outputs,
		F3TokenizationConfig(
			patch_size_xyz=(2, 2, 2),
			min_labeled_fraction=1.0,
			min_majority_fraction=0.5,
		),
		load_f3_label_consistency_alignments(alignments_path),
		F3TokenizationFigureConfig(dpi=40),
	)

	with outputs.summary_csv.open(encoding='utf-8', newline='') as file_obj:
		rows = list(csv.DictReader(file_obj))
	metadata = json.loads(outputs.metadata_json.read_text(encoding='utf-8'))
	sidecar = json.loads(result.sidecar_paths[0].read_text(encoding='utf-8'))
	markdown = outputs.summary_markdown.read_text(encoding='utf-8')

	assert len(result.png_paths) == 1
	assert result.png_paths[0].is_file()
	assert result.sidecar_paths[0].is_file()
	assert rows[0]['slice_type'] == 'inline'
	assert rows[0]['total_tokens'] == '4'
	assert rows[0]['retained_tokens'] == '4'
	assert json.loads(rows[0]['class_counts_retained'])['0'] == 2
	assert metadata['png_label_file_count'] == 1
	assert metadata['outputs'][0]['summary']['retained_fraction'] == 1.0
	assert metadata['outputs'][0]['token_plane']['slice_index'] == 10
	assert metadata['outputs'][0]['token_plane']['array_index'] == 2
	assert metadata['outputs'][0]['token_plane']['fixed_token_index'] == 1
	assert sidecar['figure_type'] == 'teacher_slice_tokenization_preview'
	assert 'Per-slice results' in markdown


def test_preview_f3_tokenization_proc_dry_run(tmp_path: Path) -> None:
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
			'label_consistency_json': str(
				inspection_dir / 'stats' / 'label_consistency.json',
			),
			'tokenization_dir': str(inspection_dir / 'quicklook' / 'tokenization'),
			'metadata_json': str(
				inspection_dir / 'stats' / 'tokenization_preview.json',
			),
			'summary_csv': str(
				inspection_dir / 'stats' / 'tokenization_summary.csv',
			),
			'summary_markdown': str(
				inspection_dir / 'stats' / 'tokenization_summary.md',
			),
			'tokenization': {
				'patch_size_xyz': [8, 8, 8],
				'min_labeled_fraction': 0.5,
				'min_majority_fraction': 0.7,
			},
			'figure': {
				'dpi': 40,
				'background': 'white',
				'output_formats': ['png'],
			},
		},
	}
	config_path = tmp_path / 'preview_f3_tokenization.yaml'
	config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/preview_f3_tokenization.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert 'stage: preview_f3_tokenization' in result.stdout
	assert 'inspection.tokenization.patch_size_xyz: [8, 8, 8]' in result.stdout
	assert 'execution: dry-run; F3 tokenization preview skipped' in result.stdout


def _classes() -> tuple[F3ClassInfo, ...]:
	return (
		F3ClassInfo(class_id=0, class_name='Class zero', rgb=(1, 2, 3)),
		F3ClassInfo(class_id=1, class_name='Class one', rgb=(35, 92, 167)),
		F3ClassInfo(class_id=2, class_name='Class two', rgb=(9, 8, 7)),
	)


def _make_png_fixture(tmp_path: Path) -> Path:
	f3_root = tmp_path / 'F3'
	(f3_root / 'interpretation' / 'train').mkdir(parents=True)
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
	labels = np.asarray(
		[
			[0, 0, 1, 1],
			[0, 1, 1, 1],
			[2, 2, 0, 0],
			[2, 2, 0, 1],
		],
		dtype=np.int32,
	)
	_write_png(
		f3_root / 'interpretation' / 'train' / '0001_labels_inline_0010.png',
		class_id_image_to_rgb(labels, _classes()),
	)
	return f3_root


def _write_label_consistency_json(
	tmp_path: Path,
	*,
	relative_paths: list[str],
) -> Path:
	path = tmp_path / 'stats' / 'label_consistency.json'
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(
			{
				'files': [
					{
						'relative_path': relative_path,
						'segy_line_mapping': {
							'slice_type': 'inline',
							'slice_index': 10,
							'array_index': 2,
							'axis_name': 'inline',
							'axis_count': 8,
							'coordinate_min': 8,
							'coordinate_max': 15,
							'resolution': 'coordinate_offset',
						},
						'label_shape_alignment': {
							'source_shape': [4, 4],
							'output_shape': [4, 4],
							'transform': 'none',
							'transpose': False,
						},
					}
					for relative_path in relative_paths
				],
			},
		),
		encoding='utf-8',
	)
	return path


def _write_png(path: Path, image: np.ndarray) -> None:
	image_module = pytest.importorskip('matplotlib.image')
	path.parent.mkdir(parents=True, exist_ok=True)
	image_module.imsave(path, image)
