from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.f3 import (
	F3ClassInfo,
	F3LineGeometry,
	F3LithologyTokenDatasetConfig,
	F3LithologyTokenDatasetInputs,
	F3LithologyTokenDatasetOutputs,
	F3LithologyTokenPolicy,
	F3SliceSplitRecord,
	build_f3_lithology_token_dataset,
	f3_slice_split_manifest,
	lithology_tokens,
	load_f3_slice_split_records,
	resolve_f3_slice_array_index,
	tokenize_f3_lithology_slice,
)
from tests.helpers import run_python_proc


def test_inline_and_crossline_slice_numbers_resolve_to_internal_indices() -> None:
	geometry = F3LineGeometry(
		shape_xyz=(6, 8, 4),
		inline_min=100,
		inline_max=105,
		crossline_min=300,
		crossline_max=307,
	)

	inline = resolve_f3_slice_array_index(
		F3SliceSplitRecord(
			relative_path='interpretation/train/a_labels_inline_0102.png',
			split='train',
			slice_type='inline',
			slice_index=102,
		),
		geometry,
	)
	crossline = resolve_f3_slice_array_index(
		F3SliceSplitRecord(
			relative_path='interpretation/validation/b_labels_crossline_0304.png',
			split='validation',
			slice_type='crossline',
			slice_index=304,
		),
		geometry,
	)

	assert inline == 2
	assert crossline == 4


def test_token_majority_labels_and_class_zero_are_retained() -> None:
	labels = np.zeros((4, 4, 4), dtype=np.int32)
	labels[1, :, :] = np.asarray(
		[
			[0, 0, 1, 1],
			[0, 1, 1, 1],
			[2, 2, 0, 0],
			[2, 2, 0, 1],
		],
		dtype=np.int32,
	)
	valid_tokens = np.ones((2, 2, 2), dtype=np.bool_)
	record = F3SliceSplitRecord(
		relative_path='interpretation/train/labels_inline_0101.png',
		split='train',
		slice_type='inline',
		slice_index=101,
	)

	result = tokenize_f3_lithology_slice(
		record,
		label_volume=labels,
		valid_tokens=valid_tokens,
		geometry=_geometry(),
		patch_size_xyz=(2, 2, 2),
		policy=F3LithologyTokenPolicy(
			min_labeled_fraction=1.0,
			min_majority_fraction=0.5,
			ignore_z_border_samples=0,
		),
		classes=_classes(),
	)

	assert result.array_index == 1
	assert result.tokenization.majority_class_ids.tolist() == [[0, 1], [2, 0]]
	assert result.usable_mask.tolist() == [[True, True], [True, True]]
	assert result.retained_tokens == 4
	assert result.to_summary_dict()['class_counts_retained'] == {
		'0': 2,
		'1': 1,
		'2': 1,
	}


def test_majority_and_labeled_fraction_drop_tokens() -> None:
	labels = np.full((2, 2, 2), -1, dtype=np.int32)
	labels[0, :, :] = np.asarray([[0, 1], [1, -1]], dtype=np.int32)
	valid_tokens = np.ones((1, 1, 1), dtype=np.bool_)
	record = F3SliceSplitRecord(
		relative_path='interpretation/train/labels_inline_0100.png',
		split='train',
		slice_type='inline',
		slice_index=100,
	)

	sparse = tokenize_f3_lithology_slice(
		record,
		label_volume=labels,
		valid_tokens=valid_tokens,
		geometry=F3LineGeometry(
			shape_xyz=(2, 2, 2),
			inline_min=100,
			inline_max=101,
			crossline_min=300,
			crossline_max=301,
		),
		patch_size_xyz=(2, 2, 2),
		policy=F3LithologyTokenPolicy(
			min_labeled_fraction=1.0,
			min_majority_fraction=0.5,
			ignore_z_border_samples=0,
		),
	)
	ambiguous = tokenize_f3_lithology_slice(
		record,
		label_volume=np.asarray([[[0, 1], [1, 0]], [[0, 0], [0, 0]]]),
		valid_tokens=valid_tokens,
		geometry=F3LineGeometry(
			shape_xyz=(2, 2, 2),
			inline_min=100,
			inline_max=101,
			crossline_min=300,
			crossline_max=301,
		),
		patch_size_xyz=(2, 2, 2),
		policy=F3LithologyTokenPolicy(
			min_labeled_fraction=1.0,
			min_majority_fraction=0.75,
			ignore_z_border_samples=0,
		),
	)

	assert sparse.tokenization.labeled_fraction.tolist() == [[0.75]]
	assert sparse.usable_mask.tolist() == [[False]]
	assert ambiguous.tokenization.majority_fraction.tolist() == [[0.5]]
	assert ambiguous.usable_mask.tolist() == [[False]]


def test_z_border_ignore_excludes_edge_samples_from_majority() -> None:
	labels = np.zeros((2, 2, 4), dtype=np.int32)
	labels[0, :, :] = np.asarray(
		[
			[1, 2, 2, 1],
			[1, 2, 2, 1],
		],
		dtype=np.int32,
	)
	record = F3SliceSplitRecord(
		relative_path='interpretation/train/labels_inline_0100.png',
		split='train',
		slice_type='inline',
		slice_index=100,
	)

	result = tokenize_f3_lithology_slice(
		record,
		label_volume=labels,
		valid_tokens=np.ones((1, 1, 1), dtype=np.bool_),
		geometry=F3LineGeometry(
			shape_xyz=(2, 2, 4),
			inline_min=100,
			inline_max=101,
			crossline_min=300,
			crossline_max=301,
		),
		patch_size_xyz=(2, 2, 4),
		policy=F3LithologyTokenPolicy(
			min_labeled_fraction=0.5,
			min_majority_fraction=1.0,
			ignore_z_border_samples=1,
		),
	)

	assert result.tokenization.majority_class_ids.tolist() == [[2]]
	assert result.tokenization.labeled_fraction.tolist() == [[0.5]]
	assert result.usable_mask.tolist() == [[True]]


def test_png_inventory_split_manifest_is_slice_level_and_not_random(
	tmp_path: Path,
) -> None:
	inventory = _write_inventory_csv(tmp_path)

	records = load_f3_slice_split_records(inventory)
	manifest = f3_slice_split_manifest(records)

	assert [record.split for record in records] == ['train', 'validation']
	assert manifest['split_unit'] == 'slice'
	assert manifest['no_random_split'] is True
	assert manifest['splits']['train'][0]['slice_type'] == 'inline'
	assert manifest['splits']['validation'][0]['slice_type'] == 'crossline'


def test_build_f3_lithology_token_dataset_outputs_npz_metadata_and_figures(
	tmp_path: Path,
) -> None:
	pytest.importorskip('matplotlib.pyplot')
	config = _write_dataset_fixture(tmp_path)

	result = build_f3_lithology_token_dataset(config)

	train = np.load(result.train_npz)
	validation = np.load(result.validation_npz)
	all_tokens = np.load(result.all_labeled_npz)
	metadata = json.loads(result.metadata_json.read_text(encoding='utf-8'))
	splits = json.loads(result.split_manifest_json.read_text(encoding='utf-8'))
	with result.class_counts_csv.open(encoding='utf-8', newline='') as file_obj:
		count_rows = list(csv.DictReader(file_obj))
	summary = result.summary_markdown.read_text(encoding='utf-8')

	assert result.train_token_count == train['labels'].shape[0]
	assert result.validation_token_count == validation['labels'].shape[0]
	assert train['features'].dtype == np.float32
	assert train['labels'].dtype == np.int64
	assert train['survey_id'][0] == 'f3_facies_benchmark'
	assert train['slice_type'][0] == 'inline'
	assert train['token_xyz'].shape[1] == 3
	assert train['voxel_center_xyz'].shape[1] == 3
	assert all_tokens['features'].shape[0] == (
		train['features'].shape[0] + validation['features'].shape[0]
	)
	assert metadata['label_source_of_truth'] == 'segy_label_volume'
	assert metadata['png_label_role'] == (
		'train_validation_slice_selection_and_visual_qc'
	)
	assert metadata['feature_source'] == _feature_source()
	assert metadata['embedding']['patch_size_xyz'] == [2, 2, 2]
	assert metadata['no_random_split'] is True
	assert splits['no_random_split'] is True
	assert any(row['split'] == 'train' and row['class_id'] == '0' for row in count_rows)
	assert 'no random token split' in summary
	assert (
		config.outputs.quicklook_dir / 'train_inline_0101_token_labels.png'
	).is_file()
	assert (
		config.outputs.quicklook_dir / 'validation_crossline_0302_token_labels.png'
	).is_file()


def test_build_f3_lithology_token_dataset_removes_train_tokens_that_overlap_validation(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _write_dataset_fixture(tmp_path)
	monkeypatch.setattr(
		lithology_tokens,
		'write_f3_lithology_token_quicklooks',
		lambda *_args, **_kwargs: (),
	)

	result = build_f3_lithology_token_dataset(config)

	train = np.load(result.train_npz)
	validation = np.load(result.validation_npz)
	metadata = json.loads(result.metadata_json.read_text(encoding='utf-8'))
	train_token_xyz = _token_xyz_set(train)
	validation_token_xyz = _token_xyz_set(validation)

	assert {(0, 1, 0), (0, 1, 1)} <= validation_token_xyz
	assert {(0, 1, 0), (0, 1, 1)}.isdisjoint(train_token_xyz)
	assert train_token_xyz & validation_token_xyz == set()
	assert train['labels'].shape[0] == 2
	assert validation['labels'].shape[0] == 4
	assert metadata['split_strategy'] == (
		'png_label_inventory_slice_split_no_random_token_split'
	)
	assert metadata['cross_split_token_overlap_resolution'] == {
		'strategy': 'validation_precedence_remove_train_duplicates',
		'overlap_token_xyz_count': 2,
		'train_rows_removed': 2,
		'validation_rows_removed': 0,
	}
	assert metadata['summary']['cross_split_duplicate_rows_removed_from_train'] == 2


def test_build_f3_lithology_token_dataset_proc_dry_run(tmp_path: Path) -> None:
	config = _write_dataset_fixture(tmp_path)
	config_path = tmp_path / 'build_f3_lithology_token_dataset.yaml'
	config_path.write_text(
		yaml.safe_dump(_config_mapping(config)),
		encoding='utf-8',
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/build_f3_lithology_token_dataset.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert 'stage: build_f3_lithology_token_dataset' in result.stdout
	assert 'token_dataset.patch_size_source: embedding metadata' in result.stdout
	assert 'execution: dry-run; F3 lithology token dataset build skipped' in (
		result.stdout
	)


def _write_dataset_fixture(tmp_path: Path) -> F3LithologyTokenDatasetConfig:
	artifact_root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	f3_root = tmp_path / 'F3'
	label_volume = _label_volume()
	seismic_volume = np.arange(label_volume.size, dtype=np.float32).reshape(
		label_volume.shape,
	)
	label_path = artifact_root / 'registry' / 'volumes' / 'f3_facies_labels.npy'
	seismic_path = artifact_root / 'registry' / 'volumes' / 'f3_seismic.npy'
	label_path.parent.mkdir(parents=True)
	np.save(label_path, label_volume)
	np.save(seismic_path, seismic_volume)
	inventory = _write_inventory_csv(tmp_path)
	class_info = _write_class_info(tmp_path)
	geometry = _write_geometry_json(tmp_path)
	embeddings_dir = artifact_root / 'embeddings' / 'f3'
	_write_embedding_artifacts(embeddings_dir)
	output_dir = (
		artifact_root
		/ 'lithology'
		/ 'f3'
		/ 'facies_benchmark_v1'
		/ 'model'
		/ 'embed'
		/ 'labels'
		/ 'token_dataset'
	)
	return F3LithologyTokenDatasetConfig(
		inputs=F3LithologyTokenDatasetInputs(
			embeddings_dir=embeddings_dir,
			label_volume=label_path,
			seismic_volume=seismic_path,
			png_label_inventory=inventory,
			class_info=class_info,
			segy_geometry_json=geometry,
			source_label_segy=f3_root / 'f3_labels.sgy',
			volume_metadata_json=label_path.with_name('f3_metadata.json'),
		),
		outputs=F3LithologyTokenDatasetOutputs(
			output_dir=output_dir,
			metadata_json=output_dir / 'token_dataset_metadata.json',
			class_counts_csv=output_dir / 'class_counts.csv',
			summary_markdown=output_dir / 'token_dataset_summary.md',
			split_manifest_json=output_dir / 'splits.json',
			quicklook_dir=output_dir / 'quicklook',
		),
		policy=F3LithologyTokenPolicy(
			min_labeled_fraction=1.0,
			min_majority_fraction=0.5,
			ignore_z_border_samples=0,
		),
		dataset={'name': 'f3_facies_benchmark', 'version': 'facies_benchmark_v1'},
		model={'tag': 'model', 'freeze_encoder': True},
		figure_dpi=40,
		feature_source=_feature_source(),
	)


def _token_xyz_set(dataset: object) -> set[tuple[int, int, int]]:
	return {tuple(int(axis) for axis in row) for row in dataset['token_xyz']}


def _config_mapping(config: F3LithologyTokenDatasetConfig) -> dict[str, object]:
	return {
		'paths': {
			'f3_root': str(config.inputs.source_label_segy.parent),
			'artifact_root': str(
				config.outputs.output_dir.parents[6],
			),
		},
		'dataset': dict(config.dataset),
		'model': dict(config.model),
		'embeddings': {'input_dir': str(config.inputs.embeddings_dir)},
		'labels': {
			'source_label_segy': str(config.inputs.source_label_segy),
			'source_label_volume': str(config.inputs.label_volume),
			'png_label_inventory': str(config.inputs.png_label_inventory),
			'class_info': str(config.inputs.class_info),
			'segy_geometry_json': str(config.inputs.segy_geometry_json),
		},
		'registry': {
			'seismic_volume': str(config.inputs.seismic_volume),
			'label_volume': str(config.inputs.label_volume),
			'metadata_json': str(config.inputs.volume_metadata_json),
		},
		'lithology': {'root': str(config.outputs.output_dir.parent)},
		'token_dataset': {
			'output_dir': str(config.outputs.output_dir),
			'split_manifest': str(config.outputs.split_manifest_json),
			'metadata_json': str(config.outputs.metadata_json),
			'class_counts_csv': str(config.outputs.class_counts_csv),
			'summary_markdown': str(config.outputs.summary_markdown),
			'quicklook_dir': str(config.outputs.quicklook_dir),
			'tokenization': config.policy.to_dict(),
			'figure': {'dpi': config.figure_dpi},
			'feature_source': dict(config.feature_source or {}),
		},
	}


def _feature_source() -> dict[str, object]:
	return {
		'kind': 'pretrained_encoder',
		'reference_model_tag': 'model',
		'embedding_spec': 'embed',
		'description': 'fixture pretrained encoder features',
	}


def _label_volume() -> np.ndarray:
	labels = np.zeros((4, 4, 4), dtype=np.int32)
	labels[:, 2, :] = np.asarray(
		[
			[0, 0, 0, 0],
			[1, 1, 1, 1],
			[2, 2, 2, 2],
			[2, 2, 2, 2],
		],
		dtype=np.int32,
	)
	labels[1, :, :] = np.asarray(
		[
			[0, 0, 1, 1],
			[0, 1, 1, 1],
			[2, 2, 0, 0],
			[2, 2, 0, 1],
		],
		dtype=np.int32,
	)
	return labels


def _write_embedding_artifacts(output_dir: Path) -> None:
	output_dir.mkdir(parents=True)
	embeddings = np.arange(2 * 2 * 2 * 3, dtype=np.float16).reshape(2, 2, 2, 3)
	np.save(output_dir / 'f3_facies_benchmark.embeddings.npy', embeddings)
	np.save(
		output_dir / 'f3_facies_benchmark.valid_tokens.npy',
		np.ones((2, 2, 2), dtype=np.bool_),
	)
	(output_dir / 'f3_facies_benchmark.embedding_metadata.json').write_text(
		json.dumps(
			{
				'patch_size': [2, 2, 2],
				'token_grid_shape': [2, 2, 2],
				'volume_shape_xyz': [4, 4, 4],
			},
			indent=2,
			sort_keys=True,
		)
		+ '\n',
		encoding='utf-8',
	)


def _write_inventory_csv(tmp_path: Path) -> Path:
	path = tmp_path / 'label_png_inventory.csv'
	path.write_text(
		(
			'relative_path,absolute_path,split,slice_type,slice_index\n'
			'interpretation/train/labels_inline_0101.png,'
			'/tmp/labels_inline_0101.png,train,inline,101\n'
			'interpretation/validation/labels_crossline_0302.png,'
			'/tmp/labels_crossline_0302.png,validation,crossline,302\n'
		),
		encoding='utf-8',
	)
	return path


def _write_class_info(tmp_path: Path) -> Path:
	path = tmp_path / 'class_info.json'
	path.write_text(
		json.dumps(
			{
				'class_count': 3,
				'classes': [class_info.to_dict() for class_info in _classes()],
			},
			indent=2,
			sort_keys=True,
		)
		+ '\n',
		encoding='utf-8',
	)
	return path


def _write_geometry_json(tmp_path: Path) -> Path:
	path = tmp_path / 'segy_geometry.json'
	path.write_text(
		json.dumps(
			{
				'segy_files': {
					'label': {
						'cube_shape': [4, 4, 4],
						'iline_min': 100,
						'iline_max': 103,
						'xline_min': 300,
						'xline_max': 303,
					},
				},
			},
			indent=2,
			sort_keys=True,
		)
		+ '\n',
		encoding='utf-8',
	)
	return path


def _geometry() -> F3LineGeometry:
	return F3LineGeometry(
		shape_xyz=(4, 4, 4),
		inline_min=100,
		inline_max=103,
		crossline_min=300,
		crossline_max=303,
	)


def _classes() -> tuple[F3ClassInfo, ...]:
	return (
		F3ClassInfo(class_id=0, class_name='Class zero', rgb=(1, 2, 3)),
		F3ClassInfo(class_id=1, class_name='Class one', rgb=(35, 92, 167)),
		F3ClassInfo(class_id=2, class_name='Class two', rgb=(9, 8, 7)),
	)
