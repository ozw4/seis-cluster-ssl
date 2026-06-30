from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.f3 import (
	F3BaselineFeatureConfig,
	F3BaselineReferenceTokenDataset,
	F3BaselineTokenDatasetOutputs,
	F3LithologyBaselineTokenDatasetConfig,
	build_f3_lithology_baseline_token_dataset,
	f3_lithology_baseline_token_dataset_config_from_mapping,
	load_token_dataset,
)
from tests.helpers import run_python_proc


def test_z_only_baseline_features_preserve_reference_split_and_labels(
	tmp_path: Path,
) -> None:
	config = _baseline_config(
		tmp_path,
		features=F3BaselineFeatureConfig(
			kind='z_only',
			polynomial_degree=2,
			normalization='minmax',
		),
	)

	result = build_f3_lithology_baseline_token_dataset(config)

	source_train = np.load(config.reference.train_tokens)
	train = np.load(result.train_npz)
	validation = np.load(result.validation_npz)
	metadata = json.loads(result.metadata_json.read_text(encoding='utf-8'))
	feature_summary = json.loads(
		result.feature_summary_json.read_text(encoding='utf-8'),
	)
	splits = json.loads(result.split_manifest_json.read_text(encoding='utf-8'))
	with result.class_counts_csv.open(encoding='utf-8', newline='') as file_obj:
		class_rows = list(csv.DictReader(file_obj))

	np.testing.assert_allclose(
		train['features'],
		np.asarray(
			[
				[0.0, 0.0],
				[1.0, 1.0],
				[0.0, 0.0],
			],
			dtype=np.float32,
		),
	)
	for key in source_train.files:
		if key != 'features':
			np.testing.assert_array_equal(train[key], source_train[key])
	np.testing.assert_array_equal(validation['labels'], np.asarray([1, 0]))
	assert train['features'].dtype == np.float32
	assert metadata['feature_source']['kind'] == 'z_only'
	assert metadata['baseline']['feature_names'] == ['z_norm', 'z_norm_power_2']
	assert feature_summary['baseline']['feature_dim'] == 2
	assert splits['split_unit'] == 'slice'
	assert any(row['split'] == 'train' and row['class_id'] == '0' for row in class_rows)
	assert result.feature_summary_markdown.is_file()
	assert load_token_dataset(result.train_npz, label='train_tokens').count == 3


def test_amplitude_stats_baseline_features_match_token_blocks(tmp_path: Path) -> None:
	seismic_path = _write_seismic_volume(tmp_path)
	config = _baseline_config(
		tmp_path,
		features=F3BaselineFeatureConfig(
			kind='amplitude_stats',
			statistics=(
				'mean',
				'std',
				'rms',
				'abs_mean',
				'min',
				'max',
				'p10',
				'p50',
				'p90',
			),
			seismic_path=seismic_path,
			feature_space='survey_normalized',
		),
	)

	result = build_f3_lithology_baseline_token_dataset(config)

	train = np.load(result.train_npz)
	metadata = json.loads(result.metadata_json.read_text(encoding='utf-8'))
	volume = np.load(seismic_path)
	first_block = volume[0:2, 0:2, 0:2]
	expected_first = np.asarray(
		[
			np.mean(first_block),
			np.std(first_block),
			np.sqrt(np.mean(np.square(first_block))),
			np.mean(np.abs(first_block)),
			np.min(first_block),
			np.max(first_block),
			np.percentile(first_block, 10),
			np.percentile(first_block, 50),
			np.percentile(first_block, 90),
		],
		dtype=np.float32,
	)

	np.testing.assert_allclose(train['features'][0], expected_first)
	assert train['features'].shape == (3, 9)
	assert metadata['baseline']['parameters']['patch_size_xyz'] == [2, 2, 2]
	assert metadata['baseline']['parameters']['feature_space'] == 'survey_normalized'
	assert (
		load_token_dataset(result.validation_npz, label='validation_tokens').count == 2
	)


def test_baseline_config_rejects_invalid_kind(tmp_path: Path) -> None:
	reference = _write_reference_token_dataset(tmp_path)
	output_dir = tmp_path / 'artifacts' / 'baselines' / 'invalid' / 'token_dataset'

	with pytest.raises(ValueError, match='baseline kind'):
		f3_lithology_baseline_token_dataset_config_from_mapping(
			{
				'paths': {'artifact_root': str(tmp_path / 'artifacts')},
				'source_token_dataset': {'directory': str(reference.root)},
				'baseline': {
					'kind': 'not_a_baseline',
					'output_dir': str(output_dir),
				},
			},
		)


def test_baseline_features_proc_dry_run_accepts_issue_style_config(
	tmp_path: Path,
) -> None:
	reference = _write_reference_token_dataset(tmp_path)
	config_path = tmp_path / 'build_baseline_features.yaml'
	config_path.write_text(
		yaml.safe_dump(
			{
				'paths': {'artifact_root': str(tmp_path / 'artifacts')},
				'source_token_dataset': {'directory': str(reference.root)},
				'baseline': {
					'kind': 'z_only',
					'output_dir': str(
						tmp_path / 'artifacts' / 'baselines' / 'z' / 'token_dataset',
					),
					'z_only': {
						'normalize': 'minmax',
						'polynomial_degree': 1,
					},
				},
			},
		),
		encoding='utf-8',
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/build_f3_lithology_baseline_features.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert 'stage: build_f3_lithology_baseline_token_dataset' in result.stdout
	assert 'baseline.kind: z_only' in result.stdout
	assert 'execution: dry-run; F3 lithology baseline token dataset build skipped' in (
		result.stdout
	)


def _baseline_config(
	tmp_path: Path,
	*,
	features: F3BaselineFeatureConfig,
) -> F3LithologyBaselineTokenDatasetConfig:
	reference = _write_reference_token_dataset(tmp_path)
	output_dir = (
		tmp_path
		/ 'artifacts'
		/ 'lithology'
		/ 'f3'
		/ 'facies_benchmark_v1'
		/ 'baselines'
		/ features.kind
		/ 'labels'
		/ 'token_dataset'
	)
	return F3LithologyBaselineTokenDatasetConfig(
		reference=reference,
		outputs=F3BaselineTokenDatasetOutputs(
			output_dir=output_dir,
			metadata_json=output_dir / 'token_dataset_metadata.json',
			feature_summary_json=output_dir / 'feature_summary.json',
			feature_summary_markdown=output_dir / 'feature_summary.md',
			split_manifest_json=output_dir / 'splits.json',
			class_counts_csv=output_dir / 'class_counts.csv',
			summary_markdown=output_dir / 'token_dataset_summary.md',
		),
		features=features,
		dataset={'name': 'f3_facies_benchmark', 'version': 'facies_benchmark_v1'},
		model={'tag': f'{features.kind}_v1', 'freeze_encoder': True},
		labels={'set': 'fixture_labels'},
		token_dataset={
			'feature_source': {
				'kind': features.kind,
				'reference_model_tag': 'reference_model',
				'embedding_spec': 'overlap_x16',
				'description': 'fixture baseline features',
			},
		},
		feature_source={
			'kind': features.kind,
			'reference_model_tag': 'reference_model',
			'embedding_spec': 'overlap_x16',
			'description': 'fixture baseline features',
		},
	)


def _write_reference_token_dataset(tmp_path: Path) -> F3BaselineReferenceTokenDataset:
	root = tmp_path / 'artifacts' / 'reference' / 'token_dataset'
	root.mkdir(parents=True, exist_ok=True)
	_write_tokens(
		root / 'train_tokens.npz',
		labels=np.asarray([0, 1, 0], dtype=np.int64),
		split='train',
		token_xyz=np.asarray([[0, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int64),
		voxel_center_z=np.asarray([0.5, 2.5, 0.5], dtype=np.float32),
	)
	_write_tokens(
		root / 'validation_tokens.npz',
		labels=np.asarray([1, 0], dtype=np.int64),
		split='validation',
		token_xyz=np.asarray([[1, 1, 1], [0, 1, 0]], dtype=np.int64),
		voxel_center_z=np.asarray([2.5, 0.5], dtype=np.float32),
	)
	metadata_json = root / 'token_dataset_metadata.json'
	metadata_json.write_text(
		json.dumps(
			{
				'artifact_type': 'f3_lithology_token_dataset',
				'dataset': {
					'name': 'f3_facies_benchmark',
					'version': 'facies_benchmark_v1',
				},
				'label_source_of_truth': 'segy_label_volume',
				'png_label_role': 'train_validation_slice_selection_and_visual_qc',
				'split_strategy': (
					'png_label_inventory_slice_split_no_random_token_split'
				),
				'no_random_split': True,
				'embedding': {
					'patch_size_xyz': [2, 2, 2],
					'token_grid_shape_xyz': [2, 2, 2],
					'embedding_dim': 4,
				},
				'geometry': {'shape_xyz': [4, 4, 4]},
				'tokenization': {
					'min_labeled_fraction': 0.5,
					'min_majority_fraction': 0.7,
					'ignore_z_border_samples': 1,
				},
				'classes': [
					{
						'class_id': 0,
						'class_name': 'Class 0',
						'rgb': [0, 0, 0],
					},
					{
						'class_id': 1,
						'class_name': 'Class 1',
						'rgb': [255, 255, 255],
					},
				],
			},
			indent=2,
			sort_keys=True,
		)
		+ '\n',
		encoding='utf-8',
	)
	splits_json = root / 'splits.json'
	splits_json.write_text(
		json.dumps({'split_unit': 'slice', 'no_random_split': True}) + '\n',
		encoding='utf-8',
	)
	return F3BaselineReferenceTokenDataset(
		train_tokens=root / 'train_tokens.npz',
		validation_tokens=root / 'validation_tokens.npz',
		metadata_json=metadata_json,
		split_manifest=splits_json,
		root=root,
	)


def _write_tokens(
	path: Path,
	*,
	labels: np.ndarray,
	split: str,
	token_xyz: np.ndarray,
	voxel_center_z: np.ndarray,
) -> None:
	count = int(labels.shape[0])
	np.savez_compressed(
		path,
		features=np.full((count, 4), -1.0, dtype=np.float32),
		labels=labels,
		survey_id=np.asarray(['f3_facies_benchmark'] * count),
		split=np.asarray([split] * count),
		slice_type=np.asarray(['inline'] * count),
		slice_index=np.arange(count, dtype=np.int64),
		token_xyz=token_xyz,
		voxel_center_xyz=np.column_stack(
			(
				np.zeros(count, dtype=np.float32),
				np.zeros(count, dtype=np.float32),
				voxel_center_z,
			),
		),
		majority_fraction=np.ones(count, dtype=np.float32),
		labeled_fraction=np.ones(count, dtype=np.float32),
	)


def _write_seismic_volume(tmp_path: Path) -> Path:
	path = tmp_path / 'artifacts' / 'registry' / 'f3_seismic.npy'
	path.parent.mkdir(parents=True, exist_ok=True)
	np.save(path, np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4))
	return path
