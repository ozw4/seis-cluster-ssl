from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.config.f3_lithology_robustness import (
	f3_lithology_split_sweep_dataset_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.robustness import (
	F3_SPLIT_INVENTORY_MANIFEST_ARTIFACT_TYPE,
	F3SplitSweepDatasetConfig,
	build_f3_lithology_split_sweep_datasets,
	load_token_dataset_npz,
	paired_token_identity_hash,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_split_sweep_builder_builds_paired_baseline_and_candidate(
	tmp_path: Path,
) -> None:
	config = _config(tmp_path)

	result = build_f3_lithology_split_sweep_datasets(config)

	assert result.manifest_json == config.output_root / 'split_dataset_manifest.json'
	assert len(result.dataset_roots) == 4
	manifest = _read_json(result.manifest_json)
	assert len(manifest['rows']) == 4

	for split_id in ('split_000', 'split_001'):
		baseline_root = _dataset_root(config, split_id, 'baseline')
		candidate_root = _dataset_root(config, split_id, 'candidate')
		assert (baseline_root / 'train_tokens.npz').is_file()
		assert (candidate_root / 'train_tokens.npz').is_file()
		baseline_hash = _identity_hash(baseline_root)
		candidate_hash = _identity_hash(candidate_root)
		assert baseline_hash == candidate_hash

		baseline_train = load_token_dataset_npz(baseline_root / 'train_tokens.npz')
		candidate_train = load_token_dataset_npz(candidate_root / 'train_tokens.npz')
		assert not np.array_equal(baseline_train.features, candidate_train.features)


def test_candidate_dataset_metadata_uses_reference_rows(tmp_path: Path) -> None:
	config = _config(tmp_path)

	build_f3_lithology_split_sweep_datasets(config)

	split_id = 'split_000'
	candidate_root = _dataset_root(config, split_id, 'candidate')
	baseline_root = _dataset_root(config, split_id, 'baseline')
	metadata = _read_json(candidate_root / 'token_dataset_metadata.json')

	assert metadata['reference_token_dataset']['root'] == str(baseline_root)
	assert (
		metadata['feature_source']['reference_model_tag']
		== config.candidate.model_tag
	)


def test_split_specific_inventory_path_is_recorded(tmp_path: Path) -> None:
	config = _config(tmp_path)

	build_f3_lithology_split_sweep_datasets(config)

	inventory = tmp_path / 'splits' / 'split_001' / 'png_label_inventory.csv'
	metadata = _read_json(
		_dataset_root(config, 'split_001', 'baseline')
		/ 'token_dataset_metadata.json',
	)
	assert metadata['inputs']['png_label_inventory'] == str(inventory)


def test_only_missing_skips_complete_outputs(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _config(tmp_path)
	build_f3_lithology_split_sweep_datasets(config)

	def fail_if_called(*_args: object, **_kwargs: object) -> object:
		raise AssertionError('token dataset builder should not be called')

	monkeypatch.setattr(
		'seis_ssl_cluster.f3.lithology.robustness.build_f3_lithology_token_dataset',
		fail_if_called,
	)

	result = build_f3_lithology_split_sweep_datasets(config, only_missing=True)

	assert len(result.rows) == 4


def test_candidate_build_fails_if_baseline_reference_is_missing(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _config(tmp_path)

	def no_outputs(*_args: object, **_kwargs: object) -> object:
		return None

	monkeypatch.setattr(
		'seis_ssl_cluster.f3.lithology.robustness.build_f3_lithology_token_dataset',
		no_outputs,
	)

	with pytest.raises(FileNotFoundError, match='baseline reference token dataset'):
		build_f3_lithology_split_sweep_datasets(config)


def _identity_hash(root: Path) -> str:
	return paired_token_identity_hash(
		load_token_dataset_npz(root / 'train_tokens.npz'),
		load_token_dataset_npz(root / 'validation_tokens.npz'),
	)


def _dataset_root(
	config: F3SplitSweepDatasetConfig,
	split_id: str,
	role: str,
) -> Path:
	model = config.baseline if role == 'baseline' else config.candidate
	return (
		config.output_root
		/ 'datasets'
		/ f'split={split_id}'
		/ f'model={model.model_tag}'
		/ 'token_dataset'
	)


def _config(tmp_path: Path) -> F3SplitSweepDatasetConfig:
	paths = _write_fixture(tmp_path)
	return f3_lithology_split_sweep_dataset_config_from_mapping(
		{
			'suite': {
				'split_inventory_manifest': str(paths['split_manifest']),
				'output_root': str(tmp_path / 'out'),
			},
			'models': {
				'baseline': {
					'model_tag': 'mae_fixture',
					'embeddings_dir': str(paths['baseline_embeddings']),
					'checkpoint': str(paths['baseline_checkpoint']),
				},
				'candidate': {
					'model_tag': 'strat_hmm_fixture',
					'embeddings_dir': str(paths['candidate_embeddings']),
					'checkpoint': str(paths['candidate_checkpoint']),
				},
			},
			'common': {
				'f3_root': str(tmp_path / 'F3'),
				'artifact_root': str(tmp_path / 'artifacts'),
				'dataset': {
					'name': 'f3_facies_benchmark',
					'version': 'facies_benchmark_v1',
				},
				'labels': {
					'source_label_segy': str(tmp_path / 'F3' / 'f3_labels.sgy'),
					'source_label_volume': str(paths['label_volume']),
					'class_info': str(paths['class_info']),
					'segy_geometry_json': str(paths['geometry']),
				},
				'registry': {
					'seismic_volume': str(paths['seismic_volume']),
					'label_volume': str(paths['label_volume']),
					'metadata_json': str(paths['volume_metadata']),
				},
				'tokenization': {
					'min_labeled_fraction': 1.0,
					'min_majority_fraction': 0.5,
					'ignore_z_border_samples': 0,
				},
			},
			'outputs': {'overwrite': False},
		},
	)


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
	artifacts = tmp_path / 'artifacts'
	split_root = tmp_path / 'splits'
	split_000 = split_root / 'split_000' / 'png_label_inventory.csv'
	split_001 = split_root / 'split_001' / 'png_label_inventory.csv'
	_write_inventory(split_000, validation='crossline')
	_write_inventory(split_001, validation='inline')
	split_manifest = split_root / 'split_inventory_manifest.json'
	_write_json(
		split_manifest,
		{
			'artifact_type': F3_SPLIT_INVENTORY_MANIFEST_ARTIFACT_TYPE,
			'rows': [
				{'split_id': 'split_000', 'png_label_inventory': str(split_000)},
				{'split_id': 'split_001', 'png_label_inventory': str(split_001)},
			],
		},
	)
	label_volume = artifacts / 'registry' / 'f3_facies_labels.npy'
	seismic_volume = artifacts / 'registry' / 'f3_seismic.npy'
	label_volume.parent.mkdir(parents=True, exist_ok=True)
	labels = _label_volume()
	np.save(label_volume, labels)
	np.save(
		seismic_volume,
		np.arange(labels.size, dtype=np.float32).reshape(labels.shape),
	)
	volume_metadata = artifacts / 'registry' / 'f3_metadata.json'
	_write_json(volume_metadata, {'shape': list(labels.shape)})
	class_info = tmp_path / 'class_info.json'
	_write_json(
		class_info,
		{
			'class_count': 3,
			'classes': [
				{'class_id': 0, 'class_name': 'Class zero', 'rgb': [1, 2, 3]},
				{'class_id': 1, 'class_name': 'Class one', 'rgb': [35, 92, 167]},
				{'class_id': 2, 'class_name': 'Class two', 'rgb': [9, 8, 7]},
			],
		},
	)
	geometry = tmp_path / 'segy_geometry.json'
	_write_json(
		geometry,
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
	)
	baseline_embeddings = artifacts / 'embeddings' / 'mae' / 'overlap_x16'
	candidate_embeddings = artifacts / 'embeddings' / 'strat_hmm' / 'overlap_x16'
	_write_embedding_artifacts(baseline_embeddings, offset=0.0)
	_write_embedding_artifacts(candidate_embeddings, offset=100.0)
	baseline_checkpoint = artifacts / 'checkpoints' / 'mae.pt'
	candidate_checkpoint = artifacts / 'checkpoints' / 'strat.pt'
	baseline_checkpoint.parent.mkdir(parents=True, exist_ok=True)
	baseline_checkpoint.write_bytes(b'baseline')
	candidate_checkpoint.write_bytes(b'candidate')
	return {
		'split_manifest': split_manifest,
		'label_volume': label_volume,
		'seismic_volume': seismic_volume,
		'volume_metadata': volume_metadata,
		'class_info': class_info,
		'geometry': geometry,
		'baseline_embeddings': baseline_embeddings,
		'candidate_embeddings': candidate_embeddings,
		'baseline_checkpoint': baseline_checkpoint,
		'candidate_checkpoint': candidate_checkpoint,
	}


def _write_inventory(path: Path, *, validation: str) -> None:
	rows = [
		{
			'relative_path': 'interpretation/inline_0101.png',
			'absolute_path': '/fixture/inline_0101.png',
			'split': 'validation' if validation == 'inline' else 'train',
			'slice_type': 'inline',
			'slice_index': '101',
		},
		{
			'relative_path': 'interpretation/crossline_0302.png',
			'absolute_path': '/fixture/crossline_0302.png',
			'split': 'validation' if validation == 'crossline' else 'train',
			'slice_type': 'crossline',
			'slice_index': '302',
		},
	]
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=tuple(rows[0].keys()))
		writer.writeheader()
		writer.writerows(rows)


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


def _write_embedding_artifacts(output_dir: Path, *, offset: float) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	embeddings = (
		np.arange(2 * 2 * 2 * 3, dtype=np.float16).reshape(2, 2, 2, 3) + offset
	)
	np.save(output_dir / 'f3_facies_benchmark.embeddings.npy', embeddings)
	np.save(
		output_dir / 'f3_facies_benchmark.valid_tokens.npy',
		np.ones((2, 2, 2), dtype=np.bool_),
	)
	_write_json(
		output_dir / 'f3_facies_benchmark.embedding_metadata.json',
		{
			'patch_size': [2, 2, 2],
			'token_grid_shape': [2, 2, 2],
			'volume_shape_xyz': [4, 4, 4],
		},
	)


def _write_json(path: Path, payload: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _read_json(path: Path) -> dict[str, object]:
	return json.loads(path.read_text(encoding='utf-8'))
