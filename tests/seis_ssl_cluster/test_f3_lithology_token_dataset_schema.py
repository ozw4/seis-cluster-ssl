from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.f3.lithology.token_dataset import (
	F3_LITHOLOGY_TOKEN_DATASET_KEYS,
	F3LithologyTokenDataset,
	load_f3_lithology_token_dataset,
	load_f3_lithology_token_dataset_summary,
	replace_token_features,
	save_f3_lithology_token_dataset,
	validate_f3_lithology_token_dataset,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_token_dataset_schema_round_trips_existing_npz_keys(tmp_path: Path) -> None:
	path = tmp_path / 'train_tokens.npz'
	dataset = _dataset()

	save_f3_lithology_token_dataset(dataset, path)
	loaded = load_f3_lithology_token_dataset(path)

	with np.load(path) as payload:
		assert tuple(payload.files) == F3_LITHOLOGY_TOKEN_DATASET_KEYS
		assert 'metadata' not in payload.files
	np.testing.assert_array_equal(loaded.labels, np.asarray([0, 1], dtype=np.int64))
	np.testing.assert_array_equal(loaded.features, dataset.features)
	assert loaded.count == 2


def test_token_dataset_schema_round_trips_metadata_when_present(
	tmp_path: Path,
) -> None:
	path = tmp_path / 'train_tokens.npz'
	metadata = {
		'feature_source': {
			'kind': 'pretrained_encoder',
			'reference_model_tag': 'model',
			'embedding_spec': 'overlap_x16',
			'description': 'fixture pretrained encoder features',
		},
	}

	save_f3_lithology_token_dataset(_dataset(metadata=metadata), path)
	loaded = load_f3_lithology_token_dataset(path)

	assert loaded.metadata == metadata
	with np.load(path) as payload:
		assert 'metadata' in payload.files
		assert json.loads(str(payload['metadata'].item())) == metadata


def test_token_dataset_summary_loads_labels_and_metadata(tmp_path: Path) -> None:
	path = tmp_path / 'train_tokens.npz'
	metadata = {'feature_source': {'kind': 'amplitude_statistics'}}

	save_f3_lithology_token_dataset(_dataset(metadata=metadata), path)
	summary = load_f3_lithology_token_dataset_summary(path)

	assert summary.count == 2
	np.testing.assert_array_equal(summary.labels, np.asarray([0, 1], dtype=np.int64))
	assert summary.metadata == metadata


def test_token_dataset_schema_accepts_class_zero_as_valid_label() -> None:
	dataset = _dataset(labels=np.asarray([0, 0], dtype=np.int64))

	validate_f3_lithology_token_dataset(dataset)


def test_replace_token_features_preserves_schema_and_records_feature_source() -> None:
	dataset = _dataset()
	replacement = np.asarray([[3.0], [4.0]], dtype=np.float32)
	feature_source = {
		'kind': 'xyz_coordinates',
		'description': 'fixture xyz baseline features',
	}

	updated = replace_token_features(
		dataset,
		replacement,
		feature_source=feature_source,
	)

	np.testing.assert_array_equal(updated.features, replacement)
	np.testing.assert_array_equal(updated.labels, dataset.labels)
	np.testing.assert_array_equal(updated.token_xyz, dataset.token_xyz)
	assert updated.metadata['feature_source'] == feature_source


def test_token_dataset_schema_rejects_non_integer_labels() -> None:
	dataset = _dataset(labels=np.asarray([0.0, 1.0], dtype=np.float32))

	with pytest.raises(TypeError, match='labels must be integer typed'):
		validate_f3_lithology_token_dataset(dataset)


def test_token_dataset_schema_rejects_bad_token_xyz_shape() -> None:
	dataset = _dataset(token_xyz=np.asarray([[0, 0], [1, 0]], dtype=np.int64))

	with pytest.raises(ValueError, match='token_xyz must have shape'):
		validate_f3_lithology_token_dataset(dataset)


def _dataset(
	*,
	labels: np.ndarray | None = None,
	token_xyz: np.ndarray | None = None,
	metadata: dict[str, object] | None = None,
) -> F3LithologyTokenDataset:
	count = 2
	return F3LithologyTokenDataset(
		features=np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32),
		labels=(
			np.asarray([0, 1], dtype=np.int64)
			if labels is None
			else np.asarray(labels)
		),
		survey_id=np.asarray(['f3', 'f3']),
		split=np.asarray(['train', 'train']),
		slice_type=np.asarray(['inline', 'crossline']),
		slice_index=np.asarray([100, 300], dtype=np.int64),
		token_xyz=(
			np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
			if token_xyz is None
			else np.asarray(token_xyz)
		),
		voxel_center_xyz=np.asarray(
			[[0.5, 0.5, 0.5], [2.5, 0.5, 0.5]],
			dtype=np.float32,
		),
		majority_fraction=np.ones(count, dtype=np.float32),
		labeled_fraction=np.ones(count, dtype=np.float32),
		metadata={} if metadata is None else metadata,
	)
