from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.f3.lithology.robustness import (
	assert_same_token_identity,
	budget_subset_metadata,
	class_stratified_subset_indices,
	load_token_dataset_npz,
	paired_token_identity_hash,
	save_token_dataset_npz,
	subset_token_dataset,
)
from seis_ssl_cluster.f3.lithology.token_dataset import F3LithologyTokenDataset

if TYPE_CHECKING:
	from pathlib import Path


def test_full_budget_returns_all_rows_in_original_order() -> None:
	labels = np.asarray([2, 1, 2, 0], dtype=np.int64)

	indices = class_stratified_subset_indices(
		labels,
		per_class_cap=None,
		seed=11,
	)

	np.testing.assert_array_equal(indices, np.asarray([0, 1, 2, 3], dtype=np.int64))


def test_class_cap_selects_at_most_cap_per_class_and_sorts_indices() -> None:
	labels = np.asarray([0, 0, 0, 1, 1, 1, 2], dtype=np.int64)

	indices = class_stratified_subset_indices(
		labels,
		per_class_cap=2,
		seed=3,
	)

	assert indices.tolist() == sorted(indices.tolist())
	assert np.count_nonzero(labels[indices] == 0) <= 2
	assert np.count_nonzero(labels[indices] == 1) <= 2
	assert np.count_nonzero(labels[indices] == 2) == 1


def test_class_cap_is_deterministic_for_same_seed() -> None:
	labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

	first = class_stratified_subset_indices(labels, per_class_cap=2, seed=7)
	second = class_stratified_subset_indices(labels, per_class_cap=2, seed=7)

	np.testing.assert_array_equal(first, second)


def test_different_seed_can_change_selected_rows() -> None:
	labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

	first = class_stratified_subset_indices(labels, per_class_cap=2, seed=1)
	second = class_stratified_subset_indices(labels, per_class_cap=2, seed=2)

	assert first.tolist() != second.tolist()


def test_require_all_classes_rejects_missing_class() -> None:
	labels = np.asarray([0, 0, 1], dtype=np.int64)

	with pytest.raises(ValueError, match='class_id 2 has zero rows'):
		class_stratified_subset_indices(
			labels,
			per_class_cap=1,
			seed=0,
			class_ids=[0, 1, 2],
		)


def test_identity_validation_passes_for_same_rows_with_different_features() -> None:
	reference = _dataset()
	candidate = _dataset(features=np.full((6, 3), 42.0, dtype=np.float32))

	assert_same_token_identity(
		reference,
		candidate,
		reference_label='mae',
		candidate_label='strat_hmm',
	)


@pytest.mark.parametrize('field', ['labels', 'token_xyz'])
def test_identity_validation_fails_if_labels_or_token_coordinates_differ(
	field: str,
) -> None:
	if field == 'labels':
		changed = _dataset(labels=np.asarray([0, 0, 1, 1, 2, 1], dtype=np.int64))
	else:
		changed = _dataset(
			token_xyz=np.asarray(
				[
					[0, 0, 0],
					[1, 0, 0],
					[9, 9, 9],
					[3, 0, 0],
					[4, 0, 0],
					[5, 0, 0],
				],
				dtype=np.int64,
			),
		)
	with pytest.raises(ValueError, match='token identity rows differ'):
		assert_same_token_identity(
			_dataset(),
			changed,
			reference_label='reference',
			candidate_label='candidate',
		)


def test_subset_preserves_non_feature_provenance_arrays_and_metadata() -> None:
	dataset = _dataset(metadata={'feature_source': {'kind': 'fixture'}})
	indices = np.asarray([0, 2, 5], dtype=np.int64)

	subset = subset_token_dataset(dataset, indices)

	np.testing.assert_array_equal(subset.labels, dataset.labels[indices])
	np.testing.assert_array_equal(subset.survey_id, dataset.survey_id[indices])
	np.testing.assert_array_equal(subset.split, dataset.split[indices])
	np.testing.assert_array_equal(subset.slice_type, dataset.slice_type[indices])
	np.testing.assert_array_equal(subset.slice_index, dataset.slice_index[indices])
	np.testing.assert_array_equal(subset.token_xyz, dataset.token_xyz[indices])
	np.testing.assert_array_equal(
		subset.voxel_center_xyz,
		dataset.voxel_center_xyz[indices],
	)
	np.testing.assert_array_equal(
		subset.majority_fraction,
		dataset.majority_fraction[indices],
	)
	np.testing.assert_array_equal(
		subset.labeled_fraction,
		dataset.labeled_fraction[indices],
	)
	assert subset.metadata == dataset.metadata


def test_load_token_dataset_npz_and_budget_metadata_are_json_safe(
	tmp_path: Path,
) -> None:
	train_path = tmp_path / 'train_tokens.npz'
	validation_path = tmp_path / 'validation_tokens.npz'
	train = subset_token_dataset(_dataset(), np.asarray([0, 2, 4], dtype=np.int64))
	validation = _dataset(split='validation')
	save_token_dataset_npz(train, train_path)
	save_token_dataset_npz(validation, validation_path)

	loaded_train = load_token_dataset_npz(train_path)
	loaded_validation = load_token_dataset_npz(validation_path)
	metadata = budget_subset_metadata(
		source_train_tokens=train_path,
		source_validation_tokens=validation_path,
		per_class_cap=1,
		subsample_seed=5,
		selected_train_dataset=loaded_train,
		validation_dataset=loaded_validation,
	)

	assert metadata['selected_train_token_count'] == 3
	assert metadata['validation_token_count'] == 6
	assert metadata['selected_class_counts'] == {'0': 1, '1': 1, '2': 1}
	assert metadata['validation_class_counts'] == {'0': 2, '1': 2, '2': 2}
	assert metadata['paired_identity_hash'] == paired_token_identity_hash(
		loaded_train,
		loaded_validation,
	)
	json.dumps(metadata, allow_nan=False)


def _dataset(
	*,
	features: np.ndarray | None = None,
	labels: np.ndarray | None = None,
	token_xyz: np.ndarray | None = None,
	split: str = 'train',
	metadata: dict[str, object] | None = None,
) -> F3LithologyTokenDataset:
	count = 6
	return F3LithologyTokenDataset(
		features=(
			np.arange(count * 3, dtype=np.float32).reshape(count, 3)
			if features is None
			else np.asarray(features, dtype=np.float32)
		),
		labels=(
			np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
			if labels is None
			else np.asarray(labels)
		),
		survey_id=np.asarray(['f3'] * count),
		split=np.asarray([split] * count),
		slice_type=np.asarray(
			['inline', 'inline', 'crossline', 'crossline', 'inline', 'crossline'],
		),
		slice_index=np.asarray([100, 101, 300, 301, 102, 302], dtype=np.int64),
		token_xyz=(
			np.asarray(
				[
					[0, 0, 0],
					[1, 0, 0],
					[2, 0, 0],
					[3, 0, 0],
					[4, 0, 0],
					[5, 0, 0],
				],
				dtype=np.int64,
			)
			if token_xyz is None
			else np.asarray(token_xyz)
		),
		voxel_center_xyz=np.asarray(
			[
				[0.5, 0.5, 0.5],
				[1.5, 0.5, 0.5],
				[2.5, 0.5, 0.5],
				[3.5, 0.5, 0.5],
				[4.5, 0.5, 0.5],
				[5.5, 0.5, 0.5],
			],
			dtype=np.float32,
		),
		majority_fraction=np.asarray(
			[0.9, 0.8, 0.85, 0.95, 0.75, 1.0],
			dtype=np.float32,
		),
		labeled_fraction=np.asarray(
			[1.0, 0.9, 0.8, 1.0, 0.85, 0.95],
			dtype=np.float32,
		),
		metadata={} if metadata is None else metadata,
	)
