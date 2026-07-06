"""Shared schema helpers for F3 lithology token dataset NPZ files."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

	from numpy.typing import NDArray

F3_LITHOLOGY_TOKEN_DATASET_KEYS = (
	'features',
	'labels',
	'survey_id',
	'split',
	'slice_type',
	'slice_index',
	'token_xyz',
	'voxel_center_xyz',
	'majority_fraction',
	'labeled_fraction',
)


@dataclass(frozen=True)
class F3LithologyTokenDataset:
	"""Flat token-level F3 lithology dataset using the existing NPZ schema."""

	features: NDArray[np.float32]
	labels: NDArray[np.integer]
	survey_id: NDArray[np.generic]
	split: NDArray[np.generic]
	slice_type: NDArray[np.generic]
	slice_index: NDArray[np.integer]
	token_xyz: NDArray[np.integer]
	voxel_center_xyz: NDArray[np.floating]
	majority_fraction: NDArray[np.floating]
	labeled_fraction: NDArray[np.floating]
	metadata: Mapping[str, object] = MappingProxyType({})

	@property
	def count(self) -> int:
		"""Return number of token rows."""
		return int(self.labels.shape[0])

	def to_npz_arrays(self) -> dict[str, NDArray[np.generic]]:
		"""Return arrays under the stable on-disk NPZ key names."""
		return {
			'features': self.features,
			'labels': self.labels,
			'survey_id': self.survey_id,
			'split': self.split,
			'slice_type': self.slice_type,
			'slice_index': self.slice_index,
			'token_xyz': self.token_xyz,
			'voxel_center_xyz': self.voxel_center_xyz,
			'majority_fraction': self.majority_fraction,
			'labeled_fraction': self.labeled_fraction,
		}


def load_f3_lithology_token_dataset(path: Path) -> F3LithologyTokenDataset:
	"""Load and validate an F3 lithology token dataset NPZ."""
	if not path.is_file():
		msg = f'F3 lithology token dataset does not exist: {path}'
		raise FileNotFoundError(msg)
	with np.load(path) as payload:
		missing = [
			name
			for name in F3_LITHOLOGY_TOKEN_DATASET_KEYS
			if name not in payload.files
		]
		if missing:
			msg = f'F3 lithology token dataset missing required field(s): {missing!r}'
			raise KeyError(msg)
		dataset = F3LithologyTokenDataset(
			features=np.asarray(payload['features'], dtype=np.float32),
			labels=np.asarray(payload['labels']),
			survey_id=np.asarray(payload['survey_id']),
			split=np.asarray(payload['split']),
			slice_type=np.asarray(payload['slice_type']),
			slice_index=np.asarray(payload['slice_index']),
			token_xyz=np.asarray(payload['token_xyz']),
			voxel_center_xyz=np.asarray(payload['voxel_center_xyz'], dtype=np.float32),
			majority_fraction=np.asarray(
				payload['majority_fraction'],
				dtype=np.float32,
			),
			labeled_fraction=np.asarray(payload['labeled_fraction'], dtype=np.float32),
		)
	validate_f3_lithology_token_dataset(dataset)
	return dataset


def save_f3_lithology_token_dataset(
	dataset: F3LithologyTokenDataset,
	path: Path,
) -> None:
	"""Validate and save an F3 lithology token dataset NPZ."""
	validate_f3_lithology_token_dataset(dataset)
	path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(path, **dataset.to_npz_arrays())


def validate_f3_lithology_token_dataset(
	dataset: F3LithologyTokenDataset,
) -> None:
	"""Validate row counts, shapes, dtypes, and finite feature values."""
	count = _validate_features_and_labels(dataset.features, dataset.labels)
	_validate_token_metadata_shapes(dataset, count)
	_validate_token_metadata_dtypes(dataset)
	_validate_finite_token_metadata(dataset)


def _validate_features_and_labels(
	features: NDArray[np.generic],
	labels: NDArray[np.generic],
) -> int:
	features = np.asarray(features)
	labels = np.asarray(labels)
	if features.ndim != 2:
		msg = f'features must be a 2D matrix; got shape={features.shape!r}'
		raise ValueError(msg)
	if not np.issubdtype(features.dtype, np.floating):
		msg = f'features must be floating point; got dtype={features.dtype}'
		raise TypeError(msg)
	if not np.all(np.isfinite(features)):
		msg = 'features must contain only finite values'
		raise ValueError(msg)
	if labels.ndim != 1:
		msg = f'labels must be a 1D vector; got shape={labels.shape!r}'
		raise ValueError(msg)
	if not np.issubdtype(labels.dtype, np.integer):
		msg = f'labels must be integer typed; got dtype={labels.dtype}'
		raise TypeError(msg)
	count = int(labels.shape[0])
	if features.shape[0] != count:
		msg = (
			'features row count must match labels; '
			f'got {features.shape[0]}, expected={count}'
		)
		raise ValueError(msg)
	return count


def _validate_token_metadata_shapes(
	dataset: F3LithologyTokenDataset,
	count: int,
) -> None:
	_validate_vector_length(dataset.survey_id, count, 'survey_id')
	_validate_vector_length(dataset.split, count, 'split')
	_validate_vector_length(dataset.slice_type, count, 'slice_type')
	_validate_vector_length(dataset.slice_index, count, 'slice_index')
	_validate_vector_length(
		dataset.majority_fraction,
		count,
		'majority_fraction',
	)
	_validate_vector_length(dataset.labeled_fraction, count, 'labeled_fraction')
	_validate_xyz(dataset.token_xyz, count, 'token_xyz')
	_validate_xyz(dataset.voxel_center_xyz, count, 'voxel_center_xyz')


def _validate_token_metadata_dtypes(dataset: F3LithologyTokenDataset) -> None:
	if not np.issubdtype(np.asarray(dataset.slice_index).dtype, np.integer):
		msg = (
			'slice_index must be integer typed; '
			f'got dtype={dataset.slice_index.dtype}'
		)
		raise TypeError(msg)
	if not np.issubdtype(np.asarray(dataset.token_xyz).dtype, np.integer):
		msg = f'token_xyz must be integer typed; got dtype={dataset.token_xyz.dtype}'
		raise TypeError(msg)


def _validate_finite_token_metadata(dataset: F3LithologyTokenDataset) -> None:
	for name, values in (
		('voxel_center_xyz', dataset.voxel_center_xyz),
		('majority_fraction', dataset.majority_fraction),
		('labeled_fraction', dataset.labeled_fraction),
	):
		if not np.all(np.isfinite(np.asarray(values))):
			msg = f'{name} must contain only finite values'
			raise ValueError(msg)


def replace_token_features(
	dataset: F3LithologyTokenDataset,
	features: NDArray[np.generic],
	*,
	feature_source: Mapping[str, object],
) -> F3LithologyTokenDataset:
	"""Return a dataset with replacement features and updated in-memory metadata."""
	replacement_features = np.asarray(features, dtype=np.float32)
	if replacement_features.ndim != 2:
		msg = (
			'replacement features must be a 2D matrix; '
			f'got shape={replacement_features.shape!r}'
		)
		raise ValueError(msg)
	if replacement_features.shape[0] != dataset.count:
		msg = (
			'replacement features row count must match labels; '
			f'got {replacement_features.shape[0]}, expected={dataset.count}'
		)
		raise ValueError(msg)
	if not np.all(np.isfinite(replacement_features)):
		msg = 'replacement features must contain only finite values'
		raise ValueError(msg)
	metadata = dict(dataset.metadata)
	metadata['feature_source'] = dict(feature_source)
	replaced = replace(
		dataset,
		features=replacement_features,
		metadata=metadata,
	)
	validate_f3_lithology_token_dataset(replaced)
	return replaced


def _validate_vector_length(
	values: NDArray[np.generic],
	count: int,
	name: str,
) -> None:
	if np.asarray(values).shape != (count,):
		msg = f'{name} must have shape {(count,)!r}; got {np.asarray(values).shape!r}'
		raise ValueError(msg)


def _validate_xyz(values: NDArray[np.generic], count: int, name: str) -> None:
	if np.asarray(values).shape != (count, 3):
		msg = f'{name} must have shape {(count, 3)!r}; got {np.asarray(values).shape!r}'
		raise ValueError(msg)
