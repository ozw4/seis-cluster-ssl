"""Two-view augmentation for amplitude Barlow Twins pretraining."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.data.crop_sampler import rng_for_sample
from seis_ssl_cluster.data.window_preprocessing import reduce_valid_mask_to_tokens

if TYPE_CHECKING:
	from seis_ssl_cluster.data.amplitude_dataset import AmplitudePretrainDataset


class BarlowTwinsPretrainDataset:
	"""Wrap one amplitude crop as two independently flipped views."""

	def __init__(
		self,
		base_dataset: AmplitudePretrainDataset,
		*,
		horizontal_flip_probability: float = 0.5,
	) -> None:
		"""Initialize the wrapper and validate its flip probability."""
		self.base_dataset = base_dataset
		self.horizontal_flip_probability = _validate_probability(
			horizontal_flip_probability,
			'horizontal_flip_probability',
		)

	def __len__(self) -> int:
		"""Return the wrapped dataset's epoch length."""
		return len(self.base_dataset)

	@property
	def epoch(self) -> int:
		"""Return the wrapped dataset's current sampling epoch."""
		return self.base_dataset.epoch

	def set_epoch(self, epoch: int) -> None:
		"""Forward the sampling epoch to the wrapped dataset."""
		self.base_dataset.set_epoch(epoch)

	def _load_base_sample(
		self,
		index: int,
	) -> tuple[
		dict[str, object],
		np.ndarray,
		np.ndarray,
		np.random.Generator,
	]:
		normalized_index = _normalize_index(index, len(self))
		base_sample = self.base_dataset[normalized_index]
		x = _require_array(base_sample, 'x')
		valid_mask = _require_array(base_sample, 'local_valid_mask')
		_validate_sample_shapes(x, valid_mask)
		rng = rng_for_sample(
			self.base_dataset.seed,
			self.epoch,
			normalized_index,
		)
		return base_sample, x, valid_mask, rng

	def __getitem__(self, index: int) -> dict[str, object]:
		"""Return two augmented copies of one preprocessed physical crop."""
		base_sample, x, valid_mask, rng = self._load_base_sample(index)
		(view_a, valid_mask_a, _), (view_b, valid_mask_b, _) = _build_horizontal_views(
			x,
			valid_mask,
			rng,
			probability=self.horizontal_flip_probability,
			require_distinct=False,
		)
		return {
			'view_a': view_a,
			'view_b': view_b,
			'valid_mask_a': valid_mask_a,
			'valid_mask_b': valid_mask_b,
			'coords': base_sample.get('coords'),
		}


class LocalBarlowTwinsPretrainDataset(BarlowTwinsPretrainDataset):
	"""Return flipped views and indices for matching physical tokens."""

	def __init__(
		self,
		base_dataset: AmplitudePretrainDataset,
		*,
		local_pairs_per_crop: int,
		horizontal_flip_probability: float = 0.5,
	) -> None:
		"""Initialize the local-pair wrapper and validate its sampling contract."""
		super().__init__(
			base_dataset,
			horizontal_flip_probability=horizontal_flip_probability,
		)
		self.local_pairs_per_crop = _validate_positive_int(
			local_pairs_per_crop,
			'local_pairs_per_crop',
		)
		if base_dataset.min_valid_token_count < self.local_pairs_per_crop:
			msg = (
				'base_dataset.min_valid_token_count must be greater than or equal '
				f'to local_pairs_per_crop ({self.local_pairs_per_crop}); got '
				f'{base_dataset.min_valid_token_count}'
			)
			raise ValueError(msg)

	def __getitem__(self, index: int) -> dict[str, object]:
		"""Return two views plus C-order indices for canonical token pairs."""
		base_sample, x, valid_mask, rng = self._load_base_sample(index)
		(
			(view_a, valid_mask_a, flip_state_a),
			(view_b, valid_mask_b, flip_state_b),
		) = _build_horizontal_views(
			x,
			valid_mask,
			rng,
			probability=self.horizontal_flip_probability,
			require_distinct=True,
		)

		canonical_indices, token_shape = _sample_canonical_token_indices(
			valid_mask,
			self.base_dataset.patch_size_xyz,
			self.local_pairs_per_crop,
			rng,
		)

		return {
			'view_a': view_a,
			'view_b': view_b,
			'valid_mask_a': valid_mask_a,
			'valid_mask_b': valid_mask_b,
			'coords': base_sample.get('coords'),
			'horizontal_flip_state_a': flip_state_a,
			'horizontal_flip_state_b': flip_state_b,
			'local_pair_indices_a': _map_token_indices_for_view(
				canonical_indices,
				token_shape,
				flip_state_a,
			),
			'local_pair_indices_b': _map_token_indices_for_view(
				canonical_indices,
				token_shape,
				flip_state_b,
			),
		}


class LocalBarlowTwinsD4TraceDropPretrainDataset(BarlowTwinsPretrainDataset):
	"""Return XY-D4 views, trace-drop metadata, and physical token pairs."""

	def __init__(
		self,
		base_dataset: AmplitudePretrainDataset,
		*,
		local_pairs_per_crop: int,
		reflection_probability: float,
		trace_drop_probability: float,
	) -> None:
		"""Initialize and validate the square-XY augmentation contract."""
		super().__init__(base_dataset)
		self.local_pairs_per_crop = _validate_positive_int(
			local_pairs_per_crop,
			'local_pairs_per_crop',
		)
		if base_dataset.min_valid_token_count < self.local_pairs_per_crop:
			msg = (
				'base_dataset.min_valid_token_count must be greater than or equal '
				f'to local_pairs_per_crop ({self.local_pairs_per_crop}); got '
				f'{base_dataset.min_valid_token_count}'
			)
			raise ValueError(msg)
		self.reflection_probability = _validate_probability(
			reflection_probability,
			'reflection_probability',
		)
		self.trace_drop_probability = _validate_probability(
			trace_drop_probability,
			'trace_drop_probability',
		)
		_validate_square_xy(base_dataset.local_crop_size_xyz, 'local crop')
		_validate_square_xy(base_dataset.patch_size_xyz, 'patch size')
		_validate_square_xy(base_dataset.token_grid_shape_xyz, 'token grid')

	def __getitem__(self, index: int) -> dict[str, object]:
		"""Return two independently augmented views of canonical token pairs."""
		base_sample, x, valid_mask, rng = self._load_base_sample(index)
		canonical_indices, token_shape = _sample_canonical_token_indices(
			valid_mask,
			self.base_dataset.patch_size_xyz,
			self.local_pairs_per_crop,
			rng,
		)
		transform_id_a = _sample_xy_d4_transform_id(
			rng,
			reflection_probability=self.reflection_probability,
		)
		transform_id_b = _sample_xy_d4_transform_id(
			rng,
			reflection_probability=self.reflection_probability,
		)
		view_a = _apply_xy_d4(x, transform_id_a, xy_axes=(1, 2))
		view_b = _apply_xy_d4(x, transform_id_b, xy_axes=(1, 2))
		valid_mask_a = _apply_xy_d4(
			valid_mask,
			transform_id_a,
			xy_axes=(0, 1),
		)
		valid_mask_b = _apply_xy_d4(
			valid_mask,
			transform_id_b,
			xy_axes=(0, 1),
		)
		local_pair_indices_a = _map_token_indices_for_d4_view(
			canonical_indices,
			token_shape,
			transform_id_a,
		)
		local_pair_indices_b = _map_token_indices_for_d4_view(
			canonical_indices,
			token_shape,
			transform_id_b,
		)
		trace_drop_count_a = _apply_trace_drop(
			view_a,
			valid_mask_a,
			rng,
			probability=self.trace_drop_probability,
		)
		trace_drop_count_b = _apply_trace_drop(
			view_b,
			valid_mask_b,
			rng,
			probability=self.trace_drop_probability,
		)

		return {
			'view_a': view_a,
			'view_b': view_b,
			'valid_mask_a': valid_mask_a,
			'valid_mask_b': valid_mask_b,
			'coords': base_sample.get('coords'),
			'xy_transform_id_a': np.asarray(transform_id_a, dtype=np.int64),
			'xy_transform_id_b': np.asarray(transform_id_b, dtype=np.int64),
			'trace_drop_count_a': np.asarray(
				trace_drop_count_a,
				dtype=np.int64,
			),
			'trace_drop_count_b': np.asarray(
				trace_drop_count_b,
				dtype=np.int64,
			),
			'local_pair_indices_a': local_pair_indices_a,
			'local_pair_indices_b': local_pair_indices_b,
		}


def _build_horizontal_views(
	x: np.ndarray,
	valid_mask: np.ndarray,
	rng: np.random.Generator,
	*,
	probability: float,
	require_distinct: bool,
) -> tuple[
	tuple[np.ndarray, np.ndarray, np.ndarray],
	tuple[np.ndarray, np.ndarray, np.ndarray],
]:
	flip_state_a = _sample_horizontal_flip_state(
		rng,
		probability=probability,
	)
	flip_state_b = _sample_horizontal_flip_state(
		rng,
		probability=probability,
	)
	if require_distinct and np.array_equal(flip_state_a, flip_state_b):
		axis = int(rng.integers(0, 2))
		flip_state_b[axis] = not bool(flip_state_b[axis])

	view_a, valid_mask_a = _augment_view(
		x,
		valid_mask,
		flip_inline=bool(flip_state_a[0]),
		flip_crossline=bool(flip_state_a[1]),
	)
	view_b, valid_mask_b = _augment_view(
		x,
		valid_mask,
		flip_inline=bool(flip_state_b[0]),
		flip_crossline=bool(flip_state_b[1]),
	)
	return (
		(view_a, valid_mask_a, flip_state_a),
		(view_b, valid_mask_b, flip_state_b),
	)


def _augment_view(
	x: np.ndarray,
	valid_mask: np.ndarray,
	*,
	flip_inline: bool,
	flip_crossline: bool,
) -> tuple[np.ndarray, np.ndarray]:
	amplitude_axes: list[int] = []
	mask_axes: list[int] = []
	if flip_inline:
		amplitude_axes.append(1)
		mask_axes.append(0)
	if flip_crossline:
		amplitude_axes.append(2)
		mask_axes.append(1)
	if not amplitude_axes:
		return x.copy(), valid_mask.copy()
	return (
		np.flip(x, axis=tuple(amplitude_axes)).copy(),
		np.flip(valid_mask, axis=tuple(mask_axes)).copy(),
	)


def _sample_canonical_token_indices(
	valid_mask: np.ndarray,
	patch_size_xyz: tuple[int, int, int],
	local_pairs_per_crop: int,
	rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[int, int, int]]:
	canonical_token_mask = reduce_valid_mask_to_tokens(
		valid_mask,
		patch_size_xyz=patch_size_xyz,
		min_valid_fraction=1.0,
	)
	valid_canonical_indices = np.flatnonzero(canonical_token_mask.ravel(order='C'))
	if valid_canonical_indices.size < local_pairs_per_crop:
		msg = (
			'base sample has fewer fully valid tokens than local_pairs_per_crop; '
			f'got {valid_canonical_indices.size} and {local_pairs_per_crop}'
		)
		raise ValueError(msg)
	canonical_indices = np.asarray(
		rng.choice(
			valid_canonical_indices,
			size=local_pairs_per_crop,
			replace=False,
		),
		dtype=np.int64,
	)
	token_shape = tuple(int(axis) for axis in canonical_token_mask.shape)
	return canonical_indices, token_shape


def _sample_xy_d4_transform_id(
	rng: np.random.Generator,
	*,
	reflection_probability: float,
) -> int:
	quarter_turns = int(rng.integers(0, 4))
	reflect_x = rng.random() < reflection_probability
	return quarter_turns + 4 * int(reflect_x)


def _apply_xy_d4(
	array: np.ndarray,
	transform_id: int,
	*,
	xy_axes: tuple[int, int],
) -> np.ndarray:
	if transform_id < 0 or transform_id > 7:
		msg = f'transform_id must be in [0, 7]; got {transform_id!r}'
		raise ValueError(msg)
	quarter_turns = transform_id % 4
	transformed = np.rot90(array, k=quarter_turns, axes=xy_axes)
	if transform_id >= 4:
		transformed = np.flip(transformed, axis=xy_axes[0])
	return transformed.copy()


def _map_token_indices_for_d4_view(
	canonical_indices: np.ndarray,
	token_shape: tuple[int, int, int],
	transform_id: int,
) -> np.ndarray:
	canonical_grid = np.arange(
		int(np.prod(token_shape)),
		dtype=np.int64,
	).reshape(token_shape)
	transformed_grid = _apply_xy_d4(
		canonical_grid,
		transform_id,
		xy_axes=(0, 1),
	)
	canonical_to_view = np.empty(transformed_grid.size, dtype=np.int64)
	canonical_to_view[transformed_grid.ravel(order='C')] = np.arange(
		transformed_grid.size,
		dtype=np.int64,
	)
	return canonical_to_view[canonical_indices]


def _apply_trace_drop(
	view: np.ndarray,
	transformed_valid_mask: np.ndarray,
	rng: np.random.Generator,
	*,
	probability: float,
) -> int:
	eligible_xy = transformed_valid_mask.any(axis=2)
	drop_xy = (rng.random(eligible_xy.shape) < probability) & eligible_xy
	drop_x, drop_y = np.nonzero(drop_xy)
	view[:, drop_x, drop_y, :] = 0.0
	return int(drop_x.size)


def _validate_square_xy(shape_xyz: tuple[int, int, int], name: str) -> None:
	if shape_xyz[0] != shape_xyz[1]:
		msg = f'{name} X/Y sizes must be equal; got {shape_xyz!r}'
		raise ValueError(msg)


def _validate_sample_shapes(x: np.ndarray, valid_mask: np.ndarray) -> None:
	if x.ndim != 4 or x.shape[0] != 1:
		msg = f'x must have shape [1, X, Y, Z]; got {x.shape!r}'
		raise ValueError(msg)
	if valid_mask.ndim != 3 or valid_mask.shape != x.shape[1:]:
		msg = (
			'local_valid_mask must have shape [X, Y, Z] matching x; '
			f'got {valid_mask.shape!r} and {x.shape!r}'
		)
		raise ValueError(msg)


def _require_array(sample: dict[str, object], key: str) -> np.ndarray:
	value = sample.get(key)
	if not isinstance(value, np.ndarray):
		msg = f'base sample {key!r} must be a NumPy array'
		raise TypeError(msg)
	return value


def _normalize_index(index: int, length: int) -> int:
	if isinstance(index, bool) or not isinstance(index, Integral):
		msg = f'index must be an integer; got {index!r}'
		raise TypeError(msg)
	normalized = int(index)
	if normalized < 0:
		normalized += length
	if normalized < 0 or normalized >= length:
		msg = f'index out of range: {index!r}'
		raise IndexError(msg)
	return normalized


def _validate_probability(value: object, name: str) -> float:
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{name} must be a real number; got {value!r}'
		raise TypeError(msg)
	probability = float(value)
	if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
		msg = f'{name} must be in [0, 1]; got {probability!r}'
		raise ValueError(msg)
	return probability


def _validate_positive_int(value: object, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	integer = int(value)
	if integer <= 0:
		msg = f'{name} must be positive; got {integer!r}'
		raise ValueError(msg)
	return integer


def _sample_horizontal_flip_state(
	rng: np.random.Generator,
	*,
	probability: float,
) -> np.ndarray:
	return np.asarray(
		[rng.random() < probability, rng.random() < probability],
		dtype=bool,
	)


def _map_token_indices_for_view(
	canonical_indices: np.ndarray,
	token_shape: tuple[int, int, int],
	flip_state: np.ndarray,
) -> np.ndarray:
	coordinates = np.asarray(
		np.unravel_index(canonical_indices, token_shape, order='C'),
		dtype=np.int64,
	)
	if bool(flip_state[0]):
		coordinates[0] = token_shape[0] - 1 - coordinates[0]
	if bool(flip_state[1]):
		coordinates[1] = token_shape[1] - 1 - coordinates[1]
	return np.asarray(
		np.ravel_multi_index(tuple(coordinates), token_shape, order='C'),
		dtype=np.int64,
	)


__all__ = [
	'BarlowTwinsPretrainDataset',
	'LocalBarlowTwinsD4TraceDropPretrainDataset',
	'LocalBarlowTwinsPretrainDataset',
]
