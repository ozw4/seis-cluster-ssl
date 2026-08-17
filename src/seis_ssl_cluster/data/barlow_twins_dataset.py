"""Two-view augmentation for amplitude Barlow Twins pretraining."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.data.crop_sampler import rng_for_sample

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

	def __getitem__(self, index: int) -> dict[str, object]:
		"""Return two augmented copies of one preprocessed physical crop."""
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
		view_a, valid_mask_a = _augment_view(
			x,
			valid_mask,
			flip_inline=bool(rng.random() < self.horizontal_flip_probability),
			flip_crossline=bool(rng.random() < self.horizontal_flip_probability),
		)
		view_b, valid_mask_b = _augment_view(
			x,
			valid_mask,
			flip_inline=bool(rng.random() < self.horizontal_flip_probability),
			flip_crossline=bool(rng.random() < self.horizontal_flip_probability),
		)
		return {
			'view_a': view_a,
			'view_b': view_b,
			'valid_mask_a': valid_mask_a,
			'valid_mask_b': valid_mask_b,
			'coords': base_sample.get('coords'),
		}


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


__all__ = ['BarlowTwinsPretrainDataset']
