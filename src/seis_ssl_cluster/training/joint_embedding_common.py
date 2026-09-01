"""Shared local-view dataset selection for joint-embedding trainers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.config.schema import (
	HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
	HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_POLICY,
	HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_POLICY,
	IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
	XY_D4_TRACE_DROP_AUGMENTATION_POLICY,
)
from seis_ssl_cluster.data.barlow_twins_dataset import (
	BarlowTwinsPretrainDataset,
	LocalBarlowTwinsD4TraceDropPretrainDataset,
	LocalBarlowTwinsPretrainDataset,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

	from seis_ssl_cluster.data.amplitude_dataset import AmplitudePretrainDataset


def build_local_joint_embedding_dataset(
	base_dataset: AmplitudePretrainDataset,
	*,
	local_pairs_per_crop: int,
	augmentations: Mapping[str, object],
) -> BarlowTwinsPretrainDataset:
	"""Build the local two-view dataset selected by augmentation policy."""
	augmentation_policy = augmentations.get('policy')
	if augmentation_policy is None:
		return LocalBarlowTwinsPretrainDataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			horizontal_flip_probability=_floating(
				augmentations,
				'horizontal_flip_probability',
			),
		)
	if augmentation_policy == IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_POLICY:
		return LocalBarlowTwinsPretrainDataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			horizontal_flip_probability=0.0,
			gaussian_noise_std=_floating(
				augmentations,
				'gaussian_noise_std',
			),
			require_distinct_horizontal_views=False,
		)
	if augmentation_policy == HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_POLICY:
		return LocalBarlowTwinsPretrainDataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			horizontal_flip_probability=_floating(
				augmentations,
				'horizontal_flip_probability',
			),
			gaussian_noise_std=_floating(
				augmentations,
				'gaussian_noise_std',
			),
		)
	if augmentation_policy == HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_POLICY:
		return LocalBarlowTwinsPretrainDataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			horizontal_flip_probability=_floating(
				augmentations,
				'horizontal_flip_probability',
			),
			trace_drop_probability=_floating(
				augmentations,
				'trace_drop_probability',
			),
		)
	if augmentation_policy == HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_POLICY:
		return LocalBarlowTwinsPretrainDataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			horizontal_flip_probability=_floating(
				augmentations,
				'horizontal_flip_probability',
			),
			z_filter_side_weight=_floating(
				augmentations,
				'z_filter_side_weight',
			),
		)
	if augmentation_policy == XY_D4_TRACE_DROP_AUGMENTATION_POLICY:
		return LocalBarlowTwinsD4TraceDropPretrainDataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			reflection_probability=_floating(
				augmentations,
				'reflection_probability',
			),
			trace_drop_probability=_floating(
				augmentations,
				'trace_drop_probability',
			),
		)
	raise ValueError(
		'unsupported local joint-embedding augmentation policy: '
		f'{augmentation_policy!r}'
	)


def _floating(values: Mapping[str, object], key: str) -> float:
	value = values.get(key)
	if isinstance(value, bool) or not isinstance(value, int | float):
		raise TypeError(f'{key} must be numeric')
	return float(value)


__all__ = ['build_local_joint_embedding_dataset']
