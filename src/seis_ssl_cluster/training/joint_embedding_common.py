"""Shared local-view dataset selection for joint-embedding trainers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.config.schema import (
	HORIZONTAL_FLIP_ASYMMETRIC_NOISE_AUGMENTATION_POLICY,
	HORIZONTAL_FLIP_COLORED_NOISE_AUGMENTATION_POLICY,
	HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
	HORIZONTAL_FLIP_GAUSSIAN_TRACE_DROP_AUGMENTATION_POLICY,
	HORIZONTAL_FLIP_LAPLACE_NOISE_AUGMENTATION_POLICY,
	HORIZONTAL_FLIP_SMOOTH_GAIN_AUGMENTATION_POLICY,
	HORIZONTAL_FLIP_TOKEN_DROPOUT_AUGMENTATION_POLICY,
	HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_POLICY,
	HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_POLICY,
	IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
	XY_D4_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
	XY_D4_TRACE_DROP_AUGMENTATION_POLICY,
	XY_ROT90_ASYMMETRIC_NOISE_AUGMENTATION_POLICY,
	XY_ROT90_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
	XY_ROT90_LAPLACE_NOISE_AUGMENTATION_POLICY,
)
from seis_ssl_cluster.data.barlow_twins_dataset import (
	BarlowTwinsPretrainDataset,
	LocalBarlowTwinsD4TraceDropPretrainDataset,
	LocalBarlowTwinsPretrainDataset,
	RegionLocalBarlowTwinsPretrainDataset,
	XyRotationLocalBarlowTwinsPretrainDataset,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

	from seis_ssl_cluster.data.amplitude_dataset import AmplitudePretrainDataset


def build_local_joint_embedding_dataset(  # noqa: C901, PLR0911, PLR0912
	base_dataset: AmplitudePretrainDataset,
	*,
	local_pairs_per_crop: int,
	augmentations: Mapping[str, object],
	positive_window_tokens: tuple[int, int, int] | None = None,
) -> BarlowTwinsPretrainDataset:
	"""Build the local two-view dataset selected by augmentation policy."""
	augmentation_policy = augmentations.get('policy')
	if augmentation_policy is None:
		return _build_flip_local_dataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			positive_window_tokens=positive_window_tokens,
			horizontal_flip_probability=_floating(
				augmentations,
				'horizontal_flip_probability',
			),
		)
	if augmentation_policy == IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_POLICY:
		return _build_flip_local_dataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			positive_window_tokens=positive_window_tokens,
			horizontal_flip_probability=0.0,
			gaussian_noise_std=_floating(
				augmentations,
				'gaussian_noise_std',
			),
			require_distinct_horizontal_views=False,
		)
	if augmentation_policy == HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_POLICY:
		return _build_flip_local_dataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			positive_window_tokens=positive_window_tokens,
			horizontal_flip_probability=_floating(
				augmentations,
				'horizontal_flip_probability',
			),
			gaussian_noise_std=_floating(
				augmentations,
				'gaussian_noise_std',
			),
		)
	if augmentation_policy == HORIZONTAL_FLIP_ASYMMETRIC_NOISE_AUGMENTATION_POLICY:
		return _build_flip_local_dataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			positive_window_tokens=positive_window_tokens,
			horizontal_flip_probability=_floating(
				augmentations,
				'horizontal_flip_probability',
			),
			gaussian_noise_std=_floating(augmentations, 'gaussian_noise_std'),
			noise_second_view_only=True,
		)
	if augmentation_policy == HORIZONTAL_FLIP_GAUSSIAN_TRACE_DROP_AUGMENTATION_POLICY:
		return _build_flip_local_dataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			positive_window_tokens=positive_window_tokens,
			horizontal_flip_probability=_floating(
				augmentations,
				'horizontal_flip_probability',
			),
			gaussian_noise_std=_floating(augmentations, 'gaussian_noise_std'),
			trace_drop_probability=_floating(
				augmentations,
				'trace_drop_probability',
			),
		)
	if augmentation_policy == HORIZONTAL_FLIP_COLORED_NOISE_AUGMENTATION_POLICY:
		return _build_flip_local_dataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			positive_window_tokens=positive_window_tokens,
			horizontal_flip_probability=_floating(
				augmentations,
				'horizontal_flip_probability',
			),
			gaussian_noise_std=_floating(augmentations, 'gaussian_noise_std'),
			noise_z_smoothing=int(_floating(augmentations, 'noise_z_smoothing')),
		)
	if augmentation_policy == HORIZONTAL_FLIP_LAPLACE_NOISE_AUGMENTATION_POLICY:
		return _build_flip_local_dataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			positive_window_tokens=positive_window_tokens,
			horizontal_flip_probability=_floating(
				augmentations,
				'horizontal_flip_probability',
			),
			gaussian_noise_std=_floating(augmentations, 'gaussian_noise_std'),
			noise_kind='laplace',
		)
	if augmentation_policy in (
		XY_ROT90_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
		XY_ROT90_LAPLACE_NOISE_AUGMENTATION_POLICY,
		XY_ROT90_ASYMMETRIC_NOISE_AUGMENTATION_POLICY,
		XY_D4_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
	):
		return XyRotationLocalBarlowTwinsPretrainDataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			rotation_only=(
				augmentation_policy != XY_D4_GAUSSIAN_NOISE_AUGMENTATION_POLICY
			),
			positive_window_tokens=positive_window_tokens,
			gaussian_noise_std=_floating(augmentations, 'gaussian_noise_std'),
			noise_kind=(
				'laplace'
				if augmentation_policy == XY_ROT90_LAPLACE_NOISE_AUGMENTATION_POLICY
				else 'gaussian'
			),
			noise_second_view_only=(
				augmentation_policy
				== XY_ROT90_ASYMMETRIC_NOISE_AUGMENTATION_POLICY
			),
		)
	if augmentation_policy == HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_POLICY:
		return _build_flip_local_dataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			positive_window_tokens=positive_window_tokens,
			horizontal_flip_probability=_floating(
				augmentations,
				'horizontal_flip_probability',
			),
			trace_drop_probability=_floating(
				augmentations,
				'trace_drop_probability',
			),
		)
	if (
		augmentation_policy
		== HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_POLICY
	):
		return _build_flip_local_dataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			positive_window_tokens=positive_window_tokens,
			horizontal_flip_probability=_floating(
				augmentations,
				'horizontal_flip_probability',
			),
			z_filter_side_weight=_floating(
				augmentations,
				'z_filter_side_weight',
			),
		)
	if augmentation_policy == HORIZONTAL_FLIP_TOKEN_DROPOUT_AUGMENTATION_POLICY:
		return _build_flip_local_dataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			positive_window_tokens=positive_window_tokens,
			horizontal_flip_probability=_floating(
				augmentations,
				'horizontal_flip_probability',
			),
			token_dropout_fraction=_floating(
				augmentations,
				'token_dropout_fraction',
			),
		)
	if augmentation_policy == HORIZONTAL_FLIP_SMOOTH_GAIN_AUGMENTATION_POLICY:
		return _build_flip_local_dataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			positive_window_tokens=positive_window_tokens,
			horizontal_flip_probability=_floating(
				augmentations,
				'horizontal_flip_probability',
			),
			smooth_gain_jitter=_floating(
				augmentations,
				'gain_jitter',
			),
		)
	if augmentation_policy == XY_D4_TRACE_DROP_AUGMENTATION_POLICY:
		_reject_positive_window(augmentation_policy, positive_window_tokens)
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


def _build_flip_local_dataset(
	base_dataset: AmplitudePretrainDataset,
	*,
	local_pairs_per_crop: int,
	positive_window_tokens: tuple[int, int, int] | None,
	**nuisance_kwargs: float | bool | str,
) -> BarlowTwinsPretrainDataset:
	if positive_window_tokens is None:
		return LocalBarlowTwinsPretrainDataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			**nuisance_kwargs,
		)
	return RegionLocalBarlowTwinsPretrainDataset(
		base_dataset,
		local_pairs_per_crop=local_pairs_per_crop,
		positive_window_tokens=positive_window_tokens,
		**nuisance_kwargs,
	)


def _reject_positive_window(
	augmentation_policy: str,
	positive_window_tokens: tuple[int, int, int] | None,
) -> None:
	if positive_window_tokens is not None:
		msg = (
			'positive_window_tokens is not supported for augmentation policy '
			f'{augmentation_policy!r}'
		)
		raise ValueError(msg)


def _floating(values: Mapping[str, object], key: str) -> float:
	value = values.get(key)
	if isinstance(value, bool) or not isinstance(value, int | float):
		msg = f'{key} must be numeric'
		raise TypeError(msg)
	return float(value)


__all__ = ['build_local_joint_embedding_dataset']
