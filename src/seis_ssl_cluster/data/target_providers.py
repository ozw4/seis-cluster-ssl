"""Contracts for adding supervision targets to amplitude samples."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from seis_ssl_cluster.stratigraphy.targets import (
	StratPseudoTargetArrays,
	StratPseudoTargetInput,
	load_pseudo_target_arrays,
)

if TYPE_CHECKING:
	from collections.abc import Sequence

	from seis_ssl_cluster.data.schema import CropRequest, SurveyManifest

XYZ = tuple[int, int, int]


@dataclass(frozen=True)
class TargetProviderContext:
	"""Immutable crop metadata supplied to a target provider."""

	manifest: SurveyManifest
	crop_request: CropRequest
	patch_size_xyz: tuple[int, int, int]
	token_start_xyz: tuple[int, int, int]
	token_size_xyz: tuple[int, int, int]
	token_valid_mask: np.ndarray


class TargetProvider(Protocol):
	"""Contract for validating and adding targets to dataset samples."""

	def validate_manifests(
		self,
		manifests: Sequence[SurveyManifest],
		*,
		local_crop_size_xyz: tuple[int, int, int],
		patch_size_xyz: tuple[int, int, int],
		token_grid_shape_xyz: tuple[int, int, int],
	) -> None:
		"""Validate provider-specific requirements for all manifests."""
		...

	def add_targets(
		self,
		sample: MutableMapping[str, object],
		context: TargetProviderContext,
	) -> None:
		"""Add supervision fields to ``sample`` in place."""
		...

	def sample_is_acceptable(self, sample: Mapping[str, object]) -> bool:
		"""Return whether a populated sample should be accepted."""
		...

	def rejection_message(
		self,
		*,
		survey_id: str,
		max_resample_attempts: int,
		last_valid_fraction: float,
	) -> str:
		"""Describe why repeated sample attempts were rejected."""
		...


class NoTargetProvider:
	"""Target provider for unsupervised samples."""

	def validate_manifests(
		self,
		manifests: Sequence[SurveyManifest],
		*,
		local_crop_size_xyz: tuple[int, int, int],
		patch_size_xyz: tuple[int, int, int],
		token_grid_shape_xyz: tuple[int, int, int],
	) -> None:
		"""Accept all manifests without provider-specific validation."""

	def add_targets(
		self,
		sample: MutableMapping[str, object],
		context: TargetProviderContext,
	) -> None:
		"""Leave the sample unchanged."""

	def sample_is_acceptable(self, sample: Mapping[str, object]) -> bool:
		"""Accept every sample."""
		del sample
		return True

	def rejection_message(
		self,
		*,
		survey_id: str,
		max_resample_attempts: int,
		last_valid_fraction: float,
	) -> str:
		"""Return an explicit diagnostic for an unreachable rejection."""
		return (
			'NoTargetProvider accepts every sample; rejection is unexpected '
			f'for survey {survey_id!r} after {max_resample_attempts} attempts '
			f'(last_valid_fraction={last_valid_fraction:.6f}).'
		)


class StratPseudoTargetProvider:
	"""Add token-grid stratigraphic pseudo-targets to amplitude samples."""

	def __init__(  # noqa: D107
		self,
		pseudo_target_inputs: Sequence[StratPseudoTargetInput],
		*,
		min_confidence: float = 0.0,
	) -> None:
		self.min_confidence = _validate_fraction(
			min_confidence,
			'min_confidence',
		)
		self._pseudo_target_inputs = _pseudo_targets_by_survey(
			pseudo_target_inputs,
		)
		self._pseudo_target_arrays: dict[str, StratPseudoTargetArrays] = {}

	def validate_manifests(
		self,
		manifests: Sequence[SurveyManifest],
		*,
		local_crop_size_xyz: tuple[int, int, int],
		patch_size_xyz: tuple[int, int, int],
		token_grid_shape_xyz: tuple[int, int, int],
	) -> None:
		"""Validate pseudo-target presence and crop coverage for each survey."""
		manifest_ids = {manifest.survey_id for manifest in manifests}
		missing_ids = sorted(manifest_ids - self._pseudo_target_inputs.keys())
		if missing_ids:
			msg = f'missing pseudo-target inputs for surveys: {missing_ids!r}'
			raise ValueError(msg)

		for manifest in manifests:
			arrays = self._pseudo_targets_for_survey(manifest.survey_id)
			required_shape = tuple(
				((shape_axis - crop_axis) // patch_axis) + token_axis
				for shape_axis, crop_axis, patch_axis, token_axis in zip(
					manifest.amplitude.shape_xyz,
					local_crop_size_xyz,
					patch_size_xyz,
					token_grid_shape_xyz,
					strict=True,
				)
			)
			if any(
				target_axis < required_axis
				for target_axis, required_axis in zip(
					arrays.labels.shape,
					required_shape,
					strict=True,
				)
			):
				msg = (
					f'survey {manifest.survey_id!r} pseudo-target grid is too '
					f'small for token-aligned crops; got {arrays.labels.shape!r}, '
					f'need at least {required_shape!r}'
				)
				raise ValueError(msg)

	def add_targets(
		self,
		sample: MutableMapping[str, object],
		context: TargetProviderContext,
	) -> None:
		"""Slice pseudo-targets for the context token window into ``sample``."""
		arrays = self._pseudo_targets_for_survey(context.manifest.survey_id)
		token_stop = tuple(
			start_axis + size_axis
			for start_axis, size_axis in zip(
				context.token_start_xyz,
				context.token_size_xyz,
				strict=True,
			)
		)
		token_slices = tuple(
			slice(start_axis, stop_axis)
			for start_axis, stop_axis in zip(
				context.token_start_xyz,
				token_stop,
				strict=True,
			)
		)
		labels = np.asarray(arrays.labels[token_slices], dtype=np.int64).copy()
		confidence = np.asarray(
			arrays.confidence[token_slices],
			dtype=np.float32,
		).copy()
		boundary_weight = np.asarray(
			arrays.boundary_weight[token_slices],
			dtype=np.float32,
		).copy()
		pseudo_valid = np.asarray(arrays.valid_tokens[token_slices], dtype=bool)
		token_valid = _require_bool_array(
			{'token_valid_mask': context.token_valid_mask},
			'token_valid_mask',
		)
		strat_valid_mask = np.logical_and(pseudo_valid, token_valid)
		labels[~strat_valid_mask] = -1
		confidence[~strat_valid_mask] = 0.0
		boundary_weight[~strat_valid_mask] = 0.0
		_validate_boundary_weight_sample(boundary_weight, strat_valid_mask)
		sample['strat_labels'] = labels
		sample['strat_confidence'] = confidence
		sample['strat_boundary_weight'] = boundary_weight
		sample['strat_valid_mask'] = strat_valid_mask.astype(bool, copy=False)

		coords = sample['coords']
		if not isinstance(coords, MutableMapping):
			msg = 'sample coords must be a mutable mapping'
			raise TypeError(msg)
		coords['token_start_xyz'] = cast('XYZ', context.token_start_xyz)
		coords['token_size_xyz'] = cast('XYZ', context.token_size_xyz)

	def sample_is_acceptable(self, sample: Mapping[str, object]) -> bool:
		"""Return whether the sample contains confident supervised tokens."""
		strat_valid = _require_bool_array(sample, 'strat_valid_mask')
		confidence = _require_float_array(sample, 'strat_confidence')
		if confidence.shape != strat_valid.shape:
			msg = (
				'strat_confidence shape must match strat_valid_mask shape; '
				f'got {confidence.shape!r} and {strat_valid.shape!r}'
			)
			raise ValueError(msg)
		boundary_weight = _require_float32_array(sample, 'strat_boundary_weight')
		_validate_boundary_weight_sample(boundary_weight, strat_valid)
		return bool(
			np.any(
				strat_valid
				& (confidence >= self.min_confidence)
				& (boundary_weight > 0.0)
			),
		)

	def rejection_message(
		self,
		*,
		survey_id: str,
		max_resample_attempts: int,
		last_valid_fraction: float,
	) -> str:
		"""Describe exhausted pseudo-target crop resampling."""
		return (
			f'survey {survey_id!r} did not produce a pseudo-target crop with at '
			f'least one valid supervised token at '
			f'min_confidence={self.min_confidence:.6f} after '
			f'max_resample_attempts={max_resample_attempts}; no positive '
			f'boundary/effective weight token satisfied the confidence threshold; '
			f'last local valid '
			f'fraction was {last_valid_fraction:.6f}.'
		)

	def _pseudo_targets_for_survey(
		self,
		survey_id: str,
	) -> StratPseudoTargetArrays:
		if survey_id not in self._pseudo_target_arrays:
			self._pseudo_target_arrays[survey_id] = load_pseudo_target_arrays(
				self._pseudo_target_inputs[survey_id],
				mmap_mode='r',
			)
		return self._pseudo_target_arrays[survey_id]


def _pseudo_targets_by_survey(
	pseudo_target_inputs: Sequence[StratPseudoTargetInput],
) -> dict[str, StratPseudoTargetInput]:
	if not pseudo_target_inputs:
		msg = 'pseudo_target_inputs must contain at least one survey'
		raise ValueError(msg)
	by_survey: dict[str, StratPseudoTargetInput] = {}
	for item in pseudo_target_inputs:
		if not isinstance(item, StratPseudoTargetInput):
			msg = (
				'pseudo_target_inputs must contain StratPseudoTargetInput '
				f'items; got {type(item).__name__}'
			)
			raise TypeError(msg)
		if item.survey_id in by_survey:
			msg = f'duplicate pseudo-target input for survey {item.survey_id!r}'
			raise ValueError(msg)
		by_survey[item.survey_id] = item
	return by_survey


def _require_bool_array(sample: Mapping[str, object], key: str) -> np.ndarray:
	value = sample[key]
	if not isinstance(value, np.ndarray):
		msg = f'{key} must be a NumPy array; got {type(value).__name__}'
		raise TypeError(msg)
	if value.dtype != np.bool_:
		msg = f'{key} dtype must be bool; got {value.dtype}'
		raise TypeError(msg)
	return value


def _require_float_array(sample: Mapping[str, object], key: str) -> np.ndarray:
	value = sample[key]
	if not isinstance(value, np.ndarray):
		msg = f'{key} must be a NumPy array; got {type(value).__name__}'
		raise TypeError(msg)
	if value.dtype.kind != 'f':
		msg = f'{key} dtype must be floating point; got {value.dtype}'
		raise TypeError(msg)
	return value


def _require_float32_array(sample: Mapping[str, object], key: str) -> np.ndarray:
	value = sample[key]
	if not isinstance(value, np.ndarray):
		msg = f'{key} must be a NumPy array; got {type(value).__name__}'
		raise TypeError(msg)
	if value.dtype != np.float32:
		msg = f'{key} dtype must be float32; got {value.dtype}'
		raise TypeError(msg)
	return value


def _validate_boundary_weight_sample(
	boundary_weight: np.ndarray,
	strat_valid_mask: np.ndarray,
) -> None:
	if boundary_weight.shape != strat_valid_mask.shape:
		msg = (
			'strat_boundary_weight shape must match strat_valid_mask shape; '
			f'got {boundary_weight.shape!r} and {strat_valid_mask.shape!r}'
		)
		raise ValueError(msg)
	if not np.all(np.isfinite(boundary_weight)):
		msg = 'strat_boundary_weight must be finite'
		raise ValueError(msg)
	if np.any((boundary_weight < 0.0) | (boundary_weight > 1.0)):
		msg = 'strat_boundary_weight values must be in [0, 1]'
		raise ValueError(msg)
	if np.any(boundary_weight[~strat_valid_mask] != 0.0):
		msg = 'strat_boundary_weight must be 0.0 where strat_valid_mask is false'
		raise ValueError(msg)


def _validate_fraction(value: object, name: str) -> float:
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{name} must be a real number; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not 0.0 <= fraction <= 1.0:
		msg = f'{name} must be in [0, 1]; got {fraction!r}'
		raise ValueError(msg)
	return fraction


__all__ = [
	'NoTargetProvider',
	'StratPseudoTargetProvider',
	'TargetProvider',
	'TargetProviderContext',
]
