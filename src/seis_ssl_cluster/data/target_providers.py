"""Contracts for adding supervision targets to amplitude samples."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.stratigraphy.lateral_targets import (
	load_multi_head_lateral_target_manifest,
)
from seis_ssl_cluster.stratigraphy.multi_head import (
	load_multi_head_target_manifest,
	validate_multi_head_target_reference,
)
from seis_ssl_cluster.stratigraphy.state_posterior import (
	load_multi_head_state_posterior_manifest,
)
from seis_ssl_cluster.stratigraphy.targets import (
	StratPseudoTargetArrays,
	StratPseudoTargetInput,
	load_pseudo_target_arrays,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus_targets import (
	load_multi_head_xy_neighbor_consensus_target_manifest,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_unanimous_targets import (
	load_multi_head_xy_neighbor_unanimous_target_manifest,
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


@dataclass(frozen=True)
class StratMultiHeadTargetInput:
	"""One validated head reference from a multi-head target manifest."""

	k: int
	survey_id: str
	labels_path: Path
	confidence_path: Path
	valid_tokens_path: Path
	metadata_path: Path
	hashes: Mapping[str, str]


@dataclass(frozen=True)
class StratMultiHeadTargetManifest:
	"""Ordered multi-head target references grouped by survey."""

	head_ks: tuple[int, ...]
	by_survey: Mapping[str, tuple[StratMultiHeadTargetInput, ...]]
	common_valid_token_sha256: Mapping[str, str]
	invalid_label_policy: str = 'minus_one'


@dataclass(frozen=True)
class StratMultiHeadLateralTargetManifest:
	"""Strict M5-LS provenance plus its compatible hard-target adapter."""

	target_manifest: StratMultiHeadTargetManifest
	target_semantics: str
	source_hard_manifest: Mapping[str, str]
	source_posterior_manifest: Mapping[str, str]
	smoothing: Mapping[str, object]


@dataclass(frozen=True)
class StratMultiHeadXYNeighborConsensusTargetManifest:
	"""Strict XY-consensus provenance plus its hard-target adapter."""

	target_manifest: StratMultiHeadTargetManifest
	target_representation: str
	target_semantics: str
	source_hard_manifest: Mapping[str, str]
	smoothing: Mapping[str, object]


@dataclass(frozen=True)
class StratMultiHeadXYNeighborUnanimousTargetManifest:
	"""Strict XY-unanimous provenance plus its hard-target adapter."""

	target_manifest: StratMultiHeadTargetManifest
	target_representation: str
	target_semantics: str
	source_hard_manifest: Mapping[str, str]
	smoothing: Mapping[str, object]


@dataclass(frozen=True)
class StratMultiHeadTargetArrays:
	"""Memory-mapped target arrays for a single survey and head."""

	labels: np.ndarray
	confidence: np.ndarray
	valid_tokens: np.ndarray


@dataclass(frozen=True)
class StratMultiHeadPosteriorInput:
	"""One validated posterior reference from a multi-head export."""

	k: int
	survey_id: str
	posterior_path: Path
	valid_tokens_path: Path
	metadata_path: Path
	hashes: Mapping[str, str]


@dataclass(frozen=True)
class StratMultiHeadPosteriorManifest:
	"""Ordered posterior references grouped by survey."""

	head_ks: tuple[int, ...]
	by_survey: Mapping[str, tuple[StratMultiHeadPosteriorInput, ...]]


@dataclass(frozen=True)
class StratMultiHeadPosteriorArrays:
	"""Process-local memory maps for one survey and posterior head."""

	posterior: np.ndarray
	valid_tokens: np.ndarray


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


class MultiHeadStratPseudoTargetProvider:
	"""Add ordered, token-aligned pseudo-targets for every manifest head."""

	def __init__(  # noqa: D107
		self,
		multi_head_target_manifest: StratMultiHeadTargetManifest | str | Path,
		*,
		min_confidence: float = 0.0,
	) -> None:
		self.min_confidence = _validate_fraction(
			min_confidence,
			'min_confidence',
		)
		self.manifest = _coerce_multi_head_target_manifest(
			multi_head_target_manifest,
		)
		self._allow_source_invalid_labels = (
			self.manifest.invalid_label_policy == 'preserve_source'
		)
		self._pseudo_target_arrays: dict[
			str, dict[int, StratMultiHeadTargetArrays]
		] = {}
		self._last_empty_head_ks: tuple[int, ...] = ()

	def validate_manifests(
		self,
		manifests: Sequence[SurveyManifest],
		*,
		local_crop_size_xyz: tuple[int, int, int],
		patch_size_xyz: tuple[int, int, int],
		token_grid_shape_xyz: tuple[int, int, int],
	) -> None:
		"""Validate manifest coverage and target-grid crop coverage."""
		missing_ids = sorted(
			{manifest.survey_id for manifest in manifests}
			- self.manifest.by_survey.keys(),
		)
		if missing_ids:
			msg = f'missing multi-head target inputs for surveys: {missing_ids!r}'
			raise ValueError(msg)
		for manifest in manifests:
			arrays_by_k = self._pseudo_targets_for_survey(manifest.survey_id)
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
			for k, arrays in arrays_by_k.items():
				if any(
					target_axis < required_axis
					for target_axis, required_axis in zip(
						arrays.labels.shape,
						required_shape,
						strict=True,
					)
				):
					msg = (
						f'survey {manifest.survey_id!r} multi-head k={k} target '
						f'grid is too small for token-aligned crops; got '
						f'{arrays.labels.shape!r}, need at least {required_shape!r}'
					)
					raise ValueError(msg)

	def add_targets(
		self,
		sample: MutableMapping[str, object],
		context: TargetProviderContext,
	) -> None:
		"""Slice every head using one crop geometry and source-validity mask."""
		arrays_by_k = self._pseudo_targets_for_survey(context.manifest.survey_id)
		token_slices = _token_slices(context)
		token_valid = _require_bool_array(
			{'token_valid_mask': context.token_valid_mask},
			'token_valid_mask',
		)
		if token_valid.shape != context.token_size_xyz:
			msg = (
				'token_valid_mask shape must match token_size_xyz; got '
				f'{token_valid.shape!r} and {context.token_size_xyz!r}'
			)
			raise ValueError(msg)

		pseudo_valid: np.ndarray | None = None
		targets: dict[str, dict[str, np.ndarray]] = {}
		for k in self.manifest.head_ks:
			arrays = arrays_by_k[k]
			labels = np.asarray(arrays.labels[token_slices])
			confidence = np.asarray(arrays.confidence[token_slices])
			current_valid = np.asarray(arrays.valid_tokens[token_slices])
			_validate_multi_head_crop(
				labels,
				confidence,
				current_valid,
				k=k,
				allow_source_invalid_labels=self._allow_source_invalid_labels,
			)
			if pseudo_valid is None:
				pseudo_valid = current_valid
			elif not np.array_equal(pseudo_valid, current_valid):
				msg = (
					'multi-head valid_tokens must match across heads; '
					f'k={self.manifest.head_ks[0]} and k={k} differ'
				)
				raise ValueError(msg)

		if pseudo_valid is None:
			raise AssertionError('multi-head target manifest has no heads')
		valid_mask = np.logical_and(pseudo_valid, token_valid)
		for k in self.manifest.head_ks:
			arrays = arrays_by_k[k]
			labels = np.asarray(arrays.labels[token_slices], dtype=np.int64).copy()
			confidence = np.asarray(
				arrays.confidence[token_slices], dtype=np.float32
			).copy()
			labels[~valid_mask] = -1
			confidence[~valid_mask] = 0.0
			boundary_weight = valid_mask.astype(np.float32, copy=True)
			_validate_multi_head_sample(
				labels,
				confidence,
				boundary_weight,
				valid_mask,
				k=k,
			)
			targets[f'k{k}'] = {
				'labels': labels,
				'confidence': confidence,
				'boundary_weight': boundary_weight,
				'valid_mask': valid_mask,
			}
		sample['strat_multi_targets'] = targets
		coords = sample['coords']
		if not isinstance(coords, MutableMapping):
			msg = 'sample coords must be a mutable mapping'
			raise TypeError(msg)
		coords['token_start_xyz'] = cast('XYZ', context.token_start_xyz)
		coords['token_size_xyz'] = cast('XYZ', context.token_size_xyz)
		coords['strat_multi_target_metadata'] = {
			'boundary_weight_source': 'valid_token_indicator',
		}

	def sample_is_acceptable(self, sample: Mapping[str, object]) -> bool:
		"""Require a positive effective-weight token after thresholding per head."""
		targets = _require_multi_head_targets(sample)
		empty_head_ks: list[int] = []
		for k in self.manifest.head_ks:
			target = targets[f'k{k}']
			labels = _require_int64_array(target, 'labels')
			confidence = _require_float32_array(target, 'confidence')
			boundary_weight = _require_float32_array(target, 'boundary_weight')
			valid_mask = _require_bool_array(target, 'valid_mask')
			_validate_multi_head_sample(
				labels,
				confidence,
				boundary_weight,
				valid_mask,
				k=k,
			)
			if not np.any(
				valid_mask
				& (confidence >= self.min_confidence)
				& (boundary_weight > 0.0),
			):
				empty_head_ks.append(k)
		self._last_empty_head_ks = tuple(empty_head_ks)
		return not empty_head_ks

	def rejection_message(
		self,
		*,
		survey_id: str,
		max_resample_attempts: int,
		last_valid_fraction: float,
	) -> str:
		"""Describe exhausted multi-head crop resampling."""
		empty_heads = ', '.join(f'K{k}' for k in self._last_empty_head_ks)
		return (
			f'survey {survey_id!r} did not produce a multi-head pseudo-target '
			f'crop with at least one valid supervised token per head at '
			f'min_confidence={self.min_confidence:.6f} after '
			f'max_resample_attempts={max_resample_attempts}; empty heads: '
			f'{empty_heads or "unknown"}; last local valid fraction was '
			f'{last_valid_fraction:.6f}.'
		)

	def _pseudo_targets_for_survey(
		self,
		survey_id: str,
	) -> dict[int, StratMultiHeadTargetArrays]:
		if survey_id not in self._pseudo_target_arrays:
			self._pseudo_target_arrays[survey_id] = {
				item.k: StratMultiHeadTargetArrays(
					labels=np.load(item.labels_path, mmap_mode='r'),
					confidence=np.load(item.confidence_path, mmap_mode='r'),
					valid_tokens=np.load(item.valid_tokens_path, mmap_mode='r'),
				)
				for item in self.manifest.by_survey[survey_id]
			}
		return self._pseudo_target_arrays[survey_id]


class MultiHeadStratPosteriorProvider:
	"""Add token-aligned soft posterior targets without hard-target fields."""

	def __init__(  # noqa: D107
		self,
		multi_head_posterior_manifest: StratMultiHeadPosteriorManifest | str | Path,
	) -> None:
		self.manifest = _coerce_multi_head_posterior_manifest(
			multi_head_posterior_manifest,
		)
		self._posterior_arrays: dict[str, dict[int, StratMultiHeadPosteriorArrays]] = {}

	def validate_manifests(
		self,
		manifests: Sequence[SurveyManifest],
		*,
		local_crop_size_xyz: tuple[int, int, int],
		patch_size_xyz: tuple[int, int, int],
		token_grid_shape_xyz: tuple[int, int, int],
	) -> None:
		"""Validate that every survey supports token-aligned posterior crops."""
		missing_ids = sorted(
			{manifest.survey_id for manifest in manifests}
			- self.manifest.by_survey.keys(),
		)
		if missing_ids:
			raise ValueError(
				f'missing multi-head posterior inputs for surveys: {missing_ids!r}',
			)
		for manifest in manifests:
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
			for k, arrays in self._posteriors_for_survey(manifest.survey_id).items():
				if any(
					axis < required_axis
					for axis, required_axis in zip(
						arrays.posterior.shape[:3], required_shape, strict=True
					)
				):
					raise ValueError(
						f'survey {manifest.survey_id!r} multi-head posterior k={k} '
						f'grid is too small for token-aligned crops; got '
						f'{arrays.posterior.shape[:3]!r}, need at least '
						f'{required_shape!r}',
					)

	def add_targets(
		self,
		sample: MutableMapping[str, object],
		context: TargetProviderContext,
	) -> None:
		"""Materialize only one common posterior crop from each memory map."""
		arrays_by_k = self._posteriors_for_survey(context.manifest.survey_id)
		token_valid = _require_bool_array(
			{'token_valid_mask': context.token_valid_mask}, 'token_valid_mask'
		)
		if token_valid.shape != context.token_size_xyz:
			raise ValueError(
				'token_valid_mask shape must match token_size_xyz; got '
				f'{token_valid.shape!r} and {context.token_size_xyz!r}',
			)
		token_slices = _token_slices(context)
		common_valid: np.ndarray | None = None
		crops: dict[int, tuple[np.ndarray, np.ndarray]] = {}
		for k in self.manifest.head_ks:
			arrays = arrays_by_k[k]
			posterior = np.asarray(
				arrays.posterior[(*token_slices, slice(None))], dtype=np.float32
			).copy()
			valid_mask = np.asarray(
				arrays.valid_tokens[token_slices], dtype=bool
			).copy()
			_validate_multi_head_posterior_crop(posterior, valid_mask, k=k)
			if common_valid is None:
				common_valid = valid_mask
			elif not np.array_equal(common_valid, valid_mask):
				raise ValueError(
					'multi-head posterior valid_tokens must match across heads; '
					f'k={self.manifest.head_ks[0]} and k={k} differ',
				)
			crops[k] = (posterior, valid_mask)
		if common_valid is None:  # pragma: no cover - manifest validation rejects this
			raise AssertionError('multi-head posterior manifest has no heads')
		effective_valid = np.logical_and(common_valid, token_valid)
		targets: dict[str, dict[str, np.ndarray]] = {}
		for k in self.manifest.head_ks:
			posterior, _ = crops[k]
			posterior[~effective_valid] = 0.0
			_validate_multi_head_posterior_crop(posterior, effective_valid, k=k)
			targets[f'k{k}'] = {
				'posterior': posterior,
				'valid_mask': effective_valid.copy(),
			}
		sample['strat_multi_posteriors'] = targets
		coords = sample['coords']
		if not isinstance(coords, MutableMapping):
			raise TypeError('sample coords must be a mutable mapping')
		coords['token_start_xyz'] = cast('XYZ', context.token_start_xyz)
		coords['token_size_xyz'] = cast('XYZ', context.token_size_xyz)

	def sample_is_acceptable(self, sample: Mapping[str, object]) -> bool:
		"""Require at least one common valid token for the posterior crop."""
		targets = _require_multi_head_posteriors(sample)
		common_valid: np.ndarray | None = None
		for k in self.manifest.head_ks:
			target = targets[f'k{k}']
			posterior = _require_float32_array(target, 'posterior')
			valid_mask = _require_bool_array(target, 'valid_mask')
			_validate_multi_head_posterior_crop(posterior, valid_mask, k=k)
			if common_valid is None:
				common_valid = valid_mask
			elif not np.array_equal(common_valid, valid_mask):
				raise ValueError('multi-head posterior valid masks must match')
		return bool(common_valid is not None and np.any(common_valid))

	def rejection_message(
		self,
		*,
		survey_id: str,
		max_resample_attempts: int,
		last_valid_fraction: float,
	) -> str:
		"""Describe an exhausted posterior crop resampling attempt."""
		return (
			f'survey {survey_id!r} did not produce a multi-head posterior crop '
			f'with at least one common valid token after '
			f'max_resample_attempts={max_resample_attempts}; last local valid '
			f'fraction was {last_valid_fraction:.6f}.'
		)

	def _posteriors_for_survey(
		self, survey_id: str
	) -> dict[int, StratMultiHeadPosteriorArrays]:
		if survey_id not in self._posterior_arrays:
			self._posterior_arrays[survey_id] = {
				item.k: StratMultiHeadPosteriorArrays(
					posterior=np.load(item.posterior_path, mmap_mode='r'),
					valid_tokens=np.load(item.valid_tokens_path, mmap_mode='r'),
				)
				for item in self.manifest.by_survey[survey_id]
			}
		return self._posterior_arrays[survey_id]


def load_strat_multi_head_target_manifest(
	path: str | Path,
) -> StratMultiHeadTargetManifest:
	"""Load #268's references without materializing pseudo-target arrays."""
	payload = load_multi_head_target_manifest(
		path,
		validate_array_semantics=False,
	)
	head_ks = tuple(cast('list[int]', payload['head_ks']))
	common = cast('Mapping[str, object]', payload['common'])
	heads = cast('Mapping[str, Mapping[str, object]]', payload['heads'])
	by_survey: dict[str, tuple[StratMultiHeadTargetInput, ...]] = {}
	for survey_id in cast('list[str]', common['survey_ids']):
		inputs: list[StratMultiHeadTargetInput] = []
		for k in head_ks:
			entry = cast(
				'Mapping[str, object]',
				heads[str(k)]['surveys'][survey_id],
			)
			inputs.append(
				StratMultiHeadTargetInput(
					k=k,
					survey_id=survey_id,
					labels_path=Path(
						str(cast('Mapping[str, object]', entry['labels'])['path'])
					),
					confidence_path=Path(
						str(cast('Mapping[str, object]', entry['confidence'])['path'])
					),
					valid_tokens_path=Path(
						str(cast('Mapping[str, object]', entry['valid_tokens'])['path'])
					),
					metadata_path=Path(
						str(cast('Mapping[str, object]', entry['metadata'])['path'])
					),
					hashes={
						name: str(cast('Mapping[str, object]', entry[name])['sha256'])
						for name in ('labels', 'confidence', 'valid_tokens', 'metadata')
					},
				),
			)
		by_survey[survey_id] = tuple(inputs)
	return StratMultiHeadTargetManifest(
		head_ks=head_ks,
		by_survey=by_survey,
		common_valid_token_sha256={
			survey_id: str(
				cast('Mapping[str, object]', common['valid_tokens_sha256'])[survey_id]
			)
			for survey_id in by_survey
		},
	)


def load_strat_multi_head_lateral_target_manifest_adapter(
	path: str | Path,
) -> StratMultiHeadLateralTargetManifest:
	"""Adapt a strict M5-LS export to the existing hard-target provider.

	The strict lateral validator is deliberately invoked before references are
	adapted.  Arrays remain paths here and are memory-mapped only by the provider
	when a crop is requested.
	"""
	payload = load_multi_head_lateral_target_manifest(
		path,
		validate_array_semantics=False,
	)
	head_ks = tuple(cast('list[int]', payload['head_ks']))
	heads = cast('Mapping[str, Mapping[str, object]]', payload['heads'])
	by_survey: dict[str, tuple[StratMultiHeadTargetInput, ...]] = {}
	first_surveys = cast('Mapping[str, object]', heads[str(head_ks[0])]['surveys'])
	for survey_id in first_surveys:
		inputs: list[StratMultiHeadTargetInput] = []
		for k in head_ks:
			entry = cast('Mapping[str, object]', heads[str(k)]['surveys'][survey_id])
			inputs.append(
				StratMultiHeadTargetInput(
					k=k,
					survey_id=str(survey_id),
					labels_path=Path(
						str(cast('Mapping[str, object]', entry['labels'])['path'])
					),
					confidence_path=Path(
						str(cast('Mapping[str, object]', entry['confidence'])['path'])
					),
					valid_tokens_path=Path(
						str(cast('Mapping[str, object]', entry['valid_tokens'])['path'])
					),
					metadata_path=Path(
						str(cast('Mapping[str, object]', entry['metadata'])['path'])
					),
					hashes={
						name: str(cast('Mapping[str, object]', entry[name])['sha256'])
						for name in ('labels', 'confidence', 'valid_tokens', 'metadata')
					},
				),
			)
		by_survey[str(survey_id)] = tuple(inputs)
	target_manifest = StratMultiHeadTargetManifest(
		head_ks=head_ks,
		by_survey=by_survey,
		common_valid_token_sha256={
			survey_id: inputs[0].hashes['valid_tokens']
			for survey_id, inputs in by_survey.items()
		},
	)
	# Validate the typed view as well, including all reference digests.
	_coerce_multi_head_target_manifest(target_manifest)
	return StratMultiHeadLateralTargetManifest(
		target_manifest=target_manifest,
		target_semantics=str(payload['target_semantics']),
		source_hard_manifest=cast('Mapping[str, str]', payload['source_hard_manifest']),
		source_posterior_manifest=cast(
			'Mapping[str, str]', payload['source_posterior_manifest']
		),
		smoothing=cast('Mapping[str, object]', payload['smoothing']),
	)


def load_strat_multi_head_xy_neighbor_consensus_target_manifest_adapter(
	path: str | Path,
) -> StratMultiHeadXYNeighborConsensusTargetManifest:
	"""Adapt a strict XY-consensus export to the hard-target provider.

	The target loader validates the distinct source-hard-only provenance before
	the existing lazy hard-label provider sees any array references.
	"""
	payload = load_multi_head_xy_neighbor_consensus_target_manifest(
		path,
		validate_array_semantics=False,
	)
	head_ks = tuple(cast('list[int]', payload['head_ks']))
	heads = cast('Mapping[str, Mapping[str, object]]', payload['heads'])
	by_survey: dict[str, tuple[StratMultiHeadTargetInput, ...]] = {}
	first_surveys = cast('Mapping[str, object]', heads[str(head_ks[0])]['surveys'])
	for survey_id in first_surveys:
		inputs: list[StratMultiHeadTargetInput] = []
		for k in head_ks:
			entry = cast('Mapping[str, object]', heads[str(k)]['surveys'][survey_id])
			inputs.append(
				StratMultiHeadTargetInput(
					k=k,
					survey_id=str(survey_id),
					labels_path=Path(
						str(cast('Mapping[str, object]', entry['labels'])['path'])
					),
					confidence_path=Path(
						str(cast('Mapping[str, object]', entry['confidence'])['path'])
					),
					valid_tokens_path=Path(
						str(cast('Mapping[str, object]', entry['valid_tokens'])['path'])
					),
					metadata_path=Path(
						str(cast('Mapping[str, object]', entry['metadata'])['path'])
					),
					hashes={
						name: str(cast('Mapping[str, object]', entry[name])['sha256'])
						for name in (
							'labels',
							'confidence',
							'valid_tokens',
							'metadata',
						)
					},
				)
			)
		by_survey[str(survey_id)] = tuple(inputs)
	target_manifest = StratMultiHeadTargetManifest(
		head_ks=head_ks,
		by_survey=by_survey,
		common_valid_token_sha256={
			survey_id: inputs[0].hashes['valid_tokens']
			for survey_id, inputs in by_survey.items()
		},
		invalid_label_policy='preserve_source',
	)
	_coerce_multi_head_target_manifest(target_manifest)
	return StratMultiHeadXYNeighborConsensusTargetManifest(
		target_manifest=target_manifest,
		target_representation=str(payload['target_representation']),
		target_semantics=str(payload['target_semantics']),
		source_hard_manifest=cast('Mapping[str, str]', payload['source_hard_manifest']),
		smoothing=cast('Mapping[str, object]', payload['smoothing']),
	)


def load_strat_multi_head_xy_neighbor_unanimous_target_manifest_adapter(
	path: str | Path,
) -> StratMultiHeadXYNeighborUnanimousTargetManifest:
	"""Adapt a strict XY-unanimous export to the existing hard provider.

	The target loader validates the distinct source-hard-only provenance before
	the existing lazy hard-label provider sees any array references.
	"""
	payload = load_multi_head_xy_neighbor_unanimous_target_manifest(
		path,
		validate_array_semantics=False,
	)
	head_ks = tuple(cast('list[int]', payload['head_ks']))
	heads = cast('Mapping[str, Mapping[str, object]]', payload['heads'])
	by_survey: dict[str, tuple[StratMultiHeadTargetInput, ...]] = {}
	first_surveys = cast('Mapping[str, object]', heads[str(head_ks[0])]['surveys'])
	for survey_id in first_surveys:
		inputs: list[StratMultiHeadTargetInput] = []
		for k in head_ks:
			entry = cast('Mapping[str, object]', heads[str(k)]['surveys'][survey_id])
			inputs.append(
				StratMultiHeadTargetInput(
					k=k,
					survey_id=str(survey_id),
					labels_path=Path(
						str(cast('Mapping[str, object]', entry['labels'])['path'])
					),
					confidence_path=Path(
						str(cast('Mapping[str, object]', entry['confidence'])['path'])
					),
					valid_tokens_path=Path(
						str(cast('Mapping[str, object]', entry['valid_tokens'])['path'])
					),
					metadata_path=Path(
						str(cast('Mapping[str, object]', entry['metadata'])['path'])
					),
					hashes={
						name: str(cast('Mapping[str, object]', entry[name])['sha256'])
						for name in (
							'labels',
							'confidence',
							'valid_tokens',
							'metadata',
						)
					},
				)
			)
		by_survey[str(survey_id)] = tuple(inputs)
	target_manifest = StratMultiHeadTargetManifest(
		head_ks=head_ks,
		by_survey=by_survey,
		common_valid_token_sha256={
			survey_id: inputs[0].hashes['valid_tokens']
			for survey_id, inputs in by_survey.items()
		},
		invalid_label_policy='preserve_source',
	)
	_coerce_multi_head_target_manifest(target_manifest)
	return StratMultiHeadXYNeighborUnanimousTargetManifest(
		target_manifest=target_manifest,
		target_representation=str(payload['target_representation']),
		target_semantics=str(payload['target_semantics']),
		source_hard_manifest=cast('Mapping[str, str]', payload['source_hard_manifest']),
		smoothing=cast('Mapping[str, object]', payload['smoothing']),
	)


def load_strat_multi_head_posterior_manifest(
	path: str | Path,
) -> StratMultiHeadPosteriorManifest:
	"""Load the strict state-posterior manifest without retaining its payload."""
	payload = load_multi_head_state_posterior_manifest(
		path,
		validate_array_semantics=False,
	)
	head_ks = tuple(cast('list[int]', payload['head_ks']))
	heads = cast('Mapping[str, Mapping[str, object]]', payload['heads'])
	first_surveys = cast('Mapping[str, object]', heads[str(head_ks[0])]['surveys'])
	by_survey: dict[str, tuple[StratMultiHeadPosteriorInput, ...]] = {}
	for survey_id in first_surveys:
		inputs: list[StratMultiHeadPosteriorInput] = []
		for k in head_ks:
			entry = cast('Mapping[str, object]', heads[str(k)]['surveys'][survey_id])
			inputs.append(
				StratMultiHeadPosteriorInput(
					k=k,
					survey_id=survey_id,
					posterior_path=Path(
						str(cast('Mapping[str, object]', entry['posterior'])['path'])
					),
					valid_tokens_path=Path(
						str(cast('Mapping[str, object]', entry['valid_tokens'])['path'])
					),
					metadata_path=Path(
						str(cast('Mapping[str, object]', entry['metadata'])['path'])
					),
					hashes={
						name: str(cast('Mapping[str, object]', entry[name])['sha256'])
						for name in ('posterior', 'valid_tokens', 'metadata')
					},
				),
			)
		by_survey[survey_id] = tuple(inputs)
	return StratMultiHeadPosteriorManifest(head_ks=head_ks, by_survey=by_survey)


def _coerce_multi_head_target_manifest(
	value: StratMultiHeadTargetManifest | str | Path,
) -> StratMultiHeadTargetManifest:
	if isinstance(value, (str, Path)):
		return load_strat_multi_head_target_manifest(value)
	if not isinstance(value, StratMultiHeadTargetManifest):
		msg = (
			'multi_head_target_manifest must be a '
			'StratMultiHeadTargetManifest or path; got '
			f'{type(value).__name__}'
		)
		raise TypeError(msg)
	_validate_multi_head_target_manifest_header(value)
	for survey_id, inputs in value.by_survey.items():
		_validate_multi_head_target_survey(value, survey_id, inputs)
	return value


def _coerce_multi_head_posterior_manifest(
	value: StratMultiHeadPosteriorManifest | str | Path,
) -> StratMultiHeadPosteriorManifest:
	if isinstance(value, (str, Path)):
		return load_strat_multi_head_posterior_manifest(value)
	if not isinstance(value, StratMultiHeadPosteriorManifest):
		raise TypeError(
			'multi_head_posterior_manifest must be a '
			'StratMultiHeadPosteriorManifest or path; got '
			f'{type(value).__name__}',
		)
	if value.head_ks != (6, 8, 10):
		raise ValueError('multi-head posterior head_ks must be canonical (6, 8, 10)')
	if not isinstance(value.by_survey, Mapping) or not value.by_survey:
		raise ValueError('multi-head posterior manifest must contain survey inputs')
	for survey_id, inputs in value.by_survey.items():
		if not isinstance(survey_id, str) or not survey_id:
			raise TypeError('multi-head posterior survey ids must be non-empty strings')
		if (
			not isinstance(inputs, tuple)
			or tuple(item.k for item in inputs) != value.head_ks
		):
			raise ValueError(
				f'multi-head posterior inputs for {survey_id!r} must match head_ks',
			)
		for item in inputs:
			_validate_multi_head_posterior_input(item, survey_id)
	return value


def _validate_multi_head_posterior_input(
	item: object,
	survey_id: str,
) -> None:
	if not isinstance(item, StratMultiHeadPosteriorInput):
		raise TypeError(
			'multi-head posterior inputs must contain '
			'StratMultiHeadPosteriorInput items',
		)
	if item.survey_id != survey_id:
		raise ValueError(
			f'multi-head posterior input survey ids must match {survey_id!r}',
		)
	if set(item.hashes) != {'posterior', 'valid_tokens', 'metadata'}:
		raise ValueError('multi-head posterior input hashes must be complete')
	for name, path in (
		('posterior', item.posterior_path),
		('valid_tokens', item.valid_tokens_path),
		('metadata', item.metadata_path),
	):
		if not isinstance(path, Path):
			raise TypeError(
				'multi-head posterior artifact paths must be Path instances',
			)
		if not path.is_file() or not isinstance(item.hashes[name], str):
			raise ValueError(f'multi-head posterior {name} reference is invalid')
		if file_sha256(path) != item.hashes[name]:
			raise ValueError(f'multi-head posterior {name} hash mismatch')
	posterior = np.load(item.posterior_path, mmap_mode='r', allow_pickle=False)
	valid = np.load(item.valid_tokens_path, mmap_mode='r', allow_pickle=False)
	_validate_multi_head_posterior_crop(posterior, valid, k=item.k)


def _validate_multi_head_target_manifest_header(
	value: StratMultiHeadTargetManifest,
) -> None:
	if (
		len(value.head_ks) < 2
		or any(isinstance(k, bool) or not isinstance(k, int) for k in value.head_ks)
		or any(k < 2 for k in value.head_ks)
		or tuple(sorted(value.head_ks)) != value.head_ks
		or len(set(value.head_ks)) != len(value.head_ks)
	):
		raise ValueError(
			'multi-head target head_ks must be unique ascending integers >= 2 '
			'with at least two heads'
		)
	if not isinstance(value.by_survey, Mapping):
		raise TypeError('multi-head target by_survey must be a mapping')
	if not isinstance(value.common_valid_token_sha256, Mapping):
		raise TypeError('multi-head target common valid-token hashes must be a mapping')
	if value.invalid_label_policy not in {'minus_one', 'preserve_source'}:
		raise ValueError(
			'multi-head target invalid_label_policy must be minus_one or '
			'preserve_source'
		)
	if not value.by_survey:
		raise ValueError('multi-head target manifest must contain at least one survey')
	if set(value.common_valid_token_sha256) != set(value.by_survey):
		raise ValueError(
			'multi-head target common valid-token hashes must match survey inputs'
		)


def _validate_multi_head_target_survey(
	manifest: StratMultiHeadTargetManifest,
	survey_id: object,
	inputs: object,
) -> None:
	if not isinstance(survey_id, str) or not survey_id:
		raise TypeError('multi-head target survey ids must be non-empty strings')
	if not isinstance(inputs, tuple):
		raise TypeError(f'multi-head target inputs for {survey_id!r} must be a tuple')
	if not all(isinstance(item, StratMultiHeadTargetInput) for item in inputs):
		invalid = next(
			item for item in inputs if not isinstance(item, StratMultiHeadTargetInput)
		)
		raise TypeError(
			'multi-head target inputs must contain StratMultiHeadTargetInput items; '
			f'got {type(invalid).__name__}'
		)
	if tuple(item.k for item in inputs) != manifest.head_ks:
		raise ValueError(
			f'multi-head target inputs for {survey_id!r} must match head_ks'
		)
	common_valid_hash = manifest.common_valid_token_sha256[survey_id]
	if not isinstance(common_valid_hash, str):
		raise TypeError(
			f'multi-head target valid-token hash for {survey_id!r} must be a string'
		)
	for item in inputs:
		_validate_multi_head_target_input(item, survey_id, common_valid_hash)


def _validate_multi_head_target_input(
	item: StratMultiHeadTargetInput,
	survey_id: str,
	common_valid_hash: str,
) -> None:
	if item.survey_id != survey_id:
		raise ValueError(f'multi-head target input survey ids must match {survey_id!r}')
	if not all(
		isinstance(path, Path)
		for path in (
			item.labels_path,
			item.confidence_path,
			item.valid_tokens_path,
			item.metadata_path,
		)
	):
		raise TypeError('multi-head target artifact paths must be Path instances')
	validate_multi_head_target_reference(
		k=item.k,
		survey_id=survey_id,
		labels_path=item.labels_path,
		confidence_path=item.confidence_path,
		valid_tokens_path=item.valid_tokens_path,
		metadata_path=item.metadata_path,
		hashes=item.hashes,
		validate_array_semantics=False,
	)
	if item.hashes['valid_tokens'] != common_valid_hash:
		raise ValueError(
			f'multi-head target valid-token hash for {survey_id!r} does not '
			'match the common contract'
		)


def _token_slices(context: TargetProviderContext) -> tuple[slice, ...]:
	token_stop = tuple(
		start_axis + size_axis
		for start_axis, size_axis in zip(
			context.token_start_xyz,
			context.token_size_xyz,
			strict=True,
		)
	)
	return tuple(
		slice(start_axis, stop_axis)
		for start_axis, stop_axis in zip(
			context.token_start_xyz,
			token_stop,
			strict=True,
		)
	)


def _require_multi_head_targets(
	sample: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
	value = sample.get('strat_multi_targets')
	if not isinstance(value, Mapping):
		msg = 'strat_multi_targets must be a mapping'
		raise TypeError(msg)
	if not all(isinstance(target, Mapping) for target in value.values()):
		msg = 'strat_multi_targets entries must be mappings'
		raise TypeError(msg)
	return cast('Mapping[str, Mapping[str, object]]', value)


def _require_multi_head_posteriors(
	sample: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
	value = sample.get('strat_multi_posteriors')
	if not isinstance(value, Mapping):
		raise TypeError('strat_multi_posteriors must be a mapping')
	if not all(isinstance(target, Mapping) for target in value.values()):
		raise TypeError('strat_multi_posteriors entries must be mappings')
	return cast('Mapping[str, Mapping[str, object]]', value)


def _require_int64_array(sample: Mapping[str, object], key: str) -> np.ndarray:
	value = sample[key]
	if not isinstance(value, np.ndarray):
		msg = f'{key} must be a NumPy array; got {type(value).__name__}'
		raise TypeError(msg)
	if value.dtype != np.int64:
		msg = f'{key} dtype must be int64; got {value.dtype}'
		raise TypeError(msg)
	return value


def _validate_multi_head_crop(
	labels: np.ndarray,
	confidence: np.ndarray,
	valid_mask: np.ndarray,
	*,
	k: int,
	allow_source_invalid_labels: bool = False,
) -> None:
	if labels.shape != confidence.shape or labels.shape != valid_mask.shape:
		msg = f'multi-head k={k} labels, confidence, and valid_tokens shapes must match'
		raise ValueError(msg)
	if labels.dtype.kind not in 'iu':
		raise TypeError(f'multi-head k={k} labels must have integer dtype')
	if confidence.dtype != np.float32:
		raise TypeError(f'multi-head k={k} confidence dtype must be float32')
	if valid_mask.dtype != np.bool_:
		raise TypeError(f'multi-head k={k} valid_tokens dtype must be bool')
	if not np.all(np.isfinite(confidence)) or np.any(
		(confidence < 0.0) | (confidence > 1.0),
	):
		raise ValueError(f'multi-head k={k} confidence must be finite and in [0, 1]')
	if np.any((labels[valid_mask] < 0) | (labels[valid_mask] >= k)):
		raise ValueError(f'multi-head k={k} valid labels must be in [0, {k - 1}]')
	if not allow_source_invalid_labels and np.any(labels[~valid_mask] != -1):
		raise ValueError(f'multi-head k={k} invalid labels must be -1')
	if np.any(confidence[~valid_mask] != 0.0):
		raise ValueError(f'multi-head k={k} invalid confidence must be 0.0')


def _validate_multi_head_sample(
	labels: np.ndarray,
	confidence: np.ndarray,
	boundary_weight: np.ndarray,
	valid_mask: np.ndarray,
	*,
	k: int,
) -> None:
	_validate_multi_head_crop(labels, confidence, valid_mask, k=k)
	if boundary_weight.dtype != np.float32:
		raise TypeError(f'multi-head k={k} boundary_weight dtype must be float32')
	if boundary_weight.shape != valid_mask.shape:
		raise ValueError(
			f'multi-head k={k} boundary_weight shape must match valid_mask'
		)
	if not np.all(np.isfinite(boundary_weight)) or np.any(
		(boundary_weight < 0.0) | (boundary_weight > 1.0),
	):
		raise ValueError(
			f'multi-head k={k} boundary_weight must be finite and in [0, 1]'
		)
	if np.any(boundary_weight[~valid_mask] != 0.0):
		raise ValueError(f'multi-head k={k} invalid boundary_weight must be 0.0')


def _validate_multi_head_posterior_crop(
	posterior: np.ndarray,
	valid_mask: np.ndarray,
	*,
	k: int,
) -> None:
	"""Validate one cropped posterior with an explicit invalid-row contract."""
	if posterior.dtype != np.float32 or posterior.ndim != 4:
		raise ValueError(f'multi-head posterior k={k} must be float32 [X,Y,Z,{k}]')
	if posterior.shape[-1] != k:
		raise ValueError(
			f'multi-head posterior k={k} last dimension must equal {k}',
		)
	if valid_mask.dtype != np.bool_ or valid_mask.shape != posterior.shape[:3]:
		raise ValueError(
			f'multi-head posterior k={k} valid mask shape/dtype mismatch',
		)
	if not np.all(np.isfinite(posterior)) or np.any(posterior < 0.0):
		raise ValueError(
			f'multi-head posterior k={k} must be finite and non-negative',
		)
	if not np.allclose(posterior[valid_mask].sum(axis=-1), 1.0, rtol=0.0, atol=2.0e-6):
		raise ValueError(
			f'multi-head posterior k={k} valid rows must sum to one',
		)
	if np.any(posterior[~valid_mask] != 0.0):
		raise ValueError(
			f'multi-head posterior k={k} invalid rows must be zero',
		)


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
	'MultiHeadStratPosteriorProvider',
	'MultiHeadStratPseudoTargetProvider',
	'NoTargetProvider',
	'StratMultiHeadLateralTargetManifest',
	'StratMultiHeadPosteriorArrays',
	'StratMultiHeadPosteriorInput',
	'StratMultiHeadPosteriorManifest',
	'StratMultiHeadTargetArrays',
	'StratMultiHeadTargetInput',
	'StratMultiHeadTargetManifest',
	'StratMultiHeadXYNeighborConsensusTargetManifest',
	'StratMultiHeadXYNeighborUnanimousTargetManifest',
	'StratPseudoTargetProvider',
	'TargetProvider',
	'TargetProviderContext',
	'load_strat_multi_head_lateral_target_manifest_adapter',
	'load_strat_multi_head_posterior_manifest',
	'load_strat_multi_head_target_manifest',
	'load_strat_multi_head_xy_neighbor_consensus_target_manifest_adapter',
	'load_strat_multi_head_xy_neighbor_unanimous_target_manifest_adapter',
]
