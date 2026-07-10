"""Contracts for adding supervision targets to amplitude samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
	from collections.abc import Mapping, MutableMapping, Sequence

	import numpy as np

	from seis_ssl_cluster.data.schema import CropRequest, SurveyManifest


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


__all__ = ['NoTargetProvider', 'TargetProvider', 'TargetProviderContext']
