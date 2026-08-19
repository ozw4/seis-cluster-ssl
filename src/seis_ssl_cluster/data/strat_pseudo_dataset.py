"""Compatibility dataset for stratigraphic pseudo-target crops."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.data.amplitude_crop_dataset import NopimsAmplitudeCropDataset
from seis_ssl_cluster.data.target_providers import (
	MultiHeadStratPosteriorProvider,
	MultiHeadStratPseudoTargetProvider,
	StratMultiHeadPosteriorManifest,
	StratMultiHeadTargetManifest,
	StratPseudoTargetProvider,
)
from seis_ssl_cluster.data.zero_mask import (
	DEFAULT_ZERO_MASK_CONFIG,
	ZeroMaskConfig,
)

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence
	from pathlib import Path

	from seis_ssl_cluster.data.normalization import AmplitudeAgcConfig
	from seis_ssl_cluster.data.schema import SurveyManifest
	from seis_ssl_cluster.data.window_preprocessing import FiniteCheckMode
	from seis_ssl_cluster.stratigraphy.targets import StratPseudoTargetInput


class NopimsStratPseudoTargetDataset:
	"""Preserve the strat dataset API using the generic crop dataset."""

	def __init__(  # noqa: D107, PLR0913, PLR0917
		self,
		manifests: Sequence[SurveyManifest],
		pseudo_target_inputs: Sequence[StratPseudoTargetInput],
		local_crop_size_xyz: Sequence[int] = (128, 128, 128),
		patch_size_xyz: Sequence[int] = (8, 8, 8),
		seed: int = 42,
		samples_per_epoch: int | None = None,
		zero_mask: ZeroMaskConfig = DEFAULT_ZERO_MASK_CONFIG,
		min_valid_fraction: float = 0.0,
		max_resample_attempts: int = 16,
		normalized_clip_abs: float | None = None,
		amplitude_agc: AmplitudeAgcConfig | Mapping[str, object] | None = None,
		finite_check_mode: FiniteCheckMode = 'strict',
		min_confidence: float = 0.0,
	) -> None:
		provider = StratPseudoTargetProvider(
			pseudo_target_inputs,
			min_confidence=min_confidence,
		)
		self._dataset = NopimsAmplitudeCropDataset(
			manifests,
			local_crop_size_xyz=local_crop_size_xyz,
			patch_size_xyz=patch_size_xyz,
			seed=seed,
			samples_per_epoch=samples_per_epoch,
			zero_mask=zero_mask,
			min_valid_fraction=min_valid_fraction,
			max_resample_attempts=max_resample_attempts,
			normalized_clip_abs=normalized_clip_abs,
			amplitude_agc=amplitude_agc,
			finite_check_mode=finite_check_mode,
			target_provider=provider,
		)

	def __len__(self) -> int:
		"""Return the configured epoch length."""
		return len(self._dataset)

	@property
	def epoch(self) -> int:
		"""Return the current shared sampling epoch."""
		return self._dataset.epoch

	def set_epoch(self, epoch: int) -> None:
		"""Set the sampling epoch used for deterministic sample draws."""
		self._dataset.set_epoch(epoch)

	def __getitem__(self, index: int) -> dict[str, object]:
		"""Return one token-aligned pseudo-target supervision sample."""
		return self._dataset[index]


class NopimsStratMultiHeadTargetDataset:
	"""Token-aligned crops with all heads from one validated target manifest."""

	def __init__(  # noqa: D107, PLR0913, PLR0917
		self,
		manifests: Sequence[SurveyManifest],
		multi_head_target_manifest: StratMultiHeadTargetManifest | str | Path,
		local_crop_size_xyz: Sequence[int] = (128, 128, 128),
		patch_size_xyz: Sequence[int] = (8, 8, 8),
		seed: int = 42,
		samples_per_epoch: int | None = None,
		zero_mask: ZeroMaskConfig = DEFAULT_ZERO_MASK_CONFIG,
		min_valid_fraction: float = 0.0,
		max_resample_attempts: int = 16,
		normalized_clip_abs: float | None = None,
		amplitude_agc: AmplitudeAgcConfig | Mapping[str, object] | None = None,
		finite_check_mode: FiniteCheckMode = 'strict',
		min_confidence: float = 0.0,
	) -> None:
		provider = MultiHeadStratPseudoTargetProvider(
			multi_head_target_manifest,
			min_confidence=min_confidence,
		)
		self._dataset = NopimsAmplitudeCropDataset(
			manifests,
			local_crop_size_xyz=local_crop_size_xyz,
			patch_size_xyz=patch_size_xyz,
			seed=seed,
			samples_per_epoch=samples_per_epoch,
			zero_mask=zero_mask,
			min_valid_fraction=min_valid_fraction,
			max_resample_attempts=max_resample_attempts,
			normalized_clip_abs=normalized_clip_abs,
			amplitude_agc=amplitude_agc,
			finite_check_mode=finite_check_mode,
			target_provider=provider,
		)

	def __len__(self) -> int:
		"""Return the configured epoch length."""
		return len(self._dataset)

	@property
	def epoch(self) -> int:
		"""Return the current shared sampling epoch."""
		return self._dataset.epoch

	def set_epoch(self, epoch: int) -> None:
		"""Set the sampling epoch used for deterministic sample draws."""
		self._dataset.set_epoch(epoch)

	def __getitem__(self, index: int) -> dict[str, object]:
		"""Return one multi-head pseudo-target supervision sample."""
		return self._dataset[index]


class NopimsStratMultiHeadPosteriorDataset:
	"""Token-aligned crops with soft posteriors from a frozen HMM export."""

	def __init__(  # noqa: D107, PLR0913, PLR0917
		self,
		manifests: Sequence[SurveyManifest],
		multi_head_posterior_manifest: StratMultiHeadPosteriorManifest | str | Path,
		local_crop_size_xyz: Sequence[int] = (128, 128, 128),
		patch_size_xyz: Sequence[int] = (8, 8, 8),
		seed: int = 42,
		samples_per_epoch: int | None = None,
		zero_mask: ZeroMaskConfig = DEFAULT_ZERO_MASK_CONFIG,
		min_valid_fraction: float = 0.0,
		max_resample_attempts: int = 16,
		normalized_clip_abs: float | None = None,
		amplitude_agc: AmplitudeAgcConfig | Mapping[str, object] | None = None,
		finite_check_mode: FiniteCheckMode = 'strict',
	) -> None:
		provider = MultiHeadStratPosteriorProvider(multi_head_posterior_manifest)
		self._dataset = NopimsAmplitudeCropDataset(
			manifests,
			local_crop_size_xyz=local_crop_size_xyz,
			patch_size_xyz=patch_size_xyz,
			seed=seed,
			samples_per_epoch=samples_per_epoch,
			zero_mask=zero_mask,
			min_valid_fraction=min_valid_fraction,
			max_resample_attempts=max_resample_attempts,
			normalized_clip_abs=normalized_clip_abs,
			amplitude_agc=amplitude_agc,
			finite_check_mode=finite_check_mode,
			target_provider=provider,
		)

	def __len__(self) -> int:
		"""Return the configured epoch length."""
		return len(self._dataset)

	@property
	def epoch(self) -> int:
		"""Return the current shared sampling epoch."""
		return self._dataset.epoch

	def set_epoch(self, epoch: int) -> None:
		"""Set the sampling epoch used for deterministic sample draws."""
		self._dataset.set_epoch(epoch)

	def __getitem__(self, index: int) -> dict[str, object]:
		"""Return one multi-head soft-posterior supervision sample."""
		return self._dataset[index]


__all__ = [
	'NopimsStratMultiHeadPosteriorDataset',
	'NopimsStratMultiHeadTargetDataset',
	'NopimsStratPseudoTargetDataset',
]
