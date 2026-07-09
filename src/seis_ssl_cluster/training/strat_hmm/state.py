"""State containers for stratigraphic HMM pretext training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Mapping

	import torch

	from seis_ssl_cluster.models.mae import AmplitudeMAE3D
	from seis_ssl_cluster.stratigraphy import OrderedPrototypeHead


@dataclass(frozen=True)
class StratHmmTrainingState:
	"""Summary state returned from one strat HMM training epoch."""

	epoch: int
	global_step: int
	metrics: dict[str, float]
	last_batch_index: int
	completed_epoch: bool


@dataclass(frozen=True)
class StratHmmResumeState:
	"""Resolved checkpoint resume location."""

	start_epoch: int
	global_step: int
	skip_batches: int


@dataclass(frozen=True)
class TrainabilitySummary:
	"""Summary of student MAE trainability after milestone-1 configuration."""

	trainable_parameter_count: int
	frozen_parameter_count: int
	trainable_names: tuple[str, ...]


@dataclass(frozen=True)
class StratHmmHeadOnlyComponents:
	"""Trainable components for strat HMM pretext training."""

	student: AmplitudeMAE3D
	teacher: AmplitudeMAE3D | None
	head: OrderedPrototypeHead
	optimizer: torch.optim.Optimizer
	mae_checkpoint_config: Mapping[str, object]
	trainability_summary: TrainabilitySummary


__all__ = [
	'StratHmmHeadOnlyComponents',
	'StratHmmResumeState',
	'StratHmmTrainingState',
	'TrainabilitySummary',
]
