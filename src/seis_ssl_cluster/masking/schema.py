"""Schema for amplitude-only spatial masking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import numpy as np


@dataclass(frozen=True)
class SpatialMaskingPlan:
	"""Spatial token masks for one amplitude sample."""

	spatial_mask: np.ndarray
	visible_spatial_mask: np.ndarray


__all__ = ['SpatialMaskingPlan']
