"""Validation helpers for amplitude-only spatial masking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from seis_ssl_cluster.masking.schema import SpatialMaskingPlan


def validate_spatial_masking_plan(plan: SpatialMaskingPlan) -> None:
	"""Validate spatial mask shape, dtype, and visibility convention."""
	_validate_bool_mask('spatial_mask', plan.spatial_mask)
	_validate_bool_mask('visible_spatial_mask', plan.visible_spatial_mask)

	if plan.visible_spatial_mask.shape != plan.spatial_mask.shape:
		msg = (
			'visible_spatial_mask shape must equal spatial_mask shape; '
			f'got {plan.visible_spatial_mask.shape!r} and {plan.spatial_mask.shape!r}'
		)
		raise ValueError(msg)
	if not np.array_equal(plan.visible_spatial_mask, np.logical_not(plan.spatial_mask)):
		msg = 'visible_spatial_mask must equal ~spatial_mask'
		raise ValueError(msg)
	if not np.any(plan.spatial_mask):
		msg = 'spatial_mask must contain at least one masked token'
		raise ValueError(msg)
	if not np.any(plan.visible_spatial_mask):
		msg = 'visible_spatial_mask must contain at least one visible token'
		raise ValueError(msg)


def _validate_bool_mask(field_name: str, value: object) -> None:
	if not isinstance(value, np.ndarray):
		msg = f'{field_name} must be a NumPy array; got {type(value).__name__}'
		raise TypeError(msg)
	if value.dtype != np.bool_:
		msg = f'{field_name} dtype must be bool; got {value.dtype}'
		raise TypeError(msg)
	if value.ndim != 3:
		msg = f'{field_name} must be 3D; got {value.ndim}D'
		raise ValueError(msg)


__all__ = ['validate_spatial_masking_plan']
