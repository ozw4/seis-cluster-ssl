"""Grouped host reads for stratigraphic HMM training diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping


def all_tensors_finite(values: Iterable[torch.Tensor]) -> bool:
	"""Check every tensor, reading one combined flag per device on the host."""
	flags: dict[torch.device, list[torch.Tensor]] = {}
	for value in values:
		flags.setdefault(value.device, []).append(torch.isfinite(value.detach()).all())
	return all(bool(torch.stack(group).all()) for group in flags.values())


def materialize_metrics(metrics: Mapping[str, torch.Tensor]) -> dict[str, float]:
	"""Read scalar metrics together without changing their precision or order.

	Group by device and dtype so stacking cannot round integers or promote mixed
	precision metrics before the original scalar-to-Python-float conversion.
	Epoch accumulation remains on the host in batch order.
	"""
	groups: dict[tuple[torch.device, torch.dtype], list[str]] = {}
	for key, value in metrics.items():
		groups.setdefault((value.device, value.dtype), []).append(key)
	result: dict[str, float] = {}
	for keys in groups.values():
		values = torch.stack([metrics[key].detach().reshape(()) for key in keys])
		result.update(zip(keys, map(float, values.cpu().tolist()), strict=True))
	return {key: result[key] for key in metrics}
