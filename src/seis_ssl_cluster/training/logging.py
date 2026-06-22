"""Small stdout logging helpers for training loops."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Mapping


def print_epoch_metrics(
	epoch: int,
	metrics: Mapping[str, float],
) -> None:
	"""Print one compact epoch metrics line."""
	parts = [f'epoch={epoch}']
	parts.extend(f'{key}={metrics[key]:.6g}' for key in sorted(metrics))
	print(' '.join(parts))


__all__ = ['print_epoch_metrics']
