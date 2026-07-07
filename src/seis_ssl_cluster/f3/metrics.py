"""Compatibility wrapper for F3 lithology metrics."""

from __future__ import annotations

from seis_ssl_cluster.f3.lithology import metrics as _metrics
from seis_ssl_cluster.f3.lithology.metrics import *  # noqa: F403


def __getattr__(name: str) -> object:
	return getattr(_metrics, name)
