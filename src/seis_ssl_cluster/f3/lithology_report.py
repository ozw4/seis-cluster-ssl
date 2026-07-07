"""Compatibility wrapper for F3 lithology report helpers.

The implementation lives under :mod:`seis_ssl_cluster.f3.lithology.report`.
This module remains to preserve existing public imports.
"""

from __future__ import annotations

from seis_ssl_cluster.f3.lithology import report as _report
from seis_ssl_cluster.f3.lithology.report import *  # noqa: F403


def __getattr__(name: str) -> object:
	return getattr(_report, name)
