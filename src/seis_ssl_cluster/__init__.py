"""Self-contained seismic SSL clustering package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from seis_ssl_cluster import _version

try:
	__version__ = version('seis-cluster-ssl')
except PackageNotFoundError:
	__version__ = _version.__version__

__all__ = ['__version__']
