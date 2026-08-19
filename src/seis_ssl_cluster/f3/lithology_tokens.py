"""Compatibility wrapper for F3 lithology token dataset builders."""

from seis_ssl_cluster.f3.lithology import tokens as _tokens
from seis_ssl_cluster.f3.lithology.tokens import *  # noqa: F403

__all__ = _tokens.__all__
