"""Shared method identity helpers for VICReg configurations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from seis_ssl_cluster.config.schema import (
	SUPPORTED_VICREG_PRETRAINING_METHODS,
)


def resolve_vicreg_pretraining_method(
	config: Mapping[str, object],
) -> str:
	"""Resolve and validate the local VICReg method identity."""
	vicreg = config.get('vicreg')
	if not isinstance(vicreg, Mapping):
		raise TypeError('config.vicreg must be a mapping')
	method = vicreg.get('method')
	if not isinstance(method, str):
		raise TypeError('config.vicreg.method must be a string')
	if method not in SUPPORTED_VICREG_PRETRAINING_METHODS:
		raise ValueError(
			'config.vicreg.method must be one of '
			f'{sorted(SUPPORTED_VICREG_PRETRAINING_METHODS)!r}; got {method!r}'
		)
	return cast('str', method)


def vicreg_config_compatibility_identity(
	config: Mapping[str, object],
) -> dict[str, object]:
	"""Return the strict method-aware VICReg objective identity."""
	resolve_vicreg_pretraining_method(config)
	vicreg = config.get('vicreg')
	if not isinstance(vicreg, Mapping):
		raise TypeError('config.vicreg must be a mapping')
	return {str(key): value for key, value in vicreg.items()}


__all__ = [
	'resolve_vicreg_pretraining_method',
	'vicreg_config_compatibility_identity',
]
