"""Shared method identity helpers for Barlow Twins configurations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from seis_ssl_cluster.config.schema import (
	BARLOW_TWINS_PRETRAINING_METHOD,
	LOCAL_BARLOW_TWINS_PRETRAINING_METHOD,
)

_SUPPORTED_PRETRAINING_METHODS = frozenset(
	{
		BARLOW_TWINS_PRETRAINING_METHOD,
		LOCAL_BARLOW_TWINS_PRETRAINING_METHOD,
	}
)


def resolve_barlow_twins_pretraining_method(
	config: Mapping[str, object],
) -> str:
	"""Resolve the method identity from a Barlow Twins stage config.

	Legacy resolved configurations omit ``barlow_twins.method`` and therefore
	resolve to the standard global Barlow Twins objective.
	"""
	barlow_twins = config.get('barlow_twins')
	if not isinstance(barlow_twins, Mapping):
		raise TypeError('config.barlow_twins must be a mapping')
	method = barlow_twins.get(
		'method',
		BARLOW_TWINS_PRETRAINING_METHOD,
	)
	if not isinstance(method, str):
		raise TypeError('config.barlow_twins.method must be a string')
	if method not in _SUPPORTED_PRETRAINING_METHODS:
		raise ValueError(
			'config.barlow_twins.method must be one of '
			f'{sorted(_SUPPORTED_PRETRAINING_METHODS)!r}; got {method!r}'
		)
	return cast('str', method)


def barlow_twins_config_compatibility_identity(
	config: Mapping[str, object],
) -> dict[str, object]:
	"""Return a method-aware identity for config compatibility checks.

	An explicitly declared standard method is equivalent to the legacy omitted
	method. Local method fields remain part of the strict identity.
	"""
	barlow_twins = config.get('barlow_twins')
	if not isinstance(barlow_twins, Mapping):
		raise TypeError('config.barlow_twins must be a mapping')
	identity = {str(key): value for key, value in barlow_twins.items()}
	if (
		resolve_barlow_twins_pretraining_method(config)
		== BARLOW_TWINS_PRETRAINING_METHOD
	):
		identity.pop('method', None)
	return identity


__all__ = [
	'barlow_twins_config_compatibility_identity',
	'resolve_barlow_twins_pretraining_method',
]
