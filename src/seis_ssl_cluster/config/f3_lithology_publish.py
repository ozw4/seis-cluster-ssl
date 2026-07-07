"""F3 lithology publish config validation."""

from __future__ import annotations

from collections.abc import Mapping

from seis_ssl_cluster.config.f3_lithology_common import (
	_max_file_size_bytes,
	_optional_non_negative_int,
	_optional_path,
	_publish_optional_bool,
)
from seis_ssl_cluster.f3 import F3LithologyPublishConfig


def f3_lithology_publish_config_from_mapping(
	value: object,
) -> F3LithologyPublishConfig:
	"""Validate and normalize the optional F3 lithology publish config."""
	if value is None:
		return F3LithologyPublishConfig()
	if not isinstance(value, Mapping):
		msg = f'publish must be a mapping; got {value!r}'
		raise TypeError(msg)
	enabled = _publish_optional_bool(value, 'enabled', default=False)
	include_figures = _publish_optional_bool(value, 'include_figures', default=True)
	output_dir = _optional_path(value, 'output_dir')
	if enabled and output_dir is None:
		msg = 'publish.output_dir must be set when publish.enabled is true'
		raise ValueError(msg)
	return F3LithologyPublishConfig(
		enabled=enabled,
		output_dir=output_dir,
		include_figures=include_figures,
		max_file_size_bytes=_max_file_size_bytes(value),
		max_prediction_figures=_optional_non_negative_int(
			value,
			'max_prediction_figures',
			default=3,
		),
	)


__all__ = ['f3_lithology_publish_config_from_mapping']
