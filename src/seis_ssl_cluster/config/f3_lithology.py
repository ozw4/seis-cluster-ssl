"""F3 lithology config validation entrypoints."""

from importlib import import_module
from typing import TYPE_CHECKING

from seis_ssl_cluster.f3.prepare_volume import f3_prepare_volume_config_from_mapping

if TYPE_CHECKING:
	from proc.seis_ssl_cluster.build_f3_lithology_report import (
		f3_lithology_publish_config_from_mapping,
		f3_lithology_report_config_from_mapping,
	)
	from proc.seis_ssl_cluster.build_f3_lithology_token_dataset import (
		f3_lithology_token_dataset_config_from_mapping,
	)
	from proc.seis_ssl_cluster.predict_f3_lithology_tokens import (
		f3_lithology_prediction_config_from_mapping,
	)
	from proc.seis_ssl_cluster.train_f3_lithology_probe import (
		f3_lithology_probe_config_from_mapping,
	)
	from proc.seis_ssl_cluster.visualize_f3_lithology_predictions import (
		f3_lithology_visualization_config_from_mapping,
	)

_LAZY_EXPORTS = {
	'f3_lithology_prediction_config_from_mapping': (
		'proc.seis_ssl_cluster.predict_f3_lithology_tokens'
	),
	'f3_lithology_probe_config_from_mapping': (
		'proc.seis_ssl_cluster.train_f3_lithology_probe'
	),
	'f3_lithology_publish_config_from_mapping': (
		'proc.seis_ssl_cluster.build_f3_lithology_report'
	),
	'f3_lithology_report_config_from_mapping': (
		'proc.seis_ssl_cluster.build_f3_lithology_report'
	),
	'f3_lithology_token_dataset_config_from_mapping': (
		'proc.seis_ssl_cluster.build_f3_lithology_token_dataset'
	),
	'f3_lithology_visualization_config_from_mapping': (
		'proc.seis_ssl_cluster.visualize_f3_lithology_predictions'
	),
}


def __getattr__(name: str) -> object:
	try:
		module_name = _LAZY_EXPORTS[name]
	except KeyError as exc:
		msg = f'module {__name__!r} has no attribute {name!r}'
		raise AttributeError(msg) from exc
	value = getattr(import_module(module_name), name)
	globals()[name] = value
	return value


def __dir__() -> list[str]:
	return sorted((*globals(), *_LAZY_EXPORTS))


__all__ = [
	'f3_lithology_prediction_config_from_mapping',
	'f3_lithology_probe_config_from_mapping',
	'f3_lithology_publish_config_from_mapping',
	'f3_lithology_report_config_from_mapping',
	'f3_lithology_token_dataset_config_from_mapping',
	'f3_lithology_visualization_config_from_mapping',
	'f3_prepare_volume_config_from_mapping',
]
