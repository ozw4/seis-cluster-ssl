"""F3 lithology baseline config validation entrypoints."""

from importlib import import_module
from typing import TYPE_CHECKING

from seis_ssl_cluster.f3.baseline_features import (
	f3_lithology_baseline_token_dataset_config_from_mapping,
)
from seis_ssl_cluster.training.random_checkpoint import (
	random_mae_checkpoint_config_from_mapping,
)

if TYPE_CHECKING:
	from proc.seis_ssl_cluster.build_f3_lithology_comparison_report import (
		f3_lithology_comparison_publish_config_from_mapping,
		f3_lithology_comparison_report_config_from_mapping,
	)

_LAZY_EXPORTS = {
	'f3_lithology_comparison_publish_config_from_mapping': (
		'proc.seis_ssl_cluster.build_f3_lithology_comparison_report'
	),
	'f3_lithology_comparison_report_config_from_mapping': (
		'proc.seis_ssl_cluster.build_f3_lithology_comparison_report'
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
	'f3_lithology_baseline_token_dataset_config_from_mapping',
	'f3_lithology_comparison_publish_config_from_mapping',
	'f3_lithology_comparison_report_config_from_mapping',
	'random_mae_checkpoint_config_from_mapping',
]
