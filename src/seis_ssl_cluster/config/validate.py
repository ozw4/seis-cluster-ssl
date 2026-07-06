"""Backward-compatible config validation entrypoints."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from seis_ssl_cluster.config.cluster_visualization import (
	resolve_cluster_visualization_config,
)
from seis_ssl_cluster.config.clustering import resolve_clustering_config
from seis_ssl_cluster.config.embedding import resolve_embedding_extraction_config
from seis_ssl_cluster.config.f3_inspection import resolve_f3_facies_inspection_config
from seis_ssl_cluster.config.manifest import resolve_manifest_build_config
from seis_ssl_cluster.config.normalization import (
	resolve_normalization_qc_config,
	resolve_normalization_stats_config,
)
from seis_ssl_cluster.config.pretraining import resolve_mae_training_config
from seis_ssl_cluster.config.results import (
	DEFAULT_ALLOWED_SUFFIXES,
	DEFAULT_LOCAL_PATH_MARKERS,
	DEFAULT_MAX_FILE_SIZE_BYTES,
	FORBIDDEN_SUFFIXES,
	LOCAL_PATH_POLICY_ERROR,
	LOCAL_PATH_POLICY_WARNING,
	PUBLISH_MANIFEST_NAME,
	PublishedItem,
	PublishItem,
	PublishManifest,
	ResultsValidationFinding,
	ResultsValidationReport,
	SkippedOptionalItem,
	publish_manifest_to_dict,
	publish_selected_results,
	validate_results_artifacts,
)
from seis_ssl_cluster.config.schema import (
	KNOWN_STAGES,
	STAGE_BUILD_MANIFESTS,
	STAGE_CLUSTER_VISUALIZATION,
	STAGE_CLUSTERING,
	STAGE_EMBEDDING_EXTRACTION,
	STAGE_MAE_TRAINING,
	STAGE_NORMALIZATION_QC,
	STAGE_NORMALIZATION_STATS,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.config.f3_baselines import (
		f3_lithology_baseline_token_dataset_config_from_mapping,
		f3_lithology_comparison_publish_config_from_mapping,
		f3_lithology_comparison_report_config_from_mapping,
		random_mae_checkpoint_config_from_mapping,
	)
	from seis_ssl_cluster.config.f3_lithology import (
		f3_lithology_prediction_config_from_mapping,
		f3_lithology_probe_config_from_mapping,
		f3_lithology_publish_config_from_mapping,
		f3_lithology_report_config_from_mapping,
		f3_lithology_token_dataset_config_from_mapping,
		f3_lithology_visualization_config_from_mapping,
		f3_prepare_volume_config_from_mapping,
	)

Config: TypeAlias = dict[str, object]
_T = TypeVar('_T', bound=Mapping[str, object])
_Resolver: TypeAlias = Callable[[Mapping[str, object]], Config]

_LAZY_COMPAT_EXPORTS = {
	'f3_lithology_baseline_token_dataset_config_from_mapping': (
		'seis_ssl_cluster.config.f3_baselines'
	),
	'f3_lithology_comparison_publish_config_from_mapping': (
		'seis_ssl_cluster.config.f3_baselines'
	),
	'f3_lithology_comparison_report_config_from_mapping': (
		'seis_ssl_cluster.config.f3_baselines'
	),
	'f3_lithology_prediction_config_from_mapping': (
		'seis_ssl_cluster.config.f3_lithology'
	),
	'f3_lithology_probe_config_from_mapping': (
		'seis_ssl_cluster.config.f3_lithology'
	),
	'f3_lithology_publish_config_from_mapping': (
		'seis_ssl_cluster.config.f3_lithology'
	),
	'f3_lithology_report_config_from_mapping': (
		'seis_ssl_cluster.config.f3_lithology'
	),
	'f3_lithology_token_dataset_config_from_mapping': (
		'seis_ssl_cluster.config.f3_lithology'
	),
	'f3_lithology_visualization_config_from_mapping': (
		'seis_ssl_cluster.config.f3_lithology'
	),
	'f3_prepare_volume_config_from_mapping': 'seis_ssl_cluster.config.f3_lithology',
	'random_mae_checkpoint_config_from_mapping': (
		'seis_ssl_cluster.config.f3_baselines'
	),
}


def validate_config(config: _T, *, stage: str) -> Config:
	"""Resolve raw config for an explicit stage selected by caller code."""
	try:
		resolver = _STAGE_RESOLVERS[stage]
	except KeyError as exc:
		msg = f'stage must be one of {sorted(KNOWN_STAGES)!r}; got {stage!r}'
		raise ValueError(msg) from exc
	return resolver(config)


def __getattr__(name: str) -> object:
	try:
		module_name = _LAZY_COMPAT_EXPORTS[name]
	except KeyError as exc:
		msg = f'module {__name__!r} has no attribute {name!r}'
		raise AttributeError(msg) from exc
	value = getattr(import_module(module_name), name)
	globals()[name] = value
	return value


def __dir__() -> list[str]:
	return sorted((*globals(), *_LAZY_COMPAT_EXPORTS))


_STAGE_RESOLVERS: dict[str, _Resolver] = {
	STAGE_BUILD_MANIFESTS: resolve_manifest_build_config,
	STAGE_NORMALIZATION_STATS: resolve_normalization_stats_config,
	STAGE_NORMALIZATION_QC: resolve_normalization_qc_config,
	STAGE_MAE_TRAINING: resolve_mae_training_config,
	STAGE_EMBEDDING_EXTRACTION: resolve_embedding_extraction_config,
	STAGE_CLUSTERING: resolve_clustering_config,
	STAGE_CLUSTER_VISUALIZATION: resolve_cluster_visualization_config,
}

__all__ = [
	'DEFAULT_ALLOWED_SUFFIXES',
	'DEFAULT_LOCAL_PATH_MARKERS',
	'DEFAULT_MAX_FILE_SIZE_BYTES',
	'FORBIDDEN_SUFFIXES',
	'LOCAL_PATH_POLICY_ERROR',
	'LOCAL_PATH_POLICY_WARNING',
	'PUBLISH_MANIFEST_NAME',
	'PublishItem',
	'PublishManifest',
	'PublishedItem',
	'ResultsValidationFinding',
	'ResultsValidationReport',
	'SkippedOptionalItem',
	'f3_lithology_baseline_token_dataset_config_from_mapping',
	'f3_lithology_comparison_publish_config_from_mapping',
	'f3_lithology_comparison_report_config_from_mapping',
	'f3_lithology_prediction_config_from_mapping',
	'f3_lithology_probe_config_from_mapping',
	'f3_lithology_publish_config_from_mapping',
	'f3_lithology_report_config_from_mapping',
	'f3_lithology_token_dataset_config_from_mapping',
	'f3_lithology_visualization_config_from_mapping',
	'f3_prepare_volume_config_from_mapping',
	'publish_manifest_to_dict',
	'publish_selected_results',
	'random_mae_checkpoint_config_from_mapping',
	'resolve_cluster_visualization_config',
	'resolve_clustering_config',
	'resolve_embedding_extraction_config',
	'resolve_f3_facies_inspection_config',
	'resolve_mae_training_config',
	'resolve_manifest_build_config',
	'resolve_normalization_qc_config',
	'resolve_normalization_stats_config',
	'validate_config',
	'validate_results_artifacts',
]
