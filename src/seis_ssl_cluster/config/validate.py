"""Compatibility exports for stage-specific config validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from seis_ssl_cluster.config.artifact_path_validation import (
	_validate_artifact_output_path,
)
from seis_ssl_cluster.config.base import (
	_resolve_base,
)
from seis_ssl_cluster.config.cluster_visualization import (
	resolve_cluster_visualization_config,
)
from seis_ssl_cluster.config.clustering import resolve_clustering_config
from seis_ssl_cluster.config.common import (
	_is_int,
	_is_number,
	_required_mapping,
	_validate_non_empty_path,
	_validate_non_empty_str,
	_validate_path,
	_validate_positive_int,
	_validate_positive_number,
	_validate_required_key,
)
from seis_ssl_cluster.config.embedding import resolve_embedding_extraction_config
from seis_ssl_cluster.config.f3_inspection import resolve_f3_facies_inspection_config
from seis_ssl_cluster.config.pretraining import resolve_mae_training_config
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

Config: TypeAlias = dict[str, object]
_T = TypeVar('_T', bound=Mapping[str, object])
_Resolver: TypeAlias = Callable[[Mapping[str, object]], Config]

_LAZY_COMPAT_EXPORTS = {
	'DEFAULT_ALLOWED_SUFFIXES': 'seis_ssl_cluster.config.results',
	'DEFAULT_LOCAL_PATH_MARKERS': 'seis_ssl_cluster.config.results',
	'DEFAULT_MAX_FILE_SIZE_BYTES': 'seis_ssl_cluster.config.results',
	'FORBIDDEN_SUFFIXES': 'seis_ssl_cluster.config.results',
	'LOCAL_PATH_POLICY_ERROR': 'seis_ssl_cluster.config.results',
	'LOCAL_PATH_POLICY_WARNING': 'seis_ssl_cluster.config.results',
	'PUBLISH_MANIFEST_NAME': 'seis_ssl_cluster.config.results',
	'PublishedItem': 'seis_ssl_cluster.config.results',
	'PublishItem': 'seis_ssl_cluster.config.results',
	'PublishManifest': 'seis_ssl_cluster.config.results',
	'ResultsValidationFinding': 'seis_ssl_cluster.config.results',
	'ResultsValidationReport': 'seis_ssl_cluster.config.results',
	'SkippedOptionalItem': 'seis_ssl_cluster.config.results',
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
	'publish_manifest_to_dict': 'seis_ssl_cluster.config.results',
	'publish_selected_results': 'seis_ssl_cluster.config.results',
	'random_mae_checkpoint_config_from_mapping': (
		'seis_ssl_cluster.config.f3_baselines'
	),
	'validate_results_artifacts': 'seis_ssl_cluster.config.results',
}

_FIXED_DISABLED_NORMALIZATION_KEYS = frozenset(
	{
		'smooth_time_depth_trend_correction',
		'trace_wise_agc',
		'patch_wise_zscore',
	},
)


def resolve_manifest_build_config(config: _T) -> Config:
	"""Validate and resolve raw config for the manifest-build entrypoint."""
	resolved, paths = _resolve_base(config, STAGE_BUILD_MANIFESTS)
	manifest = _required_mapping(resolved, 'manifest')
	_validate_non_empty_path(manifest, 'input_path_list', prefix='manifest')
	_validate_artifact_output_path(
		_validate_path(manifest, 'output_dir', prefix='manifest'),
		'manifest.output_dir',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
	)
	_validate_artifact_output_path(
		_validate_path(manifest, 'normalization_stats_dir', prefix='manifest'),
		'manifest.normalization_stats_dir',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
	)
	_validate_non_empty_str(manifest, 'output_name', prefix='manifest')
	return resolved


def resolve_normalization_stats_config(config: _T) -> Config:
	"""Validate and resolve raw config for normalization-stat preparation."""
	resolved, _paths = _resolve_base(config, STAGE_NORMALIZATION_STATS)
	manifests = _required_mapping(resolved, 'manifests')
	_validate_non_empty_path(manifests, 'train', prefix='manifests')
	normalization = _required_mapping(resolved, 'normalization')
	_validate_normalization(normalization)
	return resolved


def resolve_normalization_qc_config(config: _T) -> Config:
	"""Validate and resolve raw config for normalization QC filtering."""
	resolved, paths = _resolve_base(config, STAGE_NORMALIZATION_QC)
	manifests = _required_mapping(resolved, 'manifests')
	splits = _required_mapping(resolved, 'splits')
	qc = _required_mapping(resolved, 'qc')
	_validate_non_empty_path(manifests, 'input', prefix='manifests')
	_validate_non_empty_path(splits, 'input', prefix='splits')
	for parent, key, prefix in (
		(manifests, 'output', 'manifests'),
		(splits, 'output', 'splits'),
		(qc, 'output_json', 'qc'),
		(qc, 'excluded_surveys', 'qc'),
	):
		label = f'{prefix}.{key}'
		_validate_artifact_output_path(
			_validate_path(parent, key, prefix=prefix),
			label,
			artifact_root=paths.artifact_root,
			nopims_root=paths.nopims_root,
		)
	for key in ('min_iqr', 'max_normalized_abs'):
		_validate_required_key(qc, key, prefix='qc')
		_validate_positive_number(qc, key, prefix='qc')
	return resolved


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


def _validate_normalization(normalization: Mapping[str, object]) -> None:
	for key in ('clipping_percentiles', 'epsilon', 'max_samples', 'seed'):
		_validate_required_key(normalization, key, prefix='normalization')
	value = normalization.get('clipping_percentiles')
	if (
		not isinstance(value, list)
		or len(value) != 2
		or not all(_is_number(item) for item in value)
		or float(value[0]) >= float(value[1])
	):
		msg = 'normalization.clipping_percentiles must be two increasing numbers'
		raise ValueError(msg)
	_validate_positive_number(normalization, 'epsilon', prefix='normalization')
	_validate_positive_int(normalization, 'max_samples', prefix='normalization')
	if not _is_int(normalization.get('seed')):
		msg = (
			'normalization.seed must be an integer; '
			f'got {normalization.get("seed")!r}'
		)
		raise ValueError(msg)
	for key in sorted(_FIXED_DISABLED_NORMALIZATION_KEYS):
		if key in normalization:
			msg = (
				f'normalization.{key} is fixed disabled by the amplitude-only '
				'implementation contract and must be removed from raw YAML.'
			)
			raise ValueError(msg)


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
