"""Stage-specific validation and resolution for SeisSSLCluster configs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import TypeAlias, TypeVar

from seis_ssl_cluster.config.common import (
	_is_int,
	_is_number,
	_reject_legacy_attribute_config,
	_reject_stage_key,
	_required_mapping,
	_resolve_base,
	_ResolvedPaths,
	_validate_absolute_path,
	_validate_allowed_keys,
	_validate_artifact_output_path,
	_validate_bool,
	_validate_mapping,
	_validate_non_empty_path,
	_validate_non_empty_str,
	_validate_path,
	_validate_path_under_root,
	_validate_positive_finite_number,
	_validate_positive_int,
	_validate_positive_number,
	_validate_required_key,
	_validate_required_keys,
)
from seis_ssl_cluster.config.cluster_visualization import (
	resolve_cluster_visualization_config,
)
from seis_ssl_cluster.config.clustering import resolve_clustering_config
from seis_ssl_cluster.config.embedding import resolve_embedding_extraction_config
from seis_ssl_cluster.config.pretraining import resolve_mae_training_config
from seis_ssl_cluster.config.schema import (
	F3_FACIES_DATASET_NAME,
	F3_FACIES_DATASET_VERSION,
	F3_FACIES_INSPECTION_ARTIFACT_SUBDIR,
	F3_FACIES_INSPECTION_STAGES,
	KNOWN_STAGES,
	STAGE_BUILD_MANIFESTS,
	STAGE_CLUSTER_VISUALIZATION,
	STAGE_CLUSTERING,
	STAGE_EMBEDDING_EXTRACTION,
	STAGE_MAE_TRAINING,
	STAGE_NORMALIZATION_QC,
	STAGE_NORMALIZATION_STATS,
)

Config: TypeAlias = dict[str, object]
_T = TypeVar('_T', bound=Mapping[str, object])
_Resolver: TypeAlias = Callable[[Mapping[str, object]], Config]

_FIXED_DISABLED_NORMALIZATION_KEYS = frozenset(
	{
		'smooth_time_depth_trend_correction',
		'trace_wise_agc',
		'patch_wise_zscore',
	},
)

_F3_FACIES_INSPECTION_TOP_LEVEL = frozenset(
	{'paths', 'outputs', 'dataset', 'inspection', 'publish'},
)
_F3_FACIES_INSPECTION_REQUIRED_TOP_LEVEL = frozenset(
	{'paths', 'outputs', 'dataset', 'inspection'},
)
_F3_FACIES_INSPECTION_PATH_KEYS = frozenset({'f3_root', 'artifact_root'})
_F3_FACIES_INSPECTION_OUTPUT_KEYS = frozenset({'inspection_dir'})
_F3_FACIES_INSPECTION_DATASET_KEYS = frozenset({'name', 'version'})
_F3_FACIES_INSPECTION_PUBLISH_KEYS = frozenset(
	{'enabled', 'output_dir', 'include_figures', 'max_file_size_mb'},
)
_F3_FACIES_INSPECTION_PATH_KEY_SUFFIXES = (
	'_dir',
	'_json',
	'_csv',
	'_markdown',
	'_png',
	'_path',
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


def resolve_f3_facies_inspection_config(config: _T, *, stage: str) -> Config:
	"""Validate and resolve a raw config for an F3 facies inspection entrypoint."""
	if stage not in F3_FACIES_INSPECTION_STAGES:
		msg = (
			f'stage must be one of {sorted(F3_FACIES_INSPECTION_STAGES)!r}; '
			f'got {stage!r}'
		)
		raise ValueError(msg)
	_validate_mapping(config)
	_reject_legacy_attribute_config(config)
	_reject_stage_key(config)
	_validate_f3_facies_inspection_top_level_sections(config, stage)

	resolved = deepcopy(dict(config))
	resolved['stage'] = stage
	paths = _validate_f3_facies_inspection_paths(
		_required_mapping(resolved, 'paths'),
	)
	inspection_dir = _validate_f3_facies_inspection_outputs(
		_required_mapping(resolved, 'outputs'),
		paths=paths,
	)
	_validate_f3_facies_dataset(_required_mapping(resolved, 'dataset'))
	if 'publish' in resolved:
		_validate_f3_facies_inspection_publish(
			_required_mapping(resolved, 'publish'),
		)
	inspection = _required_mapping(resolved, 'inspection')
	if not inspection:
		msg = 'inspection must contain stage-specific settings'
		raise ValueError(msg)
	_validate_f3_facies_inspection_artifact_paths(
		inspection,
		inspection_dir=inspection_dir,
	)
	return resolved


def validate_config(config: _T, *, stage: str) -> Config:
	"""Resolve raw config for an explicit stage selected by caller code."""
	try:
		resolver = _STAGE_RESOLVERS[stage]
	except KeyError as exc:
		msg = f'stage must be one of {sorted(KNOWN_STAGES)!r}; got {stage!r}'
		raise ValueError(msg) from exc
	return resolver(config)


def _validate_f3_facies_inspection_top_level_sections(
	config: Mapping[str, object],
	stage: str,
) -> None:
	keys = set(config)
	unexpected = sorted(keys - _F3_FACIES_INSPECTION_TOP_LEVEL)
	if unexpected:
		msg = (
			f'top-level section(s) not allowed for {stage}: {unexpected!r}; '
			'allowed sections are '
			f'{sorted(_F3_FACIES_INSPECTION_TOP_LEVEL)!r}'
		)
		raise ValueError(msg)
	missing = sorted(_F3_FACIES_INSPECTION_REQUIRED_TOP_LEVEL - keys)
	if missing:
		msg = f'missing required top-level section(s) for {stage}: {missing!r}'
		raise ValueError(msg)


def _validate_f3_facies_inspection_paths(
	paths: Mapping[str, object],
) -> _ResolvedPaths:
	_validate_allowed_keys(
		paths,
		_F3_FACIES_INSPECTION_PATH_KEYS,
		prefix='paths',
	)
	return _ResolvedPaths(
		f3_root=_validate_absolute_path(paths, 'f3_root', prefix='paths'),
		artifact_root=_validate_absolute_path(paths, 'artifact_root', prefix='paths'),
	)


def _validate_f3_facies_inspection_outputs(
	outputs: Mapping[str, object],
	*,
	paths: _ResolvedPaths,
) -> Path:
	_validate_allowed_keys(
		outputs,
		_F3_FACIES_INSPECTION_OUTPUT_KEYS,
		prefix='outputs',
	)
	_validate_required_keys(
		outputs,
		_F3_FACIES_INSPECTION_OUTPUT_KEYS,
		prefix='outputs',
	)
	inspection_dir = _validate_path(outputs, 'inspection_dir', prefix='outputs')
	_validate_artifact_output_path(
		inspection_dir,
		'outputs.inspection_dir',
		artifact_root=paths.artifact_root,
		nopims_root=paths.f3_root,
		raw_root_label='paths.f3_root',
	)
	expected_relative = Path(F3_FACIES_INSPECTION_ARTIFACT_SUBDIR)
	actual_relative = inspection_dir.resolve(strict=False).relative_to(
		paths.artifact_root.resolve(strict=False),
	)
	if actual_relative != expected_relative:
		msg = (
			'outputs.inspection_dir must be paths.artifact_root / '
			f'{F3_FACIES_INSPECTION_ARTIFACT_SUBDIR!r}; got {inspection_dir}'
		)
		raise ValueError(msg)
	return inspection_dir


def _validate_f3_facies_inspection_artifact_paths(
	inspection: Mapping[str, object],
	*,
	inspection_dir: Path,
	prefix: str = 'inspection',
) -> None:
	for key, value in inspection.items():
		label = f'{prefix}.{key}'
		if isinstance(value, Mapping):
			_validate_f3_facies_inspection_artifact_paths(
				value,
				inspection_dir=inspection_dir,
				prefix=label,
			)
			continue
		if not _is_f3_facies_inspection_path_key(key):
			continue
		if not isinstance(value, str) or not value:
			msg = f'{label} must be a non-empty string; got {value!r}'
			raise TypeError(msg)
		_validate_path_under_root(
			Path(value),
			label,
			root=inspection_dir,
			root_label='outputs.inspection_dir',
		)


def _is_f3_facies_inspection_path_key(key: str) -> bool:
	return key.endswith(_F3_FACIES_INSPECTION_PATH_KEY_SUFFIXES)


def _validate_f3_facies_dataset(dataset: Mapping[str, object]) -> None:
	_validate_allowed_keys(
		dataset,
		_F3_FACIES_INSPECTION_DATASET_KEYS,
		prefix='dataset',
	)
	_validate_required_keys(
		dataset,
		_F3_FACIES_INSPECTION_DATASET_KEYS,
		prefix='dataset',
	)
	if dataset.get('name') != F3_FACIES_DATASET_NAME:
		msg = (
			f'dataset.name must be {F3_FACIES_DATASET_NAME!r}; '
			f'got {dataset.get("name")!r}'
		)
		raise ValueError(msg)
	if dataset.get('version') != F3_FACIES_DATASET_VERSION:
		msg = (
			f'dataset.version must be {F3_FACIES_DATASET_VERSION!r}; '
			f'got {dataset.get("version")!r}'
		)
		raise ValueError(msg)


def _validate_f3_facies_inspection_publish(
	publish: Mapping[str, object],
) -> None:
	_validate_allowed_keys(
		publish,
		_F3_FACIES_INSPECTION_PUBLISH_KEYS,
		prefix='publish',
	)
	if 'enabled' in publish:
		_validate_bool(publish, 'enabled', prefix='publish')
	if 'include_figures' in publish:
		_validate_bool(publish, 'include_figures', prefix='publish')
	if 'output_dir' in publish:
		_validate_path(publish, 'output_dir', prefix='publish')
	if publish.get('enabled') is True and 'output_dir' not in publish:
		msg = 'publish.output_dir is required when publish.enabled is true'
		raise ValueError(msg)
	if 'max_file_size_mb' in publish:
		_validate_positive_finite_number(
			publish,
			'max_file_size_mb',
			prefix='publish',
		)


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
	'resolve_cluster_visualization_config',
	'resolve_clustering_config',
	'resolve_embedding_extraction_config',
	'resolve_f3_facies_inspection_config',
	'resolve_mae_training_config',
	'resolve_manifest_build_config',
	'resolve_normalization_qc_config',
	'resolve_normalization_stats_config',
	'validate_config',
]
