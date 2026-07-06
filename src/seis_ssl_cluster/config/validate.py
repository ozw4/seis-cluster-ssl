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
	_required_child_mapping,
	_required_mapping,
	_resolve_base,
	_ResolvedPaths,
	_validate_absolute_path,
	_validate_allowed_keys,
	_validate_artifact_output_path,
	_validate_bool,
	_validate_fraction,
	_validate_mapping,
	_validate_non_empty_path,
	_validate_non_empty_str,
	_validate_nonnegative_finite_number,
	_validate_nonnegative_int_list,
	_validate_nopims_cluster_visualization_path,
	_validate_nopims_clustering_path,
	_validate_nopims_embedding_path,
	_validate_path,
	_validate_path_under_root,
	_validate_positive_finite_number,
	_validate_positive_int,
	_validate_positive_number,
	_validate_required_key,
	_validate_required_keys,
	_validate_unique_positive_int_list,
)
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

_CLUSTERING_EMBEDDINGS_KEYS = frozenset({'input_dir'})
_CLUSTERING_KEYS = frozenset(
	{
		'output_dir',
		'embedding_normalization',
		'residualization',
		'pca',
		'sample_tokens',
		'method',
		'k_values',
		'minibatch_size',
		'prediction_batch_size',
		'seed',
	},
)
_CLUSTERING_REQUIRED_KEYS = frozenset(
	{
		'output_dir',
		'embedding_normalization',
		'residualization',
		'pca',
		'sample_tokens',
		'method',
		'k_values',
		'minibatch_size',
		'seed',
	},
)
_CLUSTERING_RESIDUALIZATION_KEYS = frozenset(
	{
		'enabled',
		'mode',
		'group_by',
		'add_global_mean_back',
		'min_group_count',
	},
)
_CLUSTERING_RESIDUALIZATION_ENABLED_REQUIRED_KEYS = (
	_CLUSTERING_RESIDUALIZATION_KEYS
)
_CLUSTERING_PCA_KEYS = frozenset({'enabled', 'n_components', 'whiten'})
_VISUALIZATION_CLUSTERING_KEYS = frozenset({'input_dir'})
_VISUALIZATION_KEYS = frozenset(
	{
		'output_dir',
		'survey_ids',
		'modes',
		'reconstruct_voxel',
		'allow_all_surveys_for_voxel_reconstruction',
		'skip_existing_voxel_labels',
		'max_voxel_output_gib',
		'allow_large_voxel_output',
		'slice_coordinate_space',
		'xy_slices',
		'xz_slices',
		'dpi',
		'invalid_color',
		'amplitude_underlay',
		'amplitude_comparison',
		'summaries',
	},
)
_VISUALIZATION_REQUIRED_KEYS = _VISUALIZATION_KEYS
_VISUALIZATION_UNDERLAY_KEYS = frozenset({'enabled', 'alpha'})
_VISUALIZATION_COMPARISON_KEYS = frozenset({'enabled', 'alpha'})
_VISUALIZATION_SUMMARY_KEYS = frozenset({'enabled', 'include_amplitude_norm'})
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


def resolve_clustering_config(config: _T) -> Config:
	"""Validate and resolve raw config for embedding clustering."""
	resolved, paths = _resolve_base(
		config,
		STAGE_CLUSTERING,
		require_nopims_root=False,
	)
	embeddings = _required_mapping(resolved, 'embeddings')
	clustering = _required_mapping(resolved, 'clustering')
	if 'residualization' not in clustering:
		resolved['clustering']['residualization'] = {'enabled': False}
		clustering = _required_mapping(resolved, 'clustering')
	_validate_allowed_keys(
		embeddings,
		_CLUSTERING_EMBEDDINGS_KEYS,
		prefix='embeddings',
	)
	_validate_allowed_keys(clustering, _CLUSTERING_KEYS, prefix='clustering')
	_validate_required_keys(
		clustering,
		_CLUSTERING_REQUIRED_KEYS,
		prefix='clustering',
	)
	input_dir = _validate_path(embeddings, 'input_dir', prefix='embeddings')
	_validate_artifact_output_path(
		input_dir,
		'embeddings.input_dir',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
	)
	_validate_nopims_embedding_path(
		input_dir,
		'embeddings.input_dir',
		artifact_root=paths.artifact_root,
	)
	output_dir = _validate_path(clustering, 'output_dir', prefix='clustering')
	_validate_artifact_output_path(
		output_dir,
		'clustering.output_dir',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
	)
	_validate_nopims_clustering_path(
		output_dir,
		'clustering.output_dir',
		artifact_root=paths.artifact_root,
	)
	_validate_clustering_normalization(clustering)
	residualization = _required_child_mapping(
		clustering,
		'residualization',
		prefix='clustering',
	)
	_validate_clustering_residualization(residualization)
	pca = _required_child_mapping(clustering, 'pca', prefix='clustering')
	_validate_allowed_keys(pca, _CLUSTERING_PCA_KEYS, prefix='clustering.pca')
	_validate_required_keys(
		pca,
		_CLUSTERING_PCA_KEYS,
		prefix='clustering.pca',
	)
	_validate_bool(pca, 'enabled', prefix='clustering.pca')
	_validate_positive_int(pca, 'n_components', prefix='clustering.pca')
	_validate_bool(pca, 'whiten', prefix='clustering.pca')
	_validate_positive_int(clustering, 'sample_tokens', prefix='clustering')
	_validate_clustering_method(clustering)
	_validate_unique_positive_int_list(
		clustering,
		'k_values',
		prefix='clustering',
	)
	_validate_positive_int(clustering, 'minibatch_size', prefix='clustering')
	if 'prediction_batch_size' in clustering:
		_validate_positive_int(
			clustering,
			'prediction_batch_size',
			prefix='clustering',
		)
	if not _is_int(clustering.get('seed')):
		msg = f'clustering.seed must be an integer; got {clustering.get("seed")!r}'
		raise ValueError(msg)
	return resolved


def resolve_cluster_visualization_config(config: _T) -> Config:
	"""Validate and resolve raw config for cluster visualization."""
	resolved, paths = _resolve_base(
		config,
		STAGE_CLUSTER_VISUALIZATION,
		require_nopims_root=False,
	)
	clustering = _required_mapping(resolved, 'clustering')
	visualization = _required_mapping(resolved, 'visualization')
	_validate_allowed_keys(
		clustering,
		_VISUALIZATION_CLUSTERING_KEYS,
		prefix='clustering',
	)
	_validate_allowed_keys(
		visualization,
		_VISUALIZATION_KEYS,
		prefix='visualization',
	)
	_validate_required_keys(
		visualization,
		_VISUALIZATION_REQUIRED_KEYS,
		prefix='visualization',
	)
	input_dir = _validate_path(clustering, 'input_dir', prefix='clustering')
	_validate_artifact_output_path(
		input_dir,
		'clustering.input_dir',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
	)
	_validate_nopims_clustering_path(
		input_dir,
		'clustering.input_dir',
		artifact_root=paths.artifact_root,
	)
	output_dir = _validate_path(
		visualization,
		'output_dir',
		prefix='visualization',
	)
	_validate_artifact_output_path(
		output_dir,
		'visualization.output_dir',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
	)
	_validate_nopims_cluster_visualization_path(
		output_dir,
		'visualization.output_dir',
		artifact_root=paths.artifact_root,
	)
	_validate_survey_id_list(visualization)
	_validate_visualization_modes(visualization)
	_validate_bool(visualization, 'reconstruct_voxel', prefix='visualization')
	_validate_bool(
		visualization,
		'allow_all_surveys_for_voxel_reconstruction',
		prefix='visualization',
	)
	_validate_bool(
		visualization,
		'skip_existing_voxel_labels',
		prefix='visualization',
	)
	_validate_nonnegative_finite_number(
		visualization,
		'max_voxel_output_gib',
		prefix='visualization',
	)
	_validate_bool(
		visualization,
		'allow_large_voxel_output',
		prefix='visualization',
	)
	_validate_slice_coordinate_space(visualization)
	_validate_nonnegative_int_list(visualization, 'xy_slices', prefix='visualization')
	_validate_nonnegative_int_list(visualization, 'xz_slices', prefix='visualization')
	_validate_positive_int(visualization, 'dpi', prefix='visualization')
	_validate_non_empty_str(visualization, 'invalid_color', prefix='visualization')
	underlay = _required_child_mapping(
		visualization,
		'amplitude_underlay',
		prefix='visualization',
	)
	_validate_allowed_keys(
		underlay,
		_VISUALIZATION_UNDERLAY_KEYS,
		prefix='visualization.amplitude_underlay',
	)
	_validate_required_keys(
		underlay,
		_VISUALIZATION_UNDERLAY_KEYS,
		prefix='visualization.amplitude_underlay',
	)
	_validate_bool(underlay, 'enabled', prefix='visualization.amplitude_underlay')
	_validate_fraction(underlay, 'alpha', prefix='visualization.amplitude_underlay')
	comparison = _required_child_mapping(
		visualization,
		'amplitude_comparison',
		prefix='visualization',
	)
	_validate_allowed_keys(
		comparison,
		_VISUALIZATION_COMPARISON_KEYS,
		prefix='visualization.amplitude_comparison',
	)
	_validate_required_keys(
		comparison,
		_VISUALIZATION_COMPARISON_KEYS,
		prefix='visualization.amplitude_comparison',
	)
	_validate_bool(
		comparison,
		'enabled',
		prefix='visualization.amplitude_comparison',
	)
	_validate_fraction(
		comparison,
		'alpha',
		prefix='visualization.amplitude_comparison',
	)
	summaries = _required_child_mapping(
		visualization,
		'summaries',
		prefix='visualization',
	)
	_validate_allowed_keys(
		summaries,
		_VISUALIZATION_SUMMARY_KEYS,
		prefix='visualization.summaries',
	)
	_validate_required_keys(
		summaries,
		_VISUALIZATION_SUMMARY_KEYS,
		prefix='visualization.summaries',
	)
	_validate_bool(summaries, 'enabled', prefix='visualization.summaries')
	_validate_bool(
		summaries,
		'include_amplitude_norm',
		prefix='visualization.summaries',
	)
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


def _validate_clustering_residualization(
	residualization: Mapping[str, object],
) -> None:
	_validate_allowed_keys(
		residualization,
		_CLUSTERING_RESIDUALIZATION_KEYS,
		prefix='clustering.residualization',
	)
	_validate_required_key(
		residualization,
		'enabled',
		prefix='clustering.residualization',
	)
	_validate_bool(
		residualization,
		'enabled',
		prefix='clustering.residualization',
	)
	if not residualization['enabled']:
		return
	_validate_required_keys(
		residualization,
		_CLUSTERING_RESIDUALIZATION_ENABLED_REQUIRED_KEYS,
		prefix='clustering.residualization',
	)
	if residualization['mode'] != 'local_token_position':
		msg = (
			"clustering.residualization.mode must be 'local_token_position'; "
			f'got {residualization["mode"]!r}'
		)
		raise ValueError(msg)
	if residualization['group_by'] not in {'token_phase', 'local_token_position'}:
		msg = (
			'clustering.residualization.group_by must be '
			"'token_phase' or 'local_token_position'; "
			f'got {residualization["group_by"]!r}'
		)
		raise ValueError(msg)
	_validate_bool(
		residualization,
		'add_global_mean_back',
		prefix='clustering.residualization',
	)
	_validate_positive_int(
		residualization,
		'min_group_count',
		prefix='clustering.residualization',
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


def _validate_clustering_normalization(clustering: Mapping[str, object]) -> None:
	value = clustering.get('embedding_normalization')
	if value not in {'l2', 'none'}:
		msg = (
			'clustering.embedding_normalization must be "l2" or "none"; '
			f'got {value!r}'
		)
		raise ValueError(msg)


def _validate_clustering_method(clustering: Mapping[str, object]) -> None:
	value = clustering.get('method')
	if value != 'minibatch_kmeans':
		msg = 'clustering.method must be "minibatch_kmeans"'
		raise ValueError(msg)


def _validate_survey_id_list(visualization: Mapping[str, object]) -> None:
	value = visualization.get('survey_ids')
	if not isinstance(value, list) or any(
		not isinstance(item, str) or not item
		for item in value
	):
		msg = 'visualization.survey_ids must be a list of non-empty strings'
		raise ValueError(msg)


def _validate_visualization_modes(visualization: Mapping[str, object]) -> None:
	value = visualization.get('modes')
	if (
		not isinstance(value, list)
		or not value
		or any(not isinstance(item, str) for item in value)
	):
		msg = 'visualization.modes must be a non-empty list of strings'
		raise ValueError(msg)
	unknown = sorted(set(value) - {'token', 'voxel'})
	if unknown:
		msg = f'visualization.modes contains unsupported mode(s): {unknown!r}'
		raise ValueError(msg)
	if len(set(value)) != len(value):
		msg = f'visualization.modes must not contain duplicates; got {value!r}'
		raise ValueError(msg)


def _validate_slice_coordinate_space(visualization: Mapping[str, object]) -> None:
	value = visualization.get('slice_coordinate_space')
	if value != 'voxel':
		msg = (
			'visualization.slice_coordinate_space must be "voxel"; '
			f'got {value!r}'
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
