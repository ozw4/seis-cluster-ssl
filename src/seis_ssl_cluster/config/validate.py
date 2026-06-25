"""Stage-specific validation and resolution for amplitude-only configs."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from numbers import Integral, Real
from pathlib import Path
from typing import TypeAlias, TypeVar

from seis_ssl_cluster.config.schema import (
	DEFAULT_MAE_DATA_OPTIONS,
	DEFAULT_MAE_DEBUG_VISUALIZATION_OPTIONS,
	DEFAULT_MAE_LOSS_OPTIONS,
	DEFAULT_MAE_TRAIN_OPTIONS,
	DEFAULT_ZERO_MASK_CONTRACT,
	EXPECTED_VALID_MASK_MODE,
	FIXED_DATA_CONTRACT,
	FIXED_LOSS_CONTRACT,
	FIXED_MASKING_CONTRACT,
	FIXED_MODEL_CONTRACT,
	KNOWN_STAGES,
	LEGACY_ATTRIBUTE_KEY_NAMES,
	LEGACY_ATTRIBUTE_KEY_PATHS,
	MAE_DEBUG_VISUALIZATION_COLUMNS,
	MAE_DEBUG_VISUALIZATION_KEYS,
	STAGE_BUILD_MANIFESTS,
	STAGE_CLUSTER_VISUALIZATION,
	STAGE_CLUSTERING,
	STAGE_EMBEDDING_EXTRACTION,
	STAGE_MAE_TRAINING,
	STAGE_NORMALIZATION_QC,
	STAGE_NORMALIZATION_STATS,
	STAGE_PATH_KEYS,
	SUPPORTED_RECONSTRUCTION_LOSSES,
	SUPPORTED_TARGET_NORMALIZATION_MODES,
)

Config: TypeAlias = dict[str, object]
_T = TypeVar('_T', bound=Mapping[str, object])
_Resolver: TypeAlias = Callable[[Mapping[str, object]], Config]

_ALLOWED_TOP_LEVEL: dict[str, frozenset[str]] = {
	STAGE_BUILD_MANIFESTS: frozenset({'paths', 'manifest'}),
	STAGE_NORMALIZATION_STATS: frozenset(
		{'paths', 'manifests', 'normalization'},
	),
	STAGE_NORMALIZATION_QC: frozenset({'paths', 'manifests', 'splits', 'qc'}),
	STAGE_MAE_TRAINING: frozenset(
		{
			'paths',
			'manifests',
			'data',
			'zero_mask',
			'model',
			'masking',
			'loss',
			'train',
			'visualization',
		},
	),
	STAGE_EMBEDDING_EXTRACTION: frozenset(
		{'paths', 'manifests', 'embeddings', 'embedding'},
	),
	STAGE_CLUSTERING: frozenset({'paths', 'embeddings', 'clustering'}),
	STAGE_CLUSTER_VISUALIZATION: frozenset(
		{'paths', 'clustering', 'visualization'},
	),
}

_REQUIRED_TOP_LEVEL: dict[str, frozenset[str]] = {
	STAGE_BUILD_MANIFESTS: frozenset({'paths', 'manifest'}),
	STAGE_NORMALIZATION_STATS: frozenset(
		{'paths', 'manifests', 'normalization'},
	),
	STAGE_NORMALIZATION_QC: frozenset({'paths', 'manifests', 'splits', 'qc'}),
	STAGE_MAE_TRAINING: frozenset(
		{'paths', 'manifests', 'data', 'model', 'masking', 'loss', 'train'},
	),
	STAGE_EMBEDDING_EXTRACTION: frozenset(
		{'paths', 'manifests', 'embeddings', 'embedding'},
	),
	STAGE_CLUSTERING: frozenset({'paths', 'embeddings', 'clustering'}),
	STAGE_CLUSTER_VISUALIZATION: frozenset(
		{'paths', 'clustering', 'visualization'},
	),
}

_FIXED_RAW_KEYS: dict[str, frozenset[str]] = {
	'data': frozenset(FIXED_DATA_CONTRACT),
	'model': frozenset(FIXED_MODEL_CONTRACT),
	'masking': frozenset(FIXED_MASKING_CONTRACT),
	'loss': frozenset(FIXED_LOSS_CONTRACT),
}

_FIXED_DISABLED_NORMALIZATION_KEYS = frozenset(
	{
		'smooth_time_depth_trend_correction',
		'trace_wise_agc',
		'patch_wise_zscore',
	},
)

_CHECKPOINT_OWNED_EXTRACTION_SECTIONS = frozenset(
	{'data', 'model', 'masking', 'loss', 'train', 'zero_mask'},
)

_CLUSTERING_EMBEDDINGS_KEYS = frozenset({'input_dir'})
_CLUSTERING_KEYS = frozenset(
	{
		'output_dir',
		'embedding_normalization',
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
		'pca',
		'sample_tokens',
		'method',
		'k_values',
		'minibatch_size',
		'seed',
	},
)
_CLUSTERING_PCA_KEYS = frozenset({'enabled', 'n_components', 'whiten'})
_VISUALIZATION_CLUSTERING_KEYS = frozenset({'input_dir'})
_MAE_TRAINING_VISUALIZATION_KEYS = frozenset({'mae_debug'})
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


def resolve_mae_training_config(config: _T) -> Config:
	"""Validate and resolve raw config for MAE training."""
	resolved, paths = _resolve_base(
		config,
		STAGE_MAE_TRAINING,
		require_nopims_root=False,
	)
	paths_config = _required_mapping(resolved, 'paths')
	output_root = _validate_path(
		paths_config,
		'output_root',
		prefix='paths',
	)
	_validate_artifact_output_path(
		output_root,
		'paths.output_root',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
	)
	_reject_fixed_contract_keys(resolved)
	_merge_section_defaults(resolved, 'data', DEFAULT_MAE_DATA_OPTIONS)
	_merge_section_defaults(resolved, 'train', DEFAULT_MAE_TRAIN_OPTIONS)
	_merge_section_defaults(resolved, 'loss', DEFAULT_MAE_LOSS_OPTIONS)
	_merge_section_defaults(resolved, 'zero_mask', DEFAULT_ZERO_MASK_CONTRACT)

	manifests = _required_mapping(resolved, 'manifests')
	_validate_non_empty_path(manifests, 'train', prefix='manifests')
	_validate_non_empty_path(manifests, 'train_path_list', prefix='manifests')

	data = _required_mapping(resolved, 'data')
	model = _required_mapping(resolved, 'model')
	masking = _required_mapping(resolved, 'masking')
	loss = _required_mapping(resolved, 'loss')
	train = _required_mapping(resolved, 'train')

	local_crop_size = _validate_positive_int_triplet(
		data,
		'local_crop_size',
		prefix='data',
	)
	_validate_optional_fraction(data, 'min_valid_fraction', prefix='data')
	if 'max_resample_attempts' in data:
		_validate_positive_int(data, 'max_resample_attempts', prefix='data')
	if 'normalized_clip_abs' in data:
		_validate_positive_finite_number(
			data,
			'normalized_clip_abs',
			prefix='data',
		)

	patch_size = _validate_positive_int_triplet(
		model,
		'patch_size',
		prefix='model',
	)
	_validate_model(model)
	_validate_divisible_crop_patch(local_crop_size, patch_size)
	_validate_masking(masking)
	_validate_loss(loss)
	_validate_train(train)
	_validate_zero_mask(_required_mapping(resolved, 'zero_mask'))
	if 'visualization' in resolved:
		_validate_mae_training_visualization(
			_required_mapping(resolved, 'visualization'),
			output_root=output_root,
		)

	_merge_section_defaults(resolved, 'data', FIXED_DATA_CONTRACT)
	_merge_section_defaults(resolved, 'model', FIXED_MODEL_CONTRACT)
	_merge_section_defaults(resolved, 'masking', FIXED_MASKING_CONTRACT)
	_merge_section_defaults(resolved, 'loss', FIXED_LOSS_CONTRACT)
	return resolved


def resolve_embedding_extraction_config(config: _T) -> Config:
	"""Validate and resolve raw config for embedding extraction."""
	_reject_checkpoint_owned_extraction_sections(config)
	resolved, paths = _resolve_base(
		config,
		STAGE_EMBEDDING_EXTRACTION,
		require_nopims_root=False,
	)
	manifests = _required_mapping(resolved, 'manifests')
	embeddings = _required_mapping(resolved, 'embeddings')
	_validate_non_empty_path(manifests, 'input', prefix='manifests')
	_validate_non_empty_path(embeddings, 'checkpoint', prefix='embeddings')
	_validate_artifact_output_path(
		_validate_path(embeddings, 'output_dir', prefix='embeddings'),
		'embeddings.output_dir',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
	)

	embedding = _required_mapping(resolved, 'embedding')
	window_size = _validate_positive_int_triplet(
		embedding,
		'window_size',
		prefix='embedding',
	)
	overlap = _validate_nonnegative_int_triplet(
		embedding,
		'overlap',
		prefix='embedding',
	)
	_validate_overlap_less_than_window(overlap, window_size)
	_validate_embedding_output_dtype(embedding)
	_validate_positive_int(embedding, 'batch_size', prefix='embedding')
	_validate_fraction(
		embedding,
		'min_token_valid_fraction',
		prefix='embedding',
	)
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
	_validate_non_empty_path(embeddings, 'input_dir', prefix='embeddings')
	_validate_artifact_output_path(
		_validate_path(clustering, 'output_dir', prefix='clustering'),
		'clustering.output_dir',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
	)
	_validate_clustering_normalization(clustering)
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
	_validate_non_empty_path(clustering, 'input_dir', prefix='clustering')
	_validate_artifact_output_path(
		_validate_path(visualization, 'output_dir', prefix='visualization'),
		'visualization.output_dir',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
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


def validate_config(config: _T, *, stage: str) -> Config:
	"""Resolve raw config for an explicit stage selected by caller code."""
	try:
		resolver = _STAGE_RESOLVERS[stage]
	except KeyError as exc:
		msg = f'stage must be one of {sorted(KNOWN_STAGES)!r}; got {stage!r}'
		raise ValueError(msg) from exc
	return resolver(config)


def _resolve_base(
	config: Mapping[str, object],
	stage: str,
	*,
	require_nopims_root: bool = True,
) -> tuple[Config, _ResolvedPaths]:
	_validate_mapping(config)
	_reject_legacy_attribute_config(config)
	_reject_stage_key(config)
	_validate_top_level_sections(config, stage)
	resolved = deepcopy(dict(config))
	resolved['stage'] = stage
	paths = _validate_paths(
		_required_mapping(resolved, 'paths'),
		require_nopims_root=require_nopims_root,
		allowed_keys=STAGE_PATH_KEYS[stage],
	)
	return resolved, paths


class _ResolvedPaths:
	def __init__(self, *, nopims_root: Path | None, artifact_root: Path) -> None:
		self.nopims_root = nopims_root
		self.artifact_root = artifact_root


def _validate_mapping(config: Mapping[str, object]) -> None:
	if not isinstance(config, Mapping):
		msg = 'config must be a mapping'
		raise TypeError(msg)


def _reject_stage_key(config: Mapping[str, object]) -> None:
	if 'stage' in config:
		msg = (
			'stage is selected by the entrypoint; remove the top-level '
			'stage key from this YAML and choose the proc script instead.'
		)
		raise ValueError(msg)


def _validate_top_level_sections(config: Mapping[str, object], stage: str) -> None:
	allowed = _ALLOWED_TOP_LEVEL[stage]
	required = _REQUIRED_TOP_LEVEL[stage]
	keys = set(config)
	unexpected = sorted(keys - allowed)
	if unexpected:
		msg = (
			f'top-level section(s) not allowed for {stage}: {unexpected!r}; '
			f'allowed sections are {sorted(allowed)!r}'
		)
		raise ValueError(msg)
	missing = sorted(required - keys)
	if missing:
		msg = f'missing required top-level section(s) for {stage}: {missing!r}'
		raise ValueError(msg)


def _reject_fixed_contract_keys(config: Mapping[str, object]) -> None:
	for section, fixed_keys in _FIXED_RAW_KEYS.items():
		value = config.get(section)
		if not isinstance(value, Mapping):
			continue
		stale = sorted(set(value) & set(fixed_keys))
		if stale:
			labels = [f'{section}.{key}' for key in stale]
			msg = (
				f'{labels[0]} is fixed by the amplitude-only MVP config '
				'resolver and must be removed from raw YAML.'
			)
			raise ValueError(msg)


def _reject_checkpoint_owned_extraction_sections(
	config: Mapping[str, object],
) -> None:
	stale = sorted(set(config) & _CHECKPOINT_OWNED_EXTRACTION_SECTIONS)
	if stale:
		msg = (
			'embedding extraction config must not include checkpoint-owned '
			f'section(s): {stale!r}'
		)
		raise ValueError(msg)


def _reject_legacy_attribute_config(config: Mapping[str, object]) -> None:
	for path, key in _iter_mapping_keys(config):
		if path in LEGACY_ATTRIBUTE_KEY_PATHS or key in LEGACY_ATTRIBUTE_KEY_NAMES:
			msg = (
				f'{path} is a legacy multi-attribute config key and is not '
				'valid for the amplitude-only MVP; remove fixed-attribute '
				'configuration from this config.'
			)
			raise ValueError(msg)


def _validate_paths(
	paths: Mapping[str, object],
	*,
	require_nopims_root: bool,
	allowed_keys: frozenset[str] | None,
) -> _ResolvedPaths:
	if allowed_keys is not None:
		_validate_allowed_keys(paths, allowed_keys, prefix='paths')
	nopims_root: Path | None = None
	if require_nopims_root or 'nopims_root' in paths:
		nopims_root = _validate_absolute_path(paths, 'nopims_root', prefix='paths')
	if 'output_root' in paths:
		_validate_non_empty_path(paths, 'output_root', prefix='paths')
	return _ResolvedPaths(
		artifact_root=_validate_absolute_path(paths, 'artifact_root', prefix='paths'),
		nopims_root=nopims_root,
	)


def _validate_allowed_keys(
	parent: Mapping[str, object],
	allowed: frozenset[str],
	*,
	prefix: str,
) -> None:
	unexpected = sorted(set(parent) - allowed)
	if unexpected:
		labels = [f'{prefix}.{key}' for key in unexpected]
		msg = (
			f'{prefix} key(s) not allowed: {labels!r}; '
			f'allowed keys are {sorted(allowed)!r}'
		)
		raise ValueError(msg)


def _validate_required_keys(
	parent: Mapping[str, object],
	keys: frozenset[str],
	*,
	prefix: str,
) -> None:
	for key in sorted(keys):
		_validate_required_key(parent, key, prefix=prefix)


def _validate_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	path = _validate_path(parent, key, prefix=prefix)
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _validate_non_empty_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	return _validate_path(parent, key, prefix=prefix)


def _validate_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return Path(value)


def _validate_artifact_output_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
	nopims_root: Path | None,
) -> None:
	if not path.is_absolute():
		msg = f'{label} must be an absolute artifact-registry path; got {path}'
		raise ValueError(msg)
	if nopims_root is not None and _is_relative_to(path, nopims_root):
		msg = f'{label} must not be under paths.nopims_root; got {path}'
		raise ValueError(msg)
	if not _is_relative_to(path, artifact_root):
		msg = f'{label} must be under paths.artifact_root ({artifact_root}); got {path}'
		raise ValueError(msg)


def _validate_optional_output_path_under_root(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
	root: Path,
	root_label: str,
) -> None:
	value = parent.get(key)
	if value is None:
		return
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string or null; got {value!r}'
		raise TypeError(msg)
	_validate_path_under_root(
		Path(value),
		f'{prefix}.{key}',
		root=root,
		root_label=root_label,
	)


def _validate_path_under_root(
	path: Path,
	label: str,
	*,
	root: Path,
	root_label: str,
) -> None:
	if not path.is_absolute():
		msg = f'{label} must be an absolute path; got {path}'
		raise ValueError(msg)
	if not _is_relative_to(path, root):
		msg = f'{label} must be under {root_label} ({root}); got {path}'
		raise ValueError(msg)


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


def _iter_mapping_keys(
	value: object,
	prefix: str = '',
) -> Sequence[tuple[str, str]]:
	if isinstance(value, Sequence) and not isinstance(value, str | bytes):
		paths: list[tuple[str, str]] = []
		for index, child in enumerate(value):
			path = f'{prefix}[{index}]' if prefix else f'[{index}]'
			paths.extend(_iter_mapping_keys(child, path))
		return paths

	if not isinstance(value, Mapping):
		return ()

	paths: list[tuple[str, str]] = []
	for key, child in value.items():
		if not isinstance(key, str):
			continue
		path = f'{prefix}.{key}' if prefix else key
		paths.append((path, key))
		paths.extend(_iter_mapping_keys(child, path))
	return paths


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


def _validate_model(model: Mapping[str, object]) -> None:
	for key in (
		'encoder_dim',
		'encoder_depth',
		'encoder_heads',
		'decoder_dim',
		'decoder_depth',
		'decoder_heads',
	):
		_validate_positive_int(model, key, prefix='model')


def _validate_masking(masking: Mapping[str, object]) -> None:
	ratio = masking.get('spatial_mask_ratio')
	if (
		not isinstance(ratio, Real)
		or isinstance(ratio, bool)
		or ratio <= 0.0
		or ratio >= 1.0
	):
		msg = 'masking.spatial_mask_ratio must be greater than 0 and less than 1'
		raise ValueError(msg)

	_validate_positive_int_triplet(
		masking,
		'block_size_tokens',
		prefix='masking',
	)


def _validate_train(train: Mapping[str, object]) -> None:
	for key in ('batch_size', 'samples_per_epoch', 'epochs'):
		_validate_positive_int(train, key, prefix='train')
	_validate_optional_train_numbers(train)
	_validate_bool(train, 'amp', prefix='train')
	for key in ('shuffle', 'allow_overwrite_output'):
		if key in train:
			_validate_bool(train, key, prefix='train')
	_validate_optional_train_seed(train)
	_validate_optional_train_device(train)


def _validate_optional_train_numbers(train: Mapping[str, object]) -> None:
	for key in ('num_workers', 'max_steps', 'checkpoint_every_steps'):
		if key in train:
			_validate_nonnegative_int(train, key, prefix='train')
	for key in ('lr', 'grad_clip_norm'):
		if key in train:
			_validate_positive_number(train, key, prefix='train')
	if 'weight_decay' in train:
		_validate_nonnegative_number(train, 'weight_decay', prefix='train')


def _validate_optional_train_seed(train: Mapping[str, object]) -> None:
	if 'seed' in train and not _is_int(train.get('seed')):
		msg = f'train.seed must be an integer; got {train.get("seed")!r}'
		raise ValueError(msg)


def _validate_optional_train_device(train: Mapping[str, object]) -> None:
	if 'device' in train:
		value = train.get('device')
		if value not in {'auto', 'cpu', 'cuda'}:
			msg = 'train.device must be "auto", "cpu", or "cuda"'
			raise ValueError(msg)


def _validate_loss(loss: Mapping[str, object]) -> None:
	_validate_required_key(loss, 'reconstruction', prefix='loss')
	reconstruction = loss.get('reconstruction')
	if reconstruction not in SUPPORTED_RECONSTRUCTION_LOSSES:
		msg = (
			'loss.reconstruction must be one of '
			f'{sorted(SUPPORTED_RECONSTRUCTION_LOSSES)!r}; '
			f'got {reconstruction!r}'
		)
		raise ValueError(msg)

	if reconstruction == 'huber':
		_validate_required_key(loss, 'huber_delta', prefix='loss')
		_validate_positive_finite_number(loss, 'huber_delta', prefix='loss')
	elif 'huber_delta' in loss:
		msg = 'loss.huber_delta must be omitted unless loss.reconstruction is huber'
		raise ValueError(msg)

	_validate_required_key(loss, 'gradient_weight', prefix='loss')
	_validate_nonnegative_finite_number(loss, 'gradient_weight', prefix='loss')
	_validate_required_key(loss, 'visible_reconstruction_weight', prefix='loss')
	_validate_nonnegative_finite_number(
		loss,
		'visible_reconstruction_weight',
		prefix='loss',
	)
	_validate_loss_target_normalization(loss)
	if (
		'valid_mask_mode' in loss
		and loss.get('valid_mask_mode') != EXPECTED_VALID_MASK_MODE
	):
		msg = "loss.valid_mask_mode must be resolved internally as 'voxel'"
		raise ValueError(msg)


def _validate_loss_target_normalization(loss: Mapping[str, object]) -> None:
	target_normalization = _required_child_mapping(
		loss,
		'target_normalization',
		prefix='loss',
	)
	_validate_allowed_keys(
		target_normalization,
		frozenset({'mode', 'eps', 'min_std'}),
		prefix='loss.target_normalization',
	)
	_validate_required_key(
		target_normalization,
		'mode',
		prefix='loss.target_normalization',
	)
	mode = target_normalization.get('mode')
	if mode not in SUPPORTED_TARGET_NORMALIZATION_MODES:
		msg = (
			'loss.target_normalization.mode must be one of '
			f'{sorted(SUPPORTED_TARGET_NORMALIZATION_MODES)!r}; got {mode!r}'
		)
		raise ValueError(msg)
	if mode == 'none':
		for key in ('eps', 'min_std'):
			if key in target_normalization:
				msg = (
					f'loss.target_normalization.{key} must be omitted '
					"when mode is 'none'"
				)
				raise ValueError(msg)
		return
	_validate_required_key(
		target_normalization,
		'eps',
		prefix='loss.target_normalization',
	)
	_validate_required_key(
		target_normalization,
		'min_std',
		prefix='loss.target_normalization',
	)
	_validate_positive_finite_number(
		target_normalization,
		'eps',
		prefix='loss.target_normalization',
	)
	_validate_positive_finite_number(
		target_normalization,
		'min_std',
		prefix='loss.target_normalization',
	)
	if float(loss.get('gradient_weight', 0.0)) != 0.0:
		msg = (
			'loss.gradient_weight must be 0.0 when '
			"loss.target_normalization.mode is 'patch_zscore'; "
			'the current gradient loss operates in survey-normalized amplitude space'
		)
		raise ValueError(msg)


def _validate_zero_mask(zero_mask: Mapping[str, object]) -> None:
	if 'enabled' in zero_mask:
		_validate_bool(zero_mask, 'enabled', prefix='zero_mask')
	if 'zero_atol' in zero_mask:
		_validate_nonnegative_number(zero_mask, 'zero_atol', prefix='zero_mask')
	for key in ('z_sample_influence_radius', 'xy_trace_influence_radius'):
		if key in zero_mask:
			_validate_nonnegative_int(zero_mask, key, prefix='zero_mask')


def _validate_mae_training_visualization(
	visualization: Mapping[str, object],
	*,
	output_root: Path,
) -> None:
	_validate_allowed_keys(
		visualization,
		_MAE_TRAINING_VISUALIZATION_KEYS,
		prefix='visualization',
	)
	if 'mae_debug' not in visualization:
		return
	mae_debug = _required_child_mapping(
		visualization,
		'mae_debug',
		prefix='visualization',
	)
	_validate_allowed_keys(
		mae_debug,
		MAE_DEBUG_VISUALIZATION_KEYS,
		prefix='visualization.mae_debug',
	)
	_validate_mae_debug_general_fields(mae_debug, output_root=output_root)
	_validate_mae_debug_triggers(mae_debug)
	_validate_mae_debug_rendering_fields(mae_debug)


def _validate_mae_debug_general_fields(
	mae_debug: Mapping[str, object],
	*,
	output_root: Path,
) -> None:
	if 'enabled' in mae_debug:
		_validate_bool(mae_debug, 'enabled', prefix='visualization.mae_debug')
	if 'output_dir' in mae_debug:
		_validate_optional_output_path_under_root(
			mae_debug,
			'output_dir',
			prefix='visualization.mae_debug',
			root=output_root,
			root_label='paths.output_root',
		)


def _validate_mae_debug_triggers(mae_debug: Mapping[str, object]) -> None:
	for key in ('every_steps', 'every_epochs'):
		_validate_optional_positive_int(
			mae_debug,
			key,
			prefix='visualization.mae_debug',
		)
	if _mae_debug_enabled(mae_debug) and not _mae_debug_has_trigger(mae_debug):
		msg = (
			'visualization.mae_debug requires every_steps or every_epochs '
			'when enabled is true'
		)
		raise ValueError(msg)


def _validate_mae_debug_rendering_fields(mae_debug: Mapping[str, object]) -> None:
	if 'max_samples' in mae_debug:
		_validate_positive_int(
			mae_debug,
			'max_samples',
			prefix='visualization.mae_debug',
		)
	for key in ('xy_slice_index', 'xz_slice_y_index'):
		_validate_optional_nonnegative_int(
			mae_debug,
			key,
			prefix='visualization.mae_debug',
		)
	if 'dpi' in mae_debug:
		_validate_positive_int(mae_debug, 'dpi', prefix='visualization.mae_debug')
	if 'clip_percentiles' in mae_debug:
		_validate_mae_debug_clip_percentiles(mae_debug)
	if 'columns' in mae_debug:
		_validate_mae_debug_columns(mae_debug)
	for key in ('panel_width', 'panel_height'):
		if key in mae_debug:
			_validate_positive_finite_number(
				mae_debug,
				key,
				prefix='visualization.mae_debug',
			)
	if 'invalid_color' in mae_debug:
		_validate_non_empty_str(
			mae_debug,
			'invalid_color',
			prefix='visualization.mae_debug',
		)


def _mae_debug_enabled(mae_debug: Mapping[str, object]) -> bool:
	value = mae_debug.get(
		'enabled',
		DEFAULT_MAE_DEBUG_VISUALIZATION_OPTIONS['enabled'],
	)
	return bool(value)


def _mae_debug_has_trigger(mae_debug: Mapping[str, object]) -> bool:
	every_steps = mae_debug.get(
		'every_steps',
		DEFAULT_MAE_DEBUG_VISUALIZATION_OPTIONS['every_steps'],
	)
	every_epochs = mae_debug.get(
		'every_epochs',
		DEFAULT_MAE_DEBUG_VISUALIZATION_OPTIONS['every_epochs'],
	)
	return every_steps is not None or every_epochs is not None


def _validate_divisible_crop_patch(
	crop_size: Sequence[int],
	patch_size: Sequence[int],
) -> None:
	if any(
		crop % patch != 0
		for crop, patch in zip(crop_size, patch_size, strict=True)
	):
		msg = (
			'data.local_crop_size dimensions must be divisible by '
			'model.patch_size dimensions'
		)
		raise ValueError(msg)


def _merge_section_defaults(
	config: Config,
	section: str,
	defaults: Mapping[str, object],
) -> None:
	current = config.get(section)
	if current is None:
		config[section] = deepcopy(dict(defaults))
		return
	if not isinstance(current, dict):
		msg = f'{section} must be a mapping'
		raise TypeError(msg)
	config[section] = {**deepcopy(dict(defaults)), **current}


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _required_child_mapping(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{prefix}.{key} must be a mapping'
		raise TypeError(msg)
	return value


def _validate_non_empty_str(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)


def _validate_positive_int_triplet(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> tuple[int, int, int]:
	value = parent.get(key)
	if (
		not isinstance(value, list)
		or len(value) != 3
		or not all(_is_int(item) and int(item) > 0 for item in value)
	):
		msg = f'{prefix}.{key} must be a list of three positive integers'
		raise ValueError(msg)
	return (int(value[0]), int(value[1]), int(value[2]))


def _validate_nonnegative_int_triplet(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> tuple[int, int, int]:
	value = parent.get(key)
	if (
		not isinstance(value, list)
		or len(value) != 3
		or not all(_is_int(item) and int(item) >= 0 for item in value)
	):
		msg = f'{prefix}.{key} must be a list of three nonnegative integers'
		raise ValueError(msg)
	return (int(value[0]), int(value[1]), int(value[2]))


def _validate_overlap_less_than_window(
	overlap: Sequence[int],
	window_size: Sequence[int],
) -> None:
	if any(
		overlap_axis >= window_axis
		for overlap_axis, window_axis in zip(overlap, window_size, strict=True)
	):
		msg = (
			'embedding.overlap values must be less than embedding.window_size '
			f'values; got overlap={list(overlap)!r}, '
			f'window_size={list(window_size)!r}'
		)
		raise ValueError(msg)


def _validate_embedding_output_dtype(embedding: Mapping[str, object]) -> None:
	value = embedding.get('output_dtype')
	if not isinstance(value, str) or not value:
		msg = f'embedding.output_dtype must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	if value not in {'float16', 'float32'}:
		msg = (
			'embedding.output_dtype must be "float16" or "float32"; '
			f'got {value!r}'
		)
		raise ValueError(msg)


def _validate_positive_int_list(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if (
		not isinstance(value, list)
		or not value
		or not all(_is_int(item) and int(item) > 0 for item in value)
	):
		msg = f'{prefix}.{key} must be a non-empty list of positive integers'
		raise ValueError(msg)


def _validate_unique_positive_int_list(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	_validate_positive_int_list(parent, key, prefix=prefix)
	value = parent.get(key)
	if not isinstance(value, list):
		msg = f'{prefix}.{key} must be a non-empty list of positive integers'
		raise TypeError(msg)
	values = [int(item) for item in value]
	if len(set(values)) != len(values):
		msg = f'{prefix}.{key} must not contain duplicates; got {values!r}'
		raise ValueError(msg)


def _validate_nonnegative_int_list(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if (
		not isinstance(value, list)
		or any(not _is_int(item) or int(item) < 0 for item in value)
	):
		msg = f'{prefix}.{key} must be a list of nonnegative integers'
		raise ValueError(msg)


def _validate_positive_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not _is_int(value) or int(value) <= 0:
		msg = f'{prefix}.{key} must be a positive integer; got {value!r}'
		raise ValueError(msg)


def _validate_required_key(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	if key not in parent:
		msg = f'{prefix}.{key} is required'
		raise ValueError(msg)


def _validate_nonnegative_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not _is_int(value) or int(value) < 0:
		msg = f'{prefix}.{key} must be a nonnegative integer; got {value!r}'
		raise ValueError(msg)


def _validate_optional_positive_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	if key not in parent or parent.get(key) is None:
		return
	_validate_positive_int(parent, key, prefix=prefix)


def _validate_optional_nonnegative_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	if key not in parent or parent.get(key) is None:
		return
	_validate_nonnegative_int(parent, key, prefix=prefix)


def _validate_positive_number(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not _is_number(value) or float(value) <= 0.0:
		msg = f'{prefix}.{key} must be positive; got {value!r}'
		raise ValueError(msg)


def _validate_nonnegative_number(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not _is_number(value) or float(value) < 0.0:
		msg = f'{prefix}.{key} must be nonnegative; got {value!r}'
		raise ValueError(msg)


def _validate_positive_finite_number(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if (
		not _is_number(value)
		or float(value) <= 0.0
		or not math.isfinite(float(value))
	):
		msg = f'{prefix}.{key} must be a finite positive number; got {value!r}'
		raise ValueError(msg)


def _validate_nonnegative_finite_number(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if (
		not _is_number(value)
		or float(value) < 0.0
		or not math.isfinite(float(value))
	):
		msg = f'{prefix}.{key} must be a nonnegative finite number; got {value!r}'
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


def _validate_mae_debug_clip_percentiles(
	mae_debug: Mapping[str, object],
) -> None:
	value = mae_debug.get('clip_percentiles')
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 2
	):
		msg = (
			'visualization.mae_debug.clip_percentiles must contain two '
			f'finite values; got {value!r}'
		)
		raise ValueError(msg)
	low, high = value
	if not _is_number(low) or not _is_number(high):
		msg = (
			'visualization.mae_debug.clip_percentiles must contain numeric '
			f'values; got {value!r}'
		)
		raise ValueError(msg)
	low_float = float(low)
	high_float = float(high)
	if (
		not math.isfinite(low_float)
		or not math.isfinite(high_float)
		or not 0.0 <= low_float < high_float <= 100.0
	):
		msg = (
			'visualization.mae_debug.clip_percentiles must satisfy '
			f'0 <= low < high <= 100; got {value!r}'
		)
		raise ValueError(msg)


def _validate_mae_debug_columns(mae_debug: Mapping[str, object]) -> None:
	value = mae_debug.get('columns')
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or not value
		or any(not isinstance(item, str) or not item for item in value)
	):
		msg = (
			'visualization.mae_debug.columns must be a non-empty sequence '
			f'of strings; got {value!r}'
		)
		raise ValueError(msg)
	if len(set(value)) != len(value):
		msg = (
			'visualization.mae_debug.columns must not contain duplicates; '
			f'got {list(value)!r}'
		)
		raise ValueError(msg)
	unknown = sorted(set(value) - MAE_DEBUG_VISUALIZATION_COLUMNS)
	if unknown:
		msg = (
			'visualization.mae_debug.columns contains unsupported column(s): '
			f'{unknown!r}; allowed columns are '
			f'{sorted(MAE_DEBUG_VISUALIZATION_COLUMNS)!r}'
		)
		raise ValueError(msg)


def _validate_slice_coordinate_space(visualization: Mapping[str, object]) -> None:
	value = visualization.get('slice_coordinate_space')
	if value != 'voxel':
		msg = (
			'visualization.slice_coordinate_space must be "voxel"; '
			f'got {value!r}'
		)
		raise ValueError(msg)


def _validate_optional_fraction(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	if key in parent:
		_validate_fraction(parent, key, prefix=prefix)


def _validate_fraction(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if (
		not _is_number(value)
		or float(value) < 0.0
		or float(value) > 1.0
		or not math.isfinite(float(value))
	):
		msg = f'{prefix}.{key} must be between 0 and 1; got {value!r}'
		raise ValueError(msg)


def _validate_bool(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not isinstance(value, bool):
		msg = f'{prefix}.{key} must be a boolean; got {value!r}'
		raise TypeError(msg)


def _is_int(value: object) -> bool:
	return isinstance(value, Integral) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
	return isinstance(value, Real) and not isinstance(value, bool)


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
	'resolve_mae_training_config',
	'resolve_manifest_build_config',
	'resolve_normalization_qc_config',
	'resolve_normalization_stats_config',
	'validate_config',
]
