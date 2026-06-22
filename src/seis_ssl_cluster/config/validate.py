"""Stage-specific validation and resolution for amplitude-only configs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from numbers import Integral, Real
from pathlib import Path
from typing import TypeAlias, TypeVar

from seis_ssl_cluster.config.schema import (
	DEFAULT_ZERO_MASK_CONTRACT,
	EXPECTED_RECONSTRUCTION_LOSS,
	EXPECTED_VALID_MASK_MODE,
	FIXED_DATA_CONTRACT,
	FIXED_LOSS_CONTRACT,
	FIXED_MASKING_CONTRACT,
	FIXED_MODEL_CONTRACT,
	KNOWN_STAGES,
	LEGACY_ATTRIBUTE_KEY_NAMES,
	LEGACY_ATTRIBUTE_KEY_PATHS,
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
		{'paths', 'manifests', 'embeddings'},
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

_DEFAULT_EMBEDDING_LOCAL_CROP_SIZE = [128, 128, 128]


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
	if 'output_name' in manifest:
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
	_validate_optional_positive_number(qc, 'min_iqr', prefix='qc')
	_validate_optional_positive_number(qc, 'max_normalized_abs', prefix='qc')
	return resolved


def resolve_mae_training_config(config: _T) -> Config:
	"""Validate and resolve raw config for MAE training."""
	resolved, _paths = _resolve_base(config, STAGE_MAE_TRAINING)
	_reject_fixed_contract_keys(resolved)

	manifests = _required_mapping(resolved, 'manifests')
	_validate_non_empty_path(manifests, 'train', prefix='manifests')
	if 'train_path_list' in manifests:
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
	if 'zero_mask' in resolved:
		_validate_zero_mask(_required_mapping(resolved, 'zero_mask'))
	if 'visualization' in resolved:
		_required_mapping(resolved, 'visualization')

	_merge_section_defaults(resolved, 'data', FIXED_DATA_CONTRACT)
	_merge_section_defaults(resolved, 'model', FIXED_MODEL_CONTRACT)
	_merge_section_defaults(resolved, 'masking', FIXED_MASKING_CONTRACT)
	_merge_section_defaults(resolved, 'loss', FIXED_LOSS_CONTRACT)
	_merge_section_defaults(resolved, 'zero_mask', DEFAULT_ZERO_MASK_CONTRACT)
	return resolved


def resolve_embedding_extraction_config(config: _T) -> Config:
	"""Validate and resolve raw config for embedding extraction."""
	resolved, _paths = _resolve_base(config, STAGE_EMBEDDING_EXTRACTION)
	manifests = _required_mapping(resolved, 'manifests')
	embeddings = _required_mapping(resolved, 'embeddings')
	_validate_non_empty_path(manifests, 'input', prefix='manifests')
	_validate_non_empty_path(embeddings, 'checkpoint', prefix='embeddings')
	_validate_non_empty_path(embeddings, 'output_dir', prefix='embeddings')

	embedding = _optional_mapping(resolved, 'embedding')
	if 'window_size' in embedding:
		_validate_positive_int_triplet(embedding, 'window_size', prefix='embedding')
	if 'overlap' in embedding:
		_validate_nonnegative_int_triplet(embedding, 'overlap', prefix='embedding')
	if 'batch_size' in embedding:
		_validate_positive_int(embedding, 'batch_size', prefix='embedding')
	if 'min_token_valid_fraction' in embedding:
		_validate_fraction(
			embedding,
			'min_token_valid_fraction',
			prefix='embedding',
		)
	if 'output_dtype' in embedding:
		_validate_non_empty_str(embedding, 'output_dtype', prefix='embedding')

	window_size = embedding.get('window_size', _DEFAULT_EMBEDDING_LOCAL_CROP_SIZE)
	resolved['data'] = {
		**deepcopy(FIXED_DATA_CONTRACT),
		'local_crop_size': deepcopy(window_size),
	}
	return resolved


def resolve_clustering_config(config: _T) -> Config:
	"""Validate and resolve raw config for embedding clustering."""
	resolved, _paths = _resolve_base(config, STAGE_CLUSTERING)
	embeddings = _required_mapping(resolved, 'embeddings')
	clustering = _required_mapping(resolved, 'clustering')
	_validate_non_empty_path(embeddings, 'input_dir', prefix='embeddings')
	_validate_non_empty_path(clustering, 'output_dir', prefix='clustering')
	if 'k_values' in clustering:
		_validate_positive_int_list(clustering, 'k_values', prefix='clustering')
	if 'sample_tokens' in clustering:
		_validate_positive_int(clustering, 'sample_tokens', prefix='clustering')
	if 'minibatch_size' in clustering:
		_validate_positive_int(clustering, 'minibatch_size', prefix='clustering')
	if 'prediction_batch_size' in clustering:
		_validate_positive_int(
			clustering,
			'prediction_batch_size',
			prefix='clustering',
		)
	if 'seed' in clustering and not _is_int(clustering.get('seed')):
		msg = f'clustering.seed must be an integer; got {clustering.get("seed")!r}'
		raise ValueError(msg)
	return resolved


def resolve_cluster_visualization_config(config: _T) -> Config:
	"""Validate and resolve raw config for cluster visualization."""
	resolved, _paths = _resolve_base(config, STAGE_CLUSTER_VISUALIZATION)
	clustering = _required_mapping(resolved, 'clustering')
	visualization = _required_mapping(resolved, 'visualization')
	_validate_non_empty_path(clustering, 'input_dir', prefix='clustering')
	_validate_non_empty_path(visualization, 'output_dir', prefix='visualization')
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
) -> tuple[Config, _ResolvedPaths]:
	_validate_mapping(config)
	_reject_legacy_attribute_config(config)
	_reject_stage_key(config)
	_validate_top_level_sections(config, stage)
	resolved = deepcopy(dict(config))
	resolved['stage'] = stage
	paths = _validate_paths(_required_mapping(resolved, 'paths'))
	return resolved, paths


class _ResolvedPaths:
	def __init__(self, *, nopims_root: Path, artifact_root: Path) -> None:
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


def _reject_legacy_attribute_config(config: Mapping[str, object]) -> None:
	for path, key in _iter_mapping_keys(config):
		if path in LEGACY_ATTRIBUTE_KEY_PATHS or key in LEGACY_ATTRIBUTE_KEY_NAMES:
			msg = (
				f'{path} is a legacy multi-attribute config key and is not '
				'valid for the amplitude-only MVP; remove fixed-attribute '
				'configuration from this config.'
			)
			raise ValueError(msg)


def _validate_paths(paths: Mapping[str, object]) -> _ResolvedPaths:
	return _ResolvedPaths(
		nopims_root=_validate_absolute_path(paths, 'nopims_root', prefix='paths'),
		artifact_root=_validate_absolute_path(paths, 'artifact_root', prefix='paths'),
	)


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
	nopims_root: Path,
) -> None:
	if not path.is_absolute():
		msg = f'{label} must be an absolute artifact-registry path; got {path}'
		raise ValueError(msg)
	if _is_relative_to(path, nopims_root):
		msg = f'{label} must not be under paths.nopims_root; got {path}'
		raise ValueError(msg)
	if not _is_relative_to(path, artifact_root):
		msg = f'{label} must be under paths.artifact_root ({artifact_root}); got {path}'
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
	if 'clipping_percentiles' in normalization:
		value = normalization.get('clipping_percentiles')
		if (
			not isinstance(value, list)
			or len(value) != 2
			or not all(_is_number(item) for item in value)
			or float(value[0]) >= float(value[1])
		):
			msg = 'normalization.clipping_percentiles must be two increasing numbers'
			raise ValueError(msg)
	for key in ('epsilon', 'max_samples'):
		if key not in normalization:
			continue
		if key == 'epsilon':
			_validate_positive_number(normalization, key, prefix='normalization')
		else:
			_validate_positive_int(normalization, key, prefix='normalization')
	if 'seed' in normalization and not _is_int(normalization.get('seed')):
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
	if (
		'reconstruction' in loss
		and loss.get('reconstruction') != EXPECTED_RECONSTRUCTION_LOSS
	):
		msg = "loss.reconstruction must be resolved internally as 'huber'"
		raise ValueError(msg)
	if 'huber_delta' in loss:
		_validate_positive_number(loss, 'huber_delta', prefix='loss')
	if 'gradient_weight' in loss:
		_validate_nonnegative_number(loss, 'gradient_weight', prefix='loss')
	if (
		'valid_mask_mode' in loss
		and loss.get('valid_mask_mode') != EXPECTED_VALID_MASK_MODE
	):
		msg = "loss.valid_mask_mode must be resolved internally as 'voxel'"
		raise ValueError(msg)


def _validate_zero_mask(zero_mask: Mapping[str, object]) -> None:
	if 'enabled' in zero_mask:
		_validate_bool(zero_mask, 'enabled', prefix='zero_mask')
	if 'zero_atol' in zero_mask:
		_validate_nonnegative_number(zero_mask, 'zero_atol', prefix='zero_mask')
	for key in ('z_sample_influence_radius', 'xy_trace_influence_radius'):
		if key in zero_mask:
			_validate_nonnegative_int(zero_mask, key, prefix='zero_mask')


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


def _optional_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
	value = parent.get(key, {})
	if value is None:
		return {}
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
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


def _validate_optional_positive_number(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	if key in parent:
		_validate_positive_number(parent, key, prefix=prefix)


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
	if not _is_number(value) or float(value) < 0.0 or float(value) > 1.0:
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
