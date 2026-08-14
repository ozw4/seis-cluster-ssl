"""Configuration contracts for the paired F3 voxel split robustness suite."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from seis_ssl_cluster.config.f3_lithology_common import (
	_max_file_size_bytes,
	_optional_positive_int,
	_publish_optional_bool,
)
from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	VoxelDecoderSpec,
	VoxelDecoderTileSettings,
	VoxelDecoderTrainSettings,
	_factor_sequence,
	_finite_nonnegative_float,
	_finite_positive_float,
	_positive_sequence,
	_triplet,
)
from seis_ssl_cluster.models.voxel_decoder.spec import (
	VOXEL_DECODER_NORMALIZATION,
	VOXEL_DECODER_SPEC,
	VOXEL_DECODER_UPSAMPLE_MODE,
	validate_voxel_decoder_implementation,
)

DEFAULT_REPORTS_ROOT = Path('reports')
_DEFAULT_PUBLISH_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
BASELINE_MODEL_TAG = 'strat_hmm_pretext_m1_k6_topblock1_distill'
CANDIDATE_MODEL_TAG = 'strat_hmm_pretext_m2a_boundary_a050_t2_k6_topblock1_distill'
MODEL_ROLES = ('baseline', 'candidate')


@dataclass(frozen=True)
class VoxelRobustnessModel:
	"""One frozen encoder in the paired suite."""

	role: str
	model_tag: str
	embeddings_dir: Path
	checkpoint: Path | None = None


@dataclass(frozen=True)
class F3VoxelSplitDatasetSuiteConfig:
	"""Inputs shared by all split-specific voxel supervision builds."""

	split_inventory_manifest: Path
	output_root: Path
	artifact_root: Path
	f3_root: Path
	dataset: Mapping[str, str]
	source_label_volume: Path
	source_label_segy: Path
	class_info: Path
	segy_geometry_json: Path
	reference_metadata_json: Path
	reference_valid_tokens: Path
	ignore_z_border_samples: int
	overwrite: bool

	def __post_init__(self) -> None:
		"""Keep generated split supervision under the artifact root."""
		_validate_suite_output_root(
			self.output_root,
			f3_root=self.f3_root,
			label='suite.output_root',
		)


@dataclass(frozen=True)
class F3VoxelV0SplitSuiteConfig:
	"""Resolved inputs for paired full-token V0 projection jobs."""

	voxel_dataset_manifest: Path
	split_dataset_manifest: Path
	probe_run_manifest: Path
	output_root: Path
	artifact_root: Path
	f3_root: Path
	dataset: Mapping[str, str]
	models: tuple[VoxelRobustnessModel, ...]
	source_label_volume: Path
	source_label_segy: Path
	class_info: Path
	segy_geometry_json: Path
	batch_size: int
	tokenization: Mapping[str, object]
	evaluation: Mapping[str, object]
	overwrite: bool

	def __post_init__(self) -> None:
		"""Keep generated V0 artifacts under the artifact root."""
		_validate_suite_output_root(
			self.output_root,
			f3_root=self.f3_root,
			label='suite.output_root',
		)


@dataclass(frozen=True)
class F3VoxelDecoderSplitSuiteConfig:
	"""Resolved inputs and common settings for paired V1 decoder jobs."""

	voxel_dataset_manifest: Path
	output_root: Path
	artifact_root: Path
	f3_root: Path
	dataset: Mapping[str, str]
	models: tuple[VoxelRobustnessModel, ...]
	source_label_volume: Path
	source_label_segy: Path
	class_info: Path
	segy_geometry_json: Path
	decoder: VoxelDecoderSpec
	tiles: VoxelDecoderTileSettings
	train: VoxelDecoderTrainSettings
	evaluation: Mapping[str, object]
	write_probabilities: bool
	overwrite: bool

	def __post_init__(self) -> None:
		"""Keep generated decoder artifacts under the artifact root."""
		_validate_suite_output_root(
			self.output_root,
			f3_root=self.f3_root,
			label='suite.output_root',
		)


@dataclass(frozen=True)
class F3VoxelSplitRobustnessPublishConfig:
	"""Final lightweight publication settings for original and split evidence."""

	enabled: bool = False
	reports_root: Path = DEFAULT_REPORTS_ROOT
	output_dir: Path | None = None
	max_file_size_bytes: int = _DEFAULT_PUBLISH_MAX_FILE_SIZE_BYTES
	overwrite: bool = True

	def __post_init__(self) -> None:
		"""Validate publication settings."""
		if self.enabled and self.output_dir is None:
			raise ValueError(
				'publish.output_dir is required when publishing is enabled'
			)


@dataclass(frozen=True)
class F3VoxelSplitRobustnessSummaryConfig:
	"""Inputs for split-level paired V0/V1 aggregation."""

	suite_root: Path
	v0_run_manifest: Path
	v1_run_manifest: Path
	baseline_model_tag: str
	candidate_model_tag: str
	artifact_root: Path | None = None
	f3_root: Path | None = None
	original_summary_dir: Path | None = None
	publish: F3VoxelSplitRobustnessPublishConfig = field(
		default_factory=F3VoxelSplitRobustnessPublishConfig
	)

	def __post_init__(self) -> None:
		"""Validate the report root when its configured roots are available."""
		if (self.artifact_root is None) != (self.f3_root is None):
			raise ValueError('artifact_root and f3_root must be provided together')
		if self.artifact_root is not None and self.f3_root is not None:
			_validate_suite_output_root(
				self.suite_root,
				f3_root=self.f3_root,
				label='suite.root',
			)


def f3_lithology_voxel_split_dataset_config_from_mapping(
	config: Mapping[str, object],
) -> F3VoxelSplitDatasetSuiteConfig:
	"""Resolve the split-specific voxel supervision builder config."""
	_exact_keys(
		config,
		{
			'suite',
			'paths',
			'dataset',
			'labels',
			'reference_embedding',
			'voxel_dataset',
			'outputs',
		},
		'config',
	)
	suite = _mapping(config, 'suite')
	paths = _mapping(config, 'paths')
	dataset = _dataset(config)
	labels = _labels(config)
	reference = _mapping(config, 'reference_embedding')
	voxel = _mapping(config, 'voxel_dataset')
	outputs = _mapping(config, 'outputs')
	_exact_keys(suite, {'split_inventory_manifest', 'output_root'}, 'suite')
	_exact_keys(paths, {'artifact_root', 'f3_root'}, 'paths')
	_exact_keys(reference, {'metadata_json', 'valid_tokens'}, 'reference_embedding')
	_exact_keys(voxel, {'ignore_z_border_samples'}, 'voxel_dataset')
	_exact_keys(outputs, {'overwrite'}, 'outputs')
	ignore = _integer(voxel, 'ignore_z_border_samples', minimum=0)
	return F3VoxelSplitDatasetSuiteConfig(
		split_inventory_manifest=_path(suite, 'split_inventory_manifest'),
		output_root=_path(suite, 'output_root'),
		artifact_root=_path(paths, 'artifact_root'),
		f3_root=_path(paths, 'f3_root'),
		dataset=dataset,
		source_label_volume=_path(labels, 'source_label_volume'),
		source_label_segy=_path(labels, 'source_label_segy'),
		class_info=_path(labels, 'class_info'),
		segy_geometry_json=_path(labels, 'segy_geometry_json'),
		reference_metadata_json=_path(reference, 'metadata_json'),
		reference_valid_tokens=_path(reference, 'valid_tokens'),
		ignore_z_border_samples=ignore,
		overwrite=_boolean(outputs, 'overwrite'),
	)


def f3_lithology_voxel_v0_split_suite_config_from_mapping(
	config: Mapping[str, object],
) -> F3VoxelV0SplitSuiteConfig:
	"""Resolve the paired V0 split runner config."""
	_exact_keys(
		config,
		{
			'suite',
			'paths',
			'dataset',
			'models',
			'labels',
			'predictions',
			'evaluation',
			'outputs',
		},
		'config',
	)
	suite = _mapping(config, 'suite')
	paths = _mapping(config, 'paths')
	predictions = _mapping(config, 'predictions')
	evaluation = _evaluation(config)
	outputs = _mapping(config, 'outputs')
	labels = _labels(config)
	_exact_keys(
		suite,
		{
			'voxel_dataset_manifest',
			'split_dataset_manifest',
			'probe_run_manifest',
			'output_root',
		},
		'suite',
	)
	_exact_keys(paths, {'artifact_root', 'f3_root'}, 'paths')
	_exact_keys(predictions, {'batch_size', 'tokenization'}, 'predictions')
	_exact_keys(outputs, {'overwrite'}, 'outputs')
	return F3VoxelV0SplitSuiteConfig(
		voxel_dataset_manifest=_path(suite, 'voxel_dataset_manifest'),
		split_dataset_manifest=_path(suite, 'split_dataset_manifest'),
		probe_run_manifest=_path(suite, 'probe_run_manifest'),
		output_root=_path(suite, 'output_root'),
		artifact_root=_path(paths, 'artifact_root'),
		f3_root=_path(paths, 'f3_root'),
		dataset=_dataset(config),
		models=_models(config, require_checkpoint=True),
		source_label_volume=_path(labels, 'source_label_volume'),
		source_label_segy=_path(labels, 'source_label_segy'),
		class_info=_path(labels, 'class_info'),
		segy_geometry_json=_path(labels, 'segy_geometry_json'),
		batch_size=_integer(predictions, 'batch_size', minimum=1),
		tokenization=_tokenization(predictions),
		evaluation=evaluation,
		overwrite=_boolean(outputs, 'overwrite'),
	)


def f3_lithology_voxel_decoder_split_suite_config_from_mapping(
	config: Mapping[str, object],
) -> F3VoxelDecoderSplitSuiteConfig:
	"""Resolve the paired V1 decoder split runner config."""
	_exact_keys(
		config,
		{
			'suite',
			'paths',
			'dataset',
			'models',
			'labels',
			'decoder',
			'tiles',
			'train',
			'inference',
			'evaluation',
			'outputs',
		},
		'config',
	)
	suite = _mapping(config, 'suite')
	paths = _mapping(config, 'paths')
	labels = _labels(config)
	decoder = _mapping(config, 'decoder')
	tiles = _mapping(config, 'tiles')
	train = _mapping(config, 'train')
	inference = _mapping(config, 'inference')
	outputs = _mapping(config, 'outputs')
	_exact_keys(suite, {'voxel_dataset_manifest', 'output_root'}, 'suite')
	_exact_keys(paths, {'artifact_root', 'f3_root'}, 'paths')
	_exact_keys(inference, {'write_probabilities'}, 'inference')
	_exact_keys(outputs, {'overwrite'}, 'outputs')
	return F3VoxelDecoderSplitSuiteConfig(
		voxel_dataset_manifest=_path(suite, 'voxel_dataset_manifest'),
		output_root=_path(suite, 'output_root'),
		artifact_root=_path(paths, 'artifact_root'),
		f3_root=_path(paths, 'f3_root'),
		dataset=_dataset(config),
		models=_models(config, require_checkpoint=False),
		source_label_volume=_path(labels, 'source_label_volume'),
		source_label_segy=_path(labels, 'source_label_segy'),
		class_info=_path(labels, 'class_info'),
		segy_geometry_json=_path(labels, 'segy_geometry_json'),
		decoder=_decoder_settings(decoder),
		tiles=_tiles_settings(tiles),
		train=_training_settings(train),
		evaluation=_evaluation(config),
		write_probabilities=_boolean(inference, 'write_probabilities'),
		overwrite=_boolean(outputs, 'overwrite'),
	)


def f3_lithology_voxel_split_summary_config_from_mapping(
	config: Mapping[str, object],
) -> F3VoxelSplitRobustnessSummaryConfig:
	"""Resolve split-level robustness summary inputs."""
	_exact_keys(config, {'suite', 'paths', 'inputs', 'models', 'publish'}, 'config')
	suite = _mapping(config, 'suite')
	paths = _mapping(config, 'paths')
	inputs = _mapping(config, 'inputs')
	models = _mapping(config, 'models')
	publish = _mapping(config, 'publish')
	_exact_keys(suite, {'root'}, 'suite')
	_exact_keys(paths, {'artifact_root', 'f3_root', 'reports_root'}, 'paths')
	_exact_keys(
		inputs,
		{'v0_run_manifest', 'v1_run_manifest', 'original_summary_dir'},
		'inputs',
	)
	_exact_keys(models, set(MODEL_ROLES), 'models')
	_exact_keys(
		publish,
		{'enabled', 'output_dir', 'max_file_size_mb', 'overwrite'},
		'publish',
	)
	baseline = _string(models, 'baseline')
	candidate = _string(models, 'candidate')
	_validate_model_tags(baseline, candidate)
	reports_root = Path(_string(paths, 'reports_root'))
	publish_output = Path(_string(publish, 'output_dir'))
	return F3VoxelSplitRobustnessSummaryConfig(
		suite_root=_path(suite, 'root'),
		v0_run_manifest=_path(inputs, 'v0_run_manifest'),
		v1_run_manifest=_path(inputs, 'v1_run_manifest'),
		baseline_model_tag=baseline,
		candidate_model_tag=candidate,
		artifact_root=_path(paths, 'artifact_root'),
		f3_root=_path(paths, 'f3_root'),
		original_summary_dir=_path(inputs, 'original_summary_dir'),
		publish=F3VoxelSplitRobustnessPublishConfig(
			enabled=_publish_optional_bool(publish, 'enabled', default=False),
			reports_root=reports_root,
			output_dir=publish_output,
			max_file_size_bytes=_max_file_size_bytes(publish),
			overwrite=_boolean(publish, 'overwrite'),
		),
	)


def _dataset(config: Mapping[str, object]) -> Mapping[str, str]:
	dataset = _mapping(config, 'dataset')
	_exact_keys(dataset, {'name', 'version'}, 'dataset')
	return {'name': _string(dataset, 'name'), 'version': _string(dataset, 'version')}


def _labels(config: Mapping[str, object]) -> Mapping[str, object]:
	labels = _mapping(config, 'labels')
	_exact_keys(
		labels,
		{
			'source_label_volume',
			'source_label_segy',
			'class_info',
			'segy_geometry_json',
		},
		'labels',
	)
	return labels


def _evaluation(config: Mapping[str, object]) -> Mapping[str, object]:
	evaluation = _mapping(config, 'evaluation')
	_exact_keys(
		evaluation,
		{
			'monitored_class_ids',
			'boundary_tolerances',
			'boundary_region_radii',
			'chunk_size_x',
		},
		'evaluation',
	)
	resolved: dict[str, object] = {}
	for key in ('monitored_class_ids', 'boundary_tolerances', 'boundary_region_radii'):
		value = evaluation.get(key)
		if not isinstance(value, Sequence) or isinstance(value, str | bytes):
			raise TypeError(f'evaluation.{key} must be a list')
		items = tuple(
			_integer_item(item, f'evaluation.{key}', minimum=0) for item in value
		)
		if not items or len(set(items)) != len(items):
			raise ValueError(f'evaluation.{key} must be non-empty and unique')
		resolved[key] = items
	if not {3, 5}.issubset(cast('tuple[int, ...]', resolved['monitored_class_ids'])):
		raise ValueError('evaluation.monitored_class_ids must include classes 3 and 5')
	if not {2, 4}.issubset(cast('tuple[int, ...]', resolved['boundary_tolerances'])):
		raise ValueError('evaluation.boundary_tolerances must include 2 and 4')
	if not {2, 4}.issubset(cast('tuple[int, ...]', resolved['boundary_region_radii'])):
		raise ValueError('evaluation.boundary_region_radii must include 2 and 4')
	resolved['chunk_size_x'] = _integer(evaluation, 'chunk_size_x', minimum=1)
	return resolved


def _tokenization(predictions: Mapping[str, object]) -> Mapping[str, object]:
	value = _mapping(predictions, 'tokenization')
	_exact_keys(
		value,
		{
			'min_labeled_fraction',
			'min_majority_fraction',
			'ignore_z_border_samples',
		},
		'predictions.tokenization',
	)
	return {
		'min_labeled_fraction': _fraction(value, 'min_labeled_fraction'),
		'min_majority_fraction': _fraction(value, 'min_majority_fraction'),
		'ignore_z_border_samples': _integer(
			value, 'ignore_z_border_samples', minimum=0
		),
	}


def _decoder_settings(value: Mapping[str, object]) -> VoxelDecoderSpec:
	_exact_keys(
		value,
		{
			'spec',
			'embedding_dim',
			'class_count',
			'hidden_channels',
			'upsample_factors',
			'upsample_mode',
			'normalization',
		},
		'decoder',
	)
	validate_voxel_decoder_implementation(
		spec=value.get('spec'),
		upsample_mode=value.get('upsample_mode'),
		normalization=value.get('normalization'),
		field_prefix='decoder',
	)
	return VoxelDecoderSpec(
		spec=VOXEL_DECODER_SPEC,
		embedding_dim=_optional_positive_int(
			value.get('embedding_dim'), 'decoder.embedding_dim'
		),
		class_count=_optional_positive_int(
			value.get('class_count'), 'decoder.class_count'
		),
		hidden_channels=_positive_sequence(
			value.get('hidden_channels'), 'decoder.hidden_channels'
		),
		upsample_factors=_factor_sequence(value.get('upsample_factors')),
		upsample_mode=VOXEL_DECODER_UPSAMPLE_MODE,
		normalization=VOXEL_DECODER_NORMALIZATION,
	)


def _tiles_settings(value: Mapping[str, object]) -> VoxelDecoderTileSettings:
	_exact_keys(value, {'core_size_tokens', 'context_halo_tokens'}, 'tiles')
	return VoxelDecoderTileSettings(
		core_size_tokens=_triplet(
			value.get('core_size_tokens'), 'tiles.core_size_tokens', positive=True
		),
		context_halo_tokens=_triplet(
			value.get('context_halo_tokens'),
			'tiles.context_halo_tokens',
			positive=False,
		),
	)


def _training_settings(value: Mapping[str, object]) -> VoxelDecoderTrainSettings:
	_exact_keys(
		value,
		{
			'epochs',
			'batch_size',
			'learning_rate',
			'weight_decay',
			'class_weight',
			'seed',
			'num_workers',
			'amp',
			'gradient_clip_norm',
		},
		'train',
	)
	class_weight = _string(value, 'class_weight')
	if class_weight != 'balanced':
		raise ValueError("train.class_weight must be 'balanced'")
	seed = _integer(value, 'seed', minimum=0)
	num_workers = _integer(value, 'num_workers', minimum=0)
	amp = _boolean(value, 'amp')
	return VoxelDecoderTrainSettings(
		epochs=_optional_positive_int(value.get('epochs'), 'train.epochs'),
		batch_size=_optional_positive_int(value.get('batch_size'), 'train.batch_size'),
		learning_rate=_finite_positive_float(
			value.get('learning_rate'), 'train.learning_rate'
		),
		weight_decay=_finite_nonnegative_float(
			value.get('weight_decay'), 'train.weight_decay'
		),
		class_weight=class_weight,
		seed=seed,
		num_workers=num_workers,
		amp=amp,
		gradient_clip_norm=_finite_positive_float(
			value.get('gradient_clip_norm'), 'train.gradient_clip_norm'
		),
	)


def _models(
	config: Mapping[str, object], *, require_checkpoint: bool
) -> tuple[VoxelRobustnessModel, ...]:
	models = _mapping(config, 'models')
	_exact_keys(models, set(MODEL_ROLES), 'models')
	result = []
	for role in MODEL_ROLES:
		item = _mapping(models, role)
		allowed = {'model_tag', 'embeddings_dir'} | (
			{'checkpoint'} if require_checkpoint else set()
		)
		_exact_keys(item, allowed, f'models.{role}')
		result.append(
			VoxelRobustnessModel(
				role=role,
				model_tag=_string(item, 'model_tag'),
				embeddings_dir=_path(item, 'embeddings_dir'),
				checkpoint=_path(item, 'checkpoint') if require_checkpoint else None,
			)
		)
	_validate_model_tags(result[0].model_tag, result[1].model_tag)
	return tuple(result)


def _validate_model_tags(baseline: str, candidate: str) -> None:
	if baseline != BASELINE_MODEL_TAG or candidate != CANDIDATE_MODEL_TAG:
		raise ValueError(
			'M1/M2-A voxel robustness requires the fixed '
			'baseline/candidate model pair; '
			f'got baseline={baseline!r}, candidate={candidate!r}'
		)


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'{key} must be a mapping; got {value!r}')
	return value


def _string(parent: Mapping[str, object], key: str) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value.strip():
		raise TypeError(f'{key} must be a non-empty string; got {value!r}')
	return value


def _path(parent: Mapping[str, object], key: str) -> Path:
	path = Path(_string(parent, key))
	if not path.is_absolute():
		raise ValueError(f'{key} must be an absolute path; got {path}')
	return path


def _validate_suite_output_root(
	path: Path,
	*,
	f3_root: Path,
	label: str,
) -> None:
	output = Path(path)
	if not output.is_absolute():
		raise ValueError(f'{label} must be an absolute path; got {output}')
	resolved = output.resolve(strict=False)
	resolved_f3 = Path(f3_root).resolve(strict=False)
	try:
		resolved.relative_to(resolved_f3)
	except ValueError:
		return
	raise ValueError(f'{label} must be outside f3_root ({resolved_f3}); got {resolved}')


def _boolean(parent: Mapping[str, object], key: str) -> bool:
	value = parent.get(key)
	if not isinstance(value, bool):
		raise TypeError(f'{key} must be boolean; got {value!r}')
	return value


def _integer(parent: Mapping[str, object], key: str, *, minimum: int) -> int:
	value = parent.get(key)
	if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
		raise ValueError(f'{key} must be an integer >= {minimum}; got {value!r}')
	return value


def _integer_item(value: object, label: str, *, minimum: int) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
		raise ValueError(f'{label} values must be integers >= {minimum}; got {value!r}')
	return value


def _fraction(parent: Mapping[str, object], key: str) -> float:
	value = parent.get(key)
	if isinstance(value, bool) or not isinstance(value, int | float):
		raise TypeError(f'{key} must be numeric; got {value!r}')
	result = float(value)
	if not 0.0 <= result <= 1.0:
		raise ValueError(f'{key} must be in [0, 1]; got {value!r}')
	return result


def _exact_keys(value: Mapping[str, object], allowed: set[str], label: str) -> None:
	unknown = sorted(set(value) - allowed)
	missing = sorted(allowed - set(value))
	if unknown:
		raise ValueError(f'{label} contains unknown key(s): {unknown!r}')
	if missing:
		raise ValueError(f'{label} missing required key(s): {missing!r}')


__all__ = [
	'BASELINE_MODEL_TAG',
	'CANDIDATE_MODEL_TAG',
	'F3VoxelDecoderSplitSuiteConfig',
	'F3VoxelSplitDatasetSuiteConfig',
	'F3VoxelSplitRobustnessPublishConfig',
	'F3VoxelSplitRobustnessSummaryConfig',
	'F3VoxelV0SplitSuiteConfig',
	'VoxelRobustnessModel',
	'f3_lithology_voxel_decoder_split_suite_config_from_mapping',
	'f3_lithology_voxel_split_dataset_config_from_mapping',
	'f3_lithology_voxel_split_summary_config_from_mapping',
	'f3_lithology_voxel_v0_split_suite_config_from_mapping',
]
