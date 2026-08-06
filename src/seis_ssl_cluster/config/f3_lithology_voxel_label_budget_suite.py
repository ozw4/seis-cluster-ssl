"""Strict configuration for the original-split voxel label-budget run suite."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pathlib import Path

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_required_str,
	_validate_allowed_keys,
)
from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	VoxelDecoderSpec,
	VoxelDecoderTileSettings,
	VoxelDecoderTrainSettings,
)
from seis_ssl_cluster.models.voxel_decoder.spec import (
	VOXEL_DECODER_NORMALIZATION,
	VOXEL_DECODER_SPEC,
	VOXEL_DECODER_UPSAMPLE_MODE,
	validate_voxel_decoder_implementation,
)

CANONICAL_STEPS_PER_EPOCH = 440
EXPECTED_BUDGETS = ('cap25', 'cap50', 'cap100')
EXPECTED_SUBSAMPLE_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_MODEL_TAGS = {
	'mae': 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
	'm1': 'strat_hmm_pretext_m1_k6_topblock1_distill',
	'm2a': 'strat_hmm_pretext_m2a_boundary_a050_t2_k6_topblock1_distill',
}


@dataclass(frozen=True)
class VoxelLabelBudgetSuiteModel:
	"""One frozen embedding source in the three-model suite."""

	role: str
	model_tag: str
	embeddings_dir: Path


@dataclass(frozen=True)
class F3VoxelLabelBudgetSuiteConfig:
	"""All preregistered settings needed to execute the 45 decoder jobs."""

	dataset_manifest: Path
	output_root: Path
	artifact_root: Path
	f3_root: Path
	dataset: Mapping[str, str]
	models: tuple[VoxelLabelBudgetSuiteModel, ...]
	budgets: tuple[str, ...]
	subsample_seeds: tuple[int, ...]
	base_seed: int
	add_subsample_seed: bool
	full_label_decoder_runs: Mapping[str, Path]
	require_shared_train_tile_identity: bool
	labels: Mapping[str, Path]
	decoder: VoxelDecoderSpec
	tiles: VoxelDecoderTileSettings
	train: VoxelDecoderTrainSettings
	write_probabilities: bool
	evaluation: Mapping[str, object]
	report: Mapping[str, object]
	overwrite: bool
	publish_individual_reports: bool

	def __post_init__(self) -> None:
		"""Validate that suite outputs stay below the artifact root."""
		_validate_scientific_contract(self)

	@property
	def model_by_role(self) -> Mapping[str, VoxelLabelBudgetSuiteModel]:
		"""Return the three configured roles by canonical short name."""
		return {model.role: model for model in self.models}


def f3_lithology_voxel_label_budget_suite_config_from_mapping(
	config: Mapping[str, object],
) -> F3VoxelLabelBudgetSuiteConfig:
	"""Resolve the 45-job suite config and reject every unknown key."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'suite',
				'paths',
				'dataset',
				'models',
				'budgets',
				'subsample_seeds',
				'seed_policy',
				'full_label_reference',
				'labels',
				'decoder',
				'tiles',
				'train',
				'inference',
				'evaluation',
				'report',
				'outputs',
			}
		),
		prefix='config',
	)
	suite = _required_mapping(config, 'suite')
	paths = _required_mapping(config, 'paths')
	dataset = _required_mapping(config, 'dataset')
	models = _required_mapping(config, 'models')
	seed_policy = _required_mapping(config, 'seed_policy')
	full = _required_mapping(config, 'full_label_reference')
	labels = _required_mapping(config, 'labels')
	decoder = _required_mapping(config, 'decoder')
	tiles = _required_mapping(config, 'tiles')
	train = _required_mapping(config, 'train')
	inference = _required_mapping(config, 'inference')
	evaluation = _required_mapping(config, 'evaluation')
	report = _required_mapping(config, 'report')
	outputs = _required_mapping(config, 'outputs')
	_validate_allowed_keys(
		suite, frozenset({'dataset_manifest', 'output_root'}), prefix='suite'
	)
	_validate_allowed_keys(
		paths, frozenset({'artifact_root', 'f3_root'}), prefix='paths'
	)
	_validate_allowed_keys(dataset, frozenset({'name', 'version'}), prefix='dataset')
	_validate_allowed_keys(models, frozenset({'mae', 'm1', 'm2a'}), prefix='models')
	_validate_allowed_keys(
		seed_policy,
		frozenset({'base_seed', 'add_subsample_seed'}),
		prefix='seed_policy',
	)
	_validate_allowed_keys(
		full,
		frozenset(
			{
				'mae_decoder_run',
				'm1_decoder_run',
				'm2a_decoder_run',
				'require_shared_train_tile_identity',
			}
		),
		prefix='full_label_reference',
	)
	_validate_allowed_keys(
		labels,
		frozenset(
			{
				'seismic_volume',
				'source_label_volume',
				'source_label_segy',
				'png_label_inventory',
				'segy_geometry_json',
				'class_info',
			}
		),
		prefix='labels',
	)
	_validate_allowed_keys(
		decoder,
		frozenset(
			{
				'spec',
				'embedding_dim',
				'class_count',
				'hidden_channels',
				'upsample_factors',
				'upsample_mode',
				'normalization',
			}
		),
		prefix='decoder',
	)
	_validate_allowed_keys(
		tiles, frozenset({'core_size_tokens', 'context_halo_tokens'}), prefix='tiles'
	)
	_validate_allowed_keys(
		train,
		frozenset(
			{
				'epochs',
				'batch_size',
				'learning_rate',
				'weight_decay',
				'class_weight',
				'sampling_mode',
				'steps_per_epoch',
				'num_workers',
				'amp',
				'gradient_clip_norm',
			}
		),
		prefix='train',
	)
	_validate_allowed_keys(
		inference, frozenset({'write_probabilities'}), prefix='inference'
	)
	_validate_allowed_keys(
		evaluation,
		frozenset(
			{
				'monitored_class_ids',
				'boundary_tolerances',
				'boundary_region_radii',
				'chunk_size_x',
			}
		),
		prefix='evaluation',
	)
	_validate_allowed_keys(
		report,
		frozenset(
			{
				'selected_slices',
				'dpi',
				'include_confidence',
				'amplitude_clip_percentiles',
			}
		),
		prefix='report',
	)
	_validate_allowed_keys(
		outputs,
		frozenset({'overwrite', 'publish_individual_reports'}),
		prefix='outputs',
	)
	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	output_root = _required_absolute_path(suite, 'output_root', prefix='suite')
	dataset_manifest = _required_absolute_path(
		suite, 'dataset_manifest', prefix='suite'
	)
	resolved_models = tuple(
		_model(role, models)
		for role in ('mae', 'm1', 'm2a')
	)
	base_seed = _integer(
		seed_policy.get('base_seed'), 'seed_policy.base_seed', minimum=0
	)
	add_seed = _boolean(
		seed_policy.get('add_subsample_seed'), 'seed_policy.add_subsample_seed'
	)
	shared_tiles = _boolean(
		full.get('require_shared_train_tile_identity'),
		'full_label_reference.require_shared_train_tile_identity',
	)
	full_runs = {
		role: _required_absolute_path(
			full, f'{role}_decoder_run', prefix='full_label_reference'
		)
		for role in ('mae', 'm1', 'm2a')
	}
	resolved_labels = {
		key: _required_absolute_path(labels, key, prefix='labels')
		for key in (
			'seismic_volume',
			'source_label_volume',
			'source_label_segy',
			'png_label_inventory',
			'segy_geometry_json',
			'class_info',
		)
	}
	validate_voxel_decoder_implementation(
		spec=decoder.get('spec'),
		upsample_mode=decoder.get('upsample_mode'),
		normalization=decoder.get('normalization'),
		field_prefix='decoder',
	)
	train_settings = _train_settings(train, base_seed=base_seed)
	if train_settings.sampling_mode != 'uniform_tiles_with_replacement':
		raise ValueError(
			'train.sampling_mode must be uniform_tiles_with_replacement for this suite'
		)
	write_probabilities = _boolean(
		inference.get('write_probabilities'), 'inference.write_probabilities'
	)
	if write_probabilities:
		raise ValueError('inference.write_probabilities must be false')
	return F3VoxelLabelBudgetSuiteConfig(
		dataset_manifest=dataset_manifest,
		output_root=output_root,
		artifact_root=artifact_root,
		f3_root=f3_root,
		dataset={
			'name': _required_str(dataset, 'name', prefix='dataset'),
			'version': _required_str(dataset, 'version', prefix='dataset'),
		},
		models=resolved_models,
		budgets=_budget_ids(config.get('budgets')),
		subsample_seeds=_integer_list(
			config.get('subsample_seeds'), 'subsample_seeds', minimum=0
		),
		base_seed=base_seed,
		add_subsample_seed=add_seed,
		full_label_decoder_runs=full_runs,
		require_shared_train_tile_identity=shared_tiles,
		labels=resolved_labels,
		decoder=_decoder_spec(decoder),
		tiles=VoxelDecoderTileSettings(
			core_size_tokens=_triplet(
				tiles.get('core_size_tokens'), 'tiles.core_size_tokens', minimum=1
			),
			context_halo_tokens=_triplet(
				tiles.get('context_halo_tokens'),
				'tiles.context_halo_tokens',
				minimum=0,
			),
		),
		train=train_settings,
		write_probabilities=False,
		evaluation={
			'monitored_class_ids': list(
				_integer_list(
					evaluation.get('monitored_class_ids'),
					'evaluation.monitored_class_ids',
					minimum=0,
				)
			),
			'boundary_tolerances': list(
				_integer_list(
					evaluation.get('boundary_tolerances'),
					'evaluation.boundary_tolerances',
					minimum=1,
				)
			),
			'boundary_region_radii': list(
				_integer_list(
					evaluation.get('boundary_region_radii'),
					'evaluation.boundary_region_radii',
					minimum=1,
				)
			),
			'chunk_size_x': _integer(
				evaluation.get('chunk_size_x'), 'evaluation.chunk_size_x', minimum=1
			),
		},
		report=_report(report),
		overwrite=_boolean(outputs.get('overwrite'), 'outputs.overwrite'),
		publish_individual_reports=_boolean(
			outputs.get('publish_individual_reports'),
			'outputs.publish_individual_reports',
		),
	)


def _model(role: str, models: Mapping[str, object]) -> VoxelLabelBudgetSuiteModel:
	value = models.get(role)
	if not isinstance(value, Mapping):
		raise TypeError(f'models.{role} must be a mapping')
	_validate_allowed_keys(
		value, frozenset({'model_tag', 'embeddings_dir'}), prefix=f'models.{role}'
	)
	embeddings = _required_absolute_path(
		value, 'embeddings_dir', prefix=f'models.{role}'
	)
	return VoxelLabelBudgetSuiteModel(
		role=role,
		model_tag=_required_str(value, 'model_tag', prefix=f'models.{role}'),
		embeddings_dir=embeddings,
	)


def _decoder_spec(value: Mapping[str, object]) -> VoxelDecoderSpec:
	return VoxelDecoderSpec(
		spec=VOXEL_DECODER_SPEC,
		embedding_dim=_integer(
			value.get('embedding_dim'), 'decoder.embedding_dim', minimum=1
		),
		class_count=_integer(
			value.get('class_count'), 'decoder.class_count', minimum=1
		),
		hidden_channels=_integer_list(
			value.get('hidden_channels'), 'decoder.hidden_channels', minimum=1
		),
		upsample_factors=tuple(
			_triplet(item, 'decoder.upsample_factors', minimum=1)
			for item in _sequence(
				value.get('upsample_factors'), 'decoder.upsample_factors'
			)
		),
		upsample_mode=VOXEL_DECODER_UPSAMPLE_MODE,
		normalization=VOXEL_DECODER_NORMALIZATION,
	)


def _train_settings(
	value: Mapping[str, object], *, base_seed: int
) -> VoxelDecoderTrainSettings:
	class_weight = _required_str(value, 'class_weight', prefix='train')
	if class_weight != 'balanced':
		raise ValueError("train.class_weight must be 'balanced'")
	sampling_mode = _required_str(value, 'sampling_mode', prefix='train')
	steps = _integer(value.get('steps_per_epoch'), 'train.steps_per_epoch', minimum=1)
	return VoxelDecoderTrainSettings(
		epochs=_integer(value.get('epochs'), 'train.epochs', minimum=1),
		batch_size=_integer(value.get('batch_size'), 'train.batch_size', minimum=1),
		learning_rate=_positive_float(
			value.get('learning_rate'), 'train.learning_rate'
		),
		weight_decay=_nonnegative_float(
			value.get('weight_decay'), 'train.weight_decay'
		),
		class_weight=class_weight,
		seed=base_seed,
		num_workers=_integer(value.get('num_workers'), 'train.num_workers', minimum=0),
		amp=_boolean(value.get('amp'), 'train.amp'),
		gradient_clip_norm=_positive_float(
			value.get('gradient_clip_norm'), 'train.gradient_clip_norm'
		),
		sampling_mode=sampling_mode,
		steps_per_epoch=steps,
	)


def _report(value: Mapping[str, object]) -> Mapping[str, object]:
	selected = value.get('selected_slices')
	if not isinstance(selected, Mapping):
		raise TypeError('report.selected_slices must be a mapping')
	_validate_allowed_keys(
		selected, frozenset({'inline', 'crossline'}), prefix='report.selected_slices'
	)
	return {
		'selected_slices': {
			key: list(
				_integer_list(
					selected.get(key, ()),
					f'report.selected_slices.{key}',
					minimum=0,
					empty=True,
				)
			)
			for key in ('inline', 'crossline')
		},
		'dpi': _integer(value.get('dpi'), 'report.dpi', minimum=1),
		'include_confidence': _boolean(
			value.get('include_confidence'), 'report.include_confidence'
		),
		'amplitude_clip_percentiles': list(
			_float_pair(
				value.get('amplitude_clip_percentiles'),
				'report.amplitude_clip_percentiles',
			)
		),
	}


def _budget_ids(value: object) -> tuple[str, ...]:
	items = _sequence(value, 'budgets')
	result = []
	for item in items:
		if not isinstance(item, str) or not item.startswith('cap'):
			raise ValueError('budgets entries must have form cap<positive-int>')
		try:
			cap = int(item[3:])
		except ValueError as error:
			raise ValueError(
				'budgets entries must have form cap<positive-int>'
			) from error
		if cap <= 0 or item != f'cap{cap}':
			raise ValueError('budgets entries must be canonical positive caps')
		result.append(item)
	if len(set(result)) != len(result):
		raise ValueError('budgets must not contain duplicates')
	return tuple(result)


def _integer_list(
	value: object, label: str, *, minimum: int, empty: bool = False
) -> tuple[int, ...]:
	items = _sequence(value, label, empty=empty)
	result = tuple(_integer(item, label, minimum=minimum) for item in items)
	if len(set(result)) != len(result):
		raise ValueError(f'{label} must not contain duplicates')
	return result


def _sequence(value: object, label: str, *, empty: bool = False) -> tuple[object, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be a list')
	result = tuple(value)
	if not result and not empty:
		raise ValueError(f'{label} must not be empty')
	return result


def _triplet(value: object, label: str, *, minimum: int) -> tuple[int, int, int]:
	items = _sequence(value, label)
	if len(items) != 3:
		raise ValueError(f'{label} must contain three integers')
	result = tuple(_integer(item, label, minimum=minimum) for item in items)
	return (result[0], result[1], result[2])


def _integer(value: object, label: str, *, minimum: int) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
		raise ValueError(f'{label} must be an integer >= {minimum}')
	return value


def _positive_float(value: object, label: str) -> float:
	if (
		not isinstance(value, int | float)
		or isinstance(value, bool)
		or float(value) <= 0
	):
		raise ValueError(f'{label} must be positive')
	return float(value)


def _nonnegative_float(value: object, label: str) -> float:
	if (
		not isinstance(value, int | float)
		or isinstance(value, bool)
		or float(value) < 0
	):
		raise ValueError(f'{label} must be non-negative')
	return float(value)


def _float_pair(value: object, label: str) -> tuple[float, float]:
	items = _sequence(value, label)
	if len(items) != 2 or any(
		not isinstance(item, int | float) or isinstance(item, bool) for item in items
	):
		raise ValueError(f'{label} must contain two numbers')
	return (float(items[0]), float(items[1]))


def _boolean(value: object, label: str) -> bool:
	if not isinstance(value, bool):
		raise TypeError(f'{label} must be boolean')
	return value




def _validate_scientific_contract(  # noqa: C901
	config: F3VoxelLabelBudgetSuiteConfig,
) -> None:
	"""Reject drift from the preregistered M3-V-LB experiment matrix."""
	if config.dataset != {
		'name': 'f3_facies_benchmark',
		'version': 'facies_benchmark_v1',
	}:
		raise ValueError('dataset must be f3_facies_benchmark/facies_benchmark_v1')
	if config.budgets != EXPECTED_BUDGETS:
		raise ValueError(f'budgets must be exactly {list(EXPECTED_BUDGETS)!r}')
	if config.subsample_seeds != EXPECTED_SUBSAMPLE_SEEDS:
		raise ValueError(
			f'subsample_seeds must be exactly {list(EXPECTED_SUBSAMPLE_SEEDS)!r}'
		)
	if config.base_seed != 42000 or not config.add_subsample_seed:
		raise ValueError('seed policy must be decoder_seed = 42000 + subsample_seed')
	if {model.role: model.model_tag for model in config.models} != EXPECTED_MODEL_TAGS:
		raise ValueError('models must match the canonical MAE/M1/M2-A identities')
	if not config.require_shared_train_tile_identity:
		raise ValueError('full-label train tile identity matching must be required')
	expected_decoder = VoxelDecoderSpec(
		spec=VOXEL_DECODER_SPEC,
		embedding_dim=384,
		class_count=6,
		hidden_channels=(128, 64, 32),
		upsample_factors=((2, 2, 2), (2, 2, 2), (2, 2, 2)),
		upsample_mode=VOXEL_DECODER_UPSAMPLE_MODE,
		normalization=VOXEL_DECODER_NORMALIZATION,
	)
	if config.decoder != expected_decoder:
		raise ValueError('decoder must match the canonical M3-V V1 architecture')
	if config.tiles != VoxelDecoderTileSettings(
		core_size_tokens=(8, 8, 8), context_halo_tokens=(1, 1, 1)
	):
		raise ValueError('tile geometry must match the canonical M3-V V1 geometry')
	expected_train = VoxelDecoderTrainSettings(
		epochs=50,
		batch_size=1,
		learning_rate=0.001,
		weight_decay=0.0001,
		class_weight='balanced',
		seed=42000,
		num_workers=0,
		amp=True,
		gradient_clip_norm=1.0,
		sampling_mode='uniform_tiles_with_replacement',
		steps_per_epoch=CANONICAL_STEPS_PER_EPOCH,
	)
	if config.train != expected_train:
		raise ValueError(
			'train settings differ from the preregistered M3-V-LB contract'
		)
	if config.evaluation != {
		'monitored_class_ids': [3, 5],
		'boundary_tolerances': [2, 4],
		'boundary_region_radii': [2, 4],
		'chunk_size_x': 8,
	}:
		raise ValueError('evaluation settings differ from the M3-V-LB contract')
	if config.write_probabilities:
		raise ValueError('inference.write_probabilities must be false')
	if config.overwrite or config.publish_individual_reports:
		raise ValueError('suite outputs must disable overwrite and individual publish')


__all__ = [
	'CANONICAL_STEPS_PER_EPOCH',
	'EXPECTED_BUDGETS',
	'EXPECTED_MODEL_TAGS',
	'EXPECTED_SUBSAMPLE_SEEDS',
	'F3VoxelLabelBudgetSuiteConfig',
	'VoxelLabelBudgetSuiteModel',
	'f3_lithology_voxel_label_budget_suite_config_from_mapping',
]
