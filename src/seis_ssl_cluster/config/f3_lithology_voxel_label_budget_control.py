"""Strict configuration for the current-code F3 K=6 control suite.

The control intentionally has a narrower surface than the historical
three-model M3-V-LB suite: it can create outputs only for the current-code
single-head K=6 model and reads the historical MAE/M1 rows from the existing
run manifest.  Keeping that separation in the resolver prevents accidental
retraining or replacement of historical artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from seis_ssl_cluster.config.f3_lithology_common import (
	_max_file_size_bytes,
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

if TYPE_CHECKING:
	from pathlib import Path

CURRENT_K6_MODEL_ID = 'm1_current_k6'
CURRENT_K6_MODEL_TAG = 'strat_hmm_pretext_m1_current_k6_topblock1_distill_v1'
MAE_MODEL_ID = 'mae'
HISTORICAL_M1_MODEL_ID = 'm1'

EXPECTED_DATASET = {
	'name': 'f3_facies_benchmark',
	'version': 'facies_benchmark_v1',
}
EXPECTED_BUDGETS = ('cap25', 'cap50', 'cap100')
EXPECTED_SUBSAMPLE_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_COMPARISONS = (
	(CURRENT_K6_MODEL_ID, HISTORICAL_M1_MODEL_ID),
	(CURRENT_K6_MODEL_ID, MAE_MODEL_ID),
	(HISTORICAL_M1_MODEL_ID, MAE_MODEL_ID),
)
REQUIRED_LABEL_KEYS = frozenset(
	{
		'seismic_volume',
		'source_label_volume',
		'source_label_segy',
		'png_label_inventory',
		'segy_geometry_json',
		'class_info',
	}
)
CANONICAL_BASE_SEED = 42000
CANONICAL_STEPS_PER_EPOCH = 440


@dataclass(frozen=True)
class F3VoxelLabelBudgetControlCandidate:
	"""The only model whose voxel decoder jobs this suite may create."""

	model_id: str
	model_tag: str
	embeddings_dir: Path


@dataclass(frozen=True)
class F3VoxelLabelBudgetControlReferences:
	"""Read-only dataset and historical result sources for paired comparisons."""

	dataset_manifest: Path
	historical_run_manifest: Path | None
	mae_model_id: str | None
	historical_m1_model_id: str | None


@dataclass(frozen=True)
class F3VoxelLabelBudgetControlPublishConfig:
	"""Lightweight result publication settings for the current-code control."""

	enabled: bool
	results_root: Path
	output_dir: Path
	max_file_size_bytes: int
	overwrite: bool

	def __post_init__(self) -> None:
		"""Keep publishable files below the repository results root."""
		if self.max_file_size_bytes <= 0:
			raise ValueError('publish.max_file_size_bytes must be positive')


@dataclass(frozen=True)
class F3VoxelLabelBudgetControlDecisionThresholds:
	"""Preregistered descriptive thresholds for control readiness."""

	minimum_positive_budgets: int
	minimum_primary_wins: int
	drift_absolute_mean_delta: float
	drift_budget_count: int
	monitored_class_ids: tuple[int, ...]
	major_degradation_delta: float
	systematic_degradation_budget_count: int

	def __post_init__(self) -> None:
		"""Reject thresholds that do not fit the fixed 3-by-5 design."""
		_bounded_positive_int(
			self.minimum_positive_budgets,
			'decision.minimum_positive_budgets',
			maximum=len(EXPECTED_BUDGETS),
		)
		_bounded_positive_int(
			self.minimum_primary_wins,
			'decision.minimum_primary_wins',
			maximum=len(EXPECTED_SUBSAMPLE_SEEDS),
		)
		_bounded_positive_int(
			self.drift_budget_count,
			'decision.drift_budget_count',
			maximum=len(EXPECTED_BUDGETS),
		)
		_bounded_positive_int(
			self.systematic_degradation_budget_count,
			'decision.systematic_degradation_budget_count',
			maximum=len(EXPECTED_BUDGETS),
		)
		if self.monitored_class_ids != (3, 5):
			raise ValueError('decision.monitored_class_ids must be exactly [3, 5]')
		for label, value in (
			('decision.drift_absolute_mean_delta', self.drift_absolute_mean_delta),
			('decision.major_degradation_delta', self.major_degradation_delta),
		):
			if not math.isfinite(value):
				raise ValueError(f'{label} must be finite')
		if self.drift_absolute_mean_delta <= 0.0:
			raise ValueError(
				'decision.drift_absolute_mean_delta must be strictly positive'
			)
		if self.major_degradation_delta != -0.05:
			raise ValueError(
				'decision.major_degradation_delta must be exactly -0.05'
			)

	def to_dict(self) -> dict[str, object]:
		"""Return an exact JSON-compatible decision contract."""
		return {
			'minimum_positive_budgets': self.minimum_positive_budgets,
			'minimum_primary_wins': self.minimum_primary_wins,
			'drift_absolute_mean_delta': self.drift_absolute_mean_delta,
			'drift_budget_count': self.drift_budget_count,
			'monitored_class_ids': list(self.monitored_class_ids),
			'major_degradation_delta': self.major_degradation_delta,
			'systematic_degradation_budget_count': (
				self.systematic_degradation_budget_count
			),
		}


@dataclass(frozen=True)
class F3VoxelLabelBudgetControlConfig:
	"""Resolved current-K=6-only configuration for runner and summary stages."""

	artifact_root: Path
	f3_root: Path
	results_root: Path
	dataset: Mapping[str, str]
	references: F3VoxelLabelBudgetControlReferences
	candidate: F3VoxelLabelBudgetControlCandidate
	output_root: Path
	budgets: tuple[str, ...]
	subsample_seeds: tuple[int, ...]
	base_seed: int
	add_subsample_seed: bool
	labels: Mapping[str, Path]
	decoder: VoxelDecoderSpec
	tiles: VoxelDecoderTileSettings
	train: VoxelDecoderTrainSettings
	write_probabilities: bool
	evaluation: Mapping[str, object]
	report: Mapping[str, object]
	comparisons: tuple[tuple[str, str], ...]
	decision: F3VoxelLabelBudgetControlDecisionThresholds
	overwrite: bool
	publish: F3VoxelLabelBudgetControlPublishConfig
	validate_pairing_reference: bool = True

	def __post_init__(self) -> None:
		"""Validate artifact ownership and the fixed scientific control contract."""
		if set(self.labels) != REQUIRED_LABEL_KEYS:
			raise ValueError(
				'labels must define exactly the six canonical voxel decoder inputs'
			)
		_validate_scientific_contract(self)

	@property
	def dataset_manifest(self) -> Path:
		"""Return the read-only shared label-budget dataset manifest."""
		return self.references.dataset_manifest

	@property
	def historical_run_manifest(self) -> Path:
		"""Return the read-only MAE/historical-M1 run manifest."""
		if self.references.historical_run_manifest is None:
			raise ValueError('historical run manifest is not configured')
		return self.references.historical_run_manifest

	@property
	def reports_dir(self) -> Path:
		"""Return the control-owned directory for summary outputs."""
		return self.output_root / 'reports'

	@property
	def model_by_role(self) -> Mapping[str, F3VoxelLabelBudgetControlCandidate]:
		"""Expose the new writable candidate by its canonical model ID."""
		return {self.candidate.model_id: self.candidate}

	@property
	def mae_model_id(self) -> str | None:
		"""Return the historical MAE role used in paired comparisons."""
		return self.references.mae_model_id

	@property
	def historical_m1_model_id(self) -> str | None:
		"""Return the historical M1 role used in paired comparisons."""
		return self.references.historical_m1_model_id

	@property
	def job_count(self) -> int:
		"""Return the immutable current-control decoder matrix size."""
		return len(self.budgets) * len(self.subsample_seeds)

	def decoder_seed(self, subsample_seed: int) -> int:
		"""Resolve the paired decoder seed for one approved subsample seed."""
		if subsample_seed not in self.subsample_seeds:
			raise ValueError(f'unknown configured subsample seed: {subsample_seed}')
		return self.base_seed + subsample_seed

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-compatible resolved configuration snapshot."""
		return {
			'paths': {
				'artifact_root': str(self.artifact_root),
				'f3_root': str(self.f3_root),
				'results_root': str(self.results_root),
			},
			'dataset': dict(self.dataset),
			'references': {
				'dataset_manifest': str(self.dataset_manifest),
				**(
					{
						'historical_run_manifest': str(
							self.historical_run_manifest
						),
						'mae_model_id': self.mae_model_id,
						'historical_m1_model_id': self.historical_m1_model_id,
					}
					if self.validate_pairing_reference
					else {}
				),
			},
			'candidate': {
				'model_id': self.candidate.model_id,
				'model_tag': self.candidate.model_tag,
				'embeddings_dir': str(self.candidate.embeddings_dir),
			},
			'budgets': list(self.budgets),
			'subsample_seeds': list(self.subsample_seeds),
			'seed_policy': {
				'base_seed': self.base_seed,
				'add_subsample_seed': self.add_subsample_seed,
			},
			'labels': {key: str(path) for key, path in self.labels.items()},
			'decoder': {
				'spec': self.decoder.spec,
				'embedding_dim': self.decoder.embedding_dim,
				'class_count': self.decoder.class_count,
				'hidden_channels': list(self.decoder.hidden_channels),
				'upsample_factors': [
					list(item) for item in self.decoder.upsample_factors
				],
				'upsample_mode': self.decoder.upsample_mode,
				'normalization': self.decoder.normalization,
			},
			'tiles': {
				'core_size_tokens': list(self.tiles.core_size_tokens),
				'context_halo_tokens': list(self.tiles.context_halo_tokens),
			},
			'train': {
				'epochs': self.train.epochs,
				'batch_size': self.train.batch_size,
				'learning_rate': self.train.learning_rate,
				'weight_decay': self.train.weight_decay,
				'class_weight': self.train.class_weight,
				'sampling_mode': self.train.sampling_mode,
				'steps_per_epoch': self.train.steps_per_epoch,
				'num_workers': self.train.num_workers,
				'amp': self.train.amp,
				'gradient_clip_norm': self.train.gradient_clip_norm,
			},
			'inference': {'write_probabilities': self.write_probabilities},
			'evaluation': dict(self.evaluation),
			'report': dict(self.report),
			'comparisons': [list(pair) for pair in self.comparisons],
			'decision': self.decision.to_dict(),
			'outputs': {
				'output_root': str(self.output_root),
				'overwrite': self.overwrite,
			},
			'publish': {
				'enabled': self.publish.enabled,
				'output_dir': str(self.publish.output_dir),
				'max_file_size_mb': self.publish.max_file_size_bytes / (1024 * 1024),
				'overwrite': self.publish.overwrite,
			},
		}


def f3_lithology_voxel_label_budget_control_config_from_mapping(
	config: Mapping[str, object],
	*,
	validate_pairing_reference: bool = True,
) -> F3VoxelLabelBudgetControlConfig:
	"""Validate and resolve the current-code K=6 control configuration."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'references',
				'candidate',
				'budgets',
				'subsample_seeds',
				'seed_policy',
				'labels',
				'decoder',
				'tiles',
				'train',
				'inference',
				'evaluation',
				'report',
				'comparisons',
				'decision',
				'outputs',
				'publish',
			}
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	dataset = _required_mapping(config, 'dataset')
	references = _required_mapping(config, 'references')
	candidate = _required_mapping(config, 'candidate')
	seed_policy = _required_mapping(config, 'seed_policy')
	labels = _required_mapping(config, 'labels')
	decoder = _required_mapping(config, 'decoder')
	tiles = _required_mapping(config, 'tiles')
	train = _required_mapping(config, 'train')
	inference = _required_mapping(config, 'inference')
	evaluation = _required_mapping(config, 'evaluation')
	report = _required_mapping(config, 'report')
	decision = _required_mapping(config, 'decision')
	outputs = _required_mapping(config, 'outputs')
	publish = _required_mapping(config, 'publish')
	_validate_section_keys(paths, {'artifact_root', 'f3_root', 'results_root'}, 'paths')
	_validate_section_keys(dataset, {'name', 'version'}, 'dataset')
	_validate_section_keys(
		references,
		{
			'dataset_manifest',
			'historical_run_manifest',
			'mae_model_id',
			'historical_m1_model_id',
		},
		'references',
	)
	_validate_section_keys(
		candidate,
		{'model_id', 'model_tag', 'embeddings_dir'},
		'candidate',
	)
	_validate_section_keys(
		seed_policy,
		{'base_seed', 'add_subsample_seed'},
		'seed_policy',
	)
	_validate_section_keys(
		labels,
		set(REQUIRED_LABEL_KEYS),
		'labels',
	)
	_validate_section_keys(
		decoder,
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
	_validate_section_keys(
		tiles,
		{'core_size_tokens', 'context_halo_tokens'},
		'tiles',
	)
	_validate_section_keys(
		train,
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
		},
		'train',
	)
	_validate_section_keys(inference, {'write_probabilities'}, 'inference')
	_validate_section_keys(
		evaluation,
		{
			'monitored_class_ids',
			'boundary_tolerances',
			'boundary_region_radii',
			'chunk_size_x',
		},
		'evaluation',
	)
	_validate_section_keys(
		report,
		{
			'selected_slices',
			'dpi',
			'include_confidence',
			'amplitude_clip_percentiles',
		},
		'report',
	)
	_validate_section_keys(
		decision,
		{
			'minimum_positive_budgets',
			'minimum_primary_wins',
			'drift_absolute_mean_delta',
			'drift_budget_count',
			'monitored_class_ids',
			'major_degradation_delta',
			'systematic_degradation_budget_count',
		},
		'decision',
	)
	_validate_section_keys(outputs, {'output_root', 'overwrite'}, 'outputs')
	_validate_section_keys(
		publish,
		{'enabled', 'output_dir', 'max_file_size_mb', 'overwrite'},
		'publish',
	)

	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	results_root = _required_absolute_path(paths, 'results_root', prefix='paths')
	base_seed = _integer(
		seed_policy.get('base_seed'), 'seed_policy.base_seed', minimum=0
	)
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
	write_probabilities = _boolean(
		inference.get('write_probabilities'), 'inference.write_probabilities'
	)
	if write_probabilities:
		raise ValueError('inference.write_probabilities must be false')
	return F3VoxelLabelBudgetControlConfig(
		artifact_root=artifact_root,
		f3_root=f3_root,
		results_root=results_root,
		dataset={
			'name': _required_str(dataset, 'name', prefix='dataset'),
			'version': _required_str(dataset, 'version', prefix='dataset'),
		},
		references=F3VoxelLabelBudgetControlReferences(
			dataset_manifest=_required_absolute_path(
				references, 'dataset_manifest', prefix='references'
			),
			historical_run_manifest=(
				_required_absolute_path(
					references, 'historical_run_manifest', prefix='references'
				)
				if validate_pairing_reference
				else _optional_absolute_path(
					references, 'historical_run_manifest', prefix='references'
				)
			),
			mae_model_id=(
				_required_str(references, 'mae_model_id', prefix='references')
				if validate_pairing_reference
				else _optional_str(references, 'mae_model_id', prefix='references')
			),
			historical_m1_model_id=(
				_required_str(
					references, 'historical_m1_model_id', prefix='references'
				)
				if validate_pairing_reference
				else _optional_str(
					references,
					'historical_m1_model_id',
					prefix='references',
				)
			),
		),
		candidate=F3VoxelLabelBudgetControlCandidate(
			model_id=_required_str(candidate, 'model_id', prefix='candidate'),
			model_tag=_required_str(candidate, 'model_tag', prefix='candidate'),
			embeddings_dir=_required_absolute_path(
				candidate, 'embeddings_dir', prefix='candidate'
			),
		),
		output_root=_required_absolute_path(outputs, 'output_root', prefix='outputs'),
		budgets=_budget_ids(config.get('budgets')),
		subsample_seeds=_integer_list(
			config.get('subsample_seeds'), 'subsample_seeds', minimum=0
		),
		base_seed=base_seed,
		add_subsample_seed=_boolean(
			seed_policy.get('add_subsample_seed'),
			'seed_policy.add_subsample_seed',
		),
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
		train=_train_settings(train, base_seed=base_seed),
		write_probabilities=False,
		evaluation=_evaluation(evaluation),
		report=_report(report),
		comparisons=_comparison_pairs(config.get('comparisons')),
		decision=F3VoxelLabelBudgetControlDecisionThresholds(
			minimum_positive_budgets=_integer(
				decision.get('minimum_positive_budgets'),
				'decision.minimum_positive_budgets',
				minimum=1,
			),
			minimum_primary_wins=_integer(
				decision.get('minimum_primary_wins'),
				'decision.minimum_primary_wins',
				minimum=1,
			),
			drift_absolute_mean_delta=_positive_float(
				decision.get('drift_absolute_mean_delta'),
				'decision.drift_absolute_mean_delta',
			),
			drift_budget_count=_integer(
				decision.get('drift_budget_count'),
				'decision.drift_budget_count',
				minimum=1,
			),
			monitored_class_ids=_integer_list(
				decision.get('monitored_class_ids'),
				'decision.monitored_class_ids',
				minimum=0,
			),
			major_degradation_delta=_negative_float(
				decision.get('major_degradation_delta'),
				'decision.major_degradation_delta',
			),
			systematic_degradation_budget_count=_integer(
				decision.get('systematic_degradation_budget_count'),
				'decision.systematic_degradation_budget_count',
				minimum=1,
			),
		),
		overwrite=_boolean(outputs.get('overwrite'), 'outputs.overwrite'),
		publish=F3VoxelLabelBudgetControlPublishConfig(
			enabled=_boolean(publish.get('enabled'), 'publish.enabled'),
			results_root=results_root,
			output_dir=_required_absolute_path(
				publish, 'output_dir', prefix='publish'
			),
			max_file_size_bytes=_max_file_size_bytes(publish),
			overwrite=_boolean(publish.get('overwrite'), 'publish.overwrite'),
		),
		validate_pairing_reference=validate_pairing_reference,
	)


def _validate_section_keys(
	value: Mapping[str, object], allowed: set[str], prefix: str
) -> None:
	_validate_allowed_keys(value, frozenset(allowed), prefix=prefix)


def _optional_absolute_path(
	value: Mapping[str, object], key: str, *, prefix: str
) -> Path | None:
	if key not in value:
		return None
	return _required_absolute_path(value, key, prefix=prefix)


def _optional_str(
	value: Mapping[str, object], key: str, *, prefix: str
) -> str | None:
	if key not in value:
		return None
	return _required_str(value, key, prefix=prefix)


def _decoder_spec(value: Mapping[str, object]) -> VoxelDecoderSpec:
	validate_voxel_decoder_implementation(
		spec=value.get('spec'),
		upsample_mode=value.get('upsample_mode'),
		normalization=value.get('normalization'),
		field_prefix='decoder',
	)
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
		steps_per_epoch=_integer(
			value.get('steps_per_epoch'), 'train.steps_per_epoch', minimum=1
		),
	)


def _evaluation(value: Mapping[str, object]) -> Mapping[str, object]:
	return {
		'monitored_class_ids': list(
			_integer_list(
				value.get('monitored_class_ids'),
				'evaluation.monitored_class_ids',
				minimum=0,
			)
		),
		'boundary_tolerances': list(
			_integer_list(
				value.get('boundary_tolerances'),
				'evaluation.boundary_tolerances',
				minimum=1,
			)
		),
		'boundary_region_radii': list(
			_integer_list(
				value.get('boundary_region_radii'),
				'evaluation.boundary_region_radii',
				minimum=1,
			)
		),
		'chunk_size_x': _integer(
			value.get('chunk_size_x'), 'evaluation.chunk_size_x', minimum=1
		),
	}


def _report(value: Mapping[str, object]) -> Mapping[str, object]:
	selected = _required_mapping(value, 'selected_slices')
	_validate_section_keys(selected, {'inline', 'crossline'}, 'report.selected_slices')
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


def _comparison_pairs(value: object) -> tuple[tuple[str, str], ...]:
	pairs = _sequence(value, 'comparisons')
	result: list[tuple[str, str]] = []
	for index, item in enumerate(pairs):
		pair = _sequence(item, f'comparisons[{index}]')
		if len(pair) != 2 or any(
			not isinstance(member, str) or not member for member in pair
		):
			raise ValueError(
				f'comparisons[{index}] must contain exactly two non-empty model IDs'
			)
		result.append((str(pair[0]), str(pair[1])))
	if len(set(result)) != len(result):
		raise ValueError('comparisons must not contain duplicates')
	return tuple(result)


def _budget_ids(value: object) -> tuple[str, ...]:
	items = _sequence(value, 'budgets')
	result: list[str] = []
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
		or not math.isfinite(float(value))
		or float(value) <= 0.0
	):
		raise ValueError(f'{label} must be a finite positive number')
	return float(value)


def _negative_float(value: object, label: str) -> float:
	if (
		not isinstance(value, int | float)
		or isinstance(value, bool)
		or not math.isfinite(float(value))
		or float(value) >= 0.0
	):
		raise ValueError(f'{label} must be a finite negative number')
	return float(value)


def _nonnegative_float(value: object, label: str) -> float:
	if (
		not isinstance(value, int | float)
		or isinstance(value, bool)
		or not math.isfinite(float(value))
		or float(value) < 0.0
	):
		raise ValueError(f'{label} must be a finite non-negative number')
	return float(value)


def _float_pair(value: object, label: str) -> tuple[float, float]:
	items = _sequence(value, label)
	if len(items) != 2 or any(
		not isinstance(item, int | float)
		or isinstance(item, bool)
		or not math.isfinite(float(item))
		for item in items
	):
		raise ValueError(f'{label} must contain two finite numbers')
	return (float(items[0]), float(items[1]))


def _boolean(value: object, label: str) -> bool:
	if not isinstance(value, bool):
		raise TypeError(f'{label} must be boolean')
	return value


def _bounded_positive_int(value: int, label: str, *, maximum: int) -> None:
	if (
		not isinstance(value, int)
		or isinstance(value, bool)
		or not 1 <= value <= maximum
	):
		raise ValueError(f'{label} must be in [1, {maximum}]')


def _validate_scientific_contract(  # noqa: C901, PLR0912
	config: F3VoxelLabelBudgetControlConfig,
) -> None:
	if dict(config.dataset) != EXPECTED_DATASET:
		raise ValueError('dataset must be f3_facies_benchmark/facies_benchmark_v1')
	if config.validate_pairing_reference:
		if config.references.mae_model_id != MAE_MODEL_ID:
			raise ValueError(f'references.mae_model_id must be {MAE_MODEL_ID!r}')
		if config.references.historical_m1_model_id != HISTORICAL_M1_MODEL_ID:
			raise ValueError(
				'references.historical_m1_model_id must be '
				f'{HISTORICAL_M1_MODEL_ID!r}'
			)
	if config.candidate.model_id != CURRENT_K6_MODEL_ID:
		raise ValueError(f'candidate.model_id must be {CURRENT_K6_MODEL_ID!r}')
	if config.candidate.model_tag != CURRENT_K6_MODEL_TAG:
		raise ValueError(f'candidate.model_tag must be {CURRENT_K6_MODEL_TAG!r}')
	if config.budgets != EXPECTED_BUDGETS:
		raise ValueError(f'budgets must be exactly {list(EXPECTED_BUDGETS)!r}')
	if config.subsample_seeds != EXPECTED_SUBSAMPLE_SEEDS:
		raise ValueError(
			f'subsample_seeds must be exactly {list(EXPECTED_SUBSAMPLE_SEEDS)!r}'
		)
	if config.base_seed != CANONICAL_BASE_SEED or not config.add_subsample_seed:
		raise ValueError('seed policy must be decoder_seed = 42000 + subsample_seed')
	if config.comparisons != EXPECTED_COMPARISONS:
		raise ValueError(
			'comparisons must be current K6-historical M1, current K6-MAE, '
			'and historical M1-MAE in that order'
		)
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
		raise ValueError('decoder must match the canonical M3-V-LB architecture')
	if config.tiles != VoxelDecoderTileSettings(
		core_size_tokens=(8, 8, 8), context_halo_tokens=(1, 1, 1)
	):
		raise ValueError('tile geometry must match the canonical M3-V-LB geometry')
	expected_train = VoxelDecoderTrainSettings(
		epochs=50,
		batch_size=1,
		learning_rate=0.001,
		weight_decay=0.0001,
		class_weight='balanced',
		seed=CANONICAL_BASE_SEED,
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
	if config.overwrite:
		raise ValueError('outputs.overwrite must be false for the control suite')
	if config.decision.minimum_positive_budgets != 2:
		raise ValueError('decision.minimum_positive_budgets must be 2')
	if config.decision.minimum_primary_wins != 4:
		raise ValueError('decision.minimum_primary_wins must be 4')
	if config.decision.drift_absolute_mean_delta != 0.01:
		raise ValueError('decision.drift_absolute_mean_delta must be 0.01')
	if config.decision.drift_budget_count != 2:
		raise ValueError('decision.drift_budget_count must be 2')
	if config.decision.systematic_degradation_budget_count != 2:
		raise ValueError(
			'decision.systematic_degradation_budget_count must be 2'
		)


__all__ = [
	'CANONICAL_BASE_SEED',
	'CANONICAL_STEPS_PER_EPOCH',
	'CURRENT_K6_MODEL_ID',
	'CURRENT_K6_MODEL_TAG',
	'EXPECTED_BUDGETS',
	'EXPECTED_COMPARISONS',
	'EXPECTED_SUBSAMPLE_SEEDS',
	'HISTORICAL_M1_MODEL_ID',
	'MAE_MODEL_ID',
	'F3VoxelLabelBudgetControlCandidate',
	'F3VoxelLabelBudgetControlConfig',
	'F3VoxelLabelBudgetControlDecisionThresholds',
	'F3VoxelLabelBudgetControlPublishConfig',
	'F3VoxelLabelBudgetControlReferences',
	'f3_lithology_voxel_label_budget_control_config_from_mapping',
]
