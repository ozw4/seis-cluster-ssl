'''Strict configuration and audit for one Volve horizon pretraining recipe arm.

The five-way benchmark pins the Local Barlow Twins lineage to a 100-epoch
Stage 1 plus a 25-epoch Stage 2 continuation, so it cannot express a different
pretraining recipe.  This module keeps every frozen downstream contract of that
benchmark - decoder, tiles, split plan, seed, checkpoint selection and
embedding extraction - and replaces only the pretraining lineage with one
single-stage recipe that the config declares and the audit verifies against its
own checkpoint.  The paired baseline is the same canonical random encoder the
five-way benchmark uses, so recipe arms and the published random column measure
the same quantity.
'''

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.training.random_checkpoint import (
	load_checkpoint_metadata_without_weights,
)
from seis_ssl_cluster.volve.horizon_five_way_config import (
	EXPECTED_MODEL_IDENTITIES,
	FIVE_WAY_RANDOM_SEED,
	LOCAL_BARLOW_TWINS_METHOD,
	VolveHorizonFiveWayConfig,
	VolveHorizonFiveWayModelSource,
)
from seis_ssl_cluster.volve.horizon_five_way_runner import (
	VolveHorizonFiveWayJob,
	inspect_volve_horizon_five_way_job,
	resolve_volve_horizon_five_way_job,
	run_volve_horizon_five_way_job,
)
from seis_ssl_cluster.volve.horizon_five_way_sources import (
	EXPECTED_ENCODER_GEOMETRY,
	RANDOM_CHECKPOINT_STAGE,
	inspect_volve_horizon_five_way_embedding_suite,
)
from seis_ssl_cluster.volve.horizon_frozen import frozen_horizon_config_from_mapping
from seis_ssl_cluster.volve.horizon_layouts import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.volve.horizon_runner import (
	CHECKPOINT_SELECTION_IDS,
	CHECKPOINT_SELECTION_VALIDATION_MAE,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.volve.horizon_data import VolveHorizonData
	from seis_ssl_cluster.volve.horizon_five_way_sources import (
		VolveHorizonFiveWayEmbeddingSuite,
	)
	from seis_ssl_cluster.volve.horizon_frozen import (
		FrozenHorizonPlan,
		FrozenHorizonTrainSettings,
	)
	from seis_ssl_cluster.volve.horizon_tiles import HorizonTileSettings

RECIPE_ARM_BASELINE_MODEL_ID = 'random'
RECIPE_ARM_BENCHMARK_ID = 'local_bt_recipe_arm_v1'
BARLOW_TWINS_TRAINING_STAGE = 'barlow_twins_training'
RECIPE_ARM_CONDITION_COUNT = 2 * len(LAYOUT_IDS) * len(DATA_SIZE_PREFIX)

_TOP_LEVEL_REQUIRED_KEYS = frozenset(
	{
		'paths',
		'dataset',
		'inputs',
		'arm',
		'baseline',
		'outputs',
		'decoder',
		'tiles',
		'train',
	}
)
_TOP_LEVEL_KEYS = _TOP_LEVEL_REQUIRED_KEYS | frozenset(
	{'benchmark_id', 'checkpoint_selection', 'checkpoint_selections'}
)
_RECIPE_KEYS = frozenset(
	{
		'method',
		'local_pairs_per_crop',
		'positive_window_tokens',
		'augmentations',
		'epochs',
		'samples_per_epoch',
		'batch_size',
		'learning_rate',
		'weight_decay',
		'seed',
	}
)


@dataclass(frozen=True)
class VolveHorizonRecipeArmRecipe:
	'''One declared single-stage Local Barlow Twins pretraining recipe.'''

	method: str
	local_pairs_per_crop: int
	positive_window_tokens: tuple[int, int, int] | None
	augmentations: Mapping[str, object]
	epochs: int
	samples_per_epoch: int
	batch_size: int
	learning_rate: float
	weight_decay: float
	seed: int

	@property
	def steps_per_epoch(self) -> int:
		'''Return the optimizer steps one epoch of this recipe performs.'''
		return self.samples_per_epoch // self.batch_size

	@property
	def global_steps(self) -> int:
		'''Return the optimizer steps the finished checkpoint must report.'''
		return self.epochs * self.steps_per_epoch


@dataclass(frozen=True)
class VolveHorizonRecipeArmConfig:
	'''Resolved settings for one recipe arm paired with the random encoder.'''

	artifact_root: Path
	volve_root: Path
	survey_id: str
	canonical_input_metadata: Path
	arm_id: str
	arm_checkpoint: Path
	arm_embeddings_dir: Path
	recipe: VolveHorizonRecipeArmRecipe
	baseline_checkpoint: Path
	baseline_embeddings_dir: Path
	runs_root: Path
	summary_root: Path
	train: FrozenHorizonTrainSettings
	tiles: HorizonTileSettings
	checkpoint_selection: str
	checkpoint_selections: tuple[str, ...]
	benchmark_id: str

	@property
	def model_ids(self) -> tuple[str, ...]:
		'''Return the fixed arm-then-baseline comparison order.'''
		return (self.arm_id, RECIPE_ARM_BASELINE_MODEL_ID)


def volve_horizon_recipe_arm_config_from_mapping(
	config: Mapping[str, object],
) -> VolveHorizonRecipeArmConfig:
	'''Resolve the strict recipe-arm config without touching any artifact.'''
	_validate_top_level_keys(config)
	paths = _required_mapping(config, 'paths', 'config')
	dataset = _required_mapping(config, 'dataset', 'config')
	inputs = _required_mapping(config, 'inputs', 'config')
	arm = _required_mapping(config, 'arm', 'config')
	baseline = _required_mapping(config, 'baseline', 'config')
	outputs = _required_mapping(config, 'outputs', 'config')
	_validate_exact_keys(paths, frozenset({'artifact_root', 'volve_root'}), 'paths')
	_validate_exact_keys(dataset, frozenset({'survey_id'}), 'dataset')
	_validate_exact_keys(inputs, frozenset({'canonical_input_metadata'}), 'inputs')
	_validate_exact_keys(
		arm,
		frozenset({'arm_id', 'checkpoint', 'embeddings_dir', 'recipe'}),
		'arm',
	)
	_validate_exact_keys(
		baseline,
		frozenset({'checkpoint', 'embeddings_dir'}),
		'baseline',
	)
	_validate_exact_keys(
		outputs,
		frozenset({'runs_root', 'summary_root'}),
		'outputs',
	)

	artifact_root = _absolute_path(paths, 'artifact_root', 'paths')
	volve_root = _absolute_path(paths, 'volve_root', 'paths')
	runs_root = _absolute_path(outputs, 'runs_root', 'outputs')
	summary_root = _absolute_path(outputs, 'summary_root', 'outputs')
	if runs_root.resolve(strict=False) == summary_root.resolve(strict=False):
		raise ValueError('outputs.runs_root and outputs.summary_root must differ')
	for key, output_root in (
		('runs_root', runs_root),
		('summary_root', summary_root),
	):
		if not _is_relative_to(output_root, artifact_root):
			raise ValueError(f'outputs.{key} must be below paths.artifact_root')
		if _is_relative_to(output_root, volve_root):
			raise ValueError(
				f'outputs.{key} must not be below public paths.volve_root'
			)

	arm_id = _non_empty_string(arm.get('arm_id'), 'arm.arm_id')
	if arm_id == RECIPE_ARM_BASELINE_MODEL_ID:
		raise ValueError(
			f'arm.arm_id must differ from {RECIPE_ARM_BASELINE_MODEL_ID!r}'
		)
	arm_checkpoint = _absolute_path(arm, 'checkpoint', 'arm')
	arm_embeddings_dir = _absolute_path(arm, 'embeddings_dir', 'arm')
	baseline_checkpoint = _absolute_path(baseline, 'checkpoint', 'baseline')
	baseline_embeddings_dir = _absolute_path(baseline, 'embeddings_dir', 'baseline')
	if arm_checkpoint.resolve(strict=False) == baseline_checkpoint.resolve(
		strict=False
	):
		raise ValueError('arm and baseline checkpoint paths must differ')
	if arm_embeddings_dir.resolve(strict=False) == baseline_embeddings_dir.resolve(
		strict=False
	):
		raise ValueError('arm and baseline embedding directories must differ')

	# Reuse the established Volve decoder/tile/train validators verbatim so the
	# downstream contract stays identical to the five-way benchmark.
	legacy_config = frozen_horizon_config_from_mapping(
		{
			'paths': dict(paths),
			'dataset': dict(dataset),
			'inputs': dict(inputs),
			'embeddings': {
				'pretrained_dir': str(arm_embeddings_dir),
				'random_dir': str(baseline_embeddings_dir),
			},
			'outputs': {'runs_root': str(runs_root)},
			'decoder': _required_mapping(config, 'decoder', 'config'),
			'tiles': _required_mapping(config, 'tiles', 'config'),
			'train': _required_mapping(config, 'train', 'config'),
		}
	)
	checkpoint_selection = _checkpoint_selection(
		config.get('checkpoint_selection', CHECKPOINT_SELECTION_VALIDATION_MAE)
	)
	checkpoint_selections = _checkpoint_selections(
		config.get('checkpoint_selections'),
		primary=checkpoint_selection,
	)
	benchmark_id = (
		_non_empty_string(config.get('benchmark_id'), 'benchmark_id')
		if 'benchmark_id' in config
		else RECIPE_ARM_BENCHMARK_ID
	)
	return VolveHorizonRecipeArmConfig(
		artifact_root=artifact_root,
		volve_root=volve_root,
		survey_id=_non_empty_string(dataset.get('survey_id'), 'dataset.survey_id'),
		canonical_input_metadata=_absolute_path(
			inputs,
			'canonical_input_metadata',
			'inputs',
		),
		arm_id=arm_id,
		arm_checkpoint=arm_checkpoint,
		arm_embeddings_dir=arm_embeddings_dir,
		recipe=_resolve_recipe(arm.get('recipe')),
		baseline_checkpoint=baseline_checkpoint,
		baseline_embeddings_dir=baseline_embeddings_dir,
		runs_root=runs_root,
		summary_root=summary_root,
		train=legacy_config.train,
		tiles=legacy_config.tiles,
		checkpoint_selection=checkpoint_selection,
		checkpoint_selections=checkpoint_selections,
		benchmark_id=benchmark_id,
	)


def as_five_way_config(
	config: VolveHorizonRecipeArmConfig,
) -> VolveHorizonFiveWayConfig:
	'''Express the arm and its baseline in the shared two-source runner shape.'''
	arm_source = VolveHorizonFiveWayModelSource(
		model_id=config.arm_id,
		checkpoint=config.arm_checkpoint,
		embeddings_dir=config.arm_embeddings_dir,
		expected={
			'objective': config.recipe.method,
			'local_pairs_per_crop': config.recipe.local_pairs_per_crop,
			'stratigraphy_pretext': False,
		},
	)
	baseline_source = VolveHorizonFiveWayModelSource(
		model_id=RECIPE_ARM_BASELINE_MODEL_ID,
		checkpoint=config.baseline_checkpoint,
		embeddings_dir=config.baseline_embeddings_dir,
		expected=dict(EXPECTED_MODEL_IDENTITIES[RECIPE_ARM_BASELINE_MODEL_ID]),
	)
	return VolveHorizonFiveWayConfig(
		artifact_root=config.artifact_root,
		volve_root=config.volve_root,
		survey_id=config.survey_id,
		canonical_input_metadata=config.canonical_input_metadata,
		models=(arm_source, baseline_source),
		runs_root=config.runs_root,
		summary_root=config.summary_root,
		train=config.train,
		tiles=config.tiles,
		checkpoint_selection=config.checkpoint_selection,
		checkpoint_selections=config.checkpoint_selections,
		benchmark_id=config.benchmark_id,
	)


def plan_volve_horizon_recipe_arm_sources(
	config: VolveHorizonRecipeArmConfig,
) -> tuple[dict[str, object], ...]:
	'''Return the declared source plan without opening an artifact.'''
	return (
		{
			'model_id': config.arm_id,
			'role': 'recipe_arm',
			'checkpoint': str(config.arm_checkpoint),
			'embeddings_dir': str(config.arm_embeddings_dir),
			'recipe': _recipe_identity(config.recipe),
		},
		{
			'model_id': RECIPE_ARM_BASELINE_MODEL_ID,
			'role': 'baseline',
			'checkpoint': str(config.baseline_checkpoint),
			'embeddings_dir': str(config.baseline_embeddings_dir),
			'random_seed': FIVE_WAY_RANDOM_SEED,
		},
	)


def audit_volve_horizon_recipe_arm_sources(
	config: VolveHorizonRecipeArmConfig,
	*,
	model_ids: Sequence[str] | None = None,
) -> dict[str, object]:
	'''Audit the selected arm and baseline checkpoints against the config.

	Selecting one model audits only that source, so the baseline cells can run
	before the arm checkpoint exists.
	'''
	selected = _normalize_model_ids(config, model_ids)
	reports: dict[str, dict[str, object]] = {}
	if config.arm_id in selected:
		reports[config.arm_id] = _audit_recipe_arm_checkpoint(config)
	if RECIPE_ARM_BASELINE_MODEL_ID in selected:
		reports[RECIPE_ARM_BASELINE_MODEL_ID] = _audit_baseline_checkpoint(config)
	return {
		'schema_version': 1,
		'survey_id': config.survey_id,
		'model_order': list(selected),
		'sources': [reports[model_id] for model_id in selected],
	}


def inspect_volve_horizon_recipe_arm_embedding_suite(
	config: VolveHorizonRecipeArmConfig,
	*,
	source_audit: Mapping[str, object] | None = None,
	model_ids: Sequence[str] | None = None,
) -> VolveHorizonFiveWayEmbeddingSuite:
	'''Inspect the selected embeddings on the shared frozen support.'''
	selected = _normalize_model_ids(config, model_ids)
	audit = (
		audit_volve_horizon_recipe_arm_sources(config, model_ids=selected)
		if source_audit is None
		else source_audit
	)
	return inspect_volve_horizon_five_way_embedding_suite(
		as_five_way_config(config),
		source_audit=_select_audit(audit, selected),
		model_ids=selected,
	)


def plan_volve_horizon_recipe_arm_jobs(
	config: VolveHorizonRecipeArmConfig,
	*,
	model_ids: Sequence[str] | None = None,
) -> tuple[tuple[str, str, str], ...]:
	'''Enumerate every cell of the selected model subset.'''
	selected = _normalize_model_ids(config, model_ids)
	jobs = tuple(
		(model_id, layout_id, data_size)
		for model_id in selected
		for layout_id in LAYOUT_IDS
		for data_size in DATA_SIZE_PREFIX
	)
	expected_count = len(selected) * len(LAYOUT_IDS) * len(DATA_SIZE_PREFIX)
	if len(jobs) != expected_count or len(set(jobs)) != len(jobs):
		raise RuntimeError('Volve horizon recipe-arm cells must be unique')
	return jobs


def resolve_volve_horizon_recipe_arm_job(
	config: VolveHorizonRecipeArmConfig,
	*,
	model: str,
	layout: str,
	size: str,
) -> VolveHorizonFiveWayJob:
	'''Resolve one fixed cell without opening any source artifact.'''
	_normalize_model_ids(config, (model,))
	return resolve_volve_horizon_five_way_job(
		as_five_way_config(config),
		model=model,
		layout=layout,
		size=size,
	)


def inspect_volve_horizon_recipe_arm_job(
	config: VolveHorizonRecipeArmConfig,
	job: VolveHorizonFiveWayJob,
	*,
	layout_config: str | Path,
	data: VolveHorizonData | None = None,
	embedding_suite: VolveHorizonFiveWayEmbeddingSuite | None = None,
) -> FrozenHorizonPlan:
	'''Run the recipe-arm preflight and build one frozen decoder plan.'''
	suite = (
		inspect_volve_horizon_recipe_arm_embedding_suite(
			config,
			model_ids=(job.model.model_id,),
		)
		if embedding_suite is None
		else embedding_suite
	)
	return inspect_volve_horizon_five_way_job(
		job,
		layout_config=layout_config,
		data=data,
		embedding_suite=suite,
	)


def run_volve_horizon_recipe_arm_job(
	plan: FrozenHorizonPlan,
	*,
	device: str = 'auto',
	max_steps: int | None = None,
	resume: str | Path | None = None,
) -> Path | None:
	'''Execute one preflighted cell through the shared frozen decoder loop.'''
	return run_volve_horizon_five_way_job(
		plan,
		device=device,
		max_steps=max_steps,
		resume=resume,
	)


def _audit_recipe_arm_checkpoint(  # noqa: C901
	config: VolveHorizonRecipeArmConfig,
) -> dict[str, object]:
	recipe = config.recipe
	label = f'{config.arm_id} checkpoint'
	checkpoint_sha = _checkpoint_sha256(label, config.arm_checkpoint)
	payload = _load_checkpoint(label, config.arm_checkpoint)
	training_state = _required_mapping(payload, 'training_state', label)
	if training_state.get('stage') != BARLOW_TWINS_TRAINING_STAGE:
		raise ValueError(
			f'{label} training_state.stage must equal '
			f'{BARLOW_TWINS_TRAINING_STAGE!r}'
		)
	if training_state.get('completed_epoch') is not True:
		raise ValueError(f'{label} must be a completed pretraining epoch')
	if payload.get('pretraining_method') != recipe.method:
		raise ValueError(f'{label} pretraining_method must equal {recipe.method!r}')
	if payload.get('resume_count') != 0:
		raise ValueError(f'{label} must be a single uninterrupted run')
	if payload.get('epoch') != recipe.epochs:
		raise ValueError(f'{label} epoch must equal {recipe.epochs}')
	if payload.get('global_step') != recipe.global_steps:
		raise ValueError(f'{label} global_step must equal {recipe.global_steps}')
	checkpoint_config = _required_mapping(payload, 'config', label)
	if 'continuation' in checkpoint_config:
		raise ValueError(f'{label} must be single-stage without a continuation')
	_validate_encoder_geometry(
		label,
		_required_mapping(checkpoint_config, 'model', label),
	)
	barlow_twins = _required_mapping(checkpoint_config, 'barlow_twins', label)
	if barlow_twins.get('method') != recipe.method:
		raise ValueError(f'{label} barlow_twins.method must equal {recipe.method!r}')
	if barlow_twins.get('local_pairs_per_crop') != recipe.local_pairs_per_crop:
		raise ValueError(
			f'{label} barlow_twins.local_pairs_per_crop must equal '
			f'{recipe.local_pairs_per_crop}'
		)
	recorded_window = barlow_twins.get('positive_window_tokens')
	window = (
		None if recorded_window is None else _positive_triplet(recorded_window, label)
	)
	if window != recipe.positive_window_tokens:
		raise ValueError(
			f'{label} barlow_twins.positive_window_tokens must equal '
			f'{recipe.positive_window_tokens!r}'
		)
	_validate_declared_mapping(
		f'{label} augmentations',
		_required_mapping(checkpoint_config, 'augmentations', label),
		recipe.augmentations,
	)
	_validate_declared_mapping(
		f'{label} train',
		_required_mapping(checkpoint_config, 'train', label),
		{
			'epochs': recipe.epochs,
			'samples_per_epoch': recipe.samples_per_epoch,
			'batch_size': recipe.batch_size,
			'lr': recipe.learning_rate,
			'weight_decay': recipe.weight_decay,
			'seed': recipe.seed,
		},
		subset=True,
	)
	return {
		'model_id': config.arm_id,
		'role': 'recipe_arm',
		'checkpoint': str(config.arm_checkpoint),
		'checkpoint_sha256': checkpoint_sha,
		'objective': recipe.method,
		'recipe': _recipe_identity(recipe),
	}


def _audit_baseline_checkpoint(
	config: VolveHorizonRecipeArmConfig,
) -> dict[str, object]:
	label = 'random checkpoint'
	checkpoint_sha = _checkpoint_sha256(label, config.baseline_checkpoint)
	payload = _load_checkpoint(label, config.baseline_checkpoint)
	metadata = _required_mapping(payload, 'metadata', label)
	for key, expected in (
		('random_encoder_baseline', True),
		('pretrained_weights_loaded', False),
		('seed', FIVE_WAY_RANDOM_SEED),
	):
		if metadata.get(key) != expected:
			raise ValueError(f'{label} metadata.{key} must equal {expected!r}')
	training_state = _required_mapping(payload, 'training_state', label)
	if training_state.get('stage') != RANDOM_CHECKPOINT_STAGE:
		raise ValueError(
			f'{label} training_state.stage must equal {RANDOM_CHECKPOINT_STAGE!r}'
		)
	if training_state.get('checkpoint_kind') != 'random_init':
		raise ValueError(f'{label} must have checkpoint_kind random_init')
	_validate_encoder_geometry(
		label,
		_required_mapping(_required_mapping(payload, 'config', label), 'model', label),
	)
	return {
		'model_id': RECIPE_ARM_BASELINE_MODEL_ID,
		'role': 'baseline',
		'checkpoint': str(config.baseline_checkpoint),
		'checkpoint_sha256': checkpoint_sha,
		'objective': EXPECTED_MODEL_IDENTITIES[RECIPE_ARM_BASELINE_MODEL_ID][
			'objective'
		],
		'random_seed': FIVE_WAY_RANDOM_SEED,
	}


def _recipe_identity(recipe: VolveHorizonRecipeArmRecipe) -> dict[str, object]:
	return {
		'method': recipe.method,
		'local_pairs_per_crop': recipe.local_pairs_per_crop,
		'positive_window_tokens': (
			None
			if recipe.positive_window_tokens is None
			else list(recipe.positive_window_tokens)
		),
		'augmentations': dict(recipe.augmentations),
		'stages': 1,
		'epochs': recipe.epochs,
		'samples_per_epoch': recipe.samples_per_epoch,
		'batch_size': recipe.batch_size,
		'steps_per_epoch': recipe.steps_per_epoch,
		'global_steps': recipe.global_steps,
		'learning_rate': recipe.learning_rate,
		'weight_decay': recipe.weight_decay,
		'seed': recipe.seed,
	}


def _resolve_recipe(value: object) -> VolveHorizonRecipeArmRecipe:
	recipe = _required_mapping({'recipe': value}, 'recipe', 'arm')
	_validate_exact_keys(recipe, _RECIPE_KEYS, 'arm.recipe')
	method = _non_empty_string(recipe.get('method'), 'arm.recipe.method')
	if method != LOCAL_BARLOW_TWINS_METHOD:
		raise ValueError(
			f'arm.recipe.method must equal {LOCAL_BARLOW_TWINS_METHOD!r}'
		)
	augmentations = _required_mapping(recipe, 'augmentations', 'arm.recipe')
	if not augmentations:
		raise ValueError('arm.recipe.augmentations must declare the view contract')
	window_value = recipe.get('positive_window_tokens')
	batch_size = _positive_int(recipe.get('batch_size'), 'arm.recipe.batch_size')
	samples_per_epoch = _positive_int(
		recipe.get('samples_per_epoch'),
		'arm.recipe.samples_per_epoch',
	)
	if samples_per_epoch % batch_size != 0:
		raise ValueError(
			'arm.recipe.samples_per_epoch must be a whole number of batches'
		)
	return VolveHorizonRecipeArmRecipe(
		method=method,
		local_pairs_per_crop=_positive_int(
			recipe.get('local_pairs_per_crop'),
			'arm.recipe.local_pairs_per_crop',
		),
		positive_window_tokens=(
			None
			if window_value is None
			else _positive_triplet(window_value, 'arm.recipe.positive_window_tokens')
		),
		augmentations=dict(augmentations),
		epochs=_positive_int(recipe.get('epochs'), 'arm.recipe.epochs'),
		samples_per_epoch=samples_per_epoch,
		batch_size=batch_size,
		learning_rate=_positive_float(
			recipe.get('learning_rate'),
			'arm.recipe.learning_rate',
		),
		weight_decay=_nonnegative_float(
			recipe.get('weight_decay'),
			'arm.recipe.weight_decay',
		),
		seed=_nonnegative_int(recipe.get('seed'), 'arm.recipe.seed'),
	)


def _normalize_model_ids(
	config: VolveHorizonRecipeArmConfig,
	model_ids: Sequence[str] | None,
) -> tuple[str, ...]:
	if model_ids is None:
		return config.model_ids
	if isinstance(model_ids, str | bytes):
		raise TypeError('recipe-arm model selection must be a sequence of model IDs')
	selected = tuple(model_ids)
	if not selected or len(selected) != len(set(selected)):
		raise ValueError('recipe-arm model selection must be non-empty and unique')
	for model_id in selected:
		if model_id not in config.model_ids:
			raise ValueError(
				f'unknown Volve horizon recipe-arm model: {model_id!r}; '
				f'expected one of {list(config.model_ids)!r}'
			)
	return selected


def _select_audit(
	audit: Mapping[str, object],
	model_ids: tuple[str, ...],
) -> dict[str, object]:
	value = audit.get('sources')
	if not isinstance(value, list) or not all(
		isinstance(item, Mapping) for item in value
	):
		raise TypeError('recipe-arm source report sources must be a list of mappings')
	sources = cast('list[Mapping[str, object]]', value)
	by_id = {str(source.get('model_id')): source for source in sources}
	missing = [model_id for model_id in model_ids if model_id not in by_id]
	if missing:
		raise ValueError(f'recipe-arm source report is missing models: {missing!r}')
	return {
		**dict(audit),
		'model_order': list(model_ids),
		'sources': [by_id[model_id] for model_id in model_ids],
	}


def _validate_declared_mapping(
	label: str,
	actual: Mapping[str, object],
	expected: Mapping[str, object],
	*,
	subset: bool = False,
) -> None:
	if not subset and set(actual) != set(expected):
		missing = sorted(set(expected) - set(actual))
		extra = sorted(set(actual) - set(expected))
		raise ValueError(
			f'{label} keys differ from the declared recipe; '
			f'missing={missing!r}, extra={extra!r}'
		)
	for key, wanted in expected.items():
		if key not in actual:
			raise ValueError(f'{label} must declare {key!r}')
		if not _same_value(actual[key], wanted):
			raise ValueError(
				f'{label}.{key} must equal {wanted!r}; got {actual[key]!r}'
			)


def _same_value(actual: object, expected: object) -> bool:
	if isinstance(actual, bool) or isinstance(expected, bool):
		return actual is expected
	if isinstance(actual, int | float) and isinstance(expected, int | float):
		return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12)
	if isinstance(actual, Sequence) and isinstance(expected, Sequence):
		if isinstance(actual, str | bytes) or isinstance(expected, str | bytes):
			return actual == expected
		return len(actual) == len(expected) and all(
			_same_value(left, right)
			for left, right in zip(actual, expected, strict=True)
		)
	return bool(actual == expected)


def _validate_encoder_geometry(label: str, model: Mapping[str, object]) -> None:
	for key, expected in EXPECTED_ENCODER_GEOMETRY.items():
		if not _same_value(model.get(key), expected):
			raise ValueError(f'{label} model.{key} must equal {expected!r}')


def _validate_top_level_keys(config: Mapping[str, object]) -> None:
	missing = sorted(set(_TOP_LEVEL_REQUIRED_KEYS) - set(config))
	extra = sorted(set(config) - set(_TOP_LEVEL_KEYS))
	if missing or extra:
		raise ValueError(
			'config keys differ from the fixed contract; '
			f'missing={missing!r}, extra={extra!r}'
		)


def _validate_exact_keys(
	value: Mapping[str, object],
	expected: frozenset[str],
	label: str,
) -> None:
	missing = sorted(expected - set(value))
	extra = sorted(set(value) - expected)
	if missing or extra:
		raise ValueError(
			f'{label} keys differ from the fixed contract; '
			f'missing={missing!r}, extra={extra!r}'
		)


def _checkpoint_selection(value: object) -> str:
	if not isinstance(value, str):
		raise TypeError('checkpoint_selection must be a string')
	if value not in CHECKPOINT_SELECTION_IDS:
		raise ValueError(f'unknown horizon checkpoint selection: {value!r}')
	return value


def _checkpoint_selections(
	value: object,
	*,
	primary: str,
) -> tuple[str, ...]:
	if value is None:
		return (primary,)
	if not isinstance(value, list | tuple) or isinstance(value, str | bytes):
		raise TypeError('checkpoint_selections must be a sequence of strings')
	selections = tuple(_checkpoint_selection(item) for item in value)
	if not selections:
		raise ValueError('checkpoint_selections must not be empty')
	if len(selections) != len(set(selections)):
		raise ValueError('checkpoint_selections must be unique')
	if primary not in selections:
		raise ValueError(
			'checkpoint_selections must include the primary checkpoint_selection'
		)
	return selections


def _load_checkpoint(label: str, path: Path) -> Mapping[str, object]:
	try:
		return load_checkpoint_metadata_without_weights(path)
	except FileNotFoundError as error:
		raise FileNotFoundError(f'missing {label}: {path}') from error


def _checkpoint_sha256(label: str, path: Path) -> str:
	if not path.is_file():
		raise FileNotFoundError(f'missing {label}: {path}')
	return file_sha256(path)


def _required_mapping(
	value: Mapping[str, object],
	key: str,
	label: str,
) -> Mapping[str, object]:
	item = value.get(key)
	if not isinstance(item, Mapping):
		raise TypeError(f'{label}.{key} must be a mapping')
	return cast('Mapping[str, object]', item)


def _non_empty_string(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		raise ValueError(f'{label} must be a non-empty string')
	return value


def _absolute_path(value: Mapping[str, object], key: str, label: str) -> Path:
	raw = value.get(key)
	if not isinstance(raw, str) or not raw:
		raise ValueError(f'{label}.{key} must be a non-empty string')
	path = Path(raw)
	if not path.is_absolute():
		raise ValueError(f'{label}.{key} must be an absolute path')
	return path


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


def _positive_int(value: object, label: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
		raise ValueError(f'{label} must be a positive integer')
	return int(value)


def _nonnegative_int(value: object, label: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int) or value < 0:
		raise ValueError(f'{label} must be a non-negative integer')
	return int(value)


def _positive_float(value: object, label: str) -> float:
	if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
		raise ValueError(f'{label} must be a positive number')
	return float(value)


def _nonnegative_float(value: object, label: str) -> float:
	if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
		raise ValueError(f'{label} must be a non-negative number')
	return float(value)


def _positive_triplet(value: object, label: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 3
	):
		raise ValueError(f'{label} must be a length-3 positive integer sequence')
	triplet = tuple(_positive_int(item, label) for item in cast('Sequence[Any]', value))
	return (triplet[0], triplet[1], triplet[2])


__all__ = [
	'BARLOW_TWINS_TRAINING_STAGE',
	'RECIPE_ARM_BASELINE_MODEL_ID',
	'RECIPE_ARM_BENCHMARK_ID',
	'RECIPE_ARM_CONDITION_COUNT',
	'VolveHorizonRecipeArmConfig',
	'VolveHorizonRecipeArmRecipe',
	'as_five_way_config',
	'audit_volve_horizon_recipe_arm_sources',
	'inspect_volve_horizon_recipe_arm_embedding_suite',
	'inspect_volve_horizon_recipe_arm_job',
	'plan_volve_horizon_recipe_arm_jobs',
	'plan_volve_horizon_recipe_arm_sources',
	'resolve_volve_horizon_recipe_arm_job',
	'run_volve_horizon_recipe_arm_job',
	'volve_horizon_recipe_arm_config_from_mapping',
]
