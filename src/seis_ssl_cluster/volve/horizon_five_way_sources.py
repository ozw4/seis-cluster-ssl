'''Read-only checkpoint and embedding audit for Volve horizon five-way.'''

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from seis_ssl_cluster.embedding.writer import (
	EmbeddingOutputPaths,
	file_sha256,
	output_paths,
)
from seis_ssl_cluster.training.random_checkpoint import (
	load_checkpoint_metadata_without_weights,
)
from seis_ssl_cluster.volve.horizon_data import array_sha256
from seis_ssl_cluster.volve.horizon_five_way_config import (
	FIVE_WAY_HMM_K,
	FIVE_WAY_RANDOM_SEED,
	FIVE_WAY_STAGE1_EPOCHS,
	FIVE_WAY_STAGE2_EPOCHS,
	FIVE_WAY_UNFREEZE_TOP_BLOCKS,
	LOCAL_BARLOW_TWINS_METHOD,
	LOCAL_BARLOW_TWINS_PAIRS_PER_CROP,
	MAE_OBJECTIVE,
	RANDOM_OBJECTIVE,
)
from seis_ssl_cluster.volve.horizon_tiles import (
	HORIZON_WINDOW_START,
	HORIZON_WINDOW_STOP,
	frozen_survey_output_valid_mask,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.volve.horizon_five_way_config import (
		VolveHorizonFiveWayConfig,
		VolveHorizonFiveWayModelSource,
	)

VOLVE_PRETRAIN_SAMPLES_PER_EPOCH = 10_000
VOLVE_PRETRAIN_BATCH_SIZE = 4
VOLVE_LOCAL_BT_STAGE1_BATCH_SIZE = 16
FIVE_WAY_STAGE1_GLOBAL_STEPS = 250_000
FIVE_WAY_LOCAL_BT_STAGE1_GLOBAL_STEPS = 62_500
FIVE_WAY_STAGE2_GLOBAL_STEPS = 62_500
STRAT_HMM_PRETEXT_METHOD = 'strat_hmm_pretext'
RANDOM_CHECKPOINT_STAGE = 'create_random_mae_checkpoint'
EXPECTED_ENCODER_GEOMETRY: Mapping[str, object] = {
	'in_channels': 1,
	'patch_size': [8, 8, 8],
	'encoder_dim': 384,
	'encoder_depth': 8,
	'encoder_heads': 6,
}
EXPECTED_EXTRACTION_CONTRACT: Mapping[str, object] = {
	'patch_size': [8, 8, 8],
	'window_size': [128, 128, 128],
	'overlap': [64, 64, 64],
	'output_dtype': 'float16',
	'min_token_valid_fraction': 1.0,
}
SHARED_EMBEDDING_METADATA_KEYS = (
	'survey_id',
	'source_amplitude_path',
	'source_valid_mask_path',
	'volume_shape_xyz',
	'model_geometry',
	'patch_size',
	'token_grid_shape',
	'window_size',
	'overlap',
	'output_dtype',
	'precision',
	'min_token_valid_fraction',
	'normalization_stats_path',
	'normalized_clip_abs',
	'amplitude_agc',
	'finite_check_mode',
	'preprocessing',
	'preprocessing_cache',
	'zero_mask',
)
_MASK_COMPARE_CHUNK = 1 << 22
_MAX_LINEAGE_DEPTH = 4


@dataclass(frozen=True)
class VolveHorizonFiveWayEmbeddingSource:
	'''Validated files and identity for one model embedding.'''

	model_id: str
	paths: EmbeddingOutputPaths
	metadata: Mapping[str, object]
	checkpoint_identity: Mapping[str, object]
	embeddings_sha256: str
	metadata_sha256: str
	valid_tokens_sha256: str


@dataclass(frozen=True)
class VolveHorizonFiveWayEmbeddingSuite:
	'''Common geometry and per-model files of the five embeddings.'''

	sources: Mapping[str, VolveHorizonFiveWayEmbeddingSource]
	volume_shape_xyz: tuple[int, int, int]
	token_grid_shape_xyz: tuple[int, int, int]
	embedding_shape: tuple[int, int, int, int]
	embedding_dim: int
	valid_tokens_sha256: str
	model_valid_lateral_mask: np.ndarray
	model_valid_lateral_mask_sha256: str
	canonical_identity: Mapping[str, object]

	def source_by_id(self, model_id: str) -> VolveHorizonFiveWayEmbeddingSource:
		'''Return one inspected embedding source.'''
		try:
			return self.sources[model_id]
		except KeyError as error:
			raise ValueError(
				f'unknown inspected embedding model: {model_id!r}'
			) from error


def plan_volve_horizon_five_way_sources(
	config: VolveHorizonFiveWayConfig,
) -> tuple[dict[str, object], ...]:
	'''Return the fixed checkpoint plan without opening an artifact.'''
	return tuple(
		{
			'model_id': model.model_id,
			'checkpoint': str(model.checkpoint),
			'expected': dict(model.expected),
		}
		for model in config.models
	)


def plan_volve_horizon_five_way_embeddings(
	config: VolveHorizonFiveWayConfig,
) -> tuple[dict[str, object], ...]:
	'''Return deterministic embedding paths without requiring them to exist.'''
	rows: list[dict[str, object]] = []
	for model in config.models:
		paths = output_paths(model.embeddings_dir, config.survey_id)
		rows.append(
			{
				'model_id': model.model_id,
				'embeddings_dir': str(model.embeddings_dir),
				'embeddings': str(paths.embeddings),
				'valid_tokens': str(paths.valid_tokens),
				'metadata': str(paths.metadata),
				'checkpoint': str(model.checkpoint),
			}
		)
	return tuple(rows)


def audit_volve_horizon_five_way_sources(
	config: VolveHorizonFiveWayConfig,
) -> dict[str, object]:
	'''Audit the five checkpoint identities and their bounded ancestry.'''
	sources: list[dict[str, object]] = []
	for model in config.models:
		if model.model_id == 'random':
			if not sources:
				raise RuntimeError(
					'random source must follow the learned model sources'
				)
			mae_parent = _parent_identity(sources[0])
			source = _audit_random_checkpoint(model, mae_parent=mae_parent)
		else:
			source = _audit_learned_checkpoint(
				model,
				survey_id=config.survey_id,
			)
		sources.append(source)

	mae_parent = _parent_identity(sources[0])
	mae_hmm_parent = _parent_identity(sources[1])
	local_parent = _parent_identity(sources[2])
	local_hmm_parent = _parent_identity(sources[3])
	if mae_parent != mae_hmm_parent:
		raise ValueError('mae_hmm_k6 must use the same stage-1 parent as mae')
	if local_parent != local_hmm_parent:
		raise ValueError(
			'local_barlow_twins_hmm_k6 must use the same stage-1 parent as '
			'local_barlow_twins'
		)
	if mae_parent == local_parent:
		raise ValueError('MAE and Local Barlow Twins must use distinct stage-1 sources')
	return {
		'schema_version': 1,
		'survey_id': config.survey_id,
		'model_order': list(config.model_ids),
		'sources': sources,
	}


def inspect_volve_horizon_five_way_embedding_suite(  # noqa: PLR0915
	config: VolveHorizonFiveWayConfig,
	*,
	source_audit: Mapping[str, object] | None = None,
) -> VolveHorizonFiveWayEmbeddingSuite:
	'''Inspect five memory-mapped embedding artifacts on identical support.'''
	audit = (
		audit_volve_horizon_five_way_sources(config)
		if source_audit is None
		else source_audit
	)
	checkpoint_reports = _checkpoint_reports_from_audit(config, audit)
	reference_metadata: Mapping[str, object] | None = None
	reference_paths: EmbeddingOutputPaths | None = None
	reference_valid: np.ndarray | None = None
	reference_valid_sha256: str | None = None
	resolved_sources: dict[str, VolveHorizonFiveWayEmbeddingSource] = {}
	volume_shape: tuple[int, int, int] | None = None
	token_grid: tuple[int, int, int] | None = None
	embedding_shape: tuple[int, int, int, int] | None = None
	embedding_dim: int | None = None

	for model in config.models:
		paths = output_paths(model.embeddings_dir, config.survey_id)
		for path in (paths.embeddings, paths.valid_tokens, paths.metadata):
			if not path.is_file():
				raise FileNotFoundError(
					f'{model.model_id} embedding source is missing: {path}'
				)
		metadata = _read_json(paths.metadata, f'{model.model_id} embedding metadata')
		embeddings_sha256 = file_sha256(paths.embeddings)
		metadata_sha256 = file_sha256(paths.metadata)
		valid_tokens_sha256 = file_sha256(paths.valid_tokens)
		_validate_extraction_metadata(model.model_id, metadata)
		if reference_metadata is None:
			reference_metadata = metadata
			reference_paths = paths
		else:
			_validate_shared_embedding_identity(
				model.model_id,
				metadata,
				reference_metadata,
			)
		model_volume = _positive_triplet(
			metadata.get('volume_shape_xyz'),
			f'{model.model_id} volume_shape_xyz',
		)
		model_grid = _positive_triplet(
			metadata.get('token_grid_shape'),
			f'{model.model_id} token_grid_shape',
		)
		model_embedding_shape, model_embedding_dim, valid_tokens = _inspect_arrays(
			model.model_id,
			paths,
			token_grid=model_grid,
			metadata=metadata,
		)
		if reference_valid is None:
			reference_valid = valid_tokens
			reference_valid_sha256 = valid_tokens_sha256
		else:
			_validate_masks_identical(
				model.model_id,
				reference_valid,
				valid_tokens,
			)
		if volume_shape is None:
			volume_shape = model_volume
			token_grid = model_grid
			embedding_shape = model_embedding_shape
			embedding_dim = model_embedding_dim
		elif (
			model_volume != volume_shape
			or model_grid != token_grid
			or model_embedding_shape != embedding_shape
			or model_embedding_dim != embedding_dim
		):
			raise ValueError(
				f'{model.model_id} embedding geometry differs from the mae source'
			)
		checkpoint_report = checkpoint_reports[model.model_id]
		_validate_embedding_checkpoint_identity(model, metadata, checkpoint_report)
		_validate_embedding_objective(model, metadata)
		resolved_sources[model.model_id] = VolveHorizonFiveWayEmbeddingSource(
			model_id=model.model_id,
			paths=paths,
			metadata=MappingProxyType(dict(metadata)),
			checkpoint_identity=MappingProxyType(dict(checkpoint_report)),
			embeddings_sha256=embeddings_sha256,
			metadata_sha256=metadata_sha256,
			valid_tokens_sha256=valid_tokens_sha256,
		)

	if (
		reference_metadata is None
		or reference_paths is None
		or reference_valid is None
		or reference_valid_sha256 is None
		or volume_shape is None
		or token_grid is None
		or embedding_shape is None
		or embedding_dim is None
	):
		raise RuntimeError('the five-way embedding suite is empty')
	_validate_grid_geometry(volume_shape, token_grid)
	canonical_identity = _inspect_canonical_identity(
		config,
		embedding_metadata=reference_metadata,
		volume_shape=volume_shape,
	)
	_validate_valid_support(
		reference_valid,
		embedding_metadata=reference_metadata,
		volume_shape=volume_shape,
		token_grid=token_grid,
	)
	z_start = HORIZON_WINDOW_START // EXPECTED_ENCODER_GEOMETRY['patch_size'][2]
	z_stop = HORIZON_WINDOW_STOP // EXPECTED_ENCODER_GEOMETRY['patch_size'][2]
	window_valid = reference_valid[:, :, z_start:z_stop]
	if window_valid.shape[2] != z_stop - z_start:
		raise ValueError('valid-token mask does not cover the fixed horizon window')
	model_valid_lateral = frozen_survey_output_valid_mask(
		window_valid,
		replace(config.tiles, lateral_shape_xy=volume_shape[:2]),
	)
	return VolveHorizonFiveWayEmbeddingSuite(
		sources=MappingProxyType(resolved_sources),
		volume_shape_xyz=volume_shape,
		token_grid_shape_xyz=token_grid,
		embedding_shape=embedding_shape,
		embedding_dim=embedding_dim,
		valid_tokens_sha256=reference_valid_sha256,
		model_valid_lateral_mask=model_valid_lateral,
		model_valid_lateral_mask_sha256=array_sha256(model_valid_lateral),
		canonical_identity=MappingProxyType(dict(canonical_identity)),
	)


def _audit_learned_checkpoint(
	model: VolveHorizonFiveWayModelSource,
	*,
	survey_id: str,
) -> dict[str, object]:
	checkpoint_sha = _checkpoint_sha256(model.model_id, model.checkpoint)
	payload = _load_checkpoint(model.model_id, model.checkpoint)
	config = _required_mapping(payload, 'config', f'{model.model_id} checkpoint')
	_validate_stage2_budget(model.model_id, payload, config=config)
	is_hmm = bool(model.expected['stratigraphy_pretext'])
	if is_hmm:
		parent_path, pseudo_identity = _validate_hmm_checkpoint(
			model,
			payload,
			base_config=config,
		)
	else:
		parent_path = _validate_plain_continuation(model, payload, config=config)
		pseudo_identity = None
	parent_sha = _validate_stage1_parent(
		model.model_id,
		parent_path,
		expected_objective=str(model.expected['objective']),
	)
	_validate_recorded_parent_sha(
		model.model_id,
		payload,
		parent_path=parent_path,
		parent_sha=parent_sha,
		required=True,
	)
	if is_hmm:
		pseudo_identity = _validate_pseudo_target_lineage(
			model.model_id,
			cast('Mapping[str, object]', pseudo_identity),
			checkpoint_payload=payload,
			parent_path=parent_path,
			parent_sha=parent_sha,
			survey_id=survey_id,
		)
	return {
		'model_id': model.model_id,
		'checkpoint': str(model.checkpoint),
		'checkpoint_sha256': checkpoint_sha,
		'parent_checkpoint': str(parent_path),
		'parent_checkpoint_sha256': parent_sha,
		'objective': model.expected['objective'],
		'hmm_k': FIVE_WAY_HMM_K if is_hmm else None,
		'pseudo_targets': pseudo_identity,
		'stage_2': {
			'epochs': FIVE_WAY_STAGE2_EPOCHS,
			'global_steps': FIVE_WAY_STAGE2_GLOBAL_STEPS,
			'unfreeze_top_blocks': FIVE_WAY_UNFREEZE_TOP_BLOCKS,
		},
	}


def _audit_random_checkpoint(  # noqa: C901
	model: VolveHorizonFiveWayModelSource,
	*,
	mae_parent: tuple[Path, str],
) -> dict[str, object]:
	checkpoint_sha = _checkpoint_sha256(model.model_id, model.checkpoint)
	payload = _load_checkpoint(model.model_id, model.checkpoint)
	metadata = _required_mapping(payload, 'metadata', 'random checkpoint')
	for key, expected in (
		('random_encoder_baseline', True),
		('pretrained_weights_loaded', False),
		('seed', FIVE_WAY_RANDOM_SEED),
	):
		if metadata.get(key) != expected:
			raise ValueError(
				f'random checkpoint metadata.{key} must equal {expected!r}'
			)
	training_state = _required_mapping(
		payload,
		'training_state',
		'random checkpoint',
	)
	if training_state.get('stage') != RANDOM_CHECKPOINT_STAGE:
		raise ValueError(
			f'random checkpoint training_state.stage must equal '
			f'{RANDOM_CHECKPOINT_STAGE!r}'
		)
	if training_state.get('checkpoint_kind') != 'random_init':
		raise ValueError('random checkpoint must have checkpoint_kind random_init')
	for key in ('epoch', 'global_step'):
		if payload.get(key) != 0:
			raise ValueError(f'random checkpoint {key} must equal 0')
	for forbidden in ('continuation_lineage', 'stratigraphy_config'):
		if forbidden in payload:
			raise ValueError(f'random checkpoint must not contain {forbidden}')
	config = _required_mapping(payload, 'config', 'random checkpoint')
	_validate_encoder_geometry('random', config)
	mae_parent_path, mae_parent_sha = mae_parent
	mae_payload = _load_checkpoint('mae stage-1 parent', mae_parent_path)
	mae_config = _required_mapping(mae_payload, 'config', 'mae stage-1 parent')
	if _model_geometry(config) != _model_geometry(mae_config):
		raise ValueError('random checkpoint encoder geometry differs from MAE stage 1')
	reference = _required_string(
		metadata.get('reference_checkpoint'),
		'random checkpoint metadata.reference_checkpoint',
	)
	if Path(reference).resolve(strict=False) != mae_parent_path.resolve(strict=False):
		raise ValueError('random checkpoint must reference the MAE stage-1 source')
	recorded_sha = metadata.get('reference_checkpoint_sha256')
	if recorded_sha is not None and recorded_sha != mae_parent_sha:
		raise ValueError('random checkpoint reference checkpoint SHA-256 mismatch')
	return {
		'model_id': model.model_id,
		'checkpoint': str(model.checkpoint),
		'checkpoint_sha256': checkpoint_sha,
		'parent_checkpoint': str(mae_parent_path),
		'parent_checkpoint_sha256': mae_parent_sha,
		'objective': RANDOM_OBJECTIVE,
		'hmm_k': None,
		'pseudo_targets': None,
		'stage_2': None,
	}


def _validate_plain_continuation(
	model: VolveHorizonFiveWayModelSource,
	payload: Mapping[str, object],
	*,
	config: Mapping[str, object],
) -> Path:
	_validate_base_objective(model.model_id, config, str(model.expected['objective']))
	continuation = _required_mapping(
		config,
		'continuation',
		f'{model.model_id} checkpoint config',
	)
	if continuation.get('unfreeze_top_blocks') != FIVE_WAY_UNFREEZE_TOP_BLOCKS:
		raise ValueError(
			f'{model.model_id} continuation.unfreeze_top_blocks must equal '
			f'{FIVE_WAY_UNFREEZE_TOP_BLOCKS}'
		)
	training_state = _required_mapping(
		payload,
		'training_state',
		f'{model.model_id} checkpoint',
	)
	if model.model_id == 'mae':
		if training_state.get('stage') != 'train_amp_mae':
			raise ValueError(
				'mae checkpoint training_state.stage must be train_amp_mae'
			)
		if training_state.get('checkpoint_kind') != 'epoch':
			raise ValueError('mae checkpoint must be a completed epoch checkpoint')
	else:
		if training_state.get('stage') != 'barlow_twins_training':
			raise ValueError(
				f'{model.model_id} checkpoint stage must be barlow_twins_training'
			)
		if training_state.get('completed_epoch') is not True:
			raise ValueError(f'{model.model_id} checkpoint must complete its epoch')
	return _lineage_path(
		model.model_id,
		continuation.get('init_checkpoint'),
		'continuation.init_checkpoint',
	)


def _validate_hmm_checkpoint(
	model: VolveHorizonFiveWayModelSource,
	payload: Mapping[str, object],
	*,
	base_config: Mapping[str, object],
) -> tuple[Path, Mapping[str, object]]:
	_validate_base_objective(
		model.model_id,
		base_config,
		str(model.expected['objective']),
	)
	training_state = _required_mapping(
		payload,
		'training_state',
		f'{model.model_id} checkpoint',
	)
	if training_state.get('stage') != 'train_strat_hmm_pretext':
		raise ValueError(
			f'{model.model_id} checkpoint stage must be train_strat_hmm_pretext'
		)
	if training_state.get('checkpoint_kind') != 'epoch':
		raise ValueError(f'{model.model_id} must be a completed HMM epoch checkpoint')
	stratigraphy = _required_mapping(
		payload,
		'stratigraphy_config',
		f'{model.model_id} checkpoint',
	)
	head = _required_mapping(stratigraphy, 'head', f'{model.model_id} stratigraphy')
	if head.get('num_prototypes') != FIVE_WAY_HMM_K:
		raise ValueError(f'{model.model_id} HMM prototype count must equal 6')
	student = _required_mapping(
		stratigraphy,
		'student',
		f'{model.model_id} stratigraphy',
	)
	if student.get('unfreeze_top_blocks') != FIVE_WAY_UNFREEZE_TOP_BLOCKS:
		raise ValueError(f'{model.model_id} HMM must unfreeze the top encoder block')
	parent_path = _lineage_path(
		model.model_id,
		student.get('init_checkpoint'),
		'stratigraphy_config.student.init_checkpoint',
	)
	teacher = _required_mapping(
		stratigraphy,
		'teacher',
		f'{model.model_id} stratigraphy',
	)
	teacher_path = _lineage_path(
		model.model_id,
		teacher.get('checkpoint'),
		'stratigraphy_config.teacher.checkpoint',
	)
	if teacher_path.resolve(strict=False) != parent_path.resolve(strict=False):
		raise ValueError(f'{model.model_id} HMM teacher and student parents differ')
	pseudo_targets = _required_mapping(
		stratigraphy,
		'pseudo_targets',
		f'{model.model_id} stratigraphy',
	)
	if pseudo_targets.get('k') != FIVE_WAY_HMM_K:
		raise ValueError(f'{model.model_id} pseudo target K must equal 6')
	return parent_path, pseudo_targets


def _validate_stage2_budget(
	model_id: str,
	payload: Mapping[str, object],
	*,
	config: Mapping[str, object],
) -> None:
	if payload.get('epoch') != FIVE_WAY_STAGE2_EPOCHS:
		raise ValueError(
			f'{model_id} checkpoint epoch must equal {FIVE_WAY_STAGE2_EPOCHS}'
		)
	if payload.get('global_step') != FIVE_WAY_STAGE2_GLOBAL_STEPS:
		raise ValueError(
			f'{model_id} checkpoint global_step must equal '
			f'{FIVE_WAY_STAGE2_GLOBAL_STEPS}'
		)
	train: Mapping[str, object]
	if 'stratigraphy_config' in payload:
		stratigraphy = _required_mapping(
			payload,
			'stratigraphy_config',
			f'{model_id} checkpoint',
		)
		train = _required_mapping(stratigraphy, 'train', f'{model_id} stratigraphy')
	else:
		train = _required_mapping(config, 'train', f'{model_id} checkpoint config')
	_validate_train_budget(
		model_id,
		train,
		epochs=FIVE_WAY_STAGE2_EPOCHS,
		global_steps=FIVE_WAY_STAGE2_GLOBAL_STEPS,
		batch_size=VOLVE_PRETRAIN_BATCH_SIZE,
	)
	for key in ('optimizer_state_loaded', 'scheduler_state_loaded'):
		if payload.get(key) is True:
			raise ValueError(f'{model_id} checkpoint must restart Stage-2 {key}')


def _validate_stage1_parent(
	model_id: str,
	path: Path,
	*,
	expected_objective: str,
	depth: int = 0,
) -> str:
	if depth >= _MAX_LINEAGE_DEPTH:
		raise ValueError(
			f'{model_id} checkpoint lineage is deeper than {_MAX_LINEAGE_DEPTH}'
		)
	parent_sha = _checkpoint_sha256(f'{model_id} stage-1 parent', path)
	payload = _load_checkpoint(f'{model_id} stage-1 parent', path)
	config = _required_mapping(payload, 'config', f'{model_id} stage-1 parent')
	_validate_base_objective(model_id, config, expected_objective)
	if payload.get('epoch') != FIVE_WAY_STAGE1_EPOCHS:
		raise ValueError(f'{model_id} stage-1 parent epoch must equal 100')
	is_local = expected_objective == LOCAL_BARLOW_TWINS_METHOD
	expected_batch_size = (
		VOLVE_LOCAL_BT_STAGE1_BATCH_SIZE
		if is_local
		else VOLVE_PRETRAIN_BATCH_SIZE
	)
	expected_global_steps = (
		FIVE_WAY_LOCAL_BT_STAGE1_GLOBAL_STEPS
		if is_local
		else FIVE_WAY_STAGE1_GLOBAL_STEPS
	)
	if payload.get('global_step') != expected_global_steps:
		raise ValueError(
			f'{model_id} stage-1 parent global_step must equal '
			f'{expected_global_steps}'
		)
	train = _required_mapping(config, 'train', f'{model_id} stage-1 config')
	_validate_train_budget(
		model_id,
		train,
		epochs=FIVE_WAY_STAGE1_EPOCHS,
		global_steps=expected_global_steps,
		batch_size=expected_batch_size,
	)
	continuation = config.get('continuation')
	if isinstance(continuation, Mapping) and continuation.get('init_checkpoint'):
		ancestor = _lineage_path(
			model_id,
			continuation.get('init_checkpoint'),
			'stage-1 continuation.init_checkpoint',
		)
		_validate_stage1_parent(
			model_id,
			ancestor,
			expected_objective=expected_objective,
			depth=depth + 1,
		)
	return parent_sha


def _validate_train_budget(
	model_id: str,
	train: Mapping[str, object],
	*,
	epochs: int,
	global_steps: int,
	batch_size: int,
) -> None:
	checks = {
		'epochs': epochs,
		'samples_per_epoch': VOLVE_PRETRAIN_SAMPLES_PER_EPOCH,
		'batch_size': batch_size,
	}
	for key, expected in checks.items():
		if train.get(key) != expected:
			raise ValueError(
				f'{model_id} train.{key} must equal {expected}; '
				f'got {train.get(key)!r}'
			)
	derived = math.ceil(
		VOLVE_PRETRAIN_SAMPLES_PER_EPOCH / batch_size
	) * epochs
	if derived != global_steps:
		raise RuntimeError('internal Volve optimizer update budget is inconsistent')
	max_steps = train.get('max_steps')
	if max_steps is not None and max_steps != global_steps:
		raise ValueError(
			f'{model_id} train.max_steps must be null or {global_steps}; '
			f'got {max_steps!r}'
		)


def _validate_base_objective(
	model_id: str,
	config: Mapping[str, object],
	expected_objective: str,
) -> None:
	_validate_encoder_geometry(model_id, config)
	if expected_objective == MAE_OBJECTIVE:
		if config.get('stage') != 'train_amp_mae':
			raise ValueError(f'{model_id} base objective must be MAE')
	else:
		if config.get('stage') != 'barlow_twins_training':
			raise ValueError(f'{model_id} base objective must be Local Barlow Twins')
		barlow = _required_mapping(config, 'barlow_twins', f'{model_id} config')
		if barlow.get('method') != LOCAL_BARLOW_TWINS_METHOD:
			raise ValueError(
				f'{model_id} barlow_twins.method must equal '
				f'{LOCAL_BARLOW_TWINS_METHOD!r}'
			)
		if barlow.get('local_pairs_per_crop') != LOCAL_BARLOW_TWINS_PAIRS_PER_CROP:
			raise ValueError(
				f'{model_id} local_pairs_per_crop must equal '
				f'{LOCAL_BARLOW_TWINS_PAIRS_PER_CROP}'
			)
	_reject_trace_drop(config, model_id)


def _validate_encoder_geometry(
	model_id: str,
	config: Mapping[str, object],
) -> None:
	model = _required_mapping(config, 'model', f'{model_id} config')
	for key, expected in EXPECTED_ENCODER_GEOMETRY.items():
		if model.get(key) != expected:
			raise ValueError(
				f'{model_id} model.{key} must equal {expected!r}; '
				f'got {model.get(key)!r}'
			)


def _model_geometry(config: Mapping[str, object]) -> dict[str, object]:
	model = _required_mapping(config, 'model', 'checkpoint config')
	return {key: model.get(key) for key in EXPECTED_ENCODER_GEOMETRY}


def _validate_recorded_parent_sha(  # noqa: C901
	model_id: str,
	payload: Mapping[str, object],
	*,
	parent_path: Path,
	parent_sha: str,
	required: bool,
) -> None:
	recorded: list[tuple[str, str]] = []
	lineage = payload.get('continuation_lineage')
	if lineage is not None:
		if not isinstance(lineage, Mapping):
			raise TypeError(f'{model_id} continuation_lineage must be a mapping')
		lineage_path = lineage.get('init_checkpoint')
		if isinstance(lineage_path, str) and Path(lineage_path).resolve(
			strict=False
		) != parent_path.resolve(strict=False):
			raise ValueError(f'{model_id} continuation lineage parent path mismatch')
		recorded.append(
			(
				'continuation_lineage.init_checkpoint_sha256',
				_required_sha256(
					lineage.get('init_checkpoint_sha256'),
					f'{model_id} continuation lineage parent SHA-256',
				),
			)
		)
	control = payload.get('control_identity')
	if control is not None:
		if not isinstance(control, Mapping):
			raise TypeError(f'{model_id} control_identity must be a mapping')
		inputs = _required_mapping(control, 'input_identities', f'{model_id} control')
		for key in ('teacher_checkpoint', 'student_init_checkpoint'):
			identity = _required_mapping(inputs, key, f'{model_id} control inputs')
			identity_path = identity.get('path')
			if isinstance(identity_path, str) and Path(identity_path).resolve(
				strict=False
			) != parent_path.resolve(strict=False):
				raise ValueError(f'{model_id} control {key} path mismatch')
			recorded.append(
				(
					f'control_identity.input_identities.{key}.sha256',
					_required_sha256(
						identity.get('sha256'),
						f'{model_id} control {key} SHA-256',
					),
				)
			)
	checkpoint_identity = payload.get('stratigraphy_checkpoint')
	if isinstance(checkpoint_identity, Mapping):
		recorded.extend(
			(
				f'stratigraphy_checkpoint.{key}',
				_required_sha256(
					checkpoint_identity.get(key),
					f'{model_id} {key}',
				),
			)
			for key in (
				'teacher_checkpoint_sha256',
				'student_init_checkpoint_sha256',
			)
			if key in checkpoint_identity
		)
	if required and not recorded:
		raise ValueError(f'{model_id} checkpoint does not record its parent SHA-256')
	for field, value in recorded:
		if value != parent_sha:
			raise ValueError(f'{model_id} {field} does not match the parent file')


def _validate_pseudo_target_lineage(  # noqa: PLR0913
	model_id: str,
	pseudo_targets: Mapping[str, object],
	*,
	checkpoint_payload: Mapping[str, object],
	parent_path: Path,
	parent_sha: str,
	survey_id: str,
) -> dict[str, object]:
	input_dir = Path(
		_required_string(
			pseudo_targets.get('input_dir'),
			f'{model_id} pseudo_targets.input_dir',
		)
	)
	if 'trace_drop' in str(input_dir).lower():
		raise ValueError(f'{model_id} pseudo targets are from a trace-drop artifact')
	if not input_dir.is_dir():
		raise FileNotFoundError(
			f'{model_id} pseudo target directory is missing: {input_dir}'
		)
	metadata_paths = tuple(sorted(input_dir.rglob('*.pseudo_target_metadata.json')))
	if not metadata_paths:
		raise FileNotFoundError(
			f'{model_id} pseudo target metadata is missing below {input_dir}'
		)
	metadata_identities: list[dict[str, str]] = []
	for path in metadata_paths:
		metadata = _read_json(path, f'{model_id} pseudo target metadata')
		if metadata.get('survey_id') != survey_id:
			raise ValueError(
				f'{model_id} pseudo target survey_id must equal {survey_id!r}'
			)
		if metadata.get('k') != FIVE_WAY_HMM_K:
			raise ValueError(f'{model_id} pseudo target metadata K must equal 6')
		source = _required_mapping(metadata, 'source', f'{model_id} pseudo metadata')
		source_path, source_sha = _pseudo_source_checkpoint_identity(
			model_id,
			source,
		)
		if source_path.resolve(strict=False) != parent_path.resolve(strict=False):
			raise ValueError(
				f'{model_id} pseudo target source does not match its stage-1 parent'
			)
		if source_sha != parent_sha:
			raise ValueError(f'{model_id} pseudo target parent SHA-256 mismatch')
		metadata_identities.append(
			{'path': str(path), 'sha256': file_sha256(path)}
		)
	_validate_checkpoint_pseudo_target_bindings(
		model_id,
		checkpoint_payload,
		input_dir=input_dir,
		metadata_paths=metadata_paths,
		survey_id=survey_id,
	)
	return {
		'input_dir': str(input_dir),
		'metadata': metadata_identities,
		'source_checkpoint': str(parent_path),
		'source_checkpoint_sha256': parent_sha,
	}


def _pseudo_source_checkpoint_identity(
	model_id: str,
	source: Mapping[str, object],
) -> tuple[Path, str]:
	'''Resolve direct builder or exported-clustering checkpoint provenance.'''
	if 'checkpoint_path' in source or 'checkpoint_sha256' in source:
		return (
			_lineage_path(
				model_id,
				source.get('checkpoint_path'),
				'pseudo target source.checkpoint_path',
			),
			_required_sha256(
				source.get('checkpoint_sha256'),
				f'{model_id} pseudo target source checkpoint SHA-256',
			),
		)
	cluster_metadata_path = _identity_artifact_path(
		model_id,
		source.get('source_metadata_path'),
		'pseudo target source.source_metadata_path',
	)
	_validate_live_sha(
		model_id,
		cluster_metadata_path,
		source.get('source_metadata_sha256'),
		'pseudo target source metadata',
	)
	cluster_metadata = _read_json(
		cluster_metadata_path,
		f'{model_id} source cluster metadata',
	)
	embedding_input = _required_mapping(
		cluster_metadata,
		'embedding_input',
		f'{model_id} source cluster metadata',
	)
	embedding_metadata_path = _identity_artifact_path(
		model_id,
		embedding_input.get('metadata_path'),
		'source cluster embedding_input.metadata_path',
	)
	_validate_live_sha(
		model_id,
		embedding_metadata_path,
		embedding_input.get('metadata_sha256'),
		'source embedding metadata',
	)
	embedding_metadata = _read_json(
		embedding_metadata_path,
		f'{model_id} source embedding metadata',
	)
	return (
		_lineage_path(
			model_id,
			embedding_metadata.get('checkpoint_path'),
			'source embedding checkpoint_path',
		),
		_required_sha256(
			embedding_metadata.get('checkpoint_sha256'),
			f'{model_id} source embedding checkpoint SHA-256',
		),
	)


def _validate_checkpoint_pseudo_target_bindings(  # noqa: C901
	model_id: str,
	payload: Mapping[str, object],
	*,
	input_dir: Path,
	metadata_paths: tuple[Path, ...],
	survey_id: str,
) -> None:
	control = _required_mapping(payload, 'control_identity', f'{model_id} checkpoint')
	inputs = _required_mapping(control, 'input_identities', f'{model_id} control')
	value = inputs.get('pseudo_targets')
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(
			f'{model_id} control input_identities.pseudo_targets must be a list'
		)
	recorded_metadata: set[Path] = set()
	for index, item in enumerate(value):
		if not isinstance(item, Mapping):
			raise TypeError(
				f'{model_id} pseudo target identity {index} must be a mapping'
			)
		if item.get('survey_id') != survey_id:
			raise ValueError(
				f'{model_id} control pseudo_targets[{index}].survey_id must equal '
				f'{survey_id!r}'
			)
		for key in ('labels', 'confidence', 'valid_tokens', 'metadata'):
			identity = _required_mapping(
				item,
				key,
				f'{model_id} pseudo target identity {index}',
			)
			path = _identity_artifact_path(
				model_id,
				identity.get('path'),
				f'control pseudo_targets[{index}].{key}.path',
			)
			if not _is_relative_to(path, input_dir):
				raise ValueError(
					f'{model_id} control pseudo target {key} is outside input_dir'
				)
			_validate_live_sha(
				model_id,
				path,
				identity.get('sha256'),
				f'control pseudo_targets[{index}].{key}',
			)
			if key == 'metadata':
				recorded_metadata.add(path.resolve(strict=False))
		boundary_present = item.get('boundary_weight_present')
		if boundary_present is True:
			boundary = _required_mapping(
				item,
				'boundary_weight',
				f'{model_id} pseudo target identity {index}',
			)
			boundary_path = _identity_artifact_path(
				model_id,
				boundary.get('path'),
				f'control pseudo_targets[{index}].boundary_weight.path',
			)
			if not _is_relative_to(boundary_path, input_dir):
				raise ValueError(
					f'{model_id} control pseudo target boundary_weight is outside '
					'input_dir'
				)
			_validate_live_sha(
				model_id,
				boundary_path,
				boundary.get('sha256'),
				f'control pseudo_targets[{index}].boundary_weight',
			)
	expected_metadata = {
		path.resolve(strict=False) for path in metadata_paths
	}
	if recorded_metadata != expected_metadata:
		raise ValueError(
			f'{model_id} checkpoint pseudo-target metadata identities differ '
			'from the live input directory'
		)


def _identity_artifact_path(model_id: str, value: object, field: str) -> Path:
	path = Path(_required_string(value, f'{model_id} {field}'))
	if not path.is_absolute():
		raise ValueError(f'{model_id} {field} must be absolute')
	if 'trace_drop' in str(path).lower():
		raise ValueError(f'{model_id} {field} references trace-drop lineage')
	if not path.is_file():
		raise FileNotFoundError(f'{model_id} {field} is missing: {path}')
	return path


def _validate_live_sha(
	model_id: str,
	path: Path,
	value: object,
	field: str,
) -> str:
	recorded = _required_sha256(value, f'{model_id} {field} SHA-256')
	if file_sha256(path) != recorded:
		raise ValueError(f'{model_id} {field} SHA-256 differs from its live file')
	return recorded


def _reject_trace_drop(value: object, model_id: str, *, prefix: str = 'config') -> None:
	if isinstance(value, Mapping):
		for key, child in value.items():
			label = f'{prefix}.{key}'
			if key == 'trace_drop_probability':
				if isinstance(child, bool) or not isinstance(child, int | float):
					raise TypeError(f'{model_id} {label} must be numeric')
				if float(child) != 0.0:
					raise ValueError(f'{model_id} {label} must be disabled')
				continue
			_reject_trace_drop(child, model_id, prefix=label)
		return
	if isinstance(value, Sequence) and not isinstance(value, str | bytes):
		for index, child in enumerate(value):
			_reject_trace_drop(child, model_id, prefix=f'{prefix}[{index}]')
		return
	if isinstance(value, str) and 'trace_drop' in value.lower():
		raise ValueError(f'{model_id} {prefix} references trace-drop lineage')


def _validate_extraction_metadata(
	model_id: str,
	metadata: Mapping[str, object],
) -> None:
	for key, expected in EXPECTED_EXTRACTION_CONTRACT.items():
		if metadata.get(key) != expected:
			raise ValueError(
				f'{model_id} embedding metadata {key} must equal {expected!r}; '
				f'got {metadata.get(key)!r}'
			)
	geometry = _required_mapping(metadata, 'model_geometry', f'{model_id} metadata')
	for key, expected in EXPECTED_ENCODER_GEOMETRY.items():
		if geometry.get(key) != expected:
			raise ValueError(
				f'{model_id} embedding model_geometry.{key} must equal {expected!r}'
			)


def _validate_shared_embedding_identity(
	model_id: str,
	metadata: Mapping[str, object],
	reference: Mapping[str, object],
) -> None:
	for key in SHARED_EMBEDDING_METADATA_KEYS:
		if key not in metadata or key not in reference:
			raise ValueError(f'{model_id} shared embedding metadata requires {key}')
		if metadata[key] != reference[key]:
			raise ValueError(
				f'{model_id} embedding metadata {key} differs from the shared suite'
			)


def _inspect_arrays(
	model_id: str,
	paths: EmbeddingOutputPaths,
	*,
	token_grid: tuple[int, int, int],
	metadata: Mapping[str, object],
) -> tuple[tuple[int, int, int, int], int, np.ndarray]:
	embeddings = np.load(paths.embeddings, mmap_mode='r', allow_pickle=False)
	valid_tokens = np.load(paths.valid_tokens, mmap_mode='r', allow_pickle=False)
	expected_embedding_shape = (
		*token_grid,
		int(EXPECTED_ENCODER_GEOMETRY['encoder_dim']),
	)
	if tuple(embeddings.shape) != expected_embedding_shape:
		raise ValueError(
			f'{model_id} embedding array shape must equal '
			f'{expected_embedding_shape!r}; got {tuple(embeddings.shape)!r}'
		)
	if embeddings.dtype != np.dtype(np.float16):
		raise TypeError(f'{model_id} embedding array dtype must be float16')
	if np.dtype(cast('str', metadata.get('output_dtype'))) != embeddings.dtype:
		raise TypeError(f'{model_id} embedding dtype differs from metadata')
	if (
		valid_tokens.dtype != np.dtype(np.bool_)
		or tuple(valid_tokens.shape) != token_grid
	):
		raise TypeError(
			f'{model_id} valid-token array must be bool with the full token grid'
		)
	shape = cast(
		'tuple[int, int, int, int]',
		tuple(int(size) for size in embeddings.shape),
	)
	return shape, shape[-1], valid_tokens


def _validate_masks_identical(
	model_id: str,
	reference: np.ndarray,
	candidate: np.ndarray,
) -> None:
	if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
		raise ValueError(f'{model_id} valid-token mask geometry differs from mae')
	reference_flat = reference.reshape(-1)
	candidate_flat = candidate.reshape(-1)
	for start in range(0, reference_flat.shape[0], _MASK_COMPARE_CHUNK):
		stop = min(reference_flat.shape[0], start + _MASK_COMPARE_CHUNK)
		if not np.array_equal(reference_flat[start:stop], candidate_flat[start:stop]):
			raise ValueError(
				f'{model_id} valid-token mask differs from the mae valid-token mask'
			)


def _validate_embedding_checkpoint_identity(
	model: VolveHorizonFiveWayModelSource,
	metadata: Mapping[str, object],
	checkpoint_report: Mapping[str, object],
) -> None:
	checkpoint_value = _required_string(
		metadata.get('checkpoint_path'),
		f'{model.model_id} embedding checkpoint_path',
	)
	if Path(checkpoint_value).resolve(strict=False) != model.checkpoint.resolve(
		strict=False
	):
		raise ValueError(
			f'{model.model_id} embedding checkpoint_path differs from config'
		)
	sha = _required_sha256(
		metadata.get('checkpoint_sha256'),
		f'{model.model_id} embedding checkpoint_sha256',
	)
	if sha != checkpoint_report.get('checkpoint_sha256'):
		raise ValueError(
			f'{model.model_id} embedding checkpoint SHA-256 differs from source audit'
		)


def _validate_embedding_objective(  # noqa: C901
	model: VolveHorizonFiveWayModelSource,
	metadata: Mapping[str, object],
) -> None:
	objective = _required_mapping(
		metadata,
		'pretraining_objective',
		f'{model.model_id} embedding metadata',
	)
	is_local = str(model.expected['objective']) == LOCAL_BARLOW_TWINS_METHOD
	if is_local:
		if metadata.get('pretraining_method') != LOCAL_BARLOW_TWINS_METHOD:
			raise ValueError(
				f'{model.model_id} embedding pretraining_method must identify Local BT'
			)
		if objective.get('method') != LOCAL_BARLOW_TWINS_METHOD:
			raise ValueError(
				f'{model.model_id} embedding objective method must identify Local BT'
			)
		if objective.get('local_pairs_per_crop') != LOCAL_BARLOW_TWINS_PAIRS_PER_CROP:
			raise ValueError(
				f'{model.model_id} embedding local_pairs_per_crop must equal 128'
			)
	elif 'method' in objective or 'reconstruction' not in objective:
		raise ValueError(
			f'{model.model_id} embedding must carry the MAE reconstruction objective'
		)
	expects_pretext = bool(model.expected['stratigraphy_pretext'])
	pretext = metadata.get('stratigraphy_pretext')
	if not expects_pretext:
		if pretext is not None:
			raise ValueError(
				f'{model.model_id} embedding must not carry '
				'stratigraphy pretext metadata'
			)
		return
	if not isinstance(pretext, Mapping):
		raise TypeError(
			f'{model.model_id} embedding stratigraphy_pretext must be a mapping'
		)
	checks = {
		'method': STRAT_HMM_PRETEXT_METHOD,
		'base_objective': model.expected['base_objective'],
		'head_num_prototypes': FIVE_WAY_HMM_K,
		'unfreeze_top_blocks': FIVE_WAY_UNFREEZE_TOP_BLOCKS,
	}
	for key, expected in checks.items():
		if pretext.get(key) != expected:
			raise ValueError(
				f'{model.model_id} stratigraphy_pretext.{key} must equal {expected!r}'
			)
	_reject_trace_drop(pretext, model.model_id, prefix='stratigraphy_pretext')


def _validate_grid_geometry(
	volume_shape: tuple[int, int, int],
	token_grid: tuple[int, int, int],
) -> None:
	patch_size = cast('list[int]', EXPECTED_ENCODER_GEOMETRY['patch_size'])
	expected = tuple(
		math.ceil(size / patch)
		for size, patch in zip(volume_shape, patch_size, strict=True)
	)
	if token_grid != expected:
		raise ValueError('embedding token grid is inconsistent with volume geometry')


def _inspect_canonical_identity(
	config: VolveHorizonFiveWayConfig,
	*,
	embedding_metadata: Mapping[str, object],
	volume_shape: tuple[int, int, int],
) -> dict[str, object]:
	metadata = _read_json(
		config.canonical_input_metadata,
		'canonical input metadata',
	)
	if metadata.get('status') != 'PASS' or metadata.get('artifact_type') != (
		'volve_canonical_input_registration'
	):
		raise ValueError('canonical input metadata must be a PASS registration')
	identity = _required_mapping(metadata, 'scientific_identity', 'canonical metadata')
	identity_sha = _sha256_json(identity)
	if metadata.get('scientific_identity_sha256') != identity_sha:
		raise ValueError('canonical scientific identity SHA-256 mismatch')
	if identity.get('survey_id') != config.survey_id:
		raise ValueError('canonical survey identity differs from five-way config')
	if tuple(identity.get('shape_xyz', ())) != volume_shape:
		raise ValueError('canonical input shape differs from embedding volume shape')
	for key in (
		'canonical_amplitude_sha256',
		'valid_trace_mask_sha256',
		'inline_values_sha256',
		'crossline_values_sha256',
		'time_axis_sha256',
		'canonical_normalization_stats_sha256',
	):
		_required_sha256(identity.get(key), f'canonical identity {key}')
	provenance = _required_mapping(metadata, 'provenance', 'canonical metadata')
	amplitude = _required_mapping(provenance, 'amplitude', 'canonical provenance')
	public_inputs = _required_mapping(
		provenance,
		'public_inputs',
		'canonical provenance',
	)
	outputs = _required_mapping(metadata, 'outputs', 'canonical metadata')
	checks = {
		'source_amplitude_path': amplitude.get('path'),
		'source_valid_mask_path': public_inputs.get('valid_trace_mask.npy'),
		'normalization_stats_path': outputs.get('normalization_stats'),
	}
	for key, expected in checks.items():
		if embedding_metadata.get(key) != expected:
			raise ValueError(f'embedding {key} differs from canonical registration')
	for key, identity_key in (
		('source_amplitude_path', 'canonical_amplitude_sha256'),
		('source_valid_mask_path', 'valid_trace_mask_sha256'),
		('normalization_stats_path', 'canonical_normalization_stats_sha256'),
	):
		path = _identity_artifact_path(
			'canonical input',
			checks[key],
			key,
		)
		_validate_live_sha(
			'canonical input',
			path,
			identity.get(identity_key),
			identity_key,
		)
	return {
		'scientific_identity_sha256': identity_sha,
		'canonical_input_metadata_sha256': file_sha256(
			config.canonical_input_metadata
		),
		**{
			key: identity[key]
			for key in (
				'canonical_amplitude_sha256',
				'valid_trace_mask_sha256',
				'inline_values_sha256',
				'crossline_values_sha256',
				'time_axis_sha256',
				'canonical_normalization_stats_sha256',
			)
		},
	}


def _validate_valid_support(
	valid_tokens: np.ndarray,
	*,
	embedding_metadata: Mapping[str, object],
	volume_shape: tuple[int, int, int],
	token_grid: tuple[int, int, int],
) -> None:
	path = Path(
		_required_string(
			embedding_metadata.get('source_valid_mask_path'),
			'embedding source_valid_mask_path',
		)
	)
	if not path.is_file():
		raise FileNotFoundError(f'canonical source valid mask is missing: {path}')
	trace_valid = np.load(path, mmap_mode='r', allow_pickle=False)
	if trace_valid.dtype != np.bool_ or tuple(trace_valid.shape) != volume_shape[:2]:
		raise TypeError('canonical source valid mask must be bool with volume XY shape')
	patch = cast('list[int]', EXPECTED_ENCODER_GEOMETRY['patch_size'])
	expected_lateral = np.zeros(token_grid[:2], dtype=np.bool_)
	for token_x in range(token_grid[0]):
		for token_y in range(token_grid[1]):
			x0, y0 = token_x * patch[0], token_y * patch[1]
			x1, y1 = x0 + patch[0], y0 + patch[1]
			if x1 <= volume_shape[0] and y1 <= volume_shape[1]:
				expected_lateral[token_x, token_y] = bool(
					np.all(trace_valid[x0:x1, y0:y1])
				)
	expected = np.zeros(token_grid, dtype=np.bool_)
	full_z_tokens = volume_shape[2] // patch[2]
	expected[:, :, :full_z_tokens] = expected_lateral[:, :, np.newaxis]
	if np.any(valid_tokens & ~expected):
		raise ValueError('valid-token mask includes missing-trace or padding tokens')


def _parent_identity(source: Mapping[str, object]) -> tuple[Path, str]:
	return (
		Path(cast('str', source['parent_checkpoint'])),
		cast('str', source['parent_checkpoint_sha256']),
	)


def _checkpoint_reports_from_audit(
	config: VolveHorizonFiveWayConfig,
	report: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
	if report.get('survey_id') != config.survey_id:
		raise ValueError('five-way source report survey_id differs from config')
	if report.get('model_order') != list(config.model_ids):
		raise ValueError('five-way source report model_order differs from config')
	value = report.get('sources')
	if not isinstance(value, list) or not all(
		isinstance(item, Mapping) for item in value
	):
		raise TypeError('five-way source report sources must be a list of mappings')
	sources = cast('list[Mapping[str, object]]', value)
	if [source.get('model_id') for source in sources] != list(config.model_ids):
		raise ValueError('five-way source report sources differ from fixed model order')
	reports: dict[str, Mapping[str, object]] = {}
	for model, source in zip(config.models, sources, strict=True):
		checkpoint = _lineage_path(
			model.model_id,
			source.get('checkpoint'),
			'source report checkpoint',
		)
		if checkpoint.resolve(strict=False) != model.checkpoint.resolve(strict=False):
			raise ValueError(
				f'{model.model_id} source report checkpoint differs from config'
			)
		_required_sha256(
			source.get('checkpoint_sha256'),
			f'{model.model_id} source report checkpoint SHA-256',
		)
		reports[model.model_id] = source
	return reports


def _load_checkpoint(model_id: str, path: Path) -> Mapping[str, object]:
	try:
		return load_checkpoint_metadata_without_weights(path)
	except Exception as error:
		raise ValueError(
			f'{model_id} checkpoint metadata is unreadable: {path}'
		) from error


def _checkpoint_sha256(model_id: str, path: Path) -> str:
	if not path.is_file():
		raise FileNotFoundError(f'{model_id} checkpoint does not exist: {path}')
	return file_sha256(path)


def _lineage_path(model_id: str, value: object, field: str) -> Path:
	path = Path(_required_string(value, f'{model_id} {field}'))
	if 'trace_drop' in str(path).lower():
		raise ValueError(f'{model_id} {field} references trace-drop lineage')
	if not path.is_absolute():
		raise ValueError(f'{model_id} {field} must be absolute')
	return path


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


def _required_mapping(
	value: Mapping[str, object],
	key: str,
	prefix: str,
) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{prefix}.{key} must be a mapping')
	return child


def _required_string(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty string')
	return value


def _required_sha256(value: object, label: str) -> str:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise ValueError(f'{label} must be a lowercase SHA-256 digest')
	return value


def _positive_triplet(value: object, label: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, list | tuple)
		or len(value) != 3
		or any(
			not isinstance(item, int) or isinstance(item, bool) or item <= 0
			for item in value
		)
	):
		raise ValueError(f'{label} must be three positive integers')
	return cast('tuple[int, int, int]', tuple(value))


def _read_json(path: Path, label: str) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(f'{label} does not exist: {path}')
	try:
		payload: Any = json.loads(path.read_text(encoding='utf-8'))
	except (OSError, json.JSONDecodeError) as error:
		raise ValueError(f'{label} is unreadable: {path}') from error
	if not isinstance(payload, Mapping):
		raise TypeError(f'{label} must contain a JSON object')
	return cast('Mapping[str, object]', payload)


def _sha256_json(value: Mapping[str, object]) -> str:
	return hashlib.sha256(
		json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
	).hexdigest()


__all__ = [
	'EXPECTED_ENCODER_GEOMETRY',
	'EXPECTED_EXTRACTION_CONTRACT',
	'FIVE_WAY_LOCAL_BT_STAGE1_GLOBAL_STEPS',
	'FIVE_WAY_STAGE1_GLOBAL_STEPS',
	'FIVE_WAY_STAGE2_GLOBAL_STEPS',
	'RANDOM_CHECKPOINT_STAGE',
	'SHARED_EMBEDDING_METADATA_KEYS',
	'STRAT_HMM_PRETEXT_METHOD',
	'VOLVE_LOCAL_BT_STAGE1_BATCH_SIZE',
	'VOLVE_PRETRAIN_BATCH_SIZE',
	'VOLVE_PRETRAIN_SAMPLES_PER_EPOCH',
	'VolveHorizonFiveWayEmbeddingSource',
	'VolveHorizonFiveWayEmbeddingSuite',
	'audit_volve_horizon_five_way_sources',
	'inspect_volve_horizon_five_way_embedding_suite',
	'plan_volve_horizon_five_way_embeddings',
	'plan_volve_horizon_five_way_sources',
]
