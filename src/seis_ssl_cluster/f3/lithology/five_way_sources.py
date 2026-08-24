"""Read-only source audit for the F3 lithology five-way comparison."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.config.f3_lithology_five_way import (
	FIVE_WAY_DISTILLATION_WEIGHT,
	FIVE_WAY_HMM_HEAD_CONTRACT,
	FIVE_WAY_HMM_K,
	FIVE_WAY_HMM_LOSS_CONTRACT,
	FIVE_WAY_MIN_CONFIDENCE,
	FIVE_WAY_PRETEXT_TRAIN_CONTRACT,
	FIVE_WAY_RANDOM_SEED,
	FIVE_WAY_STAGE1_EPOCHS,
	FIVE_WAY_STAGE1_TRAIN_CONTRACT,
	FIVE_WAY_STAGE2_EPOCHS,
	FIVE_WAY_STAGE2_GLOBAL_STEPS,
	FIVE_WAY_STAGE2_TRAIN_CONTRACT,
	FIVE_WAY_UNFREEZE_TOP_BLOCKS,
	LOCAL_BARLOW_TWINS_METHOD,
	LOCAL_BARLOW_TWINS_OBJECTIVE_CONTRACT,
	LOCAL_BARLOW_TWINS_PAIRS_PER_CROP,
	MAE_LOSS_CONTRACT,
	MAE_MASKING_CONTRACT,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.training.random_checkpoint import (
	load_checkpoint_metadata_without_weights,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.config.f3_lithology_five_way import (
		F3FiveWayConfig,
		F3FiveWayModelSource,
	)

EXPECTED_EXTRACTION_CONTRACT: Mapping[str, object] = {
	'window_size': [128, 128, 128],
	'overlap': [64, 64, 64],
	'output_dtype': 'float16',
	'min_token_valid_fraction': 0.5,
	'patch_size': [8, 8, 8],
}
EXPECTED_ENCODER_GEOMETRY: Mapping[str, int] = {
	'encoder_dim': 384,
	'encoder_depth': 8,
	'encoder_heads': 6,
}
SHARED_METADATA_KEYS = (
	'survey_id',
	'source_amplitude_path',
	'normalization_stats_path',
	'volume_shape_xyz',
	'token_grid_shape',
	'patch_size',
	'window_size',
	'overlap',
	'output_dtype',
	'min_token_valid_fraction',
	'model_geometry',
	'precision',
	'preprocessing',
	'amplitude_agc',
	'normalized_clip_abs',
	'finite_check_mode',
	'zero_mask',
	'preprocessing_cache',
)
EXPECTED_PSEUDO_TARGET_SUFFIXES: Mapping[str, tuple[str, ...]] = {
	'mae_hmm_k6': ('ssl_hmm_continuation_v1', 'mae100'),
	'local_barlow_twins_hmm_k6': ('mae_local_bt_five_way_v1', 'local_bt100'),
}
TRACE_DROP_AUGMENTATION_KEYS = (
	'policy',
	'reflection_probability',
	'trace_drop_probability',
)
STRAT_HMM_PRETEXT_METHOD = 'strat_hmm_pretext'
RANDOM_CHECKPOINT_STAGE = 'create_random_mae_checkpoint'
_MASK_COMPARE_CHUNK = 1 << 22
_MAX_LINEAGE_DEPTH = 4


def plan_f3_lithology_five_way_sources(
	config: F3FiveWayConfig,
) -> tuple[dict[str, object], ...]:
	"""Resolve the static per-model source plan without touching artifacts."""
	survey_id = config.dataset['name']
	rows = []
	for model in config.models:
		files = output_paths(model.embeddings_dir, survey_id)
		rows.append(
			{
				'model_id': model.model_id,
				'checkpoint': str(model.checkpoint),
				'embeddings_dir': str(model.embeddings_dir),
				'embeddings': str(files.embeddings),
				'valid_tokens': str(files.valid_tokens),
				'embedding_metadata': str(files.metadata),
				'expected': dict(model.expected),
			}
		)
	return tuple(rows)


def audit_f3_lithology_five_way_sources(
	config: F3FiveWayConfig,
) -> dict[str, object]:
	"""Audit the five live sources read-only; raise on any identity drift."""
	survey_id = config.dataset['name']
	mae_checkpoint = config.model_by_id('mae').checkpoint
	reference_metadata: Mapping[str, object] | None = None
	reference_valid_tokens: Path | None = None
	sources: list[dict[str, object]] = []
	for model in config.models:
		files = output_paths(model.embeddings_dir, survey_id)
		for path in (files.embeddings, files.valid_tokens, files.metadata):
			if not path.is_file():
				raise FileNotFoundError(
					f'{model.model_id} embedding source is missing: {path}'
				)
		metadata = _read_json(files.metadata)
		if metadata.get('survey_id') != survey_id:
			raise ValueError(
				f'{model.model_id} embedding survey_id does not match '
				f'{survey_id!r}'
			)
		_validate_extraction_contract(model.model_id, metadata)
		if reference_metadata is None:
			reference_metadata = metadata
			reference_valid_tokens = files.valid_tokens
		else:
			_validate_shared_identity(model.model_id, metadata, reference_metadata)
		token_grid_shape = _token_grid_shape(model.model_id, metadata)
		_validate_arrays(model.model_id, files, token_grid_shape)
		if reference_valid_tokens is not None and not _masks_identical(
			reference_valid_tokens, files.valid_tokens
		):
			raise ValueError(
				f'{model.model_id} valid-token mask is not byte-identical to '
				'the mae valid-token mask'
			)
		checkpoint_sha256 = _validate_checkpoint_identity(model, metadata)
		_validate_objective_identity(model, metadata)
		_validate_checkpoint_payload(model, mae_checkpoint=mae_checkpoint)
		sources.append(
			{
				'model_id': model.model_id,
				'checkpoint': str(model.checkpoint),
				'checkpoint_sha256': checkpoint_sha256,
				'embeddings_dir': str(model.embeddings_dir),
				'token_grid_shape': list(token_grid_shape),
				'valid_token_mask': 'byte_identical_across_models',
			}
		)
	return {
		'survey_id': survey_id,
		'model_order': list(config.model_ids),
		'sources': sources,
	}


def _validate_extraction_contract(
	model_id: str, metadata: Mapping[str, object]
) -> None:
	for key, expected in EXPECTED_EXTRACTION_CONTRACT.items():
		if metadata.get(key) != expected:
			raise ValueError(
				f'{model_id} embedding metadata {key} must equal {expected!r}; '
				f'got {metadata.get(key)!r}'
			)
	geometry = metadata.get('model_geometry')
	if not isinstance(geometry, Mapping):
		raise TypeError(f'{model_id} embedding model_geometry must be a mapping')
	for key, expected in EXPECTED_ENCODER_GEOMETRY.items():
		if geometry.get(key) != expected:
			raise ValueError(
				f'{model_id} embedding model_geometry.{key} must equal '
				f'{expected}; got {geometry.get(key)!r}'
			)
	precision = metadata.get('precision')
	if not isinstance(precision, Mapping):
		raise TypeError(f'{model_id} embedding precision must be a mapping')
	if precision.get('amp_enabled') is not False:
		raise ValueError(f'{model_id} embedding must be extracted with amp disabled')


def _validate_shared_identity(
	model_id: str,
	metadata: Mapping[str, object],
	reference: Mapping[str, object],
) -> None:
	for key in SHARED_METADATA_KEYS:
		if metadata.get(key) != reference.get(key):
			raise ValueError(
				f'{model_id} embedding metadata {key} differs from the shared '
				'five-way extraction contract'
			)


def _token_grid_shape(
	model_id: str, metadata: Mapping[str, object]
) -> tuple[int, ...]:
	value = metadata.get('token_grid_shape')
	if (
		not isinstance(value, list)
		or len(value) != 3
		or any(not isinstance(item, int) or item <= 0 for item in value)
	):
		raise ValueError(
			f'{model_id} embedding token_grid_shape must be three positive ints'
		)
	return tuple(value)


def _validate_arrays(
	model_id: str,
	files: object,
	token_grid_shape: tuple[int, ...],
) -> None:
	embeddings = np.load(files.embeddings, mmap_mode='r', allow_pickle=False)
	valid_tokens = np.load(files.valid_tokens, mmap_mode='r', allow_pickle=False)
	expected_shape = (
		*token_grid_shape,
		EXPECTED_ENCODER_GEOMETRY['encoder_dim'],
	)
	if tuple(embeddings.shape) != expected_shape:
		raise ValueError(
			f'{model_id} embedding array shape must equal {expected_shape!r}; '
			f'got {tuple(embeddings.shape)!r}'
		)
	if embeddings.dtype != np.dtype(np.float16):
		raise ValueError(f'{model_id} embedding array dtype must be float16')
	if tuple(valid_tokens.shape) != token_grid_shape:
		raise ValueError(
			f'{model_id} valid-token array shape must equal '
			f'{token_grid_shape!r}'
		)
	if valid_tokens.dtype != np.dtype(np.bool_):
		raise ValueError(f'{model_id} valid-token array dtype must be bool')


def _masks_identical(first: Path, second: Path) -> bool:
	if first == second:
		return True
	first_mask = np.load(first, mmap_mode='r', allow_pickle=False)
	second_mask = np.load(second, mmap_mode='r', allow_pickle=False)
	if first_mask.shape != second_mask.shape:
		return False
	if first_mask.dtype != second_mask.dtype:
		return False
	first_flat = first_mask.reshape(-1)
	second_flat = second_mask.reshape(-1)
	for start in range(0, first_flat.shape[0], _MASK_COMPARE_CHUNK):
		stop = start + _MASK_COMPARE_CHUNK
		if not np.array_equal(first_flat[start:stop], second_flat[start:stop]):
			return False
	return True


def _validate_checkpoint_identity(
	model: F3FiveWayModelSource, metadata: Mapping[str, object]
) -> str:
	checkpoint_value = metadata.get('checkpoint_path')
	if not isinstance(checkpoint_value, str) or not checkpoint_value:
		raise ValueError(
			f'{model.model_id} embedding metadata checkpoint_path is required'
		)
	recorded = Path(checkpoint_value).resolve(strict=False)
	if recorded != model.checkpoint.resolve(strict=False):
		raise ValueError(
			f'{model.model_id} embedding checkpoint_path does not match the '
			'configured checkpoint'
		)
	if not model.checkpoint.is_file():
		raise FileNotFoundError(
			f'{model.model_id} checkpoint does not exist: {model.checkpoint}'
		)
	sha_value = metadata.get('checkpoint_sha256')
	if (
		not isinstance(sha_value, str)
		or len(sha_value) != 64
		or any(character not in '0123456789abcdef' for character in sha_value)
	):
		raise ValueError(
			f'{model.model_id} embedding checkpoint_sha256 must be a '
			'lowercase SHA-256 digest'
		)
	if file_sha256(model.checkpoint) != sha_value:
		raise ValueError(
			f'{model.model_id} embedding checkpoint_sha256 does not match '
			'the checkpoint file'
		)
	return sha_value


def _validate_objective_identity(
	model: F3FiveWayModelSource, metadata: Mapping[str, object]
) -> None:
	expected = model.expected
	objective = metadata.get('pretraining_objective')
	if not isinstance(objective, Mapping):
		raise ValueError(  # noqa: TRY004 - missing metadata is a value error
			f'{model.model_id} embedding metadata pretraining_objective is required'
		)
	if expected['objective'] == LOCAL_BARLOW_TWINS_METHOD:
		if metadata.get('pretraining_method') != LOCAL_BARLOW_TWINS_METHOD:
			raise ValueError(
				f'{model.model_id} embedding pretraining_method must equal '
				f'{LOCAL_BARLOW_TWINS_METHOD!r}'
			)
		if objective.get('method') != LOCAL_BARLOW_TWINS_METHOD:
			raise ValueError(
				f'{model.model_id} pretraining_objective.method must equal '
				f'{LOCAL_BARLOW_TWINS_METHOD!r}'
			)
		if (
			objective.get('local_pairs_per_crop')
			!= LOCAL_BARLOW_TWINS_PAIRS_PER_CROP
		):
			raise ValueError(
				f'{model.model_id} pretraining_objective.local_pairs_per_crop '
				f'must equal {LOCAL_BARLOW_TWINS_PAIRS_PER_CROP}'
			)
	else:
		if 'pretraining_method' in metadata:
			raise ValueError(
				f'{model.model_id} embedding must not declare a Barlow Twins '
				'pretraining_method'
			)
		if 'method' in objective or 'reconstruction' not in objective:
			raise ValueError(
				f'{model.model_id} pretraining_objective must be the MAE '
				'reconstruction objective'
			)
	_validate_pretext_identity(model, metadata)


def _validate_pretext_identity(
	model: F3FiveWayModelSource, metadata: Mapping[str, object]
) -> None:
	expected = model.expected
	pretext = metadata.get('stratigraphy_pretext')
	if not expected['stratigraphy_pretext']:
		if pretext is not None:
			raise ValueError(
				f'{model.model_id} embedding must not carry a '
				'stratigraphy_pretext identity'
			)
		return
	if not isinstance(pretext, Mapping):
		raise ValueError(  # noqa: TRY004 - missing metadata is a value error
			f'{model.model_id} embedding metadata stratigraphy_pretext is required'
		)
	checks = {
		'method': STRAT_HMM_PRETEXT_METHOD,
		'base_objective': expected['base_objective'],
		'head_num_prototypes': FIVE_WAY_HMM_K,
		'unfreeze_top_blocks': FIVE_WAY_UNFREEZE_TOP_BLOCKS,
		'distillation_weight': FIVE_WAY_DISTILLATION_WEIGHT,
	}
	for key, expected_value in checks.items():
		if pretext.get(key) != expected_value:
			raise ValueError(
				f'{model.model_id} stratigraphy_pretext.{key} must equal '
				f'{expected_value!r}; got {pretext.get(key)!r}'
			)
	target_dir = pretext.get('pseudo_target_input_dir')
	if not isinstance(target_dir, str) or not target_dir:
		raise ValueError(
			f'{model.model_id} stratigraphy_pretext.pseudo_target_input_dir '
			'is required'
		)
	if 'trace_drop' in target_dir:
		raise ValueError(
			f'{model.model_id} pseudo targets must not come from a '
			'trace-drop artifact'
		)
	suffix = EXPECTED_PSEUDO_TARGET_SUFFIXES[model.model_id]
	if tuple(Path(target_dir).parts[-len(suffix) :]) != suffix:
		raise ValueError(
			f'{model.model_id} pseudo_target_input_dir must end with '
			f'{"/".join(suffix)!r}'
		)


def _validate_checkpoint_payload(
	model: F3FiveWayModelSource, *, mae_checkpoint: Path
) -> None:
	payload = load_checkpoint_metadata_without_weights(model.checkpoint)
	base_config = payload.get('config')
	if not isinstance(base_config, Mapping):
		raise ValueError(  # noqa: TRY004 - missing config is a value error
			f'{model.model_id} checkpoint must record its resolved config'
		)
	expected = model.expected
	has_pretext = 'stratigraphy_config' in payload
	if bool(expected['stratigraphy_pretext']) != has_pretext:
		raise ValueError(
			f'{model.model_id} checkpoint stratigraphy_config presence does '
			'not match the expected objective identity'
		)
	metadata = payload.get('metadata')
	if model.model_id == 'random':
		_validate_random_checkpoint_payload(payload, mae_checkpoint=mae_checkpoint)
		return
	if isinstance(metadata, Mapping) and (
		metadata.get('random_encoder_baseline') is True
	):
		raise ValueError(
			f'{model.model_id} checkpoint must not be a random encoder baseline'
		)
	_validate_base_objective(
		model.model_id,
		base_config,
		expected_objective=_expected_base_objective(model),
		role='checkpoint base config',
	)
	_validate_fixed_budget(model, payload, base_config=base_config)


def _expected_base_objective(model: F3FiveWayModelSource) -> str:
	"""Return the SSL objective every ancestor of this slot must carry."""
	expected = model.expected
	if expected['stratigraphy_pretext']:
		return str(expected['base_objective'])
	return str(expected['objective'])


def _validate_base_objective(
	model_id: str,
	config: Mapping[str, object],
	*,
	expected_objective: str,
	role: str,
) -> None:
	if expected_objective != LOCAL_BARLOW_TWINS_METHOD:
		if config.get('stage') != 'train_amp_mae':
			raise ValueError(
				f'{model_id} {role} stage must be train_amp_mae; '
				f'got {config.get("stage")!r}'
			)
		_validate_contract(
			model_id,
			config.get('masking'),
			MAE_MASKING_CONTRACT,
			prefix=f'{role} masking',
		)
		_validate_contract(
			model_id,
			config.get('loss'),
			MAE_LOSS_CONTRACT,
			prefix=f'{role} loss',
		)
		_reject_trace_drop_augmentations(model_id, config)
		return
	if config.get('stage') != 'barlow_twins_training':
		raise ValueError(
			f'{model_id} {role} stage must be barlow_twins_training; '
			f'got {config.get("stage")!r}'
		)
	_validate_contract(
		model_id,
		config.get('barlow_twins'),
		LOCAL_BARLOW_TWINS_OBJECTIVE_CONTRACT,
		prefix=f'{role} barlow_twins',
	)
	_reject_trace_drop_augmentations(model_id, config)


def _validate_contract(
	model_id: str,
	values: object,
	contract: Mapping[str, object],
	*,
	prefix: str,
) -> None:
	"""Reject drift in the settings the comparison declares scientifically fixed."""
	if not isinstance(values, Mapping):
		raise ValueError(  # noqa: TRY004 - a missing block is a value error
			f'{model_id} does not record {prefix}'
		)
	for key, expected in contract.items():
		value = values.get(key)
		# Booleans compare equal to 0 and 1, so an identity check keeps amp: 0
		# from passing as amp: false.
		matches = value is expected if isinstance(expected, bool) else value == expected
		if not matches:
			raise ValueError(
				f'{model_id} {prefix}.{key} must equal {expected!r}; got {value!r}'
			)


def _reject_trace_drop_augmentations(
	model_id: str, config: Mapping[str, object]
) -> None:
	augmentations = config.get('augmentations')
	if not isinstance(augmentations, Mapping):
		return
	forbidden = sorted(set(augmentations) & set(TRACE_DROP_AUGMENTATION_KEYS))
	if forbidden:
		raise ValueError(
			f'{model_id} checkpoint lineage uses trace-drop augmentations: '
			f'{forbidden!r}'
		)


def _validate_ancestry(
	model_id: str,
	init_value: object,
	*,
	role: str,
	expected_objective: str,
	depth: int = 0,
) -> None:
	"""Walk one recorded lineage edge for objective and budget compliance."""
	if not isinstance(init_value, str) or not init_value:
		raise ValueError(f'{model_id} checkpoint must record {role}')
	if 'trace_drop' in init_value:
		raise ValueError(
			f'{model_id} {role} is a trace-drop artifact: {init_value}'
		)
	init_path = Path(init_value)
	if not init_path.is_file():
		# An unverifiable ancestor cannot certify the 100+25 budget, the base
		# objective, or the absence of trace drop, so the audit fails closed.
		raise FileNotFoundError(
			f'{model_id} {role} does not exist: {init_path}'
		)
	if depth >= _MAX_LINEAGE_DEPTH:
		raise ValueError(
			f'{model_id} lineage is deeper than the audit can verify at {init_path}'
		)
	try:
		base_payload = load_checkpoint_metadata_without_weights(init_path)
	except Exception as error:
		raise ValueError(
			f'{model_id} {role} is unreadable: {init_path}'
		) from error
	base_config = base_payload.get('config')
	if not isinstance(base_config, Mapping):
		raise ValueError(  # noqa: TRY004 - missing config is a value error
			f'{model_id} {role} does not record its resolved config: {init_path}'
		)
	_validate_base_objective(
		model_id,
		base_config,
		expected_objective=expected_objective,
		role=role,
	)
	if depth == 0:
		# The suite's stage-1 sources are the 100-epoch runs; a shorter base
		# breaks the 100+25 budget parity the comparison rests on.
		if base_payload.get('epoch') != FIVE_WAY_STAGE1_EPOCHS:
			raise ValueError(
				f'{model_id} {role} must be the {FIVE_WAY_STAGE1_EPOCHS} epoch '
				f'stage-1 source; got epoch {base_payload.get("epoch")!r}'
			)
		_validate_contract(
			model_id,
			base_config.get('train'),
			FIVE_WAY_STAGE1_TRAIN_CONTRACT,
			prefix=f'{role} train',
		)
	continuation = base_config.get('continuation')
	if isinstance(continuation, Mapping) and continuation.get('init_checkpoint'):
		_validate_ancestry(
			model_id,
			continuation.get('init_checkpoint'),
			role=f'{role} ancestor',
			expected_objective=expected_objective,
			depth=depth + 1,
		)


def _validate_fixed_budget(
	model: F3FiveWayModelSource,
	payload: Mapping[str, object],
	*,
	base_config: Mapping[str, object],
) -> None:
	label = model.model_id
	_validate_budget_counters(label, payload)
	if not model.expected['stratigraphy_pretext']:
		_validate_continuation_budget(model, base_config)
		return
	_validate_pretext_budget(model, payload)


def _validate_continuation_budget(
	model: F3FiveWayModelSource, base_config: Mapping[str, object]
) -> None:
	label = model.model_id
	continuation = base_config.get('continuation')
	if not isinstance(continuation, Mapping):
		raise ValueError(  # noqa: TRY004 - missing config is a value error
			f'{label} checkpoint must record the fixed-budget continuation '
			'that produced it'
		)
	if continuation.get('unfreeze_top_blocks') != FIVE_WAY_UNFREEZE_TOP_BLOCKS:
		raise ValueError(
			f'{label} checkpoint continuation.unfreeze_top_blocks must equal '
			f'{FIVE_WAY_UNFREEZE_TOP_BLOCKS}'
		)
	_validate_contract(
		label,
		base_config.get('train'),
		FIVE_WAY_STAGE2_TRAIN_CONTRACT,
		prefix='train',
	)
	_validate_ancestry(
		label,
		continuation.get('init_checkpoint'),
		role='continuation.init_checkpoint',
		expected_objective=_expected_base_objective(model),
	)


def _validate_pretext_budget(
	model: F3FiveWayModelSource, payload: Mapping[str, object]
) -> None:
	label = model.model_id
	training_state = payload.get('training_state')
	if not isinstance(training_state, Mapping) or (
		training_state.get('stage') != 'train_strat_hmm_pretext'
	):
		raise ValueError(
			f'{label} checkpoint training_state.stage must equal '
			"'train_strat_hmm_pretext'"
		)
	stratigraphy = payload.get('stratigraphy_config')
	if not isinstance(stratigraphy, Mapping):
		raise ValueError(  # noqa: TRY004 - missing config is a value error
			f'{label} checkpoint must record its stratigraphy config'
		)
	_validate_contract(
		label,
		stratigraphy.get('train'),
		FIVE_WAY_PRETEXT_TRAIN_CONTRACT,
		prefix='stratigraphy_config.train',
	)
	_validate_contract(
		label,
		stratigraphy.get('head'),
		FIVE_WAY_HMM_HEAD_CONTRACT,
		prefix='stratigraphy_config.head',
	)
	_validate_contract(
		label,
		stratigraphy.get('loss'),
		FIVE_WAY_HMM_LOSS_CONTRACT,
		prefix='stratigraphy_config.loss',
	)
	student = stratigraphy.get('student')
	if not isinstance(student, Mapping) or (
		student.get('unfreeze_top_blocks') != FIVE_WAY_UNFREEZE_TOP_BLOCKS
	):
		raise ValueError(
			f'{label} checkpoint stratigraphy_config.student.unfreeze_top_blocks '
			f'must equal {FIVE_WAY_UNFREEZE_TOP_BLOCKS}'
		)
	teacher = stratigraphy.get('teacher')
	if not isinstance(teacher, Mapping):
		raise ValueError(  # noqa: TRY004 - missing config is a value error
			f'{label} checkpoint must record stratigraphy_config.teacher'
		)
	if teacher.get('checkpoint') != student.get('init_checkpoint'):
		raise ValueError(
			f'{label} checkpoint teacher and student must start from the same '
			'stage-1 source'
		)
	_validate_ancestry(
		label,
		student.get('init_checkpoint'),
		role='stratigraphy_config.student.init_checkpoint',
		expected_objective=_expected_base_objective(model),
	)
	pseudo_targets = stratigraphy.get('pseudo_targets')
	if not isinstance(pseudo_targets, Mapping) or (
		pseudo_targets.get('k') != FIVE_WAY_HMM_K
	):
		raise ValueError(
			f'{label} checkpoint stratigraphy_config.pseudo_targets.k must equal '
			f'{FIVE_WAY_HMM_K}'
		)
	if pseudo_targets.get('min_confidence') != FIVE_WAY_MIN_CONFIDENCE:
		# A confidence gate silently shrinks supervision inside the same budget.
		raise ValueError(
			f'{label} checkpoint stratigraphy_config.pseudo_targets.'
			f'min_confidence must equal {FIVE_WAY_MIN_CONFIDENCE}'
		)
	target_dir = pseudo_targets.get('input_dir')
	if isinstance(target_dir, str) and 'trace_drop' in target_dir:
		raise ValueError(
			f'{label} checkpoint was trained on trace-drop pseudo targets: '
			f'{target_dir}'
		)


def _validate_budget_counters(
	label: str, payload: Mapping[str, object]
) -> None:
	for key, expected_value in (
		('epoch', FIVE_WAY_STAGE2_EPOCHS),
		('global_step', FIVE_WAY_STAGE2_GLOBAL_STEPS),
	):
		value = payload.get(key)
		if value != expected_value:
			raise ValueError(
				f'{label} checkpoint {key} must equal the fixed budget '
				f'{expected_value}; got {value!r}'
			)


def _validate_random_checkpoint_payload(  # noqa: C901
	payload: Mapping[str, object], *, mae_checkpoint: Path
) -> None:
	metadata = payload.get('metadata')
	if not isinstance(metadata, Mapping):
		raise TypeError('random checkpoint metadata must be a mapping')
	if metadata.get('random_encoder_baseline') is not True:
		raise ValueError(
			'random checkpoint metadata.random_encoder_baseline must equal True'
		)
	if metadata.get('pretrained_weights_loaded') is not False:
		raise ValueError(
			'random checkpoint metadata.pretrained_weights_loaded must equal False'
		)
	if metadata.get('seed') != FIVE_WAY_RANDOM_SEED:
		raise ValueError(
			f'random checkpoint metadata.seed must equal {FIVE_WAY_RANDOM_SEED}'
		)
	training_state = payload.get('training_state')
	if not isinstance(training_state, Mapping) or (
		training_state.get('checkpoint_kind') != 'random_init'
	):
		raise ValueError(
			"random checkpoint training_state.checkpoint_kind must equal "
			"'random_init'"
		)
	if training_state.get('stage') != RANDOM_CHECKPOINT_STAGE:
		raise ValueError(
			f'random checkpoint training_state.stage must equal '
			f'{RANDOM_CHECKPOINT_STAGE!r}'
		)
	# Random is an untrained representation, never a budget-matched model.
	for key in ('epoch', 'global_step'):
		if payload.get(key) != 0:
			raise ValueError(
				f'random checkpoint {key} must equal 0; got {payload.get(key)!r}'
			)
	reference_value = metadata.get('reference_checkpoint')
	if not isinstance(reference_value, str) or not reference_value:
		raise ValueError(
			'random checkpoint metadata.reference_checkpoint must be non-empty'
		)
	if Path(reference_value).resolve(strict=False) != mae_checkpoint.resolve(
		strict=False
	):
		raise ValueError(
			'random checkpoint reference_checkpoint must equal the configured '
			'mae checkpoint'
		)


def _read_json(path: Path) -> Mapping[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		raise TypeError(f'{path} must contain a JSON object')
	return payload


__all__ = [
	'EXPECTED_ENCODER_GEOMETRY',
	'EXPECTED_EXTRACTION_CONTRACT',
	'EXPECTED_PSEUDO_TARGET_SUFFIXES',
	'RANDOM_CHECKPOINT_STAGE',
	'SHARED_METADATA_KEYS',
	'STRAT_HMM_PRETEXT_METHOD',
	'TRACE_DROP_AUGMENTATION_KEYS',
	'audit_f3_lithology_five_way_sources',
	'plan_f3_lithology_five_way_sources',
]
