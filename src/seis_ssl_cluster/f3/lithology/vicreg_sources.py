"""VICReg source configuration, lineage, and screening-gate audits."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_required_str,
)
from seis_ssl_cluster.config.f3_lithology_five_way import (
	F3FiveWayConfig,
	F3FiveWayModelSource,
	f3_lithology_five_way_config_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	F3LithologyCandidateConfig,
	audit_f3_lithology_candidate_source,
)
from seis_ssl_cluster.f3.lithology.five_way_sources import (
	audit_f3_lithology_five_way_sources,
)
from seis_ssl_cluster.stratigraphy import discover_pseudo_target_inputs
from seis_ssl_cluster.training.checkpoint import load_checkpoint

SCREENING_MODEL_IDS = ('local_vicreg_100', 'random')
EXTENSION_MODEL_IDS = ('local_vicreg', 'local_vicreg_hmm_k6')
SCREENING_DATA_SIZE = 'medium'
VICREG_METHOD = 'local_vicreg_3d'
VICREG_CHECKPOINT_KIND = 'vicreg_pretraining'
VICREG_LOCAL_PAIRS_PER_CROP = 128
VICREG_STAGE1_EPOCHS = 100
VICREG_STAGE1_GLOBAL_STEPS = 62_500
VICREG_STAGE2_EPOCHS = 25
VICREG_STAGE2_GLOBAL_STEPS = 15_625
VICREG_HMM_K = 6
VICREG_HMM_TARGET_SUFFIX = (
	'pseudo_targets',
	'f3',
	'facies_benchmark_v1',
	'local_vicreg_v1',
	'vicreg100',
)
VICREG_MODEL_CONTRACT: Mapping[str, object] = {
	'patch_size': [8, 8, 8],
	'encoder_dim': 384,
	'encoder_depth': 8,
	'encoder_heads': 6,
	'decoder_dim': 256,
	'decoder_depth': 4,
	'decoder_heads': 4,
}
VICREG_OBJECTIVE_CONTRACT: Mapping[str, object] = {
	'method': VICREG_METHOD,
	'local_pairs_per_crop': VICREG_LOCAL_PAIRS_PER_CROP,
	'projector_dim': 384,
	'invariance_weight': 25.0,
	'variance_weight': 25.0,
	'covariance_weight': 1.0,
	'variance_target_std': 1.0,
	'variance_eps': 1.0e-4,
}
VICREG_AUGMENTATION_CONTRACT: Mapping[str, object] = {
	'horizontal_flip_probability': 0.5,
}
VICREG_GATE_PASS = 'VICREG_BASELINE_GATE_PASS'  # noqa: S105
VICREG_GATE_FAIL = 'VICREG_BASELINE_GATE_FAIL'


@dataclass(frozen=True)
class F3VICRegOutputRoots:
	"""Disjoint run, log, and summary roots for one benchmark suite."""

	runs_root: Path
	job_logs_root: Path
	summary_root: Path


@dataclass(frozen=True)
class F3VICRegExtensionConfig:
	"""Resolved VICReg screening and two-arm extension configuration."""

	canonical_config: Path
	screening_model: F3FiveWayModelSource
	extension_models: tuple[F3FiveWayModelSource, ...]
	screening_outputs: F3VICRegOutputRoots
	extension_outputs: F3VICRegOutputRoots
	combined_summary_root: Path

	def extension_model_by_id(self, model_id: str) -> F3FiveWayModelSource:
		"""Return one of the exact two extension sources."""
		for model in self.extension_models:
			if model.model_id == model_id:
				return model
		raise ValueError(
			f'unknown VICReg extension model: {model_id!r}; '
			f'expected one of {list(EXTENSION_MODEL_IDS)!r}'
		)


def f3_vicreg_extension_config_from_mapping(
	config: Mapping[str, object],
) -> F3VICRegExtensionConfig:
	"""Resolve the exact one-screening-source/two-extension-source schema."""
	_require_exact_keys(config, {'benchmark', 'screening', 'extension'}, 'config')
	benchmark = _required_mapping(config, 'benchmark')
	screening = _required_mapping(config, 'screening')
	extension = _required_mapping(config, 'extension')
	_require_exact_keys(benchmark, {'canonical_config'}, 'benchmark')
	_require_exact_keys(screening, {'model', 'outputs'}, 'screening')
	_require_exact_keys(extension, {'models', 'outputs'}, 'extension')

	screening_model = _resolve_source(
		screening.get('model'),
		expected_id=SCREENING_MODEL_IDS[0],
		label='screening.model',
	)
	extension_models = _resolve_extension_sources(extension.get('models'))
	screening_outputs = _resolve_output_roots(
		screening.get('outputs'), label='screening.outputs'
	)
	extension_outputs, combined_summary_root = _resolve_extension_output_roots(
		extension.get('outputs')
	)
	all_output_roots = (
		screening_outputs.runs_root,
		screening_outputs.job_logs_root,
		screening_outputs.summary_root,
		extension_outputs.runs_root,
		extension_outputs.job_logs_root,
		extension_outputs.summary_root,
		combined_summary_root,
	)
	_validate_disjoint_roots(all_output_roots, label='VICReg benchmark outputs')
	return F3VICRegExtensionConfig(
		canonical_config=_required_absolute_path(
			benchmark, 'canonical_config', prefix='benchmark'
		),
		screening_model=screening_model,
		extension_models=extension_models,
		screening_outputs=screening_outputs,
		extension_outputs=extension_outputs,
		combined_summary_root=combined_summary_root,
	)


def load_f3_vicreg_canonical_config(
	config: F3VICRegExtensionConfig,
) -> F3FiveWayConfig:
	"""Load the existing exact-five configuration and enforce read-only separation."""
	if not config.canonical_config.is_file():
		raise FileNotFoundError(
			f'canonical five-way config does not exist: {config.canonical_config}'
		)
	canonical = f3_lithology_five_way_config_from_mapping(
		load_config(config.canonical_config)
	)
	for output in (
		config.screening_outputs.runs_root,
		config.screening_outputs.job_logs_root,
		config.screening_outputs.summary_root,
		config.extension_outputs.runs_root,
		config.extension_outputs.job_logs_root,
		config.extension_outputs.summary_root,
		config.combined_summary_root,
	):
		for canonical_root in (canonical.runs_root, canonical.summary_root):
			if _paths_overlap(output, canonical_root):
				raise ValueError(
					'VICReg output overlaps read-only canonical five-way output: '
					f'{output} and {canonical_root}'
				)
	return canonical


def audit_f3_vicreg_screening_source(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Audit only the stage-1 screening source and canonical random baseline."""
	return _audit_f3_vicreg_sources(
		config,
		canonical,
		specs=((config.screening_model, 'screening', config.screening_outputs),),
	)


def audit_f3_vicreg_sources(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Audit canonical sources and all three VICReg checkpoint/embedding lineages."""
	return _audit_f3_vicreg_sources(
		config,
		canonical,
		specs=(
			(config.screening_model, 'screening', config.screening_outputs),
			(config.extension_models[0], 'control', config.extension_outputs),
			(config.extension_models[1], 'hmm', config.extension_outputs),
		),
	)


def _audit_f3_vicreg_sources(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
	*,
	specs: Sequence[tuple[F3FiveWayModelSource, str, F3VICRegOutputRoots]],
) -> dict[str, object]:
	canonical_report = audit_f3_lithology_five_way_sources(canonical)
	canonical_sources = _canonical_source_provenance(canonical)
	sources: list[dict[str, object]] = []
	canonical_random: object | None = None
	for source, role, outputs in specs:
		candidate = F3LithologyCandidateConfig(
			canonical_config=config.canonical_config,
			candidate_id=source.model_id,
			checkpoint=source.checkpoint,
			embeddings_dir=source.embeddings_dir,
			runs_root=outputs.runs_root,
			summary_root=outputs.summary_root,
		)
		provenance = audit_f3_lithology_candidate_source(candidate, canonical)
		lineage = _validate_vicreg_checkpoint(source.checkpoint, role=role)
		_validate_vicreg_embedding_metadata(
			Path(str(provenance['embedding_metadata_path'])),
			checkpoint=source.checkpoint,
			role=role,
		)
		current_random = provenance['canonical_random']
		if canonical_random is None:
			canonical_random = current_random
		elif canonical_random != current_random:
			raise ValueError('canonical random provenance changed during VICReg audit')
		sources.append({**provenance, 'role': role, 'lineage': lineage})
	_validate_shared_vicreg100_lineage(sources)
	return {
		'model_order': [source.model_id for source, _role, _outputs in specs],
		'canonical_model_order': list(canonical.model_ids),
		'canonical_source_audit': canonical_report,
		'canonical_sources': canonical_sources,
		'sources': sources,
		'canonical_random': canonical_random,
	}


def _canonical_source_provenance(
	canonical: F3FiveWayConfig,
) -> list[dict[str, object]]:
	survey_id = canonical.dataset['name']
	provenance = []
	for model in canonical.models:
		files = output_paths(model.embeddings_dir, survey_id)
		metadata = _read_json(
			files.metadata, label=f'{model.model_id} embedding metadata'
		)
		provenance.append(
			{
				'model_id': model.model_id,
				'checkpoint_sha256': file_sha256(model.checkpoint),
				'embeddings_sha256': file_sha256(files.embeddings),
				'embedding_metadata_sha256': file_sha256(files.metadata),
				'valid_tokens_sha256': file_sha256(files.valid_tokens),
				'recorded_checkpoint_sha256': metadata.get('checkpoint_sha256'),
			}
		)
	return provenance


def _validate_shared_vicreg100_lineage(
	sources: Sequence[Mapping[str, object]],
) -> None:
	by_role = {str(source.get('role')): source for source in sources}
	if set(by_role) != {'screening', 'control', 'hmm'}:
		return
	screening = by_role['screening']
	screening_path = Path(str(screening['checkpoint_path'])).resolve(strict=False)
	screening_sha256 = screening.get('checkpoint_sha256')
	for role in ('control', 'hmm'):
		lineage = by_role[role].get('lineage')
		if not isinstance(lineage, Mapping):
			raise TypeError(f'{role} VICReg lineage must be a mapping')
		source_path = Path(str(lineage.get('source_checkpoint'))).resolve(strict=False)
		if source_path != screening_path:
			raise ValueError(
				f'{role} lineage must use the screened VICReg100 checkpoint path'
			)
		if lineage.get('source_checkpoint_sha256') != screening_sha256:
			raise ValueError(
				f'{role} lineage must use the screened VICReg100 checkpoint SHA'
			)


def f3_vicreg_screening_gate_from_mapping(
	payload: Mapping[str, object], *, path: Path
) -> str:
	"""Validate and return one already-read screening summary gate."""
	gate = payload.get('gate_status')
	if gate not in {VICREG_GATE_PASS, VICREG_GATE_FAIL}:
		raise ValueError(
			f'{path} gate_status must be {VICREG_GATE_PASS!r} or '
			f'{VICREG_GATE_FAIL!r}; got {gate!r}'
		)
	return str(gate)


def _validate_vicreg_checkpoint(  # noqa: C901, PLR0912, PLR0915
	path: Path, *, role: str
) -> dict[str, object]:
	payload = load_checkpoint(path, map_location='cpu')
	if not isinstance(payload, Mapping):
		raise TypeError(f'VICReg checkpoint payload must be a mapping: {path}')
	config = _required_checkpoint_mapping(payload, 'config', path=path)
	_validate_vicreg_base_config(config, label=f'{path} config')
	if role == 'screening':
		_validate_checkpoint_counters(
			payload,
			epochs=VICREG_STAGE1_EPOCHS,
			global_steps=VICREG_STAGE1_GLOBAL_STEPS,
			label=str(path),
		)
		_validate_direct_vicreg_identity(payload, path=path)
		return {'role': role, 'source_checkpoint': None}
	if role == 'control':
		_validate_checkpoint_counters(
			payload,
			epochs=VICREG_STAGE2_EPOCHS,
			global_steps=VICREG_STAGE2_GLOBAL_STEPS,
			label=str(path),
		)
		_validate_direct_vicreg_identity(payload, path=path)
		continuation = _required_checkpoint_mapping(config, 'continuation', path=path)
		if continuation.get('unfreeze_top_blocks') != 1:
			raise ValueError(f'{path} continuation.unfreeze_top_blocks must be 1')
		source = _required_existing_path(
			continuation.get('init_checkpoint'),
			label=f'{path} continuation.init_checkpoint',
		)
		source_sha256 = _validate_stage1_source(source)
		_validate_vicreg_stage1_identity(config, source=source, label=f'{path} control')
		lineage = _required_checkpoint_mapping(
			payload, 'continuation_lineage', path=path
		)
		if lineage.get('init_checkpoint') != str(source):
			raise ValueError(
				f'{path} continuation lineage source does not match config'
			)
		if lineage.get('init_checkpoint_sha256') != source_sha256:
			raise ValueError(f'{path} continuation lineage source SHA does not match')
		return {
			'role': role,
			'source_checkpoint': str(source),
			'source_checkpoint_sha256': source_sha256,
		}
	if role != 'hmm':
		raise ValueError(f'unsupported VICReg checkpoint role: {role!r}')
	_validate_checkpoint_counters(
		payload,
		epochs=VICREG_STAGE2_EPOCHS,
		global_steps=VICREG_STAGE2_GLOBAL_STEPS,
		label=str(path),
	)
	stratigraphy = _required_checkpoint_mapping(
		payload, 'stratigraphy_config', path=path
	)
	if stratigraphy.get('stage') != 'train_strat_hmm_pretext':
		raise ValueError(f'{path} stratigraphy_config.stage is invalid')
	teacher = _required_checkpoint_mapping(stratigraphy, 'teacher', path=path)
	student = _required_checkpoint_mapping(stratigraphy, 'student', path=path)
	teacher_path = _required_existing_path(
		teacher.get('checkpoint'), label=f'{path} teacher.checkpoint'
	)
	student_path = _required_existing_path(
		student.get('init_checkpoint'), label=f'{path} student.init_checkpoint'
	)
	if teacher_path.resolve(strict=False) != student_path.resolve(strict=False):
		raise ValueError(f'{path} HMM teacher and student must use the same VICReg100')
	if student.get('unfreeze_top_blocks') != 1:
		raise ValueError(f'{path} student.unfreeze_top_blocks must be 1')
	head = _required_checkpoint_mapping(stratigraphy, 'head', path=path)
	if head.get('num_prototypes') != VICREG_HMM_K:
		raise ValueError(f'{path} HMM head num_prototypes must be {VICREG_HMM_K}')
	pseudo_targets = _required_checkpoint_mapping(
		stratigraphy, 'pseudo_targets', path=path
	)
	if pseudo_targets.get('k') != VICREG_HMM_K:
		raise ValueError(f'{path} pseudo_targets.k must be {VICREG_HMM_K}')
	if float(pseudo_targets.get('min_confidence', -1.0)) != 0.0:
		raise ValueError(f'{path} pseudo_targets.min_confidence must be 0.0')
	pseudo_target_dir = _validate_hmm_pseudo_target_dir(
		pseudo_targets.get('input_dir'), path=path
	)
	loss = _required_checkpoint_mapping(stratigraphy, 'loss', path=path)
	if float(loss.get('distillation_weight', -1.0)) != 0.2:
		raise ValueError(f'{path} distillation_weight must be 0.2')
	train = _required_checkpoint_mapping(stratigraphy, 'train', path=path)
	if train.get('epochs') != VICREG_STAGE2_EPOCHS:
		raise ValueError(f'{path} HMM config must declare 25 epochs')
	source_sha256 = _validate_stage1_source(teacher_path)
	_validate_vicreg_stage1_identity(
		config, source=teacher_path, label=f'{path} HMM base'
	)
	_validate_hmm_control_identity(
		payload,
		path=path,
		stratigraphy=stratigraphy,
		teacher_path=teacher_path,
		student_path=student_path,
		pseudo_target_dir=pseudo_target_dir,
	)
	return {
		'role': role,
		'source_checkpoint': str(teacher_path),
		'source_checkpoint_sha256': source_sha256,
		'hmm_k': VICREG_HMM_K,
	}


def _validate_hmm_pseudo_target_dir(value: object, *, path: Path) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{path} pseudo_targets.input_dir must be a non-empty string')
	target_dir = Path(value)
	if not target_dir.is_absolute():
		raise ValueError(f'{path} pseudo_targets.input_dir must be absolute')
	if tuple(target_dir.parts[-len(VICREG_HMM_TARGET_SUFFIX) :]) != (
		VICREG_HMM_TARGET_SUFFIX
	):
		raise ValueError(
			f'{path} HMM pseudo-target path must end with '
			f'{"/".join(VICREG_HMM_TARGET_SUFFIX)!r}'
		)
	return target_dir


def _validate_hmm_control_identity(  # noqa: C901, PLR0912, PLR0913
	payload: Mapping[str, object],
	*,
	path: Path,
	stratigraphy: Mapping[str, object],
	teacher_path: Path,
	student_path: Path,
	pseudo_target_dir: Path,
) -> None:
	identity = _required_checkpoint_mapping(stratigraphy, 'identity', path=path)
	model_tag = identity.get('model_tag')
	if not isinstance(model_tag, str) or not model_tag:
		raise ValueError(f'{path} identity.model_tag must be a non-empty string')
	control = _required_checkpoint_mapping(payload, 'control_identity', path=path)
	if control.get('schema_version') != 1:
		raise ValueError(f'{path} control_identity.schema_version must be 1')
	if control.get('model_tag') != model_tag:
		raise ValueError(
			f'{path} control_identity.model_tag does not match stratigraphy identity'
		)
	inputs = _required_checkpoint_mapping(control, 'input_identities', path=path)
	_validate_recorded_file_identity(
		inputs.get('teacher_checkpoint'),
		expected_path=teacher_path,
		label=f'{path} control_identity teacher checkpoint',
	)
	_validate_recorded_file_identity(
		inputs.get('student_init_checkpoint'),
		expected_path=student_path,
		label=f'{path} control_identity student init checkpoint',
	)
	recorded_targets = inputs.get('pseudo_targets')
	if not isinstance(recorded_targets, Sequence) or isinstance(
		recorded_targets, str | bytes
	):
		raise TypeError(f'{path} control_identity pseudo_targets must be a list')
	recorded_by_survey: dict[str, Mapping[str, object]] = {}
	for index, recorded in enumerate(recorded_targets):
		if not isinstance(recorded, Mapping):
			raise TypeError(
				f'{path} control_identity pseudo_targets[{index}] must be a mapping'
			)
		survey_id = recorded.get('survey_id')
		if not isinstance(survey_id, str) or not survey_id:
			raise ValueError(
				f'{path} control_identity pseudo_targets[{index}].survey_id '
				'must be non-empty'
			)
		if survey_id in recorded_by_survey:
			raise ValueError(
				f'{path} control_identity has duplicate pseudo-target survey '
				f'{survey_id!r}'
			)
		recorded_by_survey[survey_id] = recorded
	current_targets = discover_pseudo_target_inputs(pseudo_target_dir, k=VICREG_HMM_K)
	current_by_survey = {item.survey_id: item for item in current_targets}
	if set(recorded_by_survey) != set(current_by_survey):
		raise ValueError(
			f'{path} control_identity pseudo-target survey set does not match '
			'current inputs'
		)
	for survey_id, current in current_by_survey.items():
		recorded = recorded_by_survey[survey_id]
		for field, current_path in (
			('labels', current.labels_path),
			('confidence', current.confidence_path),
			('valid_tokens', current.valid_tokens_path),
			('metadata', current.metadata_path),
		):
			_validate_recorded_file_identity(
				recorded.get(field),
				expected_path=current_path,
				label=(f'{path} control_identity pseudo-target {survey_id} {field}'),
			)
		boundary_present = recorded.get('boundary_weight_present')
		if not isinstance(boundary_present, bool):
			raise TypeError(
				f'{path} control_identity pseudo-target {survey_id} '
				'boundary_weight_present must be boolean'
			)
		current_boundary_present = current.boundary_weight_path is not None
		if boundary_present != current_boundary_present:
			raise ValueError(
				f'{path} control_identity pseudo-target {survey_id} boundary '
				'presence does not match current input'
			)
		if current.boundary_weight_path is not None:
			_validate_recorded_file_identity(
				recorded.get('boundary_weight'),
				expected_path=current.boundary_weight_path,
				label=(
					f'{path} control_identity pseudo-target {survey_id} boundary_weight'
				),
			)
		elif 'boundary_weight' in recorded:
			raise ValueError(
				f'{path} control_identity pseudo-target {survey_id} has an '
				'unexpected boundary_weight identity'
			)


def _validate_recorded_file_identity(
	value: object, *, expected_path: Path, label: str
) -> None:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	recorded_path = _required_existing_path(value.get('path'), label=f'{label}.path')
	if recorded_path.resolve(strict=False) != expected_path.resolve(strict=False):
		raise ValueError(f'{label} path does not match current input')
	recorded_sha256 = _sha256(value.get('sha256'), label=f'{label}.sha256')
	if recorded_sha256 != file_sha256(expected_path):
		raise ValueError(f'{label} SHA does not match current input')


def _validate_stage1_source(path: Path) -> str:
	payload = load_checkpoint(path, map_location='cpu')
	if not isinstance(payload, Mapping):
		raise TypeError(f'VICReg100 source payload must be a mapping: {path}')
	config = _required_checkpoint_mapping(payload, 'config', path=path)
	_validate_vicreg_base_config(config, label=f'{path} config')
	if 'continuation' in config:
		raise ValueError(f'VICReg100 source must not be a continuation: {path}')
	_validate_checkpoint_counters(
		payload,
		epochs=VICREG_STAGE1_EPOCHS,
		global_steps=VICREG_STAGE1_GLOBAL_STEPS,
		label=str(path),
	)
	_validate_direct_vicreg_identity(payload, path=path)
	return file_sha256(path)


def _validate_direct_vicreg_identity(
	payload: Mapping[str, object], *, path: Path
) -> None:
	if payload.get('pretraining_method') != VICREG_METHOD:
		raise ValueError(f'{path} pretraining_method must be {VICREG_METHOD!r}')
	if payload.get('checkpoint_kind') != VICREG_CHECKPOINT_KIND:
		raise ValueError(f'{path} checkpoint_kind must be {VICREG_CHECKPOINT_KIND!r}')
	if not isinstance(payload.get('projector_state_dict'), Mapping):
		raise TypeError(f'{path} projector_state_dict must be a mapping')
	model_state = payload.get('model_state_dict')
	if not isinstance(model_state, Mapping):
		raise TypeError(f'{path} model_state_dict must be a mapping')
	if any(
		isinstance(key, str) and key.startswith(('backbone.', 'projector.'))
		for key in model_state
	):
		raise ValueError(f'{path} model_state_dict must contain bare encoder keys')


def _validate_vicreg_base_config(config: Mapping[str, object], *, label: str) -> None:
	if config.get('stage') != 'vicreg_training':
		raise ValueError(f'{label}.stage must be vicreg_training')
	_validate_contract_values(
		config.get('model'), expected=VICREG_MODEL_CONTRACT, label=f'{label}.model'
	)
	_validate_contract_values(
		config.get('vicreg'),
		expected=VICREG_OBJECTIVE_CONTRACT,
		label=f'{label}.vicreg',
	)
	augmentations = config.get('augmentations')
	if augmentations != VICREG_AUGMENTATION_CONTRACT:
		raise ValueError(
			f'{label}.augmentations must equal the forced-distinct horizontal-flip '
			f'contract {dict(VICREG_AUGMENTATION_CONTRACT)!r}'
		)


def _validate_contract_values(
	value: object, *, expected: Mapping[str, object], label: str
) -> None:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	for key, expected_value in expected.items():
		if value.get(key) != expected_value:
			raise ValueError(
				f'{label}.{key} must equal {expected_value!r}; got {value.get(key)!r}'
			)


def _validate_vicreg_stage1_identity(
	config: Mapping[str, object], *, source: Path, label: str
) -> None:
	payload = load_checkpoint(source, map_location='cpu')
	if not isinstance(payload, Mapping):
		raise TypeError(f'{label} source checkpoint payload must be a mapping')
	source_config = _required_checkpoint_mapping(payload, 'config', path=source)
	for section in ('data', 'zero_mask', 'model', 'vicreg', 'augmentations'):
		if config.get(section) != source_config.get(section):
			raise ValueError(
				f'{label} {section} identity differs from screened VICReg100'
			)


def _validate_checkpoint_counters(
	payload: Mapping[str, object],
	*,
	epochs: int,
	global_steps: int,
	label: str,
) -> None:
	if payload.get('epoch') != epochs or payload.get('global_step') != global_steps:
		raise ValueError(
			f'{label} must record epoch/global_step={epochs}/{global_steps}; '
			f'got {payload.get("epoch")!r}/{payload.get("global_step")!r}'
		)


def _validate_vicreg_embedding_metadata(  # noqa: C901, PLR0912
	path: Path, *, checkpoint: Path, role: str
) -> None:
	metadata = _read_json(path, label='VICReg embedding metadata')
	if metadata.get('pretraining_method') != VICREG_METHOD:
		raise ValueError(f'{path} pretraining_method must be {VICREG_METHOD!r}')
	payload = load_checkpoint(checkpoint, map_location='cpu')
	if not isinstance(payload, Mapping):
		raise TypeError(f'{checkpoint} payload must be a mapping')
	checkpoint_config = _required_checkpoint_mapping(payload, 'config', path=checkpoint)
	checkpoint_model = _required_checkpoint_mapping(
		checkpoint_config, 'model', path=checkpoint
	)
	model_geometry = metadata.get('model_geometry')
	if not isinstance(model_geometry, Mapping):
		raise TypeError(f'{path} model_geometry must be a mapping')
	for key, value in checkpoint_model.items():
		if model_geometry.get(key) != value:
			raise ValueError(
				f'{path} model_geometry.{key} does not match checkpoint config.model'
			)
	objective = metadata.get('pretraining_objective')
	if not isinstance(objective, Mapping) or objective.get('method') != VICREG_METHOD:
		raise ValueError(
			f'{path} pretraining_objective.method must be {VICREG_METHOD!r}'
		)
	checkpoint_vicreg = _required_checkpoint_mapping(
		checkpoint_config, 'vicreg', path=checkpoint
	)
	for key, value in checkpoint_vicreg.items():
		if objective.get(key) != value:
			raise ValueError(
				f'{path} pretraining_objective.{key} does not match checkpoint '
				'config.vicreg'
			)
	pretext = metadata.get('stratigraphy_pretext')
	if role != 'hmm':
		if pretext is not None:
			raise ValueError(f'{path} direct VICReg embedding must not declare pretext')
		return
	if not isinstance(pretext, Mapping):
		raise TypeError(f'{path} HMM embedding stratigraphy_pretext must be a mapping')
	for key, expected in (
		('method', 'strat_hmm_pretext'),
		('base_objective', VICREG_METHOD),
		('head_num_prototypes', VICREG_HMM_K),
		('unfreeze_top_blocks', 1),
		('distillation_weight', 0.2),
	):
		if pretext.get(key) != expected:
			raise ValueError(f'{path} HMM {key} must equal {expected!r}')
	pseudo_target_dir = pretext.get('pseudo_target_input_dir')
	if not isinstance(pseudo_target_dir, str) or not pseudo_target_dir:
		raise ValueError(f'{path} HMM pseudo_target_input_dir is required')
	if 'trace_drop' in pseudo_target_dir:
		raise ValueError(f'{path} HMM pseudo targets must not use trace drop')
	stratigraphy = _required_checkpoint_mapping(
		payload, 'stratigraphy_config', path=checkpoint
	)
	pseudo_targets = _required_checkpoint_mapping(
		stratigraphy, 'pseudo_targets', path=checkpoint
	)
	checkpoint_target_dir = pseudo_targets.get('input_dir')
	if pseudo_target_dir != checkpoint_target_dir:
		raise ValueError(
			f'{path} HMM pseudo_target_input_dir does not match checkpoint lineage'
		)
	if tuple(Path(pseudo_target_dir).parts[-len(VICREG_HMM_TARGET_SUFFIX) :]) != (
		VICREG_HMM_TARGET_SUFFIX
	):
		raise ValueError(
			f'{path} HMM pseudo-target path must end with '
			f'{"/".join(VICREG_HMM_TARGET_SUFFIX)!r}'
		)


def _resolve_extension_sources(value: object) -> tuple[F3FiveWayModelSource, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError('extension.models must be a list of two source mappings')
	if len(value) != len(EXTENSION_MODEL_IDS):
		raise ValueError('extension.models must contain exactly two entries')
	models = tuple(
		_resolve_source(item, expected_id=model_id, label=f'extension.models[{index}]')
		for index, (item, model_id) in enumerate(
			zip(value, EXTENSION_MODEL_IDS, strict=True)
		)
	)
	if len({model.checkpoint for model in models}) != len(models):
		raise ValueError('extension model checkpoints must be distinct')
	if len({model.embeddings_dir for model in models}) != len(models):
		raise ValueError('extension model embedding directories must be distinct')
	return models


def _resolve_source(
	value: object, *, expected_id: str, label: str
) -> F3FiveWayModelSource:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	_require_exact_keys(value, {'model_id', 'checkpoint', 'embeddings_dir'}, label)
	model_id = _required_str(value, 'model_id', prefix=label)
	if model_id != expected_id:
		raise ValueError(f'{label}.model_id must be {expected_id!r}; got {model_id!r}')
	checkpoint = _required_absolute_path(value, 'checkpoint', prefix=label)
	embeddings_dir = _required_absolute_path(value, 'embeddings_dir', prefix=label)
	if checkpoint == embeddings_dir:
		raise ValueError(f'{label} checkpoint and embeddings_dir must differ')
	return F3FiveWayModelSource(
		model_id=model_id,
		checkpoint=checkpoint,
		embeddings_dir=embeddings_dir,
		expected={
			'objective': VICREG_METHOD,
			'stratigraphy_pretext': model_id.endswith('_hmm_k6'),
		},
	)


def _resolve_output_roots(value: object, *, label: str) -> F3VICRegOutputRoots:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	_require_exact_keys(value, {'runs_root', 'job_logs_root', 'summary_root'}, label)
	return F3VICRegOutputRoots(
		runs_root=_required_absolute_path(value, 'runs_root', prefix=label),
		job_logs_root=_required_absolute_path(value, 'job_logs_root', prefix=label),
		summary_root=_required_absolute_path(value, 'summary_root', prefix=label),
	)


def _resolve_extension_output_roots(
	value: object,
) -> tuple[F3VICRegOutputRoots, Path]:
	if not isinstance(value, Mapping):
		raise TypeError('extension.outputs must be a mapping')
	_require_exact_keys(
		value,
		{'runs_root', 'job_logs_root', 'summary_root', 'combined_summary_root'},
		'extension.outputs',
	)
	return (
		F3VICRegOutputRoots(
			runs_root=_required_absolute_path(
				value, 'runs_root', prefix='extension.outputs'
			),
			job_logs_root=_required_absolute_path(
				value, 'job_logs_root', prefix='extension.outputs'
			),
			summary_root=_required_absolute_path(
				value, 'summary_root', prefix='extension.outputs'
			),
		),
		_required_absolute_path(
			value, 'combined_summary_root', prefix='extension.outputs'
		),
	)


def _validate_disjoint_roots(roots: Sequence[Path], *, label: str) -> None:
	for index, root in enumerate(roots):
		for other in roots[index + 1 :]:
			if _paths_overlap(root, other):
				raise ValueError(f'{label} must be disjoint: {root} and {other}')


def _paths_overlap(first: Path, second: Path) -> bool:
	first = first.resolve(strict=False)
	second = second.resolve(strict=False)
	return (
		first == second or first.is_relative_to(second) or second.is_relative_to(first)
	)


def _require_exact_keys(
	value: Mapping[str, object], expected: set[str], label: str
) -> None:
	keys = set(value)
	if keys != expected:
		missing = sorted(expected - keys)
		extra = sorted(str(key) for key in keys - expected)
		raise ValueError(
			f'{label} keys must be exactly {sorted(expected)!r}; '
			f'missing={missing!r}, unexpected={extra!r}'
		)


def _required_checkpoint_mapping(
	parent: Mapping[str, object], key: str, *, path: Path
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'{path} {key} must be a mapping')
	return value


def _required_existing_path(value: object, *, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty path string')
	path = Path(value)
	if not path.is_absolute():
		raise ValueError(f'{label} must be absolute')
	if not path.is_file():
		raise FileNotFoundError(f'{label} does not exist: {path}')
	return path


def _sha256(value: object, *, label: str) -> str:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise ValueError(f'{label} must be a lowercase SHA-256 digest')
	return value


def _read_json(path: Path, *, label: str) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(f'{label} is missing: {path}')
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as error:
		raise ValueError(f'{label} must contain JSON: {path}') from error
	if not isinstance(payload, Mapping):
		raise TypeError(f'{label} must contain a JSON object: {path}')
	return payload


__all__ = [
	'EXTENSION_MODEL_IDS',
	'SCREENING_DATA_SIZE',
	'SCREENING_MODEL_IDS',
	'VICREG_GATE_FAIL',
	'VICREG_GATE_PASS',
	'F3VICRegExtensionConfig',
	'F3VICRegOutputRoots',
	'audit_f3_vicreg_screening_source',
	'audit_f3_vicreg_sources',
	'f3_vicreg_extension_config_from_mapping',
	'f3_vicreg_screening_gate_from_mapping',
	'load_f3_vicreg_canonical_config',
]
