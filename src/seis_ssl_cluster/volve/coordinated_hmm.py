"""Serialize only future experiment-33 Random HMM writers, before GPU setup.

The two arms have independent locks; full and smoke of one arm share a lock.
Resource admission remains the caller's responsibility. This is not a GPU queue
or permission to interrupt an existing uncoordinated process.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.parihaka.coordinated_hmm import _file_lock, _validate_rng
from seis_ssl_cluster.stratigraphy import discover_pseudo_target_inputs
from seis_ssl_cluster.training.strat_hmm.resume import _validate_strat_resume_payload
from seis_ssl_cluster.volve.coordinated_hmm_state import validate_parent, validate_state
from seis_ssl_cluster.volve.horizon_five_way_sources import (
	_validate_pseudo_target_lineage,
)
from seis_ssl_cluster.volve.horizon_hmm_recipe import _audit_declared_clustering

EXPERIMENT = (
	Path(__file__).resolve().parents[3]
	/ 'experiments/volve/horizon_benchmark_v1/33_missing_hmm_comparison_v1'
)
RECIPES = ('random_init_hmm_k6_distill010', 'random_init_hmm_k6_distill020')
ARTIFACT_STEM = Path('pretraining/volve/horizon_benchmark_v1/missing_hmm_comparison_v1')
RUN_FILES = frozenset(
	{'latest.pt', 'best.pt', 'resolved_config.json', 'run_metadata.json'}
)


@dataclass(frozen=True)
class CoordinatedVolveHmmRun:
	"""An exact recipe and a shared full/smoke writer lock."""

	recipe: str
	config_path: Path
	config: dict[str, object]
	output_root: Path
	lock_path: Path
	smoke: bool


@dataclass(frozen=True)
class _ClusteringAuditSpec:
	"""Only the attributes consumed by the existing Volve clustering inspector."""

	hmm: Mapping[str, object]
	survey_id: str = 'volve_st10010'


def _mapping(value: object) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('coordinated Volve HMM evidence must be a mapping')
	return value


def _no_alias(path: Path) -> None:
	if not path.is_absolute() or '..' in path.parts:
		raise ValueError('coordinated Volve HMM refuses relative or aliased paths')
	for part in (path, *path.parents):
		if part.is_symlink():
			raise ValueError(f'coordinated Volve HMM refuses symlink: {part}')


def _regular(path: Path) -> None:
	_no_alias(path)
	if not path.is_file() or path.stat().st_nlink != 1:
		raise ValueError(f'coordinated Volve HMM requires an unaliased file: {path}')


def _json_hash(value: object) -> str:
	return hashlib.sha256(
		json.dumps(
			value, sort_keys=True, separators=(',', ':'), allow_nan=False
		).encode()
	).hexdigest()


def _snapshot(paths: list[Path]) -> dict[str, str]:
	result = {}
	for path in sorted(set(paths)):
		_regular(path)
		before = path.stat()
		result[str(path)] = file_sha256(path)
		after = path.stat()
		if (
			before.st_dev,
			before.st_ino,
			before.st_size,
			before.st_mtime_ns,
			before.st_ctime_ns,
		) != (
			after.st_dev,
			after.st_ino,
			after.st_size,
			after.st_mtime_ns,
			after.st_ctime_ns,
		):
			raise ValueError('coordinated Volve HMM input changed while hashing')
	return result


def classify_coordinated_volve_hmm_run(
	config: Mapping[str, object], *, config_path: Path
) -> CoordinatedVolveHmmRun | None:
	"""Classify four exact full/smoke scopes without writing or initializing CUDA."""
	output = Path(str(_mapping(config.get('paths', {})).get('output_root', '')))
	definitions = {
		recipe: EXPERIMENT / f'20_pretraining/{recipe}.yaml' for recipe in RECIPES
	}
	recipe = next(
		(
			key
			for key, path in definitions.items()
			if path.resolve() == config_path.resolve()
		),
		None,
	)
	protected = any(
		candidate.parts[-6:] == (*ARTIFACT_STEM.parts, name, kind)
		for candidate in (output, output.resolve())
		for name in RECIPES
		for kind in ('full_25ep', 'smoke_2step')
	)
	if recipe is None and not protected:
		return None
	if recipe is None:
		raise ValueError('canonical Volve HMM output requires its exact recipe config')
	_regular(config_path.absolute())
	_no_alias(output)
	expected = resolve_strat_hmm_pretext_config(load_config(definitions[recipe]))
	full = Path(str(_mapping(expected['paths'])['output_root']))
	if full.parts[-6:] != (*ARTIFACT_STEM.parts, recipe, 'full_25ep'):
		raise ValueError('canonical Volve HMM recipe has a foreign output root')
	smoke_root = full.parent / 'smoke_2step'
	if output not in (full, smoke_root):
		raise ValueError('coordinated Volve HMM output is outside the exact recipe')
	smoke = output == smoke_root
	if smoke:
		expected = copy.deepcopy(expected)
		expected['paths']['output_root'] = str(smoke_root)
		expected['train']['max_steps'] = 2
	if _json_hash(config) != _json_hash(expected):
		raise ValueError('coordinated Volve HMM config or CLI override differs')
	train = _mapping(expected['train'])
	if any(
		train.get(key) != value
		for key, value in {
			'epochs': 25,
			'samples_per_epoch': 10000,
			'batch_size': 4,
			'amp': False,
			'seed': 42,
			'device': 'cuda',
			'num_workers': 4,
			'allow_overwrite_output': False,
			'max_steps': 2 if smoke else None,
		}.items()
	) or _mapping(expected['loss'])['distillation_weight'] != (
		0.1 if recipe.endswith('010') else 0.2
	):
		raise ValueError('coordinated Volve HMM requires the immutable FP32 budget')
	lock = full.parent.parent / '.random_hmm_locks' / f'{recipe}.lock'
	_no_alias(lock)
	return CoordinatedVolveHmmRun(
		recipe, definitions[recipe], expected, output, lock, smoke
	)


def _expected_inputs(config: Mapping[str, object]) -> dict[str, object]:
	"""Read current parent and the exact single-survey, explicit-boundary targets."""
	pseudo = _mapping(config['pseudo_targets'])
	root = Path(str(pseudo['input_dir']))
	_no_alias(root)
	targets = discover_pseudo_target_inputs(root, k=6)
	if len(targets) != 1 or targets[0].survey_id != 'volve_st10010':
		raise ValueError('coordinated Volve HMM requires exactly one Volve K6 target')
	item = targets[0]
	if item.boundary_weight_path is None:
		raise ValueError('coordinated Volve HMM requires explicit boundary weights')
	files = {
		'labels': item.labels_path,
		'confidence': item.confidence_path,
		'valid_tokens': item.valid_tokens_path,
		'metadata': item.metadata_path,
		'boundary_weight': item.boundary_weight_path,
	}
	paths = list(files.values())
	parent = Path(str(_mapping(config['student'])['init_checkpoint']))
	teacher = Path(str(_mapping(config['teacher'])['checkpoint']))
	if parent != teacher:
		raise ValueError('Volve Random HMM must use its own random teacher')
	before = _snapshot([*paths, parent])
	validate_parent(config)
	metadata = json.loads(item.metadata_path.read_text(encoding='utf-8'))
	arrays = {
		name: np.load(path, mmap_mode='r', allow_pickle=False)
		for name, path in files.items()
		if name != 'metadata'
	}
	_validate_targets(arrays, metadata)
	if _snapshot([*paths, parent]) != before:
		raise ValueError('coordinated Volve HMM inputs changed during validation')
	return {
		'teacher_checkpoint': {'path': str(teacher), 'sha256': before[str(teacher)]},
		'student_init_checkpoint': {'path': str(parent), 'sha256': before[str(parent)]},
		'pseudo_targets': [
			{
				'survey_id': 'volve_st10010',
				'boundary_weight_present': True,
				**{
					name: {'path': str(path), 'sha256': before[str(path)]}
					for name, path in files.items()
				},
			}
		],
	}


def _validate_targets(
	arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]
) -> None:
	shape = tuple(metadata.get('token_grid_shape', ()))
	if (
		metadata.get('schema_version') != 2
		or metadata.get('survey_id') != 'volve_st10010'
		or metadata.get('k') != 6
		or len(shape) != 3
		or any(
			value.shape != shape or not np.isfinite(value).all()
			for value in arrays.values()
		)
	):
		raise ValueError('invalid coordinated Volve HMM target geometry or metadata')
	valid, labels = arrays['valid_tokens'], arrays['labels']
	if valid.dtype != np.bool_ or labels.dtype.kind not in 'iu' or not valid.any():
		raise ValueError('invalid coordinated Volve HMM target dtypes or valid mask')
	if (
		int(valid.sum()) != metadata.get('valid_token_count')
		or int((~valid).sum()) != metadata.get('invalid_token_count')
		or np.any((labels[valid] < 0) | (labels[valid] >= 6))
		or any(
			np.any(arrays[name][valid] != 1)
			for name in ('confidence', 'boundary_weight')
		)
		or {str(k): int((labels[valid] == k).sum()) for k in range(6)}
		!= metadata.get('label_counts')
	):
		raise ValueError('coordinated Volve HMM target values disagree with metadata')


def _input_snapshot(
	plan: CoordinatedVolveHmmRun,
) -> tuple[dict[str, object], dict[str, str]]:
	paths = [
		plan.config_path,
		*(Path(str(v)) for v in _mapping(plan.config['manifests']).values()),
	]
	before = _snapshot(paths)
	if (
		classify_coordinated_volve_hmm_run(plan.config, config_path=plan.config_path)
		!= plan
	):
		raise ValueError('coordinated Volve HMM fresh config changed during validation')
	inputs = _expected_inputs(plan.config)
	lineage = _lineage_snapshot(plan, inputs)
	if _snapshot(paths) != before or _expected_inputs(plan.config) != inputs:
		raise ValueError(
			'coordinated Volve HMM config/manifest changed during input validation'
		)
	return inputs, {**before, **lineage}


def _lineage_snapshot(
	plan: CoordinatedVolveHmmRun,
	inputs: Mapping[str, object],
) -> dict[str, str]:
	"""Reuse the source audit's parent-to-embedding-to-clustering-to-target checks."""
	definition = EXPERIMENT / '10_hmm_targets/random_init.yaml'
	first = _snapshot([definition])
	declared = load_config(definition)
	clustering = Path(str(_mapping(declared['clustering'])['output_dir']))
	embeddings = Path(str(_mapping(declared['embeddings'])['input_dir']))
	targets = Path(str(_mapping(plan.config['pseudo_targets'])['input_dir']))
	paths = [
		definition,
		clustering / 'models/k6/clustering_metadata.json',
		clustering / 'labels/k6/volve_st10010.cluster_label_metadata.json',
		clustering / 'labels/k6/volve_st10010.cluster_labels_token.npy',
		targets / 'k6/volve_st10010.pseudo_target_metadata.json',
		*(
			embeddings / f'volve_st10010.{suffix}'
			for suffix in (
				'embedding_metadata.json',
				'embeddings.npy',
				'valid_tokens.npy',
			)
		),
	]
	before = _snapshot(paths)
	if before[str(definition)] != first[str(definition)]:
		raise ValueError(
			'coordinated Volve HMM clustering config changed while loading'
		)
	parent = _mapping(inputs['student_init_checkpoint'])
	spec = _ClusteringAuditSpec(
		{
			'clustering_config': str(definition),
			'source_embeddings_dir': str(embeddings),
			'pseudo_targets_dir': str(targets),
		}
	)
	_audit_declared_clustering(spec, parent_sha=str(parent['sha256']))
	_validate_pseudo_target_lineage(
		plan.recipe,
		_mapping(plan.config['pseudo_targets']),
		checkpoint_payload={'control_identity': {'input_identities': inputs}},
		parent_path=Path(str(parent['path'])),
		parent_sha=str(parent['sha256']),
		survey_id='volve_st10010',
	)
	metadata = json.loads(paths[4].read_text(encoding='utf-8'))
	source = _mapping(metadata['source'])
	if (
		source.get('source_label_path') != str(paths[3])
		or source.get('source_label_sha256') != before[str(paths[3])]
		or not np.array_equal(
			np.load(paths[3], mmap_mode='r', allow_pickle=False),
			np.load(
				targets / 'k6/volve_st10010.hmm_labels_token.npy',
				mmap_mode='r',
				allow_pickle=False,
			),
		)
	):
		raise ValueError('coordinated Volve HMM targets differ from clustering labels')
	if _snapshot(paths) != before:
		raise ValueError('coordinated Volve HMM lineage changed during validation')
	return before


def _counters(payload: Mapping[str, object], *, smoke: bool) -> bool:
	epoch, step = payload['epoch'], payload['global_step']
	state = _mapping(payload['training_state'])
	kind, batch = state['checkpoint_kind'], state['batch_index']
	if (
		type(epoch) is not int
		or type(step) is not int
		or not (1 <= epoch <= 25 and 1 <= step <= 62500)
	):
		raise ValueError('coordinated Volve HMM counters exceed the exact budget')
	if kind == 'epoch':
		valid = batch is None and step == epoch * 2500
	elif kind == 'step':
		valid = (
			type(batch) is int
			and 0 <= batch < 2500
			and step == (epoch - 1) * 2500 + batch + 1
		)
	else:
		valid = False
	if not valid:
		raise ValueError('coordinated Volve HMM epoch/step/batch counters disagree')
	if smoke:
		if (epoch, step, kind, batch) != (1, 2, 'step', 1):
			raise ValueError('coordinated Volve HMM smoke must prove exactly two steps')
		return True
	if kind != 'epoch':
		raise ValueError('coordinated Volve HMM full resume requires an epoch boundary')
	return epoch == 25


def _validate_payload(
	payload: Mapping[str, object],
	plan: CoordinatedVolveHmmRun,
	inputs: Mapping[str, object],
) -> bool:
	_validate_strat_resume_payload(payload, amp_enabled=False)
	if any(
		key in payload
		for key in (
			'stratigraphy_checkpoint',
			'spatial_context_state_dict',
			'target_refresh_state',
			'checkpoint_selection',
			'epoch_metrics_state',
		)
	):
		raise ValueError('coordinated Volve HMM requires plain single-head state')
	_validate_rng(payload)
	_finite_rng(payload['rng_state'])
	if payload.get('scaler_state_dict') is not None:
		raise ValueError('coordinated Volve FP32 HMM must not contain a scaler')
	if payload['stratigraphy_config'] != plan.config:
		raise ValueError('coordinated Volve HMM saved full config differs')
	control = _mapping(payload.get('control_identity'))
	identity = _mapping(plan.config['identity'])
	if (
		control.get('schema_version') != 1
		or _mapping(payload['training_state']).get('schema_version') != 1
		or control.get('model_tag') != identity['model_tag']
		or control.get('scientific_identity') != identity.get('scientific_identity', {})
		or control.get('resolved_training_config_sha256') != _json_hash(plan.config)
		or control.get('input_identities') != inputs
	):
		raise ValueError(
			'coordinated Volve HMM control provenance differs from live inputs'
		)
	metrics = _mapping(payload.get('metrics'))
	if (
		not metrics
		or 'loss' not in metrics
		or any(
			type(value) not in (float, int) or not math.isfinite(value)
			for value in metrics.values()
		)
	):
		raise ValueError('coordinated Volve HMM metrics must be finite numbers')
	complete = _counters(payload, smoke=plan.smoke)
	validate_state(payload, plan.config)
	return complete


def _finite_rng(value: object) -> None:
	# setstate() accepts a NaN cached Gaussian, which poisons the next sample.
	if isinstance(value, Mapping):
		for child in value.values():
			_finite_rng(child)
	elif isinstance(value, list | tuple):
		for child in value:
			_finite_rng(child)
	elif isinstance(value, int | float | np.number) and not math.isfinite(value):
		raise ValueError('coordinated Volve HMM RNG numeric cache must be finite')


def _receipt_path(plan: CoordinatedVolveHmmRun) -> Path:
	return plan.lock_path.parent / f'{plan.recipe}.{plan.output_root.name}.inputs.json'


def _receipt(
	plan: CoordinatedVolveHmmRun,
	inputs: tuple[dict[str, object], dict[str, str]],
) -> dict[str, object]:
	return {
		'schema_version': 1,
		'artifact_type': 'coordinated_volve_hmm_inputs',
		'recipe': plan.recipe,
		'config_path': str(plan.config_path),
		'output_root': str(plan.output_root),
		'smoke': plan.smoke,
		'config_sha256': _json_hash(plan.config),
		'control_input_identities': inputs[0],
		'supporting_file_sha256': inputs[1],
	}


def _verify_receipt(
	plan: CoordinatedVolveHmmRun, expected: Mapping[str, object]
) -> None:
	path = _receipt_path(plan)
	_regular(path)
	before = _snapshot([path])
	if json.loads(path.read_text(encoding='utf-8')) != expected:
		raise ValueError('coordinated Volve HMM immutable input receipt differs')
	if _snapshot([path]) != before:
		raise ValueError('coordinated Volve HMM input receipt changed while reading')


def _create_receipt(
	plan: CoordinatedVolveHmmRun, expected: Mapping[str, object]
) -> None:
	"""Atomically create without replacement; never backfill existing checkpoints."""
	path = _receipt_path(plan)
	_no_alias(path)
	if path.exists():
		_verify_receipt(plan, expected)
		return
	if plan.output_root.exists() and any(plan.output_root.iterdir()):
		raise ValueError('coordinated Volve HMM never backfills missing input receipts')
	_no_alias(path.parent)
	fd, temporary = tempfile.mkstemp(
		prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent
	)
	temp_path = Path(temporary)
	try:
		with os.fdopen(fd, 'w', encoding='utf-8') as handle:
			handle.write(
				json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + '\n'
			)
			handle.flush()
			os.fsync(handle.fileno())
		# link() provides atomic NOREPLACE publication, unlike rename/replace.
		os.link(temp_path, path, follow_symlinks=False)
	finally:
		temp_path.unlink(missing_ok=True)
	_verify_receipt(plan, expected)


def inspect_coordinated_volve_hmm_checkpoint(plan: CoordinatedVolveHmmRun) -> bool:
	"""Read-only, CPU-only strict final/partial inspection under the caller's lock."""
	root = plan.output_root
	_no_alias(root)
	if {path.name for path in root.iterdir()} != RUN_FILES:
		raise ValueError('coordinated Volve HMM has missing or foreign output files')
	paths = [root / name for name in RUN_FILES]
	before = _snapshot(paths)
	inputs, manifests = _input_snapshot(plan)
	_verify_receipt(plan, _receipt(plan, (inputs, manifests)))
	payloads = {
		name: _mapping(
			torch.load(root / name, map_location='cpu', mmap=True, weights_only=False)
		)
		for name in ('latest.pt', 'best.pt')
	}
	latest, best = payloads['latest.pt'], payloads['best.pt']
	complete = _validate_payload(latest, plan, inputs)
	_validate_payload(best, plan, inputs)
	if (
		best['global_step'] > latest['global_step']
		or _mapping(best['metrics'])['loss'] > _mapping(latest['metrics'])['loss']
		or {
			key: value
			for key, value in _mapping(best['control_identity']).items()
			if key != 'runtime_identity'
		}
		!= {
			key: value
			for key, value in _mapping(latest['control_identity']).items()
			if key != 'runtime_identity'
		}
		or (
			best['global_step'] == latest['global_step']
			and before[str(root / 'best.pt')] != before[str(root / 'latest.pt')]
		)
	):
		raise ValueError('coordinated Volve HMM best/latest evidence disagrees')
	if (
		json.loads((root / 'resolved_config.json').read_text(encoding='utf-8'))
		!= plan.config
	):
		raise ValueError('coordinated Volve HMM resolved_config.json differs')
	metadata = json.loads((root / 'run_metadata.json').read_text(encoding='utf-8'))
	if metadata.get('control_identity') != latest['control_identity']:
		raise ValueError('coordinated Volve HMM run metadata control differs')
	if _snapshot(paths) != before or _input_snapshot(plan) != (inputs, manifests):
		raise ValueError('coordinated Volve HMM evidence changed during inspection')
	_verify_receipt(plan, _receipt(plan, (inputs, manifests)))
	return complete


def _run_training(config: Mapping[str, object], resume: Path | None) -> Path:
	from seis_ssl_cluster.training.strat_hmm import (  # noqa: PLC0415
		run_strat_hmm_pretext_training,
	)

	return run_strat_hmm_pretext_training(config, resume=resume)


def run_coordinated_volve_hmm_if_scoped(  # noqa: C901 - Explicit fail-closed dispatch.
	config: Mapping[str, object],
	*,
	config_path: Path,
	resume: Path | None = None,
	quarantine_invalid: bool = False,
) -> Path | None:
	"""Lock before GPU setup, then recheck fresh inputs and skip/resume/run safely."""
	plan = classify_coordinated_volve_hmm_run(config, config_path=config_path)
	if plan is None:
		return None
	if quarantine_invalid:
		raise ValueError('coordinated Volve HMM never quarantines or overwrites')
	checkpoint = plan.output_root / 'latest.pt'
	if resume is not None:
		_regular(resume.absolute())
		if resume.absolute() != checkpoint:
			raise ValueError('coordinated Volve HMM resume must be canonical latest.pt')
	with _file_lock(plan.lock_path):
		fresh = classify_coordinated_volve_hmm_run(config, config_path=config_path)
		if fresh != plan:
			raise ValueError('coordinated Volve HMM recipe changed while waiting')
		inputs = _input_snapshot(plan)
		_no_alias(plan.output_root)
		if plan.output_root.exists() and any(plan.output_root.iterdir()):
			if inspect_coordinated_volve_hmm_checkpoint(plan):
				print(
					'coordinated Volve HMM: strict-complete; no training or output '
					f'writes: {checkpoint}'
				)
				return checkpoint
			resume = checkpoint
		elif resume is not None:
			raise ValueError('coordinated Volve HMM resume disappeared while waiting')
		else:
			_create_receipt(plan, _receipt(plan, inputs))
		result = _run_training(plan.config, resume)
		if _input_snapshot(plan) != inputs:
			raise ValueError('coordinated Volve HMM inputs changed during training')
		if result != checkpoint or not inspect_coordinated_volve_hmm_checkpoint(plan):
			raise ValueError(
				'coordinated Volve HMM runner did not finish the exact budget'
			)
		return checkpoint
