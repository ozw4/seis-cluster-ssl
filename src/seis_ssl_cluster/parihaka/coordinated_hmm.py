"""Cooperate on the three remaining experiment-40 HMM runs without GPU waiting."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.stratigraphy import discover_pseudo_target_inputs
from seis_ssl_cluster.training.strat_hmm.resume import _validate_strat_resume_payload

try:
	import fcntl
except ImportError:  # Non-POSIX, unrelated CLI jobs retain their existing behavior.
	fcntl = None

EXPERIMENT = (
	Path(__file__).resolve().parents[3]
	/ 'experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1'
)
RECIPES = ('local_bt3/distill020', 'random/distill010', 'random/distill020')
ARTIFACT_STEM = Path(
	'pretraining/parihaka/facies_benchmark_v1/pretraining_comparison_completion_v1'
)
RUN_FILES = frozenset(
	{'latest.pt', 'best.pt', 'resolved_config.json', 'run_metadata.json'}
)
_LOCK_FDS: set[int] = set()


def _close_inherited_locks() -> None:
	"""DataLoader fork children must not retain their parent's lock descriptors."""
	for descriptor in _LOCK_FDS:
		os.close(descriptor)
	_LOCK_FDS.clear()


if hasattr(os, 'register_at_fork'):
	os.register_at_fork(after_in_child=_close_inherited_locks)


@dataclass(frozen=True)
class CoordinatedHmmRun:
	"""An exact, validated experiment recipe and its isolated execution kind."""

	recipe: str
	config_path: Path
	config: dict[str, object]
	output_root: Path
	lock_root: Path
	smoke: bool


def _mapping(value: object) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('coordinated HMM evidence must be a mapping')
	return value


def _no_symlinks(path: Path) -> None:
	if not path.is_absolute():
		raise ValueError(f'coordinated HMM path must be absolute: {path}')
	for part in (path, *path.parents):
		if part.is_symlink():
			raise ValueError(f'coordinated HMM refuses symlink: {part}')


def _regular_file(path: Path) -> None:
	_no_symlinks(path)
	if not path.is_file():
		raise ValueError(f'coordinated HMM requires a regular file: {path}')


def _json_hash(value: object) -> str:
	return hashlib.sha256(
		json.dumps(
			value, sort_keys=True, separators=(',', ':'), allow_nan=False
		).encode()
	).hexdigest()


def _file_identity(path: Path) -> dict[str, str]:
	_regular_file(path)
	return {'path': str(path), 'sha256': file_sha256(path)}


def _expected_inputs(config: Mapping[str, object]) -> dict[str, object]:
	pseudo = _mapping(config['pseudo_targets'])
	targets = discover_pseudo_target_inputs(Path(str(pseudo['input_dir'])), k=6)
	if len(targets) != 1 or targets[0].survey_id != 'parihaka':
		raise ValueError('coordinated HMM requires exactly the Parihaka K6 target set')
	item = targets[0]
	entry: dict[str, object] = {
		'survey_id': item.survey_id,
		'boundary_weight_present': item.boundary_weight_path is not None,
	}
	for name, path in (
		('labels', item.labels_path),
		('confidence', item.confidence_path),
		('valid_tokens', item.valid_tokens_path),
		('metadata', item.metadata_path),
	):
		entry[name] = _file_identity(path)
	if item.boundary_weight_path is not None:
		entry['boundary_weight'] = _file_identity(item.boundary_weight_path)
	return {
		'teacher_checkpoint': _file_identity(
			Path(str(_mapping(config['teacher'])['checkpoint']))
		),
		'student_init_checkpoint': _file_identity(
			Path(str(_mapping(config['student'])['init_checkpoint']))
		),
		'pseudo_targets': [entry],
	}


def classify_coordinated_hmm_run(
	config: Mapping[str, object], *, config_path: Path
) -> CoordinatedHmmRun | None:
	"""Return an exact allowed run, reject scope escapes, or leave other jobs alone."""
	paths = _mapping(config.get('paths', {}))
	output = Path(str(paths.get('output_root', '')))
	definitions = {
		recipe: EXPERIMENT / f'20_stage2/{recipe}/01_full_25ep.yaml'
		for recipe in RECIPES
	}
	definition_recipe = next(
		(
			recipe
			for recipe, path in definitions.items()
			if path.resolve() == config_path.resolve()
		),
		None,
	)
	output_recipe = next(
		(
			recipe
			for recipe in RECIPES
			for kind in ('full_25ep', 'smoke_1step')
			if output.resolve().parts[-7:]
			== (*ARTIFACT_STEM.parts, *Path(recipe).parts, kind)
		),
		None,
	)
	if definition_recipe is None and output_recipe is None:
		return None
	if definition_recipe is None or definition_recipe != output_recipe:
		raise ValueError('canonical Parihaka output requires its exact recipe config')
	_regular_file(config_path.absolute())
	_no_symlinks(output)
	expected = resolve_strat_hmm_pretext_config(
		load_config(definitions[definition_recipe])
	)
	full_root = Path(str(_mapping(expected['paths'])['output_root']))
	smoke_root = full_root.parent / 'smoke_1step'
	smoke = output == smoke_root
	if output not in (full_root, smoke_root):
		raise ValueError('coordinated HMM output is outside the canonical recipe')
	if smoke:
		expected = copy.deepcopy(expected)
		expected['paths']['output_root'] = str(smoke_root)
		expected['train']['max_steps'] = 1
	if config != expected:
		raise ValueError(
			'coordinated HMM config or CLI override differs from the recipe'
		)
	train = _mapping(expected['train'])
	if (
		train['epochs'] != 25
		or train['samples_per_epoch'] != 10000
		or train['batch_size'] != 16
		or train['amp'] is not False
		or train['seed'] != 42
		or train['allow_overwrite_output'] is not False
		or train['max_steps'] != (1 if smoke else None)
	):
		raise ValueError('coordinated HMM requires the immutable full/smoke budget')
	lock_root = full_root.parents[2] / '.remaining_hmm_locks'
	_no_symlinks(lock_root)
	return CoordinatedHmmRun(
		definition_recipe,
		definitions[definition_recipe],
		expected,
		output,
		lock_root,
		smoke,
	)


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
	if fcntl is None:
		raise RuntimeError(
			'coordinated Parihaka HMM execution requires POSIX file locks'
		)
	_no_symlinks(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	_no_symlinks(path.parent)
	descriptor = os.open(
		path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600
	)
	try:
		verify_open_lock(path, descriptor)
		_LOCK_FDS.add(descriptor)
		fcntl.flock(descriptor, fcntl.LOCK_EX)
		verify_open_lock(path, descriptor)
		yield
	finally:
		_LOCK_FDS.discard(descriptor)
		fcntl.flock(descriptor, fcntl.LOCK_UN)
		os.close(descriptor)


def verify_open_lock(path: Path, descriptor: int) -> None:
	"""Require an opened, single-link regular lock to still name the same inode."""
	_no_symlinks(path)
	opened = os.fstat(descriptor)
	current = path.stat(follow_symlinks=False)
	if (
		not stat.S_ISREG(opened.st_mode)
		or opened.st_nlink != 1
		or not stat.S_ISREG(current.st_mode)
		or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
	):
		raise ValueError(
			'lock must be the same regular-file inode without hardlink aliases'
		)


def _validate_rng(payload: Mapping[str, object]) -> None:
	rng = _mapping(payload['rng_state'])
	random.Random().setstate(rng['python'])  # noqa: S311 - validate saved training RNG only.
	np.random.RandomState().set_state(rng['numpy'])
	for name in ('torch', 'dataloader_generator'):
		value = rng[name]
		if (
			not isinstance(value, torch.Tensor)
			or value.dtype != torch.uint8
			or value.ndim != 1
		):
			raise ValueError(f'invalid CPU RNG state: {name}')
		torch.Generator(device='cpu').set_state(value)
	cuda = rng.get('torch_cuda')
	if (
		not isinstance(cuda, list)
		or len(cuda) != 1
		or any(
			not isinstance(value, torch.Tensor)
			or value.dtype != torch.uint8
			or value.ndim != 1
			or value.numel() != 16
			for value in cuda
		)
	):
		raise ValueError('invalid stored CUDA RNG state')


def _expected_optimizer_shapes(
	payload: Mapping[str, object], config: Mapping[str, object]
) -> list[tuple[int, ...]]:
	parent_path = Path(str(_mapping(config['student'])['init_checkpoint']))
	_regular_file(parent_path)
	parent = _mapping(torch.load(parent_path, map_location='cpu', weights_only=False))
	model = _mapping(payload['model_state_dict'])
	parent_model = _mapping(parent['model_state_dict'])
	if not model or set(model) != set(parent_model):
		raise ValueError('student state does not match the parent architecture')
	for name, value in model.items():
		original = parent_model[name]
		if (
			not isinstance(value, torch.Tensor)
			or not isinstance(original, torch.Tensor)
			or value.shape != original.shape
			or value.dtype != original.dtype
			or not torch.isfinite(value).all()
		):
			raise ValueError('invalid student tensor state')
	head = _mapping(payload['stratigraphy_state_dict'])
	head_config = _mapping(config['head'])
	projection = int(head_config['projection_dim'])
	expected_head = {
		'prototypes': (int(head_config['num_prototypes']), projection),
		'projection.weight': (
			projection,
			int(_mapping(config['model'])['encoder_dim']),
		),
		'projection.bias': (projection,),
	}
	if set(head) != set(expected_head):
		raise ValueError('prototype head state has unexpected keys')
	for name, shape in expected_head.items():
		value = head[name]
		if (
			not isinstance(value, torch.Tensor)
			or tuple(value.shape) != shape
			or not torch.isfinite(value).all()
		):
			raise ValueError('invalid prototype head tensor state')
	block = int(_mapping(config['model'])['encoder_depth']) - 1
	encoder_shapes = [
		tuple(value.shape)
		for name, value in model.items()
		if name.startswith(f'encoder.layers.{block}.')
	]
	if not encoder_shapes:
		raise ValueError('missing trainable top encoder block')
	return [*expected_head.values(), *encoder_shapes]


def _validate_optimizer(
	payload: Mapping[str, object], config: Mapping[str, object]
) -> None:
	expected_shapes = _expected_optimizer_shapes(payload, config)
	optimizer = _mapping(payload['optimizer_state_dict'])
	groups = optimizer.get('param_groups')
	states = _mapping(optimizer.get('state'))
	if not isinstance(groups, list) or len(groups) != 2 or not states:
		raise ValueError('missing optimizer state or parameter groups')
	train = _mapping(config['train'])
	for group, name, lr, count in zip(
		groups,
		('head', 'encoder'),
		(train['lr'], train['encoder_lr']),
		(3, len(expected_shapes) - 3),
		strict=True,
	):
		if (
			_mapping(group).get('name') != name
			or group.get('lr') != lr
			or group.get('weight_decay') != train['weight_decay']
			or group.get('betas') != (0.9, 0.999)
			or group.get('eps') != 1e-8
			or group.get('amsgrad') is not False
			or group.get('maximize') is not False
			or group.get('capturable') is not False
			or group.get('differentiable') is not False
			or group.get('foreach') is not None
			or group.get('fused') is not None
			or len(group.get('params', [])) != count
		):
			raise ValueError('optimizer groups differ from the scientific recipe')
	parameters: list[int] = []
	for group in groups:
		indices = _mapping(group).get('params')
		if (
			not isinstance(indices, list)
			or not indices
			or any(type(index) is not int for index in indices)
		):
			raise ValueError('invalid optimizer parameter group')
		parameters.extend(indices)
	if len(set(parameters)) != len(parameters) or set(states) != set(parameters):
		raise ValueError('optimizer parameter identities are inconsistent')
	for index, shape in zip(parameters, expected_shapes, strict=True):
		_validate_optimizer_moment(
			_mapping(states[index]), shape, payload['global_step']
		)


def _validate_optimizer_moment(
	state: Mapping[str, object], shape: tuple[int, ...], global_step: object
) -> None:
	step = state.get('step')
	if isinstance(step, torch.Tensor):
		if step.numel() != 1:
			raise ValueError('invalid optimizer step tensor')
		step = step.item()
	if (
		not isinstance(step, int | float)
		or not math.isfinite(step)
		or step != global_step
	):
		raise ValueError('optimizer step differs from checkpoint budget')
	mean, variance = state.get('exp_avg'), state.get('exp_avg_sq')
	if (
		not isinstance(mean, torch.Tensor)
		or not isinstance(variance, torch.Tensor)
		or mean.shape != variance.shape
		or tuple(mean.shape) != shape
		or not torch.isfinite(mean).all()
		or not torch.isfinite(variance).all()
	):
		raise ValueError('invalid optimizer moment tensors')


def inspect_coordinated_hmm_checkpoint(plan: CoordinatedHmmRun) -> bool:
	"""Validate exact provenance and resumability; return whether the budget is done."""
	root = plan.output_root
	_no_symlinks(root)
	if {path.name for path in root.iterdir()} != RUN_FILES:
		raise ValueError('coordinated HMM output has missing or foreign files')
	for name in RUN_FILES:
		_regular_file(root / name)
	checkpoint = root / 'latest.pt'
	before = file_sha256(checkpoint)
	payload = _mapping(torch.load(checkpoint, map_location='cpu', weights_only=False))
	_validate_strat_resume_payload(payload, amp_enabled=False)
	_validate_rng(payload)
	_validate_optimizer(payload, plan.config)
	if payload['stratigraphy_config'] != plan.config:
		raise ValueError('checkpoint resolved config differs from expected full config')
	saved = json.loads((root / 'resolved_config.json').read_text(encoding='utf-8'))
	if saved != plan.config:
		raise ValueError('saved resolved config differs from expected full config')
	control = _mapping(payload.get('control_identity'))
	identity = _mapping(plan.config['identity'])
	if (
		control.get('schema_version') != 1
		or _mapping(payload['training_state']).get('schema_version') != 1
		or control.get('model_tag') != identity['model_tag']
		or control.get('scientific_identity') != identity['scientific_identity']
		or control.get('resolved_training_config_sha256') != _json_hash(plan.config)
		or control.get('input_identities') != _expected_inputs(plan.config)
	):
		raise ValueError(
			'checkpoint scientific identity, config SHA, or live inputs differ'
		)
	metadata = json.loads((root / 'run_metadata.json').read_text(encoding='utf-8'))
	if metadata.get('control_identity') != control:
		raise ValueError('run metadata control identity differs from checkpoint')
	complete = _validate_checkpoint_counters(payload, smoke=plan.smoke)
	if file_sha256(checkpoint) != before:
		raise ValueError('checkpoint SHA changed during source audit')
	return complete


def _validate_checkpoint_counters(
	payload: Mapping[str, object], *, smoke: bool
) -> bool:
	state = _mapping(payload['training_state'])
	epoch, step = payload['epoch'], payload['global_step']
	kind, batch = state['checkpoint_kind'], state['batch_index']
	if not (1 <= epoch <= 25 and 1 <= step <= 15625):
		raise ValueError('checkpoint counters exceed the exact HMM budget')
	if kind == 'epoch':
		valid = batch is None and step == epoch * 625
	elif kind == 'step':
		valid = (
			type(batch) is int
			and 0 <= batch < 625
			and step == (epoch - 1) * 625 + batch + 1
		)
	else:
		valid = False
	if not valid:
		raise ValueError('checkpoint epoch/step/batch boundary is inconsistent')
	if smoke:
		complete = (epoch, step, kind, batch) == (1, 1, 'step', 0)
		if not complete:
			raise ValueError('smoke checkpoint must prove exactly one optimizer step')
	else:
		complete = (epoch, step, kind, batch) == (25, 15625, 'epoch', None)
		if step == 15625 and not complete:
			raise ValueError('full HMM must end at the completed epoch boundary')
	return complete


def _run_training(config: Mapping[str, object], resume: Path | None) -> Path:
	from seis_ssl_cluster.training.strat_hmm import (  # noqa: PLC0415
		run_strat_hmm_pretext_training,
	)

	return run_strat_hmm_pretext_training(config, resume=resume)


def run_coordinated_hmm_if_scoped(
	config: Mapping[str, object],
	*,
	config_path: Path,
	resume: Path | None = None,
	quarantine_invalid: bool = False,
) -> Path | None:
	"""Serialize only the exact remaining recipes, returning None for other CLI jobs."""
	plan = classify_coordinated_hmm_run(config, config_path=config_path)
	if plan is None:
		return None
	if quarantine_invalid:
		raise ValueError('coordinated HMM never quarantines or overwrites artifacts')
	checkpoint = plan.output_root / 'latest.pt'
	if resume is not None:
		_regular_file(resume.absolute())
		if resume.absolute() != checkpoint:
			raise ValueError(
				'coordinated resume must be the canonical latest checkpoint'
			)
	with (
		_file_lock(plan.lock_root / 'lane.lock'),
		_file_lock(plan.lock_root / f'{plan.recipe.replace("/", "_")}.lock'),
	):
		# Do not trust config/input or checkpoint observations made before waiting.
		fresh = classify_coordinated_hmm_run(config, config_path=config_path)
		if fresh != plan:
			raise ValueError('coordinated HMM recipe changed while waiting')
		_no_symlinks(plan.output_root)
		if plan.output_root.exists() and any(plan.output_root.iterdir()):
			if inspect_coordinated_hmm_checkpoint(plan):
				print(
					'coordinated HMM: strict-complete; no training or output writes: '
					f'{checkpoint}'
				)
				return checkpoint
			resume = checkpoint
		else:
			if resume is not None:
				raise ValueError('resume checkpoint disappeared while waiting')
			_expected_inputs(plan.config)
		result = _run_training(plan.config, resume)
		if result != checkpoint or not inspect_coordinated_hmm_checkpoint(plan):
			raise ValueError('coordinated HMM runner did not complete the exact budget')
		return checkpoint
