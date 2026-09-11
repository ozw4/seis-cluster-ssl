"""CPU-only scope, source-evidence, and process-lock contracts for experiment 40."""

# ruff: noqa: SLF001 - These focused contracts intentionally inspect lock internals.

from __future__ import annotations

import copy
import json
import multiprocessing
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.parihaka import coordinated_hmm as coordination
from seis_ssl_cluster.stratigraphy import write_pseudo_target


def _config(tmp_path, monkeypatch, recipe='random/distill020', *, smoke=False):
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	path = coordination.EXPERIMENT / f'20_stage2/{recipe}/01_full_25ep.yaml'
	raw = load_config(path)
	parent = Path(raw['teacher']['checkpoint'])
	if not parent.exists():
		parent.parent.mkdir(parents=True, exist_ok=True)
		torch.save(
			{
				'model_state_dict': {
					'encoder.layers.7.weight': torch.ones(1),
					'frozen': torch.ones(1),
				}
			},
			parent,
		)
	target_root = Path(raw['pseudo_targets']['input_dir'])
	if not target_root.exists():
		write_pseudo_target(
			target_root,
			k=6,
			survey_id='parihaka',
			labels=np.zeros((1, 1, 1), dtype=np.int32),
			confidence=np.ones((1, 1, 1), dtype=np.float32),
			valid_tokens=np.ones((1, 1, 1), dtype=np.bool_),
		)
	config = resolve_strat_hmm_pretext_config(raw)
	if smoke:
		config['paths']['output_root'] = str(
			Path(config['paths']['output_root']).parent / 'smoke_1step'
		)
		config['train']['max_steps'] = 1
	return config, path


def _plan(config, path):
	plan = coordination.classify_coordinated_hmm_run(config, config_path=path)
	assert plan is not None
	return plan


def _write_checkpoint(config, *, epoch=None, step=None):
	smoke = config['train']['max_steps'] == 1
	epoch = (1 if smoke else 25) if epoch is None else epoch
	step = (1 if smoke else epoch * 625) if step is None else step
	root = Path(config['paths']['output_root'])
	root.mkdir(parents=True, exist_ok=True)
	parent = torch.load(
		config['student']['init_checkpoint'], map_location='cpu', weights_only=False
	)
	head = {
		'prototypes': torch.zeros(6, 128),
		'projection.weight': torch.zeros(128, 384),
		'projection.bias': torch.zeros(128),
	}
	shapes = [*head.values(), parent['model_state_dict']['encoder.layers.7.weight']]
	state = {
		index: {
			'step': torch.tensor(float(step)),
			'exp_avg': torch.zeros_like(value),
			'exp_avg_sq': torch.zeros_like(value),
		}
		for index, value in enumerate(shapes)
	}
	groups = []
	for name, indices in [('head', [0, 1, 2]), ('encoder', [3])]:
		groups.append(
			{
				'name': name,
				'params': indices,
				'lr': 1e-5,
				'weight_decay': 0.05,
				'betas': (0.9, 0.999),
				'eps': 1e-8,
				'amsgrad': False,
				'maximize': False,
				'capturable': False,
				'differentiable': False,
				'foreach': None,
				'fused': None,
			}
		)
	control = {
		'schema_version': 1,
		'model_tag': config['identity']['model_tag'],
		'scientific_identity': config['identity']['scientific_identity'],
		'resolved_training_config_sha256': coordination._json_hash(config),
		'input_identities': coordination._expected_inputs(config),
	}
	payload = {
		'config': {},
		'stratigraphy_config': config,
		'model_state_dict': parent['model_state_dict'],
		'stratigraphy_state_dict': head,
		'optimizer_state_dict': {'state': state, 'param_groups': groups},
		'epoch': epoch,
		'global_step': step,
		'amp_enabled': False,
		'scaler_state_dict': None,
		'package_version': 'test',
		'metrics': {'loss': 1.0},
		'control_identity': control,
		'training_state': {
			'schema_version': 1,
			'stage': 'train_strat_hmm_pretext',
			'checkpoint_kind': 'step' if smoke else 'epoch',
			'batch_index': 0 if smoke else None,
		},
		'rng_state': {
			'python': random.getstate(),
			'numpy': np.random.get_state(),  # noqa: NPY002 - Existing checkpoint schema.
			'torch': torch.get_rng_state(),
			'dataloader_generator': torch.Generator().get_state(),
			'torch_cuda': [torch.zeros(16, dtype=torch.uint8)],
		},
	}
	for name in ('latest.pt', 'best.pt'):
		torch.save(payload, root / name)
	(root / 'resolved_config.json').write_text(json.dumps(config))
	(root / 'run_metadata.json').write_text(json.dumps({'control_identity': control}))
	return root / 'latest.pt'


@pytest.mark.parametrize('recipe', coordination.RECIPES)
@pytest.mark.parametrize('smoke', [False, True])
def test_exact_six_recipes_classify_without_writing_outputs(
	tmp_path, monkeypatch, recipe, smoke
):
	config, path = _config(tmp_path, monkeypatch, recipe, smoke=smoke)
	plan = _plan(config, path)
	assert plan.recipe == recipe
	assert plan.smoke is smoke
	assert not plan.output_root.exists()
	assert not plan.lock_root.exists()


@pytest.mark.parametrize(
	'change',
	['loss', 'budget', 'device', 'output', 'foreign_config', 'dotdot', 'alias'],
)
def test_rejects_canonical_scope_escapes(tmp_path, monkeypatch, change):
	config, path = _config(tmp_path, monkeypatch)
	if change == 'loss':
		config['loss']['distillation_weight'] = 0.1
	elif change == 'budget':
		config['train']['max_steps'] = 1
	elif change == 'device':
		config['train']['device'] = 'cpu'
	elif change == 'output':
		config['paths']['output_root'] = str(tmp_path / 'other')
	elif change == 'foreign_config':
		path = tmp_path / 'foreign.yaml'
	elif change == 'dotdot':
		output = Path(config['paths']['output_root'])
		config['paths']['output_root'] = str(
			output.parent / '..' / output.parent.name / output.name
		)
		path = tmp_path / 'foreign.yaml'
	else:
		alias = tmp_path / 'alias'
		alias.symlink_to(Path(config['paths']['output_root']), target_is_directory=True)
		config['paths']['output_root'] = str(alias)
		path = tmp_path / 'foreign.yaml'
	with pytest.raises(ValueError, match=r'coordinated|canonical'):
		coordination.run_coordinated_hmm_if_scoped(config, config_path=path)


def test_non_target_compatibility_without_posix_locks(tmp_path, monkeypatch):
	monkeypatch.setattr(coordination, 'fcntl', None)
	config = {'paths': {'output_root': str(tmp_path / 'unrelated')}}
	assert (
		coordination.run_coordinated_hmm_if_scoped(
			config, config_path=tmp_path / 'other.yaml', quarantine_invalid=True
		)
		is None
	)
	assert not (tmp_path / 'unrelated').exists()


@pytest.mark.parametrize('smoke', [False, True])
def test_complete_skip_is_cpu_only_and_preserves_every_artifact(
	tmp_path, monkeypatch, smoke
):
	config, path = _config(tmp_path, monkeypatch, smoke=smoke)
	checkpoint = _write_checkpoint(config)
	before = {
		item.name: (coordination.file_sha256(item), item.stat().st_mtime_ns)
		for item in checkpoint.parent.iterdir()
	}

	def forbidden(*_args, **_kwargs):
		raise AssertionError(
			'completed checkpoint must not initialize GPU or run training'
		)

	monkeypatch.setattr(coordination, '_run_training', forbidden)
	monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
	assert (
		coordination.run_coordinated_hmm_if_scoped(config, config_path=path)
		== checkpoint
	)
	assert {
		item.name: (coordination.file_sha256(item), item.stat().st_mtime_ns)
		for item in checkpoint.parent.iterdir()
	} == before
	lock = _plan(config, path).lock_root / 'lane.lock'
	inode = lock.stat().st_ino
	assert (
		coordination.run_coordinated_hmm_if_scoped(
			config, config_path=path, resume=checkpoint
		)
		== checkpoint
	)
	assert lock.stat().st_ino == inode


def test_partial_uses_latest_after_lock_and_preserves_config(tmp_path, monkeypatch):
	config, path = _config(tmp_path, monkeypatch)
	checkpoint = _write_checkpoint(config, epoch=2)
	calls = []

	def train(actual_config, resume):
		calls.append(resume)
		assert actual_config == config
		return _write_checkpoint(actual_config)

	monkeypatch.setattr(coordination, '_run_training', train)
	assert (
		coordination.run_coordinated_hmm_if_scoped(config, config_path=path)
		== checkpoint
	)
	assert calls == [checkpoint]


@pytest.mark.parametrize(
	'change',
	[
		'config',
		'saved_config',
		'config_sha',
		'parent',
		'target_set',
		'target_live',
		'stage',
		'batch',
		'counter',
		'cuda_missing',
		'cuda_empty',
		'cuda_shape',
		'optimizer_missing',
		'optimizer_subset',
		'optimizer_shape',
		'optimizer_lr',
		'model_shape',
		'head_shape',
	],
)
def test_invalid_checkpoint_fails_before_runner_without_writes(  # noqa: C901, PLR0912
	tmp_path, monkeypatch, change
):
	config, path = _config(tmp_path, monkeypatch)
	checkpoint = _write_checkpoint(config, epoch=2)
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	if change == 'config':
		payload['stratigraphy_config'] = copy.deepcopy(config)
		payload['stratigraphy_config']['train']['num_workers'] = 3
	elif change == 'saved_config':
		(checkpoint.parent / 'resolved_config.json').write_text('{}')
	elif change == 'config_sha':
		payload['control_identity']['resolved_training_config_sha256'] = '0' * 64
	elif change == 'parent':
		payload['control_identity']['input_identities']['teacher_checkpoint'][
			'sha256'
		] = '0' * 64
	elif change == 'target_set':
		payload['control_identity']['input_identities']['pseudo_targets'] = []
	elif change == 'target_live':
		Path(
			payload['control_identity']['input_identities']['pseudo_targets'][0][
				'labels'
			]['path']
		).write_bytes(b'changed target')
	elif change == 'stage':
		payload['training_state']['stage'] = 'train_amp_mae'
	elif change == 'batch':
		payload['training_state']['batch_index'] = 1
	elif change == 'counter':
		payload['epoch'] = 3
	elif change == 'cuda_missing':
		del payload['rng_state']['torch_cuda']
	elif change == 'cuda_empty':
		payload['rng_state']['torch_cuda'] = []
	elif change == 'cuda_shape':
		payload['rng_state']['torch_cuda'] = [torch.zeros(1, dtype=torch.uint8)]
	elif change == 'optimizer_missing':
		payload['optimizer_state_dict'] = {}
	elif change == 'optimizer_subset':
		del payload['optimizer_state_dict']['state'][3]
	elif change == 'optimizer_shape':
		payload['optimizer_state_dict']['state'][3]['exp_avg'] = torch.zeros(2)
	elif change == 'optimizer_lr':
		payload['optimizer_state_dict']['param_groups'][0]['lr'] = 1.0
	elif change == 'model_shape':
		payload['model_state_dict']['encoder.layers.7.weight'] = torch.zeros(2)
	else:
		payload['stratigraphy_state_dict']['prototypes'] = torch.zeros(6, 127)
	torch.save(payload, checkpoint)
	before = {
		item.name: coordination.file_sha256(item)
		for item in checkpoint.parent.iterdir()
	}

	def forbidden(*_args):
		raise AssertionError('invalid checkpoint must never reach the runner')

	monkeypatch.setattr(coordination, '_run_training', forbidden)
	with pytest.raises((ValueError, TypeError, KeyError)):
		coordination.run_coordinated_hmm_if_scoped(config, config_path=path)
	assert {
		item.name: coordination.file_sha256(item)
		for item in checkpoint.parent.iterdir()
	} == before


@pytest.mark.parametrize(
	'change',
	['foreign_file', 'metadata_only', 'quarantine', 'foreign_resume', 'symlink_lock'],
)
def test_refuses_unowned_output_and_lock_paths(tmp_path, monkeypatch, change):
	config, path = _config(tmp_path, monkeypatch)
	plan = _plan(config, path)
	kwargs = {}
	if change == 'foreign_file':
		_write_checkpoint(config)
		(plan.output_root / 'foreign').touch()
	elif change == 'metadata_only':
		plan.output_root.mkdir(parents=True)
		(plan.output_root / 'resolved_config.json').write_text('{}')
	elif change == 'quarantine':
		kwargs['quarantine_invalid'] = True
	elif change == 'foreign_resume':
		foreign = tmp_path / 'latest.pt'
		foreign.touch()
		kwargs['resume'] = foreign
	else:
		plan.lock_root.mkdir(parents=True)
		sentinel = tmp_path / 'sentinel'
		sentinel.write_text('untouched')
		(plan.lock_root / 'lane.lock').symlink_to(sentinel)
	with pytest.raises(ValueError, match='coordinated'):
		coordination.run_coordinated_hmm_if_scoped(config, config_path=path, **kwargs)
	if change == 'symlink_lock':
		assert sentinel.read_text() == 'untouched'


def _worker(config, path, queue, barrier):
	torch.set_num_threads(2)

	def fake_training(actual, _resume):
		queue.put(('start', time.monotonic(), str(actual['paths']['output_root'])))
		time.sleep(0.2)
		checkpoint = _write_checkpoint(actual)
		queue.put(('finish', time.monotonic(), str(checkpoint.parent)))
		return checkpoint

	coordination._run_training = fake_training
	barrier.wait(timeout=20)
	coordination.run_coordinated_hmm_if_scoped(config, config_path=Path(path))
	queue.put(('done', time.monotonic(), ''))


@pytest.mark.parametrize('pair', ['same_full', 'full_smoke', 'different_arm'])
def test_cpu_process_contention_serializes_full_smoke_and_different_arms(
	tmp_path, monkeypatch, pair
):
	first, first_path = _config(tmp_path, monkeypatch)
	second, second_path = _config(
		tmp_path,
		monkeypatch,
		'random/distill010' if pair == 'different_arm' else 'random/distill020',
		smoke=pair == 'full_smoke',
	)
	context = multiprocessing.get_context('spawn')
	queue = context.Queue()
	barrier = context.Barrier(2)
	processes = [
		context.Process(target=_worker, args=(config, str(path), queue, barrier))
		for config, path in ((first, first_path), (second, second_path))
	]
	for process in processes:
		process.start()
	for process in processes:
		process.join(timeout=45)
		assert process.exitcode == 0
	events = []
	while sum(event[0] == 'done' for event in events) < 2:
		events.append(queue.get(timeout=5))
	runs = sorted(
		(event for event in events if event[0] != 'done'), key=lambda event: event[1]
	)
	assert [event[0] for event in runs] == (
		['start', 'finish']
		if pair == 'same_full'
		else ['start', 'finish', 'start', 'finish']
	)


def _fork_probe(descriptor, identity, connection):
	try:
		opened = os.fstat(descriptor)
	except OSError:
		connection.send(True)  # noqa: FBT003 - IPC result, not a mode argument.
	else:
		# multiprocessing may reuse the closed descriptor for /dev/null in bootstrap.
		connection.send((opened.st_dev, opened.st_ino) != identity)


def test_fork_workers_close_inherited_lock_descriptor(tmp_path):
	context = multiprocessing.get_context('fork')
	parent, child = context.Pipe(duplex=False)
	with coordination._file_lock(tmp_path / 'lock'):
		descriptor = next(iter(coordination._LOCK_FDS))
		opened = os.fstat(descriptor)
		process = context.Process(
			target=_fork_probe, args=(descriptor, (opened.st_dev, opened.st_ino), child)
		)
		process.start()
		process.join(timeout=10)
		assert process.exitcode == 0
		assert parent.recv() is True
		assert os.fstat(descriptor).st_ino == (tmp_path / 'lock').stat().st_ino


@pytest.mark.parametrize('change', ['none', 'hardlink', 'replacement', 'symlink'])
def test_open_lock_verifier_checks_inode_and_aliases(tmp_path, change):
	path = tmp_path / 'lock'
	descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
	try:
		if change == 'hardlink':
			os.link(path, tmp_path / 'alias')
		elif change == 'replacement':
			path.rename(tmp_path / 'old')
			path.touch()
		elif change == 'symlink':
			path.rename(tmp_path / 'old')
			path.symlink_to(tmp_path / 'old')
		if change == 'none':
			coordination.verify_open_lock(path, descriptor)
		else:
			with pytest.raises(ValueError, match=r'lock|symlink'):
				coordination.verify_open_lock(path, descriptor)
	finally:
		os.close(descriptor)


@pytest.mark.parametrize('smoke', [False, True])
def test_cli_dry_run_never_creates_lock_or_output(tmp_path, monkeypatch, smoke):
	config, path = _config(tmp_path, monkeypatch, smoke=smoke)
	plan = _plan(config, path)
	cli = (
		coordination.EXPERIMENT.parents[3]
		/ 'proc/seis_ssl_cluster/train_strat_hmm_pretext.py'
	)
	command = [sys.executable, str(cli), '--config', str(path), '--dry-run']
	if smoke:
		command.extend(['--max-steps', '1', '--output-root', str(plan.output_root)])
	result = subprocess.run(  # noqa: S603 - Repository CLI and tmp-path fixture only.
		command,
		capture_output=True,
		text=True,
		check=True,
	)
	assert 'dry-run; training skipped' in result.stdout
	assert not plan.lock_root.exists()
	assert not plan.output_root.exists()
