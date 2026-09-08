"""CPU-only exact scope, strict checkpoint, and concurrent Volve writer tests."""

# ruff: noqa: SLF001 - Focused contracts deliberately exercise private validators.

from __future__ import annotations

import copy
import json
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.parihaka import coordinated_hmm as pari
from seis_ssl_cluster.stratigraphy import write_pseudo_target
from seis_ssl_cluster.volve import coordinated_hmm as coordination
from seis_ssl_cluster.volve import coordinated_hmm_state as state


def _tiny_template(_config):
	return {
		**{f'encoder.layers.7.parameter_{i}': torch.ones(i + 1) for i in range(12)},
		'frozen': torch.ones(3),
	}


def _tiny_head(_config):
	return {
		'prototypes': torch.zeros(6, 128),
		'projection.weight': torch.zeros(128, 384),
		'projection.bias': torch.zeros(128),
	}


def _config(tmp_path, monkeypatch, suffix='020', *, smoke=False):
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	monkeypatch.setattr(state, '_model_template', _tiny_template)
	monkeypatch.setattr(state, '_initial_head', _tiny_head)
	path = (
		coordination.EXPERIMENT
		/ f'20_pretraining/random_init_hmm_k6_distill{suffix}.yaml'
	)
	raw = load_config(path)
	Path(raw['pseudo_targets']['input_dir']).mkdir(parents=True, exist_ok=True)
	parent = Path(raw['teacher']['checkpoint'])
	new_parent = not parent.exists()
	if new_parent:
		parent.parent.mkdir(parents=True)
		parent.touch()
	config = resolve_strat_hmm_pretext_config(raw)
	if new_parent:
		torch.save(
			{
				'epoch': 0,
				'global_step': 0,
				'metadata': {
					'random_encoder_baseline': True,
					'pretrained_weights_loaded': False,
					'seed': 42,
				},
				'training_state': {
					'stage': 'create_random_mae_checkpoint',
					'checkpoint_kind': 'random_init',
				},
				'config': {
					key: copy.deepcopy(config[key])
					for key in ('paths', 'model', 'data', 'zero_mask')
				},
				'model_state_dict': _tiny_template(None),
			},
			parent,
		)
	target = Path(config['pseudo_targets']['input_dir'])
	if not (target / 'k6').exists():
		write_pseudo_target(
			target,
			k=6,
			survey_id='volve_st10010',
			labels=np.arange(6, dtype=np.int32).reshape(1, 1, 6),
			confidence=np.ones((1, 1, 6), dtype=np.float32),
			valid_tokens=np.ones((1, 1, 6), dtype=np.bool_),
			boundary_weight=np.ones((1, 1, 6), dtype=np.float32),
		)
	for filename in config['manifests'].values():
		manifest = Path(filename)
		manifest.parent.mkdir(parents=True, exist_ok=True)
		manifest.write_text('{}')
	_write_lineage(config)
	if smoke:
		config['paths']['output_root'] = str(
			Path(config['paths']['output_root']).parent / 'smoke_2step'
		)
		config['train']['max_steps'] = 2
	return config, path


def _write_lineage(config):
	definition = coordination.EXPERIMENT / '10_hmm_targets/random_init.yaml'
	declared = load_config(definition)
	clustering = Path(declared['clustering']['output_dir'])
	embeddings = Path(declared['embeddings']['input_dir'])
	embeddings.mkdir(parents=True, exist_ok=True)
	parent = config['teacher']['checkpoint']
	parent_sha = coordination.file_sha256(Path(parent))
	metadata_path = embeddings / 'volve_st10010.embedding_metadata.json'
	metadata_path.write_text(
		json.dumps(
			{
				'survey_id': 'volve_st10010',
				'checkpoint_path': parent,
				'checkpoint_sha256': parent_sha,
				'window_size': [128, 128, 128],
				'overlap': [64, 64, 64],
				'output_dtype': 'float16',
				'min_token_valid_fraction': 1.0,
				'precision': {'amp_enabled': True, 'resolved_dtype': 'bfloat16'},
			}
		)
	)
	metadata_sha = coordination.file_sha256(metadata_path)
	source = {
		'metadata_path': str(metadata_path),
		'metadata_sha256': metadata_sha,
		'embeddings_path': str(embeddings / 'volve_st10010.embeddings.npy'),
		'valid_tokens_path': str(embeddings / 'volve_st10010.valid_tokens.npy'),
	}
	np.save(source['embeddings_path'], np.zeros((1, 1, 6, 384), dtype=np.float16))
	np.save(source['valid_tokens_path'], np.ones((1, 1, 6), dtype=np.bool_))
	model_metadata = clustering / 'models/k6/clustering_metadata.json'
	model_metadata.parent.mkdir(parents=True, exist_ok=True)
	model_metadata.write_text(
		json.dumps(
			{
				'method': 'stratigraphic_hmm_kmeans',
				'k': 6,
				'normalization': 'l2',
				'random_seed': 42,
				**{
					key: declared['clustering'][key]
					for key in ('pca', 'residualization', 'stratigraphic_hmm')
				},
				'embedding_compatibility_signature': {'checkpoint_sha256': parent_sha},
				'embedding_inputs': [source],
			}
		)
	)
	label_metadata = clustering / 'labels/k6/volve_st10010.cluster_label_metadata.json'
	label_metadata.parent.mkdir(parents=True, exist_ok=True)
	label_metadata.write_text(json.dumps({'embedding_input': source}))
	labels_path = clustering / 'labels/k6/volve_st10010.cluster_labels_token.npy'
	np.save(labels_path, np.arange(6, dtype=np.int32).reshape(1, 1, 6))
	target_meta = (
		Path(config['pseudo_targets']['input_dir'])
		/ 'k6/volve_st10010.pseudo_target_metadata.json'
	)
	value = json.loads(target_meta.read_text())
	value['source'] = {
		'source_clustering_output_dir': str(clustering),
		'source_metadata_path': str(label_metadata),
		'source_metadata_sha256': coordination.file_sha256(label_metadata),
		'source_label_path': str(labels_path),
		'source_label_sha256': coordination.file_sha256(labels_path),
	}
	target_meta.write_text(json.dumps(value))


def _plan(config, path):
	result = coordination.classify_coordinated_volve_hmm_run(config, config_path=path)
	assert result is not None
	return result


def _write_checkpoint(config, *, epoch=None):
	smoke = config['train']['max_steps'] == 2
	epoch = (1 if smoke else 25) if epoch is None else epoch
	step = 2 if smoke else epoch * 2500
	root = Path(config['paths']['output_root'])
	definition = coordination.EXPERIMENT / f'20_pretraining/{root.parent.name}.yaml'
	plan = _plan(config, definition)
	plan.lock_path.parent.mkdir(parents=True, exist_ok=True)
	coordination._create_receipt(
		plan, coordination._receipt(plan, coordination._input_snapshot(plan))
	)
	root.mkdir(parents=True, exist_ok=True)
	parent = torch.load(
		config['student']['init_checkpoint'], map_location='cpu', weights_only=False
	)
	model = parent['model_state_dict']
	head = {
		'prototypes': torch.zeros(6, 128),
		'projection.weight': torch.zeros(128, 384),
		'projection.bias': torch.zeros(128),
	}
	names = [name for name in model if name.startswith('encoder.layers.7.')]
	parameters = [*head.values(), *(model[name] for name in names)]
	groups = []
	for name, indices in [('head', list(range(3))), ('encoder', list(range(3, 15)))]:
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
		'scientific_identity': {},
		'resolved_training_config_sha256': coordination._json_hash(config),
		'input_identities': coordination._expected_inputs(config),
		'initial_parameter_sha256': {
			'student_trainable': state._parameter_sha256(
				(name, model[name]) for name in names
			),
			'prototype_head': state._parameter_sha256(head.items()),
		},
		'initial_state_sha256': {
			'student': state._state_dict_sha256(model),
			'head': state._state_dict_sha256(head),
		},
	}
	payload = {
		'config': state._extraction_compatible_config(
			parent['config'],
			output_root=root,
			strat_data_config=config['data'],
			strat_zero_mask_config=config['zero_mask'],
		),
		'stratigraphy_config': config,
		'model_state_dict': model,
		'stratigraphy_state_dict': head,
		'optimizer_state_dict': {
			'param_groups': groups,
			'state': {
				index: {
					'step': torch.tensor(float(step)),
					'exp_avg': torch.zeros_like(value),
					'exp_avg_sq': torch.zeros_like(value),
				}
				for index, value in enumerate(parameters)
			},
		},
		'epoch': epoch,
		'global_step': step,
		'amp_enabled': False,
		'scaler_state_dict': None,
		'package_version': 'test',
		'metrics': {'loss': 1.0},
		'control_identity': control,
		'trainability_summary': {
			'trainable_names': names,
			'trainable_parameter_count': sum(model[name].numel() for name in names),
			'frozen_parameter_count': 3,
		},
		'training_state': {
			'schema_version': 1,
			'stage': 'train_strat_hmm_pretext',
			'checkpoint_kind': 'step' if smoke else 'epoch',
			'batch_index': 1 if smoke else None,
		},
		'rng_state': {
			'python': random.getstate(),
			'numpy': np.random.get_state(),  # noqa: NPY002 - Existing saved RNG schema.
			'torch': torch.get_rng_state(),
			'dataloader_generator': torch.Generator().get_state(),
			'torch_cuda': [torch.zeros(16, dtype=torch.uint8)],
		},
	}
	torch.save(payload, root / 'latest.pt')
	shutil.copyfile(root / 'latest.pt', root / 'best.pt')
	(root / 'resolved_config.json').write_text(json.dumps(config))
	(root / 'run_metadata.json').write_text(json.dumps({'control_identity': control}))
	return root / 'latest.pt'


def _files(root):
	return {
		str(p): (coordination.file_sha256(p), p.stat().st_mtime_ns)
		for p in root.rglob('*')
		if p.is_file()
	}


def _forbidden(*_args, **_kwargs):
	raise AssertionError('must not initialize CUDA or reach training')


@pytest.mark.parametrize('suffix', ['010', '020'])
@pytest.mark.parametrize('smoke', [False, True])
def test_four_exact_scopes_are_read_only(tmp_path, monkeypatch, suffix, smoke):
	config, path = _config(tmp_path, monkeypatch, suffix, smoke=smoke)
	plan = _plan(config, path)
	assert plan.smoke is smoke
	assert plan.recipe.endswith(suffix)
	assert not plan.output_root.exists()
	assert not plan.lock_path.exists()
	assert plan.lock_path.parent.parent == plan.output_root.parent.parent


@pytest.mark.parametrize(
	'change',
	[
		'loss',
		'budget',
		'device',
		'workers',
		'overwrite',
		'output',
		'foreign',
		'traversal',
		'alias',
		'config_alias',
		'hardlink_config',
	],
)
def test_exact_scope_rejects_foreign_or_alias_overrides(tmp_path, monkeypatch, change):
	config, path = _config(tmp_path, monkeypatch)
	if change in {'loss', 'budget', 'device', 'workers', 'overwrite'}:
		section, key, value = {
			'loss': ('loss', 'distillation_weight', 0.1),
			'budget': ('train', 'max_steps', 1),
			'device': ('train', 'device', 'cpu'),
			'workers': ('train', 'num_workers', 1),
			'overwrite': ('train', 'allow_overwrite_output', True),
		}[change]
		config[section][key] = value
	elif change == 'output':
		config['paths']['output_root'] = str(tmp_path / 'other')
	elif change == 'foreign':
		path = tmp_path / 'foreign.yaml'
	elif change == 'traversal':
		output = Path(config['paths']['output_root'])
		config['paths']['output_root'] = str(
			output.parent / '..' / output.parent.name / output.name
		)
	elif change == 'alias':
		alias = tmp_path / 'alias'
		alias.symlink_to(config['paths']['output_root'], target_is_directory=True)
		config['paths']['output_root'] = str(alias)
		path = tmp_path / 'foreign.yaml'
	else:
		alias = tmp_path / 'alias.yaml'
		if change == 'hardlink_config':
			# Never hardlink a repository file: a foreign copy has the same content.
			original = tmp_path / 'foreign.yaml'
			original.write_text(path.read_text())
			os.link(original, alias)
		else:
			alias.symlink_to(path)
		path = alias
	with pytest.raises(ValueError, match=r'coordinated|canonical'):
		coordination.run_coordinated_volve_hmm_if_scoped(config, config_path=path)


def test_unrelated_and_parihaka_configs_leave_no_outputs(tmp_path, monkeypatch):
	monkeypatch.setattr(coordination, '_file_lock', _forbidden)
	for filename, output in [
		('other.yaml', tmp_path / 'other'),
		(
			'parihaka.yaml',
			tmp_path / pari.ARTIFACT_STEM / 'random/distill010/full_25ep',
		),
	]:
		assert (
			coordination.run_coordinated_volve_hmm_if_scoped(
				{'paths': {'output_root': str(output)}},
				config_path=tmp_path / filename,
				quarantine_invalid=True,
			)
			is None
		)
	assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('smoke', [False, True])
def test_completed_skip_preserves_all_artifacts_without_cuda(
	tmp_path, monkeypatch, smoke
):
	config, path = _config(tmp_path, monkeypatch, smoke=smoke)
	checkpoint = _write_checkpoint(config)
	before = _files(checkpoint.parent)
	monkeypatch.setattr(torch.cuda, '_lazy_init', _forbidden)
	monkeypatch.setattr(coordination, '_run_training', _forbidden)
	for resume in (None, checkpoint):
		assert (
			coordination.run_coordinated_volve_hmm_if_scoped(
				config, config_path=path, resume=resume
			)
			== checkpoint
		)
	assert _files(checkpoint.parent) == before
	assert _plan(config, path).lock_path.is_file()


def test_partial_resumes_latest_only_after_lock(tmp_path, monkeypatch):
	config, path = _config(tmp_path, monkeypatch)
	checkpoint = _write_checkpoint(config, epoch=2)
	calls = []

	def train(actual, resume):
		assert pari._LOCK_FDS
		assert actual == config
		calls.append(resume)
		return _write_checkpoint(actual)

	monkeypatch.setattr(coordination, '_run_training', train)
	assert (
		coordination.run_coordinated_volve_hmm_if_scoped(config, config_path=path)
		== checkpoint
	)
	assert calls == [checkpoint]


@pytest.mark.parametrize(
	'change',
	[
		'config',
		'config_sha',
		'source_sha',
		'stage',
		'full_step',
		'counter',
		'counter_bool',
		'cuda_missing',
		'cuda_shape',
		'cpu_rng',
		'optimizer_missing',
		'optimizer_subset',
		'optimizer_order',
		'optimizer_dtype',
		'optimizer_negative',
		'optimizer_step',
		'optimizer_lr',
		'optimizer_extra',
		'model_shape',
		'model_dtype',
		'model_nan',
		'frozen_changed',
		'head_shape',
		'head_dtype',
		'metrics_nan',
		'metrics_missing',
		'scaler',
		'trainability',
		'initial_hash',
		'extraction_config',
	],
)
def test_corrupt_saved_state_is_rejected_without_writes(  # noqa: C901, PLR0912, PLR0915
	tmp_path, monkeypatch, change
):
	config, path = _config(tmp_path, monkeypatch)
	checkpoint = _write_checkpoint(config, epoch=2)
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	if change == 'config':
		payload['stratigraphy_config']['train']['num_workers'] = 3
	elif change == 'config_sha':
		payload['control_identity']['resolved_training_config_sha256'] = '0' * 64
	elif change == 'source_sha':
		payload['control_identity']['input_identities']['teacher_checkpoint'][
			'sha256'
		] = '0' * 64
	elif change == 'stage':
		payload['training_state']['stage'] = 'train_amp_mae'
	elif change == 'full_step':
		payload['training_state'].update(checkpoint_kind='step', batch_index=2499)
	elif change == 'counter':
		payload['global_step'] += 1
	elif change == 'counter_bool':
		payload['epoch'] = True
	elif change == 'cuda_missing':
		del payload['rng_state']['torch_cuda']
	elif change == 'cuda_shape':
		payload['rng_state']['torch_cuda'] = [torch.zeros(1, dtype=torch.uint8)]
	elif change == 'cpu_rng':
		payload['rng_state']['torch'] = torch.ones(1, dtype=torch.uint8)
	elif change == 'optimizer_missing':
		payload['optimizer_state_dict'] = {}
	elif change == 'optimizer_subset':
		del payload['optimizer_state_dict']['state'][14]
	elif change == 'optimizer_order':
		payload['optimizer_state_dict']['param_groups'][0]['params'].reverse()
	elif change == 'optimizer_dtype':
		payload['optimizer_state_dict']['state'][3]['exp_avg'] = torch.zeros(
			1, dtype=torch.float64
		)
	elif change == 'optimizer_negative':
		payload['optimizer_state_dict']['state'][3]['exp_avg_sq'].fill_(-1)
	elif change == 'optimizer_step':
		payload['optimizer_state_dict']['state'][3]['step'] -= 1
	elif change == 'optimizer_lr':
		payload['optimizer_state_dict']['param_groups'][0]['lr'] = 1.0
	elif change == 'optimizer_extra':
		payload['optimizer_state_dict']['state'][3]['extra'] = torch.tensor(1)
	elif change == 'model_shape':
		payload['model_state_dict']['frozen'] = torch.ones(4)
	elif change == 'model_dtype':
		payload['model_state_dict']['frozen'] = torch.ones(3, dtype=torch.float64)
	elif change == 'model_nan':
		payload['model_state_dict']['frozen'][0] = float('nan')
	elif change == 'frozen_changed':
		payload['model_state_dict']['frozen'][0] = 5.0
	elif change == 'head_shape':
		payload['stratigraphy_state_dict']['prototypes'] = torch.zeros(6, 127)
	elif change == 'head_dtype':
		payload['stratigraphy_state_dict']['prototypes'] = torch.zeros(
			6, 128, dtype=torch.float64
		)
	elif change == 'metrics_nan':
		payload['metrics']['loss'] = float('nan')
	elif change == 'metrics_missing':
		payload['metrics'] = {}
	elif change == 'scaler':
		payload['scaler_state_dict'] = {'scale': 1.0}
	elif change == 'trainability':
		payload['trainability_summary']['trainable_names'] = []
	elif change == 'initial_hash':
		payload['control_identity']['initial_state_sha256']['student'] = '0' * 64
	else:
		payload['config']['data'] = {}
	torch.save(payload, checkpoint)
	before = _files(checkpoint.parent)
	monkeypatch.setattr(coordination, '_run_training', _forbidden)
	with pytest.raises((ValueError, TypeError, KeyError, RuntimeError)):
		coordination.run_coordinated_volve_hmm_if_scoped(config, config_path=path)
	assert _files(checkpoint.parent) == before


@pytest.mark.parametrize(
	'change',
	[
		'best_nan',
		'best_newer',
		'best_higher_loss',
		'best_missing',
		'metadata',
		'saved_config',
		'foreign_file',
		'partial_files',
		'hardlink',
		'symlink',
	],
)
def test_four_file_output_contract_is_strict(tmp_path, monkeypatch, change):
	config, path = _config(tmp_path, monkeypatch)
	checkpoint = _write_checkpoint(config, epoch=2)
	root = checkpoint.parent
	if change.startswith('best_') and change != 'best_missing':
		best = torch.load(root / 'best.pt', map_location='cpu', weights_only=False)
		if change == 'best_nan':
			best['model_state_dict']['frozen'][0] = float('nan')
		elif change == 'best_newer':
			best['epoch'], best['global_step'] = 3, 7500
		else:
			best['metrics']['loss'] = 2.0
		torch.save(best, root / 'best.pt')
	elif change == 'best_missing':
		(root / 'best.pt').unlink()
	elif change == 'metadata':
		(root / 'run_metadata.json').write_text('{}')
	elif change == 'saved_config':
		(root / 'resolved_config.json').write_text('{}')
	elif change == 'foreign_file':
		(root / 'foreign').touch()
	elif change == 'partial_files':
		checkpoint.unlink()
	elif change == 'hardlink':
		os.link(checkpoint, tmp_path / 'alias.pt')
	else:
		checkpoint.rename(tmp_path / 'old.pt')
		checkpoint.symlink_to(tmp_path / 'old.pt')
	monkeypatch.setattr(coordination, '_run_training', _forbidden)
	with pytest.raises((ValueError, TypeError, KeyError)):
		coordination.run_coordinated_volve_hmm_if_scoped(config, config_path=path)


@pytest.mark.parametrize(
	'change',
	[
		'missing_target',
		'target_nan',
		'target_count',
		'target_schema',
		'target_survey',
		'parent_kind',
		'parent_shape',
		'foreign_resume',
		'quarantine',
	],
)
def test_fresh_inputs_and_foreign_resume_fail_before_gpu(tmp_path, monkeypatch, change):
	config, path = _config(tmp_path, monkeypatch)
	inputs = coordination._expected_inputs(config)
	target = inputs['pseudo_targets'][0]
	kwargs = {}
	if change == 'missing_target':
		Path(target['boundary_weight']['path']).unlink()
	elif change == 'target_nan':
		np.save(
			target['confidence']['path'], np.full((1, 1, 6), np.nan, dtype=np.float32)
		)
	elif change.startswith('target_'):
		meta = Path(target['metadata']['path'])
		value = json.loads(meta.read_text())
		key, wrong = {
			'target_count': ('valid_token_count', 5),
			'target_schema': ('schema_version', 1),
			'target_survey': ('survey_id', 'wrong'),
		}[change]
		value[key] = wrong
		meta.write_text(json.dumps(value))
	elif change.startswith('parent_'):
		parent = Path(config['teacher']['checkpoint'])
		payload = torch.load(parent, map_location='cpu', weights_only=False)
		if change == 'parent_kind':
			payload['training_state']['checkpoint_kind'] = 'epoch'
		else:
			payload['model_state_dict']['frozen'] = torch.ones(1)
		torch.save(payload, parent)
	elif change == 'foreign_resume':
		foreign = tmp_path / 'foreign.pt'
		foreign.touch()
		kwargs['resume'] = foreign
	else:
		kwargs['quarantine_invalid'] = True
	monkeypatch.setattr(coordination, '_run_training', _forbidden)
	with pytest.raises((ValueError, TypeError, KeyError, FileNotFoundError)):
		coordination.run_coordinated_volve_hmm_if_scoped(
			config, config_path=path, **kwargs
		)
	assert not Path(config['paths']['output_root']).exists()


def test_rechecks_new_completion_after_wait_without_calling_training(
	tmp_path, monkeypatch
):
	config, path = _config(tmp_path, monkeypatch)

	@contextmanager
	def lock(_path):
		_write_checkpoint(config)
		yield

	monkeypatch.setattr(coordination, '_file_lock', lock)
	monkeypatch.setattr(coordination, '_run_training', _forbidden)
	assert (
		coordination.run_coordinated_volve_hmm_if_scoped(config, config_path=path).name
		== 'latest.pt'
	)


def test_changed_config_while_waiting_is_rejected(tmp_path, monkeypatch):
	config, path = _config(tmp_path, monkeypatch)

	@contextmanager
	def lock(_path):
		monkeypatch.setattr(coordination, 'load_config', lambda _path: {'paths': {}})
		yield

	monkeypatch.setattr(coordination, '_file_lock', lock)
	monkeypatch.setattr(coordination, '_run_training', _forbidden)
	with pytest.raises((ValueError, TypeError, KeyError)):
		coordination.run_coordinated_volve_hmm_if_scoped(config, config_path=path)


def test_inputs_changed_during_training_are_rejected(tmp_path, monkeypatch):
	config, path = _config(tmp_path, monkeypatch)

	def train(actual, _resume):
		result = _write_checkpoint(actual)
		Path(actual['manifests']['train']).write_text('{"changed": true}')
		return result

	monkeypatch.setattr(coordination, '_run_training', train)
	with pytest.raises(ValueError, match='inputs changed during training'):
		coordination.run_coordinated_volve_hmm_if_scoped(config, config_path=path)


def _worker(config, path, queue, barrier, inside):
	torch.set_num_threads(2)
	state._model_template = _tiny_template
	state._initial_head = _tiny_head
	torch.cuda._lazy_init = _forbidden

	def training(actual, _resume):
		queue.put(('start', time.monotonic()))
		if inside is not None:
			inside.wait(timeout=20)
		time.sleep(0.15)
		result = _write_checkpoint(actual)
		queue.put(('finish', time.monotonic()))
		return result

	coordination._run_training = training
	barrier.wait(timeout=20)
	coordination.run_coordinated_volve_hmm_if_scoped(config, config_path=Path(path))
	queue.put(('done', time.monotonic()))


@pytest.mark.parametrize('pair', ['same_full', 'full_smoke', 'different_arm'])
def test_processes_serialize_same_arm_but_not_different_arms(
	tmp_path, monkeypatch, pair
):
	first, first_path = _config(tmp_path, monkeypatch)
	second, second_path = _config(
		tmp_path,
		monkeypatch,
		'010' if pair == 'different_arm' else '020',
		smoke=pair == 'full_smoke',
	)
	context = multiprocessing.get_context('spawn')
	queue, barrier = context.Queue(), context.Barrier(2)
	inside = context.Barrier(2) if pair == 'different_arm' else None
	processes = [
		context.Process(
			target=_worker, args=(config, str(path), queue, barrier, inside)
		)
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
	events = [
		kind
		for kind, _time in sorted(events, key=lambda item: item[1])
		if kind != 'done'
	]
	assert (
		events
		== {
			'same_full': ['start', 'finish'],
			'full_smoke': ['start', 'finish', 'start', 'finish'],
			'different_arm': ['start', 'start', 'finish', 'finish'],
		}[pair]
	)


@pytest.mark.parametrize('change', ['hardlink', 'symlink', 'fifo'])
def test_shared_lock_refuses_aliases_and_nonregular_files(
	tmp_path, monkeypatch, change
):
	config, path = _config(tmp_path, monkeypatch)
	lock = _plan(config, path).lock_path
	lock.parent.mkdir(parents=True)
	original = tmp_path / 'sentinel'
	original.write_text('untouched')
	if change == 'hardlink':
		os.link(original, lock)
	elif change == 'symlink':
		lock.symlink_to(original)
	else:
		os.mkfifo(lock)
	with pytest.raises(ValueError, match=r'lock|symlink'):
		coordination.run_coordinated_volve_hmm_if_scoped(config, config_path=path)
	assert original.read_text() == 'untouched'


def _fork_probe(fd, inode, connection):
	try:
		opened = os.fstat(fd)
	except OSError:
		connection.send(True)  # noqa: FBT003 - IPC result.
	else:
		connection.send((opened.st_dev, opened.st_ino) != inode)


def test_reused_lock_closes_inherited_fork_descriptor(tmp_path):
	context = multiprocessing.get_context('fork')
	read, write = context.Pipe(duplex=False)
	with coordination._file_lock(tmp_path / 'lock'):
		fd = next(iter(pari._LOCK_FDS))
		stat = os.fstat(fd)
		process = context.Process(
			target=_fork_probe, args=(fd, (stat.st_dev, stat.st_ino), write)
		)
		process.start()
		process.join(timeout=10)
		assert process.exitcode == 0
		assert read.recv() is True


@pytest.mark.parametrize('smoke', [False, True])
def test_cli_dry_run_remains_no_lock_no_output(tmp_path, monkeypatch, smoke):
	config, path = _config(tmp_path, monkeypatch, smoke=smoke)
	plan = _plan(config, path)
	cli = (
		coordination.EXPERIMENT.parents[3]
		/ 'proc/seis_ssl_cluster/train_strat_hmm_pretext.py'
	)
	command = [sys.executable, str(cli), '--config', str(path), '--dry-run']
	if smoke:
		command.extend(['--max-steps', '2', '--output-root', str(plan.output_root)])
	result = subprocess.run(command, capture_output=True, text=True, check=True)  # noqa: S603 - Repository CLI, temporary outputs.
	assert 'training skipped' in result.stdout
	assert not plan.output_root.exists()
	assert not plan.lock_path.exists()


@pytest.mark.parametrize(
	'marker',
	[
		'stratigraphy_checkpoint',
		'spatial_context_state_dict',
		'target_refresh_state',
		'checkpoint_selection',
		'epoch_metrics_state',
	],
)
def test_foreign_checkpoint_modes_fail_closed(tmp_path, monkeypatch, marker):
	config, path = _config(tmp_path, monkeypatch)
	checkpoint = _write_checkpoint(config)
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload[marker] = {'schema_version': 2}
	torch.save(payload, checkpoint)
	with pytest.raises((ValueError, TypeError), match=r'single-head|schema'):
		coordination.inspect_coordinated_volve_hmm_checkpoint(_plan(config, path))


@pytest.mark.parametrize('key', ['prototype_head', 'head'])
@pytest.mark.parametrize('mode', ['missing', 'foreign'])
def test_initial_head_identity_is_required_and_recomputed(
	tmp_path, monkeypatch, key, mode
):
	config, path = _config(tmp_path, monkeypatch)
	checkpoint = _write_checkpoint(config)
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	section = payload['control_identity'][
		'initial_parameter_sha256'
		if key == 'prototype_head'
		else 'initial_state_sha256'
	]
	if mode == 'missing':
		del section[key]
	else:
		section[key] = '0' * 64
	torch.save(payload, checkpoint)
	with pytest.raises(ValueError, match='initial student provenance'):
		coordination.inspect_coordinated_volve_hmm_checkpoint(_plan(config, path))


@pytest.mark.parametrize('decoupled', [None, True, False, 1])
def test_adamw_version_flag_requires_true_when_present(
	tmp_path, monkeypatch, decoupled
):
	config, path = _config(tmp_path, monkeypatch)
	checkpoint = _write_checkpoint(config)
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	if decoupled is not None:
		for group in payload['optimizer_state_dict']['param_groups']:
			group['decoupled_weight_decay'] = decoupled
	torch.save(payload, checkpoint)
	shutil.copyfile(checkpoint, checkpoint.parent / 'best.pt')
	if decoupled is None or decoupled is True:
		assert coordination.inspect_coordinated_volve_hmm_checkpoint(
			_plan(config, path)
		)
	else:
		with pytest.raises(ValueError, match='decoupled'):
			coordination.inspect_coordinated_volve_hmm_checkpoint(_plan(config, path))


def test_real_installed_torch_adamw_groups_are_compatible(tmp_path, monkeypatch):
	config, _path = _config(tmp_path, monkeypatch)
	checkpoint = _write_checkpoint(config)
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	parameters = [
		torch.nn.Parameter(value.clone())
		for value in state._model_parameters(payload, config)
	]
	optimizer = torch.optim.AdamW(
		[
			{'name': 'head', 'params': parameters[:3], 'lr': 1e-5},
			{'name': 'encoder', 'params': parameters[3:], 'lr': 1e-5},
		],
		weight_decay=0.05,
	)
	for parameter in parameters:
		parameter.grad = torch.zeros_like(parameter)
	optimizer.step()
	saved = optimizer.state_dict()
	for entry in saved['state'].values():
		entry['step'].fill_(62500)
	payload['optimizer_state_dict'] = saved
	state._optimizer(payload, config, parameters)


@pytest.mark.parametrize(
	'change',
	[
		'parent_sha',
		'source_label_sha',
		'clustering_seed',
		'clustering_parent',
		'embedding_parent',
		'label_array',
	],
)
def test_parent_clustering_target_lineage_is_checked_before_gpu(
	tmp_path, monkeypatch, change
):
	config, path = _config(tmp_path, monkeypatch)
	declared = load_config(coordination.EXPERIMENT / '10_hmm_targets/random_init.yaml')
	clustering = Path(declared['clustering']['output_dir'])
	target = (
		Path(config['pseudo_targets']['input_dir'])
		/ 'k6/volve_st10010.pseudo_target_metadata.json'
	)
	if change in {'parent_sha', 'source_label_sha'}:
		metadata = json.loads(target.read_text())
		if change == 'parent_sha':
			metadata['source'].update(
				{
					'checkpoint_path': config['teacher']['checkpoint'],
					'checkpoint_sha256': '0' * 64,
				}
			)
		else:
			metadata['source']['source_label_sha256'] = '0' * 64
		target.write_text(json.dumps(metadata))
	elif change in {'clustering_seed', 'clustering_parent'}:
		filename = clustering / 'models/k6/clustering_metadata.json'
		metadata = json.loads(filename.read_text())
		if change == 'clustering_seed':
			metadata['random_seed'] = 0
		else:
			metadata['embedding_compatibility_signature']['checkpoint_sha256'] = (
				'0' * 64
			)
		filename.write_text(json.dumps(metadata))
	elif change == 'embedding_parent':
		filename = (
			Path(declared['embeddings']['input_dir'])
			/ 'volve_st10010.embedding_metadata.json'
		)
		metadata = json.loads(filename.read_text())
		metadata['checkpoint_sha256'] = '0' * 64
		filename.write_text(json.dumps(metadata))
	else:
		np.save(
			clustering / 'labels/k6/volve_st10010.cluster_labels_token.npy',
			np.zeros((1, 1, 6), dtype=np.int32),
		)
	monkeypatch.setattr(coordination, '_run_training', _forbidden)
	with pytest.raises((ValueError, TypeError)):
		coordination.run_coordinated_volve_hmm_if_scoped(config, config_path=path)
	assert not Path(config['paths']['output_root']).exists()


def test_saved_partial_target_change_while_waiting_is_rejected(tmp_path, monkeypatch):
	config, path = _config(tmp_path, monkeypatch)
	_write_checkpoint(config, epoch=2)

	@contextmanager
	def lock(_path):
		metadata = (
			Path(config['pseudo_targets']['input_dir'])
			/ 'k6/volve_st10010.pseudo_target_metadata.json'
		)
		metadata.write_text(metadata.read_text() + '\n')
		yield

	monkeypatch.setattr(coordination, '_file_lock', lock)
	monkeypatch.setattr(coordination, '_run_training', _forbidden)
	with pytest.raises(ValueError, match=r'control provenance|input receipt'):
		coordination.run_coordinated_volve_hmm_if_scoped(config, config_path=path)


def test_older_best_runtime_receipt_remains_valid_after_resume(tmp_path, monkeypatch):
	config, path = _config(tmp_path, monkeypatch)
	checkpoint = _write_checkpoint(config, epoch=2)
	best = torch.load(checkpoint, map_location='cpu', weights_only=False)
	best['control_identity']['runtime_identity'] = {'git_commit': 'a' * 40}
	checkpoint = _write_checkpoint(config)
	torch.save(best, checkpoint.parent / 'best.pt')
	assert coordination.inspect_coordinated_volve_hmm_checkpoint(_plan(config, path))


@pytest.mark.parametrize('kind', ['python', 'numpy'])
def test_rng_cached_gaussian_nan_is_rejected(tmp_path, monkeypatch, kind):
	config, path = _config(tmp_path, monkeypatch)
	checkpoint = _write_checkpoint(config)
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	rng = list(payload['rng_state'][kind])
	rng[-1] = float('nan')
	if kind == 'numpy':
		rng[-2] = 1
	payload['rng_state'][kind] = tuple(rng)
	with pytest.raises(ValueError, match='RNG numeric cache'):
		coordination._validate_payload(
			payload, _plan(config, path), coordination._expected_inputs(config)
		)


@pytest.mark.parametrize(
	'change',
	[
		'missing',
		'malformed',
		'foreign',
		'symlink',
		'hardlink',
		'manifest',
		'source_embedding',
	],
)
def test_existing_checkpoint_requires_immutable_input_receipt(
	tmp_path, monkeypatch, change
):
	config, path = _config(tmp_path, monkeypatch)
	checkpoint = _write_checkpoint(config, epoch=2)
	plan = _plan(config, path)
	receipt = coordination._receipt_path(plan)
	before = _files(checkpoint.parent)
	if change == 'missing':
		receipt.unlink()
	elif change == 'malformed':
		receipt.write_text('{')
	elif change == 'foreign':
		receipt.write_text('{}')
	elif change == 'symlink':
		receipt.rename(tmp_path / 'original.json')
		receipt.symlink_to(tmp_path / 'original.json')
	elif change == 'hardlink':
		os.link(receipt, tmp_path / 'alias.json')
	elif change == 'manifest':
		Path(config['manifests']['train']).write_text('{"changed": true}')
	else:
		declared = load_config(
			coordination.EXPERIMENT / '10_hmm_targets/random_init.yaml'
		)
		np.save(
			Path(declared['embeddings']['input_dir']) / 'volve_st10010.embeddings.npy',
			np.ones((1, 1, 6, 384), dtype=np.float16),
		)
	monkeypatch.setattr(coordination, '_run_training', _forbidden)
	with pytest.raises((ValueError, TypeError, FileNotFoundError)):
		coordination.run_coordinated_volve_hmm_if_scoped(config, config_path=path)
	assert _files(checkpoint.parent) == before
	if change == 'missing':
		assert not receipt.exists()
		with pytest.raises(ValueError, match='never backfills'):
			coordination._create_receipt(
				plan, coordination._receipt(plan, coordination._input_snapshot(plan))
			)


@pytest.mark.parametrize('failure', ['write', 'publish', 'race'])
def test_receipt_atomic_creation_failures_never_start_gpu(
	tmp_path, monkeypatch, failure
):
	config, path = _config(tmp_path, monkeypatch)
	plan = _plan(config, path)
	receipt = coordination._receipt_path(plan)
	original_link = os.link

	def failed_fsync(_fd):
		raise OSError('injected receipt write failure')

	def failed_link(source, destination, **kwargs):
		if failure == 'race':
			Path(destination).write_text('concurrent foreign receipt')
			return original_link(source, destination, **kwargs)
		raise OSError('injected receipt publish failure')

	if failure == 'write':
		monkeypatch.setattr(os, 'fsync', failed_fsync)
	else:
		monkeypatch.setattr(os, 'link', failed_link)
	monkeypatch.setattr(coordination, '_run_training', _forbidden)
	with pytest.raises(OSError, match=r'injected|File exists'):
		coordination.run_coordinated_volve_hmm_if_scoped(config, config_path=path)
	assert not plan.output_root.exists()
	assert not list(receipt.parent.glob('*.tmp'))
	if failure == 'race':
		assert receipt.read_text() == 'concurrent foreign receipt'
	else:
		assert not receipt.exists()


def test_empty_output_retry_preserves_existing_valid_receipt(tmp_path, monkeypatch):
	config, path = _config(tmp_path, monkeypatch)
	plan = _plan(config, path)
	plan.lock_path.parent.mkdir(parents=True)
	expected = coordination._receipt(plan, coordination._input_snapshot(plan))
	coordination._create_receipt(plan, expected)
	receipt = coordination._receipt_path(plan)
	before = (receipt.read_bytes(), receipt.stat().st_mtime_ns)
	monkeypatch.setattr(
		coordination, '_run_training', lambda cfg, _resume: _write_checkpoint(cfg)
	)
	coordination.run_coordinated_volve_hmm_if_scoped(config, config_path=path)
	assert (receipt.read_bytes(), receipt.stat().st_mtime_ns) == before


def test_full_smoke_receipts_are_separate_with_one_arm_lock(tmp_path, monkeypatch):
	full, path = _config(tmp_path, monkeypatch)
	smoke, _ = _config(tmp_path, monkeypatch, smoke=True)
	full_plan, smoke_plan = _plan(full, path), _plan(smoke, path)
	assert full_plan.lock_path == smoke_plan.lock_path
	assert coordination._receipt_path(full_plan) != coordination._receipt_path(
		smoke_plan
	)
	_write_checkpoint(full)
	_write_checkpoint(smoke)
	assert coordination.inspect_coordinated_volve_hmm_checkpoint(full_plan)
	assert coordination.inspect_coordinated_volve_hmm_checkpoint(smoke_plan)
