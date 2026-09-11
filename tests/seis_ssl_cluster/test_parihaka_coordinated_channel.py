"""Scope, input receipts, and shared CPU process locks for canonical Channel cells."""

# ruff: noqa: SLF001 - Focused cooperation and filesystem safety contracts.

from __future__ import annotations

import json
import multiprocessing
import os
import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.parihaka import coordinated_channel as coordination


def invocation(tmp_path, monkeypatch, *, model=None, layout='layout_000', size='small'):
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	model = model or next(iter(coordination.MODELS))
	path = coordination.EXPERIMENT / f'40_downstream/01_{model}.yaml'
	return load_config(path), {
		'config_path': path,
		'model': model,
		'layout_id': layout,
		'data_size': size,
		'layout_config': coordination.LAYOUT_CONFIG,
	}


@pytest.mark.parametrize('model', coordination.MODELS)
@pytest.mark.parametrize('layout', [f'layout_{index:03d}' for index in range(5)])
@pytest.mark.parametrize('size', ['small', 'medium', 'large'])
def test_all_sixty_scopes_are_pure(tmp_path, monkeypatch, model, layout, size):
	raw, args = invocation(tmp_path, monkeypatch, model=model, layout=layout, size=size)
	monkeypatch.setattr(
		torch.cuda, 'is_available', lambda: pytest.fail('scope touched CUDA')
	)
	before = list(tmp_path.rglob('*'))
	scope = coordination.classify_coordinated_channel_run(raw, **args)
	assert (
		scope.output_dir
		== Path(raw['outputs']['runs_root'])
		/ f'model={model}/layout={layout}/size={size}'
	)
	assert (
		coordination.run_coordinated_channel_if_scoped(raw, **args, dry_run=True)
		is None
	)
	assert list(tmp_path.rglob('*')) == before


@pytest.mark.parametrize(
	'mutation',
	[
		'foreign_config',
		'copied_config',
		'config_alias',
		'layout_alias',
		'layout_changed',
		'wrong_model',
		'layout',
		'size',
		'max_steps',
		'validation_only',
		'evaluation_only',
		'wrong_resume',
		'resume_alias',
		'output_alias',
		'output_other',
		'budget',
		'learning_rate',
		'labels',
		'embedding',
		'unknown_key',
		'hardlink_config',
	],
)
def test_scope_escapes_fail_closed(tmp_path, monkeypatch, mutation):  # noqa: C901, PLR0912, PLR0915
	raw, args = invocation(tmp_path, monkeypatch)
	if mutation in {
		'foreign_config',
		'copied_config',
		'config_alias',
		'hardlink_config',
	}:
		path = tmp_path / 'foreign.yaml'
		if mutation == 'copied_config':
			path.write_text(args['config_path'].read_text())
		elif mutation == 'config_alias':
			path.symlink_to(args['config_path'])
		elif mutation == 'hardlink_config':
			# Never hardlink repository files; use a temporary fake definition.
			fake_experiment = tmp_path / 'experiments'
			definition = fake_experiment / f'40_downstream/01_{args["model"]}.yaml'
			definition.parent.mkdir(parents=True)
			definition.write_text(args['config_path'].read_text())
			os.link(definition, path)
			monkeypatch.setattr(coordination, 'EXPERIMENT', fake_experiment)
			args['config_path'] = definition
		else:
			path.write_text('foreign')
		if mutation != 'hardlink_config':
			args['config_path'] = path
	elif mutation == 'layout_alias':
		path = tmp_path / 'layouts.yaml'
		path.symlink_to(args['layout_config'])
		args['layout_config'] = path
	elif mutation == 'layout_changed':
		args['layout_config'] = tmp_path / 'other.yaml'
		args['layout_config'].write_text('foreign')
	elif mutation == 'wrong_model':
		args['model'] = 'random'
	elif mutation == 'layout':
		args['layout_id'] = 'layout_005'
	elif mutation == 'size':
		args['data_size'] = 'tiny'
	elif mutation in {'max_steps', 'validation_only'}:
		args[mutation] = 1 if mutation == 'max_steps' else True
	elif mutation == 'evaluation_only':
		args['evaluate_completed_validation'] = tmp_path / 'old'
	elif mutation == 'wrong_resume':
		args['resume'] = tmp_path / 'latest.pt'
	elif mutation == 'resume_alias':
		args['resume'] = (
			Path(raw['outputs']['runs_root'])
			/ f'model={args["model"]}/layout=layout_000/size=small/x/../latest.pt'
		)
	elif mutation == 'output_alias':
		raw['outputs']['runs_root'] = str(tmp_path / 'x/..' / coordination.RUNS_STEM)
	elif mutation == 'output_other':
		raw['outputs']['runs_root'] = str(tmp_path / 'other')
	elif mutation == 'budget':
		raw['train']['epochs'] = 1
	elif mutation == 'learning_rate':
		raw['train']['learning_rate'] = 0.01
	elif mutation == 'labels':
		raw['inputs']['labels_npy'] = str(tmp_path / 'labels.npy')
	elif mutation == 'embedding':
		raw['embeddings']['models'][args['model']]['dir'] = str(tmp_path / 'embedding')
	else:
		raw['foreign'] = True
	with pytest.raises((ValueError, FileNotFoundError)):
		coordination.classify_coordinated_channel_run(raw, **args)


@pytest.mark.parametrize('outside_runs', [False, True])
def test_model_traversal_into_canonical_cell_never_falls_through(
	tmp_path, monkeypatch, outside_runs
):
	raw, args = invocation(tmp_path, monkeypatch)
	model = args['model']
	alias = f'x/../model={model}'
	if outside_runs:
		raw['outputs']['runs_root'] = str(tmp_path / 'other')
		alias = f'x/../../{coordination.RUNS_STEM}/model={model}'
	raw['embeddings']['models'] = {alias: raw['embeddings']['models'][model]}
	args['model'] = alias
	args['config_path'] = tmp_path / 'foreign.yaml'
	with pytest.raises(ValueError, match='exact model recipe'):
		coordination.classify_coordinated_channel_run(raw, **args)


@pytest.mark.parametrize('model', ['mae_hmm_k6_distill010', 'random', 'pretrained'])
def test_non_target_cli_modes_remain_untouched(tmp_path, monkeypatch, model):
	monkeypatch.delenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', raising=False)
	raw = {
		'outputs': {'runs_root': str(tmp_path / coordination.RUNS_STEM)},
		'embeddings': {'models': {model: {}}},
	}
	assert (
		coordination.run_coordinated_channel_if_scoped(
			raw,
			config_path=tmp_path / 'non_target.yaml',
			model=model,
			layout_id='layout_000',
			data_size='small',
			layout_config=tmp_path / 'unused',
			max_steps=1,
			resume=tmp_path / 'latest.pt',
			validation_only=True,
			evaluate_completed_validation=tmp_path / 'old',
		)
		is None
	)


def _mock_execution(tmp_path, monkeypatch, *, state='fresh'):
	raw, args = invocation(tmp_path, monkeypatch)
	scope = coordination.classify_coordinated_channel_run(raw, **args)
	input_file = tmp_path / 'input.bin'
	input_file.write_bytes(b'stable')
	hashes = coordination._hashes([input_file])
	plan = SimpleNamespace(output_dir=scope.output_dir)
	monkeypatch.setattr(
		coordination, 'inspect_coordinated_channel_inputs', lambda _: (plan, hashes)
	)
	monkeypatch.setattr(
		coordination.decoder, '_run_identity', lambda _: {'fresh': True}
	)
	states = {'value': state}
	monkeypatch.setattr(
		coordination, 'inspect_channel_state', lambda *_: states['value']
	)

	def run(_, *, device, resume):
		assert device == 'auto'
		assert coordination.hmm._LOCK_FDS
		assert resume == (
			scope.output_dir / 'latest.pt' if state == 'partial' else None
		)
		states['value'] = 'complete'
		return scope.output_dir / 'metrics.json'

	monkeypatch.setattr(coordination.decoder, 'run_channel_decoder_job', run)
	return raw, args, scope, hashes


def test_completed_without_receipt_is_readonly_and_stale_resume_is_ignored(
	tmp_path, monkeypatch
):
	raw, args, scope, _ = _mock_execution(tmp_path, monkeypatch, state='complete')
	args['resume'] = scope.output_dir / 'latest.pt'
	monkeypatch.setattr(
		coordination.decoder,
		'run_channel_decoder_job',
		lambda *_args, **_kwargs: pytest.fail('completed cell trained'),
	)
	assert (
		coordination.run_coordinated_channel_if_scoped(raw, **args)
		== scope.output_dir / 'metrics.json'
	)
	assert not scope.receipt_path.exists()
	assert not scope.output_dir.exists()


def test_fresh_execution_seals_inputs_outside_canonical_output(tmp_path, monkeypatch):
	raw, args, scope, hashes = _mock_execution(tmp_path, monkeypatch)
	coordination.run_coordinated_channel_if_scoped(raw, **args)
	assert json.loads(scope.receipt_path.read_text()) == coordination._receipt(
		scope, hashes
	)
	assert scope.receipt_path.stat().st_nlink == 1
	assert scope.receipt_path.parent != scope.output_dir


def test_partial_without_receipt_is_never_resumed(tmp_path, monkeypatch):
	raw, args, _, _ = _mock_execution(tmp_path, monkeypatch, state='partial')
	with pytest.raises(ValueError, match='no pre-training input SHA receipt'):
		coordination.run_coordinated_channel_if_scoped(raw, **args)


def test_valid_sealed_partial_is_automatically_resumed(tmp_path, monkeypatch):
	raw, args, scope, hashes = _mock_execution(tmp_path, monkeypatch, state='partial')
	scope.receipt_path.parent.mkdir(parents=True)
	coordination._write_receipt(
		scope.receipt_path, coordination._receipt(scope, hashes)
	)
	before = scope.receipt_path.read_bytes()
	coordination.run_coordinated_channel_if_scoped(raw, **args)
	assert scope.receipt_path.read_bytes() == before


def test_receipt_atomic_create_never_replaces_existing(tmp_path):
	path = tmp_path / 'inputs.json'
	coordination._write_receipt(path, {'original': True})
	before = path.read_bytes()
	with pytest.raises(FileExistsError):
		coordination._write_receipt(path, {'replacement': True})
	assert path.read_bytes() == before
	assert list(tmp_path.iterdir()) == [path]


def test_changed_receipt_fails_even_for_completed_cell(tmp_path, monkeypatch):
	raw, args, scope, hashes = _mock_execution(tmp_path, monkeypatch, state='complete')
	scope.receipt_path.parent.mkdir(parents=True)
	coordination._write_receipt(
		scope.receipt_path, coordination._receipt(scope, {**hashes, 'foreign': 'x'})
	)
	with pytest.raises(ValueError, match='receipt differs'):
		coordination.run_coordinated_channel_if_scoped(raw, **args)


@pytest.mark.parametrize('change_marker_only', [True, False])
def test_post_training_reuse_marker_change_is_not_a_scientific_change(
	tmp_path,
	monkeypatch,
	change_marker_only,
):
	raw, args, scope, hashes = _mock_execution(tmp_path, monkeypatch)
	marker = tmp_path / 'embedding_extraction_execution.json'
	marker.write_text('fresh disposition')
	inputs = [Path(path) for path in hashes] + [marker]
	plan = SimpleNamespace(output_dir=scope.output_dir)
	monkeypatch.setattr(
		coordination,
		'inspect_coordinated_channel_inputs',
		lambda _: (plan, coordination._hashes(inputs)),
	)
	original_run = coordination.decoder.run_channel_decoder_job

	def run(*call_args, **call_kwargs):
		result = original_run(*call_args, **call_kwargs)
		(marker if change_marker_only else inputs[0]).write_text('changed')
		return result

	monkeypatch.setattr(coordination.decoder, 'run_channel_decoder_job', run)
	if change_marker_only:
		assert (
			coordination.run_coordinated_channel_if_scoped(raw, **args)
			== scope.output_dir / 'metrics.json'
		)
	else:
		with pytest.raises(ValueError, match='scientific inputs changed'):
			coordination.run_coordinated_channel_if_scoped(raw, **args)


def test_config_is_reclassified_after_lock_before_any_gpu_call(tmp_path, monkeypatch):
	raw, args = invocation(tmp_path, monkeypatch)
	load = coordination.load_config
	count = 0

	def changed(path):
		nonlocal count
		count += 1
		result = load(path)
		if count >= 2:
			result['train']['epochs'] = 1
		return result

	monkeypatch.setattr(coordination, 'load_config', changed)
	monkeypatch.setattr(
		torch.cuda, '_lazy_init', lambda: pytest.fail('CUDA before scope audit')
	)
	with pytest.raises(ValueError, match='scientific recipe'):
		coordination.run_coordinated_channel_if_scoped(raw, **args)


def _input_gate_fixture(tmp_path, monkeypatch):
	raw, args = invocation(tmp_path, monkeypatch)
	scope = coordination.classify_coordinated_channel_run(raw, **args)
	config = coordination.decoder.channel_decoder_config_from_mapping(raw)
	source = config.models[scope.model]
	paths = coordination.output_paths(source.embedding_dir, 'parihaka')
	paths.embeddings.parent.mkdir(parents=True)
	np.save(paths.embeddings, np.ones((2, 1, 1, 2), dtype=np.float16))
	np.save(paths.valid_tokens, np.ones((2, 1, 1), dtype=np.bool_))
	metadata = {'scientific': 'exact'}
	paths.metadata.write_text(json.dumps(metadata))
	marker = paths.metadata.parent / 'embedding_extraction_execution.json'
	marker.write_text(
		json.dumps(
			{
				'artifact_type': 'embedding_extraction_execution',
				'schema_version': 1,
				'encoder_input_mode': coordination.UNMASKED_ENCODER_INPUT_MODE,
				'fresh': 1,
				'reuse': 0,
				'survey_count': 1,
			}
		)
	)
	definitions = tmp_path / 'definitions'
	monkeypatch.setattr(coordination, 'EXPERIMENT', definitions)
	train = (
		definitions / f'20_stage2/{coordination.MODELS[scope.model]}/01_full_25ep.yaml'
	)
	extraction = definitions / f'30_embeddings/01_extract_{scope.model}.yaml'
	manifest = tmp_path / coordination.LABEL_STEM / 'parihaka_amplitude_manifest.json'
	for path in (
		source.expected_checkpoint,
		config.labels,
		config.labels_metadata,
		train,
		extraction,
		manifest,
	):
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text('stable')
	monkeypatch.setattr(coordination, '_inspect_source', lambda _: (train, {}))
	monkeypatch.setattr(
		coordination,
		'_expected_metadata',
		lambda *_: (metadata, [extraction, manifest]),
	)
	monkeypatch.setattr(
		coordination.decoder,
		'inspect_channel_decoder_job',
		lambda *_args, **_kwargs: SimpleNamespace(output_dir=scope.output_dir),
	)
	monkeypatch.setattr(
		torch.cuda, '_lazy_init', lambda: pytest.fail('input gate touched CUDA')
	)
	return scope, paths, marker, train, extraction


def test_input_gate_three_hashes_finite_and_stable_without_writes(
	tmp_path, monkeypatch
):
	scope, paths, _, _, _ = _input_gate_fixture(tmp_path, monkeypatch)
	before = {
		str(path): (path.stat().st_size, path.stat().st_mtime_ns)
		for path in tmp_path.rglob('*')
		if path.is_file()
	}
	plan, hashes = coordination.inspect_coordinated_channel_inputs(scope)
	assert plan.output_dir == scope.output_dir
	assert all(
		str(path) in hashes
		for path in (paths.embeddings, paths.valid_tokens, paths.metadata)
	)
	assert {
		str(path): (path.stat().st_size, path.stat().st_mtime_ns)
		for path in tmp_path.rglob('*')
		if path.is_file()
	} == before


@pytest.mark.parametrize(
	'mutation',
	[
		'temporary',
		'missing_array',
		'nan',
		'metadata',
		'marker_missing',
		'marker_incomplete',
		'changing_config',
		'changing_array',
		'symlink',
	],
)
def test_input_gate_corruption_and_publication_races_fail_closed(  # noqa: C901
	tmp_path, monkeypatch, mutation
):
	scope, paths, marker, train, _ = _input_gate_fixture(tmp_path, monkeypatch)
	if mutation == 'temporary':
		paths.embeddings_tmp.write_bytes(b'partial')
	elif mutation == 'missing_array':
		paths.valid_tokens.unlink()
	elif mutation == 'nan':
		np.save(paths.embeddings, np.full((2, 1, 1, 2), np.nan, dtype=np.float16))
	elif mutation == 'metadata':
		paths.metadata.write_text('{}')
	elif mutation == 'marker_missing':
		marker.unlink()
	elif mutation == 'marker_incomplete':
		marker.write_text('{}')
	elif mutation == 'changing_config':
		original = coordination._expected_metadata

		def changed(*args):
			train.write_text('changed during reconstruction')
			return original(*args)

		monkeypatch.setattr(coordination, '_expected_metadata', changed)
	elif mutation == 'changing_array':

		def changed_plan(*_args, **_kwargs):
			np.save(paths.embeddings, np.zeros((2, 1, 1, 2), dtype=np.float16))
			return SimpleNamespace(output_dir=scope.output_dir)

		monkeypatch.setattr(
			coordination.decoder, 'inspect_channel_decoder_job', changed_plan
		)
	else:
		target = tmp_path / 'real.npy'
		paths.valid_tokens.replace(target)
		paths.valid_tokens.symlink_to(target)
	with pytest.raises((ValueError, TypeError, FileNotFoundError)):
		coordination.inspect_coordinated_channel_inputs(scope)


def _competing_worker(root, size, ready, entered, release, results):  # noqa: PLR0913, PLR0917
	os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'] = root
	model = next(iter(coordination.MODELS))
	path = coordination.EXPERIMENT / f'40_downstream/01_{model}.yaml'
	raw = load_config(path)
	args = {
		'config_path': path,
		'model': model,
		'layout_id': 'layout_000',
		'data_size': size,
		'layout_config': coordination.LAYOUT_CONFIG,
	}
	scope = coordination.classify_coordinated_channel_run(raw, **args)
	input_file = Path(root) / 'stable.bin'
	hashes = coordination._hashes([input_file])
	plan = SimpleNamespace(output_dir=scope.output_dir)

	def inspect(_):
		assert coordination.hmm._LOCK_FDS
		return plan, hashes

	coordination.inspect_coordinated_channel_inputs = inspect
	coordination.decoder._run_identity = lambda _: {'fresh': True}
	coordination.inspect_channel_state = lambda *_: (
		'complete' if (scope.output_dir / 'metrics.json').exists() else 'fresh'
	)

	def run(_, **_kwargs):
		entered.put((os.getpid(), size))
		assert release.wait(30)
		scope.output_dir.mkdir(parents=True)
		metrics = scope.output_dir / 'metrics.json'
		metrics.write_text('{}')
		return metrics

	coordination.decoder.run_channel_decoder_job = run
	try:
		ready.put((os.getpid(), size))
		coordination.run_coordinated_channel_if_scoped(raw, **args)
		results.put(('ok', size))
	except BaseException as exc:
		results.put(('error', repr(exc)))
		raise


@pytest.mark.parametrize('same_cell', [True, False])
def test_spawned_future_main_aux_share_cell_lock_and_skip_after_wait(
	tmp_path, same_cell
):
	context = multiprocessing.get_context('spawn')
	(tmp_path / 'stable.bin').write_bytes(b'stable')
	ready, entered, results = context.Queue(), context.Queue(), context.Queue()
	release = context.Event()
	first = context.Process(
		target=_competing_worker,
		args=(str(tmp_path), 'small', ready, entered, release, results),
	)
	second = context.Process(
		target=_competing_worker,
		args=(
			str(tmp_path),
			'small' if same_cell else 'large',
			ready,
			entered,
			release,
			results,
		),
	)
	first.start()
	assert ready.get(timeout=30)[1] == 'small'
	assert entered.get(timeout=30)[1] == 'small'
	second.start()
	assert ready.get(timeout=30)[1] == ('small' if same_cell else 'large')
	if same_cell:
		# A release barrier controls the writer; the second must not enter training.
		with pytest.raises(queue.Empty):
			entered.get(timeout=3)
	else:
		assert entered.get(timeout=30)[1] == 'large'
	release.set()
	for process in (first, second):
		process.join(timeout=30)
		assert process.exitcode == 0
	assert results.get(timeout=5)[0] == results.get(timeout=5)[0] == 'ok'
