"""Verify all-source HMM zero/zero repeats preserve controls and report every cell."""

from __future__ import annotations

import copy
import csv
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.parihaka import hmm_zero_seed_repeat as module
from seis_ssl_cluster.parihaka.channel_decoder import (
	channel_decoder_config_from_mapping,
)

WORKSPACE = Path(__file__).resolve().parents[2]
DEFINITION = WORKSPACE / (
	'experiments/parihaka/facies_benchmark_v1/'
	'54_hmm_v2_dis0_con0_decoder_seed42001_v1/execution.yaml'
)


def _identity(seed, checkpoint_sha):
	return {
		'embedding': {
			'checkpoint_sha256': checkpoint_sha,
			'model_source': {
				'stratigraphy_pretext': {
					'head_ks': [6, 8, 10],
					'distillation_weight': 0.0,
					'consistency_weight': 0.0,
				}
			},
		},
		'training': {'seed': seed},
		'decoder_initial_state_sha256': f'seed{seed}',
		'supervision': 'same paired label support',
	}


@pytest.fixture
def definition(tmp_path, monkeypatch):
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(WORKSPACE))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts'))
	raw = load_config(DEFINITION)
	for arm in raw['arms']:
		baseline = channel_decoder_config_from_mapping(
			load_config(arm['baseline_config'])
		)
		checkpoint = baseline.models[arm['model']].expected_checkpoint
		checkpoint.parent.mkdir(parents=True)
		checkpoint.write_bytes(arm['source'].encode())
		arm['checkpoint_sha256'] = file_sha256(checkpoint)
		for layout in module.LAYOUTS:
			for size in module.SIZES:
				path = (
					baseline.runs_root
					/ f'model={arm["model"]}/layout={layout}/size={size}/metrics.json'
				)
				path.parent.mkdir(parents=True)
				path.write_text(
					json.dumps(
						{
							'benchmark_identity': _identity(
								42000, arm['checkpoint_sha256']
							),
							'best_epoch': 1,
							'test': dict.fromkeys(module.PUBLIC_METRIC_NAMES, 0.4),
						}
					)
				)
				arm['baseline_metrics_sha256'][f'{layout}/{size}'] = file_sha256(path)
	path = tmp_path / 'execution.yaml'
	path.write_text(yaml.safe_dump(raw))
	return path, raw


@pytest.fixture
def planning(definition, monkeypatch):
	repeat = module.load_repeat(definition[0])

	def plan(config, *, model, layout_id, data_size, layout_config):
		assert layout_config == repeat.layout_config
		return SimpleNamespace(
			config=config,
			model=model,
			layout_id=layout_id,
			data_size=data_size,
			output_dir=config.runs_root
			/ f'model={model}/layout={layout_id}/size={data_size}',
			geometry=SimpleNamespace(patch_size_xyz=(8, 8, 8)),
			tile_counts={'train': 2, 'validation': 2, 'test': 2},
		)

	def identity(plan):
		arm = next(a for a in repeat.definition['arms'] if a['model'] == plan.model)
		return _identity(plan.config.train.seed, arm['checkpoint_sha256'])

	monkeypatch.setattr(module, 'inspect_channel_decoder_job', plan)
	monkeypatch.setattr(module, 'channel_decoder_run_identity', identity)
	monkeypatch.setattr(
		module, 'decoder_initial_state_sha256', lambda _a, _b, seed: f'seed{seed}'
	)
	monkeypatch.setattr(
		module, 'inspect_completed_channel_job', lambda *_a, **_kw: {'epochs': 50}
	)
	return repeat


def test_seed_only_change_keeps_canonical_resolver_strict(definition):
	repeat = module.load_repeat(definition[0])
	for source in module.SOURCES:
		baseline, candidate = repeat.baselines[source], repeat.candidates[source]
		assert (
			replace(candidate, train=baseline.train, runs_root=baseline.runs_root)
			== baseline
		)
		assert candidate.train.seed == 42001
		assert baseline.train.seed == 42000
		assert candidate.runs_root != baseline.runs_root
	raw = load_config(repeat.definition['arms'][0]['baseline_config'])
	raw['train']['seed'] = 42001
	with pytest.raises(ValueError, match='fixed Channel benchmark'):
		channel_decoder_config_from_mapping(raw)


@pytest.mark.parametrize('change', ['size', 'seed', 'checkpoint', 'baseline', 'output'])
def test_rejects_scope_or_provenance_changes(definition, change):
	path, raw = definition
	if change == 'size':
		raw['sizes'] = ['large']
	elif change == 'seed':
		raw['decoder_seed'] = 42000
	elif change == 'checkpoint':
		raw['arms'][0]['checkpoint_sha256'] = '0' * 64
	elif change == 'baseline':
		raw['arms'][0]['baseline_metrics_sha256'].pop('layout_000/medium')
	else:
		raw['runs_root'] = str(path.parent / 'reports')
	path.write_text(yaml.safe_dump(raw))
	with pytest.raises(ValueError, match=r'thirty|hash drift|both sizes|namespace'):
		module.load_repeat(path)


def test_rejects_different_support_and_wrong_completed_seed(planning, monkeypatch):
	plan = module.inspect_plan(planning, 'mae', 'layout_000', 'medium')
	identity = copy.deepcopy(module.channel_decoder_run_identity(plan))
	identity['supervision'] = 'different label support'
	with monkeypatch.context() as scoped:
		scoped.setattr(module, 'channel_decoder_run_identity', lambda _p: identity)
		with pytest.raises(ValueError, match='other than decoder seed'):
			module.baseline_metrics(planning, plan)
	plan.output_dir.mkdir(parents=True)
	(plan.output_dir / 'metrics.json').write_text(
		json.dumps({'benchmark_identity': {}})
	)
	with pytest.raises(ValueError, match='different benchmark identity'):
		module.audit_cell(planning, plan)


def _completed_rows():
	rows = []
	for source in module.SOURCES:
		for size in module.SIZES:
			for index, layout in enumerate(module.LAYOUTS):
				row = {'source': source, 'data_size': size, 'layout_id': layout}
				for metric in module.PUBLIC_METRIC_NAMES:
					row.update(
						{
							f'baseline_{metric}': 0.4,
							f'repeat_{metric}': 0.3 + index / 100,
							f'delta_{metric}': -0.1 + index / 100,
						}
					)
				rows.append(row)
	return rows


def test_exact_publication_set_keeps_negative_results_and_all_layouts():
	rows = _completed_rows()
	contents = module.publication_contents(rows)
	assert set(contents) == set(module.PUBLISHED_FILES)
	assert len(list(csv.DictReader(io.StringIO(contents['cells.csv'])))) == 30
	groups = list(csv.DictReader(io.StringIO(contents['comparison.csv'])))
	assert len(groups) == 6
	assert all(
		float(group['delta_channel_iou_mean']) == pytest.approx(-0.08)
		for group in groups
	)
	assert all(float(group['delta_channel_iou_std']) > 0 for group in groups)
	for invalid in (rows[:-1], [*rows[:-1], rows[0]]):
		with pytest.raises(ValueError, match='thirty unique'):
			module.publication_contents(invalid)


def test_execution_resumes_own_checkpoint_and_reuses_completed_cells(
	planning, definition, monkeypatch
):
	resuming = module.inspect_plan(planning, 'mae', 'layout_000', 'medium')
	resuming.output_dir.mkdir(parents=True)
	(resuming.output_dir / 'latest.pt').write_bytes(b'own partial checkpoint')
	reused = module.inspect_plan(planning, 'mae', 'layout_001', 'medium')
	reused.output_dir.mkdir(parents=True)
	(reused.output_dir / 'metrics.json').write_text('{}')
	calls, audited, published = [], [], []

	def execute(plan, *, device, resume):
		assert device == 'cpu'
		calls.append((plan.model, plan.layout_id, plan.data_size, resume))

	monkeypatch.setattr(module, 'run_channel_decoder_job', execute)
	monkeypatch.setattr(module, 'audit_cell', lambda _r, plan: audited.append(plan))
	monkeypatch.setattr(module, 'publish_summary', lambda _r: published.append(True))
	monkeypatch.setattr(
		'sys.argv',
		['repeat', '--config', str(definition[0]), '--execute', '--device', 'cpu'],
	)
	module.main()
	assert len(calls) == 29
	assert len(audited) == 30
	assert published == [True]
	assert calls[0][3] == resuming.output_dir / 'latest.pt'
	assert all(call[3] is None for call in calls[1:])
	status = json.loads((planning.operations_root / 'status.json').read_text())
	assert status['status'] == 'complete_requested_scope'
	assert status['completed_count'] == 30


def test_execution_failure_records_error_and_stops_admission(
	planning, definition, monkeypatch
):
	calls = []

	def fail(plan, **_kwargs):
		calls.append(plan)
		raise RuntimeError('training failure')

	monkeypatch.setattr(module, 'run_channel_decoder_job', fail)
	monkeypatch.setattr(
		'sys.argv',
		['repeat', '--config', str(definition[0]), '--execute', '--device', 'cpu'],
	)
	with pytest.raises(RuntimeError, match='training failure'):
		module.main()
	assert len(calls) == 1
	status = json.loads((planning.operations_root / 'status.json').read_text())
	assert status['status'] == 'failed'
	assert status['completed_cells'] == []
	assert status['error'] == 'training failure'
