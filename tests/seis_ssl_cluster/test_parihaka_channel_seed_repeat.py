"""Preserve canonical seed contracts and isolate the five-cell requested repeat."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.parihaka import channel_seed_repeat as repeat_module
from seis_ssl_cluster.parihaka.channel_decoder import (
	channel_decoder_config_from_mapping,
)

WORKSPACE = Path(__file__).resolve().parents[2]
DEFINITION = (
	WORKSPACE
	/ 'experiments/parihaka/facies_benchmark_v1'
	/ '47_mae_no_hmm_large_decoder_seed42001_v1/execution.yaml'
)


@pytest.fixture
def definition(tmp_path, monkeypatch):
	artifact = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(artifact))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(WORKSPACE))
	raw = load_config(DEFINITION)
	baseline = channel_decoder_config_from_mapping(
		load_config(Path(raw['baseline_config']))
	)
	checkpoint = baseline.models['mae'].expected_checkpoint
	checkpoint.parent.mkdir(parents=True)
	checkpoint.write_bytes(b'fixed-encoder-test-fixture')
	raw['checkpoint_sha256'] = file_sha256(checkpoint)
	for index, layout in enumerate(repeat_module.LAYOUTS):
		path = baseline.runs_root / f'model=mae/layout={layout}/size=large/metrics.json'
		path.parent.mkdir(parents=True)
		payload = {
			'benchmark_identity': {
				'embedding': {'checkpoint_sha256': raw['checkpoint_sha256']},
				'training': {'seed': 42000},
				'decoder_initial_state_sha256': 'seed42000',
				'support': 'same-support',
			},
			'test': dict.fromkeys(
				repeat_module.PUBLIC_METRIC_NAMES, 0.4 + index * 0.05
			),
		}
		path.write_text(json.dumps(payload), encoding='utf-8')
		raw['baseline_metrics_sha256'][layout] = file_sha256(path)
	path = tmp_path / 'execution.yaml'
	path.write_text(yaml.safe_dump(raw), encoding='utf-8')
	return path, raw


@pytest.fixture
def repeat(definition):
	return repeat_module.load_repeat(definition[0])


@pytest.fixture
def planning(repeat, monkeypatch):
	def plan(config, *, model, layout_id, data_size, layout_config):
		assert model == 'mae'
		assert data_size == 'large'
		assert layout_config == repeat.layout_config
		return SimpleNamespace(
			config=config,
			model=model,
			layout_id=layout_id,
			output_dir=config.runs_root / f'model=mae/layout={layout_id}/size=large',
			geometry=SimpleNamespace(
				patch_size_xyz=(8, 8, 8),
				models={'mae': SimpleNamespace(metadata={})},
			),
			tile_counts={'train': 2, 'validation': 2, 'test': 2},
		)

	def identity(plan):
		return {
			'embedding': {'checkpoint_sha256': repeat.definition['checkpoint_sha256']},
			'training': {'seed': plan.config.train.seed},
			'decoder_initial_state_sha256': f'seed{plan.config.train.seed}',
			'support': 'same-support',
		}

	monkeypatch.setattr(repeat_module, 'inspect_channel_decoder_job', plan)
	monkeypatch.setattr(repeat_module, 'channel_decoder_run_identity', identity)
	monkeypatch.setattr(
		repeat_module,
		'decoder_initial_state_sha256',
		lambda _a, _b, seed: f'seed{seed}',
	)
	monkeypatch.setattr(
		repeat_module,
		'inspect_completed_channel_job',
		lambda *_a, **_kw: {'epochs': 50},
	)
	return repeat


def _write_candidates(repeat):
	for index, layout in enumerate(repeat_module.LAYOUTS):
		plan = repeat_module.inspect_repeat_plan(repeat, layout)
		path = plan.output_dir / 'metrics.json'
		path.parent.mkdir(parents=True)
		path.write_text(
			json.dumps(
				{
					'benchmark_identity': repeat_module.channel_decoder_run_identity(
						plan
					),
					'test': dict.fromkeys(
						repeat_module.PUBLIC_METRIC_NAMES, 0.45 + index * 0.05
					),
				}
			),
			encoding='utf-8',
		)


def test_repeat_changes_only_seed_and_output_and_selects_only_mae(repeat):
	assert tuple(repeat.candidate.models) == ('mae',)
	assert repeat.baseline.train.seed == 42000
	assert repeat.candidate.train.seed == 42001
	assert repeat.candidate.train.epochs == 50
	assert repeat.candidate.decoder == repeat.baseline.decoder
	assert repeat.candidate.tiles == repeat.baseline.tiles
	assert repeat.candidate.models == repeat.baseline.models
	assert repeat.candidate.runs_root != repeat.baseline.runs_root
	assert (
		repeat.candidate.models['mae'].expected_checkpoint
		== repeat.baseline.models['mae'].expected_checkpoint
	)


@pytest.mark.parametrize(
	('key', 'value'),
	[
		('model', 'random'),
		('data_size', 'medium'),
		('decoder_seed', 42000),
		('decoder_seed', True),
		('decoder_seed', -1),
		('layouts', ['layout_000']),
		('baseline_config_sha256', '0' * 64),
		('layout_config_sha256', '0' * 64),
		('checkpoint_sha256', '0' * 64),
		('runs_root', '/foreign-runs'),
		('summary_root', '/foreign-summary'),
		('operations_root', '/foreign-operations'),
	],
)
def test_repeat_rejects_scope_identity_and_output_drift(definition, key, value):
	path, raw = definition
	raw[key] = value
	path.write_text(yaml.safe_dump(raw), encoding='utf-8')
	with pytest.raises(ValueError, match=r'scope|hash drift|namespace'):
		repeat_module.load_repeat(path)


def test_canonical_resolver_still_rejects_other_seed(definition):
	raw = load_config(Path(definition[1]['baseline_config']))
	raw['train']['seed'] = 42001
	with pytest.raises(ValueError, match='train settings differ'):
		channel_decoder_config_from_mapping(raw)


def test_baseline_hash_and_supervision_are_checked(planning, monkeypatch):
	plan = repeat_module.inspect_repeat_plan(planning, 'layout_000')
	identity = repeat_module.channel_decoder_run_identity(plan)
	identity['support'] = 'changed-support'
	monkeypatch.setattr(
		repeat_module, 'channel_decoder_run_identity', lambda _p: identity
	)
	with pytest.raises(ValueError, match='changed supervision'):
		repeat_module.inspect_repeat_plan(planning, 'layout_000')


def test_reject_changed_baseline_metrics(planning):
	path = (
		planning.baseline.runs_root
		/ 'model=mae/layout=layout_000/size=large/metrics.json'
	)
	path.write_text('{}', encoding='utf-8')
	with pytest.raises(ValueError, match='baseline metrics hash drift'):
		repeat_module.inspect_repeat_plan(planning, 'layout_000')


def test_completed_candidate_must_match_new_seed(planning):
	_write_candidates(planning)
	plan = repeat_module.inspect_repeat_plan(planning, 'layout_000')
	path = plan.output_dir / 'metrics.json'
	metrics = json.loads(path.read_text())
	metrics['benchmark_identity']['training']['seed'] = 42000
	path.write_text(json.dumps(metrics), encoding='utf-8')
	with pytest.raises(ValueError, match='differs from the seed-repeat plan'):
		repeat_module.audit_repeat_cell(planning, plan)


def test_complete_summary_owns_exact_files_and_all_statistics(planning):
	_write_candidates(planning)
	paths = repeat_module.publish_summary(planning)
	assert {path.name for path in paths} == set(repeat_module.PUBLISHED_FILES)
	assert {path.name for path in planning.summary_root.iterdir()} == set(
		repeat_module.PUBLISHED_FILES
	)
	payload = json.loads((planning.summary_root / 'summary.json').read_text())
	assert payload['cell_count'] == 5
	assert len(payload['rows']) == 5
	assert set(payload['aggregates']) == set(repeat_module.PUBLIC_METRIC_NAMES)
	for stats in payload['aggregates'].values():
		for statistic in ('mean', 'median', 'trimmed_mean'):
			assert stats['baseline'][statistic] == pytest.approx(0.5)
			assert stats['repeat'][statistic] == pytest.approx(0.55)
		assert stats['paired_mean_delta'] == pytest.approx(0.05)
	before = {path.name: path.read_bytes() for path in paths}
	assert repeat_module.publish_summary(planning) == paths
	assert before == {path.name: path.read_bytes() for path in paths}


def test_incomplete_summary_writes_nothing(planning):
	with pytest.raises(FileNotFoundError):
		repeat_module.publish_summary(planning)
	assert not planning.summary_root.exists()


def test_summary_rejects_extra_publication_file(planning):
	_write_candidates(planning)
	repeat_module.publish_summary(planning)
	(planning.summary_root / 'foreign.json').write_text('{}', encoding='utf-8')
	with pytest.raises(ValueError, match='publication conflicts'):
		repeat_module.publish_summary(planning)


def test_dry_run_creates_no_execution_outputs(
	planning, definition, monkeypatch, capsys
):
	monkeypatch.setattr(
		'sys.argv', ['seed-repeat', '--config', str(definition[0]), '--dry-run']
	)
	repeat_module.main()
	assert not planning.operations_root.exists()
	assert not planning.candidate.runs_root.exists()
	assert capsys.readouterr().out.count('"seed": 42001') == 5


def test_feasibility_is_scoped_and_resume_uses_only_own_latest(
	planning, definition, monkeypatch
):
	path = (
		planning.candidate.runs_root
		/ 'model=mae/layout=layout_000/size=large/latest.pt'
	)
	path.parent.mkdir(parents=True)
	path.write_bytes(b'owned-partial')
	calls = []
	monkeypatch.setattr(repeat_module, '_wait_for_gpu', lambda *_args: None)
	monkeypatch.setattr(
		repeat_module,
		'run_channel_decoder_job',
		lambda plan, **kwargs: calls.append((plan, kwargs)),
	)
	monkeypatch.setattr(
		'sys.argv',
		[
			'seed-repeat',
			'--config',
			str(definition[0]),
			'--execute',
			'--layout',
			'layout_000',
			'--max-steps',
			'1',
		],
	)
	repeat_module.main()
	assert len(calls) == 1
	assert calls[0][1]['resume'] == path
	assert calls[0][1]['max_steps'] == 1
	assert (
		json.loads((planning.operations_root / 'status.json').read_text())['status']
		== 'partial'
	)


def test_full_execution_is_serial_fresh_and_reuses_only_audited_own_cells(
	planning, definition, monkeypatch
):
	calls = []

	def run(plan, *, device, max_steps, resume):
		assert device == 'cpu'
		assert max_steps is None
		assert resume is None
		calls.append(plan.layout_id)
		path = plan.output_dir / 'metrics.json'
		path.parent.mkdir(parents=True)
		path.write_text(
			json.dumps(
				{
					'benchmark_identity': repeat_module.channel_decoder_run_identity(
						plan
					),
					'test': dict.fromkeys(repeat_module.PUBLIC_METRIC_NAMES, 0.55),
				}
			),
			encoding='utf-8',
		)
		return path

	monkeypatch.setattr(repeat_module, 'run_channel_decoder_job', run)
	monkeypatch.setattr(
		'sys.argv',
		[
			'seed-repeat',
			'--config',
			str(definition[0]),
			'--execute',
			'--device',
			'cpu',
		],
	)
	repeat_module.main()
	assert calls == list(repeat_module.LAYOUTS)
	status = json.loads((planning.operations_root / 'status.json').read_text())
	assert status['status'] == 'complete_requested_scope'
	assert status['completed_layouts'] == list(repeat_module.LAYOUTS)
	calls.clear()
	repeat_module.main()
	assert calls == []


def test_summary_rejects_changed_completed_values(planning):
	_write_candidates(planning)
	repeat_module.publish_summary(planning)
	path = (
		planning.candidate.runs_root
		/ 'model=mae/layout=layout_000/size=large/metrics.json'
	)
	metrics = json.loads(path.read_text())
	metrics['test']['channel_iou'] += 0.01
	path.write_text(json.dumps(metrics), encoding='utf-8')
	with pytest.raises(ValueError, match='publication conflicts'):
		repeat_module.publish_summary(planning)
