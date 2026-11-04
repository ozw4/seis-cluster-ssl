from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from seis_ssl_cluster.config.f3_lithology_five_way import (
	f3_lithology_five_way_config_from_mapping,
)
from seis_ssl_cluster.f3.io.labels import F3ClassInfo
from seis_ssl_cluster.f3.lithology import decoder_seed_repeat as subject
from seis_ssl_cluster.f3.lithology.common_test import (
	LAYOUTS,
	SIZES,
	SUMMARY_FILES,
)
from seis_ssl_cluster.f3.lithology.common_test import (
	TestPlan as FrozenTestPlan,
)
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	decoder_config_for_job,
	resolve_f3_lithology_five_way_job,
)
from seis_ssl_cluster.f3.lithology.metrics import (
	lithology_metrics_from_confusion_matrix,
)
from tests.seis_ssl_cluster.helpers_f3_five_way import build_five_way_universe


@pytest.fixture
def baseline(tmp_path):
	config = f3_lithology_five_way_config_from_mapping(
		build_five_way_universe(tmp_path / 'source')
	)
	return resolve_f3_lithology_five_way_job(
		config, model='mae', layout='layout_000', size='small'
	)


def test_repeat_changes_only_seed_and_destination(baseline, tmp_path):
	old = decoder_config_for_job(baseline).to_dict()
	job = subject.repeat_job(baseline, tmp_path / 'repeat')
	new = decoder_config_for_job(job).to_dict()
	assert baseline.decoder_seed == 42000
	assert job.decoder_seed == 42001
	assert new['train']['seed'] == 42001
	assert new['outputs'] != old['outputs']
	new['train']['seed'] = 42000
	new['outputs'] = old['outputs']
	assert new == old
	assert job.model == baseline.model
	assert job.condition_dir == baseline.condition_dir
	assert decoder_config_for_job(baseline).to_dict() == old


def test_repeat_rejects_wrong_original_seed_and_output_collision(baseline, tmp_path):
	with pytest.raises(ValueError, match='baseline decoder seed'):
		subject.repeat_job(replace(baseline, decoder_seed=42), tmp_path / 'repeat')
	with pytest.raises(ValueError, match='overwrite'):
		subject.repeat_job(baseline, baseline.config.runs_root)


def test_resume_accepts_only_own_seed_config(baseline, tmp_path, monkeypatch):
	job = subject.repeat_job(baseline, tmp_path / 'repeat')
	job.decoder_dir.mkdir(parents=True)
	latest = job.decoder_dir / 'latest.pt'
	latest.write_bytes(b'checkpoint stub')
	cfg = decoder_config_for_job(job).to_dict()
	(job.decoder_dir / 'resolved_config.json').write_text(json.dumps(cfg))
	payload = {'checkpoint_kind': 'step', 'resolved_config': cfg}
	monkeypatch.setattr(
		subject, 'load_voxel_decoder_checkpoint', lambda *_a, **_k: payload
	)
	calls = []
	monkeypatch.setattr(
		subject,
		'run_f3_lithology_frozen_encoder_job',
		lambda *_a, **kw: calls.append(kw),
	)
	subject._run_job(job)  # noqa: SLF001
	assert calls == [{'device': 'cuda', 'resume': latest}]
	payload['resolved_config'] = decoder_config_for_job(baseline).to_dict()
	with pytest.raises(ValueError, match='different seed/config'):
		subject._run_job(job)  # noqa: SLF001
	assert len(calls) == 1


def _repeat(baseline, tmp_path):
	jobs = tuple(
		subject.repeat_job(
			replace(baseline, layout_id=layout, data_size=size), tmp_path / 'runs'
		)
		for layout in LAYOUTS
		for size in SIZES
	)
	condition = {
		'audit_id': subject.NEW_CONDITION,
		'version': 'v3',
		'source': 'local_bt',
		'head_ks': '6|8|10',
		'distillation_weight': '0.2',
		'consistency_weight': '0.1',
		'mask_token_lr': '3e-5',
		'masked_prototype_weight': '0.5',
		'visible_prototype_weight': '0.5',
	}
	config = {
		'operations_root': str(tmp_path / 'ops'),
		'test_root': str(tmp_path / 'test'),
	}
	classes = tuple(F3ClassInfo(i, str(i), (i, i, i)) for i in range(6))
	plan = FrozenTestPlan({}, classes, (), (), {'fixture': True}, 'fixed-identity')
	return subject.SeedRepeat(
		config, jobs, condition, plan, tmp_path / 'mask.npy', 'fixed-identity'
	)


def test_serial_execution_publication_and_identity_guard(
	baseline, tmp_path, monkeypatch
):
	repeat = _repeat(baseline, tmp_path)
	calls = []
	monkeypatch.setattr(subject, '_wait_for_gpu', lambda *_a: None)
	monkeypatch.setattr(
		subject,
		'_run_job',
		lambda job: calls.append((job.layout_id, job.data_size, job.decoder_seed)),
	)
	inspected = []

	def inspect(*_args, **kwargs):
		inspected.append(kwargs)

	monkeypatch.setattr(subject, 'inspect_test_cell', inspect)
	metrics = lithology_metrics_from_confusion_matrix(
		np.eye(6, dtype=np.int64), repeat.test_plan.classes
	)
	metrics['evaluation_voxel_count'] = 6
	monkeypatch.setattr(subject, 'evaluate_cell', lambda *_a: metrics)
	monkeypatch.setattr(subject.torch.cuda, 'is_available', lambda: False)
	subject.run_repeat(repeat)
	assert len(calls) == 15
	assert len(set(calls)) == 15
	assert inspected == [{'decoder_seed': 42001}] * 15
	summary = tmp_path / 'test/summary'
	assert {p.name for p in summary.iterdir()} == SUMMARY_FILES
	assert json.loads((summary / 'summary.json').read_text())['cell_count'] == 15
	status = json.loads((tmp_path / 'ops/status.json').read_text())
	assert status['status'] == 'complete'
	assert status['completed_cells'] == 15
	with pytest.raises(ValueError, match='identity drift'):
		subject.run_repeat(replace(repeat, identity='wrong'))
	assert len(calls) == 15


def test_training_failure_is_recorded_and_no_summary_published(
	baseline, tmp_path, monkeypatch
):
	repeat = _repeat(baseline, tmp_path)
	monkeypatch.setattr(subject, '_wait_for_gpu', lambda *_a: None)

	def fail(_job):
		raise RuntimeError('training failure')

	monkeypatch.setattr(subject, '_run_job', fail)
	with pytest.raises(RuntimeError, match='training failure'):
		subject.run_repeat(repeat)
	status = json.loads((tmp_path / 'ops/status.json').read_text())
	assert status['status'] == 'failed'
	assert status['completed_cells'] == 0
	assert not (tmp_path / 'test/summary').exists()


def test_no_cuda_does_not_silently_start_cpu_training(tmp_path, monkeypatch):
	monkeypatch.setattr(subject.torch.cuda, 'is_available', lambda: False)
	with pytest.raises(RuntimeError, match='CPU training fallback'):
		subject._wait_for_gpu(tmp_path, {})  # noqa: SLF001


def test_preflight_refuses_a_different_seed_or_scope(tmp_path):
	path = tmp_path / 'wrong.yaml'
	path.write_text(
		json.dumps(
			{
				'experiment_id': subject.NAMESPACE,
				'baseline_condition_id': subject.CONDITION,
				'baseline_decoder_seed': 42000,
				'decoder_seed': 43,
				'layouts': list(LAYOUTS),
				'sizes': list(SIZES),
			}
		)
	)
	with pytest.raises(ValueError, match='scope changed'):
		subject.load_repeat(path)
