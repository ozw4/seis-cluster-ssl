"""Paired source provenance, complete canonical cells and exact publication."""

from __future__ import annotations

# ruff: noqa: SLF001
import copy
import json
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster.hmm import trace_mask_summary as summary
from seis_ssl_cluster.hmm.multi_source_receipts import (
	CELL_IDENTITIES,
	METRICS,
	SOURCE_FAMILIES,
	file_sha256,
)
from seis_ssl_cluster.hmm.trace_mask_definition import CANDIDATE_IDS


def _json(path, value):
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def paired_arm(tmp_path, monkeypatch):
	experiment = tmp_path / 'experiments/parihaka/facies_benchmark_v1/v3'
	experiment.mkdir(parents=True)
	artifacts = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(artifacts))
	candidate_id = CANDIDATE_IDS['mae']
	baseline_id = 'mae100_hmm_v2_mh_k6810_distill020_cons010'
	cell = {
		'survey': 'parihaka',
		'source': 'mae',
		'candidate_id': candidate_id,
		'target_manifest_sha256': 'a' * 64,
	}
	configs, lineages, frozen = {}, {}, []
	for baseline, model in ((False, candidate_id), (True, baseline_id)):
		root = artifacts / ('baseline_runs' if baseline else 'candidate_runs')
		checkpoint = artifacts / model / 'latest.pt'
		checkpoint.parent.mkdir(parents=True)
		checkpoint.write_bytes(model.encode())
		field = 'baseline_downstream_config' if baseline else 'downstream_config'
		cell[field] = str(experiment / (field + '.yaml'))
		configs[cell[field]] = {
			'outputs': {'runs_root': str(root)},
			'embeddings': {'models': {model: {'checkpoint': str(checkpoint)}}},
		}
		lineages[baseline] = {
			'model_tag': model,
			'head_ks': [6, 8, 10],
			'epoch': 25,
			'checkpoint': {'path': str(checkpoint), 'sha256': file_sha256(checkpoint)},
			'target_manifest': {'sha256': cell['target_manifest_sha256']},
		}
		for layout, size in CELL_IDENTITIES:
			metrics = (
				root
				/ f'model={model}'
				/ f'layout={layout}'
				/ f'size={size}'
				/ 'metrics.json'
			)
			_json(
				metrics,
				{'test': {'channel_iou': 0.5 if baseline else 0.55, 'channel_f1': 0.6}},
			)
			if baseline:
				frozen.append(
					{
						'survey': 'parihaka',
						'source_family': 'mae',
						'candidate_id': model,
						'layout_id': layout,
						'data_size': size,
						'primary': 0.5,
						'secondary': 0.6,
					}
				)
	cell['baseline_train_config'] = str(experiment / 'baseline_train.yaml')
	Path(cell['baseline_train_config']).write_text(
		yaml.safe_dump({'identity': {'model_tag': baseline_id}}), encoding='utf-8'
	)
	cell['baseline_checkpoint_sha256'] = lineages[True]['checkpoint']['sha256']
	selection = experiment / 'selection.yaml'
	selection.write_text(
		yaml.safe_dump({'evidence': {'artifact_summary': 'canonical.json'}}),
		encoding='utf-8',
	)
	(experiment / 'execution.yaml').write_text(
		yaml.safe_dump({'canonical_selection': {'path': str(selection)}}),
		encoding='utf-8',
	)
	_json(artifacts / 'canonical.json', {'cells': frozen})
	monkeypatch.setattr(summary, 'load_config', lambda path: configs[str(path)])
	monkeypatch.setattr(
		summary, '_lineage', lambda _experiment, _cell, *, baseline: lineages[baseline]
	)
	calls = []

	def audit(survey, _raw, model, layout, size):
		calls.append((survey, model, layout, size))
		return {
			'benchmark_identity': {
				'model': model,
				'embedding': model,
				'decoder': 'shared',
				'layout': layout,
				'size': size,
			}
		}

	monkeypatch.setattr(summary, '_audit_cell', audit)
	return experiment, cell, configs, lineages, calls


def test_arm_audits_both_sources_and_all_15_matched_decoder_cells(paired_arm):
	experiment, cell, _configs, _lineages, calls = paired_arm
	cells, evidence = summary._audited_arm(experiment, 'parihaka', cell)
	assert len(cells) == 15
	assert len(calls) == 30
	assert len(evidence['paired_cells']) == 15
	assert evidence['source']['model_tag'] == cell['candidate_id']
	assert all(
		row['primary'] == 0.55 and row['control_primary'] == 0.5 for row in cells
	)


def test_arm_can_bind_explicit_frozen_control_without_v02_selection(paired_arm):
	experiment, cell, _configs, _lineages, calls = paired_arm
	artifacts = experiment.parents[3] / 'artifacts'
	frozen = json.loads((artifacts / 'canonical.json').read_text())['cells']
	for item in frozen:
		item.pop('survey')
	(experiment / 'execution.yaml').unlink()
	cells, _evidence = summary._audited_arm(
		experiment, 'parihaka', cell, frozen_control_cells=frozen
	)
	assert len(cells) == 15
	assert len(calls) == 30
	frozen[0]['primary'] = 0.1
	with pytest.raises(ValueError, match='canonical evidence'):
		summary._audited_arm(experiment, 'parihaka', cell, frozen_control_cells=frozen)


@pytest.mark.parametrize(
	'change',
	[
		'incomplete',
		'canonical_metric',
		'nonfinite',
		'checkpoint',
		'shared_decoder',
		'extra_layout',
	],
)
def test_arm_rejects_missing_or_drifting_evidence(paired_arm, monkeypatch, change):
	experiment, cell, configs, lineages, _calls = paired_arm
	raw = configs[cell['downstream_config']]
	model = cell['candidate_id']
	metrics = (
		Path(raw['outputs']['runs_root'])
		/ f'model={model}/layout=layout_000/size=small/metrics.json'
	)
	if change == 'incomplete':
		metrics.unlink()
	elif change == 'canonical_metric':
		baseline = configs[cell['baseline_downstream_config']]
		baseline_id = lineages[True]['model_tag']
		path = (
			Path(baseline['outputs']['runs_root'])
			/ f'model={baseline_id}/layout=layout_000/size=small/metrics.json'
		)
		_json(path, {'test': {'channel_iou': 0.4, 'channel_f1': 0.6}})
	elif change == 'nonfinite':
		_json(metrics, {'test': {'channel_iou': float('nan'), 'channel_f1': 0.6}})
	elif change == 'checkpoint':
		lineages[False]['checkpoint']['sha256'] = 'b' * 64
	elif change == 'shared_decoder':
		monkeypatch.setattr(
			summary,
			'_audit_cell',
			lambda _survey, _raw, model, _layout, _size: {
				'benchmark_identity': {
					'model': model,
					'embedding': model,
					'decoder': model,
				}
			},
		)
	else:
		(Path(raw['outputs']['runs_root']) / f'model={model}/layout=layout_999').mkdir()
	with pytest.raises((ValueError, FileNotFoundError)):
		summary._audited_arm(experiment, 'parihaka', cell)


@pytest.mark.parametrize('survey', ['f3', 'parihaka', 'volve'])
def test_primary_improvement_direction_stays_within_survey(survey):
	cells = [
		{
			'layout_id': f'layout_{i:03d}',
			'data_size': 'small',
			'candidate_id': 'v3',
			'baseline_id': 'v02',
			'primary': 0.4,
			'control_primary': 0.5,
			'secondary': 0.6,
			'control_secondary': 0.6,
		}
		for i in range(5)
	]
	row = summary.comparison_row(survey, 'mae', 'small', cells)
	assert row['primary_metric'] == METRICS[survey]['primary_metric']
	assert row['primary_improvement'] == pytest.approx(
		0.1 if survey == 'volve' else -0.1
	)
	assert row['wins'] == (5 if survey == 'volve' else 0)


@pytest.fixture(params=['f3', 'parihaka', 'volve'])
def publication(paired_arm, monkeypatch, request):
	experiment, cell, _configs, _lineages, _calls = paired_arm
	selected, evidence = summary._audited_arm(experiment, 'parihaka', cell)
	definition = {
		'survey': request.param,
		'summary_root': str(experiment / 'output'),
		'canonical_selection': {'path': 'selection.yaml', 'sha256': 'a' * 64},
		'cells': [
			{**cell, 'source': family, 'candidate_id': CANDIDATE_IDS[family]}
			for family in SOURCE_FAMILIES
		],
	}
	monkeypatch.setattr(
		summary, 'validate_trace_mask_definition', lambda _experiment: definition
	)
	monkeypatch.setattr(summary, 'validate_canonical_evidence', lambda *_args: None)

	def arm(_experiment, _survey, entry):
		return [
			{
				**row,
				'source_family': entry['source'],
				'candidate_id': entry['candidate_id'],
			}
			for row in selected
		], copy.deepcopy(evidence)

	monkeypatch.setattr(summary, '_audited_arm', arm)
	return experiment, definition


def test_summary_exact_file_set_complete_cells_and_audited_idempotence(publication):
	experiment, definition = publication
	paths = summary.summarize(experiment)
	assert {path.name for path in paths} == {
		'comparison.csv',
		'summary.json',
		'summary.md',
	}
	assert set(Path(definition['summary_root']).iterdir()) == set(paths)
	payload = json.loads(
		(Path(definition['summary_root']) / 'summary.json').read_text()
	)
	assert payload['status'] == 'complete'
	assert len(payload['cells']) == 45
	assert len(payload['rows']) == 9
	assert len(payload['evidence']) == 3
	assert payload['primary_metric'] == METRICS[definition['survey']]['primary_metric']
	assert all(arm['layout_clustered']['cluster_count'] == 5 for arm in payload['arms'])
	assert summary.summarize(experiment) == paths


@pytest.mark.parametrize(
	'name', ['comparison.csv', 'summary.json', 'summary.md', 'unexpected.pt']
)
def test_summary_reuse_rejects_any_changed_publication_file(publication, name):
	experiment, definition = publication
	summary.summarize(experiment)
	(Path(definition['summary_root']) / name).write_text('changed', encoding='utf-8')
	with pytest.raises(ValueError, match='publication'):
		summary.summarize(experiment)


def test_summary_failed_provenance_publishes_nothing(publication, monkeypatch):
	experiment, definition = publication

	def fail(*_args):
		raise ValueError('incomplete paired provenance')

	monkeypatch.setattr(summary, '_audited_arm', fail)
	with pytest.raises(ValueError, match='incomplete'):
		summary.summarize(experiment)
	assert not Path(definition['summary_root']).exists()


def test_summary_never_consumes_reports(tmp_path):
	root = tmp_path / 'reports/experiment'
	root.mkdir(parents=True)
	with pytest.raises(ValueError, match='reports'):
		summary.inspect_summary(root)
