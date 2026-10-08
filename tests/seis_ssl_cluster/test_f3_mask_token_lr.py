"""Six mask-only arms, inherited decoder contracts and exact publication."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster.hmm import mask_token_lr as sweep
from seis_ssl_cluster.hmm.multi_source_receipts import CELL_IDENTITIES
from seis_ssl_cluster.hmm.trace_mask_definition import definition_path, read_yaml

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'experiments/f3/facies_benchmark_v2' / sweep.EXPERIMENT


def _args(**values):
	return argparse.Namespace(first='smoke', last='summary', candidate=None, **values)


def test_six_independent_arms_and_exact_90_cell_plan():
	definition = sweep.validate_definition(ROOT)
	assert len(list(ROOT.rglob('*.yaml'))) == 31
	plan = sweep.command_plan(ROOT, _args())
	assert len(plan) == 115
	assert [sum(stage == name for stage, _ in plan) for name in sweep.STAGES] == [
		6,
		6,
		6,
		6,
		90,
		1,
	]
	for cell in definition['cells']:
		full = read_yaml(definition_path(cell['train_full_config'], ROOT))
		control = read_yaml(definition_path(cell['baseline_train_config'], ROOT))
		assert full['train']['epochs'] == 25
		assert full['train']['lr'] == full['train']['encoder_lr'] == 1.0e-5
		assert full['loss']['consistency_weight'] == 0.1
		assert full['teacher'] == control['teacher']
		assert full['student'] == control['student']
		assert full['pseudo_targets'] == control['pseudo_targets']
		assert full['train']['mask_token_lr'] == cell['mask_token_lr']
		args = argparse.Namespace(
			first='smoke', last='downstream', candidate=cell['candidate_id']
		)
		selected = sweep.command_plan(ROOT, args)
		assert len(selected) == 19
		assert sum(stage == 'downstream' for stage, _ in selected) == 15
		assert 'summary' not in {stage for stage, _ in selected}


@pytest.mark.parametrize(
	('first', 'last', 'candidate'),
	[
		('full', 'smoke', None),
		('smoke', 'full', 'unknown'),
		('smoke', 'summary', next(iter(sweep.CENTER_TRACE_MASK_LR_TAGS))),
	],
)
def test_invalid_or_partial_summary_plan_is_rejected(first, last, candidate):
	with pytest.raises(ValueError, match=r'reversed|unknown|all six'):
		sweep.command_plan(
			ROOT, argparse.Namespace(first=first, last=last, candidate=candidate)
		)


@pytest.fixture
def copied_definition(tmp_path, monkeypatch):
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(tmp_path))
	destination = tmp_path / ROOT.relative_to(WORKSPACE)
	shutil.copytree(ROOT, destination)
	control_root = ROOT.parent / sweep.CONTROL
	shutil.copytree(control_root, tmp_path / control_root.relative_to(WORKSPACE))
	control = read_yaml(control_root / 'execution.yaml')
	paths = [definition_path(control['canonical_selection']['path'], control_root)]
	for cell in control['cells']:
		paths.extend(
			definition_path(cell[field], control_root)
			for field in (
				'baseline_train_config',
				'baseline_embedding_config',
				'baseline_downstream_config',
			)
		)
	for path in paths:
		target = tmp_path / path.relative_to(WORKSPACE)
		target.parent.mkdir(parents=True, exist_ok=True)
		shutil.copyfile(path, target)
	return destination


@pytest.mark.parametrize(
	('section', 'key', 'value'),
	[
		('train', 'lr', 3.0e-5),
		('train', 'encoder_lr', 3.0e-5),
		('train', 'mask_token_lr', 5.0e-5),
		('train', 'seed', 99),
		('train', 'epochs', 24),
		('loss', 'consistency_weight', 0.0),
		('student', 'unfreeze_top_blocks', 2),
	],
)
def test_only_mask_token_rate_may_change(copied_definition, section, key, value):
	definition = sweep.validate_definition(copied_definition)
	path = definition_path(
		definition['cells'][0]['train_full_config'], copied_definition
	)
	raw = read_yaml(path)
	raw[section][key] = value
	path.write_text(yaml.safe_dump(raw), encoding='utf-8')
	with pytest.raises(ValueError, match='other than mask-token LR'):
		sweep.validate_definition(copied_definition)


@pytest.mark.parametrize(
	('field', 'section', 'key'),
	[
		('train_smoke_config', 'train', 'seed'),
		('embedding_config', 'embeddings', 'checkpoint'),
		('downstream_config', 'outputs', 'runs_root'),
	],
)
def test_stage_handoff_drift_is_rejected(copied_definition, field, section, key):
	definition = sweep.validate_definition(copied_definition)
	path = definition_path(definition['cells'][0][field], copied_definition)
	raw = read_yaml(path)
	raw[section][key] = 'unexpected'
	path.write_text(yaml.safe_dump(raw), encoding='utf-8')
	with pytest.raises(ValueError, match=r'inheritance|namespace'):
		sweep.validate_definition(copied_definition)


@pytest.mark.parametrize('change', ['matrix', 'rate', 'target', 'control'])
def test_definition_requires_exact_matrix_and_frozen_control(copied_definition, change):
	path = copied_definition / 'execution.yaml'
	raw = read_yaml(path)
	if change == 'matrix':
		raw['cells'].pop()
	elif change == 'rate':
		raw['cells'][0]['mask_token_lr'] = 1.0e-5
	elif change == 'target':
		raw['cells'][0]['target_manifest_sha256'] = '0' * 64
	else:
		raw['control_definition_sha256'] = '0' * 64
	path.write_text(yaml.safe_dump(raw), encoding='utf-8')
	with pytest.raises(ValueError, match=r'matrix|rate|target|control'):
		sweep.validate_definition(copied_definition)


@pytest.mark.parametrize(
	'change', ['summary_hash', 'status', 'count', 'checkpoint', 'manifest']
)
def test_live_control_requires_completed_pinned_evidence(tmp_path, monkeypatch, change):
	definition = copy.deepcopy(sweep.validate_definition(ROOT))
	definition['control_summary_sha256'] = 'summary'
	for cell in definition['cells']:
		cell['baseline_checkpoint_sha256'] = 'checkpoint'
		cell['target_manifest_sha256'] = 'manifest'
	completed = {'status': 'complete', 'cells': list(range(45))}
	digests = {
		'summary.json': 'summary',
		'latest.pt': 'checkpoint',
		'manifest.json': 'manifest',
	}
	monkeypatch.setattr(
		sweep,
		'validate_trace_mask_definition',
		lambda _root: {'summary_root': str(tmp_path)},
	)
	monkeypatch.setattr(sweep, 'validate_canonical_evidence', lambda *_args: None)
	monkeypatch.setattr(
		sweep,
		'read_yaml',
		lambda _path: {
			'paths': {'output_root': str(tmp_path / 'training')},
			'pseudo_targets': {'manifest': str(tmp_path / 'manifest.json')},
		},
	)
	monkeypatch.setattr(sweep, 'file_sha256', lambda path: digests[path.name])
	monkeypatch.setattr(sweep, 'read_json', lambda _path: completed)
	assert sweep.validate_live(ROOT, definition) is completed
	if change == 'summary_hash':
		digests['summary.json'] = 'drift'
	elif change == 'status':
		completed['status'] = 'running'
	elif change == 'count':
		completed['cells'].pop()
	elif change == 'checkpoint':
		digests['latest.pt'] = 'drift'
	else:
		digests['manifest.json'] = 'drift'
	with pytest.raises(ValueError, match=r'hash drift|45 completed'):
		sweep.validate_live(ROOT, definition)


@pytest.fixture
def complete_summary(monkeypatch):
	definition = sweep.validate_definition(ROOT)
	control = {'cells': [{'frozen_masked_v3': True}]}
	monkeypatch.setattr(sweep, 'validate_live', lambda *_args: control)
	calls = []

	def audited(_experiment, survey, cell, *, frozen_control_cells):
		assert survey == 'f3'
		assert frozen_control_cells is control['cells']
		calls.append(cell['candidate_id'])
		baseline = cell['candidate_id'].rsplit('_masklr', 1)[0]
		return [
			{
				'candidate_id': cell['candidate_id'],
				'baseline_id': baseline,
				'layout_id': layout,
				'data_size': size,
				'primary': 0.55,
				'control_primary': 0.5,
				'secondary': 0.6,
				'control_secondary': 0.58,
			}
			for layout, size in CELL_IDENTITIES
		], {'audited': True}

	monkeypatch.setattr(sweep, '_audited_arm', audited)
	return definition, calls


def test_summary_uses_masked_v3_control_and_layout_clustered_statistics(
	complete_summary,
):
	_definition, calls = complete_summary
	payload = sweep.inspect_summary(ROOT)
	assert len(calls) == 6
	assert len(payload['cells']) == 90
	assert len(payload['rows']) == 18
	assert payload['control_mask_token_lr'] == 1.0e-5
	assert all(
		row['primary_improvement'] == pytest.approx(0.05) for row in payload['rows']
	)
	assert all(arm['layout_clustered']['cluster_count'] == 5 for arm in payload['arms'])
	assert all(arm['layout_clustered']['sample_sd'] == 0 for arm in payload['arms'])


def test_exact_publication_is_idempotent_and_rejects_extra_files(
	tmp_path, monkeypatch, complete_summary
):
	_definition, _calls = complete_summary
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	paths = sweep.summarize(ROOT)
	assert {path.name for path in paths} == set(sweep.PUBLISHED_FILES)
	assert {path.name for path in paths[0].parent.iterdir()} == set(
		sweep.PUBLISHED_FILES
	)
	assert sweep.summarize(ROOT) == paths
	assert (
		len(json.loads((paths[0].parent / 'summary.json').read_text())['cells']) == 90
	)
	(paths[0].parent / 'unowned.txt').write_text('extra', encoding='utf-8')
	with pytest.raises(ValueError, match='publication drift'):
		sweep.summarize(ROOT)


def test_incomplete_results_do_not_publish(tmp_path, monkeypatch, complete_summary):
	definition, _calls = complete_summary
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))

	def missing(*_args, **_kwargs):
		raise FileNotFoundError('missing decoder metrics')

	monkeypatch.setattr(sweep, '_audited_arm', missing)
	with pytest.raises(FileNotFoundError):
		sweep.summarize(ROOT)
	output = Path(
		definition['summary_root'].replace(
			'${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}', str(tmp_path)
		)
	)
	assert not output.exists()
