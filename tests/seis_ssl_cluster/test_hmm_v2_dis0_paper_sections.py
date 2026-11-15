"""Zero-distillation source, fixed-section and review publication contracts."""

# ruff: noqa: SLF001

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.hmm.distillation_sensitivity import validate_matrix
from seis_ssl_cluster.visualization import hmm_v2_paper_sections as subject
from tests.seis_ssl_cluster.test_hmm_v2_paper_sections import _reference

MATRIX = Path(
	'experiments/hmm_v2/zero_distillation_paper_sections_v1/barlow_sources.yaml'
)


def test_barlow_sources_inherit_their_own_v2_recipes():
	matrix = validate_matrix(MATRIX)
	assert matrix['source'] == 'local_bt'
	assert matrix['new_downstream_cells'] == 3
	for survey, arms in matrix['surveys'].items():
		assert 'local_bt_rot90_asym_g060_3ep' in arms['0.0']['candidate_id']
		assert arms['0.0']['candidate_id'].endswith('distill000_cons010')
		assert arms['0.2']['execution'] == 'reuse'
		configs = arms['0.0']['configs']
		assert set(configs) == {'training', 'smoke', 'embeddings', 'downstream'}
		root = Path(configs['training']['path']).parents[2]
		assert len(list(root.rglob('*.yaml'))) == 4, survey
		if survey == 'volve':
			downstream = yaml.safe_load(Path(configs['downstream']['path']).read_text())
			assert downstream['arm']['multi_head_training_config'] == (
				'${SEIS_SSL_CLUSTER_WORKSPACE}/' + configs['training']['path']
			)


@pytest.mark.parametrize('survey', subject.SURVEYS)
def test_dis0_reuses_every_original_section_and_has_separate_outputs(survey):
	reference = _reference(survey)
	before = copy.deepcopy(reference)
	old = subject.fixed_rows(reference, survey)
	new = subject.fixed_rows(reference, survey, treatment='v2_dis0')
	assert reference == before
	assert len(new) == (20 if survey == 'volve' else 30)
	for a, b in zip(old, new, strict=True):
		assert b == dict(
			a, output_png=a['output_png'].replace('_hmm_v2_', '_hmm_v2_dis0_')
		)
	assert len(subject.expected_figure_files(new)) == len(new) + 2


@pytest.mark.parametrize(
	'change',
	[{'distillation_weight': 0.2}, {'head_ks': [4, 6, 8]}, {'consistency_weight': 0.0}],
)
def test_dis0_requires_actual_zero_weight_metadata(change):
	pretext = {
		'head_spec': 'multi_resolution_ordered_prototypes_v1',
		'head_ks': [6, 8, 10],
		'distillation_weight': 0.0,
		'consistency_weight': 0.1,
		'method': 'strat_hmm_multi_head_pretext',
	}
	subject.validate_pretext({'stratigraphy_pretext': pretext}, 'v2_dis0')
	with pytest.raises(ValueError, match='canonical v2_dis0'):
		subject.validate_pretext(
			{'stratigraphy_pretext': dict(pretext, **change)}, 'v2_dis0'
		)
	with pytest.raises(ValueError, match='canonical v2'):
		subject.validate_pretext({'stratigraphy_pretext': pretext}, 'v2')


@pytest.mark.parametrize('family', ['mae', 'local_bt', 'random'])
def test_dis0_source_must_match_completed_study_and_selected_cell(tmp_path, family):
	model = {
		'source_family': family,
		'treatment': 'v2_dis0',
		'model_id': 'candidate',
		'metrics': {'path': str(tmp_path / 'metrics.json'), 'sha256': 'correct'},
	}
	record = {
		'survey': 'f3',
		'weight': '0.0',
		'layout': 'layout_000',
		'size': 'large',
		'candidate_id': 'candidate',
		'metrics_paths': [model['metrics']['path']],
		'metrics_sha256': ['correct'],
	}
	path = tmp_path / 'study.json'
	config = {'inputs': {f'{family}_sources': {'path': str(path)}}}
	path.write_text(json.dumps({'status': 'complete', 'evidence': [record]}))
	subject._bind_dis0_source(config, model, 'f3')
	for delta in [{'metrics_sha256': ['wrong']}, {'weight': '0.2'}, {'size': 'small'}]:
		path.write_text(
			json.dumps({'status': 'complete', 'evidence': [dict(record, **delta)]})
		)
		with pytest.raises(ValueError, match='not bound'):
			subject._bind_dis0_source(config, model, 'f3')
	path.write_text(json.dumps({'status': 'running', 'evidence': [record]}))
	with pytest.raises(ValueError, match='incomplete'):
		subject._bind_dis0_source(config, model, 'f3')


def test_parent_and_reused_prediction_provenance_are_preserved(tmp_path):
	model = {'condition_id': 'mae_none', 'source_family': 'mae', 'treatment': 'none'}
	baseline = {'surveys': {'parihaka': {'models': [model]}}}
	meta = tmp_path / 'prediction_metadata.json'
	meta.write_text('{}')
	reference = {
		'models': [
			dict(
				model,
				render_prediction_metadata={
					'path': str(meta),
					'sha256': file_sha256(meta),
				},
			)
		]
	}
	selected = subject._dis0_model(model, 'parihaka', {}, baseline, reference)
	assert selected['reuse_prediction_dir'] == str(tmp_path)
	assert 'reuse_prediction_dir' not in model
	with pytest.raises(ValueError, match='parent differs'):
		subject._dis0_model(
			dict(model, model_id='other'), 'parihaka', {}, baseline, reference
		)
	meta.write_text('changed')
	with pytest.raises(ValueError, match='hash drift'):
		subject._dis0_model(model, 'parihaka', {}, baseline, reference)


def test_manifest_records_dis0_and_preserves_original_reference(tmp_path):
	reference = _reference('f3')
	before = copy.deepcopy(reference)
	record = {'path': '/artifact/source.json', 'sha256': 'bound'}
	plan = {
		'config': {
			'experiment_id': subject.DIS0_EXPERIMENT,
			'output_root': str(tmp_path),
			'inputs': dict.fromkeys(
				(
					'canonical_selection',
					'mae_sources',
					'local_bt_sources',
					'random_sources',
				),
				record,
			),
			'surveys': {'f3': {'reference_manifest': record, 'v2_manifest': record}},
		},
		'config_path': '/experiment/execution.yaml',
		'config_sha256': 'execution',
		'surveys': {
			'f3': {
				'reference': reference,
				'sources': [],
				'rows': subject.fixed_rows(reference, 'f3', treatment='v2_dis0'),
			}
		},
	}
	manifest = subject._manifest(plan, 'f3')
	assert manifest['comparison']['distillation_weight'] == 0.0
	assert manifest['comparison']['consistency_weight'] == 0.1
	assert manifest['hmm_version'] == 'HMM_v2_dis0'
	assert set(manifest['source_studies']) == {'mae', 'local_bt', 'random'}
	assert manifest['v2_reference_manifest'] == record
	assert not manifest['training']
	assert not manifest['metric_evaluation']
	assert reference == before


def test_execute_dis0_renders_all_surveys_then_publishes_review(tmp_path, monkeypatch):
	plan = {
		'config': {
			'experiment_id': subject.DIS0_EXPERIMENT,
			'output_root': str(tmp_path / 'artifacts'),
			'review_root': str(tmp_path / 'reports'),
		},
		'surveys': {s: {'rows': []} for s in subject.SURVEYS},
	}
	events = []
	monkeypatch.setattr(subject, 'inspect_plan', lambda _path: plan)
	monkeypatch.setattr(
		subject, 'render_survey', lambda _plan, survey: events.append(survey)
	)
	monkeypatch.setattr(
		subject, 'publish_dis0_review', lambda _plan, _root: events.append('review')
	)
	monkeypatch.setattr(
		sys, 'argv', ['visualize', '--config', 'unused.yaml', '--execute']
	)
	subject.main()
	assert events == [*subject.SURVEYS, 'review']


def test_review_publishes_only_index_and_links_all_eighty_figures(tmp_path):
	plan = {
		'config': {
			'experiment_id': subject.DIS0_EXPERIMENT,
			'output_root': str(tmp_path / 'artifacts'),
		},
		'surveys': {},
	}
	for survey in subject.SURVEYS:
		rows = subject.fixed_rows(_reference(survey), survey, treatment='v2_dis0')
		plan['surveys'][survey] = {'rows': rows}
		folder = (
			Path(plan['config']['output_root'])
			/ 'figures'
			/ survey
			/ 'layout_000/large'
		)
		folder.mkdir(parents=True)
		(folder / 'manifest.json').write_text('{}')
		(folder / 'selected_slices.csv').write_text('fixture')
		for row in rows:
			path = folder / row['output_png']
			path.parent.mkdir(exist_ok=True)
			Image.new('RGB', (2125, 1240)).save(path)
	root = tmp_path / 'reports/hmm_v2_dis0_paper_sections_v1'
	readme = subject.publish_dis0_review(plan, root)
	assert (
		{p.name for p in root.iterdir()} == subject.DIS0_REVIEW_FILES == {'README.md'}
	)
	assert readme.read_text().count('.png)') == 80
	assert 'layout_000/large' in readme.read_text()
	with pytest.raises(FileExistsError, match='refusing'):
		subject.publish_dis0_review(plan, root)
	# Extra files must prevent publishing even a review index.
	(folder / 'extra.npy').write_bytes(b'not a publication')
	with pytest.raises(ValueError, match='publication files'):
		subject.publish_dis0_review(plan, tmp_path / 'reports/invalid')
	assert not (tmp_path / 'reports/invalid').exists()
