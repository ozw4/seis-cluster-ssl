from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.visualization import hmm_v2_paper_sections as subject


def _reference(survey='f3'):
	domains = ('inline', 'crossline') if survey == 'volve' else subject.facies.DOMAINS
	rows = [
		{
			'domain': d,
			'array_index': i,
			'actual_coordinate': i + 100,
			'active_training_line': 'not_applicable' if d == 'time' else False,
			'output_png': f'{d}/{survey}_hmm_v1_layout_000_large_{d}_{i}.png',
		}
		for d in domains
		for i in range(10)
	]
	return (
		{
			'selection': {
				f'selected_{d}_slices': [r for r in rows if r['domain'] == d]
				for d in domains
			}
		}
		if survey == 'f3'
		else {'selected_slices': rows}
	)


def test_canonical_pretext_is_verified_from_metadata():
	pretext = {
		'head_spec': 'multi_resolution_ordered_prototypes_v1',
		'head_ks': [6, 8, 10],
		'distillation_weight': 0.2,
		'consistency_weight': 0.1,
		'method': 'strat_hmm_multi_head_pretext',
	}
	subject.validate_pretext({'stratigraphy_pretext': pretext}, 'v2')
	subject.validate_pretext({}, 'none')
	for changed in [
		{'head_ks': [4, 6, 8]},
		{'consistency_weight': 0},
		{'distillation_weight': 0.1},
		{'head_spec': 'single_head'},
	]:
		with pytest.raises(ValueError, match='canonical v2'):
			subject.validate_pretext(
				{'stratigraphy_pretext': dict(pretext, **changed)}, 'v2'
			)
	with pytest.raises(ValueError, match='non-HMM'):
		subject.validate_pretext({'stratigraphy_pretext': pretext}, 'none')


@pytest.mark.parametrize(
	('survey', 'count'), [('f3', 30), ('parihaka', 30), ('volve', 20)]
)
def test_sections_and_coordinates_are_fixed_not_reselected(survey, count):
	reference = _reference(survey)
	before = copy.deepcopy(reference)
	rows = subject.fixed_rows(reference, survey)
	assert len(rows) == count
	assert reference == before
	assert all('_hmm_v2_' in r['output_png'] for r in rows)
	assert [r['array_index'] for r in rows] == list(range(10)) * (count // 10)
	assert len(subject.expected_figure_files(rows)) == count + 2


def test_reference_rejects_training_sections_duplicates_and_unsafe_paths():
	for change in [
		{'active_training_line': True},
		{'output_png': '../escape.png'},
		{'domain': 'crossline'},
	]:
		ref = _reference()
		ref['selection']['selected_inline_slices'][0].update(change)
		with pytest.raises(ValueError, match=r'reference|unsafe'):
			subject.fixed_rows(ref, 'f3')


def test_source_hash_drift_and_report_inputs_are_rejected(tmp_path):
	path = tmp_path / 'source.json'
	path.write_text('{}')
	bound = {'path': str(path), 'sha256': file_sha256(path)}
	assert subject.verify_bound(bound) == path
	path.write_text('changed')
	with pytest.raises(ValueError, match='hash drift'):
		subject.verify_bound(bound)
	report = tmp_path / 'reports/source.json'
	report.parent.mkdir()
	report.write_text('{}')
	with pytest.raises(ValueError, match='non-report'):
		subject.verify_bound({'path': str(report), 'sha256': file_sha256(report)})


def test_atomic_render_has_exact_32_file_publication_and_no_overwrite(
	tmp_path, monkeypatch
):
	rows = subject.fixed_rows(_reference(), 'f3')
	manifest = {'training': False, 'metric_evaluation': False}
	plan = {
		'config': {'output_root': str(tmp_path / 'out')},
		'surveys': {'f3': {'rows': rows}},
	}
	monkeypatch.setattr(subject, '_predictions', lambda *_a: [])
	monkeypatch.setattr(
		subject, '_facies_prepared', lambda *_a: SimpleNamespace(manifest=manifest)
	)

	class Figure:
		def savefig(self, path, **_kwargs):
			Image.new('RGB', (2125, 1240), color='white').save(path)

	monkeypatch.setattr(subject.facies, 'render_figure', lambda *_a: Figure())
	monkeypatch.setattr(subject.plt, 'close', lambda *_a: None)
	root = subject.render_survey(plan, 'f3')
	subject.verify_publication(root, rows)
	assert len([p for p in root.rglob('*') if p.is_file()]) == 32
	assert json.loads((root / 'manifest.json').read_text())[
		'published_files'
	] == sorted(subject.expected_figure_files(rows))
	with pytest.raises(FileExistsError, match='refusing'):
		subject.render_survey(plan, 'f3')
	(root / 'extra.txt').write_text('unexpected')
	with pytest.raises(ValueError, match='publication files'):
		subject.verify_publication(root, rows)


def test_failed_render_does_not_publish_a_partial_survey(tmp_path, monkeypatch):
	plan = {
		'config': {'output_root': str(tmp_path / 'out')},
		'surveys': {'f3': {'rows': subject.fixed_rows(_reference(), 'f3')}},
	}
	monkeypatch.setattr(subject, '_predictions', lambda *_a: [])
	monkeypatch.setattr(
		subject, '_facies_prepared', lambda *_a: SimpleNamespace(manifest={})
	)

	def fail(*_args):
		raise RuntimeError('render failed')

	monkeypatch.setattr(subject.facies, 'render_figure', fail)
	with pytest.raises(RuntimeError, match='render failed'):
		subject.render_survey(plan, 'f3')
	assert not (tmp_path / 'out/figures/f3/layout_000/large').exists()


def test_cli_defaults_to_preflight_without_writes(monkeypatch):
	plan = {
		'config': {'output_root': '/unused'},
		'surveys': {
			s: {'rows': subject.fixed_rows(_reference(s), s)} for s in subject.SURVEYS
		},
	}
	monkeypatch.setattr(subject, 'inspect_plan', lambda *_a: plan)
	monkeypatch.setattr(sys, 'argv', ['visualize', '--config', 'unused.yaml'])

	def fail(*_args):
		raise AssertionError('dry-run attempted rendering')

	monkeypatch.setattr(subject, 'render_survey', fail)
	subject.main()
