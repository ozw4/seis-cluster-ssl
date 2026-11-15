"""Missing controls retain fixed cohorts, paired units and publication ownership."""

# ruff: noqa: SLF001

import json
from pathlib import Path

import pytest

from seis_ssl_cluster.f3.lithology import common_test
from seis_ssl_cluster.hmm import missing_controls as subject


def test_test_extensions_have_exact_prespecified_scope_and_require_fixed_mask():
	for namespace, expected_count in (
		(common_test.V1_ZERO_DISTILLATION_NAMESPACE, 3),
		(common_test.BARLOW_ZERO_DISTILLATION_NAMESPACE, 2),
	):
		conditions = common_test._expected_conditions(namespace)
		assert len(conditions) == expected_count
		config = {
			'experiment_id': namespace,
			'expected_conditions': expected_count,
			'expected_cells': expected_count * 15,
			'inputs': {},
		}
		with pytest.raises(ValueError, match='reference test mask'):
			common_test._scope(config)
		config['inputs']['reference_test_mask'] = {
			'path': 'fixed.npy',
			'sha256': 'frozen',
		}
		assert common_test._scope(config)[1] == conditions
	assert all(
		c[3] == '0.0'
		for c in common_test._expected_conditions(
			common_test.V1_ZERO_DISTILLATION_NAMESPACE
		)
	)
	assert {
		c[3]
		for c in common_test._expected_conditions(
			common_test.BARLOW_ZERO_DISTILLATION_NAMESPACE
		)
	} == {'0.0', '0.2'}
	with pytest.raises(ValueError, match='unsupported'):
		common_test._expected_conditions('unbounded_dynamic_test')


def test_means_reject_missing_and_duplicate_layouts_and_use_sample_sd():
	rows = [
		{'survey': 'f3', 'layout_id': layout, 'primary': i, 'secondary': i / 2}
		for i, layout in enumerate(subject.LAYOUTS)
	]
	result = subject.means(rows, ('survey',))[0]
	assert result['primary_mean'] == 2
	assert result['primary_sample_sd'] == pytest.approx(1.5811388300841898)
	with pytest.raises(ValueError, match='exactly five'):
		subject.means(rows[:-1], ('survey',))
	with pytest.raises(ValueError, match='exactly five'):
		subject.means([*rows[:-1], rows[0]], ('survey',))
	rows[0]['primary'] = float('inf')
	with pytest.raises(ValueError, match='nonfinite'):
		subject.means(rows, ('survey',))


def test_paired_factorial_contrasts_preserve_metric_direction_and_interaction():
	# In the selected layout, zero distillation raises IoU by 2 without
	# consistency and by 5 with it. All five layouts have the same effect.
	cells = []
	for survey in ('f3', 'parihaka', 'volve'):
		for source in ('mae', 'local_bt', 'random'):
			for size in subject.SIZES:
				for i, layout in enumerate(subject.LAYOUTS):
					for (dist, cons), value in {
						(0.2, 0.0): 10,
						(0.0, 0.0): 12,
						(0.2, 0.1): 11,
						(0.0, 0.1): 16,
					}.items():
						cells.append(
							{
								'survey': survey,
								'source': source,
								'data_size': size,
								'layout_id': layout,
								'distillation_weight': dist,
								'consistency_weight': cons,
								'primary': i + value
								if survey != 'volve'
								else 100 - i - value,
							}
						)
	contrasts = subject.contrasts(cells)
	assert len(contrasts) == 135
	for row in contrasts:
		assert row['paired_effect_sample_sd'] == 0
		if row['contrast'] == 'remove_distillation_at_cons000':
			assert row['paired_effect_mean'] == 2
		elif row['contrast'] == 'remove_distillation_at_cons010':
			assert row['paired_effect_mean'] == 5
		elif row['contrast'] == 'interaction':
			assert row['paired_effect_mean'] == 3


def test_completion_plan_reuses_three_large_cells_and_never_resumes_another_cell(
	tmp_path, monkeypatch
):
	barlow = {
		'surveys': {
			s: {
				'0.0': {
					'candidate_id': 'fixed_candidate',
					'configs': {'downstream': {'path': s}},
				}
			}
			for s in ('f3', 'parihaka', 'volve')
		}
	}
	monkeypatch.setattr(subject, 'load_config', lambda _: barlow)

	def metric_path(survey, _arm, layout, size):
		return (
			tmp_path
			/ survey
			/ layout
			/ size
			/ ('evaluation/metrics.json' if survey == 'f3' else 'metrics.json')
		)

	monkeypatch.setattr(subject, '_cell_path', metric_path)
	for survey in ('f3', 'parihaka', 'volve'):
		p = metric_path(survey, {}, 'layout_000', 'large')
		p.parent.mkdir(parents=True)
		p.write_text('{}')
	p = (
		metric_path('f3', {}, 'layout_000', 'medium').parent.parent
		/ 'decoder/latest.pt'
	)
	p.parent.mkdir(parents=True)
	p.write_text('owned')
	plan = subject.plan_decoders({'inputs': {'barlow_matrix': {'path': 'barlow'}}})
	assert len(plan) == 45
	assert sum(not r['complete'] for r in plan) == 42
	assert all(r['size'] == 'medium' for r in plan[:15])
	first = plan[0]['command']
	assert first[-2:] == ['--resume', str(p)]
	assert '--resume' not in plan[1]['command']


def test_exact_summary_set_and_immutable_republication(tmp_path, monkeypatch):
	monkeypatch.setattr(subject, 'study_root', lambda: tmp_path)
	definition = tmp_path / 'definition.yaml'
	definition.write_text('frozen')
	monkeypatch.setattr(subject, 'MATRIX', definition)
	monkeypatch.setattr(subject, 'prepare_test', lambda _: definition)
	names = (
		'barlow_cells',
		'barlow_comparison',
		'v1_cells',
		'v1_comparison',
		'factorial_cells',
		'factorial_comparison',
		'factorial_contrasts',
	)
	row = {
		'survey': 'f3',
		'data_size': 'large',
		'distillation_weight': 0,
		'primary_mean': 0.5,
		'primary_sample_sd': 0.1,
	}
	monkeypatch.setattr(subject, 'collect_results', lambda _: {n: [row] for n in names})
	subject.publish({'inputs': {}})
	output = tmp_path / 'summary'
	assert {p.name for p in output.iterdir()} == subject.PUBLISHED_FILES
	assert json.loads((output / 'summary.json').read_text())['factorial_cells'] == 540
	before = {p.name: p.read_bytes() for p in output.iterdir()}
	subject.publish({'inputs': {}})
	assert before == {p.name: p.read_bytes() for p in output.iterdir()}
	(output / 'unowned.csv').write_text('unsafe publication')
	with pytest.raises(ValueError, match='incomplete prior'):
		subject.publish({'inputs': {}})


def test_complete_collection_uses_f3_test_values_and_requires_full_factorial(
	tmp_path, monkeypatch
):
	monkeypatch.setattr(subject, 'root', lambda: tmp_path)
	coords = [
		(s, l, z)
		for s in ('mae', 'local_bt', 'random')
		for l in subject.LAYOUTS
		for z in subject.SIZES
	]

	def test_row(source, layout, size, version, weight, cons, primary):  # noqa: PLR0913, PLR0917
		return {
			'source': source,
			'layout_id': layout,
			'data_size': size,
			'version': version,
			'head_ks': '6' if version == 'v1' else '6|8|10',
			'distillation_weight': str(weight),
			'consistency_weight': str(cons),
			'mean_iou': str(primary),
			'macro_f1': '0.8',
		}

	base = [
		test_row(s, l, z, 'v1', w, 0, 0.7) for s, l, z in coords for w in (0.1, 0.2)
	]
	base += [
		test_row(s, l, z, 'v2', 0.2, c, 0.75) for s, l, z in coords for c in (0.0, 0.1)
	]
	new_v1 = [test_row(s, l, z, 'v1', 0.0, 0, 0.9) for s, l, z in coords]
	new_bt = [
		test_row('local_bt', l, z, 'v2', w, 0.1, 0.75 if w == 0.2 else 0.85)
		for l in subject.LAYOUTS
		for z in subject.SIZES
		for w in (0.0, 0.2)
	]
	all_cells = [
		{
			'survey': v,
			'source': s,
			'layout_id': l,
			'data_size': z,
			'primary': 999 if v == 'f3' else 0.4,
			'secondary': 0.8,
			'weight': str(w),
		}
		for v in ('f3', 'parihaka', 'volve')
		for s, l, z in coords
		for w in (0.0, 0.1, 0.2)
	]
	tables = {'v1.csv': all_cells}
	prefix = tmp_path / 'surveys/f3/facies_benchmark_v2/evaluation'
	for namespace, rows in [
		('lithology_common_test_v1', base),
		(common_test.V1_ZERO_DISTILLATION_NAMESPACE, new_v1),
		(common_test.BARLOW_ZERO_DISTILLATION_NAMESPACE, new_bt),
	]:
		tables[str(prefix / namespace / 'summary/per_layout.csv')] = rows
	for name in (
		'k6810_multi_source_evaluation_v1',
		'k6810_cons010_multi_source_evaluation_v1',
	):
		p = tmp_path / 'studies/hmm_v2' / name / 'summary/summary.json'
		p.parent.mkdir(parents=True)
		p.write_text(
			json.dumps(
				{
					'cells': [
						{
							'survey': v,
							'source_family': s,
							'layout_id': l,
							'data_size': z,
							'primary': 999 if v == 'f3' else 0.4,
							'secondary': 0.8,
						}
						for v in ('f3', 'parihaka', 'volve')
						for s, l, z in coords
					]
				}
			)
		)
	zero = [
		{
			'survey': v,
			'source': s,
			'layout_id': l,
			'data_size': z,
			'primary': 0.5,
			'secondary': 0.8,
			'evaluation_split': 'common_test' if v == 'f3' else 'test',
		}
		for v in ('f3', 'parihaka', 'volve')
		for s, l, z in coords
	]
	tables[str(tmp_path / 'studies/hmm_v2/v2_dis0_con0_v1/summary/cells.csv')] = zero
	for name in ('distillation_sensitivity_v1', 'random_distillation_ablation_v1'):
		tables[str(tmp_path / 'studies/hmm_v2' / name / 'summary/cells.csv')] = [
			{
				'survey': v,
				'layout_id': l,
				'data_size': z,
				'primary': 0.6,
				'secondary': 0.8,
				'distillation_weight': '0.0',
			}
			for v in ('f3', 'parihaka', 'volve')
			for l in subject.LAYOUTS
			for z in subject.SIZES
		]

	def read(path):
		assert isinstance(path, Path), (
			'read_rows requires a Path, matching the real CSV reader'
		)
		return [dict(r) for r in tables[str(path)]]

	monkeypatch.setattr(subject, 'read_rows', read)
	monkeypatch.setattr(
		subject,
		'load_config',
		lambda _: {
			'surveys': {
				v: {w: {} for w in ('0.0', '0.2')} for v in ('f3', 'parihaka', 'volve')
			}
		},
	)
	monkeypatch.setattr(subject, '_cell_path', lambda *_: tmp_path)
	monkeypatch.setattr(subject, '_metrics', lambda *_: (999, 0.8, 'sha'))
	matrix = {
		'inputs': {'barlow_matrix': {'path': 'barlow'}, 'v1_cells': {'path': 'v1.csv'}}
	}
	result = subject.collect_results(matrix)
	assert len(result['factorial_cells']) == 540
	assert len(result['factorial_comparison']) == 108
	assert len(result['v1_cells']) == 405
	assert len(result['v1_comparison']) == 81
	assert len(result['barlow_cells']) == 90
	assert len(result['barlow_comparison']) == 18
	assert all(
		r['primary'] != 999 for r in result['factorial_cells'] if r['survey'] == 'f3'
	)
	assert all(
		r['primary'] == 0.9
		for r in result['v1_cells']
		if r['survey'] == 'f3' and r['distillation_weight'] == 0
	)
	new_bt[1]['mean_iou'] = '.76'
	with pytest.raises(ValueError, match='control changed'):
		subject.collect_results(matrix)


def test_review_uses_exact_copies_and_refuses_partial_or_report_sources(
	tmp_path, monkeypatch
):
	artifact = tmp_path / 'artifacts'
	summary = artifact / 'study/summary'
	summary.mkdir(parents=True)
	monkeypatch.setattr(subject, 'root', lambda: artifact)
	for name in subject.PUBLISHED_FILES:
		(summary / name).write_text(name + '\n')
	(summary / 'summary.json').write_text(
		json.dumps({'status': 'complete', 'factorial_cells': 540})
	)
	output = tmp_path / 'reports/review'
	subject.publish_review(summary, output)
	assert {p.name for p in output.iterdir()} == subject.REVIEW_FILES
	for name in subject.REVIEW_COPIES:
		assert (output / name).read_bytes() == (summary / name).read_bytes()
	provenance = json.loads((output / 'provenance.json').read_text())
	assert provenance['artifact_summary'] == 'study/summary'
	assert set(provenance['source_file_sha256']) == subject.REVIEW_COPIES
	with pytest.raises(FileExistsError):
		subject.publish_review(summary, output)
	(summary / 'v1_cells.csv').unlink()
	with pytest.raises(ValueError, match='complete artifact'):
		subject.publish_review(summary, tmp_path / 'another')
	with pytest.raises(ValueError, match='artifact originals'):
		subject.publish_review(output, tmp_path / 'unsafe')


def test_dry_run_still_validates_each_survey_after_all_cells_complete(monkeypatch):
	jobs = [
		{'survey': s, 'complete': True, 'command': ['cell', s]}
		for s in ('f3', 'parihaka', 'volve')
	]
	monkeypatch.setattr(subject, 'validate', dict)
	monkeypatch.setattr(subject, 'plan_decoders', lambda _: jobs)
	monkeypatch.setattr(subject, 'operations', lambda: Path('operations'))
	monkeypatch.setattr(subject, 'prepare_test', lambda _: Path('fixed-test.yaml'))
	commands = []
	monkeypatch.setattr(subject, '_run', lambda c, *_: commands.append(c))
	plans = []
	monkeypatch.setattr(subject, 'inspect_test_plan', plans.append)
	monkeypatch.setattr('sys.argv', ['completion', '--phase', 'dry-run'])
	subject.main()
	assert commands == [['cell', s, '--dry-run'] for s in ('f3', 'parihaka', 'volve')]
	assert plans == [Path('fixed-test.yaml')]
