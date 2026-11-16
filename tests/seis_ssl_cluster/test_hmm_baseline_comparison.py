"""Signed no-HMM comparisons and exact review publication contracts."""

import json
import math

import pytest

from seis_ssl_cluster.hmm.missing_controls import PUBLISHED_FILES, REVIEW_FILES
from seis_ssl_cluster.results import hmm_baseline_comparison as subject


def cell(survey, layout='layout_000', **overrides):
	return dict(
		survey=survey,
		source='mae',
		data_size='small',
		layout_id=layout,
		distillation_weight='0.0',
		consistency_weight='0.1',
		evaluation_split='common_test' if survey == 'f3' else 'test',
		primary=0.6,
		secondary=0.7,
		decoder_seed=42000,
		model='with_hmm',
		shared_identity='fixed',
		metrics_file='hmm/metrics.json',
		**overrides,
	)


def baseline(survey, layout='layout_000'):
	r = cell(survey, layout)
	r.update(
		primary=0.5,
		secondary=0.65,
		model='without_hmm',
		metrics_file='base/metrics.json',
	)
	return r


def test_signed_improvement_and_units_do_not_invert_horizon_accuracy():
	for survey in subject.METRICS:
		a, b = cell(survey), baseline(survey)
		if survey == 'volve':
			a['primary'], b['primary'] = 8, 10
		r = subject.pair([a], [b])[0]
		assert r['primary_improvement'] == pytest.approx(
			2 if survey == 'volve' else 0.1
		)
		assert r['secondary_improvement'] == pytest.approx(0.05)
		assert r['primary_unit'] == ('samples' if survey == 'volve' else 'fraction')
		assert r['with_hmm_primary'] == a['primary']
		assert r['without_hmm_primary'] == b['primary']


def test_pairing_rejects_missing_duplicate_seed_or_support_mismatches():
	a, b = cell('f3'), baseline('f3')
	with pytest.raises(ValueError, match='missing'):
		subject.pair([a], [])
	with pytest.raises(ValueError, match='duplicate'):
		subject.pair([a], [b, b])
	with pytest.raises(ValueError, match='seed'):
		subject.pair([a], [dict(b, decoder_seed=42001)])
	with pytest.raises(ValueError, match='identity'):
		subject.pair([a], [dict(b, shared_identity='other')])
	with pytest.raises(ValueError, match='identity'):
		subject.pair([a], [dict(b, evaluation_split='validation')])


def test_sd_is_computed_from_paired_differences_with_all_negative_rows_retained():
	a = []
	b = []
	for i, (before, after) in enumerate(
		zip([0.1, 0.2, 0.3, 0.4, 0.5], [0.3, 0.1, 0.5, 0.3, 0.7], strict=True)
	):
		layout = f'layout_{i:03d}'
		a.append(dict(cell('f3', layout), primary=after))
		b.append(dict(baseline('f3', layout), primary=before))
	r = subject.summarize(subject.pair(a, b))[0]
	assert r['primary_improvement_mean'] == pytest.approx(0.08)
	assert r['primary_improvement_sample_sd'] == pytest.approx(math.sqrt(0.108 / 4))
	assert r['hmm_wins'] == 3
	assert r['hmm_losses'] == 2
	with pytest.raises(ValueError, match='five distinct'):
		subject.summarize(subject.pair(a[:-1], b))


def test_actual_evaluation_metadata_distinguishes_no_hmm_from_a_parent_directory_name(
	tmp_path,
):
	p = tmp_path / 'metrics.json'
	data = {
		'model': 'mae_from_ssl_hmm_continuation',
		'benchmark_identity': {
			'training': {'seed': 42000},
			'embedding': {'model_source': {}},
		},
		'test': {'channel_iou': 0.4, 'channel_f1': 0.6},
	}
	p.write_text(json.dumps(data))
	assert subject.evaluation(p, 'parihaka', expected_hmm=False)['primary'] == 0.4
	data['benchmark_identity']['embedding']['model_source']['stratigraphy_pretext'] = {
		'head_ks': [6, 8, 10]
	}
	p.write_text(json.dumps(data))
	with pytest.raises(ValueError, match='HMM source'):
		subject.evaluation(p, 'parihaka', expected_hmm=False)
	data['benchmark_identity']['training']['seed'] = 42001
	p.write_text(json.dumps(data))
	with pytest.raises(ValueError, match='seed'):
		subject.evaluation(p, 'parihaka', expected_hmm=True)


def test_f3_uses_recorded_model_tag_and_requires_the_common_test_support(tmp_path):
	p = tmp_path / 'metrics.json'
	cfg = tmp_path / 'config.json'
	cfg.write_text(
		json.dumps(
			{
				'train': {'seed': 42000},
				'model': {'tag': 'mae'},
				'dataset': {},
				'decoder': {},
				'tiles': {},
			}
		)
	)
	(p.with_name('evidence.json')).write_text(
		json.dumps(
			{
				'resolved_decoder_config': {'path': str(cfg)},
				'condition': {'version': 'none', 'historical_condition_id': '1'},
				'test_mask_sha256': 'mask',
				'supervision_grid': {'sha256': 'grid'},
			}
		)
	)
	data = {
		'split': 'test',
		'evaluation_voxel_count': 113979624,
		'mean_iou': 0.5,
		'macro_f1': 0.6,
	}
	p.write_text(json.dumps(data))
	assert subject.evaluation(p, 'f3', expected_hmm=False)['model'] == 'mae'
	data['split'] = 'validation'
	p.write_text(json.dumps(data))
	with pytest.raises(ValueError, match='support mismatch'):
		subject.evaluation(p, 'f3', expected_hmm=False)


def test_exact_derived_and_review_sets_preserve_the_original_scores(
	tmp_path, monkeypatch
):
	artifact = tmp_path / 'artifacts'
	artifact.mkdir()
	config = tmp_path / 'config.yaml'
	config.write_text(
		'schema_version: 1\nsource_cells: {path: original.csv, sha256: frozen}\n'
	)
	a = []
	b = []
	for survey in subject.METRICS:
		for source in ('mae', 'local_bt', 'random'):
			for size in ('small', 'medium', 'large'):
				for i in range(5):
					layout = f'layout_{i:03d}'
					b.append(
						dict(baseline(survey, layout), source=source, data_size=size)
					)
					for d in ('0.0', '0.2'):
						a.extend(
							dict(
								cell(survey, layout),
								source=source,
								data_size=size,
								distillation_weight=d,
								consistency_weight=c,
							)
							for c in ('0.0', '0.1')
						)
	monkeypatch.setattr(subject, 'collect', lambda *_: (a, b, []))
	paired = artifact / 'derived'
	subject.export(config, artifact, paired)
	assert {p.name for p in paired.iterdir()} == subject.FILES
	assert len(subject.rows(paired / 'factorial_cells.csv')) == 540
	assert len(subject.rows(paired / 'factorial_comparison.csv')) == 108
	with pytest.raises(FileExistsError):
		subject.export(config, artifact, paired)
	original = artifact / 'original'
	original.mkdir()
	for name in PUBLISHED_FILES:
		(original / name).write_text('unchanged ' + name)
	before = {p.name: p.read_bytes() for p in original.iterdir()}
	out = tmp_path / 'reports/review'
	subject.update_review(original, paired, artifact, out)
	assert {p.name for p in out.iterdir()} == REVIEW_FILES
	assert before == {p.name: p.read_bytes() for p in original.iterdir()}
	assert (out / 'factorial_cells.csv').read_bytes() == (
		paired / 'factorial_cells.csv'
	).read_bytes()
	assert (out / 'factorial_contrasts.csv').read_bytes() == (
		original / 'factorial_contrasts.csv'
	).read_bytes()
	subject.update_review(original, paired, artifact, out)
	(out / 'unowned.txt').write_text('preserve me')
	with pytest.raises(ValueError, match='unowned'):
		subject.update_review(original, paired, artifact, out)
	with pytest.raises(ValueError, match='artifacts'):
		subject.update_review(out, paired, artifact, out)
