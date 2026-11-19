"""Preserve saved endpoints and restrict deltas to audited comparison pairs."""

import json
from pathlib import Path

import pytest

from seis_ssl_cluster.results import all_experiment_comparison as comparison
from seis_ssl_cluster.results.hmm_baseline_comparison import rows, write_csv


def record(record_id, *, protocol='HMM v2', seed='42000', score='0.5'):
	return {
		'record_id': record_id,
		'source_file': f'surveys/f3/{record_id}/metrics.json',
		'source_cohort': f'surveys/f3/{record_id}',
		'source_sha256': 'a' * 64,
		'survey': 'f3',
		'initialization': 'MAE',
		'model': record_id,
		'protocol': protocol,
		'hmm_version': 'v2' if protocol == 'HMM v2' else '',
		'distillation_weight': '0.0',
		'consistency_weight': '0.0',
		'encoder_seed': '42',
		'decoder_seed': seed,
		'label_budget': 'large',
		'replicate_id': 'layout_000',
		'split': 'common_test',
		'support': 'unique_test_voxel',
		'mean_iou': score,
		'macro_f1': '0.6',
	}


def baseline_pair(observed, reference):
	return {
		'comparison_status': 'paired',
		'with_hmm_record_id': observed['record_id'],
		'without_hmm_metrics_file': reference['source_file'],
		'with_hmm_decoder_seed': observed['decoder_seed'],
		'without_hmm_decoder_seed': reference['decoder_seed'],
		'baseline_reference_status': 'historical',
		'with_hmm_primary': observed['mean_iou'],
		'without_hmm_primary': reference['mean_iou'],
		'with_hmm_secondary': observed['macro_f1'],
		'without_hmm_secondary': reference['macro_f1'],
	}


def test_publish_all_endpoints_with_both_baseline_seeds(tmp_path, monkeypatch):
	monkeypatch.setattr(comparison, 'KMEANS_STUDIES', ())
	observed = record('hmm', score='0.7')
	references = [
		record(f'baseline_{s}', protocol='none', seed=s) for s in ('42000', '42001')
	]
	standalone = record('old_kmeans', protocol='k-means')
	# Historical F3 exports use both spellings for the same common test support.
	references[0]['split'] = 'test'
	records = [observed, *references, standalone]
	catalog = tmp_path / comparison.CATALOG / 'experiment_metrics.csv'
	catalog.parent.mkdir(parents=True)
	write_csv(catalog, records)
	baseline = tmp_path / comparison.BASELINES / 'factorial_cells.csv'
	baseline.parent.mkdir(parents=True)
	write_csv(baseline, [baseline_pair(observed, r) for r in references])
	(baseline.parent / 'provenance.json').write_text(
		json.dumps(
			{
				'saved_catalog': {'sha256': comparison.sha256(catalog)},
			}
		)
	)
	output = tmp_path / 'comparison'
	metadata = comparison.export(tmp_path, output)
	assert {p.name for p in output.iterdir()} == comparison.FILES
	assert metadata['saved_evaluations'] == 4
	assert metadata['comparison_csv_rows'] == 5
	assert metadata['paired_rows'] == 2
	actual = rows(output / 'all_experiment_comparison.csv')
	paired = [r for r in actual if r['comparison_status'] == 'paired']
	assert {r['reference_decoder_seed'] for r in paired} == {'42000', '42001'}
	assert {r['same_decoder_seed'] for r in paired} == {'True', 'False'}
	assert [float(r['primary_improvement']) for r in paired] == pytest.approx(
		[0.2, 0.2]
	)
	assert {r['record_id'] for r in actual} == {r['record_id'] for r in records}
	assert all(
		r['primary_improvement'] == '' for r in actual if r['record_id'] != 'hmm'
	)
	assert comparison.export(tmp_path, output) == metadata
	(output / 'unowned.txt').touch()
	with pytest.raises(ValueError, match='unowned'):
		comparison.export(tmp_path, output)


def test_kmeans_common_test_binding_checks_original_score_and_hash(tmp_path):
	observed, reference = (
		record('hmm', score='0.7'),
		record('kmeans', protocol='k-means'),
	)
	cell = {
		'survey': 'f3',
		'layout_id': 'layout_000',
		'data_size': 'large',
		'control_id': 'hmm',
		'candidate_id': 'kmeans',
		'control_primary': 0.7,
		'primary': 0.5,
		'control_secondary': 0.6,
		'secondary': 0.6,
	}
	for role, item in (('control', observed), ('candidate', reference)):
		path = tmp_path / item['source_cohort'] / 'summary/per_layout.csv'
		path.parent.mkdir(parents=True)
		path.write_text('source scores\n')
		cell[f'{role}_metrics'] = {'path': str(path), 'sha256': comparison.sha256(path)}
	index = comparison.ComparisonIndex(tmp_path, [observed, reference])
	[(key, paired)] = index.kmeans_pairs('kmeans_zero', [cell])
	assert key == 'hmm'
	assert paired['reference_record_id'] == 'kmeans'
	assert paired['reference_split'] == 'common_test'
	assert float(paired['primary_improvement']) == pytest.approx(0.2)
	with pytest.raises(ValueError, match='score differs'):
		list(index.kmeans_pairs('kmeans_zero', [{**cell, 'primary': 0.1}]))
	Path(cell['candidate_metrics']['path']).write_text('changed source\n')
	with pytest.raises(ValueError, match='hash drift'):
		list(
			comparison.ComparisonIndex(tmp_path, [observed, reference]).kmeans_pairs(
				'kmeans_zero', [cell]
			)
		)


def test_split_support_and_duplicate_comparisons_are_rejected(tmp_path):
	observed = record('hmm')
	reference = record('baseline', protocol='none')
	for field, value in [('split', 'validation'), ('support', 'validation_labels')]:
		with pytest.raises(ValueError, match=field):
			comparison._compare(  # noqa: SLF001
				observed, {**reference, field: value}, study='test'
			)
	pair = comparison._compare(observed, reference, study='test')  # noqa: SLF001
	with pytest.raises(ValueError, match='duplicate verified comparison'):
		comparison.assemble([observed, reference], [('hmm', pair), ('hmm', pair)])
	with pytest.raises(ValueError, match='remain in artifacts'):
		comparison.ComparisonIndex(
			tmp_path / 'artifacts',
			[{**observed, 'source_file': str(tmp_path / 'reports/metrics.json')}],
		)
