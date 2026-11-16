"""All saved versions remain visible, even when a baseline cannot be paired."""

import pytest

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.results import hmm_baseline_comparison as base
from seis_ssl_cluster.results import saved_hmm_comparison as subject


def fixture(tmp_path, monkeypatch):
	artifact = tmp_path / 'artifacts'
	records = []
	for version, split in [
		('v1', 'test'),
		('v1', 'validation'),
		('v2', 'test'),
		('v3', 'test'),
		('legacy', 'test'),
	]:
		source = artifact / f'surveys/parihaka/benchmark/{version}/{split}/metrics.json'
		source.parent.mkdir(parents=True)
		source.write_text('{}')
		records.append(
			{
				'survey': 'parihaka',
				'record_id': f'{version}_{split}',
				'protocol': f'HMM {version}'
				if version != 'legacy'
				else 'multi-head (legacy)',
				'hmm_version': version,
				'source_file': str(source.relative_to(artifact)),
				'source_sha256': file_sha256(source),
				'head_ks': '6' if version == 'v1' else '6|8|10',
				'model': f'mae_hmm_{version}',
				'support': 'test_labels' if split == 'test' else 'validation_labels',
				'split': split,
				'source_cohort': f'cohort/{version}',
				'checkpoint_selection': '',
				'archived': False,
				'status': 'saved_result',
				'initialization': 'MAE',
				'label_budget': 'small',
				'replicate_id': 'layout_000',
				'distillation_weight': '0.1',
				'consistency_weight': '',
				'decoder_seed': 42001 if version == 'v3' else 42000,
				'channel_iou': 0.4,
				'channel_f1': 0.6,
			}
		)
	cell = {
		'survey': 'parihaka',
		'source': 'mae',
		'data_size': 'small',
		'layout_id': 'layout_000',
		'distillation_weight': '0.1',
		'consistency_weight': '',
		'evaluation_split': 'test',
		'primary': 0.4,
		'secondary': 0.6,
		'decoder_seed': 42000,
		'model': 'mae_hmm_v2',
		'shared_identity': 'fixed',
		'comparison_identity': 'compatible',
		'metrics_file': records[2]['source_file'],
	}
	baseline = dict(
		cell, primary=0.5, metrics_file='baseline/metrics.json', model='mae'
	)

	def measured(path, *_args, **_kwargs):
		return {
			'model': path.parent.parent.name,
			'primary': 0.4,
			'secondary': 0.6,
			'decoder_seed': 42001 if 'v3' in path.parts else 42000,
			'shared_identity': 'fixed',
			'comparison_identity': 'compatible',
		}

	monkeypatch.setattr(subject, 'evaluation', measured)
	catalog = artifact / 'catalog.csv'
	base.write_csv(catalog, records)
	ref = {'path': 'catalog.csv', 'sha256': file_sha256(catalog)}
	return artifact, cell, baseline, records, ref


def test_all_versions_splits_and_seeds_are_retained_without_fabricating_deltas(
	tmp_path, monkeypatch
):
	artifact, cell, baseline, records, ref = fixture(tmp_path, monkeypatch)
	paired, metadata = subject.expand_saved(
		base.pair([cell], [baseline]), [baseline], artifact, ref
	)
	assert len(paired) == len(records) == 5
	assert {r['with_hmm_record_id'] for r in paired} == {
		r['record_id'] for r in records
	}
	assert metadata['saved_hmm_by_version'] == {'legacy': 1, 'v1': 2, 'v2': 1, 'v3': 1}
	assert metadata['matched_rows'] == 4
	unmatched = [r for r in paired if r['comparison_status'] != 'paired']
	assert len(unmatched) == 1
	assert unmatched[0]['source_evaluation_split'] == 'validation'
	assert unmatched[0]['with_hmm_primary'] == 0.4
	assert (
		unmatched[0]['without_hmm_primary'] == unmatched[0]['primary_improvement'] == ''
	)
	assert unmatched[0]['comparison_note']
	v1 = next(r for r in paired if r['with_hmm_record_id'] == 'v1_test')
	assert v1['primary_improvement'] == pytest.approx(-0.1)
	v3 = next(r for r in paired if r['hmm_version'] == 'v3')
	assert v3['comparison_type'] == 'different_decoder_seed'
	summary = base.summarize(paired, allow_partial=True)
	assert len(summary) == 4
	assert {r['hmm_version'] for r in summary} == {'v1', 'v2', 'v3', 'legacy'}
	assert all(
		r['n'] == 1 and r['primary_improvement_sample_sd'] == '' for r in summary
	)


def test_incompatible_support_remains_visible_without_a_comparison(
	tmp_path, monkeypatch
):
	artifact, cell, baseline, _, ref = fixture(tmp_path, monkeypatch)
	baseline['shared_identity'] = baseline['comparison_identity'] = 'different_support'
	paired, metadata = subject.expand_saved(
		base.pair([cell], [dict(baseline, shared_identity='fixed')]),
		[baseline],
		artifact,
		ref,
	)
	v1 = next(r for r in paired if r['with_hmm_record_id'] == 'v1_test')
	assert v1['comparison_status'] == 'unmatched_baseline'
	assert v1['primary_improvement'] == ''
	assert 'identity' in v1['comparison_note']
	assert metadata['saved_hmm_records'] == 5


def test_source_hash_drift_and_missing_versions_are_rejected(tmp_path, monkeypatch):
	artifact, cell, baseline, records, ref = fixture(tmp_path, monkeypatch)
	path = artifact / records[0]['source_file']
	path.write_text('{"changed":true}')
	with pytest.raises(ValueError, match='source changed'):
		subject.expand_saved(base.pair([cell], [baseline]), [baseline], artifact, ref)
	path.write_text('{}')
	base.write_csv(
		artifact / 'catalog.csv', [r for r in records if r['hmm_version'] != 'v1']
	)
	ref['sha256'] = file_sha256(artifact / 'catalog.csv')
	with pytest.raises(ValueError, match='retain v1, v2 and v3'):
		subject.expand_saved(base.pair([cell], [baseline]), [baseline], artifact, ref)


def test_sources_cannot_escape_artifacts(tmp_path):
	with pytest.raises(ValueError, match='remain in artifacts'):
		subject.artifact_source(
			tmp_path / 'artifacts', str(tmp_path / 'reports/metrics.json')
		)
