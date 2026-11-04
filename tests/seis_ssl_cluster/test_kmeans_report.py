"""Publication contract for the curated k-means versus HMM comparison."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/kmeans_k6810_multi_source_v1'
PUBLISHED_FILES = {
	'README.md',
	'comparison.csv',
	'hmm_advantage_by_source_and_budget.png',
	'provenance.json',
}


def test_kmeans_report_has_exact_review_files_and_bound_provenance():
	assert {path.name for path in REPORT.iterdir()} == PUBLISHED_FILES
	assert all(path.is_file() for path in REPORT.iterdir())
	provenance = json.loads((REPORT / 'provenance.json').read_text())
	assert set(provenance['published_files']) == PUBLISHED_FILES
	assert provenance['status'] == 'complete'
	assert provenance['training_conditions'] == 9
	assert provenance['downstream_cells'] == 135
	assert provenance['comparison_rows'] == 27
	assert provenance['layouts_per_condition'] == 5
	assert set(provenance['review_files_sha256']) == PUBLISHED_FILES - {
		'provenance.json'
	}
	for name, expected in provenance['review_files_sha256'].items():
		assert hashlib.sha256((REPORT / name).read_bytes()).hexdigest() == expected
	matrix = provenance['matrix']
	assert matrix['path'] == (
		'experiments/kmeans/kmeans_k6810_multi_source_v1/matrix.yaml'
	)
	assert (
		hashlib.sha256((ROOT / matrix['path']).read_bytes()).hexdigest()
		== (matrix['sha256'])
	)
	for source in provenance['source_artifacts']:
		assert not Path(source['path']).is_absolute()
		assert '..' not in Path(source['path']).parts
		assert len(source['sha256']) == 64
	csv_source = next(
		source
		for source in provenance['source_artifacts']
		if source['path'].endswith('/summary/comparison.csv')
	)
	assert provenance['csv_normalization'] == 'crlf_to_lf'
	csv_bytes = (REPORT / 'comparison.csv').read_bytes()
	assert b'\r' not in csv_bytes
	assert (
		hashlib.sha256(csv_bytes.replace(b'\n', b'\r\n')).hexdigest()
		== csv_source['sha256']
	)
	assert (
		(REPORT / 'hmm_advantage_by_source_and_budget.png')
		.read_bytes()
		.startswith(b'\x89PNG\r\n\x1a\n')
	)


def test_kmeans_report_retains_all_conditions_and_paired_metric_units():
	with (REPORT / 'comparison.csv').open(newline='') as stream:
		rows = list(csv.DictReader(stream))
	expected = set(
		itertools.product(
			('f3', 'parihaka', 'volve'),
			('mae', 'local_bt', 'random'),
			('small', 'medium', 'large'),
		)
	)
	assert len(rows) == 27
	assert {
		(row['survey'], row['source_family'], row['data_size']) for row in rows
	} == expected
	for row in rows:
		assert int(row['layout_count']) == 5
		assert row['evaluation_split'] == (
			'validation' if row['survey'] == 'f3' else 'test'
		)
		for metric in ('primary', 'secondary'):
			candidate = float(row[f'{metric}_kmeans_mean'])
			control = float(row[f'{metric}_hmm_v2_mean'])
			difference = float(row[f'{metric}_difference_mean'])
			assert all(math.isfinite(x) for x in (candidate, control, difference))
			assert difference == pytest.approx(candidate - control, abs=1e-12)
			if metric == 'secondary' or row['survey'] != 'volve':
				assert 0 <= candidate <= 1
				assert 0 <= control <= 1
			for column in ('kmeans', 'hmm_v2', 'difference'):
				deviation = float(row[f'{metric}_{column}_sample_sd'])
				assert math.isfinite(deviation)
				assert deviation >= 0
