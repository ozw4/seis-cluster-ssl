"""Regression contracts for the saved-results catalog; no model execution."""

import csv
import json

import pytest

from seis_ssl_cluster.results.experiment_catalog import (
	FIXED_FILES,
	build_catalog,
	discover,
	flatten_metrics,
	metric_info,
	summarize_layouts,
)


def write_json(path, value):
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value))
	return path


def read_csv(path):
	with path.open(encoding='utf-8-sig', newline='') as stream:
		return list(csv.DictReader(stream))


def test_export_scope_provenance_nulls_and_exact_file_set(tmp_path):
	root = tmp_path / 'artifacts/seis_ssl_cluster'
	f3 = root / 'surveys/f3/facies_benchmark_v2/benchmark/run_a/runs'
	for layout, score in [('layout_000', 0.4), ('layout_001', 0.6)]:
		write_json(
			f3
			/ (
				'model=mae_hmm_v2_k6810_distill020_cons010/'
				f'layout={layout}/size=small/evaluation/metrics.json'
			),
			{
				'aggregation_unit': 'unique_validation_voxel',
				'mean_iou': score,
				'macro_f1': None,
				'per_class_iou': {'0': 0.5},
				'confusion_matrix': [[3, 1], [2, 4]],
			},
		)
	parihaka = write_json(
		root / 'surveys/parihaka/facies_benchmark_v1/benchmark/p1/metrics.json',
		{
			'model': 'local_bt',
			'layout_id': 'layout_000',
			'data_size': 'large',
			'validation': {'channel_iou': 0.55},
			'test': {'channel_iou': 0.51},
			'benchmark_identity': {
				'training': {'seed': 42000},
				'private_values': [17, 88],
			},
		},
	)
	(root / 'surveys/parihaka/duplicate_metrics.json').symlink_to(parihaka)
	write_json(
		root / 'surveys/volve/horizon_benchmark_v1/benchmark/v1/metrics.json',
		{
			'model': 'mae',
			'layout_id': 'layout_000',
			'data_size': 'medium',
			'validation': {'macro_mae_samples': 5.0},
			'test': {
				'primary_common': {
					'macro_mae_samples': 6.0,
					'per_horizon': {'bcu': {'mae_ms': 24.0}},
				},
				'secondary_per_horizon': {'macro_mae_samples': 7.0},
			},
		},
	)
	write_json(
		root / 'archive/f3/smoke/metrics.json',
		{'model': '</script><b>random</b>', 'accuracy': 0.1},
	)
	table = f3 / 'validation_trace_metrics.csv'
	table.parent.mkdir(parents=True, exist_ok=True)
	table.write_text(
		'slice_index,accuracy,boundary_position_mae\n0,0.123456789012345,2.5\n'
	)
	write_json(f3 / 'boundary_metrics.json', {'vertical_boundary_f1': 0.4})
	(root / 'surveys/f3/alias_directory').symlink_to(f3, target_is_directory=True)
	definition = tmp_path / 'experiments/f3/facies_benchmark_v2/101_test/config.yaml'
	definition.parent.mkdir(parents=True)
	definition.write_text(f'outputs:\n  runs_root: {f3}\n')
	output = tmp_path / 'result'
	preview = build_catalog(tmp_path, root, output, dry_run=True)
	assert preview['status'] == 'dry_run'
	assert not output.exists()
	result = build_catalog(tmp_path, root, output)
	assert result['by_survey'] == {'f3': 3, 'parihaka': 2, 'volve': 3}
	assert result['source_statuses'].get('parse_error', 0) == 0
	rows = read_csv(output / 'experiment_metrics.csv')
	assert {r['split'] for r in rows} == {'validation', 'test', 'unknown'}
	assert {r['support'] for r in rows if r['survey'] == 'volve'} == {
		'validation_labels',
		'primary_common',
		'secondary_per_horizon',
	}
	assert len([r for r in rows if r['archived'] == 'True']) == 1
	assert any(r['status'] == 'smoke/feasibility' for r in rows)
	long = read_csv(output / 'metrics_long.csv')
	assert any(
		r['metric'] == 'per_horizon.bcu.mae_ms' and r['unit'] == 'ms' for r in long
	)
	assert any(r['metric'] == 'macro_f1' and r['value'] == '' for r in long)
	assert any(
		r['metric'] == 'confusion_matrix[0][0]' and r['unit'] == 'count' for r in long
	)
	assert not any('private_values' in r['metric'] for r in long)
	aggregate = next(
		r
		for r in read_csv(output / 'layout_summary.csv')
		if r['survey'] == 'f3' and r['metric'] == 'mean_iou'
	)
	assert int(aggregate['n']) == 2
	assert float(aggregate['mean']) == pytest.approx(0.5)
	assert float(aggregate['sample_sd']) == pytest.approx(0.1414213562373095)
	source = next(
		r
		for r in read_csv(output / 'source_files.csv')
		if r['path'].endswith('validation_trace_metrics.csv')
	)
	assert (
		read_csv(output / source['bundle_table'])[0]['accuracy'] == '0.123456789012345'
	)
	assert read_csv(output / 'experiment_catalog.csv')[0]['evaluation_rows'] == '2'
	assert any(
		r['metric_path'] == 'vertical_boundary_f1'
		for r in read_csv(output / 'published_statistics.csv')
	)
	html = (output / 'index.html').read_text()
	assert '</script><b>random</b>' not in html
	assert '\\u003c/script\\u003e' in html
	manifest = json.loads((output / 'manifest.json').read_text())
	actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
	expected = FIXED_FILES | {source['bundle_table']}
	assert actual == expected == set(manifest['published_files'])
	assert set(manifest['sha256']) == expected - {'manifest.json'}
	with pytest.raises(FileExistsError):
		build_catalog(tmp_path, root, output)


def test_discovery_excludes_generated_bundles_reports_and_directory_aliases(tmp_path):
	root = tmp_path / 'artifacts/seis_ssl_cluster'
	source = write_json(root / 'surveys/f3/results/metrics.json', {'mean_iou': 0.5})
	write_json(
		root / 'studies/all_experiment_metrics_old/metrics.json', {'mean_iou': 9}
	)
	write_json(
		root / 'studies/results_figure_candidates_old/metrics.json', {'mean_iou': 8}
	)
	write_json(tmp_path / 'reports/f3/metrics.json', {'mean_iou': 7})
	(root / 'surveys/f3/alias').symlink_to(source.parent, target_is_directory=True)
	assert discover(root, tmp_path / 'output') == [source]


def test_flattening_preserves_nonfinite_missing_and_explicit_units():
	metrics = flatten_metrics(
		{
			'macro_mae_samples': float('nan'),
			'coverage': {'count': 10},
			'per_horizon': {'h1': {'mae_ms': 4}},
			'order_violation_rate': 0.2,
			'within_2_samples_count': 8,
			'best_epoch': 17,
			'benchmark_identity': {'mae_samples': 999},
		}
	)
	assert metrics == {
		'macro_mae_samples': None,
		'coverage.count': 10.0,
		'per_horizon.h1.mae_ms': 4.0,
		'order_violation_rate': 0.2,
		'within_2_samples_count': 8.0,
	}
	assert metric_info('macro_mae_samples') == ('samples', 'lower')
	assert metric_info('per_horizon.h1.mae_ms') == ('ms', 'lower')
	assert metric_info('within_2_samples_count') == ('count', 'descriptive')
	assert metric_info('order_violation_rate') == ('fraction', 'lower')


def test_conflicting_replicate_values_are_not_averaged():
	base = {'survey': 'f3', 'replicate_id': 'layout_000'}
	rows, _ = summarize_layouts(
		[
			{**base, 'record_id': '1', '_metrics': {'mean_iou': 0.4}},
			{**base, 'record_id': '2', '_metrics': {'mean_iou': 0.6}},
		]
	)
	assert rows[0]['aggregation_status'] == 'conflicting_duplicate_replicates'
	assert rows[0]['mean'] is None
	assert rows[0]['sample_sd'] is None


def test_malformed_source_is_visible_in_inventory(tmp_path):
	root = tmp_path / 'artifacts/seis_ssl_cluster'
	broken = root / 'surveys/f3/metrics.json'
	broken.parent.mkdir(parents=True)
	broken.write_text('{broken')
	output = tmp_path / 'output'
	result = build_catalog(tmp_path, root, output)
	assert result['source_statuses']['parse_error'] == 1
	assert 'JSONDecodeError' in read_csv(output / 'source_files.csv')[0]['error']


def test_legacy_f3_identity_and_random_initialization_do_not_use_parent_family(
	tmp_path,
):
	root = tmp_path / 'artifacts/seis_ssl_cluster'
	checkpoint = root / 'surveys/f3/pretraining/hmm/encoder/latest.pt'
	write_json(
		checkpoint.parent / 'resolved_config.json',
		{
			'head': {'num_prototypes': 6},
			'loss': {'distillation_weight': 0},
			'student': {
				'init_checkpoint': 'mae_local_bt_five_way_v1/random/random_init.pt'
			},
			'train': {'seed': 42},
		},
	)
	metric = (
		root
		/ 'surveys/f3/evaluation/old_study/runs'
		/ 'split_002/cap25/random_role/evaluation/metrics.json'
	)
	write_json(metric, {'mean_iou': 0.3, 'aggregation_unit': 'unique_validation_voxel'})
	write_json(
		metric.with_name('evaluation_metadata.json'),
		{'model_tag': 'random_init_hmm_v1_k6'},
	)
	write_json(
		metric.parent.parent / 'decoder/resolved_config.json',
		{
			'embeddings': {'checkpoint_path': str(checkpoint)},
			'train': {'seed': 42000},
		},
	)
	probe = (
		root
		/ 'surveys/f3/evaluation/old_study/probes/model=mae'
		/ 'budget=cap100/subsample_seed=2/linear/metrics.json'
	)
	write_json(probe, {'mean_iou': 0.4})
	write_json(
		probe.with_name('probe_config_resolved.json'),
		{'inputs': {'validation_tokens': 'validation.npz'}},
	)
	output = tmp_path / 'result'
	build_catalog(tmp_path, root, output)
	rows = read_csv(output / 'experiment_metrics.csv')
	random = next(r for r in rows if r['model'] == 'random_init_hmm_v1_k6')
	assert random['initialization'] == 'Random'
	assert random['label_budget'] == 'cap25'
	assert random['replicate_id'] == 'split_index=split_002'
	assert random['distillation_weight'] == '0'
	assert random['head_ks'] == '6'
	assert random['encoder_seed'] == '42'
	probe_row = next(r for r in rows if r['model'] == 'mae')
	assert probe_row['split'] == 'validation'
	assert probe_row['support'] == 'validation_tokens'
	assert probe_row['evaluation_kind'] == 'token_probe'
