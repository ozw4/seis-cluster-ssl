from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from seis_ssl_cluster.f3.lithology.m1_results import (
	F3StratHMMM1PublishConfig,
	F3StratHMMM1ResultsConfig,
	F3StratHMMM1ResultsResult,
	consolidate_f3_strat_hmm_m1_results,
	f3_strat_hmm_m1_results_config_from_mapping,
	publish_f3_strat_hmm_m1_results,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / 'proc' / 'seis_ssl_cluster' / 'summarize_f3_strat_hmm_m1_results.py'


def test_successful_summary_creation(tmp_path: Path) -> None:
	_require_matplotlib_agg()
	config = _write_inputs(tmp_path)

	result = consolidate_f3_strat_hmm_m1_results(config)

	assert result.summary_json.is_file()
	assert result.summary_markdown.is_file()
	assert {path.name for path in result.figure_paths} == {
		'label_budget_delta_curves.png',
		'split_index_deltas.png',
		'single_run_metric_comparison.png',
	}
	assert {path.name for path in result.table_paths} == {
		'single_split_comparison.csv',
		'label_budget_summary.csv',
		'split_index_deltas.csv',
	}
	for table_path in result.table_paths:
		assert table_path.is_file()
	for figure_path in result.figure_paths:
		assert figure_path.is_file()
		assert figure_path.stat().st_size > 0
	single_split_rows = _read_csv(
		config.output_dir / 'tables/single_split_comparison.csv',
	)
	assert [row['role'] for row in single_split_rows] == [
		'baseline',
		'candidate',
		'delta',
	]
	assert single_split_rows[2]['macro_f1'] == '0.060000'
	payload = json.loads(result.summary_json.read_text(encoding='utf-8'))
	assert payload['schema_version'] == 1
	assert payload['baseline_model'] == 'mae_baseline'
	assert payload['candidate_model'] == 'strat_hmm_m1'
	assert payload['single_split']['delta']['macro_f1'] == pytest.approx(0.06)
	assert [row['budget_id'] for row in payload['label_budget']['budgets']] == [
		'cap25',
		'cap50',
		'cap100',
		'cap250',
		'cap500',
		'full',
	]
	assert payload['label_budget']['budgets'][0] == {
		'budget_id': 'cap25',
		'per_class_cap': 25,
		'n_pairs': 2,
		'mean_delta_macro_f1': pytest.approx(0.055),
		'win_rate_macro_f1': 1.0,
		'mean_delta_mean_iou': pytest.approx(0.05),
		'win_rate_mean_iou': 1.0,
		'mean_delta_balanced_accuracy': pytest.approx(0.035),
		'win_rate_balanced_accuracy': 1.0,
	}
	assert payload['split_index']['win_rates'] == {
		'macro_f1': 1.0,
		'mean_iou': 1.0,
		'balanced_accuracy': 0.5,
	}
	markdown = result.summary_markdown.read_text(encoding='utf-8')
	assert 'HMM labels are a structured pretext signal' in markdown
	assert 'Single-run result is strong positive' in markdown
	assert 'Label-budget robustness is strongest in low-label regimes' in markdown
	assert 'positive macro F1 and mean IoU deltas on all tested splits' in markdown
	assert (
		'![Label-budget delta curves](figures/label_budget_delta_curves.png)'
		in markdown
	)
	assert '![Split/index deltas](figures/split_index_deltas.png)' in markdown
	assert result.publish_manifest is None


def test_publish_copies_expected_small_files(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_require_matplotlib_agg()
	publish_dir = tmp_path / 'results' / 'f3' / 'facies_benchmark_v1' / 'm1'
	monkeypatch.chdir(tmp_path)
	config = _write_inputs(
		tmp_path,
		publish=F3StratHMMM1PublishConfig(
			enabled=True,
			output_dir=Path('results/f3/facies_benchmark_v1/m1'),
			include_figures=True,
		),
	)

	result = consolidate_f3_strat_hmm_m1_results(config)

	published_files = {
		path.relative_to(publish_dir)
		for path in publish_dir.rglob('*')
		if path.is_file()
	}
	assert result.publish_manifest is not None
	assert published_files == {
		Path('m1_results_summary.md'),
		Path('m1_results_summary.json'),
		Path('tables/single_split_comparison.csv'),
		Path('tables/label_budget_summary.csv'),
		Path('tables/split_index_deltas.csv'),
		Path('figures/label_budget_delta_curves.png'),
		Path('figures/split_index_deltas.png'),
		Path('figures/single_run_metric_comparison.png'),
		Path('publish_manifest.json'),
	}
	assert (publish_dir / 'm1_results_summary.md').read_text(
		encoding='utf-8'
	) == result.summary_markdown.read_text(
		encoding='utf-8',
	)
	manifest_payload = json.loads(
		(publish_dir / 'publish_manifest.json').read_text(encoding='utf-8'),
	)
	assert sorted(item['target'] for item in manifest_payload['items']) == sorted(
		str(path) for path in published_files if path.name != 'publish_manifest.json'
	)


def test_publish_include_figures_false_omits_markdown_figure_links(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_require_matplotlib_agg()
	publish_dir = tmp_path / 'results' / 'f3' / 'facies_benchmark_v1' / 'm1'
	monkeypatch.chdir(tmp_path)
	config = _write_inputs(
		tmp_path,
		publish=F3StratHMMM1PublishConfig(
			enabled=True,
			output_dir=Path('results/f3/facies_benchmark_v1/m1'),
			include_figures=False,
		),
	)

	result = consolidate_f3_strat_hmm_m1_results(config)

	assert result.publish_manifest is not None
	assert not (publish_dir / 'figures').exists()
	generated_markdown = result.summary_markdown.read_text(encoding='utf-8')
	published_markdown = (publish_dir / 'm1_results_summary.md').read_text(
		encoding='utf-8',
	)
	assert '(figures/' in generated_markdown
	assert '(figures/' not in published_markdown


def test_publish_disabled_does_nothing(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_require_matplotlib_agg()
	publish_dir = tmp_path / 'results' / 'disabled'
	monkeypatch.chdir(tmp_path)
	config = _write_inputs(
		tmp_path,
		publish=F3StratHMMM1PublishConfig(
			enabled=False,
			output_dir=Path('results/disabled'),
		),
	)

	result = consolidate_f3_strat_hmm_m1_results(config)

	assert result.publish_manifest is None
	assert not publish_dir.exists()


def test_publish_refuses_prohibited_suffix(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.chdir(tmp_path)
	result = _write_publishable_result(tmp_path)
	checkpoint = tmp_path / 'artifacts' / 'model.pt'
	checkpoint.write_bytes(b'heavy')
	result = F3StratHMMM1ResultsResult(
		summary_json=result.summary_json,
		summary_markdown=result.summary_markdown,
		table_paths=(checkpoint,),
		figure_paths=(),
		warnings=(),
	)

	with pytest.raises(ValueError, match='forbidden suffix'):
		publish_f3_strat_hmm_m1_results(
			result,
			F3StratHMMM1PublishConfig(
				enabled=True,
				output_dir=Path('results'),
			),
		)


def test_publish_refuses_oversized_file(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.chdir(tmp_path)
	result = _write_publishable_result(tmp_path)
	large_table = tmp_path / 'artifacts' / 'large.csv'
	large_table.write_bytes(b'12345')
	result = F3StratHMMM1ResultsResult(
		summary_json=result.summary_json,
		summary_markdown=result.summary_markdown,
		table_paths=(large_table,),
		figure_paths=(),
		warnings=(),
	)

	with pytest.raises(ValueError, match='exceeds max_file_size_bytes'):
		publish_f3_strat_hmm_m1_results(
			result,
			F3StratHMMM1PublishConfig(
				enabled=True,
				output_dir=Path('results'),
				max_file_size_bytes=4,
			),
		)


def test_relative_publish_path_is_preserved(tmp_path: Path) -> None:
	config = _write_inputs(tmp_path)

	resolved = f3_strat_hmm_m1_results_config_from_mapping(
		{
			'inputs': {
				'baseline_comparison_csv': str(config.baseline_comparison_csv),
				'label_budget_suite_root': str(config.label_budget_suite_root),
				'split_index_suite_root': str(config.split_index_suite_root),
			},
			'models': {
				'baseline': config.baseline_model,
				'candidate': config.candidate_model,
			},
			'outputs': {'output_dir': str(config.output_dir)},
			'publish': {
				'enabled': True,
				'output_dir': 'artifacts/not-results',
			},
		}
	)
	assert resolved.publish.output_dir == Path('artifacts/not-results')


def test_absolute_publish_path_is_preserved(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.chdir(tmp_path)
	config = _write_inputs(tmp_path)

	explicit_output = tmp_path / 'outside-results' / 'm1'
	resolved = f3_strat_hmm_m1_results_config_from_mapping(
		{
			'inputs': {
				'baseline_comparison_csv': str(config.baseline_comparison_csv),
				'label_budget_suite_root': str(config.label_budget_suite_root),
				'split_index_suite_root': str(config.split_index_suite_root),
			},
			'models': {
				'baseline': config.baseline_model,
				'candidate': config.candidate_model,
			},
			'outputs': {'output_dir': str(config.output_dir)},
			'publish': {
				'enabled': True,
				'output_dir': str(explicit_output),
			},
		}
	)
	assert resolved.publish.output_dir == explicit_output


def test_missing_required_input_file_raises_clear_error(tmp_path: Path) -> None:
	config = _write_inputs(tmp_path)
	config = F3StratHMMM1ResultsConfig(
		baseline_comparison_csv=tmp_path / 'missing.csv',
		label_budget_suite_root=config.label_budget_suite_root,
		split_index_suite_root=config.split_index_suite_root,
		output_dir=config.output_dir,
		baseline_model=config.baseline_model,
		candidate_model=config.candidate_model,
	)

	with pytest.raises(FileNotFoundError, match='required input file does not exist'):
		consolidate_f3_strat_hmm_m1_results(config)


def test_baseline_candidate_row_lookup_is_deterministic(tmp_path: Path) -> None:
	_require_matplotlib_agg()
	config = _write_inputs(tmp_path, reverse_comparison_order=True)

	payload = _summary_payload(config)

	assert payload['single_split']['baseline']['accuracy'] == pytest.approx(0.70)
	assert payload['single_split']['candidate']['accuracy'] == pytest.approx(0.76)


def test_duplicate_comparison_row_fails_instead_of_guessing(tmp_path: Path) -> None:
	config = _write_inputs(tmp_path)
	_append_csv_row(
		config.baseline_comparison_csv,
		{
			'MODEL_TAG': 'mae_baseline',
			'BASELINE_TAG': '',
			'accuracy': '0.50',
			'balanced_accuracy': '0.50',
			'macro_f1': '0.50',
			'weighted_f1': '0.50',
			'mean_iou': '0.50',
		},
	)

	with pytest.raises(ValueError, match='expected exactly one comparison row'):
		consolidate_f3_strat_hmm_m1_results(config)


def test_decision_go_when_macro_f1_and_mean_iou_are_positive(
	tmp_path: Path,
) -> None:
	_require_matplotlib_agg()
	config = _write_inputs(tmp_path)

	payload = _summary_payload(config)

	assert payload['decision']['guidance'] == 'go'
	assert 'positive' in payload['decision']['summary']


def test_markdown_uses_mixed_language_when_decision_is_hold(
	tmp_path: Path,
) -> None:
	_require_matplotlib_agg()
	config = _write_inputs(tmp_path)
	comparison_rows = _read_csv(config.baseline_comparison_csv)
	for row in comparison_rows:
		if row['MODEL_TAG'] == 'strat_hmm_m1':
			row['macro_f1'] = '0.55'
	_write_csv(
		config.baseline_comparison_csv, tuple(comparison_rows[0]), comparison_rows
	)
	split_deltas = config.split_index_suite_root / 'reports' / 'split_paired_deltas.csv'
	split_rows = _read_csv(split_deltas)
	split_rows[0]['delta_mean_iou'] = '-0.01'
	_write_csv(split_deltas, tuple(split_rows[0]), split_rows)

	result = consolidate_f3_strat_hmm_m1_results(config)

	payload = json.loads(result.summary_json.read_text(encoding='utf-8'))
	markdown = result.summary_markdown.read_text(encoding='utf-8')
	assert payload['decision']['guidance'] == 'hold'
	assert 'Single-run result is mixed' in markdown
	assert 'Single-run result is strong positive' not in markdown
	assert 'Split/index robustness is mixed' in markdown
	assert 'positive macro F1 and mean IoU deltas on all tested splits' not in markdown


def test_full_budget_duplicate_seed_rows_are_collapsed_when_manifest_exposes_identity(
	tmp_path: Path,
) -> None:
	_require_matplotlib_agg()
	config = _write_inputs(tmp_path, duplicate_full_identity=True)

	payload = _summary_payload(config)

	full = next(
		row for row in payload['label_budget']['budgets'] if row['budget_id'] == 'full'
	)
	assert full['n_pairs'] == 1
	assert full['mean_delta_macro_f1'] == pytest.approx(0.02)
	assert any(
		'full duplicate label-budget rows collapsed' in warning
		for warning in payload['warnings']
	)
	assert any('balanced_accuracy' in warning for warning in payload['warnings'])


def test_full_budget_duplicate_seed_rows_warn_when_manifest_lacks_identity(
	tmp_path: Path,
) -> None:
	_require_matplotlib_agg()
	config = _write_inputs(tmp_path)
	manifest_path = config.label_budget_suite_root / 'suite_manifest.json'
	manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
	for row in manifest['rows']:
		row.pop('paired_identity_hash')
	_write_json(manifest_path, manifest)

	payload = _summary_payload(config)

	full = next(
		row for row in payload['label_budget']['budgets'] if row['budget_id'] == 'full'
	)
	assert full['n_pairs'] == 2
	assert any(
		'duplicate independence cannot be inferred' in warning
		for warning in payload['warnings']
	)


def test_empty_label_budget_rows_raise_clear_error(tmp_path: Path) -> None:
	_require_matplotlib_agg()
	config = _write_inputs(tmp_path)
	_write_csv(
		config.label_budget_suite_root / 'reports' / 'paired_deltas.csv',
		(
			'budget_id',
			'per_class_cap',
			'subsample_seed',
			'delta_accuracy',
			'delta_balanced_accuracy',
			'delta_macro_f1',
			'delta_weighted_f1',
			'delta_mean_iou',
		),
		[],
	)

	with pytest.raises(ValueError, match='csv file contains no rows'):
		consolidate_f3_strat_hmm_m1_results(config)


def test_missing_required_label_budget_raises_clear_error(tmp_path: Path) -> None:
	config = _write_inputs(tmp_path)
	paired_deltas = config.label_budget_suite_root / 'reports' / 'paired_deltas.csv'
	rows = _read_csv(paired_deltas)
	_write_csv(
		paired_deltas,
		tuple(rows[0]),
		[row for row in rows if row['budget_id'] != 'cap50'],
	)

	with pytest.raises(
		ValueError,
		match=r'missing required budget_id rows.*cap50',
	):
		consolidate_f3_strat_hmm_m1_results(config)


def test_missing_label_budget_row_id_raises_clear_error(tmp_path: Path) -> None:
	config = _write_inputs(tmp_path)
	_write_csv(
		config.label_budget_suite_root / 'reports' / 'paired_deltas.csv',
		(
			'budget_id',
			'per_class_cap',
			'subsample_seed',
			'delta_accuracy',
			'delta_balanced_accuracy',
			'delta_macro_f1',
			'delta_weighted_f1',
			'delta_mean_iou',
		),
		[
			{
				'budget_id': '',
				'per_class_cap': '25',
				'subsample_seed': '0',
				'delta_accuracy': '0.02',
				'delta_balanced_accuracy': '0.04',
				'delta_macro_f1': '0.05',
				'delta_weighted_f1': '0.03',
				'delta_mean_iou': '0.04',
			},
		],
	)

	with pytest.raises(ValueError, match="missing required value for 'budget_id'"):
		consolidate_f3_strat_hmm_m1_results(config)


def test_split_index_figure_supports_single_split(tmp_path: Path) -> None:
	_require_matplotlib_agg()
	config = _write_inputs(tmp_path, single_split_index=True)

	result = consolidate_f3_strat_hmm_m1_results(config)

	split_figure = config.output_dir / 'figures' / 'split_index_deltas.png'
	assert split_figure in result.figure_paths
	assert split_figure.is_file()
	assert split_figure.stat().st_size > 0
	payload = json.loads(result.summary_json.read_text(encoding='utf-8'))
	assert len(payload['split_index']['splits']) == 1


def test_cli_supports_config_and_dry_run(tmp_path: Path) -> None:
	config = _write_inputs(tmp_path)
	config_path = tmp_path / 'config.yaml'
	config_path.write_text(
		'\n'.join(
			(
				'inputs:',
				f'  baseline_comparison_csv: {config.baseline_comparison_csv}',
				f'  label_budget_suite_root: {config.label_budget_suite_root}',
				f'  split_index_suite_root: {config.split_index_suite_root}',
				'models:',
				'  baseline: mae_baseline',
				'  candidate: strat_hmm_m1',
				'outputs:',
				f'  output_dir: {config.output_dir}',
				'',
			),
		),
		encoding='utf-8',
	)
	env = os.environ.copy()
	env['PYTHONPATH'] = os.pathsep.join(
		(str(REPO_ROOT / 'src'), env.get('PYTHONPATH', '')),
	)

	completed = subprocess.run(  # noqa: S603
		[sys.executable, str(CLI), '--config', str(config_path), '--dry-run'],
		cwd=REPO_ROOT,
		env=env,
		text=True,
		capture_output=True,
		check=True,
		timeout=30,
	)

	assert 'stage: summarize_f3_strat_hmm_m1_results' in completed.stdout
	assert 'execution: dry-run' in completed.stdout
	assert 'outputs.figures_dir:' in completed.stdout
	assert not (config.output_dir / 'm1_results_summary.json').exists()


def test_cli_runs_end_to_end_and_publishes_synthetic_fixtures(
	tmp_path: Path,
) -> None:
	_require_matplotlib_agg()
	config = _write_inputs(tmp_path)
	publish_dir = tmp_path / 'results' / 'f3' / 'facies_benchmark_v1' / 'm1'
	config_path = tmp_path / 'config.yaml'
	config_path.write_text(
		'\n'.join(
			(
				'inputs:',
				f'  baseline_comparison_csv: {config.baseline_comparison_csv}',
				f'  label_budget_suite_root: {config.label_budget_suite_root}',
				f'  split_index_suite_root: {config.split_index_suite_root}',
				'models:',
				'  baseline: mae_baseline',
				'  candidate: strat_hmm_m1',
				'outputs:',
				f'  output_dir: {config.output_dir}',
				'publish:',
				'  enabled: true',
				'  output_dir: results/f3/facies_benchmark_v1/m1',
				'  include_figures: true',
				'  max_file_size_mb: 10',
				'',
			),
		),
		encoding='utf-8',
	)
	env = os.environ.copy()
	env['PYTHONPATH'] = os.pathsep.join(
		(str(REPO_ROOT / 'src'), env.get('PYTHONPATH', '')),
	)
	env['MPLBACKEND'] = 'Agg'

	completed = subprocess.run(  # noqa: S603
		[sys.executable, str(CLI), '--config', str(config_path)],
		cwd=tmp_path,
		env=env,
		text=True,
		capture_output=True,
		check=True,
		timeout=30,
	)

	assert 'f3_strat_hmm_m1_results.summary_json:' in completed.stdout
	assert 'published F3 strat-HMM M1 results:' in completed.stdout
	assert (config.output_dir / 'm1_results_summary.json').is_file()
	assert (config.output_dir / 'm1_results_summary.md').is_file()
	assert (config.output_dir / 'figures' / 'label_budget_delta_curves.png').is_file()
	assert (config.output_dir / 'figures' / 'split_index_deltas.png').is_file()
	assert (publish_dir / 'm1_results_summary.json').is_file()
	assert (publish_dir / 'm1_results_summary.md').is_file()
	assert (publish_dir / 'tables' / 'label_budget_summary.csv').is_file()
	assert (publish_dir / 'figures' / 'label_budget_delta_curves.png').is_file()
	manifest = json.loads(
		(publish_dir / 'publish_manifest.json').read_text(encoding='utf-8'),
	)
	created_at = _parse_created_at_utc(manifest['created_at_utc'])
	assert created_at.tzinfo == timezone.utc
	assert manifest['created_at_utc'] != '1970-01-01T00:00:00Z'


def _summary_payload(config: F3StratHMMM1ResultsConfig) -> dict[str, object]:
	result = consolidate_f3_strat_hmm_m1_results(config)
	return json.loads(result.summary_json.read_text(encoding='utf-8'))


def _parse_created_at_utc(value: object) -> datetime:
	assert isinstance(value, str)
	return datetime.fromisoformat(value.replace('Z', '+00:00'))


def _write_inputs(
	tmp_path: Path,
	*,
	reverse_comparison_order: bool = False,
	duplicate_full_identity: bool = False,
	single_split_index: bool = False,
	publish: F3StratHMMM1PublishConfig | None = None,
) -> F3StratHMMM1ResultsConfig:
	baseline_csv = tmp_path / 'comparison_table.csv'
	label_root = tmp_path / 'label_budget_m1_v1'
	split_root = tmp_path / 'split_index_m1_v1'
	output_dir = tmp_path / 'm1_results'
	_write_comparison_csv(baseline_csv, reverse_order=reverse_comparison_order)
	_write_label_budget(label_root, duplicate_full_identity=duplicate_full_identity)
	_write_split_index(split_root, single_split=single_split_index)
	return F3StratHMMM1ResultsConfig(
		baseline_comparison_csv=baseline_csv,
		label_budget_suite_root=label_root,
		split_index_suite_root=split_root,
		output_dir=output_dir,
		baseline_model='mae_baseline',
		candidate_model='strat_hmm_m1',
		publish=F3StratHMMM1PublishConfig() if publish is None else publish,
	)


def _write_publishable_result(tmp_path: Path) -> F3StratHMMM1ResultsResult:
	root = tmp_path / 'artifacts'
	summary_json = root / 'm1_results_summary.json'
	summary_markdown = root / 'm1_results_summary.md'
	table = root / 'tables' / 'single_split_comparison.csv'
	summary_json.parent.mkdir(parents=True, exist_ok=True)
	table.parent.mkdir(parents=True, exist_ok=True)
	summary_json.write_text('{"schema_version": 1}\n', encoding='utf-8')
	summary_markdown.write_text('# summary\n', encoding='utf-8')
	table.write_text('role,macro_f1\nbaseline,0.5\n', encoding='utf-8')
	return F3StratHMMM1ResultsResult(
		summary_json=summary_json,
		summary_markdown=summary_markdown,
		table_paths=(table,),
		figure_paths=(),
		warnings=(),
	)


def _write_comparison_csv(path: Path, *, reverse_order: bool) -> None:
	rows = [
		{
			'MODEL_TAG': 'mae_baseline',
			'BASELINE_TAG': '',
			'accuracy': '0.70',
			'balanced_accuracy': '0.62',
			'macro_f1': '0.60',
			'weighted_f1': '0.71',
			'mean_iou': '0.52',
		},
		{
			'MODEL_TAG': 'strat_hmm_m1',
			'BASELINE_TAG': '',
			'accuracy': '0.76',
			'balanced_accuracy': '0.65',
			'macro_f1': '0.66',
			'weighted_f1': '0.77',
			'mean_iou': '0.59',
		},
		{
			'MODEL_TAG': '',
			'BASELINE_TAG': 'z_only_v1',
			'accuracy': '0.41',
			'balanced_accuracy': '0.40',
			'macro_f1': '0.35',
			'weighted_f1': '0.43',
			'mean_iou': '0.26',
		},
	]
	if reverse_order:
		rows.reverse()
	_write_csv(
		path,
		(
			'MODEL_TAG',
			'BASELINE_TAG',
			'accuracy',
			'balanced_accuracy',
			'macro_f1',
			'weighted_f1',
			'mean_iou',
		),
		rows,
	)


def _write_label_budget(root: Path, *, duplicate_full_identity: bool) -> None:
	rows = [
		{
			'budget_id': 'cap25',
			'per_class_cap': '25',
			'subsample_seed': '0',
			'delta_accuracy': '0.02',
			'delta_balanced_accuracy': '0.04',
			'delta_macro_f1': '0.05',
			'delta_weighted_f1': '0.03',
			'delta_mean_iou': '0.04',
		},
		{
			'budget_id': 'cap25',
			'per_class_cap': '25',
			'subsample_seed': '1',
			'delta_accuracy': '0.03',
			'delta_balanced_accuracy': '0.03',
			'delta_macro_f1': '0.06',
			'delta_weighted_f1': '0.04',
			'delta_mean_iou': '0.06',
		},
		{
			'budget_id': 'cap50',
			'per_class_cap': '50',
			'subsample_seed': '0',
			'delta_accuracy': '0.02',
			'delta_balanced_accuracy': '0.03',
			'delta_macro_f1': '0.05',
			'delta_weighted_f1': '0.03',
			'delta_mean_iou': '0.05',
		},
		{
			'budget_id': 'cap100',
			'per_class_cap': '100',
			'subsample_seed': '0',
			'delta_accuracy': '0.02',
			'delta_balanced_accuracy': '0.03',
			'delta_macro_f1': '0.04',
			'delta_weighted_f1': '0.03',
			'delta_mean_iou': '0.04',
		},
		{
			'budget_id': 'cap250',
			'per_class_cap': '250',
			'subsample_seed': '0',
			'delta_accuracy': '0.02',
			'delta_balanced_accuracy': '0.02',
			'delta_macro_f1': '0.03',
			'delta_weighted_f1': '0.02',
			'delta_mean_iou': '0.03',
		},
		{
			'budget_id': 'cap500',
			'per_class_cap': '500',
			'subsample_seed': '0',
			'delta_accuracy': '0.01',
			'delta_balanced_accuracy': '0.01',
			'delta_macro_f1': '0.02',
			'delta_weighted_f1': '0.01',
			'delta_mean_iou': '0.02',
		},
		{
			'budget_id': 'full',
			'per_class_cap': '',
			'subsample_seed': '0',
			'delta_accuracy': '0.01',
			'delta_balanced_accuracy': '-0.02',
			'delta_macro_f1': '0.02',
			'delta_weighted_f1': '0.01',
			'delta_mean_iou': '0.03',
		},
		{
			'budget_id': 'full',
			'per_class_cap': '',
			'subsample_seed': '1',
			'delta_accuracy': '0.01',
			'delta_balanced_accuracy': '-0.02',
			'delta_macro_f1': '0.02',
			'delta_weighted_f1': '0.01',
			'delta_mean_iou': '0.03',
		},
	]
	_write_csv(
		root / 'reports' / 'paired_deltas.csv',
		(
			'budget_id',
			'per_class_cap',
			'subsample_seed',
			'delta_accuracy',
			'delta_balanced_accuracy',
			'delta_macro_f1',
			'delta_weighted_f1',
			'delta_mean_iou',
		),
		rows,
	)
	manifest_rows = []
	for row in rows:
		for role in ('baseline', 'candidate'):
			identity = f'hash-{row["budget_id"]}-{row["subsample_seed"]}'
			if duplicate_full_identity and row['budget_id'] == 'full':
				identity = 'hash-full-duplicated'
			manifest_row = {
				'model_role': role,
				'budget_id': row['budget_id'],
				'subsample_seed': int(row['subsample_seed']),
				'paired_identity_hash': identity,
			}
			manifest_rows.append(manifest_row)
	_write_json(root / 'suite_manifest.json', {'rows': manifest_rows})


def _write_split_index(root: Path, *, single_split: bool = False) -> None:
	rows = [
		{
			'split_id': 'split_001',
			'delta_accuracy': '0.01',
			'delta_balanced_accuracy': '-0.01',
			'delta_macro_f1': '0.02',
			'delta_weighted_f1': '0.01',
			'delta_mean_iou': '0.03',
		},
		{
			'split_id': 'split_000',
			'delta_accuracy': '0.02',
			'delta_balanced_accuracy': '0.02',
			'delta_macro_f1': '0.03',
			'delta_weighted_f1': '0.02',
			'delta_mean_iou': '0.04',
		},
	]
	_write_csv(
		root / 'reports' / 'split_paired_deltas.csv',
		(
			'split_id',
			'delta_accuracy',
			'delta_balanced_accuracy',
			'delta_macro_f1',
			'delta_weighted_f1',
			'delta_mean_iou',
		),
		rows[:1] if single_split else rows,
	)


def _require_matplotlib_agg() -> None:
	matplotlib = pytest.importorskip('matplotlib')
	matplotlib.use('Agg', force=True)


def _write_csv(
	path: Path,
	fieldnames: tuple[str, ...],
	rows: list[dict[str, str]],
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
	with path.open(newline='', encoding='utf-8') as handle:
		return list(csv.DictReader(handle))


def _append_csv_row(path: Path, row: dict[str, str]) -> None:
	with path.open('a', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=tuple(row))
		writer.writerow(row)


def _write_json(path: Path, payload: dict[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
