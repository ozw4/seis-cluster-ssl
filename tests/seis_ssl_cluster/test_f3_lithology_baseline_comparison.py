from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from seis_ssl_cluster.f3 import (
	F3LithologyComparisonReportConfig,
	build_f3_lithology_comparison_report,
)
from tests.helpers import run_python_proc


def test_f3_lithology_baseline_comparison_writes_table_report_and_figures(
	tmp_path: Path,
) -> None:
	pytest.importorskip('matplotlib.pyplot')
	search_root = _search_root(tmp_path)
	_write_probe_metrics(
		search_root,
		feature_kind='pretrained_encoder',
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		embed_spec='overlap_x16',
		macro_f1=0.72,
		mean_iou=0.55,
		per_class_f1={'3': 0.31, '5': 0.54},
	)
	_write_probe_metrics(
		search_root,
		feature_kind='z_only',
		baseline_tag='z_only_v1',
		embed_spec='z_only_degree1',
		macro_f1=0.60,
		mean_iou=0.43,
		per_class_f1={'3': 0.22, '5': 0.36},
	)
	_write_probe_metrics(
		search_root,
		feature_kind='amplitude_stats',
		baseline_tag='amplitude_stats_v1',
		embed_spec='amplitude_stats_v1',
		macro_f1=0.64,
		mean_iou=0.47,
		per_class_f1={'3': 0.25, '5': 0.41},
	)
	_write_probe_metrics(
		search_root,
		feature_kind='random_encoder',
		baseline_tag='random_encoder_amp_mae_seed42_v1',
		embed_spec='overlap_x16',
		macro_f1=0.58,
		mean_iou=0.40,
		per_class_f1={'3': 0.19, '5': 0.33},
	)
	output_dir = search_root / 'reports' / 'baseline_comparison'

	result = build_f3_lithology_comparison_report(
		F3LithologyComparisonReportConfig(
			search_root=search_root,
			output_csv=output_dir / 'comparison_table.csv',
			output_markdown=output_dir / 'comparison_report.md',
		),
	)

	rows = _read_csv(result.comparison_csv)
	markdown = result.comparison_markdown.read_text(encoding='utf-8')

	assert [row['feature_kind'] for row in rows] == [
		'pretrained_encoder',
		'z_only',
		'amplitude_stats',
		'random_encoder',
	]
	assert rows[0]['MODEL_TAG'] == (
		'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	)
	assert rows[1]['BASELINE_TAG'] == 'z_only_v1'
	assert rows[1]['MODEL_TAG'] == ''
	assert rows[1]['EMBED_SPEC'] == 'z_only_degree1'
	assert rows[3]['BASELINE_TAG'] == 'random_encoder_amp_mae_seed42_v1'
	assert 'class_3_f1' in rows[0]
	assert 'class_5_f1' in rows[0]
	assert len(result.figure_paths) == 3
	for figure_path in result.figure_paths:
		assert figure_path.is_file()
		assert figure_path.parent == output_dir / 'figures'
	assert 'pretrained encoderがz-onlyを上回るか' in markdown
	assert 'macro F1差分 +0.1200' in markdown
	assert 'pretrained encoderがrandom encoderを上回るか' in markdown
	assert 'class 5: F1差分 +0.1300' in markdown
	assert 'F3 faciesが深度だけで説明できる程度' in markdown


def test_f3_lithology_baseline_comparison_warns_for_missing_metrics(
	tmp_path: Path,
) -> None:
	search_root = _search_root(tmp_path)
	metrics_path = _write_probe_metrics(
		search_root,
		feature_kind='z_only',
		baseline_tag='z_only_v1',
		embed_spec='z_only_degree1',
		macro_f1=0.60,
		mean_iou=0.43,
		per_class_f1={'5': 0.36},
		metrics_override={'accuracy': 0.5},
	)
	missing_path = search_root / 'missing' / 'probes' / 'linear' / 'metrics.json'
	output_dir = search_root / 'reports' / 'baseline_comparison'

	result = build_f3_lithology_comparison_report(
		F3LithologyComparisonReportConfig(
			search_root=search_root,
			output_csv=output_dir / 'comparison_table.csv',
			output_markdown=output_dir / 'comparison_report.md',
			metrics_paths=(metrics_path, missing_path),
		),
	)

	assert len(result.rows) == 1
	assert any('comparison metrics missing key(s)' in item for item in result.warnings)
	assert any('missing input report component' in item for item in result.warnings)
	assert 'missing input report component' in result.comparison_markdown.read_text(
		encoding='utf-8',
	)


def test_build_f3_lithology_comparison_report_proc_dry_run(tmp_path: Path) -> None:
	output_dir = tmp_path / 'out' / 'baseline_comparison'

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/build_f3_lithology_comparison_report.py'),
		'--search-root',
		_search_root(tmp_path),
		'--output-dir',
		output_dir,
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert 'stage: build_f3_lithology_comparison_report' in result.stdout
	assert (
		f'comparison.output_csv: {output_dir / "comparison_table.csv"}'
		in result.stdout
	)
	assert 'execution: dry-run; F3 lithology comparison report skipped' in result.stdout


def test_build_f3_lithology_comparison_report_proc_dry_run_with_config(
	tmp_path: Path,
) -> None:
	search_root = _search_root(tmp_path)
	output_dir = tmp_path / 'out' / 'baseline_comparison'
	config_path = tmp_path / 'comparison.yaml'
	config_path.write_text(
		f"""
paths:
  artifact_root: {tmp_path / 'artifacts' / 'seis_ssl_cluster'}
dataset:
  name: f3_facies_benchmark
  version: facies_benchmark_v1
comparison:
  search_root: {search_root}
  output_dir: {output_dir}
  output_csv: {output_dir / 'comparison_table.csv'}
  output_markdown: {output_dir / 'comparison_report.md'}
  figure_dpi: 300
""".lstrip(),
		encoding='utf-8',
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/build_f3_lithology_comparison_report.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert 'stage: build_f3_lithology_comparison_report' in result.stdout
	assert f'comparison.search_root: {search_root}' in result.stdout
	assert (
		f'comparison.output_csv: {output_dir / "comparison_table.csv"}'
		in result.stdout
	)
	assert 'comparison.figure_dpi: 300' in result.stdout


def test_f3_lithology_probe_joblib_artifacts_are_gitignored() -> None:
	gitignore = Path('.gitignore').read_text(encoding='utf-8')

	assert '*.joblib' in gitignore


def _search_root(root: Path) -> Path:
	return (
		root
		/ 'artifacts'
		/ 'seis_ssl_cluster'
		/ 'lithology'
		/ 'f3'
		/ 'facies_benchmark_v1'
	)


def _write_probe_metrics(  # noqa: PLR0913
	search_root: Path,
	*,
	feature_kind: str,
	embed_spec: str,
	macro_f1: float,
	mean_iou: float,
	per_class_f1: dict[str, float],
	model_tag: str | None = None,
	baseline_tag: str | None = None,
	metrics_override: dict[str, object] | None = None,
) -> Path:
	label_set = 'png_slices_segy_labels_v1'
	probe_spec = 'linear_balanced_v1'
	run_root = _run_root(
		search_root,
		feature_kind=feature_kind,
		model_tag=model_tag,
		baseline_tag=baseline_tag,
		embed_spec=embed_spec,
		label_set=label_set,
	)
	probe_dir = run_root / 'probes' / probe_spec
	token_metadata = run_root / 'token_dataset' / 'token_dataset_metadata.json'
	feature_source = {
		'kind': feature_kind,
		'baseline_tag': baseline_tag,
		'embedding_spec': embed_spec,
	}
	metrics = (
		dict(metrics_override)
		if metrics_override is not None
		else {
			'accuracy': macro_f1,
			'balanced_accuracy': macro_f1,
			'macro_f1': macro_f1,
			'weighted_f1': macro_f1,
			'mean_iou': mean_iou,
			'per_class_f1': per_class_f1,
			'class_names': {'3': 'Scruff', '5': 'Zechstein'},
			'feature_source': feature_source,
		}
	)
	_write_json(probe_dir / 'metrics.json', metrics)
	_write_json(
		probe_dir / 'probe_config_resolved.json',
		{
			'model': {'tag': model_tag or baseline_tag},
			'embeddings': {'spec': embed_spec},
			'labels': {'set': label_set},
			'probe': {'spec': probe_spec},
			'inputs': {'token_dataset_metadata_json': str(token_metadata)},
			'token_dataset': {'feature_source': feature_source},
		},
	)
	_write_json(token_metadata, {'feature_source': feature_source})
	return probe_dir / 'metrics.json'


def _run_root(  # noqa: PLR0913
	search_root: Path,
	*,
	feature_kind: str,
	model_tag: str | None,
	baseline_tag: str | None,
	embed_spec: str,
	label_set: str,
) -> Path:
	if feature_kind in {'z_only', 'amplitude_stats'}:
		return search_root / 'baselines' / str(baseline_tag) / label_set
	tag = model_tag or baseline_tag
	return search_root / str(tag) / embed_spec / label_set


def _read_csv(path: Path) -> list[dict[str, str]]:
	with path.open(encoding='utf-8', newline='') as file_obj:
		return list(csv.DictReader(file_obj))


def _write_json(path: Path, payload: dict[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)
