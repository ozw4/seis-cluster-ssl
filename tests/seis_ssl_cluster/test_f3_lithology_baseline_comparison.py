from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import seis_ssl_cluster.f3.lithology_report as lithology_report_module

from proc.seis_ssl_cluster.build_f3_lithology_comparison_report import (
	f3_lithology_comparison_report_config_from_mapping,
)
from seis_ssl_cluster.f3 import (
	F3LithologyComparisonPublishConfig,
	F3LithologyComparisonReportConfig,
	build_f3_lithology_comparison_report,
	default_f3_lithology_comparison_figure_style,
	publish_f3_lithology_comparison_report,
)
from tests.helpers import run_python_proc


def test_f3_lithology_comparison_default_figure_style() -> None:
	style = default_f3_lithology_comparison_figure_style()

	assert style.font_sizes.title == 10
	assert style.font_sizes.axis_label == 9
	assert style.font_sizes.tick == 8
	assert style.font_sizes.legend == 8
	assert style.font_sizes.bar_label == 7
	assert style.figsize.metric == (6.5, 3.6)
	assert style.figsize.per_class == (8.0, 4.2)


def test_f3_lithology_metric_plot_uses_short_labels_and_unit_range(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	plt = pytest.importorskip('matplotlib.pyplot')
	figure_module = pytest.importorskip('matplotlib.figure')
	style = default_f3_lithology_comparison_figure_style()
	output_png = tmp_path / 'macro_f1_comparison.png'
	rows = [
		{
			'feature_kind': 'pretrained_encoder',
			'MODEL_TAG': 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
			'macro_f1': 0.72,
		},
		{
			'feature_kind': 'z_only',
			'BASELINE_TAG': 'z_only_v1',
			'macro_f1': 0.60,
		},
		{
			'feature_kind': 'random_encoder',
			'BASELINE_TAG': 'random_encoder_amp_mae_seed42_v1',
			'macro_f1': 0.58,
		},
	]
	closed_figures = []
	savefig_kwargs = []
	original_close = plt.close
	original_savefig = figure_module.Figure.savefig

	def _record_savefig(self: object, *args: object, **kwargs: object) -> object:
		savefig_kwargs.append(dict(kwargs))
		return original_savefig(self, *args, **kwargs)

	def _record_close(figure: object | None = None) -> None:
		if hasattr(figure, 'axes'):
			closed_figures.append(figure)

	monkeypatch.setattr(figure_module.Figure, 'savefig', _record_savefig)
	monkeypatch.setattr(plt, 'close', _record_close)

	lithology_report_module._save_metric_comparison_bar(
		rows,
		metric='macro_f1',
		title='Macro F1',
		ylabel='Macro F1',
		output_png=output_png,
		plt=plt,
		dpi=300,
		style=style,
	)

	axis = closed_figures[0].axes[0]
	tick_labels = [tick.get_text() for tick in axis.get_xticklabels()]
	assert tick_labels == ['Pretrained', 'Z only', 'Random encoder']
	assert all('amp_mae_m075' not in label for label in tick_labels)
	assert axis.get_ylim() == (0.0, 1.0)
	assert savefig_kwargs[0]['dpi'] == 300
	assert savefig_kwargs[0]['bbox_inches'] == 'tight'
	assert output_png.stat().st_size > 0
	original_close(closed_figures[0])


def test_f3_lithology_per_class_plot_uses_display_legend_and_unit_range(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	plt = pytest.importorskip('matplotlib.pyplot')
	style = default_f3_lithology_comparison_figure_style()
	output_png = tmp_path / 'per_class_f1_comparison.png'
	rows = [
		{
			'feature_kind': 'pretrained_encoder',
			'MODEL_TAG': 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
			'class_3_f1': 0.31,
			'class_5_f1': 0.54,
			'_class_names': {'3': 'Scruff', '5': 'Zechstein'},
		},
		{
			'feature_kind': 'amplitude_stats',
			'BASELINE_TAG': 'amplitude_stats_v1',
			'class_3_f1': 0.25,
			'class_5_f1': 0.41,
			'_class_names': {'3': 'Scruff', '5': 'Zechstein'},
		},
	]
	closed_figures = []
	original_close = plt.close

	def _record_close(figure: object | None = None) -> None:
		if hasattr(figure, 'axes'):
			closed_figures.append(figure)

	monkeypatch.setattr(plt, 'close', _record_close)

	lithology_report_module._save_per_class_f1_comparison(
		rows,
		output_png=output_png,
		plt=plt,
		dpi=300,
		style=style,
	)

	axis = closed_figures[0].axes[0]
	legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
	assert legend_labels == ['Pretrained', 'Amplitude stats']
	assert axis.get_ylim() == (0.0, 1.0)
	assert output_png.stat().st_size > 0
	original_close(closed_figures[0])


def test_f3_lithology_comparison_config_accepts_optional_figure_style(
	tmp_path: Path,
) -> None:
	search_root = _search_root(tmp_path)
	output_dir = tmp_path / 'out' / 'baseline_comparison'

	config = f3_lithology_comparison_report_config_from_mapping(
		{
			'paths': {
				'artifact_root': str(tmp_path / 'artifacts' / 'seis_ssl_cluster'),
			},
			'dataset': {'version': 'facies_benchmark_v1'},
			'comparison': {
				'search_root': str(search_root),
				'output_dir': str(output_dir),
				'figures': {
					'dpi': 360,
					'font_sizes': {
						'title': 11,
						'axis_label': 10,
						'tick': 9,
						'legend': 8,
						'bar_label': 7,
					},
					'figsize': {
						'metric': [7.0, 3.8],
						'per_class': [8.5, 4.4],
					},
				},
			},
		},
	)

	assert config.figure_dpi == 360
	assert config.figure_style.font_sizes.title == 11
	assert config.figure_style.font_sizes.axis_label == 10
	assert config.figure_style.font_sizes.tick == 9
	assert config.figure_style.figsize.metric == (7.0, 3.8)
	assert config.figure_style.figsize.per_class == (8.5, 4.4)


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
		feature_kind='xyz_coordinates',
		baseline_tag='xyz_coordinates_v1',
		embed_spec='xyz_coordinates_degree1',
		macro_f1=0.62,
		mean_iou=0.45,
		per_class_f1={'3': 0.24, '5': 0.38},
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
	publish_dir = (
		tmp_path
		/ 'results'
		/ 'f3'
		/ 'facies_benchmark_v1'
		/ 'baseline_comparison'
	)
	output_dir.mkdir(parents=True, exist_ok=True)
	_write_json(output_dir / 'comparison_table.json', {'rows': []})
	(output_dir / 'encoder.pt').write_bytes(b'heavy')
	(output_dir / 'embeddings.npy').write_bytes(b'heavy')
	(output_dir / 'probe.joblib').write_bytes(b'heavy')

	result = build_f3_lithology_comparison_report(
		F3LithologyComparisonReportConfig(
			search_root=search_root,
			output_csv=output_dir / 'comparison_table.csv',
			output_markdown=output_dir / 'comparison_report.md',
		),
		publish_config=F3LithologyComparisonPublishConfig(
			enabled=True,
			output_dir=publish_dir,
			include_figures=True,
		),
	)

	rows = _read_csv(result.comparison_csv)
	markdown = result.comparison_markdown.read_text(encoding='utf-8')
	published_files = {
		path.relative_to(publish_dir)
		for path in publish_dir.rglob('*')
		if path.is_file()
	}

	assert [row['feature_kind'] for row in rows] == [
		'pretrained_encoder',
		'z_only',
		'xyz_coordinates',
		'amplitude_stats',
		'random_encoder',
	]
	assert rows[0]['MODEL_TAG'] == (
		'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	)
	assert rows[1]['BASELINE_TAG'] == 'z_only_v1'
	assert rows[1]['MODEL_TAG'] == ''
	assert rows[1]['EMBED_SPEC'] == 'z_only_degree1'
	assert rows[2]['BASELINE_TAG'] == 'xyz_coordinates_v1'
	assert rows[2]['MODEL_TAG'] == ''
	assert rows[2]['EMBED_SPEC'] == 'xyz_coordinates_degree1'
	assert rows[4]['BASELINE_TAG'] == 'random_encoder_amp_mae_seed42_v1'
	assert 'class_3_f1' in rows[0]
	assert 'class_5_f1' in rows[0]
	assert len(result.figure_paths) == 3
	for figure_path in result.figure_paths:
		assert figure_path.is_file()
		assert figure_path.stat().st_size > 0
		assert figure_path.parent == output_dir / 'figures'
	assert 'pretrained encoderがz-onlyを上回るか' in markdown
	assert 'macro F1差分 +0.1200' in markdown
	assert 'pretrained encoderがxyz-coordinateを上回るか' in markdown
	assert 'pretrained encoderがrandom encoderを上回るか' in markdown
	assert 'class 5: F1差分 +0.1300' in markdown
	assert 'F3 faciesが深度だけで説明できる程度' in markdown
	assert result.publish_manifest is not None
	assert published_files == {
		Path('comparison_report.md'),
		Path('comparison_table.csv'),
		Path('comparison_table.json'),
		Path('publish_manifest.json'),
		Path('figures/macro_f1_comparison.png'),
		Path('figures/mean_iou_comparison.png'),
		Path('figures/per_class_f1_comparison.png'),
	}
	assert not any(
		path.suffix in {'.pt', '.npy', '.npz', '.joblib', '.pkl'}
		for path in published_files
	)
	published_table = publish_dir / 'comparison_table.csv'
	manifest_payload = json.loads(
		(publish_dir / 'publish_manifest.json').read_text(encoding='utf-8'),
	)
	table_entry = next(
		item
		for item in manifest_payload['items']
		if Path(item['target']).name == 'comparison_table.csv'
	)
	table_bytes = published_table.read_bytes()
	assert table_bytes == result.comparison_csv.read_bytes()
	assert table_entry['size_bytes'] == len(table_bytes)
	assert table_entry['sha256'] == hashlib.sha256(table_bytes).hexdigest()


def test_f3_lithology_baseline_comparison_publish_warns_for_missing_optional_figure(
	tmp_path: Path,
) -> None:
	output_dir = tmp_path / 'artifacts' / 'comparison'
	publish_dir = tmp_path / 'results' / 'comparison'
	output_dir.mkdir(parents=True)
	(output_dir / 'comparison_report.md').write_text('# report\n', encoding='utf-8')
	(output_dir / 'comparison_table.csv').write_text(
		'feature_kind,macro_f1\npretrained_encoder,0.7\n',
		encoding='utf-8',
	)

	manifest = publish_f3_lithology_comparison_report(
		F3LithologyComparisonReportConfig(
			search_root=tmp_path / 'artifacts',
			output_csv=output_dir / 'comparison_table.csv',
			output_markdown=output_dir / 'comparison_report.md',
		),
		F3LithologyComparisonPublishConfig(
			enabled=True,
			output_dir=publish_dir,
			include_figures=True,
		),
	)

	assert manifest is not None
	assert (publish_dir / 'comparison_report.md').is_file()
	assert (publish_dir / 'comparison_table.csv').is_file()
	assert any(
		'optional publish source does not exist' in warning
		and 'macro_f1_comparison.png' in warning
		for warning in manifest.warnings
	)
	assert not (publish_dir / 'figures/macro_f1_comparison.png').exists()


def test_f3_lithology_baseline_comparison_uses_pretrained_token_metadata(
	tmp_path: Path,
) -> None:
	search_root = _search_root(tmp_path)
	model_tag = 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	embed_spec = 'overlap_x16'
	label_set = 'png_slices_segy_labels_v1'
	probe_spec = 'linear_balanced_v1'
	run_root = _run_root(
		search_root,
		feature_kind='pretrained_encoder',
		model_tag=model_tag,
		baseline_tag=None,
		embed_spec=embed_spec,
		label_set=label_set,
	)
	probe_dir = run_root / 'probes' / probe_spec
	token_metadata = run_root / 'token_dataset' / 'token_dataset_metadata.json'
	feature_source = {
		'kind': 'pretrained_encoder',
		'reference_model_tag': model_tag,
		'embedding_spec': embed_spec,
		'description': 'pretrained MAE encoder embedding',
	}
	_write_json(
		probe_dir / 'metrics.json',
		{
			'accuracy': 0.72,
			'balanced_accuracy': 0.71,
			'macro_f1': 0.70,
			'weighted_f1': 0.73,
			'mean_iou': 0.52,
			'per_class_f1': {'5': 0.54},
			'class_names': {'5': 'Zechstein'},
		},
	)
	_write_json(
		probe_dir / 'probe_config_resolved.json',
		{
			'model': {'tag': model_tag},
			'embeddings': {'spec': embed_spec},
			'labels': {'set': label_set},
			'probe': {'spec': probe_spec},
			'inputs': {'token_dataset_metadata_json': str(token_metadata)},
		},
	)
	_write_json(token_metadata, {'feature_source': feature_source})
	output_dir = search_root / 'reports' / 'baseline_comparison'

	result = build_f3_lithology_comparison_report(
		F3LithologyComparisonReportConfig(
			search_root=search_root,
			output_csv=output_dir / 'comparison_table.csv',
			output_markdown=output_dir / 'comparison_report.md',
			metrics_paths=(probe_dir / 'metrics.json',),
		),
	)

	rows = _read_csv(result.comparison_csv)
	markdown = result.comparison_markdown.read_text(encoding='utf-8')
	assert rows[0]['FEATURE_SOURCE_REFERENCE_MODEL_TAG'] == model_tag
	assert rows[0]['FEATURE_SOURCE_DESCRIPTION'] == (
		'pretrained MAE encoder embedding'
	)
	assert 'pretrained MAE encoder embedding' in markdown
	table_row = next(
		line
		for line in markdown.splitlines()
		if line.startswith('| pretrained_encoder |')
	)
	assert '未確認' not in table_row


def test_f3_lithology_baseline_comparison_publish_requires_comparison_table(
	tmp_path: Path,
) -> None:
	output_dir = tmp_path / 'artifacts' / 'comparison'
	output_dir.mkdir(parents=True)
	(output_dir / 'comparison_report.md').write_text('# report\n', encoding='utf-8')

	with pytest.raises(FileNotFoundError, match='required publish source'):
		publish_f3_lithology_comparison_report(
			F3LithologyComparisonReportConfig(
				search_root=tmp_path / 'artifacts',
				output_csv=output_dir / 'comparison_table.csv',
				output_markdown=output_dir / 'comparison_report.md',
			),
			F3LithologyComparisonPublishConfig(
				enabled=True,
				output_dir=tmp_path / 'results' / 'comparison',
			),
		)


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
publish:
  enabled: true
  output_dir: {tmp_path / 'results' / 'baseline_comparison'}
  include_figures: true
  max_file_size_mb: 10
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
	assert 'publish.enabled: True' in result.stdout
	assert f"publish.output_dir: {tmp_path / 'results' / 'baseline_comparison'}" in (
		result.stdout
	)
	assert 'publish.include_figures: True' in result.stdout
	assert 'publish.max_file_size_bytes: 10485760' in result.stdout


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
	if feature_kind in {'z_only', 'xyz_coordinates', 'amplitude_stats'}:
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
