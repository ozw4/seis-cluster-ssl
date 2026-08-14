from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from seis_ssl_cluster.config.f3_lithology import (
	f3_lithology_report_config_from_mapping,
)
from seis_ssl_cluster.f3 import (
	F3LithologyComparisonReportConfig,
	F3LithologyPublishConfig,
	F3LithologyReportConfig,
	build_f3_lithology_comparison_report,
	build_f3_lithology_report,
)


def test_f3_lithology_report_outputs_markdown_json_and_relative_links(
	tmp_path: Path,
) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
	)
	config = _report_config(run)

	result = build_f3_lithology_report(config)

	payload = json.loads(result.report_json.read_text(encoding='utf-8'))
	markdown = result.report_markdown.read_text(encoding='utf-8')

	assert result.report_markdown == config.output_markdown
	assert result.report_json == config.output_json
	assert payload['artifact_type'] == 'f3_lithology_probe_report'
	assert payload['pretrained_encoder']['MODEL_TAG'] == (
		'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	)
	assert payload['pretrained_encoder']['EMBED_SPEC'] == 'overlap_x16'
	assert payload['pretrained_encoder']['agc_enabled'] is True
	assert payload['pretrained_encoder']['visible_loss_enabled'] is True
	assert payload['pretrained_encoder']['mask_ratio'] == 0.75
	assert payload['probe']['classifier_type'] == 'logistic_regression'
	assert payload['probe']['feature_scaling'] == 'standard'
	assert payload['probe']['class_weighting'] == 'balanced'
	assert payload['probe']['hyperparameters']['max_iter'] == 2000
	assert payload['prediction_summary'] == {'valid_token_count': 16}
	assert payload['dataset']['class_imbalance'] == {
		'class_counts': {'0': 10, '5': 6},
		'max_to_min_positive_ratio': 10 / 6,
		'total': 16,
	}
	assert (
		'- class imbalance: {"class_counts": {"0": 10, "5": 6}, '
		'"max_to_min_positive_ratio": 1.6666666666666667, "total": 16}'
		in markdown
	)
	for section in (
		'## Dataset',
		'## Pretrained encoder',
		'## Token dataset',
		'## Probe',
		'## Metrics',
		'## Figures',
		'## Interpretation',
	):
		assert section in markdown
	assert '### 良い点' in markdown
	assert '### AGCあり/なし比較' in markdown
	assert (
		'[confusion_matrix]'
		'(../../probes/linear_balanced_v1/figures/confusion_matrix.png)'
		in markdown
	)
	assert (
		'[per_class_f1]'
		'(../../probes/linear_balanced_v1/figures/per_class_f1.png)'
		in markdown
	)
	assert (
		'[validation_slice_inline_250]'
		'(../../visualizations/linear_balanced_v1/'
		'validation_inline_0250_prediction.png)'
		in markdown
	)
	assert str(tmp_path) not in _figures_section(markdown)


def test_m1_structured_report_uses_token_dataset_counts_for_class_imbalance(
	tmp_path: Path,
) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
	)
	token_dir = run['lithology_root'] / 'token_dataset'
	train_tokens = token_dir / 'train_tokens.npz'
	validation_tokens = token_dir / 'validation_tokens.npz'
	_write_token_npz(train_tokens, np.asarray([0, 5, 5], dtype=np.int64))
	_write_token_npz(validation_tokens, np.asarray([0, 0], dtype=np.int64))
	probe_config = json.loads(
		run['probe_config_json'].read_text(encoding='utf-8'),
	)
	probe_config['inputs']['train_tokens'] = str(train_tokens)
	probe_config['inputs']['validation_tokens'] = str(validation_tokens)
	_write_json(run['probe_config_json'], probe_config)

	result = build_f3_lithology_report(_report_config(run))

	payload = json.loads(result.report_json.read_text(encoding='utf-8'))
	token_dataset = payload['token_dataset']
	assert token_dataset['train_token_count'] == 3
	assert token_dataset['validation_token_count'] == 2
	assert token_dataset['class_counts']['train'] == {'0': 1, '5': 2}
	assert token_dataset['class_counts']['validation'] == {'0': 2}
	assert token_dataset['class_imbalance'] == {
		'class_counts': {'0': 3, '5': 2},
		'max_to_min_positive_ratio': 1.5,
		'total': 5,
	}
	assert payload['dataset']['class_imbalance'] == {
		'class_counts': {'0': 3, '5': 2},
		'max_to_min_positive_ratio': 1.5,
		'total': 5,
	}


def test_f3_lithology_report_uses_available_split_counts(
	tmp_path: Path,
) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
	)
	for path in (
		Path(run['lithology_root']) / 'token_dataset/token_dataset_metadata.json',
		Path(run['probe_config_json']),
	):
		payload = json.loads(path.read_text(encoding='utf-8'))
		payload['summary'].pop('train_class_counts', None)
		_write_json(path, payload)

	result = build_f3_lithology_report(_report_config(run))

	assert result.payload['dataset']['class_imbalance'] == {
		'class_counts': {'0': 2, '5': 4},
		'max_to_min_positive_ratio': 2.0,
		'total': 6,
	}


def test_f3_lithology_report_warns_when_class_counts_are_missing(
	tmp_path: Path,
) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
	)
	for path in (
		Path(run['lithology_root']) / 'token_dataset/token_dataset_metadata.json',
		Path(run['probe_config_json']),
	):
		payload = json.loads(path.read_text(encoding='utf-8'))
		payload['summary'].pop('train_class_counts', None)
		payload['summary'].pop('validation_class_counts', None)
		_write_json(path, payload)

	result = build_f3_lithology_report(_report_config(run))
	markdown = result.report_markdown.read_text(encoding='utf-8')

	assert result.payload['dataset']['class_imbalance'] == {
		'class_counts': {},
		'max_to_min_positive_ratio': None,
		'total': 0,
	}
	warning = (
		'dataset class imbalance unavailable: '
		'no token dataset class counts were found'
	)
	assert warning in result.payload['warnings']
	assert f'- {warning}' in markdown


def test_f3_lithology_report_ignores_zero_counts_in_imbalance_ratio(
	tmp_path: Path,
) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
	)
	for path in (
		Path(run['lithology_root']) / 'token_dataset/token_dataset_metadata.json',
		Path(run['probe_config_json']),
	):
		payload = json.loads(path.read_text(encoding='utf-8'))
		payload['summary']['train_class_counts'] = {'0': 0, '5': 2}
		payload['summary']['validation_class_counts'] = {'0': 0, '5': 4}
		_write_json(path, payload)

	result = build_f3_lithology_report(_report_config(run))

	assert result.payload['dataset']['class_imbalance'] == {
		'class_counts': {'0': 0, '5': 6},
		'max_to_min_positive_ratio': 1.0,
		'total': 6,
	}


def test_f3_lithology_report_writes_warning_when_metrics_are_missing(
	tmp_path: Path,
) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_vis00_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
		write_metrics=False,
	)

	result = build_f3_lithology_report(_report_config(run))

	payload = json.loads(result.report_json.read_text(encoding='utf-8'))
	markdown = result.report_markdown.read_text(encoding='utf-8')

	assert any(
		'missing input report component: metrics' in warning
		for warning in payload['warnings']
	)
	assert '## Warnings' in markdown
	assert '- missing input report component: metrics' in markdown


def test_f3_lithology_report_publish_writes_lightweight_results(
	tmp_path: Path,
) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
		prediction_figures=(
			('validation', 'inline', 150, True),
			('validation', 'crossline', 350, True),
			('validation', 'crossline', 750, True),
			('validation', 'inline', 900, True),
			('selected', 'inline', 100, True),
		),
	)
	(run['probe_dir'] / 'probe.joblib').write_bytes(b'heavy')
	(run['prediction_dir'] / 'f3_token_predictions.npy').write_bytes(b'heavy')
	output_dir = tmp_path / 'reports' / 'f3' / 'lithology_probe'

	result = build_f3_lithology_report(
		_report_config(run),
		publish_config=F3LithologyPublishConfig(
			enabled=True,
			output_dir=output_dir,
			include_figures=True,
			max_prediction_figures=3,
		),
	)

	assert result.published_files
	published_files = {
		path.relative_to(output_dir)
		for path in output_dir.rglob('*')
		if path.is_file()
	}
	assert published_files == {
		Path('report.md'),
		Path('report.json'),
		Path('metrics.json'),
		Path('metrics.csv'),
		Path('classification_report.md'),
		Path('confusion_matrix.csv'),
		Path('figures/confusion_matrix.png'),
		Path('figures/per_class_f1.png'),
		Path('figures/validation_inline_0150_prediction.png'),
		Path('figures/validation_crossline_0350_prediction.png'),
		Path('figures/validation_crossline_0750_prediction.png'),
	}
	assert {
		path.relative_to(output_dir) for path in result.published_files
	} == published_files
	assert not any(
		path.suffix in {'.joblib', '.npy', '.npz'}
		for path in published_files
	)
	markdown = (output_dir / 'report.md').read_text(encoding='utf-8')
	assert '(figures/confusion_matrix.png)' in markdown
	assert '(figures/validation_inline_0150_prediction.png)' in markdown
	assert 'validation_inline_0900_prediction.png' not in markdown
	assert 'selected_inline_0100_prediction.png' not in markdown
	published_payload = json.loads(
		(output_dir / 'report.json').read_text(encoding='utf-8'),
	)
	assert 'inputs' not in published_payload
	assert 'outputs' not in published_payload
	assert 'comparison' not in published_payload
	assert 'checkpoint_path' not in published_payload['pretrained_encoder']
	assert all('source_path' not in item for item in published_payload['figures'])


def test_f3_lithology_report_publish_warns_for_missing_optional_prediction_figure(
	tmp_path: Path,
) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
		prediction_figures=(
			('validation', 'inline', 150, True),
			('validation', 'crossline', 350, False),
		),
	)
	output_dir = tmp_path / 'reports' / 'f3' / 'lithology_probe'

	result = build_f3_lithology_report(
		_report_config(run),
		publish_config=F3LithologyPublishConfig(
			enabled=True,
			output_dir=output_dir,
			include_figures=True,
			max_prediction_figures=3,
		),
	)

	assert result.published_files
	missing_prediction = output_dir / 'figures/validation_crossline_0350_prediction.png'
	assert not missing_prediction.exists()
	assert missing_prediction not in result.published_files
	markdown = (output_dir / 'report.md').read_text(encoding='utf-8')
	assert 'validation_crossline_0350_prediction.png' not in _figures_section(markdown)


def test_f3_lithology_report_publish_requires_metrics(
	tmp_path: Path,
) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_vis00_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
		write_metrics=False,
	)

	output_dir = tmp_path / 'reports' / 'f3' / 'lithology_probe'
	with pytest.raises(FileNotFoundError, match='required publish source'):
		build_f3_lithology_report(
			_report_config(run),
			publish_config=F3LithologyPublishConfig(
				enabled=True,
				output_dir=output_dir,
			),
		)

	assert not output_dir.exists() or not any(output_dir.rglob('*'))
	local_payload = json.loads(
		(Path(run['report_dir']) / 'report.json').read_text(encoding='utf-8')
	)
	assert 'inputs' in local_payload
	assert 'outputs' in local_payload


def test_f3_lithology_report_publish_can_exclude_figures(tmp_path: Path) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
		prediction_figures=(('validation', 'inline', 150, True),),
	)
	output_dir = tmp_path / 'reports' / 'f3' / 'lithology_probe'

	result = build_f3_lithology_report(
		_report_config(run),
		publish_config=F3LithologyPublishConfig(
			enabled=True,
			output_dir=output_dir,
			include_figures=False,
		),
	)

	assert {path.relative_to(output_dir) for path in result.published_files} == {
		Path('report.md'),
		Path('report.json'),
		Path('metrics.json'),
		Path('metrics.csv'),
		Path('classification_report.md'),
		Path('confusion_matrix.csv'),
	}
	assert not (output_dir / 'figures').exists()
	assert json.loads((output_dir / 'report.json').read_text())['figures'] == []


def test_f3_lithology_report_publish_enforces_size_limit(tmp_path: Path) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
	)
	output_dir = tmp_path / 'reports' / 'f3' / 'lithology_probe'

	with pytest.raises(ValueError, match='exceeds max_file_size_bytes'):
		build_f3_lithology_report(
			_report_config(run),
			publish_config=F3LithologyPublishConfig(
				enabled=True,
				output_dir=output_dir,
				max_file_size_bytes=1,
			),
		)

	assert not output_dir.exists()


def test_f3_lithology_report_publish_rejects_source_symlink(
	tmp_path: Path,
) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
	)
	metrics_json = Path(run['metrics_json'])
	real_metrics_json = metrics_json.with_name('metrics-real.json')
	metrics_json.rename(real_metrics_json)
	metrics_json.symlink_to(real_metrics_json.name)
	output_dir = tmp_path / 'reports' / 'f3' / 'lithology_probe'

	with pytest.raises(FileNotFoundError, match='regular file'):
		build_f3_lithology_report(
			_report_config(run),
			publish_config=F3LithologyPublishConfig(
				enabled=True,
				output_dir=output_dir,
			),
		)

	assert not output_dir.exists()


def test_f3_lithology_report_publish_rejects_source_as_target(
	tmp_path: Path,
) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
	)

	with pytest.raises(ValueError, match='publish target must differ from source'):
		build_f3_lithology_report(
			_report_config(run),
			publish_config=F3LithologyPublishConfig(
				enabled=True,
				output_dir=Path(run['report_dir']),
			),
		)


@pytest.mark.parametrize('target_kind', ['symlink', 'directory'])
def test_f3_lithology_report_publish_rejects_unsafe_target(
	tmp_path: Path,
	target_kind: str,
) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
	)
	output_dir = tmp_path / 'reports' / 'f3' / 'lithology_probe'
	output_dir.mkdir(parents=True)
	report_target = output_dir / 'report.md'
	if target_kind == 'symlink':
		symlink_target = tmp_path / 'existing-report.md'
		symlink_target.write_text('unchanged\n', encoding='utf-8')
		report_target.symlink_to(symlink_target)
		expected_error = ValueError
		expected_message = 'must not be a symlink'
	else:
		report_target.mkdir()
		expected_error = IsADirectoryError
		expected_message = 'is not a file'

	with pytest.raises(expected_error, match=expected_message):
		build_f3_lithology_report(
			_report_config(run),
			publish_config=F3LithologyPublishConfig(
				enabled=True,
				output_dir=output_dir,
			),
		)


def test_f3_lithology_comparison_table_aggregates_multiple_runs(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	_write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_vis00_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
		accuracy=0.55,
	)
	_write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		embed_spec='overlap_x16',
		probe_spec='mlp_balanced_v1',
		accuracy=0.65,
		feature_source={
			'kind': 'random_encoder',
			'reference_model_tag': (
				'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
			),
			'embedding_spec': 'overlap_x16',
			'description': 'fixture random encoder features',
		},
	)
	comparison_dir = root / 'lithology' / 'f3' / 'facies_benchmark_v1' / 'reports'
	config = F3LithologyComparisonReportConfig(
		search_root=root / 'lithology' / 'f3' / 'facies_benchmark_v1',
		output_csv=comparison_dir / 'comparison_table.csv',
		output_markdown=comparison_dir / 'comparison_report.md',
	)

	result = build_f3_lithology_comparison_report(config)

	with result.comparison_csv.open(encoding='utf-8', newline='') as file_obj:
		rows = list(csv.DictReader(file_obj))
	markdown = result.comparison_markdown.read_text(encoding='utf-8')

	assert len(rows) == 2
	assert rows[0]['feature_kind'] == 'pretrained_encoder'
	assert rows[0]['MODEL_TAG'] == 'amp_mae_m075_mse_g0_patchnorm_clip8_vis00_v1'
	assert rows[0]['BASELINE_TAG'] == ''
	assert rows[0]['PROBE_SPEC'] == 'linear_balanced_v1'
	assert rows[1]['feature_kind'] == 'random_encoder'
	assert rows[1]['MODEL_TAG'] == ''
	assert rows[1]['BASELINE_TAG'] == (
		'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	)
	assert rows[1]['PROBE_SPEC'] == 'mlp_balanced_v1'
	assert rows[1]['accuracy'] == '0.65'
	assert 'class_0_f1' in rows[0]
	assert 'class_5_f1' in rows[0]
	assert 'feature_kind' in markdown
	assert '集約run数: 2' in markdown


@pytest.mark.parametrize(
	('output_section', 'output_key', 'collision_section', 'collision_key'),
	[
		('reports', 'output_json', 'probe', 'metrics_json'),
		(
			'reports',
			'output_markdown',
			'reports',
			'token_dataset_metadata_json',
		),
		('comparison', 'output_csv', 'comparison', 'output_markdown'),
		('reports', 'output_json', 'comparison', 'output_csv'),
	],
)
def test_f3_lithology_report_config_rejects_file_path_collisions(
	tmp_path: Path,
	output_section: str,
	output_key: str,
	collision_section: str,
	collision_key: str,
) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='model_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
	)
	raw = _report_config_mapping(tmp_path, run)
	raw[output_section][output_key] = raw[collision_section][collision_key]

	with pytest.raises(
		ValueError,
		match=rf'{output_section}\.{output_key}.*differ',
	):
		f3_lithology_report_config_from_mapping(raw)


def test_f3_lithology_report_config_preserves_explicit_comparison_paths(
	tmp_path: Path,
) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='model_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
	)
	raw = _report_config_mapping(tmp_path, run)
	raw['paths']['artifact_root'] = str(tmp_path / 'unrelated-artifact-root')
	raw['dataset']['version'] = 'unrelated_version'

	resolved = f3_lithology_report_config_from_mapping(raw)

	assert resolved.metrics_json == Path(raw['probe']['metrics_json'])
	assert resolved.probe_config_json == Path(
		raw['probe']['probe_config_resolved_json']
	)
	assert resolved.token_dataset_metadata_json == Path(
		raw['reports']['token_dataset_metadata_json']
	)
	assert resolved.prediction_metadata_json == Path(
		raw['predictions']['metadata_json']
	)
	assert resolved.visualization_metadata_json == Path(
		raw['visualizations']['metadata_json']
	)
	assert resolved.output_markdown == Path(raw['reports']['output_markdown'])
	assert resolved.output_json == Path(raw['reports']['output_json'])
	assert resolved.comparison is not None
	assert resolved.comparison.search_root == Path(raw['comparison']['search_root'])
	assert resolved.comparison.output_csv == Path(raw['comparison']['output_csv'])
	assert resolved.comparison.output_markdown == Path(
		raw['comparison']['output_markdown']
	)


def _report_config(run: dict[str, object]) -> F3LithologyReportConfig:
	return F3LithologyReportConfig(
		output_dir=Path(run['report_dir']),
		output_markdown=Path(run['report_dir']) / 'report.md',
		output_json=Path(run['report_dir']) / 'report.json',
		metrics_json=Path(run['metrics_json']),
		probe_config_json=Path(run['probe_config_json']),
		prediction_metadata_json=Path(run['prediction_metadata_json']),
		visualization_metadata_json=Path(run['visualization_metadata_json']),
		dataset={
			'name': 'f3_facies_benchmark',
			'version': 'facies_benchmark_v1',
		},
		model={
			'tag': run['model_tag'],
			'checkpoint': str(run['checkpoint']),
			'freeze_encoder': True,
		},
		labels={
			'set': 'png_slices_segy_labels_v1',
			'png_label_role': 'train_validation_slice_selection_and_visual_qc',
		},
		lithology={'root': str(run['lithology_root'])},
		probe={'spec': run['probe_spec'], 'metrics_json': str(run['metrics_json'])},
	)


def _report_config_mapping(
	tmp_path: Path,
	run: dict[str, object],
) -> dict[str, object]:
	report_dir = Path(run['report_dir'])
	comparison_dir = report_dir / 'comparison'
	return {
		'paths': {
			'f3_root': str(tmp_path / 'F3'),
			'artifact_root': str(run['artifact_root']),
		},
		'dataset': {
			'name': 'f3_facies_benchmark',
			'version': 'facies_benchmark_v1',
		},
		'model': {'tag': run['model_tag']},
		'labels': {'set': 'png_slices_segy_labels_v1'},
		'lithology': {'root': str(run['lithology_root'])},
		'probe': {
			'spec': run['probe_spec'],
			'metrics_json': str(run['metrics_json']),
			'probe_config_resolved_json': str(run['probe_config_json']),
		},
		'predictions': {'metadata_json': str(run['prediction_metadata_json'])},
		'visualizations': {
			'metadata_json': str(run['visualization_metadata_json']),
		},
		'reports': {
			'output_dir': str(report_dir),
			'output_markdown': str(report_dir / 'report.md'),
			'output_json': str(report_dir / 'report.json'),
			'token_dataset_metadata_json': str(
				Path(run['lithology_root'])
				/ 'token_dataset'
				/ 'token_dataset_metadata.json'
			),
		},
		'comparison': {
			'search_root': str(run['lithology_root']),
			'output_dir': str(comparison_dir),
			'output_csv': str(comparison_dir / 'comparison.csv'),
			'output_markdown': str(comparison_dir / 'comparison.md'),
		},
	}


def _write_probe_run(  # noqa: PLR0913
	root: Path,
	*,
	model_tag: str,
	embed_spec: str,
	probe_spec: str,
	accuracy: float = 0.625,
	write_metrics: bool = True,
	feature_source: dict[str, object] | None = None,
	prediction_figures: tuple[tuple[str, str, int, bool], ...] | None = None,
) -> dict[str, object]:
	artifact_root = root / 'artifacts' / 'seis_ssl_cluster'
	label_set = 'png_slices_segy_labels_v1'
	lithology_root = (
		artifact_root
		/ 'lithology'
		/ 'f3'
		/ 'facies_benchmark_v1'
		/ model_tag
		/ embed_spec
		/ label_set
	)
	probe_dir = lithology_root / 'probes' / probe_spec
	report_dir = lithology_root / 'reports' / probe_spec
	prediction_dir = lithology_root / 'predictions' / probe_spec
	visualization_dir = lithology_root / 'visualizations' / probe_spec
	token_metadata_json = (
		lithology_root / 'token_dataset' / 'token_dataset_metadata.json'
	)
	metrics_json = probe_dir / 'metrics.json'
	probe_config_json = probe_dir / 'probe_config_resolved.json'
	checkpoint = (
		artifact_root
		/ 'pretraining'
		/ 'nopims'
		/ 'pretrain_v1'
		/ model_tag
		/ 'full_100ep'
		/ 'mae_best.pt'
	)
	for path in (
		probe_dir / 'figures' / 'confusion_matrix.png',
		probe_dir / 'figures' / 'per_class_f1.png',
	):
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_bytes(b'fake-png')
	if write_metrics:
		_write_json(
			metrics_json,
			_metrics_payload(accuracy=accuracy, feature_source=feature_source),
		)
		(probe_dir / 'metrics.csv').write_text(
			'metric,value\naccuracy,0.625\n',
			encoding='utf-8',
		)
		(probe_dir / 'classification_report.md').write_text(
			'# Classification report\n',
			encoding='utf-8',
		)
		(probe_dir / 'confusion_matrix.csv').write_text(
			'class_id,0,5\n0,8,2\n5,3,3\n',
			encoding='utf-8',
		)
	_write_json(
		token_metadata_json,
		_token_metadata_payload(model_tag, feature_source=feature_source),
	)
	_write_json(
		probe_config_json,
		_probe_config_payload(
			model_tag=model_tag,
			embed_spec=embed_spec,
			probe_spec=probe_spec,
			lithology_root=lithology_root,
			probe_dir=probe_dir,
			token_metadata_json=token_metadata_json,
			checkpoint=checkpoint,
			feature_source=feature_source,
		),
	)
	prediction_metadata_json = prediction_dir / 'prediction_metadata.json'
	_write_json(prediction_metadata_json, {'summary': {'valid_token_count': 16}})
	visualization_metadata_json = visualization_dir / 'metadata.json'
	if prediction_figures is None:
		prediction_figures = (('validation', 'inline', 250, True),)
	visualization_entries = []
	for group, slice_type, slice_index, write_file in prediction_figures:
		path = visualization_dir / (
			f'{group}_{slice_type}_{slice_index:04d}_prediction.png'
		)
		if write_file:
			path.parent.mkdir(parents=True, exist_ok=True)
			path.write_bytes(b'fake-png')
		visualization_entries.append(
			{
				'path': str(path),
				'group': group,
				'slice_type': slice_type,
				'slice_index': slice_index,
			},
		)
	_write_json(
		visualization_metadata_json,
		{'figures': visualization_entries},
	)
	return {
		'artifact_root': artifact_root,
		'model_tag': model_tag,
		'probe_spec': probe_spec,
		'lithology_root': lithology_root,
		'probe_dir': probe_dir,
		'report_dir': report_dir,
		'prediction_dir': prediction_dir,
		'visualization_dir': visualization_dir,
		'metrics_json': metrics_json,
		'probe_config_json': probe_config_json,
		'prediction_metadata_json': prediction_metadata_json,
		'visualization_metadata_json': visualization_metadata_json,
		'checkpoint': checkpoint,
	}


def _metrics_payload(
	*,
	accuracy: float,
	feature_source: dict[str, object] | None = None,
) -> dict[str, object]:
	payload: dict[str, object] = {
		'accuracy': accuracy,
		'balanced_accuracy': 0.6,
		'macro_f1': 0.58,
		'weighted_f1': 0.61,
		'mean_iou': 0.42,
		'per_class_precision': {'0': 0.7, '5': 0.5},
		'per_class_recall': {'0': 0.8, '5': 0.4},
		'per_class_f1': {'0': 0.75, '5': 0.44},
		'per_class_iou': {'0': 0.6, '5': 0.28},
		'per_class_support': {'0': 10, '5': 6},
		'confusion_matrix': [[8, 2], [3, 3]],
		'confusion_matrix_row_normalized': [[0.8, 0.2], [0.5, 0.5]],
		'class_ids': [0, 5],
		'class_names': {'0': 'Background', '5': 'Zechstein'},
	}
	if feature_source is not None:
		payload['feature_source'] = dict(feature_source)
	return payload


def _token_metadata_payload(
	model_tag: str,
	*,
	feature_source: dict[str, object] | None = None,
) -> dict[str, object]:
	payload: dict[str, object] = {
		'artifact_type': 'f3_lithology_token_dataset',
		'dataset': {
			'name': 'f3_facies_benchmark',
			'version': 'facies_benchmark_v1',
		},
		'model': {'tag': model_tag},
		'label_source_of_truth': 'segy_label_volume',
		'png_label_role': 'train_validation_slice_selection_and_visual_qc',
		'geometry': {'shape_xyz': [4, 5, 6]},
		'tokenization': {
			'min_labeled_fraction': 0.5,
			'min_majority_fraction': 0.7,
			'ignore_z_border_samples': 1,
		},
		'classes': _classes(),
		'summary': {
			'train_tokens': 10,
			'validation_tokens': 6,
			'all_labeled_tokens': 16,
			'total_dropped_tokens': 4,
			'total_ambiguous_tokens': 1,
			'train_class_counts': {'0': 8, '5': 2},
			'validation_class_counts': {'0': 2, '5': 4},
		},
		'slices': [
			{'split': 'train', 'slice_type': 'inline', 'slice_index': 101},
			{'split': 'validation', 'slice_type': 'inline', 'slice_index': 250},
		],
	}
	if feature_source is not None:
		payload['feature_source'] = dict(feature_source)
	return payload


def _probe_config_payload(  # noqa: PLR0913
	*,
	model_tag: str,
	embed_spec: str,
	probe_spec: str,
	lithology_root: Path,
	probe_dir: Path,
	token_metadata_json: Path,
	checkpoint: Path,
	feature_source: dict[str, object] | None = None,
) -> dict[str, object]:
	payload: dict[str, object] = {
		'artifact_type': 'f3_lithology_probe',
		'dataset': {
			'name': 'f3_facies_benchmark',
			'version': 'facies_benchmark_v1',
		},
		'model': {
			'tag': model_tag,
			'checkpoint': str(checkpoint),
			'freeze_encoder': True,
		},
		'embeddings': {'spec': embed_spec},
		'labels': {'set': 'png_slices_segy_labels_v1'},
		'lithology': {'root': str(lithology_root)},
		'token_dataset': {
			'input_dir': str(lithology_root / 'token_dataset'),
			'feature_source': dict(feature_source or {}),
		},
		'probe': {
			'spec': probe_spec,
			'type': 'logistic_regression',
			'feature_scaling': 'standard',
			'class_weight': 'balanced',
			'max_iter': 2000,
		},
		'inputs': {'token_dataset_metadata_json': str(token_metadata_json)},
		'outputs': {
			'metrics_json': str(probe_dir / 'metrics.json'),
			'confusion_matrix_png': str(
				probe_dir / 'figures' / 'confusion_matrix.png'
			),
			'per_class_f1_png': str(probe_dir / 'figures' / 'per_class_f1.png'),
		},
		'classes': _classes(),
		'summary': {
			'train_tokens': 10,
			'validation_tokens': 6,
			'train_class_counts': {'0': 8, '5': 2},
			'validation_class_counts': {'0': 2, '5': 4},
		},
		'training_summary': {'trainer': 'sklearn.linear_model.LogisticRegression'},
	}
	if feature_source is not None:
		payload['feature_source'] = dict(feature_source)
	return payload


def _classes() -> list[dict[str, object]]:
	return [
		{'class_id': 0, 'class_name': 'Background', 'rgb': [0, 0, 0]},
		{'class_id': 5, 'class_name': 'Zechstein', 'rgb': [128, 64, 32]},
	]


def _write_token_npz(path: Path, labels: np.ndarray) -> None:
	count = int(labels.shape[0])
	path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(
		path,
		features=np.zeros((count, 2), dtype=np.float32),
		labels=labels,
		survey_id=np.asarray(['f3_facies_benchmark'] * count),
		split=np.asarray(['train'] * count),
		slice_type=np.asarray(['inline'] * count),
		slice_index=np.arange(count, dtype=np.int64),
		token_xyz=np.column_stack(
			(
				np.arange(count, dtype=np.int64),
				np.zeros(count, dtype=np.int64),
				np.zeros(count, dtype=np.int64),
			),
		),
		voxel_center_xyz=np.zeros((count, 3), dtype=np.float32),
		majority_fraction=np.ones(count, dtype=np.float32),
		labeled_fraction=np.ones(count, dtype=np.float32),
	)


def _write_json(path: Path, payload: dict[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _figures_section(markdown: str) -> str:
	start = markdown.index('## Figures')
	end = markdown.index('## Interpretation')
	return markdown[start:end]
