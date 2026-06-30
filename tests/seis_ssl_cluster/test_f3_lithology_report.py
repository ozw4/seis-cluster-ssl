from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from seis_ssl_cluster.f3 import (
	F3LithologyComparisonReportConfig,
	F3LithologyReportConfig,
	build_f3_lithology_comparison_report,
	build_f3_lithology_report,
)
from tests.helpers import run_python_proc


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
	assert rows[0]['MODEL_TAG'] == 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	assert rows[0]['PROBE_SPEC'] == 'mlp_balanced_v1'
	assert rows[0]['FEATURE_SOURCE_KIND'] == 'random_encoder'
	assert rows[0]['FEATURE_SOURCE_REFERENCE_MODEL_TAG'] == (
		'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	)
	assert rows[0]['FEATURE_SOURCE_EMBED_SPEC'] == 'overlap_x16'
	assert rows[0]['accuracy'] == '0.65'
	assert rows[1]['MODEL_TAG'] == 'amp_mae_m075_mse_g0_patchnorm_clip8_vis00_v1'
	assert rows[1]['PROBE_SPEC'] == 'linear_balanced_v1'
	assert 'class_0_f1' in rows[0]
	assert 'class_5_f1' in rows[0]
	assert 'FEATURE_SOURCE_KIND' in markdown
	assert '集約run数: 2' in markdown


def test_build_f3_lithology_report_proc_dry_run(tmp_path: Path) -> None:
	run = _write_probe_run(
		tmp_path,
		model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		embed_spec='overlap_x16',
		probe_spec='linear_balanced_v1',
	)
	artifact_root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	config = {
		'paths': {
			'f3_root': str(tmp_path / 'F3'),
			'artifact_root': str(artifact_root),
		},
		'dataset': {
			'name': 'f3_facies_benchmark',
			'version': 'facies_benchmark_v1',
		},
		'model': {
			'tag': run['model_tag'],
			'checkpoint': str(
				artifact_root
				/ 'pretraining'
				/ 'nopims'
				/ 'pretrain_v1'
				/ run['model_tag']
				/ 'full_100ep'
				/ 'mae_best.pt'
			),
			'freeze_encoder': True,
		},
		'labels': {
			'set': 'png_slices_segy_labels_v1',
			'png_label_role': 'train_validation_slice_selection_and_visual_qc',
		},
		'lithology': {'root': str(run['lithology_root'])},
		'probe': {
			'spec': run['probe_spec'],
			'metrics_json': str(run['metrics_json']),
		},
		'predictions': {'metadata_json': str(run['prediction_metadata_json'])},
		'visualizations': {'metadata_json': str(run['visualization_metadata_json'])},
		'reports': {
			'output_dir': str(run['report_dir']),
			'output_markdown': str(run['report_dir'] / 'report.md'),
			'output_json': str(run['report_dir'] / 'report.json'),
		},
	}
	config_path = tmp_path / 'build_lithology_report.yaml'
	config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/build_f3_lithology_report.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert 'stage: build_f3_lithology_report' in result.stdout
	assert 'reports.output_markdown:' in result.stdout
	assert 'comparison.output_csv:' in result.stdout
	assert 'execution: dry-run; F3 lithology report skipped' in result.stdout


def test_build_f3_lithology_report_proc_default_config_dry_run() -> None:
	result = run_python_proc(
		Path('proc/seis_ssl_cluster/build_f3_lithology_report.py'),
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert 'stage: build_f3_lithology_report' in result.stdout
	assert 'reports.output_markdown:' in result.stdout
	assert 'comparison.output_csv:' in result.stdout
	assert 'prediction_metadata.json' in result.stdout
	assert 'execution: dry-run; F3 lithology report skipped' in result.stdout


def test_default_lithology_report_config_uses_prediction_metadata_json() -> None:
	config_dir = _default_lithology_config_dir()
	predict_config = yaml.safe_load(
		(config_dir / '04_predict_volume.yaml').read_text(encoding='utf-8'),
	)
	report_config = yaml.safe_load(
		(config_dir / '06_build_lithology_report.yaml').read_text(encoding='utf-8'),
	)

	assert report_config['predictions']['metadata_json'] == (
		predict_config['predictions']['metadata_json']
	)
	assert report_config['predictions']['metadata_json'].endswith(
		'/prediction_metadata.json',
	)


def test_default_lithology_configs_use_latest_checkpoint_contract() -> None:
	config_dir = _default_lithology_config_dir()
	expected_checkpoint = (
		'/workspace/artifacts/seis_ssl_cluster/pretraining/nopims/pretrain_v1/'
		'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full_100ep/'
		'mae_latest.pt'
	)

	for yaml_path in sorted(config_dir.glob('*.yaml')):
		payload = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
		assert payload['model']['checkpoint'] == expected_checkpoint


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


def _write_probe_run(  # noqa: PLR0913
	root: Path,
	*,
	model_tag: str,
	embed_spec: str,
	probe_spec: str,
	accuracy: float = 0.625,
	write_metrics: bool = True,
	feature_source: dict[str, object] | None = None,
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
		visualization_dir / 'validation_inline_0250_prediction.png',
	):
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_bytes(b'fake-png')
	if write_metrics:
		_write_json(
			metrics_json,
			_metrics_payload(accuracy=accuracy, feature_source=feature_source),
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
	_write_json(
		visualization_metadata_json,
		{
			'figures': [
				{
					'path': str(
						visualization_dir / 'validation_inline_0250_prediction.png'
					),
					'group': 'validation',
					'slice_type': 'inline',
					'slice_index': 250,
				},
			],
		},
	)
	return {
		'artifact_root': artifact_root,
		'model_tag': model_tag,
		'probe_spec': probe_spec,
		'lithology_root': lithology_root,
		'probe_dir': probe_dir,
		'report_dir': report_dir,
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


def _write_json(path: Path, payload: dict[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _default_lithology_config_dir() -> Path:
	return (
		Path('experiments')
		/ 'f3'
		/ 'facies_benchmark_v1'
		/ '50_lithology'
		/ 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
		/ 'overlap_x16'
		/ 'png_slices_segy_labels_v1'
	)


def _figures_section(markdown: str) -> str:
	start = markdown.index('## Figures')
	end = markdown.index('## Interpretation')
	return markdown[start:end]
