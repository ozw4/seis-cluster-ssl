from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

import seis_ssl_cluster.f3.lithology.voxel_report as voxel_report_module
from seis_ssl_cluster.config.f3_lithology_voxel_report import (
	_selected_slices,
	f3_lithology_voxel_report_config_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.labels import F3ClassInfo
from seis_ssl_cluster.f3.lithology.voxel_evaluation import (
	EVALUATION_METADATA_JSON,
	EVALUATION_OUTPUT_FILES,
	METRICS_JSON,
	evaluate_f3_lithology_voxels,
)
from seis_ssl_cluster.f3.lithology.voxel_report import (
	VOXEL_EVALUATION_PUBLISH_FILES,
	F3LithologyVoxelPublishConfig,
	F3LithologyVoxelReportConfig,
	F3LithologyVoxelReportResult,
	_selected_slice_pairs,
	_validate_identity_summary,
	_write_aggregate_figures,
	build_f3_lithology_voxel_report,
	build_f3_lithology_voxel_report_payload,
	inspect_f3_lithology_voxel_report,
	publish_f3_lithology_voxel_report,
	render_f3_lithology_voxel_report_markdown,
)
from seis_ssl_cluster.f3.lithology.voxel_visualization import (
	F3LithologyVoxelFigureConfig,
)
from seis_ssl_cluster.models.voxel_decoder import (
	voxel_decoder_architecture_mapping,
)
from tests.seis_ssl_cluster.test_f3_lithology_voxel_evaluation import (
	_fixture as evaluation_fixture,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_empty_selected_slices_preserve_validation_slice_fallback() -> None:
	assert _selected_slices({}) == {}


@pytest.mark.parametrize(
	('kind', 'wording'),
	[
		('token_projection_nearest', 'V0 nearest token projection'),
		('frozen_embedding_decoder', 'V1 learned frozen-embedding decoder'),
	],
)
def test_voxel_report_is_complete_standard_json_with_v0_v1_wording(
	kind: str, wording: str
) -> None:
	payload = _payload(kind=kind)
	markdown = render_f3_lithology_voxel_report_markdown(payload)

	assert payload['prediction']['label'] == wording
	assert 'decoder' in payload['prediction']
	assert payload['supervision']['validation_precedence'] is True
	assert payload['overall_voxel_metrics']['accuracy'] == 1.0
	assert len(payload['per_slice_metrics']) == 1
	assert 'unique validation voxels' in markdown
	assert 'plane-level; not aggregate' in markdown
	if kind == 'frozen_embedding_decoder':
		assert 'frozen_embedding_decoder_nearest_voxel_ln_v1' in markdown
		assert 'frozen_embedding_decoder_v1' not in markdown
	assert json.dumps(payload, allow_nan=False)


def test_voxel_report_handles_missing_and_zero_support_classes() -> None:
	payload = _payload(kind='frozen_embedding_decoder')
	classes = payload['monitored_classes']

	assert classes[0] == {
		'class_id': 3,
		'status': 'zero_support',
		'support': 0,
		'f1': None,
		'iou': None,
		'boundary_recall': None,
	}
	assert classes[1]['class_id'] == 5
	assert classes[1]['status'] == 'missing_class'
	assert classes[1]['support'] is None


def test_voxel_report_rejects_evaluation_prediction_decoder_mismatch() -> None:
	architecture = voxel_decoder_architecture_mapping(
		embedding_dim=4,
		class_count=3,
		hidden_channels=(4,),
		upsample_factors=((1, 1, 2),),
	)
	evaluation_architecture = {**architecture, 'hidden_channels': [8]}
	with pytest.raises(ValueError, match='decoder identity mismatch'):
		voxel_report_module._report_decoder_identity(  # noqa: SLF001
			kind='frozen_embedding_decoder',
			prediction_metadata={'decoder_architecture': architecture},
			evaluation_metadata={
				'decoder_architecture': evaluation_architecture
			},
		)


def test_aggregate_figures_handle_class_missing_from_metric_maps(
	tmp_path: Path,
) -> None:
	classes = (
		F3ClassInfo(class_id=0, class_name='class zero', rgb=(0, 0, 0)),
		F3ClassInfo(class_id=3, class_name='class three', rgb=(3, 3, 3)),
		F3ClassInfo(class_id=5, class_name='class five', rgb=(5, 5, 5)),
	)
	paths = _write_aggregate_figures(
		tmp_path,
		metrics={
			'confusion_matrix': [[4, 0, 0], [0, 0, 0], [0, 0, 0]],
			'per_class_f1': {'0': 1.0, '3': 0.0},
			'per_class_iou': {'0': 1.0, '3': 0.0},
		},
		boundary={},
		regions=(),
		classes=classes,
		config=F3LithologyVoxelFigureConfig(dpi=35),
	)

	assert all(path.stat().st_size > 0 for path in paths)


def test_voxel_publish_excludes_raw_volume_and_enforces_size_guard(
	tmp_path: Path,
) -> None:
	report_dir = tmp_path / 'artifacts' / 'report'
	figure = report_dir / 'figures' / 'confusion_matrix.png'
	figure.parent.mkdir(parents=True)
	markdown = report_dir / 'report.md'
	json_path = report_dir / 'report.json'
	markdown.write_text('# report\n', encoding='utf-8')
	json_path.write_text('{}\n', encoding='utf-8')
	figure.write_bytes(b'png')
	(report_dir / 'raw.npy').write_bytes(b'raw-volume')
	payload = _payload(kind='token_projection_nearest')
	result = F3LithologyVoxelReportResult(markdown, json_path, (figure,), payload)
	config = _publish_config(tmp_path, report_dir=report_dir)
	published_files = publish_f3_lithology_voxel_report(result, config=config)

	assert published_files
	targets = [path.name for path in published_files]
	assert targets == [
		'report.md',
		'report.json',
		*VOXEL_EVALUATION_PUBLISH_FILES,
		'confusion_matrix.png',
	]
	assert 'raw.npy' not in targets
	with pytest.raises(ValueError, match='exceeds max_file_size_bytes'):
		publish_f3_lithology_voxel_report(
			result,
			config=replace(
				config,
				publish=replace(
					config.publish,
					output_dir=tmp_path / 'reports' / 'too-small',
					max_file_size_bytes=2,
				),
			),
		)


@pytest.mark.parametrize(
	('kind', 'prediction_spec'),
	[
		('token_projection_nearest', 'token_projection_nearest_v1'),
		(
			'frozen_embedding_decoder',
			'frozen_embedding_decoder_nearest_voxel_ln_v1',
		),
	],
)
def test_voxel_publish_default_dir_uses_versioned_prediction_spec(
	tmp_path: Path, kind: str, prediction_spec: str
) -> None:
	report_dir = tmp_path / 'artifacts' / 'report'
	figure = report_dir / 'figures' / 'confusion_matrix.png'
	figure.parent.mkdir(parents=True)
	markdown = report_dir / 'report.md'
	json_path = report_dir / 'report.json'
	markdown.write_text('# report\n', encoding='utf-8')
	json_path.write_text('{}\n', encoding='utf-8')
	figure.write_bytes(b'png')
	result = F3LithologyVoxelReportResult(
		markdown,
		json_path,
		(figure,),
		_payload(kind=kind),
	)
	config = _publish_config(tmp_path, report_dir=report_dir)
	config = replace(
		config,
		publish=replace(config.publish, output_dir=None),
	)

	published_files = publish_f3_lithology_voxel_report(result, config=config)

	expected_output_dir = (
		tmp_path
		/ 'reports'
		/ 'f3'
		/ 'test'
		/ 'voxel_lithology'
		/ 'model'
		/ prediction_spec
	)
	assert published_files
	assert all(path.is_relative_to(expected_output_dir) for path in published_files)


def test_voxel_publish_preserves_explicit_output_dir(tmp_path: Path) -> None:
	config = F3LithologyVoxelPublishConfig(
		enabled=True,
		reports_root=tmp_path / 'reports',
		output_dir=tmp_path / 'outside-results',
	)
	assert config.output_dir == tmp_path / 'outside-results'


def test_selected_slices_must_have_validation_metrics(tmp_path: Path) -> None:
	config = replace(
		_publish_config(tmp_path, report_dir=tmp_path / 'report'),
		selected_slices={'inline': (999,)},
	)
	with pytest.raises(ValueError, match='must be validation slices'):
		_selected_slice_pairs(
			config,
			slice_rows=({'slice_type': 'inline', 'slice_index': '100'},),
		)


def test_voxel_report_checks_evaluation_input_paths_and_hashes(
	tmp_path: Path,
) -> None:
	config = _identity_config(tmp_path)
	supervision = {
		'dataset': dict(config.dataset),
		'labels': {
			'source_label_segy': str(config.label_volume.with_name('labels.sgy'))
		},
	}
	prediction = {
		'prediction_kind': 'token_projection_nearest',
		'model_tag': 'model',
	}
	inputs = {
		name: _identity(path)
		for name, path in _evaluation_identity_paths(config).items()
	}
	evaluation = {
		'prediction_kind': 'token_projection_nearest',
		'model_tag': 'model',
		'dataset': dict(config.dataset),
		'inputs': inputs,
	}
	_validate_identity_summary(
		prediction,
		evaluation=evaluation,
		supervision=supervision,
		config=config,
	)

	inputs['prediction_metadata']['sha256'] = '0' * 64
	with pytest.raises(ValueError, match='prediction_metadata hash'):
		_validate_identity_summary(
			prediction,
			evaluation=evaluation,
			supervision=supervision,
			config=config,
		)
	inputs['prediction_metadata'] = _identity(
		config.label_volume.with_name('labels.sgy')
	)
	with pytest.raises(ValueError, match='prediction_metadata path'):
		_validate_identity_summary(
			prediction,
			evaluation=evaluation,
			supervision=supervision,
			config=config,
		)


def test_voxel_report_rejects_numeric_output_hash_mismatch(tmp_path: Path) -> None:
	config, _ = _evaluated_report_job(tmp_path)
	inspect_f3_lithology_voxel_report(config)
	(config.evaluation_input_dir / METRICS_JSON).write_text(
		'{}\n', encoding='utf-8'
	)

	with pytest.raises(ValueError, match=r'metrics\.json hash'):
		inspect_f3_lithology_voxel_report(config)


def test_voxel_report_rejects_rearranged_prediction_array(tmp_path: Path) -> None:
	config, _ = _evaluated_report_job(tmp_path)
	inspect_f3_lithology_voxel_report(config)
	predictions = np.load(
		config.prediction_input_dir / 'f3_voxel_predictions.npy',
		mmap_mode='r+',
	)
	flat = predictions.reshape(-1)
	first = int(np.flatnonzero(flat == 3)[0])
	second = int(np.flatnonzero(flat == 5)[0])
	flat[first], flat[second] = flat[second], flat[first]
	predictions.flush()

	with pytest.raises(ValueError, match='voxel_predictions hash'):
		inspect_f3_lithology_voxel_report(config)


def test_voxel_report_overwrite_removes_obsolete_selected_slice(
	tmp_path: Path,
) -> None:
	config, _ = _evaluated_report_job(tmp_path)
	build_f3_lithology_voxel_report(config)
	obsolete = config.output_dir / 'figures' / 'selected_slices' / 'obsolete.png'
	obsolete.write_bytes(b'obsolete')

	build_f3_lithology_voxel_report(replace(config, overwrite=True))

	assert not obsolete.exists()


def test_voxel_report_failure_leaves_existing_output_untouched(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config, _ = _evaluated_report_job(tmp_path)
	config.output_dir.mkdir(parents=True)
	marker = config.output_dir / 'existing.txt'
	marker.write_text('existing\n', encoding='utf-8')

	def fail_after_partial_write(output_dir: Path, **_: object) -> tuple[Path, ...]:
		(output_dir / 'partial.png').write_bytes(b'partial')
		raise RuntimeError('synthetic report failure')

	monkeypatch.setattr(
		voxel_report_module,
		'_write_aggregate_figures',
		fail_after_partial_write,
	)
	with pytest.raises(RuntimeError, match='synthetic report failure'):
		build_f3_lithology_voxel_report(replace(config, overwrite=True))

	assert marker.read_text(encoding='utf-8') == 'existing\n'
	assert not (config.output_dir / 'partial.png').exists()
	assert not tuple(
		config.output_dir.parent.glob(f'.{config.output_dir.name}.staging-*')
	)


def test_voxel_report_config_rejects_existing_output(tmp_path: Path) -> None:
	config, raw = _evaluated_report_job(tmp_path)
	config.output_dir.mkdir(parents=True)

	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		f3_lithology_voxel_report_config_from_mapping(raw)


def _payload(*, kind: str) -> dict[str, object]:
	metrics = {
		'accuracy': 1.0,
		'balanced_accuracy': 1.0,
		'macro_f1': 0.5,
		'weighted_f1': 1.0,
		'mean_iou': 0.5,
		'class_ids': [0, 3],
		'per_class_support': {'0': 4, '3': 0},
		'per_class_f1': {'0': 1.0, '3': 0.0},
		'per_class_iou': {'0': 1.0, '3': 0.0},
	}
	boundary = {
		'vertical_boundary_precision_at_1': None,
		'vertical_boundary_recall_at_1': None,
		'vertical_boundary_f1_at_1': None,
		'vertical_boundary_class_3_recall_at_1': None,
	}
	architecture = voxel_decoder_architecture_mapping(
		embedding_dim=4,
		class_count=3,
		hidden_channels=(4,),
		upsample_factors=((1, 1, 2),),
	)
	prediction_metadata = {
		'prediction_kind': kind,
		'model_tag': 'model',
		'source_identity': {'embedding': 'identity'},
		'inputs': {'decoder_checkpoint': 'best.pt'},
	}
	evaluation_metadata: dict[str, object] = {
		'policy': {'boundary_tolerances': [1]}
	}
	if kind == 'frozen_embedding_decoder':
		prediction_metadata['decoder_architecture'] = architecture
		evaluation_metadata['decoder_architecture'] = architecture
	return build_f3_lithology_voxel_report_payload(
		metrics=metrics,
		boundary_metrics=boundary,
		boundary_region_rows=(
			{
				'region': 'interior',
				'radius': '1',
				'voxel_count': '4',
				'macro_f1': '0.5',
				'mean_iou': '0.5',
			},
		),
		per_slice_rows=(
			{
				'slice_type': 'inline',
				'slice_index': '100',
				'voxel_count': '4',
				'accuracy': '1.0',
				'macro_f1': '0.5',
				'mean_iou': '0.5',
			},
		),
		prediction_metadata=prediction_metadata,
		evaluation_metadata=evaluation_metadata,
		supervision_metadata={
			'split_strategy': 'planes',
			'split_codes': {'unsupervised': 0, 'train': 1, 'validation': 2},
			'validation_precedence': True,
		},
	)


def _publish_config(
	tmp_path: Path, *, report_dir: Path
) -> F3LithologyVoxelReportConfig:
	dummy = tmp_path / 'unused'
	evaluation_dir = tmp_path / 'evaluation'
	evaluation_dir.mkdir(exist_ok=True)
	for name in EVALUATION_OUTPUT_FILES:
		(evaluation_dir / name).write_text('{}\n', encoding='utf-8')
	(evaluation_dir / EVALUATION_METADATA_JSON).write_text(
		json.dumps(
			{
				'artifact_type': 'f3_lithology_voxel_evaluation',
				'schema_version': 2,
				'outputs': {
					name: _identity(evaluation_dir / name)
					for name in EVALUATION_OUTPUT_FILES
				},
			}
		),
		encoding='utf-8',
	)
	return F3LithologyVoxelReportConfig(
		prediction_input_dir=dummy,
		voxel_dataset_input_dir=dummy,
		evaluation_input_dir=evaluation_dir,
		seismic_volume=dummy,
		label_volume=dummy,
		class_info=dummy,
		png_label_inventory=dummy,
		segy_geometry_json=dummy,
		output_dir=report_dir,
		dataset={'name': 'f3', 'version': 'test'},
		publish=F3LithologyVoxelPublishConfig(
			enabled=True,
			reports_root=tmp_path / 'reports',
			output_dir=tmp_path / 'reports' / 'published',
		),
	)


def _identity_config(tmp_path: Path) -> F3LithologyVoxelReportConfig:
	config = _publish_config(tmp_path, report_dir=tmp_path / 'report')
	prediction_dir = tmp_path / 'prediction'
	voxel_dataset_dir = tmp_path / 'voxel_dataset'
	prediction_dir.mkdir()
	voxel_dataset_dir.mkdir()
	paths = {
		'prediction_metadata': prediction_dir / 'prediction_metadata.json',
		'voxel_predictions': prediction_dir / 'f3_voxel_predictions.npy',
		'voxel_confidence': prediction_dir / 'f3_voxel_confidence.npy',
		'voxel_valid_mask': prediction_dir / 'f3_valid_voxel_mask.npy',
		'voxel_dataset_metadata': voxel_dataset_dir / 'voxel_dataset_metadata.json',
		'voxel_split_grid': voxel_dataset_dir / 'supervision_split_grid.npy',
		'label_volume': tmp_path / 'labels.npy',
		'png_label_inventory': tmp_path / 'inventory.csv',
		'segy_geometry_json': tmp_path / 'geometry.json',
		'class_info': tmp_path / 'class_info.json',
	}
	for path in paths.values():
		path.write_bytes(b'identity')
	source_label_segy = tmp_path / 'labels.sgy'
	source_label_segy.write_bytes(b'segy')
	return replace(
		config,
		prediction_input_dir=prediction_dir,
		voxel_dataset_input_dir=voxel_dataset_dir,
		label_volume=paths['label_volume'],
		png_label_inventory=paths['png_label_inventory'],
		segy_geometry_json=paths['segy_geometry_json'],
		class_info=paths['class_info'],
		dataset={'name': 'f3', 'version': 'test'},
	)


def _evaluation_identity_paths(
	config: F3LithologyVoxelReportConfig,
) -> dict[str, Path]:
	return {
		'prediction_metadata': config.prediction_input_dir / 'prediction_metadata.json',
		'voxel_predictions': config.prediction_input_dir / 'f3_voxel_predictions.npy',
		'voxel_confidence': config.prediction_input_dir / 'f3_voxel_confidence.npy',
		'voxel_valid_mask': config.prediction_input_dir / 'f3_valid_voxel_mask.npy',
		'voxel_dataset_metadata': (
			config.voxel_dataset_input_dir / 'voxel_dataset_metadata.json'
		),
		'voxel_split_grid': (
			config.voxel_dataset_input_dir / 'supervision_split_grid.npy'
		),
		'label_volume': config.label_volume,
		'png_label_inventory': config.png_label_inventory,
		'segy_geometry_json': config.segy_geometry_json,
		'class_info': config.class_info,
		'source_label_segy': config.label_volume.with_name('labels.sgy'),
	}


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _evaluated_report_job(
	tmp_path: Path,
) -> tuple[F3LithologyVoxelReportConfig, dict[str, object]]:
	evaluation_config = evaluation_fixture(tmp_path / 'evaluation-job')
	evaluate_f3_lithology_voxels(evaluation_config)
	output_dir = evaluation_config.artifact_root / 'report'
	config = F3LithologyVoxelReportConfig(
		prediction_input_dir=evaluation_config.prediction_input_dir,
		voxel_dataset_input_dir=evaluation_config.voxel_dataset_input_dir,
		evaluation_input_dir=evaluation_config.output_dir,
		seismic_volume=evaluation_config.source_label_volume,
		label_volume=evaluation_config.source_label_volume,
		class_info=evaluation_config.class_info,
		png_label_inventory=evaluation_config.png_label_inventory,
		segy_geometry_json=evaluation_config.segy_geometry_json,
		output_dir=output_dir,
		dataset=dict(evaluation_config.dataset),
		figure=F3LithologyVoxelFigureConfig(dpi=35),
	)
	raw = {
		'paths': {
			'artifact_root': str(evaluation_config.artifact_root),
			'f3_root': str(evaluation_config.f3_root),
			'reports_root': str(tmp_path / 'reports'),
		},
		'dataset': dict(evaluation_config.dataset),
		'labels': {
			'seismic_volume': str(evaluation_config.source_label_volume),
			'source_label_volume': str(evaluation_config.source_label_volume),
			'class_info': str(evaluation_config.class_info),
			'png_label_inventory': str(
				evaluation_config.png_label_inventory
			),
			'segy_geometry_json': str(evaluation_config.segy_geometry_json),
		},
		'voxel_predictions': {
			'input_dir': str(evaluation_config.prediction_input_dir)
		},
		'voxel_dataset': {
			'input_dir': str(evaluation_config.voxel_dataset_input_dir)
		},
		'evaluation': {'input_dir': str(evaluation_config.output_dir)},
		'report': {
			'selected_slices': {},
			'dpi': 35,
			'include_confidence': False,
			'amplitude_clip_percentiles': [1.0, 99.0],
		},
		'outputs': {'output_dir': str(output_dir), 'overwrite': False},
		'publish': {'enabled': False},
	}
	return config, raw
