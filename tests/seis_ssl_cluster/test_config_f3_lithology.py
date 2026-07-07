from __future__ import annotations

import json
from pathlib import Path

import seis_ssl_cluster.config.f3_lithology as f3_lithology_config
import seis_ssl_cluster.config.validate as validate_config
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology import (
	f3_lithology_prediction_config_from_mapping,
	f3_lithology_probe_config_from_mapping,
	f3_lithology_visualization_config_from_mapping,
	f3_prepare_volume_config_from_mapping,
)
from seis_ssl_cluster.config.schema import F3_FACIES_DATASET_VERSION

F3_PREPARE_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/10_prepare/01_prepare_f3_volume.yaml',
)


def test_f3_volume_prepare_config_resolves_from_stage_module() -> None:
	raw = load_config(F3_PREPARE_CONFIG)

	config = f3_prepare_volume_config_from_mapping(raw)

	assert config.dataset.version == F3_FACIES_DATASET_VERSION
	assert 'runs' not in config.outputs.volume_dir.parts


def test_f3_volume_prepare_outputs_match_artifact_paths_contract() -> None:
	raw = load_config(F3_PREPARE_CONFIG)
	config = f3_prepare_volume_config_from_mapping(raw)
	artifact_root = config.paths.artifact_root
	version = F3_FACIES_DATASET_VERSION

	assert config.outputs.volume_dir == (
		artifact_root / 'registry' / 'volumes' / 'f3' / version
	)
	assert config.outputs.manifest_path.parent == (
		artifact_root / 'registry' / 'manifests' / 'f3' / version
	)
	assert config.outputs.split_path.parent == (
		artifact_root / 'registry' / 'splits' / 'f3' / version
	)


def test_f3_lithology_config_entrypoints_reexport_from_validate_module() -> None:
	for name in (
		'f3_prepare_volume_config_from_mapping',
		'f3_lithology_token_dataset_config_from_mapping',
		'f3_lithology_probe_config_from_mapping',
		'f3_lithology_prediction_config_from_mapping',
		'f3_lithology_visualization_config_from_mapping',
		'f3_lithology_report_config_from_mapping',
		'f3_lithology_publish_config_from_mapping',
	):
		assert getattr(validate_config, name) is getattr(f3_lithology_config, name)


def test_f3_lithology_probe_config_can_resolve_without_loading_classes(
	tmp_path: Path,
) -> None:
	payload = _probe_config_mapping(tmp_path, class_info_exists=False)

	config = f3_lithology_probe_config_from_mapping(payload, load_classes=False)

	assert config.classes is None
	assert config.inputs.class_info == Path(payload['labels']['class_info'])


def test_f3_lithology_probe_config_loads_classes_by_default(tmp_path: Path) -> None:
	payload = _probe_config_mapping(tmp_path, class_info_exists=True)

	config = f3_lithology_probe_config_from_mapping(payload)

	assert config.classes is not None
	assert [item.class_id for item in config.classes] == [0, 1]


def test_f3_lithology_prediction_config_can_resolve_without_loading_classes(
	tmp_path: Path,
) -> None:
	payload = _prediction_config_mapping(tmp_path, class_info_exists=False)

	config = f3_lithology_prediction_config_from_mapping(payload, load_classes=False)

	assert config.classes is None
	assert config.inputs.class_info == Path(payload['labels']['class_info'])
	assert config.inputs.validation_tokens == (
		Path(payload['lithology']['root']) / 'token_dataset' / 'validation_tokens.npz'
	)


def test_f3_lithology_prediction_config_loads_classes_by_default(
	tmp_path: Path,
) -> None:
	payload = _prediction_config_mapping(tmp_path, class_info_exists=True)

	config = f3_lithology_prediction_config_from_mapping(payload)

	assert config.classes is not None
	assert [item.class_name for item in config.classes] == ['zero', 'one']


def test_f3_lithology_visualization_config_can_resolve_without_loading_classes(
	tmp_path: Path,
) -> None:
	payload = _visualization_config_mapping(tmp_path, class_info_exists=False)

	config = f3_lithology_visualization_config_from_mapping(
		payload,
		load_classes=False,
	)

	assert config.classes is None
	assert config.inputs.class_info == Path(payload['labels']['class_info'])


def test_f3_lithology_visualization_config_loads_classes_by_default(
	tmp_path: Path,
) -> None:
	payload = _visualization_config_mapping(tmp_path, class_info_exists=True)

	config = f3_lithology_visualization_config_from_mapping(payload)

	assert config.classes is not None
	assert [item.rgb for item in config.classes] == [(1, 2, 3), (4, 5, 6)]


def _probe_config_mapping(
	tmp_path: Path,
	*,
	class_info_exists: bool,
) -> dict[str, object]:
	roots = _roots(tmp_path)
	class_info = _class_info_path(roots['artifact_root'], exists=class_info_exists)
	token_dataset = roots['artifact_root'] / 'lithology' / 'tokens'
	probe_output = roots['artifact_root'] / 'lithology' / 'probe'
	return {
		'paths': _path_mapping(roots),
		'dataset': _dataset_mapping(),
		'model': _model_mapping(),
		'embeddings': {'input_dir': str(roots['artifact_root'] / 'embeddings')},
		'labels': {'class_info': str(class_info)},
		'lithology': {'root': str(roots['artifact_root'] / 'lithology')},
		'token_dataset': {
			'input_dir': str(token_dataset),
			'metadata_json': str(token_dataset / 'token_dataset_metadata.json'),
		},
		'probe': {
			'spec': 'linear',
			'type': 'logistic_regression',
			'output_dir': str(probe_output),
		},
	}


def _prediction_config_mapping(
	tmp_path: Path,
	*,
	class_info_exists: bool,
) -> dict[str, object]:
	roots = _roots(tmp_path)
	class_info = _class_info_path(roots['artifact_root'], exists=class_info_exists)
	prediction_output = roots['artifact_root'] / 'lithology' / 'predictions'
	return {
		'paths': _path_mapping(roots),
		'dataset': _dataset_mapping(),
		'model': _model_mapping(),
		'embeddings': {'input_dir': str(roots['artifact_root'] / 'embeddings')},
		'labels': _labels_mapping(roots, class_info),
		'lithology': {'root': str(roots['artifact_root'] / 'lithology')},
		'probe': {
			'probe_joblib': str(roots['artifact_root'] / 'lithology' / 'probe.joblib'),
			'scaler_joblib': str(
				roots['artifact_root'] / 'lithology' / 'scaler.joblib',
			),
		},
		'predictions': _predictions_mapping(prediction_output),
	}


def _visualization_config_mapping(
	tmp_path: Path,
	*,
	class_info_exists: bool,
) -> dict[str, object]:
	roots = _roots(tmp_path)
	class_info = _class_info_path(roots['artifact_root'], exists=class_info_exists)
	prediction_output = roots['artifact_root'] / 'lithology' / 'predictions'
	visualization_output = roots['artifact_root'] / 'lithology' / 'visualizations'
	predictions = _predictions_mapping(prediction_output)
	return {
		'paths': _path_mapping(roots),
		'dataset': _dataset_mapping(),
		'model': _model_mapping(),
		'labels': _labels_mapping(roots, class_info),
		'registry': {
			'seismic_volume': str(roots['artifact_root'] / 'registry' / 'f3.npy'),
		},
		'lithology': {'root': str(roots['artifact_root'] / 'lithology')},
		'probe': {'spec': 'linear'},
		'predictions': {
			'token_predictions': predictions['token_predictions'],
			'probability_volume': predictions['probability_volume'],
			'metadata_json': predictions['metadata_json'],
			'validation_slice_metrics_csv': predictions[
				'validation_slice_metrics_csv'
			],
		},
		'visualizations': {
			'output_dir': str(visualization_output),
			'metadata_json': str(visualization_output / 'metadata.json'),
			'selected_slices_dir': str(visualization_output / 'selected_slices'),
			'slices': {'inline': [100], 'crossline': [], 'z': [10]},
		},
	}


def _roots(tmp_path: Path) -> dict[str, Path]:
	return {
		'artifact_root': tmp_path / 'artifacts' / 'seis_ssl_cluster',
		'f3_root': tmp_path / 'F3',
	}


def _path_mapping(roots: dict[str, Path]) -> dict[str, str]:
	return {
		'artifact_root': str(roots['artifact_root']),
		'f3_root': str(roots['f3_root']),
	}


def _dataset_mapping() -> dict[str, str]:
	return {'name': 'f3_facies_benchmark', 'version': 'facies_benchmark_v1'}


def _model_mapping() -> dict[str, object]:
	return {'tag': 'model', 'freeze_encoder': True}


def _labels_mapping(roots: dict[str, Path], class_info: Path) -> dict[str, str]:
	return {
		'source_label_volume': str(
			roots['artifact_root'] / 'registry' / 'f3_facies_labels.npy',
		),
		'png_label_inventory': str(
			roots['artifact_root'] / 'inspection' / 'png_labels.csv',
		),
		'class_info': str(class_info),
		'segy_geometry_json': str(
			roots['artifact_root'] / 'inspection' / 'segy_geometry.json',
		),
		'source_label_segy': str(roots['f3_root'] / 'f3_labels.sgy'),
	}


def _predictions_mapping(output_dir: Path) -> dict[str, str]:
	return {
		'output_dir': str(output_dir),
		'token_predictions': str(output_dir / 'token_predictions.npy'),
		'probability_volume': str(output_dir / 'probability_volume.npy'),
		'valid_token_grid': str(output_dir / 'valid_token_grid.npy'),
		'metadata_json': str(output_dir / 'metadata.json'),
		'validation_slice_metrics_csv': str(
			output_dir / 'validation_slice_metrics.csv',
		),
	}


def _class_info_path(artifact_root: Path, *, exists: bool) -> Path:
	path = artifact_root / 'inspection' / 'f3' / 'inventory' / 'class_info.json'
	if exists:
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(
			json.dumps(
				{
					'0': {'name': 'zero', 'color': [1, 2, 3]},
					'1': {'name': 'one', 'color': [4, 5, 6]},
				},
			),
			encoding='utf-8',
		)
	return path
