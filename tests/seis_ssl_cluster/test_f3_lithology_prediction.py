from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import joblib
import numpy as np

from seis_ssl_cluster.f3 import (
	F3ClassInfo,
	F3LithologyPredictionConfig,
	F3LithologyPredictionInputs,
	F3LithologyPredictionOutputs,
	F3LithologyTokenPolicy,
	predict_f3_lithology_tokens,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


class _IdentityScaler:
	def transform(self, features: np.ndarray) -> np.ndarray:
		return np.asarray(features, dtype=np.float32)


class _TokenClassProbe:
	classes_ = np.asarray([0, 1, 2], dtype=np.int64)

	def predict_proba(self, features: np.ndarray) -> np.ndarray:
		class_index = np.clip(
			np.rint(np.asarray(features)[:, 0]).astype(np.int64),
			0,
			2,
		)
		probabilities = np.zeros((class_index.shape[0], 3), dtype=np.float32)
		probabilities[np.arange(class_index.shape[0]), class_index] = 1.0
		return probabilities

	def predict(self, features: np.ndarray) -> np.ndarray:
		probabilities = self.predict_proba(features)
		return self.classes_[np.argmax(probabilities, axis=1)]


def test_predict_f3_lithology_tokens_writes_grids_metadata_and_metrics(
	tmp_path: Path,
) -> None:
	config = write_prediction_fixture(tmp_path)

	result = predict_f3_lithology_tokens(config)

	predictions = np.load(result.token_predictions)
	probabilities = np.load(result.probability_volume)
	valid_tokens = np.load(result.valid_token_grid)
	metadata = json.loads(result.metadata_json.read_text(encoding='utf-8'))
	with result.validation_slice_metrics_csv.open(
		encoding='utf-8',
		newline='',
	) as file_obj:
		metric_rows = list(csv.DictReader(file_obj))

	assert predictions.shape == (2, 2, 2)
	assert probabilities.shape == (2, 2, 2, 3)
	assert valid_tokens.shape == (2, 2, 2)
	assert predictions[1, 1, 1] == -1
	assert np.isnan(probabilities[1, 1, 1]).all()
	assert predictions[0, 0, 0] == 0
	assert predictions[0, 0, 1] == 1
	assert result.valid_token_count == 7
	assert result.invalid_token_count == 1
	assert metadata['artifact_type'] == 'f3_lithology_token_predictions'
	assert metadata['invalid_prediction_class_id'] == -1
	assert metadata['class_probability_order'] == [0, 1, 2]
	assert metadata['summary']['validation_slice_count'] == 2
	assert {row['slice_type'] for row in metric_rows} == {'inline', 'crossline'}
	assert all(float(row['accuracy']) == 1.0 for row in metric_rows)


def test_predict_f3_lithology_tokens_writes_standard_json_for_empty_metrics(
	tmp_path: Path,
) -> None:
	config = write_prediction_fixture(tmp_path)
	valid_token_path = (
		config.inputs.embeddings_dir / 'f3_facies_benchmark.valid_tokens.npy'
	)
	valid_tokens = np.load(valid_token_path)
	valid_tokens[0, :, :] = False
	np.save(valid_token_path, valid_tokens)

	result = predict_f3_lithology_tokens(config)

	text = result.metadata_json.read_text(encoding='utf-8')
	metadata = json.loads(text)
	inline_row = next(
		row
		for row in metadata['validation_slice_metrics']
		if row['slice_type'] == 'inline'
	)

	assert 'NaN' not in text
	assert 'Infinity' not in text
	assert inline_row['token_count'] == 0
	assert inline_row['accuracy'] is None
	assert inline_row['balanced_accuracy'] is None
	assert inline_row['macro_f1'] is None
	assert inline_row['weighted_f1'] is None
	assert inline_row['mean_iou'] is None


def write_prediction_fixture(tmp_path: Path) -> F3LithologyPredictionConfig:
	artifact_root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	f3_root = tmp_path / 'F3'
	f3_root.mkdir()
	label_path = artifact_root / 'registry' / 'volumes' / 'f3_facies_labels.npy'
	label_path.parent.mkdir(parents=True)
	np.save(label_path, _dense_label_volume())
	class_info = _write_class_info(artifact_root)
	inventory = _write_inventory(artifact_root)
	geometry = _write_geometry(artifact_root)
	embeddings_dir = artifact_root / 'embeddings' / 'f3'
	_write_embedding_artifacts(embeddings_dir)
	probe_dir = artifact_root / 'lithology' / 'f3' / 'probes' / 'linear_test'
	probe_dir.mkdir(parents=True)
	probe_path = probe_dir / 'probe.joblib'
	scaler_path = probe_dir / 'scaler.joblib'
	joblib.dump(_TokenClassProbe(), probe_path)
	joblib.dump(_IdentityScaler(), scaler_path)
	output_dir = artifact_root / 'lithology' / 'f3' / 'predictions' / 'linear_test'
	return F3LithologyPredictionConfig(
		inputs=F3LithologyPredictionInputs(
			embeddings_dir=embeddings_dir,
			probe_joblib=probe_path,
			scaler_joblib=scaler_path,
			label_volume=label_path,
			class_info=class_info,
			png_label_inventory=inventory,
			segy_geometry_json=geometry,
			source_label_segy=f3_root / 'f3_labels.sgy',
		),
		outputs=F3LithologyPredictionOutputs(
			output_dir=output_dir,
			token_predictions=output_dir / 'f3_token_predictions.npy',
			probability_volume=output_dir / 'f3_token_probabilities.npy',
			valid_token_grid=output_dir / 'f3_valid_token_grid.npy',
			metadata_json=output_dir / 'prediction_metadata.json',
			validation_slice_metrics_csv=output_dir / 'validation_slice_metrics.csv',
		),
		classes=_classes(),
		token_policy=F3LithologyTokenPolicy(
			min_labeled_fraction=1.0,
			min_majority_fraction=1.0,
			ignore_z_border_samples=0,
		),
		dataset={'name': 'synthetic_f3', 'version': 'test'},
		model={'tag': 'synthetic_encoder', 'freeze_encoder': True},
		embeddings={'spec': 'synthetic', 'input_dir': str(embeddings_dir)},
		labels={'set': 'synthetic_labels'},
		lithology={'root': str(output_dir.parent)},
		probe={'spec': 'linear_test'},
		batch_size=3,
	)


def prediction_fixture_paths(config: F3LithologyPredictionConfig) -> Mapping[str, Path]:
	return {
		'artifact_root': config.outputs.output_dir.parents[4],
		'label_volume': config.inputs.label_volume,
		'class_info': config.inputs.class_info,
		'png_label_inventory': config.inputs.png_label_inventory,
		'segy_geometry_json': config.inputs.segy_geometry_json,
	}


def _dense_label_volume() -> np.ndarray:
	token_classes = _token_classes()
	labels = np.empty((4, 4, 4), dtype=np.int32)
	for token_x in range(2):
		for token_y in range(2):
			for token_z in range(2):
				labels[
					token_x * 2 : token_x * 2 + 2,
					token_y * 2 : token_y * 2 + 2,
					token_z * 2 : token_z * 2 + 2,
				] = token_classes[token_x, token_y, token_z]
	return labels


def _write_embedding_artifacts(output_dir: Path) -> None:
	output_dir.mkdir(parents=True)
	token_classes = _token_classes()
	embeddings = np.zeros((2, 2, 2, 2), dtype=np.float32)
	embeddings[..., 0] = token_classes
	np.save(output_dir / 'f3_facies_benchmark.embeddings.npy', embeddings)
	valid_tokens = np.ones((2, 2, 2), dtype=np.bool_)
	valid_tokens[1, 1, 1] = False
	np.save(output_dir / 'f3_facies_benchmark.valid_tokens.npy', valid_tokens)
	(output_dir / 'f3_facies_benchmark.embedding_metadata.json').write_text(
		json.dumps(
			{
				'patch_size': [2, 2, 2],
				'token_grid_shape': [2, 2, 2],
				'volume_shape_xyz': [4, 4, 4],
			},
			indent=2,
			sort_keys=True,
		)
		+ '\n',
		encoding='utf-8',
	)


def _token_classes() -> np.ndarray:
	return np.asarray(
		[
			[[0, 1], [1, 2]],
			[[2, 0], [1, 2]],
		],
		dtype=np.int32,
	)


def _write_class_info(root: Path) -> Path:
	path = root / 'inspection' / 'f3' / 'class_info.json'
	path.parent.mkdir(parents=True)
	path.write_text(
		json.dumps(
			{
				'class_count': 3,
				'classes': [class_info.to_dict() for class_info in _classes()],
			},
			indent=2,
			sort_keys=True,
		)
		+ '\n',
		encoding='utf-8',
	)
	return path


def _write_inventory(root: Path) -> Path:
	path = root / 'inspection' / 'f3' / 'png_label_inventory.csv'
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		(
			'relative_path,absolute_path,split,slice_type,slice_index\n'
			'interpretation/validation/labels_inline_0101.png,'
			'/tmp/labels_inline_0101.png,validation,inline,101\n'
			'interpretation/validation/labels_crossline_0302.png,'
			'/tmp/labels_crossline_0302.png,validation,crossline,302\n'
		),
		encoding='utf-8',
	)
	return path


def _write_geometry(root: Path) -> Path:
	path = root / 'inspection' / 'f3' / 'segy_geometry.json'
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(
			{
				'segy_files': {
					'label': {
						'cube_shape': [4, 4, 4],
						'iline_min': 100,
						'iline_max': 103,
						'xline_min': 300,
						'xline_max': 303,
					},
				},
			},
			indent=2,
			sort_keys=True,
		)
		+ '\n',
		encoding='utf-8',
	)
	return path


def _classes() -> tuple[F3ClassInfo, ...]:
	return (
		F3ClassInfo(class_id=0, class_name='Class zero', rgb=(1, 2, 3)),
		F3ClassInfo(class_id=1, class_name='Class one', rgb=(35, 92, 167)),
		F3ClassInfo(class_id=2, class_name='Class two', rgb=(9, 8, 7)),
	)
