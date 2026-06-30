from __future__ import annotations

import json
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pytest

from seis_ssl_cluster.f3 import (
	F3ClassInfo,
	F3LithologyProbeConfig,
	F3LithologyProbeInputs,
	F3LithologyProbeOutputs,
	F3LithologyProbeSettings,
	compute_lithology_metrics,
	train_and_evaluate_f3_lithology_probe,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_logistic_regression_probe_trains_writes_and_reloads_artifacts(
	tmp_path: Path,
) -> None:
	config, validation_features = _write_probe_fixture(
		tmp_path,
		F3LithologyProbeSettings(
			spec='linear_balanced_test',
			probe_type='logistic_regression',
			max_iter=500,
			random_state=7,
		),
	)

	result = train_and_evaluate_f3_lithology_probe(config)

	assert result.train_token_count == 30
	assert result.validation_token_count == 12
	assert result.metrics['accuracy'] >= 0.9
	for path in (
		result.probe_joblib,
		result.scaler_joblib,
		result.config_json,
		result.metrics_json,
		result.metrics_csv,
		result.confusion_matrix_csv,
		result.classification_report_md,
		result.confusion_matrix_png,
		result.per_class_f1_png,
	):
		assert path.is_file()

	probe = joblib.load(result.probe_joblib)
	scaler = joblib.load(result.scaler_joblib)
	predicted = probe.predict(scaler.transform(validation_features))
	assert predicted.shape == (12,)
	assert probe.class_weight == 'balanced'

	resolved = json.loads(result.config_json.read_text(encoding='utf-8'))
	assert resolved['encoder_finetuning'] is False
	assert resolved['probe']['type'] == 'logistic_regression'
	assert resolved['training_summary']['class_weight'] == 'balanced'
	report = result.classification_report_md.read_text(encoding='utf-8')
	assert 'macro F1' in report
	assert 'class 5 Zechstein recall' in report


def test_mlp_probe_trains_with_balanced_class_weights(tmp_path: Path) -> None:
	config, validation_features = _write_probe_fixture(
		tmp_path,
		F3LithologyProbeSettings(
			spec='mlp_balanced_test',
			probe_type='mlp',
			hidden_dims=(16,),
			dropout=0.0,
			max_epochs=80,
			early_stopping_patience=15,
			batch_size=8,
			learning_rate=0.01,
			random_state=11,
		),
	)

	result = train_and_evaluate_f3_lithology_probe(config)

	assert result.metrics['macro_f1'] >= 0.75
	probe = joblib.load(result.probe_joblib)
	scaler = joblib.load(result.scaler_joblib)
	predicted = probe.predict(scaler.transform(validation_features))
	assert set(predicted.tolist()) <= {0, 1, 5}
	assert probe.class_weight == {0: 0.5, 1: 2.0, 5: 2.0}
	assert probe.training_epochs <= 80


def test_lithology_metrics_compute_iou_and_confusion_matrix() -> None:
	metrics = compute_lithology_metrics(
		np.asarray([0, 0, 1, 1, 5, 5]),
		np.asarray([0, 1, 1, 1, 0, 5]),
		_classes(),
	)

	assert metrics['confusion_matrix'] == [
		[1, 1, 0],
		[0, 2, 0],
		[1, 0, 1],
	]
	assert np.isclose(metrics['accuracy'], 4 / 6)
	assert np.isclose(metrics['balanced_accuracy'], 2 / 3)
	assert np.isclose(metrics['per_class_iou']['0'], 1 / 3)
	assert np.isclose(metrics['per_class_iou']['1'], 2 / 3)
	assert np.isclose(metrics['per_class_iou']['5'], 1 / 2)
	assert np.isclose(metrics['mean_iou'], 0.5)


def test_invalid_probe_type_raises() -> None:
	with pytest.raises(ValueError, match='probe type'):
		F3LithologyProbeSettings(spec='bad', probe_type='svm')


def _write_probe_fixture(
	tmp_path: Path,
	settings: F3LithologyProbeSettings,
) -> tuple[F3LithologyProbeConfig, np.ndarray]:
	input_dir = tmp_path / 'token_dataset'
	input_dir.mkdir()
	output_dir = tmp_path / 'probes' / settings.spec
	train_features, train_labels = _train_arrays()
	validation_features, validation_labels = _validation_arrays()
	_write_tokens(input_dir / 'train_tokens.npz', train_features, train_labels)
	_write_tokens(
		input_dir / 'validation_tokens.npz',
		validation_features,
		validation_labels,
	)
	config = F3LithologyProbeConfig(
		inputs=F3LithologyProbeInputs(
			train_tokens=input_dir / 'train_tokens.npz',
			validation_tokens=input_dir / 'validation_tokens.npz',
		),
		outputs=F3LithologyProbeOutputs(output_dir=output_dir),
		classes=_classes(),
		probe=settings,
		dataset={'name': 'synthetic_f3'},
		model={'tag': 'synthetic_encoder', 'freeze_encoder': True},
		embeddings={'spec': 'synthetic_tokens'},
		labels={'set': 'synthetic_labels'},
		token_dataset={'input_dir': str(input_dir)},
		lithology={'root': str(tmp_path)},
	)
	return config, validation_features


def _train_arrays() -> tuple[np.ndarray, np.ndarray]:
	return (
		np.asarray(
			[
				[-2.2, -2.0],
				[-2.0, -1.8],
				[-1.8, -2.1],
				[-2.3, -1.7],
				[-1.7, -2.2],
				[-2.1, -2.3],
				[-1.9, -1.9],
				[-2.4, -2.1],
				[-1.6, -1.8],
				[-2.0, -2.4],
				[-2.1, -1.6],
				[-1.7, -2.0],
				[-2.5, -2.2],
				[-1.8, -1.7],
				[-2.2, -2.4],
				[-1.6, -2.1],
				[-2.3, -1.9],
				[-1.9, -2.5],
				[-2.4, -1.8],
				[-1.7, -2.3],
				[0.8, 1.1],
				[1.0, 0.9],
				[1.2, 1.0],
				[0.9, 1.3],
				[1.1, 0.8],
				[2.8, -0.9],
				[3.0, -1.1],
				[3.2, -1.0],
				[2.9, -1.3],
				[3.1, -0.8],
			],
			dtype=np.float32,
		),
		np.asarray([0] * 20 + [1] * 5 + [5] * 5, dtype=np.int64),
	)


def _validation_arrays() -> tuple[np.ndarray, np.ndarray]:
	return (
		np.asarray(
			[
				[-2.1, -2.0],
				[-1.8, -2.2],
				[-2.3, -1.9],
				[-1.9, -1.8],
				[0.9, 1.0],
				[1.1, 1.2],
				[1.3, 0.9],
				[0.8, 1.3],
				[2.9, -1.0],
				[3.1, -1.2],
				[2.8, -0.8],
				[3.2, -1.1],
			],
			dtype=np.float32,
		),
		np.asarray([0] * 4 + [1] * 4 + [5] * 4, dtype=np.int64),
	)


def _write_tokens(path: Path, features: np.ndarray, labels: np.ndarray) -> None:
	count = int(labels.shape[0])
	np.savez_compressed(
		path,
		features=features,
		labels=labels,
		token_xyz=np.zeros((count, 3), dtype=np.int64),
		voxel_center_xyz=np.zeros((count, 3), dtype=np.float32),
	)


def _classes() -> tuple[F3ClassInfo, ...]:
	return (
		F3ClassInfo(class_id=0, class_name='Upper North Sea', rgb=(230, 159, 0)),
		F3ClassInfo(class_id=1, class_name='Middle North Sea', rgb=(86, 180, 233)),
		F3ClassInfo(class_id=5, class_name='Zechstein', rgb=(204, 121, 167)),
	)
