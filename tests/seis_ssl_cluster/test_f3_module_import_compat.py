from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from pathlib import Path


def test_f3_io_modules_and_compat_modules_import() -> None:
	for module_name in (
		'seis_ssl_cluster.f3.labels',
		'seis_ssl_cluster.f3.segy',
		'seis_ssl_cluster.f3.prepare_volume',
		'seis_ssl_cluster.f3.io.labels',
		'seis_ssl_cluster.f3.io.segy',
		'seis_ssl_cluster.f3.io.prepare_volume',
	):
		importlib.import_module(module_name)


def test_class_info_loader_matches_compat_module(tmp_path: Path) -> None:
	class_info_path = tmp_path / 'class_info.json'
	class_info_path.write_text(
		json.dumps(
			{
				'0': {'name': 'Class zero', 'color': [0, 0, 0]},
				'1': {'name': 'Class one', 'color': [35, 92, 167]},
			},
		),
		encoding='utf-8',
	)
	compat_labels = importlib.import_module('seis_ssl_cluster.f3.labels')
	io_labels = importlib.import_module('seis_ssl_cluster.f3.io.labels')

	assert compat_labels.read_class_info(class_info_path) == io_labels.read_class_info(
		class_info_path,
	)


def test_segy_stats_helper_public_import_matches_compat_module() -> None:
	compat_segy = importlib.import_module('seis_ssl_cluster.f3.segy')
	io_segy = importlib.import_module('seis_ssl_cluster.f3.io.segy')
	values = np.asarray([0.0, 1.0, np.nan])

	assert (
		compat_segy.calculate_seismic_amplitude_stats
		is io_segy.calculate_seismic_amplitude_stats
	)
	assert compat_segy.calculate_seismic_amplitude_stats(values)['finite_count'] == 2


def test_f3_refactor_public_module_imports() -> None:
	for module_name in (
		'seis_ssl_cluster.f3.segy',
		'seis_ssl_cluster.f3.png_labels',
		'seis_ssl_cluster.f3.consistency',
		'seis_ssl_cluster.f3.lithology_tokens',
		'seis_ssl_cluster.f3.lithology_prediction',
		'seis_ssl_cluster.f3.baseline_features',
		'seis_ssl_cluster.f3.lithology_probe',
		'seis_ssl_cluster.f3.lithology.prediction',
		'seis_ssl_cluster.f3.lithology.probe',
		'seis_ssl_cluster.f3.lithology_report',
		'seis_ssl_cluster.f3.metrics',
		'seis_ssl_cluster.f3.io.segy',
		'seis_ssl_cluster.f3.lithology.metrics',
		'seis_ssl_cluster.f3.lithology.baselines',
		'seis_ssl_cluster.f3.lithology.token_dataset',
		'seis_ssl_cluster.f3.lithology.tokens',
		'seis_ssl_cluster.f3.lithology.report.comparison',
	):
		importlib.import_module(module_name)


def test_f3_lithology_token_module_import_compatibility() -> None:
	compat_tokens = importlib.import_module('seis_ssl_cluster.f3.lithology_tokens')
	lithology_tokens = importlib.import_module('seis_ssl_cluster.f3.lithology.tokens')

	assert (
		compat_tokens.build_f3_lithology_token_dataset
		is lithology_tokens.build_f3_lithology_token_dataset
	)


def test_f3_lithology_baseline_module_import_compatibility() -> None:
	compat_baselines = importlib.import_module(
		'seis_ssl_cluster.f3.baseline_features',
	)
	lithology_baselines = importlib.import_module(
		'seis_ssl_cluster.f3.lithology.baselines',
	)

	assert (
		compat_baselines.build_f3_lithology_baseline_token_dataset
		is lithology_baselines.build_f3_lithology_baseline_token_dataset
	)


def test_f3_lithology_metrics_import_compatibility() -> None:
	f3 = importlib.import_module('seis_ssl_cluster.f3')
	compat_metrics = importlib.import_module('seis_ssl_cluster.f3.metrics')
	lithology_metrics = importlib.import_module('seis_ssl_cluster.f3.lithology.metrics')

	assert f3.compute_lithology_metrics is lithology_metrics.compute_lithology_metrics
	assert (
		compat_metrics.compute_lithology_metrics
		is lithology_metrics.compute_lithology_metrics
	)


def test_f3_lithology_probe_module_import_compatibility() -> None:
	compat_probe = importlib.import_module('seis_ssl_cluster.f3.lithology_probe')
	lithology_probe = importlib.import_module('seis_ssl_cluster.f3.lithology.probe')

	assert (
		compat_probe.train_and_evaluate_f3_lithology_probe
		is lithology_probe.train_and_evaluate_f3_lithology_probe
	)
	assert compat_probe.F3TorchMLPClassifier is lithology_probe.F3TorchMLPClassifier


def test_f3_lithology_prediction_module_import_compatibility() -> None:
	compat_prediction = importlib.import_module(
		'seis_ssl_cluster.f3.lithology_prediction',
	)
	lithology_prediction = importlib.import_module(
		'seis_ssl_cluster.f3.lithology.prediction',
	)

	assert (
		compat_prediction.predict_f3_lithology_tokens
		is lithology_prediction.predict_f3_lithology_tokens
	)


def test_f3_lithology_visualization_module_import_compatibility() -> None:
	compat_visualization = importlib.import_module(
		'seis_ssl_cluster.f3.lithology_visualization',
	)
	lithology_visualization = importlib.import_module(
		'seis_ssl_cluster.f3.lithology.visualization',
	)

	assert (
		compat_visualization.visualize_f3_lithology_predictions
		is lithology_visualization.visualize_f3_lithology_predictions
	)
