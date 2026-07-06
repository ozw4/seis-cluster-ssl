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
