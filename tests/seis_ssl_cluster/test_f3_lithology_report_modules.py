from __future__ import annotations

import importlib


def test_f3_lithology_report_submodules_import() -> None:
	for module_name in (
		'seis_ssl_cluster.f3.lithology.report',
		'seis_ssl_cluster.f3.lithology.report.metrics_loader',
		'seis_ssl_cluster.f3.lithology.report.markdown',
		'seis_ssl_cluster.f3.lithology.report.figures',
		'seis_ssl_cluster.f3.lithology.report.comparison',
		'seis_ssl_cluster.f3.lithology.report.publish',
	):
		importlib.import_module(module_name)


def test_f3_lithology_report_compat_exports_same_build_function() -> None:
	compat = importlib.import_module('seis_ssl_cluster.f3.lithology_report')
	report = importlib.import_module('seis_ssl_cluster.f3.lithology.report')
	comparison = importlib.import_module(
		'seis_ssl_cluster.f3.lithology.report.comparison',
	)

	assert compat.build_f3_lithology_report is report.build_f3_lithology_report
	assert (
		comparison.build_f3_lithology_comparison_report
		is report.build_f3_lithology_comparison_report
	)
