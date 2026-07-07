"""Compatibility wrapper for F3 lithology prediction."""

from __future__ import annotations

from seis_ssl_cluster.f3.lithology.prediction import (
	VALIDATION_SLICE_METRIC_FIELDNAMES,
	F3LithologyPredictionConfig,
	F3LithologyPredictionInputs,
	F3LithologyPredictionOutputs,
	F3LithologyPredictionResult,
	predict_f3_lithology_tokens,
	read_f3_lithology_prediction_classes,
)

__all__ = [
	'VALIDATION_SLICE_METRIC_FIELDNAMES',
	'F3LithologyPredictionConfig',
	'F3LithologyPredictionInputs',
	'F3LithologyPredictionOutputs',
	'F3LithologyPredictionResult',
	'predict_f3_lithology_tokens',
	'read_f3_lithology_prediction_classes',
]
