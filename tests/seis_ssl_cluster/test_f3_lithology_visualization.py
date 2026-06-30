from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.f3 import (
	F3LithologyVisualizationConfig,
	F3LithologyVisualizationFigureConfig,
	F3LithologyVisualizationInputs,
	F3LithologyVisualizationOutputs,
	predict_f3_lithology_tokens,
	visualize_f3_lithology_predictions,
)
from tests.seis_ssl_cluster.test_f3_lithology_prediction import (
	write_prediction_fixture,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_visualize_f3_lithology_predictions_writes_png_sidecars_and_metadata(
	tmp_path: Path,
) -> None:
	pytest.importorskip('matplotlib.pyplot')
	prediction_config = write_prediction_fixture(tmp_path)
	prediction_result = predict_f3_lithology_tokens(prediction_config)
	seismic_path = prediction_config.inputs.label_volume.with_name('f3_seismic.npy')
	np.save(
		seismic_path,
		np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4),
	)
	visualization_config = F3LithologyVisualizationConfig(
		inputs=F3LithologyVisualizationInputs(
			seismic_volume=seismic_path,
			label_volume=prediction_config.inputs.label_volume,
			class_info=prediction_config.inputs.class_info,
			png_label_inventory=prediction_config.inputs.png_label_inventory,
			segy_geometry_json=prediction_config.inputs.segy_geometry_json,
			token_predictions=prediction_result.token_predictions,
			probability_volume=prediction_result.probability_volume,
			prediction_metadata_json=prediction_result.metadata_json,
			validation_slice_metrics_csv=(
				prediction_result.validation_slice_metrics_csv
			),
		),
		outputs=F3LithologyVisualizationOutputs(
			output_dir=(
				prediction_config.outputs.output_dir.parent
				/ 'visualizations'
				/ 'linear_test'
			),
			metadata_json=(
				prediction_config.outputs.output_dir.parent
				/ 'visualizations'
				/ 'linear_test'
				/ 'metadata.json'
			),
			selected_slices_dir=(
				prediction_config.outputs.output_dir.parent
				/ 'visualizations'
				/ 'linear_test'
				/ 'selected_slices'
			),
		),
		classes=prediction_config.classes,
		dataset=prediction_config.dataset,
		model=prediction_config.model,
		labels=prediction_config.labels,
		lithology=prediction_config.lithology,
		probe=prediction_config.probe,
		predictions={'input_dir': str(prediction_config.outputs.output_dir)},
		selected_slices={'inline': (), 'crossline': (), 'z': (2,)},
		figure=F3LithologyVisualizationFigureConfig(dpi=40),
	)

	result = visualize_f3_lithology_predictions(visualization_config)

	validation_png = (
		visualization_config.outputs.output_dir
		/ 'validation_inline_0101_prediction.png'
	)
	selected_png = (
		visualization_config.outputs.selected_slices_dir
		/ 'selected_z_0002_prediction.png'
	)
	sidecar = json.loads(
		validation_png.with_suffix('.json').read_text(encoding='utf-8'),
	)
	metadata = json.loads(result.metadata_json.read_text(encoding='utf-8'))

	assert validation_png.is_file()
	assert validation_png.stat().st_size > 0
	assert selected_png.is_file()
	assert len(result.png_paths) == 3
	assert sidecar['figure_type'] == 'f3_lithology_prediction_slice'
	assert sidecar['display']['z_axis'] == 'down'
	assert sidecar['display']['origin'] == 'upper'
	assert sidecar['figure_config']['background'] == 'white'
	assert sidecar['probe'] == dict(prediction_config.probe)
	assert sidecar['class_legend'][0] == {
		'class_id': 0,
		'class_name': 'Class zero',
		'rgb': [1, 2, 3],
	}
	assert sidecar['token_metrics']['accuracy'] == '1.0'
	assert sidecar['voxel_projection_metrics']['accuracy'] == 1.0
	assert metadata['artifact_type'] == 'f3_lithology_prediction_visualizations'
	assert metadata['figures'][0]['group'] == 'validation'
