"""Visualize F3 lithology token predictions on seismic slices."""

from __future__ import annotations

from pathlib import Path

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology import (
	f3_lithology_visualization_config_from_mapping,
)
from seis_ssl_cluster.f3 import (
	F3LithologyVisualizationConfig,
	visualize_f3_lithology_predictions,
)

STAGE = 'visualize_f3_lithology_predictions'
DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '50_lithology'
	/ 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	/ 'overlap_x16'
	/ 'png_slices_segy_labels_v1'
	/ '05_visualize_predictions.yaml'
)


def main() -> None:
	"""Write F3 lithology prediction figures or print a dry-run summary."""
	parser = build_config_parser(
		'Visualize F3 lithology predictions.',
		default_config=DEFAULT_CONFIG,
		dry_run_help=(
			'Validate the config and print a run summary without writing figures.'
		),
	)
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=f3_lithology_visualization_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		_print_summary(config)
		print('execution: dry-run; F3 lithology prediction visualization skipped')
		return

	result = visualize_f3_lithology_predictions(config)
	print(f'f3_lithology_visualization.metadata_json: {result.metadata_json}')
	print(f'f3_lithology_visualization.figure_count: {len(result.png_paths)}')


def _print_summary(config: F3LithologyVisualizationConfig) -> None:
	print(f'stage: {STAGE}')
	print(f'registry.seismic_volume: {config.inputs.seismic_volume}')
	print(f'labels.source_label_volume: {config.inputs.label_volume}')
	print(f'labels.class_info: {config.inputs.class_info}')
	print(f'labels.png_label_inventory: {config.inputs.png_label_inventory}')
	print(f'labels.segy_geometry_json: {config.inputs.segy_geometry_json}')
	print(f'predictions.token_predictions: {config.inputs.token_predictions}')
	print(f'predictions.probability_volume: {config.inputs.probability_volume}')
	print(f'predictions.metadata_json: {config.inputs.prediction_metadata_json}')
	print(
		'predictions.validation_slice_metrics_csv: '
		f'{config.inputs.validation_slice_metrics_csv}',
	)
	print(f'model.tag: {config.model.get("tag")}')
	print(f'model.freeze_encoder: {config.model.get("freeze_encoder")}')
	print(f'probe.spec: {config.probe.get("spec")}')
	print(f'visualizations.output_dir: {config.outputs.output_dir}')
	print(f'visualizations.metadata_json: {config.outputs.metadata_json}')
	print(f'visualizations.selected_slices_dir: {config.outputs.selected_slices_dir}')
	print(f'visualizations.slices: {config.selected_slices}')
	print(f'visualizations.figure.dpi: {config.figure.dpi}')
	print(f'visualizations.figure.z_axis: {config.figure.z_axis}')
	print(f'classes.count: {len(config.classes)}')


if __name__ == '__main__':
	main()
