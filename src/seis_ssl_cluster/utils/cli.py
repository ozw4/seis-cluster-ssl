"""Shared helpers for thin seismic SSL clustering procedure entrypoints."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.schema import (
	STAGE_BUILD_MANIFESTS,
	STAGE_CLUSTER_VISUALIZATION,
	STAGE_CLUSTERING,
	STAGE_EMBEDDING_EXTRACTION,
	STAGE_MAE_TRAINING,
	STAGE_NORMALIZATION_QC,
	STAGE_NORMALIZATION_STATS,
)


def parse_config_args(
	description: str,
	default_config: str | Path | None = None,
) -> argparse.Namespace:
	"""Parse common config and dry-run arguments."""
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument(
		'--config',
		type=Path,
		default=Path(default_config) if default_config is not None else None,
		required=default_config is None,
		help='Path to a YAML configuration file.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate the config and print a run summary without executing.',
	)
	return parser.parse_args()


def print_config_summary(
	cfg: Mapping[str, Any],
	*,
	device_override: str | None = None,
) -> None:
	"""Print a compact stage-aware summary of resolved config values."""
	paths = _mapping(cfg.get('paths'))
	stage = cfg.get('stage')
	rows = (
		[]
		if stage in {
			STAGE_EMBEDDING_EXTRACTION,
			STAGE_CLUSTERING,
			STAGE_CLUSTER_VISUALIZATION,
		}
		else _base_summary_rows(stage, paths)
	)

	if stage == STAGE_BUILD_MANIFESTS:
		manifest = _mapping(cfg.get('manifest'))
		rows.extend(
			[
				('manifest.input_path_list', manifest.get('input_path_list')),
				('manifest.output_dir', manifest.get('output_dir')),
				('manifest.output_name', manifest.get('output_name')),
				(
					'manifest.normalization_stats_dir',
					manifest.get('normalization_stats_dir'),
				),
			],
		)
	elif stage == STAGE_NORMALIZATION_STATS:
		manifests = _mapping(cfg.get('manifests'))
		normalization = _mapping(cfg.get('normalization'))
		rows.extend(
			[
				('manifests.train', manifests.get('train')),
				(
					'normalization.clipping_percentiles',
					normalization.get('clipping_percentiles'),
				),
				('normalization.epsilon', normalization.get('epsilon')),
				('normalization.max_samples', normalization.get('max_samples')),
				('normalization.seed', normalization.get('seed')),
			],
		)
	elif stage == STAGE_NORMALIZATION_QC:
		manifests = _mapping(cfg.get('manifests'))
		splits = _mapping(cfg.get('splits'))
		qc = _mapping(cfg.get('qc'))
		rows.extend(
			[
				('manifests.input', manifests.get('input')),
				('manifests.output', manifests.get('output')),
				('splits.input', splits.get('input')),
				('splits.output', splits.get('output')),
				('qc.output_json', qc.get('output_json')),
				('qc.excluded_surveys', qc.get('excluded_surveys')),
				('qc.min_iqr', qc.get('min_iqr')),
				('qc.max_normalized_abs', qc.get('max_normalized_abs')),
			],
		)
	elif stage == STAGE_MAE_TRAINING:
		_add_training_rows(rows, cfg)
	elif stage == STAGE_EMBEDDING_EXTRACTION:
		manifests = _mapping(cfg.get('manifests'))
		embeddings = _mapping(cfg.get('embeddings'))
		embedding = _mapping(cfg.get('embedding'))
		rows.extend(
			[
				('manifests.input', manifests.get('input')),
				('embeddings.checkpoint', embeddings.get('checkpoint')),
				('embeddings.output_dir', embeddings.get('output_dir')),
				('embedding.window_size', embedding.get('window_size')),
				('embedding.overlap', embedding.get('overlap')),
				('embedding.output_dtype', embedding.get('output_dtype')),
				('embedding.batch_size', embedding.get('batch_size')),
				(
					'embedding.min_token_valid_fraction',
					embedding.get('min_token_valid_fraction'),
				),
			],
		)
		if device_override is not None:
			rows.append(('device_override', device_override))
	elif stage == STAGE_CLUSTERING:
		_add_clustering_rows(rows, cfg)
	elif stage == STAGE_CLUSTER_VISUALIZATION:
		_add_cluster_visualization_rows(rows, cfg)

	for key, value in rows:
		print(f'{key}: {_format_value(value)}')


def _add_training_rows(
	rows: list[tuple[str, Any]],
	cfg: Mapping[str, Any],
) -> None:
	manifests = _mapping(cfg.get('manifests'))
	data = _mapping(cfg.get('data'))
	model = _mapping(cfg.get('model'))
	masking = _mapping(cfg.get('masking'))
	loss = _mapping(cfg.get('loss'))
	train = _mapping(cfg.get('train'))
	rows.extend(
		[
			('manifests.train', manifests.get('train')),
			('data.local_crop_size', data.get('local_crop_size')),
			('model.patch_size', model.get('patch_size')),
			('model.encoder_depth', model.get('encoder_depth')),
			('masking.spatial_mask_ratio', masking.get('spatial_mask_ratio')),
			('masking.block_size_tokens', masking.get('block_size_tokens')),
			('loss.huber_delta', loss.get('huber_delta')),
			('loss.gradient_weight', loss.get('gradient_weight')),
			('train.batch_size', train.get('batch_size')),
			('train.epochs', train.get('epochs')),
			('train.device', train.get('device')),
		],
	)


def _add_clustering_rows(
	rows: list[tuple[str, Any]],
	cfg: Mapping[str, Any],
) -> None:
	paths = _mapping(cfg.get('paths'))
	embeddings = _mapping(cfg.get('embeddings'))
	clustering = _mapping(cfg.get('clustering'))
	pca = _mapping(clustering.get('pca'))
	rows.extend(
		[
			('paths.artifact_root', paths.get('artifact_root')),
			('embeddings.input_dir', embeddings.get('input_dir')),
			('clustering.output_dir', clustering.get('output_dir')),
			(
				'clustering.embedding_normalization',
				clustering.get('embedding_normalization'),
			),
			('clustering.pca.enabled', pca.get('enabled')),
			('clustering.pca.n_components', pca.get('n_components')),
			('clustering.pca.whiten', pca.get('whiten')),
			('clustering.sample_tokens', clustering.get('sample_tokens')),
			('clustering.method', clustering.get('method')),
			('clustering.k_values', clustering.get('k_values')),
			('clustering.minibatch_size', clustering.get('minibatch_size')),
			('clustering.seed', clustering.get('seed')),
		],
	)
	if 'prediction_batch_size' in clustering:
		rows.append(
			(
				'clustering.prediction_batch_size',
				clustering.get('prediction_batch_size'),
			),
		)


def _add_cluster_visualization_rows(
	rows: list[tuple[str, Any]],
	cfg: Mapping[str, Any],
) -> None:
	paths = _mapping(cfg.get('paths'))
	clustering = _mapping(cfg.get('clustering'))
	visualization = _mapping(cfg.get('visualization'))
	rows.extend(
		[
			('paths.artifact_root', paths.get('artifact_root')),
			('clustering.input_dir', clustering.get('input_dir')),
			('visualization.output_dir', visualization.get('output_dir')),
			('visualization.survey_ids', visualization.get('survey_ids')),
			('visualization.modes', visualization.get('modes')),
			(
				'visualization.slice_coordinate_space',
				visualization.get('slice_coordinate_space'),
			),
			('visualization.xy_slices', visualization.get('xy_slices')),
			('visualization.xz_slices', visualization.get('xz_slices')),
			(
				'visualization.reconstruct_voxel',
				visualization.get('reconstruct_voxel'),
			),
			(
				'visualization.allow_all_surveys_for_voxel_reconstruction',
				visualization.get('allow_all_surveys_for_voxel_reconstruction'),
			),
			(
				'visualization.skip_existing_voxel_labels',
				visualization.get('skip_existing_voxel_labels'),
			),
			(
				'visualization.max_voxel_output_gib',
				visualization.get('max_voxel_output_gib'),
			),
			(
				'visualization.allow_large_voxel_output',
				visualization.get('allow_large_voxel_output'),
			),
		],
	)


def _base_summary_rows(
	stage: object,
	paths: Mapping[str, Any],
) -> list[tuple[str, Any]]:
	rows: list[tuple[str, Any]] = [('stage', stage)]
	if 'nopims_root' in paths:
		rows.append(('paths.nopims_root', paths.get('nopims_root')))
	rows.append(('paths.artifact_root', paths.get('artifact_root')))
	if 'output_root' in paths:
		rows.append(('paths.output_root', paths.get('output_root')))
	return rows


def run_pending_entrypoint(
	description: str,
	default_config: str | Path | None = None,
	*,
	resolve_config: Callable[[Mapping[str, object]], Mapping[str, object]],
) -> None:
	"""Validate config, print a summary, and report pending execution."""
	args = parse_config_args(description, default_config)
	config = resolve_config(load_config(args.config))
	print_config_summary(config)
	if args.dry_run:
		print('execution: dry-run; implementation pending')
		return

	message = (
		f'execution pending for stage {config.get("stage")!r}; '
		'use --dry-run to validate configuration only'
	)
	raise SystemExit(message)


def _mapping(value: object) -> Mapping[str, Any]:
	if isinstance(value, Mapping):
		return value
	return {}


def _format_value(value: object) -> str:
	if isinstance(value, bool):
		return str(value).lower()
	if isinstance(value, list):
		return ', '.join(str(item) for item in value)
	if value is None:
		return 'null'
	return str(value)


__all__ = [
	'parse_config_args',
	'print_config_summary',
	'run_pending_entrypoint',
]
