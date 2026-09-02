"""Measure fixed representation diagnostics for one overlap-subcrop PoC arm."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from proc.seis_ssl_cluster.run_f3_lithology_overlap_subcrop_poc import (
	poc_model_and_namespace,
)
from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_five_way import (
	f3_lithology_five_way_config_from_mapping,
)
from seis_ssl_cluster.embedding.representation_diagnostics import (
	DEFAULT_REPRESENTATION_LAYER_NORM_EPS,
	DEFAULT_REPRESENTATION_SAMPLE_SIZE,
	EmbeddingRepresentationSource,
	build_embedding_representation_diagnostics,
	write_embedding_representation_diagnostics,
)
from seis_ssl_cluster.embedding.writer import output_paths

if TYPE_CHECKING:
	import argparse
	from pathlib import Path

	from seis_ssl_cluster.config.f3_lithology_five_way import F3FiveWayConfig

F3_TOKEN_GRID_SHAPE = (76, 113, 32)
F3_EMBEDDING_DIM = 384
F3_VALID_MASK_SHA256 = (
	'3bfeb8db8a47420ae7671db90a7e4d6e5a07fceba27648ec76213df3c2b38fd7'
)
F3_SAMPLE_FLAT_INDICES_SHA256 = (
	'44efdbe4e7f7a50caf7c6d1658a5764b4126ac1f0da7ec7d3cc796febfe90de1'
)
DIAGNOSTIC_NAMESPACE = 'local_bt_overlap_subcrop_poc_v1'


def build_parser() -> argparse.ArgumentParser:
	"""Build the fixed PoC representation-diagnostic parser."""
	return build_config_parser(
		'Measure fixed F3 overlap-subcrop representation diagnostics.',
		config_help='Path to random_medium.yaml or <candidate_id>_medium.yaml.',
		dry_run_help='Print fixed diagnostic inputs without writing artifacts.',
	)


def representation_source_from_config(
	config: F3FiveWayConfig,
	*,
	config_path: Path,
) -> EmbeddingRepresentationSource:
	"""Infer the sole measured model source from the documented config filename."""
	model_id, source_id = poc_model_and_namespace(config_path)
	model = config.model_by_id(model_id)
	survey_id = config.dataset['name']
	paths = output_paths(model.embeddings_dir, survey_id)
	return EmbeddingRepresentationSource(
		source_id=source_id,
		survey_id=survey_id,
		checkpoint_path=model.checkpoint,
		embeddings_path=paths.embeddings,
		valid_tokens_path=paths.valid_tokens,
		metadata_path=paths.metadata,
		random_baseline=model_id == 'random',
	)


def representation_diagnostic_output_path(
	config: F3FiveWayConfig,
	*,
	source_id: str,
) -> Path:
	"""Return the fixed per-ID diagnostic artifact path."""
	return (
		config.artifact_root
		/ 'diagnostics/f3'
		/ DIAGNOSTIC_NAMESPACE
		/ 'representation'
		/ f'{source_id}.json'
	)


def diagnostic_plan(
	config: F3FiveWayConfig,
	*,
	config_path: Path,
) -> dict[str, object]:
	"""Return the write-free resolved plan shown by ``--dry-run``."""
	source = representation_source_from_config(config, config_path=config_path)
	return {
		'source_id': source.source_id,
		'source_kind': 'random_baseline' if source.random_baseline else 'candidate',
		'checkpoint': source.checkpoint_path,
		'embeddings': source.embeddings_path,
		'valid_tokens': source.valid_tokens_path,
		'metadata': source.metadata_path,
		'output': representation_diagnostic_output_path(
			config,
			source_id=source.source_id,
		),
		'token_grid_shape': F3_TOKEN_GRID_SHAPE,
		'embedding_dim': F3_EMBEDDING_DIM,
		'sample_size': DEFAULT_REPRESENTATION_SAMPLE_SIZE,
		'sample_flat_indices_sha256': F3_SAMPLE_FLAT_INDICES_SHA256,
		'layer_norm_eps': DEFAULT_REPRESENTATION_LAYER_NORM_EPS,
	}


def main() -> None:
	"""Measure and atomically persist one fixed diagnostic artifact."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_lithology_five_way_config_from_mapping,
		config_path=config_path,
	)
	plan = diagnostic_plan(config, config_path=config_path)
	if args.dry_run:
		for key, value in plan.items():
			print(f'{key}: {value}')
		print('execution: dry-run; no files written')
		return

	source = representation_source_from_config(config, config_path=config_path)
	payload = build_embedding_representation_diagnostics(
		source,
		expected_token_grid_shape=F3_TOKEN_GRID_SHAPE,
		expected_embedding_dim=F3_EMBEDDING_DIM,
		expected_valid_mask_sha256=F3_VALID_MASK_SHA256,
	)
	sampling = payload.get('sampling')
	if not isinstance(sampling, Mapping):
		raise TypeError('diagnostic payload sampling must be a mapping')
	actual_indices_sha256 = sampling.get('sample_flat_indices_sha256')
	if actual_indices_sha256 != F3_SAMPLE_FLAT_INDICES_SHA256:
		raise ValueError(
			'sampled token coordinates differ from the fixed Random-baseline contract'
		)
	output = write_embedding_representation_diagnostics(
		representation_diagnostic_output_path(config, source_id=source.source_id),
		payload,
	)
	print(f'source_id: {source.source_id}')
	metrics = payload.get('metrics')
	if not isinstance(metrics, Mapping):
		raise TypeError('diagnostic payload metrics must be a mapping')
	for key, value in metrics.items():
		print(f'{key}: {value}')
	print(f'written: {output}')


if __name__ == '__main__':
	main()
