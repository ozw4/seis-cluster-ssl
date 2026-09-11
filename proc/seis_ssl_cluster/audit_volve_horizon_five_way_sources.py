'''Audit Volve horizon five-way checkpoints and embeddings read-only.'''

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.volve.horizon_five_way_config import (
	volve_horizon_five_way_config_from_mapping,
)
from seis_ssl_cluster.volve.horizon_five_way_sources import (
	audit_volve_horizon_five_way_sources,
	inspect_volve_horizon_five_way_embedding_suite,
	plan_volve_horizon_five_way_embeddings,
	plan_volve_horizon_five_way_sources,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	'''Build the read-only five-way preflight parser.'''
	return build_config_parser(
		'Audit Volve horizon five-way checkpoints and embeddings read-only.',
		config_help='Path to the Volve horizon five-way comparison YAML.',
		dry_run_help='Print static source plans without opening any artifact.',
	)


def main() -> None:
	'''Print a static plan or run the complete read-only source preflight.'''
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=volve_horizon_five_way_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		print(
			json.dumps(
				{
					'execution': 'dry-run',
					'model_order': list(config.model_ids),
					'sources': plan_volve_horizon_five_way_sources(config),
					'embeddings': plan_volve_horizon_five_way_embeddings(config),
				},
				indent=2,
				sort_keys=True,
			)
		)
		return

	source_report = audit_volve_horizon_five_way_sources(config)
	suite = inspect_volve_horizon_five_way_embedding_suite(
		config,
		source_audit=source_report,
	)
	print(
		json.dumps(
			{
				'source_audit': source_report,
				'embedding_suite': {
					'model_order': list(suite.sources),
					'volume_shape_xyz': list(suite.volume_shape_xyz),
					'token_grid_shape_xyz': list(suite.token_grid_shape_xyz),
					'embedding_shape': list(suite.embedding_shape),
					'embedding_dim': suite.embedding_dim,
					'valid_tokens_sha256': suite.valid_tokens_sha256,
					'model_valid_lateral_mask_sha256': (
						suite.model_valid_lateral_mask_sha256
					),
					'canonical_identity': dict(suite.canonical_identity),
					'sources': {
						model_id: {
							'embeddings': str(source.paths.embeddings),
							'embeddings_sha256': source.embeddings_sha256,
							'valid_tokens': str(source.paths.valid_tokens),
							'valid_tokens_sha256': source.valid_tokens_sha256,
							'metadata': str(source.paths.metadata),
							'metadata_sha256': source.metadata_sha256,
							'checkpoint_sha256': source.checkpoint_identity[
								'checkpoint_sha256'
							],
						}
						for model_id, source in suite.sources.items()
					},
				},
			},
			indent=2,
			sort_keys=True,
		)
	)


if __name__ == '__main__':
	main()
