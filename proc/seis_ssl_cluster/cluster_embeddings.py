"""Thin entrypoint for amplitude-only embedding clustering."""

from __future__ import annotations

import importlib
from argparse import ArgumentParser
from pathlib import Path

from seis_ssl_cluster.config import load_config, resolve_clustering_config
from seis_ssl_cluster.utils.cli import print_config_summary

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[1]
	/ 'configs'
	/ 'seis_ssl_cluster'
	/ 'cluster_embeddings.yaml'
)


def main() -> None:
	"""Run embedding clustering or print a dry-run summary."""
	parser = ArgumentParser(description='Cluster amplitude-only embeddings.')
	parser.add_argument(
		'--config',
		type=Path,
		default=DEFAULT_CONFIG,
		help='Path to a YAML configuration file.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate the config and print a run summary without executing.',
	)
	args = parser.parse_args()

	config = resolve_clustering_config(load_config(args.config))
	if args.dry_run:
		print_config_summary(config)
		print('execution: dry-run; clustering skipped')
		return

	run_embedding_clustering = importlib.import_module(
		'seis_ssl_cluster.clustering.kmeans',
	).run_embedding_clustering

	result = run_embedding_clustering(config)
	for k_result in result.results:
		print(
			f'k={k_result.k}: wrote {len(k_result.label_results)} survey label '
			f'file(s) under {k_result.model_dir.parent.parent}',
		)


if __name__ == '__main__':
	main()
