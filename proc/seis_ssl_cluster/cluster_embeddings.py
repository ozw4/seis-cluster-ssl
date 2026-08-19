"""Thin entrypoint for amplitude-only embedding clustering."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config, resolve_clustering_config
from seis_ssl_cluster.utils.cli import print_config_summary

if TYPE_CHECKING:
	import argparse

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[1]
	/ 'configs'
	/ 'seis_ssl_cluster'
	/ 'cluster_embeddings.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for embedding clustering."""
	return build_config_parser(
		'Cluster amplitude-only embeddings.',
		default_config=DEFAULT_CONFIG,
	)


def main() -> None:
	"""Run embedding clustering or print a dry-run summary."""
	parser = build_parser()
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=resolve_clustering_config,
		config_path=config_path,
	)
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
