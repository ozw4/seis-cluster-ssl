"""Thin entrypoint for amplitude-only embedding extraction."""

from __future__ import annotations

from pathlib import Path

from seis_ssl_cluster.cli import (
	add_device_argument,
	add_skip_existing_argument,
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import (
	load_config,
	resolve_embedding_extraction_config,
)
from seis_ssl_cluster.embedding import run_embedding_extraction
from seis_ssl_cluster.utils.cli import print_config_summary

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[1]
	/ 'configs'
	/ 'seis_ssl_cluster'
	/ 'extract_embeddings.yaml'
)


def main() -> None:
	"""Run amplitude-only embedding extraction or print a dry-run summary."""
	parser = build_config_parser(
		'Extract amplitude-only embeddings.',
		default_config=DEFAULT_CONFIG,
	)
	add_device_argument(parser, help_text='Embedding extraction device override.')
	add_skip_existing_argument(
		parser,
		help_text='Skip survey outputs whose metadata already matches this run.',
	)
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=resolve_embedding_extraction_config,
		config_path=config_path,
	)
	if args.dry_run:
		print_config_summary(config, device_override=args.device)
		print('execution: dry-run; extraction skipped')
		return

	results = run_embedding_extraction(
		config,
		skip_existing=args.skip_existing,
		device=args.device,
	)
	for result in results:
		status = 'skipped' if result.skipped else 'written'
		print(f'{result.survey_id}: {status} {result.embeddings_path}')


if __name__ == '__main__':
	main()
