"""Audit the five frozen encoder sources of the F3 lithology comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from seis_ssl_cluster.f3.lithology.five_way_sources import (
	audit_f3_lithology_five_way_sources,
	plan_f3_lithology_five_way_sources,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for the read-only five-way source audit."""
	return build_config_parser(
		'Audit the five F3 lithology encoder sources read-only.',
		config_help='Path to the canonical five-way comparison YAML.',
		dry_run_help=(
			'Print the static source plan and model order without reading '
			'artifacts.'
		),
	)


def main() -> None:
	"""Audit the five sources or print the static dry-run plan."""
	parser = build_parser()
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_lithology_five_way_config_from_mapping,
		config_path=config_path,
	)
	print(f'model_order: {", ".join(config.model_ids)}')
	print(f'survey_id: {config.dataset["name"]}')
	if args.dry_run:
		for row in plan_f3_lithology_five_way_sources(config):
			print(f'{row["model_id"]}.checkpoint: {row["checkpoint"]}')
			print(f'{row["model_id"]}.embeddings_dir: {row["embeddings_dir"]}')
			print(f'{row["model_id"]}.expected: {row["expected"]}')
		print('execution: dry-run; live source audit skipped')
		return

	report = audit_f3_lithology_five_way_sources(config)
	for source in report['sources']:
		print(
			f'{source["model_id"]}: checkpoint_sha256='
			f'{source["checkpoint_sha256"]} '
			f'valid_token_mask={source["valid_token_mask"]}'
		)
	print('five-way source audit passed')


if __name__ == '__main__':
	main()
