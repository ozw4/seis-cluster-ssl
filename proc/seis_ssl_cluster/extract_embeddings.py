"""Thin entrypoint for amplitude-only embedding extraction."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / 'src'
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from seis_ssl_cluster.config import (  # noqa: E402
	load_config,
	resolve_embedding_extraction_config,
)
from seis_ssl_cluster.embedding import run_embedding_extraction  # noqa: E402
from seis_ssl_cluster.utils.cli import print_config_summary  # noqa: E402

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[1]
	/ 'configs'
	/ 'seis_ssl_cluster'
	/ 'extract_embeddings.yaml'
)


def main() -> None:
	"""Run amplitude-only embedding extraction or print a dry-run summary."""
	parser = ArgumentParser(description='Extract amplitude-only embeddings.')
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
	parser.add_argument(
		'--device',
		choices=('auto', 'cpu', 'cuda'),
		help='Embedding extraction device override.',
	)
	parser.add_argument(
		'--skip-existing',
		action='store_true',
		help='Skip survey outputs whose metadata already matches this run.',
	)
	args = parser.parse_args()

	config = resolve_embedding_extraction_config(load_config(args.config))
	if args.dry_run:
		print_config_summary(config)
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
