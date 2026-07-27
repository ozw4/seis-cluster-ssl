"""Summarize completed M4 six-split low-label decoder jobs."""
# ruff: noqa: D103, E501

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_split import (
	f3_lithology_voxel_label_budget_split_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_split_results import (
	publish_low_label_split_summary,
	write_low_label_split_summary,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the strict config and summary parser."""
	return build_config_parser(
		'Summarize completed M4 six-split low-label jobs.',
		config_required=True,
		dry_run_help='Validate the completed metrics input without writing.',
	)


def main() -> None:
	args = build_parser().parse_args()
	path = parse_config_path(args)
	config = resolve_config_for_cli(load_config_for_cli(path, loader=load_config), resolver=f3_lithology_voxel_label_budget_split_config_from_mapping, config_path=path)
	metrics = config.output_root / 'low_label_split_job_metrics.csv'
	if not metrics.is_file():
		raise FileNotFoundError(metrics)
	with metrics.open(encoding='utf-8', newline='') as handle:
		rows = list(csv.DictReader(handle))
	if args.dry_run:
		print(f'job_metric_row_count: {len(rows)}')
		return
	paths = write_low_label_split_summary(rows, config.output_root)
	publish_low_label_split_summary(config, paths)
	for name, output in paths.items():
		print(f'{name}: {output}')


if __name__ == '__main__':
	main()
