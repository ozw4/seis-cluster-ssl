"""Summarize the Parihaka Channel MAE/local-BT four-way benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import argparse

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.parihaka.channel_results import (
	CHANNEL_MAE_LOCAL_BT_MODEL_IDS,
	channel_summary_config_from_mapping,
	inspect_channel_model_results,
	summarize_channel_mae_local_bt_four_way,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'parihaka/facies_benchmark_v1/33_channel_mae_local_bt_four_way_v1'
	/ '03_channel_mae_local_bt_four_way.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the MAE/local-BT four-way summary parser."""
	return build_config_parser(
		'Summarize all 60 Parihaka Channel MAE/local-BT jobs.',
		default_config=DEFAULT_CONFIG,
	)


def main() -> None:
	"""Validate completeness or write the four-way summary files."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = channel_summary_config_from_mapping(raw)
	jobs = inspect_channel_model_results(
		config, model_ids=CHANNEL_MAE_LOCAL_BT_MODEL_IDS
	)
	if args.dry_run:
		print(f'complete_jobs: {len(jobs)}')
		print(f'models: {", ".join(CHANNEL_MAE_LOCAL_BT_MODEL_IDS)}')
		print(f'output_dir: {config.output_dir}')
		print('execution: dry-run; no files written')
		return
	for path in summarize_channel_mae_local_bt_four_way(config):
		print(f'output: {path}')


if __name__ == '__main__':
	main()
