"""Render the fixed F3 HMM_v1 publication comparison from completed artifacts."""

from __future__ import annotations

import argparse
import json

from seis_ssl_cluster.f3.lithology.hmm_v1_comparison_visualization import run


def main() -> None:
	"""Print a JSON preview or the generated figure manifest."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', required=True)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--overwrite', action='store_true')
	args = parser.parse_args()
	print(
		json.dumps(
			run(args.config, dry_run=args.dry_run, overwrite=args.overwrite),
			indent=2,
			allow_nan=False,
		)
	)


if __name__ == '__main__':
	main()
