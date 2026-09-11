"""Read-only CPU audit of completed F3 HMM sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seis_ssl_cluster.f3.lithology.hmm_final_source_audit import (
	audit_f3_hmm_final_sources,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the command-line parser."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		'--configs',
		type=Path,
		nargs='+',
		required=True,
		help='Full HMM training YAMLs whose final checkpoints must all pass.',
	)
	return parser


def main() -> None:
	"""Print the final source audit; exit nonzero on any incomplete/stale source."""
	args = build_parser().parse_args()
	result = audit_f3_hmm_final_sources(args.configs)
	print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
	main()
