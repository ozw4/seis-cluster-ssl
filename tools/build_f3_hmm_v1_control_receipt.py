"""Generate or reproduce the audit-only frozen F3 Condition 2b receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seis_ssl_cluster.f3.lithology.hmm_v1_freeze_receipt import (
	build_control_freeze_receipt,
	load_control_freeze_receipt,
)


def main() -> None:
	"""Validate the freeze records and emit only the small control receipt."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--freeze-root', type=Path, default=Path('reports/hmm_v1'))
	parser.add_argument('--output', type=Path)
	parser.add_argument(
		'--check',
		action='store_true',
		help='Reproduce and compare an existing receipt without writing.',
	)
	args = parser.parse_args()
	if args.check and args.output is None:
		parser.error('--check requires --output')
	receipt = build_control_freeze_receipt(args.freeze_root)
	if args.check:
		if load_control_freeze_receipt(args.output) != receipt:
			raise ValueError(
				'control receipt differs from the authoritative HMM_v1 freeze records'
			)
		print(json.dumps({'status': 'ok', 'f3_cell_count': receipt['f3_cell_count']}))
		return
	payload = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n'
	if args.output is None:
		print(payload, end='')
	else:
		with args.output.open('x', encoding='utf-8') as stream:
			stream.write(payload)


if __name__ == '__main__':
	main()
