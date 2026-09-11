"""Print read-only final-source audit evidence as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.training.strat_hmm_source_audit import audit_multi_head_source


def main() -> None:
	"""Audit all configured full recipes; dry-run only lists the intended reads."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--dry-run', action='store_true')
	args = parser.parse_args()
	config = load_config(args.config)
	paths = config.get('training_configs')
	if set(config) != {'training_configs'} or not isinstance(paths, list) or not paths:
		raise ValueError('audit config requires a nonempty training_configs list')
	paths = [Path(path) for path in paths]
	if len({path.resolve() for path in paths}) != len(paths):
		raise ValueError('duplicate training configs')
	if args.dry_run:
		result = {
			'status': 'planned',
			'training_configs': [str(path) for path in paths],
		}
	else:
		sources = [audit_multi_head_source(path) for path in paths]
		if len({source['checkpoint']['path'] for source in sources}) != len(sources):
			raise ValueError('duplicate checkpoint sources')
		if any(
			source['source_embedding'] != sources[0]['source_embedding']
			for source in sources
		):
			raise ValueError('source embedding identities differ between candidates')
		result = {'status': 'complete', 'sources': sources}
	print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
	main()
