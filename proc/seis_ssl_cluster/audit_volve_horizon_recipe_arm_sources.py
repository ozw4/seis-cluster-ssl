'''Audit declared Volve recipe checkpoints before embedding extraction.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.volve.horizon_recipe_arm import (
	audit_volve_horizon_recipe_arm_sources,
	volve_horizon_recipe_arm_config_from_mapping,
)


def main() -> None:
	'''Print the selected live checkpoint lineage audits.'''
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--models', nargs='+')
	args = parser.parse_args()
	config = volve_horizon_recipe_arm_config_from_mapping(load_config(args.config))
	result = audit_volve_horizon_recipe_arm_sources(config, model_ids=args.models)
	print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
	main()
