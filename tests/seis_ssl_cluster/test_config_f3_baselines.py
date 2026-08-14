from __future__ import annotations

import seis_ssl_cluster.config.f3_baselines as f3_baselines_config
import seis_ssl_cluster.config.validate as validate_config


def test_f3_baseline_config_entrypoints_reexport_from_validate_module() -> None:
	for name in (
		'f3_lithology_baseline_token_dataset_config_from_mapping',
		'random_mae_checkpoint_config_from_mapping',
	):
		assert getattr(validate_config, name) is getattr(f3_baselines_config, name)
