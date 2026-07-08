from __future__ import annotations

import math
from typing import TYPE_CHECKING

from seis_ssl_cluster.training import load_checkpoint
from seis_ssl_cluster.training.strat_hmm_pretraining import (
	run_strat_hmm_pretext_training,
)
from tests.seis_ssl_cluster.test_strat_hmm_pretraining_head_only import (
	_resolved_config,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_distillation_unfreeze_cpu_smoke_writes_valid_checkpoint(
	tmp_path: Path,
) -> None:
	config = _resolved_config(
		tmp_path,
		encoder_depth=2,
		unfreeze_top_blocks=1,
		distillation_weight=0.2,
		max_steps=1,
	)

	checkpoint_path = run_strat_hmm_pretext_training(config)

	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	assert payload['global_step'] == 1
	assert math.isfinite(payload['metrics']['loss_distillation'])
	assert payload['metrics']['trainable_parameter_count'] > 0.0
	assert payload['trainability_summary']['trainable_names']
