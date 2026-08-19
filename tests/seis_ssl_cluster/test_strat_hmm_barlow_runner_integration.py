from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from seis_ssl_cluster.embedding.extractor import _stratigraphy_pretext_metadata
from seis_ssl_cluster.models.amplitude_encoder_factory import (
	build_model_from_checkpoint_payload,
)
from seis_ssl_cluster.training import load_checkpoint
from seis_ssl_cluster.training.strat_hmm import run_strat_hmm_pretext_training
from tests.seis_ssl_cluster.test_strat_hmm_pretraining_head_only import (
	_k6_component_fixture,
)

if TYPE_CHECKING:
	from pathlib import Path


@pytest.mark.integration
def test_barlow_source_runner_checkpoint_loads_as_bare_encoder_and_resumes(
	tmp_path: Path,
) -> None:
	config, _, _ = _k6_component_fixture(tmp_path, base_method='barlow_twins')
	train = config['train']
	assert isinstance(train, dict)
	train['max_steps'] = 1

	checkpoint_path = run_strat_hmm_pretext_training(config)
	payload = load_checkpoint(checkpoint_path, map_location='cpu')

	assert payload['global_step'] == 1
	assert payload['config']['stage'] == 'barlow_twins_training'
	assert 'projector_state_dict' not in payload
	assert isinstance(payload['stratigraphy_state_dict'], dict)
	assert payload['stratigraphy_state_dict']

	encoder = build_model_from_checkpoint_payload(payload)
	encoder_state = encoder.state_dict()
	assert set(encoder_state) == set(payload['model_state_dict'])
	assert all(
		torch.equal(encoder_state[name], expected)
		for name, expected in payload['model_state_dict'].items()
	)
	assert set(payload['stratigraphy_state_dict']).isdisjoint(encoder_state)

	metadata = _stratigraphy_pretext_metadata(payload)
	assert metadata is not None
	assert metadata['method'] == 'strat_hmm_pretext'
	assert metadata['base_objective'] == 'barlow_twins_3d'
	assert metadata['head_num_prototypes'] == 6

	train['max_steps'] = 2
	resumed_path = run_strat_hmm_pretext_training(config, resume=checkpoint_path)
	resumed_payload = load_checkpoint(resumed_path, map_location='cpu')
	assert resumed_payload['global_step'] == 2
	assert 'projector_state_dict' not in resumed_payload
	assert resumed_payload['stratigraphy_state_dict']
