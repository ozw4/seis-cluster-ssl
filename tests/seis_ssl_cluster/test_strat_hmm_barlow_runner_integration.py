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
@pytest.mark.parametrize(
	('base_method', 'expected_base_objective', 'expected_local_pairs'),
	[
		pytest.param('barlow_twins', 'barlow_twins_3d', None, id='global'),
		pytest.param(
			'local_barlow_twins',
			'local_barlow_twins_3d',
			4,
			id='local',
		),
	],
)
def test_barlow_source_runner_checkpoint_loads_as_bare_encoder_and_resumes(
	tmp_path: Path,
	base_method: str,
	expected_base_objective: str,
	expected_local_pairs: int | None,
) -> None:
	config, _, _ = _k6_component_fixture(tmp_path, base_method=base_method)
	train = config['train']
	assert isinstance(train, dict)
	train['max_steps'] = 1

	checkpoint_path = run_strat_hmm_pretext_training(config)
	payload = load_checkpoint(checkpoint_path, map_location='cpu')

	assert payload['global_step'] == 1
	assert payload['config']['stage'] == 'barlow_twins_training'
	barlow_twins = payload['config']['barlow_twins']
	assert isinstance(barlow_twins, dict)
	if expected_local_pairs is None:
		assert barlow_twins.get('method', 'barlow_twins_3d') == (
			'barlow_twins_3d'
		)
		assert 'local_pairs_per_crop' not in barlow_twins
	else:
		assert barlow_twins['method'] == expected_base_objective
		assert barlow_twins['local_pairs_per_crop'] == expected_local_pairs
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
	assert metadata['base_objective'] == expected_base_objective
	assert metadata['head_num_prototypes'] == 6

	train['max_steps'] = 2
	resumed_path = run_strat_hmm_pretext_training(config, resume=checkpoint_path)
	resumed_payload = load_checkpoint(resumed_path, map_location='cpu')
	assert resumed_payload['global_step'] == 2
	assert 'projector_state_dict' not in resumed_payload
	assert resumed_payload['stratigraphy_state_dict']
	resumed_metadata = _stratigraphy_pretext_metadata(resumed_payload)
	assert resumed_metadata is not None
	assert resumed_metadata['base_objective'] == expected_base_objective
