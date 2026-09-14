"""Exercise completed-cell adapters against real survey evidence contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pathlib import Path

import numpy as np
import pytest
import torch

from seis_ssl_cluster.hmm.multi_source_cell import audit_completed_cell
from seis_ssl_cluster.parihaka import channel_decoder
from seis_ssl_cluster.volve.horizon_five_way_results import (
	inspect_volve_horizon_recipe_arm_cell,
)
from seis_ssl_cluster.volve.horizon_five_way_sources import (
	_validate_embedding_objective,
)
from seis_ssl_cluster.volve.horizon_recipe_arm import (
	as_five_way_config,
	inspect_volve_horizon_recipe_arm_embedding_suite,
)
from tests.seis_ssl_cluster.helpers_volve_five_way import write_five_way_completed_run
from tests.seis_ssl_cluster.helpers_volve_recipe_arm import write_recipe_arm_universe
from tests.seis_ssl_cluster.test_parihaka_channel_completion import _make_job
from tests.seis_ssl_cluster.test_parihaka_channel_decoder import (
	_generic_config_mapping,
	_write_generic_sources,
	_write_labels,
	_write_layout,
)


def _complete_horizon_budget(path: Path) -> None:
	metrics = json.loads((path / 'metrics.json').read_text())
	history = json.loads((path / 'history.json').read_text())
	for i in range(len(history), 50):
		history.append(
			{
				**history[-1],
				'epoch': i,
				'validation_macro_mae_samples': history[2][
					'validation_macro_mae_samples'
				]
				+ 1,
			}
		)
	for i, row in enumerate(history):
		row['global_step'] = (i + 1) * 2
	(path / 'history.json').write_text(json.dumps(history))
	torch.save(
		{
			'run_identity': metrics['benchmark_identity'],
			'completed': True,
			'epoch': 50,
			'next_position': 0,
			'global_step': 100,
			'best_epoch': metrics['best_epoch'],
			'history': history,
		},
		path / 'latest.pt',
	)


def test_volve_one_arm_audits_without_baseline_and_rejects_incomplete_budget(
	tmp_path: Path,
) -> None:
	universe = write_recipe_arm_universe(tmp_path, embeddings=True, arm_id='mae')
	config = universe['config']
	suite = inspect_volve_horizon_recipe_arm_embedding_suite(config, model_ids=('mae',))
	five = as_five_way_config(config)
	write_five_way_completed_run(
		five, 'mae', 'layout_000', 'small', embedding_suite=suite
	)
	cell = config.runs_root / 'model=mae/layout=layout_000/size=small'
	_complete_horizon_budget(cell)
	config.baseline_checkpoint.unlink()
	row = inspect_volve_horizon_recipe_arm_cell(
		config, layout_id='layout_000', data_size='small'
	)
	assert row['completion']['epochs'] == 50
	assert 'shared_run_identity' in row
	assert 'support_identity' in row
	latest = torch.load(cell / 'latest.pt', weights_only=False)
	latest['epoch'] = 49
	torch.save(latest, cell / 'latest.pt')
	with pytest.raises(ValueError, match='full epoch budget'):
		inspect_volve_horizon_recipe_arm_cell(
			config, layout_id='layout_000', data_size='small'
		)


def test_channel_completed_adapter_uses_public_plan_identity(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	raw = _generic_config_mapping(tmp_path)
	config = channel_decoder.channel_decoder_config_from_mapping(raw)
	_write_generic_sources(config)
	labels = np.ones((16, 16, 16), dtype=np.int8)
	labels[:, :, ::2] = 5
	_write_labels(config, labels)
	plan = channel_decoder.inspect_channel_decoder_job(
		config,
		model='mae',
		layout_id='layout_000',
		data_size='small',
		layout_config=_write_layout(tmp_path),
	)
	plan = replace(plan, tile_counts={**plan.tile_counts, 'train': 2})
	identity = channel_decoder.channel_decoder_run_identity(plan)
	_make_job(plan.output_dir)
	metrics_path = plan.output_dir / 'metrics.json'
	metrics = json.loads(metrics_path.read_text())
	metrics['benchmark_identity'] = identity
	metrics_path.write_text(json.dumps(metrics))
	for name in ('latest.pt', 'best.pt'):
		path = plan.output_dir / name
		payload = torch.load(path, weights_only=False)
		payload['run_identity'] = identity
		torch.save(payload, path)
	monkeypatch.setattr(
		channel_decoder, 'inspect_channel_decoder_job', lambda *_args, **_kwargs: plan
	)
	result = audit_completed_cell('parihaka', raw, 'mae', 'layout_000', 'small')
	assert result['benchmark_identity'] == identity
	assert result['metrics_path'] == str(metrics_path)
	metrics['benchmark_identity']['training']['seed'] = -1
	metrics_path.write_text(json.dumps(metrics))
	with pytest.raises(ValueError, match='differs from current plan'):
		audit_completed_cell('parihaka', raw, 'mae', 'layout_000', 'small')


def test_volve_multi_head_embedding_metadata_matches_extractor_contract(
	tmp_path: Path,
) -> None:
	universe = write_recipe_arm_universe(tmp_path, embeddings=True)
	config = replace(
		universe['config'], multi_head_training_config=tmp_path / 'full.yaml'
	)
	model = as_five_way_config(config).models[0]
	metadata = {
		'pretraining_method': 'local_barlow_twins_3d',
		'pretraining_objective': {
			'method': 'local_barlow_twins_3d',
			'local_pairs_per_crop': 128,
		},
		'stratigraphy_pretext': {
			'method': 'strat_hmm_multi_head_pretext',
			'base_objective': 'local_barlow_twins_3d',
			'head_spec': 'multi_resolution_ordered_prototypes_v1',
			'head_ks': [6, 8, 10],
			'unfreeze_top_blocks': 1,
			'distillation_weight': 0.2,
			'consistency_weight': 0.0,
		},
	}
	_validate_embedding_objective(model, metadata)
	metadata['stratigraphy_pretext']['method'] = 'strat_hmm_pretext'
	with pytest.raises(ValueError, match='method'):
		_validate_embedding_objective(model, metadata)
