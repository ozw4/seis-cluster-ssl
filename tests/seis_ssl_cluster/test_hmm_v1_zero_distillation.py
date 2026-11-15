"""Contracts for the separate single-head v1 zero-distillation extension."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from seis_ssl_cluster.config import (
	load_config,
	resolve_embedding_extraction_config,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	f3_lithology_candidate_config_from_mapping,
)
from seis_ssl_cluster.hmm.single_head_distillation import (
	OPT_IN,
	PUBLISHED_FILES,
	audit_single_head_source,
	comparison_rows,
	publish_summary,
	validate_matrix,
)
from seis_ssl_cluster.parihaka.channel_decoder import (
	channel_decoder_config_from_mapping,
)
from seis_ssl_cluster.training.strat_hmm.components import build_strat_hmm_components
from seis_ssl_cluster.training.strat_hmm.epoch import (
	train_strat_hmm_head_only_one_epoch,
)
from seis_ssl_cluster.training.strat_hmm.runner import run_strat_hmm_pretext_training
from seis_ssl_cluster.volve.horizon_recipe_arm import (
	as_five_way_config,
	volve_horizon_recipe_arm_config_from_mapping,
)
from tests.seis_ssl_cluster.test_strat_hmm_pretraining_head_only import (
	_raw_config,
	_single_batch_dataloader,
)


def _config(tmp_path):
	return _raw_config(
		tmp_path,
		encoder_depth=2,
		unfreeze_top_blocks=1,
		distillation_weight=0.2,
		max_steps=1,
	)


def test_legacy_guard_and_explicit_single_head_zero_route(tmp_path):
	raw = _config(tmp_path)
	raw['loss']['distillation_weight'] = 0.0
	with pytest.raises(ValueError, match='must be positive'):
		resolve_strat_hmm_pretext_config(raw)
	with pytest.raises(ValueError, match='must be positive'):
		build_strat_hmm_components(raw, device='cpu')
	raw['train'][OPT_IN] = True
	config = resolve_strat_hmm_pretext_config(raw)
	assert config['loss']['distillation_weight'] == 0
	assert config['student']['unfreeze_top_blocks'] == 1


@pytest.mark.parametrize(
	'change', ['positive_weight', 'all_frozen', 'no_prototypes', 'not_bool']
)
def test_opt_in_rejects_incompatible_recipes(tmp_path, change):
	raw = _config(tmp_path)
	raw['loss']['distillation_weight'] = 0.0
	raw['train'][OPT_IN] = True
	if change == 'positive_weight':
		raw['loss']['distillation_weight'] = 0.2
	elif change == 'all_frozen':
		raw['student']['unfreeze_top_blocks'] = 0
	elif change == 'no_prototypes':
		raw['loss']['prototype_weight'] = 0.0
	else:
		raw['train'][OPT_IN] = 'true'
	with pytest.raises(
		(ValueError, TypeError), match='single_head_zero_distillation_ablation'
	):
		resolve_strat_hmm_pretext_config(raw)


def test_zero_route_preserves_initial_head_student_and_rng(tmp_path):
	raw = _config(tmp_path)
	states = []
	for weight in (0.2, 0.0):
		candidate = deepcopy(raw)
		candidate['loss']['distillation_weight'] = weight
		if weight == 0:
			candidate['train'][OPT_IN] = True
		torch.manual_seed(42)
		c = build_strat_hmm_components(
			resolve_strat_hmm_pretext_config(candidate), device='cpu'
		)
		assert c.teacher is not None
		assert all(not p.requires_grad for p in c.teacher.parameters())
		states.append(
			(
				deepcopy(c.student.state_dict()),
				deepcopy(c.head.state_dict()),
				torch.get_rng_state(),
			)
		)
	torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)


def test_zero_distillation_updates_only_top_block(tmp_path):
	raw = _config(tmp_path)
	raw['loss']['distillation_weight'] = 0.0
	raw['train'][OPT_IN] = True
	config = resolve_strat_hmm_pretext_config(raw)
	c = build_strat_hmm_components(config, device='cpu')
	before = deepcopy(c.student.state_dict())
	state = train_strat_hmm_head_only_one_epoch(
		student=c.student,
		teacher=c.teacher,
		head=c.head,
		dataloader=_single_batch_dataloader(config),
		optimizer=c.optimizer,
		device=torch.device('cpu'),
		epoch=1,
		loss_config=config['loss'],
		pseudo_target_config=config['pseudo_targets'],
		max_steps=1,
	)
	changed = [
		name
		for name, value in c.student.state_dict().items()
		if not torch.equal(value, before[name])
	]
	assert changed
	assert all(name.startswith('encoder.layers.1.') for name in changed)
	assert state.metrics['loss_distillation'] == 0
	assert state.metrics['loss_prototype'] > 0
	assert all(p.grad is None for p in c.teacher.parameters())


def test_complete_zero_source_passes_checkpoint_audit(tmp_path):
	raw = _config(tmp_path)
	raw['student']['init_checkpoint'] = raw['teacher']['checkpoint']
	raw['loss']['distillation_weight'] = 0.0
	raw['train'].update(
		{OPT_IN: True, 'epochs': 25, 'max_steps': None, 'samples_per_epoch': 2}
	)
	path = tmp_path / 'full.yaml'
	path.write_text(yaml.safe_dump(raw))
	checkpoint = run_strat_hmm_pretext_training(resolve_strat_hmm_pretext_config(raw))
	report = audit_single_head_source(path)
	assert report['checkpoint'] == str(checkpoint)
	assert report['epoch'] == 25
	assert report['global_step'] == 50
	assert report['distillation_weight'] == 0


def _cells():
	return [
		{
			'survey': survey,
			'source': source,
			'weight': weight,
			'layout_id': f'layout_{layout:03d}',
			'data_size': size,
			'primary': 0.4 + weight + layout * 0.01,
			'secondary': 0.5,
		}
		for survey in ('f3', 'parihaka', 'volve')
		for source in ('mae', 'local_bt', 'random')
		for weight in (0.0, 0.1, 0.2)
		for layout in range(5)
		for size in ('small', 'medium', 'large')
	]


def test_complete_grid_and_signed_paired_differences():
	cells = _cells()
	rows = comparison_rows(cells)
	assert len(rows) == 81
	for r in rows:
		expected = r['distillation_weight'] - 0.2
		if r['survey'] == 'volve':
			expected = -expected
		assert r['improvement_vs_0.2_mean'] == pytest.approx(expected)
		assert r['improvement_vs_0.2_sample_sd'] == pytest.approx(0, abs=1e-15)
	for bad in (cells[:-1], [*cells, cells[0]], [cells[0], *cells[1:-1], cells[0]]):
		with pytest.raises(ValueError, match='405 unique'):
			comparison_rows(bad)


def test_exact_published_files_and_no_overwrite(tmp_path):
	root = tmp_path / 'summary'
	publish_summary(root, _cells())
	assert {p.name for p in root.iterdir()} == PUBLISHED_FILES
	with pytest.raises(FileExistsError):
		publish_summary(root, _cells())


def test_nine_arm_versioned_definition():
	matrix = validate_matrix()
	assert len(matrix['arms']) == 9
	assert matrix['new_downstream_cells'] == 135


def test_all_declared_configs_resolve_and_keep_single_head_lineage(
	tmp_path, monkeypatch
):
	for name in (
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'SEIS_SSL_CLUSTER_VOLVE_ROOT',
		'F3_ROOT',
	):
		monkeypatch.setenv(name, str(tmp_path / name))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	parsers = {
		'f3': f3_lithology_candidate_config_from_mapping,
		'parihaka': channel_decoder_config_from_mapping,
		'volve': volve_horizon_recipe_arm_config_from_mapping,
	}
	for arm in validate_matrix()['arms']:
		for kind in ('training', 'smoke'):
			raw_training = load_config(arm['configs'][kind]['path'])
			Path(raw_training['pseudo_targets']['input_dir']).mkdir(
				parents=True, exist_ok=True
			)
			for value in (
				raw_training['teacher']['checkpoint'],
				raw_training['student']['init_checkpoint'],
			):
				path = Path(value)
				path.parent.mkdir(parents=True, exist_ok=True)
				path.touch()
			config = resolve_strat_hmm_pretext_config(raw_training)
			assert 'spec' not in config['head']
			assert config['head']['num_prototypes'] == 6
		resolve_embedding_extraction_config(
			load_config(arm['configs']['embeddings']['path'])
		)
		raw = load_config(arm['configs']['downstream']['path'])
		parsed = parsers[arm['survey']](raw)
		if arm['survey'] == 'volve':
			assert parsed.single_head_training_config is not None
			assert parsed.multi_head_training_config is None
			model = as_five_way_config(parsed).models[0]
			assert model.expected['stratigraphy_pretext'] is True
			assert model.expected['hmm_k'] == 6
			raw['arm']['multi_head_training_config'] = str(tmp_path / 'other.yaml')
			with pytest.raises(ValueError, match='exactly one continuation'):
				volve_horizon_recipe_arm_config_from_mapping(raw)
