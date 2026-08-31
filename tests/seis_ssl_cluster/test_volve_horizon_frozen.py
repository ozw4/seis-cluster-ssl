'''Tests for the frozen Volve MAE versus random horizon benchmark.'''

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from seis_ssl_cluster.config import load_config, resolve_embedding_extraction_config
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.volve.horizon_data import HORIZON_NAMES
from seis_ssl_cluster.volve.horizon_frozen import (
	BEST_NAME,
	FROZEN_CONDITION_COUNT,
	LATEST_NAME,
	FrozenHorizonConfig,
	FrozenHorizonTileDataset,
	decoder_initial_state_sha256,
	enumerate_frozen_horizon_conditions,
	frozen_horizon_config_from_mapping,
	inspect_frozen_horizon_job,
	run_frozen_horizon_job,
	validation_mae_improved,
)
from seis_ssl_cluster.volve.horizon_loss import fractional_horizon_cross_entropy
from seis_ssl_cluster.volve.horizon_tiles import (
	HorizonTileSettings,
	frozen_core_output_valid_mask,
	frozen_survey_output_valid_mask,
)
from tests.seis_ssl_cluster.helpers_volve import (
	write_synthetic_frozen_horizon_data,
)

EXPERIMENT_ROOT = Path(
	'experiments/volve/horizon_benchmark_v1/30_mae_vs_random_frozen_v1'
)


class _TinyHorizonDecoder(nn.Module):
	'''A single-parameter decoder preserving the fixed output contract.'''

	def __init__(self) -> None:
		super().__init__()
		self.logits = nn.Parameter(torch.zeros(5, 216))

	def forward(
		self, embeddings: torch.Tensor, _valid: torch.Tensor
	) -> torch.Tensor:
		return self.logits.reshape(1, 5, 1, 1, 216).expand(
			embeddings.shape[0], 5, 64, 64, 216
		)


def test_configs_resolve_and_suite_contains_exactly_thirty_conditions(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts'))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_VOLVE_ROOT', str(tmp_path / 'public'))
	pretrained = resolve_embedding_extraction_config(
		load_config(EXPERIMENT_ROOT / '01_extract_pretrained_embeddings.yaml')
	)
	random_config = resolve_embedding_extraction_config(
		load_config(EXPERIMENT_ROOT / '02_extract_random_embeddings.yaml')
	)
	frozen = frozen_horizon_config_from_mapping(
		load_config(EXPERIMENT_ROOT / '03_horizon_frozen.yaml')
	)

	for extraction in (pretrained, random_config):
		assert extraction['embedding']['window_size'] == [128, 128, 128]
		assert extraction['embedding']['overlap'] == [64, 64, 64]
		assert extraction['embedding']['output_dtype'] == 'float16'
		assert extraction['embedding']['min_token_valid_fraction'] == 1.0
	assert pretrained['embeddings']['checkpoint'] != (
		random_config['embeddings']['checkpoint']
	)
	assert frozen.train.seed == 42000
	assert frozen.train.amp is True
	conditions = enumerate_frozen_horizon_conditions()
	assert len(conditions) == FROZEN_CONDITION_COUNT == 30
	assert len(set(conditions)) == 30
	assert sum(condition[0] == 'pretrained' for condition in conditions) == 15
	assert sum(condition[0] == 'random' for condition in conditions) == 15


def test_paired_job_preflight_reuses_split_and_decoder_identity(
	tmp_path: Path,
) -> None:
	config, data, layout = _write_frozen_fixture(tmp_path)
	pretrained = inspect_frozen_horizon_job(
		config,
		model='pretrained',
		layout_id='layout_002',
		data_size='medium',
		layout_config=layout,
		data=data,
	)
	random_plan = inspect_frozen_horizon_job(
		config,
		model='random',
		layout_id='layout_002',
		data_size='medium',
		layout_config=layout,
		data=data,
	)

	assert pretrained.split_plan.scientific_identity_sha256 == (
		random_plan.split_plan.scientific_identity_sha256
	)
	assert pretrained.run_identity['decoder'] == random_plan.run_identity['decoder']
	assert pretrained.run_identity['tiles'] == random_plan.run_identity['tiles']
	assert pretrained.run_identity['training'] == random_plan.run_identity['training']
	assert pretrained.run_identity['optimizer'] == {
		'name': 'adamw',
		'betas': [0.9, 0.999],
		'eps': 1.0e-8,
		'weight_decay': 1.0e-4,
	}
	assert pretrained.run_identity['objective'] == {
		'loss': 'fractional_two_bin_per_tile_horizon_macro_v1',
		'prediction': 'masked_soft_argmax_v1',
		'checkpoint_selection': 'strict_lower_validation_macro_mae_v1',
		'metrics_schema_version': 1,
	}
	assert pretrained.run_identity['optimizer'] == random_plan.run_identity['optimizer']
	assert pretrained.run_identity['objective'] == random_plan.run_identity['objective']
	assert pretrained.run_identity['canonical_scientific_identity'] == (
		random_plan.run_identity['canonical_scientific_identity']
	)
	assert pretrained.geometry.pretrained_model_source['role'] == 'pretrained'
	assert pretrained.geometry.random_model_source['role'] == 'random'
	assert (
		pretrained.geometry.pretrained_model_source['checkpoint_sha256']
		!= pretrained.geometry.random_model_source['checkpoint_sha256']
	)
	assert pretrained.geometry.valid_tokens_sha256 == file_sha256(
		pretrained.geometry.random.valid_tokens
	)
	assert pretrained.selected_embedding_paths == pretrained.geometry.pretrained
	assert random_plan.selected_embedding_paths == random_plan.geometry.random
	assert pretrained.per_horizon_counts['train'] == tuple(
		pretrained.effective_per_horizon_counts['train']
	)
	assert all(
		effective <= native
		for native, effective in zip(
			pretrained.native_per_horizon_counts['train'],
			pretrained.effective_per_horizon_counts['train'],
			strict=True,
		)
	)
	assert all(count > 0 for count in pretrained.per_horizon_counts['test'])


def test_decoder_seed_and_all_available_section_supervision_are_paired(
	tmp_path: Path,
) -> None:
	config, data, layout = _write_frozen_fixture(tmp_path)
	plan = inspect_frozen_horizon_job(
		config,
		model='pretrained',
		layout_id='layout_000',
		data_size='small',
		layout_config=layout,
		data=data,
	)
	assert plan.native_per_horizon_counts['train'] != (
		plan.effective_per_horizon_counts['train']
	)
	assert any(
		count > 0 for count in plan.excluded_by_token_validity_counts['train']
	)
	for split in ('train', 'validation', 'test', 'test_primary'):
		assert plan.excluded_by_token_validity_counts[split] == tuple(
			native - effective
			for native, effective in zip(
				plan.native_per_horizon_counts[split],
				plan.effective_per_horizon_counts[split],
				strict=True,
			)
		)
	identity = plan.split_plan.identity()

	assert identity['selection_semantics'] == (
		'explicit_section_prefix_all_available_horizon_points_v1'
	)
	assert identity['selected_physical_lines'] == {
		'inline': [100],
		'crossline': [200],
	}
	assert identity['per_horizon_counts']['train'] == {
		name: plan.native_per_horizon_counts['train'][index]
		for index, name in enumerate(HORIZON_NAMES)
	}
	assert plan.run_identity['effective_model_valid_observation_counts'][
		'train'
	] == {
		name: plan.effective_per_horizon_counts['train'][index]
		for index, name in enumerate(HORIZON_NAMES)
	}
	assert plan.run_identity['native_horizon_observation_counts']['train'] == {
		name: plan.native_per_horizon_counts['train'][index]
		for index, name in enumerate(HORIZON_NAMES)
	}
	assert plan.run_identity['excluded_by_token_validity_counts']['train'] == {
		name: plan.excluded_by_token_validity_counts['train'][index]
		for index, name in enumerate(HORIZON_NAMES)
	}
	assert plan.run_identity['decoder']['initialization_seed'] == 42000
	assert plan.run_identity['decoder']['initial_state_sha256'] == (
		decoder_initial_state_sha256()
	)


def test_token_valid_columns_expand_to_output_and_filter_dataset_masks(
	tmp_path: Path,
) -> None:
	settings = HorizonTileSettings(
		lateral_shape_xy=(24, 24), min_token_valid_fraction=1.0
	)
	survey_tokens = np.ones((3, 3, 27), dtype=np.bool_)
	survey_tokens[2, 2, 4] = False
	survey_mask = frozen_survey_output_valid_mask(survey_tokens, settings)
	assert survey_mask.shape == (24, 24)
	assert not survey_mask[16:24, 16:24].any()
	assert survey_mask[:16].all()

	tile_tokens = np.ones(settings.input_size_tokens, dtype=np.bool_)
	tile_tokens[3, 2, 4] = False
	core_mask = frozen_core_output_valid_mask(tile_tokens, settings)
	assert core_mask.shape == (64, 64)
	assert not core_mask[16:24, 8:16].any()
	assert np.count_nonzero(~core_mask) == 64

	config, data, layout = _write_frozen_fixture(tmp_path)
	plan = inspect_frozen_horizon_job(
		config,
		model='pretrained',
		layout_id='layout_000',
		data_size='small',
		layout_config=layout,
		data=data,
	)
	dataset = FrozenHorizonTileDataset(
		data=plan.data,
		plan=plan.split_plan,
		embedding_path=plan.geometry.pretrained.embeddings,
		valid_tokens_path=plan.geometry.pretrained.valid_tokens,
		settings=plan.config.tiles,
		split='train',
		records=plan.tile_records['train'],
	)
	for index in range(len(dataset)):
		item = dataset[index]
		output_valid = item['output_valid_mask'].numpy()
		supervision = item['supervision_mask'].numpy()
		assert not np.any(supervision & ~output_valid[np.newaxis, :, :])


def test_partial_edge_token_is_excluded_from_loss_validation_and_test_masks(
	tmp_path: Path,
) -> None:
	config, data, layout = _write_frozen_fixture(tmp_path, shape_xy=(65, 64))
	valid_path = output_paths(
		config.pretrained_embeddings_dir, config.survey_id
	).valid_tokens
	valid = np.load(valid_path)
	valid[2, 0, :] = False
	valid[3, 3, :] = False
	_write_paired_valid_tokens(config, valid)
	plan = inspect_frozen_horizon_job(
		config,
		model='pretrained',
		layout_id='layout_000',
		data_size='small',
		layout_config=layout,
		data=data,
	)

	assert not plan.geometry.model_valid_lateral_mask[64, :].any()
	assert all(
		record.core_start_token_xy[0] != 8
		for records in plan.tile_records.values()
		for record in records
	)
	assert all(
		count > 0
		for count in plan.excluded_by_token_validity_counts['validation']
	)
	assert all(
		count > 0 for count in plan.excluded_by_token_validity_counts['test']
	)
	assert all(
		count > 0
		for count in plan.excluded_by_token_validity_counts['test_primary']
	)

	selected_paths = plan.geometry.pretrained
	for split in ('train', 'validation', 'test'):
		dataset = FrozenHorizonTileDataset(
			data=plan.data,
			plan=plan.split_plan,
			embedding_path=selected_paths.embeddings,
			valid_tokens_path=selected_paths.valid_tokens,
			settings=plan.config.tiles,
			split=split,
			records=plan.tile_records[split],
		)
		for index in range(len(dataset)):
			item = dataset[index]
			output_valid = item['output_valid_mask']
			assert not torch.any(
				item['supervision_mask'] & ~output_valid.unsqueeze(0)
			)
			assert not torch.any(
				item['primary_evaluation_mask'] & ~output_valid.unsqueeze(0)
			)

	test_dataset = FrozenHorizonTileDataset(
		data=plan.data,
		plan=plan.split_plan,
		embedding_path=selected_paths.embeddings,
		valid_tokens_path=selected_paths.valid_tokens,
		settings=plan.config.tiles,
		split='test',
		records=plan.tile_records['test'],
	)
	item = test_dataset[0]
	assert torch.isfinite(item['target_sample_float'][:, 24, 24]).all()
	assert not item['output_valid_mask'][24, 24]
	assert not item['supervision_mask'][:, 24, 24].any()
	assert not item['primary_evaluation_mask'][:, 24, 24].any()
	logits = torch.zeros((1, 5, 64, 64, 216), dtype=torch.float32)
	target = item['target_sample_float'].unsqueeze(0)
	mask = item['supervision_mask'].unsqueeze(0)
	baseline, _ = fractional_horizon_cross_entropy(logits, target, mask)
	logits[:, :, 24, 24, :] = torch.linspace(-100.0, 100.0, 216)
	changed, _ = fractional_horizon_cross_entropy(logits, target, mask)
	assert torch.equal(baseline, changed)


def test_preflight_requires_model_valid_validation_and_primary_common_test(
	tmp_path: Path,
) -> None:
	config, data, layout = _write_frozen_fixture(tmp_path / 'validation')
	bound = np.array(data.bound_valid_mask, copy=True)
	bound[0, 20, :] = False
	bound[0, :, 20] = False
	bound[0, 20, 20] = True
	data = replace(
		data,
		bound_valid_mask=bound,
		source_present_mask=bound.copy(),
		common_bound_mask=np.all(bound, axis=0),
		continuous_strict_order_mask=np.all(bound, axis=0),
		sample_strict_order_mask=np.all(bound, axis=0),
	)
	valid = np.load(
		output_paths(
			config.pretrained_embeddings_dir, config.survey_id
		).valid_tokens
	)
	valid[2, 2, :] = False
	_write_paired_valid_tokens(config, valid)
	with pytest.raises(
		ValueError,
		match='validation has zero model-valid observations for horizons: ty_top',
	):
		inspect_frozen_horizon_job(
			config,
			model='pretrained',
			layout_id='layout_000',
			data_size='small',
			layout_config=layout,
			data=data,
		)

	config, data, layout = _write_frozen_fixture(tmp_path / 'primary')
	valid = np.load(
		output_paths(
			config.pretrained_embeddings_dir, config.survey_id
		).valid_tokens
	)
	valid[2, 2, :] = False
	_write_paired_valid_tokens(config, valid)
	with pytest.raises(ValueError, match='primary common test has zero model-valid'):
		inspect_frozen_horizon_job(
			config,
			model='pretrained',
			layout_id='layout_000',
			data_size='small',
			layout_config=layout,
			data=data,
		)


def test_preflight_rejects_mismatched_valid_tokens_and_checkpoint_roles(
	tmp_path: Path,
) -> None:
	config, data, layout = _write_frozen_fixture(tmp_path)
	random_paths = output_paths(config.random_embeddings_dir, config.survey_id)
	mask = np.load(random_paths.valid_tokens)
	mask[1, 1, 0] = False
	np.save(random_paths.valid_tokens, mask)
	with pytest.raises(ValueError, match='valid-token masks differ'):
		inspect_frozen_horizon_job(
			config,
			model='random',
			layout_id='layout_000',
			data_size='small',
			layout_config=layout,
			data=data,
		)

	config, data, layout = _write_frozen_fixture(tmp_path / 'role')
	random_paths = output_paths(config.random_embeddings_dir, config.survey_id)
	metadata = json.loads(random_paths.metadata.read_text(encoding='utf-8'))
	pretrained_metadata_path = output_paths(
		config.pretrained_embeddings_dir, config.survey_id
	).metadata
	metadata['checkpoint_sha256'] = json.loads(
		pretrained_metadata_path.read_text(encoding='utf-8')
	)['checkpoint_sha256']
	random_paths.metadata.write_text(json.dumps(metadata), encoding='utf-8')
	with pytest.raises(ValueError, match=r'does not match its file|must differ'):
		inspect_frozen_horizon_job(
			config,
			model='random',
			layout_id='layout_000',
			data_size='small',
			layout_config=layout,
			data=data,
		)


def test_two_step_cpu_resume_and_identity_mismatch_rejection(tmp_path: Path) -> None:
	config, data, layout = _write_frozen_fixture(tmp_path)
	plan = inspect_frozen_horizon_job(
		config,
		model='pretrained',
		layout_id='layout_001',
		data_size='small',
		layout_config=layout,
		data=data,
	)
	uninterrupted_plan = replace(
		plan, output_dir=plan.output_dir.with_name(f'{plan.output_dir.name}_continuous')
	)
	assert run_frozen_horizon_job(
		uninterrupted_plan,
		device='cpu',
		max_steps=2,
		decoder_factory=_TinyHorizonDecoder,
	) is None
	uninterrupted = torch.load(
		uninterrupted_plan.output_dir / LATEST_NAME,
		map_location='cpu',
		weights_only=False,
	)

	assert run_frozen_horizon_job(
		plan,
		device='cpu',
		max_steps=1,
		decoder_factory=_TinyHorizonDecoder,
	) is None
	latest = plan.output_dir / LATEST_NAME
	first = torch.load(latest, map_location='cpu', weights_only=False)
	assert first['global_step'] == 1
	expected_precision = {
		'device_type': 'cpu',
		'amp_enabled': False,
		'autocast_dtype': None,
		'scaler_required': False,
	}
	assert first['runtime_precision'] == expected_precision
	assert first['run_identity']['runtime_precision'] == expected_precision
	assert first['scaler_state_dict'] is None
	assert run_frozen_horizon_job(
		plan,
		device='cpu',
		max_steps=2,
		resume=latest,
		decoder_factory=_TinyHorizonDecoder,
	) is None
	second = torch.load(latest, map_location='cpu', weights_only=False)
	assert second['global_step'] == 2
	_assert_nested_equal(
		second['model_state_dict'], uninterrupted['model_state_dict']
	)
	_assert_nested_equal(
		second['optimizer_state_dict'], uninterrupted['optimizer_state_dict']
	)
	assert second['history'] == uninterrupted['history']
	assert second['best_epoch'] == uninterrupted['best_epoch']
	assert second['best_validation_macro_mae_samples'] == (
		uninterrupted['best_validation_macro_mae_samples']
	)

	changed_identity = {
		**plan.run_identity,
		'canonical_scientific_identity': {
			**plan.run_identity['canonical_scientific_identity'],
			'canonical_amplitude_sha256': 'b' * 64,
		},
	}
	changed = replace(plan, run_identity=changed_identity)
	with pytest.raises(ValueError, match='does not match'):
		run_frozen_horizon_job(
			changed,
			device='cpu',
			max_steps=3,
			resume=latest,
			decoder_factory=_TinyHorizonDecoder,
		)

	second['runtime_precision'] = {
		'device_type': 'cuda',
		'amp_enabled': True,
		'autocast_dtype': 'float16',
		'scaler_required': True,
	}
	torch.save(second, latest)
	with pytest.raises(ValueError, match='runtime precision'):
		run_frozen_horizon_job(
			plan,
			device='cpu',
			max_steps=3,
			resume=latest,
			decoder_factory=_TinyHorizonDecoder,
		)


def test_completed_job_selects_strict_best_and_tests_it_once(tmp_path: Path) -> None:
	config, data, layout = _write_frozen_fixture(tmp_path)
	plan = inspect_frozen_horizon_job(
		config,
		model='random',
		layout_id='layout_004',
		data_size='large',
		layout_config=layout,
		data=data,
	)
	short_train = replace(plan.config.train, epochs=2)
	short_config = replace(plan.config, train=short_train)
	short_identity = {
		**plan.run_identity,
		'training': {**plan.run_identity['training'], 'epochs': 2},
	}
	plan = replace(plan, config=short_config, run_identity=short_identity)

	metrics_path = run_frozen_horizon_job(
		plan,
		device='cpu',
		decoder_factory=_TinyHorizonDecoder,
	)

	assert metrics_path == plan.output_dir / 'metrics.json'
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	expected_precision = {
		'device_type': 'cpu',
		'amp_enabled': False,
		'autocast_dtype': None,
		'scaler_required': False,
	}
	assert metrics['runtime_precision'] == expected_precision
	assert metrics['benchmark_identity']['runtime_precision'] == expected_precision
	assert metrics['test']['evaluation_pass_count'] == 1
	assert metrics['test']['primary_common']['macro_mae_samples'] is not None
	secondary_coverage = metrics['test']['secondary_per_horizon']['coverage']
	assert secondary_coverage['eligible_count'] == sum(
		plan.per_horizon_counts['test']
	)
	assert metrics['test']['primary_common']['coverage']['eligible_count'] == sum(
		plan.effective_per_horizon_counts['test_primary']
	)
	assert metrics['validation']['coverage']['eligible_count'] == sum(
		plan.effective_per_horizon_counts['validation']
	)
	best = torch.load(
		plan.output_dir / BEST_NAME, map_location='cpu', weights_only=False
	)
	latest = torch.load(
		plan.output_dir / LATEST_NAME, map_location='cpu', weights_only=False
	)
	assert latest['completed'] is True
	assert best['runtime_precision'] == expected_precision
	assert latest['runtime_precision'] == expected_precision
	assert all(
		torch.equal(best['model_state_dict'][key], latest['model_state_dict'][key])
		for key in best['model_state_dict']
	)
	history = json.loads(
		(plan.output_dir / 'history.json').read_text(encoding='utf-8')
	)
	validation = [row['validation_macro_mae_samples'] for row in history]
	assert metrics['best_epoch'] == min(
		range(len(validation)), key=validation.__getitem__
	)
	assert not list(plan.output_dir.glob('*probab*'))
	assert {path.name for path in plan.output_dir.iterdir()} == {
		'latest.pt',
		'best.pt',
		'history.json',
		'metrics.json',
	}


def test_proc_entrypoint_exposes_required_one_job_arguments() -> None:
	module = importlib.import_module('proc.seis_ssl_cluster.run_volve_horizon_frozen')
	parser = module.build_parser()
	args = parser.parse_args(
		[
			'--model',
			'pretrained',
			'--layout',
			'layout_003',
			'--size',
			'medium',
			'--dry-run',
		]
	)
	assert args.model == 'pretrained'
	assert args.layout == 'layout_003'
	assert args.size == 'medium'
	assert args.dry_run is True
	assert args.max_steps is None
	assert args.resume is None
	assert validation_mae_improved(1.0, 2.0)
	assert not validation_mae_improved(1.0, 1.0)


def _write_frozen_fixture(
	tmp_path: Path,
	*,
	shape_xy: tuple[int, int] = (24, 24),
) -> tuple[FrozenHorizonConfig, object, Path]:
	data = write_synthetic_frozen_horizon_data(tmp_path, shape_xy=shape_xy)
	artifact_root = (tmp_path / 'artifacts').resolve()
	artifact_root.mkdir(parents=True)
	canonical_root = data.paths.valid_trace_mask.parent
	amplitude_path = canonical_root / 'amplitude.npy'
	np.save(amplitude_path, np.zeros((1,), dtype=np.float32))
	normalization_path = (
		artifact_root
		/ 'data/volve/horizon_benchmark_v1/volve.normalization_stats.json'
	)
	normalization_path.parent.mkdir(parents=True)
	normalization_path.write_text('{"synthetic": true}\n', encoding='utf-8')
	metadata_path = normalization_path.with_name(
		'volve_canonical_input_metadata.json'
	)
	identity = {
		'dataset_id': 'synthetic_frozen_volve',
		'survey_id': 'volve_st10010',
		'shape_xyz': [*shape_xy, 800],
		'canonical_amplitude_sha256': file_sha256(amplitude_path),
		'valid_trace_mask_sha256': file_sha256(data.paths.valid_trace_mask),
		'inline_values_sha256': file_sha256(data.paths.inline_values),
		'crossline_values_sha256': file_sha256(data.paths.crossline_values),
		'time_axis_sha256': file_sha256(data.paths.time_ms),
		'canonical_normalization_stats_sha256': 'a' * 64,
	}
	metadata = {
		'schema_version': 1,
		'artifact_type': 'volve_canonical_input_registration',
		'status': 'PASS',
		'scientific_identity': identity,
		'scientific_identity_sha256': _json_sha256(identity),
		'provenance': {
			'amplitude': {'path': str(amplitude_path)},
			'public_inputs': {
				'valid_trace_mask.npy': str(data.paths.valid_trace_mask),
			},
		},
		'outputs': {'normalization_stats': str(normalization_path)},
	}
	metadata_path.write_text(
		json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)
	pretrained_checkpoint = (
		artifact_root
		/ 'pretraining/volve/horizon_benchmark_v1'
		/ 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
		/ 'full_100ep/latest.pt'
	)
	pretrained_checkpoint.parent.mkdir(parents=True)
	checkpoint_config = {
		'model': _mae_model_config(),
		'train': {'epochs': 100, 'seed': 42},
	}
	torch.save(
		{
			'config': checkpoint_config,
			'epoch': 100,
			'training_state': {
				'stage': 'train_amp_mae',
				'checkpoint_kind': 'epoch',
			},
		},
		pretrained_checkpoint,
	)
	inputs_root = pretrained_checkpoint.parent / 'inputs'
	inputs_root.mkdir()
	(inputs_root / metadata_path.name).write_bytes(metadata_path.read_bytes())
	(inputs_root / normalization_path.name).write_bytes(
		normalization_path.read_bytes()
	)
	(pretrained_checkpoint.parent / 'run_metadata.json').write_text(
		json.dumps(
			{
				'input_scientific_identity_sha256': _json_sha256(identity),
				'canonical_input_metadata_sha256': file_sha256(metadata_path),
				'normalization_stats_sha256': file_sha256(normalization_path),
			}
		),
		encoding='utf-8',
	)
	random_checkpoint = (
		artifact_root
		/ 'pretraining/volve/horizon_benchmark_v1'
		/ 'random_encoder_amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_seed42_v1'
		/ 'random_init/mae_random_seed42.pt'
	)
	random_checkpoint.parent.mkdir(parents=True)
	torch.save(
		{
			'config': checkpoint_config,
			'epoch': 0,
			'metadata': {
				'random_encoder_baseline': True,
				'pretrained_weights_loaded': False,
				'seed': 42,
				'reference_model_tag': (
					'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
				),
				'reference_checkpoint': str(pretrained_checkpoint),
			},
			'training_state': {'checkpoint_kind': 'random_init'},
		},
		random_checkpoint,
	)
	pretrained_dir = artifact_root / 'embeddings/pretrained'
	random_dir = artifact_root / 'embeddings/random'
	token_shape = (
		(shape_xy[0] + 7) // 8,
		(shape_xy[1] + 7) // 8,
		100,
	)
	valid_tokens = np.ones(token_shape, dtype=np.bool_)
	valid_tokens[0, 0] = False
	if shape_xy[0] % 8:
		valid_tokens[-1, :, :] = False
	if shape_xy[1] % 8:
		valid_tokens[:, -1, :] = False
	common_metadata = {
		'survey_id': 'volve_st10010',
		'source_amplitude_path': str(amplitude_path),
		'source_valid_mask_path': str(data.paths.valid_trace_mask),
		'volume_shape_xyz': [*shape_xy, 800],
		'model_geometry': {'name': 'amp_mae3d', **_mae_model_config()},
		'patch_size': [8, 8, 8],
		'token_grid_shape': list(token_shape),
		'window_size': [128, 128, 128],
		'overlap': [64, 64, 64],
		'output_dtype': 'float16',
		'precision': {'device_type': 'cpu', 'autocast': False},
		'min_token_valid_fraction': 1.0,
		'normalization_stats_path': str(normalization_path),
		'normalized_clip_abs': 8.0,
		'amplitude_agc': {'enabled': True, 'mode': 'trace_rms_z'},
		'finite_check_mode': 'strict',
		'preprocessing': {'normalized_clip_abs': 8.0},
		'zero_mask': {'enabled': True},
		'pretraining_objective': {'reconstruction': 'mse'},
	}
	for directory, checkpoint in (
		(pretrained_dir, pretrained_checkpoint),
		(random_dir, random_checkpoint),
	):
		paths = output_paths(directory, 'volve_st10010')
		directory.mkdir(parents=True)
		np.save(paths.embeddings, np.zeros((*token_shape, 384), dtype=np.float16))
		np.save(paths.valid_tokens, valid_tokens)
		paths.metadata.write_text(
			json.dumps(
				{
					**common_metadata,
					'checkpoint_path': str(checkpoint),
					'checkpoint_sha256': file_sha256(checkpoint),
				}
			),
			encoding='utf-8',
		)
	layout_path = tmp_path / 'layouts.yaml'
	layout_path.write_text(
		yaml.safe_dump(
			{
				'selection': {
					'semantics': (
						'explicit_section_prefix_all_available_horizon_points_v1'
					)
				},
				'validation': {'inline': [120], 'crossline': [220]},
				'layouts': {
					f'layout_{index:03d}': {
						'inline': list(range(100 + 4 * index, 104 + 4 * index)),
						'crossline': list(range(200 + 4 * index, 204 + 4 * index)),
					}
					for index in range(5)
				},
			},
			sort_keys=False,
		),
		encoding='utf-8',
	)
	config = frozen_horizon_config_from_mapping(
		{
			'paths': {
				'artifact_root': str(artifact_root),
				'volve_root': str(tmp_path / 'public'),
			},
			'dataset': {'survey_id': 'volve_st10010'},
			'inputs': {'canonical_input_metadata': str(metadata_path)},
			'embeddings': {
				'pretrained_dir': str(pretrained_dir),
				'random_dir': str(random_dir),
			},
			'outputs': {'runs_root': str(artifact_root / 'runs')},
			'decoder': {
				'embedding_dim': 384,
				'class_count': 5,
				'hidden_channels': [128, 64, 32],
				'upsample_factors': [[2, 2, 2]] * 3,
				'upsample_mode': 'nearest',
				'normalization': 'voxelwise_layer_norm',
			},
			'tiles': {
				'patch_size': [8, 8, 8],
				'core_size_tokens': [8, 8, 27],
				'context_halo_tokens': [1, 1, 0],
				'window_start': 552,
				'window_stop': 768,
				'min_token_valid_fraction': 1.0,
			},
			'train': {
				'epochs': 50,
				'batch_size': 1,
				'learning_rate': 1.0e-3,
				'weight_decay': 1.0e-4,
				'sampling_mode': 'all_tiles_once',
				'seed': 42000,
				'amp': True,
				'gradient_clip_norm': 1.0,
			},
		}
	)
	return config, data, layout_path


def _write_paired_valid_tokens(
	config: FrozenHorizonConfig, valid_tokens: np.ndarray
) -> None:
	for directory in (
		config.pretrained_embeddings_dir,
		config.random_embeddings_dir,
	):
		np.save(output_paths(directory, config.survey_id).valid_tokens, valid_tokens)


def _mae_model_config() -> dict[str, object]:
	return {
		'in_channels': 1,
		'out_channels': 1,
		'patch_size': [8, 8, 8],
		'encoder_dim': 384,
		'encoder_depth': 8,
		'encoder_heads': 6,
		'decoder_dim': 256,
		'decoder_depth': 4,
		'decoder_heads': 4,
	}


def _json_sha256(value: Mapping[str, object]) -> str:
	return hashlib.sha256(
		json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
	).hexdigest()


def _assert_nested_equal(left: object, right: object) -> None:
	if isinstance(left, torch.Tensor):
		assert isinstance(right, torch.Tensor)
		assert torch.equal(left, right)
		return
	if isinstance(left, Mapping):
		assert isinstance(right, Mapping)
		assert left.keys() == right.keys()
		for key in left:
			_assert_nested_equal(left[key], right[key])
		return
	if isinstance(left, (list, tuple)):
		assert isinstance(right, type(left))
		assert len(left) == len(right)
		for left_item, right_item in zip(left, right, strict=True):
			_assert_nested_equal(left_item, right_item)
		return
	assert left == right
