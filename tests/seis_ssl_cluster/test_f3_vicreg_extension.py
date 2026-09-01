from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_five_way import FIVE_WAY_MODEL_IDS
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	FIXED_DECODER_CONTRACT,
	LAYOUT_IDS,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology import vicreg_benchmark
from seis_ssl_cluster.f3.lithology.five_way_results import (
	EXPECTED_AGGREGATION_UNIT,
	SUMMARY_METRICS,
)
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	FIVE_WAY_EVALUATION_POLICY,
	FIVE_WAY_TILE_SETTINGS,
)
from seis_ssl_cluster.f3.lithology.vicreg_benchmark import (
	BENCHMARK_SUMMARY_OUTPUT_NAMES,
	EXTENSION_MODEL_IDS,
	SCREENING_MODEL_IDS,
	SCREENING_SUMMARY_OUTPUT_NAMES,
	SEVEN_WAY_MODEL_IDS,
	VICREG_GATE_FAIL,
	VICREG_GATE_PASS,
	assert_f3_vicreg_full_benchmark_ready,
	audit_f3_vicreg_screening_source,
	audit_f3_vicreg_sources,
	f3_vicreg_extension_config_from_mapping,
	inspect_f3_vicreg_combined_results,
	inspect_f3_vicreg_extension_results,
	load_f3_vicreg_canonical_config,
	plan_f3_vicreg_extension_jobs,
	plan_f3_vicreg_screening_jobs,
	resolve_f3_vicreg_screening_job,
	summarize_f3_vicreg_combined,
	summarize_f3_vicreg_extension,
	summarize_f3_vicreg_screening,
)
from seis_ssl_cluster.stratigraphy import (
	discover_pseudo_target_inputs,
	write_pseudo_target,
)
from seis_ssl_cluster.training.checkpoint import load_checkpoint
from tests.seis_ssl_cluster.helpers_f3_five_way import (
	SURVEY_ID,
	build_five_way_universe,
	write_condition,
)

VALIDATION_VOXEL_COUNT = 496
VICREG_METHOD = 'local_vicreg_3d'


def _vicreg_config(
	*, epochs: int, continuation: Path | None = None
) -> dict[str, object]:
	config: dict[str, object] = {
		'stage': 'vicreg_training',
		'data': {
			'local_crop_size': [128, 128, 128],
			'min_valid_fraction': 0.1,
			'max_resample_attempts': 16,
		},
		'zero_mask': {
			'enabled': True,
			'zero_atol': 0.0,
			'z_sample_influence_radius': 16,
			'xy_trace_influence_radius': 1,
		},
		'model': {
			'patch_size': [8, 8, 8],
			'encoder_dim': 384,
			'encoder_depth': 8,
			'encoder_heads': 6,
			'decoder_dim': 256,
			'decoder_depth': 4,
			'decoder_heads': 4,
		},
		'augmentations': {'horizontal_flip_probability': 0.5},
		'vicreg': {
			'method': VICREG_METHOD,
			'local_pairs_per_crop': 128,
			'projector_dim': 384,
			'invariance_weight': 25.0,
			'variance_weight': 25.0,
			'covariance_weight': 1.0,
			'variance_target_std': 1.0,
			'variance_eps': 1.0e-4,
		},
		'train': {'epochs': epochs},
	}
	if continuation is not None:
		config['continuation'] = {
			'init_checkpoint': str(continuation),
			'unfreeze_top_blocks': 1,
		}
	return config


def _write_direct_checkpoint(
	path: Path,
	*,
	epochs: int,
	global_steps: int,
	continuation: Path | None = None,
) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	payload: dict[str, object] = {
		'config': _vicreg_config(epochs=epochs, continuation=continuation),
		'epoch': epochs,
		'global_step': global_steps,
		'pretraining_method': VICREG_METHOD,
		'checkpoint_kind': 'vicreg_pretraining',
		'model_state_dict': {'encoder.weight': torch.ones(1) * epochs},
		'projector_state_dict': {'weight': torch.ones(1)},
		'training_state': {
			'schema_version': 1,
			'stage': 'vicreg_training',
			'resume_boundary': 'epoch',
			'completed_epoch': True,
		},
	}
	if continuation is not None:
		payload['continuation_lineage'] = {
			'schema_version': 1,
			'init_checkpoint': str(continuation),
			'init_checkpoint_sha256': file_sha256(continuation),
			'resume_count': 0,
		}
	torch.save(payload, path)
	return path


def _write_hmm_checkpoint(path: Path, *, stage1: Path) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	pseudo_target_dir = (
		path.parents[2]
		/ 'pseudo_targets/f3/facies_benchmark_v1/local_vicreg_v1/vicreg100'
	)
	pseudo_target_paths = write_pseudo_target(
		pseudo_target_dir,
		k=6,
		survey_id=SURVEY_ID,
		labels=np.zeros((2, 2, 2), dtype=np.int32),
		confidence=np.ones((2, 2, 2), dtype=np.float32),
		valid_tokens=np.ones((2, 2, 2), dtype=np.bool_),
		boundary_weight=np.ones((2, 2, 2), dtype=np.float32),
		metadata={'fixture': 'vicreg_hmm_lineage'},
	)
	model_tag = 'f3_local_vicreg_hmm_k6_fixture'
	stratigraphy_config: dict[str, object] = {
		'stage': 'train_strat_hmm_pretext',
		'data': {'finite_check_mode': 'strict'},
		'teacher': {'checkpoint': str(stage1)},
		'student': {
			'init_checkpoint': str(stage1),
			'unfreeze_top_blocks': 1,
		},
		'identity': {'model_tag': model_tag},
		'head': {'num_prototypes': 6},
		'pseudo_targets': {
			'input_dir': str(pseudo_target_dir),
			'k': 6,
			'min_confidence': 0.0,
		},
		'loss': {'distillation_weight': 0.2},
		'train': {'epochs': 25},
	}
	control_identity = {
		'schema_version': 1,
		'model_tag': model_tag,
		'scientific_identity': {},
		'runtime_identity': {
			'git_commit': None,
			'git_status_short': '',
			'git_diff_sha256': hashlib.sha256(b'').hexdigest(),
			'finite_check_mode': 'strict',
		},
		'resolved_training_config_sha256': hashlib.sha256(
			json.dumps(
				stratigraphy_config, sort_keys=True, separators=(',', ':')
			).encode()
		).hexdigest(),
		'input_identities': {
			'teacher_checkpoint': _identity(stage1),
			'student_init_checkpoint': _identity(stage1),
			'pseudo_targets': [
				{
					'survey_id': SURVEY_ID,
					'labels': _identity(pseudo_target_paths.labels),
					'confidence': _identity(pseudo_target_paths.confidence),
					'valid_tokens': _identity(pseudo_target_paths.valid_tokens),
					'metadata': _identity(pseudo_target_paths.metadata),
					'boundary_weight_present': True,
					'boundary_weight': _identity(
						pseudo_target_paths.boundary_weight
					),
				}
			],
		},
		'initial_parameter_sha256': {
			'student_trainable': hashlib.sha256(b'student').hexdigest(),
			'prototype_head': hashlib.sha256(b'head').hexdigest(),
		},
		'initial_state_sha256': {
			'student': hashlib.sha256(b'student-state').hexdigest(),
			'head': hashlib.sha256(b'head-state').hexdigest(),
		},
	}
	torch.save(
		{
			'config': _vicreg_config(epochs=100),
			'stratigraphy_config': stratigraphy_config,
			'control_identity': control_identity,
			'epoch': 25,
			'global_step': 15_625,
			'model_state_dict': {'encoder.weight': torch.ones(1) * 6},
			'training_state': {
				'schema_version': 1,
				'stage': 'train_strat_hmm_pretext',
				'checkpoint_kind': 'epoch',
			},
		},
		path,
	)
	return path


def _write_candidate_embeddings(
	root: Path,
	*,
	checkpoint: Path,
	random_embeddings_dir: Path,
	hmm: bool,
) -> Path:
	root.mkdir(parents=True, exist_ok=True)
	random_embeddings = random_embeddings_dir / f'{SURVEY_ID}.embeddings.npy'
	random_tokens = random_embeddings_dir / f'{SURVEY_ID}.valid_tokens.npy'
	shutil.copyfile(random_embeddings, root / random_embeddings.name)
	shutil.copyfile(random_tokens, root / random_tokens.name)
	random_metadata = json.loads(
		(random_embeddings_dir / f'{SURVEY_ID}.embedding_metadata.json').read_text(
			encoding='utf-8'
		)
	)
	metadata = {
		**random_metadata,
		'checkpoint_path': str(checkpoint),
		'checkpoint_sha256': file_sha256(checkpoint),
		'pretraining_method': VICREG_METHOD,
		'pretraining_objective': {
			'method': VICREG_METHOD,
			'local_pairs_per_crop': 128,
			'projector_dim': 384,
			'invariance_weight': 25.0,
			'variance_weight': 25.0,
			'covariance_weight': 1.0,
			'variance_target_std': 1.0,
			'variance_eps': 1.0e-4,
		},
	}
	if hmm:
		payload = load_checkpoint(checkpoint, map_location='cpu')
		pseudo_target_dir = payload['stratigraphy_config']['pseudo_targets'][
			'input_dir'
		]
		metadata['stratigraphy_pretext'] = {
			'method': 'strat_hmm_pretext',
			'base_objective': VICREG_METHOD,
			'head_num_prototypes': 6,
			'unfreeze_top_blocks': 1,
			'distillation_weight': 0.2,
			'pseudo_target_input_dir': pseudo_target_dir,
		}
	(root / f'{SURVEY_ID}.embedding_metadata.json').write_text(
		json.dumps(metadata, indent=1, sort_keys=True), encoding='utf-8'
	)
	return root


def _model_source(universe: dict[str, object], model_id: str) -> dict[str, object]:
	return next(
		model for model in universe['models'] if model['model_id'] == model_id
	)


def _metric_value(
	model_id: str,
	layout_id: str,
	data_size: str,
	metric: str,
) -> float:
	model_order = (*FIVE_WAY_MODEL_IDS, 'local_vicreg_100', *EXTENSION_MODEL_IDS)
	return round(
		0.2
		+ 0.035 * model_order.index(model_id)
		+ 0.004 * LAYOUT_IDS.index(layout_id)
		+ 0.08 * DATA_SIZES.index(data_size)
		+ 0.0005 * SUMMARY_METRICS.index(metric),
		6,
	)


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _decoder_initial_state(layout_id: str, data_size: str) -> str:
	return hashlib.sha256(f'{layout_id}/{data_size}/decoder'.encode()).hexdigest()


def _write_job(
	canonical_mapping: dict[str, object],
	*,
	model: dict[str, object],
	runs_root: Path,
	layout_id: str,
	data_size: str,
) -> Path:
	model_id = str(model['model_id'])
	job_dir = (
		runs_root
		/ f'model={model_id}'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
	)
	decoder = job_dir / 'decoder'
	prediction = job_dir / 'prediction'
	evaluation = job_dir / 'evaluation'
	for directory in (decoder, prediction, evaluation):
		directory.mkdir(parents=True, exist_ok=True)
	(decoder / 'best.pt').write_text(
		f'{model_id}/{layout_id}/{data_size}', encoding='utf-8'
	)
	metrics = {
		metric: _metric_value(model_id, layout_id, data_size, metric)
		for metric in SUMMARY_METRICS
	}
	metrics.update(
		{
			'accuracy': 0.9,
			'evaluation_voxel_count': VALIDATION_VOXEL_COUNT,
			'aggregation_unit': EXPECTED_AGGREGATION_UNIT,
		}
	)
	(evaluation / 'metrics.json').write_text(
		json.dumps(metrics), encoding='utf-8'
	)
	embeddings_dir = Path(str(model['embeddings_dir']))
	prediction_metadata_path = prediction / 'prediction_metadata.json'
	prediction_metadata_path.write_text(
		json.dumps(
			{
				'artifact_type': 'f3_lithology_voxel_prediction',
				'model_tag': model_id,
				'source_identity': {
					'decoder_checkpoint': _identity(decoder / 'best.pt'),
					'artifact_identities': {
						'name': 'f3_voxel_decoder_sources',
						'embeddings': _identity(
							embeddings_dir / f'{SURVEY_ID}.embeddings.npy'
						),
						'embedding_metadata': _identity(
							embeddings_dir
							/ f'{SURVEY_ID}.embedding_metadata.json'
						),
						'valid_tokens': _identity(
							embeddings_dir / f'{SURVEY_ID}.valid_tokens.npy'
						),
					},
				},
			}
		),
		encoding='utf-8',
	)
	condition_dir = (
		Path(canonical_mapping['section_layout']['dataset_root'])
		/ 'datasets'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
		/ 'voxel_supervision'
	)
	policy = {
		key: list(value) if isinstance(value, tuple) else value
		for key, value in FIVE_WAY_EVALUATION_POLICY.items()
	}
	(evaluation / 'evaluation_metadata.json').write_text(
		json.dumps(
			{
				'dataset': dict(canonical_mapping['dataset']),
				'model_tag': model_id,
				'policy': policy,
				'inputs': {
					'prediction_metadata': _identity(prediction_metadata_path),
					'voxel_dataset_metadata': _identity(
						condition_dir / 'voxel_dataset_metadata.json'
					),
					'voxel_split_grid': _identity(
						condition_dir / 'supervision_split_grid.npy'
					),
				},
			}
		),
		encoding='utf-8',
	)
	(decoder / 'resolved_config.json').write_text(
		json.dumps(
			{
				'model': {'tag': model_id, 'freeze_encoder': True},
				'embeddings': {
					'checkpoint_path': model['checkpoint'],
					'input_dir': model['embeddings_dir'],
					'spec': 'overlap_x64',
				},
				'decoder': {
					key: list(value) if isinstance(value, tuple) else value
					for key, value in FIXED_DECODER_CONTRACT.items()
					if key
					in {
						'spec',
						'embedding_dim',
						'class_count',
						'hidden_channels',
						'upsample_factors',
						'upsample_mode',
						'normalization',
					}
				},
				'tiles': {
					key: list(value) for key, value in FIVE_WAY_TILE_SETTINGS.items()
				},
				'train': {
					key: value
					for key, value in FIXED_DECODER_CONTRACT.items()
					if key
					in {
						'epochs',
						'batch_size',
						'learning_rate',
						'weight_decay',
						'class_weight',
						'seed',
						'amp',
						'gradient_clip_norm',
						'sampling_mode',
						'steps_per_epoch',
					}
				},
			}
		),
		encoding='utf-8',
	)
	(decoder / 'run_metadata.json').write_text(
		json.dumps(
			{
				'voxel_dataset_metadata': str(
					condition_dir / 'voxel_dataset_metadata.json'
				),
				'train_tile_manifest_sha256': (
					f'{layout_id}-{data_size}-train-manifest'
				),
				'validation_tile_manifest_sha256': (
					f'{layout_id}-{data_size}-validation-manifest'
				),
				'initial_model_state_sha256': _decoder_initial_state(
					layout_id, data_size
				),
			}
		),
		encoding='utf-8',
	)
	return job_dir


@pytest.fixture
def vicreg_universe(tmp_path: Path) -> dict[str, object]:
	root = tmp_path / 'synthetic'
	canonical_mapping = build_five_way_universe(root / 'canonical')
	for layout_id in LAYOUT_IDS:
		for data_size in DATA_SIZES:
			write_condition(canonical_mapping, layout_id, data_size)
	canonical_config_path = root / 'canonical_five_way.yaml'
	canonical_config_path.parent.mkdir(parents=True, exist_ok=True)
	canonical_config_path.write_text(
		yaml.safe_dump(canonical_mapping), encoding='utf-8'
	)
	random_embeddings_dir = Path(
		_model_source(canonical_mapping, 'random')['embeddings_dir']
	)
	stage1 = _write_direct_checkpoint(
		root / 'vicreg/stage1/latest.pt', epochs=100, global_steps=62_500
	)
	control = _write_direct_checkpoint(
		root / 'vicreg/control/latest.pt',
		epochs=25,
		global_steps=15_625,
		continuation=stage1,
	)
	hmm = _write_hmm_checkpoint(root / 'vicreg/hmm/latest.pt', stage1=stage1)
	screen_embeddings = _write_candidate_embeddings(
		root / 'embeddings/screen',
		checkpoint=stage1,
		random_embeddings_dir=random_embeddings_dir,
		hmm=False,
	)
	control_embeddings = _write_candidate_embeddings(
		root / 'embeddings/control',
		checkpoint=control,
		random_embeddings_dir=random_embeddings_dir,
		hmm=False,
	)
	hmm_embeddings = _write_candidate_embeddings(
		root / 'embeddings/hmm',
		checkpoint=hmm,
		random_embeddings_dir=random_embeddings_dir,
		hmm=True,
	)
	mapping: dict[str, object] = {
		'benchmark': {'canonical_config': str(canonical_config_path)},
		'screening': {
			'model': {
				'model_id': 'local_vicreg_100',
				'checkpoint': str(stage1),
				'embeddings_dir': str(screen_embeddings),
			},
			'outputs': {
				'runs_root': str(root / 'outputs/screen/runs'),
				'job_logs_root': str(root / 'outputs/screen/logs'),
				'summary_root': str(root / 'outputs/screen/summary'),
			},
		},
		'extension': {
			'models': [
				{
					'model_id': 'local_vicreg',
					'checkpoint': str(control),
					'embeddings_dir': str(control_embeddings),
				},
				{
					'model_id': 'local_vicreg_hmm_k6',
					'checkpoint': str(hmm),
					'embeddings_dir': str(hmm_embeddings),
				},
			],
			'outputs': {
				'runs_root': str(root / 'outputs/extension/runs'),
				'job_logs_root': str(root / 'outputs/extension/logs'),
				'summary_root': str(root / 'outputs/extension/summary'),
				'combined_summary_root': str(root / 'outputs/combined_summary'),
			},
		},
	}
	config = f3_vicreg_extension_config_from_mapping(mapping)
	canonical = load_f3_vicreg_canonical_config(config)
	return {
		'root': root,
		'canonical_mapping': canonical_mapping,
		'mapping': mapping,
		'config': config,
		'canonical': canonical,
		'stage1': stage1,
		'control': control,
		'hmm': hmm,
		'pseudo_target_dir': Path(
			load_checkpoint(hmm, map_location='cpu')['stratigraphy_config'][
				'pseudo_targets'
			]['input_dir']
		),
	}


def _write_canonical_jobs(universe: dict[str, object]) -> None:
	mapping = universe['canonical_mapping']
	for model_id in FIVE_WAY_MODEL_IDS:
		model = _model_source(mapping, model_id)
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZES:
				_write_job(
					mapping,
					model=model,
					runs_root=Path(mapping['outputs']['runs_root']),
					layout_id=layout_id,
					data_size=data_size,
				)


def _write_screening_jobs(universe: dict[str, object]) -> None:
	config = universe['config']
	model = {
		'model_id': config.screening_model.model_id,
		'checkpoint': str(config.screening_model.checkpoint),
		'embeddings_dir': str(config.screening_model.embeddings_dir),
	}
	for layout_id in LAYOUT_IDS:
		_write_job(
			universe['canonical_mapping'],
			model=model,
			runs_root=config.screening_outputs.runs_root,
			layout_id=layout_id,
			data_size='medium',
		)


def _write_extension_jobs(universe: dict[str, object]) -> None:
	config = universe['config']
	for source in config.extension_models:
		model = {
			'model_id': source.model_id,
			'checkpoint': str(source.checkpoint),
			'embeddings_dir': str(source.embeddings_dir),
		}
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZES:
				_write_job(
					universe['canonical_mapping'],
					model=model,
					runs_root=config.extension_outputs.runs_root,
					layout_id=layout_id,
					data_size=data_size,
				)


def _files_snapshot(root: Path) -> dict[str, str]:
	return {
		str(path): file_sha256(path)
		for path in sorted(root.rglob('*'))
		if path.is_file()
	}


def test_exact_job_plans_and_medium_only_contract(
	vicreg_universe: dict[str, object],
) -> None:
	screening = plan_f3_vicreg_screening_jobs()
	extension = plan_f3_vicreg_extension_jobs()

	assert len(screening) == 10
	assert screening[:4] == (
		('local_vicreg_100', 'layout_000', 'medium'),
		('random', 'layout_000', 'medium'),
		('local_vicreg_100', 'layout_001', 'medium'),
		('random', 'layout_001', 'medium'),
	)
	assert len(extension) == 30
	assert extension[:4] == (
		('local_vicreg', 'layout_000', 'small'),
		('local_vicreg_hmm_k6', 'layout_000', 'small'),
		('local_vicreg', 'layout_001', 'small'),
		('local_vicreg_hmm_k6', 'layout_001', 'small'),
	)
	with pytest.raises(ValueError, match='only accepts size'):
		resolve_f3_vicreg_screening_job(
			vicreg_universe['config'],
			vicreg_universe['canonical'],
			model='local_vicreg_100',
			layout='layout_000',
			size='small',
		)

	overlap = deepcopy(vicreg_universe['mapping'])
	overlap['extension']['outputs']['summary_root'] = overlap['screening']['outputs'][
		'summary_root'
	]
	with pytest.raises(ValueError, match='must be disjoint'):
		f3_vicreg_extension_config_from_mapping(overlap)


def test_source_audits_require_common_vicreg100_lineage(
	vicreg_universe: dict[str, object],
) -> None:
	config = vicreg_universe['config']
	canonical = vicreg_universe['canonical']

	screen = audit_f3_vicreg_screening_source(config, canonical)
	full = audit_f3_vicreg_sources(config, canonical)

	assert screen['model_order'] == ['local_vicreg_100']
	assert full['model_order'] == [
		'local_vicreg_100',
		'local_vicreg',
		'local_vicreg_hmm_k6',
	]
	assert {
		source['valid_tokens_sha256'] for source in full['sources']
	} == {full['canonical_random']['valid_tokens_sha256']}

	alternate = _write_direct_checkpoint(
		vicreg_universe['root'] / 'vicreg/alternate/latest.pt',
		epochs=100,
		global_steps=62_500,
	)
	_write_hmm_checkpoint(vicreg_universe['hmm'], stage1=alternate)
	hmm_metadata_path = (
		config.extension_model_by_id('local_vicreg_hmm_k6').embeddings_dir
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	hmm_metadata = json.loads(hmm_metadata_path.read_text(encoding='utf-8'))
	hmm_metadata['checkpoint_sha256'] = file_sha256(vicreg_universe['hmm'])
	hmm_metadata_path.write_text(json.dumps(hmm_metadata), encoding='utf-8')
	with pytest.raises(ValueError, match='screened VICReg100 checkpoint path'):
		audit_f3_vicreg_sources(config, canonical)


def test_source_audit_rejects_same_path_stage1_replacement(
	vicreg_universe: dict[str, object],
) -> None:
	config = vicreg_universe['config']
	stage1 = vicreg_universe['stage1']
	stage1_payload = load_checkpoint(stage1, map_location='cpu')
	stage1_payload['model_state_dict']['encoder.weight'] = torch.full((1,), 999.0)
	torch.save(stage1_payload, stage1)
	stage1_sha256 = file_sha256(stage1)

	screen_metadata_path = (
		config.screening_model.embeddings_dir
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	screen_metadata = json.loads(
		screen_metadata_path.read_text(encoding='utf-8')
	)
	screen_metadata['checkpoint_sha256'] = stage1_sha256
	screen_metadata_path.write_text(json.dumps(screen_metadata), encoding='utf-8')

	control = vicreg_universe['control']
	control_payload = load_checkpoint(control, map_location='cpu')
	control_payload['continuation_lineage']['init_checkpoint_sha256'] = stage1_sha256
	torch.save(control_payload, control)
	control_metadata_path = (
		config.extension_model_by_id('local_vicreg').embeddings_dir
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	control_metadata = json.loads(
		control_metadata_path.read_text(encoding='utf-8')
	)
	control_metadata['checkpoint_sha256'] = file_sha256(control)
	control_metadata_path.write_text(json.dumps(control_metadata), encoding='utf-8')

	with pytest.raises(
		ValueError, match='control_identity teacher checkpoint SHA'
	):
		audit_f3_vicreg_sources(config, vicreg_universe['canonical'])


def test_source_audit_rejects_same_path_pseudo_target_replacement(
	vicreg_universe: dict[str, object],
) -> None:
	inputs = discover_pseudo_target_inputs(
		vicreg_universe['pseudo_target_dir'], k=6
	)
	with inputs[0].labels_path.open('ab') as handle:
		handle.write(b'same-path-replacement')

	with pytest.raises(ValueError, match=r'pseudo-target .* labels SHA'):
		audit_f3_vicreg_sources(
			vicreg_universe['config'], vicreg_universe['canonical']
		)


@pytest.mark.parametrize(
	('section', 'key', 'value', 'message'),
	[
		('model', 'encoder_dim', 999, 'model.encoder_dim'),
		('vicreg', 'covariance_weight', 99.0, 'vicreg.covariance_weight'),
		('data', 'min_valid_fraction', 0.99, 'data identity'),
		('zero_mask', 'enabled', False, 'zero_mask identity'),
		(
			'augmentations',
			'policy',
			'horizontal_flip_trace_drop',
			'augmentations',
		),
	],
)
def test_source_audit_rejects_control_geometry_loss_and_view_drift(
	vicreg_universe: dict[str, object],
	section: str,
	key: str,
	value: object,
	message: str,
) -> None:
	config = vicreg_universe['config']
	control = vicreg_universe['control']
	payload = load_checkpoint(control, map_location='cpu')
	payload['config'][section][key] = value
	torch.save(payload, control)
	metadata_path = (
		config.extension_model_by_id('local_vicreg').embeddings_dir
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['checkpoint_sha256'] = file_sha256(control)
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

	with pytest.raises(ValueError, match=message):
		audit_f3_vicreg_sources(config, vicreg_universe['canonical'])


@pytest.mark.parametrize(
	('field', 'value'),
	[
		('method', 'wrong_pretext'),
		('unfreeze_top_blocks', 2),
		('distillation_weight', 0.9),
		('pseudo_target_input_dir', '/wrong/pseudo_targets'),
	],
)
def test_source_audit_rejects_embedding_method_and_hmm_detail_drift(
	vicreg_universe: dict[str, object], field: str, value: object
) -> None:
	config = vicreg_universe['config']
	hmm_metadata_path = (
		config.extension_model_by_id('local_vicreg_hmm_k6').embeddings_dir
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	metadata = json.loads(hmm_metadata_path.read_text(encoding='utf-8'))
	metadata['stratigraphy_pretext'][field] = value
	hmm_metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
	with pytest.raises(ValueError, match=f'HMM {field}'):
		audit_f3_vicreg_sources(config, vicreg_universe['canonical'])


def test_source_audit_rejects_top_level_embedding_method_drift(
	vicreg_universe: dict[str, object],
) -> None:
	config = vicreg_universe['config']
	metadata_path = (
		config.extension_model_by_id('local_vicreg').embeddings_dir
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['pretraining_method'] = 'local_barlow_twins_3d'
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
	with pytest.raises(ValueError, match='pretraining_method'):
		audit_f3_vicreg_sources(config, vicreg_universe['canonical'])


def test_source_audit_rejects_hmm_target_suffix_drift(
	vicreg_universe: dict[str, object],
) -> None:
	config = vicreg_universe['config']
	hmm = vicreg_universe['hmm']
	payload = load_checkpoint(hmm, map_location='cpu')
	target_dir = str(vicreg_universe['root'] / 'wrong/pseudo_targets')
	payload['stratigraphy_config']['pseudo_targets']['input_dir'] = target_dir
	torch.save(payload, hmm)
	metadata_path = (
		config.extension_model_by_id('local_vicreg_hmm_k6').embeddings_dir
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['checkpoint_sha256'] = file_sha256(hmm)
	metadata['stratigraphy_pretext']['pseudo_target_input_dir'] = target_dir
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
	with pytest.raises(ValueError, match='pseudo-target path must end with'):
		audit_f3_vicreg_sources(config, vicreg_universe['canonical'])


@pytest.mark.parametrize(
	('section', 'key', 'value'),
	[
		('train', 'seed', 42),
		('decoder', 'hidden_channels', [64, 32]),
	],
)
def test_screening_rejects_decoder_seed_and_hyperparameter_drift(
	vicreg_universe: dict[str, object],
	section: str,
	key: str,
	value: object,
) -> None:
	_write_canonical_jobs(vicreg_universe)
	_write_screening_jobs(vicreg_universe)
	config = vicreg_universe['config']
	resolved_path = (
		config.screening_outputs.runs_root
		/ 'model=local_vicreg_100/layout=layout_000/size=medium'
		/ 'decoder/resolved_config.json'
	)
	resolved = json.loads(resolved_path.read_text(encoding='utf-8'))
	resolved[section][key] = value
	resolved_path.write_text(json.dumps(resolved), encoding='utf-8')
	with pytest.raises(ValueError, match=f'decoder resolved {section}.{key}'):
		summarize_f3_vicreg_screening(config, vicreg_universe['canonical'])


def test_screening_summary_gate_sign_atomicity_and_evidence_binding(
	vicreg_universe: dict[str, object],
) -> None:
	_write_canonical_jobs(vicreg_universe)
	_write_screening_jobs(vicreg_universe)
	config = vicreg_universe['config']
	canonical = vicreg_universe['canonical']

	result = summarize_f3_vicreg_screening(config, canonical)

	assert result['complete_jobs'] == 10
	assert result['gate_status'] == VICREG_GATE_PASS
	screening_outputs = sorted(
		path.name for path in config.screening_outputs.summary_root.iterdir()
	)
	assert screening_outputs == sorted(SCREENING_SUMMARY_OUTPUT_NAMES)
	with (config.screening_outputs.summary_root / 'paired_deltas.csv').open(
		encoding='utf-8', newline=''
	) as handle:
		paired = list(csv.DictReader(handle))
	assert len(paired) == 5
	for row in paired:
		assert row['left_model'] == 'local_vicreg_100'
		assert row['right_model'] == 'random'
		assert float(row['delta']) == pytest.approx(
			_metric_value('local_vicreg_100', row['layout_id'], 'medium', 'macro_f1')
			- _metric_value('random', row['layout_id'], 'medium', 'macro_f1')
		)
	payload = json.loads(
		(config.screening_outputs.summary_root / 'summary.json').read_text(
			encoding='utf-8'
		)
	)
	assert len(payload['layouts']) == 5
	for layout in payload['layouts']:
		assert {
			'local_vicreg_macro_f1',
			'random_macro_f1',
			'delta_local_vicreg_minus_random',
			'local_vicreg_checkpoint_sha256',
			'random_checkpoint_sha256',
			'local_vicreg_embedding_sha256',
			'random_embedding_sha256',
			'supervision_identity',
			'validation_mask_sha256',
			'decoder_initial_state_sha256',
		} <= set(layout)
		assert layout['delta_local_vicreg_minus_random'] == pytest.approx(
			layout['local_vicreg_macro_f1'] - layout['random_macro_f1']
		)
	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		summarize_f3_vicreg_screening(config, canonical)

	assert_f3_vicreg_full_benchmark_ready(config, canonical)
	metrics_path = (
		config.screening_outputs.runs_root
		/ 'model=local_vicreg_100/layout=layout_000/size=medium/evaluation/metrics.json'
	)
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	metrics['macro_f1'] += 0.01
	metrics_path.write_text(json.dumps(metrics), encoding='utf-8')
	with pytest.raises(RuntimeError, match='FULL_BENCHMARK_BLOCKED'):
		assert_f3_vicreg_full_benchmark_ready(config, canonical)


def test_screening_gate_fail_uses_mean_median_and_wins(
	vicreg_universe: dict[str, object],
) -> None:
	_write_canonical_jobs(vicreg_universe)
	_write_screening_jobs(vicreg_universe)
	config = vicreg_universe['config']
	for layout_id in LAYOUT_IDS[:3]:
		candidate = (
			config.screening_outputs.runs_root
			/ f'model=local_vicreg_100/layout={layout_id}'
			/ 'size=medium/evaluation/metrics.json'
		)
		random = (
			vicreg_universe['canonical'].runs_root
			/ f'model=random/layout={layout_id}'
			/ 'size=medium/evaluation/metrics.json'
		)
		candidate_metrics = json.loads(candidate.read_text(encoding='utf-8'))
		random_metrics = json.loads(random.read_text(encoding='utf-8'))
		candidate_metrics['macro_f1'] = random_metrics['macro_f1'] - 0.02
		candidate.write_text(json.dumps(candidate_metrics), encoding='utf-8')

	result = summarize_f3_vicreg_screening(
		config, vicreg_universe['canonical']
	)
	payload = json.loads(
		(config.screening_outputs.summary_root / 'summary.json').read_text(
			encoding='utf-8'
		)
	)
	assert result['gate_status'] == VICREG_GATE_FAIL
	assert payload['wins'] == 2
	assert payload['gate_status'] == VICREG_GATE_FAIL


def test_extension_and_combined_reports_are_exact_and_read_only(
	vicreg_universe: dict[str, object],
) -> None:
	_write_canonical_jobs(vicreg_universe)
	_write_screening_jobs(vicreg_universe)
	_write_extension_jobs(vicreg_universe)
	config = vicreg_universe['config']
	canonical = vicreg_universe['canonical']
	summarize_f3_vicreg_screening(config, canonical)
	canonical_before = _files_snapshot(canonical.runs_root)

	extension_report = inspect_f3_vicreg_extension_results(config, canonical)
	extension_result = summarize_f3_vicreg_extension(config, canonical)
	combined_report = inspect_f3_vicreg_combined_results(config, canonical)
	combined_result = summarize_f3_vicreg_combined(config, canonical)

	assert extension_report['complete_jobs'] == 30
	assert extension_result['complete_jobs'] == 30
	extension_outputs = sorted(
		path.name for path in config.extension_outputs.summary_root.iterdir()
	)
	assert extension_outputs == sorted(BENCHMARK_SUMMARY_OUTPUT_NAMES)
	with (config.extension_outputs.summary_root / 'summary_by_size.csv').open(
		encoding='utf-8', newline=''
	) as handle:
		by_size = list(csv.DictReader(handle))
	assert {row['n_layouts'] for row in by_size} == {'5'}
	assert {row['data_size'] for row in by_size} == set(DATA_SIZES)
	with (config.extension_outputs.summary_root / 'paired_deltas.csv').open(
		encoding='utf-8', newline=''
	) as handle:
		paired = list(csv.DictReader(handle))
	hmm_control = next(
		row
		for row in paired
		if row['comparison_id'] == 'local_vicreg_hmm_k6_minus_local_vicreg'
		and row['metric'] == 'macro_f1'
	)
	assert float(hmm_control['delta']) == pytest.approx(
		float(hmm_control['left_value']) - float(hmm_control['right_value'])
	)
	assert combined_report == {
		'complete_jobs': 105,
		'existing_jobs': 75,
		'extension_jobs': 30,
		'model_order': list(SEVEN_WAY_MODEL_IDS),
		'paired_delta_rows': combined_report['paired_delta_rows'],
	}
	assert combined_result['complete_jobs'] == 105
	with (config.combined_summary_root / 'comparison.csv').open(
		encoding='utf-8', newline=''
	) as handle:
		comparison = list(csv.DictReader(handle))
	assert len(comparison) == 105
	assert {row['model_id'] for row in comparison} == set(SEVEN_WAY_MODEL_IDS)
	assert _files_snapshot(canonical.runs_root) == canonical_before
	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		summarize_f3_vicreg_combined(config, canonical)


def test_missing_extra_and_source_drift_are_rejected(
	vicreg_universe: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_write_canonical_jobs(vicreg_universe)
	_write_screening_jobs(vicreg_universe)
	_write_extension_jobs(vicreg_universe)
	config = vicreg_universe['config']
	canonical = vicreg_universe['canonical']
	summarize_f3_vicreg_screening(config, canonical)
	original_plan = plan_f3_vicreg_extension_jobs()
	duplicate_plan = (*original_plan[:-1], original_plan[0])
	with monkeypatch.context() as patcher:
		patcher.setattr(
			vicreg_benchmark,
			'plan_f3_vicreg_extension_jobs',
			lambda: duplicate_plan,
		)
		with pytest.raises(ValueError, match='duplicate VICReg benchmark job evidence'):
			inspect_f3_vicreg_extension_results(config, canonical)
	missing = (
		config.extension_outputs.runs_root
		/ 'model=local_vicreg/layout=layout_004/size=large/evaluation/metrics.json'
	)
	missing.unlink()
	with pytest.raises(FileNotFoundError):
		inspect_f3_vicreg_extension_results(config, canonical)
	_write_extension_jobs(vicreg_universe)
	extra = config.extension_outputs.runs_root / 'model=foreign'
	extra.mkdir(parents=True)
	with pytest.raises(ValueError, match='unexpected VICReg run directory'):
		inspect_f3_vicreg_extension_results(config, canonical)
	extra.rmdir()

	mae_embeddings = (
		canonical.model_by_id('mae').embeddings_dir
		/ f'{SURVEY_ID}.embeddings.npy'
	)
	original_embeddings = mae_embeddings.read_bytes()
	with mae_embeddings.open('ab') as handle:
		handle.write(b'drift')
	with pytest.raises(
		ValueError, match='does not match the current configured source'
	):
		inspect_f3_vicreg_combined_results(config, canonical)
	mae_embeddings.write_bytes(original_embeddings)

	checkpoint = config.extension_model_by_id('local_vicreg').checkpoint
	with checkpoint.open('ab') as handle:
		handle.write(b'drift')
	with pytest.raises(RuntimeError, match='FULL_BENCHMARK_BLOCKED'):
		inspect_f3_vicreg_extension_results(config, canonical)


def test_116_configs_runbook_and_cli_references_are_resolvable(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	workspace = Path(__file__).resolve().parents[2]
	experiment = (
		workspace
		/ 'experiments/f3/facies_benchmark_v2/116_local_vicreg_extension_v1'
	)
	artifact_root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(workspace))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(artifact_root))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3'))
	mapping = load_config(experiment / '60_extension.yaml')
	config = f3_vicreg_extension_config_from_mapping(mapping)
	assert config.canonical_config == (
		workspace
		/ 'experiments/f3/facies_benchmark_v2'
		/ '110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml'
	)
	assert config.screening_model.model_id == SCREENING_MODEL_IDS[0]
	assert tuple(model.model_id for model in config.extension_models) == (
		EXTENSION_MODEL_IDS
	)
	for name, model_id in (
		('01_extract_local_vicreg.yaml', 'local_vicreg'),
		('02_extract_local_vicreg_hmm_k6.yaml', 'local_vicreg_hmm_k6'),
	):
		extraction = yaml.safe_load(
			(experiment / '50_embeddings' / name).read_text(encoding='utf-8')
		)
		assert extraction['embedding']['window_size'] == [128, 128, 128]
		assert extraction['embedding']['overlap'] == [64, 64, 64]
		assert extraction['embedding']['output_dtype'] == 'float16'
		assert f'/{model_id}/overlap_x64' in extraction['embeddings']['output_dir']

	readme = (experiment / 'README.md').read_text(encoding='utf-8')
	for relative in set(re.findall(r'proc/seis_ssl_cluster/[a-z0-9_]+\.py', readme)):
		assert (workspace / relative).is_file(), relative
	screen_log_suffix = config.screening_outputs.job_logs_root.relative_to(
		artifact_root
	)
	extension_log_suffix = config.extension_outputs.job_logs_root.relative_to(
		artifact_root
	)
	assert str(screen_log_suffix) in readme
	assert str(extension_log_suffix) in readme
	assert 'tee "$SCREEN_LOG_ROOT/${layout}_${model}_medium.log"' in readme
	assert 'tee "$EXTENSION_LOG_ROOT/${layout}_${model}_${size}.log"' in readme
	assert '--resume "$SCREEN_RESUME"' in readme
	assert '--resume "$EXTENSION_RESUME"' in readme
	assert readme.count('/decoder/latest.pt') >= 2
	assert 'never use a checkpoint from a\ndifferent cell' in readme
	for shell_block in re.findall(r'```bash\n(.*?)```', readme, flags=re.DOTALL):
		subprocess.run(
			['/bin/bash', '-n'],
			input=shell_block,
			text=True,
			check=True,
			capture_output=True,
		)
	assert readme.index('for size in small medium large') < readme.index(
		'for layout in layout_000', readme.index('for size in small medium large')
	)
	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			'proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py',
			'--config',
			str(experiment / '60_extension.yaml'),
			'--mode',
			'screening-source',
			'--dry-run',
		],
		cwd=workspace,
		check=True,
		capture_output=True,
		text=True,
	)
	assert 'screening_jobs: 10' in result.stdout
	assert 'live source audit skipped' in result.stdout
