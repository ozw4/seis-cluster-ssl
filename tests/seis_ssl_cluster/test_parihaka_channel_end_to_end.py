# ruff: noqa: TC003

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

import proc.seis_ssl_cluster.run_parihaka_channel_end_to_end as end_to_end_cli
import seis_ssl_cluster.parihaka.channel_end_to_end as end_to_end
from seis_ssl_cluster.data.normalization import (
	SurveyNormalizationStats,
	write_normalization_stats,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.models.voxel_decoder import VoxelDecoder3D
from seis_ssl_cluster.parihaka.channel_checkpoints import (
	CHANNEL_PRETRAINED_MODEL_TAG,
	inspect_channel_model_sources,
)
from seis_ssl_cluster.parihaka.channel_data import (
	DATA_SIZE_PREFIX,
	LAYOUT_IDS,
	SectionLines,
	selected_training_lines,
)
from seis_ssl_cluster.parihaka.channel_decoder import (
	ChannelTileDataset,
	DecoderArchitecture,
	DecoderTiles,
	EmbeddingGeometry,
)
from seis_ssl_cluster.parihaka.channel_end_to_end import (
	BEST_NAME,
	HISTORY_NAME,
	LATEST_NAME,
	METRICS_NAME,
	ChannelAmplitudeTileDataset,
	ChannelEndToEndConfig,
	ChannelEndToEndModel,
	ChannelEndToEndTrain,
	channel_end_to_end_config_from_mapping,
	channel_end_to_end_optimizer_groups,
	inspect_channel_end_to_end_job,
	resolve_channel_end_to_end_runtime,
	resolve_channel_reference_artifact,
	run_channel_end_to_end_job,
	train_channel_end_to_end_step,
)


def _raw_fixture(tmp_path: Path) -> tuple[Path, Path]:
	amplitude_path = tmp_path / 'amplitude.npy'
	labels_path = tmp_path / 'labels.npy'
	stats_path = tmp_path / 'stats.json'
	np.save(amplitude_path, np.ones((64, 64, 64), dtype=np.float32))
	labels = np.ones((64, 64, 64), dtype=np.int8)
	labels[..., ::2] = 5
	np.save(labels_path, labels)
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id='parihaka',
			source_path=amplitude_path,
			grid_order=('x', 'y', 'z'),
			clip_low_percentile=1.0,
			clip_high_percentile=99.0,
			clip_low=-2.0,
			clip_high=2.0,
			median=0.0,
			iqr=1.0,
		),
		stats_path,
	)
	artifact_dir = tmp_path / 'reference'
	artifact_dir.mkdir()
	paths = output_paths(artifact_dir, 'parihaka')
	np.save(
		paths.valid_tokens,
		np.ones((8, 8, 8), dtype=np.bool_),
	)
	paths.metadata.write_text(
		json.dumps(
			{
				'source_amplitude_path': str(amplitude_path),
				'normalization_stats_path': str(stats_path),
				'volume_shape_xyz': [64, 64, 64],
				'patch_size': [8, 8, 8],
				'token_grid_shape': [8, 8, 8],
				'min_token_valid_fraction': 1.0,
				'preprocessing': {
					'normalized_clip_abs': 8.0,
					'amplitude_agc': {'enabled': False},
					'finite_check_mode': 'strict',
				},
				'zero_mask': {'enabled': False},
			}
		),
		encoding='utf-8',
	)
	return artifact_dir, labels_path


def test_reference_artifact_uses_embedding_output_path_contract(
	tmp_path: Path,
) -> None:
	artifact_dir, _ = _raw_fixture(tmp_path)
	paths = output_paths(artifact_dir, 'parihaka')

	artifact = resolve_channel_reference_artifact(artifact_dir)

	assert artifact.metadata_path == paths.metadata
	assert artifact.valid_tokens_path == paths.valid_tokens


def _dataset(
	tmp_path: Path,
	*,
	context_halo_tokens: tuple[int, int, int] = (1, 1, 1),
) -> ChannelAmplitudeTileDataset:
	artifact_dir, labels_path = _raw_fixture(tmp_path)
	return ChannelAmplitudeTileDataset(
		reference=resolve_channel_reference_artifact(artifact_dir),
		labels_path=labels_path,
		lines=SectionLines((0,), (0,)),
		validation=SectionLines((62,), (62,)),
		reserved_training=SectionLines((0,), (0,)),
		split='train',
		context_halo_tokens=context_halo_tokens,
		training_selection_mask=np.ones((8, 8, 8), dtype=np.bool_),
	)


@pytest.mark.parametrize(
	('context_halo_tokens', 'input_size', 'core_start'),
	[
		((1, 1, 1), 80, 8),
		((4, 4, 4), 128, 32),
	],
)
def test_raw_amplitude_uses_shared_preprocessing_and_matches_reference(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	context_halo_tokens: tuple[int, int, int],
	input_size: int,
	core_start: int,
) -> None:
	dataset = _dataset(tmp_path, context_halo_tokens=context_halo_tokens)
	called = 0
	runtime_token_valid_mask: np.ndarray | None = None
	original = end_to_end.read_amplitude_crop

	def wrapped(**kwargs: object) -> object:
		nonlocal called, runtime_token_valid_mask
		called += 1
		prepared = original(**kwargs)  # type: ignore[arg-type]
		runtime_token_valid_mask = prepared.token_valid_mask.copy()
		return prepared

	monkeypatch.setattr(end_to_end, 'read_amplitude_crop', wrapped)
	item = dataset[0]
	assert called == 1
	assert item['amplitude'].dtype == torch.float32
	assert item['amplitude'].shape == (1, input_size, input_size, input_size)
	token_size = input_size // 8
	assert item['token_valid_mask'].shape == (token_size, token_size, token_size)
	assert item['labels'].shape == (input_size, input_size, input_size)
	assert item['core_mask'].shape == (input_size, input_size, input_size)
	expected_core = torch.zeros_like(item['core_mask'])
	core_stop = core_start + 64
	expected_core[
		core_start:core_stop, core_start:core_stop, core_start:core_stop
	] = True
	assert torch.equal(item['core_mask'], expected_core)
	assert not item['supervision_mask'][~expected_core].any()
	assert int(item['supervision_mask'].sum()) == dataset.records[0].supervised_voxels
	assert runtime_token_valid_mask is not None
	assert np.array_equal(item['token_valid_mask'].numpy(), runtime_token_valid_mask)
	expected_token_valid = torch.zeros_like(item['token_valid_mask'])
	halo = context_halo_tokens[0]
	expected_token_valid[halo : halo + 8, halo : halo + 8, halo : halo + 8] = True
	assert torch.equal(item['token_valid_mask'], expected_token_valid)
	voxel_valid = item['token_valid_mask']
	for axis in range(3):
		voxel_valid = voxel_valid.repeat_interleave(8, dim=axis)
	assert not item['supervision_mask'][~voxel_valid].any()


def test_context_halo_does_not_change_supervised_core_identity(
	tmp_path: Path,
) -> None:
	halo1_root = tmp_path / 'halo1'
	halo4_root = tmp_path / 'halo4'
	halo1_root.mkdir()
	halo4_root.mkdir()
	halo1 = _dataset(halo1_root, context_halo_tokens=(1, 1, 1))
	halo4 = _dataset(halo4_root, context_halo_tokens=(4, 4, 4))
	halo1_item = halo1[0]
	halo4_item = halo4[0]

	assert halo1.records == halo4.records
	assert halo1.class_counts == halo4.class_counts
	assert [record.tile_id for record in halo1.records] == [
		record.tile_id for record in halo4.records
	]
	assert torch.equal(
		halo1_item['supervision_mask'][8:72, 8:72, 8:72],
		halo4_item['supervision_mask'][32:96, 32:96, 32:96],
	)
	assert torch.equal(
		halo1_item['labels'][8:72, 8:72, 8:72],
		halo4_item['labels'][32:96, 32:96, 32:96],
	)


def test_runtime_token_valid_mismatch_is_rejected(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	dataset = _dataset(tmp_path)
	original = end_to_end.read_amplitude_crop

	def mismatched(**kwargs: object) -> object:
		prepared = original(**kwargs)  # type: ignore[arg-type]
		mask = prepared.token_valid_mask.copy()
		mask[1, 1, 1] = False
		return replace(prepared, token_valid_mask=mask)

	monkeypatch.setattr(end_to_end, 'read_amplitude_crop', mismatched)
	with pytest.raises(ValueError, match='runtime token-valid mask'):
		dataset[0]


def test_model_forward_gradient_path_and_parameter_partition() -> None:
	mae = AmplitudeMAE3D(
		patch_size_xyz=(2, 2, 2),
		encoder_dim=12,
		encoder_depth=1,
		encoder_heads=3,
		decoder_dim=12,
		decoder_depth=1,
		decoder_heads=3,
		runtime_check_mode='strict',
	)
	decoder = VoxelDecoder3D(
		embedding_dim=12,
		class_count=2,
		hidden_channels=(8,),
		upsample_factors=((2, 2, 2),),
		patch_size_xyz=(2, 2, 2),
	)
	model = ChannelEndToEndModel(mae, decoder).float()
	amplitude = torch.randn(1, 1, 8, 8, 8)
	valid = torch.ones(1, 4, 4, 4, dtype=torch.bool)
	logits = model(amplitude, valid)
	assert logits.shape == (1, 2, 8, 8, 8)
	logits.square().mean().backward()
	assert mae.patch_projection.weight.grad is not None
	assert any(parameter.grad is not None for parameter in mae.encoder.parameters())
	assert any(parameter.grad is not None for parameter in decoder.parameters())
	assert mae.mask_token.grad is None
	assert all(parameter.grad is None for parameter in mae.decoder.parameters())
	assert all(parameter.grad is None for parameter in mae.prediction_head.parameters())
	assert all(
		parameter.grad is None for parameter in mae.encoder_to_decoder.parameters()
	)
	encoder_ids = {id(parameter) for parameter in model.encoder_parameters()}
	decoder_ids = {id(parameter) for parameter in model.decoder_parameters()}
	assert encoder_ids
	assert decoder_ids
	assert encoder_ids.isdisjoint(decoder_ids)


class _PreflightFixture:
	def __init__(self, tmp_path: Path) -> None:
		self.root = tmp_path
		self.amplitude = tmp_path / 'amplitude.npy'
		self.labels = tmp_path / 'labels.npy'
		self.labels_metadata = tmp_path / 'labels_metadata.json'
		self.stats = tmp_path / 'stats.json'
		self.reference_dir = tmp_path / 'reference'
		self.reference_dir.mkdir()
		reference_paths = output_paths(self.reference_dir, 'parihaka')
		self.reference_metadata = reference_paths.metadata
		self.valid_tokens = reference_paths.valid_tokens
		self.pretrained = (
			tmp_path
			/ 'pretraining'
			/ 'parihaka'
			/ 'facies_benchmark_v1'
			/ CHANNEL_PRETRAINED_MODEL_TAG
			/ 'full_100ep'
			/ 'latest.pt'
		)
		self.random = tmp_path / 'random' / 'mae_random_seed42.pt'
		self.layout = tmp_path / 'layouts.yaml'
		self.pretrained.parent.mkdir(parents=True)
		self.random.parent.mkdir(parents=True)
		np.save(self.amplitude, np.ones((16, 16, 16), dtype=np.float32))
		labels = np.ones((16, 16, 16), dtype=np.int8)
		labels[..., ::2] = 5
		np.save(self.labels, labels)
		write_normalization_stats(
			SurveyNormalizationStats(
				survey_id='parihaka',
				source_path=self.amplitude,
				grid_order=('x', 'y', 'z'),
				clip_low_percentile=1.0,
				clip_high_percentile=99.0,
				clip_low=-2.0,
				clip_high=2.0,
				median=0.0,
				iqr=1.0,
			),
			self.stats,
		)
		np.save(self.valid_tokens, np.ones((2, 2, 2), dtype=np.bool_))
		self.write_checkpoints()
		self.write_reference_metadata()
		self.write_label_metadata()
		self.layout.write_text(
			yaml.safe_dump(
				{
					'training_selection': {
						'semantics': (
							'stable_hash_partial_section_token_footprints_v1'
						),
						'allowed_relative_error': 0.05,
						'target_train_voxel_counts': {
							'small': 448,
							'medium': 896,
							'large': 1664,
						},
					},
					'validation': {'inline': [12], 'crossline': [12]},
					'layouts': {
						f'layout_{index:03d}': {
							'inline': [
								index,
								index + 1,
								index + 2,
								index + 3,
							],
							'crossline': [
								index,
								index + 1,
								index + 2,
								index + 3,
							],
						}
						for index in range(5)
					},
				}
			),
			encoding='utf-8',
		)

	@property
	def model_config(self) -> dict[str, object]:
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

	def write_checkpoints(
		self,
		*,
		random_metadata: dict[str, object] | None = None,
		random_model_config: dict[str, object] | None = None,
	) -> None:
		pretrained_payload = {
			'model_state_dict': {
				'patch_projection.weight': torch.ones(2),
				'encoder.layers.0.weight': torch.ones(3),
				'mask_token': torch.ones(1),
			},
			'config': {'model': self.model_config},
			'training_state': {'checkpoint_kind': 'epoch'},
			'metadata': {'pretrained_weights_loaded': True},
		}
		torch.save(pretrained_payload, self.pretrained)
		metadata = {
			'random_encoder_baseline': True,
			'pretrained_weights_loaded': False,
			'seed': 42,
			'reference_checkpoint': str(self.pretrained),
			'reference_model_tag': CHANNEL_PRETRAINED_MODEL_TAG,
		}
		if random_metadata is not None:
			metadata.update(random_metadata)
		torch.save(
			{
				'model_state_dict': {
					'patch_projection.weight': torch.zeros(2),
					'encoder.layers.0.weight': torch.zeros(3),
					'mask_token': torch.zeros(1),
				},
				'config': {
					'model': random_model_config or self.model_config,
				},
				'training_state': {'checkpoint_kind': 'random_init'},
				'metadata': metadata,
			},
			self.random,
		)

	def write_reference_metadata(self, **updates: object) -> None:
		metadata = {
			'checkpoint_path': str(self.pretrained),
			'checkpoint_sha256': file_sha256(self.pretrained),
			'source_amplitude_path': str(self.amplitude),
			'normalization_stats_path': str(self.stats),
			'volume_shape_xyz': [16, 16, 16],
			'patch_size': [8, 8, 8],
			'token_grid_shape': [2, 2, 2],
			'model_geometry': {
				'embed_dim': 384,
				'depth': 8,
				'num_heads': 6,
			},
			'min_token_valid_fraction': 1.0,
			'preprocessing': {
				'normalized_clip_abs': 8.0,
				'amplitude_agc': {'enabled': False},
				'finite_check_mode': 'strict',
			},
			'zero_mask': {'enabled': False},
		}
		metadata.update(updates)
		self.reference_metadata.write_text(json.dumps(metadata), encoding='utf-8')

	def write_label_metadata(self) -> None:
		labels = np.load(self.labels, mmap_mode='r', allow_pickle=False)
		self.labels_metadata.write_text(
			json.dumps(
				{
					'artifact_type': 'parihaka_channel_labels',
					'output_labels': str(self.labels),
					'prepared_label_identity': {
						'labels_sha256': file_sha256(self.labels),
						'source_npz_path': '/data/labels.npz',
						'source_key': 'labels',
						'shape': list(labels.shape),
						'dtype': labels.dtype.name,
						'class_definition': {
							'positive_class_id': 5,
							'negative_class_ids': [1, 2, 3, 4, 6],
						},
					},
				}
			),
			encoding='utf-8',
		)

	def config(
		self,
		*,
		core_size_tokens: tuple[int, int, int] = (8, 8, 8),
		context_halo_tokens: tuple[int, int, int] = (1, 1, 1),
	) -> ChannelEndToEndConfig:
		return ChannelEndToEndConfig(
			survey_id='parihaka',
			labels=self.labels,
			labels_metadata=self.labels_metadata,
			reference_embedding_dir=self.reference_dir,
			pretrained_checkpoint=self.pretrained,
			random_checkpoint=self.random,
			runs_root=self.root / 'runs',
			output_dir=self.root / 'summary',
			four_way_output_dir=self.root / 'four_way_summary',
			decoder=DecoderArchitecture(
				embedding_dim=384,
				class_count=2,
				hidden_channels=(128, 64, 32),
				upsample_factors=((2, 2, 2),) * 3,
				upsample_mode='nearest',
				normalization='voxelwise_layer_norm',
			),
			tiles=DecoderTiles(core_size_tokens, context_halo_tokens),
			train=ChannelEndToEndTrain(
				epochs=50,
				batch_size=1,
				encoder_learning_rate=0.0001,
				decoder_learning_rate=0.001,
				weight_decay=0.0001,
				class_weight='balanced',
				sampling_mode='all_tiles_once',
				seed=42000,
				amp=False,
				gradient_clip_norm=1.0,
			),
		)

	def config_mapping(
		self,
		*,
		core_size_tokens: tuple[int, int, int] = (8, 8, 8),
		context_halo_tokens: tuple[int, int, int] = (1, 1, 1),
	) -> dict[str, object]:
		config = self.config(
			core_size_tokens=core_size_tokens,
			context_halo_tokens=context_halo_tokens,
		)
		return {
			'dataset': {'survey_id': 'parihaka'},
			'inputs': {
				'labels_npy': str(config.labels),
				'labels_metadata_json': str(config.labels_metadata),
				'reference_embedding_dir': str(config.reference_embedding_dir),
				'pretrained_checkpoint': str(config.pretrained_checkpoint),
				'random_checkpoint': str(config.random_checkpoint),
			},
			'outputs': {
				'runs_root': str(config.runs_root),
				'output_dir': str(config.output_dir),
				'four_way_output_dir': str(config.four_way_output_dir),
			},
			'decoder': {
				'embedding_dim': 384,
				'class_count': 2,
				'hidden_channels': [128, 64, 32],
				'upsample_factors': [[2, 2, 2]] * 3,
				'upsample_mode': 'nearest',
				'normalization': 'voxelwise_layer_norm',
			},
			'tiles': {
				'core_size_tokens': list(config.tiles.core_size_tokens),
				'context_halo_tokens': list(config.tiles.context_halo_tokens),
			},
			'train': {
				'epochs': 50,
				'batch_size': 1,
				'encoder_learning_rate': 0.0001,
				'decoder_learning_rate': 0.001,
				'weight_decay': 0.0001,
				'class_weight': 'balanced',
				'sampling_mode': 'all_tiles_once',
				'seed': 42000,
				'amp': False,
				'gradient_clip_norm': 1.0,
			},
		}

	def plan(
		self,
		encoder_init: str = 'pretrained',
		*,
		context_halo_tokens: tuple[int, int, int] = (1, 1, 1),
	) -> Any:
		return inspect_channel_end_to_end_job(
			self.config(context_halo_tokens=context_halo_tokens),
			encoder_init=encoder_init,
			layout_id='layout_000',
			data_size='small',
			layout_config=self.layout,
			device='cpu',
		)


def test_preflight_geometry_identity_and_condition_parity(tmp_path: Path) -> None:
	fixture = _PreflightFixture(tmp_path)
	pretrained = fixture.plan('pretrained')
	random = fixture.plan('random')
	assert pretrained.model_geometry == random.model_geometry
	assert pretrained.model_geometry.encoder_dim == 384
	assert pretrained.decoder_initial_state_sha256 == (
		random.decoder_initial_state_sha256
	)
	assert pretrained.tile_ids == random.tile_ids
	assert pretrained.class_weights == random.class_weights
	assert pretrained.output_dir != random.output_dir
	assert 'encoder_init=pretrained' in str(pretrained.output_dir)
	assert 'encoder_init=random' in str(random.output_dir)
	assert pretrained.benchmark_identity['encoder_init'] == 'pretrained'
	assert random.benchmark_identity['encoder_init'] == 'random'
	assert pretrained.benchmark_identity['optimizer'] == {
		'encoder_learning_rate': 0.0001,
		'decoder_learning_rate': 0.001,
		'weight_decay': 0.0001,
		'parameter_group_names': ['encoder', 'decoder'],
	}


def test_halo4_preflight_uses_validator_and_preserves_condition_parity(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	fixture = _PreflightFixture(tmp_path)
	validated: list[dict[str, object]] = []
	original = end_to_end.validate_context_halo_tokens

	def wrapped(**kwargs: object) -> None:
		validated.append(dict(kwargs))
		original(**kwargs)  # type: ignore[arg-type]

	monkeypatch.setattr(end_to_end, 'validate_context_halo_tokens', wrapped)
	pretrained = fixture.plan(
		'pretrained', context_halo_tokens=(4, 4, 4)
	)
	random = fixture.plan('random', context_halo_tokens=(4, 4, 4))

	assert len(validated) == 2
	assert all(
		call['context_halo_tokens'] == (4, 4, 4) for call in validated
	)
	assert pretrained.config.tiles == DecoderTiles((8, 8, 8), (4, 4, 4))
	assert pretrained.benchmark_identity['tiles'] == {
		'core_size_tokens': [8, 8, 8],
		'context_halo_tokens': [4, 4, 4],
	}
	assert random.benchmark_identity['tiles'] == pretrained.benchmark_identity['tiles']
	assert pretrained.tile_ids == random.tile_ids
	assert pretrained.split_counts == random.split_counts


def test_checkpoint_geometry_mismatch_is_rejected(tmp_path: Path) -> None:
	fixture = _PreflightFixture(tmp_path)
	random_geometry = dict(fixture.model_config)
	random_geometry['encoder_depth'] = 6
	fixture.write_checkpoints(random_model_config=random_geometry)
	fixture.write_reference_metadata()
	with pytest.raises(ValueError, match='model geometry mismatch'):
		fixture.plan()


def test_random_checkpoint_role_metadata_is_required(tmp_path: Path) -> None:
	fixture = _PreflightFixture(tmp_path)
	fixture.write_checkpoints(random_metadata={'random_encoder_baseline': False})
	fixture.write_reference_metadata()
	with pytest.raises(ValueError, match='random_encoder_baseline'):
		fixture.plan()


def test_checkpoint_roles_cannot_be_swapped(tmp_path: Path) -> None:
	fixture = _PreflightFixture(tmp_path)
	config = replace(fixture.config(), random_checkpoint=fixture.pretrained)
	with pytest.raises(ValueError, match='checkpoint_sha256 must differ'):
		inspect_channel_end_to_end_job(
			config,
			encoder_init='random',
			layout_id='layout_000',
			data_size='small',
			layout_config=fixture.layout,
			device='cpu',
		)


def test_checkpoint_source_identity_can_declare_barlow_experiment(
	tmp_path: Path,
) -> None:
	model_tag = 'barlow_twins_3d_channel_v1'
	pretrained = tmp_path / 'pretraining' / model_tag / 'latest.pt'
	random = tmp_path / 'random.pt'
	pretrained.parent.mkdir(parents=True)
	torch.save({'model_state_dict': {}}, pretrained)
	torch.save(
		{
			'model_state_dict': {},
			'metadata': {
				'random_encoder_baseline': True,
				'pretrained_weights_loaded': False,
				'seed': 42,
				'reference_checkpoint': str(pretrained),
				'reference_model_tag': model_tag,
			},
			'training_state': {'checkpoint_kind': 'random_init'},
		},
		random,
	)
	pretrained_source, random_source = inspect_channel_model_sources(
		{
			'checkpoint_path': str(pretrained),
			'checkpoint_sha256': file_sha256(pretrained),
		},
		{
			'checkpoint_path': str(random),
			'checkpoint_sha256': file_sha256(random),
		},
		pretrained_model_tag=model_tag,
		pretrained_checkpoint_suffix=(model_tag, 'latest.pt'),
	)

	assert pretrained_source['model_tag'] == model_tag
	assert random_source['reference_model_tag'] == model_tag


def test_checkpoint_file_sha_mismatch_is_rejected(tmp_path: Path) -> None:
	fixture = _PreflightFixture(tmp_path)
	payload = torch.load(fixture.pretrained, weights_only=False)
	payload['model_state_dict']['encoder.layers.0.weight'] = torch.zeros(3)
	torch.save(payload, fixture.pretrained)
	with pytest.raises(ValueError, match='does not match checkpoint file'):
		fixture.plan()


def test_reference_checkpoint_path_mismatch_is_rejected(tmp_path: Path) -> None:
	fixture = _PreflightFixture(tmp_path)
	fixture.write_reference_metadata(checkpoint_path=str(fixture.random))
	with pytest.raises(ValueError, match='does not match pretrained_checkpoint'):
		fixture.plan()


@pytest.mark.parametrize('artifact', ['labels', 'amplitude', 'valid_tokens'])
def test_preflight_rejects_volume_shape_mismatch(
	tmp_path: Path, artifact: str
) -> None:
	fixture = _PreflightFixture(tmp_path)
	if artifact == 'labels':
		np.save(fixture.labels, np.ones((15, 16, 16), dtype=np.int8))
	elif artifact == 'amplitude':
		np.save(fixture.amplitude, np.ones((15, 16, 16), dtype=np.float32))
	else:
		np.save(fixture.valid_tokens, np.ones((1, 2, 2), dtype=np.bool_))
	with pytest.raises(ValueError, match='shape'):
		fixture.plan()


def test_all_splits_require_both_classes(tmp_path: Path) -> None:
	fixture = _PreflightFixture(tmp_path)
	np.save(fixture.labels, np.ones((16, 16, 16), dtype=np.int8))
	fixture.write_label_metadata()
	with pytest.raises(ValueError, match='both Channel and non-Channel'):
		fixture.plan()


def test_frozen_and_end_to_end_supervision_counts_match(tmp_path: Path) -> None:
	fixture = _PreflightFixture(tmp_path)
	plan = fixture.plan()
	embedding_dir = tmp_path / 'frozen'
	embedding_dir.mkdir()
	paths = output_paths(embedding_dir, 'parihaka')
	np.save(paths.embeddings, np.zeros((2, 2, 2, 384), dtype=np.float16))
	geometry = EmbeddingGeometry(
		pretrained=paths,
		random=paths,
		volume_shape_xyz=(16, 16, 16),
		token_grid_shape_xyz=(2, 2, 2),
		patch_size_xyz=(8, 8, 8),
		embedding_shape=(2, 2, 2, 384),
		embedding_dim=384,
		pretrained_metadata={},
		random_metadata={},
		pretrained_model_source={},
		random_model_source={},
	)
	for split in ('train', 'validation', 'test'):
		frozen = ChannelTileDataset(
			embedding_path=paths.embeddings,
			valid_tokens_path=fixture.valid_tokens,
			labels_path=fixture.labels,
			geometry=geometry,
			lines=plan.train_lines,
			validation=plan.layouts.validation,
			reserved_training=plan.reserved_training_lines,
			split=split,
			tiles=plan.config.tiles,
			training_selection_mask=(
				np.ones(plan.reference.token_grid_shape_xyz, dtype=np.bool_)
				if split == 'train'
				else None
			),
		)
		assert frozen.class_counts == plan.split_counts[split]
		assert len(frozen) == plan.tile_counts[split]
		assert tuple(record.tile_id for record in frozen.records) == (
			plan.tile_ids[split]
		)
		end_dataset = ChannelAmplitudeTileDataset(
			reference=plan.reference,
			labels_path=fixture.labels,
			lines=plan.train_lines,
			validation=plan.layouts.validation,
			reserved_training=plan.reserved_training_lines,
			split=split,
			core_size_tokens=plan.config.tiles.core_size_tokens,
			context_halo_tokens=plan.config.tiles.context_halo_tokens,
			training_selection_mask=(
				np.ones(plan.reference.token_grid_shape_xyz, dtype=np.bool_)
				if split == 'train'
				else None
			),
		)
		assert end_dataset.records == frozen.records
		for index in range(len(frozen)):
			assert torch.equal(
				frozen[index]['supervision_mask'],
				end_dataset[index]['supervision_mask'],
			)


def test_common_test_tiles_and_counts_are_layout_and_size_invariant(
	tmp_path: Path,
) -> None:
	fixture = _PreflightFixture(tmp_path)
	plan = fixture.plan()
	common_test: tuple[tuple[int, ...], tuple[int, int]] | None = None
	common_validation: tuple[tuple[int, ...], tuple[int, int]] | None = None
	train_counts_by_size: dict[str, set[tuple[int, int]]] = {
		data_size: set() for data_size in DATA_SIZE_PREFIX
	}
	for layout_id in LAYOUT_IDS:
		for data_size in DATA_SIZE_PREFIX:
			lines = selected_training_lines(plan.layouts, layout_id, data_size)
			datasets = {
				split: ChannelAmplitudeTileDataset(
					reference=plan.reference,
					labels_path=fixture.labels,
					lines=lines,
					validation=plan.layouts.validation,
					reserved_training=plan.reserved_training_lines,
					split=split,
					core_size_tokens=plan.config.tiles.core_size_tokens,
					context_halo_tokens=plan.config.tiles.context_halo_tokens,
					training_selection_mask=(
						np.ones(plan.reference.token_grid_shape_xyz, dtype=np.bool_)
						if split == 'train'
						else None
					),
				)
				for split in ('train', 'validation', 'test')
			}
			test_identity = (
				tuple(record.tile_id for record in datasets['test'].records),
				datasets['test'].class_counts,
			)
			validation_identity = (
				tuple(record.tile_id for record in datasets['validation'].records),
				datasets['validation'].class_counts,
			)
			common_test = test_identity if common_test is None else common_test
			common_validation = (
				validation_identity
				if common_validation is None
				else common_validation
			)
			assert test_identity == common_test
			assert validation_identity == common_validation
			assert all(count > 0 for count in datasets['test'].class_counts)
			train_counts_by_size[data_size].add(datasets['train'].class_counts)
	assert len({next(iter(values)) for values in train_counts_by_size.values()}) == 3


def test_optimizer_groups_exclude_unused_mae_decoder() -> None:
	mae = AmplitudeMAE3D(
		patch_size_xyz=(2, 2, 2),
		encoder_dim=12,
		encoder_depth=1,
		encoder_heads=3,
		decoder_dim=12,
		decoder_depth=1,
		decoder_heads=3,
	)
	decoder = VoxelDecoder3D(
		embedding_dim=12,
		class_count=2,
		hidden_channels=(8,),
		upsample_factors=((2, 2, 2),),
		patch_size_xyz=(2, 2, 2),
	)
	model = ChannelEndToEndModel(mae, decoder)
	groups = channel_end_to_end_optimizer_groups(
		model,
		encoder_learning_rate=0.0001,
		decoder_learning_rate=0.001,
		weight_decay=0.0001,
	)
	assert [group['name'] for group in groups] == ['encoder', 'decoder']
	assert [group['lr'] for group in groups] == [0.0001, 0.001]
	parameter_ids = {
		id(parameter)
		for group in groups
		for parameter in group['params']
	}
	unused = [
		mae.mask_token,
		*mae.encoder_to_decoder.parameters(),
		*mae.decoder.parameters(),
		*mae.prediction_head.parameters(),
	]
	assert all(id(parameter) not in parameter_ids for parameter in unused)


def test_config_mapping_and_cli_dry_run_are_read_only(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	fixture = _PreflightFixture(tmp_path)
	config_path = tmp_path / 'config.yaml'
	config_path.write_text(
		yaml.safe_dump(fixture.config_mapping()), encoding='utf-8'
	)
	resolved = channel_end_to_end_config_from_mapping(fixture.config_mapping())
	assert resolved.pretrained_checkpoint == fixture.pretrained
	assert resolved.tiles == DecoderTiles((8, 8, 8), (1, 1, 1))
	before = {path.relative_to(tmp_path) for path in tmp_path.rglob('*')}
	monkeypatch.setattr(
		'sys.argv',
		[
			'run_parihaka_channel_end_to_end.py',
			'--config',
			str(config_path),
			'--encoder-init',
			'pretrained',
			'--layout',
			'layout_000',
			'--size',
			'small',
			'--layout-config',
			str(fixture.layout),
			'--device',
			'cpu',
			'--dry-run',
		],
	)
	end_to_end_cli.main()
	after = {path.relative_to(tmp_path) for path in tmp_path.rglob('*')}
	assert after == before
	output = capsys.readouterr().out
	assert 'encoder_init: pretrained' in output
	assert 'execution: dry-run; no files written' in output


def test_end_to_end_config_preserves_halo4(tmp_path: Path) -> None:
	fixture = _PreflightFixture(tmp_path)
	resolved = channel_end_to_end_config_from_mapping(
		fixture.config_mapping(context_halo_tokens=(4, 4, 4))
	)
	assert resolved.tiles == DecoderTiles((8, 8, 8), (4, 4, 4))


def test_end_to_end_config_rejects_changed_core_size(tmp_path: Path) -> None:
	fixture = _PreflightFixture(tmp_path)
	with pytest.raises(ValueError, match='core_size_tokens'):
		channel_end_to_end_config_from_mapping(
			fixture.config_mapping(core_size_tokens=(16, 16, 16))
		)


@pytest.mark.parametrize(
	'context_halo_tokens',
	[
		[-1, 1, 1],
		[True, 1, 1],
		[1.0, 1, 1],
		[1, 1],
	],
)
def test_end_to_end_config_rejects_invalid_context_halo(
	tmp_path: Path,
	context_halo_tokens: list[object],
) -> None:
	fixture = _PreflightFixture(tmp_path)
	raw = fixture.config_mapping()
	raw_tiles = raw['tiles']
	assert isinstance(raw_tiles, dict)
	raw_tiles['context_halo_tokens'] = context_halo_tokens
	with pytest.raises((TypeError, ValueError), match='context_halo_tokens'):
		channel_end_to_end_config_from_mapping(raw)


def test_end_to_end_config_rejects_amp_true(tmp_path: Path) -> None:
	fixture = _PreflightFixture(tmp_path)
	raw = fixture.config_mapping()
	raw['train']['amp'] = True
	with pytest.raises(ValueError, match='train settings differ'):
		channel_end_to_end_config_from_mapping(raw)


class _TinyChannelDataset(torch.utils.data.Dataset[dict[str, object]]):
	def __init__(self, count: int) -> None:
		self.count = count

	def __len__(self) -> int:
		return self.count

	def __getitem__(self, index: int) -> dict[str, object]:
		labels = (torch.arange(64).reshape(4, 4, 4) % 2).to(torch.int64)
		return {
			'amplitude': torch.linspace(-1.0, 1.0, 64).reshape(1, 4, 4, 4)
			+ index * 0.01,
			'token_valid_mask': torch.ones(2, 2, 2, dtype=torch.bool),
			'labels': labels,
			'supervision_mask': torch.ones(4, 4, 4, dtype=torch.bool),
			'core_mask': torch.ones(4, 4, 4, dtype=torch.bool),
			'tile_id': index,
		}


def _tiny_model(encoder_init: str = 'pretrained') -> ChannelEndToEndModel:
	with torch.random.fork_rng(devices=[]):
		torch.manual_seed(11 if encoder_init == 'pretrained' else 12)
		mae = AmplitudeMAE3D(
			patch_size_xyz=(2, 2, 2),
			encoder_dim=12,
			encoder_depth=1,
			encoder_heads=3,
			decoder_dim=12,
			decoder_depth=1,
			decoder_heads=3,
			runtime_check_mode='strict',
		)
	with torch.random.fork_rng(devices=[]):
		torch.manual_seed(42000)
		decoder = VoxelDecoder3D(
			embedding_dim=12,
			class_count=2,
			hidden_channels=(8,),
			upsample_factors=((2, 2, 2),),
			patch_size_xyz=(2, 2, 2),
		)
	return ChannelEndToEndModel(mae, decoder).float()


def test_fp32_cuda_runtime_disables_autocast_and_grad_scaler(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(end_to_end.torch.cuda, 'is_available', lambda: True)
	runtime = resolve_channel_end_to_end_runtime('cuda', amp=False)
	assert runtime.device_type == 'cuda'
	assert runtime.amp_enabled is False
	assert runtime.autocast_dtype is None
	assert runtime.grad_scaler_enabled is False


def _tiny_training_plan(
	tmp_path: Path,
	*,
	encoder_init: str = 'pretrained',
	name: str = 'run',
	context_halo_tokens: tuple[int, int, int] = (1, 1, 1),
) -> Any:
	fixture_root = tmp_path / f'fixture-{name}'
	fixture_root.mkdir()
	fixture = _PreflightFixture(fixture_root)
	plan = fixture.plan(
		encoder_init, context_halo_tokens=context_halo_tokens
	)
	train = replace(plan.config.train, epochs=2)
	config = replace(plan.config, train=train)
	identity = json.loads(json.dumps(plan.benchmark_identity))
	identity['training']['epochs'] = 2
	return replace(
		plan,
		config=config,
		output_dir=tmp_path / name,
		benchmark_identity=identity,
	)


def _patch_tiny_training(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(
		end_to_end,
		'build_channel_end_to_end_model',
		lambda plan: _tiny_model(plan.encoder_init),
	)
	monkeypatch.setattr(
		end_to_end,
		'_channel_end_to_end_datasets',
		lambda _plan: {
			'train': _TinyChannelDataset(3),
			'validation': _TinyChannelDataset(1),
			'test': _TinyChannelDataset(1),
		},
	)


@pytest.mark.parametrize('encoder_init', ['pretrained', 'random'])
def test_joint_step_updates_encoder_and_decoder_only(
	encoder_init: str,
) -> None:
	model = _tiny_model(encoder_init)
	groups = channel_end_to_end_optimizer_groups(
		model,
		encoder_learning_rate=0.0001,
		decoder_learning_rate=0.001,
		weight_decay=0.0001,
	)
	optimizer = torch.optim.AdamW(groups)
	encoder_before = [
		parameter.detach().clone() for parameter in model.encoder_parameters()
	]
	decoder_before = [
		parameter.detach().clone() for parameter in model.decoder_parameters()
	]
	unused_before = {
		name: value.detach().clone()
		for name, value in model.mae.state_dict().items()
		if not name.startswith(('patch_projection.', 'encoder.'))
	}
	metrics = train_channel_end_to_end_step(
		model,
		_TinyChannelDataset(1),
		0,
		optimizer,
		None,
		torch.ones(2),
		torch.device('cpu'),
		amp_enabled=False,
		grad_clip_norm=1.0,
	)
	assert np.isfinite(metrics['loss'])
	assert any(
		not torch.equal(before, after)
		for before, after in zip(
			encoder_before, model.encoder_parameters(), strict=True
		)
	)
	assert any(
		not torch.equal(before, after)
		for before, after in zip(
			decoder_before, model.decoder_parameters(), strict=True
		)
	)
	assert all(
		parameter.grad is not None and torch.isfinite(parameter.grad).all()
		for parameter in model.encoder_parameters()
	)
	assert all(
		parameter.grad is not None and torch.isfinite(parameter.grad).all()
		for parameter in model.decoder_parameters()
	)
	assert model.mae.mask_token.grad is None
	assert all(parameter.grad is None for parameter in model.mae.decoder.parameters())
	assert all(
		torch.equal(unused_before[name], value)
		for name, value in model.mae.state_dict().items()
		if name in unused_before
	)


def test_interrupted_resume_matches_uninterrupted_job(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	_patch_tiny_training(monkeypatch)
	uninterrupted = _tiny_training_plan(tmp_path, name='uninterrupted')
	resumed = replace(uninterrupted, output_dir=tmp_path / 'resumed')
	assert run_channel_end_to_end_job(uninterrupted) is not None
	assert run_channel_end_to_end_job(resumed, max_steps=2) is None
	latest_path = resumed.output_dir / LATEST_NAME
	interrupted_payload = torch.load(latest_path, weights_only=False)
	assert interrupted_payload['global_step'] == 2
	assert interrupted_payload['next_position'] == 2
	assert {
		'schema_version',
		'completed',
		'run_identity',
		'encoder_state_dict',
		'decoder_state_dict',
		'optimizer_state_dict',
		'scaler_state_dict',
		'train_loss_sum',
		'train_confusion',
		'train_voxels',
		'python_rng_state',
		'numpy_rng_state',
		'torch_cpu_rng_state',
		'torch_cuda_rng_state',
	} <= interrupted_payload.keys()
	assert run_channel_end_to_end_job(resumed, resume=latest_path) is not None
	full = torch.load(uninterrupted.output_dir / LATEST_NAME, weights_only=False)
	actual = torch.load(latest_path, weights_only=False)
	assert full['history'] == actual['history']
	for state_name in ('encoder_state_dict', 'decoder_state_dict'):
		assert full[state_name].keys() == actual[state_name].keys()
		assert all(
			torch.equal(full[state_name][key], actual[state_name][key])
			for key in full[state_name]
		)
	assert {path.name for path in resumed.output_dir.iterdir()} == {
		LATEST_NAME,
		BEST_NAME,
		HISTORY_NAME,
		METRICS_NAME,
	}
	with pytest.raises(ValueError, match='completed'):
		run_channel_end_to_end_job(resumed, resume=latest_path)


def test_resume_rejects_identity_drift_and_new_run_rejects_output(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	_patch_tiny_training(monkeypatch)
	plan = _tiny_training_plan(tmp_path, name='drift')
	assert run_channel_end_to_end_job(plan, max_steps=1) is None
	latest = plan.output_dir / LATEST_NAME
	drifted_identity = dict(plan.benchmark_identity)
	drifted_identity['encoder_init'] = 'random'
	drifted = replace(plan, benchmark_identity=drifted_identity)
	with pytest.raises(ValueError, match='does not match'):
		run_channel_end_to_end_job(drifted, resume=latest)
	precision_identity = json.loads(json.dumps(plan.benchmark_identity))
	precision_identity['runtime']['amp_enabled'] = True
	precision_drifted = replace(plan, benchmark_identity=precision_identity)
	with pytest.raises(ValueError, match='does not match'):
		run_channel_end_to_end_job(precision_drifted, resume=latest)
	with pytest.raises(FileExistsError, match='non-empty'):
		run_channel_end_to_end_job(plan)


def test_halo4_checkpoint_resumes_with_same_plan(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	_patch_tiny_training(monkeypatch)
	plan = _tiny_training_plan(
		tmp_path,
		name='halo4-resume',
		context_halo_tokens=(4, 4, 4),
	)
	assert run_channel_end_to_end_job(plan, max_steps=1) is None
	latest = plan.output_dir / LATEST_NAME
	assert run_channel_end_to_end_job(plan, resume=latest) is not None


@pytest.mark.parametrize(
	('checkpoint_halo', 'resume_halo'),
	[
		((1, 1, 1), (4, 4, 4)),
		((4, 4, 4), (1, 1, 1)),
	],
)
def test_resume_rejects_context_halo_drift(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	checkpoint_halo: tuple[int, int, int],
	resume_halo: tuple[int, int, int],
) -> None:
	_patch_tiny_training(monkeypatch)
	plan = _tiny_training_plan(
		tmp_path,
		name=f'halo-{checkpoint_halo[0]}-to-{resume_halo[0]}',
		context_halo_tokens=checkpoint_halo,
	)
	assert run_channel_end_to_end_job(plan, max_steps=1) is None
	latest = plan.output_dir / LATEST_NAME
	resume_identity = json.loads(json.dumps(plan.benchmark_identity))
	resume_identity['tiles']['context_halo_tokens'] = list(resume_halo)
	resume_config = replace(
		plan.config,
		tiles=DecoderTiles((8, 8, 8), resume_halo),
	)
	resume_plan = replace(
		plan,
		config=resume_config,
		benchmark_identity=resume_identity,
	)
	with pytest.raises(ValueError, match='does not match'):
		run_channel_end_to_end_job(resume_plan, resume=latest)


def test_strict_best_selection_and_test_reload_best_state(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	_patch_tiny_training(monkeypatch)
	plan = _tiny_training_plan(tmp_path, name='best')
	observed_states: list[dict[str, torch.Tensor]] = []

	def evaluate(
		model: ChannelEndToEndModel,
		_dataset: object,
		_weights: object,
		_device: object,
		*,
		amp_enabled: bool,
	) -> dict[str, object]:
		assert not amp_enabled
		observed_states.append(
			{
				key: value.detach().clone()
				for key, value in model.voxel_decoder.state_dict().items()
			}
		)
		return {
			'channel_iou': 0.5,
			'channel_f1': 2.0 / 3.0,
			'channel_precision': 0.5,
			'channel_recall': 1.0,
			'balanced_accuracy': 0.5,
			'confusion_matrix': [[0, 32], [0, 32]],
			'loss': 1.0,
			'supervised_voxel_count': 64,
		}

	monkeypatch.setattr(end_to_end, '_evaluate_channel_end_to_end', evaluate)
	metrics_path = run_channel_end_to_end_job(plan)
	assert metrics_path == plan.output_dir / METRICS_NAME
	assert len(observed_states) == 3
	assert all(
		torch.equal(observed_states[0][key], observed_states[2][key])
		for key in observed_states[0]
	)
	payload = json.loads(metrics_path.read_text(encoding='utf-8'))
	assert payload['best_epoch'] == 0
	assert payload['test']['confusion_matrix'] == [[0, 32], [0, 32]]
	assert payload['condition_name'] == 'finetune_pretrained'
