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
)
from seis_ssl_cluster.parihaka.channel_data import SectionLines
from seis_ssl_cluster.parihaka.channel_decoder import (
	ChannelTileDataset,
	DecoderArchitecture,
	DecoderTiles,
	EmbeddingGeometry,
)
from seis_ssl_cluster.parihaka.channel_end_to_end import (
	ChannelAmplitudeTileDataset,
	ChannelEndToEndConfig,
	ChannelEndToEndModel,
	ChannelEndToEndTrain,
	channel_end_to_end_config_from_mapping,
	channel_end_to_end_optimizer_groups,
	inspect_channel_end_to_end_job,
	resolve_channel_reference_artifact,
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
	np.save(
		artifact_dir / 'parihaka.valid_tokens.npy',
		np.ones((8, 8, 8), dtype=np.bool_),
	)
	(artifact_dir / 'parihaka.metadata.json').write_text(
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


def _dataset(tmp_path: Path) -> ChannelAmplitudeTileDataset:
	artifact_dir, labels_path = _raw_fixture(tmp_path)
	return ChannelAmplitudeTileDataset(
		reference=resolve_channel_reference_artifact(artifact_dir),
		labels_path=labels_path,
		lines=SectionLines((0,), (0,)),
		validation=SectionLines((62,), (62,)),
		test=SectionLines((63,), (63,)),
		split='train',
	)


def test_raw_amplitude_uses_shared_preprocessing_and_matches_reference(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	dataset = _dataset(tmp_path)
	called = 0
	original = end_to_end.read_amplitude_crop

	def wrapped(**kwargs: object) -> object:
		nonlocal called
		called += 1
		return original(**kwargs)  # type: ignore[arg-type]

	monkeypatch.setattr(end_to_end, 'read_amplitude_crop', wrapped)
	item = dataset[0]
	assert called == 1
	assert item['amplitude'].dtype == torch.float32
	assert item['amplitude'].shape == (1, 80, 80, 80)
	assert item['token_valid_mask'].shape == (10, 10, 10)
	assert item['labels'].shape == (80, 80, 80)
	assert item['core_mask'][8:72, 8:72, 8:72].all()
	assert int(item['supervision_mask'].sum()) == dataset.records[0].supervised_voxels


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
		self.reference_metadata = self.reference_dir / 'parihaka.metadata.json'
		self.valid_tokens = self.reference_dir / 'parihaka.valid_tokens.npy'
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
					'validation': {'inline': [12], 'crossline': [12]},
					'test': {'inline': [13], 'crossline': [13]},
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
			'encoder_depth': 12,
			'encoder_heads': 6,
			'decoder_dim': 256,
			'decoder_depth': 4,
			'decoder_heads': 8,
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
				'depth': 12,
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

	def config(self) -> ChannelEndToEndConfig:
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
			tiles=DecoderTiles((8, 8, 8), (1, 1, 1)),
			train=ChannelEndToEndTrain(
				epochs=50,
				batch_size=1,
				encoder_learning_rate=0.0001,
				decoder_learning_rate=0.001,
				weight_decay=0.0001,
				class_weight='balanced',
				sampling_mode='all_tiles_once',
				seed=42000,
				amp=True,
				gradient_clip_norm=1.0,
			),
		)

	def config_mapping(self) -> dict[str, object]:
		config = self.config()
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
				'core_size_tokens': [8, 8, 8],
				'context_halo_tokens': [1, 1, 1],
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
				'amp': True,
				'gradient_clip_norm': 1.0,
			},
		}

	def plan(self, encoder_init: str = 'pretrained') -> Any:
		return inspect_channel_end_to_end_job(
			self.config(),
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
			test=plan.layouts.test,
			split=split,
			tiles=plan.config.tiles,
		)
		assert frozen.class_counts == plan.split_counts[split]
		assert len(frozen) == plan.tile_counts[split]
		assert tuple(record.tile_id for record in frozen.records) == (
			plan.tile_ids[split]
		)


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
