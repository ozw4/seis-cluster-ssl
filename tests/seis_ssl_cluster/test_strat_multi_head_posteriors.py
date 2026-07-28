"""Coverage for soft multi-head posterior supervision."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	CropRequest,
	MultiHeadStratPosteriorProvider,
	NopimsStratMultiHeadPosteriorDataset,
	StratMultiHeadPosteriorInput,
	StratMultiHeadPosteriorManifest,
	SurveyManifest,
	SurveyNormalizationStats,
	TargetProviderContext,
	ZeroMaskConfig,
	target_providers,
	write_normalization_stats,
)
from seis_ssl_cluster.stratigraphy import MultiResolutionOrderedPrototypeHeads
from seis_ssl_cluster.stratigraphy.losses import (
	soft_categorical_cross_entropy,
	structured_hmm_prototype_loss,
)
from seis_ssl_cluster.training.collate import (
	move_batch_to_device,
	strat_multi_head_posterior_collate_fn,
)
from seis_ssl_cluster.training.dataloaders import (
	build_strat_multi_head_posterior_dataloader,
)
from seis_ssl_cluster.training.strat_hmm.losses import (
	compute_strat_hmm_multi_head_posterior_losses,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_soft_categorical_cross_entropy_is_detached_and_graph_safe() -> None:
	logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
	target = torch.tensor(
		[[[1.0, 0.0], [0.0, 0.0]]], requires_grad=True
	)
	valid_mask = torch.tensor([[True, False]])

	loss = soft_categorical_cross_entropy(logits, target, valid_mask=valid_mask)
	loss.backward()

	expected = -torch.nn.functional.log_softmax(logits, -1)[0, 0, 0]
	assert torch.allclose(loss.detach(), expected)
	assert target.grad is None
	assert logits.grad is not None
	empty = soft_categorical_cross_entropy(
		logits,
		torch.zeros_like(target),
		valid_mask=torch.zeros_like(valid_mask),
	)
	empty.backward()


def test_soft_cross_entropy_matches_hard_ce_for_one_hot_targets() -> None:
	logits = torch.tensor(
		[[[1.0, -2.0, 0.5], [-1.0, 0.0, 3.0]]], requires_grad=True
	)
	labels = torch.tensor([[0, 2]])
	valid_mask = torch.tensor([[True, True]])
	posterior = torch.nn.functional.one_hot(labels, 3).float()

	soft = soft_categorical_cross_entropy(logits, posterior, valid_mask=valid_mask)
	hard = structured_hmm_prototype_loss(
		logits,
		labels,
		valid_mask=valid_mask,
		confidence=torch.ones_like(labels, dtype=torch.float32),
		boundary_weight=torch.ones_like(labels, dtype=torch.float32),
	)

	torch.testing.assert_close(soft, hard)


def test_posterior_collate_stacks_nested_targets_and_moves_them() -> None:
	batch = strat_multi_head_posterior_collate_fn([_sample(), _sample()])
	moved = move_batch_to_device(batch, torch.device('cpu'))

	assert list(batch['strat_multi_posteriors']) == ['k6', 'k8', 'k10']
	assert batch['strat_multi_posteriors']['k8']['posterior'].shape == (2, 1, 2, 2, 8)
	assert moved['strat_multi_posteriors']['k10']['valid_mask'].device.type == 'cpu'


def test_posterior_collate_rejects_bad_head_shape() -> None:
	first = _sample()
	second = _sample()
	second['strat_multi_posteriors']['k8']['posterior'] = np.zeros(
		(1, 2, 2, 7), dtype=np.float32
	)

	with pytest.raises(ValueError, match='last dimension'):
		strat_multi_head_posterior_collate_fn([first, second])


def test_posterior_losses_average_heads_without_consistency() -> None:
	torch.manual_seed(289)
	heads = MultiResolutionOrderedPrototypeHeads(
		feature_dim=3,
		ks=(6, 8, 10),
		projection_dim=2,
		temperature=0.1,
		normalize=True,
	)
	tokens = torch.randn(1, 4, 3, requires_grad=True)
	batch = {
		'strat_multi_posteriors': {
			f'k{k}': {
				'posterior': torch.nn.functional.one_hot(
					torch.zeros((1, 4), dtype=torch.long), k
				).float(),
				'valid_mask': torch.ones((1, 4), dtype=torch.bool),
			}
			for k in (6, 8, 10)
		},
	}
	result = compute_strat_hmm_multi_head_posterior_losses(
		heads=heads,
		encoded={
			'tokens': tokens,
			'token_valid_mask': torch.tensor([[True, False, True, False]]),
		},
		teacher_encoded=None,
		batch=batch,
		loss_config={'usage_weight': 0.1},
	)

	assert torch.allclose(
		result['loss_prototype'],
		torch.stack([result[f'loss_prototype_k{k}'] for k in (6, 8, 10)]).mean(),
	)
	assert {'target_entropy_k6', 'prototype_kl_k8', 'loss_distillation'} <= set(result)
	result['loss'].backward()
	assert tokens.grad is not None


def test_posterior_losses_apply_prototype_weight() -> None:
	torch.manual_seed(290)
	heads = MultiResolutionOrderedPrototypeHeads(
		feature_dim=3,
		ks=(6, 8, 10),
		projection_dim=2,
		temperature=0.1,
		normalize=True,
	)
	tokens = torch.randn(1, 4, 3, requires_grad=True)
	batch = {
		'strat_multi_posteriors': {
			f'k{k}': {
				'posterior': torch.nn.functional.one_hot(
					torch.zeros((1, 4), dtype=torch.long), k
				).float(),
				'valid_mask': torch.ones((1, 4), dtype=torch.bool),
			}
			for k in (6, 8, 10)
		},
	}

	result = compute_strat_hmm_multi_head_posterior_losses(
		heads=heads,
		encoded={'tokens': tokens},
		teacher_encoded=None,
		batch=batch,
		loss_config={'prototype_weight': 0.0},
	)

	assert result['loss_prototype'] > 0.0
	torch.testing.assert_close(result['loss'], torch.zeros_like(result['loss']))
	result['loss'].backward()
	assert tokens.grad is not None


def test_posterior_provider_crops_memmaps_and_applies_token_mask(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	provider, paths = _posterior_provider(tmp_path)
	materialized_memmap_shapes: list[tuple[int, ...]] = []
	loaded_memmaps: list[np.ndarray] = []
	original_asarray = target_providers.np.asarray
	original_load = target_providers.np.load

	def spy_load(path: object, *args: object, **kwargs: object) -> np.ndarray:
		loaded = original_load(path, *args, **kwargs)
		if kwargs.get('mmap_mode') == 'r':
			loaded_memmaps.append(loaded)
		return loaded

	def spy_asarray(value: object, *args: object, **kwargs: object) -> np.ndarray:
		if isinstance(value, np.memmap):
			materialized_memmap_shapes.append(value.shape)
		return original_asarray(value, *args, **kwargs)

	monkeypatch.setattr(target_providers.np, 'asarray', spy_asarray)
	monkeypatch.setattr(target_providers.np, 'load', spy_load)
	token_valid_mask = np.ones((2, 3, 4), dtype=np.bool_)
	token_valid_mask[0, 0, 0] = False
	sample: dict[str, object] = {'coords': {}}

	provider.add_targets(
		sample,
		_posterior_context(tmp_path, token_valid_mask=token_valid_mask),
	)

	assert set(sample) == {'coords', 'strat_multi_posteriors'}
	targets = sample['strat_multi_posteriors']
	assert list(targets) == ['k6', 'k8', 'k10']
	for k in (6, 8, 10):
		target = targets[f'k{k}']
		expected = np.load(paths[k]['posterior'])[1:3, 1:4, 1:5].copy()
		expected[0, 0, 0] = 0.0
		np.testing.assert_array_equal(target['posterior'], expected)
		assert target['posterior'].base is None
		assert target['valid_mask'].dtype == np.bool_
		assert not target['valid_mask'][0, 0, 0]
		assert np.all(target['posterior'][~target['valid_mask']] == 0.0)
	assert len(loaded_memmaps) == 6
	assert all(isinstance(array, np.memmap) for array in loaded_memmaps)
	assert (2, 3, 4, 6) in materialized_memmap_shapes
	assert (4, 5, 6, 6) not in materialized_memmap_shapes


@pytest.mark.parametrize(
	('mutate', 'match'),
	[
		(lambda paths: _make_k8_valid_mask_mismatch(paths), 'valid_tokens must match'),
		(lambda paths: _make_k8_last_dimension_mismatch(paths), 'last dimension'),
		(lambda paths: _make_k8_nonfinite(paths), 'finite and non-negative'),
		(lambda paths: _make_k8_nonunit(paths), 'valid rows must sum to one'),
	],
)
def test_posterior_provider_rejects_malformed_crops(
	tmp_path: Path,
	mutate: object,
	match: str,
) -> None:
	provider, paths = _posterior_provider(tmp_path)
	assert callable(mutate)
	mutate(paths)

	with pytest.raises(ValueError, match=match):
		provider.add_targets({'coords': {}}, _posterior_context(tmp_path))


def test_posterior_dataset_and_dataloader_preserve_soft_only_contract(
	tmp_path: Path,
) -> None:
	posterior_manifest, _ = _posterior_manifest(tmp_path, grid_shape=(2, 2, 2))
	dataset = NopimsStratMultiHeadPosteriorDataset(
		[_dataset_survey_manifest(tmp_path)],
		posterior_manifest,
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		zero_mask=ZeroMaskConfig(enabled=False),
	)

	sample = dataset[0]
	batch = next(
		iter(
			build_strat_multi_head_posterior_dataloader(
				dataset,
				batch_size=1,
				shuffle=False,
				device='cpu',
			)
		)
	)

	assert set(sample) == {'x', 'local_valid_mask', 'strat_multi_posteriors', 'coords'}
	assert batch['strat_multi_posteriors']['k10']['posterior'].shape == (
		1,
		2,
		2,
		2,
		10,
	)
	assert batch['strat_multi_posteriors']['k6']['valid_mask'].all()


def test_posterior_dataset_rejects_crops_without_common_valid_tokens(
	tmp_path: Path,
) -> None:
	posterior_manifest, _ = _posterior_manifest(
		tmp_path,
		grid_shape=(2, 2, 2),
		valid_tokens=False,
	)
	dataset = NopimsStratMultiHeadPosteriorDataset(
		[_dataset_survey_manifest(tmp_path)],
		posterior_manifest,
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		max_resample_attempts=2,
		zero_mask=ZeroMaskConfig(enabled=False),
	)

	with pytest.raises(ValueError, match='at least one common valid token'):
		dataset[0]


def _sample() -> dict[str, object]:
	valid_mask = np.ones((1, 2, 2), dtype=np.bool_)
	return {
		'x': np.ones((1, 2, 2, 2), dtype=np.float32),
		'local_valid_mask': np.ones((2, 2, 2), dtype=np.bool_),
		'strat_multi_posteriors': {
			f'k{k}': {
				'posterior': np.full((1, 2, 2, k), 1.0 / k, dtype=np.float32),
				'valid_mask': valid_mask,
			}
			for k in (6, 8, 10)
		},
		'coords': {'survey_id': 'survey-a'},
	}


def _posterior_provider(
	tmp_path: Path,
) -> tuple[MultiHeadStratPosteriorProvider, dict[int, dict[str, Path]]]:
	manifest, paths = _posterior_manifest(tmp_path)
	return MultiHeadStratPosteriorProvider(manifest), paths


def _posterior_manifest(
	tmp_path: Path,
	*,
	grid_shape: tuple[int, int, int] = (4, 5, 6),
	valid_tokens: bool = True,
) -> tuple[StratMultiHeadPosteriorManifest, dict[int, dict[str, Path]]]:
	paths: dict[int, dict[str, Path]] = {}
	inputs: list[StratMultiHeadPosteriorInput] = []
	for k in (6, 8, 10):
		root = tmp_path / 'posteriors' / f'k{k}'
		root.mkdir(parents=True)
		indices = np.indices(grid_shape).sum(axis=0) % k
		posterior = np.eye(k, dtype=np.float32)[indices]
		mask = np.full(grid_shape, valid_tokens, dtype=np.bool_)
		if not valid_tokens:
			posterior.fill(0.0)
		paths[k] = {}
		for name, array in {'posterior': posterior, 'valid_tokens': mask}.items():
			path = root / f'{name}.npy'
			np.save(path, array)
			paths[k][name] = path
		metadata_path = root / 'metadata.json'
		metadata_path.write_text(
			json.dumps({'artifact_type': 'strat_hmm_state_posterior', 'k': k}),
			encoding='utf-8',
		)
		paths[k]['metadata'] = metadata_path
		inputs.append(
			StratMultiHeadPosteriorInput(
				k=k,
				survey_id='survey',
				posterior_path=paths[k]['posterior'],
				valid_tokens_path=paths[k]['valid_tokens'],
				metadata_path=metadata_path,
				hashes={
					name: sha256(path.read_bytes()).hexdigest()
					for name, path in paths[k].items()
				},
			)
		)
	return (
		StratMultiHeadPosteriorManifest(
			head_ks=(6, 8, 10),
			by_survey={'survey': tuple(inputs)},
		),
		paths,
	)


def _posterior_context(
	tmp_path: Path,
	*,
	token_valid_mask: np.ndarray | None = None,
) -> TargetProviderContext:
	return TargetProviderContext(
		manifest=_context_survey_manifest(tmp_path),
		crop_request=CropRequest('survey', (2, 2, 2), (4, 6, 8)),
		patch_size_xyz=(2, 2, 2),
		token_start_xyz=(1, 1, 1),
		token_size_xyz=(2, 3, 4),
		token_valid_mask=(
			np.ones((2, 3, 4), dtype=np.bool_)
			if token_valid_mask is None
			else token_valid_mask
		),
	)


def _context_survey_manifest(tmp_path: Path) -> SurveyManifest:
	return SurveyManifest(
		survey_id='survey',
		root=tmp_path,
		amplitude=AmplitudeVolumeRecord(
			survey_id='survey',
			path=tmp_path / 'amplitude.npy',
			shape_xyz=(8, 10, 12),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=tmp_path / 'stats.json',
		),
	)


def _dataset_survey_manifest(tmp_path: Path) -> SurveyManifest:
	volume_path = tmp_path / 'amplitude.npy'
	np.save(volume_path, np.ones((4, 4, 4), dtype=np.float32))
	stats_path = tmp_path / 'stats.json'
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id='survey',
			source_path=volume_path,
			grid_order=GRID_ORDER_XYZ,
			clip_low_percentile=0.0,
			clip_high_percentile=100.0,
			clip_low=-1.0,
			clip_high=1.0,
			median=0.0,
			iqr=1.0,
		),
		stats_path,
	)
	return SurveyManifest(
		survey_id='survey',
		root=tmp_path,
		amplitude=AmplitudeVolumeRecord(
			survey_id='survey',
			path=volume_path,
			shape_xyz=(4, 4, 4),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=stats_path,
		),
	)


def _make_k8_valid_mask_mismatch(paths: dict[int, dict[str, Path]]) -> None:
	valid_mask = np.load(paths[8]['valid_tokens'])
	posterior = np.load(paths[8]['posterior'])
	valid_mask[1, 1, 1] = False
	posterior[1, 1, 1] = 0.0
	np.save(paths[8]['valid_tokens'], valid_mask)
	np.save(paths[8]['posterior'], posterior)


def _make_k8_last_dimension_mismatch(paths: dict[int, dict[str, Path]]) -> None:
	posterior = np.load(paths[8]['posterior'])
	np.save(paths[8]['posterior'], np.concatenate((posterior, posterior[..., :1]), -1))


def _make_k8_nonfinite(paths: dict[int, dict[str, Path]]) -> None:
	posterior = np.load(paths[8]['posterior'])
	posterior[1, 1, 1] = np.nan
	np.save(paths[8]['posterior'], posterior)


def _make_k8_nonunit(paths: dict[int, dict[str, Path]]) -> None:
	posterior = np.load(paths[8]['posterior'])
	posterior[1, 1, 1] = 0.0
	posterior[1, 1, 1, 0] = 0.5
	np.save(paths[8]['posterior'], posterior)
