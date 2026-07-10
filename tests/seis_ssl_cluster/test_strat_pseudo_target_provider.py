from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	CropRequest,
	StratPseudoTargetProvider,
	SurveyManifest,
	TargetProviderContext,
)
from seis_ssl_cluster.stratigraphy.targets import (
	StratPseudoTargetInput,
	write_pseudo_target,
)


def test_provider_adds_token_slice_and_coords(tmp_path: Path) -> None:
	labels = np.arange(4 * 4 * 4, dtype=np.int32).reshape(4, 4, 4)
	provider = _provider(tmp_path, labels=labels)
	sample: dict[str, object] = {'coords': {'survey_id': 's1'}}
	context = _context(tmp_path, token_start_xyz=(1, 1, 1))

	provider.add_targets(sample, context)

	np.testing.assert_array_equal(sample['strat_labels'], labels[1:3, 1:3, 1:3])
	assert sample['strat_labels'].dtype == np.int64
	assert sample['strat_confidence'].dtype == np.float32
	assert sample['strat_valid_mask'].dtype == np.bool_
	assert sample['coords']['token_start_xyz'] == (1, 1, 1)
	assert sample['coords']['token_size_xyz'] == (2, 2, 2)


def test_provider_excludes_context_invalid_tokens(tmp_path: Path) -> None:
	labels = np.zeros((2, 2, 2), dtype=np.int32)
	token_valid = np.ones(labels.shape, dtype=bool)
	token_valid[0, 1, 0] = False
	provider = _provider(tmp_path, labels=labels)
	sample: dict[str, object] = {'coords': {}}

	provider.add_targets(
		sample,
		_context(tmp_path, token_valid_mask=token_valid),
	)

	assert sample['strat_labels'][0, 1, 0] == -1
	assert sample['strat_confidence'][0, 1, 0] == 0.0
	assert not sample['strat_valid_mask'][0, 1, 0]
	assert np.count_nonzero(sample['strat_valid_mask']) == 7


def test_provider_rejects_tokens_below_min_confidence(tmp_path: Path) -> None:
	labels = np.zeros((2, 2, 2), dtype=np.int32)
	provider = _provider(
		tmp_path,
		labels=labels,
		confidence=np.full(labels.shape, 0.25, dtype=np.float32),
		min_confidence=0.5,
	)
	sample: dict[str, object] = {'coords': {}}
	provider.add_targets(sample, _context(tmp_path))

	assert provider.sample_is_acceptable(sample) is False


def test_provider_validate_manifests_rejects_missing_survey(
	tmp_path: Path,
) -> None:
	provider = _provider(
		tmp_path,
		labels=np.zeros((2, 2, 2), dtype=np.int32),
	)

	with pytest.raises(ValueError, match=r'missing pseudo-target inputs.*s2'):
		_validate(provider, [_manifest(tmp_path, survey_id='s2')])


def test_provider_rejects_duplicate_pseudo_target_inputs(tmp_path: Path) -> None:
	item = _pseudo_target_input(
		tmp_path,
		labels=np.zeros((2, 2, 2), dtype=np.int32),
	)

	with pytest.raises(ValueError, match='duplicate pseudo-target input'):
		StratPseudoTargetProvider([item, item])


def test_provider_rejects_empty_pseudo_target_inputs() -> None:
	with pytest.raises(ValueError, match='at least one survey'):
		StratPseudoTargetProvider([])


def test_provider_validate_manifests_rejects_small_grid(tmp_path: Path) -> None:
	provider = _provider(
		tmp_path,
		labels=np.zeros((2, 2, 2), dtype=np.int32),
	)

	with pytest.raises(ValueError, match='pseudo-target grid is too small'):
		_validate(provider, [_manifest(tmp_path, shape_xyz=(6, 6, 6))])


def _provider(
	tmp_path: Path,
	*,
	labels: np.ndarray,
	confidence: np.ndarray | None = None,
	min_confidence: float = 0.0,
) -> StratPseudoTargetProvider:
	return StratPseudoTargetProvider(
		[
			_pseudo_target_input(
				tmp_path,
				labels=labels,
				confidence=confidence,
			)
		],
		min_confidence=min_confidence,
	)


def _pseudo_target_input(
	tmp_path: Path,
	*,
	labels: np.ndarray,
	confidence: np.ndarray | None = None,
) -> StratPseudoTargetInput:
	if confidence is None:
		confidence = np.ones(labels.shape, dtype=np.float32)
	k = max(1, int(labels.max(initial=0)) + 1)
	paths = write_pseudo_target(
		tmp_path / 'pseudo',
		k=k,
		survey_id='s1',
		labels=labels,
		confidence=confidence,
		valid_tokens=np.ones(labels.shape, dtype=bool),
	)
	return StratPseudoTargetInput(
		survey_id='s1',
		k=k,
		labels_path=paths.labels,
		confidence_path=paths.confidence,
		valid_tokens_path=paths.valid_tokens,
		metadata_path=paths.metadata,
	)


def _context(
	tmp_path: Path,
	*,
	token_start_xyz: tuple[int, int, int] = (0, 0, 0),
	token_valid_mask: np.ndarray | None = None,
) -> TargetProviderContext:
	if token_valid_mask is None:
		token_valid_mask = np.ones((2, 2, 2), dtype=bool)
	manifest = _manifest(tmp_path)
	return TargetProviderContext(
		manifest=manifest,
		crop_request=CropRequest('s1', (0, 0, 0), (4, 4, 4)),
		patch_size_xyz=(2, 2, 2),
		token_start_xyz=token_start_xyz,
		token_size_xyz=(2, 2, 2),
		token_valid_mask=token_valid_mask,
	)


def _validate(
	provider: StratPseudoTargetProvider,
	manifests: list[SurveyManifest],
) -> None:
	provider.validate_manifests(
		manifests,
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		token_grid_shape_xyz=(2, 2, 2),
	)


def _manifest(
	tmp_path: Path,
	*,
	survey_id: str = 's1',
	shape_xyz: tuple[int, int, int] = (4, 4, 4),
) -> SurveyManifest:
	return SurveyManifest(
		survey_id=survey_id,
		root=tmp_path,
		amplitude=AmplitudeVolumeRecord(
			survey_id=survey_id,
			path=Path('amplitude.npy'),
			shape_xyz=shape_xyz,
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=Path('stats.json'),
		),
	)
