from __future__ import annotations

from pathlib import Path

import numpy as np

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	CropRequest,
	NoTargetProvider,
	SurveyManifest,
	TargetProviderContext,
)


def test_target_provider_context_holds_expected_fields(tmp_path: Path) -> None:
	manifest = _manifest(tmp_path)
	crop_request = CropRequest(
		survey_id=manifest.survey_id,
		start_xyz=(8, 16, 24),
		size_xyz=(32, 32, 32),
	)
	token_valid_mask = np.ones((4, 4, 4), dtype=bool)

	context = TargetProviderContext(
		manifest=manifest,
		crop_request=crop_request,
		patch_size_xyz=(8, 8, 8),
		token_start_xyz=(1, 2, 3),
		token_size_xyz=(4, 4, 4),
		token_valid_mask=token_valid_mask,
	)

	assert context.manifest is manifest
	assert context.crop_request is crop_request
	assert context.patch_size_xyz == (8, 8, 8)
	assert context.token_start_xyz == (1, 2, 3)
	assert context.token_size_xyz == (4, 4, 4)
	assert context.token_valid_mask is token_valid_mask


def test_no_target_provider_add_targets_leaves_sample_unchanged(
	tmp_path: Path,
) -> None:
	provider = NoTargetProvider()
	sample: dict[str, object] = {'x': np.ones((1, 2, 2, 2)), 'survey_id': 's1'}
	original_keys = tuple(sample)

	provider.add_targets(sample, _context(tmp_path))

	assert tuple(sample) == original_keys


def test_no_target_provider_accepts_every_sample() -> None:
	assert NoTargetProvider().sample_is_acceptable({}) is True


def test_no_target_provider_rejection_message_is_non_empty() -> None:
	message = NoTargetProvider().rejection_message(
		survey_id='s1',
		max_resample_attempts=5,
		last_valid_fraction=0.25,
	)

	assert message


def test_target_provider_context_defers_mask_validation(tmp_path: Path) -> None:
	# The context is a transport object; concrete providers validate mask inputs.
	context = TargetProviderContext(
		manifest=_manifest(tmp_path),
		crop_request=CropRequest('s1', (0, 0, 0), (8, 8, 8)),
		patch_size_xyz=(8, 8, 8),
		token_start_xyz=(0, 0, 0),
		token_size_xyz=(1, 1, 1),
		token_valid_mask=[[True]],  # type: ignore[arg-type]
	)

	assert context.token_valid_mask == [[True]]


def _context(tmp_path: Path) -> TargetProviderContext:
	return TargetProviderContext(
		manifest=_manifest(tmp_path),
		crop_request=CropRequest('s1', (0, 0, 0), (8, 8, 8)),
		patch_size_xyz=(8, 8, 8),
		token_start_xyz=(0, 0, 0),
		token_size_xyz=(1, 1, 1),
		token_valid_mask=np.ones((1, 1, 1), dtype=bool),
	)


def _manifest(tmp_path: Path) -> SurveyManifest:
	return SurveyManifest(
		survey_id='s1',
		root=tmp_path,
		amplitude=AmplitudeVolumeRecord(
			survey_id='s1',
			path=Path('amplitude.npy'),
			shape_xyz=(32, 32, 32),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=tmp_path / 'stats.json',
		),
	)
