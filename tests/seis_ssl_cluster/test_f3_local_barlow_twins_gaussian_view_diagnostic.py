"""Tests for the experiment-local F3 Gaussian-view diagnostic."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	SurveyManifest,
	SurveyNormalizationStats,
	write_manifest_json,
	write_normalization_stats,
)

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping

DIAGNOSTIC = Path(
	'experiments/f3/facies_benchmark_v2/'
	'111_local_barlow_twins_gaussian_view_v1/diagnose_views.py'
)


@pytest.fixture(scope='module')
def diagnostic_namespace() -> dict[str, object]:
	"""Load the standalone experiment script without running its CLI."""
	return runpy.run_path(str(DIAGNOSTIC))


def test_tiny_view_diagnostic_is_deterministic_and_preserves_contracts(
	tmp_path: Path,
	diagnostic_namespace: dict[str, object],
) -> None:
	config_path = tmp_path / 'control.yaml'
	config_path.write_text('fixture: true\n', encoding='utf-8')
	config = _tiny_control_config(tmp_path)
	diagnose = cast(
		'Callable[..., dict[str, object]]',
		diagnostic_namespace['diagnose_views'],
	)

	first = diagnose(
		config,
		config_path=config_path,
		epoch=0,
		start_index=0,
		count=2,
	)
	second = diagnose(
		config,
		config_path=config_path,
		epoch=0,
		start_index=0,
		count=2,
	)

	assert first == second
	metrics = cast(
		'Mapping[str, Mapping[str, Mapping[str, float | int]]]',
		first['metrics'],
	)
	legacy = metrics['legacy']['all_valid_physical_voxels']
	std005 = metrics['gaussian_noise_std005']['all_valid_physical_voxels']
	std010 = metrics['gaussian_noise_std010']['all_valid_physical_voxels']
	assert legacy['voxel_count'] == 126
	assert legacy['paired_correlation'] == pytest.approx(1.0)
	assert legacy['paired_rms'] == 0.0
	assert std005['paired_correlation'] > std010['paired_correlation']
	assert std010['paired_rms'] == pytest.approx(
		2.0 * std005['paired_rms'],
		abs=1.0e-6,
	)
	assert std010['per_view_rms_from_unaugmented'] == pytest.approx(
		2.0 * std005['per_view_rms_from_unaugmented'],
		abs=1.0e-6,
	)
	assert (
		metrics['legacy']['sampled_pair_token_voxels']['voxel_count'] == 64
	)

	integrity = cast('Mapping[str, object]', first['integrity'])
	assert integrity['invalid_canonical_voxel_count'] == 2
	for key in (
		'legacy_aligned_view_mismatched_voxels',
		'aligned_mask_mismatched_views',
		'flip_state_mismatched_views',
		'pair_index_mismatched_views',
		'physical_pair_mismatched_views',
		'coordinate_mismatched_samples',
		'sampled_pair_invalid_voxels',
	):
		assert integrity[key] == 0
	invalid_nonzero = cast(
		'Mapping[str, int]',
		integrity['invalid_nonzero_values_across_both_views'],
	)
	invalid_mismatches = cast(
		'Mapping[str, int]',
		integrity['invalid_value_mismatches_vs_legacy'],
	)
	assert set(invalid_nonzero.values()) == {0}
	assert set(invalid_mismatches.values()) == {0}
	assert float(
		integrity['max_abs_noise_scaling_residual_std010_minus_2x_std005']
	) < 1.0e-6


def test_cli_defaults_and_explicit_output_only(
	tmp_path: Path,
	capsys: pytest.CaptureFixture[str],
	diagnostic_namespace: dict[str, object],
) -> None:
	build_parser = cast(
		'Callable[[], object]', diagnostic_namespace['_build_parser']
	)
	emit_report = cast(
		'Callable[[Mapping[str, object], Path | None], None]',
		diagnostic_namespace['_emit_report'],
	)
	args = build_parser().parse_args([])
	assert args.epoch == 0
	assert args.start_index == 0
	assert args.count == 16
	assert args.output is None

	payload = {'schema_version': 1, 'value': 0.125}
	emit_report(payload, None)
	serialized = json.dumps(payload, indent=2, sort_keys=True) + '\n'
	assert capsys.readouterr().out == serialized
	assert not list(tmp_path.iterdir())

	output = tmp_path / 'explicit/view_metrics.json'
	emit_report(payload, output)
	assert capsys.readouterr().out == serialized
	assert output.read_text(encoding='utf-8') == serialized
	with pytest.raises(FileExistsError):
		emit_report(payload, output)


def test_readme_documents_stdout_only_diagnostic() -> None:
	readme = (DIAGNOSTIC.parent / 'README.md').read_text(encoding='utf-8')

	assert 'diagnose_views.py' in readme
	assert 'writes nothing unless `--output` is supplied' in readme


def _tiny_control_config(tmp_path: Path) -> dict[str, object]:
	volume_path = tmp_path / 'survey/amplitude.npy'
	volume_path.parent.mkdir(parents=True)
	volume = np.arange(1, 65, dtype=np.float32).reshape(4, 4, 4)
	np.save(volume_path, volume)
	valid_path = tmp_path / 'survey/valid.npy'
	valid = np.ones(volume.shape, dtype=bool)
	valid[0, 0, 0] = False
	np.save(valid_path, valid)
	stats_path = tmp_path / 'normalization.json'
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id='tiny_f3',
			source_path=volume_path,
			grid_order=GRID_ORDER_XYZ,
			clip_low_percentile=0.0,
			clip_high_percentile=100.0,
			clip_low=-100.0,
			clip_high=100.0,
			median=0.0,
			iqr=64.0,
		),
		stats_path,
	)
	manifest_path = tmp_path / 'manifest.json'
	write_manifest_json(
		[
			SurveyManifest(
				survey_id='tiny_f3',
				root=tmp_path,
				amplitude=AmplitudeVolumeRecord(
					survey_id='tiny_f3',
					path=volume_path,
					shape_xyz=volume.shape,
					dtype='float32',
					grid_order=GRID_ORDER_XYZ,
					normalization_stats_path=stats_path,
					valid_mask_path=valid_path,
				),
			)
		],
		manifest_path,
	)
	return {
		'manifests': {'train': str(manifest_path)},
		'data': {
			'local_crop_size': [4, 4, 4],
			'min_valid_fraction': 0.1,
			'max_resample_attempts': 1,
			'normalized_clip_abs': 100.0,
			'amplitude_agc': {'enabled': False},
			'finite_check_mode': 'strict',
		},
		'zero_mask': {
			'enabled': False,
			'zero_atol': 0.0,
			'z_sample_influence_radius': 16,
			'xy_trace_influence_radius': 1,
		},
		'model': {'patch_size': [2, 2, 2]},
		'augmentations': {'horizontal_flip_probability': 0.5},
		'barlow_twins': {
			'method': 'local_barlow_twins_3d',
			'local_pairs_per_crop': 4,
		},
		'train': {'seed': 42, 'samples_per_epoch': 100},
	}
