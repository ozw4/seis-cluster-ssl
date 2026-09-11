"""Inference geometry and paper output contracts without training or scoring."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from seis_ssl_cluster.volve import paper_prediction as prediction
from seis_ssl_cluster.volve import paper_sections as paper
from seis_ssl_cluster.volve.horizon_metrics import soft_argmax_global_sample
from seis_ssl_cluster.volve.horizon_tiles import HorizonTileSettings

CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments/volve/horizon_benchmark_v1/34_hmm_v1_paper_sections_v1'
	/ '01_visualize_sections.yaml'
)


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
	"""Resolve all paths into test-owned directories."""
	for key in (
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'SEIS_SSL_CLUSTER_WORKSPACE',
		'SEIS_SSL_CLUSTER_VOLVE_ROOT',
	):
		monkeypatch.setenv(key, str(tmp_path))
	return prediction.load_config(CONFIG)


def test_fixed_conditions(config: dict[str, Any]) -> None:
	"""Only the six fixed conditions and the common weight are supported."""
	assert tuple(m['condition_id'] for m in config['models']) == prediction.CONDITIONS
	assert config['comparison']['fixed_distillation_weight'] == 0.2
	config['comparison']['fixed_distillation_weight'] = 0.1
	with pytest.raises(ValueError, match='fixed comparison'):
		prediction.validate_config(config)


def test_missing_environment(monkeypatch: pytest.MonkeyPatch) -> None:
	"""A missing root is named explicitly instead of searching the current directory."""
	monkeypatch.delenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', raising=False)
	with pytest.raises(ValueError, match='SEIS_SSL_CLUSTER_ARTIFACT_ROOT'):
		prediction.load_config(CONFIG)


def test_exact_once_tiles() -> None:
	"""The 84 cores cover the survey exactly once, including both partial edges."""
	settings = HorizonTileSettings((401, 720), 1.0)
	counts = np.zeros(settings.lateral_shape_xy, dtype=np.int8)
	records = prediction.tile_records(settings)
	assert len(records) == 84
	for record in records:
		counts[prediction.output_slices(record, settings.lateral_shape_xy)] += 1
	assert np.all(counts == 1)


def test_prediction_validity() -> None:
	"""NaN is confined to invalid traces; no invalid prediction is a valid horizon."""
	valid = np.array([[True, False]])
	values = np.full((5, 1, 2), 600.0)
	values[:, :, 1] = np.nan
	prediction.validate_prediction(values, valid)
	values[0, 0, 0] = 768
	with pytest.raises(ValueError, match='outside'):
		prediction.validate_prediction(values, valid)
	values[0, 0, 0] = np.nan
	with pytest.raises(ValueError, match='finite'):
		prediction.validate_prediction(values, valid)


def test_native_soft_argmax() -> None:
	"""Inference uses the trained fractional-sample expectation, not hard argmax."""
	logits = torch.zeros((1, 5, 2, 2, 216))
	actual = soft_argmax_global_sample(logits)
	torch.testing.assert_close(actual, torch.full((1, 5, 2, 2), 659.5))


@pytest.mark.parametrize(
	('fraction', 'expected'), [(0.96, 0.95), (0.92, 0.90), (0.82, 0.80)]
)
def test_selection(fraction: float, expected: float) -> None:
	"""Fixed quantiles select unique deterministic positions with ordered relaxation."""
	rows = [
		{
			'actual_coordinate': i + 9961,
			'array_index': i,
			'common_valid_fraction': fraction,
			'ground_truth_horizon_count': 5,
		}
		for i in range(20)
	]
	first, threshold = paper.choose_sections(rows, np.arange(9961, 9981))
	second, _ = paper.choose_sections(rows, np.arange(9961, 9981))
	assert first == second
	assert len({r['array_index'] for r in first}) == 10
	assert threshold == expected
	with pytest.raises(ValueError, match='fewer than ten'):
		paper.choose_sections(rows[:9], np.arange(9961, 9981))


def synthetic(config: dict[str, Any]) -> dict[str, Any]:
	"""Provide small lateral geometry with the production time window."""
	data = SimpleNamespace(
		sample_float=np.broadcast_to(
			np.arange(580, 730, 30)[:, None, None], (5, 20, 20)
		),
		bound_valid_mask=np.ones((5, 20, 20), dtype=bool),
		time_ms=np.arange(1, 851) * 4,
	)
	return {
		'config': config,
		'data': data,
		'models': config['models'],
		'seismic': np.zeros((20, 20, 850)),
		'common_valid': np.ones((20, 20), dtype=bool),
		'coords': {
			'inline': np.arange(9961, 9981),
			'crossline': np.arange(1961, 1981),
			'time': data.time_ms / 1000,
		},
		'manifest': {
			'display_crops': {'inline': [0, 20], 'crossline': [0, 20]},
			'amplitude': {'vmin': -4.0, 'vmax': 4.0},
			'active_training_lines': {'inline': [9965], 'crossline': [1965]},
			'adopted_threshold_by_domain': {},
		},
	}


@pytest.mark.parametrize('domain', ['inline', 'crossline'])
def test_layout_axes_and_active_exclusion(config: dict[str, Any], domain: str) -> None:
	"""Eight panels share real coordinates, colors and tick-free axes."""
	prepared = synthetic(config)
	rows = paper.select_domain(prepared, domain)
	assert all(
		r['actual_coordinate']
		not in prepared['manifest']['active_training_lines'][domain]
		for r in rows
	)
	values = prepared['data'].sample_float
	fig = paper.render_figure(prepared, rows[0], [values] * 6)
	try:
		assert len(fig.axes) == 8
		assert [a.get_title() for a in fig.axes] == [
			'Seismic amplitude',
			'Ground truth',
			*[m['display_name'] for m in config['models']],
		]
		assert len(fig.axes[0].lines) == 0
		assert fig.axes[0].images[0].get_cmap().name == 'seismic'
		assert fig.axes[0].images[0].get_clim() == (-4.0, 4.0)
		assert fig.axes[0].get_ylim() == pytest.approx((3.072, 2.212))
		assert fig.axes[4].get_xlabel() == ('xline' if domain == 'inline' else 'inline')
		assert fig.axes[0].get_ylabel() == 'time (s)'
		for ax in fig.axes[1:]:
			assert len(ax.lines) == 5
			assert [line.get_color() for line in ax.lines] == config['figure'][
				'horizon_colors'
			]
			assert ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == 0
			assert ax.get_subplotspec().get_gridspec().get_geometry() == (2, 4)
	finally:
		plt.close(fig)


def test_horizon_extraction_and_crop() -> None:
	"""Horizon ordering and spatial orientation are never changed or repaired."""
	volume = np.arange(5 * 6 * 7).reshape(5, 6, 7)
	np.testing.assert_array_equal(
		paper.horizon_section(volume, 'inline', 2), volume[:, 2, :]
	)
	np.testing.assert_array_equal(
		paper.horizon_section(volume, 'crossline', 2), volume[:, :, 2]
	)
	assert paper.longest_valid_interval(
		np.array([False, True, True, False, True, True])
	) == (1, 3)
	with pytest.raises(ValueError, match='no common'):
		paper.longest_valid_interval(np.zeros(5, dtype=bool))


def test_twenty_outputs(config: dict[str, Any], tmp_path: Path) -> None:
	"""Require exactly twenty readable images and twenty CSV rows."""
	prepared = synthetic(config)
	rows = [
		r for d in ('inline', 'crossline') for r in paper.select_domain(prepared, d)
	]
	for row in rows:
		path = tmp_path / row['output_png']
		path.parent.mkdir(exist_ok=True)
		fig = paper.render_figure(prepared, row, [prepared['data'].sample_float] * 6)
		try:
			fig.savefig(path, dpi=30)
		finally:
			plt.close(fig)
	paper.write_selected_csv(tmp_path / 'selected_slices.csv', rows)
	result = paper.verify_outputs(tmp_path, rows)
	assert result['total_pngs'] == 20
	assert result['counts'] == {'inline': 10, 'crossline': 10}
	(tmp_path / 'extra.png').write_bytes(b'forbidden')
	with pytest.raises(ValueError, match='count mismatch'):
		paper.verify_outputs(tmp_path, rows)


def test_overwrite_refused(
	config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Existing figures are rejected before source preflight or inference."""
	root = Path(config['outputs']['figures'])
	root.mkdir(parents=True)
	(root / 'manifest.json').write_text('{}')
	monkeypatch.setattr(paper, 'load_config', lambda _: config)
	with pytest.raises(FileExistsError):
		paper.run(CONFIG)


def test_dry_run_no_outputs(
	config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Dry-run returns serializable plans without creating output roots."""
	monkeypatch.setattr(paper, 'load_config', lambda _: config)
	monkeypatch.setattr(paper, 'prepare', lambda _: {'manifest': {'planned_pngs': 20}})
	assert paper.run(CONFIG, dry_run=True) == {'planned_pngs': 20}
	assert not Path(config['outputs']['figures']).exists()
	assert not Path(config['outputs']['predictions']).exists()


@pytest.mark.parametrize(('weight', 'k'), [(0.1, 6), (0.2, 5)])
def test_hmm_provenance_rejection(
	weight: float, k: int, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""A plausible model name cannot override the bound distillation provenance."""
	monkeypatch.setattr(prediction, 'require_hash', lambda *_: None)
	monkeypatch.setattr(
		prediction,
		'read_json',
		lambda _: {
			'checkpoint_sha256': 'encoder',
			'stratigraphy_pretext': {
				'distillation_weight': weight,
				'head_num_prototypes': k,
			},
		},
	)
	emb = {
		'embeddings_path': '/test/embeddings.npy',
		'embeddings_sha256': 'array',
		'metadata_path': '/test/metadata.json',
		'metadata_sha256': 'metadata',
		'checkpoint_sha256': 'encoder',
	}
	with pytest.raises(ValueError, match='verified weight'):
		prediction.validate_embedding(emb, {'condition_id': '2b', 'hmm': True})


def test_overwrite_owned_files_only(config: dict[str, Any]) -> None:
	"""Explicit replacement never accepts unexpected files in the output root."""
	root = Path(config['outputs']['figures'])
	root.mkdir(parents=True)
	(root / 'manifest.json').write_text(
		json.dumps(
			{
				'artifact_type': 'volve_hmm_v1_paper_sections',
				'selected_slices': [],
			}
		)
	)
	(root / 'selected_slices.csv').write_text('')
	paper.check_output_root(config, overwrite=True)
	(root / 'unrelated.txt').write_text('preserve')
	with pytest.raises(ValueError, match='unexpected files'):
		paper.check_output_root(config, overwrite=True)
	assert (root / 'unrelated.txt').read_text() == 'preserve'
