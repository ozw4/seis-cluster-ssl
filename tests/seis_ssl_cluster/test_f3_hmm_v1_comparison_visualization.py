from __future__ import annotations

import copy
import csv
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.figure import Figure

mpl.use('Agg')
import matplotlib.pyplot as plt

from seis_ssl_cluster.f3.lithology import hmm_v1_comparison_visualization as paper

REPO = Path(__file__).resolve().parents[2]
CONFIG = (
	REPO / 'experiments/f3/facies_benchmark_v2/127_hmm_v1_paper_sections_v1/'
	'01_visualize_large_layout_000.yaml'
)


@pytest.fixture
def config(monkeypatch, tmp_path):
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(REPO))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3'))
	return paper.load_config(CONFIG)


@pytest.fixture
def prepared(config):
	shape = (24, 26, 28)
	labels = (np.indices(shape).sum(axis=0) % 2).astype(np.int16)
	seismic = (
		np.arange(np.prod(shape), dtype=float).reshape(shape) / np.prod(shape) - 0.5
	)
	mask = np.ones(shape, dtype=bool)
	mask[-1, -1, -1] = False
	prediction = labels.copy()
	prediction[~mask] = -1
	arrays = [
		SimpleNamespace(predictions=prediction, valid_mask=mask) for _ in range(6)
	]
	coords = {
		'inline': np.arange(100, 124),
		'crossline': np.arange(300, 326),
		'time': np.arange(28) * 0.005,
	}
	classes = [
		{'class_id': 0, 'class_name': 'Zero', 'rgb': [35, 92, 167]},
		{'class_id': 1, 'class_name': 'One', 'rgb': [254, 219, 124]},
	]
	rows = []
	for domain in paper.DOMAINS:
		candidates = paper.slice_eligibility(
			seismic, labels, [mask] * 6, [0, 1], domain, coords[domain], []
		)
		selected, _ = paper.select_slices(candidates, coords[domain])
		rows.extend({**row, 'output_png': paper.output_png(row)} for row in selected)
	config['figure']['dpi'] = (
		25  # Unit-test output only; production parser rejects this.
	)
	manifest = {
		'amplitude': {'vmin': -0.5, 'vmax': 0.5},
		'outputs': config['outputs'].copy(),
	}
	return paper.PreparedComparison(
		config, seismic, labels, arrays, classes, coords, manifest, rows
	)


def test_condition_mapping(config):
	assert tuple(model['condition_id'] for model in config['models']) == (
		'1',
		'2b',
		'3',
		'4b',
		'5',
		'6b',
	)
	assert config['comparison']['layout_id'] == 'layout_000'
	assert config['comparison']['data_size'] == 'large'
	assert all(
		model['distillation_weight'] == 0.2
		for model in config['models']
		if model['hmm']
	)
	assert not {'2a', '4a', '6a'} & set(paper.CONDITION_ORDER)
	assert config['figure']['dpi'] >= 300


@pytest.mark.parametrize('index', [1, 3, 5])
def test_reject_per_family_weight_selection(config, index):
	config['models'][index]['distillation_weight'] = 0.1
	with pytest.raises(ValueError, match=r'weight=0\.2'):
		paper.validate_config(config)


@pytest.mark.parametrize(
	('field', 'value'), [('layout_id', 'layout_001'), ('data_size', 'small')]
)
def test_reject_alternate_comparison(config, field, value):
	config['comparison'][field] = value
	with pytest.raises(ValueError, match='layout_000 / large'):
		paper.validate_config(config)


def test_missing_environment_is_named(monkeypatch):
	for name in (
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'SEIS_SSL_CLUSTER_WORKSPACE',
		'F3_ROOT',
	):
		monkeypatch.delenv(name, raising=False)
	with pytest.raises(
		ValueError,
		match='SEIS_SSL_CLUSTER_ARTIFACT_ROOT, SEIS_SSL_CLUSTER_WORKSPACE, F3_ROOT',
	):
		paper.load_config(CONFIG)


def test_relative_paths_and_output_escape_rejected(config):
	with pytest.raises(ValueError, match='absolute resolved'):
		paper.absolute_path('relative/prediction')
	config['outputs']['root'] = config['paths']['artifact_root']
	with pytest.raises(ValueError, match='output root'):
		paper.validate_config(config)


@pytest.mark.parametrize('domain', paper.DOMAINS)
def test_axis_extraction_and_actual_coordinates(prepared, domain):
	volume = np.arange(24 * 26 * 28).reshape(24, 26, 28)
	expected = {
		'inline': volume[7, :, :].T,
		'crossline': volume[:, 7, :].T,
		'time': volume[:, :, 7],
	}
	np.testing.assert_array_equal(
		paper.extract_section(volume, domain, 7), expected[domain]
	)
	x, y, xlabel, ylabel = paper.section_axes(prepared.coords, domain)
	np.testing.assert_array_equal(
		x, prepared.coords['inline' if domain == 'crossline' else 'crossline']
	)
	np.testing.assert_array_equal(
		y, prepared.coords['inline' if domain == 'time' else 'time']
	)
	assert xlabel == ('inline' if domain == 'crossline' else 'xline')
	assert ylabel == ('inline' if domain == 'time' else 'time (s)')


@pytest.mark.parametrize(
	('raw', 'unit'),
	[(np.arange(255) * 5.0, 'milliseconds'), (np.arange(255) * 0.005, 'seconds')],
)
def test_time_unit_conversion(raw, unit):
	seconds, report = paper.time_in_seconds(raw, 5000)
	np.testing.assert_allclose(seconds, np.arange(255) * 0.005)
	assert report['detected_sample_unit'] == unit
	assert report['time_step_seconds'] == pytest.approx(0.005)


def test_time_unit_conflict_reports_evidence():
	with pytest.raises(ValueError, match=r'first=.*last=.*differences=.*interval=4000'):
		paper.time_in_seconds(np.arange(10) * 5.0, 4000)


def test_nonmonotone_coordinates_fail():
	with pytest.raises(ValueError, match='strictly monotone'):
		paper.coordinate_step(np.array([1, 3, 2]))


@pytest.mark.parametrize('domain', paper.DOMAINS)
def test_figure_layout_palette_and_orientation(prepared, domain):
	row = next(row for row in prepared.rows if row['domain'] == domain)
	fig = paper.render_figure(prepared, row)
	try:
		assert len(fig.axes) == 8
		assert {
			(ax.get_subplotspec().rowspan.start, ax.get_subplotspec().colspan.start)
			for ax in fig.axes
		} == {(r, c) for r in range(2) for c in range(4)}
		assert [ax.get_title().replace('\n', ' ').strip() for ax in fig.axes] == [
			'Seismic amplitude',
			'Ground truth',
			'MAE',
			'MAE + HMM',
			'LocalBT',
			'LocalBT + HMM',
			'Random',
			'Random + HMM',
		]
		assert len(fig.axes[0].images) == 1
		assert fig.axes[0].images[0].get_cmap().name == 'seismic'
		assert fig.axes[0].images[0].get_clim() == (-0.5, 0.5)
		palette = fig.axes[1].images[0].get_cmap()
		for ax in fig.axes[1:]:
			assert ax.images[0].get_cmap() is palette
			assert ax.images[0].get_interpolation() == 'nearest'
			np.testing.assert_allclose(
				ax.images[0].get_cmap()(0)[:3], np.array([35, 92, 167]) / 255
			)
			np.testing.assert_allclose(
				ax.images[0].get_cmap()(np.ma.masked)[:3],
				np.array([226, 226, 226]) / 255,
			)
		for ax in fig.axes:
			assert ax.get_xlim() == fig.axes[0].get_xlim()
			assert ax.get_ylim() == fig.axes[0].get_ylim()
			assert (ax.get_ylim()[0] > ax.get_ylim()[1]) == (domain != 'time')
		assert fig.axes[4].get_xlabel() == (
			'inline' if domain == 'crossline' else 'xline'
		)
		assert fig.axes[0].get_ylabel() == (
			'inline' if domain == 'time' else 'time (s)'
		)
		fig.canvas.draw()
		assert fig.get_size_inches()[0] * 25.4 == pytest.approx(180)
		assert not fig.texts
		for ax in fig.axes:
			for label in (ax.xaxis.label, ax.yaxis.label):
				if label.get_text():
					assert label.get_fontsize() == 9
					assert label.get_fontfamily() == ['Arial']
			assert ax.texts[0].get_fontsize() == 10
			assert ax.texts[0].get_fontweight() == 'bold'
			assert ax.title.get_fontfamily() == ['Arial']
			assert ax.title.get_fontsize() == 9
			assert '\n' not in ax.get_title()
			assert all(
				tick.tick1line.get_markersize() == 0
				for axis in (ax.xaxis, ax.yaxis)
				for tick in axis.get_major_ticks()
			)
			assert all(spine.get_linewidth() == 0.8 for spine in ax.spines.values())
			for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
				assert tick.get_fontsize() == 8
				assert tick.get_fontfamily() == ['Arial']
		for text in fig.legends[0].get_texts():
			assert text.get_fontsize() == 8
			assert text.get_fontfamily() == ['Arial']
		renderer = fig.canvas.get_renderer()
		for start in (0, 4):
			heights = [
				ax.title.get_window_extent(renderer).y0
				for ax in fig.axes[start : start + 4]
			]
			assert max(heights) - min(heights) == pytest.approx(0)
		for ax in fig.axes:
			title_box = ax.title.get_window_extent(renderer)
			panel_box = ax.texts[0].get_window_extent(renderer)
			assert ax.bbox.x0 - panel_box.x1 == pytest.approx(3 * fig.dpi / 72)
			assert (title_box.y0 + title_box.y1) / 2 == pytest.approx(
				(panel_box.y0 + panel_box.y1) / 2
			)
	finally:
		plt.close(fig)


def test_palette_class_zero_unknown_and_invalid():
	image = paper.categorical_image(
		np.array([[0, 1, -1]]), [0, 1], np.array([[True, True, False]])
	)
	assert image[0, 0] == 0
	assert not image.mask[0, 0]
	assert image.mask[0, 2]
	for labels, mask in [
		(np.array([[42]]), None),
		(np.array([[-1]]), None),
		(np.array([[0]]), np.array([[False]])),
	]:
		with pytest.raises(ValueError, match=r'class ID|prediction voxels'):
			paper.categorical_image(labels, [0, 1], mask)


@pytest.mark.parametrize('domain', ['inline', 'crossline'])
def test_selection_excludes_training_and_is_deterministic(prepared, domain):
	coordinates = prepared.coords[domain]
	active = [int(coordinates[8]), int(coordinates[13])]
	rows = paper.slice_eligibility(
		prepared.seismic,
		prepared.labels,
		[a.valid_mask for a in prepared.arrays],
		[0, 1],
		domain,
		coordinates,
		active,
	)
	assert not set(active) & {row['actual_coordinate'] for row in rows}
	selected, threshold = paper.select_slices(rows, coordinates)
	assert paper.select_slices(rows, coordinates) == (selected, threshold)
	assert len({row['array_index'] for row in selected}) == 10
	assert all(row['active_training_line'] is False for row in selected)
	assert all(row['ground_truth_class_ids'] == [0, 1] for row in rows)
	assert set(inspect.signature(paper.select_slices).parameters) == {
		'rows',
		'coordinates',
	}


@pytest.mark.parametrize(
	('fraction', 'threshold'), [(0.99, 0.95), (0.93, 0.90), (0.85, 0.80)]
)
def test_threshold_relaxation(prepared, fraction, threshold):
	rows = copy.deepcopy(prepared.rows[:10])
	for row in rows:
		row['common_valid_fraction'] = fraction
	assert paper.select_slices(rows, prepared.coords['inline'])[1] == threshold


def test_insufficient_candidates_fail_with_diagnostics(prepared):
	with pytest.raises(ValueError, match=r'candidates=9.*min/median/max'):
		paper.select_slices(prepared.rows[:9], prepared.coords['inline'])


def test_selection_tie_uses_small_array_index(prepared):
	rows = copy.deepcopy(prepared.rows[:10])
	for index, row in enumerate(rows):
		row['array_index'], row['actual_coordinate'] = index, float(index + 10)
	coordinates = np.array([10.0, 20.0])  # q=.05 target=10.5
	assert paper.select_slices(rows, coordinates)[0][0]['array_index'] == 0


def test_common_fraction_uses_intersection_and_finite_denominator(prepared):
	mask_a = np.ones_like(prepared.labels, dtype=bool)
	mask_b = mask_a.copy()
	mask_a[:, :2, :] = False
	mask_b[:, 2:4, :] = False
	rows = paper.slice_eligibility(
		prepared.seismic,
		prepared.labels,
		[mask_a, mask_b] * 3,
		[0, 1],
		'inline',
		prepared.coords['inline'],
		[],
	)
	assert rows[0]['common_valid_fraction'] == pytest.approx(22 / 26)


def test_nonuniform_coordinates_use_pcolormesh(prepared):
	prepared.coords['crossline'] = prepared.coords['crossline'].astype(float)
	prepared.coords['crossline'][5] += 0.25
	assert paper.drawing_method(prepared.coords, 'inline') == 'pcolormesh'
	fig = paper.render_figure(prepared, prepared.rows[0])
	try:
		assert all(len(ax.images) == 0 and len(ax.collections) == 1 for ax in fig.axes)
	finally:
		plt.close(fig)


def test_exact_output_set_and_overwrite_refusal(prepared):
	manifest = paper.write_outputs(prepared)
	root = Path(prepared.config['outputs']['root'])
	assert manifest['outputs']['actual_counts'] == {
		'inline': 10,
		'crossline': 10,
		'time': 10,
		'total': 30,
	}
	assert len(list(root.rglob('*.png'))) == 30
	assert (root / 'manifest.json').is_file()
	with (root / 'selected_slices.csv').open() as stream:
		assert len(list(csv.DictReader(stream))) == 30
	before = paper.sha256(root / 'manifest.json')
	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		paper.write_outputs(prepared)
	assert paper.sha256(root / 'manifest.json') == before
	assert not plt.get_fignums()
	(root / 'unexpected.png').write_bytes(b'not an allowed output')
	with pytest.raises(ValueError, match='file set mismatch'):
		paper.verify_outputs(root, prepared.rows)


def test_save_failure_closes_figure(prepared, monkeypatch):
	def fail(*_args, **_kwargs):
		raise OSError('save failed')

	monkeypatch.setattr(Figure, 'savefig', fail)
	with pytest.raises(OSError, match='save failed'):
		paper.write_outputs(prepared)
	assert not plt.get_fignums()


def test_dry_run_does_not_write(config, prepared, monkeypatch):
	monkeypatch.setattr(paper, 'prepare_comparison', lambda _: prepared)
	result = paper.run(CONFIG, dry_run=True)
	json.dumps(result, allow_nan=False)
	assert not Path(config['outputs']['root']).exists()


def test_input_validation_before_render(prepared):
	prepared.seismic[0, 0, 0] = np.nan
	with pytest.raises(ValueError, match='nonfinite seismic'):
		paper.validate_input_values(prepared.seismic, prepared.labels, [0, 1])
	prepared.seismic[0, 0, 0] = 0
	prepared.labels[0, 0, 0] = 7
	with pytest.raises(ValueError, match='ground truth'):
		paper.validate_input_values(prepared.seismic, prepared.labels, [0, 1])


def test_amplitude_sampling_is_global_and_deterministic(prepared, tmp_path):
	stats = tmp_path / 'stats.json'
	stats.write_text('{}')
	first = paper.amplitude_scale(
		prepared.seismic, str(stats), prepared.config['figure']
	)
	assert first == paper.amplitude_scale(
		prepared.seismic, str(stats), prepared.config['figure']
	)
	assert first['vmin'] == -first['vmax']
	stats.write_text(
		json.dumps(
			{
				'clip_low_percentile': 1.0,
				'clip_high_percentile': 99.0,
				'clip_low': -2,
				'clip_high': 1,
			}
		)
	)
	assert (
		paper.amplitude_scale(prepared.seismic, str(stats), prepared.config['figure'])[
			'vmax'
		]
		== 2
	)


def test_hmm_provenance_does_not_trust_model_name():
	model = {'hmm': True, 'model_id': 'random_self_target_hmm_k6_distill020'}
	for weight, prototypes in [(0.1, 6), (0.2, 8)]:
		with pytest.raises(ValueError, match='K/weight mismatch'):
			paper.validate_hmm_provenance(
				model,
				{
					'stratigraphy_pretext': {
						'distillation_weight': weight,
						'head_num_prototypes': prototypes,
					}
				},
				Path('/provenance.json'),
			)


def test_shared_validator_is_required(config, monkeypatch):
	def reject(_path, *, mmap_mode):
		assert mmap_mode == 'r'
		raise ValueError('shared validator rejected incomplete artifact')

	monkeypatch.setattr(paper, 'validate_f3_voxel_prediction_artifact', reject)
	with pytest.raises(ValueError, match='shared validator rejected'):
		paper.validate_model(config['models'][0], config, (24, 26, 28), [0, 1])


def test_provenance_hash_mismatch(tmp_path):
	path = tmp_path / 'metadata.json'
	path.write_text('{}')
	with pytest.raises(ValueError, match='SHA-256 mismatch'):
		paper.verified_identity({'path': str(path), 'sha256': 'wrong'})


@pytest.fixture
def model_artifact(config, tmp_path, monkeypatch):
	model = config['models'][1]
	root = Path(model['prediction_dir'])
	root.mkdir(parents=True)
	voxel_path = Path(config['inputs']['section_layout_metadata']).with_name(
		'voxel_dataset_metadata.json'
	)
	voxel_path.parent.mkdir(parents=True)
	decoder_path, embedding_path = (
		tmp_path / 'decoder.json',
		tmp_path / 'embedding.json',
	)
	decoder = {
		'dataset': paper.DATASET,
		'model': {'tag': model['model_id']},
		'voxel_dataset': {'input_dir': str(voxel_path.parent)},
		'embeddings': {'checkpoint_path': '/checkpoint.pt'},
	}
	embedding = {
		'checkpoint_path': '/checkpoint.pt',
		'stratigraphy_pretext': {'distillation_weight': 0.2, 'head_num_prototypes': 6},
	}
	voxel = {
		'dataset': paper.DATASET,
		'section_layout': {'layout_id': 'layout_000', 'data_size': 'large'},
	}
	for path, payload in [
		(decoder_path, decoder),
		(embedding_path, embedding),
		(voxel_path, voxel),
	]:
		path.write_text(json.dumps(payload))
	metadata = {
		'model_tag': model['model_id'],
		'volume_shape_xyz': [2, 3, 4],
		'class_probability_order': [0, 1],
		'coverage': {'exact_once': True},
		'prediction_kind': 'frozen_embedding_decoder',
		'summary': {},
		'inputs': {'embedding_metadata': str(embedding_path)},
		'source_identity': {
			'resolved_decoder_config': {
				'path': str(decoder_path),
				'sha256': paper.sha256(decoder_path),
			},
			'artifact_identities': {
				key: {'path': str(path), 'sha256': paper.sha256(path)}
				for key, path in [
					('embedding_metadata', embedding_path),
					('voxel_dataset_metadata', voxel_path),
				]
			},
		},
	}
	(root / 'prediction_metadata.json').write_text(json.dumps(metadata))
	artifact = SimpleNamespace(metadata=metadata, arrays=SimpleNamespace())
	monkeypatch.setattr(
		paper,
		'validate_f3_voxel_prediction_artifact',
		lambda *_args, **_kwargs: artifact,
	)
	return config, model, artifact


def test_bound_provenance_success(model_artifact):
	config, model, _ = model_artifact
	_, report = paper.validate_model(model, config, (2, 3, 4), [0, 1])
	assert report['distillation_weight'] == 0.2
	assert report['hmm_k'] == 6
	assert report['layout_id'] == 'layout_000'
	assert report['data_size'] == 'large'
	assert report['provenance_path'].endswith('embedding.json')


@pytest.mark.parametrize(
	('field', 'value', 'message'),
	[
		('model_tag', 'random', 'model identity'),
		('volume_shape_xyz', [2, 3, 5], 'volume shape'),
		('class_probability_order', [1, 0], 'class probability order'),
		('coverage', {'exact_once': False}, 'completed full-volume'),
	],
)
def test_model_metadata_mismatch(model_artifact, field, value, message):
	config, model, artifact = model_artifact
	artifact.metadata[field] = value
	with pytest.raises(ValueError, match=message):
		paper.validate_model(model, config, (2, 3, 4), [0, 1])


@pytest.mark.parametrize(
	('field', 'value'), [('layout_id', 'layout_001'), ('data_size', 'small')]
)
def test_bound_supervision_mismatch(model_artifact, field, value):
	config, model, artifact = model_artifact
	identity = artifact.metadata['source_identity']['artifact_identities'][
		'voxel_dataset_metadata'
	]
	path = Path(identity['path'])
	metadata = paper.read_json(path)
	metadata['section_layout'][field] = value
	path.write_text(json.dumps(metadata))
	identity['sha256'] = paper.sha256(path)
	with pytest.raises(ValueError, match='layout/data size'):
		paper.validate_model(model, config, (2, 3, 4), [0, 1])


def test_invalid_borders_are_cropped_identically(prepared):
	for arrays in prepared.arrays:
		arrays.valid_mask[:, :, :2] = False
		arrays.valid_mask[:, :, -2:] = False
		arrays.valid_mask[-1, :, :] = False
		arrays.predictions[~arrays.valid_mask] = -1
	crops = paper.display_crops(prepared)
	prepared.manifest['display_crops_by_domain'] = crops
	assert crops['inline']['y_min'] == 0.01
	assert crops['inline']['y_max'] == 0.125
	assert crops['crossline']['x_max'] == 122
	assert crops['time']['y_max'] == 122
	fig = paper.render_figure(prepared, prepared.rows[0])
	try:
		assert len({ax.images[0].get_array().shape for ax in fig.axes}) == 1
		assert all(
			not np.ma.getmaskarray(ax.images[0].get_array()).any() for ax in fig.axes
		)
		assert 'Invalid prediction' not in [
			text.get_text() for text in fig.legends[0].get_texts()
		]
	finally:
		plt.close(fig)


def test_internal_invalid_hole_is_not_hidden():
	mask = np.ones((20, 30), dtype=bool)
	mask[10, 15] = False
	with pytest.raises(ValueError, match='internal invalid'):
		paper.valid_display_bounds(mask)


def test_partially_invalid_edge_columns_are_trimmed():
	mask = np.ones((20, 30), dtype=bool)
	mask[:2] = False
	mask[-2:] = False
	mask[-4:, -3:] = False
	assert paper.valid_display_bounds(mask) == [2, 18, 0, 27]
