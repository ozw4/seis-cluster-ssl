# ruff: noqa: SLF001
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from proc.seis_ssl_cluster.visualize_clusters import (
	DEFAULT_CONFIG,
	run_cluster_visualization,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.visualization import clusters as cluster_vis
from seis_ssl_cluster.visualization.clusters import (
	ClusterSlice,
	ClusterSliceRequest,
	save_cluster_comparison_pngs,
	save_cluster_slice_pngs,
	stable_cluster_colors,
)
from seis_ssl_cluster.visualization.common import slice_image

if TYPE_CHECKING:
	from pathlib import Path


plt = pytest.importorskip('matplotlib.pyplot')


def test_imshow_view_helpers_match_seismic_display_rules() -> None:
	assert cluster_vis._imshow_origin_for_view('xy') == 'lower'
	assert cluster_vis._imshow_aspect_for_view('xy') == 'equal'
	assert cluster_vis._vertical_axis_label_for_view('xy') == 'y'
	assert cluster_vis._imshow_origin_for_view('xz') == 'upper'
	assert cluster_vis._imshow_aspect_for_view('xz') == 'auto'
	assert cluster_vis._vertical_axis_label_for_view('xz') == 'z (down)'
	assert cluster_vis._figsize_for_view('xy', comparison=False) == (6.0, 6.0)
	assert cluster_vis._figsize_for_view('xz', comparison=False) == (6.0, 8.5)
	assert cluster_vis._figsize_for_view('xy', comparison=True) == (12.0, 4.5)
	assert cluster_vis._figsize_for_view('xz', comparison=True) == (12.0, 8.5)


def test_xz_slice_keeps_shallow_z_first_row_and_uses_upper_origin() -> None:
	volume = np.zeros((3, 2, 4), dtype=np.int32)
	for z_index in range(volume.shape[2]):
		volume[:, :, z_index] = z_index

	image = slice_image(volume, view='xz', slice_index=1)

	assert image.shape == (4, 3)
	assert np.all(image[0, :] == 0)
	assert np.all(image[-1, :] == 3)
	assert cluster_vis._imshow_origin_for_view('xz') == 'upper'


def test_comparison_panels_share_xz_origin_and_aspect(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	fig, ax = plt.subplots()
	axes_type = type(ax)
	plt.close(fig)
	calls = []
	original_imshow = axes_type.imshow

	def recording_imshow(self: object, *args: object, **kwargs: object) -> object:
		calls.append(
			{
				'origin': kwargs.get('origin'),
				'aspect': kwargs.get('aspect'),
				'interpolation': kwargs.get('interpolation'),
			},
		)
		return original_imshow(self, *args, **kwargs)

	monkeypatch.setattr(axes_type, 'imshow', recording_imshow)
	labels = np.arange(3 * 4 * 5, dtype=np.int32).reshape(3, 4, 5) % 3
	amplitude = np.linspace(-1.0, 1.0, labels.size, dtype=np.float32).reshape(
		labels.shape,
	)

	save_cluster_comparison_pngs(
		labels,
		survey_id='survey',
		k=3,
		mode='voxel',
		output_dir=tmp_path,
		slices=ClusterSliceRequest(xz_slices=(1,)),
		amplitude=amplitude,
	)

	assert len(calls) == 4
	assert {call['origin'] for call in calls} == {'upper'}
	assert {call['aspect'] for call in calls} == {'auto'}
	assert calls[1]['interpolation'] == 'nearest'
	assert calls[3]['interpolation'] == 'nearest'


def test_xy_and_xz_cluster_pngs_are_created(tmp_path: Path) -> None:
	labels = np.arange(3 * 4 * 5, dtype=np.int32).reshape(3, 4, 5) % 3
	labels[0, :, :] = -1
	amplitude = np.linspace(-1.0, 1.0, labels.size, dtype=np.float32).reshape(
		labels.shape,
	)

	created = save_cluster_slice_pngs(
		labels,
		survey_id='survey',
		k=3,
		mode='voxel',
		output_dir=tmp_path,
		slices=ClusterSliceRequest(xy_slices=(2,), xz_slices=(1,)),
		amplitude=amplitude,
		amplitude_alpha=0.25,
	)

	assert [path.name for path in created] == [
		'survey_k3_xy_z2.png',
		'survey_k3_xz_y1.png',
	]
	assert all(path.is_file() and path.stat().st_size > 0 for path in created)


def test_xy_and_xz_cluster_comparison_pngs_are_created(tmp_path: Path) -> None:
	labels = np.arange(3 * 4 * 5, dtype=np.int32).reshape(3, 4, 5) % 3
	amplitude = np.linspace(-1.0, 1.0, labels.size, dtype=np.float32).reshape(
		labels.shape,
	)

	created = save_cluster_comparison_pngs(
		labels,
		survey_id='survey',
		k=3,
		mode='voxel',
		output_dir=tmp_path,
		slices=ClusterSliceRequest(xy_slices=(2,), xz_slices=(1,)),
		amplitude=amplitude,
		amplitude_alpha=0.25,
	)

	assert [path.name for path in created] == [
		'survey_k3_xy_z2.png',
		'survey_k3_xz_y1.png',
	]
	assert all(path.is_file() and path.stat().st_size > 0 for path in created)


def test_amplitude_underlay_changes_visible_cluster_pixels(tmp_path: Path) -> None:
	labels = np.zeros((4, 4, 1), dtype=np.int32)
	flat = np.linspace(-1.0, 1.0, labels.size, dtype=np.float32)
	first_amplitude = np.zeros_like(labels, dtype=np.float32)
	second_amplitude = flat.reshape(labels.shape)

	first = save_cluster_slice_pngs(
		labels,
		survey_id='survey',
		k=1,
		mode='voxel',
		output_dir=tmp_path / 'first',
		slices=ClusterSliceRequest(xy_slices=(0,)),
		amplitude=first_amplitude,
		amplitude_alpha=0.65,
	)[0]
	second = save_cluster_slice_pngs(
		labels,
		survey_id='survey',
		k=1,
		mode='voxel',
		output_dir=tmp_path / 'second',
		slices=ClusterSliceRequest(xy_slices=(0,)),
		amplitude=second_amplitude,
		amplitude_alpha=0.65,
	)[0]

	assert not np.array_equal(plt.imread(first), plt.imread(second))


def test_proc_visualization_token_mode_uses_amplitude_underlay(
	tmp_path: Path,
) -> None:
	first = _run_token_underlay_visualization(
		tmp_path / 'first',
		np.zeros((4, 4, 1), dtype=np.float32),
	)
	second = _run_token_underlay_visualization(
		tmp_path / 'second',
		np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(4, 4, 1),
	)

	assert not np.array_equal(plt.imread(first), plt.imread(second))


def test_proc_visualization_token_mode_writes_amplitude_comparison(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'comparison'
	input_dir = root / 'cluster_run'
	output_dir = root / 'figures'
	labels_dir = input_dir / 'labels' / 'k1'
	labels_dir.mkdir(parents=True)
	np.save(
		labels_dir / 'survey.cluster_labels_token.npy',
		np.zeros((2, 2, 1), dtype=np.int32),
	)
	amplitude_path = root / 'survey.npy'
	np.save(
		amplitude_path,
		np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(4, 4, 1),
	)
	(labels_dir / 'survey.cluster_label_metadata.json').write_text(
		json.dumps(
			{
				'source_amplitude_path': str(amplitude_path),
				'patch_size': [2, 2, 1],
			},
		)
		+ '\n',
		encoding='utf-8',
	)

	result = run_cluster_visualization(
		{
			'clustering': {'input_dir': str(input_dir)},
			'visualization': {
				'output_dir': str(output_dir),
				'modes': ['token'],
				'xy_slices': [0],
				'xz_slices': [],
				'summaries': {'enabled': False},
				'amplitude_underlay': {'enabled': False},
				'amplitude_comparison': {'enabled': True, 'alpha': 0.35},
			},
		},
	)

	assert result == {'png_count': 2, 'voxel_count': 0, 'summary_count': 0}
	assert (output_dir / 'token' / 'survey_k1_xy_z0.png').is_file()
	assert (
		output_dir / 'token_comparison' / 'survey_k1_xy_z0.png'
	).is_file()


def test_cluster_colormap_is_stable_for_same_k() -> None:
	first = stable_cluster_colors(4)
	second = stable_cluster_colors(4)

	np.testing.assert_array_equal(first.colors, second.colors)


def test_proc_visualization_writes_token_and_voxel_modes_separately(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'cluster_run'
	output_dir = tmp_path / 'figures'
	labels_dir = input_dir / 'labels' / 'k2'
	labels_dir.mkdir(parents=True)
	token_labels = np.array(
		[
			[[0, 1], [1, -1]],
			[[1, 0], [-1, 0]],
		],
		dtype=np.int32,
	)
	np.save(labels_dir / 'survey.cluster_labels_token.npy', token_labels)
	(labels_dir / 'survey.cluster_label_metadata.json').write_text(
		json.dumps(
			{
				'patch_size': [2, 2, 2],
				'volume_shape_xyz': [4, 4, 4],
			},
		)
		+ '\n',
		encoding='utf-8',
	)

	result = run_cluster_visualization(
		{
			'clustering': {'input_dir': str(input_dir)},
			'visualization': {
				'output_dir': str(output_dir),
				'survey_ids': ['survey'],
				'modes': ['token', 'voxel'],
				'reconstruct_voxel': True,
				'slice_coordinate_space': 'voxel',
				'xy_slices': [3],
				'xz_slices': [2],
				'summaries': {'enabled': False},
				'amplitude_underlay': {'enabled': False},
			},
		},
	)

	assert result['png_count'] == 4
	assert (output_dir / 'token' / 'survey_k2_xy_z3.png').is_file()
	assert (output_dir / 'token' / 'survey_k2_xz_y2.png').is_file()
	assert (output_dir / 'voxel' / 'survey_k2_xy_z3.png').is_file()
	assert (output_dir / 'voxel' / 'survey_k2_xz_y2.png').is_file()


def test_proc_visualization_survey_ids_filter_all_outputs(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'cluster_run'
	output_dir = tmp_path / 'figures'
	labels_dir = input_dir / 'labels' / 'k2'
	_make_cluster_labels(labels_dir, 'keep')
	_make_cluster_labels(labels_dir, 'drop')

	result = run_cluster_visualization(
		{
			'clustering': {'input_dir': str(input_dir)},
			'visualization': {
				'output_dir': str(output_dir),
				'survey_ids': ['keep'],
				'modes': ['token', 'voxel'],
				'reconstruct_voxel': True,
				'xy_slices': [0],
				'xz_slices': [],
				'summaries': {'enabled': True},
				'amplitude_underlay': {'enabled': False},
			},
		},
	)

	assert result == {'png_count': 2, 'voxel_count': 1, 'summary_count': 1}
	assert (output_dir / 'token' / 'keep_k2_xy_z0.png').is_file()
	assert not (output_dir / 'token' / 'drop_k2_xy_z0.png').exists()
	assert (output_dir / 'voxel' / 'keep_k2_xy_z0.png').is_file()
	assert not (labels_dir / 'drop.cluster_labels_voxel.npy').exists()
	payload = json.loads(
		(output_dir / 'k2' / 'cluster_summary.json').read_text(encoding='utf-8'),
	)
	assert payload['survey_ids'] == ['keep']
	assert payload['selected_survey_ids'] == ['keep']


def test_proc_visualization_unknown_survey_id_fails_clearly(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'cluster_run'
	_make_cluster_labels(input_dir / 'labels' / 'k2', 'known')

	with pytest.raises(ValueError, match='unknown visualization\\.survey_ids'):
		run_cluster_visualization(
			{
				'clustering': {'input_dir': str(input_dir)},
				'visualization': {
					'output_dir': str(tmp_path / 'figures'),
					'survey_ids': ['missing'],
					'modes': ['token'],
					'xy_slices': [0],
					'xz_slices': [],
					'summaries': {'enabled': False},
					'amplitude_underlay': {'enabled': False},
				},
			},
		)


def test_proc_visualization_rejects_all_survey_voxel_reconstruction(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'cluster_run'
	_make_cluster_labels(input_dir / 'labels' / 'k2', 'survey')

	with pytest.raises(ValueError, match='empty survey_ids list'):
		run_cluster_visualization(
			{
				'clustering': {'input_dir': str(input_dir)},
				'visualization': {
					'output_dir': str(tmp_path / 'figures'),
					'survey_ids': [],
					'modes': ['token'],
					'reconstruct_voxel': True,
					'xy_slices': [0],
					'xz_slices': [],
					'summaries': {'enabled': False},
					'amplitude_underlay': {'enabled': False},
				},
			},
		)


def test_proc_visualization_large_voxel_estimate_fails_before_writing(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'cluster_run'
	labels_dir = input_dir / 'labels' / 'k2'
	_make_cluster_labels(labels_dir, 'survey')

	with pytest.raises(ValueError, match='estimated voxel label output'):
		run_cluster_visualization(
			{
				'clustering': {'input_dir': str(input_dir)},
				'visualization': {
					'output_dir': str(tmp_path / 'figures'),
					'survey_ids': ['survey'],
					'modes': ['token'],
					'reconstruct_voxel': True,
					'max_voxel_output_gib': 0.0,
					'xy_slices': [0],
					'xz_slices': [],
					'summaries': {'enabled': False},
					'amplitude_underlay': {'enabled': False},
				},
			},
		)

	assert not (labels_dir / 'survey.cluster_labels_voxel.npy').exists()


def test_proc_visualization_token_only_all_surveys_is_allowed(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'cluster_run'
	output_dir = tmp_path / 'figures'
	_make_cluster_labels(input_dir / 'labels' / 'k2', 'first')
	_make_cluster_labels(input_dir / 'labels' / 'k2', 'second')

	result = run_cluster_visualization(
		{
			'clustering': {'input_dir': str(input_dir)},
			'visualization': {
				'output_dir': str(output_dir),
				'survey_ids': [],
				'modes': ['token'],
				'xy_slices': [0],
				'xz_slices': [],
				'summaries': {'enabled': False},
				'amplitude_underlay': {'enabled': False},
			},
		},
	)

	assert result == {'png_count': 2, 'voxel_count': 0, 'summary_count': 0}
	assert (output_dir / 'token' / 'first_k2_xy_z0.png').is_file()
	assert (output_dir / 'token' / 'second_k2_xy_z0.png').is_file()


def test_default_cluster_visualization_config_is_token_only() -> None:
	config = load_config(DEFAULT_CONFIG)
	visualization = config['visualization']

	assert visualization['survey_ids'] == []
	assert visualization['modes'] == ['token']
	assert visualization['reconstruct_voxel'] is False
	assert visualization['allow_all_surveys_for_voxel_reconstruction'] is False
	assert visualization['skip_existing_voxel_labels'] is True


def test_proc_visualization_maps_realistic_voxel_slices_to_token_slices(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	input_dir = tmp_path / 'cluster_run'
	output_dir = tmp_path / 'figures'
	labels_dir = input_dir / 'labels' / 'k8'
	labels_dir.mkdir(parents=True)
	np.save(
		labels_dir / 'survey.cluster_labels_token.npy',
		np.zeros((38, 38, 188), dtype=np.int32),
	)
	(labels_dir / 'survey.cluster_label_metadata.json').write_text(
		json.dumps(
			{
				'patch_size': [8, 8, 8],
				'volume_shape_xyz': [300, 300, 1501],
			},
		)
		+ '\n',
		encoding='utf-8',
	)
	titles = []
	original_set_title = plt.Axes.set_title

	def capture_title(
		self: object,
		label: str,
		*args: object,
		**kwargs: object,
	) -> object:
		titles.append(label)
		return original_set_title(self, label, *args, **kwargs)

	monkeypatch.setattr(plt.Axes, 'set_title', capture_title)

	result = run_cluster_visualization(
		{
			'clustering': {'input_dir': str(input_dir)},
			'visualization': {
				'output_dir': str(output_dir),
				'modes': ['token'],
				'slice_coordinate_space': 'voxel',
				'xy_slices': [750],
				'xz_slices': [150],
				'summaries': {'enabled': False},
				'amplitude_underlay': {'enabled': False},
			},
		},
	)

	assert result == {'png_count': 2, 'voxel_count': 0, 'summary_count': 0}
	assert (output_dir / 'token' / 'survey_k8_xy_z750.png').is_file()
	assert (output_dir / 'token' / 'survey_k8_xz_y150.png').is_file()
	assert titles == [
		'survey k=8 token XY voxel-z=750 token-z=93',
		'survey k=8 token XZ voxel-y=150 token-y=18',
	]


def test_token_titles_include_voxel_and_token_slice_indices(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	titles = []
	original_set_title = plt.Axes.set_title

	def capture_title(
		self: object,
		label: str,
		*args: object,
		**kwargs: object,
	) -> object:
		titles.append(label)
		return original_set_title(self, label, *args, **kwargs)

	monkeypatch.setattr(plt.Axes, 'set_title', capture_title)

	created = save_cluster_slice_pngs(
		np.zeros((2, 2, 2), dtype=np.int32),
		survey_id='survey',
		k=2,
		mode='token',
		output_dir=tmp_path,
		slices=ClusterSliceRequest(
			xz_slices=(ClusterSlice(array_slice_index=1, voxel_slice_index=150),),
		),
	)

	assert created[0].name == 'survey_k2_xz_y150.png'
	assert titles == ['survey k=2 token XZ voxel-y=150 token-y=1']


def test_proc_visualization_rejects_out_of_range_voxel_slices(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'cluster_run'
	labels_dir = input_dir / 'labels' / 'k2'
	labels_dir.mkdir(parents=True)
	np.save(
		labels_dir / 'survey.cluster_labels_token.npy',
		np.zeros((2, 2, 2), dtype=np.int32),
	)
	(labels_dir / 'survey.cluster_label_metadata.json').write_text(
		json.dumps(
			{
				'patch_size': [2, 2, 2],
				'volume_shape_xyz': [4, 4, 4],
			},
		)
		+ '\n',
		encoding='utf-8',
	)

	with pytest.raises(ValueError, match='xy voxel slice index out of range'):
		run_cluster_visualization(
			{
				'clustering': {'input_dir': str(input_dir)},
				'visualization': {
					'output_dir': str(tmp_path / 'figures'),
					'modes': ['token'],
					'xy_slices': [4],
					'xz_slices': [],
					'summaries': {'enabled': False},
					'amplitude_underlay': {'enabled': False},
				},
			},
		)


def test_proc_visualization_requires_patch_metadata_for_token_mapping(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'cluster_run'
	labels_dir = input_dir / 'labels' / 'k2'
	labels_dir.mkdir(parents=True)
	np.save(
		labels_dir / 'survey.cluster_labels_token.npy',
		np.zeros((2, 2, 2), dtype=np.int32),
	)

	with pytest.raises(ValueError, match='requires metadata field patch_size'):
		run_cluster_visualization(
			{
				'clustering': {'input_dir': str(input_dir)},
				'visualization': {
					'output_dir': str(tmp_path / 'figures'),
					'modes': ['token'],
					'xy_slices': [0],
					'xz_slices': [],
					'summaries': {'enabled': False},
					'amplitude_underlay': {'enabled': False},
				},
			},
		)


def test_proc_visualization_requires_volume_metadata_for_token_mapping(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'cluster_run'
	labels_dir = input_dir / 'labels' / 'k2'
	labels_dir.mkdir(parents=True)
	np.save(
		labels_dir / 'survey.cluster_labels_token.npy',
		np.zeros((2, 2, 2), dtype=np.int32),
	)
	(labels_dir / 'survey.cluster_label_metadata.json').write_text(
		json.dumps({'patch_size': [2, 2, 2]}) + '\n',
		encoding='utf-8',
	)

	with pytest.raises(ValueError, match='volume_shape_xyz or a valid'):
		run_cluster_visualization(
			{
				'clustering': {'input_dir': str(input_dir)},
				'visualization': {
					'output_dir': str(tmp_path / 'figures'),
					'modes': ['token'],
					'xy_slices': [0],
					'xz_slices': [],
					'summaries': {'enabled': False},
					'amplitude_underlay': {'enabled': False},
				},
			},
		)


@pytest.mark.parametrize('slice_value', [1.9, True])
def test_proc_visualization_rejects_non_integer_slice_values(
	tmp_path: Path,
	slice_value: object,
) -> None:
	with pytest.raises(TypeError, match=r'visualization\.xy_slices'):
		run_cluster_visualization(
			{
				'clustering': {'input_dir': str(tmp_path / 'cluster_run')},
				'visualization': {
					'output_dir': str(tmp_path / 'figures'),
					'xy_slices': [slice_value],
					'xz_slices': [0],
				},
			},
		)


def _make_cluster_labels(labels_dir: Path, survey_id: str) -> None:
	labels_dir.mkdir(parents=True, exist_ok=True)
	np.save(
		labels_dir / f'{survey_id}.cluster_labels_token.npy',
		np.array([[[0, 1], [1, -1]], [[1, 0], [-1, 0]]], dtype=np.int32),
	)
	(labels_dir / f'{survey_id}.cluster_label_metadata.json').write_text(
		json.dumps({'patch_size': [2, 2, 2], 'volume_shape_xyz': [4, 4, 4]})
		+ '\n',
		encoding='utf-8',
	)


def _run_token_underlay_visualization(root: Path, amplitude: np.ndarray) -> Path:
	input_dir = root / 'cluster_run'
	output_dir = root / 'figures'
	labels_dir = input_dir / 'labels' / 'k1'
	labels_dir.mkdir(parents=True)
	np.save(
		labels_dir / 'survey.cluster_labels_token.npy',
		np.zeros((2, 2, 1), dtype=np.int32),
	)
	amplitude_path = root / 'survey.npy'
	np.save(amplitude_path, amplitude)
	(labels_dir / 'survey.cluster_label_metadata.json').write_text(
		json.dumps(
			{
				'source_amplitude_path': str(amplitude_path),
				'patch_size': [2, 2, 1],
			},
		)
		+ '\n',
		encoding='utf-8',
	)

	result = run_cluster_visualization(
		{
			'clustering': {'input_dir': str(input_dir)},
			'visualization': {
				'output_dir': str(output_dir),
				'modes': ['token'],
				'xy_slices': [0],
				'xz_slices': [],
				'summaries': {'enabled': False},
				'amplitude_underlay': {'enabled': True, 'alpha': 0.8},
			},
		},
	)

	assert result == {'png_count': 1, 'voxel_count': 0, 'summary_count': 0}
	return output_dir / 'token' / 'survey_k1_xy_z0.png'
