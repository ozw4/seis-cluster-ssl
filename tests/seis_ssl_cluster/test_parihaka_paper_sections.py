"""One-based coordinate and orientation contracts for paper comparisons."""

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from seis_ssl_cluster.f3.lithology import hmm_v1_comparison_visualization as paper
from seis_ssl_cluster.parihaka.paper_sections import BinaryLabels, render


def test_binary_labels() -> None:
	"""Only source class 5 is Channel; source class zero is not defined."""
	labels = BinaryLabels(np.array([[[1, 5, 6]]]))
	np.testing.assert_array_equal(labels[:, :, :], [[[0, 1, 0]]])
	with pytest.raises(ValueError, match='Unknown'):
		BinaryLabels(np.array([[[0]]]))[:, :, :]


@pytest.mark.parametrize('domain', paper.DOMAINS)
def test_index_axes_and_panels(domain: str) -> None:
	"""Eight panels share one-based integer coordinates, including transposed XY."""
	path = (
		Path(__file__).resolve().parents[2]
		/ 'experiments/parihaka/facies_benchmark_v1'
		/ '41_hmm_v1_paper_sections_v1/02_visualize_sections.yaml'
	)
	config = yaml.safe_load(path.read_text())
	config['models'] = [
		{'display_name': t}
		for t in (
			'MAE',
			'MAE + HMM',
			'LocalBT',
			'LocalBT + HMM',
			'Random',
			'Random + HMM',
		)
	]
	volume = np.arange(5 * 6 * 7).reshape(5, 6, 7)
	labels = volume % 2
	arrays = [
		SimpleNamespace(
			predictions=labels, valid_mask=np.ones(volume.shape, dtype=np.bool_)
		)
		for _ in range(6)
	]
	coords = {
		d: np.arange(1, n + 1) for d, n in zip(paper.DOMAINS, volume.shape, strict=True)
	}
	rows = [{'domain': d, 'array_index': 1} for d in paper.DOMAINS]
	manifest = {
		'amplitude': {'vmin': -1, 'vmax': 1},
		'axis_mapping': {
			'inline': ['Y', 'Z'],
			'crossline': ['X', 'Z'],
			'time': ['X', 'Y'],
		},
	}
	prepared = paper.PreparedComparison(
		config,
		volume,
		labels,
		arrays,
		[
			{'class_id': 0, 'class_name': 'Non-Channel', 'rgb': [219, 241, 247]},
			{'class_id': 1, 'class_name': 'Channel', 'rgb': [208, 10, 0]},
		],
		coords,
		manifest,
		rows,
	)
	fig = render(prepared, {'domain': domain, 'array_index': 1})
	try:
		assert len(fig.axes) == 8
		assert fig.axes[4].get_xlabel() == manifest['axis_mapping'][domain][0]
		assert fig.axes[0].get_ylabel() == manifest['axis_mapping'][domain][1]
		for ax in fig.axes:
			assert min(ax.get_xticks()) >= 1
			assert min(ax.get_yticks()) >= 1
			assert ax.get_xlim() == fig.axes[0].get_xlim()
			assert ax.get_ylim() == fig.axes[0].get_ylim()
		if domain == 'time':
			np.testing.assert_array_equal(
				fig.axes[0].images[0].get_array(), volume[:, :, 1].T
			)
			assert fig.axes[0].get_xlim() == (1, 5)
			assert fig.axes[0].get_ylim() == (1, 6)
		else:
			assert fig.axes[0].get_ylim()[0] > fig.axes[0].get_ylim()[1]
	finally:
		plt.close(fig)
