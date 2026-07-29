"""Shared, side-effect-free frozen ordered-HMM replay primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path  # noqa: TC003

import joblib
import numpy as np

from seis_ssl_cluster.clustering.features import EmbeddingInput, file_sha256
from seis_ssl_cluster.clustering.residualization import read_residualizer_npz
from seis_ssl_cluster.clustering.stratigraphic_hmm import (
	prepare_feature_batch_for_indices,
	squared_euclidean_emission_costs,
	viterbi_decode_costs,
)

CANONICAL_KS = (6, 8, 10)


def load_frozen_hmm_model(
	*,
	clustering_output_dir: Path,
	clustering_config: Path | None,
	k: int,
	joblib_load: Callable[[Path], object] = joblib.load,
) -> dict[str, object]:
	"""Load one recorded model without fitting or mutating any source artifact."""
	model_dir = clustering_output_dir / 'models' / f'k{k}'
	paths = {
		name: model_dir / filename
		for name, filename in {
			'preprocessor': 'preprocessor.joblib',
			'hmm_model': 'hmm_model.joblib',
			'centers': 'cluster_centers.npy',
			'metadata': 'clustering_metadata.json',
		}.items()
	}
	if not all(path.is_file() for path in paths.values()):
		raise FileNotFoundError(f'frozen model artifacts are incomplete for k={k}')
	hmm = joblib_load(paths['hmm_model'])
	if not isinstance(hmm, Mapping):
		raise TypeError(f'hmm_model must be a mapping for k={k}')
	emission_source = hmm.get('emission_source')
	if emission_source not in {'embedding', 'z_coordinate'}:
		raise ValueError(
			'frozen hmm_model must record a valid emission_source for k={k}'
		)
	centers = np.load(paths['centers'], mmap_mode='r', allow_pickle=False)
	if centers.shape != (k, centers.shape[1]):
		raise ValueError(f'center shape does not match k={k}')
	residualizer_path = clustering_output_dir / 'models' / 'residualizer.npz'
	residualizer = (
		read_residualizer_npz(residualizer_path)
		if residualizer_path.is_file()
		else None
	)
	frozen_identity: dict[str, object] = {
		name: _reference(path) for name, path in paths.items()
	}
	if residualizer_path.is_file():
		frozen_identity['residualizer'] = _reference(residualizer_path)
	identity = dict(frozen_identity)
	if clustering_config is not None:
		identity['clustering_config'] = _reference(clustering_config)
	return {
		'preprocessor': joblib_load(paths['preprocessor']),
		'hmm': hmm,
		'centers': np.asarray(centers, dtype=np.float32),
		'residualizer': residualizer,
		'emission_source': emission_source,
		'transition_costs': np.asarray(hmm['transition_costs'], dtype=np.float32),
		'initial_costs': np.asarray(hmm['initial_state_costs'], dtype=np.float32),
		'terminal_costs': np.asarray(hmm['terminal_state_costs'], dtype=np.float32),
		'identity': identity,
		'frozen_identity': frozen_identity,
	}


def expected_boundaries(
	hmm: Mapping[str, object], *, k: int, length: int
) -> tuple[int | None, float]:
	"""Resolve the recorded ordered-path boundary prior for one trace."""
	prior = hmm.get('path_prior', {})
	if not isinstance(prior, Mapping) or not prior.get('enabled', False):
		return None, 0.0
	value = prior.get('expected_boundaries', {})
	if not isinstance(value, Mapping) or not value.get('enabled', False):
		return None, 0.0
	weight = float(value.get('weight', 0.0))
	if weight == 0.0:
		return None, 0.0
	target = k - 1 if value.get('target') == 'auto_k_minus_1' else int(value['target'])
	return min(target, length - 1), weight


def replay_frozen_hmm_trace(  # noqa: PLR0913
	embedding: EmbeddingInput,
	flat_indices: np.ndarray,
	model: Mapping[str, object],
	*,
	k: int,
	prepare_features: Callable[..., np.ndarray] = prepare_feature_batch_for_indices,
	emission_costs: Callable[
		[np.ndarray, np.ndarray], np.ndarray
	] = squared_euclidean_emission_costs,
) -> tuple[np.ndarray, np.ndarray]:
	"""Recompute frozen emission costs and the exact recorded Viterbi path."""
	features = prepare_features(
		embedding,
		flat_indices,
		residualizer=model['residualizer'],
		preprocessor=model['preprocessor'],
		emission_source=str(model['emission_source']),
	)
	costs = emission_costs(features, model['centers'])
	expected, weight = expected_boundaries(model['hmm'], k=k, length=costs.shape[0])
	labels = viterbi_decode_costs(
		costs,
		model['transition_costs'],
		initial_state_costs=model['initial_costs'],
		terminal_state_costs=model['terminal_costs'],
		expected_boundary_count=expected,
		boundary_count_weight=weight,
	)
	return costs, labels


def _reference(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


__all__ = [
	'CANONICAL_KS',
	'expected_boundaries',
	'load_frozen_hmm_model',
	'replay_frozen_hmm_trace',
]
