from __future__ import annotations

from pathlib import Path

import pytest

from seis_ssl_cluster.paths import (
	DEFAULT_ARTIFACT_ROOT,
	ArtifactPaths,
	ExperimentKey,
	ResultsPaths,
	ensure_under_root,
	reject_runs_path,
	safe_slug,
)

MODEL_TAG = 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
KEY = ExperimentKey(
	dataset='f3',
	version='facies_benchmark_v1',
	model_tag=MODEL_TAG,
	subset='overlap_x16',
	embed_spec='tokens_v1',
	cluster_spec='k4_6_8_whiten',
	viz_spec='selected_surveys',
	label_set='png_slices_segy_labels_v1',
	probe_spec='linear_balanced_v1',
	baseline_tag='z_only_v1',
	run_spec='full_100ep',
)


def test_pretraining_path_uses_pretraining_root_and_never_runs() -> None:
	path = ArtifactPaths().pretraining(KEY)

	assert path == (
		DEFAULT_ARTIFACT_ROOT
		/ 'pretraining'
		/ 'f3'
		/ 'facies_benchmark_v1'
		/ MODEL_TAG
		/ 'full_100ep'
	)
	assert 'runs' not in path.parts


def test_embeddings_path_separates_optional_subset_from_embed_spec() -> None:
	paths = ArtifactPaths()

	assert paths.embeddings(KEY) == (
		DEFAULT_ARTIFACT_ROOT
		/ 'embeddings'
		/ 'f3'
		/ 'facies_benchmark_v1'
		/ MODEL_TAG
		/ 'overlap_x16'
		/ 'tokens_v1'
	)
	assert paths.embeddings(
		ExperimentKey(
			dataset='f3',
			version='facies_benchmark_v1',
			model_tag=MODEL_TAG,
			embed_spec='overlap_x16',
		)
	) == (
		DEFAULT_ARTIFACT_ROOT
		/ 'embeddings'
		/ 'f3'
		/ 'facies_benchmark_v1'
		/ MODEL_TAG
		/ 'overlap_x16'
	)


def test_clustering_path_splits_subset_embed_spec_and_cluster_spec() -> None:
	path = ArtifactPaths().clustering(KEY)

	assert path == (
		DEFAULT_ARTIFACT_ROOT
		/ 'clustering'
		/ 'f3'
		/ 'facies_benchmark_v1'
		/ MODEL_TAG
		/ 'overlap_x16'
		/ 'tokens_v1'
		/ 'k4_6_8_whiten'
	)


def test_cluster_visualization_path_includes_viz_spec() -> None:
	path = ArtifactPaths().cluster_visualization(KEY)

	assert path == (
		DEFAULT_ARTIFACT_ROOT
		/ 'visualizations'
		/ 'clusters'
		/ 'f3'
		/ 'facies_benchmark_v1'
		/ MODEL_TAG
		/ 'overlap_x16'
		/ 'tokens_v1'
		/ 'k4_6_8_whiten'
		/ 'selected_surveys'
	)


def test_lithology_probe_artifact_paths_are_stable() -> None:
	paths = ArtifactPaths()
	base = (
		DEFAULT_ARTIFACT_ROOT
		/ 'lithology'
		/ 'f3'
		/ 'facies_benchmark_v1'
		/ MODEL_TAG
		/ 'tokens_v1'
		/ 'png_slices_segy_labels_v1'
	)

	assert paths.lithology_token_dataset(KEY) == base / 'token_dataset'
	assert paths.lithology_probe(KEY) == base / 'probes' / 'linear_balanced_v1'
	assert (
		paths.lithology_predictions(KEY)
		== base / 'predictions' / 'linear_balanced_v1'
	)
	assert (
		paths.lithology_visualizations(KEY)
		== base / 'visualizations' / 'linear_balanced_v1'
	)
	assert paths.lithology_report(KEY) == base / 'reports' / 'linear_balanced_v1'


def test_baseline_artifact_paths_are_stable() -> None:
	paths = ArtifactPaths()
	base = (
		DEFAULT_ARTIFACT_ROOT
		/ 'lithology'
		/ 'f3'
		/ 'facies_benchmark_v1'
		/ 'baselines'
		/ 'z_only_v1'
		/ 'png_slices_segy_labels_v1'
	)

	assert paths.baseline_token_dataset(KEY) == base / 'token_dataset'
	assert paths.baseline_probe(KEY) == base / 'probes' / 'linear_balanced_v1'
	assert paths.baseline_comparison_report(KEY) == (
		DEFAULT_ARTIFACT_ROOT
		/ 'lithology'
		/ 'f3'
		/ 'facies_benchmark_v1'
		/ 'reports'
		/ 'baseline_comparison'
	)


def test_results_paths_are_stable() -> None:
	paths = ResultsPaths()

	assert paths.inspection(KEY) == Path('results/f3/facies_benchmark_v1/inspection')
	assert paths.lithology_probe(KEY) == (
		Path('results')
		/ 'f3'
		/ 'facies_benchmark_v1'
		/ 'lithology_probe'
		/ MODEL_TAG
		/ 'tokens_v1'
		/ 'png_slices_segy_labels_v1'
		/ 'linear_balanced_v1'
	)
	assert paths.baseline_comparison(KEY) == (
		Path('results/f3/facies_benchmark_v1/baseline_comparison')
	)


def test_missing_required_key_raises_value_error() -> None:
	key = ExperimentKey(
		dataset='f3',
		version='facies_benchmark_v1',
		model_tag=MODEL_TAG,
		embed_spec='tokens_v1',
	)

	with pytest.raises(ValueError, match='subset'):
		ArtifactPaths().clustering(key)


@pytest.mark.parametrize(
	('field', 'value'),
	[
		('model_tag', 'bad/model'),
		('subset', 'bad/subset'),
	],
)
def test_invalid_experiment_key_components_raise_value_error(
	field: str,
	value: str,
) -> None:
	key = ExperimentKey(
		dataset='f3',
		version='facies_benchmark_v1',
		model_tag=MODEL_TAG,
		subset='overlap_x16',
		embed_spec='tokens_v1',
		cluster_spec='k4_6_8_whiten',
	)
	key = ExperimentKey(**{**key.__dict__, field: value})

	with pytest.raises(ValueError, match=field):
		ArtifactPaths().clustering(key)


@pytest.mark.parametrize('value', ['bad/path', 'bad path', '../bad', 'bad..name', ''])
def test_invalid_slug_raises_value_error(value: str) -> None:
	with pytest.raises(ValueError, match='must'):
		safe_slug(value, label='model_tag')


def test_safe_slug_returns_valid_slug_unchanged() -> None:
	assert safe_slug('Amp_MAE.v1-2', label='model_tag') == 'Amp_MAE.v1-2'


def test_ensure_under_root_detects_path_outside_root() -> None:
	root = Path('/workspace/artifacts/seis_ssl_cluster')

	with pytest.raises(ValueError, match='must be under root'):
		ensure_under_root(
			Path('/workspace/artifacts/seis_ssl_cluster/../outside'),
			root=root,
			label='output_dir',
		)


def test_reject_runs_path_rejects_runs_component() -> None:
	with pytest.raises(ValueError, match='runs/ paths'):
		reject_runs_path(Path('runs/nopims/pretrain_v1'), label='output_dir')
