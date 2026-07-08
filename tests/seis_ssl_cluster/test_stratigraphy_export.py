from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from seis_ssl_cluster.stratigraphy import (
	export_hmm_cluster_labels_as_pseudo_targets,
	load_pseudo_target_arrays,
	load_pseudo_target_metadata,
	pseudo_target_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = (
	REPO_ROOT
	/ 'proc'
	/ 'seis_ssl_cluster'
	/ 'export_strat_hmm_pseudo_targets.py'
)


def test_export_from_synthetic_hmm_labels(tmp_path: Path) -> None:
	labels = np.array([[[0, -1], [1, 2]]], dtype=np.int32)
	clustering_dir = _write_hmm_labels(tmp_path, 'survey_a', labels=labels)

	results = export_hmm_cluster_labels_as_pseudo_targets(
		clustering_output_dir=clustering_dir,
		pseudo_target_root=tmp_path / 'pseudo',
		k=3,
	)

	assert [result.survey_id for result in results] == ['survey_a']
	result = results[0]
	assert result.valid_token_count == 3
	assert result.labels_path.is_file()
	assert result.confidence_path.is_file()
	assert result.valid_tokens_path.is_file()
	assert result.metadata_path.is_file()
	arrays = load_pseudo_target_arrays(
		pseudo_target_paths(tmp_path / 'pseudo', k=3, survey_id='survey_a'),
	)
	np.testing.assert_array_equal(arrays.labels, labels)
	np.testing.assert_array_equal(arrays.valid_tokens, labels >= 0)


def test_export_confidence_is_constant_on_valid_and_zero_on_invalid(
	tmp_path: Path,
) -> None:
	labels = np.array([[[0, -1, 1]]], dtype=np.int32)
	clustering_dir = _write_hmm_labels(tmp_path, 'survey_a', k=2, labels=labels)

	export_hmm_cluster_labels_as_pseudo_targets(
		clustering_output_dir=clustering_dir,
		pseudo_target_root=tmp_path / 'pseudo',
		k=2,
		confidence=0.35,
	)

	arrays = load_pseudo_target_arrays(
		pseudo_target_paths(tmp_path / 'pseudo', k=2, survey_id='survey_a'),
	)
	np.testing.assert_array_equal(
		arrays.confidence,
		np.array([[[0.35, 0.0, 0.35]]], dtype=np.float32),
	)


def test_export_includes_source_metadata_provenance(tmp_path: Path) -> None:
	source_metadata = {
		'method': 'stratigraphic_hmm_kmeans',
		'run_id': 'source-run',
	}
	clustering_dir = _write_hmm_labels(
		tmp_path,
		'survey_a',
		source_metadata=source_metadata,
	)
	metadata_path = (
		clustering_dir
		/ 'labels'
		/ 'k3'
		/ 'survey_a.cluster_label_metadata.json'
	)

	export_hmm_cluster_labels_as_pseudo_targets(
		clustering_output_dir=clustering_dir,
		pseudo_target_root=tmp_path / 'pseudo',
		k=3,
		confidence=0.75,
	)

	metadata = load_pseudo_target_metadata(
		pseudo_target_paths(tmp_path / 'pseudo', k=3, survey_id='survey_a'),
	)
	assert metadata['source'] == {
		'export_confidence': 0.75,
		'source_clustering_output_dir': str(clustering_dir),
		'source_label_path': str(
			clustering_dir
			/ 'labels'
			/ 'k3'
			/ 'survey_a.cluster_labels_token.npy',
		),
		'source_metadata_path': str(metadata_path),
		'source_metadata_sha256': hashlib.sha256(
			metadata_path.read_bytes(),
		).hexdigest(),
		'source_method': 'stratigraphic_hmm_kmeans',
	}


def test_export_rejects_existing_outputs_without_overwrite(tmp_path: Path) -> None:
	clustering_dir = _write_hmm_labels(tmp_path, 'survey_a')
	pseudo_root = tmp_path / 'pseudo'
	export_hmm_cluster_labels_as_pseudo_targets(
		clustering_output_dir=clustering_dir,
		pseudo_target_root=pseudo_root,
		k=3,
	)

	with pytest.raises(FileExistsError, match='overwrite=True'):
		export_hmm_cluster_labels_as_pseudo_targets(
			clustering_output_dir=clustering_dir,
			pseudo_target_root=pseudo_root,
			k=3,
		)

	results = export_hmm_cluster_labels_as_pseudo_targets(
		clustering_output_dir=clustering_dir,
		pseudo_target_root=pseudo_root,
		k=3,
		overwrite=True,
	)
	assert results[0].metadata_path.is_file()


def test_cli_dry_run_validates_and_does_not_create_files(tmp_path: Path) -> None:
	clustering_dir = _write_hmm_labels(tmp_path, 'survey_a')
	pseudo_root = tmp_path / 'pseudo'

	completed = _run_cli(
		'--clustering-output-dir',
		str(clustering_dir),
		'--pseudo-target-root',
		str(pseudo_root),
		'--k',
		'3',
		'--dry-run',
	)

	assert completed.returncode == 0
	assert 'execution: dry-run; no files written' in completed.stdout
	assert not pseudo_root.exists()


def test_cli_execution_writes_expected_pseudo_target_files(tmp_path: Path) -> None:
	clustering_dir = _write_hmm_labels(tmp_path, 'survey_a')
	pseudo_root = tmp_path / 'pseudo'

	completed = _run_cli(
		'--clustering-output-dir',
		str(clustering_dir),
		'--pseudo-target-root',
		str(pseudo_root),
		'--k',
		'3',
		'--confidence',
		'0.5',
	)

	assert completed.returncode == 0
	assert 'pseudo_target_exports: 1' in completed.stdout
	paths = pseudo_target_paths(pseudo_root, k=3, survey_id='survey_a')
	assert paths.labels.is_file()
	assert paths.confidence.is_file()
	assert paths.valid_tokens.is_file()
	assert paths.metadata.is_file()
	np.testing.assert_array_equal(
		np.load(paths.confidence),
		np.array([[[0.5, 0.0, 0.5]]], dtype=np.float32),
	)


def _write_hmm_labels(
	tmp_path: Path,
	survey_id: str,
	*,
	k: int = 3,
	labels: np.ndarray | None = None,
	source_metadata: dict[str, object] | None = None,
) -> Path:
	clustering_dir = tmp_path / 'clusters'
	label_dir = clustering_dir / 'labels' / f'k{k}'
	label_dir.mkdir(parents=True)
	label_array = (
		np.array([[[0, -1, 2]]], dtype=np.int32)
		if labels is None
		else labels
	)
	np.save(label_dir / f'{survey_id}.cluster_labels_token.npy', label_array)
	if source_metadata is not None:
		(label_dir / f'{survey_id}.cluster_label_metadata.json').write_text(
			json.dumps(source_metadata, sort_keys=True) + '\n',
			encoding='utf-8',
		)
	return clustering_dir


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
	env = os.environ.copy()
	env['PYTHONPATH'] = os.pathsep.join(
		(
			str(REPO_ROOT / 'src'),
			env.get('PYTHONPATH', ''),
		),
	)
	return subprocess.run(  # noqa: S603
		[sys.executable, str(CLI_PATH), *args],
		cwd=REPO_ROOT,
		env=env,
		text=True,
		capture_output=True,
		check=False,
		timeout=30,
	)
