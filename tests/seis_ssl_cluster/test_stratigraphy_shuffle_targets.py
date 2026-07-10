from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from proc.seis_ssl_cluster.shuffle_strat_hmm_pseudo_targets import main
from seis_ssl_cluster.stratigraphy.shuffle_targets import (
	shuffle_pseudo_target_arrays,
	shuffle_strat_hmm_pseudo_targets,
)
from seis_ssl_cluster.stratigraphy.targets import (
	load_pseudo_target_arrays,
	load_pseudo_target_metadata,
	pseudo_target_paths,
	write_pseudo_target,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_global_shuffle_is_deterministic_and_preserves_contract() -> None:
	labels, confidence, valid = _arrays()
	boundary_weight = _boundary_weight(valid)

	first = shuffle_pseudo_target_arrays(
		labels,
		confidence,
		valid,
		boundary_weight=boundary_weight,
		k=3,
		seed=42,
	)
	second = shuffle_pseudo_target_arrays(
		labels,
		confidence,
		valid,
		boundary_weight=boundary_weight,
		k=3,
		seed=42,
	)
	different = shuffle_pseudo_target_arrays(
		labels,
		confidence,
		valid,
		boundary_weight=boundary_weight,
		k=3,
		seed=188,
	)

	np.testing.assert_array_equal(first.labels, second.labels)
	np.testing.assert_array_equal(first.confidence, second.confidence)
	np.testing.assert_array_equal(first.boundary_weight, second.boundary_weight)
	np.testing.assert_array_equal(first.valid_tokens, valid)
	assert not np.array_equal(first.labels[valid], different.labels[valid])
	np.testing.assert_array_equal(
		np.bincount(first.labels[valid], minlength=3),
		np.bincount(labels[valid], minlength=3),
	)
	assert np.all(first.labels[~valid] == -1)
	assert np.all(first.confidence[~valid] == 0.0)
	assert np.all(first.boundary_weight[~valid] == 0.0)
	assert sorted(
		zip(
			first.labels[valid],
			first.confidence[valid],
			first.boundary_weight[valid],
			strict=True,
		),
	) == sorted(
		zip(
			labels[valid],
			confidence[valid],
			boundary_weight[valid],
			strict=True,
		),
	)


def test_global_shuffle_modifies_fortran_contiguous_arrays() -> None:
	labels, confidence, valid = (np.asfortranarray(array) for array in _arrays())

	shuffled = shuffle_pseudo_target_arrays(
		labels,
		confidence,
		valid,
		k=3,
		seed=42,
	)

	assert shuffled.labels.flags.f_contiguous
	assert shuffled.confidence.flags.f_contiguous
	assert not np.array_equal(shuffled.labels[valid], labels[valid])
	assert sorted(
		zip(shuffled.labels[valid], shuffled.confidence[valid], strict=True),
	) == sorted(zip(labels[valid], confidence[valid], strict=True))


def test_build_writes_metadata_and_requires_explicit_overwrite(tmp_path: Path) -> None:
	source_root = tmp_path / 'source'
	output_root = tmp_path / 'output'
	labels, confidence, valid = _arrays()
	boundary_weight = _boundary_weight(valid)
	write_pseudo_target(
		source_root,
		k=3,
		survey_id='survey_a',
		labels=labels,
		confidence=confidence,
		valid_tokens=valid,
		boundary_weight=boundary_weight,
		metadata={'run_id': 'source-run'},
	)

	results = shuffle_strat_hmm_pseudo_targets(
		source_root,
		output_root,
		k=3,
		seed=42,
	)
	output = load_pseudo_target_arrays(
		pseudo_target_paths(output_root, k=3, survey_id='survey_a'),
	)
	metadata = load_pseudo_target_metadata(results[0].paths)

	np.testing.assert_array_equal(output.valid_tokens, valid)
	assert all(
		path.is_file()
		for path in (
			results[0].paths.labels,
			results[0].paths.confidence,
			results[0].paths.valid_tokens,
			results[0].paths.boundary_weight,
			results[0].paths.metadata,
		)
	)
	assert sorted(
		zip(
			output.labels[valid],
			output.confidence[valid],
			output.boundary_weight[valid],
			strict=True,
		),
	) == sorted(
		zip(
			labels[valid],
			confidence[valid],
			boundary_weight[valid],
			strict=True,
		),
	)
	assert metadata['source']['source']['run_id'] == 'source-run'
	assert metadata['shuffle'] == {
		'enabled': True,
		'label_counts_preserved': True,
		'mode': 'global_valid_tokens',
		'preserve_boundary_weight_distribution': True,
		'preserve_label_confidence_boundary_weight_tuples': True,
		'preserve_label_confidence_pairs': True,
		'seed': 42,
		'source_pseudo_target_root': str(source_root.resolve()),
		'valid_mask_preserved': True,
	}
	with pytest.raises(FileExistsError, match='use overwrite'):
		shuffle_strat_hmm_pseudo_targets(
			source_root,
			output_root,
			k=3,
			seed=42,
		)
	shuffle_strat_hmm_pseudo_targets(
		source_root,
		output_root,
		k=3,
		seed=42,
		overwrite=True,
	)


def test_build_rejects_unplanned_output_survey_even_with_overwrite(
	tmp_path: Path,
) -> None:
	source_root = tmp_path / 'source'
	output_root = tmp_path / 'output'
	labels, confidence, valid = _arrays()
	write_pseudo_target(
		source_root,
		k=3,
		survey_id='survey_a',
		labels=labels,
		confidence=confidence,
		valid_tokens=valid,
	)
	for survey_id in ('survey_a', 'orphaned_survey'):
		write_pseudo_target(
			output_root,
			k=3,
			survey_id=survey_id,
			labels=labels,
			confidence=confidence,
			valid_tokens=valid,
		)

	with pytest.raises(FileExistsError, match='unplanned survey artifacts'):
		shuffle_strat_hmm_pseudo_targets(
			source_root,
			output_root,
			k=3,
			seed=42,
			overwrite=True,
		)


def test_cli_dry_run_validates_and_prints_planned_outputs(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	source_root = tmp_path / 'source'
	output_root = tmp_path / 'output'
	labels, confidence, valid = _arrays()
	write_pseudo_target(
		source_root,
		k=3,
		survey_id='survey_a',
		labels=labels,
		confidence=confidence,
		valid_tokens=valid,
	)
	config_path = tmp_path / 'shuffle.yaml'
	config_path.write_text(
		'\n'.join(
			(
				'suite:',
				'  name: strat_hmm_m1_guardrails_v1',
				'source:',
				f'  pseudo_target_root: {source_root}',
				'  k: 3',
				'shuffle:',
				'  seed: 42',
				'  scope: global_valid_tokens',
				'  preserve_valid_token_mask: true',
				'  preserve_global_label_histogram: true',
				'  preserve_confidence_distribution: true',
				'  preserve_artifact_schema: true',
				'outputs:',
				f'  pseudo_target_root: {output_root}',
				'  overwrite: false',
			),
		)
		+ '\n',
		encoding='utf-8',
	)
	monkeypatch.setattr(
		sys,
		'argv',
		[
			'shuffle_strat_hmm_pseudo_targets.py',
			'--config',
			str(config_path),
			'--dry-run',
		],
	)

	main()

	output = capsys.readouterr().out
	planned_metadata = output_root / 'k3/survey_a.pseudo_target_metadata.json'
	assert f'planned_output: {planned_metadata}' in output
	assert 'execution: dry-run' in output
	assert not output_root.exists()


def test_cli_config_overwrite_authorizes_replacement(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	source_root = tmp_path / 'source'
	output_root = tmp_path / 'output'
	labels, confidence, valid = _arrays()
	for root in (source_root, output_root):
		write_pseudo_target(
			root,
			k=3,
			survey_id='survey_a',
			labels=labels,
			confidence=confidence,
			valid_tokens=valid,
		)
	config_path = tmp_path / 'shuffle.yaml'
	config_path.write_text(
		'\n'.join(
			(
				'suite:',
				'  name: strat_hmm_m1_guardrails_v1',
				'source:',
				f'  pseudo_target_root: {source_root}',
				'  k: 3',
				'shuffle:',
				'  seed: 42',
				'  scope: global_valid_tokens',
				'  preserve_valid_token_mask: true',
				'  preserve_global_label_histogram: true',
				'  preserve_confidence_distribution: true',
				'  preserve_artifact_schema: true',
				'outputs:',
				f'  pseudo_target_root: {output_root}',
				'  overwrite: true',
			),
		)
		+ '\n',
		encoding='utf-8',
	)
	monkeypatch.setattr(
		sys,
		'argv',
		[
			'shuffle_strat_hmm_pseudo_targets.py',
			'--config',
			str(config_path),
		],
	)

	main()

	metadata = load_pseudo_target_metadata(
		pseudo_target_paths(output_root, k=3, survey_id='survey_a'),
	)
	assert metadata['shuffle']['seed'] == 42


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	labels = np.array(
		[
			[[0, 1, -1, 2], [1, 2, 0, 1]],
			[[2, -1, 0, 1], [2, 0, 1, 2]],
		],
		dtype=np.int32,
	)
	valid = labels >= 0
	confidence = np.zeros(labels.shape, dtype=np.float32)
	confidence[valid] = np.linspace(0.1, 1.0, num=np.count_nonzero(valid))
	return labels, confidence, valid


def _boundary_weight(valid: np.ndarray) -> np.ndarray:
	boundary_weight = np.zeros(valid.shape, dtype=np.float32)
	boundary_weight[valid] = np.linspace(
		0.2,
		1.0,
		num=np.count_nonzero(valid),
	)
	return boundary_weight
