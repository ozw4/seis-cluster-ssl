from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.stratigraphy import (
	StratPseudoTargetInput,
	discover_pseudo_target_inputs,
	load_pseudo_target_arrays,
	load_pseudo_target_metadata,
	pseudo_target_paths,
	validate_pseudo_target_arrays,
	write_pseudo_target,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_pseudo_target_paths_uses_expected_filenames(tmp_path: Path) -> None:
	paths = pseudo_target_paths(tmp_path, k=4, survey_id='survey_a')

	assert paths.labels == tmp_path / 'k4' / 'survey_a.hmm_labels_token.npy'
	assert paths.confidence == tmp_path / 'k4' / 'survey_a.hmm_confidence_token.npy'
	assert paths.valid_tokens == tmp_path / 'k4' / 'survey_a.valid_tokens.npy'
	assert paths.metadata == (
		tmp_path / 'k4' / 'survey_a.pseudo_target_metadata.json'
	)


def test_write_load_and_discover_pseudo_targets(tmp_path: Path) -> None:
	labels = np.array([[[1, -1]], [[0, 2]]], dtype=np.int32)
	confidence = np.array([[[0.5, 0.0]], [[1.0, 0.25]]], dtype=np.float32)
	valid = np.array([[[True, False]], [[True, True]]], dtype=np.bool_)
	write_pseudo_target(
		tmp_path,
		k=3,
		survey_id='survey_b',
		labels=labels,
		confidence=confidence,
		valid_tokens=valid,
		metadata={'run_id': 'b'},
	)
	write_pseudo_target(
		tmp_path,
		k=3,
		survey_id='survey_a',
		labels=labels,
		confidence=confidence,
		valid_tokens=valid,
		metadata={'schema_version': 999, 'run_id': 'a'},
	)

	inputs = discover_pseudo_target_inputs(tmp_path, k=3)

	assert [item.survey_id for item in inputs] == ['survey_a', 'survey_b']
	assert all(isinstance(item, StratPseudoTargetInput) for item in inputs)
	arrays = load_pseudo_target_arrays(inputs[0])
	np.testing.assert_array_equal(arrays.labels, labels)
	np.testing.assert_array_equal(arrays.confidence, confidence)
	np.testing.assert_array_equal(arrays.valid_tokens, valid)
	metadata = load_pseudo_target_metadata(inputs[0])
	assert metadata == {
		'artifact_type': 'strat_hmm_pseudo_target',
		'invalid_token_count': 1,
		'k': 3,
		'label_counts': {'0': 1, '1': 1, '2': 1},
		'schema_version': 1,
		'source': {'run_id': 'a', 'schema_version': 999},
		'survey_id': 'survey_a',
		'token_grid_shape': [2, 1, 2],
		'valid_token_count': 3,
	}


@pytest.mark.parametrize(
	('updates', 'error'),
	[
		({'confidence': np.ones((1, 1, 1), dtype=np.float32)}, 'shapes'),
		({'labels': np.array([[0, 1]], dtype=np.int32)}, 'labels must be 3D'),
		(
			{
				'labels': np.array([[[0, 3]]], dtype=np.int32),
				'confidence': np.array([[[1.0, 1.0]]], dtype=np.float32),
				'valid_tokens': np.array([[[True, True]]], dtype=np.bool_),
			},
			r'\[0, 3\)',
		),
		({'valid_tokens': np.array([[[1, 1]]], dtype=np.int32)}, 'dtype must be bool'),
		({'confidence': np.array([[[1.0, np.nan]]])}, 'finite'),
		({'confidence': np.array([[[1.0, 1.5]]])}, r'\[0, 1\]'),
	],
)
def test_validate_pseudo_target_arrays_rejects_invalid_contracts(
	updates: dict[str, np.ndarray],
	error: str,
) -> None:
	arrays = _valid_arrays()
	arrays.update(updates)

	with pytest.raises((TypeError, ValueError), match=f'{error}.*survey_bad'):
		validate_pseudo_target_arrays(
			arrays['labels'],
			arrays['confidence'],
			arrays['valid_tokens'],
			k=3,
			survey_id='survey_bad',
		)


@pytest.mark.parametrize(
	('updates', 'error'),
	[
		(
			{'labels': np.array([[[0, 1]]], dtype=np.int32)},
			'labels must be -1',
		),
		(
			{'confidence': np.array([[[1.0, 0.2]]], dtype=np.float32)},
			'confidence must be 0.0',
		),
	],
)
def test_validate_pseudo_target_arrays_rejects_invalid_positions(
	updates: dict[str, np.ndarray],
	error: str,
) -> None:
	arrays = _valid_arrays()
	arrays.update(updates)

	with pytest.raises(ValueError, match=f'{error}.*survey_bad'):
		validate_pseudo_target_arrays(
			arrays['labels'],
			arrays['confidence'],
			arrays['valid_tokens'],
			k=3,
			survey_id='survey_bad',
		)


def test_pseudo_target_metadata_is_json_safe(tmp_path: Path) -> None:
	arrays = _valid_arrays()
	paths = write_pseudo_target(
		tmp_path,
		k=3,
		survey_id='survey_a',
		labels=arrays['labels'],
		confidence=arrays['confidence'],
		valid_tokens=arrays['valid_tokens'],
	)

	raw = paths.metadata.read_text(encoding='utf-8')
	metadata = json.loads(raw)

	assert raw.endswith('\n')
	assert metadata['token_grid_shape'] == [1, 1, 2]
	assert metadata['valid_token_count'] == 1
	assert metadata['invalid_token_count'] == 1
	assert metadata['label_counts'] == {'0': 1, '1': 0, '2': 0}
	json.dumps(metadata, allow_nan=False)


def _valid_arrays() -> dict[str, np.ndarray]:
	return {
		'labels': np.array([[[0, -1]]], dtype=np.int32),
		'confidence': np.array([[[1.0, 0.0]]], dtype=np.float32),
		'valid_tokens': np.array([[[True, False]]], dtype=np.bool_),
	}
