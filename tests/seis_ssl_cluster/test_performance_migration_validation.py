from __future__ import annotations

import numpy as np

from seis_ssl_cluster.migration import performance_validation as validation


def _embedding_artifact(
	array: np.ndarray,
	valid: np.ndarray,
) -> dict[str, object]:
	return {
		'embeddings': array,
		'valid_tokens': valid,
		'metadata': {
			'survey_id': 'f3',
			'checkpoint_sha256': 'a' * 64,
			'patch_size': [8, 8, 8],
			'token_grid_shape': [2, 2, 1],
			'window_size': [128, 128, 128],
			'overlap': [112, 64, 64],
			'output_dtype': 'float16',
			'min_token_valid_fraction': 0.5,
			'preprocessing_cache': {'effective_mode': 'off'},
		},
	}


def test_embedding_pair_exact_identity() -> None:
	array = np.ones((2, 2, 1, 3), dtype=np.float16)
	valid = np.ones((2, 2, 1), dtype=bool)
	result = validation._compare_one_embedding_pair(  # noqa: SLF001
		_embedding_artifact(array, valid),
		_embedding_artifact(array.copy(), valid.copy()),
		left_name='A',
		right_name='B',
	)

	assert result['status'] == 'EXACT'
	assert result['embedding_array_equal'] is True
	assert result['valid_token_mask_exact'] is True


def test_embedding_pair_numeric_drift_excludes_invalid_tokens() -> None:
	left = np.ones((2, 2, 1, 2), dtype=np.float16)
	right = left.copy()
	valid = np.ones((2, 2, 1), dtype=bool)
	valid[0, 0, 0] = False
	right[0, 0, 0] = 99.0
	right[1, 1, 0, 0] += np.float16(0.25)
	result = validation._compare_one_embedding_pair(  # noqa: SLF001
		_embedding_artifact(left, valid),
		_embedding_artifact(right, valid),
		left_name='A',
		right_name='B',
	)

	assert result['status'] == 'NUMERIC_DRIFT'
	assert result['different_element_count'] == 1
	assert result['absolute_error']['max'] == 0.25


def test_embedding_pair_rejects_valid_mask_mismatch() -> None:
	array = np.ones((2, 2, 1, 2), dtype=np.float16)
	left_valid = np.ones((2, 2, 1), dtype=bool)
	right_valid = left_valid.copy()
	right_valid[1, 1, 0] = False
	result = validation._compare_one_embedding_pair(  # noqa: SLF001
		_embedding_artifact(array, left_valid),
		_embedding_artifact(array, right_valid),
		left_name='A',
		right_name='B',
	)

	assert result['status'] == 'BLOCKED_NUMERIC_CONTRACT'
	assert result['valid_token_mask_exact'] is False


def test_migration_decision_priority() -> None:
	stages = {
		'preflight': {'missing_input_count': 0},
		'checkpoint_smoke': {'status': 'PASS'},
		'embedding_parity': {'comparisons': {}},
		'probe_parity': {'parity': {}},
		'hmm_parity': {
			'labels': {
				'decoded_labels_exact': False,
				'valid_token_mask_exact': True,
			},
		},
		'pseudo_target_parity': {
			'labels': {'exact': False},
			'valid_tokens': {'exact': True},
			'confidence': {'threshold_crossing_count': 1},
		},
		'benchmark': {'status': 'PASS'},
	}

	decision = validation._decide_migration(stages)  # noqa: SLF001

	assert decision['status'] == 'REBUILD_M1_REQUIRED'
	assert decision['complete'] is True


def test_migration_decision_blocks_before_rebuild() -> None:
	stages = {
		'preflight': {'missing_input_count': 0},
		'checkpoint_smoke': {'status': 'BLOCKED_NUMERIC_CONTRACT'},
		'embedding_parity': {'comparisons': {}},
		'probe_parity': {'parity': {}},
		'hmm_parity': {
			'labels': {
				'decoded_labels_exact': False,
				'valid_token_mask_exact': True,
			},
		},
		'pseudo_target_parity': {
			'labels': {'exact': False},
			'valid_tokens': {'exact': True},
			'confidence': {'threshold_crossing_count': 1},
		},
		'benchmark': {'status': 'PASS'},
	}

	decision = validation._decide_migration(stages)  # noqa: SLF001

	assert decision['status'] == 'BLOCKED_NUMERIC_CONTRACT'
