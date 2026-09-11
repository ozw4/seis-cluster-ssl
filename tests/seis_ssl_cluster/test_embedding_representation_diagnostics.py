from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from proc.seis_ssl_cluster import (
	measure_f3_overlap_subcrop_representation as diagnostic_cli,
)
from seis_ssl_cluster.config.f3_lithology_five_way import (
	f3_lithology_five_way_config_from_mapping,
)
from seis_ssl_cluster.embedding import representation_diagnostics as diagnostics
from seis_ssl_cluster.embedding.representation_diagnostics import (
	DEFAULT_REPRESENTATION_LAYER_NORM_EPS,
	REPRESENTATION_METRIC_KEYS,
	EmbeddingRepresentationSource,
	build_embedding_representation_diagnostics,
	representation_metrics,
	systematic_valid_token_indices,
	write_embedding_representation_diagnostics,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from tests.seis_ssl_cluster.helpers_f3_five_way import build_five_way_universe


def test_midpoint_systematic_sampling_uses_c_ordered_valid_tokens() -> None:
	valid = np.array(
		[[True, False, True], [True, False, True], [True, False, True]],
		dtype=bool,
	)

	indices = systematic_valid_token_indices(valid, sample_size=3)

	np.testing.assert_array_equal(indices, np.array([2, 5, 8], dtype=np.int64))


def test_midpoint_systematic_sampling_rejects_nonbool_and_too_few() -> None:
	with pytest.raises(TypeError, match='dtype must be bool'):
		systematic_valid_token_indices(np.ones(4, dtype=np.uint8), sample_size=2)
	with pytest.raises(ValueError, match='valid token count'):
		systematic_valid_token_indices(
			np.array([True, False, True]),
			sample_size=3,
		)


def test_representation_metrics_match_fixed_norm_std_rank_and_layer_norm() -> None:
	features = np.array(
		[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
		dtype=np.float16,
	)

	metrics = representation_metrics(features)

	assert tuple(metrics) == REPRESENTATION_METRIC_KEYS
	assert metrics['raw_feature_norm'] == pytest.approx(1.0)
	assert metrics['token_wise_feature_std'] == pytest.approx(math.sqrt(0.5))
	assert metrics['raw_feature_effective_rank'] == pytest.approx(2.0)
	layer_norm_scale = 0.5 / math.sqrt(0.25 + DEFAULT_REPRESENTATION_LAYER_NORM_EPS)
	assert metrics['layer_norm_feature_std'] == pytest.approx(layer_norm_scale)
	assert metrics['layer_norm_effective_rank'] == pytest.approx(1.0)


def test_representation_metrics_record_zero_rank_and_reject_nonfinite() -> None:
	metrics = representation_metrics(np.ones((4, 3), dtype=np.float16))

	assert metrics['raw_feature_effective_rank'] == 0.0
	assert metrics['layer_norm_effective_rank'] == 0.0
	with pytest.raises(FloatingPointError, match='non-finite'):
		representation_metrics(np.array([[0.0], [np.nan]], dtype=np.float64))


def test_checkpoint_identity_requires_complete_candidate_and_random_contract() -> None:
	config = {
		'barlow_twins': {'method': 'local_barlow_twins_3d'},
		'augmentations': {'policy': 'overlapping_subcrop_xy_v1'},
	}
	candidate = {
		'epoch': 10,
		'global_step': 6_250,
		'training_state': {'completed_epoch': True},
	}
	random = {
		'epoch': 0,
		'global_step': 0,
		'training_state': {'checkpoint_kind': 'random_init'},
		'metadata': {
			'random_encoder_baseline': True,
			'pretrained_weights_loaded': False,
			'seed': 42,
		},
	}

	candidate_identity = diagnostics._validate_checkpoint_identity(  # noqa: SLF001
		candidate,
		checkpoint_config=config,
		random_baseline=False,
		expected_candidate_epoch=10,
		expected_candidate_global_step=6_250,
		expected_random_seed=42,
	)
	random_identity = diagnostics._validate_checkpoint_identity(  # noqa: SLF001
		random,
		checkpoint_config={},
		random_baseline=True,
		expected_candidate_epoch=10,
		expected_candidate_global_step=6_250,
		expected_random_seed=42,
	)

	assert candidate_identity['bare_encoder_state'] is True
	assert random_identity['random_seed'] == 42
	with pytest.raises(ValueError, match='10-epoch/6250-step'):
		diagnostics._validate_checkpoint_identity(  # noqa: SLF001
			{**candidate, 'epoch': 9},
			checkpoint_config=config,
			random_baseline=False,
			expected_candidate_epoch=10,
			expected_candidate_global_step=6_250,
			expected_random_seed=42,
		)


def _write_diagnostic_source(
	tmp_path: Path,
) -> tuple[EmbeddingRepresentationSource, str]:
	checkpoint = tmp_path / 'latest.pt'
	checkpoint.write_bytes(b'checkpoint fixture')
	embeddings = tmp_path / 'survey.embeddings.npy'
	valid_tokens = tmp_path / 'survey.valid_tokens.npy'
	metadata = tmp_path / 'survey.embedding_metadata.json'
	np.save(
		embeddings,
		np.array(
			[
				[[[1.0, 0.0]], [[-1.0, 0.0]]],
				[[[0.0, 1.0]], [[0.0, -1.0]]],
			],
			dtype=np.float16,
		),
		allow_pickle=False,
	)
	np.save(valid_tokens, np.ones((2, 2, 1), dtype=bool), allow_pickle=False)
	metadata.write_text(
		json.dumps(
			{
				'survey_id': 'survey',
				'checkpoint_path': str(checkpoint),
				'checkpoint_sha256': file_sha256(checkpoint),
				'token_grid_shape': [2, 2, 1],
				'output_dtype': 'float16',
				'model_geometry': {'encoder_dim': 2},
			}
		),
		encoding='utf-8',
	)
	return (
		EmbeddingRepresentationSource(
			source_id='candidate',
			survey_id='survey',
			checkpoint_path=checkpoint,
			embeddings_path=embeddings,
			valid_tokens_path=valid_tokens,
			metadata_path=metadata,
			random_baseline=False,
		),
		file_sha256(valid_tokens),
	)


def test_diagnostic_payload_binds_fixed_sample_and_all_source_hashes(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	source, valid_sha256 = _write_diagnostic_source(tmp_path)
	payload = {
		'epoch': 10,
		'global_step': 6_250,
		'training_state': {'completed_epoch': True},
	}
	config = {
		'barlow_twins': {'method': 'local_barlow_twins_3d'},
		'augmentations': {'policy': 'overlapping_subcrop_xy_v1'},
	}
	monkeypatch.setattr(
		diagnostics,
		'load_checkpoint',
		lambda *_args, **_kwargs: payload,
	)
	monkeypatch.setattr(
		diagnostics,
		'checkpoint_config_from_payload',
		lambda _payload: config,
	)

	result = build_embedding_representation_diagnostics(
		source,
		expected_token_grid_shape=(2, 2, 1),
		expected_embedding_dim=2,
		expected_valid_mask_sha256=valid_sha256,
		sample_size=4,
	)

	assert tuple(result['metrics']) == REPRESENTATION_METRIC_KEYS
	assert result['sampling']['sample_size'] == 4
	assert (
		result['sampling']['sample_flat_indices_sha256']
		== hashlib.sha256(np.arange(4, dtype='<i8').tobytes()).hexdigest()
	)
	assert result['provenance']['checkpoint']['sha256'] == file_sha256(
		source.checkpoint_path
	)
	assert result['provenance']['embeddings']['sha256'] == file_sha256(
		source.embeddings_path
	)
	assert result['calculation']['calculation_dtype'] == 'float64'


def test_diagnostic_rejects_embedding_metadata_checkpoint_sha_mismatch(
	tmp_path: Path,
) -> None:
	source, valid_sha256 = _write_diagnostic_source(tmp_path)
	metadata = json.loads(source.metadata_path.read_text(encoding='utf-8'))
	metadata['checkpoint_sha256'] = '0' * 64
	source.metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

	with pytest.raises(ValueError, match='metadata checkpoint SHA-256'):
		build_embedding_representation_diagnostics(
			source,
			expected_token_grid_shape=(2, 2, 1),
			expected_embedding_dim=2,
			expected_valid_mask_sha256=valid_sha256,
			sample_size=4,
		)


def test_diagnostic_json_write_is_atomic_and_requires_exact_metric_keys(
	tmp_path: Path,
) -> None:
	metrics = dict.fromkeys(REPRESENTATION_METRIC_KEYS, 1.0)
	payload = {'metrics': metrics, 'source_id': 'candidate'}
	output = tmp_path / 'diagnostics/candidate.json'

	assert write_embedding_representation_diagnostics(output, payload) == output
	assert json.loads(output.read_text(encoding='utf-8')) == payload
	assert not list(output.parent.glob('*.tmp'))
	with pytest.raises(ValueError, match='exact five metric keys'):
		write_embedding_representation_diagnostics(
			tmp_path / 'invalid.json',
			{'metrics': {'raw_feature_norm': 1.0}},
		)


def _write_five_way_config(
	tmp_path: Path,
	filename: str,
) -> tuple[dict[str, object], Path]:
	mapping = build_five_way_universe(tmp_path / 'universe')
	path = tmp_path / filename
	path.write_text(yaml.safe_dump(mapping), encoding='utf-8')
	return mapping, path


@pytest.mark.parametrize(
	('filename', 'expected_model_id', 'expected_source_id', 'expected_source_kind'),
	[
		('random_medium.yaml', 'random', 'random', 'random_baseline'),
		(
			'candidate_a_medium.yaml',
			'local_barlow_twins',
			'candidate_a',
			'candidate',
		),
	],
)
def test_poc_diagnostic_cli_infers_source_from_existing_downstream_config(
	tmp_path: Path,
	filename: str,
	expected_model_id: str,
	expected_source_id: str,
	expected_source_kind: str,
) -> None:
	mapping, path = _write_five_way_config(tmp_path, filename)
	config = f3_lithology_five_way_config_from_mapping(mapping)

	source = diagnostic_cli.representation_source_from_config(
		config,
		config_path=path,
	)

	assert source.source_id == expected_source_id
	actual_source_kind = 'random_baseline' if source.random_baseline else 'candidate'
	assert actual_source_kind == expected_source_kind
	assert source.checkpoint_path == config.model_by_id(expected_model_id).checkpoint
	assert diagnostic_cli.representation_diagnostic_output_path(
		config,
		source_id=expected_source_id,
	).parts[-2:] == ('representation', f'{expected_source_id}.json')


def test_poc_diagnostic_cli_has_no_sampling_overrides_and_dry_run_writes_nothing(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	mapping, path = _write_five_way_config(tmp_path, 'candidate_a_medium.yaml')
	artifact_root = Path(mapping['paths']['artifact_root'])
	options = {
		option
		for action in diagnostic_cli.build_parser()._actions  # noqa: SLF001
		for option in action.option_strings
	}
	assert '--sample-size' not in options
	assert '--layer-norm-eps' not in options
	monkeypatch.setattr(
		sys,
		'argv',
		[
			'measure_f3_overlap_subcrop_representation.py',
			'--config',
			str(path),
			'--dry-run',
		],
	)

	diagnostic_cli.main()

	stdout = capsys.readouterr().out
	assert 'sample_size: 8192' in stdout
	assert diagnostic_cli.F3_SAMPLE_FLAT_INDICES_SHA256 in stdout
	assert 'execution: dry-run; no files written' in stdout
	assert not (artifact_root / 'diagnostics').exists()
