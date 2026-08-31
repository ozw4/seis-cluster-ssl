'''Tests for Volve five-way checkpoint and embedding preflight.'''

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml

from proc.seis_ssl_cluster import audit_volve_horizon_five_way_sources as cli
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.volve.horizon_five_way_config import (
	FIVE_WAY_MODEL_IDS,
	volve_horizon_five_way_config_from_mapping,
)
from seis_ssl_cluster.volve.horizon_five_way_sources import (
	FIVE_WAY_STAGE2_GLOBAL_STEPS,
	audit_volve_horizon_five_way_sources,
	inspect_volve_horizon_five_way_embedding_suite,
	plan_volve_horizon_five_way_embeddings,
	plan_volve_horizon_five_way_sources,
)
from tests.seis_ssl_cluster.helpers_volve_five_way import (
	five_way_config_mapping,
	five_way_embedding_sentinel,
	load_checkpoint,
	save_checkpoint,
	write_five_way_universe,
	write_json,
)


def test_source_plan_is_static_when_artifacts_are_missing(tmp_path: Path) -> None:
	config = volve_horizon_five_way_config_from_mapping(
		five_way_config_mapping(tmp_path)
	)

	sources = plan_volve_horizon_five_way_sources(config)
	embeddings = plan_volve_horizon_five_way_embeddings(config)

	assert tuple(row['model_id'] for row in sources) == FIVE_WAY_MODEL_IDS
	assert tuple(row['model_id'] for row in embeddings) == FIVE_WAY_MODEL_IDS
	assert not config.artifact_root.exists()


def test_checkpoint_audit_passes_and_reports_fixed_budgets(tmp_path: Path) -> None:
	universe = write_five_way_universe(tmp_path, embeddings=False)
	report = audit_volve_horizon_five_way_sources(universe['config'])
	sources = cast('list[dict[str, object]]', report['sources'])

	assert report['model_order'] == list(FIVE_WAY_MODEL_IDS)
	assert sources[0]['stage_2'] == {
		'epochs': 25,
		'global_steps': FIVE_WAY_STAGE2_GLOBAL_STEPS,
		'unfreeze_top_blocks': 1,
	}
	assert sources[0]['parent_checkpoint_sha256'] == sources[1][
		'parent_checkpoint_sha256'
	]
	assert sources[2]['parent_checkpoint_sha256'] == sources[3][
		'parent_checkpoint_sha256'
	]
	assert sources[0]['parent_checkpoint_sha256'] != sources[2][
		'parent_checkpoint_sha256'
	]
	assert sources[-1]['stage_2'] is None


def test_checkpoint_audit_rejects_hmm_k_and_pseudo_source_swap(
	tmp_path: Path,
) -> None:
	universe = write_five_way_universe(tmp_path / 'k', embeddings=False)
	payload = load_checkpoint(universe['checkpoints']['mae_hmm_k6'])
	payload['stratigraphy_config']['head']['num_prototypes'] = 5
	save_checkpoint(universe['checkpoints']['mae_hmm_k6'], payload)
	with pytest.raises(ValueError, match='prototype count'):
		audit_volve_horizon_five_way_sources(universe['config'])

	universe = write_five_way_universe(tmp_path / 'swap', embeddings=False)
	metadata_path = universe['pseudo_metadata']['mae_hmm_k6']
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['source'] = {
		'checkpoint_path': str(universe['parents']['local']),
		'checkpoint_sha256': file_sha256(universe['parents']['local']),
	}
	write_json(metadata_path, metadata)
	with pytest.raises(ValueError, match='does not match its stage-1 parent'):
		audit_volve_horizon_five_way_sources(universe['config'])


def test_checkpoint_audit_rejects_local_identity_and_trace_drop(
	tmp_path: Path,
) -> None:
	universe = write_five_way_universe(tmp_path / 'method', embeddings=False)
	payload = load_checkpoint(universe['checkpoints']['local_barlow_twins'])
	payload['config']['barlow_twins']['local_pairs_per_crop'] = 64
	save_checkpoint(universe['checkpoints']['local_barlow_twins'], payload)
	with pytest.raises(ValueError, match='local_pairs_per_crop'):
		audit_volve_horizon_five_way_sources(universe['config'])

	universe = write_five_way_universe(tmp_path / 'trace', embeddings=False)
	payload = load_checkpoint(universe['checkpoints']['local_barlow_twins'])
	payload['config']['augmentations']['trace_drop_probability'] = 0.1
	save_checkpoint(universe['checkpoints']['local_barlow_twins'], payload)
	with pytest.raises(ValueError, match='must be disabled'):
		audit_volve_horizon_five_way_sources(universe['config'])


def test_checkpoint_audit_rejects_random_budget_and_parent_sha(
	tmp_path: Path,
) -> None:
	universe = write_five_way_universe(tmp_path / 'random', embeddings=False)
	payload = load_checkpoint(universe['checkpoints']['random'])
	payload['metadata']['seed'] = 7
	save_checkpoint(universe['checkpoints']['random'], payload)
	with pytest.raises(ValueError, match=r'metadata.seed'):
		audit_volve_horizon_five_way_sources(universe['config'])

	universe = write_five_way_universe(tmp_path / 'budget', embeddings=False)
	payload = load_checkpoint(universe['checkpoints']['mae'])
	payload['global_step'] -= 1
	save_checkpoint(universe['checkpoints']['mae'], payload)
	with pytest.raises(ValueError, match='global_step'):
		audit_volve_horizon_five_way_sources(universe['config'])

	universe = write_five_way_universe(tmp_path / 'sha', embeddings=False)
	payload = load_checkpoint(universe['checkpoints']['mae'])
	payload['continuation_lineage']['init_checkpoint_sha256'] = 'f' * 64
	save_checkpoint(universe['checkpoints']['mae'], payload)
	with pytest.raises(ValueError, match='does not match the parent file'):
		audit_volve_horizon_five_way_sources(universe['config'])


def test_checkpoint_audit_rejects_tampered_pseudo_target_file(
	tmp_path: Path,
) -> None:
	universe = write_five_way_universe(tmp_path, embeddings=False)
	payload = load_checkpoint(universe['checkpoints']['mae_hmm_k6'])
	identities = payload['control_identity']['input_identities']['pseudo_targets']
	labels_path = Path(identities[0]['labels']['path'])
	np.save(labels_path, np.ones((1, 1, 1), dtype=np.int16))

	with pytest.raises(ValueError, match='SHA-256 differs from its live file'):
		audit_volve_horizon_five_way_sources(universe['config'])


def test_checkpoint_audit_rejects_tampered_pseudo_source_chain(
	tmp_path: Path,
) -> None:
	universe = write_five_way_universe(tmp_path, embeddings=False)
	pseudo_metadata = json.loads(
		universe['pseudo_metadata']['mae_hmm_k6'].read_text(encoding='utf-8')
	)
	cluster_metadata_path = Path(pseudo_metadata['source']['source_metadata_path'])
	cluster_metadata = json.loads(cluster_metadata_path.read_text(encoding='utf-8'))
	embedding_metadata_path = Path(
		cluster_metadata['embedding_input']['metadata_path']
	)
	with embedding_metadata_path.open('a', encoding='utf-8') as stream:
		stream.write('\n')

	with pytest.raises(ValueError, match='source embedding metadata SHA-256'):
		audit_volve_horizon_five_way_sources(universe['config'])


def test_checkpoint_audit_rejects_wrong_pseudo_target_survey(
	tmp_path: Path,
) -> None:
	universe = write_five_way_universe(tmp_path, embeddings=False)
	metadata_path = universe['pseudo_metadata']['mae_hmm_k6']
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['survey_id'] = 'wrong_survey'
	write_json(metadata_path, metadata)

	with pytest.raises(ValueError, match='pseudo target survey_id'):
		audit_volve_horizon_five_way_sources(universe['config'])


def test_embedding_suite_accepts_distinct_objectives_on_shared_support(
	tmp_path: Path,
) -> None:
	universe = write_five_way_universe(tmp_path, embeddings=True)
	suite = inspect_volve_horizon_five_way_embedding_suite(universe['config'])

	assert tuple(suite.sources) == FIVE_WAY_MODEL_IDS
	assert suite.volume_shape_xyz == (16, 16, 800)
	assert suite.token_grid_shape_xyz == (2, 2, 100)
	assert suite.embedding_shape == (2, 2, 100, 384)
	assert suite.embedding_dim == 384
	assert suite.model_valid_lateral_mask.shape == (16, 16)
	assert len(suite.valid_tokens_sha256) == 64
	assert len({source.embeddings_sha256 for source in suite.sources.values()}) == 5
	for model_id, source in suite.sources.items():
		assert np.all(
			np.load(source.paths.embeddings)
			== five_way_embedding_sentinel(model_id)
		)
		assert source.embeddings_sha256 == file_sha256(source.paths.embeddings)
		assert source.metadata_sha256 == file_sha256(source.paths.metadata)
		assert source.valid_tokens_sha256 == file_sha256(source.paths.valid_tokens)
	assert suite.sources['mae'].metadata['pretraining_objective'] != (
		suite.sources['local_barlow_twins'].metadata['pretraining_objective']
	)


def test_embedding_suite_rejects_wrong_array_embedding_dimension(
	tmp_path: Path,
) -> None:
	universe = write_five_way_universe(tmp_path, embeddings=True)
	model = universe['config'].model_by_id('mae_hmm_k6')
	paths = output_paths(model.embeddings_dir, universe['config'].survey_id)
	np.save(paths.embeddings, np.zeros((2, 2, 100, 383), dtype=np.float16))

	with pytest.raises(ValueError, match='embedding array shape must equal'):
		inspect_volve_horizon_five_way_embedding_suite(universe['config'])


def test_embedding_suite_rejects_out_of_order_supplied_source_audit(
	tmp_path: Path,
) -> None:
	universe = write_five_way_universe(tmp_path, embeddings=True)
	report = audit_volve_horizon_five_way_sources(universe['config'])
	report['sources'] = list(reversed(report['sources']))

	with pytest.raises(ValueError, match='fixed model order'):
		inspect_volve_horizon_five_way_embedding_suite(
			universe['config'],
			source_audit=report,
		)


@pytest.mark.parametrize(
	('field', 'replacement', 'message'),
	[
		('token_grid_shape', [2, 2, 99], 'token_grid_shape'),
		('window_size', [64, 128, 128], 'window_size'),
		('overlap', [32, 64, 64], 'overlap'),
		('precision', {'amp_enabled': False}, 'precision'),
	],
)
def test_embedding_suite_rejects_shared_metadata_drift(
	tmp_path: Path,
	field: str,
	replacement: object,
	message: str,
) -> None:
	universe = write_five_way_universe(tmp_path, embeddings=True)
	path = universe['embedding_metadata']['mae_hmm_k6']
	metadata = json.loads(path.read_text(encoding='utf-8'))
	metadata[field] = replacement
	write_json(path, metadata)

	with pytest.raises((TypeError, ValueError), match=message):
		inspect_volve_horizon_five_way_embedding_suite(universe['config'])


def test_embedding_suite_rejects_mask_and_checkpoint_identity_drift(
	tmp_path: Path,
) -> None:
	universe = write_five_way_universe(tmp_path / 'mask', embeddings=True)
	paths = output_paths(
		universe['config'].model_by_id('mae_hmm_k6').embeddings_dir,
		universe['config'].survey_id,
	)
	mask = np.load(paths.valid_tokens)
	mask[0, 0, 0] = False
	np.save(paths.valid_tokens, mask)
	with pytest.raises(ValueError, match='valid-token mask differs'):
		inspect_volve_horizon_five_way_embedding_suite(universe['config'])

	universe = write_five_way_universe(tmp_path / 'checkpoint', embeddings=True)
	path = universe['embedding_metadata']['mae_hmm_k6']
	metadata = json.loads(path.read_text(encoding='utf-8'))
	metadata['checkpoint_sha256'] = '0' * 64
	write_json(path, metadata)
	with pytest.raises(ValueError, match='checkpoint SHA-256'):
		inspect_volve_horizon_five_way_embedding_suite(universe['config'])


def test_embedding_suite_rejects_canonical_input_content_drift(
	tmp_path: Path,
) -> None:
	universe = write_five_way_universe(tmp_path, embeddings=True)
	metadata_path = universe['embedding_metadata']['mae']
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	amplitude_path = Path(metadata['source_amplitude_path'])
	np.save(amplitude_path, np.ones((1,), dtype=np.float32))

	with pytest.raises(ValueError, match='canonical_amplitude_sha256'):
		inspect_volve_horizon_five_way_embedding_suite(universe['config'])


def test_audit_cli_dry_run_needs_no_artifacts_and_writes_nothing(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	raw = five_way_config_mapping(tmp_path)
	config_path = tmp_path / 'five_way.yaml'
	config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')
	before = {path.relative_to(tmp_path) for path in tmp_path.rglob('*')}
	monkeypatch.setattr(
		'sys.argv',
		['audit', '--config', str(config_path), '--dry-run'],
	)
	cli.main()
	after = {path.relative_to(tmp_path) for path in tmp_path.rglob('*')}
	payload = json.loads(capsys.readouterr().out)

	assert payload['execution'] == 'dry-run'
	assert payload['model_order'] == list(FIVE_WAY_MODEL_IDS)
	assert before == after
