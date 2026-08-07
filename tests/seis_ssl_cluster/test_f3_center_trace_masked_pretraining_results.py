"""Focused contracts for experiment-104 lightweight publication."""
# ruff: noqa: E501, SLF001, TC003

from __future__ import annotations

from pathlib import Path

import pytest

import seis_ssl_cluster.f3.center_trace_masked_pretraining_results as results


def test_review_config_is_closed_and_requires_all_paths(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'artifacts'
	workspace_root = tmp_path / 'workspace'
	artifact_root.mkdir()
	workspace_root.mkdir()
	handoff = artifact_root / 'handoff.json'
	handoff.write_text('{}\n', encoding='utf-8')
	mapping = {
		'artifact_root': str(artifact_root),
		'workspace_root': str(workspace_root),
		'pretraining_handoff': str(handoff),
		'output_dir': str(workspace_root / 'review'),
	}

	config = results.f3_center_trace_masked_pretraining_review_config_from_mapping(
		mapping
	)
	assert config.pretraining_handoff == handoff.resolve()
	with pytest.raises(ValueError, match='unknown center-trace'):
		results.f3_center_trace_masked_pretraining_review_config_from_mapping(
			{**mapping, 'extra': 'value'}
		)


def _publication_fixture(tmp_path: Path) -> tuple[object, dict[str, object], dict[str, object]]:
	artifact_root = tmp_path / 'artifacts'
	workspace_root = tmp_path / 'workspace'
	artifact_root.mkdir()
	workspace_root.mkdir()
	handoff = artifact_root / 'handoff.json'
	handoff.write_text('{}\n', encoding='utf-8')
	config = results.F3CenterTraceMaskedPretrainingReviewConfig(
		artifact_root=artifact_root,
		workspace_root=workspace_root,
		pretraining_handoff=handoff,
		output_dir=workspace_root / 'review',
	)
	selection_path = artifact_root / 'selection.json'
	selection_path.write_text('{}\n', encoding='utf-8')
	diagnostics_path = artifact_root / 'diagnostics.csv'
	diagnostics_path.write_text('source\nfixture\n', encoding='utf-8')
	row = {'epoch': 1, 'global_step': 1}
	row.update(dict.fromkeys(results._DIAGNOSTIC_FIELDS, 1.0))
	row['loss_consistency_contribution'] = 0.0
	portable = {
		'status': 'PASS',
		'execution': {'git_sha': 'a' * 40, 'dirty_status': []},
		'full_run': {'final_epoch': 25, 'final_global_step': 25600},
		'selected_checkpoint': {
			'kind': 'epoch',
			'epoch': 25,
			'global_step': 25600,
			'loss': 1.0,
			'sha256': 'b' * 64,
		},
		'embedding': {
			'embeddings_shape': [76, 113, 32, 384],
			'embeddings_dtype': 'float16',
			'finite_valid_count': 1,
		},
		'execution_counts': {
			'training': {'fresh': 1, 'resume': 0},
			'embedding': {'fresh': 1, 'reuse': 0},
		},
		'pass_handoff': {'sha256': 'c' * 64},
		'training_metrics': {
			'ranges': {
				field: {'min': 1.0, 'max': 1.0}
				for field in results._DIAGNOSTIC_FIELDS
			}
		},
	}
	live = {
		'selection': {},
		'selection_path': selection_path,
		'diagnostics': [row],
		'diagnostics_path': diagnostics_path,
	}
	return config, portable, live


def test_publication_rewrites_exact_lightweight_file_set(tmp_path: Path) -> None:
	config, portable, live = _publication_fixture(tmp_path)
	first = results._publish_review(
		config, portable=portable, live=live, handoff={}
	)
	paths = [config.output_dir / name for name in results.OUTPUT_NAMES]
	first_state = {path: path.read_bytes() for path in paths}

	second = results._publish_review(
		config, portable=portable, live=live, handoff={}
	)
	second_state = {path: path.read_bytes() for path in paths}
	assert first is None
	assert second is None
	assert first_state == second_state
	assert {path.name for path in config.output_dir.iterdir()} == set(results.OUTPUT_NAMES)
	assert not (config.output_dir / 'publish_manifest.json').exists()


def test_publication_preserves_top_level_legacy_manifest(tmp_path: Path) -> None:
	config, portable, live = _publication_fixture(tmp_path)
	results._publish_review(config, portable=portable, live=live, handoff={})
	legacy = config.output_dir / 'publish_manifest.json'
	legacy.write_bytes(b'{"legacy": true}\n')
	before_bytes = legacy.read_bytes()
	before_size = legacy.stat().st_size
	before_mtime_ns = legacy.stat().st_mtime_ns

	results._publish_review(config, portable=portable, live=live, handoff={})

	assert legacy.read_bytes() == before_bytes
	assert legacy.stat().st_size == before_size
	assert legacy.stat().st_mtime_ns == before_mtime_ns
	assert {path.name for path in config.output_dir.iterdir()} == {
		*results.OUTPUT_NAMES,
		'publish_manifest.json',
	}


@pytest.mark.parametrize(
	'extra_name',
	['raw_embeddings.npy', 'checkpoint.pt', 'unexpected.md'],
)
def test_publication_rejects_unallowlisted_owned_output(
	tmp_path: Path, extra_name: str
) -> None:
	config, portable, live = _publication_fixture(tmp_path)
	config.output_dir.mkdir()
	(config.output_dir / extra_name).write_bytes(b'raw')
	with pytest.raises(ValueError, match='unallowlisted'):
		results._publish_review(
			config, portable=portable, live=live, handoff={}
		)


def test_publication_rejects_nested_legacy_manifest(tmp_path: Path) -> None:
	config, portable, live = _publication_fixture(tmp_path)
	nested = config.output_dir / 'nested'
	nested.mkdir(parents=True)
	(nested / 'publish_manifest.json').write_bytes(b'{"legacy": true}\n')

	with pytest.raises(ValueError, match='unallowlisted'):
		results._publish_review(
			config, portable=portable, live=live, handoff={}
		)


def test_publication_rejects_legacy_manifest_symlink(tmp_path: Path) -> None:
	config, portable, live = _publication_fixture(tmp_path)
	config.output_dir.mkdir()
	target = tmp_path / 'legacy.json'
	target.write_bytes(b'{"legacy": true}\n')
	(config.output_dir / 'publish_manifest.json').symlink_to(target)

	with pytest.raises(ValueError, match='symlink'):
		results._publish_review(
			config, portable=portable, live=live, handoff={}
		)


def test_live_references_reject_missing_and_stale_but_allow_explicit_files(
	tmp_path: Path,
) -> None:
	config, _portable, _live = _publication_fixture(tmp_path)
	source = config.artifact_root / 'evidence.json'
	source.write_text('{}\n', encoding='utf-8')

	with pytest.raises(FileNotFoundError, match='missing'):
		results._validate_live_reference(
			{'path': str(config.artifact_root / 'missing.json'), 'sha256': 'a' * 64},
			label='missing evidence',
		)
	with pytest.raises(ValueError, match='SHA-256'):
		results._validate_live_reference(
			{'path': str(source), 'sha256': 'a' * 64},
			label='stale evidence',
		)
	foreign = tmp_path / 'foreign.json'
	foreign.write_text('{}\n', encoding='utf-8')
	assert results._validate_live_reference(
		{'path': str(foreign), 'sha256': results.file_sha256(foreign)},
		label='explicit evidence',
	) == foreign.resolve()


def test_portable_review_paths_preserve_hashes_without_local_roots(
	tmp_path: Path,
) -> None:
	artifact_root = tmp_path / 'artifacts'
	workspace_root = tmp_path / 'workspace'
	artifact_root.mkdir()
	workspace_root.mkdir()
	config = results.F3CenterTraceMaskedPretrainingReviewConfig(
		artifact_root=artifact_root,
		workspace_root=workspace_root,
		pretraining_handoff=artifact_root / 'handoff.json',
		output_dir=workspace_root / 'review',
	)
	digest = 'a' * 64
	foreign = tmp_path / 'foreign' / 'summary.json'
	payload = results._portable_value(
		{
			'artifact_root': artifact_root,
			'artifact_child': artifact_root / 'pretraining' / 'metrics.csv',
			'workspace_root': workspace_root,
			'workspace_child': workspace_root / 'results' / 'summary.json',
			'sha256': digest,
			'foreign': foreign,
		},
		config=config,
	)
	assert payload == {
		'artifact_root': '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}',
		'artifact_child': '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/metrics.csv',
		'workspace_root': '.',
		'workspace_child': 'results/summary.json',
		'sha256': digest,
		'foreign': str(foreign),
	}


def test_inspect_live_evidence_binds_diagnostics_to_handoff_hash_and_root(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	artifact_root = tmp_path / 'artifacts'
	workspace_root = tmp_path / 'workspace'
	artifact_root.mkdir()
	workspace_root.mkdir()
	handoff_path = artifact_root / 'handoff.json'
	handoff_path.write_text('{}\n', encoding='utf-8')
	checkpoint_root = artifact_root / 'pretraining'
	checkpoint_root.mkdir()
	selected_path = checkpoint_root / 'best.pt'
	latest_path = checkpoint_root / 'latest.pt'
	selection_path = checkpoint_root / 'checkpoint_selection_summary.json'
	diagnostics_path = checkpoint_root / 'multi_head_epoch_metrics.csv'
	selected_path.write_bytes(b'selected')
	latest_path.write_bytes(b'latest')
	selection_path.write_text('{}\n', encoding='utf-8')
	diagnostics_path.write_text('diagnostics\n', encoding='utf-8')
	(checkpoint_root / 'run_metadata.json').write_text(
		'{"execution_counts": {"fresh": 1, "resume": 0}}\n',
		encoding='utf-8',
	)
	embedding_root = artifact_root / 'embeddings'
	embedding_root.mkdir()
	(embedding_root / results.EMBEDDING_EXECUTION_JSON).write_text(
		'{"artifact_type": "embedding_extraction_execution", '
		'"schema_version": 1, "fresh": 1, "reuse": 0, "survey_count": 1}\n',
		encoding='utf-8',
	)
	selected_payload = {
		'epoch': 25,
		'global_step': 25600,
		'metrics': {'loss': 1.0},
		'training_state': {'checkpoint_kind': 'epoch'},
	}
	latest_payload = {
		'epoch': 25,
		'global_step': 25600,
		'checkpoint_selection': {},
	}
	handoff = {
		'targets': {},
		'checkpoint': {
			'path': str(selected_path),
			'sha256': results.file_sha256(selected_path),
			'latest_path': str(latest_path),
			'latest_sha256': results.file_sha256(latest_path),
			'selection_history_sha256': 's' * 64,
		},
		'training_diagnostics': {
			'path': str(diagnostics_path),
			'sha256': results.file_sha256(diagnostics_path),
		},
	}
	monkeypatch.setattr(results, '_validate_source_references', lambda *_args: None)
	monkeypatch.setattr(
		results,
		'_torch_mapping',
		lambda path, _label: selected_payload
		if path == selected_path
		else latest_payload,
	)
	monkeypatch.setattr(results, '_validate_checkpoint_lineage', lambda **_kwargs: None)
	monkeypatch.setattr(results, 'checkpoint_selection_sha256', lambda _value: 's' * 64)
	monkeypatch.setattr(
		results,
		'_read_diagnostics',
		lambda _path: [
			{'epoch': epoch, 'global_step': epoch * 1024} for epoch in range(1, 26)
		],
	)
	monkeypatch.setattr(
		results,
		'_embedding_evidence',
		lambda *_args, **_kwargs: {'root': embedding_root, 'survey_count': 1},
	)

	live = results._inspect_live_evidence(handoff=handoff)
	assert live['diagnostics_reference'] == handoff['training_diagnostics']
	assert live['training_execution'] == {'fresh': 1, 'resume': 0}
	assert live['embedding_execution'] == {'fresh': 1, 'reuse': 0}

	diagnostics_path.write_bytes(b'tampered')
	with pytest.raises(ValueError, match='does not match live bytes'):
		results._inspect_live_evidence(handoff=handoff)

	foreign = artifact_root / 'foreign.csv'
	foreign.write_bytes(b'foreign')
	foreign_handoff = {
		**handoff,
		'training_diagnostics': {
			'path': str(foreign),
			'sha256': results.file_sha256(foreign),
		},
	}
	with pytest.raises(ValueError, match='does not bind selected checkpoint'):
		results._inspect_live_evidence(handoff=foreign_handoff)
