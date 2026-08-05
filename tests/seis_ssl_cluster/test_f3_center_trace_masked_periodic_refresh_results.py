"""Focused contracts for experiment-107 lightweight publication."""
# ruff: noqa: CPY001, SLF001

from __future__ import annotations

import json
from pathlib import Path

import pytest

import seis_ssl_cluster.f3.center_trace_masked_periodic_refresh_results as results


def _config(tmp_path: Path) -> results.F3CenterTraceMaskedPeriodicRefreshReviewConfig:
	artifact_root = tmp_path / 'artifacts'
	workspace_root = tmp_path / 'workspace'
	artifact_root.mkdir()
	workspace_root.mkdir()
	validation_config = workspace_root / 'validate.yaml'
	validation_config.write_text('{}\n', encoding='utf-8')
	handoff = artifact_root / 'periodic_refresh_handoff.json'
	handoff.write_text('{}\n', encoding='utf-8')
	return results.F3CenterTraceMaskedPeriodicRefreshReviewConfig(
		artifact_root=artifact_root,
		workspace_root=workspace_root,
		validation_config=validation_config,
		pretraining_handoff=handoff,
		output_dir=workspace_root / 'results',
	)


def _publication_inputs() -> tuple[
	dict[str, object], dict[str, object], dict[str, object]
]:
	commit = 'a' * 40
	execution_state = {
		'git_commit': commit,
		'git_status_short': [],
		'git_diff_sha256': 'b' * 64,
	}
	evidence = {
		'artifact_type': results._REVIEW_TYPE,
		'schema_version': 1,
		'status': 'PASS',
		'model_tag': 'model',
		'variant': 'variant',
		'execution_git_state': {
			'before': execution_state,
			'after': execution_state,
		},
		'refresh': {
			'schedule': [2, 5, 8, 11, 14, 17, 20],
			'completed_refresh_count': 0,
			'completed_generation_count': 1,
			'completed_generations': [
				{
					'generation_id': 'refresh_0000_initial',
					'refresh_after_epoch': 0,
					'source_student_state_sha256': None,
					'manifest_sha256': 'c' * 64,
				}
			],
			'chain': {'path': 'chain.json', 'sha256': 'd' * 64},
		},
		'final_checkpoint': {
			'path': 'selected.pt',
			'sha256': 'e' * 64,
			'latest_path': 'latest.pt',
			'latest_sha256': 'e' * 64,
			'schema_version': 8,
			'epoch': 25,
			'global_step': 25600,
			'selected_event': {'epoch': 25, 'checkpoint_kind': 'epoch'},
			'selection_history_sha256': 'f' * 64,
		},
		'checkpoint_summary': {
			'artifact_type': 'checkpoint-summary',
			'schema_version': 1,
			'status': 'PASS',
		},
		'final_embedding': {
			'embeddings_shape': [76, 113, 32, 384],
			'embeddings_dtype': 'float16',
			'finite_valid_count': 1,
		},
		'pass_handoff': {'path': 'handoff.json', 'sha256': '0' * 64, 'status': 'PASS'},
		'downstream': {
			'original_split_ready': True,
			'decoder_jobs_executed': 0,
			'six_split_jobs_executed': 0,
		},
	}
	# The renderer only reads this summary subset in the publication safety test.
	live = {
		'events': [{'event_type': 'generation', 'status': 'complete'}],
		'generation_details': [
			{
				'generation_index': 0,
				'generation_id': 'refresh_0000_initial',
				'refresh_after_epoch': 0,
				'source_student_state_sha256': None,
				'manifest_path': 'generation.json',
				'manifest_sha256': 'c' * 64,
				'generation_content_sha256': '1' * 64,
				'active_target_manifest_path': 'target.json',
				'active_target_manifest_sha256': '2' * 64,
				'per_k': {},
			}
		],
	}
	return evidence, live, {}


def test_review_config_is_closed_and_requires_owned_paths(tmp_path: Path) -> None:
	config = _config(tmp_path)
	mapping = {
		'artifact_root': str(config.artifact_root),
		'workspace_root': str(config.workspace_root),
		'validation_config': str(config.validation_config),
		'pretraining_handoff': str(config.pretraining_handoff),
		'output_dir': str(config.output_dir),
	}
	resolved = (
		results.f3_center_trace_masked_periodic_refresh_review_config_from_mapping(
			mapping
		)
	)
	assert resolved == config
	with pytest.raises(ValueError, match='unknown periodic refresh'):
		results.f3_center_trace_masked_periodic_refresh_review_config_from_mapping(
			{**mapping, 'extra': 'value'}
		)


def test_publication_reuses_exact_content_without_rewriting(tmp_path: Path) -> None:
	config = _config(tmp_path)
	evidence, live, handoff = _publication_inputs()
	first = results._publish_review(
		config,
		evidence=evidence,
		handoff=handoff,
		live=live,
		quarantine_invalid=False,
	)
	paths = [
		config.output_dir / name
		for name in (*results.OUTPUT_NAMES, 'publish_manifest.json')
	]
	first_state = {
		path: (path.stat().st_mtime_ns, path.read_bytes()) for path in paths
	}
	second = results._publish_review(
		config,
		evidence=evidence,
		handoff=handoff,
		live=live,
		quarantine_invalid=False,
	)
	second_state = {
		path: (path.stat().st_mtime_ns, path.read_bytes()) for path in paths
	}
	assert first.manifest_path == second.manifest_path
	assert first_state == second_state
	assert {path.name for path in config.output_dir.iterdir()} == {
		*results.OUTPUT_NAMES,
		'publish_manifest.json',
	}


@pytest.mark.parametrize(
	('field', 'value'),
	[
		('source_artifact_root', '/foreign/artifacts'),
		('output_dir', 'foreign-results'),
	],
)
def test_publication_does_not_reuse_foreign_manifest_identity(
	tmp_path: Path, field: str, value: str
) -> None:
	config = _config(tmp_path)
	evidence, live, handoff = _publication_inputs()
	results._publish_review(
		config,
		evidence=evidence,
		handoff=handoff,
		live=live,
		quarantine_invalid=False,
	)
	manifest_path = config.output_dir / 'publish_manifest.json'
	payload = json.loads(manifest_path.read_text(encoding='utf-8'))
	payload[field] = value
	manifest_path.write_text(json.dumps(payload) + '\n', encoding='utf-8')

	with pytest.raises(ValueError, match='stale or partial'):
		results._publish_review(
			config,
			evidence=evidence,
			handoff=handoff,
			live=live,
			quarantine_invalid=False,
		)


def test_publication_rejects_raw_output_and_explicitly_quarantines_it(
	tmp_path: Path,
) -> None:
	config = _config(tmp_path)
	evidence, live, handoff = _publication_inputs()
	config.output_dir.mkdir()
	(config.output_dir / 'checkpoint.pt').write_bytes(b'raw')
	with pytest.raises(ValueError, match='unallowlisted'):
		results._publish_review(
			config,
			evidence=evidence,
			handoff=handoff,
			live=live,
			quarantine_invalid=False,
		)
	results._publish_review(
		config,
		evidence=evidence,
		handoff=handoff,
		live=live,
		quarantine_invalid=True,
	)
	quarantined = list(config.output_dir.parent.glob('results.quarantine.*'))
	assert len(quarantined) == 1
	assert (quarantined[0] / 'checkpoint.pt').read_bytes() == b'raw'
	assert (config.output_dir / results.SUMMARY_JSON).is_file()


def test_publication_failure_does_not_leave_canonical_partial_tree(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _config(tmp_path)
	evidence, live, handoff = _publication_inputs()

	def fail_after_partial_stage(**kwargs: object) -> object:
		staging = kwargs['output_dir']
		assert isinstance(staging, Path)
		(staging / 'partial.json').write_text('partial\n', encoding='utf-8')
		raise RuntimeError('simulated publication failure')

	monkeypatch.setattr(
		results, 'publish_selected_results', fail_after_partial_stage
	)
	with pytest.raises(RuntimeError, match='simulated publication failure'):
		results._publish_review(
			config,
			evidence=evidence,
			handoff=handoff,
			live=live,
			quarantine_invalid=False,
		)

	assert not config.output_dir.exists()
	assert list(config.output_dir.parent.glob('.results.staging-*')) == []


def test_portable_value_removes_local_artifact_and_workspace_roots(
	tmp_path: Path,
) -> None:
	config = _config(tmp_path)
	assert results._portable_value(
		{
			'artifact': config.artifact_root / 'pretraining' / 'latest.pt',
			'workspace': config.workspace_root / 'results' / 'summary.json',
			'hash': 'a' * 64,
		},
		config=config,
	) == {
		'artifact': '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/latest.pt',
		'workspace': 'results/summary.json',
		'hash': 'a' * 64,
	}
