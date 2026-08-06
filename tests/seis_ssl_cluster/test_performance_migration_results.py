from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
	from pathlib import Path

from seis_ssl_cluster.migration.performance_results import (
	COMPLETION_MANIFEST_NAME,
	classify_metadata_diff,
	commit_staged_artifact_directory,
	compare_numeric_arrays,
	decide_migration_status,
	publish_lightweight_migration_results,
	quarantine_artifact,
	reuse_or_quarantine_artifact,
	staged_artifact_directory,
	validate_completion_manifest,
	write_completion_manifest,
)

CURRENT_SHA = 'a' * 40
HISTORICAL_SHA = 'b' * 40


def test_migration_decision_priority_and_reuse_policy() -> None:
	common = _exact_checks()
	assert decide_migration_status(**common).status == 'PASS_REUSE_EXISTING'
	assert (
		decide_migration_status(**(common | {'numeric_drift': True})).status
		== 'PASS_WITH_NUMERIC_DRIFT'
	)
	assert (
		decide_migration_status(**(common | {'probe_predictions_exact': False})).status
		== 'REEXTRACT_REQUIRED'
	)
	assert (
		decide_migration_status(**(common | {'hmm_labels_exact': False})).status
		== 'REBUILD_M1_REQUIRED'
	)
	assert (
		decide_migration_status(
			**(
				common
				| {
					'blocking_numeric_contract': True,
					'hmm_labels_exact': False,
				}
			)
		).status
		== 'BLOCKED_NUMERIC_CONTRACT'
	)


def test_metadata_diff_separates_scientific_runtime_path_and_environment() -> None:
	historical = {
		'checkpoint_sha256': 'same',
		'window_size': [128, 128, 128],
		'batch_size': 1,
		'output_dir': '/historical/out',
		'environment': {'hostname': 'old-host'},
	}
	current = {
		'checkpoint_sha256': 'drift',
		'window_size': [128, 128, 128],
		'batch_size': 4,
		'output_dir': '/current/out',
		'environment': {'hostname': 'new-host'},
	}

	diff = classify_metadata_diff(historical, current)

	assert [item.path for item in diff.scientific] == ['checkpoint_sha256']
	assert [item.path for item in diff.performance] == ['batch_size']
	assert [item.path for item in diff.path_only] == ['output_dir']
	assert [item.path for item in diff.environment] == ['environment.hostname']
	assert diff.has_scientific_drift


def test_numeric_comparison_excludes_invalid_tokens_and_records_drift() -> None:
	historical = np.asarray([[1.0, 2.0], [100.0, 100.0]], dtype=np.float16)
	current = np.asarray([[1.0, 2.25], [0.0, 0.0]], dtype=np.float16)

	comparison = compare_numeric_arrays(
		historical,
		current,
		valid_mask=np.asarray([True, False]),
	)

	assert not comparison.exact_equal
	assert comparison.valid_element_count == 2
	assert comparison.invalid_element_count == 2
	assert comparison.different_element_count == 1
	assert comparison.different_element_fraction == 0.5
	assert comparison.max_absolute_error == 0.25
	assert comparison.mean_absolute_error == 0.125
	assert comparison.nan_count_historical == 0
	assert comparison.nonfinite_mismatch_count == 0


def test_numeric_comparison_records_nonfinite_mismatch() -> None:
	comparison = compare_numeric_arrays(
		np.asarray([1.0, np.nan, np.inf]),
		np.asarray([1.0, np.nan, -np.inf]),
	)

	assert not comparison.array_equal
	assert comparison.nan_count_historical == 1
	assert comparison.nan_count_current == 1
	assert comparison.inf_count_historical == 1
	assert comparison.inf_count_current == 1
	assert comparison.nonfinite_mismatch_count == 1


def test_completion_manifest_roundtrip_reuse_and_invalid_quarantine(
	tmp_path: Path,
) -> None:
	artifact = tmp_path / 'artifact'
	artifact.mkdir()
	required = artifact / 'report.json'
	required.write_text('{"status": "ok"}\n', encoding='utf-8')
	write_completion_manifest(
		artifact,
		artifact_type='migration_test',
		schema_version=1,
		required_files=('report.json',),
		current_git_sha=CURRENT_SHA,
		historical_baseline_sha=HISTORICAL_SHA,
	)

	payload = validate_completion_manifest(
		artifact,
		expected_artifact_type='migration_test',
		expected_schema_version=1,
		expected_current_git_sha=CURRENT_SHA,
		expected_historical_baseline_sha=HISTORICAL_SHA,
		required_files=('report.json',),
	)
	assert payload['status'] == 'COMPLETE'
	reused = reuse_or_quarantine_artifact(
		artifact,
		expected_artifact_type='migration_test',
		expected_schema_version=1,
		expected_current_git_sha=CURRENT_SHA,
		expected_historical_baseline_sha=HISTORICAL_SHA,
		required_files=('report.json',),
	)
	assert reused.action == 'REUSED'

	required.write_text('{"status": "corrupt"}\n', encoding='utf-8')
	quarantined = reuse_or_quarantine_artifact(
		artifact,
		expected_artifact_type='migration_test',
		expected_schema_version=1,
		expected_current_git_sha=CURRENT_SHA,
		expected_historical_baseline_sha=HISTORICAL_SHA,
		required_files=('report.json',),
	)
	assert quarantined.action == 'QUARANTINED'
	assert quarantined.quarantine_path is not None
	assert (quarantined.quarantine_path / COMPLETION_MANIFEST_NAME).is_file()
	assert not artifact.exists()


def test_staged_directory_commits_only_after_explicit_commit(tmp_path: Path) -> None:
	final = tmp_path / 'final'
	with staged_artifact_directory(final) as staging:
		(staging / 'report.md').write_text('# staged\n', encoding='utf-8')
		assert staging.exists()
		assert not final.exists()
		committed = commit_staged_artifact_directory(staging, final)

	assert committed == final
	assert (final / 'report.md').is_file()


def test_quarantine_uses_timestamped_preserving_rename(tmp_path: Path) -> None:
	path = tmp_path / 'partial'
	path.mkdir()
	(path / 'evidence.txt').write_text('preserve', encoding='utf-8')

	quarantined = quarantine_artifact(
		path,
		reason='partial artifact',
		timestamp=datetime(2026, 7, 17, tzinfo=timezone.utc),
	)

	assert quarantined.name == 'partial.quarantine_20260717T000000Z_partial_artifact'
	assert (quarantined / 'evidence.txt').read_text(encoding='utf-8') == 'preserve'


def test_lightweight_publish_writes_exact_files_without_manifest(
	tmp_path: Path,
) -> None:
	source_root = tmp_path / 'artifacts'
	report = _write(source_root / 'reports' / 'summary.md', '# summary\n')
	metrics = _write(source_root / 'reports' / 'metrics.json', '{"f1": 1.0}\n')
	output = tmp_path / 'results' / 'migration'

	raw = _write(source_root / 'raw.npy', 'not actually an array')
	published_files = publish_lightweight_migration_results(
		summary_report=report,
		metrics_json=metrics,
		output_dir=output,
		source_artifact_root=source_root,
	)

	assert {item.relative_to(output).as_posix() for item in published_files} == {
		'summary.md',
		'tables/metrics.json',
	}
	assert (output / 'summary.md').read_bytes() == report.read_bytes()
	assert (output / 'tables/metrics.json').read_bytes() == metrics.read_bytes()
	assert not (output / raw.name).exists()
	assert not (output / 'publish_manifest.json').exists()
	(output / 'unlisted.md').write_text('not allowed', encoding='utf-8')
	with pytest.raises(ValueError, match='do not exactly match'):
		publish_lightweight_migration_results(
			summary_report=report,
			metrics_json=metrics,
			output_dir=output,
			source_artifact_root=source_root,
			allow_reuse=True,
		)

	with pytest.raises(FileNotFoundError, match='regular file'):
		publish_lightweight_migration_results(
			summary_report=source_root / 'missing.md',
			metrics_json=metrics,
			output_dir=tmp_path / 'results' / 'missing',
			source_artifact_root=source_root,
		)


def _exact_checks() -> dict[str, bool]:
	return {
		'blocking_numeric_contract': False,
		'valid_token_masks_exact': True,
		'probe_predictions_exact': True,
		'probe_confusion_matrix_exact': True,
		'primary_metrics_exact': True,
		'hmm_labels_exact': True,
		'pseudo_target_labels_exact': True,
		'pseudo_target_valid_tokens_exact': True,
		'confidence_threshold_crossing': False,
		'numeric_drift': False,
	}


def _write(path: Path, text: str) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(text, encoding='utf-8')
	return path
