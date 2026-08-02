"""Immutable source-hard-only unanimous XY target export contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from proc.seis_ssl_cluster import (
	export_strat_hmm_multi_head_xy_neighbor_unanimous_targets as unanimous_export_cli,
)
from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus import (
	smooth_xy_neighbor_unanimous_hard_labels,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus_targets import (
	MultiHeadXYNeighborConsensusTargetExportConfig,
	export_multi_head_xy_neighbor_consensus_targets,
	load_multi_head_xy_neighbor_consensus_target_manifest,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_unanimous_targets import (
	ARTIFACT_TYPE,
	TARGET_REPRESENTATION,
	TARGET_SEMANTICS,
	MultiHeadXYNeighborUnanimousTargetExportConfig,
	export_multi_head_xy_neighbor_unanimous_targets,
	load_multi_head_xy_neighbor_unanimous_target_manifest,
	plan_multi_head_xy_neighbor_unanimous_target_exports,
	resolve_multi_head_xy_neighbor_unanimous_target_export_config,
)
from tests.seis_ssl_cluster.test_strat_multi_head_xy_neighbor_consensus_targets import (
	_hard_manifest,
)


def test_export_replays_unanimous_source_labels_and_preserves_invalid_bytes(
	tmp_path: Path,
) -> None:
	hard_manifest = _hard_manifest(tmp_path)
	config = _config(tmp_path, hard_manifest)

	plans = export_multi_head_xy_neighbor_unanimous_targets(config)

	assert [plan.action for plan in plans] == ['NEW', 'NEW', 'NEW']
	payload = load_multi_head_xy_neighbor_unanimous_target_manifest(
		config.handoff_manifest
	)
	assert payload['artifact_type'] == ARTIFACT_TYPE
	assert payload['target_representation'] == TARGET_REPRESENTATION
	assert payload['target_semantics'] == TARGET_SEMANTICS
	assert payload['smoothing'] == {
		'neighborhood': 'same_z_xy_four_neighbors',
		'neighbor_order': ['x_minus', 'x_plus', 'y_minus', 'y_plus'],
		'four_valid_neighbors_minimum_agreement': 4,
		'three_valid_neighbors_minimum_agreement': 3,
		'fewer_than_three_valid_neighbors': 'unchanged',
		'tied_or_nonunique_consensus': 'unchanged',
		'center_matching_consensus': 'unchanged',
		'temporal_guard': 'internal_valid_token_source_label_bounds',
		'application': 'single_pass_synchronous_source_labels',
	}
	hard = json.loads(hard_manifest.read_text(encoding='utf-8'))

	for k in (6, 8, 10):
		entry = payload['heads'][str(k)]['surveys']['survey']  # type: ignore[index]
		source = hard['heads'][str(k)]['surveys']['survey']
		source_labels = np.load(source['labels']['path'], allow_pickle=False)
		source_valid = np.load(source['valid_tokens']['path'], allow_pickle=False)
		labels = np.load(entry['labels']['path'], allow_pickle=False)  # type: ignore[index]
		confidence = np.load(entry['confidence']['path'], allow_pickle=False)  # type: ignore[index]
		valid = np.load(entry['valid_tokens']['path'], allow_pickle=False)  # type: ignore[index]
		expected = smooth_xy_neighbor_unanimous_hard_labels(
			source_labels,
			source_valid,
		)
		np.testing.assert_array_equal(labels, expected.labels)
		np.testing.assert_array_equal(valid, source_valid)
		np.testing.assert_array_equal(labels[~valid], source_labels[~source_valid])
		assert labels.dtype == np.int32
		assert confidence.dtype == np.float32
		assert valid.dtype == np.bool_
		assert np.all(confidence[valid] == 1.0)
		assert np.all(confidence[~valid] == 0.0)
		metadata = json.loads(
			Path(entry['metadata']['path']).read_text(encoding='utf-8')  # type: ignore[index]
		)
		assert metadata['artifact_type'] == 'strat_hmm_pseudo_target'
		assert metadata['schema_version'] == 1
		assert metadata['source']['target_representation'] == TARGET_REPRESENTATION
		assert metadata['source']['target_semantics'] == TARGET_SEMANTICS
		diagnostics = payload['heads'][str(k)]['diagnostics']['aggregate']  # type: ignore[index]
		assert diagnostics['changed_token_count'] > 0
		assert diagnostics['ordered_path']['violation_count'] == 0
		assert diagnostics['empty_output_state_count'] == 0
		assert set(diagnostics['consensus_decisions']) == {
			'neighbor_count_histogram',
			'consensus_token_count',
			'change_candidate_count',
			'internal_valid_token_count',
			'order_compatible_candidate_count',
			'changed_token_count',
		}


def test_new_and_old_manifest_loaders_are_strictly_disjoint(tmp_path: Path) -> None:
	hard_manifest = _hard_manifest(tmp_path)
	unanimous = _config(tmp_path, hard_manifest)
	export_multi_head_xy_neighbor_unanimous_targets(unanimous)

	with pytest.raises(ValueError, match='unsupported XY-neighbor-consensus'):
		load_multi_head_xy_neighbor_consensus_target_manifest(
			unanimous.handoff_manifest
		)

	consensus = MultiHeadXYNeighborConsensusTargetExportConfig(
		source_hard_manifest=hard_manifest,
		output_root=tmp_path / 'consensus',
		handoff_manifest=tmp_path / 'consensus' / 'handoff.json',
	)
	export_multi_head_xy_neighbor_consensus_targets(consensus)
	with pytest.raises(ValueError, match='unsupported XY-neighbor-unanimous'):
		load_multi_head_xy_neighbor_unanimous_target_manifest(
			consensus.handoff_manifest
		)


def test_full_loader_replays_arrays_while_reference_only_loader_stays_lazy(
	tmp_path: Path,
) -> None:
	hard_manifest = _hard_manifest(tmp_path)
	config = _config(tmp_path, hard_manifest)
	export_multi_head_xy_neighbor_unanimous_targets(config)
	payload = json.loads(config.handoff_manifest.read_text(encoding='utf-8'))
	entry = payload['heads']['6']['surveys']['survey']
	labels_path = Path(entry['labels']['path'])
	labels = np.load(labels_path, allow_pickle=False)
	labels[1, 1, 5] = 4
	np.save(labels_path, labels, allow_pickle=False)
	entry['labels']['sha256'] = file_sha256(labels_path)
	head_path = config.output_root / 'bundle/k6/head_metadata.json'
	head_path.write_text(
		json.dumps(payload['heads']['6']),
		encoding='utf-8',
	)
	config.handoff_manifest.write_text(json.dumps(payload), encoding='utf-8')

	load_multi_head_xy_neighbor_unanimous_target_manifest(
		config.handoff_manifest,
		validate_array_semantics=False,
	)
	with pytest.raises(ValueError, match='smoothing replay result'):
		load_multi_head_xy_neighbor_unanimous_target_manifest(config.handoff_manifest)
	assert [
		plan.action
		for plan in plan_multi_head_xy_neighbor_unanimous_target_exports(
			config,
			only_missing=True,
		)
	] == ['QUARANTINE', 'QUARANTINE', 'QUARANTINE']


def test_only_missing_reuses_unchanged_files_and_mtimes(tmp_path: Path) -> None:
	hard_manifest = _hard_manifest(tmp_path)
	config = _config(tmp_path, hard_manifest)
	export_multi_head_xy_neighbor_unanimous_targets(config)
	paths = sorted(path for path in config.output_root.rglob('*') if path.is_file())
	before = {
		path: (file_sha256(path), path.stat().st_mtime_ns)
		for path in [*paths, config.handoff_manifest]
	}

	plans = export_multi_head_xy_neighbor_unanimous_targets(
		config,
		only_missing=True,
	)

	assert [plan.action for plan in plans] == ['REUSE', 'REUSE', 'REUSE']
	assert {
		path: (file_sha256(path), path.stat().st_mtime_ns)
		for path in before
	} == before


def test_orphan_handoff_is_quarantined_then_rebuilt(tmp_path: Path) -> None:
	hard_manifest = _hard_manifest(tmp_path)
	config = _config(tmp_path, hard_manifest)
	config.output_root.mkdir()
	config.handoff_manifest.write_text('{}\n', encoding='utf-8')

	plans = plan_multi_head_xy_neighbor_unanimous_target_exports(
		config,
		only_missing=True,
	)
	assert [plan.action for plan in plans] == ['QUARANTINE', 'QUARANTINE', 'QUARANTINE']
	export_multi_head_xy_neighbor_unanimous_targets(config, only_missing=True)

	assert config.handoff_manifest.is_file()
	assert list(config.output_root.glob('handoff.json.quarantine-*'))
	assert [
		plan.action
		for plan in plan_multi_head_xy_neighbor_unanimous_target_exports(
			config,
			only_missing=True,
		)
	] == ['REUSE', 'REUSE', 'REUSE']


def test_different_frozen_source_identity_is_an_error_not_quarantine(
	tmp_path: Path,
) -> None:
	hard_manifest = _hard_manifest(tmp_path)
	config = _config(tmp_path, hard_manifest)
	export_multi_head_xy_neighbor_unanimous_targets(config)
	other_source = tmp_path / 'other-hard-manifest.json'
	other_source.write_text(hard_manifest.read_text(encoding='utf-8'), encoding='utf-8')
	other_config = MultiHeadXYNeighborUnanimousTargetExportConfig(
		source_hard_manifest=other_source,
		output_root=config.output_root,
		handoff_manifest=config.handoff_manifest,
	)

	plans = plan_multi_head_xy_neighbor_unanimous_target_exports(
		other_config,
		only_missing=True,
	)

	assert [plan.action for plan in plans] == ['ERROR', 'ERROR', 'ERROR']
	assert all(plan.reason and 'identity differs' in plan.reason for plan in plans)


def test_existing_complete_output_requires_only_missing(tmp_path: Path) -> None:
	hard_manifest = _hard_manifest(tmp_path)
	config = _config(tmp_path, hard_manifest)
	export_multi_head_xy_neighbor_unanimous_targets(config)

	plans = plan_multi_head_xy_neighbor_unanimous_target_exports(
		config,
		only_missing=False,
	)

	assert [plan.action for plan in plans] == ['ERROR', 'ERROR', 'ERROR']
	with pytest.raises(FileExistsError, match='use --only-missing'):
		export_multi_head_xy_neighbor_unanimous_targets(config)


def test_config_and_cli_expose_only_fixed_immutable_controls(tmp_path: Path) -> None:
	hard_manifest = _hard_manifest(tmp_path)
	resolved = resolve_multi_head_xy_neighbor_unanimous_target_export_config(
		{
			'source_hard_manifest': str(hard_manifest),
			'output_root': str(tmp_path / 'output'),
		}
	)
	assert resolved.handoff_manifest.name == (
		'multi_head_xy_neighbor_unanimous_target_handoff.json'
	)
	with pytest.raises(ValueError, match='unknown XY-neighbor-unanimous'):
		resolve_multi_head_xy_neighbor_unanimous_target_export_config(
			{
				'source_hard_manifest': str(hard_manifest),
				'output_root': str(tmp_path / 'output'),
				'smoothing': {'four_valid_neighbors_minimum_agreement': 3},
			}
		)
	help_text = unanimous_export_cli.build_parser().format_help()
	assert '--dry-run' in help_text
	assert '--only-missing' in help_text


def _config(
	tmp_path: Path,
	hard_manifest: Path,
) -> MultiHeadXYNeighborUnanimousTargetExportConfig:
	return MultiHeadXYNeighborUnanimousTargetExportConfig(
		source_hard_manifest=hard_manifest,
		output_root=tmp_path / 'unanimous',
		handoff_manifest=tmp_path / 'unanimous' / 'handoff.json',
	)
