"""Target-only unanimous audit contracts."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

import seis_ssl_cluster.f3.xy_neighbor_unanimous_target_audit as audit_module
from seis_ssl_cluster.f3.xy_neighbor_unanimous_target_audit import (
	F3XYNeighborUnanimousTargetAuditConfig,
	audit_f3_xy_neighbor_unanimous_targets,
	f3_xy_neighbor_unanimous_target_audit_config_from_mapping,
	load_f3_xy_neighbor_unanimous_target_audit,
	replay_f3_xy_neighbor_unanimous_target_audit,
	validate_f3_xy_neighbor_unanimous_target_audit,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus_targets import (
	MultiHeadXYNeighborConsensusTargetExportConfig,
	export_multi_head_xy_neighbor_consensus_targets,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_unanimous_targets import (
	MultiHeadXYNeighborUnanimousTargetExportConfig,
	export_multi_head_xy_neighbor_unanimous_targets,
)
from tests.seis_ssl_cluster.test_strat_multi_head_xy_neighbor_consensus_targets import (
	_hard_manifest,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_target_audit_mapping_accepts_existing_artifact_root_directory(
	tmp_path: Path,
) -> None:
	"""The public YAML adapter must distinguish its root directory from files."""
	for path in (
		tmp_path / 'source.json',
		tmp_path / 'consensus.json',
		tmp_path / 'unanimous.json',
	):
		path.write_text('{}', encoding='utf-8')
	config = f3_xy_neighbor_unanimous_target_audit_config_from_mapping(
		{
			'artifact_root': str(tmp_path),
			'source_hard_manifest': str(tmp_path / 'source.json'),
			'xy_neighbor_consensus_target_manifest': str(tmp_path / 'consensus.json'),
			'xy_neighbor_unanimous_target_manifest': str(tmp_path / 'unanimous.json'),
			'output_path': str(tmp_path / 'output.json'),
		}
	)

	assert config.artifact_root == tmp_path.resolve()


def test_target_audit_proves_subset_and_reuses_identical_evidence(
	tmp_path: Path,
) -> None:
	hard = _hard_manifest(tmp_path)
	consensus = MultiHeadXYNeighborConsensusTargetExportConfig(
		source_hard_manifest=hard,
		output_root=tmp_path / 'consensus',
		handoff_manifest=tmp_path / 'consensus' / 'handoff.json',
	)
	unanimous = MultiHeadXYNeighborUnanimousTargetExportConfig(
		source_hard_manifest=hard,
		output_root=tmp_path / 'unanimous',
		handoff_manifest=tmp_path / 'unanimous' / 'handoff.json',
	)
	export_multi_head_xy_neighbor_consensus_targets(consensus)
	export_multi_head_xy_neighbor_unanimous_targets(unanimous)
	config = F3XYNeighborUnanimousTargetAuditConfig(
		artifact_root=tmp_path,
		source_hard_manifest=hard,
		xy_neighbor_consensus_target_manifest=consensus.handoff_manifest,
		xy_neighbor_unanimous_target_manifest=unanimous.handoff_manifest,
		output_path=tmp_path / 'unanimous' / 'target_audit.json',
	)

	first = audit_f3_xy_neighbor_unanimous_targets(config)

	assert first.action == 'WRITTEN'
	assert first.payload['status'] == 'XYUNANIM_TARGET_GO'
	for evidence in first.payload['per_k'].values():  # type: ignore[union-attr]
		assert evidence['subset_evidence']['changed_mask_subset']  # type: ignore[index]
		assert evidence['subset_evidence']['label_parity']  # type: ignore[index]
		assert evidence['unanimous_changed_token_count'] > 0  # type: ignore[index]
	before = config.output_path.stat().st_mtime_ns
	second = audit_f3_xy_neighbor_unanimous_targets(config, only_missing=True)

	assert second.action == 'REUSE_COMPLETED'
	assert config.output_path.stat().st_mtime_ns == before
	assert (
		load_f3_xy_neighbor_unanimous_target_audit(config.output_path) == first.payload
	)
	assert validate_f3_xy_neighbor_unanimous_target_audit(config) == first.payload


def test_target_audit_dry_run_does_not_publish(tmp_path: Path) -> None:
	hard = _hard_manifest(tmp_path)
	consensus = MultiHeadXYNeighborConsensusTargetExportConfig(
		source_hard_manifest=hard,
		output_root=tmp_path / 'consensus',
		handoff_manifest=tmp_path / 'consensus' / 'handoff.json',
	)
	unanimous = MultiHeadXYNeighborUnanimousTargetExportConfig(
		source_hard_manifest=hard,
		output_root=tmp_path / 'unanimous',
		handoff_manifest=tmp_path / 'unanimous' / 'handoff.json',
	)
	export_multi_head_xy_neighbor_consensus_targets(consensus)
	export_multi_head_xy_neighbor_unanimous_targets(unanimous)
	config = F3XYNeighborUnanimousTargetAuditConfig(
		artifact_root=tmp_path,
		source_hard_manifest=hard,
		xy_neighbor_consensus_target_manifest=consensus.handoff_manifest,
		xy_neighbor_unanimous_target_manifest=unanimous.handoff_manifest,
		output_path=tmp_path / 'unanimous' / 'target_audit.json',
	)

	result = audit_f3_xy_neighbor_unanimous_targets(config, dry_run=True)

	assert result.action == 'DRY_RUN'
	assert result.payload['status'] == 'XYUNANIM_TARGET_GO'
	assert not config.output_path.exists()


def test_target_audit_loader_rejects_status_or_condition_tampering(
	tmp_path: Path,
) -> None:
	hard = _hard_manifest(tmp_path)
	consensus = MultiHeadXYNeighborConsensusTargetExportConfig(
		source_hard_manifest=hard,
		output_root=tmp_path / 'consensus',
		handoff_manifest=tmp_path / 'consensus' / 'handoff.json',
	)
	unanimous = MultiHeadXYNeighborUnanimousTargetExportConfig(
		source_hard_manifest=hard,
		output_root=tmp_path / 'unanimous',
		handoff_manifest=tmp_path / 'unanimous' / 'handoff.json',
	)
	export_multi_head_xy_neighbor_consensus_targets(consensus)
	export_multi_head_xy_neighbor_unanimous_targets(unanimous)
	config = F3XYNeighborUnanimousTargetAuditConfig(
		artifact_root=tmp_path,
		source_hard_manifest=hard,
		xy_neighbor_consensus_target_manifest=consensus.handoff_manifest,
		xy_neighbor_unanimous_target_manifest=unanimous.handoff_manifest,
		output_path=tmp_path / 'unanimous' / 'target_audit.json',
	)
	audit_f3_xy_neighbor_unanimous_targets(config)
	payload = json.loads(config.output_path.read_text(encoding='utf-8'))
	payload['go_conditions']['6']['changed_token_count_positive'] = False
	config.output_path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='conditions mismatch evidence'):
		load_f3_xy_neighbor_unanimous_target_audit(config.output_path)


def test_target_audit_replay_rejects_coherent_but_stale_hold_evidence(
	tmp_path: Path,
) -> None:
	"""A syntactically coherent post-publication edit cannot bypass replay."""
	hard = _hard_manifest(tmp_path)
	consensus = MultiHeadXYNeighborConsensusTargetExportConfig(
		source_hard_manifest=hard,
		output_root=tmp_path / 'consensus',
		handoff_manifest=tmp_path / 'consensus' / 'handoff.json',
	)
	unanimous = MultiHeadXYNeighborUnanimousTargetExportConfig(
		source_hard_manifest=hard,
		output_root=tmp_path / 'unanimous',
		handoff_manifest=tmp_path / 'unanimous' / 'handoff.json',
	)
	export_multi_head_xy_neighbor_consensus_targets(consensus)
	export_multi_head_xy_neighbor_unanimous_targets(unanimous)
	config = F3XYNeighborUnanimousTargetAuditConfig(
		artifact_root=tmp_path,
		source_hard_manifest=hard,
		xy_neighbor_consensus_target_manifest=consensus.handoff_manifest,
		xy_neighbor_unanimous_target_manifest=unanimous.handoff_manifest,
		output_path=tmp_path / 'unanimous' / 'target_audit.json',
	)
	audit_f3_xy_neighbor_unanimous_targets(config)
	payload = json.loads(config.output_path.read_text(encoding='utf-8'))
	payload['per_k']['6']['unanimous_changed_token_count'] = 0
	payload['go_conditions']['6']['changed_token_count_positive'] = False
	payload['status'] = 'XYUNANIM_TARGET_HOLD'
	config.output_path.write_text(json.dumps(payload), encoding='utf-8')

	assert (
		load_f3_xy_neighbor_unanimous_target_audit(config.output_path)['status']
		== 'XYUNANIM_TARGET_HOLD'
	)
	with pytest.raises(ValueError, match='differs from replayed evidence'):
		replay_f3_xy_neighbor_unanimous_target_audit(
			config.output_path,
			artifact_root=tmp_path,
		)


def test_target_audit_quarantines_invalid_output_then_republishes(
	tmp_path: Path,
) -> None:
	hard = _hard_manifest(tmp_path)
	consensus = MultiHeadXYNeighborConsensusTargetExportConfig(
		source_hard_manifest=hard,
		output_root=tmp_path / 'consensus',
		handoff_manifest=tmp_path / 'consensus' / 'handoff.json',
	)
	unanimous = MultiHeadXYNeighborUnanimousTargetExportConfig(
		source_hard_manifest=hard,
		output_root=tmp_path / 'unanimous',
		handoff_manifest=tmp_path / 'unanimous' / 'handoff.json',
	)
	export_multi_head_xy_neighbor_consensus_targets(consensus)
	export_multi_head_xy_neighbor_unanimous_targets(unanimous)
	config = F3XYNeighborUnanimousTargetAuditConfig(
		artifact_root=tmp_path,
		source_hard_manifest=hard,
		xy_neighbor_consensus_target_manifest=consensus.handoff_manifest,
		xy_neighbor_unanimous_target_manifest=unanimous.handoff_manifest,
		output_path=tmp_path / 'unanimous' / 'target_audit.json',
	)
	audit_f3_xy_neighbor_unanimous_targets(config)
	payload = json.loads(config.output_path.read_text(encoding='utf-8'))
	payload['status'] = 'XYUNANIM_TARGET_HOLD'
	config.output_path.write_text(json.dumps(payload), encoding='utf-8')

	result = audit_f3_xy_neighbor_unanimous_targets(
		config,
		only_missing=True,
		quarantine_invalid=True,
	)

	assert result.action == 'WRITTEN'
	assert result.quarantine_path is not None
	assert result.quarantine_path.is_file()
	assert (
		load_f3_xy_neighbor_unanimous_target_audit(config.output_path)['status']
		== 'XYUNANIM_TARGET_GO'
	)


@pytest.mark.parametrize(
	('mutate', 'failed_condition'),
	[
		(
			lambda evidence: evidence.__setitem__('unanimous_changed_token_count', 0),
			'changed_token_count_positive',
		),
		(
			lambda evidence: evidence['unanimous']['spatial']['combined'].__setitem__(  # type: ignore[index]
				'disagreement_count',
				evidence['source']['spatial']['combined']['disagreement_count'],  # type: ignore[index]
			),
			'unanimous_combined_xy_disagreement_lt_source',
		),
		(
			lambda evidence: evidence['unanimous'].__setitem__(  # type: ignore[index]
				'ordered_path_violation_count', 1
			),
			'ordered_path_violation_count_zero',
		),
		(
			lambda evidence: evidence['unanimous'].__setitem__(  # type: ignore[index]
				'empty_output_state_count', 1
			),
			'empty_output_state_count_zero',
		),
		(
			lambda evidence: evidence['subset_evidence'].update(  # type: ignore[index]
				{'changed_mask_subset': False}
			),
			'unanimous_changed_mask_subset_of_3_of_4',
		),
		(
			lambda evidence: evidence['subset_evidence'].update(  # type: ignore[index]
				{'label_parity': False}
			),
			'unanimous_output_equals_3_of_4_at_unanimous_changes',
		),
	],
)
def test_target_audit_hold_conditions_are_fixed_and_transition_increase_is_not_one(
	tmp_path: Path,
	mutate: object,
	failed_condition: str,
) -> None:
	"""Only the named target conditions decide GO/HOLD; transitions are descriptive."""
	hard = _hard_manifest(tmp_path)
	consensus = MultiHeadXYNeighborConsensusTargetExportConfig(
		source_hard_manifest=hard,
		output_root=tmp_path / 'consensus',
		handoff_manifest=tmp_path / 'consensus' / 'handoff.json',
	)
	unanimous = MultiHeadXYNeighborUnanimousTargetExportConfig(
		source_hard_manifest=hard,
		output_root=tmp_path / 'unanimous',
		handoff_manifest=tmp_path / 'unanimous' / 'handoff.json',
	)
	export_multi_head_xy_neighbor_consensus_targets(consensus)
	export_multi_head_xy_neighbor_unanimous_targets(unanimous)
	config = F3XYNeighborUnanimousTargetAuditConfig(
		artifact_root=tmp_path,
		source_hard_manifest=hard,
		xy_neighbor_consensus_target_manifest=consensus.handoff_manifest,
		xy_neighbor_unanimous_target_manifest=unanimous.handoff_manifest,
		output_path=tmp_path / 'unanimous' / 'target_audit.json',
	)
	payload = audit_f3_xy_neighbor_unanimous_targets(config, dry_run=True).payload
	evidence = deepcopy(payload['per_k']['6'])  # type: ignore[index]
	assert callable(mutate)
	mutate(evidence)
	conditions = audit_module._conditions_from_evidence(evidence)  # noqa: SLF001

	assert not conditions[failed_condition]
	assert (
		audit_module._status_from_conditions({'6': conditions})  # noqa: SLF001
		== 'XYUNANIM_TARGET_HOLD'
	)
	increased = deepcopy(payload['per_k']['6'])  # type: ignore[index]
	increased['unanimous']['temporal_transition_count'] += 100  # type: ignore[index]
	increased_conditions = audit_module._conditions_from_evidence(  # noqa: SLF001
		increased
	)
	assert all(increased_conditions.values())
	assert (
		audit_module._status_from_conditions({'6': increased_conditions})  # noqa: SLF001
		== 'XYUNANIM_TARGET_GO'
	)
