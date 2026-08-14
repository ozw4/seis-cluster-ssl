"""Immutable source-hard-only XY-consensus target export contracts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from proc.seis_ssl_cluster import (
	export_strat_hmm_multi_head_xy_neighbor_consensus_targets as consensus_export_cli,
)
from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.data.target_providers import (
	MultiHeadStratPseudoTargetProvider,
	TargetProviderContext,
	load_strat_multi_head_xy_neighbor_consensus_target_manifest_adapter,
)
from seis_ssl_cluster.stratigraphy.multi_head import build_multi_head_target_manifest
from seis_ssl_cluster.stratigraphy.targets import write_pseudo_target
from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus_targets import (
	TARGET_REPRESENTATION,
	TARGET_SEMANTICS,
	MultiHeadXYNeighborConsensusTargetExportConfig,
	_temporal_transition_count,
	export_multi_head_xy_neighbor_consensus_targets,
	load_multi_head_xy_neighbor_consensus_target_manifest,
	plan_multi_head_xy_neighbor_consensus_target_exports,
	resolve_multi_head_xy_neighbor_consensus_target_export_config,
)


def test_export_replays_source_hard_labels_and_detects_rehashed_tampering(
	tmp_path: Path,
) -> None:
	"""The handoff is source-only, preserves invalid values, and is replayed."""
	hard_manifest = _hard_manifest(tmp_path)
	config = MultiHeadXYNeighborConsensusTargetExportConfig(
		source_hard_manifest=hard_manifest,
		output_root=tmp_path / 'consensus',
		handoff_manifest=tmp_path / 'consensus' / 'handoff.json',
	)
	plans = export_multi_head_xy_neighbor_consensus_targets(config)
	assert [plan.action for plan in plans] == ['NEW', 'NEW', 'NEW']
	payload = load_multi_head_xy_neighbor_consensus_target_manifest(
		config.handoff_manifest
	)
	assert payload['target_representation'] == TARGET_REPRESENTATION
	assert payload['target_semantics'] == TARGET_SEMANTICS

	for k in (6, 8, 10):
		entry = payload['heads'][str(k)]['surveys']['survey']  # type: ignore[index]
		labels = np.load(entry['labels']['path'])  # type: ignore[index]
		confidence = np.load(entry['confidence']['path'])  # type: ignore[index]
		valid = np.load(entry['valid_tokens']['path'])  # type: ignore[index]
		assert labels.dtype == np.int32
		assert confidence.dtype == np.float32
		assert valid.dtype == np.bool_
		assert labels[1, 1, 5] == 5
		assert labels[0, 0, 0] == -17
		assert confidence[1, 1, 5] == 1.0
		assert confidence[0, 0, 0] == 0.0
		for x in range(labels.shape[0]):
			for y in range(labels.shape[1]):
				assert np.all(np.diff(labels[x, y, valid[x, y]]) >= 0)

	entry = payload['heads']['6']['surveys']['survey']  # type: ignore[index]
	labels_path = Path(entry['labels']['path'])  # type: ignore[index]
	labels = np.load(labels_path)
	labels[1, 1, 5] = 4
	np.save(labels_path, labels, allow_pickle=False)
	entry['labels']['sha256'] = file_sha256(labels_path)  # type: ignore[index]
	head_path = config.output_root / 'bundle/k6/head_metadata.json'
	head = json.loads(head_path.read_text(encoding='utf-8'))
	head['surveys']['survey']['labels']['sha256'] = file_sha256(labels_path)
	head_path.write_text(json.dumps(head), encoding='utf-8')
	config.handoff_manifest.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='smoothing replay result'):
		load_multi_head_xy_neighbor_consensus_target_manifest(config.handoff_manifest)


def test_config_refuses_external_smoothing_parameters(tmp_path: Path) -> None:
	"""The fixed policy has no beta, posterior, or configurable smoothing knob."""
	hard = tmp_path / 'hard.json'
	hard.write_text('{}', encoding='utf-8')
	with pytest.raises(ValueError, match='unknown XY-neighbor-consensus'):
		resolve_multi_head_xy_neighbor_consensus_target_export_config(
			{
				'source_hard_manifest': str(hard),
				'output_root': str(tmp_path / 'output'),
				'smoothing': {'pairwise_strength_ratio': 0.25},
			}
		)


def test_export_plan_quarantines_mutated_bundle_head_metadata(tmp_path: Path) -> None:
	"""A reusable handoff must remain byte-for-byte aligned with its bundle."""
	hard_manifest = _hard_manifest(tmp_path)
	config = MultiHeadXYNeighborConsensusTargetExportConfig(
		source_hard_manifest=hard_manifest,
		output_root=tmp_path / 'consensus',
		handoff_manifest=tmp_path / 'consensus' / 'handoff.json',
	)
	export_multi_head_xy_neighbor_consensus_targets(config)
	head_path = config.output_root / 'bundle/k6/head_metadata.json'
	head = json.loads(head_path.read_text(encoding='utf-8'))
	head['diagnostics']['aggregate']['changed_token_count'] += 1
	head_path.write_text(json.dumps(head), encoding='utf-8')

	plans = plan_multi_head_xy_neighbor_consensus_target_exports(
		config,
		only_missing=True,
	)

	assert [plan.action for plan in plans] == ['QUARANTINE', 'QUARANTINE', 'QUARANTINE']
	assert all(plan.reason and 'head metadata differs' in plan.reason for plan in plans)


def test_temporal_transition_diagnostics_bridge_valid_gaps_and_allow_increases(
	tmp_path: Path,
) -> None:
	"""Source/output transitions are replayed diagnostics, never an export gate."""
	labels = np.asarray([[[0, -17, 2, 2, 4]]], dtype=np.int32)
	valid = np.asarray([[[True, False, True, True, True]]], dtype=bool)
	assert _temporal_transition_count(labels, valid) == 2

	hard_manifest = _hard_manifest(tmp_path)
	payload = json.loads(hard_manifest.read_text(encoding='utf-8'))
	for k in (6, 8, 10):
		entry = payload['heads'][str(k)]['surveys']['survey']
		labels_path = Path(entry['labels']['path'])
		source_labels = np.load(labels_path)
		# The four same-z neighbours vote 1 for this internal center.  Source
		# [0, 0, 2, ...] becomes output [0, 1, 2, ...], increasing transitions
		# while preserving the ordered valid trace.
		source_labels[1, 1, 1] = 0
		np.save(labels_path, source_labels, allow_pickle=False)
		entry['labels']['sha256'] = file_sha256(labels_path)
	hard_manifest.write_text(json.dumps(payload), encoding='utf-8')

	config = MultiHeadXYNeighborConsensusTargetExportConfig(
		source_hard_manifest=hard_manifest,
		output_root=tmp_path / 'consensus',
		handoff_manifest=tmp_path / 'consensus' / 'handoff.json',
	)
	export_multi_head_xy_neighbor_consensus_targets(config)
	manifest = load_multi_head_xy_neighbor_consensus_target_manifest(
		config.handoff_manifest
	)

	for k in (6, 8, 10):
		diagnostics = manifest['heads'][str(k)]['diagnostics']  # type: ignore[index]
		per_survey = diagnostics['per_survey']['survey']  # type: ignore[index]
		aggregate = diagnostics['aggregate']  # type: ignore[index]
		assert per_survey['temporal_transition_counts']['output'] > (  # type: ignore[index]
			per_survey['temporal_transition_counts']['source']  # type: ignore[index]
		)
		assert aggregate['temporal_transition_counts'] == per_survey[  # type: ignore[index]
			'temporal_transition_counts'
		]


def test_export_cli_exposes_immutable_resume_controls() -> None:
	"""Automation can plan, then explicitly reuse a complete publication."""
	help_text = consensus_export_cli.build_parser().format_help()
	assert '--dry-run' in help_text
	assert '--only-missing' in help_text


def test_export_uses_hard_labels_and_masks_after_embedding_artifacts_are_removed(
	tmp_path: Path,
) -> None:
	"""Consensus has no source-embedding, posterior, or metadata dependency."""
	hard_manifest = _hard_manifest(tmp_path)
	shutil.rmtree(tmp_path / 'embeddings')
	config = MultiHeadXYNeighborConsensusTargetExportConfig(
		source_hard_manifest=hard_manifest,
		output_root=tmp_path / 'consensus',
		handoff_manifest=tmp_path / 'consensus' / 'handoff.json',
	)
	assert all(
		plan.action == 'NEW'
		for plan in export_multi_head_xy_neighbor_consensus_targets(config)
	)
	assert (
		load_multi_head_xy_neighbor_consensus_target_manifest(config.handoff_manifest)[
			'target_semantics'
		]
		== TARGET_SEMANTICS
	)


def test_hard_dataset_adapter_normalizes_preserved_invalid_source_values(
	tmp_path: Path,
) -> None:
	"""The existing hard-label provider accepts a preserved invalid sentinel."""
	hard_manifest = _hard_manifest(tmp_path)
	config = MultiHeadXYNeighborConsensusTargetExportConfig(
		source_hard_manifest=hard_manifest,
		output_root=tmp_path / 'consensus',
		handoff_manifest=tmp_path / 'consensus' / 'handoff.json',
	)
	export_multi_head_xy_neighbor_consensus_targets(config)
	adapted = load_strat_multi_head_xy_neighbor_consensus_target_manifest_adapter(
		config.handoff_manifest
	)
	provider = MultiHeadStratPseudoTargetProvider(adapted.target_manifest)
	sample: dict[str, object] = {'coords': {}}
	provider.add_targets(
		sample,
		TargetProviderContext(
			manifest=type('Manifest', (), {'survey_id': 'survey'})(),
			crop_request=type('Request', (), {})(),
			patch_size_xyz=(1, 1, 1),
			token_start_xyz=(0, 0, 0),
			token_size_xyz=(3, 3, 12),
			token_valid_mask=np.ones((3, 3, 12), dtype=bool),
		),
	)
	targets = sample['strat_multi_targets']
	assert isinstance(targets, dict)
	for k in (6, 8, 10):
		assert targets[f'k{k}']['labels'][0, 0, 0] == -1


def _hard_manifest(tmp_path: Path) -> Path:
	shape = (3, 3, 12)
	valid = np.ones(shape, dtype=bool)
	valid[0, 0, 0] = False
	embeddings = tmp_path / 'embeddings'
	embeddings.mkdir()
	np.save(embeddings / 'survey.embeddings.npy', np.zeros((*shape, 3), np.float32))
	np.save(embeddings / 'survey.valid_tokens.npy', np.ones(shape, dtype=bool))
	(embeddings / 'survey.embedding_metadata.json').write_text(
		json.dumps(
			{
				'survey_id': 'survey',
				'source_amplitude_path': 'amplitude.npy',
				'checkpoint_path': 'checkpoint.pt',
				'checkpoint_sha256': 'checkpoint',
				'model_geometry': {'name': 'fixture'},
				'patch_size': [1, 1, 1],
				'token_grid_shape': list(shape),
				'window_size': [1, 1, 1],
				'overlap': [0, 0, 0],
				'normalization_stats_path': 'stats.json',
				'output_dtype': 'float32',
				'min_token_valid_fraction': 1.0,
				'zero_mask': {},
			}
		),
		encoding='utf-8',
	)
	heads: dict[int, Path] = {}
	for k in (6, 8, 10):
		labels = np.tile(np.minimum(np.arange(shape[2]), k - 1), (*shape[:2], 1))
		labels = labels.astype(np.int32)
		labels[1, 1, 5] = 4
		labels[~valid] = -1
		source_labels = (
			tmp_path / f'cluster{k}/labels/k{k}/survey.cluster_labels_token.npy'
		)
		source_labels.parent.mkdir(parents=True)
		np.save(source_labels, labels, allow_pickle=False)
		root = tmp_path / f'hard{k}'
		write_pseudo_target(
			root,
			k=k,
			survey_id='survey',
			labels=labels,
			confidence=np.asarray(valid, dtype=np.float32),
			valid_tokens=valid,
			metadata={
				'source_clustering_output_dir': str(source_labels.parents[2]),
				'source_label_path': str(source_labels),
			},
			schema_version=1,
			write_boundary_weight=False,
		)
		heads[k] = root

	replay_root = tmp_path / 'replay-k6'
	replay_labels = np.load(heads[6] / 'k6/survey.hmm_labels_token.npy')
	replay_source = (
		tmp_path / 'replay-cluster/labels/k6/survey.cluster_labels_token.npy'
	)
	replay_source.parent.mkdir(parents=True)
	np.save(replay_source, replay_labels, allow_pickle=False)
	write_pseudo_target(
		replay_root,
		k=6,
		survey_id='survey',
		labels=replay_labels,
		confidence=np.asarray(valid, dtype=np.float32),
		valid_tokens=valid,
		metadata={
			'source_clustering_output_dir': str(replay_source.parents[2]),
			'source_label_path': str(replay_source),
		},
		schema_version=1,
		write_boundary_weight=False,
	)
	path = tmp_path / 'hard-manifest.json'
	build_multi_head_target_manifest(
		manifest_path=path,
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=replay_root,
	)
	# The source-only successor must retain invalid source values verbatim.  The
	# historical hard-manifest builder enforces -1, so change only the frozen
	# hard-label payload after it has published its source reference contract.
	payload = json.loads(path.read_text(encoding='utf-8'))
	for k in (6, 8, 10):
		entry = payload['heads'][str(k)]['surveys']['survey']
		labels_path = Path(entry['labels']['path'])
		labels = np.load(labels_path)
		labels[~valid] = -17
		np.save(labels_path, labels, allow_pickle=False)
		entry['labels']['sha256'] = file_sha256(labels_path)
	path.write_text(json.dumps(payload), encoding='utf-8')
	return path
