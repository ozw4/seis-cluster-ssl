"""Contracts for the independent F3 XY-neighbour consensus experiment."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

import seis_ssl_cluster.f3.xy_neighbor_consensus_pretraining_validation as validation
import seis_ssl_cluster.f3.xy_neighbor_consensus_results as results
from seis_ssl_cluster.results import validate_results_artifacts


def test_preflight_config_is_closed_and_requires_successor_inputs(
	tmp_path: Path,
) -> None:
	artifact_root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	artifact_root.mkdir(parents=True)
	target = artifact_root / 'targets.json'
	smoke = artifact_root / 'smoke.yaml'
	full = artifact_root / 'full.yaml'
	for path in (target, smoke, full):
		path.write_text('{}\n', encoding='utf-8')
	config = {
		'artifact_root': str(artifact_root),
		'experiment_root': str(artifact_root / 'pretraining/f3/facies_benchmark_v1'),
		'target_manifest': str(target),
		'xy_neighbor_consensus_smoke_config': str(smoke),
		'xy_neighbor_consensus_full_config': str(full),
	}

	resolved = (
		validation.f3_xy_neighbor_consensus_pretraining_validation_config_from_mapping(
			config
		)
	)

	assert resolved.target_manifest == target
	with pytest.raises(ValueError, match='unknown XY-neighbour consensus'):
		validation.f3_xy_neighbor_consensus_pretraining_validation_config_from_mapping(
			{**config, 'beta_calibration': 'forbidden'}
		)
	with pytest.raises(ValueError, match='xy_neighbor_consensus_full_config'):
		validation.f3_xy_neighbor_consensus_pretraining_validation_config_from_mapping(
			{
				key: value
				for key, value in config.items()
				if key != 'xy_neighbor_consensus_full_config'
			}
		)


def test_handoff_rejects_m5_provenance_fields(tmp_path: Path) -> None:
	paths = _paths(tmp_path)
	handoff = _handoff(paths)
	paths['handoff'].write_text(json.dumps(handoff), encoding='utf-8')

	loaded = validation.load_f3_xy_neighbor_consensus_pretraining_handoff(
		paths['handoff']
	)
	assert loaded['variant'] == 'xycons1_nocons'
	handoff['targets']['source_posterior_manifest'] = _reference(  # type: ignore[index]
		paths['hard']
	)
	paths['handoff'].write_text(json.dumps(handoff), encoding='utf-8')
	with pytest.raises(ValueError, match='target keys mismatch'):
		validation.load_f3_xy_neighbor_consensus_pretraining_handoff(paths['handoff'])


def test_handoff_rejects_incomplete_target_head_hashes(tmp_path: Path) -> None:
	paths = _paths(tmp_path)
	handoff = _handoff(paths)
	handoff['targets']['xy_neighbor_consensus_target_head_hashes'] = {  # type: ignore[index]
		'6': {},
		'8': {},
		'10': {},
	}
	paths['handoff'].write_text(json.dumps(handoff), encoding='utf-8')

	with pytest.raises(ValueError, match='must contain surveys'):
		validation.load_f3_xy_neighbor_consensus_pretraining_handoff(paths['handoff'])


def test_review_publishes_portable_source_only_diagnostics(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	paths = _paths(tmp_path)
	handoff = _handoff(paths)
	paths['handoff'].write_text(json.dumps(handoff), encoding='utf-8')
	calls: list[object] = []
	target = _target_manifest(paths)

	def load_target(*_args: object, **kwargs: object) -> dict[str, object]:
		calls.append(kwargs.get('validate_array_semantics'))
		return target

	monkeypatch.setattr(
		results,
		'load_multi_head_xy_neighbor_consensus_target_manifest',
		load_target,
	)
	runtime_checks: list[tuple[object, object]] = []
	monkeypatch.setattr(
		results,
		'_validate_handoff_artifact_lineage',
		lambda _config, *, target, handoff: runtime_checks.append((target, handoff)),
	)
	config = results.f3_xy_neighbor_consensus_review_config_from_mapping(
		_review_config_mapping(paths)
	)
	with pytest.raises(ValueError, match='unknown XY-neighbour consensus'):
		results.f3_xy_neighbor_consensus_review_config_from_mapping(
			{**_review_config_mapping(paths), 'calibration_handoff': 'forbidden'}
		)

	publication = results.publish_f3_xy_neighbor_consensus_review(config)

	assert calls == [False]
	assert runtime_checks == [(target, handoff)]
	assert publication.publish_manifest is not None
	assert {path.name for path in paths['output_dir'].iterdir()} == {
		results.SUMMARY_JSON,
		results.SUMMARY_MARKDOWN,
		'publish_manifest.json',
	}
	summary = json.loads(publication.summary_json.read_text(encoding='utf-8'))
	assert summary['target_representation'] == 'xy_neighbor_consensus_hard_labels_v1'
	assert summary['head_diagnostics'][0]['changed_token_count'] == 2
	assert summary['head_diagnostics'][0]['source_temporal_transition_count'] == 6
	assert summary['head_diagnostics'][0]['output_temporal_transition_count'] == 7
	assert 'increases are allowed' in publication.summary_markdown.read_text(
		encoding='utf-8'
	)
	assert summary['source_hard_manifest']['path'].startswith(
		'${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/'
	)
	for path in paths['output_dir'].iterdir():
		if path.suffix in {'.json', '.md'}:
			text = path.read_text(encoding='utf-8')
			assert str(paths['artifact_root']) not in text
			assert str(paths['workspace_root']) not in text
	report = validate_results_artifacts(
		paths['output_dir'],
		required_files=(
			Path(results.SUMMARY_JSON),
			Path(results.SUMMARY_MARKDOWN),
			Path('publish_manifest.json'),
		),
		local_path_policy='error',
		local_path_markers=(
			f'{paths["artifact_root"]}/',
			f'{paths["workspace_root"]}/',
		),
	)
	assert report.ok, report.errors

	stale_head_hashes = _head_hashes()
	stale_head_hashes['6']['survey']['labels'] = 'f' * 64
	handoff['targets']['xy_neighbor_consensus_target_head_hashes'] = stale_head_hashes  # type: ignore[index]
	paths['handoff'].write_text(json.dumps(handoff), encoding='utf-8')
	with pytest.raises(ValueError, match='head hashes'):
		results.publish_f3_xy_neighbor_consensus_review(config)

	handoff = _handoff(paths)
	handoff['targets']['xy_neighbor_consensus_smoothing'] = {'application': 'stale'}  # type: ignore[index]
	paths['handoff'].write_text(json.dumps(handoff), encoding='utf-8')
	with pytest.raises(ValueError, match='smoothing policy'):
		results.publish_f3_xy_neighbor_consensus_review(config)

	handoff = _handoff(paths)
	handoff['targets']['temporal_transition_counts']['6']['output'] += 1  # type: ignore[index]
	paths['handoff'].write_text(json.dumps(handoff), encoding='utf-8')
	with pytest.raises(ValueError, match='temporal transition counts'):
		results.publish_f3_xy_neighbor_consensus_review(config)


def test_review_rejects_m5_checkpoint_even_with_a_matching_target_handoff(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	paths = _paths(tmp_path)
	target = _target_manifest(paths)
	handoff = _handoff(paths)
	checkpoint = paths['artifact_root'] / 'pretraining/m5-ls/best.pt'
	checkpoint.parent.mkdir(parents=True)
	checkpoint.write_bytes(b'm5-ls')
	handoff['checkpoint']['path'] = str(checkpoint)  # type: ignore[index]
	handoff['checkpoint']['sha256'] = _reference(checkpoint)['sha256']  # type: ignore[index]
	paths['handoff'].write_text(json.dumps(handoff), encoding='utf-8')
	monkeypatch.setattr(
		results,
		'load_multi_head_xy_neighbor_consensus_target_manifest',
		lambda *_args, **_kwargs: target,
	)
	monkeypatch.setattr(
		results,
		'validate_stratigraphy_checkpoint_payload',
		lambda _: None,
	)
	monkeypatch.setattr(
		results.torch,
		'load',
		lambda *_args, **_kwargs: {
			'stratigraphy_checkpoint': {
				'schema_version': 4,
				'target_representation': 'lateral_mean_field_hard_labels_v1',
			}
		},
	)
	config = results.f3_xy_neighbor_consensus_review_config_from_mapping(
		_review_config_mapping(paths)
	)

	with pytest.raises(ValueError, match='schema 5'):
		results.publish_f3_xy_neighbor_consensus_review(config)


def test_review_artifact_lineage_accepts_bound_v5_checkpoint_and_metadata(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	paths = _paths(tmp_path)
	target = _target_manifest(paths)
	handoff = _handoff(paths)
	checkpoint = paths['artifact_root'] / 'pretraining/xy/best.pt'
	checkpoint.parent.mkdir(parents=True)
	checkpoint.write_bytes(b'xy-consensus')
	checkpoint_reference = _reference(checkpoint)
	handoff['checkpoint']['path'] = checkpoint_reference['path']  # type: ignore[index]
	handoff['checkpoint']['sha256'] = checkpoint_reference['sha256']  # type: ignore[index]

	embedding_root = paths['artifact_root'] / 'embeddings/xy/overlap_x16'
	files = results.output_paths(embedding_root, 'f3_facies_benchmark')
	files.embeddings.parent.mkdir(parents=True)
	files.embeddings.write_bytes(b'embeddings')
	files.valid_tokens.write_bytes(b'valid-tokens')
	target_reference = _reference(paths['target'])
	identity = {
		'schema_version': 5,
		'head_spec': 'multi_resolution_ordered_prototypes_v1',
		'head_ks': [6, 8, 10],
		'target_representation': target['target_representation'],
		'target_semantics': target['target_semantics'],
		'xy_neighbor_consensus_target_manifest_sha256': target_reference['sha256'],
		'xy_neighbor_consensus_target_manifest': target_reference,
		'per_head_xy_neighbor_consensus_targets': _head_hashes(),
		'source_hard_manifest_sha256': _reference(paths['hard'])['sha256'],
		'xy_neighbor_consensus_smoothing': target['smoothing'],
		'consistency_policy': 'disabled_for_xy_neighbor_consensus_v1',
		'consistency_weight': 0.0,
		'consistency_beta': 0.1,
		'model_tag': handoff['model_tag'],
		'initial_student_state_sha256': 'a' * 64,
		'initial_head_state_sha256': 'b' * 64,
		'scientific_identity_sha256': 'c' * 64,
		'stratigraphy_state_sha256': 'd' * 64,
	}
	metadata = {
		'checkpoint_path': str(checkpoint),
		'checkpoint_sha256': checkpoint_reference['sha256'],
		'stratigraphy_pretext': {
			'method': 'strat_hmm_multi_head_pretext',
			'base_objective': 'amp_mae3d',
			'head_spec': identity['head_spec'],
			'head_ks': identity['head_ks'],
			'head_count': 3,
			'unfreeze_top_blocks': 1,
			'distillation_weight': 0.2,
			'prototype_weight': 1.0,
			'prototype_weight_semantics': 'mean_across_heads',
			'usage_weight': 0.005,
			'usage_weight_semantics': 'mean_across_heads',
			'consistency_policy': identity['consistency_policy'],
			'consistency_weight': identity['consistency_weight'],
			'consistency_beta': identity['consistency_beta'],
			'model_tag': identity['model_tag'],
			'target_representation': identity['target_representation'],
			'target_semantics': identity['target_semantics'],
			'xy_neighbor_consensus_target_manifest_path': target_reference['path'],
			'xy_neighbor_consensus_target_manifest_sha256': target_reference['sha256'],
			'per_head_xy_neighbor_consensus_target_sha256': identity[
				'per_head_xy_neighbor_consensus_targets'
			],
			'source_hard_manifest_sha256': identity['source_hard_manifest_sha256'],
			'xy_neighbor_consensus_smoothing': identity[
				'xy_neighbor_consensus_smoothing'
			],
			'scientific_identity_sha256': identity['scientific_identity_sha256'],
			'checkpoint_stratigraphy_state_sha256': identity[
				'stratigraphy_state_sha256'
			],
		},
	}
	files.metadata.write_text(json.dumps(metadata), encoding='utf-8')
	handoff['embedding'].update(  # type: ignore[index]
		{
			'root': str(embedding_root),
			'metadata_path': str(files.metadata),
			'metadata_sha256': _reference(files.metadata)['sha256'],
			'embeddings_sha256': _reference(files.embeddings)['sha256'],
			'valid_tokens_sha256': _reference(files.valid_tokens)['sha256'],
		}
	)
	paths['handoff'].write_text(json.dumps(handoff), encoding='utf-8')
	config = results.f3_xy_neighbor_consensus_review_config_from_mapping(
		_review_config_mapping(paths)
	)
	monkeypatch.setattr(
		results,
		'validate_stratigraphy_checkpoint_payload',
		lambda _: None,
	)
	monkeypatch.setattr(
		results.torch,
		'load',
		lambda *_args, **_kwargs: {
			'stratigraphy_checkpoint': identity,
			'epoch': 1,
			'global_step': 2,
			'metrics': {'loss': 1.0},
			'training_state': {'checkpoint_kind': 'step'},
		},
	)

	results._validate_handoff_artifact_lineage(  # noqa: SLF001
		config,
		target=target,
		handoff=handoff,
	)


def test_checkpoint_evidence_validates_latest_and_best_against_config(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	root = tmp_path / 'pretraining'
	root.mkdir()
	latest, best = root / 'latest.pt', root / 'best.pt'
	latest.touch()
	best.touch()
	training = {'paths': {'output_root': str(root)}}
	calls: list[tuple[Path, object]] = []

	def checkpoint(
		path: Path,
		*,
		expected_config: object,
	) -> dict[str, object]:
		calls.append((path, expected_config))
		return {
			'global_step': 2,
			'epoch': 1,
			'metrics': {'loss': 1.0, 'loss_consistency': 0.0},
			'training_state': {'checkpoint_kind': 'step'},
			'stratigraphy_checkpoint': {
				'initial_student_state_sha256': 'a' * 64,
				'initial_head_state_sha256': 'b' * 64,
				'consistency_weight': 0.0,
			},
		}

	monkeypatch.setattr(validation, '_checkpoint', checkpoint)
	monkeypatch.setattr(
		validation,
		'_scientific_checkpoint_contract',
		lambda *_args: None,
	)

	validation._checkpoint_evidence(  # noqa: SLF001
		training,
		expected_global_step=2,
		require_best=False,
	)

	assert calls == [(latest, training), (best, training)]


def test_embedding_metadata_identity_rejects_posterior_or_lateral_carryover() -> None:
	identity = {
		'head_spec': 'multi_resolution_ordered_prototypes_v1',
		'head_ks': [6, 8, 10],
		'model_tag': 'strat_hmm_pretext_mh_k6810_xycons1_nocons_topblock1_distill_v1',
		'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
		'target_semantics': 'xy_neighbor_consensus_hard_label_smoothing_v1',
		'xy_neighbor_consensus_target_manifest_sha256': 'a' * 64,
		'xy_neighbor_consensus_target_manifest': {
			'path': 'targets.json',
			'sha256': 'a' * 64,
		},
		'per_head_xy_neighbor_consensus_targets': {'6': {}},
		'source_hard_manifest_sha256': 'b' * 64,
		'xy_neighbor_consensus_smoothing': {'application': 'single_pass'},
		'consistency_policy': 'disabled_for_xy_neighbor_consensus_v1',
		'consistency_weight': 0.0,
		'consistency_beta': 0.1,
		'scientific_identity_sha256': 'c' * 64,
		'stratigraphy_state_sha256': 'd' * 64,
	}
	metadata = {
		'stratigraphy_pretext': {
			'method': 'strat_hmm_multi_head_pretext',
			'base_objective': 'amp_mae3d',
			'head_spec': identity['head_spec'],
			'head_ks': identity['head_ks'],
			'head_count': 3,
			'unfreeze_top_blocks': 1,
			'distillation_weight': 0.2,
			'prototype_weight': 1.0,
			'prototype_weight_semantics': 'mean_across_heads',
			'usage_weight': 0.005,
			'usage_weight_semantics': 'mean_across_heads',
			'consistency_policy': identity['consistency_policy'],
			'consistency_weight': identity['consistency_weight'],
			'consistency_beta': identity['consistency_beta'],
			'model_tag': identity['model_tag'],
			'target_representation': identity['target_representation'],
			'target_semantics': identity['target_semantics'],
			'xy_neighbor_consensus_target_manifest_path': 'targets.json',
			'xy_neighbor_consensus_target_manifest_sha256': identity[
				'xy_neighbor_consensus_target_manifest_sha256'
			],
			'per_head_xy_neighbor_consensus_target_sha256': identity[
				'per_head_xy_neighbor_consensus_targets'
			],
			'source_hard_manifest_sha256': identity['source_hard_manifest_sha256'],
			'xy_neighbor_consensus_smoothing': identity[
				'xy_neighbor_consensus_smoothing'
			],
			'scientific_identity_sha256': identity['scientific_identity_sha256'],
			'checkpoint_stratigraphy_state_sha256': identity[
				'stratigraphy_state_sha256'
			],
		}
	}

	validation._validate_embedding_stratigraphy_identity(  # noqa: SLF001
		metadata,
		identity,
	)
	for field in (
		'source_posterior_manifest_sha256',
		'posterior_manifest_path',
		'lateral_target_manifest_path',
		'per_head_lateral_target_sha256',
	):
		metadata['stratigraphy_pretext'][field] = 'd' * 64  # type: ignore[index]
		with pytest.raises(ValueError, match=field):
			validation._validate_embedding_stratigraphy_identity(  # noqa: SLF001
				metadata,
				identity,
			)
		metadata['stratigraphy_pretext'].pop(field)  # type: ignore[index]
	metadata['stratigraphy_pretext']['xy_neighbor_consensus_target_manifest_path'] = (  # type: ignore[index]
		'stale-targets.json'
	)
	with pytest.raises(ValueError, match='xy_neighbor_consensus_target_manifest_path'):
		validation._validate_embedding_stratigraphy_identity(  # noqa: SLF001
			metadata,
			identity,
		)
	metadata['stratigraphy_pretext']['xy_neighbor_consensus_target_manifest_path'] = (  # type: ignore[index]
		'targets.json'
	)
	metadata['stratigraphy_pretext']['checkpoint_stratigraphy_state_sha256'] = 'e' * 64  # type: ignore[index]
	with pytest.raises(ValueError, match='checkpoint_stratigraphy_state_sha256'):
		validation._validate_embedding_stratigraphy_identity(  # noqa: SLF001
			metadata,
			identity,
		)


@pytest.mark.parametrize(
	('field', 'value'),
	[
		('method', 'wrong_method'),
		('base_objective', 'wrong_objective'),
		('head_spec', 'wrong_head_spec'),
		('head_ks', [6, 8]),
		('head_count', 2),
		('consistency_policy', 'enabled_for_xy_neighbor_consensus_v1'),
		('consistency_weight', 0.1),
		('consistency_beta', 0.2),
	],
)
def test_embedding_metadata_identity_binds_head_and_consistency_claims(
	field: str,
	value: object,
) -> None:
	"""The extracted metadata cannot restate a different v5 training identity."""
	identity = {
		'head_spec': 'multi_resolution_ordered_prototypes_v1',
		'head_ks': [6, 8, 10],
		'model_tag': 'strat_hmm_pretext_mh_k6810_xycons1_nocons_topblock1_distill_v1',
		'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
		'target_semantics': 'xy_neighbor_consensus_hard_label_smoothing_v1',
		'xy_neighbor_consensus_target_manifest_sha256': 'a' * 64,
		'xy_neighbor_consensus_target_manifest': {
			'path': 'targets.json',
			'sha256': 'a' * 64,
		},
		'per_head_xy_neighbor_consensus_targets': {'6': {}},
		'source_hard_manifest_sha256': 'b' * 64,
		'xy_neighbor_consensus_smoothing': {'application': 'single_pass'},
		'consistency_policy': 'disabled_for_xy_neighbor_consensus_v1',
		'consistency_weight': 0.0,
		'consistency_beta': 0.1,
		'scientific_identity_sha256': 'c' * 64,
		'stratigraphy_state_sha256': 'd' * 64,
	}
	stratigraphy = {
		'method': 'strat_hmm_multi_head_pretext',
		'base_objective': 'amp_mae3d',
		'head_spec': identity['head_spec'],
		'head_ks': identity['head_ks'],
		'head_count': 3,
		'unfreeze_top_blocks': 1,
		'distillation_weight': 0.2,
		'prototype_weight': 1.0,
		'prototype_weight_semantics': 'mean_across_heads',
		'usage_weight': 0.005,
		'usage_weight_semantics': 'mean_across_heads',
		'consistency_policy': identity['consistency_policy'],
		'consistency_weight': identity['consistency_weight'],
		'consistency_beta': identity['consistency_beta'],
		'model_tag': identity['model_tag'],
		'scientific_identity_sha256': identity['scientific_identity_sha256'],
		'checkpoint_stratigraphy_state_sha256': identity['stratigraphy_state_sha256'],
		'target_representation': identity['target_representation'],
		'target_semantics': identity['target_semantics'],
		'xy_neighbor_consensus_target_manifest_path': 'targets.json',
		'xy_neighbor_consensus_target_manifest_sha256': identity[
			'xy_neighbor_consensus_target_manifest_sha256'
		],
		'per_head_xy_neighbor_consensus_target_sha256': identity[
			'per_head_xy_neighbor_consensus_targets'
		],
		'source_hard_manifest_sha256': identity['source_hard_manifest_sha256'],
		'xy_neighbor_consensus_smoothing': identity['xy_neighbor_consensus_smoothing'],
	}
	stratigraphy[field] = value

	with pytest.raises(ValueError, match=field):
		validation._validate_embedding_stratigraphy_identity(  # noqa: SLF001
			{'stratigraphy_pretext': stratigraphy},
			identity,
		)


def _review_config_mapping(paths: dict[str, Path]) -> dict[str, str]:
	return {
		'artifact_root': str(paths['artifact_root']),
		'workspace_root': str(paths['workspace_root']),
		'target_manifest': str(paths['target']),
		'pretraining_handoff': str(paths['handoff']),
		'output_dir': str(paths['output_dir']),
	}


def _paths(tmp_path: Path) -> dict[str, Path]:
	workspace_root = tmp_path / 'workspace'
	artifact_root = workspace_root / 'artifacts' / 'seis_ssl_cluster'
	artifact_root.mkdir(parents=True)
	for name in ('hard.json', 'targets.json'):
		(artifact_root / name).write_text('{}\n', encoding='utf-8')
	handoff = artifact_root / 'pretraining/handoff.json'
	handoff.parent.mkdir(parents=True)
	return {
		'workspace_root': workspace_root,
		'artifact_root': artifact_root,
		'hard': artifact_root / 'hard.json',
		'target': artifact_root / 'targets.json',
		'handoff': handoff,
		'output_dir': workspace_root / 'results/f3/xy-neighbor-consensus',
	}


def _reference(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()}


def _handoff(paths: dict[str, Path]) -> dict[str, object]:
	return {
		'artifact_type': 'f3_xy_neighbor_consensus_pretraining_handoff',
		'schema_version': 1,
		'status': 'PASS',
		'model_tag': ('strat_hmm_pretext_mh_k6810_xycons1_nocons_topblock1_distill_v1'),
		'variant': 'xycons1_nocons',
		'targets': {
			'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
			'target_semantics': 'xy_neighbor_consensus_hard_label_smoothing_v1',
			'consistency_policy': 'disabled_for_xy_neighbor_consensus_v1',
			'target_manifest': _reference(paths['target']),
			'xy_neighbor_consensus_target_head_hashes': _head_hashes(),
			'source_hard_manifest': _reference(paths['hard']),
			'xy_neighbor_consensus_smoothing': {
				'application': 'single_pass_synchronous_source_labels'
			},
			'temporal_transition_counts': _transition_counts(),
			'initial_student_state_sha256': 'a' * 64,
			'initial_head_state_sha256': 'b' * 64,
		},
		'checkpoint': {
			'path': str(paths['artifact_root'] / 'pretraining/best.pt'),
			'sha256': 'c' * 64,
			'selected_checkpoint_kind': 'step',
			'selected_epoch': 1,
			'selected_global_step': 2,
			'selected_loss': 1.0,
		},
		'embedding': {
			'root': str(paths['artifact_root'] / 'embeddings'),
			'metadata_path': str(paths['artifact_root'] / 'embeddings/metadata.json'),
			'metadata_sha256': 'd' * 64,
			'embeddings_sha256': 'e' * 64,
			'valid_tokens_sha256': 'f' * 64,
			'valid_token_count': 42,
		},
	}


def _target_manifest(paths: dict[str, Path]) -> dict[str, object]:
	return {
		'head_ks': [6, 8, 10],
		'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
		'target_semantics': 'xy_neighbor_consensus_hard_label_smoothing_v1',
		'source_hard_manifest': _reference(paths['hard']),
		'smoothing': {'application': 'single_pass_synchronous_source_labels'},
		'heads': {
			str(k): {
				'surveys': {
					'survey': {
						name: {'sha256': f'{k:x}' * 64}
						for name in ('labels', 'confidence', 'valid_tokens', 'metadata')
					}
				},
				'diagnostics': {
					'aggregate': {
						'valid_token_count': 10 * k,
						'changed_token_count': 2,
						'changed_fraction': 2 / (10 * k),
						'temporal_transition_counts': {
							'source': k,
							'output': k + 1,
						},
					}
				},
			}
			for k in (6, 8, 10)
		},
	}


def _head_hashes() -> dict[str, dict[str, dict[str, str]]]:
	return {
		str(k): {
			'survey': dict.fromkeys(
				('labels', 'confidence', 'valid_tokens', 'metadata'),
				f'{k:x}' * 64,
			),
		}
		for k in (6, 8, 10)
	}


def _transition_counts() -> dict[str, dict[str, int]]:
	return {str(k): {'source': k, 'output': k + 1} for k in (6, 8, 10)}
