"""Contracts for the schema-v1 K=6/8/10 target manifest."""

from __future__ import annotations

import importlib
import json
import shutil
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.stratigraphy.multi_head import (
	build_multi_head_target_manifest,
	compare_k6_replay,
	load_multi_head_target_manifest,
	validate_multi_head_target_manifest,
)
from seis_ssl_cluster.stratigraphy.targets import write_pseudo_target

if TYPE_CHECKING:
	from collections.abc import Callable


def test_manifest_roundtrip_rejects_hash_tamper(
	tmp_path: Path,
) -> None:
	embeddings, heads = _artifacts(tmp_path)
	manifest = tmp_path / 'manifest.json'
	payload = build_multi_head_target_manifest(
		manifest_path=manifest,
		source_embedding_dir=embeddings,
		head_roots={'6': heads[6], '8': heads[8], '10': heads[10]},
		replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
	)

	assert payload['head_ks'] == [6, 8, 10]
	assert set(payload['cross_head_diagnostics']) == {'k6_k8', 'k6_k10', 'k8_k10'}
	load_multi_head_target_manifest(manifest)
	(heads[8] / 'k8' / 'survey.hmm_labels_token.npy').write_bytes(b'tampered')
	with pytest.raises(ValueError, match='hash mismatch'):
		load_multi_head_target_manifest(manifest)


def test_manifest_rejects_replay_root_as_historical_k6_training_target(
	tmp_path: Path,
) -> None:
	embeddings, heads = _artifacts(tmp_path)

	with pytest.raises(ValueError, match='immutable historical'):
		build_multi_head_target_manifest(
			manifest_path=tmp_path / 'manifest.json',
			source_embedding_dir=embeddings,
			head_roots=heads,
			replay_k6_root=heads[6],
		)


@pytest.mark.parametrize(
	'head_ks',
	[[8, 6, 10], [6, 6, 10]],
)
def test_manifest_load_rejects_persisted_unsorted_or_duplicate_head_ks(
	tmp_path: Path,
	head_ks: list[int],
) -> None:
	embeddings, heads = _artifacts(tmp_path)
	manifest = tmp_path / 'manifest.json'
	payload = build_multi_head_target_manifest(
		manifest_path=manifest,
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
	)
	payload['head_ks'] = head_ks
	manifest.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='head_ks'):
		load_multi_head_target_manifest(manifest)


def test_manifest_supports_dynamic_increasing_head_ks(tmp_path: Path) -> None:
	"""The artifact contract is not coupled to the active K=6/8/10 experiment."""
	embeddings, heads = _artifacts(tmp_path, ks=(4, 7))
	manifest = tmp_path / 'manifest.json'
	payload = build_multi_head_target_manifest(
		manifest_path=manifest,
		source_embedding_dir=embeddings,
		head_roots=heads,
	)

	assert payload['head_ks'] == [4, 7]
	assert set(payload['cross_head_diagnostics']) == {'k4_k7'}
	assert 'k6_replay_parity' not in payload
	load_multi_head_target_manifest(manifest)


def test_k6_replay_parity_rejects_one_token_mismatch(tmp_path: Path) -> None:
	embeddings, heads = _artifacts(tmp_path)
	labels = np.tile(np.minimum(np.arange(12), 5), (2, 2, 1)).astype(np.int32)
	labels[0, 0, 0] = 1
	replay = _replay_k6_root(tmp_path, heads[6], labels=labels)
	assert not compare_k6_replay(historical_root=heads[6], replay_root=replay)['exact']
	with pytest.raises(ValueError, match='parity'):
		build_multi_head_target_manifest(
			manifest_path=tmp_path / 'manifest.json',
			source_embedding_dir=embeddings,
			head_roots=heads,
			replay_k6_root=replay,
		)


def test_k6_replay_parity_rejects_valid_mask_mismatch(tmp_path: Path) -> None:
	embeddings, heads = _artifacts(tmp_path)
	replay = _replay_k6_root(tmp_path, heads[6])
	valid_path = replay / 'k6' / 'survey.valid_tokens.npy'
	labels_path = replay / 'k6' / 'survey.hmm_labels_token.npy'
	confidence_path = replay / 'k6' / 'survey.hmm_confidence_token.npy'
	valid = np.load(valid_path)
	labels = np.load(labels_path)
	confidence = np.load(confidence_path)
	valid[0, 0, 0] = False
	labels[0, 0, 0] = -1
	confidence[0, 0, 0] = 0.0
	np.save(valid_path, valid)
	np.save(labels_path, labels)
	np.save(confidence_path, confidence)

	parity = compare_k6_replay(historical_root=heads[6], replay_root=replay)
	assert not parity['exact']
	assert parity['checks']['survey.pseudo_target_valid_tokens'] is False  # type: ignore[index]
	with pytest.raises(ValueError, match='parity'):
		build_multi_head_target_manifest(
			manifest_path=tmp_path / 'manifest.json',
			source_embedding_dir=embeddings,
			head_roots=heads,
			replay_k6_root=replay,
		)


def test_k6_replay_parity_compares_replayed_clustering_decoded_labels(
	tmp_path: Path,
) -> None:
	embeddings, heads = _artifacts(tmp_path)
	replay = _replay_k6_root(tmp_path, heads[6])
	initial = compare_k6_replay(
		historical_root=heads[6],
		replay_root=replay,
	)
	replayed_labels = Path(
		initial['replay_decoded_labels']['survey']['path'],  # type: ignore[index]
	)
	labels = np.load(replayed_labels)
	labels[0, 0, 0] = 1
	np.save(replayed_labels, labels)

	parity = compare_k6_replay(historical_root=heads[6], replay_root=replay)
	assert not parity['exact']
	assert parity['checks']['survey.decoded_labels'] is False  # type: ignore[index]
	with pytest.raises(ValueError, match='parity'):
		build_multi_head_target_manifest(
			manifest_path=tmp_path / 'manifest.json',
			source_embedding_dir=embeddings,
			head_roots=heads,
			replay_k6_root=replay,
		)


def test_manifest_rejects_missing_k6_replay_parity(tmp_path: Path) -> None:
	"""A complete manifest must preserve exact K=6 replay evidence."""
	embeddings, heads = _artifacts(tmp_path)
	payload = build_multi_head_target_manifest(
		manifest_path=tmp_path / 'manifest.json',
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
	)
	payload.pop('k6_replay_parity')

	with pytest.raises(ValueError, match='replay parity evidence'):
		validate_multi_head_target_manifest(payload)


def test_replay_config_uses_a_required_artifact_root_environment_variable(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config_path = Path(
		'experiments/f3/facies_benchmark_v1/'
		'94_strat_hmm_multi_head_k6810_v1/01_replay_hmm_k6810.yaml',
	)
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'/test/artifacts/seis_ssl_cluster',
	)
	config = load_config(config_path)

	assert config['paths'] == {
		'artifact_root': '/test/artifacts/seis_ssl_cluster',
	}

	monkeypatch.delenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT')
	with pytest.raises(
		ValueError,
		match='config environment variable is required',
	):
		load_config(config_path)


def test_manifest_rejects_target_valid_outside_source_embedding(
	tmp_path: Path,
) -> None:
	embeddings, heads = _artifacts(tmp_path)
	valid_path = embeddings / 'survey.valid_tokens.npy'
	valid = np.load(valid_path)
	valid[0, 0, 0] = False
	np.save(valid_path, valid)

	with pytest.raises(ValueError, match='valid-token mask is not a subset'):
		build_multi_head_target_manifest(
			manifest_path=tmp_path / 'manifest.json',
			source_embedding_dir=embeddings,
			head_roots=heads,
			replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
		)


def test_manifest_accepts_edge_excluded_target_mask_as_source_subset(
	tmp_path: Path,
) -> None:
	shape = (2, 2, 12)
	target_valid = np.ones(shape, dtype=np.bool_)
	target_valid[0, :, :] = False
	embeddings, heads = _artifacts(tmp_path, target_valid_tokens=target_valid)
	manifest = tmp_path / 'manifest.json'
	payload = build_multi_head_target_manifest(
		manifest_path=manifest,
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
	)

	alignment = payload['common']['source_target_alignment']['survey']  # type: ignore[index]
	assert alignment == {
		'source_valid_count': 48,
		'target_valid_count': 24,
		'excluded_from_source_count': 24,
		'target_is_subset_of_source': True,
	}
	assert (
		payload['source_embedding']['surveys']['survey']['valid_tokens_sha256']  # type: ignore[index]
		!= payload['common']['valid_tokens_sha256']['survey']  # type: ignore[index]
	)
	load_multi_head_target_manifest(manifest, validate_array_semantics=False)
	load_multi_head_target_manifest(manifest)
	payload['common']['source_target_alignment']['survey'][  # type: ignore[index]
		'target_is_subset_of_source'
	] = False
	with pytest.raises(ValueError, match='must record a subset'):
		validate_multi_head_target_manifest(
			payload,
			verify_hashes=True,
			validate_array_semantics=False,
		)


def test_manifest_loader_supports_legacy_v1_exact_mask_contract(
	tmp_path: Path,
) -> None:
	"""V1 remains loadable only for its original source-mask-equality contract."""
	embeddings, heads = _artifacts(tmp_path)
	manifest = tmp_path / 'legacy-manifest.json'
	payload = build_multi_head_target_manifest(
		manifest_path=manifest,
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
	)
	payload['schema_version'] = 1
	payload['common'].pop('source_target_alignment')  # type: ignore[index]
	manifest.write_text(json.dumps(payload), encoding='utf-8')

	assert load_multi_head_target_manifest(manifest)['schema_version'] == 1
	assert load_multi_head_target_manifest(
		manifest,
		validate_array_semantics=False,
	)['schema_version'] == 1


def test_legacy_v1_manifest_rejects_edge_excluded_target_mask(
	tmp_path: Path,
) -> None:
	shape = (2, 2, 12)
	target_valid = np.ones(shape, dtype=np.bool_)
	target_valid[0, :, :] = False
	embeddings, heads = _artifacts(tmp_path, target_valid_tokens=target_valid)
	manifest = tmp_path / 'legacy-manifest.json'
	payload = build_multi_head_target_manifest(
		manifest_path=manifest,
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
	)
	payload['schema_version'] = 1
	payload['common'].pop('source_target_alignment')  # type: ignore[index]
	manifest.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(
		ValueError, match=r'legacy v1.*does not match source embedding'
	):
		load_multi_head_target_manifest(manifest, validate_array_semantics=False)


def test_manifest_rejects_cross_head_target_mask_mismatch(tmp_path: Path) -> None:
	embeddings, heads = _artifacts(tmp_path)
	valid_path = heads[8] / 'k8' / 'survey.valid_tokens.npy'
	labels_path = heads[8] / 'k8' / 'survey.hmm_labels_token.npy'
	confidence_path = heads[8] / 'k8' / 'survey.hmm_confidence_token.npy'
	valid = np.load(valid_path)
	labels = np.load(labels_path)
	confidence = np.load(confidence_path)
	valid[0, 0, 0] = False
	labels[0, 0, 0] = -1
	confidence[0, 0, 0] = 0.0
	np.save(valid_path, valid)
	np.save(labels_path, labels)
	np.save(confidence_path, confidence)

	with pytest.raises(ValueError, match='valid-token masks differ'):
		build_multi_head_target_manifest(
			manifest_path=tmp_path / 'manifest.json',
			source_embedding_dir=embeddings,
			head_roots=heads,
			replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
		)


def test_manifest_full_validation_rechecks_target_source_subset(
	tmp_path: Path,
) -> None:
	shape = (2, 2, 12)
	target_valid = np.ones(shape, dtype=np.bool_)
	target_valid[0, :, :] = False
	embeddings, heads = _artifacts(tmp_path, target_valid_tokens=target_valid)
	source_valid_path = embeddings / 'survey.valid_tokens.npy'
	source_valid = np.load(source_valid_path)
	source_valid[0, 0, 0] = False
	np.save(source_valid_path, source_valid)
	payload = build_multi_head_target_manifest(
		manifest_path=tmp_path / 'manifest.json',
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
	)
	for k in (6, 8, 10):
		valid_path = heads[k] / f'k{k}' / 'survey.valid_tokens.npy'
		labels_path = heads[k] / f'k{k}' / 'survey.hmm_labels_token.npy'
		confidence_path = heads[k] / f'k{k}' / 'survey.hmm_confidence_token.npy'
		valid = np.load(valid_path)
		labels = np.load(labels_path)
		confidence = np.load(confidence_path)
		valid[0, 0, 0] = True
		labels[0, 0, 0] = 0
		confidence[0, 0, 0] = 1.0
		for path, array, key in (
			(valid_path, valid, 'valid_tokens'),
			(labels_path, labels, 'labels'),
			(confidence_path, confidence, 'confidence'),
		):
			np.save(path, array)
			payload['heads'][str(k)]['surveys']['survey'][key]['sha256'] = sha256(  # type: ignore[index]
				path.read_bytes(),
			).hexdigest()
	payload['common']['valid_tokens_sha256']['survey'] = payload['heads']['6'][  # type: ignore[index]
		'surveys'
	]['survey']['valid_tokens']['sha256']

	with pytest.raises(ValueError, match='not a subset'):
		validate_multi_head_target_manifest(payload)


def test_manifest_full_validation_rechecks_target_embedding_grid_shape(
	tmp_path: Path,
) -> None:
	embeddings, heads = _artifacts(tmp_path)
	payload = build_multi_head_target_manifest(
		manifest_path=tmp_path / 'manifest.json',
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
	)
	valid_path = embeddings / 'survey.valid_tokens.npy'
	np.save(valid_path, np.ones((2, 2, 11), dtype=np.bool_))
	payload['source_embedding']['surveys']['survey']['valid_tokens_sha256'] = (  # type: ignore[index]
		sha256(valid_path.read_bytes()).hexdigest()
	)

	with pytest.raises(ValueError, match='token grid does not match'):
		validate_multi_head_target_manifest(payload, verify_hashes=True)


def test_cli_dry_run_only_missing_does_not_quarantine_invalid_manifest(
	tmp_path: Path,
	capsys: pytest.CaptureFixture[str],
) -> None:
	embeddings, heads = _artifacts(tmp_path)
	manifest = tmp_path / 'manifest.json'
	manifest.write_text('{not valid JSON', encoding='utf-8')
	predictable_dry_run = manifest.with_name(f'.{manifest.name}.dry-run')
	predictable_dry_run.write_text('preserve', encoding='utf-8')
	module = importlib.import_module(
		'proc.seis_ssl_cluster.build_strat_hmm_multi_head_targets',
	)

	assert module.main(
		[
			'--source-embedding-dir',
			str(embeddings),
			'--head-root',
			f'6={heads[6]}',
			'--head-root',
			f'8={heads[8]}',
			'--head-root',
			f'10={heads[10]}',
			'--replay-k6-root',
			str(_replay_k6_root(tmp_path, heads[6])),
			'--manifest',
			str(manifest),
			'--only-missing',
			'--quarantine-invalid',
			'--dry-run',
		]
	) == 0

	assert manifest.read_text(encoding='utf-8') == '{not valid JSON'
	assert predictable_dry_run.read_text(encoding='utf-8') == 'preserve'
	assert not manifest.with_name(f'{manifest.name}.quarantine').exists()
	assert 'would quarantine:' in capsys.readouterr().out


def test_cli_only_missing_rebuilds_for_stale_requested_inputs(
	tmp_path: Path,
	capsys: pytest.CaptureFixture[str],
) -> None:
	embeddings, heads = _artifacts(tmp_path)
	alternate_embeddings = tmp_path / 'alternate_embeddings'
	shutil.copytree(embeddings, alternate_embeddings)
	alternate_k6 = tmp_path / 'alternate_k6'
	shutil.copytree(heads[6], alternate_k6)
	alternate_replay = _replay_k6_root(tmp_path, alternate_k6)
	manifest = tmp_path / 'manifest.json'
	build_multi_head_target_manifest(
		manifest_path=manifest,
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
	)
	module = importlib.import_module(
		'proc.seis_ssl_cluster.build_strat_hmm_multi_head_targets',
	)

	assert module.main(
		[
			'--source-embedding-dir',
			str(alternate_embeddings),
			'--head-root',
			f'6={alternate_k6}',
			'--head-root',
			f'8={heads[8]}',
			'--head-root',
			f'10={heads[10]}',
			'--replay-k6-root',
			str(alternate_replay),
			'--manifest',
			str(manifest),
			'--only-missing',
		]
	) == 0

	payload = load_multi_head_target_manifest(manifest)
	assert payload['source_embedding']['input_dir'] == str(alternate_embeddings)  # type: ignore[index]
	assert payload['heads']['6']['pseudo_target_root'] == str(alternate_k6)  # type: ignore[index]
	assert payload['k6_replay_parity']['replay_root'] == str(alternate_replay)  # type: ignore[index]
	assert 'reused complete manifest' not in capsys.readouterr().out


def test_cli_only_missing_refuses_stale_k6_replay_evidence(
	tmp_path: Path,
	capsys: pytest.CaptureFixture[str],
) -> None:
	embeddings, heads = _artifacts(tmp_path)
	replay = _replay_k6_root(tmp_path, heads[6])
	manifest = tmp_path / 'manifest.json'
	build_multi_head_target_manifest(
		manifest_path=manifest,
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=replay,
	)
	labels_path = replay / 'k6' / 'survey.hmm_labels_token.npy'
	labels = np.load(labels_path)
	labels[0, 0, 0] = 1
	np.save(labels_path, labels)
	module = importlib.import_module(
		'proc.seis_ssl_cluster.build_strat_hmm_multi_head_targets',
	)

	with pytest.raises(ValueError, match='existing manifest is invalid'):
		module.main(
			[
				'--source-embedding-dir',
				str(embeddings),
				'--head-root',
				f'6={heads[6]}',
				'--head-root',
				f'8={heads[8]}',
				'--head-root',
				f'10={heads[10]}',
				'--replay-k6-root',
				str(replay),
				'--manifest',
				str(manifest),
				'--only-missing',
			]
		)

	assert 'reused complete manifest' not in capsys.readouterr().out


@pytest.mark.parametrize(
	('common_key', 'value', 'match'),
	[
		('token_grid_shapes', [2, 2, 11], 'token grid does not match'),
		('valid_tokens_sha256', 'falsified', 'valid-token hash does not match'),
	],
)
def test_manifest_rejects_falsified_common_target_contract(
	tmp_path: Path, common_key: str, value: object, match: str
) -> None:
	embeddings, heads = _artifacts(tmp_path)
	payload = build_multi_head_target_manifest(
		manifest_path=tmp_path / 'manifest.json',
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
	)
	payload['common'][common_key]['survey'] = value  # type: ignore[index]

	with pytest.raises(ValueError, match=match):
		validate_multi_head_target_manifest(payload, verify_hashes=True)


def _ordered_path_violation(array: np.ndarray) -> np.ndarray:
	result = array.copy()
	result[0, 0, 2] = 0
	return result


@pytest.mark.parametrize(
	('artifact_name', 'mutate', 'match'),
	[
		(
			'labels',
			lambda array: np.where(array == 7, 6, array),
			'empty state',
		),
		(
			'labels',
			_ordered_path_violation,
			'ordered path violations',
		),
		(
			'confidence',
			lambda array: np.where(
				np.indices(array.shape)[-1] == 0,
				np.inf,
				array,
			),
			'confidence must be finite',
		),
	],
)
def test_manifest_load_revalidates_referenced_target_semantics(
	tmp_path: Path,
	artifact_name: str,
	mutate: Callable[[np.ndarray], np.ndarray],
	match: str,
) -> None:
	embeddings, heads = _artifacts(tmp_path)
	payload = build_multi_head_target_manifest(
		manifest_path=tmp_path / 'manifest.json',
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
	)
	artifact = heads[8] / 'k8' / f'survey.hmm_{artifact_name}_token.npy'
	array = np.load(artifact)
	np.save(artifact, mutate(array))
	reference = payload['heads']['8']['surveys']['survey'][artifact_name]  # type: ignore[index]
	reference['sha256'] = sha256(artifact.read_bytes()).hexdigest()  # type: ignore[index]

	with pytest.raises(ValueError, match=match):
		validate_multi_head_target_manifest(payload, verify_hashes=True)


@pytest.mark.parametrize(
	('path', 'value'),
	[
		(('heads', '6', 'surveys', 'survey', 'labels', 'unexpected'), 'value'),
		(('source_embedding', 'surveys', 'survey', 'unexpected'), 'value'),
	],
)
def test_manifest_rejects_unknown_nested_reference_fields(
	tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
	embeddings, heads = _artifacts(tmp_path)
	payload = build_multi_head_target_manifest(
		manifest_path=tmp_path / 'manifest.json',
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
	)
	entry: object = payload
	for key in path[:-1]:
		entry = entry[key]  # type: ignore[index]
	entry[path[-1]] = value  # type: ignore[index]

	with pytest.raises(ValueError, match='unknown fields'):
		validate_multi_head_target_manifest(payload)


def test_manifest_rejects_unknown_head_diagnostics_fields(tmp_path: Path) -> None:
	embeddings, heads = _artifacts(tmp_path)
	payload = build_multi_head_target_manifest(
		manifest_path=tmp_path / 'manifest.json',
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
	)
	payload['heads']['8']['diagnostics']['per_survey']['survey']['unexpected'] = 1  # type: ignore[index]

	with pytest.raises(ValueError, match='unknown fields'):
		validate_multi_head_target_manifest(payload)


def test_manifest_rejects_unknown_k6_replay_parity_fields(tmp_path: Path) -> None:
	embeddings, heads = _artifacts(tmp_path)
	payload = build_multi_head_target_manifest(
		manifest_path=tmp_path / 'manifest.json',
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
	)
	payload['k6_replay_parity']['checks']['unexpected'] = True  # type: ignore[index]

	with pytest.raises(ValueError, match='unknown fields'):
		validate_multi_head_target_manifest(payload)


def _artifacts(
	tmp_path: Path,
	*,
	ks: tuple[int, ...] = (6, 8, 10),
	source_root: Path | None = None,
	target_valid_tokens: np.ndarray | None = None,
) -> tuple[Path, dict[int, Path]]:
	shape = (2, 2, 12)
	target_valid = (
		np.ones(shape, dtype=np.bool_)
		if target_valid_tokens is None
		else np.asarray(target_valid_tokens, dtype=np.bool_)
	)
	if target_valid.shape != shape:
		raise ValueError('target_valid_tokens shape mismatch')
	embeddings = tmp_path / 'embeddings'
	embeddings.mkdir(parents=True)
	np.save(embeddings / 'survey.embeddings.npy', np.zeros((*shape, 3), np.float32))
	np.save(embeddings / 'survey.valid_tokens.npy', np.ones(shape, np.bool_))
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
			},
		),
	)
	heads: dict[int, Path] = {}
	for k in ks:
		root = tmp_path / f'head{k}'
		labels = np.tile(np.minimum(np.arange(12), k - 1), (2, 2, 1)).astype(
			np.int32,
		)
		labels[~target_valid] = -1
		confidence = np.ones(labels.shape, dtype=np.float32)
		confidence[~target_valid] = 0.0
		source_label_path = (
			(source_root or tmp_path)
			/ f'head{k}_clustering'
			/ 'labels'
			/ f'k{k}'
			/ 'survey.cluster_labels_token.npy'
		)
		source_label_path.parent.mkdir(parents=True, exist_ok=True)
		np.save(source_label_path, labels)
		write_pseudo_target(
			root,
			k=k,
			survey_id='survey',
			labels=labels,
			confidence=confidence,
			valid_tokens=target_valid,
			metadata={
				'source_clustering_output_dir': str(source_label_path.parents[2]),
				'source_label_path': str(source_label_path),
			},
			schema_version=1,
			write_boundary_weight=False,
		)
		heads[k] = root
	return embeddings, heads


def _replay_k6_root(
	tmp_path: Path,
	historical_root: Path,
	*,
	labels: np.ndarray | None = None,
) -> Path:
	index = 0
	while (tmp_path / f'replay_k6_{index}').exists():
		index += 1
	replay_root = tmp_path / f'replay_k6_{index}'
	historical_labels = np.load(
		historical_root / 'k6' / 'survey.hmm_labels_token.npy',
	)
	historical_valid = np.load(historical_root / 'k6' / 'survey.valid_tokens.npy')
	historical_confidence = np.load(
		historical_root / 'k6' / 'survey.hmm_confidence_token.npy',
	)
	replay_labels = historical_labels if labels is None else labels
	source_label_path = (
		tmp_path
		/ f'replay_k6_clustering_{index}'
		/ 'labels'
		/ 'k6'
		/ 'survey.cluster_labels_token.npy'
	)
	source_label_path.parent.mkdir(parents=True)
	np.save(source_label_path, replay_labels)
	write_pseudo_target(
		replay_root,
		k=6,
		survey_id='survey',
		labels=replay_labels,
		confidence=historical_confidence,
		valid_tokens=historical_valid,
		metadata={
			'source_clustering_output_dir': str(source_label_path.parents[2]),
			'source_label_path': str(source_label_path),
		},
		schema_version=1,
		write_boundary_weight=False,
	)
	return replay_root
