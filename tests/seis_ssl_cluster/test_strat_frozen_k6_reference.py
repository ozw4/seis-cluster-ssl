"""Immutable historical K=6 evidence without reclustering or replay targets."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from proc.seis_ssl_cluster.build_strat_hmm_multi_head_targets import main
from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.stratigraphy import multi_head
from tests.seis_ssl_cluster.test_strat_multi_head_target_manifest import (
	_artifacts,
	_replay_k6_root,
)


@pytest.fixture
def frozen_inputs(tmp_path: Path) -> dict:
	embeddings, heads = _artifacts(tmp_path, schema_version=2)
	checkpoint = tmp_path / 'parent.pt'
	checkpoint.write_bytes(b'original source checkpoint')
	metadata_path = embeddings / 'survey.embedding_metadata.json'
	metadata = json.loads(metadata_path.read_text())
	metadata.update(
		checkpoint_path=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint)
	)
	metadata_path.write_text(json.dumps(metadata))
	training = tmp_path / 'historical_training.yaml'
	training.write_text(
		yaml.safe_dump(
			{
				'pseudo_targets': {'input_dir': str(heads[6]), 'k': 6},
				'teacher': {'checkpoint': str(checkpoint)},
				'student': {'init_checkpoint': str(checkpoint)},
			}
		)
	)
	return {
		'embeddings': embeddings,
		'heads': heads,
		'checkpoint': checkpoint,
		'training': training,
		'receipt': tmp_path / 'new_namespace/frozen_k6.json',
		'manifest': tmp_path / 'new_namespace/manifest.json',
	}


def _freeze(inputs: dict, *, dry_run: bool = False) -> dict:
	return multi_head.build_frozen_k6_reference_receipt(
		receipt_path=inputs['receipt'],
		reference_training_config=inputs['training'],
		historical_root=inputs['heads'][6],
		source_embedding_dir=inputs['embeddings'],
		dry_run=dry_run,
	)


def _manifest(inputs: dict) -> dict:
	return multi_head.build_multi_head_target_manifest(
		manifest_path=inputs['manifest'],
		source_embedding_dir=inputs['embeddings'],
		head_roots=inputs['heads'],
		frozen_k6_receipt=inputs['receipt'],
	)


def test_frozen_manifest_roundtrip_never_calls_replay(
	frozen_inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
	def forbidden(**_kwargs: object) -> None:
		raise AssertionError('frozen references must not call replay')

	monkeypatch.setattr(multi_head, 'compare_k6_replay', forbidden)
	before = {
		path: file_sha256(path)
		for root in (frozen_inputs['embeddings'], frozen_inputs['heads'][6])
		for path in root.rglob('*')
		if path.is_file()
	}
	payload = _freeze(frozen_inputs)
	receipt_bytes = frozen_inputs['receipt'].read_bytes()
	assert _freeze(frozen_inputs) == payload
	assert frozen_inputs['receipt'].read_bytes() == receipt_bytes
	manifest = _manifest(frozen_inputs)
	assert 'k6_replay_parity' not in manifest
	assert set(manifest['k6_frozen_reference']['target_hashes']['survey']) == {
		'labels',
		'confidence',
		'valid_tokens',
		'metadata',
		'boundary_weight',
	}
	assert (
		multi_head.load_multi_head_target_manifest(frozen_inputs['manifest'])
		== manifest
	)
	assert {path: file_sha256(path) for path in before} == before


@pytest.mark.parametrize(
	'field', ['labels', 'confidence', 'valid_tokens', 'metadata', 'boundary_weight']
)
def test_every_historical_target_hash_is_immutable(
	frozen_inputs: dict, field: str
) -> None:
	payload = _freeze(frozen_inputs)
	_manifest(frozen_inputs)
	path = Path(payload['targets']['survey'][field]['path'])
	path.write_bytes(path.read_bytes() + b'changed')
	with pytest.raises(ValueError, match='hash mismatch'):
		multi_head.load_multi_head_target_manifest(frozen_inputs['manifest'])


@pytest.mark.parametrize('item', ['checkpoint', 'training', 'embedding', 'receipt'])
def test_frozen_manifest_rejects_source_or_receipt_drift(
	frozen_inputs: dict, item: str
) -> None:
	_freeze(frozen_inputs)
	_manifest(frozen_inputs)
	path = (
		(frozen_inputs['embeddings'] / 'survey.embeddings.npy')
		if item == 'embedding'
		else frozen_inputs[item]
	)
	path.write_bytes(path.read_bytes() + b' ')
	with pytest.raises(ValueError, match=r'hash mismatch|evidence drift'):
		multi_head.load_multi_head_target_manifest(frozen_inputs['manifest'])


def test_freeze_dry_run_publishes_nothing_and_existing_receipt_never_refreezes(
	frozen_inputs: dict,
) -> None:
	_freeze(frozen_inputs, dry_run=True)
	assert not frozen_inputs['receipt'].parent.exists()
	_freeze(frozen_inputs)
	before = frozen_inputs['receipt'].read_bytes()
	frozen_inputs['training'].write_text(frozen_inputs['training'].read_text() + '\n')
	with pytest.raises(ValueError, match='refreeze'):
		_freeze(frozen_inputs)
	assert frozen_inputs['receipt'].read_bytes() == before


def test_k6_manifest_requires_exactly_one_evidence_mode(frozen_inputs: dict) -> None:
	_freeze(frozen_inputs)
	payload = _manifest(frozen_inputs)
	for change in ('both', 'neither'):
		changed = copy.deepcopy(payload)
		if change == 'both':
			changed['k6_replay_parity'] = {}
		else:
			del changed['k6_frozen_reference']
		with pytest.raises(ValueError, match='exactly one'):
			multi_head.validate_multi_head_target_manifest(changed)


def test_cli_freezes_then_builds_and_only_missing_is_read_only(
	frozen_inputs: dict,
) -> None:
	base = [
		'--source-embedding-dir',
		str(frozen_inputs['embeddings']),
		'--frozen-k6-receipt',
		str(frozen_inputs['receipt']),
		'--only-missing',
	]
	freeze = [
		*base,
		'--freeze-k6-receipt',
		'--reference-training-config',
		str(frozen_inputs['training']),
		'--head-root',
		f'6={frozen_inputs["heads"][6]}',
	]
	assert main([*freeze, '--dry-run']) == 0
	assert not frozen_inputs['receipt'].exists()
	assert main(freeze) == 0
	build = [*base, '--manifest', str(frozen_inputs['manifest'])]
	for k, root in frozen_inputs['heads'].items():
		build += ['--head-root', f'{k}={root}']
	assert main(build) == 0
	before = frozen_inputs['manifest'].read_bytes()
	assert main(build) == 0
	assert frozen_inputs['manifest'].read_bytes() == before


def test_frozen_manifest_cannot_be_replaced_by_replay(frozen_inputs: dict) -> None:
	_freeze(frozen_inputs)
	_manifest(frozen_inputs)
	before = frozen_inputs['manifest'].read_bytes()
	replay = _replay_k6_root(
		frozen_inputs['manifest'].parent, frozen_inputs['heads'][6], schema_version=2
	)
	with pytest.raises(FileExistsError, match='frozen-reference manifest'):
		multi_head.build_multi_head_target_manifest(
			manifest_path=frozen_inputs['manifest'],
			source_embedding_dir=frozen_inputs['embeddings'],
			head_roots=frozen_inputs['heads'],
			replay_k6_root=replay,
		)
	arguments = [
		'--source-embedding-dir',
		str(frozen_inputs['embeddings']),
		'--manifest',
		str(frozen_inputs['manifest']),
		'--replay-k6-root',
		str(replay),
		'--only-missing',
	]
	for k, root in frozen_inputs['heads'].items():
		arguments += ['--head-root', f'{k}={root}']
	with pytest.raises(ValueError, match='cannot change evidence mode'):
		main(arguments)
	assert frozen_inputs['manifest'].read_bytes() == before
