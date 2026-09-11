from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from proc.seis_ssl_cluster.build_strat_hmm_multi_head_targets import main
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from tests.seis_ssl_cluster.test_strat_multi_head_target_manifest import (
	_artifacts,
	_replay_k6_root,
)

if TYPE_CHECKING:
	from typing import Any

EXPERIMENT = Path(
	'experiments/f3/facies_benchmark_v2/127_hmm_v2_multi_head_screening_v1'
)
CANDIDATES = {'46': [4, 6], '68': [6, 8], '468': [4, 6, 8], '6810': [6, 8, 10]}
NAMESPACE = 'pseudo_targets/f3/facies_benchmark_v1/hmm_v2_multi_head_screening_v1'


@pytest.fixture
def configs(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, dict[str, Any]]:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	return {
		tag: load_config(
			EXPERIMENT / '20_manifests' / f'mae100_hmm_v2_mh_k{tag}_distill020.yaml'
		)
		for tag in CANDIDATES
	}


def test_configs_bind_task01_roots_and_disjoint_outputs(
	configs: dict[str, dict[str, Any]], tmp_path: Path
) -> None:
	source = load_config(EXPERIMENT / '10_targets/01_cluster_hmm_k4810.yaml')
	outputs = set()
	for tag, config in configs.items():
		assert set(config) == {
			'source_embedding_dir',
			'head_roots',
			'replay_k6_root',
			'manifest',
		}
		assert list(config['head_roots']) == CANDIDATES[tag]
		assert config['source_embedding_dir'] == source['embeddings']['input_dir']
		for k, root in config['head_roots'].items():
			expected = (
				tmp_path
				/ 'pseudo_targets/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/mae100'
				if k == 6
				else tmp_path / NAMESPACE / 'mae100/shared'
			)
			assert Path(root) == expected
		assert (
			Path(config['replay_k6_root']) == tmp_path / NAMESPACE / 'mae100/k6_replay'
		)
		assert config['replay_k6_root'] != config['head_roots'][6]
		output = Path(config['manifest'])
		assert (
			output
			== tmp_path / NAMESPACE / f'mh_k{tag}/multi_head_target_manifest.json'
		)
		outputs.add(output.parent)
	assert len(outputs) == len(CANDIDATES)
	assert len(list((EXPERIMENT / '20_manifests').glob('*.yaml'))) == len(CANDIDATES)
	assert not list(tmp_path.iterdir())


@pytest.fixture
def live_configs(
	configs: dict[str, dict[str, Any]], tmp_path: Path
) -> dict[str, dict[str, Any]]:
	heads = {
		k: Path(root)
		for config in configs.values()
		for k, root in config['head_roots'].items()
	}
	config = configs['46']
	_artifacts(
		tmp_path,
		ks=(4, 6, 8, 10),
		embedding_root=Path(config['source_embedding_dir']),
		head_roots=heads,
		schema_version=2,
	)
	_replay_k6_root(
		tmp_path,
		heads[6],
		replay_root=Path(config['replay_k6_root']),
		schema_version=2,
	)
	return configs


def _arguments(config: dict[str, Any]) -> list[str]:
	args = [
		'--source-embedding-dir',
		config['source_embedding_dir'],
		'--manifest',
		config['manifest'],
		'--replay-k6-root',
		config['replay_k6_root'],
		'--only-missing',
	]
	for k, root in config['head_roots'].items():
		args.extend(['--head-root', f'{k}={root}'])
	return args


def test_four_configured_manifests_roundtrip_and_share_identity(
	live_configs: dict[str, dict[str, Any]],
) -> None:
	identities = []
	for tag, config in live_configs.items():
		manifest = Path(config['manifest'])
		args = _arguments(config)
		assert main([*args, '--dry-run']) == 0
		assert not manifest.exists()
		assert main(args) == 0
		payload = load_multi_head_target_manifest(manifest)
		assert payload['head_ks'] == list(config['head_roots']) == CANDIDATES[tag]
		assert payload['k6_replay_parity']['exact'] is True
		assert all(payload['k6_replay_parity']['checks'].values())
		assert payload['k6_replay_parity']['replay_root'] == config['replay_k6_root']
		for k, root in config['head_roots'].items():
			assert payload['heads'][str(k)]['pseudo_target_root'] == root
		identities.append(payload['source_embedding'])
		before = manifest.read_bytes()
		assert main(args) == 0
		assert manifest.read_bytes() == before
	assert all(identity == identities[0] for identity in identities)


@pytest.mark.parametrize('tag', CANDIDATES)
def test_every_candidate_rejects_replay_mismatch(
	live_configs: dict[str, dict[str, Any]],
	tag: str,
) -> None:
	config = live_configs[tag]
	metadata = json.loads(
		(
			Path(config['replay_k6_root']) / 'k6/survey.pseudo_target_metadata.json'
		).read_text()
	)
	decoded = Path(metadata['source']['source_label_path'])
	labels = np.load(decoded)
	labels[0, 0, 0] = 1
	np.save(decoded, labels)
	with pytest.raises(ValueError, match='parity'):
		main(_arguments(config))
	assert not Path(config['manifest']).exists()
