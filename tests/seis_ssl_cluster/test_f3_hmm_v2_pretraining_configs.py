from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from proc.seis_ssl_cluster import train_strat_hmm_pretext
from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.stratigraphy.multi_head import build_multi_head_target_manifest
from tests.seis_ssl_cluster.test_f3_hmm_v2_manifest_configs import (
	CANDIDATES,
	EXPERIMENT,
	configs,  # noqa: F401
	live_configs,  # noqa: F401
)

if TYPE_CHECKING:
	from typing import Any

BASELINE = Path(
	'experiments/f3/facies_benchmark_v1/21_ssl_hmm_continuation_v1/'
	'30_stage2/mae100/hmm/k6'
)
MODES = {
	'01_gpu_feasibility_1step.yaml': 'smoke_1step',
	'02_full_25ep.yaml': 'full_25ep',
}


def _path(tag: str, filename: str) -> Path:
	return (
		EXPERIMENT / '30_pretraining' / f'mae100_hmm_v2_mh_k{tag}_distill020' / filename
	)


def test_all_configs_preserve_condition_2b_except_multi_head_contract(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256', 'a' * 64)
	outputs = set()
	for tag, ks in CANDIDATES.items():
		candidate = f'mae100_hmm_v2_mh_k{tag}_distill020'
		manifest = load_config(EXPERIMENT / '20_manifests' / f'{candidate}.yaml')
		for filename, mode in MODES.items():
			config = load_config(_path(tag, filename))
			baseline = load_config(BASELINE / filename)
			expected = deepcopy(baseline)
			expected['paths']['output_root'] = str(
				tmp_path
				/ 'pretraining/f3/facies_benchmark_v1/hmm_v2_multi_head_screening_v1'
				/ candidate
				/ mode
			)
			expected['pseudo_targets'] = {
				'manifest': manifest['manifest'],
				'target_representation': 'hard_viterbi_labels_v1',
				'min_confidence': baseline['pseudo_targets']['min_confidence'],
			}
			del expected['head']['num_prototypes']
			expected['head'].update(
				spec='multi_resolution_ordered_prototypes_v1', ks=ks
			)
			expected['loss'].update(consistency_weight=0.0, consistency_beta=0.1)
			identity = config.pop('identity')
			assert identity == {
				'model_tag': candidate,
				'scientific_identity': {
					'experiment_role': 'multi_head_ordered_pretext',
					'variant': 'nocons',
					'head_spec': expected['head']['spec'],
					'head_ks': ks,
					'target_manifest_sha256': 'a' * 64,
					'consistency_policy': 'normalized_order_smooth_l1_v1',
				},
				'runtime_identity': {
					'device': 'cuda',
					'workers': baseline['train']['num_workers'],
				},
			}
			assert config == expected
			outputs.add(config['paths']['output_root'])
	assert len(outputs) == 8
	assert len(list((EXPERIMENT / '30_pretraining').glob('*/*.yaml'))) == 8


@pytest.mark.parametrize('tag', CANDIDATES)
@pytest.mark.parametrize('filename', MODES)
def test_training_configs_resolve_and_cli_dry_run_without_training(
	live_configs: dict[str, dict[str, Any]],  # noqa: F811
	tag: str,
	filename: str,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	build = live_configs[tag]
	manifest = Path(build['manifest'])
	build_multi_head_target_manifest(
		manifest_path=manifest,
		source_embedding_dir=Path(build['source_embedding_dir']),
		head_roots={k: Path(root) for k, root in build['head_roots'].items()},
		replay_k6_root=Path(build['replay_k6_root']),
	)
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256', file_sha256(manifest)
	)
	config = load_config(_path(tag, filename))
	# Resolution checks checkpoint existence; dry-run must never deserialize it.
	checkpoint = Path(config['teacher']['checkpoint'])
	checkpoint.parent.mkdir(parents=True, exist_ok=True)
	checkpoint.write_bytes(b'config-test checkpoint placeholder')
	resolved = resolve_strat_hmm_pretext_config(config)
	assert resolved['head']['ks'] == CANDIDATES[tag]
	assert resolved['identity']['scientific_identity']['head_ks'] == CANDIDATES[tag]
	assert (
		resolved['pseudo_targets']['target_representation'] == 'hard_viterbi_labels_v1'
	)

	def fail_training(*_args: object, **_kwargs: object) -> None:
		pytest.fail('dry-run must not start training')

	monkeypatch.setattr(
		train_strat_hmm_pretext, 'run_strat_hmm_pretext_training', fail_training
	)
	monkeypatch.setattr(
		sys, 'argv', ['train', '--config', str(_path(tag, filename)), '--dry-run']
	)
	train_strat_hmm_pretext.main()
	assert 'execution: dry-run; training skipped' in capsys.readouterr().out
	assert not Path(config['paths']['output_root']).exists()
	for section, key, value, match in (
		(
			'identity',
			'model_tag',
			'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1',
			'model_tag',
		),
		('loss', 'consistency_weight', 0.1, 'consistency_weight'),
		('head', 'ks', [4, 10], 'head'),
	):
		invalid = deepcopy(config)
		invalid[section][key] = value
		with pytest.raises(ValueError, match=match):
			resolve_strat_hmm_pretext_config(invalid)
	invalid = deepcopy(config)
	invalid['identity']['scientific_identity']['target_manifest_sha256'] = '0' * 64
	with pytest.raises(ValueError, match='target_manifest_sha256'):
		resolve_strat_hmm_pretext_config(invalid)
