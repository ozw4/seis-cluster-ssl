"""K468 preparation, portable configs, receipt isolation and exact publication."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster.config import (
	load_config,
	resolve_clustering_config,
	resolve_embedding_extraction_config,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	f3_lithology_candidate_config_from_mapping,
)
from seis_ssl_cluster.hmm import multi_source_aggregate as aggregate
from seis_ssl_cluster.hmm import multi_source_receipts as receipts
from seis_ssl_cluster.hmm import multi_source_survey as survey_summary
from seis_ssl_cluster.hmm.multi_source_driver import command_plan
from seis_ssl_cluster.hmm.multi_source_protocol import K468
from seis_ssl_cluster.parihaka.channel_decoder import (
	channel_decoder_config_from_mapping,
)
from seis_ssl_cluster.stratigraphy.multi_head import build_multi_head_target_manifest
from seis_ssl_cluster.volve.horizon_recipe_arm import (
	as_five_way_config,
	volve_horizon_recipe_arm_config_from_mapping,
)
from tests.seis_ssl_cluster.test_config_strat_hmm_multi_head import _multi_head_config
from tests.seis_ssl_cluster.test_hmm_k6810_receipts_aggregate import (
	seal,
	synthetic,  # noqa: F401
	write_json,
)
from tests.seis_ssl_cluster.test_hmm_v2_multi_source_configs import (
	configured_environment,  # noqa: F401
)
from tests.seis_ssl_cluster.test_hmm_v2_multi_source_driver import _args
from tests.seis_ssl_cluster.test_strat_multi_head_target_manifest import (
	_artifacts,
	_replay_k6_root,
)

WORKSPACE = Path(__file__).resolve().parents[2]
STUDY = WORKSPACE / 'experiments/hmm_v2/k468_multi_source_evaluation_v1'
ROOTS = tuple(sorted(WORKSPACE.glob('experiments/*/*/*hmm_v2_k468_multi_source_v1')))


@pytest.mark.usefixtures('configured_environment')
@pytest.mark.parametrize('root', ROOTS, ids=lambda p: p.parts[-3])
def test_k468_plan_and_all_stage_configs(root, tmp_path):
	plan = command_plan(root, _args())
	definition = yaml.safe_load((root / 'execution.yaml').read_text())
	survey = definition['survey']
	count = 2 if survey == 'f3' else 3
	assert len(plan) == count * 23 + 1
	assert sum(stage == 'full' for stage, _ in plan) == count
	assert sum(stage == 'downstream' for stage, _ in plan) == count * 15
	assert all('k6810' not in ' '.join(command) for _, command in plan)
	assert (
		all(K468.candidate_ids['mae'] not in ' '.join(c) for _, c in plan)
		if survey == 'f3'
		else True
	)
	fixture = _multi_head_config(tmp_path / 'teacher')
	embeddings, heads = _artifacts(tmp_path / 'targets', ks=K468.head_ks)
	manifest = tmp_path / 'manifest.json'
	build_multi_head_target_manifest(
		manifest_path=manifest,
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=_replay_k6_root(tmp_path / 'targets', heads[6]),
	)
	for entry in definition['arms'].values():
		cid = entry['candidate_id']
		targets = root / '10_targets' / entry['target_family']
		cluster = resolve_clustering_config(
			load_config(targets / K468.cluster_filename)
		)
		assert cluster['clustering']['k_values'] == [4, 8]
		assert '/surveys/' in cluster['clustering']['output_dir']
		exports = (targets / '03_export_pseudo_targets.sh').read_text()
		assert '--k 4 ' in exports
		assert '--k 8 ' in exports
		assert '--k 6 ' not in exports
		assert '--k 10 ' not in exports
		for filename in ('01_gpu_feasibility_1step.yaml', '02_full_25ep.yaml'):
			raw = load_config(root / '30_pretraining' / cid / filename)
			raw['pseudo_targets']['manifest'] = str(manifest)
			raw['identity']['scientific_identity']['target_manifest_sha256'] = (
				receipts.file_sha256(manifest)
			)
			raw['teacher']['checkpoint'] = fixture['teacher']['checkpoint']
			raw['student']['init_checkpoint'] = fixture['teacher']['checkpoint']
			resolved = resolve_strat_hmm_pretext_config(raw)
			assert resolved['head']['ks'] == [4, 6, 8]
			assert resolved['loss']['distillation_weight'] == 0.2
			assert '/surveys/' in resolved['paths']['output_root']
		embedding = resolve_embedding_extraction_config(
			load_config(root / '40_embeddings' / f'{cid}.yaml')
		)
		assert embedding['embedding']['prefetch_queue_depth'] == 2
		down = load_config(root / '50_downstream' / f'{cid}.yaml')
		assert '/surveys/' in down['outputs']['runs_root']
		if survey == 'f3':
			f3_lithology_candidate_config_from_mapping(down)
		elif survey == 'parihaka':
			channel_decoder_config_from_mapping(down)
		else:
			config = volve_horizon_recipe_arm_config_from_mapping(down)
			assert as_five_way_config(config).models[0].expected['head_ks'] == [4, 6, 8]


def test_k468_reuse_receipt_cannot_be_used_as_k6810():
	matrix = receipts.load_matrix(STUDY / 'matrix.yaml')
	assert matrix['selected_head_ks'] == [4, 6, 8]
	path = STUDY / 'f3_mae_k468_reuse_receipt.json'
	reuse = receipts.load_reuse_receipt(path, K468)
	assert len(reuse['cells']) == 15
	assert reuse['candidate_id'] == K468.candidate_ids['mae']
	with pytest.raises(ValueError, match='scientific contract drift'):
		receipts.load_reuse_receipt(path)
	with pytest.raises(ValueError, match='reuse'):
		command_plan(
			ROOTS[0],
			_args(first='full', last='full', candidate=K468.candidate_ids['mae']),
		)


@pytest.fixture
def k468_synthetic(synthetic):  # noqa: F811
	config, root = synthetic
	config['matrix'] = str(STUDY / 'matrix.yaml')
	path = Path(config['selection_receipt'])
	reuse = receipts.read_json(path)
	reuse['candidate_id'] = K468.candidate_ids['mae']
	reuse['head_ks'] = list(K468.head_ks)
	write_json(path, seal(reuse))
	for entry in config['summaries'].values():
		path = Path(entry['path'])
		payload = json.loads(path.read_text().replace('k6810', 'k468'))
		payload['selection_receipt']['sha256'] = receipts.file_sha256(
			Path(config['selection_receipt'])
		)
		payload.pop('summary_sha256')
		payload['summary_sha256'] = receipts.object_sha256(payload)
		write_json(path, payload)
		entry['sha256'] = receipts.file_sha256(path)
	return config, root


def test_k468_aggregate_publishes_correct_labels_and_exact_files(k468_synthetic):
	config, root = k468_synthetic
	payload = aggregate.inspect_aggregate(config)
	assert payload['completeness']['new_cells'] == 120
	assert len(payload['rows']) == 27
	assert len(payload['cells']) == 135
	paths = aggregate.summarize_aggregate(config)
	assert {p.name for p in paths} == {'comparison.csv', 'summary.json', 'summary.md'}
	assert set((root / 'output').iterdir()) == set(paths)
	assert 'k468_mean' in paths[0].read_text().splitlines()[0]
	assert 'k6810_mean' not in paths[0].read_text()
	assert '# K468 multi-source evaluation' in paths[2].read_text()
	with pytest.raises(FileExistsError):
		aggregate.summarize_aggregate(config)


@pytest.mark.parametrize('survey', receipts.SURVEYS)
def test_k468_survey_publishes_exact_files(k468_synthetic, monkeypatch, survey):
	config, root = k468_synthetic
	cells = receipts.read_json(Path(config['summaries'][survey]['path']))['cells']
	experiment = next(p for p in ROOTS if p.parts[-3] == survey)
	definition = yaml.safe_load(
		(experiment / '60_summary/01_paired_comparison.yaml')
		.read_text()
		.replace('${SEIS_SSL_CLUSTER_WORKSPACE}', str(WORKSPACE))
	)
	config.update(
		schema_version=1,
		survey=survey,
		artifact_root=str(root),
		experiment_root=str(experiment),
		arms=definition['arms'],
	)
	monkeypatch.setattr(
		survey_summary,
		'_audited_arm',
		lambda _config, family, _receipt: (
			[c for c in cells if c['source_family'] == family],
			{},
		),
	)
	paths = survey_summary.summarize_survey(config)
	assert {p.name for p in paths} == {'comparison.csv', 'summary.json', 'summary.md'}
	assert set((root / 'output').iterdir()) == set(paths)
	assert 'k468_mean' in paths[0].read_text().splitlines()[0]
	assert 'K468 multi-source comparison' in paths[2].read_text()


@pytest.mark.parametrize('change', ['matrix', 'candidate'])
def test_k468_aggregate_rejects_cross_study_evidence(k468_synthetic, change):
	config, root = k468_synthetic
	if change == 'matrix':
		config['matrix'] = str(
			STUDY.with_name('k6810_multi_source_evaluation_v1') / 'matrix.yaml'
		)
	else:
		entry = config['summaries']['volve']
		path = Path(entry['path'])
		payload = receipts.read_json(path)
		payload['cells'][0]['candidate_id'] = receipts.CANDIDATE_IDS['random']
		payload.pop('summary_sha256')
		payload['summary_sha256'] = receipts.object_sha256(payload)
		write_json(path, payload)
		entry['sha256'] = receipts.file_sha256(path)
	with pytest.raises(ValueError, match='drift'):
		aggregate.summarize_aggregate(config)
	assert not (root / 'output').exists()
