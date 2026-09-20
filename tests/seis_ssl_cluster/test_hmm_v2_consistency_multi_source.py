"""Consistency ablation: strict inheritance, all-new execution and publication."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster.config import (
	load_config,
	resolve_embedding_extraction_config,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	f3_lithology_candidate_config_from_mapping,
)
from seis_ssl_cluster.hmm import multi_source_aggregate as aggregate
from seis_ssl_cluster.hmm import multi_source_receipts as receipts
from seis_ssl_cluster.hmm import multi_source_survey as survey_summary
from seis_ssl_cluster.hmm.multi_source_consistency import baseline_paths
from seis_ssl_cluster.hmm.multi_source_definition import (
	validate_multi_source_experiment_definition,
)
from seis_ssl_cluster.hmm.multi_source_driver import command_plan
from seis_ssl_cluster.hmm.multi_source_protocol import (
	K468_CONS010,
	K6810_CONS010,
	experiment_protocol,
)
from seis_ssl_cluster.parihaka.channel_decoder import (
	channel_decoder_config_from_mapping,
)
from seis_ssl_cluster.stratigraphy.multi_head import build_multi_head_target_manifest
from seis_ssl_cluster.volve.horizon_recipe_arm import (
	as_five_way_config,
	volve_horizon_recipe_arm_config_from_mapping,
)
from tests.seis_ssl_cluster.test_config_strat_hmm_multi_head import _multi_head_config
from tests.seis_ssl_cluster.test_hmm_k6810_receipts_aggregate import (  # noqa: F401
	synthetic,
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
ROOTS = tuple(sorted(WORKSPACE.glob('experiments/*/*/*_cons010_multi_source_v1')))
PROTOCOLS = (K468_CONS010, K6810_CONS010)


@pytest.mark.usefixtures('configured_environment')
@pytest.mark.parametrize('root', ROOTS, ids=lambda p: p.name)
def test_all_new_arms_resolve_and_preserve_source_and_decoder(root, tmp_path):
	protocol = experiment_protocol(root)
	plan = command_plan(root, _args())
	assert len(plan) == 58
	assert sum(stage == 'full' for stage, _ in plan) == 3
	assert sum(stage == 'downstream' for stage, _ in plan) == 45
	assert not any(stage in ('targets', 'export', 'manifests') for stage, _ in plan)
	fixture = _multi_head_config(tmp_path / 'teacher')
	embeddings, heads = _artifacts(tmp_path / 'targets', ks=protocol.head_ks)
	manifest = tmp_path / 'manifest.json'
	build_multi_head_target_manifest(
		manifest_path=manifest,
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=_replay_k6_root(tmp_path / 'targets', heads[6]),
	)
	for family, cid in protocol.candidate_ids.items():
		refs = baseline_paths(root, family)
		baseline = yaml.safe_load(refs['training'].read_text())
		full = yaml.safe_load(
			(root / '30_pretraining' / cid / '02_full_25ep.yaml').read_text()
		)
		for section in (
			'pseudo_targets',
			'teacher',
			'student',
			'model',
			'data',
			'zero_mask',
			'train',
			'head',
			'manifests',
		):
			assert full[section] == baseline[section]
		for filename in ('01_gpu_feasibility_1step.yaml', '02_full_25ep.yaml'):
			raw = load_config(root / '30_pretraining' / cid / filename)
			raw['pseudo_targets']['manifest'] = str(manifest)
			raw['identity']['scientific_identity']['target_manifest_sha256'] = (
				receipts.file_sha256(manifest)
			)
			raw['teacher']['checkpoint'] = fixture['teacher']['checkpoint']
			raw['student']['init_checkpoint'] = fixture['teacher']['checkpoint']
			resolved = resolve_strat_hmm_pretext_config(raw)
			assert resolved['identity']['scientific_identity']['variant'] == 'cons010'
			assert resolved['loss']['consistency_weight'] == 0.1
			assert resolved['loss']['distillation_weight'] == 0.2
		embedding = resolve_embedding_extraction_config(
			load_config(root / '40_embeddings' / f'{cid}.yaml')
		)
		down = load_config(root / '50_downstream' / f'{cid}.yaml')
		assert protocol.namespace in down['outputs']['runs_root']
		if root.parts[-3] == 'f3':
			config = f3_lithology_candidate_config_from_mapping(down)
			assert str(config.checkpoint) == embedding['embeddings']['checkpoint']
			assert str(config.embeddings_dir) == embedding['embeddings']['output_dir']
		elif root.parts[-3] == 'parihaka':
			channel_decoder_config_from_mapping(down)
			assert (
				down['embeddings']['models'][cid]['checkpoint']
				== embedding['embeddings']['checkpoint']
			)
		else:
			config = volve_horizon_recipe_arm_config_from_mapping(down)
			assert as_five_way_config(config).models[0].expected['head_ks'] == list(
				protocol.head_ks
			)
			assert str(config.arm_checkpoint) == embedding['embeddings']['checkpoint']
		resume = command_plan(
			root, _args(first='full', last='full', candidate=cid, resume=True)
		)
		assert resume[0][1][-1].endswith(f'{cid}/full_25ep/latest.pt')


@pytest.fixture
def copied_definition(tmp_path):
	root = ROOTS[0]
	dest = tmp_path / root.relative_to(WORKSPACE)
	shutil.copytree(root, dest)
	for family in receipts.SOURCE_FAMILIES:
		for path in baseline_paths(root, family).values():
			target = tmp_path / path.relative_to(WORKSPACE)
			target.parent.mkdir(parents=True, exist_ok=True)
			shutil.copyfile(path, target)
	protocol = experiment_protocol(root)
	matrix = WORKSPACE / 'experiments/hmm_v2' / protocol.study / 'matrix.yaml'
	target = tmp_path / matrix.relative_to(WORKSPACE)
	target.parent.mkdir(parents=True)
	shutil.copyfile(matrix, target)
	return dest


@pytest.mark.parametrize(
	'change',
	[
		'weight',
		'target',
		'parent',
		'seed',
		'output',
		'decoder',
		'baseline',
		'reuse',
		'manifest_hash',
	],
)
def test_definition_rejects_scientific_drift_and_output_overlap(
	copied_definition, change
):
	root = copied_definition
	protocol = experiment_protocol(root)
	cid = protocol.candidate_ids['mae']
	path = root / '30_pretraining' / cid / '02_full_25ep.yaml'
	value = yaml.safe_load(path.read_text())
	if change == 'weight':
		value['loss']['consistency_weight'] = 0.0
	elif change == 'target':
		value['pseudo_targets']['manifest'] += '.changed'
	elif change == 'parent':
		value['student']['init_checkpoint'] = (
			value['paths']['output_root'] + '/latest.pt'
		)
	elif change == 'seed':
		value['train']['seed'] = 43
	elif change == 'output':
		value['paths']['output_root'] = yaml.safe_load(
			baseline_paths(root, 'mae')['training'].read_text()
		)['paths']['output_root']
	elif change == 'decoder':
		path = root / '50_downstream' / f'{cid}.yaml'
		value = yaml.safe_load(path.read_text())
		value['outputs']['runs_root'] += '_changed'
	elif change == 'manifest_hash':
		value['identity']['scientific_identity']['target_manifest_sha256'] = '0' * 64
	elif change == 'baseline':
		path = baseline_paths(root, 'mae')['training']
		value = yaml.safe_load(path.read_text())
		value['train']['seed'] = 43
	else:
		path = root / 'execution.yaml'
		value = yaml.safe_load(path.read_text())
		del value['arms']['mae']
	path.write_text(yaml.safe_dump(value))
	with pytest.raises(ValueError, match=r'drift|inheritance|matrix'):
		validate_multi_source_experiment_definition(root)


@pytest.fixture(params=PROTOCOLS, ids=lambda p: p.tag)
def consistency_synthetic(synthetic, request):  # noqa: F811
	config, root = synthetic
	protocol = request.param
	config.pop('selection_receipt')
	config['matrix'] = str(
		WORKSPACE / 'experiments/hmm_v2' / protocol.study / 'matrix.yaml'
	)
	for survey, entry in config['summaries'].items():
		path = Path(entry['path'])
		payload = receipts.read_json(path)
		payload.pop('selection_receipt')
		payload['matrix'] = {
			'path': config['matrix'],
			'sha256': receipts.file_sha256(Path(config['matrix'])),
		}
		for cell in payload['cells']:
			cell['candidate_id'] = protocol.candidate_ids[cell['source_family']]
			cell['reused_existing'] = False
		payload['rows'] = [
			aggregate.comparison_row(
				survey,
				family,
				size,
				[c for c in payload['cells'] if c['source_family'] == family],
				protocol,
			)
			for family in receipts.SOURCE_FAMILIES
			for size in receipts.DATA_SIZES
		]
		payload.pop('summary_sha256')
		payload['summary_sha256'] = receipts.object_sha256(payload)
		write_json(path, payload)
		entry['sha256'] = receipts.file_sha256(path)
	return config, root, protocol


def test_all_new_aggregate_exact_publication(consistency_synthetic):
	config, root, protocol = consistency_synthetic
	payload = aggregate.inspect_aggregate(config)
	assert payload['completeness']['new_arms'] == 9
	assert payload['completeness']['new_cells'] == 135
	assert payload['completeness']['reused_cells'] == 0
	assert not any(c['reused_existing'] for c in payload['cells'])
	paths = aggregate.summarize_aggregate(config)
	assert {p.name for p in paths} == {'comparison.csv', 'summary.json', 'summary.md'}
	assert set((root / 'output').iterdir()) == set(paths)
	assert f'{protocol.tag}_mean' in paths[0].read_text()
	with pytest.raises(FileExistsError):
		aggregate.summarize_aggregate(config)


@pytest.mark.parametrize('survey', receipts.SURVEYS)
def test_all_new_survey_exact_publication(consistency_synthetic, monkeypatch, survey):
	config, root, protocol = consistency_synthetic
	cells = receipts.read_json(Path(config['summaries'][survey]['path']))['cells']
	experiment = next(
		p for p in ROOTS if p.parts[-3] == survey and experiment_protocol(p) == protocol
	)
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
		lambda _c, f, _r: ([c for c in cells if c['source_family'] == f], {}),
	)
	paths = survey_summary.summarize_survey(config)
	assert {p.name for p in paths} == {'comparison.csv', 'summary.json', 'summary.md'}
	assert set((root / 'output').iterdir()) == set(paths)
	payload = receipts.read_json(paths[1])
	assert not any(c['reused_existing'] for c in payload['cells'])


@pytest.mark.parametrize('change', ['candidate', 'reuse', 'matrix', 'selection'])
def test_aggregate_rejects_zero_consistency_evidence(consistency_synthetic, change):
	config, root, _protocol = consistency_synthetic
	if change == 'selection':
		config['selection_receipt'] = str(root / 'reuse.json')
	else:
		entry = config['summaries']['f3']
		path = Path(entry['path'])
		payload = receipts.read_json(path)
		if change == 'candidate':
			payload['cells'][0]['candidate_id'] = payload['cells'][0][
				'candidate_id'
			].removesuffix('_cons010')
		elif change == 'reuse':
			payload['cells'][0]['reused_existing'] = True
		else:
			payload['matrix']['sha256'] = 'a' * 64
		payload.pop('summary_sha256')
		payload['summary_sha256'] = receipts.object_sha256(payload)
		write_json(path, payload)
		entry['sha256'] = receipts.file_sha256(path)
	with pytest.raises(ValueError, match=r'drift|cannot reuse'):
		aggregate.summarize_aggregate(config)
	assert not (root / 'output').exists()
