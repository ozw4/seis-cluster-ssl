import copy
import json
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster.hmm import multi_source_aggregate as aggregate
from seis_ssl_cluster.hmm import multi_source_receipts as receipts
from seis_ssl_cluster.hmm import multi_source_survey

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'experiments/hmm_v2/k6810_multi_source_evaluation_v1'


def seal(value):
	value.pop('receipt_sha256', None)
	return {**value, 'receipt_sha256': receipts.object_sha256(value)}


def write_json(path, value):
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value))
	return path


def metric_cells(survey):
	return [
		{
			'layout_id': layout,
			'data_size': size,
			'primary': 5.0 if survey == 'volve' else 0.4,
			'secondary': 0.5,
		}
		for layout, size in sorted(receipts.CELL_IDENTITIES)
	]


@pytest.fixture
def synthetic(tmp_path):
	controls = []
	for survey in receipts.SURVEYS:
		for family in receipts.SOURCE_FAMILIES:
			cells = metric_cells(survey)
			controls.append(
				{
					'survey': survey,
					'source_family': family,
					'condition_id': receipts.CONDITIONS[family],
					**receipts.METRICS[survey],
					'checkpoint_id': f'{survey}_{family}_k6',
					'checkpoint_path': f'{survey}/{family}/latest.pt',
					'checkpoint_sha256': 'a' * 64,
					'cells': cells,
					'canonical_cells_sha256': receipts.canonical_cells_sha256(
						survey, cells
					),
				}
			)
	control_path = write_json(
		tmp_path / 'controls.json', seal({'schema_version': 1, 'controls': controls})
	)
	cells = metric_cells('f3')
	for i, cell in enumerate(cells):
		cell['metrics_path'] = f'cell_{i}.json'
		path = write_json(
			tmp_path / cell['metrics_path'],
			{'mean_iou': cell['primary'], 'macro_f1': cell['secondary']},
		)
		cell['metrics_sha256'] = receipts.file_sha256(path)
	reuse = {
		'schema_version': 1,
		'candidate_id': receipts.CANDIDATE_IDS['mae'],
		'head_ks': [6, 8, 10],
		'distillation_weight': 0.2,
		'consistency_weight': 0.0,
		'cells': cells,
		'canonical_cells_sha256': receipts.canonical_cells_sha256('f3', cells),
	}
	for name in (
		'checkpoint',
		'embeddings',
		'valid_tokens',
		'embedding_metadata',
		'source_summary',
		'resolved_training_config',
	):
		path = tmp_path / name
		path.write_bytes(b'synthetic')
		reuse[name] = {'path': name, 'sha256': receipts.file_sha256(path)}
	reuse_path = write_json(tmp_path / 'reuse.json', seal(reuse))
	config = {
		'matrix': str(EXPERIMENT / 'matrix.yaml'),
		'control_receipt': str(control_path),
		'selection_receipt': str(reuse_path),
		'summary_root': str(tmp_path / 'output'),
		'summaries': {},
	}
	for survey in receipts.SURVEYS:
		all_cells, rows = [], []
		for family in receipts.SOURCE_FAMILIES:
			cells = metric_cells(survey)
			for cell in cells:
				cell.update(
					source_family=family,
					candidate_id=receipts.CANDIDATE_IDS[family],
					reused_existing=(survey, family) == ('f3', 'mae'),
					control_primary=cell['primary'],
					control_secondary=cell['secondary'],
				)
				if not cell['reused_existing']:
					cell['primary'] += -0.1 if survey == 'volve' else 0.1
			rows.extend(
				aggregate.comparison_row(survey, family, size, cells)
				for size in receipts.DATA_SIZES
			)
			all_cells.extend(cells)
		payload = {
			'schema_version': 1,
			'status': 'complete',
			'survey': survey,
			**receipts.METRICS[survey],
			'cells': all_cells,
			'rows': rows,
		}
		for name in ('control_receipt', 'selection_receipt'):
			path = Path(config[name])
			payload[name] = {'path': str(path), 'sha256': receipts.file_sha256(path)}
		payload['summary_sha256'] = receipts.object_sha256(payload)
		path = write_json(tmp_path / survey / 'summary.json', payload)
		config['summaries'][survey] = {
			'path': str(path),
			'sha256': receipts.file_sha256(path),
		}
	return config, tmp_path


def test_versioned_matrix_and_receipts():
	matrix = receipts.load_matrix(EXPERIMENT / 'matrix.yaml')
	assert (
		sum(
			a['execution'] == 'new'
			for s in matrix['surveys'].values()
			for a in s['arms'].values()
		)
		== 8
	)
	assert (
		len(
			receipts.load_control_receipts(EXPERIMENT / 'hmm_v1_controls_receipt.json')[
				'controls'
			]
		)
		== 9
	)
	assert (
		len(
			receipts.load_reuse_receipt(EXPERIMENT / 'f3_mae_k6810_reuse_receipt.json')[
				'cells'
			]
		)
		== 15
	)


@pytest.mark.parametrize('change', ['head', 'execution', 'tuning', 'survey'])
def test_matrix_rejects_drift(tmp_path, change):
	matrix = yaml.safe_load((EXPERIMENT / 'matrix.yaml').read_text())
	if change == 'head':
		matrix['selected_head_ks'] = [4, 6, 8]
	elif change == 'execution':
		matrix['surveys']['f3']['arms']['mae']['execution'] = 'new'
	elif change == 'tuning':
		matrix['surveys']['parihaka']['select_by_test'] = True
	else:
		del matrix['surveys']['volve']
	path = tmp_path / 'matrix.yaml'
	path.write_text(yaml.safe_dump(matrix))
	with pytest.raises(
		ValueError,
		match=r'matrix|survey|contract|drift|expected|requires|duplicate|disagrees',
	):
		receipts.load_matrix(path)


@pytest.mark.parametrize(
	'change', ['missing', 'duplicate', 'primary', 'secondary', 'checkpoint', 'family']
)
def test_control_rejects_drift(synthetic, change):
	config, _ = synthetic
	entry = copy.deepcopy(
		receipts.load_control_receipts(Path(config['control_receipt']))['controls'][0]
	)
	cells = copy.deepcopy(entry['cells'])
	checkpoint = entry['checkpoint_sha256']
	if change == 'missing':
		cells.pop()
	elif change == 'duplicate':
		cells[-1] = cells[0]
	elif change in ('primary', 'secondary'):
		cells[0][change] += 0.01
	elif change == 'checkpoint':
		checkpoint = 'b' * 64
	else:
		entry['condition_id'] = '4b'
	with pytest.raises(
		ValueError,
		match=r'matrix|survey|contract|drift|expected|requires|duplicate|disagrees',
	):
		receipts.verify_control(entry, checkpoint, cells)


@pytest.mark.parametrize(
	'change',
	[
		'candidate_id',
		'head_ks',
		'checkpoint',
		'embeddings',
		'valid_tokens',
		'source_summary',
		'cell',
	],
)
def test_reuse_rejects_drift(synthetic, change):
	config, root = synthetic
	receipt = receipts.load_reuse_receipt(Path(config['selection_receipt']))
	if change in ('candidate_id', 'head_ks'):
		receipt[change] = 'wrong'
		receipt = seal(receipt)
	elif change == 'cell':
		(root / receipt['cells'][0]['metrics_path']).write_text('{}')
	else:
		(root / receipt[change]['path']).write_bytes(b'changed')
	with pytest.raises(
		ValueError,
		match=r'matrix|survey|contract|drift|expected|requires|duplicate|disagrees',
	):
		receipts.verify_reuse_receipt(receipt, root)


def test_aggregate_exact_outputs_and_positive_improvements(synthetic):
	config, root = synthetic
	payload = aggregate.inspect_aggregate(config)
	assert len(payload['rows']) == 27
	assert len(payload['cells']) == 135
	assert sum(c['reused_existing'] for c in payload['cells']) == 15
	assert payload['completeness']['new_cells'] == 120
	assert all(
		r['primary_improvement'] > 0
		for r in payload['rows']
		if not r['reused_existing']
	)
	assert all(a['layout_clustered']['cluster_count'] == 5 for a in payload['arms'])
	paths = aggregate.summarize_aggregate(config)
	assert {p.name for p in paths} == {'comparison.csv', 'summary.json', 'summary.md'}
	assert {p.name for p in (root / 'output').iterdir()} == {p.name for p in paths}
	with pytest.raises(FileExistsError):
		aggregate.summarize_aggregate(config)


@pytest.mark.parametrize(
	'change',
	[
		'missing_survey',
		'duplicate_row',
		'missing_cell',
		'metric',
		'direction',
		'reuse',
		'source_hash',
		'receipt_hash',
	],
)
def test_aggregate_rejects_partial_or_drifting_inputs(synthetic, change):
	config, root = synthetic
	if change == 'missing_survey':
		del config['summaries']['volve']
	else:
		entry = config['summaries']['f3']
		path = Path(entry['path'])
		payload = json.loads(path.read_text())
		if change == 'duplicate_row':
			payload['rows'][-1] = payload['rows'][0]
		elif change == 'missing_cell':
			payload['cells'].pop()
		elif change == 'metric':
			payload['rows'][0]['k6810_mean'] += 0.01
		elif change == 'direction':
			payload['rows'][0]['primary_direction'] = 'lower'
		elif change == 'reuse':
			payload['cells'][0]['reused_existing'] = False
		elif change == 'receipt_hash':
			payload['control_receipt']['sha256'] = 'b' * 64
		else:
			payload['status'] = 'changed'
		payload.pop('summary_sha256')
		payload['summary_sha256'] = receipts.object_sha256(payload)
		write_json(path, payload)
		if change != 'source_hash':
			entry['sha256'] = receipts.file_sha256(path)
	with pytest.raises(
		ValueError,
		match=r'matrix|survey|contract|drift|expected|requires|duplicate|disagrees',
	):
		aggregate.summarize_aggregate(config)
	assert not (root / 'output').exists()


@pytest.mark.parametrize('survey', receipts.SURVEYS)
def test_survey_summary_complete_atomic_and_paired(synthetic, monkeypatch, survey):
	config, root = synthetic
	existing = receipts.read_json(Path(config['summaries'][survey]['path']))
	config = {
		**config,
		'schema_version': 1,
		'survey': survey,
		'artifact_root': str(root),
		'arms': {f: {} for f in receipts.SOURCE_FAMILIES},
	}

	def audit(_config, family, _receipt):
		return [c for c in existing['cells'] if c['source_family'] == family], {
			'complete': True
		}

	monkeypatch.setattr(multi_source_survey, '_audited_arm', audit)
	payload = multi_source_survey.inspect_survey(config)
	assert len(payload['cells']) == 45
	assert len(payload['rows']) == 9
	assert len(payload['arms']) == 3
	assert all(r['cell_count'] == 5 for r in payload['rows'])
	assert payload['evaluation_split'] == ('validation' if survey == 'f3' else 'test')
	paths = multi_source_survey.summarize_survey(config)
	assert {p.name for p in paths} == {'comparison.csv', 'summary.json', 'summary.md'}
	assert len(list((root / 'output').iterdir())) == 3
	with pytest.raises(FileExistsError, match='already exists'):
		multi_source_survey.summarize_survey(config)


@pytest.mark.parametrize(
	'change', ['missing', 'duplicate', 'reuse_drift', 'audit_failure']
)
def test_survey_failure_creates_no_output(synthetic, monkeypatch, change):
	config, root = synthetic
	existing = receipts.read_json(Path(config['summaries']['f3']['path']))
	config = {
		**config,
		'schema_version': 1,
		'survey': 'f3',
		'artifact_root': str(root),
		'arms': {f: {} for f in receipts.SOURCE_FAMILIES},
	}

	def audit(_config, family, _receipt):
		if change == 'audit_failure':
			raise ValueError('source audit failure')
		cells = [
			copy.deepcopy(c) for c in existing['cells'] if c['source_family'] == family
		]
		if change == 'missing':
			cells.pop()
		elif change == 'duplicate':
			cells[-1] = cells[0]
		else:
			cells[0]['primary'] += 0.1
		return cells, {}

	monkeypatch.setattr(multi_source_survey, '_audited_arm', audit)
	with pytest.raises(ValueError, match=r'expected|drift|failure'):
		multi_source_survey.summarize_survey(config)
	assert not (root / 'output').exists()


@pytest.mark.parametrize(
	('section', 'field', 'value'),
	[
		('head', 'ks', [4, 6, 8]),
		('head', 'projection_dim', 64),
		('loss', 'distillation_weight', 0.1),
		('loss', 'consistency_weight', 0.1),
		('train', 'epochs', 24),
		('train', 'seed', 43),
		('train', 'samples_per_epoch', 9999),
		('train', 'amp', True),
		('train', 'lr', 0.001),
		('train', 'allow_overwrite_output', True),
		('student', 'unfreeze_top_blocks', 2),
	],
)
def test_runtime_fixed_training_drift(section, field, value):
	path = ROOT / (
		'experiments/f3/facies_benchmark_v2/'
		'128_hmm_v2_k6810_multi_source_v1/30_pretraining/'
		'random_init_hmm_v2_mh_k6810_distill020/02_full_25ep.yaml'
	)
	config = yaml.safe_load(path.read_text())
	receipts.validate_fixed_training_config(config)
	config[section][field] = value
	with pytest.raises(ValueError, match='fixed K6810 training contract drift'):
		receipts.validate_fixed_training_config(config)


@pytest.mark.parametrize('survey', ['parihaka', 'volve'])
def test_paired_identity_drift_rejected(survey):
	if survey == 'parihaka':
		candidate = {
			'benchmark_identity': {
				'model': 'candidate',
				'decoder': 'same',
				'test_definition': 'changed',
			}
		}
		control = {
			'benchmark_identity': {
				'model': 'control',
				'decoder': 'same',
				'test_definition': 'original',
			}
		}
	else:
		candidate = dict.fromkeys(
			(
				'shared_run_identity',
				'support_identity',
				'split_plan_sha256',
				'decoder_initial_state_sha256',
				'valid_tokens_sha256',
			),
			'same',
		)
		control = {**candidate, 'support_identity': 'changed'}
	with pytest.raises(ValueError, match='paired'):
		multi_source_survey._paired_identity(survey, candidate, control)  # noqa: SLF001


@pytest.mark.parametrize('survey', receipts.SURVEYS)
@pytest.mark.parametrize('drift', [None, 'checkpoint', 'metrics', 'extra_layout'])
def test_audited_arm_binds_paths_sources_and_cell_metrics(  # noqa: C901, PLR0915
	tmp_path,
	monkeypatch,
	survey,
	drift,
):
	family = 'random'
	candidate_id = receipts.CANDIDATE_IDS[family]
	control_id = 'matching_k6'
	checkpoint = tmp_path / 'candidate.pt'
	checkpoint.write_bytes(b'candidate')
	control_checkpoint = tmp_path / 'control.pt'
	control_checkpoint.write_bytes(b'control')
	cell_values = metric_cells(survey)
	receipt = {
		'survey': survey,
		'source_family': family,
		'condition_id': '6b',
		**receipts.METRICS[survey],
		'checkpoint_id': control_id,
		'checkpoint_path': 'control.pt',
		'checkpoint_sha256': receipts.file_sha256(control_checkpoint),
		'cells': cell_values,
		'canonical_cells_sha256': receipts.canonical_cells_sha256(survey, cell_values),
	}

	def raw(model_id, path):
		value = {'outputs': {'runs_root': str(tmp_path / 'runs')}}
		if survey == 'f3':
			value['candidate'] = {'id': model_id, 'checkpoint': str(path)}
		elif survey == 'parihaka':
			value['embeddings'] = {'models': {model_id: {'checkpoint': str(path)}}}
		else:
			value['arm'] = {'arm_id': model_id, 'checkpoint': str(path)}
		return value

	configs = {
		'candidate.yaml': raw(candidate_id, checkpoint),
		'control.yaml': raw(control_id, control_checkpoint),
	}
	training = ROOT / (
		'experiments/f3/facies_benchmark_v2/'
		'128_hmm_v2_k6810_multi_source_v1/30_pretraining/'
		'random_init_hmm_v2_mh_k6810_distill020/02_full_25ep.yaml'
	)
	config = {
		'survey': survey,
		'artifact_root': str(tmp_path),
		'arms': {
			family: {
				'candidate_id': candidate_id,
				'training_config': str(training),
				'downstream_config': 'candidate.yaml',
				'control_downstream_config': 'control.yaml',
			}
		},
	}
	for model in (candidate_id, control_id):
		for cell in cell_values:
			path = (
				tmp_path
				/ 'runs'
				/ f'model={model}'
				/ f'layout={cell["layout_id"]}'
				/ f'size={cell["data_size"]}'
			)
			if survey == 'f3':
				path /= 'evaluation'
				metrics = {'mean_iou': cell['primary'], 'macro_f1': cell['secondary']}
			elif survey == 'parihaka':
				metrics = {
					'test': {
						'channel_iou': cell['primary'],
						'channel_f1': cell['secondary'],
					}
				}
			else:
				metrics = {
					'test': {
						'primary_common': {
							'macro_mae_samples': cell['primary'],
							'macro_within_2_samples': cell['secondary'],
						}
					}
				}
			write_json(path / 'metrics.json', metrics)
	monkeypatch.setattr(multi_source_survey, 'load_config', lambda p: configs[str(p)])
	monkeypatch.setattr(
		multi_source_survey,
		'audit_multi_head_source',
		lambda _p: {
			'model_tag': candidate_id,
			'head_ks': [6, 8, 10],
			'epoch': 25,
			'checkpoint': {
				'path': str(checkpoint),
				'sha256': receipts.file_sha256(checkpoint),
			},
		},
	)
	audited = []

	def audit(actual_survey, _raw, model, layout, size):
		assert actual_survey == survey
		audited.append((model, layout, size))
		if drift == 'metrics' and len(audited) == 1:
			path = (
				tmp_path
				/ 'runs'
				/ f'model={model}'
				/ f'layout={layout}'
				/ f'size={size}'
			)
			path /= 'evaluation/metrics.json' if survey == 'f3' else 'metrics.json'
			path.write_text(path.read_text() + '\n')
		return {'audited': True}

	monkeypatch.setattr(multi_source_survey, '_audit_cell', audit)
	monkeypatch.setattr(multi_source_survey, '_paired_identity', lambda *_args: None)
	if drift == 'checkpoint':
		control_checkpoint.write_bytes(b'changed')
	elif drift == 'extra_layout':
		(tmp_path / 'runs' / f'model={candidate_id}' / 'layout=unexpected').mkdir()
	if drift:
		with pytest.raises(ValueError, match=r'drift|changed|unexpected'):
			multi_source_survey._audited_arm(config, family, receipt)  # noqa: SLF001
	else:
		cells, evidence = multi_source_survey._audited_arm(config, family, receipt)  # noqa: SLF001
		assert len(cells) == 15
		assert len(evidence['paired_cells']) == 15
		assert len(audited) == 30
		assert all(c['primary'] == c['control_primary'] for c in cells)
		json.dumps(evidence)


def test_every_summary_reference_resolves_with_its_survey_loader(monkeypatch, tmp_path):
	from seis_ssl_cluster.f3.lithology.candidate_benchmark import (  # noqa: PLC0415
		f3_lithology_candidate_config_from_mapping,
	)
	from seis_ssl_cluster.parihaka.channel_decoder import (  # noqa: PLC0415
		channel_decoder_config_from_mapping,
	)
	from seis_ssl_cluster.volve.horizon_recipe_arm import (  # noqa: PLC0415
		volve_horizon_recipe_arm_config_from_mapping,
	)

	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(ROOT))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts'))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_VOLVE_ROOT', str(tmp_path / 'volve'))
	controls = receipts.load_control_receipts(
		EXPERIMENT / 'hmm_v1_controls_receipt.json'
	)
	count = 0
	for path in ROOT.glob(
		'experiments/*/*/*hmm_v2_k6810_multi_source_v1/60_summary/01_paired_comparison.yaml'
	):
		config = multi_source_survey.load_config(path)
		survey = config['survey']
		for family, entry in config['arms'].items():
			control = receipts.control_receipt_for(controls, survey, family)
			for field, model_id in (
				('downstream_config', entry['candidate_id']),
				('control_downstream_config', control['checkpoint_id']),
			):
				raw = multi_source_survey.load_config(entry[field])
				multi_source_survey._paths(raw, survey, model_id)  # noqa: SLF001
				if survey == 'f3':
					f3_lithology_candidate_config_from_mapping(raw)
				elif survey == 'parihaka':
					channel_decoder_config_from_mapping(raw)
				elif 'arm' in raw:
					volve_horizon_recipe_arm_config_from_mapping(raw)
				else:
					multi_source_survey.volve_horizon_five_way_config_from_mapping(raw)
				count += 1
	assert count == 18
	assert not (tmp_path / 'artifacts').exists()
