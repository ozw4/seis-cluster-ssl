from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from seis_ssl_cluster.config import (
	load_config,
	resolve_embedding_extraction_config,
)
from seis_ssl_cluster.config.f3_lithology_five_way import (
	FIVE_WAY_MODEL_IDS,
	f3_lithology_five_way_config_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.five_way_sources import (
	audit_f3_lithology_five_way_sources,
	plan_f3_lithology_five_way_sources,
)
from tests.seis_ssl_cluster.helpers_f3_five_way import (
	SURVEY_ID,
	TOKEN_GRID,
	build_five_way_universe,
	local_bt_objective,
	local_bt_stage1_config,
	mae_stage1_config,
)

FIVE_WAY_ROOT = Path(
	'experiments/f3/facies_benchmark_v1/110_lithology_mae_local_bt_five_way_v1'
)
FIVE_WAY_CONFIG = FIVE_WAY_ROOT / '60_five_way.yaml'
EXTRACTION_CONFIGS = {
	'mae': '01_extract_mae.yaml',
	'mae_hmm_k6': '02_extract_mae_hmm_k6.yaml',
	'local_barlow_twins': '03_extract_local_barlow_twins.yaml',
	'local_barlow_twins_hmm_k6': '04_extract_local_barlow_twins_hmm_k6.yaml',
	'random': '05_extract_random.yaml',
}


@pytest.fixture
def env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3_root'))
	return root


@pytest.fixture
def universe(tmp_path: Path) -> dict[str, object]:
	return build_five_way_universe(tmp_path / 'synthetic')


def _files_snapshot(root: Path) -> dict[str, str]:
	return {
		str(path): file_sha256(path)
		for path in sorted(root.rglob('*'))
		if path.is_file()
	}


def test_experiment_config_resolves_with_fixed_order_and_extraction_parity(
	env_root: Path,
) -> None:
	config = f3_lithology_five_way_config_from_mapping(load_config(FIVE_WAY_CONFIG))

	assert config.model_ids == FIVE_WAY_MODEL_IDS
	assert config.runs_root == (
		env_root / 'f3_lithology_benchmark/mae_local_bt_five_way_v1/runs'
	)
	assert config.summary_root == (
		env_root / 'f3_lithology_benchmark/mae_local_bt_five_way_v1/summary'
	)
	for model in config.models:
		extraction = resolve_embedding_extraction_config(
			load_config(
				FIVE_WAY_ROOT
				/ '50_embeddings'
				/ EXTRACTION_CONFIGS[model.model_id]
			)
		)
		assert extraction['embeddings']['checkpoint'] == str(model.checkpoint)
		assert extraction['embeddings']['output_dir'] == str(model.embeddings_dir)
		assert extraction['embedding']['window_size'] == [128, 128, 128]
		assert extraction['embedding']['overlap'] == [64, 64, 64]
		assert extraction['embedding']['output_dtype'] == 'float16'
		assert extraction['embedding']['batch_size'] == 1
		assert extraction['embedding']['amp'] is False
		assert extraction['embedding']['min_token_valid_fraction'] == 0.5


def test_resolver_rejects_wrong_order_unknown_ids_and_duplicates(
	universe: dict[str, object],
) -> None:
	reordered = deepcopy(universe)
	models = reordered['models']
	models[0], models[1] = models[1], models[0]
	with pytest.raises(ValueError, match='in this order'):
		f3_lithology_five_way_config_from_mapping(reordered)

	unknown = deepcopy(universe)
	unknown['models'][0]['model_id'] = 'mae_v2'
	with pytest.raises(ValueError, match='model_id must be one of'):
		f3_lithology_five_way_config_from_mapping(unknown)

	duplicated = deepcopy(universe)
	duplicated['models'][1]['embeddings_dir'] = duplicated['models'][0][
		'embeddings_dir'
	]
	with pytest.raises(ValueError, match='must be distinct'):
		f3_lithology_five_way_config_from_mapping(duplicated)

	trace_drop = deepcopy(universe)
	trace_drop['models'][2]['checkpoint'] = str(
		Path(universe['paths']['artifact_root'])
		/ 'pretraining/local_bt_d4_trace_drop/latest.pt'
	)
	with pytest.raises(ValueError, match='trace-drop'):
		f3_lithology_five_way_config_from_mapping(trace_drop)

	drifted = deepcopy(universe)
	drifted['models'][4]['expected'] = {
		'objective': 'random_encoder',
		'random_seed': 43,
		'stratigraphy_pretext': False,
	}
	with pytest.raises(ValueError, match='must restate the fixed'):
		f3_lithology_five_way_config_from_mapping(drifted)


def test_audit_passes_and_does_not_modify_sources(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	root = Path(universe['paths']['artifact_root'])
	before = _files_snapshot(root)

	report = audit_f3_lithology_five_way_sources(config)

	assert report['model_order'] == list(FIVE_WAY_MODEL_IDS)
	assert [source['model_id'] for source in report['sources']] == list(
		FIVE_WAY_MODEL_IDS
	)
	for source in report['sources']:
		assert source['valid_token_mask'] == 'byte_identical_across_models'  # noqa: S105
	assert _files_snapshot(root) == before


def test_plan_is_static_and_creates_nothing(universe: dict[str, object]) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	root = Path(universe['paths']['artifact_root'])
	missing = deepcopy(universe)
	missing['models'][0]['embeddings_dir'] = str(root / 'embeddings/missing')
	missing_config = f3_lithology_five_way_config_from_mapping(missing)
	before = _files_snapshot(root)

	rows = plan_f3_lithology_five_way_sources(missing_config)

	assert [row['model_id'] for row in rows] == list(config.model_ids)
	assert rows[0]['embeddings'].endswith(f'{SURVEY_ID}.embeddings.npy')
	assert _files_snapshot(root) == before


def test_audit_rejects_embedding_shape_and_dtype_mismatch(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	embeddings_path = (
		Path(universe['models'][1]['embeddings_dir'])
		/ f'{SURVEY_ID}.embeddings.npy'
	)
	np.save(
		embeddings_path,
		np.zeros((*TOKEN_GRID, 32), dtype=np.float16),
		allow_pickle=False,
	)
	with pytest.raises(ValueError, match='embedding array shape'):
		audit_f3_lithology_five_way_sources(config)

	np.save(
		embeddings_path,
		np.zeros((*TOKEN_GRID, 384), dtype=np.float32),
		allow_pickle=False,
	)
	with pytest.raises(ValueError, match='dtype must be float16'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_valid_token_mask_mismatch(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	mask_path = (
		Path(universe['models'][3]['embeddings_dir'])
		/ f'{SURVEY_ID}.valid_tokens.npy'
	)
	mask = np.ones(TOKEN_GRID, dtype=np.bool_)
	mask[0, 0, 0] = False
	np.save(mask_path, mask, allow_pickle=False)

	with pytest.raises(ValueError, match='not byte-identical'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_checkpoint_path_and_sha_mismatch(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	metadata_path = (
		Path(universe['models'][0]['embeddings_dir'])
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	payload = json.loads(metadata_path.read_text(encoding='utf-8'))
	payload['checkpoint_sha256'] = '0' * 64
	metadata_path.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(ValueError, match='checkpoint_sha256 does not match'):
		audit_f3_lithology_five_way_sources(config)

	payload['checkpoint_sha256'] = file_sha256(
		Path(universe['models'][1]['checkpoint'])
	)
	payload['checkpoint_path'] = universe['models'][1]['checkpoint']
	metadata_path.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(ValueError, match='checkpoint_path does not match'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_objective_identity_swaps(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	mae_metadata_path = (
		Path(universe['models'][0]['embeddings_dir'])
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	payload = json.loads(mae_metadata_path.read_text(encoding='utf-8'))
	payload['pretraining_method'] = 'local_barlow_twins_3d'
	payload['pretraining_objective'] = local_bt_objective()
	mae_metadata_path.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(ValueError, match='must not declare a Barlow Twins'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_missing_or_foreign_pretext_identity(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	hmm_metadata_path = (
		Path(universe['models'][1]['embeddings_dir'])
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	payload = json.loads(hmm_metadata_path.read_text(encoding='utf-8'))
	removed = deepcopy(payload)
	del removed['stratigraphy_pretext']
	hmm_metadata_path.write_text(json.dumps(removed), encoding='utf-8')
	with pytest.raises(ValueError, match='stratigraphy_pretext is required'):
		audit_f3_lithology_five_way_sources(config)

	swapped = deepcopy(payload)
	swapped['stratigraphy_pretext']['base_objective'] = 'local_barlow_twins_3d'
	hmm_metadata_path.write_text(json.dumps(swapped), encoding='utf-8')
	with pytest.raises(ValueError, match='base_objective'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_trace_drop_local_bt_sources(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	checkpoint = Path(universe['models'][2]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['config']['augmentations']['trace_drop_probability'] = 0.5
	torch.save(payload, checkpoint)
	sha256 = file_sha256(checkpoint)
	for index in (2,):
		metadata_path = (
			Path(universe['models'][index]['embeddings_dir'])
			/ f'{SURVEY_ID}.embedding_metadata.json'
		)
		metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
		metadata['checkpoint_sha256'] = sha256
		metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

	with pytest.raises(ValueError, match='trace-drop'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_pretext_pseudo_targets_from_trace_drop(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	metadata_path = (
		Path(universe['models'][3]['embeddings_dir'])
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	payload = json.loads(metadata_path.read_text(encoding='utf-8'))
	payload['stratigraphy_pretext']['pseudo_target_input_dir'] = (
		'/artifacts/pseudo_targets/local_bt_d4_trace_drop/local_bt100'
	)
	metadata_path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='trace-drop'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_invalid_random_metadata(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	checkpoint = Path(universe['models'][4]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['metadata']['seed'] = 43
	torch.save(payload, checkpoint)
	metadata_path = (
		Path(universe['models'][4]['embeddings_dir'])
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['checkpoint_sha256'] = file_sha256(checkpoint)
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

	with pytest.raises(ValueError, match=r'metadata\.seed must equal 42'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_random_reference_that_is_not_the_mae_checkpoint(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	checkpoint = Path(universe['models'][4]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['metadata']['reference_checkpoint'] = universe['models'][1][
		'checkpoint'
	]
	torch.save(payload, checkpoint)
	metadata_path = (
		Path(universe['models'][4]['embeddings_dir'])
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['checkpoint_sha256'] = file_sha256(checkpoint)
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

	with pytest.raises(ValueError, match='reference_checkpoint must equal'):
		audit_f3_lithology_five_way_sources(config)


def _repoint_checkpoint(universe: dict[str, object], index: int, payload) -> None:
	checkpoint = Path(universe['models'][index]['checkpoint'])
	torch.save(payload, checkpoint)
	metadata_path = (
		Path(universe['models'][index]['embeddings_dir'])
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['checkpoint_sha256'] = file_sha256(checkpoint)
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')


def test_audit_rejects_a_stage1_checkpoint_in_the_mae_slot(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	_repoint_checkpoint(
		universe,
		0,
		{
			'config': mae_stage1_config(),
			'epoch': 100,
			'global_step': 62_500,
		},
	)

	with pytest.raises(ValueError, match='fixed budget'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_an_interrupted_fixed_budget_checkpoint(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	checkpoint = Path(universe['models'][0]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['epoch'] = 12
	payload['global_step'] = 7_500
	_repoint_checkpoint(universe, 0, payload)

	with pytest.raises(ValueError, match=r'epoch must equal the fixed budget 25'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_a_missing_continuation_lineage(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	checkpoint = Path(universe['models'][2]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	del payload['config']['continuation']
	_repoint_checkpoint(universe, 2, payload)

	with pytest.raises(ValueError, match='fixed-budget continuation'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_a_trace_drop_base_in_the_local_bt_lineage(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	checkpoint = Path(universe['models'][2]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['config']['continuation']['init_checkpoint'] = (
		'/artifacts/pretraining/local_bt100/bt_continue_d4_trace_drop/latest.pt'
	)
	_repoint_checkpoint(universe, 2, payload)

	with pytest.raises(ValueError, match='trace-drop artifact'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_walks_the_local_bt_lineage_into_the_base_checkpoint(
	universe: dict[str, object],
	tmp_path: Path,
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	base = tmp_path / 'stage1_base' / 'latest.pt'
	_write_stage1_base(
		base,
		epochs=100,
		augmentations={
			'policy': 'd4_trace_drop',
			'reflection_probability': 0.5,
			'trace_drop_probability': 0.1,
		},
	)
	checkpoint = Path(universe['models'][2]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['config']['continuation']['init_checkpoint'] = str(base)
	_repoint_checkpoint(universe, 2, payload)

	with pytest.raises(ValueError, match='trace-drop augmentations'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_an_hmm_checkpoint_without_its_fixed_budget(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	checkpoint = Path(universe['models'][1]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['stratigraphy_config']['train']['epochs'] = 5
	_repoint_checkpoint(universe, 1, payload)

	with pytest.raises(
		ValueError, match=r'stratigraphy_config\.train\.epochs must equal'
	):
		audit_f3_lithology_five_way_sources(config)


CONTRACT_DRIFTS = (
	(2, ('config', 'train', 'batch_size'), 8, r'train\.batch_size must equal 16'),
	(
		2,
		('config', 'train', 'samples_per_epoch'),
		5_000,
		r'train\.samples_per_epoch must equal 10000',
	),
	(2, ('config', 'train', 'lr'), 1.0e-4, r'train\.lr must equal 1e-05'),
	(
		2,
		('config', 'barlow_twins', 'projector_dim'),
		256,
		r'barlow_twins\.projector_dim must equal 384',
	),
	(
		3,
		('stratigraphy_config', 'head', 'temperature'),
		0.5,
		r'head\.temperature must equal 0\.1',
	),
	(
		3,
		('stratigraphy_config', 'loss', 'usage_weight'),
		0.05,
		r'loss\.usage_weight must equal 0\.005',
	),
)


@pytest.mark.parametrize(('index', 'keys', 'value', 'message'), CONTRACT_DRIFTS)
def test_audit_rejects_fixed_training_contract_drift(
	universe: dict[str, object],
	index: int,
	keys: tuple[str, ...],
	value: object,
	message: str,
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	checkpoint = Path(universe['models'][index]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	block = payload
	for key in keys[:-1]:
		block = block[key]
	block[keys[-1]] = value
	_repoint_checkpoint(universe, index, payload)

	with pytest.raises(ValueError, match=message):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_a_stage1_base_with_the_same_steps_but_fewer_samples(
	universe: dict[str, object],
	tmp_path: Path,
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	base = tmp_path / 'stage1_half_samples' / 'latest.pt'
	base.parent.mkdir(parents=True, exist_ok=True)
	base_config = local_bt_stage1_config()
	base_config['train']['batch_size'] = 8
	base_config['train']['samples_per_epoch'] = 5_000
	torch.save(
		{'config': base_config, 'epoch': 100, 'global_step': 62_500}, base
	)
	checkpoint = Path(universe['models'][2]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['config']['continuation']['init_checkpoint'] = str(base)
	_repoint_checkpoint(universe, 2, payload)

	with pytest.raises(
		ValueError,
		match=r'continuation\.init_checkpoint train\.batch_size must equal 16',
	):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_a_random_checkpoint_with_training_steps(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	checkpoint = Path(universe['models'][4]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['epoch'] = 25
	payload['global_step'] = 15_625
	_repoint_checkpoint(universe, 4, payload)

	with pytest.raises(ValueError, match='random checkpoint epoch must equal 0'):
		audit_f3_lithology_five_way_sources(config)


def _write_stage1_base(path: Path, *, epochs: int, augmentations: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	config = local_bt_stage1_config()
	config['augmentations'] = augmentations
	config['train']['epochs'] = epochs
	torch.save(
		{
			'config': config,
			'epoch': epochs,
			'global_step': epochs * 625,
		},
		path,
	)


def test_audit_rejects_a_short_stage1_base(
	universe: dict[str, object],
	tmp_path: Path,
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	base = tmp_path / 'stage1_short' / 'latest.pt'
	_write_stage1_base(
		base, epochs=50, augmentations={'horizontal_flip_probability': 0.5}
	)
	checkpoint = Path(universe['models'][2]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['config']['continuation']['init_checkpoint'] = str(base)
	_repoint_checkpoint(universe, 2, payload)

	with pytest.raises(ValueError, match='must be the 100 epoch stage-1 source'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_accepts_a_full_stage1_base(
	universe: dict[str, object],
	tmp_path: Path,
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	base = tmp_path / 'stage1_full' / 'latest.pt'
	_write_stage1_base(
		base, epochs=100, augmentations={'horizontal_flip_probability': 0.5}
	)
	checkpoint = Path(universe['models'][2]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['config']['continuation']['init_checkpoint'] = str(base)
	_repoint_checkpoint(universe, 2, payload)

	report = audit_f3_lithology_five_way_sources(config)

	assert report['model_order'] == list(FIVE_WAY_MODEL_IDS)
	assert len(report['sources']) == len(FIVE_WAY_MODEL_IDS)


def test_audit_rejects_a_trace_drop_student_init_for_hmm_arms(
	universe: dict[str, object],
	tmp_path: Path,
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	base = tmp_path / 'bt_continue_d4_trace_drop' / 'latest.pt'
	_write_stage1_base(
		base,
		epochs=100,
		augmentations={
			'policy': 'xy_d4_trace_drop_v1',
			'reflection_probability': 0.5,
			'trace_drop_probability': 0.02,
		},
	)
	checkpoint = Path(universe['models'][3]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['stratigraphy_config']['student']['init_checkpoint'] = str(base)
	payload['stratigraphy_config']['teacher']['checkpoint'] = str(base)
	_repoint_checkpoint(universe, 3, payload)

	with pytest.raises(ValueError, match='trace-drop'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_teacher_and_student_from_different_sources(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	checkpoint = Path(universe['models'][1]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['stratigraphy_config']['teacher'] = {
		'checkpoint': '/stage1/other/full_100ep/latest.pt'
	}
	_repoint_checkpoint(universe, 1, payload)

	with pytest.raises(ValueError, match='same stage-1 source'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_reports_an_unreadable_ancestor_with_model_context(
	universe: dict[str, object],
	tmp_path: Path,
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	base = tmp_path / 'corrupt' / 'latest.pt'
	base.parent.mkdir(parents=True, exist_ok=True)
	base.write_bytes(b'PK\x03\x04truncated-partial-copy')
	checkpoint = Path(universe['models'][2]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['config']['continuation']['init_checkpoint'] = str(base)
	_repoint_checkpoint(universe, 2, payload)

	with pytest.raises(ValueError, match='is unreadable'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_missing_stage1_ancestor(
	universe: dict[str, object],
	tmp_path: Path,
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	base = tmp_path / 'moved_away' / 'full_100ep' / 'latest.pt'
	checkpoint = Path(universe['models'][2]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['config']['continuation']['init_checkpoint'] = str(base)
	_repoint_checkpoint(universe, 2, payload)

	with pytest.raises(FileNotFoundError, match='does not exist') as excinfo:
		audit_f3_lithology_five_way_sources(config)
	message = str(excinfo.value)
	assert 'local_barlow_twins' in message
	assert 'continuation.init_checkpoint' in message
	assert str(base) in message


def test_audit_rejects_a_lineage_deeper_than_it_can_verify(
	universe: dict[str, object],
	tmp_path: Path,
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	chain_root = tmp_path / 'chain'
	chain_root.mkdir(parents=True, exist_ok=True)
	previous: Path | None = None
	for index in range(6):
		path = chain_root / f'link_{index}' / 'latest.pt'
		path.parent.mkdir(parents=True, exist_ok=True)
		config_payload = local_bt_stage1_config()
		if previous is not None:
			config_payload['continuation'] = {
				'init_checkpoint': str(previous),
				'unfreeze_top_blocks': 1,
			}
		torch.save({'config': config_payload, 'epoch': 100}, path)
		previous = path
	checkpoint = Path(universe['models'][2]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['config']['continuation']['init_checkpoint'] = str(previous)
	_repoint_checkpoint(universe, 2, payload)

	with pytest.raises(ValueError, match='deeper than the audit can verify'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_a_local_bt_arm_continued_from_a_mae_base(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	root = Path(universe['paths']['artifact_root'])
	mae_stage1 = root / 'pretraining/stage1/mae/full_100ep/latest.pt'
	assert mae_stage1.is_file()
	checkpoint = Path(universe['models'][2]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['config']['continuation']['init_checkpoint'] = str(mae_stage1)
	_repoint_checkpoint(universe, 2, payload)

	with pytest.raises(
		ValueError, match='stage must be barlow_twins_training'
	):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_an_hmm_arm_distilled_from_the_wrong_objective(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	root = Path(universe['paths']['artifact_root'])
	mae_stage1 = root / 'pretraining/stage1/mae/full_100ep/latest.pt'
	checkpoint = Path(universe['models'][3]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['stratigraphy_config']['student']['init_checkpoint'] = str(mae_stage1)
	payload['stratigraphy_config']['teacher']['checkpoint'] = str(mae_stage1)
	_repoint_checkpoint(universe, 3, payload)

	with pytest.raises(
		ValueError, match='stage must be barlow_twins_training'
	):
		audit_f3_lithology_five_way_sources(config)


def test_audit_requires_a_recorded_teacher_for_hmm_arms(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	checkpoint = Path(universe['models'][1]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	del payload['stratigraphy_config']['teacher']
	_repoint_checkpoint(universe, 1, payload)

	with pytest.raises(ValueError, match=r'must record stratigraphy_config\.teacher'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_confidence_gated_pseudo_targets(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	checkpoint = Path(universe['models'][3]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['stratigraphy_config']['pseudo_targets']['min_confidence'] = 0.95
	_repoint_checkpoint(universe, 3, payload)

	with pytest.raises(ValueError, match='min_confidence must equal'):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_a_random_checkpoint_from_a_foreign_producer(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	checkpoint = Path(universe['models'][4]['checkpoint'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	payload['training_state']['stage'] = 'train_amp_mae'
	_repoint_checkpoint(universe, 4, payload)

	with pytest.raises(
		ValueError, match=r"stage must equal 'create_random_mae_checkpoint'"
	):
		audit_f3_lithology_five_way_sources(config)


def test_audit_rejects_preprocessing_cache_drift(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	metadata_path = (
		Path(universe['models'][2]['embeddings_dir'])
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	payload = json.loads(metadata_path.read_text(encoding='utf-8'))
	payload['preprocessing_cache']['effective_mode'] = 'chunked'
	metadata_path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='preprocessing_cache differs'):
		audit_f3_lithology_five_way_sources(config)
