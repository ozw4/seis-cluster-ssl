from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from proc.seis_ssl_cluster import run_f3_lithology_overlap_subcrop_poc as poc_cli
from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_embedding_extraction_config,
)
from seis_ssl_cluster.config.f3_lithology_five_way import (
	f3_lithology_five_way_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	f3_lithology_voxel_decoder_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology import five_way_runner
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	resolve_f3_lithology_five_way_job,
	run_f3_lithology_five_way_job,
)
from seis_ssl_cluster.training.voxel_decoder.runner import (
	_validate_source_provenance,
)
from tests.seis_ssl_cluster.helpers_f3_five_way import (
	build_five_way_universe,
	write_condition,
)

if TYPE_CHECKING:
	from types import ModuleType

ROOT = Path('experiments/f3/facies_benchmark_v1/112_local_bt_overlap_subcrop_poc_v1')
CANDIDATE_ID = 'shift04_proj384_pairs128_lambda005'
PRETRAINING = ROOT / '10_pretraining' / f'{CANDIDATE_ID}.yaml'
FEASIBILITY = ROOT / '10_pretraining/01_gpu_feasibility_1step.yaml'
EMBEDDING = ROOT / '20_embeddings' / f'{CANDIDATE_ID}.yaml'
RANDOM_DOWNSTREAM = ROOT / '30_downstream/random_medium.yaml'
CANDIDATE_DOWNSTREAM = ROOT / '30_downstream' / f'{CANDIDATE_ID}_medium.yaml'
SOURCE_PRETRAINING = Path(
	'experiments/f3/facies_benchmark_v1/22_local_barlow_twins_v1/'
	'02_full_100ep.yaml'
)
V3_FIVE_WAY = Path(
	'experiments/f3/facies_benchmark_v2/'
	'110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml'
)
V2_RANDOM_EMBEDDING = Path(
	'experiments/f3/facies_benchmark_v2/'
	'110_lithology_mae_local_bt_five_way_v2/50_embeddings/05_extract_random.yaml'
)
DECIDE = ROOT / 'decide.py'


@pytest.fixture(autouse=True)
def _config_environment(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT', '/workspace/artifacts/seis_ssl_cluster'
	)
	monkeypatch.setenv('F3_ROOT', '/home/dcuser/data/public_data/field/F3')
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', '/workspace')


def _resolved_training(path: Path) -> dict[str, object]:
	return resolve_barlow_twins_training_config(load_config(path))


def _resolved_embedding(path: Path) -> dict[str, object]:
	return resolve_embedding_extraction_config(load_config(path))


def _decision_module() -> ModuleType:
	spec = importlib.util.spec_from_file_location('f3_overlap_subcrop_decide', DECIDE)
	if spec is None or spec.loader is None:
		raise RuntimeError(f'could not load decision script: {DECIDE}')
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def _write_metrics(
	runs_root: Path,
	*,
	model_id: str,
	values: dict[str, float],
) -> None:
	for layout_id, macro_f1 in values.items():
		path = (
			runs_root
			/ f'model={model_id}'
			/ f'layout={layout_id}'
			/ 'size=medium/evaluation/metrics.json'
		)
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(
			json.dumps({
				'aggregation_unit': 'unique_validation_voxel',
				'evaluation_voxel_count': 123,
				'macro_f1': macro_f1,
				'mean_iou': macro_f1 - 0.1,
				'balanced_accuracy': macro_f1 + 0.1,
				'weighted_f1': macro_f1 + 0.2,
			}),
			encoding='utf-8',
		)


def _synthetic_poc_config(
	tmp_path: Path, *, candidate_id: str = 'candidate_a'
) -> tuple[dict[str, object], Path]:
	mapping = build_five_way_universe(tmp_path / 'synthetic')
	condition = write_condition(mapping, 'layout_001', 'medium')
	metadata_path = condition / 'section_layout_metadata.json'
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['active_lines'] = {
		'inline': [100, 101],
		'crossline': [200, 201],
	}
	metadata['identity']['per_line_contributions'] = {
		'inline:100': 128,
		'inline:101': 128,
		'crossline:200': 128,
		'crossline:201': 128,
	}
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
	root = Path(mapping['paths']['artifact_root'])
	mapping['outputs'] = {
		'runs_root': str(root / 'poc' / candidate_id / 'runs'),
		'summary_root': str(root / 'poc' / candidate_id / 'summary'),
	}
	config_path = tmp_path / f'{candidate_id}_medium.yaml'
	config_path.write_text(yaml.safe_dump(mapping), encoding='utf-8')
	return mapping, config_path


def _file_snapshot(root: Path) -> dict[str, bytes]:
	return {
		str(path.relative_to(root)): path.read_bytes()
		for path in sorted(root.rglob('*'))
		if path.is_file()
	}


def test_initial_training_config_is_exact_10_epoch_overlap_delta() -> None:
	source = _resolved_training(SOURCE_PRETRAINING)
	candidate = _resolved_training(PRETRAINING)
	expected = deepcopy(source)
	expected['paths']['output_root'] = candidate['paths']['output_root']
	expected['train']['epochs'] = 10
	expected['augmentations'] = {
		'policy': 'overlapping_subcrop_xy_v1',
		'horizontal_flip_probability': 0.5,
		'max_subcrop_shift_tokens': [4, 4, 0],
	}

	assert candidate == expected
	assert 'continuation' not in candidate
	assert (
		candidate['train']['epochs']
		* candidate['train']['samples_per_epoch']
		// candidate['train']['batch_size']
		== 6_250
	)


def test_feasibility_config_resolves_to_one_fresh_step() -> None:
	config = _resolved_training(FEASIBILITY)

	assert config['augmentations']['policy'] == 'overlapping_subcrop_xy_v1'
	assert config['train']['max_steps'] == 1
	assert config['train']['epochs'] == 1
	assert 'continuation' not in config


def test_candidate_embedding_matches_v2_random_extraction_contract() -> None:
	candidate = _resolved_embedding(EMBEDDING)
	random = _resolved_embedding(V2_RANDOM_EMBEDDING)

	assert candidate['manifests'] == random['manifests']
	assert candidate['embedding'] == random['embedding']
	output_dir = Path(candidate['embeddings']['output_dir'])
	assert output_dir.is_relative_to(
		Path('/workspace/artifacts/seis_ssl_cluster/embeddings/f3/facies_benchmark_v2')
	)
	assert output_dir.parts[-2:] == ('local_barlow_twins', 'overlap_x64')
	assert CANDIDATE_ID in output_dir.parts


def test_downstream_reuses_v3_layout_and_only_replaces_candidate_slot() -> None:
	random = f3_lithology_five_way_config_from_mapping(load_config(RANDOM_DOWNSTREAM))
	candidate = f3_lithology_five_way_config_from_mapping(
		load_config(CANDIDATE_DOWNSTREAM)
	)
	v3 = f3_lithology_five_way_config_from_mapping(load_config(V3_FIVE_WAY))

	assert random.dataset == candidate.dataset == v3.dataset
	assert random.labels == candidate.labels == v3.labels
	assert (
		random.section_layout_dataset_root
		== candidate.section_layout_dataset_root
		== v3.section_layout_dataset_root
	)
	assert random.models == candidate.models
	for model_id in ('mae', 'mae_hmm_k6', 'local_barlow_twins_hmm_k6', 'random'):
		assert candidate.model_by_id(model_id) == v3.model_by_id(model_id)
	local = candidate.model_by_id('local_barlow_twins')
	assert CANDIDATE_ID in local.checkpoint.parts
	assert CANDIDATE_ID in local.embeddings_dir.parts
	assert local.embeddings_dir.parts[-2:] == (
		'local_barlow_twins',
		'overlap_x64',
	)


def test_candidate_embedding_path_satisfies_decoder_source_provenance() -> None:
	config = f3_lithology_five_way_config_from_mapping(
		load_config(CANDIDATE_DOWNSTREAM)
	)
	job = resolve_f3_lithology_five_way_job(
		config,
		model='local_barlow_twins',
		layout='layout_001',
		size='medium',
	)
	decoder = f3_lithology_voxel_decoder_config_from_mapping(
		five_way_runner._decoder_config_mapping(job)  # noqa: SLF001
	)

	_validate_source_provenance(
		decoder,
		embedding_payload={'checkpoint_path': str(job.model.checkpoint)},
	)


def test_poc_cli_has_only_two_filename_selected_models_and_medium_size() -> None:
	options = {
		option
		for action in poc_cli.build_parser()._actions  # noqa: SLF001
		for option in action.option_strings
	}
	assert '--model' not in options
	assert '--size' not in options
	assert poc_cli.poc_model_and_namespace(Path('random_medium.yaml')) == (
		'random',
		'random',
	)
	assert poc_cli.poc_model_and_namespace(
		Path(f'{CANDIDATE_ID}_medium.yaml')
	) == ('local_barlow_twins', CANDIDATE_ID)
	with pytest.raises(ValueError, match='canonical five-way model IDs'):
		poc_cli.poc_model_and_namespace(Path('mae_medium.yaml'))
	with pytest.raises(ValueError, match='config filename'):
		poc_cli.poc_model_and_namespace(Path('candidate.yaml'))


def test_poc_cli_passes_audit_false_to_existing_runner(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_mapping, config_path = _synthetic_poc_config(tmp_path)
	called: dict[str, object] = {}

	def fake_run(job: object, **kwargs: object) -> dict[str, object]:
		called['job'] = job
		called.update(kwargs)
		return {'completed': True}

	monkeypatch.setattr(poc_cli, 'run_f3_lithology_five_way_job', fake_run)
	monkeypatch.setattr(
		sys,
		'argv',
		[
			'run_f3_lithology_overlap_subcrop_poc.py',
			'--config',
			str(config_path),
			'--layout',
			'layout_001',
			'--device',
			'cpu',
			'--max-steps',
			'1',
		],
	)

	poc_cli.main()

	job = called['job']
	assert job.model.model_id == 'local_barlow_twins'
	assert job.data_size == 'medium'
	assert called['audit_sources'] is False
	assert called['device'] == 'cpu'
	assert called['max_steps'] == 1


def test_existing_runner_audits_by_default_and_can_explicitly_skip(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	mapping = build_five_way_universe(tmp_path / 'synthetic')
	config = f3_lithology_five_way_config_from_mapping(mapping)
	job = resolve_f3_lithology_five_way_job(
		config, model='random', layout='layout_001', size='medium'
	)
	audits: list[object] = []

	def record_audit(value: object) -> None:
		audits.append(value)

	def fake_frozen(*_args: object, **_kwargs: object) -> dict[str, object]:
		return {'completed': True}

	monkeypatch.setattr(
		five_way_runner,
		'audit_f3_lithology_five_way_sources',
		record_audit,
	)
	monkeypatch.setattr(
		five_way_runner,
		'run_f3_lithology_frozen_encoder_job',
		fake_frozen,
	)

	run_f3_lithology_five_way_job(job)
	assert audits == [config]
	run_f3_lithology_five_way_job(job, audit_sources=False)
	assert audits == [config]


def test_poc_dry_run_writes_no_artifacts(
	tmp_path: Path,
) -> None:
	mapping, config_path = _synthetic_poc_config(tmp_path)
	root = Path(mapping['paths']['artifact_root'])
	before = _file_snapshot(root)

	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			'proc/seis_ssl_cluster/run_f3_lithology_overlap_subcrop_poc.py',
			'--config',
			str(config_path),
			'--layout',
			'layout_001',
			'--dry-run',
		],
		check=True,
		capture_output=True,
		text=True,
	)

	assert 'model_id: local_barlow_twins' in result.stdout
	assert 'data_size: medium' in result.stdout
	assert 'execution: dry-run; no files written' in result.stdout
	assert _file_snapshot(root) == before


@pytest.mark.parametrize(
	('candidate_value', 'expected_passed'),
	[(0.51, 1), (0.50, 0), (0.49, 0)],
)
def test_screen_is_strict_and_tie_is_loss(
	tmp_path: Path,
	candidate_value: float,
	expected_passed: int,
) -> None:
	module = _decision_module()
	random_root = tmp_path / 'random/runs'
	candidate_root = tmp_path / 'candidate/runs'
	_write_metrics(
		random_root, model_id='random', values={'layout_001': 0.50}
	)
	_write_metrics(
		candidate_root,
		model_id='local_barlow_twins',
		values={'layout_001': candidate_value},
	)

	result = module.decide(
		candidate_id=CANDIDATE_ID,
		random_runs_root=random_root,
		candidate_runs_root=candidate_root,
		mode='screen',
	)

	assert result['screen_passed'] is bool(expected_passed)
	assert result['wins'] == expected_passed
	assert result['losses_or_ties'] == 1 - expected_passed
	assert result['adopted'] is False
	assert result['layouts'][0]['candidate']['mean_iou'] == pytest.approx(
		candidate_value - 0.1
	)


@pytest.mark.parametrize(('wins', 'expected_adopted'), [(3, 0), (4, 1)])
def test_final_requires_four_of_five_strict_wins(
	tmp_path: Path,
	wins: int,
	expected_adopted: int,
) -> None:
	module = _decision_module()
	layout_ids = [f'layout_{index:03d}' for index in range(5)]
	random_root = tmp_path / 'random/runs'
	candidate_root = tmp_path / 'candidate/runs'
	_write_metrics(
		random_root,
		model_id='random',
		values=dict.fromkeys(layout_ids, 0.50),
	)
	_write_metrics(
		candidate_root,
		model_id='local_barlow_twins',
		values={
			layout_id: 0.51 if index < wins else 0.50
			for index, layout_id in enumerate(layout_ids)
		},
	)

	result = module.decide(
		candidate_id=CANDIDATE_ID,
		random_runs_root=random_root,
		candidate_runs_root=candidate_root,
		mode='final',
	)

	assert result['wins'] == wins
	assert result['losses_or_ties'] == 5 - wins
	assert result['adopted'] is bool(expected_adopted)
	assert len(result['layouts']) == 5


def test_failing_decision_writes_json_before_exit(
	tmp_path: Path,
) -> None:
	random_root = tmp_path / 'random/runs'
	candidate_root = tmp_path / CANDIDATE_ID / 'runs'
	_write_metrics(
		random_root, model_id='random', values={'layout_001': 0.50}
	)
	_write_metrics(
		candidate_root,
		model_id='local_barlow_twins',
		values={'layout_001': 0.50},
	)

	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			str(DECIDE),
			'--candidate-id',
			CANDIDATE_ID,
			'--mode',
			'screen',
			'--random-runs-root',
			str(random_root),
			'--candidate-runs-root',
			str(candidate_root),
		],
		check=False,
		capture_output=True,
		text=True,
	)

	decision_path = candidate_root.parent / 'screen_decision.json'
	assert result.returncode == 1
	assert decision_path.is_file()
	payload = json.loads(decision_path.read_text(encoding='utf-8'))
	assert payload['screen_passed'] is False
	assert payload['losses_or_ties'] == 1
	assert payload['layouts'][0]['paired_delta'] == 0.0
