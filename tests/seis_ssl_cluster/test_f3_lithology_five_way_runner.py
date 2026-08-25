from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster.config.f3_lithology_five_way import (
	FIVE_WAY_MODEL_IDS,
	f3_lithology_five_way_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	f3_lithology_voxel_decoder_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	FIXED_DECODER_CONTRACT,
	LAYOUT_IDS,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology import five_way_runner
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	inspect_f3_lithology_five_way_job,
	plan_f3_lithology_five_way_jobs,
	resolve_f3_lithology_five_way_job,
	run_f3_lithology_five_way_job,
)
from tests.seis_ssl_cluster.helpers_f3_five_way import (
	SURVEY_ID,
	build_five_way_universe,
	write_condition,
)

SMOKE_DECODER_CONTRACT = {
	'spec': FIXED_DECODER_CONTRACT['spec'],
	'embedding_dim': 384,
	'class_count': 6,
	'hidden_channels': (8, 4, 2),
	'upsample_factors': ((2, 2, 2), (2, 2, 2), (2, 2, 2)),
	'upsample_mode': FIXED_DECODER_CONTRACT['upsample_mode'],
	'normalization': FIXED_DECODER_CONTRACT['normalization'],
	'epochs': 1,
	'batch_size': 1,
	'learning_rate': 0.001,
	'weight_decay': 0.0001,
	'class_weight': 'balanced',
	'sampling_mode': 'uniform_tiles_with_replacement',
	'steps_per_epoch': 2,
	'amp': False,
	'gradient_clip_norm': 1.0,
	'write_probabilities': False,
	'seed': 42000,
}
SMOKE_TILE_SETTINGS = {
	'core_size_tokens': (2, 2, 2),
	'context_halo_tokens': (0, 0, 0),
}


@pytest.fixture
def universe(tmp_path: Path) -> dict[str, object]:
	return build_five_way_universe(tmp_path / 'synthetic')


def _files_snapshot(root: Path) -> dict[str, str]:
	return {
		str(path): file_sha256(path)
		for path in sorted(root.rglob('*'))
		if path.is_file()
	}


def _smoke(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(
		five_way_runner, 'FIXED_DECODER_CONTRACT', SMOKE_DECODER_CONTRACT
	)
	monkeypatch.setattr(
		five_way_runner, 'FIVE_WAY_TILE_SETTINGS', SMOKE_TILE_SETTINGS
	)


def test_plan_enumerates_75_unique_jobs(universe: dict[str, object]) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	jobs = plan_f3_lithology_five_way_jobs(config)

	assert len(jobs) == 75
	assert len(set(jobs)) == 75
	assert {job[0] for job in jobs} == set(FIVE_WAY_MODEL_IDS)
	assert {job[1] for job in jobs} == set(LAYOUT_IDS)
	assert {job[2] for job in jobs} == set(DATA_SIZES)


def test_resolver_rejects_invalid_model_layout_size(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	with pytest.raises(ValueError, match='unknown five-way model'):
		resolve_f3_lithology_five_way_job(
			config, model='bert', layout='layout_000', size='small'
		)
	with pytest.raises(ValueError, match='unknown layout'):
		resolve_f3_lithology_five_way_job(
			config, model='mae', layout='layout_005', size='small'
		)
	with pytest.raises(ValueError, match='unknown data size'):
		resolve_f3_lithology_five_way_job(
			config, model='mae', layout='layout_000', size='tiny'
		)


def test_models_share_condition_and_separate_outputs(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	jobs = [
		resolve_f3_lithology_five_way_job(
			config, model=model_id, layout='layout_000', size='small'
		)
		for model_id in FIVE_WAY_MODEL_IDS
	]

	assert len({job.condition_dir for job in jobs}) == 1
	assert len({job.output_dir for job in jobs}) == len(jobs)
	for job in jobs:
		assert f'model={job.model.model_id}' in str(job.output_dir)
		assert job.decoder_dir.parent == job.output_dir
		assert job.prediction_dir.parent == job.output_dir
		assert job.evaluation_dir.parent == job.output_dir


def test_decoder_contract_is_fixed_and_encoder_frozen(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	job = resolve_f3_lithology_five_way_job(
		config, model='mae', layout='layout_000', size='small'
	)
	mapping = five_way_runner._decoder_config_mapping(job)  # noqa: SLF001
	decoder_config = f3_lithology_voxel_decoder_config_from_mapping(mapping)

	assert decoder_config.model['freeze_encoder'] is True
	assert decoder_config.decoder.spec == FIXED_DECODER_CONTRACT['spec']
	assert decoder_config.decoder.embedding_dim == 384
	assert decoder_config.decoder.class_count == 6
	assert decoder_config.decoder.hidden_channels == (128, 64, 32)
	assert decoder_config.train.epochs == 50
	assert decoder_config.train.batch_size == 1
	assert decoder_config.train.learning_rate == 0.001
	assert decoder_config.train.weight_decay == 0.0001
	assert decoder_config.train.steps_per_epoch == 440
	assert decoder_config.train.sampling_mode == 'uniform_tiles_with_replacement'
	assert decoder_config.train.amp is True
	assert decoder_config.train.seed == 42000
	assert decoder_config.embeddings['checkpoint_path'] == str(
		job.model.checkpoint
	)

	tampered = dict(universe)
	tampered['decoder'] = {'epochs': 1}
	with pytest.raises(ValueError, match='decoder'):
		f3_lithology_five_way_config_from_mapping(tampered)


def test_dry_run_shares_supervision_identity_and_writes_nothing(
	universe: dict[str, object],
	tmp_path: Path,
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	write_condition(universe, 'layout_000', 'small')
	root = Path(universe['paths']['artifact_root'])
	config_path = tmp_path / 'five_way.yaml'
	config_path.write_text(yaml.safe_dump(universe), encoding='utf-8')
	before = _files_snapshot(root)

	summaries = []
	for model_id in FIVE_WAY_MODEL_IDS:
		job = resolve_f3_lithology_five_way_job(
			config, model=model_id, layout='layout_000', size='small'
		)
		summaries.append(inspect_f3_lithology_five_way_job(job))
	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			'proc/seis_ssl_cluster/run_f3_lithology_five_way.py',
			'--config',
			str(config_path),
			'--model',
			'mae',
			'--layout',
			'layout_000',
			'--size',
			'small',
			'--dry-run',
		],
		check=True,
		capture_output=True,
		text=True,
	)

	assert 'execution: dry-run; no files written' in result.stdout
	assert 'decoder_initial_state_sha256' in result.stdout
	shared_keys = (
		'condition_dir',
		'selected_token_identity_sha256',
		'train_voxel_count',
		'validation_mask_sha256',
		'validation_voxel_count',
	)
	for key in shared_keys:
		assert len({str(summary[key]) for summary in summaries}) == 1
	assert len({summary['decoder_dir'] for summary in summaries}) == 5
	assert _files_snapshot(root) == before


def test_run_fails_fast_when_source_audit_fails(
	universe: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_smoke(monkeypatch)
	config = f3_lithology_five_way_config_from_mapping(universe)
	write_condition(universe, 'layout_000', 'small')
	metadata_path = (
		Path(universe['models'][3]['embeddings_dir'])
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	payload = json.loads(metadata_path.read_text(encoding='utf-8'))
	payload['checkpoint_sha256'] = '0' * 64
	metadata_path.write_text(json.dumps(payload), encoding='utf-8')
	job = resolve_f3_lithology_five_way_job(
		config, model='mae', layout='layout_000', size='small'
	)

	with pytest.raises(ValueError, match='checkpoint_sha256'):
		run_f3_lithology_five_way_job(job, device='cpu')
	assert not job.output_dir.exists()


def test_completed_job_is_not_overwritten(
	universe: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_smoke(monkeypatch)
	config = f3_lithology_five_way_config_from_mapping(universe)
	job = resolve_f3_lithology_five_way_job(
		config, model='mae', layout='layout_000', size='small'
	)
	job.evaluation_dir.mkdir(parents=True)
	job.metrics_path.write_text('{}', encoding='utf-8')

	with pytest.raises(FileExistsError, match='already completed'):
		run_f3_lithology_five_way_job(job, device='cpu')


def test_resume_must_target_the_job_decoder_checkpoint(
	universe: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_smoke(monkeypatch)
	config = f3_lithology_five_way_config_from_mapping(universe)
	write_condition(universe, 'layout_000', 'small')
	job = resolve_f3_lithology_five_way_job(
		config, model='mae', layout='layout_000', size='small'
	)
	foreign = (
		job.config.runs_root
		/ 'model=random/layout=layout_000/size=small/decoder/latest.pt'
	)

	with pytest.raises(ValueError, match='resume must be the decoder latest'):
		run_f3_lithology_five_way_job(job, device='cpu', resume=foreign)


def test_synthetic_end_to_end_connection(
	universe: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_smoke(monkeypatch)
	config = f3_lithology_five_way_config_from_mapping(universe)
	write_condition(universe, 'layout_000', 'small')
	job = resolve_f3_lithology_five_way_job(
		config, model='mae', layout='layout_000', size='small'
	)
	inspection = inspect_f3_lithology_five_way_job(job)

	result = run_f3_lithology_five_way_job(job, device='cpu')

	assert result['completed'] is True
	metrics = json.loads(job.metrics_path.read_text(encoding='utf-8'))
	assert 0.0 <= metrics['macro_f1'] <= 1.0
	for key in ('mean_iou', 'balanced_accuracy', 'weighted_f1'):
		assert 0.0 <= metrics[key] <= 1.0
	assert metrics['evaluation_voxel_count'] > 0
	run_metadata = json.loads(
		(job.decoder_dir / 'run_metadata.json').read_text(encoding='utf-8')
	)
	assert (
		run_metadata['initial_model_state_sha256']
		== inspection['decoder_initial_state_sha256']
	)
	prediction_metadata_path = job.prediction_dir / 'prediction_metadata.json'
	assert prediction_metadata_path.is_file()
	# The summary reads its comparison identity out of these two artifacts, so a
	# completed job must carry every SHA that identity is built from.
	assert metrics['aggregation_unit'] == 'unique_validation_voxel'
	evaluation_metadata = json.loads(
		(job.evaluation_dir / 'evaluation_metadata.json').read_text(encoding='utf-8')
	)
	recorded_prediction = evaluation_metadata['inputs']['prediction_metadata']
	assert recorded_prediction['path'] == str(prediction_metadata_path)
	assert recorded_prediction['sha256'] == file_sha256(prediction_metadata_path)
	source_identity = json.loads(
		prediction_metadata_path.read_text(encoding='utf-8')
	)['source_identity']
	assert source_identity['decoder_checkpoint']['path'] == str(
		job.decoder_dir / 'best.pt'
	)
	identities = source_identity['artifact_identities']
	for key, name in (
		('embeddings', f'{SURVEY_ID}.embeddings.npy'),
		('embedding_metadata', f'{SURVEY_ID}.embedding_metadata.json'),
		('valid_tokens', f'{SURVEY_ID}.valid_tokens.npy'),
	):
		assert identities[key]['path'] == str(job.model.embeddings_dir / name)
		assert len(identities[key]['sha256']) == 64

	with pytest.raises(FileExistsError, match='already completed'):
		run_f3_lithology_five_way_job(job, device='cpu')


def test_completed_decoder_is_reused_instead_of_retrained(
	universe: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_smoke(monkeypatch)
	config = f3_lithology_five_way_config_from_mapping(universe)
	write_condition(universe, 'layout_000', 'small')
	job = resolve_f3_lithology_five_way_job(
		config, model='mae', layout='layout_000', size='small'
	)
	first = run_f3_lithology_five_way_job(job, device='cpu')
	assert first['reused_decoder'] is False
	decoder_before = _files_snapshot(job.decoder_dir)

	shutil.rmtree(job.evaluation_dir)
	shutil.rmtree(job.prediction_dir)
	second = run_f3_lithology_five_way_job(job, device='cpu')

	assert second['completed'] is True
	assert second['reused_decoder'] is True
	assert _files_snapshot(job.decoder_dir) == decoder_before
	assert job.metrics_path.is_file()


def test_completed_prediction_is_reused_when_evaluation_is_missing(
	universe: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_smoke(monkeypatch)
	config = f3_lithology_five_way_config_from_mapping(universe)
	write_condition(universe, 'layout_000', 'small')
	job = resolve_f3_lithology_five_way_job(
		config, model='mae', layout='layout_000', size='small'
	)
	run_f3_lithology_five_way_job(job, device='cpu')
	prediction_before = _files_snapshot(job.prediction_dir)

	shutil.rmtree(job.evaluation_dir)
	result = run_f3_lithology_five_way_job(job, device='cpu')

	assert result['completed'] is True
	assert _files_snapshot(job.prediction_dir) == prediction_before


def test_interrupted_decoder_requires_resume_and_then_finishes(
	universe: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_smoke(monkeypatch)
	config = f3_lithology_five_way_config_from_mapping(universe)
	write_condition(universe, 'layout_000', 'small')
	job = resolve_f3_lithology_five_way_job(
		config, model='mae', layout='layout_000', size='small'
	)
	partial = run_f3_lithology_five_way_job(job, device='cpu', max_steps=1)
	assert partial['completed'] is False

	with pytest.raises(FileExistsError, match='resume it with --resume'):
		run_f3_lithology_five_way_job(job, device='cpu')

	resumed = run_f3_lithology_five_way_job(
		job, device='cpu', resume=job.decoder_dir / 'latest.pt'
	)
	assert resumed['completed'] is True
	assert resumed['reused_decoder'] is False


def test_resume_is_rejected_once_the_decoder_is_complete(
	universe: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_smoke(monkeypatch)
	config = f3_lithology_five_way_config_from_mapping(universe)
	write_condition(universe, 'layout_000', 'small')
	job = resolve_f3_lithology_five_way_job(
		config, model='mae', layout='layout_000', size='small'
	)
	run_f3_lithology_five_way_job(job, device='cpu')
	shutil.rmtree(job.evaluation_dir)

	with pytest.raises(FileExistsError, match='already completed'):
		run_f3_lithology_five_way_job(
			job, device='cpu', resume=job.decoder_dir / 'latest.pt'
		)


def test_foreign_decoder_directory_is_rejected(
	universe: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_smoke(monkeypatch)
	config = f3_lithology_five_way_config_from_mapping(universe)
	write_condition(universe, 'layout_000', 'small')
	job = resolve_f3_lithology_five_way_job(
		config, model='mae', layout='layout_000', size='small'
	)
	run_f3_lithology_five_way_job(job, device='cpu')
	shutil.rmtree(job.evaluation_dir)
	resolved_path = job.decoder_dir / 'resolved_config.json'
	payload = json.loads(resolved_path.read_text(encoding='utf-8'))
	payload['train']['seed'] = 1
	resolved_path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='does not match this job'):
		run_f3_lithology_five_way_job(job, device='cpu')


def test_dry_run_reports_active_inline_and_crossline_lines(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	write_condition(universe, 'layout_000', 'small')
	job = resolve_f3_lithology_five_way_job(
		config, model='mae', layout='layout_000', size='small'
	)

	summary = inspect_f3_lithology_five_way_job(job)

	assert summary['inline_lines'] == [100]
	assert summary['crossline_lines'] == [200]
	assert set(summary['per_line_contributions']) == {'inline:100', 'crossline:200'}
	assert sum(summary['per_line_contributions'].values()) == (
		summary['train_voxel_count']
	)


@pytest.mark.parametrize(
	('mutation', 'match'),
	[
		({}, r'active_lines\.inline must be a non-empty list'),
		({'inline': [], 'crossline': [200]}, r'active_lines\.inline'),
		({'inlines': [100], 'crosslines': [200]}, r'active_lines\.inline'),
		({'inline': [100]}, r'active_lines\.crossline'),
	],
)
def test_dry_run_fails_loudly_on_incomplete_active_lines(
	universe: dict[str, object],
	mutation: dict[str, object],
	match: str,
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	condition = write_condition(universe, 'layout_000', 'small')
	metadata_path = condition / 'section_layout_metadata.json'
	payload = json.loads(metadata_path.read_text(encoding='utf-8'))
	payload['active_lines'] = mutation
	metadata_path.write_text(json.dumps(payload), encoding='utf-8')
	job = resolve_f3_lithology_five_way_job(
		config, model='mae', layout='layout_000', size='small'
	)

	with pytest.raises(ValueError, match=match):
		inspect_f3_lithology_five_way_job(job)


def test_interrupted_decoder_without_a_checkpoint_reports_restart_only(
	universe: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_smoke(monkeypatch)
	config = f3_lithology_five_way_config_from_mapping(universe)
	write_condition(universe, 'layout_000', 'small')
	job = resolve_f3_lithology_five_way_job(
		config, model='mae', layout='layout_000', size='small'
	)
	job.decoder_dir.mkdir(parents=True)
	(job.decoder_dir / 'resolved_config.json').write_text('{}', encoding='utf-8')

	with pytest.raises(FileExistsError, match='only be restarted'):
		run_f3_lithology_five_way_job(job, device='cpu')


def test_dry_run_rejects_per_line_contributions_that_miss_a_line(
	universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(universe)
	condition = write_condition(universe, 'layout_000', 'small')
	metadata_path = condition / 'section_layout_metadata.json'
	payload = json.loads(metadata_path.read_text(encoding='utf-8'))
	payload['identity']['per_line_contributions'] = {'inline:100': 512}
	metadata_path.write_text(json.dumps(payload), encoding='utf-8')
	job = resolve_f3_lithology_five_way_job(
		config, model='mae', layout='layout_000', size='small'
	)

	with pytest.raises(ValueError, match='must cover exactly'):
		inspect_f3_lithology_five_way_job(job)
