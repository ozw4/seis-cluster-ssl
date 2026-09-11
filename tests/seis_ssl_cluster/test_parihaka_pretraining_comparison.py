"""Scientific contracts and report boundaries for the Parihaka completion."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.parihaka import pretraining_comparison as comparison

EXPERIMENT = (
	Path(__file__).resolve().parents[2]
	/ 'experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1'
)


@pytest.mark.parametrize('source', ['local_bt3', 'random'])
@pytest.mark.parametrize(('tag', 'weight'), [('010', 0.1), ('020', 0.2)])
def test_hmm_arms_preserve_budget_and_self_target_lineage(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
	source: str,
	tag: str,
	weight: float,
) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	config = load_config(
		EXPERIMENT / f'20_stage2/{source}/distill{tag}/01_full_25ep.yaml'
	)
	cluster = load_config(
		EXPERIMENT / f'10_hmm_targets/{source}/01_cluster_hmm_k6.yaml'
	)
	assert config['teacher']['checkpoint'] == config['student']['init_checkpoint']
	assert config['student']['unfreeze_top_blocks'] == 1
	assert config['train']['epochs'] == 25
	assert config['train']['samples_per_epoch'] == 10000
	assert config['train']['batch_size'] == 16
	assert config['train']['amp'] is False
	assert config['train']['max_steps'] is None
	assert config['train']['seed'] == 42
	assert config['loss']['distillation_weight'] == weight
	assert config['pseudo_targets']['k'] == 6
	assert cluster['clustering']['method'] == 'stratigraphic_hmm_kmeans'
	assert cluster['clustering']['k_values'] == [6]
	assert cluster['clustering']['stratigraphic_hmm']['transition']['same_cost'] == 0.03
	if source == 'random':
		assert 'random_encoder_' in config['teacher']['checkpoint']
		assert 'random_encoder_' in cluster['embeddings']['input_dir']
	else:
		assert (
			'local_bt_rot90_asym_g060_v1/full_3ep/' in config['teacher']['checkpoint']
		)
		assert 'local_bt_rot90_asym_g060_v1/' in cluster['embeddings']['input_dir']


def test_checkpoint_completion_checks_stratigraphy_budget_and_loss(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	config_path = EXPERIMENT / '20_stage2/local_bt3/distill010/01_full_25ep.yaml'
	raw = load_config(config_path)
	Path(raw['pseudo_targets']['input_dir']).mkdir(parents=True)
	teacher = Path(raw['teacher']['checkpoint'])
	teacher.parent.mkdir(parents=True)
	teacher.touch()
	config = resolve_strat_hmm_pretext_config(raw)
	checkpoint = tmp_path / 'latest.pt'
	assert comparison.checkpoint_complete(checkpoint, config_path) is False
	payload = {
		'stratigraphy_config': config,
		'epoch': 25,
		'global_step': 15625,
		'training_state': {
			'stage': 'train_strat_hmm_pretext',
			'checkpoint_kind': 'epoch',
		},
		'control_identity': {
			**config['identity'],
			'input_identities': {
				'teacher_checkpoint': {
					'path': str(teacher),
					'sha256': comparison.file_sha256(teacher),
				},
			},
		},
	}
	torch.save(payload, checkpoint)
	assert comparison.checkpoint_complete(checkpoint, config_path) is True
	payload['training_state']['checkpoint_kind'] = 'step'
	torch.save(payload, checkpoint)
	assert comparison.checkpoint_complete(checkpoint, config_path) is False
	payload['training_state']['checkpoint_kind'] = 'epoch'
	payload['training_state']['stage'] = 'train_amp_mae'
	torch.save(payload, checkpoint)
	assert comparison.checkpoint_complete(checkpoint, config_path) is False
	payload['training_state']['stage'] = 'train_strat_hmm_pretext'
	payload['global_step'] = 15624
	torch.save(payload, checkpoint)
	assert comparison.checkpoint_complete(checkpoint, config_path) is False
	config['loss']['distillation_weight'] = 0.2
	torch.save(payload, checkpoint)
	with pytest.raises(ValueError, match='mismatched loss'):
		comparison.checkpoint_complete(checkpoint, config_path)
	config['loss']['distillation_weight'] = 0.1
	torch.save(payload, checkpoint)
	teacher.write_bytes(b'changed parent checkpoint')
	with pytest.raises(ValueError, match='live training input checksum mismatch'):
		comparison.checkpoint_complete(checkpoint, config_path)


def test_summary_emits_exact_three_file_set_for_all_135_cells(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	models = {f'arm_{index}': {} for index in range(9)}
	rows = [
		{
			'arm_id': arm,
			'model_id': arm,
			'layout_id': f'layout_{layout:03d}',
			'data_size': size,
			'test_channel_iou': 0.25,
		}
		for arm in models
		for layout in range(5)
		for size in ('small', 'medium', 'large')
	]
	monkeypatch.setattr(comparison, 'inspect_comparison', lambda _: (rows, models))
	config = {'models': models, 'output_dir': str(tmp_path / 'summary')}
	paths = comparison.summarize_comparison(config)
	assert {path.name for path in paths} == set(comparison.OUTPUT_NAMES)
	assert {path.name for path in paths[0].parent.iterdir()} == set(
		comparison.OUTPUT_NAMES
	)
	payload = json.loads(paths[1].read_text())
	assert payload['complete_cells'] == 135
	assert payload['complete_arms'] == 9
	assert payload['model_means']['arm_0']['all_15'] == 0.25
	with paths[0].open(newline='') as stream:
		assert len(list(csv.DictReader(stream))) == 135
	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		comparison.summarize_comparison(config)


def test_summary_rejects_missing_arm_and_missing_evidence(tmp_path: Path) -> None:
	output = tmp_path / 'summary'
	with pytest.raises(ValueError, match='exactly nine'):
		comparison.summarize_comparison({'models': {}, 'output_dir': str(output)})
	models = {
		model: {
			'model_id': model,
			'checkpoint': str(tmp_path / 'missing.pt'),
		}
		for model in comparison.EXPECTED_MODEL_IDS
	}
	with pytest.raises(FileNotFoundError):
		comparison.summarize_comparison({'models': models, 'output_dir': str(output)})
	assert not output.exists()


def test_summary_requires_exact_requested_model_ids(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	config = load_config(EXPERIMENT / '50_summary/01_all_nine_arms.yaml')
	assert set(config['models']) == set(comparison.EXPECTED_MODEL_IDS)
	for model, entry in config['models'].items():
		assert entry['model_id'] == model
		if 'hmm' in model:
			assert Path(entry['train_config']).is_file()
	config['models']['mae']['model_id'] = 'random'
	with pytest.raises(ValueError, match='model IDs must match'):
		comparison.inspect_comparison(config)
	config['models']['incorrect_arm'] = config['models'].pop('mae')
	with pytest.raises(ValueError, match='exactly nine specified'):
		comparison.inspect_comparison(config)


def test_summary_requires_decoder_completion_before_accepting_a_cell(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	config = load_config(EXPERIMENT / '50_summary/01_all_nine_arms.yaml')
	monkeypatch.setattr(comparison, '_inspect_checkpoint', lambda _: {'sha256': 'hash'})
	metrics = {'completion_test': True}
	monkeypatch.setattr(
		comparison,
		'inspect_channel_model_results',
		lambda *_, **__: {('mae', 'layout_000', 'small'): metrics},
	)
	calls: list[Path] = []

	def reject_incomplete(job_dir: Path, *, metrics: object) -> dict[str, object]:
		assert metrics == {'completion_test': True}
		calls.append(job_dir)
		raise ValueError('incomplete decoder evidence')

	monkeypatch.setattr(comparison, 'inspect_completed_channel_job', reject_incomplete)
	with pytest.raises(ValueError, match='incomplete decoder evidence'):
		comparison.inspect_comparison(config)
	assert calls == [
		Path(config['models']['mae']['runs_root'])
		/ 'model=mae/layout=layout_000/size=small'
	]


def test_gpu1_staging_changes_only_the_run_roots(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	downstream = EXPERIMENT / '40_downstream'
	canonical = load_config(downstream / '02_mae_hmm_k6_distill010.yaml')
	staging = load_config(downstream / '03_mae_hmm_k6_distill010_gpu1_staging.yaml')
	expected_root = str(
		tmp_path / 'channel_benchmark/pretraining_comparison_completion_v1/gpu1_staging'
	)
	for section in ('inputs', 'outputs'):
		assert staging[section]['runs_root'] == expected_root
		assert staging[section]['runs_root'] != canonical[section]['runs_root']
		staging[section]['runs_root'] = canonical[section]['runs_root']
	assert staging == canonical
