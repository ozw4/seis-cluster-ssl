from __future__ import annotations

import json
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.pretraining import resolve_strat_hmm_pretext_config
from seis_ssl_cluster.f3.lithology.guardrails import (
	F3_STRAT_HMM_M1_DISTILL_ONLY_MODEL_TAG,
	F3_STRAT_HMM_M1_GUARDRAIL_ROLES,
	F3_STRAT_HMM_M1_GUARDRAIL_SUITE_NAME,
	F3_STRAT_HMM_M1_SHUFFLED_HMM_MODEL_TAG,
	f3_guardrail_jobs_config_from_mapping,
	f3_guardrail_summary_config_from_mapping,
	f3_shuffled_hmm_target_config_from_mapping,
	summarize_f3_strat_hmm_m1_guardrails,
)

ROOT = Path('experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails')


@pytest.mark.parametrize(
	'filename',
	[
		'01_train_distillation_only_smoke.yaml',
		'02_train_distillation_only_full.yaml',
		'04_train_shuffled_hmm_full.yaml',
	],
)
def test_guardrail_training_configs_resolve(
	filename: str,
	tmp_path: Path,
) -> None:
	raw = load_config(ROOT / filename)
	artifact_root = tmp_path / 'artifacts'
	pseudo_targets = tmp_path / 'pseudo_targets'
	pseudo_targets.mkdir()
	checkpoint = tmp_path / 'mae.pt'
	checkpoint.touch()
	raw['paths']['artifact_root'] = str(artifact_root)
	raw['paths']['output_root'] = str(
		artifact_root / 'pretraining' / 'f3' / Path(filename).stem,
	)
	raw['pseudo_targets']['input_dir'] = str(pseudo_targets)
	raw['teacher']['checkpoint'] = str(checkpoint)
	raw['student']['init_checkpoint'] = str(checkpoint)

	resolved = resolve_strat_hmm_pretext_config(raw)

	if 'distillation_only' in filename:
		assert resolved['loss']['prototype_weight'] == 0.0
		assert resolved['loss']['usage_weight'] == 0.0
	assert resolved['student']['unfreeze_top_blocks'] == 1
	assert resolved['loss']['distillation_weight'] == 0.2


def test_shuffled_target_contract_resolves_with_preservation_guarantees() -> None:
	config = f3_shuffled_hmm_target_config_from_mapping(
		load_config(ROOT / '03_build_shuffled_hmm_pseudo_targets.yaml'),
	)

	assert config.suite_name == F3_STRAT_HMM_M1_GUARDRAIL_SUITE_NAME
	assert config.seed == 188
	assert config.shuffle_scope == 'global_valid_tokens'
	assert config.source_root != config.output_root
	assert config.overwrite is False


@pytest.mark.parametrize(
	'filename',
	[
		'05_extract_guardrail_embeddings.yaml',
		'06_build_guardrail_token_datasets.yaml',
		'07_run_guardrail_probes.yaml',
	],
)
def test_guardrail_downstream_contracts_have_stable_isolated_names(
	filename: str,
) -> None:
	config = f3_guardrail_jobs_config_from_mapping(load_config(ROOT / filename))

	assert tuple(job.role for job in config.jobs) == (
		'distillation_only',
		'shuffled_hmm',
	)
	assert tuple(job.model_tag for job in config.jobs) == (
		F3_STRAT_HMM_M1_DISTILL_ONLY_MODEL_TAG,
		F3_STRAT_HMM_M1_SHUFFLED_HMM_MODEL_TAG,
	)
	assert len({job.output_path for job in config.jobs}) == 2
	assert all(
		'strat_hmm_pretext_m1_k6_topblock1_distill' not in job.output_path.parts
		for job in config.jobs
	)


def test_guardrail_summary_reports_missing_outputs_as_pending(
	tmp_path: Path,
) -> None:
	raw = _summary_config(tmp_path, strict=False)
	_write_metrics(Path(raw['models']['baseline']['metrics_json']), offset=0.0)
	_write_metrics(Path(raw['models']['candidate']['metrics_json']), offset=0.1)
	config = f3_guardrail_summary_config_from_mapping(raw)

	result = summarize_f3_strat_hmm_m1_guardrails(config)
	payload = json.loads(result.summary_json.read_text(encoding='utf-8'))

	assert result.pending_roles == ('distillation_only', 'shuffled_hmm')
	assert tuple(row['role'] for row in payload['models']) == (
		F3_STRAT_HMM_M1_GUARDRAIL_ROLES
	)
	assert [row['status'] for row in payload['models']] == [
		'complete',
		'complete',
		'pending',
		'pending',
	]
	assert 'distillation_only | pending' in result.summary_markdown.read_text(
		encoding='utf-8',
	)


def test_guardrail_summary_strict_mode_rejects_missing_output(
	tmp_path: Path,
) -> None:
	config = f3_guardrail_summary_config_from_mapping(
		_summary_config(tmp_path, strict=True),
	)

	with pytest.raises(FileNotFoundError, match='missing guardrail metrics'):
		summarize_f3_strat_hmm_m1_guardrails(config)


def test_guardrail_summary_config_rejects_candidate_output_root(
	tmp_path: Path,
) -> None:
	raw = _summary_config(tmp_path, strict=False)
	raw['outputs']['output_dir'] = str(
		tmp_path / 'strat_hmm_pretext_m1_k6_topblock1_distill',
	)

	with pytest.raises(ValueError, match='candidate root'):
		f3_guardrail_summary_config_from_mapping(raw)


def _summary_config(tmp_path: Path, *, strict: bool) -> dict[str, object]:
	raw = load_config(ROOT / '08_summarize_guardrails.yaml')
	raw['suite']['strict'] = strict
	for role in F3_STRAT_HMM_M1_GUARDRAIL_ROLES:
		raw['models'][role]['metrics_json'] = str(tmp_path / role / 'metrics.json')
	raw['outputs']['output_dir'] = str(tmp_path / 'guardrail_summary')
	return raw


def _write_metrics(path: Path, *, offset: float) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(
			{
				'macro_f1': 0.50 + offset,
				'mean_iou': 0.40 + offset,
				'balanced_accuracy': 0.60 + offset,
				'accuracy': 0.70 + offset,
				'weighted_f1': 0.65 + offset,
			},
		)
		+ '\n',
		encoding='utf-8',
	)
