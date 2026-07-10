from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_robustness import (
	f3_lithology_label_budget_config_from_mapping,
)
from seis_ssl_cluster.config.pretraining import resolve_strat_hmm_pretext_config
from seis_ssl_cluster.f3.lithology.guardrails import (
	F3_STRAT_HMM_M1_BASELINE_MODEL_TAG,
	F3_STRAT_HMM_M1_DISTILL_ONLY_MODEL_TAG,
	F3_STRAT_HMM_M1_GUARDRAIL_MODEL_TAGS,
	F3_STRAT_HMM_M1_GUARDRAIL_ROLES,
	F3_STRAT_HMM_M1_GUARDRAIL_SUITE_NAME,
	F3_STRAT_HMM_M1_SHUFFLED_HMM_MODEL_TAG,
	f3_guardrail_jobs_config_from_mapping,
	f3_guardrail_summary_config_from_mapping,
	f3_shuffled_hmm_target_config_from_mapping,
	summarize_f3_strat_hmm_m1_guardrails,
)
from seis_ssl_cluster.stratigraphy import shuffle_strat_hmm_pseudo_targets
from seis_ssl_cluster.training import load_checkpoint
from seis_ssl_cluster.training.strat_hmm_pretraining import (
	run_strat_hmm_pretext_training,
)
from tests.seis_ssl_cluster.test_strat_hmm_pretraining_head_only import (
	_raw_config,
)

ROOT = Path('experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails')


@pytest.mark.parametrize(
	'filename',
	[
		'01_train_distillation_only_smoke.yaml',
		'02_train_distillation_only_full.yaml',
		'07_train_shuffled_hmm_smoke.yaml',
		'08_train_shuffled_hmm_full.yaml',
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


def test_shuffled_hmm_full_training_only_changes_targets_and_output() -> None:
	candidate = load_config(
		Path('experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1')
		/ '03_train_single_head_topblock_distill_full.yaml',
	)
	guardrail = load_config(ROOT / '08_train_shuffled_hmm_full.yaml')
	candidate['paths']['output_root'] = guardrail['paths']['output_root']
	candidate['pseudo_targets']['input_dir'] = guardrail['pseudo_targets']['input_dir']

	assert guardrail == candidate
	assert guardrail['paths']['output_root'].endswith(
		F3_STRAT_HMM_M1_SHUFFLED_HMM_MODEL_TAG,
	)


def test_shuffled_target_contract_resolves_with_preservation_guarantees() -> None:
	config = f3_shuffled_hmm_target_config_from_mapping(
		load_config(ROOT / '03_build_shuffled_hmm_pseudo_targets.yaml'),
	)

	assert config.suite_name == F3_STRAT_HMM_M1_GUARDRAIL_SUITE_NAME
	assert config.seed == 42
	assert config.shuffle_scope == 'global_valid_tokens'
	assert config.source_root != config.output_root
	assert config.overwrite is False


def test_training_accepts_generated_shuffled_pseudo_target_artifacts(
	tmp_path: Path,
) -> None:
	raw = _raw_config(
		tmp_path,
		encoder_depth=2,
		unfreeze_top_blocks=1,
		distillation_weight=0.2,
		max_steps=1,
	)
	source_root = Path(raw['pseudo_targets']['input_dir'])
	shuffled_root = tmp_path / 'strat_hmm_m1_guardrail_shuffled_hmm'
	shuffle_strat_hmm_pseudo_targets(
		source_root,
		shuffled_root,
		k=3,
		seed=42,
	)
	raw['pseudo_targets']['input_dir'] = str(shuffled_root)
	config = resolve_strat_hmm_pretext_config(raw)

	checkpoint_path = run_strat_hmm_pretext_training(config)
	payload = load_checkpoint(checkpoint_path, map_location='cpu')

	assert payload['global_step'] == 1
	assert payload['stratigraphy_config']['pseudo_targets']['input_dir'] == str(
		shuffled_root,
	)


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


@pytest.mark.parametrize(
	('filename', 'expected_model_tag'),
	[
		(
			'14_build_distillation_only_label_budget_datasets.yaml',
			F3_STRAT_HMM_M1_DISTILL_ONLY_MODEL_TAG,
		),
		(
			'16_build_shuffled_hmm_label_budget_datasets.yaml',
			F3_STRAT_HMM_M1_SHUFFLED_HMM_MODEL_TAG,
		),
	],
)
def test_guardrail_label_budget_configs_use_required_paired_conditions(
	filename: str,
	expected_model_tag: str,
) -> None:
	config = f3_lithology_label_budget_config_from_mapping(
		load_config(ROOT / filename),
	)

	assert config.per_class_caps == (25, 100, 500, None)
	assert config.subsample_seeds == (0, 1, 2, 3, 4)
	assert config.baseline.model_tag == F3_STRAT_HMM_M1_BASELINE_MODEL_TAG
	assert config.candidate.model_tag == expected_model_tag


def test_guardrail_summary_configures_existing_label_budget_artifacts() -> None:
	config = f3_guardrail_summary_config_from_mapping(
		load_config(ROOT / '13_summarize_guardrails.yaml'),
	)
	by_role = {model.role: model for model in config.models}

	for role in ('candidate', 'distillation_only', 'shuffled_hmm'):
		assert by_role[role].label_budget_summary_csv is not None
		assert by_role[role].label_budget_summary_csv.name == 'summary_by_budget.csv'
		assert by_role[role].label_budget_suite_manifest_json is not None
		assert (
			by_role[role].label_budget_suite_manifest_json.name
			== 'suite_manifest.json'
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
	assert len(result.warnings) == 2


def test_guardrail_summary_writes_four_model_publishable_comparison(
	tmp_path: Path,
) -> None:
	raw = _summary_config(tmp_path, strict=True)
	for index, role in enumerate(F3_STRAT_HMM_M1_GUARDRAIL_ROLES):
		_write_metrics(
			Path(raw['models'][role]['metrics_json']),
			offset=index / 100,
		)
	config = f3_guardrail_summary_config_from_mapping(raw)

	result = summarize_f3_strat_hmm_m1_guardrails(config)
	with result.comparison_table.open(newline='', encoding='utf-8') as handle:
		rows = list(csv.DictReader(handle))

	assert result.comparison_table.name == 'guardrail_comparison_table.csv'
	assert result.summary_json.name == 'guardrail_comparison_summary.json'
	assert result.summary_markdown.name == 'guardrail_comparison_report.md'
	assert [row['role'] for row in rows] == list(F3_STRAT_HMM_M1_GUARDRAIL_ROLES)
	assert tuple(rows[0]) == (
		'role',
		'model_tag',
		'status',
		'accuracy',
		'balanced_accuracy',
		'macro_f1',
		'weighted_f1',
		'mean_iou',
		*(f'class_{class_id}_f1' for class_id in range(6)),
	)


@pytest.mark.parametrize(
	('distillation_offset', 'shuffled_offset', 'expected'),
	[
		(-0.02, -0.03, 'evidence supports an ordered structured HMM'),
		(0.0, -0.03, 'generic top-block adaptation'),
		(-0.02, 0.0, 'extra CE regularization'),
	],
)
def test_guardrail_decision_changes_with_guardrail_performance(
	tmp_path: Path,
	distillation_offset: float,
	shuffled_offset: float,
	expected: str,
) -> None:
	raw = _summary_config(tmp_path, strict=True)
	for role, offset in {
		'baseline': -0.1,
		'candidate': 0.0,
		'distillation_only': distillation_offset,
		'shuffled_hmm': shuffled_offset,
	}.items():
		_write_metrics(Path(raw['models'][role]['metrics_json']), offset=offset)

	result = summarize_f3_strat_hmm_m1_guardrails(
		f3_guardrail_summary_config_from_mapping(raw),
	)
	payload = json.loads(result.summary_json.read_text(encoding='utf-8'))

	assert expected in payload['decision']['interpretation']


def test_guardrail_low_budget_deltas_are_aggregated(tmp_path: Path) -> None:
	raw = _summary_config(tmp_path, strict=True)
	for role in F3_STRAT_HMM_M1_GUARDRAIL_ROLES:
		_write_metrics(Path(raw['models'][role]['metrics_json']), offset=0.0)
	for role, delta in {
		'candidate': 0.10,
		'distillation_only': 0.04,
		'shuffled_hmm': 0.02,
	}.items():
		path = tmp_path / role / 'summary_by_budget.csv'
		_write_label_budget(path, delta=delta)
		raw['models'][role]['label_budget_summary_csv'] = str(path)
		manifest = tmp_path / role / 'suite_manifest.json'
		_write_label_budget_manifest(manifest, role=role)
		raw['models'][role]['label_budget_suite_manifest_json'] = str(manifest)

	result = summarize_f3_strat_hmm_m1_guardrails(
		f3_guardrail_summary_config_from_mapping(raw),
	)
	payload = json.loads(result.summary_json.read_text(encoding='utf-8'))
	cap25 = payload['low_budget']['budgets'][0]

	assert payload['low_budget']['status'] == 'complete'
	assert payload['low_budget']['pairing_provenance']['status'] == 'verified'
	assert (
		payload['low_budget']['basis']
		== 'difference_of_shared_baseline_paired_mean_deltas'
	)
	assert cap25['budget_id'] == 'cap25'
	assert cap25['candidate_minus_distillation_only']['macro_f1'] == pytest.approx(
		0.06,
	)
	assert cap25['candidate_minus_shuffled_hmm']['macro_f1'] == pytest.approx(0.08)


def test_guardrail_low_budget_requires_expected_budget_overlap(
	tmp_path: Path,
) -> None:
	raw = _summary_config(tmp_path, strict=False)
	for role in F3_STRAT_HMM_M1_GUARDRAIL_ROLES:
		_write_metrics(Path(raw['models'][role]['metrics_json']), offset=0.0)
	for role in ('candidate', 'distillation_only', 'shuffled_hmm'):
		budget_ids = ('cap25',) if role == 'candidate' else ('cap100',)
		summary = tmp_path / role / 'summary_by_budget.csv'
		_write_label_budget(summary, delta=0.1, budget_ids=budget_ids)
		raw['models'][role]['label_budget_summary_csv'] = str(summary)
		manifest = tmp_path / role / 'suite_manifest.json'
		_write_label_budget_manifest(manifest, role=role, budget_ids=budget_ids)
		raw['models'][role]['label_budget_suite_manifest_json'] = str(manifest)

	result = summarize_f3_strat_hmm_m1_guardrails(
		f3_guardrail_summary_config_from_mapping(raw),
	)
	payload = json.loads(result.summary_json.read_text(encoding='utf-8'))

	assert payload['low_budget']['status'] == 'partial'
	assert payload['low_budget']['pairing_provenance']['status'] == 'unverified'
	assert payload['low_budget']['budgets'] == []
	assert any(
		'missing for expected candidate budgets' in warning
		for warning in payload['warnings']
	)


def test_guardrail_low_budget_rejects_unpaired_subsampling_indices(
	tmp_path: Path,
) -> None:
	raw = _summary_config(tmp_path, strict=True)
	for role in F3_STRAT_HMM_M1_GUARDRAIL_ROLES:
		_write_metrics(Path(raw['models'][role]['metrics_json']), offset=0.0)
	for role in ('candidate', 'distillation_only'):
		summary = tmp_path / role / 'summary_by_budget.csv'
		_write_label_budget(summary, delta=0.1)
		raw['models'][role]['label_budget_summary_csv'] = str(summary)
		manifest = tmp_path / role / 'suite_manifest.json'
		_write_label_budget_manifest(
			manifest,
			role=role,
			identity_prefix='different' if role == 'distillation_only' else 'shared',
		)
		raw['models'][role]['label_budget_suite_manifest_json'] = str(manifest)

	with pytest.raises(ValueError, match='pairing provenance mismatch'):
		summarize_f3_strat_hmm_m1_guardrails(
			f3_guardrail_summary_config_from_mapping(raw),
		)


def test_guardrail_low_budget_requires_pairing_provenance(tmp_path: Path) -> None:
	raw = _summary_config(tmp_path, strict=True)
	for role in F3_STRAT_HMM_M1_GUARDRAIL_ROLES:
		_write_metrics(Path(raw['models'][role]['metrics_json']), offset=0.0)
	path = tmp_path / 'candidate' / 'summary_by_budget.csv'
	_write_label_budget(path, delta=0.1)
	raw['models']['candidate']['label_budget_summary_csv'] = str(path)

	with pytest.raises(ValueError, match='requires a suite manifest'):
		summarize_f3_strat_hmm_m1_guardrails(
			f3_guardrail_summary_config_from_mapping(raw),
		)


@pytest.mark.parametrize('manifest_defect', ['baseline_only', 'wrong_model_tag'])
@pytest.mark.parametrize('affected_role', ['candidate', 'distillation_only'])
def test_guardrail_low_budget_rejects_manifest_without_expected_model_row(
	tmp_path: Path,
	manifest_defect: str,
	affected_role: str,
) -> None:
	raw = _summary_config(tmp_path, strict=True)
	for role in F3_STRAT_HMM_M1_GUARDRAIL_ROLES:
		_write_metrics(Path(raw['models'][role]['metrics_json']), offset=0.0)
	for role in ('candidate', 'distillation_only'):
		summary = tmp_path / role / 'summary_by_budget.csv'
		_write_label_budget(summary, delta=0.1)
		raw['models'][role]['label_budget_summary_csv'] = str(summary)
		manifest = tmp_path / role / 'suite_manifest.json'
		_write_label_budget_manifest(
			manifest,
			role=role,
			include_candidate=not (
				role == affected_role and manifest_defect == 'baseline_only'
			),
			candidate_model_tag=(
				'wrong_model'
				if role == affected_role and manifest_defect == 'wrong_model_tag'
				else None
			),
		)
		raw['models'][role]['label_budget_suite_manifest_json'] = str(manifest)

	with pytest.raises(ValueError, match=r'candidate (?:row|model_tag)'):
		summarize_f3_strat_hmm_m1_guardrails(
			f3_guardrail_summary_config_from_mapping(raw),
		)


def test_guardrail_strict_summary_rejects_missing_primary_metric(
	tmp_path: Path,
) -> None:
	raw = _summary_config(tmp_path, strict=True)
	for role in F3_STRAT_HMM_M1_GUARDRAIL_ROLES:
		_write_metrics(Path(raw['models'][role]['metrics_json']), offset=0.0)
	path = Path(raw['models']['baseline']['metrics_json'])
	payload = json.loads(path.read_text(encoding='utf-8'))
	del payload['macro_f1']
	path.write_text(json.dumps(payload) + '\n', encoding='utf-8')

	with pytest.raises(
		ValueError,
		match='missing required full-budget metric macro_f1',
	):
		summarize_f3_strat_hmm_m1_guardrails(
			f3_guardrail_summary_config_from_mapping(raw),
		)


def test_guardrail_non_strict_summary_marks_missing_required_metric_incomplete(
	tmp_path: Path,
) -> None:
	raw = _summary_config(tmp_path, strict=False)
	for role in F3_STRAT_HMM_M1_GUARDRAIL_ROLES:
		_write_metrics(Path(raw['models'][role]['metrics_json']), offset=0.0)
	path = Path(raw['models']['baseline']['metrics_json'])
	payload = json.loads(path.read_text(encoding='utf-8'))
	del payload['accuracy']
	path.write_text(json.dumps(payload) + '\n', encoding='utf-8')

	result = summarize_f3_strat_hmm_m1_guardrails(
		f3_guardrail_summary_config_from_mapping(raw),
	)
	payload = json.loads(result.summary_json.read_text(encoding='utf-8'))

	assert payload['models'][0]['status'] == 'incomplete'
	assert payload['models'][0]['metrics']['accuracy'] is None
	assert any(
		'missing required full-budget metric accuracy' in warning
		for warning in result.warnings
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
	raw = load_config(ROOT / '13_summarize_guardrails.yaml')
	raw['suite']['strict'] = strict
	for role in F3_STRAT_HMM_M1_GUARDRAIL_ROLES:
		raw['models'][role]['metrics_json'] = str(tmp_path / role / 'metrics.json')
		raw['models'][role].pop('label_budget_summary_csv', None)
		raw['models'][role].pop('label_budget_suite_manifest_json', None)
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
				'per_class_f1': {
					str(class_id): 0.30 + class_id / 100 + offset
					for class_id in range(6)
				},
			},
		)
		+ '\n',
		encoding='utf-8',
	)


def _write_label_budget(
	path: Path,
	*,
	delta: float,
	budget_ids: tuple[str, ...] = ('cap25', 'cap100', 'cap500', 'full'),
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=(
				'budget_id',
				'mean_delta_macro_f1',
				'mean_delta_mean_iou',
				'mean_delta_balanced_accuracy',
			),
		)
		writer.writeheader()
		for budget_id in budget_ids:
			writer.writerow(
				{
					'budget_id': budget_id,
					'mean_delta_macro_f1': delta,
					'mean_delta_mean_iou': delta - 0.01,
					'mean_delta_balanced_accuracy': delta + 0.01,
				},
			)


def _write_label_budget_manifest(  # noqa: PLR0913
	path: Path,
	*,
	role: str,
	identity_prefix: str = 'shared',
	budget_ids: tuple[str, ...] = ('cap25', 'cap100', 'cap500', 'full'),
	include_candidate: bool = True,
	candidate_model_tag: str | None = None,
) -> None:
	model_roles = ('baseline', 'candidate') if include_candidate else ('baseline',)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(
			{
				'artifact_type': 'f3_lithology_label_budget_suite_manifest',
				'rows': [
					{
						'model_role': model_role,
						'model_tag': (
							F3_STRAT_HMM_M1_BASELINE_MODEL_TAG
							if model_role == 'baseline'
							else candidate_model_tag
							or F3_STRAT_HMM_M1_GUARDRAIL_MODEL_TAGS[role]
						),
						'budget_id': budget_id,
						'subsample_seed': 0,
						'paired_identity_hash': f'{identity_prefix}-{budget_id}',
					}
					for budget_id in budget_ids
					for model_role in model_roles
				],
			},
		)
		+ '\n',
		encoding='utf-8',
	)
