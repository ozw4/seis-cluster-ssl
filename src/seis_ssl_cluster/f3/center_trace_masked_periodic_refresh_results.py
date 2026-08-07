"""Publish lightweight review evidence for experiment 107.

The publisher deliberately consumes the complete periodic-refresh handoff and
revalidates every live checkpoint, generation, event, and embedding reference
before writing review files.  It never copies binary or array artifacts into
``results/``.
"""
# ruff: noqa: C901, CPY001, E501, PERF401, PLR0911, PLR0912, TRY300

from __future__ import annotations

import csv
import io
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3 import (
	center_trace_masked_periodic_refresh_validation as validation,
)
from seis_ssl_cluster.stratigraphy.periodic_refresh import (
	load_periodic_refresh_generation,
)
from seis_ssl_cluster.training.checkpoint import load_checkpoint
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	validate_stratigraphy_checkpoint_payload,
)

_ARTIFACT_ROOT_PLACEHOLDER = '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}'
_REVIEW_TYPE = 'f3_center_trace_masked_periodic_refresh_pretraining_review'
_MODEL_TAG = (
	'strat_hmm_pretext_mh_k6810_ctmask010_refresh3ep_hmm2_nocons_'
	'topblock1_distill_v1'
)
_VARIANT = 'ctmask010_refresh3ep_hmm2_nocons'
_SCHEDULE = (2, 5, 8, 11, 14, 17, 20)
_EXPECTED_GENERATIONS = 8

SUMMARY_JSON = 'periodic_refresh_pretraining_summary.json'
SUMMARY_MARKDOWN = 'periodic_refresh_pretraining_summary.md'
REFRESH_EVENTS_CSV = 'periodic_refresh_events.csv'
GENERATION_SUMMARY_CSV = 'periodic_refresh_generation_summary.csv'
CHECKPOINT_SUMMARY_JSON = 'periodic_refresh_checkpoint_summary.json'
PRETRAINING_HANDOFF_JSON = 'periodic_refresh_pretraining_handoff.json'
OUTPUT_NAMES = (
	SUMMARY_JSON,
	SUMMARY_MARKDOWN,
	REFRESH_EVENTS_CSV,
	GENERATION_SUMMARY_CSV,
	CHECKPOINT_SUMMARY_JSON,
	PRETRAINING_HANDOFF_JSON,
)

_EVENT_FIELDS = (
	'event_index',
	'event_type',
	'status',
	'phase',
	'checkpoint_kind',
	'epoch',
	'refresh_epoch',
	'generation_index',
	'generation_id',
	'global_step_before',
	'global_step_after',
	'refresh_phase',
	'source_student_state_sha256',
	'student_state_sha256',
	'optimizer_state_sha256',
	'active_generation_manifest_path',
	'active_generation_manifest_sha256',
	'active_generation_content_sha256',
	'active_target_manifest_path',
	'active_target_manifest_sha256',
	'output_generation_manifest_path',
	'output_generation_manifest_sha256',
	'recovered_from_completed_step',
)
_GENERATION_FIELDS = (
	'generation_index',
	'generation_id',
	'refresh_after_epoch',
	'source_student_state_sha256',
	'manifest_path',
	'manifest_sha256',
	'generation_content_sha256',
	'active_target_manifest_path',
	'active_target_manifest_sha256',
	'k',
	'valid_token_count',
	'iterations',
	'iteration_1_total_center_shift_l2',
	'iteration_2_total_center_shift_l2',
	'iteration_1_center_shift_l2_by_state',
	'iteration_2_center_shift_l2_by_state',
	'final_label_change_count',
	'final_label_change_rate',
	'final_state_counts',
	'boundary_counts',
	'transition_counts',
	'confidence_summary',
	'state_mean_z',
	'ordered_diagnostics',
	'boundary_summary',
)


@dataclass(frozen=True)
class F3CenterTraceMaskedPeriodicRefreshReviewConfig:
	"""Closed source and destination paths for the review publication."""

	artifact_root: Path
	workspace_root: Path
	validation_config: Path
	pretraining_handoff: Path
	output_dir: Path


@dataclass(frozen=True)
class F3CenterTraceMaskedPeriodicRefreshReviewResult:
	"""Review output paths."""

	output_dir: Path
	summary_json: Path
	summary_markdown: Path
	refresh_events: Path
	generation_summary: Path
	checkpoint_summary: Path
	pretraining_handoff: Path


_CONFIG_KEYS = frozenset(
	{
		'artifact_root',
		'workspace_root',
		'validation_config',
		'pretraining_handoff',
		'output_dir',
	}
)


def f3_center_trace_masked_periodic_refresh_review_config_from_mapping(
	config: Mapping[str, object],
) -> F3CenterTraceMaskedPeriodicRefreshReviewConfig:
	"""Resolve the closed review-publication configuration."""
	if not isinstance(config, Mapping):
		raise TypeError('periodic refresh review config must be a mapping')
	unknown, missing = set(config) - _CONFIG_KEYS, _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(
			f'unknown periodic refresh review config keys: {sorted(unknown)!r}'
		)
	if missing:
		raise ValueError(
			f'missing periodic refresh review config keys: {sorted(missing)!r}'
		)

	def path(key: str) -> Path:
		value = config[key]
		if not isinstance(value, str) or not value:
			raise TypeError(f'{key} must be a non-empty path string')
		return Path(value).resolve()

	result = F3CenterTraceMaskedPeriodicRefreshReviewConfig(
		artifact_root=path('artifact_root'),
		workspace_root=path('workspace_root'),
		validation_config=path('validation_config'),
		pretraining_handoff=path('pretraining_handoff'),
		output_dir=path('output_dir'),
	)
	if not result.artifact_root.is_dir() or not result.workspace_root.is_dir():
		raise FileNotFoundError(
			'artifact_root and workspace_root must be existing directories'
		)
	if not result.validation_config.is_file():
		raise FileNotFoundError(
			f'validation_config is missing: {result.validation_config}'
		)
	if not result.pretraining_handoff.is_file():
		raise FileNotFoundError(
			f'pretraining_handoff is missing: {result.pretraining_handoff}'
		)
	return result


def load_f3_center_trace_masked_periodic_refresh_review_config(
	path: str | Path,
) -> F3CenterTraceMaskedPeriodicRefreshReviewConfig:
	"""Load a review configuration through the repository YAML loader."""
	return f3_center_trace_masked_periodic_refresh_review_config_from_mapping(
		load_config(path)
	)


def publish_f3_center_trace_masked_periodic_refresh_review(
	config: F3CenterTraceMaskedPeriodicRefreshReviewConfig,
	*,
	dry_run: bool = False,
	quarantine_invalid: bool = False,
) -> F3CenterTraceMaskedPeriodicRefreshReviewResult:
	"""Revalidate live PASS evidence and publish lightweight review files."""
	handoff = validation.load_f3_center_trace_masked_periodic_refresh_handoff(
		config.pretraining_handoff
	)
	live = _inspect_live_evidence(config, handoff=handoff)
	evidence = _review_evidence(config, handoff=handoff, live=live)
	result = F3CenterTraceMaskedPeriodicRefreshReviewResult(
		output_dir=config.output_dir,
		summary_json=config.output_dir / SUMMARY_JSON,
		summary_markdown=config.output_dir / SUMMARY_MARKDOWN,
		refresh_events=config.output_dir / REFRESH_EVENTS_CSV,
		generation_summary=config.output_dir / GENERATION_SUMMARY_CSV,
		checkpoint_summary=config.output_dir / CHECKPOINT_SUMMARY_JSON,
		pretraining_handoff=config.output_dir / PRETRAINING_HANDOFF_JSON,
	)
	if dry_run:
		return result
	_publish_review(
		config,
		evidence=evidence,
		handoff=handoff,
		live=live,
		quarantine_invalid=quarantine_invalid,
	)
	return result


def _inspect_live_evidence(
	config: F3CenterTraceMaskedPeriodicRefreshReviewConfig,
	*,
	handoff: Mapping[str, object],
) -> dict[str, object]:
	"""Re-run the strict validator's read-only checks and bind the handoff."""
	validation_config = (
		validation.load_f3_center_trace_masked_periodic_refresh_validation_config(
			config.validation_config
		)
	)
	if validation_config.artifact_root != config.artifact_root:
		raise ValueError('review artifact_root differs from validation artifact_root')
	full = validation._training_config(  # noqa: SLF001
		validation_config.periodic_refresh_full_config
	)
	full_root = validation._training_output_root(  # noqa: SLF001
		full, validation_config, 'periodic full output root'
	)
	expected_handoff = full_root / 'preflight' / 'periodic_refresh_handoff.json'
	if config.pretraining_handoff != expected_handoff:
		raise ValueError('review handoff does not bind the periodic full output root')
	inputs = validation._inputs_evidence(validation_config)  # noqa: SLF001
	checkpoints = validation._checkpoint_evidence(  # noqa: SLF001
		validation_config,
		inputs=inputs,
		quarantine_invalid=False,
		dry_run=False,
	)
	embedding = validation._embedding_evidence(  # noqa: SLF001
		validation_config,
		inputs=inputs,
		checkpoint=checkpoints['checkpoint'],
	)
	validation._validate_smoke_phase_evidence(  # noqa: SLF001
		validation_config,
		inputs=inputs,
	)
	execution_path = validation._execution_evidence_path(  # noqa: SLF001
		validation_config
	)
	execution_record = validation._mapping(  # noqa: SLF001
		validation._json(execution_path),  # noqa: SLF001
		'periodic execution evidence',
	)
	if execution_record.get('phase') != 'complete':
		raise ValueError('periodic execution evidence is not complete')
	if execution_record.get('binding') != validation._execution_binding(  # noqa: SLF001
		validation_config
	):
		raise ValueError('periodic execution evidence binding drift')
	if execution_record.get('execution') != handoff['execution']:
		raise ValueError('periodic handoff execution state is stale')
	live_handoff = validation._handoff(  # noqa: SLF001
		{
			**inputs,
			**checkpoints,
			'embedding': embedding,
			'execution': handoff['execution'],
		}
	)
	if live_handoff != handoff:
		raise ValueError('periodic PASS handoff does not match live evidence')
	latest_path = Path(str(checkpoints['checkpoint']['latest_path'])).resolve()
	selected_path = Path(str(checkpoints['checkpoint']['path'])).resolve()
	latest = load_checkpoint(latest_path, map_location='cpu')
	selected = load_checkpoint(selected_path, map_location='cpu')
	selection = _mapping(latest['checkpoint_selection'], 'periodic checkpoint selection')
	validate_stratigraphy_checkpoint_payload(latest, expected_config=full)
	validate_stratigraphy_checkpoint_payload(selected, expected_config=full)
	selection_path = full_root / 'checkpoint_selection_summary.json'
	selection_summary = validation._mapping(  # noqa: SLF001
		validation._json(selection_path),  # noqa: SLF001
		'periodic checkpoint selection summary',
	)
	if selection_summary != selection:
		raise ValueError('periodic checkpoint selection summary is stale')
	events_path = full_root / 'target_refresh_events.jsonl'
	events = _load_events(events_path)
	generation_details = _generation_details(checkpoints['refresh'])
	return {
		'validation_config': validation_config,
		'inputs': inputs,
		'checkpoint': checkpoints['checkpoint'],
		'refresh': checkpoints['refresh'],
		'embedding': embedding,
		'handoff': handoff,
		'latest': latest,
		'selected': selected,
		'selection': selection,
		'selection_path': selection_path,
		'events_path': events_path,
		'events': events,
		'generation_details': generation_details,
		'execution_path': execution_path,
	}


def _review_evidence(
	config: F3CenterTraceMaskedPeriodicRefreshReviewConfig,
	*,
	handoff: Mapping[str, object],
	live: Mapping[str, object],
) -> dict[str, object]:
	"""Build the compact, portable review summary from live evidence."""
	inputs = _mapping(live['inputs'], 'live periodic inputs')
	validation_config = live['validation_config']
	if not isinstance(
		validation_config,
		validation.F3CenterTraceMaskedPeriodicRefreshValidationConfig,
	):
		raise TypeError('live validation config has an invalid type')
	baseline = _mapping(inputs['baseline'], 'periodic baseline evidence')
	checkpoint = _mapping(live['checkpoint'], 'live checkpoint evidence')
	refresh = _mapping(live['refresh'], 'live refresh evidence')
	embedding = _mapping(live['embedding'], 'live embedding evidence')
	selection = _mapping(live['selection'], 'live checkpoint selection')
	selected_event = _mapping(selection['selected'], 'selected periodic event')
	selection_history_sha256 = file_sha256(Path(str(live['selection_path'])))
	generations = _mapping_sequence(
		refresh['generations'], 'periodic generation evidence'
	)
	generation_details = _mapping_sequence(
		live['generation_details'], 'periodic generation details'
	)
	initialization = _mapping(
		inputs['initialization_checkpoints'], 'periodic initialization evidence'
	)
	execution = _mapping(handoff['execution'], 'periodic handoff execution')
	checkpoint_summary = _checkpoint_summary(
		config,
		checkpoint=checkpoint,
		refresh=refresh,
		selected_event=selected_event,
		selection_history_sha256=selection_history_sha256,
	)
	final_checkpoint = {
		'path': checkpoint['path'],
		'sha256': checkpoint['sha256'],
		'latest_path': checkpoint['latest_path'],
		'latest_sha256': checkpoint['latest_sha256'],
		'schema_version': checkpoint['schema_version'],
		'epoch': checkpoint['epoch'],
		'global_step': checkpoint['global_step'],
		'selected_event': selected_event,
		'selection_history_sha256': selection_history_sha256,
	}
	valid_hashes = _mapping(
		inputs['target_manifest'], 'periodic target evidence'
	)['common_valid_token_hashes']
	raw = {
		'artifact_type': _REVIEW_TYPE,
		'schema_version': 1,
		'status': 'PASS',
		'model_tag': _MODEL_TAG,
		'variant': _VARIANT,
		'execution_git_state': execution,
		'fixed_center_trace_parity': {
			'status': 'PASS',
			'allowed_differences': baseline['allowed_differences'],
			'center_trace_masked_config': inputs['baseline']['center_config'],
			'periodic_full_config': inputs['full_config'],
			'center_trace_masked_handoff': validation._reference(  # noqa: SLF001
				validation_config.center_trace_masked_handoff
			),
		},
		'initial_target': {
			'manifest': inputs['target_manifest'],
			'per_head_target_hashes': inputs['target_manifest'][
				'per_head_target_hashes'
			],
			'valid_token_hashes': valid_hashes,
		},
		'initial_hmm_preprocessing': {
			'fixed_preprocessing': inputs['fixed_preprocessing'],
			'initialization_checkpoints': initialization,
		},
		'refresh': {
			'schedule': list(_SCHEDULE),
			'completed_refresh_count': len(generations) - 1,
			'completed_generation_count': len(generations),
			'completed_generations': generation_details,
			'chain': {
				'path': refresh['chain_path'],
				'sha256': refresh['chain_sha256'],
			},
		},
		'optimizer_global_step_continuity': {
			'status': 'PASS',
			'final_global_step': checkpoint['global_step'],
			'expected_global_step': 25600,
			'optimizer_step': checkpoint['optimizer_step'],
			'optimizer_group_identity': checkpoint['optimizer_group_identity'],
			'refresh_global_step_unchanged': True,
		},
		'final_checkpoint': final_checkpoint,
		'checkpoint_summary': checkpoint_summary,
		'final_embedding': embedding,
		'valid_mask_parity': {
			'status': 'PASS',
			'common_valid_token_hashes': valid_hashes,
			'embedding_valid_tokens_sha256': embedding['valid_tokens_sha256'],
			'canonical_common_mask_validated': True,
		},
		'pass_handoff': {
			'path': config.pretraining_handoff,
			'sha256': file_sha256(config.pretraining_handoff),
			'status': handoff['status'],
			'live_revalidated': True,
		},
		'downstream': {
			'original_split_ready': True,
			'decoder_jobs_executed': 0,
			'six_split_jobs_executed': 0,
		},
	}
	return _mapping(
		_portable_value(raw, config=config),
		'portable periodic refresh review evidence',
	)


def render_f3_center_trace_masked_periodic_refresh_review_markdown(
	evidence: Mapping[str, object],
) -> str:
	"""Render compact review prose without raw model or array artifacts."""
	execution = _mapping(evidence['execution_git_state'], 'execution git state')
	refresh = _mapping(evidence['refresh'], 'refresh review')
	checkpoint = _mapping(evidence['final_checkpoint'], 'final checkpoint')
	embedding = _mapping(evidence['final_embedding'], 'final embedding')
	downstream = _mapping(evidence['downstream'], 'downstream review')
	lines = [
		'# F3 periodic center-trace masked pretraining review',
		'',
		f'- Status: `{evidence["status"]}`',
		f'- Model tag: `{evidence["model_tag"]}`',
		f'- Variant: `{evidence["variant"]}`',
		f'- Execution Git SHA: `{_mapping(execution["after"], "execution after")["git_commit"]}`',
		f'- Execution dirty status: `{_mapping(execution["after"], "execution after")["git_status_short"]}`',
		f'- Refreshes/generations: `{refresh["completed_refresh_count"]}` / `{refresh["completed_generation_count"]}`',
		f'- Final checkpoint: epoch `{checkpoint["epoch"]}`, global step `{checkpoint["global_step"]}`',
		f'- Selected checkpoint SHA-256: `{checkpoint["sha256"]}`',
		f'- Final embedding shape/dtype: `{embedding["embeddings_shape"]}` / `{embedding["embeddings_dtype"]}`',
		f'- Valid-token count: `{embedding["finite_valid_count"]}`',
		f'- PASS handoff: `{_mapping(evidence["pass_handoff"], "PASS handoff")["sha256"]}`',
		'',
		'## Fixed center-trace parity',
		'',
		(
			'Parity status is `PASS`; the allowed differences are recorded in the '
			'portable JSON summary. Initial target, HMM, preprocessing, and model '
			'initialization hashes were revalidated from live bytes.'
		),
		'',
		'## Refresh generations',
		'',
		'| Generation | Epoch | Source student hash | Manifest hash |',
		'| --- | ---: | --- | --- |',
	]
	for generation in _mapping_sequence(
		refresh['completed_generations'], 'completed generations'
	):
		lines.append(
			f'| `{generation["generation_id"]}` | {generation["refresh_after_epoch"]} | '
			f'`{generation["source_student_state_sha256"]}` | '
			f'`{generation["manifest_sha256"]}` |'
		)
	lines.extend(
		[
			'',
			'## Downstream status',
			'',
			f'- Original-split ready: `{downstream["original_split_ready"]}`',
			f'- Decoder jobs executed: `{downstream["decoder_jobs_executed"]}`',
			f'- Six-split jobs executed: `{downstream["six_split_jobs_executed"]}`',
		]
	)
	return '\n'.join(lines) + '\n'


def _checkpoint_summary(
	config: F3CenterTraceMaskedPeriodicRefreshReviewConfig,
	*,
	checkpoint: Mapping[str, object],
	refresh: Mapping[str, object],
	selected_event: Mapping[str, object],
	selection_history_sha256: str,
) -> dict[str, object]:
	"""Build the small checkpoint-only evidence file."""
	return _portable_value(
		{
			'artifact_type': 'f3_center_trace_masked_periodic_refresh_checkpoint_summary',
			'schema_version': 1,
			'status': 'PASS',
			'checkpoint': {
				'selected': {
					'path': checkpoint['path'],
					'sha256': checkpoint['sha256'],
				},
				'latest': {
					'path': checkpoint['latest_path'],
					'sha256': checkpoint['latest_sha256'],
				},
				'schema_version': checkpoint['schema_version'],
				'epoch': checkpoint['epoch'],
				'global_step': checkpoint['global_step'],
				'selection_policy': 'final_completed_epoch_v1',
				'selection_history_sha256': selection_history_sha256,
				'selected_event': selected_event,
			},
			'optimizer_global_step_continuity': {
				'status': 'PASS',
				'expected_step': 25600,
				'actual_step': checkpoint['optimizer_step'],
				'groups': checkpoint['optimizer_group_identity'],
				'refresh_global_step_unchanged': True,
			},
			'refresh_chain': {
				'path': refresh['chain_path'],
				'sha256': refresh['chain_sha256'],
				'generation_count': len(_mapping_sequence(
					refresh['generations'], 'refresh generations'
				)),
			},
		},
		config=config,
	)


def _publish_review(
	config: F3CenterTraceMaskedPeriodicRefreshReviewConfig,
	*,
	evidence: Mapping[str, object],
	handoff: Mapping[str, object],
	live: Mapping[str, object],
	quarantine_invalid: bool,
) -> None:
	"""Publish exactly the allowlisted review tree with exact reuse."""
	try:
		_validate_existing_output_dir(config)
	except ValueError:
		if not quarantine_invalid or not (
			config.output_dir.exists() or config.output_dir.is_symlink()
		):
			raise
		_quarantine_tree(config.output_dir)
	contents = _publication_contents(
		config,
		evidence=evidence,
		handoff=handoff,
		live=live,
	)
	if _existing_publication_matches(config, contents=contents):
		return
	if config.output_dir.exists() or config.output_dir.is_symlink():
		if not quarantine_invalid:
			raise ValueError(
				'existing periodic refresh review is stale or partial; pass '
				'--quarantine-invalid to replace it'
			)
		_quarantine_tree(config.output_dir)
	config.output_dir.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(
		tempfile.mkdtemp(
			prefix=f'.{config.output_dir.name}.staging-',
			dir=config.output_dir.parent,
		)
	)
	try:
		for name, content in contents:
			(staging / name).write_text(content, encoding='utf-8')
		staging.replace(config.output_dir)
	except Exception:
		if staging.exists():
			shutil.rmtree(staging, ignore_errors=True)
		raise
	return


def _publication_contents(
	config: F3CenterTraceMaskedPeriodicRefreshReviewConfig,
	*,
	evidence: Mapping[str, object],
	handoff: Mapping[str, object],
	live: Mapping[str, object],
) -> tuple[tuple[str, str], ...]:
	return (
		(SUMMARY_JSON, _json_text(evidence)),
		(
			SUMMARY_MARKDOWN,
			render_f3_center_trace_masked_periodic_refresh_review_markdown(
				evidence
			),
		),
		(
			REFRESH_EVENTS_CSV,
			_events_csv_text(live['events'], config=config),
		),
		(
			GENERATION_SUMMARY_CSV,
			_generation_csv_text(
				live['generation_details'], config=config
			),
		),
		(
			CHECKPOINT_SUMMARY_JSON,
			_json_text(evidence['checkpoint_summary']),
		),
		(
			PRETRAINING_HANDOFF_JSON,
			_json_text(_portable_value(handoff, config=config)),
		),
	)


def _validate_existing_output_dir(
	config: F3CenterTraceMaskedPeriodicRefreshReviewConfig,
) -> None:
	"""Reject foreign or raw files instead of replacing them in place."""
	if not config.output_dir.exists():
		return
	if config.output_dir.is_symlink():
		raise ValueError(
			f'periodic refresh output_dir must not be a symlink: {config.output_dir}'
		)
	if not config.output_dir.is_dir():
		raise ValueError(f'periodic refresh output_dir is not a directory: {config.output_dir}')
	allowed = set(OUTPUT_NAMES)
	for path in config.output_dir.rglob('*'):
		if path.is_symlink():
			raise ValueError(f'periodic refresh output must not contain symlinks: {path}')
		if path.is_dir():
			continue
		if path.relative_to(config.output_dir).as_posix() not in allowed:
			raise ValueError(
				f'periodic refresh output contains unallowlisted file: {path}'
			)


def _existing_publication_matches(
	config: F3CenterTraceMaskedPeriodicRefreshReviewConfig,
	*,
	contents: Sequence[tuple[str, str]],
) -> bool:
	output_dir = config.output_dir
	if not output_dir.is_dir():
		return False
	try:
		return all(
			(output_dir / name).is_file()
			and (output_dir / name).read_text(encoding='utf-8') == content
			for name, content in contents
		)
	except OSError:
		return False


def _load_events(path: Path) -> list[Mapping[str, object]]:
	if not path.is_file():
		raise FileNotFoundError(f'periodic refresh events are missing: {path}')
	events: list[Mapping[str, object]] = []
	for line_number, line in enumerate(
		path.read_text(encoding='utf-8').splitlines(), start=1
	):
		if not line.strip():
			continue
		try:
			value = json.loads(line)
		except json.JSONDecodeError as exc:
			raise ValueError(
				f'periodic refresh event is invalid JSON at line {line_number}'
			) from exc
		events.append(_mapping(value, f'periodic refresh event {line_number}'))
	if not events:
		raise ValueError('periodic refresh events are empty')
	return events


def _generation_details(refresh: Mapping[str, object]) -> list[dict[str, object]]:
	generations = _mapping_sequence(
		refresh['generations'], 'periodic refresh generations'
	)
	if len(generations) != _EXPECTED_GENERATIONS:
		raise ValueError('periodic refresh review requires eight generations')
	details: list[dict[str, object]] = []
	for generation in generations:
		manifest_path = Path(str(generation['manifest_path'])).resolve()
		payload = load_periodic_refresh_generation(manifest_path)
		canonical = _mapping(
			payload['canonical_multi_head_target_manifest'],
			'generation target manifest',
		)
		refresh_diagnostics = payload.get('refresh_diagnostics')
		per_k: dict[str, object] = {}
		if refresh_diagnostics is not None:
			ref = _mapping(refresh_diagnostics, 'generation refresh diagnostics')
			diagnostics = _mapping(
				json.loads(Path(str(ref['path'])).read_text(encoding='utf-8')),
				'generation refresh diagnostics',
			)
			per_k = {
				str(k): _public_diagnostics(
					_mapping(
						_mapping(diagnostics['per_k'], 'generation diagnostics')[str(k)],
						f'generation diagnostics K={k}',
					)
				)
				for k in (6, 8, 10)
			}
		details.append(
			{
				'generation_index': generation['generation_index'],
				'generation_id': generation['generation_id'],
				'refresh_after_epoch': payload['refresh_after_epoch'],
				'source_student_state_sha256': payload.get(
					'source_student_state_sha256'
				),
				'manifest_path': generation['manifest_path'],
				'manifest_sha256': generation['manifest_sha256'],
				'generation_content_sha256': generation['generation_content_sha256'],
				'active_target_manifest_path': canonical['path'],
				'active_target_manifest_sha256': canonical['sha256'],
				'per_k': per_k,
			}
		)
	return details


def _public_diagnostics(diagnostics: Mapping[str, object]) -> dict[str, object]:
	iterations = _mapping_sequence(diagnostics['iterations'], 'HMM iterations')
	return {
		'iterations': len(iterations),
		'center_shift_l2_by_iteration': [
			_mapping(item, 'HMM iteration')['total_center_shift_l2']
			for item in iterations
		],
		'center_shift_l2_by_state_by_iteration': [
			_mapping(item, 'HMM iteration')['center_shift_l2_by_state']
			for item in iterations
		],
		'final_label_change_count': diagnostics['final_label_change_count'],
		'final_label_change_rate': diagnostics['final_label_change_rate'],
		'final_state_counts': diagnostics['final_state_counts'],
		'boundary_counts': diagnostics['boundary_counts'],
		'confidence_summary': diagnostics['confidence_summary'],
		'state_mean_z': diagnostics['state_mean_z'],
		'valid_token_count': diagnostics['valid_token_count'],
		'ordered_diagnostics': diagnostics['ordered_diagnostics'],
		'boundary_summary': diagnostics['boundary_summary'],
	}


def _events_csv_text(
	events: object,
	*,
	config: F3CenterTraceMaskedPeriodicRefreshReviewConfig,
) -> str:
	rows = _mapping_sequence(events, 'periodic event rows')
	output = io.StringIO(newline='')
	writer = csv.DictWriter(output, fieldnames=_EVENT_FIELDS, lineterminator='\n')
	writer.writeheader()
	for index, event in enumerate(rows):
		row: dict[str, object] = {}
		for field in _EVENT_FIELDS:
			value = index if field == 'event_index' else event.get(field)
			row[field] = _csv_value(value, config=config)
		writer.writerow(row)
	return output.getvalue()


def _generation_csv_text(
	details: object,
	*,
	config: F3CenterTraceMaskedPeriodicRefreshReviewConfig,
) -> str:
	rows = _mapping_sequence(details, 'periodic generation rows')
	output = io.StringIO(newline='')
	writer = csv.DictWriter(output, fieldnames=_GENERATION_FIELDS, lineterminator='\n')
	writer.writeheader()
	for generation in rows:
		per_k = _mapping(generation['per_k'], 'generation per-K diagnostics')
		for k in (6, 8, 10):
			diagnostics = _mapping(per_k.get(str(k), {}), f'generation K={k}')
			shifts = diagnostics.get('center_shift_l2_by_iteration', [])
			shift_by_state = diagnostics.get(
				'center_shift_l2_by_state_by_iteration', []
			)
			row = {
				key: generation.get(key) for key in _GENERATION_FIELDS
			}
			row.update(
				{
					'k': k,
					'valid_token_count': diagnostics.get('valid_token_count'),
					'iterations': diagnostics.get('iterations'),
					'iteration_1_total_center_shift_l2': _sequence_value(shifts, 0),
					'iteration_2_total_center_shift_l2': _sequence_value(shifts, 1),
					'iteration_1_center_shift_l2_by_state': _sequence_value(
						shift_by_state, 0
					),
					'iteration_2_center_shift_l2_by_state': _sequence_value(
						shift_by_state, 1
					),
					'final_label_change_count': diagnostics.get(
						'final_label_change_count'
					),
					'final_label_change_rate': diagnostics.get('final_label_change_rate'),
					'final_state_counts': diagnostics.get('final_state_counts'),
					'boundary_counts': diagnostics.get('boundary_counts'),
					'transition_counts': diagnostics.get('transition_counts'),
					'confidence_summary': diagnostics.get('confidence_summary'),
					'state_mean_z': diagnostics.get('state_mean_z'),
					'ordered_diagnostics': diagnostics.get('ordered_diagnostics'),
					'boundary_summary': diagnostics.get('boundary_summary'),
				}
			)
			writer.writerow(
				{
					key: _csv_value(row.get(key), config=config)
					for key in _GENERATION_FIELDS
				}
			)
	return output.getvalue()


def _sequence_value(value: object, index: int) -> object:
	if isinstance(value, Sequence) and not isinstance(value, str) and len(value) > index:
		return value[index]
	return None


def _csv_value(
	value: object,
	*,
	config: F3CenterTraceMaskedPeriodicRefreshReviewConfig,
) -> str | int | float | None:
	portable = _portable_value(value, config=config)
	if portable is None or isinstance(portable, int | float | str):
		return portable
	return json.dumps(portable, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _portable_value(
	value: object,
	*,
	config: F3CenterTraceMaskedPeriodicRefreshReviewConfig,
) -> object:
	if isinstance(value, Mapping):
		return {
			str(key): _portable_value(item, config=config)
			for key, item in value.items()
		}
	if isinstance(value, list | tuple):
		return [_portable_value(item, config=config) for item in value]
	if isinstance(value, Path):
		return _portable_path(str(value), config=config)
	if isinstance(value, str):
		return _portable_path(value, config=config)
	return value


def _portable_path(
	value: str,
	*,
	config: F3CenterTraceMaskedPeriodicRefreshReviewConfig,
) -> str:
	artifact = str(config.artifact_root.resolve())
	workspace = str(config.workspace_root.resolve())
	if value == artifact:
		return _ARTIFACT_ROOT_PLACEHOLDER
	if value.startswith(f'{artifact}/'):
		return f'{_ARTIFACT_ROOT_PLACEHOLDER}{value[len(artifact):]}'
	if value == workspace:
		return '.'
	if value.startswith(f'{workspace}/'):
		return value[len(workspace) + 1 :]
	return value


def _json_text(value: object) -> str:
	return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'


def _quarantine_tree(path: Path) -> Path:
	timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
	target = path.with_name(f'{path.name}.quarantine.{timestamp}')
	if target.exists():
		raise FileExistsError(f'quarantine path already exists: {target}')
	path.replace(target)
	return target


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _mapping_sequence(value: object, label: str) -> list[Mapping[str, object]]:
	if not isinstance(value, list):
		raise TypeError(f'{label} must be a list')
	return [_mapping(item, label) for item in value]


__all__ = [
	'CHECKPOINT_SUMMARY_JSON',
	'GENERATION_SUMMARY_CSV',
	'OUTPUT_NAMES',
	'PRETRAINING_HANDOFF_JSON',
	'REFRESH_EVENTS_CSV',
	'SUMMARY_JSON',
	'SUMMARY_MARKDOWN',
	'F3CenterTraceMaskedPeriodicRefreshReviewConfig',
	'F3CenterTraceMaskedPeriodicRefreshReviewResult',
	'f3_center_trace_masked_periodic_refresh_review_config_from_mapping',
	'load_f3_center_trace_masked_periodic_refresh_review_config',
	'publish_f3_center_trace_masked_periodic_refresh_review',
	'render_f3_center_trace_masked_periodic_refresh_review_markdown',
]
