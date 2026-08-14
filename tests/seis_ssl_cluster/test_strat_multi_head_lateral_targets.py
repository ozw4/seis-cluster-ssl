"""Focused fixed-policy configuration checks for lateral hard-target export."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from joblib import dump
from sklearn.preprocessing import FunctionTransformer

from proc.seis_ssl_cluster import (
	export_strat_hmm_multi_head_lateral_targets as lateral_target_export_cli,
)
from proc.seis_ssl_cluster.export_strat_hmm_multi_head_lateral_targets import (
	build_parser,
)
from seis_ssl_cluster.clustering.features import EmbeddingInput, file_sha256
from seis_ssl_cluster.stratigraphy import lateral_targets, state_posterior
from seis_ssl_cluster.stratigraphy.lateral_targets import (
	MultiHeadLateralTargetExportConfig,
	MultiHeadLateralTargetExportPlan,
	export_multi_head_lateral_targets,
	resolve_multi_head_lateral_target_export_config,
)
from seis_ssl_cluster.stratigraphy.multi_head import (
	build_multi_head_target_manifest,
	validate_multi_head_target_reference,
)
from seis_ssl_cluster.stratigraphy.state_posterior import (
	MultiHeadStatePosteriorExportConfig,
	export_multi_head_state_posteriors,
)
from seis_ssl_cluster.stratigraphy.targets import write_pseudo_target


def test_lateral_target_config_requires_explicit_positive_beta(tmp_path) -> None:
	"""M5-LS exports must never silently select a no-op pairwise strength."""
	paths = {}
	for name in ('hard', 'posterior', 'clustering'):
		path = tmp_path / f'{name}.json'
		path.write_text('{}', encoding='utf-8')
		paths[name] = path
	config = {
		'source_hard_manifest': str(paths['hard']),
		'source_posterior_manifest': str(paths['posterior']),
		'clustering_output_dir': str(tmp_path / 'clustering-output'),
		'clustering_config': str(paths['clustering']),
		'source_embedding_dir': str(tmp_path / 'embeddings'),
		'output_root': str(tmp_path / 'output'),
		'smoothing': {'pairwise_strength_ratio': 0.0},
		'outputs': {'overwrite': False},
	}
	with pytest.raises(ValueError, match='positive and finite'):
		resolve_multi_head_lateral_target_export_config(config)
	config['smoothing'] = {'pairwise_strength_ratio': 0.25}
	resolved = resolve_multi_head_lateral_target_export_config(config)
	assert resolved.pairwise_strength_ratio == 0.25


def test_lateral_target_entrypoint_help_exposes_resume_controls() -> None:
	"""The thin export entrypoint remains directly invocable by automation."""
	help_text = build_parser().format_help()
	assert '--dry-run' in help_text
	assert '--only-missing' in help_text


def test_lateral_target_entrypoint_dry_run_reports_resolved_scales(
	tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""The dry-run report exposes the same scales returned by the planner."""
	config_path = tmp_path / 'lateral-targets.yaml'
	plans = [
		MultiHeadLateralTargetExportPlan(
			k, 'NEW', affinity_scale=0.5, emission_gap_scale=float(k)
		)
		for k in (6, 8, 10)
	]
	monkeypatch.setattr(lateral_target_export_cli, 'load_config', lambda _path: {})
	monkeypatch.setattr(
		lateral_target_export_cli,
		'resolve_multi_head_lateral_target_export_config',
		lambda _config: object(),
	)
	monkeypatch.setattr(
		lateral_target_export_cli,
		'export_multi_head_lateral_targets',
		lambda *_args, **_kwargs: plans,
	)
	assert (
		lateral_target_export_cli.main(['--config', str(config_path), '--dry-run'])
		== 0
	)
	report = capsys.readouterr().out
	assert 'resolved affinity scale: 0.5' in report
	for k in (6, 8, 10):
		assert f'k={k} resolved emission-gap scale: {k}' in report


def test_lateral_target_entrypoint_dry_run_reports_source_error_without_scales(
	tmp_path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	"""An unresolved ERROR plan reports its cause instead of requiring scales."""
	config_path = tmp_path / 'lateral-targets.yaml'
	reason = 'hard and posterior valid-mask identities differ for k=6 survey'
	plans = [
		MultiHeadLateralTargetExportPlan(k, 'ERROR', reason)
		for k in (6, 8, 10)
	]
	monkeypatch.setattr(lateral_target_export_cli, 'load_config', lambda _path: {})
	monkeypatch.setattr(
		lateral_target_export_cli,
		'resolve_multi_head_lateral_target_export_config',
		lambda _config: object(),
	)
	monkeypatch.setattr(
		lateral_target_export_cli,
		'export_multi_head_lateral_targets',
		lambda *_args, **_kwargs: plans,
	)

	assert (
		lateral_target_export_cli.main(['--config', str(config_path), '--dry-run'])
		== 0
	)
	report = capsys.readouterr().out
	for k in (6, 8, 10):
		action = f'k={k} planned action: ERROR'
		detail = f'k={k} detail: {reason}'
		assert action in report
		assert detail in report
		assert report.index(action) < report.index(detail)
	assert 'resolved affinity scale' not in report
	assert 'resolved emission-gap scale' not in report


def test_lateral_sources_require_hard_manifest_frozen_model_identity(
	tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""A matching posterior identity cannot substitute for hard-source binding."""
	hard_path = tmp_path / 'hard.json'
	posterior_path = tmp_path / 'posterior.json'
	hard_path.write_text('{}', encoding='utf-8')
	posterior_path.write_text('{}', encoding='utf-8')
	source_embedding = {'input_dir': str(tmp_path / 'embeddings')}
	hard = {
		'head_ks': [6, 8, 10],
		'source_embedding': source_embedding,
		'heads': {str(k): {'surveys': {}} for k in (6, 8, 10)},
	}
	posterior = {
		'source_hard_manifest': {
			'path': str(hard_path),
			'sha256': file_sha256(hard_path),
		},
		'source_embedding': source_embedding,
		'heads': {
			str(k): {'model': {'model': 'selected'}, 'surveys': {}}
			for k in (6, 8, 10)
		},
	}
	config = MultiHeadLateralTargetExportConfig(
		hard_path,
		posterior_path,
		tmp_path / 'clustering',
		tmp_path / 'clustering.yaml',
		tmp_path / 'embeddings',
		tmp_path / 'output',
		0.25,
		tmp_path / 'output' / 'handoff.json',
	)
	monkeypatch.setattr(
		lateral_targets, 'load_multi_head_target_manifest', lambda _: hard
	)
	monkeypatch.setattr(lateral_targets.json, 'loads', lambda _: posterior)
	monkeypatch.setattr(
		lateral_targets,
		'validate_multi_head_state_posterior_manifest',
		lambda _payload, *, validate_array_semantics: (
			None
			if validate_array_semantics
			else pytest.fail(
				'lateral source validation requested reference-only posterior'
			)
		),
	)
	monkeypatch.setattr(
		lateral_targets, '_validate_source_embedding_identity', lambda *_: None
	)
	monkeypatch.setattr(lateral_targets, 'discover_embedding_inputs', lambda _: [])
	monkeypatch.setattr(
		lateral_targets,
		'hard_source_model_identities',
		lambda _: (
			tmp_path / 'handoff.json',
			{str(k): {'model': 'hard'} for k in (6, 8, 10)},
		),
	)
	monkeypatch.setattr(
		lateral_targets,
		'load_frozen_hmm_model',
		lambda **_: {'identity': {'model': 'selected'}},
	)
	with pytest.raises(ValueError, match='differs from hard manifest for k=6'):
		lateral_targets._validate_sources(config)  # noqa: SLF001


def test_lateral_export_publishes_one_complete_bundle_before_handoff(
	tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""No individual head becomes visible before the complete bundle handoff."""
	config = MultiHeadLateralTargetExportConfig(
		tmp_path / 'hard.json',
		tmp_path / 'posterior.json',
		tmp_path / 'cluster',
		tmp_path / 'cluster.json',
		tmp_path / 'embeddings',
		tmp_path / 'output',
		0.25,
		tmp_path / 'output' / 'handoff.json',
	)
	plans = [MultiHeadLateralTargetExportPlan(k, 'NEW') for k in (6, 8, 10)]
	monkeypatch.setattr(
		lateral_targets,
		'plan_multi_head_lateral_target_exports',
		lambda *_args, **_kwargs: plans,
	)
	monkeypatch.setattr(
		lateral_targets,
		'_validate_sources',
		lambda _config: ({}, {}, {}, {6: {}, 8: {}, 10: {}}),
	)
	monkeypatch.setattr(
		lateral_targets, '_validate_frozen_source_replay', lambda *_: None
	)
	monkeypatch.setattr(lateral_targets, '_source_snapshot', lambda *_: '{}')
	monkeypatch.setattr(lateral_targets, '_validate_live_snapshot', lambda *_: None)
	monkeypatch.setattr(
		lateral_targets,
		'_affinity_scale',
		lambda *_args: (
			1.0,
			{'quantiles': {'p25': 0.25, 'p50': 0.5, 'p75': 0.75}},
		),
	)
	monkeypatch.setattr(
		lateral_targets,
		'_emission_gap_scales',
		lambda *_args: ({6: 1.0, 8: 1.0, 10: 1.0}, {6: {}, 8: {}, 10: {}}),
	)
	staging_validation: list[bool] = []

	def validate_complete_head(*_args, allow_staging: bool = False, **_kwargs) -> None:
		staging_validation.append(allow_staging)

	monkeypatch.setattr(
		lateral_targets, '_validate_complete_head', validate_complete_head
	)
	monkeypatch.setattr(
		lateral_targets, '_rebase_head_metadata', lambda *_args, **_kwargs: None
	)

	def export_head(root, _k, *_args):
		root.mkdir(parents=True)
		(root / 'head_metadata.json').write_text('{}', encoding='utf-8')

	monkeypatch.setattr(lateral_targets, '_export_head', export_head)

	publish_count = 0

	def publish(_config, _source):
		nonlocal publish_count
		publish_count += 1
		assert (config.output_root / 'bundle').is_dir()
		assert all(
			(config.output_root / 'bundle' / f'k{k}').is_dir() for k in (6, 8, 10)
		)
		assert not any((config.output_root / f'k{k}').exists() for k in (6, 8, 10))

	monkeypatch.setattr(lateral_targets, '_publish_manifest', publish)
	export_multi_head_lateral_targets(config)
	assert staging_validation == [True, True, True, False, False, False]
	assert publish_count == 1


def test_dry_run_rejects_frozen_replay_mismatch_before_planning(
	tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Dry-run must validate Viterbi replay without writing any output."""
	config = MultiHeadLateralTargetExportConfig(
		tmp_path / 'hard.json',
		tmp_path / 'posterior.json',
		tmp_path / 'cluster',
		tmp_path / 'cluster.json',
		tmp_path / 'embeddings',
		tmp_path / 'output',
		0.25,
		tmp_path / 'output' / 'handoff.json',
	)
	monkeypatch.setattr(
		lateral_targets,
		'_validate_sources',
		lambda _config: ({}, {}, {}, {6: {}, 8: {}, 10: {}}),
	)

	def reject_replay(*_args) -> None:
		raise ValueError('Viterbi replay differs from frozen hard labels: k=6 survey')

	monkeypatch.setattr(
		lateral_targets, '_validate_frozen_source_replay', reject_replay
	)
	with pytest.raises(ValueError, match='Viterbi replay differs'):
		export_multi_head_lateral_targets(config, dry_run=True)
	assert not config.output_root.exists()


def test_dry_run_reports_preflight_scales_without_creating_output(tmp_path) -> None:
	"""Both public planning paths expose the scales from their full preflight."""
	config = _real_lateral_export_fixture(tmp_path)
	planned = lateral_targets.plan_multi_head_lateral_target_exports(
		config, only_missing=True
	)
	dry_run = export_multi_head_lateral_targets(
		config, dry_run=True, only_missing=True
	)
	assert dry_run == planned
	assert all(
		plan.affinity_scale is not None and plan.affinity_scale > 0
		for plan in dry_run
	)
	assert all(
		plan.emission_gap_scale is not None and plan.emission_gap_scale > 0
		for plan in dry_run
	)
	assert not config.output_root.exists()


def test_dry_run_rejects_invalid_posterior_before_publication_mutation(
	tmp_path,
) -> None:
	"""Full preflight rejects self-consistent NaN posterior source semantics."""
	config = _real_lateral_export_fixture(tmp_path)
	export_multi_head_lateral_targets(config)
	posterior_manifest = json.loads(
		config.source_posterior_manifest.read_text(encoding='utf-8')
	)
	entry = posterior_manifest['heads']['6']['surveys']['survey']
	posterior_path = Path(entry['posterior']['path'])
	posterior = np.load(posterior_path)
	posterior[0, 0, 0, 0] = np.nan
	np.save(posterior_path, posterior, allow_pickle=False)
	entry['posterior']['sha256'] = file_sha256(posterior_path)
	config.source_posterior_manifest.write_text(
		json.dumps(posterior_manifest, sort_keys=True) + '\n',
		encoding='utf-8',
	)
	before = _public_file_snapshot(config)

	with pytest.raises(ValueError, match='posterior must be finite and non-negative'):
		export_multi_head_lateral_targets(
			config,
			dry_run=True,
			only_missing=True,
		)

	assert _public_file_snapshot(config) == before


@pytest.mark.parametrize(
	('invalid_scale', 'message'),
	[
		('zero_norm', 'zero or non-finite norm'),
		('no_xy_edges', 'affinity scale requires at least one value'),
		('non_finite_gap', 'emission gap is invalid'),
		('negative_gap', 'emission gap is invalid'),
	],
)
def test_dry_run_rejects_invalid_scale_inputs_before_output_mutation(
	tmp_path,
	monkeypatch: pytest.MonkeyPatch,
	invalid_scale: str,
	message: str,
) -> None:
	"""Every resolved-scale blocker fails in preflight with no owned output."""
	config = _real_lateral_export_fixture(tmp_path)
	source, posterior, inputs, models = lateral_targets._validate_sources(  # noqa: SLF001
		config
	)
	if invalid_scale == 'zero_norm':
		embeddings_path = config.source_embedding_dir / 'survey.embeddings.npy'
		embeddings = np.load(embeddings_path)
		embeddings[0, 0] = 0.0
		np.save(embeddings_path, embeddings, allow_pickle=False)
	elif invalid_scale == 'no_xy_edges':
		valid_path = tmp_path / 'no_xy_edges.valid_tokens.npy'
		np.save(valid_path, np.zeros((2, 1, 10), dtype=bool), allow_pickle=False)
		valid_reference = {
			'path': str(valid_path),
			'sha256': file_sha256(valid_path),
		}
		source = json.loads(json.dumps(source))
		posterior = json.loads(json.dumps(posterior))
		for payload in (source, posterior):
			payload['heads']['6']['surveys']['survey']['valid_tokens'] = valid_reference
	elif invalid_scale == 'non_finite_gap':
		monkeypatch.setattr(
			lateral_targets,
			'replay_frozen_hmm_trace',
			lambda _embedding, indices, _model, *, k: (
				np.full((indices.size, k), np.nan),
				np.zeros(indices.size, dtype=np.int32),
			),
		)
	else:
		def negative_gap_partition(
			values: np.ndarray, _index: int, *, axis: int
		) -> np.ndarray:
			del axis
			return np.column_stack(
				(
					np.ones(values.shape[0]),
					np.zeros(values.shape[0]),
					values[:, 2:],
				)
			)

		monkeypatch.setattr(
			lateral_targets.np,
			'partition',
			negative_gap_partition,
		)
	monkeypatch.setattr(
		lateral_targets,
		'_validate_sources',
		lambda _config: (source, posterior, inputs, models),
	)
	monkeypatch.setattr(
		lateral_targets, '_validate_frozen_source_replay', lambda *_args: None
	)
	with pytest.raises(ValueError, match=message):
		export_multi_head_lateral_targets(config, dry_run=True)
	assert not config.output_root.exists()


def test_affinity_disagreement_bins_use_measured_quartiles() -> None:
	"""Diagnostic bin boundaries are derived from the global affinity samples."""
	boundaries = lateral_targets._affinity_quartile_boundaries(  # noqa: SLF001
		{'quantiles': {'p25': 0.1, 'p50': 0.3, 'p75': 0.7}}, 1.0
	)
	assert boundaries == pytest.approx((np.exp(-0.7), np.exp(-0.3), np.exp(-0.1)))
	diagnostics = lateral_targets._LateralDiagnostics(  # noqa: SLF001
		2, 1.0, (0.3, 0.5, 0.7)
	)
	features = np.array(
		[
			[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
			[
				[
					[
						1.0 + np.log(0.2),
						np.sqrt(1.0 - (1.0 + np.log(0.2)) ** 2),
					],
					[
						1.0 + np.log(0.4),
						np.sqrt(1.0 - (1.0 + np.log(0.4)) ** 2),
					],
					[
						1.0 + np.log(0.6),
						np.sqrt(1.0 - (1.0 + np.log(0.6)) ** 2),
					],
					[
						1.0 + np.log(0.8),
						np.sqrt(1.0 - (1.0 + np.log(0.8)) ** 2),
					],
				]
			],
		],
		dtype=np.float64,
	)
	labels = np.array([[[0, 0, 1, 1]], [[0, 1, 0, 1]]], dtype=np.int32)
	valid = np.ones(labels.shape, dtype=bool)
	diagnostics.add_xy_edges(labels, labels, valid, features, 1.0)
	quartiles = diagnostics.finish()['xy_edge_disagreement']['affinity_quartiles']
	assert [item['edge_count'] for item in quartiles] == [1, 1, 1, 1]
	assert [item['affinity_range'] for item in quartiles] == [
		[0.0, 0.3],
		[0.3, 0.5],
		[0.5, 0.7],
		[0.7, 1.0],
	]


def test_affinity_scale_excludes_embedding_only_valid_tokens(tmp_path) -> None:
	"""Fixed affinity scale uses the hard/posterior common mask, not embedding valid."""
	features = np.array(
		[
			[[[1.0, 0.0], [1.0, 0.0]]],
			[[[0.0, 1.0], [-1.0, 0.0]]],
		],
		dtype=np.float32,
	)
	embeddings = tmp_path / 'survey.embeddings.npy'
	embedding_valid = tmp_path / 'survey.embedding-valid.npy'
	hard_valid = tmp_path / 'survey.hard-valid.npy'
	posterior_valid = tmp_path / 'survey.posterior-valid.npy'
	metadata = tmp_path / 'survey.metadata.json'
	np.save(embeddings, features, allow_pickle=False)
	np.save(embedding_valid, np.ones((2, 1, 2), dtype=bool), allow_pickle=False)
	common_valid = np.array([[[True, False]], [[True, False]]])
	np.save(hard_valid, common_valid, allow_pickle=False)
	np.save(posterior_valid, common_valid, allow_pickle=False)
	metadata.write_text('{}', encoding='utf-8')

	def ref(path) -> dict[str, str]:
		return {'path': str(path), 'sha256': lateral_targets.file_sha256(path)}

	source = {
		'heads': {'6': {'surveys': {'survey': {'valid_tokens': ref(hard_valid)}}}}
	}
	posterior = {
		'heads': {'6': {'surveys': {'survey': {'valid_tokens': ref(posterior_valid)}}}}
	}
	inputs = {
		'survey': EmbeddingInput('survey', embeddings, embedding_valid, metadata)
	}
	scale, stats = lateral_targets._affinity_scale(  # noqa: SLF001
		source, posterior, inputs
	)
	assert scale == pytest.approx(1.0)
	assert stats['sample_count'] == 1


@pytest.mark.parametrize(
	('labels', 'message'),
	[
		(
			np.array([[[0, 1, 0]]], dtype=np.int32),
			'lateral labels violate ordered paths',
		),
		(
			np.array([[[0, 0, 0]]], dtype=np.int32),
			'lateral labels contain an empty state',
		),
	],
)
def test_reusable_head_arrays_reject_ordered_and_occupancy_drift(
	labels: np.ndarray, message: str
) -> None:
	"""Planner validation must quarantine semantically invalid owned outputs."""
	with pytest.raises(ValueError, match=message):
		lateral_targets._validate_complete_head_arrays(  # noqa: SLF001
			labels,
			np.ones(labels.shape, dtype=np.float32),
			np.ones(labels.shape, dtype=bool),
			k=2,
		)


@pytest.mark.parametrize(
	('head', 'message'),
	[
		(
			{
				'model': {},
				'surveys': {},
			},
			'manifest keys mismatch',
		),
		(
			{
				'model': {},
				'surveys': {},
				'diagnostics': {},
				'unexpected': {},
			},
			'manifest keys mismatch',
		),
		(
			{
				'model': {},
				'surveys': {'survey': {}},
				'diagnostics': {},
			},
			'manifest keys mismatch',
		),
		(
			{
				'model': {},
				'surveys': {'survey': {'unexpected': {}}},
				'diagnostics': {},
			},
			'manifest keys mismatch',
		),
	],
)
def test_reusable_head_rejects_unknown_owned_metadata_keys(
	tmp_path,
	monkeypatch: pytest.MonkeyPatch,
	head: dict[str, object],
	message: str,
) -> None:
	"""Malformed owned metadata must be quarantined instead of reused."""
	head_path = tmp_path / 'k6'
	head_path.mkdir()
	(head_path / 'head_metadata.json').write_text(json.dumps(head), encoding='utf-8')
	config = MultiHeadLateralTargetExportConfig(
		tmp_path / 'hard.json',
		tmp_path / 'posterior.json',
		tmp_path / 'cluster',
		tmp_path / 'cluster.json',
		tmp_path / 'embeddings',
		tmp_path / 'output',
		0.25,
		tmp_path / 'output' / 'handoff.json',
	)
	source = {'heads': {'6': {'surveys': {'survey': {}}}}}
	posterior = {'heads': {'6': {'surveys': {'survey': {}}}}}
	monkeypatch.setattr(
		lateral_targets,
		'_validate_lateral_diagnostics',
		lambda *_args, **_kwargs: None,
	)
	monkeypatch.setattr(lateral_targets, '_resolved_scales', lambda _value: {})
	with pytest.raises(ValueError, match=message):
		lateral_targets._validate_complete_head(  # noqa: SLF001
			head_path,
			6,
			source,
			posterior,
			{6: {'identity': {}}},
			config,
		)


def test_target_valid_mask_must_match_hard_and_posterior_sources(tmp_path) -> None:
	"""A consistently altered lateral mask cannot be reused as a complete target."""
	hard_path = tmp_path / 'hard-valid.npy'
	posterior_path = tmp_path / 'posterior-valid.npy'
	np.save(hard_path, np.array([[[True, True]]]), allow_pickle=False)
	np.save(posterior_path, np.array([[[True, True]]]), allow_pickle=False)

	def ref(path) -> dict[str, str]:
		return {'path': str(path), 'sha256': lateral_targets.file_sha256(path)}

	with pytest.raises(ValueError, match='valid mask differs from source masks'):
		lateral_targets._validate_target_valid_tokens(  # noqa: SLF001
			np.array([[[True, False]]]),
			{'valid_tokens': ref(hard_path)},
			{'valid_tokens': ref(posterior_path)},
			context='k=6 survey',
		)


def test_array_reference_requires_matching_shape_and_dtype(tmp_path) -> None:
	"""Published array descriptors are part of the fail-closed contract."""
	path = tmp_path / 'labels.npy'
	array = np.zeros((1, 2, 3), dtype=np.int32)
	np.save(path, array, allow_pickle=False)
	reference = {
		'path': str(path),
		'sha256': lateral_targets.file_sha256(path),
		'shape': [1, 2, 3],
		'dtype': 'int32',
	}
	lateral_targets._validate_array_reference(  # noqa: SLF001
		reference, array, name='labels'
	)
	for field, value in (('shape', [1, 2, 4]), ('dtype', 'float32')):
		altered = {**reference, field: value}
		with pytest.raises(ValueError, match=field):
			lateral_targets._validate_array_reference(  # noqa: SLF001
				altered, array, name='labels'
			)
	missing = {key: value for key, value in reference.items() if key != 'shape'}
	with pytest.raises(ValueError, match='manifest keys mismatch'):
		lateral_targets._validate_array_reference(  # noqa: SLF001
			missing, array, name='labels'
		)


def test_lateral_metadata_requires_exact_source_and_smoothing_identity(
	tmp_path,
) -> None:
	"""Metadata cannot drift independently of its manifest entry and head scales."""
	metadata_path = tmp_path / 'survey.metadata.json'
	labels_path = tmp_path / 'labels.npy'
	valid_path = tmp_path / 'valid.npy'
	labels = np.array([[[0, 1]]], dtype=np.int32)
	valid = np.ones(labels.shape, dtype=bool)
	np.save(labels_path, labels, allow_pickle=False)
	np.save(valid_path, valid, allow_pickle=False)
	entry = {
		'metadata': {},
		'labels': {
			'path': str(labels_path),
			'sha256': lateral_targets.file_sha256(labels_path),
		},
		'valid_tokens': {
			'path': str(valid_path),
			'sha256': lateral_targets.file_sha256(valid_path),
		},
		'source_hard_labels': {'path': 'hard.npy', 'sha256': 'hard'},
		'source_posterior': {'path': 'posterior.npy', 'sha256': 'posterior'},
	}
	identity = {
		'source_embedding': {'embedding_path': 'embedding.npy', 'sha256': 'embed'},
		'smoothing': lateral_targets._smoothing_identity(0.25),  # noqa: SLF001
		'resolved_scales': {'affinity_scale': 0.5, 'emission_gap_scale': 1.5},
	}
	metadata = lateral_targets.build_pseudo_target_metadata(
		labels=labels,
		valid_tokens=valid,
		boundary_weight=valid.astype(np.float32),
		boundary_weight_source='default_unity',
		k=6,
		survey_id='survey',
		schema_version=1,
		write_boundary_weight=False,
		source_metadata={
			'target_semantics': lateral_targets.LATERAL_SMOOTHING_SEMANTICS,
			'source_label_path': 'hard.npy',
			'source_label_sha256': 'hard',
			'source_hard_labels': entry['source_hard_labels'],
			'source_posterior': entry['source_posterior'],
			'source_embedding': identity['source_embedding'],
			'smoothing': {**identity['smoothing'], **identity['resolved_scales']},
		},
	)

	def validate(payload: dict[str, object]) -> None:
		metadata_path.write_text(json.dumps(payload), encoding='utf-8')
		entry['metadata'] = {
			'path': str(metadata_path),
			'sha256': lateral_targets.file_sha256(metadata_path),
		}
		lateral_targets._validate_survey_metadata(  # noqa: SLF001
			entry, k=6, survey_id='survey', identity=identity
		)

	validate(metadata)
	for altered in (
		{**metadata, 'source': {'unexpected': 'provenance'}},
		{key: value for key, value in metadata.items() if key != 'source'},
	):
		with pytest.raises(
			ValueError, match=r'manifest keys mismatch|lateral metadata provenance'
		):
			validate(altered)


def test_lateral_export_replays_real_frozen_sources_and_reuses_complete_bundle(
	tmp_path,
) -> None:
	"""A real frozen hard/posterior/embedding chain publishes immutable targets."""
	config = _real_lateral_export_fixture(tmp_path)
	first = export_multi_head_lateral_targets(config)
	assert [plan.action for plan in first] == ['NEW', 'NEW', 'NEW']
	manifest = lateral_targets.load_multi_head_lateral_target_manifest(
		config.handoff_manifest
	)
	for k in (6, 8, 10):
		entry = manifest['heads'][str(k)]['surveys']['survey']
		labels = np.load(entry['labels']['path'], mmap_mode='r', allow_pickle=False)
		confidence = np.load(
			entry['confidence']['path'], mmap_mode='r', allow_pickle=False
		)
		valid = np.load(
			entry['valid_tokens']['path'], mmap_mode='r', allow_pickle=False
		)
		assert labels.dtype == np.int32
		assert confidence.dtype == np.float32
		assert valid.dtype == np.bool_
		assert labels.shape == confidence.shape == valid.shape == (2, 1, 10)
		assert np.all((labels[valid] >= 0) & (labels[valid] < k))
		assert np.all(labels[~valid] == -1)
		assert np.all(confidence[valid] == 1.0)
		assert np.all(confidence[~valid] == 0.0)
		assert np.all(np.diff(labels[0, 0]) >= 0)
		assert np.all(np.diff(labels[1, 0]) >= 0)
		assert np.all(np.bincount(labels[valid], minlength=k) > 0)
		validate_multi_head_target_reference(
			k=k,
			survey_id='survey',
			labels_path=entry['labels']['path'],
			confidence_path=entry['confidence']['path'],
			valid_tokens_path=entry['valid_tokens']['path'],
			metadata_path=entry['metadata']['path'],
			hashes={
				name: entry[name]['sha256']
				for name in ('labels', 'confidence', 'valid_tokens', 'metadata')
			},
			expected_token_grid_shape=list(labels.shape),
			validate_array_semantics=True,
		)
	tracked = [
		config.handoff_manifest,
		*sorted((config.output_root / 'bundle').rglob('*')),
	]
	before = {
		path: (path.stat().st_mtime_ns, file_sha256(path))
		for path in tracked
		if path.is_file()
	}
	second = export_multi_head_lateral_targets(config, only_missing=True)
	assert [plan.action for plan in second] == ['REUSE', 'REUSE', 'REUSE']
	after = {
		path: (path.stat().st_mtime_ns, file_sha256(path)) for path in before
	}
	assert after == before


def test_lateral_reference_only_validates_schema_v1_metadata_against_arrays(
	tmp_path,
) -> None:
	"""Reference-only loading still binds schema-v1 metadata to array headers."""
	config = _real_lateral_export_fixture(tmp_path)
	export_multi_head_lateral_targets(config)
	manifest = json.loads(config.handoff_manifest.read_text(encoding='utf-8'))
	entry = manifest['heads']['6']['surveys']['survey']
	metadata_path = Path(entry['metadata']['path'])
	original = json.loads(metadata_path.read_text(encoding='utf-8'))
	for field, value in (
		('token_grid_shape', [1, 1, 1]),
		('valid_token_count', original['valid_token_count'] + 1),
		('label_counts', {str(label): 0 for label in range(6)}),
	):
		altered = {**original, field: value}
		metadata_path.write_text(
			json.dumps(altered, sort_keys=True) + '\n', encoding='utf-8'
		)
		entry['metadata']['sha256'] = file_sha256(metadata_path)
		config.handoff_manifest.write_text(
			json.dumps(manifest, sort_keys=True) + '\n', encoding='utf-8'
		)
		with pytest.raises(
			ValueError, match='lateral metadata provenance differs from entry'
		):
			lateral_targets.load_multi_head_lateral_target_manifest(
				config.handoff_manifest, validate_array_semantics=False
			)


def test_lateral_reference_only_skips_all_full_replay_helpers(
	tmp_path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Runtime validation must never replay frozen HMM or smoothing semantics."""
	config = _real_lateral_export_fixture(tmp_path)
	export_multi_head_lateral_targets(config)

	def reject(*_args, **_kwargs):
		raise AssertionError('full lateral replay helper was called')

	for name in (
		'_validate_frozen_source_replay',
		'_affinity_scale',
		'_emission_gap_scales',
		'_validate_diagnostics_from_frozen_sources',
	):
		monkeypatch.setattr(lateral_targets, name, reject)
	lateral_targets.load_multi_head_lateral_target_manifest(
		config.handoff_manifest,
		validate_array_semantics=False,
	)
	with pytest.raises(AssertionError, match='full lateral replay helper'):
		lateral_targets.load_multi_head_lateral_target_manifest(
			config.handoff_manifest,
			validate_array_semantics=True,
		)


def test_lateral_reference_only_skips_nested_posterior_array_validation(
	tmp_path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Checkpoint-style loading keeps nested posterior validation reference-only."""
	config = _real_lateral_export_fixture(tmp_path)
	export_multi_head_lateral_targets(config)
	hard_load_modes: list[bool] = []
	load_hard_manifest = state_posterior.load_multi_head_target_manifest

	def load_hard(path, *, validate_array_semantics: bool):
		hard_load_modes.append(validate_array_semantics)
		return load_hard_manifest(
			path,
			validate_array_semantics=validate_array_semantics,
		)

	def reject(*_args, **_kwargs):
		raise AssertionError('nested posterior full validation was called')

	monkeypatch.setattr(
		state_posterior,
		'load_multi_head_target_manifest',
		load_hard,
	)
	monkeypatch.setattr(state_posterior, '_validate_posterior_array', reject)
	monkeypatch.setattr(
		state_posterior,
		'_validate_manifest_source_embedding_identity',
		reject,
	)
	monkeypatch.setattr(state_posterior, 'discover_embedding_inputs', reject)

	lateral_targets.load_multi_head_lateral_target_manifest(
		config.handoff_manifest,
		validate_array_semantics=False,
	)

	assert hard_load_modes == [False]


def test_lateral_reference_only_rejects_all_heads_using_same_wrong_mask(
	tmp_path,
) -> None:
	"""Cross-head agreement cannot replace the frozen source-mask identity."""
	config = _real_lateral_export_fixture(tmp_path)
	export_multi_head_lateral_targets(config)
	wrong_path = tmp_path / 'wrong.valid_tokens.npy'
	wrong = np.ones((2, 1, 10), dtype=bool)
	wrong[0, 0, 0] = False
	np.save(wrong_path, wrong, allow_pickle=False)
	wrong_reference = {
		'path': str(wrong_path),
		'sha256': file_sha256(wrong_path),
		'shape': list(wrong.shape),
		'dtype': wrong.dtype.name,
	}
	handoff = json.loads(config.handoff_manifest.read_text(encoding='utf-8'))
	for k in (6, 8, 10):
		handoff['heads'][str(k)]['surveys']['survey']['valid_tokens'] = (
			wrong_reference
		)
	config.handoff_manifest.write_text(
		json.dumps(handoff, sort_keys=True) + '\n',
		encoding='utf-8',
	)
	with pytest.raises(
		ValueError,
		match='lateral valid-mask identity differs from current sources',
	):
		lateral_targets.load_multi_head_lateral_target_manifest(
			config.handoff_manifest,
			validate_array_semantics=False,
		)


def test_lateral_cross_head_mask_mismatch_is_quarantined_and_repaired(
	tmp_path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Owned cross-head mask drift is recoverable corruption, not source drift."""
	config = _real_lateral_export_fixture(tmp_path)
	export_multi_head_lateral_targets(config)
	head_path = config.output_root / 'bundle' / 'k8' / 'head_metadata.json'
	head = json.loads(head_path.read_text(encoding='utf-8'))
	entry = head['surveys']['survey']
	valid_path = Path(entry['valid_tokens']['path'])
	valid = np.load(valid_path)
	valid[0, 0, 0] = False
	np.save(valid_path, valid, allow_pickle=False)
	entry['valid_tokens']['sha256'] = file_sha256(valid_path)
	head_path.write_text(json.dumps(head, sort_keys=True) + '\n', encoding='utf-8')

	plans = lateral_targets.plan_multi_head_lateral_target_exports(
		config,
		only_missing=True,
	)
	assert [plan.action for plan in plans] == [
		'QUARANTINE',
		'QUARANTINE',
		'QUARANTINE',
	]
	quarantined: list[Path] = []
	strict_quarantine = lateral_targets._quarantine  # noqa: SLF001

	def quarantine_existing(path: Path) -> None:
		assert path.exists()
		quarantined.append(path)
		strict_quarantine(path)

	monkeypatch.setattr(lateral_targets, '_quarantine', quarantine_existing)
	repaired = export_multi_head_lateral_targets(config, only_missing=True)
	assert [plan.action for plan in repaired] == [
		'QUARANTINE',
		'QUARANTINE',
		'QUARANTINE',
	]
	assert quarantined == [config.handoff_manifest, config.output_root / 'bundle']
	assert list(config.output_root.glob('bundle.quarantine-*'))
	assert list(config.output_root.glob(f'{config.handoff_manifest.name}.quarantine-*'))
	_assert_full_validation_and_reuse_idempotence(config)


def test_lateral_source_mask_identity_mismatch_is_error_without_output_mutation(
	tmp_path,
) -> None:
	"""A current hard/posterior mismatch is an input error and preserves output."""
	config = _real_lateral_export_fixture(tmp_path)
	export_multi_head_lateral_targets(config)
	posterior = json.loads(config.source_posterior_manifest.read_text(encoding='utf-8'))
	for k in (6, 8, 10):
		entry = posterior['heads'][str(k)]['surveys']['survey']
		valid_path = Path(entry['valid_tokens']['path'])
		valid = np.load(valid_path)
		valid[0, 0, 0] = False
		np.save(valid_path, valid, allow_pickle=False)
		entry['valid_tokens']['sha256'] = file_sha256(valid_path)
		posterior_path = Path(entry['posterior']['path'])
		values = np.load(posterior_path)
		values[0, 0, 0] = 0.0
		np.save(posterior_path, values, allow_pickle=False)
		entry['posterior']['sha256'] = file_sha256(posterior_path)
	config.source_posterior_manifest.write_text(
		json.dumps(posterior, sort_keys=True) + '\n',
		encoding='utf-8',
	)
	before = _public_file_snapshot(config)
	plans = lateral_targets.plan_multi_head_lateral_target_exports(
		config,
		only_missing=True,
	)
	assert [plan.action for plan in plans] == ['ERROR', 'ERROR', 'ERROR']
	assert all(
		'hard and posterior valid-mask identities differ' in plan.reason
		for plan in plans
	)
	assert _public_file_snapshot(config) == before


def test_lateral_cross_head_source_mask_mismatch_returns_typed_error_plans(
	tmp_path,
) -> None:
	"""One posterior head with a different mask is classified as source ERROR."""
	config = _real_lateral_export_fixture(tmp_path)
	posterior = json.loads(config.source_posterior_manifest.read_text(encoding='utf-8'))
	entry = posterior['heads']['8']['surveys']['survey']
	valid_path = Path(entry['valid_tokens']['path'])
	valid = np.load(valid_path)
	valid[0, 0, 0] = False
	np.save(valid_path, valid, allow_pickle=False)
	entry['valid_tokens']['sha256'] = file_sha256(valid_path)
	config.source_posterior_manifest.write_text(
		json.dumps(posterior, sort_keys=True) + '\n',
		encoding='utf-8',
	)

	plans = lateral_targets.plan_multi_head_lateral_target_exports(
		config,
		only_missing=True,
	)

	assert [plan.action for plan in plans] == ['ERROR', 'ERROR', 'ERROR']
	assert all(
		plan.reason is not None
		and 'hard and posterior valid-mask identities differ for k=8 survey'
		in plan.reason
		for plan in plans
	)
	assert not config.output_root.exists()


@pytest.mark.parametrize('missing', ['bundle', 'handoff'])
def test_lateral_orphan_publication_repairs_only_existing_paths(
	tmp_path,
	monkeypatch: pytest.MonkeyPatch,
	missing: str,
) -> None:
	"""Handoff-only and bundle-only generations repair bundle-first/handoff-last."""
	config = _real_lateral_export_fixture(tmp_path)
	export_multi_head_lateral_targets(config)
	bundle = config.output_root / 'bundle'
	if missing == 'bundle':
		shutil.rmtree(bundle)
		expected_quarantine = config.handoff_manifest
	else:
		config.handoff_manifest.unlink()
		expected_quarantine = bundle
	quarantined: list[Path] = []
	strict_quarantine = lateral_targets._quarantine  # noqa: SLF001

	def quarantine_existing(path: Path) -> None:
		assert path.exists()
		quarantined.append(path)
		strict_quarantine(path)

	monkeypatch.setattr(lateral_targets, '_quarantine', quarantine_existing)
	repaired = export_multi_head_lateral_targets(config, only_missing=True)
	assert [plan.action for plan in repaired] == [
		'QUARANTINE',
		'QUARANTINE',
		'QUARANTINE',
	]
	assert quarantined == [expected_quarantine]
	assert bundle.is_dir()
	assert config.handoff_manifest.is_file()
	assert list(
		expected_quarantine.parent.glob(
			f'{expected_quarantine.name}.quarantine-*'
		)
	)
	_assert_full_validation_and_reuse_idempotence(config)


def test_lateral_reuse_rejects_self_consistent_diagnostic_tampering(tmp_path) -> None:
	"""Reuse rebuilds metrics from frozen inputs, not just diagnostics files."""
	config = _real_lateral_export_fixture(tmp_path)
	export_multi_head_lateral_targets(config)
	head_path = config.output_root / 'bundle' / 'k6' / 'head_metadata.json'
	head = json.loads(head_path.read_text(encoding='utf-8'))
	diagnostics = head['diagnostics']
	diagnostics['aggregate']['ordered_path']['violation_count'] = 1
	payload = {
		'per_survey': diagnostics['per_survey'],
		'aggregate': diagnostics['aggregate'],
		'resolved_scales': diagnostics['resolved_scales'],
	}
	json_path = diagnostics['json']['path']
	csv_path = diagnostics['csv']['path']
	Path(json_path).write_text(
		json.dumps(payload, sort_keys=True) + '\n', encoding='utf-8'
	)
	lateral_targets._write_diagnostics_csv(Path(csv_path), payload)  # noqa: SLF001
	diagnostics['json']['sha256'] = file_sha256(json_path)
	diagnostics['csv']['sha256'] = file_sha256(csv_path)
	head_path.write_text(json.dumps(head, sort_keys=True) + '\n', encoding='utf-8')
	handoff = json.loads(config.handoff_manifest.read_text(encoding='utf-8'))
	handoff['heads']['6'] = head
	config.handoff_manifest.write_text(
		json.dumps(handoff, sort_keys=True) + '\n', encoding='utf-8'
	)
	with pytest.raises(ValueError, match='frozen sources'):
		lateral_targets.load_multi_head_lateral_target_manifest(
			config.handoff_manifest
		)
	plans = lateral_targets.plan_multi_head_lateral_target_exports(
		config, only_missing=True
	)
	assert [plan.action for plan in plans] == ['QUARANTINE', 'QUARANTINE', 'QUARANTINE']


def test_lateral_reuse_quarantines_structurally_malformed_metadata(tmp_path) -> None:
	"""Malformed owned metadata is recoverable corruption, not an identity clash."""
	config = _real_lateral_export_fixture(tmp_path)
	export_multi_head_lateral_targets(config)
	head_path = config.output_root / 'bundle' / 'k6' / 'head_metadata.json'
	head = json.loads(head_path.read_text(encoding='utf-8'))
	entry = head['surveys']['survey']
	metadata_path = Path(entry['metadata']['path'])
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	del metadata['source']
	metadata_path.write_text(
		json.dumps(metadata, sort_keys=True) + '\n', encoding='utf-8'
	)
	entry['metadata']['sha256'] = file_sha256(metadata_path)
	head_path.write_text(json.dumps(head, sort_keys=True) + '\n', encoding='utf-8')
	tracked = [
		config.handoff_manifest,
		*sorted((config.output_root / 'bundle').rglob('*')),
	]
	before = {
		path: (path.read_bytes(), path.stat().st_mtime_ns)
		for path in tracked
		if path.is_file()
	}
	plans = lateral_targets.plan_multi_head_lateral_target_exports(
		config, only_missing=True
	)
	assert [plan.action for plan in plans] == ['QUARANTINE', 'QUARANTINE', 'QUARANTINE']
	after = {
		path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before
	}
	assert after == before


def test_lateral_reuse_leaves_complete_bundle_for_other_source_identity_untouched(
	tmp_path,
) -> None:
	"""A complete handoff from another source identity requires another root."""
	config = _real_lateral_export_fixture(tmp_path)
	export_multi_head_lateral_targets(config)
	hard_manifest = tmp_path / 'copied_hard_manifest.json'
	hard_manifest.write_bytes(config.source_hard_manifest.read_bytes())
	posterior_manifest = tmp_path / 'copied_posterior_manifest.json'
	posterior = json.loads(config.source_posterior_manifest.read_text(encoding='utf-8'))
	posterior['source_hard_manifest'] = {
		'path': str(hard_manifest),
		'sha256': file_sha256(hard_manifest),
	}
	posterior_manifest.write_text(
		json.dumps(posterior, sort_keys=True) + '\n', encoding='utf-8'
	)
	mismatched = replace(
		config,
		source_hard_manifest=hard_manifest,
		source_posterior_manifest=posterior_manifest,
	)
	tracked = [
		mismatched.handoff_manifest,
		*sorted((mismatched.output_root / 'bundle').rglob('*')),
	]
	before = {
		path: (path.stat().st_mtime_ns, file_sha256(path))
		for path in tracked
		if path.is_file()
	}
	plans = lateral_targets.plan_multi_head_lateral_target_exports(
		mismatched, only_missing=True
	)
	assert [plan.action for plan in plans] == ['ERROR', 'ERROR', 'ERROR']
	after = {
		path: (path.stat().st_mtime_ns, file_sha256(path)) for path in before
	}
	assert after == before


def test_lateral_reuse_errors_for_a_different_current_survey_set(
	tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""A complete bundle from another survey set is not quarantined as corrupt."""
	config = _real_lateral_export_fixture(tmp_path)
	export_multi_head_lateral_targets(config)
	source, posterior, inputs, models = lateral_targets._validate_sources(  # noqa: SLF001
		config
	)
	current_source = json.loads(json.dumps(source))
	current_posterior = json.loads(json.dumps(posterior))
	for k in (6, 8, 10):
		current_source['heads'][str(k)]['surveys'] = {'other_survey': {}}
		current_posterior['heads'][str(k)]['surveys'] = {'other_survey': {}}
	monkeypatch.setattr(
		lateral_targets,
		'_validate_sources',
		lambda _config: (current_source, current_posterior, inputs, models),
	)
	monkeypatch.setattr(
		lateral_targets, '_validate_frozen_source_replay', lambda *_args: None
	)
	monkeypatch.setattr(
		lateral_targets, '_affinity_scale', lambda *_args: (1.0, {})
	)
	monkeypatch.setattr(
		lateral_targets,
		'_emission_gap_scales',
		lambda *_args: ({6: 1.0, 8: 1.0, 10: 1.0}, {6: {}, 8: {}, 10: {}}),
	)
	monkeypatch.setattr(lateral_targets, '_source_snapshot', lambda *_args: '{}')
	tracked = [
		config.handoff_manifest,
		*sorted((config.output_root / 'bundle').rglob('*')),
	]
	before = {
		path: (path.read_bytes(), path.stat().st_mtime_ns)
		for path in tracked
		if path.is_file()
	}
	plans = lateral_targets.plan_multi_head_lateral_target_exports(
		config, only_missing=True
	)
	assert [plan.action for plan in plans] == ['ERROR', 'ERROR', 'ERROR']
	after = {
		path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before
	}
	assert after == before


def _public_file_snapshot(
	config: MultiHeadLateralTargetExportConfig,
) -> dict[Path, tuple[bytes, int]]:
	paths = [
		config.handoff_manifest,
		*sorted((config.output_root / 'bundle').rglob('*')),
	]
	return {
		path: (path.read_bytes(), path.stat().st_mtime_ns)
		for path in paths
		if path.is_file()
	}


def _assert_full_validation_and_reuse_idempotence(
	config: MultiHeadLateralTargetExportConfig,
) -> None:
	lateral_targets.load_multi_head_lateral_target_manifest(config.handoff_manifest)
	before = _public_file_snapshot(config)
	reused = export_multi_head_lateral_targets(config, only_missing=True)
	assert [plan.action for plan in reused] == ['REUSE', 'REUSE', 'REUSE']
	assert _public_file_snapshot(config) == before


def _real_lateral_export_fixture(
	tmp_path,
) -> MultiHeadLateralTargetExportConfig:
	"""Create a small, fully hash-bound frozen HMM source chain."""
	shape = (2, 1, 10)
	embedding_root = tmp_path / 'embeddings'
	embedding_root.mkdir()
	features = np.empty((*shape, 2), dtype=np.float32)
	for x in range(shape[0]):
		features[x, 0, :, 0] = np.arange(shape[2], dtype=np.float32)
		features[x, 0, :, 1] = 1.0 + 0.1 * x
	np.save(embedding_root / 'survey.embeddings.npy', features, allow_pickle=False)
	valid = np.ones(shape, dtype=bool)
	np.save(embedding_root / 'survey.valid_tokens.npy', valid, allow_pickle=False)
	(embedding_root / 'survey.embedding_metadata.json').write_text(
		json.dumps(
			{
				'survey_id': 'survey',
				'source_amplitude_path': 'amplitude.npy',
				'checkpoint_path': 'checkpoint.pt',
				'checkpoint_sha256': 'checkpoint',
				'model_geometry': {'name': 'fixture'},
				'patch_size': [1, 1, 1],
				'token_grid_shape': list(shape),
				'window_size': [1, 1, 1],
				'overlap': [0, 0, 0],
				'normalization_stats_path': 'stats.json',
				'output_dtype': 'float32',
				'min_token_valid_fraction': 1.0,
				'zero_mask': {},
			}
		),
		encoding='utf-8',
	)
	clustering = tmp_path / 'clustering'
	clustering_config = tmp_path / 'clustering.yaml'
	clustering_config.write_text('frozen: true\n', encoding='utf-8')
	hard_root = tmp_path / 'hard_targets'
	model_identities: dict[str, dict[str, object]] = {}
	metadata_hashes: dict[str, str] = {}
	label_hashes: dict[str, dict[str, str]] = {}
	for k in (6, 8, 10):
		model_root = clustering / 'models' / f'k{k}'
		model_root.mkdir(parents=True)
		dump(FunctionTransformer(validate=False), model_root / 'preprocessor.joblib')
		dump(
			{
				'emission_source': 'embedding',
				'transition_costs': np.zeros((k, k), dtype=np.float32),
				'initial_state_costs': np.zeros(k, dtype=np.float32),
				'terminal_state_costs': np.zeros(k, dtype=np.float32),
			},
			model_root / 'hmm_model.joblib',
		)
		centers = np.column_stack(
			(
				np.linspace(0.0, 9.0, k, dtype=np.float32),
				np.ones(k, dtype=np.float32),
			)
		)
		np.save(model_root / 'cluster_centers.npy', centers, allow_pickle=False)
		(model_root / 'clustering_metadata.json').write_text(
			json.dumps({'k': k}), encoding='utf-8'
		)
		input_item = EmbeddingInput(
			'survey',
			embedding_root / 'survey.embeddings.npy',
			embedding_root / 'survey.valid_tokens.npy',
			embedding_root / 'survey.embedding_metadata.json',
		)
		model = lateral_targets.load_frozen_hmm_model(
			clustering_output_dir=clustering, clustering_config=clustering_config, k=k
		)
		labels = np.full(shape, -1, dtype=np.int32)
		for x in range(shape[0]):
			flat = np.arange(x * shape[2], (x + 1) * shape[2], dtype=np.int64)
			_, labels[x, 0] = lateral_targets.replay_frozen_hmm_trace(
				input_item, flat, model, k=k
			)
		label_path = (
			clustering / 'labels' / f'k{k}' / 'survey.cluster_labels_token.npy'
		)
		label_path.parent.mkdir(parents=True, exist_ok=True)
		np.save(label_path, labels, allow_pickle=False)
		write_pseudo_target(
			hard_root,
			k=k,
			survey_id='survey',
			labels=labels,
			confidence=np.ones(shape, dtype=np.float32),
			valid_tokens=valid,
			metadata={
				'source_clustering_output_dir': str(clustering),
				'source_label_path': str(label_path),
			},
			schema_version=1,
			write_boundary_weight=False,
		)
		identity = {
			name: {
				'path': str(model_root / filename),
				'sha256': file_sha256(model_root / filename),
			}
			for name, filename in {
				'preprocessor': 'preprocessor.joblib',
				'hmm_model': 'hmm_model.joblib',
				'centers': 'cluster_centers.npy',
				'metadata': 'clustering_metadata.json',
			}.items()
		}
		model_identities[str(k)] = {**identity, 'residualizer': None}
		metadata_hashes[str(k)] = identity['metadata']['sha256']
		label_hashes[str(k)] = {label_path.name: file_sha256(label_path)}
	replay_root = tmp_path / 'replay_k6'
	labels_k6 = np.load(
		clustering / 'labels' / 'k6' / 'survey.cluster_labels_token.npy'
	)
	replay_label = tmp_path / 'replay_labels' / 'survey.cluster_labels_token.npy'
	replay_label.parent.mkdir()
	np.save(replay_label, labels_k6, allow_pickle=False)
	write_pseudo_target(
		replay_root,
		k=6,
		survey_id='survey',
		labels=labels_k6,
		confidence=np.ones(shape, dtype=np.float32),
		valid_tokens=valid,
		metadata={
			'source_clustering_output_dir': str(clustering),
			'source_label_path': str(replay_label),
		},
		schema_version=1,
		write_boundary_weight=False,
	)
	(hard_root / 'multi_head_pseudo_target_export_handoff.json').write_text(
		json.dumps(
			{
				'artifact_type': 'strat_hmm_multi_head_pseudo_target_export_handoff',
				'schema_version': 2,
				'completion_status': 'COMPLETE',
				'pseudo_target_root': str(hard_root),
				'clustering': {
					'path': str(clustering),
					'config_path': str(clustering_config),
					'config_sha256': file_sha256(clustering_config),
					'metadata_sha256': metadata_hashes,
					'model_artifacts': model_identities,
					'labels': label_hashes,
				},
				'heads': {
					str(k): {'pseudo_target_root': str(hard_root / f'k{k}')}
					for k in (6, 8, 10)
				},
			}
		),
		encoding='utf-8',
	)
	hard_manifest = tmp_path / 'hard_manifest.json'
	build_multi_head_target_manifest(
		manifest_path=hard_manifest,
		source_embedding_dir=embedding_root,
		head_roots=dict.fromkeys((6, 8, 10), hard_root),
		replay_k6_root=replay_root,
	)
	posterior_root = tmp_path / 'posterior'
	posterior_config = MultiHeadStatePosteriorExportConfig(
		source_hard_manifest=hard_manifest,
		clustering_output_dir=clustering,
		source_embedding_dir=embedding_root,
		posterior_root=posterior_root,
		clustering_config=clustering_config,
		handoff_manifest=posterior_root / 'handoff.json',
	)
	export_multi_head_state_posteriors(posterior_config)
	return MultiHeadLateralTargetExportConfig(
		source_hard_manifest=hard_manifest,
		source_posterior_manifest=posterior_config.handoff_manifest,
		clustering_output_dir=clustering,
		clustering_config=clustering_config,
		source_embedding_dir=embedding_root,
		output_root=tmp_path / 'lateral',
		pairwise_strength_ratio=0.25,
		handoff_manifest=tmp_path / 'lateral' / 'handoff.json',
	)
