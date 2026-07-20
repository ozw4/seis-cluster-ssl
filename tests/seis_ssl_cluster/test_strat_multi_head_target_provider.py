"""Regression coverage for nested multi-head pseudo-target supervision."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	CropRequest,
	MultiHeadStratPseudoTargetProvider,
	StratMultiHeadTargetInput,
	StratMultiHeadTargetManifest,
	SurveyManifest,
	TargetProviderContext,
	target_providers,
)
from seis_ssl_cluster.stratigraphy.multi_head import build_multi_head_target_manifest
from tests.seis_ssl_cluster.test_strat_multi_head_target_manifest import (
	_artifacts,
	_write_positive_preflight,
)

if TYPE_CHECKING:
	from collections.abc import Callable


def test_provider_slices_ordered_heads_and_applies_shared_source_mask(
	tmp_path: Path,
) -> None:
	provider, paths = _provider(tmp_path)
	source_valid_mask = np.ones((2, 3, 4), dtype=np.bool_)
	source_valid_mask[0, 0, 0] = False
	context = _context(
		tmp_path,
		token_valid_mask=source_valid_mask,
	)
	sample: dict[str, object] = {'coords': {'survey_id': 'survey-a'}}

	provider.add_targets(sample, context)

	targets = sample['strat_multi_targets']
	assert list(targets) == ['k6', 'k8', 'k10']
	for k in (6, 8, 10):
		target = targets[f'k{k}']
		expected = np.load(paths['survey-a'][k]['labels'])[1:3, 1:4, 1:5]
		expected[0, 0, 0] = -1
		np.testing.assert_array_equal(target['labels'], expected)
		assert target['labels'].dtype == np.int64
		assert target['confidence'].dtype == np.float32
		assert target['boundary_weight'].dtype == np.float32
		assert target['valid_mask'].dtype == np.bool_
		assert target['labels'][0, 0, 0] == -1
		assert target['confidence'][0, 0, 0] == 0.0
		assert target['boundary_weight'][0, 0, 0] == 0.0
		assert not target['valid_mask'][0, 0, 0]
	assert sample['coords']['strat_multi_target_metadata'] == {
		'boundary_weight_source': 'valid_token_indicator',
	}


@pytest.mark.parametrize(
	('mutate', 'match'),
	[
		(
			lambda paths: _replace_valid_token_with_invalid_target(paths, k=8),
			'valid_tokens must match',
		),
		(
			lambda paths: np.save(
				paths['survey-a'][8]['confidence'],
				np.ones((2, 5, 6), dtype=np.float32),
			),
			'shapes must match',
		),
		(
			lambda paths: _replace_label(paths, k=10, value=10),
			'valid labels must be in',
		),
		(
			lambda paths: _replace_confidence(paths, k=6, value=np.inf),
			'confidence must be finite',
		),
	],
)
def test_provider_rejects_invalid_head_crop_contract(
	tmp_path: Path,
	mutate: Callable[[dict[str, dict[int, dict[str, Path]]]], None],
	match: str,
) -> None:
	provider, paths = _provider(tmp_path)
	mutate(paths)

	with pytest.raises((TypeError, ValueError), match=match):
		provider.add_targets({'coords': {}}, _context(tmp_path))


def test_provider_rejects_head_empty_after_confidence_threshold(
	tmp_path: Path,
) -> None:
	provider, paths = _provider(tmp_path, min_confidence=0.5)
	np.save(
		paths['survey-a'][8]['confidence'],
		np.full((4, 5, 6), 0.25, dtype=np.float32),
	)
	sample: dict[str, object] = {'coords': {}}
	provider.add_targets(sample, _context(tmp_path))

	assert provider.sample_is_acceptable(sample) is False
	assert 'K8' in provider.rejection_message(
		survey_id='survey-a',
		max_resample_attempts=2,
		last_valid_fraction=1.0,
	)


def test_provider_validates_generated_boundary_weight_contract(tmp_path: Path) -> None:
	provider, _ = _provider(tmp_path)
	sample: dict[str, object] = {'coords': {}}
	provider.add_targets(sample, _context(tmp_path))
	targets = sample['strat_multi_targets']
	targets['k6']['boundary_weight'] = np.ones((2, 3, 4), dtype=np.float64)

	with pytest.raises(TypeError, match='boundary_weight dtype must be float32'):
		provider.sample_is_acceptable(sample)


@pytest.mark.parametrize('head_ks', [(6, 6, 10), (1, 8, 10)])
def test_provider_rejects_invalid_public_manifest_head_ks(
	tmp_path: Path,
	head_ks: tuple[int, ...],
) -> None:
	provider, _ = _provider(tmp_path)

	with pytest.raises(ValueError, match='head_ks'):
		MultiHeadStratPseudoTargetProvider(
			replace(provider.manifest, head_ks=head_ks),
		)


def test_provider_rejects_hash_tampered_public_manifest_input(tmp_path: Path) -> None:
	provider, paths = _provider(tmp_path)
	labels = paths['survey-a'][6]['labels']
	np.save(labels, np.zeros((4, 5, 6), dtype=np.int32))

	with pytest.raises(ValueError, match='labels hash mismatch'):
		MultiHeadStratPseudoTargetProvider(provider.manifest)


def test_provider_manifest_path_does_not_load_target_arrays(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	embeddings, heads = _artifacts(tmp_path)
	manifest_path = tmp_path / 'multi_head_target_manifest.json'
	migration, control = _write_positive_preflight(tmp_path)
	build_multi_head_target_manifest(
		manifest_path=manifest_path,
		source_embedding_dir=embeddings,
		head_roots=heads,
		replay_k6_root=heads[6],
		migration_decision=migration,
		control_summary=control,
	)

	def fail_array_load(*_args: object, **_kwargs: object) -> object:
		raise AssertionError('manifest-path provider construction must stay lazy')

	monkeypatch.setattr(target_providers.np, 'load', fail_array_load)

	provider = MultiHeadStratPseudoTargetProvider(manifest_path)

	assert provider.manifest.head_ks == (6, 8, 10)


def test_provider_loads_each_survey_heads_once_with_mmap(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	provider, _ = _provider(tmp_path)
	original_load = target_providers.np.load
	load_calls: list[Path] = []

	def spy_load(path: object, *args: object, **kwargs: object) -> np.ndarray:
		load_calls.append(Path(path))
		assert kwargs['mmap_mode'] == 'r'
		return original_load(path, *args, **kwargs)

	monkeypatch.setattr(target_providers.np, 'load', spy_load)
	provider.add_targets({'coords': {}}, _context(tmp_path))
	provider.add_targets({'coords': {}}, _context(tmp_path))
	provider.add_targets({'coords': {}}, _context(tmp_path, survey_id='survey-b'))

	assert len(load_calls) == 18


def _provider(
	tmp_path: Path,
	*,
	min_confidence: float = 0.0,
) -> tuple[MultiHeadStratPseudoTargetProvider, dict[str, dict[int, dict[str, Path]]]]:
	paths: dict[str, dict[int, dict[str, Path]]] = {}
	by_survey: dict[str, tuple[StratMultiHeadTargetInput, ...]] = {}
	for survey_id in ('survey-a', 'survey-b'):
		paths[survey_id] = {}
		inputs: list[StratMultiHeadTargetInput] = []
		for k in (6, 8, 10):
			root = tmp_path / survey_id / f'k{k}'
			root.mkdir(parents=True)
			labels = np.repeat(
				(np.arange(4 * 5).reshape(4, 5, 1) % k),
				6,
				axis=2,
			).astype(np.int32)
			valid_tokens = np.ones(labels.shape, dtype=np.bool_)
			confidence = np.ones(labels.shape, dtype=np.float32)
			for name, array in {
				'labels': labels,
				'confidence': confidence,
				'valid_tokens': valid_tokens,
			}.items():
				path = root / f'{name}.npy'
				np.save(path, array)
				paths[survey_id].setdefault(k, {})[name] = path
			metadata_path = root / 'metadata.json'
			metadata_path.write_text(
				json.dumps(
					{
						'artifact_type': 'strat_hmm_pseudo_target',
						'schema_version': 1,
						'k': k,
						'survey_id': survey_id,
					},
				),
				encoding='utf-8',
			)
			inputs.append(
				StratMultiHeadTargetInput(
					k=k,
					survey_id=survey_id,
					labels_path=paths[survey_id][k]['labels'],
					confidence_path=paths[survey_id][k]['confidence'],
					valid_tokens_path=paths[survey_id][k]['valid_tokens'],
					metadata_path=metadata_path,
					hashes={
						name: sha256(path.read_bytes()).hexdigest()
						for name, path in {
							**paths[survey_id][k],
							'metadata': metadata_path,
						}.items()
					},
				),
			)
		by_survey[survey_id] = tuple(inputs)
	manifest = StratMultiHeadTargetManifest(
		head_ks=(6, 8, 10),
		by_survey=by_survey,
		common_valid_token_sha256={
			survey_id: inputs[0].hashes['valid_tokens']
			for survey_id, inputs in by_survey.items()
		},
	)
	return MultiHeadStratPseudoTargetProvider(
		manifest,
		min_confidence=min_confidence,
	), paths


def _replace_valid_token_with_invalid_target(
	paths: dict[str, dict[int, dict[str, Path]]], *, k: int
) -> None:
	valid = np.load(paths['survey-a'][k]['valid_tokens'])
	labels = np.load(paths['survey-a'][k]['labels'])
	confidence = np.load(paths['survey-a'][k]['confidence'])
	valid[1, 1, 1] = False
	labels[1, 1, 1] = -1
	confidence[1, 1, 1] = 0.0
	np.save(paths['survey-a'][k]['valid_tokens'], valid)
	np.save(paths['survey-a'][k]['labels'], labels)
	np.save(paths['survey-a'][k]['confidence'], confidence)


def _replace_label(
	paths: dict[str, dict[int, dict[str, Path]]], *, k: int, value: int
) -> None:
	labels = np.load(paths['survey-a'][k]['labels'])
	labels[1, 1, 1] = value
	np.save(paths['survey-a'][k]['labels'], labels)


def _replace_confidence(
	paths: dict[str, dict[int, dict[str, Path]]], *, k: int, value: float
) -> None:
	confidence = np.load(paths['survey-a'][k]['confidence'])
	confidence[1, 1, 1] = value
	np.save(paths['survey-a'][k]['confidence'], confidence)


def _context(
	tmp_path: Path,
	*,
	survey_id: str = 'survey-a',
	token_valid_mask: np.ndarray | None = None,
) -> TargetProviderContext:
	if token_valid_mask is None:
		token_valid_mask = np.ones((2, 3, 4), dtype=np.bool_)
	return TargetProviderContext(
		manifest=_manifest(tmp_path, survey_id),
		crop_request=CropRequest(survey_id, (2, 2, 2), (4, 6, 8)),
		patch_size_xyz=(2, 2, 2),
		token_start_xyz=(1, 1, 1),
		token_size_xyz=(2, 3, 4),
		token_valid_mask=token_valid_mask,
	)


def _manifest(tmp_path: Path, survey_id: str) -> SurveyManifest:
	return SurveyManifest(
		survey_id=survey_id,
		root=tmp_path,
		amplitude=AmplitudeVolumeRecord(
			survey_id=survey_id,
			path=Path('amplitude.npy'),
			shape_xyz=(8, 10, 12),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=Path('stats.json'),
		),
	)
