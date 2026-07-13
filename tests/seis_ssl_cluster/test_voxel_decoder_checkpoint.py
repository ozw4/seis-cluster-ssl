from __future__ import annotations

import pytest
import torch

from seis_ssl_cluster.training.voxel_decoder.checkpoint import (
	BEST_SELECTION_EPSILON,
	CHECKPOINT_SCHEMA_VERSION,
	load_voxel_decoder_checkpoint,
	validation_is_better,
)


def test_best_selection_uses_fixed_lexicographic_rule() -> None:
	best = {'macro_f1': 0.5, 'mean_iou': 0.4, 'loss': 1.0}
	assert validation_is_better(
		{'macro_f1': 0.6, 'mean_iou': 0.0, 'loss': 9.0}, best
	)
	assert validation_is_better(
		{'macro_f1': 0.5, 'mean_iou': 0.5, 'loss': 9.0}, best
	)
	assert validation_is_better(
		{'macro_f1': 0.5, 'mean_iou': 0.4, 'loss': 0.9}, best
	)
	assert not validation_is_better(dict(best), best)


def test_best_selection_ties_within_explicit_epsilon() -> None:
	best = {'macro_f1': 0.5, 'mean_iou': 0.4, 'loss': 1.0}
	candidate = dict(best)
	candidate['macro_f1'] = 0.5 + BEST_SELECTION_EPSILON / 2
	assert not validation_is_better(candidate, best)


def test_best_selection_rejects_invalid_metrics() -> None:
	with pytest.raises(TypeError, match='macro_f1'):
		validation_is_better({'macro_f1': None}, {'macro_f1': 1.0})


def test_loader_rejects_previous_model_semantics_schema(tmp_path) -> None:
	path = tmp_path / 'old.pt'
	payload = {
		'schema_version': CHECKPOINT_SCHEMA_VERSION - 1,
		'epoch': 0,
		'global_step': 0,
		'model_state_dict': {},
		'optimizer_state_dict': {},
		'best_selection_state': None,
		'training_history': [],
		'current_metrics': {},
		'resolved_config': {},
		'class_weights': [],
		'artifact_identities': {},
		'tile_manifest_hashes': {},
		'rng_states': {},
		'checkpoint_kind': 'completed',
	}
	torch.save(payload, path)

	with pytest.raises(ValueError, match=r'unsupported.*schema_version'):
		load_voxel_decoder_checkpoint(path)
