"""Bridge the audited Multi-Head source to the frozen Volve runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from seis_ssl_cluster.config.io import _expand_environment_variables
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.training.random_checkpoint import (
	load_checkpoint_metadata_without_weights,
)
from seis_ssl_cluster.training.strat_hmm_source_audit import audit_multi_head_source

if TYPE_CHECKING:
	from seis_ssl_cluster.volve.horizon_recipe_arm import VolveHorizonRecipeArmConfig


def audit_multi_head_recipe_source(
	config: VolveHorizonRecipeArmConfig,
) -> dict[str, object]:
	"""Bind the completed source and its declared base objective to the arm."""
	path = config.multi_head_training_config
	if path is None or config.hmm is not None:
		raise ValueError('Multi-Head source requires exactly one training recipe')
	evidence = audit_multi_head_source(path)
	if evidence['head_ks'] != [6, 8, 10] or evidence['model_tag'] != config.arm_id:
		raise ValueError('Multi-Head source candidate/head mismatch')
	checkpoint = evidence['checkpoint']
	if checkpoint['path'] != str(config.arm_checkpoint):
		raise ValueError('Multi-Head source checkpoint differs from downstream')
	raw = _expand_environment_variables(
		yaml.safe_load(
			path.read_text().replace(
				'${SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256}', '0' * 64
			)
		)
	)
	parent = raw['student']['init_checkpoint']
	payload = load_checkpoint_metadata_without_weights(config.arm_checkpoint)
	objective = (
		'local_barlow_twins_3d'
		if config.recipe.method == 'local_barlow_twins_3d'
		else 'amp_mae3d'
	)
	if payload.get('pretraining_method', 'amp_mae3d') != objective:
		raise ValueError(
			'Multi-Head source base objective differs from declared recipe'
		)
	return {
		'model_id': config.arm_id,
		'role': 'recipe_arm',
		'checkpoint': checkpoint['path'],
		'checkpoint_sha256': checkpoint['sha256'],
		'objective': objective,
		'parent_checkpoint': parent,
		'parent_checkpoint_sha256': file_sha256(parent),
		'multi_head': evidence,
	}
