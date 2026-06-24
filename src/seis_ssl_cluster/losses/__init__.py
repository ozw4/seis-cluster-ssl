"""Loss components for seismic SSL clustering."""

from seis_ssl_cluster.losses.gradient import gradient_loss_xyz
from seis_ssl_cluster.losses.mae_reconstruction import (
	mae_pretraining_loss,
	masked_patch_reconstruction_loss,
	reconstruction_target_patches_for_loss,
)
from seis_ssl_cluster.losses.target_normalization import (
	PatchTargetNormalizationResult,
	denormalize_predicted_patches,
	normalize_target_patches,
)

__all__ = [
	'gradient_loss_xyz',
	'mae_pretraining_loss',
	'masked_patch_reconstruction_loss',
	'reconstruction_target_patches_for_loss',
	'PatchTargetNormalizationResult',
	'denormalize_predicted_patches',
	'normalize_target_patches',
]
