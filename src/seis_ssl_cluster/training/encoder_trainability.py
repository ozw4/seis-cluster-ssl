"""Trainability controls for the amplitude MAE encoder."""

from __future__ import annotations

from seis_ssl_cluster.models.mae.model import AmplitudeMAE3D


def freeze_all_and_unfreeze_top_encoder_blocks(
	model: AmplitudeMAE3D,
	*,
	unfreeze_top_blocks: int,
) -> tuple[str, ...]:
	"""Freeze the model, then unfreeze only the last N encoder blocks."""
	if not isinstance(model, AmplitudeMAE3D):
		msg = f'model must be an AmplitudeMAE3D; got {type(model).__name__}'
		raise TypeError(msg)
	if isinstance(unfreeze_top_blocks, bool) or not isinstance(
		unfreeze_top_blocks,
		int,
	):
		msg = f'unfreeze_top_blocks must be an integer; got {unfreeze_top_blocks!r}'
		raise TypeError(msg)
	if unfreeze_top_blocks < 0:
		msg = f'unfreeze_top_blocks must be nonnegative; got {unfreeze_top_blocks!r}'
		raise ValueError(msg)
	if unfreeze_top_blocks > model.encoder.depth:
		msg = (
			'unfreeze_top_blocks must be less than or equal to '
			f'model.encoder.depth ({model.encoder.depth}); got {unfreeze_top_blocks}'
		)
		raise ValueError(msg)

	model.requires_grad_(requires_grad=False)
	if unfreeze_top_blocks > 0:
		for block in model.encoder.layers[-unfreeze_top_blocks:]:
			block.requires_grad_(requires_grad=True)

	return tuple(
		name for name, parameter in model.named_parameters() if parameter.requires_grad
	)


__all__ = ['freeze_all_and_unfreeze_top_encoder_blocks']
