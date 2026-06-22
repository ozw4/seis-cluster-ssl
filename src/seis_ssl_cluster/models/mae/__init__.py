"""Masked autoencoder components for seismic SSL clustering."""

from importlib import import_module
from typing import TYPE_CHECKING

from seis_ssl_cluster.models.mae.patching import (
	compute_num_patches,
	patchify_3d,
	unpatchify_3d,
)
from seis_ssl_cluster.models.mae.positional_encoding import (
	build_3d_sincos_position_embedding,
	restore_decoder_sequence,
	select_visible_tokens,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.models.mae.model import AmplitudeMAE3D

__all__ = [
	'AmplitudeMAE3D',
	'build_3d_sincos_position_embedding',
	'compute_num_patches',
	'patchify_3d',
	'restore_decoder_sequence',
	'select_visible_tokens',
	'unpatchify_3d',
]


def __getattr__(name: str) -> object:
	"""Lazily expose the full MAE model without creating import cycles."""
	if name == 'AmplitudeMAE3D':
		return import_module('seis_ssl_cluster.models.mae.model').AmplitudeMAE3D
	msg = f'module {__name__!r} has no attribute {name!r}'
	raise AttributeError(msg)
