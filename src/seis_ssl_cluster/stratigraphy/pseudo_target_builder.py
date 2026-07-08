"""Build refreshed strat HMM pseudo-target artifacts from prototype logits."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


def build_strat_hmm_pseudo_targets(
	config: Mapping[str, object],
	*,
	device: str | None = None,
	overwrite: bool | None = None,
) -> list[Path]:
	"""Refresh strat HMM pseudo-targets from a trained prototype head."""
	raise NotImplementedError(
		'full-volume strat HMM pseudo-target refresh is not implemented yet',
	)


__all__ = ['build_strat_hmm_pseudo_targets']
