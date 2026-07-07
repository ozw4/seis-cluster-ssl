"""Stratigraphic HMM clustering backend scaffold."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
	from collections.abc import Mapping


def run_stratigraphic_hmm_clustering(config: Mapping[str, object]) -> NoReturn:
	"""Run stratigraphic HMM clustering from a validated config mapping."""
	raise NotImplementedError('stratigraphic_hmm_kmeans is not implemented yet')


__all__ = ['run_stratigraphic_hmm_clustering']
