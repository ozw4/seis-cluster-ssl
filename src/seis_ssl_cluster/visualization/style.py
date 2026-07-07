"""Shared display style rules for seismic visualization views."""

from __future__ import annotations

_Z_SECTION_VIEWS = frozenset({'xz', 'yz', 'inline', 'crossline'})
_PLAN_VIEW_VIEWS = frozenset({'xy', 'timeslice', 'depth_slice', 'z'})
_VIEW_ALIASES = {
	'depthslice': 'depth_slice',
	'time_slice': 'timeslice',
}


def normalize_view_name(view: str) -> str:
	"""Return the canonical view name used by visualization helpers."""
	if not isinstance(view, str):
		msg = f'view must be a string; got {type(view).__name__}'
		raise TypeError(msg)
	name = view.strip().lower().replace('-', '_').replace(' ', '_')
	name = _VIEW_ALIASES.get(name, name)
	return validate_view_name(name)


def validate_view_name(view: str) -> str:
	"""Validate and return a canonical visualization view name."""
	if view in _Z_SECTION_VIEWS or view in _PLAN_VIEW_VIEWS:
		return view
	msg = f'unknown view: {view!r}'
	raise ValueError(msg)


def origin_for_view(view: str) -> str:
	"""Return the matplotlib image origin for a visualization view."""
	name = normalize_view_name(view)
	if name in _Z_SECTION_VIEWS:
		return 'upper'
	return 'lower'


def aspect_for_view(view: str) -> str:
	"""Return the matplotlib image aspect for a visualization view."""
	name = normalize_view_name(view)
	if name in _Z_SECTION_VIEWS:
		return 'auto'
	return 'equal'


__all__ = [
	'aspect_for_view',
	'normalize_view_name',
	'origin_for_view',
	'validate_view_name',
]
