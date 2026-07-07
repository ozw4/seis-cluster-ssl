"""F3 lithology report package."""

from __future__ import annotations

from seis_ssl_cluster.f3.lithology.report import comparison as _comparison
from seis_ssl_cluster.f3.lithology.report import figures as _figures
from seis_ssl_cluster.f3.lithology.report import markdown as _markdown
from seis_ssl_cluster.f3.lithology.report import metrics_loader as _metrics_loader
from seis_ssl_cluster.f3.lithology.report import publish as _publish
from seis_ssl_cluster.f3.lithology.report._common import (
	COMPARISON_ID_COLUMNS,
	OVERALL_METRIC_COLUMNS,
)
from seis_ssl_cluster.f3.lithology.report._core import (
	F3LithologyReportConfig,
	F3LithologyReportResult,
	build_f3_lithology_report,
)
from seis_ssl_cluster.f3.lithology.report.comparison import (
	F3LithologyComparisonReportConfig,
	F3LithologyComparisonReportResult,
	build_f3_lithology_comparison_report,
)
from seis_ssl_cluster.f3.lithology.report.figures import (
	F3LithologyComparisonFigureFontSizes,
	F3LithologyComparisonFigureSizes,
	F3LithologyComparisonFigureStyle,
	default_f3_lithology_comparison_figure_style,
)
from seis_ssl_cluster.f3.lithology.report.markdown import (
	render_f3_lithology_report_markdown,
)
from seis_ssl_cluster.f3.lithology.report.publish import (
	F3LithologyComparisonPublishConfig,
	F3LithologyPublishConfig,
	publish_f3_lithology_comparison_report,
	publish_f3_lithology_report,
)

_ATTR_MODULES = (_comparison, _figures, _markdown, _metrics_loader, _publish)
_MISSING = object()


def __getattr__(name: str) -> object:
	for module in _ATTR_MODULES:
		value = getattr(module, name, _MISSING)
		if value is not _MISSING:
			return value
	raise AttributeError(name)


__all__ = [
	'COMPARISON_ID_COLUMNS',
	'OVERALL_METRIC_COLUMNS',
	'F3LithologyComparisonFigureFontSizes',
	'F3LithologyComparisonFigureSizes',
	'F3LithologyComparisonFigureStyle',
	'F3LithologyComparisonPublishConfig',
	'F3LithologyComparisonReportConfig',
	'F3LithologyComparisonReportResult',
	'F3LithologyPublishConfig',
	'F3LithologyReportConfig',
	'F3LithologyReportResult',
	'build_f3_lithology_comparison_report',
	'build_f3_lithology_report',
	'default_f3_lithology_comparison_figure_style',
	'publish_f3_lithology_comparison_report',
	'publish_f3_lithology_report',
	'render_f3_lithology_report_markdown',
]
