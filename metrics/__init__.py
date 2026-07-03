"""Per-simulator traffic-signal metrics with on-demand computation
and per-step API-call caching.

Public surface
--------------
- :class:`BaseMetrics` — abstract base, shared API.
- :class:`CityFlowMetrics` — backend for ``cityflow.Engine``.
- :class:`SumoMetrics` — backend for ``traci``/``libsumo``.
- :class:`MossMetrics` — backend for ``moss.Engine``.
- ``METRIC_NAMES`` — list of every supported metric name.

The concrete subclasses are imported eagerly so that their
``@register`` side-effects populate :data:`BaseMetrics.REGISTERED` and
:data:`METRIC_NAMES` before any caller reads them.
"""

from metrics.base import BaseMetrics, METRIC_NAMES, MetricSpec
from metrics.cityflow import CityFlowMetrics
from metrics.moss import MossMetrics
from metrics.sumo import SumoMetrics

__all__ = [
    "BaseMetrics",
    "CityFlowMetrics",
    "MossMetrics",
    "SumoMetrics",
    "METRIC_NAMES",
    "MetricSpec",
]
