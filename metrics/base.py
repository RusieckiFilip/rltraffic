"""
Base class for traffic-signal-control metrics.

Defines the abstract :class:`BaseMetrics` and a metric registry shared by
all simulator-specific subclasses (``CityFlowMetrics``, ``SumoMetrics``,
``MossMetrics``).

Design summary
--------------
A user opts in to a subset of metrics by name when constructing the
metrics object (or, indirectly, the env).  Only those metrics are
computed; everything else is skipped.  Episode-level accumulators
(running waiting times, completed-vehicle records, ...) are activated
on demand based on which metrics are requested.

Each metric is computed from one or more **primitives** -- raw engine
calls such as ``vehicle_speeds`` or ``lane_waiting_vehicle_count``.
Primitives are evaluated lazily and memoised in
:attr:`BaseMetrics._step_cache` until the next :meth:`update` clears it,
so two metrics that share a primitive only pay the engine call once per
step.

A metric may additionally (or exclusively) provide a **per-intersection**
implementation registered with :func:`register_local`.  It returns
``{intersection_id: value}`` and is exposed through
:meth:`BaseMetrics.get_local` / :meth:`BaseMetrics.compute_all_local`,
which downstream code (local reward shaping, per-intersection info)
consumes as ``dict[intersection_id, dict[metric_name, float]]``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class MetricSpec:
    """Descriptor stored in each subclass's ``REGISTERED`` dict."""

    name: str
    compute: Callable[["BaseMetrics"], float] | None
    """``compute(self) -> float`` — invoked when the metric is requested.
    ``None`` for metrics that only have a per-intersection variant."""

    episode: bool = False
    """If True, episode-level accumulators must be enabled when this
    metric is requested."""

    compute_local: Callable[["BaseMetrics"], "dict[str, float]"] | None = None
    """``compute_local(self) -> {intersection_id: value}`` — optional
    per-intersection variant of the metric."""


# Marker attributes set by the ``@register`` / ``@register_local``
# decorators on metric methods.  Each :class:`BaseMetrics` subclass scans
# its methods (own + inherited) for these markers in ``__init_subclass__``
# and builds its private ``REGISTERED`` dict, so two subclasses can share
# a metric *name* with different implementations.
_METRIC_MARKER = "_metric_spec"
_METRIC_LOCAL_MARKER = "_metric_local_spec"


# Names of every metric supported by the package.  Populated by the
# ``@register`` decorator across all subclass module imports.
METRIC_NAMES: list[str] = []


def register(name: str, *, episode: bool = False):
    """Decorator: tag a method as the implementation of metric *name*.

    Used inside :class:`BaseMetrics` subclass bodies::

        @register("queue")
        def _queue(self) -> float:
            ...
    """

    def _wrap(fn: Callable[["BaseMetrics"], float]):
        setattr(fn, _METRIC_MARKER, (name, episode))
        if name not in METRIC_NAMES:
            METRIC_NAMES.append(name)
        return fn

    return _wrap


def register_local(name: str, *, episode: bool = False):
    """Decorator: tag a method as the *per-intersection* implementation
    of metric *name*.

    The method must return ``{intersection_id: value}``.  A metric may
    have both a global (``@register``) and a local implementation under
    the same name; they are merged into one :class:`MetricSpec`.
    """

    def _wrap(fn: Callable[["BaseMetrics"], dict[str, float]]):
        setattr(fn, _METRIC_LOCAL_MARKER, (name, episode))
        if name not in METRIC_NAMES:
            METRIC_NAMES.append(name)
        return fn

    return _wrap


class BaseMetrics(abc.ABC):
    """Common metric-management plumbing for all simulator backends."""

    # Per-subclass dict, populated by :meth:`__init_subclass__`.
    REGISTERED: dict[str, MetricSpec] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # dir() covers inherited methods too, so global and local
        # implementations may live on different classes in the MRO
        # (e.g. generic locals on BaseMetrics, globals on the backend).
        global_impls: dict[str, tuple[Callable[..., Any], bool]] = {}
        local_impls: dict[str, tuple[Callable[..., Any], bool]] = {}
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name, None)
            if not callable(attr):
                continue
            marker = getattr(attr, _METRIC_MARKER, None)
            if marker is not None:
                global_impls[marker[0]] = (attr, marker[1])
            local_marker = getattr(attr, _METRIC_LOCAL_MARKER, None)
            if local_marker is not None:
                local_impls[local_marker[0]] = (attr, local_marker[1])

        registered: dict[str, MetricSpec] = dict(getattr(cls, "REGISTERED", {}))
        for metric_name in set(global_impls) | set(local_impls):
            compute, g_episode = global_impls.get(metric_name, (None, False))
            compute_local, l_episode = local_impls.get(
                metric_name, (None, False)
            )
            registered[metric_name] = MetricSpec(
                name=metric_name,
                compute=compute,
                episode=g_episode or l_episode,
                compute_local=compute_local,
            )
        cls.REGISTERED = registered

    def __init__(
        self,
        engine: Any,
        intersections: list[Any],
        metric_names: Iterable[str] | None = None,
        *,
        delta_time: float,
    ) -> None:
        self._engine = engine
        self._intersections = intersections
        self._delta_time = float(delta_time)

        requested = list(metric_names or [])
        for n in requested:
            if n not in self.REGISTERED:
                raise ValueError(
                    f"Unknown metric '{n}'. "
                    f"Available: {sorted(self.REGISTERED.keys())}"
                )
        self._requested: list[str] = requested
        self._requested_set: set[str] = set(requested)

        # Lazily-evaluated primitives, cleared every step.
        self._step_cache: dict[str, Any] = {}

        # Cross-step bookkeeping (completed vehicles, running sums, ...).
        self._episode: dict[str, Any] = {}

        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all per-episode and per-step state.

        Called by the env when a new episode begins.
        """
        self._step_cache.clear()
        self._episode.clear()
        self._init_episode_state()

    def pre_step(self) -> None:
        """Called by the env *before* applying the action.

        Subclasses use this to snapshot per-vehicle state needed by
        metrics that compare ``t`` and ``t + delta``
        (``calculate_average_delta_of_delays_after_action``).
        """
        if self.requires("calculate_average_delta_of_delays_after_action"):
            self._snapshot_pre_action()

    def on_sim_step(self) -> None:
        """Called by the env after each *individual* simulation step.

        An ``update()`` window spans ``delta_time`` simulation steps.
        Backends whose engine only exposes per-step events (e.g. SUMO's
        ``getArrivedIDList``, which reports only the last step's
        arrivals) override this to accumulate those events so nothing is
        lost between :meth:`update` calls.  Default: no-op.
        """

    def update(self) -> None:
        """Called by the env *after* the simulation has advanced.

        Refreshes the per-step cache and runs any episode-level hooks
        that integrate per-step values into running accumulators.
        """
        self._step_cache.clear()
        self._run_step_hooks()

    def warmup(self) -> None:
        """Called by the env once after :meth:`reset`, before the first
        observation, without advancing the simulation.

        Refreshes only the per-step caches that observations read and
        that are normally produced by :meth:`update`, so the first
        observation of an episode reflects the real ``t=0`` sim state
        instead of an empty cache.  Unlike :meth:`update` it must NOT
        touch episode-level accumulators, since no ``delta_time`` has
        elapsed yet.  Default: no-op.
        """

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    @property
    def requested(self) -> list[str]:
        """Names of metrics this instance was configured to report."""
        return list(self._requested)

    def requires(self, name: str) -> bool:
        """Whether the metric *name* was requested by the user."""
        return name in self._requested_set

    def get(self, name: str) -> float:
        """Compute and return a single metric by name.

        Works even for non-requested metrics, but in that case the
        episode-level data needed to compute it may be missing and the
        result will reflect only what was tracked.
        """
        if name not in self.REGISTERED:
            raise KeyError(
                f"Unknown metric '{name}'. "
                f"Available: {sorted(self.REGISTERED.keys())}"
            )
        spec = self.REGISTERED[name]
        if spec.compute is None:
            raise KeyError(
                f"Metric '{name}' only has a per-intersection "
                f"implementation; use get_local()."
            )
        return float(spec.compute(self))

    def has_local(self, name: str) -> bool:
        """Whether metric *name* has a per-intersection implementation."""
        spec = self.REGISTERED.get(name)
        return spec is not None and spec.compute_local is not None

    def get_local(self, name: str) -> dict[str, float]:
        """Compute metric *name* per intersection: ``{ix_id: value}``."""
        if name not in self.REGISTERED:
            raise KeyError(
                f"Unknown metric '{name}'. "
                f"Available: {sorted(self.REGISTERED.keys())}"
            )
        spec = self.REGISTERED[name]
        if spec.compute_local is None:
            raise KeyError(
                f"Metric '{name}' has no per-intersection implementation."
            )
        return {
            str(ix_id): float(value)
            for ix_id, value in spec.compute_local(self).items()
        }

    def compute_all(self) -> dict[str, float]:
        """Return ``{name: value}`` for every requested metric.

        Memoised in the per-step cache so callers in the same step
        (e.g. ``_get_reward`` and ``_get_info``) share one computation.
        """
        cached = self._step_cache.get("__compute_all__")
        if cached is None:
            cached = {
                n: self.get(n)
                for n in self._requested
                if self.REGISTERED[n].compute is not None
            }
            self._step_cache["__compute_all__"] = cached
        return dict(cached)

    def compute_all_local(self) -> dict[str, dict[str, float]]:
        """Return ``{ix_id: {name: value}}`` for every requested metric
        that has a per-intersection implementation.

        Every controlled intersection appears in the result (possibly
        with an empty dict).  Memoised per step like :meth:`compute_all`.
        """
        cached = self._step_cache.get("__compute_all_local__")
        if cached is None:
            cached = {str(ix.id): {} for ix in self._intersections}
            for n in self._requested:
                if self.REGISTERED[n].compute_local is None:
                    continue
                for ix_id, value in self.get_local(n).items():
                    cached.setdefault(ix_id, {})[n] = value
            self._step_cache["__compute_all_local__"] = cached
        return {ix_id: dict(values) for ix_id, values in cached.items()}

    @abc.abstractmethod
    def lane_waiting_vehicle_count(self) -> dict[str, int]:
        """Per-lane waiting/halting vehicle counts for current step."""

    @abc.abstractmethod
    def lane_vehicle_count(self) -> dict[str, int]:
        """Per-lane vehicle counts for current step."""

    # ------------------------------------------------------------------
    # Generic per-intersection metrics (shared by all backends)
    # ------------------------------------------------------------------

    @register_local(
        "number_of_all_halting_vehicles_for_the_last_time_step_in_simulation"
    )
    def _halting_vehicles_local(self) -> dict[str, float]:
        """Halting vehicles on each intersection's *incoming lanes*.

        Note the deliberately narrower scope than the global variant,
        which counts every halting vehicle in the simulation: a local
        reward must only depend on traffic the intersection controls.
        """
        waiting = self.lane_waiting_vehicle_count()
        return {
            str(ix.id): float(
                sum(int(waiting.get(lid, 0)) for lid in ix.incoming_lanes)
            )
            for ix in self._intersections
        }

    @register_local("average_intersection_pressure")
    def _intersection_pressure_local(self) -> dict[str, float]:
        """Pressure (incoming minus outgoing vehicle count) per
        intersection."""
        counts = self.lane_vehicle_count()
        pressures: dict[str, float] = {}
        for ix in self._intersections:
            incoming = sum(int(counts.get(lid, 0)) for lid in ix.incoming_lanes)
            outgoing = sum(int(counts.get(lid, 0)) for lid in ix.outgoing_lanes)
            pressures[str(ix.id)] = float(incoming - outgoing)
        return pressures

    @register("average_intersection_pressure")
    def _average_intersection_pressure(self) -> float:
        pressures = self._intersection_pressure_local()
        if not pressures:
            return 0.0
        return float(sum(pressures.values()) / len(pressures))

    # ------------------------------------------------------------------
    # Helpers used by subclasses
    # ------------------------------------------------------------------

    def _cached(self, key: str, fn: Callable[[], Any]) -> Any:
        """Memoise *fn()* under *key* for the lifetime of the step."""
        if key not in self._step_cache:
            self._step_cache[key] = fn()
        return self._step_cache[key]

    def _requires_any(self, *names: str) -> bool:
        """True if any of the named metrics was requested."""
        return any(n in self._requested_set for n in names)

    # ------------------------------------------------------------------
    # Subclass extension points
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _init_episode_state(self) -> None:
        """Populate :attr:`_episode` with the accumulators this backend
        needs based on which metrics were requested."""

    @abc.abstractmethod
    def _run_step_hooks(self) -> None:
        """Update episode accumulators from the freshly-cached primitives.

        Called from :meth:`update` after the step cache has been cleared
        (so primitive accessors will fetch new data on demand).
        """

    @abc.abstractmethod
    def _snapshot_pre_action(self) -> None:
        """Snapshot vehicle state right before an action is applied.

        Stored in :attr:`_episode` under a backend-defined key for use
        by ``calculate_average_delta_of_delays_after_action``.
        """
