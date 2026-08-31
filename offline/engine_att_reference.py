"""P8.4b Gate 0: an independent reference for CityFlow's own average travel time.

⚠️ SKELETON.  Every function below raises :class:`NotImplementedError`; the constants are the
REGISTERED declarations from ``docs/plans/p8.4b-g0.md`` and ``BRIEF_32`` Amendments A and B and are
real.  Tests are written against this surface first, so each one fails for its own reason rather
than sharing a single import error.

Artifact format version: ``p8.4b-g0-reference/1.0``.

WHAT THIS MODULE DECIDES, AND WHAT IT DOES NOT
----------------------------------------------
``docs/reviews/T1-metric-ground-truth.md`` established that ``metrics/cityflow.py``'s
``average_travel_time`` and CityFlow's ``Engine::getAverageTravelTime`` are different quantities.
``PREREGISTRATION`` **A11** registered ``Rule R``: which definition the paper's claims rest on is
decided per scenario by an ENGINE-SEMANTICS GATE, and **A12** replaced that gate's criterion (3).
This module builds the gate's instrument and evaluates the registered criteria.

⛔ **It issues no verdict on which metric is primary** (``BRIEF_32`` section 6).  ``Rule R`` decides
that from these numbers; this module reports measurements and criterion outcomes.

THE ENGINE'S ACCOUNTING, READ FROM SOURCE
-----------------------------------------
``CityFlow/src/engine/engine.cpp:682-691``::

    getAverageTravelTime() = (cumulativeTravelTime + SUM_pool (currentTime - enterTime))
                           / (finishedVehicleCnt + |vehiclePool|)

with **no filter** on the pool -- unlike ``getRunningVehicles`` (``:780-789``), which keeps
``isReal() && (includeWaiting || isRunning())``.  Every CityFlow config in this repo sets
``laneChange`` false (11 explicitly, ``cologne1``/``cologne3`` by ``engine.cpp:53``'s default), so
there are no non-``isReal()`` shadow vehicles and ``get_vehicles(include_waiting=True)`` **is** the
whole ``vehiclePool``.  ``tests/test_engine_att_reference.py`` re-asserts that on all 13 configs
rather than inheriting it.

THE ALIGNMENT CONVENTION -- THE ONE THING TO READ BEFORE TRUSTING A NUMBER HERE
------------------------------------------------------------------------------
Snapshots are taken **immediately after each** ``Engine::next_step()``, so a snapshot labelled ``t``
is the pool while ``engine.step == t`` and ``getCurrentTime() == t * interval``.  Because
``Flow::nextStep`` creates a vehicle at the TOP of ``nextStep`` while ``step += 1`` is its LAST
statement (``engine.cpp:592``), a vehicle created during the call ``k -> k+1`` has
``enterTime == k * interval`` and first appears in the snapshot labelled ``k + 1``::

    first_seen(v) = enterTime(v) + interval                                        (A)

Because a finishing vehicle is credited ``getCurrentTime() - enterTime`` inside ``updateLocation()``
-- also before ``step += 1`` -- and is erased from the pool in the same call, it is present in the
snapshot labelled ``m`` and absent from ``m + 1``, so its engine credit is
``last_seen(v) - enterTime(v)``.  A vehicle still pooled at the final snapshot ``S`` contributes
``S * interval - enterTime(v)`` and has ``last_seen(v) = S * interval``.  **Both branches collapse to
one expression** (``BRIEF_32`` Amendment A6 approves the collapse; criterion 1 arbitrates it)::

    contribution(v) = last_seen(v) - first_seen(v) + interval,  for EVERY v         (C)
    att_reference   = SUM contribution(v) / |{v ever seen}|

FOUR RECONSTRUCTIONS, AND WHY THERE ARE FOUR
--------------------------------------------
* ``engine_population`` -- (C) over ``get_vehicles(include_waiting=True)``.  The quantity criterion 1
  compares against ``get_average_travel_time()``.
* ``entered_running`` -- (C) over ``get_vehicles(include_waiting=False)``, admitting a vehicle at the
  first second it is RUNNING.  This is ``BRIEF_32`` section 4's entered-only variant: our metric's
  population **and** our metric's clock.
* ``entered_population`` -- the same population as ``entered_running`` but with the POOL clock.
  Isolates the population effect alone.  **A12 (3a) justifies bit-identity at ``never_entered == 0``
  by "the two populations are the same set of vehicles", which is an argument about the population
  and not about the clock; this variant is that argument's quantity.**  Measured on a real episode:
  ``entered_running`` differs from the engine value by minus the mean admission latency, which is
  0.670160 s on hz1x1 maxpressure draw 1000 (465 of 1813 vehicles delayed, maximum 11 s), while
  ``entered_population`` is bit-identical to it.  See ``docs/plans/p8.4b-g0.md`` section 8.
* ``metric_cadence`` -- A12 (3c): an independent replay of ``metrics/cityflow.py``'s algorithm on the
  decision-grid subset of the running-population snapshots.  **Required, reported, NOT gating.**

⚠️ **A vehicle's ``last_seen`` is the same in both streams and this is not an assumption.**
``setRunning(true)`` has exactly one call site (``engine.cpp:508``, in ``handleWaiting``) and nothing
ever sets it back, so a vehicle is "running" from admission until it leaves the pool.  The entire
difference between the two streams is on the FIRST observation.

THE IMPORT FENCE -- A11's INDEPENDENCE CLAUSE, MECHANICALLY ENFORCED
--------------------------------------------------------------------
A11 requires the reconstruction to import **nothing** from ``metrics/`` and nothing from
``offline/admission_probe.py``.  ``BRIEF_32`` Amendment A3 scopes that to the RECONSTRUCTION: the
comparison harness may import both, because criteria 2 and 3 are *defined* as comparisons against the
probe.  The split is enforced by an AST walk over :data:`RECONSTRUCTION_SURFACE` in
``tests/test_engine_att_reference.py``, which ships a positive control.

⚠️ **Disclosed rather than hidden:** :class:`PerSecondEngineObserver` inherits
``CityFlowEnv._create_metrics`` (``envs/cityflow_env.py:232-241``), which constructs
``metrics.CityFlowMetrics``.  That is the *env's* behaviour and is required for ``att_ours`` to exist
at all.  **The reconstruction's arithmetic never reads the metrics object**, and the fence test
covers the reconstruction surface, not the env's inherited machinery.

THE FILESYSTEM-MUTATION BARRIER
-------------------------------
``output/`` in the main tree holds every checkpoint under nine manifests and is the only copy; it is
gitignored and there is no backup.  Every write goes through ``tier_sweep.assert_writable`` with each
sibling ``output/*`` directory passed as a protected root.  The whole campaign and the criteria
evaluation complete before the first byte is written, and a refused destination creates no directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "BEHAVIOUR_METHOD",
    "C1_TOLERANCE",
    "C3C_TOLERANCE",
    "C4_MIN_DRAWS",
    "C4_MIN_TIERS",
    "DETERMINISTIC_ANCHOR_TIERS",
    "EXTREME_EPISODES",
    "GATE_DRAWS",
    "GATE_SCENARIOS",
    "GATE_SEED",
    "GATE_TIERS",
    "RECONSTRUCTION_SURFACE",
    "AdmissionLatency",
    "EpisodeReconstruction",
    "ExtremeEpisode",
    "GateCell",
    "GateEpisode",
    "GateScenario",
    # NOT "PerSecondEngineObserver": the class is built lazily by observer_env_class() so that
    # importing this module needs no native cityflow, so it is not a module attribute.
    "ReconstructedAtt",
    "ScenarioCriteria",
    "VehicleWindows",
    "admission_latency",
    "build_factory",
    "build_parser",
    "cell_file_name",
    "cell_files",
    "default_work_dir",
    "env_settings_for",
    "episode_from_record",
    "evaluate_scenario",
    "gate_artifact",
    "gate_cells",
    "gate_episode",
    "main",
    "make_observer_env",
    "manifest_checkpoint",
    "metric_cadence_att",
    "reconstruct_att",
    "reconstruct_episode",
    "run_cells",
    "seeds_for",
    "thread_regime",
    "tier_corpus_dirs",
]

ARTIFACT_FORMAT_VERSION = "p8.4b-g0-reference/1.0"

#: A11 criterion 1's registered tolerance.  ``BRIEF_32`` section 5: the smallest gap between the two
#: ATT definitions anywhere in the 1,870 cells of ``docs/data/p8_4a_admission.json`` is 0.031699, so
#: 1e-4 sits 317x below the smallest semantic difference ever measured here and cannot mistake one
#: for agreement.  A deviation between 1e-4 and 1e-2 is a FAIL that must be root-caused.
C1_TOLERANCE = 1e-4

#: A12 (3c)'s tolerance.  (3c) is REQUIRED and REPORTED but explicitly **NOT GATING**: a surprise
#: about ``att_ours`` is escalated as a new finding and never folded into the gate's outcome.
C3C_TOLERANCE = 1e-4

#: A11 criterion 4's registered coverage floor: ">=7 behaviour tiers x >=3 draws".
C4_MIN_TIERS = 7
C4_MIN_DRAWS = 3

#: The corpus's seven behaviour tiers (``BRIEF_32`` Amendment A1).  A11's "seven behaviour tiers"
#: means these -- the ones ``datasets_v11/`` actually carries per scenario -- and never
#: ``admission_probe``'s declared arm set, which has five on hz1x1 and two on grid4x4.
GATE_TIERS: tuple[str, ...] = (
    "fixedtime",
    "mappo060",
    "mappo200",
    "mappo500",
    "mappo1000",
    "maxpressure",
    "random",
)

#: The three held-out draws every tier is rolled over.  A subset of ``admission_probe.PROBE_DRAWS``.
GATE_DRAWS: tuple[int, ...] = (1000, 1001, 1002)

#: The seed slot of every non-deterministic tier, from ``admission_probe.PROBE_SEEDS``.
GATE_SEED = 101

#: The tiers ``datasets_v11/`` collected once per training seed.  Everything else was collected
#: once.  Verified on disk this session for all 14 ``(scenario, tier)`` groups.
PER_SEED_TIERS: frozenset[str] = frozenset({"mappo060", "mappo200", "mappo500", "mappo1000"})

#: The training seeds those tiers were collected under, from ``admission_probe.PROBE_SEEDS``.
CORPUS_SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)

#: Tiers whose behaviour policy is deterministic, so their slot is a single ``None``.  Mirrors
#: ``admission_probe.DETERMINISTIC_ANCHOR_TIERS`` exactly.
DETERMINISTIC_ANCHOR_TIERS: frozenset[str] = frozenset({"maxpressure", "fixedtime"})

#: The arm name of a tier's own collecting policy, matching ``admission_probe.BEHAVIOUR_METHOD``.
BEHAVIOUR_METHOD = "behaviour"


@dataclass(frozen=True)
class GateScenario:
    """One gated scenario: how to find its draws, its corpus directories and its env id."""

    name: str
    scenario_key: str
    scenario_id: str
    corpus_prefix: str


#: The two scenarios ``Rule R`` binds.  ``cf_cologne3`` is OUT OF SCOPE (``BRIEF_32`` section 5):
#: only 11 draws are materialised and it is absent from ``p8_4a_admission.json`` entirely.
GATE_SCENARIOS: Mapping[str, GateScenario] = {
    "hz1x1": GateScenario(
        name="hz1x1",
        scenario_key="cityflow1x1",
        scenario_id="cityflow1x1",
        corpus_prefix="cf_hz1x1",
    ),
    "grid4x4": GateScenario(
        name="grid4x4",
        scenario_key="cityflow_grid4x4",
        scenario_id="cityflow_grid4x4",
        corpus_prefix="cf_grid4x4",
    ),
}


@dataclass(frozen=True)
class ExtremeEpisode:
    """An episode included because it is an ``entered_fraction`` extreme, not because of its tier.

    ``BRIEF_32`` Amendment A2: the extremes exist to stress the instrument where censoring is
    largest, and whether the policy is a behaviour tier or a learned arm is irrelevant to that.
    ``entered_fraction`` is the value recorded in ``docs/data/p8_4a_admission.json`` at declaration
    time; it is carried for traceability and is **not** used as a threshold.
    """

    scenario: str
    tier: str
    method: str
    seed: int
    draw_id: int
    entered_fraction: float
    which: str


#: The minimum and maximum ``entered_fraction`` episodes of each scenario, enumerated from all 1,870
#: rows of ``docs/data/p8_4a_admission.json``.  The maximum is 1.0 with 709 ties on hz1x1 and 633 on
#: grid4x4; the tie-break -- lowest ``(arm, seed, draw_id)`` lexicographically -- was registered in
#: ``docs/plans/p8.4b-g0.md`` before the outcome was looked at and approved by Amendment A2.
EXTREME_EPISODES: Mapping[str, tuple[ExtremeEpisode, ...]] = {
    "hz1x1": (
        ExtremeEpisode(
            scenario="hz1x1",
            tier="random",
            method="bc_top10",
            seed=303,
            draw_id=1009,
            entered_fraction=0.6190476190476191,
            which="min",
        ),
        ExtremeEpisode(
            scenario="hz1x1",
            tier="mappo1000",
            method="bc",
            seed=101,
            draw_id=1000,
            entered_fraction=1.0,
            which="max",
        ),
    ),
    "grid4x4": (
        ExtremeEpisode(
            scenario="grid4x4",
            tier="mappo1000",
            method="bc_top10",
            seed=404,
            draw_id=1001,
            entered_fraction=0.9441548771407298,
            which="min",
        ),
        ExtremeEpisode(
            scenario="grid4x4",
            tier="mappo1000",
            method="bc",
            seed=101,
            draw_id=1000,
            entered_fraction=1.0,
            which="max",
        ),
    ),
}

#: The names that make up LAYER A -- the reconstruction.  A11 forbids these from importing anything
#: from ``metrics/`` or from ``offline/admission_probe.py``; the harness below them may.  The
#: allowlist is data so the fence test can walk it, and the test asserts it is non-empty.
RECONSTRUCTION_SURFACE: tuple[str, ...] = (
    "VehicleWindows",
    "ReconstructedAtt",
    "AdmissionLatency",
    "EpisodeReconstruction",
    "EngineObservationRecorder",
    "reconstruct_att",
    "metric_cadence_att",
    "admission_latency",
    "reconstruct_episode",
    "observer_env_class",
    "make_observer_env",
)


# ----------------------------------------------------------------------
# LAYER A -- the reconstruction.  Imports stdlib, numpy, envs and experiments only.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class VehicleWindows:
    """When each vehicle id was first and last observed, in engine time.

    Both maps are keyed by vehicle id and hold the engine's own ``get_current_time()`` values, so a
    disagreement between our model of ``step`` and the engine's would surface rather than be assumed
    away.  The two must have identical key sets; :func:`reconstruct_att` refuses otherwise.
    """

    first_seen: Mapping[str, float]
    last_seen: Mapping[str, float]

    def ids(self) -> frozenset[str]:
        """Every vehicle id ever observed in this stream."""
        return frozenset(self.first_seen)

    def restricted_to(self, ids: Sequence[str] | frozenset[str] | set[str]) -> "VehicleWindows":
        """This stream's clocks restricted to *ids*, refusing an id this stream never saw."""
        wanted = frozenset(ids)
        unknown = sorted(wanted - frozenset(self.first_seen))
        if unknown:
            raise ValueError(
                f"{len(unknown)} of the requested ids were never observed in this stream "
                f"(first: {unknown[:3]}); restricting to an id the stream never saw would "
                "fabricate a population rather than select one"
            )
        ordered = [vid for vid in self.first_seen if vid in wanted]
        return VehicleWindows(
            first_seen={vid: self.first_seen[vid] for vid in ordered},
            last_seen={vid: self.last_seen[vid] for vid in ordered},
        )


@dataclass(frozen=True)
class ReconstructedAtt:
    """One reconstruction's value, with the sum and count it came from.

    ``total`` and ``n_ids`` are carried so a caller can re-derive ``value`` by a second route
    instead of trusting it, which is what the tests do.
    """

    value: float
    total: float
    n_ids: int


@dataclass(frozen=True)
class AdmissionLatency:
    """How long vehicles sat in a lane's insertion buffer before entering, in engine seconds.

    ``first_running_seen(v) - first_pooled_seen(v)``, over the vehicles that ever ran.  This is the
    term ``docs/plans/p8.4b-g0.md`` section 8.3 identifies as the reason A12 (3a)'s bit-identity does
    not hold under ``BRIEF_32`` section 4's entered-only definition, and it is reported per episode.
    """

    mean: float
    maximum: float
    n_delayed: int
    n_ids: int


@dataclass(frozen=True)
class EpisodeReconstruction:
    """Every reconstruction of one episode, plus the observation grid they were built on."""

    engine_population: ReconstructedAtt
    entered_running: ReconstructedAtt
    entered_population: ReconstructedAtt
    metric_cadence: ReconstructedAtt
    latency: AdmissionLatency
    interval: float
    n_observations: int


def reconstruct_att(windows: VehicleWindows, *, interval: float) -> ReconstructedAtt:
    """Equation (C): ``mean(last_seen - first_seen + interval)`` over every observed id.

    Refuses a non-positive *interval*, a key-set mismatch between the two maps, and any vehicle
    whose ``last_seen`` precedes its ``first_seen`` -- each of which would make the mean a plausible
    number computed from a broken observation stream.  An empty stream returns ``value = 0.0``,
    mirroring ``Engine::getAverageTravelTime``'s own ``n == 0 ? 0`` branch (``engine.cpp:690``).
    """
    step = float(interval)
    if not step > 0.0:
        raise ValueError(
            f"interval must be positive, got {interval!r}; a non-positive interval would rescale "
            "every contribution silently instead of refusing"
        )
    first = windows.first_seen
    last = windows.last_seen
    if set(first) != set(last):
        only_first = sorted(set(first) - set(last))
        only_last = sorted(set(last) - set(first))
        raise ValueError(
            f"the first-seen and last-seen maps observed different vehicle ids "
            f"({len(only_first)} seen only first, {len(only_last)} seen only last; "
            f"{only_first[:3]} / {only_last[:3]}). They come from ONE observation stream and can "
            "only disagree if that stream was corrupted"
        )
    total = 0.0
    for vid in first:
        span = float(last[vid]) - float(first[vid])
        if span < 0.0:
            raise ValueError(
                f"{vid} was last seen before it was first seen ({last[vid]!r} < {first[vid]!r}); "
                "the observation grid is monotonic, so a sound stream cannot produce this"
            )
        total += span + step
    count = len(first)
    if count == 0:
        return ReconstructedAtt(value=0.0, total=0.0, n_ids=0)
    return ReconstructedAtt(value=total / count, total=total, n_ids=count)


def metric_cadence_att(
    snapshots: Sequence[tuple[float, frozenset[str]]], *, delta_time: float
) -> ReconstructedAtt:
    """A12 (3c): replay ``metrics/cityflow.py``'s algorithm on the decision-grid snapshots.

    An INDEPENDENT implementation of the published algorithm, never an import of it -- that is what
    makes agreement with ``att_ours`` evidence rather than a tautology.  The algorithm, read from
    ``metrics/cityflow.py:166-253``:

    * ``prev_active`` starts empty (``:109``; ``warmup()`` is a no-op that ``CityFlowMetrics`` does
      not override), so the first grid point treats every active vehicle as new;
    * a newly active vehicle gets ``depart = t - delta_time / 2`` -- the window-midpoint estimator;
    * completion is detected as ``prev_active - active`` at the END of a window, crediting
      ``max(0, t - depart)``;
    * the reported value averages those credits together with ``max(0, t - depart)`` over the
      vehicles still active at the final grid point.

    Refuses a non-positive *delta_time*, a non-monotonic time sequence and an empty snapshot list.
    """
    window = float(delta_time)
    if not window > 0.0:
        raise ValueError(
            f"delta_time must be positive, got {delta_time!r}; the midpoint estimator divides by it"
        )
    rows = [(float(t), frozenset(active)) for t, active in snapshots]
    if not rows:
        raise ValueError(
            "there are no decision-grid snapshots to replay, so this would report 0.0 for an "
            "episode that produced no snapshot; an empty grid is a defect in the observer, not a "
            "quiet zero"
        )
    for (earlier, _), (later, _) in zip(rows, rows[1:]):
        if later <= earlier:
            raise ValueError(
                f"the decision grid is not increasing: {later!r} follows {earlier!r}. The replay "
                "diffs consecutive windows, so an out-of-order grid would invent completions"
            )

    depart: dict[str, float] = {}
    completed: list[float] = []
    previous: frozenset[str] = frozenset()
    for now, active in rows:
        for vid in active - previous:
            depart[vid] = now - window / 2.0
        for vid in previous - active:
            completed.append(max(0.0, now - depart.get(vid, now)))
        previous = active

    horizon, final_active = rows[-1]
    total = float(sum(completed))
    count = len(completed)
    for vid in final_active:
        departed_at = depart.get(vid)
        if departed_at is None:
            continue
        total += max(0.0, horizon - departed_at)
        count += 1
    if count == 0:
        return ReconstructedAtt(value=0.0, total=0.0, n_ids=0)
    return ReconstructedAtt(value=total / count, total=total, n_ids=count)


def admission_latency(pool: VehicleWindows, running: VehicleWindows) -> AdmissionLatency:
    """``first_running_seen - first_pooled_seen`` over the ids that ever ran.

    Refuses a running id the pool stream never saw, and a negative latency: the running population is
    a subset of the pool population at every instant (``engine.cpp:780-789``), so either would mean
    the two streams were not observed on the same grid.
    """
    gaps: list[float] = []
    for vid, admitted_at in running.first_seen.items():
        if vid not in pool.first_seen:
            raise ValueError(
                f"{vid} was observed running but is absent from the pool stream; the running "
                "population is a subset of the pool at every instant, so the two sets were not "
                "read at the same instant"
            )
        gap = float(admitted_at) - float(pool.first_seen[vid])
        if gap < 0.0:
            raise ValueError(
                f"{vid} was observed running at {admitted_at!r} before it was pooled at "
                f"{pool.first_seen[vid]!r}; a vehicle enters the pool before it can be admitted"
            )
        gaps.append(gap)
    if not gaps:
        return AdmissionLatency(mean=0.0, maximum=0.0, n_delayed=0, n_ids=0)
    return AdmissionLatency(
        mean=float(sum(gaps)) / len(gaps),
        maximum=float(max(gaps)),
        n_delayed=sum(1 for gap in gaps if gap > 0.0),
        n_ids=len(gaps),
    )


class EngineObservationRecorder:
    """The per-second observation state, kept OUTSIDE the env so it is testable without an engine.

    ⚠️ **The grain is the whole point.**  ``delta_time`` is 10 s, so an env-step observation grid
    quantises enter and leave times to 10 s -- which is T1's M1 defect, i.e. exactly the error this
    reference exists to be free of.  The env subclass built by :func:`observer_env_class` calls
    :meth:`observe` after **each** ``next_step()``.

    Times are the engine's own ``get_current_time()`` values, passed in by the caller, so a
    disagreement between our model of ``step`` and the engine's surfaces rather than being assumed
    away.  A decision-grid snapshot is kept only when ``engine_time`` is an exact multiple of
    ``delta_time``; that subset is what A12 (3c) replays ``metrics/cityflow.py``'s algorithm over.
    """

    def __init__(self, *, delta_time: float) -> None:
        window = float(delta_time)
        if not window > 0.0:
            raise ValueError(
                f"a recorder needs a positive window, got delta_time={delta_time!r}; it selects "
                "the decision-grid subset A12 (3c) replays our own metric over"
            )
        self.delta_time = window
        self._pool_first: dict[str, float] = {}
        self._pool_last: dict[str, float] = {}
        self._running_first: dict[str, float] = {}
        self._running_last: dict[str, float] = {}
        self._grid: list[tuple[float, frozenset[str]]] = []
        self._times: list[float] = []

    def clear(self) -> None:
        """Drop every observation, so one recorder can serve consecutive episodes."""
        self._pool_first.clear()
        self._pool_last.clear()
        self._running_first.clear()
        self._running_last.clear()
        self._grid.clear()
        self._times.clear()

    def observe(
        self,
        *,
        engine_time: float,
        pooled_ids: Sequence[str],
        running_ids: Sequence[str],
    ) -> None:
        """Record one snapshot.

        Refuses a non-increasing *engine_time* and a running id absent from *pooled_ids*: the
        running population is a subset of the pool at every instant (``engine.cpp:780-789``), so
        either would mean the two sets were not read at the same instant.
        """
        now = float(engine_time)
        if self._times and now <= self._times[-1]:
            raise ValueError(
                f"engine time did not advance: this observation is at {now!r} and the previous one "
                f"was at {self._times[-1]!r}. Every snapshot is taken after a next_step(), so a "
                "repeated or receding time means the observer and the engine are out of step"
            )
        self._times.append(now)

        for vid in pooled_ids:
            if vid not in self._pool_first:
                self._pool_first[vid] = now
            self._pool_last[vid] = now
        for vid in running_ids:
            if self._pool_last.get(vid) != now:
                raise ValueError(
                    f"{vid} is running but not pooled at time {now!r}; get_vehicles(True) is a "
                    "superset of get_vehicles(False) at every instant, so the two sets were not "
                    "read at the same instant"
                )
            if vid not in self._running_first:
                self._running_first[vid] = now
            self._running_last[vid] = now

        if now % self.delta_time == 0.0:
            self._grid.append((now, frozenset(running_ids)))

    @property
    def pool_windows(self) -> VehicleWindows:
        """``get_vehicles(include_waiting=True)`` -- the whole ``vehiclePool``."""
        return VehicleWindows(first_seen=dict(self._pool_first), last_seen=dict(self._pool_last))

    @property
    def running_windows(self) -> VehicleWindows:
        """``get_vehicles(include_waiting=False)`` -- the admitted population."""
        return VehicleWindows(
            first_seen=dict(self._running_first), last_seen=dict(self._running_last)
        )

    @property
    def decision_grid_snapshots(self) -> tuple[tuple[float, frozenset[str]], ...]:
        """The admitted sets at times that are exact multiples of ``delta_time``, for (3c)."""
        return tuple(self._grid)

    @property
    def observation_times(self) -> tuple[float, ...]:
        """Every engine time at which a snapshot was taken, in order."""
        return tuple(self._times)

    @property
    def observed_interval(self) -> float:
        """The engine's ``interval``, MEASURED as the gap between consecutive observations.

        Derived from the observation stream rather than read from the config, so a config whose
        ``interval`` did not match the engine's behaviour would raise here instead of silently
        rescaling every contribution.  Refuses a non-uniform gap and fewer than two observations.
        """
        times = self._times
        if len(times) < 2:
            raise ValueError(
                f"measuring the interval needs at least two observations, got {len(times)}; "
                "the interval is derived from the engine's own clock, never read from a config"
            )
        gaps = {later - earlier for earlier, later in zip(times, times[1:])}
        if len(gaps) != 1:
            raise ValueError(
                f"the observation grid is not uniform: gaps {sorted(gaps)[:5]}. Equation (C) adds "
                "exactly one interval per vehicle, so a ragged grid would make every contribution "
                "wrong by a different amount"
            )
        return float(next(iter(gaps)))


#: Memoised by :func:`observer_env_class`, which builds the subclass on first use so that importing
#: this module never requires the native ``cityflow`` extension.
_OBSERVER_ENV_CLASS: type[Any] | None = None


def observer_env_class() -> type[Any]:
    """Build ``PerSecondEngineObserver``, a ``CityFlowEnv`` subclass, on first use.

    ⚠️ **The class is created lazily and this is deliberate.**  ``envs/cityflow_env.py:10`` imports
    the native ``cityflow`` module at import time, so binding the subclass at module scope would make
    importing this module impossible wherever the engine is not built -- including CI, where 32
    CityFlow-gated tests currently skip cleanly.  A skeleton whose import fails turns every test in
    the file into one shared ``ImportError`` instead of its own failure.

    The subclass overrides exactly two methods:

    * ``_simulate`` -- its own ``for`` loop over ``next_step()`` with an :meth:`observe` after each.
      It never delegates to ``super()._simulate(n)``, because ``_execute_phase_plans``
      (``envs/base_traffic_env.py:547-557``) calls it with variable segment durations.
    * ``reset`` -- clears the recorder, then defers to ``super()``.  Returns ``info`` ONLY.
    """
    global _OBSERVER_ENV_CLASS
    if _OBSERVER_ENV_CLASS is not None:
        return _OBSERVER_ENV_CLASS

    from envs.cityflow_env import CityFlowEnv

    class PerSecondEngineObserver(CityFlowEnv):
        """A ``CityFlowEnv`` that records the engine's vehicle-id sets once per simulation second.

        Format version: ``p8.4b-g0-reference/1.0``.  Alignment convention: a snapshot is taken
        IMMEDIATELY AFTER each ``next_step()``, so the snapshot labelled ``t`` is the pool while
        ``engine.step == t``, and ``first_seen(v) == enterTime(v) + interval``.  See the module
        docstring.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.recorder = EngineObservationRecorder(delta_time=float(self.delta_time))

        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            self.recorder.clear()
            return super().reset(seed=seed, options=options)

        def _simulate(self, num_steps: int) -> None:
            engine = self._eng
            if engine is None:
                raise RuntimeError(
                    "the CityFlow engine is not alive, so no step can be observed; the observer "
                    "never advances the simulation without recording it"
                )
            for _ in range(int(num_steps)):
                engine.next_step()
                self.recorder.observe(
                    engine_time=float(engine.get_current_time()),
                    pooled_ids=engine.get_vehicles(include_waiting=True),
                    running_ids=engine.get_vehicles(include_waiting=False),
                )

    _OBSERVER_ENV_CLASS = PerSecondEngineObserver
    return _OBSERVER_ENV_CLASS


def reconstruct_episode(recorder: EngineObservationRecorder) -> EpisodeReconstruction:
    """All four reconstructions of the episode *recorder* has just observed."""
    interval = recorder.observed_interval
    pool = recorder.pool_windows
    running = recorder.running_windows
    return EpisodeReconstruction(
        engine_population=reconstruct_att(pool, interval=interval),
        entered_running=reconstruct_att(running, interval=interval),
        entered_population=reconstruct_att(pool.restricted_to(running.ids()), interval=interval),
        metric_cadence=metric_cadence_att(
            recorder.decision_grid_snapshots, delta_time=recorder.delta_time
        ),
        latency=admission_latency(pool, running),
        interval=interval,
        n_observations=len(recorder.observation_times),
    )


def make_observer_env(config_path: str | Path, settings: Mapping[str, Any]) -> Any:
    """Build the observer env exactly as ``experiments.envs.make_env`` builds a ``CityFlowEnv``.

    ``make_env`` returns the frozen class and cannot return a subclass, so its CityFlow branch
    (``experiments/envs.py:84-113``) is mirrored here: the same seven ``common`` keys, the same
    optional ``metrics`` / ``obs_norm`` handling, the same ``thread_num``, and
    ``experiments.envs.phase_control_cls`` imported rather than a local mapping.
    ``tests/test_engine_att_reference.py`` asserts the two constructions agree field by field.
    """
    from experiments.envs import phase_control_cls

    common: dict[str, Any] = {
        "max_steps": settings["max_steps"],
        "delta_time": settings["delta_time"],
        "global_reward_fn": settings["global_reward_fn"],
        "local_reward_fn": settings["local_reward_fn"],
        "global_reward_weight": settings["global_reward_weight"],
        "phase_control_cls": phase_control_cls(settings["control_mode"]),
        "state_features": settings["state_features"],
    }
    if settings["metrics"] is not None:
        common["metrics"] = settings["metrics"]
    if settings["obs_norm"] is not None:
        common["obs_norm"] = settings["obs_norm"]
    return observer_env_class()(
        cityflow_config_path=str(config_path),
        thread_num=settings["thread_num"],
        **common,
    )


# ----------------------------------------------------------------------
# LAYER B -- the gate harness.  May import offline.admission_probe (Amendment A3).
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class GateCell:
    """One arm-seed-draw the gate rolls."""

    scenario: str
    tier: str
    method: str
    seed: int | None
    draw_id: int
    role: str

    @property
    def arm(self) -> str:
        """``"<method>@<tier>"`` -- the arm identity carried by every episode record."""
        return f"{self.method}@{self.tier}"


@dataclass(frozen=True)
class GateEpisode:
    """One episode: every reconstruction, both engine-side reads, and the admission counts."""

    scenario: str
    tier: str
    method: str
    arm: str
    seed: int | None
    draw_id: int
    role: str
    engine_seed: int
    att_reference_engine_population: float
    att_reference_entered_running: float
    att_reference_entered_population: float
    att_reference_metric_cadence: float
    att_engine_call: float
    att_ours: float
    n_reference_ids: int
    n_entered_ids: int
    created_from_flow: int
    entered: int
    never_entered: int
    admission_latency_mean: float
    admission_latency_max: float
    n_admission_delayed: int
    interval: float
    n_observations: int
    seconds: float
    seconds_rollout: float

    @property
    def deviation_c1(self) -> float:
        """Criterion 1: the ABSOLUTE gap between the reconstruction and the engine's own call."""
        return abs(float(self.att_reference_engine_population) - float(self.att_engine_call))

    @property
    def deviation_c3c(self) -> float:
        """A12 (3c): the absolute gap between the metric-cadence replay and ``att_ours``."""
        return abs(float(self.att_reference_metric_cadence) - float(self.att_ours))

    @property
    def difference_c3a_running(self) -> float:
        """A12 (3a)/(3b) as registered: the absolute gap to the ``entered_running`` variant."""
        return abs(
            float(self.att_reference_engine_population) - float(self.att_reference_entered_running)
        )

    @property
    def difference_c3a_population(self) -> float:
        """The same gap under the population-only reading (``docs/plans`` section 8.4 (b))."""
        return abs(
            float(self.att_reference_engine_population)
            - float(self.att_reference_entered_population)
        )

    @property
    def term_population(self) -> float:
        """Decomposition term 1: restrict the POPULATION, holding the pool clock fixed.

        ``entered_population - engine_population``.  Exactly zero when ``never_entered == 0``, which
        is what A12 (3a) asserts and what A13 scores it on.
        """
        return float(self.att_reference_entered_population) - float(
            self.att_reference_engine_population
        )

    @property
    def term_clock_origin(self) -> float:
        """Decomposition term 2: move the CLOCK ORIGIN, holding the population fixed.

        ``entered_running - entered_population`` -- minus the mean admission latency, i.e. the
        insertion-buffer wait the engine counts and our metric does not.
        """
        return float(self.att_reference_entered_running) - float(
            self.att_reference_entered_population
        )

    @property
    def term_cadence(self) -> float:
        """Decomposition term 3: the 10 s decision grid.

        ``att_ours - entered_running`` -- the midpoint-departure and end-of-window-completion
        estimators of ``metrics/cityflow.py``, isolated (T1's M1).
        """
        return float(self.att_ours) - float(self.att_reference_entered_running)

    @property
    def decomposition_residual(self) -> float:
        """``|(population + clock_origin + cadence) - (att_ours - att_engine_call)|``.

        Zero whenever criteria 1 and (3c) are exact, because the three terms telescope through the
        reconstructions and the two endpoints are then the engine's own call and our own metric.
        Reported per episode so the decomposition is checkable rather than asserted.
        """
        total = float(self.att_ours) - float(self.att_engine_call)
        return abs((self.term_population + self.term_clock_origin + self.term_cadence) - total)

    def as_record(self) -> dict[str, Any]:
        """The JSON row: every constructor field, then the derived quantities beside them."""
        record = {name: getattr(self, name) for name in _EPISODE_FIELD_NAMES}
        record.update(
            {
                "deviation_c1": self.deviation_c1,
                "deviation_c3c": self.deviation_c3c,
                "difference_c3a_running": self.difference_c3a_running,
                "difference_c3a_population": self.difference_c3a_population,
                "term_population": self.term_population,
                "term_clock_origin": self.term_clock_origin,
                "term_cadence": self.term_cadence,
                "decomposition_residual": self.decomposition_residual,
                "difference_ours_minus_engine": float(self.att_ours) - float(self.att_engine_call),
            }
        )
        return record


@dataclass(frozen=True)
class ScenarioCriteria:
    """A11's four criteria as amended by A12, scored on one scenario, with observed values.

    🔒 **(3a) and (3b) are scored on the POPULATION READING, ruled by the coordinator on 2026-08-31
    (Q7, option (b)) and registered as A13.**  The reading holds the CLOCK at the pool clock and
    varies only the POPULATION, which is what A12 (3a)'s own justifying clause -- *"the two
    populations are the same set of vehicles"* -- describes.  **Zero tolerance is retained.**

    ⚠️ **The other reading is still reported, on every scenario, in
    ``c3a_max_difference_running_reading`` / ``c3b_min_difference_running_reading``**: it is
    ``BRIEF_32`` section 4's entered-only variant, whose clock starts at ADMISSION and which
    therefore also carries the clock-origin term.  Measured at n=46, (3a) fails under that reading
    (0.738 hz1x1, 0.098 grid4x4) and is exactly 0.0 under the ruled one on all 32 qualifying
    episodes.  Both are kept so the ruling is checkable from the artifact alone.

    ``c3c`` is REQUIRED and REPORTED but explicitly NOT part of ``passed`` (A12).
    """

    scenario: str
    n_episodes: int
    c1_max_deviation: float
    c1_passed: bool
    c2_exact: bool
    c2_mismatches: tuple[Mapping[str, Any], ...]
    c3a_n_qualifying: int
    c3a_max_difference: float
    c3a_passed: bool
    c3a_max_difference_running_reading: float
    c3a_passed_running_reading: bool
    c3b_n_qualifying: int
    c3b_min_difference: float
    c3b_passed: bool
    c3b_min_difference_running_reading: float
    c3b_passed_running_reading: bool
    c3c_max_deviation: float
    c3c_agrees: bool
    c4_n_tiers: int
    c4_min_draws_per_tier: int
    c4_extremes_present: tuple[str, ...]
    c4_extremes_missing: tuple[str, ...]
    c4_passed: bool
    decomposition: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        """Criteria 1, 2, (3a), (3b) and 4.  (3c) is reported and never folded in (A12)."""
        return bool(
            self.c1_passed
            and self.c2_exact
            and self.c3a_passed
            and self.c3b_passed
            and self.c4_passed
        )

    def as_record(self) -> dict[str, Any]:
        """The JSON block for this scenario."""
        record = {name: getattr(self, name) for name in _CRITERIA_FIELD_NAMES}
        record["c2_mismatches"] = [dict(m) for m in self.c2_mismatches]
        record["c4_extremes_present"] = list(self.c4_extremes_present)
        record["c4_extremes_missing"] = list(self.c4_extremes_missing)
        record["decomposition"] = dict(self.decomposition)
        record["passed"] = self.passed
        record["c3a_reading_scored"] = (
            "entered_population -- the pool clock on the entered population, i.e. population varied "
            "and clock held. Ruled 2026-08-31 as Q7 option (b) and registered as A13; the "
            "admission-clock reading is reported beside it in the *_running_reading fields"
        )
        record["c3c_is_gating"] = False
        return record


#: The constructor fields of :class:`GateEpisode` and :class:`ScenarioCriteria`, derived from the
#: dataclasses themselves so a new field cannot be silently dropped from the artifact or from the
#: work-file round trip.
_EPISODE_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(GateEpisode))
_CRITERIA_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(ScenarioCriteria))


def _scenario_spec(scenario: str) -> GateScenario:
    """The declared spec of *scenario*, refusing an undeclared name so a typo cannot invent one."""
    try:
        return GATE_SCENARIOS[scenario]
    except KeyError as exc:
        raise ValueError(
            f"{scenario!r} is not a gated scenario; this gate covers {sorted(GATE_SCENARIOS)}. "
            "cf_cologne3 is out of scope (BRIEF_32 section 5): only 11 draws are materialised and "
            "it is absent from docs/data/p8_4a_admission.json entirely"
        ) from exc


def episode_from_record(record: Mapping[str, Any]) -> GateEpisode:
    """Rebuild a :class:`GateEpisode` from one work file's row, refusing a missing field."""
    missing = [name for name in _EPISODE_FIELD_NAMES if name not in record]
    if missing:
        raise ValueError(
            f"this episode record is missing {missing}; a row written by an older schema cannot be "
            "scored against these criteria without saying which fields it lacks"
        )
    return GateEpisode(**{name: record[name] for name in _EPISODE_FIELD_NAMES})


def tier_corpus_dirs(scenario: str, tier: str, roots: Any) -> tuple[Path, ...]:
    """The corpus directories behind one ``(scenario, tier)``, refusing a missing one.

    ⚠️ **Deviation from ``admission_probe``, disclosed.**  ``admission_probe.TIER_CORPUS_DIRS``
    declares five tiers on hz1x1 and two on grid4x4 and does not cover ``mappo060`` / ``mappo200``,
    so Amendment A1's seven-tier coverage cannot be served through it.  This is the mechanical
    ``cf_<scenario>__<tier>[__seed<N>]`` map, asserted against the directories actually on disk.
    It reimplements a LOOKUP, never a policy.
    """
    prefix = _scenario_spec(scenario).corpus_prefix
    if tier not in GATE_TIERS:
        raise ValueError(
            f"{tier!r} is not a gated tier; this gate declares {list(GATE_TIERS)}, which are the "
            "seven behaviour tiers datasets_v11/ carries per scenario (BRIEF_32 Amendment A1)"
        )
    if tier in PER_SEED_TIERS:
        names = tuple(f"{prefix}__{tier}__seed{seed}" for seed in CORPUS_SEEDS)
    else:
        names = (f"{prefix}__{tier}",)
    paths = tuple(Path(roots.corpus_root) / name for name in names)
    missing = [str(p) for p in paths if not (p / "manifest.json").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{scenario}/{tier}: these corpus directories have no manifest.json: {missing}"
        )
    return paths


def env_settings_for(scenario: str, tier: str, roots: Any) -> dict[str, Any]:
    """The evaluation env settings of one tier, read from its own collection manifests.

    Every directory of the tier must agree; a disagreement raises rather than picking the first,
    because the settings decide what the episode IS.  Delegates to
    ``offline.dt_gate.env_settings_from_manifest`` rather than restating any setting.
    """
    from offline.dt_gate import env_settings_from_manifest

    seen: dict[str, list[str]] = {}
    settings: dict[str, Any] | None = None
    for directory in tier_corpus_dirs(scenario, tier, roots):
        candidate = env_settings_from_manifest(directory / "manifest.json")
        key = json.dumps(candidate, sort_keys=True, default=str)
        seen.setdefault(key, []).append(str(directory))
        if settings is None:
            settings = candidate
    if settings is None:
        raise ValueError(f"{scenario}/{tier} names no corpus directory, so it has no env settings")
    if len(seen) > 1:
        summary = {sorted(paths)[0]: len(paths) for paths in seen.values()}
        raise ValueError(
            f"{scenario}/{tier}'s corpus directories disagree on the evaluation env settings "
            f"({summary}); picking one would compare two different episodes under one tier name"
        )
    return settings


def manifest_checkpoint(scenario: str, tier: str, seed: int, roots: Any) -> tuple[Path, str]:
    """The checkpoint that collected one ``(tier, seed)``, with its recorded digest verified.

    Same shape as ``admission_probe._manifest_checkpoint``, which cannot be reused because it routes
    through ``TIER_CORPUS_DIRS``.  The path comes from the tier's own manifest ``run_metadata`` and
    the sha256 is recomputed and compared, so a file that moved or was rebuilt cannot be substituted.
    """
    suffix = f"seed{int(seed)}"
    candidates_dirs = [d for d in tier_corpus_dirs(scenario, tier, roots) if d.name.endswith(suffix)]
    if not candidates_dirs:
        raise ValueError(
            f"{scenario}/{tier} has no corpus directory for seed {seed}; the seeded tiers are "
            f"{sorted(PER_SEED_TIERS)} and the collected seeds are {list(CORPUS_SEEDS)}"
        )
    directory = candidates_dirs[0]
    metadata = json.loads((directory / "manifest.json").read_bytes())["run_metadata"]
    recorded = metadata.get("checkpoint")
    expected = metadata.get("checkpoint_sha256")
    if not recorded or not expected:
        raise ValueError(
            f"{directory}/manifest.json records no behaviour checkpoint for {tier} seed {seed}; "
            "the anchor cannot be rebuilt from a manifest that does not name its policy"
        )
    relative = Path(recorded)
    candidates = (
        [relative]
        if relative.is_absolute()
        else [Path(roots.output_root).parent / relative, Path(roots.repo_root) / relative]
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                f"{candidate}: sha256 {digest} is not the {expected} the collection manifest "
                f"records for {tier} seed {seed}; this is not the policy that collected the tier"
            )
        return candidate, digest
    raise FileNotFoundError(
        f"the behaviour checkpoint {recorded!r} recorded by {directory}/manifest.json was not "
        f"found at any of {[str(c) for c in candidates]}"
    )


def seeds_for(tier: str, method: str) -> tuple[int | None, ...]:
    """The seed slot of one cell: ``(None,)`` for the deterministic anchors, ``(101,)`` otherwise."""
    if tier not in GATE_TIERS:
        raise ValueError(f"{tier!r} is not a gated tier; this gate declares {list(GATE_TIERS)}")
    if method == BEHAVIOUR_METHOD and tier in DETERMINISTIC_ANCHOR_TIERS:
        return (None,)
    return (GATE_SEED,)


def build_factory(
    scenario: str,
    tier: str,
    method: str,
    seed: int | None,
    roots: Any,
    *,
    device: str | None,
    config_path: str | Path,
) -> tuple[Callable[[Any], Callable[[Any, dict[str, Any]], np.ndarray]], Mapping[str, Any]]:
    """The action factory for one arm-seed, and the provenance of the policy behind it.

    Every factory is imported from the module that produced the committed cells --
    ``dt_gate._maxpressure_factory`` / ``_mappo_factory``, ``method_tier_grid._random_factory`` /
    ``_fixedtime_factory``, and ``admission_probe.build_factory`` for the learned extreme arms --
    because a second implementation of a protocol is exactly how two arms stop being comparable.
    """
    if method != BEHAVIOUR_METHOD:
        from offline.admission_probe import build_factory as probe_build_factory

        factory, source = probe_build_factory(
            scenario, tier, method, seed, roots, device=device, config_path=config_path
        )
        return factory, {
            "kind": source.kind,
            "detail": source.detail,
            "checkpoint": source.checkpoint,
            "checkpoint_sha256": source.checkpoint_sha256,
        }

    from offline.dt_gate import _mappo_factory, _maxpressure_factory
    from offline.method_tier_grid import (
        _fixedtime_factory,
        _random_factory,
        fixedtime_collection_settings,
    )

    if tier == "maxpressure":
        return _maxpressure_factory, {
            "kind": "algorithmic",
            "detail": "algorithms.max_pressure.MaxPressureAgent via dt_gate._maxpressure_factory",
            "checkpoint": None,
            "checkpoint_sha256": None,
        }

    if tier == "fixedtime":
        manifest = tier_corpus_dirs(scenario, tier, roots)[0] / "manifest.json"
        collected = fixedtime_collection_settings(manifest)
        return _fixedtime_factory(config_path, collected), {
            "kind": "plan",
            "detail": (
                "offline.policies.fixed_time.make_fixedtime with k="
                f"{collected['fixed_time_k']} and the plan hash asserted against {manifest}"
            ),
            "checkpoint": None,
            "checkpoint_sha256": str(collected["fixed_time_plan_sha256"]),
        }

    if tier == "random":
        if seed is None:
            raise ValueError("the random behaviour anchor is seeded and cannot take seed=None")
        return _random_factory(int(seed)), {
            "kind": "algorithmic",
            "detail": (
                "offline.collect._make_random with numpy.random.default_rng"
                f"({int(seed)}) rebuilt per draw, via method_tier_grid._random_factory"
            ),
            "checkpoint": None,
            "checkpoint_sha256": None,
        }

    if tier in PER_SEED_TIERS:
        if seed is None:
            raise ValueError(f"the {tier} behaviour anchor is per-seed and cannot take seed=None")
        path, digest = manifest_checkpoint(scenario, tier, int(seed), roots)
        return _mappo_factory(str(path), device), {
            "kind": "checkpoint",
            "detail": (
                "agent.MAPPOAgent via dt_gate._mappo_factory; path and digest read from the "
                "collecting run's own manifest run_metadata, never guessed"
            ),
            "checkpoint": str(path),
            "checkpoint_sha256": digest,
        }

    raise ValueError(f"no behaviour factory is declared for {scenario!r}/{tier!r}")


def gate_cells(scenario: str) -> tuple[GateCell, ...]:
    """The registered cell list of one scenario: seven tiers x three draws, plus the two extremes."""
    _scenario_spec(scenario)
    cells: list[GateCell] = []
    for tier in GATE_TIERS:
        for slot in seeds_for(tier, BEHAVIOUR_METHOD):
            for draw_id in GATE_DRAWS:
                cells.append(
                    GateCell(
                        scenario=scenario,
                        tier=tier,
                        method=BEHAVIOUR_METHOD,
                        seed=slot,
                        draw_id=int(draw_id),
                        role="tier",
                    )
                )
    for extreme in EXTREME_EPISODES[scenario]:
        cells.append(
            GateCell(
                scenario=scenario,
                tier=extreme.tier,
                method=extreme.method,
                seed=int(extreme.seed),
                draw_id=int(extreme.draw_id),
                role=f"extreme_{extreme.which}",
            )
        )
    keys = [(c.scenario, c.tier, c.method, c.seed, c.draw_id) for c in cells]
    if len(set(keys)) != len(keys):
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(
            f"{scenario}'s cell list repeats {duplicates}; one episode measured twice under two "
            "roles would be counted twice by criterion 4"
        )
    return tuple(cells)


def gate_episode(
    *,
    cell: GateCell,
    config_path: str | Path,
    env_settings: Mapping[str, Any],
    scenario_id: str,
    choose_action_factory: Callable[[Any], Callable[[Any, dict[str, Any]], np.ndarray]],
    engine_seed: int,
    created: int,
) -> GateEpisode:
    """Roll one episode under the observer and reconstruct every quantity at its horizon.

    Mirrors ``admission_probe.probe_episode``'s loop -- same ``horizon_rollout(env, factory(env),
    episodes=1, seed=engine_seed)``, same ``env.close()`` in a ``finally``, same reads inserted
    between the rollout and the close -- with the observer env in place of ``make_env``'s.

    *scenario_id* is CHECKED against the cell's own scenario rather than merely accepted, so a
    caller cannot roll an hz1x1 cell against a grid4x4 env id and file it under the wrong network.
    """
    from offline.admission_probe import read_admission_at_horizon
    from offline.horizon_metric import horizon_rollout

    expected_id = _scenario_spec(cell.scenario).scenario_id
    if str(scenario_id) != expected_id:
        raise ValueError(
            f"cell {cell.arm} is on {cell.scenario}, whose env id is {expected_id!r}, but "
            f"{scenario_id!r} was supplied; the two must agree or the episode is filed under the "
            "wrong network"
        )
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"draw {cell.draw_id} has no materialised sim config at {path}; run "
            "offline.materialise_draws for the held-out pool first"
        )

    started = time.perf_counter()
    env = make_observer_env(path, env_settings)
    try:
        choose_action = choose_action_factory(env)
        rollout_started = time.perf_counter()
        rollout = horizon_rollout(env, choose_action, episodes=1, seed=int(engine_seed))
        rollout_seconds = time.perf_counter() - rollout_started
        counts = read_admission_at_horizon(env, created=int(created))
        att_engine_call = float(env._eng.get_average_travel_time())
        built = reconstruct_episode(env.recorder)
    finally:
        env.close()
    seconds = time.perf_counter() - started

    if counts.running != int(rollout.final_vehicle_count):
        raise ValueError(
            f"{cell.arm} seed {cell.seed} draw {cell.draw_id}: the horizon reader reports "
            f"{rollout.final_vehicle_count} running vehicles from info['vehicle_count'] but the "
            f"engine reports {counts.running} now. Nothing may advance the simulation between the "
            "last step and the reconstruction read, and something did"
        )

    return GateEpisode(
        scenario=cell.scenario,
        tier=cell.tier,
        method=cell.method,
        arm=cell.arm,
        seed=None if cell.seed is None else int(cell.seed),
        draw_id=int(cell.draw_id),
        role=cell.role,
        engine_seed=int(engine_seed),
        att_reference_engine_population=float(built.engine_population.value),
        att_reference_entered_running=float(built.entered_running.value),
        att_reference_entered_population=float(built.entered_population.value),
        att_reference_metric_cadence=float(built.metric_cadence.value),
        att_engine_call=att_engine_call,
        att_ours=float(rollout.per_episode_horizon[0]),
        n_reference_ids=int(built.engine_population.n_ids),
        n_entered_ids=int(built.entered_running.n_ids),
        created_from_flow=int(created),
        entered=int(counts.entered),
        never_entered=int(counts.never_entered),
        admission_latency_mean=float(built.latency.mean),
        admission_latency_max=float(built.latency.maximum),
        n_admission_delayed=int(built.latency.n_delayed),
        interval=float(built.interval),
        n_observations=int(built.n_observations),
        seconds=seconds,
        seconds_rollout=rollout_seconds,
    )


def run_cells(
    cells: Sequence[GateCell],
    *,
    roots: Any,
    engine_seed: int,
    device: str | None,
) -> list[GateEpisode]:
    """Roll every cell, in declaration order, reusing one policy per ``(tier, method, seed)``."""
    from offline.admission_probe import created_from_flow
    from offline.materialise_draws import draw_config_path

    settings_by_tier: dict[tuple[str, str], dict[str, Any]] = {}
    configs: dict[tuple[str, int], Path] = {}
    created_by_draw: dict[tuple[str, int, int], int] = {}
    factories: dict[tuple[str, str, str, int | None], Any] = {}
    produced: list[GateEpisode] = []

    for cell in cells:
        tier_key = (cell.scenario, cell.tier)
        if tier_key not in settings_by_tier:
            settings_by_tier[tier_key] = env_settings_for(cell.scenario, cell.tier, roots)
        settings = settings_by_tier[tier_key]
        horizon = int(settings["max_steps"]) * int(settings["delta_time"])
        spec = _scenario_spec(cell.scenario)

        config_key = (cell.scenario, int(cell.draw_id))
        if config_key not in configs:
            candidate = Path(
                draw_config_path(spec.scenario_key, int(cell.draw_id), out_root=roots.draws_root)
            )
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"draw {cell.draw_id} has no materialised sim config at {candidate}; run "
                    "offline.materialise_draws for the held-out pool first"
                )
            configs[config_key] = candidate
        config = configs[config_key]

        created_key = (cell.scenario, int(cell.draw_id), horizon)
        if created_key not in created_by_draw:
            created_by_draw[created_key] = created_from_flow(
                config.parent / "flow.json", horizon_seconds=horizon
            )

        factory_key = (cell.scenario, cell.tier, cell.method, cell.seed)
        if factory_key not in factories:
            factories[factory_key] = build_factory(
                cell.scenario,
                cell.tier,
                cell.method,
                cell.seed,
                roots,
                device=device,
                config_path=config,
            )[0]

        print(f"{cell.scenario}/{cell.arm} seed {cell.seed} draw {cell.draw_id}", flush=True)
        produced.append(
            gate_episode(
                cell=cell,
                config_path=config,
                env_settings=settings,
                scenario_id=spec.scenario_id,
                choose_action_factory=factories[factory_key],
                engine_seed=int(engine_seed),
                created=created_by_draw[created_key],
            )
        )

    expected = {(c.tier, c.method, c.seed, c.draw_id) for c in cells}
    got = {(e.tier, e.method, e.seed, e.draw_id) for e in produced}
    if got != expected:
        raise ValueError(
            f"{len(got)} episodes against {len(expected)} requested "
            f"(missing {len(expected - got)}, unexpected {len(got - expected)})"
        )
    return produced


def evaluate_scenario(episodes: Sequence[GateEpisode]) -> ScenarioCriteria:
    """Score A11's criteria, as amended by A12, over one scenario's episodes."""
    rows = list(episodes)
    if not rows:
        raise ValueError(
            "no episodes were supplied, so there is nothing to score; an empty scenario must "
            "refuse rather than report a vacuous pass"
        )
    names = sorted({e.scenario for e in rows})
    if len(names) != 1:
        raise ValueError(
            f"these are episodes from more than one scenario ({names}); scoring two networks under "
            "one outcome is exactly what Rule R's per-scenario grain exists to prevent"
        )
    scenario = names[0]

    # Criterion 1 -- AGREEMENT.
    c1_max = max(e.deviation_c1 for e in rows)

    # Criterion 2 -- DENOMINATOR, two independent routes, on ints, with ==.
    mismatches = tuple(
        {
            "arm": e.arm,
            "seed": e.seed,
            "draw_id": e.draw_id,
            "n_reference_ids": int(e.n_reference_ids),
            "created_from_flow": int(e.created_from_flow),
            "entered_plus_never_entered": int(e.entered) + int(e.never_entered),
        }
        for e in rows
        if int(e.n_reference_ids) != int(e.created_from_flow)
        or int(e.n_reference_ids) != int(e.entered) + int(e.never_entered)
    )

    # A12 (3a) -- EXACT bit-identity where never_entered == 0.  Zero tolerance, not 1e-4.
    # A13 (Q7 option (b), ruled 2026-08-31): scored on the POPULATION reading -- clock held at the
    # pool clock, population varied -- which is what (3a)'s own justifying clause describes.  The
    # admission-clock reading is measured and reported beside it, never substituted for it.
    uncensored = [e for e in rows if int(e.never_entered) == 0]
    c3a_max = max((e.difference_c3a_population for e in uncensored), default=0.0)
    c3a_max_running = max((e.difference_c3a_running for e in uncensored), default=0.0)

    # A12 (3b) -- a difference is REQUIRED where never_entered > 0.  Scored on the same reading as
    # (3a): A12 calls the two "a two-sided proof that the instrument distinguishes the two
    # POPULATIONS", so the pair must vary the same thing.  Both readings are reported.
    censored = [e for e in rows if int(e.never_entered) > 0]
    c3b_min = min((e.difference_c3a_population for e in censored), default=0.0)
    c3b_min_running = min((e.difference_c3a_running for e in censored), default=0.0)

    # A12 (3c) -- REQUIRED and REPORTED, never gating.
    c3c_max = max(e.deviation_c3c for e in rows)

    # A13's three-component decomposition of att_ours - att_engine, reported per scenario.
    decomposition = _decomposition_summary(rows)

    # Criterion 4 -- COVERAGE, counted over the tier cells, extremes checked by identity.
    tier_rows = [e for e in rows if e.role == "tier"]
    draws_by_tier: dict[str, set[int]] = {}
    for row in tier_rows:
        draws_by_tier.setdefault(row.tier, set()).add(int(row.draw_id))
    present: list[str] = []
    absent: list[str] = []
    for declared in EXTREME_EPISODES.get(scenario, ()):
        found = any(
            e.tier == declared.tier
            and e.method == declared.method
            and e.seed == declared.seed
            and e.draw_id == declared.draw_id
            for e in rows
        )
        (present if found else absent).append(declared.which)
    n_tiers = len(draws_by_tier)
    min_draws = min((len(v) for v in draws_by_tier.values()), default=0)

    return ScenarioCriteria(
        scenario=scenario,
        n_episodes=len(rows),
        c1_max_deviation=float(c1_max),
        c1_passed=bool(c1_max < C1_TOLERANCE),
        c2_exact=not mismatches,
        c2_mismatches=mismatches,
        c3a_n_qualifying=len(uncensored),
        c3a_max_difference=float(c3a_max),
        c3a_passed=bool(uncensored) and c3a_max == 0.0,
        c3a_max_difference_running_reading=float(c3a_max_running),
        c3a_passed_running_reading=bool(uncensored) and c3a_max_running == 0.0,
        c3b_n_qualifying=len(censored),
        c3b_min_difference=float(c3b_min),
        c3b_passed=bool(censored) and c3b_min > 0.0,
        c3b_min_difference_running_reading=float(c3b_min_running),
        c3b_passed_running_reading=bool(censored) and c3b_min_running > 0.0,
        c3c_max_deviation=float(c3c_max),
        c3c_agrees=bool(c3c_max < C3C_TOLERANCE),
        c4_n_tiers=int(n_tiers),
        c4_min_draws_per_tier=int(min_draws),
        c4_extremes_present=tuple(sorted(present)),
        c4_extremes_missing=tuple(sorted(absent)),
        c4_passed=bool(n_tiers >= C4_MIN_TIERS and min_draws >= C4_MIN_DRAWS and not absent),
        decomposition=decomposition,
    )


def _decomposition_summary(rows: Sequence[GateEpisode]) -> dict[str, Any]:
    """A13's three-component decomposition of ``att_ours - att_engine``, summarised per scenario.

    ::

        att_ours - att_engine  =  population  +  clock_origin  +  cadence

    * **population** -- restrict from every CREATED vehicle to the ones that entered, pool clock
      held fixed.  Exactly zero when ``never_entered == 0``, which is what (3a) scores.
    * **clock_origin** -- move the clock from pool entry to admission, population held fixed.  This
      is minus the mean admission latency: the insertion-buffer wait the engine counts and
      ``metrics/cityflow.py`` does not.
    * **cadence** -- the 10 s decision grid's midpoint-departure and end-of-window-completion
      estimators (T1's M1).

    The identity is EXACT rather than approximate whenever criteria 1 and (3c) are exact, because
    the three terms telescope through the reconstructions and the endpoints are then the engine's
    own call and our own metric.  ``residual_max`` reports that, so a reader can check the
    decomposition instead of taking it on trust.
    """
    if not rows:
        raise ValueError("the decomposition needs at least one episode to summarise")

    def summarise(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        middle, odd = divmod(len(ordered), 2)
        median = ordered[middle] if odd else (ordered[middle - 1] + ordered[middle]) / 2.0
        return {
            "min": float(ordered[0]),
            "median": float(median),
            "max": float(ordered[-1]),
            "mean": float(sum(ordered) / len(ordered)),
        }

    residuals = [e.decomposition_residual for e in rows]
    return {
        "identity": "att_ours - att_engine = population + clock_origin + cadence",
        "n_episodes": len(rows),
        "population": summarise([e.term_population for e in rows]),
        "clock_origin": summarise([e.term_clock_origin for e in rows]),
        "cadence": summarise([e.term_cadence for e in rows]),
        "total": summarise([float(e.att_ours) - float(e.att_engine_call) for e in rows]),
        "residual_max": float(max(residuals)),
        "n_episodes_with_exactly_zero_residual": sum(1 for r in residuals if r == 0.0),
    }


def thread_regime() -> dict[str, Any]:
    """The thread pin, READ AT RUN TIME and never assumed.

    ``dt_gate.runtime_provenance`` records ``torch.get_num_threads()`` and nothing else about
    threading (checked: ``offline/dt_gate.py:626-685``), and ``torch.set_num_threads()`` is a
    DIFFERENT KNOB from ``OMP_NUM_THREADS`` -- a recorded ``torch_num_threads = 1`` does not
    establish which regime produced a timing.  So the environment variables are read here.
    """
    import torch

    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "torch_num_threads": int(torch.get_num_threads()),
        "note": (
            "the three environment variables are read from os.environ at run time and are None "
            "when unset; torch_num_threads is a DIFFERENT knob, set by torch.set_num_threads(), "
            "so a recorded 1 there says nothing about the OpenMP regime"
        ),
    }


def gate_artifact(
    *,
    episodes: Sequence[GateEpisode],
    criteria: Mapping[str, ScenarioCriteria],
    timing: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble ``docs/data/p8_4b_g0_reference.json``.

    Carries the registered criteria verbatim, every episode, every scenario's outcome with its
    observed values, and the provenance.  It carries **no** verdict on the metric: the payload is
    passed through ``admission_probe.assert_no_science_verdict`` before it is returned.
    """
    from offline.admission_probe import assert_no_science_verdict, code_provenance

    rows = list(episodes)
    if not rows:
        raise ValueError(
            "the artifact was given an empty episode list, so it would record a gate that "
            "measured nothing"
        )
    if not criteria:
        raise ValueError(
            "the artifact carries no scored scenario, so its criteria block would be empty while "
            "its episode block was not"
        )

    payload: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": (
            "P8.4b Gate 0: an independent, per-second reconstruction of CityFlow's own "
            "Engine::getAverageTravelTime, scored against A11's four criteria as amended by A12"
        ),
        "what_this_does_not_say": [
            "no verdict on which ATT definition the paper's claims rest on: Rule R decides that "
            "per scenario, from these numbers, and this artifact does not anticipate it",
            "no claim that either definition is wrong; both compute what their sources say",
            "criterion (3c) is REQUIRED and REPORTED but NOT GATING (A12), so a disagreement "
            "there is a new finding about our own metric and is escalated, never folded in here",
        ],
        "registered": {
            "governed_by": "PREREGISTRATION A11 as amended by A12, tag v1.2-prereg-a12",
            "declared_in": "docs/plans/p8.4b-g0.md, before any gate number existed",
            "scenarios": sorted(GATE_SCENARIOS),
            "tiers": list(GATE_TIERS),
            "draws": list(GATE_DRAWS),
            "seed": GATE_SEED,
            "deterministic_anchor_tiers": sorted(DETERMINISTIC_ANCHOR_TIERS),
            "extreme_episodes": {
                name: [
                    {
                        "which": e.which,
                        "tier": e.tier,
                        "method": e.method,
                        "seed": e.seed,
                        "draw_id": e.draw_id,
                        "entered_fraction_at_declaration": e.entered_fraction,
                    }
                    for e in declared
                ]
                for name, declared in sorted(EXTREME_EPISODES.items())
            },
            "criteria": {
                "c1": "max |engine-population reconstruction - the engine's own call| < 1e-4 s",
                "c2": "the reconstruction's vehicle count equals created, exactly, every episode",
                "c3a": (
                    "where never_entered == 0 the engine-population and entered-only "
                    "reconstructions are BIT-IDENTICAL; zero tolerance"
                ),
                "c3b": "where never_entered > 0 the two MUST differ, and the difference is reported",
                "c3c": (
                    "the decision-grid reconstruction reproduces our own horizon metric to 1e-4 s; "
                    "required and reported, NOT gating"
                ),
                "c4": ">=7 behaviour tiers x >=3 draws, plus both entered_fraction extremes",
            },
            "tolerances": {"c1": C1_TOLERANCE, "c3a": 0.0, "c3c": C3C_TOLERANCE},
            "alignment_convention": (
                "a snapshot is taken immediately after each next_step(), so first_seen(v) == "
                "enterTime(v) + interval and contribution(v) == last_seen - first_seen + interval "
                "for every vehicle"
            ),
        },
        "criteria": {name: block.as_record() for name, block in sorted(criteria.items())},
        "episodes": [e.as_record() for e in rows],
        "timing": dict(timing),
        "provenance": {
            **dict(provenance),
            "thread_regime": thread_regime(),
            "code_provenance": code_provenance(),
        },
    }
    assert_no_science_verdict(payload)
    return payload


def default_work_dir(output_root: str | Path) -> Path:
    """``<output_root>/p8_4b_g0`` -- this task's own work directory."""
    return Path(output_root) / "p8_4b_g0"


def cell_file_name(scenario: str, tier: str, method: str, seed: int | None, draw_id: int) -> str:
    """``reference_<scenario>_<tier>_<method>_seed<seed|none>_draw<draw>.json``, one per episode."""
    slot = "none" if seed is None else str(int(seed))
    return f"reference_{scenario}_{tier}_{method}_seed{slot}_draw{int(draw_id)}.json"


def cell_files(work_dir: str | Path) -> tuple[Path, ...]:
    """Every episode file written into *work_dir*, sorted."""
    return tuple(sorted(Path(work_dir).glob("reference_*.json")))


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser.

    ⚠️ **Divergence from ``admission_probe``, ruled by Amendment A4 and disclosed here**: this module
    ships ``--tiers`` (``nargs="+"``) where ``admission_probe`` has ``--tier`` (singular, one cell per
    process).  Criterion 4 is a property of a SET of tiers, so the plural is the shape this gate
    needs, and the campaign is a minute rather than an hour.

    ⚠️ ``allow_abbrev`` is FALSE on every parser here.  With argparse's default, ``--tier`` would be
    silently accepted as an abbreviation of ``--tiers`` -- so a command line copied from
    ``admission_probe`` would run and mean something the operator did not type.  On a CLI where a
    mistyped flag changes what was measured, a refusal is the only safe behaviour.
    """
    parser = argparse.ArgumentParser(
        prog="python -m offline.engine_att_reference",
        description="P8.4b Gate 0: reconstruct CityFlow's own ATT and score A11's criteria",
        allow_abbrev=False,
    )
    parser.add_argument("--repo-root", default=".", help="this worktree (holds docs/data)")
    parser.add_argument("--corpus-root", default="datasets_v11")
    parser.add_argument("--draws-root", default="scenarios/draws")
    parser.add_argument("--output-root", default="output", help="the MAIN tree's output/")
    parser.add_argument(
        "--work-dir", default=None, help="per-episode files; default <output-root>/p8_4b_g0"
    )
    parser.add_argument("--engine-seed", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--protect",
        action="append",
        default=[],
        metavar="PATH",
        help="an extra read-only root, on top of the corpus and every sibling output/ directory",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    gate = sub.add_parser("gate", help="roll the requested cells and write one file per episode",
                          allow_abbrev=False)
    gate.add_argument("--scenario", required=True, choices=sorted(GATE_SCENARIOS))
    gate.add_argument("--tiers", nargs="+", default=list(GATE_TIERS))
    gate.add_argument("--draws", type=int, nargs="+", default=list(GATE_DRAWS))
    gate.add_argument(
        "--extremes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the entered_fraction extremes (Amendment A2); --no-extremes drops them",
    )

    report = sub.add_parser("report", help="assemble docs/data/p8_4b_g0_reference.json",
                            allow_abbrev=False)
    report.add_argument("--out", default="docs/data/p8_4b_g0_reference.json")
    return parser


def _roots_of(args: argparse.Namespace) -> Any:
    """Build the ``admission_probe.ProbeRoots`` this module reads and writes through."""
    from offline.admission_probe import ProbeRoots

    output_root = Path(args.output_root)
    work_dir = Path(args.work_dir) if args.work_dir else default_work_dir(output_root)
    return ProbeRoots(
        repo_root=Path(args.repo_root),
        corpus_root=Path(args.corpus_root),
        draws_root=Path(args.draws_root),
        output_root=output_root,
        work_dir=work_dir,
    )


def _protected_roots(args: argparse.Namespace, roots: Any) -> tuple[Path, ...]:
    from offline.admission_probe import default_protected_roots
    from offline.tier_sweep import protected_roots_from

    return tuple(default_protected_roots(roots)) + tuple(protected_roots_from(args.protect))


def _run_gate(args: argparse.Namespace) -> int:
    """Roll the selected cells and write one file per episode, after the barrier."""
    from offline.dt_gate import runtime_provenance
    from offline.offline_baselines import pin_torch_threads
    from offline.tier_sweep import assert_writable, write_json_guarded

    pin_torch_threads(args.torch_threads)
    roots = _roots_of(args)
    protected = _protected_roots(args, roots)

    tiers = [str(t) for t in args.tiers]
    unknown = [t for t in tiers if t not in GATE_TIERS]
    if unknown:
        raise ValueError(f"unknown tiers {unknown}; this gate declares {list(GATE_TIERS)}")
    draws = [int(d) for d in args.draws]
    selected = [
        cell
        for cell in gate_cells(args.scenario)
        if (cell.role == "tier" and cell.tier in tiers and cell.draw_id in draws)
        or (cell.role.startswith("extreme") and bool(args.extremes))
    ]
    if not selected:
        raise ValueError(
            f"the selection --tiers {tiers} --draws {draws} matches no cell of {args.scenario}; "
            "a run that measures nothing must refuse rather than write an empty result"
        )

    started = time.perf_counter()
    episodes = run_cells(
        selected, roots=roots, engine_seed=int(args.engine_seed), device=args.device
    )
    elapsed = time.perf_counter() - started

    runtime = runtime_provenance()
    regime = thread_regime()
    work = Path(roots.work_dir)
    # The barrier: every destination is validated BEFORE the first directory is created.
    destinations = [
        assert_writable(
            work / cell_file_name(e.scenario, e.tier, e.method, e.seed, e.draw_id), protected
        )
        for e in episodes
    ]
    work.mkdir(parents=True, exist_ok=True)
    for episode_row, destination in zip(episodes, destinations):
        write_json_guarded(
            {
                "format_version": ARTIFACT_FORMAT_VERSION,
                **episode_row.as_record(),
                "runtime": runtime,
                "thread_regime": regime,
            },
            destination,
            protected,
        )

    criteria = evaluate_scenario(episodes)
    print(
        f"{args.scenario}: {len(episodes)} episodes in {elapsed:.1f} s "
        f"({elapsed / len(episodes):.2f} s/episode)",
        flush=True,
    )
    print(
        f"  c1 max deviation {criteria.c1_max_deviation!r} (passed={criteria.c1_passed}); "
        f"c2 exact={criteria.c2_exact}",
        flush=True,
    )
    print(f"  wrote {len(destinations)} episode files under {work}", flush=True)
    return 0


def _run_report(args: argparse.Namespace) -> int:
    """Assemble, score, write, and exit non-zero when any scenario FAILS."""
    from offline.dt_gate import runtime_provenance
    from offline.tier_sweep import assert_writable, write_json_guarded

    roots = _roots_of(args)
    protected = _protected_roots(args, roots)
    files = cell_files(roots.work_dir)
    if not files:
        raise FileNotFoundError(
            f"no episode files under {roots.work_dir}; run the gate subcommand first"
        )

    episodes: list[GateEpisode] = []
    for path in files:
        record = json.loads(path.read_bytes())
        if record.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"{path} records format_version {record.get('format_version')!r}, not "
                f"{ARTIFACT_FORMAT_VERSION!r}; a row from another schema cannot be scored here"
            )
        episodes.append(episode_from_record(record))

    by_scenario: dict[str, list[GateEpisode]] = {}
    for row in episodes:
        by_scenario.setdefault(row.scenario, []).append(row)
    criteria = {name: evaluate_scenario(rows) for name, rows in sorted(by_scenario.items())}

    payload = gate_artifact(
        episodes=episodes,
        criteria=criteria,
        timing={
            "seconds_total": float(sum(e.seconds for e in episodes)),
            "seconds_rollout_total": float(sum(e.seconds_rollout for e in episodes)),
            "n_episodes": len(episodes),
            "note": (
                "seconds is the whole per-episode cost including env construction and the policy "
                "load; seconds_rollout isolates the observed simulation"
            ),
        },
        provenance={"runtime": runtime_provenance(), "episode_files": [p.name for p in files]},
    )

    destination = Path(args.repo_root) / args.out
    assert_writable(destination, protected)
    write_json_guarded(payload, destination, protected)

    failed = [name for name, block in criteria.items() if not block.passed]
    for name, block in criteria.items():
        print(
            f"{name}: n={block.n_episodes} c1={block.c1_max_deviation!r} "
            f"c2_exact={block.c2_exact} c3a={block.c3a_max_difference!r} "
            f"(admission-clock reading {block.c3a_max_difference_running_reading!r}) "
            f"c3b_min={block.c3b_min_difference!r} c3c={block.c3c_max_deviation!r} "
            f"tiers={block.c4_n_tiers} draws={block.c4_min_draws_per_tier} "
            f"passed={block.passed}",
            flush=True,
        )
    print(f"wrote {destination}", flush=True)
    if failed:
        print(f"GATE FAILED on {failed}", flush=True)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.  Returns a non-zero exit code when the gate FAILS.

    A check that reports by printing is not a check: a failing criterion must make the process exit
    non-zero so a caller cannot mistake the report for a pass.
    """
    args = build_parser().parse_args(argv)
    if args.command == "gate":
        return _run_gate(args)
    return _run_report(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
