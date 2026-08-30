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
from dataclasses import dataclass
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
    "PerSecondEngineObserver",
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
        raise NotImplementedError

    def restricted_to(self, ids: Sequence[str] | frozenset[str] | set[str]) -> "VehicleWindows":
        """This stream's clocks restricted to *ids*, refusing an id this stream never saw."""
        raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def admission_latency(pool: VehicleWindows, running: VehicleWindows) -> AdmissionLatency:
    """``first_running_seen - first_pooled_seen`` over the ids that ever ran.

    Refuses a running id the pool stream never saw, and a negative latency: the running population is
    a subset of the pool population at every instant (``engine.cpp:780-789``), so either would mean
    the two streams were not observed on the same grid.
    """
    raise NotImplementedError


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
        raise NotImplementedError

    def clear(self) -> None:
        """Drop every observation, so one recorder can serve consecutive episodes."""
        raise NotImplementedError

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
        raise NotImplementedError

    @property
    def pool_windows(self) -> VehicleWindows:
        """``get_vehicles(include_waiting=True)`` -- the whole ``vehiclePool``."""
        raise NotImplementedError

    @property
    def running_windows(self) -> VehicleWindows:
        """``get_vehicles(include_waiting=False)`` -- the admitted population."""
        raise NotImplementedError

    @property
    def decision_grid_snapshots(self) -> tuple[tuple[float, frozenset[str]], ...]:
        """The admitted sets at times that are exact multiples of ``delta_time``, for (3c)."""
        raise NotImplementedError

    @property
    def observation_times(self) -> tuple[float, ...]:
        """Every engine time at which a snapshot was taken, in order."""
        raise NotImplementedError

    @property
    def observed_interval(self) -> float:
        """The engine's ``interval``, MEASURED as the gap between consecutive observations.

        Derived from the observation stream rather than read from the config, so a config whose
        ``interval`` did not match the engine's behaviour would raise here instead of silently
        rescaling every contribution.  Refuses a non-uniform gap and fewer than two observations.
        """
        raise NotImplementedError


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
    raise NotImplementedError


def reconstruct_episode(recorder: EngineObservationRecorder) -> EpisodeReconstruction:
    """All four reconstructions of the episode *recorder* has just observed."""
    raise NotImplementedError


def make_observer_env(config_path: str | Path, settings: Mapping[str, Any]) -> Any:
    """Build the observer env exactly as ``experiments.envs.make_env`` builds a ``CityFlowEnv``.

    ``make_env`` returns the frozen class and cannot return a subclass, so its CityFlow branch
    (``experiments/envs.py:84-113``) is mirrored here: the same seven ``common`` keys, the same
    optional ``metrics`` / ``obs_norm`` handling, the same ``thread_num``, and
    ``experiments.envs.phase_control_cls`` imported rather than a local mapping.
    ``tests/test_engine_att_reference.py`` asserts the two constructions agree field by field.
    """
    raise NotImplementedError


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
        raise NotImplementedError


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
        """Criterion 1: ``|engine-population reconstruction - get_average_travel_time()|``."""
        raise NotImplementedError

    @property
    def deviation_c3c(self) -> float:
        """A12 (3c): ``|metric-cadence reconstruction - att_ours|``.  Reported, not gating."""
        raise NotImplementedError

    @property
    def difference_c3a_running(self) -> float:
        """A12 (3a)/(3b) as registered: engine-population minus ``entered_running``."""
        raise NotImplementedError

    @property
    def difference_c3a_population(self) -> float:
        """The same difference under the population-only reading (``docs/plans`` section 8.4 (b))."""
        raise NotImplementedError

    @property
    def cadence_component(self) -> float:
        """``att_ours - entered_running`` -- the 10 s-grid quantisation term, isolated."""
        raise NotImplementedError

    def as_record(self) -> dict[str, Any]:
        """The JSON row, with keys sorted by the artifact writer rather than here."""
        raise NotImplementedError


@dataclass(frozen=True)
class ScenarioCriteria:
    """A11's four criteria as amended by A12, scored on one scenario, with observed values.

    ⚠️ ``c3a_passed`` scores the criterion **exactly as registered** -- against ``entered_running``,
    which is ``BRIEF_32`` section 4's entered-only variant and the definition A12 did not amend.
    ``c3a_passed_population_reading`` reports the same criterion under the alternative reading raised
    as Q7 in ``docs/plans/p8.4b-g0.md`` section 8.  **Reinterpreting a registered criterion is an
    amendment and is not this module's to make**, so the gate's outcome uses the registered one and
    the alternative is reported beside it.

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
    c3a_max_difference_population_reading: float
    c3a_passed_population_reading: bool
    c3b_n_qualifying: int
    c3b_min_difference: float
    c3b_passed: bool
    c3c_max_deviation: float
    c3c_agrees: bool
    c4_n_tiers: int
    c4_min_draws_per_tier: int
    c4_extremes_present: tuple[str, ...]
    c4_extremes_missing: tuple[str, ...]
    c4_passed: bool

    @property
    def passed(self) -> bool:
        """Criteria 1, 2, (3a), (3b) and 4.  (3c) is reported and never folded in (A12)."""
        raise NotImplementedError

    def as_record(self) -> dict[str, Any]:
        """The JSON block for this scenario."""
        raise NotImplementedError


def tier_corpus_dirs(scenario: str, tier: str, roots: Any) -> tuple[Path, ...]:
    """The corpus directories behind one ``(scenario, tier)``, refusing a missing one.

    ⚠️ **Deviation from ``admission_probe``, disclosed.**  ``admission_probe.TIER_CORPUS_DIRS``
    declares five tiers on hz1x1 and two on grid4x4 and does not cover ``mappo060`` / ``mappo200``,
    so Amendment A1's seven-tier coverage cannot be served through it.  This is the mechanical
    ``cf_<scenario>__<tier>[__seed<N>]`` map, asserted against the directories actually on disk.
    It reimplements a LOOKUP, never a policy.
    """
    raise NotImplementedError


def env_settings_for(scenario: str, tier: str, roots: Any) -> dict[str, Any]:
    """The evaluation env settings of one tier, read from its own collection manifests.

    Every directory of the tier must agree; a disagreement raises rather than picking the first,
    because the settings decide what the episode IS.  Delegates to
    ``offline.dt_gate.env_settings_from_manifest`` rather than restating any setting.
    """
    raise NotImplementedError


def manifest_checkpoint(scenario: str, tier: str, seed: int, roots: Any) -> tuple[Path, str]:
    """The checkpoint that collected one ``(tier, seed)``, with its recorded digest verified.

    Same shape as ``admission_probe._manifest_checkpoint``, which cannot be reused because it routes
    through ``TIER_CORPUS_DIRS``.  The path comes from the tier's own manifest ``run_metadata`` and
    the sha256 is recomputed and compared, so a file that moved or was rebuilt cannot be substituted.
    """
    raise NotImplementedError


def seeds_for(tier: str, method: str) -> tuple[int | None, ...]:
    """The seed slot of one cell: ``(None,)`` for the deterministic anchors, ``(101,)`` otherwise."""
    raise NotImplementedError


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
    raise NotImplementedError


def gate_cells(scenario: str) -> tuple[GateCell, ...]:
    """The registered cell list of one scenario: seven tiers x three draws, plus the two extremes."""
    raise NotImplementedError


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
    """
    raise NotImplementedError


def run_cells(
    cells: Sequence[GateCell],
    *,
    roots: Any,
    engine_seed: int,
    device: str | None,
) -> list[GateEpisode]:
    """Roll every cell, in declaration order, reusing one policy per ``(tier, method, seed)``."""
    raise NotImplementedError


def evaluate_scenario(episodes: Sequence[GateEpisode]) -> ScenarioCriteria:
    """Score A11's criteria, as amended by A12, over one scenario's episodes."""
    raise NotImplementedError


def thread_regime() -> dict[str, Any]:
    """The thread pin, READ AT RUN TIME and never assumed.

    ``dt_gate.runtime_provenance`` records ``torch.get_num_threads()`` and nothing else about
    threading (checked: ``offline/dt_gate.py:626-685``), and ``torch.set_num_threads()`` is a
    DIFFERENT KNOB from ``OMP_NUM_THREADS`` -- a recorded ``torch_num_threads = 1`` does not
    establish which regime produced a timing.  So the environment variables are read here.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def default_work_dir(output_root: str | Path) -> Path:
    """``<output_root>/p8_4b_g0`` -- this task's own work directory."""
    raise NotImplementedError


def cell_file_name(scenario: str, tier: str, method: str, seed: int | None, draw_id: int) -> str:
    """``reference_<scenario>_<tier>_<method>_seed<seed|none>_draw<draw>.json``, one per episode."""
    raise NotImplementedError


def cell_files(work_dir: str | Path) -> tuple[Path, ...]:
    """Every episode file written into *work_dir*, sorted."""
    raise NotImplementedError


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser.

    ⚠️ **Divergence from ``admission_probe``, ruled by Amendment A4 and disclosed here**: this module
    ships ``--tiers`` (``nargs="+"``) where ``admission_probe`` has ``--tier`` (singular, one cell per
    process).  Criterion 4 is a property of a SET of tiers, so the plural is the shape this gate
    needs, and the campaign is a minute rather than an hour.
    """
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.  Returns a non-zero exit code when the gate FAILS.

    A check that reports by printing is not a check: a failing criterion must make the process exit
    non-zero so a caller cannot mistake the report for a pass.
    """
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
