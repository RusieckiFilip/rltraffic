"""P8.4a: how many vehicles each arm actually admits into the network.

⚠️ SKELETON.  Every function below raises :class:`NotImplementedError`; the constants are the
REGISTERED declarations from ``docs/plans/p8.4a.md`` and ``BRIEF_31`` Amendment A and are real.
Tests are written against this surface first, so each one fails for its own reason rather than
sharing a single import error.

Artifact format version: ``p8.4a-admission/1.0``.

WHAT THIS TASK DECIDES
----------------------
``docs/reviews/T1-metric-ground-truth.md`` established that ``metrics/cityflow.py``'s
``average_travel_time`` averages over the vehicles the metric has *seen running*
(``get_vehicles(include_waiting=False)``, ``metrics/cityflow.py:60`` and ``:159``), while CityFlow's
own ``getAverageTravelTime`` (``CityFlow/src/engine/engine.cpp:682-691``) averages over the whole
``vehiclePool`` -- every vehicle ever created.  A policy that leaves vehicles in a lane's insertion
buffer therefore shrinks our metric's denominator *and* drops its slowest members.

**So the question this module answers, and nothing else:** do the offline-learned arms admit fewer
vehicles than the behaviour policy of their own tier?

⛔ **This module does not fix the metric, does not re-derive ATT (that is P8.4b), and issues no
verdict on P5.2's headline** (``BRIEF_31`` section 6).  ``average_travel_time`` computes what its
docstring says; **it measures a different population than the field's, and that is the finding.**

THE EPISODE IS NOT PERTURBED, AND THAT IS WHY THE REPRODUCTION CHECK CAN BE EXACT
--------------------------------------------------------------------------------
Every episode goes through :func:`offline.horizon_metric.horizon_rollout` -- the same function
``offline.dt_gate.evaluate_arm`` calls -- with the same env construction, the same settings source
(``dt_gate.env_settings_from_manifest`` on the tier's own collection manifest), the same
``engine_seed`` and the same action factories.  The probe reads the engine and the metrics object
**after** the rollout returns and **before** ``env.close()``: no extra ``next_step``, no subclass, no
monkeypatch, no per-second sampling.

``evaluate_arm`` is not reused because it closes its env in a ``finally`` block, which destroys the
state this module must read.  :func:`probe_episode` is therefore a copy of its body plus four
post-rollout reads.  **That duplication is deliberate and is disclosed here rather than hidden**: the
alternative is editing a function that seven merged campaigns' numbers came out of.

THE MEASUREMENT IDENTITY
------------------------
Read from ``CityFlow/src/engine/engine.cpp`` (verified in source, not inferred):

* ``activeVehicleCount`` rises **only** in ``handleWaiting()`` (``:509``), when a vehicle leaves a
  lane's waiting buffer and starts running, and falls **only** at ``:309``, when it finishes.
* ``getRunningVehicles(includeWaiting)`` (``:780-789``) walks ``vehiclePool`` keeping
  ``isReal() && (includeWaiting || isRunning())``.  Every draw config sets ``laneChange: false``, so
  the ``isReal()`` clause is inert here.
* A vehicle joins ``vehiclePool`` at creation (``pushVehicle``, ``:605-613``) and is erased only when
  it finishes (``:308``).

Therefore, at the horizon::

    never_entered = |get_vehicles(include_waiting=True)| - |get_vehicles(include_waiting=False)|   (R1)
    entered       = |metrics._episode["depart_time"]|                                              (R2)
    entered       = |metrics._episode["completed"]| + |get_vehicles(include_waiting=False)|         (R3)
    created       = entered + never_entered                                                        (R4)
    created       = #{flow entries with startTime <= horizon_seconds - 1}                          (R5)

**R1 is engine-side and exact.  R2 and R3 are metric-side.  R5 parses the flow file and touches no
engine at all.**  :func:`reconcile_admission` asserts ``R2 == R3`` and ``R4 == R5`` with ``==`` on
Python ``int``s, per episode; a mismatch raises and the episode is not recorded.

⚠️ **The bound in R5 is load-bearing.**  ``scenarios/draws/cityflow1x1/draw_1000/flow.json`` holds
1821 entries but its largest ``startTime`` is 3658, against a 3600 s episode: **8 entries never
fire**.  Reading that draw's ``provenance.json`` ``n_vehicles`` (1821) as ``created`` would be wrong
by 8 vehicles.  R5 is exact for this corpus because every entry has ``startTime == endTime``, which
:func:`created_from_flow` **asserts rather than assumes**, and ``Flow::nextStep``
(``CityFlow/src/flow/flow.cpp:6-22``) then emits exactly one vehicle at ``currentTime == startTime``.

WHAT IS RECORDED, AND AT WHICH GRAIN
------------------------------------
Per episode: ``created``, ``entered``, ``never_entered``, ``entered_fraction``, ``att_ours``
(the A1 primary metric at the horizon), ``att_engine`` (CityFlow's own definition at the same
instant), ``horizon_vehicle_count``, ``episode_reward`` and ``seconds``.  **Both ATT definitions on
every episode**: that is ``PREREGISTRATION`` A5's withdrawn co-report restored and widened, and
``BRIEF_31`` section 3 rules that the entered count is measured first rather than co-reported.

Per cell: the per-seed admission ratio **always**, never a bare pooled mean (``BRIEF_31`` section 5).

REGISTERED SCORING (docs/plans/p8.4a.md section 4, Amendment A section A3/A6)
-----------------------------------------------------------------------------
``r = sum(entered) / sum(created)`` over a cell -- a population ratio, not a mean of per-episode
ratios.  ``spread = max_seed r - min_seed r``, which is 0 for a single-seeded arm.

* **E1**  ``deficit = r(behaviour) - r(arm)``; ``Delta = max(spread_behaviour, spread_arm)``.
  ``deficit <= 0`` holds, ``0 < deficit <= Delta`` is close, ``deficit > Delta`` is falsified.
  **Any arm with ``deficit > 0`` at all escalates to the full 100 held-out draws.**
  ⭐ The permissive ``Delta`` governs the verdict; the escalation trigger sits at zero.  **Neither may
  be loosened without the other being re-argued** (Amendment A3).
* **E2**  every ``mappo1000`` arm sits at ``r >= 0.99``.  The null control.
* **E3**  re-registered by Amendment A6 over the tiers that exist in P4.6 -- ``mappo1000``,
  ``mappo500``, ``maxpressure``, ``fixedtime``, ``random`` -- predicting a monotone-ish admission
  profile with ``mappo1000`` highest and ``random`` lowest, **scored on hz1x1 only**, because E3's
  falsifier was written against an hz1x1 measurement.
  🔒 **grid4x4's profile is reported as a MEASUREMENT WITH ITS OWN ROW, never as a scoping
  exclusion** (Amendment A4): ``never_entered`` there is expected to be 0 or near it, and a
  materially non-zero value is a finding in its own right.

THE FILESYSTEM-MUTATION BARRIER
-------------------------------
``output/`` in the main tree holds every checkpoint under nine manifests and is the only copy; it is
gitignored and there is no backup.  Every write here goes through ``tier_sweep.assert_writable`` with
each sibling ``output/*`` directory passed as a protected root, so a path bug refuses instead of
overwriting.  Validation completes before the first byte is written and a refused run creates no
directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "BEHAVIOUR_METHOD",
    "E2_ADMISSION_FLOOR",
    "E3_SCORED_SCENARIO",
    "ESCALATION_DRAWS",
    "PROBE_DRAWS",
    "PROBE_SCENARIOS",
    "PROBE_SEEDS",
    "SCIENCE_VERDICT_STRINGS",
    "AdmissionCounts",
    "AdmissionEpisode",
    "ArmSource",
    "CellSummary",
    "DrawRestoration",
    "ProbeRoots",
    "ProbeScenario",
    "ReferenceCheck",
    "admission_artifact",
    "admission_spread",
    "assert_no_science_verdict",
    "build_factory",
    "cell_admission_ratio",
    "check_against_reference",
    "committed_reference",
    "created_from_flow",
    "default_protected_roots",
    "env_settings_for",
    "paired_admission_difference",
    "per_seed_admission_ratios",
    "probe_cell",
    "probe_episode",
    "read_admission_at_horizon",
    "reconcile_admission",
    "restore_draws",
    "score_e1",
    "score_e2",
    "score_e3",
    "seeds_for",
    "summarise_cell",
]

ARTIFACT_FORMAT_VERSION = "p8.4a-admission/1.0"

#: The ten held-out draws declared in ``docs/plans/p8.4a.md`` section 3, before any number existed.
PROBE_DRAWS: tuple[int, ...] = tuple(range(1000, 1010))

#: The full held-out pool an escalated arm is re-run over (``BRIEF_31`` section 4).
ESCALATION_DRAWS: tuple[int, ...] = tuple(range(1000, 1100))

#: The registered training seeds, reused so an arm's five slots line up with its committed cell.
PROBE_SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)

BEHAVIOUR_METHOD = "behaviour"

#: E2's registered band: anything below this on a ``mappo1000`` arm indicts the probe, not the science.
E2_ADMISSION_FLOOR = 0.99

#: E3 is scored on this scenario only (Amendment A4).  grid4x4 is reported, never scored against E3.
E3_SCORED_SCENARIO = "hz1x1"

#: Strings that would constitute a verdict on the SCIENCE.  ``BRIEF_31`` section 6 forbids all of
#: them: this artifact reports admission and scores the three registered instrument predictions, and
#: says nothing about whether P5.2's headline survives.
SCIENCE_VERDICT_STRINGS = frozenset(
    {
        "artefact",
        "artifact_result",
        "headline_safe",
        "headline_unsafe",
        "metric_is_wrong",
        "result_is_safe",
        "explained_by_admission",
        "not_an_artefact",
    }
)


@dataclass(frozen=True)
class ProbeScenario:
    """One scenario's cell inventory, fixed in ``docs/plans/p8.4a.md`` section 3."""

    name: str
    scenario_key: str
    scenario_id: str
    sim_config: str
    tiers: tuple[str, ...]
    methods: tuple[str, ...]


#: The two scenarios carrying headline claims.  hz1x1's method list is
#: ``offline.method_tier_grid.METHODS``; grid4x4's is ``offline.tier_sweep.METHODS`` -- read from
#: those modules by a test rather than trusted here, because ``BRIEF_31`` section 4 wrote both arm
#: lists from memory of a neighbouring task and got both wrong (Amendment A6).
PROBE_SCENARIOS: Mapping[str, ProbeScenario] = {
    "hz1x1": ProbeScenario(
        name="hz1x1",
        scenario_key="cityflow1x1",
        scenario_id="cityflow1x1",
        sim_config="configs/sim/cityflow1x1.json",
        tiers=("mappo1000", "mappo500", "maxpressure", "fixedtime", "random"),
        methods=("bc", "bc_top10", "iql", "dt"),
    ),
    "grid4x4": ProbeScenario(
        name="grid4x4",
        scenario_key="cityflow_grid4x4",
        scenario_id="cityflow_grid4x4",
        sim_config="configs/sim/cityflow_grid4x4.json",
        tiers=("mappo1000", "random"),
        methods=("dt_spatial", "dt_nomix", "bc", "bc_top10", "bc_top10_perix", "iql"),
    ),
}

#: The corpus directories behind each ``(scenario, tier)``, used to read the collection manifest the
#: evaluation env settings and the behaviour policy's own checkpoint come out of.
TIER_CORPUS_DIRS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "hz1x1": {
        "mappo1000": tuple(f"cf_hz1x1__mappo1000__seed{s}" for s in PROBE_SEEDS),
        "mappo500": tuple(f"cf_hz1x1__mappo500__seed{s}" for s in PROBE_SEEDS),
        "maxpressure": ("cf_hz1x1__maxpressure",),
        "fixedtime": ("cf_hz1x1__fixedtime",),
        "random": ("cf_hz1x1__random",),
    },
    "grid4x4": {
        "mappo1000": tuple(f"cf_grid4x4__mappo1000__seed{s}" for s in PROBE_SEEDS),
        "random": ("cf_grid4x4__random",),
    },
}


@dataclass(frozen=True)
class ProbeRoots:
    """Where everything this module reads and writes lives.

    ``repo_root`` is this worktree (committed artifacts under ``docs/data/``); ``output_root`` is the
    MAIN tree's ``output/``, which holds the only copy of every checkpoint and eval cell and is read
    by absolute path, never symlinked in.
    """

    repo_root: Path
    corpus_root: Path
    draws_root: Path
    output_root: Path
    work_dir: Path


# ----------------------------------------------------------------------
# The measurement identity
# ----------------------------------------------------------------------


def created_from_flow(flow_path: str | Path, *, horizon_seconds: int) -> int:
    """R5: how many vehicles a draw's flow file creates inside a *horizon_seconds* episode.

    ``Flow::nextStep`` (``CityFlow/src/flow/flow.cpp:6-22``) is called once per engine step with
    ``currentTime`` taking the values ``0 .. horizon_seconds - 1``.  An entry fires when
    ``currentTime >= startTime`` and ``currentTime <= endTime``, so an entry with
    ``startTime == endTime`` emits exactly one vehicle, at ``currentTime == startTime``, and only if
    ``startTime <= horizon_seconds - 1``.

    ⚠️ **The ``startTime == endTime`` shape is ASSERTED, not assumed.**  A multi-vehicle entry
    (``startTime < endTime``) would emit ``floor((endTime - startTime) / interval) + 1`` vehicles and
    silently break this count, so it raises instead.
    """
    raise NotImplementedError


@dataclass(frozen=True)
class AdmissionCounts:
    """The reconciled admission counts of one episode at its horizon."""

    created: int
    entered: int
    never_entered: int
    completed: int
    running: int
    waiting: int

    @property
    def entered_fraction(self) -> float:
        """``entered / created``; ``0.0`` when a draw creates no vehicles at all."""
        raise NotImplementedError


def reconcile_admission(
    *,
    n_running: int,
    n_with_waiting: int,
    n_depart_time: int,
    n_completed: int,
    created_from_flow: int,
) -> AdmissionCounts:
    """R1-R5, cross-checked with ``==`` on ``int``s, or raise.

    Two independent routes to ``entered`` (the metric's ``depart_time`` set against
    ``completed + running``) and two to ``created`` (the engine's pool arithmetic against the flow
    file) must agree exactly.  ``==`` and not ``isclose``: these are counts, and a route that is one
    vehicle out is a defect, not a tolerance.
    """
    raise NotImplementedError


def read_admission_at_horizon(env: Any, *, created: int) -> AdmissionCounts:
    """Read the live engine and metrics object at the horizon and reconcile them.

    Called after :func:`offline.horizon_metric.horizon_rollout` returns and before ``env.close()``.
    Nothing here advances the simulation: ``get_vehicles`` and the metrics' ``_episode`` dict are
    reads.  ``_episode["depart_time"]`` and ``_episode["completed"]`` are populated on every step
    regardless of the requested metric set, because ``metrics/base.py:208-215``'s ``update`` calls
    ``_run_step_hooks`` unconditionally -- which is why this does not need
    ``count_of_vehicles_completing_journey`` to be requested, and must not, since the collection
    manifests do not request it.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------
# One episode
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissionEpisode:
    """One arm, one seed, one draw: both ATT definitions and the admission counts behind them."""

    scenario: str
    tier: str
    method: str
    arm: str
    seed: int | None
    draw_id: int
    created: int
    entered: int
    never_entered: int
    entered_fraction: float
    completed_at_horizon: int
    running_at_horizon: int
    waiting_at_horizon: int
    att_ours: float
    att_engine: float
    horizon_vehicle_count: float
    episode_reward: float
    seconds: float

    def as_record(self) -> dict[str, Any]:
        """The JSON row, with keys sorted by the artifact writer rather than here."""
        raise NotImplementedError


def probe_episode(
    *,
    scenario: str,
    tier: str,
    method: str,
    arm: str,
    seed: int | None,
    draw_id: int,
    config_path: str | Path,
    env_settings: Mapping[str, Any],
    scenario_id: str,
    choose_action_factory: Callable[[Any], Callable[[Any, dict[str, Any]], np.ndarray]],
    engine_seed: int,
    created: int,
) -> AdmissionEpisode:
    """Roll one episode and read the admission counts at its horizon.

    The body mirrors ``dt_gate.evaluate_arm``'s loop exactly -- same ``EnvSpec``, same
    ``horizon_rollout(env, factory(env), episodes=1, seed=engine_seed)``, same ``env.close()`` in a
    ``finally`` -- with the engine and metrics read inserted between the rollout and the close.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------
# Wiring: env settings, arms, seeds
# ----------------------------------------------------------------------


def env_settings_for(scenario: str, tier: str, roots: ProbeRoots) -> dict[str, Any]:
    """The evaluation env settings of one tier, read from its own collection manifests.

    Every directory of the tier must agree; a disagreement raises rather than picking the first,
    because the settings decide what the episode IS and a silent pick would compare two things.
    """
    raise NotImplementedError


def seeds_for(scenario: str, tier: str, method: str) -> tuple[int | None, ...]:
    """The seed slots of one cell.

    Five for every learned arm and for the stochastic or checkpointed anchors; a single ``None`` for
    ``behaviour@maxpressure`` and ``behaviour@fixedtime``, which are deterministic and whose
    committed cells carry ``seed: null``.
    """
    raise NotImplementedError


@dataclass(frozen=True)
class ArmSource:
    """Where an arm's policy came from, recorded so a cell can be traced to a file."""

    kind: str
    detail: str
    checkpoint: str | None = None
    checkpoint_sha256: str | None = None


def build_factory(
    scenario: str,
    tier: str,
    method: str,
    seed: int | None,
    roots: ProbeRoots,
    *,
    device: str | None,
    config_path: str | Path,
) -> tuple[Callable[[Any], Callable[[Any, dict[str, Any]], np.ndarray]], ArmSource]:
    """The action factory for one arm-seed, and the provenance of the policy behind it.

    Every factory is imported from the module that produced the committed cell rather than
    reimplemented -- ``dt_gate._maxpressure_factory`` / ``_mappo_factory``,
    ``method_tier_grid._random_factory`` / ``_fixedtime_factory`` / ``_dt_factory``,
    ``offline_baselines._baseline_factory``, ``agent.SpatialDTAgent`` -- because a second
    implementation of a protocol is exactly how two arms stop being comparable.

    **MAPPO anchors are read, not guessed:** the checkpoint path and its sha256 come from the tier's
    own collection manifest ``run_metadata``, and the digest is verified before the policy is built.
    """
    raise NotImplementedError


def probe_cell(
    *,
    scenario: str,
    tier: str,
    method: str,
    draw_ids: Sequence[int],
    roots: ProbeRoots,
    engine_seed: int,
    device: str | None,
    seeds: Sequence[int | None] | None = None,
) -> list[AdmissionEpisode]:
    """Probe every ``(seed, draw)`` of one cell, in seed-then-draw order."""
    raise NotImplementedError


# ----------------------------------------------------------------------
# The exact-reproduction check (docs/plans/p8.4a.md section 5, conflict C1)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceCheck:
    """The outcome of comparing a probed cell against its committed ``att_horizon``."""

    source: str
    n_compared: int
    n_missing: int
    mismatches: tuple[Mapping[str, Any], ...]

    @property
    def exact(self) -> bool:
        """True only when every probed episode matched and none was missing."""
        raise NotImplementedError


def committed_reference(
    scenario: str, tier: str, method: str, roots: ProbeRoots
) -> tuple[dict[tuple[int | None, int], float], str]:
    """The committed ``att_horizon`` of one cell, keyed by ``(seed, draw_id)``, and its source path.

    ``BRIEF_31``'s Definition of Done named ``att_per_step[-1]``, which is a **corpus** field; a
    held-out evaluation stores ``att_horizon``.  Amendment A6 accepted the correction and called it
    stronger.  Sources, in the order this function tries them:

    * hz1x1, all 25 cells -- ``docs/data/p4_6_grid.json``, arm ``"<method>@<tier>"``.
    * grid4x4 ``mappo1000``, the five P5.1 arms and the behaviour anchor --
      ``output/p5_1/eval_<method>.json``, arm ``"<method>@grid4x4_mappo1000"``.
    * grid4x4 ``mappo1000``, ``bc_top10_perix`` -- ``output/p5_2/eval_mappo1000_bc_top10_perix.json``.
    * grid4x4 ``random``, the six method arms -- ``output/p5_2/eval_random_<method>.json``.
    * grid4x4 ``random``, the behaviour anchor -- ``output/p5_2/eval_mappo1000_random.json``, which
      ``tier_sweep.campaign_cell_manifest():384-394`` records as doubling as the random tier's
      behaviour anchor.  **This module asserts the env-settings equality that claim rests on rather
      than inheriting it.**
    """
    raise NotImplementedError


def check_against_reference(
    episodes: Sequence[AdmissionEpisode],
    reference: Mapping[tuple[int | None, int], float],
    source: str,
) -> ReferenceCheck:
    """Compare every probed ``att_ours`` against its committed value with ``==``.

    ``==`` and never ``isclose``: a replay that is faithful reproduces the float exactly, and a
    tolerance here would hide the one failure mode this check exists to catch -- that the probe is
    measuring a different episode than the one the paper reported.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------
# Scoring (docs/plans/p8.4a.md section 4, Amendment A3/A4/A6)
# ----------------------------------------------------------------------


def cell_admission_ratio(episodes: Sequence[AdmissionEpisode]) -> float:
    """``sum(entered) / sum(created)`` -- a population ratio, not a mean of per-episode ratios."""
    raise NotImplementedError


def per_seed_admission_ratios(
    episodes: Sequence[AdmissionEpisode],
) -> dict[str, float]:
    """The admission ratio of each seed slot, keyed by ``str(seed)`` and sorted.

    Reported always: ``BRIEF_31`` section 5 forbids a bare pooled mean, and A5's own error was a
    statistic computed on a subset and stated of the population.
    """
    raise NotImplementedError


def admission_spread(ratios: Mapping[str, float]) -> float:
    """``max - min`` over the per-seed ratios; ``0.0`` for a single-seeded arm."""
    raise NotImplementedError


@dataclass(frozen=True)
class CellSummary:
    """One cell, summarised at the grain the artifact reports."""

    scenario: str
    tier: str
    method: str
    arm: str
    n_episodes: int
    seeds: tuple[str, ...]
    draw_ids: tuple[int, ...]
    created_total: int
    entered_total: int
    never_entered_total: int
    admission_ratio: float
    per_seed_admission: Mapping[str, float]
    admission_spread: float
    att_ours_mean: float
    att_engine_mean: float
    att_difference_mean: float
    horizon_vehicle_count_mean: float
    seconds_total: float

    def as_record(self) -> dict[str, Any]:
        """The JSON block for one cell."""
        raise NotImplementedError


def summarise_cell(episodes: Sequence[AdmissionEpisode]) -> CellSummary:
    """Summarise one cell, refusing a mixed-arm input."""
    raise NotImplementedError


def paired_admission_difference(
    arm_episodes: Sequence[AdmissionEpisode],
    behaviour_episodes: Sequence[AdmissionEpisode],
) -> dict[str, Any]:
    """Paired ``entered_fraction`` difference over shared draws, with a 95 % CI.

    Pairing is by ``(seed, draw_id)``.  When the anchor is single-seeded -- ``seed=None``, as
    ``behaviour@maxpressure`` and ``behaviour@fixedtime`` are -- the one anchor episode of a draw
    pairs against every arm seed's episode of that draw, and the record says so.
    """
    raise NotImplementedError


def score_e1(cells: Mapping[str, Mapping[str, CellSummary]]) -> dict[str, Any]:
    """E1, scored exactly as registered.

    ``deficit = r(behaviour) - r(arm)``, ``Delta = max(spread_behaviour, spread_arm)``.
    ``deficit <= 0`` -> ``holds``; ``0 < deficit <= Delta`` -> ``close``; ``deficit > Delta`` ->
    ``falsified``.  **Escalation to the full 100 draws is triggered by ``deficit > 0``, whatever
    ``Delta`` says** -- the permissive verdict threshold is only acceptable because the escalation
    threshold is zero (Amendment A3).
    """
    raise NotImplementedError


def score_e2(cells: Mapping[str, Mapping[str, CellSummary]]) -> dict[str, Any]:
    """E2: every ``mappo1000`` arm at ``r >= 0.99``, on both scenarios.  The null control."""
    raise NotImplementedError


def score_e3(cells: Mapping[str, Mapping[str, CellSummary]]) -> dict[str, Any]:
    """E3 as re-registered by Amendment A6, scored on hz1x1's five behaviour anchors only.

    The prediction is a monotone-ish admission profile with ``mappo1000`` highest and ``random``
    lowest, scored as ``r(behaviour, mappo1000) > r(behaviour, random)`` with the full five-tier
    profile and its rank order reported.

    🔒 grid4x4's profile is returned under ``grid4x4_profile`` **as a measurement with its own row**
    (Amendment A4).  It is not scored against E3 and is not described as excluded from it: E3's
    falsifier -- *"a flat profile indicts the replay"* -- was written against an hz1x1 measurement,
    and applying it to grid4x4 would condemn a correct result.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------
# Draw restoration (Amendment A1: Gate -1)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DrawRestoration:
    """What ``offline.materialise_draws`` did, reported rather than trusted."""

    scenario_key: str
    survivors: tuple[int, ...]
    restored: tuple[int, ...]
    actions: Mapping[str, str]
    flow_sha256: Mapping[str, str]
    n_vehicles: Mapping[str, int]
    survivors_reproduced: bool


def restore_draws(
    sim_config: str | Path,
    *,
    survivors: Sequence[int],
    wanted: Sequence[int],
    out_root: str | Path,
) -> DrawRestoration:
    """Gate -1: regenerate the survivors FIRST, report byte-identity, then create what is missing.

    ``offline.materialise_draws.materialise`` is no-op-or-refuse: it returns ``action == "kept"``
    only when :func:`offline.materialise_draws._existing_conflict` finds byte equality of the
    rendered files and field equality of the provenance record, and raises ``FileExistsError``
    without writing anything when it does not.  ``force`` is never passed here, so ``"replaced"`` is
    unreachable -- and is asserted against regardless.

    🔒 **If any survivor does not come back ``"kept"``, this raises and nothing new is written.**
    Every downstream grid4x4 number would otherwise be measured on draws that are not P5.2's,
    undetectably.  *The tool being no-op-or-refuse is the mechanism; reporting what it found is the
    evidence.*
    """
    raise NotImplementedError


# ----------------------------------------------------------------------
# The artifact
# ----------------------------------------------------------------------


def default_protected_roots(roots: ProbeRoots) -> tuple[Path, ...]:
    """Every directory this task may never write under.

    The corpus, and every immediate child of ``output/`` other than this task's own work directory.
    Resolution happens in ``tier_sweep.protected_roots_from``, so a relative path, a ``..``
    traversal and a symlink all resolve into the protected root and are refused.
    """
    raise NotImplementedError


def assert_no_science_verdict(payload: Any) -> None:
    """Refuse to emit a verdict on the science anywhere in the artifact.

    Runs ``method_tier_grid.assert_no_verdicts`` for the equivalence-verdict class this repo already
    forbids, then walks the payload again for :data:`SCIENCE_VERDICT_STRINGS` -- the words that would
    turn a measurement of admission into a claim about whether P5.2's headline survives, which
    ``BRIEF_31`` section 6 reserves.
    """
    raise NotImplementedError


def admission_artifact(
    *,
    cells: Mapping[str, Mapping[str, CellSummary]],
    episodes: Sequence[AdmissionEpisode],
    references: Mapping[str, ReferenceCheck],
    restoration: Mapping[str, DrawRestoration],
    timing: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble ``docs/data/p8_4a_admission.json``.

    Carries the registered rules verbatim, every episode, every cell at per-seed grain, the
    reference checks, the draw-restoration report, the per-cell timing P8.4b's cost model needs, and
    E1/E2/E3 scored.  It carries **no** verdict on the science.
    """
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code."""
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
