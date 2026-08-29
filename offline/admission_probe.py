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

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
    "WorkingDirectoryCheck",
    "assert_cwd_renders_the_recorded_scenario_dir",
    "assert_no_science_verdict",
    "build_factory",
    "cell_admission_ratio",
    "cell_files",
    "code_provenance",
    "check_against_reference",
    "committed_reference",
    "created_from_flow",
    "default_protected_roots",
    "env_settings_for",
    "escalation_targets",
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
    "work_file_name",
]

ARTIFACT_FORMAT_VERSION = "p8.4a-admission/1.0"

#: The ten held-out draws declared in ``docs/plans/p8.4a.md`` section 3, before any number existed.
PROBE_DRAWS: tuple[int, ...] = tuple(range(1000, 1010))

#: The full held-out pool an escalated arm is re-run over (``BRIEF_31`` section 4).
ESCALATION_DRAWS: tuple[int, ...] = tuple(range(1000, 1100))

#: The registered training seeds, reused so an arm's five slots line up with its committed cell.
PROBE_SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)

BEHAVIOUR_METHOD = "behaviour"

#: Tiers whose behaviour policy is deterministic, so their committed cells carry one ``seed: null``
#: slot rather than five.  ``maxpressure`` is a function of the state and ``fixedtime`` of the clock.
DETERMINISTIC_ANCHOR_TIERS: frozenset[str] = frozenset({"maxpressure", "fixedtime"})

#: The declared training budget every checkpoint this module loads must record.  Both campaigns ran
#: at 40,000 and the loaders refuse anything else, which is the mechanical form of "no online model
#: selection" (``PREREGISTRATION`` section 6).
DECLARED_GRADIENT_STEPS = 40_000

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
    horizon = int(horizon_seconds)
    if horizon <= 0:
        raise ValueError(
            f"horizon_seconds must be positive, got {horizon_seconds!r}; an episode of no length "
            "creates no vehicles and the count would be vacuous"
        )
    path = Path(flow_path)
    entries = json.loads(path.read_bytes())
    if not isinstance(entries, list):
        raise ValueError(f"{path}: a CityFlow flow file is a JSON list, got {type(entries).__name__}")

    created = 0
    for index, entry in enumerate(entries):
        if "startTime" not in entry or "endTime" not in entry:
            raise ValueError(
                f"{path}: flow entry {index} has no startTime/endTime, so it cannot be counted"
            )
        start = int(entry["startTime"])
        end = int(entry["endTime"])
        if start != end:
            raise ValueError(
                f"{path}: flow entry {index} has startTime {start} != endTime {end}. This count "
                "models one vehicle per entry, which is exact only for the single-vehicle shape "
                "every materialised draw uses; a repeating entry emits "
                "floor((endTime - startTime) / interval) + 1 vehicles and would be undercounted"
            )
        if start <= horizon - 1:
            created += 1
    return created


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
        if self.created == 0:
            return 0.0
        return self.entered / self.created


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
    counts = {
        "n_running": int(n_running),
        "n_with_waiting": int(n_with_waiting),
        "n_depart_time": int(n_depart_time),
        "n_completed": int(n_completed),
        "created_from_flow": int(created_from_flow),
    }
    for name, value in counts.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    never_entered = counts["n_with_waiting"] - counts["n_running"]
    if never_entered < 0:
        raise ValueError(
            f"never_entered is {never_entered}: get_vehicles(include_waiting=True) returned "
            f"{counts['n_with_waiting']} against {counts['n_running']} without waiting, but R1 "
            "makes the first a superset of the second (engine.cpp:780-789). A negative difference "
            "is a defect in the read, not a measurement"
        )

    entered_r2 = counts["n_depart_time"]
    entered_r3 = counts["n_completed"] + counts["n_running"]
    if entered_r2 != entered_r3:
        raise ValueError(
            f"the two routes to entered disagree: R2 (the metric's depart_time set) says "
            f"{entered_r2} and R3 (completed + running) says {entered_r3}. They can only differ "
            "if a vehicle entered and left inside one decision window, or if the metric's "
            "bookkeeping is broken; either way this episode is not interpretable"
        )

    created_r4 = entered_r2 + never_entered
    if created_r4 != counts["created_from_flow"]:
        raise ValueError(
            f"the two routes to created disagree: R4 (entered + never_entered) says {created_r4} "
            f"and R5 (the flow file, counting entries that fire inside the horizon) says "
            f"{counts['created_from_flow']}. Either R5's single-vehicle-per-entry assumption or "
            "the episode horizon is wrong, and the admission ratio would then be measured against "
            "the wrong denominator"
        )

    return AdmissionCounts(
        created=created_r4,
        entered=entered_r2,
        never_entered=never_entered,
        completed=counts["n_completed"],
        running=counts["n_running"],
        waiting=never_entered,
    )


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
    engine = getattr(env, "_eng", None)
    metrics = getattr(env, "_metrics", None)
    if engine is None or metrics is None or not hasattr(metrics, "_episode"):
        raise TypeError(
            f"{type(env).__name__} exposes no live CityFlow engine and metrics pair "
            "(_eng / _metrics._episode), so its admission counts cannot be read. This probe is "
            "CityFlow-only by construction: SUMO and MOSS report travel time from the engine and "
            "have no insertion buffer of this shape"
        )

    n_running = len(engine.get_vehicles(include_waiting=False))
    n_with_waiting = len(engine.get_vehicles(include_waiting=True))
    episode_state = metrics._episode
    return reconcile_admission(
        n_running=n_running,
        n_with_waiting=n_with_waiting,
        n_depart_time=len(episode_state["depart_time"]),
        n_completed=len(episode_state["completed"]),
        created_from_flow=int(created),
    )


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
    seconds_rollout: float = 0.0

    def as_record(self) -> dict[str, Any]:
        """The JSON row, with keys sorted by the artifact writer rather than here.

        ``seconds`` is the whole per-episode cost -- env construction, policy construction, rollout
        and the horizon reads -- because that is the quantity P8.4b's cost model needs.
        ``seconds_rollout`` isolates the simulation from the ``torch.load`` that ``evaluate_arm``'s
        contract puts in front of every draw.
        """
        return {
            "scenario": self.scenario,
            "tier": self.tier,
            "method": self.method,
            "arm": self.arm,
            "seed": self.seed,
            "draw_id": int(self.draw_id),
            "created": int(self.created),
            "entered": int(self.entered),
            "never_entered": int(self.never_entered),
            "entered_fraction": float(self.entered_fraction),
            "completed_at_horizon": int(self.completed_at_horizon),
            "running_at_horizon": int(self.running_at_horizon),
            "waiting_at_horizon": int(self.waiting_at_horizon),
            "att_ours": float(self.att_ours),
            "att_engine": float(self.att_engine),
            "att_difference": float(self.att_ours - self.att_engine),
            "horizon_vehicle_count": float(self.horizon_vehicle_count),
            "episode_reward": float(self.episode_reward),
            "seconds": float(self.seconds),
            "seconds_rollout": float(self.seconds_rollout),
        }


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
    from experiments.config import EnvSpec
    from experiments.envs import make_env

    from offline.horizon_metric import horizon_rollout

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"draw {draw_id} has no materialised sim config at {path}; run "
            "offline.materialise_draws for the held-out pool first"
        )

    started = time.perf_counter()
    env = make_env(
        EnvSpec(
            id=scenario_id,
            backend="cityflow",
            paths={"config": str(path)},
            settings=dict(env_settings),
        )
    )
    try:
        choose_action = choose_action_factory(env)
        rollout_started = time.perf_counter()
        rollout = horizon_rollout(env, choose_action, episodes=1, seed=int(engine_seed))
        rollout_seconds = time.perf_counter() - rollout_started
        counts = read_admission_at_horizon(env, created=int(created))
        att_engine = float(env._eng.get_average_travel_time())
    finally:
        env.close()
    seconds = time.perf_counter() - started

    if counts.running != int(rollout.final_vehicle_count):
        raise ValueError(
            f"{arm} seed {seed} draw {draw_id}: the horizon reader reports "
            f"{rollout.final_vehicle_count} running vehicles from info['vehicle_count'] but the "
            f"engine reports {counts.running} now. Nothing may advance the simulation between the "
            "last step and the admission read, and something did"
        )

    return AdmissionEpisode(
        scenario=scenario,
        tier=tier,
        method=method,
        arm=arm,
        seed=None if seed is None else int(seed),
        draw_id=int(draw_id),
        created=counts.created,
        entered=counts.entered,
        never_entered=counts.never_entered,
        entered_fraction=counts.entered_fraction,
        completed_at_horizon=counts.completed,
        running_at_horizon=counts.running,
        waiting_at_horizon=counts.waiting,
        att_ours=float(rollout.per_episode_horizon[0]),
        att_engine=att_engine,
        horizon_vehicle_count=float(rollout.final_vehicle_count),
        episode_reward=float(rollout.episode_reward),
        seconds=seconds,
        seconds_rollout=rollout_seconds,
    )


# ----------------------------------------------------------------------
# Wiring: env settings, arms, seeds
# ----------------------------------------------------------------------


def env_settings_for(scenario: str, tier: str, roots: ProbeRoots) -> dict[str, Any]:
    """The evaluation env settings of one tier, read from its own collection manifests.

    Every directory of the tier must agree; a disagreement raises rather than picking the first,
    because the settings decide what the episode IS and a silent pick would compare two things.
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


def tier_corpus_dirs(scenario: str, tier: str, roots: ProbeRoots) -> tuple[Path, ...]:
    """The corpus directories behind one ``(scenario, tier)``, refusing a missing one."""
    try:
        names = TIER_CORPUS_DIRS[scenario][tier]
    except KeyError as exc:
        raise ValueError(
            f"no corpus directories are declared for {scenario!r}/{tier!r}; the probe inventory is "
            "docs/plans/p8.4a.md section 3 and does not cover this cell"
        ) from exc
    paths = tuple(Path(roots.corpus_root) / name for name in names)
    missing = [str(p) for p in paths if not (p / "manifest.json").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{scenario}/{tier}: these corpus directories have no manifest.json: {missing}"
        )
    return paths


def seeds_for(scenario: str, tier: str, method: str) -> tuple[int | None, ...]:
    """The seed slots of one cell.

    Five for every learned arm and for the stochastic or checkpointed anchors; a single ``None`` for
    ``behaviour@maxpressure`` and ``behaviour@fixedtime``, which are deterministic and whose
    committed cells carry ``seed: null``.
    """
    _assert_declared_cell(scenario, tier, method)
    if method == BEHAVIOUR_METHOD and tier in DETERMINISTIC_ANCHOR_TIERS:
        return (None,)
    return PROBE_SEEDS


def _assert_declared_cell(scenario: str, tier: str, method: str) -> ProbeScenario:
    """Refuse a cell the plan does not declare, so a typo cannot invent one."""
    if scenario not in PROBE_SCENARIOS:
        raise ValueError(
            f"{scenario!r} is not a probed scenario; docs/plans/p8.4a.md section 3 declares "
            f"{sorted(PROBE_SCENARIOS)}"
        )
    spec = PROBE_SCENARIOS[scenario]
    if tier not in spec.tiers:
        raise ValueError(
            f"{tier!r} is not a probed tier of {scenario}; the declared tiers are {list(spec.tiers)}"
        )
    if method != BEHAVIOUR_METHOD and method not in spec.methods:
        raise ValueError(
            f"{method!r} is not a probed method of {scenario}; the declared arms are "
            f"{list(spec.methods)} plus {BEHAVIOUR_METHOD!r}"
        )
    return spec


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
    _assert_declared_cell(scenario, tier, method)
    if method == BEHAVIOUR_METHOD:
        return _behaviour_factory(scenario, tier, seed, roots, device=device, config_path=config_path)
    return _method_factory(scenario, tier, method, seed, roots, device=device)


def _behaviour_factory(
    scenario: str,
    tier: str,
    seed: int | None,
    roots: ProbeRoots,
    *,
    device: str | None,
    config_path: str | Path,
) -> tuple[Callable[[Any], Any], ArmSource]:
    """The policy that COLLECTED a tier, rebuilt from that tier's own manifest."""
    from offline.dt_gate import _mappo_factory, _maxpressure_factory
    from offline.method_tier_grid import (
        _fixedtime_factory,
        _random_factory,
        fixedtime_collection_settings,
    )

    if tier == "maxpressure":
        return _maxpressure_factory, ArmSource(
            kind="algorithmic",
            detail="algorithms.max_pressure.MaxPressureAgent via dt_gate._maxpressure_factory",
        )

    if tier == "fixedtime":
        manifest = tier_corpus_dirs(scenario, tier, roots)[0] / "manifest.json"
        collected = fixedtime_collection_settings(manifest)
        factory = _fixedtime_factory(config_path, collected)
        return factory, ArmSource(
            kind="plan",
            detail=(
                "offline.policies.fixed_time.make_fixedtime with k="
                f"{collected['fixed_time_k']} and the plan hash asserted against {manifest}"
            ),
            checkpoint_sha256=str(collected["fixed_time_plan_sha256"]),
        )

    if tier == "random":
        if seed is None:
            raise ValueError("the random behaviour anchor is seeded and cannot take seed=None")
        return _random_factory(int(seed)), ArmSource(
            kind="algorithmic",
            detail=(
                "offline.collect._make_random with numpy.random.default_rng"
                f"({int(seed)}) rebuilt per draw, via method_tier_grid._random_factory"
            ),
        )

    if tier in ("mappo1000", "mappo500"):
        if seed is None:
            raise ValueError(f"the {tier} behaviour anchor is per-seed and cannot take seed=None")
        path, digest = _manifest_checkpoint(scenario, tier, int(seed), roots)
        return _mappo_factory(str(path), device), ArmSource(
            kind="checkpoint",
            detail=(
                "agent.MAPPOAgent via dt_gate._mappo_factory; path and digest read from the "
                "collecting run's own manifest run_metadata, never guessed"
            ),
            checkpoint=str(path),
            checkpoint_sha256=digest,
        )

    raise ValueError(f"no behaviour factory is declared for {scenario!r}/{tier!r}")


def _manifest_checkpoint(
    scenario: str, tier: str, seed: int, roots: ProbeRoots
) -> tuple[Path, str]:
    """The checkpoint that collected one ``(tier, seed)``, with its recorded digest verified.

    The corpus manifest records ``checkpoint`` **and** ``checkpoint_sha256``; the path is stored
    relative to the repository root that ran the collection, so it is resolved against the tree that
    actually holds ``output/``.  The digest is then recomputed and compared, so a file that moved,
    was migrated or was rebuilt cannot be substituted silently.
    """
    directory = next(
        d for d in tier_corpus_dirs(scenario, tier, roots) if d.name.endswith(f"seed{seed}")
    )
    metadata = json.loads((directory / "manifest.json").read_bytes())["run_metadata"]
    recorded = metadata.get("checkpoint")
    expected = metadata.get("checkpoint_sha256")
    if not recorded or not expected:
        raise ValueError(
            f"{directory}/manifest.json records no behaviour checkpoint for {tier} seed {seed}; "
            "the anchor cannot be rebuilt from a manifest that does not name its policy"
        )
    relative = Path(recorded)
    candidates = [relative] if relative.is_absolute() else [
        Path(roots.output_root).parent / relative,
        Path(roots.repo_root) / relative,
    ]
    for candidate in candidates:
        if candidate.is_file():
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


def _method_checkpoint(scenario: str, tier: str, method: str, seed: int, roots: ProbeRoots) -> Path:
    """Where the model behind a cell's COMMITTED numbers lives.

    ⚠️ **The grid4x4 ``mappo1000`` rows are the trap this table exists for.**  P5.2 reused P5.1's
    cells for ``dt_spatial``, ``dt_nomix``, ``bc``, ``bc_top10`` and ``iql`` at that tier but
    retrained its own baselines anyway: ``output/p5_2/checkpoints/grid4x4_mappo1000_bc_seed101.pt``
    and ``output/p5_1/checkpoints/grid4x4_mappo1000_bc_seed101.pt`` are **different files**
    (verified by ``cmp`` on 2026-08-28), and the reported cell came from the P5.1 one.  Loading the
    P5.2 file would produce a plausible number that reproduces nothing.
    """
    out = Path(roots.output_root)
    if scenario == "hz1x1":
        if tier == "mappo1000":
            if method == "dt":
                return out / "p4_dt" / f"dt_seed{seed}.pt"
            return out / "p4_4" / "checkpoints" / f"{method}_seed{seed}.pt"
        return out / "p4_6" / "checkpoints" / f"{tier}_{method}_seed{seed}.pt"
    if scenario == "grid4x4":
        name = f"grid4x4_{tier}_{method}_seed{seed}.pt"
        if tier == "mappo1000" and method != "bc_top10_perix":
            return out / "p5_1" / "checkpoints" / name
        return out / "p5_2" / "checkpoints" / name
    raise ValueError(f"no checkpoint layout is declared for scenario {scenario!r}")


def _method_factory(
    scenario: str,
    tier: str,
    method: str,
    seed: int | None,
    roots: ProbeRoots,
    *,
    device: str | None,
) -> tuple[Callable[[Any], Any], ArmSource]:
    """A trained arm, loaded through the same loader that produced its committed cell."""
    from offline.offline_baselines import _baseline_factory

    if seed is None:
        raise ValueError(f"{method}@{tier} is a five-seed arm and cannot take seed=None")
    path = _method_checkpoint(scenario, tier, method, int(seed), roots)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; {method}@{tier} on {scenario} cannot be probed without the "
            "checkpoint that produced its committed cell"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    if method in ("dt_spatial", "dt_nomix"):
        from offline.spatial_mixing import assert_declared_budget

        def spatial(env: Any) -> Callable[[Any, dict[str, Any]], np.ndarray]:
            from agent.SpatialDTAgent import SpatialDTAgent

            assert_declared_budget(str(path), DECLARED_GRADIENT_STEPS, method)
            agent = SpatialDTAgent.from_checkpoint(env, str(path), device=device)
            return lambda _env, info: agent.act(info, explore=False, update_memory=True)

        return spatial, ArmSource(
            kind="checkpoint",
            detail="agent.SpatialDTAgent, budget asserted by spatial_mixing.assert_declared_budget",
            checkpoint=str(path),
            checkpoint_sha256=digest,
        )

    if method == "dt":
        from offline.method_tier_grid import TIERS as HZ_TIERS
        from offline.method_tier_grid import _dt_factory

        target = float(HZ_TIERS[tier].target_rtg)
        return _dt_factory(str(path), DECLARED_GRADIENT_STEPS, target, device), ArmSource(
            kind="checkpoint",
            detail=(
                "agent.DTAgent via method_tier_grid._dt_factory with the declared target_rtg "
                f"{target}; for mappo1000 that is exactly docs/data/p4_training.json's recorded "
                "target, so this path and P4's load_gate_checkpoint condition identically"
            ),
            checkpoint=str(path),
            checkpoint_sha256=digest,
        )

    if method in ("bc", "bc_top10", "bc_top10_perix", "iql"):
        return _baseline_factory(method, str(path), DECLARED_GRADIENT_STEPS, device), ArmSource(
            kind="checkpoint",
            detail="agent BC/IQL via offline_baselines._baseline_factory",
            checkpoint=str(path),
            checkpoint_sha256=digest,
        )

    raise ValueError(f"no action factory is declared for {method!r} at {scenario}/{tier}")


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
) -> tuple[list[AdmissionEpisode], dict[str, ArmSource]]:
    """Probe every ``(seed, draw)`` of one cell, in seed-then-draw order.

    Returns the episodes and the provenance of the policy behind each seed slot, so a cell can be
    traced to the exact file it came from.  ``created`` is computed once per draw and reused across
    seeds: it is a property of the flow file, not of the policy, and computing it once is also what
    makes "identical in every cell" checkable at report time.
    """
    from offline.materialise_draws import draw_config_path

    spec = _assert_declared_cell(scenario, tier, method)
    settings = env_settings_for(scenario, tier, roots)
    horizon = int(settings["max_steps"]) * int(settings["delta_time"])
    slots = tuple(seeds) if seeds is not None else seeds_for(scenario, tier, method)
    arm = f"{method}@{tier}"

    configs: dict[int, Path] = {}
    created: dict[int, int] = {}
    for draw_id in draw_ids:
        config = Path(
            draw_config_path(spec.scenario_key, int(draw_id), out_root=roots.draws_root)
        )
        if not config.is_file():
            raise FileNotFoundError(
                f"draw {draw_id} has no materialised sim config at {config}; run "
                "offline.materialise_draws for the held-out pool first"
            )
        configs[int(draw_id)] = config
        created[int(draw_id)] = created_from_flow(
            config.parent / "flow.json", horizon_seconds=horizon
        )

    produced: list[AdmissionEpisode] = []
    sources: dict[str, ArmSource] = {}
    for slot in slots:
        factory, source = build_factory(
            scenario,
            tier,
            method,
            slot,
            roots,
            device=device,
            config_path=configs[int(draw_ids[0])],
        )
        sources[str(slot)] = source
        print(f"{scenario}/{arm} seed {slot} over {len(draw_ids)} draws", flush=True)
        for draw_id in draw_ids:
            produced.append(
                probe_episode(
                    scenario=scenario,
                    tier=tier,
                    method=method,
                    arm=arm,
                    seed=slot,
                    draw_id=int(draw_id),
                    config_path=configs[int(draw_id)],
                    env_settings=settings,
                    scenario_id=spec.scenario_id,
                    choose_action_factory=factory,
                    engine_seed=int(engine_seed),
                    created=created[int(draw_id)],
                )
            )

    expected = {(slot, int(d)) for slot in slots for d in draw_ids}
    got = {(e.seed, e.draw_id) for e in produced}
    if got != expected:
        raise ValueError(
            f"{scenario}/{arm}: {len(got)} episodes against {len(expected)} requested "
            f"(missing {len(expected - got)}, unexpected {len(got - expected)})"
        )
    return produced, sources


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
        """True only when every probed episode matched and none was missing.

        ``n_compared == 0`` is deliberately **not** exact: an empty comparison is the shape a
        silently mis-keyed lookup takes, and reporting it as a pass is how a cell with no reference
        at all would read as verified.
        """
        return self.n_compared > 0 and self.n_missing == 0 and not self.mismatches


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
    _assert_declared_cell(scenario, tier, method)
    if scenario == "hz1x1":
        path = Path(roots.repo_root) / "docs/data/p4_6_grid.json"
        return _reference_from(path, f"{method}@{tier}"), str(path)

    out = Path(roots.output_root)
    if tier == "mappo1000":
        if method == "bc_top10_perix":
            path = out / "p5_2" / "eval_mappo1000_bc_top10_perix.json"
            return _reference_from(path, "bc_top10_perix@mappo1000"), str(path)
        path = out / "p5_1" / f"eval_{method}.json"
        return _reference_from(path, f"{method}@grid4x4_mappo1000"), str(path)

    if tier == "random":
        if method == BEHAVIOUR_METHOD:
            # tier_sweep.campaign_cell_manifest():384-394 records that this one cell serves every
            # tier AND doubles as the random tier's behaviour anchor, on the ground that the env
            # settings are identical across all four tier manifests and the factory is a function
            # of the seed alone.  The settings half of that ground is asserted here rather than
            # inherited: an inherited "(verified)" is exactly the claim A5 turned out to be.
            here = env_settings_for(scenario, "random", roots)
            there = env_settings_for(scenario, "mappo1000", roots)
            if here != there:
                raise ValueError(
                    "the grid4x4 random and mappo1000 tiers disagree on the evaluation env "
                    "settings, so eval_mappo1000_random.json cannot serve as the random tier's "
                    "behaviour anchor; tier_sweep.campaign_cell_manifest's note rests on that "
                    "equality and it does not hold in this tree"
                )
            path = out / "p5_2" / "eval_mappo1000_random.json"
            source = (
                f"{path} (the shared collapse reference, which doubles as the random tier's "
                "behaviour anchor; env-settings equality re-verified here)"
            )
            return _reference_from(path, "random@mappo1000"), source
        path = out / "p5_2" / f"eval_random_{method}.json"
        return _reference_from(path, f"{method}@random"), str(path)

    raise ValueError(f"no committed reference is declared for {scenario}/{method}@{tier}")


def _reference_from(path: Path, arm: str) -> dict[tuple[int | None, int], float]:
    """Every ``(seed, draw_id) -> att_horizon`` of one arm in one committed artifact."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist, so {arm} has no committed att_horizon to reproduce; the "
            "fidelity check cannot be skipped by treating a missing reference as agreement"
        )
    payload = json.loads(path.read_bytes())
    rows = [row for row in payload["episodes"] if row["arm"] == arm]
    if not rows:
        arms = sorted({str(row["arm"]) for row in payload["episodes"]})
        raise ValueError(f"{path} holds no episodes for arm {arm!r}; it holds {arms}")
    reference: dict[tuple[int | None, int], float] = {}
    for row in rows:
        seed = None if row.get("seed") is None else int(row["seed"])
        key = (seed, int(row["draw_id"]))
        if key in reference:
            raise ValueError(f"{path}: {arm} has two episodes for seed {seed} draw {key[1]}")
        reference[key] = float(row["att_horizon"])
    return reference


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
    compared = 0
    missing = 0
    mismatches: list[Mapping[str, Any]] = []
    for episode in episodes:
        key = (episode.seed, int(episode.draw_id))
        if key not in reference:
            missing += 1
            continue
        compared += 1
        committed = reference[key]
        if episode.att_ours != committed:
            mismatches.append(
                {
                    "arm": episode.arm,
                    "seed": episode.seed,
                    "draw_id": int(episode.draw_id),
                    "probed": float(episode.att_ours),
                    "committed": float(committed),
                    "difference": float(episode.att_ours - committed),
                }
            )
    return ReferenceCheck(
        source=source,
        n_compared=compared,
        n_missing=missing,
        mismatches=tuple(mismatches),
    )


# ----------------------------------------------------------------------
# Scoring (docs/plans/p8.4a.md section 4, Amendment A3/A4/A6)
# ----------------------------------------------------------------------


def cell_admission_ratio(episodes: Sequence[AdmissionEpisode]) -> float:
    """``sum(entered) / sum(created)`` -- a population ratio, not a mean of per-episode ratios."""
    if not episodes:
        raise ValueError("cell_admission_ratio received no episodes")
    created = sum(int(e.created) for e in episodes)
    if created == 0:
        raise ValueError(
            "every episode in this cell created zero vehicles, so an admission ratio is undefined"
        )
    return sum(int(e.entered) for e in episodes) / created


def per_seed_admission_ratios(
    episodes: Sequence[AdmissionEpisode],
) -> dict[str, float]:
    """The admission ratio of each seed slot, keyed by ``str(seed)`` and sorted.

    Reported always: ``BRIEF_31`` section 5 forbids a bare pooled mean, and A5's own error was a
    statistic computed on a subset and stated of the population.
    """
    if not episodes:
        raise ValueError("per_seed_admission_ratios received no episodes")
    by_seed: dict[str, list[AdmissionEpisode]] = {}
    for episode in episodes:
        by_seed.setdefault(str(episode.seed), []).append(episode)
    return {seed: cell_admission_ratio(rows) for seed, rows in sorted(by_seed.items())}


def admission_spread(ratios: Mapping[str, float]) -> float:
    """``max - min`` over the per-seed ratios; ``0.0`` for a single-seeded arm."""
    if not ratios:
        raise ValueError("admission_spread received no per-seed ratios")
    values = [float(v) for v in ratios.values()]
    return float(max(values) - min(values))


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
        return {
            "scenario": self.scenario,
            "tier": self.tier,
            "method": self.method,
            "arm": self.arm,
            "n_episodes": int(self.n_episodes),
            "seeds": list(self.seeds),
            "draw_ids": list(self.draw_ids),
            "created_total": int(self.created_total),
            "entered_total": int(self.entered_total),
            "never_entered_total": int(self.never_entered_total),
            "admission_ratio": float(self.admission_ratio),
            "per_seed_admission": dict(self.per_seed_admission),
            "admission_spread": float(self.admission_spread),
            "att_ours_mean": float(self.att_ours_mean),
            "att_engine_mean": float(self.att_engine_mean),
            "att_difference_mean": float(self.att_difference_mean),
            "horizon_vehicle_count_mean": float(self.horizon_vehicle_count_mean),
            "seconds_total": float(self.seconds_total),
            "seconds_per_episode": float(self.seconds_total / self.n_episodes),
        }


def summarise_cell(episodes: Sequence[AdmissionEpisode]) -> CellSummary:
    """Summarise one cell, refusing a mixed-arm input."""
    if not episodes:
        raise ValueError("summarise_cell received no episodes")
    arms = {e.arm for e in episodes}
    if len(arms) != 1:
        raise ValueError(
            f"summarise_cell describes ONE cell but was given the arms {sorted(arms)}; mixing two "
            "is how a tier ends up wearing two labels"
        )
    scenarios = {e.scenario for e in episodes}
    if len(scenarios) != 1:
        raise ValueError(
            f"summarise_cell was given episodes from the scenarios {sorted(scenarios)}"
        )
    keys = [(e.seed, e.draw_id) for e in episodes]
    if len(set(keys)) != len(keys):
        raise ValueError("summarise_cell was given the same (seed, draw) twice")

    per_seed = per_seed_admission_ratios(episodes)
    first = episodes[0]
    return CellSummary(
        scenario=first.scenario,
        tier=first.tier,
        method=first.method,
        arm=first.arm,
        n_episodes=len(episodes),
        seeds=tuple(sorted(per_seed)),
        draw_ids=tuple(sorted({int(e.draw_id) for e in episodes})),
        created_total=sum(int(e.created) for e in episodes),
        entered_total=sum(int(e.entered) for e in episodes),
        never_entered_total=sum(int(e.never_entered) for e in episodes),
        admission_ratio=cell_admission_ratio(episodes),
        per_seed_admission=per_seed,
        admission_spread=admission_spread(per_seed),
        att_ours_mean=float(np.mean([e.att_ours for e in episodes])),
        att_engine_mean=float(np.mean([e.att_engine for e in episodes])),
        att_difference_mean=float(np.mean([e.att_ours - e.att_engine for e in episodes])),
        horizon_vehicle_count_mean=float(
            np.mean([e.horizon_vehicle_count for e in episodes])
        ),
        seconds_total=float(sum(e.seconds for e in episodes)),
    )


def paired_admission_difference(
    arm_episodes: Sequence[AdmissionEpisode],
    behaviour_episodes: Sequence[AdmissionEpisode],
) -> dict[str, Any]:
    """Paired ``entered_fraction`` difference over shared draws, with a 95 % CI.

    Pairing is by ``(seed, draw_id)``.  When the anchor is single-seeded -- ``seed=None``, as
    ``behaviour@maxpressure`` and ``behaviour@fixedtime`` are -- the one anchor episode of a draw
    pairs against every arm seed's episode of that draw, and the record says so.
    """
    from offline.dt_gate import mean_ci95

    if not arm_episodes:
        raise ValueError("paired_admission_difference received no arm episodes")
    if not behaviour_episodes:
        raise ValueError("paired_admission_difference received no behaviour episodes")

    anchor_seeds = {e.seed for e in behaviour_episodes}
    single_seeded = anchor_seeds == {None}
    if single_seeded:
        anchor = {int(e.draw_id): e.entered_fraction for e in behaviour_episodes}
    else:
        anchor = {(e.seed, int(e.draw_id)): e.entered_fraction for e in behaviour_episodes}

    differences: list[float] = []
    for episode in arm_episodes:
        key: Any = int(episode.draw_id) if single_seeded else (episode.seed, int(episode.draw_id))
        if key not in anchor:
            raise ValueError(
                f"{episode.arm} seed {episode.seed} draw {episode.draw_id} has no behaviour "
                f"episode to pair with (the anchor covers {sorted(map(str, anchor))}); dropping it "
                "would shrink the denominator silently"
            )
        differences.append(episode.entered_fraction - anchor[key])

    stats = mean_ci95(differences)
    return {
        "n_pairs": int(stats.n),
        "mean": float(stats.mean),
        "std": float(stats.std),
        "ci95": float(stats.ci95),
        "anchor_is_single_seeded": bool(single_seeded),
        "quantity": "arm entered/created minus behaviour entered/created, paired per episode",
        "draw_ids": sorted({int(e.draw_id) for e in arm_episodes}),
    }


def score_e1(cells: Mapping[str, Mapping[str, CellSummary]]) -> dict[str, Any]:
    """E1, scored exactly as registered.

    ``deficit = r(behaviour) - r(arm)``, ``Delta = max(spread_behaviour, spread_arm)``.
    ``deficit <= 0`` -> ``holds``; ``0 < deficit <= Delta`` -> ``close``; ``deficit > Delta`` ->
    ``falsified``.  **Escalation to the full 100 draws is triggered by ``deficit > 0``, whatever
    ``Delta`` says** -- the permissive verdict threshold is only acceptable because the escalation
    threshold is zero (Amendment A3).
    """
    arms: list[dict[str, Any]] = []
    for scenario in sorted(cells):
        scenario_cells = cells[scenario]
        anchors = {
            summary.tier: summary
            for summary in scenario_cells.values()
            if summary.method == BEHAVIOUR_METHOD
        }
        for arm in sorted(scenario_cells):
            summary = scenario_cells[arm]
            if summary.method == BEHAVIOUR_METHOD:
                continue
            anchor = anchors.get(summary.tier)
            if anchor is None:
                raise ValueError(
                    f"{scenario}/{arm} has no behaviour anchor at tier {summary.tier!r}; E1 is a "
                    "comparison against the policy that produced the arm's data and cannot be "
                    "scored without it"
                )
            deficit = anchor.admission_ratio - summary.admission_ratio
            delta = max(anchor.admission_spread, summary.admission_spread)
            if deficit <= 0.0:
                status = "holds"
            elif deficit <= delta:
                status = "close"
            else:
                status = "falsified"
            arms.append(
                {
                    "scenario": scenario,
                    "tier": summary.tier,
                    "method": summary.method,
                    "arm": arm,
                    "admission_ratio_arm": float(summary.admission_ratio),
                    "admission_ratio_behaviour": float(anchor.admission_ratio),
                    "per_seed_admission_arm": dict(summary.per_seed_admission),
                    "per_seed_admission_behaviour": dict(anchor.per_seed_admission),
                    "spread_arm": float(summary.admission_spread),
                    "spread_behaviour": float(anchor.admission_spread),
                    "deficit": float(deficit),
                    "delta": float(delta),
                    "status": status,
                    "escalate": bool(deficit > 0.0),
                }
            )

    return {
        "prediction": (
            "every learned arm admits at least as many vehicles as the behaviour policy of its own "
            "tier, per cell, on shared draws"
        ),
        "rule": (
            "deficit = r(behaviour) - r(arm); Delta = max(spread_behaviour, spread_arm); "
            "deficit <= 0 holds, 0 < deficit <= Delta is close, deficit > Delta is falsified; "
            "ANY deficit > 0 escalates that arm to the full 100 held-out draws"
        ),
        "registered_in": "docs/plans/p8.4a.md section 4, approved as BRIEF_31 Amendment A3",
        "n_arms": len(arms),
        "n_holds": sum(1 for a in arms if a["status"] == "holds"),
        "n_close": sum(1 for a in arms if a["status"] == "close"),
        "n_falsified": sum(1 for a in arms if a["status"] == "falsified"),
        "escalated_arms": [f"{a['scenario']}/{a['arm']}" for a in arms if a["escalate"]],
        "arms": arms,
    }


def score_e2(cells: Mapping[str, Mapping[str, CellSummary]]) -> dict[str, Any]:
    """E2: every ``mappo1000`` arm at ``r >= 0.99``, on both scenarios.  The null control."""
    arms: list[dict[str, Any]] = []
    for scenario in sorted(cells):
        for arm in sorted(cells[scenario]):
            summary = cells[scenario][arm]
            if summary.tier != "mappo1000":
                continue
            arms.append(
                {
                    "scenario": scenario,
                    "tier": summary.tier,
                    "arm": arm,
                    "admission_ratio": float(summary.admission_ratio),
                    "per_seed_admission": dict(summary.per_seed_admission),
                    "passes": bool(summary.admission_ratio >= E2_ADMISSION_FLOOR),
                }
            )
    return {
        "prediction": "every mappo1000 arm sits at approximately 100 % admission",
        "role": (
            "the null control: anything materially below the floor here indicts the probe before "
            "it indicts the science, and that direction of inference is registered"
        ),
        "floor": E2_ADMISSION_FLOOR,
        "n_arms": len(arms),
        "n_below": sum(1 for a in arms if not a["passes"]),
        "arms": arms,
    }


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

    def profile(scenario: str) -> list[dict[str, Any]]:
        scenario_cells = cells.get(scenario, {})
        order = PROBE_SCENARIOS[scenario].tiers if scenario in PROBE_SCENARIOS else ()
        rows: list[dict[str, Any]] = []
        for tier in order:
            summary = scenario_cells.get(f"{BEHAVIOUR_METHOD}@{tier}")
            if summary is None:
                continue
            rows.append(
                {
                    "tier": tier,
                    "admission_ratio": float(summary.admission_ratio),
                    "per_seed_admission": dict(summary.per_seed_admission),
                    "never_entered_total": int(summary.never_entered_total),
                    "created_total": int(summary.created_total),
                }
            )
        return rows

    hz_profile = profile(E3_SCORED_SCENARIO)
    ratios = {row["tier"]: row["admission_ratio"] for row in hz_profile}
    holds: bool | None = None
    if "mappo1000" in ratios and "random" in ratios:
        holds = bool(ratios["mappo1000"] > ratios["random"])
    monotone = all(
        hz_profile[i]["admission_ratio"] >= hz_profile[i + 1]["admission_ratio"]
        for i in range(len(hz_profile) - 1)
    )

    return {
        "prediction": (
            "over hz1x1's five behaviour anchors the admission profile is monotone-ish with "
            "mappo1000 highest and random lowest"
        ),
        "registered_in": "BRIEF_31 Amendment A6, which re-registered E3 over the tiers P4.6 has",
        "context_not_prediction": (
            "T1's 100.0 / 90.8 / 92.9 / 65.0 for mappo1000 / fixedtime / mappo060 / random was "
            "measured on TRAINING draws with the corpus behaviour policies, and mappo060 is not a "
            "P4.6 tier; it is context and is not the prediction"
        ),
        "scored_scenario": E3_SCORED_SCENARIO,
        "scored_as": "admission_ratio(behaviour@mappo1000) > admission_ratio(behaviour@random)",
        "profile": hz_profile,
        "rank_order": [row["tier"] for row in sorted(
            hz_profile, key=lambda r: -float(r["admission_ratio"])
        )],
        "holds": holds,
        "monotone": bool(monotone),
        "grid4x4_note": (
            "reported as a measurement with its own row, never as a scoping exclusion (Amendment "
            "A4): never_entered there is expected to be 0 or near it for every arm, and a "
            "materially non-zero value is a finding in its own right"
        ),
        "grid4x4_profile": profile("grid4x4"),
    }


# ----------------------------------------------------------------------
# Draw restoration (Amendment A1: Gate -1)
# ----------------------------------------------------------------------


def code_provenance() -> dict[str, Any]:
    """The tree the CODE was imported from, which is NOT the process working directory.

    🚨 **Review BL-2 / Amendment I2.**  ``dt_gate.runtime_provenance`` records
    ``git rev-parse HEAD`` **in the process CWD**, and Amendment A2 deliberately puts the CWD in the
    MAIN tree so that re-materialising a draw does not hit ``DEFERRED`` 61's false ``BLOCKED``.  The
    main tree's HEAD is whatever branch it happens to have checked out -- for this campaign,
    ``task/p5.3b-nortg-campaign`` -- so both committed artifacts recorded a commit **from another
    task's branch that contains none of this code**.

    ⛔ **The fix is not to move the CWD back**; that reopens ``DEFERRED`` 61.  The CWD and the code
    root are two different facts and the provenance must record the second.  This resolves the tree
    from this module's own ``__file__``, so it is correct however the process was launched and
    whatever ``PYTHONPATH`` pointed at it.
    """
    import subprocess

    root = Path(__file__).resolve().parent.parent

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True, text=True, check=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "code_root": str(root),
        "module": str(Path(__file__).resolve()),
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "working_directory": str(Path.cwd()),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "note": (
            "code_root is resolved from this module's __file__, not from the working directory. "
            "runtime.git_commit beside it comes from dt_gate.runtime_provenance, which reads "
            "git rev-parse HEAD in the CWD -- and BRIEF_31 Amendment A2 puts the CWD in the main "
            "tree, so the two can and do differ (review P8.4a BL-2)"
        ),
    }


@dataclass(frozen=True)
class WorkingDirectoryCheck:
    """What the process working directory would render, against what the draws on disk record."""

    scenario_key: str
    here: Path
    recorded: Path | None
    checked_draw: int | None
    matches: bool


def assert_cwd_renders_the_recorded_scenario_dir(
    sim_config: str | Path,
    *,
    scenario_key: str,
    draw_ids: Sequence[int],
    out_root: str | Path,
) -> WorkingDirectoryCheck:
    """Amendment E4: refuse a working directory that would render a different scenario dir.

    A materialised ``cityflow.json`` embeds ``dir`` as an ABSOLUTE path resolved against the process
    working directory, because CityFlow requires one.  ``_existing_conflict`` compares rendered files
    **before** any provenance field, so re-materialising from another tree refuses with
    ``cityflow.json differs byte-for-byte`` -- a message that names a scenario file and reads as
    though the demand changed.  **It has not**: measured on 2026-08-28, all ten grid4x4 held-out
    draws regenerate a byte-identical ``flow.json`` from a second worktree while all ten
    ``cityflow.json`` differ.

    E4 rules that the embedded path is NOT normalised -- that would change rendering semantics in
    merged code for a diagnostic's convenience -- and that the refusal is made legible instead.

    Returns a check describing what was compared.  When no requested draw exists yet there is
    nothing to disagree with and any working directory is legitimate, so ``recorded`` is ``None``.
    """
    from offline.materialise_draws import (
        CITYFLOW_CONFIG_FILENAME,
        _scenario_dir,
        draw_dir,
    )

    here = Path(_scenario_dir(sim_config))
    agreed: int | None = None
    for draw_id in draw_ids:
        config = (
            Path(draw_dir(scenario_key, int(draw_id), out_root=out_root))
            / CITYFLOW_CONFIG_FILENAME
        )
        if not config.is_file():
            continue
        recorded = Path(os.path.normpath(json.loads(config.read_bytes()).get("dir", "")))
        if recorded == here:
            # Keep going: a pool assembled by several processes can agree on its first draw and
            # disagree later, and DEFERRED 55 keeps producing exactly those (review MINOR 1).
            agreed = agreed or int(draw_id)
            continue
        raise RuntimeError(
            f"this process's working directory is {Path.cwd()}, from which {sim_config} resolves "
            f"its scenario to {here}; the draws already in {Path(out_root)} were rendered against "
            f"{recorded} (read from draw {int(draw_id)}). A materialised cityflow.json embeds that "
            "directory as an absolute path, so re-materialising from here would be refused for a "
            "rendered-config difference.\n\n"
            "  THIS IS NOT A STATEMENT ABOUT THE DEMAND. The drawn demand is a pure function of "
            "(source, base_seed, draw_id) and is identical from either tree -- measured on "
            "2026-08-28 across all ten cityflow_grid4x4 held-out draws, flow.json identical 10/10. "
            "Only the config wrapper's embedded path differs.\n\n"
            f"  Re-run with the working directory set to the tree holding {recorded}."
        )
    if agreed is not None:
        return WorkingDirectoryCheck(
            scenario_key=scenario_key,
            here=here,
            recorded=here,
            checked_draw=agreed,
            matches=True,
        )
    return WorkingDirectoryCheck(
        scenario_key=scenario_key, here=here, recorded=None, checked_draw=None, matches=True
    )


@dataclass(frozen=True)
class DrawRestoration:
    """What ``offline.materialise_draws`` did, reported rather than trusted."""

    scenario_key: str
    survivors: tuple[int, ...]
    #: The draws requested BEYOND the declared survivors.  ⚠️ Review MINOR 5: this is the set the
    #: second materialise call was asked for, NOT the set it wrote -- an id here whose ``actions``
    #: entry is ``kept`` already existed and was re-verified rather than created.  Read ``actions``
    #: for what happened; this field says only what was requested.
    requested_beyond_survivors: tuple[int, ...]
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
    from offline.materialise_draws import materialise, scenario_key_for_config

    survivor_ids = [int(d) for d in survivors]
    wanted_ids = [int(d) for d in wanted]

    # Amendment E4: the working-directory check runs FIRST, before a single draw is built, so a
    # wrong directory costs a message rather than 90 draws of rendering.
    assert_cwd_renders_the_recorded_scenario_dir(
        sim_config,
        scenario_key=scenario_key_for_config(sim_config),
        draw_ids=wanted_ids,
        out_root=out_root,
    )
    overlap = sorted(set(survivor_ids) & set(wanted_ids))
    if overlap != sorted(survivor_ids):
        raise ValueError(
            f"the survivors {survivor_ids} must be a subset of the wanted draws {wanted_ids}; "
            "verifying draws the campaign will not use proves nothing about the ones it will"
        )

    actions: dict[str, str] = {}
    flow_sha256: dict[str, str] = {}
    n_vehicles: dict[str, int] = {}
    scenario_key = ""

    verified = materialise(sim_config, survivor_ids, out_root=out_root, force=False)
    for record in verified:
        scenario_key = record.scenario_key
        actions[str(record.draw_id)] = record.action
        flow_sha256[str(record.draw_id)] = record.flow_sha256
        n_vehicles[str(record.draw_id)] = int(record.n_vehicles)
    unreproduced = sorted(d for d, a in actions.items() if a != "kept")
    if unreproduced:
        raise ValueError(
            f"the surviving draws {unreproduced} came back {[actions[d] for d in unreproduced]} "
            "rather than 'kept', so they are not byte-identical to what P5.2 evaluated on. Nothing "
            "further has been written; this is BLOCKED, because every grid4x4 number measured "
            "afterwards would be measured on different demand, undetectably"
        )

    remaining = [d for d in wanted_ids if d not in set(survivor_ids)]
    restored: list[int] = []
    if remaining:
        for record in materialise(sim_config, remaining, out_root=out_root, force=False):
            scenario_key = record.scenario_key
            actions[str(record.draw_id)] = record.action
            flow_sha256[str(record.draw_id)] = record.flow_sha256
            n_vehicles[str(record.draw_id)] = int(record.n_vehicles)
            restored.append(int(record.draw_id))
        replaced = sorted(d for d in map(str, remaining) if actions[d] == "replaced")
        if replaced:
            raise ValueError(
                f"draws {replaced} were REPLACED, which force=False makes unreachable; the "
                "materialiser's contract has changed and this run may not be trusted"
            )

    return DrawRestoration(
        scenario_key=scenario_key,
        survivors=tuple(survivor_ids),
        requested_beyond_survivors=tuple(sorted(restored)),
        actions=actions,
        flow_sha256=flow_sha256,
        n_vehicles=n_vehicles,
        survivors_reproduced=True,
    )


# ----------------------------------------------------------------------
# The artifact
# ----------------------------------------------------------------------


def default_protected_roots(roots: ProbeRoots) -> tuple[Path, ...]:
    """Every directory this task may never write under.

    The corpus, and every immediate child of ``output/`` other than this task's own work directory.
    Resolution happens in ``tier_sweep.protected_roots_from``, so a relative path, a ``..``
    traversal and a symlink all resolve into the protected root and are refused.
    """
    from offline.tier_sweep import protected_roots_from

    output = Path(roots.output_root)
    work = Path(roots.work_dir).resolve()
    candidates: list[Path] = [Path(roots.corpus_root)]
    if output.is_dir():
        for child in sorted(output.iterdir()):
            if child.is_dir() and child.resolve() != work:
                candidates.append(child)
    return protected_roots_from(candidates)


def assert_no_science_verdict(payload: Any) -> None:
    """Refuse to emit a verdict on the science anywhere in the artifact.

    Runs ``method_tier_grid.assert_no_verdicts`` for the equivalence-verdict class this repo already
    forbids, then walks the payload again for :data:`SCIENCE_VERDICT_STRINGS` -- the words that would
    turn a measurement of admission into a claim about whether P5.2's headline survives, which
    ``BRIEF_31`` section 6 reserves.
    """
    from offline.method_tier_grid import assert_no_verdicts

    assert_no_verdicts(payload)

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif isinstance(node, str) and node in SCIENCE_VERDICT_STRINGS:
            raise ValueError(
                f"{path}: {node!r} is a verdict on the science, and BRIEF_31 section 6 reserves "
                "that. This artifact measures admission and scores three instrument predictions; "
                "it says nothing about whether P5.2's headline survives"
            )

    walk(payload, "artifact")


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
    draws_per_cell = {len(cell.draw_ids) for block in cells.values() for cell in block.values()}
    if len(draws_per_cell) != 1:
        raise ValueError(
            f"the cells of one artifact span {sorted(draws_per_cell)} draws; the 10-draw and "
            "100-draw grains are reported as separate artifacts precisely so that a 100-draw arm "
            "is never scored against a 10-draw anchor"
        )
    n_draws = next(iter(draws_per_cell))
    grain = f"{n_draws} held-out draws" + (" (escalated)" if n_draws > len(PROBE_DRAWS) else "")

    created_by_draw: dict[str, set[int]] = {}
    for episode in episodes:
        created_by_draw.setdefault(
            f"{episode.scenario}/{episode.draw_id}", set()
        ).add(int(episode.created))
    inconsistent = {k: sorted(v) for k, v in created_by_draw.items() if len(v) != 1}
    if inconsistent:
        raise ValueError(
            f"created is a property of a draw and must be identical in every cell that used it, "
            f"but these draws disagree: {inconsistent}"
        )

    payload: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": (
            "P8.4a: how many vehicles each arm admits, measured first rather than co-reported "
            "(BRIEF_31 section 3), with both ATT definitions on every episode"
        ),
        "what_this_does_not_say": [
            "the metric is not wrong; it computes what its docstring says, and it measures a "
            "different population than the field's",
            "no conclusion about P5.2's headline beyond what E1 measures",
            "no novelty claim: BRIEF_31 section 7 rules that unsearched",
        ],
        "registered": {
            "draws": list(PROBE_DRAWS),
            "grain": grain,
            "draws_per_cell": sorted(draws_per_cell),
            "escalation_draws": [ESCALATION_DRAWS[0], ESCALATION_DRAWS[-1]],
            "seeds": list(PROBE_SEEDS),
            "declared_in": "docs/plans/p8.4a.md sections 3 and 4, before any number existed",
            "identity": (
                "never_entered = |get_vehicles(True)| - |get_vehicles(False)|; "
                "entered = |depart_time| = completed + running; "
                "created = entered + never_entered = flow entries firing inside the horizon"
            ),
        },
        "created_per_draw": {
            key: sorted(values)[0] for key, values in sorted(created_by_draw.items())
        },
        "draw_restoration": {
            scenario: {
                "scenario_key": record.scenario_key,
                "survivors": list(record.survivors),
                "requested_beyond_survivors": list(record.requested_beyond_survivors),
                "n_written": sum(1 for a in record.actions.values() if a == "written"),
                "n_kept": sum(1 for a in record.actions.values() if a == "kept"),
                "actions": dict(record.actions),
                "flow_sha256": dict(record.flow_sha256),
                "n_vehicles": dict(record.n_vehicles),
                "survivors_reproduced": bool(record.survivors_reproduced),
            }
            for scenario, record in sorted(restoration.items())
        },
        "reference_checks": {
            arm: {
                "source": check.source,
                "n_compared": int(check.n_compared),
                "n_missing": int(check.n_missing),
                "exact": bool(check.exact),
                "mismatches": [dict(m) for m in check.mismatches],
            }
            for arm, check in sorted(references.items())
        },
        "cells": {
            scenario: {arm: cells[scenario][arm].as_record() for arm in sorted(cells[scenario])}
            for scenario in sorted(cells)
        },
        "e1": score_e1(cells),
        "e2": score_e2(cells),
        "e3": score_e3(cells),
        "timing": dict(timing),
        "episodes": [e.as_record() for e in episodes],
        "provenance": {
            **dict(provenance),
            # Review BL-2 / Amendment I2: `runtime.git_commit` below is the CWD's HEAD, and the CWD
            # is the MAIN tree by Amendment A2's ruling.  The commit that actually produced these
            # numbers is in `code_provenance`.
            "code_provenance": code_provenance(),
            "caveat": (
                "runtime.git_commit (if present) is git rev-parse HEAD in the process WORKING "
                "DIRECTORY, which BRIEF_31 Amendment A2 requires to be the main tree; that tree's "
                "HEAD belongs to whichever branch it has checked out and may contain none of this "
                "code. Read code_provenance.git_commit for the code that produced these numbers"
            ),
        },
    }
    assert_no_science_verdict(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser.

    One cell per process, mirroring every campaign in this repo: a job that dies takes one cell with
    it and the resume path is "run the cells that have no file".
    """
    parser = argparse.ArgumentParser(
        prog="python -m offline.admission_probe",
        description="P8.4a: measure how many vehicles each arm admits, and both ATT definitions",
    )
    parser.add_argument("--repo-root", default=".", help="this worktree (holds docs/data)")
    parser.add_argument("--corpus-root", default="datasets_v11")
    parser.add_argument("--draws-root", default="scenarios/draws")
    parser.add_argument("--output-root", default="output", help="the MAIN tree's output/")
    parser.add_argument("--work-dir", default="output/p8_4a")
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

    restore = sub.add_parser("restore-draws", help="Gate -1: verify the survivors, then fill gaps")
    restore.add_argument("--scenario", required=True, choices=sorted(PROBE_SCENARIOS))
    restore.add_argument("--survivors", type=int, nargs="+", required=True)
    restore.add_argument("--wanted", type=int, nargs="+", required=True)

    for name, help_text in (
        ("timing", "Gate 0: probe one cell and report seconds per episode"),
        ("probe", "probe one cell and write its work file"),
    ):
        cell_parser = sub.add_parser(name, help=help_text)
        cell_parser.add_argument("--scenario", required=True, choices=sorted(PROBE_SCENARIOS))
        cell_parser.add_argument("--tier", required=True)
        cell_parser.add_argument("--method", required=True)
        cell_parser.add_argument("--draws", type=int, nargs="+", default=list(PROBE_DRAWS))
        cell_parser.add_argument("--seeds", type=int, nargs="+", default=None)
        cell_parser.add_argument(
            "--escalated",
            action="store_true",
            help="use the full held-out pool and write under a distinct name",
        )

    report = sub.add_parser("report", help="assemble docs/data/p8_4a_admission.json")
    report.add_argument("--out", default="docs/data/p8_4a_admission.json")
    report.add_argument(
        "--escalated",
        action="store_true",
        help="report the 100-draw grain instead of the 10-draw one; the two are NEVER mixed and "
        "each is written as its own internally consistent artifact",
    )

    escalate = sub.add_parser(
        "escalation-plan", help="print the cells the registered rule sends to 100 draws"
    )
    escalate.add_argument("--artifact", default="docs/data/p8_4a_admission.json")
    return parser


def _run_escalation_plan(args: argparse.Namespace) -> int:
    """Derive the escalation list from the scored artifact, never from a hand-written list."""
    payload = json.loads((Path(args.repo_root) / args.artifact).read_bytes())
    targets = escalation_targets(payload["e1"])
    print(f"# {len(targets)} cells at {len(ESCALATION_DRAWS)} draws, derived from {args.artifact}")
    print("# rule: any arm with deficit > 0, plus the behaviour anchor of its own tier")
    for scenario, tier, method in targets:
        print(f"{scenario} {tier} {method}")
    return 0


def _roots_of(args: argparse.Namespace) -> ProbeRoots:
    return ProbeRoots(
        repo_root=Path(args.repo_root),
        corpus_root=Path(args.corpus_root),
        draws_root=Path(args.draws_root),
        output_root=Path(args.output_root),
        work_dir=Path(args.work_dir),
    )


#: The suffix that marks a cell rolled over the full held-out pool rather than the ten probe draws.
ESCALATED_SUFFIX = "_full"


def work_file_name(scenario: str, tier: str, method: str, *, escalated: bool = False) -> str:
    """``admission_<scenario>_<tier>_<method>[_full].json`` -- one file per cell per grain."""
    suffix = ESCALATED_SUFFIX if escalated else ""
    return f"admission_{scenario}_{tier}_{method}{suffix}.json"


def cell_files(work_dir: str | Path, *, escalated: bool) -> tuple[Path, ...]:
    """The cell files of ONE draw grain, sorted.

    🔒 **The two grains are never mixed, and this selector is the only thing that guarantees it.**
    Scoring a 100-draw arm against a 10-draw anchor puts two denominators under one label, which is
    the shape of the error ``PREREGISTRATION`` A5 made and T1 found.  Each grain is reported as its
    own internally consistent artifact instead.
    """
    found: list[Path] = []
    for path in sorted(Path(work_dir).glob("admission_*.json")):
        is_escalated = path.stem.endswith(ESCALATED_SUFFIX)
        if is_escalated == bool(escalated):
            found.append(path)
    return tuple(found)


def escalation_targets(scored_e1: Mapping[str, Any]) -> tuple[tuple[str, str, str], ...]:
    """Every ``(scenario, tier, method)`` the registered escalation rule requires at 100 draws.

    That is each arm with ``deficit > 0`` **plus the behaviour anchor of its own tier**: E1 is a
    comparison, and a 100-draw arm may only be compared against a 100-draw anchor.
    """
    by_key = {
        f"{row['scenario']}/{row['arm']}": row for row in scored_e1.get("arms", ())
    }
    targets: set[tuple[str, str, str]] = set()
    for key in scored_e1.get("escalated_arms", ()):
        row = by_key.get(key)
        if row is None:
            raise ValueError(
                f"{key} is named in escalated_arms but has no scored row, so its tier -- and "
                "therefore the behaviour anchor it must be compared against -- is unknown"
            )
        scenario, tier = str(row["scenario"]), str(row["tier"])
        targets.add((scenario, tier, str(row["method"]) if "method" in row else
                     str(row["arm"]).split("@")[0]))
        targets.add((scenario, tier, BEHAVIOUR_METHOD))
    return tuple(sorted(targets))


def _run_cell(args: argparse.Namespace, *, write: bool) -> int:
    from offline.dt_gate import runtime_provenance
    from offline.offline_baselines import pin_torch_threads
    from offline.tier_sweep import assert_writable, protected_roots_from, write_json_guarded

    pin_torch_threads(args.torch_threads)
    roots = _roots_of(args)
    protected = default_protected_roots(roots) + protected_roots_from(args.protect)
    draws = list(ESCALATION_DRAWS) if args.escalated else [int(d) for d in args.draws]
    seeds = None if args.seeds is None else tuple(int(s) for s in args.seeds)

    started = time.perf_counter()
    episodes, sources = probe_cell(
        scenario=args.scenario,
        tier=args.tier,
        method=args.method,
        draw_ids=draws,
        roots=roots,
        engine_seed=int(args.engine_seed),
        device=args.device,
        seeds=seeds,
    )
    elapsed = time.perf_counter() - started

    summary = summarise_cell(episodes)
    reference, source = committed_reference(args.scenario, args.tier, args.method, roots)
    check = check_against_reference(episodes, reference, source)

    print(
        f"  {args.scenario}/{summary.arm}: admission {summary.admission_ratio:.6f} "
        f"({summary.entered_total}/{summary.created_total}), att_ours "
        f"{summary.att_ours_mean:.4f}, att_engine {summary.att_engine_mean:.4f}, "
        f"n={summary.n_episodes}, {elapsed / summary.n_episodes:.3f} s/episode",
        flush=True,
    )
    print(
        f"  reference {source}: compared {check.n_compared}, missing {check.n_missing}, "
        f"mismatches {len(check.mismatches)}, exact={check.exact}",
        flush=True,
    )
    for mismatch in check.mismatches[:5]:
        print(f"    MISMATCH {mismatch}", flush=True)

    if not write:
        return 0 if check.exact else 1

    destination = Path(roots.work_dir) / work_file_name(
        args.scenario, args.tier, args.method, escalated=bool(args.escalated)
    )
    assert_writable(destination, protected)
    Path(roots.work_dir).mkdir(parents=True, exist_ok=True)
    write_json_guarded(
        {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "scenario": args.scenario,
            "tier": args.tier,
            "method": args.method,
            "arm": summary.arm,
            "engine_seed": int(args.engine_seed),
            "escalated": bool(args.escalated),
            "draw_ids": draws,
            "seconds_wall": elapsed,
            "seconds_per_episode": elapsed / summary.n_episodes,
            "arm_sources": {
                slot: {
                    "kind": src.kind,
                    "detail": src.detail,
                    "checkpoint": src.checkpoint,
                    "checkpoint_sha256": src.checkpoint_sha256,
                }
                for slot, src in sorted(sources.items())
            },
            "cell": summary.as_record(),
            "reference": {
                "source": check.source,
                "n_compared": check.n_compared,
                "n_missing": check.n_missing,
                "exact": check.exact,
                "mismatches": [dict(m) for m in check.mismatches],
            },
            "episodes": [e.as_record() for e in episodes],
            "runtime": runtime_provenance(),
        },
        destination,
        protected,
    )
    print(f"  wrote {destination}", flush=True)
    return 0 if check.exact else 1


def _run_restore(args: argparse.Namespace) -> int:
    """Gate -1, and the report of what it found is the deliverable, not a side effect."""
    from offline.tier_sweep import assert_writable, protected_roots_from, write_json_guarded

    roots = _roots_of(args)
    protected = default_protected_roots(roots) + protected_roots_from(args.protect)
    spec = PROBE_SCENARIOS[args.scenario]
    record = restore_draws(
        Path(args.repo_root) / spec.sim_config,
        survivors=args.survivors,
        wanted=args.wanted,
        out_root=args.draws_root,
    )
    print(f"survivors reproduced byte-identically: {sorted(record.survivors)}", flush=True)
    for draw_id in sorted(record.actions, key=int):
        print(
            f"  draw {draw_id}: {record.actions[draw_id]}  "
            f"n_vehicles={record.n_vehicles[draw_id]}  flow_sha256={record.flow_sha256[draw_id]}",
            flush=True,
        )

    destination = Path(roots.work_dir) / "draw_restoration.json"
    assert_writable(destination, protected)
    Path(roots.work_dir).mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if destination.is_file():
        existing = json.loads(destination.read_bytes())
    existing[args.scenario] = {
        "scenario_key": record.scenario_key,
        "survivors": list(record.survivors),
        "requested_beyond_survivors": list(record.requested_beyond_survivors),
        "n_written": sum(1 for a in record.actions.values() if a == "written"),
        "n_kept": sum(1 for a in record.actions.values() if a == "kept"),
        "actions": dict(record.actions),
        "flow_sha256": dict(record.flow_sha256),
        "n_vehicles": dict(record.n_vehicles),
        "survivors_reproduced": bool(record.survivors_reproduced),
        "sim_config": str(Path(args.repo_root) / spec.sim_config),
    }
    write_json_guarded(existing, destination, protected)
    print(f"  wrote {destination}", flush=True)
    return 0


def _run_report(args: argparse.Namespace) -> int:
    from offline.dt_gate import runtime_provenance
    from offline.tier_sweep import assert_writable, protected_roots_from, write_json_guarded

    roots = _roots_of(args)
    protected = default_protected_roots(roots) + protected_roots_from(args.protect)
    work = Path(roots.work_dir)
    escalated = bool(getattr(args, "escalated", False))
    files = cell_files(work, escalated=escalated)
    if not files:
        grain = "100-draw" if escalated else "10-draw"
        raise FileNotFoundError(
            f"no {grain} cell files under {work}; run the probe subcommand first"
        )

    cells: dict[str, dict[str, CellSummary]] = {}
    references: dict[str, ReferenceCheck] = {}
    episodes: list[AdmissionEpisode] = []
    timing: dict[str, Any] = {"per_cell": {}}
    restoration: dict[str, DrawRestoration] = {}

    for path in files:
        payload = json.loads(path.read_bytes())
        if bool(payload.get("escalated")) != escalated:
            raise ValueError(
                f"{path} records escalated={payload.get('escalated')} but was selected for the "
                f"escalated={escalated} grain; the filename and the payload disagree and mixing "
                "the two grains would put two denominators under one label"
            )
        rows = [
            AdmissionEpisode(
                scenario=row["scenario"],
                tier=row["tier"],
                method=row["method"],
                arm=row["arm"],
                seed=row["seed"],
                draw_id=int(row["draw_id"]),
                created=int(row["created"]),
                entered=int(row["entered"]),
                never_entered=int(row["never_entered"]),
                entered_fraction=float(row["entered_fraction"]),
                completed_at_horizon=int(row["completed_at_horizon"]),
                running_at_horizon=int(row["running_at_horizon"]),
                waiting_at_horizon=int(row["waiting_at_horizon"]),
                att_ours=float(row["att_ours"]),
                att_engine=float(row["att_engine"]),
                horizon_vehicle_count=float(row["horizon_vehicle_count"]),
                episode_reward=float(row["episode_reward"]),
                seconds=float(row["seconds"]),
                seconds_rollout=float(row.get("seconds_rollout", 0.0)),
            )
            for row in payload["episodes"]
        ]
        episodes.extend(rows)
        scenario = payload["scenario"]
        arm = payload["arm"]
        cells.setdefault(scenario, {})[arm] = summarise_cell(rows)
        reference = payload["reference"]
        references[f"{scenario}/{arm}"] = ReferenceCheck(
            source=reference["source"],
            n_compared=int(reference["n_compared"]),
            n_missing=int(reference["n_missing"]),
            mismatches=tuple(reference["mismatches"]),
        )
        timing["per_cell"][f"{scenario}/{arm}"] = {
            "seconds_wall": float(payload["seconds_wall"]),
            "seconds_per_episode": float(payload["seconds_per_episode"]),
            "n_episodes": len(rows),
        }

    restoration_file = work / "draw_restoration.json"
    if restoration_file.is_file():
        for scenario, record in json.loads(restoration_file.read_bytes()).items():
            restoration[scenario] = DrawRestoration(
                scenario_key=record["scenario_key"],
                survivors=tuple(record["survivors"]),
                requested_beyond_survivors=tuple(record["requested_beyond_survivors"]),
                actions=record["actions"],
                flow_sha256=record["flow_sha256"],
                n_vehicles=record["n_vehicles"],
                survivors_reproduced=bool(record["survivors_reproduced"]),
            )

    timing["seconds_total"] = float(sum(e.seconds for e in episodes))
    timing["n_episodes"] = len(episodes)
    timing["note"] = (
        "seconds is the whole per-episode cost including env construction and the per-draw "
        "policy load; seconds_rollout isolates the simulation. Both are needed by P8.4b"
    )

    payload = admission_artifact(
        cells=cells,
        episodes=episodes,
        references=references,
        restoration=restoration,
        timing=timing,
        provenance={
            "runtime": runtime_provenance(),
            "cell_files": [p.name for p in files],
        },
    )
    destination = Path(args.repo_root) / args.out
    assert_writable(destination, protected)
    write_json_guarded(payload, destination, protected)

    exact = sum(1 for c in references.values() if c.exact)
    print(f"cells: {sum(len(v) for v in cells.values())}, episodes: {len(episodes)}", flush=True)
    print(f"reference checks exact: {exact}/{len(references)}", flush=True)
    print(f"E1 holds/close/falsified: {payload['e1']['n_holds']}/"
          f"{payload['e1']['n_close']}/{payload['e1']['n_falsified']}", flush=True)
    print(f"E2 arms below the floor: {payload['e2']['n_below']}/{payload['e2']['n_arms']}", flush=True)
    print(f"E3 holds: {payload['e3']['holds']}, monotone: {payload['e3']['monotone']}", flush=True)
    print(f"wrote {destination}", flush=True)
    return 0 if exact == len(references) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "restore-draws":
        return _run_restore(args)
    if args.command == "timing":
        return _run_cell(args, write=False)
    if args.command == "probe":
        return _run_cell(args, write=True)
    if args.command == "escalation-plan":
        return _run_escalation_plan(args)
    return _run_report(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
