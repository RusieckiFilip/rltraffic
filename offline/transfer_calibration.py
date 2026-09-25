"""P7.2b: the target-domain SUMO probe and the cross-domain return-prompt calibration.

Artifact format version: ``p7.2b-calibration/1.0`` -- one file, ``docs/data/p7_2b_calibration.json``,
written by :func:`report` from the per-draw chunks under ``output/p7_2b/``.

WHAT THIS MODULE EXECUTES, AND WHAT IT MAY NOT DECIDE
------------------------------------------------------
``PREREGISTRATION`` **A17** fixed the protocol on 2026-09-13, before any target-domain probe return
existed: the rule (Rule B), the registered statistic (``S = mean``), the probe (MaxPressure, SUMO,
draws 201-300, one episode per draw at ``reset(seed=1000)``), the two subjects, the H3 arm's prompt
(Rule B, ``mean``, k = 100) and the exploratory status of the calibrated-vs-naive contrast.  **Nothing
here chooses any of that.**  Every rule function is imported from :mod:`offline.rtg_calibration` and
every source-domain statistic is READ from ``docs/data/p4_3_probe.json``; this module measures the
target-domain half and does the arithmetic.

**It evaluates nothing.**  The zero-shot point, the calibrated-vs-naive contrast and every rho are
P7.3's, on the held-out pool.  The two smoke episodes exist to prove the mechanics of the path -- a
CityFlow-trained DT driving a SUMO env through :mod:`offline.aligned_env` with the RTG advancing --
and their outcome-shaped quantities are FENCED (see below).

THE FENCE (Amendment A3)
------------------------
``horizon_rollout`` returns ``att_horizon`` and ``episode_reward``; the RTG series ends at
``rtg_last = target - sum(rewards)``, which **is** the episode return in disguise.  All four --
``att_horizon``, ``episode_reward``, ``rtg_last`` and the per-decision RTG series -- are written into
the smoke chunk under :data:`FENCED_KEY` and :func:`report` **refuses to copy that key into
``docs/data/``**.  What the smoke reports is mechanics: the decision count, that the first RTG equals
the target, that the RTG advanced on every decision whose reward was non-zero, the in-support count
against the registered range, that every action was in range, the engine-read type set and regime,
and the wall time.  A smoke that printed an ATT would be a zero-shot number seen before P7.3's brief
is written.

TWO RANGES, AND WHY BOTH ARE RECORDED (Amendment A1)
-----------------------------------------------------
:func:`offline.rtg_calibration.training_rtg_range` returns the checkpoint's ``stats.rtg`` range, which
is the range over the **split**: 216,000 rows for ``mix50`` (600 streams x 360), 72,000 for
``mappo1000``.  That is the number **A17(c) registered**, so it is the range the in-support diagnostic
is computed against, and it is recorded as ``support_range_over_the_split`` with its ``n_rows``.
Beside it the artifact records ``training_set_return_min = -rtg_scale``, the bound of the TRAINING SET
the target itself came from (``-40223`` for ``mix50``, ``-9991`` for ``mappo1000``, where the two
coincide).  **The two differ by 71 on a 40,000 scale for ``mix50``; the diagnostic never selects and
no claim rests on which bound is used.**

DISJOINTNESS IS ASSERTED AGAINST THE UNION (Amendment A7.1)
------------------------------------------------------------
Both checkpoints' ``provenance.training_draw_ids`` are ``1..200`` -- the ``mix50`` checkpoint's field
records the candidate pool, **not** the 152 draws its tier trained on, which live in
``docs/data/p4_7_declaration.json:tiers.mix50.training_draws``.  The assertion therefore runs against
the UNION of both checkpoint fields, the declaration's 152 and the held-out pool 1000-1099; the
superset is the safe direction, and the artifact records all three sources with their sizes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "CANARY_MAX_SECONDS",
    "DECLARED_GRADIENT_STEPS",
    "DEFAULT_ENGINE_SEED",
    "FENCED_KEY",
    "P4_3_PROBE_ARTIFACT",
    "P4_3_PROBE_SHA256",
    "SCENARIO_KEY",
    "SMOKE_DRAW_ID",
    "SUBJECTS",
    "ProbeRecord",
    "SubjectFacts",
    "CANARY_FACT_NAMES",
    "CANARY_RECORD_NAME",
    "CANARY_REFERENCE_ATT_HORIZON",
    "CANARY_REFERENCE_DECISIONS",
    "CANARY_REFERENCE_LOCAL_RETURN",
    "build_parser",
    "assert_logged_corpus_matches_probe",
    "check_canary",
    "chunk_is_reusable",
    "disjointness_record",
    "format_canary_line",
    "in_support_position",
    "main",
    "parse_canary_line",
    "probe_chunk_path",
    "record_canary",
    "probe_returns_from_chunks",
    "report",
    "rtg_advanced_every_decision",
    "run_smoke",
    "run_sumo_probe",
    "statistics_table",
    "subject_facts",
    "targets_table",
]

ARTIFACT_FORMAT_VERSION = "p7.2b-calibration/1.0"

#: A17(b): the scenario key, the seed, the fenced smoke draw and the declared budget.
SCENARIO_KEY = "cityflow1x1"
DEFAULT_ENGINE_SEED = 1000
SMOKE_DRAW_ID = 5
DECLARED_GRADIENT_STEPS = 40000

#: Amendment A3: the smoke chunk's fenced block.  ``report`` refuses to emit this key.
FENCED_KEY = "fenced_do_not_report"

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: The source-domain statistics are READ from P4.3's committed artifact and never recomputed.
P4_3_PROBE_ARTIFACT = _REPO_ROOT / "docs" / "data" / "p4_3_probe.json"

#: Amendment A5: the digest is pinned here AS A DECLARATION -- *these statistics came from THAT
#: artifact* -- and recorded again in every chunk AS EVIDENCE.  ``report`` refuses if the file, this
#: constant and the chunks do not all agree.  P4.3's artifact regenerates byte-identically
#: (``docs/returns/P4.3.md`` §16.0), so this constant moves only in a commit that also moves the
#: artifact, which is a reviewed act.
P4_3_PROBE_SHA256 = "dc12f8b7e791bdf1a2097fc372a0e8918f67b40b8bf531f7bbde802f07b12a74"

#: A17(c)'s two registered subjects, each with its five seeds.  The paths are under the MAIN tree's
#: ``output/``; the loader takes a root so a test can point elsewhere.
SUBJECTS: dict[str, dict[str, Any]] = {
    "mappo1000": {"subdir": "p4_dt", "stem": "dt_seed"},
    "mix50": {"subdir": "p4_7/checkpoints", "stem": "mix50_dt_seed"},
}
TRAINING_SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)


@dataclass(frozen=True)
class ProbeRecord:
    """One SUMO probe episode, with the engine reads A17(b) requires."""

    draw_id: int
    local_return: float
    local_return_from_lanes: float
    att_horizon: float
    horizon_vehicle_count: float
    decisions: int
    engine_seed_requested: int
    engine_seed_drawn: int
    n_teleports: int
    vehicle_types_seen: tuple[str, ...]
    time_to_teleport_option: str
    seconds: float


@dataclass(frozen=True)
class SubjectFacts:
    """What a subject's checkpoints declare, read from the payloads and agreed across five seeds."""

    subject: str
    best_source_return: float
    rtg_scale: float
    support_range_over_the_split: tuple[float, float]
    n_rows: int
    training_set_return_min: float
    checkpoints: tuple[str, ...]
    state_dim: int
    context_length: int


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


#: ``p4_3_probe.json:env_settings.max_steps`` -- one episode is exactly this many decisions.
EXPECTED_DECISIONS = 360

#: A17(b): the only vehicle type any probe episode may observe.
PARITY_VTYPE_ID = "cf_parity"

#: A15(c): the regime SUMO must report.
EXPECTED_TIME_TO_TELEPORT = "-1"

#: ``PROJECT_PLAN`` §7's machine-health gate, recipe in ``BRIEF_36`` §3.3.  The driver refuses to
#: start above this, and :func:`report` lifts the measured value into the artifact so a reader can
#: see the rate basis of the run rather than take it on trust (Amendment D2).  The driver carries
#: the same literal and a test asserts the two agree.
CANARY_MAX_SECONDS = 2.0

#: THE CANARY'S OTHER HALF (Amendment E1.2).  ``PROJECT_PLAN`` §7 names both in one sentence -- a
#: canary measures *the machine and the engine* -- but ``BRIEF_36`` §3.3 wrote only the timing, the
#: implementation matched the brief, and the pre-flight cleared it against the brief.  So runs 1 and
#: 2 printed the engine's answers to the pane and compared none of them: a CityFlow that had
#: silently changed its arithmetic would have passed a 0.89 s canary.
#:
#: The three references are the same episode read three ways, all on record and all bit-identical:
#: draw 0 is the NOMINAL control (its ``flow.json`` is byte-identical to the shipped one), so
#:   * ``-32648.0`` is P4.3's probe return on it (2026-08-13), and equals the P0.2 baseline's
#:     MaxPressure ``episode_reward`` (2026-08, three seeds);
#:   * ``247.75089149261333`` is P7.1's ``att_env_mean`` for ``cityflow__maxpressure``
#:     (2026-09-12, n = 5, five identical episodes);
#:   * 360 decisions is ``max_steps`` from the same committed settings.
#: Compared under ``==``: every measurement on record agrees to the last digit, so a near-miss is a
#: finding about the engine, not a tolerance to widen.
CANARY_REFERENCE_LOCAL_RETURN = -32648.0
CANARY_REFERENCE_ATT_HORIZON = 247.75089149261333
#: The canary's decision count is the same 360 the probe episodes run, from the same settings.
CANARY_REFERENCE_DECISIONS = EXPECTED_DECISIONS

#: The four names a canary line must carry.  A line that is missing one is refused rather than
#: recorded with a hole, because :func:`check_canary` reads a missing name as ``None`` and would
#: then refuse for the wrong reason.
CANARY_FACT_NAMES: tuple[str, ...] = (
    "att_horizon",
    "decisions",
    "local_return",
    "two_routes_agree",
)

#: Amendment E1.4: THIS run's canary, in the work directory, written by the driver after the token
#: and swept up by the manifest's ``find … -name '*.json'``.  Before it existed, :func:`report` built
#: the artifact's canary block from the probe CHUNKS, and the chunks are reused across re-rolls -- so
#: run 3's artifact reported run 1's 0.89 s beside ``checked_against``, implying a check that the
#: code performing it did not yet exist to perform.  From this commit on, a work directory without
#: this file is not a run.
CANARY_RECORD_NAME = "canary.json"

#: ``PREREGISTRATION`` §5: the pool a probe may never touch.
HELD_OUT_DRAWS: tuple[int, ...] = tuple(range(1000, 1100))


def probe_chunk_path(draw_id: int, *, work_dir: str | Path) -> Path:
    """Pure path arithmetic: where one draw's probe chunk lives."""
    return Path(work_dir) / f"probe_draw_{int(draw_id):04d}.json"


def chunk_is_reusable(payload: Mapping[str, Any], *, draw_id: int) -> bool:
    """Whether a chunk may be skipped on resume -- judged from its CONTENT, never its verdict.

    The stored ``two_routes_agree`` is exactly what a hand-made or half-written chunk would lie
    about, so the two returns are compared again here.  Any exception means "not reusable": a chunk
    this function cannot read is one the campaign must re-run.

    ⚠️ **A syntactically valid non-object is judged here, not left to explode later** (Amendment B2,
    pre-flight minor 1).  ``json.loads`` accepts ``[]``, ``5`` and ``"text"`` as happily as an
    object, and ``payload.get`` on any of them raises ``AttributeError`` -- which the ``except``
    below does not catch, so the probe died with a traceback and left the file in place.  A chunk
    that is not a mapping is simply not reusable, and the existing move-aside path handles it.
    """
    if not isinstance(payload, Mapping):
        return False
    try:
        if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
            return False
        if int(payload["draw_id"]) != int(draw_id):
            return False
        if int(payload["decisions"]) != EXPECTED_DECISIONS:
            return False
        if float(payload["local_return"]) != float(payload["local_return_from_lanes"]):
            return False
        if int(payload["n_teleports"]) != 0:
            return False
        if list(payload["vehicle_types_seen"]) != [PARITY_VTYPE_ID]:
            return False
        if str(payload["time_to_teleport_option"]) != EXPECTED_TIME_TO_TELEPORT:
            return False
        if str(payload["p4_3_probe_sha256"]) != P4_3_PROBE_SHA256:
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _record_from_chunk(payload: Mapping[str, Any]) -> ProbeRecord:
    return ProbeRecord(
        draw_id=int(payload["draw_id"]),
        local_return=float(payload["local_return"]),
        local_return_from_lanes=float(payload["local_return_from_lanes"]),
        att_horizon=float(payload["att_horizon"]),
        horizon_vehicle_count=float(payload["horizon_vehicle_count"]),
        decisions=int(payload["decisions"]),
        engine_seed_requested=int(payload["engine_seed_requested"]),
        engine_seed_drawn=int(payload["engine_seed_drawn"]),
        n_teleports=int(payload["n_teleports"]),
        vehicle_types_seen=tuple(payload["vehicle_types_seen"]),
        time_to_teleport_option=str(payload["time_to_teleport_option"]),
        seconds=float(payload["seconds"]),
    )


def _move_aside(path: Path) -> Path:
    """Move an unusable chunk into ``failed/`` rather than overwriting it.

    The same reasoning as P7.1's ``failed_chunk_destination``: a chunk that failed its own
    re-validation is evidence about a run, and ``report``'s glob must not see it.
    """
    destination = path.parent / "failed" / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while destination.exists():
        suffix += 1
        destination = destination.with_name(f"{path.stem}.{suffix}{path.suffix}")
    path.replace(destination)
    return destination


def run_sumo_probe(
    draw_ids: Sequence[int],
    *,
    out_root: str | Path,
    work_dir: str | Path,
    engine_seed: int = DEFAULT_ENGINE_SEED,
    canary_seconds: float | None = None,
) -> list[ProbeRecord]:
    """A17(b)'s probe: one MaxPressure episode per draw on SUMO, with the engine reads.

    The loop mirrors :func:`offline.rtg_calibration.run_probe` (fresh env per draw,
    ``reset(seed=engine_seed)``, ``policy.act(info)`` -> ``env.step``, post-step infos accumulated,
    break on terminate/truncate, ``samples[-1]`` for the horizon reading) and differs in exactly
    three declared ways, each because A17(b) requires it:

    1. the env is **SUMO**, built through ``collect_style_args`` -> ``_build_env_spec`` ->
       ``make_env`` -- the route the repo already exercises on SUMO;
    2. the two return routes are **compared** here.  ``run_probe`` records both and leaves the
       comparison to its caller, so "mirrors ``run_probe``" must not be read as inheriting a check
       that is not there;
    3. four quantities are read **from the running engine**: the teleport count (per step), the
       effective vehicle type, the ``time-to-teleport`` option, and the seed SUMO actually received.

    The env is the plain one, not P7.1's observer: the observer counts teleports but has no
    vehicle-type read at all, and its instrumentation costs about 60 % more per episode.
    """
    from offline.materialise_draws import parity_sumocfg_path

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    probe_sha = _sha256_file(P4_3_PROBE_ARTIFACT)
    if probe_sha != P4_3_PROBE_SHA256:
        raise ValueError(
            f"{P4_3_PROBE_ARTIFACT} has sha256 {probe_sha} but this module pins "
            f"{P4_3_PROBE_SHA256}; the source-domain statistics would come from a different "
            "artifact than the one declared"
        )

    records: list[ProbeRecord] = []
    for raw_draw_id in draw_ids:
        draw_id = int(raw_draw_id)
        chunk_path = probe_chunk_path(draw_id, work_dir=work)
        if chunk_path.is_file():
            try:
                existing = json.loads(chunk_path.read_bytes())
            except json.JSONDecodeError:
                existing = {}
            if chunk_is_reusable(existing, draw_id=draw_id):
                records.append(_record_from_chunk(existing))
                continue
            _move_aside(chunk_path)

        config_path = parity_sumocfg_path(SCENARIO_KEY, draw_id, out_root=out_root)
        if not config_path.is_file():
            raise FileNotFoundError(
                f"probe draw {draw_id} has no parity configuration at {config_path}; P7.2a "
                "materialises the band into the MAIN tree's scenarios/draws"
            )

        record = _roll_one_episode(draw_id, config_path, engine_seed=int(engine_seed))
        _write_json(
            chunk_path,
            {
                "format_version": ARTIFACT_FORMAT_VERSION,
                "draw_id": record.draw_id,
                "scenario_key": SCENARIO_KEY,
                "local_return": record.local_return,
                "local_return_from_lanes": record.local_return_from_lanes,
                "two_routes_agree": True,
                "att_horizon": record.att_horizon,
                "horizon_vehicle_count": record.horizon_vehicle_count,
                "decisions": record.decisions,
                "engine_seed_requested": record.engine_seed_requested,
                "engine_seed_drawn": record.engine_seed_drawn,
                "n_teleports": record.n_teleports,
                "vehicle_types_seen": list(record.vehicle_types_seen),
                "time_to_teleport_option": record.time_to_teleport_option,
                "config_path": str(config_path),
                "config_sha256": _sha256_file(config_path),
                "p4_3_probe_sha256": probe_sha,
                "seconds": record.seconds,
                "canary_seconds": canary_seconds,
                **_git_provenance(),
            },
        )
        records.append(record)
    return records


def _roll_sumo_maxpressure_episode(
    env: Any, policy: Any, *, engine_seed: int
) -> tuple[list[Mapping[str, Any]], list[float], float, dict[str, Any]]:
    """One SUMO MaxPressure episode with A17(b)'s four engine reads: the ONE loop, shared.

    Extracted from :func:`_roll_one_episode` verbatim when the multi-intersection probe (P7.3d C5)
    needed the same episode, so that the two entry points cannot drift -- the same reason
    ``rtg_calibration._roll_maxpressure_episode`` exists on the CityFlow side.  The duplication was
    caught by ``test_the_module_contains_no_bare_dt_act_call``, which pins this module to exactly
    ONE ``policy.act(...)`` call; that test was not touched.

    The reads: the teleport counter is accumulated per step AND after the loop, and the effective
    vehicle type is sampled on the first step on which anybody is present as well as at the
    horizon, so an early wrong type cannot hide behind a horizon-only read.
    """
    info = env.reset(seed=int(engine_seed))
    option = str(env._sumo.simulation.getOption("time-to-teleport"))
    types_seen: set[str] = set()
    teleports = 0
    post_step: list[Mapping[str, Any]] = []
    samples: list[float] = []
    last_vehicle_count = 0.0
    for _ in range(int(env.max_steps)):
        teleports += len(env._sumo.simulation.getStartingTeleportIDList())
        if not types_seen:
            types_seen.update(
                env._sumo.vehicle.getTypeID(v) for v in env._sumo.vehicle.getIDList()
            )
        # Named `policy`, not `agent`: MaxPressureAgent.act takes no `explore` keyword, so a bare
        # call here is correct -- and the name keeps it distinguishable from a DT's call, which
        # must never be bare (Amendment E1).
        action = policy.act(info)
        _reward, terminated, truncated, info = env.step(action)
        post_step.append(info)
        samples.append(float(info.get("average_travel_time", 0.0)))
        last_vehicle_count = float(info.get("vehicle_count", 0.0))
        if terminated or truncated:
            break
    teleports += len(env._sumo.simulation.getStartingTeleportIDList())
    types_seen.update(env._sumo.vehicle.getTypeID(v) for v in env._sumo.vehicle.getIDList())
    reads = {
        "n_teleports": teleports,
        "vehicle_types_seen": sorted(types_seen),
        "time_to_teleport_option": option,
        "engine_seed_drawn": int(env._engine_seed),
    }
    return post_step, samples, last_vehicle_count, reads


def _roll_one_episode(
    draw_id: int, config_path: Path, *, engine_seed: int
) -> ProbeRecord:
    """One SUMO MaxPressure episode with A17(b)'s engine reads, or a refusal.

    A named seam, not decoration: the campaign's resume path (move a bad chunk aside, roll a fresh
    one) is testable without a simulator only if the roll can be substituted, and a resume rule
    that is only ever exercised by a real episode is a rule nobody tests on the failure branch.
    """
    from algorithms.max_pressure import MaxPressureAgent
    from experiments.envs import make_env
    from offline.collect import _build_env_spec
    from offline.rtg_calibration import episode_return_two_routes
    from offline.sumo_att_reference import collect_style_args

    import time

    args = collect_style_args(
        "sumo", "maxpressure", config_path, sentinel_out_dir="/nonexistent"
    )
    started = time.perf_counter()
    env = make_env(_build_env_spec(args))
    try:
        intersections = list(env.intersections)
        if len(intersections) != 1:
            raise ValueError(
                f"the probe records one intersection's return and this scenario has "
                f"{len(intersections)}; A17(b) registers a single-intersection probe"
            )
        ix_id = str(intersections[0].id)
        lanes = list(intersections[0].incoming_lanes)
        policy = MaxPressureAgent(env)
        post_step, samples, last_vehicle_count, reads = _roll_sumo_maxpressure_episode(
            env, policy, engine_seed=int(engine_seed)
        )
        option = str(reads["time_to_teleport_option"])
        types_seen = set(reads["vehicle_types_seen"])
        teleports = int(reads["n_teleports"])
        engine_seed_drawn = int(reads["engine_seed_drawn"])
    finally:
        env.close()
    seconds = time.perf_counter() - started

    from_rewards, from_lanes = episode_return_two_routes(
        post_step, ix_id=ix_id, incoming_lanes=lanes
    )

    # A17(b)'s refusals. A single failure refuses the draw AND the run.
    if from_rewards != from_lanes:
        raise ValueError(
            f"draw {draw_id}: the two return routes disagree ({from_rewards!r} against "
            f"{from_lanes!r}); A17(b) requires equality under =="
        )
    if teleports != 0:
        raise ValueError(
            f"draw {draw_id}: {teleports} teleport(s) on a configuration that requested "
            "time-to-teleport -1 (A15(c))"
        )
    if sorted(types_seen) != [PARITY_VTYPE_ID]:
        raise ValueError(
            f"draw {draw_id}: the engine ran vehicle type(s) {sorted(types_seen)}, not "
            f"[{PARITY_VTYPE_ID!r}]; the parity contract is not what this episode measured"
        )
    if option != EXPECTED_TIME_TO_TELEPORT:
        raise ValueError(
            f"draw {draw_id}: SUMO reports time-to-teleport {option!r}, not "
            f"{EXPECTED_TIME_TO_TELEPORT!r}"
        )
    if len(post_step) != EXPECTED_DECISIONS:
        raise ValueError(
            f"draw {draw_id}: {len(post_step)} decisions, not {EXPECTED_DECISIONS}"
        )

    record = ProbeRecord(
        draw_id=draw_id,
        local_return=from_rewards,
        local_return_from_lanes=from_lanes,
        att_horizon=samples[-1] if samples else 0.0,
        horizon_vehicle_count=last_vehicle_count,
        decisions=len(post_step),
        engine_seed_requested=int(engine_seed),
        engine_seed_drawn=engine_seed_drawn,
        n_teleports=teleports,
        vehicle_types_seen=tuple(sorted(types_seen)),
        time_to_teleport_option=option,
        seconds=seconds,
    )
    return record


# ======================================================================================
# The calibration PER INTERSECTION (P7.3d: BRIEF_39 C3a, Amendments A1, A4, B.1; A17(e), A20, A21)
# ======================================================================================

#: The draws-tree key and the registered subject of the grid4x4 point (A20(a)).
GRID4X4_SCENARIO_KEY = "cityflow_grid4x4"
GRID4X4_SUBJECT = "mappo1000_dt_nomix_h4"
GRID4X4_CHECKPOINT_SUBDIR = "p5_2/checkpoints"
GRID4X4_CHECKPOINT_STEM = "grid4x4_mappo1000_dt_nomix_h4_seed"

#: A20(a)'s five checkpoints BY DIGEST.  A20 records that they were "pinned in no committed file
#: until P7.3d's declaration artifact records them": this is that pin, re-measured 2026-09-19, and
#: ``docs/data/p7_3d_calibration.json`` carries it as evidence.  It moves only with the artifact.
GRID4X4_CHECKPOINT_SHA256: dict[int, str] = {
    101: "329fb6b87fc8cf5c27530b2fe4d45212ea9c0775db3fb1e636a6bbc0a5004279",
    202: "f541358530399cc8282d4b8af63da7a83195bc421706714df0d460ff0d132fcc",
    303: "48076dab687da6b88ee0059b12e5edd49cea7d363deec84d30597bdc26e29906",
    404: "4b61bc064af2984e0d2e9666f03485ca901b87337a3a30c29c77302813eb4729",
    505: "09bd310ddc6ed337835d8a8fe9d29a16968cf2d339fabaae91927b4808e34cdc",
}

#: The checkpoint format the subject is stored in (Amendment A1: the identity-graph control of the
#: spatial architecture, read by ``SpatialDTAgent``; never a ``dt-checkpoint/1.0``).
SPATIAL_CHECKPOINT_FORMAT = "spatial-dt-checkpoint/1.0"

#: A20(b): the registered budget, and the two that are RECORDED and never evaluated.
REGISTERED_K = 100
REGISTERED_STATISTIC = "mean"
ROLE_REGISTERED = "registered_prompt"
ROLE_RECORDED_ONLY = "recorded_not_evaluated"


@dataclass(frozen=True)
class SubjectFactsPerIntersection:
    """What the subject's five checkpoints declare PER INTERSECTION, agreed across the seeds."""

    subject: str
    scenario_id: str
    intersection_ids: tuple[str, ...]
    best_source_return: dict[str, float]
    rtg_scale: dict[str, float]
    support_range: dict[str, tuple[float, float]]
    n_rows: dict[str, int]
    training_draw_ids: tuple[int, ...]
    checkpoints: tuple[str, ...]
    checkpoint_sha256: dict[int, str]
    state_dim: int
    context_length: int
    n_head: int
    gradient_steps: int


def subject_facts_per_intersection(
    *,
    output_root: str | Path,
    expected_sha256: Mapping[int, str] | None = None,
) -> SubjectFactsPerIntersection:
    """Read the registered grid4x4 subject's per-intersection constants from its five checkpoints.

    **Which field each constant is read from, and why (Amendment A4).**  ``R_best_source,i`` is
    ``payload["target_rtg"][i]`` -- P5.2's declared prompt, *the maximum episode return in THIS
    intersection's training streams* -- which is the repo's established reading on hangzhou
    (:func:`subject_facts`, ``best_source_return = payload["target_rtg"]``).  ``rtg_scale_i`` is
    ``payload["rtg_scale"][i]`` and is never recalibrated (A17(a)).  ``payload["stats"]["rtg"]`` is
    a PER-WINDOW summary -- its ``max`` is 0.0 on every intersection, the return-to-go at an
    episode's last step -- and bounds the SUPPORT only; A17(e)'s wording pointed at it for
    ``R_best`` and the payload says otherwise.

    Every refusal names what it found: a checkpoint absent or at a digest other than
    *expected_sha256* (default: A20(a)'s five, :data:`GRID4X4_CHECKPOINT_SHA256`); a payload that is
    not the identity-graph control in the spatial format (Amendment A1) -- wrong format version,
    ``spatial_mixing`` on, a mask that is not the identity; a gradient-step count other than the
    declared one; an id set that differs between the prompt, the scale, the statistics and the
    recorded order; and any seed that disagrees with seed 101 on any intersection's constant --
    the seed varies the training RNG, never the prompt.
    """
    import numpy as np

    pins = dict(GRID4X4_CHECKPOINT_SHA256 if expected_sha256 is None else expected_sha256)
    root = Path(output_root)
    paths = {
        seed: root / GRID4X4_CHECKPOINT_SUBDIR / f"{GRID4X4_CHECKPOINT_STEM}{seed}.pt"
        for seed in TRAINING_SEEDS
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{GRID4X4_SUBJECT}: checkpoints absent: {missing}")

    digests: dict[int, str] = {}
    for seed, path in paths.items():
        digests[seed] = _sha256_file(path)
        if digests[seed] != pins.get(seed):
            raise ValueError(
                f"{GRID4X4_SUBJECT} seed {seed}: {path.name} hashes to {digests[seed]}, not the "
                f"registered {pins.get(seed)}; A20(a) registers the subject BY DIGEST, and a "
                "different file under the same name is a different subject"
            )

    per_seed: dict[int, dict[str, Any]] = {}
    for seed, path in paths.items():
        payload = _load_payload(path)
        label = f"{GRID4X4_SUBJECT} seed {seed}"
        version = str(payload.get("format_version"))
        if version != SPATIAL_CHECKPOINT_FORMAT:
            raise ValueError(
                f"{label}: checkpoint format {version!r} is not {SPATIAL_CHECKPOINT_FORMAT!r}; the "
                "registered subject is the identity-graph control stored in the spatial format"
            )
        config = payload["config"]
        if bool(config.get("spatial_mixing")):
            raise ValueError(
                f"{label}: spatial_mixing is on; A20(a) registers the NON-mixing control "
                "(dt_nomix_h4), and the mixing arm is out of this task's scope"
            )
        ids = [str(ix) for ix in payload["intersection_ids"]]
        mask = np.asarray(payload["spatial_mask"], dtype=np.bool_)
        if mask.shape != (len(ids), len(ids)) or not bool((mask == np.eye(len(ids), dtype=np.bool_)).all()):
            raise ValueError(
                f"{label}: the recorded spatial mask is not the {len(ids)} x {len(ids)} identity, "
                "so intersections could attend to each other; that is not the registered subject"
            )
        steps = int(payload["provenance"]["gradient_steps"])
        if steps != DECLARED_GRADIENT_STEPS:
            raise ValueError(
                f"{label}: the checkpoint records {steps} gradient steps, not the declared "
                f"{DECLARED_GRADIENT_STEPS}"
            )
        scenario_id = str(payload["scenario_id"])
        summaries = payload["stats"]["rtg"].get(scenario_id, {})
        for name, keys in (
            ("target_rtg", payload["target_rtg"]),
            ("rtg_scale", payload["rtg_scale"]),
            ("stats.rtg", summaries),
        ):
            absent = [ix for ix in ids if ix not in keys]
            extra = [str(ix) for ix in keys if str(ix) not in ids]
            if absent or extra:
                raise ValueError(
                    f"{label}: {name} does not cover exactly the recorded intersections "
                    f"(missing {absent[:4]}, unexpected {extra[:4]})"
                )
        per_seed[seed] = {
            "scenario_id": scenario_id,
            "ids": ids,
            "target_rtg": {ix: float(payload["target_rtg"][ix]) for ix in ids},
            "rtg_scale": {ix: float(payload["rtg_scale"][ix]) for ix in ids},
            "support": {ix: (float(summaries[ix]["min"]), float(summaries[ix]["max"])) for ix in ids},
            "n_rows": {ix: int(summaries[ix]["count"]) for ix in ids},
            "draw_ids": tuple(int(d) for d in payload["stats"]["draw_ids"]),
            "state_dim": int(config["state_dim"]),
            "context_length": int(config["context_length"]),
            "n_head": int(config["n_head"]),
            "gradient_steps": steps,
        }

    reference_seed = TRAINING_SEEDS[0]
    reference = per_seed[reference_seed]
    for seed, facts in per_seed.items():
        for field in ("target_rtg", "rtg_scale", "support", "n_rows"):
            for ix in reference["ids"]:
                if facts["ids"] != reference["ids"] or facts[field][ix] != reference[field][ix]:
                    raise ValueError(
                        f"{GRID4X4_SUBJECT} seed {seed}: intersection {ix!r} disagrees with seed "
                        f"{reference_seed} on {field} ({facts[field].get(ix)!r} against "
                        f"{reference[field][ix]!r}); the seed varies the training RNG, not the prompt"
                    )
        for field in ("scenario_id", "draw_ids", "state_dim", "context_length", "n_head"):
            if facts[field] != reference[field]:
                raise ValueError(
                    f"{GRID4X4_SUBJECT} seed {seed}: {field} {facts[field]!r} differs from seed "
                    f"{reference_seed}'s {reference[field]!r}"
                )

    return SubjectFactsPerIntersection(
        subject=GRID4X4_SUBJECT,
        scenario_id=reference["scenario_id"],
        intersection_ids=tuple(reference["ids"]),
        best_source_return=dict(reference["target_rtg"]),
        rtg_scale=dict(reference["rtg_scale"]),
        support_range=dict(reference["support"]),
        n_rows=dict(reference["n_rows"]),
        training_draw_ids=reference["draw_ids"],
        checkpoints=tuple(str(path) for path in paths.values()),
        checkpoint_sha256=digests,
        state_dim=reference["state_dim"],
        context_length=reference["context_length"],
        n_head=reference["n_head"],
        gradient_steps=reference["gradient_steps"],
    )


def per_intersection_statistics(
    sumo_returns: Mapping[int, Mapping[str, float]],
    cityflow_returns: Mapping[int, Mapping[str, float]],
    *,
    intersection_ids: Sequence[str],
) -> dict[str, Any]:
    """``S = mean`` per intersection and per nested budget k, for BOTH domains, never pooled.

    *sumo_returns* and *cityflow_returns* map ``draw_id -> {intersection_id: episode return}`` --
    the two probes of A17(b) and Amendment A4.  Budgets are A17(b)'s nested prefixes in draw order,
    ``k in PROBE_K_VALUES`` = draws ``201 .. 200+k``, the SAME draws in both domains: a ratio of two
    statistics over different demand would not be a property of the two engines.  The statistic is
    :func:`offline.rtg_calibration.probe_statistic`, unchanged, applied to ONE intersection's
    returns at a time (A17(e): no pooling).
    """
    from offline.rtg_calibration import PROBE_K_VALUES, probe_statistic

    ids = [str(ix) for ix in intersection_ids]
    if sorted(sumo_returns) != sorted(cityflow_returns):
        only_sumo = sorted(set(sumo_returns) - set(cityflow_returns))
        only_cityflow = sorted(set(cityflow_returns) - set(sumo_returns))
        raise ValueError(
            "the two probes do not cover the same draws (only on SUMO: "
            f"{only_sumo[:5]}; only on CityFlow: {only_cityflow[:5]}); Rule B's ratio is formed "
            "over the SAME demand in both domains"
        )
    draws = sorted(int(d) for d in sumo_returns)
    for label, returns in (("SUMO", sumo_returns), ("CityFlow", cityflow_returns)):
        for draw in draws:
            absent = [ix for ix in ids if ix not in returns[draw]]
            if absent:
                raise ValueError(
                    f"the {label} probe of draw {draw} records no return for intersection(s) "
                    f"{absent[:4]!r}; a statistic over the intersections that happen to be "
                    "present would be a statistic of a different population"
                )

    table: dict[str, Any] = {}
    for k in PROBE_K_VALUES:
        if len(draws) < k:
            raise ValueError(f"k={k} needs {k} probe episodes and only {len(draws)} are recorded")
        prefix = draws[:k]
        table[f"k{k}"] = {
            "k": k,
            "draw_ids": [prefix[0], prefix[-1]],
            "statistic": REGISTERED_STATISTIC,
            "per_intersection": {
                ix: {
                    "sumo": probe_statistic(
                        [float(sumo_returns[d][ix]) for d in prefix], REGISTERED_STATISTIC
                    ),
                    "cityflow": probe_statistic(
                        [float(cityflow_returns[d][ix]) for d in prefix], REGISTERED_STATISTIC
                    ),
                }
                for ix in ids
            },
        }
    return table


def per_intersection_targets(
    facts: SubjectFactsPerIntersection, statistics: Mapping[str, Any]
) -> dict[str, Any]:
    """A17(e)'s Rule B per intersection, at every recorded k, with its role and support position.

    ``target_i(k) = R_best_source,i x (S_sumo,i(k) / S_cityflow,i(k))`` through
    :func:`offline.rtg_calibration.rule_b_target`, whose association (the ratio formed first) is
    pinned by a test and makes the in-domain case an exact identity.  **Only k = 100 carries the
    registered role** (A20(b)); k = 5 and 20 are recorded and never evaluated.  The in-support
    position is each target against THAT intersection's own training range -- a diagnostic that
    never selects (A8).
    """
    from offline.rtg_calibration import rule_b_target

    out: dict[str, Any] = {}
    for ix in facts.intersection_ids:
        low, high = facts.support_range[ix]
        per_k: dict[str, Any] = {}
        for key in sorted(statistics, key=lambda name: int(statistics[name]["k"])):
            cell = statistics[key]["per_intersection"][ix]
            k = int(statistics[key]["k"])
            target = rule_b_target(
                best_source_return=facts.best_source_return[ix],
                probe_source_stat=cell["cityflow"],
                probe_target_stat=cell["sumo"],
            )
            per_k[key] = {
                "k": k,
                "rule": "B",
                "statistic": REGISTERED_STATISTIC,
                "role": ROLE_REGISTERED if k == REGISTERED_K else ROLE_RECORDED_ONLY,
                "target": target,
                "inputs": {
                    "best_source_return": facts.best_source_return[ix],
                    "probe_target_stat": cell["sumo"],
                    "probe_source_stat": cell["cityflow"],
                },
                "in_support": in_support_position(target, rtg_min=low, rtg_max=high),
            }
        out[ix] = per_k
    return out


#: Format of one per-intersection probe chunk and of ``docs/data/p7_3d_calibration.json``.
PER_INTERSECTION_FORMAT_VERSION = "p7.3d-calibration/1.0"

#: The two probe domains, and the chunk prefix each writes.
PROBE_DOMAINS: tuple[str, ...] = ("cityflow", "sumo")


def per_intersection_chunk_path(
    domain: str, draw_id: int, *, work_dir: str | Path
) -> Path:
    """Pure path arithmetic: where one draw's per-intersection probe chunk lives."""
    if domain not in PROBE_DOMAINS:
        raise ValueError(f"unknown probe domain {domain!r}; the two are {list(PROBE_DOMAINS)}")
    return Path(work_dir) / f"probe_{domain}_draw_{int(draw_id):04d}.json"


def per_intersection_chunk_is_reusable(
    payload: Mapping[str, Any],
    *,
    domain: str,
    draw_id: int,
    scenario_key: str,
    intersection_ids: Sequence[str],
    expected_decisions: int,
) -> bool:
    """Whether a chunk may be skipped on resume -- judged from its CONTENT, never its verdict.

    :func:`chunk_is_reusable`'s rule, per intersection: the stored ``two_routes_agree`` is exactly
    what a half-written or hand-made chunk would lie about, so the two returns are compared AGAIN
    here, id by id, over the ids the caller expects IN ORDER -- the env's order is part of the
    record (contract C1), and a chunk missing an id is not a chunk with fewer intersections but a
    chunk whose returns cannot be read.  Anything unreadable is simply not reusable; the caller's
    move-aside path handles it.
    """
    if not isinstance(payload, Mapping):
        return False
    try:
        if payload.get("format_version") != PER_INTERSECTION_FORMAT_VERSION:
            return False
        if str(payload["domain"]) != str(domain):
            return False
        if int(payload["draw_id"]) != int(draw_id):
            return False
        if str(payload["scenario_key"]) != str(scenario_key):
            return False
        if [str(ix) for ix in payload["intersection_ids"]] != [str(ix) for ix in intersection_ids]:
            return False
        if int(payload["decisions"]) != int(expected_decisions):
            return False
        by_reward = payload["local_return"]
        by_lanes = payload["local_return_from_lanes"]
        for ix in intersection_ids:
            if float(by_reward[str(ix)]) != float(by_lanes[str(ix)]):
                return False
        if str(domain) == "sumo":
            # A17(b)'s engine reads are part of a SUMO chunk's IDENTITY, not a verdict it stores:
            # a chunk recording a teleport, DEFAULT_VEHTYPE or the 300 s default describes an
            # episode that did not run under the registered regime, and reusing it on a resume
            # would put that episode's returns into the calibration hours later.
            if int(payload["n_teleports"]) != 0:
                return False
            if [str(t) for t in payload["vehicle_types_seen"]] != [PARITY_VTYPE_ID]:
                return False
            if str(payload["time_to_teleport_option"]) != EXPECTED_TIME_TO_TELEPORT:
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def run_cityflow_probe_per_intersection(
    draw_ids: Sequence[int],
    *,
    scenario_key: str,
    out_root: str | Path,
    work_dir: str | Path,
    engine_seed: int = DEFAULT_ENGINE_SEED,
    canary_seconds: float | None = None,
) -> dict[int, dict[str, float]]:
    """Amendment A4's source-domain probe: one CityFlow MaxPressure episode per draw, per id.

    Rule B needs ``S(R_probe_cityflow)`` and grid4x4 has none: on hangzhou it was READ from P4.3's
    committed artifact, which covers one intersection of another scenario.  So it is measured here,
    on the SAME draws the SUMO probe uses (201-300), with P4.3's own env settings -- read from that
    artifact rather than restated -- and ``reset(seed=engine_seed)`` on a fresh env per draw.

    One chunk per draw, atomic, resumable **by content**: a chunk that still satisfies
    :func:`per_intersection_chunk_is_reusable` is reused without starting an engine; one that does
    not is moved to ``failed/`` -- evidence about a run is not overwritten -- and re-rolled.
    CityFlow's engine seed is inert (``PREREGISTRATION`` §5) and is recorded anyway, because a
    field that is recorded on one domain and not the other is a field a reader must guess about.
    """
    import time

    from offline.rtg_calibration import run_probe_per_intersection

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    settings = _p4_3_env_settings()
    expected_decisions = int(settings["max_steps"])
    config_for_draw = _cityflow_config_for_draw(scenario_key, out_root)

    returns: dict[int, dict[str, float]] = {}
    for raw_draw_id in draw_ids:
        draw_id = int(raw_draw_id)
        chunk_path = per_intersection_chunk_path("cityflow", draw_id, work_dir=work)
        if chunk_path.is_file():
            try:
                existing = json.loads(chunk_path.read_bytes())
            except json.JSONDecodeError:
                existing = {}
            ids = (
                [str(ix) for ix in existing["intersection_ids"]]
                if isinstance(existing, Mapping) and "intersection_ids" in existing
                else []
            )
            if ids and per_intersection_chunk_is_reusable(
                existing,
                domain="cityflow",
                draw_id=draw_id,
                scenario_key=scenario_key,
                intersection_ids=ids,
                expected_decisions=expected_decisions,
            ):
                returns[draw_id] = {ix: float(existing["local_return"][ix]) for ix in ids}
                continue
            _move_aside(chunk_path)

        started = time.perf_counter()
        (episode,) = run_probe_per_intersection(
            draw_ids=[draw_id],
            config_for_draw=config_for_draw,
            env_settings=settings,
            scenario_id=scenario_key,
            engine_seed=int(engine_seed),
        )
        seconds = time.perf_counter() - started
        config_path = Path(config_for_draw(draw_id))
        _write_json(
            chunk_path,
            {
                "format_version": PER_INTERSECTION_FORMAT_VERSION,
                "domain": "cityflow",
                "draw_id": episode.draw_id,
                "scenario_key": scenario_key,
                "intersection_ids": list(episode.local_return),
                "local_return": dict(episode.local_return),
                "local_return_from_lanes": dict(episode.local_return_from_lanes),
                "two_routes_agree": True,
                "decisions": episode.decisions,
                "engine_seed_requested": int(engine_seed),
                "att_horizon": episode.att_horizon,
                "horizon_vehicle_count": episode.horizon_vehicle_count,
                "config_path": str(config_path),
                "config_sha256": _sha256_file(config_path),
                "env_settings_source": "docs/data/p4_3_probe.json:env_settings",
                "p4_3_probe_sha256": _sha256_file(P4_3_PROBE_ARTIFACT),
                "seconds": seconds,
                "canary_seconds": canary_seconds,
                **_git_provenance(),
            },
        )
        returns[draw_id] = dict(episode.local_return)
    return returns


def _p4_3_env_settings() -> dict[str, Any]:
    """P4.3's recorded probe settings, READ from the committed artifact and never restated.

    Measured 2026-09-19: they equal the grid4x4 corpus's own settings on every key but
    ``compare_with`` (a harness key, which P4.3's artifact strips), on all five manifests -- so the
    probe's reward stream is the stream the return-to-go advances on.  A test asserts it.
    """
    payload = json.loads(P4_3_PROBE_ARTIFACT.read_bytes())
    return dict(payload["env_settings"])


def _cityflow_config_for_draw(
    scenario_key: str, out_root: str | Path
) -> Callable[[int], Path]:
    """The draw -> CityFlow sim config lookup, through the draws tree's own path arithmetic."""
    from offline.materialise_draws import draw_config_path

    def lookup(draw_id: int) -> Path:
        return draw_config_path(scenario_key, int(draw_id), out_root=out_root)

    return lookup


def _roll_sumo_probe_episode_per_intersection(
    config_path: Path, *, engine_seed: int
) -> tuple[list[tuple[str, list[str]]], list[Mapping[str, Any]], list[float], float, dict[str, Any]]:
    """One SUMO MaxPressure episode on a multi-intersection scenario, with A17(b)'s engine reads.

    A named seam, for :func:`_roll_one_episode`'s reason: the resume path and the five refusals
    are testable without a simulator only if the roll can be substituted.

    The env is the PLAIN one -- not the observer and not aligned.  MaxPressure's pressure is a
    difference over the env's own SUMO lane ids, so it must not go through A16's door
    (``BRIEF_37`` Amendment A2), and the observer's instrumentation costs about 60 % per episode
    for a quantity this probe does not use.  The four engine reads are taken FROM THE RUNNING
    ENGINE, and the type set is sampled on the first step on which anybody is present as well as
    at the horizon, so an early wrong type cannot hide behind a horizon-only read.
    """
    from algorithms.max_pressure import MaxPressureAgent
    from experiments.envs import make_env
    from offline.collect import _build_env_spec
    from offline.sumo_att_reference import collect_style_args

    args = collect_style_args(
        "sumo", "maxpressure", config_path, sentinel_out_dir="/nonexistent"
    )
    env = make_env(_build_env_spec(args))
    try:
        ids_and_lanes = [
            (str(ix.id), [str(lane) for lane in ix.incoming_lanes]) for ix in env.intersections
        ]
        policy = MaxPressureAgent(env)
        post_step, samples, last_vehicle_count, reads = _roll_sumo_maxpressure_episode(
            env, policy, engine_seed=int(engine_seed)
        )
    finally:
        env.close()
    return ids_and_lanes, post_step, samples, last_vehicle_count, reads


def _sumo_config_for_draw(scenario_key: str, out_root: str | Path) -> Callable[[int], Path]:
    """The draw -> teleport-free parity ``.sumocfg`` lookup, through the draws tree's arithmetic."""
    from offline.materialise_draws import parity_sumocfg_path

    def lookup(draw_id: int) -> Path:
        return parity_sumocfg_path(scenario_key, int(draw_id), out_root=out_root)

    return lookup


def run_sumo_probe_per_intersection(
    draw_ids: Sequence[int],
    *,
    scenario_key: str,
    out_root: str | Path,
    work_dir: str | Path,
    engine_seed: int = DEFAULT_ENGINE_SEED,
    canary_seconds: float | None = None,
) -> dict[int, dict[str, float]]:
    """A17(b)'s probe on a multi-intersection pair: one MaxPressure episode per draw, per id.

    The TARGET-domain half of Rule B's ratio, and the counterpart of
    :func:`run_cityflow_probe_per_intersection`: the same draws, the same seed rule
    (``reset(seed=engine_seed)`` on a fresh env per draw), the same chunk contract, the same
    resume-by-content -- and, on this side, A17(b)'s five refusals per episode.

    **All five stop the run, not just the draw.**  Four are engine reads taken while the episode
    is running (teleports, the effective vehicle type, the ``time-to-teleport`` option) plus the
    decision count; the fifth is the two-route return equality, applied PER INTERSECTION and
    refusing by name.  A probe episode that silently ran ``DEFAULT_VEHTYPE`` would calibrate the
    registered prompt against the +49 % travel-time confound the parity contract exists to
    remove, and a teleport would remove a stuck vehicle from one engine's metric and not the
    other's -- neither may be averaged into a statistic and discovered afterwards.
    """
    import time

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    config_for_draw = _sumo_config_for_draw(scenario_key, out_root)

    returns: dict[int, dict[str, float]] = {}
    for raw_draw_id in draw_ids:
        draw_id = int(raw_draw_id)
        chunk_path = per_intersection_chunk_path("sumo", draw_id, work_dir=work)
        if chunk_path.is_file():
            try:
                existing = json.loads(chunk_path.read_bytes())
            except json.JSONDecodeError:
                existing = {}
            ids = (
                [str(ix) for ix in existing["intersection_ids"]]
                if isinstance(existing, Mapping) and "intersection_ids" in existing
                else []
            )
            if ids and per_intersection_chunk_is_reusable(
                existing,
                domain="sumo",
                draw_id=draw_id,
                scenario_key=scenario_key,
                intersection_ids=ids,
                expected_decisions=EXPECTED_DECISIONS,
            ):
                returns[draw_id] = {ix: float(existing["local_return"][ix]) for ix in ids}
                continue
            _move_aside(chunk_path)

        config_path = Path(config_for_draw(draw_id))
        if not config_path.is_file():
            raise FileNotFoundError(
                f"probe draw {draw_id} has no parity configuration at {config_path}; P7.3d C1 "
                "materialises the band into the MAIN tree's scenarios/draws"
            )
        started = time.perf_counter()
        ids_and_lanes, post_step, samples, last_vehicle_count, reads = (
            _roll_sumo_probe_episode_per_intersection(config_path, engine_seed=int(engine_seed))
        )
        seconds = time.perf_counter() - started

        if len(ids_and_lanes) < 2:
            raise ValueError(
                f"draw {draw_id}: this scenario has {len(ids_and_lanes)} intersection(s); "
                "run_sumo_probe is the single-intersection entry point and its scalar record is "
                "the registered one"
            )
        if len(post_step) != EXPECTED_DECISIONS:
            raise ValueError(
                f"draw {draw_id}: {len(post_step)} decisions, not {EXPECTED_DECISIONS}"
            )
        if int(reads["n_teleports"]) != 0:
            raise ValueError(
                f"draw {draw_id}: {reads['n_teleports']} teleport(s) on a configuration that "
                "requested time-to-teleport -1 (A15(c))"
            )
        if list(reads["vehicle_types_seen"]) != [PARITY_VTYPE_ID]:
            raise ValueError(
                f"draw {draw_id}: the engine ran vehicle type(s) "
                f"{list(reads['vehicle_types_seen'])}, not [{PARITY_VTYPE_ID!r}]; the parity "
                "contract is not what this episode measured"
            )
        if str(reads["time_to_teleport_option"]) != EXPECTED_TIME_TO_TELEPORT:
            raise ValueError(
                f"draw {draw_id}: SUMO reports time-to-teleport "
                f"{str(reads['time_to_teleport_option'])!r}, not {EXPECTED_TIME_TO_TELEPORT!r}"
            )

        from offline.rtg_calibration import episode_return_two_routes

        by_reward: dict[str, float] = {}
        by_lanes: dict[str, float] = {}
        for ix_id, lanes in ids_and_lanes:
            from_rewards, from_lanes = episode_return_two_routes(
                post_step, ix_id=ix_id, incoming_lanes=lanes
            )
            if from_rewards != from_lanes:
                raise ValueError(
                    f"draw {draw_id}: the two return routes disagree on intersection {ix_id!r} "
                    f"({from_rewards!r} from the reward stream against {from_lanes!r} from its "
                    "lane waiting counts); A17(b) requires equality under ==, per intersection"
                )
            by_reward[ix_id] = from_rewards
            by_lanes[ix_id] = from_lanes

        _write_json(
            chunk_path,
            {
                "format_version": PER_INTERSECTION_FORMAT_VERSION,
                "domain": "sumo",
                "draw_id": draw_id,
                "scenario_key": scenario_key,
                "intersection_ids": [ix for ix, _ in ids_and_lanes],
                "local_return": by_reward,
                "local_return_from_lanes": by_lanes,
                "two_routes_agree": True,
                "decisions": len(post_step),
                "engine_seed_requested": int(engine_seed),
                "engine_seed_drawn": int(reads["engine_seed_drawn"]),
                "n_teleports": int(reads["n_teleports"]),
                "vehicle_types_seen": list(reads["vehicle_types_seen"]),
                "time_to_teleport_option": str(reads["time_to_teleport_option"]),
                "att_horizon": samples[-1] if samples else 0.0,
                "horizon_vehicle_count": last_vehicle_count,
                "config_path": str(config_path),
                "config_sha256": _sha256_file(config_path),
                "seconds": seconds,
                "canary_seconds": canary_seconds,
                **_git_provenance(),
            },
        )
        returns[draw_id] = dict(by_reward)
    return returns


def per_intersection_returns_from_chunks(
    domain: str, *, work_dir: str | Path
) -> dict[int, dict[str, float]]:
    """``{draw_id: {intersection_id: return}}`` read back from one domain's chunks on disk.

    The artifact is built from what the chunks SAY, re-read from disk, not from what the run that
    wrote them returned in memory: the chunks are the evidence, and a resumed run's numbers must
    come from the same place a fresh one's do.
    """
    if domain not in PROBE_DOMAINS:
        raise ValueError(f"unknown probe domain {domain!r}; the two are {list(PROBE_DOMAINS)}")
    out: dict[int, dict[str, float]] = {}
    for path in sorted(Path(work_dir).glob(f"probe_{domain}_draw_*.json")):
        payload = json.loads(path.read_bytes())
        if str(payload.get("format_version")) != PER_INTERSECTION_FORMAT_VERSION:
            raise ValueError(
                f"{path}: chunk format {payload.get('format_version')!r} is not "
                f"{PER_INTERSECTION_FORMAT_VERSION!r}"
            )
        ids = [str(ix) for ix in payload["intersection_ids"]]
        out[int(payload["draw_id"])] = {ix: float(payload["local_return"][ix]) for ix in ids}
    return out


def report_per_intersection_calibration(
    *,
    work_dir: str | Path,
    out_path: str | Path,
    output_root: str | Path,
    scenario_key: str = GRID4X4_SCENARIO_KEY,
    expected_sha256: Mapping[int, str] | None = None,
    held_out_draws: Sequence[int] = HELD_OUT_DRAWS,
) -> dict[str, Any]:
    """A17(e)'s calibration artifact for a multi-intersection pair, from BOTH probes' chunks.

    **Format version** ``p7.3d-calibration/1.0``.  Per intersection: the two constants read from
    the subject's checkpoints with the FIELD each came from stated (Amendment A4), both probe
    statistics at every budget, the Rule B target at every budget with only ``k = 100`` carrying
    the registered role (A20(b)), and the in-support position -- a diagnostic that never selects.

    **Built from the chunks on disk, never from a run's memory**, so a resumed run's artifact is
    the same as a fresh one's.  **Every refusal precedes the write**, and the write is atomic: a
    refused report leaves no file at all.  The refusals: both halves must cover exactly the same
    draws and every intersection; the ids must be the subject's own, in its order; the probe band
    must be disjoint from the draws the CHECKPOINT says it trained on and from the held-out pool;
    and every budget must have the episodes it declares.
    """
    from offline.rtg_calibration import assert_probe_draws_disjoint

    work = Path(work_dir)
    facts = subject_facts_per_intersection(
        output_root=output_root, expected_sha256=expected_sha256
    )
    by_domain = {
        domain: per_intersection_returns_from_chunks(domain, work_dir=work)
        for domain in PROBE_DOMAINS
    }
    counts = {domain: len(rows) for domain, rows in by_domain.items()}
    if counts["cityflow"] != counts["sumo"] or not by_domain["sumo"]:
        raise ValueError(
            f"the two probe halves do not cover the same draws: {counts['cityflow']} cityflow "
            f"chunk(s) against {counts['sumo']} sumo chunk(s) in {work}. Rule B's ratio is formed "
            "over the SAME demand in both domains, so a missing half is a refusal, not a smaller k"
        )
    draws = sorted(by_domain["sumo"])
    for domain, rows in by_domain.items():
        for draw in draws:
            found = [str(ix) for ix in rows.get(draw, {})]
            if found != list(facts.intersection_ids):
                raise ValueError(
                    f"the {domain} chunk of draw {draw} records intersections {found[:4]}, not "
                    f"the subject's {list(facts.intersection_ids)[:4]} in its order; a statistic "
                    "over a different population is a different statistic"
                )
    assert_probe_draws_disjoint(
        draws,
        training_draw_ids=facts.training_draw_ids,
        held_out_draws=held_out_draws,
    )

    statistics_table = per_intersection_statistics(
        by_domain["sumo"], by_domain["cityflow"], intersection_ids=facts.intersection_ids
    )
    targets = per_intersection_targets(facts, statistics_table)

    per_intersection: dict[str, Any] = {}
    for ix in facts.intersection_ids:
        low, high = facts.support_range[ix]
        budgets: dict[str, Any] = {}
        for key, entry in targets[ix].items():
            budgets[key] = {
                "k": entry["k"],
                "rule": entry["rule"],
                "statistic": entry["statistic"],
                "role": entry["role"],
                "target": entry["target"],
                "probe_target_stat": entry["inputs"]["probe_target_stat"],
                "probe_source_stat": entry["inputs"]["probe_source_stat"],
                "draw_ids": statistics_table[key]["draw_ids"],
                "in_support": entry["in_support"],
            }
        per_intersection[ix] = {
            "best_source_return": facts.best_source_return[ix],
            "rtg_scale": facts.rtg_scale[ix],
            "support_range": [low, high],
            "n_training_rows": facts.n_rows[ix],
            "budgets": budgets,
        }

    artifact: dict[str, Any] = {
        "format_version": PER_INTERSECTION_FORMAT_VERSION,
        "registered_in": "PREREGISTRATION A17(e), A20(b), A21(a); BRIEF_39 C3a, Amendment A4",
        "scenario_key": scenario_key,
        "subject": facts.subject,
        "registered_statistic": REGISTERED_STATISTIC,
        "registered_k": REGISTERED_K,
        "n_draws": len(draws),
        "draw_ids": [draws[0], draws[-1]],
        "intersection_ids": list(facts.intersection_ids),
        "checkpoint_sha256": {str(seed): digest for seed, digest in facts.checkpoint_sha256.items()},
        "declared_gradient_steps": facts.gradient_steps,
        "architecture": {
            "state_dim": facts.state_dim,
            "context_length": facts.context_length,
            "n_head": facts.n_head,
            "spatial_mixing": False,
        },
        "fields_read": {
            "best_source_return": (
                'payload["target_rtg"][i] -- P5.2\'s declared per-intersection prompt, the '
                "maximum episode return in THAT intersection's training streams, and the repo's "
                "established reading (transfer_calibration.subject_facts on hangzhou). Amendment "
                "A4: A17(e)'s wording pointed at stats[\"rtg\"], which is a different quantity."
            ),
            "rtg_scale": 'payload["rtg_scale"][i]; never recalibrated (A17(a)), only the target moves',
            "support_range": (
                'payload["stats"]["rtg"][scenario][i] min/max -- a per-window summary whose max is '
                "0.0 on every intersection (the return-to-go at an episode's last step), so it "
                "bounds the SUPPORT and is not a source of any target (Amendment A4)"
            ),
            "probe_source_stat": "the mean over this intersection's 100 CityFlow probe returns",
            "probe_target_stat": "the mean over this intersection's 100 SUMO probe returns",
        },
        "probe_returns": {
            domain: {str(draw): rows[draw] for draw in draws}
            for domain, rows in by_domain.items()
        },
        "disjointness": {
            "probe_draw_ids": [draws[0], draws[-1]],
            "training_draw_ids": [facts.training_draw_ids[0], facts.training_draw_ids[-1]],
            "held_out_draw_ids": [int(held_out_draws[0]), int(held_out_draws[-1])],
            "checked_by": "offline.rtg_calibration.assert_probe_draws_disjoint",
        },
        "per_intersection": per_intersection,
        "what_this_does_not_say": [
            "These are PROBE returns -- MaxPressure's, on both engines -- and a prompt computed "
            "from them. Nothing here is an evaluation of the subject, on either backend.",
            "The targets at k = 5 and k = 20 are RECORDED and never evaluated (A20(b)); only "
            "k = 100 carries the registered role.",
            "The in-support position is a diagnostic and selects nothing (A8, BRIEF_15 12.1); a "
            "target outside its intersection's training range is reported, not adjusted.",
            "Rule B rescales a level by a ratio of probe means and so assumes the return scale "
            "shifts multiplicatively between backends (A17's stated limit); the in-support "
            "diagnostic is reported precisely because that assumption can fail.",
        ],
        **_git_provenance(),
    }

    # ---- the last refusal, then the write ----
    for ix in facts.intersection_ids:
        if per_intersection[ix]["budgets"][f"k{REGISTERED_K}"]["role"] != ROLE_REGISTERED:
            raise AssertionError(f"{ix} carries no registered prompt at k={REGISTERED_K}")
    _write_json(out_path, artifact)
    return artifact


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomic, sorted, newline-terminated -- so a chunk is either whole or absent."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)


def _git_provenance() -> dict[str, Any]:
    from offline.materialise_draws import _git_commit

    commit, dirty = _git_commit()
    return {"git_commit": commit, "git_dirty": dirty}


def _load_payload(path: str | Path) -> Mapping[str, Any]:
    import torch

    return torch.load(Path(path), map_location="cpu", weights_only=False)


def subject_facts(subject: str, *, output_root: str | Path) -> SubjectFacts:
    """Read one subject's constants from its five checkpoints and assert they agree.

    ``support_range_over_the_split`` is what :func:`offline.rtg_calibration.training_rtg_range`
    returns -- the ``stats.rtg`` range, which covers the SPLIT (216,000 rows for ``mix50``) -- and it
    is the range **A17(c) registered**.  ``training_set_return_min`` is ``-rtg_scale``, the bound of
    the training set the target itself came from.  They coincide for ``mappo1000`` and differ by 71
    on a 40,000 scale for ``mix50``; the diagnostic never selects and no claim rests on which bound
    is used (Amendment A1).
    """
    from offline.rtg_calibration import training_rtg_range

    if subject not in SUBJECTS:
        raise ValueError(f"unknown subject {subject!r}; A17(c) registers {sorted(SUBJECTS)}")
    spec = SUBJECTS[subject]
    root = Path(output_root)
    paths = [root / spec["subdir"] / f"{spec['stem']}{seed}.pt" for seed in TRAINING_SEEDS]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{subject}: checkpoints absent: {missing}")

    constants: list[tuple[Any, ...]] = []
    for path in paths:
        payload = _load_payload(path)
        stats = payload["stats"]["rtg"]
        pairs = [(s, i, v) for s, per_ix in stats.items() for i, v in per_ix.items()]
        if len(pairs) != 1:
            raise ValueError(
                f"{path}: the statistics cover {len(pairs)} (scenario, intersection) pairs; this "
                "task measures one support range"
            )
        summary = pairs[0][2]
        constants.append(
            (
                float(payload["target_rtg"]),
                float(payload["rtg_scale"]),
                int(payload["config"]["state_dim"]),
                int(payload["config"]["context_length"]),
                int(payload["provenance"]["gradient_steps"]),
                float(summary["min"]),
                float(summary["max"]),
                int(summary["count"]),
            )
        )
    if len(set(constants)) != 1:
        raise ValueError(
            f"{subject}: the five seeds disagree on the prompt constants {sorted(set(constants))}; "
            "the seed varies the training RNG, not the prompt"
        )
    target_rtg, rtg_scale, state_dim, context_length, steps, low, high, count = constants[0]
    if steps != DECLARED_GRADIENT_STEPS:
        raise ValueError(
            f"{subject}: the checkpoints record {steps} gradient steps, not the declared "
            f"{DECLARED_GRADIENT_STEPS}"
        )
    # The same range, by the registered helper rather than by this function's own reading.
    assert training_rtg_range(paths[0]) == (low, high)
    return SubjectFacts(
        subject=subject,
        best_source_return=target_rtg,
        rtg_scale=rtg_scale,
        support_range_over_the_split=(low, high),
        n_rows=count,
        training_set_return_min=-rtg_scale,
        checkpoints=tuple(str(p) for p in paths),
        state_dim=state_dim,
        context_length=context_length,
    )


def disjointness_record(draw_ids: Sequence[int], *, output_root: str | Path) -> dict[str, Any]:
    """Amendment A7.1: assert the probe band is disjoint from the UNION of every training source.

    Both checkpoints' ``provenance.training_draw_ids`` are ``1..200``; the ``mix50`` field records
    the candidate pool, **not** the 152 draws its tier trained on, which live in the declaration.
    Asserting against the union means the weaker source cannot make the assertion pass alone.
    """
    from offline.rtg_calibration import assert_probe_draws_disjoint

    root = Path(output_root)
    declaration = json.loads((_REPO_ROOT / "docs" / "data" / "p4_7_declaration.json").read_bytes())
    sources: list[dict[str, Any]] = []
    training: set[int] = set()
    for subject, spec in SUBJECTS.items():
        path = root / spec["subdir"] / f"{spec['stem']}101.pt"
        ids = [int(d) for d in _load_payload(path)["provenance"]["training_draw_ids"]]
        training.update(ids)
        sources.append(
            {"name": f"{subject}:provenance.training_draw_ids", "n": len(ids), "path": str(path)}
        )
    tier_ids = [int(d) for d in declaration["tiers"]["mix50"]["training_draws"]]
    training.update(tier_ids)
    sources.append(
        {
            "name": "p4_7_declaration:tiers.mix50.training_draws",
            "n": len(tier_ids),
            "path": "docs/data/p4_7_declaration.json",
        }
    )
    sources.append({"name": "held_out_pool", "n": len(HELD_OUT_DRAWS), "path": "PREREGISTRATION §5"})

    requested = [int(d) for d in draw_ids]
    assert_probe_draws_disjoint(
        requested, training_draw_ids=sorted(training), held_out_draws=HELD_OUT_DRAWS
    )
    for source in sources:
        if source["name"] == "held_out_pool":
            pool: set[int] = set(HELD_OUT_DRAWS)
        elif source["name"].startswith("p4_7_declaration"):
            pool = set(tier_ids)
        else:
            pool = {
                int(d)
                for d in _load_payload(
                    Path(source["path"])
                )["provenance"]["training_draw_ids"]
            }
        source["overlap_with_probe"] = len(pool & set(requested))
    return {
        "disjoint": True,
        "probe_draws": [min(requested), max(requested)],
        "n_probe_draws": len(requested),
        "sources": sources,
        "union_size": len(training | set(HELD_OUT_DRAWS)),
        "the_trap": (
            "both checkpoints' provenance.training_draw_ids are 1..200; the mix50 field records "
            "the candidate pool, not the 152 draws of tiers.mix50.training_draws, so the "
            "assertion runs against the union of all three sources"
        ),
    }


def probe_returns_from_chunks(work_dir: str | Path) -> list[tuple[int, float]]:
    """``(draw_id, local_return)`` per chunk, draw-ordered, refusing a gap or a duplicate."""
    work = Path(work_dir)
    rows: dict[int, float] = {}
    for path in sorted(work.glob("probe_draw_*.json")):
        payload = json.loads(path.read_bytes())
        draw_id = int(payload["draw_id"])
        if draw_id in rows:
            raise ValueError(f"draw {draw_id} appears in more than one chunk under {work}")
        rows[draw_id] = float(payload["local_return"])
    if not rows:
        raise ValueError(f"no probe chunks under {work}")
    expected = set(range(min(rows), max(rows) + 1))
    gaps = sorted(expected - set(rows))
    if gaps:
        raise ValueError(
            f"the probe band has {len(gaps)} missing draw(s) (first {gaps[:5]}); a missing draw "
            "silently shortens the band and moves every statistic"
        )
    return [(draw_id, rows[draw_id]) for draw_id in sorted(rows)]


def statistics_table(returns_by_draw: Sequence[tuple[int, float]]) -> dict[str, Any]:
    """The SUMO statistics per k and S, beside the CityFlow ones READ from P4.3's artifact."""
    from offline.rtg_calibration import PROBE_K_VALUES, PROBE_STATISTICS, probe_statistic

    ordered = sorted(returns_by_draw)
    committed = json.loads(P4_3_PROBE_ARTIFACT.read_bytes())["budgets"]
    table: dict[str, Any] = {
        "read_from": {
            "path": str(P4_3_PROBE_ARTIFACT.relative_to(_REPO_ROOT)),
            "sha256": _sha256_file(P4_3_PROBE_ARTIFACT),
            "never_recomputed": "A17(b): S(R_probe_cityflow) is read from this artifact, not re-run",
        }
    }
    for k in PROBE_K_VALUES:
        if len(ordered) < k:
            raise ValueError(f"k={k} needs {k} probe episodes and only {len(ordered)} are recorded")
        prefix = ordered[:k]
        values = [value for _, value in prefix]
        cell: dict[str, Any] = {"draw_ids": [prefix[0][0], prefix[-1][0]], "k": k}
        for statistic in PROBE_STATISTICS:
            cell[statistic] = {
                "sumo": probe_statistic(values, statistic),
                "cityflow": committed[f"k{k}"]["statistics"][statistic],
            }
        table[f"k{k}"] = cell
    return table


def in_support_position(target: float, *, rtg_min: float, rtg_max: float) -> dict[str, Any]:
    """Where a target sits relative to a range: below / inside / above, and the margin.

    Arithmetic, never a selector (A8; ``BRIEF_15`` §12.1).  Double-computed: the verdict is also
    obtained from :func:`offline.rtg_calibration.in_support_counts` on the one-element trajectory
    ``[target]``, so the classification is checked against the registered function rather than
    trusted.
    """
    from offline.rtg_calibration import in_support_counts

    value = float(target)
    low = float(rtg_min)
    high = float(rtg_max)
    if value < low:
        position, margin = "below", value - low
    elif value > high:
        position, margin = "above", value - high
    else:
        position, margin = "inside", 0.0

    counts = in_support_counts([value], rtg_min=low, rtg_max=high)
    by_registered_function = (
        "inside" if counts.in_support else ("below" if counts.below else "above")
    )
    if by_registered_function != position:
        raise AssertionError(
            f"the two routes disagree on {value}: {position} against {by_registered_function}"
        )
    return {
        "target_rtg": value,
        "position": position,
        "margin": margin,
        "range": [low, high],
    }


def targets_table(
    statistics: Mapping[str, Any],
    returns_by_draw: Sequence[tuple[int, float]],
    facts: Mapping[str, SubjectFacts],
) -> dict[str, Any]:
    """Every target per subject x rule x S x k, with the registered prompt marked.

    Eight rows per subject: Rule B over {mean, max} x {5, 20, 100} (six), Rule A at ``q = 1.0``
    (one) and the naive prompt (one).  **Rule B / mean / k = 100 is A17(c)'s registered prompt**;
    everything else is an ablation.  Rule A's secondary quantiles are computed as VALUES ONLY, as
    P4.3 did, and are not targets.
    """
    from offline.rtg_calibration import (
        PROBE_K_VALUES,
        PROBE_STATISTICS,
        RULE_A_K,
        RULE_A_QUANTILE,
        SECONDARY_QUANTILES,
        rule_a_target,
        rule_b_target,
    )

    ordered = sorted(returns_by_draw)
    out: dict[str, Any] = {}
    for subject, subject_facts_ in facts.items():
        rows: list[dict[str, Any]] = []
        for k in PROBE_K_VALUES:
            for statistic in PROBE_STATISTICS:
                cell = statistics[f"k{k}"][statistic]
                target = rule_b_target(
                    best_source_return=subject_facts_.best_source_return,
                    probe_source_stat=cell["cityflow"],
                    probe_target_stat=cell["sumo"],
                )
                registered = statistic == "mean" and k == 100
                rows.append(
                    {
                        "rule": "rule_b",
                        "statistic": statistic,
                        "k": k,
                        "target_rtg": target,
                        "role": "registered_prompt" if registered else "ablation",
                        "inputs": {
                            "best_source_return": subject_facts_.best_source_return,
                            "probe_source_stat": cell["cityflow"],
                            "probe_target_stat": cell["sumo"],
                        },
                    }
                )
        rule_a_values = [value for _, value in ordered[:RULE_A_K]]
        rows.append(
            {
                "rule": "rule_a",
                "statistic": f"q{RULE_A_QUANTILE}",
                "k": RULE_A_K,
                "target_rtg": rule_a_target(rule_a_values, RULE_A_QUANTILE),
                "role": "ablation",
                "inputs": {"quantile": RULE_A_QUANTILE, "n": len(rule_a_values)},
            }
        )
        rows.append(
            {
                "rule": "naive",
                "statistic": "none",
                "k": None,
                "target_rtg": subject_facts_.best_source_return,
                "role": "ablation",
                "inputs": {"best_source_return": subject_facts_.best_source_return},
            }
        )
        for row in rows:
            row["in_support"] = in_support_position(
                row["target_rtg"],
                rtg_min=subject_facts_.support_range_over_the_split[0],
                rtg_max=subject_facts_.support_range_over_the_split[1],
            )
        out[subject] = rows
        out.setdefault("rule_a_secondary_values", {})[subject] = {
            f"q{q}": rule_a_target(rule_a_values, q) for q in SECONDARY_QUANTILES
        }
    return out


def rtg_advanced_every_decision(
    rtg_series: Sequence[float], rewards_in_info: Sequence[float | None]
) -> bool:
    """Did the RTG move exactly on the decisions whose driving reward was non-zero?

    ⚠️ **THE ALIGNMENT IS SHIFTED BY ONE, AND THAT IS NOT AN OFF-BY-ONE -- IT IS THE AGENT'S RULE**
    (Amendment D1; the first version of this check got it wrong and the campaign recorded ``False``
    for a mechanism that was working).  ``run_smoke`` records ``agent.current_rtg()`` **before**
    ``agent.act(info_t)``, and ``DTAgent.act`` updates ``reward_sum`` **inside** that call
    (``agent/DTAgent.py``: ``current_rtg`` is ``target - reward_sum``; ``act`` adds
    ``0.0 if step == 0 else self._reward_for(...)``).  So::

        rtg[t] - rtg[t-1] == -r(info_{t-1})          for t >= 2
        rtg[1] - rtg[0]   == 0                        ALWAYS

    The second line is the agent forcing the step-0 reward to ``0.0`` regardless of what the info
    carries, so index 1 is **forced unchanged** and is not evidence either way.  Comparing the
    change at ``t`` with ``r(info_t)`` -- the current info's reward -- returns ``False`` as soon as
    two consecutive rewards differ, which on a real episode is immediately.

    Returns ``True`` when every decision agrees with the rule.  A missing reward on a decision that
    needs one makes it ``False``: an absent reward cannot be shown to be zero.
    """
    if len(rtg_series) != len(rewards_in_info):
        raise ValueError(
            f"the RTG series has {len(rtg_series)} entries and the reward series "
            f"{len(rewards_in_info)}; they are recorded per decision and must agree"
        )
    for index in range(1, len(rtg_series)):
        changed = rtg_series[index] != rtg_series[index - 1]
        if index == 1:
            expected = False  # the agent's step-0 rule; see the docstring
        else:
            reward = rewards_in_info[index - 1]
            if reward is None:
                return False
            expected = float(reward) != 0.0
        if changed != expected:
            return False
    return True


def registered_prompt_for(
    subject: str, *, work_dir: str | Path, output_root: str | Path
) -> tuple[float, SubjectFacts]:
    """The subject's Rule B / mean / k=100 target, through the SAME path ``report`` uses.

    One code path for the target means the smoke cannot condition on a number the artifact does not
    also publish -- which is why the smoke runs AFTER the probe (Amendment A4).
    """
    rows = probe_returns_from_chunks(work_dir)
    statistics = statistics_table(rows)
    facts = subject_facts(subject, output_root=output_root)
    table = targets_table(statistics, rows, {subject: facts})
    registered = [row for row in table[subject] if row["role"] == "registered_prompt"]
    if len(registered) != 1:
        raise ValueError(f"{subject}: {len(registered)} registered prompts, expected exactly one")
    return float(registered[0]["target_rtg"]), facts


def run_smoke(
    subject: str,
    *,
    out_root: str | Path,
    work_dir: str | Path,
    output_root: str | Path,
    draw_id: int = SMOKE_DRAW_ID,
    seed: int = DEFAULT_ENGINE_SEED,
) -> dict[str, Any]:
    """The fenced proof of mechanics: a CityFlow-trained DT driving SUMO through the aligned door.

    **It evaluates nothing.**  What is recorded is that the path runs: the decision count, that the
    first RTG is the target, that the RTG advanced exactly on the decisions whose reward was
    non-zero, how many decisions sat inside the registered range, that every action was in range,
    and what the engine reported about type and regime.  ``att_horizon``, ``episode_reward``,
    ``rtg_last`` and the RTG series go under :data:`FENCED_KEY`, which :func:`report` refuses to
    emit (Amendment A3).
    """
    import time

    import numpy as np

    from offline.aligned_env import aligned_sumo_env_for_draw
    from offline.horizon_metric import horizon_rollout
    from offline.rtg_calibration import agent_with_target, in_support_counts

    target, facts = registered_prompt_for(subject, work_dir=work_dir, output_root=output_root)
    checkpoint = Path(facts.checkpoints[0])

    env = aligned_sumo_env_for_draw(SCENARIO_KEY, draw_id, out_root=out_root)
    rtg_series: list[float] = []
    rewards_in_info: list[float | None] = []
    actions: list[int] = []
    started = time.perf_counter()
    try:
        agent = agent_with_target(
            env,
            checkpoint,
            declared_gradient_steps=DECLARED_GRADIENT_STEPS,
            target_rtg=target,
        )
        ix_id = str(list(env.intersections)[0].id)
        option = str(env._sumo.simulation.getOption("time-to-teleport"))

        def choose(_env: Any, info: Mapping[str, Any]) -> Any:
            rtg_series.append(float(agent.current_rtg()[ix_id]))
            payload = info["intersections"][ix_id]
            rewards_in_info.append(
                None if "reward" not in payload else float(payload["reward"])
            )
            # explore=False is the DECLARED EVALUATION PATH (DTAgent.act's own docstring):
            # the argmax over masked logits. The default is explore=True, which samples from
            # the masked softmax through an unseeded torch.multinomial -- and the first run of
            # this smoke took that default, which is why n_decisions_in_support moved 271 ->
            # 231 between two runs of the same seed, checkpoint and draw.
            # FIFTEEN other call sites in this repository pass explore=False, and the smoke must
            # match them or it demonstrates a path P7.3 will not take (Amendment E1, corrected by
            # E1.1). Counted by AST over offline/ and experiments/ -- calls carrying an explicit
            # explore=False keyword, so docstrings cannot inflate it -- as dt_gate x2,
            # method_tier_grid x3, offline_baselines x2, and one each in admission_probe,
            # att_rederivation, collect, rtg_ablation, rtg_calibration, spatial_mixing,
            # tier_sweep and experiments/runner.py. This call is the sixteenth.
            action = agent.act(info, explore=False, update_memory=True)
            actions.append(int(np.asarray(action).reshape(-1)[0]))
            return action

        rollout = horizon_rollout(env, choose, 1, int(seed))
        types_seen = sorted(
            {env._sumo.vehicle.getTypeID(v) for v in env._sumo.vehicle.getIDList()}
        )
    finally:
        env.close()
    seconds = time.perf_counter() - started

    advanced_correctly = rtg_advanced_every_decision(rtg_series, rewards_in_info)

    counts = in_support_counts(
        rtg_series,
        rtg_min=facts.support_range_over_the_split[0],
        rtg_max=facts.support_range_over_the_split[1],
    )
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "subject": subject,
        "draw_id": int(draw_id),
        "checkpoint": str(checkpoint),
        "target_rtg": target,
        "decisions": len(actions),
        "rtg_first": rtg_series[0] if rtg_series else None,
        "rtg_advanced_every_decision": advanced_correctly,
        "n_decisions_in_support": counts.in_support,
        "actions_in_range": all(0 <= action < 8 for action in actions),
        "vehicle_types_seen": types_seen,
        "time_to_teleport_option": option,
        "seconds": seconds,
        FENCED_KEY: {
            "why": (
                "A17 and BRIEF_36 §2 fence the smoke's outcome: it is a zero-shot number and P7.3's "
                "brief is not written. report() refuses to copy this key into docs/data/"
            ),
            "att_horizon": rollout.att_horizon,
            "episode_reward": rollout.episode_reward,
            "rtg_last": rtg_series[-1] if rtg_series else None,
            "rtg_series": rtg_series,
        },
        **_git_provenance(),
    }


#: The only smoke fields that may reach ``docs/data/``.  A whitelist, not a blacklist: a new fenced
#: field added later is excluded by default rather than by being remembered.
_SMOKE_PUBLISHED_FIELDS: tuple[str, ...] = (
    "subject",
    "draw_id",
    "checkpoint",
    "target_rtg",
    "decisions",
    "rtg_first",
    "rtg_advanced_every_decision",
    "n_decisions_in_support",
    "actions_in_range",
    "vehicle_types_seen",
    "time_to_teleport_option",
    "seconds",
)


def _validate_chunk(payload: Mapping[str, Any], path: Path) -> None:
    """Every A17(b) property a probe chunk must carry, each refusal naming its own reason."""
    if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"{path.name}: format_version {payload.get('format_version')!r} is not "
            f"{ARTIFACT_FORMAT_VERSION!r}"
        )
    if float(payload["local_return"]) != float(payload["local_return_from_lanes"]):
        raise ValueError(
            f"{path.name}: the two routes disagree ({payload['local_return']!r} against "
            f"{payload['local_return_from_lanes']!r}); A17(b) requires equality under =="
        )
    if int(payload["n_teleports"]) != 0:
        raise ValueError(
            f"{path.name}: {payload['n_teleports']} teleport(s) under A15(c)'s teleport-free regime"
        )
    if list(payload["vehicle_types_seen"]) != [PARITY_VTYPE_ID]:
        raise ValueError(
            f"{path.name}: the engine ran {payload['vehicle_types_seen']!r}, not "
            f"[{PARITY_VTYPE_ID!r}]"
        )
    if str(payload["time_to_teleport_option"]) != EXPECTED_TIME_TO_TELEPORT:
        raise ValueError(
            f"{path.name}: SUMO reported time-to-teleport "
            f"{payload['time_to_teleport_option']!r}"
        )
    if int(payload["decisions"]) != EXPECTED_DECISIONS:
        raise ValueError(f"{path.name}: {payload['decisions']} decisions, not {EXPECTED_DECISIONS}")
    if str(payload.get("p4_3_probe_sha256")) != P4_3_PROBE_SHA256:
        raise ValueError(
            f"{path.name}: it records source statistics from sha256 "
            f"{payload.get('p4_3_probe_sha256')!r}, not the pinned {P4_3_PROBE_SHA256!r}"
        )


def _read_canary_record(work: Path) -> dict[str, Any]:
    """Read ``<work>/canary.json`` and RE-RUN :func:`check_canary` on its facts.

    Amendment E1.4 item 2.  The record is a claim, and a claim is not evidence: ``report`` checks
    the facts itself rather than trusting that whatever wrote the file checked them.  A missing
    file is a refusal, not a fallback -- from that amendment on, a work directory without one is
    not a run, and the previous behaviour (fall back on the probe chunks' ``canary_seconds``) is
    precisely how run 3's artifact came to publish run 1's canary as its own.
    """
    path = work / CANARY_RECORD_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: this run wrote no {CANARY_RECORD_NAME}. The driver writes it from the canary "
            "line immediately after the token is consumed; a work directory without one is not a "
            "run, and the chunks' canary_seconds is the canary of whoever ROLLED them, not of "
            "whoever is reporting (Amendment E1.4)"
        )
    record = json.loads(path.read_bytes())
    if not isinstance(record, dict):
        raise ValueError(f"{CANARY_RECORD_NAME}: the record is a {type(record).__name__}, not an object")
    facts = record.get("facts")
    if not isinstance(facts, dict):
        raise ValueError(
            f"{CANARY_RECORD_NAME}: 'facts' is {type(facts).__name__}, not an object; the four "
            "observed values are what the artifact reports and what check_canary compares"
        )
    try:
        seconds = float(record["seconds"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"{CANARY_RECORD_NAME}: 'seconds' is {record.get('seconds')!r}, not a number"
        ) from None
    check_canary(facts)
    # The coerced values last, so they win over the raw ones this dict is built from.
    return {**record, "seconds": seconds, "facts": facts}


def report(*, work_dir: str | Path, out_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Build the committed artifact from the chunks. Every refusal precedes every write."""
    work = Path(work_dir)
    target_path = Path(out_path)

    on_disk = _sha256_file(P4_3_PROBE_ARTIFACT)
    if on_disk != P4_3_PROBE_SHA256:
        raise ValueError(
            f"{P4_3_PROBE_ARTIFACT.name}: sha256 {on_disk} against the pinned {P4_3_PROBE_SHA256}"
        )

    # Amendment E1.4 items 1-2: THIS run's canary, from the file the driver parked after the token.
    canary_record = _read_canary_record(work)

    chunks: list[Mapping[str, Any]] = []
    for path in sorted(work.glob("probe_draw_*.json")):
        payload = json.loads(path.read_bytes())
        _validate_chunk(payload, path)
        chunks.append(payload)
    if not chunks:
        raise ValueError(f"no probe chunks under {work}")

    # Amendment E1.4 item 3: the artifact reports the chunks' own commit, so the chunks must carry
    # ONE. They are rolled once and reused by every re-roll, so this is the revision that produced
    # the probe table -- which is not, in general, the revision that wrote the artifact.
    chunk_commits = {chunk.get("git_commit") for chunk in chunks}
    if len(chunk_commits) != 1:
        raise ValueError(
            f"the probe chunks carry {len(chunk_commits)} distinct git commits "
            f"({sorted(str(commit) for commit in chunk_commits)}); they were rolled by more than "
            "one commit, so no single revision describes the code that produced the probe table"
        )
    chunk_commit = chunk_commits.pop()

    rows = probe_returns_from_chunks(work)
    statistics = statistics_table(rows)
    facts = {name: subject_facts(name, output_root=output_root) for name in SUBJECTS}
    targets = targets_table(statistics, rows, facts)
    disjointness = disjointness_record([draw_id for draw_id, _ in rows], output_root=output_root)

    smoke: list[dict[str, Any]] = []
    for path in sorted(work.glob("smoke_*.json")):
        payload = json.loads(path.read_bytes())
        if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(f"{path.name}: format_version {payload.get('format_version')!r}")
        absent = [field for field in _SMOKE_PUBLISHED_FIELDS if field not in payload]
        if absent:
            raise ValueError(
                f"{path.name}: the smoke chunk is missing {absent}; a published field that is "
                "absent would silently shrink what the artifact reports about the mechanics"
            )
        smoke.append({field: payload[field] for field in _SMOKE_PUBLISHED_FIELDS})

    # Amendment D2: the rate basis of the run, lifted from the chunks rather than taken on trust.
    # One campaign, one canary -- more than one distinct value means two runs' chunks were mixed,
    # which would make every "seconds" in the table incomparable.
    canary_values = {chunk.get("canary_seconds") for chunk in chunks}
    if len(canary_values) != 1:
        raise ValueError(
            f"the chunks carry {len(canary_values)} distinct canary values "
            f"({sorted(v for v in canary_values if v is not None)}); one campaign has one canary, "
            "so these chunks come from more than one run and their timings are not comparable"
        )
    canary_value = canary_values.pop()

    # ---------------------------------------------------------------- the canary block
    # Amendment E1.4 items 3 and 4.  TWO canaries exist and run 3's artifact conflated them: the
    # block's own seconds and verdict describe THE RUN THAT WROTE THIS FILE, read from canary.json
    # and re-checked above; the probe chunks' canary is reported beside it under its own name,
    # because the chunks are rolled once and reused unchanged by every re-roll.  `checked_against`
    # sits in the first of those and nowhere else -- it names the references the OBSERVED values
    # beside it were compared with, so a reader of docs/data/ alone can redo the comparison
    # instead of taking a "checked" flag on trust.
    this_run_seconds = float(canary_record["seconds"])
    canary_block: dict[str, Any] = {
        "seconds": this_run_seconds,
        "threshold_seconds": CANARY_MAX_SECONDS,
        "verdict": "at speed" if this_run_seconds <= CANARY_MAX_SECONDS else "throttled",
        "observed": dict(canary_record["facts"]),
        "checked_against": {
            "local_return": CANARY_REFERENCE_LOCAL_RETURN,
            "att_horizon": CANARY_REFERENCE_ATT_HORIZON,
            "decisions": CANARY_REFERENCE_DECISIONS,
            "two_routes_agree": True,
            "comparison": "== (draw 0 is the nominal control; every measurement on record agrees)",
        },
        "source": (
            f"output/p7_2b/{CANARY_RECORD_NAME}, written by the driver from the canary line right "
            "after the token; the same line is appended to the manifested logs/canary.log, and "
            "report re-ran check_canary on these facts before writing this file"
        ),
        "git_commit": canary_record.get("git_commit"),
        "git_dirty": canary_record.get("git_dirty"),
        "recipe": (
            "PROJECT_PLAN section 7's rule; run_probe on CityFlow draw 0 through the committed "
            "settings (BRIEF_36 section 3.3). A guest that reports a low load can still be running "
            "on a throttled host, so the rate basis is measured in the same session or not written "
            "down."
        ),
        "probe_chunks_canary": {
            "seconds": None if canary_value is None else float(canary_value),
            "git_commit": chunk_commit,
            "correctness_half": "not recorded in these chunks",
            "what_this_is": (
                "the canary_seconds field the probe chunks carry, identical across all of them. A "
                "re-roll reuses chunks by content and leaves them untouched, so this is the rate "
                "basis of the probe TABLE and not necessarily of the run that wrote this file. The "
                "chunks record a duration and no engine facts, which is what correctness_half says."
            ),
        },
    }

    artifact: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "registered_in": "PREREGISTRATION A17",
        "scenario_key": SCENARIO_KEY,
        "canary": canary_block,
        "probe": [
            {
                key: chunk[key]
                for key in (
                    "draw_id",
                    "local_return",
                    "local_return_from_lanes",
                    "att_horizon",
                    "horizon_vehicle_count",
                    "decisions",
                    "engine_seed_requested",
                    "engine_seed_drawn",
                    "n_teleports",
                    "vehicle_types_seen",
                    "seconds",
                    "canary_seconds",
                )
            }
            for chunk in sorted(chunks, key=lambda c: int(c["draw_id"]))
        ],
        "statistics": statistics,
        "subjects": {
            name: {
                "best_source_return": subject.best_source_return,
                "rtg_scale": subject.rtg_scale,
                "support_range_over_the_split": list(subject.support_range_over_the_split),
                "n_rows": subject.n_rows,
                "training_set_return_min": subject.training_set_return_min,
                "the_two_bounds": (
                    "support_range_over_the_split is the checkpoint's stats.rtg range, which A17(c) "
                    "registered and which the diagnostic is computed against; "
                    "training_set_return_min is -rtg_scale, the bound of the training set the "
                    "target came from. They coincide for mappo1000 and differ by 71 on a 40,000 "
                    "scale for mix50. The diagnostic never selects and no claim rests on which "
                    "bound is used."
                ),
                "checkpoints": list(subject.checkpoints),
                "state_dim": subject.state_dim,
                "context_length": subject.context_length,
            }
            for name, subject in facts.items()
        },
        "targets": {name: targets[name] for name in facts},
        "rule_a_secondary_values": targets.get("rule_a_secondary_values", {}),
        "disjointness": disjointness,
        "smoke": smoke,
        "what_this_does_not_say": (
            "This task evaluates nothing. No held-out draw was rolled, no rho was computed and no "
            "ATT or return of any DT is reported anywhere: the smoke's outcome is fenced. The "
            "zero-shot point, the calibrated-vs-naive contrast (exploratory per PREREGISTRATION "
            "§2 and A17(d)) and every transfer-curve number are P7.3's."
        ),
        **_git_provenance(),
    }

    serialised = json.dumps(artifact, indent=2, sort_keys=True)
    if FENCED_KEY in serialised:
        raise AssertionError(
            f"{FENCED_KEY} reached the artifact; the fence exists so a zero-shot number cannot be "
            "published before P7.3's brief is written"
        )
    _write_json(target_path, artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m offline.transfer_calibration",
        description=(
            "P7.2b: the SUMO target-domain probe and A17's return-prompt calibration. "
            "Evaluates nothing."
        ),
    )
    parser.add_argument("--draws-root", default="/home/filip/rltraffic/scenarios/draws")
    parser.add_argument("--output-root", default="/home/filip/rltraffic/output")
    parser.add_argument("--work-dir", default="/home/filip/rltraffic/output/p7_2b")
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "docs" / "data"))
    parser.add_argument("--engine-seed", type=int, default=DEFAULT_ENGINE_SEED)
    parser.add_argument("--canary-seconds", type=float, default=None)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    probe = subparsers.add_parser("probe", help="one MaxPressure episode per draw, on SUMO")
    probe.add_argument("--draws-range", type=int, nargs=2, metavar=("START", "END"))

    smoke = subparsers.add_parser("smoke", help="the fenced mechanics proof for one subject")
    smoke.add_argument("--subject", required=True, choices=sorted(SUBJECTS))
    smoke.add_argument("--draw-id", type=int, default=SMOKE_DRAW_ID)

    subparsers.add_parser("report", help="write docs/data/p7_2b_calibration.json")
    subparsers.add_parser("canary", help="the machine-health canary (PROJECT_PLAN §7)")

    record = subparsers.add_parser(
        "record-canary", help="park the canary line in the work directory (Amendment E1.4)"
    )
    record.add_argument("--line", required=True)

    # P7.3c (BRIEF_41 C1-C2): the grid4x4 corpus run's three module calls, in the driver's order.
    corpus_defaults = [GRID4X4_CORPUS_DRAWS[0], GRID4X4_CORPUS_DRAWS[-1] + 1]
    preflight = subparsers.add_parser(
        "corpus-preflight",
        help="P7.3c: the corpus run's pre-token checks -- inputs by digest, disjointness (writes nothing)",
    )
    preflight.add_argument("--draws-range", type=int, nargs=2, metavar=("START", "END"), default=corpus_defaults)
    preflight.add_argument("--data-dir", default=str(_REPO_ROOT / "docs" / "data"))
    collect_corpus = subparsers.add_parser(
        "collect-corpus", help="P7.3c: the probe replayed through the logger, one episode per draw"
    )
    collect_corpus.add_argument("--corpus-dir", required=True)
    collect_corpus.add_argument("--scenario-key", default=GRID4X4_SCENARIO_KEY)
    collect_corpus.add_argument(
        "--draws-range", type=int, nargs=2, metavar=("START", "END"), default=corpus_defaults
    )
    gate = subparsers.add_parser(
        "corpus-gate", help="P7.3c: A17(f)'s gate per intersection; writes its record only if it passes"
    )
    gate.add_argument("--corpus-dir", required=True)
    gate.add_argument("--draws-range", type=int, nargs=2, metavar=("START", "END"), default=corpus_defaults)
    gate.add_argument("--data-dir", default=str(_REPO_ROOT / "docs" / "data"))
    gate.add_argument("--record", required=True)
    return parser


def _draws_range(values: Sequence[int]) -> range:
    """A half-open ``--draws-range START END``, refused when it selects nothing."""
    start, end = (int(value) for value in values)
    if end <= start:
        raise ValueError(f"--draws-range is half-open [START, END); [{start}, {end}) selects no draw")
    return range(start, end)


def canary_seconds() -> tuple[float, dict[str, Any]]:
    """``run_probe`` on CityFlow draw 0 through the committed settings, timed.

    ``PROJECT_PLAN`` §7's rule, recipe in ``BRIEF_36`` §3.3: a guest that reports a low load can
    still be running on a throttled host, so a timing needs a canary measured in the same session.
    """
    import time

    from offline.materialise_draws import draw_config_path
    from offline.rtg_calibration import run_probe

    probe = json.loads(P4_3_PROBE_ARTIFACT.read_bytes())
    started = time.perf_counter()
    episodes = run_probe(
        draw_ids=[0],
        config_for_draw=lambda draw_id: draw_config_path(
            SCENARIO_KEY, draw_id, out_root="/home/filip/rltraffic/scenarios/draws"
        ),
        env_settings=probe["env_settings"],
        scenario_id=str(probe["scenario_id"]),
        engine_seed=int(probe["engine_seed"]),
    )
    elapsed = time.perf_counter() - started
    episode = episodes[0]
    return elapsed, {
        "decisions": episode.decisions,
        "local_return": episode.local_return,
        "att_horizon": episode.att_horizon,
        "two_routes_agree": episode.local_return == episode.local_return_from_lanes,
    }


def check_canary(facts: Mapping[str, Any]) -> None:
    """Raise unless the canary episode reproduced the engine's recorded answers exactly.

    The timing half says the machine is at speed; this half says the ENGINE still computes what it
    computed when the references were measured.  A canary that only times is a canary that would
    have passed while CityFlow returned different numbers (Amendment E1.2).

    Compared under ``==`` and not within a tolerance: draw 0 is the nominal control and every
    measurement of it on record -- P4.3's probe return, the P0.2 baseline, P7.1's five identical
    episodes -- agrees to the last digit.  A near-miss is a finding about the engine.
    """
    observed_return = facts.get("local_return")
    if observed_return != CANARY_REFERENCE_LOCAL_RETURN:
        raise ValueError(
            f"canary local_return {observed_return!r} != the recorded "
            f"{CANARY_REFERENCE_LOCAL_RETURN!r}; the engine did not reproduce draw 0"
        )
    observed_att = facts.get("att_horizon")
    if observed_att != CANARY_REFERENCE_ATT_HORIZON:
        raise ValueError(
            f"canary att_horizon {observed_att!r} != the recorded "
            f"{CANARY_REFERENCE_ATT_HORIZON!r}; the engine did not reproduce draw 0"
        )
    if facts.get("two_routes_agree") is not True:
        raise ValueError(
            f"canary two_routes_agree is {facts.get('two_routes_agree')!r}, not True; the "
            "per-intersection return and the lane recomputation disagree on the control episode"
        )
    observed_decisions = facts.get("decisions")
    if observed_decisions != CANARY_REFERENCE_DECISIONS:
        raise ValueError(
            f"canary decisions {observed_decisions!r} != {CANARY_REFERENCE_DECISIONS!r}; the "
            "control episode did not run to the horizon"
        )


#: The canary line's grammar, as two literals both halves of the round trip share:
#: ``canary <seconds:.2f> s <JSON facts>``.
_CANARY_LINE_PREFIX = "canary "
_CANARY_LINE_SEPARATOR = " s "



def assert_logged_corpus_matches_probe(
    corpus_dir: str | Path,
    artifact_path: str | Path = _REPO_ROOT / "docs" / "data" / "p7_2b_calibration.json",
    *,
    draw_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """A17(f): every logged episode reproduces P7.2b's probe return bit-for-bit.

    ``PREREGISTRATION`` A17(f), executed (``BRIEF_37`` §3.2).  P7.3 collects the k-shot corpora by
    running the same probe through the trajectory logger; per draw, the logged episode's return
    must equal P7.2b's probe return **bit-for-bit**, and ``engine_seed_drawn`` must equal.
    **100/100 or P7.3 stops.**

    WHY THE TWO NUMBERS ARE THE SAME DEFINITION, not two that happen to agree
    ------------------------------------------------------------------------
    The probe appends ``float(payload["reward"])`` over the post-step infos
    (:func:`offline.rtg_calibration.episode_return_two_routes`, ``rtg_calibration.py:345``); the
    logger stores ``float(payload["reward"])`` from the same field of the same info
    (``trajectory_logger.py:794``).  Same expression, same object, same order, and hz1x1 has ONE
    intersection, so the sum of ``ix0_local_reward`` **is** ``local_return``.  Agreement confirms
    the arithmetic; this paragraph is the claim.

    INTEGRALITY IS CHECKED BEFORE EQUALITY, and that ordering is load-bearing
    -------------------------------------------------------------------------
    The logger stores float32 (``trajectory_logger.py:118``) and the probe sums float64, so ``==``
    is achievable only because this reward family is a vehicle COUNT: integers are exact in float32
    to 2**24 and an episode total is about 3e4, six orders below.  Measured over 6,480 stored values
    of the committed ``datasets_v11`` corpus -- every one integral.  If that ever stops holding, the
    gate must fail saying *the reward is not integral* rather than *the transfer is broken*, so the
    check runs first and names itself.  **A non-integral reward is a finding, never a tolerance.**
    """
    import numpy as np

    corpus = Path(corpus_dir)
    artifact = json.loads(Path(artifact_path).read_bytes())
    probe = {int(row["draw_id"]): row for row in artifact["probe"]}

    manifest = json.loads((corpus / "manifest.json").read_bytes())
    drawn_by_id = {
        int(entry["draw_id"]): entry
        for entry in manifest.get("run_metadata", {}).get("sumo_draws", [])
    }

    episodes: dict[int, Path] = {}
    for path in sorted(corpus.glob("*.npz")):
        with np.load(path) as payload:
            draw = int(payload["flow_draw"])
        if draw in episodes:
            raise ValueError(
                f"draw {draw} has more than one logged episode in {corpus}; A17(f) compares one "
                "episode per draw and would otherwise silently use whichever sorted last"
            )
        episodes[draw] = path

    wanted = sorted(episodes) if draw_ids is None else [int(d) for d in draw_ids]
    missing = [d for d in wanted if d not in episodes]
    if missing:
        raise ValueError(f"the corpus has no logged episode for draw(s) {missing}")
    unknown = [d for d in wanted if d not in probe]
    if unknown:
        raise ValueError(
            f"{Path(artifact_path).name} records no probe row for draw(s) {unknown}, so there is "
            "nothing to compare them against"
        )

    rows: list[dict[str, Any]] = []
    for draw in wanted:
        with np.load(episodes[draw]) as payload:
            rewards = np.asarray(payload["ix0_local_reward"])
        if not bool(np.all(rewards == np.rint(rewards))):
            offending = rewards[rewards != np.rint(rewards)]
            raise ValueError(
                f"draw {draw}: {offending.size} of {rewards.size} stored per-step rewards are NOT "
                f"integral (first: {float(offending[0])!r}). The corpus stores float32 and the "
                "probe sums float64, and A17(f)'s bit-for-bit equality is achievable only because "
                "this reward family is a vehicle count. A non-integral reward means the two are no "
                "longer the same arithmetic -- a finding about the reward, not a tolerance to widen"
            )
        logged_return = float(np.sum(rewards.astype(np.float64)))
        expected_return = float(probe[draw]["local_return"])
        expected_seed = int(probe[draw]["engine_seed_drawn"])
        logged_seed = int(drawn_by_id.get(draw, {}).get("engine_seed_drawn", -1))
        matches = logged_return == expected_return and logged_seed == expected_seed
        rows.append(
            {
                "draw_id": draw,
                "matches": matches,
                "logged_return": logged_return,
                "probe_local_return": expected_return,
                "difference": logged_return - expected_return,
                "logged_engine_seed_drawn": logged_seed,
                "probe_engine_seed_drawn": expected_seed,
                "episode": episodes[draw].name,
            }
        )

    bad = [row for row in rows if not row["matches"]]
    if bad:
        first = bad[0]
        raise ValueError(
            f"A17(f) FAILED on {len(bad)} of {len(rows)} draw(s); the first is draw "
            f"{first['draw_id']}: logged return {first['logged_return']!r} against the probe's "
            f"{first['probe_local_return']!r} (difference {first['difference']!r}), "
            f"engine_seed_drawn {first['logged_engine_seed_drawn']} against "
            f"{first['probe_engine_seed_drawn']}. SUMO is deterministic under a fixed seed, so "
            "this is a wiring defect and P7.3 stops here rather than reporting a transfer number "
            "built on an episode that is not the one the prompt was calibrated from"
        )
    return {
        "n_checked": len(rows),
        "n_matching": len(rows) - len(bad),
        "all_match": not bad,
        "artifact_sha256": _sha256_file(artifact_path),
        "rows": rows,
    }


# ======================================================================================
# P7.3c (BRIEF_41 C1, Amendment A): the grid4x4 SUMO corpus -- the door's argv, the pre-token record,
# and A17(f)'s gate PER INTERSECTION, keyed by id
# ======================================================================================

#: P7.3d's committed zero-shot artifact.  Here it is the source of the ONE engine seed every corpus episode
#: must record (Amendment A, Q14: 437485271 on all 700 of its cells), read at this pin and never from the
#: gitignored probe chunks.  It moves only in a commit that also moves the artifact.
P7_3D_ZERO_SHOT_ARTIFACT_NAME = "p7_3d_grid4x4.json"
P7_3D_ZERO_SHOT_ARTIFACT_SHA256 = "c63c371ff14d208d16b9fbfa6d3daa31975679c90b20b5a4b44fbed60760b0b7"
#: A17(b)'s probe band, which the corpus replays: draws 201-300 (the calibration artifact's
#: ``disjointness.probe_draw_ids``; a test compares the two).
GRID4X4_CORPUS_DRAWS: tuple[int, ...] = tuple(range(201, 301))
#: The gate's record, written by the corpus driver only after the gate passes.
CORPUS_GATE_FORMAT_VERSION = "p7.3c-corpus-gate/1.0"
#: The corpus run's pre-token record.
CORPUS_PREFLIGHT_FORMAT_VERSION = "p7.3c-corpus-preflight/1.0"
_P7_3D_CALIBRATION_NAME = "p7_3d_calibration.json"


def _p7_3d_calibration_pin() -> str:
    """The calibration artifact's ONE pin, ``transfer_curve.P7_3D_CALIBRATION_SHA256`` -- imported, never retyped.

    Imported at call time: ``offline.transfer_curve`` imports this module.
    """
    from offline.transfer_curve import P7_3D_CALIBRATION_SHA256

    return P7_3D_CALIBRATION_SHA256


def _pinned_json(path: Path, pin: str) -> dict[str, Any]:
    """A committed artifact, digest-checked BEFORE it is parsed."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} is absent; it is a committed artifact under docs/data/")
    digest = _sha256_file(path)
    if digest != pin:
        raise ValueError(
            f"{path}: sha256 {digest} is not the pinned {pin}. The corpus gate reads registered values from "
            "THAT file and no other; an edited copy could change a reference without leaving a trace"
        )
    return json.loads(path.read_bytes())


def registered_engine_seed_drawn(data_dir: str | Path = _REPO_ROOT / "docs" / "data") -> int:
    """The engine seed every committed zero-shot cell records -- the value each corpus episode must record.

    Amendment A (Q14): read from ``p7_3d_grid4x4.json`` at :data:`P7_3D_ZERO_SHOT_ARTIFACT_SHA256`.  A18(c)'s
    rule makes it one number (the env RNG's first draw under ``reset(seed=1000)``), so the 700 cells must agree
    on ONE value and on the requested 1000, or the artifact is not the one A24 read.
    """
    artifact = _pinned_json(Path(data_dir) / P7_3D_ZERO_SHOT_ARTIFACT_NAME, P7_3D_ZERO_SHOT_ARTIFACT_SHA256)
    drawn = sorted({int(cell["engine_seed_drawn"]) for cell in artifact["cells"]})
    requested = sorted({int(cell["engine_seed_requested"]) for cell in artifact["cells"]})
    if len(drawn) != 1 or requested != [DEFAULT_ENGINE_SEED]:
        raise ValueError(
            f"{P7_3D_ZERO_SHOT_ARTIFACT_NAME}: its cells record engine seeds drawn {drawn[:4]} for requested "
            f"{requested[:4]}; A18(c) makes that ONE seed drawn from requested {DEFAULT_ENGINE_SEED}"
        )
    return drawn[0]


def logged_corpus_argv(
    scenario_key: str,
    draw_ids: Sequence[int],
    *,
    out_dir: str | Path,
    draws_root: str | Path,
    engine_seed: int = DEFAULT_ENGINE_SEED,
) -> list[str]:
    """``offline.collect``'s argv for A17(f)'s corpus: the PROBE replayed through the logger.

    The env settings are :data:`offline.transfer_gate.COLLECT_SETTINGS` -- the object the probe's own
    ``collect_style_args`` parses (``offline/sumo_att_reference.py:1232-1270``) -- appended whole, so the corpus
    runs the probe's env by construction rather than by a retyped list.  One episode per draw at
    ``--base-seed`` 1000: episode 0 of every draw gets ``reset(seed=1000)`` (A18(c)), and the flow randomiser
    behind ``flow_draw_sha256`` uses the draws tree's own base seed.  The sim config is the CityFlow one, whose
    stem is the scenario key the corpus is logged under (A24(b)).  ``--overwrite`` is never passed: an
    existing corpus is refused by the logger, never replaced.
    """
    from offline.transfer_gate import COLLECT_SETTINGS

    config = _REPO_ROOT / "configs" / "sim" / f"{scenario_key}.json"
    if not config.is_file():
        raise FileNotFoundError(
            f"{config} is absent: the scenario key IS the CityFlow sim config's stem, and the corpus is "
            "logged under it"
        )
    draws = [int(draw) for draw in draw_ids]
    if not draws:
        raise ValueError("no draws: a corpus is one episode per declared draw, and none was declared")
    return [
        "--backend", "sumo",
        "--env-config", str(config),
        "--policy", "maxpressure",
        "--flow-draws", *[str(draw) for draw in draws],
        "--episodes", "1",
        "--base-seed", str(int(engine_seed)),
        "--out-dir", str(out_dir),
        "--draws-root", str(draws_root),
        *COLLECT_SETTINGS,
    ]


def collect_logged_probe_corpus(
    scenario_key: str,
    draw_ids: Sequence[int],
    *,
    out_dir: str | Path,
    draws_root: str | Path,
    engine_seed: int = DEFAULT_ENGINE_SEED,
) -> list[str]:
    """Collect A17(f)'s corpus through ``offline.collect.main`` -- the door, not a copy of it.

    ``collect.main`` builds each draw's OBSERVED env, logs the aligned view, counts teleports and collisions
    on every simulated second, and refuses an episode with either before writing it (``BRIEF_41`` C1).
    Returns the argv, for the caller's provenance; a non-zero exit raises.
    """
    from offline import collect

    argv = logged_corpus_argv(
        scenario_key, draw_ids, out_dir=out_dir, draws_root=draws_root, engine_seed=engine_seed
    )
    code = collect.main(argv)
    if code != 0:
        raise RuntimeError(f"offline.collect exited {code} on the corpus argv; nothing was verified")
    return argv


def grid4x4_corpus_disjointness_record(
    draw_ids: Sequence[int],
    *,
    output_root: str | Path,
    expected_sha256: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """``BRIEF_41`` C1(v): the corpus band against the draws the subject TRAINED on and the held-out pool.

    The training draws are read from A20(a)'s five checkpoints (``stats.draw_ids``, through
    :func:`subject_facts_per_intersection`, which digest-checks every file first) -- the set the corpus of the
    subject actually used, never the registered 1-999 pool.  Run by the corpus driver before its token, i.e.
    before the first episode, and again, from the calibration artifact's own record, by the gate.
    """
    from offline.rtg_calibration import assert_probe_draws_disjoint

    requested = sorted({int(draw) for draw in draw_ids})
    if not requested:
        raise ValueError("the corpus draw set is empty; there is nothing to check")
    facts = subject_facts_per_intersection(output_root=output_root, expected_sha256=expected_sha256)
    assert_probe_draws_disjoint(
        requested, training_draw_ids=facts.training_draw_ids, held_out_draws=HELD_OUT_DRAWS
    )
    training = sorted(int(draw) for draw in facts.training_draw_ids)
    return {
        "disjoint": True,
        "checked_by": "offline.rtg_calibration.assert_probe_draws_disjoint",
        "subject": facts.subject,
        "checkpoint_sha256": {str(seed): digest for seed, digest in sorted(facts.checkpoint_sha256.items())},
        "probe_draws": [requested[0], requested[-1]],
        "n_probe_draws": len(requested),
        "training_draw_ids": [training[0], training[-1]],
        "n_training_draws": len(training),
        "training_draws_source": "stats.draw_ids of A20(a)'s five checkpoints, digest-checked",
        "held_out_draws": [min(HELD_OUT_DRAWS), max(HELD_OUT_DRAWS)],
    }


def corpus_preflight_record(
    draw_ids: Sequence[int],
    *,
    output_root: str | Path,
    data_dir: str | Path = _REPO_ROOT / "docs" / "data",
) -> dict[str, Any]:
    """Every pre-token check of the corpus run that needs Python, in one record (format
    ``p7.3c-corpus-preflight/1.0``).

    In order: the calibration artifact and the zero-shot artifact at their pins (the gate's two references);
    grid4x4's alignment resolved -- which locates RESCO's network and verifies its pinned digests -- and its
    ids equal to the artifact's; the band disjoint from the subject's training draws and the held-out pool
    (:func:`grid4x4_corpus_disjointness_record`); and every draw of the band carrying a recorded probe return.
    Nothing is written here; the driver decides whether the record is kept.
    """
    from offline.aligned_env import alignment_for_scenario_key

    data = Path(data_dir)
    calibration = _pinned_json(data / _P7_3D_CALIBRATION_NAME, _p7_3d_calibration_pin())
    engine_seed_drawn = registered_engine_seed_drawn(data)
    scenario_key = str(calibration["scenario_key"])
    ids = [str(ix) for ix in calibration["intersection_ids"]]
    alignment = alignment_for_scenario_key(scenario_key)
    if sorted(str(ix) for ix in alignment.intersections) != sorted(ids):
        raise ValueError(
            f"{scenario_key}'s alignment covers {sorted(alignment.intersections)[:4]}..., not the artifact's "
            f"{sorted(ids)[:4]}...; the corpus would be logged under ids the gate cannot key"
        )
    disjointness = grid4x4_corpus_disjointness_record(draw_ids, output_root=output_root)
    probe = calibration["probe_returns"]["sumo"]
    unrecorded = sorted(int(draw) for draw in draw_ids if str(int(draw)) not in probe)
    if unrecorded:
        raise ValueError(
            f"{_P7_3D_CALIBRATION_NAME} records no SUMO probe return for draw(s) {unrecorded[:5]}; A17(f) "
            "compares the corpus with the probe, so the corpus band must be the probe band"
        )
    return {
        "format_version": CORPUS_PREFLIGHT_FORMAT_VERSION,
        "registered_in": "PREREGISTRATION A17(f), A24(b); BRIEF_41 C1(v), C2, Amendment A (Q14, Q21)",
        "scenario_key": scenario_key,
        "inputs": {
            "calibration_artifact": _P7_3D_CALIBRATION_NAME,
            "calibration_sha256": _p7_3d_calibration_pin(),
            "zero_shot_artifact": P7_3D_ZERO_SHOT_ARTIFACT_NAME,
            "zero_shot_artifact_sha256": P7_3D_ZERO_SHOT_ARTIFACT_SHA256,
            "alignment_scenario": str(alignment.scenario),
            "n_intersections": len(ids),
        },
        "engine_seed_requested": DEFAULT_ENGINE_SEED,
        "engine_seed_drawn": engine_seed_drawn,
        "disjointness": disjointness,
    }


def assert_logged_corpus_matches_probe_per_intersection(
    corpus_dir: str | Path,
    *,
    data_dir: str | Path = _REPO_ROOT / "docs" / "data",
    draw_ids: Sequence[int] = GRID4X4_CORPUS_DRAWS,
) -> dict[str, Any]:
    """A17(f) on a multi-intersection pair: every logged episode reproduces P7.3d's probe, per intersection.

    ``PREREGISTRATION`` A24(b) applied to grid4x4 (``BRIEF_41`` C1(iv)): every episode's per-intersection
    return must equal ``p7_3d_calibration.json:probe_returns.sumo[draw][ix]`` BIT FOR BIT, on every requested
    draw and all sixteen intersections, and every episode must record zero teleports and zero collisions
    counted on EVERY simulated second.  **Any mismatch or event refuses the corpus and P7.3c stops before any
    training runs.**  Returns the record the driver writes (format ``p7.3c-corpus-gate/1.0``).

    THE RETURN, AND ITS ALIGNMENT CONVENTION (C6)
    ---------------------------------------------
    Intersection *i*'s return is the sum of its ``T`` outcome rows, ``ix{k}_local_reward`` -- the reward read
    from the ``info`` returned by each step -- which is the probe's ``episode_return_two_routes`` route 1 over
    the same post-step infos, the same ``float(payload["reward"])`` of the same field.  The logger stores
    float32 and the probe summed float64, so ``==`` is achievable only because this reward family is a
    vehicle count: integrality is checked FIRST, per intersection, and a non-integral reward is a finding,
    never a tolerance (the hz1x1 gate's argument, :func:`assert_logged_corpus_matches_probe`, unchanged).

    KEYED BY ID, NEVER BY POSITION
    ------------------------------
    The logger names its arrays by POSITION in the env's order (``ix{k}_*``) and records that order in the
    episode's own ``ix_ids``.  Each intersection is located through THAT array, in THAT episode, so a corpus
    logged in any order compares each id with its own probe return.

    THE REFUSALS, IN ORDER, each naming what it found
    ------------------------------------------------
    the two committed artifacts at their pins, before either is parsed; the manifest in the logger's
    format and under the artifact's scenario key; the draws disjoint from the subject's training draws and
    the held-out pool (re-asserted from the calibration artifact's own record, its ``[1, 200]`` read as the
    whole range -- which can only refuse more); one episode per draw, listed in the manifest, exactly the
    requested set; each draw's door record -- counted every simulated second, zero teleports and collisions,
    ``cf_parity`` only, ``time-to-teleport -1``, and the engine seed every committed zero-shot cell records;
    per episode the registered horizon, the artifact's id set and seed 1000; then integrality and ``==``.
    """
    import numpy as np

    from offline.collect import ENGINE_EVENT_GRAIN
    from offline.rtg_calibration import assert_probe_draws_disjoint
    from offline.trajectory_logger import FORMAT_VERSION as LOGGED_FORMAT_VERSION

    corpus = Path(corpus_dir)
    data = Path(data_dir)
    requested = sorted({int(draw) for draw in draw_ids})
    if not requested:
        raise ValueError("no draws were requested, so nothing would be compared")

    # 1. The two committed references, each at its pin BEFORE it is parsed.
    calibration = _pinned_json(data / _P7_3D_CALIBRATION_NAME, _p7_3d_calibration_pin())
    engine_seed_drawn = registered_engine_seed_drawn(data)
    ids = [str(ix) for ix in calibration["intersection_ids"]]
    scenario_key = str(calibration["scenario_key"])
    probe = calibration["probe_returns"]["sumo"]

    # 2. The manifest: the logger's format, logged under the CityFlow scenario key (A24(b)).
    manifest = json.loads((corpus / "manifest.json").read_bytes())
    if str(manifest.get("format_version")) != LOGGED_FORMAT_VERSION:
        raise ValueError(
            f"{corpus}: manifest format {manifest.get('format_version')!r}, not the logger's "
            f"{LOGGED_FORMAT_VERSION!r}"
        )
    metadata = manifest.get("run_metadata") or {}
    if str(metadata.get("scenario_id")) != scenario_key:
        raise ValueError(
            f"{corpus}: scenario_id {metadata.get('scenario_id')!r}, not {scenario_key!r}. A24(b) logs the corpus "
            "under the CityFlow scenario key, the key the subject's statistics are stored under"
        )

    # 3. Disjointness, re-asserted from the pinned artifact's own record.
    record = calibration["disjointness"]
    low, high = (int(value) for value in record["training_draw_ids"])
    held_out = [int(value) for value in record["held_out_draw_ids"]]
    if held_out != [min(HELD_OUT_DRAWS), max(HELD_OUT_DRAWS)]:
        raise ValueError(
            f"{_P7_3D_CALIBRATION_NAME} records the held-out pool as {held_out}, not "
            f"{[min(HELD_OUT_DRAWS), max(HELD_OUT_DRAWS)]}"
        )
    assert_probe_draws_disjoint(
        requested, training_draw_ids=range(low, high + 1), held_out_draws=HELD_OUT_DRAWS
    )

    # 4. The episodes: listed in the manifest, one per draw, exactly the requested set.
    listed = {str(entry["filename"]): int(entry["flow_draw"]) for entry in manifest.get("episodes", [])}
    on_disk = sorted(path.name for path in corpus.glob("*.npz"))
    if sorted(listed) != on_disk:
        raise ValueError(
            f"{corpus}: the manifest lists {len(listed)} episode(s) and {len(on_disk)} are on disk "
            f"(unlisted {sorted(set(on_disk) - set(listed))[:3]}, absent {sorted(set(listed) - set(on_disk))[:3]})"
        )
    episodes: dict[int, Path] = {}
    for name in on_disk:
        with np.load(corpus / name) as payload:
            draw = int(payload["flow_draw"])
        if draw != listed[name]:
            raise ValueError(f"{name}: the episode's flow_draw {draw} disagrees with the manifest's {listed[name]}")
        if draw in episodes:
            raise ValueError(
                f"draw {draw} has more than one logged episode in {corpus}; A17(f) compares one episode per draw"
            )
        episodes[draw] = corpus / name
    missing = [draw for draw in requested if draw not in episodes]
    if missing:
        raise ValueError(f"the corpus has no logged episode for draw(s) {missing[:10]}")
    extra = sorted(set(episodes) - set(requested))
    if extra:
        raise ValueError(
            f"{corpus} holds draw(s) {extra[:10]} that were not requested; a corpus is exactly its declared draws"
        )
    unrecorded = [draw for draw in requested if str(draw) not in probe]
    if unrecorded:
        raise ValueError(
            f"{_P7_3D_CALIBRATION_NAME} records no SUMO probe return for draw(s) {unrecorded[:10]}, so there "
            "is nothing to compare them against"
        )

    # 5. Each draw's door record: dense counts, zero events, the regime, the one engine seed.
    door = {int(entry["draw_id"]): entry for entry in metadata.get("sumo_draws") or []}
    n_teleports = n_collisions = 0
    for draw in requested:
        entry = door.get(draw)
        if entry is None:
            raise ValueError(f"draw {draw}: the manifest carries no sumo_draws record for it")
        if (
            "n_teleports" not in entry
            or "n_collisions" not in entry
            or entry.get("engine_events_counted") != ENGINE_EVENT_GRAIN
        ):
            raise ValueError(
                f"draw {draw}: the corpus records no engine-event count taken on every simulated second "
                f"({ENGINE_EVENT_GRAIN}); it was collected through a door without DEFERRED 93's counter"
            )
        if int(entry["n_teleports"]) or int(entry["n_collisions"]):
            raise ValueError(
                f"draw {draw}: {int(entry['n_teleports'])} teleport(s) and {int(entry['n_collisions'])} "
                "collision(s) counted every simulated second; A24(b) requires zero of each"
            )
        types = list(entry.get("vehicle_types_seen") or [])
        if types != [PARITY_VTYPE_ID]:
            raise ValueError(f"draw {draw}: vehicle types {types!r}, not {[PARITY_VTYPE_ID]!r}")
        option = str(entry.get("time_to_teleport_option"))
        if option != EXPECTED_TIME_TO_TELEPORT:
            raise ValueError(f"draw {draw}: time-to-teleport {option!r}, not {EXPECTED_TIME_TO_TELEPORT!r}")
        if int(entry.get("engine_seed_requested", -1)) != DEFAULT_ENGINE_SEED:
            raise ValueError(
                f"draw {draw}: engine_seed_requested {entry.get('engine_seed_requested')!r}, not A18(c)'s "
                f"{DEFAULT_ENGINE_SEED}"
            )
        if int(entry.get("engine_seed_drawn", -1)) != engine_seed_drawn:
            raise ValueError(
                f"draw {draw}: engine_seed_drawn {entry.get('engine_seed_drawn')!r}, not {engine_seed_drawn} -- "
                f"the seed all of {P7_3D_ZERO_SHOT_ARTIFACT_NAME}'s cells record (Amendment A, Q14)"
            )
        n_teleports += int(entry["n_teleports"])
        n_collisions += int(entry["n_collisions"])

    # 6. Per episode, per intersection BY ID: integrality first, then ==.
    rows: list[dict[str, Any]] = []
    mismatches: list[tuple[int, str, float, float]] = []
    for draw in requested:
        returns: dict[str, float] = {}
        with np.load(episodes[draw]) as payload:
            decisions = int(payload["episode_length"])
            if decisions != EXPECTED_DECISIONS:
                raise ValueError(
                    f"draw {draw}: {decisions} decision(s), not {EXPECTED_DECISIONS}; the episode did not run to "
                    "the registered horizon"
                )
            logged_ids = [str(value) for value in payload["ix_ids"].tolist()]
            if sorted(logged_ids) != sorted(ids):
                raise ValueError(
                    f"draw {draw}: the episode's intersections differ from the artifact's (missing "
                    f"{sorted(set(ids) - set(logged_ids))}, unknown {sorted(set(logged_ids) - set(ids))}); "
                    "A24(b) logs the corpus under the CityFlow intersection ids"
                )
            if int(payload["engine_seed"]) != DEFAULT_ENGINE_SEED:
                raise ValueError(
                    f"draw {draw}: the episode was reset with seed {int(payload['engine_seed'])}, not "
                    f"{DEFAULT_ENGINE_SEED} (A18(c))"
                )
            for ix in ids:
                position = logged_ids.index(ix)
                rewards = np.asarray(payload[f"ix{position}_local_reward"])
                off = rewards != np.rint(rewards)
                if bool(off.any()):
                    raise ValueError(
                        f"draw {draw}, intersection {ix!r}: {int(off.sum())} of {rewards.size} stored per-step "
                        f"rewards are NOT integral (first: {float(rewards[off][0])!r}). The corpus stores float32 "
                        "and the probe summed float64; bit-for-bit equality holds only because this reward is a "
                        "vehicle count -- a finding about the reward, never a tolerance to widen"
                    )
                returns[ix] = float(np.sum(rewards.astype(np.float64)))
        matching = 0
        for ix in ids:
            expected = float(probe[str(draw)][ix])
            if returns[ix] == expected:
                matching += 1
            else:
                mismatches.append((draw, ix, returns[ix], expected))
        rows.append(
            {
                "draw_id": draw,
                "episode": episodes[draw].name,
                "n_intersections": len(ids),
                "n_matching": matching,
                "all_match": matching == len(ids),
            }
        )
    if mismatches:
        draw, ix, logged, expected = mismatches[0]
        raise ValueError(
            f"A17(f) FAILED on {len(mismatches)} of {len(requested) * len(ids)} (draw, intersection) pair(s); "
            f"the first is draw {draw}, intersection {ix!r}: logged return {logged!r} against the probe's "
            f"{expected!r} (difference {logged - expected!r}). SUMO is deterministic under a fixed seed, so this "
            "is a wiring defect, and P7.3c stops here, before any training (A24(b))"
        )
    n_checked = len(requested) * len(ids)
    return {
        "format_version": CORPUS_GATE_FORMAT_VERSION,
        "registered_in": "PREREGISTRATION A17(f), A24(b); BRIEF_41 C1(iv), Amendment A (Q14)",
        "scenario_key": scenario_key,
        "corpus_dir": str(corpus),
        "calibration_artifact": _P7_3D_CALIBRATION_NAME,
        "calibration_sha256": _p7_3d_calibration_pin(),
        "zero_shot_artifact": P7_3D_ZERO_SHOT_ARTIFACT_NAME,
        "zero_shot_artifact_sha256": P7_3D_ZERO_SHOT_ARTIFACT_SHA256,
        "engine_seed_requested": DEFAULT_ENGINE_SEED,
        "engine_seed_drawn": engine_seed_drawn,
        "engine_events": {"grain": ENGINE_EVENT_GRAIN, "n_teleports": n_teleports, "n_collisions": n_collisions},
        "disjointness": {
            "training_draw_ids": [low, high],
            "held_out_draw_ids": held_out,
            "source": f"{_P7_3D_CALIBRATION_NAME}:disjointness, [low, high] read as the whole range",
        },
        "n_draws": len(requested),
        "n_intersections": len(ids),
        "n_checked": n_checked,
        "n_matching": n_checked,
        "all_match": True,
        "rows": rows,
    }


def format_canary_line(seconds: float, facts: Mapping[str, Any]) -> str:
    """``canary <seconds> s <JSON facts>`` -- the one line the driver captures and parses back.

    The facts are **JSON with sorted keys**, not a Python ``repr``.  Runs 1-3 printed a ``repr``,
    which is why their observed values could only ever be read by a human: ``{'decisions': 360}``
    is not JSON, so nothing could parse it into the artifact, and the artifact fell back on the
    probe chunks (Amendment E1.4).  Every value here is a plain Python ``float``/``int``/``bool``
    -- :func:`offline.rtg_calibration.episode_return_two_routes` appends ``float(...)`` and
    ``run_probe`` sets ``att_horizon`` from a ``float(...)`` sample -- so ``json.dumps`` cannot
    fail on them.
    """
    return (
        f"{_CANARY_LINE_PREFIX}{seconds:.2f}{_CANARY_LINE_SEPARATOR}"
        f"{json.dumps(dict(facts), sort_keys=True)}"
    )


def parse_canary_line(line: str) -> tuple[float, dict[str, Any]]:
    """Inverse of :func:`format_canary_line`; raise ``ValueError`` on anything else.

    Refused rather than salvaged, in every case: this line is the only route by which the engine's
    observed answers reach ``canary.json`` and from there the artifact, so a line that does not
    parse must stop the campaign rather than produce a record with a hole in it.  A missing fact
    name in particular: :func:`check_canary` reads an absent name as ``None`` and would then refuse
    for the wrong reason, reporting a wrong value where the truth is a missing one.
    """
    text = line.strip()
    if "\n" in text:
        raise ValueError(f"canary line is more than one line: {line!r}")
    if not text.startswith(_CANARY_LINE_PREFIX):
        raise ValueError(f"canary line does not start with {_CANARY_LINE_PREFIX!r}: {line!r}")
    head, separator, payload = text.partition(_CANARY_LINE_SEPARATOR)
    if not separator:
        raise ValueError(f"canary line has no {_CANARY_LINE_SEPARATOR!r} separator: {line!r}")
    try:
        seconds = float(head[len(_CANARY_LINE_PREFIX) :])
    except ValueError:
        raise ValueError(f"canary line carries no parsable seconds: {line!r}") from None
    try:
        facts = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"canary line's facts are not JSON: {payload!r}") from exc
    if not isinstance(facts, dict):
        raise ValueError(
            f"canary line's facts are a {type(facts).__name__}, not a JSON object: {payload!r}"
        )
    missing = [name for name in CANARY_FACT_NAMES if name not in facts]
    if missing:
        raise ValueError(f"canary line is missing the facts {missing}: {payload!r}")
    return seconds, facts


def record_canary(line: str, *, work_dir: str | Path) -> Path:
    """Park the driver's canary line in ``<work_dir>/canary.json``; return the path written.

    Amendment E1.4 item 1.  The driver runs this immediately after the token is consumed, so the
    observed values reach a file the manifest hashes -- not only a pane that is gone by the time
    anyone reads the artifact.

    **It deliberately does not call :func:`check_canary`.**  Its job is to preserve what the engine
    answered, including an answer that fails the check: that is the whole of Amendment E1.2's
    lesson, and :func:`report` is the checker (item 2).  In practice the driver cannot reach here
    with a failing canary anyway -- the canary stage exits non-zero and ``set -euo pipefail`` aborts
    the pipeline before the token.

    The line is parsed before the path is touched, so a refused line creates nothing, not even the
    work directory (the same barrier Amendment B1 put in front of the token).
    """
    seconds, facts = parse_canary_line(line)
    path = Path(work_dir) / CANARY_RECORD_NAME
    _write_json(
        path,
        {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "line": line,
            "seconds": seconds,
            "facts": facts,
            "threshold_seconds": CANARY_MAX_SECONDS,
            **_git_provenance(),
        },
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Run one stage; returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        if args.stage == "canary":
            elapsed, facts = canary_seconds()
            # PRINT FIRST, THEN CHECK: on a mismatch the observed values must be visible in the
            # pane and in canary.log, or the refusal says only that something was wrong.
            print(format_canary_line(elapsed, facts), flush=True)
            check_canary(facts)
            return 0
        if args.stage == "record-canary":
            written = record_canary(args.line, work_dir=args.work_dir)
            print(f"wrote {written}", flush=True)
            return 0
        if args.stage == "probe":
            start, end = args.draws_range or (PROBE_DRAW_START_DEFAULT, PROBE_DRAW_END_DEFAULT)
            records = run_sumo_probe(
                range(start, end),
                out_root=args.draws_root,
                work_dir=args.work_dir,
                engine_seed=args.engine_seed,
                canary_seconds=args.canary_seconds,
            )
            print(f"probe: {len(records)} draws, all two-route equal, 0 teleports", flush=True)
            return 0
        if args.stage == "smoke":
            payload = run_smoke(
                args.subject,
                out_root=args.draws_root,
                work_dir=args.work_dir,
                output_root=args.output_root,
                draw_id=args.draw_id,
            )
            _write_json(Path(args.work_dir) / f"smoke_{args.subject}.json", payload)
            print(
                f"smoke {args.subject}: decisions={payload['decisions']} "
                f"actions_in_range={payload['actions_in_range']} "
                f"rtg_advanced={payload['rtg_advanced_every_decision']} "
                f"in_support={payload['n_decisions_in_support']} (outcome fenced)",
                flush=True,
            )
            return 0
        if args.stage == "report":
            out_path = Path(args.out_dir) / "p7_2b_calibration.json"
            artifact = report(
                work_dir=args.work_dir, out_path=out_path, output_root=args.output_root
            )
            print(f"wrote {out_path}: {len(artifact['probe'])} probe rows", flush=True)
            return 0
        if args.stage == "corpus-preflight":
            checked = corpus_preflight_record(
                _draws_range(args.draws_range), output_root=args.output_root, data_dir=args.data_dir
            )
            disjoint = checked["disjointness"]
            inputs = checked["inputs"]
            print(
                f"corpus_preflight PASSED: draws {disjoint['probe_draws'][0]}-{disjoint['probe_draws'][1]} "
                f"({disjoint['n_probe_draws']}) disjoint from the subject's {disjoint['n_training_draws']} "
                f"training draws and the held-out pool; calibration {inputs['calibration_sha256'][:8]}, "
                f"zero-shot artifact {inputs['zero_shot_artifact_sha256'][:8]}, "
                f"{inputs['n_intersections']} intersections aligned; engine_seed_drawn "
                f"{checked['engine_seed_drawn']}",
                flush=True,
            )
            return 0
        if args.stage == "collect-corpus":
            draws = _draws_range(args.draws_range)
            collect_logged_probe_corpus(
                args.scenario_key,
                draws,
                out_dir=args.corpus_dir,
                draws_root=args.draws_root,
                engine_seed=args.engine_seed,
            )
            print(
                f"collect_corpus DONE: draws {draws[0]}-{draws[-1]} logged into {args.corpus_dir}",
                flush=True,
            )
            return 0
        if args.stage == "corpus-gate":
            verdict = assert_logged_corpus_matches_probe_per_intersection(
                args.corpus_dir, data_dir=args.data_dir, draw_ids=_draws_range(args.draws_range)
            )
            # The record is a WRITE, so it follows every refusal: a refused corpus leaves no record.
            _write_json(args.record, verdict)
            events = verdict["engine_events"]
            print(
                f"A17(f) {verdict['n_matching']}/{verdict['n_checked']} MATCH on {verdict['n_draws']} draw(s) "
                f"x {verdict['n_intersections']} intersections; {events['n_teleports']} teleport(s) and "
                f"{events['n_collisions']} collision(s) counted every simulated second",
                flush=True,
            )
            return 0
    except (ValueError, FileNotFoundError, AssertionError) as exc:
        print(f"transfer_calibration: {exc}", flush=True)
        return 1
    return 2


#: The registered band, as CLI defaults.
PROBE_DRAW_START_DEFAULT = 201
PROBE_DRAW_END_DEFAULT = 301


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
