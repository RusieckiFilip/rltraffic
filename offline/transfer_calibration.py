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
from typing import Any, Mapping, Sequence

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
        # Named `policy`, not `agent`: MaxPressureAgent.act takes no `explore` keyword, so a
        # bare call here is correct -- and the name keeps it distinguishable from the DT's
        # call in run_smoke, which must never be bare (Amendment E1).
        policy = MaxPressureAgent(env)

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
                # the first step on which anybody is present, so an early wrong type cannot
                # hide behind a horizon-only read
                types_seen.update(
                    env._sumo.vehicle.getTypeID(v) for v in env._sumo.vehicle.getIDList()
                )
            action = policy.act(info)
            _reward, terminated, truncated, info = env.step(action)
            post_step.append(info)
            samples.append(float(info.get("average_travel_time", 0.0)))
            last_vehicle_count = float(info.get("vehicle_count", 0.0))
            if terminated or truncated:
                break
        teleports += len(env._sumo.simulation.getStartingTeleportIDList())
        types_seen.update(
            env._sumo.vehicle.getTypeID(v) for v in env._sumo.vehicle.getIDList()
        )
        engine_seed_drawn = int(env._engine_seed)
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
    return parser


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
    except (ValueError, FileNotFoundError, AssertionError) as exc:
        print(f"transfer_calibration: {exc}", flush=True)
        return 1
    return 2


#: The registered band, as CLI defaults.
PROBE_DRAW_START_DEFAULT = 201
PROBE_DRAW_END_DEFAULT = 301


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
