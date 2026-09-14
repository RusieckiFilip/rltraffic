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
    "build_parser",
    "chunk_is_reusable",
    "disjointness_record",
    "in_support_position",
    "main",
    "probe_chunk_path",
    "probe_returns_from_chunks",
    "report",
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
    """
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
    ``reset(seed=engine_seed)``, ``agent.act(info)`` -> ``env.step``, post-step infos accumulated,
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
    from algorithms.max_pressure import MaxPressureAgent
    from experiments.envs import make_env
    from offline.collect import _build_env_spec
    from offline.materialise_draws import parity_sumocfg_path
    from offline.rtg_calibration import episode_return_two_routes
    from offline.sumo_att_reference import collect_style_args

    import time

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
            agent = MaxPressureAgent(env)

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
                action = agent.act(info)
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
            action = agent.act(info)
            actions.append(int(np.asarray(action).reshape(-1)[0]))
            return action

        rollout = horizon_rollout(env, choose, 1, int(seed))
        types_seen = sorted(
            {env._sumo.vehicle.getTypeID(v) for v in env._sumo.vehicle.getIDList()}
        )
    finally:
        env.close()
    seconds = time.perf_counter() - started

    # The RTG advanced exactly where the reward was non-zero: rtg[t] - rtg[t-1] == -r_{t-1}, and
    # r_{t-1} is the reward carried by the info handed to decision t.
    advanced_correctly = True
    for index in range(1, len(rtg_series)):
        reward = rewards_in_info[index]
        changed = rtg_series[index] != rtg_series[index - 1]
        if reward is None or bool(reward != 0.0) != changed:
            advanced_correctly = False
            break

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


def report(*, work_dir: str | Path, out_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Build the committed artifact from the chunks. Every refusal precedes every write."""
    work = Path(work_dir)
    target_path = Path(out_path)

    on_disk = _sha256_file(P4_3_PROBE_ARTIFACT)
    if on_disk != P4_3_PROBE_SHA256:
        raise ValueError(
            f"{P4_3_PROBE_ARTIFACT.name}: sha256 {on_disk} against the pinned {P4_3_PROBE_SHA256}"
        )

    chunks: list[Mapping[str, Any]] = []
    for path in sorted(work.glob("probe_draw_*.json")):
        payload = json.loads(path.read_bytes())
        _validate_chunk(payload, path)
        chunks.append(payload)
    if not chunks:
        raise ValueError(f"no probe chunks under {work}")

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

    artifact: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "registered_in": "PREREGISTRATION A17",
        "scenario_key": SCENARIO_KEY,
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run one stage; returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        if args.stage == "canary":
            elapsed, facts = canary_seconds()
            print(f"canary {elapsed:.2f} s {facts}", flush=True)
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
