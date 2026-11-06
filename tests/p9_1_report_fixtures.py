"""A synthetic P9.1 run tree at the campaign's full size, for the report's end-to-end T-regress (``BRIEF_46`` G4; plan
C6: "T-regress for each artifact (rebuilt byte for byte from fixture run directories)").

No format of its own: every file is written in the format the campaign writes it -- ``p9.1-run/1.0`` run records,
route (i)'s cell payloads (``episodes``), P8.4b's per-episode files (``att_rederivation.cell_file_name``), the
``p9.1-actions/1.0`` P3a records, the driver's Unix-time GPU trace, the driver's captures with one token line per
session (``BRIEF_46`` G.2), each IQL run's ``diagnostics`` block and the D4RL re-run's (Q18, F.2-3) -- and the three
declarations are written by the campaign's own ``declare_*`` functions on the real corpus.  Alignment: every episode is a held-out ``(seed, draw)`` of
``dt_gate``'s protocol (5 seeds x 100 draws).  A NEW cell's values are synthetic and exact in binary (a level that is a
multiple of 1/4 plus a per-episode term that is a multiple of 1/64); an OLD cell's ``att_ours`` is its committed value
(its re-roll must reproduce it) and its ``att_engine`` that value minus :data:`ENGINE_SHIFT_OLD`.

The output root links every entry of the main tree's ``output/`` except ``p9_1`` and ``p9_1_runs`` -- the registered
grid cells, P8.4b's campaign, the P5 checkpoints, read and never written -- beside the synthetic ``p9_1`` and
``p9_1_runs``.  Two features are planted for the tests to find: one gap in the GPU trace inside one run's bracket
(:data:`PAUSED_RUN`), and one repeated action sequence in one P3a record (:data:`REPEATED`).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import offline.corrective_report as rep
import offline.corrective_round as cr
from offline import att_rederivation as ar
from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS

#: The first run's start (Unix seconds); runs follow ``training_runs()``'s order, each ``RUN_SECONDS + index`` long.
T0 = 1_800_000_000.0
RUN_SECONDS = 600.0
RUN_SPACING = 1_000.0
#: The run whose bracket spans the trace's one gap, and the gap's place inside it (seconds after its start, length).
PAUSED_RUN = ("hz1x1", "mix50", "iql_t", 202)
GAP_OFFSET, GAP_LENGTH = 100.0, 300.0
#: P3a: on this cell and seed, these two draws share one joint action sequence (the only repeat in the tree).
REPEATED = ("hz1x1", "dt@random_fresh", 202, (1000, 1001))
ENGINE_SHIFT_NEW = 3.0
ENGINE_SHIFT_OLD = 7.0
TRACE_STEP = 20.0
#: The driver's sessions in the campaign's shape (BRIEF_46 G.2): (stage, the run number it trains first or None, the
#: commit and the run tree its token line names).  A training session starts SESSION_LEAD seconds before its first run;
#: collect, d4rl and evaluate train nothing.
SESSIONS: tuple[tuple[str, int | None, str, str], ...] = (
    ("collect", None, "140149e", "/trees/run"),
    ("campaign", 1, "cb61ee7", "/trees/run-c"),
    ("d4rl", None, "cb61ee7", "/trees/run-c"),
    *(("train", first, "c58afb8", "/trees/run-d") for first in (4, 66, 126, 155, 188, 212, 226, 236)),
    *(("evaluate", None, "c58afb8", "/trees/run-d") for _ in range(3)),
)
SESSION_LEAD = 50.0
#: What hz1x1's route (i) files carry in ``runtime.git_commit``: the main tree's HEAD when they were written (``BRIEF_46``
#: S.2), never the code's commit. The grid's route (i) files carry no ``runtime`` block, as the campaign's do not.
ROUTE_I_RUNTIME_COMMIT = "25410f4" + "0123456789abcdef0123456789abcdef0"


def iql_diagnostics_of(index: int) -> tuple[float, float, float]:
    """An IQL run's final Q loss, final V loss and AWR near-zero fraction, exact in binary, fixed by its run index."""
    return 2.0 + index / 4, 1.0 + index / 8, (index % 128) / 128


def d4rl_diagnostics_of(position: int) -> tuple[float, float, float]:
    """The fixture re-run's ``iql_t`` cell's three values, by its seed's position in ``TRAINING_SEEDS``."""
    return 80.0 + position / 4, 8.0 + position / 8, (64 + position) / 128


def session_starts() -> list[float]:
    """Each :data:`SESSIONS` entry's start, in whole Unix seconds: a training session SESSION_LEAD before its first
    run, d4rl between run 3's end and run 4's session, collect a day before the campaign, the evaluation after the
    last run, and each later evaluation 20,000 s after the one before."""
    starts = []
    for stage, first, _, _ in SESSIONS:
        if first is not None:
            starts.append(T0 + RUN_SPACING * (first - 1) - SESSION_LEAD)
        elif stage == "collect":
            starts.append(T0 - 86_400.0)
        elif stage == "d4rl":
            starts.append(T0 + RUN_SPACING * 2 + RUN_SECONDS + 100.0)
        else:
            starts.append(T0 + RUN_SPACING * 256 + 20_000.0 * sum(s == "evaluate" for s, _, _, _ in
                                                                    SESSIONS[:len(starts)]))
    return starts


def stamp_of(start: float) -> str:
    """The driver's stamp for a session starting at *start*: ``date -u +%Y%m%dT%H%M%SZ``."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(start))


def training_session_of(number: int) -> tuple[str, str, str, str]:
    """The session that trains run *number* (1-based), found by run number: (stage, stamp, commit, run tree)."""
    chosen = max(i for i, (_, first, _, _) in enumerate(SESSIONS) if first is not None and first <= number)
    stage, _, commit, tree = SESSIONS[chosen]
    return stage, stamp_of(session_starts()[chosen]), commit, tree


def level(scenario: str, key: str) -> float:
    """A new cell's level: 100 plus a multiple of 1/4 below 40, fixed by the cell's name."""
    digest = int(hashlib.sha256(f"{scenario}:{key}".encode()).hexdigest()[:8], 16)
    return 100.0 + (digest % 160) / 4.0


def term(key: str, seed: int, draw: int) -> float:
    """A per-episode term exact in binary, within +-14/64, different per family, seed and draw."""
    h = sum(map(ord, key))
    return ((h + 7 * seed + (13 + h % 7) * draw) % 29 - 14) / 64.0


def companions(seed: int | None, draw: int) -> dict[str, Any]:
    """The entry companions, functions of the episode alone, so the two re-rolls of one old episode agree."""
    s = -1 if seed is None else int(seed)
    created = 1200 + (s + draw) % 50
    entered = created - (3 * s + draw) % 7
    return {"created": created, "entered": entered, "never_entered": created - entered,
            "entered_fraction": entered / created}


def new_episode(scenario: str, key: str, seed: int, draw: int) -> dict[str, float]:
    """A new cell's synthetic episode: ``att_ours``, ``att_engine`` and route (i)'s two companions."""
    ours = level(scenario, key) + term(key, seed, draw)
    return {"att_ours": ours, "att_engine": ours - ENGINE_SHIFT_NEW, "episode_reward": -8.0 * ours,
            "horizon_vehicle_count": 1000.0 + seed % 7 + draw % 13}


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _p8_4b_file(scenario: str, arm: str, tier: str, method: str, seed: int | None, draw: int, att_ours: float,
                att_engine: float, *, episode_reward: float, horizon_vehicle_count: float) -> dict[str, Any]:
    """One episode in P8.4b's per-episode format (the keys ``att_rederivation.run_campaign`` writes)."""
    return {"format_version": ar.ARTIFACT_FORMAT_VERSION, "scenario": scenario, "arm": arm, "tier": tier,
            "method": method, "seed": seed, "draw_id": draw, "att_ours": att_ours, "att_engine": att_engine,
            "att_difference": att_engine - att_ours, "committed_att_ours": att_ours, "reproduces_committed": True,
            **companions(seed, draw), "episode_reward": episode_reward, "horizon_vehicle_count": horizon_vehicle_count,
            "completed_at_horizon": 900, "running_at_horizon": 40, "waiting_at_horizon": 20, "seconds": 1.0,
            "seconds_rollout": 0.9, "policy_source": {"kind": "fixture"}}


def _digest(*parts: Any) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()


@dataclass(frozen=True)
class FixtureTree:
    roots: cr.Roots
    d4rl_path: Path
    run_brackets: dict[tuple[str, str, str, int], tuple[float, float]]


def write_tree(root: Path, *, main_output: Path, corpus_root: Path, draws_root: Path, repo_root: Path) -> FixtureTree:
    """The tree under *root*: ``output/`` (links plus the synthetic ``p9_1`` and ``p9_1_runs``) and ``d4rl/``."""
    output = root / "output"
    output.mkdir(parents=True)
    for entry in sorted(main_output.iterdir()):
        if entry.name not in (cr.RUN_DIR, cr.RUNS_DIR):
            (output / entry.name).symlink_to(entry)
    (output / cr.RUN_DIR).mkdir()
    (output / cr.RUNS_DIR).mkdir()
    roots = cr.Roots(repo_root=repo_root, output_root=output, corpus_root=corpus_root, draws_root=draws_root)
    runs_root = Path(roots.runs_root)

    # The declarations, by the campaign's own functions on the real corpus (the collect stage's last three steps).
    cr.declare_mixture(roots)
    cr.declare_random_fresh(roots, "hz1x1")
    cr.declare_random_fresh(roots, "grid4x4")

    # Every training run's write-once record, in training_runs() order.
    brackets: dict[tuple[str, str, str, int], tuple[float, float]] = {}
    for index, (cell, seed) in enumerate(cr.training_runs()):
        started = T0 + RUN_SPACING * index
        seconds = RUN_SECONDS + index
        fields = cr.recipe_fields(cell)
        record = {"format_version": cr.RUN_RECORD_FORMAT_VERSION, "scenario": cell.scenario, "tier": cell.tier,
                  "arm": cell.arm, "seed": int(seed), **fields, "label": cr.LABELS[cell.point],
                  "gradient_steps": int(fields["declared_steps"]), "seconds": seconds, "clock": cr.CLOCK_LABEL,
                  "started_unix": started, "finished_unix": started + seconds,
                  "checkpoint": str(cr.checkpoint_path(roots, cell.scenario, cell.tier, cell.arm, int(seed))),
                  "canonical_digest": _digest("canonical", cell.key, seed), "file_sha256": _digest("file", cell.key, seed),
                  "window_means": [3.0, 2.0, 1.5, 1.25], "plateaued": index % 3 == 0, "final_loss": 1.25,
                  "diagnostics": None, "device": "cuda", "thread_regime": {"torch": 1}}
        if fields["base"] == "iql":
            q, v, near_zero = iql_diagnostics_of(index)
            record["diagnostics"] = {"final_q_loss": q, "final_v_loss": v, "awr_weight_near_zero_fraction": near_zero,
                                     "final_policy_loss": 1.25, "diagnostic_positions": 20000,
                                     "target_dropout": fields["target_dropout"]}
        if cell.scenario == "grid4x4" and cell.arm == "dt_nomix":
            record["deterministic"] = False
        _write(cr.run_record_path(roots, cell.scenario, cell.tier, cell.arm, int(seed)), record)
        brackets[(cell.scenario, cell.tier, cell.arm, int(seed))] = (started, started + seconds)

    # The driver's captures: one token line per session (BRIEF_46 G.2), each followed by a line of its stage's output.
    captures: dict[str, list[str]] = {}
    for (stage, _, commit, tree), start in zip(SESSIONS, session_starts()):
        stamp = stamp_of(start)
        captures.setdefault(f"{stage}_capture.txt", []).extend([
            f"token consumed; stage {stage} {stamp} from {tree} at {commit}; logs {runs_root / 'logs' / stamp}",
            f"stage {stage} {stamp}: SESSION COMPLETE (fixture)"])
    for name, lines in captures.items():
        (runs_root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # The driver's Unix-time GPU trace over the whole stage, with one gap inside PAUSED_RUN's bracket.
    first, last = min(b[0] for b in brackets.values()), max(b[1] for b in brackets.values())
    gap_start = brackets[PAUSED_RUN][0] + GAP_OFFSET
    samples, t = [], first - TRACE_STEP
    while t <= last + TRACE_STEP:
        if not gap_start < t < gap_start + GAP_LENGTH:
            samples.append(f"{t:.3f},1024,50")
        t += TRACE_STEP
    trace = runs_root / "logs" / "20261028T000000Z" / "gpu_trace.csv"
    trace.parent.mkdir(parents=True)
    trace.write_text("\n".join(samples) + "\n", encoding="utf-8")

    # Every new cell: route (i)'s complete 5 x 100 payload and route (ii)'s 500 per-episode files, equal under ==.
    values: dict[tuple[str, str, int, int], dict[str, float]] = {}
    for cell in cr.evaluation_cells():
        episodes = []
        for seed in TRAINING_SEEDS:
            for draw in HELD_OUT_DRAWS:
                v = new_episode(cell.scenario, cell.key, int(seed), int(draw))
                values[(cell.scenario, cell.key, int(seed), int(draw))] = v
                episodes.append({"arm": cell.key, "seed": int(seed), "draw_id": int(draw), "att_horizon": v["att_ours"],
                                 "episode_reward": v["episode_reward"],
                                 "horizon_vehicle_count": v["horizon_vehicle_count"]})
                _write(runs_root / "eval_ii" / cell.scenario / ar.cell_file_name(
                    ar.CellKey(cell.scenario, cell.key, int(seed), int(draw))),
                    _p8_4b_file(cell.scenario, cell.key, cell.tier, cell.arm, int(seed), int(draw), v["att_ours"],
                                v["att_engine"], episode_reward=v["episode_reward"],
                                horizon_vehicle_count=v["horizon_vehicle_count"]))
        payload = {"format_version": "fixture", "arm": cell.key, "tier": cell.tier, "method": cell.arm,
                   "episodes": episodes}
        if cell.scenario == "hz1x1":
            payload["runtime"] = {"git_commit": ROUTE_I_RUNTIME_COMMIT, "dirty": False}
        _write(cr.route_i_output(roots, cell), payload)

    # Every old hz1x1 cell's re-roll: each committed episode reproduced, att_engine its value minus ENGINE_SHIFT_OLD.
    for artifact, arm in cr.HZ1X1_OLD_CELLS:
        method, tier = ar.normalise_arm("hz1x1", arm)
        payload = json.loads((repo_root / artifact).read_text(encoding="utf-8"))
        for episode in payload["episodes"]:
            if str(episode["arm"]) != arm:
                continue
            seed = None if episode.get("seed") is None else int(episode["seed"])
            ours = float(episode["att_horizon"])
            _write(runs_root / "old_cells" / "hz1x1" / ar.cell_file_name(
                ar.CellKey("hz1x1", arm, seed, int(episode["draw_id"]))),
                _p8_4b_file("hz1x1", arm, tier, method, seed, int(episode["draw_id"]), ours, ours - ENGINE_SHIFT_OLD,
                            episode_reward=float(episode["episode_reward"]),
                            horizon_vehicle_count=float(episode["horizon_vehicle_count"])))

    # P3a's records: the new cells' values, the registered references' committed ones; one planted repeat.
    for cell in cr.p3a_cells():
        if cell in cr.REGISTERED_RANDOM_REFERENCES:
            committed = cr._registered_reference_values(roots, cell)
            keyed = {(int(k[2]), int(k[3])): float(v) for k, v in committed.items()}
        else:
            keyed = {(s, d): values[(cell.scenario, cell.key, s, d)]["att_ours"] for s in TRAINING_SEEDS
                     for d in HELD_OUT_DRAWS}
        columns = 16 if cell.scenario == "grid4x4" else 1
        rows = []
        for (seed, draw), att in sorted(keyed.items()):
            shown = draw
            if (cell.scenario, cell.key, seed) == REPEATED[:3] and draw in REPEATED[3]:
                shown = REPEATED[3][0]
            rows.append({"seed": seed, "draw_id": draw, "att_ours": att, "n_decisions": 360,
                         "action_sequence_sha256": _digest(cell.scenario, cell.key, seed, shown),
                         "per_intersection_sha256": [_digest(cell.scenario, cell.key, seed, shown, i)
                                                     for i in range(columns)]})
        _write(cr._p3a_path(roots, cell), {"format_version": cr.ACTIONS_FORMAT_VERSION, "scenario": cell.scenario,
                                           "arm": cell.key, "registered_in": "A27(d)", "digest": "fixture",
                                           "episodes": rows})

    # The D4RL re-run's output with its compare-with block (calibration/d4rl_run.py's keys the report reads): compared
    # with the committed P8.3 artifact, by its digest, and each cell's diagnostics block.
    d4rl = root / "d4rl" / rep.D4RL_OUTPUT_NAME
    registered = repo_root / "docs" / "data" / "p8_3_d4rl_calibration.json"
    cells = []
    for i, seed in enumerate(TRAINING_SEEDS):
        q, v, near_zero = d4rl_diagnostics_of(i)
        cells.append({"arm": "iql_t", "seed": int(seed), "training_seconds": 380.0 + i, "wall_seconds": 400.0 + i,
                      "gradient_steps": 40000,
                      "diagnostics": {"final_q_loss": q, "final_v_loss": v, "awr_weight_near_zero_fraction": near_zero,
                                      "diagnostic_positions": 20000, "target_dropout": "off"}})
    _write(d4rl, {"compare_with": {"registered": "docs/data/p8_3_d4rl_calibration.json", "fixture": True,
                                   "path": str(registered), "sha256": hashlib.sha256(registered.read_bytes()).hexdigest()},
                  "cells": cells})
    return FixtureTree(roots=roots, d4rl_path=d4rl, run_brackets=brackets)
