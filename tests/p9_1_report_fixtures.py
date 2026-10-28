"""A synthetic P9.1 run tree at the campaign's full size, for the report's end-to-end T-regress (``BRIEF_46`` G4; plan
C6: "T-regress for each artifact (rebuilt byte for byte from fixture run directories)").

No format of its own: every file is written in the format the campaign writes it -- ``p9.1-run/1.0`` run records,
route (i)'s cell payloads (``episodes``), P8.4b's per-episode files (``att_rederivation.cell_file_name``), the
``p9.1-actions/1.0`` P3a records, the driver's Unix-time GPU trace -- and the three declarations are written by the
campaign's own ``declare_*`` functions on the real corpus.  Alignment: every episode is a held-out ``(seed, draw)`` of
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
        if cell.scenario == "grid4x4" and cell.arm == "dt_nomix":
            record["deterministic"] = False
        _write(cr.run_record_path(roots, cell.scenario, cell.tier, cell.arm, int(seed)), record)
        brackets[(cell.scenario, cell.tier, cell.arm, int(seed))] = (started, started + seconds)

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
        _write(cr.route_i_output(roots, cell), {"format_version": "fixture", "arm": cell.key, "tier": cell.tier,
                                                "method": cell.arm, "episodes": episodes})

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

    # The D4RL re-run's output with its compare-with block (calibration/d4rl_run.py's keys the report reads).
    d4rl = root / "d4rl" / rep.D4RL_OUTPUT_NAME
    _write(d4rl, {"compare_with": {"registered": "docs/data/p8_3_d4rl_calibration.json", "fixture": True},
                  "cells": [{"arm": "iql_t", "seed": int(seed), "training_seconds": 380.0 + i, "wall_seconds": 400.0 + i,
                             "gradient_steps": 40000} for i, seed in enumerate(TRAINING_SEEDS)]})
    return FixtureTree(roots=roots, d4rl_path=d4rl, run_brackets=brackets)
